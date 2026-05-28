"""Single-shot Bonsai image generation via stdin/stdout.

Reads JSON from stdin, writes PNG to the specified output directory, then
writes a JSON result to stdout and exits. Intended to be spawned as a
short-lived subprocess so VRAM is released after each request.

Must be run using the Bonsai project's Python interpreter
(``Bonsai-Image-Demo/.venv/bin/python``) so all ML dependencies
(prism-image-studio, mflux, backend_gpu) are importable.

Environment:
    BONSAI_IMAGE_DEMO_DIR  Path to the Bonsai-Image-Demo repo root.
                           Default: ~/Projects/Bonsai-Image-Demo

Input (stdin):
    {
        "prompt":      "..." (required),
        "output_dir":  "/path/to/outputs" (default: BONSAI_IMAGE_DEMO_DIR/outputs/mcp),
        "seed":        null | int (default: random 31-bit),
        "steps":       4 (default),
        "width":       512 (default),
        "height":      512 (default),
        "model":       null | "ternary-mlx" | "ternary-gemlite" | "binary-mlx" | "binary-gemlite"
    }

Output (stdout):
    { "status": "ok",    "output_path": "...", "seed": ..., "duration_seconds": ..., ... }
    { "status": "error", "error": "..." }
"""
import json
import logging
import os
import secrets
import sys
import time
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("bonsai-generate-wrapper")

BONSAI_IMAGE_DEMO_DIR = (
    Path(os.environ.get(
        "BONSAI_IMAGE_DEMO_DIR",
        str(Path.home() / "Projects" / "Bonsai-Image-Demo"),
    )).resolve()
)
MODELS_DIR = BONSAI_IMAGE_DEMO_DIR / "models"

TRITON_CACHE_DIR = BONSAI_IMAGE_DEMO_DIR / "outputs" / ".triton_cache"
GEMLITE_PERSIST_PATH = BONSAI_IMAGE_DEMO_DIR / "outputs" / ".gemlite_cache" / "autotune.json"
if sys.platform != "darwin":
    TRITON_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    GEMLITE_PERSIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TRITON_CACHE_DIR", str(TRITON_CACHE_DIR))

MODELS = {
    "ternary-mlx":     ("bonsai-ternary-mlx",     "bonsai-image-4B-ternary-mlx"),
    "ternary-gemlite": ("bonsai-ternary-gemlite", "bonsai-image-4B-ternary-gemlite"),
    "binary-mlx":      ("bonsai-binary-mlx",      "bonsai-image-4B-binary-mlx"),
    "binary-gemlite":  ("bonsai-binary-gemlite",  "bonsai-image-4B-binary-gemlite"),
}


def default_model() -> str:
    return "ternary-mlx" if sys.platform == "darwin" else "ternary-gemlite"


def require_model_dir(model: str) -> Path:
    _, subdir = MODELS[model]
    model_root = MODELS_DIR / subdir
    if not model_root.exists():
        raise FileNotFoundError(
            f"Model directory not found: {model_root}\n"
            f"Run: ./scripts/download_model.sh --model {model}"
        )
    return model_root


# ── macOS path ─────────────────────────────────────────────────────────────

def generate_macos(
    prompt: str, seed: int, width: int, height: int, steps: int, model: str,
) -> tuple[bytes, dict[str, float]]:
    log.info("setup: importing FluxPipeline (MLX) ...")
    setup_t0 = time.perf_counter()
    from backend.pipeline import FluxPipeline, PipelineConfig

    backend_id, _ = MODELS[model]
    model_root = require_model_dir(model)
    pipeline = FluxPipeline(PipelineConfig(
        backend=backend_id,
        baked_model_path=str(model_root),
        te_4bit=True,
        evict_text_encoder=True,
    ))
    setup_s = time.perf_counter() - setup_t0

    log.info("diffusion (steps=%d size=%dx%d seed=%d) ...", steps, width, height, seed)
    diff_t0 = time.perf_counter()
    png_bytes = pipeline.generate_png(
        prompt=prompt, seed=seed, steps=steps, height=height, width=width,
    )
    diffusion_s = time.perf_counter() - diff_t0

    return png_bytes, {"setup_s": setup_s, "diffusion_s": diffusion_s}


# ── Linux path ─────────────────────────────────────────────────────────────

def _find_subdir(root: Path, *hints: str) -> Path:
    matches = [
        p for p in root.iterdir()
        if p.is_dir() and any(h in p.name for h in hints)
    ]
    if not matches:
        present = ", ".join(sorted(p.name for p in root.iterdir() if p.is_dir())) or "(empty)"
        raise FileNotFoundError(
            f"No subdir matching {hints!r} under {root}. Present: {present}"
        )
    matches.sort(key=lambda p: len(p.name), reverse=True)
    return matches[0]


def generate_linux(
    prompt: str, seed: int, width: int, height: int, steps: int, model: str,
) -> tuple[bytes, dict[str, float]]:
    log.info("setup: importing GpuPipeline (gemlite) ...")
    setup_t0 = time.perf_counter()

    from backend_gpu.pipeline_gpu import GpuPipeline
    from gemlite.core import GemLiteLinearTriton

    backend_id, _ = MODELS[model]
    model_root = require_model_dir(model)
    text_encoder_dir = _find_subdir(model_root, "text_encoder")
    transformer_kwarg = {
        "bonsai-binary-gemlite":  "binary_transformer_path",
        "bonsai-ternary-gemlite": "ternary_transformer_path",
    }[backend_id]

    pipeline = GpuPipeline(
        backend=backend_id,
        **{transformer_kwarg: str(_find_subdir(model_root, "transformer"))},
        text_encoder_path=str(text_encoder_dir),
        vae_path=str(_find_subdir(model_root, "vae")),
        tokenizer_path=str(text_encoder_dir / "tokenizer"),
    )

    if GEMLITE_PERSIST_PATH.exists():
        GemLiteLinearTriton.load_config(str(GEMLITE_PERSIST_PATH), print_error=False)

    pipeline.prewarm()
    setup_s = time.perf_counter() - setup_t0

    log.info("diffusion (steps=%d size=%dx%d seed=%d) ...", steps, width, height, seed)
    diff_t0 = time.perf_counter()
    png_bytes = pipeline.generate_png(
        prompt=prompt, seed=seed, steps=steps, height=height, width=width,
    )
    diffusion_s = time.perf_counter() - diff_t0

    GemLiteLinearTriton.cache_config(str(GEMLITE_PERSIST_PATH))

    return png_bytes, {"setup_s": setup_s, "diffusion_s": diffusion_s}


# ── main ───────────────────────────────────────────────────────────────────

def main() -> None:
    raw = sys.stdin.read()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        result = {"status": "error", "error": f"Invalid JSON on stdin: {e}"}
        sys.stdout.write(json.dumps(result))
        sys.exit(1)

    prompt = data.get("prompt", "")
    if not prompt:
        result = {"status": "error", "error": "Missing required field: prompt"}
        sys.stdout.write(json.dumps(result))
        sys.exit(1)

    output_dir = Path(data.get("output_dir", str(BONSAI_IMAGE_DEMO_DIR / "outputs" / "mcp")))
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = data.get("seed")
    if seed is None:
        seed = secrets.randbits(31)
    elif not isinstance(seed, int) or isinstance(seed, bool):
        result = {"status": "error", "error": f"'seed' must be an integer or null, got {type(seed).__name__}"}
        sys.stdout.write(json.dumps(result))
        sys.exit(1)
    try:
        steps = int(data.get("steps", 4))
        width = int(data.get("width", 512))
        height = int(data.get("height", 512))
    except (TypeError, ValueError) as e:
        result = {"status": "error", "error": f"Invalid numeric parameter: {e}"}
        sys.stdout.write(json.dumps(result))
        sys.exit(1)
    model = data.get("model") or default_model()

    if model not in MODELS:
        result = {
            "status": "error",
            "error": f"Unknown model {model!r}. Valid: {', '.join(sorted(MODELS))}",
        }
        sys.stdout.write(json.dumps(result))
        sys.exit(1)

    for dim, name in ((width, "width"), (height, "height")):
        if not 256 <= dim <= 2048:
            result = {
                "status": "error",
                "error": f"{name} {dim} out of range — must be 256–2048",
            }
            sys.stdout.write(json.dumps(result))
            sys.exit(1)
        if dim % 16:
            result = {
                "status": "error",
                "error": f"{name} {dim} must be a multiple of 16",
            }
            sys.stdout.write(json.dumps(result))
            sys.exit(1)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"image_{ts}_seed{seed}.png"

    wall_t0 = time.perf_counter()
    try:
        if sys.platform == "darwin":
            png_bytes, stages = generate_macos(prompt, seed, width, height, steps, model)
        else:
            png_bytes, stages = generate_linux(prompt, seed, width, height, steps, model)
    except Exception as e:
        log.exception("generation failed")
        result = {"status": "error", "error": str(e)}
        sys.stdout.write(json.dumps(result))
        sys.exit(1)

    wall_s = time.perf_counter() - wall_t0
    output_path.write_bytes(png_bytes)

    result = {
        "status": "ok",
        "output_path": str(output_path),
        "seed": seed,
        "duration_seconds": round(wall_s, 3),
        "stages": {k: round(v, 3) for k, v in stages.items()},
        "model": model,
        "width": width,
        "height": height,
    }
    sys.stdout.write(json.dumps(result))


if __name__ == "__main__":
    main()
