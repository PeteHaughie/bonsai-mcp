"""FastMCP server exposing Bonsai image generation as an MCP tool.

Spawns a short-lived subprocess per request so VRAM is only allocated
during generation and released when the subprocess exits.

Configuration via environment variables:
    BONSAI_IMAGE_DEMO_DIR      Path to the Bonsai-Image-Demo repo.
                               Default: ~/Projects/Bonsai-Image-Demo
    BONSAI_OUTPUT_DIR          Directory for generated PNGs.
                               Default: {BONSAI_IMAGE_DEMO_DIR}/outputs/mcp
    BONSAI_GENERATE_WRAPPER    Path to the wrapper subprocess script.
                               Default: alongside this file (generate_wrapper.py)
    BONSAI_MODEL               Default model variant override (auto-detected
                               by platform if unset).
"""
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Literal

from fastmcp import FastMCP

mcp = FastMCP("Bonsai Image Generator")

def _getenv_path(key: str, default: str) -> Path:
    """Read a filesystem path from the environment, falling back to a default.

    Args:
        key: Environment variable name.
        default: Fallback path string if the env var is unset or empty.

    Returns:
        Absolute, resolved Path.
    """
    val = os.environ.get(key)
    if not val:
        val = default
    return Path(val).expanduser().resolve()


def _resolve_bonsai_dir() -> Path:
    """Resolve the Bonsai-Image-Demo repo root from the environment."""
    return _getenv_path("BONSAI_IMAGE_DEMO_DIR", "~/Projects/Bonsai-Image-Demo")


def _resolve_output_dir() -> Path:
    """Resolve the directory where generated PNGs are written."""
    bonsai = _resolve_bonsai_dir()
    return _getenv_path("BONSAI_OUTPUT_DIR", str(bonsai / "outputs" / "mcp"))


def _resolve_wrapper() -> Path:
    """Resolve the path to the subprocess wrapper script."""
    default = str(Path(__file__).parent / "generate_wrapper.py")
    return _getenv_path("BONSAI_GENERATE_WRAPPER", default)


def _resolve_python() -> Path:
    """Resolve the Bonsai project's venv Python interpreter."""
    return _resolve_bonsai_dir() / ".venv" / "bin" / "python"


def _resolve_model(override: str | None) -> str | None:
    """Resolve the model variant: tool arg wins, then env var, then None for auto."""
    return override or os.environ.get("BONSAI_MODEL") or None


def _validate_size(name: str, value: int) -> str | None:
    """Validate an image dimension.

    Args:
        name: Label ("width" or "height") for error messages.
        value: Dimension in pixels.

    Returns:
        An error string if invalid, or None if valid.
    """
    if not 256 <= value <= 2048:
        return f"{name} {value} out of range — must be 256–2048"
    if value % 16:
        return f"{name} {value} must be a multiple of 16"
    return None


@mcp.tool(description="Generate an image using the Bonsai 4B model.")
def generate_image(
    prompt: str,
    seed: int | None = None,
    steps: int = 4,
    width: int = 512,
    height: int = 512,
    model: Literal["ternary-mlx", "ternary-gemlite", "binary-mlx", "binary-gemlite"] | None = None,
) -> dict[str, Any]:
    for dim, name in ((width, "width"), (height, "height")):
        err = _validate_size(name, dim)
        if err:
            return {"status": "error", "error": err}

    python_path = _resolve_python()
    wrapper_path = _resolve_wrapper()
    output_dir = _resolve_output_dir()

    if not python_path.exists():
        return {
            "status": "error",
            "error": f"Python interpreter not found at {python_path}. "
                     f"Is Bonsai-Image-Demo set up (run setup.sh)?",
        }
    if not wrapper_path.exists():
        return {
            "status": "error",
            "error": f"Wrapper script not found at {wrapper_path}. "
                     f"Check BONSAI_GENERATE_WRAPPER.",
        }

    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "prompt": prompt,
        "output_dir": str(output_dir),
        "seed": seed,
        "steps": steps,
        "width": width,
        "height": height,
    }
    resolved_model = _resolve_model(model)
    if resolved_model:
        payload["model"] = resolved_model

    try:
        result = subprocess.run(
            [str(python_path), str(wrapper_path)],
            input=json.dumps(payload).encode(),
            capture_output=True,
            timeout=600,
        )
    except subprocess.TimeoutExpired:
        return {"status": "error", "error": "Generation timed out after 600s"}
    except FileNotFoundError:
        return {"status": "error", "error": f"Could not execute {python_path}"}
    except OSError as e:
        return {"status": "error", "error": f"Subprocess error: {e}"}

    if result.stdout:
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError as e:
            return {
                "status": "error",
                "error": f"Failed to parse wrapper output: {e}",
                "raw_stdout": result.stdout.decode(errors="replace"),
            }

    return {
        "status": "error",
        "error": f"Wrapper produced no output (exit code {result.returncode})",
        "stderr": result.stderr.decode(errors="replace") if result.stderr else "",
    }


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
