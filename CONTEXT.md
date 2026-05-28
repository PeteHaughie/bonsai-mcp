# bonsai-mcp domain glossary

## Project

**bonsai-mcp** — MCP server that wraps Bonsai-Image-Demo's text-to-image pipeline for AI agent use. Lives at `~/Projects/bonsai-mcp`.

**Bonsai-Image-Demo** — The upstream repo containing the Bonsai 4B model, FastAPI backend, Next.js frontend, and CLI tools. The source of truth for model pipelines and weights. Pointed to by `BONSAI_IMAGE_DEMO_DIR`.

## Components

**MCP server** — The long-lived FastMCP process (`server.py`). Exposes one tool (`generate_image`). Does not load ML libraries — delegates to the wrapper.

**Wrapper** — The short-lived subprocess (`generate_wrapper.py`). Runs inside Bonsai-Image-Demo's venv so all ML deps are importable. Reads JSON from stdin, writes PNG to disk, writes JSON result to stdout, exits. VRAM is live only during this process.

## Model concepts

**Model variant** — A specific combination of quantization format and backend: `ternary-mlx`, `ternary-gemlite`, `binary-mlx`, `binary-gemlite`.

**Backend** — The compute platform: `mlx` (Apple Silicon / Metal) or `gemlite` (Linux / CUDA + Triton). Determined by `sys.platform`.

**Checkpoint** — The model weights on disk under `Bonsai-Image-Demo/models/`. ~4 GB for ternary, ~2 GB for binary. Distinct from VRAM usage at runtime.

## Parameter conventions

**`steps`** — Inference / diffusion steps. 4 is the recommended default. Higher = slower but slightly better quality.

**`seed`** — 31-bit integer for deterministic generation. When unset, a random value is chosen and returned.

**`width` / `height`** — Output dimensions in pixels. Must be multiples of 16. Valid range: 256–2048.

## Platform behavior

- **macOS (darwin)**: Uses `FluxPipeline` from `prism-image-studio` (MLX/Metal). Cold start ~few seconds. Precompiled Metal shaders — no per-shape JIT.
- **Linux**: Uses `GpuPipeline` from `prism-image-studio-backend-gpu` (gemlite/HQQ on CUDA). Cold start 25–30s (Triton JIT + gemlite autotune per shape). Results cached under `outputs/.triton_cache/` and `outputs/.gemlite_cache/`.
