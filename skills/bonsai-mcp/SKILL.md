---
name: bonsai-mcp
description: Generate images using the Bonsai 4B text-to-image diffusion model via the bonsai-mcp MCP server. Use when the user wants to generate an image, create a picture, visualize a concept, or make AI art.
---

An MCP tool called **`generate_image`** (on the `bonsai` server) is already configured and available in this environment. When the user asks for an image, call it.

> Do NOT run shell commands for image generation. Use the `generate_image` MCP tool only.

**Before proceeding:** list your available MCP tools. Confirm `generate_image` is present before continuing.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | `string` | — | Text description of the image |
| `seed` | `int` | random 31-bit | Deterministic seed |
| `steps` | `int` | `4` | Inference steps (4 is recommended; higher = slower, better quality) |
| `width` | `int` | `512` | Width in pixels (multiple of 16, 256–2048) |
| `height` | `int` | `512` | Height in pixels (multiple of 16, 256–2048) |
| `model` | `enum` | auto-detect | `ternary-mlx`, `ternary-gemlite`, `binary-mlx`, or `binary-gemlite` |

## Returns

```json
{
  "status": "ok",
  "output_path": "/path/to/outputs/mcp/image_20260528_120000_seed42.png",
  "seed": 42,
  "duration_seconds": 3.2,
  "stages": { "setup_s": 2.1, "diffusion_s": 1.1 },
  "model": "ternary-mlx",
  "width": 512,
  "height": 512
}
```

On error, `status` is `"error"` and an `"error"` field describes the problem.

## Setup & reference (handled by the operator, not the agent)

The `bonsai` MCP server is already running. The operator has completed all prerequisites:

- Bonsai-Image-Demo repo at `BONSAI_IMAGE_DEMO_DIR`
- `./setup.sh` and `./scripts/download_model.sh` run
- Model weights at `models/bonsai-image-4B-*`
- Hardware: macOS Apple Silicon (MLX) or Linux NVIDIA GPU (CUDA)

| Env var | Purpose |
|---------|---------|
| `BONSAI_IMAGE_DEMO_DIR` | Path to Bonsai-Image-Demo repo |
| `BONSAI_MODEL` | Default model variant override |
| `BONSAI_GENERATE_WRAPPER` | Override path to the wrapper script |
| `BONSAI_OUTPUT_DIR` | Where generated PNGs are saved |

- Dimensions must be multiples of 16 (recommended: 512×512 fast, 1024×1024 quality)
- Generated PNGs land at `{BONSAI_OUTPUT_DIR}`
- Platform behavior: macOS MLX (few-second cold start), Linux CUDA/gemlite (25–30s cold start)
- VRAM allocated only during generation, freed on subprocess exit
