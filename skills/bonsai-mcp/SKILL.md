---
name: bonsai-mcp
description: Generate images using the Bonsai 4B text-to-image diffusion model via the bonsai-mcp MCP server. Use when the user wants to generate an image, create a picture, visualize a concept, or make AI art.
---

# Bonsai Image Generation

Generate images using the Bonsai 4B text-to-image diffusion model via the `bonsai-mcp` MCP server.

## Prerequisites

1. **Bonsai-Image-Demo project** must be fully set up at the path specified by `BONSAI_IMAGE_DEMO_DIR`:
   - `./setup.sh` has been run
   - `./scripts/download_model.sh` has been run (or the specific variant downloaded)
   - Model weights are present under `models/bonsai-image-4B-*`
2. **Hardware**: macOS Apple Silicon (MLX) or Linux with NVIDIA GPU (CUDA/gemlite)
3. **MCP server** `bonsai-mcp` is running and configured in the agent harness

## MCP server config

The agent harness must define a `bonsai` MCP server pointing at `bonsai-mcp`:

```json
{
  "mcpServers": {
    "bonsai": {
      "command": "python",
      "args": ["-m", "bonsai_mcp.server"],
      "env": {
        "BONSAI_IMAGE_DEMO_DIR": "/path/to/Bonsai-Image-Demo"
      }
    }
  }
}
```

| Env var | Purpose |
|---------|---------|
| `BONSAI_IMAGE_DEMO_DIR` | Path to Bonsai-Image-Demo repo (required) |
| `BONSAI_MODEL` | Default model variant (tool arg takes precedence) |
| `BONSAI_GENERATE_WRAPPER` | Override path to the wrapper script |
| `BONSAI_OUTPUT_DIR` | Where generated PNGs are saved |

## Tools

### `generate_image`

Generate an image from a text prompt.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | `string` | — | Text description of the image to generate |
| `seed` | `int` | random 31-bit | Deterministic seed for reproducible results |
| `steps` | `int` | `4` | Inference steps (4 is recommended; more is slower but higher quality) |
| `width` | `int` | `512` | Width in pixels (must be multiple of 16, 256–2048) |
| `height` | `int` | `512` | Height in pixels (must be multiple of 16, 256–2048) |
| `model` | `enum` | auto-detect | `ternary-mlx`, `ternary-gemlite`, `binary-mlx`, or `binary-gemlite` |

**Returns:**

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

## Platform behavior

- **macOS (MLX)**: Cold start is a few seconds (Metal shaders are precompiled binaries). The default auto-detected model is `ternary-mlx`.
- **Linux (CUDA/gemlite)**: Cold start is 25–30s (imports + 4 GB model load + Triton JIT + gemlite autotune). The default auto-detected model is `ternary-gemlite`.

The subprocess exits after generation, releasing all VRAM. Each call pays the cold-start cost. For repeated generation at high throughput, use `serve.sh` directly instead.

## VRAM

VRAM is allocated only during the subprocess lifetime. Once the subprocess exits (~seconds after the tool returns), all GPU memory is freed. No GPU resources are consumed while the MCP server is idle.

## Recommended sizes

Dimensions must be multiples of 16.

| Aspect | Fast (~0.25MP) | Quality (~1MP) |
|--------|----------------|----------------|
| Square (1:1) | 512×512 | 1024×1024 |
| Landscape (3:2) | 624×416 | 1248×832 |
| Portrait (2:3) | 416×624 | 832×1248 |
| Wide (2:1) | 704×352 | 1408×704 |
| Tall (1:2) | 352×704 | 704×1408 |

## Output directory

Generated PNGs are written to `{BONSAI_OUTPUT_DIR}` (default: `{BONSAI_IMAGE_DEMO_DIR}/outputs/mcp/`) with filenames like `image_{timestamp}_seed{seed}.png`. The output path is returned in the tool result.

## Model variants

| Model key | Backend | Quality | Checkpoint size (disk) |
|-----------|---------|---------|------------------------|
| `ternary-mlx` | macOS MLX | Recommended | ~4 GB |
| `ternary-gemlite` | Linux CUDA | Recommended | ~4 GB |
| `binary-mlx` | macOS MLX | Lower | ~2 GB |
| `binary-gemlite` | Linux CUDA | Lower | ~2 GB |
