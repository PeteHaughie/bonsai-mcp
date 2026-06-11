---
name: bonsai-mcp
description: Generate images using the Bonsai 4B text-to-image diffusion model via the bonsai-mcp MCP server. Use when the user wants to generate an image, create a picture, visualize a concept, or make AI art.
---

An MCP tool called **`generate_image`** (on the `bonsai` server) is already configured and injected into the **Huxley daemon's model tool definitions** (`huxley tools`). The underlying LLM can call it via OpenAI `tool_calls` when you prompt it through the daemon's API at `http://localhost:8083/v1/chat/completions`.

> Do NOT run shell commands for image generation. Do NOT look for `generate_image` in your own agent tool list — it is a **model-level injected tool**, not a direct agent tool.

**How to use it:** Simply ask the model to generate an image. The model will automatically call `generate_image` via `tool_calls`:

```bash
curl -s http://localhost:8083/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "alpha", "messages": [{"role": "user", "content": "generate an image of..."}], "max_tokens": 800}'
```

**To verify the tool is registered:** run `huxley tools` and confirm `mcp:bonsai|generate_image` appears in the output.

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
  "output_path": "/path/to/outputs/mcp/image.png",
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

The `bonsai` MCP server is already running and set up. Prerequisites completed, model weights present.

| Env var | Purpose |
|---------|---------|
| `BONSAI_IMAGE_DEMO_DIR` | Path to Bonsai-Image-Demo repo |
| `BONSAI_MODEL` | Default model variant override |
| `BONSAI_GENERATE_WRAPPER` | Override path to the wrapper script |
| `BONSAI_OUTPUT_DIR` | Where generated PNGs are saved |

- Dimensions must be multiples of 16 (recommended: 512×512 fast, 1024×1024 quality)
- Generated PNGs land at `{BONSAI_OUTPUT_DIR}`
