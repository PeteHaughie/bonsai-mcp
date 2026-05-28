# bonsai-mcp

MCP server for AI agent-driven image generation using the [Bonsai 4B](https://prismml.com) model.

Spawns a short-lived subprocess per request — VRAM is allocated only during generation and released when the process exits.

## Prerequisites

- [Bonsai-Image-Demo](https://github.com/PrismML-Eng/Bonsai-Image-Demo) must be fully set up (`./setup.sh`, models downloaded)
- macOS Apple Silicon or Linux with NVIDIA GPU (CUDA)

## Install

```bash
cd ~/Projects/bonsai-mcp
uv sync
```

## Configuration

| Env var | Default | Purpose |
|---------|---------|---------|
| `BONSAI_IMAGE_DEMO_DIR` | `~/Projects/Bonsai-Image-Demo` | Bonsai project root |
| `BONSAI_OUTPUT_DIR` | `{BONSAI_IMAGE_DEMO_DIR}/outputs/mcp` | Where generated PNGs land |
| `BONSAI_GENERATE_WRAPPER` | `src/bonsai_mcp/generate_wrapper.py` | Path to the subprocess wrapper script |
| `BONSAI_MODEL` | auto (ternary-mlx / ternary-gemlite) | Default model variant (tool arg takes precedence) |

## Usage

```bash
# Start the MCP server (stdio transport for agents)
bonsai-mcp

# Or with an env override:
BONSAI_IMAGE_DEMO_DIR=/path/to/Bonsai-Image-Demo bonsai-mcp
```
