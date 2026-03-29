# Contributing

This file covers local build and development workflows for this repository.

## Prerequisites

- `just`
- `cmake`
- Rust toolchain (`cargo`)
- Node.js 24 + npm (for UI development)

**macOS**: Apple Silicon. Metal is used automatically.

**Linux NVIDIA**: x86_64 with an NVIDIA GPU. Requires the CUDA toolkit (`nvcc` in your `PATH`). On Arch Linux, CUDA is typically at `/opt/cuda`; on Ubuntu/Debian it's at `/usr/local/cuda`. Auto-detection finds the right SM architecture for your GPU.

**Linux AMD**: ROCm/HIP is supported when ROCm is installed. Typical installs expose `hipcc`, `hipconfig`, and `rocm-smi` under `/opt/rocm/bin`.

**Linux Vulkan**: Vulkan is supported when the Vulkan development files and `glslc` are installed. On Ubuntu/Debian, install `libvulkan-dev glslc`. On Arch Linux, install `vulkan-headers shaderc`.

## Build from source

Build everything (llama.cpp fork, mesh binary, and UI production build):

```bash
just build
```

On Linux, `just build` auto-detects CUDA vs ROCm vs Vulkan. For NVIDIA, make sure `nvcc` is in your `PATH` first:

```bash
# Arch Linux
PATH=/opt/cuda/bin:$PATH just build

# Ubuntu/Debian
PATH=/usr/local/cuda/bin:$PATH just build
```

For NVIDIA builds, the script auto-detects your GPU's CUDA architecture. To override:

```bash
just build cuda_arch=90   # e.g. H100
```

For AMD ROCm builds, you can force the backend explicitly:

```bash
just build backend=rocm
```

To override the AMD GPU target list:

```bash
just build backend=rocm rocm_arch="gfx90a;gfx942;gfx1100"
```

For Vulkan builds, force the backend explicitly:

```bash
just build backend=vulkan
```

For CPU-only builds (no GPU acceleration):

```bash
just build backend=cpu
```

Create a portable bundle:

```bash
just bundle
```

## UI development workflow

Use this two-terminal flow for UI development.

Terminal A (run `mesh-llm` yourself):

```bash
mesh-llm --port 9337 --console 3131
```

If `mesh-llm` is not on your `PATH`:

```bash
./target/release/mesh-llm --port 9337 --console 3131
```

Terminal B (run Vite with HMR):

```bash
just ui-dev
```

Open:

```text
http://127.0.0.1:5173
```

`ui-dev` defaults:

- Serves on `127.0.0.1:5173`
- Proxies `/api/*` to `http://127.0.0.1:3131`

Overrides:

```bash
# Different backend API origin for /api proxy
just ui-dev http://127.0.0.1:4141

# Different Vite dev port
just ui-dev http://127.0.0.1:3131 5174
```

## Useful commands

```bash
just stop             # stop mesh/rpc/llama processes
just test             # quick test against :9337
just --list           # list all recipes
```
