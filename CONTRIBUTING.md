# Contributing

Join the [#mesh-llm channel on the Goose Discord](https://discord.gg/goose-oss) for discussion and questions.

This file covers local build and development workflows for this repository.

## Prerequisites

- `just`
- `cmake`
- Rust toolchain (`cargo`)
- Node.js 24 + pnpm (for UI development)

**macOS**: Apple Silicon. Metal is used automatically.

**Linux NVIDIA**: x86_64 with an NVIDIA GPU. Requires the CUDA toolkit (`nvcc` in your `PATH`). On Arch Linux, CUDA is typically at `/opt/cuda`; on Ubuntu/Debian it's at `/usr/local/cuda`. Auto-detection finds the right SM architecture for your GPU.

**Linux AMD**: ROCm/HIP is supported when ROCm is installed. Typical installs expose `hipcc`, `hipconfig`, and `rocm-smi` under `/opt/rocm/bin`.

**Linux Vulkan**: Vulkan is supported when the Vulkan development files and `glslc` are installed. On Ubuntu/Debian, install `libvulkan-dev glslc`. On Arch Linux, install `vulkan-headers shaderc`.

**Windows**: `just build` auto-detects `cuda`, `hip`/`rocm`, `vulkan`, or `cpu`. You can override with `just build backend=cuda` (or `rocm`, `vulkan`, `cpu`). Metal is not supported on Windows.

## Build from source

Build everything (patched llama.cpp, mesh binary, and UI production build):

```bash
just build
```

`just build` builds the mesh binary in release mode. For day-to-day iteration
where the final release link is the slow step, use the debug-profile shortcut:

```bash
just build-dev
```

You can also keep the normal recipe shape and select the profile explicitly:

```bash
MESH_LLM_BUILD_PROFILE=dev just build
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

On Windows, you can override the detected backend if needed:

```powershell
just build backend=vulkan
just build backend=cpu
just build backend=cuda cuda_arch=90
```

Windows release bundles use dedicated Windows release recipes:

```powershell
just release-build-cuda-windows
just release-bundle-cuda-windows v0.X.0
```

GitHub Actions uses hosted `windows-2022` runners for compile-only Windows CI. The release workflow keeps the Windows release build/publish block commented out for now, so Windows release packaging is currently local-only via the `*-windows` `just` recipes above.

Create a portable bundle:

```bash
just bundle
```

## UI development workflow

The React console and embedded asset crate live in `crates/mesh-llm-ui/`.
The host binary serves the built assets through the management API.

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
just check-release    # release-target/docs/workflow parity check
just compat-smoke ~/.cache/huggingface/hub/<model>.gguf   # optional 2-node + 1-client Python/Node/LiteLLM smoke
just --list           # list all recipes
```

On native Windows, `just check-release` runs the host-safe Rust/doc invariant subset and skips the Bash-only `install.sh` / `package-release.sh` parity checks. Run it on macOS or Linux when you need full shell parity coverage.

## CI / GitHub Actions

CI uses [`dorny/paths-filter`](https://github.com/dorny/paths-filter) to skip jobs when unchanged areas of the repo are modified. A `changes` detection job runs first on every push and PR, then each build job gates on its output.

For the repo's CI design rules and workflow responsibilities, see [`docs/CI_GUIDANCE.md`](docs/CI_GUIDANCE.md).

### What triggers what

| Changed paths                                                                                           | `linux` / `macos` | `linux_cuda` / `linux_rocm` / `linux_vulkan` / `windows` |
| ------------------------------------------------------------------------------------------------------- | ----------------- | -------------------------------------------------------- |
| `crates/mesh-llm-host-runtime/**`, `crates/mesh-llm-system/**`, `Cargo.*`, `Justfile`, `scripts/**`, `crates/mesh-llm/build.rs`, `crates/mesh-llm-plugin/**`, `crates/mesh-llm/tests/**`, `crates/mesh-llm-protocol/proto/**` | ✅ runs           | ✅ runs                                                  |
| `crates/mesh-llm-ui/**`                                                                                        | ✅ runs           | ⏭ skipped                                               |
| `**/*.md`, `docs/**`, anything else                                                                     | ⏭ skipped        | ⏭ skipped                                               |
| Manual `workflow_dispatch`                                                                              | ✅ runs           | ✅ runs                                                  |

### Verifying path filtering works

To confirm builds are skipped on a docs-only change, open a PR and push a commit that touches only a `.md` file (e.g. add a blank line to `README.md`). All build jobs should appear as **Skipped** in the Actions tab — only the `changes` job runs.

To confirm UI-only changes skip the GPU backend jobs, push a commit touching only `crates/mesh-llm-ui/**`. The `linux` and `macos` jobs run; `linux_cuda`, `linux_rocm`, `linux_vulkan`, and `windows` are skipped.

### Adding new paths

If you add a new Rust crate, build script, or test directory, add its path to the `rust` filter in `.github/workflows/ci.yml` under the `changes` job so it correctly triggers the build matrix.

## GPU benchmark execution

GPU bandwidth benchmarks are launched through the `mesh-llm` binary itself rather than standalone benchmark executables. The public command remains:

```bash
mesh-llm gpus benchmark
```

Internally, mesh-llm runs a hidden `benchmark` subcommand in a narrow subprocess boundary so native backend hangs and stdout capture stay isolated from the main process.

Standard builds support benchmark execution only for the backends wired into the normal build flow:

- macOS Apple Silicon: Metal
- Linux / Windows NVIDIA: CUDA
- Linux / Windows AMD: HIP / ROCm

Intel GPU benchmark execution is not currently supported in standard `just build` flows, so runtime benchmark selection intentionally skips Intel GPUs.

## Protocol Backward Compatibility

Any change to `crates/mesh-llm-host-runtime/src/protocol/` or `crates/mesh-client/src/protocol/` requires backward-compatibility tests before merging.

Embedded clients (iOS, macOS, Android) are permanently supported. Protocol changes that break embedded client compatibility are breaking changes.

Run the protocol compatibility tests after any protocol change:

```bash
cargo test -p mesh-llm --test protocol_compat_v0_client
cargo test -p mesh-llm --test protocol_convert_matrix
```

See [`docs/design/EMBEDDED_CLIENT_ADR.md`](docs/design/EMBEDDED_CLIENT_ADR.md) for the full compatibility policy and rationale.
