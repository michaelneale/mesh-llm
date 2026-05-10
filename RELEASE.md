# Releasing mesh-llm

## Prerequisites

- `just` installed
- Rust toolchain installed
- `cmake` and a native compiler installed
- Node/npm installed for the UI build
- `gh` CLI authenticated if publishing manually

## Build

```bash
just build
```

`just build` prepares the pinned upstream `llama.cpp` checkout, applies the
Mesh-LLM ABI patch queue from `third_party/llama.cpp/patches`, builds the
patched static ABI libraries, builds the UI, and builds the `mesh-llm` binary.

The release bundle is now a single `mesh-llm` runtime binary. External
`llama-server`, `rpc-server`, and `llama-moe-*` binaries are not packaged.

## Bundle

```bash
just bundle
```

This creates `/tmp/mesh-bundle.tar.gz` containing `mesh-llm`.

Platform release archives are created with:

```bash
just release-build
just release-bundle v0.X.Y
```

The current GitHub Actions release workflow publishes macOS aarch64, Linux
x86_64 CPU, Linux ARM64 CPU, Linux CUDA, Linux CUDA Blackwell, Linux ROCm,
Linux Vulkan, Windows CPU, Windows CUDA, Windows ROCm, and Windows Vulkan
bundles. The Linux ARM64 artifact is named
`mesh-llm-aarch64-unknown-linux-gnu.tar.gz`; CUDA lanes are named
`mesh-llm-x86_64-unknown-linux-gnu-cuda.tar.gz` and
`mesh-llm-x86_64-unknown-linux-gnu-cuda-blackwell.tar.gz`.

Windows release artifacts use the `x86_64-pc-windows-msvc` target triple and
`.zip` archives.

On native Windows, `just check-release` still runs the Rust/docs/workflow invariant checks, but it skips the Bash-only `install.sh` and `scripts/package-release.sh` parity checks.

## Smoke Test

```bash
mkdir /tmp/test-bundle
tar xzf /tmp/mesh-bundle.tar.gz -C /tmp/test-bundle --strip-components=1
/tmp/test-bundle/mesh-llm --model Qwen2.5-3B
rm -rf /tmp/test-bundle
```

Verify:

- the process starts without looking for `llama-server` or `rpc-server`;
- `/api/status` returns valid JSON;
- `/v1/models` lists the resolved model refs;
- `/v1/chat/completions` can generate through the embedded runtime.

## Publish

Push a `v*` tag to run `.github/workflows/release.yml`.
