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
