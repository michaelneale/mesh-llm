# llama.cpp Patch Queue

mesh-llm builds llama.cpp from upstream `ggml-org/llama.cpp` plus a local patch
queue. The old fork workflow has been retired for normal builds.

## Source Layout

```text
third_party/llama.cpp/upstream.txt   pinned upstream commit
third_party/llama.cpp/patches/       Mesh-LLM patch queue
.deps/llama.cpp/                     prepared checkout, ignored by git
```

`LLAMA_CPP_SHA` is temporarily kept as a compatibility mirror of
`third_party/llama.cpp/upstream.txt`. New scripts should treat
`third_party/llama.cpp/upstream.txt` as the source of truth.

## What's In The Patch Queue

The current queue is the flattened form of the former Mesh-LLM llama.cpp fork.
It contains patches for:

- RPC optimizations: zero-transfer GGUF loading, allocation-size caching, and
  direct worker-to-worker tensor transfers
- MoE analysis support: expert mask routing and `llama-moe-analyze`
- Mesh hooks: virtual LLM callback hooks used by inter-model collaboration

All runtime behavior remains external-process based in this migration:
mesh-llm still launches `rpc-server` and `llama-server`.

## Preparing llama.cpp

```bash
scripts/prepare-llama.sh pinned
```

This:

1. clones or fetches upstream `ggml-org/llama.cpp` into `.deps/llama.cpp`
2. checks out the pinned upstream SHA from `third_party/llama.cpp/upstream.txt`
3. applies all patches in `third_party/llama.cpp/patches/` with `git am --3way`
4. writes diagnostic SHAs under `.deps/llama.cpp/.git/`
5. creates a best-effort `llama.cpp -> .deps/llama.cpp` compatibility symlink

To test the patch queue against current upstream without moving the pin:

```bash
scripts/prepare-llama.sh latest
```

To test a specific upstream commit:

```bash
scripts/prepare-llama.sh <upstream-sha>
```

## Building

Use `just build`; do not manually build llama.cpp for normal development.

The platform build scripts call `scripts/prepare-llama.sh` before CMake and
then build from `.deps/llama.cpp/build`.

Important backend flags are preserved from the fork workflow:

- `GGML_RPC=ON`
- `BUILD_SHARED_LIBS=OFF`
- `LLAMA_OPENSSL=OFF`
- Metal on macOS
- CPU/CUDA/ROCm/Vulkan backend selection on Linux
- `GGML_CUDA_FA_ALL_QUANTS=ON` by default for CUDA release correctness

## Updating The Upstream Pin

```bash
scripts/prepare-llama.sh latest
just build
cargo test -p mesh-llm --lib
```

If the patch queue applies and validation passes:

```bash
cp third_party/llama.cpp/upstream.txt /tmp/old-llama-upstream.txt
git -C .deps/llama.cpp rev-parse "$(cat .deps/llama.cpp/.git/mesh-llm-upstream-sha)" > third_party/llama.cpp/upstream.txt
cp third_party/llama.cpp/upstream.txt LLAMA_CPP_SHA
```

Then commit the pin update with any patch refreshes.

## Refreshing Patches

Use a temporary llama.cpp working tree for patch authoring, then export patches
with `git format-patch`.

The queue should remain ordered by responsibility:

1. RPC patches
2. MoE analysis patches
3. Mesh hook patches
4. Future llama-stage ABI patches, when intentionally added

Do not add llama-stage ABI patches as part of the fork-to-patch-queue migration.
That integration is tracked separately in
[LLAMA_STAGE_INTEGRATION_PLAN.md](LLAMA_STAGE_INTEGRATION_PLAN.md).

## Compatibility Notes

The prepared checkout is intentionally outside tracked source. If tooling still
expects `llama.cpp/build/bin`, the prepare script creates a compatibility
symlink when possible. New scripts should prefer `.deps/llama.cpp/build/bin` or
the `MESH_LLM_LLAMA_DIR` override.

The mesh protocol is unchanged by this migration.
