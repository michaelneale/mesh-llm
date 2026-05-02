---
name: skippy-llama-patch-changes
description: Use this skill when changing mesh-llm's skippy llama.cpp ABI shim, runtime hooks, model introspection, tensor filtering, activation-frame execution, GGUF writer surface, upstream pin, or skippy patch queue.
metadata:
  short-description: Maintain the skippy llama.cpp patch queue
---

# skippy-llama-patch-changes

Use this skill when changing the skippy ABI surface carried in
`third_party/skippy-llama.cpp/patches`.

## Boundaries

- Keep durable skippy llama-side changes in
  `third_party/skippy-llama.cpp/patches/*.patch`.
- Keep the skippy upstream pin in `third_party/skippy-llama.cpp/upstream.txt`.
- Do not edit `.deps/skippy-llama.cpp` as the final artifact; regenerate the
  patch queue from commits.
- Keep mesh orchestration, protocol compatibility, lifecycle, model management,
  and API status behavior in Rust.
- Prefer one ABI capability per patch.

## Local Flow

Prepare the pinned checkout and current patch queue:

```bash
scripts/prepare-skippy-llama.sh pinned
```

For llama-side editing, work in `.deps/skippy-llama.cpp` or another llama.cpp
checkout where commits can be named and inspected. Base the branch on the
pinned upstream, then carry the skippy patch commits on top.

After editing and committing in that checkout, regenerate the skippy patch
queue from the upstream base:

```bash
rm -rf /Users/jdumay/code/mesh-llm/third_party/skippy-llama.cpp/patches
mkdir -p /Users/jdumay/code/mesh-llm/third_party/skippy-llama.cpp/patches
git -C .deps/skippy-llama.cpp format-patch \
  --output-directory /Users/jdumay/code/mesh-llm/third_party/skippy-llama.cpp/patches \
  "$(cat .deps/skippy-llama.cpp/.skippy-upstream-sha)..HEAD"
```

## Validation

Validate patch application in a clean checkout:

```bash
tmp_llama="$(mktemp -d /tmp/mesh-skippy-llama.XXXXXX)"
rm -rf "$tmp_llama"
LLAMA_WORKDIR="$tmp_llama" scripts/prepare-skippy-llama.sh pinned
```

For Rust fallout, run cargo commands serially:

```bash
cargo fmt --all -- --check
cargo check -p mesh-llm
cargo test -p skippy-runtime --lib
cargo test -p skippy-server --lib
cargo test -p mesh-llm --lib
```

Patch files are mail-format artifacts. Do not hand-normalize them in a way that
breaks `git am`.
