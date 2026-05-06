---
name: llama-stage-patch-changes
description: Use this skill when changing mesh-llm's llama-stage.cpp ABI shim, runtime hooks, model introspection, tensor filtering, activation-frame execution, GGUF writer surface, upstream pin, or stage patch queue.
metadata:
  short-description: Maintain the llama-stage.cpp patch queue
---

# llama-stage-patch-changes

Use this skill when changing the stage ABI surface carried in
`third_party/llama-stage.cpp/patches`.

## Boundaries

- Keep durable llama stage-side changes in
  `third_party/llama-stage.cpp/patches/*.patch`.
- Keep the stage upstream pin in `third_party/llama-stage.cpp/upstream.txt`.
- Do not edit `.deps/llama-stage.cpp` as the final artifact; regenerate the
  patch queue from commits.
- Keep mesh orchestration, protocol compatibility, lifecycle, model management,
  and API status behavior in Rust.
- Prefer one ABI capability per patch.

## Local Flow

Prepare the pinned checkout and current patch queue:

```bash
scripts/prepare-llama-stage.sh pinned
```

For llama-side editing, work in `.deps/llama-stage.cpp` or another llama.cpp
checkout where commits can be named and inspected. Base the branch on the
pinned upstream, then carry the stage ABI patch commits on top.

After editing and committing in that checkout, regenerate the stage patch queue
from the upstream base:

```bash
rm -rf /Users/jdumay/code/mesh-llm/third_party/llama-stage.cpp/patches
mkdir -p /Users/jdumay/code/mesh-llm/third_party/llama-stage.cpp/patches
git -C .deps/llama-stage.cpp format-patch \
  --output-directory /Users/jdumay/code/mesh-llm/third_party/llama-stage.cpp/patches \
  "$(cat .deps/llama-stage.cpp/.llama-stage-upstream-sha)..HEAD"
```

## Validation

Validate patch application in a clean checkout:

```bash
tmp_llama="$(mktemp -d /tmp/mesh-llama-stage.XXXXXX)"
rm -rf "$tmp_llama"
LLAMA_WORKDIR="$tmp_llama" scripts/prepare-llama-stage.sh pinned
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
