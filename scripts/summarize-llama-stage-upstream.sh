#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OLD_SHA="${1:?old upstream sha required}"
NEW_SHA="${2:?new upstream sha required}"
LLAMA_WORKDIR="${3:-${LLAMA_WORKDIR:-$ROOT/.deps/llama-stage.cpp}}"

if [[ ! -d "$LLAMA_WORKDIR/.git" ]]; then
  echo "llama checkout not found: $LLAMA_WORKDIR" >&2
  exit 1
fi

git -C "$LLAMA_WORKDIR" fetch origin master --tags >/dev/null 2>&1 || true

CHANGED_FILES="$(git -C "$LLAMA_WORKDIR" diff --name-only "$OLD_SHA..$NEW_SHA" || true)"

commit_base_url() {
  local remote_url
  remote_url="$(git -C "$LLAMA_WORKDIR" remote get-url origin 2>/dev/null || true)"

  case "$remote_url" in
    git@github.com:*)
      remote_url="https://github.com/${remote_url#git@github.com:}"
      ;;
    https://github.com/*)
      ;;
    *)
      echo "https://github.com/ggml-org/llama.cpp/commit"
      return
      ;;
  esac

  remote_url="${remote_url%.git}"
  echo "$remote_url/commit"
}

commit_list() {
  local base_url="$1"
  git -C "$LLAMA_WORKDIR" log --reverse --format='%H%x09%h%x09%s' "$OLD_SHA..$NEW_SHA" |
    while IFS=$'\t' read -r full_sha short_sha subject; do
      printf -- '- [%s](%s/%s) %s\n' "$short_sha" "$base_url" "$full_sha" "$subject"
    done
}

area_count() {
  local pattern="$1"
  awk -v pattern="$pattern" 'NF && $0 ~ pattern { count++ } END { print count + 0 }' <<< "$CHANGED_FILES"
}

runtime_count="$(area_count '^(src/llama|include/llama|src/models/)')"
gguf_count="$(area_count '(^gguf|gguf|ggml.*gguf|src/llama-model-loader)')"
metal_count="$(area_count '(^ggml/src/ggml-metal|metal)')"
cuda_count="$(area_count '(^ggml/src/ggml-cuda|cuda|CUDA)')"
tokenizer_count="$(area_count '(vocab|token|tokenizer|unicode)')"
build_count="$(area_count '(^CMakeLists.txt|^cmake/|CMake|Makefile|scripts/)')"
test_count="$(area_count '(^tests/|^examples/)')"
upstream_commit_base_url="$(commit_base_url)"
upstream_commit_list="$(commit_list "$upstream_commit_base_url")"
if [[ -z "$upstream_commit_list" ]]; then
  upstream_commit_list="_No upstream commits found._"
fi
if [[ -n "${SKIPPY_CORRECTNESS_MODEL:-}" ]]; then
  correctness_checked="x"
else
  correctness_checked=" "
fi

cat <<EOF
## llama-stage.cpp Upstream Pin Update

Previous pin: \`$OLD_SHA\`
New pin: \`$NEW_SHA\`

## Upstream Summary

Generated from:

\`\`\`bash
git log --oneline $OLD_SHA..$NEW_SHA
git diff --stat $OLD_SHA..$NEW_SHA
\`\`\`

### Notable Areas

- Runtime / inference: $runtime_count changed files
- GGUF / model loading: $gguf_count changed files
- Metal backend: $metal_count changed files
- CUDA backend: $cuda_count changed files
- Tokenizer / vocab: $tokenizer_count changed files
- Build / CMake / scripts: $build_count changed files
- Tests / examples: $test_count changed files

## Diffstat

\`\`\`text
$(git -C "$LLAMA_WORKDIR" diff --stat "$OLD_SHA..$NEW_SHA")
\`\`\`

## Upstream Commits

$upstream_commit_list

## Validation

- [x] Applied Mesh-LLM llama-stage.cpp patch stack
- [x] Built patched llama stage stage ABI
- [x] Built imported skippy Rust crates
- [x] Ran targeted Rust checks/tests
- [$correctness_checked] Ran \`skippy-correctness single-step\`
- [$correctness_checked] Ran \`skippy-correctness chain\`

Correctness smokes are checked when \`SKIPPY_CORRECTNESS_MODEL\` is
available to the workflow runner.
EOF
