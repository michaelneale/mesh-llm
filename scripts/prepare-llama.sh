#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-pinned}"

LLAMA_UPSTREAM_URL="${LLAMA_UPSTREAM_URL:-https://github.com/ggml-org/llama.cpp.git}"
LLAMA_WORKDIR="${LLAMA_WORKDIR:-$ROOT/.deps/llama.cpp}"
PIN_FILE="${LLAMA_PIN_FILE:-$ROOT/third_party/llama.cpp/upstream.txt}"
PATCH_DIR="${LLAMA_PATCH_DIR:-$ROOT/third_party/llama.cpp/patches}"

if [[ ! -f "$PIN_FILE" ]]; then
  echo "missing llama upstream pin: $PIN_FILE" >&2
  exit 1
fi

if [[ ! -d "$PATCH_DIR" ]]; then
  echo "missing llama patch directory: $PATCH_DIR" >&2
  exit 1
fi

mkdir -p "$(dirname "$LLAMA_WORKDIR")"

if [[ ! -d "$LLAMA_WORKDIR/.git" ]]; then
  rm -rf "$LLAMA_WORKDIR"
  git clone --filter=blob:none "$LLAMA_UPSTREAM_URL" "$LLAMA_WORKDIR"
fi

git -C "$LLAMA_WORKDIR" am --abort >/dev/null 2>&1 || true
git -C "$LLAMA_WORKDIR" remote set-url origin "$LLAMA_UPSTREAM_URL"
git -C "$LLAMA_WORKDIR" fetch origin master --tags
git -C "$LLAMA_WORKDIR" config user.name "${GIT_AUTHOR_NAME:-Mesh-LLM CI}"
git -C "$LLAMA_WORKDIR" config user.email "${GIT_AUTHOR_EMAIL:-ci@mesh-llm.local}"

case "$MODE" in
  pinned)
    TARGET_SHA="$(tr -d '[:space:]' < "$PIN_FILE")"
    ;;
  latest)
    TARGET_SHA="$(git -C "$LLAMA_WORKDIR" rev-parse origin/master)"
    ;;
  *)
    TARGET_SHA="$MODE"
    ;;
esac

# The llama.cpp checkout is a generated dependency worktree. Local edits there
# should live in third_party/llama.cpp/patches, so reset before switching pins.
git -C "$LLAMA_WORKDIR" reset --hard HEAD
git -C "$LLAMA_WORKDIR" clean -fdx
git -C "$LLAMA_WORKDIR" checkout --force --detach "$TARGET_SHA"
git -C "$LLAMA_WORKDIR" reset --hard "$TARGET_SHA"
git -C "$LLAMA_WORKDIR" clean -fdx

printf '%s\n' "$TARGET_SHA" > "$LLAMA_WORKDIR/.mesh-llm-upstream-sha"

PATCHES=()
while IFS= read -r patch; do
  PATCHES+=("$patch")
done < <(find "$PATCH_DIR" -maxdepth 1 -type f -name '*.patch' | sort)
if (( ${#PATCHES[@]} > 0 )); then
  git -C "$LLAMA_WORKDIR" am --3way "${PATCHES[@]}"
fi

git -C "$LLAMA_WORKDIR" rev-parse HEAD > "$LLAMA_WORKDIR/.mesh-llm-patched-sha"

echo "prepared llama.cpp"
echo "  upstream: $TARGET_SHA"
echo "  patched:  $(cat "$LLAMA_WORKDIR/.mesh-llm-patched-sha")"
echo "  workdir:  $LLAMA_WORKDIR"
