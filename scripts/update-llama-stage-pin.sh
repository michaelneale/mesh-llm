#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_WORKDIR="${LLAMA_WORKDIR:-$ROOT/.deps/llama-stage.cpp}"
PIN_FILE="${LLAMA_PIN_FILE:-$ROOT/third_party/llama-stage.cpp/upstream.txt}"

if [[ -f "$LLAMA_WORKDIR/.llama-stage-upstream-sha" ]]; then
  NEW_SHA="$(tr -d '[:space:]' < "$LLAMA_WORKDIR/.llama-stage-upstream-sha")"
elif [[ -d "$LLAMA_WORKDIR/.git" ]]; then
  NEW_SHA="$(git -C "$LLAMA_WORKDIR" rev-parse HEAD)"
else
  echo "llama checkout not found: $LLAMA_WORKDIR" >&2
  exit 1
fi

printf '%s\n' "$NEW_SHA" > "$PIN_FILE"
echo "updated $PIN_FILE to $NEW_SHA"
