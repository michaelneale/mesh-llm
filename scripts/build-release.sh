#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

LLAMA_DIR="${MESH_LLM_LLAMA_DIR:-$REPO_ROOT/.deps/llama.cpp}"
UI_DIR="$REPO_ROOT/crates/mesh-llm/ui"

os_name="$(uname -s)"
case "$os_name" in
    Darwin)
        BACKEND="${LLAMA_STAGE_BACKEND:-metal}"
        ;;
    Linux)
        BACKEND="${LLAMA_STAGE_BACKEND:-cpu}"
        ;;
    *)
        echo "Unsupported OS for release build: $os_name" >&2
        exit 1
        ;;
esac

if [[ -z "${LLAMA_STAGE_BUILD_DIR:-}" && -n "${SKIPPY_LLAMA_BUILD_DIR:-}" ]]; then
    export LLAMA_STAGE_BUILD_DIR="$SKIPPY_LLAMA_BUILD_DIR"
fi
if [[ -z "${LLAMA_STAGE_BUILD_DIR:-}" ]]; then
    if [[ "$BACKEND" == "cpu" ]]; then
        export LLAMA_STAGE_BUILD_DIR="$LLAMA_DIR/build-stage-abi-static"
    else
        export LLAMA_STAGE_BUILD_DIR="$LLAMA_DIR/build-stage-abi-$BACKEND"
    fi
fi

echo "Preparing patched llama.cpp ABI checkout..."
LLAMA_WORKDIR="$LLAMA_DIR" "$SCRIPT_DIR/prepare-llama.sh" "${MESH_LLM_LLAMA_PIN_SHA:-pinned}"

echo "Building patched llama.cpp ABI ($BACKEND)..."
LLAMA_WORKDIR="$LLAMA_DIR" \
    LLAMA_BUILD_DIR="$LLAMA_STAGE_BUILD_DIR" \
    LLAMA_STAGE_BACKEND="$BACKEND" \
    "$SCRIPT_DIR/build-llama.sh"

echo "Building UI..."
"$SCRIPT_DIR/build-ui.sh" "$UI_DIR"

echo "Building mesh-llm..."
(cd "$REPO_ROOT" && cargo build --release --locked -p mesh-llm)
