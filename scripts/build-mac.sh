#!/usr/bin/env zsh
# build-mac.sh — build patched llama.cpp ABI libraries + mesh-llm on macOS.

setopt errexit nounset pipefail

SCRIPT_DIR="${0:A:h}"
REPO_ROOT="${SCRIPT_DIR:h}"

LLAMA_DIR="${MESH_LLM_LLAMA_DIR:-$REPO_ROOT/.deps/llama.cpp}"
MESH_DIR="$REPO_ROOT/crates/mesh-llm"
UI_DIR="$REPO_ROOT/crates/mesh-llm-ui"

rustc_wrapper=""

configure_rust_cache() {
    if (( $+commands[sccache] )); then
        rustc_wrapper="$(command -v sccache)"
        echo "Using Rust compiler wrapper: $rustc_wrapper"
    fi
}

export LLAMA_STAGE_BUILD_DIR="${LLAMA_STAGE_BUILD_DIR:-${SKIPPY_LLAMA_BUILD_DIR:-$LLAMA_DIR/build-stage-abi-metal}}"

echo "Preparing patched llama.cpp ABI checkout..."
LLAMA_WORKDIR="$LLAMA_DIR" "$SCRIPT_DIR/prepare-llama.sh" "${MESH_LLM_LLAMA_PIN_SHA:-pinned}"

echo "Building patched llama.cpp ABI (metal)..."
LLAMA_WORKDIR="$LLAMA_DIR" \
    LLAMA_BUILD_DIR="$LLAMA_STAGE_BUILD_DIR" \
    LLAMA_STAGE_BACKEND="${LLAMA_STAGE_BACKEND:-metal}" \
    "$SCRIPT_DIR/build-llama.sh"

if [[ -d "$MESH_DIR" ]]; then
    if [[ -d "$UI_DIR" ]]; then
        "$SCRIPT_DIR/build-ui.sh" "$UI_DIR"
    fi

    configure_rust_cache
    if [[ -n "$rustc_wrapper" ]]; then
        (cd "$REPO_ROOT" && RUSTC_WRAPPER="$rustc_wrapper" cargo build --release -p mesh-llm)
    else
        (cd "$REPO_ROOT" && cargo build --release -p mesh-llm)
    fi

    echo "Mesh binary: target/release/mesh-llm"
fi
