#!/usr/bin/env bash
# build-linux-rocm.sh — build llama.cpp (ROCm/HIP) + mesh-llm on Linux
#
# Usage: scripts/build-linux-rocm.sh [amdgpu_targets]
#   amdgpu_targets  Semicolon-separated AMDGPU targets, e.g.
#                   "gfx90a;gfx942;gfx1100". If omitted, a broad default is used.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

LLAMA_DIR="${MESH_LLM_LLAMA_DIR:-$REPO_ROOT/.deps/llama.cpp}"
BUILD_DIR="$LLAMA_DIR/build"
MESH_DIR="$REPO_ROOT/crates/mesh-llm"
UI_DIR="$MESH_DIR/ui"

AMDGPU_TARGETS="${1:-gfx90a;gfx942;gfx1100;gfx1101;gfx1102;gfx1200;gfx1201}"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"

if [[ ! -d "$ROCM_PATH" ]]; then
    echo "Error: ROCm not found at $ROCM_PATH" >&2
    exit 1
fi

export ROCM_PATH
export PATH="$ROCM_PATH/bin:$ROCM_PATH/llvm/bin:$PATH"

if ! command -v hipconfig >/dev/null 2>&1; then
    echo "Error: hipconfig not found. Ensure ROCm is installed and PATH includes $ROCM_PATH/bin." >&2
    exit 1
fi

compiler_launcher_flags=()

configure_compiler_cache() {
    local cache_bin=""
    if command -v sccache >/dev/null 2>&1; then
        cache_bin="sccache"
    elif command -v ccache >/dev/null 2>&1; then
        cache_bin="ccache"
    else
        return 0
    fi

    echo "Using compiler cache: $cache_bin"
    compiler_launcher_flags=(
        -DCMAKE_C_COMPILER_LAUNCHER="$cache_bin"
        -DCMAKE_CXX_COMPILER_LAUNCHER="$cache_bin"
        -DCMAKE_HIP_COMPILER_LAUNCHER="$cache_bin"
    )
}

LLAMA_WORKDIR="$LLAMA_DIR" "$SCRIPT_DIR/prepare-llama.sh" "${MESH_LLM_LLAMA_PIN_SHA:-pinned}"

echo "Using ROCm from $ROCM_PATH"
echo "Building for AMDGPU targets: $AMDGPU_TARGETS"

configure_compiler_cache

HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
cmake -B "$BUILD_DIR" -S "$LLAMA_DIR" \
    -DGGML_HIP=ON \
    -DGGML_CUDA=OFF \
    -DGGML_VULKAN=OFF \
    -DGGML_METAL=OFF \
    -DGGML_RPC=ON \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DBUILD_SHARED_LIBS=OFF \
    -DLLAMA_OPENSSL=OFF \
    -DAMDGPU_TARGETS="$AMDGPU_TARGETS" \
    "${compiler_launcher_flags[@]}"

cmake --build "$BUILD_DIR" --config Release -j"$(nproc)"
echo "llama.cpp ROCm build complete: $BUILD_DIR/bin/"

LLAMA_STAGE_WORKDIR="${LLAMA_STAGE_WORKDIR:-${SKIPPY_LLAMA_WORKDIR:-$REPO_ROOT/.deps/llama-stage.cpp}}"
LLAMA_STAGE_BACKEND_NAME="${LLAMA_STAGE_BACKEND:-${SKIPPY_LLAMA_BACKEND:-cpu}}"
if [[ -z "${LLAMA_STAGE_BUILD_DIR:-}" && -n "${SKIPPY_LLAMA_BUILD_DIR:-}" ]]; then
    export LLAMA_STAGE_BUILD_DIR="$SKIPPY_LLAMA_BUILD_DIR"
fi
if [[ -z "${LLAMA_STAGE_BUILD_DIR:-}" ]]; then
    if [[ "$LLAMA_STAGE_BACKEND_NAME" == "cpu" ]]; then
        export LLAMA_STAGE_BUILD_DIR="$LLAMA_STAGE_WORKDIR/build-stage-abi-static"
    else
        export LLAMA_STAGE_BUILD_DIR="$LLAMA_STAGE_WORKDIR/build-stage-abi-$LLAMA_STAGE_BACKEND_NAME"
    fi
fi

echo "Building llama-stage ABI ($LLAMA_STAGE_BACKEND_NAME)..."
LLAMA_WORKDIR="$LLAMA_STAGE_WORKDIR" "$SCRIPT_DIR/prepare-llama-stage.sh" "${LLAMA_STAGE_PIN_SHA:-pinned}"
LLAMA_WORKDIR="$LLAMA_STAGE_WORKDIR" \
    LLAMA_BUILD_DIR="$LLAMA_STAGE_BUILD_DIR" \
    LLAMA_STAGE_BACKEND="$LLAMA_STAGE_BACKEND_NAME" \
    LLAMA_STAGE_AMDGPU_TARGETS="$AMDGPU_TARGETS" \
    "$SCRIPT_DIR/build-llama-stage.sh"

if [[ -d "$MESH_DIR" ]]; then
    if [[ -d "$UI_DIR" ]]; then
        "$SCRIPT_DIR/build-ui.sh" "$UI_DIR"
    fi
    echo "Building mesh-llm..."
    (cd "$REPO_ROOT" && cargo build --release --locked -p mesh-llm)
    echo "Mesh binary: target/release/mesh-llm"
fi
