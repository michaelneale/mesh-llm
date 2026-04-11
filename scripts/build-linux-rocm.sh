#!/usr/bin/env bash
# build-linux-rocm.sh — build llama.cpp (ROCm/HIP) + mesh-llm on Linux
#
# Usage: scripts/build-linux-rocm.sh [amdgpu_targets]
#   amdgpu_targets  Semicolon-separated AMDGPU targets, e.g.
#                   "gfx90a;gfx942;gfx1100". If omitted, a broad default is used.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

LLAMA_DIR="$REPO_ROOT/llama.cpp"
BUILD_DIR="$LLAMA_DIR/build"
MESH_DIR="$REPO_ROOT/mesh-llm"
UI_DIR="$MESH_DIR/ui"

AMDGPU_TARGETS="${1:-gfx90a;gfx942;gfx1100;gfx1101;gfx1102;gfx1200;gfx1201}"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
LLAMA_PIN_SHA="${MESH_LLM_LLAMA_PIN_SHA:-}"

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
        return
    fi

    echo "Using compiler cache: $cache_bin"
    compiler_launcher_flags=(
        -DCMAKE_C_COMPILER_LAUNCHER="$cache_bin"
        -DCMAKE_CXX_COMPILER_LAUNCHER="$cache_bin"
        -DCMAKE_HIP_COMPILER_LAUNCHER="$cache_bin"
    )
}

if [[ ! -d "$LLAMA_DIR" ]]; then
    if [[ -n "$LLAMA_PIN_SHA" ]]; then
        echo "Cloning michaelneale/llama.cpp pinned to $LLAMA_PIN_SHA..."
        git clone -b upstream-latest --depth 1 \
            https://github.com/michaelneale/llama.cpp.git "$LLAMA_DIR"
        if ! (cd "$LLAMA_DIR" && git cat-file -e "${LLAMA_PIN_SHA}^{commit}" 2>/dev/null); then
            echo "Pinned SHA not on upstream-latest tip, fetching explicitly..."
            (cd "$LLAMA_DIR" && git fetch --depth 1 origin "$LLAMA_PIN_SHA")
        fi
        (cd "$LLAMA_DIR" && git checkout --detach "$LLAMA_PIN_SHA")
    else
        echo "Cloning michaelneale/llama.cpp (upstream-latest)..."
        git clone -b upstream-latest \
            https://github.com/michaelneale/llama.cpp.git "$LLAMA_DIR"
    fi
else
    cd "$LLAMA_DIR"
    if [[ -n "$LLAMA_PIN_SHA" ]]; then
        if ! git cat-file -e "${LLAMA_PIN_SHA}^{commit}" 2>/dev/null; then
            echo "Fetching pinned llama.cpp SHA $LLAMA_PIN_SHA..."
            git fetch --depth 1 origin "$LLAMA_PIN_SHA"
        fi
        CURRENT_SHA="$(git rev-parse HEAD)"
        if [[ "$CURRENT_SHA" != "$LLAMA_PIN_SHA" ]]; then
            echo "Checking out pinned llama.cpp SHA $LLAMA_PIN_SHA (was $CURRENT_SHA)..."
            git checkout --detach "$LLAMA_PIN_SHA"
        else
            echo "llama.cpp already at pinned SHA $LLAMA_PIN_SHA, no checkout needed"
        fi
    else
        CURRENT_BRANCH=$(git branch --show-current)
        if [[ "$CURRENT_BRANCH" != "upstream-latest" ]]; then
            echo "⚠️  llama.cpp is on branch '$CURRENT_BRANCH', switching to upstream-latest..."
            git checkout upstream-latest
        fi
        echo "Pulling latest upstream-latest from origin..."
        git pull --ff-only origin upstream-latest
    fi
    cd "$REPO_ROOT"
fi

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

if [[ -d "$MESH_DIR" ]]; then
    if [[ -d "$UI_DIR" ]]; then
        "$SCRIPT_DIR/build-ui.sh" "$UI_DIR"
    fi
    echo "Building mesh-llm..."
    (cd "$REPO_ROOT" && cargo build --release --locked -p mesh-llm)
    echo "Mesh binary: target/release/mesh-llm"
fi
