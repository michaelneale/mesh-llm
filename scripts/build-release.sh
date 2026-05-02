#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

LLAMA_DIR="${MESH_LLM_LLAMA_DIR:-$REPO_ROOT/.deps/llama.cpp}"
BUILD_DIR="$LLAMA_DIR/build"
UI_DIR="$REPO_ROOT/crates/mesh-llm/ui"

detect_jobs() {
    if command -v nproc >/dev/null 2>&1; then
        nproc
    elif command -v sysctl >/dev/null 2>&1; then
        sysctl -n hw.ncpu
    else
        echo 4
    fi
}

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
    )
}

os_name="$(uname -s)"
cmake_flags=(
    -B "$BUILD_DIR"
    -S "$LLAMA_DIR"
    -DGGML_RPC=ON
    -DBUILD_SHARED_LIBS=OFF
    -DLLAMA_OPENSSL=OFF
)

case "$os_name" in
    Darwin)
        cmake_flags+=(
            -DGGML_METAL=ON
        )
        ;;
    Linux)
        cmake_flags+=(
            -DGGML_CUDA=OFF
            -DGGML_HIP=OFF
            -DGGML_METAL=OFF
            # Release Linux CPU artifacts must stay portable across GitHub runners.
            -DGGML_NATIVE=OFF
            -DGGML_VULKAN=OFF
        )
        ;;
    *)
        echo "Unsupported OS for release build: $os_name" >&2
        exit 1
        ;;
esac

if command -v ninja >/dev/null 2>&1; then
    cmake_flags=(-G Ninja "${cmake_flags[@]}")
fi

configure_compiler_cache
cmake_flags+=("${compiler_launcher_flags[@]}")

LLAMA_WORKDIR="$LLAMA_DIR" "$SCRIPT_DIR/prepare-llama.sh" "${MESH_LLM_LLAMA_PIN_SHA:-pinned}"

echo "Configuring llama.cpp for $os_name..."
cmake "${cmake_flags[@]}"

echo "Building llama.cpp..."
cmake --build "$BUILD_DIR" --config Release --parallel "$(detect_jobs)"

SKIPPY_LLAMA_WORKDIR="${SKIPPY_LLAMA_WORKDIR:-$REPO_ROOT/.deps/skippy-llama.cpp}"
export SKIPPY_LLAMA_BUILD_DIR="${SKIPPY_LLAMA_BUILD_DIR:-$SKIPPY_LLAMA_WORKDIR/build-stage-abi-static}"

echo "Building Skippy stage ABI..."
LLAMA_WORKDIR="$SKIPPY_LLAMA_WORKDIR" "$SCRIPT_DIR/prepare-skippy-llama.sh" "${SKIPPY_LLAMA_PIN_SHA:-pinned}"
LLAMA_WORKDIR="$SKIPPY_LLAMA_WORKDIR" \
    LLAMA_BUILD_DIR="$SKIPPY_LLAMA_BUILD_DIR" \
    "$SCRIPT_DIR/build-skippy-llama.sh"

echo "Building UI..."
"$SCRIPT_DIR/build-ui.sh" "$UI_DIR"

echo "Building mesh-llm..."
(
    cd "$REPO_ROOT"
    cargo build --release --locked -p mesh-llm
)
