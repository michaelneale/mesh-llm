#!/usr/bin/env zsh
# build-mac.sh — build llama.cpp + mesh-llm on macOS Apple Silicon
#
# Usage:
#   scripts/build-mac.sh

setopt errexit nounset pipefail

SCRIPT_DIR="${0:A:h}"
REPO_ROOT="${SCRIPT_DIR:h}"

LLAMA_DIR="${MESH_LLM_LLAMA_DIR:-$REPO_ROOT/.deps/llama.cpp}"
BUILD_DIR="$LLAMA_DIR/build"
MESH_DIR="$REPO_ROOT/mesh-llm"
UI_DIR="$MESH_DIR/ui"

compiler_launcher_flags=()
rustc_wrapper=""

detect_jobs() {
    sysctl -n hw.ncpu 2>/dev/null || echo 4
}

configure_compiler_cache() {
    local cache_bin=""
    if (( $+commands[sccache] )); then
        cache_bin="sccache"
        rustc_wrapper="$cache_bin"
    elif (( $+commands[ccache] )); then
        cache_bin="ccache"
    else
        return 0
    fi

    echo "Using compiler cache: $cache_bin"
    compiler_launcher_flags=(
        -DCMAKE_C_COMPILER_LAUNCHER="$cache_bin"
        -DCMAKE_CXX_COMPILER_LAUNCHER="$cache_bin"
    )

    if [[ -n "$rustc_wrapper" ]]; then
        echo "Using Rust compiler wrapper: $rustc_wrapper"
    fi
}

stage_dev_runtime_binaries() {
    local backend="$1"
    local target_dir="$2"
    local source_bin_dir="$BUILD_DIR/bin"

    mkdir -p "$target_dir"
    rm -f "$target_dir/rpc-server" "$target_dir/llama-server"
    rm -f "$target_dir"/rpc-server-*(N) "$target_dir"/llama-server-*(N)

    for name in rpc-server llama-server; do
        local source="$source_bin_dir/$name"
        if [[ ! -f "$source" ]]; then
            echo "Error: expected llama.cpp binary not found: $source" >&2
            exit 1
        fi
        cp "$source" "$target_dir/$name-$backend"
    done

    for name in llama-moe-analyze llama-moe-split; do
        local source="$source_bin_dir/$name"
        if [[ -f "$source" ]]; then
            cp "$source" "$target_dir/$name"
        fi
    done

    echo "Staged llama.cpp runtime binaries in $target_dir with '$backend' flavor names."
}

LLAMA_WORKDIR="$LLAMA_DIR" "$SCRIPT_DIR/prepare-llama.sh" "${MESH_LLM_LLAMA_PIN_SHA:-pinned}"

configure_compiler_cache

cmake_flags=(
    -B "$BUILD_DIR"
    -S "$LLAMA_DIR"
    -DGGML_METAL=ON
    -DGGML_RPC=ON
    -DBUILD_SHARED_LIBS=OFF
    -DLLAMA_OPENSSL=OFF
)

if (( $+commands[ninja] )); then
    cmake_flags=(-G Ninja "${cmake_flags[@]}")
fi

cmake_flags+=("${compiler_launcher_flags[@]}")

echo "Configuring llama.cpp for macOS..."
cmake "${cmake_flags[@]}"

echo "Building llama.cpp..."
cmake --build "$BUILD_DIR" --config Release --parallel "$(detect_jobs)"
echo "Build complete: $BUILD_DIR/bin/"

if [[ -d "$MESH_DIR" ]]; then
    echo "Building mesh-llm..."
    if [[ -d "$UI_DIR" ]]; then
        "$SCRIPT_DIR/build-ui.sh" "$UI_DIR"
    fi

    if [[ -n "$rustc_wrapper" ]]; then
        (
            cd "$REPO_ROOT"
            RUSTC_WRAPPER="$rustc_wrapper" cargo build --release
        )
    else
        (
            cd "$REPO_ROOT"
            cargo build --release
        )
    fi

    stage_dev_runtime_binaries "metal" "$REPO_ROOT/target/release"
    echo "Mesh binary: target/release/mesh-llm"
fi
