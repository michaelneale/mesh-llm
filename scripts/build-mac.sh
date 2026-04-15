#!/usr/bin/env zsh
# build-mac.sh — build llama.cpp + mesh-llm on macOS Apple Silicon
#
# Usage:
#   scripts/build-mac.sh

setopt errexit nounset pipefail

SCRIPT_DIR="${0:A:h}"
REPO_ROOT="${SCRIPT_DIR:h}"

LLAMA_DIR="$REPO_ROOT/llama.cpp"
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

LLAMA_BRANCH="${LLAMA_BRANCH:-master}"
LLAMA_REPO="https://github.com/Mesh-LLM/llama.cpp.git"

# Read pinned SHA from LLAMA_CPP_SHA if it exists
LLAMA_PIN_SHA=""
if [[ -f "$REPO_ROOT/LLAMA_CPP_SHA" ]]; then
    LLAMA_PIN_SHA="$(tr -d '[:space:]' < "$REPO_ROOT/LLAMA_CPP_SHA")"
fi

clone_or_update_llama() {
    if [[ ! -d "$LLAMA_DIR" ]]; then
        echo "Cloning Mesh-LLM/llama.cpp ($LLAMA_BRANCH branch)..."
        git clone -b "$LLAMA_BRANCH" "$LLAMA_REPO" "$LLAMA_DIR"
    else
        pushd "$LLAMA_DIR" >/dev/null
        local current_branch
        current_branch="$(git branch --show-current 2>/dev/null || true)"
        if [[ -n "$current_branch" && "$current_branch" != "$LLAMA_BRANCH" ]]; then
            echo "⚠️  llama.cpp is on branch '$current_branch', switching to $LLAMA_BRANCH..."
            git checkout "$LLAMA_BRANCH"
        fi
        echo "Pulling latest $LLAMA_BRANCH from origin..."
        if ! git pull --ff-only origin "$LLAMA_BRANCH" 2>/dev/null; then
            echo "⚠️  git pull failed (detached HEAD or offline) — will pin to SHA if available"
        fi
        popd >/dev/null
    fi

    # Pin to exact SHA if LLAMA_CPP_SHA exists
    if [[ -n "$LLAMA_PIN_SHA" ]]; then
        pushd "$LLAMA_DIR" >/dev/null
        local current_sha
        current_sha="$(git rev-parse HEAD)"
        if [[ "$current_sha" != "$LLAMA_PIN_SHA" ]]; then
            echo "Pinning llama.cpp to SHA $LLAMA_PIN_SHA (was ${current_sha:0:12})..."
            git fetch origin "$LLAMA_PIN_SHA" 2>/dev/null || git fetch origin
            git checkout "$LLAMA_PIN_SHA" --detach
        else
            echo "llama.cpp already at pinned SHA ${LLAMA_PIN_SHA:0:12}"
        fi
        popd >/dev/null
    fi
}

clone_or_update_llama

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

    echo "Mesh binary: target/release/mesh-llm"
fi
