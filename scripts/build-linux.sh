#!/usr/bin/env bash
# build-linux.sh — build llama.cpp (CUDA) + mesh-llm on Linux
#
# Usage: scripts/build-linux.sh [--clean] [cuda_arch]
#   --clean     Wipe the build dir before cmake (required on arch change).
#   cuda_arch   SM integer for CMAKE_CUDA_ARCHITECTURES (e.g. 87, 90, 120).
#               If omitted, scripts/detect-cuda-arch.sh is invoked to detect it.
#
# Must be run from the repository root.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

LLAMA_DIR="$REPO_ROOT/llama.cpp"
BUILD_DIR="$LLAMA_DIR/build"
MESH_DIR="$REPO_ROOT/mesh-llm"
UI_DIR="$MESH_DIR/ui"

CLEAN=0
CUDA_ARCH=""

for ARG in "$@"; do
    case "$ARG" in
        --clean) CLEAN=1 ;;
        *)       [[ -z "$CUDA_ARCH" ]] && CUDA_ARCH="$ARG" ;;
    esac
done

if [[ -z "$CUDA_ARCH" ]]; then
    echo "No cuda_arch specified — running auto-detection..."
    CUDA_ARCH="$("$SCRIPT_DIR/detect-cuda-arch.sh")"
    echo "Using SM ${CUDA_ARCH}"
fi

# Locate nvcc — check PATH first, then common install locations
if ! command -v nvcc &>/dev/null; then
    for CANDIDATE in /usr/local/cuda/bin /opt/cuda/bin /usr/cuda/bin; do
        if [[ -x "$CANDIDATE/nvcc" ]]; then
            export PATH="$CANDIDATE:$PATH"
            break
        fi
    done
fi

if ! command -v nvcc &>/dev/null; then
    echo "Error: nvcc not found. Install the CUDA toolkit and ensure nvcc is in your PATH." >&2
    echo "  Arch Linux:    sudo pacman -S cuda" >&2
    echo "  Ubuntu/Debian: sudo apt install nvidia-cuda-toolkit" >&2
    exit 1
fi

echo "Using nvcc: $(command -v nvcc) ($(nvcc --version | grep release | awk '{print $5}' | tr -d ','))"

if [[ ! -d "$LLAMA_DIR" ]]; then
    echo "Cloning michaelneale/llama.cpp (rebase-upstream-master)..."
    git clone -b rebase-upstream-master \
        https://github.com/michaelneale/llama.cpp.git "$LLAMA_DIR"
else
    cd "$LLAMA_DIR"
    CURRENT_BRANCH=$(git branch --show-current)
    if [[ "$CURRENT_BRANCH" != "rebase-upstream-master" ]]; then
        echo "⚠️  llama.cpp is on branch '$CURRENT_BRANCH', switching to rebase-upstream-master..."
        git checkout rebase-upstream-master
    fi
    echo "Pulling latest rebase-upstream-master from origin..."
    git pull --ff-only origin rebase-upstream-master
    cd "$REPO_ROOT"
fi

if [[ "$CLEAN" -eq 1 && -d "$BUILD_DIR" ]]; then
    echo "Cleaning build dir..."
    rm -rf "$BUILD_DIR"
fi

cmake -B "$BUILD_DIR" -S "$LLAMA_DIR" \
    -DGGML_CUDA=ON \
    -DGGML_METAL=OFF \
    -DGGML_RPC=ON \
    -DBUILD_SHARED_LIBS=OFF \
    -DLLAMA_OPENSSL=OFF \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH"

cmake --build "$BUILD_DIR" --config Release -j"$(nproc)"
echo "llama.cpp build complete: $BUILD_DIR/bin/"

if [[ -d "$MESH_DIR" ]]; then
    if [[ -d "$UI_DIR" ]]; then
        echo "Building mesh-llm UI..."
        (cd "$UI_DIR" && npm ci && npm run build)
    fi
    echo "Building mesh-llm..."
    (cd "$MESH_DIR" && cargo build --release)
    echo "Mesh binary: target/release/mesh-llm"
fi
