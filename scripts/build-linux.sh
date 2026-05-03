#!/usr/bin/env bash
# build-linux.sh — build patched llama.cpp ABI libraries + mesh-llm on Linux
#
# Usage:
#   scripts/build-linux.sh [--clean] [--skip-ui] [--backend cpu|cuda|rocm|vulkan] [--cuda-arch SM_LIST] [--rocm-arch GFX_LIST]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

LLAMA_DIR="${MESH_LLM_LLAMA_DIR:-$REPO_ROOT/.deps/llama.cpp}"
LLAMA_BUILD_ROOT="${MESH_LLM_LLAMA_BUILD_ROOT:-$REPO_ROOT/.deps/llama-build}"
MESH_DIR="$REPO_ROOT/crates/mesh-llm"
UI_DIR="$MESH_DIR/ui"

CLEAN=0
SKIP_UI="${MESH_LLM_SKIP_UI:-0}"
BACKEND=""
CUDA_ARCH=""
ROCM_ARCH=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --clean)
            CLEAN=1
            shift
            ;;
        --skip-ui)
            SKIP_UI=1
            shift
            ;;
        --backend)
            BACKEND="${2:-}"
            shift 2
            ;;
        --cuda-arch)
            CUDA_ARCH="${2:-}"
            shift 2
            ;;
        --rocm-arch)
            ROCM_ARCH="${2:-}"
            shift 2
            ;;
        *)
            [[ -z "$CUDA_ARCH" ]] && CUDA_ARCH="$1"
            shift
            ;;
    esac
done

detect_backend() {
    if command -v nvidia-smi &>/dev/null || command -v tegrastats &>/dev/null || command -v nvcc &>/dev/null; then
        echo cuda
        return
    fi
    if command -v rocm-smi &>/dev/null || command -v rocminfo &>/dev/null || command -v hipcc &>/dev/null || [[ -x /opt/rocm/bin/hipcc ]]; then
        echo rocm
        return
    fi
    if command -v glslc &>/dev/null; then
        if command -v vulkaninfo &>/dev/null && vulkaninfo --summary >/dev/null 2>&1; then
            echo vulkan
            return
        fi
        if pkg-config --exists vulkan 2>/dev/null || [[ -n "${VULKAN_SDK:-}" ]]; then
            echo vulkan
            return
        fi
    fi
    echo cpu
}

locate_nvcc() {
    if command -v nvcc &>/dev/null; then
        return 0
    fi
    for candidate in /usr/local/cuda/bin /opt/cuda/bin /usr/cuda/bin; do
        if [[ -x "$candidate/nvcc" ]]; then
            export PATH="$candidate:$PATH"
            return 0
        fi
    done
    return 1
}

locate_hip_toolchain() {
    if command -v hipcc &>/dev/null; then
        return 0
    fi
    for candidate in /opt/rocm/bin /usr/lib/rocm/bin /usr/local/rocm/bin; do
        if [[ -x "$candidate/hipcc" ]]; then
            export PATH="$candidate:$PATH"
            return 0
        fi
    done
    return 1
}

locate_vulkan_toolchain() {
    if ! command -v glslc &>/dev/null; then
        if [[ -n "${VULKAN_SDK:-}" && -x "$VULKAN_SDK/bin/glslc" ]]; then
            export PATH="$VULKAN_SDK/bin:$PATH"
        else
            return 1
        fi
    fi

    if pkg-config --exists vulkan 2>/dev/null ||
        [[ -f /usr/include/vulkan/vulkan.h || -f /usr/local/include/vulkan/vulkan.h ]] ||
        [[ -n "${VULKAN_SDK:-}" && -f "$VULKAN_SDK/include/vulkan/vulkan.h" ]]; then
        return 0
    fi

    return 1
}

sanitize_build_component() {
    printf '%s' "$1" | tr ';, /:' '_____' | tr -cd 'A-Za-z0-9_.-'
}

default_llama_build_dir_for_backend() {
    local backend="$1"
    local suffix="$backend"
    case "$backend" in
        cpu)
            suffix="cpu"
            ;;
        cuda)
            suffix="cuda-sm$(sanitize_build_component "$CUDA_ARCH")"
            ;;
        rocm)
            suffix="rocm-$(sanitize_build_component "$ROCM_ARCH")"
            ;;
    esac
    printf '%s/build-stage-abi-%s\n' "$LLAMA_BUILD_ROOT" "$suffix"
}

if [[ -z "$BACKEND" ]]; then
    BACKEND="$(detect_backend)"
fi

case "$BACKEND" in
    cuda)
        locate_nvcc || {
            echo "Error: nvcc not found. Install the CUDA toolkit and ensure nvcc is in PATH." >&2
            exit 1
        }
        if [[ -z "$CUDA_ARCH" ]]; then
            CUDA_ARCH="$("$SCRIPT_DIR/detect-cuda-arch.sh")"
        fi
        echo "Building Linux backend: CUDA ($CUDA_ARCH)"
        ;;
    rocm|hip)
        locate_hip_toolchain || {
            echo "Error: hipcc not found. Install ROCm and ensure hipcc is in PATH." >&2
            exit 1
        }
        if [[ -z "$ROCM_ARCH" ]]; then
            ROCM_ARCH="$("$SCRIPT_DIR/detect-rocm-arch.sh")"
        fi
        BACKEND="rocm"
        echo "Building Linux backend: ROCm/HIP ($ROCM_ARCH)"
        ;;
    vulkan)
        locate_vulkan_toolchain || {
            echo "Error: Vulkan development files or glslc not found." >&2
            exit 1
        }
        echo "Building Linux backend: Vulkan"
        ;;
    cpu)
        echo "Building Linux backend: CPU"
        ;;
    *)
        echo "Error: unsupported backend '$BACKEND' (expected cpu, cuda, rocm, or vulkan)." >&2
        exit 1
        ;;
esac

if [[ -z "${LLAMA_STAGE_BUILD_DIR:-}" && -n "${SKIPPY_LLAMA_BUILD_DIR:-}" ]]; then
    export LLAMA_STAGE_BUILD_DIR="$SKIPPY_LLAMA_BUILD_DIR"
fi
if [[ -z "${LLAMA_STAGE_BUILD_DIR:-}" ]]; then
    export LLAMA_STAGE_BUILD_DIR="$(default_llama_build_dir_for_backend "$BACKEND")"
fi
echo "Using llama.cpp build dir: $LLAMA_STAGE_BUILD_DIR"

if [[ "$CLEAN" -eq 1 ]]; then
    rm -rf "$LLAMA_STAGE_BUILD_DIR"
fi

echo "Preparing patched llama.cpp ABI checkout..."
LLAMA_WORKDIR="$LLAMA_DIR" "$SCRIPT_DIR/prepare-llama.sh" "${MESH_LLM_LLAMA_PIN_SHA:-pinned}"

echo "Building patched llama.cpp ABI ($BACKEND)..."
LLAMA_WORKDIR="$LLAMA_DIR" \
    LLAMA_BUILD_DIR="$LLAMA_STAGE_BUILD_DIR" \
    LLAMA_STAGE_BACKEND="$BACKEND" \
    LLAMA_STAGE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
    LLAMA_STAGE_AMDGPU_TARGETS="$ROCM_ARCH" \
    "$SCRIPT_DIR/build-llama.sh"

if [[ "$SKIP_UI" == "1" ]]; then
    echo "Skipping UI build (MESH_LLM_SKIP_UI=1 or --skip-ui)."
elif [[ -d "$UI_DIR" ]]; then
    "$SCRIPT_DIR/build-ui.sh" "$UI_DIR"
fi

if [[ "${MESH_LLM_BUILD_PROFILE:-release}" == "dev" || "${MESH_LLM_BUILD_PROFILE:-release}" == "debug" ]]; then
    echo "Building mesh-llm (profile: dev, bin only)..."
    (cd "$REPO_ROOT" && cargo build -p mesh-llm --bin mesh-llm)
    echo "Mesh binary: target/debug/mesh-llm"
else
    echo "Building mesh-llm (profile: release)..."
    (cd "$REPO_ROOT" && cargo build --release -p mesh-llm)
    echo "Mesh binary: target/release/mesh-llm"
fi
