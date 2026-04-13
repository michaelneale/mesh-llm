#!/usr/bin/env bash
# build-linux.sh — build llama.cpp + mesh-llm on Linux
#
# Usage:
#   scripts/build-linux.sh [--clean] [--backend cpu|cuda|rocm|vulkan] [--cuda-arch SM_LIST] [--rocm-arch GFX_LIST]
#
# Examples:
#   scripts/build-linux.sh
#   scripts/build-linux.sh --backend cpu
#   scripts/build-linux.sh --backend cuda --cuda-arch '120;86'
#   scripts/build-linux.sh --backend rocm --rocm-arch 'gfx942;gfx90a'
#   scripts/build-linux.sh --backend vulkan
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
BACKEND=""
CUDA_ARCH=""
ROCM_ARCH=""
LLAMA_TARGETS="${MESH_LLM_LLAMA_TARGETS:-}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --clean)
            CLEAN=1
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
            # Backward compatibility: treat a bare arg as cuda_arch.
            [[ -z "$CUDA_ARCH" ]] && CUDA_ARCH="$1"
            shift
            ;;
    esac
done

detect_backend() {
    if command -v nvidia-smi &>/dev/null; then
        echo cuda
        return 0
    fi
    if command -v tegrastats &>/dev/null; then
        echo cuda
        return 0
    fi
    if command -v nvcc &>/dev/null; then
        echo cuda
        return 0
    fi
    if command -v rocm-smi &>/dev/null; then
        echo rocm
        return 0
    fi
    if command -v rocminfo &>/dev/null; then
        echo rocm
        return 0
    fi
    if command -v hipcc &>/dev/null; then
        echo rocm
        return 0
    fi
    if [[ -x /opt/rocm/bin/hipcc ]]; then
        echo rocm
        return 0
    fi
    if command -v glslc &>/dev/null; then
        if command -v vulkaninfo &>/dev/null && vulkaninfo --summary >/dev/null 2>&1; then
            echo vulkan
            return 0
        fi
        if pkg-config --exists vulkan 2>/dev/null; then
            echo vulkan
            return 0
        fi
        if [[ -n "${VULKAN_SDK:-}" ]]; then
            echo vulkan
            return 0
        fi
    fi
    echo cpu
}

locate_nvcc() {
    if command -v nvcc &>/dev/null; then
        return 0
    fi
    for CANDIDATE in /usr/local/cuda/bin /opt/cuda/bin /usr/cuda/bin; do
        if [[ -x "$CANDIDATE/nvcc" ]]; then
            export PATH="$CANDIDATE:$PATH"
            return 0
        fi
    done
    return 1
}

locate_hip_toolchain() {
    if command -v hipcc &>/dev/null; then
        return 0
    fi
    for CANDIDATE in /opt/rocm/bin /usr/lib/rocm/bin /usr/local/rocm/bin; do
        if [[ -x "$CANDIDATE/hipcc" ]]; then
            export PATH="$CANDIDATE:$PATH"
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

    if pkg-config --exists vulkan 2>/dev/null; then
        return 0
    fi

    if [[ -f /usr/include/vulkan/vulkan.h || -f /usr/local/include/vulkan/vulkan.h ]]; then
        return 0
    fi

    if [[ -n "${VULKAN_SDK:-}" ]]; then
        export CMAKE_PREFIX_PATH="${VULKAN_SDK}${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
        if [[ -f "$VULKAN_SDK/include/vulkan/vulkan.h" ]]; then
            return 0
        fi
    fi

    return 1
}

compiler_launcher_flags=()

configure_compiler_cache() {
    local backend="$1"
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
    )

    case "$backend" in
        cuda)
            compiler_launcher_flags+=(-DCMAKE_CUDA_COMPILER_LAUNCHER="$cache_bin")
            ;;
        rocm)
            compiler_launcher_flags+=(-DCMAKE_HIP_COMPILER_LAUNCHER="$cache_bin")
            ;;
    esac
}

if [[ -z "$BACKEND" ]]; then
    BACKEND="$(detect_backend)"
fi

case "$BACKEND" in
    cuda)
        locate_nvcc || {
            echo "Error: nvcc not found. Install the CUDA toolkit and ensure nvcc is in your PATH." >&2
            echo "  Arch Linux:    sudo pacman -S cuda" >&2
            echo "  Ubuntu/Debian: sudo apt install nvidia-cuda-toolkit" >&2
            exit 1
        }
        if [[ -z "$CUDA_ARCH" ]]; then
            echo "No cuda_arch specified — running auto-detection..."
            CUDA_ARCH="$("$SCRIPT_DIR/detect-cuda-arch.sh")"
            echo "Using SM ${CUDA_ARCH}"
        fi
        echo "Building Linux backend: CUDA"
        echo "Using nvcc: $(command -v nvcc) ($(nvcc --version | grep release | awk '{print $5}' | tr -d ','))"
        ;;
    rocm)
        locate_hip_toolchain || {
            echo "Error: hipcc not found. Install ROCm and ensure hipcc is in your PATH." >&2
            echo "  Typical location: /opt/rocm/bin/hipcc" >&2
            exit 1
        }
        if [[ -z "$ROCM_ARCH" ]]; then
            echo "No rocm_arch specified — running auto-detection..."
            ROCM_ARCH="$("$SCRIPT_DIR/detect-rocm-arch.sh")"
            echo "Using AMDGPU_TARGETS ${ROCM_ARCH}"
        fi
        echo "Building Linux backend: ROCm/HIP"
        echo "Using hipcc: $(command -v hipcc)"
        ;;
    vulkan)
        locate_vulkan_toolchain || {
            echo "Error: Vulkan SDK/development files not found." >&2
            echo "  Need both the Vulkan headers/loader and 'glslc' in your PATH." >&2
            echo "  Ubuntu/Debian: sudo apt install libvulkan-dev glslc" >&2
            echo "  Arch Linux:    sudo pacman -S vulkan-headers shaderc" >&2
            exit 1
        }
        echo "Building Linux backend: Vulkan"
        echo "Using glslc: $(command -v glslc)"
        ;;
    cpu)
        echo "Building Linux backend: CPU only (no GPU acceleration)"
        ;;
    *)
        echo "Error: unsupported backend '$BACKEND' (expected 'cpu', 'cuda', 'rocm', or 'vulkan')." >&2
        exit 1
        ;;
esac

# MESH_LLM_LLAMA_PIN_SHA pins the llama.cpp checkout to a specific commit and
# disables the `git pull` that would otherwise move the working tree forward.
# This is required by the cross-PR llama.cpp / CUDA artifact cache in
# .github/workflows/ci.yml: the cache key embeds the resolved upstream SHA, so
# the actual checkout MUST match that SHA byte-for-byte or the restored
# `llama.cpp/build/` directory will be inconsistent with the source tree and
# cmake will silently rebuild things.
#
# When unset (the default for local `just build`), behaviour is unchanged:
# clone-or-pull `upstream-latest` HEAD as before.
LLAMA_PIN_SHA="${MESH_LLM_LLAMA_PIN_SHA:-}"

if [[ ! -d "$LLAMA_DIR" ]]; then
    if [[ -n "$LLAMA_PIN_SHA" ]]; then
        echo "Cloning michaelneale/llama.cpp pinned to $LLAMA_PIN_SHA..."
        # Shallow clone of upstream-latest first (the common case is that
        # $LLAMA_PIN_SHA == upstream-latest HEAD because ci.yml resolves it
        # via `git ls-remote ... refs/heads/upstream-latest`). If the branch
        # has moved between resolve and clone, fall back to fetching the
        # specific commit.
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
        # Pinned mode: do NOT pull. Fetch the requested SHA if missing and
        # check it out in detached HEAD. Skipping `git pull` is the whole
        # point — it keeps the working tree byte-identical to what the cache
        # key promises.
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

if [[ "$CLEAN" -eq 1 && -d "$BUILD_DIR" ]]; then
    echo "Cleaning build dir..."
    rm -rf "$BUILD_DIR"
fi

configure_compiler_cache "$BACKEND"

cmake_flags=(
    -B "$BUILD_DIR"
    -S "$LLAMA_DIR"
    -DGGML_RPC=ON
    -DBUILD_SHARED_LIBS=OFF
    -DLLAMA_OPENSSL=OFF
)

if [[ "$BACKEND" == "cpu" ]]; then
    cmake_flags+=(
        -DGGML_CUDA=OFF
        -DGGML_HIP=OFF
        -DGGML_VULKAN=OFF
        -DGGML_METAL=OFF
    )
elif [[ "$BACKEND" == "cuda" ]]; then
    # GGML_CUDA_FA_ALL_QUANTS compiles the full matrix of FlashAttention
    # kernels so mismatched K/V cache quantization types (e.g. K=q8_0, V=q4_0)
    # don't hit BEST_FATTN_KERNEL_NONE and crash the rpc-server.
    # Required for any asymmetric KV cache; the default (ON) is what user-
    # facing release artifacts must ship. Tracking:
    # https://github.com/ggml-org/llama.cpp/issues/20866
    #
    # CI may opt out via MESH_LLM_CUDA_FA_ALL_QUANTS=off because ci.yml does
    # only a --version smoke test on the CUDA binary and never exercises the
    # asymmetric KV cache path. Dropping the flag shrinks the FlashAttention
    # kernel matrix drastically (~177 fattn .cu instantiations \u2192 a fraction)
    # and cuts llama.cpp CUDA compile time significantly. NEVER use this
    # opt-out for release builds.
    CUDA_FA_ALL_QUANTS_FLAG="-DGGML_CUDA_FA_ALL_QUANTS=ON"
    if [[ "${MESH_LLM_CUDA_FA_ALL_QUANTS:-on}" == "off" ]]; then
        CUDA_FA_ALL_QUANTS_FLAG="-DGGML_CUDA_FA_ALL_QUANTS=OFF"
        echo "GGML_CUDA_FA_ALL_QUANTS disabled via MESH_LLM_CUDA_FA_ALL_QUANTS=off (CI opt-out)"
    fi
    cmake_flags+=(
        -DGGML_CUDA=ON
        "$CUDA_FA_ALL_QUANTS_FLAG"
        -DGGML_HIP=OFF
        -DGGML_VULKAN=OFF
        -DGGML_METAL=OFF
        -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH"
    )
elif [[ "$BACKEND" == "rocm" ]]; then
    if command -v hipconfig &>/dev/null; then
        export HIPCXX="$(hipconfig -l)/clang"
        export HIP_PATH="$(hipconfig -R)"
    fi
    cmake_flags+=(
        -DGGML_CUDA=OFF
        -DGGML_HIP=ON
        -DGGML_VULKAN=OFF
        -DGGML_METAL=OFF
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON
        -DAMDGPU_TARGETS="$ROCM_ARCH"
    )
else
    cmake_flags+=(
        -DGGML_CUDA=OFF
        -DGGML_HIP=OFF
        -DGGML_VULKAN=ON
        -DGGML_METAL=OFF
    )
fi

cmake_flags+=("${compiler_launcher_flags[@]}")

cmake "${cmake_flags[@]}"

# Post-configure assertion: guarantee the CUDA cmake cache reflects the
# intended GGML_CUDA_FA_ALL_QUANTS state. The default path must ship ON; the
# CI opt-out must explicitly pass MESH_LLM_CUDA_FA_ALL_QUANTS=off. Tracking:
# https://github.com/ggml-org/llama.cpp/issues/20866
if [[ "$BACKEND" == "cuda" ]]; then
    EXPECTED_FA_ALL_QUANTS="ON"
    if [[ "${MESH_LLM_CUDA_FA_ALL_QUANTS:-on}" == "off" ]]; then
        EXPECTED_FA_ALL_QUANTS="OFF"
    fi
    if ! grep -q "^GGML_CUDA_FA_ALL_QUANTS:BOOL=${EXPECTED_FA_ALL_QUANTS}" "$BUILD_DIR/CMakeCache.txt"; then
        echo "ERROR: GGML_CUDA_FA_ALL_QUANTS is not ${EXPECTED_FA_ALL_QUANTS} in $BUILD_DIR/CMakeCache.txt" >&2
        echo "       Expected state derived from MESH_LLM_CUDA_FA_ALL_QUANTS=${MESH_LLM_CUDA_FA_ALL_QUANTS:-on}." >&2
        echo "       Release builds MUST ship ON (asymmetric K/V cache crash risk)." >&2
        echo "       See scripts/build-linux.sh and ggml-org/llama.cpp#20866." >&2
        exit 1
    fi
fi

build_args=(
    --build "$BUILD_DIR"
    --config Release
    -j"$(nproc)"
)

if [[ -n "$LLAMA_TARGETS" ]]; then
    read -r -a target_array <<< "$LLAMA_TARGETS"
    if [[ "${#target_array[@]}" -gt 0 ]]; then
        echo "Limiting llama.cpp build targets to: ${target_array[*]}"
        build_args+=(--target "${target_array[@]}")
    fi
fi

cmake "${build_args[@]}"
echo "llama.cpp build complete: $BUILD_DIR/bin/"

if [[ -d "$MESH_DIR" ]]; then
    if [[ -d "$UI_DIR" ]]; then
        "$SCRIPT_DIR/build-ui.sh" "$UI_DIR"
    fi

    # MESH_LLM_BUILD_PROFILE=dev|debug lets CI opt into dev profile (single
    # target subdir, only the bin target — same shape as linux+macos jobs).
    # Default stays release so local `just build` is unchanged.
    if [[ "${MESH_LLM_BUILD_PROFILE:-release}" == "dev" || "${MESH_LLM_BUILD_PROFILE:-release}" == "debug" ]]; then
        echo "Building mesh-llm (profile: dev, bin only)..."
        (cd "$REPO_ROOT" && cargo build -p mesh-llm --bin mesh-llm)
        echo "Mesh binary: target/debug/mesh-llm"
    else
        echo "Building mesh-llm (profile: release)..."
        (cd "$MESH_DIR" && cargo build --release)
        echo "Mesh binary: target/release/mesh-llm"
    fi
fi
