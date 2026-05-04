#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

LLAMA_WORKDIR="${LLAMA_WORKDIR:-$ROOT/.deps/llama.cpp}"
LLAMA_BUILD_ROOT="${MESH_LLM_LLAMA_BUILD_ROOT:-$ROOT/.deps/llama-build}"
LLAMA_BACKEND="${LLAMA_STAGE_BACKEND:-${SKIPPY_LLAMA_BACKEND:-${LLAMA_BACKEND:-cpu}}}"

case "$LLAMA_BACKEND" in
  cpu|cuda|rocm|hip|vulkan|metal) ;;
  *)
    echo "unsupported LLAMA_STAGE_BACKEND: $LLAMA_BACKEND" >&2
    echo "expected one of: cpu, cuda, rocm, hip, vulkan, metal" >&2
    exit 1
    ;;
esac

sanitize_build_component() {
  printf '%s' "$1" | tr ';, /:' '_____' | tr -cd 'A-Za-z0-9_.-'
}

default_build_dir_for_backend() {
  local suffix="$LLAMA_BACKEND"
  case "$LLAMA_BACKEND" in
    cpu)
      suffix="cpu"
      ;;
    cuda)
      local cuda_arch="${LLAMA_STAGE_CUDA_ARCHITECTURES:-${SKIPPY_CUDA_ARCHITECTURES:-}}"
      suffix="cuda-sm$(sanitize_build_component "$cuda_arch")"
      ;;
    rocm|hip)
      local amdgpu_targets="${LLAMA_STAGE_AMDGPU_TARGETS:-${SKIPPY_AMDGPU_TARGETS:-}}"
      suffix="rocm-$(sanitize_build_component "$amdgpu_targets")"
      ;;
  esac
  printf '%s/build-stage-abi-%s\n' "$LLAMA_BUILD_ROOT" "$suffix"
}

detect_jobs() {
  if [[ -n "${CMAKE_BUILD_PARALLEL_LEVEL:-}" ]]; then
    echo "$CMAKE_BUILD_PARALLEL_LEVEL"
  elif command -v nproc >/dev/null 2>&1; then
    nproc
  elif command -v sysctl >/dev/null 2>&1; then
    sysctl -n hw.ncpu
  else
    echo 4
  fi
}

if [[ -z "${LLAMA_BUILD_DIR:-}" ]]; then
  if [[ -n "${LLAMA_STAGE_BUILD_DIR:-}" ]]; then
    LLAMA_BUILD_DIR="$LLAMA_STAGE_BUILD_DIR"
  elif [[ -n "${SKIPPY_LLAMA_BUILD_DIR:-}" ]]; then
    LLAMA_BUILD_DIR="$SKIPPY_LLAMA_BUILD_DIR"
  else
    LLAMA_BUILD_DIR="$(default_build_dir_for_backend)"
  fi
fi

if [[ ! -d "$LLAMA_WORKDIR/.git" ]]; then
  echo "llama checkout not found: $LLAMA_WORKDIR" >&2
    echo "run: just llama-prepare" >&2
  exit 1
fi

SCCACHE_BIN="${SCCACHE:-${SCCACHE_PATH:-}}"
if [[ -z "$SCCACHE_BIN" ]] && command -v sccache >/dev/null 2>&1; then
  SCCACHE_BIN="$(command -v sccache)"
fi

CMAKE_ARGS=(
  -S "$LLAMA_WORKDIR"
  -B "$LLAMA_BUILD_DIR"
  -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"
  -DBUILD_SHARED_LIBS=OFF
  -DGGML_NATIVE="${LLAMA_STAGE_GGML_NATIVE:-${SKIPPY_GGML_NATIVE:-OFF}}"
  -DLLAMA_BUILD_EXAMPLES=OFF
  -DLLAMA_BUILD_TESTS=OFF
  -DLLAMA_CURL=OFF
)

if command -v ninja >/dev/null 2>&1; then
  CMAKE_ARGS=(-G Ninja "${CMAKE_ARGS[@]}")
  echo "using CMake generator: Ninja"
fi

case "$LLAMA_BACKEND" in
  cuda)
    CMAKE_ARGS+=(-DGGML_CUDA=ON)
    CUDA_ARCHITECTURES="${LLAMA_STAGE_CUDA_ARCHITECTURES:-${SKIPPY_CUDA_ARCHITECTURES:-}}"
    if [[ -n "$CUDA_ARCHITECTURES" ]]; then
      CMAKE_ARGS+=(-DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHITECTURES")
    fi
    ;;
  rocm|hip)
    CMAKE_ARGS+=(-DGGML_HIP=ON)
    AMDGPU_TARGETS="${LLAMA_STAGE_AMDGPU_TARGETS:-${SKIPPY_AMDGPU_TARGETS:-}}"
    if [[ -n "$AMDGPU_TARGETS" ]]; then
      CMAKE_ARGS+=(-DAMDGPU_TARGETS="$AMDGPU_TARGETS")
    fi
    ;;
  vulkan)
    CMAKE_ARGS+=(-DGGML_VULKAN=ON)
    ;;
  metal)
    CMAKE_ARGS+=(-DGGML_METAL=ON)
    ;;
esac

USE_SCCACHE="${LLAMA_STAGE_USE_SCCACHE:-${SKIPPY_USE_SCCACHE:-1}}"
if [[ "$USE_SCCACHE" != "0" && -n "$SCCACHE_BIN" ]]; then
  CMAKE_ARGS+=(
    -DCMAKE_C_COMPILER_LAUNCHER="$SCCACHE_BIN"
    -DCMAKE_CXX_COMPILER_LAUNCHER="$SCCACHE_BIN"
  )
  case "$LLAMA_BACKEND" in
    cuda)
      CMAKE_ARGS+=(-DCMAKE_CUDA_COMPILER_LAUNCHER="$SCCACHE_BIN")
      ;;
    rocm|hip)
      CMAKE_ARGS+=(-DCMAKE_HIP_COMPILER_LAUNCHER="$SCCACHE_BIN")
      ;;
  esac
  echo "using sccache for llama.cpp C/C++ compilation: $SCCACHE_BIN"
elif [[ "$USE_SCCACHE" != "0" ]]; then
  echo "sccache not found; llama.cpp build will run without compiler caching" >&2
else
  CMAKE_ARGS+=(-DGGML_CCACHE=OFF)
fi

if [[ "$#" -gt 0 ]]; then
  CMAKE_ARGS+=("$@")
fi

cmake "${CMAKE_ARGS[@]}"

cmake --build "$LLAMA_BUILD_DIR" --config "${CMAKE_BUILD_TYPE:-Release}" --parallel "$(detect_jobs)" --target llama llama-common mtmd

echo "built patched llama.cpp"
echo "  backend:   $LLAMA_BACKEND"
echo "  build dir: $LLAMA_BUILD_DIR"
