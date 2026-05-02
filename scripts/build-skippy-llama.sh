#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

LLAMA_WORKDIR="${LLAMA_WORKDIR:-$ROOT/.deps/skippy-llama.cpp}"
LLAMA_BACKEND="${SKIPPY_LLAMA_BACKEND:-${LLAMA_BACKEND:-cpu}}"

case "$LLAMA_BACKEND" in
  cpu|cuda|rocm|hip|vulkan) ;;
  *)
    echo "unsupported SKIPPY_LLAMA_BACKEND: $LLAMA_BACKEND" >&2
    echo "expected one of: cpu, cuda, rocm, hip, vulkan" >&2
    exit 1
    ;;
esac

if [[ -z "${LLAMA_BUILD_DIR:-}" ]]; then
  if [[ "$LLAMA_BACKEND" == "cpu" ]]; then
    LLAMA_BUILD_DIR="$LLAMA_WORKDIR/build-stage-abi-static"
  else
    LLAMA_BUILD_DIR="$LLAMA_WORKDIR/build-stage-abi-$LLAMA_BACKEND"
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
  -DLLAMA_BUILD_EXAMPLES=OFF
  -DLLAMA_BUILD_TESTS=OFF
  -DLLAMA_CURL=OFF
)

case "$LLAMA_BACKEND" in
  cuda)
    CMAKE_ARGS+=(-DGGML_CUDA=ON)
    if [[ -n "${SKIPPY_CUDA_ARCHITECTURES:-}" ]]; then
      CMAKE_ARGS+=(-DCMAKE_CUDA_ARCHITECTURES="$SKIPPY_CUDA_ARCHITECTURES")
    fi
    ;;
  rocm|hip)
    CMAKE_ARGS+=(-DGGML_HIP=ON)
    if [[ -n "${SKIPPY_AMDGPU_TARGETS:-}" ]]; then
      CMAKE_ARGS+=(-DAMDGPU_TARGETS="$SKIPPY_AMDGPU_TARGETS")
    fi
    ;;
  vulkan)
    CMAKE_ARGS+=(-DGGML_VULKAN=ON)
    ;;
esac

if [[ "${SKIPPY_USE_SCCACHE:-1}" != "0" && -n "$SCCACHE_BIN" ]]; then
  CMAKE_ARGS+=(
    -DCMAKE_C_COMPILER_LAUNCHER="$SCCACHE_BIN"
    -DCMAKE_CXX_COMPILER_LAUNCHER="$SCCACHE_BIN"
  )
  echo "using sccache for llama.cpp C/C++ compilation: $SCCACHE_BIN"
elif [[ "${SKIPPY_USE_SCCACHE:-1}" != "0" ]]; then
  echo "sccache not found; llama.cpp build will run without compiler caching" >&2
fi

if [[ "$#" -gt 0 ]]; then
  CMAKE_ARGS+=("$@")
fi

cmake "${CMAKE_ARGS[@]}"

cmake --build "$LLAMA_BUILD_DIR" --config "${CMAKE_BUILD_TYPE:-Release}" --target llama llama-common mtmd

echo "built patched llama.cpp"
echo "  backend:   $LLAMA_BACKEND"
echo "  build dir: $LLAMA_BUILD_DIR"
