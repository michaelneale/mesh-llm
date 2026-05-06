#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="${SKIPPY_CACHE_RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_DIR="${1:-/tmp/skippy-cache-family-bench-${RUN_ID}}"

PREFIX_TOKENS="${PREFIX_TOKENS:-128}"
RUNTIME_LANE_COUNT="${RUNTIME_LANE_COUNT:-1}"
LLAMA_PARALLEL="${LLAMA_PARALLEL:-1}"
LLAMA_REPEATS="${LLAMA_REPEATS:-3}"
CACHE_HIT_REPEATS="${CACHE_HIT_REPEATS:-3}"
LLAMA_STAGE_BUILD_DIR="${LLAMA_STAGE_BUILD_DIR:-${ROOT}/.deps/llama-build/build-stage-abi-cpu}"
LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-${LLAMA_STAGE_BUILD_DIR}/bin/llama-server}"
SKIPPY_CORRECTNESS_BIN="${SKIPPY_CORRECTNESS_BIN:-${ROOT}/target/debug/skippy-correctness}"

FULL_DIR="${OUTPUT_DIR}/full-gguf"
USECASE_DIR="${OUTPUT_DIR}/use-cases"
REPORT="${OUTPUT_DIR}/readme-tables.md"

if [[ "${SKIPPY_CACHE_SKIP_BUILD:-0}" != "1" ]]; then
  (cd "$ROOT" && just build)
  (cd "$ROOT" && cargo build -p skippy-correctness)
fi

COMMON_ARGS=(
  --runtime-lane-count "$RUNTIME_LANE_COUNT"
  --llama-parallel "$LLAMA_PARALLEL"
  --llama-repeats "$LLAMA_REPEATS"
  --cache-hit-repeats "$CACHE_HIT_REPEATS"
  --prefix-tokens "$PREFIX_TOKENS"
  --llama-stage-build-dir "$LLAMA_STAGE_BUILD_DIR"
  --llama-server-bin "$LLAMA_SERVER_BIN"
  --skippy-correctness-bin "$SKIPPY_CORRECTNESS_BIN"
)

(cd "$ROOT" && python3 evals/skippy-cache-production-bench.py \
  --output-dir "$FULL_DIR" \
  "${COMMON_ARGS[@]}")

(cd "$ROOT" && python3 evals/skippy-cache-production-bench.py \
  --output-dir "$USECASE_DIR" \
  --use-case all \
  "${COMMON_ARGS[@]}")

(cd "$ROOT" && python3 evals/skippy-cache-family-report.py \
  --input "${FULL_DIR}/production-cache-bench.json" \
  --input "${USECASE_DIR}/production-cache-bench.json" \
  --output "$REPORT")

printf 'Wrote raw full-GGUF results: %s\n' "${FULL_DIR}/production-cache-bench.json"
printf 'Wrote raw use-case results: %s\n' "${USECASE_DIR}/production-cache-bench.json"
printf 'Wrote README tables: %s\n' "$REPORT"
