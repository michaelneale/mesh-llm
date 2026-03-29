#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
BENCH_SCRIPT="$SCRIPT_DIR/bench-backends.sh"
DEFAULT_MATRIX_FILE="$SCRIPT_DIR/model-matrix.json"
DEFAULT_OUTPUT_ROOT="$REPO_DIR/evals/results/backend-benchmarking/matrix-$(date +%Y%m%d-%H%M%S)"

MATRIX_FILE="$DEFAULT_MATRIX_FILE"
OUTPUT_ROOT="$DEFAULT_OUTPUT_ROOT"
RUNS=3
WARMUP_RUNS=1
CASES_FILE=""
ONLY_ID=""
INCLUDE_DISABLED=0
REQUIRE_ALL=0

usage() {
    cat <<EOF
Usage:
  $(basename "$0") [options]

Options:
  --matrix-file PATH       Model matrix JSON (default: $DEFAULT_MATRIX_FILE)
  --output-root PATH       Output root dir (default: $DEFAULT_OUTPUT_ROOT)
  --runs N                 Measured runs per case/backend (default: $RUNS)
  --warmup-runs N          Warmup runs per case/backend (default: $WARMUP_RUNS)
  --cases-file PATH        Override benchmark cases file
  --only ID                Run one matrix entry by id
  --include-disabled       Include entries with "enabled": false
  --require-all            Fail if any selected model pair is missing
  --help                   Show this help
EOF
}

expand_path() {
    python3 - "$1" <<'PY'
import os
import sys
print(os.path.expandvars(os.path.expanduser(sys.argv[1])))
PY
}

resolve_matrix_relative_path() {
    local raw="$1"
    local base_dir="$2"
    if [[ -z "$raw" ]]; then
        echo ""
        return 0
    fi
    python3 - "$raw" "$base_dir" <<'PY'
import os
import sys

raw = os.path.expandvars(os.path.expanduser(sys.argv[1]))
base_dir = sys.argv[2]
if os.path.isabs(raw):
    print(raw)
else:
    print(os.path.normpath(os.path.join(base_dir, raw)))
PY
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --matrix-file)
            MATRIX_FILE="$2"
            shift 2
            ;;
        --output-root)
            OUTPUT_ROOT="$2"
            shift 2
            ;;
        --runs)
            RUNS="$2"
            shift 2
            ;;
        --warmup-runs)
            WARMUP_RUNS="$2"
            shift 2
            ;;
        --cases-file)
            CASES_FILE="$2"
            shift 2
            ;;
        --only)
            ONLY_ID="$2"
            shift 2
            ;;
        --include-disabled)
            INCLUDE_DISABLED=1
            shift
            ;;
        --require-all)
            REQUIRE_ALL=1
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage
            exit 1
            ;;
    esac
done

if [[ ! -f "$MATRIX_FILE" ]]; then
    echo "Missing matrix file: $MATRIX_FILE" >&2
    exit 1
fi

MATRIX_DIR="$(cd "$(dirname "$MATRIX_FILE")" && pwd)"

mkdir -p "$OUTPUT_ROOT"

MATRIX_ROWS=()
while IFS= read -r line; do
    MATRIX_ROWS+=("$line")
done < <(
    python3 - "$MATRIX_FILE" "$ONLY_ID" "$INCLUDE_DISABLED" <<'PY'
import json
import sys

matrix_path = sys.argv[1]
only_id = sys.argv[2]
include_disabled = sys.argv[3] == "1"

with open(matrix_path, "r", encoding="utf-8") as fh:
    rows = json.load(fh)

for row in rows:
    if only_id and row["id"] != only_id:
        continue
    if not include_disabled and not row.get("enabled", False):
        continue
    phases = row.get("phases") or [{}]
    for phase in phases:
        print("\t".join([
            row["id"],
            str(row.get("enabled", False)).lower(),
            row.get("notes", ""),
            row["llama_model"],
            row["mlx_model"],
            phase.get("id", "main"),
            phase.get("notes", ""),
            phase.get("cases_file", ""),
            str(phase.get("warmup_runs", "")),
        ]))
PY
)

if [[ "${#MATRIX_ROWS[@]}" -eq 0 ]]; then
    echo "No matrix entries selected." >&2
    exit 1
fi

echo "Output root: $OUTPUT_ROOT"
echo ""

for row in "${MATRIX_ROWS[@]}"; do
    IFS=$'\t' read -r entry_id enabled notes llama_model_raw mlx_model_raw phase_id phase_notes phase_cases_file_raw phase_warmup_runs <<<"$row"
    llama_model="$(expand_path "$llama_model_raw")"
    mlx_model="$(expand_path "$mlx_model_raw")"
    phase_cases_file="$(resolve_matrix_relative_path "$phase_cases_file_raw" "$MATRIX_DIR")"

    echo "=== $entry_id ==="
    echo "notes: $notes"
    echo "phase: $phase_id"
    if [[ -n "$phase_notes" ]]; then
        echo "phase notes: $phase_notes"
    fi

    missing=0
    if [[ ! -e "$llama_model" ]]; then
        echo "skip: missing llama model $llama_model"
        missing=1
    fi
    if [[ ! -e "$mlx_model" ]]; then
        echo "skip: missing mlx model $mlx_model"
        missing=1
    fi

    if [[ "$missing" == "1" ]]; then
        if [[ "$REQUIRE_ALL" == "1" ]]; then
            exit 1
        fi
        echo ""
        continue
    fi

    pair_output="$OUTPUT_ROOT/$entry_id/$phase_id"
    selected_warmup_runs="$WARMUP_RUNS"
    if [[ -n "$phase_warmup_runs" ]]; then
        selected_warmup_runs="$phase_warmup_runs"
    fi
    cmd=(
        "$BENCH_SCRIPT"
        --llama-model "$llama_model"
        --mlx-model "$mlx_model"
        --runs "$RUNS"
        --warmup-runs "$selected_warmup_runs"
        --output-dir "$pair_output"
    )
    if [[ -n "$CASES_FILE" ]]; then
        cmd+=(--cases-file "$CASES_FILE")
    elif [[ -n "$phase_cases_file" ]]; then
        cmd+=(--cases-file "$phase_cases_file")
    fi

    "${cmd[@]}"
    echo ""
done

echo "Matrix run complete: $OUTPUT_ROOT"
