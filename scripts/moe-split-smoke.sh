#!/usr/bin/env bash
# moe-split-smoke.sh — direct smoke tests for llama-moe-split across known MoE families.
#
# Usage:
#   scripts/moe-split-smoke.sh <bin-dir> [family...]
#   scripts/moe-split-smoke.sh <bin-dir> --strict all
#
# Families:
#   qwen3-a3b
#   qwen3-next
#   glm-deepseek2
#   olmoe
#   qwen35moe
#   all
#
# The script:
#   1. resolves a known model path from local cache
#   2. produces 2-way shards for group 0 and group 1
#   3. validates each shard by loading it with llama-server
#
# Missing families are skipped by default. Pass --strict to fail when a requested
# family cannot be resolved locally.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/moe-split-smoke.sh <bin-dir> [options] [family...]

Options:
  --strict           Fail if a requested family is not found locally
  --keep-artifacts   Keep the temp directory instead of deleting it
  --ctx-size <n>     Context size to use during shard validation (default: 4096)
  --max-wait <secs>  Max seconds to wait for shard health (default: 180)
  --help             Show this help

Families:
  qwen3-a3b
  qwen3-next
  glm-deepseek2
  olmoe
  qwen35moe
  all
EOF
}

if [[ $# -lt 1 ]]; then
    usage >&2
    exit 1
fi

BIN_DIR="$1"
shift

STRICT=0
KEEP_ARTIFACTS=0
CTX_SIZE="${CTX_SIZE:-4096}"
MAX_WAIT="${MAX_WAIT:-180}"
declare -a FAMILIES=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --strict)
            STRICT=1
            shift
            ;;
        --keep-artifacts)
            KEEP_ARTIFACTS=1
            shift
            ;;
        --ctx-size)
            CTX_SIZE="$2"
            shift 2
            ;;
        --max-wait)
            MAX_WAIT="$2"
            shift 2
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            FAMILIES+=("$1")
            shift
            ;;
    esac
done

if [[ ${#FAMILIES[@]} -eq 0 ]]; then
    FAMILIES=("all")
fi

if [[ "${FAMILIES[0]}" == "all" ]]; then
    FAMILIES=("qwen3-a3b" "qwen3-next" "glm-deepseek2" "olmoe" "qwen35moe")
fi

LLAMA_MOE_SPLIT="$BIN_DIR/llama-moe-split"
LLAMA_SERVER="$BIN_DIR/llama-server"
if [[ ! -x "$LLAMA_MOE_SPLIT" ]]; then
    echo "❌ Missing binary: $LLAMA_MOE_SPLIT" >&2
    exit 1
fi
if [[ ! -x "$LLAMA_SERVER" ]]; then
    echo "❌ Missing binary: $LLAMA_SERVER" >&2
    exit 1
fi

HF_HUB_ROOT="${HF_HOME:-$HOME/.cache/huggingface}/hub"
MODEL_DIR_FALLBACK="$HOME/.models"
WORKDIR="$(mktemp -d /tmp/moe-split-smoke.XXXXXX)"

cleanup() {
    pkill -f "[/]llama-server .*${WORKDIR}" 2>/dev/null || true
    if [[ "$KEEP_ARTIFACTS" -ne 1 ]]; then
        rm -rf "$WORKDIR"
    else
        echo "Keeping artifacts in $WORKDIR"
    fi
}
trap cleanup EXIT

find_model_by_name() {
    local candidate="$1"
    local path=""

    if [[ -d "$HF_HUB_ROOT" ]]; then
        path="$(find "$HF_HUB_ROOT" -type f -name "$candidate" 2>/dev/null | head -n 1 || true)"
        if [[ -n "$path" ]]; then
            printf '%s\n' "$path"
            return 0
        fi
    fi

    if [[ -d "$MODEL_DIR_FALLBACK" ]]; then
        path="$(find "$MODEL_DIR_FALLBACK" -type f -name "$candidate" 2>/dev/null | head -n 1 || true)"
        if [[ -n "$path" ]]; then
            printf '%s\n' "$path"
            return 0
        fi
    fi

    return 1
}

resolve_family() {
    local family="$1"

    case "$family" in
        qwen3-a3b)
            printf '%s\n' "${MOE_SPLIT_SMOKE_QWEN3_A3B_MODEL:-$(find_model_by_name "Qwen3-30B-A3B-Q4_K_M.gguf" || true)}"
            ;;
        qwen3-next)
            printf '%s\n' "${MOE_SPLIT_SMOKE_QWEN3_NEXT_MODEL:-$(find_model_by_name "Qwen3-Coder-Next-Q4_K_M-00001-of-00004.gguf" || true)}"
            ;;
        glm-deepseek2)
            printf '%s\n' "${MOE_SPLIT_SMOKE_GLM_MODEL:-$(find_model_by_name "GLM-4.7-Flash-Q4_K_M.gguf" || true)}"
            ;;
        olmoe)
            if [[ -n "${MOE_SPLIT_SMOKE_OLMOE_MODEL:-}" ]]; then
                printf '%s\n' "$MOE_SPLIT_SMOKE_OLMOE_MODEL"
                return 0
            fi
            local path=""
            path="$(find_model_by_name "OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf" || true)"
            if [[ -z "$path" ]]; then
                path="$(find_model_by_name "OLMoE-1B-7B-0125-Instruct-Q4_K_M.gguf" || true)"
            fi
            printf '%s\n' "$path"
            ;;
        qwen35moe)
            printf '%s\n' "${MOE_SPLIT_SMOKE_QWEN35MOE_MODEL:-$(find_model_by_name "qwen3.5-moe-0.87B-d0.8B-Q2_K.gguf" || true)}"
            ;;
        *)
            echo "❌ Unknown family: $family" >&2
            return 1
            ;;
    esac
}

wait_for_shard_health() {
    local port="$1"
    local pid="$2"
    local log="$3"

    for i in $(seq 1 "$MAX_WAIT"); do
        if curl -sf "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
            return 0
        fi
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "  ❌ llama-server exited before shard became healthy"
            tail -n 80 "$log" || true
            return 1
        fi
        sleep 1
    done

    echo "  ❌ timed out waiting for shard health after ${MAX_WAIT}s"
    tail -n 80 "$log" || true
    return 1
}

validate_group() {
    local family="$1"
    local model="$2"
    local group_id="$3"
    local idx="$4"

    local family_dir="$WORKDIR/$family"
    mkdir -p "$family_dir"

    local shard_path="$family_dir/group-${group_id}.gguf"
    local split_log="$family_dir/group-${group_id}.split.log"
    local load_log="$family_dir/group-${group_id}.load.log"
    local port=$((19100 + idx * 10 + group_id))

    echo "  → splitting group ${group_id}"
    if ! "$LLAMA_MOE_SPLIT" \
        -m "$model" \
        --groups 2 \
        --group-id "$group_id" \
        --output "$shard_path" \
        >"$split_log" 2>&1; then
        echo "  ❌ llama-moe-split failed for ${family} group ${group_id}"
        tail -n 80 "$split_log" || true
        return 1
    fi

    echo "  → validating group ${group_id}"
    "$LLAMA_SERVER" \
        -m "$shard_path" \
        --host 127.0.0.1 \
        --port "$port" \
        --ctx-size "$CTX_SIZE" \
        --no-webui \
        --no-warmup \
        >"$load_log" 2>&1 &
    local pid=$!

    if ! wait_for_shard_health "$port" "$pid" "$load_log"; then
        kill "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
        return 1
    fi

    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    echo "  ✅ group ${group_id} healthy"
}

declare -a PASSED=()
declare -a FAILED=()
declare -a SKIPPED=()

echo "=== moe-split family smoke ==="
echo "  bin-dir:  $BIN_DIR"
echo "  workdir:  $WORKDIR"
echo "  families: ${FAMILIES[*]}"

family_index=0
for family in "${FAMILIES[@]}"; do
    family_index=$((family_index + 1))
    model_path="$(resolve_family "$family")"
    if [[ -z "$model_path" ]]; then
        if [[ "$STRICT" -eq 1 ]]; then
            echo "❌ missing model for family: $family"
            exit 1
        fi
        echo "⚠️  skipping $family (model not found locally)"
        SKIPPED+=("$family")
        continue
    fi

    echo
    echo "=== $family ==="
    echo "  model: $model_path"

    if validate_group "$family" "$model_path" 0 "$family_index" \
        && validate_group "$family" "$model_path" 1 "$family_index"; then
        PASSED+=("$family")
    else
        FAILED+=("$family")
    fi
done

echo
echo "=== Summary ==="
echo "  passed:  ${PASSED[*]:-none}"
echo "  failed:  ${FAILED[*]:-none}"
echo "  skipped: ${SKIPPED[*]:-none}"

if [[ ${#FAILED[@]} -gt 0 ]]; then
    exit 1
fi

echo "✅ moe-split smoke passed"
