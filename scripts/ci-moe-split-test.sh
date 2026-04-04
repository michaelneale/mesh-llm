#!/usr/bin/env bash
# ci-moe-split-test.sh — verify MoE expert splitting produces valid, loadable shards.
#
# Downloads a small MoE model (~598MB Q2_K), splits it into 2 shards using
# llama-moe-split, then validates each shard by starting llama-server and
# confirming it reaches healthy state and produces inference tokens.
#
# Usage: scripts/ci-moe-split-test.sh <bin-dir> <model-path>
#
# Expects llama-moe-split and llama-server in <bin-dir>.
# Exits 0 on success, 1 on failure.

set -euo pipefail

BIN_DIR="$1"
MODEL="$2"

LLAMA_MOE_SPLIT="$BIN_DIR/llama-moe-split"
LLAMA_SERVER="$BIN_DIR/llama-server"
SHARD0_PORT=19200
SHARD1_PORT=19201
MAX_WAIT=120

WORKDIR="$(mktemp -d /tmp/ci-moe-split-test.XXXXXX)"

echo "=== CI MoE Split Test ==="
echo "  bin-dir:  $BIN_DIR"
echo "  model:    $MODEL"
echo "  workdir:  $WORKDIR"
echo "  os:       $(uname -s)"

if [[ ! -x "$LLAMA_MOE_SPLIT" ]]; then
    echo "❌ Missing binary: $LLAMA_MOE_SPLIT"
    exit 1
fi
if [[ ! -x "$LLAMA_SERVER" ]]; then
    echo "❌ Missing binary: $LLAMA_SERVER"
    exit 1
fi
if [[ ! -f "$MODEL" ]]; then
    echo "❌ Missing model: $MODEL"
    exit 1
fi

SHARD0_PID=""
SHARD1_PID=""

cleanup() {
    [[ -n "$SHARD0_PID" ]] && kill "$SHARD0_PID" 2>/dev/null || true
    [[ -n "$SHARD1_PID" ]] && kill "$SHARD1_PID" 2>/dev/null || true
    wait "$SHARD0_PID" 2>/dev/null || true
    wait "$SHARD1_PID" 2>/dev/null || true
    rm -rf "$WORKDIR"
}
trap cleanup EXIT

# ── Step 1: Split into 2 shards ──
echo ""
echo "--- Step 1: Splitting model into 2 shards ---"

"$LLAMA_MOE_SPLIT" \
    -m "$MODEL" \
    --groups 2 \
    --group-id 0 \
    -o "$WORKDIR/shard-0.gguf" \
    2>&1 | tee "$WORKDIR/split-0.log" | grep -E 'Arch|Expert|Group|Done|error|warning'

if [[ ! -f "$WORKDIR/shard-0.gguf" ]]; then
    echo "❌ Shard 0 not produced"
    exit 1
fi

"$LLAMA_MOE_SPLIT" \
    -m "$MODEL" \
    --groups 2 \
    --group-id 1 \
    -o "$WORKDIR/shard-1.gguf" \
    2>&1 | tee "$WORKDIR/split-1.log" | grep -E 'Arch|Expert|Group|Done|error|warning'

if [[ ! -f "$WORKDIR/shard-1.gguf" ]]; then
    echo "❌ Shard 1 not produced"
    exit 1
fi

echo "✅ Both shards produced"
ls -lh "$WORKDIR"/shard-*.gguf

# ── Step 2: Load each shard and test inference ──

wait_healthy() {
    local port="$1"
    local pid="$2"
    local label="$3"
    local log="$4"

    for i in $(seq 1 "$MAX_WAIT"); do
        if curl -sf "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
            echo "  ✅ $label healthy (${i}s)"
            return 0
        fi
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "  ❌ $label: llama-server exited unexpectedly"
            tail -20 "$log"
            return 1
        fi
        sleep 1
    done

    echo "  ❌ $label: timed out after ${MAX_WAIT}s"
    tail -20 "$log"
    return 1
}

test_inference() {
    local port="$1"
    local label="$2"

    local resp
    resp=$(curl -sf "http://127.0.0.1:${port}/v1/completions" \
        -H "Content-Type: application/json" \
        -d '{"model":"test","prompt":"1+1=","max_tokens":8,"temperature":0}' 2>&1) || {
        echo "  ❌ $label: inference request failed"
        return 1
    }

    local tokens
    tokens=$(echo "$resp" | python3 -c "import json,sys; print(json.load(sys.stdin)['usage']['completion_tokens'])" 2>/dev/null) || {
        echo "  ❌ $label: failed to parse response: $resp"
        return 1
    }

    if [[ "$tokens" -gt 0 ]]; then
        echo "  ✅ $label: produced $tokens tokens"
        return 0
    else
        echo "  ❌ $label: zero tokens produced"
        return 1
    fi
}

echo ""
echo "--- Step 2: Loading shard 0 ---"
"$LLAMA_SERVER" \
    -m "$WORKDIR/shard-0.gguf" \
    --host 127.0.0.1 \
    --port "$SHARD0_PORT" \
    -c 512 \
    --no-webui \
    --no-warmup \
    >"$WORKDIR/server-0.log" 2>&1 &
SHARD0_PID=$!

wait_healthy "$SHARD0_PORT" "$SHARD0_PID" "Shard 0" "$WORKDIR/server-0.log" || exit 1
test_inference "$SHARD0_PORT" "Shard 0" || exit 1

kill "$SHARD0_PID" 2>/dev/null; wait "$SHARD0_PID" 2>/dev/null || true
SHARD0_PID=""

echo ""
echo "--- Step 3: Loading shard 1 ---"
"$LLAMA_SERVER" \
    -m "$WORKDIR/shard-1.gguf" \
    --host 127.0.0.1 \
    --port "$SHARD1_PORT" \
    -c 512 \
    --no-webui \
    --no-warmup \
    >"$WORKDIR/server-1.log" 2>&1 &
SHARD1_PID=$!

wait_healthy "$SHARD1_PORT" "$SHARD1_PID" "Shard 1" "$WORKDIR/server-1.log" || exit 1
test_inference "$SHARD1_PORT" "Shard 1" || exit 1

kill "$SHARD1_PID" 2>/dev/null; wait "$SHARD1_PID" 2>/dev/null || true
SHARD1_PID=""

echo ""
echo "✅ CI MoE split test passed — both shards split, load, and serve inference"
