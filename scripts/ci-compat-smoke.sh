#!/usr/bin/env bash
# ci-compat-smoke.sh - run OpenAI client compatibility probes against mesh-llm.
#
# Usage: scripts/ci-compat-smoke.sh <mesh-llm-binary> <bin-dir> <model-path>

set -euo pipefail

MESH_LLM="${1:?Usage: $0 <mesh-llm-binary> <bin-dir> <model-path>}"
BIN_DIR="${2:?Usage: $0 <mesh-llm-binary> <bin-dir> <model-path>}"
MODEL="${3:?Usage: $0 <mesh-llm-binary> <bin-dir> <model-path>}"
API_PORT="${MESH_COMPAT_API_PORT:-9348}"
CONSOLE_PORT="${MESH_COMPAT_CONSOLE_PORT:-3142}"
MAX_WAIT="${MESH_COMPAT_MAX_WAIT:-180}"
LOG="${MESH_COMPAT_LOG:-/tmp/mesh-llm-compat-ci.log}"

echo "=== CI OpenAI Compatibility Smoke ==="
echo "  mesh-llm:  $MESH_LLM"
echo "  bin-dir:   $BIN_DIR (compatibility placeholder)"
echo "  model:     $MODEL"
echo "  api port:  $API_PORT"
echo "  console:   $CONSOLE_PORT"

if [[ ! -x "$MESH_LLM" ]]; then
    echo "Missing executable mesh-llm binary: $MESH_LLM" >&2
    exit 1
fi

if [[ ! -f "$MODEL" ]]; then
    echo "Missing model file: $MODEL" >&2
    exit 1
fi

"$MESH_LLM" \
    serve \
    --model "$MODEL" \
    --no-draft \
    --device CPU \
    --ctx-size "${MESH_COMPAT_CTX_SIZE:-256}" \
    --port "$API_PORT" \
    --console "$CONSOLE_PORT" \
    --headless \
    >"$LOG" 2>&1 &
MESH_PID=$!

cleanup() {
    kill "$MESH_PID" 2>/dev/null || true
    pkill -P "$MESH_PID" 2>/dev/null || true
    sleep 1
    kill -9 "$MESH_PID" 2>/dev/null || true
    wait "$MESH_PID" 2>/dev/null || true
    echo "--- compat mesh-llm log tail ---"
    tail -100 "$LOG" 2>/dev/null || true
    echo "--- end log ---"
}
trap cleanup EXIT

BASE_URL="http://127.0.0.1:${API_PORT}/v1"
MODEL_ID=""
for i in $(seq 1 "$MAX_WAIT"); do
    if ! kill -0 "$MESH_PID" 2>/dev/null; then
        echo "mesh-llm exited unexpectedly" >&2
        tail -120 "$LOG" >&2 || true
        exit 1
    fi

    MODELS_JSON="$(curl -sf "${BASE_URL}/models" 2>/dev/null || true)"
    MODEL_ID="$(
        printf '%s' "$MODELS_JSON" |
            python3 -c 'import json,sys; data=json.load(sys.stdin).get("data", []); print(data[0].get("id", "") if data else "")' 2>/dev/null ||
            echo ""
    )"
    if [[ -n "$MODEL_ID" ]]; then
        echo "OpenAI endpoint ready with model: $MODEL_ID"
        break
    fi

    if [[ "$i" -eq "$MAX_WAIT" ]]; then
        echo "Timed out waiting for OpenAI endpoint" >&2
        tail -120 "$LOG" >&2 || true
        exit 1
    fi
    sleep 1
done

python3 scripts/ci-openai-python-smoke.py --base-url "$BASE_URL"
python3 scripts/ci-litellm-smoke.py --base-url "$BASE_URL" --model "$MODEL_ID"
python3 scripts/ci-langchain-openai-smoke.py --base-url "$BASE_URL" --model "$MODEL_ID"
NODE_PATH="${NODE_PATH:-$(npm root -g 2>/dev/null || true)}" node scripts/ci-openai-node-smoke.cjs --base-url "$BASE_URL"

echo "OpenAI compatibility smoke passed"
