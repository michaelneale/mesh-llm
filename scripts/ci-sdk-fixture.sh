#!/usr/bin/env bash
# Start a skippy-backed mesh-llm node and expose environment for SDK smoke tests.

set -euo pipefail

if [[ "$#" -lt 5 ]]; then
    echo "Usage: $0 <mesh-llm-binary> <bin-dir> <model-path> -- <command...>" >&2
    exit 1
fi

MESH_LLM="$1"
BIN_DIR="$2"
MODEL="$3"
shift 3

if [[ "${1:-}" != "--" ]]; then
    echo "Usage: $0 <mesh-llm-binary> <bin-dir> <model-path> -- <command...>" >&2
    exit 1
fi
shift

API_PORT="${MESH_SDK_API_PORT:-9347}"
CONSOLE_PORT="${MESH_SDK_CONSOLE_PORT:-3141}"
MAX_WAIT="${MESH_SDK_MAX_WAIT:-180}"
LOG="${MESH_SDK_LOG:-/tmp/mesh-llm-sdk-ci.log}"

echo "=== SDK Fixture ==="
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
    echo "Missing model: $MODEL" >&2
    exit 1
fi

"$MESH_LLM" \
    serve \
    --model "$MODEL" \
    --no-draft \
    --device CPU \
    --ctx-size "${MESH_SDK_CTX_SIZE:-256}" \
    --port "$API_PORT" \
    --console "$CONSOLE_PORT" \
    >"$LOG" 2>&1 &
MESH_PID=$!

descendant_pids() {
    local pid="$1"
    local children
    children="$(pgrep -P "$pid" 2>/dev/null || true)"
    for child in $children; do
        descendant_pids "$child"
        printf '%s\n' "$child"
    done
}

cleanup() {
    local children
    children="$(descendant_pids "$MESH_PID" | sort -u || true)"
    kill "$MESH_PID" 2>/dev/null || true
    if [[ -n "$children" ]]; then
        printf '%s\n' "$children" | xargs kill 2>/dev/null || true
    fi
    sleep 1
    kill -9 "$MESH_PID" 2>/dev/null || true
    if [[ -n "$children" ]]; then
        printf '%s\n' "$children" | xargs kill -9 2>/dev/null || true
    fi
    wait "$MESH_PID" 2>/dev/null || true
    echo "--- SDK fixture log tail ---"
    tail -100 "$LOG" 2>/dev/null || true
    echo "--- end log ---"
}
trap cleanup EXIT

STATUS_JSON=""
TOKEN=""
for i in $(seq 1 "$MAX_WAIT"); do
    if ! kill -0 "$MESH_PID" 2>/dev/null; then
        echo "mesh-llm exited unexpectedly" >&2
        tail -120 "$LOG" >&2 || true
        exit 1
    fi

    STATUS_JSON="$(curl -sf "http://127.0.0.1:${CONSOLE_PORT}/api/status" 2>/dev/null || true)"
    READY="$(
        printf '%s' "$STATUS_JSON" | python3 -c 'import json,sys
try:
    print(json.load(sys.stdin).get("llama_ready", False))
except Exception:
    print(False)' 2>/dev/null || echo "False"
    )"
    TOKEN="$(
        printf '%s' "$STATUS_JSON" | python3 -c 'import json,sys
try:
    print(json.load(sys.stdin).get("token", ""))
except Exception:
    print("")' 2>/dev/null || echo ""
    )"

    if [[ "$READY" == "True" && -n "$TOKEN" ]]; then
        break
    fi

    if [[ "$i" -eq "$MAX_WAIT" ]]; then
        echo "Timed out waiting for SDK fixture readiness" >&2
        tail -120 "$LOG" >&2 || true
        exit 1
    fi
    sleep 1
done

MODELS_JSON="$(curl -sf "http://127.0.0.1:${API_PORT}/v1/models")"
MODEL_ID="$(
    printf '%s' "$MODELS_JSON" | python3 -c 'import json,sys
data = json.load(sys.stdin).get("data", [])
print(data[0]["id"] if data else "")'
)"

if [[ -z "$MODEL_ID" ]]; then
    echo "No models returned from /v1/models" >&2
    printf '%s\n' "$MODELS_JSON" >&2
    exit 1
fi

export MESH_SDK_INVITE_TOKEN="$TOKEN"
export MESH_SDK_MODEL_ID="$MODEL_ID"
export MESH_SDK_API_PORT="$API_PORT"
export MESH_SDK_CONSOLE_PORT="$CONSOLE_PORT"
export MESH_CLIENT_API_BASE="http://127.0.0.1:${API_PORT}"

echo "SDK fixture ready:"
echo "  model: $MODEL_ID"

"$@"
