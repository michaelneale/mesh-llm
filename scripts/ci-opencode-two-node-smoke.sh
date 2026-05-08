#!/usr/bin/env bash
# ci-opencode-two-node-smoke.sh - run OpenCode smoke through a client node.
#
# Usage: scripts/ci-opencode-two-node-smoke.sh <mesh-llm-binary> <bin-dir> <model-path>
#
# The host serves the model. A passive client joins the host and exposes its own
# OpenAI-compatible API. The OpenCode smoke then targets the client API so
# requests exercise client -> host mesh routing and tunneling.

set -euo pipefail

MESH_LLM="${1:?Usage: $0 <mesh-llm-binary> <bin-dir> <model-path>}"
BIN_DIR="${2:?Usage: $0 <mesh-llm-binary> <bin-dir> <model-path>}"
MODEL="${3:?Usage: $0 <mesh-llm-binary> <bin-dir> <model-path>}"

HOST_API_PORT="${MESH_TWO_NODE_HOST_API_PORT:-9357}"
HOST_CONSOLE_PORT="${MESH_TWO_NODE_HOST_CONSOLE_PORT:-3151}"
HOST_BIND_PORT="${MESH_TWO_NODE_HOST_BIND_PORT:-53547}"
CLIENT_API_PORT="${MESH_TWO_NODE_CLIENT_API_PORT:-9358}"
CLIENT_CONSOLE_PORT="${MESH_TWO_NODE_CLIENT_CONSOLE_PORT:-3152}"
MAX_WAIT="${MESH_TWO_NODE_MAX_WAIT:-240}"
HOST_LOG="${MESH_TWO_NODE_HOST_LOG:-/tmp/mesh-llm-opencode-host.log}"
CLIENT_LOG="${MESH_TWO_NODE_CLIENT_LOG:-/tmp/mesh-llm-opencode-client.log}"
CLIENT_STABLE_PROBES="${MESH_TWO_NODE_CLIENT_STABLE_PROBES:-5}"

echo "=== CI OpenCode Two-Node Smoke ==="
echo "  mesh-llm:       $MESH_LLM"
echo "  bin-dir:        $BIN_DIR (compatibility placeholder)"
echo "  model:          $MODEL"
echo "  host api:       $HOST_API_PORT"
echo "  host console:   $HOST_CONSOLE_PORT"
echo "  host bind:      $HOST_BIND_PORT"
echo "  client api:     $CLIENT_API_PORT"
echo "  client console: $CLIENT_CONSOLE_PORT"
echo "  stable probes:  $CLIENT_STABLE_PROBES"

if [[ ! -x "$MESH_LLM" ]]; then
    echo "Missing executable mesh-llm binary: $MESH_LLM" >&2
    exit 1
fi
if [[ ! -f "$MODEL" ]]; then
    echo "Missing model: $MODEL" >&2
    exit 1
fi

descendant_pids() {
    local pid="$1"
    local children
    children="$(pgrep -P "$pid" 2>/dev/null || true)"
    for child in $children; do
        descendant_pids "$child"
        printf '%s\n' "$child"
    done
}

kill_tree() {
    local pid="${1:-}"
    [[ -n "$pid" ]] || return 0
    local children
    children="$(descendant_pids "$pid" | sort -u || true)"
    kill "$pid" 2>/dev/null || true
    if [[ -n "$children" ]]; then
        printf '%s\n' "$children" | xargs kill 2>/dev/null || true
    fi
    sleep 1
    kill -9 "$pid" 2>/dev/null || true
    if [[ -n "$children" ]]; then
        printf '%s\n' "$children" | xargs kill -9 2>/dev/null || true
    fi
    wait "$pid" 2>/dev/null || true
}

HOST_PID=""
CLIENT_PID=""
cleanup() {
    kill_tree "$CLIENT_PID"
    kill_tree "$HOST_PID"
    echo "--- host log tail ---"
    tail -120 "$HOST_LOG" 2>/dev/null || true
    echo "--- client log tail ---"
    tail -120 "$CLIENT_LOG" 2>/dev/null || true
    echo "--- end logs ---"
}
trap cleanup EXIT

"$MESH_LLM" \
    serve \
    --model "$MODEL" \
    --no-draft \
    --device CPU \
    --ctx-size "${MESH_TWO_NODE_CTX_SIZE:-1024}" \
    --port "$HOST_API_PORT" \
    --console "$HOST_CONSOLE_PORT" \
    --bind-port "$HOST_BIND_PORT" \
    --headless \
    >"$HOST_LOG" 2>&1 &
HOST_PID=$!

TOKEN=""
HOST_MODEL_ID=""
for i in $(seq 1 "$MAX_WAIT"); do
    if ! kill -0 "$HOST_PID" 2>/dev/null; then
        echo "host exited unexpectedly" >&2
        tail -120 "$HOST_LOG" >&2 || true
        exit 1
    fi

    STATUS_JSON="$(curl -sf "http://127.0.0.1:${HOST_CONSOLE_PORT}/api/status" 2>/dev/null || true)"
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
        MODELS_JSON="$(curl -sf "http://127.0.0.1:${HOST_API_PORT}/v1/models")"
        HOST_MODEL_ID="$(
            printf '%s' "$MODELS_JSON" | python3 -c 'import json,sys
data=json.load(sys.stdin).get("data", [])
print(data[0].get("id", "") if data else "")'
        )"
        if [[ -n "$HOST_MODEL_ID" ]]; then
            echo "Host ready after ${i}s with model: $HOST_MODEL_ID"
            break
        fi
    fi

    if [[ "$i" -eq "$MAX_WAIT" ]]; then
        echo "timed out waiting for host readiness" >&2
        tail -120 "$HOST_LOG" >&2 || true
        exit 1
    fi
    sleep 1
done

"$MESH_LLM" \
    client \
    --join "$TOKEN" \
    --port "$CLIENT_API_PORT" \
    --console "$CLIENT_CONSOLE_PORT" \
    --headless \
    >"$CLIENT_LOG" 2>&1 &
CLIENT_PID=$!

CLIENT_STABLE_COUNT=0
for i in $(seq 1 "$MAX_WAIT"); do
    if ! kill -0 "$CLIENT_PID" 2>/dev/null; then
        echo "client exited unexpectedly" >&2
        tail -120 "$CLIENT_LOG" >&2 || true
        exit 1
    fi

    CLIENT_MODELS_JSON="$(curl -sf "http://127.0.0.1:${CLIENT_API_PORT}/v1/models" 2>/dev/null || true)"
    if python3 - "$HOST_MODEL_ID" "$CLIENT_MODELS_JSON" <<'PY' 2>/dev/null; then
import json
import sys

expected = sys.argv[1]
data = json.loads(sys.argv[2]).get("data", [])
if not any(item.get("id") == expected for item in data):
    raise SystemExit(1)
PY
        CLIENT_STABLE_COUNT=$((CLIENT_STABLE_COUNT + 1))
        if [[ "$CLIENT_STABLE_COUNT" -ge "$CLIENT_STABLE_PROBES" ]]; then
            echo "Client routed /v1/models stably after ${i}s"
            break
        fi
    else
        CLIENT_STABLE_COUNT=0
    fi

    if [[ "$i" -eq "$MAX_WAIT" ]]; then
        echo "timed out waiting for stable client /v1/models to include host model" >&2
        echo "$CLIENT_MODELS_JSON" >&2
        tail -120 "$CLIENT_LOG" >&2 || true
        exit 1
    fi
    sleep 1
done

MESH_OPENCODE_BASE_URL="http://127.0.0.1:${CLIENT_API_PORT}/v1" \
MESH_OPENCODE_MODEL="$HOST_MODEL_ID" \
scripts/ci-opencode-smoke.sh

echo "OpenCode two-node smoke passed"
