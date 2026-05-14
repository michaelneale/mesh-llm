#!/usr/bin/env bash
# ci-two-node-client-serving-smoke.sh - verify client-to-serving-node routing.
#
# Usage: scripts/ci-two-node-client-serving-smoke.sh <mesh-llm-binary> <bin-dir> <model-path>
#
# The host serves the model. A passive client joins the host and exposes its own
# OpenAI-compatible API. The smoke targets the client API so requests exercise
# client -> host mesh routing and tunneling.

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
HOST_LOG="${MESH_TWO_NODE_HOST_LOG:-/tmp/mesh-llm-two-node-host.log}"
CLIENT_LOG="${MESH_TWO_NODE_CLIENT_LOG:-/tmp/mesh-llm-two-node-client.log}"
CLIENT_STABLE_PROBES="${MESH_TWO_NODE_CLIENT_STABLE_PROBES:-5}"

echo "=== CI Two-Node Client/Serving Smoke ==="
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
    --log-format json \
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
    --log-format json \
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

WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/mesh-two-node-client-serving.XXXXXX")"
CHAT_RESPONSE="${WORK_DIR}/chat-response.json"
STREAM_RESPONSE="${WORK_DIR}/chat-stream.txt"

python3 - "$HOST_MODEL_ID" "${WORK_DIR}/chat-payload.json" <<'PY'
import json
import sys

model, path = sys.argv[1:3]
payload = {
    "model": model,
    "messages": [
        {"role": "system", "content": "You are a terse CI smoke probe."},
        {"role": "user", "content": "Reply with one short sentence."},
    ],
    "stream": False,
    "max_tokens": 16,
    "temperature": 0,
}
with open(path, "w", encoding="utf-8") as fh:
    json.dump(payload, fh)
PY

curl -fsS --max-time 120 \
    "http://127.0.0.1:${CLIENT_API_PORT}/v1/chat/completions" \
    -H 'content-type: application/json' \
    -d @"${WORK_DIR}/chat-payload.json" \
    -o "$CHAT_RESPONSE"

python3 - "$CHAT_RESPONSE" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as fh:
    body = json.load(fh)
if body.get("object") != "chat.completion":
    raise SystemExit(f"unexpected chat object: {body.get('object')!r}")
if not isinstance(body.get("id"), str) or not body["id"]:
    raise SystemExit("chat response id was not a non-empty string")
if not body.get("choices"):
    raise SystemExit("chat response had no choices")
print("Client-routed non-stream chat response validated")
PY

python3 - "$HOST_MODEL_ID" "${WORK_DIR}/stream-payload.json" <<'PY'
import json
import sys

model, path = sys.argv[1:3]
payload = {
    "model": model,
    "messages": [{"role": "user", "content": "Say ok."}],
    "stream": True,
    "max_tokens": 8,
    "temperature": 0,
}
with open(path, "w", encoding="utf-8") as fh:
    json.dump(payload, fh)
PY

curl -fsS --max-time 120 \
    "http://127.0.0.1:${CLIENT_API_PORT}/v1/chat/completions" \
    -H 'content-type: application/json' \
    -d @"${WORK_DIR}/stream-payload.json" \
    -o "$STREAM_RESPONSE"

python3 - "$STREAM_RESPONSE" <<'PY'
import json
import sys

top_ids = set()
chunk_count = 0
done = False
for line in open(sys.argv[1], encoding="utf-8", errors="replace"):
    line = line.strip()
    if not line.startswith("data: "):
        continue
    data = line[6:]
    if data == "[DONE]":
        done = True
        continue
    body = json.loads(data)
    if not isinstance(body.get("id"), str) or not body["id"]:
        raise SystemExit("stream chunk id was not a non-empty string")
    top_ids.add(body["id"])
    chunk_count += 1

if chunk_count == 0:
    raise SystemExit("stream response had no chunks")
if len(top_ids) != 1:
    raise SystemExit(f"stream response used unstable top-level ids: {sorted(top_ids)}")
if not done:
    raise SystemExit("stream response did not finish with [DONE]")
print("Client-routed streaming chat response validated")
PY

echo "Two-node client/serving smoke passed"
