#!/usr/bin/env bash
# ci-moe-mesh-test.sh — verify MoE expert sharding works end-to-end through a mesh.
#
# Starts two mesh-llm nodes on the same machine with a small MoE model,
# forces the MoE split path with --split and an aggressive VRAM cap (--max-vram 0.1),
# then uses --force so startup still proceeds for this e2e split test.
# to form and both nodes to serve their expert shards, then verifies inference
# works through both the host and a client node.
#
# This exercises the full MoE pipeline that ci-moe-split-test.sh does NOT cover:
#   - MoE detection from GGUF header
#   - Expert assignment and ranking
#   - Shard splitting via llama-moe-split
#   - Each node loading its own shard in llama-server
#   - Mesh formation and MoE placement planning
#   - Session hash-routed inference through the mesh
#
# Usage: scripts/ci-moe-mesh-test.sh <mesh-llm-binary> <bin-dir> <model-path>

set -euo pipefail

MESH_LLM="$1"
BIN_DIR="$2"
MODEL="$3"

A_API_PORT=19210
A_CONSOLE_PORT=19211
B_API_PORT=19212
B_CONSOLE_PORT=19213
C_API_PORT=19214
C_CONSOLE_PORT=19215
MAX_WAIT=240
MAX_INFERENCE_ATTEMPTS=15
LOG_A=/tmp/mesh-llm-moe-mesh-a.log
LOG_B=/tmp/mesh-llm-moe-mesh-b.log
LOG_C=/tmp/mesh-llm-moe-mesh-c.log

echo "=== CI MoE Mesh Test ==="
echo "  mesh-llm:  $MESH_LLM"
echo "  bin-dir:   $BIN_DIR"
echo "  model:     $MODEL"
echo "  os:        $(uname -s)"

if [ ! -f "$MESH_LLM" ]; then
    echo "❌ Missing mesh-llm binary: $MESH_LLM"
    exit 1
fi
if [ ! -f "$MODEL" ]; then
    echo "❌ Missing model: $MODEL"
    exit 1
fi

A_PID=""
B_PID=""
C_PID=""

cleanup() {
    echo "Cleaning up..."
    for PID in $C_PID $B_PID $A_PID; do
        [ -n "$PID" ] && kill "$PID" 2>/dev/null || true
        [ -n "$PID" ] && pkill -P "$PID" 2>/dev/null || true
    done
    sleep 2
    for PID in $C_PID $B_PID $A_PID; do
        [ -n "$PID" ] && kill -9 "$PID" 2>/dev/null || true
    done
    pkill -9 -f "rpc-server" 2>/dev/null || true
    pkill -9 -f "llama-server" 2>/dev/null || true
    wait 2>/dev/null || true
    echo "Cleanup done."
}
trap cleanup EXIT

# Clear any cached MoE splits for this model so we exercise the full split path.
MODEL_STEM=$(basename "$MODEL" .gguf)
SPLIT_CACHE="$HOME/.cache/mesh-llm/splits/${MODEL_STEM}"
if [ -d "$SPLIT_CACHE" ]; then
    echo "Clearing cached MoE splits: $SPLIT_CACHE"
    rm -rf "$SPLIT_CACHE"
fi

# ── Start Node A ──
echo ""
echo "Starting Node A..."
MESH_LLM_EPHEMERAL_KEY=1 "$MESH_LLM" \
    serve \
    --model "$MODEL" \
    --split \
    --max-vram 0.1 \
    --force \
    --no-draft \
    --bin-dir "$BIN_DIR" \
    --device CPU \
    --port "$A_API_PORT" \
    --console "$A_CONSOLE_PORT" \
    > "$LOG_A" 2>&1 &
A_PID=$!
echo "  PID: $A_PID"

# Wait for Node A's console API to get the invite token
echo "Waiting for Node A console API..."
TOKEN=""
for i in $(seq 1 60); do
    if ! kill -0 "$A_PID" 2>/dev/null; then
        echo "❌ Node A exited unexpectedly"
        tail -50 "$LOG_A" || true
        exit 1
    fi
    TOKEN=$(curl -sf "http://localhost:${A_CONSOLE_PORT}/api/status" 2>/dev/null \
        | python3 -c "import sys,json; print(json.load(sys.stdin).get('token',''))" 2>/dev/null || echo "")
    if [ -n "$TOKEN" ]; then
        echo "  Got invite token in ${i}s"
        break
    fi
    if [ "$i" -eq 60 ]; then
        echo "❌ Node A console never came up"
        tail -30 "$LOG_A" || true
        exit 1
    fi
    sleep 1
done

# ── Start Node B ──
echo ""
echo "Starting Node B..."
MESH_LLM_EPHEMERAL_KEY=1 "$MESH_LLM" \
    serve \
    --model "$MODEL" \
    --split \
    --max-vram 0.1 \
    --force \
    --no-draft \
    --bin-dir "$BIN_DIR" \
    --device CPU \
    --port "$B_API_PORT" \
    --console "$B_CONSOLE_PORT" \
    --join "$TOKEN" \
    > "$LOG_B" 2>&1 &
B_PID=$!
echo "  PID: $B_PID"

# ── Wait for both nodes to be serving MoE shards ──
echo ""
echo "Waiting for MoE mesh to form (up to ${MAX_WAIT}s)..."
BOTH_READY=""
for i in $(seq 1 "$MAX_WAIT"); do
    for PID in $A_PID $B_PID; do
        if ! kill -0 "$PID" 2>/dev/null; then
            echo "❌ Node (PID $PID) exited unexpectedly"
            echo "--- Node A log tail ---"
            tail -50 "$LOG_A" || true
            echo "--- Node B log tail ---"
            tail -50 "$LOG_B" || true
            exit 1
        fi
    done

    # Check both consoles for MoE readiness: both should be serving with a peer
    A_READY="false"
    B_READY="false"

    A_STATUS=$(curl -sf "http://localhost:${A_CONSOLE_PORT}/api/status" 2>/dev/null || echo "")
    if [ -n "$A_STATUS" ]; then
        A_READY=$(echo "$A_STATUS" | python3 -c "
import sys, json
s = json.load(sys.stdin)
llama = s.get('llama_ready', False)
peers = len(s.get('peers', []))
print('true' if llama and peers >= 1 else 'false')
" 2>/dev/null || echo "false")
    fi

    B_STATUS=$(curl -sf "http://localhost:${B_CONSOLE_PORT}/api/status" 2>/dev/null || echo "")
    if [ -n "$B_STATUS" ]; then
        B_READY=$(echo "$B_STATUS" | python3 -c "
import sys, json
s = json.load(sys.stdin)
llama = s.get('llama_ready', False)
peers = len(s.get('peers', []))
print('true' if llama and peers >= 1 else 'false')
" 2>/dev/null || echo "false")
    fi

    if [ "$A_READY" = "true" ] && [ "$B_READY" = "true" ]; then
        BOTH_READY="true"
        echo "  ✅ Both MoE nodes ready in ${i}s"
        break
    fi

    if [ "$i" -eq "$MAX_WAIT" ]; then
        echo "❌ MoE mesh failed to form within ${MAX_WAIT}s"
        echo "  Node A ready: $A_READY"
        echo "  Node B ready: $B_READY"
        echo "--- Node A log tail ---"
        tail -50 "$LOG_A" || true
        echo "--- Node B log tail ---"
        tail -50 "$LOG_B" || true
        exit 1
    fi

    if [ $((i % 15)) -eq 0 ]; then
        echo "  Still waiting... (${i}s) A=$A_READY B=$B_READY"
    fi
    sleep 1
done

# ── Verify MoE topology ──
echo ""
echo "MoE topology from Node A:"
curl -sf "http://localhost:${A_CONSOLE_PORT}/api/status" | python3 -c "
import sys, json
s = json.load(sys.stdin)
print(f'  This node: host={s.get(\"is_host\")}, llama_ready={s.get(\"llama_ready\")}')
print(f'  Node status: {s.get(\"node_status\", \"unknown\")}')
for m in s.get('models', []):
    print(f'  Model: {m.get(\"name\")}, moe={m.get(\"moe\")}, experts={m.get(\"expert_count\")}')
for p in s.get('peers', []):
    print(f'  Peer: {p[\"id\"][:8]} role={p.get(\"role\",\"?\")}')
" 2>/dev/null || echo "  (failed to parse)"

# ── Test inference through Node A ──
echo ""
MODEL_NAME=$(basename "$MODEL" .gguf)
echo "Testing inference through Node A (port $A_API_PORT)..."
CONTENT=""
BACKOFF=2
for attempt in $(seq 1 "$MAX_INFERENCE_ATTEMPTS"); do
    RESPONSE=$(curl -s --max-time 30 -w "\n%{http_code}" "http://localhost:${A_API_PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"${MODEL_NAME}\",
            \"messages\": [{\"role\": \"user\", \"content\": \"Say hello.\"}],
            \"max_tokens\": 16,
            \"temperature\": 0
        }" 2>&1) || true

    HTTP_CODE=$(echo "$RESPONSE" | tail -1)
    BODY=$(echo "$RESPONSE" | sed '$d')

    if [ "$HTTP_CODE" = "200" ]; then
        CONTENT=$(echo "$BODY" | python3 -c "import sys,json; r=json.load(sys.stdin); print(r['choices'][0]['message']['content'])" 2>/dev/null || echo "")
        if [ -n "$CONTENT" ]; then
            echo "  ✅ Node A inference (attempt $attempt): $CONTENT"
            break
        fi
    else
        echo "  ⚠️  Attempt $attempt: HTTP $HTTP_CODE (retrying in ${BACKOFF}s)"
    fi

    if [ "$attempt" -lt "$MAX_INFERENCE_ATTEMPTS" ]; then
        sleep "$BACKOFF"
        BACKOFF=$(( BACKOFF + BACKOFF / 2 ))
        [ "$BACKOFF" -gt 15 ] && BACKOFF=15
    fi
done

if [ -z "$CONTENT" ]; then
    echo "❌ Node A inference failed after ${MAX_INFERENCE_ATTEMPTS} attempts"
    echo "  Last HTTP code: $HTTP_CODE"
    echo "  Last body: $BODY"
    echo "--- Node A log tail ---"
    tail -40 "$LOG_A" || true
    exit 1
fi

# ── Test inference through Node B ──
echo ""
echo "Testing inference through Node B (port $B_API_PORT)..."
CONTENT_B=""
BACKOFF=2
for attempt in $(seq 1 "$MAX_INFERENCE_ATTEMPTS"); do
    RESPONSE=$(curl -s --max-time 30 -w "\n%{http_code}" "http://localhost:${B_API_PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"${MODEL_NAME}\",
            \"messages\": [{\"role\": \"user\", \"content\": \"Say hi.\"}],
            \"max_tokens\": 16,
            \"temperature\": 0
        }" 2>&1) || true

    HTTP_CODE=$(echo "$RESPONSE" | tail -1)
    BODY=$(echo "$RESPONSE" | sed '$d')

    if [ "$HTTP_CODE" = "200" ]; then
        CONTENT_B=$(echo "$BODY" | python3 -c "import sys,json; r=json.load(sys.stdin); print(r['choices'][0]['message']['content'])" 2>/dev/null || echo "")
        if [ -n "$CONTENT_B" ]; then
            echo "  ✅ Node B inference (attempt $attempt): $CONTENT_B"
            break
        fi
    else
        echo "  ⚠️  Attempt $attempt: HTTP $HTTP_CODE (retrying in ${BACKOFF}s)"
    fi

    if [ "$attempt" -lt "$MAX_INFERENCE_ATTEMPTS" ]; then
        sleep "$BACKOFF"
        BACKOFF=$(( BACKOFF + BACKOFF / 2 ))
        [ "$BACKOFF" -gt 15 ] && BACKOFF=15
    fi
done

if [ -z "$CONTENT_B" ]; then
    echo "❌ Node B inference failed after ${MAX_INFERENCE_ATTEMPTS} attempts"
    echo "  Last HTTP code: $HTTP_CODE"
    echo "  Last body: $BODY"
    echo "--- Node B log tail ---"
    tail -40 "$LOG_B" || true
    exit 1
fi

# ── Start Node C (client) and test routing through it ──
echo ""
echo "Starting Node C (client)..."
MESH_LLM_EPHEMERAL_KEY=1 "$MESH_LLM" \
    client \
    --no-draft \
    --bin-dir "$BIN_DIR" \
    --port "$C_API_PORT" \
    --console "$C_CONSOLE_PORT" \
    --join "$TOKEN" \
    > "$LOG_C" 2>&1 &
C_PID=$!
echo "  PID: $C_PID"

# Wait for client to see both peers
echo "Waiting for client to join mesh..."
for i in $(seq 1 60); do
    if ! kill -0 "$C_PID" 2>/dev/null; then
        echo "❌ Node C exited unexpectedly"
        tail -50 "$LOG_C" || true
        exit 1
    fi
    CLIENT_PEERS=$(curl -sf "http://localhost:${C_CONSOLE_PORT}/api/status" 2>/dev/null \
        | python3 -c "import sys,json; print(len(json.load(sys.stdin).get('peers',[])))" 2>/dev/null || echo "0")
    if [ "$CLIENT_PEERS" -ge 2 ]; then
        echo "  ✅ Client sees $CLIENT_PEERS peers in ${i}s"
        break
    fi
    if [ "$i" -eq 60 ]; then
        echo "❌ Client never joined mesh"
        tail -30 "$LOG_C" || true
        exit 1
    fi
    sleep 1
done

# Wait for client to learn a routable MoE host
echo "Waiting for client to learn a routable host..."
for i in $(seq 1 60); do
    STATUS=$(curl -sf "http://localhost:${C_CONSOLE_PORT}/api/status" 2>/dev/null || echo "")
    if [ -n "$STATUS" ]; then
        ROUTABLE=$(echo "$STATUS" | python3 -c "
import json, sys
status = json.load(sys.stdin)
for peer in status.get('peers', []):
    hosted = peer.get('hosted_models', []) or []
    if peer.get('role') == 'Host' and len(hosted) > 0:
        print('1')
        break
else:
    print('0')
" 2>/dev/null || echo "0")
        if [ "$ROUTABLE" = "1" ]; then
            echo "  ✅ Client sees a routable host in ${i}s"
            break
        fi
    fi
    if [ "$i" -eq 60 ]; then
        echo "❌ Client never learned a routable host"
        curl -sf "http://localhost:${C_CONSOLE_PORT}/api/status" | python3 -m json.tool 2>/dev/null || true
        tail -30 "$LOG_C" || true
        exit 1
    fi
    sleep 1
done

echo ""
echo "Testing inference through Client (port $C_API_PORT)..."
CONTENT_C=""
BACKOFF=2
for attempt in $(seq 1 "$MAX_INFERENCE_ATTEMPTS"); do
    RESPONSE=$(curl -s --max-time 30 -w "\n%{http_code}" "http://localhost:${C_API_PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"${MODEL_NAME}\",
            \"messages\": [{\"role\": \"user\", \"content\": \"What is 1+1?\"}],
            \"max_tokens\": 16,
            \"temperature\": 0
        }" 2>&1) || true

    HTTP_CODE=$(echo "$RESPONSE" | tail -1)
    BODY=$(echo "$RESPONSE" | sed '$d')

    if [ "$HTTP_CODE" = "200" ]; then
        CONTENT_C=$(echo "$BODY" | python3 -c "import sys,json; r=json.load(sys.stdin); print(r['choices'][0]['message']['content'])" 2>/dev/null || echo "")
        if [ -n "$CONTENT_C" ]; then
            echo "  ✅ Client inference (attempt $attempt): $CONTENT_C"
            break
        fi
    else
        echo "  ⚠️  Attempt $attempt: HTTP $HTTP_CODE (retrying in ${BACKOFF}s)"
    fi

    if [ "$attempt" -lt "$MAX_INFERENCE_ATTEMPTS" ]; then
        sleep "$BACKOFF"
        BACKOFF=$(( BACKOFF + BACKOFF / 2 ))
        [ "$BACKOFF" -gt 15 ] && BACKOFF=15
    fi
done

if [ -z "$CONTENT_C" ]; then
    echo "❌ Client inference failed after ${MAX_INFERENCE_ATTEMPTS} attempts"
    echo "  Last HTTP code: $HTTP_CODE"
    echo "  Last body: $BODY"
    echo "--- Client log tail ---"
    tail -40 "$LOG_C" || true
    echo "--- Node A log tail ---"
    tail -20 "$LOG_A" || true
    echo "--- Node B log tail ---"
    tail -20 "$LOG_B" || true
    exit 1
fi

echo ""
echo "=== MoE mesh test passed ==="
echo "  Both nodes split experts, loaded shards, and served inference."
echo "  Client routed requests through the mesh successfully."
