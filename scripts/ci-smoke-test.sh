#!/usr/bin/env bash
# ci-smoke-test.sh — start mesh-llm with a tiny model, run one inference request, shut down.
#
# Usage: scripts/ci-smoke-test.sh <mesh-llm-binary> <bin-dir> <model-path> [mmproj-path]
#
# Expects llama-server and rpc-server in <bin-dir>.
# Exits 0 on success, 1 on failure.

set -euo pipefail

MESH_LLM="$1"
BIN_DIR="$2"
MODEL="$3"
MMPROJ="${4:-}"
API_PORT=9337
CONSOLE_PORT=3131
MAX_WAIT=180  # seconds to wait for model load on CPU
LOG=/tmp/mesh-llm-ci.log

echo "=== CI Smoke Test ==="
echo "  mesh-llm:  $MESH_LLM"
echo "  bin-dir:   $BIN_DIR"
echo "  model:     $MODEL"
if [ -n "$MMPROJ" ]; then
    echo "  mmproj:    $MMPROJ"
fi
echo "  api port:  $API_PORT"
echo "  os:        $(uname -s)"

# Verify binaries exist
ls -la "$BIN_DIR"/rpc-server* "$BIN_DIR"/llama-server* 2>/dev/null || true
if [ ! -f "$MESH_LLM" ]; then
    echo "❌ Missing mesh-llm binary: $MESH_LLM"
    exit 1
fi

# Start mesh-llm in background
ARGS=(
    serve
    --model "$MODEL"
    --no-draft
    --bin-dir "$BIN_DIR"
    --device CPU
    --port "$API_PORT"
    --console "$CONSOLE_PORT"
)

if [ -n "$MMPROJ" ]; then
    ARGS+=(--mmproj "$MMPROJ")
fi

echo "Starting mesh-llm..."
"$MESH_LLM" "${ARGS[@]}" > "$LOG" 2>&1 &
MESH_PID=$!
echo "  PID: $MESH_PID"

cleanup() {
    echo "Shutting down mesh-llm (PID $MESH_PID)..."
    kill "$MESH_PID" 2>/dev/null || true
    # Also kill any child processes
    pkill -P "$MESH_PID" 2>/dev/null || true
    # Give them a moment then force-kill stragglers
    sleep 2
    kill -9 "$MESH_PID" 2>/dev/null || true
    pkill -9 -f rpc-server 2>/dev/null || true
    pkill -9 -f llama-server 2>/dev/null || true
    wait "$MESH_PID" 2>/dev/null || true
    echo "Cleanup done."
}
trap cleanup EXIT

# Wait for llama_ready
echo "Waiting for model to load (up to ${MAX_WAIT}s)..."
for i in $(seq 1 "$MAX_WAIT"); do
    if ! kill -0 "$MESH_PID" 2>/dev/null; then
        echo "❌ mesh-llm exited unexpectedly"
        echo "--- Log tail ---"
        tail -50 "$LOG" || true
        exit 1
    fi

    READY=$(curl -sf "http://localhost:${CONSOLE_PORT}/api/status" 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin).get('llama_ready', False))" 2>/dev/null || echo "False")
    if [ "$READY" = "True" ]; then
        echo "✅ Model loaded in ${i}s"
        break
    fi

    if [ "$i" -eq "$MAX_WAIT" ]; then
        echo "❌ Model failed to load within ${MAX_WAIT}s"
        echo "--- Log tail ---"
        tail -80 "$LOG" || true
        exit 1
    fi

    if [ $((i % 15)) -eq 0 ]; then
        echo "  Still waiting... (${i}s)"
    fi
    sleep 1
done

# Test inference
echo "Testing /v1/chat/completions..."
RESPONSE=$(curl -sf "http://localhost:${API_PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "any",
        "messages": [{"role": "user", "content": "Say hello in exactly 3 words."}],
        "max_tokens": 32,
        "temperature": 0
    }' 2>&1)

if [ $? -ne 0 ]; then
    echo "❌ Inference request failed"
    echo "$RESPONSE"
    echo "--- Log tail ---"
    tail -50 "$LOG" || true
    exit 1
fi

# Verify response has content
CONTENT=$(echo "$RESPONSE" | python3 -c "import sys,json; r=json.load(sys.stdin); print(r['choices'][0]['message']['content'])" 2>/dev/null || echo "")
if [ -z "$CONTENT" ]; then
    echo "❌ Empty response from inference"
    echo "Raw response: $RESPONSE"
    exit 1
fi

echo "✅ Inference response: $CONTENT"

# Test model=auto with hooks — routes through smart router with inter-model
# collaboration hooks enabled. No peers available so hooks return action:none,
# model generates normally.
echo "Testing model=auto (virtual LLM hooks)..."
AUTO_HOOK_RESPONSE=$(curl -sf "http://localhost:${API_PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "auto",
        "messages": [{"role": "user", "content": "Say hi."}],
        "max_tokens": 16,
        "temperature": 0
    }' 2>&1)

AUTO_HOOK_CONTENT=$(echo "$AUTO_HOOK_RESPONSE" | python3 -c "import sys,json; r=json.load(sys.stdin); print(r['choices'][0]['message']['content'])" 2>/dev/null || echo "")
if [ -z "$AUTO_HOOK_CONTENT" ]; then
    echo "❌ model=auto returned empty response"
    echo "Raw: $AUTO_HOOK_RESPONSE"
    echo "--- Log tail ---"
    tail -30 "$LOG" || true
    exit 1
fi
echo "✅ model=auto response: $AUTO_HOOK_CONTENT"

# Test /v1/models endpoint
echo "Testing /v1/models..."
MODELS=$(curl -sf "http://localhost:${API_PORT}/v1/models" 2>&1)
MODEL_COUNT=$(echo "$MODELS" | python3 -c "import sys,json; print(len(json.load(sys.stdin).get('data',[])))" 2>/dev/null || echo "0")
if [ "$MODEL_COUNT" -eq 0 ]; then
    echo "❌ No models in /v1/models"
    echo "$MODELS"
    exit 1
fi
echo "✅ /v1/models returned $MODEL_COUNT model(s)"



echo ""
echo "=== All smoke tests passed ==="
