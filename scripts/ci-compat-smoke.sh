#!/usr/bin/env bash
# ci-compat-smoke.sh — start a single mesh-llm node, then run SDK compatibility smokes.
#
# This test validates that openai-python, openai-node, litellm, and langchain-openai
# all work correctly against mesh-llm's OpenAI-compatible API. It uses a solo node
# (no split, no mesh) since split-mode routing is already tested by ci-split-test.sh.
#
# Usage: scripts/ci-compat-smoke.sh <mesh-llm-binary> <bin-dir> <model-path> [mmproj-path]

set -euo pipefail

MESH_LLM="$1"
BIN_DIR="$2"
MODEL="$3"
MMPROJ="${4:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
NODE_BIN="${NODE_BIN:-node}"
NPM_BIN="${NPM_BIN:-npm}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

API_PORT=9337
CONSOLE_PORT=3131
MAX_WAIT=240
WORKDIR="$(mktemp -d)"
LOG="$WORKDIR/mesh-llm.log"
NODE_SDK_DIR="$WORKDIR/openai-node"

echo "=== Compat Smoke Test ==="
echo "  mesh-llm:   $MESH_LLM"
echo "  bin-dir:    $BIN_DIR"
echo "  model:      $MODEL"
if [ -n "$MMPROJ" ]; then
    echo "  mmproj:     $MMPROJ"
fi
echo "  workdir:    $WORKDIR"

if [ ! -f "$MESH_LLM" ]; then
    echo "❌ Missing mesh-llm binary: $MESH_LLM"
    exit 1
fi

cleanup() {
    set +e
    if [ -n "${MESH_PID:-}" ]; then
        kill "$MESH_PID" 2>/dev/null || true
        pkill -P "$MESH_PID" 2>/dev/null || true
    fi
    sleep 2
    if [ -n "${MESH_PID:-}" ]; then
        kill -9 "$MESH_PID" 2>/dev/null || true
    fi
    pkill -9 -f "[/]rpc-server" 2>/dev/null || true
    pkill -9 -f "[/]llama-server" 2>/dev/null || true
    rm -rf "$WORKDIR"
}
trap cleanup EXIT

fail_with_logs() {
    local message="$1"
    echo "❌ $message"
    echo "--- mesh-llm log ---"
    tail -80 "$LOG" 2>/dev/null || true
    exit 1
}

assert_pid_alive() {
    local pid="$1"
    local name="$2"
    if ! kill -0 "$pid" 2>/dev/null; then
        fail_with_logs "$name exited unexpectedly"
    fi
}

# ── Start solo node ──

echo "Starting mesh-llm (solo)..."
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
"$MESH_LLM" "${ARGS[@]}" >"$LOG" 2>&1 &
MESH_PID=$!

echo "Waiting for model to load (up to ${MAX_WAIT}s)..."
for i in $(seq 1 "$MAX_WAIT"); do
    assert_pid_alive "$MESH_PID" "mesh-llm"
    READY=$("$PYTHON_BIN" -c '
import json, sys, urllib.request
try:
    r = urllib.request.urlopen("http://127.0.0.1:'"$CONSOLE_PORT"'/api/status", timeout=2)
    print(json.load(r).get("llama_ready", False))
except Exception:
    print("False")
' 2>/dev/null || echo "False")
    if [ "$READY" = "True" ]; then
        echo "✅ Model loaded in ${i}s"
        break
    fi
    if [ "$i" -eq "$MAX_WAIT" ]; then
        fail_with_logs "model did not load within ${MAX_WAIT}s"
    fi
    if [ $((i % 20)) -eq 0 ]; then
        echo "  Still waiting... (${i}s)"
    fi
    sleep 1
done

# Wait for the public OpenAI surface to publish a concrete model id.
echo "Waiting for OpenAI API to publish a model..."
MODEL_NAME=""
for i in $(seq 1 60); do
    MODELS_JSON="$(curl -sf "http://127.0.0.1:${API_PORT}/v1/models" 2>/dev/null || true)"
    MODEL_NAME="$(echo "$MODELS_JSON" | "$PYTHON_BIN" -c '
import json, sys
try:
    data = json.load(sys.stdin).get("data", [])
except Exception:
    data = []
print(data[0].get("id", "") if data else "")
' 2>/dev/null || echo "")"
    if [ -n "$MODEL_NAME" ]; then
        echo "✅ OpenAI API ready in ${i}s"
        echo "Using model: $MODEL_NAME"
        break
    fi
    if [ "$i" -eq 60 ]; then
        fail_with_logs "OpenAI API did not publish a model"
    fi
    sleep 1
done

# Verify inference works before running SDK smokes
echo "Verifying inference readiness..."
for i in $(seq 1 30); do
    READY_PAYLOAD=$(cat <<JSON
{
    "model": "$MODEL_NAME",
    "messages": [{"role": "user", "content": "Reply with the single word ready."}],
    "max_tokens": 8,
    "temperature": 0
}
JSON
)
    RESPONSE=$(curl -s --max-time 10 -w '\n%{http_code}' \
        "http://127.0.0.1:${API_PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "$READY_PAYLOAD" 2>/dev/null || true)
    HTTP_CODE=$(printf '%s\n' "$RESPONSE" | tail -n 1)
    if [ "$HTTP_CODE" = "200" ]; then
        echo "✅ Inference ready"
        break
    fi
    if [ "$i" -eq 30 ]; then
        fail_with_logs "inference never became ready (last HTTP $HTTP_CODE)"
    fi
    sleep 1
done

# ── SDK smoke helper ──

ensure_openai_node_sdk() {
    if ! command -v "$NODE_BIN" >/dev/null 2>&1; then
        fail_with_logs "node is not installed"
    fi
    if ! command -v "$NPM_BIN" >/dev/null 2>&1; then
        fail_with_logs "npm is not installed"
    fi
    mkdir -p "$NODE_SDK_DIR"
    "$NPM_BIN" install --silent --prefix "$NODE_SDK_DIR" openai >/dev/null
}

# ── Run SDK smokes (all hit the solo node directly — no tunnel) ──

BASE_URL="http://127.0.0.1:${API_PORT}/v1"

echo "Running official openai-python smoke..."
"$PYTHON_BIN" "$REPO_ROOT/scripts/ci-openai-python-smoke.py" \
    --base-url "$BASE_URL"

echo "Running official openai-node smoke..."
ensure_openai_node_sdk
NODE_PATH="$NODE_SDK_DIR/node_modules" "$NODE_BIN" \
    "$REPO_ROOT/scripts/ci-openai-node-smoke.cjs" \
    --base-url "$BASE_URL"

echo "Running LiteLLM smoke..."
"$PYTHON_BIN" "$REPO_ROOT/scripts/ci-litellm-smoke.py" \
    --base-url "$BASE_URL" \
    --model "$MODEL_NAME"

echo "Running langchain-openai smoke..."
"$PYTHON_BIN" "$REPO_ROOT/scripts/ci-langchain-openai-smoke.py" \
    --base-url "$BASE_URL" \
    --model "$MODEL_NAME"

echo ""
echo "=== Compat smoke passed ==="
