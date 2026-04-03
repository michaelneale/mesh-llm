#!/usr/bin/env bash
# ci-gguf-smoke-test.sh — start mesh-llm with a GGUF model,
# run one inference request, then shut down.
#
# Usage:
#   scripts/ci-gguf-smoke-test.sh <mesh-llm-binary> <bin-dir> <gguf-model-path> [prompt] [expect-contains] [forbid-contains] [expect-exact] [prompt-suite-json]

set -euo pipefail

MESH_LLM="$1"
BIN_DIR="$2"
MODEL="$3"
PROMPT_TEXT="${4:-Reply with exactly: blue}"
EXPECT_CONTAINS="${5:-}"
FORBID_CONTAINS="${6:-}"
EXPECT_EXACT="${7:-}"
PROMPT_SUITE_JSON="${8:-}"
MAX_WAIT=300
LOG=/tmp/mesh-llm-ci-gguf.log

pick_free_port() {
    python3 - <<'PY'
import socket
s = socket.socket()
s.bind(("127.0.0.1", 0))
print(s.getsockname()[1])
s.close()
PY
}

API_PORT="$(pick_free_port)"
CONSOLE_PORT="$(pick_free_port)"
while [ "$API_PORT" = "$CONSOLE_PORT" ]; do
    CONSOLE_PORT="$(pick_free_port)"
done

echo "=== CI GGUF Smoke Test ==="
echo "  mesh-llm:  $MESH_LLM"
echo "  bin-dir:   $BIN_DIR"
echo "  model:     $MODEL"
echo "  api port:  $API_PORT"
echo "  os:        $(uname -s)"
echo "  prompt:    $PROMPT_TEXT"

ls -la "$BIN_DIR"/rpc-server* "$BIN_DIR"/llama-server* 2>/dev/null || true
if [ ! -f "$MESH_LLM" ]; then
    echo "❌ Missing mesh-llm binary: $MESH_LLM"
    exit 1
fi

echo "Starting mesh-llm..."
"$MESH_LLM" \
    --model "$MODEL" \
    --no-draft \
    --bin-dir "$BIN_DIR" \
    --port "$API_PORT" \
    --console "$CONSOLE_PORT" \
    > "$LOG" 2>&1 &
MESH_PID=$!
echo "  PID: $MESH_PID"

cleanup() {
    echo "Shutting down mesh-llm (PID $MESH_PID)..."
    if kill -0 "$MESH_PID" 2>/dev/null; then
        # Collect child/grandchild PIDs before killing the parent so we can
        # clean up rpc-server and llama-server without touching unrelated processes.
        CHILD_PIDS=()
        while IFS= read -r cpid; do
            CHILD_PIDS+=("$cpid")
        done < <(pgrep -P "$MESH_PID" 2>/dev/null || true)

        kill "$MESH_PID" 2>/dev/null || true
        sleep 2
        kill -9 "$MESH_PID" 2>/dev/null || true
        for cpid in "${CHILD_PIDS[@]}"; do
            kill -9 "$cpid" 2>/dev/null || true
        done
        wait "$MESH_PID" 2>/dev/null || true
    fi
    echo "Cleanup done."
}
trap cleanup EXIT

echo "Waiting for model to load (up to ${MAX_WAIT}s)..."
for i in $(seq 1 "$MAX_WAIT"); do
    if ! kill -0 "$MESH_PID" 2>/dev/null; then
        echo "❌ mesh-llm exited unexpectedly"
        echo "--- Log tail ---"
        tail -80 "$LOG" || true
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

run_chat_case() {
    local prompt_text="$1"
    local expect_contains="$2"
    local forbid_contains="$3"
    local expect_exact="$4"
    local case_label="$5"

    echo "Testing /v1/chat/completions ($case_label)..."
    local curl_body
    curl_body=$(python3 - "$prompt_text" <<'PY'
import json, sys
prompt = sys.argv[1]
print(json.dumps({
    "model": "any",
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": 32,
    "temperature": 0,
    "enable_thinking": False
}))
PY
)
    local response
    if ! response=$(curl -sf "http://localhost:${API_PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "$curl_body" 2>&1); then
        echo "❌ Inference request failed"
        echo "$response"
        echo "--- Log tail ---"
        tail -80 "$LOG" || true
        exit 1
    fi

    local content
    content=$(echo "$response" | python3 -c "import sys,json; r=json.load(sys.stdin); print(r['choices'][0]['message']['content'])" 2>/dev/null || echo "")
    if [ -z "$content" ]; then
        echo "❌ Empty response from inference"
        echo "Raw response: $response"
        exit 1
    fi

    if echo "$content" | grep -F "<think>" >/dev/null 2>&1; then
        echo "❌ Unexpected reasoning output with enable_thinking=false"
        echo "Content: $content"
        exit 1
    fi

    if [ -n "$expect_contains" ] && ! echo "$content" | grep -F "$expect_contains" >/dev/null 2>&1; then
        echo "❌ Response did not contain expected text: $expect_contains"
        echo "Content: $content"
        exit 1
    fi

    if [ -n "$expect_exact" ]; then
        local normalized_content normalized_expected
        normalized_content=$(printf '%s' "$content" | python3 -c "import sys; print(sys.stdin.read().strip())")
        normalized_expected=$(printf '%s' "$expect_exact" | python3 -c "import sys; print(sys.stdin.read().strip())")
        if [ "$normalized_content" != "$normalized_expected" ]; then
            echo "❌ Response did not exactly match expected text"
            echo "Expected: $normalized_expected"
            echo "Content:  $normalized_content"
            exit 1
        fi
    fi

    if [ -n "$forbid_contains" ] && echo "$content" | grep -F "$forbid_contains" >/dev/null 2>&1; then
        echo "❌ Response contained forbidden text: $forbid_contains"
        echo "Content: $content"
        exit 1
    fi

    local finish_reason
    finish_reason=$(echo "$response" | python3 -c "import sys,json; r=json.load(sys.stdin); print(r['choices'][0].get('finish_reason',''))" 2>/dev/null || echo "")
    if [ -z "$finish_reason" ]; then
        echo "❌ Missing finish_reason in response"
        echo "Raw response: $response"
        exit 1
    fi

    echo "✅ Inference response: $content"
}

run_chat_case "$PROMPT_TEXT" "$EXPECT_CONTAINS" "$FORBID_CONTAINS" "$EXPECT_EXACT" "primary"

if [ -n "$PROMPT_SUITE_JSON" ]; then
    echo "Running extra prompt suite..."
    python3 - "$PROMPT_SUITE_JSON" <<'PY' | while IFS=$'\t' read -r label prompt expect_contains forbid_contains expect_exact; do
import json, sys
suite = json.loads(sys.argv[1])
for index, case in enumerate(suite, start=1):
    print("\t".join([
        str(case.get("label", f"case-{index}")),
        str(case.get("prompt", "")),
        str(case.get("expect_contains", "")),
        str(case.get("forbid_contains", "")),
        str(case.get("expect_exact", "")),
    ]))
PY
        run_chat_case "$prompt" "$expect_contains" "$forbid_contains" "$expect_exact" "$label"
    done
fi

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
echo "=== GGUF smoke test passed ==="
