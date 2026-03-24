#!/bin/bash
# Reasoning compatibility test — validates the API contract around thinking.
#
# Requires a local llama-server with a thinking model (Qwen3, MiniMax, etc.).
# The test spins up its own llama-server with --reasoning-budget 0 and validates:
#
#   1. Default requests produce NO reasoning (fast, no thinking)
#   2. API users can opt-in to thinking via chat_template_kwargs
#   3. Streaming works correctly in both modes
#   4. Standard OpenAI-compat requests are unaffected
#
# Usage:
#   ./tests/test_reasoning_compat.sh [model_path]
#
# If no model path given, tries ~/.models/Qwen3-8B-Q4_K_M.gguf
set -e

MODEL="${1:-$HOME/.models/Qwen3-8B-Q4_K_M.gguf}"
PORT=18099
PASS=0
FAIL=0

total_pass() { PASS=$((PASS + 1)); echo "  ✅ $1"; }
total_fail() { FAIL=$((FAIL + 1)); echo "  ❌ $1: $2"; }

if [ ! -f "$MODEL" ]; then
    echo "ERROR: Model not found: $MODEL"
    echo "Usage: $0 [model_path]"
    echo "Needs a thinking-capable model (Qwen3, MiniMax, etc.)"
    exit 1
fi

# Find llama-server
LLAMA_SERVER=""
for p in \
    "$(dirname "$0")/../bin/llama-server" \
    /usr/local/bin/llama-server \
    /opt/homebrew/bin/llama-server \
    llama-server; do
    if command -v "$p" >/dev/null 2>&1 || [ -x "$p" ]; then
        LLAMA_SERVER="$p"
        break
    fi
done
if [ -z "$LLAMA_SERVER" ]; then
    echo "ERROR: llama-server not found"
    exit 1
fi

cleanup() {
    set +e
    [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null
    sleep 0.5
    [ -n "$SERVER_PID" ] && kill -9 "$SERVER_PID" 2>/dev/null
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Results: $PASS passed, $FAIL failed"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    [ "$FAIL" -gt 0 ] && exit 1 || true
}
trap cleanup EXIT

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Reasoning Compatibility Test"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Model: $MODEL"
echo "Server: $LLAMA_SERVER"
echo "Port: $PORT"
echo ""

# Kill any existing server on our port
pkill -f "llama-server.*--port $PORT" 2>/dev/null || true
sleep 1

# Start llama-server with reasoning disabled (our production config)
echo "🚀 Starting llama-server with --reasoning-budget 0 ..."
$LLAMA_SERVER \
    -m "$MODEL" \
    --port $PORT \
    -ngl 99 \
    --reasoning-format deepseek \
    --reasoning-budget 0 \
    -c 2048 \
    > /tmp/reasoning_test.log 2>&1 &
SERVER_PID=$!

# Wait for ready
for i in $(seq 1 60); do
    if curl -s "http://localhost:$PORT/health" 2>/dev/null | grep -q "ok"; then
        echo "  Ready after ${i}s"
        break
    fi
    sleep 1
done
if ! curl -s "http://localhost:$PORT/health" 2>/dev/null | grep -q "ok"; then
    echo "ERROR: llama-server failed to start. Log:"
    tail -20 /tmp/reasoning_test.log
    exit 1
fi

# Confirm thinking is disabled at startup
if grep -q "thinking = 0" /tmp/reasoning_test.log; then
    echo "  Confirmed: thinking = 0 at startup"
else
    echo "  WARNING: Could not confirm thinking=0 in log"
fi
echo ""

# ── Helper ──
# Send a non-streaming request, extract reasoning and content
api_request() {
    local desc="$1"
    local body="$2"
    curl -s --max-time 60 "http://localhost:$PORT/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "$body" 2>/dev/null
}

# ── Test 1: Default request — no thinking ──
echo "── Test 1: Default request produces no reasoning ──"
RESP=$(api_request "default" '{"model":"test","messages":[{"role":"user","content":"What is 2+2? One word."}],"max_tokens":128}')
R_LEN=$(echo "$RESP" | python3 -c "import sys,json; c=json.load(sys.stdin)['choices'][0]['message']; print(len(c.get('reasoning_content','') or ''))" 2>/dev/null || echo "-1")
CONTENT=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message'].get('content','').strip()[:100])" 2>/dev/null || echo "")
if [ "$R_LEN" = "0" ] && [ -n "$CONTENT" ]; then
    total_pass "No reasoning ($R_LEN chars), content: \"$CONTENT\""
else
    total_fail "Default no-think" "reasoning=$R_LEN, content=\"$CONTENT\""
fi

# ── Test 2: Explicit enable_thinking=false — no thinking ──
echo ""
echo "── Test 2: Explicit enable_thinking=false — no thinking ──"
RESP=$(api_request "explicit-off" '{"model":"test","messages":[{"role":"user","content":"What is 2+2? One word."}],"max_tokens":128,"chat_template_kwargs":{"enable_thinking":false}}')
R_LEN=$(echo "$RESP" | python3 -c "import sys,json; c=json.load(sys.stdin)['choices'][0]['message']; print(len(c.get('reasoning_content','') or ''))" 2>/dev/null || echo "-1")
CONTENT=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message'].get('content','').strip()[:100])" 2>/dev/null || echo "")
if [ "$R_LEN" = "0" ] && [ -n "$CONTENT" ]; then
    total_pass "No reasoning, content: \"$CONTENT\""
else
    total_fail "Explicit off" "reasoning=$R_LEN, content=\"$CONTENT\""
fi

# ── Test 3: Opt-in to thinking via chat_template_kwargs ──
echo ""
echo "── Test 3: Opt-in enable_thinking=true produces reasoning ──"
RESP=$(api_request "opt-in" '{"model":"test","messages":[{"role":"user","content":"What is 2+2? One word."}],"max_tokens":1024,"chat_template_kwargs":{"enable_thinking":true}}')
R_LEN=$(echo "$RESP" | python3 -c "import sys,json; c=json.load(sys.stdin)['choices'][0]['message']; print(len(c.get('reasoning_content','') or ''))" 2>/dev/null || echo "-1")
CONTENT=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message'].get('content','').strip()[:100])" 2>/dev/null || echo "")
if [ "$R_LEN" -gt 0 ]; then
    total_pass "Reasoning produced ($R_LEN chars), content: \"$CONTENT\""
else
    total_fail "Opt-in thinking" "reasoning=$R_LEN (expected >0)"
fi

# ── Test 4: Streaming default — no thinking ──
echo ""
echo "── Test 4: Streaming default — no reasoning_content deltas ──"
STREAM_RESULT=$(curl -s --max-time 30 -N "http://localhost:$PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"test","messages":[{"role":"user","content":"What is 2+2? One word."}],"max_tokens":128,"stream":true}' \
    2>/dev/null | python3 -c "
import sys, json
r = ''; c = ''
for line in sys.stdin:
    line = line.strip()
    if not line.startswith('data: '): continue
    data = line[6:].strip()
    if not data or data == '[DONE]': continue
    try:
        chunk = json.loads(data)
        delta = chunk.get('choices', [{}])[0].get('delta', {})
        if delta.get('reasoning_content'): r += delta['reasoning_content']
        if delta.get('content'): c += delta['content']
    except: pass
print(f'{len(r)}|{c.strip()[:100]}')
" 2>/dev/null)
STREAM_R=$(echo "$STREAM_RESULT" | cut -d'|' -f1)
STREAM_C=$(echo "$STREAM_RESULT" | cut -d'|' -f2)
if [ "$STREAM_R" = "0" ] && [ -n "$STREAM_C" ]; then
    total_pass "Streaming: no reasoning, content: \"$STREAM_C\""
else
    total_fail "Streaming default" "reasoning=$STREAM_R, content=\"$STREAM_C\""
fi

# ── Test 5: Streaming with opt-in thinking ──
echo ""
echo "── Test 5: Streaming opt-in — reasoning_content deltas present ──"
STREAM_RESULT=$(curl -s --max-time 60 -N "http://localhost:$PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"test","messages":[{"role":"user","content":"What is 2+2? One word."}],"max_tokens":1024,"stream":true,"chat_template_kwargs":{"enable_thinking":true}}' \
    2>/dev/null | python3 -c "
import sys, json
r = ''; c = ''
for line in sys.stdin:
    line = line.strip()
    if not line.startswith('data: '): continue
    data = line[6:].strip()
    if not data or data == '[DONE]': continue
    try:
        chunk = json.loads(data)
        delta = chunk.get('choices', [{}])[0].get('delta', {})
        if delta.get('reasoning_content'): r += delta['reasoning_content']
        if delta.get('content'): c += delta['content']
    except: pass
print(f'{len(r)}|{c.strip()[:100]}')
" 2>/dev/null)
STREAM_R=$(echo "$STREAM_RESULT" | cut -d'|' -f1)
STREAM_C=$(echo "$STREAM_RESULT" | cut -d'|' -f2)
if [ "$STREAM_R" -gt 0 ]; then
    total_pass "Streaming opt-in: reasoning=$STREAM_R chars, content: \"$STREAM_C\""
else
    total_fail "Streaming opt-in" "reasoning=$STREAM_R (expected >0)"
fi

# ── Test 6: Standard OpenAI-compat request ──
echo ""
echo "── Test 6: Standard OpenAI-compatible request works ──"
RESP=$(api_request "openai-compat" '{"model":"test","messages":[{"role":"user","content":"Write a haiku about computers"}],"max_tokens":128,"temperature":0.7}')
FINISH=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0].get('finish_reason',''))" 2>/dev/null || echo "")
USAGE=$(echo "$RESP" | python3 -c "import sys,json; u=json.load(sys.stdin).get('usage',{}); print(f\"prompt={u.get('prompt_tokens','?')} completion={u.get('completion_tokens','?')}\")" 2>/dev/null || echo "")
CONTENT=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message'].get('content','').strip()[:100])" 2>/dev/null || echo "")
if [ "$FINISH" = "stop" ] && [ -n "$CONTENT" ]; then
    total_pass "OpenAI-compat: finish=$FINISH, $USAGE, content: \"$CONTENT\""
else
    total_fail "OpenAI-compat" "finish=$FINISH, content=\"$CONTENT\""
fi

# ── Test 7: /v1/models endpoint ──
echo ""
echo "── Test 7: /v1/models returns model list ──"
MODELS=$(curl -s --max-time 5 "http://localhost:$PORT/v1/models" 2>/dev/null)
MODEL_COUNT=$(echo "$MODELS" | python3 -c "import sys,json; print(len(json.load(sys.stdin).get('data',[])))" 2>/dev/null || echo "0")
if [ "$MODEL_COUNT" -ge 1 ]; then
    total_pass "/v1/models returns $MODEL_COUNT model(s)"
else
    total_fail "/v1/models" "Expected >=1, got $MODEL_COUNT"
fi
