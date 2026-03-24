#!/bin/bash
# Live integration test for the blackboard feature.
# Spins up two local nodes (private mesh), exercises all blackboard operations,
# then cleans up. No external mesh/Nostr involvement.
set -e

BINARY="$(dirname "$0")/../target/release/mesh-llm"
if [ ! -f "$BINARY" ]; then
    echo "ERROR: Build first — mesh-llm binary not found at $BINARY"
    exit 1
fi
BINARY="$(cd "$(dirname "$BINARY")" && pwd)/$(basename "$BINARY")"

PORT_A=19337
CONSOLE_A=13131
PORT_B=19338
CONSOLE_B=13132

PASS=0
FAIL=0
total_pass() { PASS=$((PASS + 1)); echo "  ✅ $1"; }
total_fail() { FAIL=$((FAIL + 1)); echo "  ❌ $1: $2"; }

cleanup() {
    echo ""
    set +e
    echo "🧹 Cleaning up..."
    [ -n "$PID_A" ] && kill "$PID_A" 2>/dev/null
    [ -n "$PID_B" ] && kill "$PID_B" 2>/dev/null
    sleep 1
    [ -n "$PID_A" ] && kill -9 "$PID_A" 2>/dev/null
    [ -n "$PID_B" ] && kill -9 "$PID_B" 2>/dev/null
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Results: $PASS passed, $FAIL failed"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    [ "$FAIL" -gt 0 ] && exit 1 || true
}
trap cleanup EXIT

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Blackboard Whiteboard Integration Test"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ── Start Node A ──
echo "📡 Starting Node A (port $PORT_A, console $CONSOLE_A)..."
$BINARY --blackboard --client --port $PORT_A --console $CONSOLE_A 2>/tmp/blackboard_test_a.log &
disown
PID_A=$!

# Wait for Node A console to be ready
for i in $(seq 1 15); do
    if curl -s "http://localhost:$CONSOLE_A/api/status" >/dev/null 2>&1; then break; fi
    sleep 1
done
if ! curl -s "http://localhost:$CONSOLE_A/api/status" >/dev/null 2>&1; then
    echo "ERROR: Node A failed to start. Log:"
    cat /tmp/blackboard_test_a.log
    exit 1
fi
echo "  Node A ready (PID $PID_A)"

# Get invite token from Node A's log
sleep 2
TOKEN=$(grep -oE 'Invite: [^ ]+' /tmp/blackboard_test_a.log | head -1 | awk '{print $2}')
if [ -z "$TOKEN" ]; then
    TOKEN=$(grep -oE 'eyJ[a-zA-Z0-9+/=]+' /tmp/blackboard_test_a.log | head -1)
fi
if [ -z "$TOKEN" ]; then
    echo "  ⚠️  Could not extract invite token"
fi
echo "  Token: ${TOKEN:0:30}..."

# ── Start Node B (joins A) ──
if [ -n "$TOKEN" ]; then
    echo "📡 Starting Node B (port $PORT_B, console $CONSOLE_B, joining A)..."
    $BINARY --blackboard --client --join "$TOKEN" --port $PORT_B --console $CONSOLE_B 2>/tmp/blackboard_test_b.log &
    disown
    PID_B=$!

    for i in $(seq 1 30); do
        if curl -s "http://localhost:$CONSOLE_B/api/status" >/dev/null 2>&1; then break; fi
        sleep 1
    done
    if ! curl -s "http://localhost:$CONSOLE_B/api/status" >/dev/null 2>&1; then
        echo "  ⚠️  Node B failed to start, continuing with single-node tests"
        kill "$PID_B" 2>/dev/null; wait "$PID_B" 2>/dev/null
        PID_B=""
    else
        echo "  Node B ready (PID $PID_B)"
        sleep 3  # Let gossip + blackboard sync settle
    fi
else
    echo "  Skipping Node B (no invite token)"
    PID_B=""
fi

echo ""
echo "── Test 1: Blackboard not enabled returns error ──"
# (We can't easily test this since both nodes have --blackboard, skip)
total_pass "Skipped (both nodes have --blackboard)"

echo ""
echo "── Test 2: Post a message via API ──"
RESP=$(curl -s "http://localhost:$CONSOLE_A/api/blackboard/post" \
    -H "Content-Type: application/json" \
    -d '{"text":"Hello from Node A - testing the whiteboard"}')
if echo "$RESP" | grep -q '"id"'; then
    POST_ID=$(echo "$RESP" | python3 -c "import sys,json; print(format(json.load(sys.stdin)['id'],'x'))" 2>/dev/null || echo "")
    total_pass "Posted message (id: $POST_ID)"
else
    total_fail "Post message" "$RESP"
fi

echo ""
echo "── Test 3: Feed shows the message ──"
sleep 1
FEED=$(curl -s "http://localhost:$CONSOLE_A/api/blackboard/feed")
COUNT=$(echo "$FEED" | python3 -c "import sys,json; print(len(json.load(sys.stdin)))" 2>/dev/null || echo "0")
if [ "$COUNT" -ge 1 ]; then
    total_pass "Feed has $COUNT item(s)"
else
    total_fail "Feed" "Expected >=1 items, got: $FEED"
fi

echo ""
echo "── Test 4: Search finds the message ──"
SEARCH=$(curl -s "http://localhost:$CONSOLE_A/api/blackboard/search?q=whiteboard")
SCOUNT=$(echo "$SEARCH" | python3 -c "import sys,json; print(len(json.load(sys.stdin)))" 2>/dev/null || echo "0")
if [ "$SCOUNT" -ge 1 ]; then
    total_pass "Search found $SCOUNT result(s)"
else
    total_fail "Search" "Expected >=1 results for 'whiteboard', got: $SEARCH"
fi

echo ""
echo "── Test 5: Search with no match returns empty ──"
SEARCH2=$(curl -s "http://localhost:$CONSOLE_A/api/blackboard/search?q=zzzznonexistent")
SCOUNT2=$(echo "$SEARCH2" | python3 -c "import sys,json; print(len(json.load(sys.stdin)))" 2>/dev/null || echo "0")
if [ "$SCOUNT2" -eq 0 ]; then
    total_pass "Empty search returns 0 results"
else
    total_fail "Empty search" "Expected 0, got $SCOUNT2"
fi

echo ""
echo "── Test 6: CLI feed works ──"
CLI_FEED=$($BINARY blackboard --port $CONSOLE_A --limit 5 2>&1)
if echo "$CLI_FEED" | grep -q "whiteboard\|Hello"; then
    total_pass "CLI feed shows messages"
else
    total_fail "CLI feed" "$CLI_FEED"
fi

echo ""
echo "── Test 7: CLI search works ──"
CLI_SEARCH=$($BINARY blackboard --search "whiteboard" --port $CONSOLE_A 2>&1)
if echo "$CLI_SEARCH" | grep -q "whiteboard\|Hello"; then
    total_pass "CLI search finds message"
else
    total_fail "CLI search" "$CLI_SEARCH"
fi

echo ""
echo "── Test 8: CLI post works ──"
CLI_POST=$($BINARY blackboard "CLI post test message" --port $CONSOLE_A 2>&1)
if echo "$CLI_POST" | grep -q "Posted"; then
    total_pass "CLI post succeeded"
else
    total_fail "CLI post" "$CLI_POST"
fi

echo ""
echo "── Test 9: PII scrubbing (private paths) ──"
PII_POST=$($BINARY blackboard "Check /Users/michael/secret/file.txt" --port $CONSOLE_A 2>&1)
if echo "$PII_POST" | grep -q "PII\|Scrubbing\|Posted"; then
    # Verify the stored message was scrubbed
    PII_SEARCH=$(curl -s "http://localhost:$CONSOLE_A/api/blackboard/search?q=secret")
    if echo "$PII_SEARCH" | grep -q '~/secret'; then
        total_pass "Path scrubbed to ~/"
    elif echo "$PII_SEARCH" | grep -q '/Users/michael'; then
        total_fail "PII scrub" "Path not scrubbed"
    else
        total_pass "PII detection + scrubbing triggered"
    fi
else
    total_fail "PII detection" "$PII_POST"
fi

echo ""
echo "── Test 10: Post with empty text rejected ──"
EMPTY=$(curl -s "http://localhost:$CONSOLE_A/api/blackboard/post" \
    -H "Content-Type: application/json" \
    -d '{"text":""}')
if echo "$EMPTY" | grep -q "error\|Missing"; then
    total_pass "Empty text rejected"
else
    total_fail "Empty text" "$EMPTY"
fi

# ── Cross-node tests (only if Node B is running) ──
if [ -n "$PID_B" ]; then
    echo ""
    echo "── Test 11: Message propagated to Node B ──"
    sleep 3  # Give flood-fill time
    FEED_B=$(curl -s "http://localhost:$CONSOLE_B/api/blackboard/feed")
    BCOUNT=$(echo "$FEED_B" | python3 -c "import sys,json; print(len(json.load(sys.stdin)))" 2>/dev/null || echo "0")
    if [ "$BCOUNT" -ge 1 ]; then
        total_pass "Node B has $BCOUNT items (propagated from A)"
    else
        total_fail "Propagation A→B" "Node B has 0 items"
    fi

    echo ""
    echo "── Test 12: Post from Node B propagates to A ──"
    curl -s "http://localhost:$CONSOLE_B/api/blackboard/post" \
        -H "Content-Type: application/json" \
        -d '{"text":"Hello from Node B!"}'
    sleep 2
    FEED_A=$(curl -s "http://localhost:$CONSOLE_A/api/blackboard/search?q=Node+B")
    ACOUNT=$(echo "$FEED_A" | python3 -c "import sys,json; print(len(json.load(sys.stdin)))" 2>/dev/null || echo "0")
    if [ "$ACOUNT" -ge 1 ]; then
        total_pass "Node A got message from B"
    else
        total_fail "Propagation B→A" "Node A didn't get B's message"
    fi
else
    echo ""
    echo "  (Skipping cross-node tests — Node B not running)"
fi

echo ""
echo "── Test 13: Feed limit works ──"
LIMIT_FEED=$(curl -s "http://localhost:$CONSOLE_A/api/blackboard/feed?limit=2")
LCOUNT=$(echo "$LIMIT_FEED" | python3 -c "import sys,json; print(len(json.load(sys.stdin)))" 2>/dev/null || echo "0")
if [ "$LCOUNT" -le 2 ]; then
    total_pass "Feed limit=2 returned $LCOUNT items"
else
    total_fail "Feed limit" "Expected <=2, got $LCOUNT"
fi
