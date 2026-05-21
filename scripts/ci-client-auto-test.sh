#!/usr/bin/env bash
# ci-client-auto-test.sh — verify `mesh-llm client --auto` boots its API
# AND actually joins the public mesh.
#
# Two invariants:
#   1. Even when Nostr discovery returns dead/stale peers, the management API
#      on :3132 must come up. A broken implementation thrashes retrying dead
#      peers and never binds the console port.
#   2. With public Nostr discovery available, the node must actually join a
#      mesh — i.e. /api/status reports mesh_id set AND (peers non-empty OR
#      first_joined_mesh_ts set). This catches regressions where discovery,
#      gossip, or join-handshake silently break and the node sits idle.
#
# Usage: scripts/ci-client-auto-test.sh <mesh-llm-binary>
#
# Exits 0 only if both invariants hold; 1 otherwise.

set -euo pipefail

MESH_LLM="${1:?Usage: $0 <mesh-llm-binary>}"
CONSOLE_PORT=3132        # avoid clashing with other CI steps
API_PORT=9338
MAX_WAIT=120             # seconds — generous for Nostr discovery + join attempt
LOG=/tmp/mesh-llm-client-auto.log

echo "=== CI Client-Auto Test ==="
echo "  mesh-llm:     $MESH_LLM"
echo "  console port: $CONSOLE_PORT"
echo "  api port:     $API_PORT"
echo "  max wait:     ${MAX_WAIT}s"
echo "  os:           $(uname -s)"

if [ ! -f "$MESH_LLM" ]; then
    echo "❌ Missing mesh-llm binary: $MESH_LLM"
    exit 1
fi

# Start mesh-llm client --auto in background
echo "Starting mesh-llm client --auto..."
"$MESH_LLM" \
    --log-format json \
    client \
    --auto \
    --port "$API_PORT" \
    --console "$CONSOLE_PORT" \
    > "$LOG" 2>&1 &
MESH_PID=$!
echo "  PID: $MESH_PID"

cleanup() {
    echo "Shutting down mesh-llm (PID $MESH_PID)..."
    kill "$MESH_PID" 2>/dev/null || true
    pkill -P "$MESH_PID" 2>/dev/null || true
    sleep 1
    kill -9 "$MESH_PID" 2>/dev/null || true
    wait "$MESH_PID" 2>/dev/null || true
    echo "--- Log (last 80 lines) ---"
    tail -80 "$LOG" 2>/dev/null || true
    echo "--- End log ---"
    echo "Cleanup done."
}
trap cleanup EXIT

# Wait for the console API to become reachable.
# This is the core assertion: the management API MUST come up even when
# the node is struggling to connect to dead peers.
echo "Waiting for console API on port ${CONSOLE_PORT} (up to ${MAX_WAIT}s)..."
API_UP=false
for i in $(seq 1 "$MAX_WAIT"); do
    # Check process is still alive
    if ! kill -0 "$MESH_PID" 2>/dev/null; then
        echo "⚠️  mesh-llm exited (may be expected if no meshes found)"
        echo "--- Log tail ---"
        tail -40 "$LOG" 2>/dev/null || true
        # If it exited because "No meshes found after 5 minutes" that's a
        # different (expected) failure. The test only fails if the API never
        # came up while the process was alive.
        if grep -q "No meshes found after" "$LOG" 2>/dev/null; then
            echo "❌ Process exited with 'no meshes found'."
            echo "   Either the public mesh is currently down, or discovery is broken."
            exit 1
        fi
        echo "❌ mesh-llm exited unexpectedly"
        exit 1
    fi

    # Try to hit the console API
    if curl -sf --max-time 2 "http://localhost:${CONSOLE_PORT}/api/status" > /dev/null 2>&1; then
        echo "✅ Console API is up after ${i}s"
        API_UP=true
        break
    fi

    if [ $((i % 10)) -eq 0 ]; then
        echo "  Still waiting... (${i}s)"
        # Show last few log lines for debugging
        tail -3 "$LOG" 2>/dev/null | sed 's/^/    /' || true
    fi
    sleep 1
done

if [ "$API_UP" = true ]; then
    # Bonus: verify the status endpoint returns valid JSON
    echo "Verifying /api/status response..."
    STATUS=$(curl -sf --max-time 5 "http://localhost:${CONSOLE_PORT}/api/status" 2>&1 || echo "")
    if [ -n "$STATUS" ]; then
        # Check it's valid JSON with expected fields
        if echo "$STATUS" | python3 -c "import sys,json; d=json.load(sys.stdin); assert 'version' in d or 'peers' in d or 'models' in d" 2>/dev/null; then
            echo "✅ /api/status returns valid JSON"
        else
            echo "⚠️  /api/status returned something but couldn't validate: $STATUS"
        fi
    fi

    # Verify /v1/models is reachable (through the proxy port)
    echo "Checking /v1/models on port ${API_PORT}..."
    if curl -sf --max-time 5 "http://localhost:${API_PORT}/v1/models" > /dev/null 2>&1; then
        echo "✅ /v1/models is reachable"
    else
        echo "⚠️  /v1/models not reachable (may be expected with no live peers)"
    fi

    # Required "actually joined a mesh" signal.
    #
    # Predicate: mesh_id set AND peers non-empty.
    #
    # We deliberately do NOT accept `first_joined_mesh_ts` as evidence of a
    # successful public-mesh join. When `client --auto` finds no candidates
    # via Nostr discovery, it falls through to `run_auto_start_new_mesh`,
    # which generates a fresh local mesh_id AND sets first_joined_mesh_ts
    # to "now" with zero peers. That standalone-fallback path would satisfy
    # a `mesh_id OR first_joined_mesh_ts` predicate and silently pass CI
    # while the node is talking to nobody — exactly the regression this
    # test is meant to catch.
    #
    # The only honest signal that public discovery + join actually worked
    # end-to-end is at least one real peer. We poll for up to JOIN_WAIT
    # seconds; if no peer ever appears, this step fails.
    JOIN_WAIT=60
    echo "Polling /api/status for at least one peer (mesh_id set AND peers non-empty, up to ${JOIN_WAIT}s)..."
    JOINED=false
    for j in $(seq 1 "$JOIN_WAIT"); do
        STATUS=$(curl -sf --max-time 5 "http://localhost:${CONSOLE_PORT}/api/status" 2>/dev/null || echo "")
        if [ -n "$STATUS" ] && echo "$STATUS" | python3 -c '
import json, sys
d = json.load(sys.stdin)
mesh_id = d.get("mesh_id")
peers = d.get("peers") or []
if mesh_id and peers:
    sys.exit(0)
sys.exit(1)
' 2>/dev/null; then
            MESH_ID=$(echo "$STATUS" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("mesh_id",""))')
            PEER_COUNT=$(echo "$STATUS" | python3 -c 'import json,sys; print(len(json.load(sys.stdin).get("peers") or []))')
            echo "✅ Joined mesh after ${j}s — mesh_id=${MESH_ID} peers=${PEER_COUNT}"
            JOINED=true
            break
        fi
        if [ $((j % 10)) -eq 0 ]; then
            echo "  Still waiting for at least one peer... (${j}s)"
        fi
        sleep 1
    done

    if [ "$JOINED" = false ]; then
        echo ""
        echo "❌ Console API came up but the node never saw any peer within ${JOIN_WAIT}s."
        echo "   Required signal: /api/status with mesh_id set AND peers non-empty."
        echo "   (first_joined_mesh_ts alone is NOT accepted — it is also set by the"
        echo "    standalone-fallback path when no public mesh is discovered.)"
        if grep -q "No meshes found yet" "$LOG" 2>/dev/null; then
            echo "   Log indicates public Nostr discovery returned no meshes."
            echo "   Either the public mesh is currently down, or discovery is broken."
        fi
        echo "   Last /api/status body:"
        echo "$STATUS" | sed 's/^/     /'
        exit 1
    fi

    echo ""
    echo "=== Client-auto test passed ==="
    exit 0
fi

echo ""
# Distinguish between "stuck with dead peers" vs "no meshes at all"
if grep -q "No meshes found yet" "$LOG" 2>/dev/null; then
    echo "❌ Console API never became reachable within ${MAX_WAIT}s"
    echo "   Node is stuck in Nostr discovery retry loop (no meshes found)."
    echo "   The API should be bound early, even during discovery retries."
elif grep -qE "Joining:|Joined mesh" "$LOG" 2>/dev/null; then
    echo "❌ Console API never became reachable within ${MAX_WAIT}s"
    echo "   Node joined a mesh but the API never came up."
    echo "   This likely means the node is stuck connecting to dead peers."
else
    echo "❌ Console API never became reachable within ${MAX_WAIT}s"
    echo "   Unknown state — check logs above."
fi
exit 1
