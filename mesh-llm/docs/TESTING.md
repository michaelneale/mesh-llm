# Testing mesh-llm

## Local inspection

### 0. Inspect local GPUs

```bash
mesh-llm gpus
mesh-llm gpus --json | jq .
mesh-llm gpu benchmark --json | jq .
```

- Prints local GPU entries with stable IDs, backend devices, VRAM, unified-memory status, and cached bandwidth when a fingerprint is available
- `--json` emits machine-readable inventory and benchmark payloads suitable for automation

### 0a. Startup config smoke

Create `~/.mesh-llm/config.toml`:

```toml
version = 1

[gpu]
assignment = "auto"

[[models]]
model = "Qwen2.5-3B"

[[models]]
model = "/absolute/path/to/qwen2.5-vl.gguf"
mmproj = "/absolute/path/to/mmproj.gguf"
ctx_size = 8192
```

Then start:

```bash
mesh-llm serve
```

- Both configured startup models should be considered for launch
- If `[[models]]` is empty, `mesh-llm serve` should print a `⚠️` warning, show help, and exit cleanly
- Explicit `--model` or `--gguf` should ignore configured `[[models]]`
- Explicit `--ctx-size` should override configured `ctx_size`

### 0b. Pinned startup smoke

First inspect the valid local IDs:

```bash
mesh-llm gpus
mesh-llm gpus --json | jq .
```

Then create `~/.mesh-llm/config.toml` with a real pinnable stable ID from that output (for example `pci:*`, `uuid:*`, or `metal:*`, not fallback IDs like `index:*` or backend-device names):

```toml
version = 1

[gpu]
assignment = "pinned"

[[models]]
model = "Qwen2.5-3B"
gpu_id = "pci:0000:65:00.0"
```

Start the node:

```bash
mesh-llm serve
```

- Startup should succeed only when `gpu_id` matches a valid local pinnable stable ID from `mesh-llm gpus`
- If the pinned ID is missing, ambiguous, unsupported, or stale, startup should fail closed before local launch
- Explicit `mesh-llm serve --model ...` should still bypass configured `[[models]]` and therefore bypass config-owned pinned IDs
- Do not use GPU indexes, `index:*`, or backend-device names like `CUDA0` / `HIP0` / `MTL0` as `gpu_id`

## Single-model permutations

### 1. Solo (single node)

```bash
mesh-llm serve --model Qwen2.5-3B --console
```

- API on `:9337`, console on `:3131`
- Console: `host=true, peers=0`
- llama-server has 1 RPC entry (self)

### 1a. Headless mode (API-only, no embedded UI)

```bash
mesh-llm serve --model Qwen2.5-3B --headless --console 3131
```

- API on `:9337`, management API on `:3131`
- `GET /api/status` returns 200 with normal JSON
- `GET /` returns 404 (web console routes are disabled)
- `GET /dashboard`, `GET /chat`, and `/assets/*` also return 404
- The management API (`/api/*`) remains fully accessible

This mode is intended for headless server deployments where the embedded web UI is not needed.

### 2. Two GPU nodes, model fits on host

```bash
# node A (more VRAM, becomes host)
mesh-llm serve --model Qwen2.5-32B --bind-port 7842
# node B (joins)
mesh-llm serve --model Qwen2.5-32B --join <TOKEN>
```

- Both nodes run solo (no split) — each is its own host
- API works from both nodes on `:9337`

### 3. Two GPU nodes, forced split

```bash
# host with --split
mesh-llm serve --model Qwen2.5-32B --bind-port 7842 --split
# worker joins
mesh-llm serve --model Qwen2.5-32B --join <TOKEN>
```

- `--split` forces tensor split even when model fits on host
- llama-server has 2 RPC entries
- Tensor split proportional to VRAM (e.g. `0.67,0.33`)
- Draft model auto-detected and used

### 4. Two GPU nodes, model too big for one

When the model exceeds host VRAM, split happens automatically without `--split`.

### 5. Lite client (no GPU)

```bash
mesh-llm client --join <TOKEN> --port 9555
```

- Uses ephemeral key (unique identity, works on same machine as GPU node)
- `/v1/models` lists all served models from gossip
- API tunneled to correct host per model via QUIC
- VRAM total excludes client

## Multi-model permutations

### 6. Two nodes, different models

```bash
# node A: seeds mesh with two models, serves 3B
mesh-llm serve --model Qwen2.5-3B --model GLM-4.7-Flash --console
# node B: joins, auto-assigned to GLM (needed, on disk)
mesh-llm serve --join <TOKEN>
```

- `/v1/models` on either node lists both models
- Requesting GLM from node A routes via QUIC to node B
- Requesting 3B from node B routes via QUIC to node A
- Both run solo (no tensor split)
- Console shows both models warm with node counts

Compatibility result:
- Verified on 2026-04-02 with the current `codex/model-identity-design` branch on node 1 and the latest GitHub release `v0.54.0` on node 2.
- Node 1 served `Llama-3.2-1B-Instruct-Q4_K_M`; node 2 served `Qwen3-4B-Q4_K_M`.
- `/api/models` and `/v1/models` agreed on the same warm model list from both nodes.
- Chat from node 1 to node 2's model succeeded, and chat from node 2 to node 1's model succeeded.

### 7. Auto-assignment

```bash
# seeder declares two models
mesh-llm serve --model Qwen2.5-3B --model GLM-4.7-Flash
# joiner with no --model
mesh-llm serve --join <TOKEN>
```

- Joiner scans the Hugging Face cache and picks an unserved model already on disk
- Log: "Selected to serve GLM-4.7-Flash (needed by mesh, already on disk)"

### 8. Lite client with multi-model

```bash
# GPU nodes running as above
mesh-llm client --join <TOKEN> --port 9555
```

- Client sees all models via gossip (ephemeral key = unique identity)
- `/v1/models` lists all served models
- Routes to correct host per model
- Streaming works through cross-model QUIC tunnel

### 9. Unload a model

```bash
mesh-llm unload GLM-4.7-Flash-Q4_K_M
```

- Node serving that model exits cleanly
- Other nodes unaffected
- Model goes cold in console

### 9a. Local runtime load/unload and local status view

```bash
# Running node
mesh-llm serve --model Qwen2.5-0.5B-Instruct-Q4_K_M --console

# Operator surface
mesh-llm load Llama-3.2-1B-Instruct-Q4_K_M
mesh-llm status
mesh-llm unload Llama-3.2-1B-Instruct-Q4_K_M

# REST surface
curl localhost:3131/api/runtime
curl localhost:3131/api/runtime/processes
curl -X POST localhost:3131/api/runtime/models \
  -H 'Content-Type: application/json' \
  -d '{"model":"Llama-3.2-1B-Instruct-Q4_K_M"}'
curl -X DELETE localhost:3131/api/runtime/models/Llama-3.2-1B-Instruct-Q4_K_M
```

- `mesh-llm status` shows the local models currently backed by running inference processes, including PID when present
- `GET /api/runtime` and `GET /api/runtime/processes` agree with the CLI output
- Loading a small local model adds it to `/v1/models` without restarting the node
- Unloading any local model removes it cleanly without terminating the mesh-llm process

### 10. Console model picker

- Dropdown appears when >1 warm model
- Switching models highlights the serving node in topology view
- Chat routes to selected model via API proxy

### 11. Console live-state and wakeable capacity

```bash
cd mesh-llm/ui/
npm run test:run
npm run typecheck
just build
```

- Live badges show only `Client`, `Standby`, `Loading`, and `Serving`
- Wakeable capacity renders in a separate section from topology peers and live nodes
- Wakeable entries do not appear in the topology peer list
- Validation uses `npm run test:run`, `npm run typecheck`, and `just build`

## Mesh Identity

### 16. Mesh ID generation (originator)

```bash
# With --mesh-name (deterministic ID)
mesh-llm serve --model Qwen2.5-3B --mesh-name "test-mesh"
```

- Log: `📌 Mesh ID: <hex>`
- `~/.mesh-llm/last-mesh` contains the same hex
- Restart with same `--mesh-name` → same mesh ID (deterministic)
- Different `--mesh-name` → different mesh ID

### 17. Mesh ID propagation (joiner)

```bash
# Originator
mesh-llm serve --model Qwen2.5-3B --mesh-name "test-mesh"
# Joiner
mesh-llm serve --model Qwen2.5-3B --join <TOKEN>
```

- Joiner log: `📌 Mesh ID: <same hex as originator>`
- Joiner's `~/.mesh-llm/last-mesh` matches originator's mesh ID
- Mesh ID arrives via gossip (worker nodes) or routing table (passive clients)

### 18. Sticky mesh preference

- Join a mesh → `~/.mesh-llm/last-mesh` saved
- On next `--auto`, `score_mesh()` adds +500 for meshes with matching `mesh_id`
- If that mesh is dead (not on Nostr), scoring proceeds normally without bonus

## Bootstrap Proxy

### 19. Instant API during GPU bootstrap

```bash
# Originator (already running)
mesh-llm serve --model Qwen2.5-3B --port 8090
# Joiner
mesh-llm serve --model Qwen2.5-3B --join <TOKEN> --port 8091
```

- Joiner log: `⚡ API ready (bootstrap): http://localhost:8091`
- BEFORE `rpc-server` or `llama-server` starts on joiner:
  - `curl localhost:8091/v1/models` → lists mesh models
  - `curl localhost:8091/v1/chat/completions` → inference via tunnel to originator
- Log: `⚡ Bootstrap proxy handing off to full API proxy`
- After handoff, API continues working (now served locally or via election)

### 20. Bootstrap proxy not started for originator

```bash
mesh-llm serve --model Qwen2.5-3B
```

- No `⚡ API ready (bootstrap)` message (only joiners get bootstrap proxy)
- API port opens only after election resolves

## No-Arg Behavior & Management API

### 21. No-arg help

```bash
mesh-llm
```

- Prints the same usage/help text as `mesh-llm --help`
- No ports are bound
- `curl localhost:3131/api/status` fails to connect


### 22. Join via console

```bash
mesh-llm client --auto
# In browser: http://localhost:3131 → Discover → Join
# Or via API:
curl -X POST localhost:3131/api/join -H 'Content-Type: application/json' -d '{"token":"..."}'
```

- `/api/join` triggers full flow: connect → gossip → assign model → download → serve
- Console updates: status, peers, model name all reflect new state
- Inference port starts working after model loads

### 23. Management API while serving

```bash
mesh-llm serve --auto
# After serving:
curl localhost:3131/api/status   # JSON: node, peers, routing, mesh_id, mesh_name
curl localhost:3131/api/events   # SSE stream
curl 'localhost:3131/api/search?q=qwen&catalog=true&artifact=gguf&limit=5' # JSON search results
curl localhost:3131/api/discover # Nostr meshes (current mesh marked by mesh_id)
```

- `/api/status` includes `mesh_id` and `mesh_name`
- SSE events push every 2s and on topology changes
- `/api/search` returns 200 JSON with canonical model refs for matching results
- Discover results can be matched to current mesh by `mesh_id`

### 24. HTTP proxy single-request connection contract

- Send a routed inference request, then pipeline or reuse the same TCP
  connection for a second request.
- Verify only the first request reaches the selected upstream.
- Verify the proxy closes the routed connection after the first response.
- Verify the upstream-observed request includes `Connection: close`.

## Resilience

### 11. Dead peer cleanup

- Kill a node with `kill -9`
- Cleanup happens in ~41s via the reconnect-gossip-probe mechanism:
  1. QUIC detects connection drop (~5-30s depending on idle timeout and relay state)
  2. Reconnect attempt with 10s gossip probe timeout
  3. Gossip probe fails → `remove_peer` called immediately
- Heartbeat also detects dead peers (60s interval, 2 consecutive failures) as a fallback
- On-use detection: tunnel failure → immediate death broadcast via stream 0x06
- Dead model goes cold, peer removed from list, death broadcast to mesh
- Dead peer won't be re-added by gossip (dead_peers set)
- Console updates automatically

## MoE Smoke Tests

These are the minimum smoke tests for leader-planned MoE recovery. They should be kept green as the runtime changes.

### 10z. Direct `moe-split` family smoke

Run the direct splitter smoke before remote mesh experiments:

```bash
just moe-split-smoke
```

Or target specific families:

```bash
scripts/moe-split-smoke.sh llama.cpp/build/bin glm-deepseek2 qwen3-a3b
```

Current preferred family matrix:

| Family | Preferred model |
|---|---|
| `qwen3-a3b` | `Qwen3-30B-A3B-Q4_K_M` |
| `qwen3-next` | `Qwen3-Coder-Next-Q4_K_M-00001-of-00004.gguf` |
| `glm-deepseek2` | `GLM-4.7-Flash-Q4_K_M` |
| `olmoe` | `OLMoE-1B-7B-0924-Instruct-Q4_K_M` or `OLMoE-1B-7B-0125-Instruct-Q4_K_M` |

What it validates:

- `llama-moe-split` can generate both sides of a `2`-way split.
- Each shard can be loaded by `llama-server`.
- Splitter regressions fail early with a direct tool-level repro, before the mesh runtime is involved.

### 10zz. Live MoE inference smoke

Use this once a real split is already up for a family that passes direct `moe-split` validation.

```bash
just moe-live-smoke \
  model=Qwen3-30B-A3B-Q4_K_M \
  api_url=http://studio54.local:9337 \
  console_url=http://studio54.local:3131
```

To validate more than one console view of the same deployment, call the script directly:

```bash
scripts/moe-live-smoke.sh \
  --expected-nodes 2 \
  Qwen3-30B-A3B-Q4_K_M \
  http://studio54.local:9337 \
  http://studio54.local:3131 \
  http://build.local:3131
```

What it validates:

- `/api/status` reports the model as `warm`
- `node_count` and `active_nodes` agree with the expected MoE topology
- `/v1/models` exposes the model through the chosen API
- `/v1/chat/completions` succeeds through the mesh

Current preferred live-inference families on `studio54.local + build.local`:

| Family | Preferred model | Notes |
|---|---|---|
| `qwen3-a3b` | `Qwen3-30B-A3B-Q4_K_M` | Main live control case |
| `glm-deepseek2` | `GLM-4.7-Flash-Q4_K_M` | Expect slower shard-1 startup on `build.local` |
| `olmoe` | `OLMoE-1B-7B-0924-Instruct-Q4_K_M` | Use `--split --max-vram 4.0` on this pair to force the live MoE path |

Coverage note:

- `just moe-split-smoke` is the CI-safe family gate.
- `scripts/moe-live-smoke.sh` is the manual or remote-runner integration check after actual mesh nodes are up.

### 11a. Two-node MoE split collapses to one survivor

```bash
# node A
mesh-llm --model Qwen3-Coder-Next-Q4_K_M --auto --no-self-update

# node B
mesh-llm --model Qwen3-Coder-Next-Q4_K_M --auto --no-self-update --split --join <TOKEN>
```

- Verify the two-node split comes up and `/v1/chat/completions` works.
- Kill one node with `kill -9`.
- The survivor should stop counting the dead shard in the active MoE set.
- The survivor should reconfigure to a one-node topology and serving should recover without manual restart.

### 11b. Three-node MoE split shrinks to two survivors

- Start a three-node split on the same exact MoE model identity.
- Kill one shard node.
- The remaining two nodes should re-elect on `n_nodes = 2`.
- The dead shard should disappear from the active MoE target map.
- Serving should recover on the two-node topology without killing the survivors.

### 11c. Recovered node does not cause immediate flap

- Start a two-node or three-node MoE split.
- Kill one shard node and wait for the deployment to fail down.
- Restart the dead node.
- Verify the node reappears in mesh membership.
- Verify the deployment does **not** immediately expand back up on the first healthy signal.
- Verify the deployment still does **not** expand during the quiet window after probation expires.
- Verify the leader keeps the existing healthy shard set unless the larger topology is explicitly needed.
- If the recovered node enables a materially better plan, verify scale-up happens only after the quiet window, not before.

### 11d. Full-coverage fallback replica

- Start a split deployment where one additional node can serve the full expert set for the same exact model identity.
- Kill an active shard.
- Verify request routing can fail over to the full-coverage target instead of blindly retrying a different partial shard.
- Verify the surviving cluster keeps serving while the leader recomputes the split on the remaining active shard set.

### 11e. Flaky network does not churn the MoE topology

- Run a two-node or three-node MoE split.
- Inject transient packet loss or briefly interrupt control traffic without killing the process.
- Verify the deployment does not oscillate between `N` and `N-1` on a single blip.
- Verify serving stays available when the shard remains reachable for inference.

### 12. Node rejoin

- Kill a node, restart it with `--join <token>`
- Rejoin loop (60s) reconnects to bootstrap if connection drops
- Inbound reconnection clears dead_peers entry
- Model goes warm again, cross-model routing resumes

### 13. Gossip stability

- Regossip after becoming host should NOT cause restart loops
- Log should show "still host, no restart needed" on re-check
- llama-server starts exactly once per election (not 5-9 times)
- Heartbeat gossip doesn't re-discover dead peers (discover_peers=false)

## Control-Plane Protocol (Protobuf v1)

The control plane uses QUIC ALPN `mesh-llm/1` with the `meshllm.node.v1` protobuf schema. All five scoped control-plane streams use 4-byte LE framing followed by protobuf bytes.

| Stream | Type | Format |
|--------|------|--------|
| 0x01 | GOSSIP | protobuf `GossipFrame` |
| 0x03 | TUNNEL_MAP | protobuf `TunnelMap` |
| 0x05 | ROUTE_REQUEST | protobuf `RouteTableRequest` / `RouteTable` |
| 0x06 | PEER_DOWN | protobuf `PeerDown` |
| 0x07 | PEER_LEAVING | protobuf `PeerLeaving` |

Raw TCP relay streams (0x02 RPC, 0x04 HTTP) are unchanged.


### Verifying protobuf gossip in logs

After two nodes connect, look for log lines indicating gossip was exchanged:

```
DEBUG mesh: gossip received from <peer_id>
DEBUG mesh: admitted peer <peer_id>
```

Absence of JSON-related log lines for streams 0x01/0x03/0x05/0x06/0x07 confirms the protobuf path is active.

## Single-machine testing with ephemeral keys

Set `MESH_LLM_EPHEMERAL_KEY=1` to give a second process a unique identity on the same machine.
Without this, both processes share `~/.mesh-llm/key` and appear as the same node.

### 14. Forced split on one machine

```bash
# Terminal 1: host with --split
mesh-llm serve --model Qwen2.5-3B --port 9337 --split --console

# Terminal 2: worker with ephemeral key
MESH_LLM_EPHEMERAL_KEY=1 mesh-llm serve --model Qwen2.5-3B --join <TOKEN> --port 9338 --split --max-vram 1
```

- Host starts solo, then re-elects with split when worker joins
- Worker becomes rpc-server, proxies API to host
- Tensor split proportional to VRAM (e.g. `0.98,0.02`)
- Kill worker → host detects via heartbeat (~60s), reverts to solo mode

### 15. Passive client on one machine

```bash
# Terminal 1: host
mesh-llm serve --model Qwen2.5-3B --port 9337

# Terminal 2: client (the client surface uses an ephemeral key automatically)
mesh-llm client --join <TOKEN> --port 9338
```

- Client connects without gossip (no peer list entry on host)
- `/v1/models` returns models from routing table
- Inference routes through QUIC tunnel to host
- Host does NOT see client in its peer list (zero per-client state)

## Deploy to remote node

```bash
just bundle
# scp, then on remote:
codesign -s - ~/mesh-bundle/mesh-llm ~/mesh-bundle/rpc-server ~/mesh-bundle/llama-server
xattr -cr ~/mesh-bundle/mesh-llm ~/mesh-bundle/rpc-server ~/mesh-bundle/llama-server
```

Must codesign + xattr after every scp or macOS kills the binary (exit 137).

## Cleanup

```bash
pkill -f mesh-llm; pkill -f rpc-server; pkill -f llama-server
```

Always kill all three — child processes can orphan.
