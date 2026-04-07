# Humans

Developer and contributor reference for mesh-llm. Architecture, testing, release process, roadmap, and plugin design — everything a human working on the project needs.

## Principles

These guide every design decision. When in doubt, come back here.

- **One command to run.** `mesh-llm serve --auto` should be all you need. No config files, no coordination, no prerequisites beyond the binary.
- **Batteries included.** Models download automatically. Backends are bundled. The web console ships inside the binary. Nothing extra to install.
- **Sensible defaults.** Solo when the model fits. Split only when it has to. Draft models auto-paired. Context sized to VRAM. Thinking off when it hurts more than it helps.
- **Always compatible in the mesh.** Older and newer nodes must coexist. Protocol negotiation (`mesh-llm/0` ↔ `mesh-llm/1`) keeps mixed meshes working. Rolling upgrades, not flag days.
- **Public and private.** `--auto` joins public meshes for experimentation. `--join <token>` creates private meshes for production. Same binary, same API, same behavior.
- **Support as many platforms as possible.** macOS Metal, Linux CUDA, Linux ROCm, Linux Vulkan, Linux CPU, Jetson/Tegra, Windows. If it has compute, it should be able to join.

---

## Architecture

A Rust sidecar that turns llama.cpp RPC into a peer-to-peer mesh. Nodes find each other over QUIC (via [iroh](https://iroh.computer)), form a mesh of tunnels, and llama.cpp runs unmodified on top — rpc-server and llama-server just see local TCP sockets.

### Source layout

```text
mesh-llm/src/
├── lib.rs                 crate entrypoint, module wiring, version, public re-exports
├── main.rs                binary entrypoint
├── api/                   management API, status shaping, HTTP routing
├── cli/                   clap types, subcommands, command handlers
├── inference/             election, launch, pipeline splits, MoE orchestration
├── mesh/                  peer membership, gossip, routing tables, QUIC node behavior
├── models/                catalog, search, GGUF metadata, inventory, resolution
├── network/               proxying, tunnels, affinity, Nostr discovery, endpoint rewrite
├── plugin/                external plugin host, MCP bridge, transport, config
├── plugins/               built-in plugins shipped with mesh-llm (blackboard, lemonade)
├── protocol/              control-plane protocol versions and conversions
├── runtime/               top-level startup flows and local runtime coordination
└── system/                hardware detection, benchmarking, self-update
```

### Node roles

```rust
enum NodeRole {
    Worker,                      // rpc-server, provides GPU compute
    Host { http_port: u16 },     // llama-server + rpc-server, serves HTTP API
    Client,                      // no compute, just API access via tunnel
}
```

Roles are exchanged via gossip. Preferred peers use `meshllm.node.v1` protobuf on QUIC ALPN `mesh-llm/1`; legacy peers may still negotiate `mesh-llm/0` and use the older JSON gossip payloads. A node transitions Worker → Host when elected.

A newly connected peer is quarantined until it sends a valid `GossipFrame` with `gen = 1` (quarantine-until-gossip admission model). Only streams 0x01 (GOSSIP) and 0x05 (ROUTE_REQUEST) are accepted before admission.

### Control-plane protocol

The control plane prefers QUIC ALPN `mesh-llm/1` using the `meshllm.node.v1` protobuf schema. Scoped control-plane streams on `/1` use 4-byte LE framing followed by protobuf bytes. For backward compatibility, peers may also negotiate `mesh-llm/0`, which preserves the legacy JSON/raw payloads on those same streams.

Mixed meshes containing both `/0` and `/1` nodes are supported. `/0` links are compatibility mode only, so they do not carry protobuf-only fields.

See [mesh-llm/docs/message_protocol.md](mesh-llm/docs/message_protocol.md) for the full wire format specification.

### QUIC stream types

Single QUIC connection per peer, multiplexed by 1-byte prefix:

| Byte | Type | Purpose | Format |
|------|------|---------|--------|
| 0x01 | GOSSIP | Peer announcements (role, serving, VRAM, models, demand, mesh_id) | protobuf `GossipFrame` |
| 0x02 | TUNNEL_RPC | TCP relay to remote rpc-server | raw TCP relay |
| 0x03 | TUNNEL_MAP | B2B tunnel port map exchange | protobuf `TunnelMap` |
| 0x04 | TUNNEL_HTTP | TCP relay to remote llama-server HTTP | raw TCP relay |
| 0x05 | ROUTE_REQUEST | Routing table for passive nodes (hosts + models) | protobuf `RouteTableRequest` / `RouteTable` |
| 0x06 | PEER_DOWN | Death broadcast (immediate, from any node that detects a death) | protobuf `PeerDown` |
| 0x07 | PEER_LEAVING | Clean shutdown broadcast (ctrl-c) | protobuf `PeerLeaving` |

Streams 0x02 and 0x04 are raw TCP relay tunnels and are not subject to protobuf framing or generation validation.

### Multi-model routing

Different nodes serve different models. The API proxy on each node peeks at the `model` field in POST bodies and routes to the correct host via QUIC tunnel. One model per node — no VRAM double-commitment. Solo by default if VRAM ≥ model_size × 1.1.

#### HTTP/1.1 connection contract

The proxy buffers and routes exactly one HTTP request per client connection. The forwarded upstream request is rewritten to `Connection: close`. Clients should open a fresh connection for each routed inference request.

### Mesh identity

Every mesh has a stable `mesh_id`:
- **Named mesh**: `hash(name + originator_nostr_pubkey)` — deterministic, unique per creator
- **Unnamed mesh**: random UUID, persisted to `~/.mesh-llm/mesh-id`

Propagated via gossip, routing tables, and Nostr listings. Saved to `~/.mesh-llm/last-mesh` for sticky preference scoring.

### Bootstrap proxy

When joining, a tunnel-only API proxy starts immediately on the local port — before rpc-server or llama-server are ready. Requests are tunneled to mesh hosts via QUIC. When the real `api_proxy` is ready, it takes over. This gives instant API access within seconds of `mesh-llm serve --join`.

### Passive mode

Two flavors, one code path (`run_passive()`):
- **`--client`**: pure consumer, ephemeral key, no gossip, routing table only
- **Standby GPU**: has VRAM + models on disk, watches for topology changes, promotes when needed

Passive nodes get routing tables via `STREAM_ROUTE_REQUEST` (0x05), not full gossip. Scales to hundreds of clients without O(n²) gossip cost.

### Demand-aware rebalancing

- `record_request(model)` increments per-model counter on every API proxy request
- `snapshot_request_rates()` computes delta each gossip cycle (requests/min)
- Rates gossiped in `PeerAnnouncement.request_rates`
- Promotion triggers: (1) model with 0 servers, (2) ≥3x demand imbalance + ≥10 req/min, (3) single hot model ≥10 req/min

### Latency-aware tensor split

When a model requires splitting across nodes:
1. Filter candidates by `rtt_ms < 80ms`
2. Sort by RTT ascending (unknown RTT sorts last)
3. Greedily accumulate VRAM until `≥ model_size × 1.1`
4. Stop — don't add unnecessary high-latency peers

### Event-driven peer management

- **Reconnect-gossip-probe** — when a QUIC connection drops, reconnect and await gossip with a 10s timeout. Dead peer cleanup typically completes in ~41s after `kill -9`.
- **60s heartbeat** with 2-consecutive-failure threshold (fallback path)
- **Death broadcasts** (`STREAM_PEER_DOWN`) for immediate notification
- **Clean shutdown** (`STREAM_PEER_LEAVING`) on ctrl-c
- **Dead peers set** prevents gossip from re-adding killed nodes

### B2B direct transfer

When the model is split across workers, activation tensors flow directly between workers (1 hop) instead of through the host (2 hops). Each node broadcasts `{EndpointId → tunnel_port}` via `STREAM_TUNNEL_MAP`. `rewrite.rs` intercepts `REGISTER_PEER` and rewrites ports for local tunnels.

### Management API (port 3131)

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/status` | GET | Live mesh state (JSON): node, peers, routing, targets |
| `/api/models` | GET | Mesh model inventory |
| `/api/events` | GET | SSE stream of status updates (2s interval + on change) |
| `/api/discover` | GET | Browse Nostr-published meshes |
| `/api/join` | POST | Join a mesh by invite token |
| `/api/chat` | POST | Proxy to inference API |
| `/` | GET | Embedded web dashboard |

### Hardware detection

`hardware.rs` collects GPU and host info at startup via the `Collector` trait:

| Implementation | Platform | Source |
|---|---|---|
| `DefaultCollector` | macOS (Metal/CPU) | `system_profiler`, `vm_stat` |
| `DefaultCollector` | Linux NVIDIA | `/proc/driver/nvidia`, `nvidia-smi` |
| `DefaultCollector` | Linux AMD | `/sys/class/drm`, `rocm-smi` |
| `TegraCollector` | Jetson / Tegra | sysfs + `tegrastats` |

`--enumerate-host` opts in to sharing GPU name and hostname in gossip (default: off for privacy).

### Nostr discovery

Opt-in mesh advertisement via Nostr relays (NIP-89, kind 31990):
- `--publish`: republish listing every 60s (TTL 120s)
- `--auto`: discover meshes, score them, health-probe, join best
- `score_mesh()`: region match (+200), capacity, node count, VRAM, sticky preference (+500)

---

## Contributing

Join the [#mesh-llm channel on the Goose Discord](https://discord.gg/goose-oss) for discussion and questions.

### Prerequisites

- `just`
- `cmake`
- Rust toolchain (`cargo`)
- Node.js 24 + npm (for UI development)

**macOS**: Apple Silicon. Metal is used automatically.

**Linux NVIDIA**: x86_64 with an NVIDIA GPU. Requires the CUDA toolkit (`nvcc` in your `PATH`).

**Linux AMD**: ROCm/HIP when ROCm is installed.

**Linux Vulkan**: Vulkan development files and `glslc`.

**Windows**: `just build` auto-detects `cuda`, `hip`/`rocm`, `vulkan`, or `cpu`.

### Build from source

```bash
just build
```

On Linux, auto-detects CUDA vs ROCm vs Vulkan. Override:

```bash
just build backend=rocm
just build backend=vulkan
just build backend=cpu
just build cuda_arch=90
```

Create a portable bundle:

```bash
just bundle
```

### UI development

Terminal A:

```bash
mesh-llm --port 9337 --console 3131
```

Terminal B:

```bash
just ui-dev
```

Open `http://127.0.0.1:5173`. Proxies `/api/*` to `http://127.0.0.1:3131`.

### Useful commands

```bash
just stop             # stop mesh/rpc/llama processes
just test             # quick test against :9337
just --list           # list all recipes
```

### CI

CI uses `dorny/paths-filter` to skip jobs when unchanged areas are modified. Docs-only changes skip all builds. UI-only changes skip GPU backend builds.

### Benchmark binaries

Optional — not built by `just build`. See `benchmarks/` and the `just benchmark-build-*` recipes.

---

## Testing

### Local inspection

```bash
mesh-llm gpus                  # inspect local GPUs
```

### Startup config smoke

Create `~/.mesh-llm/config.toml` with `[[models]]` entries, then `mesh-llm serve`. Verify:

- Both configured models are considered for launch
- Empty `[[models]]` prints a warning and exits
- `--model` / `--gguf` ignores configured models
- `--ctx-size` overrides configured `ctx_size`

### Single-model scenarios

**1. Solo** — `mesh-llm serve --model Qwen2.5-3B --console`. API on `:9337`, console on `:3131`.

**2. Two GPU nodes, model fits** — both run solo, API works from both.

**3. Forced split** — `--split` forces tensor split. llama-server has 2 RPC entries.

**4. Model too big** — split happens automatically without `--split`.

**5. Lite client** — `mesh-llm client --join <TOKEN>`. Ephemeral key, API tunneled via QUIC, host does not see client in peer list.

### Multi-model scenarios

**6. Two nodes, different models** — `/v1/models` lists both, cross-model routing works via QUIC.

**7. Auto-assignment** — joiner with no `--model` picks an unserved model already on disk.

**8. Lite client with multi-model** — client sees all models, routes to correct host per model.

**9. Unload** — `mesh-llm unload GLM-4.7-Flash-Q4_K_M`. Node exits cleanly, other nodes unaffected.

**9a. Runtime load/unload** — `mesh-llm load`, `mesh-llm status`, `mesh-llm unload`, and the REST equivalents on `/api/runtime`.

**10. Console model picker** — dropdown when >1 warm model, switching highlights serving node.

### Mesh identity

**16.** Named mesh (`--mesh-name`) produces deterministic ID. Different names → different IDs.

**17.** Joiner receives mesh ID via gossip/routing table.

**18.** Sticky preference: `~/.mesh-llm/last-mesh` saved on join, +500 score on next `--auto`.

### Bootstrap proxy

**19.** Joiner gets `⚡ API ready (bootstrap)` before GPU loads. Inference works through tunnel immediately.

**20.** Originator does not get bootstrap proxy.

### Resilience

**11.** Dead peer cleanup: ~41s via reconnect-gossip-probe. Heartbeat as fallback. Tunnel failure → immediate death broadcast.

**12.** Node rejoin: reconnects via 60s loop, dead_peers cleared on inbound reconnection.

**13.** Gossip stability: no restart loops, llama-server starts exactly once per election.

### MoE smoke tests

**10z.** `just moe-split-smoke` — direct splitter validation across model families.

**10zz.** `just moe-live-smoke` — live inference against a running MoE deployment.

**11a-11e.** MoE recovery scenarios: two-node collapse, three-node shrink, no-flap rejoin, full-coverage fallback, flaky network stability.

### Protocol compatibility

Start a `mesh-llm/0` node and join a `mesh-llm/1` mesh. Connection should negotiate `/0`, complete gossip, and exchange legacy payloads without breaking.

### Single-machine testing

Set `MESH_LLM_EPHEMERAL_KEY=1` for a second process on the same machine:

```bash
# Terminal 1
mesh-llm serve --model Qwen2.5-3B --port 9337 --split --console

# Terminal 2
MESH_LLM_EPHEMERAL_KEY=1 mesh-llm serve --model Qwen2.5-3B --join <TOKEN> --port 9338 --split --max-vram 1
```

### Deploy to remote node

```bash
just bundle
# scp, then on remote:
codesign -s - ~/mesh-bundle/mesh-llm ~/mesh-bundle/rpc-server ~/mesh-bundle/llama-server
xattr -cr ~/mesh-bundle/mesh-llm ~/mesh-bundle/rpc-server ~/mesh-bundle/llama-server
```

Must codesign + xattr after every scp or macOS kills the binary (exit 137).

### Cleanup

```bash
pkill -f mesh-llm; pkill -f rpc-server; pkill -f llama-server
```

Always kill all three — child processes can orphan.

---

## Release process

### Prerequisites

- `just`, `cmake`, `cargo`, `gh` CLI authenticated, llama.cpp fork cloned

### Steps

**1. Build fresh:**

```bash
just build
```

**2. Verify no homebrew deps:**

```bash
otool -L llama.cpp/build/bin/llama-server | grep -v /System | grep -v /usr/lib
otool -L llama.cpp/build/bin/rpc-server | grep -v /System | grep -v /usr/lib
otool -L target/release/mesh-llm | grep -v /System | grep -v /usr/lib
```

**3. Bundle:**

```bash
just bundle
```

Bundle naming: macOS bundles `*-metal`, Linux `*-cpu`/`*-cuda`/`*-rocm`/`*-vulkan`.

**4. Smoke test:**

```bash
mkdir /tmp/test-bundle && tar xzf /tmp/mesh-bundle.tar.gz -C /tmp/test-bundle --strip-components=1
/tmp/test-bundle/mesh-llm --model Qwen2.5-3B
# Verify inference at localhost:9337, then Ctrl+C
rm -rf /tmp/test-bundle
```

**5. Release:**

```bash
just release v0.X.0
```

Run from a clean `main` branch. Bumps version, refreshes `Cargo.lock`, commits as `v0.X.0: release`, pushes main, then pushes the tag.

**6. CI builds and publishes** — pushing a `v*` tag triggers `.github/workflows/release.yml` which builds for macOS, Linux (CPU/CUDA/ROCm/Vulkan), and Windows (CPU/CUDA/ROCm/Vulkan), then creates the GitHub release.

**7. Verify assets** — check that all platform bundles exist on the release page.

### Windows release recipes

```powershell
just release-build-windows
just release-build-cuda-windows
just release-build-amd-windows
just release-build-vulkan-windows
```

### Notes

- Unversioned `mesh-bundle.tar.gz` kept for compatibility with direct archive installs
- Release bundles use flavor-specific `rpc-server-<flavor>` / `llama-server-<flavor>` names
- `codesign` and `xattr` may be needed on receiving machines for macOS Gatekeeper

---

## Roadmap

High-level directions. Not promises — just things we're thinking about.

### Done

- **Smart model router** — heuristic classifier (Code/Reasoning/Chat/Creative/ToolCall), task-dominant scoring, multi-model per node
- **MoE expert sharding** — auto-detect, overlapping expert assignments, split locally, session-sticky routing, zero cross-node traffic
- **Blackboard** — shared ephemeral messages, multi-term search, PII scrub, MCP server, agent skill
- **Demand-based rebalancing** — unified demand map, standby auto-promote
- **Resilience** — Nostr re-discovery, llama-server watchdog, multi-host load balancing, VRAM-scaled context

### In progress / next

- **Connection stability** — relay decay investigation, health monitoring, periodic reconnect
- **Production relay infrastructure** — dedicated relays in key regions
- **SSD expert streaming** — run giant MoE models from NVMe on a single node (see [flash-moe](https://github.com/danveloper/flash-moe))
- **Agent launcher** — `mesh-llm run goose/pi/opencode` as one-command agent launch
- **Single binary** — compile llama.cpp into the Rust binary via llama-cpp-2
- **Mobile chat app** — scan QR code, join mesh, chat. "AirDrop for AI"

### TODO backlog

- **Mixture of Models** — route requests to specialized models by task type
- **Multi-model per host** — spare VRAM could serve multiple models simultaneously
- **Peer-to-peer model transfer** — fetch models from mesh peers instead of HuggingFace
- **Context-aware routing** — hosts advertise `n_ctx`, router skips hosts that can't fit the request
- **Retry on 400** — try next host instead of forwarding context overflow errors
- **Multi-node tensor split recovery** — re-split across remaining peers when one dies
- **Vision routing** — auto-route image requests to vision-capable models

---

## Plugins

mesh-llm has a plugin architecture for extending functionality without embedding feature logic in core.

### Current built-ins

- **blackboard** — shared mesh message feed + MCP surface
- **lemonade** — external OpenAI-compatible inference endpoint bridge (AMD Lemonade Server)

### Design summary

A plugin is a local service process launched by mesh-llm. The system has three core pieces:

1. One long-lived control connection per plugin process
2. Zero or more short-lived negotiated streams for large or streaming data
3. One declarative plugin manifest that the host projects into MCP, HTTP, and inference surfaces

mesh-llm owns: plugin lifecycle, IPC, HTTP serving, MCP serving, capability routing, mesh transport.

A plugin owns: its feature logic, local state, operation/resource/prompt handlers.

### Plugin author DSL

```rust
let plugin = mesh_llm_plugin::plugin! {
    metadata: PluginMetadata::new(...),
    startup_policy: PluginStartupPolicy::PrivateMeshOnly,
    provides: [capability("notes.v1")],
    mcp: [
        tool("search").description("Search notes").input::<SearchArgs>().handle(search),
        resource("notes://latest").name("Latest Notes").handle(read_latest),
    ],
    http: [
        get("/search").handle(search),
        post("/notes").handle(post_note),
    ],
    inference: [
        openai_http("local-llm", "http://127.0.0.1:8080/v1").managed_by_plugin(false),
    ],
    health: |_context| Box::pin(async move { Ok("ok".to_string()) }),
};
```

### Implementation plan

| Phase | What |
|---|---|
| 1 | Protocol and manifest types |
| 2 | Host runtime core (control connection, streams, health) |
| 3 | Manifest-driven MCP projection |
| 4 | Manifest-driven HTTP bindings |
| 5 | Capability resolution |
| 6 | Endpoint registration |
| 7 | Migrate existing built-ins |
| 8 | Validation plugins (llama backend, MLX, Ollama, Lemonade) |
| 9 | Host-owned plugin crypto API |

For the full plugin architecture spec, see [mesh-llm/docs/PLUGINS.md](mesh-llm/docs/PLUGINS.md). For the sequencing plan, see [mesh-llm/docs/PLUGINS_PLAN.md](mesh-llm/docs/PLUGINS_PLAN.md).

---

## Other docs

These remain as focused reference files:

| File | Content |
|---|---|
| [mesh-llm/docs/message_protocol.md](mesh-llm/docs/message_protocol.md) | Wire protocol spec |
| [mesh-llm/docs/MULTI_MODAL.md](mesh-llm/docs/MULTI_MODAL.md) | Multimodal capability details |
| [mesh-llm/docs/MoE_PLAN.md](mesh-llm/docs/MoE_PLAN.md) | MoE expert sharding design |
| [mesh-llm/docs/MoE_DEPLOY_DESIGN.md](mesh-llm/docs/MoE_DEPLOY_DESIGN.md) | MoE auto-deploy implementation |
| [mesh-llm/docs/MoE_SPLIT_REPORT.md](mesh-llm/docs/MoE_SPLIT_REPORT.md) | MoE splitting validation |
| [mesh-llm/docs/MOE_ISLANDS.md](mesh-llm/docs/MOE_ISLANDS.md) | Expert co-activation clustering |
| [mesh-llm/docs/MOE_STRATEGY_BENCHMARKS.md](mesh-llm/docs/MOE_STRATEGY_BENCHMARKS.md) | MoE strategy benchmarks |
| [mesh-llm/docs/MODEL_ROUTER.md](mesh-llm/docs/MODEL_ROUTER.md) | Model router design |
| [mesh-llm/docs/ROUTER_V2.md](mesh-llm/docs/ROUTER_V2.md) | Router v2 adaptive proposal |
| [mesh-llm/docs/ROUTER_BENCHMARKS.md](mesh-llm/docs/ROUTER_BENCHMARKS.md) | Router benchmarks |
| [mesh-llm/docs/PREFIX_AFFINITY_BENCHMARKS.md](mesh-llm/docs/PREFIX_AFFINITY_BENCHMARKS.md) | Prefix affinity benchmarks |
| [mesh-llm/docs/PLUGINS.md](mesh-llm/docs/PLUGINS.md) | Full plugin architecture spec |
| [mesh-llm/docs/PLUGINS_PLAN.md](mesh-llm/docs/PLUGINS_PLAN.md) | Plugin implementation sequencing |
| [fly/README.md](fly/README.md) | Fly.io deployment |
| [relay/README.md](relay/README.md) | Self-hosted iroh relay |
| [evals/README.md](evals/README.md) | Router evaluation suite |
