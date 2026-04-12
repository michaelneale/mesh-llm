# mesh-llm Design

A Rust sidecar that turns llama.cpp RPC into a peer-to-peer mesh. Nodes find
each other over QUIC (via [iroh](https://iroh.computer)), form a mesh of
tunnels, and llama.cpp runs unmodified on top — rpc-server and llama-server
just see local TCP sockets.

## Architecture

```
src/
├── main.rs                  CLI args, orchestration (auto, idle, passive)
├── lib.rs                   Crate root re-exports
├── api/                     Management API (:3131): status, events, discover, join
├── cli/                     Clap types, command parsing, command handlers
├── crypto/                  Key management, envelope encryption, keychain
├── inference/
│   ├── election.rs          Per-model host election, tensor split, llama-server lifecycle
│   ├── launch.rs            rpc-server and llama-server process management
│   ├── moe.rs               MoE detection, expert rankings, split orchestration
│   └── pipeline.rs          Inference pipeline coordination
├── mesh/mod.rs              Node struct, QUIC endpoint, gossip, peer management, mesh identity
├── models/
│   ├── capabilities.rs      Vision/audio/multimodal/reasoning capability inference
│   ├── catalog.rs           Model catalog and HuggingFace downloads
│   ├── resolve.rs           Model path resolution, mmproj lookup
│   └── ...                  GGUF parsing, inventory, search, topology
├── network/
│   ├── proxy.rs             HTTP proxy: request parsing, model routing, response helpers
│   ├── router.rs            Request classification, model scoring, multimodal routing
│   ├── tunnel.rs            TCP ↔ QUIC relay (RPC + HTTP), B2B rewrite map
│   ├── nostr.rs             Nostr discovery, score_mesh(), smart_auto()
│   ├── affinity.rs          Prefix-affinity request routing
│   └── rewrite.rs           REGISTER_PEER interception and endpoint rewriting
├── plugins/
│   └── blobstore/           Request-scoped media object storage for multimodal
├── protocol/                Wire protocol types, protobuf encoding/decoding
├── runtime/                 Top-level process orchestration, startup coordination
└── system/                  Hardware detection, benchmarking, self-update
```

## Node Roles

```rust
enum NodeRole {
    Worker,                      // rpc-server, provides GPU compute
    Host { http_port: u16 },     // llama-server + rpc-server, serves HTTP API
    Client,                      // no compute, just API access via tunnel
}
```

Roles are exchanged via gossip. Preferred peers use `meshllm.node.v1` protobuf on QUIC ALPN `mesh-llm/1`; legacy peers may still negotiate `mesh-llm/0` and use the older JSON gossip payloads. A node transitions Worker → Host when elected.

A newly connected peer is quarantined until it sends a valid `GossipFrame` with `gen = 1` (quarantine-until-gossip admission model). Only streams 0x01 (GOSSIP) and 0x05 (ROUTE_REQUEST) are accepted before admission. All other streams are rejected until the peer is admitted.

## Control-Plane Protocol

The control plane prefers QUIC ALPN `mesh-llm/1` using the `meshllm.node.v1` protobuf schema. Scoped control-plane streams on `/1` use 4-byte LE framing followed by protobuf bytes. For backward compatibility, peers may also negotiate `mesh-llm/0`, which preserves the legacy JSON/raw payloads on those same streams.

Mixed meshes containing both `/0` and `/1` nodes are supported. `/0` links are compatibility mode only, so they do not carry protobuf-only fields.

See [message_protocol.md](../../message_protocol.md) for the full wire format specification.

## QUIC Stream Types

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

## Multi-Model

Different nodes serve different models. The API proxy on each node peeks at
the `model` field in POST bodies and routes to the correct host via QUIC tunnel.

- **One model per node** — no VRAM double-commitment
- **Solo by default** — if VRAM ≥ model_size × 1.1, run solo
- **Per-model election groups** — nodes serving the same model elect a host independently
- **Auto-assignment** — joiners without `--model` get assigned based on mesh needs and what's on disk

### HTTP/1.1 Connection Contract

For routed inference requests, the proxy buffers and routes exactly one HTTP
request per client connection:

- The full request is framed first (`Content-Length` or chunked) before routing.
- The forwarded upstream request is rewritten to `Connection: close`.
- After the buffered request is written upstream, the proxy only relays the
  response back to the client.
- Additional client bytes on the same connection are ignored and dropped when
  the connection closes; they are not replayed to the already-selected
  upstream.

This is an intentional safety tradeoff. The proxy does not currently implement
per-request routing for persistent HTTP/1.1 keep-alive or pipelined multi-
request connections. Clients should open a fresh connection for each routed
inference request.

## Mesh Identity

Every mesh has a stable `mesh_id`:
- **Named mesh**: `hash(name + originator_nostr_pubkey)` — deterministic, unique per creator
- **Unnamed mesh**: random UUID, persisted to `~/.mesh-llm/mesh-id`

Propagated via gossip (`PeerAnnouncement.mesh_id`) and routing table (`RoutingTable.mesh_id`).
Published in Nostr listings (`MeshListing.mesh_id`).
Saved to `~/.mesh-llm/last-mesh` on successful join for sticky preference scoring.

## Bootstrap Proxy

When joining an existing mesh, a tunnel-only API proxy starts immediately on the
local port — before rpc-server or llama-server are ready. Requests are tunneled to
mesh hosts via QUIC. When the real `api_proxy` is ready, it takes over the listener.

This gives instant API access (within seconds of `mesh-llm serve --join`) while the local
GPU loads its model in the background.

## Local Node Config

`mesh-llm serve` owns startup model configuration. By default it reads
`~/.mesh-llm/config.toml`, which now serves as the unified local node config for:

- startup models under `[[models]]`
- local GPU startup policy under `[gpu]`
- plugin declarations under `[[plugin]]`

Phase 2 keeps this config intentionally local-node only. There is no authored mesh-wide
`[[nodes]]` state yet.

CLI precedence is by concern:

- explicit `--model` or `--gguf` ignores configured `[[models]]`
- explicit `--ctx-size` overrides configured `ctx_size`
- plugin config continues to load from the same file

Pinned GPU startup is also local-node only:

- `[gpu].assignment = "pinned"` means each configured `[[models]]` entry must carry its own `gpu_id`
- valid IDs come from the local `mesh-llm gpus` / `mesh-llm gpus --json` inventory surface
- pin resolution is host-local and fail-closed: missing, ambiguous, unsupported, or stale IDs abort startup and config push for that node instead of silently falling back to auto placement
- explicit CLI `--model` / `--gguf` still bypass configured `[[models]]`, so they do not inherit config-owned pinned IDs

Bare `mesh-llm serve` is the config-owned path. If `[[models]]` is empty, it warns,
prints help, and exits cleanly. Background services use that path directly.

## Passive Mode

Two flavors, one code path (`run_passive()`):
- **`--client`**: pure consumer, ephemeral key, no gossip, routing table only
- **Standby GPU**: has VRAM + models on disk, watches for topology changes, promotes when needed

Passive nodes get routing tables via `STREAM_ROUTE_REQUEST` (0x05), not full gossip.
Scales to hundreds of clients without O(n²) gossip cost.

## Demand-Aware Rebalancing

- `record_request(model)` increments per-model counter on every API proxy request
- `snapshot_request_rates()` computes delta each gossip cycle (requests/min)
- Rates gossipped in `PeerAnnouncement.request_rates`
- Standby nodes check on 60s timer + topology changes via `tokio::select!`
- Promotion triggers: (1) model with 0 servers, (2) ≥3x demand imbalance + ≥10 req/min, (3) single hot model ≥10 req/min

## Latency-Aware Tensor Split

When a model requires splitting across nodes:
1. Filter candidates by `rtt_ms < 80ms`
2. Sort by RTT ascending (unknown RTT sorts last)
3. Greedily accumulate VRAM until `≥ model_size × 1.1`
4. Stop — don't add unnecessary high-latency peers

## Event-Driven Peer Management

- **Reconnect-gossip-probe** — when a QUIC connection drops, the node reconnects and awaits gossip with a 10s timeout. If gossip fails, the peer is removed immediately. Dead peer cleanup typically completes in ~41s after `kill -9`.
- **60s heartbeat** with 2-consecutive-failure threshold (fallback path)
- **Death broadcasts** (`STREAM_PEER_DOWN`, protobuf) for immediate notification
- **Clean shutdown** (`STREAM_PEER_LEAVING`, protobuf) on ctrl-c — only removes the sender, not other peers
- **Dead peers set** prevents gossip from re-adding killed nodes
- **Tunnel failure detection** triggers immediate death broadcast

## B2B Direct Transfer

When the model is split across workers, activation tensors flow directly
between workers (1 hop) instead of through the host (2 hops):
1. Each node broadcasts `{EndpointId → tunnel_port}` via `STREAM_TUNNEL_MAP`
2. `rewrite.rs` intercepts `REGISTER_PEER` and rewrites ports for local tunnels
3. llama.cpp's `PUSH_TENSOR_TO_PEER` goes directly between workers

## Management API (port 3131)

Separate from the inference API (port 9337). Serves mesh management endpoints
and the embedded web dashboard.

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/status` | GET | Live mesh state (JSON): node, peers, routing, targets |
| `/api/models` | GET | Mesh model inventory for the dashboard and operators |
| `/api/events` | GET | SSE stream of status updates (2s interval + on change) |
| `/api/discover` | GET | Browse Nostr-published meshes |
| `/api/join` | POST | Join a mesh by invite token `{"token":"..."}` |
| `/api/chat` | POST | Proxy to inference API (`/v1/chat/completions`) |
| `/` | GET | Embedded web dashboard |

The dashboard is a thin client. Live node state comes from `/api/status` and
`/api/events`, while model inventory comes from `/api/models`. Mesh management
works without the HTML via curl/scripts.

Always enabled on port 3131 (configurable with `--console <port>`).

## No-Arg Behavior

`mesh-llm` with no arguments prints the standard CLI help and exits.

Management API and inference listeners are only started by active modes such as
`--model`, `--join`, `--auto`, or `--client`. This avoids surprising port binds
for users who run the binary just to check usage.

## Hardware Detection

`hardware.rs` collects GPU and host info at startup via the `Collector` trait:

```rust
trait Collector {
    fn collect(&self) -> Vec<Metric>;
}
```

| Implementation | Platform | Source |
|---|---|---|
| `DefaultCollector` | macOS (Metal/CPU) | `system_profiler`, `vm_stat` |
| `DefaultCollector` | Linux NVIDIA | `/proc/driver/nvidia`, `nvidia-smi` |
| `DefaultCollector` | Linux AMD | `/sys/class/drm`, `rocm-smi` |
| `TegraCollector` | Jetson / Tegra | sysfs + `tegrastats` |

`survey()` calls all applicable collectors and returns a `HardwareSurvey` with `gpu_name`, `gpu_vram` (per-GPU bytes), `gpu_reserved` (per-GPU reserved or unavailable bytes when the platform reports a true reserved/unavailable metric), `vram_bytes` (total), `hostname`, `is_soc`, and per-device `GpuFacts` entries. Benchmark-derived memory-bandwidth and compute-throughput hints are attached later when cached or freshly measured results are available. ROCm `rocm-smi --showmeminfo` and Intel `xpu-smi` discovery expose live used-memory counters, so mesh-llm intentionally omits `gpu_reserved` for those backends instead of reinterpreting used bytes as reserved memory.

### Gossip Fields

`PeerAnnouncement` fields carried in the `meshllm.node.v1` protobuf `GossipFrame`:

| Field | Type | Description |
|---|---|---|
| `gpu_name` | `Option<String>` | Comma-separated GPU model names |
| `hostname` | `Option<String>` | System hostname |
| `is_soc` | `Option<bool>` | True for Tegra/Jetson (unified memory) |
| `gpu_vram` | `Option<String>` | Comma-separated per-GPU VRAM in bytes |
| `gpu_reserved_bytes` | `Option<String>` | Comma-separated per-GPU reserved bytes when the platform reports a true reserved/unavailable metric |
| `gpu_mem_bandwidth_gbps` | `Option<String>` | Comma-separated per-GPU memory bandwidth measurements or cached benchmark results |
| `gpu_compute_tflops_fp32` | `Option<String>` | Comma-separated per-GPU FP32 compute-throughput hints |
| `gpu_compute_tflops_fp16` | `Option<String>` | Comma-separated per-GPU FP16 compute-throughput hints |
| `available_model_metadata` | `repeated CompactModelMetadata` | GGUF-derived metadata per available model |
| `available_model_sizes` | `map<string, uint64>` | File sizes in bytes per model name |
| `mesh_id` | `optional string` | Stable mesh identity (self entry only) |
| `demand` | `repeated ModelDemandEntry` | Per-model demand entries (self entry only) |

GGUF-derived metadata (architecture, quantization type, tokenizer, RoPE parameters, expert counts) is transported via `CompactModelMetadata` in the `available_model_metadata` field. This lets peers learn model capabilities without downloading the file. The `ScannedModel` type in the proto schema carries the same information for catalog-level model listings. Current gossip sanitization still strips `available_models`, `available_model_metadata`, and `available_model_sizes` before sending announcements on the wire, so these schema fields remain compatibility surface rather than a second transitive model-inventory source.

### `--enumerate-host` Flag

Controls whether the host-identifying inventory fields (`gpu_name`, `hostname`, `gpu_vram`, and `gpu_reserved_bytes`) appear in gossip. `is_soc` is always sent. Benchmark-derived bandwidth and compute hints remain additive optional fields when available. `gpu_reserved_bytes` stays omitted on backends such as ROCm and Intel where the tooling does not report a true reserved/unavailable memory metric. Default: `false` (privacy-preserving; peers see VRAM totals but not GPU model or hostname).

```
--enumerate-host    # opt in: peers learn your GPU name and hostname
```

### API Shape

`GET /api/status` — self node:
```json
{
  "my_hostname": "carrack",
  "my_is_soc": false,
  "gpus": [{"name": "NVIDIA RTX 5090", "vram_bytes": 34359738368, "reserved_bytes": 1073741824, "mem_bandwidth_gbps": 1792.0, "compute_tflops_fp32": 104.8, "compute_tflops_fp16": 209.6}]
}
```

For ROCm and Intel hosts, `reserved_bytes` is omitted because their standard CLI telemetry exposes live used-memory counters rather than a true reserved/system-memory value.

`peers[]` entries (only when peer has `--enumerate-host`):
```json
{"hostname": "lemony-28", "is_soc": true, "gpus": [{"name": "Tegra AGX Orin", "vram_bytes": 0}]}
```

## Nostr Discovery

Opt-in mesh advertisement via Nostr relays (NIP-89, kind 31990):
- `--publish`: republish listing every 60s (TTL 120s)
- `--auto`: discover meshes, score them, health-probe, join best
- Publish watchdog: if publisher dies, another node takes over
- `score_mesh()`: region match (+200), capacity, node count, VRAM, sticky preference (+500)
- `smart_auto()`: picks best mesh or recommends starting new one with models for your VRAM
