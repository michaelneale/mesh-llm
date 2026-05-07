# mesh-llm crate

Rust implementation of mesh-llm: a peer-to-peer control plane for embedded
skippy/llama inference over QUIC, with distributed routing, model
orchestration, plugin hosting, and a local management API.

For install and end-user usage, see the [project README](../../README.md). For deeper architecture and test flows, see [DESIGN.md](../../docs/design/DESIGN.md), [METRICS.md](../../docs/design/METRICS.md), [TESTING.md](../../docs/design/TESTING.md), [message_protocol.md](../../docs/design/message_protocol.md), and [LLAMA_STAGE_INTEGRATION_PLAN.md](../../docs/design/LLAMA_STAGE_INTEGRATION_PLAN.md).

## Source layout

The crate root stays intentionally small:

```text
src/
├── lib.rs                 crate entrypoint, module wiring, version, public re-exports
├── main.rs                binary entrypoint
├── api/                   management API, status shaping, HTTP routing
├── cli/                   clap types, subcommands, command handlers
├── inference/             embedded skippy serving, stage deployment, hooks
├── mesh/                  peer membership, gossip, routing tables, QUIC node behavior
├── models/                catalog, search, GGUF metadata, inventory, resolution
├── network/               proxying, tunnels, affinity, Nostr discovery, endpoint rewrite
├── plugin/                external plugin host, MCP bridge, transport, config
├── plugins/               built-in plugins shipped with mesh-llm
├── protocol/              control-plane protocol versions and conversions
├── runtime/               top-level startup flows and local runtime coordination
└── system/                hardware detection, benchmarking, self-update
```

Notable built-ins under `src/plugins/` today:

```text
plugins/
├── blackboard/            shared mesh message feed + MCP surface
├── blobstore/             request-scoped media object storage for multimodal
├── openai_endpoint/       external OpenAI-compatible inference endpoint bridge
└── telemetry/             opt-in OTLP metrics-only local runtime telemetry
```

## Runtime model

- `mesh-llm` owns the user-facing OpenAI-compatible API on `:9337`. Requests are routed by full model ref.
- The management API and web console live on `:3131`. Pass `--headless` to disable the embedded web UI while keeping the management API (`/api/*`) available on that port.
- Models that fit run as embedded single-stage skippy runtimes. Larger certified models can run as staged splits over the `skippy-stage/1` control plane.
- Stage topology is planned by mesh from peer/device inventory, package metadata, and reviewed family capability records. Replans withdraw old routes before publishing replacement stage-0 routes.
- Routing and demand tracking are mesh-wide. Nodes can serve different models at the same time.
- Discovery is optional and Nostr-backed. Private meshes work with explicit join tokens only.

The mesh control plane uses protocol `mesh-llm/1` with protobuf framing for mesh
traffic. Staged runtime traffic uses `skippy-stage/1`, keeping the new stage
protocol separate from mesh gossip compatibility.

## API surface

The management API exposes the state the UI uses directly:

- `GET /api/status` for node, peer, and routing state, including enriched `gpus[]` hardware entries with per-device VRAM, optional reserved bytes when the backend reports a true reserved/unavailable metric, memory bandwidth, and compute-throughput hints
- `GET /api/events` for live updates
- `GET /api/models` for mesh model inventory and `GET /api/runtime*` for loaded model/process state
- `GET /api/search` for read-only catalog or Hugging Face model search, returning the same JSON payload shape as `mesh-llm models search --json`
- `GET`/`POST`/`DELETE /api/model-interests` for local explicit-interest submission and readback using canonical model refs such as `org/repo@rev:variant`
- `GET /api/model-targets` for ranked model targets derived from explicit interest, active demand, and current serving visibility, with raw `signals` kept separate from derived `target_rank`/`wanted` hints
- `GET /api/discover` for mesh discovery results
- `GET /api/plugins` plus per-plugin tool endpoints
- `GET /api/blackboard/feed`, `GET /api/blackboard/search`, `POST /api/blackboard/post`

The OpenAI-compatible inference API remains on `http://localhost:9337/v1`, including `/v1/models`.

## Plugins and MCP

Plugin hosting now lives in `src/plugin/` rather than a crate-root module. mesh-llm supports:

- built-in plugins shipped with the binary
- external executable plugins declared in `~/.mesh-llm/config.toml`
- MCP exposure through the plugin bridge

`mesh-llm serve` also loads startup model config from the same file. The blackboard plugin is auto-registered unless explicitly disabled in config. Useful entry points:

```bash
mesh-llm plugin list
mesh-llm blackboard
mesh-llm blackboard --search "routing"
mesh-llm client --join <token> blackboard --mcp
```

Unified local config example:

```toml
version = 1

[gpu]
assignment = "pinned"

[[models]]
model = "unsloth/Qwen3-8B-GGUF:Q4_K_M"
gpu_id = "pci:0000:65:00.0"

[[models]]
model = "bartowski/Qwen2.5-VL-7B-Instruct-GGUF/qwen2.5-vl-7b-instruct-q4_k_m.gguf"
mmproj = "bartowski/Qwen2.5-VL-7B-Instruct-GGUF/mmproj-f16.gguf"
ctx_size = 8192
gpu_id = "uuid:GPU-12345678"

[[plugin]]
name = "blackboard"
enabled = true

[[plugin]]
name = "my-plugin"
command = "/absolute/path/to/plugin-binary"
args = ["--stdio"]
```

`mesh-llm serve` uses `~/.mesh-llm/config.toml` by default, or `--config /path/to/config.toml`.
Explicit `--model` or `--gguf` ignores configured `[[models]]`, and explicit `--ctx-size`
overrides configured `ctx_size` for the selected startup models.
Bare `mesh-llm serve` warns, shows help, and exits if `[[models]]` is empty.

When `[gpu].assignment = "pinned"`, every configured `[[models]]` entry must include a `gpu_id` taken from the pinnable stable IDs shown by `mesh-llm gpus` / `mesh-llm gpus --json`. Some fallback `stable_id` values may still be printed for inventory (`index:*`, backend-device names, and similar fallbacks), but those are not valid config identities. Pinned configs fail closed when an ID is missing, ambiguous, unsupported by the selected backend, or no longer resolves on the current host.

## Discovery and mesh modes

Opt-in Nostr discovery:

```bash
mesh-llm serve --model Qwen/Qwen2.5-3B-Instruct-GGUF:Q4_K_M --publish --mesh-name "Sydney Lab" --region AU
mesh-llm discover
mesh-llm discover --model GLM --region AU
mesh-llm serve --auto
mesh-llm gpus
```

Named meshes still work as a strict discovery filter:

```bash
mesh-llm serve --auto --model unsloth/GLM-4.7-Flash-GGUF:Q4_K_M --mesh-name "poker-night"
```

No-arg behavior remains intentionally simple:

```bash
mesh-llm
```

It prints `--help` and exits without binding the console or API ports.

## Development notes

- Build and test from the repo root with `just`; do not invoke ad-hoc build commands.
- Keep new code inside the owning domain module instead of adding new crate-root files.
- When changing protocol behavior, preserve compatibility unless a breaking change is explicitly intended.

## Live demo

**[Try it now](https://mesh-llm-console.fly.dev/)** — web console connected to the default public mesh. Runs as a client on Fly.io, no GPU.
