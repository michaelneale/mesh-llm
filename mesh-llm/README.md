# mesh-llm crate

Rust implementation of mesh-llm: a peer-to-peer control plane for llama.cpp inference over QUIC, with distributed routing, model orchestration, plugin hosting, and a local management API.

For install and end-user usage, see the [project README](../README.md). For deeper architecture and test flows, see [docs/DESIGN.md](docs/DESIGN.md), [docs/TESTING.md](docs/TESTING.md), and [docs/message_protocol.md](docs/message_protocol.md).

## Source layout

The crate root stays intentionally small:

```text
src/
├── lib.rs                 crate entrypoint, module wiring, version, public re-exports
├── main.rs                binary entrypoint
├── api/                   management API, status shaping, HTTP routing
├── cli/                   clap types, subcommands, command handlers
├── inference/             election, launch, pipeline splits, MoE orchestration
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
└── example/               standalone example plugin crate
```

## Runtime model

- `mesh-llm` owns the user-facing OpenAI-compatible API on `:9337`. Requests are routed by model.
- The management API and web console live on `:3131`.
- Dense models that fit run locally. Dense models that do not fit can be split with pipeline parallelism.
- MoE models are handled through expert-aware orchestration in `inference/moe.rs`.
- Routing and demand tracking are mesh-wide. Nodes can serve different models at the same time.
- Discovery is optional and Nostr-backed. Private meshes work with explicit join tokens only.

The current control plane prefers protocol `mesh-llm/1` with protobuf framing, while keeping backward-compatible support for older `mesh-llm/0` peers in `src/protocol/`.

## API surface

The management API exposes the state the UI uses directly:

- `GET /api/status` for node, peer, and routing state
- `GET /api/events` for live updates
- `GET /api/models` and runtime endpoints for loaded model/process state
- `GET /api/discover` for mesh discovery results
- `GET /api/plugins` plus per-plugin tool endpoints
- `GET /api/blackboard/feed`, `GET /api/blackboard/search`, `POST /api/blackboard/post`

The OpenAI-compatible inference API remains on `http://localhost:9337/v1`, including `/v1/models`.

## Plugins and MCP

Plugin hosting now lives in `src/plugin/` rather than a crate-root module. mesh-llm supports:

- built-in plugins shipped with the binary
- external executable plugins declared in `~/.mesh-llm/config.toml`
- MCP exposure through the plugin bridge

The blackboard plugin is auto-registered unless explicitly disabled in config. Useful entry points:

```bash
mesh-llm plugin list
mesh-llm blackboard
mesh-llm blackboard --search "routing"
mesh-llm --client --join <token> blackboard --mcp
```

External plugins are configured as executables, for example:

```toml
[[plugin]]
name = "my-plugin"
command = "/absolute/path/to/plugin-binary"
args = ["--stdio"]
```

## Discovery and mesh modes

Opt-in Nostr discovery:

```bash
mesh-llm --model Qwen2.5-3B --publish --mesh-name "Sydney Lab" --region AU
mesh-llm discover
mesh-llm discover --model GLM --region AU
mesh-llm --auto
```

Named meshes still work as a strict discovery filter:

```bash
mesh-llm --auto --model GLM-4.7-Flash-Q4_K_M --mesh-name "poker-night"
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
