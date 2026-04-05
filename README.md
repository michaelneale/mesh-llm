# Mesh LLM

![Mesh LLM logo](docs/mesh-llm-logo.svg)

![Mesh LLM](mesh.png)

Mesh LLM lets you pool spare GPU capacity across machines and expose the result as one OpenAI-compatible API.

If a model fits on one machine, it runs there. If it does not, Mesh LLM automatically spreads the work across the mesh:

- Dense models use pipeline parallelism.
- MoE models use expert sharding with zero cross-node inference traffic.
- Every node gets the same local API at `http://localhost:9337/v1`.

## Why people use it

- Run models larger than a single machine can hold.
- Turn a few uneven boxes into one shared inference pool.
- Give agents a local OpenAI-compatible endpoint instead of wiring each tool by hand.
- Keep the setup simple: start one node, add more later.

## Quick start

Install the latest release:

```bash
curl -fsSL https://raw.githubusercontent.com/michaelneale/mesh-llm/main/install.sh | bash
```

Then start a node:

```bash
mesh-llm --auto
```

That command:

- picks a suitable bundled backend for your machine
- downloads a model if needed
- joins the best public mesh
- exposes an OpenAI-compatible API at `http://localhost:9337/v1`
- starts the web console at `http://localhost:3131`

Check what is available:

```bash
curl -s http://localhost:9337/v1/models | jq '.data[].id'
```

Send a request:

```bash
curl http://localhost:9337/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"GLM-4.7-Flash-Q4_K_M","messages":[{"role":"user","content":"hello"}]}'
```

## Common workflows

### 1. Try the public mesh

```bash
mesh-llm --auto
```

This is the easiest way to see the system working end to end.

### 2. Start a private mesh

```bash
mesh-llm --model Qwen2.5-32B
```

This starts serving a model, opens the local API and console, and prints an invite token for other machines.

### 3. Build from source

```bash
git clone https://github.com/michaelneale/mesh-llm
cd mesh-llm
just build
```

Requires: `just`, `cmake`, Rust toolchain, Node.js 24 + npm. NVIDIA GPU builds need `nvcc` (CUDA toolkit). AMD GPU builds need ROCm/HIP. Vulkan GPU builds need the Vulkan development files plus `glslc`. CPU-only and Jetson/Tegra also work. For source builds, `just build` auto-detects CUDA vs ROCm vs Vulkan on Linux, or you can force `backend=rocm` or `backend=vulkan`. See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

Windows source builds are also supported for `cuda`, `rocm`/`hip`, `vulkan`, and `cpu` via `just build`. Metal remains macOS-only. Tagged GitHub releases now publish Windows `.zip` bundles for `cpu`, `cuda`, `rocm`, and `vulkan`, and you can generate the same artifacts locally with `just release-build-windows`, `just release-build-cuda-windows`, `just release-build-amd-windows`, `just release-build-vulkan-windows`, and the matching `release-bundle-*-windows` recipes.

## Run
Once installed, you can run:

```bash
mesh-llm --auto                            # join the best public mesh, start serving
```

That's it. Downloads a model for your hardware, connects to other nodes, and gives you an OpenAI-compatible API at `http://localhost:9337`.

Or start your own:
```bash
mesh-llm --model Qwen2.5-32B              # downloads model (~20GB), starts API + web console
mesh-llm --model Qwen2.5-3B               # or a small model first (~2GB)
```

Add another machine:
```bash
mesh-llm --join <token>                    # token printed by the first machine
```

Or discover and join public meshes:
```bash
mesh-llm --auto                            # find and join the best mesh
mesh-llm --client --auto                   # join as API-only client (no GPU)
```

## How it works

Every node gets an OpenAI-compatible API at `http://localhost:9337/v1`. Distribution is automatic — you just say `mesh-llm --model X` and the mesh figures out the best strategy:

- **Model fits on one machine?** → runs solo, full speed, no network overhead
- **Dense model too big?** → pipeline parallelism — layers split across nodes
- **MoE model too big?** → expert parallelism — experts split across nodes, zero cross-node traffic

If a node has enough VRAM, it always runs the full model. Splitting only happens when it has to.
Currently using a lightly forked version of llama.cpp (see the Justfile for where it pulls branch from).

**Pipeline parallelism** — for dense models that don't fit on one machine, layers are distributed across nodes proportional to VRAM. llama-server runs on the highest-VRAM node and coordinates via RPC. Each rpc-server loads only its assigned layers from local disk. Latency-aware: peers are selected by lowest RTT first, with an 80ms hard cap — high-latency nodes stay in the mesh as API clients but don't participate in splits.

**MoE expert parallelism** — Mixture-of-Experts models (Qwen3-MoE, GLM, OLMoE, Mixtral, DeepSeek — increasingly the best-performing architectures) are auto-detected from the GGUF header. The mesh reads expert routing statistics to identify which experts matter most, then assigns each node an overlapping shard: a shared core of critical experts replicated everywhere, plus unique experts distributed across nodes. Each node gets a standalone GGUF with the full trunk + its expert subset and runs its own independent llama-server — zero cross-node traffic during inference. Sessions are hash-routed to nodes for KV cache locality.

**Multi-model** — different nodes serve different models simultaneously. The API proxy peeks at the `model` field in each request and routes to the right node via QUIC tunnel. `/v1/models` lists everything available.

**Demand-aware rebalancing** — a unified demand map tracks which models the mesh wants (from `--model` flags, API requests, and gossip). Demand signals propagate infectiously across all nodes and decay naturally via TTL. Standby nodes auto-promote to serve unserved models with active demand, or rebalance when one model is significantly hotter than others. When a model loses its last server, standby nodes detect it within ~60s.

**Latency design** — the key insight is that HTTP streaming is latency-tolerant while RPC is latency-multiplied. llama-server always runs on the same box as the GPU. The mesh tunnels HTTP, so cross-network latency only affects time-to-first-token, not per-token throughput. RPC only crosses the network for pipeline splits where the model physically doesn't fit on one machine.

### Network optimizations

- **Zero-transfer GGUF loading** — `SET_TENSOR_GGUF` tells rpc-server to read weights from local disk. Dropped model load from 111s → 5s.
- **RPC round-trip reduction** — cached `get_alloc_size`, skip GGUF lookups for intermediates. Per-token round-trips: 558 → 8.
- **Direct server-to-server transfers** — intermediate tensors pushed directly between rpc-servers via TCP, not relayed through the client.
- **Speculative decoding** — draft model runs locally on the host, proposes tokens verified in one batched forward pass. +38% throughput on code (75% acceptance).

## Usage

### Start a mesh
```bash
mesh-llm --model Qwen2.5-32B
```
Starts serving a model and prints an invite token. This mesh is **private** — only people you share the token with can join.

To make it **public** (discoverable by others via `--auto`):
```bash
mesh-llm --model Qwen2.5-32B --publish
```

### Join a mesh
```bash
mesh-llm --join <token>                    # join with invite token (GPU node)
mesh-llm --client --join <token>           # join as API-only client (no GPU)
```

### Named mesh (buddy mode)
```bash
mesh-llm --auto --model GLM-4.7-Flash-Q4_K_M --mesh-name "poker-night"
```
Everyone runs the same command. First person creates it, everyone else discovers "poker-night" and joins automatically. `--mesh-name` implies `--publish` — named meshes are always published to the directory.

### Auto-discover
```bash
mesh-llm --auto                            # discover, join, and serve a model
mesh-llm --client --auto                   # join as API-only client (no GPU)
mesh-llm discover                          # browse available meshes
```

### Multi-model
```bash
mesh-llm --model Qwen2.5-32B --model GLM-4.7-Flash

# Route by model name
curl localhost:9337/v1/chat/completions -d '{"model":"GLM-4.7-Flash-Q4_K_M", ...}'
```
Different nodes serve different models. The API proxy routes by the `model` field.

### No-arg behavior
```bash
mesh-llm                                   # no args — prints --help and exits
```
Does not start the console or bind any ports. Use the CLI flags shown in `--help` to start or join a mesh.

## Background service

To install it as a per-user background service:

```bash
curl -fsSL https://raw.githubusercontent.com/michaelneale/mesh-llm/main/install.sh | bash -s -- --service
```

To seed the service with a custom startup command on first install:

```bash
curl -fsSL https://raw.githubusercontent.com/michaelneale/mesh-llm/main/install.sh | bash -s -- --service --service-args '--model Qwen2.5-3B'
```

Service installs are user-scoped:

- macOS installs a `launchd` agent at `~/Library/LaunchAgents/com.mesh-llm.mesh-llm.plist`
- Linux installs a `systemd --user` unit at `~/.config/systemd/user/mesh-llm.service`
- Shared environment config lives in `~/.config/mesh-llm/service.env`

The two platforms handle launch args differently:

- macOS: `launchd` runs `~/.config/mesh-llm/run-service.sh`, which reads `~/.config/mesh-llm/service.args`. `service.args` is one `mesh-llm` CLI argument per line. The installer creates it with `--auto` by default and preserves your edits on reinstall unless you pass `--service-args` again.
- Linux: the installer writes the `mesh-llm` argv directly into `ExecStart=` in `~/.config/systemd/user/mesh-llm.service`. If you pass `--service-args`, those replace the current unit args; otherwise the installer preserves the existing unit args on reinstall.

`service.env` is optional and shared by both platforms. Use plain `KEY=value` lines, for example:

```text
MESH_LLM_NO_SELF_UPDATE=1
```

If you edit the Linux unit manually, reload and restart it:

```bash
systemctl --user daemon-reload
systemctl --user restart mesh-llm.service
```

On Linux this is a user service, so if you want it to keep running after reboot before login, enable lingering once:

```bash
sudo loginctl enable-linger "$USER"
```

## Mesh config

An optional TOML file lets you record per-node assignments for your mesh. The default path is `~/.mesh-llm/mesh.toml`. If the file does not exist, mesh-llm starts with an empty default config and no error.

Override the path with the `--mesh-config <path>` flag or the `MESH_LLM_MESH_CONFIG` environment variable:

```bash
mesh-llm --auto --mesh-config /etc/mesh/my-mesh.toml
MESH_LLM_MESH_CONFIG=/etc/mesh/my-mesh.toml mesh-llm --auto
```

A minimal config looks like this:

```toml
version = 1

[[nodes]]
node_id = "abc12345"  # use the short node_id shown by /api/status

[[nodes.models]]
name = "Qwen2.5-7B"

[[nodes]]
node_id = "def67890"

[[nodes.models]]
name = "GLM-4.7-Flash"
ctx_size = 4096
```

Each entry in the `nodes` array has a `node_id` field. Use the same short node ID string the running system exposes in `/api/status` and the UI, not an arbitrary nickname. Per-node fields include an optional `hostname` and an optional `placement_mode` (`"pooled"` or `"separate"`). Each model in the `models` array is identified by its `name`. Optional per-model fields include `ctx_size`, `gpu_index`, `model_key`, `path`, `moe_experts`, and `split` (`{ start, end, total }` absolute layer range). The `split` field is persisted in the file but not acted on at runtime.

Config is loaded and projected at startup as temporary inert scaffolding for future mesh-config activation. Today it is used for validation and local-node projection only; it does not yet auto-apply models, change election, alter routing, launch processes, or retain a long-lived runtime config state beyond startup. This temporary behavior will be expanded in a later change, so the startup wiring should remain in place.

## Web console

```bash
mesh-llm --model Qwen2.5-32B    # dashboard at http://localhost:3131
```

Live topology, VRAM bars per node, model picker, built-in chat. Everything comes from `/api/status` (JSON) and `/api/events` (SSE).

## Multimodal Support

mesh-llm supports multimodal requests on:

- `POST /v1/chat/completions`
- `POST /v1/responses`

The console supports image, audio, and file attachments. Large attachments use request-scoped blob upload rather than permanent storage.

### Current support matrix

| Family / model type | Vision | Audio | Notes |
|---|---|---|---|
| `Qwen3-VL`, `Qwen3VL` | yes | no | Example: `Qwen3VL-2B-Instruct-Q4_K_M` |
| `Qwen2-VL`, `Qwen2.5-VL` | yes | no | Vision-capable Qwen VL families |
| `LLaVA`, `mllama`, `PaliGemma`, `Idefics`, `Molmo`, `InternVL`, `GLM-4V`, `Ovis`, `Florence` | yes | no | Detected as vision-capable families |
| `Qwen2-Audio` | no | yes | Audio-capable family |
| `SeaLLM-Audio` | no | yes | Audio-capable family |
| `Ultravox` | no | yes | Audio-capable family |
| `Omni` | no or metadata-dependent | yes | Example: `Qwen2.5-Omni-3B-Q4_K_M` |
| `Whisper` | no | yes | Audio-capable family |
| Any GGUF with `mmproj` sidecar | yes | depends | Strong local signal for vision support |
| Any model with `vision_config` / vision token IDs | yes | depends | Promoted by metadata |
| Any model with `audio_config` / audio token IDs | depends | yes | Promoted by metadata |
| Generic `multimodal`, `-vl`, `image`, `video`, `voice` naming only | likely | likely | Hint only, not a strong routing guarantee |

Notes:

- `yes` means mesh-llm treats the model as runtime-capable for routing and UI.
- `likely` means mesh-llm shows a weaker hint but does not rely on it as a hard capability.
- Mixed image+audio requests work only when the selected model/runtime actually supports both modalities.
- Non-goals: `POST /v1/audio/transcriptions`, `POST /v1/audio/speech`, and `v1/realtime`.

For the full capability and transport details, see [mesh-llm/docs/MULTI_MODAL.md](mesh-llm/docs/MULTI_MODAL.md).

### Development

Build-from-source and UI development instructions are in [CONTRIBUTING.md](CONTRIBUTING.md).

## Using with agents

mesh-llm exposes an OpenAI-compatible API on `localhost:9337`. Any tool that supports custom OpenAI endpoints works. `/v1/models` lists available models; the `model` field in requests routes to the right node.

For built-in launcher integrations (`goose`, `claude`):

- If a mesh is already running locally on `--port`, it is reused.
- If not, `mesh-llm` auto-starts a background client node that auto-joins the mesh.
- If `--model` is omitted, the launcher picks the strongest tool-capable model available on the mesh.
- When the harness exits (e.g. `claude` quits), the auto-started node is cleaned up automatically.

### goose

[Goose](https://github.com/block/goose) is available as both CLI (`goose session`) and desktop app (Goose.app).

```bash
mesh-llm goose
```

Use a specific model (example: MiniMax):

```bash
mesh-llm goose --model MiniMax-M2.5-Q4_K_M
```

This command writes/updates `~/.config/goose/custom_providers/mesh.json` and launches Goose.

### pi

1. Start a mesh client:
```bash
mesh-llm --client --auto --port 9337
```

2. Check what models are available:
```bash
curl -s http://localhost:9337/v1/models | jq '.data[].id'
```

If you want the mesh to be discoverable via `--auto`, publish it:

```bash
mesh-llm --model Qwen2.5-32B --publish
```

### 3. Add another machine

```bash
mesh-llm --join <token>
```

Use `--client` if the machine should join without serving a model:

```bash
mesh-llm --client --join <token>
```

### 4. Create a named mesh for a group

```bash
mesh-llm --auto --model GLM-4.7-Flash-Q4_K_M --mesh-name "poker-night"
```

Everyone runs the same command. The first node creates the mesh, the rest discover and join it automatically.

### 5. Serve more than one model

```bash
mesh-llm --model Qwen2.5-32B --model GLM-4.7-Flash
```

Requests are routed by the `model` field:

```bash
curl localhost:9337/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"GLM-4.7-Flash-Q4_K_M","messages":[{"role":"user","content":"hello"}]}'
```

## How it works

Mesh LLM keeps the user-facing surface simple: talk to `localhost:9337`, pick a model, and let the mesh decide how to serve it.

- If a model fits on one machine, it runs there with no network overhead.
- If a dense model does not fit, layers are split across low-latency peers.
- If an MoE model does not fit, experts are split across nodes and requests are hash-routed for cache locality.
- Different nodes can serve different models at the same time.

Each node also exposes a management API and web console on port `3131`.

## Install notes

The installer currently targets macOS and Linux release bundles. Windows coming soon.

To force a specific bundled flavor during install:

```bash
curl -fsSL https://raw.githubusercontent.com/michaelneale/mesh-llm/main/install.sh | MESH_LLM_INSTALL_FLAVOR=vulkan bash
```

Installed release bundles use flavor-specific llama.cpp binaries:

- macOS: `metal`
- Linux: `cpu`, `cuda`, `rocm`, `vulkan`

To update a bundle install to the latest release:

```bash
mesh-llm update
```

If you build from source, always use `just`:

```bash
git clone https://github.com/michaelneale/mesh-llm
cd mesh-llm
just build
```

Requirements and backend-specific build notes are in [CONTRIBUTING.md](CONTRIBUTING.md).

## Web console

When a node is running, open:

```text
http://localhost:3131
```

The console shows live topology, VRAM usage, loaded models, and built-in chat. It is backed by `/api/status` and `/api/events`.

You can also try the hosted demo:

**[mesh-llm-console.fly.dev](https://mesh-llm-console.fly.dev/)**

## More docs

- [docs/USAGE.md](docs/USAGE.md) for service installs, model commands, storage, and runtime control
- [docs/AGENTS.md](docs/AGENTS.md) for Goose, Claude Code, pi, OpenCode, curl, and blackboard usage
- [docs/BENCHMARKS.md](docs/BENCHMARKS.md) for benchmark numbers and context
- [CONTRIBUTING.md](CONTRIBUTING.md) for local development and build workflows
- [PLUGINS.md](PLUGINS.md) for the plugin system and blackboard internals
- [mesh-llm/README.md](mesh-llm/README.md) for Rust crate structure
- [ROADMAP.md](ROADMAP.md) for future work

## Community

Join the [#mesh-llm channel on the Goose Discord](https://discord.gg/goose-oss) for discussion and support.
