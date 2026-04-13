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
curl -fsSL https://raw.githubusercontent.com/Mesh-LLM/mesh-llm/main/install.sh | bash
```

Then start a node:

```bash
mesh-llm serve --auto
```

Inspect local GPU identity:

```bash
mesh-llm gpus
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
mesh-llm serve --auto
```

This is the easiest way to see the system working end to end.

### 2. Start a private mesh

```bash
mesh-llm serve --model Qwen2.5-32B
```

This starts serving a model, opens the local API and console, and prints an invite token for other machines.

### 3. Build from source

```bash
git clone https://github.com/Mesh-LLM/mesh-llm
cd mesh-llm
just build
```

Requires: `just`, `cmake`, Rust toolchain, Node.js 24 + npm. NVIDIA GPU builds need `nvcc` (CUDA toolkit). AMD GPU builds need ROCm/HIP. Vulkan GPU builds need the Vulkan development files plus `glslc`. CPU-only and Jetson/Tegra also work. For source builds, `just build` auto-detects CUDA vs ROCm vs Vulkan on Linux, or you can force `backend=rocm` or `backend=vulkan`. See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

Windows source builds are also supported for `cuda`, `rocm`/`hip`, `vulkan`, and `cpu` via `just build`. Metal remains macOS-only. Tagged GitHub releases currently publish macOS bundles plus Linux CPU, Linux ARM64 CPU, Linux CUDA, Linux ROCm, and Linux Vulkan bundles. The Linux ARM64 CPU artifact is `mesh-llm-aarch64-unknown-linux-gnu.tar.gz`. In install and release contexts, `arm64` and `aarch64` mean the same 64-bit ARM target, and generic 32-bit ARM is not a published release target. Windows publish jobs are currently commented out in `.github/workflows/release.yml`, but you can still generate the matching local Windows artifacts with `just release-build-windows`, `just release-build-cuda-windows`, `just release-build-rocm-windows`, `just release-build-vulkan-windows`, and the matching `release-bundle-*-windows` recipes.

## Run
Once installed, you can run:

```bash
mesh-llm serve --auto                      # join the best public mesh, start serving
```

That's it. Downloads a model for your hardware, connects to other nodes, and gives you an OpenAI-compatible API at `http://localhost:9337`.

Or start your own:
```bash
mesh-llm serve --model Qwen2.5-32B        # downloads model (~20GB), starts API + web console
mesh-llm serve --model Qwen2.5-3B         # or a small model first (~2GB)
```

Add another machine:
```bash
mesh-llm serve --join <token>              # token printed by the first machine
```

Or discover and join public meshes:
```bash
mesh-llm serve --auto                      # find and join the best mesh
mesh-llm client --auto                     # join as API-only client (no GPU)
```

## How it works

Every node gets an OpenAI-compatible API at `http://localhost:9337/v1`. Distribution is automatic — you just say `mesh-llm serve --model X` and the mesh figures out the best strategy:

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
mesh-llm serve --model Qwen2.5-32B
```
Starts serving a model and prints an invite token. This mesh is **private** — only people you share the token with can join.

To make it **public** (discoverable by others via `--auto`):
```bash
mesh-llm serve --model Qwen2.5-32B --publish
```

### Join a mesh
```bash
mesh-llm serve --join <token>              # join with invite token (GPU node)
mesh-llm client --join <token>             # join as API-only client (no GPU)
```

### Named mesh (buddy mode)
```bash
mesh-llm serve --auto --model GLM-4.7-Flash-Q4_K_M --mesh-name "poker-night"
```
Everyone runs the same command. First person creates it, everyone else discovers "poker-night" and joins automatically. `--mesh-name` implies `--publish` — named meshes are always published to the directory.

### Auto-discover
```bash
mesh-llm serve --auto                      # discover, join, and serve a model
mesh-llm client --auto                     # join as API-only client (no GPU)
mesh-llm discover                          # browse available meshes
mesh-llm gpus                              # inspect local GPUs and stable IDs
```

### Multi-model
```bash
mesh-llm serve --model Qwen2.5-32B --model GLM-4.7-Flash

# Route by model name
curl localhost:9337/v1/chat/completions -d '{"model":"GLM-4.7-Flash-Q4_K_M", ...}'
```
Different nodes serve different models. The API proxy routes by the `model` field.

### Inspect local GPUs
```bash
mesh-llm gpus
mesh-llm gpus --json
mesh-llm gpu benchmark --json
```

`mesh-llm gpus` prints local GPU entries, backend device names, stable IDs, VRAM, unified-memory state, and cached bandwidth when a benchmark fingerprint is already available. Add `--json` for machine-readable inventory output, or run `mesh-llm gpu benchmark --json` to refresh the local fingerprint and print the benchmark result as JSON.

Use only pinnable `Stable ID` / `stable_id` values from `mesh-llm gpus` or `mesh-llm gpus --json` for pinned startup config. Stable-ID fallback values such as `index:*` or backend-device names like `CUDA0` / `HIP0` / `MTL0` can still be printed for inventory purposes, but they are not valid pin targets.

### Startup config

`mesh-llm serve` can now load startup models from `~/.mesh-llm/config.toml`:

```toml
version = 1

[gpu]
assignment = "pinned"

[[models]]
model = "Qwen3-8B-Q4_K_M"
gpu_id = "pci:0000:65:00.0"

[[models]]
model = "bartowski/Qwen2.5-VL-7B-Instruct-GGUF/qwen2.5-vl-7b-instruct-q4_k_m.gguf"
mmproj = "bartowski/Qwen2.5-VL-7B-Instruct-GGUF/mmproj-f16.gguf"
ctx_size = 8192
gpu_id = "uuid:GPU-12345678"

[[plugin]]
name = "blackboard"
enabled = true
```

Start with the default config path:

```bash
mesh-llm serve
```

If no startup models are configured, `mesh-llm serve` prints a `⚠️` warning, shows help, and exits.

Or point at a different file:

```bash
mesh-llm serve --config /path/to/config.toml
```

Precedence rules:

- Explicit `--model` or `--gguf` ignores configured `[[models]]`.
- Explicit `--ctx-size` overrides configured `ctx_size` for the selected startup models.
- Plugin entries still live in the same file.

Pinned startup notes:

- `assignment = "pinned"` requires every configured `[[models]]` entry to include a `gpu_id`.
- Valid `gpu_id` values come from the pinnable stable IDs reported by `mesh-llm gpus` / `mesh-llm gpus --json`, not fallback inventory IDs.
- Pinned configs fail closed when a configured ID is missing, ambiguous, unsupported on the local backend, or no longer resolves on the current machine.
- Explicit `--model` / `--gguf` still bypass configured `[[models]]`, so they also bypass config-owned pinned `gpu_id` values.

### No-arg behavior
```bash
mesh-llm                                   # no args — prints --help and exits
```
Does not start the console or bind any ports. Use the CLI flags shown in `--help` to start or join a mesh.

## Background service

To install it as a per-user background service:

```bash
curl -fsSL https://raw.githubusercontent.com/Mesh-LLM/mesh-llm/main/install.sh | bash -s -- --service
```

Service installs are user-scoped:

- macOS installs a `launchd` agent at `~/Library/LaunchAgents/com.mesh-llm.mesh-llm.plist`
- Linux installs a `systemd --user` unit at `~/.config/systemd/user/mesh-llm.service`
- Shared environment config lives in `~/.config/mesh-llm/service.env`
- Startup models live in `~/.mesh-llm/config.toml`

The two platforms handle launch startup the same way:

- macOS: `launchd` runs `~/.config/mesh-llm/run-service.sh`, which loads `service.env` and executes `mesh-llm serve`.
- Linux: the installer writes `mesh-llm serve` directly into `ExecStart=` in `~/.config/systemd/user/mesh-llm.service`.

The background service no longer stores custom startup args. Configure startup models in `~/.mesh-llm/config.toml` instead.

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

## Web console

```bash
mesh-llm serve --model Qwen2.5-32B    # dashboard at http://localhost:3131
```

Live topology, per-node GPU capacity, model picker, and built-in chat. Everything comes from `/api/status` (JSON) and `/api/events` (SSE).

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
mesh-llm client --auto --port 9337
```

2. Check what models are available:
```bash
curl -s http://localhost:9337/v1/models | jq '.data[].id'
```

### Lemonade

mesh-llm ships a built-in `lemonade` plugin that registers a local [Lemonade Server](https://lemonade-server.ai) as another OpenAI-compatible backend. For setup and verification steps, see [docs/USAGE.md](docs/USAGE.md#lemonade-integration).

If you want the mesh to be discoverable via `--auto`, publish it:

```bash
mesh-llm serve --model Qwen2.5-32B --publish
```

### 3. Add another machine

```bash
mesh-llm serve --join <token>
```

Use `mesh-llm client` if the machine should join without serving a model:

```bash
mesh-llm client --join <token>
```

### 4. Create a named mesh for a group

```bash
mesh-llm serve --auto --model GLM-4.7-Flash-Q4_K_M --mesh-name "poker-night"
```

Everyone runs the same command. The first node creates the mesh, the rest discover and join it automatically.

### 5. Serve more than one model

```bash
mesh-llm serve --model Qwen2.5-32B --model GLM-4.7-Flash
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
curl -fsSL https://raw.githubusercontent.com/Mesh-LLM/mesh-llm/main/install.sh | MESH_LLM_INSTALL_FLAVOR=vulkan bash
```

Installed release bundles use flavor-specific llama.cpp binaries:

- macOS: `metal`
- Linux: `cpu`, `cuda`, `rocm`, `vulkan`
- Linux ARM64 CPU: `cpu` (asset triple: `aarch64-unknown-linux-gnu`)

For release and install naming, `arm64` and `aarch64` both refer to the same 64-bit ARM target. Generic 32-bit ARM is not a published release target.

To update a bundle install to the latest release:

```bash
mesh-llm update
```

To install a specific bundled release tag:

```bash
mesh-llm update --version v0.X.Y
```

If you build from source, always use `just`:

```bash
git clone https://github.com/Mesh-LLM/mesh-llm
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
