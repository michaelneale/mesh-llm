# Mesh LLM

![Mesh LLM logo](docs/mesh-llm-logo.svg)

![Mesh LLM](mesh.png)

Mesh LLM lets you pool spare GPU capacity across machines and expose the result as one OpenAI-compatible API.

If a model fits on one machine, it runs there. If it does not, Mesh LLM automatically spreads the work across the mesh:

- Dense models use pipeline parallelism.
- MoE models use expert sharding with zero cross-node inference traffic.
- Every node gets the same local API at `http://localhost:9337/v1`.

## Principles

These guide every design decision:

- **One command to run.** `mesh-llm serve --auto` should be all you need. No config files, no coordination, no prerequisites beyond the binary.
- **Batteries included.** Models download automatically. Backends are bundled. The web console ships inside the binary. Nothing extra to install.
- **Sensible defaults.** Solo when the model fits. Split only when it has to. Draft models auto-paired. Context sized to VRAM. Thinking off when it hurts more than it helps.
- **Always compatible in the mesh.** Older and newer nodes must coexist. Protocol negotiation (`mesh-llm/0` ↔ `mesh-llm/1`) keeps mixed meshes working. Rolling upgrades, not flag days.
- **Public and private.** `--auto` joins public meshes for experimentation. `--join <token>` creates private meshes for production. Same binary, same API, same behavior.
- **Support as many platforms as possible.** macOS Metal, Linux CUDA, Linux ROCm, Linux Vulkan, Linux CPU, Jetson/Tegra, Windows. If it has compute, it should be able to join.

## Quick start

Install the latest release:

```bash
curl -fsSL https://raw.githubusercontent.com/michaelneale/mesh-llm/main/install.sh | bash
```

Then start a node:

```bash
mesh-llm serve --auto
```

That command:

- picks a suitable bundled backend for your machine
- downloads a model if needed
- joins the best public mesh
- exposes an OpenAI-compatible API at `http://localhost:9337/v1`
- starts the web console at `http://localhost:3131`

Inspect local GPUs:

```bash
mesh-llm gpus
```

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

Starts serving a model, opens the local API and console, and prints an invite token for other machines.

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

### 6. Publish for auto-discovery

```bash
mesh-llm serve --model Qwen2.5-32B --publish
```

## How it works

Mesh LLM keeps the user-facing surface simple: talk to `localhost:9337`, pick a model, and let the mesh decide how to serve it.

- If a model fits on one machine, it runs there with no network overhead.
- If a dense model does not fit, layers are split across low-latency peers.
- If an MoE model does not fit, experts are split across nodes and requests are hash-routed for cache locality.
- Different nodes can serve different models at the same time.

Each node also exposes a management API and web console on port `3131`.

Currently using a lightly forked version of llama.cpp (see the Justfile for where it pulls the branch from).

**Pipeline parallelism** — for dense models that don't fit on one machine, layers are distributed across nodes proportional to VRAM. llama-server runs on the highest-VRAM node and coordinates via RPC. Each rpc-server loads only its assigned layers from local disk. Latency-aware: peers are selected by lowest RTT first, with an 80ms hard cap — high-latency nodes stay in the mesh as API clients but don't participate in splits.

**MoE expert parallelism** — Mixture-of-Experts models (Qwen3-MoE, GLM, OLMoE, Mixtral, DeepSeek — increasingly the best-performing architectures) are auto-detected from the GGUF header. The mesh reads expert routing statistics to identify which experts matter most, then assigns each node an overlapping shard: a shared core of critical experts replicated everywhere, plus unique experts distributed across nodes. Each node gets a standalone GGUF with the full trunk + its expert subset and runs its own independent llama-server — zero cross-node traffic during inference. Sessions are hash-routed to nodes for KV cache locality.

**Multi-model** — different nodes serve different models simultaneously. The API proxy peeks at the `model` field in each request and routes to the right node via QUIC tunnel. `/v1/models` lists everything available.

**Demand-aware rebalancing** — a unified demand map tracks which models the mesh wants (from `--model` flags, API requests, and gossip). Demand signals propagate infectiously across all nodes and decay naturally via TTL. Standby nodes auto-promote to serve unserved models with active demand, or rebalance when one model is significantly hotter than others.

**Latency design** — HTTP streaming is latency-tolerant while RPC is latency-multiplied. llama-server always runs on the same box as the GPU. The mesh tunnels HTTP, so cross-network latency only affects time-to-first-token, not per-token throughput. RPC only crosses the network for pipeline splits where the model physically doesn't fit on one machine.

### Network optimizations

- **Zero-transfer GGUF loading** — `SET_TENSOR_GGUF` tells rpc-server to read weights from local disk. Dropped model load from 111s → 5s.
- **RPC round-trip reduction** — cached `get_alloc_size`, skip GGUF lookups for intermediates. Per-token round-trips: 558 → 8.
- **Direct server-to-server transfers** — intermediate tensors pushed directly between rpc-servers via TCP, not relayed through the client.
- **Speculative decoding** — draft model runs locally on the host, proposes tokens verified in one batched forward pass. +38% throughput on code (75% acceptance).

## Benchmarks

GLM-4.7-Flash-Q4_K_M (17GB), tested on an M4 Max and a Mac mini M4 over Wi-Fi:

| Configuration | tok/s |
|---|---|
| Solo (no mesh) | 68 |
| 2-node split (85/15) | 21 |
| 3-node split (62/31/8) | 12-13 |

Cross-network from Sydney to Queensland at roughly 20ms RTT measured 10-25 tok/s. In those runs, the overhead was dominated by per-token RPC latency.

Stock llama.cpp RPC transfers about 16.88GB on connect. This fork uses local GGUF loading on peers, which cuts that to 0 bytes transferred in about 9 seconds.

## Install

The installer currently targets macOS and Linux release bundles.

```bash
curl -fsSL https://raw.githubusercontent.com/michaelneale/mesh-llm/main/install.sh | bash
```

The installer probes your machine, recommends a flavor, and asks what to install.

To force a specific bundled flavor:

```bash
curl -fsSL https://raw.githubusercontent.com/michaelneale/mesh-llm/main/install.sh | MESH_LLM_INSTALL_FLAVOR=vulkan bash
```

Installed release bundles use flavor-specific llama.cpp binaries:

- macOS: `metal`
- Linux: `cpu`, `cuda`, `rocm`, `vulkan`

To update:

```bash
mesh-llm update
```

### Build from source

```bash
git clone https://github.com/michaelneale/mesh-llm
cd mesh-llm
just build
```

Requires: `just`, `cmake`, Rust toolchain, Node.js 24 + npm. NVIDIA GPU builds need `nvcc` (CUDA toolkit). AMD GPU builds need ROCm/HIP. Vulkan GPU builds need the Vulkan development files plus `glslc`. CPU-only and Jetson/Tegra also work.

For source builds, `just build` auto-detects CUDA vs ROCm vs Vulkan on Linux, or you can force `backend=rocm` or `backend=vulkan`.

Windows source builds are also supported. Tagged GitHub releases publish Windows `.zip` bundles for `cpu`, `cuda`, `rocm`, and `vulkan`.

See [HUMANS.md](HUMANS.md) for full build details and development workflow.

## Usage reference

### Specifying models

`mesh-llm serve --model` accepts several formats. Hugging Face-backed models are cached on first use.

```bash
mesh-llm serve --model Qwen3-8B
mesh-llm serve --model Qwen3-8B-Q4_K_M
mesh-llm serve --model https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf
mesh-llm serve --model bartowski/Llama-3.2-3B-Instruct-GGUF/Llama-3.2-3B-Instruct-Q4_K_M.gguf
mesh-llm serve --gguf ~/my-models/custom-model.gguf
mesh-llm serve --gguf ~/my-models/qwen3.5-4b.gguf --mmproj ~/my-models/mmproj-BF16.gguf
```

### Model catalog

```bash
mesh-llm download
mesh-llm download 32b
mesh-llm download 72b --draft
```

Draft pairings for speculative decoding:

| Model | Size | Draft | Draft size |
|---|---|---|---|
| Qwen2.5 (3B/7B/14B/32B/72B) | 2-47GB | Qwen2.5-0.5B | 491MB |
| Qwen3-32B | 20GB | Qwen3-0.6B | 397MB |
| Llama-3.3-70B | 43GB | Llama-3.2-1B | 760MB |
| Gemma-3-27B | 17GB | Gemma-3-1B | 780MB |

### Model commands

```bash
mesh-llm models recommended
mesh-llm models installed
mesh-llm models search qwen 8b
mesh-llm models search --catalog qwen
mesh-llm models show Qwen/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf
mesh-llm models download Qwen/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf
mesh-llm models updates --check
mesh-llm models updates --all
mesh-llm models updates Qwen/Qwen3-8B-GGUF
```

### Startup config

`mesh-llm serve` loads startup models from `~/.mesh-llm/config.toml`:

```toml
version = 1

[gpu]
assignment = "auto"

[[models]]
model = "Qwen3-8B-Q4_K_M"

[[models]]
model = "bartowski/Qwen2.5-VL-7B-Instruct-GGUF/qwen2.5-vl-7b-instruct-q4_k_m.gguf"
mmproj = "bartowski/Qwen2.5-VL-7B-Instruct-GGUF/mmproj-f16.gguf"
ctx_size = 8192

[[plugin]]
name = "blackboard"
enabled = true
```

```bash
mesh-llm serve                             # uses default config
mesh-llm serve --config /path/to/config.toml
```

Precedence:

- Explicit `--model` or `--gguf` ignores configured `[[models]]`.
- Explicit `--ctx-size` overrides configured `ctx_size`.
- Plugin entries stay in the same file.
- If `[[models]]` is empty, `mesh-llm serve` prints a warning and exits.

### No-arg behavior

```bash
mesh-llm                                   # prints --help and exits
```

Does not start the console or bind any ports.

### Local runtime control

```bash
mesh-llm load Llama-3.2-1B-Instruct-Q4_K_M
mesh-llm unload Llama-3.2-1B-Instruct-Q4_K_M
mesh-llm status
```

Management API:

```bash
curl localhost:3131/api/runtime
curl localhost:3131/api/runtime/processes
curl -X POST localhost:3131/api/runtime/models \
  -H 'Content-Type: application/json' \
  -d '{"model":"Llama-3.2-1B-Instruct-Q4_K_M"}'
curl -X DELETE localhost:3131/api/runtime/models/Llama-3.2-1B-Instruct-Q4_K_M
```

### Model storage

- Hugging Face repo snapshots are the canonical managed model store.
- Arbitrary local GGUF files work through `mesh-llm serve --gguf`.
- MoE split artifacts are cached under `~/.cache/mesh-llm/splits/`.

## Background service

Install as a per-user background service:

```bash
curl -fsSL https://raw.githubusercontent.com/michaelneale/mesh-llm/main/install.sh | bash -s -- --service
```

- macOS: `launchd` agent at `~/Library/LaunchAgents/com.mesh-llm.mesh-llm.plist`
- Linux: `systemd --user` unit at `~/.config/systemd/user/mesh-llm.service`
- Shared environment config: `~/.config/mesh-llm/service.env`
- Startup models: `~/.mesh-llm/config.toml`

On Linux, enable lingering for reboot persistence:

```bash
sudo loginctl enable-linger "$USER"
```

## Web console

```bash
mesh-llm serve --model Qwen2.5-32B    # dashboard at http://localhost:3131
```

Live topology, VRAM bars per node, model picker, built-in chat. Everything comes from `/api/status` (JSON) and `/api/events` (SSE).

Try the hosted demo: **[mesh-llm-console.fly.dev](https://mesh-llm-console.fly.dev/)**

## Multimodal support

mesh-llm supports multimodal requests on `/v1/chat/completions` and `/v1/responses`. The console supports image, audio, and file attachments.

| Family | Vision | Audio |
|---|---|---|
| Qwen3-VL, Qwen2.5-VL | yes | no |
| LLaVA, mllama, PaliGemma, Molmo, InternVL, GLM-4V | yes | no |
| Qwen2-Audio, Ultravox, Whisper | no | yes |
| Qwen2.5-Omni | metadata-dependent | yes |
| Any GGUF with `mmproj` sidecar | yes | depends |

For full details, see [mesh-llm/docs/MULTI_MODAL.md](mesh-llm/docs/MULTI_MODAL.md).

## Using with agents

mesh-llm exposes an OpenAI-compatible API on `localhost:9337`. Any tool that supports custom OpenAI endpoints works. `/v1/models` lists available models; the `model` field routes to the right node.

### Goose

```bash
mesh-llm goose
mesh-llm goose --model MiniMax-M2.5-Q4_K_M
```

Writes/updates `~/.config/goose/custom_providers/mesh.json` and launches [Goose](https://github.com/block/goose).

### Claude Code

```bash
mesh-llm claude
mesh-llm claude --model MiniMax-M2.5-Q4_K_M
```

### pi

```bash
mesh-llm client --auto --port 9337
curl -s http://localhost:9337/v1/models | jq '.data[].id'
pi --model mesh/MiniMax-M2.5-Q4_K_M
```

### OpenCode

```bash
OPENAI_API_KEY=dummy OPENAI_BASE_URL=http://localhost:9337/v1 opencode -m openai/GLM-4.7-Flash-Q4_K_M
```

### curl or any OpenAI client

```bash
curl http://localhost:9337/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"GLM-4.7-Flash-Q4_K_M","messages":[{"role":"user","content":"hello"}]}'
```

### Launcher auto-start

For built-in launcher commands (`goose`, `claude`):

- If a mesh is already running locally, it is reused.
- Otherwise mesh-llm starts a background client node and auto-joins.
- If `--model` is omitted, the launcher picks the strongest tool-capable model.
- When the harness exits, the auto-started node is cleaned up.

### Lemonade

mesh-llm ships a built-in `lemonade` plugin that registers a local [Lemonade Server](https://lemonade-server.ai) as another OpenAI-compatible backend.

Enable in `~/.mesh-llm/config.toml`:

```toml
[[plugin]]
name = "lemonade"
enabled = true
```

Set a custom endpoint if needed:

```bash
export MESH_LLM_LEMONADE_BASE_URL=http://127.0.0.1:8000/api/v1
```

### Blackboard

Share status, findings, and questions across the mesh:

```bash
mesh-llm blackboard "STATUS: [org/repo branch:main] refactoring billing module"
mesh-llm blackboard --search "billing refactor"
mesh-llm blackboard install-skill        # install agent skill
mesh-llm blackboard --mcp                # run as MCP server
```

## Plugins

mesh-llm has a plugin architecture for extending functionality. Bundled plugins (blackboard, lemonade) are auto-registered. External plugins can be declared in config:

```toml
[[plugin]]
name = "my-plugin"
command = "/path/to/plugin-binary"
args = ["--stdio"]
```

See [HUMANS.md](HUMANS.md#plugins) for the plugin overview, and [mesh-llm/docs/PLUGINS.md](mesh-llm/docs/PLUGINS.md) for the full architecture spec.

## More docs

- [HUMANS.md](HUMANS.md) — architecture, development, testing, release process, roadmap, plugins
- [AGENTS.md](AGENTS.md) — instructions for AI coding agents working on this repo

## Community

Join the [#mesh-llm channel on the Goose Discord](https://discord.gg/goose-oss) for discussion and support.
