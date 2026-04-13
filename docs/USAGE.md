# Usage Guide

This page keeps the longer operational reference out of the top-level README.

For command-by-command CLI usage, model resolution rules, and JSON automation examples, see [CLI.md](./CLI.md).

## Installation details

Install the latest release bundle:

```bash
curl -fsSL https://raw.githubusercontent.com/Mesh-LLM/mesh-llm/main/install.sh | bash
```

To opt into the latest published prerelease bundle instead:

```bash
curl -fsSL https://raw.githubusercontent.com/Mesh-LLM/mesh-llm/main/install.sh | bash -s -- --pre-release
```

The installer probes your machine, recommends a flavor, and asks what to install.

For a non-interactive install, set the flavor explicitly:

```bash
curl -fsSL https://raw.githubusercontent.com/Mesh-LLM/mesh-llm/main/install.sh | MESH_LLM_INSTALL_FLAVOR=vulkan bash
```

Release bundles install flavor-specific llama.cpp binaries:

- macOS: `rpc-server-metal`, `llama-server-metal`
- Linux CPU: `rpc-server-cpu`, `llama-server-cpu`
- Linux CUDA: `rpc-server-cuda`, `llama-server-cuda`
- Linux ROCm: `rpc-server-rocm`, `llama-server-rocm`
- Linux Vulkan: `rpc-server-vulkan`, `llama-server-vulkan`

If you keep more than one flavor in the same `bin` directory, choose one explicitly:

```bash
mesh-llm serve --llama-flavor vulkan --model Qwen2.5-32B
```

Source builds must use `just`:

```bash
git clone https://github.com/Mesh-LLM/mesh-llm
cd mesh-llm
just build
```

Requirements:

- `just`
- `cmake`
- Rust toolchain
- Node.js 24 + npm

Backend-specific notes:

- NVIDIA builds require `nvcc`
- AMD builds require ROCm/HIP
- Vulkan builds require the Vulkan development files and `glslc`
- CPU-only and Jetson/Tegra are also supported

For full build details, see [CONTRIBUTING.md](../CONTRIBUTING.md).

## Common commands

```bash
mesh-llm serve --auto
mesh-llm serve --model Qwen2.5-32B
mesh-llm serve --join <token>
mesh-llm client --auto
mesh-llm gpus
mesh-llm discover
```

If you run `mesh-llm` with no arguments, it prints `--help` and exits. It does not start the console or bind ports until you choose a mode.
Bare `mesh-llm serve` loads startup models from `[[models]]` in `~/.mesh-llm/config.toml`.

## Background service

To install Mesh LLM as a per-user background service:

```bash
curl -fsSL https://raw.githubusercontent.com/Mesh-LLM/mesh-llm/main/install.sh | bash -s -- --service
```

Service installs are user-scoped:

- macOS installs a `launchd` agent at `~/Library/LaunchAgents/com.mesh-llm.mesh-llm.plist`
- Linux installs a `systemd --user` unit at `~/.config/systemd/user/mesh-llm.service`
- Shared environment config lives in `~/.config/mesh-llm/service.env`
- Startup models live in `~/.mesh-llm/config.toml`

Platform behavior:

- macOS loads `service.env` and then executes `mesh-llm serve`
- Linux writes `mesh-llm serve` directly into `ExecStart=`

The background service no longer stores custom startup args. Configure startup models in `~/.mesh-llm/config.toml` instead.

Optional shared environment file example:

```text
MESH_LLM_NO_SELF_UPDATE=1
```

If you edit the Linux unit manually:

```bash
systemctl --user daemon-reload
systemctl --user restart mesh-llm.service
```

If you want the service to survive reboot before login:

```bash
sudo loginctl enable-linger "$USER"
```

## Model catalog

List or fetch models from the built-in catalog:

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

## Specifying models

`mesh-llm serve --model` accepts several formats. Hugging Face-backed models are cached in the standard Hugging Face cache on first use.

```bash
mesh-llm serve --model Qwen3-8B
mesh-llm serve --model Qwen3-8B-Q4_K_M
mesh-llm serve --model https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf
mesh-llm serve --model bartowski/Llama-3.2-3B-Instruct-GGUF/Llama-3.2-3B-Instruct-Q4_K_M.gguf
mesh-llm serve --gguf ~/my-models/custom-model.gguf
mesh-llm serve --gguf ~/my-models/qwen3.5-4b.gguf --mmproj ~/my-models/mmproj-BF16.gguf
```

## Startup config

`mesh-llm serve` also loads startup models from `~/.mesh-llm/config.toml` by default.

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

Use the default config:

```bash
mesh-llm serve
```

If no startup models are configured, `mesh-llm serve` prints a `⚠️` warning, shows help, and exits.

Or an explicit path:

```bash
mesh-llm serve --config /path/to/config.toml
```

Config precedence:

- Explicit `--model` or `--gguf` ignores configured `[[models]]`.
- Explicit `--ctx-size` overrides configured `ctx_size` for the selected startup models.
- `mmproj` is optional and only used when that startup model needs a projector sidecar.
- Plugin entries stay in the same file.

## Lemonade integration

mesh-llm includes a built-in `lemonade` plugin for routing requests to a local [Lemonade Server](https://lemonade-server.ai) through the same `http://localhost:9337/v1` API that mesh-llm already exposes.

Start Lemonade first, either with the Lemonade Desktop app or with the CLI:

```bash
lemonade-server serve
curl -s http://localhost:8000/api/v1/models | jq '.data[].id'
```

The plugin uses `http://localhost:8000/api/v1` by default. To point at a different Lemonade endpoint, set:

```bash
export MESH_LLM_LEMONADE_BASE_URL=http://127.0.0.1:8000/api/v1
```

Then enable the plugin in `~/.mesh-llm/config.toml`:

```toml
[[plugin]]
name = "lemonade"
enabled = true
```

Start mesh-llm normally:

```bash
mesh-llm serve --model Qwen3-8B-Q4_K_M
```

After startup, mesh-llm should include Lemonade-hosted models in its own model list:

```bash
curl -s http://localhost:9337/v1/models | jq '.data[].id'
```

Requests sent to mesh-llm with a Lemonade model ID are forwarded to Lemonade:

```bash
curl http://localhost:9337/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-0.6B-GGUF",
    "messages": [
      {"role": "user", "content": "hello"}
    ]
  }'
```

Notes:

- mesh-llm does not start or supervise Lemonade; run it separately with the Desktop app or CLI.
- Use the exact model ID returned by Lemonade's `/api/v1/models`.
- If you use the mesh-llm background service, add `MESH_LLM_LEMONADE_BASE_URL=...` to `~/.config/mesh-llm/service.env`.

Useful model commands:

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

## Model storage

- Hugging Face repo snapshots are the canonical managed model store.
- Flat `~/.models/` storage is no longer scanned for managed models.
- Arbitrary local GGUF files still work through `mesh-llm serve --gguf`.
- MoE split artifacts are cached under `~/.cache/mesh-llm/splits/`.

## Inspect local GPUs

```bash
mesh-llm gpus
mesh-llm gpus --json
mesh-llm gpu benchmark --json
```

This prints the local GPU inventory with stable IDs, backend device names, VRAM, unified-memory status, and cached bandwidth when a benchmark fingerprint is already present. Add `--json` for machine-readable inventory output, or run `mesh-llm gpu benchmark --json` to refresh the cached fingerprint and print the benchmark summary as JSON.

## Local runtime control

Stage one supports local-only hot load and unload on a running node.

```bash
mesh-llm load Llama-3.2-1B-Instruct-Q4_K_M
mesh-llm unload Llama-3.2-1B-Instruct-Q4_K_M
mesh-llm status
```

Management API endpoints:

```bash
curl localhost:3131/api/runtime
curl localhost:3131/api/runtime/processes
curl -X POST localhost:3131/api/runtime/models \
  -H 'Content-Type: application/json' \
  -d '{"model":"Llama-3.2-1B-Instruct-Q4_K_M"}'
curl -X DELETE localhost:3131/api/runtime/models/Llama-3.2-1B-Instruct-Q4_K_M
```

This stage is intentionally node-local. Mesh-wide rebalancing and distributed load/unload come later.
