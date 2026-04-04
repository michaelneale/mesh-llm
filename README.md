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

### Model selection and storage

```bash
# Catalog name (fuzzy match — finds Qwen3-8B-Q4_K_M)
mesh-llm --model Qwen3-8B

# Full catalog name
mesh-llm --model Qwen3-8B-Q4_K_M

# MLX catalog name
mesh-llm --model Qwen3-4B-MLX --mlx

# HuggingFace URL (any GGUF)
mesh-llm --model https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf

# HuggingFace shorthand (org/repo/file.gguf)
mesh-llm --model bartowski/Llama-3.2-3B-Instruct-GGUF/Llama-3.2-3B-Instruct-Q4_K_M.gguf

# HuggingFace repo shorthand (works when the repo has one clear primary artifact)
mesh-llm --model mlx-community/Qwen2.5-0.5B-Instruct-4bit --mlx

# Prefer GGUF or MLX when a repo has multiple candidates
mesh-llm --model some-org/some-repo --gguf
mesh-llm --model some-org/some-repo --mlx

# Local file path (legacy/raw file mode)
mesh-llm --gguf-file ~/my-models/custom-model.gguf

# Local MLX model path
mesh-llm --mlx-file ~/my-models/qwen3-mlx/model.safetensors
```

Catalog models are downloaded with resume support. Use the `models` subcommands to browse, inspect, and fetch exact refs.

MLX catalog entries use explicit `-MLX` names so they stay distinct from the GGUF catalog entries.

- Hugging Face repo snapshots are the canonical managed model store.
- `~/.models/` is deprecated and will be removed in a future release.
- Arbitrary local GGUF files remain supported through `--gguf-file`.
- MLX runtime selection is explicit: use `--mlx` with `--model`, or `--mlx-file` for a local MLX path.
- MoE split artifacts are cached separately under `~/.cache/mesh-llm/splits/`.

### MLX status

MLX is available on macOS as an experimental local backend for supported text models.

- `--model ... --mlx` is required for MLX runtime selection.
- `--mlx-file` is the explicit local-path MLX mode.
- MLX currently bypasses the existing distributed/split GGUF path and runs locally on the Mac serving node.
- On startup, MLX prints an experimental warning and points users to the GitHub issues page if they hit problems.

### Backend support matrix

This is the practical support snapshot for the backends in this branch. The `🦙 GGUF / llama` column reflects the families we currently document, catalog, or smoke through the llama.cpp path here, not every model llama.cpp may support upstream.

| Family | Example `🦙 GGUF / llama` model | `🦙 GGUF / llama` | Example `🍎 MLX` model | `🍎 MLX` | Notes |
|---|---|---:|---|---:|---|
| Llama | `Llama-3.2-3B-Instruct-Q4_K_M` | ✅ | `Llama-3.2-3B-Instruct-MLX` | ✅ | Dense text |
| Qwen2 | `Qwen2.5-3B-Instruct-Q4_K_M` | ✅ | `Qwen2.5-3B-Instruct-MLX` | ✅ | Dense text |
| Qwen3 | `Qwen3-8B-Q4_K_M` | ✅ | `Qwen3-4B-MLX` | ✅ | Reasoning and non-thinking flows |
| Gemma 2 | `gemma-2-2b-it-Q4_K_M` | ✅ | `Gemma-2-2B-it-MLX` | ✅ | Dense text |
| Gemma 3 | `Gemma-3-12B-it-Q4_K_M` | ✅ | `mlx-community/gemma-3-1b-it-qat-4bit` | ✅ | Text path only on MLX |
| Gemma 4 | `gemma-4-E4B-it-Q4_K_M` | ✅ | `Gemma-4-E4B-it-MLX` | ✅ | Text path only on MLX |
| GLM4 | `GLM-4.7-Flash-Q4_K_M` | ✅ | `GLM-4-9B-0414-MLX` | ✅ | Dense text |
| LFM2 | — | — | `LFM2-350M-MLX` | ✅ | MLX-only today in this repo |
| DeepSeekV3 / Kimi-K2 | — | — | `mlx-community/Kimi-K2-Instruct-4bit` | ✅ | Implemented on MLX, not in live smoke matrix |
| gpt-oss | — | — | `openai/gpt-oss` MLX repos | ✅ | Implemented on MLX, not in live smoke matrix |
| Kimi Linear | — | — | `kimi_linear` MLX repos | ✅ | Separate MLX family, not in live smoke matrix |

Current MLX limitations:

- experimental
- macOS only
- text only
- larger families like DeepSeekV3 / Kimi-K2, `gpt-oss`, and `Kimi Linear` are implemented but not yet part of the live macOS smoke matrix
- distributed/split MLX serving is still future work

Useful commands:

```bash
mesh-llm models recommended      # list built-in recommended models
mesh-llm models installed        # list installed local models
mesh-llm models search qwen 8b   # search Hugging Face GGUF repos
mesh-llm models search --catalog qwen
mesh-llm models show Qwen/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf
mesh-llm models show Qwen3-4B-MLX
mesh-llm models download Qwen/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf
mesh-llm models download Qwen3-4B-MLX
mesh-llm models download mlx-community/Qwen2.5-0.5B-Instruct-4bit
mesh-llm models download some-org/some-repo --mlx
mesh-llm models migrate          # inspect deprecated ~/.models content
mesh-llm models migrate --apply  # materialize recognized HF-backed models into the HF cache
mesh-llm models updates --check  # check cached HF repos for newer upstream revisions
mesh-llm models updates --all    # refresh all cached HF repos
mesh-llm models updates Qwen/Qwen3-8B-GGUF
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
