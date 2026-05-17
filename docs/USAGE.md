# Usage Guide

Use this operational reference for installation details, service mode, model
storage, and runtime control.

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

Release bundles install the `mesh-llm` host binary plus the flavor-specific
native runtime libraries it embeds. Normal serving runs inside the `mesh-llm`
host process, which loads the Skippy/llama.cpp stage runtime directly.

Published bundle flavors include macOS, Linux CPU, Linux ARM64 CPU, Linux CUDA,
Linux CUDA Blackwell, Linux ROCm, Linux Vulkan, Windows CPU, Windows CUDA,
Windows ROCm, and Windows Vulkan. Metal remains macOS-only.

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
mesh-llm serve --discover "my-mesh"
mesh-llm client --auto
mesh-llm gpus
mesh-llm discover
mesh-llm discover --name "my-mesh"
```

Mesh workflow details live in [MESHES.md](MESHES.md). Big-model split serving
lives in [SKIPPY_SPLITS.md](SKIPPY_SPLITS.md).

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

The background service reads startup models from `~/.mesh-llm/config.toml`.

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

Use the persisted TOML for future starts or reloads. It does not rewrite active
sessions in place, and request payload values still win over any request
defaults from the file.

The example below shows every configuration section with annotations. All
sections and fields are optional unless noted.

```toml
# ~/.mesh-llm/config.toml
#
# Comprehensive configuration reference.
#
# Precedence (highest → lowest):
#   explicit request field value
#   → per-model config ([[models]] entry)
#   → [defaults.*] global config
#   → family / topology policy
#   → built-in runtime defaults
#
# Request defaults are merged ONLY at the OpenAI frontend boundary when the
# incoming request field is absent or null. They never enter StageConfig,
# protobuf, or any lower runtime layer.

version = 1

# ---------------------------------------------------------------------------
# GPU assignment policy
# ---------------------------------------------------------------------------

[gpu]
# "auto"   — let the planner pick the best visible device (default)
# "pinned" — require an explicit device= in every model or in [defaults.hardware]
assignment = "auto"
parallel   = 2        # total parallel inference slots across all models

# ---------------------------------------------------------------------------
# Node identity and network
# ---------------------------------------------------------------------------

[owner_control]
bind           = "0.0.0.0:7447"          # QUIC listen address
advertise_addr = "203.0.113.10:18443"    # address announced to peers

# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------

[telemetry]
enabled  = true
endpoint = "http://localhost:4317"       # OTLP collector

# ---------------------------------------------------------------------------
# Global defaults — applied to every model that does not override the field
# ---------------------------------------------------------------------------

# --- Context, batching, and KV cache -------------------------------------
[defaults.model_fit]
ctx_size         = 8192          # context window size (tokens)
batch            = 512           # n_batch — prompt-processing chunk
ubatch           = 128           # n_ubatch — micro-batch within a batch
cache_type_k     = "auto"        # KV key dtype: auto f16 f32 bf16 q8_0 q4_0 …
cache_type_v     = "auto"        # KV value dtype (same enum)
flash_attention  = "auto"        # auto on off
kv_cache_policy  = "balanced"    # macro preset: auto quality balanced saver
                                 #   quality  → f16/f16, no forced RAM cap
                                 #   balanced → preserve runtime defaults
                                 #   saver    → low-memory dtypes + offload
                                 # explicit cache_type_k/v always wins over preset
kv_offload       = "auto"        # bool or "auto" — KV residency / offload policy
kv_unified       = "auto"        # bool or "auto" — unified KV layout (schema-reserved)
cache_ram_mib    = 0             # byte cap for KV cache in MiB; 0 = no cap (schema-reserved)
cache_idle_slots = 0             # idle slot retention count (schema-reserved)
prompt_cache     = "auto"        # bool or "auto" — reuse previous prompt KV
swa_full         = false         # sliding-window attention (model-family specific)

# exact-prefix cache sub-section
[defaults.model_fit.prefix_cache]
enabled              = true
max_entries          = 64
max_bytes            = 0         # 0 = no explicit byte cap
min_tokens           = 64
shared_stride_tokens = 32        # stride for shared-prefix record matching
shared_record_limit  = 4         # max retained shared-prefix records
payload_mode         = "auto"    # resident-kv kv-recurrent full-state auto

# Schema-reserved fields (accepted but not yet wired to runtime):
# keep_tokens          = 256     # session prompt retention
# context_shift        = "auto"  # long-context shift
# checkpoint_interval  = 100     # KV checkpoint cadence
# checkpoint_count     = 5       # KV checkpoint retention
# lookup_cache_static  = "/path/to/static.cache"
# lookup_cache_dynamic = "/path/to/dynamic.cache"

# --- Hardware and model loading ------------------------------------------
[defaults.hardware]
model_runtime    = "auto"        # backend: auto cpu cuda rocm metal vulkan
device           = "auto"        # device id/index, e.g. "cuda:0" or "0"
gpu_layers       = "auto"        # integer >= -1, or "auto" (all layers)
placement        = "auto"        # planner placement strategy enum
split_mode       = "auto"        # multi-GPU split: auto none layer row
main_gpu         = 0             # primary device index for split_mode tuning
safety_margin_gb = 2.0           # reserved headroom; maps to fit_target_mib
fit_target_mib   = 0             # explicit allocatable-memory target (MiB)
                                 # do NOT write derived values back into TOML
fit_context      = "auto"        # bool or "auto" — estimator context-fit mode
mmap             = "auto"        # bool or "auto" — memory-mapped model load
mlock            = false         # pin model pages in RAM
direct_io        = false         # bypass page cache for model reads
repack           = false         # backend-specific repack flag
op_offload       = false         # backend-specific op-offload flag
no_host_buffer   = false         # backend-specific host-buffer flag
warmup           = "auto"        # bool or "auto" — post-load warmup pass
check_tensors    = false         # tensor-validation at load time (debug)

# multi-GPU tensor split (per-GPU ratio list or backend-native string)
# tensor_split = [0.6, 0.4]

# staged (skippy) layer ownership — set by planner; override only when manual
# stage_layer_start = 0
# stage_layer_end   = 15

# model artifact — typically set per-model; unusual in [defaults]
# model_path  = "/models/default.gguf"
# hf_repo     = "org/model-GGUF"
# hf_file     = "model-q4_k_m.gguf"
# mmproj      = "mmproj-f16.gguf"

# LoRA adapters and control vectors
# lora_adapters   = ["/adapters/adapter-1.gguf"]
# control_vectors = ["/vectors/cv-1.gguf"]

# MoE (Mixture-of-Experts) routing
# cpu_moe   = "auto"   # bool or "auto"
# n_cpu_moe = 0        # number of experts to route to CPU

# --- Throughput, scheduling, and CPU -------------------------------------
[defaults.throughput]
parallel             = 1           # concurrent request slots
continuous_batching  = "auto"      # bool or "auto"
threads              = 8           # CPU inference thread count
threads_batch        = 4           # CPU batch-processing thread count
tuning_profile       = "balanced"  # macro preset: throughput balanced saver
                                   #   throughput → larger batch/ubatch, more parallel
                                   #   balanced   → preserve runtime defaults
                                   #   saver      → smaller batch/ubatch, lower parallel
                                   # explicit low-level fields always win over preset
slot_prompt_similarity = 0.5       # slot-reuse heuristic threshold
priority               = "normal"  # scheduler priority hint (integer or string)

# CPU affinity and NUMA (advanced — usually leave unset)
# cpu_affinity = "0-7"
# numa         = "distribute"
# poll         = "auto"   # bool or "auto" — polling strategy

# Rejected in model config — stays operational/host-level:
# threads_http       — HTTP worker pool
# sleep_idle_seconds — power-management idle

# --- Skippy staged serving -----------------------------------------------
[defaults.skippy]
activation_wire_dtype           = "auto"    # auto f16 f32 bf16 q8 q4 q2
binary_stage_transport          = "auto"    # auto on off
prefill_chunking                = "fixed"   # fixed schedule none
prefill_chunk_size              = 512       # tokens per prefill chunk
lifecycle_startup_timeout_ms    = 30000     # stage startup grace period (ms)
lifecycle_readiness_interval_ms = 250       # readiness poll interval (ms)
lifecycle_health_interval_ms    = 5000      # health-check interval (ms)

# Staged-only / manual topology (set by planner; override carefully)
# stage_model_path       = "/packages/stage-0.pkg"
# stage_role             = "prefill"
# stage_topology         = "2-stage-split"
# prefill_chunk_schedule = "128,256,512"   # custom progressive schedule

# --- Speculative decoding ------------------------------------------------
[defaults.speculative]
mode                       = "auto"          # auto off draft ngram lookahead
draft_selection_policy     = "auto"          # auto manual heuristic
pairing_fault              = "warn_disable"  # warn_disable fail_open fail_closed
draft_max_tokens           = 16
draft_min_tokens           = 1
draft_acceptance_threshold = 0.0             # 0.0 = use runtime default
spec_default               = "auto"          # bool or "auto"

# Draft model source (per-model is more typical; these are global fallbacks)
# draft_model_path = "/models/draft.gguf"
# draft_hf_repo    = "org/draft-GGUF"
# draft_hf_file    = "draft-q4_k_m.gguf"

# Draft hardware (leave unset to share host model's device)
# draft_gpu_layers   = -1
# draft_device       = "cuda:1"
# draft_threads      = 4
# draft_cache_type_k = "q8_0"
# draft_cache_type_v = "q8_0"

# N-gram speculative (when mode = "ngram")
# ngram_min = 1
# ngram_max = 5

# --- Request defaults (merged at OpenAI frontend only) -------------------
[defaults.request_defaults]
# Sampling — explicit request values always win
temperature       = 0.8
top_p             = 0.95
top_k             = -1           # -1 = disabled
min_p             = 0.05
typical_p         = 1.0
top_nsigma        = 0.0
dynatemp_range    = 0.0
dynatemp_exponent = 1.0
repeat_penalty    = 1.1
repeat_last_n     = 64
presence_penalty  = 0.0
frequency_penalty = 0.0
seed              = -1           # -1 = random

# Mirostat sampling (alternative to top_p/top_k)
mirostat_mode          = 0       # 0 off, 1 v1, 2 v2
mirostat_entropy       = 5.0
mirostat_learning_rate = 0.1

# Stop sequences (string or list of strings)
stop = ["<|im_end|>", "</s>"]

# Token budget
max_tokens = 2048
ignore_eos = false

# Sampler ordering (leave unset to use runtime default)
# samplers         = ["top_k", "top_p", "temperature"]
# sampler_sequence = "kpt"

# Logit bias: token_id → bias delta (TOML inline table)
# logit_bias = { "12345" = -2.0, "67890" = 1.5 }

# Reasoning (for thinking models)
reasoning_format  = "auto"   # auto none deepseek deepseek-legacy hidden
reasoning_enabled = "auto"   # bool or "auto" / "on" / "off"
reasoning_budget  = "auto"   # integer token budget, or "auto"

# Chat template (leave unset to use model's embedded template)
# chat_template      = "chatml"
# chat_template_file = "/path/to.jinja"
# jinja              = false
# skip_chat_parsing  = false

# System prompt injected at the start of every conversation
# system_prompt = "You are a helpful assistant."

# Schema-reserved (accepted, not yet wired):
#   dry, xtc, adaptive  — advanced sampler bags
#   backend_sampling    — raw backend sampling passthrough
#   grammar, json_schema, logprobs
#   prefill_assistant, chat_template_kwargs

# --- Multimodal ----------------------------------------------------------
[defaults.multimodal]
mmproj           = "default-mmproj-f16.gguf"  # vision projector path or HF ref
mmproj_offload   = "auto"                     # bool or "auto"
image_min_tokens = 0
image_max_tokens = 4096

# Schema-reserved (accepted, not yet wired):
#   mmproj_url  — projector URL source
#   embeddings, reranking, pooling, vocoder

# --- Advanced server (operational — reject most in model config) ---------
[defaults.advanced.server]
alias = "my-cluster"   # friendly name shown in /api/status
# host, port, reuse_port, timeout, metrics, slots, props, and api_prefix are
# operational or rejected here, not model-settings controls.

# ===========================================================================
# Per-model entries — each [[models]] block overrides specific defaults
# ===========================================================================

# ---------------------------------------------------------------------------
# Example 1: GPU-heavy model with staged serving and speculative decoding
# ---------------------------------------------------------------------------

[[models]]
model = "Qwen/Qwen3-8B:Q4_K_M"

[models.model_fit]
ctx_size        = 16384
batch           = 1024
ubatch           = 256
cache_type_k    = "f16"
cache_type_v    = "f16"
kv_cache_policy = "quality"    # overrides global "balanced"
flash_attention  = "on"
prompt_cache     = true

[models.model_fit.prefix_cache]
enabled     = true
max_entries = 128
min_tokens  = 128

[models.hardware]
device            = "cuda:0"
gpu_layers        = 99          # all layers on GPU
fit_target_mib    = 22528       # 22 GiB target
stage_layer_start = 0           # staged split: this node owns layers 0–15
stage_layer_end   = 15
split_mode        = "layer"
tensor_split      = [0.6, 0.4]  # two-GPU split ratios
main_gpu          = 0
mmap              = true
warmup            = true
lora_adapters     = ["/adapters/qwen-chat-v2.gguf"]

[models.throughput]
parallel            = 4
continuous_batching = true
tuning_profile      = "throughput"
threads             = 16
threads_batch       = 8

[models.skippy]
activation_wire_dtype  = "f16"
prefill_chunking       = "schedule"
prefill_chunk_size     = 256
prefill_chunk_schedule = "128,256,512,1024"

[models.speculative]
mode                   = "draft"
draft_model_path       = "/models/qwen3-0.6b-q8.gguf"
draft_selection_policy = "manual"
pairing_fault          = "warn_disable"
draft_max_tokens       = 8
draft_gpu_layers       = 28
draft_device           = "cuda:1"
draft_cache_type_k     = "q8_0"
draft_cache_type_v     = "q8_0"

[models.request_defaults]
temperature      = 0.7
top_p            = 0.9
repeat_penalty   = 1.05
max_tokens       = 4096
reasoning_format = "hidden"
reasoning_budget = 512
system_prompt    = "You are a helpful coding assistant."
stop             = ["<|im_end|>"]

[models.multimodal]
mmproj           = "Qwen/Qwen2.5-VL-7B-Instruct-GGUF/mmproj-f16.gguf"
mmproj_offload   = true
image_max_tokens = 8192

[models.advanced.server]
alias = "qwen3-8b"

# ---------------------------------------------------------------------------
# Example 2: CPU-only small model, minimal config
# ---------------------------------------------------------------------------

[[models]]
model = "bartowski/gemma-3-1b-it-GGUF/gemma-3-1b-it-Q4_K_M.gguf"

[models.hardware]
model_runtime = "cpu"
gpu_layers    = 0
mmap          = true

[models.model_fit]
ctx_size = 4096
batch    = 128
ubatch   = 64

[models.throughput]
threads        = 4
threads_batch  = 4
tuning_profile = "saver"

[models.request_defaults]
temperature = 0.9
max_tokens  = 512

[models.advanced.server]
alias = "gemma-tiny"

# ---------------------------------------------------------------------------
# Example 3: MoE model with CPU expert offload
# ---------------------------------------------------------------------------

[[models]]
model = "bartowski/Mixtral-8x7B-Instruct-v0.1-GGUF/Mixtral-8x7B-Instruct-v0.1-Q4_K_M.gguf"

[models.hardware]
device         = "cuda:0"
gpu_layers     = 32
cpu_moe        = true
n_cpu_moe      = 4             # route 4 experts to CPU
split_mode     = "row"
placement      = "auto"
fit_target_mib = 20480

[models.model_fit]
ctx_size        = 8192
kv_cache_policy = "saver"

[models.throughput]
parallel = 2
threads  = 8

[models.advanced.server]
alias = "mixtral-8x7b"

# ---------------------------------------------------------------------------
# Example 4: Vision model from Hugging Face
# ---------------------------------------------------------------------------

[[models]]
model = "Qwen/Qwen2.5-VL-7B-Instruct-GGUF/qwen2.5-vl-7b-instruct-q4_k_m.gguf"

[models.hardware]
hf_repo    = "Qwen/Qwen2.5-VL-7B-Instruct-GGUF"
hf_file    = "qwen2.5-vl-7b-instruct-q4_k_m.gguf"
device     = "cuda:0"
gpu_layers = 99

[models.multimodal]
mmproj           = "bartowski/Qwen2.5-VL-7B-Instruct-GGUF/mmproj-f16.gguf"
mmproj_offload   = true
image_min_tokens = 16
image_max_tokens = 16384

[models.model_fit]
ctx_size = 8192

[models.advanced.server]
alias = "qwen-vl"

# ---------------------------------------------------------------------------
# Plugin declarations
# ---------------------------------------------------------------------------

[[plugin]]
name    = "blackboard"
enabled = true

# [[plugin]]
# name    = "openai-endpoint"
# url     = "http://localhost:8000/api/v1"
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

- Request values override per-model config, which override `[defaults.*]`, which
  override family or topology policy, which finally override built-in runtime
  defaults.
- Request defaults only fill missing or null request fields at the OpenAI
  frontend boundary. Explicit request values win, and those defaults never
  become `StageConfig`, runtime load structs, protobuf payloads, or lower-layer
  runtime settings.
- Explicit `--model` or `--gguf` ignores configured `[[models]]`.
- Explicit `--ctx-size` overrides configured `ctx_size` for the selected startup
  models.
- `mmproj` is optional and only used when that startup model needs a projector
  sidecar.
- `skippy.*` staged-serving controls stay staged-only. `activation_wire_dtype`,
  prefill controls, speculative draft controls, and manual stage layer ranges
  apply only when the model is started in staged mode.
- `safety_margin_gb` resolves to `hardware.fit_target_mib` by subtracting the
  reserved MiB from detected allocatable memory, and the derived target is not
  written back into TOML.
- Changing this file affects future starts or reloads, not active sessions.
- Plugin entries stay in the same file.

## Lemonade integration

Use the `openai-endpoint` plugin to route requests to a local [Lemonade Server](https://lemonade-server.ai) through the same `http://localhost:9337/v1` API that mesh-llm exposes.

Start Lemonade first, either with the Lemonade Desktop app or with the CLI:

```bash
lemonade-server serve
curl -s http://localhost:8000/api/v1/models | jq '.data[].id'
```

Then enable the plugin in `~/.mesh-llm/config.toml`:

```toml
[[plugin]]
name = "openai-endpoint"
url = "http://localhost:8000/api/v1"
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
- The URL can also be set via `MESH_LLM_OPENAI_ENDPOINT_URL` env var (config takes precedence).

Useful model commands:

```bash
mesh-llm models recommended
mesh-llm models installed
mesh-llm models search qwen 8b
mesh-llm models search --catalog qwen
mesh-llm models show Qwen/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf
mesh-llm models download Qwen/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf
mesh-llm models package unsloth/Qwen3-8B-GGUF:Q4_K_M --dry-run
mesh-llm models updates --check
mesh-llm models updates --all
mesh-llm models updates Qwen/Qwen3-8B-GGUF
mesh-llm models cleanup
mesh-llm models prune
```

## Model storage

- Hugging Face repo snapshots are the canonical managed model store.
- Managed model scans use Hugging Face repo snapshots.
- Arbitrary local GGUF files still work through `mesh-llm serve --gguf`.
- Skippy materialized stage GGUFs are derived cache and can be preview-pruned
  with `mesh-llm models prune`.

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

## Owner-control plane

Owner-control is the operator lane for config and inventory actions. It does **not** replace the public mesh plane used for join, gossip, routing, or inference. Config and inventory mutation are exclusive to `mesh-llm-control/1`; the old mesh-plane config stream IDs are reserved but no longer carry protobuf request/response handling.

### Bootstrap contract

- New control clients need an explicit owner-control endpoint token.
- Read the local bootstrap policy from `GET /api/runtime/control-bootstrap` or `mesh-llm runtime bootstrap --json`.
- If no explicit endpoint is supplied, the current client contract returns `ControlEndpointRequired`.
- If an explicit endpoint is configured and fails, the client stays on owner-control and reports a structured failure. It does **not** silently fall back to mesh-plane config streams.

### Transport and fallback matrix

| Caller / target | Result |
|---|---|
| New client + explicit endpoint | Use `mesh-llm-control/1` only; no silent legacy downgrade |
| New client + no endpoint | `ControlEndpointRequired` |
| New client ↔ old node with no endpoint | `ControlEndpointRequired` by default |
| Old client + new node | Legacy mesh-plane config stream IDs are reserved but rejected as unsupported/unknown |
| Old node ↔ new node public mesh join/routing | Public mesh ALPN negotiation, gossip, and routing remain compatible; owner-control is not required for join/routing |
| Old client + old node | Unchanged old-node behavior outside this release |

### Operator commands

Inspect the local bootstrap policy:

```bash
mesh-llm runtime bootstrap --port 3131 --json
curl -s localhost:3131/api/runtime/control-bootstrap | jq .
```

Run owner-control requests through the local management API using an explicit endpoint token:

```bash
mesh-llm runtime get-config --port 3131 --endpoint '<control-endpoint>' --json
mesh-llm runtime refresh-inventory --port 3131 --endpoint '<control-endpoint>' --json
mesh-llm runtime apply-config \
  --port 3131 \
  --endpoint '<control-endpoint>' \
  --expected-revision 7 \
  --config /absolute/path/to/config.toml \
  --json
```

Equivalent REST calls:

```bash
curl -s -X POST localhost:3131/api/runtime/control/get-config \
  -H 'Content-Type: application/json' \
  -d '{"endpoint":"<control-endpoint>"}' | jq .

curl -s -X POST localhost:3131/api/runtime/control/refresh-inventory \
  -H 'Content-Type: application/json' \
  -d '{"endpoint":"<control-endpoint>"}' | jq .

curl -s -X POST localhost:3131/api/runtime/control/apply-config \
  -H 'Content-Type: application/json' \
  -d '{
    "endpoint":"<control-endpoint>",
    "expected_revision":7,
    "config":{"version":1}
  }' | jq .
```

### Failure modes

| Error | Meaning | Typical operator action |
|---|---|---|
| `ControlEndpointRequired` / `control_endpoint_required` | No explicit endpoint was supplied | Read `runtime bootstrap`, then retry with the advertised endpoint token |
| `ControlUnsupported` / `control_unsupported` | Target accepted the connection path but does not speak `mesh-llm-control/1` | Verify the endpoint token targets an owner-control listener |
| `ControlUnavailable` / `control_unavailable` | Endpoint token, listener, network path, or local owner key loading failed | Verify the endpoint token, listener status, and local owner keystore/passphrase |
| `Unauthorized` / `unauthorized` | Same-owner handshake failed | Check that both nodes use the same owner identity and that the local key can be unlocked |
| `RevisionConflict` / `revision_conflict` | Apply request used a stale `expected_revision` | Re-read config, merge, and retry with the current revision |
| `LegacyJsonUnsupported` / `legacy_json_unsupported` | A legacy mesh-plane frame hit `mesh-llm-control/1` | Fix the caller to use owner-control protobuf frames |

### Transition note

Treat owner-control as the only lane for operator config and inventory clients. Legacy mesh-plane config stream IDs remain reserved for compatibility bookkeeping, but current nodes do not handle config subscribe/push requests on `mesh-llm/1`.

### Mixed-version QA harness

Use the task harness when you need executable evidence for mixed-version routing or owner-control bootstrap:

```bash
scripts/qa-control-plane-mixed-version.sh \
  --released-binary ./target/qa/released/mesh-llm \
  --current-binary ./target/debug/mesh-llm \
  --evidence-dir .sisyphus/evidence
```

Loopback-only routing/owner-control smoke:

```bash
scripts/qa-control-plane-mixed-version.sh \
  --released-binary ./target/qa/released/mesh-llm \
  --current-binary ./target/debug/mesh-llm \
  --evidence-dir .sisyphus/evidence \
  --local-only
```

Owner-control bootstrap lane only:

```bash
scripts/qa-control-plane-mixed-version.sh \
  --released-binary ./target/qa/released/mesh-llm \
  --current-binary ./target/debug/mesh-llm \
  --evidence-dir .sisyphus/evidence \
  --local-only \
  --config-only
```

To validate the harness contract without starting processes or writing evidence, add `--print-plan`; it prints the planned public, loopback, and owner-control result names as JSON.

`--config-only` skips public-mesh probes and focuses on the owner-control migration lane:

- loopback released/current private-mesh coexistence in both directions
- current-branch proof that new clients prefer `mesh-llm-control/1`
- current-branch proof that missing endpoints fail with `ControlEndpointRequired`
- current-node `runtime bootstrap` / `runtime get-config` evidence when owner-control is enabled

Each real run writes a timestamped evidence directory with `manifest.json`, `commands.jsonl`, `results.jsonl`, `summary.md`, `summary.json`, `versions/*.txt`, process logs, and grouped status/model/chat/control payloads.

If the local bootstrap payload reports `enabled=false`, the harness records a `PREREQ` result explaining that a signed same-owner keystore is required before runtime owner-control requests can be proven on that machine. That is an explicit prerequisite report, not a silent pass.
