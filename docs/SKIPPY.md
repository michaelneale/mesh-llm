# Skippy Integration Plan

This document captures the migration plan for fully replacing mesh-llm's
current `llama-server` + `rpc-server` serving path with the skippy staged
runtime.

The goal is to make skippy the only model-serving runtime inside mesh-llm.
Mesh-llm remains responsible for the public product surface: mesh membership,
model resolution, routing, demand tracking, the OpenAI-compatible API, runtime
control, and user-facing status.

## Direction

Skippy lets mesh-llm move model execution into Rust-owned runtime orchestration
instead of patched `llama-server` and `rpc-server` subprocesses. The migration
plan should close parity gaps rather than preserve a long-term legacy backend.

The target shape is:

```text
OpenAI client
    |
    v
mesh-llm OpenAI-compatible API
    |
    v
mesh router / demand / affinity / target selection
    |
    v
skippy backend
    |
    +-- single in-process stage for local serving
    +-- multi-stage layer pipeline across mesh peers
```

This replaces the current split-serving model:

```text
host llama-server + worker rpc-server + RPC tunnel/rewrite
```

with:

```text
stage 0 runtime + stage 1..N runtimes + plain activation transport
```

## Scope

Bring over these skippy crates:

| Crate | Why mesh needs it |
|---|---|
| `skippy-ffi` | Rust FFI bindings to the llama.cpp ABI |
| `skippy-runtime` | Safe model/session/runtime wrapper and layer package materialization |
| `skippy-protocol` | Stage wire protocol and shared stage config types |
| `skippy-server` | Stage transport and OpenAI driver code to refactor into embeddable APIs |
| `skippy-topology` | Layer topology planning and family/split policy |
| `skippy-metrics` | Runtime telemetry helpers |
| `metrics-server` | Benchmark/debug OTLP ingest, run lifecycle, DuckDB storage, and report export |
| `openai-frontend` | Shared OpenAI-compatible request/response, streaming, responses API compatibility, structured-output, tool-call, logprob, and backend contract |
| `llama-model-slice` | Later package tooling for producing layer packages |

Keep the first mesh integration focused on staged serving, OpenAI compatibility,
model lifecycle, topology planning, and package materialization. Prefix-cache
sidecars and model-free speculative history services are outside this branch.

## Important Boundary

Skippy replaces mesh-llm's current llama.cpp patch queue for serving behavior,
but it does not remove the need for a patched llama.cpp yet. Skippy has its own
stage ABI patch queue. The replacement is:

- remove mesh's patched `llama-server`, `rpc-server`, mesh hooks, RPC rewrite
  path, and legacy split tooling;
- adopt the llama-stage ABI build and runtime path;
- keep mesh-llm as the user-facing product and orchestration layer.

## Lessons From PR #421

PR #421 proved the staged direction in mesh-llm with live two-machine
orchestration, including a Qwen3-235B layer package split across a Mac Studio
and MacBook over iroh. The implementation in that PR is intentionally
experimental, but several findings should be carried forward.

What to keep:

- Layer packages must be first-class model artifacts. Detect
  `model-package.json`, support `hf://...` package refs, and keep package
  identity separate from plain GGUF identity.
- Stage traffic should use plain byte relay over the mesh tunnel. The old RPC
  port-rewrite path is not appropriate for skippy activation traffic.
- Layer-parallel splits should not inherit the old RPC RTT cutoff directly.
  Stage pipelines have different latency and bandwidth tradeoffs from tensor
  RPC offload.
- Remote stage commands should send package identity, manifest hash, model id,
  layer range, selected files, activation wire policy, and topology/run ids.
  They should not assume matching local paths on every peer.
- Downstream/final stages should become ready before upstream stages and the
  driver start sending work.
- Mesh should route to a stage-0 OpenAI driver or direct backend handle, but
  mesh should continue to own `/v1` compatibility and routing policy.
- Stage readiness must be explicit. A spawned process or loaded runtime is not
  routable until the stage and driver have passed readiness checks.

What to avoid:

- Do not leak child processes to keep stages alive. Use managed handles with
  explicit shutdown, or in-process cancellation handles once embedded.
- Do not store a single global `stage_inbound_port`. Stage routing must be
  keyed by model, topology/run id, and stage id so reloads and multi-model
  serving do not collide.
- Do not hardcode activation width, context size, or port arithmetic. Derive
  activation width from model metadata or reviewed family capability, take
  context from mesh config, and allocate ports explicitly.
- Do not make skippy-server the public product API. Mesh keeps routing and
  user-facing product behavior, while `openai-frontend` provides the shared
  OpenAI-compatible request/response contract used by mesh and skippy.

## Architecture Target

### Single Node

For a model that fits locally, mesh should be able to run skippy as an
in-process backend:

```text
mesh-llm
  +-- OpenAI API / router
  +-- skippy runtime model
  +-- skippy sessions
```

This path should replace local `llama-server` serving once the required
lifecycle, device, OpenAI compatibility, direct-GGUF package, status, and
multimodal parity gates are met.

Current branch status:

- `openai-frontend` has been imported and mesh's local OpenAI request/response
  adapters delegate to it.
- `skippy-server` exposes embeddable runtime and OpenAI backend handles.
- `skippy-server` is split by concern so request parsing/sampling helpers,
  utility helpers, socket connection fallback, and tests are no longer all
  carried in the serving monoliths.
- Runtime-control loads can select `--serving-backend skippy` and route a
  direct GGUF through an embedded skippy OpenAI backend instead of
  `llama-server`.
- Normal startup can select `--serving-backend skippy`; this skips
  `rpc-server`, loads the configured direct GGUF as a single skippy
  `RuntimeSlice` stage, publishes a normal local mesh target, and preserves
  startup load/unload lifecycle through the managed model controller.
- The legacy llama election loop remains the default while skippy is hidden
  behind the backend selector. The next replacement step is multi-peer stage
  topology planning and activation transport, after which the legacy
  `llama-server`/`rpc-server` startup path can be retired.
- `metrics-server` is in the workspace with OTLP ingest, DuckDB-backed run
  lifecycle/report export, `just metrics-server` recipes, and agent workflow
  documentation.

### Multi Node

For a model that needs multiple machines, mesh should plan a contiguous stage
topology:

```text
node A: stage 0, layers 0..N, driver/routing target
node B: stage 1, layers N..M
node C: stage 2, layers M..end
```

Activations flow downstream between stages and predicted-token replies flow
back upstream. The transport can be direct TCP when reachable or TCP relayed
over the existing iroh tunnel when not.

### Mesh Responsibilities

Mesh continues to own:

- node discovery, admission, gossip, and version compatibility;
- model resolution, model interests, local inventory, and download policy;
- stage topology selection and per-peer assignment;
- public `/v1` routing behavior and compatibility tests;
- routing, target selection, demand tracking, and affinity;
- runtime control, status, logs, and dashboard state.

### Skippy Responsibilities

Skippy owns:

- model/session execution through the stage ABI;
- stage protocol types, the `skippy-stage/1` QUIC ALPN, protobuf frames in
  `skippy-protocol`, and activation wire encoding;
- layer package loading/materialization;
- topology planning primitives and family capability policy;
- backend telemetry emitted by runtime/stage execution.

`openai-frontend` sits between those two ownership areas. It should be the
shared OpenAI-compatible surface crate for:

- request and response structs;
- `/v1/responses` compatibility adapters;
- streaming Server-Sent Events chunks;
- OpenAI-style error bodies;
- backend traits for chat/completions/models;
- structured-output, tool-call, and logprob request/response support;
- compatibility fixtures shared by mesh and skippy.

Mesh should compose those types with mesh-specific routing, authorization,
model selection, demand tracking, multimodal object handling, and management
APIs. Skippy should implement the backend trait for local/staged execution.
Endpoint compatibility that is not mesh-specific should move into
`openai-frontend` so mesh does not carry a second OpenAI compatibility layer.

## Auto LLM / Virtual LLM Hooks

The migration must preserve mesh's Auto LLM / Virtual LLM behavior. Today that
feature is powered by patched `llama-server` callbacks into mesh's
`/mesh/hook` route, with handlers in `inference::virtual_llm` and peer
consultation in `inference::consult`.

This behavior is not automatically preserved by the skippy startup path because
skippy does not run patched `llama-server`. The replacement must move the hook
points into Rust-owned request orchestration before the legacy backend is
removed.

Required hook parity:

- preserve the auto-routed `model=auto` behavior that injects
  `mesh_hooks: true`;
- preserve the recursion guard where peer consultations set
  `mesh_hooks: false`;
- preserve media fallback hooks for images/audio/video that the selected model
  cannot handle;
- preserve uncertainty and drift hooks that consult different peer models and
  inject context or hints;
- preserve hook debug/testing controls such as `MESH_HOOK_DEBUG` and
  `--mesh-hook-debug`, or replace them with equivalent mesh-owned controls;
- expose hook decisions, peer consultation failures, and injected context in
  trace/log output without leaking private request content into public status;
- add compatibility fixtures proving the same request bodies produce equivalent
  hook actions under skippy and the legacy llama backend.

Target ownership:

- mesh owns hook policy, peer selection, recursion guards, and request-level
  routing decisions;
- `openai-frontend` should expose typed preflight/prefill/generation hook
  extension points so hooks do not require raw JSON reparsing;
- skippy should provide runtime signals needed by the hooks, such as media
  rejection before tokenization, post-prefill uncertainty, and mid-generation
  drift/entropy windows.

Until this parity exists, skippy can be used as a hidden backend selector, but
the legacy llama backend cannot be deleted for Auto LLM users.

Current branch status:

- `openai-frontend` defines typed hook policy extension points for
  pre-chat/media fallback, post-prefill uncertainty, and mid-generation drift;
- mesh passes an in-process hook policy into skippy's embedded OpenAI backend;
- skippy chat requests with `mesh_hooks: true` now call the existing
  `inference::virtual_llm` media fallback before skippy applies the chat
  template;
- peer consultations still force `mesh_hooks: false` to preserve the recursion
  guard;
- skippy's llama.cpp ABI patch queue now exposes `skippy-signals.h` for
  first-token and generation-window telemetry;
- skippy chat generation now calls post-prefill uncertainty and mid-generation
  drift hooks in-process using runtime token/window signals;
- injected hook text is materialized into the active skippy session as hidden
  continuation context, while the typed request is updated so future hook
  consultations see the same context;
- skippy hook policy has fixture coverage for `mesh_hooks` injection,
  recursion guard, media fallback, uncertainty, drift, and debug forcing
  controls;
- operators can force skippy hook paths during testing with
  `MESH_HOOK_DEBUG_FORCE=pre_inference,post_prefill,mid_generation` and can
  override the injected debug text with `MESH_HOOK_DEBUG_TEXT`.

## Lifecycle and Device Parity

The skippy migration must preserve mesh-llm's current model lifecycle and
hardware-selection semantics. The adapter should make these requirements
explicit instead of treating them as incidental behavior of `llama-server`.

### Model Lifecycle

Mesh must continue to support:

- loading a model at startup from CLI/TOML configuration;
- loading additional models through runtime control APIs;
- unloading a model without stopping the whole mesh node;
- reloading a model after config changes or topology changes;
- reporting `starting`, `loading`, `ready`, `failed`, `stopping`, and
  `stopped` states;
- withdrawing routing targets before a backend is stopped;
- cleaning up sessions, memory, logs, and runtime status on shutdown.

The implementation should replace process handles with explicit skippy handles:

```rust
struct SkippyModelHandle {
    model_id: String,
    topology_id: Option<String>,
    stages: Vec<SkippyStageHandle>,
    driver: SkippyDriverHandle,
    status: BackendStatus,
}
```

The exact type names can change, but the ownership model should not: a handle
must own the loaded runtime resources and provide explicit readiness and
shutdown. Dropping a handle should not be the only cleanup path.

### Device Pinning

Mesh must preserve pinned GPU/device behavior. Current users can constrain a
model to a specific backend device through CLI/TOML configuration and per-model
startup planning. Skippy must expose an equivalent path.

Required behavior:

- preserve per-model GPU assignment and pinned-device selection;
- reject conflicting explicit device and pinned assignment combinations;
- surface the selected device in status/dashboard output;
- pass selected device through both single-node and staged launches;
- keep CPU placement behavior explicit when no GPU backend is available.

Skippy's current Rust `RuntimeConfig` has `n_gpu_layers`, but device selection
must be added if the current C ABI cannot express it. The replacement plan
therefore includes a skippy ABI/config extension for selected device, plus tests
that prove pinned models load on the requested backend device and reject invalid
device names before becoming routable.

Current branch status:

- mesh passes a normalized selected-device descriptor into skippy stage config
  for pinned startup/runtime loads;
- skippy runtime status preserves that selected-device descriptor;
- skippy's llama.cpp ABI accepts an explicit selected backend device string
  and applies it before model load, including Metal, Vulkan, CUDA/HIP/ROCm, and
  explicit CPU placement;
- invalid selected backend device names fail during model open before the model
  is marked routable.

Device ownership should use a two-layer model:

- mesh owns a normalized device descriptor for config, inventory, planning,
  pinning decisions, and status;
- the skippy ABI accepts a backend device string that maps directly to
  llama.cpp's device selector, such as `Metal0`, `CUDA0`, `Vulkan1`, or `CPU`.

Mesh should probe and validate the selected backend device before load, pass the
backend device string into skippy, and report the resolved normalized descriptor
and raw backend device string in runtime status.

### Runtime Knobs

The skippy adapter must preserve the user-visible runtime knobs mesh already
supports:

- context size from CLI/TOML;
- per-model concurrency/parallelism limits;
- backend admission and rate limiting;
- KV cache type policy where applicable;
- draft/speculative settings only when parity exists;
- model source/provenance in status and usage records;
- log paths and error reporting usable from the existing dashboard/API.

During burn-in, subprocess-backed skippy can sit behind the same handle trait as
the final in-process backend. That gives us crash isolation while the native
runtime path proves itself. This is a development migration aid only; the target
product shape is skippy-owned serving without `llama-server`.

### Direct GGUF As Package

Mesh should continue to support direct GGUF model references. A plain GGUF
should materialize as a synthetic single-model package in skippy runtime terms
rather than taking a separate serving path.

Required behavior:

- preserve existing GGUF, split-GGUF, catalog, and Hugging Face resolution
  inputs;
- derive a package identity, manifest hash, tensor metadata, layer count,
  activation width, and capability metadata from direct GGUF inputs;
- use the same package abstraction for local single-stage serving and
  multi-stage layer planning, but do not treat an arbitrary `--gguf` path as a
  distributable stage package;
- cache synthetic package metadata under mesh's model storage, without copying
  the full model unless slicing/materialization requires it;
- surface synthetic package provenance in status so users can still understand
  which GGUF file is loaded.

### Materialization Cache

Materialized skippy stage artifacts are derived cache, not durable user-owned
model data. The source model or package remains the durable artifact.

Required behavior:

- `models delete <model>` should remove the source model/package and all
  derived materialized stage artifacts for that source;
- `models prune` should be allowed to remove stale materialized artifacts before
  deleting durable source models;
- active runtimes and active topologies must pin their materialized artifacts so
  cleanup cannot delete files that are in use;
- synthetic direct-GGUF package manifests and stage slices should be
  regenerable from the source artifact;
- runtime status should show materialized artifact path, size, source identity,
  and whether it is currently pinned by an active runtime.

### Exact KV/Recurrent Cache Policy

Mesh should bring over the exact BLAKE3-deduplicated KV/recurrent cache path
that won the Qwen benchmark work, but it should live inside the skippy runtime
path rather than as the old standalone `kv-server`.

Default policy:

- the global default is `auto`, which behaves as off for unknown or
  uncertified families;
- mesh may enable the exact cache by default only for reviewed
  model-family/runtime configurations with correctness and benchmark evidence;
- unknown families, new quantization/layout combinations, or unreviewed
  recurrent models require explicit per-model opt-in;
- per-model config overrides the global policy, including the byte cap;
- a loaded model/stage owns its cache budget, and active sessions pin cache
  entries they are using until release.

Correctness requirements:

- cache restore is exact only, never fuzzy;
- cache identity includes the model/package hash, layer range, tokenizer and
  chat-template identity, runtime options that affect state, and the exact token
  prefix;
- BLAKE3 content hashes deduplicate identical KV/recurrent pages, but a page is
  reusable only after the full cache identity matches;
- recurrent state support must come from reviewed family capability data before
  it can be enabled by `auto`.

Operational requirements:

- expose `mode`, `max_bytes`, `used_bytes`, `deduped_bytes`, hit/miss counts,
  restored-prefix tokens, and eviction counts in runtime status and metrics;
- enforce a byte-based cap per loaded model/stage, with explicit eviction rather
  than unbounded growth;
- make cache contents derived runtime state, not durable model data, so model
  deletion/prune can discard it safely;
- keep a global emergency off switch for burn-in and incident response.

Example shape:

```toml
[skippy.cache]
mode = "auto"
max_bytes = "4GiB"

[[models]]
model = "hf://Qwen/..."
serving_backend = "skippy"

[models.skippy_cache]
mode = "exact"
max_bytes = "8GiB"
```

## Multimodal Parity

Multimodal support is part of the replacement plan. Mesh should not keep
`llama-server` as the long-term vision/audio fallback.

Mesh currently owns multimodal concerns that are outside a plain text runtime:

- request-scoped media/blob handling;
- OpenAI multi-part message normalization;
- model capability advertisement for vision, audio, and multimodal support;
- modality-aware routing;
- `mmproj` discovery/loading for llama.cpp backends;
- UI/API status for multimodal capability and active routing.

Required skippy replacement work:

- add skippy runtime support for llama.cpp multimodal projector loading;
- pass resolved projector/package metadata through skippy configs;
- preserve mesh blob/media normalization before request execution;
- keep multimodal capability advertisement and routing semantics;
- test OpenAI multipart image/audio/file request shapes through
  `openai-frontend` and mesh routing;
- expose loaded projector/media capability in runtime status.

The implementation can land in stages, but full removal of `llama-server`
requires skippy multimodal parity for the model families mesh advertises as
multimodal.

The multimodal parity suite should be defined by mesh-advertised capability:
if mesh advertises a skippy-backed model as vision, audio, or multimodal
capable, skippy must load the required projector/media runtime pieces, accept
the matching OpenAI multipart request shape, route by the same capability
semantics, execute successfully, and report the capability/status accurately.

The first required execution targets are:

- Qwen VL / Qwen2.5-VL / Qwen3-VL style image-text models;
- LLaVA-style GGUF plus `mmproj` models as the generic llama.cpp-compatible
  projector baseline;
- Gemma multimodal variants present in the mesh catalog or local inventory.

Audio request/blob plumbing should be covered in API compatibility tests. Audio
model execution becomes required when mesh has a concrete catalog/runtime audio
model path to advertise.

## Runtime Status Parity

Mesh currently polls llama.cpp `/metrics` and `/slots` and maps them into the
dashboard/API. Skippy should not emulate those endpoints internally, but mesh
does need equivalent product state.

Preferred direction:

- introduce a backend-neutral runtime status model in mesh as the public
  response shape immediately;
- map current llama metrics into that model only during migration;
- have skippy publish structured status directly through its handle and
  telemetry stream;
- keep any old llama-shaped fields only as deprecated migration views where
  existing clients need time to move;
- expose skippy-native fields for stage topology and per-stage health from the
  backend-neutral shape.

Status should answer these questions without requiring llama-compatible
endpoints:

- is the model routable, loading, failed, stopping, or stopped?
- which backend device is selected for each local stage?
- how many requests are admitted, active, queued, completed, failed, or
  cancelled?
- what is each stage's readiness, endpoint, layer range, context size,
  activation width, wire dtype, and package identity?
- what are current token counters, throughput estimates, and last error?
- where are the logs and materialized package artifacts?

The public runtime status response should introduce a backend-neutral shape
immediately, for example `runtime.backend`, `runtime.models`, and
`runtime.stages`. Avoid naming new public status fields after llama.cpp. If a
temporary compatibility view is needed, it should be clearly deprecated and
derived from the backend-neutral model.

Current branch status:

- `/api/status` exposes backend-neutral `runtime.backend`, `runtime.models`,
  and `runtime.stages`;
- stage status includes topology/run id, backend, package/source identity,
  materialized artifact path and size when locally visible, pin state,
  projector path, multimodal flag, stage id/index, node id, layer range,
  readiness state, bind address, activation width, wire dtype, selected device,
  context size, last error, and shutdown generation;
- `/api/runtime/stages` exposes the same staged-serving state plus topology
  assignments for dashboard/debug views;
- request counters and token/throughput estimates still come from mesh routing
  metrics and skippy telemetry; they should be promoted into the same
  backend-neutral runtime model before the old llama-shaped metrics view is
  removed.

## Metrics Debug Workflow

Skippy's metrics-server workflow should move into mesh so experimental feature
work can keep the same measure/debug loop:

```text
mesh/skippy stage runtime
    -. best-effort OTLP summaries .->
metrics-server
    -> metrics.duckdb
    -> report.json
```

`metrics-server` is not on the request path. Mesh and skippy stages must keep
running if telemetry export is unavailable, slow, or drops events. The server
owns benchmark/debug ingest, run lifecycle, DuckDB storage, and report export;
stage runtime code owns only best-effort emission and stable `skippy-metrics`
names.

Current branch status:

- `metrics-server` is a workspace crate in mesh;
- `just metrics-server` starts a local collector with HTTP and OTLP/gRPC
  endpoints;
- `.agents/skills/metrics-server` documents the run/debug workflow for agents;
- raw OTLP retention remains an explicit debug mode via
  `--debug-retain-raw-otlp`.
- `docs/SKIPPY_RUNTIME_README.md` preserves the standalone skippy README as
  background context, with a mesh-specific note that standalone `kv-server` and
  `ngram-pool` remain outside this replacement scope.

## PR Readiness Notes

This branch has moved from planning into an integration branch for replacing
mesh's old llama-server/rpc-server serving path with the embedded skippy
runtime. It imports the skippy runtime/protocol/topology crates, introduces
`openai-frontend`, adds the metrics workflow, and routes mesh serving through
the embedded runtime.

Mergeability notes:

- `origin/main` has been merged into this branch during PR preparation.
- Conflict resolution preserved the skippy-owned workspace crates and the
  `openai-frontend` request/response adapter shims over the older mesh-local
  OpenAI compatibility code.
- Main's shell/runtime updates were retained where they do not conflict with
  the skippy backend selector and runtime lifecycle.

Local burn-in checklist for this branch:

- `cargo test -p skippy-server --lib`
- `cargo test -p skippy-protocol --lib`
- `cargo test -p openai-frontend --lib`
- `cargo test -p metrics-server`
- `cargo check -p mesh-llm`
- `cargo test -p mesh-llm inference::skippy --lib`
- `cargo test -p mesh-llm --lib`
- metrics-server fixture ingest/finalize/report export

Legacy reachability audit:

- the `--serving-backend` selector has been removed;
- legacy `rpc-server` startup and direct RPC tunnel/rewrite paths have been
  removed;
- direct GGUFs route through the embedded skippy runtime as single-stage fake
  packages;
- stage splits route through topology planning, downstream `LoadStage`
  readiness, and stage-0 route publication;
- `mesh_hook` routes and `inference::virtual_llm` remain as the Rust-owned hook
  policy surface used by the embedded runtime.
- No standalone `kv-server`, `ngram-pool`, or `ngram-pool-server` crates are
  present in mesh. Mentions in `SKIPPY_RUNTIME_README.md` are imported
  standalone-skippy background, not planned mesh runtime components.

## Stage Failure Recovery

If a stage fails during an active topology, mesh should replan the topology.
The failed topology/run is no longer considered routable.

Required behavior:

- fail in-flight requests that were using the failed topology;
- withdraw the failed topology from routing before cleanup;
- tear down or quarantine surviving stages from that run;
- mark the failed stage/node/package details in runtime status and events;
- build a new topology with a new run id from the remaining eligible peers;
- start downstream/final stages before upstream/driver stages;
- advertise the replacement route only after every required stage is ready.

The first implementation should not attempt in-flight request resume. Stage
state is distributed across per-stage KV, sequence ids, driver state, and
transport links, so correctness is clearer if active generations fail and the
next request uses a fresh topology. Checkpoint/resume can be revisited later as
a separate high-availability feature.

Current branch status:

- active skippy topologies are monitored for stage status loss, refresh
  failures, failed states, stopping states, and stopped states;
- on failure, mesh withdraws the routable target, clears the active HTTP port,
  marks the failed stage/run in backend-neutral status, stops/quarantines the
  old run, and lets the election loop replan with a fresh run id;
- failed peers are temporarily quarantined so the immediate replacement plan is
  built from the remaining eligible peers;
- active generations are not resumed across a topology failure; the active
  request fails and the next request uses the replacement topology once ready.

## Migration Queue

### 1. Import Pure Crates

Copy in and compile:

- `skippy-protocol`
- `skippy-topology`
- `skippy-metrics`
- `openai-frontend`

This PR should not alter serving behavior.

### 2. Add Skippy ABI Build

Bring over:

- `skippy-ffi`
- `skippy-runtime`
- llama.cpp ABI build scripts

Add a mesh build path for the skippy ABI. This step must also add any missing
ABI/config support needed for mesh parity, especially selected device and
multimodal projector loading.

### 3. Make Skippy Embeddable

Refactor `skippy-server` toward host-friendly Rust APIs:

- construct runtime/stage configs from Rust structs;
- start a stage or local driver from a function, not only CLI args;
- return a handle with readiness, status, logs, and explicit shutdown;
- expose model load/unload operations that mesh can call from runtime control;
- expose or pass through selected device, context size, concurrency, and cache
  settings;
- expose structured state for mesh dashboard/runtime status;
- expose synthetic package loading for direct GGUF inputs;
- keep subprocess mode available only as a temporary burn-in fallback.

### 4. Move OpenAI Compatibility Into `openai-frontend`

Extend `openai-frontend` before switching mesh serving:

- move `/v1/responses` request/response translation out of mesh;
- support structured output end to end, including request parsing, constraint
  enforcement through the backend/runtime, non-streaming responses, streaming
  responses, and OpenAI-compatible errors;
- support tool calling end to end, including request parsing, prompt/template
  representation, assistant `tool_calls`, tool result messages, streamed
  tool-call deltas, and OpenAI-compatible errors;
- support logprobs end to end, including token scoring from the backend/runtime,
  chat logprobs, completion logprobs, top logprobs, streaming responses, and
  non-streaming responses;
- keep shared fixtures for chat, completions, responses, streaming, errors,
  tool calls, structured output, and logprobs.

There is no partial compatibility mode for these features in the replacement
plan. If a client can use structured output, tool calls, or logprobs through
the current serving surface, the skippy replacement must preserve that behavior
or return the same class of OpenAI-compatible error for invalid requests.

Mesh-specific behavior, such as blob resolution, routing policy, request
affinity, model selection, and object-store cleanup, should remain in mesh.

Mesh should not lose its optimized OpenAI ingress path. Today mesh reads one raw
HTTP request, extracts lightweight metadata for routing, and only parses the
full JSON body when compatibility translation, blob resolution, or local
execution requires it. `openai-frontend` should preserve that shape by exposing
reusable parser/normalizer/response-adapter primitives in addition to any Axum
router it provides. Mesh's public ingress should be able to:

- inspect model/session/token-budget metadata without fully deserializing
  messages;
- forward raw request bytes unchanged for remote/plugin targets when no
  normalization is needed;
- parse and rewrite the JSON body only once when `/v1/responses`,
  max-token aliases, structured output, tool calls, logprobs, or multimodal
  blobs require normalization;
- hand a parsed request directly to a local skippy backend without reparsing;
- stream responses through the shared adapters without buffering the full
  response body.

Embeddings can be deferred for now. The replacement plan does not need
`/v1/embeddings` parity before removing `llama-server`.

Current branch status:

- `openai-frontend` owns `/v1/chat/completions`, `/v1/completions`, and
  `/v1/responses` request/response shapes, streaming SSE adapters, OpenAI error
  bodies, tool-call fields, structured-output fields, and logprob fields;
- frontend fixture coverage accepts and translates tool, structured-output,
  logprob, streaming, and responses requests without mesh carrying a second
  public OpenAI model;
- mesh ingress still keeps the raw-byte fast path and only normalizes when
  compatibility translation or blob rewriting requires it;
- skippy runtime execution now returns OpenAI-compatible unsupported-feature
  errors for structured-output constraints, tool-call execution, and logprob
  scoring until runtime support lands, so requested behavior is not silently
  ignored.

### 5. Add `inference/skippy`

Create a mesh-owned backend module under `crates/mesh-llm/src/inference/skippy/`.
Start with single-node serving and make this the path that grows to full
replacement parity.

The first target is text parity through mesh's existing routing, runtime
control, and backend rate limiting. Use `openai-frontend` as the shared
request/response and backend trait layer rather than adding another mesh-local
OpenAI model of the world.

This module is also where mesh's lifecycle semantics are preserved. It should
adapt mesh runtime-control requests into skippy load/unload/reload operations
and make staged backends look like any other mesh backend to the router,
dashboard, and management API.

### 6. Add Direct GGUF Package Materialization

Make the skippy package abstraction cover existing mesh model inputs:

- direct local GGUF;
- split GGUF;
- catalog assets;
- Hugging Face GGUF references;
- future layer-package references.

Direct GGUF should materialize into a synthetic package manifest for local
runtime loading and status provenance. Multi-stage serving should require a
package-backed source or an explicit materialization step; it should not assume a
local `--gguf` path exists on remote workers.

### 7. Add Stage Coordination

Add an additive, versioned mesh stream/control protocol for remote stage
commands.

The command should include:

- schema version;
- topology id and run id;
- public model id and resolved package identity;
- package manifest hash;
- assigned layer range;
- stage id and stage index;
- selected package files or file patterns;
- activation width and wire dtype;
- bind/listen information;
- shutdown/reload generation.

The stage protocol is new and does not need backward compatibility. Keep it out
of `mesh-llm/1` so mixed-version mesh membership, gossip, routing, and public
OpenAI proxy traffic continue to interoperate even when a peer cannot
participate in skippy stage execution.

### 8. Wire Stage Transport

Stage traffic uses the `skippy-stage/1` QUIC ALPN. Each accepted stage
connection carries skippy-owned stream kinds for control and activation
transport, then length-prefixed `skippy-protocol` protobuf frames. The
activation transport switches to raw activation bytes after the
`StageTransportOpen` frame.

### 9. Replace Split Serving

Use `skippy-topology` and mesh peer capacity data to replace the old dense RPC
split path.

Important policy changes:

- layer packages with split demand should use skippy topology, not RPC;
- high RTT should be evaluated as a stage pipeline cost, not as an RPC hard
  cutoff;
- each stage owns its own layer range and KV for that range;
- family split safety should come from skippy's reviewed/inferred
  `FamilyCapabilityRecord` data, including recurrent ranges, split
  constraints, sideband requirements, exact-state mobility, and wire dtype
  validation;
- stage assignments and readiness should be visible in status/gossip.

### 10. Preserve Auto LLM Hooks

Move the patched `llama-server` hook behavior into the mesh/skippy path.

Implementation checkpoints:

- [x] route `model=auto` skippy requests through the same hook policy as legacy
  requests;
- [x] call the media fallback hook before tokenization when the selected model
  cannot consume attached media;
- [x] call post-prefill uncertainty and mid-generation drift hooks using runtime
  signals supplied by skippy;
- [x] preserve `inference::virtual_llm` behavior and peer consultation
  semantics;
- [x] keep `/mesh/hook` available as the local Rust-owned hook policy route;
- [x] add fixture tests for `mesh_hooks` injection, recursion guard, media
  fallback, uncertainty hint, drift hint behavior, and debug forcing controls.

### 11. Remove Old Serving Paths

After lifecycle, device pinning, OpenAI compatibility, direct GGUF packages,
staged serving, runtime status, multimodal parity, and Auto LLM hook parity are
proven:

- [x] delete `rpc-server` launch and pidfile paths from the serving runtime;
- [x] delete RPC port rewrite;
- [x] replace patched `llama-server` mesh hooks with Rust-owned skippy hook
  points;
- [x] delete legacy expert split serving from request serving;
- [x] simplify election around topology planning and backend target readiness;
- [ ] retire stale legacy build/test/docker artifacts that only existed to
  package external `llama-server` or `rpc-server` binaries.

## Data Model Additions

Mesh needs explicit staged-serving state. Suggested concepts:

```rust
struct StageTopologyInstance {
    topology_id: String,
    run_id: String,
    model_id: String,
    package_ref: String,
    manifest_sha256: String,
    stages: Vec<StageAssignment>,
}

struct StageAssignment {
    stage_id: String,
    stage_index: u32,
    node_id: iroh::EndpointId,
    layer_start: u32,
    layer_end: u32,
    endpoint: StageEndpoint,
}

struct StageRuntimeStatus {
    topology_id: String,
    run_id: String,
    stage_id: String,
    status: StageStatus,
    context_length: Option<u32>,
    activation_width: Option<u32>,
    wire_dtype: Option<String>,
    selected_device: Option<String>,
    backend_pid: Option<u32>,
    log_path: Option<PathBuf>,
}

struct SkippyModelRuntime {
    model_id: String,
    public_model_ref: String,
    status: BackendStatus,
    selected_device: Option<String>,
    context_length: Option<u32>,
    parallelism: usize,
    topology: Option<StageTopologyInstance>,
}
```

This should be additive to existing gossip. Older nodes must continue to
operate using the existing protocol.

## Near-Term Recommendation

The next PR should import the pure skippy crates and compile them in the mesh
workspace without changing runtime behavior. After that, bring in the ABI build,
runtime wrapper, device-selection extension, and direct-GGUF synthetic package
path. Only then should mesh start a skippy backend, first single-node and then
multi-node.
