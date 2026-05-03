# llama-stage-runtime Integration Plan

## Purpose

This document captures the planned integration of
`/Users/jdumay/code/llama-stage-runtime` into mesh-llm.

The goals are:

- replace the llama.cpp fork workflow with an upstream llama.cpp pin plus a
  local patch queue
- keep the same backend and performance envelope as the current external
  llama.cpp binaries across CPU, Metal, CUDA, Vulkan, and ROCm
- statically link llama.cpp behind a C ABI where possible
- add an in-process llama backend for mesh-llm, eventually replacing the
  external `llama-server` path for local serving
- make it easy to move the llama-stage-runtime source tree into this repo later

## Current State

mesh-llm's historical baseline depended on a pinned Mesh-LLM llama.cpp fork.
The first migration step converts those fork commits into a local patch queue.
Those patches cover:

- llama.cpp RPC optimizations
- MoE expert analysis and splitting tools
- mesh hooks used by the virtual LLM path

Runtime orchestration is process based:

- `rpc-server` is launched for local worker compute
- `llama-server` is launched as the local OpenAI-compatible inference server
- mesh-llm proxies local and remote requests to those HTTP servers
- QUIC tunnels carry HTTP traffic and, for dense split mode, llama.cpp RPC traffic

`llama-stage-runtime` provides a separate proof of the same source management
model:

- upstream `ggml-org/llama.cpp` is pinned in
  `third_party/llama.cpp/upstream.txt`
- patches live in `third_party/llama.cpp/patches/*.patch`
- `just llama-prepare` checks out upstream and applies patches
- `just llama-build` builds static llama.cpp archives
- Rust crates link the patched static libraries through `llama-stage-ffi`

## Feasibility Summary

The patch-queue build migration is the first step and intentionally excludes
the llama-stage ABI patches.

The in-process runtime migration is also feasible, but it is not a drop-in
replacement for `llama-server`. mesh-llm currently relies on behavior owned by
`llama-server`, including:

- OpenAI-compatible HTTP endpoints
- streaming response framing
- chat templates and request formatting
- sampling options and defaults
- slot scheduling and parallel request handling
- multimodal `mmproj` handling
- reasoning format and reasoning budget flags
- speculative decoding
- mesh hook callbacks
- current dense split integration over llama.cpp RPC

The embedded path should therefore be introduced as a second backend behind an
internal abstraction, then promoted only after it reaches compatibility and
performance parity.

## Design Constraints

### Performance Parity Is Mandatory

Embedding llama.cpp must not become a CPU/Metal-only shortcut. It must preserve
the current performance envelope for all release targets:

| Target | Requirement |
| --- | --- |
| CPU | Same portable CPU build behavior as release artifacts |
| Metal | Same Apple Silicon Metal acceleration and static shader behavior |
| CUDA | Same CUDA backend flags, including release-safe Flash Attention quant support |
| Vulkan | Same Vulkan backend selection and shader/toolchain expectations |
| ROCm | Same HIP/ROCm backend flags and architecture targeting |

The embedded build must reuse the same backend matrix as the current external
binary build scripts. In particular, CUDA release builds must retain
`GGML_CUDA_FA_ALL_QUANTS=ON`, because mesh-llm intentionally uses asymmetric KV
cache quantization for mid-size models.

### Source Migration Should Stay Easy

The first integration should preserve the llama-stage source shape rather than
folding it into unrelated mesh-llm modules. That keeps a later move from
`/Users/jdumay/code/llama-stage-runtime` into this repository mechanical.

Preferred eventual layout:

```text
llama-stage/
  crates/
    llama-stage-ffi/
    llama-stage-runtime/
    llama-stage-protocol/
    llama-stage-server/
    llama-model-slice/
    llama-stage-correctness/
    llama-stage-metrics/
    llama-stage-bench/
    metrics-server/
  scripts/
    prepare-llama.sh
    build-llama.sh
    update-llama-pin.sh
  third_party/
    llama.cpp/
      upstream.txt
      patches/
```

mesh-llm can then depend on `llama-stage-*` crates by workspace path.

### Protocol Compatibility Still Applies

The mesh protocol must remain compatible across versions. Introducing an
embedded backend should not change gossip, OpenAI routing, QUIC tunnel stream
types, or node role semantics until explicit compatibility handling exists.

If staged activation transport later replaces llama.cpp RPC for dense split
mode, that should be designed as an additive capability advertised through
gossip, with fallback to the current external RPC split path.

## Proposed Architecture

Introduce an inference backend abstraction owned by `crates/mesh-llm/src/inference/`.

```text
InferenceBackend
  ExternalLlamaBackend
    launches rpc-server and llama-server
    preserves current behavior

  EmbeddedLlamaBackend
    loads patched static llama.cpp through llama-stage-runtime
    exposes an internal OpenAI-compatible local service surface
    initially supports a narrow text-only local serving path
```

The existing process backend remains the default until the embedded backend
passes compatibility and performance gates.

For early integration, keep mesh-llm's routing and tunneling model stable:

- local requests still target a local backend port or equivalent local handle
- remote requests still tunnel through QUIC using the existing HTTP path
- dense split mode continues using external llama.cpp RPC unless explicitly
  running an experimental staged-runtime split
- MoE sharded serving continues using current shard artifacts and process
  orchestration until the embedded backend can serve those shards equivalently

## Build System Plan

### Phase 1: Adopt Upstream Pin Plus Patch Queue

Add llama patch queue files to the mesh repo:

```text
third_party/llama.cpp/upstream.txt
third_party/llama.cpp/patches/
```

Convert the current Mesh-LLM llama.cpp fork commits into ordered patches. Add
or reconcile the llama-stage-runtime ABI patches in a later pass.

Patch groups should stay reviewable:

```text
0001-00xx  mesh RPC patches
00xx-00xx  mesh MoE patches
00xx-00xx  mesh hook patches
00xx-00xx  llama-stage ABI/runtime patches
```

Build scripts should prepare llama.cpp by:

1. reading `third_party/llama.cpp/upstream.txt`
2. cloning/fetching upstream `ggml-org/llama.cpp`
3. checking out the pinned upstream SHA
4. applying patches with `git am --3way`
5. writing prepared and patched SHAs for diagnostics

The existing `LLAMA_CPP_SHA` file is kept temporarily as a compatibility mirror
of `upstream.txt` while release and CI scripts are updated.

### Phase 2: Use One Backend Build Matrix

Refactor the current platform scripts so external and embedded builds share the
same backend flag decisions.

The shared configuration should cover:

- backend selection: CPU, Metal, CUDA, Vulkan, ROCm
- architecture flags: CUDA SM list and ROCm gfx list
- `BUILD_SHARED_LIBS=OFF`
- `LLAMA_OPENSSL=OFF` or equivalent current llama option
- `GGML_RPC=ON` while the external RPC path is still shipped
- CUDA Flash Attention quant flags
- release portability flags such as Linux `GGML_NATIVE=OFF`
- compiler cache configuration

The embedded static ABI build should not grow a separate, divergent CMake flag
set. If the external CUDA build gets a safety flag, the embedded CUDA build
should inherit it.

### Phase 3: Link llama-stage Crates

Initially, use a workspace path dependency while preserving source shape:

```toml
llama-stage-runtime = { path = "../llama-stage/crates/llama-stage-runtime" }
```

If the source has not yet moved into this repo, a temporary local path to
`/Users/jdumay/code/llama-stage-runtime` can be used only for experimental
branches. Mainline should not depend on an absolute local path.

## Runtime Migration Plan

### Milestone 1: External Behavior Preserved

Replace the fork checkout flow with upstream pin plus patches while keeping the
current external `rpc-server` and `llama-server` runtime.

Acceptance criteria:

- `just build` works on the current platform
- release scripts still package flavor-specific external binaries
- `cargo check -p mesh-llm` passes
- existing local solo serving works
- no mesh protocol changes

### Milestone 2: Embedded Backend Skeleton

Add an embedded backend behind a feature flag or hidden CLI/config option.

The skeleton should:

- statically link the patched llama.cpp ABI through `llama-stage-runtime`
- load a local GGUF model in process
- expose health/readiness to mesh-llm's existing state machinery
- keep current external backend as the default
- be easy to disable at build time if a platform link issue appears

Acceptance criteria:

- a small local model can be loaded and unloaded in process
- lifecycle state appears in `/api/status`
- process cleanup still works when external llama processes are not present

### Milestone 3: Text-Only Local Serving

Implement a local OpenAI-compatible shim over the embedded backend for a narrow
text-only path.

Start with:

- `/v1/models`
- non-streaming `/v1/chat/completions`
- basic prompt formatting
- tokenization through the llama C ABI
- greedy or minimal sampling only as an experimental mode

Do not claim parity with llama-server yet.

Acceptance criteria:

- same mesh request routing can target external or embedded local backend
- remote clients can reach the embedded local backend through existing QUIC HTTP
  tunnel behavior
- errors map into the existing OpenAI transport error handling

### Milestone 4: Request and Sampling Compatibility

Close the gap with the llama-server behavior that users rely on:

- streaming SSE responses
- common OpenAI sampling fields
- repeat penalty defaults
- max token handling
- stop sequences
- chat template behavior
- reasoning fields and defaults
- slot/parallel request limits matching `BACKEND_PROXY_MAX_INFLIGHT`

Acceptance criteria:

- representative clients such as Goose, Claude Code via OpenAI-compatible mode,
  curl, and the console chat path behave the same as with external llama-server
- concurrency is bounded and does not create an unbounded deferred queue
- output quality and latency are comparable for the same model and backend

### Milestone 5: Backend Performance Parity

Run side-by-side performance checks for external vs embedded on every backend
that mesh-llm ships:

- CPU
- Metal
- CUDA
- Vulkan
- ROCm

Measure at minimum:

- model load time
- time to first token
- tokens per second
- memory usage and VRAM residency
- behavior with current KV cache quantization defaults
- behavior under concurrent requests

Acceptance criteria:

- embedded performance is within an agreed tolerance of external llama-server
  for each supported backend
- any backend-specific regression is documented and either fixed or explicitly
  excluded from embedded support for that release

### Milestone 6: Multimodal, MoE, and Mesh Hooks

Port higher-level behavior after the basic text path is stable:

- `mmproj` and image/audio request handling
- MoE shard serving behavior
- current mesh hook behavior used by virtual LLM
- speculative decoding
- dense split mode replacement or coexistence with staged activation transport

Acceptance criteria:

- existing multimodal tests and manual smokes pass
- MoE split artifacts serve correctly through embedded mode
- virtual LLM behavior has either equivalent Rust-owned hook points or an
  explicit reason to stay on external llama-server

## Dense Split Strategy

Do not replace llama.cpp RPC split mode as part of the first embedded milestone.

The current dense split path is mature enough to preserve while the embedded
backend proves local serving. Later, staged activation transport can be
introduced as a new capability:

```text
current:
  llama-server -> llama.cpp RPC -> rpc-server

future experimental:
  mesh-llm embedded stage -> QUIC stage transport -> mesh-llm embedded stage
```

The future path should be additive and negotiated. Older nodes and nodes without
embedded support must keep working through the existing external RPC path.

## Risks

### ABI Surface Is Lower Level Than llama-server

`llama-stage-runtime` exposes model/session/token/activation primitives. It does
not currently provide all llama-server product behavior. mesh-llm will need a
compatibility layer, not just an FFI call site.

### Static Linking Varies By Backend

CPU and Metal static linking are already close to the current
llama-stage-runtime shape. CUDA, ROCm, and Vulkan need explicit validation
because the Rust binary will inherit backend runtime library requirements that
were previously attached to standalone llama.cpp binaries.

### Build Matrix Drift

If external and embedded builds use separate CMake flag logic, performance and
bug fixes will diverge quickly. A shared backend flag generator is the safest
path.

### Mesh Hook Ownership

The current mesh hook behavior is implemented inside patched llama-server code.
Embedding is a chance to move those hooks into Rust-owned request orchestration,
but that requires careful behavior matching.

### Protocol Compatibility

Any replacement for RPC split mode touches distributed execution semantics.
It must be advertised as a capability and rolled out with fallback behavior.

## Recommended Near-Term PRs

1. Add upstream pin and patch queue scaffolding without changing runtime
   behavior.
2. Convert current fork commits into patches and prove `just build` still works.
3. Import llama-stage crates under a preserved `llama-stage/` subtree.
4. Refactor build scripts so backend flags are shared by external and embedded
   builds.
5. Add an experimental embedded backend skeleton behind a disabled-by-default
   flag.
6. Add benchmark fixtures comparing external and embedded execution per backend.

## Validation Checklist

Before embedded mode can become default for any backend:

- `just build` succeeds for that backend
- `cargo check -p mesh-llm` succeeds
- local solo inference succeeds
- `/v1/models` and `/v1/chat/completions` behave like external llama-server
- streaming behavior matches external llama-server
- concurrency limits are enforced
- memory usage is comparable
- token throughput is comparable
- released binary compatibility is preserved for mixed-version meshes
- fallback to external backend remains available
