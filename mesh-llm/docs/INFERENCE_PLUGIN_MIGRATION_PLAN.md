# Inference Plugin Migration Plan

This plan turns the inference plugin architecture note into an implementation sequence.

The intent is to move backend-specific runtime behavior into plugins without moving mesh-wide planning and routing policy out of `mesh-llm`.

## Scope

This plan covers:

- the current local and distributed llama runtime path
- the MLX runtime shape from PR `#103`
- the richer MoE planning and ranking policy work from `codex/moe-ranking-benchmark`

It does not attempt to move the entire inference subsystem into plugins.

## Architectural Split

### Core remains responsible for

- mesh election
- inference target routing
- endpoint inventory and health
- served-model advertisement
- MoE placement planning
- ranking policy selection
- fallback and failover planning

### Plugins become responsible for

- backend runtime startup and shutdown
- model loading
- backend-specific local serving endpoints
- backend-specific worker helpers
- backend-specific analysis tooling

## Contracts

We will move toward two backend-facing contracts.

### `InferenceEndpointProvider`

Owns serving/runtime execution:

- start local runtime
- start distributed host runtime
- start distributed worker runtime
- stop runtime
- describe runtime endpoint and context

### `MoeRankingProvider`

Optional backend-specific ranking/analysis provider:

- inspect backend-specific model topology
- generate ranking artifacts
- expose provenance and strategy metadata

Llama is expected to implement both contracts eventually.

MLX is expected to implement `InferenceEndpointProvider` first.

## Phase Plan

### Phase 1: extract backend-neutral runtime contract

Create a backend-neutral home for:

- inference server handle
- inference server process
- backend runtime request/response types as needed

This is a no-behavior-change refactor that makes `launch.rs` llama-specific again.

### Phase 2: define provider-facing request shapes

Add explicit request structs for:

- local runtime start
- distributed host runtime start
- distributed worker runtime start

These are still host-owned types. No plugin wiring yet.

### Phase 3: introduce provider adapters in core

Add a host-side provider dispatch layer that can:

- call the built-in llama implementation
- later call plugin-backed providers

At this phase, llama can still be implemented in-process in core behind the provider adapter.

### Phase 4: move local llama runtime behind the provider

Replace direct local `llama-server` launch calls with the provider contract.

Primary target:

- `runtime/local.rs`

### Phase 5: port MLX onto the same provider contract

Keep the same local endpoint semantics:

- local OpenAI-compatible HTTP surface
- host-owned proxy remains unchanged

### Phase 6: move distributed llama runtime behind the provider

Replace direct distributed launch logic with provider-owned execution:

- host `llama-server`
- worker `rpc-server`
- backend-specific launch flags

Core election still decides placement.

### Phase 7: split MoE ranking generation from MoE ranking policy

Keep in core:

- ranking strategy choice
- plan replacement rules
- fallback coverage logic
- routing table updates

Move toward provider ownership for:

- `llama-moe-analyze`
- GGUF/backend-specific ranking artifact generation

## Immediate Tasks

The next concrete tasks are:

1. Extract the backend-neutral runtime handle/process contract.
2. Add request structs for local and distributed runtime startup.
3. Switch local runtime code to import the provider contract rather than `launch.rs`.
4. Sketch the built-in llama provider adapter without changing behavior yet.

## Sync Branches

This work should be mirrored across three architecture branches:

- `codex/mlx-llama-plugin-architecture`
- `codex/moe-plugin-architecture-sync`
- `codex/mlx-plugin-architecture-sync`

The rule is:

- design notes stay identical where possible
- neutral contract extractions should be mirrored
- backend-specific follow-up work can diverge after the shared contract layer is in place
