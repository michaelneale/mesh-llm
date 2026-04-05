# Inference Plugin Migration Plan

This plan turns the inference plugin architecture note into an implementation sequence.

The intent is to move backend-specific runtime behavior into plugins without moving mesh-wide planning and routing policy out of `mesh-llm`.

It assumes the pure plugin DSL from [PLUGINS.md](/Users/jdumay/.codex/worktrees/59c2/mesh-llm/PLUGINS.md):

- `provides`
- `mcp`
- `http`
- `inference`

Inference backends should register through `inference`, not through ad hoc host-specific wiring.

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

In the plugin DSL, this should map cleanly to `inference` contributions for both:

- attached external endpoints
- plugin-hosted backends

### `MoeRankingProvider`

Optional backend-specific ranking/analysis provider:

- inspect backend-specific model topology
- generate ranking artifacts
- expose provenance and strategy metadata

If ranking utilities are projected outward for humans or tooling, those should appear under `mcp` or `http`, not as a separate top-level plugin concept.

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

That provider layer should already be shaped so the eventual plugin contribution is just an `inference` entry, not a separate bespoke registration path.

### Phase 4: move local llama runtime behind the provider

Replace direct local `llama-server` launch calls with the provider contract.

Primary target:

- `runtime/local.rs`

### Phase 5: port MLX onto the same provider contract

Keep the same local endpoint semantics:

- local OpenAI-compatible HTTP surface
- host-owned proxy remains unchanged

The MLX plugin should be a clean proof that the `inference` section is sufficient for a plugin-hosted backend.

### Phase 6: move distributed llama runtime behind the provider

Replace direct distributed launch logic with provider-owned execution:

- host `llama-server`
- worker `rpc-server`
- backend-specific launch flags

Core election still decides placement.

The resulting llama plugin should still read as one coherent plugin with:

- `inference` for the runtime backend
- optional `mcp` or `http` projections for debugging or operator tooling
- `provides` for any stable contracts we want core to depend on

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

1. Replace the remaining built-in llama execution path with the new plugin-managed llama plugin where the branch is ready.
2. Continue splitting MoE ranking generation from host policy on the MoE sync branch.
3. Mirror neutral launch-setting and provider-registry improvements across the sync branches as they become ready.
4. Keep the inference-provider seam aligned with the pure plugin model described in `PLUGINS.md` without prematurely porting implementation work onto the plugin branch.

## Current Status

The following no-behavior-change groundwork is already in place on this branch:

- `InferenceServerHandle` and `InferenceServerProcess` now live in `inference/provider.rs`
- local and distributed llama endpoint launch sites now build `InferenceEndpointRequest`
- worker helper launch sites now build `InferenceWorkerRequest`
- a built-in `BuiltinLlamaProvider` adapter now owns the call from core into llama launch code
- provider selection now goes through a named built-in provider seam instead of hard-coding llama at the orchestration call sites
- this MLX sync branch now also has a built-in `BuiltinMlxProvider` for local endpoint startup
- provider selection now returns an explicit selection object with provider id, label, and capabilities rather than a bare runtime handle
- provider lookup now goes through a small registry-shaped seam rather than direct selector logic, so plugin-backed providers can slot in later without another call-site rewrite
- the provider registry can now admit explicitly registered non-built-in providers, with registered providers taking precedence over the built-in fallback descriptors
- provider requests can now carry an explicit preferred provider id, so future plugin-backed backends can be selected by descriptor instead of only by model-path matching
- the shared inference layer now has a host-side plugin registration adapter that turns a plugin-style inference registration into a preferred-only provider descriptor
- the provider contract now advertises explicit capabilities so orchestration can ask what a backend supports instead of inferring it indirectly
- worker-runtime startup is now gated by provider capabilities instead of being assumed unconditionally in the shared runtime path
- MoE GGUF detection and cached-ranking lookup now route through a backend-facing `MoeRankingProvider` on the provider seam, so ranking generation can move out of core without rewriting election policy
- plugin-managed inference endpoints can now be surfaced from plugin manifests and synced into the provider registry as managed provider descriptors
- plugin-managed endpoint launch now goes through a real host/plugin handshake (`inference/ensure_endpoint`) instead of stopping at registration-only descriptors
- plugin-managed worker launch now goes through a matching host/plugin handshake (`inference/ensure_worker`)
- plugin-managed MoE shard preparation now has a host/plugin handshake (`inference/prepare_moe_shard`) and the provider contract uses it instead of calling split logic directly
- managed provider manifests can now declare selection metadata and provider capabilities, so host registration no longer hard-codes those properties
- managed-provider model-path matchers now apply across local runtime, distributed-host runtime, and worker-runtime selection, not just local runtime
- plugin initialization now carries host launch hints such as resolved `bin_dir` and preferred binary flavor, so managed inference plugins can launch backend runtimes without rediscovering host policy on their own
- this MLX sync branch now has a built-in plugin-managed llama provider scaffold that:
  - declares a managed GGUF-backed inference endpoint
  - serves `inference/ensure_endpoint`
  - serves `inference/ensure_worker`
  - serves `inference/prepare_moe_shard`
  - registers into the runtime provider registry and overrides the built-in llama provider for GGUF local, distributed, and worker selection

## Sync Branches

This work should be mirrored across three architecture branches:

- `codex/mlx-llama-plugin-architecture`
- `codex/moe-plugin-architecture-sync`
- `codex/mlx-plugin-architecture-sync`

The rule is:

- design notes stay identical where possible
- neutral contract extractions should be mirrored
- backend-specific follow-up work can diverge after the shared contract layer is in place
