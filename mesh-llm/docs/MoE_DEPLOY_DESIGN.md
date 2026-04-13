# MoE Auto-Deploy — Implementation Notes

This documents how MoE expert sharding is implemented in mesh-llm. Originally a design doc; updated to reflect the actual implementation.

## User Experience

```bash
# MoE auto-detected — splits if needed, runs solo if it fits
mesh-llm serve --model Qwen3-30B-A3B-Q4_K_M

# Hidden test/debug override: force splitting even if the model fits locally
mesh-llm serve --model Qwen3-30B-A3B-Q4_K_M --split
```

The system detects MoE from the GGUF header, computes expert assignments, splits the GGUF per node, and each node runs its own llama-server. Sessions are hash-routed. No manual steps.

Normal operators should use plain `mesh-llm --model ...` or `--auto`.

- `--split` remains a hidden test/debug override for forcing the MoE path.
- `--max-vram` is a supported resource-budget knob when you want the planner to behave as if less VRAM is available.

## Planned Deployment Policy

The current runtime uses **leader-planned automatic MoE deployment**.

That means:

- the leader computes one deployment plan for the exact model identity currently being served
- nodes advertise resources and health, not their own preferred MoE strategy
- `auto` is the normal path; the mesh decides whether to run solo, split, or split with extra redundancy
- nodes either participate in the chosen plan or do not join that split

The deployment plan should choose:

- participating nodes
- shard count
- ranking source
- overlap / redundancy level
- whether a full-coverage fallback replica is feasible

## Failure Handling

The goal is graceful degradation, not perfect steadiness on a broken topology.

- **Active shard failure should fail down quickly.**
  If a request to an active shard fails, the deployment should be treated as invalid and reconfigured across survivors.
- **Heartbeat is supporting evidence, not the main signal.**
  Heartbeats are useful for liveness, but request-path failure matters more because it directly affects serving.
- **Do not retry to another partial shard by default.**
  Partial shards are not interchangeable.
- **Retry directly only to a full-coverage target.**
  If a node can serve the full expert set for the same exact model identity, it is a safe failover target.

## Recovery Handling

- **Recover up cautiously.**
  When a lost node comes back, it should rejoin the mesh first but remain out of active MoE placement until it has stayed healthy for a short probation window.
- **Use a quiet window before scale-up.**
  After probation, the leader should still wait for a short quiet period before reconsidering a larger topology.
- **Only expand when the plan is materially better.**
  A recovered node being reachable again is not enough by itself. The larger plan should improve fallback coverage, active capacity, or redundancy enough to justify a rebuild.
- **Avoid topology flapping.**
  The system should fail down faster than it scales back up.
- **Recompute when membership changes.**
  As nodes join or leave, the leader should re-plan the deployment based on the current healthy set and the available resource budget.

## Redundancy Handling

If the cluster has spare capacity, the leader should use it to improve resilience automatically rather than exposing manual runtime knobs.

Possible outcomes:

- extra overlap
- replicated hot experts
- a full-coverage fallback replica

The choice should come from the observed cluster resources and health, not conflicting per-node flags.

## Implementation

### Step 1: Detect MoE (`moe.rs`)

`detect_moe(path)` reads the GGUF header — looks for `*.expert_count` and `*.expert_used_count` KV pairs. Returns `GgufMoeInfo` or None. Takes ~1ms.

### Step 2: Decide solo vs split (`election.rs`)

In `election_loop()`, after detecting MoE via `lookup_moe_config()`:

```
if model.is_moe && (force_split || !model_fits_locally) && node_count >= 2:
    → moe_election_loop() — each node gets a different shard GGUF
else:
    → normal election — solo or tensor split
```

`lookup_moe_config()` checks the current ranking sources in descending quality:
1. **published full analyze** from `meshllm/moe-rankings`
2. **cached full analyze**
3. **published micro-analyze** from `meshllm/moe-rankings`
4. **cached/imported micro-analyze**
5. **peer-first micro-analyze on cold start**
6. **sequential fallback**

Published rankings are downloaded through the normal Hugging Face cache and remain there. mesh-llm does not copy dataset artifacts into its own cache.

### Step 3: Compute assignments (`moe.rs`)

`compute_assignments(ranking, n_nodes, min_experts)`:
- Shared core = top `min_experts` by gate mass (replicated to every node)
- Remaining experts distributed round-robin across nodes
- Returns `Vec<NodeAssignment>` — each has `experts`, `n_shared`, `n_unique`

The leader now chooses placement automatically from cluster resources instead of exposing MoE split-planning knobs at runtime. The current planner keeps a healthy active shard set stable, uses overlap-based redundancy in the active split, and reserves a full-coverage fallback replica when there is enough spare capacity.
Recover-up is intentionally conservative: a recovered node must pass probation, then a quiet window, and then the leader only scales back up if the candidate plan is materially better than the current healthy one.

### Step 4: Split GGUF (`moe.rs` → `llama-moe-split`)

`run_split()` calls the external `llama-moe-split` tool with `--expert-list`. Produces a self-contained GGUF: full trunk + selected experts + adjusted router gates + updated metadata.

Splits are cached at `~/.cache/mesh-llm/splits/<model>/<n>-nodes/node-<i>.gguf`. Invalidated implicitly when node count changes (different directory).

### Step 5: Independent llama-servers

Each node runs `llama-server` with its split GGUF. No `--rpc`, no tensor splitting. Each node is fully independent with its own KV cache.

`moe_election_loop()` manages the lifecycle: start, restart on mesh changes, kill on shutdown.

### Step 6: Session routing (`proxy.rs` + `election.rs`)

`extract_session_hint()` parses `user` or `session_id` from the request body. `get_moe_target()` hashes it to pick a node. `MoeLocal`/`MoeRemote` targets handle local vs QUIC-tunneled forwarding.

## What's NOT implemented from the original design

- **Shard distribution over QUIC** — the design proposed pushing shards from host to workers. Instead, every node splits locally from its own copy of the full GGUF. Simpler, but requires every node to have the full model on disk.
- **Probe-based placement** — hash routing is used instead. Both nodes are equivalent with sufficient overlap.
- **Global redundancy optimization** — the current leader plan is intentionally conservative and deterministic. It does not yet search all possible shard/fallback layouts.
- **Probe-based placement** — hash routing plus full-coverage failover is still used instead of prompt probing.

## Open Questions (from original design, still open)

1. **Node count changes**: Re-splitting when a 3rd node joins a 2-node mesh. Currently handled — `moe_election_loop` detects the change, re-computes assignments, re-splits if the new split doesn't exist in cache.
2. **Minimum viable calibration per model**: The 50% default is conservative. Different models may need more or less. Only Qwen3-30B-A3B has been properly calibrated (36% = 46/128 experts).
3. **Can we skip `moe-analyze`?** The 50% fallback works but wastes storage. Gate norms from GGUF weights (no inference needed) might give a cheap approximation of expert importance.

## Future Direction: Descriptor-Carried MoE Topology

As mesh-llm moves toward protocol-level `ServedModelDescriptor` objects, MoE should stop depending only on local GGUF inspection and instead consume `ModelTopology.moe` when it is available.

Planned source priority:

1. published `moe-analyze` data in `meshllm/moe-rankings`
2. local cached `moe-analyze` output for that exact descriptor identity
3. Hugging Face metadata for exact `repository + revision + artifact`
4. GGUF header fallback

This gives us two important properties:

- **revision-aware grouping**: nodes can confirm they are serving the same exact MoE snapshot before coordinating
- **clean future analysis flow**: `moe-analyze` can improve topology later without changing the contract for how topology is identified

Contribution flow:

- use `mesh-llm moe analyze full` or `mesh-llm moe analyze micro` to generate local rankings
- use `mesh-llm moe share` to open contribution PRs against `meshllm/moe-rankings`
- when `serve` had to generate a local ranking because no published one was available, it should suggest `mesh-llm moe share <model>`

Longer term, we may also use lighter-weight local signals such as short warm-up inference or recent router statistics to improve unknown models. That data should remain explicitly lower-confidence than `moe-analyze`, and should be treated as a hinting/calibration layer rather than the canonical topology source.
