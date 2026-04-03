# MoE Expert Sharding — Design & Status

Distribute MoE models across mesh nodes using **overlapping expert shards** with zero cross-node inference traffic. Each node holds the full trunk plus a subset of experts. Sessions are hash-routed to nodes.

See [ROADMAP.md](../../ROADMAP.md) for how this fits into mesh-llm.

## What's Implemented

All core phases are complete and integrated into mesh-llm.

### Detection (`moe.rs`)
- `detect_moe()` reads `expert_count` from GGUF header in ~1ms. Any MoE model works — no catalog entry needed.
- Auto-detected in `election.rs` at model load time.

### Ranking
- **Cached rankings**: `moe::load_cached_ranking()` loads from the mesh-llm cache under `~/.cache/mesh-llm/moe-rankings/`.
- **Dynamic analysis**: runtime can materialize cached rankings via `micro-analyze` or full `moe-analyze`.
- **Fallback**: no ranking → conservative 50% shared core with sequential expert IDs.
- **Tool**: `llama-moe-analyze` (in `llama.cpp/tools/moe-analyze/`) runs inference on sample prompts and exports per-expert gate mass CSV.

### Splitting (`moe.rs` + `llama-moe-split`)
- `compute_assignments()` implements the overlap strategy: shared core (top N experts by gate mass) replicated to every node, remaining experts distributed uniquely.
- `run_split()` calls `llama-moe-split` to produce per-node GGUFs (trunk + expert subset). Cached at `~/.cache/mesh-llm/splits/<model>/<n>-nodes/node-<i>.gguf`.
- `llama-moe-split` (in `llama.cpp/tools/moe-split/`) slices expert tensors, gathers router gate rows, clamps `expert_used_count`. Supports `--groups`, `--expert-list`, `--ranking-file`.

### Mesh Integration (`election.rs`)
- `moe_election_loop()` handles the full lifecycle: detect MoE → compute assignments → split if needed → start llama-server with shard → rebuild on mesh changes.
- **Solo mode**: model fits locally → load full model, no splitting.
- **Multi-node mode**: model doesn't fit or `--split` forced → each node gets its own shard, runs its own llama-server independently.
- `moe_shard_index()` determines which shard this node gets based on sorted node IDs.
- `build_moe_targets()` publishes the MoE target map so the proxy knows all MoE nodes.

### Session Routing (`election.rs` + `proxy.rs`)
- `get_moe_target()` hashes a session hint (user field, session_id, or conversation_id) to pick a node. Pure hash — deterministic, sticky.
- `extract_session_hint()` parses the hint from HTTP request body.
- `MoeLocal` / `MoeRemote` variants in `InferenceTarget` handle local vs QUIC-tunneled forwarding.

### Tested
- OLMoE-1B-7B: 2 nodes over WAN (225ms RTT Sydney↔Sydney), both shards coherent.
- Qwen3-30B-A3B: local quality validation, 87/128 experts per node = excellent.
- GLM-4.7-Flash-Q4_K_M: MoE auto-detected (64 experts, top-4), fits locally → solo mode, no split. If split, mesh-llm now prefers cached or freshly computed analysis over the sequential fallback.

## Leader-Planned Auto MoE

Current behavior:

- **Leader computes one plan per exact model identity.**
  - participating nodes
  - shard count
  - ranking source
  - overlap and redundancy
  - whether a full-coverage fallback replica is feasible
- **Followers advertise facts, not policies.**
  - model identity present or not
  - available VRAM / RAM
  - health and stability
  - bandwidth / RTT when available
- **No mixed per-node MoE strategies.**
  One deployment gets one plan. Nodes either participate in that plan or sit out.
- **`auto` is the default runtime behavior.**
  The system should pick solo, split, or split-with-redundancy based on current resources without requiring flags such as grouping or overlap mode.

### Failure and Recovery Policy

The deployment objective is to keep serving as much as possible while avoiding topology flapping.

- **Fail down quickly when an active shard is unusable.**
  A shard request failure is stronger evidence than a heartbeat miss and should trigger prompt reconfiguration across survivors.
- **Do not blindly retry on another partial shard.**
  Partial shards do not generally contain interchangeable expert sets.
- **Retry directly only to full-coverage targets.**
  If another node has the full expert set for the same exact model identity, it is a valid failover target.
- **Recover up cautiously.**
  When a lost node reappears, re-admit it to mesh membership first, then keep it out of active MoE placement until it has stayed healthy for a short stability window.
- **Use extra capacity for resilience when available.**
  If the cluster has spare memory, the leader may choose extra overlap, replicated hot experts, or a full-coverage fallback replica instead of maximizing packing efficiency.

### Current Result

This should give mesh-llm the following MoE behavior:

- `A + B -> A` quickly when `B` disappears
- `A + B + C -> A + B` quickly when `C` disappears
- stable serving on the reduced topology instead of waiting for manual restart
- cautious expansion back to larger splits only after the recovered node proves healthy
- optional direct failover to a full-coverage replica when one exists

## What's NOT Implemented

### No probe-based session placement (planned)
The current design uses hash routing — sessions are assigned to nodes deterministically. The original plan proposed fan-out probes where each node scores "how well does my shard match this prompt" and the best node gets the session. This was unnecessary for the 2-node case with sufficient overlap (68%+) — both nodes produce equivalent quality. Probing becomes important with more nodes, less overlap, or sharper expert specialization. With scale testing on larger models coming soon, this is next on the list.

### Remaining limits
- The leader currently prefers a conservative automatic plan:
  - keep the existing active shard set when it is still healthy
  - reserve a dedicated full-coverage fallback when spare nodes exist
  - otherwise use overlap-based redundancy in the active split
- This is intentionally simpler than a full global packing solver. It does not yet optimize across all possible redundancy layouts or cost models.

### No scale testing on large models
Phase 5 in TODO. Mixtral 8×22B (~80GB) and Qwen3-235B-A22B (~130GB) are the real targets where expert sharding provides value (models that don't fit on one machine). Not tested yet.

## Future Evolution: Topology via Model Descriptors

As mesh-llm moves to canonical per-model descriptors in the gossip protocol, MoE should evolve to consume `ModelTopology` from those descriptors instead of treating MoE as a GGUF-only local concern.

Planned direction:

- **Descriptor-first topology**: every served or available model can advertise:
  - canonical identity (`repository`, `revision`, `artifact`)
  - capabilities (`vision`, `reasoning`, `tool_use`, `moe`)
  - optional `topology.moe`
- **Initial MoE topology sources**:
  1. cached `moe-analyze` or `micro-analyze` results
  2. Hugging Face metadata such as `num_experts` and `num_experts_per_tok`
  3. GGUF header fallback when no stronger source exists
- **What goes into `ModelMoeInfo` first**:
  - `expert_count`
  - `used_expert_count`
  - optional `min_experts_per_node`
  - a source label such as `catalog`, `hf_metadata`, or `gguf_header`

This means future MoE coordination can become revision-aware:

- nodes can tell whether they are talking about the same exact model snapshot
- MoE grouping can reject mixed revisions cleanly
- cached analysis can be keyed by `repository + revision + artifact`

### Planned `moe-analyze` integration

`moe-analyze` remains the path to high-quality ranking data.

Expected evolution:

- when a model is detected as MoE but only has fallback topology, mesh-llm can still run conservatively
- if `llama-moe-analyze` is available, mesh-llm can run it in the background for that exact model revision
- the resulting ranking should be cached as descriptor-aligned topology data, not as an ad hoc local guess
- improved rankings should only take effect on the next reload or re-election, never mid-run

This gives a clean progression:

1. **HF / precomputed topology** — immediate compatibility
2. **fallback topology** — safe but conservative operation
3. **`moe-analyze` ranking** — optimized expert placement for later runs

### Possible future live local inference path

There is also room for a lighter-weight live local path later, but it should remain explicitly second-tier to `moe-analyze`.

Possibilities include:

- collecting router statistics from short local warm-up prompts
- estimating expert importance from recent local traffic
- using weight-derived approximations when full analysis is unavailable

This data would be useful for:

- improving placement when no precomputed ranking exists
- prioritizing which unknown MoE models deserve a full `moe-analyze`
- informing probe-based session placement on larger meshes

But it should not replace `moe-analyze` as the canonical high-confidence ranking source without further validation.

## Key Findings

From [MoE_SPLIT_REPORT.md](MoE_SPLIT_REPORT.md):

- **Expert 0 dominance**: In Qwen3-30B-A3B, expert 0 captures 25% of all gate mass. The top 46 experts (36%) form the minimum viable set.
- **Minimum viable threshold**: ~50% of experts needed per node for coherent output (model-dependent). Below that → degenerate loops.
- **Overlap makes probing unnecessary**: With 68%+ overlap, both nodes handle all prompt types equally well. Hash routing is sufficient.
- **Contiguous slicing is naive**: Expert importance isn't correlated with expert ID. Informed grouping (by gate mass ranking) is essential.
- **Split GGUFs are faster**: Smaller expert tensors → less memory pressure → ~8% speed improvement (110 vs 101 t/s on Qwen3-30B-A3B).

## Architecture

### MoE mode (multi-node)
```
Client → Proxy ─→ Node0 (llama-server, shard-0.gguf)
               ├→ Node1 (llama-server, shard-1.gguf)  
               └→ Node2 (llama-server, shard-2.gguf)
```
- N independent llama-servers, zero cross-node inference traffic
- Per-node KV cache → N× total context capacity
- Session-sticky hash routing

### vs Tensor split (dense models)
```
Client → Proxy → Host (llama-server --rpc worker1,worker2)
                    ↕ RPC          ↕ RPC
                Worker1          Worker2
```
- One llama-server, distributed computation
- Cross-node traffic per token (tensor activations)
- Single KV cache on host
