# MoE Islands — Expert Co-Activation Clustering for Distributed Inference

Improve distributed MoE inference by clustering experts that fire together
into islands and routing sessions to the island that best matches the prompt's
expert activation pattern. Each island holds the full trunk plus its expert
cluster. Sessions are pinned to an island after classification.

This builds on the existing MoE sharding (full trunk + expert subset per node)
with two improvements: smarter expert grouping and smarter session routing.

## What Changes vs Current Sharding

| | Current | Islands |
|---|---------|---------|
| **Expert grouping** | Gate mass ranking — hottest experts shared, rest distributed sequentially | Co-activation clustering — experts that fire together go to the same island |
| **Session routing** | Hash of session hint — deterministic, prompt-unaware | Offline classifier — routes prompt to the island whose expert cluster best matches |
| **Trunk** | Full trunk per node | Full trunk per node (same) |
| **Cross-node traffic** | Zero | Zero (same) |
| **When it matters** | Works well with 2 nodes, 68%+ overlap | Matters at 4+ nodes where overlap decreases and expert specialisation becomes important |

## Why Clustering Matters

Current sharding distributes experts by individual gate mass ranking: the
hottest experts (by aggregate probability) are shared, the rest are dealt out
round-robin. This ignores that experts fire **in groups** — a code prompt
activates a cluster of code-related experts together, not randomly.

With 2 nodes and high overlap (68%+), this doesn't matter — both nodes
cover most expert combinations. With more nodes and less overlap per node,
a prompt may need experts scattered across multiple islands. Clustering
experts by co-activation ensures each island is self-contained for a class
of prompts.

## Design

### Offline: Build Islands

Run once per model (like `moe-analyze` today).

1. **Collect co-activation data** — extend `moe-analyze` to record, for each
   token, which experts were in top-K. Build a `[n_expert × n_expert]`
   co-firing count matrix.

2. **Cluster experts into K islands** — group experts that frequently fire
   together. Standard clustering (spectral, k-means on co-fire vectors) on
   the co-firing matrix. Each expert belongs to exactly one island.
   K = number of nodes.

3. **Build a prompt classifier** — from the co-activation data, learn which
   prompt characteristics (topic, language, structure) map to which island.
   This can be simple (keyword/embedding based) or use the router's own
   patterns from the analysis data.

4. **Split the model** — use existing `moe-split` with the island-based expert
   assignments instead of gate-mass-ranked assignments. Each island GGUF
   contains the full trunk + its expert cluster. Same split format as today.

### Runtime: Route and Serve

1. **Prompt arrives at proxy**
2. **Proxy classifies prompt** → picks island using offline classifier
3. **Route to island node** — same QUIC tunnel routing as today
4. **Island runs full inference** — trunk + its experts, start to finish
5. **Session pinned to island** for follow-up turns

If the classifier is wrong, the island still works — the router picks the
best available experts from the island's subset. Same graceful degradation
as current sharding. Quality is slightly lower than if routed to the
optimal island, but not broken.

## What This Doesn't Solve

**Models where the trunk doesn't fit on one node.** The trunk (attention
layers, embeddings, output head) is dense — every token passes through
every trunk layer. You can't split it across nodes without pipeline
parallelism (token hops per layer boundary).

| Model | Trunk est. | Fits on 24GB? | Fits on 48GB? |
|-------|-----------|---------------|---------------|
| Qwen3-30B-A3B | ~1.7 GB | ✅ | ✅ |
| Qwen3-235B-A22B | ~15 GB | ✅ | ✅ |
| DeepSeek-V3 (671B) | ~40 GB | ❌ | ✅ |

For DeepSeek-V3 on 24GB nodes, islands alone aren't enough — you'd also need
tensor splitting (RPC) for the trunk within each island, or pipeline
parallelism. That's a hybrid approach (island for experts + pipeline/tensor
split for trunk) which is possible but significantly more complex.

For Qwen3-235B on 24GB+ nodes, islands work as described.

## VRAM Budget Example

**Qwen3-235B-A22B (Q4_K_M ≈ 130GB) across 4 × 48GB nodes:**

| Component | Per-island | Notes |
|-----------|-----------|-------|
| Full trunk | ~15 GB | Replicated |
| Expert cluster (32 of 128) | ~29 GB | 1/4 of expert params |
| KV cache (Q8_0) | ~3 GB | Per-island, independent |
| **Total** | **~47 GB** | Fits in 48GB |

**DeepSeek-V3 (Q4_K_M ≈ 370GB) across 8 × 64GB nodes:**

| Component | Per-island | Notes |
|-----------|-----------|-------|
| Full trunk | ~40 GB | Replicated |
| Expert cluster (32 of 256) | ~16 GB | 1/8 of expert params (some shared) |
| KV cache (Q4_0) | ~4 GB | Per-island, independent |
| **Total** | **~60 GB** | Fits in 64GB |

## Implementation

### Phase 1: Co-activation analysis
Extend `moe-analyze` to output expert co-firing matrix. Add clustering to
produce island assignments for K nodes. Output format: same as existing
ranking CSV but grouped by island.

### Phase 2: Classifier
Build offline prompt → island classifier from the co-activation data.
Simplest version: embed the prompt, compare to island centroids from
the analysis data.

### Phase 3: Wire into mesh-llm
- `compute_assignments()` accepts island-based groupings instead of
  (or in addition to) gate-mass rankings
- Proxy uses classifier for routing instead of session hash
- Everything else (election, gossip, split, launch) stays the same

### What Needs Building

| Feature | Where | Complexity |
|---------|-------|------------|
| Expert co-activation matrix | `moe-analyze` | Medium |
| Expert clustering (K islands) | `moe-analyze` or offline script | Medium |
| Prompt → island classifier | mesh-llm `proxy.rs` | Medium |
| Island-aware `compute_assignments()` | `moe.rs` | Low — different input, same split logic |

No changes to llama-server, llama.cpp, or the GGUF format.

## References

- Current MoE implementation: [MoE_PLAN.md](MoE_PLAN.md)
- Expert ranking data: [MoE_SPLIT_REPORT.md](MoE_SPLIT_REPORT.md)
- `moe-analyze` source: `llama.cpp/tools/moe-analyze/moe-analyze.cpp`
- `moe-split` source: `llama.cpp/tools/moe-split/`
- Expert mask API: `llama_model_set_expert_mask()` in `llama.h`
