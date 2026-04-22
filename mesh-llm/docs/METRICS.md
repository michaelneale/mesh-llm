# Metrics

Metrics in mesh-llm are part of the control plane, not just an observability layer bolted on afterward. Routing decisions, failover logic, and placement heuristics all depend on locally observed outcomes. The metrics system exists to make those observations legible to the runtime, to operators, and eventually to longer-horizon analysis.

The core philosophy: **local first, bounded, routing-adjacent, outcome-aware**. Metrics that stay on one node carry no protocol risk. Metrics that leave the node become compatibility decisions. Latency without outcome context is misleading. Cardinality without bounds is a liability.

## Taxonomy

Every metric in this system belongs to one of three layers and one of three scopes.

### Layers

**Runtime** metrics feed directly into routing, failover, or placement decisions. They need to be fast, cheap, and accurate enough to act on. Examples: inflight request count, per-target success rate, timeout rate.

**Information** metrics are primarily for operator and API visibility. They don't drive runtime decisions but they let operators understand what the node is doing. Examples: request outcome summaries, service mix shares, per-model throughput.

**Strategy** metrics are useful for roadmap and longer-horizon analysis: model popularity trends, demand vs. supply mismatch, failure mode distributions. These are not yet implemented but are tracked as proposed work.

### Scopes

**Local-only** metrics are measured and consumed on one node. They never leave the node and carry no protocol compatibility risk.

**Peer-advertised** metrics are published by a node to its peers via gossip. These are compatibility-sensitive. Once a field is gossiped, older nodes will receive it and must handle it gracefully. Adding, removing, or redefining peer-advertised fields is a protocol change.

**Mesh-derived** metrics are computed locally from existing peer state, API snapshots, or gossip data. They don't require new protocol fields but they do depend on the accuracy and freshness of the underlying data.

### The scope contract

The dangerous mistake is letting a local heuristic accidentally become mesh-wide truth. A metric that looks like a local observation but gets gossiped to peers can corrupt routing on nodes that don't share the same context. Every metric must be tagged with its scope before it's exposed anywhere. The scope tag is the safety rail for mixed-version meshes.

## Design Principles

1. **Consumer-first.** Every metric needs a primary consumer: a runtime control loop, an operator/UI surface, or a strategy/planning use case. A metric with no consumer is not a core metric.

2. **Scope-first.** Every metric is tagged with its scope before it's exposed. This is not optional. Scope determines protocol risk.

3. **Additive-first.** Land on local runtime or API surfaces first. Promote to gossip only when remote peers genuinely need the data for correct behavior, not just for convenience.

4. **Outcome over vanity.** Latency alone is misleading. A fast timeout is still a failure. Every latency or throughput metric should be paired with success/failure/degradation signals.

5. **Bounded cardinality.** Keying by `(model, target)` is fine locally. As a long-lived mesh-wide contract it can explode. Bounds must be explicit and enforced in the implementation.

## Implemented Metrics

The metrics described in this section are the current implementation. The code-level registry in [`src/network/metrics.rs`](../src/network/metrics.rs) is the authoritative classification source: it defines `MetricLayer`, `MetricScope`, `MetricGroupMetadata`, and the `ROUTING_METRIC_GROUPS` array that enumerates all five exported groups.

Storage is bounded: 1-hour TTL, maximum 128 tracked models, maximum 16 targets per model. Stale entries are pruned on snapshot. When bounds are exceeded, the least-recently-updated entries are evicted first.

All currently implemented metrics are `local-only`. None of them are new gossip or protocol fields.

### `/api/status` `routing_metrics`

Layer: **Information** | Scope: **local-only**

Current-node routing outcome summary for operator and API inspection. These counters reflect traffic fronted by this node only. They are not mesh-wide aggregates.

| Field(s) | Meaning |
| --- | --- |
| `request_count`, `successful_requests`, `success_rate` | Request-level routing outcomes for traffic fronted by this node |
| `retry_count`, `failover_count` | Local reroute and failover pressure |
| `attempt_timeout_count`, `attempt_unavailable_count`, `attempt_context_overflow_count`, `attempt_reject_count` | Attempt outcome breakdown observed by this node |
| `avg_queue_wait_ms`, `avg_attempt_ms` | Bounded timing summary from locally observed attempts |
| `avg_tokens_per_second`, `completion_tokens_observed`, `throughput_samples` | Throughput summary from locally observed successful attempts |

### `/api/status` `routing_metrics.local_node`

Layer: **Runtime** | Scope: **local-only**

Current-node routing pressure and lightweight utilization proxies. These are intentionally not a complete node utilization model.

| Field(s) | Meaning |
| --- | --- |
| `current_inflight_requests`, `peak_inflight_requests` | Live and recent peak request pressure on this node |
| `local_attempt_count`, `remote_attempt_count`, `endpoint_attempt_count` | Attempt mix by local, remote, and endpoint targets |
| `avg_queue_wait_ms`, `avg_attempt_ms`, `avg_tokens_per_second`, `completion_tokens_observed`, `throughput_samples` | Current-node latency and throughput proxies for locally observed attempts |

### `/api/status` `routing_metrics.pressure`

Layer: **Information** | Scope: **local-only**

Current-node service mix for requests fronted by this node. These shares are derived from local routing outcomes and are not mesh-wide demand or serving totals.

| Field(s) | Meaning |
| --- | --- |
| `fronted_request_count`, `locally_served_request_count`, `remotely_served_request_count`, `endpoint_request_count` | Service mix for requests fronted by this node |
| `local_service_share`, `remote_service_share`, `endpoint_service_share` | Normalized shares for locally fronted traffic |

### `/api/models[]` `routing_metrics`

Layer: **Information** | Scope: **local-only**

Per-model routing outcome summary observed on the current node. These values describe what this node has seen while routing requests for that model. They are not aggregated across the mesh.

| Field(s) | Meaning |
| --- | --- |
| `request_count`, `successful_requests`, `success_rate` | Per-model request outcomes observed locally |
| `retry_count`, `failover_count` | Per-model instability and recovery pressure observed locally |
| `attempt_timeout_count`, `attempt_unavailable_count`, `attempt_context_overflow_count`, `attempt_reject_count` | Per-model attempt outcome breakdown observed locally |
| `avg_queue_wait_ms`, `avg_attempt_ms`, `avg_tokens_per_second`, `completion_tokens_observed`, `throughput_samples` | Per-model bounded timing and throughput summary observed locally |

### `/api/models[]` `routing_metrics.targets[]`

Layer: **Runtime** | Scope: **local-only**

Per-target routing outcome memory observed on the current node. These entries are the most route-adjacent data in the system: they record what happened the last time this node tried each target for a given model, and they feed directly into routing decisions.

| Field(s) | Meaning |
| --- | --- |
| `target`, `kind`, `last_updated_secs_ago` | Which target was observed, its kind (local/remote/endpoint), and how recent the memory is |
| `attempt_count`, `success_count`, `success_rate` | Per-target success history observed locally |
| `timeout_count`, `timeout_rate`, `unavailable_count`, `context_overflow_count`, `reject_count` | Per-target failure and degradation breakdown observed locally |
| `avg_queue_wait_ms`, `avg_attempt_ms`, `avg_tokens_per_second`, `completion_tokens_observed`, `throughput_samples` | Per-target latency and throughput summary observed locally |

## Pre-existing Metrics

These metrics existed before the formal metrics system but fit the taxonomy. They are listed here for completeness.

### From `/api/status`

| Metric | Layer | Scope |
| --- | --- | --- |
| `inflight_requests` | Runtime | local-only |
| Routing affinity counters | Runtime | local-only |
| Peer `rtt_ms` | Runtime | mesh-derived |
| `inference_perf` / TTFT by target | Runtime | local-only |

### From `/api/models`

| Metric | Layer | Scope |
| --- | --- | --- |
| `request_count`, `last_active_secs_ago` | Information | mesh-derived |
| `active_demand` | Runtime | mesh-derived |
| `node_count`, `active_nodes` | Information | mesh-derived |
| `mesh_vram_gb` | Information | mesh-derived |
| Capability flags: `vision`, `audio`, `reasoning`, `tool_use`, `moe` | Runtime | peer-advertised |

### From gossip

| Metric | Layer | Scope |
| --- | --- | --- |
| `served_model_runtime` presence | Runtime | peer-advertised |
| Model demand signal | Runtime | peer-advertised |

## Proposed Metrics

These metrics are not yet implemented. They are organized by layer. Each entry includes the metric name, why it matters, and its intended scope.

### Runtime (proposed)

These are the highest-priority proposals because they directly affect routing quality.

| Metric | Why it matters | Scope |
| --- | --- | --- |
| Request success rate by model+target | Core routing signal; without it, the router can't distinguish a degraded target from a healthy one | local-only |
| Timeout rate by model+target | Timeouts are the most expensive failure mode; they block the request slot for the full timeout window | local-only |
| Retry/failover count by model | Measures routing instability; high retry counts indicate the primary target selection is wrong | local-only |
| Reject/unavailable/context-overflow rate | Distinguishes capacity exhaustion from network failure from model-level rejection | local-only |
| Request queue/wait time before dispatch | Measures pre-routing delay; high queue wait indicates the node is a bottleneck before any target is tried | local-only |
| Tokens/sec or completion throughput by model+target | Throughput is the primary quality signal for inference; latency alone doesn't capture it | local-only |
| Active server count vs. demanded server count by model | Measures serving deficit; if demand exceeds active servers, the mesh is underserving the model | mesh-derived |
| GPU occupancy / estimated node utilization | Routing needs a utilization signal to avoid overloading nodes that are already near capacity | local-only |
| Cold-start / model-activation time | Affects TTFT for the first request after a model loads; important for demand-aware rebalancing | local-only |
| Split-eligibility / split-failure rate | Measures how often pipeline or expert splits fail to form; split failures fall back to single-node serving | local-only |
| MoE expert pressure / skew indicators | Uneven expert load across nodes degrades MoE throughput; skew detection enables rebalancing | local-only |
| Client-only vs. host-serving pressure | Distinguishes nodes that are only routing from nodes that are also serving; affects placement decisions | local-only |

### Information (proposed)

These improve operator trust and debuggability without affecting runtime behavior.

| Metric | Why it matters | Scope |
| --- | --- | --- |
| Clear node utilization summary | Operators need a single number to understand how loaded a node is | local-only |
| Clear client utilization summary | Client nodes have different load profiles than serving nodes | local-only |
| Route outcome summary | A per-request breakdown of what happened: which target was tried, what the outcome was | local-only |
| "Why this route was chosen" summary | Routing decisions are opaque today; a brief explanation improves debuggability | local-only |
| Serving deficit / unmet demand by model | Shows which models have active demand but no server; useful for capacity planning | mesh-derived |
| Model hotness tier (hot/warm/cold) | A coarse classification of model activity; useful for UI and operator dashboards | mesh-derived |
| Per-model TTFT/throughput summaries | Aggregated quality metrics per model for operator review | local-only |
| Discovery summary quality | How many peers were discovered, how many joined, how many are healthy | mesh-derived |

### Strategy (proposed)

These are useful for roadmap and longer-horizon analysis. They don't need to be fast or cheap.

| Metric | Why it matters | Scope |
| --- | --- | --- |
| Model popularity over time | Identifies which models are growing or shrinking in demand | mesh-derived |
| Demand vs. supply mismatch by model | Persistent mismatch indicates the mesh needs more capacity for specific models | mesh-derived |
| Client utilization vs. node utilization | Measures how much of the mesh's capacity is being used by clients vs. serving nodes | mesh-derived |
| Auto-router traffic share | How much traffic is routed by the auto-router vs. explicit model selection | local-only |
| Direct model-routed vs. auto-routed traffic | Measures how often callers specify a model vs. letting the mesh decide | local-only |
| Hot-path failure modes | Which failure types dominate on the most-used models | local-only |
| Discovery-to-join conversion | How many discovered meshes result in a join; measures discovery quality | local-only |
| Capability demand mix | Which capability types (vision, audio, reasoning, tool_use, moe) are most requested | local-only |
| Latency/throughput pain by topology type | Whether single-node, pipeline-split, or expert-split topologies have different quality profiles | local-only |

## Scope Safety

Four risks to keep in mind when adding or promoting metrics.

**Protocol risk.** Any metric that leaves the node becomes a compatibility decision. Gossip fields, API fields consumed by peers, and any data that flows across the QUIC connection must be additive and backward-compatible. Older nodes will receive new fields and must handle them gracefully. Newer nodes must handle missing fields from older peers.

**Cardinality risk.** Per-model, per-target keying is fine locally with explicit bounds. As a long-lived mesh-wide contract, the same keying can produce unbounded state across the mesh. The current implementation enforces 128 models and 16 targets per model. Any proposal to gossip per-model or per-target data must address cardinality explicitly.

**Semantic risk.** Ambiguous metrics destabilize routing. A metric named "utilization" that means different things on different nodes will produce inconsistent routing decisions. Every metric needs a precise definition of what it measures and what it doesn't.

**Latency-only risk.** TTFT without outcome context is misleading. A node that times out quickly looks fast. A node that rejects requests looks available. Latency metrics must always be paired with outcome breakdowns.

## Direction

New metrics should expand outward from what already exists: local runtime state → local API surfaces → operator/UI summaries → mesh-derived aggregates → compatibility-sensitive protocol fields.

**Strengthen operator visibility first.** The most valuable near-term work is route outcome summaries, per-request routing explanations, serving deficit signals, and model hotness tiers. These are all local-only or mesh-derived and carry no protocol risk.

**Derive strategy metrics from existing state.** Model popularity, demand vs. supply mismatch, and failure mode distributions can be computed from current gossip and API state without new protocol fields. Time-windowed aggregation may require persistent or ring-buffer storage; evaluate per metric whether the accuracy tradeoff is worth it.

**Promote to protocol only when runtime correctness requires it.** The bar for adding a gossip field is that remote peers need the data to make correct routing decisions, not just better ones. Any metric that enters gossip or is consumed by peers must be additive, optional, and backward-compatible with older nodes.

## Open Questions

**Which utilization estimate should drive runtime decisions?** GPU occupancy, inflight request count, and queue wait time all proxy utilization differently. They disagree under different load patterns. The right signal may be a composite, but composites are harder to reason about and debug.

**Auto-router tracking: request-level, token-level, or both?** Request-level tracking is cheap and easy. Token-level tracking is more accurate for capacity planning but requires streaming token counts through the routing path. The right granularity depends on what decisions the auto-router needs to make.

**Route explanation: `/api/status` or a dedicated diagnostics endpoint?** Embedding route explanations in `/api/status` keeps the surface simple but adds noise for operators who don't need it. A dedicated `/api/diagnostics` or per-request trace endpoint is cleaner but adds API surface.

**Which strategy metrics are derivable cheaply without persistent storage?** Some strategy metrics (model popularity over time, demand trends) require time-windowed aggregation. Without persistent storage, they can only be approximated from current state. The tradeoff between accuracy and implementation complexity needs to be evaluated per metric.

**MoE expert-pressure signal stability?** Expert load is highly request-dependent and can shift rapidly. A pressure signal that's too noisy will cause unnecessary rebalancing. A signal that's too smooth will miss real skew. The right smoothing window and threshold are unknown without production data.
