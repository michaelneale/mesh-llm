# MoE Placement Planning

This document records how deployment-oriented MoE split planning relates to the canonical `moe-analyze` dataset.

## Decision

Placement and split recommendations are **derived outputs**, not part of the first canonical `moe-analyze` artifact.

The canonical Hugging Face dataset stores the measured `moe-analyze` result for an exact:

- `source_repo`
- `source_revision`
- `format`
- `distribution_id`
- `analyzer_id`

That canonical artifact is the stable input for later deployment planning.

## Why Placement Is Separate

Expert rankings are measured artifacts.

Placement plans depend on deployment assumptions that are expected to change:

- VRAM per node
- node count
- KV/cache reserve
- replication policy for hot experts
- placement heuristic
- operating constraints such as network balance or heterogenous nodes

Because those assumptions are not stable, they should not be treated as the same immutable artifact as the ranking itself.

## Current Approach

1. Run and publish canonical `moe-analyze`.
2. Recalculate placement later from the full analysis result.

This means:

- the dataset remains focused on durable measured data
- deployment plans can be recomputed for different hardware classes
- `mesh-llm` or a separate planner can derive splits on demand

## Example Derived Inputs

Once a full `moe-analyze` artifact exists, a placement planner can derive a split using inputs such as:

- `target_vram_per_node`
- `target_node_count`
- `reserved_vram_per_node`
- `estimated_shared_bytes`
- `estimated_expert_bytes`
- `replication_policy`
- `placement_strategy`

Examples:

- best split for `16 GB` nodes
- best split for `24 GB` nodes
- best split for `8` nodes vs `16` nodes

## Optional Future Dataset Projection

If placement outputs become expensive or worth publishing, they can be stored as a second derived artifact family without changing the canonical ranking contract.

For example:

```text
data/<namespace>/<repo>/<revision>/<format>/<distribution_id>/<analyzer_id>/
  metadata.json
  ranking.csv
  run.log
  placement/
    16gb-v1/
      summary.json
      plan.json
```

That projection is optional and should remain clearly separated from the canonical `moe-analyze` artifact.
