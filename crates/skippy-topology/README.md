# skippy-topology

Topology planning and model-family capability policy.

`skippy-topology` turns model layers, node capacity, requested split points,
and optional family capability records into a validated stage plan. It is a pure
Rust planning crate: no model loading, process launching, sockets, telemetry, or
GGUF writing.

## Architecture Role

Mesh, benchmark, and correctness flows ask this crate whether a split plan is
acceptable before slicing models or starting servers. The returned plan carries
stage ranges, peer/device placement, boundary decisions, wire dtype policy, and
diagnostics.

```mermaid
flowchart LR
    L["LayerSpec list<br/>attention/recurrent flags"] --> P["topology planner"]
    N["NodeSpec list"] --> P
    F["FamilyCapabilityRecord<br/>reviewed or inferred"] --> P
    I["mesh inventory<br/>peers + device capacity"] --> P
    S["requested splits"] --> P
    P --> Plan["TopologyPlan"]
    Plan --> Stages["StagePlan<br/>layer ranges + nodes"]
    Plan --> Bounds["BoundaryPlan<br/>accepted/rejected<br/>wire dtype"]
    Plan --> Diag["diagnostics<br/>reason codes"]
    Stages --> Mesh["mesh coordinator<br/>LoadStage downstream-to-upstream"]
    Bounds --> Mesh
    Diag --> Mesh
```

## Family Policy Flow

```mermaid
sequenceDiagram
    participant C as caller
    participant T as skippy-topology
    participant R as reviewed registry
    participant P as planner

    C->>T: infer_family_capability(model_id, layers, width)
    T->>R: match reviewed identity
    R-->>T: optional capability record
    T-->>C: reviewed or heuristic capability
    C->>P: plan_contiguous_with_splits(request, splits)
    P-->>C: TopologyPlan with diagnostics
```

Reviewed capability records live in
`crates/skippy-topology/capabilities/reviewed-family-capabilities.json`.

## Responsibilities

- validate contiguous layer ranges and split boundaries
- produce even contiguous plans or explicit split plans
- classify stages as stateless, attention-KV, recurrent, or mixed
- reject family-forbidden boundaries such as shared KV producer/consumer cuts
- report activation wire dtype and payload sizing
- infer capabilities for reviewed and known dense/recurrent families

Use this crate before `skippy-model-package`, mesh stage deployment,
`skippy-prompt`, or `skippy-bench` commits to a runnable stage layout. When new
peers or devices make a better split possible, mesh replans by preparing the
replacement topology, waiting for readiness, and only then publishing the new
stage-0 route.
