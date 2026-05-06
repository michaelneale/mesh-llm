# PR: Make Skippy split topology layer-aware and preparation-driven

## Summary

This PR changes Skippy split serving so model downloads and layer preparation no
longer count against inference or stage-control timeouts.

Nodes now report the layer ranges they can serve or prepare, the coordinator
plans from currently usable capacity, and slower nodes can join later through a
candidate topology once their assigned layers are downloaded, materialized, and
loaded.

The protocol change is intentionally breaking and is scoped to the Skippy stage
control protocol. `mesh.proto` gossip is not changed.

## User Impact

- Split serving can start with the best topology available now instead of
  waiting for every selected node to download layers.
- Slow or cold nodes can join a split after preparation completes.
- Model download time is treated as topology preparation, not request latency.
- Status output can explain what each node is doing, including layer range,
  phase, progress when known, and failure reason.
- Existing ready topologies keep serving while a better candidate topology is
  prepared in the background.

## Architecture

The coordinator uses the existing mesh QUIC connection as transport, but all new
messages are Skippy-owned stage-control protobufs in
`crates/skippy-protocol/proto/stage.proto`.

No full layer inventory, progress, or preparation state is added to `mesh.proto`.
The coordinator asks peers directly over the Skippy stage control stream.

```mermaid
flowchart LR
    C["Coordinator"]
    W1["Worker A"]
    W2["Worker B"]
    P["skippy-protocol stage control"]
    M["mesh QUIC transport"]

    C --> M
    M --> P
    P --> W1
    P --> W2

    C -- "GetLayerInventory" --> W1
    W1 -- "LayerInventory" --> C
    C -- "PrepareStage" --> W2
    W2 -- "PrepareAccepted" --> C
    W2 -- "StageStatusUpdate" --> C
```

## Protocol

The Skippy stage protocol is now the source of truth for layer inventory and
stage preparation.

Today, `LoadStage` is synchronous from the coordinator's point of view: the
coordinator sends one request and waits for the remote node to download or
resolve model artifacts, materialize the assigned stage, start the binary stage,
and return `StageReady`. That makes model download time part of the control RPC
timeout.

This PR replaces that meaning with an explicit asynchronous preparation
protocol. The coordinator first asks what a node has, then assigns preparation
work, and then observes status until the stage is ready.

```mermaid
sequenceDiagram
    participant C as Coordinator
    participant W as Worker

    rect rgb(255, 240, 240)
    Note over C,W: Old synchronous load
    C->>W: LoadStage(layers 12..24)
    W->>W: download package parts
    W->>W: materialize stage GGUF
    W->>W: start stage runtime
    W-->>C: StageReady
    Note over C,W: One RPC timeout covers all work
    end

    rect rgb(240, 248, 255)
    Note over C,W: New async preparation
    C->>W: GetLayerInventory(package, manifest)
    W-->>C: LayerInventory(ready/materialized/available/missing)
    C->>W: PrepareStage(layers 12..24)
    W-->>C: PrepareAccepted(state=ASSIGNED)
    W-->>C: StageStatusUpdate(DOWNLOADING)
    W-->>C: StageStatusUpdate(MATERIALIZING)
    W-->>C: StageStatusUpdate(READY)
    Note over C,W: Download is preparation, not control timeout
    end
```

New protocol concepts:

- `LayerRange`
- `LayerInventory`
- `LayerRangeAvailability`
- `GetLayerInventory`
- `PrepareStage`
- `CancelPrepareStage`
- `StagePreparationStatus`
- `StageStatusUpdate`

Preparation states:

- `ASSIGNED`
- `DOWNLOADING`
- `AVAILABLE`
- `MATERIALIZING`
- `MATERIALIZED`
- `LOADING`
- `READY`
- `FAILED`
- `CANCELLED`

### Wire-Level Changes

`StageControlRequest` adds new command variants:

- `get_layer_inventory`
- `prepare_stage`
- `cancel_prepare_stage`
- `stage_status_update`

`StageControlResponse` adds new response variants:

- `layer_inventory`
- `prepare_stage_accepted`
- `stage_preparation_status`
- `stage_status_ack`

`StageRuntimeState` is replaced or expanded with preparation-specific states so
callers do not need to overload `STARTING` for download, materialization, and
runtime startup.

The existing `StageStatus` payload is extended into a preparation-aware status:

- package identity: `package_ref`, `manifest_sha256`
- assigned layer range: `layer_start`, `layer_end`
- state: assignment/download/materialization/loading/ready/failure
- optional progress: `bytes_done`, `bytes_total`
- runtime endpoint: `bind_addr`, present only when the stage is ready
- error string for terminal failures

### Semantic Changes

`LoadStage` is no longer used by split serving as a blocking
download-and-start operation. For split serving, the new flow is:

1. `GetLayerInventory`
2. `PrepareStage`
3. `StageStatusUpdate` push messages, with `GetStageStatus` polling as fallback
4. coordinator activates a topology only after every required stage is `READY`

`LoadStage` can either be removed in the protocol break or kept only as a
compatibility shim for local/single-stage paths that already have local model
artifacts. It must not be used for remote split layer downloads.

### Why This Is Breaking

This intentionally changes the meaning of stage readiness. In the old protocol,
a successful `LoadStage -> StageReady` meant the remote runtime was loaded and
ready to receive activations. In the new protocol, accepting work and becoming
ready are separate events.

Old nodes do not understand inventory, prepare, or status-update commands. New
coordinators should require the new Skippy stage protocol generation for the
layer-inventory split path.

Base mesh gossip remains unchanged; the break is isolated to Skippy stage
control messages.

```mermaid
stateDiagram-v2
    [*] --> ASSIGNED
    ASSIGNED --> DOWNLOADING: package parts missing
    ASSIGNED --> AVAILABLE: layer files local
    DOWNLOADING --> AVAILABLE: files local
    AVAILABLE --> MATERIALIZING
    MATERIALIZING --> MATERIALIZED
    MATERIALIZED --> LOADING
    LOADING --> READY

    ASSIGNED --> FAILED
    DOWNLOADING --> FAILED
    MATERIALIZING --> FAILED
    LOADING --> FAILED

    ASSIGNED --> CANCELLED
    DOWNLOADING --> CANCELLED
    MATERIALIZING --> CANCELLED
    MATERIALIZED --> CANCELLED
```

## Coordinator Flow

The coordinator creates an active topology only from capacity that can serve
now. Nodes that still need downloads are prepared as background candidates and
can join after they reach `READY`.

```mermaid
sequenceDiagram
    participant C as Coordinator
    participant A as Ready Node
    participant B as Cold Node
    participant R as Runtime

    C->>A: GetLayerInventory(package_ref, manifest_sha256)
    A-->>C: ready/materialized/available ranges
    C->>B: GetLayerInventory(package_ref, manifest_sha256)
    B-->>C: missing or preparing ranges

    C->>C: Plan active topology from ready capacity
    C->>A: PrepareStage(active generation)
    A-->>C: PrepareAccepted
    A-->>C: StageStatusUpdate(READY)
    C->>R: Activate active topology

    C->>C: Score better candidate topology
    C->>B: PrepareStage(candidate generation)
    B-->>C: PrepareAccepted
    B-->>C: StageStatusUpdate(DOWNLOADING)
    B-->>C: StageStatusUpdate(MATERIALIZING)
    B-->>C: StageStatusUpdate(READY)
    C->>R: Cut over to candidate topology
```

## Planning Policy

The coordinator ranks topology inputs by readiness and cost:

1. `READY`
2. `MATERIALIZED`
3. `AVAILABLE`
4. `MISSING`, only for background candidates

The initial active topology avoids required downloads. Background preparation is
started only when a candidate is materially better.

Material improvement includes:

- Adding enough capacity to serve a model that otherwise cannot run.
- Reducing the worst-stage layer count by a meaningful threshold.
- Improving VRAM headroom or stage balance.
- Replacing a failed or unhealthy stage.
- Bringing more eligible nodes into the topology when balance improves.

The coordinator should penalize:

- Required download bytes.
- Topology churn.
- Recent failures for the same node/package/range.
- Preparing too many candidates at once.

```mermaid
flowchart TD
    I["Collect inventory"]
    A["Plan active topology from ready capacity"]
    S["Serve requests"]
    C["Score candidate using all eligible nodes"]
    B{"Candidate materially better?"}
    P["Prepare missing ranges in background"]
    R{"All candidate stages READY?"}
    X["Cut over to candidate generation"]
    F["Keep active topology"]

    I --> A --> S
    I --> C --> B
    B -- "no" --> F
    B -- "yes" --> P --> R
    R -- "no" --> F
    R -- "yes" --> X --> S
```

## Timeout Model

Timeouts are split by responsibility:

- Assignment/control timeout: short, covers accepting a control message.
- Preparation timeout: long or disabled by default, covers download and
  materialization.
- Stage start timeout: bounded, starts after local artifacts are ready.
- Inference first-byte timeout: starts only after a ready topology is active.

Downloads do not count against inference or stage-control response time.

## Failure and Cancellation

Failures are recorded per node and `{package_ref, manifest_sha256, layer_range}`
with backoff. The coordinator keeps serving the active topology if candidate
preparation fails.

`CancelPrepareStage` cancels stale candidate work. Workers may keep already
downloaded package files because they are useful cache, but should stop
materialization or runtime loading when practical.

## API and UI

`/api/status` should expose enough state to explain split preparation:

- node id
- package ref and manifest
- layer range
- state and phase
- bytes done and total when known
- current active or candidate generation
- failure reason

Example user-visible states:

- `stage-1: ready, layers 0..12`
- `stage-2: downloading, layers 12..24`
- `stage-3: materializing, layers 24..32`
- `stage-4: failed, checksum mismatch`

## Validation

Recommended validation:

- `cargo test -p skippy-protocol --lib`
- `cargo test -p mesh-llm inference::skippy --lib`
- `cargo test -p mesh-llm --lib` for stage-control conversion and runtime status
- Two-node forced split with one warm node and one cold node.
- Slow or throttled model download on a worker; verify active topology serves
  before the cold node joins.
- Candidate replan and cutover after the cold node reaches `READY`.

## Compatibility

This is a breaking Skippy stage protocol change. Mixed old/new split-stage
nodes are not supported for the new layer-inventory preparation path.

Base mesh gossip remains unchanged. Non-skippy mesh behavior is not affected.
