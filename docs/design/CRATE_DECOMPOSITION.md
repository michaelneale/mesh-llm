# mesh-llm crate decomposition plan

This plan decomposes the current `mesh-llm` crate in two layers:

1. Move shared or pure code into existing crates where ownership is already clear.
2. Extract app-specific host subsystems into focused `mesh-llm-*` crates.

The root `mesh-llm` crate should become a thin binary and app assembly layer:
parse CLI, configure logging, assemble runtime dependencies, and call `run()`.

## Target dependency graph

```mermaid
flowchart TD
    model_ref["model-ref"]
    model_artifact["model-artifact"]
    model_resolver["model-resolver"]
    model_hf["model-hf"]

    types["mesh-llm-types"]
    identity["mesh-llm-identity"]
    protocol["mesh-llm-protocol"]
    client["mesh-client / mesh-api"]
    control_plane["mesh-llm-control-plane"]
    routing["mesh-llm-routing"]
    system["mesh-llm-system"]
    plugins["mesh-llm-plugin-host"]
    runtime_data["mesh-llm-runtime-data"]
    host_runtime["mesh-llm-host-runtime"]
    gateway["mesh-llm-gateway"]
    api_server["mesh-llm-api-server"]
    ui["mesh-llm-ui"]
    cli["mesh-llm-cli"]
    app["mesh-llm"]

    model_ref --> types
    model_artifact --> types
    model_resolver --> types
    model_hf --> types

    types --> identity
    types --> protocol
    identity --> protocol
    protocol --> client
    protocol --> control_plane

    control_plane --> routing
    routing --> host_runtime
    system --> host_runtime
    plugins --> host_runtime
    runtime_data --> host_runtime
    control_plane --> host_runtime

    host_runtime --> gateway
    host_runtime --> api_server
    ui --> api_server
    host_runtime --> cli
    gateway --> app
    api_server --> app
    cli --> app
```

## Proposed destinations

| Current area | Destination | Reason |
| --- | --- | --- |
| `protocol/`, generated proto, ALPN and stream IDs, frame validation | New `mesh-llm-protocol` | Shared by host and client; protocol compatibility deserves a small, explicit crate. |
| Shared mesh/client types such as `PeerAnnouncement`, `PeerInfo`, `NodeRole`, and model descriptors | New `mesh-llm-types`, or fold into `mesh-llm-protocol` if it remains small | Breaks the current `protocol -> mesh -> models/crypto` coupling. |
| Shared signing, ownership, and envelope crypto | New `mesh-llm-identity` | Both `mesh-client` and the host need this; keychain and storage can stay behind host-only features. |
| `mesh/` gossip, heartbeat, membership, peer state, config sync | New `mesh-llm-control-plane` | This avoids the self-referential `mesh-llm-mesh` name and describes the subsystem's actual role: control-plane membership and coordination, not model execution. |
| `network/router.rs`, `network/affinity.rs`, route scoring, request placement, election-adjacent logic | New `mesh-llm-routing` | Routing and placement should be reusable without pulling in process runtime or CLI UI. |
| `network/proxy.rs`, `network/tunnel.rs`, HTTP ingress glue | New `mesh-llm-gateway` | This is the network edge around OpenAI/API traffic. |
| `network/openai/*` | Existing `openai-frontend` | Adapters, schemas, and response translation fit the existing OpenAI frontend crate better than a new crate. |
| `plugin/` host runtime, MCP bridge, transport, config support | New `mesh-llm-plugin-host` | Keep host-side plugin orchestration separate from `mesh-llm-plugin`, which should remain the plugin author API. |
| `plugins/blobstore`, `plugins/blackboard`, telemetry, OpenAI endpoint plugin | Initially submodules of `mesh-llm-plugin-host`; later first-party plugin crates if needed | These do not all need crates yet. Extract only when boundaries harden. |
| `runtime_data/` | New `mesh-llm-runtime-data` | Shared by runtime, API, plugins, and CLI dashboard. |
| `system/hardware.rs`, backend detection, benchmark primitives | New `mesh-llm-system` | Local-machine concerns should stay out of mesh/protocol crates. |
| `system/autoupdate.rs`, `release_target.rs` | Keep in root app or move to `mesh-llm-cli` | App distribution behavior, not core system modeling. |
| `models/resolve`, `catalog`, `remote_catalog`, `search`, download code | Existing `model-resolver` and `model-hf` | Avoid creating `mesh-llm-models` for code that already belongs to model infrastructure. |
| `models/capabilities.rs`, `models/topology.rs`, `models/gguf.rs` | Existing `model-artifact`, `model-ref`, or `model-resolver` depending on final ownership | These are model metadata concerns, not host runtime concerns. |
| `inference/skippy/*` | Existing `skippy-*` crates where possible | The host should orchestrate Skippy rather than own Skippy package, topology, and runtime internals. |
| `inference/election.rs`, split planning | `mesh-llm-routing`, or existing shared client inference modules if needed by embedded clients | Election is placement logic rather than process runtime. |
| React console | `mesh-llm-ui` | The console owns its package metadata, build, generated assets, Rust embedding surface, and UI conventions instead of living under the host binary crate. |
| `api/` | New `mesh-llm-api-server` | Management API routes can depend on `mesh-llm-runtime-data` and serve assets produced by `mesh-llm-ui`, but should not own the console source. |
| `cli/` | New `mesh-llm-cli` | Large dependency surface: Clap, Ratatui, terminal progress, dashboard output. Extract late. |
| `runtime/` | New `mesh-llm-host-runtime` | Extract late because it coordinates almost every subsystem today. |

## Expected benefits

The split should pay off in both engineering velocity and operational safety:

- Faster targeted builds and checks: pure/shared crates such as protocol, types, identity, routing, and runtime data can be checked without rebuilding the full host binary, UI, Skippy runtime wiring, or terminal dashboard stack.
- Smaller CI blast radius: changes can run crate-specific validation first, reserving `just build` and distributed/runtime tests for changes that actually touch those layers.
- Better mergeability: smaller ownership boundaries reduce conflicts in large files such as `runtime/mod.rs`, `mesh/mod.rs`, and broad crate-level module glue.
- Clearer reviews: PRs can be reviewed by subsystem ownership, such as protocol compatibility, routing behavior, plugin host behavior, UI packaging, or model resolution, instead of asking reviewers to reason about the entire binary crate.
- Cleaner dependency hygiene: CLI, TUI, keychain, Hugging Face, OpenAI frontend, Skippy, and platform hardware dependencies can stay attached to the crates that need them instead of leaking through `mesh-llm`.
- Safer protocol evolution: protobuf generation, frame validation, ALPN/stream IDs, and compatibility tests can live in `mesh-llm-protocol`, making mixed-version compatibility easier to audit.
- Better embedded-client reuse: shared protocol, identity, routing, and type crates give `mesh-client` and `mesh-api` stable dependencies without depending on host-only runtime code.
- More predictable documentation ownership: each crate README can explain the subsystem boundary, while top-level docs and Mermaid diagrams describe composition instead of implementation detail.
- Easier incremental extraction: once the dependency graph points inward toward shared crates and outward toward app assembly, later moves become mechanical instead of architectural surgery.

## Refactors before extraction

The main blocker is dependency direction. Before extracting crates, reduce direct calls from domain modules into app-level modules:

- Add a `MeshEventSink` or equivalent callback instead of calling `crate::cli::output` from `mesh/`.
- Move pinned-GPU preflight out of `mesh/` and into runtime or plugin config handling.
- Put protocol-facing shared types in `mesh-llm-types` so protocol conversion does not depend on `crate::mesh`.
- Keep generated protobuf and frame validation in `mesh-llm-protocol`.
- Make host-only integrations explicit through traits or feature-gated adapters.

## Documentation scope

The crate split should include the documentation migration as part of the same scope, not as follow-up cleanup:

- Add a `README.md` for every new crate that explains ownership, public API boundaries, dependency expectations, and how the crate fits into the host runtime.
- Update existing crate READMEs when code moves into or out of those crates, especially `mesh-client`, `mesh-api`, `mesh-llm-plugin`, `openai-frontend`, `model-*`, and `skippy-*`.
- Update top-level docs that describe repository structure, architecture, runtime composition, plugin ownership, model resolution, OpenAI routing, and console/API packaging.
- Update Mermaid diagrams anywhere they describe the crate graph, runtime architecture, protocol/control-plane flow, API/UI packaging, or plugin/data ownership.
- Keep the root `README.md`, `docs/README.md`, and `crates/mesh-llm/README.md` aligned so new contributors can find the owning crate for a subsystem without reading implementation details first.

## Extraction order

1. Extract `mesh-llm-types`, `mesh-llm-identity`, and `mesh-llm-protocol`.
2. Move model metadata, resolve, search, and download code into existing `model-*` crates.
3. Move OpenAI adapter and schema code into `openai-frontend`.
4. Push Skippy-owned logic into existing `skippy-*` crates.
5. Extract `mesh-llm-runtime-data`.
6. Decouple `mesh/` from CLI, system, and runtime, then extract `mesh-llm-control-plane`.
7. Extract routing and election logic into `mesh-llm-routing`.
8. Keep the React console and asset embedding surface in `mesh-llm-ui`.
9. Extract plugin host, gateway, and API server crates.
10. Extract `mesh-llm-cli`.
11. Add new crate READMEs and update affected existing READMEs, docs, and Mermaid diagrams as each ownership boundary moves.
12. Leave root `mesh-llm` as the thin binary and app assembly crate.

The guiding rule is to split by dependency direction and ownership, not by folder size. The first win is making protocol, types, and identity shared and boring; after that, the remaining crate boundaries should become much easier to see.
