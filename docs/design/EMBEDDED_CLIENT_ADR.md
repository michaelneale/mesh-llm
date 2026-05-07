# ADR: Native Embedded SDK for mesh-llm

## Status

Accepted — supersedes any informal discussion in issue #188

---

## Context

Native iOS, macOS, and Android apps need to join mesh-llm meshes without running a localhost HTTP server, spawning a sidecar process, or invoking the CLI. The existing mesh-llm binary is a desktop-first process that binds ports, spawns child processes, and reads credentials from the filesystem. None of that is acceptable inside a sandboxed mobile app.

Issue #188 identified the core need: extract the passive Rust client from mesh-llm, expose it via uniffi-rs, and ship Swift and Kotlin bindings that app developers can drop into their projects. The result must be App Store-safe: no subprocess spawning, no ambient port binding, no filesystem credential loading.

The passive client role already exists in the mesh-llm codebase as `NodeRole::Client`. It joins a mesh via an invite token, discovers available models, routes inference requests through QUIC tunnels, and receives streamed responses. It does not run llama-server, rpc-server, or any inference process. Extracting that role into a standalone crate is the foundation of this work.

---

## Decision

### FFI Toolchain

**uniffi-rs v0.31+** is the chosen FFI layer. It generates Swift and Kotlin bindings from a single Rust interface definition, supports async functions natively, and has an active maintenance track. The alternatives were evaluated and rejected:

- **swift-bridge**: Rust-only. No Kotlin support. Rejected.
- **diplomat**: Less mature ecosystem, smaller community, fewer production deployments. Rejected.
- **cbindgen**: C ABI only. No async support. Rejected.

### Crate Structure

Wave 1 splits the workspace into four new crates immediately:

- `mesh-client` — pure Rust client logic, no FFI, no host code
- `mesh-host-core` — host-side orchestration, macOS-only, behind a feature flag
- `mesh-api-ffi` — uniffi-rs bridge, depends on `mesh-client` and optionally `mesh-host-core`
- `mesh-llm-test-harness` — shared test infrastructure, `FixtureMesh` lives here

### Phases

The work is organized into seven phases:

- **Phase 1**: Workspace skeleton, ADR, and toolchain POC
- **Phase 2**: Extract protocol types and wire format from `mesh-llm`
- **Phase 3**: Extract client networking (QUIC, gossip, routing) into `mesh-client`
- **Phase 4**: Implement `MeshClient` and `MeshHost` against the extracted core
- **Phase 5**: uniffi-rs bindings, Swift package, Kotlin artifact
- **Phase 6**: Integration tests, platform CI, and wave-end gates
- **Phase 7**: Document existing `mesh-llm/0` and `mesh-llm/1` dual-support as a compatibility guarantee for embedded clients. This phase records the existing behavior; it does not design a new protocol.

### Development Discipline

TDD across all surfaces. Every extraction follows the recipe: write a failing test in the destination crate, copy the implementation byte-for-byte, add a transitional re-export shim in the source crate, verify `just build && cargo test --workspace` passes, then commit atomically.

---

## Cargo Feature Topology

`mesh-api-ffi` has a `host` feature that is **off by default**.

| Build target | Cargo flags |
|---|---|
| iOS / Android | `--no-default-features` |
| macOS host | `--features host` |

`mesh-client` has a strict dependency policy. Its dependency tree is allowed to include the transport, protocol, crypto, parsing, and async crates needed for an embedded QUIC client, including the current set used in this PR such as `iroh`, `tokio`, `prost`, `bytes`, `rustls`, `quinn`, `serde`, `serde_json`, `thiserror`, `anyhow`, `tracing`, `sha2`, `ed25519-dalek`, `hex`, `uuid`, `url`, `http`, `base64`, `nostr-sdk`, `crypto_box`, `rand`, `async-trait`, and `httparse`.

The actual CI-enforced constraint is that `mesh-client` must not pull in desktop/host-oriented crates such as credential stores, CLI frameworks, or plugin/runtime host dependencies. This keeps the mobile binary size predictable and prevents accidental inclusion of desktop-only code.

`mesh-api-ffi` must not directly depend on `iroh`, `tokio`, or `prost`. It reaches those through `mesh-client` only.

---

## MVP API (Closed List — Authoritative)

This is the complete, closed API surface for the embedded SDK. No methods may be added without a new ADR revision.

### MeshClient

`MeshClient` is the primary type for mobile and embedded use. It is available on all platforms.

```rust
impl MeshClient {
    pub fn new(keypair: OwnerKeypair, token: InviteToken) -> Result<Self, MeshError>;
    pub async fn join(&mut self) -> Result<(), MeshError>;
    pub async fn list_models(&self) -> Result<Vec<Model>, MeshError>;
    pub async fn chat(&self, request: ChatRequest, listener: Box<dyn EventListener>) -> Result<RequestId, MeshError>;
    pub async fn responses(&self, request: ResponsesRequest, listener: Box<dyn EventListener>) -> Result<RequestId, MeshError>;
    pub async fn cancel(&self, request_id: RequestId) -> Result<(), MeshError>;
    pub async fn status(&self) -> Result<MeshStatus, MeshError>;
    pub async fn disconnect(&mut self) -> Result<(), MeshError>;
    pub async fn reconnect(&mut self) -> Result<(), MeshError>;
}
```

Nine methods. No more, no fewer.

`chat` maps to the `/v1/chat/completions` endpoint. `responses` maps to `/v1/responses`. Both deliver tokens incrementally via `EventListener`. `cancel` terminates an in-flight request by `RequestId`. `reconnect` is async and is intended to be awaited from foreground-resume handling after the app returns from background.

### MeshHost

`MeshHost` is macOS-only and lives behind `#[cfg(feature = "host")]`. It is not available on iOS or Android.

```rust
impl MeshHost {
    pub fn new(keypair: OwnerKeypair, config: HostConfig) -> Result<Self, MeshError>;
    pub async fn start(&mut self) -> Result<InviteToken, MeshError>;
    pub async fn stop(&mut self) -> Result<(), MeshError>;
    pub async fn load_model(&mut self, model: &str) -> Result<(), MeshError>;
    pub async fn unload_model(&mut self, model: &str) -> Result<(), MeshError>;
    pub async fn host_status(&self) -> Result<HostStatus, MeshError>;
}
```

Six methods. `start` returns an `InviteToken` that the host app can share with clients. `load_model` and `unload_model` manage the active model set at runtime.

---

## Extraction Map

Code from `crates/mesh-llm-host-runtime/src/` falls into three categories.

| Category | Modules | Notes |
|---|---|---|
| A: Extract verbatim | `src/protocol/*`, `src/network/{router,affinity,rewrite}`, `src/models/{catalog,capabilities,gguf}`, `src/crypto/{keys,mod}` pure parts | No semantic changes. Copy byte-for-byte, add re-export shim. |
| B: Extract with rewrite | `src/network/{nostr,proxy,tunnel}`, `src/mesh/mod.rs` (~50%), `src/inference/{election}` (~60-70%) | I/O becomes trait-based. Filesystem and process calls are removed or abstracted. |
| C: Host-only (stays in mesh-llm) | `src/{runtime,api,cli,system,plugin,plugins}/*`, `src/inference/launch.rs`, `src/crypto/{keystore,keychain}.rs`, `src/models/{local,resolve,maintenance}.rs` | Desktop-only. Not extracted. |

Category B modules require the most care. The extraction recipe applies: failing test first, then minimal implementation, then re-export shim, then commit.

---

## Out of Scope

The following are explicitly excluded from this ADR and from all waves of this work:

1. Browser SDK
2. iOS host mode
3. Android host mode
4. Localhost HTTP proxy wrapper
5. Plugin host in SDK
6. Web console assets
7. Nostr publishing from SDK
8. Model download management from SDK

Any future work on these items requires a separate ADR.

---

## Mobile Platform Constraints

### App Store (iOS)

A `PrivacyInfo.xcprivacy` file is required and must be embedded inside the XCFramework bundle, not shipped separately. Apple's App Store review rejects frameworks that access privacy-sensitive APIs without a declared reason.

### Android

From August 2025, the Play Store requires 16KB page size alignment. All Android builds must pass the linker flag `-Wl,-z,max-page-size=16384`. This is a hard requirement, not optional.

### iOS Backgrounding

iOS suspends apps in the background. The `reconnect()` method exists specifically to re-establish the QUIC connection after the app returns to the foreground. App code must call `reconnect()` from a `UIApplication.willEnterForegroundNotification` observer. The SDK does not attempt automatic reconnection.

### Tokio Runtime

The embedded SDK runs a dedicated Tokio runtime on its own OS thread. Shutdown uses `runtime.shutdown_timeout(Duration::from_secs(5))` called from that dedicated thread. This avoids blocking the main thread and prevents the 5-second watchdog from triggering on iOS.

### Credentials

Credentials are passed by constructor (`OwnerKeypair` in `MeshClient::new` and `MeshHost::new`). The SDK never reads credentials from the filesystem. This is a hard constraint for App Store compliance and sandbox safety.

---

## Protocol Compatibility

The mesh-llm control plane supports two protocol versions:

- `mesh-llm/0`: legacy JSON/raw payloads, preserved for backward compatibility
- `mesh-llm/1`: protobuf framing with `meshllm.node.v1` schema, preferred for all new connections

Both versions are already in production. Mixed meshes containing `/0` and `/1` nodes are supported today. The embedded client is a permanently supported client type and will negotiate the highest version the host supports.

The `mesh-llm/0` ↔ `mesh-llm/1` dual-support scheme is a compatibility guarantee for all embedded clients (iOS, macOS, Android). Embedded clients are not a transitional concern — they are a first-class, permanent part of the mesh-llm client ecosystem. Any change to `src/protocol/` that breaks embedded client compatibility is a breaking change and requires a new ADR revision before it may be merged.

Any change to `src/protocol/` requires a backward-compatibility test before merging. This applies to both the existing mesh-llm binary and the new embedded SDK. Breaking the wire format without an explicit version bump is not acceptable.

---

## Test Strategy

All extraction work follows TDD:

1. **RED**: Write a failing test in the destination crate (`mesh-client`, `mesh-api-ffi`, etc.)
2. **GREEN**: Copy the minimal implementation to make it pass
3. **REFACTOR**: Clean up, remove duplication, add the transitional re-export shim in the source crate

`FixtureMesh` in `mesh-llm-test-harness` has a hard cap: 1 struct, 3 public methods (`new`, `invite_token`, `Drop`), no traits, no generics. It exists to give tests a real mesh endpoint without pulling in the full mesh-llm binary.

Wave-end gates run before any wave is considered complete:

```bash
just build && cargo test --workspace && cargo clippy --workspace -- -D warnings
```

All three must exit 0. Clippy warnings are errors. No exceptions.

---

## Consequences

### Enables

Native iOS, macOS, and Android apps can join mesh-llm meshes without localhost HTTP, sidecar processes, or CLI invocation. App developers get a typed Swift or Kotlin API backed by the same QUIC transport and protocol as the desktop binary.

### Costs

- Ongoing maintenance of the FFI surface. Every API change must be reflected in the uniffi definition and regenerated bindings.
- uniffi version pinning. Upgrading uniffi may require regenerating all bindings and updating consumer packages.
- Extraction discipline. Category B modules require careful rewriting to remove I/O assumptions. Rushing this step risks subtle behavioral differences between the embedded client and the desktop binary.

### Alternatives Rejected

- **swift-bridge**: Rust-only. No Kotlin support. Would require a separate solution for Android, doubling the maintenance surface.
- **diplomat**: Less mature, smaller ecosystem, fewer production deployments at the time of this decision.
- **cbindgen**: C ABI only. No async support. Would require manual async bridging on both Swift and Kotlin sides, negating most of the benefit.
