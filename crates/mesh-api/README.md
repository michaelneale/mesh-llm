# mesh-api

`mesh-api` is the public Rust client SDK for embedding Mesh in applications.

This is the crate that Rust-native consumers should depend on when they want to:

- join a mesh
- list models
- submit chat or responses requests
- observe client events
- manage connection lifecycle

Layering:

- `crates/mesh-client/` implements the low-level client behavior
- `crates/mesh-api/` exposes the stable Rust SDK surface
- `crates/mesh-api-ffi/` wraps `crates/mesh-api/` for Swift, Kotlin, and other native
  bindings

If an API is meant for app integration, it should live here rather than in
`crates/mesh-client/`.
