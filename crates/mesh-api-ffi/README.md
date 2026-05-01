# mesh-api-ffi

`mesh-api-ffi` exposes the Mesh client SDK through a native FFI layer for
language bindings.

This crate is the bridge used by the generated Swift and Kotlin SDKs. It should
stay thin and map the public Rust API from `crates/mesh-api/` into an FFI-safe surface.

Layering:

- `crates/mesh-client/` implements low-level client behavior
- `crates/mesh-api/` defines the public Rust SDK
- `crates/mesh-api-ffi/` adapts that SDK for cross-language consumers

Application code should usually depend on `crates/mesh-api/` directly unless it is
building a non-Rust binding.
