# SDK

This directory contains the language-specific Mesh SDK packages built on top of
the shared native client core.

Current SDKs:

- `swift/` for Apple platforms
- `kotlin/` for Android and JVM consumers

These SDK packages should stay thin. Shared client behavior belongs in the Rust
SDK crates:

- `crates/mesh-client/` for the low-level client implementation
- `crates/mesh-api/` for the public Rust client API
- `crates/mesh-api-ffi/` for the UniFFI/native bridge used by language SDKs

If you add another top-level SDK here, include a `README.md` in that SDK
directory explaining its packaging and public surface.
