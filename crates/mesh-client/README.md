# mesh-client

`mesh-client` is the low-level Rust client implementation crate for embedded
Mesh integrations.

This crate owns client-side protocol, transport, and runtime behavior used by
higher-level SDK surfaces. It is not intended to be the primary application
integration boundary.

Most consumers should depend on:

- `crates/mesh-api/` for the public Rust client SDK

Language bindings should generally reach this crate through:

- `crates/mesh-api/`
- `crates/mesh-api-ffi/`

Keep this crate implementation-focused. Public, app-facing ergonomics should be
added in `crates/mesh-api/`, not here.

Client requests should preserve the full model ref chosen by the caller. Model
resolution, stage topology, and runtime lifecycle remain server-side mesh
responsibilities.
