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

Shared protocol-facing model/type definitions are owned by
`crates/mesh-llm-types/` and re-exported here where existing client call sites
expect them. Keep pure shared data there instead of adding host-runtime
dependencies to this crate.

Shared owner identity and envelope crypto are owned by
`crates/mesh-llm-identity/` and re-exported here for compatibility with existing
client call sites.

Shared protobuf types and frame helpers are owned by `crates/mesh-llm-protocol/`
and re-exported here for compatibility with existing client call sites.

Shared routing targets and model placement helpers are owned by
`crates/mesh-llm-routing/` and re-exported here for compatibility with existing
client call sites. Client-only runtime process details stay in this crate.

GGUF artifact metadata scanning is owned by `crates/model-artifact/` and
re-exported here for compatibility. Keep file-format parsing in model
infrastructure crates rather than in client runtime code.

Client requests should preserve the full model ref chosen by the caller. Model
resolution, stage topology, and runtime lifecycle remain server-side mesh
responsibilities.
