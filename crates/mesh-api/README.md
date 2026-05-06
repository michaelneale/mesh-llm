# mesh-api

`mesh-api` is the public Rust client SDK for embedding Mesh in applications.

This is the crate that Rust-native consumers should depend on when they want to:

- join a mesh
- discover public meshes
- smart-auto connect to the best public mesh
- list models
- submit chat or responses requests
- observe client events
- manage connection lifecycle

Layering:

- `crates/mesh-client/` implements the low-level client behavior
- `crates/mesh-api/` exposes the stable Rust SDK surface
- `crates/mesh-api-ffi/` wraps `crates/mesh-api/` for Swift, Kotlin, and other native
  bindings

This SDK talks to mesh's public API and OpenAI surface. It does not embed
skippy, plan stage topology, or manage local model packages directly.

If an API is meant for app integration, it should live here rather than in
`crates/mesh-client/`.

Typical public-mesh flow:

```rust
use mesh_api::{discover_public_meshes, ClientBuilder, OwnerKeypair, PublicMeshQuery};

# async fn demo() -> Result<(), Box<dyn std::error::Error>> {
let meshes = discover_public_meshes(PublicMeshQuery::default()).await?;
let selected = meshes.first().expect("at least one public mesh");

let mut client = selected
    .client_builder(OwnerKeypair::generate())?
    .build()?;
client.join().await?;
# Ok(())
# }
```
