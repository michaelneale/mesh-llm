#![forbid(unsafe_code)]

mod client;
mod discover;
pub mod events;

pub use client::{
    ChatMessage, ChatRequest, ClientBuilder, ClientConfig, MeshApiError, MeshClient, Model,
    RequestId, ResponsesRequest, Status, MAX_RECONNECT_ATTEMPTS,
};
pub use discover::{
    create_auto_client, discover_public_meshes, AutoConnectResult, PublicMesh, PublicMeshQuery,
};
pub use identity::OwnerKeypair;
pub use token::InviteToken;

mod identity;
mod token;
