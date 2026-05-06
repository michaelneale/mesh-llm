#![allow(unused)]

use mesh_api::{
    ClientBuilder, InviteToken, MeshClient, Model, OwnerKeypair, PublicMesh, PublicMeshQuery,
    Status,
};
use std::str::FromStr;

#[test]
fn client_builder_with_keypair_and_token() {
    let kp = OwnerKeypair::generate();
    let token = InviteToken::from_str("mesh-test:abc123").expect("valid token");
    let _builder = ClientBuilder::new(kp, token);
}

#[test]
fn client_builder_builds_mesh_client() {
    let kp = OwnerKeypair::generate();
    let token = InviteToken::from_str("mesh-test:abc123").expect("valid token");
    let builder = ClientBuilder::new(kp, token);
    let _client: MeshClient = builder.build().expect("build");
}

#[test]
fn mesh_client_has_reconnect_method() {
    fn _assert_reconnect(c: &mut MeshClient) {
        drop(c.reconnect());
    }
}

#[test]
fn public_mesh_query_is_constructible() {
    let _query = PublicMeshQuery::default();
}

#[test]
fn public_mesh_can_build_client_builder() {
    let mesh = PublicMesh {
        invite_token: "mesh-test:abc123".to_string(),
        serving: vec!["Qwen".to_string()],
        wanted: vec![],
        on_disk: vec![],
        total_vram_bytes: 32_000_000_000,
        node_count: 2,
        client_count: 1,
        max_clients: 0,
        name: Some("mesh-llm".to_string()),
        region: Some("AU".to_string()),
        mesh_id: Some("mesh-1".to_string()),
        publisher_npub: "npub1test".to_string(),
        published_at: 1,
        expires_at: Some(2),
    };
    let builder = mesh.client_builder(OwnerKeypair::generate());
    assert!(builder.is_ok());
}
