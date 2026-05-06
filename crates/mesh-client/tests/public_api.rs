#![allow(unused)]
use mesh_client::{ClientBuilder, InviteToken, MeshClient, Model, OwnerKeypair, Status};
use std::str::FromStr;

#[test]
fn client_builder_with_keypair_and_token() {
    let kp = OwnerKeypair::generate();
    let token = InviteToken::from_str("mesh-test:abc123").expect("valid token");
    let _builder = ClientBuilder::new(kp, token);
    // Compile-time check of the API shape
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
    // Compile-time check that reconnect() exists
    fn _assert_reconnect(c: &mut MeshClient) {
        drop(c.reconnect());
    }
}
