use mesh_client::client::builder::{ClientBuilder, InviteToken, RequestId};
use mesh_client::crypto::keys::OwnerKeypair;
use std::str::FromStr;

#[test]
fn cancel_unknown_request_id_is_noop() {
    let kp = OwnerKeypair::generate();
    let token = InviteToken::from_str("test-token").unwrap();
    let client = ClientBuilder::new(kp, token).build().unwrap();

    client.cancel(RequestId("unknown-id".to_string()));
    client.cancel(RequestId("another-unknown".to_string()));
}

#[test]
fn cancel_already_completed_request_is_noop() {
    let kp = OwnerKeypair::generate();
    let token = InviteToken::from_str("test-token").unwrap();
    let client = ClientBuilder::new(kp, token).build().unwrap();

    client.cancel(RequestId("completed-id".to_string()));
}
