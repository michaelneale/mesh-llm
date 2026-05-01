use mesh_client::client::builder::{ClientBuilder, InviteToken, MAX_RECONNECT_ATTEMPTS};
use mesh_client::crypto::keys::OwnerKeypair;
use std::str::FromStr;

#[tokio::test]
async fn max_reconnect_attempts_is_ten() {
    assert_eq!(MAX_RECONNECT_ATTEMPTS, 10);
}

#[tokio::test]
async fn reconnect_resets_attempt_counter() {
    let kp = OwnerKeypair::generate();
    let token = InviteToken::from_str("test-token").unwrap();
    let mut client = ClientBuilder::new(kp, token).build().unwrap();

    client.reconnect_attempts = 7;
    client.reconnect().await.unwrap();

    assert_eq!(
        client.reconnect_attempts, 0,
        "manual reconnect() must reset the attempt counter"
    );
}

#[tokio::test]
async fn reconnect_attempts_does_not_exceed_max() {
    let kp = OwnerKeypair::generate();
    let token = InviteToken::from_str("test-token").unwrap();
    let client = ClientBuilder::new(kp, token).build().unwrap();

    assert!(
        client.reconnect_attempts <= MAX_RECONNECT_ATTEMPTS,
        "attempt counter must never exceed MAX_RECONNECT_ATTEMPTS"
    );
}

#[tokio::test]
async fn explicit_disconnect_sets_user_disconnected_flag() {
    let kp = OwnerKeypair::generate();
    let token = InviteToken::from_str("test-token").unwrap();
    let mut client = ClientBuilder::new(kp, token).build().unwrap();

    client.join().await.unwrap();
    assert!(!client.user_disconnected);

    client.disconnect().await;
    assert!(
        client.user_disconnected,
        "explicit disconnect() must set user_disconnected so auto-reconnect does not fire"
    );
}

#[tokio::test]
async fn reconnect_clears_user_disconnected_flag() {
    let kp = OwnerKeypair::generate();
    let token = InviteToken::from_str("test-token").unwrap();
    let mut client = ClientBuilder::new(kp, token).build().unwrap();

    client.disconnect().await;
    assert!(client.user_disconnected);

    client.reconnect().await.unwrap();
    assert!(
        !client.user_disconnected,
        "reconnect() must clear user_disconnected to allow future auto-reconnect"
    );
}

#[tokio::test]
#[ignore]
async fn auto_reconnect_retries_with_backoff_on_real_network() {
    // This test requires a real mesh network endpoint and is skipped in CI.
    // Manual verification: start a mesh node, connect, kill the node, observe
    // that try_auto_reconnect() fires with growing delays up to MAX_RECONNECT_ATTEMPTS.
}
