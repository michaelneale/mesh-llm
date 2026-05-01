use mesh_client::crypto::{keys::OwnerKeypair, open_message, seal_message};

#[test]
fn seal_open_roundtrip() {
    let sender = OwnerKeypair::generate();
    let recipient = OwnerKeypair::generate();
    let payload = b"hello from mesh-client";
    let ts = 1_700_000_000_000u64;

    let envelope = seal_message(
        &sender,
        &recipient.encryption_public_key(),
        "test.roundtrip",
        payload,
        ts,
    )
    .expect("seal");

    let opened = open_message(&recipient, &envelope).expect("open");
    assert_eq!(opened.payload, payload);
    assert_eq!(opened.message_type, "test.roundtrip");
    assert_eq!(opened.timestamp_unix_ms, ts);
    assert_eq!(opened.sender_owner_id, sender.owner_id());
}

#[test]
fn keypair_bytes_roundtrip() {
    let kp = OwnerKeypair::generate();
    let signing = kp.signing_bytes().to_vec();
    let encryption = kp.encryption_bytes().to_vec();
    let restored = OwnerKeypair::from_bytes(&signing, &encryption).expect("roundtrip");
    assert_eq!(kp.owner_id(), restored.owner_id());
    assert_eq!(
        kp.encryption_public_key().as_bytes(),
        restored.encryption_public_key().as_bytes(),
    );
}
