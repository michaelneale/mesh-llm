use mesh_client::crypto::keys::OwnerKeypair;
use mesh_client::crypto::provider::{InMemoryKeyProvider, KeyProvider};

#[test]
fn in_memory_key_provider_roundtrip() {
    let kp = OwnerKeypair::generate();
    let seed = [42u8; 32];
    let mesh_id = Some(b"test-mesh".to_vec());
    let provider = InMemoryKeyProvider::new(kp.clone(), mesh_id.clone(), seed);

    let retrieved_kp = provider.owner_keypair().unwrap();
    assert_eq!(
        kp.verifying_key().as_bytes(),
        retrieved_kp.verifying_key().as_bytes()
    );
    assert_eq!(provider.mesh_id().unwrap(), mesh_id);
    assert_eq!(provider.node_id_seed().unwrap(), seed);
}

#[test]
fn in_memory_key_provider_no_mesh_id() {
    let kp = OwnerKeypair::generate();
    let provider = InMemoryKeyProvider::new(kp, None, [0u8; 32]);
    assert!(provider.mesh_id().unwrap().is_none());
}
