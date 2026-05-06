use mesh_client::inference::election::{
    should_be_host_for_model, total_model_bytes, InferenceTarget,
};

#[test]
fn election_types_accessible() {
    let target = InferenceTarget::None;
    assert!(matches!(target, InferenceTarget::None));
}

#[test]
fn total_model_bytes_nonexistent_returns_zero() {
    let bytes = total_model_bytes(std::path::Path::new("/nonexistent/model.gguf"));
    assert_eq!(bytes, 0);
}

#[test]
fn total_model_bytes_real_file() {
    let path = std::env::temp_dir().join("mesh_election_test_model.gguf");
    std::fs::write(&path, b"fake gguf content").unwrap();
    let bytes = total_model_bytes(&path);
    let _ = std::fs::remove_file(&path);
    assert_eq!(bytes, 17);
}

#[test]
fn should_be_host_when_no_peers() {
    use iroh::{EndpointId, SecretKey};
    let my_id = EndpointId::from(SecretKey::from_bytes(&[0x01; 32]).public());
    assert!(should_be_host_for_model(my_id, 48_000_000_000, &[]));
}
