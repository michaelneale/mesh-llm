use mesh_client::models::capabilities::ModelCapabilities;
use mesh_client::models::catalog::MODEL_CATALOG;
use mesh_client::models::gguf::scan_gguf_compact_meta;

fn push_gguf_string(bytes: &mut Vec<u8>, value: &str) {
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value.as_bytes());
}

fn push_u32_kv(bytes: &mut Vec<u8>, key: &str, value: u32) {
    push_gguf_string(bytes, key);
    bytes.extend_from_slice(&4u32.to_le_bytes());
    bytes.extend_from_slice(&value.to_le_bytes());
}

#[test]
fn catalog_has_entries() {
    assert!(MODEL_CATALOG.iter().count() > 0);
}

#[test]
fn capabilities_default_is_none() {
    let caps = ModelCapabilities::default();
    assert!(!caps.multimodal);
    assert!(!caps.moe);
}

#[test]
fn gguf_parse_minimal_fixture() {
    use std::io::Write;

    // Minimal valid GGUF: magic + version=3 + n_tensors=0 + n_kv=0
    let mut fixture = Vec::<u8>::new();
    fixture.extend_from_slice(b"GGUF"); // magic
    fixture.extend_from_slice(&3u32.to_le_bytes()); // version
    fixture.extend_from_slice(&0i64.to_le_bytes()); // n_tensors
    fixture.extend_from_slice(&0i64.to_le_bytes()); // n_kv

    let tmp = std::env::temp_dir().join("mesh-client-models-extraction.gguf");
    std::fs::File::create(&tmp)
        .unwrap()
        .write_all(&fixture)
        .unwrap();
    let meta = scan_gguf_compact_meta(&tmp);
    assert!(meta.is_some(), "should parse minimal GGUF fixture");
    let meta = meta.unwrap();
    assert_eq!(meta.context_length, 0);
    assert_eq!(meta.expert_count, 0);
    let _ = std::fs::remove_file(&tmp);
}

#[test]
fn gguf_public_api_derives_value_length_from_kv_heads() {
    use std::io::Write;

    let mut fixture = Vec::<u8>::new();
    fixture.extend_from_slice(b"GGUF");
    fixture.extend_from_slice(&3u32.to_le_bytes());
    fixture.extend_from_slice(&0i64.to_le_bytes());
    fixture.extend_from_slice(&2i64.to_le_bytes());
    push_u32_kv(&mut fixture, "llama.embedding_length", 4096);
    push_u32_kv(&mut fixture, "llama.attention.head_count_kv", 8);

    let tmp = std::env::temp_dir().join("mesh-client-models-extraction-kv-heads.gguf");
    std::fs::File::create(&tmp)
        .unwrap()
        .write_all(&fixture)
        .unwrap();
    let meta = scan_gguf_compact_meta(&tmp).expect("should parse GGUF fixture");
    assert_eq!(meta.head_count, 0);
    assert_eq!(meta.head_count_kv, Some(8));
    assert_eq!(meta.key_length, 0);
    assert_eq!(meta.value_length, 512);
    let _ = std::fs::remove_file(&tmp);
}
