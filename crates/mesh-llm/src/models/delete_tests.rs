use std::ffi::OsString;
use std::path::{Path, PathBuf};

use serial_test::serial;

use crate::models::delete::{delete_model_by_identifier, resolve_model_identifier};
use crate::models::resolve::resolve_huggingface_file_from_sibling_entries;

fn unique_temp_dir(prefix: &str) -> PathBuf {
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("mesh-llm-{prefix}-{stamp}"))
}

fn restore_env(key: &str, previous: Option<OsString>) {
    if let Some(value) = previous {
        std::env::set_var(key, value);
    } else {
        std::env::remove_var(key);
    }
}

fn create_cache_repo_file(
    root: &Path,
    repo_id: &str,
    revision: &str,
    relative_file: &str,
    size_bytes: usize,
) -> PathBuf {
    let repo_dir = root.join(format!("models--{}", repo_id.replace('/', "--")));
    let refs_dir = repo_dir.join("refs");
    let snapshot_dir = repo_dir.join("snapshots").join(revision);
    std::fs::create_dir_all(&refs_dir).unwrap();
    std::fs::create_dir_all(
        snapshot_dir.join(Path::new(relative_file).parent().unwrap_or(Path::new(""))),
    )
    .unwrap();
    std::fs::write(refs_dir.join("main"), revision).unwrap();

    let path = snapshot_dir.join(relative_file);
    std::fs::write(&path, vec![0u8; size_bytes]).unwrap();
    path
}

#[tokio::test]
async fn resolve_model_identifier_rejects_filesystem_paths() {
    let err = resolve_model_identifier("/tmp/model.gguf")
        .await
        .unwrap_err();
    assert!(err
        .to_string()
        .contains("does not support filesystem paths"));
}

#[tokio::test]
async fn resolve_model_identifier_rejects_direct_urls() {
    let err = resolve_model_identifier("https://huggingface.co/org/repo/resolve/main/model.gguf")
        .await
        .unwrap_err();
    assert!(err.to_string().contains("does not support direct URLs"));
}

#[tokio::test]
#[serial]
async fn resolve_model_identifier_returns_all_split_shards_from_selector_ref() {
    let prev_hub_cache = std::env::var_os("HF_HUB_CACHE");
    let prev_hf_home = std::env::var_os("HF_HOME");
    let prev_xdg = std::env::var_os("XDG_CACHE_HOME");

    let temp = unique_temp_dir("delete-split-resolve");
    let shard1 = create_cache_repo_file(
        &temp,
        "bartowski/GLM-5-UD-IQ2_XXS-GGUF",
        "abcdef1234567890",
        "GLM-5-UD-IQ2_XXS-00001-of-00002.gguf",
        4,
    );
    let shard2 = create_cache_repo_file(
        &temp,
        "bartowski/GLM-5-UD-IQ2_XXS-GGUF",
        "abcdef1234567890",
        "GLM-5-UD-IQ2_XXS-00002-of-00002.gguf",
        4,
    );
    let unrelated = create_cache_repo_file(
        &temp,
        "bartowski/GLM-5-UD-IQ2_XXS-GGUF",
        "abcdef1234567890",
        "GLM-5-UD-IQ2_XXS-Q4_K_M.gguf",
        4,
    );

    std::env::set_var("HF_HUB_CACHE", &temp);
    std::env::remove_var("HF_HOME");
    std::env::remove_var("XDG_CACHE_HOME");

    let resolved = resolve_model_identifier("bartowski/GLM-5-UD-IQ2_XXS-GGUF:UD-IQ2_XXS")
        .await
        .unwrap();
    assert_eq!(resolved, vec![shard1.clone(), shard2.clone()]);
    assert!(unrelated.exists());

    let _ = std::fs::remove_dir_all(&temp);
    restore_env("HF_HUB_CACHE", prev_hub_cache);
    restore_env("HF_HOME", prev_hf_home);
    restore_env("XDG_CACHE_HOME", prev_xdg);
}

#[tokio::test]
#[serial]
async fn delete_model_by_identifier_removes_only_the_resolved_split_shards() {
    let prev_hub_cache = std::env::var_os("HF_HUB_CACHE");
    let prev_hf_home = std::env::var_os("HF_HOME");
    let prev_xdg = std::env::var_os("XDG_CACHE_HOME");

    let temp = unique_temp_dir("delete-split-target");
    let shard1 = create_cache_repo_file(
        &temp,
        "bartowski/GLM-5-UD-IQ2_XXS-GGUF",
        "abcdef1234567890",
        "GLM-5-UD-IQ2_XXS-00001-of-00002.gguf",
        4,
    );
    let shard2 = create_cache_repo_file(
        &temp,
        "bartowski/GLM-5-UD-IQ2_XXS-GGUF",
        "abcdef1234567890",
        "GLM-5-UD-IQ2_XXS-00002-of-00002.gguf",
        4,
    );
    let unrelated = create_cache_repo_file(
        &temp,
        "bartowski/GLM-5-UD-IQ2_XXS-GGUF",
        "abcdef1234567890",
        "GLM-5-UD-IQ2_XXS-Q4_K_M.gguf",
        4,
    );

    std::env::set_var("HF_HUB_CACHE", &temp);
    std::env::remove_var("HF_HOME");
    std::env::remove_var("XDG_CACHE_HOME");

    let expected_deleted = vec![
        shard1.canonicalize().unwrap(),
        shard2.canonicalize().unwrap(),
    ];
    let result = delete_model_by_identifier("bartowski/GLM-5-UD-IQ2_XXS-GGUF:UD-IQ2_XXS")
        .await
        .unwrap();
    assert_eq!(result.deleted_paths, expected_deleted);
    assert!(!shard1.exists());
    assert!(!shard2.exists());
    assert!(unrelated.exists());

    let _ = std::fs::remove_dir_all(&temp);
    restore_env("HF_HUB_CACHE", prev_hub_cache);
    restore_env("HF_HOME", prev_hf_home);
    restore_env("XDG_CACHE_HOME", prev_xdg);
}

#[tokio::test]
#[serial]
async fn delete_model_by_identifier_supports_dotted_quant_selector_refs() {
    let prev_hub_cache = std::env::var_os("HF_HUB_CACHE");
    let prev_hf_home = std::env::var_os("HF_HOME");
    let prev_xdg = std::env::var_os("XDG_CACHE_HOME");

    let temp = unique_temp_dir("delete-dotted-selector");
    let q2 = create_cache_repo_file(
        &temp,
        "Flexan/kshitijthakkar-qwen3.5-moe-0.87B-d0.8B-GGUF",
        "a9b8adbec2cc87479c772dac1944f313b4036c26",
        "qwen3.5-moe-0.87B-d0.8B.Q2_K.gguf",
        4,
    );
    let q4 = create_cache_repo_file(
        &temp,
        "Flexan/kshitijthakkar-qwen3.5-moe-0.87B-d0.8B-GGUF",
        "a9b8adbec2cc87479c772dac1944f313b4036c26",
        "qwen3.5-moe-0.87B-d0.8B.Q4_K_M.gguf",
        4,
    );

    std::env::set_var("HF_HUB_CACHE", &temp);
    std::env::remove_var("HF_HOME");
    std::env::remove_var("XDG_CACHE_HOME");

    let resolved =
        resolve_model_identifier("Flexan/kshitijthakkar-qwen3.5-moe-0.87B-d0.8B-GGUF:Q2_K")
            .await
            .unwrap();
    assert_eq!(resolved, vec![q2.clone()]);

    let expected_deleted = vec![q2.canonicalize().unwrap()];
    let result =
        delete_model_by_identifier("Flexan/kshitijthakkar-qwen3.5-moe-0.87B-d0.8B-GGUF:Q2_K")
            .await
            .unwrap();
    assert_eq!(result.deleted_paths, expected_deleted);
    assert!(!q2.exists());
    assert!(q4.exists());

    let _ = std::fs::remove_dir_all(&temp);
    restore_env("HF_HUB_CACHE", prev_hub_cache);
    restore_env("HF_HOME", prev_hf_home);
    restore_env("XDG_CACHE_HOME", prev_xdg);
}

#[tokio::test]
#[serial]
async fn resolve_model_identifier_repo_ref_matches_shared_resolver_semantics() {
    let prev_hub_cache = std::env::var_os("HF_HUB_CACHE");
    let prev_hf_home = std::env::var_os("HF_HOME");
    let prev_xdg = std::env::var_os("XDG_CACHE_HOME");

    let temp = unique_temp_dir("delete-default-repo");
    let shard1 = create_cache_repo_file(
        &temp,
        "bartowski/GLM-5-UD-IQ2_XXS-GGUF",
        "abcdef1234567890",
        "GLM-5-UD-IQ2_XXS-00001-of-00002.gguf",
        64,
    );
    let shard2 = create_cache_repo_file(
        &temp,
        "bartowski/GLM-5-UD-IQ2_XXS-GGUF",
        "abcdef1234567890",
        "GLM-5-UD-IQ2_XXS-00002-of-00002.gguf",
        64,
    );
    let bf16 = create_cache_repo_file(
        &temp,
        "bartowski/GLM-5-UD-IQ2_XXS-GGUF",
        "abcdef1234567890",
        "BF16/GLM-5-UD-BF16.gguf",
        128,
    );

    std::env::set_var("HF_HUB_CACHE", &temp);
    std::env::remove_var("HF_HOME");
    std::env::remove_var("XDG_CACHE_HOME");

    let sibling_entries = vec![
        ("GLM-5-UD-IQ2_XXS-00001-of-00002.gguf".to_string(), Some(64)),
        ("GLM-5-UD-IQ2_XXS-00002-of-00002.gguf".to_string(), Some(64)),
        ("BF16/GLM-5-UD-BF16.gguf".to_string(), Some(128)),
    ];
    let selected = resolve_huggingface_file_from_sibling_entries(
        "bartowski/GLM-5-UD-IQ2_XXS-GGUF",
        Some("main"),
        "",
        &sibling_entries,
    )
    .await
    .unwrap();

    let resolved = resolve_model_identifier("bartowski/GLM-5-UD-IQ2_XXS-GGUF")
        .await
        .unwrap();
    let resolved: Vec<PathBuf> = resolved
        .into_iter()
        .map(|path| path.canonicalize().unwrap())
        .collect();
    let expected = if selected == "BF16/GLM-5-UD-BF16.gguf" {
        vec![bf16.canonicalize().unwrap()]
    } else {
        vec![
            shard1.canonicalize().unwrap(),
            shard2.canonicalize().unwrap(),
        ]
    };
    assert_eq!(resolved, expected);

    let _ = std::fs::remove_dir_all(&temp);
    restore_env("HF_HUB_CACHE", prev_hub_cache);
    restore_env("HF_HOME", prev_hf_home);
    restore_env("XDG_CACHE_HOME", prev_xdg);
}
