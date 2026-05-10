//! Integration tests that exercise real HuggingFace API calls and downloads.
//!
//! These tests hit the network and are **ignored by default**. Run them with:
//!
//! ```sh
//! cargo test -p model-hf --test hf_download -- --ignored
//! ```
//!
//! The download tests use `jc-builds/SmolLM2-135M-Instruct-Q4_K_M-GGUF` — the
//! same tiny model already cached in CI — so they add no new external dependency.
//!
//! The split-GGUF resolution tests use `unsloth/gemma-3-27b-it-GGUF` for file
//! listing only (no download).

use model_artifact::{resolve_model_artifact_ref, ModelArtifactFile, ModelFormat, ModelRepository};
use model_hf::HfModelRepository;
use std::path::Path;

/// Build a repository pointed at a fresh temp cache directory.
fn make_repo(tmp: &Path) -> HfModelRepository {
    HfModelRepository::builder()
        .cache_dir(tmp.join("hf-cache"))
        .build()
        .expect("build HfModelRepository")
}

// ---------------------------------------------------------------------------
// API-only tests (no file downloads — just HF HTTP API calls)
// ---------------------------------------------------------------------------

/// Verify we can resolve a revision (commit SHA) for a known public repo.
#[tokio::test]
#[ignore]
async fn resolve_revision_returns_commit_sha() {
    let tmp = tempfile::tempdir().unwrap();
    let repo = make_repo(tmp.path());

    let sha = repo
        .resolve_revision("jc-builds/SmolLM2-135M-Instruct-Q4_K_M-GGUF", Some("main"))
        .await
        .expect("resolve_revision");

    // HF commit SHAs are 40-char hex strings.
    assert_eq!(sha.len(), 40, "expected 40-char SHA, got: {sha}");
    assert!(
        sha.chars().all(|c| c.is_ascii_hexdigit()),
        "SHA contains non-hex chars: {sha}"
    );
    eprintln!("resolved revision: {sha}");
}

/// Verify list_files returns the expected GGUF file for a single-file repo.
#[tokio::test]
#[ignore]
async fn list_files_single_gguf_repo() {
    let tmp = tempfile::tempdir().unwrap();
    let repo = make_repo(tmp.path());

    let sha = repo
        .resolve_revision("jc-builds/SmolLM2-135M-Instruct-Q4_K_M-GGUF", Some("main"))
        .await
        .unwrap();
    let files = repo
        .list_files("jc-builds/SmolLM2-135M-Instruct-Q4_K_M-GGUF", &sha)
        .await
        .expect("list_files");

    let gguf_files: Vec<&ModelArtifactFile> =
        files.iter().filter(|f| f.path.ends_with(".gguf")).collect();
    assert_eq!(
        gguf_files.len(),
        1,
        "expected exactly 1 GGUF file, got: {gguf_files:?}"
    );
    assert_eq!(
        gguf_files[0].path, "SmolLM2-135M-Instruct.Q4_K_M.gguf",
        "unexpected GGUF filename"
    );
    eprintln!(
        "listed {n} files, GGUF size: {size:?} bytes",
        n = files.len(),
        size = gguf_files[0].size_bytes,
    );
}

/// Verify list_files finds split-GGUF shards in a multi-quant repo.
///
/// Uses `unsloth/gemma-3-27b-it-GGUF` which has `BF16/...-00001-of-00002.gguf`
/// split files. This is API-only — no files are downloaded.
#[tokio::test]
#[ignore]
async fn list_files_split_gguf_repo() {
    let tmp = tempfile::tempdir().unwrap();
    let repo = make_repo(tmp.path());

    let sha = repo
        .resolve_revision("unsloth/gemma-3-27b-it-GGUF", Some("main"))
        .await
        .unwrap();
    let files = repo
        .list_files("unsloth/gemma-3-27b-it-GGUF", &sha)
        .await
        .expect("list_files");

    let bf16_splits: Vec<&ModelArtifactFile> = files
        .iter()
        .filter(|f| f.path.starts_with("BF16/") && f.path.ends_with(".gguf"))
        .collect();

    assert!(
        bf16_splits.len() >= 2,
        "expected ≥2 BF16 split shards, got {}: {:?}",
        bf16_splits.len(),
        bf16_splits.iter().map(|f| &f.path).collect::<Vec<_>>()
    );

    let has_shard_1 = bf16_splits.iter().any(|f| f.path.contains("-00001-of-"));
    let has_shard_2 = bf16_splits.iter().any(|f| f.path.contains("-00002-of-"));
    assert!(has_shard_1, "missing shard 00001");
    assert!(has_shard_2, "missing shard 00002");

    eprintln!("found {n} BF16 split shards", n = bf16_splits.len());
}

/// Verify the full resolve_model_artifact_ref pipeline against a real repo.
/// This calls resolve_revision + list_files + artifact selection — no download.
#[tokio::test]
#[ignore]
async fn resolve_artifact_ref_single_gguf() {
    let tmp = tempfile::tempdir().unwrap();
    let repo = make_repo(tmp.path());

    let artifact =
        resolve_model_artifact_ref("jc-builds/SmolLM2-135M-Instruct-Q4_K_M-GGUF:Q4_K_M", &repo)
            .await
            .expect("resolve_model_artifact_ref");

    assert_eq!(artifact.format, ModelFormat::Gguf);
    assert_eq!(artifact.primary_file, "SmolLM2-135M-Instruct.Q4_K_M.gguf");
    assert_eq!(artifact.files.len(), 1);
    assert_eq!(
        artifact.source_repo,
        "jc-builds/SmolLM2-135M-Instruct-Q4_K_M-GGUF"
    );
    assert_eq!(artifact.source_revision.len(), 40);
    assert_eq!(artifact.selector.as_deref(), Some("Q4_K_M"));
    eprintln!(
        "resolved artifact: {} @ {} → {}",
        artifact.model_id, artifact.source_revision, artifact.primary_file
    );
}

/// Verify resolve_model_artifact_ref correctly resolves split-GGUF shards.
/// API-only — no files are downloaded.
#[tokio::test]
#[ignore]
async fn resolve_artifact_ref_split_gguf() {
    let tmp = tempfile::tempdir().unwrap();
    let repo = make_repo(tmp.path());

    let artifact = resolve_model_artifact_ref("unsloth/gemma-3-27b-it-GGUF:BF16", &repo)
        .await
        .expect("resolve_model_artifact_ref for split GGUF");

    assert_eq!(artifact.format, ModelFormat::Gguf);
    assert!(
        artifact.primary_file.contains("-00001-of-"),
        "primary_file should be shard 00001, got: {}",
        artifact.primary_file
    );
    assert!(
        artifact.files.len() >= 2,
        "expected ≥2 split shards, got {}",
        artifact.files.len()
    );

    // Verify shard ordering.
    let paths: Vec<&str> = artifact.files.iter().map(|f| f.path.as_str()).collect();
    for (i, path) in paths.iter().enumerate() {
        let expected_part = format!("-{:05}-of-", i + 1);
        assert!(
            path.contains(&expected_part),
            "shard {i} should contain '{expected_part}', got: {path}"
        );
    }
    eprintln!(
        "resolved split artifact: {} → {} shards, primary: {}",
        artifact.model_id,
        artifact.files.len(),
        artifact.primary_file
    );
}

// ---------------------------------------------------------------------------
// Download tests (actually fetch files from HuggingFace)
// ---------------------------------------------------------------------------

/// Download the actual GGUF file via the Rust HF client and verify the result.
///
/// Uses `jc-builds/SmolLM2-135M-Instruct-Q4_K_M-GGUF` (~100 MB). The same
/// model is already cached in CI, and this test uses a dedicated temp cache
/// directory so it always exercises the real download path.
#[tokio::test]
#[ignore]
async fn download_single_gguf_file() {
    let tmp = tempfile::tempdir().unwrap();
    let repo = make_repo(tmp.path());

    let sha = repo
        .resolve_revision("jc-builds/SmolLM2-135M-Instruct-Q4_K_M-GGUF", Some("main"))
        .await
        .unwrap();

    let path = repo
        .download_file(
            "jc-builds/SmolLM2-135M-Instruct-Q4_K_M-GGUF",
            &sha,
            "SmolLM2-135M-Instruct.Q4_K_M.gguf",
        )
        .await
        .expect("download_file");

    assert!(
        path.exists(),
        "downloaded file should exist: {}",
        path.display()
    );

    let metadata = std::fs::metadata(&path).expect("file metadata");
    let size = metadata.len();
    // SmolLM2-135M Q4_K_M is ~100 MB.
    assert!(
        size > 50_000_000,
        "downloaded file should be >50MB, got {size} bytes"
    );
    assert!(
        size < 200_000_000,
        "downloaded file should be <200MB, got {size} bytes"
    );

    // Verify the file starts with the GGUF magic bytes.
    let mut magic = [0u8; 4];
    let mut file = std::fs::File::open(&path).unwrap();
    std::io::Read::read_exact(&mut file, &mut magic).unwrap();
    assert_eq!(&magic, b"GGUF", "file should start with GGUF magic");

    eprintln!("downloaded {size} bytes to {}", path.display());
}

/// Full pipeline: resolve → download → identity.
///
/// Exercises the complete code path a user hits when running
/// `mesh-llm serve --model jc-builds/SmolLM2-135M-Instruct-Q4_K_M-GGUF:Q4_K_M`.
#[tokio::test]
#[ignore]
async fn full_resolve_download_identity_pipeline() {
    let tmp = tempfile::tempdir().unwrap();
    let repo = make_repo(tmp.path());

    // Step 1: Resolve artifact reference.
    let artifact =
        resolve_model_artifact_ref("jc-builds/SmolLM2-135M-Instruct-Q4_K_M-GGUF:Q4_K_M", &repo)
            .await
            .expect("resolve artifact");
    eprintln!(
        "step 1 — resolved: {} ({} files)",
        artifact.model_id,
        artifact.files.len()
    );

    // Step 2: Download all artifact files.
    let downloaded_paths = repo
        .download_artifact_files(&artifact)
        .await
        .expect("download artifact files");
    assert_eq!(
        downloaded_paths.len(),
        artifact.files.len(),
        "should download same number of files as resolved"
    );

    for path in &downloaded_paths {
        assert!(
            path.exists(),
            "downloaded file should exist: {}",
            path.display()
        );
        let size = std::fs::metadata(path).unwrap().len();
        assert!(size > 0, "downloaded file should not be empty");
        eprintln!("step 2 — downloaded: {} ({size} bytes)", path.display());
    }

    // Step 3: Verify identity for the downloaded path.
    let primary_path = &downloaded_paths[0];
    let identity = repo
        .identity_for_path(primary_path)
        .expect("identity_for_path should succeed for a file in the HF cache");

    assert_eq!(
        identity.repo_id,
        "jc-builds/SmolLM2-135M-Instruct-Q4_K_M-GGUF"
    );
    assert_eq!(identity.file, "SmolLM2-135M-Instruct.Q4_K_M.gguf");
    assert_eq!(identity.revision, artifact.source_revision);
    assert_eq!(identity.selector.as_deref(), Some("Q4_K_M"));
    assert!(
        !identity.canonical_ref.is_empty(),
        "canonical_ref should not be empty"
    );
    eprintln!(
        "step 3 — identity: model_id={}, canonical_ref={}",
        identity.model_id, identity.canonical_ref
    );
}

/// Verify that resolving a non-existent repo returns an error.
#[tokio::test]
#[ignore]
async fn resolve_nonexistent_repo_returns_error() {
    let tmp = tempfile::tempdir().unwrap();
    let repo = make_repo(tmp.path());

    let result = repo
        .resolve_revision("mesh-llm-ci/this-repo-does-not-exist-12345", Some("main"))
        .await;

    assert!(result.is_err(), "expected error for nonexistent repo");
    eprintln!("got expected error: {}", result.unwrap_err());
}

/// Verify that downloading a non-existent file returns an error.
#[tokio::test]
#[ignore]
async fn download_nonexistent_file_returns_error() {
    let tmp = tempfile::tempdir().unwrap();
    let repo = make_repo(tmp.path());

    let sha = repo
        .resolve_revision("jc-builds/SmolLM2-135M-Instruct-Q4_K_M-GGUF", Some("main"))
        .await
        .unwrap();

    let result = repo
        .download_file(
            "jc-builds/SmolLM2-135M-Instruct-Q4_K_M-GGUF",
            &sha,
            "nonexistent-file.gguf",
        )
        .await;

    assert!(result.is_err(), "expected error for nonexistent file");
    eprintln!("got expected error: {}", result.unwrap_err());
}
