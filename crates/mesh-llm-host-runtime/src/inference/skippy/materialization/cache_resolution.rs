use std::{
    cmp::Reverse,
    fs,
    path::{Path, PathBuf},
    time::SystemTime,
};

use anyhow::{Context, Result};

use super::{manifest_artifact_bytes, safe_manifest_file_path, verify_cached_hf_package_files};

pub(super) fn cached_package_snapshots(
    cache_dir: &Path,
    repo_folder: &str,
) -> Result<Vec<PathBuf>> {
    let snapshots_dir = cache_dir.join(repo_folder).join("snapshots");
    let Ok(entries) = fs::read_dir(&snapshots_dir) else {
        return Ok(Vec::new());
    };
    let mut snapshots = Vec::new();
    for entry in entries {
        let entry =
            entry.with_context(|| format!("read snapshot entry {}", snapshots_dir.display()))?;
        let path = entry.path();
        if path.join("model-package.json").is_file() {
            snapshots.push(path);
        }
    }
    snapshots.sort_by_key(|path| cached_snapshot_sort_key(path));
    Ok(snapshots)
}

pub(super) fn resolve_cached_hf_package_snapshot(
    package_dir: &Path,
    layer_start: u32,
    layer_end: u32,
    include_embeddings: bool,
    include_output: bool,
) -> Result<Option<String>> {
    if !should_prefer_cached_snapshot_for_request(
        package_dir,
        layer_start,
        layer_end,
        include_embeddings,
        include_output,
    )? {
        tracing::debug!(
            package_dir = %package_dir.display(),
            "cached HF layer package snapshot is metadata-only or incomplete; looking for a better snapshot"
        );
        return Ok(None);
    }
    verify_cached_hf_package_files(
        package_dir,
        layer_start,
        layer_end,
        include_embeddings,
        include_output,
    )
}

fn cached_snapshot_sort_key(path: &Path) -> (Reverse<SystemTime>, PathBuf) {
    let modified = fs::metadata(path)
        .and_then(|metadata| metadata.modified())
        .unwrap_or(SystemTime::UNIX_EPOCH);
    (Reverse(modified), path.to_path_buf())
}

fn should_prefer_cached_snapshot_for_request(
    package_dir: &Path,
    _layer_start: u32,
    _layer_end: u32,
    _include_embeddings: bool,
    _include_output: bool,
) -> Result<bool> {
    // Always verify that the snapshot has its declared layer artifacts on
    // disk.  A metadata-only snapshot (model-package.json + shared/ but no
    // layers/) must never be selected — not even for metadata-only probes —
    // because the snapshot hash gets baked into the canonical package_ref
    // and propagated to all later resolution calls (split stage loads,
    // activation-width inference, etc.) which then fail when the layers
    // are missing.
    cached_snapshot_has_declared_layer_artifacts(package_dir)
}

fn cached_snapshot_has_declared_layer_artifacts(package_dir: &Path) -> Result<bool> {
    let manifest_contents =
        fs::read(package_dir.join("model-package.json")).context("read cached package manifest")?;
    let manifest: serde_json::Value =
        serde_json::from_slice(&manifest_contents).context("parse cached package manifest")?;
    let Some(layers) = manifest.get("layers").and_then(|layers| layers.as_array()) else {
        return Ok(false);
    };
    if layers.is_empty() {
        return Ok(false);
    }
    for layer in layers {
        let Some(path) = layer.get("path").and_then(|path| path.as_str()) else {
            return Ok(false);
        };
        let path = safe_manifest_file_path(path)?;
        let Ok(metadata) = fs::metadata(package_dir.join(path)) else {
            return Ok(false);
        };
        if !metadata.is_file() {
            return Ok(false);
        }
        if let Some(expected_bytes) = manifest_artifact_bytes(layer) {
            if metadata.len() != expected_bytes {
                return Ok(false);
            }
        }
    }
    Ok(true)
}
