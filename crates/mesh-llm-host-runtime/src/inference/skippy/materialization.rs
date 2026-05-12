use std::{
    fs,
    io::Write,
    path::{Component, Path, PathBuf},
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use anyhow::{bail, Context, Result};
use hf_hub::{DownloadEvent, Progress, ProgressEvent, ProgressHandler};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use skippy_protocol::{LoadMode, StageConfig};
use skippy_runtime::package::{
    self, LayerPackageInfo, PackageIntegrityOptions, PackageStageRequest,
};

use crate::cli::output::{emit_event, interactive_tui_active, ModelProgressStatus, OutputEvent};
use crate::cli::terminal_progress::{start_spinner, SpinnerHandle};

use super::StageLoadRequest;

mod cache_resolution;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum StagePackageRef {
    LocalPackage(PathBuf),
    HuggingFacePackage {
        repo: String,
        revision: Option<String>,
    },
    SyntheticDirectGguf(PathBuf),
}

impl StagePackageRef {
    pub(crate) fn parse(value: &str) -> Result<Self> {
        if let Some(rest) = value.strip_prefix("hf://") {
            let (repo, revision) = if let Some((repo, revision)) = rest.split_once('@') {
                (repo, Some(revision.to_string()))
            } else if let Some(index) = rest.rfind(':') {
                (&rest[..index], Some(rest[index + 1..].to_string()))
            } else {
                (rest, None)
            };
            if repo.split('/').count() != 2 || repo.contains(':') || repo.contains('@') {
                bail!("HF package repo id must look like namespace/repo");
            }
            return Ok(Self::HuggingFacePackage {
                repo: repo.to_string(),
                revision,
            });
        }

        let path = PathBuf::from(value);
        if path.join("model-package.json").is_file() {
            return Ok(Self::LocalPackage(path));
        }
        if path.extension().and_then(|ext| ext.to_str()) == Some("gguf") {
            return Ok(Self::SyntheticDirectGguf(path));
        }

        bail!("not a skippy package ref: {value}");
    }

    pub(crate) fn is_distributable_package(&self) -> bool {
        matches!(
            self,
            Self::LocalPackage(_) | Self::HuggingFacePackage { .. }
        )
    }

    pub(crate) fn as_package_ref(&self) -> Option<String> {
        match self {
            Self::LocalPackage(path) => Some(path.to_string_lossy().to_string()),
            Self::HuggingFacePackage { repo, revision } => Some(match revision {
                Some(revision) => format!("hf://{repo}@{revision}"),
                None => format!("hf://{repo}"),
            }),
            Self::SyntheticDirectGguf(_) => None,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct StagePackageInfo {
    pub(crate) package_ref: String,
    pub(crate) package_dir: PathBuf,
    pub(crate) manifest_sha256: String,
    pub(crate) model_id: String,
    pub(crate) source_model_path: String,
    pub(crate) source_model_sha256: String,
    pub(crate) source_model_bytes: Option<u64>,
    pub(crate) layer_count: u32,
    pub(crate) activation_width: u32,
    pub(crate) projector_path: Option<String>,
    pub(crate) layers: Vec<StagePackageLayerInfo>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct StagePackageLayerInfo {
    pub(crate) layer_index: u32,
    pub(crate) tensor_count: usize,
    pub(crate) tensor_bytes: u64,
    pub(crate) artifact_bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct MaterializedStageArtifact {
    pub(crate) path: PathBuf,
    pub(crate) manifest_sha256: String,
    pub(crate) source_model_path: String,
    pub(crate) source_model_sha256: String,
    pub(crate) source_model_bytes: Option<u64>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ResolvedStagePackage {
    pub(crate) local_ref: String,
    pub(crate) source_model_path: String,
    pub(crate) source_model_sha256: String,
    pub(crate) source_model_bytes: Option<u64>,
}

#[derive(Debug)]
pub(crate) struct MaterializedStagePin {
    path: PathBuf,
}

impl Drop for MaterializedStagePin {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct PinFile {
    artifact_path: PathBuf,
    package_ref: String,
    topology_id: String,
    run_id: String,
    stage_id: String,
}

pub(crate) fn configure_materialized_stage_cache() {
    if std::env::var_os("SKIPPY_MATERIALIZED_DIR").is_none() {
        std::env::set_var("SKIPPY_MATERIALIZED_DIR", materialized_stage_cache_dir());
    }
}

pub(crate) fn materialized_stage_cache_dir() -> PathBuf {
    crate::models::mesh_llm_cache_dir().join("skippy-stages")
}

#[derive(Clone, Debug)]
struct LayerPackageDownloadProgressState {
    downloaded: u64,
    total: u64,
    bytes_per_sec: Option<f64>,
    last_draw: Option<Instant>,
    showed_progress: bool,
}

struct LayerPackageDownloadProgress {
    label: String,
    file: String,
    package_scope: Option<Arc<LayerPackageDownloadScope>>,
    completed_before: usize,
    preflight_spinner: Mutex<Option<SpinnerHandle>>,
    state: Mutex<LayerPackageDownloadProgressState>,
}

struct LayerPackageDownloadScope {
    package: String,
    total_files: usize,
    state: Mutex<LayerPackageDownloadScopeState>,
}

#[derive(Debug)]
struct LayerPackageDownloadScopeState {
    announced: bool,
    drawn_line: bool,
}

impl LayerPackageDownloadScope {
    fn new(label: &str, total_files: usize) -> Self {
        Self {
            package: layer_package_progress_package(label).to_string(),
            total_files,
            state: Mutex::new(LayerPackageDownloadScopeState {
                announced: false,
                drawn_line: false,
            }),
        }
    }

    fn has_drawn(&self) -> bool {
        self.state
            .lock()
            .map(|state| state.announced || state.drawn_line)
            .unwrap_or(false)
    }

    fn complete_count(&self, completed: usize) -> usize {
        completed.min(self.total_files)
    }

    fn draw(
        &self,
        file: &str,
        completed_files: usize,
        downloaded: u64,
        total: u64,
        bytes_per_sec: Option<f64>,
        force: bool,
    ) {
        let Ok(mut scope_state) = self.state.lock() else {
            return;
        };
        if !scope_state.announced {
            eprintln!(
                "\r\x1b[K📦 Downloading layer package {} ({} file(s))",
                self.package, self.total_files
            );
            scope_state.announced = true;
        }
        let percent = if total == 0 {
            0
        } else {
            ((downloaded as f64 / total as f64) * 1000.0).round() as usize
        };
        let percent_major = (percent.min(1000)) / 10;
        let percent_minor = (percent.min(1000)) % 10;
        let speed_suffix = bytes_per_sec
            .filter(|bytes_per_sec| *bytes_per_sec > 0.0)
            .map(|bytes_per_sec| {
                format!(
                    " at {}/s",
                    format_layer_package_download_bytes(bytes_per_sec as u64)
                )
            })
            .unwrap_or_default();
        eprint!(
            "\r\x1b[K   ⏬ {} {:>3}.{:01}% ({}/{}){}   📄 files {}/{} complete",
            layer_package_artifact_display_for_package(&self.package, file),
            percent_major,
            percent_minor,
            format_layer_package_download_bytes(downloaded),
            format_layer_package_download_bytes(total),
            speed_suffix,
            self.complete_count(completed_files),
            self.total_files,
        );
        let _ = std::io::stderr().flush();
        scope_state.drawn_line = true;
        if force {
            eprintln!();
            scope_state.drawn_line = false;
        }
    }
}

impl LayerPackageDownloadProgress {
    fn new(
        label: String,
        file: String,
        total_bytes: Option<u64>,
        package_scope: Option<Arc<LayerPackageDownloadScope>>,
        completed_before: usize,
    ) -> Self {
        let preflight_spinner = if interactive_tui_active()
            || package_scope
                .as_ref()
                .is_some_and(|scope| scope.has_drawn())
        {
            None
        } else {
            Some(start_spinner(&format!("Preparing download {file}")))
        };
        Self {
            label,
            file,
            package_scope,
            completed_before,
            preflight_spinner: Mutex::new(preflight_spinner),
            state: Mutex::new(LayerPackageDownloadProgressState {
                downloaded: 0,
                total: total_bytes.unwrap_or(0),
                bytes_per_sec: None,
                last_draw: None,
                showed_progress: false,
            }),
        }
    }

    fn emit(
        &self,
        downloaded_bytes: Option<u64>,
        total_bytes: Option<u64>,
        status: ModelProgressStatus,
    ) {
        let _ = emit_event(OutputEvent::ModelDownloadProgress {
            label: self.label.clone(),
            file: Some(self.file.clone()),
            downloaded_bytes,
            total_bytes,
            status,
        });
    }

    fn emit_ensuring(&self) {
        if !interactive_tui_active() {
            return;
        }
        let total = self
            .state
            .lock()
            .ok()
            .and_then(|state| (state.total > 0).then_some(state.total));
        self.emit(None, total, ModelProgressStatus::Ensuring);
    }

    fn emit_ready(&self, path: &Path) {
        let total = fs::metadata(path)
            .ok()
            .map(|metadata| metadata.len())
            .or_else(|| {
                self.state
                    .lock()
                    .ok()
                    .and_then(|state| (state.total > 0).then_some(state.total))
            });
        if interactive_tui_active() {
            self.emit(total, total, ModelProgressStatus::Ready);
            return;
        }
        if let Ok(mut spinner) = self.preflight_spinner.lock() {
            spinner.take();
        }
        let showed_progress = self
            .state
            .lock()
            .map(|state| state.showed_progress)
            .unwrap_or(false);
        if let Some(scope) = &self.package_scope {
            if !showed_progress {
                let total = total.unwrap_or(0);
                scope.draw(
                    &self.file,
                    self.completed_before + 1,
                    total,
                    total,
                    None,
                    true,
                );
            }
            return;
        }
        if !showed_progress {
            let file = layer_package_artifact_display(&self.label, &self.file);
            match total {
                Some(total) if total > 0 => eprintln!(
                    "   ✅ Ready {} ({})",
                    file,
                    format_layer_package_download_bytes(total)
                ),
                _ => eprintln!("   ✅ Ready {}", file),
            }
        }
    }

    fn draw(&self, state: &mut LayerPackageDownloadProgressState, force: bool) {
        if !force && state.downloaded == 0 && state.total == 0 {
            return;
        }
        let now = Instant::now();
        if !force
            && state
                .last_draw
                .is_some_and(|last| now.duration_since(last) < Duration::from_millis(150))
        {
            return;
        }
        state.last_draw = Some(now);
        state.showed_progress = true;
        if interactive_tui_active() {
            self.emit(
                (state.downloaded > 0).then_some(state.downloaded),
                (state.total > 0).then_some(state.total),
                ModelProgressStatus::Downloading,
            );
            return;
        }
        if let Ok(mut spinner) = self.preflight_spinner.lock() {
            spinner.take();
        }
        if let Some(scope) = &self.package_scope {
            let completed = if force {
                self.completed_before + 1
            } else {
                self.completed_before
            };
            scope.draw(
                &self.file,
                completed,
                state.downloaded,
                state.total,
                state.bytes_per_sec,
                force,
            );
        } else {
            draw_layer_package_file_progress(
                &layer_package_artifact_display(&self.label, &self.file),
                state.downloaded,
                state.total,
                state.bytes_per_sec,
                force,
            );
        }
    }
}

impl Drop for LayerPackageDownloadProgress {
    fn drop(&mut self) {
        if let Ok(mut spinner) = self.preflight_spinner.lock() {
            spinner.take();
        }
    }
}

impl ProgressHandler for LayerPackageDownloadProgress {
    fn on_progress(&self, event: &ProgressEvent) {
        let ProgressEvent::Download(event) = event else {
            return;
        };
        let Ok(mut state) = self.state.lock() else {
            return;
        };
        match event {
            DownloadEvent::Start { total_bytes, .. } => {
                if *total_bytes > 0 {
                    state.total = state.total.max(*total_bytes);
                }
            }
            DownloadEvent::Progress { files } => {
                if !files.is_empty() {
                    let downloaded: u64 = files.iter().map(|file| file.bytes_completed).sum();
                    state.downloaded = state.downloaded.max(downloaded);
                    let total: u64 = files.iter().map(|file| file.total_bytes).sum();
                    if total > 0 {
                        state.total = state.total.max(total);
                    }
                }
            }
            DownloadEvent::AggregateProgress {
                bytes_completed,
                total_bytes,
                bytes_per_sec,
            } => {
                state.downloaded = state.downloaded.max(*bytes_completed);
                if *total_bytes > 0 {
                    state.total = state.total.max(*total_bytes);
                }
                state.bytes_per_sec = *bytes_per_sec;
            }
            DownloadEvent::Complete => {
                if state.total > 0 {
                    state.downloaded = state.total;
                }
                state.bytes_per_sec = None;
            }
        }
        let should_show_progress = state.downloaded > 0 || state.total > 0;
        let force = matches!(event, DownloadEvent::Complete) && should_show_progress;
        if should_show_progress {
            self.draw(&mut state, force);
        } else if matches!(event, DownloadEvent::Complete) {
            if let Ok(mut spinner) = self.preflight_spinner.lock() {
                spinner.take();
            }
        }
    }
}

fn format_layer_package_download_bytes(bytes: u64) -> String {
    if bytes >= 1_000_000_000 {
        format!("{:.1}GB", bytes as f64 / 1e9)
    } else if bytes >= 1_000_000 {
        format!("{:.0}MB", bytes as f64 / 1e6)
    } else if bytes >= 1_000 {
        format!("{:.0}KB", bytes as f64 / 1e3)
    } else {
        format!("{bytes}B")
    }
}

fn layer_package_progress_package(label: &str) -> &str {
    label.strip_prefix("layer package ").unwrap_or(label)
}

fn layer_package_progress_repo(package: &str) -> &str {
    package
        .split_once('@')
        .map(|(repo, _)| repo)
        .unwrap_or(package)
}

fn layer_package_artifact_display(label: &str, file: &str) -> String {
    layer_package_artifact_display_for_package(layer_package_progress_package(label), file)
}

fn layer_package_artifact_display_for_package(package: &str, file: &str) -> String {
    let repo = layer_package_progress_repo(package);
    if file.starts_with(repo) || file.starts_with('/') {
        file.to_string()
    } else {
        format!("{repo}/{file}")
    }
}

fn draw_layer_package_file_progress(
    file: &str,
    downloaded: u64,
    total: u64,
    bytes_per_sec: Option<f64>,
    force: bool,
) {
    let percent = if total == 0 {
        0
    } else {
        ((downloaded as f64 / total as f64) * 1000.0).round() as usize
    };
    let percent_major = (percent.min(1000)) / 10;
    let percent_minor = (percent.min(1000)) % 10;
    let speed_suffix = bytes_per_sec
        .filter(|bytes_per_sec| *bytes_per_sec > 0.0)
        .map(|bytes_per_sec| {
            format!(
                " at {}/s",
                format_layer_package_download_bytes(bytes_per_sec as u64)
            )
        })
        .unwrap_or_default();
    eprint!(
        "\r\x1b[K   ⏬ {} {:>3}.{:01}% ({}/{}){}",
        file,
        percent_major,
        percent_minor,
        format_layer_package_download_bytes(downloaded),
        format_layer_package_download_bytes(total),
        speed_suffix,
    );
    let _ = std::io::stderr().flush();
    if force {
        eprintln!();
    }
}

pub(crate) fn is_layer_package_ref(value: &str) -> bool {
    StagePackageRef::parse(value).is_ok_and(|package_ref| package_ref.is_distributable_package())
}

/// Resolve an `hf://` package ref to a local directory, downloading the manifest,
/// shared components (metadata, embeddings, output head), and assigned layer files
/// using the `hf_hub` Rust library.
///
/// Returns the local directory path containing the package files.
/// If `package_ref` is already a local package path, validates its manifest paths
/// and returns it.
/// Resolve a layer package from the local HF cache without touching the HF SDK.
/// Verifies that needed files exist locally; returns the snapshot dir path.
fn resolve_local_package_files(
    package_dir: &Path,
    layer_start: u32,
    layer_end: u32,
    include_embeddings: bool,
    include_output: bool,
) -> Result<String> {
    let manifest_path = package_dir.join("model-package.json");
    let manifest_contents = fs::read(&manifest_path).context("read local package manifest")?;
    let manifest: serde_json::Value =
        serde_json::from_slice(&manifest_contents).context("parse local package manifest")?;

    // Verify shared/metadata.gguf exists
    let metadata_path = manifest
        .pointer("/shared/metadata/path")
        .and_then(|v| v.as_str())
        .context("manifest missing /shared/metadata/path")?;
    let metadata_path = safe_manifest_file_path(metadata_path)?;
    anyhow::ensure!(
        package_dir.join(&metadata_path).is_file(),
        "missing shared metadata: {}",
        metadata_path.display()
    );
    if include_embeddings {
        if let Some(path) = manifest
            .pointer("/shared/embeddings/path")
            .and_then(|v| v.as_str())
        {
            let path = safe_manifest_file_path(path)?;
            anyhow::ensure!(
                package_dir.join(&path).is_file(),
                "missing shared embeddings: {}",
                path.display()
            );
        }
    }
    if include_output {
        if let Some(path) = manifest
            .pointer("/shared/output/path")
            .and_then(|v| v.as_str())
        {
            let path = safe_manifest_file_path(path)?;
            anyhow::ensure!(
                package_dir.join(&path).is_file(),
                "missing shared output: {}",
                path.display()
            );
        }
    }
    // Verify needed layer files exist
    if let Some(layers) = manifest.get("layers").and_then(|l| l.as_array()) {
        for (i, layer) in layers.iter().enumerate() {
            let idx = layer
                .get("layer_index")
                .and_then(|v| v.as_u64())
                .unwrap_or(i as u64) as u32;
            if idx >= layer_start && idx < layer_end {
                if let Some(path) = layer.get("path").and_then(|a| a.as_str()) {
                    let path = safe_manifest_file_path(path)?;
                    anyhow::ensure!(
                        package_dir.join(&path).is_file(),
                        "missing layer file: {}",
                        path.display()
                    );
                }
            }
        }
    }
    Ok(package_dir.to_string_lossy().to_string())
}

fn package_integrity_cache_dir() -> PathBuf {
    crate::models::mesh_llm_cache_dir().join("skippy-package-integrity")
}

fn is_metadata_only_package_inspection(
    layer_start: u32,
    layer_end: u32,
    include_embeddings: bool,
    include_output: bool,
) -> bool {
    layer_start == layer_end && !include_embeddings && !include_output
}

fn verify_resolved_hf_package_files(
    package_dir: &Path,
    layer_start: u32,
    layer_end: u32,
    include_embeddings: bool,
    include_output: bool,
) -> Result<String> {
    let local_ref = resolve_local_package_files(
        package_dir,
        layer_start,
        layer_end,
        include_embeddings,
        include_output,
    )?;
    let metadata_only = is_metadata_only_package_inspection(
        layer_start,
        layer_end,
        include_embeddings,
        include_output,
    );
    let options = PackageIntegrityOptions::verify_with_cache(package_integrity_cache_dir());
    let report = if metadata_only {
        package::verify_layer_package_metadata_integrity(&local_ref, &options)
    } else {
        let request = PackageStageRequest {
            model_id: "hf-layer-package".to_string(),
            topology_id: "hf-layer-package-resolver".to_string(),
            package_ref: local_ref.clone(),
            stage_id: format!("layers-{layer_start}-{layer_end}"),
            layer_start,
            layer_end,
            include_embeddings,
            include_output,
        };
        package::verify_layer_package_integrity(&request, &options)
    }
    .map_err(|error| anyhow::anyhow!("verify resolved HF layer package artifacts: {error:#}"))?;
    tracing::debug!(
        artifacts = report.artifacts,
        verified_artifacts = report.verified_artifacts,
        cached_artifacts = report.cached_artifacts,
        manifest_sha256 = %report.manifest_sha256,
        metadata_only,
        "verified resolved HF layer package artifacts"
    );
    Ok(local_ref)
}

fn missing_cached_package_artifact(error: &anyhow::Error) -> bool {
    let message = error.to_string();
    message.starts_with("missing shared metadata:")
        || message.starts_with("missing shared embeddings:")
        || message.starts_with("missing shared output:")
        || message.starts_with("missing layer file:")
}

fn verify_cached_hf_package_files(
    package_dir: &Path,
    layer_start: u32,
    layer_end: u32,
    include_embeddings: bool,
    include_output: bool,
) -> Result<Option<String>> {
    match verify_resolved_hf_package_files(
        package_dir,
        layer_start,
        layer_end,
        include_embeddings,
        include_output,
    ) {
        Ok(local_ref) => Ok(Some(local_ref)),
        Err(error) if missing_cached_package_artifact(&error) => {
            tracing::debug!(
                package_dir = %package_dir.display(),
                error = %error,
                "cached HF layer package snapshot is incomplete; downloading missing artifacts"
            );
            Ok(None)
        }
        Err(error) => Err(error),
    }
}

fn manifest_artifact_bytes(artifact: &serde_json::Value) -> Option<u64> {
    artifact
        .get("artifact_bytes")
        .and_then(|value| value.as_u64())
}

fn layer_package_progress_label(repo: &str, revision: &str) -> String {
    if revision == "main" {
        format!("layer package {repo}")
    } else {
        format!("layer package {repo}@{revision}")
    }
}

fn download_layer_package_file(
    model_api: &hf_hub::HFRepositorySync,
    revision: &str,
    label: &str,
    file_name: &str,
    total_bytes: Option<u64>,
    package_scope: Option<Arc<LayerPackageDownloadScope>>,
    completed_before: usize,
) -> Result<PathBuf> {
    let progress = Arc::new(LayerPackageDownloadProgress::new(
        label.to_string(),
        file_name.to_string(),
        total_bytes,
        package_scope,
        completed_before,
    ));
    progress.emit_ensuring();
    let progress_handler: Progress = Some(progress.clone());
    let path = model_api
        .download_file(
            &hf_hub::RepoDownloadFileParams::builder()
                .filename(file_name.to_string())
                .revision(revision.to_string())
                .progress(progress_handler)
                .build(),
        )
        .with_context(|| format!("download layer package file: {file_name}"))?;
    progress.emit_ready(&path);
    Ok(path)
}

pub(crate) fn resolve_hf_package_to_local(
    package_ref: &str,
    layer_start: u32,
    layer_end: u32,
    include_embeddings: bool,
    include_output: bool,
) -> Result<String> {
    let parsed = StagePackageRef::parse(package_ref)?;
    let (repo, revision) = match &parsed {
        StagePackageRef::HuggingFacePackage { repo, revision } => (
            repo.clone(),
            revision.clone().unwrap_or_else(|| "main".to_string()),
        ),
        StagePackageRef::LocalPackage(path) => {
            return resolve_local_package_files(
                path,
                layer_start,
                layer_end,
                include_embeddings,
                include_output,
            );
        }
        _ => return Ok(package_ref.to_string()),
    };

    // Try to resolve from the local HF cache first — avoids the HF SDK entirely,
    // which is critical on NFS (where flock fails) and inside async runtimes
    // (where the sync SDK wrapper panics with "Cannot start a runtime").
    let cache_dir = crate::models::huggingface_hub_cache_dir();
    let repo_folder = format!("models--{}", repo.replace('/', "--"));
    let revision_cache_path = safe_manifest_file_path(&revision)
        .with_context(|| format!("invalid HF revision for local cache lookup: {revision}"))?;
    let ref_path = cache_dir
        .join(&repo_folder)
        .join("refs")
        .join(&revision_cache_path);
    let direct_snapshot_dir = cache_dir
        .join(&repo_folder)
        .join("snapshots")
        .join(&revision_cache_path);
    if direct_snapshot_dir.join("model-package.json").is_file() {
        if let Some(local_ref) = cache_resolution::resolve_cached_hf_package_snapshot(
            &direct_snapshot_dir,
            layer_start,
            layer_end,
            include_embeddings,
            include_output,
        )? {
            return Ok(local_ref);
        }
    }
    if let Ok(commit_hash) = fs::read_to_string(&ref_path) {
        let commit_hash = commit_hash.trim();
        let commit_hash_path = safe_manifest_file_path(commit_hash).with_context(|| {
            format!("invalid HF cache commit hash for local cache lookup: {commit_hash}")
        })?;
        let snapshot_dir = cache_dir
            .join(&repo_folder)
            .join("snapshots")
            .join(commit_hash_path);
        if snapshot_dir.join("model-package.json").is_file() {
            if let Some(local_ref) = cache_resolution::resolve_cached_hf_package_snapshot(
                &snapshot_dir,
                layer_start,
                layer_end,
                include_embeddings,
                include_output,
            )? {
                return Ok(local_ref);
            }
        }
    }
    let downloaded = crate::models::run_hf_sync(move || {
        download_hf_package_to_local_sync(
            &repo,
            &revision,
            layer_start,
            layer_end,
            include_embeddings,
            include_output,
        )
    })?;

    // Metadata-only probes (layer_start == layer_end == 0) download the
    // manifest and shared metadata but no layer files.  The downloaded
    // snapshot may be a skeleton whose hash must not propagate through
    // topology configs and stage loads.  Re-scan the local cache for a
    // snapshot that has at least one real layer artifact.
    //
    // Real stage loads (layer_start < layer_end) always download the
    // requested layer range, so the downloaded snapshot is guaranteed to
    // have the needed files — no fallback scan needed.
    let is_metadata_only = layer_start == 0 && layer_end == 0;
    if is_metadata_only {
        let downloaded_dir = std::path::Path::new(&downloaded);
        if downloaded_dir.join("model-package.json").is_file()
            && cache_resolution::resolve_cached_hf_package_snapshot(
                downloaded_dir,
                layer_start,
                layer_end,
                include_embeddings,
                include_output,
            )?
            .is_none()
        {
            // Downloaded snapshot is a skeleton — find one with real layers.
            let cache_dir = crate::models::huggingface_hub_cache_dir();
            for snapshot_dir in
                cache_resolution::cached_package_snapshots(&cache_dir, &repo_folder)?
            {
                if snapshot_dir.as_path() == downloaded_dir {
                    continue;
                }
                if let Ok(Some(better)) = cache_resolution::resolve_cached_hf_package_snapshot(
                    &snapshot_dir,
                    layer_start,
                    layer_end,
                    include_embeddings,
                    include_output,
                ) {
                    tracing::debug!(
                        downloaded = %downloaded,
                        better = %better,
                        "post-download: preferring cached snapshot with layer artifacts over skeleton"
                    );
                    return Ok(better);
                }
            }
        }
    }

    Ok(downloaded)
}

fn download_hf_package_to_local_sync(
    repo: &str,
    revision: &str,
    layer_start: u32,
    layer_end: u32,
    include_embeddings: bool,
    include_output: bool,
) -> Result<String> {
    let api = crate::models::build_hf_api(false)?;
    let (owner, name) = repo.split_once('/').context("invalid HF repo format")?;
    let model_api = api.model(owner, name);
    let progress_label = layer_package_progress_label(repo, revision);

    // Download manifest first
    let manifest_path = download_layer_package_file(
        &model_api,
        revision,
        &progress_label,
        "model-package.json",
        None,
        None,
        0,
    )
    .context("download layer package manifest")?;

    let package_dir = manifest_path
        .parent()
        .context("manifest has no parent directory")?
        .to_path_buf();

    // Read manifest to determine which files we need
    let manifest_contents = fs::read(&manifest_path).context("read package manifest")?;
    let manifest: serde_json::Value =
        serde_json::from_slice(&manifest_contents).context("parse package manifest")?;

    // Collect the files we need to download
    let mut needed_files: Vec<(PathBuf, Option<u64>)> = Vec::new();

    // Always need shared/metadata.gguf — required for materialization
    let metadata_artifact = manifest
        .pointer("/shared/metadata")
        .context("manifest missing required /shared/metadata")?;
    let metadata_path = metadata_artifact
        .get("path")
        .and_then(|v| v.as_str())
        .context("manifest missing required /shared/metadata/path")?;
    needed_files.push((
        safe_manifest_file_path(metadata_path)?,
        manifest_artifact_bytes(metadata_artifact),
    ));
    if include_embeddings {
        if let Some(artifact) = manifest.pointer("/shared/embeddings") {
            if let Some(path) = artifact.get("path").and_then(|v| v.as_str()) {
                needed_files.push((
                    safe_manifest_file_path(path)?,
                    manifest_artifact_bytes(artifact),
                ));
            }
        }
    }
    if include_output {
        if let Some(artifact) = manifest.pointer("/shared/output") {
            if let Some(path) = artifact.get("path").and_then(|v| v.as_str()) {
                needed_files.push((
                    safe_manifest_file_path(path)?,
                    manifest_artifact_bytes(artifact),
                ));
            }
        }
    }

    // Layer files for assigned range — use explicit layer_index if present,
    // fall back to array position.
    if let Some(layers) = manifest.get("layers").and_then(|l| l.as_array()) {
        for (i, layer) in layers.iter().enumerate() {
            let idx = layer
                .get("layer_index")
                .and_then(|v| v.as_u64())
                .unwrap_or(i as u64) as u32;
            if idx >= layer_start && idx < layer_end {
                if let Some(path) = layer.get("path").and_then(|a| a.as_str()) {
                    needed_files.push((
                        safe_manifest_file_path(path)?,
                        manifest_artifact_bytes(layer),
                    ));
                }
            }
        }
    }
    if layer_start == 0 {
        if let Some(projectors) = manifest.get("projectors").and_then(|p| p.as_array()) {
            for projector in projectors {
                if let Some(path) = projector.get("path").and_then(|value| value.as_str()) {
                    needed_files.push((
                        safe_manifest_file_path(path)?,
                        manifest_artifact_bytes(projector),
                    ));
                }
            }
        }
    }

    let missing_files: Vec<_> = needed_files
        .iter()
        .filter(|(file, _)| !package_dir.join(file).is_file())
        .collect();
    let package_scope = Arc::new(LayerPackageDownloadScope::new(
        &progress_label,
        missing_files.len() + 1,
    ));

    // Download each needed file
    for (index, (file, total_bytes)) in missing_files.into_iter().enumerate() {
        let file_name = file.to_string_lossy().to_string();
        download_layer_package_file(
            &model_api,
            revision,
            &progress_label,
            &file_name,
            *total_bytes,
            Some(Arc::clone(&package_scope)),
            index + 1,
        )
        .with_context(|| format!("download layer package file: {file_name}"))?;
    }

    verify_resolved_hf_package_files(
        &package_dir,
        layer_start,
        layer_end,
        include_embeddings,
        include_output,
    )
}

fn safe_manifest_file_path(path: &str) -> Result<PathBuf> {
    anyhow::ensure!(!path.is_empty(), "manifest file path is empty");
    let path = Path::new(path);
    let mut components = path.components();
    let Some(first) = components.next() else {
        bail!("manifest file path is empty");
    };
    anyhow::ensure!(
        matches!(first, Component::Normal(_))
            && components.all(|component| matches!(component, Component::Normal(_))),
        "manifest file path must be a safe relative path: {}",
        path.display()
    );
    Ok(path.to_path_buf())
}

pub(crate) fn ensure_package_manifest_sha(package_ref: &str, expected_sha256: &str) -> Result<()> {
    if expected_sha256.trim().is_empty() {
        return Ok(());
    }
    anyhow::ensure!(
        expected_sha256.len() == 64 && expected_sha256.chars().all(|ch| ch.is_ascii_hexdigit()),
        "package manifest sha256 must be a hex SHA-256 digest"
    );
    let manifest_path = Path::new(package_ref).join("model-package.json");
    let manifest_contents = fs::read(&manifest_path).context("read package manifest")?;
    let actual_sha = hex::encode(Sha256::digest(&manifest_contents));
    anyhow::ensure!(
        actual_sha.eq_ignore_ascii_case(expected_sha256),
        "package manifest sha256 mismatch"
    );
    Ok(())
}

pub(crate) fn inspect_stage_package(package_ref: &str) -> Result<StagePackageInfo> {
    // Resolve hf:// to local for inspection, downloading the manifest and any
    // shared package metadata that resolver path needs.
    let local_ref = resolve_hf_package_to_local(package_ref, 0, 0, false, false)?;
    let info = package::inspect_layer_package(&local_ref)
        .with_context(|| format!("inspect skippy layer package {package_ref}"))?;
    stage_package_info(package_ref, info)
}

/// Resolve an `hf://` package ref in a stage load request to a local directory.
/// Returns the resolved local path if the package ref needed resolution, or `None`
/// if it was already local / not a layer package.
pub(crate) fn resolve_stage_load_package(
    load: &StageLoadRequest,
) -> Result<Option<ResolvedStagePackage>> {
    if load.load_mode != LoadMode::LayerPackage {
        return Ok(None);
    }
    let is_first = load.layer_start == 0;
    let is_final = load.downstream.is_none();
    let include_embeddings = is_first || is_final;
    // Resolve hf:// to a local package directory, verifying the needed package
    // files exist without materializing them into a single GGUF on disk.
    let local_ref = resolve_hf_package_to_local(
        &load.package_ref,
        load.layer_start,
        load.layer_end,
        include_embeddings,
        is_final, // include_output
    )?;
    ensure_package_manifest_sha(&local_ref, &load.manifest_sha256)?;
    let info = package::inspect_layer_package(&local_ref)
        .with_context(|| format!("inspect resolved layer package {}", load.package_ref))?;
    Ok(Some(ResolvedStagePackage {
        local_ref,
        source_model_path: info.source_model_path,
        source_model_sha256: info.source_model_sha256,
        source_model_bytes: info.source_model_bytes,
    }))
}

pub(crate) fn materialize_stage_config(
    config: &StageConfig,
) -> Result<Option<(MaterializedStageArtifact, MaterializedStagePin)>> {
    if config.load_mode != LoadMode::LayerPackage {
        return Ok(None);
    }
    let package_ref = config
        .model_path
        .as_deref()
        .or(config.package_ref.as_deref())
        .context("layer-package config is missing package ref")?;
    let is_first = config.layer_start == 0;
    let is_final = config.downstream.is_none();
    let include_embeddings = is_first || is_final;
    let include_output = is_final;
    // Resolve hf:// to local dir with needed files downloaded
    let local_ref = resolve_hf_package_to_local(
        package_ref,
        config.layer_start,
        config.layer_end,
        include_embeddings,
        include_output,
    )?;
    if let Some(expected_manifest_sha) = config.manifest_sha256.as_deref() {
        ensure_package_manifest_sha(&local_ref, expected_manifest_sha)?;
    }
    let request = package_stage_request(
        &config.model_id,
        &config.topology_id,
        &local_ref,
        &config.stage_id,
        config.layer_start,
        config.layer_end,
        is_final,
    );
    let materialized = package::materialize_layer_package_details(&request).with_context(|| {
        format!(
            "materialize skippy stage package {} layers {}..{}",
            config.stage_id, config.layer_start, config.layer_end
        )
    })?;
    let info = package::inspect_layer_package(&local_ref)?;
    let artifact = MaterializedStageArtifact {
        path: materialized.output_path,
        manifest_sha256: materialized.manifest_sha256,
        source_model_path: info.source_model_path,
        source_model_sha256: info.source_model_sha256,
        source_model_bytes: info.source_model_bytes,
    };
    let pin = pin_materialized_stage(
        &artifact.path,
        &local_ref,
        &config.topology_id,
        &config.run_id,
        &config.stage_id,
    )?;
    Ok(Some((artifact, pin)))
}

pub(crate) fn prune_unpinned_materialized_stages() -> Result<usize> {
    let root = materialized_stage_cache_dir();
    if !root.is_dir() {
        return Ok(0);
    }
    let pins = active_pin_artifacts(&root)?;
    let mut removed = 0usize;
    for entry in fs::read_dir(&root).with_context(|| format!("read {}", root.display()))? {
        let path = entry?.path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("gguf") {
            continue;
        }
        if pins.iter().any(|pin| pin == &path) {
            continue;
        }
        fs::remove_file(&path).with_context(|| format!("remove {}", path.display()))?;
        removed += 1;
    }
    for entry in fs::read_dir(&root).with_context(|| format!("read {}", root.display()))? {
        let path = entry?.path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("json") {
            continue;
        }
        let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if !file_name.starts_with("source-") {
            continue;
        }
        let Ok(bytes) = fs::read(&path) else {
            continue;
        };
        let Ok(index) = serde_json::from_slice::<SourceIndex>(&bytes) else {
            continue;
        };
        if !index.artifact_path.exists() && !pins.iter().any(|pin| pin == &index.artifact_path) {
            let _ = fs::remove_file(path);
        }
    }
    Ok(removed)
}

pub(crate) fn remove_materialized_stages_for_sources(sources: &[PathBuf]) -> Result<usize> {
    let candidates = materialized_stage_removal_candidates(sources)?;
    let mut removed = 0usize;
    for candidate in candidates {
        if candidate.artifact_path.exists() {
            fs::remove_file(&candidate.artifact_path)
                .with_context(|| format!("remove {}", candidate.artifact_path.display()))?;
            removed += 1;
        }
        let _ = fs::remove_file(candidate.source_index_path);
    }
    Ok(removed)
}

pub(crate) fn materialized_stages_for_sources(sources: &[PathBuf]) -> Result<Vec<PathBuf>> {
    Ok(materialized_stage_removal_candidates(sources)?
        .into_iter()
        .filter(|candidate| candidate.artifact_path.exists())
        .map(|candidate| candidate.artifact_path)
        .collect())
}

fn materialized_stage_removal_candidates(
    sources: &[PathBuf],
) -> Result<Vec<MaterializedStageRemovalCandidate>> {
    if sources.is_empty() {
        return Ok(Vec::new());
    }
    let root = materialized_stage_cache_dir();
    if !root.is_dir() {
        return Ok(Vec::new());
    }
    let source_strings = sources
        .iter()
        .map(|path| path.to_string_lossy().to_string())
        .collect::<Vec<_>>();
    let pins = active_pin_artifacts(&root)?;
    let mut candidates = Vec::new();
    for entry in fs::read_dir(&root).with_context(|| format!("read {}", root.display()))? {
        let path = entry?.path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("json") {
            continue;
        }
        let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if !file_name.starts_with("source-") {
            continue;
        }
        let Ok(bytes) = fs::read(&path) else {
            continue;
        };
        let Ok(index) = serde_json::from_slice::<SourceIndex>(&bytes) else {
            continue;
        };
        if !source_strings
            .iter()
            .any(|source| source == &index.source_model_path)
        {
            continue;
        }
        if pins.iter().any(|pin| pin == &index.artifact_path) {
            continue;
        }
        candidates.push(MaterializedStageRemovalCandidate {
            artifact_path: index.artifact_path,
            source_index_path: path,
        });
    }
    candidates.sort_by(|left, right| left.artifact_path.cmp(&right.artifact_path));
    Ok(candidates)
}

#[derive(Debug)]
struct MaterializedStageRemovalCandidate {
    artifact_path: PathBuf,
    source_index_path: PathBuf,
}

fn stage_package_info(package_ref: &str, info: LayerPackageInfo) -> Result<StagePackageInfo> {
    let activation_width = info.activation_width.with_context(|| {
        format!(
            "layer package {package_ref} is missing activation_width; rebuild the package manifest"
        )
    })?;
    Ok(StagePackageInfo {
        package_ref: package_ref.to_string(),
        package_dir: info.package_dir,
        manifest_sha256: info.manifest_sha256,
        model_id: info.model_id,
        source_model_path: info.source_model_path,
        source_model_sha256: info.source_model_sha256,
        source_model_bytes: info.source_model_bytes,
        layer_count: info.layer_count,
        activation_width,
        projector_path: info
            .projectors
            .first()
            .map(|projector| projector.path.to_string_lossy().to_string()),
        layers: info
            .layers
            .into_iter()
            .map(|layer| StagePackageLayerInfo {
                layer_index: layer.layer_index,
                tensor_count: layer.tensor_count,
                tensor_bytes: layer.tensor_bytes,
                artifact_bytes: layer.artifact_bytes,
            })
            .collect(),
    })
}

fn package_stage_request(
    model_id: &str,
    topology_id: &str,
    package_ref: &str,
    stage_id: &str,
    layer_start: u32,
    layer_end: u32,
    is_final_stage: bool,
) -> PackageStageRequest {
    PackageStageRequest {
        model_id: model_id.to_string(),
        topology_id: topology_id.to_string(),
        package_ref: package_ref.to_string(),
        stage_id: stage_id.to_string(),
        layer_start,
        layer_end,
        include_embeddings: layer_start == 0 || is_final_stage,
        include_output: is_final_stage,
    }
}

fn pin_materialized_stage(
    artifact_path: &Path,
    package_ref: &str,
    topology_id: &str,
    run_id: &str,
    stage_id: &str,
) -> Result<MaterializedStagePin> {
    let root = materialized_stage_cache_dir();
    let pin_dir = root.join("pins");
    fs::create_dir_all(&pin_dir).with_context(|| format!("create {}", pin_dir.display()))?;
    let pin = PinFile {
        artifact_path: artifact_path.to_path_buf(),
        package_ref: package_ref.to_string(),
        topology_id: topology_id.to_string(),
        run_id: run_id.to_string(),
        stage_id: stage_id.to_string(),
    };
    let pin_path = pin_dir.join(format!(
        "{}.json",
        cache_key(&format!(
            "{package_ref}\0{topology_id}\0{run_id}\0{stage_id}"
        ))
    ));
    fs::write(&pin_path, serde_json::to_vec_pretty(&pin)?)
        .with_context(|| format!("write {}", pin_path.display()))?;
    write_source_index(artifact_path, &pin)?;
    Ok(MaterializedStagePin { path: pin_path })
}

#[derive(Debug, Serialize, Deserialize)]
struct SourceIndex {
    artifact_path: PathBuf,
    source_model_path: String,
}

fn write_source_index(artifact_path: &Path, pin: &PinFile) -> Result<()> {
    let root = materialized_stage_cache_dir();
    let Ok(info) = package::inspect_layer_package(&pin.package_ref) else {
        return Ok(());
    };
    let index = SourceIndex {
        artifact_path: artifact_path.to_path_buf(),
        source_model_path: info.source_model_path,
    };
    let path = root.join(format!(
        "source-{}.json",
        cache_key(&format!(
            "{}\0{}",
            index.source_model_path,
            artifact_path.to_string_lossy()
        ))
    ));
    fs::write(path, serde_json::to_vec_pretty(&index)?).context("write source index")?;
    Ok(())
}

fn active_pin_artifacts(root: &Path) -> Result<Vec<PathBuf>> {
    let pin_dir = root.join("pins");
    if !pin_dir.is_dir() {
        return Ok(Vec::new());
    }
    let mut artifacts = Vec::new();
    for entry in fs::read_dir(&pin_dir).with_context(|| format!("read {}", pin_dir.display()))? {
        let path = entry?.path();
        let Ok(bytes) = fs::read(&path) else {
            continue;
        };
        let Ok(pin) = serde_json::from_slice::<PinFile>(&bytes) else {
            continue;
        };
        artifacts.push(pin.artifact_path);
    }
    Ok(artifacts)
}

fn cache_key(input: &str) -> String {
    let digest = Sha256::digest(input.as_bytes());
    let mut out = String::with_capacity(24);
    for byte in &digest[..12] {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::OsString;

    use serial_test::serial;
    use skippy_protocol::{FlashAttentionType, LoadMode};

    fn restore_env(key: &str, previous: Option<OsString>) {
        if let Some(value) = previous {
            std::env::set_var(key, value);
        } else {
            std::env::remove_var(key);
        }
    }

    fn sha256_hex(bytes: &[u8]) -> String {
        hex::encode(Sha256::digest(bytes))
    }

    fn write_local_package_fixture(root: &Path) -> (PathBuf, String) {
        fs::create_dir_all(root.join("shared")).unwrap();
        fs::create_dir_all(root.join("layers")).unwrap();
        fs::write(root.join("shared/metadata.gguf"), b"metadata").unwrap();
        fs::write(root.join("shared/embeddings.gguf"), b"embeddings").unwrap();
        fs::write(root.join("shared/output.gguf"), b"output").unwrap();
        fs::write(root.join("layers/layer-000.gguf"), b"layer").unwrap();
        let manifest = serde_json::json!({
            "schema_version": 1,
            "model_id": "model-a",
            "source_model": {
                "path": "model-a.gguf",
                "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "files": [
                    {
                        "path": "model-a.gguf",
                        "size_bytes": 123,
                        "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                    }
                ]
            },
            "format": "layer-package",
            "layer_count": 1,
            "activation_width": 4096,
            "shared": {
                "metadata": {
                    "path": "shared/metadata.gguf",
                    "tensor_count": 1,
                    "tensor_bytes": 1,
                    "artifact_bytes": 8,
                    "sha256": sha256_hex(b"metadata")
                },
                "embeddings": {
                    "path": "shared/embeddings.gguf",
                    "tensor_count": 1,
                    "tensor_bytes": 1,
                    "artifact_bytes": 10,
                    "sha256": sha256_hex(b"embeddings")
                },
                "output": {
                    "path": "shared/output.gguf",
                    "tensor_count": 1,
                    "tensor_bytes": 1,
                    "artifact_bytes": 6,
                    "sha256": sha256_hex(b"output")
                }
            },
            "layers": [
                {
                    "layer_index": 0,
                    "path": "layers/layer-000.gguf",
                    "tensor_count": 1,
                    "tensor_bytes": 1,
                    "artifact_bytes": 5,
                    "sha256": sha256_hex(b"layer")
                }
            ],
            "skippy_abi_version": "0.1.0"
        });
        let manifest_bytes = serde_json::to_vec_pretty(&manifest).unwrap();
        let manifest_sha = sha256_hex(&manifest_bytes);
        fs::write(root.join("model-package.json"), manifest_bytes).unwrap();
        (root.to_path_buf(), manifest_sha)
    }

    fn stage_load_request_for_package(
        package_dir: &Path,
        manifest_sha256: String,
    ) -> StageLoadRequest {
        StageLoadRequest {
            topology_id: "topology-a".to_string(),
            run_id: "run-a".to_string(),
            model_id: "model-a".to_string(),
            backend: "skippy".to_string(),
            package_ref: package_dir.to_string_lossy().to_string(),
            manifest_sha256,
            stage_id: "stage-0".to_string(),
            stage_index: 0,
            layer_start: 0,
            layer_end: 1,
            model_path: Some(package_dir.to_string_lossy().to_string()),
            source_model_bytes: None,
            projector_path: None,
            selected_device: None,
            bind_addr: "127.0.0.1:0".to_string(),
            activation_width: 4096,
            wire_dtype: crate::inference::skippy::StageWireDType::F16,
            ctx_size: 8192,
            lane_count: 1,
            n_batch: None,
            n_ubatch: None,
            n_gpu_layers: -1,
            cache_type_k: "f16".to_string(),
            cache_type_v: "f16".to_string(),
            flash_attn_type: FlashAttentionType::Auto,
            shutdown_generation: 1,
            coordinator_term: 0,
            coordinator_id: None,
            lease_until_unix_ms: 0,
            load_mode: LoadMode::LayerPackage,
            upstream: None,
            downstream: None,
        }
    }

    struct EnvRestore {
        key: &'static str,
        previous: Option<OsString>,
    }

    impl Drop for EnvRestore {
        fn drop(&mut self) {
            restore_env(self.key, self.previous.take());
        }
    }

    fn write_cached_package_snapshot(snapshot: &Path, layer_sha: String) {
        fs::create_dir_all(snapshot.join("shared")).unwrap();
        fs::create_dir_all(snapshot.join("layers")).unwrap();
        fs::write(snapshot.join("shared/metadata.gguf"), b"metadata").unwrap();
        fs::write(snapshot.join("layers/layer-000.gguf"), b"layer").unwrap();
        fs::write(
            snapshot.join("model-package.json"),
            serde_json::to_vec_pretty(&serde_json::json!({
                "schema_version": 1,
                "model_id": "model-a",
                "source_model": {
                    "path": "model-a.gguf",
                    "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                    "files": [
                        {
                            "path": "model-a.gguf",
                            "size_bytes": 123,
                            "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                        }
                    ]
                },
                "format": "layer-package",
                "layer_count": 1,
                "activation_width": 4096,
                "shared": {
                    "metadata": {
                        "path": "shared/metadata.gguf",
                        "tensor_count": 1,
                        "tensor_bytes": 1,
                        "artifact_bytes": 8,
                        "sha256": sha256_hex(b"metadata")
                    },
                    "embeddings": {
                        "path": "shared/metadata.gguf",
                        "tensor_count": 1,
                        "tensor_bytes": 1,
                        "artifact_bytes": 8,
                        "sha256": sha256_hex(b"metadata")
                    },
                    "output": {
                        "path": "shared/metadata.gguf",
                        "tensor_count": 1,
                        "tensor_bytes": 1,
                        "artifact_bytes": 8,
                        "sha256": sha256_hex(b"metadata")
                    }
                },
                "layers": [
                    {
                        "layer_index": 0,
                        "path": "layers/layer-000.gguf",
                        "tensor_count": 1,
                        "tensor_bytes": 1,
                        "artifact_bytes": 5,
                        "sha256": layer_sha
                    }
                ],
                "skippy_abi_version": "0.1.0",
            }))
            .unwrap(),
        )
        .unwrap();
    }

    #[test]
    fn layer_package_ref_detects_local_manifest_dir() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("model-package.json"), "{}").unwrap();

        assert!(is_layer_package_ref(&dir.path().to_string_lossy()));
        assert!(!is_layer_package_ref("/tmp/not-a-package"));
        assert!(is_layer_package_ref("hf://Mesh-LLM/demo-package"));
    }

    #[test]
    fn package_ref_distinguishes_direct_gguf_from_distributable_packages() {
        let direct = StagePackageRef::parse("/models/model.gguf").unwrap();
        assert_eq!(
            direct,
            StagePackageRef::SyntheticDirectGguf(PathBuf::from("/models/model.gguf"))
        );
        assert!(!direct.is_distributable_package());
        assert!(direct.as_package_ref().is_none());

        let hf = StagePackageRef::parse("hf://Mesh-LLM/demo-package@abc123").unwrap();
        assert!(hf.is_distributable_package());
        assert_eq!(
            hf.as_package_ref().as_deref(),
            Some("hf://Mesh-LLM/demo-package@abc123")
        );
    }

    #[test]
    fn layer_package_artifact_display_names_repo_and_file_without_revision() {
        assert_eq!(
            layer_package_artifact_display(
                "layer package meshllm/demo-package@abc123",
                "layers/layer-005.gguf"
            ),
            "meshllm/demo-package/layers/layer-005.gguf"
        );
        assert_eq!(
            layer_package_artifact_display(
                "layer package meshllm/demo-package",
                "model-package.json"
            ),
            "meshllm/demo-package/model-package.json"
        );
    }

    #[test]
    fn safe_manifest_file_path_rejects_escaping_paths() {
        assert_eq!(
            safe_manifest_file_path("shared/metadata.gguf").unwrap(),
            PathBuf::from("shared/metadata.gguf")
        );

        for path in [
            "",
            "/tmp/metadata.gguf",
            "../metadata.gguf",
            "shared/../metadata.gguf",
        ] {
            let error = safe_manifest_file_path(path).unwrap_err().to_string();
            assert!(
                error.contains("manifest file path"),
                "unexpected error for {path:?}: {error}"
            );
        }
    }

    #[test]
    fn local_package_resolution_rejects_manifest_traversal() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(
            dir.path().join("model-package.json"),
            serde_json::json!({
                "shared": {
                    "metadata": { "path": "../metadata.gguf" }
                },
                "layers": []
            })
            .to_string(),
        )
        .unwrap();

        let error = resolve_local_package_files(dir.path(), 0, 0, false, false)
            .unwrap_err()
            .to_string();
        assert!(error.contains("safe relative path"), "{error}");
    }

    #[test]
    fn local_package_ref_resolution_rejects_manifest_traversal() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(
            dir.path().join("model-package.json"),
            serde_json::json!({
                "shared": {
                    "metadata": { "path": "../metadata.gguf" }
                },
                "layers": []
            })
            .to_string(),
        )
        .unwrap();

        let error = resolve_hf_package_to_local(&dir.path().to_string_lossy(), 0, 0, false, false)
            .unwrap_err()
            .to_string();
        assert!(error.contains("safe relative path"), "{error}");
    }

    #[test]
    fn cached_hf_package_verification_treats_missing_artifacts_as_incomplete_cache() {
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir_all(dir.path().join("shared")).unwrap();
        fs::write(dir.path().join("shared/metadata.gguf"), b"metadata").unwrap();
        fs::write(
            dir.path().join("model-package.json"),
            serde_json::json!({
                "shared": {
                    "metadata": { "path": "shared/metadata.gguf" },
                    "embeddings": { "path": "shared/embeddings.gguf" },
                    "output": { "path": "shared/output.gguf" }
                },
                "layers": []
            })
            .to_string(),
        )
        .unwrap();

        let resolved = verify_cached_hf_package_files(dir.path(), 0, 0, true, false).unwrap();

        assert_eq!(resolved, None);
    }

    #[test]
    fn resolve_stage_load_package_requires_expected_manifest_sha() {
        let dir = tempfile::tempdir().unwrap();
        let (package_dir, manifest_sha) = write_local_package_fixture(dir.path());

        let load = stage_load_request_for_package(&package_dir, manifest_sha.clone());
        let resolved = resolve_stage_load_package(&load).unwrap();
        assert_eq!(
            resolved.as_ref().map(|package| package.local_ref.as_str()),
            Some(package_dir.to_str().unwrap())
        );

        let mut mismatched = stage_load_request_for_package(&package_dir, "0".repeat(64));
        mismatched.package_ref = package_dir.to_string_lossy().to_string();
        let error = resolve_stage_load_package(&mismatched)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("package manifest sha256 mismatch"),
            "{error}"
        );
    }

    #[test]
    #[serial]
    fn hf_package_resolution_rejects_revision_cache_traversal() {
        let prev_hf_home = std::env::var_os("HF_HOME");
        let prev_hf_cache = std::env::var_os("HF_HUB_CACHE");
        let prev_huggingface_cache = std::env::var_os("HUGGINGFACE_HUB_CACHE");

        let temp = tempfile::tempdir().unwrap();
        std::env::set_var("HF_HOME", temp.path());
        std::env::remove_var("HF_HUB_CACHE");
        std::env::remove_var("HUGGINGFACE_HUB_CACHE");

        let error = resolve_hf_package_to_local("hf://owner/repo@../../evil", 0, 0, false, false)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("invalid HF revision") || error.contains("safe relative path"),
            "{error}"
        );

        restore_env("HF_HOME", prev_hf_home);
        restore_env("HF_HUB_CACHE", prev_hf_cache);
        restore_env("HUGGINGFACE_HUB_CACHE", prev_huggingface_cache);
    }

    #[test]
    #[serial]
    fn hf_package_resolution_rejects_ref_target_cache_traversal() {
        let prev_hf_home = std::env::var_os("HF_HOME");
        let prev_hf_cache = std::env::var_os("HF_HUB_CACHE");
        let prev_huggingface_cache = std::env::var_os("HUGGINGFACE_HUB_CACHE");

        let temp = tempfile::tempdir().unwrap();
        std::env::set_var("HF_HOME", temp.path());
        std::env::remove_var("HF_HUB_CACHE");
        std::env::remove_var("HUGGINGFACE_HUB_CACHE");

        let refs_dir = temp
            .path()
            .join("hub")
            .join("models--owner--repo")
            .join("refs");
        fs::create_dir_all(&refs_dir).unwrap();
        fs::write(refs_dir.join("main"), "../../evil").unwrap();

        let error = resolve_hf_package_to_local("hf://owner/repo", 0, 0, false, false)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("invalid HF cache commit hash") || error.contains("safe relative path"),
            "{error}"
        );

        restore_env("HF_HOME", prev_hf_home);
        restore_env("HF_HUB_CACHE", prev_hf_cache);
        restore_env("HUGGINGFACE_HUB_CACHE", prev_huggingface_cache);
    }

    #[test]
    #[serial]
    fn hf_package_resolution_uses_direct_snapshot_revision_cache() {
        let prev_hf_home = std::env::var_os("HF_HOME");
        let prev_hf_cache = std::env::var_os("HF_HUB_CACHE");
        let prev_huggingface_cache = std::env::var_os("HUGGINGFACE_HUB_CACHE");

        let temp = tempfile::tempdir().unwrap();
        std::env::set_var("HF_HOME", temp.path());
        std::env::remove_var("HF_HUB_CACHE");
        std::env::remove_var("HUGGINGFACE_HUB_CACHE");

        let snapshot = temp
            .path()
            .join("hub")
            .join("models--owner--repo")
            .join("snapshots")
            .join("abc123");
        write_cached_package_snapshot(&snapshot, sha256_hex(b"layer"));

        let resolved =
            resolve_hf_package_to_local("hf://owner/repo@abc123", 0, 1, false, false).unwrap();

        assert_eq!(PathBuf::from(resolved), snapshot);

        restore_env("HF_HOME", prev_hf_home);
        restore_env("HF_HUB_CACHE", prev_hf_cache);
        restore_env("HUGGINGFACE_HUB_CACHE", prev_huggingface_cache);
    }

    #[test]
    #[serial]
    /// With an explicit pinned revision that has all requested layers, the
    /// cache lookup returns it directly without downloading or scanning other
    /// snapshots.  A stale snapshot with different content must NOT be picked.
    fn pinned_revision_resolves_directly_from_cache() {
        let prev_hf_home = std::env::var_os("HF_HOME");
        let prev_hf_cache = std::env::var_os("HF_HUB_CACHE");
        let prev_huggingface_cache = std::env::var_os("HUGGINGFACE_HUB_CACHE");
        let prev_xdg_cache = std::env::var_os("XDG_CACHE_HOME");

        let temp = tempfile::tempdir().unwrap();
        std::env::set_var("HF_HOME", temp.path().join("hf"));
        std::env::set_var("XDG_CACHE_HOME", temp.path().join("mesh-cache"));
        std::env::remove_var("HF_HUB_CACHE");
        std::env::remove_var("HUGGINGFACE_HUB_CACHE");

        let repo_cache = temp
            .path()
            .join("hf")
            .join("hub")
            .join("models--owner--repo");

        // Create a complete snapshot at the pinned revision.
        let pinned_snapshot = repo_cache.join("snapshots").join("abc123");
        write_cached_package_snapshot(&pinned_snapshot, sha256_hex(b"layer"));

        // Create a stale snapshot that also has layers — must NOT be used.
        let stale_snapshot = repo_cache.join("snapshots").join("old-stale");
        write_cached_package_snapshot(&stale_snapshot, sha256_hex(b"layer"));

        let resolved =
            resolve_hf_package_to_local("hf://owner/repo@abc123", 0, 0, false, false).unwrap();

        assert_eq!(PathBuf::from(resolved), pinned_snapshot);

        restore_env("HF_HOME", prev_hf_home);
        restore_env("HF_HUB_CACHE", prev_hf_cache);
        restore_env("HUGGINGFACE_HUB_CACHE", prev_huggingface_cache);
        restore_env("XDG_CACHE_HOME", prev_xdg_cache);
    }

    #[test]
    #[serial]
    fn hf_package_metadata_only_cache_resolution_uses_metadata_integrity_scope() {
        let prev_hf_home = std::env::var_os("HF_HOME");
        let prev_hf_cache = std::env::var_os("HF_HUB_CACHE");
        let prev_huggingface_cache = std::env::var_os("HUGGINGFACE_HUB_CACHE");
        let prev_xdg_cache = std::env::var_os("XDG_CACHE_HOME");

        let temp = tempfile::tempdir().unwrap();
        std::env::set_var("HF_HOME", temp.path().join("hf"));
        std::env::set_var("XDG_CACHE_HOME", temp.path().join("mesh-cache"));
        std::env::remove_var("HF_HUB_CACHE");
        std::env::remove_var("HUGGINGFACE_HUB_CACHE");

        let snapshot = temp
            .path()
            .join("hf")
            .join("hub")
            .join("models--owner--repo")
            .join("snapshots")
            .join("abc123");
        write_cached_package_snapshot(
            &snapshot,
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".to_string(),
        );

        let resolved =
            resolve_hf_package_to_local("hf://owner/repo@abc123", 0, 0, false, false).unwrap();
        assert_eq!(PathBuf::from(resolved), snapshot);

        let info = inspect_stage_package("hf://owner/repo@abc123").unwrap();
        assert_eq!(info.model_id, "model-a");
        assert_eq!(info.layer_count, 1);

        fs::write(snapshot.join("shared/metadata.gguf"), b"metadota").unwrap();
        let error = resolve_hf_package_to_local("hf://owner/repo@abc123", 0, 0, false, false)
            .unwrap_err()
            .to_string();
        assert!(error.contains("checksum mismatch"), "{error}");
        assert!(error.contains("shared/metadata.gguf"), "{error}");

        restore_env("HF_HOME", prev_hf_home);
        restore_env("HF_HUB_CACHE", prev_hf_cache);
        restore_env("HUGGINGFACE_HUB_CACHE", prev_huggingface_cache);
        restore_env("XDG_CACHE_HOME", prev_xdg_cache);
    }

    #[test]
    #[serial]
    fn hf_package_resolution_verifies_cached_snapshot_artifact_checksums() {
        let prev_hf_home = std::env::var_os("HF_HOME");
        let prev_hf_cache = std::env::var_os("HF_HUB_CACHE");
        let prev_huggingface_cache = std::env::var_os("HUGGINGFACE_HUB_CACHE");
        let prev_xdg_cache = std::env::var_os("XDG_CACHE_HOME");

        let temp = tempfile::tempdir().unwrap();
        std::env::set_var("HF_HOME", temp.path().join("hf"));
        std::env::set_var("XDG_CACHE_HOME", temp.path().join("mesh-cache"));
        std::env::remove_var("HF_HUB_CACHE");
        std::env::remove_var("HUGGINGFACE_HUB_CACHE");

        let snapshot = temp
            .path()
            .join("hf")
            .join("hub")
            .join("models--owner--repo")
            .join("snapshots")
            .join("abc123");
        write_cached_package_snapshot(
            &snapshot,
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".to_string(),
        );

        let error = resolve_hf_package_to_local("hf://owner/repo@abc123", 0, 1, false, false)
            .unwrap_err()
            .to_string();

        assert!(error.contains("checksum mismatch"), "{error}");

        restore_env("HF_HOME", prev_hf_home);
        restore_env("HF_HUB_CACHE", prev_hf_cache);
        restore_env("HUGGINGFACE_HUB_CACHE", prev_huggingface_cache);
        restore_env("XDG_CACHE_HOME", prev_xdg_cache);
    }

    #[test]
    fn resolved_stage_load_package_keeps_local_path_out_of_source_identity() {
        let dir = tempfile::tempdir().unwrap();
        write_cached_package_snapshot(dir.path(), sha256_hex(b"layer"));
        let manifest_bytes = fs::read(dir.path().join("model-package.json")).unwrap();
        let manifest_sha256 = sha256_hex(&manifest_bytes);
        let load = StageLoadRequest {
            topology_id: "topology-a".to_string(),
            run_id: "run-a".to_string(),
            model_id: "model-a".to_string(),
            backend: "skippy".to_string(),
            package_ref: dir.path().to_string_lossy().to_string(),
            manifest_sha256,
            stage_id: "stage-0".to_string(),
            stage_index: 0,
            layer_start: 0,
            layer_end: 1,
            model_path: None,
            source_model_bytes: None,
            projector_path: None,
            selected_device: None,
            bind_addr: "127.0.0.1:0".to_string(),
            activation_width: 4096,
            wire_dtype: crate::inference::skippy::StageWireDType::F16,
            ctx_size: 512,
            lane_count: 1,
            n_batch: None,
            n_ubatch: None,
            n_gpu_layers: 0,
            cache_type_k: "f16".to_string(),
            cache_type_v: "f16".to_string(),
            flash_attn_type: skippy_protocol::FlashAttentionType::Auto,
            shutdown_generation: 0,
            coordinator_term: 0,
            coordinator_id: None,
            lease_until_unix_ms: 0,
            load_mode: LoadMode::LayerPackage,
            upstream: None,
            downstream: None,
        };

        let resolved = resolve_stage_load_package(&load)
            .unwrap()
            .expect("layer package should resolve");

        assert_eq!(resolved.local_ref, dir.path().to_string_lossy());
        assert_eq!(resolved.source_model_path, "model-a.gguf");
        assert_eq!(
            resolved.source_model_sha256,
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        );
    }

    #[test]
    #[serial]
    fn materialized_stage_preview_matches_source_removal_candidates() {
        let prev_xdg = std::env::var_os("XDG_CACHE_HOME");
        let _xdg_restore = EnvRestore {
            key: "XDG_CACHE_HOME",
            previous: prev_xdg,
        };

        let temp = tempfile::tempdir().unwrap();
        std::env::set_var("XDG_CACHE_HOME", temp.path());

        let root = materialized_stage_cache_dir();
        fs::create_dir_all(&root).unwrap();
        let source = temp
            .path()
            .join("source-package")
            .join("model-package.json");
        fs::create_dir_all(source.parent().unwrap()).unwrap();
        fs::write(&source, b"{}").unwrap();
        let fixture_id = cache_key(&temp.path().to_string_lossy());
        let artifact = root.join(format!("stage-{fixture_id}.gguf"));
        fs::write(&artifact, b"stage").unwrap();
        let index = SourceIndex {
            artifact_path: artifact.clone(),
            source_model_path: source.to_string_lossy().to_string(),
        };
        let index_path = root.join(format!("source-{fixture_id}.json"));
        fs::write(&index_path, serde_json::to_vec_pretty(&index).unwrap()).unwrap();
        let unreadable_index_path = root.join(format!("source-unreadable-{fixture_id}.json"));
        fs::create_dir(&unreadable_index_path).unwrap();

        let preview = materialized_stages_for_sources(std::slice::from_ref(&source)).unwrap();
        assert_eq!(preview, vec![artifact.clone()]);

        let removed =
            remove_materialized_stages_for_sources(std::slice::from_ref(&source)).unwrap();
        assert_eq!(removed, 1);
        assert!(!artifact.exists());
        assert!(!index_path.exists());
        fs::remove_dir(unreadable_index_path).unwrap();
    }

    /// Integration test: resolves package metadata without downloading layer files from HF.
    /// Run with: cargo test -p mesh-llm resolve_hf_downloads_metadata_only -- --ignored
    #[test]
    #[ignore]
    fn resolve_hf_downloads_metadata_only() {
        let package_ref = "hf://meshllm/Qwen3-235B-A22B-UD-Q4_K_XL-layers";
        // Request 0 layers — should download manifest/shared metadata, but no layer files.
        let local_path = resolve_hf_package_to_local(package_ref, 0, 0, false, false).unwrap();
        let manifest = std::path::Path::new(&local_path).join("model-package.json");
        assert!(
            manifest.is_file(),
            "manifest should exist at {}",
            manifest.display()
        );

        // Verify manifest is valid JSON with expected fields
        let contents = std::fs::read_to_string(&manifest).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&contents).unwrap();
        assert_eq!(parsed["schema_version"], 1);
        assert!(parsed["layers"].as_array().unwrap().len() > 50);

        // Verify the function didn't request any layer downloads
        // (we can't check the cache dir because previous test runs may have cached files)
    }

    /// Integration test: downloads manifest + a single layer file.
    /// Run with: cargo test -p mesh-llm resolve_hf_downloads_single_layer -- --ignored
    #[test]
    #[ignore]
    fn resolve_hf_downloads_single_layer() {
        let package_ref = "hf://meshllm/Qwen3-235B-A22B-UD-Q4_K_XL-layers";
        // Request just layer 0
        let local_path = resolve_hf_package_to_local(package_ref, 0, 1, false, false).unwrap();
        let manifest = std::path::Path::new(&local_path).join("model-package.json");
        assert!(manifest.is_file());

        // Read manifest to find layer 0's artifact path
        let contents = std::fs::read_to_string(&manifest).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&contents).unwrap();
        let layer0_artifact = parsed["layers"][0]["path"].as_str().unwrap();

        // Verify that specific layer file was downloaded
        let layer0_path = std::path::Path::new(&local_path).join(layer0_artifact);
        assert!(
            layer0_path.is_file(),
            "layer 0 should be downloaded at {}",
            layer0_path.display()
        );
        // Should be non-trivial size (layer files are typically > 1 MB)
        let size = std::fs::metadata(&layer0_path).unwrap().len();
        assert!(size > 1_000_000, "layer file should be > 1MB, got {size}");
    }
}
