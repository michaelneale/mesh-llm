use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::Arc,
};

use anyhow::Result;
use skippy_protocol::LoadMode;
use tokio::sync::Mutex;

use crate::inference::skippy::materialization::{inspect_stage_package, is_layer_package_ref};

use super::{
    preparation_status_from_load, SourceModelKind, StageInventoryRequest, StageLoadRequest,
    StagePackagePrefetcher, StagePreparationState, StagePreparationStatus, StagePrepareRequest,
};

#[derive(Clone, Debug)]
pub(super) struct InventorySource {
    pub(super) path: PathBuf,
    pub(super) bytes: Option<u64>,
    pub(super) layer_count: u32,
    pub(super) kind: SourceModelKind,
}

pub(super) fn resolve_inventory_source(request: &StageInventoryRequest) -> Option<InventorySource> {
    if is_layer_package_ref(&request.package_ref) {
        let info = inspect_stage_package(&request.package_ref).ok()?;
        return Some(InventorySource {
            path: info.package_dir,
            bytes: info.source_model_bytes,
            layer_count: info.layer_count,
            kind: SourceModelKind::LayerPackage,
        });
    }

    for candidate in inventory_source_candidates(request) {
        if !candidate.exists() {
            continue;
        }
        let layer_count = crate::inference::skippy::infer_layer_count(&candidate).ok()?;
        let kind = if is_split_gguf_path(&candidate) {
            SourceModelKind::SplitGguf
        } else {
            SourceModelKind::PlainGguf
        };
        let bytes = crate::inference::election::total_model_bytes(&candidate);
        return Some(InventorySource {
            path: candidate,
            bytes: Some(bytes),
            layer_count,
            kind,
        });
    }
    None
}

pub(super) fn inventory_source_candidates(request: &StageInventoryRequest) -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    if let Some(path) = request.package_ref.strip_prefix("gguf://") {
        if !path.is_empty() {
            candidates.push(PathBuf::from(path));
        }
    }
    if !request.model_id.is_empty() {
        candidates.push(crate::models::find_model_path(&request.model_id));
    }
    candidates
}

fn is_split_gguf_path(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .and_then(model_ref::split_gguf_shard_info)
        .is_some()
}

pub(super) async fn run_stage_prepare_task(
    preparations: Arc<Mutex<HashMap<String, StagePreparationStatus>>>,
    key: String,
    request: StagePrepareRequest,
    package_prefetcher: Option<Arc<dyn StagePackagePrefetcher>>,
) {
    let load = request.load.clone();
    update_preparation(
        &preparations,
        &key,
        preparation_status_from_load(&load, StagePreparationState::Resolving),
    )
    .await;
    let peer_prefetch_error =
        prefetch_stage_package_if_needed(&preparations, &key, &request, package_prefetcher).await;
    if peer_prefetch_error.is_none()
        && load.load_mode != LoadMode::LayerPackage
        && !is_layer_package_ref(&load.package_ref)
    {
        update_preparation(
            &preparations,
            &key,
            preparation_status_from_load(&load, StagePreparationState::Downloading),
        )
        .await;
    }
    let result = prepare_stage_source(&load).await;
    let state = match result {
        Ok(PrepareSourceResult { bytes_total }) => {
            let mut status = preparation_status_from_load(&load, StagePreparationState::Available);
            status.bytes_done = bytes_total;
            status.bytes_total = bytes_total;
            status
        }
        Err(error) => {
            let mut status = preparation_status_from_load(&load, StagePreparationState::Failed);
            status.error = Some(match peer_prefetch_error {
                Some(prefetch_error) => {
                    format!("{error}; peer artifact prefetch failed: {prefetch_error}")
                }
                None => error.to_string(),
            });
            status
        }
    };
    update_preparation(&preparations, &key, state).await;
}

async fn prefetch_stage_package_if_needed(
    preparations: &Arc<Mutex<HashMap<String, StagePreparationStatus>>>,
    key: &str,
    request: &StagePrepareRequest,
    package_prefetcher: Option<Arc<dyn StagePackagePrefetcher>>,
) -> Option<String> {
    let load = &request.load;
    if load.load_mode != LoadMode::LayerPackage && !is_layer_package_ref(&load.package_ref) {
        return None;
    }
    let prefetcher = package_prefetcher?;
    update_preparation(
        preparations,
        key,
        preparation_status_from_load(load, StagePreparationState::Downloading),
    )
    .await;
    match prefetcher.prefetch_stage_package(request).await {
        Ok(()) => None,
        Err(error) => {
            tracing::debug!(
                topology_id = %load.topology_id,
                run_id = %load.run_id,
                stage_id = %load.stage_id,
                "peer artifact prefetch failed, falling back to local/HF resolver: {error}"
            );
            Some(error.to_string())
        }
    }
}

struct PrepareSourceResult {
    bytes_total: Option<u64>,
}

async fn prepare_stage_source(load: &StageLoadRequest) -> Result<PrepareSourceResult> {
    if load.load_mode == LoadMode::LayerPackage || is_layer_package_ref(&load.package_ref) {
        let info = inspect_stage_package(&load.package_ref)?;
        return Ok(PrepareSourceResult {
            bytes_total: info.source_model_bytes,
        });
    }

    for candidate in [
        load.model_path.as_deref(),
        Some(load.model_id.as_str()),
        load.package_ref.strip_prefix("gguf://"),
    ]
    .into_iter()
    .flatten()
    .filter(|candidate| !candidate.is_empty())
    {
        match crate::models::resolve_model_spec_with_progress(Path::new(candidate), true).await {
            Ok(path) => {
                let bytes_total = crate::inference::election::total_model_bytes(&path);
                return Ok(PrepareSourceResult {
                    bytes_total: Some(bytes_total),
                });
            }
            Err(last_error) => {
                tracing::debug!(
                    stage_id = %load.stage_id,
                    candidate,
                    error = %last_error,
                    "stage source prepare candidate failed"
                );
            }
        }
    }
    anyhow::bail!("stage source model is not available")
}

async fn update_preparation(
    preparations: &Arc<Mutex<HashMap<String, StagePreparationStatus>>>,
    key: &str,
    status: StagePreparationStatus,
) {
    preparations.lock().await.insert(key.to_string(), status);
}
