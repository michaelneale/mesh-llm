use super::*;
use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use super::inventory::inventory_source_candidates;
use anyhow::{anyhow, Result};
use skippy_protocol::{FlashAttentionType, LoadMode, StageDevice};
use tokio::sync::{oneshot, Mutex as TokioMutex};

fn load_request() -> StageLoadRequest {
    StageLoadRequest {
        topology_id: "topology-a".to_string(),
        run_id: "run-a".to_string(),
        model_id: "model-a".to_string(),
        backend: "skippy".to_string(),
        package_ref: "pkg-a".to_string(),
        manifest_sha256: "sha256".to_string(),
        stage_id: "stage-0".to_string(),
        stage_index: 0,
        layer_start: 0,
        layer_end: 12,
        model_path: Some("/models/model.gguf".to_string()),
        source_model_bytes: Some(64 * 1024 * 1024 * 1024),
        projector_path: Some("/models/mmproj.gguf".to_string()),
        selected_device: Some(StageDevice {
            backend_device: "CUDA0".to_string(),
            stable_id: Some("GPU-123".to_string()),
            index: Some(0),
            vram_bytes: Some(24_000_000_000),
        }),
        bind_addr: "127.0.0.1:0".to_string(),
        activation_width: 4096,
        wire_dtype: StageWireDType::F16,
        ctx_size: 8192,
        lane_count: 3,
        n_batch: Some(2048),
        n_ubatch: Some(512),
        n_gpu_layers: -1,
        cache_type_k: "f16".to_string(),
        cache_type_v: "q8_0".to_string(),
        flash_attn_type: FlashAttentionType::Enabled,
        shutdown_generation: 7,
        load_mode: LoadMode::RuntimeSlice,
        upstream: None,
        downstream: Some(StagePeerDescriptor {
            stage_id: "stage-1".to_string(),
            stage_index: 1,
            endpoint: "127.0.0.1:9001".to_string(),
            node_id: None,
        }),
    }
}

struct BlockingPackagePrefetcher {
    started: TokioMutex<Option<oneshot::Sender<()>>>,
    release: TokioMutex<Option<oneshot::Receiver<Result<()>>>>,
}

impl BlockingPackagePrefetcher {
    fn new() -> (Self, oneshot::Receiver<()>, oneshot::Sender<Result<()>>) {
        let (started_tx, started_rx) = oneshot::channel();
        let (release_tx, release_rx) = oneshot::channel();
        (
            Self {
                started: TokioMutex::new(Some(started_tx)),
                release: TokioMutex::new(Some(release_rx)),
            },
            started_rx,
            release_tx,
        )
    }
}

#[async_trait::async_trait]
impl StagePackagePrefetcher for BlockingPackagePrefetcher {
    async fn prefetch_stage_package(&self, _request: &StagePrepareRequest) -> Result<()> {
        if let Some(started) = self.started.lock().await.take() {
            let _ = started.send(());
        }
        let Some(release) = self.release.lock().await.take() else {
            return Ok(());
        };
        release
            .await
            .unwrap_or_else(|_| Err(anyhow!("prefetch cancelled")))
    }
}

#[test]
fn stage_config_preserves_backend_neutral_load_fields() {
    let request = load_request();
    let config = stage_config(&request, None, None).unwrap();

    assert_eq!(config.topology_id, "topology-a");
    assert_eq!(config.run_id, "run-a");
    assert_eq!(config.model_id, "model-a");
    assert_eq!(config.package_ref.as_deref(), Some("pkg-a"));
    assert_eq!(config.manifest_sha256.as_deref(), Some("sha256"));
    assert_eq!(
        config.source_model_path.as_deref(),
        Some("/models/model.gguf")
    );
    assert!(config.materialized_path.is_none());
    assert!(!config.materialized_pinned);
    assert_eq!(config.stage_id, "stage-0");
    assert_eq!(config.stage_index, 0);
    assert_eq!(config.layer_start, 0);
    assert_eq!(config.layer_end, 12);
    assert_eq!(config.lane_count, 3);
    assert_eq!(config.n_batch, Some(2048));
    assert_eq!(config.n_ubatch, Some(512));
    assert_eq!(config.model_path.as_deref(), Some("/models/model.gguf"));
    assert_eq!(
        config.projector_path.as_deref(),
        Some("/models/mmproj.gguf")
    );
    assert_eq!(config.flash_attn_type, FlashAttentionType::Enabled);
    assert_eq!(
        config
            .selected_device
            .as_ref()
            .map(|d| d.backend_device.as_str()),
        Some("CUDA0")
    );
    assert_eq!(
        config.downstream.as_ref().map(|d| d.stage_id.as_str()),
        Some("stage-1")
    );
    assert!(config.filter_tensors_on_load);
}

#[test]
fn stage_config_prefers_package_source_identity_over_local_ref() {
    let mut request = load_request();
    request.load_mode = LoadMode::LayerPackage;
    request.model_path = Some("/tmp/hf-cache/snapshots/abc123".to_string());
    request.source_model_bytes = Some(123);
    let package = super::super::materialization::ResolvedStagePackage {
        local_ref: "/tmp/hf-cache/snapshots/abc123".to_string(),
        source_model_path: "model-a.gguf".to_string(),
        source_model_sha256: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
            .to_string(),
        source_model_bytes: Some(456),
    };

    let config = stage_config(&request, None, Some(&package)).unwrap();

    assert_eq!(
        config.model_path.as_deref(),
        Some("/tmp/hf-cache/snapshots/abc123")
    );
    assert_eq!(config.source_model_path.as_deref(), Some("model-a.gguf"));
    assert_eq!(
        config.source_model_sha256.as_deref(),
        Some("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
    );
    assert_eq!(config.source_model_bytes, Some(456));
}

#[test]
fn stage_config_rejects_empty_selected_backend_device() {
    let mut request = load_request();
    request.selected_device = Some(StageDevice {
        backend_device: String::new(),
        stable_id: Some("uuid:GPU-123".into()),
        index: Some(0),
        vram_bytes: Some(24_000_000_000),
    });

    let err = stage_config(&request, None, None).unwrap_err().to_string();

    assert!(err.contains("selected backend device"));
}

#[test]
fn stage_status_filter_matches_optional_identity_fields() {
    let load = load_request();
    assert!(StageStatusFilter {
        topology_id: Some("topology-a".to_string()),
        run_id: None,
        stage_id: Some("stage-0".to_string()),
    }
    .matches(&load));
    assert!(!StageStatusFilter {
        topology_id: Some("other".to_string()),
        run_id: None,
        stage_id: None,
    }
    .matches(&load));
}

#[test]
fn materialize_stage_bind_addr_replaces_ephemeral_port() {
    let bind_addr = materialize_stage_bind_addr("127.0.0.1:0".parse().unwrap()).unwrap();
    assert_eq!(bind_addr.ip().to_string(), "127.0.0.1");
    assert_ne!(bind_addr.port(), 0);
}

#[tokio::test]
async fn binary_stage_ready_probe_waits_for_wire_handshake() {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let bind_addr = listener.local_addr().unwrap();
    let server = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(75));
        let (mut stream, _) = listener.accept().unwrap();
        skippy_protocol::binary::send_ready(&mut stream).unwrap();
    });

    let started = Instant::now();
    wait_for_binary_stage_ready(bind_addr, Duration::from_secs(2))
        .await
        .unwrap();
    assert!(started.elapsed() >= Duration::from_millis(50));
    server.join().unwrap();
}

#[tokio::test]
async fn prepare_stage_records_background_source_availability() {
    let file = tempfile::NamedTempFile::new().unwrap();
    let path = file.path().to_string_lossy().to_string();
    let mut load = load_request();
    load.model_path = Some(path.clone());
    load.package_ref = "gguf:///definitely/missing/model.gguf".to_string();
    load.downstream = None;
    let mut state = StageControlState::default();

    let accepted = state
        .prepare(StagePrepareRequest {
            load: load.clone(),
            coordinator_id: None,
        })
        .await
        .unwrap();

    assert!(accepted.accepted);
    assert_eq!(accepted.status.state, StagePreparationState::Assigned);

    let mut last_state = StagePreparationState::Assigned;
    for _ in 0..20 {
        let inventory = state
            .inventory(StageInventoryRequest {
                model_id: load.model_id.clone(),
                package_ref: load.package_ref.clone(),
                manifest_sha256: load.manifest_sha256.clone(),
            })
            .await;
        if let Some(status) = inventory
            .preparing_ranges
            .iter()
            .find(|status| status.stage_id == load.stage_id)
        {
            last_state = status.state;
            if status.state == StagePreparationState::Available {
                assert_eq!(status.bytes_done, Some(0));
                assert_eq!(status.bytes_total, Some(0));
                return;
            }
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }

    panic!("prepare did not become available, last state: {last_state:?}");
}

#[tokio::test]
async fn prepare_layer_package_stays_downloading_while_peer_prefetch_is_pending() {
    let mut load = load_request();
    load.load_mode = LoadMode::LayerPackage;
    load.package_ref = "missing-layer-package".to_string();
    load.manifest_sha256 = "a".repeat(64);
    load.downstream = None;

    let (prefetcher, started_rx, release_tx) = BlockingPackagePrefetcher::new();
    let mut state = StageControlState {
        package_prefetcher: Some(Arc::new(prefetcher)),
        ..Default::default()
    };

    let accepted = state
        .prepare(StagePrepareRequest {
            load: load.clone(),
            coordinator_id: None,
        })
        .await
        .unwrap();

    assert_eq!(accepted.status.state, StagePreparationState::Assigned);
    started_rx.await.expect("prefetch must start");
    tokio::time::sleep(Duration::from_millis(50)).await;

    let inventory = state
        .inventory(StageInventoryRequest {
            model_id: load.model_id.clone(),
            package_ref: load.package_ref.clone(),
            manifest_sha256: load.manifest_sha256.clone(),
        })
        .await;
    let status = inventory
        .preparing_ranges
        .iter()
        .find(|status| status.stage_id == load.stage_id)
        .expect("prepare status must stay visible");
    assert_eq!(status.state, StagePreparationState::Downloading);

    let _ = release_tx.send(Err(anyhow!("peer stalled")));
}

#[tokio::test]
async fn prepare_layer_package_fails_only_after_peer_prefetch_and_local_resolution_fail() {
    let mut load = load_request();
    load.load_mode = LoadMode::LayerPackage;
    load.package_ref = "missing-layer-package".to_string();
    load.manifest_sha256 = "a".repeat(64);
    load.downstream = None;

    let (prefetcher, started_rx, release_tx) = BlockingPackagePrefetcher::new();
    let mut state = StageControlState {
        package_prefetcher: Some(Arc::new(prefetcher)),
        ..Default::default()
    };

    state
        .prepare(StagePrepareRequest {
            load: load.clone(),
            coordinator_id: None,
        })
        .await
        .unwrap();
    started_rx.await.expect("prefetch must start");
    release_tx.send(Err(anyhow!("peer unavailable"))).unwrap();

    for _ in 0..40 {
        let inventory = state
            .inventory(StageInventoryRequest {
                model_id: load.model_id.clone(),
                package_ref: load.package_ref.clone(),
                manifest_sha256: load.manifest_sha256.clone(),
            })
            .await;
        if let Some(status) = inventory
            .preparing_ranges
            .iter()
            .find(|status| status.stage_id == load.stage_id)
        {
            if status.state == StagePreparationState::Failed {
                let error = status.error.as_deref().unwrap_or_default();
                assert!(error.contains("not a skippy package ref"));
                assert!(error.contains("peer artifact prefetch failed"));
                return;
            }
            assert_ne!(status.state, StagePreparationState::Failed);
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }

    panic!("prepare did not fail after peer prefetch and local resolution failed");
}

#[tokio::test]
async fn inventory_retains_failed_prepare_status() {
    let load = load_request();
    let key = stage_key(&load.topology_id, &load.run_id, &load.stage_id);
    let state = StageControlState::default();
    let mut failed = preparation_status_from_load(&load, StagePreparationState::Failed);
    failed.error = Some("source unavailable".to_string());
    state.preparations.lock().await.insert(key, failed);

    let inventory = state
        .inventory(StageInventoryRequest {
            model_id: load.model_id.clone(),
            package_ref: load.package_ref.clone(),
            manifest_sha256: load.manifest_sha256.clone(),
        })
        .await;

    let status = inventory
        .preparing_ranges
        .iter()
        .find(|status| status.stage_id == load.stage_id)
        .expect("failed prepare status should remain visible");
    assert_eq!(status.state, StagePreparationState::Failed);
    assert_eq!(status.error.as_deref(), Some("source unavailable"));
}

#[test]
fn inventory_source_candidates_prefer_explicit_gguf_ref() {
    let request = StageInventoryRequest {
        model_id: "catalog-model".to_string(),
        package_ref: "gguf:///tmp/source-model.gguf".to_string(),
        manifest_sha256: "sha256".to_string(),
    };

    let candidates = inventory_source_candidates(&request);

    assert_eq!(
        candidates[0],
        std::path::PathBuf::from("/tmp/source-model.gguf")
    );
}

#[test]
fn stage_load_timeout_keeps_existing_floor_without_size_hint() {
    let mut request = load_request();
    request.source_model_bytes = None;
    request.load_mode = LoadMode::RuntimeSlice;

    assert_eq!(stage_load_timeout(&request), Duration::from_secs(900));
}

#[test]
fn stage_load_timeout_scales_with_size_hints_for_all_load_modes() {
    let mut request = load_request();
    request.source_model_bytes = Some(170 * 1024 * 1024 * 1024);
    request.load_mode = LoadMode::RuntimeSlice;

    assert_eq!(stage_load_timeout(&request), Duration::from_secs(1360));

    request.load_mode = LoadMode::LayerPackage;
    assert_eq!(stage_load_timeout(&request), Duration::from_secs(1360));

    request.source_model_bytes = Some(u64::MAX);
    assert_eq!(stage_load_timeout(&request), Duration::from_secs(14400));
}
