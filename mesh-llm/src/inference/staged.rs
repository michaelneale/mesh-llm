#![allow(dead_code)]
//! Staged pipeline inference: split models across mesh nodes by layer range.
//!
//! Each node runs a `skippy-server` stage with a contiguous range of layers.
//! Activations flow between stages via TCP (tunneled over QUIC when nodes
//! aren't directly reachable).
//!
//! This replaces:
//! - Dense distributed (rpc-server offload) — layers split natively
//! - MoE expert sharding — MoE layers contain all experts, split by layer
//! - Solo llama-server — single stage with all layers

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Parsed layer package manifest (model-package.json).
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PackageManifest {
    pub schema_version: u32,
    pub model_id: String,
    pub layer_count: u32,
    pub shared: SharedArtifacts,
    pub layers: Vec<LayerArtifact>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SharedArtifacts {
    pub metadata: ArtifactInfo,
    pub embeddings: ArtifactInfo,
    pub output: ArtifactInfo,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ArtifactInfo {
    pub path: String,
    pub tensor_count: u32,
    pub tensor_bytes: u64,
    pub artifact_bytes: u64,
    pub sha256: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LayerArtifact {
    pub layer_index: u32,
    pub path: String,
    pub tensor_count: u32,
    pub tensor_bytes: u64,
    pub artifact_bytes: u64,
    pub sha256: String,
}

/// A stage assignment: which node runs which layers.
#[derive(Debug, Clone)]
pub struct StageAssignment {
    /// Index in the pipeline (0 = first stage, receives tokens)
    pub stage_index: u32,
    /// Peer endpoint ID (None = local node)
    pub peer_id: Option<iroh::EndpointId>,
    /// Layer range [start, end)
    pub layer_start: u32,
    pub layer_end: u32,
    /// Bytes of layer data this stage needs to download/hold
    pub weight_bytes: u64,
}

/// Complete topology plan for a staged model.
#[derive(Debug, Clone)]
pub struct StagePlan {
    pub model_id: String,
    pub layer_count: u32,
    pub stages: Vec<StageAssignment>,
    /// Which stage index runs the driver (OpenAI frontend + tokenizer)
    pub driver_stage: u32,
}

/// Node capability for topology planning.
#[derive(Debug, Clone)]
pub struct NodeCapability {
    pub peer_id: Option<iroh::EndpointId>, // None = local
    pub available_bytes: u64,              // RAM/VRAM available for model weights
    pub is_local: bool,
}

// ─── Manifest ────────────────────────────────────────────────────────────────

impl PackageManifest {
    /// Load manifest from a package directory.
    pub fn from_dir(dir: &Path) -> Result<Self> {
        let manifest_path = dir.join("model-package.json");
        let content = std::fs::read_to_string(&manifest_path)
            .with_context(|| format!("read manifest {}", manifest_path.display()))?;
        serde_json::from_str(&content)
            .with_context(|| format!("parse manifest {}", manifest_path.display()))
    }

    /// Total model weight bytes (all layers + shared).
    pub fn total_bytes(&self) -> u64 {
        let layer_bytes: u64 = self.layers.iter().map(|l| l.artifact_bytes).sum();
        layer_bytes
            + self.shared.metadata.artifact_bytes
            + self.shared.embeddings.artifact_bytes
            + self.shared.output.artifact_bytes
    }

    /// Average bytes per layer.
    pub fn bytes_per_layer(&self) -> u64 {
        if self.layers.is_empty() {
            return 0;
        }
        let layer_bytes: u64 = self.layers.iter().map(|l| l.artifact_bytes).sum();
        layer_bytes / self.layers.len() as u64
    }
}

// ─── Topology Planning ───────────────────────────────────────────────────────

/// Plan how to split a model across available nodes.
///
/// Strategy: assign layers proportional to available memory. The node with
/// the most memory gets the most layers. First stage (layer 0) always goes
/// to the node that will run the driver (typically the node with most RAM,
/// since it also needs the tokenizer).
pub fn plan_topology(
    manifest: &PackageManifest,
    nodes: &[NodeCapability],
) -> Result<StagePlan> {
    if nodes.is_empty() {
        anyhow::bail!("no nodes available for topology planning");
    }

    // Single node: all layers on one stage
    if nodes.len() == 1 {
        return Ok(StagePlan {
            model_id: manifest.model_id.clone(),
            layer_count: manifest.layer_count,
            stages: vec![StageAssignment {
                stage_index: 0,
                peer_id: nodes[0].peer_id,
                layer_start: 0,
                layer_end: manifest.layer_count,
                weight_bytes: manifest.total_bytes(),
            }],
            driver_stage: 0,
        });
    }

    // Multi-node: assign layers proportional to available memory
    let total_available: u64 = nodes.iter().map(|n| n.available_bytes).sum();
    let bytes_per_layer = manifest.bytes_per_layer();
    let layer_count = manifest.layer_count;

    let mut stages = Vec::new();
    let mut layer_cursor: u32 = 0;

    for (i, node) in nodes.iter().enumerate() {
        let is_last = i == nodes.len() - 1;

        let layers_for_node = if is_last {
            // Last node gets remaining layers
            layer_count - layer_cursor
        } else {
            // Proportional assignment
            let fraction = node.available_bytes as f64 / total_available as f64;
            let assigned = (fraction * layer_count as f64).round() as u32;
            // Ensure at least 1 layer per node
            assigned.max(1).min(layer_count - layer_cursor - (nodes.len() as u32 - i as u32 - 1))
        };

        let layer_start = layer_cursor;
        let layer_end = layer_cursor + layers_for_node;
        let weight_bytes = manifest.layers[layer_start as usize..layer_end as usize]
            .iter()
            .map(|l| l.artifact_bytes)
            .sum::<u64>()
            + if layer_start == 0 {
                manifest.shared.embeddings.artifact_bytes + manifest.shared.metadata.artifact_bytes
            } else {
                manifest.shared.metadata.artifact_bytes
            }
            + if layer_end == layer_count {
                manifest.shared.output.artifact_bytes
            } else {
                0
            };

        stages.push(StageAssignment {
            stage_index: i as u32,
            peer_id: node.peer_id,
            layer_start,
            layer_end,
            weight_bytes,
        });

        layer_cursor = layer_end;
    }

    // Driver runs on stage-0 (has embeddings for tokenizer)
    Ok(StagePlan {
        model_id: manifest.model_id.clone(),
        layer_count: manifest.layer_count,
        stages,
        driver_stage: 0,
    })
}

// ─── Stage Server Config ─────────────────────────────────────────────────────

/// Generate the JSON config for a stage server.
pub fn stage_config_json(
    plan: &StagePlan,
    stage: &StageAssignment,
    package_dir: &Path,
    run_id: &str,
    bind_port: u16,
    downstream_addr: Option<&str>, // tcp://host:port for next stage
    upstream_addr: Option<&str>,   // tcp://host:port for previous stage
) -> serde_json::Value {
    let mut config = serde_json::json!({
        "run_id": run_id,
        "model_id": plan.model_id,
        "model_path": package_dir.to_str().unwrap_or(""),
        "load_mode": "layer-package",
        "stage_id": format!("stage-{}", stage.stage_index),
        "stage_index": stage.stage_index,
        "topology_id": run_id,
        "layer_start": stage.layer_start,
        "layer_end": stage.layer_end,
        "n_gpu_layers": -1,
        "ctx_size": 4096,
        "filter_tensors_on_load": true,
        "bind_addr": format!("0.0.0.0:{bind_port}"),
    });

    if let Some(downstream) = downstream_addr {
        config["downstream"] = serde_json::json!({
            "endpoint": downstream,
            "stage_id": format!("stage-{}", stage.stage_index + 1),
            "stage_index": stage.stage_index + 1,
        });
    }

    if let Some(upstream) = upstream_addr {
        config["upstream"] = serde_json::json!({
            "endpoint": upstream,
            "stage_id": format!("stage-{}", stage.stage_index.saturating_sub(1)),
            "stage_index": stage.stage_index.saturating_sub(1),
        });
    }

    config
}

// ─── Layer Download ──────────────────────────────────────────────────────────

/// Determine which files a stage needs to download from HF.
pub fn files_for_stage(manifest: &PackageManifest, stage: &StageAssignment) -> Vec<String> {
    let mut files = vec![
        "model-package.json".to_string(),
        manifest.shared.metadata.path.clone(),
    ];

    // First stage needs embeddings
    if stage.layer_start == 0 {
        files.push(manifest.shared.embeddings.path.clone());
    }

    // Last stage needs output (and embeddings for tied-weight models)
    if stage.layer_end == manifest.layer_count {
        files.push(manifest.shared.output.path.clone());
        if stage.layer_start > 0 {
            // Tied embeddings — final stage also needs embeddings
            files.push(manifest.shared.embeddings.path.clone());
        }
    }

    // Layer files for this stage's range
    for layer in &manifest.layers[stage.layer_start as usize..stage.layer_end as usize] {
        files.push(layer.path.clone());
    }

    files
}

/// Build the HF download include patterns for a stage.
pub fn hf_include_patterns(manifest: &PackageManifest, stage: &StageAssignment) -> Vec<String> {
    let files = files_for_stage(manifest, stage);
    files
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn test_manifest(layer_count: u32) -> PackageManifest {
        PackageManifest {
            schema_version: 1,
            model_id: "test/model".to_string(),
            layer_count,
            shared: SharedArtifacts {
                metadata: ArtifactInfo {
                    path: "shared/metadata.gguf".into(),
                    tensor_count: 0,
                    tensor_bytes: 0,
                    artifact_bytes: 1_000_000,
                    sha256: "abc".into(),
                },
                embeddings: ArtifactInfo {
                    path: "shared/embeddings.gguf".into(),
                    tensor_count: 1,
                    tensor_bytes: 350_000_000,
                    artifact_bytes: 356_000_000,
                    sha256: "def".into(),
                },
                output: ArtifactInfo {
                    path: "shared/output.gguf".into(),
                    tensor_count: 2,
                    tensor_bytes: 510_000_000,
                    artifact_bytes: 516_000_000,
                    sha256: "ghi".into(),
                },
            },
            layers: (0..layer_count)
                .map(|i| LayerArtifact {
                    layer_index: i,
                    path: format!("layers/layer-{:03}.gguf", i),
                    tensor_count: 12,
                    tensor_bytes: 1_400_000_000,
                    artifact_bytes: 1_407_000_000,
                    sha256: format!("layer{i}"),
                })
                .collect(),
        }
    }

    #[test]
    fn single_node_gets_all_layers() {
        let manifest = test_manifest(94);
        let nodes = vec![NodeCapability {
            peer_id: None,
            available_bytes: 256_000_000_000,
            is_local: true,
        }];
        let plan = plan_topology(&manifest, &nodes).unwrap();
        assert_eq!(plan.stages.len(), 1);
        assert_eq!(plan.stages[0].layer_start, 0);
        assert_eq!(plan.stages[0].layer_end, 94);
    }

    #[test]
    fn two_nodes_split_proportional() {
        let manifest = test_manifest(94);
        let nodes = vec![
            NodeCapability {
                peer_id: None,
                available_bytes: 256_000_000_000, // 256 GB
                is_local: true,
            },
            NodeCapability {
                peer_id: Some(iroh::EndpointId::from_bytes(&[1; 32]).unwrap()),
                available_bytes: 64_000_000_000, // 64 GB
                is_local: false,
            },
        ];
        let plan = plan_topology(&manifest, &nodes).unwrap();
        assert_eq!(plan.stages.len(), 2);
        // 256/(256+64) = 0.8 → ~75 layers for node A
        assert!(plan.stages[0].layer_end > 60);
        assert!(plan.stages[0].layer_end < 85);
        assert_eq!(plan.stages[1].layer_end, 94);
        // Stage 1 starts where stage 0 ends
        assert_eq!(plan.stages[1].layer_start, plan.stages[0].layer_end);
    }

    #[test]
    fn files_for_first_stage() {
        let manifest = test_manifest(4);
        let stage = StageAssignment {
            stage_index: 0,
            peer_id: None,
            layer_start: 0,
            layer_end: 2,
            weight_bytes: 0,
        };
        let files = files_for_stage(&manifest, &stage);
        assert!(files.contains(&"shared/metadata.gguf".to_string()));
        assert!(files.contains(&"shared/embeddings.gguf".to_string()));
        assert!(files.contains(&"layers/layer-000.gguf".to_string()));
        assert!(files.contains(&"layers/layer-001.gguf".to_string()));
        assert!(!files.contains(&"shared/output.gguf".to_string()));
    }

    #[test]
    fn files_for_last_stage() {
        let manifest = test_manifest(4);
        let stage = StageAssignment {
            stage_index: 1,
            peer_id: None,
            layer_start: 2,
            layer_end: 4,
            weight_bytes: 0,
        };
        let files = files_for_stage(&manifest, &stage);
        assert!(files.contains(&"shared/metadata.gguf".to_string()));
        assert!(files.contains(&"shared/output.gguf".to_string()));
        assert!(files.contains(&"shared/embeddings.gguf".to_string())); // tied weights
        assert!(files.contains(&"layers/layer-002.gguf".to_string()));
        assert!(files.contains(&"layers/layer-003.gguf".to_string()));
        assert!(!files.contains(&"layers/layer-000.gguf".to_string()));
    }
}

// ─── Stage Process Launch ────────────────────────────────────────────────────

use std::process::Stdio;
use tokio::process::{Child, Command};

/// A running stage server process.
pub struct StageProcess {
    pub stage_index: u32,
    pub port: u16,
    pub child: Child,
    pub config_path: PathBuf,
}

impl Drop for StageProcess {
    fn drop(&mut self) {
        // Best-effort kill on drop
        let _ = self.child.start_kill();
    }
}

/// Configuration for launching a stage server.
pub struct StageLaunchConfig {
    pub binary_path: PathBuf,
    pub package_dir: PathBuf,
    pub stage: StageAssignment,
    pub bind_port: u16,
    pub activation_width: u32,
    pub activation_wire_dtype: String,
    pub downstream_addr: Option<String>,
    pub upstream_addr: Option<String>,
    pub run_dir: PathBuf,
    pub run_id: String,
    pub model_id: String,
}

/// Launch a stage server process.
pub async fn launch_stage_server(config: StageLaunchConfig) -> Result<StageProcess> {
    let stage_id = format!("stage-{}", config.stage.stage_index);

    // Write config JSON
    let config_json = serde_json::json!({
        "run_id": config.run_id,
        "model_id": config.model_id,
        "model_path": config.package_dir.to_str().unwrap_or(""),
        "load_mode": "layer-package",
        "stage_id": &stage_id,
        "stage_index": config.stage.stage_index,
        "topology_id": &config.run_id,
        "layer_start": config.stage.layer_start,
        "layer_end": config.stage.layer_end,
        "n_gpu_layers": -1,
        "ctx_size": 4096,
        "filter_tensors_on_load": true,
        "bind_addr": format!("0.0.0.0:{}", config.bind_port),
        "downstream": config.downstream_addr.as_ref().map(|addr| serde_json::json!({
            "endpoint": addr,
            "stage_id": format!("stage-{}", config.stage.stage_index + 1),
            "stage_index": config.stage.stage_index + 1,
        })),
        "upstream": config.upstream_addr.as_ref().map(|addr| serde_json::json!({
            "endpoint": addr,
            "stage_id": format!("stage-{}", config.stage.stage_index.saturating_sub(1)),
            "stage_index": config.stage.stage_index.saturating_sub(1),
        })),
    });

    let config_path = config.run_dir.join(format!("{stage_id}.json"));
    std::fs::create_dir_all(&config.run_dir)?;
    std::fs::write(&config_path, serde_json::to_string_pretty(&config_json)?)?;

    let log_path = config.run_dir.join(format!("{stage_id}.log"));
    let log_file = std::fs::File::create(&log_path)?;

    let child = Command::new(&config.binary_path)
        .arg("serve-binary")
        .arg("--config")
        .arg(&config_path)
        .arg("--activation-width")
        .arg(config.activation_width.to_string())
        .arg("--activation-wire-dtype")
        .arg(&config.activation_wire_dtype)
        .stdout(Stdio::from(log_file.try_clone()?))
        .stderr(Stdio::from(log_file))
        .kill_on_drop(true)
        .spawn()
        .with_context(|| format!("spawn skippy-server for {stage_id}"))?;

    Ok(StageProcess {
        stage_index: config.stage.stage_index,
        port: config.bind_port,
        child,
        config_path,
    })
}

/// Configuration for launching the OpenAI driver (connects to stage chain).
pub struct DriverLaunchConfig {
    pub binary_path: PathBuf,
    pub package_dir: PathBuf,
    pub first_stage_addr: String,
    pub bind_port: u16,
    pub activation_width: u32,
    pub activation_wire_dtype: String,
    pub model_id: String,
    pub run_dir: PathBuf,
    pub run_id: String,
    pub layer_start: u32,
    pub layer_end: u32,
}

/// A running OpenAI driver process.
pub struct DriverProcess {
    pub port: u16,
    pub child: Child,
}

impl Drop for DriverProcess {
    fn drop(&mut self) {
        let _ = self.child.start_kill();
    }
}

/// Launch the OpenAI driver that connects to the stage chain.
/// This provides the /v1/chat/completions endpoint.
pub async fn launch_driver(config: DriverLaunchConfig) -> Result<DriverProcess> {
    // The driver needs a stage config for the first stage (to load tokenizer)
    let driver_config = serde_json::json!({
        "run_id": config.run_id,
        "model_id": config.model_id,
        "model_path": config.package_dir.to_str().unwrap_or(""),
        "load_mode": "layer-package",
        "stage_id": "driver",
        "stage_index": 0,
        "topology_id": &config.run_id,
        "layer_start": config.layer_start,
        "layer_end": config.layer_end,
        "n_gpu_layers": 0,
        "ctx_size": 4096,
        "filter_tensors_on_load": true,
        "bind_addr": format!("127.0.0.1:{}", config.bind_port + 100),
    });

    let config_path = config.run_dir.join("driver.json");
    std::fs::create_dir_all(&config.run_dir)?;
    std::fs::write(&config_path, serde_json::to_string_pretty(&driver_config)?)?;

    let log_path = config.run_dir.join("driver.log");
    let log_file = std::fs::File::create(&log_path)?;

    let child = Command::new(&config.binary_path)
        .arg("serve-openai")
        .arg("--config")
        .arg(&config_path)
        .arg("--bind-addr")
        .arg(format!("127.0.0.1:{}", config.bind_port))
        .arg("--first-stage-addr")
        .arg(&config.first_stage_addr)
        .arg("--activation-wire-dtype")
        .arg(&config.activation_wire_dtype)
        .stdout(Stdio::from(log_file.try_clone()?))
        .stderr(Stdio::from(log_file))
        .kill_on_drop(true)
        .spawn()
        .context("spawn skippy-server serve-openai driver")?;

    Ok(DriverProcess {
        port: config.bind_port,
        child,
    })
}

// ─── Integration with Election Loop ─────────────────────────────────────────


/// Parameters for starting staged inference (replaces StartLlamaParams).
pub struct StartSkippyParams<'a> {
    pub bin_dir: &'a Path,
    pub model_path: &'a Path,
    pub model_name: &'a str,
    /// Available memory per node: (peer_id, available_bytes).
    /// None peer_id = local node.
    pub node_capacities: Vec<NodeCapability>,
    /// Tunnel ports for each peer (peer_id → local tunnel port).
    /// Stage downstream connections go through these.
    pub tunnel_ports: &'a std::collections::HashMap<iroh::EndpointId, u16>,
    pub ctx_size: u32,
}

/// Result of starting staged inference — compatible with election loop expectations.
pub struct SkippyInferenceResult {
    /// Port where OpenAI HTTP is available (same as llama-server would provide)
    pub http_port: u16,
    /// All spawned processes (stages + driver)
    pub processes: Vec<StageProcess>,
    pub driver: DriverProcess,
    /// Context length
    pub context_length: u32,
}

/// Start staged inference: plan topology, launch stages, launch driver.
/// Returns the HTTP port for the OpenAI endpoint.
pub async fn start_skippy(params: StartSkippyParams<'_>) -> Result<SkippyInferenceResult> {
    let binary_path = params.bin_dir.join("skippy-server");
    anyhow::ensure!(
        binary_path.exists(),
        "skippy-server not found at {}",
        binary_path.display()
    );

    // Determine if this is a layer package or plain GGUF
    let is_layer_package = params.model_path.join("model-package.json").is_file();

    let run_id = format!("mesh-{}", std::process::id());
    let run_dir = std::env::temp_dir().join("mesh-llm-staged").join(&run_id);
    std::fs::create_dir_all(&run_dir)?;

    if !is_layer_package {
        // Single GGUF, single stage — just run serve-openai directly
        let http_port = find_free_port().await?;
        let config_json = serde_json::json!({
            "run_id": &run_id,
            "model_id": params.model_name,
            "model_path": params.model_path.to_str().unwrap_or(""),
            "load_mode": "runtime-slice",
            "stage_id": "stage-0",
            "stage_index": 0,
            "topology_id": &run_id,
            "layer_start": 0,
            "layer_end": 999, // Will be clamped by the model's actual layer count
            "n_gpu_layers": -1,
            "ctx_size": params.ctx_size,
            "filter_tensors_on_load": true,
            "bind_addr": format!("127.0.0.1:{}", http_port + 100),
        });
        let config_path = run_dir.join("stage-0.json");
        std::fs::write(&config_path, serde_json::to_string_pretty(&config_json)?)?;

        let log_path = run_dir.join("driver.log");
        let log_file = std::fs::File::create(&log_path)?;

        let child = Command::new(&binary_path)
            .arg("serve-openai")
            .arg("--config")
            .arg(&config_path)
            .arg("--bind-addr")
            .arg(format!("127.0.0.1:{http_port}"))
            .stdout(Stdio::from(log_file.try_clone()?))
            .stderr(Stdio::from(log_file))
            .kill_on_drop(true)
            .spawn()
            .context("spawn skippy-server serve-openai (solo)")?;

        // Wait for it to be ready
        wait_for_http_ready(http_port, 120).await?;

        return Ok(SkippyInferenceResult {
            http_port,
            processes: vec![],
            driver: DriverProcess { port: http_port, child },
            context_length: params.ctx_size,
        });
    }

    // Layer package: plan topology and launch stages
    tracing::info!("parsing layer package manifest from {}", params.model_path.display());
    let manifest = PackageManifest::from_dir(params.model_path)?;
    tracing::info!("manifest parsed: {} layers, model_id={}", manifest.layer_count, manifest.model_id);
    let plan = plan_topology(&manifest, &params.node_capacities)?;
    tracing::info!("topology planned: {} stages", plan.stages.len());

    // Single local stage: run serve-openai directly (no separate stage server needed)
    if plan.stages.len() == 1 && plan.stages[0].peer_id.is_none() {
        let http_port = find_free_port().await?;
        let stage = &plan.stages[0];
        let config_json = serde_json::json!({
            "run_id": &run_id,
            "model_id": &manifest.model_id,
            "model_path": params.model_path.to_str().unwrap_or(""),
            "load_mode": "layer-package",
            "stage_id": "stage-0",
            "stage_index": 0,
            "topology_id": &run_id,
            "layer_start": stage.layer_start,
            "layer_end": stage.layer_end,
            "n_gpu_layers": -1,
            "ctx_size": params.ctx_size,
            "filter_tensors_on_load": true,
            "bind_addr": format!("127.0.0.1:{}", http_port + 100),
        });
        let config_path = run_dir.join("stage-0.json");
        std::fs::write(&config_path, serde_json::to_string_pretty(&config_json)?)?;

        let log_path = run_dir.join("driver.log");
        let log_file = std::fs::File::create(&log_path)?;

        tracing::info!("launching serve-openai (single local stage, layer-package) on port {http_port}");
        let child = Command::new(&binary_path)
            .arg("serve-openai")
            .arg("--config")
            .arg(&config_path)
            .arg("--bind-addr")
            .arg(format!("127.0.0.1:{http_port}"))
            .stdout(Stdio::from(log_file.try_clone()?))
            .stderr(Stdio::from(log_file))
            .kill_on_drop(true)
            .spawn()
            .context("spawn skippy-server serve-openai (single stage layer-package)")?;

        wait_for_http_ready(http_port, 120).await?;

        return Ok(SkippyInferenceResult {
            http_port,
            processes: vec![],
            driver: DriverProcess { port: http_port, child },
            context_length: params.ctx_size,
        });
    }

    let base_port: u16 = find_free_port().await?;
    let mut stage_processes = Vec::new();

    // Launch stages in reverse order (last stage first, so it's listening when upstream connects)
    for (i, stage) in plan.stages.iter().enumerate().rev() {
        let stage_port = base_port + 1 + i as u16;
        let downstream_addr = if i < plan.stages.len() - 1 {
            let next_stage = &plan.stages[i + 1];
            if let Some(peer_id) = next_stage.peer_id {
                // Remote stage: connect through tunnel
                let tunnel_port = params.tunnel_ports.get(&peer_id)
                    .with_context(|| format!("no tunnel port for peer running stage-{}", i + 1))?;
                Some(format!("tcp://127.0.0.1:{tunnel_port}"))
            } else {
                // Local stage: direct connection
                Some(format!("tcp://127.0.0.1:{}", base_port + 2 + i as u16))
            }
        } else {
            None
        };

        if stage.peer_id.is_some() {
            // Remote stage — peer launches it. Skip for now (gossip coordination).
            // TODO: tell peer to start its stage via gossip command
            continue;
        }

        let process = launch_stage_server(StageLaunchConfig {
            binary_path: binary_path.clone(),
            package_dir: params.model_path.to_path_buf(),
            stage: stage.clone(),
            bind_port: stage_port,
            activation_width: 4096, // TODO: read from manifest/metadata
            activation_wire_dtype: "f16".to_string(),
            downstream_addr,
            upstream_addr: None, // Stages accept upstream connections, don't initiate
            run_dir: run_dir.clone(),
            run_id: run_id.clone(),
            model_id: manifest.model_id.clone(),
        }).await?;

        stage_processes.push(process);
    }

    // Wait for stages to be ready (they need to load models + materialize)
    tokio::time::sleep(std::time::Duration::from_secs(5)).await;

    // Launch the OpenAI driver connecting to stage-0
    let first_stage_port = base_port + 1;
    let http_port = base_port;

    let driver = launch_driver(DriverLaunchConfig {
        binary_path,
        package_dir: params.model_path.to_path_buf(),
        first_stage_addr: format!("tcp://127.0.0.1:{first_stage_port}"),
        bind_port: http_port,
        activation_width: 4096,
        activation_wire_dtype: "f16".to_string(),
        model_id: manifest.model_id.clone(),
        run_dir: run_dir.clone(),
        run_id,
        layer_start: 0,
        layer_end: 1,
    }).await?;

    // Wait for driver to be ready
    wait_for_http_ready(http_port, 120).await?;

    Ok(SkippyInferenceResult {
        http_port,
        processes: stage_processes,
        driver,
        context_length: params.ctx_size,
    })
}

async fn find_free_port() -> Result<u16> {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    Ok(listener.local_addr()?.port())
}

async fn wait_for_http_ready(port: u16, timeout_secs: u64) -> Result<()> {
    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(timeout_secs);
    loop {
        if tokio::time::Instant::now() > deadline {
            anyhow::bail!("staged inference server not ready after {timeout_secs}s on port {port}");
        }
        match tokio::net::TcpStream::connect(format!("127.0.0.1:{port}")).await {
            Ok(_) => return Ok(()),
            Err(_) => tokio::time::sleep(std::time::Duration::from_millis(500)).await,
        }
    }
}

// --- Stage command protocol (host ↔ peer coordination) ---

/// Command sent from host to peer to start a stage server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StartStageCommand {
    pub package_ref: String,        // e.g. "hf://meshllm/Qwen3-235B-A22B-UD-Q4_K_XL-layers"
    pub layer_start: u32,
    pub layer_end: u32,
    pub stage_index: u32,
    pub bind_port: u16,
    pub activation_width: u32,
    pub activation_wire_dtype: String,
    pub model_id: String,
    pub run_id: String,
}

/// Response from peer when stage is ready.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageReadyResponse {
    pub status: String,             // "ready" or "error"
    pub port: u16,
    pub error: Option<String>,
}

/// Handle an incoming stage command on the peer side.
/// Downloads layers, materializes, starts skippy-server, responds when ready.
pub async fn handle_stage_command(
    command: StartStageCommand,
    bin_dir: &Path,
) -> StageReadyResponse {
    let binary_path = bin_dir.join("skippy-server");
    if !binary_path.exists() {
        return StageReadyResponse {
            status: "error".to_string(),
            port: 0,
            error: Some(format!("skippy-server not found at {}", binary_path.display())),
        };
    }

    let run_dir = std::env::temp_dir()
        .join("mesh-llm-staged")
        .join(&command.run_id);
    if let Err(e) = std::fs::create_dir_all(&run_dir) {
        return StageReadyResponse {
            status: "error".to_string(),
            port: 0,
            error: Some(format!("create run dir: {e}")),
        };
    }

    let stage_id = format!("stage-{}", command.stage_index);
    let config_json = serde_json::json!({
        "run_id": &command.run_id,
        "model_id": &command.model_id,
        "model_path": &command.package_ref,
        "load_mode": "layer-package",
        "stage_id": &stage_id,
        "stage_index": command.stage_index,
        "topology_id": &command.run_id,
        "layer_start": command.layer_start,
        "layer_end": command.layer_end,
        "n_gpu_layers": -1,
        "ctx_size": 4096,
        "filter_tensors_on_load": true,
        "bind_addr": format!("0.0.0.0:{}", command.bind_port),
    });

    let config_path = run_dir.join(format!("{stage_id}.json"));
    if let Err(e) = std::fs::write(&config_path, serde_json::to_string_pretty(&config_json).unwrap_or_default()) {
        return StageReadyResponse {
            status: "error".to_string(),
            port: 0,
            error: Some(format!("write config: {e}")),
        };
    }

    let log_path = run_dir.join(format!("{stage_id}.log"));
    let log_file = match std::fs::File::create(&log_path) {
        Ok(f) => f,
        Err(e) => return StageReadyResponse {
            status: "error".to_string(),
            port: 0,
            error: Some(format!("create log: {e}")),
        },
    };

    let child = Command::new(&binary_path)
        .arg("serve-binary")
        .arg("--config")
        .arg(&config_path)
        .arg("--activation-width")
        .arg(command.activation_width.to_string())
        .arg("--activation-wire-dtype")
        .arg(&command.activation_wire_dtype)
        .stdout(Stdio::from(log_file.try_clone().unwrap()))
        .stderr(Stdio::from(log_file))
        .kill_on_drop(true)
        .spawn();

    match child {
        Ok(_child) => {
            // Wait for the stage to start listening
            let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(180);
            loop {
                if tokio::time::Instant::now() > deadline {
                    return StageReadyResponse {
                        status: "error".to_string(),
                        port: command.bind_port,
                        error: Some("stage server did not become ready in 180s".to_string()),
                    };
                }
                match tokio::net::TcpStream::connect(format!("127.0.0.1:{}", command.bind_port)).await {
                    Ok(_) => break,
                    Err(_) => tokio::time::sleep(std::time::Duration::from_millis(500)).await,
                }
            }
            StageReadyResponse {
                status: "ready".to_string(),
                port: command.bind_port,
                error: None,
            }
        }
        Err(e) => StageReadyResponse {
            status: "error".to_string(),
            port: 0,
            error: Some(format!("spawn skippy-server: {e}")),
        },
    }
}

/// Send a start-stage command to a peer and wait for ready response.
pub async fn send_stage_command(
    conn: &iroh::endpoint::Connection,
    command: &StartStageCommand,
) -> Result<StageReadyResponse> {
    let (mut send, mut recv) = conn.open_bi().await?;
    send.write_all(&[crate::protocol::STREAM_STAGE_COMMAND]).await?;
    let payload = serde_json::to_vec(command)?;
    let len = (payload.len() as u32).to_le_bytes();
    send.write_all(&len).await?;
    send.write_all(&payload).await?;
    send.finish()?;

    // Read response
    let mut len_buf = [0u8; 4];
    recv.read_exact(&mut len_buf).await?;
    let resp_len = u32::from_le_bytes(len_buf) as usize;
    let mut resp_buf = vec![0u8; resp_len];
    recv.read_exact(&mut resp_buf).await?;

    let response: StageReadyResponse = serde_json::from_slice(&resp_buf)?;
    Ok(response)
}
