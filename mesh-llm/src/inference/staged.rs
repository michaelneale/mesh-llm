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
