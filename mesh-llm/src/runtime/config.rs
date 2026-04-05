use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::path::{Path, PathBuf};

pub const AUTHORED_CONFIG_VERSION: u32 = 1;

// Path helpers

fn mesh_root_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".mesh-llm")
}

/// Returns the resolved path to the mesh config file.
///
/// Precedence:
/// 1. `override_path` argument (from `--mesh-config` CLI flag)
/// 2. `MESH_LLM_MESH_CONFIG` environment variable
/// 3. Default: `~/.mesh-llm/mesh.toml`
pub fn mesh_config_path(override_path: Option<PathBuf>) -> PathBuf {
    mesh_config_path_with_env(override_path, |key| std::env::var(key))
}

fn mesh_config_path_with_env(
    override_path: Option<PathBuf>,
    env_lookup: impl for<'a> Fn(&'a str) -> Result<String, std::env::VarError>,
) -> PathBuf {
    if let Some(path) = override_path {
        return path;
    }
    if let Ok(env_path) = env_lookup("MESH_LLM_MESH_CONFIG") {
        return PathBuf::from(env_path);
    }
    mesh_root_dir().join("mesh.toml")
}

/// Returns the path to the per-node runtime config file.
///
/// This file holds the projected runtime view for the local node,
/// derived from the authored mesh config. Default: `~/.mesh-llm/node.toml`.
//
// DEFERRED: Not called at runtime yet. This is the activation point for
// per-node config persistence once the config UI or runtime startup begins
// writing `node.toml`. Do not remove.
#[allow(dead_code)]
pub fn node_config_path() -> PathBuf {
    mesh_root_dir().join("node.toml")
}

/// Loads the authored mesh config from the resolved path.
///
/// Returns a default empty config (version 1, no nodes) when the file does not
/// exist. Malformed TOML or unsupported versions are errors.
pub fn load_mesh_config(override_path: Option<PathBuf>) -> Result<AuthoredMeshConfig> {
    let path = mesh_config_path(override_path);
    AuthoredMeshConfig::load(&path)
}

// DEFERRED: Called only by AuthoredMeshConfig::save and NodeConfig::save, both of which
// are deferred pending config-UI activation. This is the shared atomic-write
// implementation for all config persistence. Do not remove.
#[allow(dead_code)]
fn save_toml_atomically<T: Serialize>(value: &T, path: &Path, label: &str) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).with_context(|| {
                format!("failed to create config directory {}", parent.display())
            })?;
        }
    }
    let raw =
        toml::to_string_pretty(value).with_context(|| format!("failed to serialize {label}"))?;

    let file_name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("config.toml");
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let tmp_path = path.with_file_name(format!(".{file_name}.{nanos}.tmp"));

    let mut tmp_file = std::fs::File::create(&tmp_path)
        .with_context(|| format!("failed to create temp config {}", tmp_path.display()))?;
    tmp_file
        .write_all(raw.as_bytes())
        .with_context(|| format!("failed to write temp config {}", tmp_path.display()))?;
    tmp_file
        .sync_all()
        .with_context(|| format!("failed to sync temp config {}", tmp_path.display()))?;

    #[cfg(windows)]
    {
        match std::fs::rename(&tmp_path, path) {
            Ok(()) => Ok(()),
            Err(rename_err) if rename_err.kind() == std::io::ErrorKind::AlreadyExists => {
                if let Err(remove_err) = std::fs::remove_file(path) {
                    let _ = std::fs::remove_file(&tmp_path);
                    return Err(remove_err).with_context(|| {
                        format!(
                            "failed to remove existing config {} after rename via {} failed: {}",
                            path.display(),
                            tmp_path.display(),
                            rename_err
                        )
                    });
                }

                std::fs::rename(&tmp_path, path)
                    .map_err(|retry_err| {
                        let _ = std::fs::remove_file(&tmp_path);
                        retry_err
                    })
                    .with_context(|| {
                        format!(
                            "failed to atomically replace config {} via {}",
                            path.display(),
                            tmp_path.display()
                        )
                    })
            }
            Err(rename_err) => {
                let _ = std::fs::remove_file(&tmp_path);
                Err(rename_err).with_context(|| {
                    format!(
                        "failed to atomically replace config {} via {}",
                        path.display(),
                        tmp_path.display()
                    )
                })
            }
        }
    }

    #[cfg(not(windows))]
    {
        std::fs::rename(&tmp_path, path).with_context(|| {
            format!(
                "failed to atomically replace config {} via {}",
                path.display(),
                tmp_path.display()
            )
        })
    }
}

// Mesh-level runtime config (projected view of all nodes, no authored fields)

/// Runtime view of the entire mesh configuration. Derived from
/// [`AuthoredMeshConfig`] by stripping split, gpu_index, and model_key.
/// Suitable for the API layer and runtime reads.
// DEFERRED: Not yet constructed at runtime. Activation point: API layer reads
// mesh-wide config from node.toml files or a shared config store.
// Do not remove — this type and its methods are the planned surface for future
// mesh-config query/broadcast logic.
#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct MeshConfig {
    #[serde(default)]
    pub nodes: Vec<NodeConfig>,
}

impl MeshConfig {
    // DEFERRED: Will be called by the API layer to read a shared mesh-config store.
    #[allow(dead_code)]
    pub fn load(path: &Path) -> Result<Self> {
        if !path.exists() {
            return Ok(Self::default());
        }
        let raw = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read config {}", path.display()))?;
        toml::from_str(&raw).with_context(|| format!("failed to parse config {}", path.display()))
    }

    // DEFERRED: Will be called by runtime/API layer to look up a node's config.
    #[allow(dead_code)]
    pub fn for_node(&self, node_id: &str) -> Option<NodeConfig> {
        self.nodes
            .iter()
            .find(|node| node.node_id == node_id)
            .cloned()
    }
}

// Authored config types (user-editable, full fidelity)

/// The authored mesh configuration. Authored by the operator and stored at
/// `~/.mesh-llm/mesh.toml`. Contains split metadata, gpu placement, and model
/// keys that are used by the config UI and broadcast to peers, but which are
/// stripped before being passed to launch logic.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AuthoredMeshConfig {
    pub version: u32,
    #[serde(default)]
    pub nodes: Vec<AuthoredNodeConfig>,
}

/// Per-node authored configuration entry.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AuthoredNodeConfig {
    pub node_id: String,
    pub hostname: Option<String>,
    #[serde(default)]
    pub placement_mode: PlacementMode,
    #[serde(default)]
    pub models: Vec<AuthoredModelAssignment>,
}

/// How the models on this node are placed across GPUs.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PlacementMode {
    /// All models share one GPU pool (default).
    #[default]
    Pooled,
    /// Each model (or split fragment) targets a specific GPU ordinal.
    Separate,
}

/// Absolute layer-range split for a dense model across two GPU slots.
///
/// Example: a 33-layer model split at layer 21 uses two assignments:
/// `{ start: 0, end: 21, total: 33 }` and `{ start: 21, end: 33, total: 33 }`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelSplit {
    pub start: u32,
    pub end: u32,
    pub total: u32,
}

/// A single model assignment in the authored config. A model can appear more
/// than once on a node (two entries = two GPU slots for a split model).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AuthoredModelAssignment {
    /// Display/launch name for the model.
    pub name: String,
    /// Optional stable key used to identify the model in the catalog and for
    /// ctx_size lookups (e.g. `"mk-qwen3-30b"`).
    pub model_key: Option<String>,
    /// Layer-range split for multi-GPU dense-model placement.
    pub split: Option<ModelSplit>,
    /// Explicit path to the GGUF file; overrides catalog resolution.
    pub path: Option<String>,
    /// Context window size for this model assignment.
    pub ctx_size: Option<u32>,
    /// Number of MoE experts to activate (for mixture-of-experts models).
    pub moe_experts: Option<u32>,
    /// GPU ordinal this assignment targets when `placement_mode = "separate"`.
    #[serde(default)]
    pub gpu_index: Option<u32>,
}

// Runtime config types (stripped-down, inert, no split/gpu_index)

/// Runtime view of a single model assignment. Split, gpu_index, and model_key
/// are dropped; only launch-relevant fields survive.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct ModelAssignment {
    pub name: String,
    pub path: Option<String>,
    pub ctx_size: Option<u32>,
    pub moe_experts: Option<u32>,
}

/// Runtime view of a single node's config. Produced by
/// [`AuthoredMeshConfig::for_node_runtime`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct NodeConfig {
    pub node_id: String,
    pub hostname: Option<String>,
    #[serde(default)]
    pub models: Vec<ModelAssignment>,
}

// AuthoredMeshConfig impl

impl Default for AuthoredMeshConfig {
    fn default() -> Self {
        Self {
            version: AUTHORED_CONFIG_VERSION,
            nodes: Vec::new(),
        }
    }
}

impl AuthoredMeshConfig {
    /// Loads an [`AuthoredMeshConfig`] from `path`.
    ///
    /// Returns the default empty config when the file does not exist.
    /// Returns an error for malformed TOML or unsupported versions.
    pub fn load(path: &Path) -> Result<Self> {
        if !path.exists() {
            return Ok(Self::default());
        }
        let raw = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read mesh config {}", path.display()))?;
        let parsed: Self = toml::from_str(&raw)
            .with_context(|| format!("failed to parse mesh config {}", path.display()))?;
        match parsed.version {
            AUTHORED_CONFIG_VERSION => {}
            other => bail!("unsupported authored config version {other}"),
        }
        let mut seen = std::collections::HashSet::new();
        for node in &parsed.nodes {
            if !seen.insert(node.node_id.as_str()) {
                bail!(
                    "duplicate node_id {:?} in mesh config {}",
                    node.node_id,
                    path.display()
                );
            }
        }
        Ok(parsed)
    }

    /// Persists this config to `path` using an atomic temp-file-then-rename strategy.
    ///
    /// The parent directory is created automatically if it does not exist.
    // DEFERRED: Will be called by the config UI when the operator writes a new mesh.toml.
    #[allow(dead_code)]
    pub fn save(&self, path: &Path) -> Result<()> {
        let mut normalized = self.clone();
        normalized.version = AUTHORED_CONFIG_VERSION;
        save_toml_atomically(&normalized, path, "mesh config")
    }

    /// Returns the runtime view for a specific node, or `None` if the node is
    /// not in the authored config. Split, `gpu_index`, and `model_key` are
    /// dropped — only `name`, `path`, `ctx_size`, and `moe_experts` survive.
    pub fn for_node_runtime(&self, node_id: &str) -> Option<NodeConfig> {
        self.nodes
            .iter()
            .find(|node| node.node_id == node_id)
            .map(AuthoredNodeConfig::to_runtime)
    }

    /// Returns the `ctx_size` configured for `model_name` on `node_id`.
    ///
    /// Matches by `name` or `model_key`.
    // DEFERRED: Will be called by launch logic to resolve ctx_size from authored config.
    #[allow(dead_code)]
    pub fn model_ctx_size(&self, node_id: &str, model_name: &str) -> Option<u32> {
        self.nodes
            .iter()
            .find(|n| n.node_id == node_id)
            .and_then(|n| {
                n.models
                    .iter()
                    .find(|m| m.name == model_name || m.model_key.as_deref() == Some(model_name))
            })
            .and_then(|m| m.ctx_size)
    }
}

impl AuthoredNodeConfig {
    /// Converts this authored node config to its inert runtime view.
    pub fn to_runtime(&self) -> NodeConfig {
        NodeConfig {
            node_id: self.node_id.clone(),
            hostname: self.hostname.clone(),
            models: self
                .models
                .iter()
                .map(AuthoredModelAssignment::to_runtime)
                .collect(),
        }
    }
}

impl AuthoredModelAssignment {
    /// Converts this authored model assignment to its inert runtime view.
    /// `split`, `gpu_index`, and `model_key` are intentionally dropped.
    pub fn to_runtime(&self) -> ModelAssignment {
        ModelAssignment {
            name: self.name.clone(),
            path: self.path.clone(),
            ctx_size: self.ctx_size,
            moe_experts: self.moe_experts,
        }
    }
}

impl NodeConfig {
    /// Loads a [`NodeConfig`] from `path`. Returns `Ok(None)` if the file does
    /// not exist; errors on malformed TOML.
    // DEFERRED: Will be called at startup to read node.toml once runtime
    // startup begins writing per-node projected config to disk.
    #[allow(dead_code)]
    pub fn load(path: &Path) -> Result<Option<Self>> {
        if !path.exists() {
            return Ok(None);
        }
        let raw = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read node config {}", path.display()))?;
        let parsed = toml::from_str(&raw)
            .with_context(|| format!("failed to parse node config {}", path.display()))?;
        Ok(Some(parsed))
    }

    // DEFERRED: Will be called at startup to persist the projected node.toml
    // once runtime startup activates per-node config writes.
    #[allow(dead_code)]
    pub fn save(&self, path: &Path) -> Result<()> {
        save_toml_atomically(self, path, "node config")
    }
}

// Hydration state (in-memory, inert)

/// In-memory holder for the local node's authored mesh configuration.
/// Populated at startup after the mesh node identity is established.
/// Intentionally inert — no mutations to runtime targets, launch, or gossip.
// DEFERRED: Fields are stored but not yet read. `authored` will be used by the
// config UI and API layer; `local` will drive model launch once hydration is
// activated. Do not remove — this struct is the in-memory bridge between
// mesh.toml and future launch/routing behavior.
#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct LocalMeshConfigState {
    /// The full authored mesh configuration as loaded from disk.
    pub authored: AuthoredMeshConfig,
    /// The projected runtime view for the local node, or `None` if the local
    /// node's identity was not found in the authored config.
    pub local: Option<NodeConfig>,
}

/// Hydrates a [`LocalMeshConfigState`] from the authored config and the local
/// node ID. Returns `local: None` if the node is not in the authored config.
///
/// Intentionally inert: does not mutate any runtime state.
pub fn hydrate_local_mesh_config(cfg: AuthoredMeshConfig, node_id: &str) -> LocalMeshConfigState {
    let local = cfg.for_node_runtime(node_id);
    LocalMeshConfigState {
        authored: cfg,
        local,
    }
}

// Tests

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(label: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("mesh_llm_{label}_{nanos}"));
        std::fs::create_dir_all(&dir).expect("create test temp dir");
        dir
    }

    // Wire format: [[nodes]] array with node_id inside (ported from PR #130)

    #[test]
    fn authored_config_supports_schema_v1_gpu_placement() {
        let raw = r#"
            version = 1

            [[nodes]]
            node_id = "node-a"
            hostname = "alpha.local"
            placement_mode = "separate"

            [[nodes.models]]
            name = "Qwen3-30B-A3B-Q4_K_M"
            model_key = "mk-qwen3-30b"
            split = { start = 0, end = 21, total = 33 }
            gpu_index = 0

            [[nodes.models]]
            name = "Qwen3-30B-A3B-Q4_K_M"
            model_key = "mk-qwen3-30b"
            split = { start = 21, end = 33, total = 33 }
            gpu_index = 1
        "#;

        let parsed: AuthoredMeshConfig = toml::from_str(raw).unwrap();
        assert_eq!(parsed.version, AUTHORED_CONFIG_VERSION);
        assert_eq!(parsed.nodes.len(), 1);
        assert_eq!(parsed.nodes[0].placement_mode, PlacementMode::Separate);
        assert_eq!(parsed.nodes[0].models.len(), 2);
        assert_eq!(parsed.nodes[0].models[0].gpu_index, Some(0));
        assert_eq!(parsed.nodes[0].models[1].gpu_index, Some(1));
        assert_eq!(
            parsed.nodes[0].models[0].split,
            Some(ModelSplit {
                start: 0,
                end: 21,
                total: 33
            })
        );
        assert_eq!(
            parsed.nodes[0].models[1].split,
            Some(ModelSplit {
                start: 21,
                end: 33,
                total: 33
            })
        );

        // Verify round-trip serialization
        let serialized = toml::to_string_pretty(&parsed).unwrap();
        assert!(serialized.contains("version = 1"));
        assert!(serialized.contains("placement_mode = \"separate\""));
        assert!(serialized.contains("gpu_index = 0"));
        assert!(serialized.contains("gpu_index = 1"));
    }

    #[test]
    fn authored_config_round_trip_with_split_and_model_key() {
        let config = AuthoredMeshConfig {
            version: AUTHORED_CONFIG_VERSION,
            nodes: vec![AuthoredNodeConfig {
                node_id: "node-a".into(),
                hostname: Some("alpha.local".into()),
                placement_mode: PlacementMode::Separate,
                models: vec![
                    AuthoredModelAssignment {
                        name: "Qwen3-30B-A3B-Q4_K_M".into(),
                        model_key: Some("mk-qwen3-30b".into()),
                        split: Some(ModelSplit {
                            start: 0,
                            end: 21,
                            total: 33,
                        }),
                        path: Some("/Users/test/.models/Qwen3-30B-A3B-Q4_K_M.gguf".into()),
                        ctx_size: Some(8192),
                        moe_experts: Some(24),
                        gpu_index: Some(0),
                    },
                    AuthoredModelAssignment {
                        name: "Qwen3-30B-A3B-Q4_K_M".into(),
                        model_key: Some("mk-qwen3-30b".into()),
                        split: Some(ModelSplit {
                            start: 21,
                            end: 33,
                            total: 33,
                        }),
                        path: None,
                        ctx_size: Some(8192),
                        moe_experts: Some(24),
                        gpu_index: Some(1),
                    },
                ],
            }],
        };

        let raw = toml::to_string_pretty(&config).unwrap();
        let parsed: AuthoredMeshConfig = toml::from_str(&raw).unwrap();
        assert_eq!(parsed, config);
    }

    // Version handling

    #[test]
    fn authored_config_missing_file_defaults_to_versioned_empty() {
        let dir = temp_dir("missing");
        let missing = dir.join("mesh.toml");

        let loaded = AuthoredMeshConfig::load(&missing).unwrap();
        assert_eq!(loaded.version, AUTHORED_CONFIG_VERSION);
        assert!(loaded.nodes.is_empty());
    }

    #[test]
    fn authored_config_rejects_unsupported_version() {
        let dir = temp_dir("unsupported_ver");
        let v2_path = dir.join("mesh-v2.toml");
        std::fs::write(&v2_path, "version = 2\n\n[[nodes]]\nnode_id = \"node-a\"\n").unwrap();

        let result = AuthoredMeshConfig::load(&v2_path);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("unsupported authored config version 2"),
            "unexpected error: {msg}"
        );

        // version 1 still works
        let v1_path = dir.join("mesh-v1.toml");
        let valid = AuthoredMeshConfig::default();
        valid.save(&v1_path).unwrap();
        let saved_raw = std::fs::read_to_string(&v1_path).unwrap();
        assert!(saved_raw.contains("version = 1"));
    }

    #[test]
    fn authored_config_rejects_invalid_placement_mode() {
        let invalid = r#"
            version = 1
            [[nodes]]
            node_id = "node-a"
            placement_mode = "invalid_mode"
        "#;
        let result = toml::from_str::<AuthoredMeshConfig>(invalid);
        assert!(
            result.is_err(),
            "invalid placement_mode should be rejected by serde"
        );
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("placement_mode"),
            "error should mention placement_mode, got: {msg}"
        );
    }

    // Runtime projection: for_node_runtime drops split/gpu_index/model_key

    #[test]
    fn authored_for_node_runtime_drops_split_metadata() {
        let config = AuthoredMeshConfig {
            version: AUTHORED_CONFIG_VERSION,
            nodes: vec![AuthoredNodeConfig {
                node_id: "node-a".into(),
                hostname: Some("alpha.local".into()),
                placement_mode: PlacementMode::Pooled,
                models: vec![AuthoredModelAssignment {
                    name: "Qwen3-8B-Q4_K_M".into(),
                    model_key: Some("mk-qwen3-8b".into()),
                    split: Some(ModelSplit {
                        start: 0,
                        end: 33,
                        total: 33,
                    }),
                    path: Some("/models/Qwen3-8B-Q4_K_M.gguf".into()),
                    ctx_size: Some(4096),
                    moe_experts: None,
                    gpu_index: Some(0),
                }],
            }],
        };

        let runtime = config.for_node_runtime("node-a").unwrap();
        assert_eq!(runtime.node_id, "node-a");
        assert_eq!(runtime.models.len(), 1);
        assert_eq!(runtime.models[0].name, "Qwen3-8B-Q4_K_M");
        assert_eq!(runtime.models[0].ctx_size, Some(4096));
        assert_eq!(
            runtime.models[0].path.as_deref(),
            Some("/models/Qwen3-8B-Q4_K_M.gguf")
        );
        // Split, gpu_index, model_key must be absent from the runtime type
        // (compile-time guarantee: ModelAssignment has no such fields)
    }

    #[test]
    fn for_node_runtime_returns_none_for_unknown_node() {
        let config = AuthoredMeshConfig {
            version: AUTHORED_CONFIG_VERSION,
            nodes: vec![AuthoredNodeConfig {
                node_id: "node-a".into(),
                hostname: None,
                placement_mode: PlacementMode::Pooled,
                models: vec![],
            }],
        };
        assert!(config.for_node_runtime("node-z").is_none());
    }

    #[test]
    fn for_node_runtime_allows_same_model_twice_for_split() {
        // Two entries for the same model = multi-GPU split scenario — must be allowed
        let config = AuthoredMeshConfig {
            version: AUTHORED_CONFIG_VERSION,
            nodes: vec![AuthoredNodeConfig {
                node_id: "node-a".into(),
                hostname: None,
                placement_mode: PlacementMode::Separate,
                models: vec![
                    AuthoredModelAssignment {
                        name: "Qwen3-30B".into(),
                        model_key: None,
                        split: Some(ModelSplit {
                            start: 0,
                            end: 21,
                            total: 33,
                        }),
                        path: None,
                        ctx_size: None,
                        moe_experts: None,
                        gpu_index: Some(0),
                    },
                    AuthoredModelAssignment {
                        name: "Qwen3-30B".into(),
                        model_key: None,
                        split: Some(ModelSplit {
                            start: 21,
                            end: 33,
                            total: 33,
                        }),
                        path: None,
                        ctx_size: None,
                        moe_experts: None,
                        gpu_index: Some(1),
                    },
                ],
            }],
        };
        let runtime = config.for_node_runtime("node-a").unwrap();
        // Both slots survive in the runtime view
        assert_eq!(runtime.models.len(), 2);
        assert_eq!(runtime.models[0].name, "Qwen3-30B");
        assert_eq!(runtime.models[1].name, "Qwen3-30B");
    }

    // model_ctx_size helper

    #[test]
    fn model_ctx_size_matches_by_name_and_model_key() {
        let config = AuthoredMeshConfig {
            version: AUTHORED_CONFIG_VERSION,
            nodes: vec![AuthoredNodeConfig {
                node_id: "node-a".into(),
                hostname: None,
                placement_mode: PlacementMode::Pooled,
                models: vec![AuthoredModelAssignment {
                    name: "Qwen3-8B-Q4_K_M".into(),
                    model_key: Some("mk-qwen3-8b".into()),
                    split: None,
                    path: None,
                    ctx_size: Some(8192),
                    moe_experts: None,
                    gpu_index: None,
                }],
            }],
        };

        // Match by name
        assert_eq!(
            config.model_ctx_size("node-a", "Qwen3-8B-Q4_K_M"),
            Some(8192)
        );
        // Match by model_key
        assert_eq!(config.model_ctx_size("node-a", "mk-qwen3-8b"), Some(8192));
        // Unknown model → None
        assert_eq!(config.model_ctx_size("node-a", "unknown"), None);
        // Unknown node → None
        assert_eq!(config.model_ctx_size("node-z", "Qwen3-8B-Q4_K_M"), None);
    }

    // Persistence: save/load round-trip

    #[test]
    fn authored_config_save_load_round_trip() {
        let dir = temp_dir("round_trip");
        let path = dir.join("mesh.toml");

        let original = AuthoredMeshConfig {
            version: AUTHORED_CONFIG_VERSION,
            nodes: vec![AuthoredNodeConfig {
                node_id: "worker1".into(),
                hostname: Some("worker1.local".into()),
                placement_mode: PlacementMode::Pooled,
                models: vec![AuthoredModelAssignment {
                    name: "Qwen2.5-7B".into(),
                    model_key: Some("mk-qwen-7b".into()),
                    split: None,
                    path: None,
                    ctx_size: Some(4096),
                    moe_experts: None,
                    gpu_index: None,
                }],
            }],
        };

        original.save(&path).expect("save should succeed");
        let loaded = AuthoredMeshConfig::load(&path).expect("load should succeed");
        assert_eq!(loaded, original);
    }

    #[test]
    fn authored_config_save_creates_parent_dirs_atomically() {
        let base = temp_dir("save_creates_dir");
        let nested = base.join("deep").join("nested").join("mesh.toml");

        let cfg = AuthoredMeshConfig::default();
        cfg.save(&nested)
            .expect("save to nested path should succeed");

        assert!(nested.exists(), "file should exist after save");
        let raw = std::fs::read_to_string(&nested).unwrap();
        assert!(raw.contains("version = 1"));
        AuthoredMeshConfig::load(&nested).expect("saved file should re-parse cleanly");
    }

    #[test]
    fn mesh_config_load_reports_invalid_toml() {
        let dir = temp_dir("invalid_toml");
        let path = dir.join("bad.toml");
        std::fs::write(&path, "version = 1\n[unclosed_section\n").unwrap();

        let result = load_mesh_config(Some(path));
        assert!(result.is_err(), "malformed TOML should return Err");
    }

    #[test]
    fn mesh_config_load_missing_file_returns_default() {
        let dir = temp_dir("missing_file");
        let path = dir.join("does_not_exist.toml");

        let result = load_mesh_config(Some(path));
        let cfg = result.expect("missing file should return Ok with default config");
        assert_eq!(cfg.version, AUTHORED_CONFIG_VERSION);
        assert!(cfg.nodes.is_empty());
    }

    // ModelSplit: absolute layer ranges

    #[test]
    fn model_split_round_trips_as_inline_table() {
        let raw = r#"
            version = 1
            [[nodes]]
            node_id = "n"
            [[nodes.models]]
            name = "GLM"
            split = { start = 0, end = 16, total = 32 }
        "#;
        let cfg: AuthoredMeshConfig = toml::from_str(raw).unwrap();
        let split = cfg.nodes[0].models[0].split.as_ref().unwrap();
        assert_eq!(split.start, 0);
        assert_eq!(split.end, 16);
        assert_eq!(split.total, 32);
    }

    // PlacementMode: pooled is the default

    #[test]
    fn placement_mode_defaults_to_pooled() {
        let raw = "version = 1\n[[nodes]]\nnode_id = \"n\"\n";
        let cfg: AuthoredMeshConfig = toml::from_str(raw).unwrap();
        assert_eq!(cfg.nodes[0].placement_mode, PlacementMode::Pooled);
    }

    // MeshConfig: mesh-level runtime view

    #[test]
    fn mesh_config_for_node_extracts_specific_node() {
        let config = MeshConfig {
            nodes: vec![
                NodeConfig {
                    node_id: "node-a".into(),
                    hostname: Some("alpha.local".into()),
                    models: vec![],
                },
                NodeConfig {
                    node_id: "node-b".into(),
                    hostname: Some("beta.local".into()),
                    models: vec![ModelAssignment {
                        name: "Qwen3-8B-Q4_K_M".into(),
                        path: None,
                        ctx_size: Some(4096),
                        moe_experts: None,
                    }],
                },
            ],
        };

        let node = config.for_node("node-b").unwrap();
        assert_eq!(node.node_id, "node-b");
        assert_eq!(node.hostname.as_deref(), Some("beta.local"));
        assert_eq!(node.models.len(), 1);
        assert_eq!(node.models[0].name, "Qwen3-8B-Q4_K_M");
        assert!(config.for_node("missing-node").is_none());
    }

    #[test]
    fn mesh_config_missing_fields_default_cleanly() {
        let parsed: MeshConfig = toml::from_str("[[nodes]]\nnode_id = \"node-a\"\n").unwrap();

        assert_eq!(parsed.nodes.len(), 1);
        assert_eq!(parsed.nodes[0].node_id, "node-a");
        assert!(parsed.nodes[0].hostname.is_none());
        assert!(parsed.nodes[0].models.is_empty());
    }

    #[test]
    fn mesh_config_round_trip() {
        let config = MeshConfig {
            nodes: vec![NodeConfig {
                node_id: "node-a".into(),
                hostname: Some("alpha.local".into()),
                models: vec![
                    ModelAssignment {
                        name: "Qwen3-30B-A3B-Q4_K_M".into(),
                        path: Some("/Users/test/.models/Qwen3-30B-A3B-Q4_K_M.gguf".into()),
                        ctx_size: Some(8192),
                        moe_experts: Some(24),
                    },
                    ModelAssignment {
                        name: "Qwen2.5-Coder-7B-Q4_K_M".into(),
                        path: None,
                        ctx_size: Some(4096),
                        moe_experts: None,
                    },
                ],
            }],
        };

        let raw = toml::to_string_pretty(&config).unwrap();
        let parsed: MeshConfig = toml::from_str(&raw).unwrap();
        assert_eq!(parsed, config);
    }

    // NodeConfig persistence: save/load via node.toml

    #[test]
    fn node_config_round_trip_save_and_load() {
        let dir = temp_dir("node_round_trip");
        let path = dir.join("node.toml");

        let node = NodeConfig {
            node_id: "node-a".into(),
            hostname: Some("alpha.local".into()),
            models: vec![ModelAssignment {
                name: "Qwen3-8B-Q4_K_M".into(),
                path: Some("/models/Qwen3-8B-Q4_K_M.gguf".into()),
                ctx_size: Some(4096),
                moe_experts: None,
            }],
        };

        node.save(&path).expect("save should succeed");
        let loaded = NodeConfig::load(&path)
            .expect("load should succeed")
            .expect("should not be None for existing file");
        assert_eq!(loaded, node);
    }

    #[test]
    fn node_config_load_missing_returns_none() {
        let dir = temp_dir("node_missing");
        let missing = dir.join("missing-node.toml");

        let loaded = NodeConfig::load(&missing).expect("load should not error");
        assert!(loaded.is_none());
    }

    // Path helpers: mesh.toml and node.toml are under ~/.mesh-llm/

    #[test]
    fn mesh_and_node_paths_are_rooted_under_mesh_llm_home() {
        let mesh_path = mesh_config_path(None);
        let node_path = node_config_path();

        assert!(
            mesh_path.ends_with(Path::new(".mesh-llm").join("mesh.toml")),
            "mesh path: {}",
            mesh_path.display()
        );
        assert!(
            node_path.ends_with(Path::new(".mesh-llm").join("node.toml")),
            "node path: {}",
            node_path.display()
        );
    }

    // gpu_index validation (ported from PR #130)

    #[test]
    fn authored_config_rejects_invalid_placement_fields() {
        let invalid_placement_mode =
            "version = 1\n[[nodes]]\nnode_id = \"node-a\"\nplacement_mode = \"invalid\"\n";
        let err = toml::from_str::<AuthoredMeshConfig>(invalid_placement_mode).unwrap_err();
        assert!(err.to_string().contains("placement_mode"));

        let negative_gpu_index = "version = 1\n[[nodes]]\nnode_id = \"node-a\"\n[[nodes.models]]\nname = \"m\"\ngpu_index = -1\n";
        assert!(toml::from_str::<AuthoredMeshConfig>(negative_gpu_index).is_err());

        let float_gpu_index = "version = 1\n[[nodes]]\nnode_id = \"node-a\"\n[[nodes.models]]\nname = \"m\"\ngpu_index = 1.5\n";
        assert!(toml::from_str::<AuthoredMeshConfig>(float_gpu_index).is_err());

        let string_gpu_index = "version = 1\n[[nodes]]\nnode_id = \"node-a\"\n[[nodes.models]]\nname = \"m\"\ngpu_index = \"0\"\n";
        assert!(toml::from_str::<AuthoredMeshConfig>(string_gpu_index).is_err());
    }

    #[test]
    fn mesh_config_rejects_duplicate_node_assignments() {
        let dir = temp_dir("duplicate_node_id");
        let path = dir.join("mesh.toml");
        let toml = "version = 1\n\
            [[nodes]]\nnode_id = \"node-a\"\n\
            [[nodes]]\nnode_id = \"node-a\"\n";
        std::fs::write(&path, toml).unwrap();
        let err = AuthoredMeshConfig::load(&path).unwrap_err();
        assert!(
            err.to_string().contains("duplicate node_id"),
            "unexpected error: {err}"
        );
    }

    // CLI wiring tests (unchanged from Task 4)

    #[test]
    fn cli_mesh_config_flag_parses_override_path() {
        use crate::cli::Cli;
        use clap::Parser;

        let args = vec!["mesh-llm", "--mesh-config", "/tmp/test.toml"];
        let cli = Cli::try_parse_from(args).expect("should parse successfully");
        assert_eq!(cli.mesh_config, Some(PathBuf::from("/tmp/test.toml")));
    }

    #[test]
    fn cli_mesh_config_flag_does_not_conflict_with_plugin_config() {
        use crate::cli::Cli;
        use clap::Parser;

        let args = vec![
            "mesh-llm",
            "--config",
            "/tmp/plugin.toml",
            "--mesh-config",
            "/tmp/mesh.toml",
        ];
        let cli = Cli::try_parse_from(args).expect("should parse successfully");
        assert_eq!(cli.config, Some(PathBuf::from("/tmp/plugin.toml")));
        assert_eq!(cli.mesh_config, Some(PathBuf::from("/tmp/mesh.toml")));
    }

    #[test]
    fn cli_mesh_config_flag_remains_hidden_from_default_help() {
        use crate::cli::Cli;
        use clap::CommandFactory;

        let help_text = Cli::command().render_help().to_string();
        assert!(
            !help_text.contains("--mesh-config"),
            "--mesh-config should not appear in default help output"
        );
    }

    // Hydration (Task 5 tests, updated for new types)

    #[test]
    fn runtime_hydrates_mesh_config_without_model_application() {
        let cfg = AuthoredMeshConfig {
            version: AUTHORED_CONFIG_VERSION,
            nodes: vec![AuthoredNodeConfig {
                node_id: "node-a".into(),
                hostname: None,
                placement_mode: PlacementMode::Pooled,
                models: vec![AuthoredModelAssignment {
                    name: "Qwen2.5-7B".into(),
                    model_key: None,
                    split: None,
                    path: None,
                    ctx_size: Some(4096),
                    moe_experts: None,
                    gpu_index: None,
                }],
            }],
        };
        let state = hydrate_local_mesh_config(cfg, "node-a");
        assert!(state.local.is_some());
        let local = state.local.unwrap();
        assert_eq!(local.models.len(), 1);
        assert_eq!(local.models[0].name, "Qwen2.5-7B");
    }

    #[test]
    fn runtime_hydration_returns_none_for_unknown_node() {
        let cfg = AuthoredMeshConfig {
            version: AUTHORED_CONFIG_VERSION,
            nodes: vec![AuthoredNodeConfig {
                node_id: "node-a".into(),
                hostname: None,
                placement_mode: PlacementMode::Pooled,
                models: vec![],
            }],
        };
        let state = hydrate_local_mesh_config(cfg, "node-z");
        assert!(state.local.is_none());
    }

    #[test]
    fn runtime_hydration_returns_none_for_empty_config() {
        let cfg = AuthoredMeshConfig::default();
        let state = hydrate_local_mesh_config(cfg, "any-node");
        assert!(state.local.is_none());
    }

    // Edge cases

    #[test]
    fn mesh_config_path_env_var_is_respected() {
        let sentinel = "/tmp/mesh-llm-env-test-sentinel.toml";
        let path = mesh_config_path_with_env(None, |key| {
            if key == "MESH_LLM_MESH_CONFIG" {
                Ok(sentinel.to_string())
            } else {
                Err(std::env::VarError::NotPresent)
            }
        });
        assert_eq!(path, PathBuf::from(sentinel));
    }

    #[test]
    fn mesh_config_path_cli_arg_overrides_env_var() {
        let path = mesh_config_path_with_env(Some(PathBuf::from("/tmp/cli.toml")), |_| {
            Ok("/tmp/env.toml".to_string())
        });
        assert_eq!(path, PathBuf::from("/tmp/cli.toml"));
    }

    #[test]
    fn for_node_runtime_with_empty_models_returns_some() {
        let config = AuthoredMeshConfig {
            version: AUTHORED_CONFIG_VERSION,
            nodes: vec![AuthoredNodeConfig {
                node_id: "node-a".into(),
                hostname: Some("alpha.local".into()),
                placement_mode: PlacementMode::Pooled,
                models: vec![],
            }],
        };
        let runtime = config.for_node_runtime("node-a").unwrap();
        assert_eq!(runtime.node_id, "node-a");
        assert!(runtime.models.is_empty());
    }

    #[test]
    fn model_ctx_size_returns_first_match_for_split_model() {
        // When the same model appears twice (split scenario), the first entry wins.
        let config = AuthoredMeshConfig {
            version: AUTHORED_CONFIG_VERSION,
            nodes: vec![AuthoredNodeConfig {
                node_id: "node-a".into(),
                hostname: None,
                placement_mode: PlacementMode::Separate,
                models: vec![
                    AuthoredModelAssignment {
                        name: "Qwen3-30B".into(),
                        model_key: None,
                        split: Some(ModelSplit {
                            start: 0,
                            end: 21,
                            total: 33,
                        }),
                        path: None,
                        ctx_size: Some(8192),
                        moe_experts: None,
                        gpu_index: Some(0),
                    },
                    AuthoredModelAssignment {
                        name: "Qwen3-30B".into(),
                        model_key: None,
                        split: Some(ModelSplit {
                            start: 21,
                            end: 33,
                            total: 33,
                        }),
                        path: None,
                        ctx_size: Some(4096),
                        moe_experts: None,
                        gpu_index: Some(1),
                    },
                ],
            }],
        };
        assert_eq!(config.model_ctx_size("node-a", "Qwen3-30B"), Some(8192));
    }

    #[test]
    fn placement_mode_serializes_as_snake_case_strings() {
        #[derive(Serialize, Deserialize)]
        struct Wrapper {
            mode: PlacementMode,
        }

        let pooled: Wrapper = toml::from_str("mode = \"pooled\"").unwrap();
        assert_eq!(pooled.mode, PlacementMode::Pooled);

        let separate: Wrapper = toml::from_str("mode = \"separate\"").unwrap();
        assert_eq!(separate.mode, PlacementMode::Separate);

        let raw = toml::to_string_pretty(&Wrapper {
            mode: PlacementMode::Separate,
        })
        .unwrap();
        assert!(
            raw.contains("\"separate\""),
            "expected snake_case serialize, got: {raw}"
        );
    }

    #[test]
    fn model_split_zero_length_range_is_permitted_by_schema() {
        // start == end is allowed by the schema; callers are responsible for validation.
        let raw = "version = 1\n[[nodes]]\nnode_id = \"n\"\n[[nodes.models]]\nname = \"m\"\nsplit = { start = 16, end = 16, total = 32 }\n";
        let cfg: AuthoredMeshConfig = toml::from_str(raw).unwrap();
        let split = cfg.nodes[0].models[0].split.as_ref().unwrap();
        assert_eq!(split.start, 16);
        assert_eq!(split.end, 16);
        assert_eq!(split.total, 32);
    }

    #[test]
    fn authored_config_save_normalizes_version_field() {
        let dir = temp_dir("save_normalizes_version");
        let path = dir.join("mesh.toml");

        let cfg = AuthoredMeshConfig {
            version: 99,
            nodes: vec![],
        };
        cfg.save(&path).expect("save should succeed");

        let raw = std::fs::read_to_string(&path).unwrap();
        assert!(
            raw.contains("version = 1"),
            "expected version=1 after normalization, got: {raw}"
        );
        AuthoredMeshConfig::load(&path).expect("saved file must re-parse cleanly as v1");
    }

    #[test]
    fn mesh_config_load_from_file_round_trip() {
        let dir = temp_dir("mesh_config_load_from_file");
        let path = dir.join("mesh.toml");

        let raw = "[[nodes]]\nnode_id = \"node-a\"\nhostname = \"alpha.local\"\n[[nodes.models]]\nname = \"Qwen3-8B\"\nctx_size = 4096\n";
        std::fs::write(&path, raw).unwrap();

        let loaded = MeshConfig::load(&path).expect("load should succeed");
        assert_eq!(loaded.nodes.len(), 1);
        assert_eq!(loaded.nodes[0].node_id, "node-a");
        assert_eq!(loaded.nodes[0].models[0].name, "Qwen3-8B");
        assert_eq!(loaded.nodes[0].models[0].ctx_size, Some(4096));
    }

    #[test]
    fn mesh_config_load_missing_file_returns_empty() {
        let dir = temp_dir("mesh_config_load_missing");
        let path = dir.join("does_not_exist.toml");
        let loaded =
            MeshConfig::load(&path).expect("missing file should return Ok with empty config");
        assert!(loaded.nodes.is_empty());
    }

    #[test]
    fn node_config_save_creates_parent_dirs() {
        let base = temp_dir("node_save_creates_dir");
        let nested = base.join("deep").join("node.toml");

        let node = NodeConfig {
            node_id: "node-a".into(),
            hostname: None,
            models: vec![],
        };
        node.save(&nested)
            .expect("save to nested path should succeed");

        assert!(nested.exists());
        let loaded = NodeConfig::load(&nested).unwrap().unwrap();
        assert_eq!(loaded.node_id, "node-a");
    }
}
