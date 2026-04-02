use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::path::{Path, PathBuf};

pub const AUTHORED_CONFIG_VERSION: u32 = 1;

fn mesh_root_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".mesh-llm")
}

pub fn mesh_config_path() -> PathBuf {
    mesh_root_dir().join("mesh.toml")
}

pub fn node_config_path() -> PathBuf {
    mesh_root_dir().join("node.toml")
}

fn save_toml_atomically<T: Serialize>(value: &T, path: &Path, label: &str) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create config directory {}", parent.display()))?;
    }
    let raw =
        toml::to_string_pretty(value).with_context(|| format!("Failed to serialize {label}"))?;

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
        .with_context(|| format!("Failed to create temp config {}", tmp_path.display()))?;
    tmp_file
        .write_all(raw.as_bytes())
        .with_context(|| format!("Failed to write temp config {}", tmp_path.display()))?;
    tmp_file
        .sync_all()
        .with_context(|| format!("Failed to sync temp config {}", tmp_path.display()))?;
    std::fs::rename(&tmp_path, path).with_context(|| {
        format!(
            "Failed to atomically replace config {} via {}",
            path.display(),
            tmp_path.display()
        )
    })
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct MeshConfig {
    #[serde(default)]
    pub nodes: Vec<NodeConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AuthoredMeshConfig {
    pub version: u32,
    #[serde(default)]
    pub nodes: Vec<AuthoredNodeConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AuthoredNodeConfig {
    pub node_id: String,
    pub hostname: Option<String>,
    #[serde(default)]
    pub placement_mode: PlacementMode,
    #[serde(default)]
    pub models: Vec<AuthoredModelAssignment>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PlacementMode {
    #[default]
    Pooled,
    Separate,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelSplit {
    pub start: u32,
    pub end: u32,
    pub total: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AuthoredModelAssignment {
    pub name: String,
    pub model_key: Option<String>,
    pub split: Option<ModelSplit>,
    pub path: Option<String>,
    pub ctx_size: Option<u32>,
    pub moe_experts: Option<u32>,
    #[serde(default)]
    pub gpu_index: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct NodeConfig {
    pub node_id: String,
    pub hostname: Option<String>,
    #[serde(default)]
    pub models: Vec<ModelAssignment>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelAssignment {
    pub name: String,
    pub path: Option<String>,
    pub ctx_size: Option<u32>,
    pub moe_experts: Option<u32>,
}

impl MeshConfig {
    pub fn load(path: &Path) -> Result<Self> {
        if !path.exists() {
            return Ok(Self::default());
        }
        let raw = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read config {}", path.display()))?;
        toml::from_str(&raw).with_context(|| format!("Failed to parse config {}", path.display()))
    }

    pub fn for_node(&self, node_id: &str) -> Option<NodeConfig> {
        self.nodes
            .iter()
            .find(|node| node.node_id == node_id)
            .cloned()
    }
}

impl Default for AuthoredMeshConfig {
    fn default() -> Self {
        Self {
            version: AUTHORED_CONFIG_VERSION,
            nodes: Vec::new(),
        }
    }
}

impl AuthoredMeshConfig {
    pub fn load(path: &Path) -> Result<Self> {
        if !path.exists() {
            return Ok(Self::default());
        }
        let raw = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read config {}", path.display()))?;
        let mut parsed: Self = toml::from_str(&raw)
            .with_context(|| format!("Failed to parse config {}", path.display()))?;
        match parsed.version {
            AUTHORED_CONFIG_VERSION => Ok(parsed),
            other => bail!("Unsupported authored config version {other}"),
        }
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let mut normalized = self.clone();
        normalized.version = AUTHORED_CONFIG_VERSION;
        save_toml_atomically(&normalized, path, "mesh config")
    }

    pub fn for_node_runtime(&self, node_id: &str) -> Option<NodeConfig> {
        self.nodes
            .iter()
            .find(|node| node.node_id == node_id)
            .map(AuthoredNodeConfig::to_runtime)
    }

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
    pub fn load(path: &Path) -> Result<Option<Self>> {
        if !path.exists() {
            return Ok(None);
        }
        let raw = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read node config {}", path.display()))?;
        let parsed = toml::from_str(&raw)
            .with_context(|| format!("Failed to parse node config {}", path.display()))?;
        Ok(Some(parsed))
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        save_toml_atomically(self, path, "node config")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_round_trip() {
        let config = MeshConfig {
            nodes: vec![
                NodeConfig {
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
                },
                NodeConfig {
                    node_id: "node-b".into(),
                    hostname: None,
                    models: vec![ModelAssignment {
                        name: "GLM-4.7-Flash-Q4_K_M".into(),
                        path: Some("/models/GLM-4.7-Flash-Q4_K_M.gguf".into()),
                        ctx_size: None,
                        moe_experts: Some(16),
                    }],
                },
            ],
        };

        let raw = toml::to_string_pretty(&config).unwrap();
        let parsed: MeshConfig = toml::from_str(&raw).unwrap();
        assert_eq!(parsed, config);
    }

    #[test]
    fn authored_config_round_trip_with_split_and_model_key() {
        let config = AuthoredMeshConfig {
            version: AUTHORED_CONFIG_VERSION,
            nodes: vec![AuthoredNodeConfig {
                node_id: "node-a".into(),
                hostname: Some("alpha.local".into()),
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
                placement_mode: PlacementMode::Separate,
            }],
        };

        let raw = toml::to_string_pretty(&config).unwrap();
        let parsed: AuthoredMeshConfig = toml::from_str(&raw).unwrap();
        assert_eq!(parsed, config);
    }

    #[test]
    fn authored_config_missing_file_defaults_to_versioned_empty() {
        let mut dir = std::env::temp_dir();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        dir.push(format!("mesh-llm-authored-config-missing-{nanos}"));
        std::fs::create_dir_all(&dir).unwrap();
        let missing = dir.join("mesh.toml");

        let loaded = AuthoredMeshConfig::load(&missing).unwrap();
        assert_eq!(loaded.version, AUTHORED_CONFIG_VERSION);
        assert!(loaded.nodes.is_empty());
    }

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
    }

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

        let serialized = toml::to_string_pretty(&parsed).unwrap();
        assert!(serialized.contains("version = 1"));
        assert!(serialized.contains("placement_mode = \"separate\""));
        assert!(serialized.contains("gpu_index = 0"));
        assert!(serialized.contains("gpu_index = 1"));
    }

    #[test]
    fn authored_config_rejects_unsupported_version() {
        let mut dir = std::env::temp_dir();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        dir.push(format!("mesh-llm-authored-config-unsupported-ver-{nanos}"));
        std::fs::create_dir_all(&dir).unwrap();

        let v2_path = dir.join("mesh-v2.toml");
        std::fs::write(
            &v2_path,
            r#"
                version = 2

                [[nodes]]
                node_id = "node-a"

                [[nodes.models]]
                name = "Qwen3-8B-Q4_K_M"
            "#,
        )
        .unwrap();

        let result = AuthoredMeshConfig::load(&v2_path);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Unsupported authored config version 2"));

        let v1_path = dir.join("mesh-v1.toml");
        let valid = AuthoredMeshConfig {
            version: AUTHORED_CONFIG_VERSION,
            nodes: vec![],
        };
        valid.save(&v1_path).unwrap();
        let saved = std::fs::read_to_string(&v1_path).unwrap();
        assert!(saved.contains("version = 1"));
    }

    #[test]
    fn authored_config_rejects_invalid_placement_fields() {
        let invalid_placement_mode = r#"
            version = 1

            [[nodes]]
            node_id = "node-a"
            placement_mode = "invalid"
        "#;
        let placement_err =
            toml::from_str::<AuthoredMeshConfig>(invalid_placement_mode).unwrap_err();
        assert!(placement_err.to_string().contains("placement_mode"));

        let invalid_gpu_index_negative = r#"
            version = 1

            [[nodes]]
            node_id = "node-a"
            placement_mode = "separate"

            [[nodes.models]]
            name = "Qwen3-8B-Q4_K_M"
            gpu_index = -1
        "#;
        assert!(toml::from_str::<AuthoredMeshConfig>(invalid_gpu_index_negative).is_err());

        let invalid_gpu_index_non_integer = r#"
            version = 1

            [[nodes]]
            node_id = "node-a"
            placement_mode = "separate"

            [[nodes.models]]
            name = "Qwen3-8B-Q4_K_M"
            gpu_index = 1.5
        "#;
        assert!(toml::from_str::<AuthoredMeshConfig>(invalid_gpu_index_non_integer).is_err());

        let invalid_gpu_index_non_numeric = r#"
            version = 1

            [[nodes]]
            node_id = "node-a"
            placement_mode = "separate"

            [[nodes.models]]
            name = "Qwen3-8B-Q4_K_M"
            gpu_index = "0"
        "#;
        assert!(toml::from_str::<AuthoredMeshConfig>(invalid_gpu_index_non_numeric).is_err());
    }

    #[test]
    fn config_for_node_extracts_specific_node() {
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
    fn config_missing_fields_default_cleanly() {
        let parsed: MeshConfig = toml::from_str(
            r#"
            [[nodes]]
            node_id = "node-a"
            "#,
        )
        .unwrap();

        assert_eq!(parsed.nodes.len(), 1);
        assert_eq!(parsed.nodes[0].node_id, "node-a");
        assert!(parsed.nodes[0].hostname.is_none());
        assert!(parsed.nodes[0].models.is_empty());
    }

    #[test]
    fn mesh_and_node_paths_are_rooted_under_mesh_llm_home() {
        let mesh_path = mesh_config_path();
        let node_path = node_config_path();

        assert!(mesh_path.ends_with(".mesh-llm/mesh.toml"));
        assert!(node_path.ends_with(".mesh-llm/node.toml"));
    }

    #[test]
    fn node_config_round_trip_save_and_load() {
        let mut dir = std::env::temp_dir();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        dir.push(format!("mesh-llm-node-config-test-{nanos}"));
        std::fs::create_dir_all(&dir).unwrap();
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

        node.save(&path).unwrap();
        let loaded = NodeConfig::load(&path).unwrap().unwrap();

        assert_eq!(loaded, node);
    }

    #[test]
    fn node_config_load_missing_returns_none() {
        let mut dir = std::env::temp_dir();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        dir.push(format!("mesh-llm-node-config-missing-{nanos}"));
        std::fs::create_dir_all(&dir).unwrap();
        let missing = dir.join("missing-node.toml");

        let loaded = NodeConfig::load(&missing).unwrap();
        assert!(loaded.is_none());
    }
}
