use super::{PluginSummary, BLACKBOARD_PLUGIN_ID, BLOBSTORE_PLUGIN_ID, LEMONADE_PLUGIN_ID};
use anyhow::{bail, Context, Result};
use mesh_llm_plugin::MeshVisibility;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct MeshConfig {
    #[serde(default)]
    pub version: Option<u32>,
    #[serde(default)]
    pub gpu: GpuConfig,
    #[serde(default)]
    pub models: Vec<ModelConfigEntry>,
    #[serde(rename = "plugin", default)]
    pub plugins: Vec<PluginConfigEntry>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct GpuConfig {
    #[serde(default)]
    pub assignment: GpuAssignment,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum GpuAssignment {
    #[default]
    Auto,
    Pinned,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ModelConfigEntry {
    pub model: String,
    #[serde(default)]
    pub mmproj: Option<String>,
    #[serde(default)]
    pub ctx_size: Option<u32>,
    #[serde(default)]
    pub gpu_id: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct PluginConfigEntry {
    pub name: String,
    #[serde(default)]
    pub enabled: Option<bool>,
    #[serde(default)]
    pub command: Option<String>,
    #[serde(default)]
    pub args: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct ResolvedPlugins {
    pub externals: Vec<ExternalPluginSpec>,
    pub inactive: Vec<PluginSummary>,
}

#[derive(Clone, Debug)]
pub struct ExternalPluginSpec {
    pub name: String,
    pub command: String,
    pub args: Vec<String>,
}

#[derive(Clone, Copy, Debug)]
pub struct PluginHostMode {
    pub mesh_visibility: MeshVisibility,
}

pub fn config_path(override_path: Option<&Path>) -> Result<PathBuf> {
    if let Some(path) = override_path {
        return Ok(path.to_path_buf());
    }
    if let Ok(path) = std::env::var("MESH_LLM_CONFIG") {
        return Ok(PathBuf::from(path));
    }
    let home = dirs::home_dir().context("Cannot determine home directory")?;
    Ok(home.join(".mesh-llm").join("config.toml"))
}

pub fn load_config(override_path: Option<&Path>) -> Result<MeshConfig> {
    let path = config_path(override_path)?;
    if !path.exists() {
        return Ok(MeshConfig::default());
    }
    let raw = std::fs::read_to_string(&path)
        .with_context(|| format!("Failed to read config {}", path.display()))?;
    let config: MeshConfig = toml::from_str(&raw)
        .with_context(|| format!("Failed to parse config {}", path.display()))?;
    validate_config(&config).with_context(|| format!("Invalid config {}", path.display()))?;
    Ok(config)
}

pub(crate) fn validate_config(config: &MeshConfig) -> Result<()> {
    if let Some(version) = config.version {
        if version != 1 {
            bail!("unsupported config version {version}; expected version = 1");
        }
    }
    for (index, model) in config.models.iter().enumerate() {
        if model.model.trim().is_empty() {
            bail!("models[{index}].model must not be empty");
        }
        if let Some(mmproj) = &model.mmproj {
            if mmproj.trim().is_empty() {
                bail!("models[{index}].mmproj must not be empty when set");
            }
        }
        match config.gpu.assignment {
            GpuAssignment::Auto => {
                if model.gpu_id.is_some() {
                    bail!("models[{index}].gpu_id must not be set when gpu.assignment = \"auto\"");
                }
            }
            GpuAssignment::Pinned => match &model.gpu_id {
                Some(gpu_id) if !gpu_id.trim().is_empty() => {}
                _ => {
                    bail!(
                        "models[{index}].gpu_id must be set to a non-empty value when gpu.assignment = \"pinned\""
                    );
                }
            },
        }
    }
    Ok(())
}

pub fn resolve_plugins(config: &MeshConfig, _host_mode: PluginHostMode) -> Result<ResolvedPlugins> {
    let mut externals = Vec::new();
    let inactive = Vec::new();
    let mut names = BTreeMap::<String, ()>::new();
    let mut blackboard_enabled = true;
    let mut blobstore_enabled = true;
    let mut lemonade_enabled = false;
    for entry in &config.plugins {
        if names.insert(entry.name.clone(), ()).is_some() {
            bail!("Duplicate plugin entry '{}'", entry.name);
        }
        let enabled = entry.enabled.unwrap_or(true);
        if entry.name == BLACKBOARD_PLUGIN_ID {
            if entry.command.is_some() || !entry.args.is_empty() {
                bail!(
                    "Plugin '{}' is served by mesh-llm itself; only `enabled` may be set",
                    BLACKBOARD_PLUGIN_ID
                );
            }
            blackboard_enabled = enabled;
            continue;
        }
        if entry.name == BLOBSTORE_PLUGIN_ID {
            if entry.command.is_some() || !entry.args.is_empty() {
                bail!(
                    "Plugin '{}' is served by mesh-llm itself; only `enabled` may be set",
                    BLOBSTORE_PLUGIN_ID
                );
            }
            blobstore_enabled = enabled;
            continue;
        }
        if entry.name == LEMONADE_PLUGIN_ID {
            if entry.command.is_some() || !entry.args.is_empty() {
                bail!(
                    "Plugin '{}' is served by mesh-llm itself; only `enabled` may be set",
                    LEMONADE_PLUGIN_ID
                );
            }
            lemonade_enabled = enabled;
            continue;
        }
        if !enabled {
            continue;
        }
        let command = entry
            .command
            .clone()
            .with_context(|| format!("Plugin '{}' is enabled but missing command", entry.name))?;
        externals.push(ExternalPluginSpec {
            name: entry.name.clone(),
            command,
            args: entry.args.clone(),
        });
    }

    if blackboard_enabled {
        externals.insert(0, blackboard_plugin_spec()?);
    }
    if lemonade_enabled {
        externals.push(lemonade_plugin_spec()?);
    }
    if blobstore_enabled {
        externals.push(blobstore_plugin_spec()?);
    }

    Ok(ResolvedPlugins {
        externals,
        inactive,
    })
}

pub fn blackboard_plugin_spec() -> Result<ExternalPluginSpec> {
    let command = std::env::current_exe()
        .context("Cannot determine mesh-llm executable path")?
        .display()
        .to_string();
    Ok(ExternalPluginSpec {
        name: BLACKBOARD_PLUGIN_ID.to_string(),
        command,
        args: vec!["--plugin".into(), BLACKBOARD_PLUGIN_ID.into()],
    })
}

pub fn blobstore_plugin_spec() -> Result<ExternalPluginSpec> {
    let command = std::env::current_exe()
        .context("Cannot determine mesh-llm executable path")?
        .display()
        .to_string();
    Ok(ExternalPluginSpec {
        name: BLOBSTORE_PLUGIN_ID.to_string(),
        command,
        args: vec!["--plugin".into(), BLOBSTORE_PLUGIN_ID.into()],
    })
}

pub fn lemonade_plugin_spec() -> Result<ExternalPluginSpec> {
    let command = std::env::current_exe()
        .context("Cannot determine mesh-llm executable path")?
        .display()
        .to_string();
    Ok(ExternalPluginSpec {
        name: LEMONADE_PLUGIN_ID.to_string(),
        command,
        args: vec!["--plugin".into(), LEMONADE_PLUGIN_ID.into()],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_unified_config_keeps_plugins_and_models() {
        let config: MeshConfig = toml::from_str(
            r#"
version = 1

[gpu]
assignment = "auto"

[[models]]
model = "Qwen3-8B-Q4_K_M"
ctx_size = 8192

[[models]]
model = "bartowski/Qwen2.5-VL-7B-Instruct-GGUF/model.gguf"
mmproj = "bartowski/Qwen2.5-VL-7B-Instruct-GGUF/mmproj.gguf"

[[plugin]]
name = "demo"
command = "/tmp/demo"
"#,
        )
        .unwrap();

        assert_eq!(config.version, Some(1));
        assert_eq!(config.gpu.assignment, GpuAssignment::Auto);
        assert_eq!(config.models.len(), 2);
        assert_eq!(config.models[0].model, "Qwen3-8B-Q4_K_M");
        assert_eq!(config.models[0].ctx_size, Some(8192));
        assert_eq!(config.models[0].gpu_id, None);
        assert_eq!(
            config.models[1].mmproj.as_deref(),
            Some("bartowski/Qwen2.5-VL-7B-Instruct-GGUF/mmproj.gguf")
        );
        assert_eq!(config.models[1].gpu_id, None);
        assert_eq!(config.plugins.len(), 1);
        assert_eq!(config.plugins[0].name, "demo");
    }

    #[test]
    fn pinned_gpu_config_accepted_pinned_config() {
        let config: MeshConfig = toml::from_str(
            r#"
version = 1

[gpu]
assignment = "pinned"

[[models]]
model = "Qwen3-8B-Q4_K_M"
gpu_id = "pci:0000:65:00.0"
ctx_size = 8192
"#,
        )
        .unwrap();

        validate_config(&config).unwrap();
        assert_eq!(config.models[0].gpu_id.as_deref(), Some("pci:0000:65:00.0"));
    }

    #[test]
    fn pinned_gpu_config_missing_gpu_id_rejected() {
        let config = MeshConfig {
            gpu: GpuConfig {
                assignment: GpuAssignment::Pinned,
            },
            models: vec![ModelConfigEntry {
                model: "Qwen3-8B-Q4_K_M".into(),
                mmproj: None,
                ctx_size: None,
                gpu_id: None,
            }],
            ..MeshConfig::default()
        };

        let err = validate_config(&config).unwrap_err();
        assert!(err.to_string().contains(
            "models[0].gpu_id must be set to a non-empty value when gpu.assignment = \"pinned\""
        ));
    }

    #[test]
    fn pinned_gpu_config_empty_gpu_id_rejected() {
        let config = MeshConfig {
            gpu: GpuConfig {
                assignment: GpuAssignment::Pinned,
            },
            models: vec![ModelConfigEntry {
                model: "Qwen3-8B-Q4_K_M".into(),
                mmproj: None,
                ctx_size: None,
                gpu_id: Some("  \t  ".into()),
            }],
            ..MeshConfig::default()
        };

        let err = validate_config(&config).unwrap_err();
        assert!(err.to_string().contains(
            "models[0].gpu_id must be set to a non-empty value when gpu.assignment = \"pinned\""
        ));
    }

    #[test]
    fn pinned_gpu_config_auto_assignment_rejects_gpu_id() {
        let config = MeshConfig {
            gpu: GpuConfig {
                assignment: GpuAssignment::Auto,
            },
            models: vec![ModelConfigEntry {
                model: "Qwen3-8B-Q4_K_M".into(),
                mmproj: None,
                ctx_size: None,
                gpu_id: Some("pci:0000:65:00.0".into()),
            }],
            ..MeshConfig::default()
        };

        let err = validate_config(&config).unwrap_err();
        assert!(err
            .to_string()
            .contains("models[0].gpu_id must not be set when gpu.assignment = \"auto\""));
    }

    #[test]
    fn pinned_gpu_config_preserves_accepted_gpu_id_string_exactly() {
        let raw = r#"
version = 1

[gpu]
assignment = "pinned"

[[models]]
model = "Qwen3-8B-Q4_K_M"
gpu_id = "  pci:0000:65:00.0  "
"#;

        let config: MeshConfig = toml::from_str(raw).unwrap();
        validate_config(&config).unwrap();

        assert_eq!(
            config.models[0].gpu_id.as_deref(),
            Some("  pci:0000:65:00.0  ")
        );
    }
}
