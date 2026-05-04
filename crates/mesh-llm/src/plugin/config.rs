use super::{PluginSummary, BLACKBOARD_PLUGIN_ID, BLOBSTORE_PLUGIN_ID, OPENAI_ENDPOINT_PLUGIN_ID};
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
    #[serde(default)]
    pub moe: MoeConfig,
}

/// Mesh-LLM MoE tuning block. Today it only configures trunk/experts storage;
/// extend here when we grow more MoE-specific knobs (routing strategy, etc.).
#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct MoeConfig {
    #[serde(default)]
    pub storage: MoeStorageConfig,
}

/// How per-node MoE GGUFs are physically laid out on disk.
///
/// * `Monolithic` (default): each node holds a self-contained GGUF with trunk
///   + its expert subset — today's behavior. Safe default so existing
///   deployments keep working with zero config change.
/// * `Split`: one shared trunk GGUF on `trunk_path` (typically fast NAS) plus
///   per-node expert shard GGUFs on `experts_path`. Eliminates trunk
///   duplication across the mesh; pairs with llama.cpp's
///   `--model-trunk`/`--model-experts` load path.
#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum MoeStorageMode {
    #[default]
    Monolithic,
    Split,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct MoeStorageConfig {
    #[serde(default)]
    pub mode: MoeStorageMode,
    /// Root directory for the shared trunk GGUFs. Required when
    /// `mode = "split"`. Must be an absolute path; typically a fast NAS
    /// mounted identically on every mesh node.
    #[serde(default)]
    pub trunk_path: Option<PathBuf>,
    /// Root directory for per-node expert shard GGUFs. Required when
    /// `mode = "split"`. May equal `trunk_path` (single NAS mount), or point
    /// at a per-node fast local SSD for hot-path latency.
    #[serde(default)]
    pub experts_path: Option<PathBuf>,
    /// Optional per-node local override: if set, this node's experts shard is
    /// resolved under `experts_local_override/...` instead of
    /// `experts_path/...`. Trunk always comes from `trunk_path`. Useful when
    /// the mesh has heterogeneous storage (some nodes local SSD, some NAS).
    #[serde(default)]
    pub experts_local_override: Option<PathBuf>,
    /// Walk every page of the trunk mmap at load so the first inference
    /// token does not pay a NAS page-fault cost. Default on (cheap when the
    /// file is already in RAM; meaningful when it isn't). Ignored in
    /// monolithic mode.
    #[serde(default = "default_true")]
    pub prefault_trunk: bool,
    /// `mlock()` just the trunk mmap region. Off by default because it
    /// requires `RLIMIT_MEMLOCK >= trunk_size`; mesh-llm validates the rlimit
    /// Rust-side before launching llama-server so users get a clear error
    /// instead of a silent mlock failure.
    #[serde(default)]
    pub mlock_trunk: bool,
}

impl Default for MoeStorageConfig {
    fn default() -> Self {
        Self {
            mode: MoeStorageMode::default(),
            trunk_path: None,
            experts_path: None,
            experts_local_override: None,
            prefault_trunk: true,
            mlock_trunk: false,
        }
    }
}

fn default_true() -> bool {
    true
}

/// Resolve the trunk root honoring the `MESH_LLM_NAS_ROOT` env fallback
/// (derived subdir `<root>/mesh-llm/trunks`). Returns `None` only when
/// neither the config nor the env is set.
pub fn resolve_trunk_root(cfg: &MoeStorageConfig) -> Option<PathBuf> {
    if let Some(p) = &cfg.trunk_path {
        return Some(p.clone());
    }
    if let Ok(root) = std::env::var("MESH_LLM_NAS_ROOT") {
        return Some(PathBuf::from(root).join("mesh-llm").join("trunks"));
    }
    None
}

/// Resolve the experts root honoring the per-node local override and then
/// `MESH_LLM_NAS_ROOT` (derived subdir `<root>/mesh-llm/experts`).
pub fn resolve_experts_root(cfg: &MoeStorageConfig) -> Option<PathBuf> {
    if let Some(p) = &cfg.experts_local_override {
        return Some(p.clone());
    }
    if let Some(p) = &cfg.experts_path {
        return Some(p.clone());
    }
    if let Ok(root) = std::env::var("MESH_LLM_NAS_ROOT") {
        return Some(PathBuf::from(root).join("mesh-llm").join("experts"));
    }
    None
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct GpuConfig {
    #[serde(default)]
    pub assignment: GpuAssignment,
    #[serde(default)]
    pub parallel: Option<usize>,
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
    #[serde(default)]
    pub parallel: Option<usize>,
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
    /// Base URL for inference endpoint plugins (e.g. http://localhost:8000/v1).
    #[serde(default)]
    pub url: Option<String>,
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
    /// Backend URL for inference endpoint plugins.
    pub url: Option<String>,
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
    if let Some(parallel) = config.gpu.parallel {
        if parallel < 1 {
            bail!("gpu.parallel must be at least 1, got {parallel}");
        }
    }
    validate_moe_storage(&config.moe.storage)?;
    for (index, model) in config.models.iter().enumerate() {
        if model.model.trim().is_empty() {
            bail!("models[{index}].model must not be empty");
        }
        if let Some(mmproj) = &model.mmproj {
            if mmproj.trim().is_empty() {
                bail!("models[{index}].mmproj must not be empty when set");
            }
        }
        if let Some(parallel) = model.parallel {
            if parallel < 1 {
                bail!("models[{index}].parallel must be at least 1, got {parallel}");
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

fn validate_moe_storage(cfg: &MoeStorageConfig) -> Result<()> {
    // Paths must be absolute in both modes — monolithic ignores them at runtime
    // but a relative path is almost always a typo that would fail silently
    // the day someone flips mode to split.
    for (field, path) in [
        ("trunk_path", cfg.trunk_path.as_ref()),
        ("experts_path", cfg.experts_path.as_ref()),
        ("experts_local_override", cfg.experts_local_override.as_ref()),
    ] {
        if let Some(p) = path {
            if !p.is_absolute() {
                bail!(
                    "moe.storage.{field} = {} must be an absolute path",
                    p.display()
                );
            }
        }
    }
    if matches!(cfg.mode, MoeStorageMode::Split) {
        if resolve_trunk_root(cfg).is_none() {
            bail!(
                "moe.storage.mode = \"split\" requires moe.storage.trunk_path \
                 (or MESH_LLM_NAS_ROOT env)"
            );
        }
        if resolve_experts_root(cfg).is_none() {
            bail!(
                "moe.storage.mode = \"split\" requires moe.storage.experts_path \
                 (or MESH_LLM_NAS_ROOT env)"
            );
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
    let mut openai_endpoint_enabled = false;
    let mut openai_endpoint_url: Option<String> = None;
    for entry in &config.plugins {
        if names.insert(entry.name.clone(), ()).is_some() {
            bail!("Duplicate plugin entry '{}'", entry.name);
        }
        let enabled = entry.enabled.unwrap_or(true);
        if entry.name == BLACKBOARD_PLUGIN_ID {
            if entry.command.is_some() || !entry.args.is_empty() || entry.url.is_some() {
                bail!(
                    "Plugin '{}' is served by mesh-llm itself; only `enabled` may be set",
                    BLACKBOARD_PLUGIN_ID
                );
            }
            blackboard_enabled = enabled;
            continue;
        }
        if entry.name == BLOBSTORE_PLUGIN_ID {
            if entry.command.is_some() || !entry.args.is_empty() || entry.url.is_some() {
                bail!(
                    "Plugin '{}' is served by mesh-llm itself; only `enabled` may be set",
                    BLOBSTORE_PLUGIN_ID
                );
            }
            blobstore_enabled = enabled;
            continue;
        }
        if entry.name == OPENAI_ENDPOINT_PLUGIN_ID {
            if entry.command.is_some() || !entry.args.is_empty() {
                bail!(
                    "Plugin '{}' is served by mesh-llm itself; only `enabled` and `url` may be set",
                    OPENAI_ENDPOINT_PLUGIN_ID
                );
            }
            openai_endpoint_enabled = enabled;
            if let Some(ref url) = entry.url {
                openai_endpoint_url = Some(url.clone());
            }
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
            url: None,
        });
    }

    if blackboard_enabled {
        externals.insert(0, blackboard_plugin_spec()?);
    }
    if openai_endpoint_enabled {
        let mut spec = openai_endpoint_plugin_spec()?;
        spec.url = openai_endpoint_url;
        externals.push(spec);
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
        args: vec![
            "--log-format".into(),
            "json".into(),
            "--plugin".into(),
            BLACKBOARD_PLUGIN_ID.into(),
        ],
        url: None,
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
        args: vec![
            "--log-format".into(),
            "json".into(),
            "--plugin".into(),
            BLOBSTORE_PLUGIN_ID.into(),
        ],
        url: None,
    })
}

pub fn openai_endpoint_plugin_spec() -> Result<ExternalPluginSpec> {
    let command = std::env::current_exe()
        .context("Cannot determine mesh-llm executable path")?
        .display()
        .to_string();
    Ok(ExternalPluginSpec {
        name: OPENAI_ENDPOINT_PLUGIN_ID.to_string(),
        command,
        args: vec![
            "--log-format".into(),
            "json".into(),
            "--plugin".into(),
            OPENAI_ENDPOINT_PLUGIN_ID.into(),
        ],
        url: None,
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
                parallel: None,
            },
            models: vec![ModelConfigEntry {
                model: "Qwen3-8B-Q4_K_M".into(),
                mmproj: None,
                ctx_size: None,
                gpu_id: None,
                parallel: None,
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
                parallel: None,
            },
            models: vec![ModelConfigEntry {
                model: "Qwen3-8B-Q4_K_M".into(),
                mmproj: None,
                ctx_size: None,
                gpu_id: Some("  \t  ".into()),
                parallel: None,
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
                parallel: None,
            },
            models: vec![ModelConfigEntry {
                model: "Qwen3-8B-Q4_K_M".into(),
                mmproj: None,
                ctx_size: None,
                gpu_id: Some("pci:0000:65:00.0".into()),
                parallel: None,
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
gpu_id = " pci:0000:65:00.0 "
"#;

        let config: MeshConfig = toml::from_str(raw).unwrap();
        validate_config(&config).unwrap();

        assert_eq!(
            config.models[0].gpu_id.as_deref(),
            Some(" pci:0000:65:00.0 ")
        );
    }

    // ── gpu.parallel validation ──

    #[test]
    fn gpu_parallel_field_deserializes_from_toml() {
        let config: MeshConfig = toml::from_str(
            r#"
version = 1

[gpu]
assignment = "auto"
parallel = 8

[[models]]
model = "Qwen3-8B-Q4_K_M"
"#,
        )
        .unwrap();

        assert_eq!(config.gpu.parallel, Some(8));
    }

    #[test]
    fn gpu_parallel_defaults_to_none_when_omitted() {
        let config: MeshConfig = toml::from_str(
            r#"
version = 1

[gpu]
assignment = "auto"

[[models]]
model = "Qwen3-8B-Q4_K_M"
"#,
        )
        .unwrap();

        assert_eq!(config.gpu.parallel, None);
    }

    #[test]
    fn gpu_parallel_zero_rejected() {
        let config = MeshConfig {
            gpu: GpuConfig {
                assignment: GpuAssignment::Auto,
                parallel: Some(0),
            },
            models: vec![ModelConfigEntry {
                model: "Qwen3-8B-Q4_K_M".into(),
                mmproj: None,
                ctx_size: None,
                gpu_id: None,
                parallel: None,
            }],
            ..MeshConfig::default()
        };

        let err = validate_config(&config).unwrap_err();
        assert!(
            err.to_string()
                .contains("gpu.parallel must be at least 1, got 0"),
            "unexpected error message: {err}"
        );
    }

    #[test]
    fn gpu_parallel_one_accepted() {
        let config = MeshConfig {
            gpu: GpuConfig {
                assignment: GpuAssignment::Auto,
                parallel: Some(1),
            },
            models: vec![ModelConfigEntry {
                model: "Qwen3-8B-Q4_K_M".into(),
                mmproj: None,
                ctx_size: None,
                gpu_id: None,
                parallel: None,
            }],
            ..MeshConfig::default()
        };

        validate_config(&config).unwrap();
    }

    #[test]
    fn gpu_parallel_none_accepted() {
        let config = MeshConfig {
            gpu: GpuConfig {
                assignment: GpuAssignment::Auto,
                parallel: None,
            },
            models: vec![ModelConfigEntry {
                model: "Qwen3-8B-Q4_K_M".into(),
                mmproj: None,
                ctx_size: None,
                gpu_id: None,
                parallel: None,
            }],
            ..MeshConfig::default()
        };

        validate_config(&config).unwrap();
    }

    #[test]
    fn gpu_parallel_large_value_accepted() {
        let config = MeshConfig {
            gpu: GpuConfig {
                assignment: GpuAssignment::Auto,
                parallel: Some(64),
            },
            models: vec![ModelConfigEntry {
                model: "Qwen3-8B-Q4_K_M".into(),
                mmproj: None,
                ctx_size: None,
                gpu_id: None,
                parallel: None,
            }],
            ..MeshConfig::default()
        };

        validate_config(&config).unwrap();
    }

    #[test]
    fn gpu_parallel_unwrap_or_default_is_4() {
        fn parsed_parallel(value: Option<usize>) -> usize {
            value.unwrap_or(4)
        }

        assert_eq!(parsed_parallel(None), 4);
        assert_eq!(parsed_parallel(Some(1)), 1);
        assert_eq!(parsed_parallel(Some(8)), 8);
        assert_eq!(parsed_parallel(Some(64)), 64);
    }

    #[test]
    fn per_model_parallel_valid_value_accepted() {
        let config = MeshConfig {
            models: vec![ModelConfigEntry {
                model: "Qwen3-8B-Q4_K_M".into(),
                mmproj: None,
                ctx_size: None,
                gpu_id: None,
                parallel: Some(8),
            }],
            ..MeshConfig::default()
        };
        validate_config(&config).unwrap();
    }

    #[test]
    fn per_model_parallel_zero_rejected() {
        let config = MeshConfig {
            models: vec![ModelConfigEntry {
                model: "Qwen3-8B-Q4_K_M".into(),
                mmproj: None,
                ctx_size: None,
                gpu_id: None,
                parallel: Some(0),
            }],
            ..MeshConfig::default()
        };
        let err = validate_config(&config).unwrap_err();
        assert!(
            err.to_string()
                .contains("models[0].parallel must be at least 1"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn per_model_parallel_none_accepted() {
        let config = MeshConfig {
            models: vec![ModelConfigEntry {
                model: "Qwen3-8B-Q4_K_M".into(),
                mmproj: None,
                ctx_size: None,
                gpu_id: None,
                parallel: None,
            }],
            ..MeshConfig::default()
        };
        validate_config(&config).unwrap();
    }

    // ── MoE trunk/experts split storage validation ──

    use serial_test::serial;

    // Guard against leaking MESH_LLM_NAS_ROOT across tests that race in
    // parallel test threads. Tests touching this env must also carry
    // `#[serial]` so they don't stomp on each other's setup.
    fn with_env_nas_root<T>(value: Option<&str>, body: impl FnOnce() -> T) -> T {
        let prev = std::env::var("MESH_LLM_NAS_ROOT").ok();
        match value {
            Some(v) => std::env::set_var("MESH_LLM_NAS_ROOT", v),
            None => std::env::remove_var("MESH_LLM_NAS_ROOT"),
        }
        let out = body();
        match prev {
            Some(v) => std::env::set_var("MESH_LLM_NAS_ROOT", v),
            None => std::env::remove_var("MESH_LLM_NAS_ROOT"),
        }
        out
    }

    #[test]
    #[serial(mesh_llm_nas_root)]
    fn moe_default_is_monolithic_and_empty_configs_still_parse() {
        // Existing configs must keep working unchanged — no [moe] section.
        let config: MeshConfig = toml::from_str(
            r#"
version = 1
[[models]]
model = "Qwen3-30B-A3B-Instruct-2507"
ctx_size = 4096
"#,
        )
        .unwrap();

        assert_eq!(config.moe.storage.mode, MoeStorageMode::Monolithic);
        assert!(config.moe.storage.trunk_path.is_none());
        assert!(config.moe.storage.experts_path.is_none());
        assert!(config.moe.storage.prefault_trunk); // default-on
        assert!(!config.moe.storage.mlock_trunk);
        with_env_nas_root(None, || validate_config(&config).unwrap());
    }

    #[test]
    #[serial(mesh_llm_nas_root)]
    fn moe_split_without_paths_or_env_is_rejected() {
        let config: MeshConfig = toml::from_str(
            r#"
version = 1
[moe.storage]
mode = "split"
"#,
        )
        .unwrap();

        let err = with_env_nas_root(None, || validate_config(&config).unwrap_err());
        let msg = err.to_string();
        assert!(
            msg.contains("trunk_path") || msg.contains("experts_path"),
            "unexpected error: {msg}"
        );
    }

    #[test]
    #[serial(mesh_llm_nas_root)]
    fn moe_split_with_absolute_paths_is_accepted() {
        let config: MeshConfig = toml::from_str(
            r#"
version = 1
[moe.storage]
mode = "split"
trunk_path = "/nas/mesh-llm/trunks"
experts_path = "/nas/mesh-llm/experts"
prefault_trunk = true
mlock_trunk = false
"#,
        )
        .unwrap();

        with_env_nas_root(None, || validate_config(&config).unwrap());
        assert_eq!(config.moe.storage.mode, MoeStorageMode::Split);
        assert_eq!(
            config.moe.storage.trunk_path.as_deref(),
            Some(Path::new("/nas/mesh-llm/trunks"))
        );
        assert!(config.moe.storage.prefault_trunk);
    }

    #[test]
    #[serial(mesh_llm_nas_root)]
    fn moe_split_rejects_relative_paths() {
        let config: MeshConfig = toml::from_str(
            r#"
version = 1
[moe.storage]
mode = "split"
trunk_path = "relative/trunks"
experts_path = "/abs/experts"
"#,
        )
        .unwrap();

        let err = with_env_nas_root(None, || validate_config(&config).unwrap_err());
        assert!(
            err.to_string().contains("must be an absolute path"),
            "unexpected error: {err}"
        );
    }

    #[test]
    #[serial(mesh_llm_nas_root)]
    fn moe_split_falls_back_to_env_nas_root() {
        let config: MeshConfig = toml::from_str(
            r#"
version = 1
[moe.storage]
mode = "split"
"#,
        )
        .unwrap();

        with_env_nas_root(Some("/nas-root"), || {
            validate_config(&config).unwrap();
            assert_eq!(
                resolve_trunk_root(&config.moe.storage).unwrap(),
                Path::new("/nas-root/mesh-llm/trunks")
            );
            assert_eq!(
                resolve_experts_root(&config.moe.storage).unwrap(),
                Path::new("/nas-root/mesh-llm/experts")
            );
        });
    }

    #[test]
    #[serial(mesh_llm_nas_root)]
    fn moe_monolithic_rejects_relative_trunk_path_even_though_it_is_unused() {
        // Reject typos that would silently do the wrong thing if the user
        // later flips mode to split without revisiting paths.
        let config: MeshConfig = toml::from_str(
            r#"
version = 1
[moe.storage]
trunk_path = "relative/trunks"
"#,
        )
        .unwrap();

        let err = with_env_nas_root(None, || validate_config(&config).unwrap_err());
        assert!(err.to_string().contains("must be an absolute path"));
    }

    #[test]
    #[serial(mesh_llm_nas_root)]
    fn moe_experts_local_override_wins_over_experts_path() {
        let cfg = MoeStorageConfig {
            mode: MoeStorageMode::Split,
            trunk_path: Some(PathBuf::from("/nas/trunks")),
            experts_path: Some(PathBuf::from("/nas/experts")),
            experts_local_override: Some(PathBuf::from("/fast-local/experts")),
            prefault_trunk: true,
            mlock_trunk: false,
        };
        with_env_nas_root(None, || {
            assert_eq!(
                resolve_experts_root(&cfg).unwrap(),
                Path::new("/fast-local/experts")
            );
        });
    }
}
