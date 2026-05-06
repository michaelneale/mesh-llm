use super::{
    PluginSummary, BLACKBOARD_PLUGIN_ID, BLOBSTORE_PLUGIN_ID, OPENAI_ENDPOINT_PLUGIN_ID,
    SURVEY_PLUGIN_ID,
};
use anyhow::{bail, Context, Result};
use mesh_llm_plugin::MeshVisibility;
use serde::{Deserialize, Serialize};
use skippy_protocol::FlashAttentionType;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct MeshConfig {
    #[serde(default)]
    pub version: Option<u32>,
    #[serde(default)]
    pub gpu: GpuConfig,
    #[serde(default)]
    pub telemetry: TelemetryConfig,
    #[serde(default)]
    pub models: Vec<ModelConfigEntry>,
    #[serde(rename = "plugin", default)]
    pub plugins: Vec<PluginConfigEntry>,
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
    #[serde(default)]
    pub cache_type_k: Option<String>,
    #[serde(default)]
    pub cache_type_v: Option<String>,
    #[serde(default)]
    pub batch: Option<u32>,
    #[serde(default)]
    pub ubatch: Option<u32>,
    #[serde(default)]
    pub flash_attention: Option<FlashAttentionType>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct TelemetryConfig {
    #[serde(default)]
    pub enabled: Option<bool>,
    #[serde(default)]
    pub service_name: Option<String>,
    #[serde(default)]
    pub endpoint: Option<String>,
    #[serde(default)]
    pub headers: BTreeMap<String, String>,
    #[serde(default)]
    pub export_interval_secs: Option<u64>,
    #[serde(default)]
    pub queue_size: Option<usize>,
    #[serde(default)]
    pub prompt_shape_metrics: bool,
    #[serde(default)]
    pub metrics: TelemetryMetricsConfig,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct TelemetryMetricsConfig {
    #[serde(default)]
    pub endpoint: Option<String>,
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
    validate_telemetry_config(&config.telemetry)?;
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
        if let Some(cache_type_k) = &model.cache_type_k {
            if cache_type_k.trim().is_empty() {
                bail!("models[{index}].cache_type_k must not be empty when set");
            }
        }
        if let Some(cache_type_v) = &model.cache_type_v {
            if cache_type_v.trim().is_empty() {
                bail!("models[{index}].cache_type_v must not be empty when set");
            }
        }
        if model.batch == Some(0) {
            bail!("models[{index}].batch must be at least 1 when set");
        }
        if model.ubatch == Some(0) {
            bail!("models[{index}].ubatch must be at least 1 when set");
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

fn validate_telemetry_config(config: &TelemetryConfig) -> Result<()> {
    if let Some(service_name) = &config.service_name {
        if service_name.trim().is_empty() {
            bail!("telemetry.service_name must not be empty when set");
        }
    }
    if let Some(endpoint) = &config.endpoint {
        if endpoint.trim().is_empty() {
            bail!("telemetry.endpoint must not be empty when set");
        }
    }
    if let Some(endpoint) = &config.metrics.endpoint {
        if endpoint.trim().is_empty() {
            bail!("telemetry.metrics.endpoint must not be empty when set");
        }
    }
    for key in config.headers.keys() {
        if key.trim().is_empty() {
            bail!("telemetry.headers keys must not be empty");
        }
    }
    if let Some(export_interval_secs) = config.export_interval_secs {
        if export_interval_secs < 1 {
            bail!("telemetry.export_interval_secs must be at least 1");
        }
    }
    if let Some(queue_size) = config.queue_size {
        if queue_size < 1 {
            bail!("telemetry.queue_size must be at least 1");
        }
    }
    if config.prompt_shape_metrics {
        bail!("telemetry.prompt_shape_metrics is not supported yet and must remain false");
    }
    Ok(())
}

pub(crate) fn survey_plugin_enabled(config: &MeshConfig) -> bool {
    config
        .plugins
        .iter()
        .find(|entry| entry.name == SURVEY_PLUGIN_ID)
        .map(|entry| entry.enabled.unwrap_or(true))
        .unwrap_or(false)
}

pub fn resolve_plugins(config: &MeshConfig, _host_mode: PluginHostMode) -> Result<ResolvedPlugins> {
    let mut externals = Vec::new();
    let inactive = Vec::new();
    let mut names = BTreeMap::<String, ()>::new();
    let mut blackboard_enabled = true;
    let mut blobstore_enabled = true;
    let mut openai_endpoint_enabled = false;
    let mut openai_endpoint_url: Option<String> = None;
    let mut survey_enabled = false;
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
        if entry.name == SURVEY_PLUGIN_ID {
            if entry.command.is_some() || !entry.args.is_empty() || entry.url.is_some() {
                bail!(
                    "Plugin '{}' is served by mesh-llm itself; only `enabled` may be set",
                    SURVEY_PLUGIN_ID
                );
            }
            survey_enabled = enabled;
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
    if survey_enabled {
        let insert_at = usize::from(blackboard_enabled).min(externals.len());
        externals.insert(insert_at, survey_plugin_spec()?);
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

pub fn survey_plugin_spec() -> Result<ExternalPluginSpec> {
    let command = std::env::current_exe()
        .context("Cannot determine mesh-llm executable path")?
        .display()
        .to_string();
    Ok(ExternalPluginSpec {
        name: SURVEY_PLUGIN_ID.to_string(),
        command,
        args: vec![
            "--log-format".into(),
            "json".into(),
            "--plugin".into(),
            SURVEY_PLUGIN_ID.into(),
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
        assert_eq!(config.models[0].cache_type_k, None);
        assert_eq!(config.models[0].cache_type_v, None);
        assert_eq!(config.models[0].batch, None);
        assert_eq!(config.models[0].ubatch, None);
        assert_eq!(config.models[0].flash_attention, None);
        assert_eq!(
            config.models[1].mmproj.as_deref(),
            Some("bartowski/Qwen2.5-VL-7B-Instruct-GGUF/mmproj.gguf")
        );
        assert_eq!(config.models[1].gpu_id, None);
        assert_eq!(config.plugins.len(), 1);
        assert_eq!(config.plugins[0].name, "demo");
    }

    #[test]
    fn telemetry_config_deserializes_standard_metrics_settings() {
        let config: MeshConfig = toml::from_str(
            r#"
version = 1

[telemetry]
enabled = true
service_name = "mesh-llm"
endpoint = "https://otel.example.com"
headers = { "authorization" = "Bearer TOKEN" }
export_interval_secs = 15
queue_size = 2048
prompt_shape_metrics = false

[telemetry.metrics]
endpoint = "https://otel.example.com/v1/metrics"

[[plugin]]
name = "survey"
enabled = true
"#,
        )
        .unwrap();

        assert_eq!(config.telemetry.enabled, Some(true));
        assert_eq!(config.telemetry.service_name.as_deref(), Some("mesh-llm"));
        assert_eq!(
            config.telemetry.endpoint.as_deref(),
            Some("https://otel.example.com")
        );
        assert_eq!(
            config.telemetry.metrics.endpoint.as_deref(),
            Some("https://otel.example.com/v1/metrics")
        );
        assert_eq!(
            config
                .telemetry
                .headers
                .get("authorization")
                .map(String::as_str),
            Some("Bearer TOKEN")
        );
        assert_eq!(config.telemetry.export_interval_secs, Some(15));
        assert_eq!(config.telemetry.queue_size, Some(2048));
        assert!(!config.telemetry.prompt_shape_metrics);
    }

    #[test]
    fn telemetry_config_rejects_zero_queue_size() {
        let config: MeshConfig = toml::from_str(
            r#"
[telemetry]
queue_size = 0
"#,
        )
        .unwrap();

        let err = validate_config(&config).unwrap_err();
        assert!(
            err.to_string()
                .contains("telemetry.queue_size must be at least 1"),
            "unexpected error: {err}"
        );
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
                cache_type_k: None,
                cache_type_v: None,
                batch: None,
                ubatch: None,
                flash_attention: None,
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
                cache_type_k: None,
                cache_type_v: None,
                batch: None,
                ubatch: None,
                flash_attention: None,
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
                cache_type_k: None,
                cache_type_v: None,
                batch: None,
                ubatch: None,
                flash_attention: None,
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
                cache_type_k: None,
                cache_type_v: None,
                batch: None,
                ubatch: None,
                flash_attention: None,
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
                cache_type_k: None,
                cache_type_v: None,
                batch: None,
                ubatch: None,
                flash_attention: None,
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
                cache_type_k: None,
                cache_type_v: None,
                batch: None,
                ubatch: None,
                flash_attention: None,
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
                cache_type_k: None,
                cache_type_v: None,
                batch: None,
                ubatch: None,
                flash_attention: None,
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
                cache_type_k: None,
                cache_type_v: None,
                batch: None,
                ubatch: None,
                flash_attention: None,
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
                cache_type_k: None,
                cache_type_v: None,
                batch: None,
                ubatch: None,
                flash_attention: None,
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
                cache_type_k: None,
                cache_type_v: None,
                batch: None,
                ubatch: None,
                flash_attention: None,
            }],
            ..MeshConfig::default()
        };
        validate_config(&config).unwrap();
    }

    #[test]
    fn model_runtime_overrides_deserialize_from_toml() {
        let config: MeshConfig = toml::from_str(
            r#"
version = 1

[gpu]
assignment = "auto"

[[models]]
model = "Qwen3-8B-Q4_K_M"
cache_type_k = "q8_0"
cache_type_v = "q4_0"
batch = 2048
ubatch = 512
flash_attention = "enabled"
"#,
        )
        .unwrap();

        assert_eq!(config.models[0].cache_type_k.as_deref(), Some("q8_0"));
        assert_eq!(config.models[0].cache_type_v.as_deref(), Some("q4_0"));
        assert_eq!(config.models[0].batch, Some(2048));
        assert_eq!(config.models[0].ubatch, Some(512));
        assert_eq!(
            config.models[0].flash_attention,
            Some(FlashAttentionType::Enabled)
        );
    }

    #[test]
    fn model_cache_type_k_empty_rejected() {
        let config = MeshConfig {
            models: vec![ModelConfigEntry {
                model: "Qwen3-8B-Q4_K_M".into(),
                mmproj: None,
                ctx_size: None,
                gpu_id: None,
                parallel: None,
                cache_type_k: Some("   ".into()),
                cache_type_v: None,
                batch: None,
                ubatch: None,
                flash_attention: None,
            }],
            ..MeshConfig::default()
        };

        let err = validate_config(&config).unwrap_err();
        assert!(err
            .to_string()
            .contains("models[0].cache_type_k must not be empty when set"));
    }

    #[test]
    fn model_cache_type_v_empty_rejected() {
        let config = MeshConfig {
            models: vec![ModelConfigEntry {
                model: "Qwen3-8B-Q4_K_M".into(),
                mmproj: None,
                ctx_size: None,
                gpu_id: None,
                parallel: None,
                cache_type_k: None,
                cache_type_v: Some("   ".into()),
                batch: None,
                ubatch: None,
                flash_attention: None,
            }],
            ..MeshConfig::default()
        };

        let err = validate_config(&config).unwrap_err();
        assert!(err
            .to_string()
            .contains("models[0].cache_type_v must not be empty when set"));
    }

    #[test]
    fn model_batch_zero_rejected() {
        let config = MeshConfig {
            models: vec![ModelConfigEntry {
                model: "Qwen3-8B-Q4_K_M".into(),
                mmproj: None,
                ctx_size: None,
                gpu_id: None,
                parallel: None,
                cache_type_k: None,
                cache_type_v: None,
                batch: Some(0),
                ubatch: None,
                flash_attention: None,
            }],
            ..MeshConfig::default()
        };

        let err = validate_config(&config).unwrap_err();
        assert!(err
            .to_string()
            .contains("models[0].batch must be at least 1 when set"));
    }

    #[test]
    fn model_ubatch_zero_rejected() {
        let config = MeshConfig {
            models: vec![ModelConfigEntry {
                model: "Qwen3-8B-Q4_K_M".into(),
                mmproj: None,
                ctx_size: None,
                gpu_id: None,
                parallel: None,
                cache_type_k: None,
                cache_type_v: None,
                batch: None,
                ubatch: Some(0),
                flash_attention: None,
            }],
            ..MeshConfig::default()
        };

        let err = validate_config(&config).unwrap_err();
        assert!(err
            .to_string()
            .contains("models[0].ubatch must be at least 1 when set"));
    }
}
