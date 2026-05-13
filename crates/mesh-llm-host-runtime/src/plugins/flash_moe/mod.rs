use std::sync::Arc;
use std::time::Duration;

use anyhow::{bail, Context, Result};
use mesh_llm_plugin::{
    capability, plugin_server_info, PluginMetadata, PluginRuntime, PluginStartupPolicy,
};
use tokio::process::{Child, Command};
use tokio::sync::Mutex;

const ENV_COMMAND: &str = "MESH_LLM_FLASH_MOE_COMMAND";
const ENV_ARGS_JSON: &str = "MESH_LLM_FLASH_MOE_ARGS_JSON";
const ENV_URL: &str = "MESH_LLM_FLASH_MOE_URL";
const ENDPOINT_ID: &str = "flash-moe";
const HEALTH_TIMEOUT: Duration = Duration::from_secs(1);
const INSTALL_HINT: &str = "Install Flash-MoE separately and set \
                           MESH_LLM_FLASH_MOE_COMMAND to its infer binary, \
                           or set MESH_LLM_FLASH_MOE_URL to an already-running \
                           Flash-MoE /v1 endpoint.";

#[derive(Clone, Debug, PartialEq, Eq)]
enum FlashMoeSource {
    Managed {
        command: String,
        args: Vec<String>,
        port: u16,
    },
    External {
        base_url: String,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct FlashMoeConfig {
    source: FlashMoeSource,
}

impl FlashMoeConfig {
    fn from_env() -> Result<Self> {
        Self::from_values(
            std::env::var(ENV_COMMAND).ok(),
            std::env::var(ENV_ARGS_JSON).ok(),
            std::env::var(ENV_URL).ok(),
        )
    }

    fn from_values(
        command: Option<String>,
        args_json: Option<String>,
        url: Option<String>,
    ) -> Result<Self> {
        let command = command
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty());
        let url = url
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty());
        if command.is_some() && url.is_some() {
            bail!("flash-moe plugin accepts either {ENV_COMMAND} or {ENV_URL}, not both");
        }
        if let Some(base_url) = url {
            return Ok(Self {
                source: FlashMoeSource::External {
                    base_url: normalize_base_url(&base_url),
                },
            });
        }
        let command = command.with_context(|| {
            format!("flash-moe plugin requires {ENV_COMMAND} or {ENV_URL}. {INSTALL_HINT}")
        })?;
        let args = match args_json {
            Some(raw) if !raw.trim().is_empty() => serde_json::from_str::<Vec<String>>(&raw)
                .with_context(|| format!("parse {ENV_ARGS_JSON}"))?,
            _ => Vec::new(),
        };
        let port = allocate_local_port().context("allocate flash-moe endpoint port")?;
        Ok(Self {
            source: FlashMoeSource::Managed {
                command,
                args,
                port,
            },
        })
    }

    fn endpoint_base_url(&self) -> String {
        match &self.source {
            FlashMoeSource::Managed { port, .. } => format!("http://127.0.0.1:{port}/v1"),
            FlashMoeSource::External { base_url } => base_url.clone(),
        }
    }

    fn managed(&self) -> bool {
        matches!(self.source, FlashMoeSource::Managed { .. })
    }
}

#[derive(Clone)]
struct FlashMoeState {
    config: FlashMoeConfig,
    child: Arc<Mutex<Option<Child>>>,
}

impl FlashMoeState {
    fn new(config: FlashMoeConfig) -> Self {
        Self {
            config,
            child: Arc::new(Mutex::new(None)),
        }
    }

    async fn ensure_started(&self) -> Result<()> {
        let FlashMoeSource::Managed {
            command,
            args,
            port,
        } = &self.config.source
        else {
            return Ok(());
        };

        let mut child = self.child.lock().await;
        if let Some(existing) = child.as_mut() {
            match existing.try_wait().context("poll flash-moe process")? {
                Some(status) => bail!("flash-moe exited before readiness: {status}"),
                None => return Ok(()),
            }
        }

        let managed_args = managed_command_args(args, *port)?;
        let mut command_builder = Command::new(command);
        command_builder.args(&managed_args);
        command_builder.stdin(std::process::Stdio::null());
        command_builder.stdout(std::process::Stdio::null());
        command_builder.stderr(std::process::Stdio::inherit());
        command_builder.kill_on_drop(true);
        let spawned = command_builder
            .spawn()
            .with_context(|| format!("launch flash-moe backend via {command}. {INSTALL_HINT}"))?;
        *child = Some(spawned);
        Ok(())
    }

    async fn health(&self) -> Result<String> {
        match &self.config.source {
            FlashMoeSource::External { base_url } => Ok(format!("external_url={base_url}")),
            FlashMoeSource::Managed { port, .. } => {
                let mut child = self.child.lock().await;
                match child.as_mut() {
                    Some(process) => {
                        if let Some(status) =
                            process.try_wait().context("poll flash-moe process")?
                        {
                            *child = None;
                            bail!("flash-moe exited with {status}");
                        }
                    }
                    None => return Ok(format!("starting endpoint=http://127.0.0.1:{port}/v1")),
                }
                drop(child);

                let health_url = format!("http://127.0.0.1:{port}/health");
                match reqwest::Client::builder()
                    .timeout(HEALTH_TIMEOUT)
                    .build()
                    .context("build flash-moe health client")?
                    .get(&health_url)
                    .send()
                    .await
                {
                    Ok(response) if response.status().is_success() => {
                        Ok(format!("endpoint=http://127.0.0.1:{port}/v1"))
                    }
                    Ok(response) => Ok(format!(
                        "starting endpoint=http://127.0.0.1:{port}/v1 health_status={}",
                        response.status()
                    )),
                    Err(error) => Ok(format!(
                        "starting endpoint=http://127.0.0.1:{port}/v1 detail={error}"
                    )),
                }
            }
        }
    }
}

fn normalize_base_url(value: &str) -> String {
    value.trim().trim_end_matches('/').to_string()
}

fn allocate_local_port() -> Result<u16> {
    let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
    let port = listener.local_addr()?.port();
    drop(listener);
    Ok(port)
}

fn managed_command_args(user_args: &[String], port: u16) -> Result<Vec<String>> {
    if user_args
        .iter()
        .any(|arg| arg == "--serve" || arg.starts_with("--serve="))
    {
        bail!("mesh-llm owns the flash-moe --serve port; remove --serve from plugin args");
    }
    let mut args = user_args.to_vec();
    args.push("--serve".to_string());
    args.push(port.to_string());
    Ok(args)
}

fn build_plugin_from_config(name: String, config: FlashMoeConfig) -> mesh_llm_plugin::SimplePlugin {
    let endpoint_base = config.endpoint_base_url();
    let state = FlashMoeState::new(config);
    let startup_state = state.clone();
    let health_state = state.clone();
    let inference_endpoint = if state.config.managed() {
        mesh_llm_plugin::inference::provider(ENDPOINT_ID, endpoint_base.clone())
    } else {
        mesh_llm_plugin::inference::openai_http(ENDPOINT_ID, endpoint_base.clone())
    };

    mesh_llm_plugin::plugin! {
        metadata: PluginMetadata::new(
            name,
            crate::VERSION,
            plugin_server_info(
                "mesh-flash-moe",
                crate::VERSION,
                "Flash-MoE SSD Expert Streaming Provider",
                "Registers a flash-moe OpenAI-compatible endpoint for single-node SSD expert streaming.",
                Some(
                    "Configure [[plugin]] name = \"flash-moe\" with either command/args \
                     for a managed flash-moe process or url for an already-running endpoint. \
                     Flash-MoE itself is installed separately.",
                ),
            ),
        ),
        startup_policy: PluginStartupPolicy::Any,
        provides: [
            capability("endpoint:inference"),
            capability("endpoint:inference/openai_compatible"),
            capability("backend:flash-moe"),
            capability("backend:ssd-expert-streaming"),
        ],
        inference: [
            inference_endpoint,
        ],
        health: move |_context| {
            let health_state = health_state.clone();
            Box::pin(async move { health_state.health().await })
        },
        on_initialized: move |_context| {
            let startup_state = startup_state.clone();
            Box::pin(async move { startup_state.ensure_started().await })
        },
    }
}

fn build_plugin(name: String) -> Result<mesh_llm_plugin::SimplePlugin> {
    let config = FlashMoeConfig::from_env()?;
    Ok(build_plugin_from_config(name, config))
}

pub(crate) async fn run_plugin(name: String) -> Result<()> {
    PluginRuntime::run(build_plugin(name)?).await
}

#[cfg(test)]
mod tests {
    use mesh_llm_plugin::Plugin;

    use super::*;

    #[test]
    fn managed_command_args_append_owned_serve_port() {
        let args = managed_command_args(
            &[
                "--model".to_string(),
                "/models/qwen3.5".to_string(),
                "--tokens".to_string(),
                "128".to_string(),
            ],
            8123,
        )
        .unwrap();

        assert_eq!(
            args,
            vec![
                "--model",
                "/models/qwen3.5",
                "--tokens",
                "128",
                "--serve",
                "8123"
            ]
        );
    }

    #[test]
    fn managed_command_args_reject_user_supplied_serve_port() {
        let err = managed_command_args(&["--serve".to_string(), "9000".to_string()], 8123)
            .expect_err("user-owned serve port must be rejected");
        assert!(err.to_string().contains("--serve"));
    }

    #[test]
    fn config_accepts_external_url_mode() {
        let config = FlashMoeConfig::from_values(
            None,
            None,
            Some(" http://127.0.0.1:8000/v1/ ".to_string()),
        )
        .unwrap();

        assert_eq!(
            config.source,
            FlashMoeSource::External {
                base_url: "http://127.0.0.1:8000/v1".to_string()
            }
        );
        assert_eq!(config.endpoint_base_url(), "http://127.0.0.1:8000/v1");
    }

    #[test]
    fn config_rejects_command_and_url_together() {
        let err = FlashMoeConfig::from_values(
            Some("/opt/flash-moe/infer".to_string()),
            None,
            Some("http://127.0.0.1:8000/v1".to_string()),
        )
        .expect_err("command and url must be mutually exclusive");
        assert!(err.to_string().contains("not both"));
    }

    #[test]
    fn config_missing_source_explains_external_install_boundary() {
        let err = FlashMoeConfig::from_values(None, None, None)
            .expect_err("flash-moe requires a managed command or attached endpoint");
        let message = err.to_string();

        assert!(message.contains("Install Flash-MoE separately"));
        assert!(message.contains(ENV_COMMAND));
        assert!(message.contains(ENV_URL));
    }

    #[tokio::test]
    async fn managed_start_failure_explains_external_install_boundary() {
        let state = FlashMoeState::new(FlashMoeConfig {
            source: FlashMoeSource::Managed {
                command: "/definitely/missing/flash-moe/infer".to_string(),
                args: Vec::new(),
                port: 8123,
            },
        });

        let message = state
            .ensure_started()
            .await
            .expect_err("missing external flash-moe binary must fail")
            .to_string();

        assert!(message.contains("Install Flash-MoE separately"));
        assert!(message.contains("/definitely/missing/flash-moe/infer"));
    }

    #[test]
    fn managed_manifest_declares_plugin_owned_openai_endpoint() {
        let plugin = build_plugin_from_config(
            "flash-moe".to_string(),
            FlashMoeConfig {
                source: FlashMoeSource::Managed {
                    command: "/opt/flash-moe/infer".to_string(),
                    args: vec!["--model".to_string(), "/models/qwen3.5".to_string()],
                    port: 8123,
                },
            },
        );
        let manifest = plugin.manifest().expect("manifest");
        let endpoint = manifest.endpoints.first().expect("endpoint");

        assert_eq!(endpoint.endpoint_id, ENDPOINT_ID);
        assert_eq!(
            endpoint.address.as_deref(),
            Some("http://127.0.0.1:8123/v1")
        );
        assert_eq!(endpoint.protocol.as_deref(), Some("openai_compatible"));
        assert!(endpoint.supports_streaming);
        assert!(endpoint.managed_by_plugin);
        assert!(manifest
            .capabilities
            .iter()
            .any(|capability| capability == "backend:ssd-expert-streaming"));
    }

    #[test]
    fn external_manifest_declares_attached_openai_endpoint() {
        let plugin = build_plugin_from_config(
            "flash-moe".to_string(),
            FlashMoeConfig {
                source: FlashMoeSource::External {
                    base_url: "http://127.0.0.1:8000/v1".to_string(),
                },
            },
        );
        let manifest = plugin.manifest().expect("manifest");
        let endpoint = manifest.endpoints.first().expect("endpoint");

        assert_eq!(
            endpoint.address.as_deref(),
            Some("http://127.0.0.1:8000/v1")
        );
        assert!(!endpoint.managed_by_plugin);
    }
}
