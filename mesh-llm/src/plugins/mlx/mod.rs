use anyhow::{anyhow, Context, Result};
use mesh_llm_plugin::{
    async_trait, plugin_server_info, EnsureInferenceEndpointRequest,
    EnsureInferenceEndpointResponse, InferenceEndpointDescriptor, InferenceLocalModelMatcher,
    InferenceProviderCapabilitiesDescriptor, Plugin, PluginContext, PluginResult, PluginRuntime,
};
use rmcp::model::ServerInfo;
use std::path::{Path, PathBuf};

struct ActiveEndpoint {
    endpoint_id: String,
    model_dir: PathBuf,
    model_name: String,
    port: u16,
    context_length: u32,
    handle: crate::inference::provider::InferenceServerHandle,
}

pub struct MlxPlugin {
    name: String,
    active: Option<ActiveEndpoint>,
}

impl MlxPlugin {
    fn new(name: String) -> Self {
        Self { name, active: None }
    }

    fn normalize_model_dir(path: &Path) -> Result<PathBuf> {
        let Some(dir) = crate::mlx::mlx_model_dir(path) else {
            anyhow::bail!(
                "MLX model path must point to a compatible model directory or a file inside one: {}",
                path.display()
            );
        };
        if !crate::mlx::is_mlx_model_dir(dir) {
            anyhow::bail!(
                "MLX model path is not a supported MLX model: {}",
                path.display()
            );
        }
        Ok(dir.to_path_buf())
    }

    fn model_name(model_path: &Path) -> String {
        if let Some(dir) = crate::mlx::mlx_model_dir(model_path) {
            if let Some(identity) =
                crate::models::huggingface_identity_for_path(&dir.join("config.json"))
            {
                if let Some(name) = identity.repo_id.rsplit('/').next() {
                    return name.to_string();
                }
            }
            if let Some(name) = dir.file_name().and_then(|value| value.to_str()) {
                return name.to_string();
            }
        }

        model_path
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string()
    }

    async fn ensure_endpoint(
        &mut self,
        request: EnsureInferenceEndpointRequest,
    ) -> Result<EnsureInferenceEndpointResponse> {
        let requested_port = request
            .requested_port
            .ok_or_else(|| anyhow!("MLX endpoint launch requires a requested port"))?;
        let model_path = PathBuf::from(&request.model_path);
        let model_dir = Self::normalize_model_dir(&model_path)?;
        let model_name = Self::model_name(&model_path);

        if let Some(active) = &self.active {
            if active.endpoint_id == request.endpoint_id
                && active.model_dir == model_dir
                && active.port == requested_port
            {
                return Ok(EnsureInferenceEndpointResponse {
                    address: format!("http://127.0.0.1:{}", active.port),
                    backend_label: "mlx".into(),
                    context_length: active.context_length,
                });
            }
        }

        if let Some(active) = self.active.take() {
            active.handle.shutdown().await;
        }

        let process = crate::mlx::start_mlx_server(&model_dir, model_name.clone(), requested_port)
            .await
            .with_context(|| {
                format!(
                    "Start MLX server for '{}' on port {}",
                    model_dir.display(),
                    requested_port
                )
            })?;
        let context_length = process.context_length;
        let handle = process.handle;
        self.active = Some(ActiveEndpoint {
            endpoint_id: request.endpoint_id,
            model_dir,
            model_name,
            port: requested_port,
            context_length,
            handle,
        });

        Ok(EnsureInferenceEndpointResponse {
            address: format!("http://127.0.0.1:{requested_port}"),
            backend_label: "mlx".into(),
            context_length,
        })
    }
}

#[async_trait]
impl Plugin for MlxPlugin {
    fn plugin_id(&self) -> &str {
        &self.name
    }

    fn plugin_version(&self) -> String {
        crate::VERSION.to_string()
    }

    fn server_info(&self) -> ServerInfo {
        plugin_server_info(
            "mesh-mlx",
            crate::VERSION,
            "MLX inference provider",
            "Plugin-managed MLX inference backend for Apple Silicon.",
            Some("Use inference/ensure_endpoint to launch a local MLX OpenAI-compatible endpoint."),
        )
    }

    async fn health(&mut self, _context: &mut PluginContext<'_>) -> Result<String> {
        Ok(match &self.active {
            Some(active) => format!("serving {} on :{}", active.model_name, active.port),
            None => "idle".into(),
        })
    }

    async fn ensure_inference_endpoint(
        &mut self,
        request: EnsureInferenceEndpointRequest,
        _context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<EnsureInferenceEndpointResponse>> {
        Ok(Some(self.ensure_endpoint(request).await.map_err(
            |err| mesh_llm_plugin::PluginError::internal(err.to_string()),
        )?))
    }

    async fn list_inference_endpoints(
        &mut self,
        _context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<Vec<InferenceEndpointDescriptor>>> {
        Ok(Some(vec![InferenceEndpointDescriptor {
            endpoint_id: "local-mlx".into(),
            address: self
                .active
                .as_ref()
                .map(|active| format!("http://127.0.0.1:{}", active.port)),
            supports_streaming: true,
            local_model_matcher: InferenceLocalModelMatcher::MlxModelDir,
            provider_capabilities: InferenceProviderCapabilitiesDescriptor {
                supports_local_runtime: true,
                supports_distributed_host_runtime: false,
                requires_worker_runtime: false,
                supports_moe_shard_runtime: false,
            },
        }]))
    }
}

pub(crate) async fn run_plugin(name: String) -> anyhow::Result<()> {
    PluginRuntime::run(MlxPlugin::new(name)).await
}
