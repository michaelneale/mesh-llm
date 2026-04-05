use anyhow::{anyhow, Context, Result};
use mesh_llm_plugin::{
    async_trait, plugin_server_info, EnsureInferenceEndpointRequest,
    EnsureInferenceEndpointResponse, EnsureInferenceWorkerRequest, EnsureInferenceWorkerResponse,
    InferenceEndpointDescriptor, InferenceLocalModelMatcher,
    InferenceProviderCapabilitiesDescriptor, Plugin, PluginContext, PluginError, PluginResult,
    PluginRuntime, PrepareMoeShardRequest, PrepareMoeShardResponse,
};
use rmcp::model::ServerInfo;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug)]
struct ActiveEndpoint {
    request: EnsureInferenceEndpointRequest,
    model_path: PathBuf,
    port: u16,
    context_length: u32,
    handle: crate::inference::provider::InferenceServerHandle,
}

#[derive(Clone, Debug)]
struct ActiveWorker {
    request: EnsureInferenceWorkerRequest,
    port: u16,
}

pub struct LlamaPlugin {
    name: String,
    bin_dir: Option<PathBuf>,
    binary_flavor: Option<crate::inference::launch::BinaryFlavor>,
    active_endpoint: Option<ActiveEndpoint>,
    active_worker: Option<ActiveWorker>,
}

impl LlamaPlugin {
    fn new(name: String) -> Self {
        Self {
            name,
            bin_dir: None,
            binary_flavor: None,
            active_endpoint: None,
            active_worker: None,
        }
    }

    fn parse_binary_flavor(
        value: Option<&str>,
    ) -> Result<Option<crate::inference::launch::BinaryFlavor>> {
        match value {
            None => Ok(None),
            Some("cpu") => Ok(Some(crate::inference::launch::BinaryFlavor::Cpu)),
            Some("cuda") => Ok(Some(crate::inference::launch::BinaryFlavor::Cuda)),
            Some("rocm") => Ok(Some(crate::inference::launch::BinaryFlavor::Rocm)),
            Some("vulkan") => Ok(Some(crate::inference::launch::BinaryFlavor::Vulkan)),
            Some("metal") => Ok(Some(crate::inference::launch::BinaryFlavor::Metal)),
            Some(other) => Err(anyhow!("Unsupported binary flavor '{other}'")),
        }
    }

    fn bin_dir(&self) -> Result<PathBuf> {
        if let Some(path) = &self.bin_dir {
            return Ok(path.clone());
        }
        crate::runtime::detect_bin_dir()
    }

    fn build_endpoint_request(
        request: &EnsureInferenceEndpointRequest,
    ) -> Result<crate::inference::provider::InferenceEndpointRequest> {
        let listen_port = request
            .requested_port
            .ok_or_else(|| anyhow!("Llama endpoint launch requires a requested port"))?;
        let endpoint_request = if request.worker_tunnel_ports.is_empty() {
            crate::inference::provider::InferenceEndpointRequest::local(
                &request.model_path,
                listen_port,
                request.model_bytes,
                request.local_vram_bytes,
            )
        } else {
            crate::inference::provider::InferenceEndpointRequest::distributed_host(
                &request.model_path,
                listen_port,
                request.worker_tunnel_ports.clone(),
                request.model_bytes,
                request.local_vram_bytes,
            )
            .with_tensor_split(request.tensor_split.clone())
            .with_draft_model_path(request.draft_model_path.as_deref())
            .with_draft_max(request.draft_max.unwrap_or(0))
            .with_total_group_vram_bytes(request.total_group_vram_bytes)
        };
        Ok(endpoint_request
            .with_ctx_size_override(request.ctx_size_override)
            .with_mmproj_path(request.mmproj_path.as_deref()))
    }

    async fn ensure_endpoint(
        &mut self,
        request: EnsureInferenceEndpointRequest,
    ) -> Result<EnsureInferenceEndpointResponse> {
        let endpoint_request = Self::build_endpoint_request(&request)?;
        let model_path = endpoint_request.model_path.clone();

        if let Some(active) = &self.active_endpoint {
            if active.request == request {
                return Ok(EnsureInferenceEndpointResponse {
                    address: format!("http://127.0.0.1:{}", active.port),
                    backend_label: "llama".into(),
                    context_length: active.context_length,
                });
            }
        }

        if let Some(active) = self.active_endpoint.take() {
            active.handle.shutdown().await;
        }

        let process = crate::inference::launch::start_llama_server(
            &self.bin_dir()?,
            self.binary_flavor,
            &endpoint_request,
        )
        .await
        .with_context(|| {
            format!(
                "Start llama endpoint '{}' for {}",
                request.endpoint_id,
                endpoint_request.model_path.display()
            )
        })?;

        let context_length = process.context_length;
        let port = process.listen_port;
        let handle = process.handle;
        self.active_endpoint = Some(ActiveEndpoint {
            request,
            model_path,
            port,
            context_length,
            handle,
        });

        Ok(EnsureInferenceEndpointResponse {
            address: format!("http://127.0.0.1:{port}"),
            backend_label: "llama".into(),
            context_length,
        })
    }

    async fn ensure_worker(
        &mut self,
        request: EnsureInferenceWorkerRequest,
    ) -> Result<EnsureInferenceWorkerResponse> {
        if let Some(active) = &self.active_worker {
            if active.request == request {
                return Ok(EnsureInferenceWorkerResponse { port: active.port });
            }
        }

        let worker_request = crate::inference::provider::InferenceWorkerRequest::default()
            .with_model_path(request.model_path.as_deref())
            .with_device_hint(request.device_hint.clone());
        let port = crate::inference::launch::start_rpc_server(
            &self.bin_dir()?,
            self.binary_flavor,
            &worker_request,
        )
        .await
        .with_context(|| "Start rpc-server for plugin-managed llama worker".to_string())?;

        self.active_worker = Some(ActiveWorker { request, port });
        Ok(EnsureInferenceWorkerResponse { port })
    }

    async fn prepare_moe_shard(
        &mut self,
        request: PrepareMoeShardRequest,
    ) -> Result<PrepareMoeShardResponse> {
        let assignment = crate::inference::moe::NodeAssignment {
            experts: request.experts,
            n_shared: request.n_shared as usize,
            n_unique: request.n_unique as usize,
        };
        crate::inference::moe::run_split(
            &self.bin_dir()?,
            Path::new(&request.model_path),
            &assignment,
            Path::new(&request.output_path),
        )
        .with_context(|| format!("Prepare MoE shard for {}", request.model_path))?;

        Ok(PrepareMoeShardResponse::default())
    }
}

#[async_trait]
impl Plugin for LlamaPlugin {
    fn plugin_id(&self) -> &str {
        &self.name
    }

    fn plugin_version(&self) -> String {
        crate::VERSION.to_string()
    }

    fn server_info(&self) -> ServerInfo {
        plugin_server_info(
            "mesh-llama",
            crate::VERSION,
            "llama inference provider",
            "Plugin-managed llama.cpp inference backend.",
            Some(
                "Uses inference/ensure_endpoint, inference/ensure_worker, and inference/prepare_moe_shard to manage llama.cpp runtime processes.",
            ),
        )
    }

    async fn initialize(
        &mut self,
        request: mesh_llm_plugin::PluginInitializeRequest,
        _context: &mut PluginContext<'_>,
    ) -> PluginResult<()> {
        let launch_info = request
            .host_launch_info()
            .map_err(|err| PluginError::internal(err.to_string()))?;
        self.bin_dir = launch_info.bin_dir.map(PathBuf::from);
        self.binary_flavor = Self::parse_binary_flavor(launch_info.binary_flavor.as_deref())
            .map_err(|err| PluginError::internal(err.to_string()))?;
        Ok(())
    }

    async fn health(&mut self, _context: &mut PluginContext<'_>) -> Result<String> {
        if let Some(active) = &self.active_endpoint {
            return Ok(format!(
                "serving {} on :{}",
                active.model_path.display(),
                active.port
            ));
        }
        if let Some(active) = &self.active_worker {
            return Ok(format!("worker on :{}", active.port));
        }
        Ok("idle".into())
    }

    async fn ensure_inference_endpoint(
        &mut self,
        request: EnsureInferenceEndpointRequest,
        _context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<EnsureInferenceEndpointResponse>> {
        Ok(Some(
            self.ensure_endpoint(request)
                .await
                .map_err(|err| PluginError::internal(err.to_string()))?,
        ))
    }

    async fn ensure_inference_worker(
        &mut self,
        request: EnsureInferenceWorkerRequest,
        _context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<EnsureInferenceWorkerResponse>> {
        Ok(Some(
            self.ensure_worker(request)
                .await
                .map_err(|err| PluginError::internal(err.to_string()))?,
        ))
    }

    async fn prepare_moe_shard(
        &mut self,
        request: PrepareMoeShardRequest,
        _context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<PrepareMoeShardResponse>> {
        Ok(Some(
            self.prepare_moe_shard(request)
                .await
                .map_err(|err| PluginError::internal(err.to_string()))?,
        ))
    }

    async fn list_inference_endpoints(
        &mut self,
        _context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<Vec<InferenceEndpointDescriptor>>> {
        Ok(Some(vec![InferenceEndpointDescriptor {
            endpoint_id: "local-llama".into(),
            address: self
                .active_endpoint
                .as_ref()
                .map(|active| format!("http://127.0.0.1:{}", active.port)),
            supports_streaming: true,
            local_model_matcher: InferenceLocalModelMatcher::GgufModelFile,
            provider_capabilities: InferenceProviderCapabilitiesDescriptor {
                supports_local_runtime: true,
                supports_distributed_host_runtime: true,
                requires_worker_runtime: true,
                supports_moe_shard_runtime: true,
            },
        }]))
    }
}

pub(crate) async fn run_plugin(name: String) -> anyhow::Result<()> {
    PluginRuntime::run(LlamaPlugin::new(name)).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_supported_binary_flavors() {
        assert_eq!(
            LlamaPlugin::parse_binary_flavor(Some("cpu")).unwrap(),
            Some(crate::inference::launch::BinaryFlavor::Cpu)
        );
        assert_eq!(
            LlamaPlugin::parse_binary_flavor(Some("cuda")).unwrap(),
            Some(crate::inference::launch::BinaryFlavor::Cuda)
        );
        assert_eq!(
            LlamaPlugin::parse_binary_flavor(Some("metal")).unwrap(),
            Some(crate::inference::launch::BinaryFlavor::Metal)
        );
        assert!(LlamaPlugin::parse_binary_flavor(Some("invalid")).is_err());
    }

    #[test]
    fn builds_local_endpoint_request_from_plugin_request() {
        let request = LlamaPlugin::build_endpoint_request(&EnsureInferenceEndpointRequest {
            endpoint_id: "local-llama".into(),
            model_path: "/tmp/model.gguf".into(),
            requested_port: Some(8123),
            model_bytes: 123,
            local_vram_bytes: 456,
            ctx_size_override: Some(8192),
            worker_tunnel_ports: Vec::new(),
            tensor_split: None,
            draft_model_path: None,
            draft_max: None,
            mmproj_path: None,
            total_group_vram_bytes: None,
        })
        .unwrap();

        assert_eq!(request.listen_port, 8123);
        assert_eq!(request.model_path, PathBuf::from("/tmp/model.gguf"));
        assert_eq!(request.model_bytes, 123);
        assert_eq!(request.local_vram_bytes, 456);
        assert_eq!(request.ctx_size_override, Some(8192));
        assert!(request.worker_tunnel_ports.is_empty());
    }

    #[test]
    fn builds_distributed_endpoint_request_from_plugin_request() {
        let request = LlamaPlugin::build_endpoint_request(&EnsureInferenceEndpointRequest {
            endpoint_id: "local-llama".into(),
            model_path: "/tmp/model.gguf".into(),
            requested_port: Some(8124),
            model_bytes: 123,
            local_vram_bytes: 456,
            ctx_size_override: Some(16384),
            worker_tunnel_ports: vec![7001, 7002],
            tensor_split: Some("3,5".into()),
            draft_model_path: Some("/tmp/draft.gguf".into()),
            draft_max: Some(8),
            mmproj_path: Some("/tmp/mmproj.gguf".into()),
            total_group_vram_bytes: Some(48_000_000_000),
        })
        .unwrap();

        assert_eq!(request.listen_port, 8124);
        assert_eq!(request.worker_tunnel_ports, vec![7001, 7002]);
        assert_eq!(request.tensor_split.as_deref(), Some("3,5"));
        assert_eq!(
            request.draft_model_path.as_deref(),
            Some(Path::new("/tmp/draft.gguf"))
        );
        assert_eq!(request.draft_max, 8);
        assert_eq!(
            request.mmproj_path.as_deref(),
            Some(Path::new("/tmp/mmproj.gguf"))
        );
        assert_eq!(request.total_group_vram_bytes, Some(48_000_000_000));
    }
}
