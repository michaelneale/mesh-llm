pub mod blobstore;
mod config;
pub(crate) mod mcp;
mod runtime;
mod support;
mod transport;

use anyhow::{bail, Context, Result};
pub use mesh_llm_plugin::proto;
use rmcp::model::ErrorCode;
use rmcp::model::ServerInfo;
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex};

#[allow(unused_imports)]
pub use self::config::ExternalPluginSpec;
pub use self::config::{
    config_path, load_config, resolve_plugins, PluginHostMode, ResolvedPlugins,
};
use self::runtime::ExternalPlugin;
pub(crate) use self::support::parse_optional_json;
use self::support::{format_args_for_log, format_slice_for_log, format_tool_names_for_log};
use self::transport::make_instance_id;
#[cfg(all(test, unix))]
use self::transport::unix_socket_path;
#[cfg(test)]
use mesh_llm_plugin::MeshVisibility;

pub const BLACKBOARD_PLUGIN_ID: &str = "blackboard";
pub const BLOBSTORE_PLUGIN_ID: &str = "blobstore";
pub const MLX_PLUGIN_ID: &str = "mlx";
pub(crate) const PROTOCOL_VERSION: u32 = mesh_llm_plugin::PROTOCOL_VERSION;
const CONNECT_TIMEOUT_SECS: u64 = 10;
const REQUEST_TIMEOUT_SECS: u64 = 30;
const HEALTH_CHECK_INTERVAL_SECS: u64 = 15;

#[derive(Clone, Debug)]
pub enum PluginMeshEvent {
    Channel {
        plugin_id: String,
        message: proto::ChannelMessage,
    },
    BulkTransfer {
        plugin_id: String,
        message: proto::BulkTransferMessage,
    },
}

#[derive(Clone, Debug, Serialize)]
pub struct ToolSummary {
    pub name: String,
    pub description: String,
    pub input_schema_json: String,
}

#[derive(Clone, Debug)]
pub struct ToolCallResult {
    pub content_json: String,
    pub is_error: bool,
}

#[derive(Clone, Debug)]
pub struct RpcResult {
    pub result_json: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ManagedInferenceEndpoint {
    pub plugin_name: String,
    pub endpoint_id: String,
    pub address: Option<String>,
    pub supports_streaming: bool,
    pub local_model_matcher: mesh_llm_plugin::InferenceLocalModelMatcher,
    pub provider_capabilities: mesh_llm_plugin::InferenceProviderCapabilitiesDescriptor,
}

pub(crate) type BridgeFuture<T> = Pin<Box<dyn Future<Output = T> + Send>>;

pub trait PluginRpcBridge: Send + Sync {
    fn handle_request(
        &self,
        plugin_name: String,
        method: String,
        params_json: String,
    ) -> BridgeFuture<Result<RpcResult, proto::ErrorResponse>>;

    fn handle_notification(
        &self,
        plugin_name: String,
        method: String,
        params_json: String,
    ) -> BridgeFuture<()>;
}

#[derive(Clone, Debug, Serialize)]
pub struct PluginSummary {
    pub name: String,
    pub kind: String,
    pub enabled: bool,
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub capabilities: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub command: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub args: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub tools: Vec<ToolSummary>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Clone)]
pub struct PluginManager {
    inner: Arc<PluginManagerInner>,
}

struct PluginManagerInner {
    plugins: BTreeMap<String, ExternalPlugin>,
    inactive: BTreeMap<String, PluginSummary>,
    rpc_bridge: Arc<Mutex<Option<Arc<dyn PluginRpcBridge>>>>,
    #[cfg(test)]
    bridged_plugins: BTreeSet<String>,
}

impl PluginManager {
    pub async fn start(
        specs: &ResolvedPlugins,
        host_mode: PluginHostMode,
        mesh_tx: mpsc::Sender<PluginMeshEvent>,
    ) -> Result<Self> {
        if specs.externals.is_empty() {
            tracing::info!("Plugin manager: no plugins enabled");
        } else {
            let names = specs
                .externals
                .iter()
                .map(|spec| spec.name.as_str())
                .collect::<Vec<_>>()
                .join(", ");
            tracing::info!(
                "Plugin manager: loading {} plugin(s): {}",
                specs.externals.len(),
                names
            );
        }

        let rpc_bridge = Arc::new(Mutex::new(None));
        let instance_id = make_instance_id();
        let mut plugins = BTreeMap::new();
        for spec in &specs.externals {
            tracing::info!(
                plugin = %spec.name,
                command = %spec.command,
                args = %format_args_for_log(&spec.args),
                "Loading plugin"
            );
            let plugin = match ExternalPlugin::spawn(
                spec,
                instance_id.clone(),
                host_mode,
                mesh_tx.clone(),
                rpc_bridge.clone(),
            )
            .await
            {
                Ok(plugin) => plugin,
                Err(err) => {
                    tracing::error!(
                        plugin = %spec.name,
                        error = %err,
                        "Plugin failed to load"
                    );
                    return Err(err);
                }
            };
            let summary = plugin.summary().await;
            tracing::info!(
                plugin = %summary.name,
                version = %summary.version.as_deref().unwrap_or("unknown"),
                capabilities = %format_slice_for_log(&summary.capabilities),
                tools = %format_tool_names_for_log(&summary.tools),
                "Plugin loaded successfully"
            );
            plugins.insert(spec.name.clone(), plugin);
        }
        let manager = Self {
            inner: Arc::new(PluginManagerInner {
                plugins,
                inactive: specs
                    .inactive
                    .iter()
                    .cloned()
                    .map(|summary| (summary.name.clone(), summary))
                    .collect(),
                rpc_bridge,
                #[cfg(test)]
                bridged_plugins: BTreeSet::new(),
            }),
        };
        manager.start_supervisor();
        Ok(manager)
    }

    #[cfg(test)]
    pub fn for_test_bridge(plugin_names: &[&str], bridge: Arc<dyn PluginRpcBridge>) -> Self {
        Self {
            inner: Arc::new(PluginManagerInner {
                plugins: BTreeMap::new(),
                inactive: BTreeMap::new(),
                rpc_bridge: Arc::new(Mutex::new(Some(bridge))),
                bridged_plugins: plugin_names
                    .iter()
                    .map(|name| (*name).to_string())
                    .collect(),
            }),
        }
    }

    pub async fn list(&self) -> Vec<PluginSummary> {
        let mut summaries =
            Vec::with_capacity(self.inner.plugins.len() + self.inner.inactive.len());
        for plugin in self.inner.plugins.values() {
            summaries.push(plugin.summary().await);
        }
        summaries.extend(self.inner.inactive.values().cloned());
        summaries.sort_by(|a, b| a.name.cmp(&b.name));
        summaries
    }

    pub async fn is_enabled(&self, name: &str) -> bool {
        if let Some(plugin) = self.inner.plugins.get(name) {
            plugin.is_enabled_running().await
        } else if cfg!(test) && self.is_test_bridge_enabled(name) {
            true
        } else {
            false
        }
    }

    pub fn is_available(&self, name: &str) -> bool {
        self.inner.plugins.contains_key(name) || self.is_test_bridge_enabled(name)
    }

    pub async fn tools(&self, name: &str) -> Result<Vec<ToolSummary>> {
        if let Some(summary) = self.inner.inactive.get(name) {
            bail!(
                "Plugin '{}' is disabled: {}",
                name,
                summary.error.as_deref().unwrap_or("unavailable")
            );
        }
        let plugin = self
            .inner
            .plugins
            .get(name)
            .with_context(|| format!("Unknown plugin '{name}'"))?;
        plugin.list_tools().await
    }

    pub async fn call_tool(
        &self,
        plugin_name: &str,
        tool_name: &str,
        arguments_json: &str,
    ) -> Result<ToolCallResult> {
        if let Some(summary) = self.inner.inactive.get(plugin_name) {
            bail!(
                "Plugin '{}' is disabled: {}",
                plugin_name,
                summary.error.as_deref().unwrap_or("unavailable")
            );
        }
        let plugin = self
            .inner
            .plugins
            .get(plugin_name)
            .with_context(|| format!("Unknown plugin '{plugin_name}'"))?;
        plugin.call_tool(tool_name, arguments_json).await
    }

    pub async fn mcp_request<T, P>(&self, plugin_name: &str, method: &str, params: P) -> Result<T>
    where
        T: serde::de::DeserializeOwned,
        P: Serialize,
    {
        if self.is_test_bridge_enabled(plugin_name) {
            let bridge = self
                .inner
                .rpc_bridge
                .lock()
                .await
                .clone()
                .with_context(|| format!("No bridge configured for test plugin '{plugin_name}'"))?;
            let params_json = serde_json::to_string(&params)
                .with_context(|| format!("Serialize params for test plugin '{plugin_name}'"))?;
            let result = bridge
                .handle_request(plugin_name.to_string(), method.to_string(), params_json)
                .await
                .map_err(|err| self::support::plugin_error(plugin_name, method, &err))?;
            return serde_json::from_str(&result.result_json)
                .with_context(|| format!("Decode response from test plugin '{plugin_name}'"));
        }
        if let Some(summary) = self.inner.inactive.get(plugin_name) {
            bail!(
                "Plugin '{}' is disabled: {}",
                plugin_name,
                summary.error.as_deref().unwrap_or("unavailable")
            );
        }
        let plugin = self
            .inner
            .plugins
            .get(plugin_name)
            .with_context(|| format!("Unknown plugin '{plugin_name}'"))?;
        plugin.mcp_request(method, params).await
    }

    pub async fn ensure_managed_inference_endpoint(
        &self,
        plugin_name: &str,
        endpoint_id: &str,
        model_path: &std::path::Path,
        requested_port: Option<u16>,
        ctx_size_override: Option<u32>,
    ) -> Result<mesh_llm_plugin::EnsureInferenceEndpointResponse> {
        self.mcp_request(
            plugin_name,
            "inference/ensure_endpoint",
            mesh_llm_plugin::EnsureInferenceEndpointRequest {
                endpoint_id: endpoint_id.to_string(),
                model_path: model_path.display().to_string(),
                requested_port,
                ctx_size_override,
            },
        )
        .await
    }

    pub async fn managed_inference_endpoints(&self) -> Result<Vec<ManagedInferenceEndpoint>> {
        #[cfg(test)]
        let plugin_names = {
            let mut plugin_names = self
                .inner
                .plugins
                .keys()
                .cloned()
                .collect::<BTreeSet<String>>();
            plugin_names.extend(self.inner.bridged_plugins.iter().cloned());
            plugin_names
        };
        #[cfg(not(test))]
        let plugin_names = self
            .inner
            .plugins
            .keys()
            .cloned()
            .collect::<BTreeSet<String>>();

        let mut endpoints = Vec::new();
        for plugin_name in plugin_names {
            let descriptors = match self
                .mcp_request::<Vec<mesh_llm_plugin::InferenceEndpointDescriptor>, _>(
                    &plugin_name,
                    "inference/list_endpoints",
                    serde_json::json!({}),
                )
                .await
            {
                Ok(descriptors) => descriptors,
                Err(err) if is_method_not_found_error(&err) => continue,
                Err(err) => {
                    return Err(err).with_context(|| {
                        format!("List managed inference endpoints for '{plugin_name}'")
                    });
                }
            };
            for descriptor in descriptors {
                endpoints.push(ManagedInferenceEndpoint {
                    plugin_name: plugin_name.clone(),
                    endpoint_id: descriptor.endpoint_id,
                    address: descriptor.address,
                    supports_streaming: descriptor.supports_streaming,
                    local_model_matcher: descriptor.local_model_matcher,
                    provider_capabilities: descriptor.provider_capabilities,
                });
            }
        }
        endpoints.sort_by(|a, b| {
            a.plugin_name
                .cmp(&b.plugin_name)
                .then_with(|| a.endpoint_id.cmp(&b.endpoint_id))
        });
        Ok(endpoints)
    }

    pub async fn mcp_notify<P>(&self, plugin_name: &str, method: &str, params: P) -> Result<()>
    where
        P: Serialize,
    {
        if self.is_test_bridge_enabled(plugin_name) {
            let bridge = self
                .inner
                .rpc_bridge
                .lock()
                .await
                .clone()
                .with_context(|| format!("No bridge configured for test plugin '{plugin_name}'"))?;
            let params_json = serde_json::to_string(&params)
                .with_context(|| format!("Serialize params for test plugin '{plugin_name}'"))?;
            bridge
                .handle_notification(plugin_name.to_string(), method.to_string(), params_json)
                .await;
            return Ok(());
        }
        if let Some(summary) = self.inner.inactive.get(plugin_name) {
            bail!(
                "Plugin '{}' is disabled: {}",
                plugin_name,
                summary.error.as_deref().unwrap_or("unavailable")
            );
        }
        let plugin = self
            .inner
            .plugins
            .get(plugin_name)
            .with_context(|| format!("Unknown plugin '{plugin_name}'"))?;
        plugin.mcp_notify(method, params).await
    }

    fn is_test_bridge_enabled(&self, _plugin_name: &str) -> bool {
        #[cfg(test)]
        {
            return self.inner.bridged_plugins.contains(_plugin_name);
        }
        #[allow(unreachable_code)]
        false
    }

    pub async fn list_server_infos(&self) -> Vec<(String, ServerInfo)> {
        let mut infos = Vec::new();
        for (name, plugin) in &self.inner.plugins {
            if let Ok(info) = plugin.server_info().await {
                infos.push((name.clone(), info));
            }
        }
        infos
    }

    pub async fn set_rpc_bridge(&self, bridge: Option<Arc<dyn PluginRpcBridge>>) {
        *self.inner.rpc_bridge.lock().await = bridge;
    }

    pub async fn dispatch_channel_message(&self, event: PluginMeshEvent) -> Result<()> {
        let PluginMeshEvent::Channel { plugin_id, message } = event else {
            bail!("expected plugin channel event");
        };
        let Some(plugin) = self.inner.plugins.get(&plugin_id) else {
            tracing::debug!(
                "Dropping channel message for unloaded plugin '{}'",
                plugin_id
            );
            return Ok(());
        };
        plugin.send_channel_message(message).await
    }

    pub async fn dispatch_bulk_transfer_message(&self, event: PluginMeshEvent) -> Result<()> {
        let PluginMeshEvent::BulkTransfer { plugin_id, message } = event else {
            bail!("expected plugin bulk transfer event");
        };
        let Some(plugin) = self.inner.plugins.get(&plugin_id) else {
            tracing::debug!(
                "Dropping bulk transfer message for unloaded plugin '{}'",
                plugin_id
            );
            return Ok(());
        };
        plugin.send_bulk_transfer_message(message).await
    }

    pub async fn broadcast_mesh_event(&self, event: proto::MeshEvent) -> Result<()> {
        for plugin in self.inner.plugins.values() {
            plugin.send_mesh_event(event.clone()).await?;
        }
        Ok(())
    }

    fn start_supervisor(&self) {
        let manager = self.clone();
        tokio::spawn(async move {
            let mut ticker =
                tokio::time::interval(std::time::Duration::from_secs(HEALTH_CHECK_INTERVAL_SECS));
            loop {
                ticker.tick().await;
                for plugin in manager.inner.plugins.values() {
                    if let Err(err) = plugin.supervise().await {
                        tracing::warn!(
                            plugin = %plugin.name(),
                            error = %err,
                            "Plugin supervision round failed"
                        );
                    }
                }
            }
        });
    }
}

fn is_method_not_found_error(err: &anyhow::Error) -> bool {
    err.to_string()
        .contains(&format!("(code {})", ErrorCode::METHOD_NOT_FOUND.0))
}

pub async fn run_plugin_process(name: String) -> Result<()> {
    match name.as_str() {
        BLACKBOARD_PLUGIN_ID => crate::plugins::blackboard::run_plugin(name).await,
        BLOBSTORE_PLUGIN_ID => crate::plugins::blobstore::run_plugin(name).await,
        #[cfg(target_os = "macos")]
        MLX_PLUGIN_ID => crate::plugins::mlx::run_plugin(name).await,
        _ => bail!("Unknown built-in plugin '{}'", name),
    }
}

#[cfg(test)]
mod tests {
    use super::config::{MeshConfig, PluginConfigEntry};
    use super::*;

    fn private_host_mode() -> PluginHostMode {
        PluginHostMode {
            mesh_visibility: MeshVisibility::Private,
        }
    }

    #[test]
    fn resolves_default_blackboard_plugin() {
        let resolved = resolve_plugins(&MeshConfig::default(), private_host_mode()).unwrap();
        #[cfg(target_os = "macos")]
        assert_eq!(resolved.externals.len(), 3);
        #[cfg(not(target_os = "macos"))]
        assert_eq!(resolved.externals.len(), 2);
        assert_eq!(resolved.externals[0].name, BLACKBOARD_PLUGIN_ID);
        assert_eq!(resolved.externals[1].name, BLOBSTORE_PLUGIN_ID);
        #[cfg(target_os = "macos")]
        assert_eq!(resolved.externals[2].name, MLX_PLUGIN_ID);
        assert!(resolved.inactive.is_empty());
    }

    #[test]
    fn blackboard_can_be_disabled() {
        let config = MeshConfig {
            plugins: vec![PluginConfigEntry {
                name: BLACKBOARD_PLUGIN_ID.into(),
                enabled: Some(false),
                command: None,
                args: Vec::new(),
            }],
            ..MeshConfig::default()
        };
        let resolved = resolve_plugins(&config, private_host_mode()).unwrap();
        #[cfg(target_os = "macos")]
        assert_eq!(resolved.externals.len(), 2);
        #[cfg(not(target_os = "macos"))]
        assert_eq!(resolved.externals.len(), 1);
        assert_eq!(resolved.externals[0].name, BLOBSTORE_PLUGIN_ID);
        #[cfg(target_os = "macos")]
        assert_eq!(resolved.externals[1].name, MLX_PLUGIN_ID);
        assert!(resolved.inactive.is_empty());
    }

    #[test]
    fn blobstore_can_be_disabled() {
        let config = MeshConfig {
            plugins: vec![PluginConfigEntry {
                name: BLOBSTORE_PLUGIN_ID.into(),
                enabled: Some(false),
                command: None,
                args: Vec::new(),
            }],
            ..MeshConfig::default()
        };
        let resolved = resolve_plugins(&config, private_host_mode()).unwrap();
        #[cfg(target_os = "macos")]
        assert_eq!(resolved.externals.len(), 2);
        #[cfg(not(target_os = "macos"))]
        assert_eq!(resolved.externals.len(), 1);
        assert_eq!(resolved.externals[0].name, BLACKBOARD_PLUGIN_ID);
        #[cfg(target_os = "macos")]
        assert_eq!(resolved.externals[1].name, MLX_PLUGIN_ID);
        assert!(resolved.inactive.is_empty());
    }

    #[test]
    fn blackboard_is_resolved_on_public_meshes() {
        let resolved = resolve_plugins(
            &MeshConfig::default(),
            PluginHostMode {
                mesh_visibility: MeshVisibility::Public,
            },
        )
        .unwrap();
        #[cfg(target_os = "macos")]
        assert_eq!(resolved.externals.len(), 3);
        #[cfg(not(target_os = "macos"))]
        assert_eq!(resolved.externals.len(), 2);
        assert_eq!(resolved.externals[0].name, BLACKBOARD_PLUGIN_ID);
        assert_eq!(resolved.externals[1].name, BLOBSTORE_PLUGIN_ID);
        #[cfg(target_os = "macos")]
        assert_eq!(resolved.externals[2].name, MLX_PLUGIN_ID);
        assert!(resolved.inactive.is_empty());
    }

    #[test]
    fn resolves_external_plugin() {
        let config = MeshConfig {
            plugins: vec![PluginConfigEntry {
                name: "demo".into(),
                enabled: Some(true),
                command: Some("/tmp/demo".into()),
                args: vec!["--flag".into()],
            }],
            ..MeshConfig::default()
        };
        let resolved = resolve_plugins(&config, private_host_mode()).unwrap();
        #[cfg(target_os = "macos")]
        assert_eq!(resolved.externals.len(), 4);
        #[cfg(not(target_os = "macos"))]
        assert_eq!(resolved.externals.len(), 3);
        assert_eq!(resolved.externals[0].name, BLACKBOARD_PLUGIN_ID);
        assert_eq!(resolved.externals[1].name, "demo");
        assert_eq!(resolved.externals[2].name, BLOBSTORE_PLUGIN_ID);
        #[cfg(target_os = "macos")]
        assert_eq!(resolved.externals[3].name, MLX_PLUGIN_ID);
        assert!(resolved.inactive.is_empty());
    }

    #[cfg(target_os = "macos")]
    struct EnsureEndpointTestBridge;

    #[cfg(target_os = "macos")]
    impl PluginRpcBridge for EnsureEndpointTestBridge {
        fn handle_request(
            &self,
            plugin_name: String,
            method: String,
            params_json: String,
        ) -> BridgeFuture<Result<RpcResult, proto::ErrorResponse>> {
            Box::pin(async move {
                assert_eq!(plugin_name, MLX_PLUGIN_ID);
                match method.as_str() {
                    "inference/list_endpoints" => Ok(RpcResult {
                        result_json: serde_json::to_string(&vec![
                            mesh_llm_plugin::InferenceEndpointDescriptor {
                                endpoint_id: "local-mlx".into(),
                                address: None,
                                supports_streaming: true,
                                local_model_matcher:
                                    mesh_llm_plugin::InferenceLocalModelMatcher::MlxModelDir,
                                provider_capabilities:
                                    mesh_llm_plugin::InferenceProviderCapabilitiesDescriptor {
                                        supports_local_runtime: true,
                                        supports_distributed_host_runtime: false,
                                        requires_worker_runtime: false,
                                        supports_moe_shard_runtime: false,
                                    },
                            },
                        ])
                        .unwrap(),
                    }),
                    "inference/ensure_endpoint" => {
                        let request: mesh_llm_plugin::EnsureInferenceEndpointRequest =
                            serde_json::from_str(&params_json).unwrap();
                        assert_eq!(request.endpoint_id, "local-mlx");
                        Ok(RpcResult {
                            result_json: serde_json::to_string(
                                &mesh_llm_plugin::EnsureInferenceEndpointResponse {
                                    address: format!(
                                        "http://127.0.0.1:{}",
                                        request.requested_port.unwrap_or(8123)
                                    ),
                                    backend_label: "mlx".into(),
                                    context_length: 32768,
                                },
                            )
                            .unwrap(),
                        })
                    }
                    _ => Err(proto::ErrorResponse {
                        code: ErrorCode::METHOD_NOT_FOUND.0,
                        message: format!("Unsupported plugin method '{method}'"),
                        data_json: String::new(),
                    }),
                }
            })
        }

        fn handle_notification(
            &self,
            _plugin_name: String,
            _method: String,
            _params_json: String,
        ) -> BridgeFuture<()> {
            Box::pin(async move {})
        }
    }

    #[cfg(target_os = "macos")]
    #[tokio::test]
    async fn managed_inference_endpoints_include_mlx_when_available() {
        let plugin_manager =
            PluginManager::for_test_bridge(&[MLX_PLUGIN_ID], Arc::new(EnsureEndpointTestBridge));
        let endpoints = plugin_manager.managed_inference_endpoints().await.unwrap();
        assert_eq!(
            endpoints,
            vec![ManagedInferenceEndpoint {
                plugin_name: MLX_PLUGIN_ID.into(),
                endpoint_id: "local-mlx".into(),
                address: None,
                supports_streaming: true,
                local_model_matcher: mesh_llm_plugin::InferenceLocalModelMatcher::MlxModelDir,
                provider_capabilities: mesh_llm_plugin::InferenceProviderCapabilitiesDescriptor {
                    supports_local_runtime: true,
                    supports_distributed_host_runtime: false,
                    requires_worker_runtime: false,
                    supports_moe_shard_runtime: false,
                },
            }]
        );
    }

    #[cfg(target_os = "macos")]
    struct NoInferenceEndpointBridge;

    #[cfg(target_os = "macos")]
    impl PluginRpcBridge for NoInferenceEndpointBridge {
        fn handle_request(
            &self,
            _plugin_name: String,
            method: String,
            _params_json: String,
        ) -> BridgeFuture<Result<RpcResult, proto::ErrorResponse>> {
            Box::pin(async move {
                Err(proto::ErrorResponse {
                    code: ErrorCode::METHOD_NOT_FOUND.0,
                    message: format!("Unsupported plugin method '{method}'"),
                    data_json: String::new(),
                })
            })
        }

        fn handle_notification(
            &self,
            _plugin_name: String,
            _method: String,
            _params_json: String,
        ) -> BridgeFuture<()> {
            Box::pin(async move {})
        }
    }

    #[cfg(target_os = "macos")]
    #[tokio::test]
    async fn managed_inference_endpoints_skip_plugins_without_inference_support() {
        let plugin_manager =
            PluginManager::for_test_bridge(&["demo"], Arc::new(NoInferenceEndpointBridge));
        let endpoints = plugin_manager.managed_inference_endpoints().await.unwrap();
        assert!(endpoints.is_empty());
    }

    #[cfg(target_os = "macos")]
    #[tokio::test]
    async fn ensure_managed_inference_endpoint_routes_through_plugin_bridge() {
        let plugin_manager =
            PluginManager::for_test_bridge(&[MLX_PLUGIN_ID], Arc::new(EnsureEndpointTestBridge));
        let response = plugin_manager
            .ensure_managed_inference_endpoint(
                MLX_PLUGIN_ID,
                "local-mlx",
                std::path::Path::new("/tmp/model"),
                Some(8123),
                Some(32768),
            )
            .await
            .unwrap();
        assert_eq!(response.address, "http://127.0.0.1:8123");
        assert_eq!(response.backend_label, "mlx");
        assert_eq!(response.context_length, 32768);
    }

    #[test]
    fn instance_ids_include_pid_and_random_suffix() {
        let instance_id = make_instance_id();
        let prefix = format!("p{}-", std::process::id());
        assert!(instance_id.starts_with(&prefix));
        assert_eq!(instance_id.len(), prefix.len() + 8);
        assert!(instance_id[prefix.len()..]
            .chars()
            .all(|ch| ch.is_ascii_hexdigit()));
    }

    #[cfg(unix)]
    #[test]
    fn unix_socket_path_is_namespaced_by_instance_id() {
        let path = unix_socket_path("p1234-deadbeef", "Pipes").unwrap();
        assert_eq!(
            path.file_name().and_then(|value| value.to_str()),
            Some("p1234-deadbeef-Pipes.sock")
        );
    }

    #[cfg(windows)]
    #[test]
    fn windows_pipe_name_is_namespaced_by_instance_id() {
        assert_eq!(
            windows_pipe_name("p1234-deadbeef", "Pipes"),
            r"\\.\pipe\mesh-llm-p1234-deadbeef-Pipes"
        );
    }
}
