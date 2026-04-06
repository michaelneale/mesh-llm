mod config;
pub(crate) mod mcp;
mod runtime;
pub(crate) mod stapler;
mod support;
mod transport;

use anyhow::{anyhow, bail, Context, Result};
pub use mesh_llm_plugin::proto;
use rmcp::model::ServerInfo;
use rmcp::model::{
    CompleteRequestParams, CompleteResult, GetPromptRequestParams, GetPromptResult,
    ReadResourceRequestParams, ReadResourceResult,
};
use serde::Serialize;
use serde_json::{json, Value};
use std::collections::BTreeMap;
#[cfg(test)]
use std::collections::BTreeSet;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, Mutex};
use url::Url;

#[allow(unused_imports)]
pub use self::config::ExternalPluginSpec;
#[allow(unused_imports)]
pub use self::config::{
    config_path, load_config, resolve_plugins, MeshConfig, ModelConfigEntry, PluginHostMode,
    ResolvedPlugins,
};
use self::runtime::ExternalPlugin;
pub(crate) use self::support::parse_optional_json;
use self::support::{format_args_for_log, format_slice_for_log, format_tool_names_for_log};
#[cfg(all(test, unix))]
use self::transport::unix_socket_path;
#[cfg(all(test, windows))]
use self::transport::windows_pipe_name;
use self::transport::{connect_side_stream, make_instance_id, LocalStream};
#[cfg(test)]
use mesh_llm_plugin::MeshVisibility;

pub const BLACKBOARD_PLUGIN_ID: &str = "blackboard";
pub const BLOBSTORE_PLUGIN_ID: &str = "blobstore";
pub const LEMONADE_PLUGIN_ID: &str = "lemonade";
pub const BLACKBOARD_CAPABILITY: &str = "blackboard.v1";
pub(crate) const PROTOCOL_VERSION: u32 = mesh_llm_plugin::PROTOCOL_VERSION;
const CONNECT_TIMEOUT_SECS: u64 = 10;
const REQUEST_TIMEOUT_SECS: u64 = 30;
const HEALTH_CHECK_INTERVAL_SECS: u64 = 15;
const ENDPOINT_STARTUP_GRACE_SECS: u64 = 30;
const ENDPOINT_FAILURE_THRESHOLD: u32 = 2;

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

pub(crate) type BridgeFuture<T> = Pin<Box<dyn Future<Output = T> + Send>>;
#[cfg(test)]
type TestStreamFuture = Pin<Box<dyn Future<Output = Result<LocalStream>> + Send>>;
#[cfg(test)]
type TestStreamHandler = Arc<dyn Fn(proto::OpenStreamRequest) -> TestStreamFuture + Send + Sync>;

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
    pub manifest: Option<PluginManifestOverview>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct PluginManifestOverview {
    pub operations: usize,
    pub resources: usize,
    pub resource_templates: usize,
    pub prompts: usize,
    pub completions: usize,
    pub http_bindings: usize,
    pub endpoints: usize,
    pub mesh_channels: usize,
    pub mesh_event_subscriptions: usize,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub capabilities: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct PluginEndpointSummary {
    pub plugin_name: String,
    pub plugin_status: String,
    pub endpoint_id: String,
    pub state: String,
    pub available: bool,
    pub kind: String,
    pub transport_kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub protocol: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub address: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub args: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub namespace: Option<String>,
    pub supports_streaming: bool,
    pub managed_by_plugin: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub models: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct PluginCapabilityProvider {
    pub capability: String,
    pub plugin_name: String,
    pub plugin_status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub endpoint_id: Option<String>,
    pub available: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct EndpointHealthRecord {
    state: String,
    available: bool,
    detail: Option<String>,
    models: Vec<String>,
}

#[derive(Clone, Debug)]
struct EndpointHealthState {
    record: EndpointHealthRecord,
    first_checked_at: Instant,
    consecutive_failures: u32,
}

#[derive(Clone, Debug)]
pub struct InferenceEndpointRoute {
    pub plugin_name: String,
    pub endpoint_id: String,
    pub address: String,
    pub models: Vec<String>,
}

#[derive(Clone)]
pub struct PluginManager {
    inner: Arc<PluginManagerInner>,
}

struct PluginManagerInner {
    plugins: BTreeMap<String, ExternalPlugin>,
    inactive: BTreeMap<String, PluginSummary>,
    endpoint_health: Arc<Mutex<BTreeMap<String, EndpointHealthState>>>,
    rpc_bridge: Arc<Mutex<Option<Arc<dyn PluginRpcBridge>>>>,
    #[cfg(test)]
    bridged_plugins: BTreeSet<String>,
    #[cfg(test)]
    test_endpoints: Arc<Mutex<Vec<PluginEndpointSummary>>>,
    #[cfg(test)]
    test_inference_endpoints: Arc<Mutex<Vec<InferenceEndpointRoute>>>,
    #[cfg(test)]
    test_manifests: Arc<Mutex<BTreeMap<String, proto::PluginManifest>>>,
    #[cfg(test)]
    test_stream_handlers: Arc<Mutex<BTreeMap<String, TestStreamHandler>>>,
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
                endpoint_health: Arc::new(Mutex::new(BTreeMap::new())),
                rpc_bridge,
                #[cfg(test)]
                bridged_plugins: BTreeSet::new(),
                #[cfg(test)]
                test_endpoints: Arc::new(Mutex::new(Vec::new())),
                #[cfg(test)]
                test_inference_endpoints: Arc::new(Mutex::new(Vec::new())),
                #[cfg(test)]
                test_manifests: Arc::new(Mutex::new(BTreeMap::new())),
                #[cfg(test)]
                test_stream_handlers: Arc::new(Mutex::new(BTreeMap::new())),
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
                endpoint_health: Arc::new(Mutex::new(BTreeMap::new())),
                rpc_bridge: Arc::new(Mutex::new(Some(bridge))),
                bridged_plugins: plugin_names
                    .iter()
                    .map(|name| (*name).to_string())
                    .collect(),
                test_endpoints: Arc::new(Mutex::new(Vec::new())),
                test_inference_endpoints: Arc::new(Mutex::new(Vec::new())),
                test_manifests: Arc::new(Mutex::new(BTreeMap::new())),
                test_stream_handlers: Arc::new(Mutex::new(BTreeMap::new())),
            }),
        }
    }

    pub async fn list(&self) -> Vec<PluginSummary> {
        #[cfg(test)]
        if self.inner.plugins.is_empty() && self.inner.inactive.is_empty() {
            let manifests = self.inner.test_manifests.lock().await.clone();
            if !manifests.is_empty() {
                let mut summaries = manifests
                    .into_iter()
                    .map(|(name, manifest)| PluginSummary {
                        name,
                        kind: "bridge".into(),
                        enabled: true,
                        status: "running".into(),
                        version: None,
                        capabilities: manifest.capabilities.clone(),
                        command: None,
                        args: Vec::new(),
                        tools: Vec::new(),
                        manifest: Some(plugin_manifest_overview(&manifest)),
                        error: None,
                    })
                    .collect::<Vec<_>>();
                summaries.sort_by(|a, b| a.name.cmp(&b.name));
                return summaries;
            }
        }
        let mut summaries =
            Vec::with_capacity(self.inner.plugins.len() + self.inner.inactive.len());
        for plugin in self.inner.plugins.values() {
            summaries.push(plugin.summary().await);
        }
        summaries.extend(self.inner.inactive.values().cloned());
        summaries.sort_by(|a, b| a.name.cmp(&b.name));
        summaries
    }

    pub async fn endpoints(&self) -> Result<Vec<PluginEndpointSummary>> {
        #[cfg(test)]
        if self.inner.plugins.is_empty() && self.inner.inactive.is_empty() {
            let mut endpoints = self.inner.test_endpoints.lock().await.clone();
            endpoints.sort_by(|a, b| {
                a.plugin_name
                    .cmp(&b.plugin_name)
                    .then_with(|| a.endpoint_id.cmp(&b.endpoint_id))
            });
            if !endpoints.is_empty() {
                return Ok(endpoints);
            }
        }
        let summaries = self.list().await;
        let endpoint_health = self.inner.endpoint_health.lock().await.clone();
        let mut endpoints = Vec::new();
        for summary in summaries {
            let Ok(Some(manifest)) = self.manifest(&summary.name).await else {
                continue;
            };
            for endpoint in manifest.endpoints {
                let health = endpoint_health
                    .get(&endpoint_key(&summary.name, &endpoint.endpoint_id))
                    .map(|state| state.record.clone())
                    .unwrap_or_else(|| endpoint_record_from_plugin_status(&summary));
                endpoints.push(PluginEndpointSummary {
                    plugin_name: summary.name.clone(),
                    plugin_status: summary.status.clone(),
                    endpoint_id: endpoint.endpoint_id,
                    state: health.state,
                    available: health.available,
                    kind: endpoint_kind_name(endpoint.kind).to_string(),
                    transport_kind: endpoint_transport_kind_name(endpoint.transport_kind)
                        .to_string(),
                    protocol: endpoint.protocol,
                    address: endpoint.address,
                    args: endpoint.args,
                    namespace: endpoint.namespace,
                    supports_streaming: endpoint.supports_streaming,
                    managed_by_plugin: endpoint.managed_by_plugin,
                    detail: health.detail,
                    models: health.models,
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

    #[cfg(test)]
    pub async fn set_test_endpoints(&self, endpoints: Vec<PluginEndpointSummary>) {
        *self.inner.test_endpoints.lock().await = endpoints;
    }

    #[cfg(test)]
    pub async fn set_test_inference_endpoints(&self, endpoints: Vec<InferenceEndpointRoute>) {
        *self.inner.test_inference_endpoints.lock().await = endpoints;
    }

    #[cfg(test)]
    pub async fn set_test_manifests(&self, manifests: BTreeMap<String, proto::PluginManifest>) {
        *self.inner.test_manifests.lock().await = manifests;
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
        if self.is_test_bridge_enabled(plugin_name) {
            let bridge = self
                .inner
                .rpc_bridge
                .lock()
                .await
                .clone()
                .with_context(|| format!("No bridge configured for test plugin '{plugin_name}'"))?;
            let arguments = parse_optional_json(arguments_json)?;
            let params_json = serde_json::to_string(&serde_json::json!({
                "name": tool_name,
                "arguments": arguments,
            }))
            .with_context(|| format!("Serialize tool call for test plugin '{plugin_name}'"))?;
            let result = bridge
                .handle_request(plugin_name.to_string(), "tools/call".into(), params_json)
                .await
                .map_err(|err| anyhow!("{}", err.message))?;
            let decoded: rmcp::model::CallToolResult = serde_json::from_str(&result.result_json)
                .with_context(|| format!("Decode tool result from test plugin '{plugin_name}'"))?;
            return Ok(ToolCallResult {
                content_json: normalize_test_tool_result_content(&decoded)?,
                is_error: decoded.is_error.unwrap_or(false),
            });
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
        plugin.call_tool(tool_name, arguments_json).await
    }

    pub async fn invoke_operation(
        &self,
        plugin_name: &str,
        operation_name: &str,
        input_json: &str,
    ) -> Result<ToolCallResult> {
        self.call_tool(plugin_name, operation_name, input_json)
            .await
    }

    pub async fn inference_models(&self) -> Result<Vec<String>> {
        let mut models = Vec::new();
        for endpoint in self.inference_endpoints().await? {
            models.extend(endpoint.models);
        }
        models.sort();
        models.dedup();
        Ok(models)
    }

    pub async fn inference_endpoint_for_model(
        &self,
        model: &str,
    ) -> Result<Option<InferenceEndpointRoute>> {
        let mut endpoints = self.inference_endpoints().await?;
        endpoints.sort_by(|a, b| {
            a.plugin_name
                .cmp(&b.plugin_name)
                .then_with(|| a.endpoint_id.cmp(&b.endpoint_id))
        });
        Ok(endpoints
            .into_iter()
            .find(|endpoint| endpoint.models.iter().any(|candidate| candidate == model)))
    }

    pub async fn capability_providers(&self) -> Result<Vec<PluginCapabilityProvider>> {
        let summaries = self.list().await;
        let endpoint_health = self.inner.endpoint_health.lock().await.clone();
        let mut providers = Vec::new();
        for summary in summaries {
            let Ok(Some(manifest)) = self.manifest(&summary.name).await else {
                continue;
            };

            let plugin_default = endpoint_record_from_plugin_status(&summary);
            for capability in &manifest.capabilities {
                providers.push(PluginCapabilityProvider {
                    capability: capability.clone(),
                    plugin_name: summary.name.clone(),
                    plugin_status: summary.status.clone(),
                    endpoint_id: None,
                    available: plugin_default.available,
                    detail: plugin_default.detail.clone(),
                });
            }

            for endpoint in &manifest.endpoints {
                let health = endpoint_health
                    .get(&endpoint_key(&summary.name, &endpoint.endpoint_id))
                    .map(|state| state.record.clone())
                    .unwrap_or_else(|| endpoint_record_from_plugin_status(&summary));
                let endpoint_capabilities = endpoint_declared_capabilities(endpoint);
                for capability in endpoint_capabilities {
                    providers.push(PluginCapabilityProvider {
                        capability,
                        plugin_name: summary.name.clone(),
                        plugin_status: summary.status.clone(),
                        endpoint_id: Some(endpoint.endpoint_id.clone()),
                        available: health.available,
                        detail: health.detail.clone(),
                    });
                }
            }
        }
        providers.sort_by(|a, b| {
            a.capability
                .cmp(&b.capability)
                .then_with(|| a.plugin_name.cmp(&b.plugin_name))
                .then_with(|| a.endpoint_id.cmp(&b.endpoint_id))
        });
        Ok(providers)
    }

    pub async fn provider_for_capability(
        &self,
        capability: &str,
    ) -> Result<Option<PluginCapabilityProvider>> {
        let mut providers = self.capability_providers().await?;
        providers.sort_by(|a, b| {
            b.available
                .cmp(&a.available)
                .then_with(|| a.plugin_name.cmp(&b.plugin_name))
                .then_with(|| a.endpoint_id.cmp(&b.endpoint_id))
        });
        Ok(providers
            .into_iter()
            .find(|provider| provider.capability == capability))
    }

    pub async fn available_provider_for_capability(
        &self,
        capability: &str,
    ) -> Result<Option<PluginCapabilityProvider>> {
        Ok(self
            .provider_for_capability(capability)
            .await?
            .filter(|provider| provider.available))
    }

    pub async fn is_capability_available(&self, capability: &str) -> bool {
        self.available_provider_for_capability(capability)
            .await
            .ok()
            .flatten()
            .is_some()
    }

    pub async fn invoke_operation_by_capability(
        &self,
        capability: &str,
        operation_name: &str,
        input_json: &str,
    ) -> Result<ToolCallResult> {
        let provider = self
            .available_provider_for_capability(capability)
            .await?
            .ok_or_else(|| anyhow!("No provider for capability '{capability}'"))?;
        self.invoke_operation(&provider.plugin_name, operation_name, input_json)
            .await
    }

    pub async fn get_prompt(
        &self,
        plugin_name: &str,
        prompt_name: &str,
        params: GetPromptRequestParams,
    ) -> Result<GetPromptResult> {
        self.invoke_service_json(
            plugin_name,
            proto::ServiceKind::Prompt,
            prompt_name,
            &params,
        )
        .await
    }

    pub async fn read_resource(
        &self,
        plugin_name: &str,
        resource_uri: &str,
        params: ReadResourceRequestParams,
    ) -> Result<ReadResourceResult> {
        self.invoke_service_json(
            plugin_name,
            proto::ServiceKind::Resource,
            resource_uri,
            &params,
        )
        .await
    }

    pub async fn complete(
        &self,
        plugin_name: &str,
        argument_ref: &str,
        params: CompleteRequestParams,
    ) -> Result<CompleteResult> {
        self.invoke_service_json(
            plugin_name,
            proto::ServiceKind::Completion,
            argument_ref,
            &params,
        )
        .await
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
                .map_err(|err| anyhow!("{}", err.message))?;
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

    async fn invoke_service_json<T, P>(
        &self,
        plugin_name: &str,
        kind: proto::ServiceKind,
        service_name: &str,
        params: &P,
    ) -> Result<T>
    where
        T: serde::de::DeserializeOwned,
        P: Serialize,
    {
        if self.is_test_bridge_enabled(plugin_name) {
            let method = match kind {
                proto::ServiceKind::Operation => "tools/call",
                proto::ServiceKind::Prompt => "prompts/get",
                proto::ServiceKind::Resource => "resources/read",
                proto::ServiceKind::Completion => "completion/complete",
                proto::ServiceKind::Unspecified => {
                    bail!("Service kind is required for test plugin '{plugin_name}'")
                }
            };
            if method == "tools/call" {
                let arguments = serde_json::to_value(params).with_context(|| {
                    format!("Serialize operation params for test plugin '{plugin_name}'")
                })?;
                return self
                    .mcp_request(
                        plugin_name,
                        method,
                        serde_json::json!({
                            "name": service_name,
                            "arguments": arguments,
                        }),
                    )
                    .await;
            }
            return self.mcp_request(plugin_name, method, params).await;
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
        let input_json = serde_json::to_string(params)
            .with_context(|| format!("Serialize service params for plugin '{plugin_name}'"))?;
        let response = plugin
            .invoke_service(kind, service_name, &input_json)
            .await?;
        serde_json::from_str(&response.output_json).with_context(|| {
            format!(
                "Decode service response '{}' from plugin '{}'",
                service_name, plugin_name
            )
        })
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

    pub async fn manifest(&self, plugin_name: &str) -> Result<Option<proto::PluginManifest>> {
        if self.is_test_bridge_enabled(plugin_name) {
            #[cfg(test)]
            if let Some(manifest) = self
                .inner
                .test_manifests
                .lock()
                .await
                .get(plugin_name)
                .cloned()
            {
                return Ok(Some(manifest));
            }
            bail!(
                "Plugin '{}' does not expose a manifest in bridge mode",
                plugin_name
            );
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
        plugin.manifest().await
    }

    pub async fn manifest_json(&self, plugin_name: &str) -> Result<Option<Value>> {
        Ok(self
            .manifest(plugin_name)
            .await?
            .as_ref()
            .map(plugin_manifest_to_json))
    }

    pub async fn set_rpc_bridge(&self, bridge: Option<Arc<dyn PluginRpcBridge>>) {
        *self.inner.rpc_bridge.lock().await = bridge;
    }

    #[cfg(test)]
    pub async fn set_test_stream_handler<F>(&self, plugin_name: &str, handler: F)
    where
        F: Fn(proto::OpenStreamRequest) -> TestStreamFuture + Send + Sync + 'static,
    {
        self.inner
            .test_stream_handlers
            .lock()
            .await
            .insert(plugin_name.to_string(), Arc::new(handler));
    }

    pub async fn dispatch_channel_message(&self, event: PluginMeshEvent) -> Result<()> {
        let PluginMeshEvent::Channel { plugin_id, message } = event else {
            bail!("expected plugin channel event");
        };
        if !self
            .plugin_declares_mesh_channel(&plugin_id, &message.channel)
            .await
        {
            tracing::debug!(
                plugin = %plugin_id,
                channel = %message.channel,
                "Dropping channel message for undeclared mesh channel"
            );
            return Ok(());
        }
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
        if !self
            .plugin_declares_mesh_channel(&plugin_id, &message.channel)
            .await
        {
            tracing::debug!(
                plugin = %plugin_id,
                channel = %message.channel,
                "Dropping bulk transfer for undeclared mesh channel"
            );
            return Ok(());
        }
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
        for (name, plugin) in &self.inner.plugins {
            if !self.plugin_subscribes_mesh_event(name, event.kind).await {
                continue;
            }
            plugin.send_mesh_event(event.clone()).await?;
        }
        Ok(())
    }

    pub async fn plugin_declares_mesh_channel(&self, plugin_name: &str, channel: &str) -> bool {
        self.manifest(plugin_name)
            .await
            .ok()
            .flatten()
            .is_some_and(|manifest| manifest_declares_mesh_channel(&manifest, channel))
    }

    pub async fn plugin_subscribes_mesh_event(&self, plugin_name: &str, kind: i32) -> bool {
        self.manifest(plugin_name)
            .await
            .ok()
            .flatten()
            .is_some_and(|manifest| manifest_subscribes_mesh_event(&manifest, kind))
    }

    pub async fn open_stream(
        &self,
        plugin_name: &str,
        request: proto::OpenStreamRequest,
    ) -> Result<proto::OpenStreamResponse> {
        if self.is_test_bridge_enabled(plugin_name) {
            bail!(
                "Plugin '{}' does not support stream control in bridge mode",
                plugin_name
            );
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
        plugin.open_stream(request).await
    }

    pub(crate) async fn connect_stream(
        &self,
        plugin_name: &str,
        request: proto::OpenStreamRequest,
    ) -> Result<LocalStream> {
        #[cfg(test)]
        if let Some(handler) = self
            .inner
            .test_stream_handlers
            .lock()
            .await
            .get(plugin_name)
            .cloned()
        {
            return handler(request).await;
        }
        let response = self.open_stream(plugin_name, request).await?;
        if !response.accepted {
            bail!(
                "Plugin '{}' rejected stream request: {}",
                plugin_name,
                response.message.as_deref().unwrap_or("no reason provided")
            );
        }
        let endpoint = response.endpoint.as_deref().with_context(|| {
            format!(
                "Plugin '{}' accepted stream request without an endpoint",
                plugin_name
            )
        })?;
        connect_side_stream(endpoint, response.transport_kind).await
    }

    fn start_supervisor(&self) {
        let manager = self.clone();
        tokio::spawn(async move {
            let mut ticker =
                tokio::time::interval(std::time::Duration::from_secs(HEALTH_CHECK_INTERVAL_SECS));
            loop {
                ticker.tick().await;
                let plugin_names = manager.inner.plugins.keys().cloned().collect::<Vec<_>>();
                for plugin_name in plugin_names {
                    let Some(plugin) = manager.inner.plugins.get(&plugin_name) else {
                        continue;
                    };
                    if let Err(err) = plugin.supervise().await {
                        tracing::warn!(
                            plugin = %plugin.name(),
                            error = %err,
                            "Plugin supervision round failed"
                        );
                    }
                    if let Err(err) = manager.refresh_plugin_endpoints(&plugin_name).await {
                        tracing::warn!(
                            plugin = %plugin_name,
                            error = %err,
                            "Endpoint supervision round failed"
                        );
                    }
                }
            }
        });
    }

    async fn refresh_plugin_endpoints(&self, plugin_name: &str) -> Result<()> {
        let summary = if let Some(plugin) = self.inner.plugins.get(plugin_name) {
            plugin.summary().await
        } else if let Some(summary) = self.inner.inactive.get(plugin_name) {
            summary.clone()
        } else {
            self.clear_plugin_endpoint_health(plugin_name).await;
            return Ok(());
        };

        let manifest = self.manifest(plugin_name).await.ok().flatten();
        let Some(manifest) = manifest else {
            self.clear_plugin_endpoint_health(plugin_name).await;
            return Ok(());
        };

        let now = Instant::now();
        let prefix = format!("{plugin_name}:");
        let previous = self
            .inner
            .endpoint_health
            .lock()
            .await
            .iter()
            .filter_map(|(key, value)| {
                key.strip_prefix(&prefix)
                    .map(|endpoint_id| (endpoint_id.to_string(), value.clone()))
            })
            .collect::<BTreeMap<_, _>>();
        let mut endpoint_states = BTreeMap::new();
        for endpoint in manifest.endpoints {
            let key = endpoint.endpoint_id.clone();
            let health =
                endpoint_health_for_summary(&summary, &endpoint, previous.get(&key), now).await;
            endpoint_states.insert(endpoint_key(plugin_name, &key), health);
        }

        let mut registry = self.inner.endpoint_health.lock().await;
        registry.retain(|key, _| !key.starts_with(&format!("{plugin_name}:")));
        registry.extend(endpoint_states);
        Ok(())
    }

    async fn clear_plugin_endpoint_health(&self, plugin_name: &str) {
        let mut registry = self.inner.endpoint_health.lock().await;
        registry.retain(|key, _| !key.starts_with(&format!("{plugin_name}:")));
    }

    async fn inference_endpoints(&self) -> Result<Vec<InferenceEndpointRoute>> {
        #[cfg(test)]
        if self.inner.plugins.is_empty() && self.inner.inactive.is_empty() {
            let mut endpoints = self.inner.test_inference_endpoints.lock().await.clone();
            endpoints.sort_by(|a, b| {
                a.plugin_name
                    .cmp(&b.plugin_name)
                    .then_with(|| a.endpoint_id.cmp(&b.endpoint_id))
            });
            if !endpoints.is_empty() {
                return Ok(endpoints);
            }
        }
        let summaries = self.list().await;
        let endpoint_health = self.inner.endpoint_health.lock().await.clone();
        let mut endpoints = Vec::new();
        for summary in summaries {
            let Ok(Some(manifest)) = self.manifest(&summary.name).await else {
                continue;
            };
            for endpoint in manifest.endpoints {
                if proto::EndpointKind::try_from(endpoint.kind)
                    .unwrap_or(proto::EndpointKind::Unspecified)
                    != proto::EndpointKind::Inference
                {
                    continue;
                }
                let Some(address) = endpoint.address.clone() else {
                    continue;
                };
                let health = endpoint_health
                    .get(&endpoint_key(&summary.name, &endpoint.endpoint_id))
                    .map(|state| state.record.clone())
                    .unwrap_or_else(|| endpoint_record_from_plugin_status(&summary));
                if !health.available {
                    continue;
                }
                endpoints.push(InferenceEndpointRoute {
                    plugin_name: summary.name.clone(),
                    endpoint_id: endpoint.endpoint_id,
                    address,
                    models: health.models,
                });
            }
        }
        Ok(endpoints)
    }
}

#[cfg(test)]
pub(crate) async fn connect_test_side_stream(
    endpoint: &str,
    transport_kind: i32,
) -> Result<LocalStream> {
    connect_side_stream(endpoint, transport_kind).await
}

pub(crate) fn plugin_manifest_overview(manifest: &proto::PluginManifest) -> PluginManifestOverview {
    PluginManifestOverview {
        operations: manifest.operations.len(),
        resources: manifest.resources.len(),
        resource_templates: manifest.resource_templates.len(),
        prompts: manifest.prompts.len(),
        completions: manifest.completions.len(),
        http_bindings: manifest.http_bindings.len(),
        endpoints: manifest.endpoints.len(),
        mesh_channels: manifest.mesh_channels.len(),
        mesh_event_subscriptions: manifest.mesh_event_subscriptions.len(),
        capabilities: manifest.capabilities.clone(),
    }
}

pub(crate) fn plugin_manifest_to_json(manifest: &proto::PluginManifest) -> Value {
    json!({
        "operations": manifest.operations.iter().map(|operation| {
            json!({
                "name": operation.name,
                "description": operation.description,
                "input_schema_json": operation.input_schema_json,
                "output_schema_json": operation.output_schema_json,
                "title": operation.title,
            })
        }).collect::<Vec<_>>(),
        "resources": manifest.resources.iter().map(|resource| {
            json!({
                "uri": resource.uri,
                "name": resource.name,
                "description": resource.description,
                "mime_type": resource.mime_type,
            })
        }).collect::<Vec<_>>(),
        "resource_templates": manifest.resource_templates.iter().map(|resource| {
            json!({
                "uri_template": resource.uri_template,
                "name": resource.name,
                "description": resource.description,
                "mime_type": resource.mime_type,
            })
        }).collect::<Vec<_>>(),
        "prompts": manifest.prompts.iter().map(|prompt| {
            json!({
                "name": prompt.name,
                "description": prompt.description,
            })
        }).collect::<Vec<_>>(),
        "completions": manifest.completions.iter().map(|completion| {
            json!({
                "argument_ref": completion.argument_ref,
                "description": completion.description,
            })
        }).collect::<Vec<_>>(),
        "http_bindings": manifest.http_bindings.iter().map(|binding| {
            json!({
                "binding_id": binding.binding_id,
                "method": http_method_name(binding.method),
                "path": binding.path,
                "operation_name": binding.operation_name,
                "request_body_mode": http_body_mode_name(binding.request_body_mode),
                "response_body_mode": http_body_mode_name(binding.response_body_mode),
                "request_schema_json": binding.request_schema_json,
                "response_schema_json": binding.response_schema_json,
            })
        }).collect::<Vec<_>>(),
        "endpoints": manifest.endpoints.iter().map(|endpoint| {
            json!({
                "endpoint_id": endpoint.endpoint_id,
                "kind": endpoint_kind_name(endpoint.kind),
                "transport_kind": endpoint_transport_kind_name(endpoint.transport_kind),
                "protocol": endpoint.protocol,
                "address": endpoint.address,
                "args": endpoint.args,
                "namespace": endpoint.namespace,
                "supports_streaming": endpoint.supports_streaming,
                "managed_by_plugin": endpoint.managed_by_plugin,
            })
        }).collect::<Vec<_>>(),
        "mesh_channels": manifest.mesh_channels.iter().map(|channel| {
            json!({
                "name": channel.name,
            })
        }).collect::<Vec<_>>(),
        "mesh_event_subscriptions": manifest.mesh_event_subscriptions.iter().map(|subscription| {
            json!({
                "kind": mesh_event_kind_name(subscription.kind),
            })
        }).collect::<Vec<_>>(),
        "capabilities": manifest.capabilities,
    })
}

fn manifest_declares_mesh_channel(manifest: &proto::PluginManifest, channel: &str) -> bool {
    manifest
        .mesh_channels
        .iter()
        .any(|entry| entry.name == channel)
}

fn manifest_subscribes_mesh_event(manifest: &proto::PluginManifest, kind: i32) -> bool {
    manifest
        .mesh_event_subscriptions
        .iter()
        .any(|entry| entry.kind == kind)
}

fn http_method_name(value: i32) -> &'static str {
    match proto::HttpMethod::try_from(value).unwrap_or(proto::HttpMethod::Unspecified) {
        proto::HttpMethod::Get => "GET",
        proto::HttpMethod::Post => "POST",
        proto::HttpMethod::Put => "PUT",
        proto::HttpMethod::Patch => "PATCH",
        proto::HttpMethod::Delete => "DELETE",
        proto::HttpMethod::Unspecified => "UNSPECIFIED",
    }
}

fn http_body_mode_name(value: i32) -> &'static str {
    match proto::HttpBodyMode::try_from(value).unwrap_or(proto::HttpBodyMode::Unspecified) {
        proto::HttpBodyMode::Buffered => "buffered",
        proto::HttpBodyMode::Streamed => "streamed",
        proto::HttpBodyMode::Unspecified => "unspecified",
    }
}

fn mesh_event_kind_name(value: i32) -> &'static str {
    match proto::mesh_event::Kind::try_from(value).unwrap_or(proto::mesh_event::Kind::Unspecified) {
        proto::mesh_event::Kind::PeerUp => "peer_up",
        proto::mesh_event::Kind::PeerDown => "peer_down",
        proto::mesh_event::Kind::PeerUpdated => "peer_updated",
        proto::mesh_event::Kind::LocalAccepting => "local_accepting",
        proto::mesh_event::Kind::LocalStandby => "local_standby",
        proto::mesh_event::Kind::MeshIdUpdated => "mesh_id_updated",
        proto::mesh_event::Kind::Unspecified => "unspecified",
    }
}

fn endpoint_kind_name(value: i32) -> &'static str {
    match proto::EndpointKind::try_from(value).unwrap_or(proto::EndpointKind::Unspecified) {
        proto::EndpointKind::Inference => "inference",
        proto::EndpointKind::Mcp => "mcp",
        proto::EndpointKind::Unspecified => "unspecified",
    }
}

fn endpoint_transport_kind_name(value: i32) -> &'static str {
    match proto::EndpointTransportKind::try_from(value)
        .unwrap_or(proto::EndpointTransportKind::Unspecified)
    {
        proto::EndpointTransportKind::EndpointTransportHttp => "http",
        proto::EndpointTransportKind::EndpointTransportUnixSocket => "unix_socket",
        proto::EndpointTransportKind::EndpointTransportStdio => "stdio",
        proto::EndpointTransportKind::EndpointTransportNamedPipe => "named_pipe",
        proto::EndpointTransportKind::EndpointTransportTcp => "tcp",
        proto::EndpointTransportKind::Unspecified => "unspecified",
    }
}

fn endpoint_record_from_plugin_status(summary: &PluginSummary) -> EndpointHealthRecord {
    if !summary.enabled || summary.status == "disabled" {
        return EndpointHealthRecord {
            state: "unavailable".into(),
            available: false,
            detail: summary.error.clone(),
            models: Vec::new(),
        };
    }

    match summary.status.as_str() {
        "running" => EndpointHealthRecord {
            state: "healthy".into(),
            available: true,
            detail: None,
            models: Vec::new(),
        },
        "starting" | "restarting" => EndpointHealthRecord {
            state: "starting".into(),
            available: false,
            detail: summary.error.clone(),
            models: Vec::new(),
        },
        "degraded" => EndpointHealthRecord {
            state: "unhealthy".into(),
            available: false,
            detail: summary.error.clone(),
            models: Vec::new(),
        },
        _ => EndpointHealthRecord {
            state: "unavailable".into(),
            available: false,
            detail: summary.error.clone(),
            models: Vec::new(),
        },
    }
}

fn endpoint_state_from_plugin_status(summary: &PluginSummary, now: Instant) -> EndpointHealthState {
    EndpointHealthState {
        record: endpoint_record_from_plugin_status(summary),
        first_checked_at: now,
        consecutive_failures: 0,
    }
}

fn endpoint_key(plugin_name: &str, endpoint_id: &str) -> String {
    format!("{plugin_name}:{endpoint_id}")
}

fn endpoint_declared_capabilities(endpoint: &proto::EndpointManifest) -> Vec<String> {
    match proto::EndpointKind::try_from(endpoint.kind).unwrap_or(proto::EndpointKind::Unspecified) {
        proto::EndpointKind::Inference => {
            let mut capabilities = vec!["endpoint:inference".into()];
            if let Some(protocol) = endpoint.protocol.as_deref() {
                capabilities.push(format!("endpoint:inference/{protocol}"));
            }
            capabilities
        }
        proto::EndpointKind::Mcp => {
            let mut capabilities = vec!["endpoint:mcp".into()];
            if let Some(namespace) = endpoint.namespace.as_deref() {
                capabilities.push(format!("endpoint:mcp/{namespace}"));
            }
            capabilities
        }
        proto::EndpointKind::Unspecified => Vec::new(),
    }
}

fn normalize_test_tool_result_content(result: &rmcp::model::CallToolResult) -> Result<String> {
    if let Some(value) = &result.structured_content {
        return serde_json::to_string(value).map_err(Into::into);
    }
    if let Some(text) = result.content.first().and_then(|content| content.as_text()) {
        return Ok(text.text.clone());
    }
    serde_json::to_string(&result.content).map_err(Into::into)
}

async fn endpoint_health_for_summary(
    summary: &PluginSummary,
    endpoint: &proto::EndpointManifest,
    previous: Option<&EndpointHealthState>,
    now: Instant,
) -> EndpointHealthState {
    if summary.status != "running" {
        return endpoint_state_from_plugin_status(summary, now);
    }

    let probe = probe_endpoint(endpoint)
        .await
        .unwrap_or(EndpointHealthRecord {
            state: "healthy".into(),
            available: true,
            detail: None,
            models: Vec::new(),
        });
    apply_endpoint_probe(previous, probe, now)
}

fn apply_endpoint_probe(
    previous: Option<&EndpointHealthState>,
    probe: EndpointHealthRecord,
    now: Instant,
) -> EndpointHealthState {
    let first_checked_at = previous.map(|state| state.first_checked_at).unwrap_or(now);

    if probe.available {
        return EndpointHealthState {
            record: probe,
            first_checked_at,
            consecutive_failures: 0,
        };
    }

    let failure_streak = previous
        .map(|state| state.consecutive_failures.saturating_add(1))
        .unwrap_or(1);
    let within_startup_grace =
        now.duration_since(first_checked_at) < Duration::from_secs(ENDPOINT_STARTUP_GRACE_SECS);
    let was_available = previous
        .map(|state| state.record.available)
        .unwrap_or(false);

    let record = if !was_available && within_startup_grace {
        EndpointHealthRecord {
            state: "starting".into(),
            available: false,
            detail: probe.detail,
            models: Vec::new(),
        }
    } else if was_available && failure_streak < ENDPOINT_FAILURE_THRESHOLD {
        EndpointHealthRecord {
            state: "degraded".into(),
            available: true,
            detail: probe.detail,
            models: Vec::new(),
        }
    } else {
        EndpointHealthRecord {
            state: "unhealthy".into(),
            available: false,
            detail: probe.detail,
            models: Vec::new(),
        }
    };

    EndpointHealthState {
        record,
        first_checked_at,
        consecutive_failures: failure_streak,
    }
}

async fn probe_endpoint(endpoint: &proto::EndpointManifest) -> Option<EndpointHealthRecord> {
    match (
        proto::EndpointKind::try_from(endpoint.kind).unwrap_or(proto::EndpointKind::Unspecified),
        proto::EndpointTransportKind::try_from(endpoint.transport_kind)
            .unwrap_or(proto::EndpointTransportKind::Unspecified),
    ) {
        (proto::EndpointKind::Inference, proto::EndpointTransportKind::EndpointTransportHttp) => {
            let protocol = endpoint.protocol.as_deref().unwrap_or_default();
            if protocol.eq_ignore_ascii_case("openai_compatible") {
                return Some(
                    probe_openai_compatible_http_endpoint(endpoint.address.as_deref()?).await,
                );
            }
            None
        }
        _ => None,
    }
}

async fn probe_openai_compatible_http_endpoint(address: &str) -> EndpointHealthRecord {
    let models_url = match endpoint_models_url(address) {
        Some(url) => url,
        None => {
            return EndpointHealthRecord {
                state: "unhealthy".into(),
                available: false,
                detail: Some(format!("invalid endpoint address '{address}'")),
                models: Vec::new(),
            };
        }
    };

    let client = match reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(3))
        .build()
    {
        Ok(client) => client,
        Err(err) => {
            return EndpointHealthRecord {
                state: "unhealthy".into(),
                available: false,
                detail: Some(format!("build health probe client: {err}")),
                models: Vec::new(),
            };
        }
    };

    match client.get(models_url.clone()).send().await {
        Ok(response) if response.status().is_success() => EndpointHealthRecord {
            state: "healthy".into(),
            available: true,
            detail: Some(format!("GET {} -> {}", models_url, response.status())),
            models: parse_models_response(response).await.unwrap_or_default(),
        },
        Ok(response) => EndpointHealthRecord {
            state: "unhealthy".into(),
            available: false,
            detail: Some(format!("GET {} -> {}", models_url, response.status())),
            models: Vec::new(),
        },
        Err(err) => EndpointHealthRecord {
            state: "unhealthy".into(),
            available: false,
            detail: Some(format!("GET {} failed: {}", models_url, err)),
            models: Vec::new(),
        },
    }
}

async fn parse_models_response(response: reqwest::Response) -> Result<Vec<String>> {
    let body = response.json::<Value>().await?;
    let models = body
        .get("data")
        .and_then(|value| value.as_array())
        .into_iter()
        .flatten()
        .filter_map(|entry| entry.get("id").and_then(|id| id.as_str()))
        .map(|id| id.to_string())
        .collect::<Vec<_>>();
    Ok(models)
}

fn endpoint_models_url(address: &str) -> Option<Url> {
    let mut url = Url::parse(address).ok()?;
    let mut path = url.path().trim_end_matches('/').to_string();
    if path.is_empty() {
        path = "/v1".into();
    }
    if !path.ends_with("/models") {
        if path.ends_with("/v1") || path.ends_with("/api/v1") {
            path.push_str("/models");
        } else {
            path.push_str("/v1/models");
        }
    }
    url.set_path(&path);
    url.set_query(None);
    Some(url)
}

pub async fn run_plugin_process(name: String) -> Result<()> {
    match name.as_str() {
        BLACKBOARD_PLUGIN_ID => crate::plugins::blackboard::run_plugin(name).await,
        BLOBSTORE_PLUGIN_ID => crate::plugins::blobstore::run_plugin(name).await,
        LEMONADE_PLUGIN_ID => crate::plugins::lemonade::run_plugin(name).await,
        _ => bail!("Unknown built-in plugin '{}'", name),
    }
}

#[cfg(test)]
mod tests {
    use super::config::{MeshConfig, PluginConfigEntry};
    use super::*;
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    fn private_host_mode() -> PluginHostMode {
        PluginHostMode {
            mesh_visibility: MeshVisibility::Private,
        }
    }

    async fn spawn_fake_models_server(
        responses: Vec<(&'static str, &'static str)>,
    ) -> (String, tokio::task::JoinHandle<()>, Arc<AtomicUsize>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let requests = Arc::new(AtomicUsize::new(0));
        let requests_seen = requests.clone();
        let handle = tokio::spawn(async move {
            for (status, body) in responses {
                let (mut stream, _) = listener.accept().await.unwrap();
                let mut buf = vec![0u8; 4096];
                let _ = stream.read(&mut buf).await.unwrap();
                requests_seen.fetch_add(1, Ordering::SeqCst);
                let response = format!(
                    "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len(),
                );
                stream.write_all(response.as_bytes()).await.unwrap();
                let _ = stream.shutdown().await;
            }
        });
        (format!("http://{addr}/api/v1"), handle, requests)
    }

    #[test]
    fn resolves_default_blackboard_plugin() {
        let resolved = resolve_plugins(&MeshConfig::default(), private_host_mode()).unwrap();
        assert_eq!(resolved.externals.len(), 2);
        assert_eq!(resolved.externals[0].name, BLACKBOARD_PLUGIN_ID);
        assert_eq!(resolved.externals[1].name, BLOBSTORE_PLUGIN_ID);
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
        assert_eq!(resolved.externals.len(), 1);
        assert_eq!(resolved.externals[0].name, BLOBSTORE_PLUGIN_ID);
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
        assert_eq!(resolved.externals.len(), 1);
        assert_eq!(resolved.externals[0].name, BLACKBOARD_PLUGIN_ID);
        assert!(resolved.inactive.is_empty());
    }

    #[test]
    fn lemonade_can_be_enabled_explicitly() {
        let config = MeshConfig {
            plugins: vec![PluginConfigEntry {
                name: LEMONADE_PLUGIN_ID.into(),
                enabled: Some(true),
                command: None,
                args: Vec::new(),
            }],
            ..MeshConfig::default()
        };
        let resolved = resolve_plugins(&config, private_host_mode()).unwrap();
        assert_eq!(resolved.externals.len(), 3);
        assert_eq!(resolved.externals[0].name, BLACKBOARD_PLUGIN_ID);
        assert_eq!(resolved.externals[1].name, LEMONADE_PLUGIN_ID);
        assert_eq!(resolved.externals[2].name, BLOBSTORE_PLUGIN_ID);
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
        assert_eq!(resolved.externals.len(), 2);
        assert_eq!(resolved.externals[0].name, BLACKBOARD_PLUGIN_ID);
        assert_eq!(resolved.externals[1].name, BLOBSTORE_PLUGIN_ID);
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
        assert_eq!(resolved.externals.len(), 3);
        assert_eq!(resolved.externals[0].name, BLACKBOARD_PLUGIN_ID);
        assert_eq!(resolved.externals[1].name, "demo");
        assert_eq!(resolved.externals[2].name, BLOBSTORE_PLUGIN_ID);
        assert!(resolved.inactive.is_empty());
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

    fn running_summary() -> PluginSummary {
        PluginSummary {
            name: "demo".into(),
            kind: "external".into(),
            enabled: true,
            status: "running".into(),
            version: None,
            capabilities: Vec::new(),
            command: None,
            args: Vec::new(),
            tools: Vec::new(),
            manifest: None,
            error: None,
        }
    }

    #[test]
    fn running_plugin_endpoints_are_healthy() {
        let summary = running_summary();
        assert_eq!(
            endpoint_record_from_plugin_status(&summary),
            EndpointHealthRecord {
                state: "healthy".into(),
                available: true,
                detail: None,
                models: Vec::new(),
            }
        );
    }

    #[test]
    fn restarting_plugin_endpoints_are_not_available() {
        let summary = PluginSummary {
            status: "restarting".into(),
            error: Some("timed out".into()),
            ..running_summary()
        };
        assert_eq!(
            endpoint_record_from_plugin_status(&summary),
            EndpointHealthRecord {
                state: "starting".into(),
                available: false,
                detail: Some("timed out".into()),
                models: Vec::new(),
            }
        );
    }

    #[test]
    fn first_probe_failure_stays_in_startup_grace() {
        let now = Instant::now();
        let state = apply_endpoint_probe(
            None,
            EndpointHealthRecord {
                state: "unhealthy".into(),
                available: false,
                detail: Some("GET /models failed".into()),
                models: Vec::new(),
            },
            now,
        );
        assert_eq!(state.record.state, "starting");
        assert!(!state.record.available);
        assert_eq!(state.consecutive_failures, 1);
    }

    #[test]
    fn healthy_endpoint_degrades_before_becoming_unhealthy() {
        let now = Instant::now();
        let healthy = EndpointHealthState {
            record: EndpointHealthRecord {
                state: "healthy".into(),
                available: true,
                detail: None,
                models: vec!["demo".into()],
            },
            first_checked_at: now - Duration::from_secs(ENDPOINT_STARTUP_GRACE_SECS + 1),
            consecutive_failures: 0,
        };

        let degraded = apply_endpoint_probe(
            Some(&healthy),
            EndpointHealthRecord {
                state: "unhealthy".into(),
                available: false,
                detail: Some("503".into()),
                models: Vec::new(),
            },
            now,
        );
        assert_eq!(degraded.record.state, "degraded");
        assert!(degraded.record.available);
        assert_eq!(degraded.consecutive_failures, 1);

        let unhealthy = apply_endpoint_probe(
            Some(&degraded),
            EndpointHealthRecord {
                state: "unhealthy".into(),
                available: false,
                detail: Some("503".into()),
                models: Vec::new(),
            },
            now + Duration::from_secs(HEALTH_CHECK_INTERVAL_SECS),
        );
        assert_eq!(unhealthy.record.state, "unhealthy");
        assert!(!unhealthy.record.available);
        assert_eq!(unhealthy.consecutive_failures, 2);
    }

    #[test]
    fn unhealthy_endpoint_recovers_immediately_on_success() {
        let now = Instant::now();
        let unhealthy = EndpointHealthState {
            record: EndpointHealthRecord {
                state: "unhealthy".into(),
                available: false,
                detail: Some("503".into()),
                models: Vec::new(),
            },
            first_checked_at: now - Duration::from_secs(ENDPOINT_STARTUP_GRACE_SECS + 1),
            consecutive_failures: ENDPOINT_FAILURE_THRESHOLD,
        };

        let recovered = apply_endpoint_probe(
            Some(&unhealthy),
            EndpointHealthRecord {
                state: "healthy".into(),
                available: true,
                detail: None,
                models: vec!["demo".into()],
            },
            now,
        );
        assert_eq!(recovered.record.state, "healthy");
        assert!(recovered.record.available);
        assert_eq!(recovered.record.models, vec!["demo".to_string()]);
        assert_eq!(recovered.consecutive_failures, 0);
    }

    #[test]
    fn models_probe_url_extends_openai_v1_base() {
        let url = endpoint_models_url("http://localhost:8000/v1").unwrap();
        assert_eq!(url.as_str(), "http://localhost:8000/v1/models");
    }

    #[test]
    fn models_probe_url_extends_api_v1_base() {
        let url = endpoint_models_url("http://localhost:8000/api/v1").unwrap();
        assert_eq!(url.as_str(), "http://localhost:8000/api/v1/models");
    }

    #[tokio::test]
    async fn openai_http_endpoint_probe_extracts_models_from_fake_server() {
        let (address, handle, requests) = spawn_fake_models_server(vec![(
            "200 OK",
            r#"{"data":[{"id":"lemonade-small"},{"id":"lemonade-large"}]}"#,
        )])
        .await;

        let health = probe_openai_compatible_http_endpoint(&address).await;
        assert!(health.available);
        assert_eq!(health.state, "healthy");
        assert_eq!(
            health.models,
            vec!["lemonade-small".to_string(), "lemonade-large".to_string()]
        );
        assert_eq!(requests.load(Ordering::SeqCst), 1);

        handle.await.unwrap();
    }

    #[tokio::test]
    async fn openai_http_endpoint_probe_marks_503_unavailable() {
        let (address, handle, requests) =
            spawn_fake_models_server(vec![("503 Service Unavailable", r#"{"error":"warming"}"#)])
                .await;

        let health = probe_openai_compatible_http_endpoint(&address).await;
        assert!(!health.available);
        assert_eq!(health.state, "unhealthy");
        assert!(health.models.is_empty());
        assert!(health
            .detail
            .as_deref()
            .unwrap_or_default()
            .contains("503 Service Unavailable"));
        assert_eq!(requests.load(Ordering::SeqCst), 1);

        handle.await.unwrap();
    }

    #[tokio::test]
    async fn openai_http_endpoint_probe_recovers_when_fake_server_recovers() {
        let (address, handle, requests) = spawn_fake_models_server(vec![
            ("503 Service Unavailable", r#"{"error":"warming"}"#),
            ("200 OK", r#"{"data":[{"id":"lemonade-recovered"}]}"#),
        ])
        .await;

        let first = probe_openai_compatible_http_endpoint(&address).await;
        assert!(!first.available);
        assert_eq!(first.state, "unhealthy");

        let second = probe_openai_compatible_http_endpoint(&address).await;
        assert!(second.available);
        assert_eq!(second.state, "healthy");
        assert_eq!(second.models, vec!["lemonade-recovered".to_string()]);
        assert_eq!(requests.load(Ordering::SeqCst), 2);

        handle.await.unwrap();
    }

    #[test]
    fn endpoint_declares_inference_capabilities() {
        let endpoint = proto::EndpointManifest {
            endpoint_id: "demo".into(),
            kind: proto::EndpointKind::Inference as i32,
            transport_kind: proto::EndpointTransportKind::EndpointTransportHttp as i32,
            protocol: Some("openai_compatible".into()),
            address: Some("http://localhost:8000/api/v1".into()),
            args: Vec::new(),
            namespace: None,
            supports_streaming: true,
            managed_by_plugin: false,
        };
        assert_eq!(
            endpoint_declared_capabilities(&endpoint),
            vec![
                "endpoint:inference".to_string(),
                "endpoint:inference/openai_compatible".to_string()
            ]
        );
    }
}
