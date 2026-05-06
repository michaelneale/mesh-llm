use anyhow::{anyhow, Context, Result};
use rmcp::{
    model::{
        CallToolRequestParams, CallToolResult, CancelTaskParams, CancelTaskResult, ClientResult,
        CompleteRequestParams, CompleteResult, CreateElicitationRequest,
        CreateElicitationRequestParams, CreateMessageRequestParams, CustomNotification,
        CustomRequest, ErrorCode, GetPromptRequestParams, GetPromptResult, GetTaskInfoParams,
        GetTaskPayloadResult, GetTaskResult, GetTaskResultParams, Implementation,
        ListPromptsResult, ListResourceTemplatesResult, ListResourcesResult, ListTasksResult,
        ListToolsResult, LoggingMessageNotificationParam, PaginatedRequestParams, PingRequest,
        ReadResourceRequestParams, ReadResourceResult, ResourceUpdatedNotificationParam,
        ServerCapabilities, ServerInfo, ServerNotification, ServerRequest, SetLevelRequestParams,
        SubscribeRequestParams, UnsubscribeRequestParams,
    },
    service::{NotificationContext, Peer, RequestContext, RunningService},
    transport::{io::stdio, StreamableHttpClientTransport, TokioChildProcess},
    ErrorData, RoleClient, RoleServer, ServerHandler, ServiceExt,
};
use serde::Serialize;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use tokio::net::TcpStream;
#[cfg(unix)]
use tokio::net::UnixStream;
use tokio::process::Command;
use tokio::sync::Mutex;

use crate::plugin::stapler;
use crate::plugin::{self, PluginEndpointSummary, PluginManager, PluginRpcBridge, RpcResult};

#[derive(Clone)]
enum ToolTarget {
    Plugin {
        plugin_name: String,
        tool_name: String,
    },
    External {
        endpoint: ExternalMcpEndpoint,
        tool_name: String,
    },
}

#[derive(Clone)]
struct ToolRef {
    target: ToolTarget,
    tool: rmcp::model::Tool,
}

#[derive(Clone)]
enum PromptTarget {
    Plugin {
        plugin_name: String,
        prompt_name: String,
    },
    External {
        endpoint: ExternalMcpEndpoint,
        prompt_name: String,
    },
}

#[derive(Clone)]
struct PromptRef {
    target: PromptTarget,
}

#[derive(Clone)]
enum ResourceTarget {
    Plugin {
        plugin_name: String,
        resource_uri: String,
    },
    External {
        endpoint: ExternalMcpEndpoint,
        original_uri: String,
    },
}

#[derive(Clone)]
struct ResourceRef {
    target: ResourceTarget,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum ExternalMcpTransport {
    Stdio { command: String, args: Vec<String> },
    Http { uri: String },
    Tcp { address: String },
    UnixSocket { path: String },
    NamedPipe { name: String },
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ExternalMcpEndpoint {
    key: String,
    plugin_name: String,
    endpoint_id: String,
    transport: ExternalMcpTransport,
    namespace_prefix: String,
}

impl ExternalMcpEndpoint {
    fn from_summary(summary: PluginEndpointSummary) -> Option<Self> {
        if !summary.available || summary.kind != "mcp" {
            return None;
        }
        let local_namespace = summary
            .namespace
            .unwrap_or_else(|| summary.endpoint_id.clone());
        let plugin_name = summary.plugin_name;
        let transport = match summary.transport_kind.as_str() {
            "stdio" => ExternalMcpTransport::Stdio {
                command: summary.address?,
                args: summary.args,
            },
            "http" => ExternalMcpTransport::Http {
                uri: summary.address?,
            },
            "tcp" => ExternalMcpTransport::Tcp {
                address: summary.address?,
            },
            "unix_socket" => ExternalMcpTransport::UnixSocket {
                path: summary.address?,
            },
            "named_pipe" => ExternalMcpTransport::NamedPipe {
                name: summary.address?,
            },
            _ => return None,
        };
        Some(Self {
            key: format!("{}:{}", plugin_name, summary.endpoint_id),
            plugin_name: plugin_name.clone(),
            endpoint_id: summary.endpoint_id,
            transport,
            namespace_prefix: format!("{}.{}", plugin_name, local_namespace),
        })
    }

    fn canonical_name(&self, local_name: &str) -> String {
        format!("{}.{}", self.namespace_prefix, local_name)
    }

    fn canonical_resource_uri(&self, original_uri: &str) -> String {
        format!(
            "mesh-mcp://{}/{}/resource/{}",
            self.plugin_name,
            self.endpoint_id,
            urlencoding::encode(original_uri)
        )
    }

    fn canonical_resource_template_uri(&self, original_uri_template: &str) -> String {
        format!(
            "mesh-mcp://{}/{}/template/{}",
            self.plugin_name,
            self.endpoint_id,
            urlencoding::encode(original_uri_template)
        )
    }

    fn transport_label(&self) -> String {
        match &self.transport {
            ExternalMcpTransport::Stdio { command, .. } => command.clone(),
            ExternalMcpTransport::Http { uri } => uri.clone(),
            ExternalMcpTransport::Tcp { address } => address.clone(),
            ExternalMcpTransport::UnixSocket { path } => path.clone(),
            ExternalMcpTransport::NamedPipe { name } => name.clone(),
        }
    }
}

#[derive(Clone)]
struct ExternalMcpClient {
    peer: Peer<RoleClient>,
    running: Arc<Mutex<RunningService<RoleClient, ()>>>,
}

impl ExternalMcpClient {
    async fn connect(endpoint: &ExternalMcpEndpoint) -> Result<Self> {
        let running = match &endpoint.transport {
            ExternalMcpTransport::Stdio { command, args } => {
                let mut child = Command::new(command);
                child.args(args);
                let transport = TokioChildProcess::new(child).with_context(|| {
                    format!(
                        "Spawn external MCP endpoint '{}:{}' with command '{}'",
                        endpoint.plugin_name, endpoint.endpoint_id, command
                    )
                })?;
                ().serve(transport).await.map_err(anyhow::Error::from)
            }
            ExternalMcpTransport::Http { uri } => {
                let transport = StreamableHttpClientTransport::from_uri(uri.clone());
                ().serve(transport).await.map_err(anyhow::Error::from)
            }
            ExternalMcpTransport::Tcp { address } => {
                let stream = TcpStream::connect(address).await.with_context(|| {
                    format!(
                        "Connect TCP external MCP endpoint '{}:{}' at '{}'",
                        endpoint.plugin_name, endpoint.endpoint_id, address
                    )
                })?;
                ().serve(stream).await.map_err(anyhow::Error::from)
            }
            ExternalMcpTransport::UnixSocket { path } => {
                #[cfg(unix)]
                {
                    let stream = UnixStream::connect(path).await.with_context(|| {
                        format!(
                            "Connect Unix socket MCP endpoint '{}:{}' at '{}'",
                            endpoint.plugin_name, endpoint.endpoint_id, path
                        )
                    })?;
                    ().serve(stream).await.map_err(anyhow::Error::from)
                }
                #[cfg(not(unix))]
                {
                    let _ = path;
                    Err(anyhow!(
                        "Unix socket MCP endpoint '{}:{}' is unsupported on this platform",
                        endpoint.plugin_name,
                        endpoint.endpoint_id
                    ))
                }
            }
            ExternalMcpTransport::NamedPipe { name } => {
                #[cfg(windows)]
                {
                    let client = tokio::net::windows::named_pipe::ClientOptions::new()
                        .open(name)
                        .with_context(|| {
                            format!(
                                "Connect named pipe MCP endpoint '{}:{}' at '{}'",
                                endpoint.plugin_name, endpoint.endpoint_id, name
                            )
                        })?;
                    ().serve(client).await.map_err(anyhow::Error::from)
                }
                #[cfg(not(windows))]
                {
                    let _ = name;
                    Err(anyhow!(
                        "Named pipe MCP endpoint '{}:{}' is unsupported on this platform",
                        endpoint.plugin_name,
                        endpoint.endpoint_id
                    ))
                }
            }
        }
        .with_context(|| {
            format!(
                "Connect to external MCP endpoint '{}:{}' via '{}'",
                endpoint.plugin_name,
                endpoint.endpoint_id,
                endpoint.transport_label()
            )
        })?;
        let peer = running.peer().clone();
        Ok(Self {
            peer,
            running: Arc::new(Mutex::new(running)),
        })
    }

    async fn is_closed(&self) -> bool {
        self.running.lock().await.is_closed()
    }
}

#[derive(Clone, Default)]
struct ExternalMcpPool {
    clients: Arc<Mutex<BTreeMap<String, Arc<ExternalMcpClient>>>>,
    #[cfg(test)]
    test_clients: Arc<Mutex<BTreeMap<String, Arc<ExternalMcpClient>>>>,
}

impl ExternalMcpPool {
    async fn retain_active(&self, active_keys: &BTreeSet<String>) {
        let mut clients = self.clients.lock().await;
        clients.retain(|key, _| active_keys.contains(key));
        #[cfg(test)]
        {
            let mut test_clients = self.test_clients.lock().await;
            test_clients.retain(|key, _| active_keys.contains(key));
        }
    }

    async fn client_for(
        &self,
        endpoint: &ExternalMcpEndpoint,
    ) -> Result<Arc<ExternalMcpClient>, ErrorData> {
        #[cfg(test)]
        if let Some(client) = self.test_clients.lock().await.get(&endpoint.key).cloned() {
            return Ok(client);
        }

        if let Some(client) = self.clients.lock().await.get(&endpoint.key).cloned() {
            if !client.is_closed().await {
                return Ok(client);
            }
            self.clients.lock().await.remove(&endpoint.key);
        }

        let client = Arc::new(
            ExternalMcpClient::connect(endpoint)
                .await
                .map_err(internal_error)?,
        );
        self.clients
            .lock()
            .await
            .insert(endpoint.key.clone(), client.clone());
        Ok(client)
    }

    #[cfg(test)]
    async fn register_test_client(&self, endpoint_key: &str, client: Arc<ExternalMcpClient>) {
        self.test_clients
            .lock()
            .await
            .insert(endpoint_key.to_string(), client);
    }
}

#[derive(Clone, Default)]
struct ActiveBridge {
    peer: Arc<Mutex<Option<Peer<RoleServer>>>>,
}

impl ActiveBridge {
    async fn set_peer(&self, peer: Peer<RoleServer>) {
        *self.peer.lock().await = Some(peer);
    }

    async fn current_peer(&self) -> Result<Peer<RoleServer>, plugin::proto::ErrorResponse> {
        self.peer
            .lock()
            .await
            .clone()
            .ok_or_else(|| proto_error::internal("No active MCP client session"))
    }
}

impl PluginRpcBridge for ActiveBridge {
    fn handle_request(
        &self,
        _plugin_name: String,
        method: String,
        params_json: String,
    ) -> crate::plugin::BridgeFuture<Result<RpcResult, plugin::proto::ErrorResponse>> {
        let this = self.clone();
        Box::pin(async move {
            let peer: Peer<RoleServer> = this.current_peer().await?;
            let params = parse_optional_value(&params_json)?;
            let result_json = match method.as_str() {
                "ping" => {
                    let result: ClientResult = peer
                        .send_request(ServerRequest::PingRequest(PingRequest::default()))
                        .await
                        .map_err(proto_error::from_service)?;
                    match result {
                        ClientResult::EmptyResult(result) => to_json_string(&result),
                        _ => Err(proto_error::internal("unexpected ping response")),
                    }
                }
                "roots/list" => {
                    to_json_string(&peer.list_roots().await.map_err(proto_error::from_service)?)
                }
                "sampling/createMessage" => {
                    let params =
                        deserialize_required::<CreateMessageRequestParams>(params, &method)?;
                    to_json_string(
                        &peer
                            .create_message(params)
                            .await
                            .map_err(proto_error::from_service)?,
                    )
                }
                "elicitation/create" => {
                    let params =
                        deserialize_required::<CreateElicitationRequestParams>(params, &method)?;
                    let result: ClientResult = peer
                        .send_request(ServerRequest::CreateElicitationRequest(
                            CreateElicitationRequest::new(params),
                        ))
                        .await
                        .map_err(proto_error::from_service)?;
                    match result {
                        ClientResult::CreateElicitationResult(result) => to_json_string(&result),
                        _ => Err(proto_error::internal("unexpected elicitation response")),
                    }
                }
                _ => {
                    let result: ClientResult = peer
                        .send_request(ServerRequest::CustomRequest(CustomRequest::new(
                            method.clone(),
                            params,
                        )))
                        .await
                        .map_err(proto_error::from_service)?;
                    match result {
                        ClientResult::CustomResult(result) => to_json_string(&result),
                        _ => Err(proto_error::internal("unexpected custom response")),
                    }
                }
            }
            .map_err(|mut err| {
                err.message = format!("bridge request '{method}': {}", err.message);
                err
            })?;

            Ok(RpcResult { result_json })
        })
    }

    fn handle_notification(
        &self,
        _plugin_name: String,
        method: String,
        params_json: String,
    ) -> crate::plugin::BridgeFuture<()> {
        let this = self.clone();
        Box::pin(async move {
            let Ok(peer): Result<Peer<RoleServer>, _> = this.current_peer().await else {
                return;
            };
            let params = match parse_optional_value(&params_json) {
                Ok(params) => params,
                Err(_) => return,
            };

            match method.as_str() {
                "notifications/resources/updated" => {
                    if let Ok(params) =
                        deserialize_required::<ResourceUpdatedNotificationParam>(params, &method)
                    {
                        let _ = peer.notify_resource_updated(params).await;
                    }
                }
                "notifications/resources/list_changed" => {
                    let _ = peer.notify_resource_list_changed().await;
                }
                "notifications/tools/list_changed" => {
                    let _ = peer.notify_tool_list_changed().await;
                }
                "notifications/prompts/list_changed" => {
                    let _ = peer.notify_prompt_list_changed().await;
                }
                "notifications/message" => {
                    if let Ok(params) =
                        deserialize_required::<LoggingMessageNotificationParam>(params, &method)
                    {
                        let _ = peer.notify_logging_message(params).await;
                    }
                }
                _ => {
                    let _ = peer
                        .send_notification(ServerNotification::CustomNotification(
                            CustomNotification::new(method, params),
                        ))
                        .await;
                }
            }
        })
    }
}

#[derive(Clone)]
pub struct PluginMcpServer {
    plugin_manager: PluginManager,
    bridge: ActiveBridge,
    external_mcp: ExternalMcpPool,
}

impl PluginMcpServer {
    fn new(plugin_manager: PluginManager, bridge: ActiveBridge) -> Self {
        Self {
            plugin_manager,
            bridge,
            external_mcp: ExternalMcpPool::default(),
        }
    }

    async fn active_external_mcp_endpoints(&self) -> Result<Vec<ExternalMcpEndpoint>, ErrorData> {
        let passive_endpoint_summaries = self
            .plugin_manager
            .endpoints()
            .await
            .map_err(internal_error)?;
        let endpoints = passive_endpoint_summaries
            .into_iter()
            .filter_map(ExternalMcpEndpoint::from_summary)
            .collect::<Vec<_>>();
        let active_keys = endpoints
            .iter()
            .map(|endpoint| endpoint.key.clone())
            .collect::<BTreeSet<_>>();
        self.external_mcp.retain_active(&active_keys).await;
        Ok(endpoints)
    }

    async fn discover_tools(&self) -> Result<BTreeMap<String, ToolRef>, ErrorData> {
        let mut tools = BTreeMap::new();
        for (plugin_name, _) in self.plugin_manager.list_server_infos().await {
            let manifest = self
                .plugin_manager
                .manifest(&plugin_name)
                .await
                .map_err(internal_error)?;
            if let Some(manifest) = manifest.filter(|manifest| !manifest.operations.is_empty()) {
                for operation in &manifest.operations {
                    let raw_name = operation.name.clone();
                    for mcp_name in tool_aliases(&plugin_name, &raw_name) {
                        tools.insert(
                            mcp_name.clone(),
                            ToolRef {
                                target: ToolTarget::Plugin {
                                    plugin_name: plugin_name.clone(),
                                    tool_name: raw_name.clone(),
                                },
                                tool: stapler::operation(mcp_name, operation),
                            },
                        );
                    }
                }
            }
        }
        for endpoint in self.active_external_mcp_endpoints().await? {
            let client = match self.external_mcp.client_for(&endpoint).await {
                Ok(client) => client,
                Err(err) => {
                    tracing::warn!(
                        plugin = %endpoint.plugin_name,
                        endpoint = %endpoint.endpoint_id,
                        error = %err,
                        "Skipping external MCP endpoint during tool discovery"
                    );
                    continue;
                }
            };
            let listed = match client.peer.list_all_tools().await {
                Ok(tools) => tools,
                Err(err) => {
                    tracing::warn!(
                        plugin = %endpoint.plugin_name,
                        endpoint = %endpoint.endpoint_id,
                        error = %err,
                        "Failed to list tools from external MCP endpoint"
                    );
                    continue;
                }
            };
            for tool in listed {
                let raw_name = tool.name.to_string();
                let canonical_name = endpoint.canonical_name(&raw_name);
                let mut namespaced = tool.clone();
                namespaced.name = canonical_name.clone().into();
                tools.insert(
                    canonical_name,
                    ToolRef {
                        target: ToolTarget::External {
                            endpoint: endpoint.clone(),
                            tool_name: raw_name,
                        },
                        tool: namespaced,
                    },
                );
            }
        }
        Ok(tools)
    }

    async fn discover_prompts(&self) -> Result<BTreeMap<String, PromptRef>, ErrorData> {
        let mut prompts = BTreeMap::new();
        for (plugin_name, _) in self.plugin_manager.list_server_infos().await {
            let manifest = self
                .plugin_manager
                .manifest(&plugin_name)
                .await
                .map_err(internal_error)?;
            if let Some(manifest) = manifest.filter(|manifest| !manifest.prompts.is_empty()) {
                for prompt in &manifest.prompts {
                    prompts.insert(
                        canonical_name(&plugin_name, &prompt.name),
                        PromptRef {
                            target: PromptTarget::Plugin {
                                plugin_name: plugin_name.clone(),
                                prompt_name: prompt.name.clone(),
                            },
                        },
                    );
                }
            }
        }
        for endpoint in self.active_external_mcp_endpoints().await? {
            let client = match self.external_mcp.client_for(&endpoint).await {
                Ok(client) => client,
                Err(err) => {
                    tracing::warn!(
                        plugin = %endpoint.plugin_name,
                        endpoint = %endpoint.endpoint_id,
                        error = %err,
                        "Skipping external MCP endpoint during prompt discovery"
                    );
                    continue;
                }
            };
            let listed = match client.peer.list_all_prompts().await {
                Ok(prompts) => prompts,
                Err(err) => {
                    tracing::warn!(
                        plugin = %endpoint.plugin_name,
                        endpoint = %endpoint.endpoint_id,
                        error = %err,
                        "Failed to list prompts from external MCP endpoint"
                    );
                    continue;
                }
            };
            for prompt in listed {
                prompts.insert(
                    endpoint.canonical_name(&prompt.name),
                    PromptRef {
                        target: PromptTarget::External {
                            endpoint: endpoint.clone(),
                            prompt_name: prompt.name,
                        },
                    },
                );
            }
        }
        Ok(prompts)
    }

    async fn refresh_peer(&self, peer: Peer<RoleServer>) {
        self.bridge.set_peer(peer).await;
    }

    async fn discover_resources(&self) -> Result<BTreeMap<String, ResourceRef>, ErrorData> {
        let mut resources = BTreeMap::new();
        for (plugin_name, _) in self.plugin_manager.list_server_infos().await {
            let manifest = self
                .plugin_manager
                .manifest(&plugin_name)
                .await
                .map_err(internal_error)?;
            if let Some(manifest) = manifest.filter(|manifest| !manifest.resources.is_empty()) {
                for resource in manifest.resources {
                    resources.insert(
                        resource.uri.clone(),
                        ResourceRef {
                            target: ResourceTarget::Plugin {
                                plugin_name: plugin_name.clone(),
                                resource_uri: resource.uri,
                            },
                        },
                    );
                }
            }
        }
        for endpoint in self.active_external_mcp_endpoints().await? {
            let client = match self.external_mcp.client_for(&endpoint).await {
                Ok(client) => client,
                Err(err) => {
                    tracing::warn!(
                        plugin = %endpoint.plugin_name,
                        endpoint = %endpoint.endpoint_id,
                        error = %err,
                        "Skipping external MCP endpoint during resource discovery"
                    );
                    continue;
                }
            };
            let listed = match client.peer.list_all_resources().await {
                Ok(resources) => resources,
                Err(err) => {
                    tracing::warn!(
                        plugin = %endpoint.plugin_name,
                        endpoint = %endpoint.endpoint_id,
                        error = %err,
                        "Failed to list resources from external MCP endpoint"
                    );
                    continue;
                }
            };
            for resource in listed {
                resources.insert(
                    endpoint.canonical_resource_uri(&resource.raw.uri),
                    ResourceRef {
                        target: ResourceTarget::External {
                            endpoint: endpoint.clone(),
                            original_uri: resource.raw.uri,
                        },
                    },
                );
            }
        }
        Ok(resources)
    }

    async fn broadcast_notification<P>(&self, method: &str, params: P)
    where
        P: Serialize + Clone,
    {
        for (plugin_name, _) in self.plugin_manager.list_server_infos().await {
            let _ = self
                .plugin_manager
                .mcp_notify(&plugin_name, method, params.clone())
                .await;
        }
    }
}

impl ServerHandler for PluginMcpServer {
    async fn initialize(
        &self,
        request: rmcp::model::InitializeRequestParams,
        context: RequestContext<RoleServer>,
    ) -> Result<ServerInfo, ErrorData> {
        if context.peer.peer_info().is_none() {
            context.peer.set_peer_info(request);
        }
        self.refresh_peer(context.peer.clone()).await;
        Ok(self.get_info())
    }

    async fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        context: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, ErrorData> {
        self.refresh_peer(context.peer.clone()).await;
        Ok(ListToolsResult {
            tools: self
                .discover_tools()
                .await?
                .into_values()
                .map(|entry| entry.tool)
                .collect(),
            meta: None,
            next_cursor: None,
        })
    }

    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        context: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, ErrorData> {
        self.refresh_peer(context.peer.clone()).await;
        let tools = self.discover_tools().await?;
        let Some(tool_ref) = tools.get(request.name.as_ref()) else {
            return Err(ErrorData::invalid_params(
                format!("Unknown MCP tool '{}'", request.name),
                None,
            ));
        };
        match &tool_ref.target {
            ToolTarget::Plugin {
                plugin_name,
                tool_name,
            } => {
                let arguments = request
                    .arguments
                    .map(Value::Object)
                    .unwrap_or_else(|| serde_json::json!({}));
                let result = self
                    .plugin_manager
                    .invoke_operation(plugin_name, tool_name, &arguments.to_string())
                    .await
                    .map_err(internal_error)?;
                Ok(operation_result_to_call_tool_result(result))
            }
            ToolTarget::External {
                endpoint,
                tool_name,
            } => {
                let client = self.external_mcp.client_for(endpoint).await?;
                let mut params = CallToolRequestParams::new(tool_name.clone());
                if let Some(arguments) = request.arguments {
                    params = params.with_arguments(arguments);
                }
                if let Some(task) = request.task {
                    params = params.with_task(task);
                }
                if let Some(meta) = request.meta {
                    params.meta = Some(meta);
                }
                client.peer.call_tool(params).await.map_err(internal_error)
            }
        }
    }

    async fn list_prompts(
        &self,
        _request: Option<PaginatedRequestParams>,
        context: RequestContext<RoleServer>,
    ) -> Result<ListPromptsResult, ErrorData> {
        self.refresh_peer(context.peer.clone()).await;
        let mut prompts = Vec::new();
        for (plugin_name, _) in self.plugin_manager.list_server_infos().await {
            let manifest = self
                .plugin_manager
                .manifest(&plugin_name)
                .await
                .map_err(internal_error)?;
            if let Some(manifest) = manifest.filter(|manifest| !manifest.prompts.is_empty()) {
                prompts.extend(manifest.prompts.into_iter().map(|prompt| {
                    stapler::prompt(canonical_name(&plugin_name, &prompt.name), &prompt)
                }));
            }
        }
        for endpoint in self.active_external_mcp_endpoints().await? {
            let client = match self.external_mcp.client_for(&endpoint).await {
                Ok(client) => client,
                Err(err) => {
                    tracing::warn!(
                        plugin = %endpoint.plugin_name,
                        endpoint = %endpoint.endpoint_id,
                        error = %err,
                        "Skipping external MCP endpoint during prompt listing"
                    );
                    continue;
                }
            };
            let listed = match client.peer.list_all_prompts().await {
                Ok(prompts) => prompts,
                Err(err) => {
                    tracing::warn!(
                        plugin = %endpoint.plugin_name,
                        endpoint = %endpoint.endpoint_id,
                        error = %err,
                        "Failed to list prompts from external MCP endpoint"
                    );
                    continue;
                }
            };
            prompts.extend(listed.into_iter().map(|mut prompt| {
                prompt.name = endpoint.canonical_name(&prompt.name);
                prompt
            }));
        }
        Ok(ListPromptsResult {
            prompts,
            meta: None,
            next_cursor: None,
        })
    }

    async fn get_prompt(
        &self,
        request: GetPromptRequestParams,
        context: RequestContext<RoleServer>,
    ) -> Result<GetPromptResult, ErrorData> {
        self.refresh_peer(context.peer.clone()).await;
        let prompts = self.discover_prompts().await?;
        let Some(entry) = prompts.get(request.name.as_str()) else {
            return Err(ErrorData::invalid_params(
                format!("Unknown MCP prompt '{}'", request.name),
                None,
            ));
        };

        match &entry.target {
            PromptTarget::Plugin {
                plugin_name,
                prompt_name,
            } => {
                let mut params = GetPromptRequestParams::new(prompt_name.clone());
                if let Some(arguments) = request.arguments {
                    params = params.with_arguments(arguments);
                }
                if let Some(meta) = request.meta {
                    params.meta = Some(meta);
                }

                self.plugin_manager
                    .get_prompt(plugin_name, prompt_name, params)
                    .await
                    .map_err(internal_error)
            }
            PromptTarget::External {
                endpoint,
                prompt_name,
            } => {
                let client = self.external_mcp.client_for(endpoint).await?;
                let mut params = GetPromptRequestParams::new(prompt_name.clone());
                if let Some(arguments) = request.arguments {
                    params = params.with_arguments(arguments);
                }
                if let Some(meta) = request.meta {
                    params.meta = Some(meta);
                }
                client.peer.get_prompt(params).await.map_err(internal_error)
            }
        }
    }

    async fn list_resources(
        &self,
        _request: Option<PaginatedRequestParams>,
        context: RequestContext<RoleServer>,
    ) -> Result<ListResourcesResult, ErrorData> {
        self.refresh_peer(context.peer.clone()).await;
        let mut resources = Vec::new();
        for (plugin_name, _) in self.plugin_manager.list_server_infos().await {
            let manifest = self
                .plugin_manager
                .manifest(&plugin_name)
                .await
                .map_err(internal_error)?;
            if let Some(manifest) = manifest.filter(|manifest| !manifest.resources.is_empty()) {
                resources.extend(manifest.resources.iter().map(stapler::resource));
            }
        }
        for endpoint in self.active_external_mcp_endpoints().await? {
            let client = match self.external_mcp.client_for(&endpoint).await {
                Ok(client) => client,
                Err(err) => {
                    tracing::warn!(
                        plugin = %endpoint.plugin_name,
                        endpoint = %endpoint.endpoint_id,
                        error = %err,
                        "Skipping external MCP endpoint during resource listing"
                    );
                    continue;
                }
            };
            let listed = match client.peer.list_all_resources().await {
                Ok(resources) => resources,
                Err(err) => {
                    tracing::warn!(
                        plugin = %endpoint.plugin_name,
                        endpoint = %endpoint.endpoint_id,
                        error = %err,
                        "Failed to list resources from external MCP endpoint"
                    );
                    continue;
                }
            };
            resources.extend(listed.into_iter().map(|mut resource| {
                resource.raw.name = endpoint.canonical_name(&resource.raw.name);
                resource.raw.uri = endpoint.canonical_resource_uri(&resource.raw.uri);
                resource
            }));
        }
        Ok(ListResourcesResult {
            resources,
            meta: None,
            next_cursor: None,
        })
    }

    async fn list_resource_templates(
        &self,
        _request: Option<PaginatedRequestParams>,
        context: RequestContext<RoleServer>,
    ) -> Result<ListResourceTemplatesResult, ErrorData> {
        self.refresh_peer(context.peer.clone()).await;
        let mut resource_templates = Vec::new();
        for (plugin_name, _) in self.plugin_manager.list_server_infos().await {
            let manifest = self
                .plugin_manager
                .manifest(&plugin_name)
                .await
                .map_err(internal_error)?;
            if let Some(manifest) =
                manifest.filter(|manifest| !manifest.resource_templates.is_empty())
            {
                resource_templates.extend(
                    manifest
                        .resource_templates
                        .iter()
                        .map(stapler::resource_template),
                );
            }
        }
        for endpoint in self.active_external_mcp_endpoints().await? {
            let client = match self.external_mcp.client_for(&endpoint).await {
                Ok(client) => client,
                Err(err) => {
                    tracing::warn!(
                        plugin = %endpoint.plugin_name,
                        endpoint = %endpoint.endpoint_id,
                        error = %err,
                        "Skipping external MCP endpoint during resource template listing"
                    );
                    continue;
                }
            };
            let listed = match client.peer.list_all_resource_templates().await {
                Ok(templates) => templates,
                Err(err) => {
                    tracing::warn!(
                        plugin = %endpoint.plugin_name,
                        endpoint = %endpoint.endpoint_id,
                        error = %err,
                        "Failed to list resource templates from external MCP endpoint"
                    );
                    continue;
                }
            };
            resource_templates.extend(listed.into_iter().map(|mut template| {
                template.raw.name = endpoint.canonical_name(&template.raw.name);
                template.raw.uri_template =
                    endpoint.canonical_resource_template_uri(&template.raw.uri_template);
                template
            }));
        }
        Ok(ListResourceTemplatesResult {
            resource_templates,
            meta: None,
            next_cursor: None,
        })
    }

    async fn read_resource(
        &self,
        request: ReadResourceRequestParams,
        context: RequestContext<RoleServer>,
    ) -> Result<ReadResourceResult, ErrorData> {
        self.refresh_peer(context.peer.clone()).await;
        if let Some(resource_ref) = self.discover_resources().await?.get(&request.uri).cloned() {
            match resource_ref.target {
                ResourceTarget::Plugin {
                    plugin_name,
                    resource_uri,
                } => {
                    let mut params = ReadResourceRequestParams::new(resource_uri);
                    if let Some(meta) = request.meta {
                        params.meta = Some(meta);
                    }
                    return self
                        .plugin_manager
                        .read_resource(&plugin_name, &request.uri, params)
                        .await
                        .map_err(internal_error);
                }
                ResourceTarget::External {
                    endpoint,
                    original_uri,
                } => {
                    let client = self.external_mcp.client_for(&endpoint).await?;
                    let mut params = ReadResourceRequestParams::new(original_uri);
                    if let Some(meta) = request.meta {
                        params.meta = Some(meta);
                    }
                    let mut result = client
                        .peer
                        .read_resource(params)
                        .await
                        .map_err(internal_error)?;
                    for content in &mut result.contents {
                        match content {
                            rmcp::model::ResourceContents::TextResourceContents { uri, .. }
                            | rmcp::model::ResourceContents::BlobResourceContents { uri, .. } => {
                                *uri = request.uri.clone();
                            }
                        }
                    }
                    return Ok(result);
                }
            }
        }
        try_plugins(&self.plugin_manager, "resources/read", request).await
    }

    async fn subscribe(
        &self,
        request: SubscribeRequestParams,
        context: RequestContext<RoleServer>,
    ) -> Result<(), ErrorData> {
        self.refresh_peer(context.peer.clone()).await;
        if let Some(resource_ref) = self.discover_resources().await?.get(&request.uri).cloned() {
            match resource_ref.target {
                ResourceTarget::Plugin { .. } => {}
                ResourceTarget::External {
                    endpoint,
                    original_uri,
                } => {
                    let client = self.external_mcp.client_for(&endpoint).await?;
                    let mut params = SubscribeRequestParams::new(original_uri);
                    if let Some(meta) = request.meta {
                        params.meta = Some(meta);
                    }
                    return client.peer.subscribe(params).await.map_err(internal_error);
                }
            }
        }
        try_plugins::<(), _>(&self.plugin_manager, "resources/subscribe", request).await
    }

    async fn unsubscribe(
        &self,
        request: UnsubscribeRequestParams,
        context: RequestContext<RoleServer>,
    ) -> Result<(), ErrorData> {
        self.refresh_peer(context.peer.clone()).await;
        if let Some(resource_ref) = self.discover_resources().await?.get(&request.uri).cloned() {
            match resource_ref.target {
                ResourceTarget::Plugin { .. } => {}
                ResourceTarget::External {
                    endpoint,
                    original_uri,
                } => {
                    let client = self.external_mcp.client_for(&endpoint).await?;
                    let mut params = UnsubscribeRequestParams::new(original_uri);
                    if let Some(meta) = request.meta {
                        params.meta = Some(meta);
                    }
                    return client
                        .peer
                        .unsubscribe(params)
                        .await
                        .map_err(internal_error);
                }
            }
        }
        try_plugins::<(), _>(&self.plugin_manager, "resources/unsubscribe", request).await
    }

    async fn complete(
        &self,
        mut request: CompleteRequestParams,
        context: RequestContext<RoleServer>,
    ) -> Result<CompleteResult, ErrorData> {
        self.refresh_peer(context.peer.clone()).await;
        if let Some(name) = request.r#ref.as_prompt_name() {
            let prompts = self.discover_prompts().await?;
            let Some(entry) = prompts.get(name) else {
                return Err(ErrorData::invalid_params(
                    format!("Unknown MCP prompt reference '{}'", name),
                    None,
                ));
            };
            match &entry.target {
                PromptTarget::Plugin {
                    plugin_name,
                    prompt_name,
                } => {
                    if let rmcp::model::Reference::Prompt(prompt) = &mut request.r#ref {
                        prompt.name = prompt_name.clone();
                    }
                    return self
                        .plugin_manager
                        .complete(plugin_name, prompt_name, request)
                        .await
                        .map_err(internal_error);
                }
                PromptTarget::External {
                    endpoint,
                    prompt_name,
                } => {
                    if let rmcp::model::Reference::Prompt(prompt) = &mut request.r#ref {
                        prompt.name = prompt_name.clone();
                    }
                    let client = self.external_mcp.client_for(endpoint).await?;
                    return client.peer.complete(request).await.map_err(internal_error);
                }
            }
        }
        try_plugins(&self.plugin_manager, "completion/complete", request).await
    }

    async fn set_level(
        &self,
        request: SetLevelRequestParams,
        context: RequestContext<RoleServer>,
    ) -> Result<(), ErrorData> {
        self.refresh_peer(context.peer.clone()).await;
        let mut first_error = None;
        for (plugin_name, server_info) in self.plugin_manager.list_server_infos().await {
            if server_info.capabilities.logging.is_none() {
                continue;
            }
            if let Err(err) = self
                .plugin_manager
                .mcp_request::<(), _>(&plugin_name, "logging/setLevel", request.clone())
                .await
            {
                first_error.get_or_insert(err);
            }
        }
        if let Some(err) = first_error {
            Err(internal_error(err))
        } else {
            Ok(())
        }
    }

    async fn list_tasks(
        &self,
        _request: Option<PaginatedRequestParams>,
        context: RequestContext<RoleServer>,
    ) -> Result<ListTasksResult, ErrorData> {
        self.refresh_peer(context.peer.clone()).await;
        let mut tasks = Vec::new();
        for (plugin_name, server_info) in self.plugin_manager.list_server_infos().await {
            if server_info.capabilities.tasks.is_none() {
                continue;
            }
            let result: ListTasksResult = self
                .plugin_manager
                .mcp_request(
                    &plugin_name,
                    "tasks/list",
                    Option::<PaginatedRequestParams>::None,
                )
                .await
                .map_err(internal_error)?;
            tasks.extend(result.tasks);
        }
        Ok(ListTasksResult::new(tasks))
    }

    async fn get_task_info(
        &self,
        request: GetTaskInfoParams,
        context: RequestContext<RoleServer>,
    ) -> Result<GetTaskResult, ErrorData> {
        self.refresh_peer(context.peer.clone()).await;
        try_plugins(&self.plugin_manager, "tasks/get", request).await
    }

    async fn get_task_result(
        &self,
        request: GetTaskResultParams,
        context: RequestContext<RoleServer>,
    ) -> Result<GetTaskPayloadResult, ErrorData> {
        self.refresh_peer(context.peer.clone()).await;
        try_plugins(&self.plugin_manager, "tasks/result", request).await
    }

    async fn cancel_task(
        &self,
        request: CancelTaskParams,
        context: RequestContext<RoleServer>,
    ) -> Result<CancelTaskResult, ErrorData> {
        self.refresh_peer(context.peer.clone()).await;
        try_plugins(&self.plugin_manager, "tasks/cancel", request).await
    }

    async fn on_cancelled(
        &self,
        notification: rmcp::model::CancelledNotificationParam,
        context: NotificationContext<RoleServer>,
    ) {
        self.refresh_peer(context.peer.clone()).await;
        self.broadcast_notification("notifications/cancelled", notification)
            .await;
    }

    async fn on_progress(
        &self,
        notification: rmcp::model::ProgressNotificationParam,
        context: NotificationContext<RoleServer>,
    ) {
        self.refresh_peer(context.peer.clone()).await;
        self.broadcast_notification("notifications/progress", notification)
            .await;
    }

    async fn on_initialized(&self, context: NotificationContext<RoleServer>) {
        self.refresh_peer(context.peer.clone()).await;
        self.broadcast_notification("notifications/initialized", serde_json::json!({}))
            .await;
    }

    async fn on_roots_list_changed(&self, context: NotificationContext<RoleServer>) {
        self.refresh_peer(context.peer.clone()).await;
        self.broadcast_notification("notifications/roots/list_changed", serde_json::json!({}))
            .await;
    }

    async fn on_custom_notification(
        &self,
        notification: CustomNotification,
        context: NotificationContext<RoleServer>,
    ) {
        self.refresh_peer(context.peer.clone()).await;
        self.broadcast_notification(
            &notification.method,
            notification.params.unwrap_or(serde_json::Value::Null),
        )
        .await;
    }

    async fn on_custom_request(
        &self,
        request: CustomRequest,
        context: RequestContext<RoleServer>,
    ) -> Result<rmcp::model::CustomResult, ErrorData> {
        self.refresh_peer(context.peer.clone()).await;
        try_plugins(
            &self.plugin_manager,
            &request.method,
            request.params.unwrap_or(serde_json::Value::Null),
        )
        .await
    }

    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(
            ServerCapabilities::builder()
                .enable_logging()
                .enable_completions()
                .enable_prompts()
                .enable_prompts_list_changed()
                .enable_resources()
                .enable_resources_list_changed()
                .enable_resources_subscribe()
                .enable_tools()
                .enable_tool_list_changed()
                .enable_tasks()
                .build(),
        )
        .with_server_info(
            Implementation::new("mesh-plugins", env!("CARGO_PKG_VERSION"))
                .with_title("Mesh Plugin MCP")
                .with_description(
                    "Re-exposes mesh-llm plugins as a single MCP server with the standard MCP surface.",
                ),
        )
        .with_instructions(
            "Running plugins are aggregated into one MCP server. Tool and prompt names are namespaced as <plugin>.<name> to avoid collisions.",
        )
    }
}

pub async fn run_mcp_server(plugin_manager: PluginManager) -> Result<()> {
    let bridge = ActiveBridge::default();
    plugin_manager
        .set_rpc_bridge(Some(Arc::new(bridge.clone())))
        .await;

    let result = async {
        let server = PluginMcpServer::new(plugin_manager.clone(), bridge);
        server.serve(stdio()).await?.waiting().await?;
        Ok::<(), anyhow::Error>(())
    }
    .await;

    plugin_manager.set_rpc_bridge(None).await;
    result
}

fn internal_error(err: impl std::fmt::Display) -> ErrorData {
    ErrorData::internal_error(err.to_string(), None)
}

fn to_json_string<T: Serialize>(value: &T) -> Result<String, plugin::proto::ErrorResponse> {
    serde_json::to_string(value).map_err(|err| proto_error::from_anyhow(err.into()))
}

fn parse_optional_value(
    raw: &str,
) -> Result<Option<serde_json::Value>, plugin::proto::ErrorResponse> {
    plugin::parse_optional_json(raw).map_err(proto_error::from_anyhow)
}

fn deserialize_required<T: serde::de::DeserializeOwned>(
    value: Option<serde_json::Value>,
    method: &str,
) -> Result<T, plugin::proto::ErrorResponse> {
    let value = value.unwrap_or(serde_json::Value::Null);
    serde_json::from_value(value).map_err(|err| plugin::proto::ErrorResponse {
        code: ErrorCode::INVALID_PARAMS.0,
        message: format!("Invalid params for '{method}': {err}"),
        data_json: String::new(),
    })
}

async fn try_plugins<T, P>(
    plugin_manager: &PluginManager,
    method: &str,
    params: P,
) -> Result<T, ErrorData>
where
    T: serde::de::DeserializeOwned,
    P: Serialize + Clone,
{
    let mut last_error = None;
    for (plugin_name, _) in plugin_manager.list_server_infos().await {
        match plugin_manager
            .mcp_request::<T, _>(&plugin_name, method, params.clone())
            .await
        {
            Ok(value) => return Ok(value),
            Err(err) => last_error = Some(err),
        }
    }
    Err(internal_error(
        last_error.unwrap_or_else(|| anyhow!("No plugin handled '{method}'")),
    ))
}

fn operation_result_to_call_tool_result(result: plugin::ToolCallResult) -> CallToolResult {
    let mut call_result = match serde_json::from_str::<Value>(&result.content_json) {
        Ok(value) => CallToolResult::structured(value),
        Err(_) => CallToolResult::success(vec![rmcp::model::Content::text(result.content_json)]),
    };
    call_result.is_error = Some(result.is_error);
    call_result
}

fn tool_aliases(plugin_name: &str, tool_name: &str) -> Vec<String> {
    let canonical = canonical_name(plugin_name, tool_name);
    let mut names = vec![canonical];
    if plugin_name == plugin::BLACKBOARD_PLUGIN_ID {
        names.push(format!("blackboard_{tool_name}"));
    }
    names
}

fn canonical_name(plugin_name: &str, local_name: &str) -> String {
    format!("{plugin_name}.{local_name}")
}

mod proto_error {
    use anyhow::Error;
    use rmcp::{model::ErrorCode, ServiceError};

    pub fn from_anyhow(err: Error) -> crate::plugin::proto::ErrorResponse {
        crate::plugin::proto::ErrorResponse {
            code: ErrorCode::INTERNAL_ERROR.0,
            message: err.to_string(),
            data_json: String::new(),
        }
    }

    pub fn from_service(err: ServiceError) -> crate::plugin::proto::ErrorResponse {
        crate::plugin::proto::ErrorResponse {
            code: ErrorCode::INTERNAL_ERROR.0,
            message: err.to_string(),
            data_json: String::new(),
        }
    }

    pub fn internal(message: impl Into<String>) -> crate::plugin::proto::ErrorResponse {
        crate::plugin::proto::ErrorResponse {
            code: ErrorCode::INTERNAL_ERROR.0,
            message: message.into(),
            data_json: String::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plugin::PluginEndpointSummary;
    use axum::Router;
    use rmcp::model::{
        AnnotateAble, CallToolResult, GetPromptResult, Implementation, ListPromptsResult,
        ListResourceTemplatesResult, ListResourcesResult, ListToolsResult, Prompt, PromptMessage,
        PromptMessageContent, PromptMessageRole, RawResource, RawResourceTemplate,
        ReadResourceRequestParams, ReadResourceResult, ResourceContents, ServerCapabilities,
        ServerInfo, Tool,
    };
    use rmcp::service::RequestContext;
    use rmcp::transport::streamable_http_server::{
        session::local::LocalSessionManager, StreamableHttpService,
    };
    use serde_json::json;
    use std::path::PathBuf;

    struct NoopBridge;

    impl PluginRpcBridge for NoopBridge {
        fn handle_request(
            &self,
            _plugin_name: String,
            _method: String,
            _params_json: String,
        ) -> crate::plugin::BridgeFuture<Result<RpcResult, plugin::proto::ErrorResponse>> {
            Box::pin(async move { Err(proto_error::internal("unexpected test bridge request")) })
        }

        fn handle_notification(
            &self,
            _plugin_name: String,
            _method: String,
            _params_json: String,
        ) -> crate::plugin::BridgeFuture<()> {
            Box::pin(async {})
        }
    }

    struct FakeExternalMcpServer;

    impl ServerHandler for FakeExternalMcpServer {
        fn get_info(&self) -> ServerInfo {
            ServerInfo::new(
                ServerCapabilities::builder()
                    .enable_tools()
                    .enable_prompts()
                    .enable_resources()
                    .build(),
            )
            .with_server_info(Implementation::new("fake-external", "test"))
        }

        async fn list_tools(
            &self,
            _request: Option<PaginatedRequestParams>,
            _context: RequestContext<RoleServer>,
        ) -> Result<ListToolsResult, ErrorData> {
            Ok(ListToolsResult::with_all_items(vec![Tool::new(
                "echo",
                "Echo a message",
                Arc::new(
                    serde_json::json!({
                        "type": "object",
                        "properties": {
                            "message": { "type": "string" }
                        }
                    })
                    .as_object()
                    .cloned()
                    .unwrap(),
                ),
            )]))
        }

        async fn call_tool(
            &self,
            request: CallToolRequestParams,
            _context: RequestContext<RoleServer>,
        ) -> Result<CallToolResult, ErrorData> {
            let message = request
                .arguments
                .as_ref()
                .and_then(|args| args.get("message"))
                .and_then(|value| value.as_str())
                .unwrap_or("missing");
            Ok(CallToolResult::structured(json!({
                "echo": message,
                "tool": request.name.to_string(),
            })))
        }

        async fn list_prompts(
            &self,
            _request: Option<PaginatedRequestParams>,
            _context: RequestContext<RoleServer>,
        ) -> Result<ListPromptsResult, ErrorData> {
            Ok(ListPromptsResult::with_all_items(vec![Prompt::new(
                "brief",
                Some("Write a short brief"),
                None::<Vec<_>>,
            )]))
        }

        async fn get_prompt(
            &self,
            request: GetPromptRequestParams,
            _context: RequestContext<RoleServer>,
        ) -> Result<GetPromptResult, ErrorData> {
            Ok(GetPromptResult::new(vec![PromptMessage::new(
                PromptMessageRole::User,
                PromptMessageContent::text(format!("Prompt: {}", request.name)),
            )])
            .with_description("External prompt"))
        }

        async fn list_resources(
            &self,
            _request: Option<PaginatedRequestParams>,
            _context: RequestContext<RoleServer>,
        ) -> Result<ListResourcesResult, ErrorData> {
            Ok(ListResourcesResult::with_all_items(vec![RawResource::new(
                "note://one",
                "First Note",
            )
            .with_description("External note")
            .no_annotation()]))
        }

        async fn list_resource_templates(
            &self,
            _request: Option<PaginatedRequestParams>,
            _context: RequestContext<RoleServer>,
        ) -> Result<ListResourceTemplatesResult, ErrorData> {
            Ok(ListResourceTemplatesResult::with_all_items(vec![
                RawResourceTemplate::new("note://{id}", "Note Template").no_annotation(),
            ]))
        }

        async fn read_resource(
            &self,
            request: ReadResourceRequestParams,
            _context: RequestContext<RoleServer>,
        ) -> Result<ReadResourceResult, ErrorData> {
            Ok(ReadResourceResult::new(vec![ResourceContents::text(
                format!("resource:{}", request.uri),
                request.uri,
            )]))
        }
    }

    async fn fake_external_client() -> Arc<ExternalMcpClient> {
        let (client_stream, server_stream) = tokio::io::duplex(16 * 1024);
        tokio::spawn(async move {
            let _ = FakeExternalMcpServer
                .serve(server_stream)
                .await
                .unwrap()
                .waiting()
                .await;
        });
        let running = ().serve(client_stream).await.unwrap();
        Arc::new(ExternalMcpClient {
            peer: running.peer().clone(),
            running: Arc::new(Mutex::new(running)),
        })
    }

    async fn spawn_fake_external_tcp_endpoint() -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap().to_string();
        tokio::spawn(async move {
            let Ok((stream, _)) = listener.accept().await else {
                return;
            };
            let _ = FakeExternalMcpServer
                .serve(stream)
                .await
                .unwrap()
                .waiting()
                .await;
        });
        address
    }

    async fn spawn_fake_external_http_endpoint() -> String {
        let service: StreamableHttpService<FakeExternalMcpServer, LocalSessionManager> =
            StreamableHttpService::new(
                || Ok(FakeExternalMcpServer),
                Default::default(),
                Default::default(),
            );
        let router = Router::new().nest_service("/mcp", service);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let _ = axum::serve(listener, router).await;
        });
        format!("http://{address}/mcp")
    }

    #[cfg(unix)]
    async fn spawn_fake_external_unix_endpoint() -> PathBuf {
        let path =
            std::env::temp_dir().join(format!("mesh-llm-mcp-{}.sock", rand::random::<u64>()));
        let _ = std::fs::remove_file(&path);
        let listener = tokio::net::UnixListener::bind(&path).unwrap();
        tokio::spawn(async move {
            let Ok((stream, _)) = listener.accept().await else {
                return;
            };
            let _ = FakeExternalMcpServer
                .serve(stream)
                .await
                .unwrap()
                .waiting()
                .await;
        });
        path
    }

    async fn test_server_with_external_endpoint() -> PluginMcpServer {
        let plugin_manager = PluginManager::for_test_bridge(&[], Arc::new(NoopBridge));
        plugin_manager
            .set_test_endpoints(vec![PluginEndpointSummary {
                plugin_name: "adapter".into(),
                plugin_status: "running".into(),
                endpoint_id: "notes".into(),
                state: "healthy".into(),
                available: true,
                kind: "mcp".into(),
                transport_kind: "stdio".into(),
                protocol: None,
                address: Some("fake-external".into()),
                args: Vec::new(),
                namespace: Some("notes".into()),
                supports_streaming: false,
                managed_by_plugin: false,
                detail: None,
                models: Vec::new(),
            }])
            .await;
        let server = PluginMcpServer::new(plugin_manager, ActiveBridge::default());
        server
            .external_mcp
            .register_test_client("adapter:notes", fake_external_client().await)
            .await;
        server
    }

    #[test]
    fn external_endpoint_namespaces_tools_under_plugin_and_endpoint_namespace() {
        let endpoint = ExternalMcpEndpoint {
            key: "adapter:notes".into(),
            plugin_name: "adapter".into(),
            endpoint_id: "notes".into(),
            transport: ExternalMcpTransport::Stdio {
                command: "fake".into(),
                args: Vec::new(),
            },
            namespace_prefix: "adapter.notes".into(),
        };
        assert_eq!(endpoint.canonical_name("echo"), "adapter.notes.echo");
        assert_eq!(
            endpoint.canonical_resource_uri("note://one"),
            "mesh-mcp://adapter/notes/resource/note%3A%2F%2Fone"
        );
    }

    #[tokio::test]
    async fn external_mcp_endpoint_is_aggregated_into_discovery() {
        let server = test_server_with_external_endpoint().await;

        let tools = server.discover_tools().await.unwrap();
        assert!(tools.contains_key("adapter.notes.echo"));

        let prompts = server.discover_prompts().await.unwrap();
        assert!(prompts.contains_key("adapter.notes.brief"));

        let resources = server.discover_resources().await.unwrap();
        assert!(resources.contains_key("mesh-mcp://adapter/notes/resource/note%3A%2F%2Fone"));

        let endpoint = server
            .active_external_mcp_endpoints()
            .await
            .unwrap()
            .remove(0);
        let client = server.external_mcp.client_for(&endpoint).await.unwrap();
        let result = client
            .peer
            .call_tool(
                CallToolRequestParams::new("echo").with_arguments(
                    serde_json::json!({ "message": "hello" })
                        .as_object()
                        .cloned()
                        .unwrap(),
                ),
            )
            .await
            .unwrap();
        assert_eq!(
            result.structured_content,
            Some(json!({"echo": "hello", "tool": "echo"}))
        );
    }

    #[tokio::test]
    async fn unavailable_external_mcp_endpoint_is_skipped_from_discovery() {
        let plugin_manager = PluginManager::for_test_bridge(&[], Arc::new(NoopBridge));
        plugin_manager
            .set_test_endpoints(vec![PluginEndpointSummary {
                plugin_name: "adapter".into(),
                plugin_status: "running".into(),
                endpoint_id: "notes".into(),
                state: "unhealthy".into(),
                available: false,
                kind: "mcp".into(),
                transport_kind: "stdio".into(),
                protocol: None,
                address: Some("fake-external".into()),
                args: Vec::new(),
                namespace: Some("notes".into()),
                supports_streaming: false,
                managed_by_plugin: false,
                detail: Some("warming".into()),
                models: Vec::new(),
            }])
            .await;
        let server = PluginMcpServer::new(plugin_manager, ActiveBridge::default());

        let tools = server.discover_tools().await.unwrap();
        assert!(!tools.contains_key("adapter.notes.echo"));
    }

    #[tokio::test]
    async fn tcp_external_mcp_endpoint_is_aggregated() {
        let address = spawn_fake_external_tcp_endpoint().await;
        let plugin_manager = PluginManager::for_test_bridge(&[], Arc::new(NoopBridge));
        plugin_manager
            .set_test_endpoints(vec![PluginEndpointSummary {
                plugin_name: "adapter".into(),
                plugin_status: "running".into(),
                endpoint_id: "notes".into(),
                state: "healthy".into(),
                available: true,
                kind: "mcp".into(),
                transport_kind: "tcp".into(),
                protocol: None,
                address: Some(address),
                args: Vec::new(),
                namespace: Some("notes".into()),
                supports_streaming: false,
                managed_by_plugin: false,
                detail: None,
                models: Vec::new(),
            }])
            .await;
        let server = PluginMcpServer::new(plugin_manager, ActiveBridge::default());
        let tools = server.discover_tools().await.unwrap();
        assert!(tools.contains_key("adapter.notes.echo"));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn unix_socket_external_mcp_endpoint_is_aggregated() {
        let path = spawn_fake_external_unix_endpoint().await;
        let plugin_manager = PluginManager::for_test_bridge(&[], Arc::new(NoopBridge));
        plugin_manager
            .set_test_endpoints(vec![PluginEndpointSummary {
                plugin_name: "adapter".into(),
                plugin_status: "running".into(),
                endpoint_id: "notes".into(),
                state: "healthy".into(),
                available: true,
                kind: "mcp".into(),
                transport_kind: "unix_socket".into(),
                protocol: None,
                address: Some(path.display().to_string()),
                args: Vec::new(),
                namespace: Some("notes".into()),
                supports_streaming: false,
                managed_by_plugin: false,
                detail: None,
                models: Vec::new(),
            }])
            .await;
        let server = PluginMcpServer::new(plugin_manager, ActiveBridge::default());
        let tools = server.discover_tools().await.unwrap();
        assert!(tools.contains_key("adapter.notes.echo"));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn http_external_mcp_endpoint_summary_is_recognized() {
        let endpoint = ExternalMcpEndpoint::from_summary(PluginEndpointSummary {
            plugin_name: "adapter".into(),
            plugin_status: "running".into(),
            endpoint_id: "remote".into(),
            state: "healthy".into(),
            available: true,
            kind: "mcp".into(),
            transport_kind: "http".into(),
            protocol: Some("streamable_http".into()),
            address: Some("http://127.0.0.1:9000/mcp".into()),
            args: Vec::new(),
            namespace: Some("remote".into()),
            supports_streaming: true,
            managed_by_plugin: false,
            detail: None,
            models: Vec::new(),
        })
        .expect("http endpoint");
        assert_eq!(endpoint.canonical_name("echo"), "adapter.remote.echo");
        assert_eq!(
            endpoint.transport,
            ExternalMcpTransport::Http {
                uri: "http://127.0.0.1:9000/mcp".into()
            }
        );
    }

    #[tokio::test]
    async fn http_external_mcp_endpoint_is_aggregated() {
        let uri = spawn_fake_external_http_endpoint().await;
        let plugin_manager = PluginManager::for_test_bridge(&[], Arc::new(NoopBridge));
        plugin_manager
            .set_test_endpoints(vec![PluginEndpointSummary {
                plugin_name: "adapter".into(),
                plugin_status: "running".into(),
                endpoint_id: "remote".into(),
                state: "healthy".into(),
                available: true,
                kind: "mcp".into(),
                transport_kind: "http".into(),
                protocol: Some("streamable_http".into()),
                address: Some(uri),
                args: Vec::new(),
                namespace: Some("remote".into()),
                supports_streaming: true,
                managed_by_plugin: false,
                detail: None,
                models: Vec::new(),
            }])
            .await;
        let server = PluginMcpServer::new(plugin_manager, ActiveBridge::default());
        let tools = server.discover_tools().await.unwrap();
        assert!(tools.contains_key("adapter.remote.echo"));
        let prompts = server.discover_prompts().await.unwrap();
        assert!(prompts.contains_key("adapter.remote.brief"));
    }
}
