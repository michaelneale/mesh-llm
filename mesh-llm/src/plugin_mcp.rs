use anyhow::{anyhow, Result};
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
    service::{NotificationContext, Peer, RequestContext},
    transport::io::stdio,
    ErrorData, RoleServer, ServerHandler, ServiceExt,
};
use serde::Serialize;
use std::collections::BTreeMap;
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::plugin::{self, PluginManager, PluginRpcBridge, RpcResult};

#[derive(Clone)]
struct PluginToolRef {
    plugin_name: String,
    tool_name: String,
    tool: rmcp::model::Tool,
}

#[derive(Clone)]
struct PluginPromptRef {
    plugin_name: String,
    prompt_name: String,
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
}

impl PluginMcpServer {
    fn new(plugin_manager: PluginManager, bridge: ActiveBridge) -> Self {
        Self {
            plugin_manager,
            bridge,
        }
    }

    async fn discover_tools(&self) -> Result<BTreeMap<String, PluginToolRef>, ErrorData> {
        let mut tools = BTreeMap::new();
        for (plugin_name, server_info) in self.plugin_manager.list_server_infos().await {
            if server_info.capabilities.tools.is_none() {
                continue;
            }
            let result: ListToolsResult = self
                .plugin_manager
                .mcp_request(
                    &plugin_name,
                    "tools/list",
                    Option::<PaginatedRequestParams>::None,
                )
                .await
                .map_err(internal_error)?;
            for tool in result.tools {
                let raw_name = tool.name.to_string();
                for mcp_name in tool_aliases(&plugin_name, &raw_name) {
                    let mut namespaced = tool.clone();
                    namespaced.name = mcp_name.clone().into();
                    tools.insert(
                        mcp_name.clone(),
                        PluginToolRef {
                            plugin_name: plugin_name.clone(),
                            tool_name: raw_name.clone(),
                            tool: namespaced,
                        },
                    );
                }
            }
        }
        Ok(tools)
    }

    async fn discover_prompts(&self) -> Result<BTreeMap<String, PluginPromptRef>, ErrorData> {
        let mut prompts = BTreeMap::new();
        for (plugin_name, server_info) in self.plugin_manager.list_server_infos().await {
            if server_info.capabilities.prompts.is_none() {
                continue;
            }
            let result: ListPromptsResult = self
                .plugin_manager
                .mcp_request(
                    &plugin_name,
                    "prompts/list",
                    Option::<PaginatedRequestParams>::None,
                )
                .await
                .map_err(internal_error)?;
            for prompt in result.prompts {
                prompts.insert(
                    canonical_name(&plugin_name, &prompt.name),
                    PluginPromptRef {
                        plugin_name: plugin_name.clone(),
                        prompt_name: prompt.name,
                    },
                );
            }
        }
        Ok(prompts)
    }

    async fn refresh_peer(&self, peer: Peer<RoleServer>) {
        self.bridge.set_peer(peer).await;
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

        let mut params = CallToolRequestParams::new(tool_ref.tool_name.clone());
        if let Some(arguments) = request.arguments {
            params = params.with_arguments(arguments);
        }
        if let Some(task) = request.task {
            params = params.with_task(task);
        }
        if let Some(meta) = request.meta {
            params.meta = Some(meta);
        }

        self.plugin_manager
            .mcp_request(&tool_ref.plugin_name, "tools/call", params)
            .await
            .map_err(internal_error)
    }

    async fn list_prompts(
        &self,
        _request: Option<PaginatedRequestParams>,
        context: RequestContext<RoleServer>,
    ) -> Result<ListPromptsResult, ErrorData> {
        self.refresh_peer(context.peer.clone()).await;
        let mut prompts = Vec::new();
        for (plugin_name, server_info) in self.plugin_manager.list_server_infos().await {
            if server_info.capabilities.prompts.is_none() {
                continue;
            }
            let result: ListPromptsResult = self
                .plugin_manager
                .mcp_request(
                    &plugin_name,
                    "prompts/list",
                    Option::<PaginatedRequestParams>::None,
                )
                .await
                .map_err(internal_error)?;
            for mut prompt in result.prompts {
                prompt.name = canonical_name(&plugin_name, &prompt.name);
                prompts.push(prompt);
            }
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

        let mut params = GetPromptRequestParams::new(entry.prompt_name.clone());
        if let Some(arguments) = request.arguments {
            params = params.with_arguments(arguments);
        }
        if let Some(meta) = request.meta {
            params.meta = Some(meta);
        }

        self.plugin_manager
            .mcp_request(&entry.plugin_name, "prompts/get", params)
            .await
            .map_err(internal_error)
    }

    async fn list_resources(
        &self,
        _request: Option<PaginatedRequestParams>,
        context: RequestContext<RoleServer>,
    ) -> Result<ListResourcesResult, ErrorData> {
        self.refresh_peer(context.peer.clone()).await;
        let mut resources = Vec::new();
        for (plugin_name, server_info) in self.plugin_manager.list_server_infos().await {
            if server_info.capabilities.resources.is_none() {
                continue;
            }
            let result: ListResourcesResult = self
                .plugin_manager
                .mcp_request(
                    &plugin_name,
                    "resources/list",
                    Option::<PaginatedRequestParams>::None,
                )
                .await
                .map_err(internal_error)?;
            resources.extend(result.resources);
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
        for (plugin_name, server_info) in self.plugin_manager.list_server_infos().await {
            if server_info.capabilities.resources.is_none() {
                continue;
            }
            let result: ListResourceTemplatesResult = self
                .plugin_manager
                .mcp_request(
                    &plugin_name,
                    "resources/templates/list",
                    Option::<PaginatedRequestParams>::None,
                )
                .await
                .map_err(internal_error)?;
            resource_templates.extend(result.resource_templates);
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
        try_plugins(&self.plugin_manager, "resources/read", request).await
    }

    async fn subscribe(
        &self,
        request: SubscribeRequestParams,
        context: RequestContext<RoleServer>,
    ) -> Result<(), ErrorData> {
        self.refresh_peer(context.peer.clone()).await;
        try_plugins::<(), _>(&self.plugin_manager, "resources/subscribe", request).await
    }

    async fn unsubscribe(
        &self,
        request: UnsubscribeRequestParams,
        context: RequestContext<RoleServer>,
    ) -> Result<(), ErrorData> {
        self.refresh_peer(context.peer.clone()).await;
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
            if let rmcp::model::Reference::Prompt(prompt) = &mut request.r#ref {
                prompt.name = entry.prompt_name.clone();
            }
            return self
                .plugin_manager
                .mcp_request(&entry.plugin_name, "completion/complete", request)
                .await
                .map_err(internal_error);
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
