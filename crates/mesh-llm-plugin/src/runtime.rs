use anyhow::{bail, Result};
use rmcp::model::{
    CallToolResult, CancelTaskParams, CancelTaskResult, CompleteRequestParams, CompleteResult,
    GetPromptRequestParams, GetPromptResult, GetTaskInfoParams, GetTaskPayloadResult,
    GetTaskResult, GetTaskResultParams, ListPromptsResult, ListResourceTemplatesResult,
    ListResourcesResult, ListTasksResult, ListToolsResult, PaginatedRequestParams,
    ReadResourceRequestParams, ReadResourceResult, ServerInfo, SetLevelRequestParams,
    SubscribeRequestParams, UnsubscribeRequestParams,
};
use serde::de::DeserializeOwned;
use std::collections::BTreeMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::{
    context::PluginContext,
    error::{PluginError, PluginResult, PluginRpcResult},
    helpers::{
        json_response, parse_get_prompt_request, parse_read_resource_request, parse_rpc_params,
        parse_tool_call_request, CompletionRouter, PromptRouter, ResourceRouter, TaskRouter,
        ToolCallRequest, ToolRouter,
    },
    io::{connect_from_env, read_envelope, write_envelope, LocalStream},
    proto, PROTOCOL_VERSION,
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MeshVisibility {
    #[default]
    Private,
    Public,
}

impl MeshVisibility {
    fn from_proto(value: i32) -> Self {
        match proto::MeshVisibility::try_from(value).unwrap_or(proto::MeshVisibility::Unspecified) {
            proto::MeshVisibility::Public => Self::Public,
            _ => Self::Private,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PluginStartupPolicy {
    #[default]
    Any,
    PrivateMeshOnly,
    PublicMeshOnly,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct PluginInitializeRequest {
    pub host_protocol_version: u32,
    pub host_version: String,
    pub host_info_json: String,
    pub mesh_visibility: MeshVisibility,
}

impl From<proto::InitializeRequest> for PluginInitializeRequest {
    fn from(value: proto::InitializeRequest) -> Self {
        Self {
            host_protocol_version: value.host_protocol_version,
            host_version: value.host_version,
            host_info_json: value.host_info_json,
            mesh_visibility: MeshVisibility::from_proto(value.mesh_visibility),
        }
    }
}

#[derive(Clone)]
pub struct PluginMetadata {
    plugin_id: String,
    plugin_version: String,
    server_info: ServerInfo,
    capabilities: Vec<String>,
    manifest: Option<proto::PluginManifest>,
    startup_policy: PluginStartupPolicy,
}

impl PluginMetadata {
    pub fn new(
        plugin_id: impl Into<String>,
        plugin_version: impl Into<String>,
        server_info: ServerInfo,
    ) -> Self {
        Self {
            plugin_id: plugin_id.into(),
            plugin_version: plugin_version.into(),
            server_info,
            capabilities: Vec::new(),
            manifest: None,
            startup_policy: PluginStartupPolicy::Any,
        }
    }

    pub fn with_capabilities(mut self, capabilities: Vec<String>) -> Self {
        self.capabilities = capabilities;
        self
    }

    pub fn with_manifest(mut self, manifest: proto::PluginManifest) -> Self {
        self.manifest = Some(manifest);
        self
    }

    pub fn with_startup_policy(mut self, startup_policy: PluginStartupPolicy) -> Self {
        self.startup_policy = startup_policy;
        self
    }
}

type InitializeFuture<'a> = Pin<Box<dyn Future<Output = PluginResult<()>> + Send + 'a>>;
type InitFuture<'a> = Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>>;
type HealthFuture<'a> = Pin<Box<dyn Future<Output = Result<String>> + Send + 'a>>;
type OpenStreamFuture<'a> =
    Pin<Box<dyn Future<Output = PluginResult<Option<proto::OpenStreamResponse>>> + Send + 'a>>;
type RpcMethodFuture<'a> = Pin<Box<dyn Future<Output = PluginRpcResult> + Send + 'a>>;
type SubscribeFuture<'a> = Pin<Box<dyn Future<Output = PluginResult<()>> + Send + 'a>>;
type SetLogLevelFuture<'a> = Pin<Box<dyn Future<Output = PluginResult<()>> + Send + 'a>>;

type InitializeHandler = Arc<
    dyn for<'a, 'ctx> Fn(
            PluginInitializeRequest,
            &'a mut PluginContext<'ctx>,
        ) -> InitializeFuture<'a>
        + Send
        + Sync,
>;
type InitHandler =
    Arc<dyn for<'a, 'ctx> Fn(&'a mut PluginContext<'ctx>) -> InitFuture<'a> + Send + Sync>;
type HealthHandler =
    Arc<dyn for<'a, 'ctx> Fn(&'a mut PluginContext<'ctx>) -> HealthFuture<'a> + Send + Sync>;
type SubscribeHandler = Arc<
    dyn for<'a, 'ctx> Fn(SubscribeRequestParams, &'a mut PluginContext<'ctx>) -> SubscribeFuture<'a>
        + Send
        + Sync,
>;
type UnsubscribeHandler = Arc<
    dyn for<'a, 'ctx> Fn(
            UnsubscribeRequestParams,
            &'a mut PluginContext<'ctx>,
        ) -> SubscribeFuture<'a>
        + Send
        + Sync,
>;
type SetLogLevelHandler = Arc<
    dyn for<'a, 'ctx> Fn(
            SetLevelRequestParams,
            &'a mut PluginContext<'ctx>,
        ) -> SetLogLevelFuture<'a>
        + Send
        + Sync,
>;
type ChannelHandler = Arc<
    dyn for<'a, 'ctx> Fn(proto::ChannelMessage, &'a mut PluginContext<'ctx>) -> InitFuture<'a>
        + Send
        + Sync,
>;
type BulkHandler = Arc<
    dyn for<'a, 'ctx> Fn(proto::BulkTransferMessage, &'a mut PluginContext<'ctx>) -> InitFuture<'a>
        + Send
        + Sync,
>;
type MeshEventHandler = Arc<
    dyn for<'a, 'ctx> Fn(proto::MeshEvent, &'a mut PluginContext<'ctx>) -> InitFuture<'a>
        + Send
        + Sync,
>;
type RpcMethodHandler = Arc<
    dyn for<'a, 'ctx> Fn(proto::RpcRequest, &'a mut PluginContext<'ctx>) -> RpcMethodFuture<'a>
        + Send
        + Sync,
>;
type OpenStreamHandler = Arc<
    dyn for<'a, 'ctx> Fn(
            proto::OpenStreamRequest,
            &'a mut PluginContext<'ctx>,
        ) -> OpenStreamFuture<'a>
        + Send
        + Sync,
>;
type CancelStreamHandler = Arc<
    dyn for<'a, 'ctx> Fn(
            proto::CancelStreamNotification,
            &'a mut PluginContext<'ctx>,
        ) -> InitFuture<'a>
        + Send
        + Sync,
>;
type CloseStreamHandler = Arc<
    dyn for<'a, 'ctx> Fn(
            proto::CloseStreamNotification,
            &'a mut PluginContext<'ctx>,
        ) -> InitFuture<'a>
        + Send
        + Sync,
>;
type StreamErrorHandler = Arc<
    dyn for<'a, 'ctx> Fn(proto::StreamError, &'a mut PluginContext<'ctx>) -> InitFuture<'a>
        + Send
        + Sync,
>;

#[crate::async_trait]
pub trait Plugin: Send {
    fn plugin_id(&self) -> &str;
    fn plugin_version(&self) -> String;
    fn server_info(&self) -> ServerInfo;

    fn capabilities(&self) -> Vec<String> {
        Vec::new()
    }

    fn manifest(&self) -> Option<proto::PluginManifest> {
        None
    }

    async fn initialize(
        &mut self,
        _request: PluginInitializeRequest,
        _context: &mut PluginContext<'_>,
    ) -> PluginResult<()> {
        Ok(())
    }

    async fn on_initialized(&mut self, _context: &mut PluginContext<'_>) -> Result<()> {
        Ok(())
    }

    async fn health(&mut self, _context: &mut PluginContext<'_>) -> Result<String> {
        Ok("ok".into())
    }

    async fn list_tools(
        &mut self,
        _context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<ListToolsResult>> {
        Ok(None)
    }

    async fn call_tool(
        &mut self,
        _request: ToolCallRequest,
        _context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<CallToolResult>> {
        Ok(None)
    }

    async fn list_prompts(
        &mut self,
        _request: Option<PaginatedRequestParams>,
        _context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<ListPromptsResult>> {
        Ok(None)
    }

    async fn get_prompt(
        &mut self,
        _request: GetPromptRequestParams,
        _context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<GetPromptResult>> {
        Ok(None)
    }

    async fn list_resources(
        &mut self,
        _request: Option<PaginatedRequestParams>,
        _context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<ListResourcesResult>> {
        Ok(None)
    }

    async fn read_resource(
        &mut self,
        _request: ReadResourceRequestParams,
        _context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<ReadResourceResult>> {
        Ok(None)
    }

    async fn list_resource_templates(
        &mut self,
        _request: Option<PaginatedRequestParams>,
        _context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<ListResourceTemplatesResult>> {
        Ok(None)
    }

    async fn subscribe_resource(
        &mut self,
        _request: SubscribeRequestParams,
        _context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<()>> {
        Ok(None)
    }

    async fn unsubscribe_resource(
        &mut self,
        _request: UnsubscribeRequestParams,
        _context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<()>> {
        Ok(None)
    }

    async fn complete(
        &mut self,
        _request: CompleteRequestParams,
        _context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<CompleteResult>> {
        Ok(None)
    }

    async fn set_log_level(
        &mut self,
        _request: SetLevelRequestParams,
        _context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<()>> {
        Ok(None)
    }

    async fn list_tasks(
        &mut self,
        _request: Option<PaginatedRequestParams>,
        _context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<ListTasksResult>> {
        Ok(None)
    }

    async fn get_task_info(
        &mut self,
        _request: GetTaskInfoParams,
        _context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<GetTaskResult>> {
        Ok(None)
    }

    async fn get_task_result(
        &mut self,
        _request: GetTaskResultParams,
        _context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<GetTaskPayloadResult>> {
        Ok(None)
    }

    async fn cancel_task(
        &mut self,
        _request: CancelTaskParams,
        _context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<CancelTaskResult>> {
        Ok(None)
    }

    async fn invoke_service(
        &mut self,
        request: proto::InvokeServiceRequest,
        context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<proto::InvokeServiceResponse>> {
        match proto::ServiceKind::try_from(request.kind).unwrap_or(proto::ServiceKind::Unspecified)
        {
            proto::ServiceKind::Operation => {
                let arguments = parse_service_input::<serde_json::Value>(&request.input_json)?;
                let tool_request = ToolCallRequest {
                    name: request.service_name,
                    arguments,
                };
                match self.call_tool(tool_request, context).await? {
                    Some(result) => Ok(Some(proto::InvokeServiceResponse {
                        output_json: normalize_call_tool_output(&result)?,
                        is_error: result.is_error.unwrap_or(false),
                    })),
                    None => Ok(None),
                }
            }
            proto::ServiceKind::Prompt => {
                let params = parse_service_input::<GetPromptRequestParams>(&request.input_json)?;
                match self.get_prompt(params, context).await? {
                    Some(result) => Ok(Some(proto::InvokeServiceResponse {
                        output_json: serialize_service_output(&result)?,
                        is_error: false,
                    })),
                    None => Ok(None),
                }
            }
            proto::ServiceKind::Resource => {
                let params = parse_service_input::<ReadResourceRequestParams>(&request.input_json)?;
                match self.read_resource(params, context).await? {
                    Some(result) => Ok(Some(proto::InvokeServiceResponse {
                        output_json: serialize_service_output(&result)?,
                        is_error: false,
                    })),
                    None => Ok(None),
                }
            }
            proto::ServiceKind::Completion => {
                let params = parse_service_input::<CompleteRequestParams>(&request.input_json)?;
                match self.complete(params, context).await? {
                    Some(result) => Ok(Some(proto::InvokeServiceResponse {
                        output_json: serialize_service_output(&result)?,
                        is_error: false,
                    })),
                    None => Ok(None),
                }
            }
            proto::ServiceKind::Unspecified => Err(PluginError::invalid_request(
                "Service invocation kind is required",
            )),
        }
    }

    async fn handle_rpc(
        &mut self,
        request: proto::RpcRequest,
        context: &mut PluginContext<'_>,
    ) -> PluginRpcResult {
        match request.method.as_str() {
            "tools/list" => match self.list_tools(context).await? {
                Some(result) => json_response(&result),
                None => Err(PluginError::method_not_found(
                    "Unsupported MCP method 'tools/list'",
                )),
            },
            "tools/call" => {
                let tool_call = parse_tool_call_request(&request)?;
                match self.call_tool(tool_call, context).await? {
                    Some(result) => json_response(&result),
                    None => Err(PluginError::method_not_found(
                        "Unsupported MCP method 'tools/call'",
                    )),
                }
            }
            "prompts/list" => {
                let params: Option<PaginatedRequestParams> = parse_rpc_params(&request)?;
                match self.list_prompts(params, context).await? {
                    Some(result) => json_response(&result),
                    None => Err(PluginError::method_not_found(
                        "Unsupported MCP method 'prompts/list'",
                    )),
                }
            }
            "prompts/get" => {
                let params = parse_get_prompt_request(&request)?;
                match self.get_prompt(params, context).await? {
                    Some(result) => json_response(&result),
                    None => Err(PluginError::method_not_found(
                        "Unsupported MCP method 'prompts/get'",
                    )),
                }
            }
            "resources/list" => {
                let params: Option<PaginatedRequestParams> = parse_rpc_params(&request)?;
                match self.list_resources(params, context).await? {
                    Some(result) => json_response(&result),
                    None => Err(PluginError::method_not_found(
                        "Unsupported MCP method 'resources/list'",
                    )),
                }
            }
            "resources/read" => {
                let params = parse_read_resource_request(&request)?;
                match self.read_resource(params, context).await? {
                    Some(result) => json_response(&result),
                    None => Err(PluginError::method_not_found(
                        "Unsupported MCP method 'resources/read'",
                    )),
                }
            }
            "resources/templates/list" => {
                let params: Option<PaginatedRequestParams> = parse_rpc_params(&request)?;
                match self.list_resource_templates(params, context).await? {
                    Some(result) => json_response(&result),
                    None => Err(PluginError::method_not_found(
                        "Unsupported MCP method 'resources/templates/list'",
                    )),
                }
            }
            "resources/subscribe" => {
                let params: SubscribeRequestParams = parse_rpc_params(&request)?;
                match self.subscribe_resource(params, context).await? {
                    Some(()) => json_response(&serde_json::json!({})),
                    None => Err(PluginError::method_not_found(
                        "Unsupported MCP method 'resources/subscribe'",
                    )),
                }
            }
            "resources/unsubscribe" => {
                let params: UnsubscribeRequestParams = parse_rpc_params(&request)?;
                match self.unsubscribe_resource(params, context).await? {
                    Some(()) => json_response(&serde_json::json!({})),
                    None => Err(PluginError::method_not_found(
                        "Unsupported MCP method 'resources/unsubscribe'",
                    )),
                }
            }
            "completion/complete" => {
                let params: CompleteRequestParams = parse_rpc_params(&request)?;
                match self.complete(params, context).await? {
                    Some(result) => json_response(&result),
                    None => Err(PluginError::method_not_found(
                        "Unsupported MCP method 'completion/complete'",
                    )),
                }
            }
            "logging/setLevel" => {
                let params: SetLevelRequestParams = parse_rpc_params(&request)?;
                match self.set_log_level(params, context).await? {
                    Some(()) => json_response(&serde_json::json!({})),
                    None => Err(PluginError::method_not_found(
                        "Unsupported MCP method 'logging/setLevel'",
                    )),
                }
            }
            "tasks/list" => {
                let params: Option<PaginatedRequestParams> = parse_rpc_params(&request)?;
                match self.list_tasks(params, context).await? {
                    Some(result) => json_response(&result),
                    None => Err(PluginError::method_not_found(
                        "Unsupported MCP method 'tasks/list'",
                    )),
                }
            }
            "tasks/get" => {
                let params: GetTaskInfoParams = parse_rpc_params(&request)?;
                match self.get_task_info(params, context).await? {
                    Some(result) => json_response(&result),
                    None => Err(PluginError::method_not_found(
                        "Unsupported MCP method 'tasks/get'",
                    )),
                }
            }
            "tasks/result" => {
                let params: GetTaskResultParams = parse_rpc_params(&request)?;
                match self.get_task_result(params, context).await? {
                    Some(result) => json_response(&result),
                    None => Err(PluginError::method_not_found(
                        "Unsupported MCP method 'tasks/result'",
                    )),
                }
            }
            "tasks/cancel" => {
                let params: CancelTaskParams = parse_rpc_params(&request)?;
                match self.cancel_task(params, context).await? {
                    Some(result) => json_response(&result),
                    None => Err(PluginError::method_not_found(
                        "Unsupported MCP method 'tasks/cancel'",
                    )),
                }
            }
            _ => Err(PluginError::method_not_found(format!(
                "Unsupported MCP method '{}'",
                request.method
            ))),
        }
    }

    async fn on_rpc_notification(
        &mut self,
        _notification: proto::RpcNotification,
        _context: &mut PluginContext<'_>,
    ) -> Result<()> {
        Ok(())
    }

    async fn on_channel_message(
        &mut self,
        _message: proto::ChannelMessage,
        _context: &mut PluginContext<'_>,
    ) -> Result<()> {
        Ok(())
    }

    async fn on_bulk_transfer_message(
        &mut self,
        _message: proto::BulkTransferMessage,
        _context: &mut PluginContext<'_>,
    ) -> Result<()> {
        Ok(())
    }

    async fn on_mesh_event(
        &mut self,
        _event: proto::MeshEvent,
        _context: &mut PluginContext<'_>,
    ) -> Result<()> {
        Ok(())
    }

    async fn open_stream(
        &mut self,
        _request: proto::OpenStreamRequest,
        _context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<proto::OpenStreamResponse>> {
        Ok(None)
    }

    async fn on_cancel_stream(
        &mut self,
        _notification: proto::CancelStreamNotification,
        _context: &mut PluginContext<'_>,
    ) -> Result<()> {
        Ok(())
    }

    async fn on_close_stream(
        &mut self,
        _notification: proto::CloseStreamNotification,
        _context: &mut PluginContext<'_>,
    ) -> Result<()> {
        Ok(())
    }

    async fn on_stream_error(
        &mut self,
        _error: proto::StreamError,
        _context: &mut PluginContext<'_>,
    ) -> Result<()> {
        Ok(())
    }

    async fn on_host_error(
        &mut self,
        error: proto::ErrorResponse,
        _context: &mut PluginContext<'_>,
    ) -> Result<()> {
        bail!("host error: {}", error.message)
    }
}

pub struct SimplePlugin {
    metadata: PluginMetadata,
    operation_router: Option<ToolRouter>,
    prompt_router: Option<PromptRouter>,
    resource_router: Option<ResourceRouter>,
    completion_router: Option<CompletionRouter>,
    task_router: Option<TaskRouter>,
    initialize_handler: Option<InitializeHandler>,
    on_initialized: Option<InitHandler>,
    health_handler: Option<HealthHandler>,
    subscribe_handler: Option<SubscribeHandler>,
    unsubscribe_handler: Option<UnsubscribeHandler>,
    set_log_level_handler: Option<SetLogLevelHandler>,
    channel_handler: Option<ChannelHandler>,
    bulk_handler: Option<BulkHandler>,
    mesh_event_handler: Option<MeshEventHandler>,
    open_stream_handler: Option<OpenStreamHandler>,
    cancel_stream_handler: Option<CancelStreamHandler>,
    close_stream_handler: Option<CloseStreamHandler>,
    stream_error_handler: Option<StreamErrorHandler>,
}

impl SimplePlugin {
    pub fn new(metadata: PluginMetadata) -> Self {
        Self {
            metadata,
            operation_router: None,
            prompt_router: None,
            resource_router: None,
            completion_router: None,
            task_router: None,
            initialize_handler: None,
            on_initialized: None,
            health_handler: None,
            subscribe_handler: None,
            unsubscribe_handler: None,
            set_log_level_handler: None,
            channel_handler: None,
            bulk_handler: None,
            mesh_event_handler: None,
            open_stream_handler: None,
            cancel_stream_handler: None,
            close_stream_handler: None,
            stream_error_handler: None,
        }
    }

    pub fn with_capabilities(mut self, capabilities: Vec<String>) -> Self {
        self.metadata = self.metadata.with_capabilities(capabilities);
        self
    }

    pub fn with_manifest(mut self, manifest: proto::PluginManifest) -> Self {
        self.metadata = self.metadata.with_manifest(manifest);
        self
    }

    pub fn with_startup_policy(mut self, startup_policy: PluginStartupPolicy) -> Self {
        self.metadata = self.metadata.with_startup_policy(startup_policy);
        self
    }

    pub fn with_operation_router(mut self, router: ToolRouter) -> Self {
        self.operation_router = Some(router);
        self
    }

    pub fn with_prompt_router(mut self, router: PromptRouter) -> Self {
        self.prompt_router = Some(router);
        self
    }

    pub fn with_resource_router(mut self, router: ResourceRouter) -> Self {
        self.resource_router = Some(router);
        self
    }

    pub fn with_completion_router(mut self, router: CompletionRouter) -> Self {
        self.completion_router = Some(router);
        self
    }

    pub fn with_task_router(mut self, router: TaskRouter) -> Self {
        self.task_router = Some(router);
        self
    }

    pub fn on_initialize<F>(mut self, handler: F) -> Self
    where
        F: for<'a, 'ctx> Fn(
                PluginInitializeRequest,
                &'a mut PluginContext<'ctx>,
            ) -> InitializeFuture<'a>
            + Send
            + Sync
            + 'static,
    {
        self.initialize_handler = Some(Arc::new(handler));
        self
    }

    pub fn on_initialized<F>(mut self, handler: F) -> Self
    where
        F: for<'a, 'ctx> Fn(&'a mut PluginContext<'ctx>) -> InitFuture<'a> + Send + Sync + 'static,
    {
        self.on_initialized = Some(Arc::new(handler));
        self
    }

    pub fn with_health<F>(mut self, handler: F) -> Self
    where
        F: for<'a, 'ctx> Fn(&'a mut PluginContext<'ctx>) -> HealthFuture<'a>
            + Send
            + Sync
            + 'static,
    {
        self.health_handler = Some(Arc::new(handler));
        self
    }

    pub fn with_subscribe_resource<F>(mut self, handler: F) -> Self
    where
        F: for<'a, 'ctx> Fn(
                SubscribeRequestParams,
                &'a mut PluginContext<'ctx>,
            ) -> SubscribeFuture<'a>
            + Send
            + Sync
            + 'static,
    {
        self.subscribe_handler = Some(Arc::new(handler));
        self
    }

    pub fn with_unsubscribe_resource<F>(mut self, handler: F) -> Self
    where
        F: for<'a, 'ctx> Fn(
                UnsubscribeRequestParams,
                &'a mut PluginContext<'ctx>,
            ) -> SubscribeFuture<'a>
            + Send
            + Sync
            + 'static,
    {
        self.unsubscribe_handler = Some(Arc::new(handler));
        self
    }

    pub fn with_set_log_level<F>(mut self, handler: F) -> Self
    where
        F: for<'a, 'ctx> Fn(
                SetLevelRequestParams,
                &'a mut PluginContext<'ctx>,
            ) -> SetLogLevelFuture<'a>
            + Send
            + Sync
            + 'static,
    {
        self.set_log_level_handler = Some(Arc::new(handler));
        self
    }

    pub fn on_channel_message<F>(mut self, handler: F) -> Self
    where
        F: for<'a, 'ctx> Fn(proto::ChannelMessage, &'a mut PluginContext<'ctx>) -> InitFuture<'a>
            + Send
            + Sync
            + 'static,
    {
        self.channel_handler = Some(Arc::new(handler));
        self
    }

    pub fn on_bulk_transfer_message<F>(mut self, handler: F) -> Self
    where
        F: for<'a, 'ctx> Fn(
                proto::BulkTransferMessage,
                &'a mut PluginContext<'ctx>,
            ) -> InitFuture<'a>
            + Send
            + Sync
            + 'static,
    {
        self.bulk_handler = Some(Arc::new(handler));
        self
    }

    pub fn on_mesh_event<F>(mut self, handler: F) -> Self
    where
        F: for<'a, 'ctx> Fn(proto::MeshEvent, &'a mut PluginContext<'ctx>) -> InitFuture<'a>
            + Send
            + Sync
            + 'static,
    {
        self.mesh_event_handler = Some(Arc::new(handler));
        self
    }

    pub fn on_open_stream<F>(mut self, handler: F) -> Self
    where
        F: for<'a, 'ctx> Fn(
                proto::OpenStreamRequest,
                &'a mut PluginContext<'ctx>,
            ) -> OpenStreamFuture<'a>
            + Send
            + Sync
            + 'static,
    {
        self.open_stream_handler = Some(Arc::new(handler));
        self
    }

    pub fn on_cancel_stream<F>(mut self, handler: F) -> Self
    where
        F: for<'a, 'ctx> Fn(
                proto::CancelStreamNotification,
                &'a mut PluginContext<'ctx>,
            ) -> InitFuture<'a>
            + Send
            + Sync
            + 'static,
    {
        self.cancel_stream_handler = Some(Arc::new(handler));
        self
    }

    pub fn on_close_stream<F>(mut self, handler: F) -> Self
    where
        F: for<'a, 'ctx> Fn(
                proto::CloseStreamNotification,
                &'a mut PluginContext<'ctx>,
            ) -> InitFuture<'a>
            + Send
            + Sync
            + 'static,
    {
        self.close_stream_handler = Some(Arc::new(handler));
        self
    }

    pub fn on_stream_error<F>(mut self, handler: F) -> Self
    where
        F: for<'a, 'ctx> Fn(proto::StreamError, &'a mut PluginContext<'ctx>) -> InitFuture<'a>
            + Send
            + Sync
            + 'static,
    {
        self.stream_error_handler = Some(Arc::new(handler));
        self
    }
}

pub struct InternalRpcPluginBuilder {
    plugin: SimplePlugin,
    rpc_handlers: BTreeMap<String, RpcMethodHandler>,
}

impl InternalRpcPluginBuilder {
    pub fn new(metadata: PluginMetadata) -> Self {
        Self {
            plugin: SimplePlugin::new(metadata),
            rpc_handlers: BTreeMap::new(),
        }
    }

    pub fn with_capabilities(mut self, capabilities: Vec<String>) -> Self {
        self.plugin = self.plugin.with_capabilities(capabilities);
        self
    }

    pub fn with_manifest(mut self, manifest: proto::PluginManifest) -> Self {
        self.plugin = self.plugin.with_manifest(manifest);
        self
    }

    pub fn with_startup_policy(mut self, startup_policy: PluginStartupPolicy) -> Self {
        self.plugin = self.plugin.with_startup_policy(startup_policy);
        self
    }

    pub fn with_operation_router(mut self, router: ToolRouter) -> Self {
        self.plugin = self.plugin.with_operation_router(router);
        self
    }

    pub fn with_health<F>(mut self, handler: F) -> Self
    where
        F: for<'a, 'ctx> Fn(&'a mut PluginContext<'ctx>) -> HealthFuture<'a>
            + Send
            + Sync
            + 'static,
    {
        self.plugin = self.plugin.with_health(handler);
        self
    }

    pub fn rpc_method<F>(mut self, method: impl Into<String>, handler: F) -> Self
    where
        F: for<'a, 'ctx> Fn(proto::RpcRequest, &'a mut PluginContext<'ctx>) -> RpcMethodFuture<'a>
            + Send
            + Sync
            + 'static,
    {
        self.rpc_handlers.insert(method.into(), Arc::new(handler));
        self
    }

    pub fn build(self) -> InternalRpcPlugin {
        InternalRpcPlugin {
            plugin: self.plugin,
            rpc_handlers: self.rpc_handlers,
        }
    }
}

pub struct InternalRpcPlugin {
    plugin: SimplePlugin,
    rpc_handlers: BTreeMap<String, RpcMethodHandler>,
}

#[crate::async_trait]
impl Plugin for InternalRpcPlugin {
    fn plugin_id(&self) -> &str {
        self.plugin.plugin_id()
    }

    fn plugin_version(&self) -> String {
        self.plugin.plugin_version()
    }

    fn server_info(&self) -> ServerInfo {
        self.plugin.server_info()
    }

    fn capabilities(&self) -> Vec<String> {
        self.plugin.capabilities()
    }

    fn manifest(&self) -> Option<proto::PluginManifest> {
        self.plugin.manifest()
    }

    async fn initialize(
        &mut self,
        request: PluginInitializeRequest,
        context: &mut PluginContext<'_>,
    ) -> PluginResult<()> {
        self.plugin.initialize(request, context).await
    }

    async fn on_initialized(&mut self, context: &mut PluginContext<'_>) -> Result<()> {
        <SimplePlugin as Plugin>::on_initialized(&mut self.plugin, context).await
    }

    async fn health(&mut self, context: &mut PluginContext<'_>) -> Result<String> {
        self.plugin.health(context).await
    }

    async fn list_tools(
        &mut self,
        context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<ListToolsResult>> {
        self.plugin.list_tools(context).await
    }

    async fn call_tool(
        &mut self,
        request: ToolCallRequest,
        context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<CallToolResult>> {
        self.plugin.call_tool(request, context).await
    }

    async fn handle_rpc(
        &mut self,
        request: proto::RpcRequest,
        context: &mut PluginContext<'_>,
    ) -> PluginRpcResult {
        if let Some(handler) = self.rpc_handlers.get(&request.method).cloned() {
            return handler(request, context).await;
        }
        self.plugin.handle_rpc(request, context).await
    }
}

#[crate::async_trait]
impl Plugin for SimplePlugin {
    fn plugin_id(&self) -> &str {
        &self.metadata.plugin_id
    }

    fn plugin_version(&self) -> String {
        self.metadata.plugin_version.clone()
    }

    fn server_info(&self) -> ServerInfo {
        self.metadata.server_info.clone()
    }

    fn capabilities(&self) -> Vec<String> {
        self.metadata.capabilities.clone()
    }

    fn manifest(&self) -> Option<proto::PluginManifest> {
        self.metadata.manifest.clone()
    }

    async fn initialize(
        &mut self,
        request: PluginInitializeRequest,
        context: &mut PluginContext<'_>,
    ) -> PluginResult<()> {
        match self.metadata.startup_policy {
            PluginStartupPolicy::Any => {}
            PluginStartupPolicy::PrivateMeshOnly
                if request.mesh_visibility != MeshVisibility::Private =>
            {
                return Err(PluginError::startup_disabled(format!(
                    "Plugin '{}' requires a private mesh",
                    self.metadata.plugin_id
                )));
            }
            PluginStartupPolicy::PublicMeshOnly
                if request.mesh_visibility != MeshVisibility::Public =>
            {
                return Err(PluginError::startup_disabled(format!(
                    "Plugin '{}' requires a public mesh",
                    self.metadata.plugin_id
                )));
            }
            PluginStartupPolicy::PrivateMeshOnly | PluginStartupPolicy::PublicMeshOnly => {}
        }
        match &self.initialize_handler {
            Some(handler) => handler(request, context).await,
            None => Ok(()),
        }
    }

    async fn on_initialized(&mut self, context: &mut PluginContext<'_>) -> Result<()> {
        match &self.on_initialized {
            Some(handler) => handler(context).await,
            None => Ok(()),
        }
    }

    async fn health(&mut self, context: &mut PluginContext<'_>) -> Result<String> {
        match &self.health_handler {
            Some(handler) => handler(context).await,
            None => Ok("ok".into()),
        }
    }

    async fn list_tools(
        &mut self,
        _context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<ListToolsResult>> {
        Ok(self
            .operation_router
            .as_ref()
            .map(|router| router.list_tools_result()))
    }

    async fn call_tool(
        &mut self,
        request: ToolCallRequest,
        context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<CallToolResult>> {
        match &self.operation_router {
            Some(router) => Ok(Some(router.call(request, context).await?)),
            None => Ok(None),
        }
    }

    async fn list_prompts(
        &mut self,
        _request: Option<PaginatedRequestParams>,
        _context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<ListPromptsResult>> {
        Ok(self
            .prompt_router
            .as_ref()
            .map(|router| router.list_prompts_result()))
    }

    async fn get_prompt(
        &mut self,
        request: GetPromptRequestParams,
        context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<GetPromptResult>> {
        match &self.prompt_router {
            Some(router) => Ok(Some(router.get(request, context).await?)),
            None => Ok(None),
        }
    }

    async fn list_resources(
        &mut self,
        _request: Option<PaginatedRequestParams>,
        _context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<ListResourcesResult>> {
        Ok(self
            .resource_router
            .as_ref()
            .map(|router| router.list_resources_result()))
    }

    async fn read_resource(
        &mut self,
        request: ReadResourceRequestParams,
        context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<ReadResourceResult>> {
        match &self.resource_router {
            Some(router) => Ok(Some(router.read(request, context).await?)),
            None => Ok(None),
        }
    }

    async fn list_resource_templates(
        &mut self,
        _request: Option<PaginatedRequestParams>,
        _context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<ListResourceTemplatesResult>> {
        Ok(self
            .resource_router
            .as_ref()
            .map(|router| router.list_resource_templates_result()))
    }

    async fn subscribe_resource(
        &mut self,
        request: SubscribeRequestParams,
        context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<()>> {
        match &self.subscribe_handler {
            Some(handler) => Ok(Some(handler(request, context).await?)),
            None => Ok(None),
        }
    }

    async fn unsubscribe_resource(
        &mut self,
        request: UnsubscribeRequestParams,
        context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<()>> {
        match &self.unsubscribe_handler {
            Some(handler) => Ok(Some(handler(request, context).await?)),
            None => Ok(None),
        }
    }

    async fn complete(
        &mut self,
        request: CompleteRequestParams,
        context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<CompleteResult>> {
        match &self.completion_router {
            Some(router) => Ok(Some(router.complete(request, context).await?)),
            None => Ok(None),
        }
    }

    async fn set_log_level(
        &mut self,
        request: SetLevelRequestParams,
        context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<()>> {
        match &self.set_log_level_handler {
            Some(handler) => Ok(Some(handler(request, context).await?)),
            None => Ok(None),
        }
    }

    async fn list_tasks(
        &mut self,
        request: Option<PaginatedRequestParams>,
        context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<ListTasksResult>> {
        match &self.task_router {
            Some(router) => router.list_tasks(request, context).await,
            None => Ok(None),
        }
    }

    async fn get_task_info(
        &mut self,
        request: GetTaskInfoParams,
        context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<GetTaskResult>> {
        match &self.task_router {
            Some(router) => router.get_task_info(request, context).await,
            None => Ok(None),
        }
    }

    async fn get_task_result(
        &mut self,
        request: GetTaskResultParams,
        context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<GetTaskPayloadResult>> {
        match &self.task_router {
            Some(router) => router.get_task_result(request, context).await,
            None => Ok(None),
        }
    }

    async fn cancel_task(
        &mut self,
        request: CancelTaskParams,
        context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<CancelTaskResult>> {
        match &self.task_router {
            Some(router) => router.cancel_task(request, context).await,
            None => Ok(None),
        }
    }

    async fn on_channel_message(
        &mut self,
        message: proto::ChannelMessage,
        context: &mut PluginContext<'_>,
    ) -> Result<()> {
        match &self.channel_handler {
            Some(handler) => handler(message, context).await,
            None => Ok(()),
        }
    }

    async fn on_bulk_transfer_message(
        &mut self,
        message: proto::BulkTransferMessage,
        context: &mut PluginContext<'_>,
    ) -> Result<()> {
        match &self.bulk_handler {
            Some(handler) => handler(message, context).await,
            None => Ok(()),
        }
    }

    async fn on_mesh_event(
        &mut self,
        event: proto::MeshEvent,
        context: &mut PluginContext<'_>,
    ) -> Result<()> {
        match &self.mesh_event_handler {
            Some(handler) => handler(event, context).await,
            None => Ok(()),
        }
    }

    async fn open_stream(
        &mut self,
        request: proto::OpenStreamRequest,
        context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<proto::OpenStreamResponse>> {
        match &self.open_stream_handler {
            Some(handler) => handler(request, context).await,
            None => Ok(None),
        }
    }

    async fn on_cancel_stream(
        &mut self,
        notification: proto::CancelStreamNotification,
        context: &mut PluginContext<'_>,
    ) -> Result<()> {
        match &self.cancel_stream_handler {
            Some(handler) => handler(notification, context).await,
            None => Ok(()),
        }
    }

    async fn on_close_stream(
        &mut self,
        notification: proto::CloseStreamNotification,
        context: &mut PluginContext<'_>,
    ) -> Result<()> {
        match &self.close_stream_handler {
            Some(handler) => handler(notification, context).await,
            None => Ok(()),
        }
    }

    async fn on_stream_error(
        &mut self,
        error: proto::StreamError,
        context: &mut PluginContext<'_>,
    ) -> Result<()> {
        match &self.stream_error_handler {
            Some(handler) => handler(error, context).await,
            None => Ok(()),
        }
    }
}

pub struct PluginRuntime;

impl PluginRuntime {
    pub async fn run<P: Plugin>(plugin: P) -> Result<()> {
        let stream = connect_from_env().await?;
        Self::run_with_stream(plugin, stream).await
    }

    pub async fn run_with_stream<P: Plugin>(mut plugin: P, mut stream: LocalStream) -> Result<()> {
        loop {
            let envelope = read_envelope(&mut stream).await?;
            let request_id = envelope.request_id;
            let plugin_id = plugin.plugin_id().to_string();

            match envelope.payload {
                Some(proto::envelope::Payload::InitializeRequest(request)) => {
                    let init_result = {
                        let mut context = PluginContext {
                            stream: &mut stream,
                            plugin_id: &plugin_id,
                        };
                        plugin
                            .initialize(PluginInitializeRequest::from(request), &mut context)
                            .await
                    };
                    if let Err(err) = init_result {
                        let response = proto::Envelope {
                            protocol_version: PROTOCOL_VERSION,
                            plugin_id: plugin_id.clone(),
                            request_id,
                            payload: Some(proto::envelope::Payload::ErrorResponse(
                                err.into_error_response(),
                            )),
                        };
                        write_envelope(&mut stream, &response).await?;
                        break;
                    }
                    let response = proto::Envelope {
                        protocol_version: PROTOCOL_VERSION,
                        plugin_id: plugin_id.clone(),
                        request_id,
                        payload: Some(proto::envelope::Payload::InitializeResponse(
                            proto::InitializeResponse {
                                plugin_id: plugin_id.clone(),
                                plugin_protocol_version: PROTOCOL_VERSION,
                                plugin_version: plugin.plugin_version(),
                                server_info_json: serde_json::to_string(&plugin.server_info())?,
                                capabilities: plugin.capabilities(),
                                manifest: plugin.manifest(),
                            },
                        )),
                    };
                    write_envelope(&mut stream, &response).await?;
                    let mut context = PluginContext {
                        stream: &mut stream,
                        plugin_id: &plugin_id,
                    };
                    plugin.on_initialized(&mut context).await?;
                }
                Some(proto::envelope::Payload::HealthRequest(_)) => {
                    let detail = {
                        let mut context = PluginContext {
                            stream: &mut stream,
                            plugin_id: &plugin_id,
                        };
                        plugin.health(&mut context).await?
                    };
                    let response = proto::Envelope {
                        protocol_version: PROTOCOL_VERSION,
                        plugin_id: plugin_id.clone(),
                        request_id,
                        payload: Some(proto::envelope::Payload::HealthResponse(
                            proto::HealthResponse {
                                status: proto::health_response::Status::Ok as i32,
                                detail,
                            },
                        )),
                    };
                    write_envelope(&mut stream, &response).await?;
                }
                Some(proto::envelope::Payload::ShutdownRequest(_)) => {
                    let response = proto::Envelope {
                        protocol_version: PROTOCOL_VERSION,
                        plugin_id: plugin_id.clone(),
                        request_id,
                        payload: Some(proto::envelope::Payload::ShutdownResponse(
                            proto::ShutdownResponse {},
                        )),
                    };
                    write_envelope(&mut stream, &response).await?;
                    break;
                }
                Some(proto::envelope::Payload::RpcRequest(request)) => {
                    let payload = {
                        let mut context = PluginContext {
                            stream: &mut stream,
                            plugin_id: &plugin_id,
                        };
                        match plugin.handle_rpc(request, &mut context).await {
                            Ok(payload) => payload,
                            Err(err) => {
                                proto::envelope::Payload::ErrorResponse(err.into_error_response())
                            }
                        }
                    };
                    write_envelope(
                        &mut stream,
                        &proto::Envelope {
                            protocol_version: PROTOCOL_VERSION,
                            plugin_id: plugin_id.clone(),
                            request_id,
                            payload: Some(payload),
                        },
                    )
                    .await?;
                }
                Some(proto::envelope::Payload::InvokeServiceRequest(request)) => {
                    let payload = {
                        let mut context = PluginContext {
                            stream: &mut stream,
                            plugin_id: &plugin_id,
                        };
                        match plugin.invoke_service(request, &mut context).await {
                            Ok(Some(response)) => {
                                proto::envelope::Payload::InvokeServiceResponse(response)
                            }
                            Ok(None) => proto::envelope::Payload::ErrorResponse(
                                PluginError::method_not_found("Unsupported service invocation")
                                    .into_error_response(),
                            ),
                            Err(err) => {
                                proto::envelope::Payload::ErrorResponse(err.into_error_response())
                            }
                        }
                    };
                    write_envelope(
                        &mut stream,
                        &proto::Envelope {
                            protocol_version: PROTOCOL_VERSION,
                            plugin_id: plugin_id.clone(),
                            request_id,
                            payload: Some(payload),
                        },
                    )
                    .await?;
                }
                Some(proto::envelope::Payload::RpcNotification(notification)) => {
                    let mut context = PluginContext {
                        stream: &mut stream,
                        plugin_id: &plugin_id,
                    };
                    plugin
                        .on_rpc_notification(notification, &mut context)
                        .await?;
                }
                Some(proto::envelope::Payload::ChannelMessage(message)) => {
                    let mut context = PluginContext {
                        stream: &mut stream,
                        plugin_id: &plugin_id,
                    };
                    plugin.on_channel_message(message, &mut context).await?;
                }
                Some(proto::envelope::Payload::BulkTransferMessage(message)) => {
                    let mut context = PluginContext {
                        stream: &mut stream,
                        plugin_id: &plugin_id,
                    };
                    plugin
                        .on_bulk_transfer_message(message, &mut context)
                        .await?;
                }
                Some(proto::envelope::Payload::MeshEvent(event)) => {
                    let mut context = PluginContext {
                        stream: &mut stream,
                        plugin_id: &plugin_id,
                    };
                    plugin.on_mesh_event(event, &mut context).await?;
                }
                Some(proto::envelope::Payload::OpenStreamRequest(request)) => {
                    let payload = {
                        let mut context = PluginContext {
                            stream: &mut stream,
                            plugin_id: &plugin_id,
                        };
                        match plugin.open_stream(request, &mut context).await {
                            Ok(Some(response)) => {
                                proto::envelope::Payload::OpenStreamResponse(response)
                            }
                            Ok(None) => proto::envelope::Payload::ErrorResponse(
                                PluginError::method_not_found(
                                    "Unsupported stream control message 'open_stream'",
                                )
                                .into_error_response(),
                            ),
                            Err(err) => {
                                proto::envelope::Payload::ErrorResponse(err.into_error_response())
                            }
                        }
                    };
                    write_envelope(
                        &mut stream,
                        &proto::Envelope {
                            protocol_version: PROTOCOL_VERSION,
                            plugin_id: plugin_id.clone(),
                            request_id,
                            payload: Some(payload),
                        },
                    )
                    .await?;
                }
                Some(proto::envelope::Payload::CancelStreamNotification(notification)) => {
                    let mut context = PluginContext {
                        stream: &mut stream,
                        plugin_id: &plugin_id,
                    };
                    plugin.on_cancel_stream(notification, &mut context).await?;
                }
                Some(proto::envelope::Payload::CloseStreamNotification(notification)) => {
                    let mut context = PluginContext {
                        stream: &mut stream,
                        plugin_id: &plugin_id,
                    };
                    plugin.on_close_stream(notification, &mut context).await?;
                }
                Some(proto::envelope::Payload::StreamError(error)) => {
                    let mut context = PluginContext {
                        stream: &mut stream,
                        plugin_id: &plugin_id,
                    };
                    plugin.on_stream_error(error, &mut context).await?;
                }
                Some(proto::envelope::Payload::ErrorResponse(error)) => {
                    let mut context = PluginContext {
                        stream: &mut stream,
                        plugin_id: &plugin_id,
                    };
                    plugin.on_host_error(error, &mut context).await?;
                }
                _ => {}
            }
        }

        Ok(())
    }
}

fn parse_service_input<T: DeserializeOwned>(input_json: &str) -> PluginResult<T> {
    let input = if input_json.trim().is_empty() {
        "null"
    } else {
        input_json
    };
    serde_json::from_str(input)
        .map_err(|err| PluginError::invalid_params(format!("Invalid service input JSON: {err}")))
}

fn serialize_service_output<T: Serialize>(value: &T) -> PluginResult<String> {
    serde_json::to_string(value)
        .map_err(|err| PluginError::internal(format!("Serialize service output: {err}")))
}

fn normalize_call_tool_output(result: &CallToolResult) -> PluginResult<String> {
    if let Some(value) = &result.structured_content {
        return serialize_service_output(value);
    }
    if let Some(text) = result.content.first().and_then(|content| content.as_text()) {
        return Ok(text.text.clone());
    }
    serialize_service_output(&result.content)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{mcp, plugin, plugin_server_info};
    use rmcp::model::{
        ArgumentInfo, PromptMessage, PromptMessageContent, PromptMessageRole, Reference,
    };
    use serde_json::json;

    #[derive(Clone, Debug, Default, Deserialize, Serialize, schemars::JsonSchema)]
    struct DemoArgs {
        #[serde(default)]
        message: String,
    }

    async fn disconnected_stream() -> LocalStream {
        #[cfg(unix)]
        {
            let (client, _server) = tokio::net::UnixStream::pair().unwrap();
            LocalStream::Unix(client)
        }
        #[cfg(windows)]
        {
            panic!("runtime tests are only implemented for unix");
        }
    }

    #[tokio::test]
    async fn invoke_service_dispatches_operation_prompt_resource_and_completion() {
        let mut plugin = plugin! {
            metadata: PluginMetadata::new(
                "demo",
                "1.0.0",
                plugin_server_info("demo", "1.0.0", "Demo", "Demo plugin", None::<String>),
            ),
            mcp: [
                mcp::tool("echo")
                    .description("Echo input")
                    .input::<DemoArgs>()
                    .handle(|args, _context| Box::pin(async move {
                        Ok(json!({ "echo": args.message }))
                    })),
                mcp::resource("demo://state")
                    .name("State")
                    .handle(|request, _context| Box::pin(async move {
                        Ok(crate::read_resource_result(vec![
                            rmcp::model::ResourceContents::text("state", request.uri),
                        ]))
                    })),
                mcp::prompt("brief")
                    .description("Brief")
                    .handle(|request, _context| Box::pin(async move {
                        Ok(crate::get_prompt_result(vec![PromptMessage::new(
                            PromptMessageRole::User,
                            PromptMessageContent::text(format!("brief:{}", request.name)),
                        )]))
                    })),
                mcp::completion("prompt.brief.topic")
                    .handle(|_request, _context| Box::pin(async move {
                        crate::complete_result(vec!["alpha".into()])
                    })),
            ],
        };

        let mut stream = disconnected_stream().await;
        let mut context = PluginContext {
            stream: &mut stream,
            plugin_id: "demo",
        };

        let op = plugin
            .invoke_service(
                proto::InvokeServiceRequest {
                    kind: proto::ServiceKind::Operation as i32,
                    service_name: "echo".into(),
                    input_json: json!({ "message": "hello" }).to_string(),
                },
                &mut context,
            )
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&op.output_json).unwrap(),
            json!({"echo": "hello"})
        );

        let prompt = plugin
            .invoke_service(
                proto::InvokeServiceRequest {
                    kind: proto::ServiceKind::Prompt as i32,
                    service_name: "brief".into(),
                    input_json: serde_json::to_string(&GetPromptRequestParams::new("brief"))
                        .unwrap(),
                },
                &mut context,
            )
            .await
            .unwrap()
            .unwrap();
        let prompt_result: GetPromptResult = serde_json::from_str(&prompt.output_json).unwrap();
        assert_eq!(prompt_result.messages.len(), 1);

        let resource = plugin
            .invoke_service(
                proto::InvokeServiceRequest {
                    kind: proto::ServiceKind::Resource as i32,
                    service_name: "demo://state".into(),
                    input_json: serde_json::to_string(&ReadResourceRequestParams::new(
                        "demo://state",
                    ))
                    .unwrap(),
                },
                &mut context,
            )
            .await
            .unwrap()
            .unwrap();
        let resource_result: ReadResourceResult =
            serde_json::from_str(&resource.output_json).unwrap();
        assert_eq!(resource_result.contents.len(), 1);

        let completion = plugin
            .invoke_service(
                proto::InvokeServiceRequest {
                    kind: proto::ServiceKind::Completion as i32,
                    service_name: "brief".into(),
                    input_json: serde_json::to_string(&CompleteRequestParams::new(
                        Reference::for_prompt("brief"),
                        ArgumentInfo {
                            name: "topic".into(),
                            value: "a".into(),
                        },
                    ))
                    .unwrap(),
                },
                &mut context,
            )
            .await
            .unwrap()
            .unwrap();
        let completion_result: CompleteResult =
            serde_json::from_str(&completion.output_json).unwrap();
        assert_eq!(
            completion_result.completion.values,
            vec![String::from("alpha")]
        );
    }
}
