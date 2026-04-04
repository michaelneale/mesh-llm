//! Shared runtime and protocol helpers for mesh-llm plugins.
//!
//! For simple tools-only plugins, implement [`Plugin::list_tools`] and
//! [`Plugin::call_tool`] instead of overriding raw RPC dispatch. Prompt and
//! resource plugins can likewise use the typed prompt/resource hooks before
//! falling back to [`Plugin::handle_rpc`] for custom MCP methods.

mod context;
mod error;
mod helpers;
mod io;
mod runtime;

pub use async_trait::async_trait;
pub use context::PluginContext;
pub use error::{PluginError, PluginResult, PluginRpcResult, STARTUP_DISABLED_ERROR_CODE};
pub use helpers::{
    accept_bulk_transfer_message, bulk_transfer_message, bulk_transfer_sequence,
    cancel_task_result, channel_message, complete_result, empty_object_schema, get_prompt_result,
    get_task_payload_result, get_task_result, json_bytes, json_channel_message,
    json_reply_channel_message, json_response, json_schema_for, json_schema_tool, json_string,
    list_prompts, list_resource_templates, list_resources, list_tasks, list_tools,
    parse_get_prompt_request, parse_optional_json, parse_read_resource_request, parse_rpc_params,
    parse_tool_call_request, plugin_server_info, plugin_server_info_full, prompt, prompt_argument,
    read_resource_result, resource_template, structured_tool_result, task, text_resource,
    tool_error, tool_with_schema, BulkTransferSequence, CompletionFuture, CompletionRouter,
    JsonToolFuture, PromptFuture, PromptRouter, ResourceFuture, ResourceRouter, SubscriptionSet,
    TaskCancelFuture, TaskInfoFuture, TaskListFuture, TaskRecord, TaskResultFuture, TaskRouter,
    TaskStore, ToolCallRequest, ToolFuture, ToolRouter,
};
pub use io::{
    connect_from_env, read_envelope, send_bulk_transfer_message, send_channel_message,
    write_envelope, LocalStream,
};
pub use runtime::{
    EnsureInferenceEndpointRequest, EnsureInferenceEndpointResponse, InferenceEndpointDescriptor,
    InferenceLocalModelMatcher, InferenceProviderCapabilitiesDescriptor, MeshVisibility, Plugin,
    PluginInitializeRequest, PluginMetadata, PluginRuntime, PluginStartupPolicy, SimplePlugin,
};

#[allow(dead_code)]
pub mod proto {
    include!(concat!(env!("OUT_DIR"), "/meshllm.plugin.v1.rs"));
}

pub const PROTOCOL_VERSION: u32 = 1;
