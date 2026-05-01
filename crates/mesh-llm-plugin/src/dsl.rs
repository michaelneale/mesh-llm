use std::marker::PhantomData;
use std::sync::Arc;

use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde::Serialize;

use crate::helpers::{
    json_schema_operation, prompt as prompt_definition,
    resource_template as resource_template_definition, text_resource, CompletionFuture,
    CompletionRouter, JsonOperationFuture, OperationRouter, PromptFuture, PromptRouter,
    ResourceFuture, ResourceRouter,
};
use crate::manifest::{
    capability as capability_entry, completion as completion_entry, mcp_http_endpoint,
    mcp_stdio_endpoint, mcp_tcp_endpoint, mcp_unix_socket_endpoint, openai_http_inference_endpoint,
    operation as operation_entry, prompt_service as prompt_entry, resource as resource_entry,
    resource_template_service as resource_template_entry, EndpointBuilder, ManifestEntry,
    PluginManifestBuilder,
};
use crate::runtime::{PluginMetadata, SimplePlugin};
use crate::PluginContext;

fn ensure_description(current: Option<String>, fallback: &str) -> String {
    current.unwrap_or_else(|| fallback.to_string())
}

fn normalize_http_operation_name(method: &str, path: &str) -> String {
    let mut value = format!("__http_{}_{}", method.to_ascii_lowercase(), path);
    value = value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() {
                ch.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect::<String>();
    value.trim_matches('_').to_string()
}

fn template_prefix(uri_template: &str) -> String {
    uri_template
        .split('{')
        .next()
        .unwrap_or(uri_template)
        .to_string()
}

pub struct DeclarativePluginBuilder {
    metadata: PluginMetadata,
    manifest: PluginManifestBuilder,
    operation_router: Option<OperationRouter>,
    prompt_router: Option<PromptRouter>,
    resource_router: Option<ResourceRouter>,
    completion_router: Option<CompletionRouter>,
    plugin_customizers: Vec<Box<dyn FnOnce(SimplePlugin) -> SimplePlugin>>,
}

impl DeclarativePluginBuilder {
    pub fn new(metadata: PluginMetadata) -> Self {
        Self {
            metadata,
            manifest: PluginManifestBuilder::new(),
            operation_router: None,
            prompt_router: None,
            resource_router: None,
            completion_router: None,
            plugin_customizers: Vec::new(),
        }
    }

    pub fn provide<T: Into<ManifestEntry>>(mut self, item: T) -> Self {
        match item.into() {
            ManifestEntry::Capability(capability) => {
                self.manifest.push_item(capability_entry(capability));
            }
            _ => panic!("provides entries must be capabilities"),
        }
        self
    }

    pub fn mcp_item<T: Into<McpItem>>(mut self, item: T) -> Self {
        item.into().apply(&mut self);
        self
    }

    pub fn http_item<T: Into<HttpItem>>(mut self, item: T) -> Self {
        item.into().apply(&mut self);
        self
    }

    pub fn inference_item<T: Into<InferenceItem>>(mut self, item: T) -> Self {
        item.into().apply(&mut self);
        self
    }

    pub fn mesh_item<T: Into<ManifestEntry>>(mut self, item: T) -> Self {
        match item.into() {
            ManifestEntry::MeshChannel(channel) => self.manifest.push_item(channel),
            _ => panic!("mesh entries must be mesh channels"),
        }
        self
    }

    pub fn event_item<T: Into<ManifestEntry>>(mut self, item: T) -> Self {
        match item.into() {
            ManifestEntry::MeshEventSubscription(subscription) => {
                self.manifest.push_item(subscription);
            }
            _ => panic!("event entries must be mesh event subscriptions"),
        }
        self
    }

    pub fn startup_policy(mut self, startup_policy: crate::runtime::PluginStartupPolicy) -> Self {
        self.metadata = self.metadata.with_startup_policy(startup_policy);
        self
    }

    pub fn customize<F>(mut self, customizer: F) -> Self
    where
        F: FnOnce(SimplePlugin) -> SimplePlugin + 'static,
    {
        self.plugin_customizers.push(Box::new(customizer));
        self
    }

    fn ensure_operation_router(&mut self) -> &mut OperationRouter {
        self.operation_router
            .get_or_insert_with(OperationRouter::new)
    }

    fn ensure_prompt_router(&mut self) -> &mut PromptRouter {
        self.prompt_router.get_or_insert_with(PromptRouter::new)
    }

    fn ensure_resource_router(&mut self) -> &mut ResourceRouter {
        self.resource_router.get_or_insert_with(ResourceRouter::new)
    }

    fn ensure_completion_router(&mut self) -> &mut CompletionRouter {
        self.completion_router
            .get_or_insert_with(CompletionRouter::new)
    }

    pub fn build(self) -> SimplePlugin {
        let Self {
            metadata,
            manifest,
            operation_router,
            prompt_router,
            resource_router,
            completion_router,
            plugin_customizers,
        } = self;
        let manifest = manifest.build();
        let mut plugin = SimplePlugin::new(
            metadata
                .with_capabilities(manifest.capabilities.clone())
                .with_manifest(manifest),
        );
        if let Some(router) = operation_router {
            plugin = plugin.with_operation_router(router);
        }
        if let Some(router) = prompt_router {
            plugin = plugin.with_prompt_router(router);
        }
        if let Some(router) = resource_router {
            plugin = plugin.with_resource_router(router);
        }
        if let Some(router) = completion_router {
            plugin = plugin.with_completion_router(router);
        }
        for customizer in plugin_customizers {
            plugin = customizer(plugin);
        }
        plugin
    }
}

pub enum McpItem {
    Tool(LocalToolRegistration),
    Resource(LocalResourceRegistration),
    ResourceTemplate(LocalResourceTemplateRegistration),
    Prompt(LocalPromptRegistration),
    Completion(LocalCompletionRegistration),
    ExternalEndpoint(EndpointBuilder),
}

impl McpItem {
    fn apply(self, builder: &mut DeclarativePluginBuilder) {
        match self {
            Self::Tool(item) => {
                builder.manifest.push_item(item.manifest);
                (item.register)(builder.ensure_operation_router());
            }
            Self::Resource(item) => {
                builder.manifest.push_item(item.manifest);
                (item.register)(builder.ensure_resource_router());
            }
            Self::ResourceTemplate(item) => {
                builder.manifest.push_item(item.manifest);
                (item.register)(builder.ensure_resource_router());
            }
            Self::Prompt(item) => {
                builder.manifest.push_item(item.manifest);
                (item.register)(builder.ensure_prompt_router());
            }
            Self::Completion(item) => {
                builder.manifest.push_item(item.manifest);
                (item.register)(builder.ensure_completion_router());
            }
            Self::ExternalEndpoint(endpoint) => {
                builder.manifest.push_item(endpoint);
            }
        }
    }
}

pub struct McpExternalBuilder {
    endpoint: EndpointBuilder,
}

impl McpExternalBuilder {
    pub fn arg(mut self, arg: impl Into<String>) -> Self {
        self.endpoint = self.endpoint.arg(arg);
        self
    }

    pub fn args<I, S>(mut self, args: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.endpoint = self.endpoint.args(args);
        self
    }

    pub fn namespace(mut self, namespace: impl Into<String>) -> Self {
        self.endpoint = self.endpoint.namespace(namespace);
        self
    }

    pub fn supports_streaming(mut self, supports_streaming: bool) -> Self {
        self.endpoint = self.endpoint.supports_streaming(supports_streaming);
        self
    }
}

impl From<McpExternalBuilder> for McpItem {
    fn from(value: McpExternalBuilder) -> Self {
        Self::ExternalEndpoint(value.endpoint)
    }
}

pub enum HttpItem {
    Route(LocalHttpRouteRegistration),
}

impl HttpItem {
    fn apply(self, builder: &mut DeclarativePluginBuilder) {
        match self {
            Self::Route(item) => {
                builder.manifest.push_item(item.operation_manifest);
                builder.manifest.push_item(item.http_manifest);
                (item.register)(builder.ensure_operation_router());
            }
        }
    }
}

pub enum InferenceItem {
    Endpoint(EndpointBuilder),
}

pub struct InferenceEndpointBuilder {
    endpoint: EndpointBuilder,
}

impl InferenceEndpointBuilder {
    pub fn managed_by_plugin(mut self, managed_by_plugin: bool) -> Self {
        self.endpoint = self.endpoint.managed_by_plugin(managed_by_plugin);
        self
    }

    pub fn supports_streaming(mut self, supports_streaming: bool) -> Self {
        self.endpoint = self.endpoint.supports_streaming(supports_streaming);
        self
    }

    pub fn protocol(mut self, protocol: impl Into<String>) -> Self {
        self.endpoint = self.endpoint.protocol(protocol);
        self
    }
}

impl From<InferenceEndpointBuilder> for InferenceItem {
    fn from(value: InferenceEndpointBuilder) -> Self {
        Self::Endpoint(value.endpoint)
    }
}

impl InferenceItem {
    fn apply(self, builder: &mut DeclarativePluginBuilder) {
        match self {
            Self::Endpoint(endpoint) => {
                builder.manifest.push_item(endpoint);
            }
        }
    }
}

type OperationRouterRegistration = Box<dyn Fn(&mut OperationRouter) + Send + Sync>;
type PromptRouterRegistration = Box<dyn Fn(&mut PromptRouter) + Send + Sync>;
type ResourceRouterRegistration = Box<dyn Fn(&mut ResourceRouter) + Send + Sync>;
type CompletionRouterRegistration = Box<dyn Fn(&mut CompletionRouter) + Send + Sync>;

pub struct LocalToolRegistration {
    manifest: ManifestEntry,
    register: OperationRouterRegistration,
}

pub struct LocalResourceRegistration {
    manifest: ManifestEntry,
    register: ResourceRouterRegistration,
}

pub struct LocalResourceTemplateRegistration {
    manifest: ManifestEntry,
    register: ResourceRouterRegistration,
}

pub struct LocalPromptRegistration {
    manifest: ManifestEntry,
    register: PromptRouterRegistration,
}

pub struct LocalCompletionRegistration {
    manifest: ManifestEntry,
    register: CompletionRouterRegistration,
}

pub struct LocalHttpRouteRegistration {
    operation_manifest: ManifestEntry,
    http_manifest: ManifestEntry,
    register: OperationRouterRegistration,
}

pub mod mcp {
    use super::*;

    pub fn tool(name: impl Into<String>) -> McpToolBuilder<serde_json::Value> {
        McpToolBuilder {
            name: name.into(),
            description: None,
            title: None,
            output_schema_json: None,
            _input: PhantomData,
        }
    }

    pub fn resource(uri: impl Into<String>) -> McpResourceBuilder {
        McpResourceBuilder {
            uri: uri.into(),
            name: None,
            description: None,
            mime_type: None,
        }
    }

    pub fn resource_template(uri_template: impl Into<String>) -> McpResourceTemplateBuilder {
        McpResourceTemplateBuilder {
            uri_template: uri_template.into(),
            name: None,
            description: None,
            mime_type: None,
        }
    }

    pub fn prompt(name: impl Into<String>) -> McpPromptBuilder {
        McpPromptBuilder {
            name: name.into(),
            description: None,
        }
    }

    pub fn completion(argument_ref: impl Into<String>) -> McpCompletionBuilder {
        McpCompletionBuilder {
            argument_ref: argument_ref.into(),
            description: None,
        }
    }

    pub fn external_stdio(
        endpoint_id: impl Into<String>,
        command: impl Into<String>,
    ) -> McpExternalBuilder {
        McpExternalBuilder {
            endpoint: mcp_stdio_endpoint(endpoint_id, command),
        }
    }

    pub fn external_http(
        endpoint_id: impl Into<String>,
        address: impl Into<String>,
    ) -> McpExternalBuilder {
        McpExternalBuilder {
            endpoint: mcp_http_endpoint(endpoint_id, address),
        }
    }

    pub fn external_tcp(
        endpoint_id: impl Into<String>,
        address: impl Into<String>,
    ) -> McpExternalBuilder {
        McpExternalBuilder {
            endpoint: mcp_tcp_endpoint(endpoint_id, address),
        }
    }

    pub fn external_unix_socket(
        endpoint_id: impl Into<String>,
        address: impl Into<String>,
    ) -> McpExternalBuilder {
        McpExternalBuilder {
            endpoint: mcp_unix_socket_endpoint(endpoint_id, address),
        }
    }
}

pub mod http {
    use super::*;

    pub fn get(path: impl Into<String>) -> HttpRouteBuilder<serde_json::Value> {
        HttpRouteBuilder::new("GET", path)
    }

    pub fn post(path: impl Into<String>) -> HttpRouteBuilder<serde_json::Value> {
        HttpRouteBuilder::new("POST", path)
    }

    pub fn put(path: impl Into<String>) -> HttpRouteBuilder<serde_json::Value> {
        HttpRouteBuilder::new("PUT", path)
    }

    pub fn patch(path: impl Into<String>) -> HttpRouteBuilder<serde_json::Value> {
        HttpRouteBuilder::new("PATCH", path)
    }

    pub fn delete(path: impl Into<String>) -> HttpRouteBuilder<serde_json::Value> {
        HttpRouteBuilder::new("DELETE", path)
    }
}

pub mod inference {
    use super::*;

    pub fn openai_http(
        endpoint_id: impl Into<String>,
        address: impl Into<String>,
    ) -> InferenceEndpointBuilder {
        InferenceEndpointBuilder {
            endpoint: openai_http_inference_endpoint(endpoint_id, address),
        }
    }

    pub fn provider(
        endpoint_id: impl Into<String>,
        address: impl Into<String>,
    ) -> InferenceEndpointBuilder {
        InferenceEndpointBuilder {
            endpoint: openai_http_inference_endpoint(endpoint_id, address).managed_by_plugin(true),
        }
    }
}

pub struct McpToolBuilder<TArgs> {
    name: String,
    description: Option<String>,
    title: Option<String>,
    output_schema_json: Option<String>,
    _input: PhantomData<TArgs>,
}

impl<TArgs> McpToolBuilder<TArgs> {
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    pub fn title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    pub fn input<TNext: DeserializeOwned + JsonSchema + Send + 'static>(
        self,
    ) -> McpToolBuilder<TNext> {
        McpToolBuilder {
            name: self.name,
            description: self.description,
            title: self.title,
            output_schema_json: self.output_schema_json,
            _input: PhantomData,
        }
    }

    pub fn output<TOutput: JsonSchema>(mut self) -> Self {
        self.output_schema_json = Some(
            crate::json_string(&crate::json_schema_for::<TOutput>())
                .unwrap_or_else(|_| "{}".into()),
        );
        self
    }

    pub fn handle<TResult, F>(self, handler: F) -> McpItem
    where
        TArgs: DeserializeOwned + JsonSchema + Send + 'static,
        TResult: Serialize + Send + 'static,
        F: for<'a, 'ctx> Fn(TArgs, &'a mut PluginContext<'ctx>) -> JsonOperationFuture<'a, TResult>
            + Send
            + Sync
            + 'static,
    {
        let name = self.name.clone();
        let description = ensure_description(self.description, &name);
        let title = self.title.clone();
        let handler = Arc::new(handler);
        let manifest = operation_entry::<TArgs>(name.clone(), description.clone());
        let manifest = if let Some(title) = &title {
            manifest.title(title.clone())
        } else {
            manifest
        };
        let manifest = match self.output_schema_json {
            Some(output_schema_json) => {
                let mut inner =
                    if let ManifestEntry::Operation(inner) = ManifestEntry::from(manifest) {
                        inner
                    } else {
                        unreachable!()
                    };
                inner.output_schema_json = Some(output_schema_json);
                ManifestEntry::Operation(inner)
            }
            None => manifest.into(),
        };
        let register = Box::new(move |router: &mut OperationRouter| {
            let mut tool = json_schema_operation::<TArgs>(name.clone(), description.clone());
            if let Some(title) = &title {
                tool = tool.with_title(title.clone());
            }
            let handler = Arc::clone(&handler);
            router.add_json::<TArgs, TResult, _>(tool, move |args, context| {
                let handler = Arc::clone(&handler);
                handler(args, context)
            });
        });
        McpItem::Tool(LocalToolRegistration { manifest, register })
    }
}

pub struct McpResourceBuilder {
    uri: String,
    name: Option<String>,
    description: Option<String>,
    mime_type: Option<String>,
}

impl McpResourceBuilder {
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    pub fn mime_type(mut self, mime_type: impl Into<String>) -> Self {
        self.mime_type = Some(mime_type.into());
        self
    }

    pub fn handle<F>(self, handler: F) -> McpItem
    where
        F: for<'a, 'ctx> Fn(
                rmcp::model::ReadResourceRequestParams,
                &'a mut PluginContext<'ctx>,
            ) -> ResourceFuture<'a>
            + Send
            + Sync
            + 'static,
    {
        let uri = self.uri.clone();
        let name = self.name.unwrap_or_else(|| uri.clone());
        let description = self.description.clone();
        let mime_type = self.mime_type.clone();
        let handler = Arc::new(handler);
        let mut manifest = resource_entry(uri.clone(), name.clone());
        if let Some(description) = description.clone() {
            manifest = manifest.description(description);
        }
        if let Some(mime_type) = mime_type.clone() {
            manifest = manifest.mime_type(mime_type);
        }
        let register = Box::new(move |router: &mut ResourceRouter| {
            let mut resource = text_resource(uri.clone(), name.clone());
            if let Some(description) = description.clone() {
                resource.raw.description = Some(description);
            }
            if let Some(mime_type) = mime_type.clone() {
                resource.raw.mime_type = Some(mime_type);
            }
            let handler = Arc::clone(&handler);
            router.add_exact(resource, move |request, context| {
                let handler = Arc::clone(&handler);
                handler(request, context)
            });
        });
        McpItem::Resource(LocalResourceRegistration {
            manifest: manifest.into(),
            register,
        })
    }
}

pub struct McpResourceTemplateBuilder {
    uri_template: String,
    name: Option<String>,
    description: Option<String>,
    mime_type: Option<String>,
}

impl McpResourceTemplateBuilder {
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    pub fn mime_type(mut self, mime_type: impl Into<String>) -> Self {
        self.mime_type = Some(mime_type.into());
        self
    }

    pub fn handle<F>(self, handler: F) -> McpItem
    where
        F: for<'a, 'ctx> Fn(
                rmcp::model::ReadResourceRequestParams,
                &'a mut PluginContext<'ctx>,
            ) -> ResourceFuture<'a>
            + Send
            + Sync
            + 'static,
    {
        let uri_template = self.uri_template.clone();
        let name = self.name.unwrap_or_else(|| uri_template.clone());
        let description = self.description.clone();
        let mime_type = self.mime_type.clone();
        let prefix = template_prefix(&uri_template);
        let handler = Arc::new(handler);
        let mut manifest = resource_template_entry(uri_template.clone(), name.clone());
        if let Some(description) = description.clone() {
            manifest = manifest.description(description);
        }
        if let Some(mime_type) = mime_type.clone() {
            manifest = manifest.mime_type(mime_type);
        }
        let register = Box::new(move |router: &mut ResourceRouter| {
            let mut template = resource_template_definition(uri_template.clone(), name.clone());
            if let Some(description) = description.clone() {
                template.raw.description = Some(description);
            }
            if let Some(mime_type) = mime_type.clone() {
                template.raw.mime_type = Some(mime_type);
            }
            let handler = Arc::clone(&handler);
            router.add_prefix_template(template, prefix.clone(), move |request, context| {
                let handler = Arc::clone(&handler);
                handler(request, context)
            });
        });
        McpItem::ResourceTemplate(LocalResourceTemplateRegistration {
            manifest: manifest.into(),
            register,
        })
    }
}

pub struct McpPromptBuilder {
    name: String,
    description: Option<String>,
}

impl McpPromptBuilder {
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    pub fn handle<F>(self, handler: F) -> McpItem
    where
        F: for<'a, 'ctx> Fn(
                rmcp::model::GetPromptRequestParams,
                &'a mut PluginContext<'ctx>,
            ) -> PromptFuture<'a>
            + Send
            + Sync
            + 'static,
    {
        let name = self.name.clone();
        let description = self.description.clone();
        let handler = Arc::new(handler);
        let mut manifest = prompt_entry(name.clone());
        if let Some(description) = description.clone() {
            manifest = manifest.description(description);
        }
        let register = Box::new(move |router: &mut PromptRouter| {
            let prompt = prompt_definition(
                name.clone(),
                description.clone().unwrap_or_default(),
                None::<Vec<_>>,
            );
            let handler = Arc::clone(&handler);
            router.add(prompt, move |request, context| {
                let handler = Arc::clone(&handler);
                handler(request, context)
            });
        });
        McpItem::Prompt(LocalPromptRegistration {
            manifest: manifest.into(),
            register,
        })
    }
}

pub struct McpCompletionBuilder {
    argument_ref: String,
    description: Option<String>,
}

impl McpCompletionBuilder {
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    pub fn handle<F>(self, handler: F) -> McpItem
    where
        F: for<'a, 'ctx> Fn(
                rmcp::model::CompleteRequestParams,
                &'a mut PluginContext<'ctx>,
            ) -> CompletionFuture<'a>
            + Send
            + Sync
            + 'static,
    {
        let argument_ref = self.argument_ref.clone();
        let handler = Arc::new(handler);
        let mut manifest = completion_entry(argument_ref.clone());
        if let Some(description) = self.description.clone() {
            manifest = manifest.description(description);
        }
        let register = Box::new(move |router: &mut CompletionRouter| {
            let handler = Arc::clone(&handler);
            if let Some((prompt_name, argument_name)) = parse_prompt_argument_ref(&argument_ref) {
                router.add_prompt_argument(prompt_name, argument_name, move |request, context| {
                    let handler = Arc::clone(&handler);
                    handler(request, context)
                });
            }
        });
        McpItem::Completion(LocalCompletionRegistration {
            manifest: manifest.into(),
            register,
        })
    }
}

pub struct HttpRouteBuilder<TArgs> {
    method: &'static str,
    path: String,
    description: Option<String>,
    binding_id: Option<String>,
    request_schema_json: Option<String>,
    response_schema_json: Option<String>,
    request_body_mode: i32,
    response_body_mode: i32,
    _input: PhantomData<TArgs>,
}

impl<TArgs> HttpRouteBuilder<TArgs> {
    fn new(method: &'static str, path: impl Into<String>) -> Self {
        Self {
            method,
            path: path.into(),
            description: None,
            binding_id: None,
            request_schema_json: None,
            response_schema_json: None,
            request_body_mode: crate::proto::HttpBodyMode::Buffered as i32,
            response_body_mode: crate::proto::HttpBodyMode::Buffered as i32,
            _input: PhantomData,
        }
    }

    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    pub fn binding_id(mut self, binding_id: impl Into<String>) -> Self {
        self.binding_id = Some(binding_id.into());
        self
    }

    pub fn input<TNext: DeserializeOwned + JsonSchema + Send + 'static>(
        self,
    ) -> HttpRouteBuilder<TNext> {
        HttpRouteBuilder {
            method: self.method,
            path: self.path,
            description: self.description,
            binding_id: self.binding_id,
            request_schema_json: Some(
                crate::json_string(&crate::json_schema_for::<TNext>())
                    .unwrap_or_else(|_| "{}".into()),
            ),
            response_schema_json: self.response_schema_json,
            request_body_mode: self.request_body_mode,
            response_body_mode: self.response_body_mode,
            _input: PhantomData,
        }
    }

    pub fn output<TOutput: JsonSchema>(mut self) -> Self {
        self.response_schema_json = Some(
            crate::json_string(&crate::json_schema_for::<TOutput>())
                .unwrap_or_else(|_| "{}".into()),
        );
        self
    }

    pub fn stream_request(mut self) -> Self {
        self.request_body_mode = crate::proto::HttpBodyMode::Streamed as i32;
        self
    }

    pub fn stream_response(mut self) -> Self {
        self.response_body_mode = crate::proto::HttpBodyMode::Streamed as i32;
        self
    }

    pub fn sse(mut self) -> Self {
        self.response_body_mode = crate::proto::HttpBodyMode::Streamed as i32;
        self
    }

    pub fn handle<TResult, F>(self, handler: F) -> HttpItem
    where
        TArgs: DeserializeOwned + JsonSchema + Send + 'static,
        TResult: Serialize + Send + 'static,
        F: for<'a, 'ctx> Fn(TArgs, &'a mut PluginContext<'ctx>) -> JsonOperationFuture<'a, TResult>
            + Send
            + Sync
            + 'static,
    {
        let method = self.method;
        let path = if self.path.starts_with('/') {
            self.path.clone()
        } else {
            format!("/{}", self.path)
        };
        let operation_name = self
            .binding_id
            .clone()
            .unwrap_or_else(|| normalize_http_operation_name(method, &path));
        let description = ensure_description(self.description, &operation_name);
        let handler = Arc::new(handler);

        let mut op_inner = if let ManifestEntry::Operation(inner) =
            operation_entry::<TArgs>(operation_name.clone(), description.clone()).into()
        {
            inner
        } else {
            unreachable!()
        };
        op_inner.output_schema_json = self.response_schema_json.clone();

        let mut binding_inner = if let ManifestEntry::HttpBinding(inner) = match method {
            "GET" => crate::manifest::http_get(path.clone(), operation_name.clone()).into(),
            "POST" => crate::manifest::http_post(path.clone(), operation_name.clone()).into(),
            "PUT" => crate::manifest::http_put(path.clone(), operation_name.clone()).into(),
            "PATCH" => crate::manifest::http_patch(path.clone(), operation_name.clone()).into(),
            "DELETE" => crate::manifest::http_delete(path.clone(), operation_name.clone()).into(),
            _ => unreachable!(),
        } {
            inner
        } else {
            unreachable!()
        };
        binding_inner.binding_id = operation_name.clone();
        binding_inner.request_schema_json = self.request_schema_json.clone();
        binding_inner.response_schema_json = self.response_schema_json.clone();
        binding_inner.request_body_mode = self.request_body_mode;
        binding_inner.response_body_mode = self.response_body_mode;

        let register = Box::new(move |router: &mut OperationRouter| {
            let handler = Arc::clone(&handler);
            router.add_json::<TArgs, TResult, _>(
                json_schema_operation::<TArgs>(operation_name.clone(), description.clone()),
                move |args, context| {
                    let handler = Arc::clone(&handler);
                    handler(args, context)
                },
            );
        });

        HttpItem::Route(LocalHttpRouteRegistration {
            operation_manifest: ManifestEntry::Operation(op_inner),
            http_manifest: ManifestEntry::HttpBinding(binding_inner),
            register,
        })
    }
}

fn parse_prompt_argument_ref(argument_ref: &str) -> Option<(String, String)> {
    let remainder = argument_ref.strip_prefix("prompt.")?;
    let (prompt_name, argument_name) = remainder.rsplit_once('.')?;
    Some((prompt_name.to_string(), argument_name.to_string()))
}
