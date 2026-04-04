use crate::{helpers::json_string, json_schema_for, proto};
use schemars::JsonSchema;

#[derive(Clone, Debug)]
pub enum ManifestEntry {
    Capability(String),
    Operation(proto::OperationManifest),
    Resource(proto::ResourceManifest),
    ResourceTemplate(proto::ResourceTemplateManifest),
    Prompt(proto::PromptManifest),
    Completion(proto::CompletionManifest),
    HttpBinding(proto::HttpBindingManifest),
    Endpoint(proto::EndpointManifest),
    MeshChannel(proto::MeshChannelManifest),
    MeshEventSubscription(proto::MeshEventSubscriptionManifest),
}

#[derive(Clone, Debug, Default)]
pub struct PluginManifestBuilder {
    manifest: proto::PluginManifest,
}

impl PluginManifestBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn item<T: Into<ManifestEntry>>(mut self, item: T) -> Self {
        self.push(item.into());
        self
    }

    pub fn build(self) -> proto::PluginManifest {
        self.manifest
    }

    pub fn push_item<T: Into<ManifestEntry>>(&mut self, item: T) {
        self.push(item.into());
    }

    fn push(&mut self, item: ManifestEntry) {
        match item {
            ManifestEntry::Capability(capability) => self.manifest.capabilities.push(capability),
            ManifestEntry::Operation(operation) => self.manifest.operations.push(operation),
            ManifestEntry::Resource(resource) => self.manifest.resources.push(resource),
            ManifestEntry::ResourceTemplate(template) => {
                self.manifest.resource_templates.push(template);
            }
            ManifestEntry::Prompt(prompt) => self.manifest.prompts.push(prompt),
            ManifestEntry::Completion(completion) => {
                self.manifest.completions.push(completion);
            }
            ManifestEntry::HttpBinding(binding) => self.manifest.http_bindings.push(binding),
            ManifestEntry::Endpoint(endpoint) => self.manifest.endpoints.push(endpoint),
            ManifestEntry::MeshChannel(channel) => self.manifest.mesh_channels.push(channel),
            ManifestEntry::MeshEventSubscription(subscription) => {
                self.manifest.mesh_event_subscriptions.push(subscription);
            }
        }
    }
}

pub fn plugin_manifest() -> PluginManifestBuilder {
    PluginManifestBuilder::new()
}

pub fn capability(name: impl Into<String>) -> ManifestEntry {
    ManifestEntry::Capability(name.into())
}

pub fn mesh_channel(name: impl Into<String>) -> ManifestEntry {
    ManifestEntry::MeshChannel(proto::MeshChannelManifest { name: name.into() })
}

pub fn mesh_event_subscription(kind: proto::mesh_event::Kind) -> ManifestEntry {
    ManifestEntry::MeshEventSubscription(proto::MeshEventSubscriptionManifest { kind: kind as i32 })
}

pub fn mesh_event_peer_up() -> ManifestEntry {
    mesh_event_subscription(proto::mesh_event::Kind::PeerUp)
}

pub fn mesh_event_peer_down() -> ManifestEntry {
    mesh_event_subscription(proto::mesh_event::Kind::PeerDown)
}

pub fn mesh_event_peer_updated() -> ManifestEntry {
    mesh_event_subscription(proto::mesh_event::Kind::PeerUpdated)
}

pub fn mesh_event_local_accepting() -> ManifestEntry {
    mesh_event_subscription(proto::mesh_event::Kind::LocalAccepting)
}

pub fn mesh_event_local_standby() -> ManifestEntry {
    mesh_event_subscription(proto::mesh_event::Kind::LocalStandby)
}

pub fn mesh_event_mesh_id_updated() -> ManifestEntry {
    mesh_event_subscription(proto::mesh_event::Kind::MeshIdUpdated)
}

impl From<proto::MeshChannelManifest> for ManifestEntry {
    fn from(value: proto::MeshChannelManifest) -> Self {
        Self::MeshChannel(value)
    }
}

impl From<proto::MeshEventSubscriptionManifest> for ManifestEntry {
    fn from(value: proto::MeshEventSubscriptionManifest) -> Self {
        Self::MeshEventSubscription(value)
    }
}

#[derive(Clone, Debug)]
pub struct OperationBuilder {
    inner: proto::OperationManifest,
}

pub fn operation<Input: JsonSchema>(
    name: impl Into<String>,
    description: impl Into<String>,
) -> OperationBuilder {
    OperationBuilder {
        inner: proto::OperationManifest {
            name: name.into(),
            description: description.into(),
            input_schema_json: schema_json::<Input>(),
            output_schema_json: None,
            title: None,
        },
    }
}

impl OperationBuilder {
    pub fn title(mut self, title: impl Into<String>) -> Self {
        self.inner.title = Some(title.into());
        self
    }

    pub fn output_schema<Output: JsonSchema>(mut self) -> Self {
        self.inner.output_schema_json = Some(schema_json::<Output>());
        self
    }
}

impl From<OperationBuilder> for ManifestEntry {
    fn from(value: OperationBuilder) -> Self {
        Self::Operation(value.inner)
    }
}

#[derive(Clone, Debug)]
pub struct ResourceBuilder {
    inner: proto::ResourceManifest,
}

pub fn resource(uri: impl Into<String>, name: impl Into<String>) -> ResourceBuilder {
    ResourceBuilder {
        inner: proto::ResourceManifest {
            uri: uri.into(),
            name: name.into(),
            description: None,
            mime_type: None,
        },
    }
}

impl ResourceBuilder {
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.inner.description = Some(description.into());
        self
    }

    pub fn mime_type(mut self, mime_type: impl Into<String>) -> Self {
        self.inner.mime_type = Some(mime_type.into());
        self
    }
}

impl From<ResourceBuilder> for ManifestEntry {
    fn from(value: ResourceBuilder) -> Self {
        Self::Resource(value.inner)
    }
}

#[derive(Clone, Debug)]
pub struct ResourceTemplateBuilder {
    inner: proto::ResourceTemplateManifest,
}

pub fn resource_template_service(
    uri_template: impl Into<String>,
    name: impl Into<String>,
) -> ResourceTemplateBuilder {
    ResourceTemplateBuilder {
        inner: proto::ResourceTemplateManifest {
            uri_template: uri_template.into(),
            name: name.into(),
            description: None,
            mime_type: None,
        },
    }
}

impl ResourceTemplateBuilder {
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.inner.description = Some(description.into());
        self
    }

    pub fn mime_type(mut self, mime_type: impl Into<String>) -> Self {
        self.inner.mime_type = Some(mime_type.into());
        self
    }
}

impl From<ResourceTemplateBuilder> for ManifestEntry {
    fn from(value: ResourceTemplateBuilder) -> Self {
        Self::ResourceTemplate(value.inner)
    }
}

#[derive(Clone, Debug)]
pub struct PromptBuilder {
    inner: proto::PromptManifest,
}

pub fn prompt_service(name: impl Into<String>) -> PromptBuilder {
    PromptBuilder {
        inner: proto::PromptManifest {
            name: name.into(),
            description: None,
        },
    }
}

impl PromptBuilder {
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.inner.description = Some(description.into());
        self
    }
}

impl From<PromptBuilder> for ManifestEntry {
    fn from(value: PromptBuilder) -> Self {
        Self::Prompt(value.inner)
    }
}

#[derive(Clone, Debug)]
pub struct CompletionBuilder {
    inner: proto::CompletionManifest,
}

pub fn completion(argument_ref: impl Into<String>) -> CompletionBuilder {
    CompletionBuilder {
        inner: proto::CompletionManifest {
            argument_ref: argument_ref.into(),
            description: None,
        },
    }
}

impl CompletionBuilder {
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.inner.description = Some(description.into());
        self
    }
}

impl From<CompletionBuilder> for ManifestEntry {
    fn from(value: CompletionBuilder) -> Self {
        Self::Completion(value.inner)
    }
}

#[derive(Clone, Debug)]
pub struct HttpBindingBuilder {
    inner: proto::HttpBindingManifest,
}

pub fn http_binding(
    method: proto::HttpMethod,
    path: impl Into<String>,
    operation_name: impl Into<String>,
) -> HttpBindingBuilder {
    let path = normalize_path(path.into());
    let operation_name = operation_name.into();
    HttpBindingBuilder {
        inner: proto::HttpBindingManifest {
            binding_id: default_binding_id(&path, &operation_name),
            method: method as i32,
            path,
            operation_name: Some(operation_name),
            request_body_mode: proto::HttpBodyMode::Buffered as i32,
            response_body_mode: proto::HttpBodyMode::Buffered as i32,
            request_schema_json: None,
            response_schema_json: None,
        },
    }
}

pub fn http_get(path: impl Into<String>, operation_name: impl Into<String>) -> HttpBindingBuilder {
    http_binding(proto::HttpMethod::Get, path, operation_name)
}

pub fn http_post(path: impl Into<String>, operation_name: impl Into<String>) -> HttpBindingBuilder {
    http_binding(proto::HttpMethod::Post, path, operation_name)
}

pub fn http_put(path: impl Into<String>, operation_name: impl Into<String>) -> HttpBindingBuilder {
    http_binding(proto::HttpMethod::Put, path, operation_name)
}

pub fn http_patch(
    path: impl Into<String>,
    operation_name: impl Into<String>,
) -> HttpBindingBuilder {
    http_binding(proto::HttpMethod::Patch, path, operation_name)
}

pub fn http_delete(
    path: impl Into<String>,
    operation_name: impl Into<String>,
) -> HttpBindingBuilder {
    http_binding(proto::HttpMethod::Delete, path, operation_name)
}

impl HttpBindingBuilder {
    pub fn binding_id(mut self, binding_id: impl Into<String>) -> Self {
        self.inner.binding_id = binding_id.into();
        self
    }

    pub fn request_schema<Request: JsonSchema>(mut self) -> Self {
        self.inner.request_schema_json = Some(schema_json::<Request>());
        self
    }

    pub fn response_schema<Response: JsonSchema>(mut self) -> Self {
        self.inner.response_schema_json = Some(schema_json::<Response>());
        self
    }

    pub fn streamed_request(mut self) -> Self {
        self.inner.request_body_mode = proto::HttpBodyMode::Streamed as i32;
        self
    }

    pub fn streamed_response(mut self) -> Self {
        self.inner.response_body_mode = proto::HttpBodyMode::Streamed as i32;
        self
    }

    pub fn buffered_request(mut self) -> Self {
        self.inner.request_body_mode = proto::HttpBodyMode::Buffered as i32;
        self
    }

    pub fn buffered_response(mut self) -> Self {
        self.inner.response_body_mode = proto::HttpBodyMode::Buffered as i32;
        self
    }
}

impl From<HttpBindingBuilder> for ManifestEntry {
    fn from(value: HttpBindingBuilder) -> Self {
        Self::HttpBinding(value.inner)
    }
}

#[derive(Clone, Debug)]
pub struct EndpointBuilder {
    inner: proto::EndpointManifest,
}

pub fn openai_http_inference_endpoint(
    endpoint_id: impl Into<String>,
    address: impl Into<String>,
) -> EndpointBuilder {
    EndpointBuilder {
        inner: proto::EndpointManifest {
            endpoint_id: endpoint_id.into(),
            kind: proto::EndpointKind::Inference as i32,
            transport_kind: proto::EndpointTransportKind::EndpointTransportHttp as i32,
            protocol: Some("openai_compatible".into()),
            address: Some(address.into()),
            args: Vec::new(),
            namespace: None,
            supports_streaming: true,
            managed_by_plugin: false,
        },
    }
}

pub fn mcp_stdio_endpoint(
    endpoint_id: impl Into<String>,
    command: impl Into<String>,
) -> EndpointBuilder {
    EndpointBuilder {
        inner: proto::EndpointManifest {
            endpoint_id: endpoint_id.into(),
            kind: proto::EndpointKind::Mcp as i32,
            transport_kind: proto::EndpointTransportKind::EndpointTransportStdio as i32,
            protocol: None,
            address: Some(command.into()),
            args: Vec::new(),
            namespace: None,
            supports_streaming: false,
            managed_by_plugin: false,
        },
    }
}

pub fn mcp_http_endpoint(
    endpoint_id: impl Into<String>,
    address: impl Into<String>,
) -> EndpointBuilder {
    EndpointBuilder {
        inner: proto::EndpointManifest {
            endpoint_id: endpoint_id.into(),
            kind: proto::EndpointKind::Mcp as i32,
            transport_kind: proto::EndpointTransportKind::EndpointTransportHttp as i32,
            protocol: Some("streamable_http".into()),
            address: Some(address.into()),
            args: Vec::new(),
            namespace: None,
            supports_streaming: true,
            managed_by_plugin: false,
        },
    }
}

pub fn mcp_tcp_endpoint(
    endpoint_id: impl Into<String>,
    address: impl Into<String>,
) -> EndpointBuilder {
    EndpointBuilder {
        inner: proto::EndpointManifest {
            endpoint_id: endpoint_id.into(),
            kind: proto::EndpointKind::Mcp as i32,
            transport_kind: proto::EndpointTransportKind::EndpointTransportTcp as i32,
            protocol: None,
            address: Some(address.into()),
            args: Vec::new(),
            namespace: None,
            supports_streaming: false,
            managed_by_plugin: false,
        },
    }
}

pub fn mcp_unix_socket_endpoint(
    endpoint_id: impl Into<String>,
    address: impl Into<String>,
) -> EndpointBuilder {
    EndpointBuilder {
        inner: proto::EndpointManifest {
            endpoint_id: endpoint_id.into(),
            kind: proto::EndpointKind::Mcp as i32,
            transport_kind: proto::EndpointTransportKind::EndpointTransportUnixSocket as i32,
            protocol: None,
            address: Some(address.into()),
            args: Vec::new(),
            namespace: None,
            supports_streaming: false,
            managed_by_plugin: false,
        },
    }
}

impl EndpointBuilder {
    pub fn protocol(mut self, protocol: impl Into<String>) -> Self {
        self.inner.protocol = Some(protocol.into());
        self
    }

    pub fn namespace(mut self, namespace: impl Into<String>) -> Self {
        self.inner.namespace = Some(namespace.into());
        self
    }

    pub fn arg(mut self, arg: impl Into<String>) -> Self {
        self.inner.args.push(arg.into());
        self
    }

    pub fn args<I, S>(mut self, args: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.inner.args.extend(args.into_iter().map(Into::into));
        self
    }

    pub fn supports_streaming(mut self, supports_streaming: bool) -> Self {
        self.inner.supports_streaming = supports_streaming;
        self
    }

    pub fn managed_by_plugin(mut self, managed_by_plugin: bool) -> Self {
        self.inner.managed_by_plugin = managed_by_plugin;
        self
    }
}

impl From<EndpointBuilder> for ManifestEntry {
    fn from(value: EndpointBuilder) -> Self {
        Self::Endpoint(value.inner)
    }
}

fn schema_json<T: JsonSchema>() -> String {
    json_string(&json_schema_for::<T>()).unwrap_or_else(|_| "{}".into())
}

fn normalize_path(path: String) -> String {
    if path.is_empty() {
        "/".into()
    } else if path.starts_with('/') {
        path
    } else {
        format!("/{path}")
    }
}

fn default_binding_id(path: &str, operation_name: &str) -> String {
    let candidate = if !operation_name.trim().is_empty() {
        operation_name
    } else {
        path.trim_matches('/')
    };
    let sanitized = candidate
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() {
                ch.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect::<String>();
    let sanitized = sanitized.trim_matches('_');
    if sanitized.is_empty() {
        "root".into()
    } else {
        sanitized.into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{inference, mcp, plugin_server_info, Plugin, PluginMetadata};

    #[allow(dead_code)]
    #[derive(serde::Deserialize, schemars::JsonSchema)]
    struct DemoInput {
        value: String,
    }

    #[allow(dead_code)]
    #[derive(serde::Serialize, schemars::JsonSchema)]
    struct DemoOutput {
        echoed: String,
    }

    #[test]
    fn macro_builds_manifest_entries() {
        let manifest = crate::plugin_manifest![
            capability("demo.v1"),
            mesh_channel("demo.v1"),
            mesh_event_peer_up(),
            operation::<DemoInput>("echo", "Echo input").title("Echo"),
            http_post("/echo", "echo")
                .request_schema::<DemoInput>()
                .response_schema::<DemoOutput>(),
            mcp_stdio_endpoint("notes", "demo-mcp").arg("--serve"),
        ];

        assert_eq!(manifest.capabilities, vec!["demo.v1"]);
        assert_eq!(manifest.operations.len(), 1);
        assert_eq!(manifest.http_bindings.len(), 1);
        assert_eq!(manifest.endpoints.len(), 1);
        assert_eq!(manifest.mesh_channels.len(), 1);
        assert_eq!(manifest.mesh_event_subscriptions.len(), 1);
        assert_eq!(manifest.http_bindings[0].binding_id, "echo");
        assert_eq!(manifest.endpoints[0].args, vec!["--serve"]);
    }

    #[test]
    fn streaming_http_builder_sets_modes() {
        let entry: ManifestEntry = http_post("/upload", "upload")
            .streamed_request()
            .streamed_response()
            .into();
        let ManifestEntry::HttpBinding(binding) = entry else {
            panic!("expected http binding");
        };
        assert_eq!(
            binding.request_body_mode,
            proto::HttpBodyMode::Streamed as i32
        );
        assert_eq!(
            binding.response_body_mode,
            proto::HttpBodyMode::Streamed as i32
        );
    }

    #[test]
    fn plugin_macro_builds_simple_plugin_with_manifest() {
        let plugin = crate::plugin! {
            metadata: PluginMetadata::new(
                "demo",
                "1.0.0",
                plugin_server_info("demo", "1.0.0", "Demo", "Demo plugin", None::<String>),
            ),
            provides: [capability("demo.v1")],
            mesh: [mesh_channel("demo.v1")],
            events: [mesh_event_peer_up()],
            mcp: [
                mcp::tool("echo")
                    .description("Echo input")
                    .input::<DemoInput>()
                    .handle(|args, _context| Box::pin(async move {
                        Ok(DemoOutput { echoed: args.value })
                    })),
                mcp::external_stdio("stdio", "demo-mcp"),
            ],
            http: [
                crate::http::post("/echo")
                    .description("Echo input")
                    .input::<DemoInput>()
                    .output::<DemoOutput>()
                    .handle(|args, _context| Box::pin(async move {
                        Ok(DemoOutput { echoed: args.value })
                    })),
            ],
            inference: [
                inference::openai_http("local", "http://127.0.0.1:8080/v1"),
            ],
        };

        let manifest = plugin.manifest().expect("manifest");
        assert_eq!(plugin.capabilities(), vec!["demo.v1"]);
        assert_eq!(manifest.capabilities, vec!["demo.v1"]);
        assert_eq!(manifest.operations.len(), 2);
        assert_eq!(manifest.http_bindings.len(), 1);
        assert_eq!(manifest.endpoints.len(), 2);
        assert_eq!(manifest.mesh_channels.len(), 1);
        assert_eq!(manifest.mesh_event_subscriptions.len(), 1);
    }

    #[test]
    fn declarative_macro_builds_local_mcp_entries() {
        let plugin = crate::plugin! {
            metadata: PluginMetadata::new(
                "demo",
                "1.0.0",
                plugin_server_info("demo", "1.0.0", "Demo", "Demo plugin", None::<String>),
            ),
            provides: [capability("demo.v1")],
            mcp: [
                mcp::tool("echo")
                    .description("Echo input")
                    .input::<DemoInput>()
                    .handle(|args, _context| Box::pin(async move {
                        Ok(DemoOutput { echoed: args.value })
                    })),
                mcp::resource("demo://snapshot")
                    .name("Snapshot")
                    .handle(|request, _context| Box::pin(async move {
                        Ok(crate::read_resource_result(vec![
                            rmcp::model::ResourceContents::text("snapshot", request.uri),
                        ]))
                    })),
                mcp::prompt("brief")
                    .description("Brief prompt")
                    .handle(|request, _context| Box::pin(async move {
                        Ok(crate::get_prompt_result(vec![
                            rmcp::model::PromptMessage::new(
                                rmcp::model::PromptMessageRole::User,
                                rmcp::model::PromptMessageContent::text(request.name),
                            ),
                        ]))
                    })),
                mcp::completion("prompt.brief.topic")
                    .description("Topic completion")
                    .handle(|_request, _context| Box::pin(async move {
                        crate::complete_result(vec!["alpha".into()])
                    })),
            ],
        };

        let manifest = plugin.manifest().expect("manifest");
        assert_eq!(manifest.operations.len(), 1);
        assert_eq!(manifest.resources.len(), 1);
        assert_eq!(manifest.prompts.len(), 1);
        assert_eq!(manifest.completions.len(), 1);
    }
}
