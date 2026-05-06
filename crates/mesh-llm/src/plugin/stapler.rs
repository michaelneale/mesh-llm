use super::proto;
use rmcp::model::{
    AnnotateAble, Prompt, RawResource, RawResourceTemplate, Resource, ResourceTemplate, Tool,
};
use std::sync::Arc;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct HttpBindingRoute {
    pub method: &'static str,
    pub route_path: String,
    pub operation_name: Option<String>,
}

pub(crate) fn operation(exposed_name: String, manifest: &proto::OperationManifest) -> Tool {
    let mut operation = Tool::new(
        exposed_name,
        manifest.description.clone(),
        Arc::new(parse_input_schema(&manifest.input_schema_json)),
    );
    if let Some(title) = &manifest.title {
        operation = operation.with_title(title.clone());
    }
    if let Some(output_schema_json) = &manifest.output_schema_json {
        if let Ok(schema) = serde_json::from_str::<serde_json::Value>(output_schema_json) {
            if let Some(schema) = schema.as_object() {
                operation.output_schema = Some(Arc::new(schema.clone()));
            }
        }
    }
    operation
}

pub(crate) fn prompt(exposed_name: String, manifest: &proto::PromptManifest) -> Prompt {
    Prompt::new(exposed_name, manifest.description.clone(), None::<Vec<_>>)
}

pub(crate) fn resource(manifest: &proto::ResourceManifest) -> Resource {
    let mut resource = RawResource::new(&manifest.uri, &manifest.name);
    if let Some(description) = &manifest.description {
        resource = resource.with_description(description.clone());
    }
    if let Some(mime_type) = &manifest.mime_type {
        resource = resource.with_mime_type(mime_type.clone());
    }
    resource.no_annotation()
}

pub(crate) fn resource_template(manifest: &proto::ResourceTemplateManifest) -> ResourceTemplate {
    let mut resource = RawResourceTemplate::new(&manifest.uri_template, &manifest.name);
    if let Some(description) = &manifest.description {
        resource = resource.with_description(description.clone());
    }
    if let Some(mime_type) = &manifest.mime_type {
        resource = resource.with_mime_type(mime_type.clone());
    }
    resource.no_annotation()
}

pub(crate) fn http_binding_route(
    plugin_name: &str,
    manifest: &proto::HttpBindingManifest,
) -> Option<HttpBindingRoute> {
    let method = match proto::HttpMethod::try_from(manifest.method).ok()? {
        proto::HttpMethod::Get => "GET",
        proto::HttpMethod::Post => "POST",
        proto::HttpMethod::Put => "PUT",
        proto::HttpMethod::Patch => "PATCH",
        proto::HttpMethod::Delete => "DELETE",
        proto::HttpMethod::Unspecified => return None,
    };
    Some(HttpBindingRoute {
        method,
        route_path: format!(
            "/api/plugins/{plugin_name}/http{}",
            normalize_route_path(&manifest.path)
        ),
        operation_name: manifest.operation_name.clone(),
    })
}

pub(crate) fn normalize_route_path(path: &str) -> String {
    if path.is_empty() {
        "/".into()
    } else if path.starts_with('/') {
        path.to_string()
    } else {
        format!("/{path}")
    }
}

fn parse_input_schema(input_schema_json: &str) -> serde_json::Map<String, serde_json::Value> {
    serde_json::from_str::<serde_json::Value>(input_schema_json)
        .ok()
        .and_then(|value| value.as_object().cloned())
        .unwrap_or_else(|| {
            serde_json::json!({
                "type": "object",
                "additionalProperties": true
            })
            .as_object()
            .cloned()
            .unwrap()
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_schema_falls_back_for_invalid_json() {
        let manifest = proto::OperationManifest {
            name: "echo".into(),
            description: "Echo input".into(),
            input_schema_json: "not-json".into(),
            output_schema_json: None,
            title: None,
        };
        let operation = operation("demo.echo".into(), &manifest);
        assert_eq!(
            operation
                .input_schema
                .get("type")
                .and_then(|value| value.as_str()),
            Some("object")
        );
        assert_eq!(
            operation
                .input_schema
                .get("additionalProperties")
                .and_then(|value| value.as_bool()),
            Some(true)
        );
    }

    #[test]
    fn resource_preserves_description_and_mime_type() {
        let manifest = proto::ResourceManifest {
            uri: "demo://snapshot".into(),
            name: "Snapshot".into(),
            description: Some("Current state".into()),
            mime_type: Some("application/json".into()),
        };
        let resource = resource(&manifest);
        assert_eq!(resource.raw.description.as_deref(), Some("Current state"));
        assert_eq!(resource.raw.mime_type.as_deref(), Some("application/json"));
    }

    #[test]
    fn http_binding_route_uses_plugin_namespace() {
        let manifest = proto::HttpBindingManifest {
            binding_id: "feed".into(),
            method: proto::HttpMethod::Get as i32,
            path: "/feed".into(),
            operation_name: Some("feed".into()),
            request_body_mode: proto::HttpBodyMode::Buffered as i32,
            response_body_mode: proto::HttpBodyMode::Buffered as i32,
            request_schema_json: None,
            response_schema_json: None,
        };
        let route = http_binding_route("blackboard", &manifest).unwrap();
        assert_eq!(route.method, "GET");
        assert_eq!(route.route_path, "/api/plugins/blackboard/http/feed");
        assert_eq!(route.operation_name.as_deref(), Some("feed"));
    }
}
