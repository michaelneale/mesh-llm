use super::proto;
use rmcp::model::{
    AnnotateAble, Prompt, RawResource, RawResourceTemplate, Resource, ResourceTemplate, Tool,
};
use std::sync::Arc;

pub(crate) fn mcp_tool(exposed_name: String, manifest: &proto::McpToolManifest) -> Tool {
    let mut tool = Tool::new(
        exposed_name,
        manifest.description.clone(),
        Arc::new(parse_input_schema(&manifest.input_schema_json)),
    );
    if let Some(title) = &manifest.title {
        tool = tool.with_title(title.clone());
    }
    if let Some(output_schema_json) = &manifest.output_schema_json {
        if let Ok(schema) = serde_json::from_str::<serde_json::Value>(output_schema_json) {
            if let Some(schema) = schema.as_object() {
                tool.output_schema = Some(Arc::new(schema.clone()));
            }
        }
    }
    tool
}

pub(crate) fn mcp_prompt(exposed_name: String, manifest: &proto::McpPromptManifest) -> Prompt {
    Prompt::new(exposed_name, manifest.description.clone(), None::<Vec<_>>)
}

pub(crate) fn mcp_resource(manifest: &proto::McpResourceManifest) -> Resource {
    let mut resource = RawResource::new(&manifest.uri, &manifest.name);
    if let Some(description) = &manifest.description {
        resource = resource.with_description(description.clone());
    }
    if let Some(mime_type) = &manifest.mime_type {
        resource = resource.with_mime_type(mime_type.clone());
    }
    resource.no_annotation()
}

pub(crate) fn mcp_resource_template(
    manifest: &proto::McpResourceTemplateManifest,
) -> ResourceTemplate {
    let mut resource = RawResourceTemplate::new(&manifest.uri_template, &manifest.name);
    if let Some(description) = &manifest.description {
        resource = resource.with_description(description.clone());
    }
    if let Some(mime_type) = &manifest.mime_type {
        resource = resource.with_mime_type(mime_type.clone());
    }
    resource.no_annotation()
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
        let manifest = proto::McpToolManifest {
            name: "echo".into(),
            description: "Echo input".into(),
            input_schema_json: "not-json".into(),
            output_schema_json: None,
            title: None,
        };
        let tool = mcp_tool("demo.echo".into(), &manifest);
        assert_eq!(
            tool.input_schema
                .get("type")
                .and_then(|value| value.as_str()),
            Some("object")
        );
        assert_eq!(
            tool.input_schema
                .get("additionalProperties")
                .and_then(|value| value.as_bool()),
            Some(true)
        );
    }

    #[test]
    fn resource_preserves_description_and_mime_type() {
        let manifest = proto::McpResourceManifest {
            uri: "demo://snapshot".into(),
            name: "Snapshot".into(),
            description: Some("Current state".into()),
            mime_type: Some("application/json".into()),
        };
        let resource = mcp_resource(&manifest);
        assert_eq!(resource.raw.description.as_deref(), Some("Current state"));
        assert_eq!(resource.raw.mime_type.as_deref(), Some("application/json"));
    }
}
