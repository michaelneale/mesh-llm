use super::{proto, ToolSummary};
use anyhow::{anyhow, Context, Result};
use rmcp::model::ServerInfo;
use serde::Serialize;

pub(crate) fn serialize_params<T: Serialize>(params: T) -> Result<String> {
    serde_json::to_string(&params).context("serialize plugin RPC params")
}

pub(crate) fn parse_optional_json(raw: &str) -> Result<Option<serde_json::Value>> {
    if raw.trim().is_empty() {
        Ok(None)
    } else {
        Ok(Some(
            serde_json::from_str(raw).context("parse plugin JSON payload")?,
        ))
    }
}

pub(crate) fn plugin_error(
    plugin_name: &str,
    method: &str,
    err: &proto::ErrorResponse,
) -> anyhow::Error {
    if err.data_json.trim().is_empty() {
        anyhow!(
            "Plugin '{}' failed '{}' (code {}): {}",
            plugin_name,
            method,
            err.code,
            err.message
        )
    } else {
        anyhow!(
            "Plugin '{}' failed '{}' (code {}): {} ({})",
            plugin_name,
            method,
            err.code,
            err.message,
            err.data_json
        )
    }
}

pub(crate) fn summarize_capabilities(server_info: &ServerInfo, extra: &[String]) -> Vec<String> {
    let mut capabilities = extra.to_vec();
    let caps = &server_info.capabilities;
    if caps.tools.is_some() {
        capabilities.push("mcp:tools".into());
    }
    if caps.prompts.is_some() {
        capabilities.push("mcp:prompts".into());
    }
    if caps.resources.is_some() {
        capabilities.push("mcp:resources".into());
    }
    if caps.completions.is_some() {
        capabilities.push("mcp:completions".into());
    }
    if caps.logging.is_some() {
        capabilities.push("mcp:logging".into());
    }
    if caps.tasks.is_some() {
        capabilities.push("mcp:tasks".into());
    }
    if let Some(extensions) = &caps.extensions {
        for key in extensions.keys() {
            capabilities.push(format!("mcp:extension:{key}"));
        }
    }
    capabilities.sort();
    capabilities.dedup();
    capabilities
}

pub(crate) fn format_args_for_log(args: &[String]) -> String {
    if args.is_empty() {
        "[]".to_string()
    } else {
        format!("[{}]", args.join(", "))
    }
}

pub(crate) fn format_slice_for_log(values: &[String]) -> String {
    if values.is_empty() {
        "[]".to_string()
    } else {
        format!("[{}]", values.join(", "))
    }
}

pub(crate) fn format_tool_names_for_log(tools: &[ToolSummary]) -> String {
    let names = tools
        .iter()
        .map(|tool| tool.name.clone())
        .collect::<Vec<_>>();
    format_slice_for_log(&names)
}
