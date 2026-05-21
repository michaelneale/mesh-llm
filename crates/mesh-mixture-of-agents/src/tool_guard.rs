//! Allowed-tool enforcement.
//!
//! Workers sometimes hallucinate tool names — e.g. proposing
//! `execute_typescript` when only `shell` was declared on the request.
//! This module demotes those proposals to `Uncertainty` before they
//! reach arbitration, so a hallucinated name can't win consensus or
//! get emitted as a `tool_call` to the client.

use crate::normalize::{OutputKind, WorkerOutput};

/// Demote a `ToolProposal` whose `tool_name` is not in `allowed_tools`
/// to `Uncertainty`. This guards downstream arbitration from worker
/// hallucinations like proposing `execute_typescript` when only `shell`
/// is declared.
///
/// If `allowed_tools` is empty (no tools were declared on the request
/// body) we leave the proposal as-is — the arbiter will route to the
/// reducer which has the same call-site policy.
pub(crate) fn enforce_allowed_tools(
    output: &mut WorkerOutput,
    allowed_tools: &[String],
    model: &str,
) {
    if allowed_tools.is_empty() {
        return;
    }
    if output.kind != OutputKind::ToolProposal {
        return;
    }
    let Some(ref name) = output.tool_name else {
        return;
    };
    if allowed_tools.iter().any(|t| t == name) {
        return;
    }
    tracing::warn!(
        "moa: worker {model} proposed unknown tool {name:?}, demoting to uncertainty \
         (allowed: {allowed_tools:?})"
    );
    output.kind = OutputKind::Uncertainty;
    output.tool_name = None;
    output.tool_arguments = None;
    // Drop confidence so this proposal doesn't outrank real ones in any
    // tie-breaking path that still inspects it.
    output.confidence = 0.0;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::worker::WorkerRole;
    use serde_json::json;

    fn proposal(tool: &str) -> WorkerOutput {
        WorkerOutput {
            kind: OutputKind::ToolProposal,
            confidence: 0.9,
            tool_name: Some(tool.to_string()),
            tool_arguments: Some(json!({"path": "README.md"})),
            payload: format!("calling {tool}"),
            model: "alpha".into(),
            role: WorkerRole::Strong,
            elapsed_ms: 0,
        }
    }

    #[test]
    fn allowed_tool_passes_through() {
        let mut out = proposal("read_file");
        enforce_allowed_tools(&mut out, &["read_file".into()], "alpha");
        assert_eq!(out.kind, OutputKind::ToolProposal);
        assert_eq!(out.tool_name.as_deref(), Some("read_file"));
        assert!(out.tool_arguments.is_some());
        assert!((out.confidence - 0.9).abs() < f32::EPSILON);
    }

    #[test]
    fn unknown_tool_is_demoted() {
        let mut out = proposal("execute_typescript");
        enforce_allowed_tools(&mut out, &["shell".into()], "alpha");
        assert_eq!(out.kind, OutputKind::Uncertainty);
        assert!(out.tool_name.is_none());
        assert!(out.tool_arguments.is_none());
        assert_eq!(
            out.confidence, 0.0,
            "demoted proposals must drop confidence so they don't outrank real answers",
        );
    }

    #[test]
    fn empty_allowed_list_is_noop() {
        // No tools declared on the request — don't second-guess the worker
        // here; the reducer applies the same policy downstream.
        let mut out = proposal("anything");
        enforce_allowed_tools(&mut out, &[], "alpha");
        assert_eq!(out.kind, OutputKind::ToolProposal);
        assert_eq!(out.tool_name.as_deref(), Some("anything"));
    }

    #[test]
    fn non_proposal_outputs_untouched() {
        let mut out = WorkerOutput {
            kind: OutputKind::Answer,
            confidence: 0.7,
            tool_name: None,
            tool_arguments: None,
            payload: "Tokyo.".into(),
            model: "beta".into(),
            role: WorkerRole::Fast,
            elapsed_ms: 0,
        };
        enforce_allowed_tools(&mut out, &["read_file".into()], "beta");
        assert_eq!(out.kind, OutputKind::Answer);
        assert!((out.confidence - 0.7).abs() < f32::EPSILON);
    }
}
