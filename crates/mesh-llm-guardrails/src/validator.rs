//! Response validator. Ported from forge
//! `src/forge/guardrails/response_validator.py` (v0.6.0).
//!
//! Stateless — safe to reuse across turns and sessions.

use crate::nudges::{retry_nudge, unknown_tool_nudge};
use crate::rescue::rescue_tool_call;
use crate::types::{LlmResponse, Nudge, NudgeKind, ToolCall};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationResult {
    pub tool_calls: Option<Vec<ToolCall>>,
    pub nudge: Option<Nudge>,
    pub needs_retry: bool,
}

#[derive(Debug, Clone)]
pub struct ResponseValidator {
    pub tool_names: Vec<String>,
    pub rescue_enabled: bool,
}

impl ResponseValidator {
    pub fn new(tool_names: Vec<String>, rescue_enabled: bool) -> Self {
        Self {
            tool_names,
            rescue_enabled,
        }
    }

    pub fn validate(&self, response: LlmResponse) -> ValidationResult {
        match response {
            LlmResponse::Text(text) => {
                if self.rescue_enabled {
                    let rescued = rescue_tool_call(&text, &self.tool_names);
                    if !rescued.is_empty() {
                        return ValidationResult {
                            tool_calls: Some(rescued),
                            nudge: None,
                            needs_retry: false,
                        };
                    }
                }
                ValidationResult {
                    tool_calls: None,
                    nudge: Some(Nudge {
                        role: "user".into(),
                        content: retry_nudge(&text),
                        kind: NudgeKind::Retry,
                        tier: 0,
                    }),
                    needs_retry: true,
                }
            }
            LlmResponse::ToolCalls(tool_calls) => {
                let unknown = tool_calls
                    .iter()
                    .find(|tc| !self.tool_names.iter().any(|t| t == &tc.tool));
                if let Some(bad) = unknown {
                    return ValidationResult {
                        tool_calls: None,
                        nudge: Some(Nudge {
                            role: "user".into(),
                            content: unknown_tool_nudge(&bad.tool, &self.tool_names),
                            kind: NudgeKind::UnknownTool,
                            tier: 0,
                        }),
                        needs_retry: true,
                    };
                }
                ValidationResult {
                    tool_calls: Some(tool_calls),
                    nudge: None,
                    needs_retry: false,
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn validator() -> ResponseValidator {
        ResponseValidator::new(vec!["read_file".into(), "respond".into()], true)
    }

    #[test]
    fn rescues_text_to_tool_calls() {
        let v = validator();
        let r = v.validate(LlmResponse::Text(
            r#"{"tool": "read_file", "args": {"path": "x"}}"#.into(),
        ));
        assert!(!r.needs_retry);
        let calls = r.tool_calls.unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].tool, "read_file");
    }

    #[test]
    fn bare_text_returns_retry_nudge() {
        let v = validator();
        let r = v.validate(LlmResponse::Text("just chatting".into()));
        assert!(r.needs_retry);
        assert_eq!(r.nudge.unwrap().kind, NudgeKind::Retry);
    }

    #[test]
    fn unknown_tool_returns_unknown_tool_nudge() {
        let v = validator();
        let r = v.validate(LlmResponse::ToolCalls(vec![ToolCall::new(
            "nuke",
            json!({}),
        )]));
        assert!(r.needs_retry);
        assert_eq!(r.nudge.unwrap().kind, NudgeKind::UnknownTool);
    }

    #[test]
    fn known_tool_passes_through() {
        let v = validator();
        let r = v.validate(LlmResponse::ToolCalls(vec![ToolCall::new(
            "read_file",
            json!({"path": "x"}),
        )]));
        assert!(!r.needs_retry);
        assert_eq!(r.tool_calls.unwrap().len(), 1);
    }

    #[test]
    fn rescue_disabled_skips_text_parse() {
        let v = ResponseValidator::new(vec!["read_file".into()], false);
        let r = v.validate(LlmResponse::Text(
            r#"{"tool": "read_file", "args": {"path": "x"}}"#.into(),
        ));
        assert!(r.needs_retry);
    }
}
