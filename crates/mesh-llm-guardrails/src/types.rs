//! Shared guardrail types — `ToolCall`, `Nudge`, `LlmResponse`,
//! `GuardrailAction`, `MessageType`.
//!
//! These mirror the forge core types but stay free of any framework
//! ties so they can live in any decorator / facade.

use serde_json::Value;

/// Validated tool invocation returned by the rescue parser or
/// extracted from a structured `tool_calls` channel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCall {
    pub tool: String,
    /// JSON arguments — kept as a `Value` so we don't lose type
    /// information when the model returned non-string fields.
    pub args: Value,
    pub reasoning: Option<String>,
}

impl ToolCall {
    pub fn new(tool: impl Into<String>, args: Value) -> Self {
        Self {
            tool: tool.into(),
            args,
            reasoning: None,
        }
    }
}

/// What kind of response the model produced — a list of structured
/// tool calls, or bare text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LlmResponse {
    ToolCalls(Vec<ToolCall>),
    Text(String),
}

/// Message-type tags used by the compactor to decide what to cut
/// first. Mesh-llm requests don't carry these tags natively; the
/// compactor classifies messages from their role and shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MessageType {
    SystemPrompt,
    UserInput,
    ToolCall,
    ToolResult,
    Reasoning,
    TextResponse,
    StepNudge,
    PrerequisiteNudge,
    RetryNudge,
    ContextWarning,
    Summary,
}

/// A corrective message to inject back into the conversation when
/// the model produced something unusable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Nudge {
    pub role: String,
    pub content: String,
    pub kind: NudgeKind,
    /// 0 = N/A, 1-3 = escalating tier (for step nudges).
    pub tier: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NudgeKind {
    Retry,
    UnknownTool,
    Step,
    Prerequisite,
}

/// What `Guardrails::check` says the caller should do next.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GuardrailAction {
    /// Tool calls are validated; let them through to the caller.
    Execute { tool_calls: Vec<ToolCall> },
    /// Inject `nudge` and re-run the inference.
    Retry { nudge: Nudge },
    /// Same as `Retry` but signals a premature terminal attempt —
    /// the step enforcer fired.
    StepBlocked { nudge: Nudge },
    /// Retry / error budget exhausted; surface the last text to the
    /// caller, do not 5xx.
    Fatal { reason: String },
}
