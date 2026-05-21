//! `Guardrails` facade — bundles `ResponseValidator`, `StepEnforcer`,
//! and `ErrorTracker` into a two-method API. Ported from forge
//! `src/forge/guardrails/guardrails.py` (v0.6.0), with the PR #72
//! fix applied so `record()` passes `args` through to the step
//! enforcer.

use std::collections::{BTreeSet, HashMap};

use crate::error_tracker::ErrorTracker;
use crate::step_enforcer::{Prereq, StepEnforcer};
use crate::types::{GuardrailAction, LlmResponse, ToolCall};
use crate::validator::ResponseValidator;

#[derive(Debug, Clone)]
pub struct GuardrailConfig {
    pub tool_names: Vec<String>,
    pub terminal_tools: BTreeSet<String>,
    pub required_steps: Vec<String>,
    pub tool_prerequisites: HashMap<String, Vec<Prereq>>,
    pub max_retries: u32,
    pub max_tool_errors: u32,
    pub rescue_enabled: bool,
    pub max_premature_attempts: u32,
    pub max_prereq_violations: u32,
}

impl Default for GuardrailConfig {
    fn default() -> Self {
        Self {
            tool_names: Vec::new(),
            terminal_tools: BTreeSet::new(),
            required_steps: Vec::new(),
            tool_prerequisites: HashMap::new(),
            max_retries: 3,
            max_tool_errors: 2,
            rescue_enabled: true,
            max_premature_attempts: 3,
            max_prereq_violations: 2,
        }
    }
}

pub struct Guardrails {
    validator: ResponseValidator,
    enforcer: StepEnforcer,
    errors: ErrorTracker,
}

impl Guardrails {
    pub fn new(cfg: GuardrailConfig) -> Self {
        let validator = ResponseValidator::new(cfg.tool_names.clone(), cfg.rescue_enabled);
        let enforcer = StepEnforcer::new(
            cfg.required_steps,
            cfg.terminal_tools,
            cfg.tool_prerequisites,
            cfg.max_premature_attempts,
            cfg.max_prereq_violations,
        );
        let errors = ErrorTracker::new(cfg.max_retries, cfg.max_tool_errors);
        Self {
            validator,
            enforcer,
            errors,
        }
    }

    /// Run all guardrail checks against a single LLM response.
    pub fn check(&mut self, response: LlmResponse) -> GuardrailAction {
        let validation = self.validator.validate(response);
        if validation.needs_retry {
            self.errors.record_retry();
            if self.errors.retries_exhausted() {
                return GuardrailAction::Fatal {
                    reason: "too many consecutive bad responses".into(),
                };
            }
            return GuardrailAction::Retry {
                nudge: validation.nudge.expect("retry path always has a nudge"),
            };
        }
        self.errors.reset_retries();

        let tool_calls = validation
            .tool_calls
            .expect("non-retry validation produces tool_calls");

        // Required-step enforcement.
        let step = self.enforcer.check(&tool_calls);
        if step.needs_nudge {
            if self.enforcer.premature_exhausted() {
                return GuardrailAction::Fatal {
                    reason: "model repeatedly skipped required steps".into(),
                };
            }
            return GuardrailAction::StepBlocked {
                nudge: step.nudge.expect("needs_nudge path always has a nudge"),
            };
        }

        GuardrailAction::Execute { tool_calls }
    }

    /// Record successfully-executed tool calls (with their args, so
    /// arg-matched prerequisites work). Returns true if the terminal
    /// tool was reached and all required steps are now satisfied —
    /// caller can stop looping.
    ///
    /// Mirrors forge PR #72: the args are passed through to the step
    /// tracker so prerequisite checks see them.
    pub fn record(&mut self, executed: &[ToolCall]) -> bool {
        let mut terminal_hit = false;
        for call in executed {
            self.enforcer.record(&call.tool, call.args.clone());
            if self.enforcer.terminal_tools.contains(&call.tool) {
                terminal_hit = true;
            }
        }
        self.errors.reset_errors();
        self.enforcer.reset_premature();
        terminal_hit && self.enforcer.is_satisfied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn cfg() -> GuardrailConfig {
        GuardrailConfig {
            tool_names: vec!["read".into(), "respond".into()],
            terminal_tools: ["respond"].into_iter().map(String::from).collect(),
            required_steps: vec!["read".into()],
            ..Default::default()
        }
    }

    #[test]
    fn rescue_followed_by_step_block_then_satisfaction() {
        let mut g = Guardrails::new(cfg());
        // Step 1: bare-text → rescue salvages a respond() call → it's
        // a terminal call with required steps unsatisfied → StepBlocked.
        let action = g.check(LlmResponse::Text(
            r#"{"tool": "respond", "args": {"message": "hi"}}"#.into(),
        ));
        assert!(matches!(action, GuardrailAction::StepBlocked { .. }));

        // Step 2: read() executes — satisfies required steps.
        let action = g.check(LlmResponse::ToolCalls(vec![ToolCall::new(
            "read",
            json!({}),
        )]));
        assert!(matches!(action, GuardrailAction::Execute { .. }));
        let done = g.record(&[ToolCall::new("read", json!({}))]);
        assert!(!done, "respond not yet called");

        // Step 3: respond() now succeeds.
        let action = g.check(LlmResponse::ToolCalls(vec![ToolCall::new(
            "respond",
            json!({"message": "done"}),
        )]));
        assert!(matches!(action, GuardrailAction::Execute { .. }));
        let done = g.record(&[ToolCall::new("respond", json!({"message": "done"}))]);
        assert!(done);
    }

    #[test]
    fn retry_budget_eventually_fatal() {
        let mut g = Guardrails::new(GuardrailConfig {
            tool_names: vec!["read".into()],
            max_retries: 2,
            ..Default::default()
        });
        // 3 consecutive retries → exhausted (strictly greater than max).
        for _ in 0..3 {
            let _ = g.check(LlmResponse::Text("garbage".into()));
        }
        let action = g.check(LlmResponse::Text("garbage".into()));
        assert!(matches!(action, GuardrailAction::Fatal { .. }));
    }
}
