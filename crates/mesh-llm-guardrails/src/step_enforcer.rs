//! Required-step tracking and premature terminal nudges.
//! Ported from forge `src/forge/guardrails/step_enforcer.py` and
//! `src/forge/core/steps.py` (v0.6.0).
//!
//! Default-off for ingress traffic — agents calling mesh-llm directly
//! don't declare required steps. The MoA side can construct it with a
//! non-empty config when worker roles grow a notion of required tools.

use std::collections::{BTreeSet, HashMap};

use serde_json::Value;

use crate::nudges::{prerequisite_nudge, step_nudge};
use crate::types::{Nudge, NudgeKind, ToolCall};

/// One prerequisite entry. `Name(t)` = "any prior call to `t`
/// satisfies this prereq". `MatchArg { tool, arg }` = "a prior call
/// to `tool` with the same value of arg `arg` satisfies this prereq".
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Prereq {
    Name(String),
    MatchArg { tool: String, arg: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrerequisiteCheck {
    pub satisfied: bool,
    pub missing: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct StepTracker {
    pub required_steps: Vec<String>,
    completed: BTreeSet<String>,
    executed: HashMap<String, Vec<Value>>,
}

impl StepTracker {
    pub fn new(required_steps: Vec<String>) -> Self {
        Self {
            required_steps,
            completed: BTreeSet::new(),
            executed: HashMap::new(),
        }
    }

    pub fn record(&mut self, tool_name: &str, args: Value) {
        self.completed.insert(tool_name.to_string());
        self.executed
            .entry(tool_name.to_string())
            .or_default()
            .push(args);
    }

    pub fn is_satisfied(&self) -> bool {
        self.required_steps
            .iter()
            .all(|s| self.completed.contains(s))
    }

    pub fn pending(&self) -> Vec<String> {
        self.required_steps
            .iter()
            .filter(|s| !self.completed.contains(s.as_str()))
            .cloned()
            .collect()
    }

    pub fn check_prerequisites(&self, args: &Value, prerequisites: &[Prereq]) -> PrerequisiteCheck {
        let mut missing = Vec::new();
        for prereq in prerequisites {
            match prereq {
                Prereq::Name(t) => {
                    if !self.executed.contains_key(t) {
                        missing.push(t.clone());
                    }
                }
                Prereq::MatchArg { tool, arg } => {
                    let Some(calls) = self.executed.get(tool) else {
                        missing.push(tool.clone());
                        continue;
                    };
                    let required_value = args.get(arg);
                    let any_match = calls.iter().any(|c| c.get(arg) == required_value);
                    if !any_match {
                        missing.push(tool.clone());
                    }
                }
            }
        }
        let satisfied = missing.is_empty();
        PrerequisiteCheck { satisfied, missing }
    }

    pub fn completed_steps(&self) -> &BTreeSet<String> {
        &self.completed
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StepCheck {
    pub nudge: Option<Nudge>,
    pub needs_nudge: bool,
}

#[derive(Debug, Clone)]
pub struct StepEnforcer {
    pub tracker: StepTracker,
    pub terminal_tools: BTreeSet<String>,
    pub tool_prerequisites: HashMap<String, Vec<Prereq>>,
    pub max_premature_attempts: u32,
    pub max_prereq_violations: u32,
    premature_attempts: u32,
    consecutive_prereq_violations: u32,
}

impl StepEnforcer {
    pub fn new(
        required_steps: Vec<String>,
        terminal_tools: BTreeSet<String>,
        tool_prerequisites: HashMap<String, Vec<Prereq>>,
        max_premature_attempts: u32,
        max_prereq_violations: u32,
    ) -> Self {
        Self {
            tracker: StepTracker::new(required_steps),
            terminal_tools,
            tool_prerequisites,
            max_premature_attempts,
            max_prereq_violations,
            premature_attempts: 0,
            consecutive_prereq_violations: 0,
        }
    }

    /// Check whether `tool_calls` include a premature terminal call.
    /// Mutates `self` only on the premature path (escalates tier).
    pub fn check(&mut self, tool_calls: &[ToolCall]) -> StepCheck {
        let attempted = tool_calls
            .iter()
            .find(|tc| self.terminal_tools.contains(&tc.tool));

        match attempted {
            Some(tc) if !self.tracker.is_satisfied() => {
                self.premature_attempts += 1;
                let tier = self.premature_attempts.min(3) as u8;
                StepCheck {
                    nudge: Some(Nudge {
                        role: "user".into(),
                        content: step_nudge(&tc.tool, &self.tracker.pending(), tier),
                        kind: NudgeKind::Step,
                        tier,
                    }),
                    needs_nudge: true,
                }
            }
            _ => StepCheck {
                nudge: None,
                needs_nudge: false,
            },
        }
    }

    /// Whole-batch prerequisite check.
    pub fn check_prerequisites(&mut self, tool_calls: &[ToolCall]) -> StepCheck {
        for tc in tool_calls {
            let Some(prereqs) = self.tool_prerequisites.get(&tc.tool) else {
                continue;
            };
            if prereqs.is_empty() {
                continue;
            }
            let result = self.tracker.check_prerequisites(&tc.args, prereqs);
            if !result.satisfied {
                self.consecutive_prereq_violations += 1;
                return StepCheck {
                    nudge: Some(Nudge {
                        role: "user".into(),
                        content: prerequisite_nudge(&tc.tool, &result.missing),
                        kind: NudgeKind::Prerequisite,
                        tier: 0,
                    }),
                    needs_nudge: true,
                };
            }
        }
        StepCheck {
            nudge: None,
            needs_nudge: false,
        }
    }

    pub fn record(&mut self, tool_name: &str, args: Value) {
        self.tracker.record(tool_name, args);
    }

    pub fn is_satisfied(&self) -> bool {
        self.tracker.is_satisfied()
    }

    pub fn premature_exhausted(&self) -> bool {
        self.premature_attempts > self.max_premature_attempts
    }

    pub fn prereq_exhausted(&self) -> bool {
        self.consecutive_prereq_violations > self.max_prereq_violations
    }

    pub fn reset_premature(&mut self) {
        self.premature_attempts = 0;
    }

    pub fn reset_prereq_violations(&mut self) {
        self.consecutive_prereq_violations = 0;
    }

    pub fn terminal_reached(&self, tool_calls: &[ToolCall]) -> bool {
        tool_calls
            .iter()
            .any(|tc| self.terminal_tools.contains(&tc.tool))
            && self.tracker.is_satisfied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn terminal(names: &[&str]) -> BTreeSet<String> {
        names.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn premature_terminal_returns_nudge_and_escalates() {
        let mut e = StepEnforcer::new(
            vec!["read".into(), "plan".into()],
            terminal("submit".split_whitespace().collect::<Vec<_>>().as_slice()),
            HashMap::new(),
            3,
            2,
        );
        let calls = vec![ToolCall::new("submit", json!({}))];
        let c1 = e.check(&calls);
        assert!(c1.needs_nudge);
        assert_eq!(c1.nudge.unwrap().tier, 1);
        let c2 = e.check(&calls);
        assert_eq!(c2.nudge.unwrap().tier, 2);
        let c3 = e.check(&calls);
        assert_eq!(c3.nudge.unwrap().tier, 3);
        let c4 = e.check(&calls);
        assert_eq!(c4.nudge.unwrap().tier, 3, "tier clamps at 3");
        assert!(e.premature_exhausted());
    }

    #[test]
    fn satisfied_steps_allow_terminal() {
        let mut e = StepEnforcer::new(
            vec!["read".into()],
            terminal(&["submit"]),
            HashMap::new(),
            3,
            2,
        );
        e.record("read", json!({}));
        let calls = vec![ToolCall::new("submit", json!({}))];
        let c = e.check(&calls);
        assert!(!c.needs_nudge);
        assert!(e.terminal_reached(&calls));
    }

    #[test]
    fn arg_matched_prerequisite_requires_same_value() {
        let mut prereqs = HashMap::new();
        prereqs.insert(
            "write".into(),
            vec![Prereq::MatchArg {
                tool: "read".into(),
                arg: "path".into(),
            }],
        );
        let mut e = StepEnforcer::new(vec![], BTreeSet::new(), prereqs, 3, 2);
        e.record("read", json!({"path": "a.txt"}));
        // Write to a different path: missing prereq.
        let calls = vec![ToolCall::new("write", json!({"path": "b.txt"}))];
        let c = e.check_prerequisites(&calls);
        assert!(c.needs_nudge);
        // Write to the same path: satisfied.
        let calls = vec![ToolCall::new("write", json!({"path": "a.txt"}))];
        let c = e.check_prerequisites(&calls);
        assert!(!c.needs_nudge);
    }
}
