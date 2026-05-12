//! Deterministic arbitration of worker outputs.
//!
//! The arbiter uses code, not models, to decide the outcome.
//! Models are only called (via the reducer) when there's genuine ambiguity.
//!
//! Decision priority:
//! 1. Unanimous tool proposal → emit tool call
//! 2. High-confidence tool proposal with no dissent → emit tool call
//! 3. Unanimous answers → pick highest confidence
//! 4. Conflicting outputs → escalate to reducer
//! 5. All uncertainty → escalate to reducer

use crate::normalize::{OutputKind, WorkerOutput};
use serde_json::Value;

/// What the arbiter decided.
#[derive(Debug)]
pub enum Decision {
    /// Emit a text answer.
    Answer(String),
    /// Emit a tool call.
    ToolCall { name: String, arguments: Value },
    /// Ambiguous — needs the reducer model.
    NeedsReducer { reason: String },
}

/// Arbitrate worker outputs into a single decision.
pub fn arbitrate(outputs: &[WorkerOutput], has_tools: bool) -> Decision {
    if outputs.is_empty() {
        return Decision::NeedsReducer {
            reason: "no worker outputs".into(),
        };
    }

    if outputs.len() == 1 {
        return single_output_decision(&outputs[0], has_tools);
    }

    let tool_proposals: Vec<&WorkerOutput> = outputs
        .iter()
        .filter(|o| o.kind == OutputKind::ToolProposal)
        .collect();
    let answers: Vec<&WorkerOutput> = outputs
        .iter()
        .filter(|o| o.kind == OutputKind::Answer)
        .collect();
    let critiques: Vec<&WorkerOutput> = outputs
        .iter()
        .filter(|o| o.kind == OutputKind::Critique)
        .collect();
    let uncertainties: Vec<&WorkerOutput> = outputs
        .iter()
        .filter(|o| o.kind == OutputKind::Uncertainty)
        .collect();

    // If everyone is uncertain, reducer
    if uncertainties.len() == outputs.len() {
        return Decision::NeedsReducer {
            reason: "all workers uncertain".into(),
        };
    }

    // ── Tool call arbitration ────────────────────────────────────

    if has_tools && !tool_proposals.is_empty() {
        // Check if any critique opposes the tool call
        let has_tool_dissent = critiques.iter().any(|c| {
            c.payload.to_lowercase().contains("don't")
                || c.payload.to_lowercase().contains("should not")
                || c.payload.to_lowercase().contains("no tool")
        });

        if has_tool_dissent {
            return Decision::NeedsReducer {
                reason: "tool proposal with dissenting critique".into(),
            };
        }

        // All tool proposals agree on the same tool?
        let tool_names: Vec<&str> = tool_proposals
            .iter()
            .filter_map(|o| o.tool_name.as_deref())
            .collect();

        if !tool_names.is_empty() {
            // If some workers propose tools and others answer directly, conflict
            if !answers.is_empty() {
                return Decision::NeedsReducer {
                    reason: "some workers propose tools, others answer directly".into(),
                };
            }

            let first = tool_names[0];
            let unanimous = tool_names.iter().all(|n| *n == first);

            if unanimous {
                // Pick the highest-confidence proposal's arguments
                let best = tool_proposals
                    .iter()
                    .max_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap())
                    .unwrap();
                return Decision::ToolCall {
                    name: first.to_string(),
                    arguments: best
                        .tool_arguments
                        .clone()
                        .unwrap_or(Value::Object(Default::default())),
                };
            }

            // Different tools proposed — check if one is clearly dominant
            let max_conf = tool_proposals
                .iter()
                .max_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap())
                .unwrap();
            let others_low = tool_proposals
                .iter()
                .filter(|o| o.tool_name != max_conf.tool_name)
                .all(|o| o.confidence < 0.5);

            if max_conf.confidence > 0.7 && others_low {
                return Decision::ToolCall {
                    name: max_conf.tool_name.clone().unwrap_or_default(),
                    arguments: max_conf
                        .tool_arguments
                        .clone()
                        .unwrap_or(Value::Object(Default::default())),
                };
            }

            return Decision::NeedsReducer {
                reason: format!("conflicting tool proposals: {}", tool_names.join(" vs ")),
            };
        }

        // Tool proposals without extractable names — single high-confidence?
        if tool_proposals.len() == 1 && tool_proposals[0].confidence > 0.6 {
            return Decision::NeedsReducer {
                reason: "tool proposal without parseable tool name".into(),
            };
        }
    }

    // ── Answer arbitration ───────────────────────────────────────

    if !answers.is_empty() {
        // Pick the highest-confidence answer
        let best = answers
            .iter()
            .max_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap())
            .unwrap();

        // If confidence is low and there's critique, reducer
        if best.confidence < 0.5 && !critiques.is_empty() {
            return Decision::NeedsReducer {
                reason: "low confidence answer with critique".into(),
            };
        }

        return Decision::Answer(best.payload.clone());
    }

    // Only critiques and/or uncertainty — reducer
    Decision::NeedsReducer {
        reason: "no clear answer or tool proposal".into(),
    }
}

fn single_output_decision(output: &WorkerOutput, has_tools: bool) -> Decision {
    match output.kind {
        OutputKind::ToolProposal if has_tools => {
            if let Some(ref name) = output.tool_name {
                Decision::ToolCall {
                    name: name.clone(),
                    arguments: output
                        .tool_arguments
                        .clone()
                        .unwrap_or(Value::Object(Default::default())),
                }
            } else {
                Decision::Answer(output.payload.clone())
            }
        }
        OutputKind::Uncertainty => Decision::NeedsReducer {
            reason: "single worker uncertain".into(),
        },
        _ => Decision::Answer(output.payload.clone()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::worker::WorkerRole;

    fn make_output(kind: OutputKind, confidence: f32, payload: &str) -> WorkerOutput {
        WorkerOutput {
            kind,
            confidence,
            tool_name: None,
            tool_arguments: None,
            payload: payload.to_string(),
            model: "test".to_string(),
            role: WorkerRole::Generalist,
            elapsed_ms: 0,
        }
    }

    fn make_tool_output(confidence: f32, tool: &str, args: Value) -> WorkerOutput {
        WorkerOutput {
            kind: OutputKind::ToolProposal,
            confidence,
            tool_name: Some(tool.to_string()),
            tool_arguments: Some(args),
            payload: "propose tool".to_string(),
            model: "test".to_string(),
            role: WorkerRole::Generalist,
            elapsed_ms: 0,
        }
    }

    #[test]
    fn unanimous_answer_picks_highest_confidence() {
        let outputs = vec![
            make_output(OutputKind::Answer, 0.7, "Paris"),
            make_output(OutputKind::Answer, 0.9, "Paris is the capital"),
        ];
        match arbitrate(&outputs, false) {
            Decision::Answer(text) => assert!(text.contains("Paris")),
            other => panic!("expected Answer, got {other:?}"),
        }
    }

    #[test]
    fn unanimous_tool_proposal() {
        let outputs = vec![
            make_tool_output(0.8, "read_file", serde_json::json!({"path": "a.rs"})),
            make_tool_output(0.7, "read_file", serde_json::json!({"path": "a.rs"})),
        ];
        match arbitrate(&outputs, true) {
            Decision::ToolCall { name, .. } => assert_eq!(name, "read_file"),
            other => panic!("expected ToolCall, got {other:?}"),
        }
    }

    #[test]
    fn conflicting_tools_needs_reducer() {
        let outputs = vec![
            make_tool_output(0.6, "read_file", serde_json::json!({})),
            make_tool_output(0.6, "web_search", serde_json::json!({})),
        ];
        match arbitrate(&outputs, true) {
            Decision::NeedsReducer { reason } => assert!(reason.contains("conflicting")),
            other => panic!("expected NeedsReducer, got {other:?}"),
        }
    }

    #[test]
    fn tool_vs_answer_needs_reducer() {
        let outputs = vec![
            make_tool_output(0.7, "read_file", serde_json::json!({})),
            make_output(OutputKind::Answer, 0.8, "I can answer that directly"),
        ];
        match arbitrate(&outputs, true) {
            Decision::NeedsReducer { reason } => assert!(reason.contains("some workers")),
            other => panic!("expected NeedsReducer, got {other:?}"),
        }
    }

    #[test]
    fn all_uncertain_needs_reducer() {
        let outputs = vec![
            make_output(OutputKind::Uncertainty, 0.2, "not sure"),
            make_output(OutputKind::Uncertainty, 0.3, "hard to say"),
        ];
        match arbitrate(&outputs, false) {
            Decision::NeedsReducer { reason } => assert!(reason.contains("uncertain")),
            other => panic!("expected NeedsReducer, got {other:?}"),
        }
    }
}
