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

/// Pick the best tool proposal: prefer proposals that have arguments,
/// then by confidence.  A proposal without arguments (e.g. from a fast
/// worker that only got tool names in the system prompt) should lose to
/// one that has actual arguments.
fn best_tool_proposal<'a>(proposals: &[&'a WorkerOutput]) -> &'a WorkerOutput {
    proposals
        .iter()
        .copied()
        .max_by(|a, b| {
            let a_has_args = a.tool_arguments.is_some()
                && a.tool_arguments.as_ref() != Some(&Value::Object(Default::default()));
            let b_has_args = b.tool_arguments.is_some()
                && b.tool_arguments.as_ref() != Some(&Value::Object(Default::default()));
            a_has_args
                .cmp(&b_has_args)
                .then(a.confidence.partial_cmp(&b.confidence).unwrap())
        })
        .unwrap()
}

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
                let best = best_tool_proposal(&tool_proposals);
                return Decision::ToolCall {
                    name: first.to_string(),
                    arguments: best
                        .tool_arguments
                        .clone()
                        .unwrap_or(Value::Object(Default::default())),
                };
            }

            // Different tools proposed — check if one is clearly dominant
            let max_conf = best_tool_proposal(&tool_proposals);
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

/// Try to decide early with a partial set of worker outputs.
///
/// Returns `Some(decision)` if we can confidently resolve without waiting
/// for more workers.  Returns `None` if we need to keep waiting.
///
/// `total_workers` is how many were dispatched.
/// `total_finished` includes both successful outputs AND failed workers,
/// so we know when there's no point waiting for more.
pub fn try_early_decision(
    outputs: &[WorkerOutput],
    total_workers: usize,
    total_finished: usize,
    has_tools: bool,
) -> Option<Decision> {
    if outputs.is_empty() {
        // All workers finished but none succeeded
        if total_finished >= total_workers {
            return None; // Let the caller handle the empty-outputs case
        }
        return None;
    }

    let remaining = total_workers.saturating_sub(total_finished);

    // ── Only one worker will ever respond ───────────────────────────
    // If we have 1 successful output and no more workers are coming,
    // return it immediately — no point waiting.
    if outputs.len() == 1 && remaining == 0 {
        return Some(single_output_decision(&outputs[0], has_tools));
    }

    // ── Single output with others still pending ─────────────────────
    // Wait for at least one more so we have a chance to detect disagreement —
    // UNLESS most other workers have already failed, in which case return
    // the sole survivor immediately rather than waiting for stragglers.
    if outputs.len() < 2 && remaining > 0 {
        let failed_count = total_finished - outputs.len();
        let majority_failed = failed_count > 0 && failed_count >= total_workers / 2;
        if !majority_failed {
            return None;
        }
        // Majority failed — return the sole survivor
        tracing::info!(
            "moa: early exit — sole survivor, {failed_count}/{total_workers} workers failed",
        );
        return Some(single_output_decision(&outputs[0], has_tools));
    }

    // ── 2+ outputs: check for consensus ─────────────────────────────

    let answers: Vec<&WorkerOutput> = outputs
        .iter()
        .filter(|o| o.kind == OutputKind::Answer)
        .collect();
    let tool_proposals: Vec<&WorkerOutput> = outputs
        .iter()
        .filter(|o| o.kind == OutputKind::ToolProposal)
        .collect();

    // All agree on an answer — no need to wait for stragglers
    if answers.len() >= 2 && tool_proposals.is_empty() {
        let best = answers
            .iter()
            .max_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap())
            .unwrap();
        if best.confidence >= 0.5 {
            tracing::info!(
                "moa: early exit — {} workers agree on answer (conf={:.2}), {} still pending",
                answers.len(),
                best.confidence,
                remaining,
            );
            return Some(Decision::Answer(best.payload.clone()));
        }
    }

    // All agree on the same tool call
    if has_tools && tool_proposals.len() >= 2 && answers.is_empty() {
        let tool_names: Vec<&str> = tool_proposals
            .iter()
            .filter_map(|o| o.tool_name.as_deref())
            .collect();
        if !tool_names.is_empty() {
            let first = tool_names[0];
            let unanimous = tool_names.iter().all(|n| *n == first);
            if unanimous {
                let best = best_tool_proposal(&tool_proposals);
                tracing::info!(
                    "moa: early exit — {} workers agree on tool '{}', {} still pending",
                    tool_proposals.len(),
                    first,
                    remaining,
                );
                return Some(Decision::ToolCall {
                    name: first.to_string(),
                    arguments: best
                        .tool_arguments
                        .clone()
                        .unwrap_or(serde_json::Value::Object(Default::default())),
                });
            }
        }
    }

    // Conflict detected early — some say tool, some say answer.
    // Escalate to reducer now, don't wait for more conflicting opinions.
    if !tool_proposals.is_empty() && !answers.is_empty() {
        tracing::info!(
            "moa: early escalation — {} tool proposals vs {} answers, {} still pending",
            tool_proposals.len(),
            answers.len(),
            remaining,
        );
        return Some(Decision::NeedsReducer {
            reason: "some workers propose tools, others answer directly".into(),
        });
    }

    // Not enough signal yet — keep waiting
    None
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

    // ── Early decision tests ────────────────────────────────────

    #[test]
    fn early_decision_none_with_one_of_three() {
        let outputs = vec![make_output(OutputKind::Answer, 0.9, "Paris")];
        // 1 of 3 — too early to decide
        assert!(try_early_decision(&outputs, 3, outputs.len(), false).is_none());
    }

    #[test]
    fn early_decision_consensus_two_of_three() {
        let outputs = vec![
            make_output(OutputKind::Answer, 0.8, "Paris"),
            make_output(OutputKind::Answer, 0.9, "Paris is the capital"),
        ];
        // 2 of 3 agree — early exit
        match try_early_decision(&outputs, 3, outputs.len(), false) {
            Some(Decision::Answer(text)) => assert!(text.contains("Paris")),
            other => panic!("expected early Answer, got {other:?}"),
        }
    }

    #[test]
    fn early_decision_tool_consensus() {
        let outputs = vec![
            make_tool_output(0.8, "read_file", serde_json::json!({"path": "a.rs"})),
            make_tool_output(0.7, "read_file", serde_json::json!({"path": "a.rs"})),
        ];
        match try_early_decision(&outputs, 3, outputs.len(), true) {
            Some(Decision::ToolCall { name, .. }) => assert_eq!(name, "read_file"),
            other => panic!("expected early ToolCall, got {other:?}"),
        }
    }

    #[test]
    fn early_decision_conflict_escalates() {
        let outputs = vec![
            make_tool_output(0.7, "read_file", serde_json::json!({})),
            make_output(OutputKind::Answer, 0.8, "I know the answer"),
        ];
        match try_early_decision(&outputs, 3, outputs.len(), true) {
            Some(Decision::NeedsReducer { .. }) => {}
            other => panic!("expected early NeedsReducer, got {other:?}"),
        }
    }

    #[test]
    fn early_decision_single_survivor() {
        // 1 success out of 3, other 2 failed — should return the single answer
        let outputs = vec![make_output(OutputKind::Answer, 0.8, "Paris")];
        // total_workers=3, total_finished=3 (1 success + 2 failures), remaining=0
        match try_early_decision(&outputs, 3, 3, false) {
            Some(Decision::Answer(text)) => assert!(text.contains("Paris")),
            other => panic!("expected early Answer for sole survivor, got {other:?}"),
        }
    }

    #[test]
    fn early_decision_low_confidence_waits() {
        let outputs = vec![
            make_output(OutputKind::Answer, 0.3, "maybe Paris"),
            make_output(OutputKind::Answer, 0.4, "could be Paris"),
        ];
        // Both answers but low confidence — should wait for more
        assert!(try_early_decision(&outputs, 3, outputs.len(), false).is_none());
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

    #[test]
    fn early_decision_sole_survivor_majority_failed() {
        // 1 success, 3 failures, 1 still pending — majority failed, return sole survivor
        let outputs = vec![make_output(OutputKind::Answer, 0.8, "Paris")];
        // total_workers=5, total_finished=4 (1 success + 3 failures), remaining=1
        match try_early_decision(&outputs, 5, 4, false) {
            Some(Decision::Answer(text)) => assert!(text.contains("Paris")),
            other => {
                panic!("expected early Answer for sole survivor (majority failed), got {other:?}")
            }
        }
    }

    #[test]
    fn early_decision_sole_survivor_minority_failed_waits() {
        // 1 success, 1 failure, 3 still pending — minority failed, wait for more
        let outputs = vec![make_output(OutputKind::Answer, 0.8, "Paris")];
        // total_workers=5, total_finished=2 (1 success + 1 failure), remaining=3
        assert!(try_early_decision(&outputs, 5, 2, false).is_none());
    }

    #[test]
    fn best_tool_proposal_prefers_arguments() {
        let without_args = WorkerOutput {
            kind: OutputKind::ToolProposal,
            confidence: 0.9,
            tool_name: Some("read_file".into()),
            tool_arguments: None,
            payload: "calling read_file".into(),
            model: "fast-model".into(),
            role: crate::worker::WorkerRole::Fast,
            elapsed_ms: 100,
        };
        let with_args = WorkerOutput {
            kind: OutputKind::ToolProposal,
            confidence: 0.6,
            tool_name: Some("read_file".into()),
            tool_arguments: Some(serde_json::json!({"path": "/tmp/test.txt"})),
            payload: "calling read_file".into(),
            model: "strong-model".into(),
            role: crate::worker::WorkerRole::Strong,
            elapsed_ms: 3000,
        };
        let proposals = vec![&without_args, &with_args];
        let best = best_tool_proposal(&proposals);
        assert_eq!(best.model, "strong-model");
        assert!(best.tool_arguments.is_some());
    }

    #[test]
    fn best_tool_proposal_falls_back_to_confidence() {
        let a = WorkerOutput {
            kind: OutputKind::ToolProposal,
            confidence: 0.6,
            tool_name: Some("read_file".into()),
            tool_arguments: Some(serde_json::json!({"path": "/a.txt"})),
            payload: "calling read_file".into(),
            model: "model-a".into(),
            role: crate::worker::WorkerRole::Specialist,
            elapsed_ms: 2000,
        };
        let b = WorkerOutput {
            kind: OutputKind::ToolProposal,
            confidence: 0.9,
            tool_name: Some("read_file".into()),
            tool_arguments: Some(serde_json::json!({"path": "/b.txt"})),
            payload: "calling read_file".into(),
            model: "model-b".into(),
            role: crate::worker::WorkerRole::Strong,
            elapsed_ms: 3000,
        };
        let proposals = vec![&a, &b];
        let best = best_tool_proposal(&proposals);
        // Both have args, so confidence wins
        assert_eq!(best.model, "model-b");
    }
}
