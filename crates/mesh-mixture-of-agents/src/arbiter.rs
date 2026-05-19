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
                .then(a.confidence.total_cmp(&b.confidence))
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
            .max_by(|a, b| a.confidence.total_cmp(&b.confidence))
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

    // Workers agree on an answer — but agreement means the *content*
    // overlaps, not just that they all produced an Answer-kind output.
    // Two workers saying "Paris" and "Berlin" must not be treated as
    // consensus. Find the largest cluster of content-similar answers
    // and only early-exit if it's ≥2 workers AND a majority of answers.
    if answers.len() >= 2 && tool_proposals.is_empty() {
        if let Some((cluster_size, best)) = largest_agreeing_cluster(&answers) {
            let majority = cluster_size * 2 >= answers.len();
            if majority && best.confidence >= 0.5 {
                tracing::info!(
                    "moa: early exit — {}/{} workers agree on answer (conf={:.2}), {} still pending",
                    cluster_size,
                    answers.len(),
                    best.confidence,
                    remaining,
                );
                return Some(Decision::Answer(best.payload.clone()));
            }
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

/// Tokens that flip meaning. Always preserved as content (even if they'd
/// fail the length / stopword filters), AND a leftover negation in either
/// side of a comparison blocks clustering regardless of subset relation.
const NEGATION_TOKENS: &[&str] = &[
    // Standalone negations
    "no", "not", "never", "none", "nor", "without", "neither", "cannot",
    // Joined contractions (apostrophe stripped)
    "dont", "doesnt", "didnt", "wont", "wouldnt", "cant", "shouldnt", "isnt", "arent", "wasnt",
    "werent",
    // Split contractions (apostrophe → space → two tokens; the prefix
    // is what survives because the `t` half is <3 chars and gets filtered).
    "don", "doesn", "didn", "won", "wouldn", "shouldn", "isn", "aren", "wasn", "weren", "hadn",
    "haven", "hasn",
];

/// Content-bearing tokens extracted from an answer payload.
///
/// Pipeline:
/// 1. Lowercase the text and replace non-alphanumeric chars with spaces.
/// 2. Split on whitespace.
/// 3. Drop stopwords and tokens shorter than 3 chars — UNLESS they're
///    in `NEGATION_TOKENS` (always kept, because they flip meaning) or
///    pure digits (kept so "42" survives).
///
/// The output is a deduplicated set: token *presence* matters, not
/// frequency. That matches how the subset-containment rule below
/// reasons about agreement.
fn content_tokens(text: &str) -> std::collections::HashSet<String> {
    // Common English stopwords. Intentionally small — scaffolding words
    // we don't want to dominate the comparison.
    const STOPWORDS: &[&str] = &[
        "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "in", "into", "is",
        "it", "its", "of", "on", "or", "out", "so", "that", "the", "their", "them", "then",
        "there", "these", "they", "this", "those", "to", "was", "were", "will", "with",
    ];

    text.to_lowercase()
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { ' ' })
        .collect::<String>()
        .split_whitespace()
        .filter(|tok| {
            if NEGATION_TOKENS.contains(tok) {
                return true;
            }
            if STOPWORDS.contains(tok) {
                return false;
            }
            // Keep digit-only tokens regardless of length (so "42" survives).
            if tok.chars().all(|c| c.is_ascii_digit()) {
                return true;
            }
            tok.len() >= 3
        })
        .map(|t| t.to_string())
        .collect()
}

/// Does the symmetric difference between `a` and `b` contain any
/// negation token? If yes, one side has a negation the other lacks,
/// and the two answers must not cluster even when subset would hold.
fn has_negation_mismatch(
    a: &std::collections::HashSet<String>,
    b: &std::collections::HashSet<String>,
) -> bool {
    a.symmetric_difference(b)
        .any(|tok| NEGATION_TOKENS.contains(&tok.as_str()))
}

/// Find the largest cluster of answers that agree on content.
///
/// Two answers agree iff:
/// - the smaller content-token set is a subset of the larger, AND
/// - their symmetric difference contains no negation tokens
///
/// Rationale: a terse answer ("Paris") and a verbose one ("Paris is
/// the capital of France") share all of the short answer's content
/// tokens, so the terse one is a subset of the verbose one. Two
/// genuinely conflicting answers ("…is Paris" vs "…is Berlin") each
/// have a leftover content token the other lacks. And an answer that
/// adds a negation ("use grep" vs "do not use grep") is rejected by
/// the negation-mismatch guard even when subset would otherwise hold.
///
/// Returns the cluster size and the highest-confidence member, or `None`
/// if no cluster has ≥ 2 members. The chosen representative is the
/// member with the *largest* content-token set (preferring the more
/// complete answer), tie-broken by confidence.
///
/// Greedy single-pass clustering — fine for the N ≤ ~8 worker case
/// MoA actually sees.
fn largest_agreeing_cluster<'a>(answers: &[&'a WorkerOutput]) -> Option<(usize, &'a WorkerOutput)> {
    let token_sets: Vec<_> = answers.iter().map(|o| content_tokens(&o.payload)).collect();
    let mut best: Option<(usize, &WorkerOutput)> = None;

    for (i, anchor_tokens) in token_sets.iter().enumerate() {
        // Build cluster anchored at i: include j if i's tokens ⊆ j's
        // tokens OR j's tokens ⊆ i's tokens.
        let mut members: Vec<&WorkerOutput> = vec![answers[i]];
        let mut member_token_counts: Vec<usize> = vec![anchor_tokens.len()];
        for (j, other_tokens) in token_sets.iter().enumerate() {
            if i == j {
                continue;
            }
            // Negation mismatch in either direction blocks clustering
            // even when subset would otherwise hold: "use grep" is a
            // subset of "do not use grep" by tokens, but they obviously
            // disagree.
            if has_negation_mismatch(anchor_tokens, other_tokens) {
                continue;
            }
            // An empty token set means the answer was all stopwords /
            // sub-3-char filler. That has no content signal — don't let
            // it cluster with anything, even though `{} ⊆ X` is
            // technically true for every X.
            if anchor_tokens.is_empty() || other_tokens.is_empty() {
                continue;
            }
            if other_tokens.is_subset(anchor_tokens) || anchor_tokens.is_subset(other_tokens) {
                members.push(answers[j]);
                member_token_counts.push(other_tokens.len());
            }
        }
        if members.len() < 2 {
            continue;
        }
        // Representative = member with the most content tokens (most
        // complete answer); tie-break on confidence.
        let representative = members
            .iter()
            .zip(member_token_counts.iter())
            .max_by(|(a, a_n), (b, b_n)| {
                a_n.cmp(b_n)
                    .then_with(|| a.confidence.total_cmp(&b.confidence))
            })
            .map(|(o, _)| *o)
            .unwrap();
        match best {
            Some((size, _)) if size >= members.len() => {}
            _ => best = Some((members.len(), representative)),
        }
    }
    best
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
    fn early_decision_requires_content_agreement() {
        // Two high-confidence answers that disagree on the answer noun.
        // The old code wrongly early-exited on the highest-confidence
        // one. The new subset rule sees `{paris}` and `{berlin}` as
        // distinct leftovers, so neither side is a subset of the other.
        let outputs = vec![
            make_output(OutputKind::Answer, 0.9, "The capital of France is Paris"),
            make_output(OutputKind::Answer, 0.8, "The capital of France is Berlin"),
        ];
        let res = try_early_decision(&outputs, 3, outputs.len(), false);
        assert!(
            res.is_none(),
            "disagreeing answers should not trigger early-exit, got {res:?}"
        );
    }

    #[test]
    fn early_decision_agrees_on_terse_vs_verbose() {
        // The most common real-world agreement pattern: one worker is
        // terse, another is verbose, both correct. Terse content tokens
        // ⊆ verbose content tokens → cluster.
        let outputs = vec![
            make_output(OutputKind::Answer, 0.7, "Paris"),
            make_output(OutputKind::Answer, 0.9, "Paris is the capital of France"),
        ];
        match try_early_decision(&outputs, 3, outputs.len(), false) {
            // Representative picks the most complete (most tokens) member.
            Some(Decision::Answer(text)) => {
                let lower = text.to_lowercase();
                assert!(lower.contains("paris"), "expected Paris, got {text:?}");
                assert!(
                    lower.contains("capital") || lower == "paris",
                    "expected verbose representative or single 'paris', got {text:?}"
                );
            }
            other => panic!("expected agreement on terse-vs-verbose, got {other:?}"),
        }
    }

    #[test]
    fn early_decision_majority_cluster_wins() {
        // Two agreeing answers + one outlier. Cluster of 2 is a majority
        // of 3 finished answers → early-exit fires.
        let outputs = vec![
            make_output(OutputKind::Answer, 0.9, "Paris"),
            make_output(OutputKind::Answer, 0.8, "Paris is the capital"),
            make_output(OutputKind::Answer, 0.6, "I think it's Lyon"),
        ];
        match try_early_decision(&outputs, 4, outputs.len(), false) {
            Some(Decision::Answer(text)) => assert!(
                text.to_lowercase().contains("paris"),
                "should pick from the agreeing cluster, got {text:?}"
            ),
            other => panic!("expected early Answer on majority cluster, got {other:?}"),
        }
    }

    #[test]
    fn early_decision_disagreement_with_shared_scaffolding_still_blocks() {
        // High token overlap from shared scaffolding ("the capital of
        // France is X") used to false-positive a similarity check.
        // With subset containment, each answer has a distinct leftover
        // (paris, berlin, madrid) so no cluster forms.
        let outputs = vec![
            make_output(OutputKind::Answer, 0.9, "The capital of France is Paris"),
            make_output(OutputKind::Answer, 0.9, "The capital of France is Berlin"),
            make_output(OutputKind::Answer, 0.5, "The capital of France is Madrid"),
        ];
        let res = try_early_decision(&outputs, 4, outputs.len(), false);
        assert!(
            res.is_none(),
            "three disagreeing answers should not early-exit, got {res:?}"
        );
    }

    #[test]
    fn early_decision_negation_blocks_agreement() {
        // The negation guard keeps "not" / "dont" etc. as content tokens,
        // so an answer with negation is not a subset of the affirmative
        // version even when all other tokens match.
        let outputs = vec![
            make_output(OutputKind::Answer, 0.9, "You should use grep"),
            make_output(OutputKind::Answer, 0.8, "You should not use grep"),
        ];
        let res = try_early_decision(&outputs, 3, outputs.len(), false);
        assert!(
            res.is_none(),
            "affirmative vs negated answer should not cluster, got {res:?}"
        );
    }

    #[test]
    fn early_decision_dont_blocks_agreement() {
        // Same idea with a contraction. After punctuation stripping
        // "don't" → "dont", which is in NEGATION_TOKENS.
        let outputs = vec![
            make_output(OutputKind::Answer, 0.9, "Do that"),
            make_output(OutputKind::Answer, 0.8, "Don't do that"),
        ];
        let res = try_early_decision(&outputs, 3, outputs.len(), false);
        assert!(
            res.is_none(),
            "affirmative vs negated should not cluster, got {res:?}"
        );
    }

    #[test]
    fn early_decision_numeric_answers_cluster() {
        // Numeric answers survive the length filter and cluster cleanly.
        let outputs = vec![
            make_output(OutputKind::Answer, 0.9, "42"),
            make_output(OutputKind::Answer, 0.8, "The answer is 42"),
        ];
        match try_early_decision(&outputs, 3, outputs.len(), false) {
            Some(Decision::Answer(text)) => assert!(text.contains("42")),
            other => panic!("expected numeric agreement, got {other:?}"),
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
