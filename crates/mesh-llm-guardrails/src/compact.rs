//! Tiered in-place compaction. Ported from forge
//! `src/forge/context/strategies.py::TieredCompact` and the threshold
//! helpers from `src/forge/context/manager.py` (v0.6.0).
//!
//! ## Approximation vs. forge
//!
//! Forge tags every message with a [`MessageType`](crate::types::MessageType)
//! at construction time and uses that tag plus `step_index` to decide
//! what to cut. Mesh-llm's [`ChatMessage`] doesn't carry either — the
//! request is reconstructed from the wire each turn. So we classify
//! messages here from `role` + shape:
//!
//! - `role == "system"`                    → `SystemPrompt`
//! - first `role == "user"` (index 0 or 1) → `UserInput`
//! - `role == "user"` with our
//!   `extra["mesh_guardrail_nudge"] == "..."` marker → matching nudge type
//! - `role == "assistant"` with `tool_calls` → `ToolCall`
//! - `role == "assistant"` with text only   → `TextResponse`
//! - `role == "tool"`                       → `ToolResult`
//!
//! Iteration boundaries (forge's `step_index`) are approximated:
//! every `assistant{tool_calls}` opens a new iteration; the following
//! `tool` messages belong to that iteration.
//!
//! ## Phases (same as forge)
//!
//! 1. Drop step / retry / prereq nudges; truncate `tool` results to
//!    [`TRUNCATE_CHARS`] chars.
//! 2. Phase 1 + drop tool results entirely.
//! 3. Phase 2 + drop reasoning / text-response assistant turns. Only
//!    `assistant{tool_calls}` skeletons remain in the cut zone.
//!
//! Phase 3 is the known weak link — dropping reasoning often makes the
//! model "forget what it's doing". The decorator gates phase 3 behind
//! [`CompactConfig::enable_phase3`] (default `false`).

use openai_frontend::chat::{ChatMessage, MessageContent};
use serde_json::Value;

use crate::types::MessageType;

pub const NUDGE_MARKER_KEY: &str = "mesh_guardrail_nudge";

#[derive(Debug, Clone)]
pub struct CompactConfig {
    pub keep_recent: usize,
    /// Phase 1 trigger as a fraction of `n_ctx`.
    pub phase1_threshold: f32,
    /// Phase 2 trigger as a fraction of `n_ctx`.
    pub phase2_threshold: f32,
    /// Phase 3 trigger as a fraction of `n_ctx`. Only consulted if
    /// `enable_phase3 == true`.
    pub phase3_threshold: f32,
    /// Phase 3 drops reasoning. Off by default — see module doc.
    pub enable_phase3: bool,
    pub truncate_chars: usize,
    /// Thresholds at which a one-shot context warning is injected.
    /// Sorted ascending. Each fires at most once per request (we're
    /// stateless across requests, so this just means "fire the
    /// highest crossed threshold").
    pub warning_thresholds: Vec<f32>,
}

impl Default for CompactConfig {
    fn default() -> Self {
        Self {
            keep_recent: 2,
            phase1_threshold: 0.75,
            phase2_threshold: 0.75,
            phase3_threshold: 0.75,
            enable_phase3: false,
            truncate_chars: 200,
            warning_thresholds: vec![0.5, 0.65, 0.8],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompactOutcome {
    pub phase_reached: u8,
    pub messages_before: usize,
    pub messages_after: usize,
    pub tokens_before: usize,
    pub tokens_after: usize,
}

/// char/4 heuristic, matching forge.
pub fn estimate_tokens(messages: &[ChatMessage]) -> usize {
    messages.iter().map(message_text_len).sum::<usize>() / 4
}

fn message_text_len(m: &ChatMessage) -> usize {
    match m.content.as_ref() {
        Some(MessageContent::Text(s)) => s.len(),
        Some(MessageContent::Parts(parts)) => parts
            .iter()
            .filter_map(|p| p.text.as_deref())
            .map(str::len)
            .sum(),
        Some(MessageContent::Other(v)) => v.to_string().len(),
        None => 0,
    }
}

/// Classify a message at position `idx` in `messages`. Position is
/// needed because the first `user` message is treated as the original
/// user input and protected.
fn classify(messages: &[ChatMessage], idx: usize) -> MessageType {
    let m = &messages[idx];
    if m.role == "system" {
        return MessageType::SystemPrompt;
    }
    if m.role == "tool" {
        return MessageType::ToolResult;
    }
    if m.role == "assistant" {
        if m.extra.contains_key("tool_calls") {
            return MessageType::ToolCall;
        }
        return MessageType::TextResponse;
    }
    if m.role == "user" {
        // First user message at or before idx == 1 is the original
        // input. Beyond that, check for an explicit nudge marker we
        // injected ourselves — otherwise treat as user input (don't
        // touch arbitrary user messages).
        let first_user_idx = messages.iter().position(|m| m.role == "user");
        if Some(idx) == first_user_idx {
            return MessageType::UserInput;
        }
        if let Some(kind) = m.extra.get(NUDGE_MARKER_KEY).and_then(Value::as_str) {
            return match kind {
                "step" => MessageType::StepNudge,
                "prerequisite" => MessageType::PrerequisiteNudge,
                "retry" => MessageType::RetryNudge,
                "context_warning" => MessageType::ContextWarning,
                _ => MessageType::UserInput,
            };
        }
        return MessageType::UserInput;
    }
    // Unknown role — preserve.
    MessageType::UserInput
}

/// Walk the message list and assign an iteration index to each
/// message. An iteration opens at each `assistant{tool_calls}` and
/// the following `tool` messages share that iteration's index.
/// Messages before the first iteration get `None`.
fn iteration_indices(messages: &[ChatMessage]) -> Vec<Option<usize>> {
    let mut out = Vec::with_capacity(messages.len());
    let mut current: Option<usize> = None;
    let mut counter: usize = 0;
    for m in messages {
        if m.role == "assistant" && m.extra.contains_key("tool_calls") {
            counter += 1;
            current = Some(counter);
            out.push(current);
        } else if m.role == "tool" {
            out.push(current);
        } else {
            // System / user / bare-text assistant: not part of an
            // iteration. Anything *after* the first iteration that
            // isn't a tool turn opens a fresh boundary on the next
            // iteration.
            out.push(None);
        }
    }
    out
}

/// Find the boundary index: everything **before** this index in
/// `messages` is eligible for compaction. Mirrors forge's
/// `_find_eligible_end`.
fn find_eligible_end(messages: &[ChatMessage], keep_recent: usize) -> usize {
    if messages.len() <= 2 {
        return messages.len();
    }
    let iters = iteration_indices(messages);
    // Distinct iteration ids from index 2 onward.
    let mut seen: Vec<usize> = Vec::new();
    for it in iters.iter().skip(2).flatten() {
        if seen.last() != Some(it) {
            seen.push(*it);
        }
    }
    if seen.len() <= keep_recent {
        return 2;
    }
    let cutoff = seen[seen.len() - keep_recent];
    for (i, it) in iters.iter().enumerate().skip(2) {
        if let Some(idx) = it {
            if *idx >= cutoff {
                return i;
            }
        }
    }
    messages.len()
}

/// Drop the message's textual content while keeping its role,
/// tool_calls, and other metadata. Returns true if anything was
/// changed.
fn truncate_text(m: &mut ChatMessage, limit: usize) -> bool {
    let Some(content) = m.content.as_mut() else {
        return false;
    };
    match content {
        MessageContent::Text(s) if s.len() > limit => {
            let mut idx = limit;
            while idx > 0 && !s.is_char_boundary(idx) {
                idx -= 1;
            }
            let removed = s.len() - idx;
            s.truncate(idx);
            s.push_str(&format!("\n[Truncated — {removed} chars removed]"));
            true
        }
        _ => false,
    }
}

/// Apply tiered compaction in place. The first two messages
/// (system / user_input — whichever come first) are always preserved.
pub fn compact(
    messages: &mut Vec<ChatMessage>,
    n_ctx: u32,
    config: &CompactConfig,
) -> CompactOutcome {
    let messages_before = messages.len();
    let tokens_before = estimate_tokens(messages);
    let budget = n_ctx as usize;

    let t1 = ((budget as f32) * config.phase1_threshold) as usize;
    let t2 = ((budget as f32) * config.phase2_threshold) as usize;
    let t3 = ((budget as f32) * config.phase3_threshold) as usize;

    if tokens_before < t1 {
        return CompactOutcome {
            phase_reached: 0,
            messages_before,
            messages_after: messages_before,
            tokens_before,
            tokens_after: tokens_before,
        };
    }

    let eligible_end = find_eligible_end(messages, config.keep_recent);

    // ---------- Phase 1 ----------
    let phase1: Vec<ChatMessage> = messages
        .iter()
        .enumerate()
        .filter_map(|(i, m)| {
            if i < 2 || i >= eligible_end {
                return Some(m.clone());
            }
            let kind = classify(messages, i);
            match kind {
                MessageType::StepNudge
                | MessageType::PrerequisiteNudge
                | MessageType::RetryNudge => None,
                MessageType::ToolResult => {
                    let mut clone = m.clone();
                    truncate_text(&mut clone, config.truncate_chars);
                    Some(clone)
                }
                _ => Some(m.clone()),
            }
        })
        .collect();

    if estimate_tokens(&phase1) < t2 {
        let tokens_after = estimate_tokens(&phase1);
        let messages_after = phase1.len();
        *messages = phase1;
        return CompactOutcome {
            phase_reached: 1,
            messages_before,
            messages_after,
            tokens_before,
            tokens_after,
        };
    }

    // ---------- Phase 2 ----------
    let phase2: Vec<ChatMessage> = messages
        .iter()
        .enumerate()
        .filter_map(|(i, m)| {
            if i < 2 || i >= eligible_end {
                return Some(m.clone());
            }
            let kind = classify(messages, i);
            match kind {
                MessageType::StepNudge
                | MessageType::PrerequisiteNudge
                | MessageType::RetryNudge
                | MessageType::ToolResult => None,
                _ => Some(m.clone()),
            }
        })
        .collect();

    if !config.enable_phase3 || estimate_tokens(&phase2) < t3 {
        let tokens_after = estimate_tokens(&phase2);
        let messages_after = phase2.len();
        *messages = phase2;
        return CompactOutcome {
            phase_reached: 2,
            messages_before,
            messages_after,
            tokens_before,
            tokens_after,
        };
    }

    // ---------- Phase 3 ----------
    let phase3: Vec<ChatMessage> = messages
        .iter()
        .enumerate()
        .filter_map(|(i, m)| {
            if i < 2 || i >= eligible_end {
                return Some(m.clone());
            }
            let kind = classify(messages, i);
            match kind {
                MessageType::StepNudge
                | MessageType::PrerequisiteNudge
                | MessageType::RetryNudge
                | MessageType::ToolResult
                | MessageType::Reasoning
                | MessageType::TextResponse => None,
                _ => Some(m.clone()),
            }
        })
        .collect();

    let tokens_after = estimate_tokens(&phase3);
    let messages_after = phase3.len();
    *messages = phase3;
    CompactOutcome {
        phase_reached: 3,
        messages_before,
        messages_after,
        tokens_before,
        tokens_after,
    }
}

/// Default escalating context warning. Returns `None` if the request
/// has not crossed any configured threshold.
pub fn threshold_warning(tokens: usize, n_ctx: u32, thresholds: &[f32]) -> Option<String> {
    if n_ctx == 0 || thresholds.is_empty() {
        return None;
    }
    let budget = n_ctx as usize;
    let pct = (tokens as f32) / (budget as f32);
    // Find the *highest* threshold that has been crossed.
    let mut highest: Option<f32> = None;
    for t in thresholds {
        if pct >= *t {
            highest = Some(*t);
        }
    }
    let crossed = highest?;
    Some(format_warning(tokens, budget, pct, crossed))
}

fn format_warning(tokens: usize, budget: usize, pct: f32, crossed: f32) -> String {
    if crossed >= 0.80 {
        format!(
            "[Context usage: {pct:.0}% ({tokens} / {budget} tokens). \
             Context is nearly full. Older tool results and reasoning will be \
             compacted soon — key information may be lost. Summarize critical \
             findings now and prioritize completing the current task.]",
            pct = pct * 100.0,
        )
    } else if crossed >= 0.65 {
        format!(
            "[Context usage: {pct:.0}% ({tokens} / {budget} tokens). \
             Context is filling up. When compaction triggers, older tool results \
             and reasoning will be condensed. Be concise in your responses and \
             front-load important information.]",
            pct = pct * 100.0,
        )
    } else {
        format!(
            "[Context usage: {pct:.0}% ({tokens} / {budget} tokens). \
             Be mindful of context usage.]",
            pct = pct * 100.0,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Map};

    fn msg(role: &str, content: &str) -> ChatMessage {
        ChatMessage {
            role: role.into(),
            content: Some(MessageContent::Text(content.into())),
            extra: Map::new().into_iter().collect(),
        }
    }

    fn assistant_with_tool_calls(text: &str) -> ChatMessage {
        let mut m = msg("assistant", text);
        m.extra.insert(
            "tool_calls".into(),
            json!([{"id": "1", "function": {"name": "f"}}]),
        );
        m
    }

    fn tool_result(content: &str) -> ChatMessage {
        msg("tool", content)
    }

    #[test]
    fn no_compaction_below_threshold() {
        let mut msgs = vec![msg("system", "you are helpful"), msg("user", "hello")];
        let outcome = compact(&mut msgs, 1024, &CompactConfig::default());
        assert_eq!(outcome.phase_reached, 0);
        assert_eq!(msgs.len(), 2);
    }

    #[test]
    fn phase1_drops_nudges_and_truncates_tool_results() {
        let big_result = "x".repeat(2000);
        let mut msgs = vec![
            msg("system", "you are helpful"),
            msg("user", "do work"),
            assistant_with_tool_calls(""),
            tool_result(&big_result),
            assistant_with_tool_calls(""),
            tool_result(&big_result),
            assistant_with_tool_calls(""),
            tool_result(&big_result),
        ];
        // Inject a nudge in the cut zone.
        let mut nudge = msg("user", "try again");
        nudge.extra.insert(NUDGE_MARKER_KEY.into(), json!("retry"));
        msgs.insert(4, nudge);

        let cfg = CompactConfig {
            keep_recent: 1,
            phase1_threshold: 0.05,
            phase2_threshold: 0.99,
            ..Default::default()
        };
        let outcome = compact(&mut msgs, 1024, &cfg);
        assert_eq!(outcome.phase_reached, 1);
        // Nudge was dropped.
        assert!(!msgs.iter().any(|m| m
            .extra
            .get(NUDGE_MARKER_KEY)
            .map(|v| v == "retry")
            .unwrap_or(false)));
        // Earlier tool results got truncated; the most-recent one is preserved.
        let truncated_count = msgs
            .iter()
            .filter(|m| {
                m.role == "tool"
                    && match m.content.as_ref() {
                        Some(MessageContent::Text(s)) => s.contains("[Truncated"),
                        _ => false,
                    }
            })
            .count();
        assert!(truncated_count >= 1);
    }

    #[test]
    fn phase3_disabled_stops_at_phase2() {
        let big = "y".repeat(5000);
        let mut msgs = vec![msg("system", "you are helpful"), msg("user", "do work")];
        for _ in 0..6 {
            msgs.push(assistant_with_tool_calls(&big));
            msgs.push(tool_result(&big));
        }
        let cfg = CompactConfig {
            keep_recent: 1,
            phase1_threshold: 0.01,
            phase2_threshold: 0.01,
            phase3_threshold: 0.01,
            enable_phase3: false,
            ..Default::default()
        };
        let outcome = compact(&mut msgs, 256, &cfg);
        assert!(
            outcome.phase_reached <= 2,
            "phase {} should not exceed 2 when disabled",
            outcome.phase_reached
        );
    }

    #[test]
    fn threshold_warning_picks_highest_crossed() {
        let cfg_thresholds = vec![0.5, 0.65, 0.8];
        // 90% — fires the 0.8 warning.
        let w = threshold_warning(90, 100, &cfg_thresholds).unwrap();
        assert!(w.contains("nearly full"));
        // 70% — fires the 0.65 warning.
        let w = threshold_warning(70, 100, &cfg_thresholds).unwrap();
        assert!(w.contains("filling up"));
        // 30% — no warning.
        assert!(threshold_warning(30, 100, &cfg_thresholds).is_none());
    }
}
