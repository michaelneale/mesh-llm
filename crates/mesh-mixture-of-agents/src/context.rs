//! Context packing — tailor what each worker sees.
//!
//! Full context enters the gateway, but workers get role-shaped slices of
//! the REAL context — the agent's actual system prompt, messages, and tool
//! definitions.  The gateway does not replace the agent's prompt with a
//! synthetic "you are a worker" envelope.  It augments with a short preamble
//! and varies the depth per role:
//!
//! - Fast:       system prompt + last user msg + tool names only
//! - Specialist: system prompt + last 4 msgs + tool summaries
//! - Strong:     system prompt + last 10 msgs + full tool schemas
//! - Reducer:    system prompt + worker outputs + full tool schemas

use crate::normalize::WorkerOutput;
use crate::session::Session;
use crate::worker::WorkerRole;
use serde_json::{json, Value};

/// Packed context ready to send to a worker.
pub struct PackedContext {
    pub messages: Vec<Value>,
    pub max_tokens: u32,
    /// Tool definitions to forward (if any).  `None` means don't send tools.
    pub tools: Option<Value>,
}

/// Build a context packet for a worker based on its role.
///
/// Each worker gets a slice of the real conversation — the agent's actual
/// system prompt and messages — not a synthetic replacement.  The depth of
/// the slice and tool detail varies by role.
pub fn pack_for_worker(session: &Session, role: WorkerRole, _has_tools: bool) -> PackedContext {
    match role {
        WorkerRole::Fast => pack_fast(session),
        WorkerRole::Specialist => pack_specialist(session),
        WorkerRole::Strong | WorkerRole::Generalist | WorkerRole::Reducer => pack_strong(session),
    }
}

// ── MoA preamble ─────────────────────────────────────────────────────
// A short addition to the system prompt.  Does NOT replace the agent's
// system prompt — it's prepended so the model still sees the original
// instructions.

const MOA_PREAMBLE: &str = "\
[Multiple models are analyzing this request in parallel. \
Respond with your best answer or tool call. Be direct.]";

/// Augment the agent's system prompt with the MoA preamble.
/// If there's no system prompt, create one with just the preamble.
fn augmented_system_prompt(session: &Session) -> String {
    match session.system_prompt() {
        Some(sp) => format!("{MOA_PREAMBLE}\n\n{sp}"),
        None => MOA_PREAMBLE.to_string(),
    }
}

/// Augmented system prompt with a compact tool catalogue appended.
fn system_with_tool_names(session: &Session) -> String {
    let mut prompt = augmented_system_prompt(session);
    let names = session.tool_names();
    if !names.is_empty() {
        prompt.push_str(&format!("\n\nAvailable tools: {}", names.join(", ")));
    }
    prompt
}

fn system_with_tool_summaries(session: &Session) -> String {
    let mut prompt = augmented_system_prompt(session);
    let summaries = session.tool_summaries();
    if !summaries.is_empty() {
        prompt.push_str("\n\nAvailable tools:");
        for s in &summaries {
            prompt.push_str(&format!("\n  - {s}"));
        }
    }
    prompt
}

// ── Fast worker ──────────────────────────────────────────────────────
// System prompt + last user message + tool names only.
// Smallest context, quickest to respond.

fn pack_fast(session: &Session) -> PackedContext {
    let system = system_with_tool_names(session);
    let user_text = session.last_user_text();

    // Per-request sessions: the caller owns the multi-turn loop and
    // sends the full history each request. Continuation context lives
    // in `session.messages()`; this path intentionally trims to just
    // the last user message to keep the fast worker's context small.
    PackedContext {
        messages: vec![
            json!({"role": "system", "content": system}),
            json!({"role": "user", "content": user_text}),
        ],
        max_tokens: 256,
        tools: None, // Fast worker doesn't get tool schemas — just names
    }
}

// ── Specialist worker ────────────────────────────────────────────────
// System prompt + last 4 messages + tool name+description summaries.

fn pack_specialist(session: &Session) -> PackedContext {
    let system = system_with_tool_summaries(session);

    let mut messages = vec![json!({"role": "system", "content": system})];

    // Recent messages — skip system (already included), skip raw tool results
    // (they'd confuse models that don't have the tool_call context)
    let recent = session.recent_messages(4);
    for msg in &recent {
        let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("");
        if role == "user" || (role == "assistant" && msg.get("tool_calls").is_none()) {
            messages.push(msg.clone());
        }
    }

    // Ensure the last message is the current user turn
    let user_text = session.last_user_text();
    if messages
        .last()
        .and_then(|m| m.get("content").and_then(|c| c.as_str()))
        != Some(&user_text)
    {
        messages.push(json!({"role": "user", "content": user_text}));
    }

    PackedContext {
        messages,
        max_tokens: 512,
        tools: session.tools().cloned(), // Specialist gets full schemas for native tool_calls
    }
}

// ── Strong worker ────────────────────────────────────────────────────
// System prompt + last 10 messages + full tool schemas forwarded natively.
// This worker gets the deepest context and the actual tool definitions so
// it can produce native tool_calls if the backend supports it.

fn pack_strong(session: &Session) -> PackedContext {
    let system = augmented_system_prompt(session);

    let mut messages = vec![json!({"role": "system", "content": system})];

    // Deep recent history — include tool result messages too since this
    // worker gets full tool schemas and can understand the context
    let recent = session.recent_messages(10);
    for msg in &recent {
        let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("");
        if role != "system" && !role.is_empty() {
            messages.push(msg.clone());
        }
    }

    let user_text = session.last_user_text();
    if messages
        .last()
        .and_then(|m| m.get("content").and_then(|c| c.as_str()))
        != Some(&user_text)
    {
        messages.push(json!({"role": "user", "content": user_text}));
    }

    // Forward the real tool schemas — the strong worker can produce native
    // tool_calls through the OpenAI API
    let tools = session.tools().cloned();

    PackedContext {
        messages,
        max_tokens: 1024,
        tools,
    }
}

// ── Reducer / conflict resolution ────────────────────────────────────

/// Build context for the reducer when arbitration is needed.
///
/// The reducer gets: agent's system prompt + worker outputs + full tool
/// schemas.  It sees what the workers proposed and makes the final call.
pub fn pack_for_reducer(
    session: &Session,
    outputs: &[WorkerOutput],
    reason: &str,
    _has_tools: bool,
) -> (Vec<Value>, Option<Value>) {
    let user_text = session.last_user_text();

    let mut system_parts = vec![
        augmented_system_prompt(session),
        String::new(),
        format!("Multiple models analyzed this request and disagreed. Reason: {reason}"),
        "Review their outputs below and produce ONE final response — either a direct answer \
         or a tool call. Be concise."
            .to_string(),
    ];

    // Worker outputs
    system_parts.push(String::new());
    system_parts.push("## Worker outputs".to_string());
    for (i, output) in outputs.iter().enumerate() {
        system_parts.push(format!("\n[Worker {} — {}]:", i + 1, output.model,));
        let payload = if output.payload.len() > 500 {
            format!("{}...", crate::worker::truncate_chars(&output.payload, 497))
        } else {
            output.payload.clone()
        };
        system_parts.push(payload);
        if let Some(ref tool) = output.tool_name {
            system_parts.push(format!("  → Proposed tool: {tool}"));
            if let Some(ref args) = output.tool_arguments {
                system_parts.push(format!("  → Arguments: {args}"));
            }
        }
    }

    let tools = session.tools().cloned();

    (
        vec![
            json!({"role": "system", "content": system_parts.join("\n")}),
            json!({"role": "user", "content": user_text}),
        ],
        tools,
    )
}

/// Build context for a tool-result turn (reducer only, not full fan-out).
///
/// The reducer gets: agent's system prompt + the original conversation
/// including assistant tool_call messages and the corresponding tool result
/// messages, plus full tool schemas so it can propose the next call.
///
/// We forward the raw message sequence rather than summarizing, because
/// the reducer model needs to see the tool_call → tool result pairs in
/// their native OpenAI format to reason about what happened and decide
/// what to do next.
pub fn pack_for_tool_result_turn(
    session: &Session,
    _has_tools: bool,
) -> (Vec<Value>, Option<Value>) {
    let system = augmented_system_prompt(session);

    let mut messages = vec![json!({"role": "system", "content": system})];

    // Forward the tail of the conversation that includes tool_call + tool
    // result messages. Walk backwards to find the assistant message that
    // proposed the tool call(s), then include everything from there forward.
    let all = session.all_messages();
    let mut start_idx = all.len().saturating_sub(10); // default: last 10

    // Try to find the assistant tool_call message that triggered these results
    for (i, msg) in all.iter().enumerate().rev() {
        let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("");
        if role == "assistant" && msg.get("tool_calls").is_some() {
            // Include one user message before the tool_call for context
            start_idx = i.saturating_sub(1);
            break;
        }
    }

    for msg in &all[start_idx..] {
        let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("");
        if role != "system" && !role.is_empty() {
            messages.push(msg.clone());
        }
    }

    let tools = session.tools().cloned();

    (messages, tools)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::normalize::{OutputKind, WorkerOutput};
    use serde_json::json;

    fn user_msg(text: &str) -> Value {
        json!({"role": "user", "content": text})
    }
    fn system_msg(text: &str) -> Value {
        json!({"role": "system", "content": text})
    }
    fn assistant_msg(text: &str) -> Value {
        json!({"role": "assistant", "content": text})
    }
    fn tools_two() -> Value {
        json!([
            {"type": "function", "function": {
                "name": "read_file",
                "description": "Read a file from disk",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}
            }},
            {"type": "function", "function": {
                "name": "web_search",
                "description": "Search the web",
                "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}
            }},
        ])
    }

    fn session_with(messages: &[Value], tools: Option<Value>) -> Session {
        let mut s = Session::new();
        s.ingest(messages, &tools);
        s
    }

    /// Helper: extract the system message content from a packed message vec.
    fn system_text(messages: &[Value]) -> String {
        messages
            .iter()
            .find(|m| m.get("role").and_then(|r| r.as_str()) == Some("system"))
            .and_then(|m| m.get("content").and_then(|c| c.as_str()))
            .unwrap_or("")
            .to_string()
    }

    // ── pack_for_worker: shape contract per role ─────────────────────

    #[test]
    fn fast_worker_has_system_user_only_no_tools() {
        let s = session_with(
            &[
                system_msg("You are a helpful assistant."),
                user_msg("first"),
                assistant_msg("first reply"),
                user_msg("second"),
            ],
            Some(tools_two()),
        );
        let packed = pack_for_worker(&s, WorkerRole::Fast, true);

        assert_eq!(packed.max_tokens, 256, "fast worker token budget");
        assert!(
            packed.tools.is_none(),
            "fast worker must not receive tool schemas"
        );
        assert_eq!(packed.messages.len(), 2, "fast = system + last user only");
        assert_eq!(
            packed.messages[0].get("role").and_then(|r| r.as_str()),
            Some("system"),
        );
        assert_eq!(
            packed.messages[1].get("role").and_then(|r| r.as_str()),
            Some("user"),
        );
        assert_eq!(
            packed.messages[1].get("content").and_then(|c| c.as_str()),
            Some("second"),
            "fast worker sees only the LAST user message",
        );

        // Tool *names* appear in system prompt; full schemas do not.
        let sys = system_text(&packed.messages);
        assert!(
            sys.contains("read_file"),
            "tool names present in system: {sys}"
        );
        assert!(
            sys.contains("web_search"),
            "tool names present in system: {sys}"
        );
        assert!(
            !sys.contains("\"parameters\""),
            "fast worker system must not contain JSON Schema fragments: {sys}",
        );
    }

    #[test]
    fn specialist_worker_has_summaries_and_native_tools() {
        let s = session_with(
            &[
                system_msg("Agent SP."),
                user_msg("m1"),
                assistant_msg("r1"),
                user_msg("m2"),
                assistant_msg("r2"),
                user_msg("m3"),
            ],
            Some(tools_two()),
        );
        let packed = pack_for_worker(&s, WorkerRole::Specialist, true);

        assert_eq!(packed.max_tokens, 512, "specialist token budget");
        assert!(
            packed.tools.is_some(),
            "specialist must receive full native tool schemas",
        );
        // Tool *summaries* (name + description) must be in the system prompt.
        let sys = system_text(&packed.messages);
        assert!(sys.contains("read_file"));
        assert!(
            sys.contains("Read a file"),
            "specialist system should include tool descriptions: {sys}",
        );

        // Last message is the latest user turn ("m3").
        let last = packed.messages.last().unwrap();
        assert_eq!(last.get("role").and_then(|r| r.as_str()), Some("user"));
        assert_eq!(last.get("content").and_then(|c| c.as_str()), Some("m3"));
    }

    #[test]
    fn strong_worker_has_deep_history_and_native_tools() {
        // Build a session with many turns so we can verify depth.
        let mut msgs = vec![system_msg("Agent ST.")];
        for i in 0..8 {
            msgs.push(user_msg(&format!("u{i}")));
            msgs.push(assistant_msg(&format!("a{i}")));
        }
        msgs.push(user_msg("final"));
        let s = session_with(&msgs, Some(tools_two()));

        let packed = pack_for_worker(&s, WorkerRole::Strong, true);

        assert_eq!(packed.max_tokens, 1024, "strong token budget");
        assert!(
            packed.tools.is_some(),
            "strong must receive full native tool schemas",
        );
        // Strong gets up to last 10 messages on top of the system prompt,
        // so it should see deeper history than the specialist's 4-message window.
        assert!(
            packed.messages.len() >= 6,
            "strong worker should retain deep history, got {} messages",
            packed.messages.len(),
        );
        let last = packed.messages.last().unwrap();
        assert_eq!(last.get("content").and_then(|c| c.as_str()), Some("final"));
    }

    #[test]
    fn generalist_and_reducer_roles_use_strong_shape() {
        let s = session_with(&[system_msg("Agent."), user_msg("hi")], Some(tools_two()));
        let g = pack_for_worker(&s, WorkerRole::Generalist, true);
        let r = pack_for_worker(&s, WorkerRole::Reducer, true);
        assert_eq!(g.max_tokens, 1024);
        assert_eq!(r.max_tokens, 1024);
        assert!(g.tools.is_some());
        assert!(r.tools.is_some());
    }

    // ── MoA preamble: augment, don't replace ─────────────────────────

    #[test]
    fn preamble_augments_existing_system_prompt() {
        let s = session_with(
            &[
                system_msg("CUSTOM_AGENT_INSTRUCTIONS_MARKER"),
                user_msg("hi"),
            ],
            None,
        );
        let packed = pack_for_worker(&s, WorkerRole::Strong, false);
        let sys = system_text(&packed.messages);
        assert!(
            sys.contains("CUSTOM_AGENT_INSTRUCTIONS_MARKER"),
            "agent's original system prompt must survive: {sys}",
        );
        assert!(
            sys.contains("Multiple models"),
            "MoA preamble must be present: {sys}",
        );
    }

    #[test]
    fn preamble_only_when_no_system_prompt() {
        let s = session_with(&[user_msg("hi")], None);
        let packed = pack_for_worker(&s, WorkerRole::Strong, false);
        let sys = system_text(&packed.messages);
        assert!(
            !sys.is_empty(),
            "should synthesize a system prompt from preamble"
        );
        assert!(sys.contains("Multiple models"));
    }

    // ── pack_for_reducer: includes reason + worker outputs ───────────

    fn worker_out(model: &str, payload: &str) -> WorkerOutput {
        WorkerOutput {
            kind: OutputKind::Answer,
            confidence: 0.6,
            tool_name: None,
            tool_arguments: None,
            payload: payload.to_string(),
            model: model.to_string(),
            role: WorkerRole::Strong,
            elapsed_ms: 0,
        }
    }

    #[test]
    fn reducer_context_includes_reason_and_worker_payloads() {
        let s = session_with(
            &[
                system_msg("Agent R."),
                user_msg("which is bigger, 7^3 or 350?"),
            ],
            Some(tools_two()),
        );
        let outputs = vec![
            worker_out("alpha", "It's 7^3 = 343, smaller than 350."),
            worker_out("beta", "350 is bigger."),
        ];
        let (messages, tools) = pack_for_reducer(&s, &outputs, "tie between answers", true);

        let sys = system_text(&messages);
        assert!(
            sys.contains("tie between answers"),
            "reason must appear in reducer system: {sys}",
        );
        assert!(sys.contains("alpha"), "worker model labels must appear");
        assert!(sys.contains("beta"));
        assert!(sys.contains("7^3 = 343"));
        assert!(sys.contains("350 is bigger"));
        assert!(
            tools.is_some(),
            "reducer should still have native tool schemas",
        );

        // Last message should be the user's actual query.
        let last = messages.last().unwrap();
        assert_eq!(last.get("role").and_then(|r| r.as_str()), Some("user"));
        assert_eq!(
            last.get("content").and_then(|c| c.as_str()),
            Some("which is bigger, 7^3 or 350?"),
        );
    }

    #[test]
    fn reducer_truncates_long_worker_payloads() {
        let s = session_with(&[user_msg("go")], None);
        let big = "x".repeat(2000);
        let outputs = vec![worker_out("alpha", &big)];

        let (messages, _tools) = pack_for_reducer(&s, &outputs, "conflict", false);
        let sys = system_text(&messages);

        // Long payloads must be truncated (cap is ~500 chars + ellipsis).
        // The full 2000-char string must NOT appear verbatim.
        assert!(
            !sys.contains(&big),
            "reducer must truncate long worker payloads to keep context bounded",
        );
        assert!(
            sys.contains("..."),
            "truncated payloads should be marked with an ellipsis",
        );
    }
}
