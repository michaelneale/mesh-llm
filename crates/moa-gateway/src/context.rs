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

    // Include running summary for multi-turn context
    let mut system_full = system;
    if session.turn_count() > 1 {
        let summary = session.running_summary();
        if !summary.is_empty() {
            system_full.push_str(&format!("\n\nConversation so far:\n{summary}"));
        }
    }

    PackedContext {
        messages: vec![
            json!({"role": "system", "content": system_full}),
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
        tools: None, // Specialist gets summaries in system prompt, not full schemas
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
            format!("{}...", &output.payload[..497])
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
/// The reducer gets: agent's system prompt + tool result + enough context
/// to produce a final answer or propose the next tool call.
pub fn pack_for_tool_result_turn(
    session: &Session,
    _has_tools: bool,
) -> (Vec<Value>, Option<Value>) {
    let mut system_parts = vec![augmented_system_prompt(session)];

    // Include recent tool results
    let tool_results = session.recent_tool_results();
    if !tool_results.is_empty() {
        system_parts.push(String::new());
        system_parts.push("Tool results:".to_string());
        for (name, result) in &tool_results {
            let short = if result.len() > 1000 {
                format!("{}...", &result[..997])
            } else {
                result.clone()
            };
            system_parts.push(format!("{name}() → {short}"));
        }
        system_parts.push(String::new());
        system_parts.push(
            "Use the tool results above to answer the user's question, or call another tool \
             if more work is needed."
                .to_string(),
        );
    }

    let user_text = session.last_user_text();
    let tools = session.tools().cloned();

    (
        vec![
            json!({"role": "system", "content": system_parts.join("\n")}),
            json!({"role": "user", "content": user_text}),
        ],
        tools,
    )
}
