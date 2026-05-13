//! Context packing — tailor what each worker sees.
//!
//! Full context enters the gateway, but workers get role-shaped packets.
//! A fast model gets the current task + tool names.
//! A specialist gets recent history + compact tool descriptions.
//! The reducer gets worker outputs + full tool schemas + original task.

use crate::normalize::WorkerOutput;
use crate::session::Session;
use crate::worker::WorkerRole;
use serde_json::{json, Value};

/// Packed context ready to send to a worker.
pub struct PackedContext {
    pub messages: Vec<Value>,
    pub max_tokens: u32,
}

/// Build a context packet for a worker based on its role.
pub fn pack_for_worker(session: &Session, role: WorkerRole, has_tools: bool) -> PackedContext {
    // When tools are present (agentic use via Goose/pi/etc.), pass through
    // the original messages faithfully — the client's system prompt already
    // has tool instructions the model needs to follow.  Wrapping them in a
    // MoA envelope confuses models.
    if has_tools {
        return pack_passthrough(session, role);
    }

    match role {
        WorkerRole::Fast => pack_fast(session),
        WorkerRole::Specialist => pack_specialist(session),
        WorkerRole::Strong => pack_strong(session),
        WorkerRole::Generalist => pack_strong(session),
        WorkerRole::Reducer => pack_strong(session),
    }
}

/// Passthrough mode: forward the original messages to the worker as-is.
/// Used when the client (Goose, pi, etc.) has already set up tools and
/// system prompts — we don't want to interfere.
fn pack_passthrough(session: &Session, role: WorkerRole) -> PackedContext {
    let messages = session.all_messages();
    let max_tokens = match role {
        WorkerRole::Fast => 512,
        WorkerRole::Specialist => 1024,
        WorkerRole::Strong | WorkerRole::Generalist | WorkerRole::Reducer => 2048,
    };
    PackedContext {
        messages,
        max_tokens,
    }
}

/// Fast worker: current task only, tool names, short max_tokens.
fn pack_fast(session: &Session) -> PackedContext {
    let user_text = session.last_user_text();
    let mut system_parts = vec![
        "You are a fast analysis worker in a multi-model ensemble.".to_string(),
        "Given the task below, produce ONLY the structured response below with NO explanation, \
         reasoning, or preamble. Start your response directly with 'kind:'."
            .to_string(),
        String::new(),
        "kind: answer | tool_proposal".to_string(),
        "confidence: 0.0-1.0".to_string(),
        "payload: your response text".to_string(),
        String::new(),
        "IMPORTANT: Output ONLY the fields above. No thinking, no explanation, \
         no markdown. Just the key: value lines starting with 'kind:'."
            .to_string(),
    ];

    // Add running summary if there's history
    if session.turn_count() > 1 {
        let summary = session.running_summary();
        if !summary.is_empty() {
            system_parts.push(String::new());
            system_parts.push(format!("Session context:\n{summary}"));
        }
    }

    PackedContext {
        messages: vec![
            json!({"role": "system", "content": system_parts.join("\n")}),
            json!({"role": "user", "content": user_text}),
        ],
        max_tokens: 256,
    }
}

/// Specialist worker: recent history, compact tool descriptions.
fn pack_specialist(session: &Session) -> PackedContext {
    let user_text = session.last_user_text();
    let mut system_parts = vec![
        "You are a specialist worker in a multi-model ensemble.".to_string(),
        "Analyze the task and produce ONLY the structured response below. \
         Start your response directly with 'kind:' — no preamble."
            .to_string(),
        String::new(),
        "kind: answer | tool_proposal".to_string(),
        "confidence: 0.0-1.0".to_string(),
        "payload: your detailed response".to_string(),
        String::new(),
        "Output ONLY the fields above. No thinking, no explanation, no markdown.".to_string(),
    ];

    // Include running summary for multi-turn context
    if session.turn_count() > 1 {
        let summary = session.running_summary();
        if !summary.is_empty() {
            system_parts.push(String::new());
            system_parts.push(format!("Session context:\n{summary}"));
        }
    }

    let mut messages = vec![json!({"role": "system", "content": system_parts.join("\n")})];

    let recent = session.recent_messages(4);
    for msg in &recent {
        let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("");
        if role == "user" || (role == "assistant" && msg.get("tool_calls").is_none()) {
            messages.push(msg.clone());
        }
    }

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
    }
}

/// Strong worker: more context, full tool descriptions where possible.
fn pack_strong(session: &Session) -> PackedContext {
    let user_text = session.last_user_text();
    let mut system_parts = vec![
        "You are a strong reasoning worker in a multi-model ensemble.".to_string(),
        "Analyze the task thoroughly and produce ONLY the structured response below. \
         Start your response directly with 'kind:' — no preamble."
            .to_string(),
        String::new(),
        "kind: answer | tool_proposal".to_string(),
        "confidence: 0.0-1.0".to_string(),
        "payload: your thorough response".to_string(),
        String::new(),
        "Output ONLY the fields above. No thinking, no explanation, no markdown.".to_string(),
    ];

    // Include system prompt if present
    if let Some(sp) = session.system_prompt() {
        system_parts.push(String::new());
        system_parts.push(format!("Original system prompt: {sp}"));
    }

    let mut messages = vec![json!({"role": "system", "content": system_parts.join("\n")})];

    let recent = session.recent_messages(10);
    for msg in &recent {
        let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("");
        if role != "system" && !role.is_empty() {
            messages.push(msg.clone());
        }
    }

    if messages
        .last()
        .and_then(|m| m.get("content").and_then(|c| c.as_str()))
        != Some(&user_text)
    {
        messages.push(json!({"role": "user", "content": user_text}));
    }

    PackedContext {
        messages,
        max_tokens: 1024,
    }
}

/// Build context for the reducer when arbitration is needed.
pub fn pack_for_reducer(
    session: &Session,
    outputs: &[WorkerOutput],
    reason: &str,
    has_tools: bool,
) -> Vec<Value> {
    let user_text = session.last_user_text();

    let mut system_parts = vec![
        "You are the final decision maker. Multiple models have analyzed a task.".to_string(),
        format!("Reason for escalation: {reason}"),
        String::new(),
        "Review the worker outputs below and produce ONE final response.".to_string(),
        "Start your response directly with 'kind:' — no preamble, no thinking, no explanation."
            .to_string(),
        String::new(),
        "kind: answer | tool_proposal".to_string(),
        "confidence: 0.0-1.0".to_string(),
        "tool: (only if tool_proposal) tool name".to_string(),
        "arguments: (only if tool_proposal) tool arguments as JSON".to_string(),
        "payload: your final response".to_string(),
        String::new(),
        "Output ONLY the fields above.".to_string(),
    ];

    if has_tools {
        if let Some(tools) = session.tools() {
            if let Some(tools_array) = tools.as_array() {
                system_parts.push(String::new());
                system_parts.push("Available tools:".to_string());
                for tool in tools_array {
                    if let Ok(compact) = serde_json::to_string(tool) {
                        system_parts.push(format!("  {compact}"));
                    }
                }
            }
        }
    }

    // Add worker outputs
    system_parts.push(String::new());
    system_parts.push("## Worker outputs".to_string());
    for (i, output) in outputs.iter().enumerate() {
        system_parts.push(format!(
            "[Worker {} — {} ({}), {:?}, confidence {:.2}]:",
            i + 1,
            output.model,
            output.role.label(),
            output.kind,
            output.confidence,
        ));
        // Truncate very long payloads for the reducer
        let payload = if output.payload.len() > 500 {
            format!("{}...", &output.payload[..497])
        } else {
            output.payload.clone()
        };
        system_parts.push(payload);
        if let Some(ref tool) = output.tool_name {
            system_parts.push(format!("  Proposed tool: {tool}"));
            if let Some(ref args) = output.tool_arguments {
                system_parts.push(format!("  Arguments: {args}"));
            }
        }
        system_parts.push(String::new());
    }

    vec![
        json!({"role": "system", "content": system_parts.join("\n")}),
        json!({"role": "user", "content": user_text}),
    ]
}

/// Build context for a tool-result turn (reducer only, not full fan-out).
pub fn pack_for_tool_result_turn(session: &Session, has_tools: bool) -> Vec<Value> {
    let _user_text = session.last_user_text();

    let mut system_parts = vec![
        "You received a tool result. Use it to produce a final answer \
         or propose the next tool call if more work is needed."
            .to_string(),
        String::new(),
        "Respond with:".to_string(),
        "kind: answer | tool_proposal".to_string(),
        "confidence: 0.0-1.0".to_string(),
        "tool: (if tool_proposal) tool name".to_string(),
        "arguments: (if tool_proposal) tool arguments as JSON".to_string(),
        "payload: your response".to_string(),
    ];

    if has_tools {
        if let Some(tools) = session.tools() {
            if let Some(tools_array) = tools.as_array() {
                system_parts.push(String::new());
                system_parts.push("Available tools:".to_string());
                for tool in tools_array {
                    if let Ok(compact) = serde_json::to_string(tool) {
                        system_parts.push(format!("  {compact}"));
                    }
                }
            }
        }
    }

    // Include recent tool results
    let tool_results = session.recent_tool_results();
    if !tool_results.is_empty() {
        system_parts.push(String::new());
        system_parts.push("## Recent tool results".to_string());
        for (name, result) in &tool_results {
            system_parts.push(format!("{name}: {result}"));
        }
    }

    // Build messages: system + user context.
    // Do NOT pass raw tool/assistant-with-tool_calls messages — many backends
    // reject them or require exact schema alignment.  Instead, fold tool
    // history into the system prompt above (via recent_tool_results) and
    // re-state the user's question.
    let user_text = session.last_user_text();
    vec![
        json!({"role": "system", "content": system_parts.join("\n")}),
        json!({"role": "user", "content": user_text}),
    ]
}
