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
    match role {
        WorkerRole::Fast => pack_fast(session, has_tools),
        WorkerRole::Specialist => pack_specialist(session, has_tools),
        WorkerRole::Strong => pack_strong(session, has_tools),
        WorkerRole::Generalist => pack_strong(session, has_tools),
        WorkerRole::Reducer => pack_strong(session, has_tools),
    }
}

/// Fast worker: current task only, tool names, short max_tokens.
fn pack_fast(session: &Session, has_tools: bool) -> PackedContext {
    let user_text = session.last_user_text();
    let mut system_parts = vec![
        "You are a fast analysis worker in a multi-model ensemble.".to_string(),
        "Given the task below, produce a structured response:".to_string(),
        String::new(),
        "kind: answer | tool_proposal | critique | uncertainty".to_string(),
        "confidence: 0.0-1.0".to_string(),
        "tool: (optional) tool name".to_string(),
        "arguments: (optional) tool arguments as JSON".to_string(),
        "payload: your response text".to_string(),
        String::new(),
        "Be concise. If you think a tool should be called, say so. Do not \
         actually execute tools."
            .to_string(),
    ];

    if has_tools {
        let names = session.tool_names();
        if !names.is_empty() {
            system_parts.push(String::new());
            system_parts.push(format!("Available tools: {}", names.join(", ")));
        }
    }

    // Add a brief summary if there's history
    if session.turn_count() > 1 {
        let recent = session.recent_messages(4);
        let summary = summarize_messages(&recent);
        if !summary.is_empty() {
            system_parts.push(String::new());
            system_parts.push(format!("Context: {summary}"));
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
fn pack_specialist(session: &Session, has_tools: bool) -> PackedContext {
    let user_text = session.last_user_text();
    let mut system_parts = vec![
        "You are a specialist worker in a multi-model ensemble.".to_string(),
        "Analyze the task carefully and produce a structured response:".to_string(),
        String::new(),
        "kind: answer | tool_proposal | critique | uncertainty".to_string(),
        "confidence: 0.0-1.0".to_string(),
        "tool: (optional) tool name".to_string(),
        "arguments: (optional) tool arguments as JSON".to_string(),
        "payload: your detailed response".to_string(),
        String::new(),
        "If a tool should be called, specify which one and with what arguments. \
         Do not execute tools yourself."
            .to_string(),
    ];

    if has_tools {
        let summaries = session.tool_summaries();
        if !summaries.is_empty() {
            system_parts.push(String::new());
            system_parts.push("Available tools:".to_string());
            for s in &summaries {
                system_parts.push(format!("  - {s}"));
            }
        }
    }

    // Include recent conversation context
    let recent = session.recent_messages(6);
    let mut messages = vec![json!({"role": "system", "content": system_parts.join("\n")})];

    // Add recent history (skip system messages, keep user/assistant/tool)
    for msg in &recent {
        let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("");
        if role != "system" && role != "" {
            messages.push(msg.clone());
        }
    }

    // If the last message isn't the current user turn, add it
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
fn pack_strong(session: &Session, has_tools: bool) -> PackedContext {
    let user_text = session.last_user_text();
    let mut system_parts = vec![
        "You are a strong reasoning worker in a multi-model ensemble.".to_string(),
        "Analyze the task thoroughly and produce a structured response:".to_string(),
        String::new(),
        "kind: answer | tool_proposal | critique | uncertainty".to_string(),
        "confidence: 0.0-1.0".to_string(),
        "tool: (optional) tool name".to_string(),
        "arguments: (optional) tool arguments as JSON".to_string(),
        "payload: your thorough response".to_string(),
        String::new(),
        "Think step by step. If a tool should be called, specify exactly which one \
         and with what arguments. If you can answer directly, do so with high \
         confidence. Do not execute tools yourself."
            .to_string(),
    ];

    if has_tools {
        // Give strong worker full tool schemas
        if let Some(tools) = session.tools() {
            if let Some(tools_array) = tools.as_array() {
                system_parts.push(String::new());
                system_parts.push("Available tools (full schema):".to_string());
                for tool in tools_array {
                    if let Ok(compact) = serde_json::to_string(tool) {
                        system_parts.push(format!("  {compact}"));
                    }
                }
            }
        }
    }

    // Include system prompt if present
    if let Some(sp) = session.system_prompt() {
        system_parts.push(String::new());
        system_parts.push(format!("Original system prompt: {sp}"));
    }

    let mut messages = vec![json!({"role": "system", "content": system_parts.join("\n")})];

    // Include more history for the strong worker
    let recent = session.recent_messages(10);
    for msg in &recent {
        let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("");
        if role != "system" && role != "" {
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
        "Either answer directly or propose exactly one tool call.".to_string(),
        String::new(),
        "Respond with:".to_string(),
        "kind: answer | tool_proposal".to_string(),
        "confidence: 0.0-1.0".to_string(),
        "tool: (if tool_proposal) tool name".to_string(),
        "arguments: (if tool_proposal) tool arguments as JSON".to_string(),
        "payload: your final response".to_string(),
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

/// Quick summary of messages for compact context.
fn summarize_messages(messages: &[Value]) -> String {
    let mut parts = Vec::new();
    for msg in messages {
        let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("?");
        let content = msg
            .get("content")
            .and_then(|c| c.as_str())
            .unwrap_or("[non-text]");
        let truncated = if content.len() > 100 {
            format!("{}...", &content[..97])
        } else {
            content.to_string()
        };
        parts.push(format!("{role}: {truncated}"));
    }
    parts.join(" | ")
}
