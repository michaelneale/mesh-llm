//! Worker dispatch — send requests to model endpoints and get raw text back.

use crate::Endpoint;
use serde_json::{json, Value};
use std::time::Duration;

/// Worker role determines the system prompt and context shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerRole {
    /// Fast small model — classify, quick proposal.
    Fast,
    /// Specialist — code, domain knowledge.
    Specialist,
    /// Strong reasoner — deeper analysis.
    Strong,
    /// General-purpose worker.
    Generalist,
    /// Reducer/finalizer — only invoked for arbitration.
    Reducer,
}

impl WorkerRole {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Fast => "fast",
            Self::Specialist => "specialist",
            Self::Strong => "strong",
            Self::Generalist => "generalist",
            Self::Reducer => "reducer",
        }
    }
}

/// A worker assignment: which endpoint plays which role.
pub struct Assignment {
    pub endpoint: Endpoint,
    pub role: WorkerRole,
}

/// Assign roles to endpoints.
///
/// Heuristic: more endpoints = more specialization.
/// With 2 endpoints: fast + strong.
/// With 3+: fast + specialist(s) + strong.
pub fn assign_roles(endpoints: &[Endpoint], _has_tools: bool) -> Vec<Assignment> {
    if endpoints.is_empty() {
        return vec![];
    }
    if endpoints.len() == 1 {
        return vec![Assignment {
            endpoint: endpoints[0].clone(),
            role: WorkerRole::Generalist,
        }];
    }

    let mut assignments = Vec::new();

    // First endpoint = fast worker
    assignments.push(Assignment {
        endpoint: endpoints[0].clone(),
        role: WorkerRole::Fast,
    });

    // Middle endpoints = specialists
    for ep in &endpoints[1..endpoints.len() - 1] {
        assignments.push(Assignment {
            endpoint: ep.clone(),
            role: WorkerRole::Specialist,
        });
    }

    // Last endpoint = strong (also serves as reducer when needed)
    assignments.push(Assignment {
        endpoint: endpoints.last().unwrap().clone(),
        role: WorkerRole::Strong,
    });

    assignments
}

/// Call a worker endpoint.  Returns the raw assistant text content.
pub async fn call(
    http: &reqwest::Client,
    endpoint: &Endpoint,
    messages: &[Value],
    max_tokens: u32,
    timeout: Duration,
) -> Result<String, String> {
    let url = format!("{}/chat/completions", endpoint.base_url);

    let body = json!({
        "model": endpoint.model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.3,
        "stream": false,
    });

    let resp = http
        .post(&url)
        .json(&body)
        .timeout(timeout)
        .send()
        .await
        .map_err(|e| format!("request failed: {e}"))?;

    let status = resp.status();
    if !status.is_success() {
        let body = resp.text().await.unwrap_or_default();
        return Err(format!("HTTP {}: {}", status, &body[..body.len().min(200)]));
    }

    let resp_body: Value = resp
        .json()
        .await
        .map_err(|e| format!("response parse failed: {e}"))?;

    // Extract assistant content — handle both text and tool_call responses
    let message = &resp_body["choices"][0]["message"];

    // If the model returned tool_calls natively, serialize them into text
    // so the normalizer can parse them uniformly
    if let Some(tool_calls) = message.get("tool_calls").and_then(|tc| tc.as_array()) {
        if !tool_calls.is_empty() {
            // Format as structured text the normalizer understands
            let tc = &tool_calls[0];
            let name = tc
                .pointer("/function/name")
                .and_then(|n| n.as_str())
                .unwrap_or("unknown");
            let args = tc
                .pointer("/function/arguments")
                .and_then(|a| a.as_str())
                .unwrap_or("{}");
            return Ok(format!(
                "kind: tool_proposal\ntool: {name}\narguments: {args}\nconfidence: 0.9\npayload: I propose calling {name}",
            ));
        }
    }

    let content = message
        .get("content")
        .and_then(|c| c.as_str())
        .unwrap_or("")
        .to_string();

    // Some providers (ollama) put thinking into a separate "reasoning" field.
    // Use it as fallback if content is empty.
    let reasoning = message
        .get("reasoning")
        .and_then(|r| r.as_str())
        .unwrap_or("")
        .to_string();

    // Strip <think>...</think> tags from content
    let stripped = strip_thinking(&content);

    if !stripped.is_empty() {
        return Ok(stripped);
    }

    // Content was empty or only thinking — try the thinking itself
    let thinking = extract_thinking(&content);
    if !thinking.is_empty() {
        return Ok(thinking);
    }

    // Fall back to the reasoning field (ollama puts thinking there)
    if !reasoning.is_empty() {
        return Ok(reasoning);
    }

    Err("empty response".into())
}

/// Strip `<think>...</think>` tags, return the remaining content.
fn strip_thinking(text: &str) -> String {
    let mut result = text.to_string();
    while let Some(start) = result.find("<think>") {
        if let Some(end) = result[start..].find("</think>") {
            result = format!(
                "{}{}",
                &result[..start],
                &result[start + end + "</think>".len()..]
            );
        } else {
            // Unclosed — remove from <think> to end
            result = result[..start].to_string();
            break;
        }
    }
    result.trim().to_string()
}

/// Extract content inside `<think>` tags.
fn extract_thinking(text: &str) -> String {
    if let Some(start) = text.find("<think>") {
        let after = &text[start + "<think>".len()..];
        if let Some(end) = after.find("</think>") {
            return after[..end].trim().to_string();
        }
        // Unclosed — take everything after <think>
        return after.trim().to_string();
    }
    String::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn assign_two_endpoints() {
        let eps = vec![
            Endpoint {
                base_url: "http://a".into(),
                model: "small".into(),
            },
            Endpoint {
                base_url: "http://b".into(),
                model: "big".into(),
            },
        ];
        let assignments = assign_roles(&eps, false);
        assert_eq!(assignments.len(), 2);
        assert_eq!(assignments[0].role, WorkerRole::Fast);
        assert_eq!(assignments[1].role, WorkerRole::Strong);
    }

    #[test]
    fn assign_three_endpoints() {
        let eps = vec![
            Endpoint {
                base_url: "http://a".into(),
                model: "small".into(),
            },
            Endpoint {
                base_url: "http://b".into(),
                model: "mid".into(),
            },
            Endpoint {
                base_url: "http://c".into(),
                model: "big".into(),
            },
        ];
        let assignments = assign_roles(&eps, true);
        assert_eq!(assignments.len(), 3);
        assert_eq!(assignments[0].role, WorkerRole::Fast);
        assert_eq!(assignments[1].role, WorkerRole::Specialist);
        assert_eq!(assignments[2].role, WorkerRole::Strong);
    }
}
