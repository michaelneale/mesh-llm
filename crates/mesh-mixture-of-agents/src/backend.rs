//! Backend abstraction for calling models.
//!
//! The gateway doesn't care whether the model is local HTTP, remote QUIC,
//! or something else entirely — it talks to backends through the
//! [`ModelBackend`] trait. The default [`HttpBackend`] talks to any
//! OpenAI-compatible HTTP endpoint and is suitable for standalone/test
//! use. The mesh host-runtime provides mesh-native backends that dispatch
//! local models via direct HTTP and remote models via QUIC tunnel.

use crate::worker;
use serde_json::{json, Value};
use std::time::Duration;

// ─── Sampling params ─────────────────────────────────────────────────

/// Sampling hyperparameters sent to backend models.
/// Workers get higher temperature for diversity; reducer gets lower for precision.
#[derive(Debug, Clone, Copy)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_p: f32,
}

impl SamplingParams {
    /// High-diversity settings for MoA workers — encourages each model
    /// to explore different parts of the solution space.
    pub fn worker() -> Self {
        Self {
            temperature: 0.8,
            top_p: 0.95,
        }
    }

    /// Low-variance settings for the reducer — precise synthesis.
    pub fn reducer() -> Self {
        Self {
            temperature: 0.3,
            top_p: 0.9,
        }
    }
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self::reducer()
    }
}

// ─── Backend trait ───────────────────────────────────────────────────

/// Abstraction for calling a model.  The gateway doesn't care whether
/// the model is local HTTP, remote QUIC, or something else entirely.
#[async_trait::async_trait]
pub trait ModelBackend: Send + Sync + 'static {
    /// Call the model with the given messages (and optionally tools).
    /// Returns the full JSON response body from the model.
    async fn chat_completion(
        &self,
        model: &str,
        messages: &[Value],
        tools: Option<&Value>,
        max_tokens: u32,
        timeout: Duration,
        sampling: SamplingParams,
    ) -> Result<Value, String>;
}

/// Default HTTP backend — calls any OpenAI-compatible endpoint.
pub struct HttpBackend {
    pub base_url: String,
    http: reqwest::Client,
}

impl HttpBackend {
    pub fn new(base_url: String) -> Self {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(120))
            .build()
            .unwrap_or_default();
        Self { base_url, http }
    }
}

#[async_trait::async_trait]
impl ModelBackend for HttpBackend {
    async fn chat_completion(
        &self,
        model: &str,
        messages: &[Value],
        tools: Option<&Value>,
        max_tokens: u32,
        timeout: Duration,
        sampling: SamplingParams,
    ) -> Result<Value, String> {
        let url = format!("{}/chat/completions", self.base_url);
        let mut body = json!({
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": sampling.temperature,
            "top_p": sampling.top_p,
            "stream": false,
        });
        if let Some(tools) = tools {
            body.as_object_mut()
                .unwrap()
                .insert("tools".to_string(), tools.clone());
        }

        let resp = self
            .http
            .post(&url)
            .json(&body)
            .timeout(timeout)
            .send()
            .await
            .map_err(|e| format!("request failed: {e}"))?;

        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(format!("HTTP {status}: {}", &text[..text.len().min(200)]));
        }

        resp.json::<Value>()
            .await
            .map_err(|e| format!("response parse: {e}"))
    }
}

// ─── Model entry ─────────────────────────────────────────────────────

/// A model available for MoA fan-out.
#[derive(Clone)]
pub struct ModelEntry {
    /// Model name (as used in the API).
    pub name: String,
    /// Index into the backends vec.  Multiple models can share a backend
    /// (e.g. all models behind the same proxy) or each have their own.
    pub backend_index: usize,
}

// ─── Backend call + text extraction ──────────────────────────────────

/// Call a backend and extract the assistant text from the response.
/// Retries once on HTTP 429 (rate limit) after the server's `retry-after`
/// delay (default 1s).
pub(crate) async fn call_backend(
    backend: &dyn ModelBackend,
    model: &str,
    messages: &[Value],
    tools: Option<&Value>,
    max_tokens: u32,
    timeout: Duration,
    sampling: SamplingParams,
) -> Result<String, String> {
    match backend
        .chat_completion(model, messages, tools, max_tokens, timeout, sampling)
        .await
    {
        Ok(resp) => extract_text_from_response(&resp),
        Err(e) if e.contains("429") => {
            // Parse retry-after from error message if present, default 1s
            let delay = parse_retry_after(&e).unwrap_or(1);
            tracing::info!("moa: 429 from {model}, retrying after {delay}s");
            tokio::time::sleep(Duration::from_secs(delay)).await;
            let resp = backend
                .chat_completion(model, messages, tools, max_tokens, timeout, sampling)
                .await?;
            extract_text_from_response(&resp)
        }
        Err(e) => Err(e),
    }
}

/// Extract retry-after seconds from an error message containing "retry-after: N".
fn parse_retry_after(err: &str) -> Option<u64> {
    let lower = err.to_lowercase();
    lower
        .find("retry-after:")
        .map(|i| &err[i + 12..])
        .and_then(|s| s.split_whitespace().next())
        .and_then(|s| s.trim().parse::<u64>().ok())
}

/// Extract assistant text from a chat completion response body.
fn extract_text_from_response(resp: &Value) -> Result<String, String> {
    let message = &resp["choices"][0]["message"];

    // Native tool_calls → KV format for normalizer
    if let Some(tool_calls) = message.get("tool_calls").and_then(|tc| tc.as_array()) {
        if let Some(tc) = tool_calls.first() {
            let name = tc
                .pointer("/function/name")
                .and_then(|n| n.as_str())
                .unwrap_or("unknown");
            let args = tc
                .pointer("/function/arguments")
                .and_then(|a| a.as_str())
                .unwrap_or("{}");
            return Ok(format!(
                "kind: tool_proposal\ntool: {name}\narguments: {args}\nconfidence: 0.9\npayload: calling {name}",
            ));
        }
    }

    let content = message
        .get("content")
        .and_then(|c| c.as_str())
        .unwrap_or("")
        .to_string();

    let stripped = worker::strip_thinking(&content);
    if !stripped.is_empty() {
        return Ok(stripped);
    }

    let thinking = worker::extract_thinking(&content);
    if !thinking.is_empty() {
        return Ok(thinking);
    }

    let reasoning = message
        .get("reasoning")
        .and_then(|r| r.as_str())
        .unwrap_or("");
    if !reasoning.is_empty() {
        return Ok(reasoning.to_string());
    }

    Err("empty response".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_retry_after_from_header() {
        let err = "HTTP 429: retry-after: 2\r\ncontent-type: application/json";
        assert_eq!(parse_retry_after(err), Some(2));
    }

    #[test]
    fn parse_retry_after_missing() {
        let err = "HTTP 429: Too Many Requests";
        assert_eq!(parse_retry_after(err), None);
    }

    #[test]
    fn parse_retry_after_case_insensitive() {
        let err = "Retry-After: 5";
        assert_eq!(parse_retry_after(err), Some(5));
    }

    #[test]
    fn worker_sampling_high_diversity() {
        let s = SamplingParams::worker();
        assert!(s.temperature > 0.5, "workers need high temp for diversity");
        assert!(s.top_p > 0.9, "workers need high top_p for diversity");
    }

    #[test]
    fn reducer_sampling_low_variance() {
        let s = SamplingParams::reducer();
        assert!(s.temperature <= 0.4, "reducer needs low temp for precision");
        assert!(s.top_p <= 0.95, "reducer needs bounded top_p");
    }

    #[test]
    fn default_sampling_is_reducer() {
        let d = SamplingParams::default();
        let r = SamplingParams::reducer();
        assert!((d.temperature - r.temperature).abs() < f32::EPSILON);
        assert!((d.top_p - r.top_p).abs() < f32::EPSILON);
    }
}
