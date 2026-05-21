//! `OpenAiBackend` decorators that engage the guardrails and the
//! compactor at the host layer.
//!
//! These wrap a host's inner backend at construction so every
//! consumer of that backend (direct ingress, MoA workers, virtual-LLM
//! consults) benefits without per-call-site logic.
//!
//! ## First-pass scope
//!
//! - **Buffered tools only.** `chat_completion_stream` short-circuits
//!   to the inner backend when no tools are present (or guarding is
//!   disabled for this request). A late-rewrite streaming path is a
//!   later cut.
//! - **Step enforcer off-by-default.** Required steps aren't declared
//!   on raw agent traffic, so the decorator constructs the facade
//!   with empty step / terminal lists. Tests can pass a non-empty
//!   config.
//! - **`respond` injection off-by-default.** Flipping it on is a
//!   single config change but it mutates the request's tools array,
//!   which is invasive — turn it on only with explicit opt-in until
//!   the validation gate runs.

use std::collections::BTreeSet;
use std::sync::Arc;

use async_trait::async_trait;
use openai_frontend::backend::{
    ChatCompletionStream, CompletionStream, OpenAiBackend, OpenAiRequestContext, OpenAiResult,
};
use openai_frontend::chat::{
    AssistantMessage, ChatCompletionRequest, ChatCompletionResponse, ChatMessage, MessageContent,
};
use openai_frontend::common::{FinishReason, Usage};
use openai_frontend::completions::{CompletionRequest, CompletionResponse};
use openai_frontend::hooks::inject_text_into_chat_messages;
use openai_frontend::models::ModelObject;
use serde_json::Value;

use crate::compact::{self, CompactConfig, NUDGE_MARKER_KEY};
use crate::facade::{GuardrailConfig, Guardrails};
use crate::respond::{extract_respond_message, inject_respond_tool, RESPOND_TOOL_NAME};
use crate::types::{GuardrailAction, LlmResponse, Nudge, NudgeKind, ToolCall};

/// Per-request override field that turns the guardrail decorator on
/// or off explicitly. Mirrors the existing `mesh_hooks` pattern in
/// `openai-frontend/src/hooks.rs`.
pub const MESH_GUARDRAILS_FIELD: &str = "mesh_guardrails";

/// Per-request override field that turns the compactor on or off.
pub const MESH_COMPACT_FIELD: &str = "mesh_compact";

#[derive(Debug, Clone)]
pub struct GuardrailBackendConfig {
    pub max_retries: u32,
    pub rescue_enabled: bool,
    pub inject_respond_tool: bool,
    /// Tools considered "terminal" — used only when paired with
    /// `required_steps`. Empty by default (the step enforcer is a
    /// no-op for ingress traffic).
    pub terminal_tools: BTreeSet<String>,
    pub required_steps: Vec<String>,
}

impl Default for GuardrailBackendConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            rescue_enabled: true,
            inject_respond_tool: false,
            terminal_tools: BTreeSet::new(),
            required_steps: Vec::new(),
        }
    }
}

pub struct GuardrailBackend {
    inner: Arc<dyn OpenAiBackend>,
    config: GuardrailBackendConfig,
}

impl GuardrailBackend {
    pub fn new(inner: Arc<dyn OpenAiBackend>, config: GuardrailBackendConfig) -> Self {
        Self { inner, config }
    }

    fn override_for_request(&self, request: &ChatCompletionRequest) -> Option<bool> {
        request
            .extra
            .get(MESH_GUARDRAILS_FIELD)
            .and_then(Value::as_bool)
    }

    fn should_guard(&self, request: &ChatCompletionRequest) -> bool {
        if let Some(b) = self.override_for_request(request) {
            return b;
        }
        // Default policy: nothing to validate if there are no tools.
        matches!(
            request.tools.as_ref().and_then(Value::as_array),
            Some(arr) if !arr.is_empty()
        )
    }

    fn tool_names_from_request(request: &ChatCompletionRequest) -> Vec<String> {
        let Some(arr) = request.tools.as_ref().and_then(Value::as_array) else {
            return Vec::new();
        };
        arr.iter()
            .filter_map(|t| t.get("function"))
            .filter_map(|f| f.get("name"))
            .filter_map(Value::as_str)
            .map(str::to_string)
            .collect()
    }
}

#[async_trait]
impl OpenAiBackend for GuardrailBackend {
    async fn models(&self) -> OpenAiResult<Vec<ModelObject>> {
        self.inner.models().await
    }

    async fn chat_completion(
        &self,
        mut request: ChatCompletionRequest,
    ) -> OpenAiResult<ChatCompletionResponse> {
        if !self.should_guard(&request) {
            return self.inner.chat_completion(request).await;
        }

        // Optionally inject the synthetic `respond` tool so the model
        // is steered toward structured replies even for chat-style
        // questions.
        if self.config.inject_respond_tool {
            if let Some(tools) = request.tools.as_mut() {
                let _ = inject_respond_tool(tools);
            }
        }

        let mut tool_names = Self::tool_names_from_request(&request);
        if self.config.inject_respond_tool && !tool_names.iter().any(|t| t == RESPOND_TOOL_NAME) {
            tool_names.push(RESPOND_TOOL_NAME.to_string());
        }

        let mut state = Guardrails::new(GuardrailConfig {
            tool_names,
            terminal_tools: self.config.terminal_tools.clone(),
            required_steps: self.config.required_steps.clone(),
            max_retries: self.config.max_retries,
            rescue_enabled: self.config.rescue_enabled,
            ..Default::default()
        });

        let mut last_response: Option<ChatCompletionResponse> = None;

        for _ in 0..self.config.max_retries.saturating_add(1) {
            let response = self.inner.chat_completion(request.clone()).await?;
            let llm = response_to_llm_response(&response);
            match state.check(llm.clone()) {
                GuardrailAction::Execute { tool_calls } => {
                    return Ok(finalize_response(
                        response,
                        tool_calls,
                        self.config.inject_respond_tool,
                    ));
                }
                GuardrailAction::Retry { nudge } | GuardrailAction::StepBlocked { nudge } => {
                    push_nudge(&mut request.messages, &nudge);
                    last_response = Some(response);
                }
                GuardrailAction::Fatal { reason } => {
                    tracing::warn!(
                        target = "mesh_llm_guardrails",
                        reason = %reason,
                        "guardrails exhausted; returning last response as text"
                    );
                    return Ok(passthrough_last_text(response));
                }
            }
        }

        Ok(last_response.map(passthrough_last_text).unwrap_or_else(|| {
            // Should be unreachable because we always make at
            // least one inner call, but be defensive.
            ChatCompletionResponse::new(request.model.clone(), String::new(), Usage::default())
        }))
    }

    async fn chat_completion_stream(
        &self,
        request: ChatCompletionRequest,
        context: OpenAiRequestContext,
    ) -> OpenAiResult<ChatCompletionStream> {
        // First-pass: only short-circuit for non-guarded requests.
        // Streaming through guardrails requires a late-rewrite SSE
        // path that's not in scope here.
        if !self.should_guard(&request) {
            return self.inner.chat_completion_stream(request, context).await;
        }
        // For guarded streaming, we still pass through unchanged
        // until the buffered rewrite lands. Mirrors the design doc's
        // "open issue 1".
        self.inner.chat_completion_stream(request, context).await
    }

    async fn completion(&self, request: CompletionRequest) -> OpenAiResult<CompletionResponse> {
        self.inner.completion(request).await
    }

    async fn completion_stream(
        &self,
        request: CompletionRequest,
        context: OpenAiRequestContext,
    ) -> OpenAiResult<CompletionStream> {
        self.inner.completion_stream(request, context).await
    }
}

/// Convert an OpenAI-style assistant message into `LlmResponse`
/// shape so the validator can chew on it.
fn response_to_llm_response(response: &ChatCompletionResponse) -> LlmResponse {
    let Some(choice) = response.choices.first() else {
        return LlmResponse::Text(String::new());
    };
    let msg = &choice.message;
    if let Some(tool_calls) = msg.tool_calls.as_ref().and_then(Value::as_array) {
        let calls: Vec<ToolCall> = tool_calls
            .iter()
            .filter_map(parse_openai_tool_call)
            .collect();
        if !calls.is_empty() {
            return LlmResponse::ToolCalls(calls);
        }
    }
    LlmResponse::Text(msg.content.clone().unwrap_or_default())
}

fn parse_openai_tool_call(value: &Value) -> Option<ToolCall> {
    let function = value.get("function")?;
    let name = function.get("name").and_then(Value::as_str)?;
    let raw_args = function.get("arguments");
    let args = match raw_args {
        Some(Value::String(s)) => {
            serde_json::from_str(s).unwrap_or(Value::Object(Default::default()))
        }
        Some(other) => other.clone(),
        None => Value::Object(Default::default()),
    };
    Some(ToolCall::new(name, args))
}

/// Rebuild the assistant message from the (possibly rescued) tool
/// calls. If the rescued list contains a single `respond` call and
/// the caller didn't ask for `respond` originally, collapse it into
/// a plain `content` string.
fn finalize_response(
    mut response: ChatCompletionResponse,
    tool_calls: Vec<ToolCall>,
    respond_was_injected: bool,
) -> ChatCompletionResponse {
    if let Some(choice) = response.choices.first_mut() {
        if respond_was_injected && tool_calls.len() == 1 && tool_calls[0].tool == RESPOND_TOOL_NAME
        {
            let message = extract_respond_message(&tool_calls[0].args).unwrap_or_default();
            choice.message = AssistantMessage {
                role: "assistant",
                content: Some(message),
                reasoning_content: None,
                tool_calls: None,
            };
            choice.finish_reason = Some(FinishReason::Stop);
        } else {
            choice.message.tool_calls = Some(tool_calls_to_openai(&tool_calls));
            choice.message.content = None;
            choice.finish_reason = Some(FinishReason::ToolCalls);
        }
    }
    response
}

fn tool_calls_to_openai(calls: &[ToolCall]) -> Value {
    Value::Array(
        calls
            .iter()
            .enumerate()
            .map(|(i, c)| {
                serde_json::json!({
                    "id": format!("call_{i}"),
                    "type": "function",
                    "function": {
                        "name": c.tool,
                        "arguments": c.args.to_string(),
                    },
                })
            })
            .collect(),
    )
}

/// On `Fatal`, hand the last text back to the caller instead of
/// erroring. Mirrors forge `proxy/handler.py`.
fn passthrough_last_text(mut response: ChatCompletionResponse) -> ChatCompletionResponse {
    if let Some(choice) = response.choices.first_mut() {
        // Keep the text as-is so callers get *something* to react to.
        if choice.message.content.is_none() {
            choice.message.content = Some(String::new());
        }
        choice.message.tool_calls = None;
        choice.finish_reason = Some(FinishReason::Stop);
    }
    response
}

fn push_nudge(messages: &mut Vec<ChatMessage>, nudge: &Nudge) {
    let mut extra: std::collections::BTreeMap<String, Value> = Default::default();
    let marker = match nudge.kind {
        NudgeKind::Retry => "retry",
        NudgeKind::UnknownTool => "retry", // grouped with retry for compaction priority
        NudgeKind::Step => "step",
        NudgeKind::Prerequisite => "prerequisite",
    };
    extra.insert(NUDGE_MARKER_KEY.into(), Value::String(marker.into()));
    messages.push(ChatMessage {
        role: nudge.role.clone(),
        content: Some(MessageContent::Text(nudge.content.clone())),
        extra,
    });
}

// ---------------------------------------------------------------------------
// CompactingBackend
// ---------------------------------------------------------------------------

pub struct CompactingBackend {
    inner: Arc<dyn OpenAiBackend>,
    n_ctx: u32,
    config: CompactConfig,
    inject_threshold_warning: bool,
    /// Most recent `usage.total_tokens` reported by the inner backend.
    /// Lives here (not in a forge-style `ContextManager`) because mesh
    /// hosts are stateless turn-to-turn — this value is just a
    /// last-known hint, used to override the char/4 heuristic when the
    /// inner backend's tokenizer disagrees (as it usually does, by a
    /// lot, on tool catalogs). See forge `ContextManager`.
    last_known_tokens: std::sync::atomic::AtomicU32,
}

impl CompactingBackend {
    pub fn new(
        inner: Arc<dyn OpenAiBackend>,
        n_ctx: u32,
        config: CompactConfig,
        inject_threshold_warning: bool,
    ) -> Self {
        Self {
            inner,
            n_ctx,
            config,
            inject_threshold_warning,
            last_known_tokens: std::sync::atomic::AtomicU32::new(0),
        }
    }

    fn override_for_request(&self, request: &ChatCompletionRequest) -> Option<bool> {
        request
            .extra
            .get(MESH_COMPACT_FIELD)
            .and_then(Value::as_bool)
    }

    /// Best available estimate of how many tokens are about to hit
    /// the model. Prefers the inner backend's last reported
    /// `usage.total_tokens` (real tokenization) over the char/4
    /// heuristic, when present. Falls back to char/4 on the first
    /// turn of a session.
    fn estimate_tokens(&self, request: &ChatCompletionRequest) -> usize {
        let heuristic = compact::estimate_tokens(&request.messages);
        let last_known = self
            .last_known_tokens
            .load(std::sync::atomic::Ordering::Relaxed) as usize;
        if last_known == 0 {
            return heuristic;
        }
        // On a subsequent turn, the new request has *more* than the
        // last total (we appended a user/tool message + maybe nudges).
        // Use the larger of (last_known_tokens, char/4) so we err on
        // the side of compacting earlier rather than later. This is
        // the "actual prompt was bigger than I thought" guard that
        // catches dense tool-schema bloat — forge does the same.
        std::cmp::max(last_known, heuristic)
    }

    fn should_compact(&self, request: &ChatCompletionRequest) -> bool {
        if let Some(b) = self.override_for_request(request) {
            return b;
        }
        if self.n_ctx == 0 {
            return false;
        }
        let tokens = self.estimate_tokens(request);
        let budget = self.n_ctx as usize;
        let trigger = ((budget as f32) * self.config.phase1_threshold) as usize;
        tokens >= trigger
    }

    fn apply(&self, request: &mut ChatCompletionRequest) {
        let _outcome = compact::compact(&mut request.messages, self.n_ctx, &self.config);
        if !self.inject_threshold_warning {
            return;
        }
        let tokens = self.estimate_tokens(request);
        if let Some(warning) =
            compact::threshold_warning(tokens, self.n_ctx, &self.config.warning_thresholds)
        {
            inject_text_into_chat_messages(&mut request.messages, warning);
        }
    }

    fn record_usage(&self, total_tokens: u32) {
        if total_tokens > 0 {
            self.last_known_tokens
                .store(total_tokens, std::sync::atomic::Ordering::Relaxed);
        }
    }
}

#[async_trait]
impl OpenAiBackend for CompactingBackend {
    async fn models(&self) -> OpenAiResult<Vec<ModelObject>> {
        self.inner.models().await
    }

    async fn chat_completion(
        &self,
        mut request: ChatCompletionRequest,
    ) -> OpenAiResult<ChatCompletionResponse> {
        if self.should_compact(&request) {
            self.apply(&mut request);
        }
        let response = self.inner.chat_completion(request).await?;
        self.record_usage(response.usage.total_tokens);
        Ok(response)
    }

    async fn chat_completion_stream(
        &self,
        mut request: ChatCompletionRequest,
        context: OpenAiRequestContext,
    ) -> OpenAiResult<ChatCompletionStream> {
        if self.should_compact(&request) {
            self.apply(&mut request);
        }
        // Streaming: we don't intercept the SSE for usage in this
        // first cut. Subsequent non-streaming calls (and goose mixes
        // both) still update the hint.
        self.inner.chat_completion_stream(request, context).await
    }

    async fn completion(&self, request: CompletionRequest) -> OpenAiResult<CompletionResponse> {
        self.inner.completion(request).await
    }

    async fn completion_stream(
        &self,
        request: CompletionRequest,
        context: OpenAiRequestContext,
    ) -> OpenAiResult<CompletionStream> {
        self.inner.completion_stream(request, context).await
    }
}

// ---------------------------------------------------------------------------
// Construction helpers — used by the host runtime at wiring time.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct WrapConfig {
    pub guardrails_enabled: bool,
    pub guardrails: GuardrailBackendConfig,
    pub compact_enabled: bool,
    pub small_ctx_threshold: u32,
    pub compact: CompactConfig,
    pub inject_threshold_warning: bool,
}

impl Default for WrapConfig {
    fn default() -> Self {
        Self {
            guardrails_enabled: true,
            guardrails: GuardrailBackendConfig::default(),
            compact_enabled: true,
            small_ctx_threshold: 8192,
            compact: CompactConfig::default(),
            inject_threshold_warning: true,
        }
    }
}

/// Decide whether to wrap `inner` with `CompactingBackend`,
/// `GuardrailBackend`, both, or neither — based on `model_name` and
/// `n_ctx`.
///
/// Order matches the design doc:
/// `GuardrailBackend -> CompactingBackend -> inner`. Guardrail
/// retries re-enter the compactor below it, so a retry sees a fresh
/// compaction pass.
pub fn maybe_wrap_backend(
    inner: Arc<dyn OpenAiBackend>,
    model_name: &str,
    n_ctx: u32,
    config: &WrapConfig,
) -> Arc<dyn OpenAiBackend> {
    let mut backend = inner;

    if config.compact_enabled && n_ctx > 0 && n_ctx <= config.small_ctx_threshold {
        backend = Arc::new(CompactingBackend::new(
            backend,
            n_ctx,
            config.compact.clone(),
            config.inject_threshold_warning,
        ));
    }

    if config.guardrails_enabled && mesh_llm_routing::is_small_tier_name(model_name) {
        backend = Arc::new(GuardrailBackend::new(backend, config.guardrails.clone()));
    }

    backend
}

#[cfg(test)]
mod tests {
    use super::*;
    use openai_frontend::backend::{OpenAiBackend, OpenAiRequestContext, OpenAiResult};
    use openai_frontend::chat::{
        ChatCompletionRequest, ChatCompletionResponse, ChatMessage, MessageContent,
    };
    use openai_frontend::common::Usage;
    use openai_frontend::errors::OpenAiError;
    use openai_frontend::models::ModelObject;
    use std::collections::BTreeMap;
    use std::sync::Mutex;

    /// Test backend that returns scripted responses turn by turn.
    struct ScriptedBackend {
        responses: Mutex<Vec<ChatCompletionResponse>>,
        seen: Mutex<Vec<ChatCompletionRequest>>,
    }

    impl ScriptedBackend {
        fn new(responses: Vec<ChatCompletionResponse>) -> Self {
            Self {
                responses: Mutex::new(responses),
                seen: Mutex::new(Vec::new()),
            }
        }

        fn calls(&self) -> Vec<ChatCompletionRequest> {
            self.seen.lock().unwrap().clone()
        }
    }

    #[async_trait]
    impl OpenAiBackend for ScriptedBackend {
        async fn models(&self) -> OpenAiResult<Vec<ModelObject>> {
            Ok(vec![])
        }
        async fn chat_completion(
            &self,
            request: ChatCompletionRequest,
        ) -> OpenAiResult<ChatCompletionResponse> {
            self.seen.lock().unwrap().push(request);
            let mut q = self.responses.lock().unwrap();
            if q.is_empty() {
                return Err(OpenAiError::unsupported("no more scripted responses"));
            }
            Ok(q.remove(0))
        }
        async fn chat_completion_stream(
            &self,
            _request: ChatCompletionRequest,
            _context: OpenAiRequestContext,
        ) -> OpenAiResult<ChatCompletionStream> {
            Err(OpenAiError::unsupported("not used in tests"))
        }
    }

    fn bare_text_response(model: &str, text: &str) -> ChatCompletionResponse {
        ChatCompletionResponse::new(model, text, Usage::default())
    }

    fn tool_call_response(model: &str, tool: &str, args_json: &str) -> ChatCompletionResponse {
        let mut resp = ChatCompletionResponse::new(model, "", Usage::default());
        let choice = resp.choices.first_mut().unwrap();
        choice.message.content = None;
        choice.message.tool_calls = Some(serde_json::json!([{
            "id": "call_0",
            "type": "function",
            "function": {"name": tool, "arguments": args_json},
        }]));
        resp
    }

    fn user_message(text: &str) -> ChatMessage {
        ChatMessage {
            role: "user".into(),
            content: Some(MessageContent::Text(text.into())),
            extra: BTreeMap::new(),
        }
    }

    fn request_with_read_tool() -> ChatCompletionRequest {
        ChatCompletionRequest {
            model: "test-model-8B".into(),
            messages: vec![user_message("hi")],
            stream: false,
            max_tokens: None,
            max_completion_tokens: None,
            temperature: None,
            top_p: None,
            n: None,
            logprobs: None,
            top_logprobs: None,
            presence_penalty: None,
            frequency_penalty: None,
            logit_bias: None,
            response_format: None,
            tools: Some(serde_json::json!([{
                "type": "function",
                "function": {"name": "read_file"},
            }])),
            tool_choice: None,
            parallel_tool_calls: None,
            user: None,
            stop: None,
            seed: None,
            reasoning: None,
            reasoning_effort: None,
            prompt_cache_key: None,
            prompt_cache_retention: None,
            stream_options: None,
            extra: BTreeMap::new(),
        }
    }

    #[tokio::test]
    async fn passthrough_when_no_tools() {
        let inner = Arc::new(ScriptedBackend::new(vec![bare_text_response("m", "ok")]));
        let backend = GuardrailBackend::new(inner.clone(), GuardrailBackendConfig::default());
        let mut req = request_with_read_tool();
        req.tools = None;
        let resp = backend.chat_completion(req).await.unwrap();
        assert_eq!(
            resp.choices[0].message.content.as_deref(),
            Some("ok"),
            "no tools means no guarding"
        );
        assert_eq!(inner.calls().len(), 1);
    }

    #[tokio::test]
    async fn rescues_bare_text_tool_call_on_first_try() {
        let inner = Arc::new(ScriptedBackend::new(vec![bare_text_response(
            "m",
            r#"{"tool": "read_file", "args": {"path": "x"}}"#,
        )]));
        let backend = GuardrailBackend::new(
            inner.clone(),
            GuardrailBackendConfig {
                max_retries: 3,
                rescue_enabled: true,
                ..Default::default()
            },
        );
        let resp = backend
            .chat_completion(request_with_read_tool())
            .await
            .unwrap();
        let tc = resp.choices[0]
            .message
            .tool_calls
            .as_ref()
            .expect("rescued call should populate tool_calls");
        let arr = tc.as_array().unwrap();
        assert_eq!(arr.len(), 1);
        assert_eq!(arr[0]["function"]["name"], "read_file");
        assert_eq!(inner.calls().len(), 1, "single rescue should not retry");
    }

    #[tokio::test]
    async fn retries_then_succeeds_with_nudge_in_messages() {
        let inner = Arc::new(ScriptedBackend::new(vec![
            bare_text_response("m", "I'm just chatting"),
            tool_call_response("m", "read_file", r#"{"path": "y"}"#),
        ]));
        let backend = GuardrailBackend::new(
            inner.clone(),
            GuardrailBackendConfig {
                max_retries: 3,
                rescue_enabled: false,
                ..Default::default()
            },
        );
        let resp = backend
            .chat_completion(request_with_read_tool())
            .await
            .unwrap();
        let tc = resp.choices[0].message.tool_calls.as_ref().unwrap();
        assert_eq!(tc.as_array().unwrap()[0]["function"]["name"], "read_file");
        let calls = inner.calls();
        assert_eq!(calls.len(), 2);
        // Second call should contain the injected nudge.
        let second = &calls[1];
        let nudge_present = second
            .messages
            .iter()
            .any(|m| m.extra.get(NUDGE_MARKER_KEY).and_then(Value::as_str) == Some("retry"));
        assert!(nudge_present, "retry nudge must be in second-call messages");
    }

    #[tokio::test]
    async fn fatal_falls_through_as_text() {
        let inner = Arc::new(ScriptedBackend::new(vec![
            bare_text_response("m", "garbage 1"),
            bare_text_response("m", "garbage 2"),
            bare_text_response("m", "garbage 3"),
            bare_text_response("m", "garbage 4"),
            bare_text_response("m", "garbage 5"),
        ]));
        let backend = GuardrailBackend::new(
            inner.clone(),
            GuardrailBackendConfig {
                max_retries: 2,
                rescue_enabled: false,
                ..Default::default()
            },
        );
        let resp = backend
            .chat_completion(request_with_read_tool())
            .await
            .unwrap();
        // The last bare text must be returned, not an error.
        assert!(resp.choices[0].message.content.is_some());
        assert!(resp.choices[0].message.tool_calls.is_none());
    }

    #[tokio::test]
    async fn maybe_wrap_skips_large_model_and_large_ctx() {
        let inner: Arc<dyn OpenAiBackend> =
            Arc::new(ScriptedBackend::new(vec![bare_text_response("m", "ok")]));
        let cfg = WrapConfig::default();
        // Large model name, large ctx → no wraps.
        let wrapped = maybe_wrap_backend(inner.clone(), "MiniMax-M2.5", 131_072, &cfg);
        assert!(
            Arc::ptr_eq(&wrapped, &inner),
            "large model + large ctx should bypass both decorators"
        );
    }

    fn response_with_usage(model: &str, total: u32) -> ChatCompletionResponse {
        let usage = Usage::new(total.saturating_sub(2), 2);
        let mut r = ChatCompletionResponse::new(model, "ok", usage);
        if let Some(c) = r.choices.first_mut() {
            c.finish_reason = Some(FinishReason::Stop);
        }
        r
    }

    #[tokio::test]
    async fn compact_uses_reported_usage_after_first_turn() {
        // Inner backend reports a huge total_tokens — much larger
        // than the char/4 heuristic of the inbound messages. After
        // the first turn, the compactor's estimate should be the
        // reported usage and trigger on the next turn even though
        // the request *looks* small to char/4.
        let inner = Arc::new(ScriptedBackend::new(vec![
            response_with_usage("m", 7_000),
            response_with_usage("m", 7_500),
        ]));
        let cfg = CompactConfig {
            phase1_threshold: 0.5, // trigger at 4096
            ..Default::default()
        };
        let backend = CompactingBackend::new(inner.clone(), 8192, cfg, false);

        let mut req = ChatCompletionRequest {
            model: "m".into(),
            messages: vec![user_message("hi")], // char/4 ≈ 0 tokens
            stream: false,
            max_tokens: None,
            max_completion_tokens: None,
            temperature: None,
            top_p: None,
            n: None,
            logprobs: None,
            top_logprobs: None,
            presence_penalty: None,
            frequency_penalty: None,
            logit_bias: None,
            response_format: None,
            tools: None,
            tool_choice: None,
            parallel_tool_calls: None,
            user: None,
            stop: None,
            seed: None,
            reasoning: None,
            reasoning_effort: None,
            prompt_cache_key: None,
            prompt_cache_retention: None,
            stream_options: None,
            extra: BTreeMap::new(),
        };

        // Turn 1: char/4 says we're empty; no compaction triggers.
        // (We assert via inner-call count not changing the request
        // structure — easier signal is that record_usage stored 7000.)
        let _ = backend.chat_completion(req.clone()).await.unwrap();
        let stored = backend
            .last_known_tokens
            .load(std::sync::atomic::Ordering::Relaxed);
        assert_eq!(stored, 7_000, "usage should have been recorded");

        // Turn 2: estimate_tokens now returns 7000 (≥ 4096 trigger),
        // so should_compact returns true even though the request
        // body is tiny.
        req.messages.push(user_message("more"));
        assert!(
            backend.should_compact(&req),
            "second turn should compact based on reported usage, not char/4"
        );
    }

    #[tokio::test]
    async fn maybe_wrap_engages_for_small_model() {
        let inner: Arc<dyn OpenAiBackend> =
            Arc::new(ScriptedBackend::new(vec![bare_text_response("m", "ok")]));
        let cfg = WrapConfig::default();
        // Small model name, large ctx → guardrails only.
        let wrapped = maybe_wrap_backend(inner.clone(), "Qwen3-8B", 131_072, &cfg);
        assert!(
            !Arc::ptr_eq(&wrapped, &inner),
            "small model should wrap at least with guardrails"
        );
    }
}
