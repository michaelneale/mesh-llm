use std::sync::Arc;

use async_trait::async_trait;
use openai_frontend::{
    chat_mesh_hooks_enabled, first_chat_media, inject_text_into_chat_messages,
    ChatCompletionRequest, ChatHookAction, ChatHookOutcome, ChatMediaKind, GenerationHookSignals,
    OpenAiHookPolicy, OpenAiResult, PrefillHookSignals,
};
use serde_json::Value;

use crate::{inference::virtual_llm, mesh};

const PREFILL_ENTROPY_THRESHOLD: f64 = 3.0;
const PREFILL_MARGIN_THRESHOLD: f64 = 0.05;
const MID_GENERATION_MIN_DECODED: i64 = 12;
const MID_GENERATION_REPETITION_THRESHOLD: u32 = 3;

#[derive(Clone)]
pub(crate) struct MeshAutoHookPolicy {
    executor: Arc<dyn VirtualHookExecutor>,
    debug: HookDebugConfig,
}

impl MeshAutoHookPolicy {
    pub(crate) fn new(node: mesh::Node) -> Arc<Self> {
        Arc::new(Self {
            executor: Arc::new(NodeVirtualHookExecutor { node }),
            debug: HookDebugConfig::from_env(),
        })
    }

    #[cfg(test)]
    fn new_with_executor(
        executor: Arc<dyn VirtualHookExecutor>,
        debug: HookDebugConfig,
    ) -> Arc<Self> {
        Arc::new(Self { executor, debug })
    }
}

#[async_trait]
impl OpenAiHookPolicy for MeshAutoHookPolicy {
    async fn before_chat_completion(
        &self,
        request: &mut ChatCompletionRequest,
    ) -> OpenAiResult<ChatHookOutcome> {
        if !chat_mesh_hooks_enabled(request) {
            return Ok(ChatHookOutcome::none());
        }

        if let Some(outcome) = self.debug.forced_outcome(HookPoint::BeforeChat) {
            return Ok(outcome);
        }

        let Some(media) = first_chat_media(&request.messages) else {
            return Ok(ChatHookOutcome::none());
        };

        let trigger = media_trigger(media.kind);
        let response = self
            .executor
            .handle_image(trigger, &request.model, &media.url, &media.user_text)
            .await;
        Ok(virtual_hook_response_to_outcome(&response))
    }

    async fn after_prefill(
        &self,
        request: &mut ChatCompletionRequest,
        signals: PrefillHookSignals,
    ) -> OpenAiResult<ChatHookOutcome> {
        if !chat_mesh_hooks_enabled(request) {
            return Ok(ChatHookOutcome::none());
        }

        if let Some(outcome) = self.debug.forced_outcome(HookPoint::AfterPrefill) {
            return Ok(outcome);
        }

        if signals.first_token_entropy <= PREFILL_ENTROPY_THRESHOLD
            || signals.first_token_margin >= PREFILL_MARGIN_THRESHOLD
        {
            return Ok(ChatHookOutcome::none());
        }

        let messages = chat_messages_as_values(&request.messages);
        let response = self
            .executor
            .handle_uncertain(
                &request.model,
                &messages,
                signals.first_token_entropy,
                signals.first_token_margin,
            )
            .await;
        Ok(virtual_hook_response_to_outcome(&response))
    }

    async fn mid_generation(
        &self,
        request: &mut ChatCompletionRequest,
        signals: GenerationHookSignals,
    ) -> OpenAiResult<ChatHookOutcome> {
        if !chat_mesh_hooks_enabled(request) {
            return Ok(ChatHookOutcome::none());
        }

        if let Some(outcome) = self.debug.forced_outcome(HookPoint::MidGeneration) {
            return Ok(outcome);
        }

        if signals.n_decoded < MID_GENERATION_MIN_DECODED
            || !mid_generation_signals_should_fire(&signals)
        {
            return Ok(ChatHookOutcome::none());
        }

        let messages = chat_messages_as_values(&request.messages);
        let response = self
            .executor
            .handle_drift(&request.model, &messages, signals.n_decoded)
            .await;
        Ok(virtual_hook_response_to_outcome(&response))
    }
}

#[async_trait]
trait VirtualHookExecutor: Send + Sync {
    async fn handle_image(
        &self,
        trigger: &str,
        model: &str,
        media_url: &str,
        user_text: &str,
    ) -> Value;

    async fn handle_uncertain(
        &self,
        model: &str,
        messages: &[Value],
        entropy: f64,
        margin: f64,
    ) -> Value;

    async fn handle_drift(&self, model: &str, messages: &[Value], n_decoded: i64) -> Value;
}

struct NodeVirtualHookExecutor {
    node: mesh::Node,
}

#[async_trait]
impl VirtualHookExecutor for NodeVirtualHookExecutor {
    async fn handle_image(
        &self,
        trigger: &str,
        model: &str,
        media_url: &str,
        user_text: &str,
    ) -> Value {
        virtual_llm::handle_image(&self.node, trigger, model, media_url, user_text).await
    }

    async fn handle_uncertain(
        &self,
        model: &str,
        messages: &[Value],
        entropy: f64,
        margin: f64,
    ) -> Value {
        virtual_llm::handle_uncertain(&self.node, model, messages, entropy, margin).await
    }

    async fn handle_drift(&self, model: &str, messages: &[Value], n_decoded: i64) -> Value {
        virtual_llm::handle_drift(&self.node, model, messages, n_decoded).await
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct HookDebugForce {
    before_chat: bool,
    after_prefill: bool,
    mid_generation: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct HookDebugConfig {
    force: HookDebugForce,
    injected_text: Option<String>,
}

impl HookDebugConfig {
    fn from_env() -> Self {
        Self {
            force: std::env::var("MESH_HOOK_DEBUG_FORCE")
                .ok()
                .map(|value| HookDebugForce::parse(&value))
                .unwrap_or_default(),
            injected_text: std::env::var("MESH_HOOK_DEBUG_TEXT").ok(),
        }
    }

    #[cfg(test)]
    fn force_all(text: impl Into<String>) -> Self {
        Self {
            force: HookDebugForce {
                before_chat: true,
                after_prefill: true,
                mid_generation: true,
            },
            injected_text: Some(text.into()),
        }
    }

    fn forced_outcome(&self, point: HookPoint) -> Option<ChatHookOutcome> {
        if !self.force.matches(point) {
            return None;
        }
        Some(ChatHookOutcome::injected(
            self.injected_text
                .clone()
                .unwrap_or_else(|| format!("[Mesh hook debug: forced {}]\n\n", point.label())),
        ))
    }
}

impl HookDebugForce {
    fn parse(value: &str) -> Self {
        let mut force = Self::default();
        for token in value
            .split([',', ';', ' ', '|'])
            .map(str::trim)
            .filter(|token| !token.is_empty())
        {
            match token.to_ascii_lowercase().as_str() {
                "1" | "true" | "all" => {
                    force.before_chat = true;
                    force.after_prefill = true;
                    force.mid_generation = true;
                }
                "pre_inference"
                | "before_chat"
                | "before_chat_completion"
                | "media"
                | "media_fallback" => force.before_chat = true,
                "post_prefill" | "after_prefill" | "uncertain" | "uncertainty" => {
                    force.after_prefill = true;
                }
                "mid_generation" | "drift" => force.mid_generation = true,
                _ => {
                    tracing::warn!("unknown MESH_HOOK_DEBUG_FORCE token: {token}");
                }
            }
        }
        force
    }

    fn matches(self, point: HookPoint) -> bool {
        match point {
            HookPoint::BeforeChat => self.before_chat,
            HookPoint::AfterPrefill => self.after_prefill,
            HookPoint::MidGeneration => self.mid_generation,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HookPoint {
    BeforeChat,
    AfterPrefill,
    MidGeneration,
}

impl HookPoint {
    fn label(self) -> &'static str {
        match self {
            HookPoint::BeforeChat => "pre_inference",
            HookPoint::AfterPrefill => "post_prefill",
            HookPoint::MidGeneration => "mid_generation",
        }
    }
}

fn media_trigger(kind: ChatMediaKind) -> &'static str {
    match kind {
        ChatMediaKind::Image => "images_no_multimodal",
        ChatMediaKind::Audio => "audio_no_support",
        ChatMediaKind::Video => "video_no_support",
    }
}

fn virtual_hook_response_to_outcome(response: &Value) -> ChatHookOutcome {
    match response.get("action").and_then(Value::as_str) {
        Some("inject") => response
            .get("text")
            .and_then(Value::as_str)
            .map(ChatHookOutcome::injected)
            .unwrap_or_else(ChatHookOutcome::none),
        _ => ChatHookOutcome::none(),
    }
}

fn chat_messages_as_values(messages: &[openai_frontend::ChatMessage]) -> Vec<Value> {
    serde_json::to_value(messages)
        .ok()
        .and_then(|value| value.as_array().cloned())
        .unwrap_or_default()
}

fn mid_generation_signals_should_fire(signals: &GenerationHookSignals) -> bool {
    let sustained_entropy = signals.window_tokens > 0
        && signals.high_entropy_count.saturating_mul(4) >= signals.window_tokens.saturating_mul(3);
    sustained_entropy || signals.repetition_count >= MID_GENERATION_REPETITION_THRESHOLD
}

fn apply_chat_hook_outcome(request: &mut ChatCompletionRequest, outcome: &ChatHookOutcome) {
    for action in &outcome.actions {
        match action {
            ChatHookAction::InjectText { text } => {
                inject_text_into_chat_messages(&mut request.messages, text.clone());
            }
            ChatHookAction::None => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use openai_frontend::{MessageContent, MessageContentPart};
    use serde_json::json;

    use super::*;

    #[derive(Debug, Clone, PartialEq)]
    enum RecordedHookCall {
        Image {
            trigger: String,
            model: String,
            media_url: String,
            user_text: String,
        },
        Uncertain {
            model: String,
            entropy: f64,
            margin: f64,
            messages_len: usize,
        },
        Drift {
            model: String,
            n_decoded: i64,
            messages_len: usize,
        },
    }

    #[derive(Default)]
    struct RecordingHookExecutor {
        calls: Mutex<Vec<RecordedHookCall>>,
    }

    impl RecordingHookExecutor {
        fn calls(&self) -> Vec<RecordedHookCall> {
            self.calls.lock().unwrap().clone()
        }
    }

    #[async_trait]
    impl VirtualHookExecutor for RecordingHookExecutor {
        async fn handle_image(
            &self,
            trigger: &str,
            model: &str,
            media_url: &str,
            user_text: &str,
        ) -> Value {
            self.calls.lock().unwrap().push(RecordedHookCall::Image {
                trigger: trigger.to_string(),
                model: model.to_string(),
                media_url: media_url.to_string(),
                user_text: user_text.to_string(),
            });
            json!({"action": "inject", "text": "[media fallback]\n\n"})
        }

        async fn handle_uncertain(
            &self,
            model: &str,
            messages: &[Value],
            entropy: f64,
            margin: f64,
        ) -> Value {
            self.calls
                .lock()
                .unwrap()
                .push(RecordedHookCall::Uncertain {
                    model: model.to_string(),
                    entropy,
                    margin,
                    messages_len: messages.len(),
                });
            json!({"action": "inject", "text": "\n\nReference answer: uncertain\n\n"})
        }

        async fn handle_drift(&self, model: &str, messages: &[Value], n_decoded: i64) -> Value {
            self.calls.lock().unwrap().push(RecordedHookCall::Drift {
                model: model.to_string(),
                n_decoded,
                messages_len: messages.len(),
            });
            json!({"action": "inject", "text": "\n\nReference answer: drift\n\n"})
        }
    }

    fn policy_with_recorder(
        debug: HookDebugConfig,
    ) -> (Arc<MeshAutoHookPolicy>, Arc<RecordingHookExecutor>) {
        let executor = Arc::new(RecordingHookExecutor::default());
        (
            MeshAutoHookPolicy::new_with_executor(executor.clone(), debug),
            executor,
        )
    }

    fn text_request(mesh_hooks: bool) -> ChatCompletionRequest {
        serde_json::from_value(json!({
            "model": "auto",
            "messages": [{"role": "user", "content": "hello"}],
            "mesh_hooks": mesh_hooks
        }))
        .unwrap()
    }

    fn image_request(mesh_hooks: bool) -> ChatCompletionRequest {
        serde_json::from_value(json!({
            "model": "auto",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is this?"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}
                ]
            }],
            "mesh_hooks": mesh_hooks
        }))
        .unwrap()
    }

    fn uncertain_signals() -> PrefillHookSignals {
        PrefillHookSignals {
            first_token_entropy: PREFILL_ENTROPY_THRESHOLD + 0.1,
            first_token_margin: PREFILL_MARGIN_THRESHOLD - 0.01,
        }
    }

    fn calm_prefill_signals() -> PrefillHookSignals {
        PrefillHookSignals {
            first_token_entropy: 0.1,
            first_token_margin: 0.9,
        }
    }

    fn drift_signals() -> GenerationHookSignals {
        GenerationHookSignals {
            n_decoded: MID_GENERATION_MIN_DECODED,
            window_tokens: 16,
            mean_entropy: 4.2,
            max_entropy: 5.1,
            mean_margin: 0.02,
            min_margin: 0.01,
            high_entropy_count: 12,
            repetition_count: 0,
        }
    }

    fn calm_generation_signals() -> GenerationHookSignals {
        GenerationHookSignals {
            n_decoded: 1,
            window_tokens: 16,
            mean_entropy: 0.2,
            max_entropy: 0.4,
            mean_margin: 0.8,
            min_margin: 0.7,
            high_entropy_count: 0,
            repetition_count: 0,
        }
    }

    #[test]
    fn virtual_hook_response_to_outcome_maps_inject_action() {
        let outcome = virtual_hook_response_to_outcome(&json!({
            "action": "inject",
            "text": "[Image description: cat]\n\n"
        }));

        assert_eq!(
            outcome,
            ChatHookOutcome::injected("[Image description: cat]\n\n")
        );
    }

    #[test]
    fn apply_chat_hook_outcome_injects_into_typed_messages() {
        let mut request: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "auto",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is this?"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}
                ]
            }],
            "mesh_hooks": true
        }))
        .unwrap();

        apply_chat_hook_outcome(
            &mut request,
            &ChatHookOutcome::injected("[Image description: cat]\n\n"),
        );

        let Some(MessageContent::Parts(parts)) = &request.messages[0].content else {
            panic!("expected multipart content");
        };
        assert_eq!(
            parts.first(),
            Some(&MessageContentPart {
                content_type: "text".to_string(),
                text: Some("[Image description: cat]\n\n".to_string()),
                extra: Default::default(),
            })
        );
    }

    #[test]
    fn media_trigger_matches_legacy_hook_triggers() {
        assert_eq!(media_trigger(ChatMediaKind::Image), "images_no_multimodal");
        assert_eq!(media_trigger(ChatMediaKind::Audio), "audio_no_support");
        assert_eq!(media_trigger(ChatMediaKind::Video), "video_no_support");
    }

    #[tokio::test]
    async fn mesh_hooks_disabled_skips_all_skippy_hook_points() {
        let (policy, executor) = policy_with_recorder(HookDebugConfig::force_all("[forced]\n"));
        let mut request = image_request(false);

        assert_eq!(
            policy.before_chat_completion(&mut request).await.unwrap(),
            ChatHookOutcome::none()
        );
        assert_eq!(
            policy
                .after_prefill(&mut request, uncertain_signals())
                .await
                .unwrap(),
            ChatHookOutcome::none()
        );
        assert_eq!(
            policy
                .mid_generation(&mut request, drift_signals())
                .await
                .unwrap(),
            ChatHookOutcome::none()
        );
        assert!(executor.calls().is_empty());
    }

    #[tokio::test]
    async fn media_fallback_hook_calls_legacy_trigger_and_injects() {
        let (policy, executor) = policy_with_recorder(HookDebugConfig::default());
        let mut request = image_request(true);

        let outcome = policy.before_chat_completion(&mut request).await.unwrap();

        assert_eq!(outcome, ChatHookOutcome::injected("[media fallback]\n\n"));
        assert_eq!(
            executor.calls(),
            vec![RecordedHookCall::Image {
                trigger: "images_no_multimodal".to_string(),
                model: "auto".to_string(),
                media_url: "data:image/png;base64,abc".to_string(),
                user_text: "what is this?".to_string(),
            }]
        );
    }

    #[tokio::test]
    async fn uncertainty_hook_calls_executor_above_thresholds() {
        let (policy, executor) = policy_with_recorder(HookDebugConfig::default());
        let mut request = text_request(true);

        let outcome = policy
            .after_prefill(&mut request, uncertain_signals())
            .await
            .unwrap();

        assert_eq!(
            outcome,
            ChatHookOutcome::injected("\n\nReference answer: uncertain\n\n")
        );
        assert_eq!(
            executor.calls(),
            vec![RecordedHookCall::Uncertain {
                model: "auto".to_string(),
                entropy: PREFILL_ENTROPY_THRESHOLD + 0.1,
                margin: PREFILL_MARGIN_THRESHOLD - 0.01,
                messages_len: 1,
            }]
        );
    }

    #[tokio::test]
    async fn uncertainty_hook_ignores_calm_prefill_without_debug_force() {
        let (policy, executor) = policy_with_recorder(HookDebugConfig::default());
        let mut request = text_request(true);

        let outcome = policy
            .after_prefill(&mut request, calm_prefill_signals())
            .await
            .unwrap();

        assert_eq!(outcome, ChatHookOutcome::none());
        assert!(executor.calls().is_empty());
    }

    #[tokio::test]
    async fn drift_hook_calls_executor_for_generation_window() {
        let (policy, executor) = policy_with_recorder(HookDebugConfig::default());
        let mut request = text_request(true);

        let outcome = policy
            .mid_generation(&mut request, drift_signals())
            .await
            .unwrap();

        assert_eq!(
            outcome,
            ChatHookOutcome::injected("\n\nReference answer: drift\n\n")
        );
        assert_eq!(
            executor.calls(),
            vec![RecordedHookCall::Drift {
                model: "auto".to_string(),
                n_decoded: MID_GENERATION_MIN_DECODED,
                messages_len: 1,
            }]
        );
    }

    #[tokio::test]
    async fn drift_hook_ignores_calm_generation_without_debug_force() {
        let (policy, executor) = policy_with_recorder(HookDebugConfig::default());
        let mut request = text_request(true);

        let outcome = policy
            .mid_generation(&mut request, calm_generation_signals())
            .await
            .unwrap();

        assert_eq!(outcome, ChatHookOutcome::none());
        assert!(executor.calls().is_empty());
    }

    #[tokio::test]
    async fn debug_force_injects_without_media_or_signal_thresholds() {
        let (policy, executor) = policy_with_recorder(HookDebugConfig::force_all("[forced]\n"));
        let mut request = text_request(true);

        assert_eq!(
            policy.before_chat_completion(&mut request).await.unwrap(),
            ChatHookOutcome::injected("[forced]\n")
        );
        assert_eq!(
            policy
                .after_prefill(&mut request, calm_prefill_signals())
                .await
                .unwrap(),
            ChatHookOutcome::injected("[forced]\n")
        );
        assert_eq!(
            policy
                .mid_generation(&mut request, calm_generation_signals())
                .await
                .unwrap(),
            ChatHookOutcome::injected("[forced]\n")
        );
        assert!(executor.calls().is_empty());
    }

    #[test]
    fn debug_force_parses_legacy_and_skippy_hook_names() {
        let force = HookDebugForce::parse(
            "pre_inference,post_prefill,mid_generation,media_fallback,uncertainty,drift",
        );

        assert!(force.before_chat);
        assert!(force.after_prefill);
        assert!(force.mid_generation);
    }

    #[test]
    fn mid_generation_signals_fire_on_sustained_entropy() {
        assert!(mid_generation_signals_should_fire(&GenerationHookSignals {
            n_decoded: 16,
            window_tokens: 16,
            mean_entropy: 4.2,
            max_entropy: 5.1,
            mean_margin: 0.02,
            min_margin: 0.01,
            high_entropy_count: 12,
            repetition_count: 0,
        }));
    }

    #[test]
    fn mid_generation_signals_fire_on_repetition() {
        assert!(mid_generation_signals_should_fire(&GenerationHookSignals {
            n_decoded: 16,
            window_tokens: 16,
            mean_entropy: 0.4,
            max_entropy: 0.8,
            mean_margin: 0.6,
            min_margin: 0.3,
            high_entropy_count: 0,
            repetition_count: 3,
        }));
    }

    #[test]
    fn mid_generation_signals_ignore_calm_window() {
        assert!(!mid_generation_signals_should_fire(
            &GenerationHookSignals {
                n_decoded: 16,
                window_tokens: 16,
                mean_entropy: 0.4,
                max_entropy: 0.8,
                mean_margin: 0.6,
                min_margin: 0.3,
                high_entropy_count: 1,
                repetition_count: 0,
            }
        ));
    }
}
