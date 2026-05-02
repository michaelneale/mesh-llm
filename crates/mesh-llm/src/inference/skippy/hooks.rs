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
    node: mesh::Node,
}

impl MeshAutoHookPolicy {
    pub(crate) fn new(node: mesh::Node) -> Arc<Self> {
        Arc::new(Self { node })
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

        let Some(media) = first_chat_media(&request.messages) else {
            return Ok(ChatHookOutcome::none());
        };

        let trigger = media_trigger(media.kind);
        let response = virtual_llm::handle_image(
            &self.node,
            trigger,
            &request.model,
            &media.url,
            &media.user_text,
        )
        .await;
        Ok(virtual_hook_response_to_outcome(&response))
    }

    async fn after_prefill(
        &self,
        request: &mut ChatCompletionRequest,
        signals: PrefillHookSignals,
    ) -> OpenAiResult<ChatHookOutcome> {
        if !chat_mesh_hooks_enabled(request)
            || signals.first_token_entropy <= PREFILL_ENTROPY_THRESHOLD
            || signals.first_token_margin >= PREFILL_MARGIN_THRESHOLD
        {
            return Ok(ChatHookOutcome::none());
        }

        let messages = chat_messages_as_values(&request.messages);
        let response = virtual_llm::handle_uncertain(
            &self.node,
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
        if !chat_mesh_hooks_enabled(request)
            || signals.n_decoded < MID_GENERATION_MIN_DECODED
            || !mid_generation_signals_should_fire(&signals)
        {
            return Ok(ChatHookOutcome::none());
        }

        let messages = chat_messages_as_values(&request.messages);
        let response =
            virtual_llm::handle_drift(&self.node, &request.model, &messages, signals.n_decoded)
                .await;
        Ok(virtual_hook_response_to_outcome(&response))
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
    use openai_frontend::{MessageContent, MessageContentPart};
    use serde_json::json;

    use super::*;

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
