use std::sync::Arc;

use async_trait::async_trait;
use openai_frontend::{
    chat_mesh_hooks_enabled, first_chat_media, inject_text_into_chat_messages,
    ChatCompletionRequest, ChatHookAction, ChatHookOutcome, ChatMediaKind, OpenAiHookPolicy,
    OpenAiResult,
};
use serde_json::Value;

use crate::{inference::virtual_llm, mesh};

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
        let outcome = virtual_hook_response_to_outcome(&response);
        apply_chat_hook_outcome(request, &outcome);
        Ok(outcome)
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
}
