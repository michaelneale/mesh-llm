use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use crate::{
    backend::{
        ChatCompletionStream, CompletionStream, OpenAiBackend, OpenAiRequestContext, OpenAiResult,
    },
    chat::{
        ChatCompletionRequest, ChatCompletionResponse, ChatMessage, MessageContent,
        MessageContentPart,
    },
    completions::{CompletionRequest, CompletionResponse},
    models::ModelObject,
};

pub const MESH_HOOKS_FIELD: &str = "mesh_hooks";

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ChatHookOutcome {
    pub actions: Vec<ChatHookAction>,
}

impl ChatHookOutcome {
    pub fn none() -> Self {
        Self::default()
    }

    pub fn injected(text: impl Into<String>) -> Self {
        Self {
            actions: vec![ChatHookAction::InjectText { text: text.into() }],
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChatHookAction {
    InjectText { text: String },
    None,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PrefillHookSignals {
    pub first_token_entropy: f64,
    pub first_token_margin: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GenerationHookSignals {
    pub n_decoded: i64,
    pub window_tokens: u32,
    pub mean_entropy: f64,
    pub max_entropy: f64,
    pub mean_margin: f64,
    pub min_margin: f64,
    pub high_entropy_count: u32,
    pub repetition_count: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatMediaRef {
    pub kind: ChatMediaKind,
    pub url: String,
    pub user_text: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatMediaKind {
    Image,
    Audio,
    Video,
}

#[async_trait]
pub trait OpenAiHookPolicy: Send + Sync + 'static {
    async fn before_chat_completion(
        &self,
        _request: &mut ChatCompletionRequest,
    ) -> OpenAiResult<ChatHookOutcome> {
        Ok(ChatHookOutcome::none())
    }

    async fn after_prefill(
        &self,
        _request: &mut ChatCompletionRequest,
        _signals: PrefillHookSignals,
    ) -> OpenAiResult<ChatHookOutcome> {
        Ok(ChatHookOutcome::none())
    }

    async fn mid_generation(
        &self,
        _request: &mut ChatCompletionRequest,
        _signals: GenerationHookSignals,
    ) -> OpenAiResult<ChatHookOutcome> {
        Ok(ChatHookOutcome::none())
    }
}

pub struct HookedOpenAiBackend {
    backend: Arc<dyn OpenAiBackend>,
    hooks: Arc<dyn OpenAiHookPolicy>,
}

impl HookedOpenAiBackend {
    pub fn new(backend: Arc<dyn OpenAiBackend>, hooks: Arc<dyn OpenAiHookPolicy>) -> Self {
        Self { backend, hooks }
    }
}

#[async_trait]
impl OpenAiBackend for HookedOpenAiBackend {
    async fn models(&self) -> OpenAiResult<Vec<ModelObject>> {
        self.backend.models().await
    }

    async fn chat_completion(
        &self,
        mut request: ChatCompletionRequest,
    ) -> OpenAiResult<ChatCompletionResponse> {
        let outcome = self.hooks.before_chat_completion(&mut request).await?;
        apply_chat_hook_outcome(&mut request, &outcome);
        self.backend.chat_completion(request).await
    }

    async fn chat_completion_stream(
        &self,
        mut request: ChatCompletionRequest,
        context: OpenAiRequestContext,
    ) -> OpenAiResult<ChatCompletionStream> {
        let outcome = self.hooks.before_chat_completion(&mut request).await?;
        apply_chat_hook_outcome(&mut request, &outcome);
        self.backend.chat_completion_stream(request, context).await
    }

    async fn completion(&self, request: CompletionRequest) -> OpenAiResult<CompletionResponse> {
        self.backend.completion(request).await
    }

    async fn completion_stream(
        &self,
        request: CompletionRequest,
        context: OpenAiRequestContext,
    ) -> OpenAiResult<CompletionStream> {
        self.backend.completion_stream(request, context).await
    }
}

pub fn chat_mesh_hooks_enabled(request: &ChatCompletionRequest) -> bool {
    request
        .extra
        .get(MESH_HOOKS_FIELD)
        .and_then(Value::as_bool)
        .unwrap_or(false)
}

pub fn set_chat_mesh_hooks_enabled(request: &mut ChatCompletionRequest, enabled: bool) {
    request
        .extra
        .insert(MESH_HOOKS_FIELD.to_string(), Value::Bool(enabled));
}

pub fn inject_text_into_chat_messages(messages: &mut Vec<ChatMessage>, text: impl Into<String>) {
    let text = text.into();
    if text.is_empty() {
        return;
    }

    if let Some(message) = messages
        .iter_mut()
        .rev()
        .find(|message| message.role == "user")
    {
        inject_text_into_message(message, text);
    } else {
        messages.push(ChatMessage {
            role: "user".to_string(),
            content: Some(MessageContent::Text(text)),
            extra: Default::default(),
        });
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

pub fn first_chat_media(messages: &[ChatMessage]) -> Option<ChatMediaRef> {
    messages
        .iter()
        .rev()
        .find(|message| message.role == "user")
        .and_then(media_from_message)
}

fn inject_text_into_message(message: &mut ChatMessage, text: String) {
    match message.content.take() {
        Some(MessageContent::Text(existing)) => {
            message.content = Some(MessageContent::Text(format!("{text}{existing}")));
        }
        Some(MessageContent::Parts(mut parts)) => {
            parts.insert(
                0,
                MessageContentPart {
                    content_type: "text".to_string(),
                    text: Some(text),
                    extra: Default::default(),
                },
            );
            message.content = Some(MessageContent::Parts(parts));
        }
        Some(MessageContent::Other(_)) | None => {
            message.content = Some(MessageContent::Text(text));
        }
    }
}

fn media_from_message(message: &ChatMessage) -> Option<ChatMediaRef> {
    let parts = match message.content.as_ref()? {
        MessageContent::Parts(parts) => parts,
        MessageContent::Text(_) | MessageContent::Other(_) => return None,
    };
    let user_text = parts
        .iter()
        .filter(|part| part.content_type == "text")
        .filter_map(|part| part.text.as_deref())
        .collect::<Vec<_>>()
        .join("\n");
    for part in parts {
        if let Some(media) = media_from_part(part, &user_text) {
            return Some(media);
        }
    }
    None
}

fn media_from_part(part: &MessageContentPart, user_text: &str) -> Option<ChatMediaRef> {
    let kind = match part.content_type.as_str() {
        "image_url" | "input_image" | "image" => ChatMediaKind::Image,
        "input_audio" | "audio" | "audio_url" => ChatMediaKind::Audio,
        "input_video" | "video" | "video_url" => ChatMediaKind::Video,
        _ => return None,
    };
    let url = media_url(part)?;
    Some(ChatMediaRef {
        kind,
        url,
        user_text: user_text.to_string(),
    })
}

fn media_url(part: &MessageContentPart) -> Option<String> {
    for key in ["image_url", "input_image", "image", "audio", "video", "url"] {
        if let Some(value) = part.extra.get(key) {
            if let Some(url) = value.as_str() {
                return Some(url.to_string());
            }
            if let Some(url) = value.get("url").and_then(Value::as_str) {
                return Some(url.to_string());
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use serde_json::json;

    use super::*;
    use crate::Usage;

    struct RecordingBackend {
        seen: Mutex<Option<ChatCompletionRequest>>,
    }

    #[async_trait]
    impl OpenAiBackend for RecordingBackend {
        async fn models(&self) -> OpenAiResult<Vec<ModelObject>> {
            Ok(vec![ModelObject::new("auto")])
        }

        async fn chat_completion(
            &self,
            request: ChatCompletionRequest,
        ) -> OpenAiResult<ChatCompletionResponse> {
            *self.seen.lock().unwrap() = Some(request.clone());
            Ok(ChatCompletionResponse::new(
                request.model,
                "ok",
                Usage::new(0, 0),
            ))
        }

        async fn chat_completion_stream(
            &self,
            request: ChatCompletionRequest,
            _context: OpenAiRequestContext,
        ) -> OpenAiResult<ChatCompletionStream> {
            *self.seen.lock().unwrap() = Some(request);
            Ok(Box::pin(futures_util::stream::empty()))
        }
    }

    struct InjectingHook;

    #[async_trait]
    impl OpenAiHookPolicy for InjectingHook {
        async fn before_chat_completion(
            &self,
            _request: &mut ChatCompletionRequest,
        ) -> OpenAiResult<ChatHookOutcome> {
            Ok(ChatHookOutcome::injected("[hint]\n"))
        }
    }

    #[test]
    fn chat_mesh_hooks_enabled_reads_extra_flag() {
        let mut request: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "auto",
            "messages": [{"role": "user", "content": "hello"}],
            "mesh_hooks": true
        }))
        .unwrap();

        assert!(chat_mesh_hooks_enabled(&request));

        set_chat_mesh_hooks_enabled(&mut request, false);

        assert!(!chat_mesh_hooks_enabled(&request));
    }

    #[test]
    fn first_chat_media_extracts_image_url_and_user_text() {
        let request: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "auto",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is this?"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}
                ]
            }]
        }))
        .unwrap();

        let media = first_chat_media(&request.messages).expect("media");

        assert_eq!(media.kind, ChatMediaKind::Image);
        assert_eq!(media.url, "data:image/png;base64,abc");
        assert_eq!(media.user_text, "what is this?");
    }

    #[test]
    fn image_only_message_with_mesh_hooks_is_valid_before_hook_injection() {
        let request: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "auto",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}
                ]
            }],
            "mesh_hooks": true
        }))
        .unwrap();

        request.validate().unwrap();
    }

    #[test]
    fn inject_text_into_chat_messages_prepends_last_user_text() {
        let mut request: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "auto",
            "messages": [{"role": "user", "content": "original"}]
        }))
        .unwrap();

        inject_text_into_chat_messages(&mut request.messages, "[hint]\n");

        assert_eq!(
            request.messages[0].content,
            Some(MessageContent::Text("[hint]\noriginal".to_string()))
        );
    }

    #[tokio::test]
    async fn hooked_backend_applies_injection_once_before_forwarding() {
        let backend = Arc::new(RecordingBackend {
            seen: Mutex::new(None),
        });
        let hooked = HookedOpenAiBackend::new(backend.clone(), Arc::new(InjectingHook));
        let request: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "auto",
            "messages": [{"role": "user", "content": "original"}],
            "mesh_hooks": true
        }))
        .unwrap();

        hooked.chat_completion(request).await.unwrap();

        let seen = backend.seen.lock().unwrap().clone().unwrap();
        assert_eq!(
            seen.messages[0].content,
            Some(MessageContent::Text("[hint]\noriginal".to_string()))
        );
    }
}
