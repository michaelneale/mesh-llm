use std::{
    pin::Pin,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

use async_trait::async_trait;
use futures_core::Stream;

use crate::{
    chat::{ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse},
    completions::{CompletionChunk, CompletionRequest, CompletionResponse},
    errors::OpenAiError,
    models::ModelObject,
};

pub type ChatCompletionStream =
    Pin<Box<dyn Stream<Item = OpenAiResult<ChatCompletionChunk>> + Send + 'static>>;
pub type CompletionStream =
    Pin<Box<dyn Stream<Item = OpenAiResult<CompletionChunk>> + Send + 'static>>;

pub type OpenAiResult<T> = Result<T, OpenAiError>;

#[derive(Debug, Clone, Default)]
pub struct CancellationToken {
    cancelled: Arc<AtomicBool>,
}

impl CancellationToken {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Relaxed);
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Relaxed)
    }
}

#[derive(Debug, Clone, Default)]
pub struct OpenAiRequestContext {
    cancellation: CancellationToken,
}

impl OpenAiRequestContext {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn cancellation_token(&self) -> CancellationToken {
        self.cancellation.clone()
    }

    pub fn cancel(&self) {
        self.cancellation.cancel();
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancellation.is_cancelled()
    }
}

#[async_trait]
pub trait OpenAiBackend: Send + Sync + 'static {
    async fn models(&self) -> OpenAiResult<Vec<ModelObject>>;

    async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> OpenAiResult<ChatCompletionResponse>;

    async fn chat_completion_stream(
        &self,
        request: ChatCompletionRequest,
        context: OpenAiRequestContext,
    ) -> OpenAiResult<ChatCompletionStream>;

    async fn completion(&self, _request: CompletionRequest) -> OpenAiResult<CompletionResponse> {
        Err(OpenAiError::unsupported(
            "/v1/completions is not supported by this backend",
        ))
    }

    async fn completion_stream(
        &self,
        _request: CompletionRequest,
        _context: OpenAiRequestContext,
    ) -> OpenAiResult<CompletionStream> {
        Err(OpenAiError::unsupported(
            "/v1/completions streaming is not supported by this backend",
        ))
    }
}

pub(crate) type SharedBackend = Arc<dyn OpenAiBackend>;
