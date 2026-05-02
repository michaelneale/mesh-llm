use std::{
    convert::Infallible,
    future::Future,
    pin::Pin,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex,
    },
    task::{Context, Poll},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use axum::{
    body::Body,
    extract::{rejection::JsonRejection, DefaultBodyLimit, State},
    http::{header::HeaderName, HeaderValue, Method, Request, StatusCode, Uri},
    middleware::{self, Next},
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Response,
    },
    routing::{get, post},
    Json, Router,
};
use futures_util::{stream, Stream, StreamExt};
use serde::Serialize;
use serde_json::Value;

use crate::{
    backend::{
        CancellationToken, OpenAiBackend, OpenAiRequestContext, OpenAiResult, SharedBackend,
    },
    chat::{ChatCompletionChunk, ChatCompletionRequest},
    completions::CompletionRequest,
    errors::OpenAiError,
    models::ModelsResponse,
    responses::{
        chunk_delta_text, normalize_openai_compat_request, responses_stream_completed_event,
        responses_stream_created_event, responses_stream_delta_event_with_logprobs,
        responses_stream_text_done_event, translate_chat_completion_response_to_responses,
        usage_to_responses_usage, ResponseAdapterMode, ResponseSseState,
    },
    sse::{done_event, json_event},
};

#[derive(Clone)]
struct FrontendState {
    backend: SharedBackend,
    config: OpenAiFrontendConfig,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OpenAiFrontendConfig {
    pub max_request_body_bytes: usize,
    pub backend_timeout: Option<Duration>,
}

impl OpenAiFrontendConfig {
    pub const DEFAULT_MAX_REQUEST_BODY_BYTES: usize = 4 * 1024 * 1024;
    pub const DEFAULT_BACKEND_TIMEOUT: Duration = Duration::from_secs(300);

    pub fn with_max_request_body_bytes(mut self, max_request_body_bytes: usize) -> Self {
        self.max_request_body_bytes = max_request_body_bytes;
        self
    }

    pub fn with_backend_timeout(mut self, backend_timeout: Duration) -> Self {
        self.backend_timeout = Some(backend_timeout);
        self
    }

    pub fn without_backend_timeout(mut self) -> Self {
        self.backend_timeout = None;
        self
    }
}

impl Default for OpenAiFrontendConfig {
    fn default() -> Self {
        Self {
            max_request_body_bytes: Self::DEFAULT_MAX_REQUEST_BODY_BYTES,
            backend_timeout: Some(Self::DEFAULT_BACKEND_TIMEOUT),
        }
    }
}

pub fn router<B>(backend: Arc<B>) -> Router
where
    B: OpenAiBackend,
{
    router_for(backend)
}

pub fn router_for(backend: Arc<dyn OpenAiBackend>) -> Router {
    router_for_with_config(backend, OpenAiFrontendConfig::default())
}

pub fn router_with_config<B>(backend: Arc<B>, config: OpenAiFrontendConfig) -> Router
where
    B: OpenAiBackend,
{
    router_for_with_config(backend, config)
}

pub fn router_for_with_config(
    backend: Arc<dyn OpenAiBackend>,
    config: OpenAiFrontendConfig,
) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/healthz", get(health))
        .route("/readyz", get(ready))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .route("/v1/responses", post(responses))
        .method_not_allowed_fallback(method_not_allowed)
        .fallback(not_found)
        .layer(middleware::from_fn(request_id_middleware))
        .layer(DefaultBodyLimit::max(config.max_request_body_bytes))
        .with_state(FrontendState { backend, config })
}

#[derive(Debug, Clone, Copy, Serialize)]
struct HealthResponse {
    status: &'static str,
}

async fn health() -> Json<HealthResponse> {
    Json(HealthResponse { status: "ok" })
}

async fn ready(State(state): State<FrontendState>) -> Result<Json<HealthResponse>, OpenAiError> {
    backend_call(&state, "models", state.backend.models()).await?;
    Ok(Json(HealthResponse { status: "ready" }))
}

async fn models(State(state): State<FrontendState>) -> Result<Json<ModelsResponse>, OpenAiError> {
    let data = backend_call(&state, "models", state.backend.models()).await?;
    Ok(Json(ModelsResponse {
        object: "list",
        data,
    }))
}

async fn chat_completions(
    State(state): State<FrontendState>,
    payload: Result<Json<ChatCompletionRequest>, JsonRejection>,
) -> Result<Response, OpenAiError> {
    let Json(request) = json_payload(payload)?;
    request.validate()?;
    if request.stream {
        let include_usage = request.include_usage();
        let model = request.model.clone();
        let context = OpenAiRequestContext::new();
        let cancellation = context.cancellation_token();
        let stream = backend_call(
            &state,
            "chat_completion_stream",
            state.backend.chat_completion_stream(request, context),
        )
        .await?;
        let prelude = stream::once(async move { json_event(&ChatCompletionChunk::role(model)) });
        let events = prelude
            .chain(stream.filter_map(move |item| async move {
                match item {
                    Ok(chunk) if !include_usage && chunk.usage.is_some() => None,
                    Ok(chunk) => Some(json_event(&chunk)),
                    Err(error) => Some(json_event(&error.body())),
                }
            }))
            .chain(stream::once(async { done_event() }));
        Ok(sse_response(events, cancellation))
    } else {
        Ok(Json(
            backend_call(
                &state,
                "chat_completion",
                state.backend.chat_completion(request),
            )
            .await?,
        )
        .into_response())
    }
}

async fn responses(
    State(state): State<FrontendState>,
    payload: Result<Json<Value>, JsonRejection>,
) -> Result<Response, OpenAiError> {
    let Json(mut value) = json_payload(payload)?;
    let normalization = normalize_openai_compat_request("/v1/responses", &mut value)?;
    let request: ChatCompletionRequest = serde_json::from_value(value).map_err(|error| {
        OpenAiError::invalid_request(format!("invalid Responses request: {error}"))
    })?;
    request.validate()?;
    match normalization.response_adapter {
        ResponseAdapterMode::OpenAiResponsesStream => {
            let context = OpenAiRequestContext::new();
            let cancellation = context.cancellation_token();
            let state_machine = Arc::new(Mutex::new(ResponseSseState::new(request.model.clone())));
            let stream = backend_call(
                &state,
                "responses_stream",
                state.backend.chat_completion_stream(request, context),
            )
            .await?;
            let body_state = state_machine.clone();
            let body_events = stream.flat_map(move |item| {
                let mut out = Vec::new();
                let mut state_machine = body_state
                    .lock()
                    .expect("responses stream state lock poisoned");
                match item {
                    Ok(chunk) => {
                        if !state_machine.created_emitted {
                            state_machine.model = chunk.model.clone();
                            out.push(
                                Event::default()
                                    .event("response.created")
                                    .json_data(&responses_stream_created_event(
                                        &state_machine.model,
                                        state_machine.created_at,
                                    ))
                                    .unwrap_or_else(|_| Event::default().data("{}")),
                            );
                            state_machine.created_emitted = true;
                        }
                        if let Some(delta) = chunk_delta_text(&chunk) {
                            let logprobs = chunk
                                .choices
                                .first()
                                .and_then(|choice| choice.logprobs.clone());
                            state_machine.output_text.push_str(&delta);
                            out.push(
                                Event::default()
                                    .event("response.output_text.delta")
                                    .json_data(&responses_stream_delta_event_with_logprobs(
                                        &state_machine.item_id,
                                        &delta,
                                        logprobs,
                                    ))
                                    .unwrap_or_else(|_| Event::default().data("{}")),
                            );
                        }
                        if let Some(usage) = chunk.usage.as_ref() {
                            state_machine.usage = Some(usage_to_responses_usage(usage));
                        }
                    }
                    Err(error) => {
                        out.push(
                            Event::default()
                                .event("error")
                                .json_data(&error.body())
                                .unwrap_or_else(|_| Event::default().data("{}")),
                        );
                    }
                }
                stream::iter(out.into_iter().map(Ok::<_, Infallible>))
            });
            let tail_events = stream::once(async move {
                let mut state_machine = state_machine
                    .lock()
                    .expect("responses stream state lock poisoned");
                let mut out = Vec::new();
                if !state_machine.created_emitted {
                    out.push(
                        Event::default()
                            .event("response.created")
                            .json_data(&responses_stream_created_event(
                                &state_machine.model,
                                state_machine.created_at,
                            ))
                            .unwrap_or_else(|_| Event::default().data("{}")),
                    );
                    state_machine.created_emitted = true;
                }
                out.push(
                    Event::default()
                        .event("response.output_text.done")
                        .json_data(&responses_stream_text_done_event(
                            &state_machine.item_id,
                            &state_machine.output_text,
                        ))
                        .unwrap_or_else(|_| Event::default().data("{}")),
                );
                out.push(
                    Event::default()
                        .event("response.completed")
                        .json_data(&responses_stream_completed_event(
                            &state_machine.response_id,
                            state_machine.created_at,
                            &state_machine.model,
                            &state_machine.item_id,
                            &state_machine.output_text,
                            state_machine.usage.clone(),
                        ))
                        .unwrap_or_else(|_| Event::default().data("{}")),
                );
                out
            })
            .flat_map(|out| stream::iter(out.into_iter().map(Ok::<_, Infallible>)));
            let events = body_events
                .chain(tail_events)
                .chain(stream::once(async { done_event() }));
            Ok(sse_response(events, cancellation))
        }
        _ => {
            let response =
                backend_call(&state, "responses", state.backend.chat_completion(request)).await?;
            let translated = translate_chat_completion_response_to_responses(&response)?;
            Ok(Json(translated).into_response())
        }
    }
}

async fn completions(
    State(state): State<FrontendState>,
    payload: Result<Json<CompletionRequest>, JsonRejection>,
) -> Result<Response, OpenAiError> {
    let Json(request) = json_payload(payload)?;
    request.validate()?;
    if request.stream {
        let include_usage = request.include_usage();
        let context = OpenAiRequestContext::new();
        let cancellation = context.cancellation_token();
        let stream = backend_call(
            &state,
            "completion_stream",
            state.backend.completion_stream(request, context),
        )
        .await?;
        let events = stream
            .filter_map(move |item| async move {
                match item {
                    Ok(chunk) if !include_usage && chunk.usage.is_some() => None,
                    Ok(chunk) => Some(json_event(&chunk)),
                    Err(error) => Some(json_event(&error.body())),
                }
            })
            .chain(stream::once(async { done_event() }));
        Ok(sse_response(events, cancellation))
    } else {
        Ok(
            Json(backend_call(&state, "completion", state.backend.completion(request)).await?)
                .into_response(),
        )
    }
}

async fn backend_call<T, F>(
    state: &FrontendState,
    operation: &'static str,
    future: F,
) -> OpenAiResult<T>
where
    F: Future<Output = OpenAiResult<T>>,
{
    match state.config.backend_timeout {
        Some(timeout) => tokio::time::timeout(timeout, future).await.map_err(|_| {
            OpenAiError::timeout(format!(
                "{operation} timed out after {} ms",
                timeout.as_millis()
            ))
        })?,
        None => future.await,
    }
}

fn json_payload<T>(payload: Result<Json<T>, JsonRejection>) -> Result<Json<T>, OpenAiError> {
    payload.map_err(|rejection| {
        if rejection.status() == StatusCode::PAYLOAD_TOO_LARGE {
            return OpenAiError::payload_too_large(format!("request body too large: {rejection}"));
        }
        OpenAiError::invalid_request(format!("invalid JSON request body: {rejection}"))
    })
}

async fn not_found(uri: Uri) -> OpenAiError {
    OpenAiError::route_not_found(uri)
}

async fn method_not_allowed(method: Method) -> OpenAiError {
    OpenAiError::method_not_allowed(method)
}

static REQUEST_COUNTER: AtomicU64 = AtomicU64::new(1);
static REQUEST_ID_HEADER: HeaderName = HeaderName::from_static("x-request-id");

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequestId(pub String);

async fn request_id_middleware(mut request: Request<Body>, next: Next) -> Response {
    let request_id = request_id_from_headers(request.headers()).unwrap_or_else(new_request_id);
    let method = request.method().clone();
    let uri = request.uri().clone();
    request
        .extensions_mut()
        .insert(RequestId(request_id.clone()));

    let mut response = next.run(request).await;
    if let Ok(value) = HeaderValue::from_str(&request_id) {
        response
            .headers_mut()
            .insert(REQUEST_ID_HEADER.clone(), value);
    }
    tracing::info!(
        request_id = %request_id,
        method = %method,
        uri = %uri,
        status = %response.status(),
        "openai frontend request"
    );
    response
}

fn request_id_from_headers(headers: &axum::http::HeaderMap) -> Option<String> {
    headers
        .get(&REQUEST_ID_HEADER)
        .and_then(|value| value.to_str().ok())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToString::to_string)
}

fn new_request_id() -> String {
    let counter = REQUEST_COUNTER.fetch_add(1, Ordering::Relaxed);
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    format!("req-{millis}-{counter}")
}

fn sse_response<S>(events: S, cancellation: CancellationToken) -> Response
where
    S: Stream<Item = Result<Event, Infallible>> + Send + 'static,
{
    Sse::new(CancelOnDropSseStream::new(events, cancellation))
        .keep_alive(KeepAlive::default())
        .into_response()
}

struct CancelOnDropSseStream {
    inner: Pin<Box<dyn Stream<Item = Result<Event, Infallible>> + Send + 'static>>,
    cancellation: CancellationToken,
}

impl CancelOnDropSseStream {
    fn new<S>(inner: S, cancellation: CancellationToken) -> Self
    where
        S: Stream<Item = Result<Event, Infallible>> + Send + 'static,
    {
        Self {
            inner: Box::pin(inner),
            cancellation,
        }
    }
}

impl Stream for CancelOnDropSseStream {
    type Item = Result<Event, Infallible>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.inner.as_mut().poll_next(cx)
    }
}

impl Drop for CancelOnDropSseStream {
    fn drop(&mut self) {
        self.cancellation.cancel();
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use async_trait::async_trait;
    use axum::{
        body::Body,
        http::{Request, StatusCode},
    };
    use futures_util::stream;
    use http_body_util::BodyExt;
    use serde_json::{json, Value};
    use tower::ServiceExt;

    use super::*;
    use crate::{
        backend::{
            CancellationToken, ChatCompletionStream, CompletionStream, OpenAiRequestContext,
            OpenAiResult,
        },
        chat::{
            messages_to_plain_prompt, AssistantMessage, ChatCompletionChoice,
            ChatCompletionResponse, ChatMessage, MessageContent, MessageContentPart,
        },
        common::Usage,
        completions::{CompletionPrompt, CompletionResponse},
        errors::{already_openai_error, map_upstream_error_body, OpenAiErrorKind},
        models::ModelObject,
        FinishReason,
    };

    struct FakeBackend;

    #[async_trait]
    impl OpenAiBackend for FakeBackend {
        async fn models(&self) -> OpenAiResult<Vec<ModelObject>> {
            Ok(vec![ModelObject::new("org/repo:Q4_K_M")])
        }

        async fn chat_completion(
            &self,
            request: ChatCompletionRequest,
        ) -> OpenAiResult<ChatCompletionResponse> {
            if request.model == "missing" {
                return Err(OpenAiError::model_not_found(request.model));
            }
            if request.model == "unsupported-feature" {
                return Err(OpenAiError::unsupported(
                    "structured output is parsed but not yet implemented by skippy runtime",
                ));
            }
            if request.model == "tool-call" {
                return Ok(ChatCompletionResponse {
                    id: "chatcmpl_tool".to_string(),
                    object: "chat.completion",
                    created: 123,
                    model: request.model,
                    choices: vec![ChatCompletionChoice {
                        index: 0,
                        message: AssistantMessage {
                            role: "assistant",
                            content: Some("calling lookup".to_string()),
                            tool_calls: Some(json!([{
                                "id": "call_123",
                                "type": "function",
                                "function": {
                                    "name": "lookup",
                                    "arguments": "{\"city\":\"Sydney\"}"
                                }
                            }])),
                        },
                        logprobs: Some(json!({
                            "content": [{
                                "token": "calling",
                                "logprob": -0.2
                            }]
                        })),
                        finish_reason: Some(FinishReason::ToolCalls),
                    }],
                    usage: Usage::new(3, 2),
                });
            }
            Ok(ChatCompletionResponse::new(
                request.model,
                format!("echo: {}", messages_to_plain_prompt(&request.messages)),
                Usage::new(3, 2),
            ))
        }

        async fn chat_completion_stream(
            &self,
            request: ChatCompletionRequest,
            _context: OpenAiRequestContext,
        ) -> OpenAiResult<ChatCompletionStream> {
            if request.model == "missing" {
                return Err(OpenAiError::model_not_found(request.model));
            }
            if request.model == "stream-error" {
                return Ok(Box::pin(stream::iter(vec![Err(OpenAiError::backend(
                    "stream backend failed",
                ))])));
            }
            if request.model == "stream-logprobs" {
                let model = request.model;
                return Ok(Box::pin(stream::iter(vec![
                    Ok(ChatCompletionChunk {
                        id: "chatcmpl_stream_logprobs".to_string(),
                        object: "chat.completion.chunk",
                        created: 123,
                        model: model.clone(),
                        choices: vec![crate::chat::ChatCompletionChunkChoice {
                            index: 0,
                            delta: crate::chat::ChatCompletionDelta {
                                role: None,
                                content: Some("tok".to_string()),
                                tool_calls: None,
                            },
                            logprobs: Some(json!({
                                "content": [{
                                    "token": "tok",
                                    "logprob": -0.1
                                }]
                            })),
                            finish_reason: None,
                        }],
                        usage: None,
                    }),
                    Ok(ChatCompletionChunk::done(model)),
                ])));
            }
            let model = request.model;
            Ok(Box::pin(stream::iter(vec![
                Ok(ChatCompletionChunk::delta(model.clone(), "hel")),
                Ok(ChatCompletionChunk::delta(model.clone(), "lo")),
                Ok(ChatCompletionChunk::usage(model.clone(), Usage::new(3, 2))),
                Ok(ChatCompletionChunk::done_with_reason(
                    model,
                    FinishReason::Length,
                )),
            ])))
        }

        async fn completion(&self, request: CompletionRequest) -> OpenAiResult<CompletionResponse> {
            Ok(CompletionResponse::new(
                request.model,
                format!("echo: {}", request.prompt.text_lossy()),
                Usage::new(2, 1),
            ))
        }

        async fn completion_stream(
            &self,
            request: CompletionRequest,
            _context: OpenAiRequestContext,
        ) -> OpenAiResult<CompletionStream> {
            let model = request.model;
            Ok(Box::pin(stream::iter(vec![
                Ok(crate::CompletionChunk::delta(model.clone(), "a")),
                Ok(crate::CompletionChunk::usage(
                    model.clone(),
                    Usage::new(2, 1),
                )),
                Ok(crate::CompletionChunk::done_with_reason(
                    model,
                    FinishReason::Length,
                )),
            ])))
        }
    }

    struct SlowBackend;

    #[async_trait]
    impl OpenAiBackend for SlowBackend {
        async fn models(&self) -> OpenAiResult<Vec<ModelObject>> {
            tokio::time::sleep(Duration::from_millis(50)).await;
            Ok(vec![ModelObject::new("slow-model")])
        }

        async fn chat_completion(
            &self,
            _request: ChatCompletionRequest,
        ) -> OpenAiResult<ChatCompletionResponse> {
            unreachable!("slow backend test only calls readiness")
        }

        async fn chat_completion_stream(
            &self,
            _request: ChatCompletionRequest,
            _context: OpenAiRequestContext,
        ) -> OpenAiResult<ChatCompletionStream> {
            unreachable!("slow backend test only calls readiness")
        }

        async fn completion(
            &self,
            _request: CompletionRequest,
        ) -> OpenAiResult<CompletionResponse> {
            unreachable!("slow backend test only calls readiness")
        }

        async fn completion_stream(
            &self,
            _request: CompletionRequest,
            _context: OpenAiRequestContext,
        ) -> OpenAiResult<CompletionStream> {
            unreachable!("slow backend test only calls readiness")
        }
    }

    struct CancellationBackend {
        token: Arc<Mutex<Option<CancellationToken>>>,
    }

    #[async_trait]
    impl OpenAiBackend for CancellationBackend {
        async fn models(&self) -> OpenAiResult<Vec<ModelObject>> {
            Ok(vec![ModelObject::new("cancel-model")])
        }

        async fn chat_completion(
            &self,
            _request: ChatCompletionRequest,
        ) -> OpenAiResult<ChatCompletionResponse> {
            unreachable!("cancellation backend test only calls streaming")
        }

        async fn chat_completion_stream(
            &self,
            _request: ChatCompletionRequest,
            context: OpenAiRequestContext,
        ) -> OpenAiResult<ChatCompletionStream> {
            *self.token.lock().expect("token lock poisoned") = Some(context.cancellation_token());
            Ok(Box::pin(stream::pending()))
        }
    }

    #[test]
    fn messages_to_plain_prompt_extracts_text_parts() {
        let messages = vec![
            ChatMessage {
                role: "system".to_string(),
                content: Some(MessageContent::Text("system text".to_string())),
                extra: Default::default(),
            },
            ChatMessage {
                role: "user".to_string(),
                content: Some(MessageContent::Parts(vec![MessageContentPart {
                    content_type: "text".to_string(),
                    text: Some("part text".to_string()),
                    extra: Default::default(),
                }])),
                extra: Default::default(),
            },
        ];
        assert_eq!(
            messages_to_plain_prompt(&messages),
            "system text\npart text"
        );
    }

    #[test]
    fn max_completion_tokens_takes_precedence() {
        let request: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "test",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 10,
            "max_completion_tokens": 3
        }))
        .unwrap();
        assert_eq!(request.effective_max_tokens(), Some(3));
    }

    #[test]
    fn completion_prompt_text_lossy_for_string_arrays() {
        assert_eq!(
            CompletionPrompt::ManyText(vec!["one".to_string(), "two".to_string()]).text_lossy(),
            "one\ntwo"
        );
    }

    #[test]
    fn strict_error_body_uses_openai_shape() {
        let error = OpenAiError::from_kind(
            StatusCode::SERVICE_UNAVAILABLE,
            OpenAiErrorKind::ServiceUnavailable,
            "upstream down",
        );
        let value = serde_json::to_value(error.body()).unwrap();
        assert_eq!(value["error"]["message"], "upstream down");
        assert_eq!(value["error"]["type"], "server_error");
        assert_eq!(value["error"]["code"], "service_unavailable");
    }

    #[test]
    fn upstream_error_body_maps_llama_error_shape() {
        let body = br#"{"type":"exceed_context_size_error","message":"too long"}"#;
        let mapped = map_upstream_error_body(400, body).unwrap();
        let value: Value = serde_json::from_slice(&mapped).unwrap();
        assert_eq!(value["error"]["message"], "too long");
        assert_eq!(value["error"]["type"], "invalid_request_error");
        assert_eq!(value["error"]["code"], "context_length_exceeded");
    }

    #[test]
    fn already_openai_error_passthrough_is_detected() {
        let value = json!({
            "error": {
                "message": "bad request",
                "type": "invalid_request_error",
                "param": null,
                "code": "invalid_value"
            }
        });
        assert!(already_openai_error(&value));
        let body = serde_json::to_vec(&value).unwrap();
        assert_eq!(map_upstream_error_body(400, &body), None);
    }

    #[tokio::test]
    async fn models_route_returns_model_list() {
        let app = router_for(Arc::new(FakeBackend));
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/v1/models")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = response_body_json(response).await;
        assert_eq!(body["object"], "list");
        assert_eq!(body["data"][0]["id"], "org/repo:Q4_K_M");
    }

    #[tokio::test]
    async fn health_route_returns_liveness_probe() {
        let app = router_for(Arc::new(FakeBackend));
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = response_body_json(response).await;
        assert_eq!(body["status"], "ok");
    }

    #[tokio::test]
    async fn readiness_route_checks_backend_models() {
        let app = router_for(Arc::new(FakeBackend));
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/readyz")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = response_body_json(response).await;
        assert_eq!(body["status"], "ready");
    }

    #[tokio::test]
    async fn backend_timeout_returns_openai_error_shape() {
        let app = router_for_with_config(
            Arc::new(SlowBackend),
            OpenAiFrontendConfig::default().with_backend_timeout(Duration::from_millis(1)),
        );
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/readyz")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::GATEWAY_TIMEOUT);
        let body = response_body_json(response).await;
        assert_eq!(body["error"]["type"], "server_error");
        assert_eq!(body["error"]["code"], "timeout");
    }

    #[tokio::test]
    async fn request_id_is_returned_on_success_and_errors() {
        let app = router_for(Arc::new(FakeBackend));
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .header("x-request-id", "client-req-1")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.headers()["x-request-id"], "client-req-1");

        let app = router_for(Arc::new(FakeBackend));
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/not-here")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert!(response.headers().get("x-request-id").is_some());
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn unknown_routes_return_openai_error_shape() {
        let app = router_for(Arc::new(FakeBackend));
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/v1/unknown")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
        let body = response_body_json(response).await;
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert_eq!(body["error"]["code"], "not_found");
    }

    #[tokio::test]
    async fn unsupported_methods_return_openai_error_shape() {
        for (method, path) in [
            ("POST", "/v1/models"),
            ("GET", "/v1/chat/completions"),
            ("GET", "/v1/completions"),
        ] {
            let app = router_for(Arc::new(FakeBackend));
            let response = app
                .oneshot(
                    Request::builder()
                        .method(method)
                        .uri(path)
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::METHOD_NOT_ALLOWED);
            let body = response_body_json(response).await;
            assert_eq!(body["error"]["type"], "invalid_request_error");
            assert_eq!(body["error"]["code"], "method_not_allowed");
        }
    }

    #[tokio::test]
    async fn chat_completion_route_returns_openai_shape() {
        let response = post_json(
            "/v1/chat/completions",
            json!({
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}]
            }),
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = response_body_json(response).await;
        assert_eq!(body["object"], "chat.completion");
        assert_eq!(body["choices"][0]["message"]["role"], "assistant");
        assert_eq!(body["choices"][0]["message"]["content"], "echo: hi");
        assert_eq!(body["usage"]["total_tokens"], 5);
    }

    #[tokio::test]
    async fn chat_completion_stream_route_returns_sse() {
        let response = post_json(
            "/v1/chat/completions",
            json!({
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": true,
                "stream_options": {"include_usage": true}
            }),
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = response_body_text(response).await;
        assert!(body.contains(r#""role":"assistant""#));
        assert!(body.contains(r#""content":"hel""#));
        assert!(body.contains(r#""finish_reason":"length""#));
        assert!(body.contains(r#""total_tokens":5"#));
        assert!(body.contains("data: [DONE]"));
    }

    #[tokio::test]
    async fn chat_completion_stream_suppresses_usage_unless_requested() {
        let response = post_json(
            "/v1/chat/completions",
            json!({
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": true
            }),
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = response_body_text(response).await;
        assert!(!body.contains(r#""total_tokens":5"#));
        assert!(body.contains("data: [DONE]"));
    }

    #[tokio::test]
    async fn chat_completion_stream_frames_backend_errors() {
        let response = post_json(
            "/v1/chat/completions",
            json!({
                "model": "stream-error",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": true
            }),
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = response_body_text(response).await;
        assert!(body.contains(r#""error":{"#));
        assert!(body.contains(r#""code":"service_unavailable""#));
        assert!(body.contains("data: [DONE]"));
    }

    #[tokio::test]
    async fn dropping_stream_response_cancels_request_context() {
        let token = Arc::new(Mutex::new(None));
        let app = router(Arc::new(CancellationBackend {
            token: token.clone(),
        }));
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        json!({
                            "model": "cancel-model",
                            "messages": [{"role": "user", "content": "hello"}],
                            "stream": true
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let cancellation = token
            .lock()
            .expect("token lock poisoned")
            .clone()
            .expect("backend saw request context");
        assert!(!cancellation.is_cancelled());

        drop(response);
        tokio::task::yield_now().await;

        assert!(cancellation.is_cancelled());
    }

    #[tokio::test]
    async fn chat_completion_route_maps_backend_errors() {
        let response = post_json(
            "/v1/chat/completions",
            json!({
                "model": "missing",
                "messages": [{"role": "user", "content": "hi"}]
            }),
        )
        .await;
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
        let body = response_body_json(response).await;
        assert_eq!(body["error"]["code"], "model_not_found");
    }

    #[tokio::test]
    async fn chat_completion_route_rejects_empty_messages() {
        let response = post_json(
            "/v1/chat/completions",
            json!({
                "model": "test-model",
                "messages": []
            }),
        )
        .await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = response_body_json(response).await;
        assert_eq!(body["error"]["type"], "invalid_request_error");
    }

    #[tokio::test]
    async fn chat_completion_route_rejects_multiple_choices_until_supported() {
        let response = post_json(
            "/v1/chat/completions",
            json!({
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
                "n": 2
            }),
        )
        .await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = response_body_json(response).await;
        assert_eq!(body["error"]["code"], "unsupported_model_feature");
    }

    #[tokio::test]
    async fn chat_completion_route_accepts_tools_structured_output_and_logprobs() {
        let response = post_json(
            "/v1/chat/completions",
            json!({
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [{"type": "function", "function": {"name": "lookup"}}],
                "tool_choice": "auto",
                "parallel_tool_calls": true,
                "response_format": {"type": "json_schema", "json_schema": {"name": "answer", "schema": {"type": "object"}}},
                "logprobs": true,
                "top_logprobs": 2
            }),
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn chat_completion_route_accepts_noop_parity_fields() {
        let response = post_json(
            "/v1/chat/completions",
            json!({
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
                "n": 1,
                "tools": [],
                "response_format": {"type": "text"}
            }),
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn responses_route_translates_to_chat_and_back() {
        let response = post_json(
            "/v1/responses",
            json!({
                "model": "test-model",
                "instructions": "be concise",
                "input": "hi",
                "max_output_tokens": 12,
                "tools": [{"type": "function", "function": {"name": "lookup"}}],
                "response_format": {"type": "json_schema", "json_schema": {"name": "answer", "schema": {"type": "object"}}},
                "logprobs": true,
                "top_logprobs": 1
            }),
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = response_body_json(response).await;
        assert_eq!(body["object"], "response");
        assert_eq!(body["output_text"], "echo: be concise\nhi");
        assert_eq!(body["usage"]["input_tokens"], 3);
        assert_eq!(body["usage"]["output_tokens"], 2);
    }

    #[tokio::test]
    async fn responses_route_preserves_tool_calls_and_logprobs() {
        let response = post_json(
            "/v1/responses",
            json!({
                "model": "tool-call",
                "input": "hi",
                "tools": [{"type": "function", "function": {"name": "lookup"}}],
                "logprobs": true,
                "top_logprobs": 1
            }),
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = response_body_json(response).await;
        assert_eq!(body["output_text"], "calling lookup");
        assert_eq!(
            body["output"][0]["content"][0]["logprobs"]["content"][0]["token"],
            "calling"
        );
        assert_eq!(body["output"][1]["type"], "function_call");
        assert_eq!(body["output"][1]["call_id"], "call_123");
        assert_eq!(body["finish_reason"], "tool_calls");
    }

    #[tokio::test]
    async fn responses_route_preserves_backend_unsupported_errors() {
        let response = post_json(
            "/v1/responses",
            json!({
                "model": "unsupported-feature",
                "input": "hi",
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "answer", "schema": {"type": "object"}}
                }
            }),
        )
        .await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = response_body_json(response).await;
        assert_eq!(body["error"]["code"], "unsupported_model_feature");
        assert!(body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("structured output"));
    }

    #[tokio::test]
    async fn responses_stream_route_returns_responses_sse() {
        let response = post_json(
            "/v1/responses",
            json!({
                "model": "test-model",
                "input": "hi",
                "stream": true,
                "stream_options": {"include_usage": true}
            }),
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = response_body_text(response).await;
        assert!(body.contains("event: response.created"));
        assert!(body.contains("event: response.output_text.delta"));
        assert!(body.contains("event: response.completed"));
        assert!(body.contains(r#""output_text":"hello""#));
        assert!(body.contains("data: [DONE]"));
    }

    #[tokio::test]
    async fn responses_stream_route_preserves_logprobs() {
        let response = post_json(
            "/v1/responses",
            json!({
                "model": "stream-logprobs",
                "input": "hi",
                "stream": true,
                "logprobs": true
            }),
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = response_body_text(response).await;
        assert!(body.contains("event: response.output_text.delta"));
        assert!(body.contains(r#""logprobs":{"content":[{"logprob":-0.1,"token":"tok"}]}"#));
    }

    #[tokio::test]
    async fn completion_route_returns_openai_shape() {
        let response = post_json(
            "/v1/completions",
            json!({
                "model": "test-model",
                "prompt": "hi"
            }),
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = response_body_json(response).await;
        assert_eq!(body["object"], "text_completion");
        assert_eq!(body["choices"][0]["text"], "echo: hi");
        assert_eq!(body["usage"]["total_tokens"], 3);
    }

    #[tokio::test]
    async fn completion_stream_route_returns_sse() {
        let response = post_json(
            "/v1/completions",
            json!({
                "model": "test-model",
                "prompt": "hi",
                "stream": true,
                "stream_options": {"include_usage": true}
            }),
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = response_body_text(response).await;
        assert!(body.contains(r#""text":"a""#));
        assert!(body.contains(r#""finish_reason":"length""#));
        assert!(body.contains(r#""total_tokens":3"#));
        assert!(body.contains("data: [DONE]"));
    }

    #[tokio::test]
    async fn completion_stream_suppresses_usage_unless_requested() {
        let response = post_json(
            "/v1/completions",
            json!({
                "model": "test-model",
                "prompt": "hi",
                "stream": true
            }),
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = response_body_text(response).await;
        assert!(!body.contains(r#""total_tokens":3"#));
        assert!(body.contains("data: [DONE]"));
    }

    #[tokio::test]
    async fn completion_route_rejects_empty_prompt() {
        let response = post_json(
            "/v1/completions",
            json!({
                "model": "test-model",
                "prompt": ""
            }),
        )
        .await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = response_body_json(response).await;
        assert_eq!(body["error"]["code"], "invalid_value");
    }

    #[tokio::test]
    async fn completion_route_accepts_token_prompts() {
        let response = post_json(
            "/v1/completions",
            json!({
                "model": "test-model",
                "prompt": [1, 2, 3]
            }),
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn completion_route_accepts_logprobs() {
        let response = post_json(
            "/v1/completions",
            json!({
                "model": "test-model",
                "prompt": "hi",
                "logprobs": 2
            }),
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn completion_route_accepts_single_choice_controls() {
        let response = post_json(
            "/v1/completions",
            json!({
                "model": "test-model",
                "prompt": "hi",
                "n": 1,
                "best_of": 1
            }),
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn missing_content_type_maps_to_strict_error_shape() {
        let app = router_for(Arc::new(FakeBackend));
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .body(Body::from(
                        json!({
                            "model": "test-model",
                            "messages": [{"role": "user", "content": "hi"}]
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = response_body_json(response).await;
        assert_eq!(body["error"]["type"], "invalid_request_error");
    }

    #[tokio::test]
    async fn invalid_json_maps_to_strict_error_shape() {
        let app = router_for(Arc::new(FakeBackend));
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from("{"))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = response_body_json(response).await;
        assert_eq!(body["error"]["type"], "invalid_request_error");
    }

    #[tokio::test]
    async fn oversized_json_maps_to_strict_error_shape() {
        let app = router_for_with_config(
            Arc::new(FakeBackend),
            OpenAiFrontendConfig::default().with_max_request_body_bytes(64),
        );
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(json!({
                        "model": "test-model",
                        "messages": [{"role": "user", "content": "this body is intentionally much larger than sixty four bytes"}]
                    }).to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
        let body = response_body_json(response).await;
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert_eq!(body["error"]["code"], "payload_too_large");
    }

    async fn post_json(path: &str, value: Value) -> axum::response::Response {
        let app = router_for(Arc::new(FakeBackend));
        app.oneshot(
            Request::builder()
                .method("POST")
                .uri(path)
                .header("content-type", "application/json")
                .body(Body::from(value.to_string()))
                .unwrap(),
        )
        .await
        .unwrap()
    }

    async fn response_body_json(response: axum::response::Response) -> Value {
        let body = response.into_body().collect().await.unwrap().to_bytes();
        serde_json::from_slice(&body).unwrap()
    }

    async fn response_body_text(response: axum::response::Response) -> String {
        let body = response.into_body().collect().await.unwrap().to_bytes();
        String::from_utf8(body.to_vec()).unwrap()
    }
}
