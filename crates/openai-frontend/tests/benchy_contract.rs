use std::sync::Arc;

use async_trait::async_trait;
use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use futures_util::stream;
use http_body_util::BodyExt;
use openai_frontend::{
    messages_to_plain_prompt, ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse,
    ChatCompletionStream, CompletionChunk, CompletionRequest, CompletionResponse, CompletionStream,
    FinishReason, ModelObject, OpenAiBackend, OpenAiError, OpenAiRequestContext, OpenAiResult,
    Usage,
};
use serde_json::{json, Value};
use tower::ServiceExt;

struct BenchyBackend;

const BENCHY_MODEL_ID: &str = "org/repo:Q4_K_M";

#[async_trait]
impl OpenAiBackend for BenchyBackend {
    async fn models(&self) -> OpenAiResult<Vec<ModelObject>> {
        Ok(vec![ModelObject::new(BENCHY_MODEL_ID)])
    }

    async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> OpenAiResult<ChatCompletionResponse> {
        ensure_model(&request.model)?;
        let completion_tokens = request.effective_max_tokens().unwrap_or(2);
        Ok(ChatCompletionResponse::new(
            request.model,
            format!(
                "benchy echo: {}",
                messages_to_plain_prompt(&request.messages)
            ),
            Usage::new(9, completion_tokens),
        ))
    }

    async fn chat_completion_stream(
        &self,
        request: ChatCompletionRequest,
        _context: OpenAiRequestContext,
    ) -> OpenAiResult<ChatCompletionStream> {
        ensure_model(&request.model)?;
        let completion_tokens = request.effective_max_tokens().unwrap_or(2);
        let model = request.model;
        Ok(Box::pin(stream::iter(vec![
            Ok(ChatCompletionChunk::delta(model.clone(), "tok")),
            Ok(ChatCompletionChunk::delta(model.clone(), "en")),
            Ok(ChatCompletionChunk::usage(
                model.clone(),
                Usage::new(9, completion_tokens),
            )),
            Ok(ChatCompletionChunk::done_with_reason(
                model,
                FinishReason::Length,
            )),
        ])))
    }

    async fn completion(&self, request: CompletionRequest) -> OpenAiResult<CompletionResponse> {
        ensure_model(&request.model)?;
        let completion_tokens = request.max_tokens.unwrap_or(2);
        Ok(CompletionResponse::new(
            request.model,
            format!("benchy completion: {}", request.prompt.text_lossy()),
            Usage::new(4, completion_tokens),
        ))
    }

    async fn completion_stream(
        &self,
        request: CompletionRequest,
        _context: OpenAiRequestContext,
    ) -> OpenAiResult<CompletionStream> {
        ensure_model(&request.model)?;
        let completion_tokens = request.max_tokens.unwrap_or(2);
        let model = request.model;
        Ok(Box::pin(stream::iter(vec![
            Ok(CompletionChunk::delta(model.clone(), "tok")),
            Ok(CompletionChunk::usage(
                model.clone(),
                Usage::new(4, completion_tokens),
            )),
            Ok(CompletionChunk::done_with_reason(
                model,
                FinishReason::Length,
            )),
        ])))
    }
}

fn ensure_model(model: &str) -> OpenAiResult<()> {
    if model == BENCHY_MODEL_ID {
        Ok(())
    } else {
        Err(OpenAiError::model_not_found(model))
    }
}

#[tokio::test]
async fn benchy_can_discover_models() {
    let response = request("GET", "/v1/models", Value::Null).await;
    assert_eq!(response.status(), StatusCode::OK);

    let body = response_json(response).await;
    assert_eq!(body["object"], "list");
    assert_eq!(body["data"][0]["id"], BENCHY_MODEL_ID);
    assert_eq!(body["data"][0]["object"], "model");
}

#[tokio::test]
async fn benchy_non_stream_chat_warmup_has_usage() {
    let response = request(
        "POST",
        "/v1/chat/completions",
        json!({
            "model": BENCHY_MODEL_ID,
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 3
        }),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);

    let body = response_json(response).await;
    assert_eq!(body["object"], "chat.completion");
    assert_eq!(body["model"], BENCHY_MODEL_ID);
    assert_eq!(body["choices"][0]["message"]["role"], "assistant");
    assert_eq!(
        body["choices"][0]["message"]["content"],
        "benchy echo: hello"
    );
    assert_eq!(body["usage"]["prompt_tokens"], 9);
    assert_eq!(body["usage"]["completion_tokens"], 3);
    assert_eq!(body["usage"]["total_tokens"], 12);
}

#[tokio::test]
async fn benchy_wrong_model_returns_model_not_found() {
    let response = request(
        "POST",
        "/v1/chat/completions",
        json!({
            "model": "org/repo:Q5_K_M",
            "messages": [{"role": "user", "content": "hello"}]
        }),
    )
    .await;
    assert_eq!(response.status(), StatusCode::NOT_FOUND);

    let body = response_json(response).await;
    assert_eq!(body["error"]["type"], "invalid_request_error");
    assert_eq!(body["error"]["code"], "model_not_found");
}

#[tokio::test]
async fn benchy_stream_chat_contract_has_role_content_usage_and_done() {
    let response = request(
        "POST",
        "/v1/chat/completions",
        json!({
            "model": BENCHY_MODEL_ID,
            "messages": [{"role": "user", "content": "hello"}],
            "stream": true,
            "stream_options": {"include_usage": true},
            "max_tokens": 2
        }),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers()["content-type"].to_str().unwrap(),
        "text/event-stream"
    );

    let data = sse_data(response_text(response).await);
    assert_eq!(data.last().unwrap(), "[DONE]");
    assert!(data
        .iter()
        .any(|line| line.contains(r#""role":"assistant""#)));
    assert!(data.iter().any(|line| line.contains(r#""content":"tok""#)));
    assert!(data.iter().any(|line| line.contains(r#""content":"en""#)));
    assert!(data
        .iter()
        .any(|line| line.contains(r#""finish_reason":"length""#)));
    assert!(data
        .iter()
        .any(|line| line.contains(r#""total_tokens":11"#)));
}

#[tokio::test]
async fn benchy_stream_chat_omits_usage_without_stream_option() {
    let response = request(
        "POST",
        "/v1/chat/completions",
        json!({
            "model": BENCHY_MODEL_ID,
            "messages": [{"role": "user", "content": "hello"}],
            "stream": true,
            "max_tokens": 2
        }),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);

    let text = response_text(response).await;
    assert!(!text.contains(r#""usage":{"#));
    assert!(text.contains("data: [DONE]"));
}

#[tokio::test]
async fn legacy_completions_surface_stays_available_for_other_clients() {
    let response = request(
        "POST",
        "/v1/completions",
        json!({
            "model": BENCHY_MODEL_ID,
            "prompt": "hello",
            "max_tokens": 4
        }),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);

    let body = response_json(response).await;
    assert_eq!(body["object"], "text_completion");
    assert_eq!(body["choices"][0]["text"], "benchy completion: hello");
    assert_eq!(body["usage"]["total_tokens"], 8);
}

async fn request(method: &str, path: &str, value: Value) -> axum::response::Response {
    let app = openai_frontend::router_for(Arc::new(BenchyBackend));
    let body = if value.is_null() {
        Body::empty()
    } else {
        Body::from(value.to_string())
    };
    app.oneshot(
        Request::builder()
            .method(method)
            .uri(path)
            .header("content-type", "application/json")
            .body(body)
            .unwrap(),
    )
    .await
    .unwrap()
}

async fn response_json(response: axum::response::Response) -> Value {
    serde_json::from_slice(&response_bytes(response).await).unwrap()
}

async fn response_text(response: axum::response::Response) -> String {
    String::from_utf8(response_bytes(response).await.to_vec()).unwrap()
}

async fn response_bytes(response: axum::response::Response) -> axum::body::Bytes {
    response.into_body().collect().await.unwrap().to_bytes()
}

fn sse_data(text: String) -> Vec<String> {
    text.lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .map(ToString::to_string)
        .collect()
}
