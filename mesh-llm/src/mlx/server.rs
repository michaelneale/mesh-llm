//! In-process HTTP server for MLX inference.
//!
//! Drop-in replacement for llama-server — speaks the same OpenAI-compatible
//! HTTP API on a local port so the existing proxy routes to it unchanged.

use super::model::{self, MlxModel};
use crate::inference::launch::{InferenceServerHandle, InferenceServerProcess};
use anyhow::{Context, Result};
use mlx_rs::Array;
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::sync::{watch, Mutex};

/// Shared inference state behind the server.
struct InferState {
    model: MlxModel,
    model_name: String,
}

/// Start the MLX inference server on the given port.
/// Returns an in-process inference server handle plus a death channel.
///
/// This is the MLX equivalent of `launch::start_llama_server` — the caller
/// gets back the same process wrapper used by the llama.cpp backend.
pub async fn start_mlx_server(
    model_dir: &std::path::Path,
    model_name: String,
    port: u16,
) -> Result<InferenceServerProcess> {
    // Load model on a blocking thread (touches disk + GPU init)
    let dir = model_dir.to_path_buf();
    let model = tokio::task::spawn_blocking(move || MlxModel::load(&dir))
        .await
        .context("MLX model load panicked")??;

    tracing::info!(
        "MLX server: model loaded — {} layers, vocab={}, serving on :{port}",
        model.config.num_hidden_layers,
        model.config.vocab_size,
    );

    let state = Arc::new(Mutex::new(InferState { model, model_name }));

    let listener = TcpListener::bind(format!("127.0.0.1:{port}"))
        .await
        .with_context(|| format!("MLX server: failed to bind port {port}"))?;

    let (death_tx, death_rx) = tokio::sync::oneshot::channel();
    let (shutdown_tx, mut shutdown_rx) = watch::channel(false);

    tokio::spawn(async move {
        loop {
            tokio::select! {
                changed = shutdown_rx.changed() => {
                    if changed.is_ok() && *shutdown_rx.borrow() {
                        break;
                    }
                }
                accept = listener.accept() => {
                    let (stream, _addr) = match accept {
                        Ok(s) => s,
                        Err(e) => {
                            tracing::warn!("MLX server: accept error: {e}");
                            continue;
                        }
                    };
                    let state = state.clone();
                    tokio::spawn(async move {
                        if let Err(e) = handle_connection(stream, state).await {
                            tracing::debug!("MLX server: connection error: {e}");
                        }
                    });
                }
            }
        }
        drop(death_tx);
    });

    Ok(InferenceServerProcess {
        handle: InferenceServerHandle::in_process(shutdown_tx),
        death_rx,
    })
}

/// Parse a raw HTTP request from the stream and dispatch.
async fn handle_connection(
    mut stream: tokio::net::TcpStream,
    state: Arc<Mutex<InferState>>,
) -> Result<()> {
    let _ = stream.set_nodelay(true);

    let mut buf = vec![0u8; 64 * 1024];
    let mut filled = 0usize;

    // Read until we have complete headers
    loop {
        let n = stream.read(&mut buf[filled..]).await?;
        if n == 0 {
            return Ok(());
        }
        filled += n;
        if filled > 4 && buf[..filled].windows(4).any(|w| w == b"\r\n\r\n") {
            break;
        }
        if filled >= buf.len() {
            send_response(&mut stream, 413, "Request too large").await?;
            return Ok(());
        }
    }

    // Find header/body split
    let header_end = buf[..filled]
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .unwrap()
        + 4;

    // Parse method, path, content-length from headers (own the strings so buf is free)
    let header_str = String::from_utf8_lossy(&buf[..header_end]).to_string();

    let first_line = header_str.lines().next().unwrap_or("");
    let parts: Vec<&str> = first_line.split_whitespace().collect();
    let method = if parts.len() >= 2 {
        parts[0].to_string()
    } else {
        String::new()
    };
    let path = if parts.len() >= 2 {
        parts[1].to_string()
    } else {
        String::new()
    };

    let content_length: usize = header_str
        .lines()
        .find_map(|line| {
            let lower = line.to_ascii_lowercase();
            if lower.starts_with("content-length:") {
                lower.split(':').nth(1)?.trim().parse().ok()
            } else {
                None
            }
        })
        .unwrap_or(0);

    // Read remaining body if needed
    let body_so_far = filled - header_end;
    if body_so_far < content_length {
        let remaining = content_length - body_so_far;
        if filled + remaining > buf.len() {
            buf.resize(filled + remaining, 0);
        }
        let mut read = 0;
        while read < remaining {
            let n = stream
                .read(&mut buf[filled + read..filled + remaining])
                .await?;
            if n == 0 {
                break;
            }
            read += n;
        }
        filled += read;
    }

    let body = &buf[header_end..filled.min(header_end + content_length)];

    match (method.as_str(), path.as_str()) {
        ("GET", "/health") => {
            send_response(&mut stream, 200, r#"{"status":"ok"}"#).await?;
        }
        ("GET", "/v1/models") | ("GET", "/models") => {
            let state = state.lock().await;
            let resp = serde_json::json!({
                "object": "list",
                "data": [{
                    "id": &state.model_name,
                    "object": "model",
                    "owned_by": "mlx",
                }]
            });
            send_response(&mut stream, 200, &resp.to_string()).await?;
        }
        ("POST", "/v1/chat/completions") => {
            handle_chat_completions(&mut stream, body, state).await?;
        }
        ("POST", "/v1/completions") => {
            handle_completions(&mut stream, body, state).await?;
        }
        _ => {
            send_response(&mut stream, 404, r#"{"error":"not found"}"#).await?;
        }
    }

    Ok(())
}

/// Handle POST /v1/chat/completions — the main inference endpoint.
async fn handle_chat_completions(
    stream: &mut tokio::net::TcpStream,
    body: &[u8],
    state: Arc<Mutex<InferState>>,
) -> Result<()> {
    let req: serde_json::Value =
        serde_json::from_slice(body).context("invalid JSON in chat completions request")?;

    let messages = req["messages"]
        .as_array()
        .context("missing messages array")?;
    let stream_mode = req["stream"].as_bool().unwrap_or(false);
    let max_tokens = req["max_tokens"].as_u64().unwrap_or(2048) as usize;
    let model_field = req["model"].as_str().unwrap_or("");

    // Build prompt from messages using a simple chat template
    let prompt = build_chat_prompt(messages);

    if stream_mode {
        generate_streaming(stream, &prompt, max_tokens, model_field, state).await
    } else {
        generate_blocking(stream, &prompt, max_tokens, model_field, state).await
    }
}

/// Handle POST /v1/completions — raw text completion.
async fn handle_completions(
    stream: &mut tokio::net::TcpStream,
    body: &[u8],
    state: Arc<Mutex<InferState>>,
) -> Result<()> {
    let req: serde_json::Value =
        serde_json::from_slice(body).context("invalid JSON in completions request")?;

    let prompt = req["prompt"].as_str().unwrap_or("").to_string();
    let stream_mode = req["stream"].as_bool().unwrap_or(false);
    let max_tokens = req["max_tokens"].as_u64().unwrap_or(2048) as usize;
    let model_field = req["model"].as_str().unwrap_or("");

    if stream_mode {
        generate_streaming(stream, &prompt, max_tokens, model_field, state).await
    } else {
        generate_blocking(stream, &prompt, max_tokens, model_field, state).await
    }
}

/// Non-streaming: run full generation, return one JSON response.
async fn generate_blocking(
    stream: &mut tokio::net::TcpStream,
    prompt: &str,
    max_tokens: usize,
    model_field: &str,
    state: Arc<Mutex<InferState>>,
) -> Result<()> {
    let prompt = prompt.to_string();
    let model_field = model_field.to_string();
    let (text, prompt_tokens, completion_tokens) =
        tokio::task::spawn_blocking(move || -> Result<(String, usize, usize)> {
            let mut state = state.blocking_lock();
            run_inference(&mut state, &prompt, max_tokens)
        })
        .await??;

    let resp = serde_json::json!({
        "id": format!("chatcmpl-mlx-{}", std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_millis()),
        "object": "chat.completion",
        "model": model_field,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": text,
            },
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
    });
    send_response(stream, 200, &resp.to_string()).await
}

/// Streaming: send SSE chunks as tokens are generated.
async fn generate_streaming(
    stream: &mut tokio::net::TcpStream,
    prompt: &str,
    max_tokens: usize,
    model_field: &str,
    state: Arc<Mutex<InferState>>,
) -> Result<()> {
    // Channel for token-by-token streaming
    let (tx, mut rx) = tokio::sync::mpsc::channel::<Option<String>>(64);

    let prompt = prompt.to_string();
    tokio::task::spawn_blocking(move || {
        let mut state = state.blocking_lock();
        let _ = run_inference_streaming(&mut state, &prompt, max_tokens, &tx);
        let _ = tx.blocking_send(None); // signal done
    });

    // Send SSE headers
    let header = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n"
    );
    stream.write_all(header.as_bytes()).await?;

    let id = format!(
        "chatcmpl-mlx-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis()
    );

    while let Some(maybe_token) = rx.recv().await {
        match maybe_token {
            Some(text) => {
                let chunk = serde_json::json!({
                    "id": &id,
                    "object": "chat.completion.chunk",
                    "model": &model_field,
                    "choices": [{
                        "index": 0,
                        "delta": { "content": text },
                        "finish_reason": null,
                    }]
                });
                let sse = format!("data: {}\n\n", chunk);
                if stream.write_all(sse.as_bytes()).await.is_err() {
                    break;
                }
            }
            None => {
                // Final chunk with finish_reason
                let chunk = serde_json::json!({
                    "id": &id,
                    "object": "chat.completion.chunk",
                    "model": &model_field,
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }]
                });
                let sse = format!("data: {}\n\ndata: [DONE]\n\n", chunk);
                let _ = stream.write_all(sse.as_bytes()).await;
                break;
            }
        }
    }

    let _ = stream.shutdown().await;
    Ok(())
}

/// Prefill all prompt tokens in one forward pass.
fn prefill(model: &MlxModel, prompt_tokens: &[u32], caches: &mut [model::KVCache]) -> Result<u32> {
    let input = Array::from_slice(prompt_tokens, &[1, prompt_tokens.len() as i32]);
    let logits = model.forward(&input, caches)?;
    model::argmax_last(&logits)
}

/// Run inference synchronously (called from blocking thread).
fn run_inference(
    state: &mut InferState,
    prompt: &str,
    max_tokens: usize,
) -> Result<(String, usize, usize)> {
    let encoding = state
        .model
        .tokenizer
        .encode(prompt, false)
        .map_err(|e| anyhow::anyhow!("tokenizer encode: {e}"))?;
    let prompt_tokens: Vec<u32> = encoding.get_ids().to_vec();
    let prompt_len = prompt_tokens.len();

    let mut caches = state.model.new_caches();

    // Chunked prefill
    let mut next_token = prefill(&state.model, &prompt_tokens, &mut caches)?;

    let mut generated: Vec<u32> = vec![next_token];

    // Decode
    for _ in 1..max_tokens {
        if is_eos(next_token, &state.model.config) {
            break;
        }
        let input = Array::from_slice(&[next_token], &[1, 1]);
        let logits = state.model.forward(&input, &mut caches)?;
        next_token = model::argmax_last(&logits)?;
        generated.push(next_token);
    }

    // Remove trailing EOS
    if let Some(&last) = generated.last() {
        if is_eos(last, &state.model.config) {
            generated.pop();
        }
    }

    let text = state
        .model
        .tokenizer
        .decode(&generated, true)
        .map_err(|e| anyhow::anyhow!("tokenizer decode: {e}"))?;

    Ok((text, prompt_len, generated.len()))
}

/// Run inference with per-token callback for streaming.
fn run_inference_streaming(
    state: &mut InferState,
    prompt: &str,
    max_tokens: usize,
    tx: &tokio::sync::mpsc::Sender<Option<String>>,
) -> Result<()> {
    let encoding = state
        .model
        .tokenizer
        .encode(prompt, false)
        .map_err(|e| anyhow::anyhow!("tokenizer encode: {e}"))?;
    let prompt_tokens: Vec<u32> = encoding.get_ids().to_vec();

    let mut caches = state.model.new_caches();

    // Chunked prefill
    let mut next_token = prefill(&state.model, &prompt_tokens, &mut caches)?;

    // Decode + stream
    for _ in 0..max_tokens {
        if is_eos(next_token, &state.model.config) {
            break;
        }

        let text = state
            .model
            .tokenizer
            .decode(&[next_token], true)
            .map_err(|e| anyhow::anyhow!("tokenizer decode: {e}"))?;

        if !text.is_empty() {
            if tx.blocking_send(Some(text)).is_err() {
                break; // client disconnected
            }
        }

        let input = Array::from_slice(&[next_token], &[1, 1]);
        let logits = state.model.forward(&input, &mut caches)?;
        next_token = model::argmax_last(&logits)?;
    }

    Ok(())
}

fn is_eos(token: u32, config: &model::ModelConfig) -> bool {
    config.eos_token_id.contains(&token)
}

fn build_chat_prompt(messages: &[serde_json::Value]) -> String {
    // Qwen/ChatML format — works for Qwen2, Qwen2.5, Qwen3, etc.
    let mut prompt = String::new();
    for msg in messages {
        let role = msg["role"].as_str().unwrap_or("user");
        let content = msg["content"].as_str().unwrap_or("");
        prompt.push_str(&format!("<|im_start|>{role}\n{content}<|im_end|>\n"));
    }
    prompt.push_str("<|im_start|>assistant\n");
    prompt
}

async fn send_response(stream: &mut tokio::net::TcpStream, status: u16, body: &str) -> Result<()> {
    let status_text = match status {
        200 => "OK",
        404 => "Not Found",
        413 => "Payload Too Large",
        500 => "Internal Server Error",
        _ => "Unknown",
    };
    let response = format!(
        "HTTP/1.1 {status} {status_text}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    stream.write_all(response.as_bytes()).await?;
    let _ = stream.shutdown().await;
    Ok(())
}
