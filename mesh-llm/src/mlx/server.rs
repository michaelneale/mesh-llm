//! In-process HTTP server for MLX inference.
//!
//! Drop-in replacement for llama-server — speaks the same OpenAI-compatible
//! HTTP API on a local port so the existing proxy routes to it unchanged.

use super::model::{self, MlxModel};
use super::sampling::{Sampler, SamplingParams, StopBuffer};
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
    /// Prompt cache: KV caches + token IDs from the last request.
    /// On the next request, we find the longest common prefix and
    /// skip re-prefilling those tokens — huge win for agent workloads
    /// where the system prompt + conversation history grows incrementally.
    prompt_cache: Option<PromptCache>,
}

struct PromptCache {
    tokens: Vec<u32>,
    caches: Vec<model::KVCache>,
}

#[derive(Clone)]
struct GenerationConfig {
    max_tokens: usize,
    sampling: SamplingParams,
    stop_sequences: Vec<String>,
}

struct GenerationOutcome {
    text: String,
    prompt_tokens: usize,
    completion_tokens: usize,
    finish_reason: &'static str,
}

enum StreamEvent {
    Text(String),
    Done(&'static str),
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

    let state = Arc::new(Mutex::new(InferState {
        model,
        model_name,
        prompt_cache: None,
    }));

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
    let generation = parse_generation_config(&req);
    let model_field = req["model"].as_str().unwrap_or("");

    let prompt = {
        let state = state.lock().await;
        state.model.prompt_template.render_messages(messages)
    };

    if stream_mode {
        generate_streaming(stream, &prompt, generation, model_field, state).await
    } else {
        generate_blocking(stream, &prompt, generation, model_field, state).await
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
    let generation = parse_generation_config(&req);
    let model_field = req["model"].as_str().unwrap_or("");

    if stream_mode {
        generate_streaming(stream, &prompt, generation, model_field, state).await
    } else {
        generate_blocking(stream, &prompt, generation, model_field, state).await
    }
}

/// Non-streaming: run full generation, return one JSON response.
async fn generate_blocking(
    stream: &mut tokio::net::TcpStream,
    prompt: &str,
    generation: GenerationConfig,
    model_field: &str,
    state: Arc<Mutex<InferState>>,
) -> Result<()> {
    let prompt = prompt.to_string();
    let model_field = model_field.to_string();
    let outcome = tokio::task::spawn_blocking(move || -> Result<GenerationOutcome> {
        let mut state = state.blocking_lock();
        run_inference(&mut state, &prompt, &generation)
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
                "content": outcome.text,
            },
            "finish_reason": outcome.finish_reason,
        }],
        "usage": {
            "prompt_tokens": outcome.prompt_tokens,
            "completion_tokens": outcome.completion_tokens,
            "total_tokens": outcome.prompt_tokens + outcome.completion_tokens,
        }
    });
    send_response(stream, 200, &resp.to_string()).await
}

/// Streaming: send SSE chunks as tokens are generated.
async fn generate_streaming(
    stream: &mut tokio::net::TcpStream,
    prompt: &str,
    generation: GenerationConfig,
    model_field: &str,
    state: Arc<Mutex<InferState>>,
) -> Result<()> {
    // Channel for token-by-token streaming
    let (tx, mut rx) = tokio::sync::mpsc::channel::<StreamEvent>(64);

    let prompt = prompt.to_string();
    tokio::task::spawn_blocking(move || {
        let mut state = state.blocking_lock();
        let finish_reason =
            run_inference_streaming(&mut state, &prompt, &generation, &tx).unwrap_or("stop");
        let _ = tx.blocking_send(StreamEvent::Done(finish_reason));
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
            StreamEvent::Text(text) => {
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
            StreamEvent::Done(finish_reason) => {
                // Final chunk with finish_reason
                let chunk = serde_json::json!({
                    "id": &id,
                    "object": "chat.completion.chunk",
                    "model": &model_field,
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": finish_reason,
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

/// Prefill prompt tokens. Uses chunked prefill (with eval barriers between
/// chunks) to keep the computation graph and peak memory bounded. The chunk
/// size of 2048 matches mlx-lm's default `prefill_step_size`. For prompts
/// ≤2048 tokens, there is only one chunk so no overhead.
const PREFILL_STEP_SIZE: usize = 2048;

fn prefill_logits(
    model: &MlxModel,
    prompt_tokens: &[u32],
    caches: &mut [model::KVCache],
) -> Result<Array> {
    let total = prompt_tokens.len();

    if total <= PREFILL_STEP_SIZE {
        // Small prompt — single forward pass, no eval barriers
        let input = Array::from_slice(prompt_tokens, &[1, total as i32]);
        return model.forward(&input, caches);
    }

    // Large prompt — chunk to avoid huge computation graphs
    let mut pos = 0;
    while total - pos > PREFILL_STEP_SIZE {
        let chunk = &prompt_tokens[pos..pos + PREFILL_STEP_SIZE];
        let input = Array::from_slice(chunk, &[1, PREFILL_STEP_SIZE as i32]);
        model.forward(&input, caches)?;
        mlx_rs::transforms::eval(caches.iter().flat_map(|c| c.arrays()))?;
        pos += PREFILL_STEP_SIZE;
    }

    // Final chunk — get logits for the first generated token
    let last_chunk = &prompt_tokens[pos..];
    let input = Array::from_slice(last_chunk, &[1, last_chunk.len() as i32]);
    model.forward(&input, caches)
}

/// Find the longest common prefix between cached tokens and new tokens.
fn common_prefix_len(cached: &[u32], new: &[u32]) -> usize {
    cached
        .iter()
        .zip(new.iter())
        .take_while(|(a, b)| a == b)
        .count()
}

/// Set up caches for a new request, reusing the prompt cache if possible.
/// Returns (caches, tokens_to_prefill) where tokens_to_prefill is the
/// suffix of prompt_tokens that still needs to be forwarded.
fn setup_caches_with_reuse<'a>(
    state: &mut InferState,
    prompt_tokens: &'a [u32],
) -> (Vec<model::KVCache>, &'a [u32]) {
    if let Some(ref cached) = state.prompt_cache {
        let prefix_len = common_prefix_len(&cached.tokens, prompt_tokens);
        if prefix_len > 0 {
            // Reuse cached KV — trim to prefix length and return suffix
            let mut caches = state.prompt_cache.take().unwrap().caches;
            for c in &mut caches {
                c.trim_to(prefix_len);
            }
            tracing::info!(
                "MLX prompt cache: reusing {prefix_len}/{} tokens ({} new)",
                prompt_tokens.len(),
                prompt_tokens.len() - prefix_len,
            );
            return (caches, &prompt_tokens[prefix_len..]);
        }
    }
    // No cache hit — fresh caches
    (state.model.new_caches(), prompt_tokens)
}

/// Save caches + tokens for future reuse.
fn save_prompt_cache(state: &mut InferState, tokens: Vec<u32>, caches: Vec<model::KVCache>) {
    state.prompt_cache = Some(PromptCache { tokens, caches });
}

/// Run inference synchronously (called from blocking thread).
fn run_inference(
    state: &mut InferState,
    prompt: &str,
    generation: &GenerationConfig,
) -> Result<GenerationOutcome> {
    let encoding = state
        .model
        .tokenizer
        .encode(prompt, false)
        .map_err(|e| anyhow::anyhow!("tokenizer encode: {e}"))?;
    let prompt_tokens: Vec<u32> = encoding.get_ids().to_vec();
    let prompt_len = prompt_tokens.len();

    let (mut caches, suffix) = setup_caches_with_reuse(state, &prompt_tokens);
    let mut sampler = Sampler::new(generation.sampling.clone());
    let mut stop_buffer = StopBuffer::new(generation.stop_sequences.clone());

    if generation.max_tokens == 0 {
        if !suffix.is_empty() {
            let _ = prefill_logits(&state.model, suffix, &mut caches)?;
        }
        save_prompt_cache(state, prompt_tokens, caches);
        return Ok(GenerationOutcome {
            text: String::new(),
            prompt_tokens: prompt_len,
            completion_tokens: 0,
            finish_reason: "length",
        });
    }

    // Prefill only the new suffix
    let mut next_token = if suffix.is_empty() {
        // Entire prompt was cached — re-forward last token to get logits
        let last = prompt_tokens[prompt_tokens.len() - 1];
        // Rewind one position so the last token gets forwarded
        let rewind_to = caches[0].offset.saturating_sub(1);
        for c in &mut caches {
            c.trim_to(rewind_to);
        }
        let input = Array::from_slice(&[last], &[1, 1]);
        let logits = state.model.forward(&input, &mut caches)?;
        sampler.sample_next_token(&logits)?
    } else {
        let logits = prefill_logits(&state.model, suffix, &mut caches)?;
        sampler.sample_next_token(&logits)?
    };

    let mut text = String::new();
    let mut completion_tokens = 0usize;
    let mut finish_reason = "length";

    // Decode
    for _ in 0..generation.max_tokens {
        if is_eos(next_token, &state.model.config) {
            finish_reason = "stop";
            break;
        }
        completion_tokens += 1;
        let piece = state
            .model
            .tokenizer
            .decode(&[next_token], true)
            .map_err(|e| anyhow::anyhow!("tokenizer decode: {e}"))?;
        let chunk = stop_buffer.push(&piece);
        text.push_str(&chunk.emit);
        if chunk.matched {
            finish_reason = "stop";
            break;
        }
        let input = Array::from_slice(&[next_token], &[1, 1]);
        let logits = state.model.forward(&input, &mut caches)?;
        next_token = sampler.sample_next_token(&logits)?;
    }

    text.push_str(&stop_buffer.finish());

    // Save prompt cache for next request (prompt only, not generated tokens)
    save_prompt_cache(state, prompt_tokens, caches);

    Ok(GenerationOutcome {
        text,
        prompt_tokens: prompt_len,
        completion_tokens,
        finish_reason,
    })
}

/// Run inference with per-token callback for streaming.
fn run_inference_streaming(
    state: &mut InferState,
    prompt: &str,
    generation: &GenerationConfig,
    tx: &tokio::sync::mpsc::Sender<StreamEvent>,
) -> Result<&'static str> {
    let encoding = state
        .model
        .tokenizer
        .encode(prompt, false)
        .map_err(|e| anyhow::anyhow!("tokenizer encode: {e}"))?;
    let prompt_tokens: Vec<u32> = encoding.get_ids().to_vec();

    let (mut caches, suffix) = setup_caches_with_reuse(state, &prompt_tokens);
    let mut sampler = Sampler::new(generation.sampling.clone());
    let mut stop_buffer = StopBuffer::new(generation.stop_sequences.clone());

    if generation.max_tokens == 0 {
        if !suffix.is_empty() {
            let _ = prefill_logits(&state.model, suffix, &mut caches)?;
        }
        save_prompt_cache(state, prompt_tokens, caches);
        return Ok("length");
    }

    // Prefill only the new suffix
    let mut next_token = if suffix.is_empty() {
        let rewind_to = caches[0].offset.saturating_sub(1);
        for c in &mut caches {
            c.trim_to(rewind_to);
        }
        let last = prompt_tokens[prompt_tokens.len() - 1];
        let input = Array::from_slice(&[last], &[1, 1]);
        let logits = state.model.forward(&input, &mut caches)?;
        sampler.sample_next_token(&logits)?
    } else {
        let logits = prefill_logits(&state.model, suffix, &mut caches)?;
        sampler.sample_next_token(&logits)?
    };

    let mut finish_reason = "length";

    // Decode + stream
    for _ in 0..generation.max_tokens {
        if is_eos(next_token, &state.model.config) {
            finish_reason = "stop";
            break;
        }

        let piece = state
            .model
            .tokenizer
            .decode(&[next_token], true)
            .map_err(|e| anyhow::anyhow!("tokenizer decode: {e}"))?;
        let chunk = stop_buffer.push(&piece);
        if !chunk.emit.is_empty() {
            if tx.blocking_send(StreamEvent::Text(chunk.emit)).is_err() {
                break; // client disconnected
            }
        }
        if chunk.matched {
            finish_reason = "stop";
            break;
        }

        let input = Array::from_slice(&[next_token], &[1, 1]);
        let logits = state.model.forward(&input, &mut caches)?;
        next_token = sampler.sample_next_token(&logits)?;
    }

    let tail = stop_buffer.finish();
    if !tail.is_empty() {
        let _ = tx.blocking_send(StreamEvent::Text(tail));
    }

    // Save prompt cache for next request
    save_prompt_cache(state, prompt_tokens, caches);

    Ok(finish_reason)
}

fn is_eos(token: u32, config: &model::ModelConfig) -> bool {
    config.eos_token_id.contains(&token)
}

fn parse_generation_config(req: &serde_json::Value) -> GenerationConfig {
    GenerationConfig {
        max_tokens: req["max_tokens"].as_u64().unwrap_or(2048) as usize,
        sampling: SamplingParams {
            temperature: req["temperature"].as_f64().unwrap_or(0.0) as f32,
            top_p: req["top_p"].as_f64().unwrap_or(1.0) as f32,
            top_k: req["top_k"]
                .as_u64()
                .map(|value| value as usize)
                .filter(|value| *value > 0),
            seed: req["seed"].as_u64(),
        },
        stop_sequences: parse_stop_sequences(req.get("stop")),
    }
}

fn parse_stop_sequences(stop: Option<&serde_json::Value>) -> Vec<String> {
    match stop {
        Some(serde_json::Value::String(text)) if !text.is_empty() => vec![text.clone()],
        Some(serde_json::Value::Array(items)) => items
            .iter()
            .filter_map(|value| value.as_str())
            .filter(|value| !value.is_empty())
            .map(ToOwned::to_owned)
            .collect(),
        _ => Vec::new(),
    }
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
