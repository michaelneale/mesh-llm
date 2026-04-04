//! Lemonade integration — discover and connect to a running Lemonade server,
//! or launch `lemond` as a subprocess.
//!
//! Lemonade is a local LLM server that provides GPU and NPU acceleration via
//! multiple backends (llama.cpp, FastFlowLM, whisper.cpp, stable-diffusion.cpp,
//! Kokoro TTS). It exposes an OpenAI-compatible API.
//!
//! Two modes:
//!   1. **Discover** — detect a running Lemonade on the default port (or a
//!      configured port), health-check it, and register its models.
//!   2. **Launch** — start `lemond` as a subprocess (same pattern as llama-server).

use super::launch::{InferenceServerHandle, InferenceServerProcess};
use anyhow::{Context, Result};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// Default Lemonade port.
pub const DEFAULT_PORT: u16 = 13305;

/// Check if a Lemonade server is healthy at the given port.
pub async fn health_check(port: u16) -> bool {
    let addr = format!("127.0.0.1:{port}");
    let Ok(mut stream) = tokio::net::TcpStream::connect(&addr).await else {
        return false;
    };
    let request =
        format!("GET /v1/health HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\n\r\n");
    if stream.write_all(request.as_bytes()).await.is_err() {
        return false;
    }
    let mut response = vec![0u8; 2048];
    let Ok(n) = stream.read(&mut response).await else {
        return false;
    };
    let response = String::from_utf8_lossy(&response[..n]);
    response.contains("200 OK")
}

/// Query `/v1/models` on a running Lemonade and return the model IDs.
pub async fn list_models(port: u16) -> Result<Vec<String>> {
    let addr = format!("127.0.0.1:{port}");
    let mut stream = tokio::net::TcpStream::connect(&addr)
        .await
        .context("connect to lemonade")?;
    let request =
        format!("GET /v1/models HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\n\r\n");
    stream.write_all(request.as_bytes()).await?;

    let mut response = Vec::new();
    stream.read_to_end(&mut response).await?;
    let response = String::from_utf8_lossy(&response);

    // Find the JSON body after headers
    let body = response
        .find("\r\n\r\n")
        .map(|pos| &response[pos + 4..])
        .unwrap_or("");

    let json: serde_json::Value = serde_json::from_str(body)
        .context("parse lemonade /v1/models response")?;

    let models = json
        .get("data")
        .and_then(|d| d.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|m| m.get("id").and_then(|id| id.as_str()).map(String::from))
                .collect()
        })
        .unwrap_or_default();

    Ok(models)
}

/// Connect to an already-running Lemonade server (external process, we don't own it).
///
/// Returns an `InferenceServerProcess` with a no-op shutdown and a death channel
/// that fires when the health check fails.
pub async fn connect_external(port: u16) -> Result<InferenceServerProcess> {
    anyhow::ensure!(
        health_check(port).await,
        "Lemonade server not healthy at port {port}"
    );

    let (death_tx, death_rx) = tokio::sync::oneshot::channel();

    // Watchdog: poll /v1/health every 5s, fire death_tx if it goes away.
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(std::time::Duration::from_secs(5)).await;
            if !health_check(port).await {
                tracing::warn!("Lemonade server on port {port} is no longer healthy");
                drop(death_tx);
                return;
            }
        }
    });

    Ok(InferenceServerProcess {
        handle: InferenceServerHandle::external(),
        death_rx,
        context_length: 0, // unknown for external servers
    })
}

/// Discover a running Lemonade server and return (port, models).
/// Returns None if no Lemonade server is found.
pub async fn discover() -> Option<(u16, Vec<String>)> {
    if !health_check(DEFAULT_PORT).await {
        return None;
    }
    let models = list_models(DEFAULT_PORT).await.ok()?;
    if models.is_empty() {
        return None;
    }
    Some((DEFAULT_PORT, models))
}

/// Discover a running Lemonade server and connect to it.
/// Returns (port, model_names, process) or None.
pub async fn discover_and_connect() -> Option<(u16, Vec<String>, InferenceServerProcess)> {
    let (port, models) = discover().await?;
    let process = connect_external(port).await.ok()?;
    Some((port, models, process))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Run with: cargo test -p mesh-llm lemonade_live -- --ignored --nocapture
    /// Requires a running Lemonade server on port 8000.
    #[tokio::test]
    #[ignore]
    async fn lemonade_live_health() {
        let healthy = health_check(8000).await;
        assert!(healthy, "Lemonade should be healthy on port 8000");

        let not_healthy = health_check(13305).await;
        // May or may not be running on default port — just don't crash
        eprintln!("health_check(13305) = {not_healthy}");
    }

    #[tokio::test]
    #[ignore]
    async fn lemonade_live_list_models() {
        let models = list_models(8000).await.expect("should list models");
        eprintln!("models: {models:?}");
        assert!(!models.is_empty(), "should have at least one model loaded");
    }

    #[tokio::test]
    #[ignore]
    async fn lemonade_live_connect() {
        let process = connect_external(8000).await.expect("should connect");
        assert!(process.handle.is_external());
        eprintln!("connected, context_length={}", process.context_length);
    }

    /// Verify that a raw HTTP request to Lemonade's port works the same way
    /// the proxy would forward it (TCP connect, write raw HTTP, read response).
    #[tokio::test]
    #[ignore]
    async fn lemonade_live_chat_completion() {
        let body = serde_json::json!({
            "model": "Qwen3-0.6B-GGUF",
            "messages": [{"role": "user", "content": "Say hi"}],
            "max_tokens": 10
        });
        let body_str = body.to_string();
        let request = format!(
            "POST /v1/chat/completions HTTP/1.1\r\n\
             Host: 127.0.0.1:8000\r\n\
             Content-Type: application/json\r\n\
             Content-Length: {}\r\n\
             Connection: close\r\n\r\n{}",
            body_str.len(),
            body_str
        );

        let mut stream = tokio::net::TcpStream::connect("127.0.0.1:8000")
            .await
            .expect("connect");
        stream
            .write_all(request.as_bytes())
            .await
            .expect("write request");

        let mut response = Vec::new();
        stream.read_to_end(&mut response).await.expect("read response");
        let response = String::from_utf8_lossy(&response);

        eprintln!("response preview: {}", &response[..response.len().min(500)]);
        assert!(response.contains("200 OK"), "should get 200 OK");
        assert!(
            response.contains("chat.completion"),
            "should contain chat.completion object"
        );
    }
}
