//! Built-in MiniMax cloud API plugin.
//!
//! Registers the MiniMax inference API as an OpenAI-compatible endpoint inside
//! mesh-llm.  Because the host proxy only forwards plain HTTP, this plugin
//! starts a local HTTP proxy that injects the `Authorization: Bearer` header
//! and forwards requests to `https://api.minimax.io/v1` (or a custom URL set
//! via `MINIMAX_BASE_URL`).
//!
//! Enable by adding to `~/.mesh-llm/config.toml`:
//! ```toml
//! [[plugin]]
//! name = "minimax"
//! ```
//! and setting `MINIMAX_API_KEY` in the environment.

use anyhow::{Context, Result};
use mesh_llm_plugin::{
    capability, plugin_server_info, PluginMetadata, PluginRuntime, PluginStartupPolicy,
};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio_stream::StreamExt;

const DEFAULT_MINIMAX_BASE_URL: &str = "https://api.minimax.io/v1";

/// MiniMax models exposed through the plugin.
const MINIMAX_MODELS: &[&str] = &["MiniMax-M2.7", "MiniMax-M2.7-highspeed"];

/// Hardcoded `/v1/models` response body returned by the local proxy so that
/// the host health-check does not need to call the MiniMax API.
fn models_response_body() -> String {
    let entries: Vec<String> = MINIMAX_MODELS
        .iter()
        .map(|id| {
            format!(
                r#"{{"id":"{id}","object":"model","owned_by":"minimax","capabilities":["text"]}}"#
            )
        })
        .collect();
    format!(r#"{{"object":"list","data":[{}]}}"#, entries.join(","))
}

fn minimax_api_key() -> Option<String> {
    std::env::var("MINIMAX_API_KEY")
        .ok()
        .filter(|k| !k.is_empty())
}

fn minimax_base_url() -> String {
    std::env::var("MINIMAX_BASE_URL")
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| DEFAULT_MINIMAX_BASE_URL.to_string())
}

/// Start a local HTTP proxy and return the port it is listening on.
async fn start_proxy(api_key: String, base_url: String) -> Result<u16> {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .context("MiniMax plugin: failed to bind local proxy port")?;
    let port = listener
        .local_addr()
        .context("MiniMax plugin: local address unavailable")?
        .port();
    tokio::spawn(accept_loop(listener, api_key, base_url));
    Ok(port)
}

/// Accept connections on the local proxy socket.
async fn accept_loop(listener: TcpListener, api_key: String, base_url: String) {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(300))
        .build()
        .expect("MiniMax plugin: failed to build HTTP client");
    loop {
        let Ok((stream, _)) = listener.accept().await else {
            break;
        };
        let api_key = api_key.clone();
        let base_url = base_url.clone();
        let client = client.clone();
        tokio::spawn(async move {
            let _ = handle_connection(stream, client, api_key, base_url).await;
        });
    }
}

const MAX_HEADERS: usize = 64;
const MAX_REQUEST_BYTES: usize = 8 * 1024 * 1024; // 8 MiB

/// Handle one inbound HTTP connection from the mesh-llm proxy.
async fn handle_connection(
    mut stream: tokio::net::TcpStream,
    client: reqwest::Client,
    api_key: String,
    base_url: String,
) -> Result<()> {
    let _ = stream.set_nodelay(true);

    // ── Read until we see the end of the HTTP headers ────────────────────────
    let mut buf = vec![0u8; MAX_REQUEST_BYTES];
    let mut total = 0usize;
    loop {
        let n = stream
            .read(&mut buf[total..])
            .await
            .context("read from client")?;
        if n == 0 {
            return Ok(());
        }
        total += n;
        if buf[..total].windows(4).any(|w| w == b"\r\n\r\n") {
            break;
        }
        if total >= buf.len() {
            return Ok(());
        }
    }

    // ── Parse HTTP request headers ────────────────────────────────────────────
    let mut headers_buf = [httparse::EMPTY_HEADER; MAX_HEADERS];
    let mut req = httparse::Request::new(&mut headers_buf);
    let header_len = match req.parse(&buf[..total])? {
        httparse::Status::Complete(n) => n,
        httparse::Status::Partial => return Ok(()),
    };

    let method = req.method.unwrap_or("POST").to_string();
    let path = req.path.unwrap_or("/").to_string();

    let mut content_length: Option<usize> = None;
    let mut content_type = String::from("application/json");
    for h in req.headers.iter() {
        if h.name.eq_ignore_ascii_case("content-length") {
            content_length = String::from_utf8_lossy(h.value).trim().parse().ok();
        }
        if h.name.eq_ignore_ascii_case("content-type") {
            content_type = String::from_utf8_lossy(h.value).trim().to_string();
        }
    }

    // ── Health-check shortcut: serve /v1/models locally ───────────────────────
    if method.eq_ignore_ascii_case("GET") && path.contains("/models") {
        let body = models_response_body();
        let resp = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nAccess-Control-Allow-Origin: *\r\n\r\n{}",
            body.len(),
            body
        );
        let _ = stream.write_all(resp.as_bytes()).await;
        let _ = stream.shutdown().await;
        return Ok(());
    }

    // ── Read request body ─────────────────────────────────────────────────────
    let already_have = &buf[header_len..total];
    let body: Vec<u8> = if let Some(cl) = content_length {
        if cl <= already_have.len() {
            already_have[..cl].to_vec()
        } else {
            let mut body = already_have.to_vec();
            body.resize(cl, 0);
            stream
                .read_exact(&mut body[already_have.len()..])
                .await
                .context("read body from client")?;
            body
        }
    } else {
        already_have.to_vec()
    };

    // ── Forward to MiniMax ────────────────────────────────────────────────────
    let upstream_url = format!("{}{}", base_url.trim_end_matches('/'), path);
    let method_val =
        reqwest::Method::from_bytes(method.as_bytes()).unwrap_or(reqwest::Method::POST);

    let upstream_resp = client
        .request(method_val, &upstream_url)
        .header(
            reqwest::header::AUTHORIZATION,
            format!("Bearer {api_key}"),
        )
        .header(reqwest::header::CONTENT_TYPE, &content_type)
        .body(body)
        .send()
        .await
        .context("forward to MiniMax API")?;

    let status = upstream_resp.status();
    let resp_content_type = upstream_resp
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("application/json")
        .to_string();

    let is_streaming = resp_content_type.contains("text/event-stream");

    if is_streaming {
        // ── Streaming: chunked transfer encoding ──────────────────────────────
        let header = format!(
            "HTTP/1.1 {status}\r\nContent-Type: {resp_content_type}\r\nTransfer-Encoding: chunked\r\nCache-Control: no-cache\r\n\r\n",
        );
        if stream.write_all(header.as_bytes()).await.is_err() {
            return Ok(());
        }
        let mut byte_stream = upstream_resp.bytes_stream();
        while let Some(chunk) = byte_stream.next().await {
            match chunk {
                Ok(bytes) if !bytes.is_empty() => {
                    let chunk_header = format!("{:x}\r\n", bytes.len());
                    if stream.write_all(chunk_header.as_bytes()).await.is_err() {
                        break;
                    }
                    if stream.write_all(&bytes).await.is_err() {
                        break;
                    }
                    if stream.write_all(b"\r\n").await.is_err() {
                        break;
                    }
                }
                Ok(_) => {}
                Err(_) => break,
            }
        }
        let _ = stream.write_all(b"0\r\n\r\n").await;
    } else {
        // ── Non-streaming: buffered response ──────────────────────────────────
        let resp_bytes = upstream_resp
            .bytes()
            .await
            .context("read response body from MiniMax")?;
        let header = format!(
            "HTTP/1.1 {status}\r\nContent-Type: {resp_content_type}\r\nContent-Length: {}\r\nAccess-Control-Allow-Origin: *\r\n\r\n",
            resp_bytes.len()
        );
        let _ = stream.write_all(header.as_bytes()).await;
        let _ = stream.write_all(&resp_bytes).await;
    }

    let _ = stream.shutdown().await;
    Ok(())
}

fn build_minimax_plugin(name: String, proxy_port: u16) -> mesh_llm_plugin::SimplePlugin {
    let local_url = format!("http://127.0.0.1:{proxy_port}/v1");

    mesh_llm_plugin::plugin! {
        metadata: PluginMetadata::new(
            name.clone(),
            crate::VERSION,
            plugin_server_info(
                "mesh-minimax",
                crate::VERSION,
                "MiniMax Cloud API Plugin",
                "Registers the MiniMax cloud inference API as an OpenAI-compatible endpoint.",
                Some(
                    "Proxies requests to api.minimax.io with MINIMAX_API_KEY authentication.",
                ),
            ),
        ),
        startup_policy: PluginStartupPolicy::Any,
        provides: [
            capability("endpoint:inference"),
            capability("endpoint:inference/openai_compatible"),
        ],
        inference: [
            mesh_llm_plugin::inference::provider("minimax", local_url),
        ],
        health: move |_context| {
            let port = proxy_port;
            Box::pin(async move {
                Ok(format!("proxy_url=http://127.0.0.1:{port}/v1"))
            })
        },
    }
}

pub(crate) async fn run_plugin(name: String) -> Result<()> {
    let api_key = minimax_api_key().context(
        "MINIMAX_API_KEY environment variable is not set; MiniMax plugin cannot start",
    )?;
    let base_url = minimax_base_url();
    let proxy_port = start_proxy(api_key, base_url).await?;
    PluginRuntime::run(build_minimax_plugin(name, proxy_port)).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn models_response_is_valid_json() {
        let body = models_response_body();
        let parsed: serde_json::Value = serde_json::from_str(&body).expect("valid JSON");
        let data = parsed["data"].as_array().expect("data array");
        assert_eq!(data.len(), MINIMAX_MODELS.len());
        let ids: Vec<&str> = data
            .iter()
            .filter_map(|m| m["id"].as_str())
            .collect();
        for model in MINIMAX_MODELS {
            assert!(ids.contains(model), "missing model {model}");
        }
    }

    #[test]
    fn minimax_base_url_defaults_to_api_minimax_io() {
        // Without env var set the default should point to the international API.
        // We cannot unset the env var in a test without unsafe, so we just verify
        // the constant is correct.
        assert!(
            DEFAULT_MINIMAX_BASE_URL.starts_with("https://api.minimax.io"),
            "default base URL must be the MiniMax international API"
        );
    }

    #[test]
    fn minimax_models_list_contains_m2_7() {
        assert!(
            MINIMAX_MODELS.contains(&"MiniMax-M2.7"),
            "MiniMax-M2.7 must be in the model list"
        );
        assert!(
            MINIMAX_MODELS.contains(&"MiniMax-M2.7-highspeed"),
            "MiniMax-M2.7-highspeed must be in the model list"
        );
    }

    #[tokio::test]
    async fn proxy_serves_models_endpoint() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::net::TcpStream;

        // Start proxy with a dummy key (models endpoint doesn't need a real key)
        let port = start_proxy("dummy-key".into(), DEFAULT_MINIMAX_BASE_URL.into())
            .await
            .expect("proxy starts");

        // Connect and send a GET /v1/models request
        let mut client = TcpStream::connect(format!("127.0.0.1:{port}"))
            .await
            .expect("connect");
        client
            .write_all(b"GET /v1/models HTTP/1.1\r\nHost: localhost\r\n\r\n")
            .await
            .expect("send");

        let mut resp = vec![0u8; 4096];
        let n = client.read(&mut resp).await.expect("read");
        let resp_str = String::from_utf8_lossy(&resp[..n]);

        assert!(resp_str.starts_with("HTTP/1.1 200"), "expected 200, got: {resp_str}");
        assert!(
            resp_str.contains("MiniMax-M2.7"),
            "response should list MiniMax-M2.7"
        );
    }
}
