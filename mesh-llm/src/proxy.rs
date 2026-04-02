//! HTTP proxy plumbing — request parsing, model routing, response helpers.
//!
//! Used by the API proxy (port 9337), bootstrap proxy, and passive mode.
//! All inference traffic flows through these functions.

use crate::{
    affinity::{prepare_remote_targets_for_request, AffinityRouter, PreparedTargets},
    election, mesh, router,
};
use anyhow::{anyhow, bail, Context, Result};
use std::time::Duration;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

const MAX_HEADER_BYTES: usize = 64 * 1024;
const MAX_BODY_BYTES: usize = 8 * 1024 * 1024;
const MAX_CHUNKED_WIRE_BYTES: usize = MAX_BODY_BYTES * 6 + 64 * 1024;
const MAX_HEADERS: usize = 64;
const MAX_RESPONSE_BODY_PREVIEW_BYTES: usize = 4 * 1024;
const REQUEST_TOKEN_MARGIN: u32 = 256;

#[derive(Debug, Clone, Copy)]
struct HttpReadLimits {
    max_header_bytes: usize,
    max_body_bytes: usize,
    max_chunked_wire_bytes: usize,
}

const HTTP_READ_LIMITS: HttpReadLimits = HttpReadLimits {
    max_header_bytes: MAX_HEADER_BYTES,
    max_body_bytes: MAX_BODY_BYTES,
    max_chunked_wire_bytes: MAX_CHUNKED_WIRE_BYTES,
};

/// Parsed header metadata extracted via httparse.
struct ParsedHeaders {
    header_end: usize,
    method: String,
    path: String,
    content_length: Option<usize>,
    is_chunked: bool,
    expects_continue: bool,
}

#[derive(Debug)]
pub struct BufferedHttpRequest {
    pub raw: Vec<u8>,
    pub method: String,
    pub path: String,
    pub body_json: Option<serde_json::Value>,
    pub model_name: Option<String>,
    pub session_hint: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineProxyResult {
    Handled,
    FallbackToDirect,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RouteAttemptResult {
    Delivered { status_code: u16 },
    RetryableUnavailable,
    RetryableContextOverflow,
}

struct ParsedResponseHeaders {
    header_end: usize,
    status_code: u16,
    content_length: Option<usize>,
}

struct ResponseProbe {
    buffered: Vec<u8>,
    status_code: u16,
    retryable_context_overflow: bool,
}

// ── Request parsing ──

/// Read and buffer one HTTP request for routing decisions.
///
/// This reads complete headers plus the full request body when body framing is
/// known via `Content-Length` or `Transfer-Encoding: chunked`. The raw request
/// bytes are preserved so the chosen upstream sees the original payload.
pub async fn read_http_request(stream: &mut TcpStream) -> Result<BufferedHttpRequest> {
    read_http_request_with_limits(stream, HTTP_READ_LIMITS).await
}

async fn read_http_request_with_limits(
    stream: &mut TcpStream,
    limits: HttpReadLimits,
) -> Result<BufferedHttpRequest> {
    let mut raw = Vec::with_capacity(8192);
    let parsed = read_until_headers_parsed(stream, &mut raw, limits.max_header_bytes).await?;

    let header_end = parsed.header_end;

    let body = if parsed.is_chunked {
        let mut sent_continue = false;
        loop {
            if let Some((consumed, decoded)) =
                try_decode_chunked_body(&raw[header_end..], limits.max_body_bytes)?
            {
                raw.truncate(header_end + consumed);
                break decoded;
            }
            if !sent_continue && parsed.expects_continue {
                stream.write_all(b"HTTP/1.1 100 Continue\r\n\r\n").await?;
                sent_continue = true;
            }
            read_more(stream, &mut raw).await?;
            if raw.len().saturating_sub(header_end) > limits.max_chunked_wire_bytes {
                bail!(
                    "HTTP chunked wire body exceeds {} bytes",
                    limits.max_chunked_wire_bytes
                );
            }
        }
    } else if let Some(content_length) = parsed.content_length {
        if content_length > limits.max_body_bytes {
            bail!("HTTP body exceeds {} bytes", limits.max_body_bytes);
        }
        let body_end = header_end + content_length;
        let mut sent_continue = false;
        while raw.len() < body_end {
            if !sent_continue && parsed.expects_continue && content_length > 0 {
                stream.write_all(b"HTTP/1.1 100 Continue\r\n\r\n").await?;
                sent_continue = true;
            }
            read_more(stream, &mut raw).await?;
        }
        raw.truncate(body_end);
        raw[header_end..body_end].to_vec()
    } else {
        raw.truncate(header_end);
        Vec::new()
    };

    let body_json = if body.is_empty() {
        None
    } else {
        serde_json::from_slice(&body).ok()
    };
    let model_name = body_json.as_ref().and_then(extract_model_from_json);
    let session_hint = body_json.as_ref().and_then(extract_session_hint_from_json);
    let raw = finalize_forwarded_request(raw, header_end, parsed.expects_continue)?;

    Ok(BufferedHttpRequest {
        raw,
        method: parsed.method,
        path: parsed.path,
        body_json,
        model_name,
        session_hint,
    })
}

fn finalize_forwarded_request(
    mut raw: Vec<u8>,
    header_end: usize,
    strip_expect: bool,
) -> Result<Vec<u8>> {
    let body = raw.split_off(header_end);
    // Re-parse with httparse so we iterate over validated header structs.
    let mut headers_buf = [httparse::EMPTY_HEADER; MAX_HEADERS];
    let mut req = httparse::Request::new(&mut headers_buf);
    let _ = req.parse(&raw).context("re-parse headers for forwarding")?;

    let method = req.method.unwrap_or("GET");
    let path = req.path.unwrap_or("/");
    let version = req.version.unwrap_or(1);

    let mut rebuilt = format!("{method} {path} HTTP/1.{version}\r\n");

    for header in req.headers.iter() {
        let name = header.name;
        if name.eq_ignore_ascii_case("connection") {
            continue;
        }
        if strip_expect && name.eq_ignore_ascii_case("expect") {
            continue;
        }
        let value = std::str::from_utf8(header.value).unwrap_or("");
        rebuilt.push_str(&format!("{name}: {value}\r\n"));
    }

    // The proxy buffers exactly one request for routing, so force a single-request
    // connection contract upstream instead of reusing the client connection blindly.
    rebuilt.push_str("Connection: close\r\n\r\n");

    let mut forwarded = rebuilt.into_bytes();
    forwarded.extend_from_slice(&body);
    Ok(forwarded)
}

/// Read from the stream until httparse can fully parse the request headers.
/// Returns parsed metadata; `buf` contains all bytes read so far (headers +
/// any trailing body bytes that arrived in the same read).
async fn read_until_headers_parsed(
    stream: &mut TcpStream,
    buf: &mut Vec<u8>,
    max_header_bytes: usize,
) -> Result<ParsedHeaders> {
    loop {
        let mut headers_buf = [httparse::EMPTY_HEADER; MAX_HEADERS];
        let mut req = httparse::Request::new(&mut headers_buf);
        match req.parse(buf) {
            Ok(httparse::Status::Complete(header_end)) => {
                let method = req.method.unwrap_or("GET").to_string();
                let path = req.path.unwrap_or("/").to_string();

                let mut content_length = None;
                let mut is_chunked = false;
                let mut expects_continue = false;

                for header in req.headers.iter() {
                    if header.name.eq_ignore_ascii_case("content-length") {
                        let val = std::str::from_utf8(header.value)
                            .context("invalid Content-Length encoding")?;
                        content_length = Some(
                            val.trim()
                                .parse::<usize>()
                                .with_context(|| format!("invalid Content-Length: {val}"))?,
                        );
                    } else if header.name.eq_ignore_ascii_case("transfer-encoding") {
                        let val = std::str::from_utf8(header.value).unwrap_or("");
                        is_chunked = val
                            .split(',')
                            .any(|part| part.trim().eq_ignore_ascii_case("chunked"));
                    } else if header.name.eq_ignore_ascii_case("expect") {
                        let val = std::str::from_utf8(header.value).unwrap_or("");
                        expects_continue = val
                            .split(',')
                            .any(|part| part.trim().eq_ignore_ascii_case("100-continue"));
                    }
                }

                // RFC 7230 §3.3.3: if both Transfer-Encoding and Content-Length
                // are present, Transfer-Encoding wins and Content-Length is ignored.
                if is_chunked {
                    content_length = None;
                }

                return Ok(ParsedHeaders {
                    header_end,
                    method,
                    path,
                    content_length,
                    is_chunked,
                    expects_continue,
                });
            }
            Ok(httparse::Status::Partial) => {
                if buf.len() >= max_header_bytes {
                    bail!("HTTP headers exceed {max_header_bytes} bytes");
                }
                read_more(stream, buf).await?;
            }
            Err(e) => bail!("HTTP parse error: {e}"),
        }
    }
}

async fn read_more(stream: &mut TcpStream, buf: &mut Vec<u8>) -> Result<()> {
    let mut chunk = [0u8; 8192];
    let n = stream.read(&mut chunk).await?;
    if n == 0 {
        bail!("unexpected EOF while reading HTTP request");
    }
    buf.extend_from_slice(&chunk[..n]);
    Ok(())
}

fn try_decode_chunked_body(buf: &[u8], max_body_bytes: usize) -> Result<Option<(usize, Vec<u8>)>> {
    let mut pos = 0usize;
    let mut decoded = Vec::new();

    loop {
        let Some(line_end_rel) = buf[pos..].windows(2).position(|window| window == b"\r\n") else {
            return Ok(None);
        };
        let line_end = pos + line_end_rel;
        let size_line = std::str::from_utf8(&buf[pos..line_end]).context("invalid chunk header")?;
        let size_text = size_line.split(';').next().unwrap_or("").trim();
        let size = usize::from_str_radix(size_text, 16)
            .with_context(|| format!("invalid chunk size: {size_text}"))?;
        pos = line_end + 2;

        if size == 0 {
            if buf.len() < pos + 2 {
                return Ok(None);
            }
            if &buf[pos..pos + 2] == b"\r\n" {
                return Ok(Some((pos + 2, decoded)));
            }
            let Some(trailer_end_rel) = buf[pos..]
                .windows(4)
                .position(|window| window == b"\r\n\r\n")
            else {
                return Ok(None);
            };
            return Ok(Some((pos + trailer_end_rel + 4, decoded)));
        }

        if buf.len() < pos + size + 2 {
            return Ok(None);
        }
        decoded.extend_from_slice(&buf[pos..pos + size]);
        pos += size;

        if &buf[pos..pos + 2] != b"\r\n" {
            return Err(anyhow!("invalid chunk terminator"));
        }
        pos += 2;

        if decoded.len() > max_body_bytes {
            bail!("HTTP chunked body exceeds {max_body_bytes} bytes");
        }
    }
}

fn extract_model_from_json(body: &serde_json::Value) -> Option<String> {
    body.get("model")
        .and_then(|value| value.as_str())
        .map(ToString::to_string)
}

fn extract_session_hint_from_json(body: &serde_json::Value) -> Option<String> {
    ["user", "session_id"].into_iter().find_map(|key| {
        body.get(key)
            .and_then(|value| value.as_str())
            .map(ToString::to_string)
    })
}

fn response_first_byte_timeout() -> Duration {
    std::env::var("MESH_LLM_TUNNEL_FIRST_BYTE_TIMEOUT_SECS")
        .ok()
        .and_then(|raw| raw.parse::<u64>().ok())
        .filter(|secs| *secs > 0)
        .map(Duration::from_secs)
        .unwrap_or_else(|| Duration::from_secs(60))
}

fn saturating_u32(value: usize) -> u32 {
    value.try_into().unwrap_or(u32::MAX)
}

fn ceil_div_u32(value: u32, divisor: u32) -> u32 {
    value.saturating_add(divisor - 1) / divisor
}

fn request_budget_tokens(body: &serde_json::Value) -> Option<u32> {
    let serialized = serde_json::to_vec(body).ok()?;
    let prompt_tokens = ceil_div_u32(saturating_u32(serialized.len()), 4);
    let completion_tokens = [
        "max_completion_tokens",
        "max_tokens",
        "max_output_tokens",
        "n_predict",
    ]
    .into_iter()
    .find_map(|key| body.get(key).and_then(|value| value.as_u64()))
    .map(|value| value.min(u32::MAX as u64) as u32)
    .unwrap_or(0);
    Some(
        prompt_tokens
            .saturating_add(completion_tokens)
            .saturating_add(REQUEST_TOKEN_MARGIN),
    )
}

fn reorder_candidates_by_context<T: Clone>(
    candidates: &[(T, Option<u32>)],
    required_tokens: Option<u32>,
) -> Vec<T> {
    let Some(required_tokens) = required_tokens else {
        return candidates
            .iter()
            .map(|(candidate, _)| candidate.clone())
            .collect();
    };

    let mut adequate = Vec::new();
    let mut unknown = Vec::new();
    for (candidate, context_length) in candidates {
        match context_length {
            Some(value) if *value >= required_tokens => adequate.push(candidate.clone()),
            Some(_) => {}
            None => unknown.push(candidate.clone()),
        }
    }

    if adequate.is_empty() && unknown.is_empty() {
        candidates
            .iter()
            .map(|(candidate, _)| candidate.clone())
            .collect()
    } else {
        adequate.extend(unknown);
        adequate
    }
}

async fn order_remote_hosts_by_context(
    node: &mesh::Node,
    model: &str,
    body_json: Option<&serde_json::Value>,
    hosts: &[iroh::EndpointId],
) -> Vec<iroh::EndpointId> {
    let required_tokens = body_json.and_then(request_budget_tokens);
    let mut candidates = Vec::with_capacity(hosts.len());
    for host in hosts {
        candidates.push((*host, node.peer_model_context_length(*host, model).await));
    }
    reorder_candidates_by_context(&candidates, required_tokens)
}

async fn order_targets_by_context(
    node: &mesh::Node,
    model: &str,
    body_json: Option<&serde_json::Value>,
    targets: &[election::InferenceTarget],
) -> Vec<election::InferenceTarget> {
    let required_tokens = body_json.and_then(request_budget_tokens);
    let mut candidates = Vec::with_capacity(targets.len());
    for target in targets {
        let context_length = match target {
            election::InferenceTarget::Local(_) | election::InferenceTarget::MoeLocal(_) => {
                node.local_model_context_length(model).await
            }
            election::InferenceTarget::Remote(peer_id)
            | election::InferenceTarget::MoeRemote(peer_id) => {
                node.peer_model_context_length(*peer_id, model).await
            }
            election::InferenceTarget::None => None,
        };
        candidates.push((target.clone(), context_length));
    }
    reorder_candidates_by_context(&candidates, required_tokens)
}

fn move_target_first<T: PartialEq>(targets: &mut [T], target: &T) -> bool {
    if let Some(pos) = targets.iter().position(|candidate| candidate == target) {
        targets[..=pos].rotate_right(1);
        true
    } else {
        false
    }
}

fn response_message_text(json: &serde_json::Value) -> Option<String> {
    fn value_to_text(value: &serde_json::Value) -> Option<String> {
        match value {
            serde_json::Value::String(text) => Some(text.clone()),
            serde_json::Value::Object(map) => map
                .get("message")
                .and_then(value_to_text)
                .or_else(|| map.get("error").and_then(value_to_text)),
            _ => None,
        }
    }

    value_to_text(json)
}

fn is_retryable_context_overflow_response(body: &[u8]) -> bool {
    let text = serde_json::from_slice::<serde_json::Value>(body)
        .ok()
        .and_then(|json| response_message_text(&json))
        .unwrap_or_else(|| String::from_utf8_lossy(body).to_string())
        .to_ascii_lowercase();

    let mentions_context = [
        "context", "n_ctx", "ctx", "prompt", "token", "slot", "window",
    ]
    .into_iter()
    .any(|needle| text.contains(needle));
    let mentions_limit = [
        "exceed",
        "overflow",
        "too long",
        "too many",
        "greater than",
        "longer than",
        "limit",
        "maximum",
    ]
    .into_iter()
    .any(|needle| text.contains(needle));

    mentions_context && mentions_limit
}

pub fn is_models_list_request(method: &str, path: &str) -> bool {
    let path = path.split('?').next().unwrap_or(path);
    method == "GET" && (path == "/v1/models" || path == "/models")
}

pub fn is_drop_request(method: &str, path: &str) -> bool {
    let path = path.split('?').next().unwrap_or(path);
    method == "POST" && path == "/mesh/drop"
}

pub fn pipeline_request_supported(path: &str, body: &serde_json::Value) -> bool {
    let path = path.split('?').next().unwrap_or(path);
    path == "/v1/chat/completions"
        && body
            .get("messages")
            .map(|messages| messages.is_array())
            .unwrap_or(false)
}

fn try_parse_response_headers(buf: &[u8]) -> Result<Option<ParsedResponseHeaders>> {
    let mut headers_buf = [httparse::EMPTY_HEADER; MAX_HEADERS];
    let mut response = httparse::Response::new(&mut headers_buf);
    match response.parse(buf) {
        Ok(httparse::Status::Complete(header_end)) => {
            let mut content_length = None;
            for header in response.headers.iter() {
                if header.name.eq_ignore_ascii_case("content-length") {
                    let value = std::str::from_utf8(header.value)
                        .context("invalid response Content-Length encoding")?;
                    content_length =
                        Some(value.trim().parse::<usize>().with_context(|| {
                            format!("invalid response Content-Length: {value}")
                        })?);
                }
            }
            Ok(Some(ParsedResponseHeaders {
                header_end,
                status_code: response.code.unwrap_or(0),
                content_length,
            }))
        }
        Ok(httparse::Status::Partial) => Ok(None),
        Err(err) => Err(anyhow!("HTTP response parse error: {err}")),
    }
}

async fn read_response_chunk<R: AsyncRead + Unpin>(
    reader: &mut R,
    buf: &mut Vec<u8>,
    with_timeout: bool,
) -> Result<usize> {
    let mut chunk = [0u8; 8192];
    let read_result = if with_timeout {
        tokio::time::timeout(response_first_byte_timeout(), reader.read(&mut chunk))
            .await
            .map_err(|_| {
                anyhow!(
                    "upstream sent no response within {:.3}s",
                    response_first_byte_timeout().as_secs_f64()
                )
            })?
    } else {
        reader.read(&mut chunk).await
    }?;
    if read_result == 0 {
        bail!("unexpected EOF while reading HTTP response");
    }
    buf.extend_from_slice(&chunk[..read_result]);
    Ok(read_result)
}

async fn probe_http_response<R: AsyncRead + Unpin>(reader: &mut R) -> Result<ResponseProbe> {
    let mut buffered = Vec::with_capacity(8192);
    let parsed = loop {
        if let Some(parsed) = try_parse_response_headers(&buffered)? {
            break parsed;
        }
        let first_read = buffered.is_empty();
        read_response_chunk(reader, &mut buffered, first_read).await?;
        if buffered.len() > MAX_HEADER_BYTES {
            bail!("HTTP response headers exceed {MAX_HEADER_BYTES} bytes");
        }
    };

    let preview_len = if parsed.status_code == 400 {
        parsed
            .content_length
            .map(|value| value.min(MAX_RESPONSE_BODY_PREVIEW_BYTES))
            .unwrap_or(0)
    } else {
        0
    };
    while buffered.len() < parsed.header_end + preview_len {
        read_response_chunk(reader, &mut buffered, false).await?;
    }

    let retryable_context_overflow = parsed.status_code == 400
        && preview_len > 0
        && is_retryable_context_overflow_response(
            &buffered[parsed.header_end..parsed.header_end + preview_len],
        );

    Ok(ResponseProbe {
        buffered,
        status_code: parsed.status_code,
        retryable_context_overflow,
    })
}

async fn relay_probed_response<R: AsyncRead + Unpin>(
    mut tcp_stream: &mut TcpStream,
    reader: &mut R,
    probe: ResponseProbe,
    retry_context_overflow: bool,
) -> Result<RouteAttemptResult> {
    if retry_context_overflow && probe.retryable_context_overflow {
        return Ok(RouteAttemptResult::RetryableContextOverflow);
    }

    tcp_stream.write_all(&probe.buffered).await?;
    if let Err(err) = tokio::io::copy(reader, &mut tcp_stream).await {
        tracing::debug!("response relay ended after headers were committed: {err}");
    }
    let _ = tcp_stream.shutdown().await;
    Ok(RouteAttemptResult::Delivered {
        status_code: probe.status_code,
    })
}

async fn route_local_attempt(
    node: &mesh::Node,
    tcp_stream: &mut TcpStream,
    port: u16,
    prefetched: &[u8],
    retry_context_overflow: bool,
) -> RouteAttemptResult {
    match TcpStream::connect(format!("127.0.0.1:{port}")).await {
        Ok(mut upstream) => {
            let _inflight = node.begin_inflight_request();
            let _ = upstream.set_nodelay(true);
            if let Err(err) = upstream.write_all(prefetched).await {
                tracing::warn!(
                    "API proxy: failed to forward buffered request to local llama-server on {port}: {err}"
                );
                return RouteAttemptResult::RetryableUnavailable;
            }
            match probe_http_response(&mut upstream).await {
                Ok(probe) => {
                    let status_code = probe.status_code;
                    match relay_probed_response(
                        tcp_stream,
                        &mut upstream,
                        probe,
                        retry_context_overflow,
                    )
                    .await
                    {
                        Ok(result) => result,
                        Err(err) => {
                            tracing::debug!("API proxy (local) ended after commit: {err}");
                            RouteAttemptResult::Delivered { status_code }
                        }
                    }
                }
                Err(err) => {
                    tracing::warn!(
                        "API proxy: failed to read local response from llama-server on {port}: {err}"
                    );
                    RouteAttemptResult::RetryableUnavailable
                }
            }
        }
        Err(err) => {
            tracing::warn!("API proxy: can't reach llama-server on {port}: {err}");
            RouteAttemptResult::RetryableUnavailable
        }
    }
}

async fn route_remote_attempt(
    node: &mesh::Node,
    tcp_stream: &mut TcpStream,
    host_id: iroh::EndpointId,
    prefetched: &[u8],
    retry_context_overflow: bool,
) -> RouteAttemptResult {
    match node.open_http_tunnel(host_id).await {
        Ok((mut quic_send, mut quic_recv)) => {
            if let Err(err) = quic_send.write_all(prefetched).await {
                tracing::warn!(
                    "API proxy: failed to forward buffered request to host {}: {err}",
                    host_id.fmt_short()
                );
                return RouteAttemptResult::RetryableUnavailable;
            }
            match probe_http_response(&mut quic_recv).await {
                Ok(probe) => {
                    let status_code = probe.status_code;
                    match relay_probed_response(
                        tcp_stream,
                        &mut quic_recv,
                        probe,
                        retry_context_overflow,
                    )
                    .await
                    {
                        Ok(result) => result,
                        Err(err) => {
                            tracing::debug!("API proxy (remote) ended after commit: {err}");
                            RouteAttemptResult::Delivered { status_code }
                        }
                    }
                }
                Err(err) => {
                    tracing::warn!(
                        "API proxy: failed to read response from host {}: {err}",
                        host_id.fmt_short()
                    );
                    RouteAttemptResult::RetryableUnavailable
                }
            }
        }
        Err(err) => {
            tracing::warn!(
                "API proxy: can't tunnel to host {}: {err}",
                host_id.fmt_short()
            );
            RouteAttemptResult::RetryableUnavailable
        }
    }
}

fn should_learn_affinity(status_code: u16) -> bool {
    (200..400).contains(&status_code)
}

// ── Model-aware tunnel routing ──

/// The common request-handling path used by idle proxy, passive proxy, and bootstrap proxy.
///
/// Peeks at the HTTP request, handles `/v1/models`, resolves the target host
/// by model name (or falls back to any host), and tunnels the request via QUIC.
///
/// Set `track_demand` to record requests for demand-based rebalancing.
pub async fn handle_mesh_request(
    node: mesh::Node,
    tcp_stream: TcpStream,
    track_demand: bool,
    affinity: AffinityRouter,
) {
    let mut tcp_stream = tcp_stream;
    let request = match read_http_request(&mut tcp_stream).await {
        Ok(v) => v,
        Err(_) => return,
    };
    let body_json = request.body_json.as_ref();

    // Handle /v1/models
    if is_models_list_request(&request.method, &request.path) {
        let served = node.models_being_served().await;
        let _ = send_models_list(tcp_stream, &served).await;
        return;
    }

    // Demand tracking for rebalancing (done after routing so we track the actual model used)
    // We'll track below after routing resolves the effective model

    // Smart routing: if no model specified (or model="auto"), classify and pick
    let routed_model =
        if request.model_name.is_none() || request.model_name.as_deref() == Some("auto") {
            if let Some(body_json) = request.body_json.as_ref() {
                let cl = router::classify(&body_json);
                let served = node.models_being_served().await;
                let available: Vec<(&str, f64)> =
                    served.iter().map(|name| (name.as_str(), 0.0)).collect();
                let picked = router::pick_model_classified(&cl, &available);
                if let Some(name) = picked {
                    tracing::info!(
                        "router: {:?}/{:?} tools={} → {name}",
                        cl.category,
                        cl.complexity,
                        cl.needs_tools
                    );
                    Some(name.to_string())
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };
    let effective_model = routed_model.or(request.model_name.clone());

    // Demand tracking for rebalancing
    if track_demand {
        if let Some(ref name) = effective_model {
            node.record_request(name);
        }
    }

    // Resolve target hosts by model name, fall back to any host
    let target_hosts = if let Some(ref name) = effective_model {
        node.hosts_for_model(name).await
    } else {
        vec![]
    };
    let target_hosts = if target_hosts.is_empty() {
        match node.any_host().await {
            Some(p) => vec![p.id],
            None => {
                let _ = send_503(tcp_stream).await;
                return;
            }
        }
    } else {
        target_hosts
    };
    let prepared = effective_model
        .as_ref()
        .map(|name| prepare_remote_targets_for_request(name, &target_hosts, body_json, &affinity))
        .unwrap_or(PreparedTargets {
            ordered: target_hosts
                .iter()
                .copied()
                .map(election::InferenceTarget::Remote)
                .collect(),
            learn_prefix_hash: None,
            cached_target: None,
        });
    let target_hosts: Vec<iroh::EndpointId> = prepared
        .ordered
        .iter()
        .filter_map(|target| match target {
            election::InferenceTarget::Remote(host_id) => Some(*host_id),
            _ => None,
        })
        .collect();
    let target_hosts = if let Some(name) = effective_model.as_deref() {
        let ordered = order_remote_hosts_by_context(&node, name, body_json, &target_hosts).await;
        if let (Some(prefix_hash), Some(cached_target)) =
            (prepared.learn_prefix_hash, prepared.cached_target.as_ref())
        {
            if let election::InferenceTarget::Remote(cached_host) = cached_target {
                let required_tokens = body_json.and_then(request_budget_tokens);
                let cached_context = node.peer_model_context_length(*cached_host, name).await;
                if matches!(
                    (required_tokens, cached_context),
                    (Some(required), Some(context)) if context < required
                ) {
                    affinity.forget_target(name, prefix_hash, cached_target);
                }
            }
        }
        ordered
    } else {
        target_hosts
    };

    // Try each host in order — if tunnel fails, retry with next.
    // On first failure, trigger background gossip refresh so future requests
    // have a fresh routing table (doesn't block the retry loop).
    let mut last_retryable = false;
    let mut refreshed = false;
    for target_host in &target_hosts {
        match route_remote_attempt(&node, &mut tcp_stream, *target_host, &request.raw, true).await {
            RouteAttemptResult::Delivered { status_code } => {
                if should_learn_affinity(status_code) {
                    if let (Some(name), Some(prefix_hash)) =
                        (effective_model.as_ref(), prepared.learn_prefix_hash)
                    {
                        let target = election::InferenceTarget::Remote(*target_host);
                        affinity.learn_target(name, prefix_hash, &target);
                    }
                }
                return;
            }
            RouteAttemptResult::RetryableContextOverflow => {
                if let (Some(name), Some(prefix_hash), Some(cached_target)) = (
                    effective_model.as_ref(),
                    prepared.learn_prefix_hash,
                    prepared.cached_target.as_ref(),
                ) {
                    let failed = election::InferenceTarget::Remote(*target_host);
                    if cached_target == &failed {
                        affinity.forget_target(name, prefix_hash, &failed);
                    }
                }
                tracing::warn!(
                    "Host {} rejected request with context overflow-style 400, trying next",
                    target_host.fmt_short()
                );
                last_retryable = true;
            }
            RouteAttemptResult::RetryableUnavailable => {
                if let (Some(name), Some(prefix_hash), Some(cached_target)) = (
                    effective_model.as_ref(),
                    prepared.learn_prefix_hash,
                    prepared.cached_target.as_ref(),
                ) {
                    let failed = election::InferenceTarget::Remote(*target_host);
                    if cached_target == &failed {
                        affinity.forget_target(name, prefix_hash, &failed);
                    }
                }
                tracing::warn!(
                    "Failed to tunnel to host {}, trying next",
                    target_host.fmt_short()
                );
                last_retryable = true;
                // Background refresh on first failure — non-blocking
                if !refreshed {
                    let refresh_node = node.clone();
                    tokio::spawn(async move {
                        refresh_node.gossip_one_peer().await;
                    });
                    refreshed = true;
                }
            }
        }
    }
    // All hosts failed
    if last_retryable {
        tracing::warn!("All hosts failed for model {:?}", effective_model);
    }
    let _ = send_503(tcp_stream).await;
}

async fn route_attempt_for_target(
    node: &mesh::Node,
    tcp_stream: &mut TcpStream,
    target: &election::InferenceTarget,
    prefetched: &[u8],
    retry_context_overflow: bool,
) -> RouteAttemptResult {
    match target {
        election::InferenceTarget::Local(port) | election::InferenceTarget::MoeLocal(port) => {
            route_local_attempt(node, tcp_stream, *port, prefetched, retry_context_overflow).await
        }
        election::InferenceTarget::Remote(host_id)
        | election::InferenceTarget::MoeRemote(host_id) => {
            route_remote_attempt(
                node,
                tcp_stream,
                *host_id,
                prefetched,
                retry_context_overflow,
            )
            .await
        }
        election::InferenceTarget::None => RouteAttemptResult::RetryableUnavailable,
    }
}

pub async fn route_model_request(
    node: mesh::Node,
    tcp_stream: TcpStream,
    targets: &election::ModelTargets,
    model: &str,
    parsed_body: Option<&serde_json::Value>,
    prefetched: &[u8],
    affinity: &AffinityRouter,
) -> bool {
    let mut tcp_stream = tcp_stream;
    let ordered_candidates =
        order_targets_by_context(&node, model, parsed_body, &targets.candidates(model)).await;
    if ordered_candidates.is_empty() {
        return false;
    }

    let mut filtered_targets = targets.clone();
    filtered_targets
        .targets
        .insert(model.to_string(), ordered_candidates.clone());
    let selection = crate::affinity::select_model_target_for_request(
        &filtered_targets,
        model,
        parsed_body,
        affinity,
    );
    if matches!(selection.target, election::InferenceTarget::None) {
        let _ = send_503(tcp_stream).await;
        return true;
    }

    if let (Some(prefix_hash), Some(cached_target)) = (
        selection.learn_prefix_hash,
        selection.cached_target.as_ref(),
    ) {
        let required_tokens = parsed_body.and_then(request_budget_tokens);
        let cached_context = match cached_target {
            election::InferenceTarget::Local(_) | election::InferenceTarget::MoeLocal(_) => {
                node.local_model_context_length(model).await
            }
            election::InferenceTarget::Remote(peer_id)
            | election::InferenceTarget::MoeRemote(peer_id) => {
                node.peer_model_context_length(*peer_id, model).await
            }
            election::InferenceTarget::None => None,
        };
        if matches!(
            (required_tokens, cached_context),
            (Some(required), Some(context)) if context < required
        ) {
            affinity.forget_target(model, prefix_hash, cached_target);
        }
    }

    let mut ordered = ordered_candidates;
    move_target_first(&mut ordered, &selection.target);
    let mut refreshed = false;
    for target in ordered {
        match route_attempt_for_target(&node, &mut tcp_stream, &target, prefetched, true).await {
            RouteAttemptResult::Delivered { status_code } => {
                if should_learn_affinity(status_code) {
                    if let Some(prefix_hash) = selection.learn_prefix_hash {
                        affinity.learn_target(model, prefix_hash, &target);
                    }
                }
                return true;
            }
            RouteAttemptResult::RetryableContextOverflow => {
                if let (Some(prefix_hash), Some(cached_target)) = (
                    selection.learn_prefix_hash,
                    selection.cached_target.as_ref(),
                ) {
                    if cached_target == &target {
                        affinity.forget_target(model, prefix_hash, &target);
                    }
                }
                tracing::warn!("Target {target:?} rejected request with context overflow-style 400, trying next");
            }
            RouteAttemptResult::RetryableUnavailable => {
                if let (Some(prefix_hash), Some(cached_target)) = (
                    selection.learn_prefix_hash,
                    selection.cached_target.as_ref(),
                ) {
                    if cached_target == &target {
                        affinity.forget_target(model, prefix_hash, &target);
                    }
                }
                if !refreshed {
                    let refresh_node = node.clone();
                    tokio::spawn(async move {
                        refresh_node.gossip_one_peer().await;
                    });
                    refreshed = true;
                }
                tracing::warn!("Target {target:?} unavailable, trying next");
            }
        }
    }

    let _ = send_503(tcp_stream).await;
    true
}

/// Route a request to a known inference target (local llama-server or remote host).
///
/// Used by the API proxy after election has determined the target.
pub async fn route_to_target(
    node: mesh::Node,
    tcp_stream: TcpStream,
    target: election::InferenceTarget,
    prefetched: &[u8],
) -> bool {
    let mut tcp_stream = tcp_stream;
    tracing::info!("API proxy: routing to target {target:?}");
    match route_attempt_for_target(&node, &mut tcp_stream, &target, prefetched, false).await {
        RouteAttemptResult::Delivered { .. } => true,
        RouteAttemptResult::RetryableContextOverflow | RouteAttemptResult::RetryableUnavailable => {
            let _ = send_503(tcp_stream).await;
            false
        }
    }
}

// ── Response helpers ──

pub async fn send_models_list(mut stream: TcpStream, models: &[String]) -> std::io::Result<()> {
    let data: Vec<serde_json::Value> = models
        .iter()
        .map(|m| {
            let capabilities = crate::models::installed_model_capabilities(m);
            let has_vision = capabilities.supports_vision_runtime();
            let mut caps = vec!["text"];
            if has_vision {
                caps.push("vision");
            }
            if capabilities.reasoning_label().is_some() {
                caps.push("reasoning");
            }
            let display_name = crate::models::installed_model_display_name(m);
            serde_json::json!({
                "id": m,
                "display_name": display_name,
                "object": "model",
                "owned_by": "mesh-llm",
                "capabilities": caps,
                "vision_status": capabilities.vision_status(),
                "reasoning_status": capabilities.reasoning_status(),
            })
        })
        .collect();
    let body = serde_json::json!({ "object": "list", "data": data }).to_string();
    let resp = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nAccess-Control-Allow-Origin: *\r\n\r\n{}",
        body.len(), body
    );
    stream.write_all(resp.as_bytes()).await?;
    stream.shutdown().await?;
    Ok(())
}

pub async fn send_json_ok(mut stream: TcpStream, data: &serde_json::Value) -> std::io::Result<()> {
    let body = data.to_string();
    let resp = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        body.len(),
        body
    );
    stream.write_all(resp.as_bytes()).await?;
    stream.shutdown().await?;
    Ok(())
}

pub async fn send_400(mut stream: TcpStream, msg: &str) -> std::io::Result<()> {
    let body = format!("{{\"error\":\"{msg}\"}}");
    let resp = format!(
        "HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        body.len(), body
    );
    stream.write_all(resp.as_bytes()).await?;
    stream.shutdown().await?;
    Ok(())
}

pub async fn send_error(mut stream: TcpStream, code: u16, msg: &str) -> std::io::Result<()> {
    let status = match code {
        404 => "Not Found",
        409 => "Conflict",
        422 => "Unprocessable Content",
        _ => "Bad Request",
    };
    let body = serde_json::json!({"error": msg}).to_string();
    let resp = format!(
        "HTTP/1.1 {code} {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        body.len(), body
    );
    stream.write_all(resp.as_bytes()).await?;
    stream.shutdown().await?;
    Ok(())
}

pub async fn send_503(mut stream: TcpStream) -> std::io::Result<()> {
    let body = r#"{"error":"No inference server available — election in progress"}"#;
    let resp = format!(
        "HTTP/1.1 503 Service Unavailable\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        body.len(), body
    );
    stream.write_all(resp.as_bytes()).await?;
    stream.shutdown().await?;
    Ok(())
}

/// Pipeline-aware HTTP proxy for local targets.
///
/// Instead of TCP tunneling, this:
/// 1. Parses the HTTP request body
/// 2. Calls the planner model for a pre-plan
/// 3. Injects the plan into the request
/// 4. Forwards to the strong model via HTTP
/// 5. Streams the response back to the client
pub async fn pipeline_proxy_local(
    client_stream: &mut TcpStream,
    request_path: &str,
    mut body: serde_json::Value,
    planner_port: u16,
    planner_model: &str,
    strong_port: u16,
    node: &mesh::Node,
) -> PipelineProxyResult {
    if !pipeline_request_supported(request_path, &body) {
        tracing::debug!("pipeline: request path/body not eligible, falling back to direct proxy");
        return PipelineProxyResult::FallbackToDirect;
    }

    // Extract whether this is a streaming request
    let is_streaming = body
        .get("stream")
        .and_then(|s| s.as_bool())
        .unwrap_or(false);

    // Pre-plan: ask the small model
    let http_client = reqwest::Client::new();
    let planner_url = format!("http://127.0.0.1:{planner_port}");
    let messages = body
        .get("messages")
        .and_then(|m| m.as_array())
        .cloned()
        .unwrap_or_default();

    match crate::pipeline::pre_plan(&http_client, &planner_url, planner_model, &messages).await {
        Ok(plan) => {
            tracing::info!(
                "pipeline: pre-plan by {} in {}ms — {}",
                plan.model_used,
                plan.elapsed_ms,
                plan.plan_text.chars().take(200).collect::<String>()
            );
            crate::pipeline::inject_plan(&mut body, &plan);
        }
        Err(e) => {
            tracing::warn!("pipeline: pre-plan failed ({e}), falling back to direct proxy");
            return PipelineProxyResult::FallbackToDirect;
        }
    }

    // Forward to strong model — use reqwest for full HTTP handling
    let strong_url = format!("http://127.0.0.1:{strong_port}/v1/chat/completions");

    let _inflight = node.begin_inflight_request();

    if is_streaming {
        // Streaming: forward SSE chunks to client
        match http_client.post(&strong_url).json(&body).send().await {
            Ok(resp) => {
                let status = resp.status();
                let content_type = resp
                    .headers()
                    .get("content-type")
                    .and_then(|v| v.to_str().ok())
                    .unwrap_or("text/event-stream")
                    .to_string();

                // Send HTTP response headers
                let header = format!(
                    "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nTransfer-Encoding: chunked\r\nCache-Control: no-cache\r\n\r\n",
                );
                if client_stream.write_all(header.as_bytes()).await.is_err() {
                    return PipelineProxyResult::Handled;
                }

                // Stream body chunks
                use tokio_stream::StreamExt;
                let mut stream = resp.bytes_stream();
                while let Some(chunk) = stream.next().await {
                    match chunk {
                        Ok(bytes) => {
                            // HTTP chunked encoding
                            let chunk_header = format!("{:x}\r\n", bytes.len());
                            if client_stream
                                .write_all(chunk_header.as_bytes())
                                .await
                                .is_err()
                            {
                                break;
                            }
                            if client_stream.write_all(&bytes).await.is_err() {
                                break;
                            }
                            if client_stream.write_all(b"\r\n").await.is_err() {
                                break;
                            }
                        }
                        Err(e) => {
                            tracing::debug!("pipeline: stream error: {e}");
                            break;
                        }
                    }
                }
                // Terminal chunk
                let _ = client_stream.write_all(b"0\r\n\r\n").await;
                let _ = client_stream.shutdown().await;
                PipelineProxyResult::Handled
            }
            Err(e) => {
                tracing::warn!(
                    "pipeline: strong model request failed: {e}, falling back to direct proxy"
                );
                PipelineProxyResult::FallbackToDirect
            }
        }
    } else {
        // Non-streaming: simple request/response
        match http_client.post(&strong_url).json(&body).send().await {
            Ok(resp) => {
                let status = resp.status();
                match resp.bytes().await {
                    Ok(resp_bytes) => {
                        let header = format!(
                            "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
                            resp_bytes.len()
                        );
                        let _ = client_stream.write_all(header.as_bytes()).await;
                        let _ = client_stream.write_all(&resp_bytes).await;
                        let _ = client_stream.shutdown().await;
                        PipelineProxyResult::Handled
                    }
                    Err(e) => {
                        tracing::warn!(
                            "pipeline: response read failed: {e}, falling back to direct proxy"
                        );
                        PipelineProxyResult::FallbackToDirect
                    }
                }
            }
            Err(e) => {
                tracing::warn!(
                    "pipeline: strong model request failed: {e}, falling back to direct proxy"
                );
                PipelineProxyResult::FallbackToDirect
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::net::TcpListener;

    async fn read_request_from_parts_with_limits(
        parts: Vec<Vec<u8>>,
        limits: HttpReadLimits,
    ) -> BufferedHttpRequest {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            read_http_request_with_limits(&mut stream, limits)
                .await
                .unwrap()
        });

        let client = tokio::spawn(async move {
            let mut stream = TcpStream::connect(addr).await.unwrap();
            for part in parts {
                stream.write_all(&part).await.unwrap();
            }
        });

        client.await.unwrap();
        server.await.unwrap()
    }

    async fn read_request_from_parts(parts: Vec<Vec<u8>>) -> BufferedHttpRequest {
        read_request_from_parts_with_limits(parts, HTTP_READ_LIMITS).await
    }

    fn build_chunked_request(body: &[u8], chunks: &[usize]) -> Vec<u8> {
        let mut out = b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nTransfer-Encoding: chunked\r\n\r\n".to_vec();
        let mut pos = 0usize;
        for &chunk_len in chunks {
            let end = pos + chunk_len;
            out.extend_from_slice(format!("{chunk_len:x}\r\n").as_bytes());
            out.extend_from_slice(&body[pos..end]);
            out.extend_from_slice(b"\r\n");
            pos = end;
        }
        out.extend_from_slice(b"0\r\n\r\n");
        out
    }

    fn build_chunked_request_one_byte_chunks(body: &[u8], extension_len: usize) -> Vec<u8> {
        let mut out = b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nTransfer-Encoding: chunked\r\n\r\n".to_vec();
        let extension = "x".repeat(extension_len);
        for byte in body {
            out.extend_from_slice(b"1");
            if !extension.is_empty() {
                out.extend_from_slice(b";");
                out.extend_from_slice(extension.as_bytes());
            }
            out.extend_from_slice(b"\r\n");
            out.push(*byte);
            out.extend_from_slice(b"\r\n");
        }
        out.extend_from_slice(b"0\r\n\r\n");
        out
    }

    #[test]
    fn test_pipeline_request_supported_chat_completions() {
        let body = serde_json::json!({"messages":[{"role":"user","content":"hi"}]});
        assert!(pipeline_request_supported(
            "/v1/chat/completions?stream=1",
            &body
        ));
    }

    #[test]
    fn test_pipeline_request_supported_rejects_other_endpoint() {
        let body = serde_json::json!({"messages":[{"role":"user","content":"hi"}]});
        assert!(!pipeline_request_supported("/v1/responses", &body));
    }

    #[test]
    fn test_pipeline_request_supported_rejects_missing_messages() {
        let body = serde_json::json!({"input":"hi"});
        assert!(!pipeline_request_supported("/v1/chat/completions", &body));
    }

    #[test]
    fn test_request_budget_tokens_includes_output_budget_and_margin() {
        let body = serde_json::json!({
            "model": "qwen",
            "max_tokens": 512,
            "messages": [{"role": "user", "content": "hello world"}],
        });

        let budget = request_budget_tokens(&body).unwrap();
        assert!(budget >= 512 + REQUEST_TOKEN_MARGIN);
    }

    #[test]
    fn test_reorder_candidates_by_context_prefers_known_fit_then_unknown() {
        let ordered = reorder_candidates_by_context(
            &[(1u8, Some(4096)), (2u8, None), (3u8, Some(16384))],
            Some(8192),
        );

        assert_eq!(ordered, vec![3, 2]);
    }

    #[test]
    fn test_reorder_candidates_by_context_falls_back_when_all_known_too_small() {
        let ordered =
            reorder_candidates_by_context(&[(1u8, Some(4096)), (2u8, Some(6144))], Some(8192));

        assert_eq!(ordered, vec![1, 2]);
    }

    #[test]
    fn test_is_retryable_context_overflow_response_detects_llama_style_message() {
        let body = br#"{"error":{"message":"prompt tokens exceed context window (n_ctx=4096)"}}"#;
        assert!(is_retryable_context_overflow_response(body));
        assert!(!is_retryable_context_overflow_response(
            br#"{"error":{"message":"missing required field: messages"}}"#
        ));
    }

    #[tokio::test]
    async fn test_read_http_request_fragmented_post_body() {
        let body =
            br#"{"model":"qwen","user":"alice","messages":[{"role":"user","content":"hi"}]}"#;
        let headers = format!(
            "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
            body.len()
        );

        let request = read_request_from_parts(vec![
            headers.as_bytes()[..40].to_vec(),
            headers.as_bytes()[40..].to_vec(),
            body[..12].to_vec(),
            body[12..].to_vec(),
        ])
        .await;

        assert_eq!(request.method, "POST");
        assert_eq!(request.path, "/v1/chat/completions");
        assert_eq!(request.model_name.as_deref(), Some("qwen"));
        assert_eq!(request.session_hint.as_deref(), Some("alice"));
        assert_eq!(request.body_json.unwrap()["messages"][0]["content"], "hi");
    }

    #[tokio::test]
    async fn test_read_http_request_large_body_over_32k() {
        let large = "x".repeat(40_000);
        let body = serde_json::json!({
            "model": "qwen",
            "messages": [{"role": "user", "content": large}],
        })
        .to_string();
        let request = format!(
            "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
            body.len(),
            body
        );

        let request = read_request_from_parts(vec![request.into_bytes()]).await;

        assert_eq!(request.model_name.as_deref(), Some("qwen"));
        let body_json = request.body_json.unwrap();
        let content = body_json["messages"][0]["content"].as_str().unwrap();
        assert_eq!(content.len(), 40_000);
    }

    #[tokio::test]
    async fn test_read_http_request_chunked_body() {
        let body = br#"{"model":"auto","session_id":"sess-42","messages":[{"role":"user","content":"hello"}]}"#;
        let request = build_chunked_request(body, &[18, 17, body.len() - 35]);

        let request = read_request_from_parts(vec![request]).await;

        assert_eq!(request.model_name.as_deref(), Some("auto"));
        assert_eq!(request.session_hint.as_deref(), Some("sess-42"));
        assert_eq!(
            request.body_json.unwrap()["messages"][0]["content"],
            "hello"
        );
    }

    #[tokio::test]
    async fn test_read_http_request_chunked_body_allows_wire_overhead() {
        let limits = HttpReadLimits {
            max_header_bytes: MAX_HEADER_BYTES,
            max_body_bytes: 256,
            max_chunked_wire_bytes: 4 * 1024,
        };
        let large = "x".repeat(48);
        let body = serde_json::json!({
            "model": "qwen",
            "messages": [{"role": "user", "content": large}],
        })
        .to_string();
        let request = build_chunked_request_one_byte_chunks(body.as_bytes(), 16);

        let request = read_request_from_parts_with_limits(vec![request], limits).await;

        assert_eq!(request.model_name.as_deref(), Some("qwen"));
        assert!(request.raw.len() > limits.max_body_bytes);
        let body_json = request.body_json.unwrap();
        let content = body_json["messages"][0]["content"].as_str().unwrap();
        assert_eq!(content.len(), 48);
    }

    #[tokio::test]
    async fn test_read_http_request_expect_100_continue() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let body = br#"{"model":"qwen","user":"bob","messages":[]}"#.to_vec();
        let headers = format!(
            "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\nExpect: 100-continue\r\n\r\n",
            body.len()
        );

        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            read_http_request(&mut stream).await.unwrap()
        });

        let client = tokio::spawn(async move {
            let mut stream = TcpStream::connect(addr).await.unwrap();
            stream.write_all(headers.as_bytes()).await.unwrap();

            let mut interim = [0u8; 64];
            let n = stream.read(&mut interim).await.unwrap();
            assert_eq!(
                std::str::from_utf8(&interim[..n]).unwrap(),
                "HTTP/1.1 100 Continue\r\n\r\n"
            );

            stream.write_all(&body).await.unwrap();
        });

        client.await.unwrap();
        let request = server.await.unwrap();
        assert_eq!(request.model_name.as_deref(), Some("qwen"));
        assert_eq!(request.session_hint.as_deref(), Some("bob"));
        let raw = String::from_utf8(request.raw).unwrap();
        assert!(!raw.contains("Expect: 100-continue"));
        assert!(raw.contains("Connection: close"));
    }

    #[tokio::test]
    async fn test_read_http_request_truncates_pipelined_follow_up_bytes() {
        let request = read_request_from_parts(vec![
            b"GET /v1/models HTTP/1.1\r\nHost: localhost\r\n\r\nGET /mesh/drop HTTP/1.1\r\nHost: localhost\r\n\r\n"
                .to_vec(),
        ])
        .await;

        let raw = String::from_utf8(request.raw).unwrap();
        assert!(raw.starts_with("GET /v1/models HTTP/1.1\r\n"));
        assert!(!raw.contains("/mesh/drop"));
        assert!(raw.contains("Connection: close\r\n\r\n"));
    }
}
