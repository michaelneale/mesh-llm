//! HTTP proxy plumbing — request parsing, model routing, response helpers.
//!
//! Used by the API proxy (port 9337), bootstrap proxy, and passive mode.
//! All inference traffic flows through these functions.

use crate::inference::election;
use crate::mesh;
use crate::network::affinity::{
    prepare_remote_targets_for_request, AffinityRouter, PreparedTargets, TargetSelection,
};
use crate::network::openai::response_adapter;
use crate::network::router;
use crate::network::target_health::TargetHealthOutcome;
use crate::plugin;
use anyhow::{anyhow, bail, Context, Result};
// moa imports relocated into moa_gateway.rs (sole user after merge)
use serde::Deserialize;
use serde_json::{Map, Value};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use url::Url;

const MAX_HEADER_BYTES: usize = 64 * 1024;
const MAX_BODY_BYTES: usize = 8 * 1024 * 1024;
const MAX_OBJECT_UPLOAD_BODY_BYTES: usize = 64 * 1024 * 1024;
const MAX_CHUNKED_WIRE_BYTES: usize = MAX_BODY_BYTES * 6 + 64 * 1024;
const MAX_OBJECT_UPLOAD_CHUNKED_WIRE_BYTES: usize = MAX_OBJECT_UPLOAD_BODY_BYTES * 6 + 64 * 1024;
const MAX_HEADERS: usize = 64;
const MAX_RESPONSE_BODY_PREVIEW_BYTES: usize = 4 * 1024;
const MAX_ERROR_RESPONSE_BYTES: usize = 256 * 1024;
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
    body_json_attempted: bool,
    body_bytes: Option<Vec<u8>>,
    pub body_len_bytes: usize,
    pub completion_tokens: Option<u32>,
    pub model_name: Option<String>,
    pub request_object_request_ids: Vec<String>,
    pub response_adapter: ResponseAdapter,
}

impl BufferedHttpRequest {
    pub fn ensure_body_json(&mut self) {
        if self.body_json.is_none() && !self.body_json_attempted {
            self.body_json = self
                .body_bytes
                .as_deref()
                .and_then(|body| serde_json::from_slice(body).ok())
                .or_else(|| parse_json_body_from_http_request(&self.raw));
            self.body_json_attempted = true;
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResponseAdapter {
    None,
    OpenAiChatCompletionsStream,
    OpenAiResponsesJson,
    OpenAiResponsesStream,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineProxyResult {
    Handled,
    FallbackToDirect,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RouteAttemptResult {
    Delivered {
        status_code: u16,
        completion_tokens: Option<u64>,
    },
    RetryableTimeout,
    RetryableUnavailable,
    RetryableContextOverflow,
    ClientDisconnected,
}

fn route_attempt_result_label(result: &RouteAttemptResult) -> &'static str {
    match result {
        RouteAttemptResult::Delivered { .. } => "delivered",
        RouteAttemptResult::RetryableTimeout => "retryable_timeout",
        RouteAttemptResult::RetryableUnavailable => "retryable_unavailable",
        RouteAttemptResult::RetryableContextOverflow => "retryable_context_overflow",
        RouteAttemptResult::ClientDisconnected => "client_disconnected",
    }
}

fn target_health_outcome_for_attempt(result: &RouteAttemptResult) -> TargetHealthOutcome {
    match result {
        RouteAttemptResult::Delivered { status_code, .. } if (200..300).contains(status_code) => {
            TargetHealthOutcome::Success
        }
        RouteAttemptResult::Delivered { status_code, .. } if (500..600).contains(status_code) => {
            TargetHealthOutcome::Unavailable
        }
        RouteAttemptResult::Delivered { .. } => TargetHealthOutcome::Rejected,
        RouteAttemptResult::RetryableTimeout => TargetHealthOutcome::Timeout,
        RouteAttemptResult::RetryableUnavailable => TargetHealthOutcome::Unavailable,
        RouteAttemptResult::RetryableContextOverflow => TargetHealthOutcome::ContextOverflow,
        RouteAttemptResult::ClientDisconnected => TargetHealthOutcome::ClientDisconnected,
    }
}

fn is_disconnect_kind(kind: std::io::ErrorKind) -> bool {
    matches!(
        kind,
        std::io::ErrorKind::BrokenPipe
            | std::io::ErrorKind::ConnectionReset
            | std::io::ErrorKind::ConnectionAborted
            | std::io::ErrorKind::NotConnected
            | std::io::ErrorKind::UnexpectedEof
    )
}

fn is_client_disconnect_error(err: &anyhow::Error) -> bool {
    err.chain().any(|cause| {
        cause
            .downcast_ref::<std::io::Error>()
            .map(|io_err| is_disconnect_kind(io_err.kind()))
            .unwrap_or(false)
    })
}

fn is_timeout_error(err: &anyhow::Error) -> bool {
    err.chain().any(|cause| {
        cause
            .downcast_ref::<std::io::Error>()
            .map(|io_err| io_err.kind() == std::io::ErrorKind::TimedOut)
            .unwrap_or(false)
            || cause.is::<tokio::time::error::Elapsed>()
    })
}

struct ParsedResponseHeaders {
    header_end: usize,
    status_code: u16,
    content_length: Option<usize>,
    content_type: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct RequestMetadata {
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    stream: Option<bool>,
    #[serde(default)]
    max_completion_tokens: Option<u32>,
    #[serde(default)]
    max_tokens: Option<u32>,
    #[serde(default)]
    max_output_tokens: Option<u32>,
    #[serde(default)]
    n_predict: Option<u32>,
}

#[derive(Clone)]
struct ResponseProbe {
    buffered: Vec<u8>,
    header_end: usize,
    status_code: u16,
    retryable_context_overflow: bool,
}

#[derive(Debug)]
struct RequestNormalization {
    changed: bool,
    rewritten_path: Option<String>,
    response_adapter: ResponseAdapter,
}

struct RequestRewriteOutcome {
    body_json: Option<serde_json::Value>,
    request_object_request_ids: Vec<String>,
    request_path: String,
    response_adapter: ResponseAdapter,
    rewritten_body: Option<Vec<u8>>,
}

struct ExternalEndpointTarget {
    host: String,
    port: u16,
    forwarded: Vec<u8>,
}

struct ResponsesStreamRelayState {
    created_at: i64,
    response_id: String,
    item_id: String,
    model: String,
    output_text: String,
    usage: Option<serde_json::Value>,
    observed_completion_tokens: Option<u64>,
    sequence_number: i32,
    created_emitted: bool,
    output_item_emitted: bool,
}

impl ResponsesStreamRelayState {
    fn new() -> Self {
        let created_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|duration| duration.as_secs() as i64)
            .unwrap_or(0);
        Self {
            created_at,
            response_id: format!("resp_{created_at}"),
            item_id: format!("msg_{created_at}"),
            model: String::new(),
            output_text: String::new(),
            usage: None,
            observed_completion_tokens: None,
            sequence_number: 0,
            created_emitted: false,
            output_item_emitted: false,
        }
    }

    fn next_sequence_number(&mut self) -> i32 {
        self.sequence_number = self.sequence_number.saturating_add(1);
        self.sequence_number
    }
}

// ── Request parsing ──

/// Read and buffer one HTTP request for routing decisions.
///
/// This reads complete headers plus the full request body when body framing is
/// known via `Content-Length` or `Transfer-Encoding: chunked`. The raw request
/// bytes are preserved so the chosen upstream sees the original payload.
pub async fn read_http_request(stream: &mut TcpStream) -> Result<BufferedHttpRequest> {
    read_http_request_with_limits(stream, HTTP_READ_LIMITS, None).await
}

pub async fn read_http_request_with_plugin_manager(
    stream: &mut TcpStream,
    plugin_manager: Option<&plugin::PluginManager>,
) -> Result<BufferedHttpRequest> {
    read_http_request_with_limits(stream, HTTP_READ_LIMITS, plugin_manager).await
}

async fn read_http_request_with_limits(
    stream: &mut TcpStream,
    limits: HttpReadLimits,
    plugin_manager: Option<&plugin::PluginManager>,
) -> Result<BufferedHttpRequest> {
    let mut raw = Vec::with_capacity(8192);
    let parsed = read_until_headers_parsed(stream, &mut raw, limits.max_header_bytes).await?;
    let body_limits = body_limits_for_path(&parsed.path, limits);
    let header_end = parsed.header_end;
    let body =
        read_buffered_request_body(stream, &mut raw, &parsed, header_end, body_limits).await?;

    let metadata = if body.is_empty() {
        None
    } else {
        serde_json::from_slice::<RequestMetadata>(&body).ok()
    };
    let requires_json_transform =
        request_requires_json_transform(&parsed.path, &body, plugin_manager.is_some());
    let rewrite = rewrite_request_body_for_forwarding(
        &parsed.path,
        &body,
        plugin_manager,
        requires_json_transform,
    )
    .await?;
    let mut response_adapter = rewrite.response_adapter;
    if response_adapter == ResponseAdapter::None
        && parsed.path.split('?').next().unwrap_or(&parsed.path) == "/v1/chat/completions"
        && metadata.as_ref().and_then(|value| value.stream) == Some(true)
    {
        response_adapter = ResponseAdapter::OpenAiChatCompletionsStream;
    }
    let model_name = metadata.as_ref().and_then(|value| value.model.clone());
    let completion_tokens = metadata.as_ref().and_then(|value| {
        value
            .max_completion_tokens
            .or(value.max_tokens)
            .or(value.max_output_tokens)
            .or(value.n_predict)
    });
    let raw = finalize_forwarded_request(
        raw,
        header_end,
        parsed.expects_continue,
        Some(&rewrite.request_path),
        rewrite.rewritten_body.as_deref(),
    )?;
    let body_len_bytes = body.len();
    let body_bytes = if body.is_empty() { None } else { Some(body) };

    Ok(BufferedHttpRequest {
        raw,
        method: parsed.method,
        path: rewrite.request_path,
        body_json: rewrite.body_json,
        body_json_attempted: requires_json_transform,
        body_bytes,
        body_len_bytes,
        completion_tokens,
        model_name,
        request_object_request_ids: rewrite.request_object_request_ids,
        response_adapter,
    })
}

async fn read_buffered_request_body(
    stream: &mut TcpStream,
    raw: &mut Vec<u8>,
    parsed: &ParsedHeaders,
    header_end: usize,
    body_limits: HttpReadLimits,
) -> Result<Vec<u8>> {
    if parsed.is_chunked {
        return read_chunked_request_body(stream, raw, parsed, header_end, body_limits).await;
    }
    if let Some(content_length) = parsed.content_length {
        return read_fixed_length_request_body(
            stream,
            raw,
            parsed,
            header_end,
            content_length,
            body_limits,
        )
        .await;
    }
    raw.truncate(header_end);
    Ok(Vec::new())
}

async fn read_chunked_request_body(
    stream: &mut TcpStream,
    raw: &mut Vec<u8>,
    parsed: &ParsedHeaders,
    header_end: usize,
    body_limits: HttpReadLimits,
) -> Result<Vec<u8>> {
    let mut sent_continue = false;
    loop {
        if let Some((consumed, decoded)) =
            try_decode_chunked_body(&raw[header_end..], body_limits.max_body_bytes)?
        {
            raw.truncate(header_end + consumed);
            return Ok(decoded);
        }
        if !sent_continue && parsed.expects_continue {
            stream.write_all(b"HTTP/1.1 100 Continue\r\n\r\n").await?;
            sent_continue = true;
        }
        read_more(stream, raw).await?;
        if raw.len().saturating_sub(header_end) > body_limits.max_chunked_wire_bytes {
            bail!(
                "HTTP chunked wire body exceeds {} bytes",
                body_limits.max_chunked_wire_bytes
            );
        }
    }
}

async fn read_fixed_length_request_body(
    stream: &mut TcpStream,
    raw: &mut Vec<u8>,
    parsed: &ParsedHeaders,
    header_end: usize,
    content_length: usize,
    body_limits: HttpReadLimits,
) -> Result<Vec<u8>> {
    if content_length > body_limits.max_body_bytes {
        bail!("HTTP body exceeds {} bytes", body_limits.max_body_bytes);
    }
    let body_end = header_end + content_length;
    let mut sent_continue = false;
    while raw.len() < body_end {
        if !sent_continue && parsed.expects_continue && content_length > 0 {
            stream.write_all(b"HTTP/1.1 100 Continue\r\n\r\n").await?;
            sent_continue = true;
        }
        read_more(stream, raw).await?;
    }
    raw.truncate(body_end);
    Ok(raw[header_end..body_end].to_vec())
}

async fn rewrite_request_body_for_forwarding(
    path: &str,
    body: &[u8],
    plugin_manager: Option<&plugin::PluginManager>,
    requires_json_transform: bool,
) -> Result<RequestRewriteOutcome> {
    let mut outcome = RequestRewriteOutcome {
        body_json: None,
        request_object_request_ids: Vec::new(),
        request_path: path.to_string(),
        response_adapter: ResponseAdapter::None,
        rewritten_body: None,
    };
    if !requires_json_transform {
        return Ok(outcome);
    }

    outcome.body_json = serde_json::from_slice(body).ok();
    let Some(body_json) = outcome.body_json.as_mut() else {
        return Ok(outcome);
    };

    let normalization = normalize_openai_compat_request(path, body_json)?;
    let mut changed = normalization.changed;
    if let Some(rewritten_path) = normalization.rewritten_path {
        outcome.request_path = rewritten_path;
    }
    outcome.response_adapter = normalization.response_adapter;
    if let Some(plugin_manager) = plugin_manager {
        let resolved_request_ids =
            resolve_request_object_references(&outcome.request_path, body_json, plugin_manager)
                .await?;
        if !resolved_request_ids.is_empty() {
            outcome.request_object_request_ids = resolved_request_ids;
            changed = true;
        }
    }
    if changed {
        outcome.rewritten_body = Some(
            serde_json::to_vec(body_json)
                .context("serialize normalized OpenAI-compatible request body")?,
        );
    }
    Ok(outcome)
}

fn body_limits_for_path(path: &str, default: HttpReadLimits) -> HttpReadLimits {
    let path_only = path.split('?').next().unwrap_or(path);
    if path_only == "/api/objects" {
        HttpReadLimits {
            max_header_bytes: default.max_header_bytes,
            max_body_bytes: MAX_OBJECT_UPLOAD_BODY_BYTES,
            max_chunked_wire_bytes: MAX_OBJECT_UPLOAD_CHUNKED_WIRE_BYTES,
        }
    } else {
        default
    }
}

fn finalize_forwarded_request(
    mut raw: Vec<u8>,
    header_end: usize,
    strip_expect: bool,
    rewritten_path: Option<&str>,
    rewritten_body: Option<&[u8]>,
) -> Result<Vec<u8>> {
    let original_body = raw.split_off(header_end);
    // Re-parse with httparse so we iterate over validated header structs.
    let mut headers_buf = [httparse::EMPTY_HEADER; MAX_HEADERS];
    let mut req = httparse::Request::new(&mut headers_buf);
    let _ = req.parse(&raw).context("re-parse headers for forwarding")?;

    let method = req.method.unwrap_or("GET");
    let path = rewritten_path.unwrap_or_else(|| req.path.unwrap_or("/"));
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
        if rewritten_body.is_some()
            && (name.eq_ignore_ascii_case("content-length")
                || name.eq_ignore_ascii_case("transfer-encoding"))
        {
            continue;
        }
        let value = std::str::from_utf8(header.value).unwrap_or("");
        rebuilt.push_str(&format!("{name}: {value}\r\n"));
    }
    if let Some(body) = rewritten_body {
        rebuilt.push_str(&format!("Content-Length: {}\r\n", body.len()));
    }

    // The proxy buffers exactly one request for routing, so force a single-request
    // connection contract upstream instead of reusing the client connection blindly.
    rebuilt.push_str("Connection: close\r\n\r\n");

    let mut forwarded = rebuilt.into_bytes();
    forwarded.extend_from_slice(rewritten_body.unwrap_or(&original_body));
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

fn request_requires_json_transform(path: &str, body: &[u8], plugin_manager_present: bool) -> bool {
    let path_only = path.split('?').next().unwrap_or(path);
    if body.is_empty() {
        return false;
    }
    if path_only == "/v1/responses" {
        return true;
    }
    if path_only != "/v1/chat/completions" {
        return false;
    }

    let body_text = match std::str::from_utf8(body) {
        Ok(text) => text,
        Err(_) => return false,
    };

    body_text.contains("\"max_completion_tokens\"")
        || body_text.contains("\"max_output_tokens\"")
        || (plugin_manager_present
            && (body_text.contains("mesh://blob/")
                || body_text.contains("\"blob_token\"")
                || body_text.contains("\"mesh_token\"")
                || body_text.contains("\"input_audio\"")
                || body_text.contains("\"input_image\"")))
}

fn parse_json_body_from_http_request(raw: &[u8]) -> Option<serde_json::Value> {
    let header_end = raw.windows(4).position(|window| window == b"\r\n\r\n")? + 4;
    serde_json::from_slice(&raw[header_end..]).ok()
}

fn normalize_openai_compat_request(
    path: &str,
    body: &mut serde_json::Value,
) -> Result<RequestNormalization> {
    let normalized = openai_frontend::normalize_openai_compat_request(path, body)?;
    let response_adapter = match normalized.response_adapter {
        openai_frontend::ResponseAdapterMode::None => ResponseAdapter::None,
        openai_frontend::ResponseAdapterMode::OpenAiResponsesJson => {
            ResponseAdapter::OpenAiResponsesJson
        }
        openai_frontend::ResponseAdapterMode::OpenAiResponsesStream => {
            ResponseAdapter::OpenAiResponsesStream
        }
    };
    Ok(RequestNormalization {
        changed: normalized.changed,
        rewritten_path: normalized.rewritten_path,
        response_adapter,
    })
}

fn request_id_from_body(body: &serde_json::Value) -> Option<String> {
    body.get("request_id")
        .and_then(|value| value.as_str())
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn mesh_blob_token_from_url(url: &str) -> Option<String> {
    let path = url.strip_prefix("mesh://blob/")?;
    let mut parts = path.split('/').filter(|part| !part.trim().is_empty());
    let _client_id = parts.next()?;
    let token = parts.next()?;
    if parts.next().is_some() {
        return None;
    }
    Some(token.to_string())
}

fn blob_token_from_container(container: &serde_json::Value) -> Option<String> {
    container
        .get("url")
        .and_then(|value| value.as_str())
        .and_then(mesh_blob_token_from_url)
        .or_else(|| {
            ["mesh_token", "blob_token", "token"]
                .into_iter()
                .find_map(|key| {
                    container
                        .get(key)
                        .and_then(|value| value.as_str())
                        .map(str::trim)
                        .filter(|value| !value.is_empty())
                        .map(ToString::to_string)
                })
        })
}

fn data_url(mime_type: &str, bytes_base64: &str) -> String {
    format!("data:{mime_type};base64,{bytes_base64}")
}

fn audio_format_from_mime_type(mime_type: &str) -> Option<&'static str> {
    match mime_type {
        "audio/wav" | "audio/x-wav" => Some("wav"),
        "audio/mpeg" | "audio/mp3" => Some("mp3"),
        "audio/flac" => Some("flac"),
        "audio/ogg" | "audio/opus" => Some("ogg"),
        "audio/webm" => Some("webm"),
        _ => None,
    }
}

enum MediaRefAction {
    DataUrlContainer { container_key: &'static str },
    InputAudio,
}

fn block_media_ref_action(block: &serde_json::Value) -> Option<(MediaRefAction, String)> {
    for key in [
        "image_url",
        "audio_url",
        "image",
        "audio",
        "input_image",
        "file",
        "input_file",
    ] {
        let Some(container) = block.get(key) else {
            continue;
        };
        let Some(token) = blob_token_from_container(container) else {
            continue;
        };
        return Some((
            MediaRefAction::DataUrlContainer { container_key: key },
            token,
        ));
    }

    let input_audio = block.get("input_audio")?;
    let token = blob_token_from_container(input_audio)?;
    Some((MediaRefAction::InputAudio, token))
}

async fn resolve_request_object_references(
    path: &str,
    body: &mut serde_json::Value,
    plugin_manager: &plugin::PluginManager,
) -> Result<Vec<String>> {
    let path_only = path.split('?').next().unwrap_or(path);
    if path_only != "/v1/chat/completions" {
        return Ok(Vec::new());
    }
    let request_id = request_id_from_body(body);
    let Some(messages) = body
        .get_mut("messages")
        .and_then(|value| value.as_array_mut())
    else {
        return Ok(Vec::new());
    };

    let mut request_ids = Vec::new();
    let mut blob_cache: HashMap<String, crate::plugins::blobstore::GetRequestObjectResponse> =
        HashMap::new();
    for message in messages.iter_mut() {
        let Some(blocks) = message
            .get_mut("content")
            .and_then(|value| value.as_array_mut())
        else {
            continue;
        };
        for block in blocks.iter_mut() {
            let Some((action, token)) = block_media_ref_action(block) else {
                continue;
            };
            let blob = if let Some(cached) = blob_cache.get(&token) {
                cached.clone()
            } else {
                let fetched = crate::plugins::blobstore::get_request_object(
                    plugin_manager,
                    crate::plugins::blobstore::GetRequestObjectRequest {
                        token: token.clone(),
                        request_id: request_id.clone(),
                    },
                )
                .await?;
                blob_cache.insert(token.clone(), fetched.clone());
                fetched
            };
            if !request_ids
                .iter()
                .any(|existing| existing == &blob.request_id)
            {
                request_ids.push(blob.request_id.clone());
            }
            match action {
                MediaRefAction::DataUrlContainer { container_key } => {
                    if let Some(container) = block
                        .get_mut(container_key)
                        .and_then(|value| value.as_object_mut())
                    {
                        container.insert(
                            "url".into(),
                            serde_json::Value::String(data_url(
                                &blob.mime_type,
                                &blob.bytes_base64,
                            )),
                        );
                        container.remove("mesh_token");
                        container.remove("blob_token");
                        container.remove("token");
                    }
                }
                MediaRefAction::InputAudio => {
                    if let Some(container) = block
                        .get_mut("input_audio")
                        .and_then(|value| value.as_object_mut())
                    {
                        container.insert(
                            "data".into(),
                            serde_json::Value::String(blob.bytes_base64.clone()),
                        );
                        if let Some(format) = audio_format_from_mime_type(&blob.mime_type) {
                            container
                                .entry("format")
                                .or_insert_with(|| serde_json::Value::String(format.to_string()));
                        }
                        container.insert(
                            "mime_type".into(),
                            serde_json::Value::String(blob.mime_type.clone()),
                        );
                        container.remove("url");
                        container.remove("mesh_token");
                        container.remove("blob_token");
                        container.remove("token");
                    }
                }
            }
        }
    }

    Ok(request_ids)
}

pub async fn release_request_objects(node: &mesh::Node, request_ids: &[String]) {
    if request_ids.is_empty() {
        return;
    }
    let Some(plugin_manager) = node.plugin_manager().await else {
        return;
    };
    for request_id in request_ids {
        if let Err(err) = crate::plugins::blobstore::complete_request(
            &plugin_manager,
            crate::plugins::blobstore::FinishRequestRequest {
                request_id: request_id.clone(),
            },
        )
        .await
        {
            tracing::warn!(
                request_id,
                error = %err,
                "blobstore: failed to release request-scoped objects"
            );
        }
    }
}

/// Remote first-byte timeout: 5 minutes. This covers the full round trip
/// through the QUIC tunnel including remote prefill. Concurrent requests
/// on a loaded host can legitimately take minutes. A truly dead QUIC
/// connection will reset/error much faster than this (QUIC idle timeout,
/// connection loss detection). The old 60s default caused spurious 503s
/// when the remote host was alive but busy.
fn response_first_byte_timeout() -> Duration {
    Duration::from_secs(5 * 60)
}

fn saturating_u32(value: usize) -> u32 {
    value.try_into().unwrap_or(u32::MAX)
}

fn ceil_div_u32(value: u32, divisor: u32) -> u32 {
    value.saturating_add(divisor - 1) / divisor
}

#[cfg(test)]
fn request_budget_tokens(body: &serde_json::Value) -> Option<u32> {
    let serialized = serde_json::to_vec(body).ok()?;
    let completion_tokens = [
        "max_completion_tokens",
        "max_tokens",
        "max_output_tokens",
        "n_predict",
    ]
    .into_iter()
    .find_map(|key| body.get(key).and_then(|value| value.as_u64()))
    .map(|value| value.min(u32::MAX as u64) as u32);
    request_budget_tokens_from_parts(serialized.len(), completion_tokens)
}

pub(crate) fn request_budget_tokens_from_parts(
    body_len_bytes: usize,
    completion_tokens: Option<u32>,
) -> Option<u32> {
    if body_len_bytes == 0 {
        return None;
    }
    let prompt_tokens = ceil_div_u32(saturating_u32(body_len_bytes), 4);
    Some(
        prompt_tokens
            .saturating_add(completion_tokens.unwrap_or(0))
            .saturating_add(REQUEST_TOKEN_MARGIN),
    )
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct TargetThroughputRank {
    avg_tokens_per_second_milli: u64,
    throughput_samples: u64,
    local_observation: bool,
}

#[derive(Clone)]
struct RankedTarget<T> {
    index: usize,
    candidate: T,
    context_length: Option<u32>,
    throughput: Option<TargetThroughputRank>,
}

const LOCAL_THROUGHPUT_PRECEDENCE_SAMPLES: u64 = 3;
const TARGET_THROUGHPUT_MAX_SCORE_SAMPLES: u64 = 32;

fn target_throughput_rank_key(throughput: Option<TargetThroughputRank>) -> (bool, bool, u64, u64) {
    let Some(throughput) = throughput else {
        return (false, false, 0, 0);
    };
    if throughput.avg_tokens_per_second_milli == 0 || throughput.throughput_samples == 0 {
        return (false, false, 0, 0);
    }
    let sample_weight = throughput
        .throughput_samples
        .min(TARGET_THROUGHPUT_MAX_SCORE_SAMPLES);
    (
        true,
        throughput.local_observation,
        throughput.avg_tokens_per_second_milli,
        sample_weight,
    )
}

fn sort_ranked_targets<T>(targets: &mut [RankedTarget<T>]) {
    targets.sort_by(|a, b| {
        target_throughput_rank_key(b.throughput)
            .cmp(&target_throughput_rank_key(a.throughput))
            .then_with(|| a.index.cmp(&b.index))
    });
}

fn reorder_candidates_by_context_and_throughput<T: Clone>(
    candidates: &[(T, Option<u32>, Option<TargetThroughputRank>)],
    required_tokens: Option<u32>,
) -> Vec<T> {
    let ranked = candidates
        .iter()
        .enumerate()
        .map(
            |(index, (candidate, context_length, throughput))| RankedTarget {
                index,
                candidate: candidate.clone(),
                context_length: *context_length,
                throughput: *throughput,
            },
        )
        .collect::<Vec<_>>();

    let Some(required_tokens) = required_tokens else {
        let mut ranked = ranked;
        sort_ranked_targets(&mut ranked);
        return ranked.into_iter().map(|ranked| ranked.candidate).collect();
    };

    let mut adequate = Vec::new();
    let mut unknown = Vec::new();
    for ranked in ranked {
        match ranked.context_length {
            Some(value) if value >= required_tokens => adequate.push(ranked),
            Some(_) => {}
            None => unknown.push(ranked),
        }
    }

    if adequate.is_empty() && unknown.is_empty() {
        let mut fallback = candidates
            .iter()
            .enumerate()
            .map(
                |(index, (candidate, context_length, throughput))| RankedTarget {
                    index,
                    candidate: candidate.clone(),
                    context_length: *context_length,
                    throughput: *throughput,
                },
            )
            .collect::<Vec<_>>();
        sort_ranked_targets(&mut fallback);
        return fallback
            .into_iter()
            .map(|ranked| ranked.candidate)
            .collect();
    }

    sort_ranked_targets(&mut adequate);
    sort_ranked_targets(&mut unknown);
    adequate
        .into_iter()
        .chain(unknown)
        .map(|ranked| ranked.candidate)
        .collect()
}

fn local_target_throughput_rank(
    node: &mesh::Node,
    model: &str,
    target: &election::InferenceTarget,
) -> Option<TargetThroughputRank> {
    let attempt_target = match target {
        election::InferenceTarget::Local(port) => {
            crate::network::metrics::AttemptTarget::Local(format!("127.0.0.1:{port}"))
        }
        election::InferenceTarget::Remote(peer_id) => {
            crate::network::metrics::AttemptTarget::Remote(peer_id.fmt_short().to_string())
        }
        election::InferenceTarget::None => return None,
    };
    node.routing_metrics()
        .throughput_hint_for_target(model, attempt_target)
        .map(|hint| TargetThroughputRank {
            avg_tokens_per_second_milli: hint.avg_tokens_per_second_milli,
            throughput_samples: hint.throughput_samples,
            local_observation: true,
        })
}

async fn remote_target_throughput_rank(
    node: &mesh::Node,
    model: &str,
    peer_id: iroh::EndpointId,
) -> Option<TargetThroughputRank> {
    let target = election::InferenceTarget::Remote(peer_id);
    let local = local_target_throughput_rank(node, model, &target);
    if local
        .map(|hint| hint.throughput_samples >= LOCAL_THROUGHPUT_PRECEDENCE_SAMPLES)
        .unwrap_or(false)
    {
        return local;
    }

    let gossiped = node
        .peer_model_throughput_hint(peer_id, model)
        .await
        .map(|hint| TargetThroughputRank {
            avg_tokens_per_second_milli: hint.avg_tokens_per_second_milli,
            throughput_samples: hint.throughput_samples,
            local_observation: false,
        });
    gossiped.or(local)
}

async fn order_remote_hosts_by_context(
    node: &mesh::Node,
    model: &str,
    required_tokens: Option<u32>,
    hosts: &[iroh::EndpointId],
) -> Vec<iroh::EndpointId> {
    let mut candidates = Vec::with_capacity(hosts.len());
    for host in hosts {
        candidates.push((
            *host,
            node.peer_model_context_length(*host, model).await,
            remote_target_throughput_rank(node, model, *host).await,
        ));
    }
    reorder_candidates_by_context_and_throughput(&candidates, required_tokens)
}

async fn order_targets_by_context(
    node: &mesh::Node,
    model: &str,
    required_tokens: Option<u32>,
    targets: &[election::InferenceTarget],
) -> Vec<election::InferenceTarget> {
    let mut candidates = Vec::with_capacity(targets.len());
    for target in targets {
        let context_length = match target {
            election::InferenceTarget::Local(_) => node.local_model_context_length(model).await,
            election::InferenceTarget::Remote(peer_id) => {
                node.peer_model_context_length(*peer_id, model).await
            }
            election::InferenceTarget::None => None,
        };
        let throughput = match target {
            election::InferenceTarget::Remote(peer_id) => {
                remote_target_throughput_rank(node, model, *peer_id).await
            }
            _ => local_target_throughput_rank(node, model, target),
        };
        candidates.push((target.clone(), context_length, throughput));
    }
    reorder_candidates_by_context_and_throughput(&candidates, required_tokens)
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

fn parse_completion_tokens_from_json_body(body: &[u8]) -> Option<u64> {
    let json = serde_json::from_slice::<serde_json::Value>(body).ok()?;
    let usage = json.get("usage")?;
    usage
        .get("completion_tokens")
        .or_else(|| usage.get("output_tokens"))
        .and_then(|value| value.as_u64())
}

fn response_is_event_stream(headers: &ParsedResponseHeaders) -> bool {
    headers
        .content_type
        .as_deref()
        .map(|value| {
            value
                .split(';')
                .next()
                .unwrap_or(value)
                .trim()
                .eq_ignore_ascii_case("text/event-stream")
        })
        .unwrap_or(false)
}

fn synthetic_id_seed() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or(0)
}

#[derive(Debug, Default)]
struct ChatStreamNormalizationState {
    completion_id: Option<String>,
    tool_call_ids: HashMap<u64, String>,
    synthetic_seed: Option<u128>,
}

impl ChatStreamNormalizationState {
    fn seed(&mut self) -> u128 {
        *self.synthetic_seed.get_or_insert_with(synthetic_id_seed)
    }

    fn completion_id(&mut self, object: &Map<String, Value>) -> String {
        if let Some(existing) = self.completion_id.clone() {
            return existing;
        }
        let id = object
            .get("id")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(ToString::to_string)
            .unwrap_or_else(|| format!("chatcmpl-mesh-{}", self.seed()));
        self.completion_id = Some(id.clone());
        id
    }

    fn tool_call_id(&mut self, index: u64, tool_call: &Map<String, Value>) -> String {
        if let Some(existing) = self.tool_call_ids.get(&index) {
            return existing.clone();
        }
        let id = tool_call
            .get("id")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(ToString::to_string)
            .unwrap_or_else(|| format!("call_mesh_{}_{}", self.seed(), index));
        self.tool_call_ids.insert(index, id.clone());
        id
    }

    fn normalize_data(&mut self, data: &str) -> String {
        let Ok(mut value) = serde_json::from_str::<Value>(data) else {
            return data.to_string();
        };
        let Some(object) = value.as_object_mut() else {
            return data.to_string();
        };

        let completion_id = self.completion_id(object);
        object.insert("id".into(), Value::String(completion_id));

        let Some(choices) = object.get_mut("choices").and_then(Value::as_array_mut) else {
            return serde_json::to_string(&value).unwrap_or_else(|_| data.to_string());
        };
        for choice in choices {
            let Some(tool_calls) = choice
                .get_mut("delta")
                .and_then(Value::as_object_mut)
                .and_then(|delta| delta.get_mut("tool_calls"))
                .and_then(Value::as_array_mut)
            else {
                continue;
            };
            for (position, tool_call) in tool_calls.iter_mut().enumerate() {
                let Some(tool_call_object) = tool_call.as_object_mut() else {
                    continue;
                };
                let index = tool_call_object
                    .get("index")
                    .and_then(Value::as_u64)
                    .unwrap_or(position as u64);
                let id = self.tool_call_id(index, tool_call_object);
                tool_call_object.insert("id".into(), Value::String(id));
            }
        }

        serde_json::to_string(&value).unwrap_or_else(|_| data.to_string())
    }
}

async fn relay_normalized_chat_completion_stream<R: AsyncRead + Unpin>(
    tcp_stream: &mut TcpStream,
    reader: &mut R,
    probe: ResponseProbe,
    retry_context_overflow: bool,
) -> Result<RouteAttemptResult> {
    if retry_context_overflow && probe.retryable_context_overflow {
        return Ok(RouteAttemptResult::RetryableContextOverflow);
    }

    if !(200..300).contains(&probe.status_code) {
        return relay_error_response(tcp_stream, reader, probe).await;
    }

    let parsed = try_parse_response_headers(&probe.buffered)?
        .ok_or_else(|| anyhow!("incomplete HTTP response"))?;
    if !response_is_event_stream(&parsed) {
        return relay_success_response(tcp_stream, reader, probe, parsed).await;
    }

    let mut carry = String::from_utf8_lossy(&probe.buffered[parsed.header_end..]).to_string();
    let mut state = ChatStreamNormalizationState::default();
    let mut observed_completion_tokens = None;
    let header = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n";
    tcp_stream.write_all(header.as_bytes()).await?;

    let mut done_seen = false;
    loop {
        let mut processed = 0usize;
        while let Some(frame_end_rel) = carry[processed..].find("\n\n") {
            let frame_end = processed + frame_end_rel;
            let frame = &carry[processed..frame_end];
            processed = frame_end + 2;
            let data_lines = frame
                .lines()
                .filter_map(|line| line.strip_prefix("data:"))
                .map(str::trim_start)
                .collect::<Vec<_>>();
            if data_lines.is_empty() {
                continue;
            }
            let data = data_lines.join("\n");
            if data == "[DONE]" {
                done_seen = true;
                response_adapter::write_chunked_sse_event(tcp_stream, None, "[DONE]").await?;
                break;
            }

            if observed_completion_tokens.is_none() {
                observed_completion_tokens =
                    parse_completion_tokens_from_json_body(data.as_bytes());
            }
            let normalized = state.normalize_data(&data);
            response_adapter::write_chunked_sse_event(tcp_stream, None, &normalized).await?;
        }
        if processed > 0 {
            carry = carry[processed..].to_string();
        }

        if done_seen {
            break;
        }

        let mut chunk = [0u8; 8192];
        let n = reader.read(&mut chunk).await?;
        if n == 0 {
            break;
        }
        let new_data = String::from_utf8_lossy(&chunk[..n]);
        carry.push_str(&new_data);
        if carry.contains('\r') {
            carry = carry.replace("\r\n", "\n");
        }
    }

    let _ = tcp_stream.write_all(b"0\r\n\r\n").await;
    let _ = tcp_stream.shutdown().await;
    Ok(RouteAttemptResult::Delivered {
        status_code: probe.status_code,
        completion_tokens: observed_completion_tokens,
    })
}

fn delivered_attempt_outcome(status_code: u16) -> crate::network::metrics::AttemptOutcome {
    match status_code {
        200..=299 => crate::network::metrics::AttemptOutcome::Success,
        400..=499 => crate::network::metrics::AttemptOutcome::Rejected,
        500..=599 => crate::network::metrics::AttemptOutcome::Unavailable,
        _ => crate::network::metrics::AttemptOutcome::Rejected,
    }
}

fn request_outcome_for_status(
    status_code: u16,
    service: crate::network::metrics::RequestService,
) -> crate::network::metrics::RequestOutcome {
    match status_code {
        200..=299 => crate::network::metrics::RequestOutcome::Success(service),
        _ => crate::network::metrics::RequestOutcome::Rejected(service),
    }
}

async fn relay_translated_responses_stream<R: AsyncRead + Unpin>(
    tcp_stream: &mut TcpStream,
    reader: &mut R,
    probe: ResponseProbe,
    retry_context_overflow: bool,
) -> Result<RouteAttemptResult> {
    fn should_parse_stream_chunk(data: &str, model_missing: bool, usage_missing: bool) -> bool {
        model_missing
            || usage_missing
            || data.contains("\"delta\"")
            || data.contains("\"content\"")
            || data.contains("\"logprobs\"")
            || data.contains("\"usage\"")
    }

    if retry_context_overflow && probe.retryable_context_overflow {
        return Ok(RouteAttemptResult::RetryableContextOverflow);
    }

    if !(200..300).contains(&probe.status_code) {
        return relay_error_response(tcp_stream, reader, probe).await;
    }

    let parsed = try_parse_response_headers(&probe.buffered)?
        .ok_or_else(|| anyhow!("incomplete HTTP response"))?;
    let mut carry = String::from_utf8_lossy(&probe.buffered[parsed.header_end..]).to_string();
    let mut state = ResponsesStreamRelayState::new();
    let header = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n";
    tcp_stream.write_all(header.as_bytes()).await?;

    let mut done_seen = false;
    loop {
        let mut processed = 0usize;
        while let Some(frame_end_rel) = carry[processed..].find("\n\n") {
            let frame_end = processed + frame_end_rel;
            let frame = &carry[processed..frame_end];
            processed = frame_end + 2;
            let data_lines = frame
                .lines()
                .filter_map(|line| line.strip_prefix("data:"))
                .map(str::trim_start)
                .collect::<Vec<_>>();
            if data_lines.is_empty() {
                continue;
            }
            let data = data_lines.join("\n");
            if data == "[DONE]" {
                done_seen = true;
                break;
            }

            if !should_parse_stream_chunk(&data, state.model.is_empty(), state.usage.is_none()) {
                continue;
            }

            process_translated_responses_frame(tcp_stream, &mut state, &data).await?;
        }
        if processed > 0 {
            carry = carry[processed..].to_string();
        }

        if done_seen {
            break;
        }

        let mut chunk = [0u8; 8192];
        let n = reader.read(&mut chunk).await?;
        if n == 0 {
            break;
        }
        let new_data = String::from_utf8_lossy(&chunk[..n]);
        carry.push_str(&new_data);
        // Normalize CRLF so frame parsing works for both LF and CRLF upstreams
        if carry.contains('\r') {
            carry = carry.replace("\r\n", "\n");
        }
    }

    finish_translated_responses_stream(tcp_stream, &mut state).await?;
    response_adapter::write_chunked_sse_event(tcp_stream, Some("done"), "[DONE]").await?;
    let _ = tcp_stream.write_all(b"0\r\n\r\n").await;
    let _ = tcp_stream.shutdown().await;
    Ok(RouteAttemptResult::Delivered {
        status_code: probe.status_code,
        completion_tokens: state.observed_completion_tokens,
    })
}

async fn process_translated_responses_frame(
    tcp_stream: &mut TcpStream,
    state: &mut ResponsesStreamRelayState,
    data: &str,
) -> Result<()> {
    let chunk = openai_frontend::parse_chat_stream_chunk(data)
        .context("parse typed upstream chat stream chunk")?;
    update_translated_responses_model(state, &chunk);
    emit_translated_response_created(tcp_stream, state).await?;
    emit_translated_reasoning_delta(tcp_stream, state, &chunk).await?;
    emit_translated_output_delta(tcp_stream, state, &chunk).await?;
    update_translated_responses_usage(state, &chunk);
    Ok(())
}

fn update_translated_responses_model(
    state: &mut ResponsesStreamRelayState,
    chunk: &openai_frontend::responses::ChatCompletionStreamChunk,
) {
    if let Some(chunk_model) = chunk.model.as_deref().filter(|_| state.model.is_empty()) {
        state.model = chunk_model.to_string();
    }
}

async fn emit_translated_response_created(
    tcp_stream: &mut TcpStream,
    state: &mut ResponsesStreamRelayState,
) -> Result<()> {
    if state.created_emitted || state.model.is_empty() {
        return Ok(());
    }
    let sequence_number = state.next_sequence_number();
    let created = serde_json::to_string(
        &response_adapter::responses_stream_created_event_with_sequence(
            &state.model,
            state.created_at,
            sequence_number,
        ),
    )
    .context("serialize response.created stream event")?;
    response_adapter::write_chunked_sse_event(tcp_stream, Some("response.created"), &created)
        .await?;
    state.created_emitted = true;
    Ok(())
}

async fn emit_translated_reasoning_delta(
    tcp_stream: &mut TcpStream,
    state: &mut ResponsesStreamRelayState,
    chunk: &openai_frontend::responses::ChatCompletionStreamChunk,
) -> Result<()> {
    let Some(delta) = chunk
        .choices
        .first()
        .and_then(|choice| choice.delta.as_ref())
        .and_then(|delta| delta.reasoning_content.as_deref())
    else {
        return Ok(());
    };
    let sequence_number = state.next_sequence_number();
    let event = serde_json::to_string(
        &response_adapter::responses_stream_reasoning_delta_event_with_sequence(
            &state.item_id,
            delta,
            sequence_number,
        ),
    )
    .context("serialize response.reasoning_text.delta event")?;
    response_adapter::write_chunked_sse_event(
        tcp_stream,
        Some("response.reasoning_text.delta"),
        &event,
    )
    .await?;
    Ok(())
}

async fn emit_translated_output_delta(
    tcp_stream: &mut TcpStream,
    state: &mut ResponsesStreamRelayState,
    chunk: &openai_frontend::responses::ChatCompletionStreamChunk,
) -> Result<()> {
    let Some(delta) = chunk
        .choices
        .first()
        .and_then(|choice| choice.delta.as_ref())
        .and_then(|delta| delta.content.as_deref())
    else {
        return Ok(());
    };
    emit_translated_output_item_prelude(tcp_stream, state).await?;
    let logprobs = chunk
        .choices
        .first()
        .and_then(|choice| choice.logprobs.clone());
    state.output_text.push_str(delta);
    let sequence_number = state.next_sequence_number();
    let event = serde_json::to_string(
        &response_adapter::responses_stream_delta_event_with_logprobs_and_sequence(
            &state.item_id,
            delta,
            logprobs,
            sequence_number,
        ),
    )
    .context("serialize response.output_text.delta event")?;
    response_adapter::write_chunked_sse_event(
        tcp_stream,
        Some("response.output_text.delta"),
        &event,
    )
    .await?;
    Ok(())
}

async fn emit_translated_output_item_prelude(
    tcp_stream: &mut TcpStream,
    state: &mut ResponsesStreamRelayState,
) -> Result<()> {
    if state.output_item_emitted {
        return Ok(());
    }
    let item_added_sequence_number = state.next_sequence_number();
    let item_added =
        serde_json::to_string(&response_adapter::responses_stream_output_item_added_event(
            &state.item_id,
            item_added_sequence_number,
        ))
        .context("serialize response.output_item.added event")?;
    response_adapter::write_chunked_sse_event(
        tcp_stream,
        Some("response.output_item.added"),
        &item_added,
    )
    .await?;
    let part_added_sequence_number = state.next_sequence_number();
    let part_added = serde_json::to_string(
        &response_adapter::responses_stream_content_part_added_event(
            &state.item_id,
            part_added_sequence_number,
        ),
    )
    .context("serialize response.content_part.added event")?;
    response_adapter::write_chunked_sse_event(
        tcp_stream,
        Some("response.content_part.added"),
        &part_added,
    )
    .await?;
    state.output_item_emitted = true;
    Ok(())
}

fn update_translated_responses_usage(
    state: &mut ResponsesStreamRelayState,
    chunk: &openai_frontend::responses::ChatCompletionStreamChunk,
) {
    if state.usage.is_none() {
        state.usage = chunk
            .usage
            .as_ref()
            .map(response_adapter::stream_usage_to_responses_usage);
    }
    if state.observed_completion_tokens.is_none() {
        state.observed_completion_tokens = chunk
            .usage
            .as_ref()
            .and_then(|usage| usage.completion_tokens);
    }
}

async fn finish_translated_responses_stream(
    tcp_stream: &mut TcpStream,
    state: &mut ResponsesStreamRelayState,
) -> Result<()> {
    emit_translated_fallback_created(tcp_stream, state).await?;
    emit_translated_output_item_prelude(tcp_stream, state).await?;
    let text_done_sequence_number = state.next_sequence_number();
    emit_translated_stream_done_event(
        tcp_stream,
        Some("response.output_text.done"),
        serde_json::to_string(
            &response_adapter::responses_stream_text_done_event_with_sequence(
                &state.item_id,
                &state.output_text,
                text_done_sequence_number,
            ),
        )
        .context("serialize response.output_text.done event")?,
    )
    .await?;
    let content_part_done_sequence_number = state.next_sequence_number();
    emit_translated_stream_done_event(
        tcp_stream,
        Some("response.content_part.done"),
        serde_json::to_string(&response_adapter::responses_stream_content_part_done_event(
            &state.item_id,
            &state.output_text,
            content_part_done_sequence_number,
        ))
        .context("serialize response.content_part.done event")?,
    )
    .await?;
    let output_item_done_sequence_number = state.next_sequence_number();
    emit_translated_stream_done_event(
        tcp_stream,
        Some("response.output_item.done"),
        serde_json::to_string(&response_adapter::responses_stream_output_item_done_event(
            &state.item_id,
            &state.output_text,
            output_item_done_sequence_number,
        ))
        .context("serialize response.output_item.done event")?,
    )
    .await?;
    let completed_sequence_number = state.next_sequence_number();
    let completed = serde_json::to_string(
        &response_adapter::responses_stream_completed_event_with_sequence(
            &state.response_id,
            state.created_at,
            &state.model,
            &state.item_id,
            &state.output_text,
            state.usage.clone(),
            completed_sequence_number,
        ),
    )
    .context("serialize response.completed event")?;
    response_adapter::write_chunked_sse_event(tcp_stream, Some("response.completed"), &completed)
        .await?;
    Ok(())
}

async fn emit_translated_fallback_created(
    tcp_stream: &mut TcpStream,
    state: &mut ResponsesStreamRelayState,
) -> Result<()> {
    if state.created_emitted {
        return Ok(());
    }
    let sequence_number = state.next_sequence_number();
    let created = serde_json::to_string(
        &response_adapter::responses_stream_created_event_with_sequence(
            &state.model,
            state.created_at,
            sequence_number,
        ),
    )
    .context("serialize response.created stream event")?;
    response_adapter::write_chunked_sse_event(tcp_stream, Some("response.created"), &created)
        .await?;
    state.created_emitted = true;
    Ok(())
}

async fn emit_translated_stream_done_event(
    tcp_stream: &mut TcpStream,
    event_name: Option<&str>,
    payload: String,
) -> Result<()> {
    response_adapter::write_chunked_sse_event(tcp_stream, event_name, &payload).await?;
    Ok(())
}

async fn relay_translated_responses_json<R: AsyncRead + Unpin>(
    tcp_stream: &mut TcpStream,
    reader: &mut R,
    probe: ResponseProbe,
    retry_context_overflow: bool,
) -> Result<RouteAttemptResult> {
    if retry_context_overflow && probe.retryable_context_overflow {
        return Ok(RouteAttemptResult::RetryableContextOverflow);
    }

    if !(200..300).contains(&probe.status_code) {
        return relay_error_response(tcp_stream, reader, probe).await;
    }
    let mut buffered = probe.buffered;
    reader.read_to_end(&mut buffered).await?;

    let parsed = try_parse_response_headers(&buffered)?
        .ok_or_else(|| anyhow!("incomplete HTTP response"))?;
    let translated_body =
        response_adapter::translate_chat_completion_to_responses(&buffered[parsed.header_end..])?;
    let completion_tokens = parse_completion_tokens_from_json_body(&translated_body);
    let header = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        translated_body.len()
    );
    tcp_stream.write_all(header.as_bytes()).await?;
    tcp_stream.write_all(&translated_body).await?;
    let _ = tcp_stream.shutdown().await;
    Ok(RouteAttemptResult::Delivered {
        status_code: probe.status_code,
        completion_tokens,
    })
}

/// Inject `"mesh_hooks": true/false` into the JSON body of an HTTP request.
///
/// Inserts the field right after the opening `{` in the body, then rebuilds
/// the Content-Length header to match.
pub fn inject_mesh_hooks_flag(raw: &mut Vec<u8>, enabled: bool) {
    let Some(header_end) = raw.windows(4).position(|w| w == b"\r\n\r\n").map(|i| i + 4) else {
        return;
    };
    let body = &raw[header_end..];
    let Some(brace) = body.iter().position(|&b| b == b'{') else {
        return;
    };

    // Build new body with mesh_hooks injected after opening brace
    let fragment = if enabled {
        &b"\"mesh_hooks\":true,"[..]
    } else {
        &b"\"mesh_hooks\":false,"[..]
    };
    let mut new_body = Vec::with_capacity(body.len() + fragment.len());
    new_body.extend_from_slice(&body[..brace + 1]);
    new_body.extend_from_slice(fragment);
    new_body.extend_from_slice(&body[brace + 1..]);

    // Rebuild headers with correct Content-Length
    let headers = std::str::from_utf8(&raw[..header_end - 4]).unwrap_or("");
    let mut rebuilt = String::new();
    for line in headers.split("\r\n") {
        if line.to_ascii_lowercase().starts_with("content-length:") {
            rebuilt.push_str(&format!("Content-Length: {}", new_body.len()));
        } else {
            rebuilt.push_str(line);
        }
        rebuilt.push_str("\r\n");
    }
    rebuilt.push_str("\r\n");

    let mut result = rebuilt.into_bytes();
    result.extend_from_slice(&new_body);
    *raw = result;
}

/// Rewrite the JSON body `model` field and rebuild Content-Length.
pub fn rewrite_model_field(request: &mut BufferedHttpRequest, model: &str) {
    let Some(header_end) = request
        .raw
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .map(|i| i + 4)
    else {
        return;
    };

    let Ok(mut body) = serde_json::from_slice::<serde_json::Value>(&request.raw[header_end..])
    else {
        return;
    };
    let Some(object) = body.as_object_mut() else {
        return;
    };

    object.insert(
        "model".to_string(),
        serde_json::Value::String(model.to_string()),
    );
    let Ok(new_body) = serde_json::to_vec(&body) else {
        return;
    };

    let headers = std::str::from_utf8(&request.raw[..header_end - 4]).unwrap_or("");
    let mut rebuilt = String::new();
    for line in headers.split("\r\n") {
        if line.to_ascii_lowercase().starts_with("content-length:") {
            rebuilt.push_str(&format!("Content-Length: {}", new_body.len()));
        } else {
            rebuilt.push_str(line);
        }
        rebuilt.push_str("\r\n");
    }
    rebuilt.push_str("\r\n");

    let mut raw = rebuilt.into_bytes();
    raw.extend_from_slice(&new_body);

    request.raw = raw;
    request.body_len_bytes = new_body.len();
    request.body_bytes = Some(new_body);
    request.body_json = Some(body);
    request.body_json_attempted = true;
    request.model_name = Some(model.to_string());
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
            let mut content_type = None;
            for header in response.headers.iter() {
                if header.name.eq_ignore_ascii_case("content-length") {
                    let value = std::str::from_utf8(header.value)
                        .context("invalid response Content-Length encoding")?;
                    content_length =
                        Some(value.trim().parse::<usize>().with_context(|| {
                            format!("invalid response Content-Length: {value}")
                        })?);
                } else if header.name.eq_ignore_ascii_case("content-type") {
                    content_type = Some(
                        std::str::from_utf8(header.value)
                            .context("invalid response Content-Type encoding")?
                            .trim()
                            .to_string(),
                    );
                }
            }
            Ok(Some(ParsedResponseHeaders {
                header_end,
                status_code: response.code.unwrap_or(0),
                content_length,
                content_type,
            }))
        }
        Ok(httparse::Status::Partial) => Ok(None),
        Err(err) => Err(anyhow!("HTTP response parse error: {err}")),
    }
}

/// Read the next chunk of HTTP response data without any timeout.
/// Used for continuation reads after the first byte has already arrived.
async fn read_response_chunk<R: AsyncRead + Unpin>(
    reader: &mut R,
    buf: &mut Vec<u8>,
) -> Result<usize> {
    let mut chunk = [0u8; 8192];
    let read_result = reader.read(&mut chunk).await?;
    if read_result == 0 {
        bail!("unexpected EOF while reading HTTP response");
    }
    buf.extend_from_slice(&chunk[..read_result]);
    Ok(read_result)
}

async fn probe_http_response<R: AsyncRead + Unpin>(reader: &mut R) -> Result<ResponseProbe> {
    probe_http_response_with_timeout(reader, response_first_byte_timeout()).await
}

/// Like `probe_http_response` but with a much longer timeout suitable for
/// the local OpenAI surface. Prefill on a busy or slow machine can
/// legitimately take minutes (large prompts, concurrent slot contention,
/// slower hardware). We still bound the wait to catch a truly wedged local
/// runtime path.
async fn probe_http_response_local<R: AsyncRead + Unpin>(reader: &mut R) -> Result<ResponseProbe> {
    probe_http_response_with_timeout(reader, local_response_first_byte_timeout()).await
}

/// Local OpenAI surface timeout: 10 minutes. This is a safety net for a wedged
/// local runtime path, not a latency budget. Normal prefill even on slow
/// hardware with large prompts and concurrent slots completes well within this
/// window.
fn local_response_first_byte_timeout() -> Duration {
    Duration::from_secs(10 * 60)
}

async fn probe_http_response_with_timeout<R: AsyncRead + Unpin>(
    reader: &mut R,
    timeout: Duration,
) -> Result<ResponseProbe> {
    let started = Instant::now();
    let mut buffered = Vec::with_capacity(8192);
    let parsed = loop {
        if let Some(parsed) = try_parse_response_headers(&buffered)? {
            break parsed;
        }
        let first_read = buffered.is_empty();
        if first_read {
            let mut chunk = [0u8; 8192];
            let read_result = tokio::time::timeout(timeout, reader.read(&mut chunk))
                .await
                .map_err(|_| {
                    anyhow!(
                        "upstream sent no response within {:.3}s",
                        timeout.as_secs_f64()
                    )
                })??;
            if read_result == 0 {
                bail!("unexpected EOF while reading HTTP response");
            }
            buffered.extend_from_slice(&chunk[..read_result]);
        } else {
            read_response_chunk(reader, &mut buffered).await?;
        }
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
        read_response_chunk(reader, &mut buffered).await?;
    }

    let retryable_context_overflow = parsed.status_code == 400
        && preview_len > 0
        && is_retryable_context_overflow_response(
            &buffered[parsed.header_end..parsed.header_end + preview_len],
        );
    tracing::debug!(
        status_code = parsed.status_code,
        header_bytes = parsed.header_end,
        probe_ms = started.elapsed().as_millis(),
        "openai transport: upstream response probe complete"
    );

    Ok(ResponseProbe {
        buffered,
        header_end: parsed.header_end,
        status_code: parsed.status_code,
        retryable_context_overflow,
    })
}

fn reason_phrase(status_code: u16) -> &'static str {
    match status_code {
        400 => "Bad Request",
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not Found",
        429 => "Too Many Requests",
        500 => "Internal Server Error",
        501 => "Not Implemented",
        502 => "Bad Gateway",
        503 => "Service Unavailable",
        _ => "Error",
    }
}

fn remap_error_http_response(
    status_code: u16,
    header_end: usize,
    full_response: &[u8],
) -> Option<Vec<u8>> {
    if status_code < 400 || header_end > full_response.len() {
        return None;
    }
    let mapped_body =
        openai_frontend::map_upstream_error_body(status_code, &full_response[header_end..])?;
    let header = format!(
        "HTTP/1.1 {} {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        status_code,
        reason_phrase(status_code),
        mapped_body.len()
    );
    let mut response = header.into_bytes();
    response.extend_from_slice(&mapped_body);
    Some(response)
}

fn oversized_error_http_response(status_code: u16) -> Vec<u8> {
    let body = serde_json::json!({
        "error": {
            "message": "upstream error response exceeded proxy limit",
            "type": "server_error",
            "param": serde_json::Value::Null,
            "code": "upstream_error_too_large",
        }
    })
    .to_string();
    format!(
        "HTTP/1.1 {} {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        status_code,
        reason_phrase(status_code),
        body.len(),
        body
    )
    .into_bytes()
}

async fn relay_error_response<R: AsyncRead + Unpin>(
    tcp_stream: &mut TcpStream,
    reader: &mut R,
    probe: ResponseProbe,
) -> Result<RouteAttemptResult> {
    let status_code = probe.status_code;
    let header_end = probe.header_end;
    let mut buffered = probe.buffered;
    let mut limited = reader.take((MAX_ERROR_RESPONSE_BYTES + 1) as u64);
    if let Err(err) = limited.read_to_end(&mut buffered).await {
        tracing::debug!("error response relay read ended before EOF: {err}");
    }
    let outgoing = if buffered.len().saturating_sub(header_end) > MAX_ERROR_RESPONSE_BYTES {
        tracing::warn!(
            "upstream error body exceeded {} bytes for status {}",
            MAX_ERROR_RESPONSE_BYTES,
            status_code
        );
        oversized_error_http_response(status_code)
    } else {
        remap_error_http_response(status_code, header_end, &buffered).unwrap_or(buffered)
    };
    tcp_stream.write_all(&outgoing).await?;
    let _ = tcp_stream.shutdown().await;
    Ok(RouteAttemptResult::Delivered {
        status_code,
        completion_tokens: None,
    })
}

async fn relay_success_response<R: AsyncRead + Unpin>(
    tcp_stream: &mut TcpStream,
    reader: &mut R,
    probe: ResponseProbe,
    parsed: ParsedResponseHeaders,
) -> Result<RouteAttemptResult> {
    if let Some(content_length) = parsed.content_length {
        const MAX_SUCCESS_METRICS_BODY_BYTES: usize = 1024 * 1024;
        if content_length <= MAX_SUCCESS_METRICS_BODY_BYTES {
            let mut buffered = probe.buffered;
            while buffered.len() < parsed.header_end + content_length {
                read_response_chunk(reader, &mut buffered).await?;
            }
            let completion_tokens =
                parse_completion_tokens_from_json_body(&buffered[parsed.header_end..]);
            tcp_stream.write_all(&buffered).await?;
            let _ = tcp_stream.shutdown().await;
            return Ok(RouteAttemptResult::Delivered {
                status_code: probe.status_code,
                completion_tokens,
            });
        }
    }

    tcp_stream.write_all(&probe.buffered).await?;
    if let Err(err) = tokio::io::copy(reader, &mut *tcp_stream).await {
        tracing::debug!("response relay ended after headers were committed: {err}");
    }
    let _ = tcp_stream.shutdown().await;
    Ok(RouteAttemptResult::Delivered {
        status_code: probe.status_code,
        completion_tokens: None,
    })
}

async fn relay_probed_response<R: AsyncRead + Unpin>(
    tcp_stream: &mut TcpStream,
    reader: &mut R,
    probe: ResponseProbe,
    retry_context_overflow: bool,
    response_adapter: ResponseAdapter,
) -> Result<RouteAttemptResult> {
    if let Some(result) = relay_adapted_response(
        tcp_stream,
        reader,
        probe.clone(),
        retry_context_overflow,
        response_adapter,
    )
    .await?
    {
        return Ok(result);
    }

    if retry_context_overflow && probe.retryable_context_overflow {
        return Ok(RouteAttemptResult::RetryableContextOverflow);
    }
    if !(200..300).contains(&probe.status_code) {
        return relay_error_response(tcp_stream, reader, probe).await;
    }

    let parsed = try_parse_response_headers(&probe.buffered)?
        .ok_or_else(|| anyhow!("incomplete HTTP response"))?;
    relay_success_response(tcp_stream, reader, probe, parsed).await
}

async fn relay_adapted_response<R: AsyncRead + Unpin>(
    tcp_stream: &mut TcpStream,
    reader: &mut R,
    probe: ResponseProbe,
    retry_context_overflow: bool,
    response_adapter: ResponseAdapter,
) -> Result<Option<RouteAttemptResult>> {
    match response_adapter {
        ResponseAdapter::OpenAiChatCompletionsStream => Ok(Some(
            relay_normalized_chat_completion_stream(
                tcp_stream,
                reader,
                probe,
                retry_context_overflow,
            )
            .await?,
        )),
        ResponseAdapter::OpenAiResponsesJson => Ok(Some(
            relay_translated_responses_json(tcp_stream, reader, probe, retry_context_overflow)
                .await?,
        )),
        ResponseAdapter::OpenAiResponsesStream => Ok(Some(
            relay_translated_responses_stream(tcp_stream, reader, probe, retry_context_overflow)
                .await?,
        )),
        ResponseAdapter::None => Ok(None),
    }
}

async fn route_local_attempt(
    node: &mesh::Node,
    tcp_stream: &mut TcpStream,
    port: u16,
    prefetched: &[u8],
    retry_context_overflow: bool,
    response_adapter: ResponseAdapter,
) -> RouteAttemptResult {
    match TcpStream::connect(format!("127.0.0.1:{port}")).await {
        Ok(mut upstream) => {
            let _inflight = node.begin_inflight_request();
            let _ = upstream.set_nodelay(true);
            if let Err(err) = upstream.write_all(prefetched).await {
                tracing::warn!(
                    "API proxy: failed to forward buffered request to local OpenAI surface on {port}: {err}"
                );
                return RouteAttemptResult::RetryableUnavailable;
            }
            route_local_attempt_after_forward(
                tcp_stream,
                &mut upstream,
                port,
                retry_context_overflow,
                response_adapter,
            )
            .await
        }
        Err(err) => {
            tracing::warn!("API proxy: can't reach local OpenAI surface on {port}: {err}");
            RouteAttemptResult::RetryableUnavailable
        }
    }
}

async fn route_local_attempt_after_forward(
    tcp_stream: &mut TcpStream,
    upstream: &mut TcpStream,
    port: u16,
    retry_context_overflow: bool,
    response_adapter: ResponseAdapter,
) -> RouteAttemptResult {
    match probe_http_response_local(upstream).await {
        Ok(probe) => {
            let result = relay_attempted_response(
                tcp_stream,
                upstream,
                probe,
                retry_context_overflow,
                response_adapter,
                "API proxy (local): downstream client disconnected during relay",
                "API proxy (local) ended after commit",
            )
            .await;
            if matches!(result, RouteAttemptResult::ClientDisconnected) {
                let _ = upstream.shutdown().await;
            }
            result
        }
        Err(err) => {
            tracing::warn!(
                "API proxy: failed to read local response from OpenAI surface on {port}: {err}"
            );
            retryable_route_result_from_error(&err)
        }
    }
}

async fn route_remote_attempt(
    node: &mesh::Node,
    tcp_stream: &mut TcpStream,
    host_id: iroh::EndpointId,
    prefetched: &[u8],
    retry_context_overflow: bool,
    response_adapter: ResponseAdapter,
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
            route_remote_attempt_after_forward(
                tcp_stream,
                &mut quic_recv,
                host_id,
                retry_context_overflow,
                response_adapter,
            )
            .await
        }
        Err(err) => {
            tracing::warn!(
                "API proxy: can't tunnel to host {}: {err}",
                host_id.fmt_short()
            );
            retryable_route_result_from_error(&err)
        }
    }
}

async fn route_remote_attempt_after_forward(
    tcp_stream: &mut TcpStream,
    quic_recv: &mut iroh::endpoint::RecvStream,
    host_id: iroh::EndpointId,
    retry_context_overflow: bool,
    response_adapter: ResponseAdapter,
) -> RouteAttemptResult {
    match probe_http_response(quic_recv).await {
        Ok(probe) => {
            relay_attempted_response(
                tcp_stream,
                quic_recv,
                probe,
                retry_context_overflow,
                response_adapter,
                "API proxy (remote): downstream client disconnected during relay",
                "API proxy (remote) ended after commit",
            )
            .await
        }
        Err(err) => {
            tracing::warn!(
                "API proxy: failed to read response from host {}: {err}",
                host_id.fmt_short()
            );
            retryable_route_result_from_error(&err)
        }
    }
}

async fn route_http_endpoint_attempt(
    tcp_stream: &mut TcpStream,
    base_url: &str,
    prefetched: &[u8],
    request_path: &str,
    retry_context_overflow: bool,
    response_adapter: ResponseAdapter,
) -> RouteAttemptResult {
    let target = match build_external_endpoint_target(base_url, request_path, prefetched) {
        Ok(target) => target,
        Err(()) => return RouteAttemptResult::RetryableUnavailable,
    };
    let mut upstream = match connect_external_endpoint(base_url, &target).await {
        Ok(upstream) => upstream,
        Err(result) => return result,
    };
    if let Err(result) = forward_external_endpoint_request(&mut upstream, base_url, &target).await {
        return result;
    }
    route_http_endpoint_attempt_after_forward(
        tcp_stream,
        &mut upstream,
        base_url,
        retry_context_overflow,
        response_adapter,
    )
    .await
}

async fn connect_external_endpoint(
    base_url: &str,
    target: &ExternalEndpointTarget,
) -> std::result::Result<TcpStream, RouteAttemptResult> {
    match TcpStream::connect(format!("{}:{}", target.host, target.port)).await {
        Ok(upstream) => Ok(upstream),
        Err(err) => {
            tracing::warn!(
                "API proxy: can't reach external inference endpoint {}: {}",
                base_url,
                err
            );
            Err(if err.kind() == std::io::ErrorKind::TimedOut {
                RouteAttemptResult::RetryableTimeout
            } else {
                RouteAttemptResult::RetryableUnavailable
            })
        }
    }
}

async fn forward_external_endpoint_request(
    upstream: &mut TcpStream,
    base_url: &str,
    target: &ExternalEndpointTarget,
) -> std::result::Result<(), RouteAttemptResult> {
    let _ = upstream.set_nodelay(true);
    match upstream.write_all(&target.forwarded).await {
        Ok(()) => Ok(()),
        Err(err) => {
            tracing::warn!(
                "API proxy: failed to forward buffered request to external endpoint {}: {}",
                base_url,
                err
            );
            Err(RouteAttemptResult::RetryableUnavailable)
        }
    }
}

async fn route_http_endpoint_attempt_after_forward(
    tcp_stream: &mut TcpStream,
    upstream: &mut TcpStream,
    base_url: &str,
    retry_context_overflow: bool,
    response_adapter: ResponseAdapter,
) -> RouteAttemptResult {
    match probe_http_response(upstream).await {
        Ok(probe) => {
            let result = relay_attempted_response(
                tcp_stream,
                upstream,
                probe,
                retry_context_overflow,
                response_adapter,
                "API proxy (external endpoint): downstream client disconnected during relay",
                "API proxy (external endpoint) ended after commit",
            )
            .await;
            if matches!(result, RouteAttemptResult::ClientDisconnected) {
                let _ = upstream.shutdown().await;
            }
            result
        }
        Err(err) => {
            tracing::warn!(
                "API proxy: failed to read response from external endpoint {}: {}",
                base_url,
                err
            );
            retryable_route_result_from_error(&err)
        }
    }
}

async fn relay_attempted_response<R: AsyncRead + Unpin>(
    tcp_stream: &mut TcpStream,
    reader: &mut R,
    probe: ResponseProbe,
    retry_context_overflow: bool,
    response_adapter: ResponseAdapter,
    disconnect_message: &str,
    commit_message: &str,
) -> RouteAttemptResult {
    let status_code = probe.status_code;
    match relay_probed_response(
        tcp_stream,
        reader,
        probe,
        retry_context_overflow,
        response_adapter,
    )
    .await
    {
        Ok(result) => result,
        Err(err) => {
            if is_client_disconnect_error(&err) {
                tracing::info!("{disconnect_message}");
                return RouteAttemptResult::ClientDisconnected;
            }
            tracing::debug!("{commit_message}: {err}");
            RouteAttemptResult::Delivered {
                status_code,
                completion_tokens: None,
            }
        }
    }
}

fn retryable_route_result_from_error(err: &anyhow::Error) -> RouteAttemptResult {
    if is_timeout_error(err) {
        RouteAttemptResult::RetryableTimeout
    } else {
        RouteAttemptResult::RetryableUnavailable
    }
}

fn attempt_outcome_for_result(
    result: &RouteAttemptResult,
) -> crate::network::metrics::AttemptOutcome {
    match result {
        RouteAttemptResult::Delivered { status_code, .. } => {
            delivered_attempt_outcome(*status_code)
        }
        RouteAttemptResult::RetryableTimeout => crate::network::metrics::AttemptOutcome::Timeout,
        RouteAttemptResult::RetryableUnavailable => {
            crate::network::metrics::AttemptOutcome::Unavailable
        }
        RouteAttemptResult::RetryableContextOverflow => {
            crate::network::metrics::AttemptOutcome::ContextOverflow
        }
        RouteAttemptResult::ClientDisconnected => {
            crate::network::metrics::AttemptOutcome::Unavailable
        }
    }
}

fn completion_tokens_for_result(result: &RouteAttemptResult) -> Option<u64> {
    match result {
        RouteAttemptResult::Delivered {
            completion_tokens, ..
        } => *completion_tokens,
        _ => None,
    }
}

fn request_service_for_target(
    target: &election::InferenceTarget,
) -> crate::network::metrics::RequestService {
    match target {
        election::InferenceTarget::Local(_) => crate::network::metrics::RequestService::Local,
        election::InferenceTarget::Remote(_) | election::InferenceTarget::None => {
            crate::network::metrics::RequestService::Remote
        }
    }
}

enum AutoModelResolution {
    Model(Option<String>),
    UnsupportedMedia,
}

enum MeshTargetResolution {
    Hosts(Vec<iroh::EndpointId>),
    ModelUnavailable(String),
    NoHostsAvailable,
}

struct MeshRequestPlan {
    effective_model: Option<String>,
    auto_session_key: Option<u64>,
    prepared: PreparedTargets,
    target_hosts: Vec<iroh::EndpointId>,
}

enum MeshRequestFailure {
    UnsupportedMedia,
    ModelUnavailable(String),
    NoHostsAvailable,
}

struct MeshAttemptState {
    route_started: Instant,
    attempts: usize,
    last_retryable: bool,
    refreshed: bool,
}

enum MeshAttemptDisposition {
    Continue,
    Return,
}

fn build_external_endpoint_target(
    base_url: &str,
    request_path: &str,
    prefetched: &[u8],
) -> std::result::Result<ExternalEndpointTarget, ()> {
    let (url, host) = parse_external_endpoint_url(base_url)?;
    let port = url.port_or_known_default().unwrap_or(80);
    let forward_path = endpoint_forward_path(&url, request_path);
    let forwarded =
        rewrite_external_endpoint_request(base_url, prefetched, &forward_path, &host, port)?;
    Ok(ExternalEndpointTarget {
        host,
        port,
        forwarded,
    })
}

fn parse_external_endpoint_url(base_url: &str) -> std::result::Result<(Url, String), ()> {
    let url = parse_external_endpoint_base_url(base_url)?;
    validate_external_endpoint_scheme(base_url, &url)?;
    let host = parse_external_endpoint_host(base_url, &url)?;
    Ok((url, host))
}

fn parse_external_endpoint_base_url(base_url: &str) -> std::result::Result<Url, ()> {
    Url::parse(base_url).map_err(|err| {
        tracing::warn!("API proxy: invalid external inference endpoint '{base_url}': {err}");
    })
}

fn validate_external_endpoint_scheme(base_url: &str, url: &Url) -> std::result::Result<(), ()> {
    if url.scheme() == "http" {
        return Ok(());
    }
    tracing::warn!(
        "API proxy: unsupported external inference endpoint scheme '{}' for {}",
        url.scheme(),
        base_url
    );
    Err(())
}

fn parse_external_endpoint_host(base_url: &str, url: &Url) -> std::result::Result<String, ()> {
    url.host_str().map(str::to_string).ok_or_else(|| {
        tracing::warn!("API proxy: missing host in external inference endpoint {base_url}");
    })
}

fn rewrite_external_endpoint_request(
    base_url: &str,
    prefetched: &[u8],
    forward_path: &str,
    host: &str,
    port: u16,
) -> std::result::Result<Vec<u8>, ()> {
    match rewrite_http_request_target(prefetched, forward_path, host, port) {
        Ok(forwarded) => Ok(forwarded),
        Err(err) => {
            tracing::warn!(
                "API proxy: failed to rewrite buffered request for external endpoint {}: {}",
                base_url,
                err
            );
            Err(())
        }
    }
}

fn endpoint_forward_path(base_url: &Url, request_path: &str) -> String {
    let (path_only, query) = request_path
        .split_once('?')
        .map(|(path, query)| (path, Some(query)))
        .unwrap_or((request_path, None));
    let base_path = base_url.path().trim_end_matches('/');
    let mapped_path = if base_path.is_empty() || base_path == "/" {
        path_only.to_string()
    } else if let Some(suffix) = path_only.strip_prefix("/v1") {
        if base_path.ends_with("/v1") {
            format!("{base_path}{suffix}")
        } else {
            format!("{base_path}/v1{suffix}")
        }
    } else if let Some(suffix) = path_only.strip_prefix("/models") {
        format!("{base_path}{suffix}")
    } else {
        format!("{base_path}{path_only}")
    };
    match query {
        Some(query) if !query.is_empty() => format!("{mapped_path}?{query}"),
        _ => mapped_path,
    }
}

fn rewrite_http_request_target(
    raw: &[u8],
    new_path: &str,
    host: &str,
    port: u16,
) -> Result<Vec<u8>> {
    let header_end = raw
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .map(|idx| idx + 4)
        .context("missing HTTP header terminator")?;
    let header_text =
        std::str::from_utf8(&raw[..header_end - 4]).context("invalid HTTP headers")?;
    let mut lines = header_text.split("\r\n");
    let request_line = lines.next().context("missing HTTP request line")?;
    let mut request_parts = request_line.split_whitespace();
    let method = request_parts.next().context("missing HTTP method")?;
    let _old_path = request_parts.next().context("missing HTTP path")?;
    let version = request_parts.next().unwrap_or("HTTP/1.1");

    let mut rebuilt = format!("{method} {new_path} {version}\r\n");
    let mut saw_host = false;
    for line in lines {
        if let Some((name, _value)) = line.split_once(':') {
            if name.eq_ignore_ascii_case("host") {
                rebuilt.push_str(&format!("Host: {host}:{port}\r\n"));
                saw_host = true;
                continue;
            }
        }
        rebuilt.push_str(line);
        rebuilt.push_str("\r\n");
    }
    if !saw_host {
        rebuilt.push_str(&format!("Host: {host}:{port}\r\n"));
    }
    rebuilt.push_str("\r\n");

    let mut bytes = rebuilt.into_bytes();
    bytes.extend_from_slice(&raw[header_end..]);
    Ok(bytes)
}

fn should_learn_affinity(status_code: u16) -> bool {
    (200..400).contains(&status_code)
}

fn cached_auto_model_satisfies_media_requirements(
    model: &str,
    media: &router::MediaRequirements,
    descriptors: &[mesh::ServedModelDescriptor],
) -> bool {
    let caps = capabilities_for_model(model, descriptors);
    router::model_satisfies_media_requirements(&caps, media)
}

pub(crate) fn capabilities_for_model(
    model: &str,
    descriptors: &[mesh::ServedModelDescriptor],
) -> crate::models::ModelCapabilities {
    descriptor_for_model(descriptors, model)
        .filter(|descriptor| descriptor.capabilities_known)
        .map(|descriptor| descriptor.capabilities)
        .unwrap_or_else(|| crate::models::installed_model_capabilities(model))
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
    let plugin_manager = node.plugin_manager().await;
    let mut request =
        match read_http_request_with_plugin_manager(&mut tcp_stream, plugin_manager.as_ref()).await
        {
            Ok(v) => v,
            Err(err) => {
                let _ = send_400(tcp_stream, &err.to_string()).await;
                return;
            }
        };

    // Handle /v1/models
    if is_models_list_request(&request.method, &request.path) {
        let served = node.models_being_served().await;
        let descriptors = node.served_model_descriptors().await;
        let _ = send_models_list_with_descriptors(tcp_stream, &served, &descriptors).await;
        return;
    }

    // MoA routing directive: `model: "mesh"` triggers mixture-of-agents
    // fan-out. Orchestration happens here, regardless of whether this node
    // is serving models locally — the worker pool is built from gossip.
    // On a pure --client node every backend is remote (QUIC tunnels to
    // peers serving each model); on a host node the locally-served model
    // is wired directly to its skippy port via the targets table.
    //
    // try_handle_moa self-gates on the model name and returns the stream
    // back unchanged if this isn't a MoA request, so we can call it
    // unconditionally here.
    let moa_model_name = request.model_name.clone();
    let tcp_stream = match crate::network::openai::moa_gateway::try_handle_moa(
        &node,
        tcp_stream,
        &mut request,
        moa_model_name.as_deref(),
        None, // passive path has no local targets table
    )
    .await
    {
        Some(stream) => stream,
        None => {
            // MoA handled the request and consumed the stream.
            release_request_objects(&node, &request.request_object_request_ids).await;
            return;
        }
    };

    let plan = match build_mesh_request_plan(&node, &mut request, track_demand, &affinity).await {
        Ok(plan) => plan,
        Err(failure) => {
            handle_mesh_request_failure(&node, tcp_stream, &request, failure).await;
            return;
        }
    };
    if let Some(tcp_stream) =
        route_mesh_request_attempts(&node, tcp_stream, &request, &plan, &affinity).await
    {
        finish_exhausted_mesh_request(
            &node,
            tcp_stream,
            plan.effective_model.as_deref(),
            plan.target_hosts.len(),
            &affinity,
        )
        .await;
    }
    release_request_objects(&node, &request.request_object_request_ids).await;
}

async fn build_mesh_request_plan(
    node: &mesh::Node,
    request: &mut BufferedHttpRequest,
    track_demand: bool,
    affinity: &AffinityRouter,
) -> std::result::Result<MeshRequestPlan, MeshRequestFailure> {
    let served = node.models_being_served().await;
    let descriptors = node.served_model_descriptors().await;
    rewrite_public_model_alias(request, &served, &descriptors);

    let is_auto_request =
        request.model_name.is_none() || request.model_name.as_deref() == Some("auto");
    let auto_session_key = auto_session_key_for_request(request, is_auto_request);
    let effective_model = match resolve_auto_model_request(
        node,
        request,
        &served,
        &descriptors,
        is_auto_request,
        auto_session_key,
        affinity,
    )
    .await
    {
        AutoModelResolution::Model(model) => model.or(request.model_name.clone()),
        AutoModelResolution::UnsupportedMedia => return Err(MeshRequestFailure::UnsupportedMedia),
    };
    rewrite_effective_model(request, effective_model.as_deref());
    if is_auto_request {
        inject_mesh_hooks_flag(&mut request.raw, true);
    }
    if track_demand {
        if let Some(name) = effective_model.as_deref() {
            node.record_request(name);
        }
    }

    let resolved_hosts = match resolve_mesh_target_hosts(node, effective_model.as_deref()).await {
        MeshTargetResolution::Hosts(hosts) => hosts,
        MeshTargetResolution::ModelUnavailable(model) => {
            return Err(MeshRequestFailure::ModelUnavailable(model));
        }
        MeshTargetResolution::NoHostsAvailable => return Err(MeshRequestFailure::NoHostsAvailable),
    };

    let required_tokens =
        request_budget_tokens_from_parts(request.body_len_bytes, request.completion_tokens);
    let prepared = prepare_mesh_targets(
        request,
        effective_model.as_deref(),
        &resolved_hosts,
        affinity,
    );
    let target_hosts = order_mesh_target_hosts(
        node,
        effective_model.as_deref(),
        required_tokens,
        &prepared,
        affinity,
    )
    .await;
    Ok(MeshRequestPlan {
        effective_model,
        auto_session_key,
        prepared,
        target_hosts,
    })
}

fn rewrite_effective_model(request: &mut BufferedHttpRequest, effective_model: Option<&str>) {
    if let Some(name) = effective_model {
        if request.model_name.as_deref() != Some(name) {
            rewrite_model_field(request, name);
        }
    }
}

fn prepare_mesh_targets(
    request: &mut BufferedHttpRequest,
    effective_model: Option<&str>,
    target_hosts: &[iroh::EndpointId],
    affinity: &AffinityRouter,
) -> PreparedTargets {
    if effective_model.is_some() && target_hosts.len() > 1 {
        request.ensure_body_json();
    }
    let body_json = request.body_json.as_ref();
    effective_model
        .map(|name| prepare_remote_targets_for_request(name, target_hosts, body_json, affinity))
        .unwrap_or(PreparedTargets {
            ordered: target_hosts
                .iter()
                .copied()
                .map(election::InferenceTarget::Remote)
                .collect(),
            learn_prefix_hash: None,
            cached_target: None,
        })
}

async fn order_mesh_target_hosts(
    node: &mesh::Node,
    effective_model: Option<&str>,
    required_tokens: Option<u32>,
    prepared: &PreparedTargets,
    affinity: &AffinityRouter,
) -> Vec<iroh::EndpointId> {
    let target_hosts: Vec<iroh::EndpointId> = prepared
        .ordered
        .iter()
        .filter_map(|target| match target {
            election::InferenceTarget::Remote(host_id) => Some(*host_id),
            _ => None,
        })
        .collect();
    let Some(name) = effective_model else {
        return target_hosts;
    };
    let mut ordered =
        order_remote_hosts_by_context(node, name, required_tokens, &target_hosts).await;
    if let (Some(prefix_hash), Some(election::InferenceTarget::Remote(cached_host))) =
        (prepared.learn_prefix_hash, prepared.cached_target.as_ref())
    {
        let cached_context = node.peer_model_context_length(*cached_host, name).await;
        if matches!((required_tokens, cached_context), (Some(required), Some(context)) if context < required)
        {
            affinity.forget_target(
                name,
                prefix_hash,
                &election::InferenceTarget::Remote(*cached_host),
            );
        } else {
            move_target_first(&mut ordered, cached_host);
        }
    }
    ordered
}

async fn handle_mesh_request_failure(
    node: &mesh::Node,
    tcp_stream: TcpStream,
    request: &BufferedHttpRequest,
    failure: MeshRequestFailure,
) {
    let mut tcp_stream = Some(tcp_stream);
    match failure {
        MeshRequestFailure::UnsupportedMedia => {
            let _ = send_error(
                tcp_stream.take().unwrap(),
                422,
                "no served model can satisfy the requested media inputs",
            )
            .await;
        }
        MeshRequestFailure::ModelUnavailable(model) => {
            node.record_routed_request(
                Some(&model),
                0,
                crate::network::metrics::RequestOutcome::Unavailable,
            );
            tracing::warn!(
                "API proxy: model {:?} not available, no hosts serving it",
                model
            );
            let _ = send_error(
                tcp_stream.take().unwrap(),
                429,
                &format!("model {:?} not currently available — retry later", model),
            )
            .await;
        }
        MeshRequestFailure::NoHostsAvailable => {
            node.record_routed_request(
                None,
                0,
                crate::network::metrics::RequestOutcome::Unavailable,
            );
            let _ = send_503(
                tcp_stream.take().unwrap(),
                "no peers serving any model (mesh empty or gossip stale)",
            )
            .await;
        }
    }
    release_request_objects(node, &request.request_object_request_ids).await;
}

async fn route_mesh_request_attempts(
    node: &mesh::Node,
    mut tcp_stream: TcpStream,
    request: &BufferedHttpRequest,
    plan: &MeshRequestPlan,
    affinity: &AffinityRouter,
) -> Option<TcpStream> {
    let effective_model = plan.effective_model.as_deref();
    let auto_session_key = plan.auto_session_key;
    let prepared = &plan.prepared;
    let target_hosts = &plan.target_hosts;
    let total_targets = target_hosts.len();
    let mut state = MeshAttemptState {
        route_started: Instant::now(),
        attempts: 0,
        last_retryable: false,
        refreshed: false,
    };
    for (idx, target_host) in target_hosts.iter().enumerate() {
        state.attempts += 1;
        let attempt_started = Instant::now();
        let attempt_result = route_remote_attempt(
            node,
            &mut tcp_stream,
            *target_host,
            &request.raw,
            idx + 1 < total_targets,
            request.response_adapter,
        )
        .await;
        let attempt_target = election::InferenceTarget::Remote(*target_host);
        record_mesh_request_attempt(
            node,
            effective_model,
            &attempt_target,
            attempt_started.duration_since(state.route_started),
            attempt_started.elapsed(),
            &attempt_result,
        );
        affinity.record_target_outcome(
            effective_model,
            &attempt_target,
            target_health_outcome_for_attempt(&attempt_result),
        );
        let mut context = MeshAttemptResultContext {
            node,
            effective_model,
            auto_session_key,
            prepared,
            attempt_target: &attempt_target,
            target_host: *target_host,
            state: &mut state,
            affinity,
        };
        match handle_mesh_attempt_result(&mut context, attempt_result) {
            MeshAttemptDisposition::Continue => continue,
            MeshAttemptDisposition::Return => return None,
        }
    }
    if state.last_retryable {
        tracing::warn!("All hosts failed for model {:?}", effective_model);
        if let Some(key) = auto_session_key {
            tracing::debug!(
                "auto: all hosts failed for cached model, forgetting session {key:016x}"
            );
            affinity.forget_auto_model(key);
        }
    }
    node.record_routed_request(
        effective_model,
        state.attempts,
        crate::network::metrics::RequestOutcome::Unavailable,
    );
    Some(tcp_stream)
}

fn record_mesh_request_attempt(
    node: &mesh::Node,
    effective_model: Option<&str>,
    attempt_target: &election::InferenceTarget,
    queue_wait: Duration,
    attempt_time: Duration,
    attempt_result: &RouteAttemptResult,
) {
    if matches!(attempt_result, RouteAttemptResult::ClientDisconnected) {
        return;
    }
    node.record_inference_attempt(
        effective_model,
        attempt_target,
        queue_wait,
        attempt_time,
        attempt_outcome_for_result(attempt_result),
        completion_tokens_for_result(attempt_result),
    );
}

struct MeshAttemptResultContext<'a> {
    node: &'a mesh::Node,
    effective_model: Option<&'a str>,
    auto_session_key: Option<u64>,
    prepared: &'a PreparedTargets,
    attempt_target: &'a election::InferenceTarget,
    target_host: iroh::EndpointId,
    state: &'a mut MeshAttemptState,
    affinity: &'a AffinityRouter,
}

fn handle_mesh_attempt_result(
    context: &mut MeshAttemptResultContext<'_>,
    attempt_result: RouteAttemptResult,
) -> MeshAttemptDisposition {
    match attempt_result {
        RouteAttemptResult::Delivered { status_code, .. } => {
            handle_delivered_mesh_attempt(context, status_code)
        }
        RouteAttemptResult::RetryableContextOverflow => handle_retryable_context_overflow(context),
        RouteAttemptResult::RetryableTimeout => handle_retryable_mesh_timeout(context),
        RouteAttemptResult::RetryableUnavailable => handle_retryable_mesh_unavailable(context),
        RouteAttemptResult::ClientDisconnected => {
            tracing::info!(
                "Downstream client disconnected while routing to host {}",
                context.target_host.fmt_short()
            );
            MeshAttemptDisposition::Return
        }
    }
}

fn handle_delivered_mesh_attempt(
    context: &MeshAttemptResultContext<'_>,
    status_code: u16,
) -> MeshAttemptDisposition {
    if should_learn_affinity(status_code) {
        if let (Some(name), Some(prefix_hash)) =
            (context.effective_model, context.prepared.learn_prefix_hash)
        {
            context
                .affinity
                .learn_target(name, prefix_hash, context.attempt_target);
        }
    } else if let Some(key) = context
        .auto_session_key
        .filter(|_| (500..600).contains(&status_code))
    {
        tracing::debug!(
            "auto: upstream returned {status_code}, forgetting cached model for session {key:016x}"
        );
        context.affinity.forget_auto_model(key);
    }
    context.node.record_routed_request(
        context.effective_model,
        context.state.attempts,
        request_outcome_for_status(status_code, crate::network::metrics::RequestService::Remote),
    );
    MeshAttemptDisposition::Return
}

fn handle_retryable_context_overflow(
    context: &mut MeshAttemptResultContext<'_>,
) -> MeshAttemptDisposition {
    forget_mesh_cached_target(
        context.effective_model,
        context.prepared,
        context.attempt_target,
        context.affinity,
    );
    tracing::warn!(
        "Host {} rejected request with context overflow-style 400, trying next",
        context.target_host.fmt_short()
    );
    context.state.last_retryable = true;
    MeshAttemptDisposition::Continue
}

fn handle_retryable_mesh_timeout(
    context: &mut MeshAttemptResultContext<'_>,
) -> MeshAttemptDisposition {
    tracing::warn!(
        "Host {} timed out, trying next",
        context.target_host.fmt_short()
    );
    context.state.last_retryable = true;
    spawn_mesh_refresh_once(context.node, &mut context.state.refreshed);
    MeshAttemptDisposition::Continue
}

fn handle_retryable_mesh_unavailable(
    context: &mut MeshAttemptResultContext<'_>,
) -> MeshAttemptDisposition {
    forget_mesh_cached_target(
        context.effective_model,
        context.prepared,
        context.attempt_target,
        context.affinity,
    );
    tracing::warn!(
        "Failed to tunnel to host {}, trying next",
        context.target_host.fmt_short()
    );
    context.state.last_retryable = true;
    spawn_mesh_refresh_once(context.node, &mut context.state.refreshed);
    MeshAttemptDisposition::Continue
}

fn forget_mesh_cached_target(
    effective_model: Option<&str>,
    prepared: &PreparedTargets,
    failed_target: &election::InferenceTarget,
    affinity: &AffinityRouter,
) {
    if let (Some(name), Some(prefix_hash), Some(cached_target)) = (
        effective_model,
        prepared.learn_prefix_hash,
        prepared.cached_target.as_ref(),
    ) {
        if cached_target == failed_target {
            affinity.forget_target(name, prefix_hash, failed_target);
        }
    }
}

fn spawn_mesh_refresh_once(node: &mesh::Node, refreshed: &mut bool) {
    if *refreshed {
        return;
    }
    let refresh_node = node.clone();
    tokio::spawn(async move {
        refresh_node.gossip_one_peer().await;
    });
    *refreshed = true;
}

async fn finish_exhausted_mesh_request(
    node: &mesh::Node,
    tcp_stream: TcpStream,
    effective_model: Option<&str>,
    total_targets: usize,
    affinity: &AffinityRouter,
) {
    let reason = format!(
        "all {} tunnel(s) to hosts for {:?} failed (mesh request)",
        total_targets, effective_model,
    );
    let _ = affinity;
    let _ = node;
    let _ = send_503(tcp_stream, &reason).await;
}

fn auto_session_key_for_request(
    request: &mut BufferedHttpRequest,
    is_auto_request: bool,
) -> Option<u64> {
    if !is_auto_request {
        return None;
    }
    request.ensure_body_json();
    request
        .body_json
        .as_ref()
        .and_then(|body| crate::network::affinity::auto_model_session_key(Some(body)))
}

async fn resolve_auto_model_request(
    node: &mesh::Node,
    request: &mut BufferedHttpRequest,
    served: &[String],
    descriptors: &[mesh::ServedModelDescriptor],
    is_auto_request: bool,
    auto_session_key: Option<u64>,
    affinity: &AffinityRouter,
) -> AutoModelResolution {
    if !is_auto_request {
        return AutoModelResolution::Model(None);
    }
    request.ensure_body_json();
    let Some(body_json) = request.body_json.as_ref() else {
        return AutoModelResolution::Model(None);
    };
    let media = router::media_requirements(body_json);
    if let Some(model) =
        lookup_cached_auto_model(node, descriptors, affinity, auto_session_key, &media).await
    {
        return AutoModelResolution::Model(Some(model));
    }

    let cl = router::classify(body_json);
    // Build candidates with observed throughput so pick_model_classified
    // can weight by locally-measured tok/s where samples exist.
    let routing_metrics = node.routing_metrics();
    let with_caps: Vec<router::RoutingCandidate<'_>> = served
        .iter()
        .map(|name| {
            let caps = capabilities_for_model(name, descriptors);
            let (tps_hint, throughput_samples) = routing_metrics
                .tps_for_model(name)
                .map(|(tps, samples)| (Some(tps), samples))
                .unwrap_or((None, 0));
            router::RoutingCandidate {
                name: name.as_str(),
                caps,
                tps_hint,
                throughput_samples,
            }
        })
        .collect();
    let Some(available) = router::filter_media_compatible_candidates(&with_caps, &media) else {
        return AutoModelResolution::UnsupportedMedia;
    };
    let picked = router::pick_model_classified(&cl, &available).map(str::to_string);
    if let Some(name) = picked.as_deref() {
        tracing::info!(
            "router: {:?}/{:?} tools={} media={} → {name}",
            cl.category,
            cl.complexity,
            cl.needs_tools,
            cl.has_media_inputs
        );
        if let Some(key) = auto_session_key {
            affinity.remember_auto_model(key, name);
        }
    }
    AutoModelResolution::Model(picked)
}

async fn lookup_cached_auto_model(
    node: &mesh::Node,
    descriptors: &[mesh::ServedModelDescriptor],
    affinity: &AffinityRouter,
    auto_session_key: Option<u64>,
    media: &router::MediaRequirements,
) -> Option<String> {
    let key = auto_session_key?;
    let model = affinity.lookup_auto_model(key)?;
    if let Some(reason) =
        cached_auto_model_reclassify_reason(node, &model, media, descriptors).await
    {
        tracing::debug!("auto: cached model {model} {reason}, reclassifying");
        affinity.forget_auto_model(key);
        return None;
    }
    tracing::debug!("auto: reusing cached model {model} for session {key:016x}");
    Some(model)
}

async fn cached_auto_model_reclassify_reason(
    node: &mesh::Node,
    model: &str,
    media: &router::MediaRequirements,
    descriptors: &[mesh::ServedModelDescriptor],
) -> Option<&'static str> {
    if cached_auto_model_missing(node, model).await {
        return Some("no longer served");
    }
    cached_auto_model_needs_reclassify(model, media, descriptors)
        .then_some("cannot satisfy media requirements")
}

async fn cached_auto_model_missing(node: &mesh::Node, model: &str) -> bool {
    node.hosts_for_model(model).await.is_empty()
}

fn cached_auto_model_needs_reclassify(
    model: &str,
    media: &router::MediaRequirements,
    descriptors: &[mesh::ServedModelDescriptor],
) -> bool {
    !cached_auto_model_satisfies_media_requirements(model, media, descriptors)
}

async fn resolve_mesh_target_hosts(
    node: &mesh::Node,
    effective_model: Option<&str>,
) -> MeshTargetResolution {
    let target_hosts = if let Some(name) = effective_model {
        node.hosts_for_model(name).await
    } else {
        Vec::new()
    };
    if !target_hosts.is_empty() {
        return MeshTargetResolution::Hosts(target_hosts);
    }
    if let Some(model) = effective_model {
        return MeshTargetResolution::ModelUnavailable(model.to_string());
    }
    match node.any_host().await {
        Some(peer) => MeshTargetResolution::Hosts(vec![peer.id]),
        None => MeshTargetResolution::NoHostsAvailable,
    }
}

async fn route_attempt_for_target(
    node: &mesh::Node,
    tcp_stream: &mut TcpStream,
    target: &election::InferenceTarget,
    prefetched: &[u8],
    retry_context_overflow: bool,
    response_adapter: ResponseAdapter,
) -> RouteAttemptResult {
    match target {
        election::InferenceTarget::Local(port) => {
            route_local_attempt(
                node,
                tcp_stream,
                *port,
                prefetched,
                retry_context_overflow,
                response_adapter,
            )
            .await
        }
        election::InferenceTarget::Remote(host_id) => {
            route_remote_attempt(
                node,
                tcp_stream,
                *host_id,
                prefetched,
                retry_context_overflow,
                response_adapter,
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
    request: &BufferedHttpRequest,
    required_tokens: Option<u32>,
    affinity: &AffinityRouter,
) -> bool {
    let args = RouteModelRequestArgs {
        node,
        tcp_stream,
        targets,
        model,
        request,
        required_tokens,
        affinity,
    };
    route_model_request_inner(args).await
}

struct RouteModelRequestArgs<'a> {
    node: mesh::Node,
    tcp_stream: TcpStream,
    targets: &'a election::ModelTargets,
    model: &'a str,
    request: &'a BufferedHttpRequest,
    required_tokens: Option<u32>,
    affinity: &'a AffinityRouter,
}

struct RouteModelState {
    route_started: Instant,
    attempts: usize,
    refreshed: bool,
}

enum RouteModelDisposition {
    Continue,
    Return(bool),
}

async fn route_model_request_inner(args: RouteModelRequestArgs<'_>) -> bool {
    let RouteModelRequestArgs {
        node,
        tcp_stream,
        targets,
        model,
        request,
        required_tokens,
        affinity,
    } = args;
    let route_started = Instant::now();
    let mut tcp_stream = tcp_stream;
    let ordered_candidates =
        order_targets_by_context(&node, model, required_tokens, &targets.candidates(model)).await;
    let ordered_candidates = affinity.route_eligible_candidates(model, &ordered_candidates);
    if ordered_candidates.is_empty() {
        record_route_model_unavailable(&node, model, 0);
        return false;
    }

    let selection = crate::network::affinity::select_model_target_from_candidates(
        targets,
        &ordered_candidates,
        model,
        request.body_json.as_ref(),
        affinity,
    );
    if matches!(selection.target, election::InferenceTarget::None) {
        return send_route_model_none_target(&node, tcp_stream, model).await;
    }
    forget_route_model_context_mismatch(&node, model, required_tokens, &selection, affinity).await;

    let mut ordered = ordered_candidates;
    move_target_first(&mut ordered, &selection.target);
    let total_targets = ordered.len();
    let mut state = RouteModelState {
        route_started,
        attempts: 0,
        refreshed: false,
    };
    for (idx, target) in ordered.into_iter().enumerate() {
        state.attempts += 1;
        let attempt_started = Instant::now();
        let retry_context_overflow = idx + 1 < total_targets;
        let attempt_result = route_attempt_for_target(
            &node,
            &mut tcp_stream,
            &target,
            &request.raw,
            retry_context_overflow,
            request.response_adapter,
        )
        .await;
        let queue_wait = attempt_started.duration_since(route_started);
        let attempt_time = attempt_started.elapsed();
        record_route_model_attempt(
            &node,
            model,
            &target,
            queue_wait,
            attempt_time,
            &attempt_result,
        );
        affinity.record_target_outcome(
            Some(model),
            &target,
            target_health_outcome_for_attempt(&attempt_result),
        );
        tracing::info!(
            model = model,
            target = ?target,
            attempt = state.attempts,
            total_targets = total_targets,
            outcome = route_attempt_result_label(&attempt_result),
            attempt_ms = attempt_started.elapsed().as_millis(),
            total_route_ms = route_started.elapsed().as_millis(),
            "openai route_model_request attempt"
        );
        match handle_route_model_attempt_result(
            &node,
            model,
            &target,
            &selection,
            attempt_result,
            &mut state,
            affinity,
        ) {
            RouteModelDisposition::Continue => continue,
            RouteModelDisposition::Return(result) => {
                return finalize_route_model_result(
                    &node,
                    model,
                    request,
                    route_started,
                    state.attempts,
                    result,
                    &target,
                )
            }
        }
    }

    finish_exhausted_route_model_request(&node, tcp_stream, model, total_targets, &state).await;
    true
}

fn record_route_model_unavailable(node: &mesh::Node, model: &str, attempts: usize) {
    node.record_routed_request(
        Some(model),
        attempts,
        crate::network::metrics::RequestOutcome::Unavailable,
    );
}

async fn send_route_model_none_target(
    node: &mesh::Node,
    tcp_stream: TcpStream,
    model: &str,
) -> bool {
    record_route_model_unavailable(node, model, 0);
    let _ = send_503(
        tcp_stream,
        &format!("target for model '{model}' resolved to None (election in progress or host down)"),
    )
    .await;
    true
}

async fn finish_exhausted_route_model_request(
    node: &mesh::Node,
    tcp_stream: TcpStream,
    model: &str,
    total_targets: usize,
    state: &RouteModelState,
) {
    let _ = send_503(
        tcp_stream,
        &format!("all {} target(s) for model '{model}' failed", total_targets),
    )
    .await;
    record_route_model_unavailable(node, model, state.attempts);
    tracing::warn!(
        model = model,
        attempts = state.attempts,
        route_ms = state.route_started.elapsed().as_millis(),
        "openai route_model_request exhausted targets"
    );
}

async fn forget_route_model_context_mismatch(
    node: &mesh::Node,
    model: &str,
    required_tokens: Option<u32>,
    selection: &TargetSelection,
    affinity: &AffinityRouter,
) {
    let (Some(prefix_hash), Some(cached_target)) = (
        selection.learn_prefix_hash,
        selection.cached_target.as_ref(),
    ) else {
        return;
    };
    let cached_context = match cached_target {
        election::InferenceTarget::Local(_) => node.local_model_context_length(model).await,
        election::InferenceTarget::Remote(peer_id) => {
            node.peer_model_context_length(*peer_id, model).await
        }
        election::InferenceTarget::None => None,
    };
    if matches!((required_tokens, cached_context), (Some(required), Some(context)) if context < required)
    {
        affinity.forget_target(model, prefix_hash, cached_target);
    }
}

fn handle_route_model_attempt_result(
    node: &mesh::Node,
    model: &str,
    target: &election::InferenceTarget,
    selection: &TargetSelection,
    attempt_result: RouteAttemptResult,
    state: &mut RouteModelState,
    affinity: &AffinityRouter,
) -> RouteModelDisposition {
    match attempt_result {
        RouteAttemptResult::Delivered { status_code, .. } => handle_delivered_route_model_attempt(
            node,
            model,
            target,
            selection,
            status_code,
            state,
            affinity,
        ),
        RouteAttemptResult::RetryableContextOverflow => {
            handle_retryable_route_model_context(model, target, selection, affinity)
        }
        RouteAttemptResult::RetryableTimeout => {
            handle_retryable_route_model_timeout(node, model, target, selection, state, affinity)
        }
        RouteAttemptResult::RetryableUnavailable => handle_retryable_route_model_unavailable(
            node, model, target, selection, state, affinity,
        ),
        RouteAttemptResult::ClientDisconnected => {
            tracing::info!(
                model = model,
                attempts = state.attempts,
                route_ms = state.route_started.elapsed().as_millis(),
                "openai route_model_request downstream disconnected"
            );
            RouteModelDisposition::Return(true)
        }
    }
}

fn handle_delivered_route_model_attempt(
    node: &mesh::Node,
    model: &str,
    target: &election::InferenceTarget,
    selection: &TargetSelection,
    status_code: u16,
    state: &RouteModelState,
    affinity: &AffinityRouter,
) -> RouteModelDisposition {
    if should_learn_affinity(status_code) {
        if let Some(prefix_hash) = selection.learn_prefix_hash {
            affinity.learn_target(model, prefix_hash, target);
        }
    }
    node.record_routed_request(
        Some(model),
        state.attempts,
        request_outcome_for_status(status_code, request_service_for_target(target)),
    );
    tracing::info!(
        model = model,
        attempts = state.attempts,
        status_code = status_code,
        route_ms = state.route_started.elapsed().as_millis(),
        "openai route_model_request delivered"
    );
    RouteModelDisposition::Return(true)
}

fn handle_retryable_route_model_context(
    model: &str,
    target: &election::InferenceTarget,
    selection: &TargetSelection,
    affinity: &AffinityRouter,
) -> RouteModelDisposition {
    forget_selected_route_model_target(model, target, selection, affinity);
    tracing::warn!(
        "Target {target:?} rejected request with context overflow-style 400, trying next"
    );
    RouteModelDisposition::Continue
}

fn handle_retryable_route_model_timeout(
    node: &mesh::Node,
    model: &str,
    target: &election::InferenceTarget,
    selection: &TargetSelection,
    state: &mut RouteModelState,
    affinity: &AffinityRouter,
) -> RouteModelDisposition {
    forget_selected_route_model_target(model, target, selection, affinity);
    spawn_mesh_refresh_once(node, &mut state.refreshed);
    tracing::warn!("Target {target:?} timed out, trying next");
    RouteModelDisposition::Continue
}

fn handle_retryable_route_model_unavailable(
    node: &mesh::Node,
    model: &str,
    target: &election::InferenceTarget,
    selection: &TargetSelection,
    state: &mut RouteModelState,
    affinity: &AffinityRouter,
) -> RouteModelDisposition {
    forget_selected_route_model_target(model, target, selection, affinity);
    spawn_mesh_refresh_once(node, &mut state.refreshed);
    tracing::warn!("Target {target:?} unavailable, trying next");
    RouteModelDisposition::Continue
}

fn forget_selected_route_model_target(
    model: &str,
    target: &election::InferenceTarget,
    selection: &TargetSelection,
    affinity: &AffinityRouter,
) {
    if let (Some(prefix_hash), Some(cached_target)) = (
        selection.learn_prefix_hash,
        selection.cached_target.as_ref(),
    ) {
        if cached_target == target {
            affinity.forget_target(model, prefix_hash, target);
        }
    }
}

fn finalize_route_model_result(
    _node: &mesh::Node,
    _model: &str,
    _request: &BufferedHttpRequest,
    _route_started: Instant,
    _attempts: usize,
    result: bool,
    _target: &election::InferenceTarget,
) -> bool {
    result
}

fn record_route_model_attempt(
    node: &mesh::Node,
    model: &str,
    target: &election::InferenceTarget,
    queue_wait: Duration,
    attempt_time: Duration,
    attempt_result: &RouteAttemptResult,
) {
    if matches!(attempt_result, RouteAttemptResult::ClientDisconnected) {
        return;
    }
    node.record_inference_attempt(
        Some(model),
        target,
        queue_wait,
        attempt_time,
        attempt_outcome_for_result(attempt_result),
        completion_tokens_for_result(attempt_result),
    );
}

/// Route a request to a known inference target (local OpenAI surface or remote host).
///
/// Used by the API proxy after election has determined the target.
pub async fn route_to_target(
    node: mesh::Node,
    tcp_stream: TcpStream,
    model: Option<&str>,
    target: election::InferenceTarget,
    prefetched: &[u8],
    response_adapter: ResponseAdapter,
) -> bool {
    let route_started = Instant::now();
    let mut tcp_stream = tcp_stream;
    tracing::info!("API proxy: routing to target {target:?}");
    let result = route_attempt_for_target(
        &node,
        &mut tcp_stream,
        &target,
        prefetched,
        false,
        response_adapter,
    )
    .await;
    node.record_inference_attempt(
        model,
        &target,
        Duration::ZERO,
        route_started.elapsed(),
        attempt_outcome_for_result(&result),
        completion_tokens_for_result(&result),
    );
    tracing::info!(
        target = ?target,
        outcome = route_attempt_result_label(&result),
        route_ms = route_started.elapsed().as_millis(),
        "openai route_to_target result"
    );
    match result {
        RouteAttemptResult::Delivered {
            status_code,
            completion_tokens: _,
        } => {
            let service = request_service_for_target(&target);
            node.record_routed_request(model, 1, request_outcome_for_status(status_code, service));
            true
        }
        RouteAttemptResult::RetryableTimeout
        | RouteAttemptResult::RetryableContextOverflow
        | RouteAttemptResult::RetryableUnavailable => {
            node.record_routed_request(
                model,
                1,
                crate::network::metrics::RequestOutcome::Unavailable,
            );
            let _ = send_503(
                tcp_stream,
                &format!("single target {target:?} unavailable (route_to_target)"),
            )
            .await;
            false
        }
        RouteAttemptResult::ClientDisconnected => true,
    }
}

pub async fn route_http_endpoint_request(
    node: &mesh::Node,
    model: Option<&str>,
    tcp_stream: &mut TcpStream,
    base_url: &str,
    prefetched: &[u8],
    request_path: &str,
    response_adapter: ResponseAdapter,
) -> bool {
    let started = Instant::now();
    let result = route_http_endpoint_attempt(
        tcp_stream,
        base_url,
        prefetched,
        request_path,
        false,
        response_adapter,
    )
    .await;
    node.record_endpoint_attempt(
        model,
        base_url,
        Duration::ZERO,
        started.elapsed(),
        attempt_outcome_for_result(&result),
        completion_tokens_for_result(&result),
    );
    tracing::info!(
        endpoint = base_url,
        path = request_path,
        outcome = route_attempt_result_label(&result),
        route_ms = started.elapsed().as_millis(),
        "openai route_http_endpoint_request result"
    );
    match result {
        RouteAttemptResult::Delivered {
            status_code,
            completion_tokens: _,
        } => {
            node.record_routed_request(
                model,
                1,
                request_outcome_for_status(
                    status_code,
                    crate::network::metrics::RequestService::Endpoint,
                ),
            );
            true
        }
        RouteAttemptResult::RetryableTimeout
        | RouteAttemptResult::RetryableContextOverflow
        | RouteAttemptResult::RetryableUnavailable => {
            node.record_routed_request(
                model,
                1,
                crate::network::metrics::RequestOutcome::Unavailable,
            );
            false
        }
        RouteAttemptResult::ClientDisconnected => true,
    }
}

// ── Response helpers ──

pub async fn send_models_list_with_descriptors(
    mut stream: TcpStream,
    models: &[String],
    descriptors: &[mesh::ServedModelDescriptor],
) -> std::io::Result<()> {
    let body = models_list_json(models, descriptors).to_string();
    let resp = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nAccess-Control-Allow-Origin: *\r\n\r\n{}",
        body.len(), body
    );
    stream.write_all(resp.as_bytes()).await?;
    stream.shutdown().await?;
    Ok(())
}

fn models_list_json(
    models: &[String],
    descriptors: &[mesh::ServedModelDescriptor],
) -> serde_json::Value {
    let mut seen = std::collections::HashSet::new();
    let data: Vec<serde_json::Value> = models
        .iter()
        .filter_map(|m| {
            let descriptor = descriptor_for_model(descriptors, m);
            let public_id = public_model_id(m, descriptor);
            if !seen.insert(public_id.clone()) {
                return None;
            }
            let capabilities = capabilities_for_model(m, descriptors);
            let has_multimodal = capabilities.supports_multimodal_runtime();
            let has_vision = capabilities.supports_vision_runtime();
            let has_audio = capabilities.supports_audio_runtime();
            let mut caps = vec!["text"];
            if has_multimodal {
                caps.push("multimodal");
            }
            if has_vision {
                caps.push("vision");
            }
            if has_audio {
                caps.push("audio");
            }
            if capabilities.reasoning_label().is_some() {
                caps.push("reasoning");
            }
            let display_name = if public_id == *m {
                crate::models::installed_model_display_name(m)
            } else {
                public_id.clone()
            };
            Some(serde_json::json!({
                "id": public_id,
                "display_name": display_name,
                "object": "model",
                "owned_by": "mesh-llm",
                "capabilities": caps,
                "multimodal_status": capabilities.multimodal_status(),
                "vision_status": capabilities.vision_status(),
                "audio_status": capabilities.audio_status(),
                "reasoning_status": capabilities.reasoning_status(),
            }))
        })
        .collect();

    serde_json::json!({ "object": "list", "data": data })
}

pub fn rewrite_public_model_alias(
    request: &mut BufferedHttpRequest,
    models: &[String],
    descriptors: &[mesh::ServedModelDescriptor],
) {
    let Some(requested) = request.model_name.as_deref() else {
        return;
    };
    if requested == "auto" || models.iter().any(|model| model == requested) {
        return;
    }
    let Some(internal) = internal_model_for_public_id(requested, models, descriptors) else {
        return;
    };
    rewrite_model_field(request, &internal);
}

fn internal_model_for_public_id(
    requested: &str,
    models: &[String],
    descriptors: &[mesh::ServedModelDescriptor],
) -> Option<String> {
    models.iter().find_map(|model| {
        let descriptor = descriptor_for_model(descriptors, model);
        (public_model_id(model, descriptor) == requested).then(|| model.clone())
    })
}

fn descriptor_for_model<'a>(
    descriptors: &'a [mesh::ServedModelDescriptor],
    model_name: &str,
) -> Option<&'a mesh::ServedModelDescriptor> {
    descriptors
        .iter()
        .find(|descriptor| descriptor.identity.model_name == model_name)
}

fn public_model_id(model_name: &str, descriptor: Option<&mesh::ServedModelDescriptor>) -> String {
    // A descriptor with an `artifact` field has enough information to
    // produce a public ID that round-trips to the same model. Without
    // it, the HuggingFace path collapses to just the repo name and
    // silently drops the quant-tag suffix the resolver needs (PR #566
    // review feedback — "some IDs in /v1/models dropped quant
    // suffixes"). Only use the descriptor-derived id when it can be
    // lossless; otherwise prefer the on-disk file (authoritative for
    // local models), and finally the internal model_name (which
    // always carries the quant suffix our resolver knows how to
    // route).
    if let Some(descriptor) = descriptor {
        if descriptor_can_produce_lossless_id(&descriptor.identity) {
            if let Some(id) = public_model_id_from_identity(&descriptor.identity) {
                return id;
            }
        }
    }

    if let Some(id) = public_model_id_from_local_path(model_name) {
        return id;
    }

    model_name.to_string()
}

/// A descriptor identity carries enough information for
/// `public_model_id_from_identity` to produce an ID that round-trips
/// to the same model. For HuggingFace that means the `artifact` field
/// (the GGUF file name) is present so the quant selector can be
/// derived. Catalog identities always carry a `canonical_ref` with the
/// selector baked in.
fn descriptor_can_produce_lossless_id(identity: &mesh::ServedModelIdentity) -> bool {
    match identity.source_kind {
        mesh::ModelSourceKind::HuggingFace => identity.artifact.is_some(),
        mesh::ModelSourceKind::Catalog => identity.canonical_ref.is_some(),
        mesh::ModelSourceKind::LocalGguf
        | mesh::ModelSourceKind::DirectUrl
        | mesh::ModelSourceKind::Unknown => false,
    }
}

fn public_model_id_from_identity(identity: &mesh::ServedModelIdentity) -> Option<String> {
    match identity.source_kind {
        mesh::ModelSourceKind::HuggingFace => identity
            .repository
            .as_deref()
            .and_then(|repo| public_huggingface_model_ref(repo, identity.artifact.as_deref()))
            .or_else(|| {
                identity
                    .canonical_ref
                    .as_deref()
                    .and_then(|model_ref| model_ref::ModelRef::parse(model_ref).ok())
                    .map(|model_ref| model_ref.display_id())
            }),
        mesh::ModelSourceKind::Catalog => identity
            .canonical_ref
            .as_deref()
            .and_then(|model_ref| model_ref::ModelRef::parse(model_ref).ok())
            .map(|model_ref| model_ref.display_id()),
        mesh::ModelSourceKind::LocalGguf
        | mesh::ModelSourceKind::DirectUrl
        | mesh::ModelSourceKind::Unknown => None,
    }
}

fn public_model_id_from_local_path(model_name: &str) -> Option<String> {
    let path = crate::models::find_model_path(model_name);
    if !path.is_file() {
        return None;
    }
    if path.extension().and_then(|extension| extension.to_str()) != Some("gguf") {
        return None;
    }
    Some(crate::models::model_ref_for_path(&path))
}

fn public_huggingface_model_ref(repo: &str, artifact: Option<&str>) -> Option<String> {
    // `artifact` can be either a GGUF filename (e.g. `Falcon-Q4_K_M.gguf`)
    // or an already-extracted quant selector (e.g. `Q4_K_M` or
    // `qwen2.5-3b-instruct-q4_k_m`, when the descriptor was built from
    // a parsed `ModelRef::selector`). Handle both — if the artifact
    // looks like a quant selector use it directly; otherwise try to
    // pull a selector out of the filename.
    let selector = artifact.and_then(|a| {
        model_ref::quant_selector_from_gguf_file(a)
            .or_else(|| (!a.is_empty() && !a.ends_with(".gguf")).then(|| a.to_string()))
    });
    Some(model_ref::format_model_ref(repo, None, selector.as_deref()))
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

/// RFC 7230 tchar set for header field names: ASCII alphanumeric plus
/// `!#$%&'*+-.^_`|~`. We additionally forbid `:` because it terminates
/// the field-name in the wire grammar. Used to reject caller-provided
/// header names that could carry CR/LF or other injection bytes.
pub(crate) fn is_valid_header_name(name: &str) -> bool {
    !name.is_empty()
        && name.bytes().all(|b| {
            b.is_ascii_alphanumeric()
                || matches!(
                    b,
                    b'!' | b'#'
                        | b'$'
                        | b'%'
                        | b'&'
                        | b'\''
                        | b'*'
                        | b'+'
                        | b'-'
                        | b'.'
                        | b'^'
                        | b'_'
                        | b'`'
                        | b'|'
                        | b'~'
                )
        })
}

/// Append a single `name: value` header line if `name` is a valid HTTP
/// header field name. CR/LF in `value` is stripped defensively. Used by
/// the `*_with_headers` writers below so a malformed header from a
/// future caller can't inject extra headers / smuggle a response.
pub(crate) fn append_safe_header(headers: &mut String, name: &str, value: &str) {
    if !is_valid_header_name(name) {
        tracing::warn!(
            "openai transport: dropping header with invalid name `{name}` (RFC 7230 tchar required)"
        );
        return;
    }
    let safe_value: String = value.chars().filter(|c| *c != '\r' && *c != '\n').collect();
    headers.push_str(name);
    headers.push_str(": ");
    headers.push_str(&safe_value);
    headers.push_str("\r\n");
}

/// Like `send_json_ok` but allows the caller to append arbitrary response
/// headers (e.g. `x-moa-*` observability headers).
///
/// Header names must satisfy the RFC 7230 tchar grammar (ASCII
/// alphanumeric + a small symbol set); invalid names are dropped with a
/// warning rather than written verbatim. Values are stripped of CR/LF.
pub async fn send_json_ok_with_headers(
    mut stream: TcpStream,
    data: &serde_json::Value,
    extra_headers: &[(&str, String)],
) -> std::io::Result<()> {
    let body = data.to_string();
    let mut headers = String::from("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n");
    for (name, value) in extra_headers {
        append_safe_header(&mut headers, name, value);
    }
    headers.push_str(&format!("Content-Length: {}\r\n\r\n", body.len()));
    stream.write_all(headers.as_bytes()).await?;
    stream.write_all(body.as_bytes()).await?;
    stream.shutdown().await?;
    Ok(())
}

/// Send a JSON body with a non-200 status and the given extra headers.
///
/// The body is sent verbatim — caller controls the shape. Use for cases
/// where the in-band payload is already a structured error (e.g. MoA's
/// `error_response`) and we still want to attach observability headers
/// while signalling failure via the HTTP status line.
pub async fn send_json_with_status_and_headers(
    mut stream: TcpStream,
    code: u16,
    data: &serde_json::Value,
    extra_headers: &[(&str, String)],
) -> std::io::Result<()> {
    let status = match code {
        400 => "Bad Request",
        404 => "Not Found",
        409 => "Conflict",
        422 => "Unprocessable Content",
        429 => "Too Many Requests",
        500 => "Internal Server Error",
        502 => "Bad Gateway",
        503 => "Service Unavailable",
        504 => "Gateway Timeout",
        _ => "Error",
    };
    let body = data.to_string();
    let mut headers = format!("HTTP/1.1 {code} {status}\r\nContent-Type: application/json\r\n");
    for (name, value) in extra_headers {
        append_safe_header(&mut headers, name, value);
    }
    headers.push_str(&format!("Content-Length: {}\r\n\r\n", body.len()));
    stream.write_all(headers.as_bytes()).await?;
    stream.write_all(body.as_bytes()).await?;
    stream.shutdown().await?;
    Ok(())
}

pub async fn send_400(mut stream: TcpStream, msg: &str) -> std::io::Result<()> {
    let body = serde_json::to_vec(&serde_json::json!({ "error": msg }))
        .expect("serializing JSON error response should not fail");
    let headers = format!(
        "HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
        body.len()
    );
    stream.write_all(headers.as_bytes()).await?;
    stream.write_all(&body).await?;
    stream.shutdown().await?;
    Ok(())
}

pub async fn send_error(mut stream: TcpStream, code: u16, msg: &str) -> std::io::Result<()> {
    let status = match code {
        404 => "Not Found",
        409 => "Conflict",
        422 => "Unprocessable Content",
        429 => "Too Many Requests",
        _ => "Bad Request",
    };
    let body = serde_json::json!({"error": msg}).to_string();
    let retry_after = if code == 429 {
        "Retry-After: 5\r\n"
    } else {
        ""
    };
    let resp = format!(
        "HTTP/1.1 {code} {status}\r\nContent-Type: application/json\r\n{retry_after}Content-Length: {}\r\n\r\n{}",
        body.len(), body
    );
    stream.write_all(resp.as_bytes()).await?;
    stream.shutdown().await?;
    Ok(())
}

pub async fn send_503(stream: TcpStream, reason: &str) -> std::io::Result<()> {
    tracing::warn!("503 → client: {reason}");
    send_503_inner(stream, reason).await
}

async fn send_503_inner(mut stream: TcpStream, reason: &str) -> std::io::Result<()> {
    let body = serde_json::json!({"error": reason}).to_string();
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

    let http_client = reqwest::Client::new();
    let planner_url = format!("http://127.0.0.1:{planner_port}");
    if !pipeline_preplan_request(&http_client, &planner_url, planner_model, &mut body).await {
        return PipelineProxyResult::FallbackToDirect;
    }

    let strong_url = format!("http://127.0.0.1:{strong_port}/v1/chat/completions");
    let _inflight = node.begin_inflight_request();
    let is_streaming = pipeline_streaming_requested(&body);
    if is_streaming {
        pipeline_proxy_streaming(client_stream, &http_client, &strong_url, &body).await
    } else {
        pipeline_proxy_non_streaming(client_stream, &http_client, &strong_url, &body).await
    }
}

fn pipeline_streaming_requested(body: &serde_json::Value) -> bool {
    body.get("stream")
        .and_then(|value| value.as_bool())
        .unwrap_or(false)
}

async fn pipeline_preplan_request(
    http_client: &reqwest::Client,
    planner_url: &str,
    planner_model: &str,
    body: &mut serde_json::Value,
) -> bool {
    let messages = body
        .get("messages")
        .and_then(|messages| messages.as_array())
        .cloned()
        .unwrap_or_default();
    match crate::inference::pipeline::pre_plan(http_client, planner_url, planner_model, &messages)
        .await
    {
        Ok(plan) => {
            tracing::info!(
                "pipeline: pre-plan by {} in {}ms — {}",
                plan.model_used,
                plan.elapsed_ms,
                plan.plan_text.chars().take(200).collect::<String>()
            );
            crate::inference::pipeline::inject_plan(body, &plan);
            true
        }
        Err(err) => {
            tracing::warn!("pipeline: pre-plan failed ({err}), falling back to direct proxy");
            false
        }
    }
}

async fn pipeline_proxy_streaming(
    client_stream: &mut TcpStream,
    http_client: &reqwest::Client,
    strong_url: &str,
    body: &serde_json::Value,
) -> PipelineProxyResult {
    match http_client.post(strong_url).json(body).send().await {
        Ok(resp) => relay_pipeline_streaming_response(client_stream, resp).await,
        Err(err) => {
            tracing::warn!(
                "pipeline: strong model request failed: {err}, falling back to direct proxy"
            );
            PipelineProxyResult::FallbackToDirect
        }
    }
}

async fn relay_pipeline_streaming_response(
    client_stream: &mut TcpStream,
    resp: reqwest::Response,
) -> PipelineProxyResult {
    let status = resp.status();
    let content_type = resp
        .headers()
        .get("content-type")
        .and_then(|value| value.to_str().ok())
        .unwrap_or("text/event-stream")
        .to_string();
    let header = format!(
        "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nTransfer-Encoding: chunked\r\nCache-Control: no-cache\r\n\r\n",
    );
    if client_stream.write_all(header.as_bytes()).await.is_err() {
        return PipelineProxyResult::Handled;
    }

    use tokio_stream::StreamExt;
    let mut stream = resp.bytes_stream();
    while let Some(chunk) = stream.next().await {
        match chunk {
            Ok(bytes) if write_pipeline_chunk(client_stream, &bytes).await.is_err() => break,
            Ok(_) => {}
            Err(err) => {
                tracing::debug!("pipeline: stream error: {err}");
                break;
            }
        }
    }
    let _ = client_stream.write_all(b"0\r\n\r\n").await;
    let _ = client_stream.shutdown().await;
    PipelineProxyResult::Handled
}

async fn write_pipeline_chunk(client_stream: &mut TcpStream, bytes: &[u8]) -> std::io::Result<()> {
    let chunk_header = format!("{:x}\r\n", bytes.len());
    client_stream.write_all(chunk_header.as_bytes()).await?;
    client_stream.write_all(bytes).await?;
    client_stream.write_all(b"\r\n").await
}

async fn pipeline_proxy_non_streaming(
    client_stream: &mut TcpStream,
    http_client: &reqwest::Client,
    strong_url: &str,
    body: &serde_json::Value,
) -> PipelineProxyResult {
    match http_client.post(strong_url).json(body).send().await {
        Ok(resp) => relay_pipeline_non_streaming_response(client_stream, resp).await,
        Err(err) => {
            tracing::warn!(
                "pipeline: strong model request failed: {err}, falling back to direct proxy"
            );
            PipelineProxyResult::FallbackToDirect
        }
    }
}

async fn relay_pipeline_non_streaming_response(
    client_stream: &mut TcpStream,
    resp: reqwest::Response,
) -> PipelineProxyResult {
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
        Err(err) => {
            tracing::warn!("pipeline: response read failed: {err}, falling back to direct proxy");
            PipelineProxyResult::FallbackToDirect
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::net::TcpListener;

    // ── Header-name validation ──────────────────────────────────────

    #[test]
    fn is_valid_header_name_accepts_normal_observability_headers() {
        assert!(is_valid_header_name("x-moa-elapsed-ms"));
        assert!(is_valid_header_name("X-MoA-Workers"));
        assert!(is_valid_header_name("Content-Type"));
        assert!(is_valid_header_name("x-request-id"));
    }

    #[test]
    fn is_valid_header_name_rejects_injection_attempts() {
        // Regression for PR #566 review item #5c: header NAMES were not
        // sanitized, only values. A name carrying CR/LF or a colon would
        // smuggle extra headers / split the response.
        assert!(!is_valid_header_name("x-evil\r\nSet-Cookie"));
        assert!(!is_valid_header_name("x-evil\nSet-Cookie"));
        assert!(!is_valid_header_name("x-evil: hijacked"));
        assert!(!is_valid_header_name("x evil")); // space inside name
        assert!(!is_valid_header_name(""));
    }

    #[test]
    fn append_safe_header_drops_invalid_name() {
        let mut buf = String::new();
        append_safe_header(&mut buf, "x-evil\r\nSet-Cookie", "bad");
        assert!(buf.is_empty(), "invalid name must be dropped, got {buf:?}");
    }

    #[test]
    fn append_safe_header_strips_crlf_from_value() {
        let mut buf = String::new();
        append_safe_header(&mut buf, "x-ok", "ok\r\nSet-Cookie: hijack");
        assert!(
            buf.starts_with("x-ok: okSet-Cookie: hijack\r\n"),
            "value CRLF must be stripped; got {buf:?}"
        );
        assert_eq!(buf.matches("\r\n").count(), 1);
    }

    fn hf_descriptor(model_name: &str) -> mesh::ServedModelDescriptor {
        mesh::ServedModelDescriptor {
            identity: mesh::ServedModelIdentity {
                model_name: model_name.to_string(),
                source_kind: mesh::ModelSourceKind::HuggingFace,
                repository: Some("tiiuae/Falcon-H1-1.5B-Instruct-GGUF".to_string()),
                revision: Some("0d3a6cfe25fb4eeab0153fb8623aac5b69d6bd0a".to_string()),
                artifact: Some("Falcon-H1-1.5B-Instruct-Q4_K_M.gguf".to_string()),
                canonical_ref: Some(
                    "tiiuae/Falcon-H1-1.5B-Instruct-GGUF@0d3a6cfe25fb4eeab0153fb8623aac5b69d6bd0a/Falcon-H1-1.5B-Instruct-Q4_K_M.gguf"
                        .to_string(),
                ),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    fn catalog_model_ref_descriptor(model_name: &str) -> mesh::ServedModelDescriptor {
        mesh::ServedModelDescriptor {
            identity: mesh::ServedModelIdentity {
                model_name: model_name.to_string(),
                source_kind: mesh::ModelSourceKind::Catalog,
                canonical_ref: Some("tiiuae/Falcon-H1-1.5B-Instruct-GGUF:Q4_K_M".to_string()),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    fn local_gguf_descriptor(model_name: &str) -> mesh::ServedModelDescriptor {
        mesh::ServedModelDescriptor {
            identity: mesh::ServedModelIdentity {
                model_name: model_name.to_string(),
                source_kind: mesh::ModelSourceKind::LocalGguf,
                local_file_name: Some(format!("{model_name}.gguf")),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    fn local_gguf_descriptor_with_capabilities(
        model_name: &str,
        capabilities: crate::models::ModelCapabilities,
    ) -> mesh::ServedModelDescriptor {
        mesh::ServedModelDescriptor {
            capabilities_known: true,
            capabilities,
            ..local_gguf_descriptor(model_name)
        }
    }

    async fn read_request_from_parts_with_limits(
        parts: Vec<Vec<u8>>,
        limits: HttpReadLimits,
    ) -> BufferedHttpRequest {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            read_http_request_with_limits(&mut stream, limits, None)
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

    #[test]
    fn models_list_uses_public_huggingface_model_ref_ids() {
        let models = vec!["Falcon-H1-1.5B-Instruct-Q4_K_M".to_string()];
        let descriptors = vec![hf_descriptor(&models[0])];

        let body = models_list_json(&models, &descriptors);

        assert_eq!(
            body["data"][0]["id"],
            "tiiuae/Falcon-H1-1.5B-Instruct-GGUF:Q4_K_M"
        );
        assert_eq!(
            body["data"][0]["display_name"],
            "tiiuae/Falcon-H1-1.5B-Instruct-GGUF:Q4_K_M"
        );
        assert_eq!(body["data"][0]["owned_by"], "mesh-llm");
    }

    #[test]
    fn models_list_id_preserves_quant_suffix_when_descriptor_has_no_artifact() {
        // Regression for PR #566 review feedback: the gateway's view of a
        // model's public ID must include enough information to route a
        // request back to that exact model. When a `ServedModelDescriptor`
        // for a HuggingFace model has no `artifact` field (because the
        // descriptor was built without inspecting the GGUF file on disk),
        // `public_huggingface_model_ref` collapses the public ID to just
        // the repo name — dropping the quant-tag suffix the internal
        // `model_name` carries. The model is then advertised in `/v1/models`
        // under a shorter ID than the resolver knows how to route.
        //
        // Symptom on a real 2-node mesh: the studio's Qwen3-0.6B-GGUF
        // shows as `unsloth/Qwen3-0.6B-GGUF:BF16` (descriptor has
        // artifact), but the gateway-local Qwen2.5-3B-Instruct-GGUF
        // shows as `Qwen/Qwen2.5-3B-Instruct-GGUF` (descriptor has no
        // artifact). A client doing the natural thing — read /v1/models,
        // call /v1/chat/completions with the listed id — then 404s on
        // remote models because the resolver doesn't know the short id.
        //
        // Acceptable behaviour: the public ID either round-trips to the
        // same model, OR includes the quant suffix the internal name
        // carries.
        let models = vec!["Qwen/Qwen2.5-3B-Instruct-GGUF:qwen2.5-3b-instruct-q4_k_m".to_string()];
        let descriptor = mesh::ServedModelDescriptor {
            identity: mesh::ServedModelIdentity {
                model_name: models[0].clone(),
                source_kind: mesh::ModelSourceKind::HuggingFace,
                repository: Some("Qwen/Qwen2.5-3B-Instruct-GGUF".to_string()),
                // No artifact — this is the field whose absence loses the
                // quant suffix.
                artifact: None,
                ..Default::default()
            },
            ..Default::default()
        };
        let descriptors = vec![descriptor];

        let body = models_list_json(&models, &descriptors);
        let public_id = body["data"][0]["id"].as_str().unwrap_or_default();

        // The public ID must NOT silently drop the quant suffix that the
        // internal model_name carries. Acceptable IDs:
        //   * the full internal name, OR
        //   * the repo with a quant tag we can route back to.
        assert!(
            public_id == models[0]
                || public_id
                    .strip_prefix("Qwen/Qwen2.5-3B-Instruct-GGUF:")
                    .is_some_and(|tag| !tag.is_empty()),
            "public id must keep enough information to route back; got {public_id:?}, \
             internal model_name was {:?}",
            models[0]
        );
    }

    #[test]
    fn models_list_uses_catalog_model_ref_ids() {
        let models = vec!["Falcon-H1-1.5B-Instruct-Q4_K_M".to_string()];
        let descriptors = vec![catalog_model_ref_descriptor(&models[0])];

        let body = models_list_json(&models, &descriptors);

        assert_eq!(
            body["data"][0]["id"],
            "tiiuae/Falcon-H1-1.5B-Instruct-GGUF:Q4_K_M"
        );
    }

    #[test]
    fn models_list_keeps_local_gguf_model_name_ids() {
        let models = vec!["smollm2-a".to_string()];
        let descriptors = vec![local_gguf_descriptor(&models[0])];

        let body = models_list_json(&models, &descriptors);

        assert_eq!(body["data"][0]["id"], "smollm2-a");
        assert_eq!(body["data"][0]["display_name"], "smollm2-a");
    }

    #[test]
    fn models_list_uses_descriptor_capabilities_not_filename_heuristics() {
        let models = vec!["Qwen3VL-2B-Instruct-Q4_K_M".to_string()];
        let descriptors = vec![local_gguf_descriptor_with_capabilities(
            &models[0],
            crate::models::ModelCapabilities::default(),
        )];

        let body = models_list_json(&models, &descriptors);

        assert_eq!(body["data"][0]["capabilities"], serde_json::json!(["text"]));
        assert_eq!(body["data"][0]["vision_status"], "none");
        assert_eq!(body["data"][0]["multimodal_status"], "none");
    }

    #[test]
    fn models_list_uses_static_fallback_for_unknown_descriptor_capabilities() {
        let models = vec!["Qwen3VL-2B-Instruct-Q4_K_M".to_string()];
        let descriptors = vec![local_gguf_descriptor(&models[0])];

        let body = models_list_json(&models, &descriptors);
        let capabilities = body["data"][0]["capabilities"].as_array().unwrap();

        assert!(capabilities.iter().any(|cap| cap == "multimodal"));
        assert!(capabilities.iter().any(|cap| cap == "vision"));
        assert_eq!(body["data"][0]["vision_status"], "supported");
        assert_eq!(body["data"][0]["multimodal_status"], "supported");
    }

    #[test]
    fn models_list_reports_runtime_verified_projector_capabilities() {
        let models = vec!["Qwen3VL-2B-Instruct-Q4_K_M".to_string()];
        let descriptors = vec![local_gguf_descriptor_with_capabilities(
            &models[0],
            crate::models::ModelCapabilities {
                multimodal: true,
                vision: crate::models::CapabilityLevel::Supported,
                ..Default::default()
            },
        )];

        let body = models_list_json(&models, &descriptors);
        let capabilities = body["data"][0]["capabilities"].as_array().unwrap();

        assert!(capabilities.iter().any(|cap| cap == "multimodal"));
        assert!(capabilities.iter().any(|cap| cap == "vision"));
        assert_eq!(body["data"][0]["vision_status"], "supported");
        assert_eq!(body["data"][0]["multimodal_status"], "supported");
    }

    #[test]
    fn public_model_alias_rewrites_request_to_internal_model_name() {
        let models = vec!["Falcon-H1-1.5B-Instruct-Q4_K_M".to_string()];
        let descriptors = vec![catalog_model_ref_descriptor(&models[0])];
        let body = serde_json::json!({
            "model": "tiiuae/Falcon-H1-1.5B-Instruct-GGUF:Q4_K_M",
            "messages": [{"role": "user", "content": "hello"}]
        });
        let body_bytes = serde_json::to_vec(&body).unwrap();
        let mut raw = format!(
            "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Length: {}\r\n\r\n",
            body_bytes.len()
        )
        .into_bytes();
        raw.extend_from_slice(&body_bytes);
        let mut request = BufferedHttpRequest {
            raw,
            method: "POST".to_string(),
            path: "/v1/chat/completions".to_string(),
            body_json: Some(body),
            body_json_attempted: true,
            body_bytes: Some(body_bytes),
            body_len_bytes: 0,
            completion_tokens: None,
            model_name: Some("tiiuae/Falcon-H1-1.5B-Instruct-GGUF:Q4_K_M".to_string()),
            request_object_request_ids: Vec::new(),
            response_adapter: ResponseAdapter::None,
        };

        rewrite_public_model_alias(&mut request, &models, &descriptors);

        assert_eq!(request.model_name.as_deref(), Some(models[0].as_str()));
        assert_eq!(request.body_json.as_ref().unwrap()["model"], models[0]);
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
    fn test_route_attempt_result_label_values() {
        assert_eq!(
            route_attempt_result_label(&RouteAttemptResult::Delivered {
                status_code: 200,
                completion_tokens: None,
            }),
            "delivered"
        );
        assert_eq!(
            route_attempt_result_label(&RouteAttemptResult::RetryableTimeout),
            "retryable_timeout"
        );
        assert_eq!(
            route_attempt_result_label(&RouteAttemptResult::RetryableUnavailable),
            "retryable_unavailable"
        );
        assert_eq!(
            route_attempt_result_label(&RouteAttemptResult::RetryableContextOverflow),
            "retryable_context_overflow"
        );
        assert_eq!(
            route_attempt_result_label(&RouteAttemptResult::ClientDisconnected),
            "client_disconnected"
        );
    }

    #[test]
    fn test_target_health_outcome_for_attempt_values() {
        assert_eq!(
            target_health_outcome_for_attempt(&RouteAttemptResult::Delivered {
                status_code: 200,
                completion_tokens: None,
            }),
            TargetHealthOutcome::Success
        );
        assert_eq!(
            target_health_outcome_for_attempt(&RouteAttemptResult::Delivered {
                status_code: 503,
                completion_tokens: None,
            }),
            TargetHealthOutcome::Unavailable
        );
        assert_eq!(
            target_health_outcome_for_attempt(&RouteAttemptResult::Delivered {
                status_code: 400,
                completion_tokens: None,
            }),
            TargetHealthOutcome::Rejected
        );
        assert_eq!(
            target_health_outcome_for_attempt(&RouteAttemptResult::RetryableContextOverflow),
            TargetHealthOutcome::ContextOverflow
        );
        assert_eq!(
            target_health_outcome_for_attempt(&RouteAttemptResult::RetryableTimeout),
            TargetHealthOutcome::Timeout
        );
    }

    #[test]
    fn test_cached_auto_model_rejects_text_model_for_image_request() {
        let body = serde_json::json!({
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}
                ]
            }]
        });
        let media = router::media_requirements(&body);

        assert!(!cached_auto_model_satisfies_media_requirements(
            "Qwen3-8B-Q4_K_M",
            &media,
            &[]
        ));
        assert!(cached_auto_model_satisfies_media_requirements(
            "Qwen3.5-0.8B-Vision-Q4_K_M",
            &media,
            &[]
        ));
    }

    #[test]
    fn cached_auto_model_rejects_descriptor_text_only_even_when_name_looks_vision() {
        let body = serde_json::json!({
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}
                ]
            }]
        });
        let media = router::media_requirements(&body);
        let model = "Qwen3VL-2B-Instruct-Q4_K_M";
        let descriptors = vec![local_gguf_descriptor_with_capabilities(
            model,
            crate::models::ModelCapabilities::default(),
        )];

        assert!(!cached_auto_model_satisfies_media_requirements(
            model,
            &media,
            &descriptors
        ));
    }

    #[test]
    fn cached_auto_model_uses_static_fallback_for_unknown_descriptor_capabilities() {
        let body = serde_json::json!({
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}
                ]
            }]
        });
        let media = router::media_requirements(&body);
        let model = "Qwen3VL-2B-Instruct-Q4_K_M";
        let descriptors = vec![local_gguf_descriptor(model)];

        assert!(cached_auto_model_satisfies_media_requirements(
            model,
            &media,
            &descriptors
        ));
    }

    #[test]
    fn test_parse_completion_tokens_from_json_body_supports_chat_and_responses_shapes() {
        let chat = serde_json::json!({
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}
        });
        let responses = serde_json::json!({
            "usage": {"input_tokens": 5, "output_tokens": 4, "total_tokens": 9}
        });

        assert_eq!(
            parse_completion_tokens_from_json_body(chat.to_string().as_bytes()),
            Some(3)
        );
        assert_eq!(
            parse_completion_tokens_from_json_body(responses.to_string().as_bytes()),
            Some(4)
        );
    }

    #[test]
    fn chat_stream_normalizer_adds_missing_tool_call_id() {
        let mut state = ChatStreamNormalizationState {
            synthetic_seed: Some(42),
            ..Default::default()
        };
        let normalized = state.normalize_data(
            r#"{"id":"chatcmpl-a","object":"chat.completion.chunk","created":1,"model":"test","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"type":"function","function":{"name":"read_file","arguments":"{\"path\":\"AGENTS.md\"}"}}]},"finish_reason":null}]}"#,
        );
        let parsed: serde_json::Value = serde_json::from_str(&normalized).unwrap();

        assert_eq!(parsed["id"], "chatcmpl-a");
        assert_eq!(
            parsed["choices"][0]["delta"]["tool_calls"][0]["id"],
            "call_mesh_42_0"
        );
    }

    #[test]
    fn chat_stream_normalizer_keeps_completion_and_tool_ids_stable() {
        let mut state = ChatStreamNormalizationState {
            synthetic_seed: Some(7),
            ..Default::default()
        };

        let first = state.normalize_data(
            r#"{"id":"chatcmpl-first","object":"chat.completion.chunk","created":1,"model":"test","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}"#,
        );
        let second = state.normalize_data(
            r#"{"id":"chatcmpl-second","object":"chat.completion.chunk","created":2,"model":"test","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"type":"function","function":{"arguments":"{}"}}]},"finish_reason":null}]}"#,
        );
        let third = state.normalize_data(
            r#"{"id":"chatcmpl-third","object":"chat.completion.chunk","created":3,"model":"test","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"type":"function","function":{"arguments":" more"}}]},"finish_reason":null}]}"#,
        );
        let first: serde_json::Value = serde_json::from_str(&first).unwrap();
        let second: serde_json::Value = serde_json::from_str(&second).unwrap();
        let third: serde_json::Value = serde_json::from_str(&third).unwrap();

        assert_eq!(first["id"], "chatcmpl-first");
        assert_eq!(second["id"], "chatcmpl-first");
        assert_eq!(third["id"], "chatcmpl-first");
        assert_eq!(
            second["choices"][0]["delta"]["tool_calls"][0]["id"],
            "call_mesh_7_0"
        );
        assert_eq!(
            third["choices"][0]["delta"]["tool_calls"][0]["id"],
            "call_mesh_7_0"
        );
    }

    #[tokio::test]
    async fn test_is_timeout_error_accepts_concrete_timeout_types_only() {
        let io_timeout = anyhow::Error::from(std::io::Error::new(
            std::io::ErrorKind::TimedOut,
            "socket timed out",
        ));
        let elapsed_timeout = anyhow::Error::from(
            tokio::time::timeout(Duration::from_millis(1), std::future::pending::<()>())
                .await
                .unwrap_err(),
        );
        let generic_timeout_text = anyhow::anyhow!("context timeout budget exceeded");

        assert!(is_timeout_error(&io_timeout));
        assert!(is_timeout_error(&elapsed_timeout));
        assert!(!is_timeout_error(&generic_timeout_text));
    }

    #[test]
    fn test_normalize_openai_compat_request_translates_responses_input() {
        let mut body = serde_json::json!({
            "model": "test",
            "instructions": "be concise",
            "max_output_tokens": 256,
            "input": [{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "describe this"},
                    {"type": "input_image", "image_url": "mesh://blob/client-1/token-1"},
                    {"type": "input_audio", "audio_url": "mesh://blob/client-1/token-2"}
                ]
            }]
        });

        let normalization = normalize_openai_compat_request("/v1/responses", &mut body).unwrap();

        assert!(normalization.changed);
        assert_eq!(
            normalization.rewritten_path.as_deref(),
            Some("/v1/chat/completions")
        );
        assert_eq!(
            normalization.response_adapter,
            ResponseAdapter::OpenAiResponsesJson
        );
        assert_eq!(body["max_tokens"], 256);
        assert!(body.get("max_output_tokens").is_none());
        assert_eq!(body["messages"][0]["role"], "system");
        assert_eq!(body["messages"][0]["content"], "be concise");
        assert_eq!(body["messages"][1]["role"], "user");
        assert_eq!(body["messages"][1]["content"][0]["type"], "text");
        assert_eq!(body["messages"][1]["content"][1]["type"], "image_url");
        assert_eq!(
            body["messages"][1]["content"][1]["image_url"]["url"],
            "mesh://blob/client-1/token-1"
        );
        assert_eq!(body["messages"][1]["content"][2]["type"], "input_audio");
        assert_eq!(
            body["messages"][1]["content"][2]["input_audio"]["url"],
            "mesh://blob/client-1/token-2"
        );
    }

    #[test]
    fn test_normalize_openai_compat_request_marks_streaming_responses_adapter() {
        let mut body = serde_json::json!({
            "model": "test",
            "stream": true,
            "input": "hello",
        });
        let normalization = normalize_openai_compat_request("/v1/responses", &mut body).unwrap();
        assert_eq!(
            normalization.response_adapter,
            ResponseAdapter::OpenAiResponsesStream
        );
        assert_eq!(
            normalization.rewritten_path.as_deref(),
            Some("/v1/chat/completions")
        );
        assert_eq!(body["messages"][0]["content"], "hello");
    }

    #[test]
    fn test_translate_chat_completion_to_responses_json() {
        let translated = response_adapter::translate_chat_completion_to_responses(
            serde_json::json!({
                "id": "chatcmpl_123",
                "object": "chat.completion",
                "created": 1234,
                "model": "test-model",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "hello from mesh"},
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 3,
                    "total_tokens": 8
                }
            })
            .to_string()
            .as_bytes(),
        )
        .unwrap();
        let response: serde_json::Value = serde_json::from_slice(&translated).unwrap();

        assert_eq!(response["object"], "response");
        assert_eq!(response["model"], "test-model");
        assert_eq!(response["output_text"], "hello from mesh");
        assert_eq!(response["output"][0]["content"][0]["type"], "output_text");
        assert_eq!(response["usage"]["input_tokens"], 5);
        assert_eq!(response["usage"]["output_tokens"], 3);
        assert_eq!(response["usage"]["total_tokens"], 8);
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
    fn test_mesh_blob_token_from_url_requires_client_id_segment() {
        assert_eq!(
            mesh_blob_token_from_url("mesh://blob/client-1/token-123"),
            Some("token-123".to_string())
        );
        assert_eq!(mesh_blob_token_from_url("mesh://blob/token-123"), None);
        assert_eq!(
            mesh_blob_token_from_url("mesh://blob/client-1/token-123/extra"),
            None
        );
    }

    #[test]
    fn test_reorder_candidates_by_context_prefers_known_fit_then_unknown() {
        let ordered = reorder_candidates_by_context_and_throughput(
            &[
                (1u8, Some(4096), None),
                (2u8, None, None),
                (3u8, Some(16384), None),
            ],
            Some(8192),
        );

        assert_eq!(ordered, vec![3, 2]);
    }

    #[test]
    fn test_reorder_candidates_by_context_falls_back_when_all_known_too_small() {
        let ordered = reorder_candidates_by_context_and_throughput(
            &[(1u8, Some(4096), None), (2u8, Some(6144), None)],
            Some(8192),
        );

        assert_eq!(ordered, vec![1, 2]);
    }

    #[test]
    fn test_reorder_candidates_without_throughput_preserves_stable_order() {
        let ordered = reorder_candidates_by_context_and_throughput(
            &[
                (1u8, Some(8192), None),
                (2u8, Some(8192), None),
                (3u8, None, None),
            ],
            Some(4096),
        );

        assert_eq!(ordered, vec![1, 2, 3]);
    }

    #[test]
    fn test_reorder_candidates_by_throughput_prefers_stronger_hint() {
        let ordered = reorder_candidates_by_context_and_throughput(
            &[
                (
                    1u8,
                    Some(8192),
                    Some(TargetThroughputRank {
                        avg_tokens_per_second_milli: 10_000,
                        throughput_samples: 4,
                        local_observation: false,
                    }),
                ),
                (
                    2u8,
                    Some(8192),
                    Some(TargetThroughputRank {
                        avg_tokens_per_second_milli: 40_000,
                        throughput_samples: 4,
                        local_observation: false,
                    }),
                ),
            ],
            Some(4096),
        );

        assert_eq!(ordered, vec![2, 1]);
    }

    #[test]
    fn test_reorder_candidates_uses_samples_as_tiebreaker_not_multiplier() {
        let ordered = reorder_candidates_by_context_and_throughput(
            &[
                (
                    1u8,
                    Some(8192),
                    Some(TargetThroughputRank {
                        avg_tokens_per_second_milli: 20_000,
                        throughput_samples: 32,
                        local_observation: false,
                    }),
                ),
                (
                    2u8,
                    Some(8192),
                    Some(TargetThroughputRank {
                        avg_tokens_per_second_milli: 40_000,
                        throughput_samples: 2,
                        local_observation: false,
                    }),
                ),
            ],
            Some(4096),
        );

        assert_eq!(ordered, vec![2, 1]);
    }

    #[test]
    fn test_reorder_candidates_keeps_context_fit_ahead_of_faster_unknown() {
        let ordered = reorder_candidates_by_context_and_throughput(
            &[
                (
                    1u8,
                    Some(8192),
                    Some(TargetThroughputRank {
                        avg_tokens_per_second_milli: 10_000,
                        throughput_samples: 4,
                        local_observation: false,
                    }),
                ),
                (
                    2u8,
                    None,
                    Some(TargetThroughputRank {
                        avg_tokens_per_second_milli: 90_000,
                        throughput_samples: 16,
                        local_observation: false,
                    }),
                ),
            ],
            Some(4096),
        );

        assert_eq!(ordered, vec![1, 2]);
    }

    #[test]
    fn test_reorder_candidates_weights_local_observations_above_gossip() {
        let ordered = reorder_candidates_by_context_and_throughput(
            &[
                (
                    1u8,
                    Some(8192),
                    Some(TargetThroughputRank {
                        avg_tokens_per_second_milli: 20_000,
                        throughput_samples: 3,
                        local_observation: true,
                    }),
                ),
                (
                    2u8,
                    Some(8192),
                    Some(TargetThroughputRank {
                        avg_tokens_per_second_milli: 50_000,
                        throughput_samples: 4,
                        local_observation: false,
                    }),
                ),
            ],
            Some(4096),
        );

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

    #[test]
    fn test_endpoint_forward_path_maps_v1_requests_onto_api_v1_base() {
        let url = Url::parse("http://localhost:8000/api/v1").unwrap();
        let forwarded = endpoint_forward_path(&url, "/v1/chat/completions?stream=true");
        assert_eq!(forwarded, "/api/v1/chat/completions?stream=true");
    }

    #[test]
    fn test_rewrite_http_request_target_updates_request_line_and_host() {
        let raw = b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost:9337\r\nContent-Type: application/json\r\nContent-Length: 2\r\n\r\n{}";
        let rewritten =
            rewrite_http_request_target(raw, "/api/v1/chat/completions", "localhost", 8000)
                .unwrap();
        let rewritten = String::from_utf8(rewritten).unwrap();
        assert!(rewritten.starts_with("POST /api/v1/chat/completions HTTP/1.1\r\n"));
        assert!(rewritten.contains("\r\nHost: localhost:8000\r\n"));
        assert!(rewritten.ends_with("\r\n\r\n{}"));
    }

    #[test]
    fn test_remap_error_http_response_rewrites_llama_error_body() {
        let upstream = b"HTTP/1.1 404 Not Found\r\nContent-Type: application/json\r\nContent-Length: 52\r\n\r\n{\"type\":\"not_found_error\",\"message\":\"model missing\"}";
        let header_end = upstream
            .windows(4)
            .position(|w| w == b"\r\n\r\n")
            .map(|idx| idx + 4)
            .unwrap();
        let remapped = remap_error_http_response(404, header_end, upstream)
            .expect("llama error should be remapped");
        let remapped_text = String::from_utf8(remapped).unwrap();

        assert!(remapped_text.starts_with("HTTP/1.1 404 Not Found\r\n"));
        assert!(remapped_text.contains("\r\nContent-Type: application/json\r\n"));
        assert!(remapped_text.contains("\"type\":\"invalid_request_error\""));
        assert!(remapped_text.contains("\"code\":\"model_not_found\""));
        assert!(remapped_text.contains("\"message\":\"model missing\""));
    }

    #[test]
    fn test_remap_error_http_response_keeps_openai_error_passthrough() {
        let upstream = b"HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\nContent-Length: 110\r\n\r\n{\"error\":{\"message\":\"bad request\",\"type\":\"invalid_request_error\",\"param\":null,\"code\":\"invalid_value\"}}";
        let header_end = upstream
            .windows(4)
            .position(|w| w == b"\r\n\r\n")
            .map(|idx| idx + 4)
            .unwrap();
        assert!(remap_error_http_response(400, header_end, upstream).is_none());
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

        assert!(request.body_json.is_none());
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

        let mut request = read_request_from_parts(vec![request.into_bytes()]).await;

        assert_eq!(request.model_name.as_deref(), Some("qwen"));
        request.ensure_body_json();
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

        assert!(request.body_json.is_none());
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

        let mut request = read_request_from_parts_with_limits(vec![request], limits).await;

        assert_eq!(request.model_name.as_deref(), Some("qwen"));
        assert!(request.raw.len() > limits.max_body_bytes);
        request.ensure_body_json();
        let body_json = request.body_json.unwrap();
        let content = body_json["messages"][0]["content"].as_str().unwrap();
        assert_eq!(content.len(), 48);
    }

    #[tokio::test]
    async fn test_read_http_request_allows_large_object_upload_body() {
        let body = vec![b'x'; MAX_BODY_BYTES + 1];
        let headers = format!(
            "POST /api/objects HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/octet-stream\r\nContent-Length: {}\r\n\r\n",
            body.len()
        )
        .into_bytes();

        let request = read_request_from_parts(vec![headers, body.clone()]).await;

        assert_eq!(request.path, "/api/objects");
        assert!(request.raw.ends_with(&body));
        assert!(request.body_json.is_none());
        assert!(request.request_object_request_ids.is_empty());
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

    /// `probe_http_response_local` uses a much longer timeout (10 min)
    /// than `probe_http_response` (5 min), because local prefill can
    /// legitimately take minutes under load.
    ///
    /// This test sends a response after a 2s delay and verifies that
    /// `probe_http_response_local` waits for it (well within its 10-min
    /// window) rather than failing at the shorter remote timeout.
    #[tokio::test]
    async fn test_probe_http_response_local_tolerates_slow_first_byte() {
        use tokio::io::AsyncWriteExt;

        let (client, mut server) = tokio::io::duplex(4096);
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_secs(2)).await;
            let _ = server
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n{}")
                .await;
        });

        let mut reader = client;
        let result = super::probe_http_response_local(&mut reader).await;
        assert!(
            result.is_ok(),
            "probe_http_response_local should NOT timeout for slow local responses"
        );
        assert_eq!(result.unwrap().status_code, 200);
    }

    #[tokio::test]
    async fn test_send_error_429_includes_retry_after() {
        use tokio::io::AsyncReadExt;
        use tokio::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            super::send_error(stream, 429, "model not available")
                .await
                .unwrap();
        });

        let mut client = tokio::net::TcpStream::connect(addr).await.unwrap();
        let mut buf = vec![0u8; 4096];
        let mut total = 0;
        loop {
            let n = client.read(&mut buf[total..]).await.unwrap();
            if n == 0 {
                break;
            }
            total += n;
        }
        let response = String::from_utf8_lossy(&buf[..total]);

        assert!(response.starts_with("HTTP/1.1 429 Too Many Requests\r\n"));
        assert!(response.contains("Retry-After: 5\r\n"));
        assert!(response.contains("model not available"));

        server.await.unwrap();
    }

    #[test]
    fn test_inject_mesh_hooks_enabled() {
        let mut raw = b"POST /v1/chat/completions HTTP/1.1\r\nContent-Length: 25\r\n\r\n{\"model\":\"auto\",\"n\":1}".to_vec();
        inject_mesh_hooks_flag(&mut raw, true);
        let body_start = raw.windows(4).position(|w| w == b"\r\n\r\n").unwrap() + 4;
        let body = std::str::from_utf8(&raw[body_start..]).unwrap();
        assert!(body.starts_with("{\"mesh_hooks\":true,"), "body: {body}");
        // Content-Length must match actual body length
        let cl_line = std::str::from_utf8(&raw[..body_start])
            .unwrap()
            .lines()
            .find(|l| l.to_ascii_lowercase().starts_with("content-length:"))
            .unwrap();
        let declared: usize = cl_line.split(':').nth(1).unwrap().trim().parse().unwrap();
        assert_eq!(declared, raw.len() - body_start);
    }

    #[test]
    fn test_inject_mesh_hooks_disabled() {
        let mut raw = b"POST /v1/chat/completions HTTP/1.1\r\nContent-Length: 25\r\n\r\n{\"model\":\"auto\",\"n\":1}".to_vec();
        inject_mesh_hooks_flag(&mut raw, false);
        let body_start = raw.windows(4).position(|w| w == b"\r\n\r\n").unwrap() + 4;
        let body = std::str::from_utf8(&raw[body_start..]).unwrap();
        assert!(body.starts_with("{\"mesh_hooks\":false,"), "body: {body}");
    }

    #[test]
    fn test_inject_mesh_hooks_no_body() {
        let mut raw = b"GET /v1/models HTTP/1.1\r\nHost: localhost\r\n\r\n".to_vec();
        let before = raw.clone();
        inject_mesh_hooks_flag(&mut raw, true);
        assert_eq!(raw, before, "GET with no body should be unchanged");
    }

    #[test]
    fn test_rewrite_model_field_updates_body_and_content_length() {
        let mut request = BufferedHttpRequest {
            raw: b"POST /v1/chat/completions HTTP/1.1\r\nContent-Length: 45\r\n\r\n{\"model\":\"auto\",\"messages\":[],\"mesh_hooks\":true}".to_vec(),
            method: "POST".to_string(),
            path: "/v1/chat/completions".to_string(),
            body_json: None,
            body_json_attempted: false,
            body_bytes: None,
            body_len_bytes: 45,
            completion_tokens: None,
            model_name: Some("auto".to_string()),
            request_object_request_ids: Vec::new(),
            response_adapter: ResponseAdapter::None,
        };

        rewrite_model_field(&mut request, "SmolLM2-135M-Instruct-Q8_0");

        let body_start = request
            .raw
            .windows(4)
            .position(|w| w == b"\r\n\r\n")
            .unwrap()
            + 4;
        let body: serde_json::Value = serde_json::from_slice(&request.raw[body_start..]).unwrap();
        assert_eq!(body["model"], "SmolLM2-135M-Instruct-Q8_0");
        assert_eq!(body["mesh_hooks"], true);
        assert_eq!(
            request.model_name.as_deref(),
            Some("SmolLM2-135M-Instruct-Q8_0")
        );

        let cl_line = std::str::from_utf8(&request.raw[..body_start])
            .unwrap()
            .lines()
            .find(|line| line.to_ascii_lowercase().starts_with("content-length:"))
            .unwrap();
        let declared: usize = cl_line.split(':').nth(1).unwrap().trim().parse().unwrap();
        assert_eq!(declared, request.raw.len() - body_start);
        assert_eq!(declared, request.body_len_bytes);
    }
}
