//! HTTP proxy plumbing — request parsing, model routing, response helpers.
//!
//! Used by the API proxy (port 9337), bootstrap proxy, and passive mode.
//! All inference traffic flows through these functions.

use crate::inference::election;
use crate::mesh;
use crate::network::affinity::{
    prepare_remote_targets_for_request, AffinityRouter, PreparedTargets,
};
use crate::network::openai::adapter;
use crate::network::openai::errors;
use crate::network::openai::request_adapter;
use crate::network::openai::response_adapter;
use crate::network::router;
use crate::plugin;
use anyhow::{anyhow, bail, Context, Result};
use serde::Deserialize;
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
    pub session_hint: Option<String>,
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
    Delivered { status_code: u16 },
    RetryableUnavailable,
    RetryableContextOverflow,
    ClientDisconnected,
}

fn route_attempt_result_label(result: &RouteAttemptResult) -> &'static str {
    match result {
        RouteAttemptResult::Delivered { .. } => "delivered",
        RouteAttemptResult::RetryableUnavailable => "retryable_unavailable",
        RouteAttemptResult::RetryableContextOverflow => "retryable_context_overflow",
        RouteAttemptResult::ClientDisconnected => "client_disconnected",
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

struct ParsedResponseHeaders {
    header_end: usize,
    status_code: u16,
    content_length: Option<usize>,
}

#[derive(Debug, Default, Deserialize)]
struct RequestMetadata {
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    user: Option<String>,
    #[serde(default)]
    session_id: Option<String>,
    #[serde(default)]
    max_completion_tokens: Option<u32>,
    #[serde(default)]
    max_tokens: Option<u32>,
    #[serde(default)]
    max_output_tokens: Option<u32>,
    #[serde(default)]
    n_predict: Option<u32>,
}

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

    let body = if parsed.is_chunked {
        let mut sent_continue = false;
        loop {
            if let Some((consumed, decoded)) =
                try_decode_chunked_body(&raw[header_end..], body_limits.max_body_bytes)?
            {
                raw.truncate(header_end + consumed);
                break decoded;
            }
            if !sent_continue && parsed.expects_continue {
                stream.write_all(b"HTTP/1.1 100 Continue\r\n\r\n").await?;
                sent_continue = true;
            }
            read_more(stream, &mut raw).await?;
            if raw.len().saturating_sub(header_end) > body_limits.max_chunked_wire_bytes {
                bail!(
                    "HTTP chunked wire body exceeds {} bytes",
                    body_limits.max_chunked_wire_bytes
                );
            }
        }
    } else if let Some(content_length) = parsed.content_length {
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
            read_more(stream, &mut raw).await?;
        }
        raw.truncate(body_end);
        raw[header_end..body_end].to_vec()
    } else {
        raw.truncate(header_end);
        Vec::new()
    };

    let metadata = if body.is_empty() {
        None
    } else {
        serde_json::from_slice::<RequestMetadata>(&body).ok()
    };
    let mut body_json = None;
    let mut request_object_request_ids = Vec::new();
    let mut request_path = parsed.path.clone();
    let mut response_adapter = ResponseAdapter::None;
    let requires_json_transform =
        request_requires_json_transform(&parsed.path, &body, plugin_manager.is_some());
    let rewritten_body = if requires_json_transform {
        body_json = serde_json::from_slice(&body).ok();
        if let Some(body_json) = body_json.as_mut() {
            let normalization = normalize_openai_compat_request(&parsed.path, body_json)?;
            let mut changed = normalization.changed;
            if let Some(rewritten_path) = normalization.rewritten_path {
                request_path = rewritten_path;
            }
            response_adapter = normalization.response_adapter;
            if let Some(plugin_manager) = plugin_manager {
                let resolved_request_ids =
                    resolve_request_object_references(&request_path, body_json, plugin_manager)
                        .await?;
                if !resolved_request_ids.is_empty() {
                    request_object_request_ids = resolved_request_ids;
                    changed = true;
                }
            }
            if changed {
                Some(
                    serde_json::to_vec(body_json)
                        .context("serialize normalized OpenAI-compatible request body")?,
                )
            } else {
                None
            }
        } else {
            None
        }
    } else {
        None
    };
    let model_name = metadata.as_ref().and_then(|value| value.model.clone());
    let session_hint = metadata
        .as_ref()
        .and_then(|value| value.user.clone().or_else(|| value.session_id.clone()));
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
        Some(&request_path),
        rewritten_body.as_deref(),
    )?;
    let body_len_bytes = body.len();
    let body_bytes = if body.is_empty() { None } else { Some(body) };

    Ok(BufferedHttpRequest {
        raw,
        method: parsed.method,
        path: request_path,
        body_json,
        body_json_attempted: requires_json_transform,
        body_bytes,
        body_len_bytes,
        completion_tokens,
        model_name,
        session_hint,
        request_object_request_ids,
        response_adapter,
    })
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
    let normalized = request_adapter::normalize_openai_compat_request(path, body)?;
    let response_adapter = match normalized.response_adapter {
        request_adapter::ResponseAdapterMode::None => ResponseAdapter::None,
        request_adapter::ResponseAdapterMode::OpenAiResponsesJson => {
            ResponseAdapter::OpenAiResponsesJson
        }
        request_adapter::ResponseAdapterMode::OpenAiResponsesStream => {
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
    required_tokens: Option<u32>,
    hosts: &[iroh::EndpointId],
) -> Vec<iroh::EndpointId> {
    let mut candidates = Vec::with_capacity(hosts.len());
    for host in hosts {
        candidates.push((*host, node.peer_model_context_length(*host, model).await));
    }
    reorder_candidates_by_context(&candidates, required_tokens)
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
    let created_at = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_secs() as i64)
        .unwrap_or(0);
    let response_id = format!("resp_{created_at}");
    let item_id = format!("msg_{created_at}");
    let mut model = String::new();
    let mut output_text = String::new();
    let mut usage = None;
    let header = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n";
    tcp_stream.write_all(header.as_bytes()).await?;

    let mut created_emitted = false;
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

            if !should_parse_stream_chunk(&data, model.is_empty(), usage.is_none()) {
                continue;
            }

            let chunk = adapter::parse_chat_stream_chunk(&data)?;
            if let Some(chunk_model) = chunk.model.as_deref() {
                if model.is_empty() {
                    model = chunk_model.to_string();
                }
            }
            // Emit response.created once we have the model from the first chunk.
            if !created_emitted && !model.is_empty() {
                let created = serde_json::to_string(
                    &response_adapter::responses_stream_created_event(&model, created_at),
                )
                .context("serialize response.created stream event")?;
                response_adapter::write_chunked_sse_event(
                    tcp_stream,
                    Some("response.created"),
                    &created,
                )
                .await?;
                created_emitted = true;
            }
            if let Some(delta) = chunk
                .choices
                .first()
                .and_then(|choice| choice.delta.as_ref())
                .and_then(|delta| delta.content.as_deref())
            {
                output_text.push_str(delta);
                let event = serde_json::to_string(&response_adapter::responses_stream_delta_event(
                    &item_id, delta,
                ))
                .context("serialize response.output_text.delta event")?;
                response_adapter::write_chunked_sse_event(
                    tcp_stream,
                    Some("response.output_text.delta"),
                    &event,
                )
                .await?;
            }
            if usage.is_none() {
                usage = chunk
                    .usage
                    .as_ref()
                    .map(response_adapter::stream_usage_to_responses_usage);
            }
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

    // If upstream sent no model field at all (e.g. empty stream), still emit response.created.
    if !created_emitted {
        let created = serde_json::to_string(&response_adapter::responses_stream_created_event(
            &model, created_at,
        ))
        .context("serialize response.created stream event")?;
        response_adapter::write_chunked_sse_event(tcp_stream, Some("response.created"), &created)
            .await?;
    }

    let text_done = serde_json::to_string(&response_adapter::responses_stream_text_done_event(
        &item_id,
        &output_text,
    ))
    .context("serialize response.output_text.done event")?;
    response_adapter::write_chunked_sse_event(
        tcp_stream,
        Some("response.output_text.done"),
        &text_done,
    )
    .await?;
    let completed = serde_json::to_string(&response_adapter::responses_stream_completed_event(
        &response_id,
        created_at,
        &model,
        &item_id,
        &output_text,
        usage,
    ))
    .context("serialize response.completed event")?;
    response_adapter::write_chunked_sse_event(tcp_stream, Some("response.completed"), &completed)
        .await?;
    response_adapter::write_chunked_sse_event(tcp_stream, Some("done"), "[DONE]").await?;
    let _ = tcp_stream.write_all(b"0\r\n\r\n").await;
    let _ = tcp_stream.shutdown().await;
    Ok(RouteAttemptResult::Delivered {
        status_code: probe.status_code,
    })
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
    let header = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        translated_body.len()
    );
    tcp_stream.write_all(header.as_bytes()).await?;
    tcp_stream.write_all(&translated_body).await?;
    let _ = tcp_stream.shutdown().await;
    Ok(RouteAttemptResult::Delivered {
        status_code: probe.status_code,
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
/// the local backend proxy (which fronts llama-server). Prefill on a busy
/// or slow machine can legitimately take minutes (large prompts,
/// concurrent slot contention, slower hardware). We still bound the wait
/// to catch a truly wedged local backend proxy path.
async fn probe_http_response_local<R: AsyncRead + Unpin>(reader: &mut R) -> Result<ResponseProbe> {
    probe_http_response_with_timeout(reader, local_response_first_byte_timeout()).await
}

/// Local backend proxy timeout: 10 minutes. This is a safety net for a
/// wedged local proxy path, not a latency budget. Normal prefill even on
/// slow hardware with large prompts and concurrent slots completes well
/// within this window.
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
    let mapped_body = errors::map_upstream_error_body(status_code, &full_response[header_end..])?;
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
    Ok(RouteAttemptResult::Delivered { status_code })
}

async fn relay_probed_response<R: AsyncRead + Unpin>(
    mut tcp_stream: &mut TcpStream,
    reader: &mut R,
    probe: ResponseProbe,
    retry_context_overflow: bool,
    response_adapter: ResponseAdapter,
) -> Result<RouteAttemptResult> {
    if response_adapter == ResponseAdapter::OpenAiResponsesJson {
        return relay_translated_responses_json(tcp_stream, reader, probe, retry_context_overflow)
            .await;
    }
    if response_adapter == ResponseAdapter::OpenAiResponsesStream {
        return relay_translated_responses_stream(
            tcp_stream,
            reader,
            probe,
            retry_context_overflow,
        )
        .await;
    }

    if retry_context_overflow && probe.retryable_context_overflow {
        return Ok(RouteAttemptResult::RetryableContextOverflow);
    }
    if !(200..300).contains(&probe.status_code) {
        return relay_error_response(tcp_stream, reader, probe).await;
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
    response_adapter: ResponseAdapter,
) -> RouteAttemptResult {
    match TcpStream::connect(format!("127.0.0.1:{port}")).await {
        Ok(mut upstream) => {
            let _inflight = node.begin_inflight_request();
            let _ = upstream.set_nodelay(true);
            if let Err(err) = upstream.write_all(prefetched).await {
                tracing::warn!(
                    "API proxy: failed to forward buffered request to local backend proxy on {port}: {err}"
                );
                return RouteAttemptResult::RetryableUnavailable;
            }
            match probe_http_response_local(&mut upstream).await {
                Ok(probe) => {
                    let status_code = probe.status_code;
                    match relay_probed_response(
                        tcp_stream,
                        &mut upstream,
                        probe,
                        retry_context_overflow,
                        response_adapter,
                    )
                    .await
                    {
                        Ok(result) => result,
                        Err(err) => {
                            if is_client_disconnect_error(&err) {
                                let _ = upstream.shutdown().await;
                                tracing::info!(
                                    "API proxy (local): downstream client disconnected during relay"
                                );
                                return RouteAttemptResult::ClientDisconnected;
                            }
                            tracing::debug!("API proxy (local) ended after commit: {err}");
                            RouteAttemptResult::Delivered { status_code }
                        }
                    }
                }
                Err(err) => {
                    tracing::warn!(
                        "API proxy: failed to read local response from backend proxy on {port}: {err}"
                    );
                    RouteAttemptResult::RetryableUnavailable
                }
            }
        }
        Err(err) => {
            tracing::warn!("API proxy: can't reach local backend proxy on {port}: {err}");
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
            match probe_http_response(&mut quic_recv).await {
                Ok(probe) => {
                    let status_code = probe.status_code;
                    match relay_probed_response(
                        tcp_stream,
                        &mut quic_recv,
                        probe,
                        retry_context_overflow,
                        response_adapter,
                    )
                    .await
                    {
                        Ok(result) => result,
                        Err(err) => {
                            if is_client_disconnect_error(&err) {
                                tracing::info!(
                                    "API proxy (remote): downstream client disconnected during relay"
                                );
                                return RouteAttemptResult::ClientDisconnected;
                            }
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

async fn route_http_endpoint_attempt(
    tcp_stream: &mut TcpStream,
    base_url: &str,
    prefetched: &[u8],
    request_path: &str,
    retry_context_overflow: bool,
    response_adapter: ResponseAdapter,
) -> RouteAttemptResult {
    let url = match Url::parse(base_url) {
        Ok(url) => url,
        Err(err) => {
            tracing::warn!("API proxy: invalid external inference endpoint '{base_url}': {err}");
            return RouteAttemptResult::RetryableUnavailable;
        }
    };
    if url.scheme() != "http" {
        tracing::warn!(
            "API proxy: unsupported external inference endpoint scheme '{}' for {}",
            url.scheme(),
            base_url
        );
        return RouteAttemptResult::RetryableUnavailable;
    }

    let host = match url.host_str() {
        Some(host) => host,
        None => {
            tracing::warn!("API proxy: missing host in external inference endpoint {base_url}");
            return RouteAttemptResult::RetryableUnavailable;
        }
    };
    let port = url.port_or_known_default().unwrap_or(80);
    let forward_path = endpoint_forward_path(&url, request_path);
    let forwarded = match rewrite_http_request_target(prefetched, &forward_path, host, port) {
        Ok(forwarded) => forwarded,
        Err(err) => {
            tracing::warn!(
                "API proxy: failed to rewrite buffered request for external endpoint {}: {}",
                base_url,
                err
            );
            return RouteAttemptResult::RetryableUnavailable;
        }
    };

    match TcpStream::connect(format!("{host}:{port}")).await {
        Ok(mut upstream) => {
            let _ = upstream.set_nodelay(true);
            if let Err(err) = upstream.write_all(&forwarded).await {
                tracing::warn!(
                    "API proxy: failed to forward buffered request to external endpoint {}: {}",
                    base_url,
                    err
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
                        response_adapter,
                    )
                    .await
                    {
                        Ok(result) => result,
                        Err(err) => {
                            if is_client_disconnect_error(&err) {
                                let _ = upstream.shutdown().await;
                                tracing::info!(
                                    "API proxy (external endpoint): downstream client disconnected during relay"
                                );
                                return RouteAttemptResult::ClientDisconnected;
                            }
                            tracing::debug!(
                                "API proxy (external endpoint) ended after commit: {err}"
                            );
                            RouteAttemptResult::Delivered { status_code }
                        }
                    }
                }
                Err(err) => {
                    tracing::warn!(
                        "API proxy: failed to read response from external endpoint {}: {}",
                        base_url,
                        err
                    );
                    RouteAttemptResult::RetryableUnavailable
                }
            }
        }
        Err(err) => {
            tracing::warn!(
                "API proxy: can't reach external inference endpoint {}: {}",
                base_url,
                err
            );
            RouteAttemptResult::RetryableUnavailable
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
        let _ = send_models_list(tcp_stream, &served).await;
        return;
    }

    // Demand tracking for rebalancing (done after routing so we track the actual model used)
    // We'll track below after routing resolves the effective model

    // Smart routing: if no model specified (or model="auto"), classify and pick
    let routed_model =
        if request.model_name.is_none() || request.model_name.as_deref() == Some("auto") {
            request.ensure_body_json();
            if let Some(body_json) = request.body_json.as_ref() {
                let cl = router::classify(&body_json);
                let served = node.models_being_served().await;
                let media = router::media_requirements(body_json);
                let available: Vec<(&str, f64)> = served
                    .iter()
                    .filter(|name| {
                        let caps = crate::models::installed_model_capabilities(name);
                        (!media.needs_vision || caps.vision_label().is_some())
                            && (!media.needs_audio || caps.audio_label().is_some())
                    })
                    .map(|name| (name.as_str(), 0.0))
                    .collect();
                let available: Vec<(&str, f64)> = if available.is_empty() {
                    served.iter().map(|name| (name.as_str(), 0.0)).collect()
                } else {
                    available
                };
                let picked = router::pick_model_classified(&cl, &available);
                if let Some(name) = picked {
                    tracing::info!(
                        "router: {:?}/{:?} tools={} media={} → {name}",
                        cl.category,
                        cl.complexity,
                        cl.needs_tools,
                        cl.has_media_inputs
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

    // Enable mesh hooks for auto-routed requests. When the smart router
    // picks the model, hooks allow the local model to consult peers during
    // inference (e.g. caption images via a vision peer, get a second opinion
    // on uncertain answers). Explicit model names pass through without hooks.
    if request.model_name.is_none() || request.model_name.as_deref() == Some("auto") {
        inject_mesh_hooks_flag(&mut request.raw, true);
    }

    // Demand tracking for rebalancing
    if track_demand {
        if let Some(ref name) = effective_model {
            node.record_request(name);
        }
    }

    // Resolve target hosts by model name
    let target_hosts = if let Some(ref name) = effective_model {
        node.hosts_for_model(name).await
    } else {
        vec![]
    };
    let target_hosts = if target_hosts.is_empty() && effective_model.is_some() {
        // Named model requested but no host serves it — tell the agent to retry.
        let model = effective_model.as_deref().unwrap();
        tracing::warn!("API proxy: model {model:?} not available, no hosts serving it");
        let _ = send_error(
            tcp_stream,
            429,
            &format!("model {model:?} not currently available — retry later"),
        )
        .await;
        release_request_objects(&node, &request.request_object_request_ids).await;
        return;
    } else if target_hosts.is_empty() {
        // No model specified and no hosts at all
        match node.any_host().await {
            Some(p) => vec![p.id],
            None => {
                let _ = send_503(
                    tcp_stream,
                    "no peers serving any model (mesh empty or gossip stale)",
                )
                .await;
                release_request_objects(&node, &request.request_object_request_ids).await;
                return;
            }
        }
    } else {
        target_hosts
    };
    let required_tokens =
        request_budget_tokens_from_parts(request.body_len_bytes, request.completion_tokens);
    if effective_model.is_some() && target_hosts.len() > 1 {
        request.ensure_body_json();
    }
    let body_json = request.body_json.as_ref();
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
        let ordered =
            order_remote_hosts_by_context(&node, name, required_tokens, &target_hosts).await;
        if let (Some(prefix_hash), Some(cached_target)) =
            (prepared.learn_prefix_hash, prepared.cached_target.as_ref())
        {
            if let election::InferenceTarget::Remote(cached_host) = cached_target {
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
    let total_targets = target_hosts.len();
    for (idx, target_host) in target_hosts.iter().enumerate() {
        let retry_context_overflow = idx + 1 < total_targets;
        match route_remote_attempt(
            &node,
            &mut tcp_stream,
            *target_host,
            &request.raw,
            retry_context_overflow,
            request.response_adapter,
        )
        .await
        {
            RouteAttemptResult::Delivered { status_code } => {
                if should_learn_affinity(status_code) {
                    if let (Some(name), Some(prefix_hash)) =
                        (effective_model.as_ref(), prepared.learn_prefix_hash)
                    {
                        let target = election::InferenceTarget::Remote(*target_host);
                        affinity.learn_target(name, prefix_hash, &target);
                    }
                }
                release_request_objects(&node, &request.request_object_request_ids).await;
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
            RouteAttemptResult::ClientDisconnected => {
                tracing::info!(
                    "Downstream client disconnected while routing to host {}",
                    target_host.fmt_short()
                );
                release_request_objects(&node, &request.request_object_request_ids).await;
                return;
            }
        }
    }
    // All hosts failed
    if last_retryable {
        tracing::warn!("All hosts failed for model {:?}", effective_model);
    }
    let reason = format!(
        "all {} tunnel(s) to hosts for {:?} failed (mesh request)",
        total_targets, effective_model,
    );
    let _ = send_503(tcp_stream, &reason).await;
    release_request_objects(&node, &request.request_object_request_ids).await;
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
        election::InferenceTarget::Local(port) | election::InferenceTarget::MoeLocal(port) => {
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
        election::InferenceTarget::Remote(host_id)
        | election::InferenceTarget::MoeRemote(host_id) => {
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
    parsed_body: Option<&serde_json::Value>,
    required_tokens: Option<u32>,
    prefetched: &[u8],
    response_adapter: ResponseAdapter,
    affinity: &AffinityRouter,
) -> bool {
    let route_started = Instant::now();
    let mut tcp_stream = tcp_stream;
    let ordered_candidates =
        order_targets_by_context(&node, model, required_tokens, &targets.candidates(model)).await;
    if ordered_candidates.is_empty() {
        return false;
    }

    let selection = crate::network::affinity::select_model_target_from_candidates(
        targets,
        &ordered_candidates,
        model,
        parsed_body,
        affinity,
    );
    if matches!(selection.target, election::InferenceTarget::None) {
        let _ = send_503(
            tcp_stream,
            &format!(
                "target for model '{model}' resolved to None (election in progress or host down)"
            ),
        )
        .await;
        return true;
    }

    if let (Some(prefix_hash), Some(cached_target)) = (
        selection.learn_prefix_hash,
        selection.cached_target.as_ref(),
    ) {
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
    let total_targets = ordered.len();
    let mut attempts = 0usize;
    let mut refreshed = false;
    for (idx, target) in ordered.into_iter().enumerate() {
        attempts += 1;
        let attempt_started = Instant::now();
        let retry_context_overflow = idx + 1 < total_targets;
        let attempt_result = route_attempt_for_target(
            &node,
            &mut tcp_stream,
            &target,
            prefetched,
            retry_context_overflow,
            response_adapter,
        )
        .await;
        tracing::info!(
            model = model,
            target = ?target,
            attempt = attempts,
            total_targets = total_targets,
            outcome = route_attempt_result_label(&attempt_result),
            attempt_ms = attempt_started.elapsed().as_millis(),
            total_route_ms = route_started.elapsed().as_millis(),
            "openai route_model_request attempt"
        );
        match attempt_result {
            RouteAttemptResult::Delivered { status_code } => {
                if should_learn_affinity(status_code) {
                    if let Some(prefix_hash) = selection.learn_prefix_hash {
                        affinity.learn_target(model, prefix_hash, &target);
                    }
                }
                tracing::info!(
                    model = model,
                    attempts = attempts,
                    status_code = status_code,
                    route_ms = route_started.elapsed().as_millis(),
                    "openai route_model_request delivered"
                );
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
            RouteAttemptResult::ClientDisconnected => {
                tracing::info!(
                    model = model,
                    attempts = attempts,
                    route_ms = route_started.elapsed().as_millis(),
                    "openai route_model_request downstream disconnected"
                );
                return true;
            }
        }
    }

    let _ = send_503(
        tcp_stream,
        &format!("all {} target(s) for model '{model}' failed", total_targets),
    )
    .await;
    tracing::warn!(
        model = model,
        attempts = attempts,
        route_ms = route_started.elapsed().as_millis(),
        "openai route_model_request exhausted targets"
    );
    true
}

pub async fn route_moe_request(
    node: mesh::Node,
    tcp_stream: TcpStream,
    targets: &election::ModelTargets,
    model: &str,
    session_hint: &str,
    _parsed_body: Option<&serde_json::Value>,
    required_tokens: Option<u32>,
    prefetched: &[u8],
) -> bool {
    let route_started = Instant::now();
    let mut tcp_stream = tcp_stream;
    let Some(primary_target) = targets.get_moe_target(session_hint) else {
        let _ = send_503(
            tcp_stream,
            &format!("no MoE target for model '{model}' session '{session_hint}'"),
        )
        .await;
        return false;
    };
    let mut ordered = order_targets_by_context(
        &node,
        model,
        required_tokens,
        &targets.get_moe_failover_targets(session_hint),
    )
    .await;
    if ordered.is_empty() {
        let _ = send_503(
            tcp_stream,
            &format!("no MoE failover targets for model '{model}'"),
        )
        .await;
        return false;
    }
    move_target_first(&mut ordered, &primary_target);

    let total_targets = ordered.len();
    let mut attempts = 0usize;
    let mut refreshed = false;
    for (idx, target) in ordered.into_iter().enumerate() {
        attempts += 1;
        let attempt_started = Instant::now();
        let retry_context_overflow = idx + 1 < total_targets;
        let attempt_result = route_attempt_for_target(
            &node,
            &mut tcp_stream,
            &target,
            prefetched,
            retry_context_overflow,
            ResponseAdapter::None,
        )
        .await;
        tracing::info!(
            model = model,
            session_hint = session_hint,
            target = ?target,
            attempt = attempts,
            total_targets = total_targets,
            outcome = route_attempt_result_label(&attempt_result),
            attempt_ms = attempt_started.elapsed().as_millis(),
            total_route_ms = route_started.elapsed().as_millis(),
            "openai route_moe_request attempt"
        );
        match attempt_result {
            RouteAttemptResult::Delivered { status_code } => {
                tracing::info!(
                    model = model,
                    session_hint = session_hint,
                    attempts = attempts,
                    status_code = status_code,
                    route_ms = route_started.elapsed().as_millis(),
                    "openai route_moe_request delivered"
                );
                return true;
            }
            RouteAttemptResult::RetryableContextOverflow => {
                tracing::warn!("MoE target {target:?} rejected request with context overflow-style 400, trying next");
            }
            RouteAttemptResult::RetryableUnavailable => {
                if let election::InferenceTarget::MoeRemote(peer_id) = &target {
                    node.handle_peer_death(*peer_id).await;
                }
                if !refreshed {
                    let refresh_node = node.clone();
                    tokio::spawn(async move {
                        refresh_node.gossip_one_peer().await;
                    });
                    refreshed = true;
                }
                tracing::warn!("MoE target {target:?} unavailable, trying next");
            }
            RouteAttemptResult::ClientDisconnected => {
                tracing::info!(
                    model = model,
                    session_hint = session_hint,
                    attempts = attempts,
                    route_ms = route_started.elapsed().as_millis(),
                    "openai route_moe_request downstream disconnected"
                );
                return true;
            }
        }
    }

    let _ = send_503(
        tcp_stream,
        &format!("all MoE targets for model '{model}' failed"),
    )
    .await;
    tracing::warn!(
        model = model,
        session_hint = session_hint,
        attempts = attempts,
        route_ms = route_started.elapsed().as_millis(),
        "openai route_moe_request exhausted targets"
    );
    true
}

/// Route a request to a known inference target (local backend proxy or remote host).
///
/// Used by the API proxy after election has determined the target.
pub async fn route_to_target(
    node: mesh::Node,
    tcp_stream: TcpStream,
    target: election::InferenceTarget,
    prefetched: &[u8],
    response_adapter: ResponseAdapter,
) -> bool {
    let route_started = Instant::now();
    let mut tcp_stream = tcp_stream;
    tracing::info!("API proxy: routing to target {target:?}");
    let moe_remote_id = match &target {
        election::InferenceTarget::MoeRemote(host_id) => Some(*host_id),
        _ => None,
    };
    let result = route_attempt_for_target(
        &node,
        &mut tcp_stream,
        &target,
        prefetched,
        false,
        response_adapter,
    )
    .await;
    tracing::info!(
        target = ?target,
        outcome = route_attempt_result_label(&result),
        route_ms = route_started.elapsed().as_millis(),
        "openai route_to_target result"
    );
    match result {
        RouteAttemptResult::Delivered { .. } => true,
        RouteAttemptResult::RetryableContextOverflow | RouteAttemptResult::RetryableUnavailable => {
            if let Some(moe_host_id) = moe_remote_id {
                node.handle_peer_death(moe_host_id).await;
            }
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
    tracing::info!(
        endpoint = base_url,
        path = request_path,
        outcome = route_attempt_result_label(&result),
        route_ms = started.elapsed().as_millis(),
        "openai route_http_endpoint_request result"
    );
    match result {
        RouteAttemptResult::Delivered { .. } => true,
        RouteAttemptResult::RetryableContextOverflow | RouteAttemptResult::RetryableUnavailable => {
            false
        }
        RouteAttemptResult::ClientDisconnected => true,
    }
}

// ── Response helpers ──

pub async fn send_models_list(mut stream: TcpStream, models: &[String]) -> std::io::Result<()> {
    let data: Vec<serde_json::Value> = models
        .iter()
        .map(|m| {
            let capabilities = crate::models::installed_model_capabilities(m);
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
            let display_name = crate::models::installed_model_display_name(m);
            serde_json::json!({
                "id": m,
                "display_name": display_name,
                "object": "model",
                "owned_by": "mesh-llm",
                "capabilities": caps,
                "multimodal_status": capabilities.multimodal_status(),
                "vision_status": capabilities.vision_status(),
                "audio_status": capabilities.audio_status(),
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

    match crate::inference::pipeline::pre_plan(&http_client, &planner_url, planner_model, &messages)
        .await
    {
        Ok(plan) => {
            tracing::info!(
                "pipeline: pre-plan by {} in {}ms — {}",
                plan.model_used,
                plan.elapsed_ms,
                plan.plan_text.chars().take(200).collect::<String>()
            );
            crate::inference::pipeline::inject_plan(&mut body, &plan);
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
            route_attempt_result_label(&RouteAttemptResult::Delivered { status_code: 200 }),
            "delivered"
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
        assert_eq!(request.session_hint.as_deref(), Some("alice"));
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
        assert_eq!(request.session_hint.as_deref(), Some("sess-42"));
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

    /// `probe_http_response_local` uses a much longer timeout (10 min)
    /// than `probe_http_response` (5 min), because local llama-server
    /// prefill can legitimately take minutes under load.
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
}
