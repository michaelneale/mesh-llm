use anyhow::{anyhow, bail, Context, Result};

use crate::network::transport::TransportIo;

const MAX_HEADER_BYTES: usize = 64 * 1024;
const MAX_BODY_BYTES: usize = 8 * 1024 * 1024;
const MAX_OBJECT_UPLOAD_BODY_BYTES: usize = 64 * 1024 * 1024;
const MAX_CHUNKED_WIRE_BYTES: usize = MAX_BODY_BYTES * 6 + 64 * 1024;
const MAX_OBJECT_UPLOAD_CHUNKED_WIRE_BYTES: usize = MAX_OBJECT_UPLOAD_BODY_BYTES * 6 + 64 * 1024;
const MAX_HEADERS: usize = 64;

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

struct ParsedHeaders {
    header_end: usize,
    method: String,
    path: String,
    content_length: Option<usize>,
    is_chunked: bool,
    expects_continue: bool,
}

struct RequestNormalization {
    changed: bool,
    rewritten_path: Option<String>,
    response_adapter: ResponseAdapter,
}

#[derive(Debug)]
pub struct BufferedHttpRequest {
    pub raw: Vec<u8>,
    pub method: String,
    pub path: String,
    pub body_json: Option<serde_json::Value>,
    pub model_name: Option<String>,
    pub session_hint: Option<String>,
    pub request_object_request_ids: Vec<String>,
    pub response_adapter: ResponseAdapter,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResponseAdapter {
    None,
    OpenAiResponsesJson,
    OpenAiResponsesStream,
}

pub fn is_models_list_request(method: &str, path: &str) -> bool {
    let path = path.split('?').next().unwrap_or(path);
    method == "GET" && (path == "/v1/models" || path == "/models")
}

pub fn pipeline_request_supported(path: &str, body: &serde_json::Value) -> bool {
    let path = path.split('?').next().unwrap_or(path);
    path == "/v1/chat/completions"
        && body
            .get("messages")
            .map(|messages| messages.is_array())
            .unwrap_or(false)
}

pub async fn read_http_request(transport: &mut dyn TransportIo) -> Result<BufferedHttpRequest> {
    let mut raw = Vec::with_capacity(8192);
    let parsed =
        read_until_headers_parsed(transport, &mut raw, HTTP_READ_LIMITS.max_header_bytes).await?;
    let limits = body_limits_for_path(&parsed.path, HTTP_READ_LIMITS);
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
                transport
                    .write_all(b"HTTP/1.1 100 Continue\r\n\r\n")
                    .await?;
                sent_continue = true;
            }
            read_more(transport, &mut raw).await?;
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
                transport
                    .write_all(b"HTTP/1.1 100 Continue\r\n\r\n")
                    .await?;
                sent_continue = true;
            }
            read_more(transport, &mut raw).await?;
        }
        raw.truncate(body_end);
        raw[header_end..body_end].to_vec()
    } else {
        raw.truncate(header_end);
        Vec::new()
    };

    let mut body_json: Option<serde_json::Value> = if body.is_empty() {
        None
    } else {
        serde_json::from_slice(&body).ok()
    };
    let mut request_path = parsed.path.clone();
    let mut response_adapter = ResponseAdapter::None;
    let rewritten_body = if let Some(body_json) = body_json.as_mut() {
        let normalization = normalize_openai_compat_request(&parsed.path, body_json)?;
        if let Some(rewritten_path) = normalization.rewritten_path {
            request_path = rewritten_path;
        }
        response_adapter = normalization.response_adapter;
        if normalization.changed {
            Some(
                serde_json::to_vec(body_json)
                    .context("serialize normalized OpenAI-compatible request body")?,
            )
        } else {
            None
        }
    } else {
        None
    };
    let model_name = body_json.as_ref().and_then(extract_model_from_json);
    let session_hint = body_json.as_ref().and_then(extract_session_hint_from_json);
    let raw = finalize_forwarded_request(
        raw,
        header_end,
        parsed.expects_continue,
        Some(&request_path),
        rewritten_body.as_deref(),
    )?;

    Ok(BufferedHttpRequest {
        raw,
        method: parsed.method,
        path: request_path,
        body_json,
        model_name,
        session_hint,
        request_object_request_ids: Vec::new(),
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

    rebuilt.push_str("Connection: close\r\n\r\n");

    let mut forwarded = rebuilt.into_bytes();
    forwarded.extend_from_slice(rewritten_body.unwrap_or(&original_body));
    Ok(forwarded)
}

async fn read_until_headers_parsed(
    transport: &mut dyn TransportIo,
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
                read_more(transport, buf).await?;
            }
            Err(e) => bail!("HTTP parse error: {e}"),
        }
    }
}

async fn read_more(transport: &mut dyn TransportIo, buf: &mut Vec<u8>) -> Result<()> {
    let mut chunk = [0u8; 8192];
    let n = transport.read(&mut chunk).await?;
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

fn path_only(path: &str) -> &str {
    path.split('?').next().unwrap_or(path)
}

fn rewrite_path_preserving_query(path: &str, new_path: &str) -> String {
    match path.split_once('?') {
        Some((_, query)) => format!("{new_path}?{query}"),
        None => new_path.to_string(),
    }
}

fn alias_max_tokens(object: &mut serde_json::Map<String, serde_json::Value>) -> bool {
    let mut changed = false;
    for alias in ["max_completion_tokens", "max_output_tokens"] {
        let Some(value) = object.remove(alias) else {
            continue;
        };
        changed = true;
        object.entry("max_tokens".to_string()).or_insert(value);
    }
    changed
}

fn normalize_openai_compat_request(
    path: &str,
    body: &mut serde_json::Value,
) -> Result<RequestNormalization> {
    let Some(object) = body.as_object_mut() else {
        return Ok(RequestNormalization {
            changed: false,
            rewritten_path: None,
            response_adapter: ResponseAdapter::None,
        });
    };

    let mut changed = alias_max_tokens(object);
    let mut rewritten_path = None;
    let mut response_adapter = ResponseAdapter::None;

    if path_only(path) == "/v1/responses" {
        let is_stream = object
            .get("stream")
            .and_then(|value| value.as_bool())
            .unwrap_or(false);
        changed |= translate_openai_responses_input(object)?;
        rewritten_path = Some(rewrite_path_preserving_query(path, "/v1/chat/completions"));
        response_adapter = if is_stream {
            ResponseAdapter::OpenAiResponsesStream
        } else {
            ResponseAdapter::OpenAiResponsesJson
        };
    }

    Ok(RequestNormalization {
        changed,
        rewritten_path,
        response_adapter,
    })
}

fn map_response_role(role: &str) -> String {
    match role {
        "developer" => "system".to_string(),
        other => other.to_string(),
    }
}

fn object_or_url_container(
    value: Option<&serde_json::Value>,
    fallback_url: Option<&str>,
) -> Option<serde_json::Map<String, serde_json::Value>> {
    match value {
        Some(serde_json::Value::Object(map)) => Some(map.clone()),
        Some(serde_json::Value::String(url)) => Some(serde_json::Map::from_iter([(
            "url".to_string(),
            serde_json::Value::String(url.clone()),
        )])),
        _ => fallback_url.map(|url| {
            serde_json::Map::from_iter([(
                "url".to_string(),
                serde_json::Value::String(url.to_string()),
            )])
        }),
    }
}

fn translate_responses_content_item(item: &serde_json::Value) -> Result<serde_json::Value> {
    let Some(object) = item.as_object() else {
        return Ok(serde_json::json!({
            "type": "text",
            "text": item.as_str().unwrap_or_default(),
        }));
    };
    let item_type = object
        .get("type")
        .and_then(|value| value.as_str())
        .unwrap_or("text");

    match item_type {
        "input_text" | "text" => {
            let text = object
                .get("text")
                .and_then(|value| value.as_str())
                .unwrap_or_default();
            Ok(serde_json::json!({"type": "text", "text": text}))
        }
        "input_image" | "image_url" | "image" => {
            let container = object_or_url_container(
                object.get("image_url").or_else(|| object.get("image")),
                object.get("url").and_then(|value| value.as_str()),
            )
            .ok_or_else(|| anyhow!("responses input_image block is missing image_url/url"))?;
            Ok(serde_json::json!({"type": "image_url", "image_url": container}))
        }
        "input_audio" | "audio" | "audio_url" => {
            let mut container = object_or_url_container(
                object
                    .get("input_audio")
                    .or_else(|| object.get("audio_url")),
                object.get("url").and_then(|value| value.as_str()),
            )
            .unwrap_or_default();
            for key in [
                "data",
                "format",
                "mime_type",
                "mesh_token",
                "blob_token",
                "token",
            ] {
                if let Some(value) = object.get(key) {
                    container
                        .entry(key.to_string())
                        .or_insert_with(|| value.clone());
                }
            }
            if container.is_empty() {
                bail!("responses input_audio block is missing input_audio/audio_url/url");
            }
            Ok(serde_json::json!({"type": "input_audio", "input_audio": container}))
        }
        "input_file" | "file" => {
            let mut container = object_or_url_container(
                object.get("input_file").or_else(|| object.get("file")),
                object.get("url").and_then(|value| value.as_str()),
            )
            .ok_or_else(|| anyhow!("responses input_file block is missing input_file/file/url"))?;
            for key in [
                "mime_type",
                "file_name",
                "filename",
                "mesh_token",
                "blob_token",
                "token",
            ] {
                if let Some(value) = object.get(key) {
                    container
                        .entry(key.to_string())
                        .or_insert_with(|| value.clone());
                }
            }
            Ok(serde_json::json!({"type": "input_file", "input_file": container}))
        }
        other => bail!("unsupported /v1/responses content block type '{other}'"),
    }
}

fn collapse_blocks_if_text_only(blocks: Vec<serde_json::Value>) -> serde_json::Value {
    if blocks.len() == 1 {
        if let Some(text) = blocks[0].get("text").and_then(|value| value.as_str()) {
            return serde_json::Value::String(text.to_string());
        }
    }
    serde_json::Value::Array(blocks)
}

fn translate_responses_message_content(content: &serde_json::Value) -> Result<serde_json::Value> {
    match content {
        serde_json::Value::String(text) => Ok(serde_json::Value::String(text.clone())),
        serde_json::Value::Array(items) => {
            let blocks = items
                .iter()
                .map(translate_responses_content_item)
                .collect::<Result<Vec<_>>>()?;
            Ok(collapse_blocks_if_text_only(blocks))
        }
        serde_json::Value::Object(_) => Ok(collapse_blocks_if_text_only(vec![
            translate_responses_content_item(content)?,
        ])),
        _ => bail!("unsupported /v1/responses input content shape"),
    }
}

fn translate_responses_input_message(
    message: &serde_json::Value,
) -> Result<serde_json::Map<String, serde_json::Value>> {
    let Some(object) = message.as_object() else {
        bail!("unsupported /v1/responses message shape");
    };

    let role = map_response_role(
        object
            .get("role")
            .and_then(|value| value.as_str())
            .unwrap_or("user"),
    );
    let content_value = object
        .get("content")
        .map(translate_responses_message_content)
        .transpose()?
        .unwrap_or_else(|| serde_json::Value::String(String::new()));

    Ok(serde_json::Map::from_iter([
        ("role".to_string(), serde_json::Value::String(role)),
        ("content".to_string(), content_value),
    ]))
}

fn translate_responses_input_to_messages(
    input: &serde_json::Value,
) -> Result<Vec<serde_json::Value>> {
    match input {
        serde_json::Value::String(text) => Ok(vec![serde_json::json!({
            "role": "user",
            "content": text,
        })]),
        serde_json::Value::Array(items) => {
            let looks_like_messages = items.iter().all(|item| {
                item.as_object()
                    .map(|object| object.contains_key("role") || object.contains_key("content"))
                    .unwrap_or(false)
            });
            if looks_like_messages {
                items
                    .iter()
                    .map(translate_responses_input_message)
                    .map(|result| result.map(serde_json::Value::Object))
                    .collect()
            } else {
                let content = translate_responses_message_content(input)?;
                Ok(vec![serde_json::json!({
                    "role": "user",
                    "content": content,
                })])
            }
        }
        serde_json::Value::Object(object) => {
            if object.contains_key("role") || object.contains_key("content") {
                Ok(vec![serde_json::Value::Object(
                    translate_responses_input_message(input)?,
                )])
            } else {
                let content = translate_responses_message_content(input)?;
                Ok(vec![serde_json::json!({
                    "role": "user",
                    "content": content,
                })])
            }
        }
        _ => bail!("unsupported /v1/responses input shape"),
    }
}

fn translate_openai_responses_input(
    object: &mut serde_json::Map<String, serde_json::Value>,
) -> Result<bool> {
    let mut changed = false;
    let mut messages = Vec::new();

    if let Some(instructions_value) = object.remove("instructions") {
        if let Some(instructions) = instructions_value.as_str().map(str::trim) {
            if !instructions.is_empty() {
                messages.push(serde_json::json!({
                    "role": "system",
                    "content": instructions,
                }));
            }
        }
        changed = true;
    }

    if let Some(input) = object.remove("input") {
        messages.extend(translate_responses_input_to_messages(&input)?);
        changed = true;
    } else if let Some(existing_messages) = object.remove("messages") {
        messages.extend(translate_responses_input_to_messages(&existing_messages)?);
        changed = true;
    }

    if !messages.is_empty() {
        object.insert("messages".to_string(), serde_json::Value::Array(messages));
    }

    for key in [
        "include",
        "output",
        "output_text",
        "parallel_tool_calls",
        "previous_response_id",
        "reasoning",
        "store",
        "text",
        "truncation",
    ] {
        if object.remove(key).is_some() {
            changed = true;
        }
    }

    Ok(changed)
}
