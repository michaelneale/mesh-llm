use tokio::io::AsyncWriteExt;
use tokio::net::TcpStream;

pub(super) fn http_body_text(raw: &[u8]) -> &str {
    let body_start = raw
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .map(|idx| idx + 4)
        .unwrap_or(raw.len());
    std::str::from_utf8(&raw[body_start..]).unwrap_or("")
}

pub(super) async fn respond_error(
    stream: &mut TcpStream,
    code: u16,
    msg: &str,
) -> anyhow::Result<()> {
    let body = serde_json::to_string(&serde_json::json!({"error": msg}))
        .unwrap_or_else(|_| r#"{"error":"internal error"}"#.to_string());
    let status = match code {
        400 => "Bad Request",
        404 => "Not Found",
        409 => "Conflict",
        422 => "Unprocessable Content",
        405 => "Method Not Allowed",
        500 => "Internal Server Error",
        502 => "Bad Gateway",
        503 => "Service Unavailable",
        _ => "Unknown",
    };
    let resp = format!(
        "HTTP/1.1 {code} {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        body.len(), body
    );
    stream.write_all(resp.as_bytes()).await?;
    Ok(())
}

pub(super) async fn respond_json<T: serde::Serialize>(
    stream: &mut TcpStream,
    code: u16,
    value: &T,
) -> anyhow::Result<()> {
    let json = serde_json::to_string(value)?;
    let status = match code {
        200 => "OK",
        201 => "Created",
        202 => "Accepted",
        400 => "Bad Request",
        403 => "Forbidden",
        404 => "Not Found",
        409 => "Conflict",
        429 => "Too Many Requests",
        500 => "Internal Server Error",
        503 => "Service Unavailable",
        _ => "OK",
    };
    let resp = format!(
        "HTTP/1.1 {code} {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        json.len(),
        json
    );
    stream.write_all(resp.as_bytes()).await?;
    Ok(())
}

pub(super) async fn respond_runtime_error(stream: &mut TcpStream, msg: &str) -> anyhow::Result<()> {
    respond_error(stream, crate::api::classify_runtime_error(msg), msg).await
}

pub(super) async fn respond_bytes(
    stream: &mut TcpStream,
    code: u16,
    status: &str,
    content_type: &str,
    body: &[u8],
) -> anyhow::Result<()> {
    respond_bytes_cached(stream, code, status, content_type, "no-cache", body).await
}

pub(super) async fn respond_bytes_cached(
    stream: &mut TcpStream,
    code: u16,
    status: &str,
    content_type: &str,
    cache_control: &str,
    body: &[u8],
) -> anyhow::Result<()> {
    let header = format!(
        "HTTP/1.1 {code} {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nCache-Control: {cache_control}\r\n\r\n",
        body.len()
    );
    stream.write_all(header.as_bytes()).await?;
    stream.write_all(body).await?;
    Ok(())
}
