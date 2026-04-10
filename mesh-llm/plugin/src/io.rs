use anyhow::{bail, Context, Result};
use prost::Message;
use std::path::PathBuf;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

use crate::{proto, PROTOCOL_VERSION};

pub enum LocalStream {
    #[cfg(unix)]
    Unix(tokio::net::UnixStream),
    #[cfg(windows)]
    PipeClient(tokio::net::windows::named_pipe::NamedPipeClient),
    #[cfg(windows)]
    PipeServer(tokio::net::windows::named_pipe::NamedPipeServer),
}

impl LocalStream {
    async fn write_all(&mut self, bytes: &[u8]) -> Result<()> {
        match self {
            #[cfg(unix)]
            LocalStream::Unix(stream) => stream.write_all(bytes).await?,
            #[cfg(windows)]
            LocalStream::PipeClient(stream) => stream.write_all(bytes).await?,
            #[cfg(windows)]
            LocalStream::PipeServer(stream) => stream.write_all(bytes).await?,
        }
        Ok(())
    }

    async fn read_exact(&mut self, bytes: &mut [u8]) -> Result<()> {
        match self {
            #[cfg(unix)]
            LocalStream::Unix(stream) => {
                let _ = stream.read_exact(bytes).await?;
            }
            #[cfg(windows)]
            LocalStream::PipeClient(stream) => {
                let _ = stream.read_exact(bytes).await?;
            }
            #[cfg(windows)]
            LocalStream::PipeServer(stream) => {
                let _ = stream.read_exact(bytes).await?;
            }
        }
        Ok(())
    }

    pub async fn write_all_bytes(&mut self, bytes: &[u8]) -> Result<()> {
        self.write_all(bytes).await
    }

    pub async fn read_exact_bytes(&mut self, bytes: &mut [u8]) -> Result<()> {
        self.read_exact(bytes).await
    }
}

pub enum LocalListener {
    #[cfg(unix)]
    Unix(tokio::net::UnixListener, PathBuf),
    #[cfg(windows)]
    Pipe(String, tokio::net::windows::named_pipe::NamedPipeServer),
}

impl LocalListener {
    pub async fn accept(self) -> Result<LocalStream> {
        match self {
            #[cfg(unix)]
            LocalListener::Unix(listener, path) => {
                let (stream, _) = listener.accept().await?;
                let _ = std::fs::remove_file(path);
                Ok(LocalStream::Unix(stream))
            }
            #[cfg(windows)]
            LocalListener::Pipe(_name, server) => {
                server.connect().await?;
                Ok(LocalStream::PipeServer(server))
            }
        }
    }

    pub fn endpoint(&self) -> String {
        match self {
            #[cfg(unix)]
            LocalListener::Unix(_, path) => path.display().to_string(),
            #[cfg(windows)]
            LocalListener::Pipe(name, _) => name.clone(),
        }
    }

    pub fn transport_kind(&self) -> i32 {
        #[cfg(unix)]
        {
            proto::StreamTransportKind::StreamUnixSocket as i32
        }
        #[cfg(windows)]
        {
            proto::StreamTransportKind::StreamNamedPipe as i32
        }
    }

    pub fn open_stream_response(
        &self,
        request: &proto::OpenStreamRequest,
    ) -> proto::OpenStreamResponse {
        proto::OpenStreamResponse {
            stream_id: request.stream_id.clone(),
            accepted: true,
            transport_kind: self.transport_kind(),
            endpoint: Some(self.endpoint()),
            token: None,
            expires_at_unix_ms: None,
            message: None,
        }
    }
}

pub async fn connect_from_env() -> Result<LocalStream> {
    let endpoint = std::env::var("MESH_LLM_PLUGIN_ENDPOINT")
        .context("MESH_LLM_PLUGIN_ENDPOINT is not set for plugin process")?;
    let transport =
        std::env::var("MESH_LLM_PLUGIN_TRANSPORT").unwrap_or_else(|_| default_transport().into());

    match transport.as_str() {
        #[cfg(unix)]
        "unix" => Ok(LocalStream::Unix(
            tokio::net::UnixStream::connect(&endpoint).await?,
        )),
        #[cfg(windows)]
        "pipe" => Ok(LocalStream::PipeClient(
            tokio::net::windows::named_pipe::ClientOptions::new().open(&endpoint)?,
        )),
        _ => bail!("Unsupported plugin transport '{transport}'"),
    }
}

pub async fn bind_side_stream(plugin_id: &str, stream_id: &str) -> Result<LocalListener> {
    #[cfg(unix)]
    {
        let path = std::env::temp_dir().join(format!(
            "mesh-llm-side-{}-{}.sock",
            sanitize_component(plugin_id),
            sanitize_component(stream_id)
        ));
        if path.exists() {
            let _ = std::fs::remove_file(&path);
        }
        let listener = tokio::net::UnixListener::bind(&path)
            .with_context(|| format!("Failed to bind side stream socket {}", path.display()))?;
        Ok(LocalListener::Unix(listener, path))
    }
    #[cfg(windows)]
    {
        let endpoint = format!(
            r"\\.\pipe\mesh-llm-side-{}-{}",
            sanitize_component(plugin_id),
            sanitize_component(stream_id)
        );
        let server = tokio::net::windows::named_pipe::ServerOptions::new()
            .create(&endpoint)
            .with_context(|| format!("Failed to create side stream pipe {endpoint}"))?;
        Ok(LocalListener::Pipe(endpoint, server))
    }
}

pub async fn write_envelope(stream: &mut LocalStream, envelope: &proto::Envelope) -> Result<()> {
    let mut body = Vec::new();
    envelope.encode(&mut body)?;
    stream.write_all(&(body.len() as u32).to_le_bytes()).await?;
    stream.write_all(&body).await?;
    Ok(())
}

pub async fn read_envelope(stream: &mut LocalStream) -> Result<proto::Envelope> {
    let mut len_buf = [0u8; 4];
    stream.read_exact(&mut len_buf).await?;
    let len = u32::from_le_bytes(len_buf) as usize;
    if len > 16 * 1024 * 1024 {
        bail!("Plugin frame too large");
    }
    let mut body = vec![0u8; len];
    stream.read_exact(&mut body).await?;
    Ok(proto::Envelope::decode(body.as_slice())?)
}

pub async fn send_channel_message(
    stream: &mut LocalStream,
    plugin_id: &str,
    message: proto::ChannelMessage,
) -> Result<()> {
    write_envelope(
        stream,
        &proto::Envelope {
            protocol_version: PROTOCOL_VERSION,
            plugin_id: plugin_id.to_string(),
            request_id: 0,
            payload: Some(proto::envelope::Payload::ChannelMessage(message)),
        },
    )
    .await
}

pub async fn send_bulk_transfer_message(
    stream: &mut LocalStream,
    plugin_id: &str,
    message: proto::BulkTransferMessage,
) -> Result<()> {
    write_envelope(
        stream,
        &proto::Envelope {
            protocol_version: PROTOCOL_VERSION,
            plugin_id: plugin_id.to_string(),
            request_id: 0,
            payload: Some(proto::envelope::Payload::BulkTransferMessage(message)),
        },
    )
    .await
}

fn default_transport() -> &'static str {
    #[cfg(unix)]
    {
        "unix"
    }
    #[cfg(windows)]
    {
        "pipe"
    }
}

fn sanitize_component(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
                ch
            } else {
                '_'
            }
        })
        .collect()
}
