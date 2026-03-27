use anyhow::{bail, Context, Result};
use prost::Message;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

use crate::{proto, PROTOCOL_VERSION};

pub enum LocalStream {
    #[cfg(unix)]
    Unix(tokio::net::UnixStream),
    #[cfg(windows)]
    PipeClient(tokio::net::windows::named_pipe::NamedPipeClient),
}

impl LocalStream {
    async fn write_all(&mut self, bytes: &[u8]) -> Result<()> {
        match self {
            #[cfg(unix)]
            LocalStream::Unix(stream) => stream.write_all(bytes).await?,
            #[cfg(windows)]
            LocalStream::PipeClient(stream) => stream.write_all(bytes).await?,
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
        }
        Ok(())
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
