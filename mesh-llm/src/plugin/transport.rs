use super::runtime::PluginRuntime;
use super::{proto, PluginMeshEvent, PluginRpcBridge, PluginSummary, PROTOCOL_VERSION};
use anyhow::{anyhow, bail, Context, Result};
use rand::Rng;
use rmcp::model::ErrorCode;
use std::collections::HashMap;
use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::sync::{mpsc, oneshot, Mutex};

pub(crate) enum LocalStream {
    #[cfg(unix)]
    Unix(tokio::net::UnixStream),
    #[cfg(windows)]
    PipeServer(tokio::net::windows::named_pipe::NamedPipeServer),
    #[cfg(windows)]
    PipeClient(tokio::net::windows::named_pipe::NamedPipeClient),
}

pub(crate) enum LocalListener {
    #[cfg(unix)]
    Unix(tokio::net::UnixListener, PathBuf),
    #[cfg(windows)]
    Pipe(String, tokio::net::windows::named_pipe::NamedPipeServer),
}

type ConnectionLoopFn = fn(
    LocalStream,
    mpsc::Receiver<super::proto::Envelope>,
    Arc<Mutex<HashMap<u64, oneshot::Sender<Result<super::proto::Envelope>>>>>,
    mpsc::Sender<PluginMeshEvent>,
    String,
    Arc<Mutex<PluginSummary>>,
    Arc<Mutex<Option<Arc<dyn PluginRpcBridge>>>>,
    Arc<Mutex<Option<PluginRuntime>>>,
    mpsc::Sender<super::proto::Envelope>,
    u64,
) -> Pin<Box<dyn Future<Output = ()> + Send>>;

pub(crate) const CONNECTION_LOOP: ConnectionLoopFn =
    |mut stream,
     mut outbound_rx,
     pending,
     mesh_tx,
     plugin_name,
     summary,
     rpc_bridge,
     runtime,
     outbound_tx,
     generation| {
        Box::pin(async move {
            let result: Result<()> = async {
                loop {
                    tokio::select! {
                        maybe_outbound = outbound_rx.recv() => {
                            let Some(envelope) = maybe_outbound else {
                                break;
                            };
                            write_envelope(&mut stream, &envelope).await?;
                        }
                        inbound = read_envelope(&mut stream) => {
                            let envelope = inbound?;
                            let request_id = envelope.request_id;
                            let plugin_id_from_env = envelope.plugin_id.clone();
                            let payload = envelope.payload.clone();
                            match payload {
                                Some(super::proto::envelope::Payload::ChannelMessage(message)) => {
                                    let plugin_id = if plugin_id_from_env.is_empty() {
                                        plugin_name.clone()
                                    } else {
                                        plugin_id_from_env
                                    };
                                    let _ = mesh_tx
                                        .send(PluginMeshEvent::Channel { plugin_id, message })
                                        .await;
                                }
                                Some(super::proto::envelope::Payload::BulkTransferMessage(message)) => {
                                    let plugin_id = if plugin_id_from_env.is_empty() {
                                        plugin_name.clone()
                                    } else {
                                        plugin_id_from_env
                                    };
                                    let _ = mesh_tx
                                        .send(PluginMeshEvent::BulkTransfer {
                                            plugin_id,
                                            message,
                                        })
                                        .await;
                                }
                                Some(super::proto::envelope::Payload::RpcRequest(request)) => {
                                    forward_plugin_request(
                                        plugin_name.clone(),
                                        request_id,
                                        request,
                                        rpc_bridge.clone(),
                                        outbound_tx.clone(),
                                    );
                                }
                                Some(super::proto::envelope::Payload::RpcNotification(notification)) => {
                                    forward_plugin_notification(
                                        plugin_name.clone(),
                                        notification,
                                        rpc_bridge.clone(),
                                    );
                                }
                                _ => {
                                    let responder = pending.lock().await.remove(&request_id);
                                    if let Some(responder) = responder {
                                        let _ = responder.send(Ok(envelope));
                                    } else {
                                        tracing::debug!(
                                            "Plugin '{}' sent an unsolicited response id={}",
                                            plugin_name,
                                            request_id
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
                Ok(())
            }
            .await;

            if let Err(err) = result {
                tracing::warn!(
                    plugin = %plugin_name,
                    error = %err,
                    "Plugin connection closed"
                );
            }

            {
                let mut runtime = runtime.lock().await;
                if runtime.as_ref().map(|runtime| runtime.generation) == Some(generation) {
                    *runtime = None;
                    let mut summary = summary.lock().await;
                    summary.status = "stopped".into();
                    summary.error = Some(format!("Plugin '{}' disconnected", plugin_name));
                }
            }

            let mut pending = pending.lock().await;
            for (_, responder) in pending.drain() {
                let _ = responder.send(Err(anyhow!("Plugin '{}' disconnected", plugin_name)));
            }
        })
    };

pub(crate) use CONNECTION_LOOP as connection_loop;

impl LocalListener {
    pub(crate) async fn accept(self) -> Result<LocalStream> {
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

    pub(crate) fn endpoint(&self) -> String {
        match self {
            #[cfg(unix)]
            LocalListener::Unix(_, path) => path.display().to_string(),
            #[cfg(windows)]
            LocalListener::Pipe(name, _) => name.clone(),
        }
    }

    pub(crate) fn transport_name(&self) -> &'static str {
        #[cfg(unix)]
        {
            "unix"
        }
        #[cfg(windows)]
        {
            "pipe"
        }
    }
}

impl LocalStream {
    pub(crate) async fn write_all(&mut self, bytes: &[u8]) -> Result<()> {
        match self {
            #[cfg(unix)]
            LocalStream::Unix(stream) => stream.write_all(bytes).await?,
            #[cfg(windows)]
            LocalStream::PipeServer(stream) => stream.write_all(bytes).await?,
            #[cfg(windows)]
            LocalStream::PipeClient(stream) => stream.write_all(bytes).await?,
        }
        Ok(())
    }

    pub(crate) async fn shutdown(&mut self) -> Result<()> {
        match self {
            #[cfg(unix)]
            LocalStream::Unix(stream) => stream.shutdown().await?,
            #[cfg(windows)]
            LocalStream::PipeServer(stream) => stream.shutdown().await?,
            #[cfg(windows)]
            LocalStream::PipeClient(stream) => stream.shutdown().await?,
        }
        Ok(())
    }

    pub(crate) async fn read(&mut self, bytes: &mut [u8]) -> Result<usize> {
        let read = match self {
            #[cfg(unix)]
            LocalStream::Unix(stream) => stream.read(bytes).await?,
            #[cfg(windows)]
            LocalStream::PipeServer(stream) => stream.read(bytes).await?,
            #[cfg(windows)]
            LocalStream::PipeClient(stream) => stream.read(bytes).await?,
        };
        Ok(read)
    }

    async fn read_exact(&mut self, bytes: &mut [u8]) -> Result<()> {
        match self {
            #[cfg(unix)]
            LocalStream::Unix(stream) => {
                let _ = stream.read_exact(bytes).await?;
            }
            #[cfg(windows)]
            LocalStream::PipeServer(stream) => {
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

pub(crate) async fn bind_local_listener(instance_id: &str, name: &str) -> Result<LocalListener> {
    #[cfg(unix)]
    {
        let path = unix_socket_path(instance_id, name)?;
        let dir = path
            .parent()
            .context("Plugin socket path is missing a parent directory")?;
        std::fs::create_dir_all(dir)
            .with_context(|| format!("Failed to create plugin runtime dir {}", dir.display()))?;
        if path.exists() {
            let _ = std::fs::remove_file(&path);
        }
        let listener = tokio::net::UnixListener::bind(&path)
            .with_context(|| format!("Failed to bind plugin socket {}", path.display()))?;
        Ok(LocalListener::Unix(listener, path))
    }
    #[cfg(windows)]
    {
        let endpoint = windows_pipe_name(instance_id, name);
        let server = tokio::net::windows::named_pipe::ServerOptions::new()
            .create(&endpoint)
            .with_context(|| format!("Failed to create plugin pipe {endpoint}"))?;
        return Ok(LocalListener::Pipe(endpoint, server));
    }
}

pub(crate) async fn connect_side_stream(
    endpoint: &str,
    transport_kind: i32,
) -> Result<LocalStream> {
    match proto::StreamTransportKind::try_from(transport_kind)
        .unwrap_or(proto::StreamTransportKind::Unspecified)
    {
        #[cfg(unix)]
        proto::StreamTransportKind::StreamUnixSocket => Ok(LocalStream::Unix(
            tokio::net::UnixStream::connect(endpoint)
                .await
                .with_context(|| format!("Failed to connect side stream socket {endpoint}"))?,
        )),
        #[cfg(windows)]
        proto::StreamTransportKind::StreamNamedPipe => Ok(LocalStream::PipeClient(
            tokio::net::windows::named_pipe::ClientOptions::new()
                .open(endpoint)
                .with_context(|| format!("Failed to connect side stream pipe {endpoint}"))?,
        )),
        _ => bail!(
            "Unsupported side stream transport kind '{}'",
            transport_kind
        ),
    }
}

#[cfg(unix)]
fn runtime_dir() -> Result<PathBuf> {
    let home = dirs::home_dir().context("Cannot determine home directory")?;
    Ok(home.join(".mesh-llm").join("run").join("plugins"))
}

pub(crate) fn make_instance_id() -> String {
    let pid = std::process::id();
    let random = rand::rng().random::<u32>();
    format!("p{pid}-{random:08x}")
}

#[cfg(unix)]
pub(crate) fn unix_socket_path(instance_id: &str, name: &str) -> Result<PathBuf> {
    Ok(runtime_dir()?.join(format!("{instance_id}-{name}.sock")))
}

#[cfg(windows)]
pub(crate) fn windows_pipe_name(instance_id: &str, name: &str) -> String {
    format!(r"\\.\pipe\mesh-llm-{instance_id}-{name}")
}

pub(crate) async fn write_envelope(
    stream: &mut LocalStream,
    envelope: &super::proto::Envelope,
) -> Result<()> {
    let mut body = Vec::new();
    prost::Message::encode(envelope, &mut body)?;
    stream.write_all(&(body.len() as u32).to_le_bytes()).await?;
    stream.write_all(&body).await?;
    Ok(())
}

pub(crate) async fn read_envelope(stream: &mut LocalStream) -> Result<super::proto::Envelope> {
    let mut len_buf = [0u8; 4];
    stream.read_exact(&mut len_buf).await?;
    let len = u32::from_le_bytes(len_buf) as usize;
    if len > 16 * 1024 * 1024 {
        bail!("Plugin frame too large");
    }
    let mut body = vec![0u8; len];
    stream.read_exact(&mut body).await?;
    Ok(prost::Message::decode(body.as_slice())?)
}

fn forward_plugin_request(
    plugin_name: String,
    request_id: u64,
    request: super::proto::RpcRequest,
    rpc_bridge: Arc<Mutex<Option<Arc<dyn PluginRpcBridge>>>>,
    outbound_tx: mpsc::Sender<super::proto::Envelope>,
) {
    tokio::spawn(async move {
        let bridge = rpc_bridge.lock().await.clone();
        let payload = match bridge {
            Some(bridge) => match bridge
                .handle_request(
                    plugin_name.clone(),
                    request.method.clone(),
                    request.params_json.clone(),
                )
                .await
            {
                Ok(result) => {
                    super::proto::envelope::Payload::RpcResponse(super::proto::RpcResponse {
                        result_json: result.result_json,
                    })
                }
                Err(err) => super::proto::envelope::Payload::ErrorResponse(err),
            },
            None => super::proto::envelope::Payload::ErrorResponse(super::proto::ErrorResponse {
                code: ErrorCode::INTERNAL_ERROR.0,
                message: "No active MCP bridge".into(),
                data_json: String::new(),
            }),
        };

        let _ = outbound_tx
            .send(super::proto::Envelope {
                protocol_version: PROTOCOL_VERSION,
                plugin_id: plugin_name,
                request_id,
                payload: Some(payload),
            })
            .await;
    });
}

fn forward_plugin_notification(
    plugin_name: String,
    notification: super::proto::RpcNotification,
    rpc_bridge: Arc<Mutex<Option<Arc<dyn PluginRpcBridge>>>>,
) {
    tokio::spawn(async move {
        if let Some(bridge) = rpc_bridge.lock().await.clone() {
            bridge
                .handle_notification(plugin_name, notification.method, notification.params_json)
                .await;
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(unix)]
    #[tokio::test]
    async fn host_can_connect_to_plugin_side_stream() {
        let request = proto::OpenStreamRequest {
            stream_id: "stream-test".into(),
            purpose: proto::StreamPurpose::HttpResponseBody as i32,
            mode: proto::StreamMode::RawBytes as i32,
            bidirectional: true,
            content_type: Some("application/octet-stream".into()),
            correlation_id: None,
            metadata_json: None,
            expected_bytes: None,
            idle_timeout_ms: None,
        };

        let listener = mesh_llm_plugin::bind_side_stream("demo-plugin", &request.stream_id)
            .await
            .unwrap();
        let response = listener.open_stream_response(&request);

        let accept_task = tokio::spawn(async move {
            let mut plugin_stream = listener.accept().await.unwrap();
            let mut incoming = [0u8; 5];
            plugin_stream.read_exact_bytes(&mut incoming).await.unwrap();
            assert_eq!(&incoming, b"hello");
            plugin_stream.write_all_bytes(b"world").await.unwrap();
        });

        let mut host_stream = connect_side_stream(
            response.endpoint.as_deref().unwrap(),
            response.transport_kind,
        )
        .await
        .unwrap();
        host_stream.write_all(b"hello").await.unwrap();
        let mut reply = [0u8; 5];
        host_stream.read_exact(&mut reply).await.unwrap();
        assert_eq!(&reply, b"world");

        accept_task.await.unwrap();
    }
}
