use super::runtime::PluginRuntime;
use super::{
    PluginInferenceEvent, PluginMeshEvent, PluginRpcBridge, PluginSummary, PROTOCOL_VERSION,
};
use anyhow::{anyhow, bail, Context, Result};
use rand::Rng;
use rmcp::model::ErrorCode;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::sync::{mpsc, oneshot, Mutex};

pub(crate) enum LocalStream {
    #[cfg(unix)]
    Unix(tokio::net::UnixStream),
    #[cfg(windows)]
    PipeServer(tokio::net::windows::named_pipe::NamedPipeServer),
}

pub(crate) enum LocalListener {
    #[cfg(unix)]
    Unix(tokio::net::UnixListener, PathBuf),
    #[cfg(windows)]
    Pipe(String, tokio::net::windows::named_pipe::NamedPipeServer),
}

pub(crate) async fn connection_loop(
    mut stream: LocalStream,
    mut outbound_rx: mpsc::Receiver<super::proto::Envelope>,
    pending: Arc<Mutex<HashMap<u64, oneshot::Sender<Result<super::proto::Envelope>>>>>,
    mesh_tx: mpsc::Sender<PluginMeshEvent>,
    plugin_name: String,
    summary: Arc<Mutex<PluginSummary>>,
    rpc_bridge: Arc<Mutex<Option<Arc<dyn PluginRpcBridge>>>>,
    runtime: Arc<Mutex<Option<PluginRuntime>>>,
    outbound_tx: mpsc::Sender<super::proto::Envelope>,
    generation: u64,
    inference_tx: Option<mpsc::Sender<PluginInferenceEvent>>,
) {
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
                            if let Some(event) = try_parse_inference_notification(
                                &plugin_name,
                                &notification,
                            ) {
                                if let Some(ref tx) = inference_tx {
                                    let tx = tx.clone();
                                    tokio::spawn(async move {
                                        let _ = tx.send(event).await;
                                    });
                                }
                            } else {
                                forward_plugin_notification(
                                    plugin_name.clone(),
                                    notification,
                                    rpc_bridge.clone(),
                                );
                            }
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
}

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
    async fn write_all(&mut self, bytes: &[u8]) -> Result<()> {
        match self {
            #[cfg(unix)]
            LocalStream::Unix(stream) => stream.write_all(bytes).await?,
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
            LocalStream::PipeServer(stream) => {
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
        return Ok(LocalListener::Unix(listener, path));
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
fn windows_pipe_name(instance_id: &str, name: &str) -> String {
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

/// Check if a plugin notification is an inference registration/unregistration.
/// Returns Some(event) if it is, None otherwise (let it go to the RPC bridge).
fn try_parse_inference_notification(
    plugin_name: &str,
    notification: &super::proto::RpcNotification,
) -> Option<PluginInferenceEvent> {
    match notification.method.as_str() {
        "inference/register" => {
            let params: serde_json::Value = serde_json::from_str(&notification.params_json).ok()?;
            let model = params.get("model")?.as_str()?.to_string();
            let port = params.get("port")?.as_u64()? as u16;
            let backend = params
                .get("backend")
                .and_then(|b| b.as_str())
                .unwrap_or("plugin")
                .to_string();
            tracing::info!(
                plugin = %plugin_name,
                model = %model,
                port = port,
                backend = %backend,
                "Plugin registering inference backend"
            );
            Some(PluginInferenceEvent::Register {
                plugin_id: plugin_name.to_string(),
                model,
                port,
                backend,
            })
        }
        "inference/unregister" => {
            let params: serde_json::Value = serde_json::from_str(&notification.params_json).ok()?;
            let model = params.get("model")?.as_str()?.to_string();
            let port = params.get("port")?.as_u64()? as u16;
            tracing::info!(
                plugin = %plugin_name,
                model = %model,
                port = port,
                "Plugin unregistering inference backend"
            );
            Some(PluginInferenceEvent::Unregister {
                plugin_id: plugin_name.to_string(),
                model,
                port,
            })
        }
        _ => None,
    }
}
