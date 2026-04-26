use super::config::{ExternalPluginSpec, PluginHostMode};
use super::plugin_manifest_overview;
use super::support::{plugin_error, serialize_params, summarize_capabilities};
use super::transport::{bind_local_listener, connection_loop};
use super::{
    proto, PluginMeshEvent, PluginRpcBridge, PluginSummary, ToolCallResult, ToolSummary,
    CONNECT_TIMEOUT_SECS, PROTOCOL_VERSION, REQUEST_TIMEOUT_SECS,
};
use anyhow::{bail, Context, Result};
use mesh_llm_plugin::{MeshVisibility, STARTUP_DISABLED_ERROR_CODE};
use rmcp::model::{InitializeRequestParams, ServerInfo};
use serde::Serialize;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::process::{Child, Command};
use tokio::sync::{mpsc, oneshot, Mutex};

pub(crate) struct ExternalPlugin {
    spec: ExternalPluginSpec,
    instance_id: String,
    host_mode: PluginHostMode,
    summary: Arc<Mutex<PluginSummary>>,
    server_info: Arc<Mutex<Option<ServerInfo>>>,
    manifest: Arc<Mutex<Option<proto::PluginManifest>>>,
    runtime: Arc<Mutex<Option<PluginRuntime>>>,
    mesh_tx: mpsc::Sender<super::PluginMeshEvent>,
    rpc_bridge: Arc<Mutex<Option<Arc<dyn PluginRpcBridge>>>>,
    restart_lock: Arc<Mutex<()>>,
    next_request_id: AtomicU64,
    next_generation: AtomicU64,
}

pub(crate) struct PluginRuntime {
    pub(crate) generation: u64,
    pub(crate) _child: Child,
    pub(crate) outbound_tx: mpsc::Sender<proto::Envelope>,
    pub(crate) pending: Arc<Mutex<HashMap<u64, oneshot::Sender<Result<proto::Envelope>>>>>,
}

impl ExternalPlugin {
    pub(crate) async fn spawn(
        spec: &ExternalPluginSpec,
        instance_id: String,
        host_mode: PluginHostMode,
        mesh_tx: mpsc::Sender<PluginMeshEvent>,
        rpc_bridge: Arc<Mutex<Option<Arc<dyn PluginRpcBridge>>>>,
    ) -> Result<Self> {
        let plugin = Self {
            spec: spec.clone(),
            instance_id,
            host_mode,
            summary: Arc::new(Mutex::new(PluginSummary {
                name: spec.name.clone(),
                kind: "external".into(),
                enabled: true,
                status: "starting".into(),
                pid: None,
                version: None,
                capabilities: Vec::new(),
                command: Some(spec.command.clone()),
                args: spec.args.clone(),
                tools: Vec::new(),
                manifest: None,
                error: None,
            })),
            server_info: Arc::new(Mutex::new(None)),
            manifest: Arc::new(Mutex::new(None)),
            runtime: Arc::new(Mutex::new(None)),
            mesh_tx,
            rpc_bridge,
            restart_lock: Arc::new(Mutex::new(())),
            next_request_id: AtomicU64::new(1),
            next_generation: AtomicU64::new(1),
        };
        if let Err(err) = plugin.ensure_running().await {
            if plugin.is_disabled().await {
                return Ok(plugin);
            }
            return Err(err);
        }
        Ok(plugin)
    }

    pub(crate) fn name(&self) -> &str {
        &self.spec.name
    }

    pub(crate) async fn summary(&self) -> PluginSummary {
        let mut summary = self.summary.lock().await.clone();
        summary.manifest = self
            .manifest
            .lock()
            .await
            .as_ref()
            .map(plugin_manifest_overview);
        summary
    }

    pub(crate) async fn supervise(&self) -> Result<()> {
        if self.is_disabled().await {
            return Ok(());
        }
        if self.is_stopping().await {
            return Ok(());
        }
        self.ensure_running().await?;
        let response = self
            .request(proto::envelope::Payload::HealthRequest(
                proto::HealthRequest {},
            ))
            .await?;
        match response.payload {
            Some(proto::envelope::Payload::HealthResponse(resp))
                if resp.status == proto::health_response::Status::Ok as i32 =>
            {
                let mut summary = self.summary.lock().await;
                summary.status = "running".into();
                summary.error = None;
                Ok(())
            }
            Some(proto::envelope::Payload::HealthResponse(resp)) => {
                self.handle_runtime_failure(
                    None,
                    format!("health check reported status {}", resp.status),
                )
                .await;
                self.ensure_running().await
            }
            Some(proto::envelope::Payload::ErrorResponse(err)) => {
                self.handle_runtime_failure(None, err.message).await;
                self.ensure_running().await
            }
            _ => {
                self.handle_runtime_failure(None, "unexpected health payload".into())
                    .await;
                self.ensure_running().await
            }
        }
    }

    async fn ensure_running(&self) -> Result<()> {
        if let Some(reason) = self.disabled_reason().await {
            bail!("Plugin '{}' is disabled: {}", self.spec.name, reason);
        }
        if self.runtime.lock().await.is_some() {
            return Ok(());
        }
        let _guard = self.restart_lock.lock().await;
        if self.runtime.lock().await.is_some() {
            return Ok(());
        }

        {
            let mut summary = self.summary.lock().await;
            summary.status = "starting".into();
            summary.pid = None;
            summary.error = None;
        }

        let listener = bind_local_listener(&self.instance_id, &self.spec.name).await?;
        let endpoint = listener.endpoint();
        let transport = listener.transport_name();
        tracing::debug!(
            plugin = %self.spec.name,
            endpoint = %endpoint,
            transport,
            "Waiting for plugin connection"
        );

        let mut child = Command::new(&self.spec.command);
        child.args(&self.spec.args);
        child.env("MESH_LLM_PLUGIN_ENDPOINT", &endpoint);
        child.env("MESH_LLM_PLUGIN_TRANSPORT", transport);
        child.env("MESH_LLM_PLUGIN_NAME", &self.spec.name);
        if let Some(ref url) = self.spec.url {
            child.env("MESH_LLM_OPENAI_ENDPOINT_URL", url);
        }
        child.stdin(std::process::Stdio::null());
        child.stdout(std::process::Stdio::null());
        child.stderr(std::process::Stdio::inherit());
        child.kill_on_drop(true);

        let child = child.spawn().with_context(|| {
            format!(
                "Failed to launch plugin '{}' via {}",
                self.spec.name, self.spec.command
            )
        })?;
        let pid = child.id();
        self.summary.lock().await.pid = pid;

        let stream = tokio::time::timeout(
            std::time::Duration::from_secs(CONNECT_TIMEOUT_SECS),
            listener.accept(),
        )
        .await
        .with_context(|| format!("Timed out waiting for plugin '{}'", self.spec.name))??;

        let (outbound_tx, outbound_rx) = mpsc::channel(256);
        let pending = Arc::new(Mutex::new(HashMap::new()));
        let generation = self.next_generation.fetch_add(1, Ordering::Relaxed);
        let outbound_tx_for_runtime = outbound_tx.clone();
        *self.runtime.lock().await = Some(PluginRuntime {
            generation,
            _child: child,
            outbound_tx,
            pending: pending.clone(),
        });
        tokio::spawn(connection_loop(
            stream,
            outbound_rx,
            pending,
            self.mesh_tx.clone(),
            self.spec.name.clone(),
            self.summary.clone(),
            self.rpc_bridge.clone(),
            self.runtime.clone(),
            outbound_tx_for_runtime,
            generation,
        ));

        let (_, outbound_tx, pending) = self.runtime_handles().await?;
        let init_result: Result<proto::InitializeResponse> = async {
            let host_info_json = serde_json::to_string(&InitializeRequestParams::default())?;
            let response = self
                .request_once(
                    generation,
                    outbound_tx.clone(),
                    pending.clone(),
                    proto::envelope::Payload::InitializeRequest(proto::InitializeRequest {
                        host_protocol_version: PROTOCOL_VERSION,
                        host_version: crate::VERSION.to_string(),
                        host_info_json,
                        mesh_visibility: proto_mesh_visibility(self.host_mode.mesh_visibility),
                    }),
                )
                .await?;

            let init = match response.payload {
                Some(proto::envelope::Payload::InitializeResponse(resp)) => resp,
                Some(proto::envelope::Payload::ErrorResponse(err))
                    if err.code == STARTUP_DISABLED_ERROR_CODE =>
                {
                    self.mark_disabled(generation, err.message).await;
                    bail!("Plugin '{}' is disabled", self.spec.name);
                }
                Some(proto::envelope::Payload::ErrorResponse(err)) => {
                    bail!(
                        "Plugin '{}' rejected initialize: {}",
                        self.spec.name,
                        err.message
                    )
                }
                _ => bail!(
                    "Plugin '{}' returned an unexpected initialize payload",
                    self.spec.name
                ),
            };

            if init.plugin_id != self.spec.name {
                bail!(
                    "Plugin '{}' identified itself as '{}'",
                    self.spec.name,
                    init.plugin_id
                );
            }
            if init.plugin_protocol_version != PROTOCOL_VERSION {
                bail!(
                    "Plugin '{}' uses protocol {}, host uses {}",
                    self.spec.name,
                    init.plugin_protocol_version,
                    PROTOCOL_VERSION
                );
            }

            Ok(init)
        }
        .await;
        let init = match init_result {
            Ok(init) => init,
            Err(err) => {
                if self.is_disabled().await {
                    return Err(err);
                }
                self.handle_runtime_failure(
                    Some(generation),
                    format!("Plugin '{}' failed initialize: {err}", self.spec.name),
                )
                .await;
                return Err(err);
            }
        };

        let server_info: ServerInfo =
            serde_json::from_str(&init.server_info_json).with_context(|| {
                format!(
                    "Plugin '{}' returned invalid server_info_json",
                    self.spec.name
                )
            })?;
        *self.server_info.lock().await = Some(server_info.clone());
        *self.manifest.lock().await = init.manifest.clone();

        let tools = init
            .manifest
            .as_ref()
            .map(manifest_tool_summaries)
            .unwrap_or_default();
        let mut summary = self.summary.lock().await;
        summary.status = "running".into();
        summary.version = Some(init.plugin_version);
        let mut declared_capabilities = init.capabilities;
        if let Some(manifest) = init.manifest {
            declared_capabilities.extend(manifest.capabilities);
        }
        summary.capabilities = summarize_capabilities(&server_info, &declared_capabilities);
        summary.tools = tools;
        summary.error = None;
        Ok(())
    }

    pub(crate) async fn server_info(&self) -> Result<ServerInfo> {
        self.ensure_running().await?;
        self.server_info
            .lock()
            .await
            .clone()
            .with_context(|| format!("Plugin '{}' did not publish server info", self.spec.name))
    }

    pub(crate) async fn manifest(&self) -> Result<Option<proto::PluginManifest>> {
        self.ensure_running().await?;
        Ok(self.manifest.lock().await.clone())
    }

    pub(crate) async fn open_stream(
        &self,
        request: proto::OpenStreamRequest,
    ) -> Result<proto::OpenStreamResponse> {
        let response = self
            .request(proto::envelope::Payload::OpenStreamRequest(request))
            .await?;
        match response.payload {
            Some(proto::envelope::Payload::OpenStreamResponse(resp)) => Ok(resp),
            Some(proto::envelope::Payload::ErrorResponse(err)) => {
                Err(plugin_error(&self.spec.name, "open_stream", &err))
            }
            _ => bail!(
                "Plugin '{}' returned an unexpected payload for 'open_stream'",
                self.spec.name
            ),
        }
    }

    pub(crate) async fn list_tools(&self) -> Result<Vec<ToolSummary>> {
        Ok(self
            .manifest
            .lock()
            .await
            .clone()
            .map(|manifest| manifest_tool_summaries(&manifest))
            .unwrap_or_default())
    }

    pub(crate) async fn shutdown(&self) {
        {
            let mut summary = self.summary.lock().await;
            summary.status = "shutting down".into();
            summary.error = None;
        }

        let runtime = self.runtime.lock().await.take();
        if let Some(runtime) = runtime {
            let mut pending = runtime.pending.lock().await;
            for (_, response) in pending.drain() {
                let _ = response.send(Err(anyhow::anyhow!("plugin shutting down")));
            }
        }

        *self.server_info.lock().await = None;
        *self.manifest.lock().await = None;

        let mut summary = self.summary.lock().await;
        summary.status = "stopped".into();
        summary.pid = None;
        summary.version = None;
        summary.capabilities.clear();
        summary.tools.clear();
        summary.error = None;
    }

    pub(crate) async fn call_tool(
        &self,
        tool_name: &str,
        arguments_json: &str,
    ) -> Result<ToolCallResult> {
        let response = self
            .invoke_service(proto::ServiceKind::Operation, tool_name, arguments_json)
            .await?;
        Ok(ToolCallResult {
            content_json: response.output_json,
            is_error: response.is_error,
        })
    }

    pub(crate) async fn invoke_service(
        &self,
        kind: proto::ServiceKind,
        service_name: &str,
        input_json: &str,
    ) -> Result<proto::InvokeServiceResponse> {
        let response = self
            .request(proto::envelope::Payload::InvokeServiceRequest(
                proto::InvokeServiceRequest {
                    kind: kind as i32,
                    service_name: service_name.to_string(),
                    input_json: input_json.to_string(),
                },
            ))
            .await?;
        match response.payload {
            Some(proto::envelope::Payload::InvokeServiceResponse(resp)) => Ok(resp),
            Some(proto::envelope::Payload::ErrorResponse(err)) => {
                Err(plugin_error(&self.spec.name, "invoke_service", &err))
            }
            _ => bail!(
                "Plugin '{}' returned an unexpected payload for 'invoke_service'",
                self.spec.name
            ),
        }
    }

    pub(crate) async fn mcp_request<T, P>(&self, method: &str, params: P) -> Result<T>
    where
        T: serde::de::DeserializeOwned,
        P: Serialize,
    {
        let params_json = serialize_params(params)?;
        let response = self
            .request(proto::envelope::Payload::RpcRequest(proto::RpcRequest {
                method: method.to_string(),
                params_json,
            }))
            .await?;
        match response.payload {
            Some(proto::envelope::Payload::RpcResponse(resp)) => {
                serde_json::from_str(&resp.result_json).with_context(|| {
                    format!(
                        "Plugin '{}' returned invalid result for '{}'",
                        self.spec.name, method
                    )
                })
            }
            Some(proto::envelope::Payload::ErrorResponse(err)) => {
                Err(plugin_error(&self.spec.name, method, &err))
            }
            _ => bail!(
                "Plugin '{}' returned an unexpected RPC payload for '{}'",
                self.spec.name,
                method
            ),
        }
    }

    pub(crate) async fn mcp_notify<P>(&self, method: &str, params: P) -> Result<()>
    where
        P: Serialize,
    {
        self.send_unsolicited(
            proto::envelope::Payload::RpcNotification(proto::RpcNotification {
                method: method.to_string(),
                params_json: serialize_params(params)?,
            }),
            method,
        )
        .await
    }

    pub(crate) async fn send_channel_message(&self, message: proto::ChannelMessage) -> Result<()> {
        self.send_unsolicited(
            proto::envelope::Payload::ChannelMessage(message),
            "messages",
        )
        .await
    }

    pub(crate) async fn send_bulk_transfer_message(
        &self,
        message: proto::BulkTransferMessage,
    ) -> Result<()> {
        self.send_unsolicited(
            proto::envelope::Payload::BulkTransferMessage(message),
            "bulk transfers",
        )
        .await
    }

    pub(crate) async fn send_mesh_event(&self, event: proto::MeshEvent) -> Result<()> {
        self.send_unsolicited(proto::envelope::Payload::MeshEvent(event), "mesh events")
            .await
    }

    async fn request(&self, payload: proto::envelope::Payload) -> Result<proto::Envelope> {
        for attempt in 0..2 {
            self.ensure_running().await?;
            let (generation, outbound_tx, pending) = self.runtime_handles().await?;
            match self
                .request_once(generation, outbound_tx, pending, payload.clone())
                .await
            {
                Ok(response) => return Ok(response),
                Err(err) if attempt == 0 => {
                    tracing::debug!(
                        plugin = %self.spec.name,
                        error = %err,
                        "Retrying plugin request after restart"
                    );
                }
                Err(err) => return Err(err),
            }
        }
        bail!("Plugin '{}' request failed after restart", self.spec.name)
    }

    async fn send_unsolicited(&self, payload: proto::envelope::Payload, kind: &str) -> Result<()> {
        for attempt in 0..2 {
            self.ensure_running().await?;
            let (generation, outbound_tx, _) = self.runtime_handles().await?;
            let envelope = proto::Envelope {
                protocol_version: PROTOCOL_VERSION,
                plugin_id: self.spec.name.clone(),
                request_id: 0,
                payload: Some(payload.clone()),
            };
            if outbound_tx.send(envelope).await.is_ok() {
                return Ok(());
            }
            self.handle_runtime_failure(
                Some(generation),
                format!("Plugin '{}' is not accepting {kind}", self.spec.name),
            )
            .await;
            if attempt == 1 {
                break;
            }
        }
        bail!("Plugin '{}' is not accepting {}", self.spec.name, kind)
    }

    async fn runtime_handles(
        &self,
    ) -> Result<(
        u64,
        mpsc::Sender<proto::Envelope>,
        Arc<Mutex<HashMap<u64, oneshot::Sender<Result<proto::Envelope>>>>>,
    )> {
        let runtime = self.runtime.lock().await;
        let runtime = runtime
            .as_ref()
            .with_context(|| format!("Plugin '{}' is not running", self.spec.name))?;
        Ok((
            runtime.generation,
            runtime.outbound_tx.clone(),
            runtime.pending.clone(),
        ))
    }

    async fn request_once(
        &self,
        generation: u64,
        outbound_tx: mpsc::Sender<proto::Envelope>,
        pending: Arc<Mutex<HashMap<u64, oneshot::Sender<Result<proto::Envelope>>>>>,
        payload: proto::envelope::Payload,
    ) -> Result<proto::Envelope> {
        let request_id = self.next_request_id.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = oneshot::channel();
        pending.lock().await.insert(request_id, tx);

        let envelope = proto::Envelope {
            protocol_version: PROTOCOL_VERSION,
            plugin_id: self.spec.name.clone(),
            request_id,
            payload: Some(payload),
        };

        if let Err(_send_err) = outbound_tx.send(envelope).await {
            pending.lock().await.remove(&request_id);
            self.handle_runtime_failure(
                Some(generation),
                format!("Plugin '{}' is not accepting requests", self.spec.name),
            )
            .await;
            bail!("Plugin '{}' is not accepting requests", self.spec.name);
        }

        match tokio::time::timeout(std::time::Duration::from_secs(REQUEST_TIMEOUT_SECS), rx).await {
            Ok(Ok(resp)) => resp,
            Ok(Err(_)) => {
                self.handle_runtime_failure(
                    Some(generation),
                    format!("Plugin '{}' dropped the response channel", self.spec.name),
                )
                .await;
                bail!("Plugin '{}' dropped the response channel", self.spec.name);
            }
            Err(_) => {
                pending.lock().await.remove(&request_id);
                self.handle_runtime_failure(
                    Some(generation),
                    format!("Plugin '{}' timed out", self.spec.name),
                )
                .await;
                bail!("Plugin '{}' timed out", self.spec.name);
            }
        }
    }

    async fn handle_runtime_failure(&self, generation: Option<u64>, reason: String) {
        let mut runtime = self.runtime.lock().await;
        let should_clear = generation
            .map(|generation| runtime.as_ref().map(|r| r.generation) == Some(generation))
            .unwrap_or(true);
        if should_clear {
            *runtime = None;
        }
        drop(runtime);
        let mut summary = self.summary.lock().await;
        summary.status = "restarting".into();
        summary.pid = None;
        summary.error = Some(reason);
    }

    async fn disabled_reason(&self) -> Option<String> {
        let summary = self.summary.lock().await;
        if summary.status == "disabled" {
            Some(
                summary
                    .error
                    .clone()
                    .unwrap_or_else(|| "disabled".to_string()),
            )
        } else {
            None
        }
    }

    async fn is_disabled(&self) -> bool {
        self.disabled_reason().await.is_some()
    }

    async fn is_stopping(&self) -> bool {
        let summary = self.summary.lock().await;
        matches!(summary.status.as_str(), "shutting down" | "stopped")
    }

    async fn mark_disabled(&self, generation: u64, reason: String) {
        let mut runtime = self.runtime.lock().await;
        if runtime.as_ref().map(|runtime| runtime.generation) == Some(generation) {
            *runtime = None;
        }
        drop(runtime);

        let mut server_info = self.server_info.lock().await;
        *server_info = None;
        drop(server_info);

        let mut manifest = self.manifest.lock().await;
        *manifest = None;
        drop(manifest);

        let mut summary = self.summary.lock().await;
        summary.enabled = false;
        summary.status = "disabled".into();
        summary.pid = None;
        summary.version = None;
        summary.capabilities.clear();
        summary.tools.clear();
        summary.error = Some(reason);
    }
}

fn manifest_tool_summaries(manifest: &proto::PluginManifest) -> Vec<ToolSummary> {
    manifest
        .operations
        .iter()
        .map(|operation| ToolSummary {
            name: operation.name.clone(),
            description: operation.description.clone(),
            input_schema_json: operation.input_schema_json.clone(),
        })
        .collect()
}

fn proto_mesh_visibility(mesh_visibility: MeshVisibility) -> i32 {
    match mesh_visibility {
        MeshVisibility::Private => proto::MeshVisibility::Private as i32,
        MeshVisibility::Public => proto::MeshVisibility::Public as i32,
    }
}
