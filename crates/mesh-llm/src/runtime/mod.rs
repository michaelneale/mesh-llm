pub(crate) mod config_state;
mod discovery;
pub mod instance;
mod interactive;
mod local;
mod proxy;
pub(crate) mod wakeable;

use self::discovery::{nostr_rediscovery, start_new_mesh};
use self::interactive::InitialPromptMode;
use self::local::{
    add_runtime_local_target, add_serving_assignment, advertise_model_ready, local_process_payload,
    remove_runtime_local_target, remove_serving_assignment, resolved_model_name,
    set_advertised_model_context, start_runtime_local_model, withdraw_advertised_model,
    LocalRuntimeModelHandle, LocalRuntimeModelStartSpec, ManagedModelController, RuntimeEvent,
};
use self::proxy::{api_proxy, bootstrap_proxy};
use crate::api;
use crate::cli::output::{
    emit_event, ConsoleSessionMode, DashboardAcceptedRequestBucket, DashboardEndpointRow,
    DashboardModelRow, DashboardProcessRow, DashboardSnapshot, DashboardSnapshotFuture,
    DashboardSnapshotProvider, OutputEvent, RuntimeStatus,
};
use crate::cli::{Cli, Command, RuntimeSurface};
use crate::crypto::{
    default_keystore_path, default_trust_store_path, keystore_exists, keystore_metadata,
    load_keystore, load_owner_keypair_from_keychain, load_trust_store, OwnerKeychainLoadError,
};
use crate::inference::{election, launch, skippy};
use crate::mesh;
use crate::mesh::NodeRole;
use crate::models;
use crate::models::catalog;
use crate::network::{affinity, nostr, router, tunnel};
use crate::plugin;
use crate::system::{autoupdate, benchmark, hardware};
use anyhow::{Context, Result};
use clap::{CommandFactory, Parser};
use std::cell::Cell;
use std::collections::HashMap;
use std::io::{self, IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};
use std::time::{Duration, Instant};
use tracing_subscriber::fmt::MakeWriter;
use zeroize::Zeroizing;

const PRETTY_DASHBOARD_INVENTORY_CACHE_TTL: Duration = Duration::from_secs(5);

thread_local! {
    static ROUTING_TRACING_STDERR: Cell<bool> = const { Cell::new(false) };
}

#[derive(Clone, Copy, Default)]
struct MeshTracingStderr;

struct MeshTracingStderrWriter {
    level: tracing::Level,
    target: String,
    buffer: Vec<u8>,
}

impl MeshTracingStderrWriter {
    fn new(level: tracing::Level, target: impl Into<String>) -> Self {
        Self {
            level,
            target: target.into(),
            buffer: Vec::new(),
        }
    }

    fn drain_complete_lines(&mut self) -> io::Result<()> {
        while let Some(newline_index) = self.buffer.iter().position(|byte| *byte == b'\n') {
            let line = self.buffer.drain(..=newline_index).collect::<Vec<_>>();
            self.write_line(&line)?;
        }
        Ok(())
    }

    fn drain_remainder(&mut self) -> io::Result<()> {
        if self.buffer.is_empty() {
            return Ok(());
        }

        let line = std::mem::take(&mut self.buffer);
        self.write_line(&line)
    }

    fn write_line(&self, line: &[u8]) -> io::Result<()> {
        let message = String::from_utf8_lossy(line)
            .trim_end_matches(['\r', '\n'])
            .to_string();
        if message.trim().is_empty() {
            return Ok(());
        }

        if self.should_route_to_dashboard() {
            return self.route_line_to_dashboard(message);
        }

        write_stderr_line(&message)
    }

    fn should_route_to_dashboard(&self) -> bool {
        !self.target.starts_with("mesh_llm::cli::output")
            && crate::cli::output::interactive_tui_active()
    }

    fn route_line_to_dashboard(&self, message: String) -> io::Result<()> {
        ROUTING_TRACING_STDERR.with(|routing| {
            if routing.get() {
                return write_stderr_line(&message);
            }

            routing.set(true);
            let event = match self.level {
                tracing::Level::ERROR => crate::cli::output::OutputEvent::Error {
                    message: message.clone(),
                    context: Some("stderr".to_string()),
                },
                tracing::Level::WARN => crate::cli::output::OutputEvent::Warning {
                    message: message.clone(),
                    context: Some("stderr".to_string()),
                },
                _ => crate::cli::output::OutputEvent::Info {
                    message: message.clone(),
                    context: Some("stderr".to_string()),
                },
            };
            let result =
                crate::cli::output::emit_event(event).or_else(|_| write_stderr_line(&message));
            routing.set(false);
            result
        })
    }
}

impl Write for MeshTracingStderrWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.buffer.extend_from_slice(buf);
        self.drain_complete_lines()?;
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        self.drain_remainder()
    }
}

impl Drop for MeshTracingStderrWriter {
    fn drop(&mut self) {
        let _ = self.drain_remainder();
    }
}

impl<'writer> MakeWriter<'writer> for MeshTracingStderr {
    type Writer = MeshTracingStderrWriter;

    fn make_writer(&'writer self) -> Self::Writer {
        MeshTracingStderrWriter::new(tracing::Level::INFO, "tracing")
    }

    fn make_writer_for(&'writer self, meta: &tracing::Metadata<'_>) -> Self::Writer {
        MeshTracingStderrWriter::new(*meta.level(), meta.target())
    }
}

fn write_stderr_line(message: &str) -> io::Result<()> {
    let mut stderr = io::stderr().lock();
    stderr.write_all(message.as_bytes())?;
    stderr.write_all(b"\n")?;
    stderr.flush()
}

fn current_time_unix_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

fn publication_state_from_update(update: nostr::PublishStateUpdate) -> api::PublicationState {
    match update {
        nostr::PublishStateUpdate::Public => api::PublicationState::Public,
        nostr::PublishStateUpdate::PublishFailed => api::PublicationState::PublishFailed,
    }
}

#[allow(dead_code)]
struct RuntimeDashboardSnapshotProvider {
    node: mesh::Node,
    local_processes: Arc<tokio::sync::Mutex<Vec<api::RuntimeProcessPayload>>>,
    plugin_manager: Option<plugin::PluginManager>,
    api_port: u16,
    console_port: Option<u16>,
    headless: bool,
    inventory_snapshot_cache: Arc<tokio::sync::Mutex<CachedDashboardInventorySnapshot>>,
    inventory_snapshot_ttl: Duration,
    inventory_snapshot_loader:
        Arc<dyn Fn() -> crate::models::LocalModelInventorySnapshot + Send + Sync>,
}

#[cfg(test)]
struct RuntimeDashboardSnapshotProviderTestOptions {
    api_port: u16,
    console_port: Option<u16>,
    headless: bool,
    inventory_snapshot_ttl: Duration,
    inventory_snapshot_loader:
        Arc<dyn Fn() -> crate::models::LocalModelInventorySnapshot + Send + Sync>,
}

#[derive(Clone, Default)]
struct CachedDashboardInventorySnapshot {
    snapshot: crate::models::LocalModelInventorySnapshot,
    captured_at: Option<Instant>,
}

impl RuntimeDashboardSnapshotProvider {
    fn new(
        node: mesh::Node,
        local_processes: Arc<tokio::sync::Mutex<Vec<api::RuntimeProcessPayload>>>,
        plugin_manager: Option<plugin::PluginManager>,
        api_port: u16,
        console_port: Option<u16>,
        headless: bool,
    ) -> Self {
        Self {
            node,
            local_processes,
            plugin_manager,
            api_port,
            console_port,
            headless,
            inventory_snapshot_cache: Arc::new(tokio::sync::Mutex::new(
                CachedDashboardInventorySnapshot::default(),
            )),
            inventory_snapshot_ttl: PRETTY_DASHBOARD_INVENTORY_CACHE_TTL,
            inventory_snapshot_loader: Arc::new(|| {
                crate::models::scan_local_inventory_snapshot_with_progress(|_| {})
            }),
        }
    }

    #[cfg(test)]
    fn with_inventory_loader(
        node: mesh::Node,
        local_processes: Arc<tokio::sync::Mutex<Vec<api::RuntimeProcessPayload>>>,
        plugin_manager: Option<plugin::PluginManager>,
        options: RuntimeDashboardSnapshotProviderTestOptions,
    ) -> Self {
        Self {
            node,
            local_processes,
            plugin_manager,
            api_port: options.api_port,
            console_port: options.console_port,
            headless: options.headless,
            inventory_snapshot_cache: Arc::new(tokio::sync::Mutex::new(
                CachedDashboardInventorySnapshot::default(),
            )),
            inventory_snapshot_ttl: options.inventory_snapshot_ttl,
            inventory_snapshot_loader: options.inventory_snapshot_loader,
        }
    }

    async fn inventory_snapshot(&self) -> crate::models::LocalModelInventorySnapshot {
        {
            let cache = self.inventory_snapshot_cache.lock().await;
            if let Some(captured_at) = cache.captured_at {
                if captured_at.elapsed() < self.inventory_snapshot_ttl {
                    return cache.snapshot.clone();
                }
            }
        }

        let inventory_snapshot_loader = self.inventory_snapshot_loader.clone();
        let snapshot = match tokio::task::spawn_blocking(move || inventory_snapshot_loader()).await
        {
            Ok(snapshot) => snapshot,
            Err(err) => {
                tracing::warn!("pretty dashboard inventory snapshot failed: {err}");
                crate::models::LocalModelInventorySnapshot::default()
            }
        };

        let mut cache = self.inventory_snapshot_cache.lock().await;
        cache.snapshot = snapshot.clone();
        cache.captured_at = Some(Instant::now());
        snapshot
    }
}

impl DashboardSnapshotProvider for RuntimeDashboardSnapshotProvider {
    fn snapshot(&self) -> DashboardSnapshotFuture<'_> {
        let node = self.node.clone();
        let local_processes = self.local_processes.clone();
        let api_port = self.api_port;
        let console_port = self.console_port;
        let headless = self.headless;
        let plugin_manager = self.plugin_manager.clone();
        let provider = self;

        Box::pin(async move {
            let process_rows = local_processes.lock().await.clone();
            let request_metrics = node.local_request_metrics_snapshot();
            let accepted_request_counts_len = request_metrics.accepted_request_counts.len();
            let inventory_snapshot = provider.inventory_snapshot().await;
            let metadata_by_name = inventory_snapshot.metadata_by_name;
            let mut loaded_model_rows = Vec::with_capacity(process_rows.len());
            for process in &process_rows {
                let metadata = metadata_by_name.get(&process.name);
                loaded_model_rows.push(DashboardModelRow {
                    name: process.name.clone(),
                    role: dashboard_role_for_local_process(process),
                    status: runtime_status_from_process_status(&process.status),
                    port: Some(process.port),
                    device: Some(process.backend.clone()),
                    slots: Some(process.slots),
                    quantization: metadata
                        .map(|model| model.quantization_type.trim())
                        .filter(|value| !value.is_empty())
                        .map(str::to_string),
                    ctx_size: node
                        .local_model_context_length(&process.name)
                        .await
                        .or_else(|| {
                            metadata
                                .map(|model| model.context_length)
                                .filter(|value| *value > 0)
                        }),
                    file_size_gb: inventory_snapshot
                        .size_by_name
                        .get(&process.name)
                        .map(|size| *size as f64 / 1e9),
                });
            }
            loaded_model_rows.sort_by(|left, right| left.name.cmp(&right.name));

            let mut webserver_rows =
                build_dashboard_endpoint_rows(api_port, console_port, headless);
            if let Some(plugin_manager) = plugin_manager {
                webserver_rows.extend(plugin_dashboard_endpoint_rows(&plugin_manager).await);
            }
            sort_dashboard_endpoint_rows(&mut webserver_rows);

            DashboardSnapshot {
                llama_process_rows: process_rows
                    .into_iter()
                    .map(|process| DashboardProcessRow {
                        name: process.name,
                        backend: process.backend,
                        status: runtime_status_from_process_status(&process.status),
                        port: process.port,
                        pid: process.pid,
                    })
                    .collect(),
                webserver_rows,
                loaded_model_rows,
                current_inflight_requests: node.inflight_requests(),
                accepted_request_buckets: request_metrics
                    .accepted_request_counts
                    .into_iter()
                    .enumerate()
                    .map(|(index, accepted_count)| DashboardAcceptedRequestBucket {
                        second_offset: accepted_request_counts_len.saturating_sub(1 + index) as u32,
                        accepted_count,
                    })
                    .collect(),
                latency_samples_ms: request_metrics.latency_samples_ms,
            }
        })
    }
}

#[allow(dead_code)]
fn runtime_status_from_process_status(status: &str) -> RuntimeStatus {
    match status {
        "ready" => RuntimeStatus::Ready,
        "shutting down" | "shutting_down" => RuntimeStatus::ShuttingDown,
        "stopped" => RuntimeStatus::Stopped,
        "exited" => RuntimeStatus::Exited,
        "warning" => RuntimeStatus::Warning,
        "error" => RuntimeStatus::Error,
        _ => RuntimeStatus::Starting,
    }
}

#[allow(dead_code)]
fn runtime_status_from_plugin_status(status: &str) -> RuntimeStatus {
    match status {
        "running" | "ready" => RuntimeStatus::Ready,
        "shutting down" | "shutting_down" => RuntimeStatus::ShuttingDown,
        "stopped" | "disabled" => RuntimeStatus::Stopped,
        "error" => RuntimeStatus::Error,
        "restarting" => RuntimeStatus::Warning,
        _ => RuntimeStatus::Starting,
    }
}

#[allow(dead_code)]
fn dashboard_role_for_local_process(_process: &api::RuntimeProcessPayload) -> Option<String> {
    // `local_processes` only tracks local model-serving processes that own a ready
    // listening port on this node, so the pretty-only Loaded Models panel should
    // present them as host entries rather than inferring from event text.
    Some("host".to_string())
}

#[allow(dead_code)]
fn build_dashboard_endpoint_rows(
    api_port: u16,
    console_port: Option<u16>,
    headless: bool,
) -> Vec<DashboardEndpointRow> {
    let mut rows = vec![DashboardEndpointRow {
        label: "OpenAI-compatible API".to_string(),
        status: RuntimeStatus::Ready,
        url: format!("http://localhost:{api_port}"),
        port: api_port,
        pid: None,
    }];
    if let Some(console_port) = console_port.filter(|_| !headless) {
        rows.push(DashboardEndpointRow {
            label: "Web console".to_string(),
            status: RuntimeStatus::Ready,
            url: format!("http://localhost:{console_port}"),
            port: console_port,
            pid: None,
        });
    }
    sort_dashboard_endpoint_rows(&mut rows);
    rows
}

fn sort_dashboard_endpoint_rows(rows: &mut [DashboardEndpointRow]) {
    rows.sort_by(|left, right| {
        dashboard_endpoint_sort_bucket(left)
            .cmp(&dashboard_endpoint_sort_bucket(right))
            .then_with(|| left.label.cmp(&right.label))
    });
}

fn dashboard_endpoint_sort_bucket(row: &DashboardEndpointRow) -> u8 {
    if row.label.starts_with("Plugin: ") {
        1
    } else {
        0
    }
}

#[allow(dead_code)]
async fn plugin_dashboard_endpoint_rows(
    plugin_manager: &plugin::PluginManager,
) -> Vec<DashboardEndpointRow> {
    plugin_manager
        .list()
        .await
        .into_iter()
        .map(|summary| {
            let url = plugin_dashboard_command_name(&summary);
            DashboardEndpointRow {
                label: format!("Plugin: {}", summary.name),
                status: runtime_status_from_plugin_status(&summary.status),
                url,
                port: 0,
                pid: summary.pid,
            }
        })
        .collect()
}

fn plugin_dashboard_command_name(summary: &plugin::PluginSummary) -> String {
    summary
        .command
        .as_deref()
        .filter(|command| !command.is_empty())
        .and_then(|command| {
            Path::new(command)
                .file_name()
                .and_then(|name| name.to_str())
                .filter(|name| !name.is_empty())
        })
        .unwrap_or(&summary.kind)
        .to_string()
}

fn runtime_process_payload_with_status(
    name: &str,
    handle: &LocalRuntimeModelHandle,
    status: &str,
) -> api::RuntimeProcessPayload {
    api::RuntimeProcessPayload {
        name: name.to_string(),
        backend: handle.backend.clone(),
        status: status.to_string(),
        port: handle.port,
        pid: handle.pid(),
        slots: handle.slots,
        context_length: Some(handle.context_length),
    }
}

async fn upsert_dashboard_process(
    shared: &Arc<tokio::sync::Mutex<Vec<api::RuntimeProcessPayload>>>,
    process: api::RuntimeProcessPayload,
) {
    let mut guard = shared.lock().await;
    guard.retain(|existing| existing.name != process.name);
    guard.push(process);
    guard.sort_by_key(|process| process.name.to_lowercase());
}

async fn remove_dashboard_process(
    shared: &Arc<tokio::sync::Mutex<Vec<api::RuntimeProcessPayload>>>,
    model_name: &str,
) {
    shared
        .lock()
        .await
        .retain(|process| process.name != model_name);
}

struct StartupLocalModelTask {
    node: mesh::Node,
    tunnel_mgr: tunnel::Manager,
    target_tx: Arc<tokio::sync::watch::Sender<election::ModelTargets>>,
    model_path: PathBuf,
    model_name: String,
    primary_model_name: String,
    mmproj_path: Option<PathBuf>,
    ctx_size: Option<u32>,
    pinned_gpu: Option<StartupPinnedGpuTarget>,
    slots: usize,
    stop_rx: tokio::sync::watch::Receiver<bool>,
    dashboard_processes: Arc<tokio::sync::Mutex<Vec<api::RuntimeProcessPayload>>>,
    console_state: Option<api::MeshApi>,
    api_port: u16,
    startup_ready_reporter: StartupReadyReporter,
    startup_load_gate: Arc<tokio::sync::Mutex<()>>,
    input_handler_enabled: bool,
    interactive_started: Arc<AtomicBool>,
    interactive_control_tx: tokio::sync::mpsc::UnboundedSender<api::RuntimeControlRequest>,
    interactive_console_state: Option<api::MeshApi>,
}

async fn startup_local_model_loop(params: StartupLocalModelTask) {
    let StartupLocalModelTask {
        node,
        tunnel_mgr,
        target_tx,
        model_path,
        model_name,
        primary_model_name,
        mmproj_path,
        ctx_size,
        pinned_gpu,
        slots,
        mut stop_rx,
        dashboard_processes,
        console_state,
        api_port,
        startup_ready_reporter,
        startup_load_gate,
        input_handler_enabled,
        interactive_started,
        interactive_control_tx,
        interactive_console_state,
    } = params;

    let startup_load_guard = startup_load_gate.lock().await;
    let (loaded_name, handle, mut death_rx) =
        match start_runtime_local_model(LocalRuntimeModelStartSpec {
            node: &node,
            model_path: &model_path,
            mmproj_override: mmproj_path.as_deref(),
            ctx_size_override: ctx_size,
            pinned_gpu: pinned_gpu.as_ref(),
            slots,
        })
        .await
        {
            Ok(loaded) => loaded,
            Err(err) => {
                let _ = emit_event(OutputEvent::Error {
                    message: format!("Failed to start model {model_name}: {err:#}"),
                    context: Some(format!("model={model_name}")),
                });
                update_startup_target(&target_tx, &model_name, election::InferenceTarget::None);
                if let Some(cs) = console_state {
                    cs.update(false, false).await;
                }
                return;
            }
        };
    drop(startup_load_guard);

    add_runtime_local_target(&target_tx, &loaded_name, handle.port);
    tunnel_mgr.set_http_port(handle.port);
    node.set_role(NodeRole::Host {
        http_port: api_port,
    })
    .await;
    set_advertised_model_context(&node, &loaded_name, Some(handle.context_length)).await;
    advertise_model_ready(&node, &primary_model_name, &loaded_name).await;
    let payload = local_process_payload(
        &loaded_name,
        &handle.backend,
        handle.port,
        handle.pid(),
        slots,
        handle.context_length,
    );
    upsert_dashboard_process(&dashboard_processes, payload.clone()).await;
    if let Some(ref cs) = console_state {
        cs.upsert_local_process(payload).await;
        cs.update(true, true).await;
    }
    update_pi_models_json(&loaded_name, api_port);
    startup_ready_reporter.mark_ready_and_maybe_emit(&loaded_name);
    let _ = emit_event(OutputEvent::ModelReady {
        model: loaded_name.clone(),
        internal_port: Some(handle.port),
        role: Some(handle.backend.clone()),
    });
    let _ = emit_event(OutputEvent::Info {
        message: format!(
            "Startup-loaded {} model '{}' on :{}",
            handle.backend, loaded_name, handle.port
        ),
        context: None,
    });

    if input_handler_enabled
        && loaded_name == primary_model_name
        && !interactive_started.swap(true, Ordering::AcqRel)
        && std::io::stdin().is_terminal()
    {
        if let Some(cs) = interactive_console_state {
            interactive::spawn_handler(
                interactive_control_tx,
                cs,
                crate::cli::output::OutputManager::global(),
                InitialPromptMode::Deferred,
            );
        }
    }

    tokio::select! {
        _ = &mut death_rx => {
            let _ = emit_event(OutputEvent::Warning {
                message: format!("Startup model '{loaded_name}' exited unexpectedly"),
                context: Some(format!("model={loaded_name} port={}", handle.port)),
            });
        }
        res = stop_rx.changed() => {
            let _ = res;
        }
    }

    let port = handle.port;
    remove_runtime_local_target(&target_tx, &loaded_name, port);
    tunnel_mgr.set_http_port(0);
    withdraw_advertised_model(&node, &loaded_name).await;
    set_advertised_model_context(&node, &loaded_name, None).await;
    upsert_dashboard_process(
        &dashboard_processes,
        runtime_process_payload_with_status(&loaded_name, &handle, "shutting down"),
    )
    .await;
    if let Some(ref cs) = console_state {
        cs.upsert_local_process(runtime_process_payload_with_status(
            &loaded_name,
            &handle,
            "shutting down",
        ))
        .await;
    }
    handle.shutdown().await;
    remove_dashboard_process(&dashboard_processes, &loaded_name).await;
    if let Some(cs) = console_state {
        cs.remove_local_process(&loaded_name).await;
        cs.update(false, false).await;
    }
    let _ = emit_event(OutputEvent::Info {
        message: format!("Stopped startup model '{}' from :{}", loaded_name, port),
        context: None,
    });
}

fn update_startup_target(
    target_tx: &Arc<tokio::sync::watch::Sender<election::ModelTargets>>,
    model_name: &str,
    target: election::InferenceTarget,
) {
    let mut targets = target_tx.borrow().clone();
    targets.targets.insert(model_name.to_string(), vec![target]);
    target_tx.send_replace(targets);
}

fn bridge_publication_state(
    console_state: api::MeshApi,
    mut status_rx: tokio::sync::watch::Receiver<Option<nostr::PublishStateUpdate>>,
) {
    tokio::spawn(async move {
        let mut pending = *status_rx.borrow_and_update();
        loop {
            if let Some(update) = pending.take() {
                console_state
                    .set_publication_state(publication_state_from_update(update))
                    .await;
            }

            if status_rx.changed().await.is_err() {
                break;
            }
            pending = *status_rx.borrow_and_update();
        }
    });
}

fn write_stderr_newline() {
    let _ = std::io::stderr().write_all(b"\n");
}

fn emit_shutdown(reason: Option<String>) {
    crate::inference::launch::mark_runtime_shutting_down();
    let _ = emit_event(OutputEvent::Shutdown { reason });
}

#[derive(Clone)]
struct StartupReadyReporter {
    ready_by_model: Arc<Mutex<HashMap<String, bool>>>,
    emitted: Arc<AtomicBool>,
    primary_model: String,
    api_port: u16,
    console_port: Option<u16>,
}

impl StartupReadyReporter {
    fn new(
        models: &[String],
        primary_model: String,
        api_port: u16,
        console_port: Option<u16>,
    ) -> Self {
        let ready_by_model = models.iter().cloned().map(|model| (model, false)).collect();
        Self {
            ready_by_model: Arc::new(Mutex::new(ready_by_model)),
            emitted: Arc::new(AtomicBool::new(false)),
            primary_model,
            api_port,
            console_port,
        }
    }

    fn mark_ready_and_maybe_emit(&self, model_name: &str) {
        let models_count = {
            let mut ready_by_model = self
                .ready_by_model
                .lock()
                .expect("startup readiness mutex poisoned");
            if let Some(entry) = ready_by_model.get_mut(model_name) {
                *entry = true;
            }
            if ready_by_model.values().all(|ready| *ready) {
                Some(ready_by_model.len())
            } else {
                None
            }
        };

        let Some(models_count) = models_count else {
            return;
        };

        if self.emitted.swap(true, Ordering::SeqCst) {
            return;
        }

        let api_url = format!("http://localhost:{}", self.api_port);
        let console_url = self
            .console_port
            .map(|port| format!("http://localhost:{port}"));
        let pi_command = Some(format!(
            "mesh-llm pi --host 127.0.0.1:{} --model {}",
            self.api_port,
            crate::cli::shell::single_quote(&self.primary_model)
        ));
        let goose_command = Some(format!(
            "GOOSE_PROVIDER=openai OPENAI_HOST={api_url} OPENAI_API_KEY=mesh GOOSE_MODEL={} goose session",
            self.primary_model
        ));
        let _ = emit_event(OutputEvent::RuntimeReady {
            api_url,
            console_url,
            api_port: self.api_port,
            console_port: self.console_port,
            models_count: Some(models_count),
            pi_command,
            goose_command,
        });
        let _ = crate::cli::output::OutputManager::global().schedule_ready_prompt();
    }
}

async fn record_first_joined_mesh_ts(node: &mesh::Node) {
    let now_ms = current_time_unix_ms();
    node.set_first_joined_mesh_ts_if_absent(now_ms).await;
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct StartupModelSpec {
    model_ref: PathBuf,
    mmproj_ref: Option<PathBuf>,
    ctx_size: Option<u32>,
    gpu_id: Option<String>,
    config_owned: bool,
    parallel: Option<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct StartupPinnedGpuTarget {
    pub(crate) index: usize,
    pub(crate) stable_id: String,
    pub(crate) backend_device: String,
    pub(crate) vram_bytes: u64,
}

#[derive(Clone, Debug)]
struct StartupModelPlan {
    declared_ref: String,
    resolved_path: PathBuf,
    mmproj_path: Option<PathBuf>,
    ctx_size: Option<u32>,
    gpu_id: Option<String>,
    pinned_gpu: Option<StartupPinnedGpuTarget>,
    parallel: Option<usize>,
}

fn resolve_runtime_owner_key_path(cli: &Cli) -> Result<Option<PathBuf>> {
    if let Some(path) = cli.owner_key.clone() {
        return Ok(Some(path));
    }

    let default_path = default_keystore_path()?;
    if keystore_exists(&default_path) {
        Ok(Some(default_path))
    } else {
        Ok(None)
    }
}

fn resolve_owner_passphrase(path: &Path) -> Result<Option<Zeroizing<String>>> {
    let info = keystore_metadata(path)?;
    if !info.encrypted {
        return Ok(None);
    }

    if let Ok(passphrase) = std::env::var("MESH_LLM_OWNER_PASSPHRASE") {
        return Ok(Some(Zeroizing::new(passphrase)));
    }

    if std::io::stdin().is_terminal() && std::io::stderr().is_terminal() {
        let prompt = format!("Enter owner keystore passphrase for {}: ", path.display());
        let passphrase = rpassword::prompt_password_stderr(&prompt)?;
        return Ok(Some(Zeroizing::new(passphrase)));
    }

    Err(crate::crypto::CryptoError::MissingPassphrase.into())
}

fn load_owner_keypair_for_runtime(path: &Path) -> Result<crate::crypto::OwnerKeypair> {
    let info = keystore_metadata(path)?;
    if info.encrypted && std::env::var("MESH_LLM_OWNER_PASSPHRASE").is_err() {
        match load_owner_keypair_from_keychain(path) {
            Ok(keypair) => return Ok(keypair),
            Err(OwnerKeychainLoadError::NoEntry)
            | Err(OwnerKeychainLoadError::Crypto(crate::crypto::CryptoError::DecryptionFailed))
            | Err(OwnerKeychainLoadError::Crypto(
                crate::crypto::CryptoError::KeychainUnavailable { .. },
            ))
            | Err(OwnerKeychainLoadError::Crypto(
                crate::crypto::CryptoError::KeychainAccessDenied { .. },
            )) => {}
            Err(OwnerKeychainLoadError::Crypto(err)) => {
                return Err(err)
                    .with_context(|| format!("Failed to load owner keystore {}", path.display()));
            }
        }
    }

    let passphrase = resolve_owner_passphrase(path)?;
    load_keystore(path, passphrase.as_deref().map(|value| value.as_str()))
        .with_context(|| format!("Failed to load owner keystore {}", path.display()))
}

fn owner_runtime_config(cli: &Cli) -> Result<mesh::OwnerRuntimeConfig> {
    let trust_store_path = default_trust_store_path()?;
    let trust_store = load_trust_store(&trust_store_path)
        .with_context(|| format!("Failed to load trust store {}", trust_store_path.display()))?
        .merged_with_trusted_owners(&cli.trust_owner);
    let trust_policy = cli.trust_policy.unwrap_or(trust_store.policy);

    let keypair = match resolve_runtime_owner_key_path(cli)? {
        Some(path) => match load_owner_keypair_for_runtime(&path) {
            Ok(keypair) => Some(keypair),
            Err(err) if !cli.owner_required => {
                let _ = emit_event(OutputEvent::Warning {
                    message: format!(
                        "Owner identity unavailable: {err}. Starting without owner attestation."
                    ),
                    context: Some(path.display().to_string()),
                });
                None
            }
            Err(err) => return Err(err),
        },
        None if cli.owner_required => {
            anyhow::bail!(
                "Owner identity is required but no keystore was found. Use --owner-key or run `mesh-llm auth init`."
            );
        }
        None => None,
    };

    Ok(mesh::OwnerRuntimeConfig {
        keypair,
        node_label: cli.node_label.clone(),
        trust_store,
        trust_policy,
    })
}

/// Wait for either SIGINT (ctrl-c) or SIGTERM. Without this, an unhandled
/// SIGTERM aborts the process before runtime cleanup can run.
async fn wait_shutdown_signal() {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{signal, SignalKind};
        let mut term = match signal(SignalKind::terminate()) {
            Ok(s) => s,
            Err(_) => {
                let _ = tokio::signal::ctrl_c().await;
                return;
            }
        };
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {}
            _ = term.recv() => {}
        }
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
    }
}

pub(crate) async fn run() -> Result<()> {
    crate::inference::launch::clear_runtime_shutting_down();
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("mesh_inference=info".parse()?)
                .add_directive("nostr_relay_pool=off".parse()?)
                .add_directive("nostr_sdk=warn".parse()?)
                .add_directive("noq_proto::connection=warn".parse()?),
        )
        .with_writer(MeshTracingStderr)
        .init();

    // --help-advanced: print full help with all hidden options and commands visible
    if std::env::args().any(|a| a == "--help-advanced") {
        let mut cmd = Cli::command();
        // Unhide all arguments
        let args: Vec<clap::Id> = cmd.get_arguments().map(|a| a.get_id().clone()).collect();
        for id in args {
            cmd = cmd.mut_arg(id, |a| a.hide(false));
        }
        // Unhide all subcommands
        let sub_names: Vec<String> = cmd
            .get_subcommands()
            .map(|s| s.get_name().to_string())
            .collect();
        for name in sub_names {
            cmd = cmd.mut_subcommand(name, |s| s.hide(false));
        }
        cmd.print_help().ok();
        write_stderr_newline();
        std::process::exit(0);
    }

    if std::env::args_os().len() == 1 {
        Cli::command().print_help().ok();
        std::process::exit(0);
    }

    let normalized_args = crate::cli::normalize_runtime_surface_args(std::env::args_os());
    let mut cli = Cli::parse_from(normalized_args.normalized.clone());
    crate::cli::output::OutputManager::init_global(
        cli.log_format,
        initial_console_session_mode(normalized_args.explicit_surface),
    );

    if let Some(warning) = crate::cli::legacy_runtime_surface_warning(
        &cli,
        &normalized_args.original,
        normalized_args.explicit_surface,
    ) {
        let _ = emit_event(OutputEvent::Warning {
            message: warning,
            context: None,
        });
    }

    if let Some(name) = cli.plugin.clone() {
        return plugin::run_plugin_process(name).await;
    }

    let checked_updates = autoupdate::maybe_auto_update(&cli).await?;

    // Finish the release check before startup continues.
    if !checked_updates && !matches!(cli.command, Some(Command::Update { .. })) {
        autoupdate::check_for_update().await;
    }

    if crate::cli::commands::dispatch(&cli).await? {
        return Ok(());
    }

    let config = plugin::load_config(cli.config.as_deref())?;
    let cli_has_explicit_models = cli_has_explicit_models(&cli);
    let has_config_models = !config.models.is_empty();
    let has_startup_models = cli_has_explicit_models || has_config_models;

    // Acquire the per-instance runtime directory and flock (skip for --client).
    // Wrap in Arc so it can be cheaply shared with local model tasks.
    let runtime: Option<Arc<crate::runtime::instance::InstanceRuntime>> = if !cli.client {
        match crate::runtime::instance::InstanceRuntime::acquire(std::process::id()) {
            Ok(rt) => Some(Arc::new(rt)),
            Err(e) => {
                tracing::warn!("failed to acquire instance runtime: {e}");
                None
            }
        }
    } else {
        None
    };

    // Write owner.json into the runtime dir so sibling-instance discovery can find us.
    if let Some(ref rt) = runtime {
        let started_at =
            crate::runtime::instance::validate::current_process_start_time_unix().unwrap_or(0);
        let owner_meta = serde_json::json!({
            "pid": std::process::id(),
            "api_port": cli.console,
            "version": crate::VERSION,
            "started_at_unix": started_at,
            "mesh_llm_binary": std::env::current_exe()
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_default(),
        });
        let owner_path = rt.dir().join("owner.json");
        if let Ok(json) = serde_json::to_string_pretty(&owner_meta) {
            let _ = crate::runtime::instance::write_text_file_atomic(&owner_path, &json);
        }
    }

    // Reap stale runtime state from dead sibling instances (skip for client).
    // This intentionally happens after subcommand dispatch so control commands
    // targeting a live instance don't kill it before sending the request.
    // Uses scoped reap that only touches runtime dirs whose owner has died (flock-released).
    if !cli.client {
        if let Some(ref rt) = runtime {
            match crate::runtime::instance::runtime_root() {
                Ok(root) => {
                    let my_dir = rt.dir().to_path_buf();
                    match crate::runtime::instance::reap::reap_cross_runtime_orphans(&root, &my_dir)
                        .await
                    {
                        Ok(summary) => {
                            if summary.children_killed > 0 || summary.dirs_gc_d > 0 {
                                let _ = emit_event(OutputEvent::Info {
                                    message: format!(
                                        "Reaped {} orphan children from {} dead instance(s)",
                                        summary.children_killed, summary.dirs_gc_d
                                    ),
                                    context: None,
                                });
                            }
                        }
                        Err(e) => tracing::warn!("cross-runtime reap failed: {e}"),
                    }
                }
                Err(e) => tracing::warn!("runtime_root resolution failed during reap: {e}"),
            }
        }
    }

    // Publication intent is now explicit only: --publish gates Nostr discovery.
    // --mesh-name alone never implies publication (Issue #240).

    // Warn users who used to rely on --mesh-name auto-publishing
    if let Some(mesh_name) = cli.mesh_name.as_ref().filter(|_| !cli.publish) {
        let _ = emit_event(OutputEvent::Info {
            message: format!(
                "Mesh named '{}' — private by default. Add --publish to make it publicly discoverable.",
                mesh_name
            ),
            context: None,
        });
    }

    // --- Public-to-private identity transition ---
    // If the previous run was public (--auto or --publish) but this run is
    // private, clear the stored identity so the private mesh gets a fresh key
    // that isn't associated with the old public listing.
    let is_public = cli.auto || cli.publish;
    if is_public {
        mesh::mark_was_public();
    } else if mesh::was_previously_public() {
        let _ = emit_event(OutputEvent::Info {
            message: "Previous run was public — rotating identity for private mesh".to_string(),
            context: None,
        });
        mesh::clear_public_identity();
    }

    let mut auto_join_candidates: Vec<(String, Option<String>)> = Vec::new();

    // --- Auto-discover ---
    if cli.auto && cli.join.is_empty() {
        cli.nostr_discovery = true;
        let _ = emit_event(OutputEvent::DiscoveryStarting {
            source: "Nostr auto-discovery".to_string(),
        });

        let relays = nostr_relays(&cli.nostr_relay);
        let filter = nostr::MeshFilter {
            model: None,
            min_vram_gb: None,
            region: None,
        };
        let meshes = match nostr::discover(&relays, &filter, None).await {
            Ok(meshes) => meshes,
            Err(err) => {
                let _ = emit_event(OutputEvent::DiscoveryFailed {
                    message: "Nostr auto-discovery failed".to_string(),
                    detail: Some(err.to_string()),
                });
                return Err(err);
            }
        };

        let my_vram_gb = mesh::detect_vram_bytes_capped(cli.max_vram) as f64 / 1e9;
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let last_mesh_id = mesh::load_last_mesh_id();
        let target_name = cli.mesh_name.as_deref();
        // When the user did not target a specific mesh, `--auto` only joins
        // the community mesh (unnamed or name == "mesh-llm"). Other named
        // meshes are still publicly discoverable on Nostr, but the user has
        // to opt in by name. Hide them from the listing so the output matches
        // what auto will actually consider.
        let listed: Vec<&nostr::DiscoveredMesh> = if target_name.is_some() {
            meshes.iter().collect()
        } else {
            meshes
                .iter()
                .filter(|m| nostr::is_auto_eligible(m))
                .collect()
        };
        for m in &listed {
            let score = nostr::score_mesh(m, now, last_mesh_id.as_deref());
            let _ = emit_event(OutputEvent::MeshFound {
                mesh: m.listing.name.as_deref().unwrap_or("unnamed").to_string(),
                peers: m.listing.node_count,
                region: m.listing.region.clone(),
            });
            tracing::debug!(
                "Nostr auto-discovery candidate: {} score={} nodes={} vram_gb={:.0} clients={}",
                m.listing.name.as_deref().unwrap_or("unnamed"),
                score,
                m.listing.node_count,
                m.listing.total_vram_bytes as f64 / 1e9,
                m.listing.client_count
            );
        }

        match nostr::smart_auto(&meshes, my_vram_gb, target_name) {
            nostr::AutoDecision::Join { candidates } => {
                if cli.client {
                    // Clients skip health probe — joining itself is the test.
                    // Queue all candidates so we can fall back if the top one is unreachable.
                    let (_, mesh) = &candidates[0];
                    if cli.mesh_name.is_none() {
                        if let Some(ref name) = mesh.listing.name {
                            cli.mesh_name = Some(name.clone());
                        }
                    }
                    let _ = emit_event(OutputEvent::DiscoveryJoined {
                        mesh: mesh
                            .listing
                            .name
                            .as_deref()
                            .unwrap_or("unnamed")
                            .to_string(),
                    });
                    for (token, _) in &candidates {
                        cli.join.push(token.clone());
                    }
                } else {
                    // GPU nodes: try to join each candidate directly.
                    // No ephemeral probe — it fails when the target has a firewall
                    // even though the real join (via relay) would succeed.
                    let mut joined = false;
                    for (token, mesh) in &candidates {
                        let _ = emit_event(OutputEvent::MeshFound {
                            mesh: mesh
                                .listing
                                .name
                                .as_deref()
                                .unwrap_or("unnamed")
                                .to_string(),
                            peers: mesh.listing.node_count,
                            region: mesh.listing.region.clone(),
                        });
                        auto_join_candidates.push((token.clone(), mesh.listing.name.clone()));
                        joined = true;
                    }
                    if !joined {
                        let _ = emit_event(OutputEvent::DiscoveryFailed {
                            message: "No meshes found — starting new".to_string(),
                            detail: None,
                        });
                        let models = nostr::default_models_for_vram(my_vram_gb);
                        start_new_mesh(&mut cli, &models, my_vram_gb, has_startup_models);
                    }
                }
            }
            nostr::AutoDecision::StartNew { models } => {
                if cli.client {
                    // Retry discovery — meshes may appear later
                    let _ = emit_event(OutputEvent::DiscoveryFailed {
                        message: "No meshes found yet — retrying in 15s...".to_string(),
                        detail: None,
                    });
                    let mut found = false;
                    for attempt in 1..=20 {
                        tokio::time::sleep(std::time::Duration::from_secs(15)).await;
                        let _ = emit_event(OutputEvent::DiscoveryStarting {
                            source: format!("Nostr retry {attempt}/20"),
                        });
                        if let Ok(retry_meshes) = nostr::discover(&relays, &filter, None).await {
                            if let nostr::AutoDecision::Join { candidates } =
                                nostr::smart_auto(&retry_meshes, my_vram_gb, target_name)
                            {
                                let (_, mesh) = &candidates[0];
                                if cli.mesh_name.is_none() {
                                    if let Some(ref name) = mesh.listing.name {
                                        cli.mesh_name = Some(name.clone());
                                    }
                                }
                                let _ = emit_event(OutputEvent::DiscoveryJoined {
                                    mesh: mesh
                                        .listing
                                        .name
                                        .as_deref()
                                        .unwrap_or("unnamed")
                                        .to_string(),
                                });
                                for (token, _) in &candidates {
                                    cli.join.push(token.clone());
                                }
                                found = true;
                                break;
                            }
                        } else {
                            let _ = emit_event(OutputEvent::DiscoveryFailed {
                                message: format!("Retry {attempt}/20 discovery failed"),
                                detail: None,
                            });
                        }
                    }
                    if !found {
                        let _ = emit_event(OutputEvent::DiscoveryFailed {
                            message: "No meshes found after 5 minutes of retrying.".to_string(),
                            detail: None,
                        });
                        anyhow::bail!("No meshes found after 5 minutes of retrying.");
                    }
                } else {
                    start_new_mesh(&mut cli, &models, my_vram_gb, has_startup_models);
                }
            }
        }
    }

    // --- Validation ---
    if cli.client && (!cli.model.is_empty() || !cli.gguf.is_empty()) {
        anyhow::bail!("--client and --model are mutually exclusive");
    }
    if let Some(mmproj) = &cli.mmproj {
        anyhow::ensure!(!cli.client, "--mmproj cannot be used with --client");
        anyhow::ensure!(
            !cli.model.is_empty() || !cli.gguf.is_empty(),
            "--mmproj requires an explicit primary model via --model or --gguf"
        );
        anyhow::ensure!(
            mmproj.is_file(),
            "mmproj path is not a file: {}",
            mmproj.display()
        );
    }
    let startup_specs = build_startup_model_specs(&cli, &config)?;
    if should_show_serve_config_help(normalized_args.explicit_surface, &cli, &startup_specs) {
        let config_path = plugin::config_path(cli.config.as_deref()).unwrap_or_else(|_| {
            dirs::home_dir()
                .unwrap_or_else(|| PathBuf::from("~"))
                .join(".mesh-llm")
                .join("config.toml")
        });
        let _ = emit_event(OutputEvent::Warning {
            message: "`mesh-llm serve` needs at least one startup model. Add `[[models]]` or pass `--model` / `--gguf` explicitly.".to_string(),
            context: Some(config_path.display().to_string()),
        });
        Cli::command().print_help().ok();
        write_stderr_newline();
        return Ok(());
    }
    let mut startup_models = resolve_startup_models(&startup_specs).await?;
    let bin_dir = match &cli.bin_dir {
        Some(d) => d.clone(),
        None => detect_bin_dir()?,
    };
    preflight_config_owned_startup_models(
        &config,
        &startup_specs,
        &mut startup_models,
        cli.llama_flavor,
        None,
    )?;
    let resolved_models: Vec<PathBuf> = startup_models
        .iter()
        .map(|model| model.resolved_path.clone())
        .collect();
    {
        let update_check_paths = resolved_models.clone();
        match tokio::task::spawn_blocking(move || {
            models::warn_about_updates_for_paths(&update_check_paths);
        })
        .await
        {
            Ok(()) => {}
            Err(err) => {
                let _ = emit_event(OutputEvent::Warning {
                    message: format!("Could not join Hugging Face update check task: {err}"),
                    context: None,
                });
            }
        }
    }

    // Build requested model names from all resolved models
    // Strip split GGUF suffix so "MiniMax-M2.5-Q4_K_M-00001-of-00004" → "MiniMax-M2.5-Q4_K_M"
    let requested_model_names: Vec<String> = resolved_models
        .iter()
        .filter_map(|m| {
            m.file_stem()
                .and_then(|s| s.to_str())
                .map(router::strip_split_suffix_owned)
        })
        .collect();

    run_auto(
        cli,
        config,
        startup_models,
        requested_model_names,
        bin_dir,
        runtime,
        auto_join_candidates,
    )
    .await
}

/// Resolve a model path: local file, catalog name, or HuggingFace URL.
async fn resolve_model(input: &std::path::Path) -> Result<PathBuf> {
    models::resolve_model_spec(input).await
}

fn cli_has_explicit_models(cli: &Cli) -> bool {
    !cli.model.is_empty() || !cli.gguf.is_empty()
}

fn build_startup_model_specs(
    cli: &Cli,
    config: &plugin::MeshConfig,
) -> Result<Vec<StartupModelSpec>> {
    if cli.client {
        return Ok(Vec::new());
    }

    let mut specs = Vec::new();
    if cli_has_explicit_models(cli) {
        for path in &cli.gguf {
            if !path.exists() {
                anyhow::bail!("GGUF file not found: {}", path.display());
            }
            specs.push(StartupModelSpec {
                model_ref: path.clone(),
                mmproj_ref: None,
                ctx_size: cli.ctx_size,
                gpu_id: None,
                config_owned: false,
                parallel: None,
            });
        }
        for model in &cli.model {
            specs.push(StartupModelSpec {
                model_ref: model.clone(),
                mmproj_ref: None,
                ctx_size: cli.ctx_size,
                gpu_id: None,
                config_owned: false,
                parallel: None,
            });
        }
        if let Some(mmproj) = &cli.mmproj {
            if let Some(primary) = specs.first_mut() {
                primary.mmproj_ref = Some(mmproj.clone());
            }
        }
        return Ok(specs);
    }

    for model in &config.models {
        specs.push(StartupModelSpec {
            model_ref: PathBuf::from(model.model.clone()),
            mmproj_ref: model.mmproj.as_ref().map(PathBuf::from),
            ctx_size: cli.ctx_size.or(model.ctx_size),
            gpu_id: model.gpu_id.clone(),
            config_owned: true,
            parallel: model.parallel,
        });
    }
    Ok(specs)
}

async fn resolve_startup_models(specs: &[StartupModelSpec]) -> Result<Vec<StartupModelPlan>> {
    let mut plans = Vec::with_capacity(specs.len());
    for spec in specs {
        let resolved_path = resolve_model(&spec.model_ref).await?;
        let mmproj_path = match spec.mmproj_ref.as_ref() {
            Some(mmproj) => Some(resolve_model(mmproj).await?),
            None => None,
        };
        plans.push(StartupModelPlan {
            declared_ref: spec.model_ref.to_string_lossy().to_string(),
            resolved_path,
            mmproj_path,
            ctx_size: spec.ctx_size,
            gpu_id: spec.gpu_id.clone(),
            pinned_gpu: None,
            parallel: spec.parallel,
        });
    }
    Ok(plans)
}

fn preflight_config_owned_startup_models(
    config: &plugin::MeshConfig,
    specs: &[StartupModelSpec],
    plans: &mut [StartupModelPlan],
    binary_flavor: Option<launch::BinaryFlavor>,
    backend_probe: Option<&launch::BinaryBackendDeviceProbe>,
) -> Result<()> {
    if config.gpu.assignment != plugin::GpuAssignment::Pinned {
        return Ok(());
    }

    let binary_flavor = backend_probe
        .and_then(|probe| probe.flavor)
        .or(binary_flavor);
    let mut survey = hardware::query(pinned_startup_preflight_metrics());
    if binary_flavor == Some(launch::BinaryFlavor::Vulkan) {
        hardware::augment_gpu_facts_with_vulkan_devices(&mut survey.gpus);
    }
    apply_backend_devices_for_flavor(&mut survey.gpus, binary_flavor);
    preflight_config_owned_startup_models_with_gpus(
        config,
        specs,
        plans,
        &survey.gpus,
        backend_probe,
    )
}

fn apply_backend_devices_for_flavor(
    gpus: &mut [hardware::GpuFacts],
    binary_flavor: Option<launch::BinaryFlavor>,
) {
    let Some(binary_flavor) = binary_flavor else {
        return;
    };

    for gpu in gpus {
        gpu.backend_device = launch::backend_device_for_flavor(gpu.index, binary_flavor);
    }
}

fn pinned_startup_preflight_metrics() -> &'static [hardware::Metric] {
    &[
        hardware::Metric::GpuName,
        hardware::Metric::GpuFacts,
        hardware::Metric::VramBytes,
        hardware::Metric::IsSoc,
    ]
}

fn preflight_config_owned_startup_models_with_gpus(
    config: &plugin::MeshConfig,
    specs: &[StartupModelSpec],
    plans: &mut [StartupModelPlan],
    gpus: &[hardware::GpuFacts],
    backend_probe: Option<&launch::BinaryBackendDeviceProbe>,
) -> Result<()> {
    if config.gpu.assignment != plugin::GpuAssignment::Pinned {
        return Ok(());
    }

    anyhow::ensure!(
        specs.len() == plans.len(),
        "startup model preflight received mismatched specs/plans"
    );

    for (spec, plan) in specs.iter().zip(plans.iter_mut()) {
        if !spec.config_owned {
            continue;
        }

        let resolved_gpu = hardware::resolve_pinned_gpu(plan.gpu_id.as_deref(), gpus)
            .map_err(anyhow::Error::new)
            .with_context(|| {
                format!(
                    "startup model '{}' failed pinned GPU preflight",
                    plan.declared_ref
                )
            })?;

        let stable_id = resolved_gpu.stable_id.clone().ok_or_else(|| {
            anyhow::anyhow!(
                "startup model '{}' resolved pinned GPU at index {} without a stable_id",
                plan.declared_ref,
                resolved_gpu.index
            )
        })?;

        let backend_device = resolved_gpu
            .backend_device
            .clone()
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "startup model '{}' resolved pinned GPU '{}' at index {} without a backend_device",
                    plan.declared_ref,
                    stable_id,
                    resolved_gpu.index
                )
            })
            .with_context(|| {
                format!(
                    "startup model '{}' failed pinned GPU preflight",
                    plan.declared_ref
                )
            })?;
        let backend_device = if let Some(probe) = backend_probe {
            launch::resolve_requested_device_from_available(
                &probe.available_devices,
                &probe.path,
                &backend_device,
            )
            .with_context(|| {
                format!(
                    "startup model '{}' failed pinned GPU preflight",
                    plan.declared_ref
                )
            })?
        } else {
            backend_device
        };

        plan.pinned_gpu = Some(StartupPinnedGpuTarget {
            index: resolved_gpu.index,
            stable_id,
            backend_device,
            vram_bytes: resolved_gpu.vram_bytes,
        });
    }

    Ok(())
}

fn should_show_serve_config_help(
    explicit_surface: Option<RuntimeSurface>,
    cli: &Cli,
    startup_specs: &[StartupModelSpec],
) -> bool {
    explicit_surface == Some(RuntimeSurface::Serve)
        && !cli.client
        && startup_specs.is_empty()
        && !cli.auto
        && cli.join.is_empty()
        && cli.discover.is_none()
}

fn initial_console_session_mode(explicit_surface: Option<RuntimeSurface>) -> ConsoleSessionMode {
    initial_console_session_mode_for_surface(
        explicit_surface,
        interactive::current_console_session_mode(),
    )
}

fn initial_console_session_mode_for_surface(
    explicit_surface: Option<RuntimeSurface>,
    current_mode: ConsoleSessionMode,
) -> ConsoleSessionMode {
    match explicit_surface {
        Some(RuntimeSurface::Serve) => current_mode,
        _ => ConsoleSessionMode::None,
    }
}

/// Pick which model this node should serve.
///
/// Priority:
/// 1. Models the mesh needs that we already have on disk
/// 2. Models in the mesh catalog that nobody is serving yet (on disk preferred)
///
/// Parse a catalog size string like "18.3GB" or "491MB" into bytes.
fn parse_size_str(s: &str) -> u64 {
    let s = s.trim();
    if let Some(gb) = s.strip_suffix("GB") {
        (gb.parse::<f64>().unwrap_or(0.0) * 1e9) as u64
    } else if let Some(mb) = s.strip_suffix("MB") {
        (mb.parse::<f64>().unwrap_or(0.0) * 1e6) as u64
    } else {
        0
    }
}

/// Pick which model this node should serve, based on demand signals.
///
/// Priority:
/// 1. Unserved models with active demand that we have on disk (hottest first)
/// 2. Underserved models with demand that we have on disk
/// 3. Unserved models with demand that we can download from catalog
/// 4. Standby if everything is covered
async fn pick_model_assignment(node: &mesh::Node, local_models: &[String]) -> Option<String> {
    let peers = node.peers().await;

    // Get active demand — the unified "what does the mesh want?"
    let demand = node.active_demand().await;

    if demand.is_empty() {
        // No API requests yet — log what the mesh is serving for visibility
        let served: Vec<String> = peers.iter().flat_map(|p| p.routable_models()).collect();
        if !served.is_empty() {
            let _ = emit_event(OutputEvent::Info {
                message: format!(
                    "No demand yet — mesh is serving {:?}, staying standby until needed",
                    served
                ),
                context: None,
            });
        } else {
            let _ = emit_event(OutputEvent::Info {
                message: "No demand signals — no models requested".to_string(),
                context: None,
            });
        }
        return None;
    }

    let _ = emit_event(OutputEvent::Info {
        message: format!("Active demand: {:?}", demand.keys().collect::<Vec<_>>()),
        context: None,
    });

    // Count how many nodes are serving each model
    let mut serving_count: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();
    for p in &peers {
        for served_model in p.routable_models() {
            *serving_count.entry(served_model).or_default() += 1;
        }
    }

    let my_vram = node.vram_bytes();

    /// Check if a model fits in our VRAM. Returns false and logs if it doesn't.
    fn model_fits(model: &str, my_vram: u64) -> bool {
        let model_path = models::find_model_path(model);
        let model_bytes = std::fs::metadata(&model_path)
            .map(|md| md.len())
            .unwrap_or(0);
        let needed = (model_bytes as f64 * 1.1) as u64;
        if model_bytes > 0 && needed > my_vram {
            let _ = emit_event(OutputEvent::Info {
                message: format!(
                    "Skipping {} — needs {:.1}GB, we have {:.1}GB",
                    model,
                    needed as f64 / 1e9,
                    my_vram as f64 / 1e9
                ),
                context: None,
            });
            return false;
        }
        true
    }

    // Sort demand entries by request_count descending (hottest first)
    let mut demand_sorted: Vec<(String, mesh::ModelDemand)> = demand.into_iter().collect();
    demand_sorted.sort_by_key(|entry| std::cmp::Reverse(entry.1.request_count));

    // Priority 1: Unserved models on disk, ordered by demand
    let mut candidates: Vec<String> = Vec::new();
    for (m, _d) in &demand_sorted {
        if serving_count.get(m).copied().unwrap_or(0) == 0
            && local_models.contains(m)
            && model_fits(m, my_vram)
        {
            candidates.push(m.clone());
        }
    }

    if !candidates.is_empty() {
        // If multiple, pick deterministically so concurrent joiners spread out
        if candidates.len() > 1 {
            let my_id = node.id();
            let id_bytes = my_id.as_bytes();
            let hash = id_bytes
                .iter()
                .fold(0u64, |acc, &b| acc.wrapping_mul(31).wrapping_add(b as u64));
            let idx = (hash as usize) % candidates.len();
            let pick = &candidates[idx];
            let _ = emit_event(OutputEvent::Info {
                message: format!(
                    "Assigned to serve {} (unserved, on disk, {} candidates, by demand)",
                    pick,
                    candidates.len()
                ),
                context: None,
            });
            return Some(pick.clone());
        }
        let pick = &candidates[0];
        let _ = emit_event(OutputEvent::Info {
            message: format!("Assigned to serve {} (unserved, on disk, by demand)", pick),
            context: None,
        });
        return Some(pick.clone());
    }

    // Priority 2: Underserved models on disk (fewer servers than others)
    let max_count = serving_count.values().copied().max().unwrap_or(0);
    let mut underserved: Vec<(String, usize, u64)> = Vec::new(); // (model, servers, demand)
    for (m, d) in &demand_sorted {
        let count = serving_count.get(m).copied().unwrap_or(0);
        if count < max_count && local_models.contains(m) && model_fits(m, my_vram) {
            underserved.push((m.clone(), count, d.request_count));
        }
    }
    if !underserved.is_empty() {
        // Pick the least-served, breaking ties by highest demand
        underserved.sort_by_key(|(_, count, demand)| (*count, std::cmp::Reverse(*demand)));
        let (pick, count, _) = &underserved[0];
        let max_model = serving_count
            .iter()
            .max_by_key(|(_, &v)| v)
            .map(|(k, _)| k.as_str())
            .unwrap_or("?");
        let _ = emit_event(OutputEvent::Info {
            message: format!(
                "Assigned to serve {} ({} servers vs {} has {}) — rebalancing",
                pick, count, max_model, max_count
            ),
            context: None,
        });
        return Some(pick.clone());
    }

    // Priority 3: Unserved models we can download from catalog
    let mut downloadable: Vec<(String, u64)> = Vec::new(); // (model, demand)
    for (m, d) in &demand_sorted {
        if serving_count.get(m).copied().unwrap_or(0) > 0 {
            continue;
        }
        if let Some(cat) = catalog::find_model(m) {
            let size_bytes = parse_size_str(&cat.size);
            let needed = (size_bytes as f64 * 1.1) as u64;
            if needed <= my_vram {
                downloadable.push((m.clone(), d.request_count));
            } else {
                let _ = emit_event(OutputEvent::Info {
                    message: format!(
                        "Skipping {} — needs {:.1}GB, we have {:.1}GB",
                        m,
                        needed as f64 / 1e9,
                        my_vram as f64 / 1e9
                    ),
                    context: None,
                });
            }
        }
    }
    if !downloadable.is_empty() {
        // Pick hottest downloadable, with node-ID hash for tie-breaking
        if downloadable.len() > 1 {
            let my_id = node.id();
            let id_bytes = my_id.as_bytes();
            let hash = id_bytes
                .iter()
                .fold(0u64, |acc, &b| acc.wrapping_mul(31).wrapping_add(b as u64));
            let idx = (hash as usize) % downloadable.len();
            let (pick, _) = &downloadable[idx];
            let _ = emit_event(OutputEvent::Info {
                message: format!(
                    "Assigned to serve {} (unserved, will download, by demand)",
                    pick
                ),
                context: None,
            });
            return Some(pick.clone());
        }
        let (pick, _) = &downloadable[0];
        let _ = emit_event(OutputEvent::Info {
            message: format!(
                "Assigned to serve {} (unserved, will download, by demand)",
                pick
            ),
            context: None,
        });
        return Some(pick.clone());
    }

    // Everything with demand is covered
    let all_covered = demand_sorted
        .iter()
        .all(|(m, _)| serving_count.get(m).copied().unwrap_or(0) > 0);
    if all_covered {
        let _ = emit_event(OutputEvent::Info {
            message: "All demanded models are covered — staying on standby".to_string(),
            context: None,
        });
    }

    None
}

/// Check if a standby node should promote to serve a model.
/// Uses demand signals — promotes for unserved models with active demand,
/// or for demand-based rebalancing when one model is much hotter than others.
///
/// Rebalancing uses `last_active` to gate on recency (only models active within
/// the last 60 minutes are considered), then `request_count / servers` for
/// relative hotness among those recent models.
async fn check_unserved_model(node: &mesh::Node, local_models: &[String]) -> Option<String> {
    let peers = node.peers().await;
    let demand = node.active_demand().await;

    if demand.is_empty() {
        return None;
    }

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let mut serving_count: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();
    for p in &peers {
        for served_model in p.routable_models() {
            *serving_count.entry(served_model).or_default() += 1;
        }
    }

    let my_vram = node.vram_bytes();

    // Only consider models with recent activity (last 60 minutes).
    // This prevents stale cumulative request_count from triggering promotions
    // for models that were popular hours ago but idle now.
    const RECENT_SECS: u64 = 3600;

    // Priority 1: promote for models with active demand and ZERO servers
    // Sort by demand (hottest first)
    let mut unserved: Vec<(String, u64)> = Vec::new();
    for (m, d) in &demand {
        if serving_count.get(m).copied().unwrap_or(0) == 0 && local_models.contains(m) {
            let model_path = models::find_model_path(m);
            let model_bytes = std::fs::metadata(&model_path)
                .map(|md| md.len())
                .unwrap_or(0);
            let needed = (model_bytes as f64 * 1.1) as u64;
            if model_bytes > 0 && needed > my_vram {
                continue;
            }
            unserved.push((m.clone(), d.request_count));
        }
    }
    if !unserved.is_empty() {
        unserved.sort_by_key(|(_, count)| std::cmp::Reverse(*count));
        return Some(unserved[0].0.clone());
    }

    // Priority 2: demand-based rebalancing.
    // Only consider models with recent activity, then use request_count / servers
    // for relative hotness. Promote if one model is significantly hotter than others.
    let mut ratios: Vec<(String, f64)> = Vec::new();
    for (m, d) in &demand {
        if now.saturating_sub(d.last_active) > RECENT_SECS {
            continue;
        }
        let servers = serving_count.get(m).copied().unwrap_or(0) as f64;
        if servers > 0.0 && d.request_count > 0 && local_models.contains(m) {
            let model_path = models::find_model_path(m);
            let model_bytes = std::fs::metadata(&model_path)
                .map(|md| md.len())
                .unwrap_or(0);
            let needed = (model_bytes as f64 * 1.1) as u64;
            if model_bytes > 0 && needed > my_vram {
                continue;
            }
            ratios.push((m.clone(), d.request_count as f64 / servers));
        }
    }

    if !ratios.is_empty() {
        ratios.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let (hottest_model, hottest_ratio) = &ratios[0];
        let coldest_ratio = if ratios.len() >= 2 {
            ratios[ratios.len() - 1].1
        } else {
            0.0
        };
        let should_promote = if ratios.len() >= 2 {
            *hottest_ratio >= coldest_ratio * 3.0 && *hottest_ratio >= 10.0
        } else {
            *hottest_ratio >= 10.0
        };

        if should_promote {
            let _ = emit_event(OutputEvent::Info {
                message: format!(
                    "Promoting to serve {} — demand {:.0} req/server (coldest: {:.0})",
                    hottest_model, hottest_ratio, coldest_ratio
                ),
                context: None,
            });
            return Some(hottest_model.clone());
        }
    }

    None
}

pub(crate) fn load_resolved_plugins(cli: &Cli) -> Result<plugin::ResolvedPlugins> {
    let config = plugin::load_config(cli.config.as_deref())?;
    resolve_plugins_from_config(&config, cli)
}

fn resolve_plugins_from_config(
    config: &plugin::MeshConfig,
    cli: &Cli,
) -> Result<plugin::ResolvedPlugins> {
    plugin::resolve_plugins(config, plugin_host_mode(cli))
}

fn plugin_host_mode(cli: &Cli) -> plugin::PluginHostMode {
    plugin::PluginHostMode {
        mesh_visibility: if cli.publish || cli.nostr_discovery {
            mesh_llm_plugin::MeshVisibility::Public
        } else {
            mesh_llm_plugin::MeshVisibility::Private
        },
    }
}

fn node_display_name(cli: &Cli, node: &mesh::Node) -> String {
    cli.name
        .clone()
        .or_else(|| std::env::var("USER").ok())
        .or_else(|| std::env::var("USERNAME").ok())
        .unwrap_or_else(|| node.id().fmt_short().to_string())
}

async fn join_mesh_for_mcp(cli: &Cli, node: &mesh::Node) -> Result<()> {
    if !cli.join.is_empty() {
        for token in &cli.join {
            match node.join(token).await {
                Ok(()) => {
                    if node.mesh_id().await.is_some() {
                        record_first_joined_mesh_ts(node).await;
                    }
                    let _ = emit_event(OutputEvent::Info {
                        message: "Joined mesh".to_string(),
                        context: None,
                    });
                    return Ok(());
                }
                Err(err) => tracing::warn!("Failed to join via token: {err}"),
            }
        }
        anyhow::bail!("Failed to join any peer for MCP mode");
    }

    if cli.auto || cli.discover.is_some() {
        let relays = nostr_relays(&cli.nostr_relay);
        let filter = nostr::MeshFilter {
            model: None,
            min_vram_gb: None,
            region: cli.region.clone(),
        };
        let target_name = cli.discover.as_deref().or(cli.mesh_name.as_deref());
        let _ = emit_event(OutputEvent::DiscoveryStarting {
            source: "Nostr discovery".to_string(),
        });
        let meshes = match nostr::discover(&relays, &filter, None).await {
            Ok(meshes) => meshes,
            Err(err) => {
                let _ = emit_event(OutputEvent::DiscoveryFailed {
                    message: "Nostr discovery failed".to_string(),
                    detail: Some(err.to_string()),
                });
                return Err(err);
            }
        };
        match nostr::smart_auto(&meshes, 0.0, target_name) {
            nostr::AutoDecision::Join { candidates } => {
                let mut last_err: Option<anyhow::Error> = None;
                for (token, mesh) in &candidates {
                    let _ = emit_event(OutputEvent::MeshFound {
                        mesh: mesh
                            .listing
                            .name
                            .as_deref()
                            .unwrap_or("unnamed")
                            .to_string(),
                        peers: mesh.listing.node_count,
                        region: mesh.listing.region.clone(),
                    });
                    match node.join(token).await {
                        Ok(()) => {
                            if node.mesh_id().await.is_some() {
                                record_first_joined_mesh_ts(node).await;
                            }
                            let _ = emit_event(OutputEvent::DiscoveryJoined {
                                mesh: mesh
                                    .listing
                                    .name
                                    .as_deref()
                                    .unwrap_or("unnamed")
                                    .to_string(),
                            });
                            last_err = None;
                            break;
                        }
                        Err(err) => {
                            let _ = emit_event(OutputEvent::DiscoveryFailed {
                                message: format!(
                                    "Failed to join mesh {}",
                                    mesh.listing.name.as_deref().unwrap_or("unnamed")
                                ),
                                detail: Some(err.to_string()),
                            });
                            tracing::warn!("Failed to join mesh candidate: {err}");
                            last_err = Some(err);
                        }
                    }
                }
                if let Some(err) = last_err {
                    return Err(err);
                }
            }
            nostr::AutoDecision::StartNew { .. } => {
                let _ = emit_event(OutputEvent::DiscoveryFailed {
                    message: "No mesh found for MCP mode".to_string(),
                    detail: Some("Pass --join or start a mesh first.".to_string()),
                });
                anyhow::bail!("No mesh found for MCP mode. Pass --join or start a mesh first.");
            }
        }
    }

    Ok(())
}

pub(crate) async fn run_plugin_mcp(cli: &Cli) -> Result<()> {
    let resolved_plugins = load_resolved_plugins(cli)?;
    let owner_config = owner_runtime_config(cli)?;
    let (node, _channels) = mesh::Node::start(
        NodeRole::Client,
        &cli.relay,
        cli.bind_port,
        Some(0.0),
        !cli.no_enumerate_host,
        Some(owner_config),
        cli.config.as_deref(),
    )
    .await?;
    node.start_accepting();
    node.set_display_name(node_display_name(cli, &node)).await;
    node.start_heartbeat();
    node.start_relay_health_monitor();
    join_mesh_for_mcp(cli, &node).await?;

    let (plugin_mesh_tx, plugin_mesh_rx) = tokio::sync::mpsc::channel(256);
    let plugin_manager =
        plugin::PluginManager::start(&resolved_plugins, plugin_host_mode(cli), plugin_mesh_tx)
            .await?;
    node.set_plugin_manager(plugin_manager.clone()).await;
    node.start_plugin_channel_forwarder(plugin_mesh_rx);

    if plugin_manager.list().await.is_empty() {
        tracing::warn!("No plugins are enabled for MCP exposure");
    }

    plugin::mcp::run_mcp_server(plugin_manager).await
}

pub(crate) use self::discovery::{check_mesh, nostr_relays};

async fn store_benchmark_metrics(
    mem_arc: std::sync::Arc<tokio::sync::Mutex<Option<Vec<f64>>>>,
    fp32_arc: std::sync::Arc<tokio::sync::Mutex<Option<Vec<f64>>>>,
    fp16_arc: std::sync::Arc<tokio::sync::Mutex<Option<Vec<f64>>>>,
    result: Option<&benchmark::BenchmarkResult>,
) {
    *mem_arc.lock().await = result.map(|r| r.mem_bandwidth_gbps.clone());
    *fp32_arc.lock().await = result.and_then(|r| r.compute_tflops_fp32.clone());
    *fp16_arc.lock().await = result.and_then(|r| r.compute_tflops_fp16.clone());
}

/// Serve mode: join the mesh and serve local models through the embedded runtime.
async fn run_auto(
    mut cli: Cli,
    config: plugin::MeshConfig,
    startup_models: Vec<StartupModelPlan>,
    requested_model_names: Vec<String>,
    bin_dir: PathBuf,
    runtime: Option<std::sync::Arc<crate::runtime::instance::InstanceRuntime>>,
    auto_join_candidates: Vec<(String, Option<String>)>,
) -> Result<()> {
    let resolved_plugins = resolve_plugins_from_config(&config, &cli)?;
    let api_port = cli.port;
    // Export management API port for runtime hook callbacks.
    // Must be the console/management port (default 3131), not the proxy port
    // (default 9337), so callbacks do not loop through the OpenAI surface.
    std::env::set_var("MESH_API_PORT", cli.console.to_string());
    skippy::configure_materialized_stage_cache();
    let console_port = Some(cli.console);
    let is_client = cli.client;
    let resolved_models: Vec<PathBuf> = startup_models
        .iter()
        .map(|model| model.resolved_path.clone())
        .collect();

    // Scan local models on disk
    let local_models = if is_client {
        vec![]
    } else {
        models::scan_local_models()
    };
    tracing::info!("Local models on disk: {:?}", local_models);

    // Start mesh node — clients use ephemeral key (unique identity per run)
    let role = if is_client {
        NodeRole::Client
    } else {
        NodeRole::Worker
    };
    let owner_config = owner_runtime_config(&cli)?;
    // Clients report 0 VRAM so they're never assigned a model to serve
    let max_vram = if is_client { Some(0.0) } else { cli.max_vram };
    let (node, channels) = mesh::Node::start(
        role,
        &cli.relay,
        cli.bind_port,
        max_vram,
        !cli.no_enumerate_host,
        Some(owner_config),
        cli.config.as_deref(),
    )
    .await?;
    node.set_stage_control_sender(skippy::spawn_stage_control_loop())
        .await;
    node.start_accepting();
    let token = node.invite_token();
    node.set_display_name(node_display_name(&cli, &node)).await;
    let (plugin_mesh_tx, plugin_mesh_rx) = tokio::sync::mpsc::channel(256);
    let plugin_manager =
        plugin::PluginManager::start(&resolved_plugins, plugin_host_mode(&cli), plugin_mesh_tx)
            .await?;
    node.set_plugin_manager(plugin_manager.clone()).await;
    node.start_plugin_channel_forwarder(plugin_mesh_rx);

    // Advertise what we have on disk and what we want the mesh to serve
    node.set_available_models(local_models.clone()).await;
    node.set_requested_models(requested_model_names.clone())
        .await;

    // Start periodic health check to detect dead peers
    node.start_heartbeat();
    node.start_relay_health_monitor();

    // Launch memory bandwidth benchmark in background (non-blocking)
    // Skip for client nodes — they have no GPU to benchmark
    if !is_client {
        let mem_arc = node.gpu_mem_bandwidth_gbps.clone();
        let compute_fp32_arc = node.gpu_compute_tflops_fp32.clone();
        let compute_fp16_arc = node.gpu_compute_tflops_fp16.clone();
        let bin_dir_clone = bin_dir.clone();
        let node_bench = node.clone();
        tokio::spawn(async move {
            let result = tokio::time::timeout(
                std::time::Duration::from_secs(30),
                tokio::task::spawn_blocking(move || {
                    let hw = hardware::survey();
                    if hw.gpu_count == 0 {
                        tracing::debug!("no GPUs detected — skipping memory bandwidth benchmark");
                        return None;
                    }
                    benchmark::run_or_load(&hw, &bin_dir_clone, benchmark::BENCHMARK_TIMEOUT)
                }),
            )
            .await
            .map_err(|_| {
                tracing::warn!("benchmark timed out after 30s — bandwidth will not be gossiped")
            })
            .ok()
            .and_then(|r| r.ok())
            .flatten();

            if let Some(ref run) = result {
                let total: f64 = run.mem_bandwidth_gbps.iter().sum();
                tracing::info!(
                    "Memory bandwidth fingerprint: {} GPUs, {:.1} GB/s total",
                    run.mem_bandwidth_gbps.len(),
                    total
                );
                for (i, gbps) in run.mem_bandwidth_gbps.iter().enumerate() {
                    tracing::debug!("  GPU {}: {:.1} GB/s", i, gbps);
                }
                if let Some(fp32s) = &run.compute_tflops_fp32 {
                    let total_fp32: f64 = fp32s.iter().sum();
                    tracing::info!(
                        "Compute FP32 TFLOPS: {} GPUs, {:.1} TFLOPS total",
                        fp32s.len(),
                        total_fp32
                    );
                    for (i, tf) in fp32s.iter().enumerate() {
                        tracing::debug!("  GPU {}: {:.1} TF32", i, tf);
                    }
                }
                if let Some(fp16s) = &run.compute_tflops_fp16 {
                    let total_fp16: f64 = fp16s.iter().sum();
                    tracing::info!(
                        "Compute FP16 TFLOPS: {} GPUs, {:.1} TFLOPS total",
                        fp16s.len(),
                        total_fp16
                    );
                    for (i, tf) in fp16s.iter().enumerate() {
                        tracing::debug!("  GPU {}: {:.1} TF16", i, tf);
                    }
                }
            }
            store_benchmark_metrics(
                mem_arc.clone(),
                compute_fp32_arc.clone(),
                compute_fp16_arc.clone(),
                result.as_ref(),
            )
            .await;
            node_bench.regossip().await;
        });
    } else {
        tracing::debug!("client node — skipping memory bandwidth benchmark");
    }

    // Join mesh if --join was given or auto-discovery queued fallback candidates.
    if !cli.join.is_empty() || !auto_join_candidates.is_empty() {
        let mut joined = false;
        let join_attempts: Vec<(String, Option<String>)> = if !cli.join.is_empty() {
            cli.join
                .iter()
                .cloned()
                .map(|token| (token, None))
                .collect()
        } else {
            auto_join_candidates.clone()
        };
        let mut successful_join: Option<(String, Option<String>)> = None;

        for (t, mesh_name) in &join_attempts {
            match node.join(t).await {
                Ok(()) => {
                    if node.mesh_id().await.is_some() {
                        record_first_joined_mesh_ts(&node).await;
                    }
                    let _ = emit_event(OutputEvent::Info {
                        message: "Joined mesh".to_string(),
                        context: None,
                    });
                    joined = true;
                    successful_join = Some((t.clone(), mesh_name.clone()));
                    break;
                }
                Err(e) => tracing::warn!("Failed to join via token: {e}"),
            }
        }

        if cli.join.is_empty() {
            cli.join.clear();
            if let Some((token, mesh_name)) = successful_join {
                cli.join.push(token);
                if cli.mesh_name.is_none() {
                    if let Some(name) = mesh_name {
                        cli.mesh_name = Some(name);
                    }
                }
            }
        }

        if !joined {
            let _ = emit_event(OutputEvent::Warning {
                message: "Failed to join any peer — running standalone".to_string(),
                context: None,
            });
        }

        // Save mesh_id for sticky preference after gossip propagates it
        {
            let save_node = node.clone();
            tokio::spawn(async move {
                // Wait for gossip to propagate mesh_id
                tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                if let Some(id) = save_node.mesh_id().await {
                    record_first_joined_mesh_ts(&save_node).await;
                    mesh::save_last_mesh_id(&id);
                    tracing::info!("Mesh ID: {id}");
                }
            });
        }

        let mesh_id = node
            .mesh_id()
            .await
            .unwrap_or_else(|| "pending".to_string());
        let _ = emit_event(OutputEvent::InviteToken {
            token: token.clone(),
            mesh_id,
            mesh_name: cli.mesh_name.clone(),
        });

        // Periodic rejoin: re-connect to bootstrap tokens every 60s.
        // No-op if already connected (connect_to_peer returns early).
        // Recovers from dropped connections without manual intervention.
        let rejoin_node = node.clone();
        let rejoin_tokens: Vec<String> = cli.join.clone();
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(std::time::Duration::from_secs(60)).await;
                for t in &rejoin_tokens {
                    if let Err(e) = rejoin_node.join(t).await {
                        tracing::debug!("Rejoin failed: {e}");
                    }
                }
            }
        });

        // Nostr re-discovery: if we joined via --auto (Nostr discovery) and lose
        // all peers, re-discover and join a new mesh. This handles the case where
        // the original mesh publisher restarts with a new identity.
        if cli.auto {
            let rediscover_node = node.clone();
            let rediscover_relays = nostr_relays(&cli.nostr_relay);
            let rediscover_relay_urls = cli.relay.clone();
            let rediscover_mesh_name = cli.mesh_name.clone();
            tokio::spawn(async move {
                nostr_rediscovery(
                    rediscover_node,
                    rediscover_relays,
                    rediscover_relay_urls,
                    rediscover_mesh_name,
                )
                .await;
            });
        }
    } else {
        // Originator — generate mesh_id
        let nostr_pubkey = if cli.publish {
            nostr::load_or_create_keys()
                .ok()
                .map(|k| k.public_key().to_hex())
        } else {
            None
        };
        let mesh_id = mesh::generate_mesh_id(cli.mesh_name.as_deref(), nostr_pubkey.as_deref());
        node.set_mesh_id_force(mesh_id.clone()).await;
        record_first_joined_mesh_ts(&node).await;
        mesh::save_last_mesh_id(&mesh_id);
        tracing::info!("Mesh ID: {mesh_id}");
        let _ = emit_event(OutputEvent::InviteToken {
            token: token.clone(),
            mesh_id: mesh_id.clone(),
            mesh_name: cli.mesh_name.clone(),
        });
        let _ = emit_event(OutputEvent::WaitingForPeers { detail: None });

        // Originator also re-discovers: if we started solo and a matching mesh
        // already exists on Nostr, we should join it instead of staying alone.
        if cli.auto {
            let rediscover_node = node.clone();
            let rediscover_relays = nostr_relays(&cli.nostr_relay);
            let rediscover_relay_urls = cli.relay.clone();
            let rediscover_mesh_name = cli.mesh_name.clone();
            tokio::spawn(async move {
                nostr_rediscovery(
                    rediscover_node,
                    rediscover_relays,
                    rediscover_relay_urls,
                    rediscover_mesh_name,
                )
                .await;
            });
        }
    }

    let affinity_router = affinity::AffinityRouter::new();

    // Start bootstrap proxy if joining an existing mesh.
    // This gives instant API access via tunnel while our GPU loads.
    let mut bootstrap_listener_tx = if !cli.join.is_empty() {
        let (stop_tx, stop_rx) =
            tokio::sync::mpsc::channel::<tokio::sync::oneshot::Sender<tokio::net::TcpListener>>(1);
        let boot_node = node.clone();
        let boot_port = api_port;
        let boot_affinity = affinity_router.clone();
        tokio::spawn(async move {
            bootstrap_proxy(boot_node, boot_port, stop_rx, cli.listen_all, boot_affinity).await;
        });
        Some(stop_tx)
    } else {
        None
    };

    let primary_startup_model = startup_models.first().cloned();

    // Decide which model THIS node will serve
    let model = if let Some(primary) = primary_startup_model.as_ref() {
        // First startup model is what we serve (already resolved/downloaded)
        primary.resolved_path.clone()
    } else {
        // No --model: try to find a model on disk that the mesh needs
        let _ = emit_event(OutputEvent::WaitingForPeers {
            detail: Some("No --model specified, checking local models against mesh...".to_string()),
        });

        // Give gossip a moment to propagate
        tokio::time::sleep(std::time::Duration::from_secs(5)).await;

        let assignment = pick_model_assignment(&node, &local_models).await;
        // If no demand-based assignment but we have VRAM, use auto pack's primary model
        let assignment = if assignment.is_none() && cli.auto && !is_client {
            let pack = nostr::auto_model_pack(node.vram_bytes() as f64 / 1e9);
            if !pack.is_empty() {
                Some(pack[0].clone())
            } else {
                assignment
            }
        } else {
            assignment
        };
        if let Some(model_name) = assignment {
            let _ = emit_event(OutputEvent::HostElected {
                model: model_name.clone(),
                host: node.id().fmt_short().to_string(),
                role: Some("host".to_string()),
                capacity_gb: Some(node.vram_bytes() as f64 / 1e9),
            });
            let model_path = models::find_model_path(&model_name);
            if model_path.exists() {
                model_path
            } else if let Some(cat) = catalog::find_model(&model_name) {
                // Model not on disk but in catalog — download it
                let _ = emit_event(OutputEvent::Info {
                    message: format!("Downloading {model_name} for mesh..."),
                    context: None,
                });
                catalog::download_model(cat).await?
            } else {
                model_path
            }
        } else {
            // Nothing on disk matches — go passive, act as proxy
            // Stop bootstrap proxy first (run_passive binds its own listener)
            drop(bootstrap_listener_tx.take());
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            // If a model becomes unserved while we're standby, we'll promote
            if is_client {
                let _ = emit_event(OutputEvent::PassiveMode {
                    role: "client".to_string(),
                    status: RuntimeStatus::Starting,
                    capacity_gb: None,
                    models_on_disk: None,
                    detail: Some("Running as client — proxying requests to mesh".to_string()),
                });
            } else {
                let _ = emit_event(OutputEvent::PassiveMode {
                    role: "standby".to_string(),
                    status: RuntimeStatus::Starting,
                    capacity_gb: Some(node.vram_bytes() as f64 / 1e9),
                    models_on_disk: Some(local_models.clone()),
                    detail: Some(
                        "No matching model on disk — running as standby GPU node. Proxying requests to other nodes. Will activate when needed."
                            .to_string(),
                    ),
                });
            }
            match run_passive(&cli, node.clone(), is_client, plugin_manager.clone()).await? {
                Some(model_name) => {
                    // Promoted! Resolve the model path and continue to serving
                    models::find_model_path(&model_name)
                }
                None => return Ok(()), // clean shutdown
            }
        }
    };

    let model_name = {
        let stem = model
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
        // Strip split GGUF suffix: "MiniMax-M2.5-Q4_K_M-00001-of-00004" → "MiniMax-M2.5-Q4_K_M"
        router::strip_split_suffix_owned(&stem)
    };

    // Set model source for gossip (so other joiners can discover it too)
    let model_source = primary_startup_model
        .as_ref()
        .map(|model| model.declared_ref.clone())
        .unwrap_or_else(|| model_name.clone());
    node.set_model_source(model_source).await;
    // Declare which models this node may serve, but do not advertise them as
    // live/routable until their local processes have passed health checks.
    let all_declared = build_serving_list(&resolved_models, &model_name);
    node.set_serving_models(all_declared.clone()).await;
    node.set_hosted_models(Vec::new()).await;
    node.set_models(all_declared.clone()).await;
    // Re-gossip so peers learn our catalog/requested state without prematurely
    // routing requests to not-yet-ready local processes.
    node.regossip().await;

    // Drain stale pidfiles from our own runtime dir before starting local runtime models.
    if let Some(ref rt) = runtime {
        let _ = crate::runtime::instance::reap::reap_own_stale_pidfiles(rt.dir()).await;
    }

    // Serve mode (non-client) always has the InstanceRuntime acquired above.
    // The fallback was only relevant during the T1-T11 staging when acquisition
    // wasn't yet wired into run() — keep an explicit error here so any future
    // refactor that drops the acquire surfaces immediately instead of panicking
    // mid-spawn from a child task.
    let _runtime = runtime
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("serve mode requires an instance runtime"))?
        .clone();

    let tunnel_mgr =
        tunnel::Manager::start(node.clone(), channels.rpc, channels.http, channels.stage).await?;

    // Election publishes per-model targets
    let (target_tx, target_rx) = tokio::sync::watch::channel(election::ModelTargets::default());
    let target_tx = std::sync::Arc::new(target_tx);

    // Runtime control for local load/unload of extra models.
    let (control_tx, mut control_rx) =
        tokio::sync::mpsc::unbounded_channel::<api::RuntimeControlRequest>();
    let (runtime_event_tx, mut runtime_event_rx) =
        tokio::sync::mpsc::unbounded_channel::<RuntimeEvent>();
    let mut runtime_models: HashMap<String, LocalRuntimeModelHandle> = HashMap::new();
    let mut managed_models: HashMap<String, ManagedModelController> = HashMap::new();
    let dashboard_processes = Arc::new(tokio::sync::Mutex::new(Vec::new()));
    let input_handler_enabled = crate::cli::output::OutputManager::global()
        .console_session_mode()
        .is_some();

    // Take over listener from bootstrap proxy (if running), or bind a new one
    let existing_listener = if let Some(tx) = bootstrap_listener_tx {
        let (resp_tx, resp_rx) = tokio::sync::oneshot::channel();
        let _ = tx.send(resp_tx).await;
        // Wait for bootstrap to hand back the TcpListener
        resp_rx.await.ok()
    } else {
        None
    };

    // API proxy: model-aware routing
    let proxy_node = node.clone();
    let proxy_rx = target_rx.clone();
    let proxy_affinity = affinity_router.clone();
    let api_control_tx = control_tx.clone();
    let api_proxy_handle = tokio::spawn(async move {
        api_proxy(
            proxy_node,
            api_port,
            proxy_rx,
            api_control_tx,
            existing_listener,
            cli.listen_all,
            proxy_affinity,
        )
        .await;
    });

    // Console (optional)
    let model_name_for_console = model_name.clone();
    let mut console_server_handle = None;
    let console_state = if let Some(cport) = console_port {
        let model_size_bytes = election::total_model_bytes(&model);
        let runtime_data_collector = node.runtime_data_collector();
        let runtime_data_producer =
            runtime_data_collector.producer(crate::runtime_data::RuntimeDataSource {
                scope: "runtime",
                plugin_data_key: None,
                plugin_endpoint_key: None,
            });
        let cs = api::MeshApi::new(api::MeshApiConfig {
            node: node.clone(),
            model_name: model_name_for_console.clone(),
            api_port,
            model_size_bytes,
            plugin_manager: plugin_manager.clone(),
            affinity_router: affinity_router.clone(),
            runtime_data_collector,
            runtime_data_producer,
        });
        cs.set_primary_backend("skippy".into()).await;
        cs.set_runtime_control(control_tx.clone()).await;
        cs.set_nostr_relays(nostr_relays(&cli.nostr_relay)).await;
        cs.set_nostr_discovery(cli.nostr_discovery).await;
        if let Some(draft) = &cli.draft {
            let dn = draft
                .file_stem()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();
            cs.set_draft_name(dn).await;
        }
        if let Some(ref name) = cli.mesh_name {
            cs.set_mesh_name(name.clone()).await;
        }
        let cs2 = cs.clone();
        let console_rx = target_rx.clone();
        let mn = model_name_for_console.clone();
        console_server_handle = Some(tokio::spawn(async move {
            // Console still takes old-style InferenceTarget for now — adapt
            let (adapted_tx, adapted_rx) =
                tokio::sync::watch::channel(election::InferenceTarget::None);
            tokio::spawn(async move {
                let mut rx = console_rx;
                loop {
                    let targets = rx.borrow().clone();
                    let target = targets.get(&mn);
                    adapted_tx.send_replace(target);
                    if rx.changed().await.is_err() {
                        break;
                    }
                }
            });
            api::start(cport, cs2, adapted_rx, cli.listen_all, cli.headless).await;
        }));
        Some(cs)
    } else {
        None
    };

    crate::cli::output::OutputManager::global().register_dashboard_snapshot_provider(Arc::new(
        RuntimeDashboardSnapshotProvider::new(
            node.clone(),
            dashboard_processes.clone(),
            Some(plugin_manager.clone()),
            api_port,
            console_port,
            cli.headless,
        ),
    ));

    if !is_client {
        if let Some(ref cs) = console_state {
            if let Ok(root) = crate::runtime::instance::runtime_root() {
                let runtime_data_producer = cs.runtime_data_producer().await;
                if let Ok(initial) =
                    crate::runtime::instance::scan_local_instances(&root, std::process::id()).await
                {
                    crate::runtime::instance::publish_local_instance_scan_results(
                        &runtime_data_producer,
                        initial,
                    );
                }
                crate::runtime::instance::spawn_local_instance_scanner(
                    root,
                    std::process::id(),
                    runtime_data_producer,
                );
            }
        }
    }

    tracing::info!("Starting embedded runtime for model: {model_name}");
    let node2 = node.clone();
    let tunnel_mgr2 = tunnel_mgr.clone();
    let model2 = model.clone();
    let slots = primary_startup_model
        .as_ref()
        .and_then(|m| m.parallel)
        .or(config.gpu.parallel)
        .unwrap_or(4);
    let cb_console_port = console_port;
    let model_name_for_election = model_name.clone();
    let primary_target_tx = target_tx.clone();
    let console_state_for_election = console_state.clone();
    let interactive_console_state = console_state.clone();
    let interactive_control_tx = control_tx.clone();
    let interactive_started = Arc::new(AtomicBool::new(false));
    let primary_model_name_for_advertise = model_name.clone();
    let startup_model_names: Vec<String> = startup_models
        .iter()
        .map(|model| {
            router::strip_split_suffix_owned(
                &model
                    .resolved_path
                    .file_stem()
                    .unwrap_or_default()
                    .to_string_lossy(),
            )
        })
        .collect();
    let startup_ready_reporter = StartupReadyReporter::new(
        &startup_model_names,
        model_name.clone(),
        api_port,
        cb_console_port,
    );
    let startup_load_gate = Arc::new(tokio::sync::Mutex::new(()));
    let primary_startup_ready_reporter = startup_ready_reporter.clone();
    let primary_mmproj = primary_startup_model
        .as_ref()
        .and_then(|model| model.mmproj_path.clone());
    let primary_ctx_size = primary_startup_model
        .as_ref()
        .and_then(|model| model.ctx_size);
    let primary_pinned_gpu = primary_startup_model
        .as_ref()
        .and_then(|model| model.pinned_gpu.clone());
    let (primary_stop_tx, primary_stop_rx) = tokio::sync::watch::channel(false);
    let dashboard_processes_for_primary_task = dashboard_processes.clone();
    let primary_startup_load_gate = startup_load_gate.clone();
    let primary_task = tokio::spawn(async move {
        startup_local_model_loop(StartupLocalModelTask {
            node: node2,
            tunnel_mgr: tunnel_mgr2,
            target_tx: primary_target_tx,
            model_path: model2,
            model_name: model_name_for_election,
            primary_model_name: primary_model_name_for_advertise,
            mmproj_path: primary_mmproj,
            ctx_size: primary_ctx_size,
            pinned_gpu: primary_pinned_gpu,
            slots,
            stop_rx: primary_stop_rx,
            dashboard_processes: dashboard_processes_for_primary_task,
            console_state: console_state_for_election,
            api_port,
            startup_ready_reporter: primary_startup_ready_reporter,
            startup_load_gate: primary_startup_load_gate,
            input_handler_enabled,
            interactive_started,
            interactive_control_tx,
            interactive_console_state,
        })
        .await;
    });
    managed_models.insert(
        model_name.clone(),
        ManagedModelController {
            stop_tx: primary_stop_tx,
            task: primary_task,
        },
    );

    // Additional model election loops (multi-model per node)
    // Each additional model gets its own embedded runtime task.
    // They share the same target_tx so the proxy sees all models.
    if startup_models.len() > 1 {
        // Announce all models to mesh
        let all_names: Vec<String> = startup_models
            .iter()
            .map(|m| {
                m.resolved_path
                    .file_stem()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string()
            })
            .collect();
        let _ = emit_event(OutputEvent::MultiModelMode {
            count: all_names.len(),
            models: all_names.clone(),
        });
        node.set_models(all_names).await;
        node.regossip().await;

        for extra_model in startup_models.iter().skip(1) {
            let extra_name = {
                let stem = extra_model
                    .resolved_path
                    .file_stem()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string();
                router::strip_split_suffix_owned(&stem)
            };
            let extra_node = node.clone();
            let extra_tunnel = tunnel_mgr.clone();
            let extra_path = extra_model.resolved_path.clone();
            let extra_mmproj = extra_model.mmproj_path.clone();
            let extra_ctx_size = extra_model.ctx_size;
            let extra_pinned_gpu = extra_model.pinned_gpu.clone();
            let extra_target_tx = target_tx.clone();
            let extra_model_name = extra_name.clone();
            let api_port_extra = api_port;
            let slots = extra_model.parallel.or(config.gpu.parallel).unwrap_or(4);
            let extra_console_state = console_state.clone();
            let extra_startup_ready_reporter = startup_ready_reporter.clone();
            let extra_startup_load_gate = startup_load_gate.clone();
            let primary_model_name_for_extra = model_name.clone();
            let managed_model_name = extra_name.clone();
            let (extra_stop_tx, extra_stop_rx) = tokio::sync::watch::channel(false);
            let dashboard_processes_for_extra_task = dashboard_processes.clone();
            let extra_control_tx = control_tx.clone();
            let extra_task = tokio::spawn(async move {
                startup_local_model_loop(StartupLocalModelTask {
                    node: extra_node,
                    tunnel_mgr: extra_tunnel,
                    target_tx: extra_target_tx,
                    model_path: extra_path,
                    model_name: extra_model_name,
                    primary_model_name: primary_model_name_for_extra,
                    mmproj_path: extra_mmproj,
                    ctx_size: extra_ctx_size,
                    pinned_gpu: extra_pinned_gpu,
                    slots,
                    stop_rx: extra_stop_rx,
                    dashboard_processes: dashboard_processes_for_extra_task,
                    console_state: extra_console_state,
                    api_port: api_port_extra,
                    startup_ready_reporter: extra_startup_ready_reporter,
                    startup_load_gate: extra_startup_load_gate,
                    input_handler_enabled: false,
                    interactive_started: Arc::new(AtomicBool::new(true)),
                    interactive_control_tx: extra_control_tx,
                    interactive_console_state: None,
                })
                .await;
            });
            managed_models.insert(
                managed_model_name,
                ManagedModelController {
                    stop_tx: extra_stop_tx,
                    task: extra_task,
                },
            );
        }
    }

    // Nostr publish loop (if --publish) or watchdog (if --auto, to take over if publisher dies)
    let nostr_publisher = if cli.publish {
        match nostr::load_or_create_keys() {
            Ok(nostr_keys) => {
                let relays = nostr_relays(&cli.nostr_relay);
                let pub_node = node.clone();
                let pub_name = cli.mesh_name.clone();
                let pub_region = cli.region.clone();
                let pub_max_clients = cli.max_clients;
                let (status_tx, status_rx) = tokio::sync::watch::channel(None);
                if let Some(ref cs) = console_state {
                    bridge_publication_state(cs.clone(), status_rx);
                }
                Some(tokio::spawn(async move {
                    nostr::publish_loop(
                        pub_node,
                        nostr_keys,
                        nostr::PublishLoopConfig {
                            relays,
                            name: pub_name,
                            region: pub_region,
                            max_clients: pub_max_clients,
                            interval_secs: 60,
                            status_tx: Some(status_tx),
                        },
                    )
                    .await;
                }))
            }
            Err(e) => {
                let _ = emit_event(OutputEvent::Warning {
                    message: format!(
                        "Publishing to Nostr failed: {e}. Mesh is running privately — add --publish after fixing the issue to make discoverable."
                    ),
                    context: cli.mesh_name.as_ref().map(|mesh_name| format!("mesh={mesh_name}")),
                });
                tracing::warn!("Nostr publish failed: {e}");
                if let Some(ref cs) = console_state {
                    cs.set_publication_state(api::PublicationState::PublishFailed)
                        .await;
                }
                None
            }
        }
    } else if cli.auto {
        // Watchdog: if we joined via --auto, watch for the publisher to die and take over
        let relays = nostr_relays(&cli.nostr_relay);
        let wd_node = node.clone();
        let wd_name = cli.mesh_name.clone();
        let wd_region = cli.region.clone();
        let watchdog_status_rx = console_state.as_ref().map(|cs| {
            let (status_tx, status_rx) = tokio::sync::watch::channel(None);
            bridge_publication_state(cs.clone(), status_rx);
            status_tx
        });
        Some(tokio::spawn(async move {
            nostr::publish_watchdog(wd_node, relays, wd_name, wd_region, 120, watchdog_status_rx)
                .await;
        }))
    } else {
        None
    };

    // Wait for SIGINT/SIGTERM or runtime model control commands.
    let primary_model_name = model_name.clone();
    loop {
        tokio::select! {
            _ = wait_shutdown_signal() => {
                emit_shutdown(None);
                break;
            }
            Some(cmd) = control_rx.recv() => {
                match cmd {
                    api::RuntimeControlRequest::Load { spec, resp } => {
                        let mut assigned_runtime_model: Option<String> = None;
                        let result = async {
                            let model_path = resolve_model(&PathBuf::from(&spec)).await?;
                            let runtime_model_name = resolved_model_name(&model_path);
                            let already_loaded = managed_models.contains_key(&runtime_model_name)
                                || runtime_models.contains_key(&runtime_model_name);
                            anyhow::ensure!(
                                !already_loaded,
                                "model '{runtime_model_name}' is already loaded"
                            );

                            // Look up per-model parallel from TOML config by matching the
                            // spec string against [[models]].model entries. Falls back to
                            // gpu.parallel or default 4 when no entry matches.
                            let slots = config
                                .models
                                .iter()
                                .find(|m| m.model == spec)
                                .and_then(|m| m.parallel)
                                .or(config.gpu.parallel)
                                .unwrap_or(4);

                            assigned_runtime_model = Some(runtime_model_name.clone());
                            add_serving_assignment(&node, &primary_model_name, &runtime_model_name)
                                .await;
                            let (loaded_name, handle, death_rx) = start_runtime_local_model(
                                LocalRuntimeModelStartSpec {
                                    node: &node,
                                    model_path: &model_path,
                                    mmproj_override: None,
                                    ctx_size_override: cli.ctx_size,
                                    pinned_gpu: None,
                                    slots,
                                },
                            )
                            .await?;

                            add_runtime_local_target(&target_tx, &loaded_name, handle.port);
                            set_advertised_model_context(
                                &node,
                                &loaded_name,
                                Some(handle.context_length),
                            )
                            .await;
                            advertise_model_ready(&node, &primary_model_name, &loaded_name).await;
                            node.set_available_models(models::scan_local_models()).await;
                            let payload = local_process_payload(
                                &loaded_name,
                                &handle.backend,
                                handle.port,
                                handle.pid(),
                                slots,
                                handle.context_length,
                            );
                            upsert_dashboard_process(&dashboard_processes, payload.clone())
                                .await;
                            if let Some(ref cs) = console_state {
                                cs.upsert_local_process(payload).await;
                            }

                            let event_tx = runtime_event_tx.clone();
                            let event_name = loaded_name.clone();
                            let event_port = handle.port;
                            tokio::spawn(async move {
                                let _ = death_rx.await;
                                let _ = event_tx.send(RuntimeEvent::Exited {
                                    model: event_name,
                                    port: event_port,
                                });
                            });

                            let _ = emit_event(OutputEvent::Info {
                                message: format!(
                                    "Runtime-loaded {} model '{}' on :{}",
                                    handle.backend,
                                    loaded_name,
                                    handle.port
                                ),
                                context: None,
                            });
                            runtime_models.insert(loaded_name.clone(), handle);
                            Ok(loaded_name)
                        }
                        .await;
                        if let Err(err) = &result {
                            let _ = err;
                            if let Some(name) = assigned_runtime_model.as_deref() {
                                remove_serving_assignment(&node, name).await;
                            }
                        }
                        let _ = resp.send(result);
                    }
                    api::RuntimeControlRequest::Unload { model, resp } => {
                        let result = if let Some(handle) = runtime_models.remove(&model) {
                            let port = handle.port;
                            remove_runtime_local_target(&target_tx, &model, port);
                            withdraw_advertised_model(&node, &model).await;
                            upsert_dashboard_process(
                                &dashboard_processes,
                                runtime_process_payload_with_status(&model, &handle, "shutting down"),
                            )
                            .await;
                            if let Some(ref cs) = console_state {
                                cs.upsert_local_process(runtime_process_payload_with_status(
                                    &model,
                                    &handle,
                                    "shutting down",
                                ))
                                .await;
                            }
                            tokio::time::sleep(std::time::Duration::from_millis(300)).await;
                            handle.shutdown().await;
                            remove_serving_assignment(&node, &model).await;
                            remove_dashboard_process(&dashboard_processes, &model).await;
                            if let Some(ref cs) = console_state {
                                cs.remove_local_process(&model).await;
                            }
                            let _ = emit_event(OutputEvent::Info {
                                message: format!("Unloaded local model '{}' from :{}", model, port),
                                context: None,
                            });
                            Ok(())
                        } else if let Some(controller) = managed_models.remove(&model) {
                            let _ = controller.stop_tx.send(true);
                            let _ = controller.task.await;
                            withdraw_advertised_model(&node, &model).await;
                            remove_serving_assignment(&node, &model).await;
                            remove_dashboard_process(&dashboard_processes, &model).await;
                            if let Some(ref cs) = console_state {
                                cs.remove_local_process(&model).await;
                            }
                            let _ = emit_event(OutputEvent::Info {
                                message: format!("Unloaded managed model '{}'", model),
                                context: None,
                            });
                            Ok(())
                        } else {
                            Err(anyhow::anyhow!("model '{model}' is not loaded"))
                        };
                        let _ = resp.send(result);
                    }
                    api::RuntimeControlRequest::Shutdown => {
                        emit_shutdown(None);
                        break;
                    }
                }
            }
            Some(event) = runtime_event_rx.recv() => {
                match event {
                    RuntimeEvent::Exited { model, port } => {
                        let matches = runtime_models
                            .get(&model)
                            .map(|handle| handle.port == port)
                            .unwrap_or(false);
                        if matches {
                            if let Some(handle) = runtime_models.remove(&model) {
                                upsert_dashboard_process(
                                    &dashboard_processes,
                                    runtime_process_payload_with_status(&model, &handle, "exited"),
                                )
                                .await;
                                if let Some(ref cs) = console_state {
                                    cs.upsert_local_process(runtime_process_payload_with_status(
                                        &model, &handle, "exited",
                                    ))
                                    .await;
                                }
                                handle.shutdown().await;
                            }
                            remove_runtime_local_target(&target_tx, &model, port);
                            withdraw_advertised_model(&node, &model).await;
                            remove_serving_assignment(&node, &model).await;
                            let _ = emit_event(OutputEvent::Warning {
                                message: format!("Runtime model '{model}' exited unexpectedly"),
                                context: Some(format!("model={model} port={port}")),
                            });
                        }
                    }
                }
            }
        }
    }

    // Announce clean departure to peers
    node.broadcast_leaving().await;

    // Clean up Nostr listing on shutdown
    if cli.publish {
        if let Ok(keys) = nostr::load_or_create_keys() {
            let relays = nostr_relays(&cli.nostr_relay);
            if let Ok(publisher) = nostr::Publisher::new(keys, &relays).await {
                let _ = publisher.unpublish().await;
                let _ = emit_event(OutputEvent::Info {
                    message: "Removed Nostr listing".to_string(),
                    context: None,
                });
            }
        }
    }
    if let Some(handle) = nostr_publisher {
        handle.abort();
    }

    plugin_manager.shutdown().await;
    api_proxy_handle.abort();
    let _ = api_proxy_handle.await;
    if let Some(handle) = console_server_handle {
        handle.abort();
        let _ = handle.await;
    }

    for (name, handle) in runtime_models.drain() {
        let shutting_down_payload =
            runtime_process_payload_with_status(&name, &handle, "shutting down");
        upsert_dashboard_process(&dashboard_processes, shutting_down_payload.clone()).await;
        if let Some(ref cs) = console_state {
            cs.upsert_local_process(shutting_down_payload).await;
        }
        remove_runtime_local_target(&target_tx, &name, handle.port);
        withdraw_advertised_model(&node, &name).await;
        remove_serving_assignment(&node, &name).await;
        let stopped_payload = runtime_process_payload_with_status(&name, &handle, "stopped");
        handle.shutdown().await;
        upsert_dashboard_process(&dashboard_processes, stopped_payload.clone()).await;
        if let Some(ref cs) = console_state {
            cs.upsert_local_process(stopped_payload).await;
        }
    }

    // Signal each local model loop to stop, then give it a short window to
    // shut down the embedded runtime cleanly.
    for (_, controller) in managed_models.drain() {
        let _ = controller.stop_tx.send(true);
        let mut task = controller.task;
        match tokio::time::timeout(std::time::Duration::from_secs(3), &mut task).await {
            Ok(join_result) => {
                let _ = join_result;
            }
            Err(_) => {
                tracing::warn!("local model task did not stop within 3s during shutdown");
                task.abort();
                let _ = task.await;
            }
        }
    }

    node.set_serving_models(Vec::new()).await;
    node.set_hosted_models(Vec::new()).await;
    if let Some(rt) = runtime {
        let outstanding_refs = std::sync::Arc::strong_count(&rt);
        if outstanding_refs == 1 {
            let dir = rt.dir().to_path_buf();
            drop(rt);
            let _ = std::fs::remove_dir_all(&dir);
        } else {
            tracing::warn!(
                outstanding_refs,
                "skipping runtime directory removal during shutdown because runtime references remain"
            );
        }
    }
    Ok(())
}

/// Used by both --client (pure consumer) and standby GPU nodes (no matching model).
/// If `create_node` is true, creates a new Node (--client path). Otherwise reuses existing.
/// Run as passive node (client or standby GPU).
/// Returns Ok(Some(model_name)) if a standby GPU should promote to serve a model.
/// Returns Ok(None) on clean shutdown.
async fn run_passive(
    cli: &Cli,
    node: mesh::Node,
    is_client: bool,
    plugin_manager: plugin::PluginManager,
) -> Result<Option<String>> {
    let local_port = cli.port;
    let affinity_router = affinity::AffinityRouter::new();
    node.set_display_name(node_display_name(cli, &node)).await;
    let mut passive_publication_state = None;
    let mut passive_publication_rx = None;

    // Nostr publishing (if --publish, for standby GPU nodes advertising capacity)
    if cli.publish && !is_client {
        let pub_node = node.clone();
        match nostr::load_or_create_keys() {
            Ok(nostr_keys) => {
                let relays = nostr_relays(&cli.nostr_relay);
                let pub_name = cli.mesh_name.clone();
                let pub_region = cli.region.clone();
                let pub_max_clients = cli.max_clients;
                let (status_tx, status_rx) = tokio::sync::watch::channel(None);
                passive_publication_rx = Some(status_rx);
                tokio::spawn(async move {
                    nostr::publish_loop(
                        pub_node,
                        nostr_keys,
                        nostr::PublishLoopConfig {
                            relays,
                            name: pub_name,
                            region: pub_region,
                            max_clients: pub_max_clients,
                            interval_secs: 60,
                            status_tx: Some(status_tx),
                        },
                    )
                    .await;
                });
            }
            Err(e) => {
                let _ = emit_event(OutputEvent::Warning {
                    message: format!(
                        "Publishing to Nostr failed: {e}. Standby node is running privately — add --publish after fixing the issue to make discoverable."
                    ),
                    context: cli.mesh_name.as_ref().map(|mesh_name| format!("mesh={mesh_name}")),
                });
                tracing::warn!("Passive Nostr publish failed: {e}");
                passive_publication_state = Some(api::PublicationState::PublishFailed);
            }
        }
    } else if cli.auto && !is_client {
        // Watchdog: take over publishing if the original publisher dies
        let relays = nostr_relays(&cli.nostr_relay);
        let wd_node = node.clone();
        let wd_name = cli.mesh_name.clone();
        let wd_region = cli.region.clone();
        let (status_tx, status_rx) = tokio::sync::watch::channel(None);
        passive_publication_rx = Some(status_rx);
        tokio::spawn(async move {
            nostr::publish_watchdog(wd_node, relays, wd_name, wd_region, 120, Some(status_tx))
                .await;
        });
    }

    // Wait briefly for gossip to propagate
    tokio::time::sleep(std::time::Duration::from_secs(2)).await;

    let served = node.models_being_served().await;
    if !served.is_empty() {
        let _ = emit_event(OutputEvent::Info {
            message: format!("Models available in mesh: {:?}", served),
            context: None,
        });
    }

    let addr = if cli.listen_all {
        "0.0.0.0"
    } else {
        "127.0.0.1"
    };
    let listener = tokio::net::TcpListener::bind(format!("{addr}:{local_port}"))
        .await
        .with_context(|| format!("Failed to bind to port {local_port}"))?;
    if is_client {
        let _ = emit_event(OutputEvent::PassiveMode {
            role: "client".to_string(),
            status: RuntimeStatus::Ready,
            capacity_gb: None,
            models_on_disk: None,
            detail: Some("Client ready".to_string()),
        });
    } else {
        let _ = emit_event(OutputEvent::PassiveMode {
            role: "standby".to_string(),
            status: RuntimeStatus::Ready,
            capacity_gb: Some(node.vram_bytes() as f64 / 1e9),
            models_on_disk: None,
            detail: Some("Standby ready".to_string()),
        });
    }
    let _ = emit_event(OutputEvent::ApiReady {
        url: format!("http://localhost:{local_port}"),
    });
    if cli.headless {
        let _ = emit_event(OutputEvent::Info {
            message: format!("Management API: http://localhost:{}", cli.console),
            context: None,
        });
    } else {
        let _ = emit_event(OutputEvent::WebserverReady {
            url: format!("http://localhost:{}", cli.console),
        });
    }

    // Console
    let (control_tx, mut control_rx) =
        tokio::sync::mpsc::unbounded_channel::<api::RuntimeControlRequest>();
    let cport = cli.console;
    let dashboard_processes = Arc::new(tokio::sync::Mutex::new(Vec::new()));
    let label = if is_client {
        "(client)".to_string()
    } else {
        "(standby)".to_string()
    };
    let runtime_data_collector = node.runtime_data_collector();
    let runtime_data_producer =
        runtime_data_collector.producer(crate::runtime_data::RuntimeDataSource {
            scope: "runtime",
            plugin_data_key: None,
            plugin_endpoint_key: None,
        });
    let console_state = api::MeshApi::new(api::MeshApiConfig {
        node: node.clone(),
        model_name: label,
        api_port: local_port,
        model_size_bytes: 0,
        plugin_manager: plugin_manager.clone(),
        affinity_router: affinity_router.clone(),
        runtime_data_collector,
        runtime_data_producer,
    });
    console_state
        .set_nostr_relays(nostr_relays(&cli.nostr_relay))
        .await;
    console_state.set_nostr_discovery(cli.nostr_discovery).await;
    if is_client {
        console_state.set_client(true).await;
    }
    // Both clients and standby nodes can proxy requests through the mesh
    console_state.update(false, true).await;
    if let Some(state) = passive_publication_state {
        console_state.set_publication_state(state).await;
    }
    if let Some(status_rx) = passive_publication_rx {
        bridge_publication_state(console_state.clone(), status_rx);
    }
    let (_tx, rx) = tokio::sync::watch::channel(election::InferenceTarget::None);
    let la = cli.listen_all;
    let headless = cli.headless;
    let console_state_for_server = console_state.clone();
    let mut console_server_handle = Some(tokio::spawn(async move {
        api::start(cport, console_state_for_server, rx, la, headless).await;
    }));
    crate::cli::output::OutputManager::global().register_dashboard_snapshot_provider(Arc::new(
        RuntimeDashboardSnapshotProvider::new(
            node.clone(),
            dashboard_processes,
            Some(plugin_manager.clone()),
            local_port,
            Some(cport),
            headless,
        ),
    ));
    if crate::cli::output::OutputManager::global()
        .console_session_mode()
        .is_some()
        && std::io::stdin().is_terminal()
    {
        // Spawn input handler for both Dashboard and line-oriented Fallback modes;
        // spawn_handler internally selects the variant.
        interactive::spawn_handler(
            control_tx.clone(),
            console_state.clone(),
            crate::cli::output::OutputManager::global(),
            InitialPromptMode::Immediate,
        );
    }

    // Heartbeat (started in run_auto) handles periodic gossip via random-K.
    // No extra gossip loop needed here.

    // Reactive rebalancing: watch for topology changes and promote if needed.
    // Only for standby GPU nodes (not clients — they never serve).
    let (promote_tx, mut promote_rx) = tokio::sync::mpsc::channel::<String>(1);
    if !is_client {
        let watch_node = node.clone();
        let mut peer_rx = node.peer_change_rx.clone();
        let local_models = models::scan_local_models();
        tokio::spawn(async move {
            // Wait for initial mesh settle
            tokio::time::sleep(std::time::Duration::from_secs(10)).await;
            // Periodic demand check interval (aligned with gossip cycle)
            let mut demand_interval = tokio::time::interval(std::time::Duration::from_secs(60));
            demand_interval.tick().await; // consume first immediate tick
            loop {
                // Wait for EITHER a topology change OR periodic demand check
                tokio::select! {
                    res = peer_rx.changed() => {
                        if res.is_err() { break; }
                        // Debounce — multiple changes can fire in quick succession
                        tokio::time::sleep(std::time::Duration::from_secs(3)).await;
                        // Drain any queued changes
                        while peer_rx.has_changed().unwrap_or(false) {
                            let _ = peer_rx.borrow_and_update();
                        }
                    }
                    _ = demand_interval.tick() => {
                        // Periodic check for demand-based rebalancing
                    }
                }
                // Check if there's an unserved or demand-imbalanced model we can handle
                if let Some(model_name) = check_unserved_model(&watch_node, &local_models).await {
                    let _ = emit_event(OutputEvent::HostElected {
                        model: model_name.clone(),
                        host: watch_node.id().fmt_short().to_string(),
                        role: Some("host".to_string()),
                        capacity_gb: Some(watch_node.vram_bytes() as f64 / 1e9),
                    });
                    let _ = promote_tx.send(model_name).await;
                    break;
                }
            }
        });
    }

    loop {
        tokio::select! {
            accept_result = listener.accept() => {
                let (tcp_stream, addr) = accept_result?;
                tcp_stream.set_nodelay(true)?;
                tracing::info!("Connection from {addr}");
                let node = node.clone();
                let affinity = affinity_router.clone();
                tokio::spawn(crate::network::proxy::handle_mesh_request(
                    node, tcp_stream, true, affinity,
                ));
            }
            Some(model_name) = promote_rx.recv() => {
                return Ok(Some(model_name));
            }
            Some(cmd) = control_rx.recv() => {
                if let api::RuntimeControlRequest::Shutdown = cmd {
                    emit_shutdown(None);
                    plugin_manager.shutdown().await;
                    if let Some(handle) = console_server_handle.take() {
                        handle.abort();
                        let _ = handle.await;
                    }
                    node.broadcast_leaving().await;
                    return Ok(None);
                }
            }
            _ = wait_shutdown_signal() => {
                emit_shutdown(None);
                plugin_manager.shutdown().await;
                if let Some(handle) = console_server_handle.take() {
                    handle.abort();
                    let _ = handle.await;
                }
                node.broadcast_leaving().await;
                return Ok(None);
            }
        }
    }
}

fn detect_bin_dir() -> Result<PathBuf> {
    let exe = std::env::current_exe().context("Failed to determine own binary path")?;
    let dir = exe.parent().context("Binary has no parent directory")?;
    Ok(dir.to_path_buf())
}

/// Update ~/.pi/agent/models.json to include a "mesh" provider.
fn update_pi_models_json(model_id: &str, port: u16) {
    let Some(home) = dirs::home_dir() else { return };
    let models_path = home.join(".pi/agent/models.json");

    let mut root: serde_json::Value = if models_path.exists() {
        match std::fs::read_to_string(&models_path) {
            Ok(s) => serde_json::from_str(&s).unwrap_or_else(|_| serde_json::json!({})),
            Err(_) => serde_json::json!({}),
        }
    } else {
        serde_json::json!({})
    };

    let providers = root.as_object_mut().and_then(|r| {
        r.entry("providers")
            .or_insert_with(|| serde_json::json!({}));
        r.get_mut("providers")?.as_object_mut()
    });
    let Some(providers) = providers else { return };

    let mesh = serde_json::json!({
        "baseUrl": format!("http://localhost:{port}/v1"),
        "api": "openai-completions",
        "apiKey": "mesh",
        "models": [{
            "id": model_id,
            "name": model_id,
            "reasoning": false,
            "input": ["text"],
            "contextWindow": 32768,
            "maxTokens": 8192,
            "compat": {
                "supportsUsageInStreaming": false,
                "maxTokensField": "max_tokens",
                "supportsDeveloperRole": false
            }
        }]
    });

    providers.insert("mesh".to_string(), mesh);

    if let Some(parent) = models_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Ok(json) = serde_json::to_string_pretty(&root) {
        if let Err(e) = std::fs::write(&models_path, json) {
            tracing::warn!("Failed to update {}: {e}", models_path.display());
        }
    }
}

/// Resolve Nostr relay URLs from CLI or defaults.
/// Build the list of models this node is serving for gossip announcement.
/// `resolved_models` comes from explicit `--model` args (may be empty for `--auto`).
/// `model_name` is the actual model we're about to serve (always set).
/// The primary model must always appear in the result.
fn build_serving_list(resolved_models: &[PathBuf], model_name: &str) -> Vec<String> {
    let clean_name = router::strip_split_suffix_owned(model_name);
    let mut all: Vec<String> = resolved_models
        .iter()
        .map(|m| {
            let stem = m
                .file_stem()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();
            // Strip split GGUF suffix: "Model-00001-of-00004" → "Model"
            router::strip_split_suffix_owned(&stem)
        })
        .collect();
    if !all.contains(&clean_name) {
        all.insert(0, clean_name);
    }
    all
}

#[cfg(test)]
fn format_console_ready_line(headless: bool, console_port: u16) -> String {
    if headless {
        format!("  Management API: http://localhost:{console_port}")
    } else {
        format!("  Console: http://localhost:{console_port}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::local::{huggingface_repo_folder_name, huggingface_snapshot_path};
    use crate::plugin::{GpuAssignment, GpuConfig, ModelConfigEntry};
    use crate::system::hardware::GpuFacts;
    use hf_hub::RepoType;
    use serial_test::serial;
    use std::path::Path;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
    use std::time::Duration;

    fn restore_env(key: &str, value: Option<std::ffi::OsString>) {
        if let Some(value) = value {
            std::env::set_var(key, value);
        } else {
            std::env::remove_var(key);
        }
    }

    async fn build_test_mesh_api() -> api::MeshApi {
        let node = mesh::Node::new_for_tests(mesh::NodeRole::Worker)
            .await
            .unwrap();
        let resolved_plugins = plugin::ResolvedPlugins {
            externals: vec![],
            inactive: vec![],
        };
        let (mesh_tx, _mesh_rx) = tokio::sync::mpsc::channel(1);
        let plugin_manager = plugin::PluginManager::start(
            &resolved_plugins,
            plugin::PluginHostMode {
                mesh_visibility: mesh_llm_plugin::MeshVisibility::Private,
            },
            mesh_tx,
        )
        .await
        .unwrap();
        let runtime_data_collector = crate::runtime_data::RuntimeDataCollector::new();
        let runtime_data_producer =
            runtime_data_collector.producer(crate::runtime_data::RuntimeDataSource {
                scope: "runtime",
                plugin_data_key: None,
                plugin_endpoint_key: None,
            });
        api::MeshApi::new(api::MeshApiConfig {
            node,
            model_name: "test-model".to_string(),
            api_port: 3131,
            model_size_bytes: 0,
            plugin_manager,
            affinity_router: affinity::AffinityRouter::default(),
            runtime_data_collector,
            runtime_data_producer,
        })
    }

    #[test]
    fn plugin_dashboard_command_name_trims_base_path() {
        let summary = plugin::PluginSummary {
            name: "browser".to_string(),
            kind: "stdio".to_string(),
            enabled: true,
            status: "running".to_string(),
            pid: Some(4242),
            version: None,
            capabilities: Vec::new(),
            command: Some("/Users/test/dev/mesh/plugins/browser-tools".to_string()),
            args: Vec::new(),
            tools: Vec::new(),
            manifest: None,
            error: None,
        };

        assert_eq!(plugin_dashboard_command_name(&summary), "browser-tools");
    }

    #[tokio::test]
    async fn dashboard_snapshot_provider_reuses_cached_inventory_within_ttl() {
        let node = mesh::Node::new_for_tests(mesh::NodeRole::Worker)
            .await
            .expect("test node should initialize");
        let local_processes = Arc::new(tokio::sync::Mutex::new(Vec::new()));
        let load_count = Arc::new(AtomicUsize::new(0));
        let load_count_for_loader = load_count.clone();
        let provider = RuntimeDashboardSnapshotProvider::with_inventory_loader(
            node,
            local_processes,
            None,
            RuntimeDashboardSnapshotProviderTestOptions {
                api_port: 9337,
                console_port: Some(3131),
                headless: false,
                inventory_snapshot_ttl: Duration::from_secs(60),
                inventory_snapshot_loader: Arc::new(move || {
                    load_count_for_loader.fetch_add(1, AtomicOrdering::SeqCst);
                    crate::models::LocalModelInventorySnapshot::default()
                }),
            },
        );

        let _ = provider.snapshot().await;
        let _ = provider.snapshot().await;

        assert_eq!(load_count.load(AtomicOrdering::SeqCst), 1);
    }

    #[tokio::test]
    async fn dashboard_snapshot_provider_uses_runtime_ctx_and_inventory_file_size() {
        let node = mesh::Node::new_for_tests(mesh::NodeRole::Worker)
            .await
            .expect("test node should initialize");
        let model_name = "Runtime-Model".to_string();
        set_advertised_model_context(&node, &model_name, Some(8192)).await;
        let local_processes = Arc::new(tokio::sync::Mutex::new(vec![api::RuntimeProcessPayload {
            name: model_name.clone(),
            backend: "CUDA0".to_string(),
            status: "ready".to_string(),
            port: 4001,
            pid: 1234,
            slots: 4,
            context_length: Some(8192),
        }]));
        let inventory_model_name = model_name.clone();
        let provider = RuntimeDashboardSnapshotProvider::with_inventory_loader(
            node,
            local_processes,
            None,
            RuntimeDashboardSnapshotProviderTestOptions {
                api_port: 9337,
                console_port: Some(3131),
                headless: false,
                inventory_snapshot_ttl: Duration::from_secs(60),
                inventory_snapshot_loader: Arc::new(move || {
                    let mut snapshot = crate::models::LocalModelInventorySnapshot::default();
                    snapshot
                        .size_by_name
                        .insert(inventory_model_name.clone(), 24_000_000_000);
                    snapshot.metadata_by_name.insert(
                        inventory_model_name.clone(),
                        crate::proto::node::CompactModelMetadata {
                            model_key: inventory_model_name.clone(),
                            context_length: 4096,
                            quantization_type: "Q4_K_M".to_string(),
                            ..Default::default()
                        },
                    );
                    snapshot
                }),
            },
        );

        let snapshot = provider.snapshot().await;
        assert_eq!(snapshot.loaded_model_rows.len(), 1);
        assert_eq!(snapshot.loaded_model_rows[0].slots, Some(4));
        assert_eq!(snapshot.loaded_model_rows[0].ctx_size, Some(8192));
        assert_eq!(snapshot.loaded_model_rows[0].file_size_gb, Some(24.0));
        assert_eq!(
            snapshot.loaded_model_rows[0].quantization.as_deref(),
            Some("Q4_K_M")
        );
    }

    fn synthetic_gpu(
        index: usize,
        stable_id: Option<&str>,
        backend_device: Option<&str>,
    ) -> GpuFacts {
        GpuFacts {
            index,
            display_name: format!("GPU {index}"),
            backend_device: backend_device.map(str::to_string),
            vram_bytes: 24_000_000_000,
            reserved_bytes: None,
            mem_bandwidth_gbps: None,
            compute_tflops_fp32: None,
            compute_tflops_fp16: None,
            unified_memory: false,
            stable_id: stable_id.map(str::to_string),
            pci_bdf: None,
            vendor_uuid: None,
            metal_registry_id: None,
            dxgi_luid: None,
            pnp_instance_id: None,
        }
    }

    #[tokio::test]
    #[serial]
    async fn resolve_model_accepts_short_catalog_name_from_hf_cache() {
        let prev_hub_cache = std::env::var_os("HF_HUB_CACHE");
        let prev_hf_home = std::env::var_os("HF_HOME");
        let prev_xdg = std::env::var_os("XDG_CACHE_HOME");

        let cache_root = std::env::temp_dir().join(format!(
            "mesh-llm-short-name-cache-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&cache_root).unwrap();
        std::env::set_var("HF_HUB_CACHE", &cache_root);
        std::env::remove_var("HF_HOME");
        std::env::remove_var("XDG_CACHE_HOME");

        let repo_id = "bartowski/Llama-3.2-1B-Instruct-GGUF";
        let repo_dir = cache_root.join(huggingface_repo_folder_name(repo_id, RepoType::Model));
        std::fs::create_dir_all(repo_dir.join("refs")).unwrap();
        std::fs::write(repo_dir.join("refs").join("main"), "test-commit").unwrap();
        let model_path = huggingface_snapshot_path(repo_id, RepoType::Model, "test-commit")
            .join("Llama-3.2-1B-Instruct-Q4_K_M.gguf");
        std::fs::create_dir_all(model_path.parent().unwrap()).unwrap();
        std::fs::write(&model_path, b"gguf").unwrap();

        let resolved = resolve_model(Path::new("Llama-3.2-1B-Instruct-Q4_K_M"))
            .await
            .unwrap();
        assert_eq!(resolved, model_path);

        let _ = std::fs::remove_dir_all(&cache_root);
        restore_env("HF_HUB_CACHE", prev_hub_cache);
        restore_env("HF_HOME", prev_hf_home);
        restore_env("XDG_CACHE_HOME", prev_xdg);
    }

    #[tokio::test]
    #[serial]
    async fn resolve_model_accepts_non_catalog_name_from_hf_cache() {
        let prev_hub_cache = std::env::var_os("HF_HUB_CACHE");
        let prev_hf_home = std::env::var_os("HF_HOME");
        let prev_xdg = std::env::var_os("XDG_CACHE_HOME");

        let cache_root = std::env::temp_dir().join(format!(
            "mesh-llm-non-catalog-cache-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&cache_root).unwrap();
        std::env::set_var("HF_HUB_CACHE", &cache_root);
        std::env::remove_var("HF_HOME");
        std::env::remove_var("XDG_CACHE_HOME");

        let repo_id = "someone/Custom-GGUF";
        let repo_dir = cache_root.join(huggingface_repo_folder_name(repo_id, RepoType::Model));
        std::fs::create_dir_all(repo_dir.join("refs")).unwrap();
        std::fs::write(repo_dir.join("refs").join("main"), "test-commit").unwrap();
        let model_path = huggingface_snapshot_path(repo_id, RepoType::Model, "test-commit")
            .join("Custom-Model-Q4_K_M.gguf");
        std::fs::create_dir_all(model_path.parent().unwrap()).unwrap();
        std::fs::write(&model_path, b"gguf").unwrap();

        let resolved_by_stem = resolve_model(Path::new("Custom-Model-Q4_K_M"))
            .await
            .unwrap();
        assert_eq!(resolved_by_stem, model_path);

        let resolved_by_filename = resolve_model(Path::new("Custom-Model-Q4_K_M.gguf"))
            .await
            .unwrap();
        assert_eq!(resolved_by_filename, model_path);

        let _ = std::fs::remove_dir_all(&cache_root);
        restore_env("HF_HUB_CACHE", prev_hub_cache);
        restore_env("HF_HOME", prev_hf_home);
        restore_env("XDG_CACHE_HOME", prev_xdg);
    }

    async fn wait_for_condition<F, Fut>(timeout: Duration, mut check: F)
    where
        F: FnMut() -> Fut,
        Fut: std::future::Future<Output = bool>,
    {
        let deadline = tokio::time::Instant::now() + timeout;
        loop {
            if check().await {
                return;
            }
            assert!(
                tokio::time::Instant::now() < deadline,
                "timed out waiting for test condition"
            );
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }

    #[test]
    fn test_build_serving_list_auto_no_resolved() {
        // --auto: resolved_models is empty, model picked dynamically
        let resolved: Vec<PathBuf> = vec![];
        let result = build_serving_list(&resolved, "Qwen3-30B-A3B-Q4_K_M");
        assert_eq!(result, vec!["Qwen3-30B-A3B-Q4_K_M"]);
    }

    #[test]
    fn test_build_serving_list_explicit_single_model() {
        // --model Qwen3-30B: resolved_models has the model
        let resolved = vec![PathBuf::from(
            "/home/.cache/huggingface/hub/Qwen3-30B-A3B-Q4_K_M.gguf",
        )];
        let result = build_serving_list(&resolved, "Qwen3-30B-A3B-Q4_K_M");
        assert_eq!(result, vec!["Qwen3-30B-A3B-Q4_K_M"]);
        // No duplicate
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_build_serving_list_explicit_multi_model() {
        // --model A --model B: both resolved
        let resolved = vec![
            PathBuf::from("/home/.cache/huggingface/hub/Qwen3-30B-A3B-Q4_K_M.gguf"),
            PathBuf::from("/home/.cache/huggingface/hub/Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf"),
        ];
        let result = build_serving_list(&resolved, "Qwen3-30B-A3B-Q4_K_M");
        assert_eq!(
            result,
            vec!["Qwen3-30B-A3B-Q4_K_M", "Qwen2.5-Coder-7B-Instruct-Q4_K_M"]
        );
    }

    #[test]
    fn test_build_serving_list_split_gguf() {
        // Split GGUF: file is "MiniMax-M2.5-Q4_K_M-00001-of-00004.gguf"
        // Serving list should strip the split suffix
        let resolved = vec![PathBuf::from(
            "/home/.cache/huggingface/hub/MiniMax-M2.5-Q4_K_M-00001-of-00004.gguf",
        )];
        let result = build_serving_list(&resolved, "MiniMax-M2.5-Q4_K_M");
        assert_eq!(result, vec!["MiniMax-M2.5-Q4_K_M"]);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_build_serving_list_split_gguf_model_name_also_has_suffix() {
        // If model_name also has the suffix (from dynamic pick), strip it too
        let resolved = vec![PathBuf::from(
            "/home/.cache/huggingface/hub/MiniMax-M2.5-Q4_K_M-00001-of-00004.gguf",
        )];
        let result = build_serving_list(&resolved, "MiniMax-M2.5-Q4_K_M-00001-of-00004");
        assert_eq!(result, vec!["MiniMax-M2.5-Q4_K_M"]);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_build_startup_model_specs_prefers_cli_models_over_config() {
        let cli = Cli::parse_from([
            "mesh-llm",
            "--model",
            "Qwen3-8B-Q4_K_M",
            "--ctx-size",
            "4096",
        ]);
        let config = plugin::MeshConfig {
            models: vec![plugin::ModelConfigEntry {
                model: "Ignored-Model".into(),
                mmproj: Some("/tmp/ignored-mmproj.gguf".into()),
                ctx_size: Some(8192),
                gpu_id: None,
                parallel: None,
            }],
            ..plugin::MeshConfig::default()
        };

        let specs = build_startup_model_specs(&cli, &config).unwrap();
        assert_eq!(specs.len(), 1);
        assert_eq!(specs[0].model_ref, PathBuf::from("Qwen3-8B-Q4_K_M"));
        assert_eq!(specs[0].mmproj_ref, None);
        assert_eq!(specs[0].ctx_size, Some(4096));
        assert_eq!(specs[0].gpu_id, None);
        assert!(!specs[0].config_owned);
    }

    #[test]
    fn test_build_startup_model_specs_uses_config_models_when_cli_is_empty() {
        let cli = Cli::parse_from(["mesh-llm", "--ctx-size", "4096"]);
        let config = plugin::MeshConfig {
            models: vec![
                plugin::ModelConfigEntry {
                    model: "Qwen3-8B-Q4_K_M".into(),
                    mmproj: None,
                    ctx_size: Some(8192),
                    gpu_id: None,
                    parallel: None,
                },
                plugin::ModelConfigEntry {
                    model: "bartowski/Qwen2.5-VL/model.gguf".into(),
                    mmproj: Some("bartowski/Qwen2.5-VL/mmproj.gguf".into()),
                    ctx_size: Some(16384),
                    gpu_id: None,
                    parallel: None,
                },
            ],
            ..plugin::MeshConfig::default()
        };

        let specs = build_startup_model_specs(&cli, &config).unwrap();
        assert_eq!(specs.len(), 2);
        assert_eq!(specs[0].model_ref, PathBuf::from("Qwen3-8B-Q4_K_M"));
        assert_eq!(specs[0].ctx_size, Some(4096));
        assert_eq!(specs[0].gpu_id, None);
        assert!(specs[0].config_owned);
        assert_eq!(
            specs[1].mmproj_ref,
            Some(PathBuf::from("bartowski/Qwen2.5-VL/mmproj.gguf"))
        );
        assert_eq!(specs[1].ctx_size, Some(4096));
        assert_eq!(specs[1].gpu_id, None);
        assert!(specs[1].config_owned);
    }

    #[test]
    fn test_build_startup_model_specs_ignores_config_models_for_client() {
        let cli = Cli::parse_from(["mesh-llm", "--client"]);
        let config = plugin::MeshConfig {
            models: vec![plugin::ModelConfigEntry {
                model: "Qwen3-8B-Q4_K_M".into(),
                mmproj: None,
                ctx_size: Some(8192),
                gpu_id: None,
                parallel: None,
            }],
            ..plugin::MeshConfig::default()
        };

        let specs = build_startup_model_specs(&cli, &config).unwrap();
        assert!(specs.is_empty());
    }

    #[test]
    fn pinned_gpu_startup_preflight_uses_config_gpu_id() {
        let cli = Cli::parse_from(["mesh-llm"]);
        let config = plugin::MeshConfig {
            gpu: plugin::GpuConfig {
                assignment: plugin::GpuAssignment::Pinned,
                parallel: None,
            },
            models: vec![plugin::ModelConfigEntry {
                model: "Qwen3-8B-Q4_K_M".into(),
                mmproj: None,
                ctx_size: Some(8192),
                gpu_id: Some("pci:0000:65:00.0".into()),
                parallel: None,
            }],
            ..plugin::MeshConfig::default()
        };
        let specs = build_startup_model_specs(&cli, &config).unwrap();
        let mut plans = vec![StartupModelPlan {
            declared_ref: "Qwen3-8B-Q4_K_M".into(),
            resolved_path: PathBuf::from("/tmp/Qwen3-8B-Q4_K_M.gguf"),
            mmproj_path: None,
            ctx_size: Some(8192),
            gpu_id: specs[0].gpu_id.clone(),
            pinned_gpu: None,
            parallel: None,
        }];
        let gpus = vec![
            synthetic_gpu(0, Some("pci:0000:65:00.0"), Some("CUDA0")),
            synthetic_gpu(1, Some("pci:0000:b3:00.0"), Some("CUDA1")),
        ];

        preflight_config_owned_startup_models_with_gpus(&config, &specs, &mut plans, &gpus, None)
            .unwrap();

        assert_eq!(plans[0].gpu_id.as_deref(), Some("pci:0000:65:00.0"));
        assert_eq!(
            plans[0].pinned_gpu,
            Some(StartupPinnedGpuTarget {
                index: 0,
                stable_id: "pci:0000:65:00.0".into(),
                backend_device: "CUDA0".into(),
                vram_bytes: 24_000_000_000,
            })
        );
    }

    #[test]
    fn pinned_gpu_startup_preflight_synthesizes_backend_from_binary_flavor() {
        let mut gpus = vec![
            synthetic_gpu(0, Some("pci:0000:65:00.0"), Some("CUDA0")),
            synthetic_gpu(1, Some("pci:0000:b3:00.0"), Some("ROCm1")),
        ];

        apply_backend_devices_for_flavor(&mut gpus, Some(launch::BinaryFlavor::Vulkan));

        assert_eq!(gpus[0].backend_device.as_deref(), Some("Vulkan0"));
        assert_eq!(gpus[1].backend_device.as_deref(), Some("Vulkan1"));
    }

    #[test]
    fn pinned_gpu_startup_preflight_rejects_synthesized_backend_missing_from_probe() {
        let config = plugin::MeshConfig {
            gpu: plugin::GpuConfig {
                assignment: plugin::GpuAssignment::Pinned,
                parallel: None,
            },
            ..plugin::MeshConfig::default()
        };
        let specs = vec![StartupModelSpec {
            model_ref: PathBuf::from("Qwen3-8B-Q4_K_M"),
            mmproj_ref: None,
            ctx_size: Some(4096),
            gpu_id: Some("pci:0000:b3:00.0".into()),
            config_owned: true,
            parallel: None,
        }];
        let mut plans = vec![StartupModelPlan {
            declared_ref: "Qwen3-8B-Q4_K_M".into(),
            resolved_path: PathBuf::from("/tmp/Qwen3-8B-Q4_K_M.gguf"),
            mmproj_path: None,
            ctx_size: Some(4096),
            gpu_id: Some("pci:0000:b3:00.0".into()),
            pinned_gpu: None,
            parallel: None,
        }];
        let gpus = vec![synthetic_gpu(1, Some("pci:0000:b3:00.0"), Some("Vulkan1"))];
        let backend_probe = launch::BinaryBackendDeviceProbe {
            path: PathBuf::from("/tmp/backend-vulkan"),
            flavor: Some(launch::BinaryFlavor::Vulkan),
            available_devices: vec!["Vulkan0".into(), "CPU".into()],
        };

        let err = preflight_config_owned_startup_models_with_gpus(
            &config,
            &specs,
            &mut plans,
            &gpus,
            Some(&backend_probe),
        )
        .unwrap_err();
        let message = format!("{err:#}");

        assert!(message.contains("failed pinned GPU preflight"));
        assert!(message.contains("requested device Vulkan1 is not supported"));
        assert!(message.contains("Available devices: Vulkan0, CPU"));
    }

    #[test]
    fn pinned_gpu_startup_preflight_canonicalizes_rocm_hip_alias_from_probe() {
        let config = plugin::MeshConfig {
            gpu: plugin::GpuConfig {
                assignment: plugin::GpuAssignment::Pinned,
                parallel: None,
            },
            ..plugin::MeshConfig::default()
        };
        let specs = vec![StartupModelSpec {
            model_ref: PathBuf::from("Qwen3-8B-Q4_K_M"),
            mmproj_ref: None,
            ctx_size: Some(4096),
            gpu_id: Some("pci:0000:b3:00.0".into()),
            config_owned: true,
            parallel: None,
        }];
        let mut plans = vec![StartupModelPlan {
            declared_ref: "Qwen3-8B-Q4_K_M".into(),
            resolved_path: PathBuf::from("/tmp/Qwen3-8B-Q4_K_M.gguf"),
            mmproj_path: None,
            ctx_size: Some(4096),
            gpu_id: Some("pci:0000:b3:00.0".into()),
            pinned_gpu: None,
            parallel: None,
        }];
        let gpus = vec![synthetic_gpu(1, Some("pci:0000:b3:00.0"), Some("ROCm1"))];
        let backend_probe = launch::BinaryBackendDeviceProbe {
            path: PathBuf::from("/tmp/backend-rocm"),
            flavor: Some(launch::BinaryFlavor::Rocm),
            available_devices: vec!["HIP1".into(), "CPU".into()],
        };

        preflight_config_owned_startup_models_with_gpus(
            &config,
            &specs,
            &mut plans,
            &gpus,
            Some(&backend_probe),
        )
        .unwrap();

        assert_eq!(plans[0].pinned_gpu.as_ref().unwrap().backend_device, "HIP1");
    }

    #[test]
    fn pinned_gpu_startup_preflight_keeps_detected_backend_without_resolved_flavor() {
        let mut gpus = vec![synthetic_gpu(0, Some("pci:0000:65:00.0"), Some("CUDA0"))];

        apply_backend_devices_for_flavor(&mut gpus, None);

        assert_eq!(gpus[0].backend_device.as_deref(), Some("CUDA0"));
    }

    #[test]
    fn pinned_gpu_startup_preflight_requests_per_gpu_vram_metrics() {
        let metrics = pinned_startup_preflight_metrics();

        assert_eq!(metrics.len(), 4);
        assert!(metrics.contains(&hardware::Metric::GpuName));
        assert!(metrics.contains(&hardware::Metric::GpuFacts));
        assert!(metrics.contains(&hardware::Metric::VramBytes));
        assert!(metrics.contains(&hardware::Metric::IsSoc));
    }

    #[test]
    fn pinned_gpu_startup_preflight_cli_models_bypass_config_gpu_id() {
        let cli = Cli::parse_from(["mesh-llm", "--model", "Qwen3-8B-Q4_K_M"]);
        let config = plugin::MeshConfig {
            gpu: plugin::GpuConfig {
                assignment: plugin::GpuAssignment::Pinned,
                parallel: None,
            },
            models: vec![plugin::ModelConfigEntry {
                model: "Ignored-Model".into(),
                mmproj: None,
                ctx_size: Some(8192),
                gpu_id: Some("pci:0000:65:00.0".into()),
                parallel: None,
            }],
            ..plugin::MeshConfig::default()
        };
        let specs = build_startup_model_specs(&cli, &config).unwrap();
        let mut plans = vec![StartupModelPlan {
            declared_ref: "Qwen3-8B-Q4_K_M".into(),
            resolved_path: PathBuf::from("/tmp/Qwen3-8B-Q4_K_M.gguf"),
            mmproj_path: None,
            ctx_size: None,
            gpu_id: specs[0].gpu_id.clone(),
            pinned_gpu: None,
            parallel: None,
        }];
        let gpus = vec![synthetic_gpu(0, Some("pci:0000:65:00.0"), Some("CUDA0"))];

        preflight_config_owned_startup_models_with_gpus(&config, &specs, &mut plans, &gpus, None)
            .unwrap();

        assert_eq!(specs[0].gpu_id, None);
        assert!(!specs[0].config_owned);
        assert_eq!(plans[0].gpu_id, None);
        assert_eq!(plans[0].pinned_gpu, None);
    }

    #[test]
    fn pinned_gpu_startup_preflight_missing_gpu_id_fails_closed() {
        let config = plugin::MeshConfig {
            gpu: plugin::GpuConfig {
                assignment: plugin::GpuAssignment::Pinned,
                parallel: None,
            },
            ..plugin::MeshConfig::default()
        };
        let specs = vec![StartupModelSpec {
            model_ref: PathBuf::from("Qwen3-8B-Q4_K_M"),
            mmproj_ref: None,
            ctx_size: None,
            gpu_id: None,
            config_owned: true,
            parallel: None,
        }];
        let mut plans = vec![StartupModelPlan {
            declared_ref: "Qwen3-8B-Q4_K_M".into(),
            resolved_path: PathBuf::from("/tmp/Qwen3-8B-Q4_K_M.gguf"),
            mmproj_path: None,
            ctx_size: None,
            gpu_id: None,
            pinned_gpu: None,
            parallel: None,
        }];
        let gpus = vec![synthetic_gpu(0, Some("pci:0000:65:00.0"), Some("CUDA0"))];

        let err = preflight_config_owned_startup_models_with_gpus(
            &config, &specs, &mut plans, &gpus, None,
        )
        .unwrap_err();
        let message = format!("{err:#}");

        assert!(message.contains("failed pinned GPU preflight"));
        assert!(message.contains("missing configured gpu_id"));
    }

    #[test]
    fn pinned_gpu_startup_preflight_stores_resolved_pinned_target_in_plan() {
        let config = plugin::MeshConfig {
            gpu: plugin::GpuConfig {
                assignment: plugin::GpuAssignment::Pinned,
                parallel: None,
            },
            ..plugin::MeshConfig::default()
        };
        let specs = vec![StartupModelSpec {
            model_ref: PathBuf::from("Qwen3-8B-Q4_K_M"),
            mmproj_ref: None,
            ctx_size: Some(4096),
            gpu_id: Some("uuid:GPU-123".into()),
            config_owned: true,
            parallel: None,
        }];
        let mut plans = vec![StartupModelPlan {
            declared_ref: "Qwen3-8B-Q4_K_M".into(),
            resolved_path: PathBuf::from("/tmp/Qwen3-8B-Q4_K_M.gguf"),
            mmproj_path: None,
            ctx_size: Some(4096),
            gpu_id: Some("uuid:GPU-123".into()),
            pinned_gpu: None,
            parallel: None,
        }];
        let gpus = vec![synthetic_gpu(3, Some("uuid:GPU-123"), Some("CUDA3"))];

        preflight_config_owned_startup_models_with_gpus(&config, &specs, &mut plans, &gpus, None)
            .unwrap();

        let pinned_gpu = plans[0].pinned_gpu.as_ref().unwrap();
        assert_eq!(pinned_gpu.index, 3);
        assert_eq!(pinned_gpu.stable_id, "uuid:GPU-123");
        assert_eq!(pinned_gpu.backend_device, "CUDA3");
        assert_eq!(pinned_gpu.vram_bytes, 24_000_000_000);
    }

    #[test]
    fn pinned_gpu_startup_preflight_rejects_resolved_gpu_without_backend_device() {
        let config = plugin::MeshConfig {
            gpu: plugin::GpuConfig {
                assignment: plugin::GpuAssignment::Pinned,
                parallel: None,
            },
            ..plugin::MeshConfig::default()
        };
        let specs = vec![StartupModelSpec {
            model_ref: PathBuf::from("Qwen3-8B-Q4_K_M"),
            mmproj_ref: None,
            ctx_size: Some(4096),
            gpu_id: Some("uuid:GPU-123".into()),
            config_owned: true,
            parallel: None,
        }];
        let mut plans = vec![StartupModelPlan {
            declared_ref: "Qwen3-8B-Q4_K_M".into(),
            resolved_path: PathBuf::from("/tmp/Qwen3-8B-Q4_K_M.gguf"),
            mmproj_path: None,
            ctx_size: Some(4096),
            gpu_id: Some("uuid:GPU-123".into()),
            pinned_gpu: None,
            parallel: None,
        }];
        let gpus = vec![synthetic_gpu(3, Some("uuid:GPU-123"), None)];

        let err = preflight_config_owned_startup_models_with_gpus(
            &config, &specs, &mut plans, &gpus, None,
        )
        .unwrap_err();
        let message = format!("{err:#}");

        assert!(message.contains("failed pinned GPU preflight"));
        assert!(message.contains("without a backend_device"));
    }

    #[test]
    fn pinned_gpu_startup_preflight_unresolvable_gpu_id_fails_closed() {
        let config = plugin::MeshConfig {
            gpu: plugin::GpuConfig {
                assignment: plugin::GpuAssignment::Pinned,
                parallel: None,
            },
            ..plugin::MeshConfig::default()
        };
        let specs = vec![StartupModelSpec {
            model_ref: PathBuf::from("Qwen3-8B-Q4_K_M"),
            mmproj_ref: None,
            ctx_size: None,
            gpu_id: Some("pci:0000:b3:00.0".into()),
            config_owned: true,
            parallel: None,
        }];
        let mut plans = vec![StartupModelPlan {
            declared_ref: "Qwen3-8B-Q4_K_M".into(),
            resolved_path: PathBuf::from("/tmp/Qwen3-8B-Q4_K_M.gguf"),
            mmproj_path: None,
            ctx_size: None,
            gpu_id: Some("pci:0000:b3:00.0".into()),
            pinned_gpu: None,
            parallel: None,
        }];
        let gpus = vec![synthetic_gpu(0, Some("pci:0000:65:00.0"), Some("CUDA0"))];

        let err = preflight_config_owned_startup_models_with_gpus(
            &config, &specs, &mut plans, &gpus, None,
        )
        .unwrap_err();
        let message = format!("{err:#}");

        assert!(message.contains("failed pinned GPU preflight"));
        assert!(message.contains("did not match any available pinnable GPU"));
    }

    #[test]
    fn test_should_show_serve_config_help_for_bare_serve_without_models() {
        let cli = Cli::parse_from(["mesh-llm"]);
        let startup_specs = Vec::new();

        assert!(should_show_serve_config_help(
            Some(RuntimeSurface::Serve),
            &cli,
            &startup_specs
        ));
    }

    #[test]
    fn test_should_not_show_serve_config_help_when_models_are_present() {
        let cli = Cli::parse_from(["mesh-llm"]);
        let startup_specs = vec![StartupModelSpec {
            model_ref: PathBuf::from("Qwen3-8B-Q4_K_M"),
            mmproj_ref: None,
            ctx_size: None,
            gpu_id: None,
            config_owned: false,
            parallel: None,
        }];

        assert!(!should_show_serve_config_help(
            Some(RuntimeSurface::Serve),
            &cli,
            &startup_specs
        ));
    }

    #[test]
    fn test_should_not_show_serve_config_help_for_client_surface() {
        let cli = Cli::parse_from(["mesh-llm", "--client"]);
        let startup_specs = Vec::new();

        assert!(!should_show_serve_config_help(
            Some(RuntimeSurface::Client),
            &cli,
            &startup_specs
        ));
    }

    #[test]
    fn test_should_not_show_serve_config_help_for_auto_serve_without_models() {
        let cli = Cli::parse_from(["mesh-llm", "--auto"]);
        let startup_specs = Vec::new();

        assert!(!should_show_serve_config_help(
            Some(RuntimeSurface::Serve),
            &cli,
            &startup_specs
        ));
    }

    #[test]
    fn test_should_not_show_serve_config_help_for_join_serve_without_models() {
        let cli = Cli::parse_from(["mesh-llm", "--join", "token"]);
        let startup_specs = Vec::new();

        assert!(!should_show_serve_config_help(
            Some(RuntimeSurface::Serve),
            &cli,
            &startup_specs
        ));
    }

    #[test]
    fn initial_pretty_session_mode_allows_dashboard_only_for_serve_surface() {
        assert_eq!(
            initial_console_session_mode_for_surface(
                Some(RuntimeSurface::Serve),
                ConsoleSessionMode::InteractiveDashboard
            ),
            ConsoleSessionMode::InteractiveDashboard
        );

        assert_eq!(
            initial_console_session_mode_for_surface(
                Some(RuntimeSurface::Client),
                ConsoleSessionMode::InteractiveDashboard
            ),
            ConsoleSessionMode::None
        );

        assert_eq!(
            initial_console_session_mode_for_surface(
                None,
                ConsoleSessionMode::InteractiveDashboard
            ),
            ConsoleSessionMode::None
        );
    }

    #[test]
    fn dashboard_endpoint_rows_keep_builtins_grouped_before_plugins() {
        let mut rows = vec![
            DashboardEndpointRow {
                label: "Plugin: zebra".to_string(),
                status: RuntimeStatus::Ready,
                url: "zebra".to_string(),
                port: 0,
                pid: Some(1001),
            },
            DashboardEndpointRow {
                label: "Web console".to_string(),
                status: RuntimeStatus::Ready,
                url: "http://localhost:3131".to_string(),
                port: 3131,
                pid: None,
            },
            DashboardEndpointRow {
                label: "Plugin: alpha".to_string(),
                status: RuntimeStatus::Ready,
                url: "alpha".to_string(),
                port: 0,
                pid: Some(1000),
            },
            DashboardEndpointRow {
                label: "OpenAI-compatible API".to_string(),
                status: RuntimeStatus::Ready,
                url: "http://localhost:9337".to_string(),
                port: 9337,
                pid: None,
            },
        ];

        sort_dashboard_endpoint_rows(&mut rows);

        let labels = rows.into_iter().map(|row| row.label).collect::<Vec<_>>();
        assert_eq!(
            labels,
            vec![
                "OpenAI-compatible API".to_string(),
                "Web console".to_string(),
                "Plugin: alpha".to_string(),
                "Plugin: zebra".to_string(),
            ]
        );
    }

    #[tokio::test]
    async fn test_runtime_load_unload_regossips_across_nodes() {
        let host = mesh::Node::new_for_tests(mesh::NodeRole::Worker)
            .await
            .unwrap();
        let observer = mesh::Node::new_for_tests(mesh::NodeRole::Worker)
            .await
            .unwrap();

        host.set_role(mesh::NodeRole::Host { http_port: 9337 })
            .await;
        host.set_serving_models(vec!["Primary".into()]).await;
        host.set_hosted_models(vec!["Primary".into()]).await;

        observer.sync_from_peer_for_tests(&host).await;

        wait_for_condition(Duration::from_secs(5), || {
            let observer = observer.clone();
            let host_id = host.id();
            async move {
                observer.peers().await.iter().any(|peer| {
                    peer.id == host_id
                        && peer.routes_model("Primary")
                        && !peer.routes_model("Runtime")
                })
            }
        })
        .await;

        add_serving_assignment(&host, "Primary", "Runtime").await;
        advertise_model_ready(&host, "Primary", "Runtime").await;
        observer.sync_from_peer_for_tests(&host).await;

        wait_for_condition(Duration::from_secs(5), || {
            let observer = observer.clone();
            let host_id = host.id();
            async move {
                observer.peers().await.iter().any(|peer| {
                    peer.id == host_id
                        && peer.is_assigned_model("Runtime")
                        && peer.routes_model("Runtime")
                        && peer.routable_models()
                            == vec!["Primary".to_string(), "Runtime".to_string()]
                })
            }
        })
        .await;

        remove_serving_assignment(&host, "Runtime").await;
        withdraw_advertised_model(&host, "Runtime").await;
        observer.sync_from_peer_for_tests(&host).await;

        wait_for_condition(Duration::from_secs(5), || {
            let observer = observer.clone();
            let host_id = host.id();
            async move {
                observer.peers().await.iter().any(|peer| {
                    peer.id == host_id
                        && peer.routes_model("Primary")
                        && !peer.is_assigned_model("Runtime")
                        && !peer.routes_model("Runtime")
                        && peer.routable_models() == vec!["Primary".to_string()]
                })
            }
        })
        .await;
    }

    #[tokio::test]
    async fn test_benchmark_result_bandwidth_still_works() {
        let mem_arc = std::sync::Arc::new(tokio::sync::Mutex::new(None));
        let fp32_arc = std::sync::Arc::new(tokio::sync::Mutex::new(None));
        let fp16_arc = std::sync::Arc::new(tokio::sync::Mutex::new(None));
        let result = benchmark::BenchmarkResult {
            mem_bandwidth_gbps: vec![10.5, 20.0],
            compute_tflops_fp32: None,
            compute_tflops_fp16: None,
        };

        store_benchmark_metrics(
            mem_arc.clone(),
            fp32_arc.clone(),
            fp16_arc.clone(),
            Some(&result),
        )
        .await;

        assert_eq!(*mem_arc.lock().await, Some(vec![10.5, 20.0]));
        assert!(fp32_arc.lock().await.is_none());
        assert!(fp16_arc.lock().await.is_none());
    }

    #[test]
    fn headless_host_logs_management_api_without_console_url() {
        let line = format_console_ready_line(true, 3131);
        assert!(
            line.contains("Management API"),
            "expected 'Management API' in headless output, got: {line}"
        );
        assert!(
            !line.contains("Console:"),
            "headless output must not contain 'Console:', got: {line}"
        );
    }

    #[test]
    fn default_host_mode_still_logs_console_url() {
        let line = format_console_ready_line(false, 3131);
        assert!(
            line.contains("Console:"),
            "expected 'Console:' in default output, got: {line}"
        );
        assert!(
            !line.contains("Management API"),
            "default output must not contain 'Management API', got: {line}"
        );
    }

    #[test]
    fn active_startup_passes_headless_to_management_server() {
        let headless_line = format_console_ready_line(true, 9090);
        let normal_line = format_console_ready_line(false, 9090);
        assert_ne!(
            headless_line, normal_line,
            "headless and non-headless output must differ"
        );
        assert!(headless_line.contains("9090"));
        assert!(normal_line.contains("9090"));
    }

    #[test]
    fn headless_passive_mode_preserves_api_without_ui() {
        let line = format_console_ready_line(true, 3131);
        assert!(
            line.contains("Management API"),
            "passive headless output must contain 'Management API', got: {line}"
        );
        assert!(
            !line.contains("Console:"),
            "passive headless output must not contain 'Console:', got: {line}"
        );
    }

    #[test]
    fn passive_headless_promotion_keeps_ui_disabled() {
        let promoted_line = format_console_ready_line(true, 3131);
        assert!(
            promoted_line.contains("Management API"),
            "promoted headless node must still advertise Management API, got: {promoted_line}"
        );
        assert!(
            !promoted_line.contains("Console:"),
            "promoted headless node must not show Console: URL, got: {promoted_line}"
        );
    }

    #[test]
    fn default_passive_mode_still_serves_ui_when_not_headless() {
        let line = format_console_ready_line(false, 3131);
        assert!(
            line.contains("Console:"),
            "default passive output must contain 'Console:', got: {line}"
        );
        assert!(
            !line.contains("Management API"),
            "default passive output must not contain 'Management API', got: {line}"
        );
    }

    // ---------------------------------------------------------------------------
    // Per-model parallel (slots) resolution tests
    // ---------------------------------------------------------------------------

    /// Scenario 1: No global `gpu.parallel` set; a specific model entry has
    /// `parallel = 1`. The model's override value must be applied correctly.
    #[test]
    fn per_model_parallel_override_applied_when_no_global() {
        let config_models = [ModelConfigEntry {
            model: "my-model".to_string(),
            mmproj: None,
            ctx_size: None,
            gpu_id: None,
            parallel: Some(1),
        }];
        let gpu_config = GpuConfig::default(); // no parallel set

        // Simulate load handler lookup by spec name
        let slots = config_models
            .iter()
            .find(|m| m.model == "my-model")
            .and_then(|m| m.parallel)
            .or(gpu_config.parallel)
            .unwrap_or(4);

        assert_eq!(
            slots, 1,
            "model-specific parallel=1 should win when no global"
        );
    }

    /// Scenario 2: Two models in config — only the second one specifies a
    /// `parallel` value. The slot assignment must land on the correct model.
    #[test]
    fn per_model_parallel_applies_to_correct_model() {
        let config_models = [
            ModelConfigEntry {
                model: "model-a".to_string(),
                mmproj: None,
                ctx_size: None,
                gpu_id: None,
                parallel: None, // no override
            },
            ModelConfigEntry {
                model: "model-b".to_string(),
                mmproj: None,
                ctx_size: None,
                gpu_id: None,
                parallel: Some(3), // only this one has an override
            },
        ];
        let gpu_config = GpuConfig::default();

        // Model A: falls back to default (no model entry match → default 4)
        let slots_a = config_models
            .iter()
            .find(|m| m.model == "model-a")
            .and_then(|m| m.parallel)
            .or(gpu_config.parallel)
            .unwrap_or(4);
        assert_eq!(
            slots_a, 4,
            "model-a should get default 4 when it has no parallel entry"
        );

        // Model B: gets its own explicit value
        let slots_b = config_models
            .iter()
            .find(|m| m.model == "model-b")
            .and_then(|m| m.parallel)
            .or(gpu_config.parallel)
            .unwrap_or(4);
        assert_eq!(slots_b, 3, "model-b should get its own parallel=3 override");
    }

    /// Scenario 3: Two models. First has NO parallel setting, second has
    /// `parallel = 2`, and global `gpu.parallel = 3`. The first model should
    /// fall through to the global (3), while the second uses its own (2).
    #[test]
    fn per_model_parallel_fallback_to_global_for_missing_entry() {
        let config_models = [
            ModelConfigEntry {
                model: "first".to_string(),
                mmproj: None,
                ctx_size: None,
                gpu_id: None,
                parallel: None, // missing — should use global fallback
            },
            ModelConfigEntry {
                model: "second".to_string(),
                mmproj: None,
                ctx_size: None,
                gpu_id: None,
                parallel: Some(2), // explicit override
            },
        ];
        let gpu_config = GpuConfig {
            assignment: GpuAssignment::Auto,
            parallel: Some(3), // global default
        };

        // First model: no per-model value → falls back to gpu.parallel = 3
        let slots_first = config_models
            .iter()
            .find(|m| m.model == "first")
            .and_then(|m| m.parallel)
            .or(gpu_config.parallel)
            .unwrap_or(4);
        assert_eq!(
            slots_first, 3,
            "missing model parallel should fall back to gpu.parallel=3"
        );

        // Second model: its own value wins over global
        let slots_second = config_models
            .iter()
            .find(|m| m.model == "second")
            .and_then(|m| m.parallel)
            .or(gpu_config.parallel)
            .unwrap_or(4);
        assert_eq!(
            slots_second, 2,
            "model-specific parallel=2 should win over global gpu.parallel=3"
        );
    }

    // ---------------------------------------------------------------------------
    // Publication-state matrix (Issue #240)
    // ---------------------------------------------------------------------------

    /// Helper to build a minimal `Cli` for publication-state tests.
    fn make_cli(args: &[&str]) -> crate::cli::Cli {
        crate::cli::Cli::try_parse_from(args).unwrap()
    }

    #[test]
    fn mesh_name_does_not_force_publish() {
        let cli = make_cli(&[
            "mesh-llm",
            "--model",
            "dummy-model",
            "--mesh-name",
            "my-mesh",
        ]);
        assert!(!cli.publish, "mesh_name alone must not set publish");
        assert_eq!(cli.mesh_name.as_deref(), Some("my-mesh"));
    }

    #[test]
    fn explicit_publish_remains_enabled() {
        let cli = make_cli(&["mesh-llm", "--model", "dummy-model", "--publish"]);
        assert!(
            cli.publish,
            "explicit --publish must set publish=true even without mesh_name"
        );
    }

    #[test]
    fn publish_with_mesh_name_is_public_and_named() {
        let cli = make_cli(&[
            "mesh-llm",
            "--model",
            "dummy-model",
            "--publish",
            "--mesh-name",
            "named-public",
        ]);
        assert!(cli.publish, "publish + mesh_name must keep publish=true");
        assert_eq!(
            cli.mesh_name.as_deref(),
            Some("named-public"),
            "mesh_name must be preserved alongside publish"
        );
    }

    #[test]
    fn auto_without_publish_stays_private() {
        let cli = make_cli(&["mesh-llm", "--model", "dummy-model", "--auto"]);
        assert!(!cli.publish, "--auto alone must not imply publish");
        assert!(cli.auto, "--auto flag should still be true");
    }

    /// Task 2: Named private mesh keeps private identity (no implicit publish).
    #[test]
    fn named_private_mesh_keeps_private_identity() {
        // A named mesh without --publish must have publish=false.
        // The is_public gate in runtime startup uses `cli.auto || cli.publish`,
        // so a named-only mesh should NOT trigger public identity handling.
        let cli = make_cli(&[
            "mesh-llm",
            "--model",
            "dummy-model",
            "--mesh-name",
            "private-named",
        ]);
        assert!(!cli.publish);
        assert!(!cli.auto);
        let is_public = cli.auto || cli.publish;
        assert!(
            !is_public,
            "named-only mesh must be treated as private for identity purposes"
        );
    }

    /// Task 3: start_new_mesh helper does not auto-enable publish.
    #[test]
    fn start_new_mesh_does_not_auto_enable_publish() {
        use crate::runtime::discovery::start_new_mesh;
        let mut cli = make_cli(&["mesh-llm", "--model", "dummy-model"]);
        assert!(!cli.publish, "precondition: publish starts false");
        start_new_mesh(&mut cli, &["dummy-model".to_string()], 16.0, false);
        assert!(
            !cli.publish,
            "start_new_mesh must NOT set publish=true when it was not requested"
        );
    }

    /// Task 3: Explicit --publish survives start_new_mesh unchanged.
    #[test]
    fn start_new_mesh_preserves_explicit_publish() {
        use crate::runtime::discovery::start_new_mesh;
        let mut cli = make_cli(&["mesh-llm", "--model", "dummy-model", "--publish"]);
        assert!(cli.publish, "precondition: publish is true");
        start_new_mesh(&mut cli, &["dummy-model".to_string()], 16.0, false);
        assert!(
            cli.publish,
            "explicit --publish must survive start_new_mesh call"
        );
    }

    #[test]
    fn publish_state_updates_map_to_api_states() {
        assert_eq!(
            publication_state_from_update(nostr::PublishStateUpdate::Public),
            api::PublicationState::Public
        );
        assert_eq!(
            publication_state_from_update(nostr::PublishStateUpdate::PublishFailed),
            api::PublicationState::PublishFailed
        );
    }

    #[tokio::test]
    async fn publication_bridge_keeps_private_until_a_real_publish_outcome_arrives() {
        let state = build_test_mesh_api().await;
        let (status_tx, status_rx) = tokio::sync::watch::channel(None);
        bridge_publication_state(state.clone(), status_rx);

        assert_eq!(state.publication_state().await.as_str(), "private");

        status_tx
            .send(Some(nostr::PublishStateUpdate::Public))
            .unwrap();
        wait_for_condition(Duration::from_secs(2), || {
            let state = state.clone();
            async move { state.publication_state().await.as_str() == "public" }
        })
        .await;

        status_tx
            .send(Some(nostr::PublishStateUpdate::PublishFailed))
            .unwrap();
        wait_for_condition(Duration::from_secs(2), || {
            let state = state.clone();
            async move { state.publication_state().await.as_str() == "publish_failed" }
        })
        .await;
    }

    #[test]
    fn test_console_session_mode_serve_uses_interactive_mode() {
        use crate::cli::RuntimeSurface;

        // When explicit_surface is Some(RuntimeSurface::Serve), should preserve current mode
        let result = initial_console_session_mode_for_surface(
            Some(RuntimeSurface::Serve),
            ConsoleSessionMode::InteractiveDashboard,
        );
        assert_eq!(result, ConsoleSessionMode::InteractiveDashboard);
    }

    #[test]
    fn test_console_session_mode_non_serve_uses_none() {
        use crate::cli::RuntimeSurface;

        // When explicit_surface is Some(RuntimeSurface::Client), should use None mode
        let result = initial_console_session_mode_for_surface(
            Some(RuntimeSurface::Client),
            ConsoleSessionMode::InteractiveDashboard,
        );
        assert_eq!(result, ConsoleSessionMode::None);
    }

    #[test]
    fn test_console_session_mode_no_explicit_surface_uses_none() {
        // When explicit_surface is None, should use None mode
        let result = initial_console_session_mode_for_surface(
            None,
            ConsoleSessionMode::InteractiveDashboard,
        );
        assert_eq!(result, ConsoleSessionMode::None);
    }
}
