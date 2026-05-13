use std::{
    net::SocketAddr,
    sync::{Arc, Mutex},
};

use anyhow::{Context, Result};
use openai_frontend::OpenAiBackend;
use skippy_protocol::{StageConfig, StageTopology};
use tokio::{sync::oneshot, task::JoinHandle};

use crate::{
    binary_transport::{serve_binary_stage_with_shutdown, BinaryStageOptions},
    config::validate_config,
    frontend::{serve_embedded_openai_with_shutdown, EmbeddedOpenAiArgs},
    http::{serve_stage_http_with_shutdown, StageHttpOptions},
    runtime_state::{load_runtime, RuntimeSessionStats, RuntimeState},
    telemetry::{lifecycle_attrs, now_unix_nanos, Telemetry, TelemetryLevel, TelemetryStats},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EmbeddedState {
    Starting,
    Ready,
    Stopping,
    Stopped,
    Failed,
}

#[derive(Clone, Debug)]
pub struct EmbeddedRuntimeStatus {
    pub state: EmbeddedState,
    pub run_id: String,
    pub topology_id: String,
    pub model_id: String,
    pub stage_id: String,
    pub stage_index: u32,
    pub layer_start: u32,
    pub layer_end: u32,
    pub runtime_loaded: bool,
    pub started_at_unix_nanos: i64,
    pub stopped_at_unix_nanos: Option<i64>,
    pub last_error: Option<String>,
    pub sessions: RuntimeSessionStats,
    pub telemetry: TelemetryStats,
}

#[derive(Clone, Debug)]
pub struct EmbeddedServerStatus {
    pub name: &'static str,
    pub bind_addr: SocketAddr,
    pub state: EmbeddedState,
    pub started_at_unix_nanos: i64,
    pub stopped_at_unix_nanos: Option<i64>,
    pub last_error: Option<String>,
}

#[derive(Clone)]
pub struct EmbeddedRuntimeOptions {
    pub config: StageConfig,
    pub topology: Option<StageTopology>,
    pub metrics_otlp_grpc: Option<String>,
    pub telemetry_queue_capacity: usize,
    pub telemetry_level: TelemetryLevel,
}

pub struct SkippyRuntimeHandle {
    config: Arc<StageConfig>,
    topology: Option<Arc<StageTopology>>,
    runtime: Arc<Mutex<RuntimeState>>,
    telemetry: Telemetry,
    status: Arc<Mutex<RuntimeHandleState>>,
}

#[derive(Debug)]
struct RuntimeHandleState {
    state: EmbeddedState,
    started_at_unix_nanos: i64,
    stopped_at_unix_nanos: Option<i64>,
    last_error: Option<String>,
}

impl SkippyRuntimeHandle {
    pub fn load(options: EmbeddedRuntimeOptions) -> Result<Self> {
        validate_config(&options.config, options.topology.as_ref())?;
        let telemetry = Telemetry::new(
            options.metrics_otlp_grpc,
            options.telemetry_queue_capacity,
            options.config.clone(),
            options.telemetry_level,
        );
        telemetry.emit(
            "stage.embedded_runtime_load_start",
            lifecycle_attrs(&options.config),
        );
        let runtime = load_runtime(&options.config)?
            .with_context(|| format!("stage {} requires model_path", options.config.stage_id))?;
        telemetry.emit(
            "stage.embedded_runtime_ready",
            lifecycle_attrs(&options.config),
        );
        Ok(Self {
            config: Arc::new(options.config),
            topology: options.topology.map(Arc::new),
            runtime,
            telemetry,
            status: Arc::new(Mutex::new(RuntimeHandleState {
                state: EmbeddedState::Ready,
                started_at_unix_nanos: now_unix_nanos(),
                stopped_at_unix_nanos: None,
                last_error: None,
            })),
        })
    }

    pub fn config(&self) -> &StageConfig {
        &self.config
    }

    pub fn topology(&self) -> Option<&StageTopology> {
        self.topology.as_deref()
    }

    pub fn runtime(&self) -> Arc<Mutex<RuntimeState>> {
        self.runtime.clone()
    }

    pub fn telemetry(&self) -> Telemetry {
        self.telemetry.clone()
    }

    pub fn status(&self) -> EmbeddedRuntimeStatus {
        let handle = self.status.lock().expect("runtime status lock poisoned");
        let sessions = self
            .runtime
            .lock()
            .expect("runtime lock poisoned")
            .session_stats();
        EmbeddedRuntimeStatus {
            state: handle.state,
            run_id: self.config.run_id.clone(),
            topology_id: self.config.topology_id.clone(),
            model_id: self.config.model_id.clone(),
            stage_id: self.config.stage_id.clone(),
            stage_index: self.config.stage_index,
            layer_start: self.config.layer_start,
            layer_end: self.config.layer_end,
            runtime_loaded: matches!(handle.state, EmbeddedState::Ready | EmbeddedState::Stopping),
            started_at_unix_nanos: handle.started_at_unix_nanos,
            stopped_at_unix_nanos: handle.stopped_at_unix_nanos,
            last_error: handle.last_error.clone(),
            sessions,
            telemetry: self.telemetry.stats(),
        }
    }

    pub fn shutdown(&self) {
        let mut status = self.status.lock().expect("runtime status lock poisoned");
        if status.state == EmbeddedState::Stopped {
            return;
        }
        status.state = EmbeddedState::Stopped;
        status.stopped_at_unix_nanos = Some(now_unix_nanos());
        self.telemetry.emit(
            "stage.embedded_runtime_stopped",
            lifecycle_attrs(&self.config),
        );
    }
}

impl Drop for SkippyRuntimeHandle {
    fn drop(&mut self) {
        self.shutdown();
    }
}

pub struct EmbeddedServerHandle {
    status: Arc<Mutex<ServerHandleState>>,
    shutdown: Option<oneshot::Sender<()>>,
    task: Option<JoinHandle<Result<()>>>,
}

#[derive(Debug)]
struct ServerHandleState {
    name: &'static str,
    bind_addr: SocketAddr,
    state: EmbeddedState,
    started_at_unix_nanos: i64,
    stopped_at_unix_nanos: Option<i64>,
    last_error: Option<String>,
}

impl EmbeddedServerHandle {
    pub fn status(&self) -> EmbeddedServerStatus {
        let status = self.status.lock().expect("server status lock poisoned");
        EmbeddedServerStatus {
            name: status.name,
            bind_addr: status.bind_addr,
            state: status.state,
            started_at_unix_nanos: status.started_at_unix_nanos,
            stopped_at_unix_nanos: status.stopped_at_unix_nanos,
            last_error: status.last_error.clone(),
        }
    }

    pub async fn shutdown(mut self) -> Result<()> {
        if let Some(shutdown) = self.shutdown.take() {
            let _ = shutdown.send(());
        }
        let task = self.task.take().expect("server task already taken");
        task.await?
    }

    pub fn abort(mut self) {
        self.shutdown.take();
        if let Some(task) = self.task.take() {
            task.abort();
        }
        let mut status = self.status.lock().expect("server status lock poisoned");
        status.state = EmbeddedState::Stopped;
        status.stopped_at_unix_nanos = Some(now_unix_nanos());
    }
}

impl Drop for EmbeddedServerHandle {
    fn drop(&mut self) {
        if let Some(shutdown) = self.shutdown.take() {
            let _ = shutdown.send(());
        }
    }
}

pub fn start_stage_http(options: StageHttpOptions) -> EmbeddedServerHandle {
    let bind_addr = options.bind_addr;
    spawn_async_server("stage-http", bind_addr, |shutdown| async move {
        serve_stage_http_with_shutdown(options, async move {
            let _ = shutdown.await;
        })
        .await
    })
}

pub fn start_embedded_openai(args: EmbeddedOpenAiArgs) -> EmbeddedServerHandle {
    let bind_addr = args.bind_addr;
    spawn_async_server("openai", bind_addr, |shutdown| async move {
        serve_embedded_openai_with_shutdown(args, async move {
            let _ = shutdown.await;
        })
        .await
    })
}

pub fn start_openai_backend(
    bind_addr: SocketAddr,
    backend: Arc<dyn OpenAiBackend>,
) -> EmbeddedServerHandle {
    spawn_async_server("openai-backend", bind_addr, move |shutdown| async move {
        let listener = tokio::net::TcpListener::bind(bind_addr).await?;
        axum::serve(listener, openai_frontend::router_for(backend))
            .with_graceful_shutdown(async move {
                let _ = shutdown.await;
            })
            .await?;
        Ok(())
    })
}

pub fn start_binary_stage(options: BinaryStageOptions) -> EmbeddedServerHandle {
    let bind_addr = options.bind_addr;
    let status = Arc::new(Mutex::new(ServerHandleState {
        name: "binary-stage",
        bind_addr,
        state: EmbeddedState::Starting,
        started_at_unix_nanos: now_unix_nanos(),
        stopped_at_unix_nanos: None,
        last_error: None,
    }));
    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    let task_status = status.clone();
    let runtime = tokio::runtime::Handle::current();
    let task = tokio::task::spawn_blocking(move || {
        {
            let mut status = task_status.lock().expect("server status lock poisoned");
            status.state = EmbeddedState::Ready;
        }
        let result = runtime.block_on(serve_binary_stage_with_shutdown(options, async move {
            let _ = shutdown_rx.await;
        }));
        finish_server_status(&task_status, &result);
        result
    });
    EmbeddedServerHandle {
        status,
        shutdown: Some(shutdown_tx),
        task: Some(task),
    }
}

fn spawn_async_server<F, Fut>(
    name: &'static str,
    bind_addr: SocketAddr,
    serve: F,
) -> EmbeddedServerHandle
where
    F: FnOnce(oneshot::Receiver<()>) -> Fut + Send + 'static,
    Fut: std::future::Future<Output = Result<()>> + Send + 'static,
{
    let status = Arc::new(Mutex::new(ServerHandleState {
        name,
        bind_addr,
        state: EmbeddedState::Starting,
        started_at_unix_nanos: now_unix_nanos(),
        stopped_at_unix_nanos: None,
        last_error: None,
    }));
    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    let task_status = status.clone();
    let task = tokio::spawn(async move {
        {
            let mut status = task_status.lock().expect("server status lock poisoned");
            status.state = EmbeddedState::Ready;
        }
        let result = serve(shutdown_rx).await;
        finish_server_status(&task_status, &result);
        result
    });
    EmbeddedServerHandle {
        status,
        shutdown: Some(shutdown_tx),
        task: Some(task),
    }
}

fn finish_server_status(status: &Arc<Mutex<ServerHandleState>>, result: &Result<()>) {
    let mut status = status.lock().expect("server status lock poisoned");
    status.stopped_at_unix_nanos = Some(now_unix_nanos());
    match result {
        Ok(()) => {
            status.state = EmbeddedState::Stopped;
        }
        Err(error) => {
            status.state = EmbeddedState::Failed;
            status.last_error = Some(error.to_string());
        }
    }
}
