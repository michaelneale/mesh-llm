use crate::cli::LogFormat;
use ansi_to_tui::IntoText as _;
use anyhow::Error as AnyhowError;
use chrono::{Local, SecondsFormat, Utc};
use crossterm::{
    cursor::{Hide, MoveTo, Show},
    event::{DisableMouseCapture, EnableMouseCapture},
    execute,
    terminal::{Clear, ClearType, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    backend::TestBackend,
    buffer::Buffer,
    layout::{Alignment, Constraint, Direction, Flex, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{
        Block, BorderType, Cell, Clear as RatatuiClear, HighlightSpacing, Padding, Paragraph, Row,
        Scrollbar, ScrollbarOrientation, ScrollbarState, StatefulWidget, Table, TableState, Widget,
    },
    Frame, Terminal,
};
use serde_json::{json, Map, Value};
use std::collections::{BTreeSet, VecDeque};
use std::fmt::Write as FmtWrite;
use std::future::Future;
use std::io::{self, Write};
use std::pin::Pin;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, OnceLock, RwLock,
};
use tokio::time::{self, Duration, Instant, MissedTickBehavior};

// Wave 1 scaffold: this API exists before runtime wiring lands in later tasks.

#[allow(dead_code)]
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RuntimeStatus {
    Starting,
    Ready,
    ShuttingDown,
    Stopped,
    Exited,
    Warning,
    Error,
}

impl RuntimeStatus {
    fn as_str(&self) -> &'static str {
        match self {
            RuntimeStatus::Starting => "starting",
            RuntimeStatus::Ready => "ready",
            RuntimeStatus::ShuttingDown => "shutting down",
            RuntimeStatus::Stopped => "stopped",
            RuntimeStatus::Exited => "exited",
            RuntimeStatus::Warning => "warning",
            RuntimeStatus::Error => "error",
        }
    }
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ConsoleSessionMode {
    InteractiveDashboard,
    Fallback,
}

#[allow(dead_code)]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DashboardProcessRow {
    pub name: String,
    pub backend: String,
    pub status: RuntimeStatus,
    pub port: u16,
    pub pid: u32,
}

#[allow(dead_code)]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DashboardEndpointRow {
    pub label: String,
    pub status: RuntimeStatus,
    pub url: String,
    pub port: u16,
    pub pid: Option<u32>,
}

#[allow(dead_code)]
#[derive(Clone, Debug, PartialEq)]
pub struct DashboardModelRow {
    pub name: String,
    pub role: Option<String>,
    pub status: RuntimeStatus,
    pub port: Option<u16>,
    pub device: Option<String>,
    pub slots: Option<usize>,
    pub quantization: Option<String>,
    pub ctx_size: Option<u32>,
    pub file_size_gb: Option<f64>,
}

#[allow(dead_code)]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DashboardAcceptedRequestBucket {
    pub second_offset: u32,
    pub accepted_count: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ModelProgressStatus {
    Ensuring,
    Downloading,
    Ready,
}

impl ModelProgressStatus {
    fn as_str(&self) -> &'static str {
        match self {
            ModelProgressStatus::Ensuring => "ensuring",
            ModelProgressStatus::Downloading => "downloading",
            ModelProgressStatus::Ready => "ready",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ModelProgressState {
    label: String,
    file: Option<String>,
    downloaded_bytes: Option<u64>,
    total_bytes: Option<u64>,
    status: ModelProgressStatus,
}

#[derive(Clone, Debug, PartialEq)]
struct StartupProgressState {
    completed_steps: usize,
    total_steps: usize,
    detail: String,
}

#[derive(Clone, Debug, PartialEq)]
struct LoadingProgressState {
    ratio: f64,
    detail: String,
}

#[allow(dead_code)]
#[derive(Clone, Debug, PartialEq)]
pub struct DashboardSnapshot {
    pub llama_process_rows: Vec<DashboardProcessRow>,
    pub webserver_rows: Vec<DashboardEndpointRow>,
    pub loaded_model_rows: Vec<DashboardModelRow>,
    pub current_inflight_requests: u64,
    pub accepted_request_buckets: Vec<DashboardAcceptedRequestBucket>,
    pub latency_samples_ms: Vec<u64>,
}

impl Default for DashboardSnapshot {
    fn default() -> Self {
        Self {
            llama_process_rows: Vec::new(),
            webserver_rows: Vec::new(),
            loaded_model_rows: Vec::new(),
            current_inflight_requests: 0,
            accepted_request_buckets: (0..30)
                .map(|second_offset| DashboardAcceptedRequestBucket {
                    second_offset,
                    accepted_count: 0,
                })
                .collect(),
            latency_samples_ms: Vec::new(),
        }
    }
}

#[allow(dead_code)]
pub type DashboardSnapshotFuture<'a> = Pin<Box<dyn Future<Output = DashboardSnapshot> + Send + 'a>>;

#[allow(dead_code)]
pub trait DashboardSnapshotProvider: Send + Sync {
    fn snapshot(&self) -> DashboardSnapshotFuture<'_>;
}

const DEFAULT_PRETTY_DASHBOARD_EVENT_HISTORY_LIMIT: usize = 1000;
const PRETTY_DASHBOARD_REQUEST_WINDOW_BUCKETS: usize = 30;
const PRETTY_DASHBOARD_REQUEST_MAX_WINDOW_SECS: u32 = 24 * 60 * 60;
const PRETTY_DASHBOARD_PANEL_COUNT: usize = 6;
const PRETTY_TUI_REDRAW_INTERVAL: Duration = Duration::from_millis(33);
const PRETTY_TUI_SNAPSHOT_INTERVAL: Duration = Duration::from_millis(250);
const PRETTY_TUI_MODEL_CARD_HEIGHT: usize = 7;
const PRETTY_TUI_LIST_HIGHLIGHT_SYMBOL_WIDTH: u16 = 2;
const PRETTY_TUI_REQUEST_GRAPH_GUIDE_SYMBOL: &str = "·";
const PRETTY_TUI_REQUEST_GRAPH_BASELINE_SYMBOL: &str = "─";
const PRETTY_TUI_STARTUP_PROGRESS_MIN_STEPS: usize = 12;
const PRETTY_TUI_JOIN_TOKEN_PANEL_HEIGHT: u16 = 5;
const PRETTY_TUI_JOIN_TOKEN_HORIZONTAL_PADDING: u16 = 2;
const PRETTY_TUI_JOIN_TOKEN_COPY_BUTTON_LABEL: &str = " Copy ";
const PRETTY_TUI_EVENTS_COLUMN_PERCENT: u16 = 44;
const PRETTY_TUI_REMAINING_COLUMN_WEIGHT: u16 = 1;
const PRETTY_TUI_WEBSERVER_PROCESS_HEADER_LABEL: &str = "PROCESSES";
const PRETTY_TUI_MIN_DASHBOARD_WIDTH: u16 = 60;
const PRETTY_TUI_SPLASH_ANSI: &[u8] = include_bytes!("assets/pretty-tui-splash.ans");

static PRETTY_TUI_SPLASH_TEXT: OnceLock<Option<Text<'static>>> = OnceLock::new();
static PRETTY_TUI_READY_LOGO_TEXT: OnceLock<Option<Text<'static>>> = OnceLock::new();

#[derive(Clone, Copy)]
struct TuiTheme {
    surface: Color,
    surface_raised: Color,
    text: Color,
    muted: Color,
    dim: Color,
    accent: Color,
    accent_soft: Color,
    success: Color,
    warning: Color,
    error: Color,
    selection_bg: Color,
    status_bar: Style,
}

const fn tui_theme() -> TuiTheme {
    TuiTheme {
        surface: Color::Rgb(8, 10, 14),
        surface_raised: Color::Rgb(18, 22, 29),
        text: Color::Rgb(220, 226, 235),
        muted: Color::Rgb(138, 150, 166),
        dim: Color::Rgb(72, 82, 96),
        accent: Color::Rgb(69, 211, 255),
        accent_soft: Color::Rgb(84, 142, 188),
        success: Color::Rgb(95, 214, 130),
        warning: Color::Rgb(232, 190, 84),
        error: Color::Rgb(238, 93, 108),
        selection_bg: Color::Rgb(31, 40, 52),
        status_bar: Style::new()
            .fg(Color::Rgb(220, 226, 235))
            .bg(Color::Rgb(18, 22, 29)),
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TuiEventListRenderer {
    Legacy,
    Scrollbar,
}

const PRETTY_TUI_EVENT_LEVEL_WIDTH: usize = 4;

const _: TuiEventListRenderer = TuiEventListRenderer::Legacy;

impl TuiEventListRenderer {
    const ACTIVE: Self = Self::Scrollbar;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TuiKeyEvent {
    Tab,
    BackTab,
    Backspace,
    Enter,
    Escape,
    Left,
    Right,
    Up,
    Down,
    PageUp,
    PageDown,
    Interrupt,
    Char(char),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TuiEvent {
    Key(TuiKeyEvent),
    Resize { columns: u16, rows: u16 },
    MouseDown { column: u16, row: u16 },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TuiControlFlow {
    Continue,
    Quit,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum OutputLevel {
    Info,
    Warn,
    Error,
}

impl OutputLevel {
    fn as_str(&self) -> &'static str {
        match self {
            OutputLevel::Info => "info",
            OutputLevel::Warn => "warn",
            OutputLevel::Error => "error",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LlamaInstanceKind {
    RpcServer,
    LlamaServer,
}

impl LlamaInstanceKind {
    fn as_str(&self) -> &'static str {
        match self {
            LlamaInstanceKind::RpcServer => "rpc-server",
            LlamaInstanceKind::LlamaServer => "llama-server",
        }
    }

    fn sort_key(&self) -> u8 {
        match self {
            LlamaInstanceKind::RpcServer => 0,
            LlamaInstanceKind::LlamaServer => 1,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MoeSummary {
    pub experts: u32,
    pub top_k: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MoeDistributionSummary {
    pub leader: String,
    pub active_nodes: usize,
    pub fallback_nodes: usize,
    pub shard_index: usize,
    pub shard_count: usize,
    pub ranking_source: String,
    pub ranking_origin: String,
    pub overlap: usize,
    pub shared_experts: usize,
    pub unique_experts: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MoeStatusSummary {
    pub phase: String,
    pub detail: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MoeAnalysisProgressSummary {
    pub mode: String,
    pub spinner: String,
    pub current: usize,
    pub total: Option<usize>,
    pub elapsed_secs: u64,
}

fn format_moe_analysis_progress_line(
    model_name: &str,
    mode: &str,
    spinner: &str,
    current: usize,
    total: Option<usize>,
    elapsed: std::time::Duration,
) -> String {
    let mode_label = format!("MoE {mode}");
    let progress = match total {
        Some(total) if total > 0 => format!(
            "{:>5.1}%  {}/{}",
            (current as f64 / total as f64) * 100.0,
            current,
            total
        ),
        Some(total) => format!("       0/{}", total),
        None => "starting".to_string(),
    };
    let spinner_progress = format!("{spinner} {progress}");
    format!(
        "🧩 [{}] {:<17} {}  {:>3}s",
        model_name,
        mode_label,
        spinner_progress,
        elapsed.as_secs()
    )
}

fn strip_leading_severity_icon(message: &str) -> &str {
    message
        .strip_prefix("⚠️")
        .or_else(|| message.strip_prefix("❌"))
        .map(str::trim_start)
        .unwrap_or(message)
}

fn format_invite_mesh_label(mesh_name: Option<&str>, mesh_id: &str) -> String {
    match mesh_name.map(str::trim).filter(|name| !name.is_empty()) {
        Some(name) => format!("{name} ({mesh_id})"),
        None => mesh_id.to_string(),
    }
}

#[allow(dead_code)]
#[derive(Clone, Debug, PartialEq)]
pub enum OutputEvent {
    Info {
        message: String,
        context: Option<String>,
    },
    Startup {
        version: String,
        message: Option<String>,
    },
    NodeIdentity {
        node_id: String,
        mesh_id: Option<String>,
    },
    InviteToken {
        token: String,
        mesh_id: String,
        mesh_name: Option<String>,
    },
    DiscoveryStarting {
        source: String,
    },
    MeshFound {
        mesh: String,
        peers: usize,
        region: Option<String>,
    },
    DiscoveryJoined {
        mesh: String,
    },
    DiscoveryFailed {
        message: String,
        detail: Option<String>,
    },
    WaitingForPeers {
        detail: Option<String>,
    },
    PassiveMode {
        role: String,
        status: RuntimeStatus,
        capacity_gb: Option<f64>,
        models_on_disk: Option<Vec<String>>,
        detail: Option<String>,
    },
    PeerJoined {
        peer_id: String,
        label: Option<String>,
    },
    PeerLeft {
        peer_id: String,
        reason: Option<String>,
    },
    ModelQueued {
        model: String,
    },
    ModelLoading {
        model: String,
        source: Option<String>,
    },
    ModelLoaded {
        model: String,
        bytes: Option<u64>,
        moe: Option<MoeSummary>,
    },
    MoeDetected {
        model: String,
        moe: MoeSummary,
        fits_locally: Option<bool>,
        capacity_gb: Option<f64>,
        model_gb: Option<f64>,
    },
    MoeDistribution {
        model: String,
        moe: MoeSummary,
        distribution: MoeDistributionSummary,
    },
    MoeStatus {
        model: String,
        status: MoeStatusSummary,
    },
    MoeAnalysisProgress {
        model: String,
        progress: MoeAnalysisProgressSummary,
    },
    HostElected {
        model: String,
        host: String,
        role: Option<String>,
        capacity_gb: Option<f64>,
    },
    RpcServerStarting {
        port: u16,
        device: String,
        log_path: Option<String>,
    },
    RpcReady {
        port: u16,
        device: String,
        log_path: Option<String>,
    },
    LlamaStarting {
        model: Option<String>,
        http_port: u16,
        ctx_size: Option<u32>,
        log_path: Option<String>,
    },
    LlamaReady {
        model: Option<String>,
        port: u16,
        ctx_size: Option<u32>,
        log_path: Option<String>,
    },
    ModelReady {
        model: String,
        internal_port: Option<u16>,
        role: Option<String>,
    },
    MultiModelMode {
        count: usize,
        models: Vec<String>,
    },
    WebserverStarting {
        url: String,
    },
    WebserverReady {
        url: String,
    },
    ApiStarting {
        url: String,
    },
    ApiReady {
        url: String,
    },
    RuntimeReady {
        api_url: String,
        console_url: Option<String>,
        api_port: u16,
        console_port: Option<u16>,
        models_count: Option<usize>,
        pi_command: Option<String>,
        goose_command: Option<String>,
    },
    ModelDownloadProgress {
        label: String,
        file: Option<String>,
        downloaded_bytes: Option<u64>,
        total_bytes: Option<u64>,
        status: ModelProgressStatus,
    },
    RequestRouted {
        model: String,
        target: String,
    },
    Warning {
        message: String,
        context: Option<String>,
    },
    Error {
        message: String,
        context: Option<String>,
    },
    Shutdown {
        reason: Option<String>,
    },
}

impl OutputEvent {
    fn event_name(&self) -> &'static str {
        match self {
            OutputEvent::Info { .. } => "info",
            OutputEvent::Startup { .. } => "startup",
            OutputEvent::NodeIdentity { .. } => "node_identity",
            OutputEvent::InviteToken { .. } => "invite_token",
            OutputEvent::DiscoveryStarting { .. } => "discovery_starting",
            OutputEvent::MeshFound { .. } => "mesh_found",
            OutputEvent::DiscoveryJoined { .. } => "discovery_joined",
            OutputEvent::DiscoveryFailed { .. } => "discovery_failed",
            OutputEvent::WaitingForPeers { .. } => "waiting_for_peers",
            OutputEvent::PassiveMode { .. } => "passive_mode",
            OutputEvent::PeerJoined { .. } => "peer_joined",
            OutputEvent::PeerLeft { .. } => "peer_left",
            OutputEvent::ModelQueued { .. } => "model_queued",
            OutputEvent::ModelLoading { .. } => "model_loading",
            OutputEvent::ModelLoaded { .. } => "model_loaded",
            OutputEvent::MoeDetected { .. } => "moe_detected",
            OutputEvent::MoeDistribution { .. } => "moe_distribution",
            OutputEvent::MoeStatus { .. } => "moe_status",
            OutputEvent::MoeAnalysisProgress { .. } => "moe_analysis_progress",
            OutputEvent::HostElected { .. } => "host_elected",
            OutputEvent::RpcServerStarting { .. } => "rpc_server_starting",
            OutputEvent::RpcReady { .. } => "rpc_ready",
            OutputEvent::LlamaStarting { .. } => "llama_starting",
            OutputEvent::LlamaReady { .. } => "llama_ready",
            OutputEvent::ModelReady { .. } => "model_ready",
            OutputEvent::MultiModelMode { .. } => "multi_model_mode",
            OutputEvent::WebserverStarting { .. } => "webserver_starting",
            OutputEvent::WebserverReady { .. } => "webserver_ready",
            OutputEvent::ApiStarting { .. } => "api_starting",
            OutputEvent::ApiReady { .. } => "api_ready",
            OutputEvent::RuntimeReady { .. } => "ready",
            OutputEvent::ModelDownloadProgress { .. } => "model_download_progress",
            OutputEvent::RequestRouted { .. } => "request_routed",
            OutputEvent::Warning { .. } => "warning",
            OutputEvent::Error { .. } => "error",
            OutputEvent::Shutdown { .. } => "shutdown",
        }
    }

    fn level(&self) -> OutputLevel {
        match self {
            OutputEvent::Warning { .. } => OutputLevel::Warn,
            OutputEvent::Error { .. } => OutputLevel::Error,
            _ => OutputLevel::Info,
        }
    }

    fn message(&self) -> String {
        match self {
            OutputEvent::Info { message, .. } => message.clone(),
            OutputEvent::Startup { message, .. } => message
                .clone()
                .unwrap_or_else(|| "mesh-llm starting".to_string()),
            OutputEvent::NodeIdentity { node_id, mesh_id } => match mesh_id {
                Some(mesh_id) => format!("node {node_id} joined mesh {mesh_id}"),
                None => format!("node {node_id} initialized"),
            },
            OutputEvent::InviteToken {
                mesh_id,
                mesh_name,
                ..
            } => {
                let mesh_label = format_invite_mesh_label(mesh_name.as_deref(), mesh_id);
                format!("invite token ready for mesh {mesh_label}")
            }
            OutputEvent::DiscoveryStarting { source } => format!("discovering mesh via {source}"),
            OutputEvent::MeshFound { mesh, peers, .. } => {
                format!("discovered mesh {mesh} ({peers} peer(s))")
            }
            OutputEvent::DiscoveryJoined { mesh } => format!("joined mesh {mesh}"),
            OutputEvent::DiscoveryFailed { message, detail } => match detail {
                Some(detail) => format!("{message}: {detail}"),
                None => message.clone(),
            },
            OutputEvent::WaitingForPeers { detail } => detail
                .clone()
                .unwrap_or_else(|| "waiting for peers".to_string()),
            OutputEvent::PassiveMode {
                role,
                status,
                capacity_gb,
                models_on_disk,
                detail,
            } => {
                let mut line = detail
                    .clone()
                    .unwrap_or_else(|| format!("{role} {}", status.as_str()));
                if let Some(capacity_gb) = capacity_gb {
                    line.push_str(&format!(" ({capacity_gb:.1}GB capacity)"));
                }
                if let Some(models_on_disk) = models_on_disk {
                    if !models_on_disk.is_empty() {
                        line.push_str(&format!(" models={}", models_on_disk.join(", ")));
                    }
                }
                line
            }
            OutputEvent::PeerJoined { peer_id, .. } => format!("peer {peer_id} joined"),
            OutputEvent::PeerLeft { peer_id, .. } => format!("peer {peer_id} left"),
            OutputEvent::ModelQueued { model } => format!("queued model {model}"),
            OutputEvent::ModelLoading { model, .. } => format!("loading model {model}"),
            OutputEvent::ModelLoaded { model, .. } => format!("loaded model {model}"),
            OutputEvent::MoeDetected {
                model,
                moe,
                fits_locally,
                capacity_gb,
                model_gb,
            } => {
                let mut line = format!("MoE model detected for {model}: {} experts, top-{}", moe.experts, moe.top_k);
                match fits_locally {
                    Some(true) => {
                        if let (Some(capacity_gb), Some(model_gb)) = (capacity_gb, model_gb) {
                            line.push_str(&format!(
                                " — fits locally ({capacity_gb:.1}GB capacity for {model_gb:.1}GB model)"
                            ));
                        } else {
                            line.push_str(" — fits locally");
                        }
                    }
                    Some(false) => {
                        if let (Some(capacity_gb), Some(model_gb)) = (capacity_gb, model_gb) {
                            line.push_str(&format!(
                                " — waiting for peers ({capacity_gb:.1}GB capacity for {model_gb:.1}GB model)"
                            ));
                        } else {
                            line.push_str(" — waiting for peers");
                        }
                    }
                    None => {}
                }
                line
            }
            OutputEvent::MoeDistribution {
                model,
                moe,
                distribution,
            } => format!(
                "MoE split mode ({model}: {} experts, top-{}) — leader={} active={} fallback={} shard {}/{} ranking={} origin={} overlap={} shared={} unique={}",
                moe.experts,
                moe.top_k,
                distribution.leader,
                distribution.active_nodes,
                distribution.fallback_nodes,
                distribution.shard_index,
                distribution.shard_count,
                distribution.ranking_source,
                distribution.ranking_origin,
                distribution.overlap,
                distribution.shared_experts,
                distribution.unique_experts
            ),
            OutputEvent::MoeStatus { model, status } => {
                format!("{}: {} [{}]", status.phase, status.detail, model)
            }
            OutputEvent::MoeAnalysisProgress { model, progress } => {
                format_moe_analysis_progress_line(
                    model,
                    &progress.mode,
                    &progress.spinner,
                    progress.current,
                    progress.total,
                    std::time::Duration::from_secs(progress.elapsed_secs),
                )
            }
            OutputEvent::HostElected {
                model, host, role, ..
            } => match role {
                Some(role) => format!("{model} elected {host} as {role}"),
                None => format!("{model} elected {host} as host"),
            },
            OutputEvent::RpcServerStarting { port, log_path, .. } => {
                let msg = format!("rpc-server starting on port {port}");
                if let Some(path) = log_path {
                    format!("{msg}\n  ↳ log={path}")
                } else {
                    msg
                }
            }
            OutputEvent::RpcReady { port, log_path, .. } => {
                let msg = format!("rpc-server ready on port {port}");
                if let Some(path) = log_path {
                    format!("{msg}\n  ↳ log={path}")
                } else {
                    msg
                }
            }
            OutputEvent::LlamaStarting { http_port, log_path, .. } => {
                let msg = format!("llama-server starting on port {http_port}");
                if let Some(path) = log_path {
                    format!("{msg}\n  ↳ log={path}")
                } else {
                    msg
                }
            }
            OutputEvent::LlamaReady { port, log_path, .. } => {
                let msg = format!("llama-server ready on port {port}");
                if let Some(path) = log_path {
                    format!("{msg}\n  ↳ log={path}")
                } else {
                    msg
                }
            }
            OutputEvent::ModelReady {
                model,
                internal_port,
                ..
            } => match internal_port {
                Some(port) => format!("model {model} ready on port {port}"),
                None => format!("model {model} ready"),
            },
            OutputEvent::WebserverStarting { url } => format!("web console starting at {url}"),
            OutputEvent::WebserverReady { url } => format!("web console ready at {url}"),
            OutputEvent::ApiStarting { url } => format!("api starting at {url}"),
            OutputEvent::ApiReady { url } => format!("api ready at {url}"),
            OutputEvent::RuntimeReady { .. } => "mesh-llm runtime ready".to_string(),
            OutputEvent::ModelDownloadProgress {
                label,
                file,
                downloaded_bytes,
                total_bytes,
                status,
            } => format_model_download_progress_message(
                label,
                file.as_deref(),
                *downloaded_bytes,
                *total_bytes,
                status,
            ),
            OutputEvent::MultiModelMode { count, models } => {
                if models.is_empty() {
                    format!("🔀 Multi-model mode: {count} model(s)")
                } else {
                    format!(
                        "🔀 Multi-model mode: {count} model(s) — {}",
                        models.join(", ")
                    )
                }
            }
            OutputEvent::RequestRouted { model, target } => {
                format!("routed request for {model} to {target}")
            }
            OutputEvent::Warning { message, .. } => message.clone(),
            OutputEvent::Error { message, .. } => message.clone(),
            OutputEvent::Shutdown { reason } => reason
                .clone()
                .unwrap_or_else(|| "mesh-llm shutting down".to_string()),
        }
    }

    fn summary_line(&self) -> String {
        match self {
            OutputEvent::Info { message, context } => match context {
                Some(context) => format!("{context}: {message}"),
                None => message.clone(),
            },
            OutputEvent::DiscoveryStarting { source } => {
                format!("🔍 discovering mesh via {source}")
            }
            OutputEvent::MeshFound {
                mesh,
                peers,
                region,
            } => match region {
                Some(region) => {
                    format!("📡 discovered mesh {mesh} ({peers} peer(s)) region={region}")
                }
                None => format!("📡 discovered mesh {mesh} ({peers} peer(s))"),
            },
            OutputEvent::DiscoveryJoined { mesh } => format!("✅ joined mesh {mesh}"),
            OutputEvent::DiscoveryFailed { message, detail } => match detail {
                Some(detail) => format!("⚠️ {message}: {detail}"),
                None => format!("⚠️ {message}"),
            },
            OutputEvent::InviteToken {
                token,
                mesh_id,
                mesh_name,
            } => {
                let mesh_label = format_invite_mesh_label(mesh_name.as_deref(), mesh_id);
                format!("📡 Invite created for mesh {mesh_label}: {token}")
            }
            OutputEvent::WaitingForPeers { detail } => detail
                .clone()
                .map(|detail| format!("⏳ {detail}"))
                .unwrap_or_else(|| "⏳ Waiting for peers...".to_string()),
            OutputEvent::PassiveMode {
                role,
                status,
                capacity_gb,
                models_on_disk,
                detail,
            } => {
                let prefix = if role == "client" { "📡" } else { "💤" };
                let mut line = match status {
                    RuntimeStatus::Ready => format!("{prefix} {role} ready"),
                    _ => format!(
                        "{prefix} {}",
                        detail.clone().unwrap_or_else(|| format!("{role} active"))
                    ),
                };
                if let Some(capacity_gb) = capacity_gb {
                    line.push_str(&format!(" ({capacity_gb:.1}GB capacity)"));
                }
                if let Some(models_on_disk) = models_on_disk {
                    if !models_on_disk.is_empty() {
                        line.push_str(&format!(" models={}", models_on_disk.join(", ")));
                    }
                }
                line
            }
            OutputEvent::MoeDetected {
                model,
                moe,
                fits_locally,
                capacity_gb,
                model_gb,
            } => {
                let mut line = format!("🧩 [{model}] MoE model detected: {} experts, top-{}", moe.experts, moe.top_k);
                match fits_locally {
                    Some(true) => {
                        if let (Some(capacity_gb), Some(model_gb)) = (capacity_gb, model_gb) {
                            line.push_str(&format!(
                                " — fits locally ({capacity_gb:.1}GB capacity for {model_gb:.1}GB model)"
                            ));
                        } else {
                            line.push_str(" — fits locally");
                        }
                    }
                    Some(false) => {
                        if let (Some(capacity_gb), Some(model_gb)) = (capacity_gb, model_gb) {
                            line.push_str(&format!(
                                " — waiting for peers ({capacity_gb:.1}GB capacity for {model_gb:.1}GB model)"
                            ));
                        } else {
                            line.push_str(" — waiting for peers");
                        }
                    }
                    None => {}
                }
                line
            }
            OutputEvent::MoeDistribution {
                model,
                moe,
                distribution,
            } => format!(
                "🧩 [{model}] MoE split mode — leader={} active={} fallback={} I am shard {}/{} (ranking={} origin={}, overlap={}); My experts: {} ({} shared + {} unique)",
                distribution.leader,
                distribution.active_nodes,
                distribution.fallback_nodes,
                distribution.shard_index,
                distribution.shard_count,
                distribution.ranking_source,
                distribution.ranking_origin,
                distribution.overlap,
                moe.experts,
                distribution.shared_experts,
                distribution.unique_experts
            ),
            OutputEvent::MoeStatus { model, status } => {
                format!("🧩 [{model}] {}: {}", status.phase, status.detail)
            }
            OutputEvent::MoeAnalysisProgress { model, progress } => format_moe_analysis_progress_line(
                model,
                &progress.mode,
                &progress.spinner,
                progress.current,
                progress.total,
                std::time::Duration::from_secs(progress.elapsed_secs),
            ),
            OutputEvent::HostElected {
                model,
                host,
                role,
                capacity_gb,
            } => match (role, capacity_gb) {
                (Some(role), Some(capacity)) => {
                    format!("🗳 {model} elected {host} as {role} ({capacity:.1}GB capacity)")
                }
                (Some(role), None) => format!("🗳 {model} elected {host} as {role}"),
                (None, Some(capacity)) => {
                    format!("🗳 {model} elected {host} ({capacity:.1}GB capacity)")
                }
                (None, None) => format!("🗳 {model} elected {host}"),
            },
            OutputEvent::PeerJoined { peer_id, label } => match label {
                Some(label) => format!("🤝 Peer joined: {label} ({peer_id})"),
                None => format!("🤝 Peer joined: {peer_id}"),
            },
            OutputEvent::PeerLeft { peer_id, reason } => match reason {
                Some(reason) => format!("👋 Peer left: {peer_id} ({reason})"),
                None => format!("👋 Peer left: {peer_id}"),
            },
            OutputEvent::ModelLoaded { model, bytes, moe } => {
                let mut line = format!("📦 Model loaded: {model}");
                if let Some(bytes) = bytes {
                    line.push_str(&format!(
                        " ({})",
                        if *bytes >= 1_000_000_000 {
                            format!("{:.1}GB", *bytes as f64 / 1e9)
                        } else if *bytes >= 1_000_000 {
                            format!("{:.0}MB", *bytes as f64 / 1e6)
                        } else if *bytes >= 1_000 {
                            format!("{:.0}KB", *bytes as f64 / 1e3)
                        } else {
                            format!("{bytes}B")
                        }
                    ));
                }
                if let Some(moe) = moe {
                    line.push_str(&format!(
                        " [MoE experts={} top_k={}]",
                        moe.experts, moe.top_k
                    ));
                }
                line
            }
            OutputEvent::RpcServerStarting { port, device, .. } => {
                format!("🧵 rpc-server starting: port={port} device={device}")
            }
            OutputEvent::LlamaStarting {
                model,
                http_port,
                ctx_size,
                ..
            } => {
                let mut line = format!("🦙 llama-server starting: port={http_port}");
                if let Some(model) = model {
                    line.push_str(&format!(" model={model}"));
                }
                if let Some(ctx_size) = ctx_size {
                    line.push_str(&format!(" ctx={ctx_size}"));
                }
                line
            }
            OutputEvent::LlamaReady { model, port, .. } => match model {
                Some(model) => format!("✅ {model} ready on internal port {port}"),
                None => format!("✅ llama-server ready on port {port}"),
            },
            OutputEvent::RuntimeReady { models_count, .. } => match models_count {
                Some(count) => format!("✅ Mesh runtime ready ({count} model(s))"),
                None => "✅ Mesh runtime ready".to_string(),
            },
            OutputEvent::ModelDownloadProgress {
                label,
                file,
                downloaded_bytes,
                total_bytes,
                status,
            } => format_model_download_progress_message(
                label,
                file.as_deref(),
                *downloaded_bytes,
                *total_bytes,
                status,
            ),
            OutputEvent::Error { context, message } => match context {
                Some(context) => format!("{context}: {}", strip_leading_severity_icon(message)),
                None => strip_leading_severity_icon(message).to_string(),
            },
            OutputEvent::Warning { message, context } => match context {
                Some(context) => format!("{context}: {}", strip_leading_severity_icon(message)),
                None => strip_leading_severity_icon(message).to_string(),
            },
            _ => self.message().to_string(),
        }
    }

    fn json_fields(&self) -> Map<String, Value> {
        let value = match self {
            OutputEvent::Info { message, context } => {
                json!({ "message": message, "context": context })
            }
            OutputEvent::Startup { version, .. } => json!({ "version": version }),
            OutputEvent::NodeIdentity { node_id, mesh_id } => {
                json!({ "node_id": node_id, "mesh_id": mesh_id })
            }
            OutputEvent::InviteToken {
                token,
                mesh_id,
                mesh_name,
            } => {
                json!({ "token": token, "mesh_id": mesh_id, "mesh_name": mesh_name })
            }
            OutputEvent::DiscoveryStarting { source } => json!({ "source": source }),
            OutputEvent::MeshFound {
                mesh,
                peers,
                region,
            } => json!({ "mesh": mesh, "peers": peers, "region": region }),
            OutputEvent::DiscoveryJoined { mesh } => json!({ "mesh": mesh }),
            OutputEvent::DiscoveryFailed { message, detail } => {
                json!({ "message": message, "detail": detail })
            }
            OutputEvent::WaitingForPeers { detail } => json!({ "detail": detail }),
            OutputEvent::PassiveMode {
                role,
                status,
                capacity_gb,
                models_on_disk,
                detail,
            } => json!({
                "role": role,
                "status": status.as_str(),
                "capacity_gb": capacity_gb,
                "models_on_disk": models_on_disk,
                "detail": detail,
            }),
            OutputEvent::PeerJoined { peer_id, label } => {
                json!({ "peer_id": peer_id, "label": label })
            }
            OutputEvent::PeerLeft { peer_id, reason } => {
                json!({ "peer_id": peer_id, "reason": reason })
            }
            OutputEvent::ModelQueued { model } => json!({ "model": model }),
            OutputEvent::ModelLoading { model, source } => {
                json!({ "model": model, "source": source })
            }
            OutputEvent::ModelLoaded { model, bytes, moe } => json!({
                "model": model,
                "bytes": bytes,
                "moe": moe.as_ref().map(|moe| json!({
                    "experts": moe.experts,
                    "top_k": moe.top_k,
                })),
            }),
            OutputEvent::MoeDetected {
                model,
                moe,
                fits_locally,
                capacity_gb,
                model_gb,
            } => json!({
                "model": model,
                "moe": {
                    "experts": moe.experts,
                    "top_k": moe.top_k,
                },
                "fits_locally": fits_locally,
                "capacity_gb": capacity_gb,
                "model_gb": model_gb,
            }),
            OutputEvent::MoeDistribution {
                model,
                moe,
                distribution,
            } => json!({
                "model": model,
                "moe": {
                    "experts": moe.experts,
                    "top_k": moe.top_k,
                },
                "distribution": {
                    "leader": distribution.leader,
                    "active_nodes": distribution.active_nodes,
                    "fallback_nodes": distribution.fallback_nodes,
                    "shard_index": distribution.shard_index,
                    "shard_count": distribution.shard_count,
                    "ranking_source": distribution.ranking_source,
                    "ranking_origin": distribution.ranking_origin,
                    "overlap": distribution.overlap,
                    "shared_experts": distribution.shared_experts,
                    "unique_experts": distribution.unique_experts,
                }
            }),
            OutputEvent::MoeStatus { model, status } => json!({
                "model": model,
                "status": {
                    "phase": status.phase,
                    "detail": status.detail,
                }
            }),
            OutputEvent::MoeAnalysisProgress { model, progress } => json!({
                "model": model,
                "progress": {
                    "mode": progress.mode,
                    "spinner": progress.spinner,
                    "current": progress.current,
                    "total": progress.total,
                    "elapsed_secs": progress.elapsed_secs,
                }
            }),
            OutputEvent::HostElected {
                model,
                host,
                role,
                capacity_gb,
            } => json!({ "model": model, "host": host, "role": role, "capacity_gb": capacity_gb }),
            OutputEvent::RpcServerStarting {
                port,
                device,
                log_path,
            }
            | OutputEvent::RpcReady {
                port,
                device,
                log_path,
            } => json!({ "port": port, "device": device, "log_path": log_path }),
            OutputEvent::LlamaStarting {
                model,
                http_port,
                ctx_size,
                log_path,
            } => json!({
                "model": model,
                "http_port": http_port,
                "ctx_size": ctx_size,
                "log_path": log_path,
            }),
            OutputEvent::LlamaReady {
                model,
                port,
                ctx_size,
                log_path,
            } => json!({
                "model": model,
                "port": port,
                "ctx_size": ctx_size,
                "log_path": log_path,
            }),
            OutputEvent::ModelReady {
                model,
                internal_port,
                role,
            } => json!({
                "model": model,
                "port": internal_port,
                "internal_port": internal_port,
                "role": role,
            }),
            OutputEvent::MultiModelMode { count, models } => {
                json!({ "count": count, "models": models })
            }
            OutputEvent::WebserverStarting { url }
            | OutputEvent::WebserverReady { url }
            | OutputEvent::ApiStarting { url }
            | OutputEvent::ApiReady { url } => json!({ "url": url }),
            OutputEvent::RuntimeReady {
                api_url,
                console_url,
                api_port,
                console_port,
                models_count,
                pi_command,
                goose_command,
            } => json!({
                "api_url": api_url,
                "console_url": console_url,
                "api_port": api_port,
                "console_port": console_port,
                "models_count": models_count,
                "pi_command": pi_command,
                "goose_command": goose_command,
            }),
            OutputEvent::ModelDownloadProgress {
                label,
                file,
                downloaded_bytes,
                total_bytes,
                status,
            } => json!({
                "label": label,
                "file": file,
                "downloaded_bytes": downloaded_bytes,
                "total_bytes": total_bytes,
                "status": status.as_str(),
            }),
            OutputEvent::RequestRouted { model, target } => {
                json!({ "model": model, "target": target })
            }
            OutputEvent::Warning { message, context } => {
                json!({ "warning": message, "context": context })
            }
            OutputEvent::Error { message, context } => {
                json!({
                    "error": message,
                    "context": context,
                    "error_type": classify_error_type(message, context.as_deref()),
                })
            }
            OutputEvent::Shutdown { reason } => json!({ "reason": reason }),
        };

        match value {
            Value::Object(map) => map,
            _ => Map::new(),
        }
    }
}

fn format_model_download_progress_message(
    label: &str,
    file: Option<&str>,
    downloaded_bytes: Option<u64>,
    total_bytes: Option<u64>,
    status: &ModelProgressStatus,
) -> String {
    let target = file.unwrap_or(label);
    match status {
        ModelProgressStatus::Ensuring => format!("ensuring model {target}"),
        ModelProgressStatus::Downloading => match (downloaded_bytes, total_bytes) {
            (Some(downloaded), Some(total)) if total > 0 => format!(
                "downloading model {target} {}/{}",
                format_display_bytes(downloaded),
                format_display_bytes(total)
            ),
            (Some(downloaded), _) if downloaded > 0 => {
                format!(
                    "downloading model {target} {}",
                    format_display_bytes(downloaded)
                )
            }
            _ => format!("downloading model {target}"),
        },
        ModelProgressStatus::Ready => match total_bytes {
            Some(total) if total > 0 => {
                format!("model {target} ready ({})", format_display_bytes(total))
            }
            _ => format!("model {target} ready"),
        },
    }
}

fn format_display_bytes(bytes: u64) -> String {
    if bytes >= 1_000_000_000 {
        format!("{:.1}GB", bytes as f64 / 1e9)
    } else if bytes >= 1_000_000 {
        format!("{:.0}MB", bytes as f64 / 1e6)
    } else if bytes >= 1_000 {
        format!("{:.0}KB", bytes as f64 / 1e3)
    } else {
        format!("{bytes}B")
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LlamaInstanceState {
    pub kind: LlamaInstanceKind,
    pub port: u16,
    pub status: RuntimeStatus,
    pub device: Option<String>,
    pub model: Option<String>,
    pub ctx_size: Option<u32>,
    pub log_path: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RunningModelState {
    pub model: String,
    pub status: RuntimeStatus,
    pub internal_port: Option<u16>,
    pub role: Option<String>,
    pub capacity_gb: Option<f64>,
    pub moe: Option<MoeSummary>,
    pub moe_distribution: Option<MoeDistributionSummary>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PassiveModeState {
    pub role: String,
    pub status: RuntimeStatus,
    pub capacity_gb: Option<f64>,
    pub models_on_disk: Vec<String>,
    pub detail: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MultiModelModeState {
    pub count: usize,
    pub models: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EndpointState {
    pub label: String,
    pub status: RuntimeStatus,
    pub url: String,
    pub details: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MeshEventState {
    pub timestamp: String,
    pub level: OutputLevel,
    pub summary: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DashboardPanel {
    JoinToken,
    Events,
    LlamaCpp,
    Webserver,
    Models,
    Requests,
}

impl DashboardPanel {
    const ALL: [Self; PRETTY_DASHBOARD_PANEL_COUNT] = [
        Self::JoinToken,
        Self::Events,
        Self::LlamaCpp,
        Self::Webserver,
        Self::Models,
        Self::Requests,
    ];

    const fn index(self) -> usize {
        match self {
            Self::JoinToken => 0,
            Self::Events => 1,
            Self::LlamaCpp => 2,
            Self::Webserver => 3,
            Self::Models => 4,
            Self::Requests => 5,
        }
    }

    fn next(self) -> Self {
        Self::ALL[(self.index() + 1) % Self::ALL.len()]
    }

    fn previous(self) -> Self {
        Self::ALL[(self.index() + Self::ALL.len() - 1) % Self::ALL.len()]
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct DashboardPanelViewState {
    scroll_offset: usize,
    selected_row: Option<usize>,
    viewport_rows: usize,
}

impl Default for DashboardPanelViewState {
    fn default() -> Self {
        Self {
            scroll_offset: 0,
            selected_row: None,
            viewport_rows: 1,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct DashboardLayoutWidget {
    rows: usize,
    selectable: bool,
}

impl DashboardLayoutWidget {
    fn new(rows: usize, selectable: bool) -> Self {
        Self {
            rows: rows.max(1),
            selectable,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct DashboardLayoutState {
    widgets: [DashboardLayoutWidget; PRETTY_DASHBOARD_PANEL_COUNT],
}

impl DashboardLayoutState {
    fn new(
        events_rows: usize,
        llama_rows: usize,
        webserver_rows: usize,
        models_rows: usize,
        requests_rows: usize,
    ) -> Self {
        Self {
            widgets: [
                DashboardLayoutWidget::new(1, false),
                DashboardLayoutWidget::new(events_rows, true),
                DashboardLayoutWidget::new(llama_rows, true),
                DashboardLayoutWidget::new(webserver_rows, true),
                DashboardLayoutWidget::new(models_rows, false),
                DashboardLayoutWidget::new(requests_rows, false),
            ],
        }
    }

    fn rows_for(self, panel: DashboardPanel) -> usize {
        self.widgets[panel.index()].rows.max(1)
    }

    fn rows_are_selectable_for(self, panel: DashboardPanel) -> bool {
        self.widgets[panel.index()].selectable
    }
}

impl Default for DashboardLayoutState {
    fn default() -> Self {
        Self::new(1, 1, 1, 1, 1)
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct DashboardEventsFilterState {
    query: String,
    editing: bool,
}

impl DashboardEventsFilterState {
    fn is_active(&self) -> bool {
        !self.query.is_empty()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct DashboardJoinTokenState {
    token: String,
    mesh_id: String,
    mesh_name: Option<String>,
    copy_status: DashboardJoinTokenCopyStatus,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum DashboardJoinTokenCopyStatus {
    Idle,
    Copied,
    Failed(String),
}

impl DashboardJoinTokenState {
    fn new(token: String, mesh_id: String, mesh_name: Option<String>) -> Self {
        Self {
            token,
            mesh_id,
            mesh_name,
            copy_status: DashboardJoinTokenCopyStatus::Idle,
        }
    }

    fn mesh_label(&self) -> String {
        format_invite_mesh_label(self.mesh_name.as_deref(), &self.mesh_id)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct DashboardRequestHistoryState {
    current_inflight_requests: u64,
    accepted_request_buckets: Vec<DashboardAcceptedRequestBucket>,
    latency_samples_ms: Vec<u64>,
    history_limit: usize,
}

impl Default for DashboardRequestHistoryState {
    fn default() -> Self {
        Self {
            current_inflight_requests: 0,
            accepted_request_buckets: (0..PRETTY_DASHBOARD_REQUEST_MAX_WINDOW_SECS)
                .map(|second_offset| DashboardAcceptedRequestBucket {
                    second_offset,
                    accepted_count: 0,
                })
                .collect(),
            latency_samples_ms: Vec::new(),
            history_limit: PRETTY_DASHBOARD_REQUEST_MAX_WINDOW_SECS as usize,
        }
    }
}

impl DashboardRequestHistoryState {
    fn from_snapshot(snapshot: &DashboardSnapshot) -> Self {
        Self {
            current_inflight_requests: snapshot.current_inflight_requests,
            accepted_request_buckets: normalize_request_buckets(&snapshot.accepted_request_buckets),
            latency_samples_ms: snapshot.latency_samples_ms.clone(),
            history_limit: PRETTY_DASHBOARD_REQUEST_MAX_WINDOW_SECS as usize,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
enum DashboardRequestWindow {
    #[default]
    SixtySeconds,
    TenMinutes,
    SixtyMinutes,
    TwelveHours,
    TwentyFourHours,
}

impl DashboardRequestWindow {
    const ALL: [Self; 5] = [
        Self::SixtySeconds,
        Self::TenMinutes,
        Self::SixtyMinutes,
        Self::TwelveHours,
        Self::TwentyFourHours,
    ];

    fn label(self) -> &'static str {
        match self {
            Self::SixtySeconds => "60s",
            Self::TenMinutes => "10m",
            Self::SixtyMinutes => "60m",
            Self::TwelveHours => "12h",
            Self::TwentyFourHours => "24h",
        }
    }

    fn bucket_label(self) -> &'static str {
        match self {
            Self::SixtySeconds => "2s buckets",
            Self::TenMinutes => "20s buckets",
            Self::SixtyMinutes => "2m buckets",
            Self::TwelveHours => "30m buckets",
            Self::TwentyFourHours => "60m buckets",
        }
    }

    fn seconds(self) -> u32 {
        match self {
            Self::SixtySeconds => 60,
            Self::TenMinutes => 10 * 60,
            Self::SixtyMinutes => 60 * 60,
            Self::TwelveHours => 12 * 60 * 60,
            Self::TwentyFourHours => 24 * 60 * 60,
        }
    }

    fn bucket_seconds(self) -> u32 {
        match self {
            Self::TwelveHours => 30 * 60,
            Self::TwentyFourHours => 60 * 60,
            _ => self.seconds() / PRETTY_DASHBOARD_REQUEST_WINDOW_BUCKETS as u32,
        }
    }

    fn bar_width_cap(self) -> Option<u16> {
        match self {
            Self::TwelveHours | Self::TwentyFourHours => Some(1),
            _ => None,
        }
    }

    fn preferred_bar_gap(self) -> u16 {
        match self {
            Self::TwelveHours | Self::TwentyFourHours => 1,
            _ => 0,
        }
    }

    fn previous(self) -> Self {
        let index = Self::ALL
            .iter()
            .position(|window| *window == self)
            .unwrap_or_default();
        Self::ALL[index.saturating_sub(1)]
    }

    fn next(self) -> Self {
        let index = Self::ALL
            .iter()
            .position(|window| *window == self)
            .unwrap_or_default();
        Self::ALL[(index + 1).min(Self::ALL.len() - 1)]
    }
}

#[derive(Clone, Debug, PartialEq)]
enum DashboardAction {
    OutputEvent(OutputEvent),
    SnapshotUpdated(DashboardSnapshot),
    FocusNextPanel,
    FocusPreviousPanel,
    ToggleEventsFollow,
    StartEventsFilterEdit,
    InsertEventsFilterChar(char),
    BackspaceEventsFilter,
    ConfirmEventsFilter,
    CancelEventsFilter,
    SelectPreviousRequestWindow,
    SelectNextRequestWindow,
    SetJoinTokenCopyStatus(DashboardJoinTokenCopyStatus),
    #[cfg(test)]
    SetPanelScroll {
        panel: DashboardPanel,
        scroll_offset: usize,
    },
    #[cfg(test)]
    SetPanelSelection {
        panel: DashboardPanel,
        selected_row: Option<usize>,
    },
    Resize(DashboardLayoutState),
}

#[derive(Clone, Debug, PartialEq)]
pub struct DashboardState {
    session_started_at: Instant,
    version: Option<String>,
    node_id: Option<String>,
    mesh_id: Option<String>,
    runtime_ready: bool,
    peer_ids: BTreeSet<String>,
    llama_instances: Vec<LlamaInstanceState>,
    multi_model_mode: Option<MultiModelModeState>,
    passive_mode: Option<PassiveModeState>,
    running_models: Vec<RunningModelState>,
    webserver: Option<EndpointState>,
    api: Option<EndpointState>,
    mesh_events: VecDeque<MeshEventState>,
    mesh_event_limit: usize,
    panel_focus: DashboardPanel,
    panel_layout: DashboardLayoutState,
    panel_view_states: [DashboardPanelViewState; PRETTY_DASHBOARD_PANEL_COUNT],
    events_follow: bool,
    events_filter: DashboardEventsFilterState,
    llama_process_rows: Vec<DashboardProcessRow>,
    webserver_rows: Vec<DashboardEndpointRow>,
    loaded_model_rows: Vec<DashboardModelRow>,
    request_history: DashboardRequestHistoryState,
    request_window: DashboardRequestWindow,
    join_token: Option<DashboardJoinTokenState>,
    terminal_size: Option<(u16, u16)>,
    model_progress: Option<ModelProgressState>,
    startup_progress: Option<StartupProgressState>,
    startup_milestones: BTreeSet<String>,
    shutdown_in_progress: bool,
}

impl Default for DashboardState {
    fn default() -> Self {
        let panel_layout = DashboardLayoutState::default();
        let mut state = Self {
            session_started_at: Instant::now(),
            version: None,
            node_id: None,
            mesh_id: None,
            runtime_ready: false,
            peer_ids: BTreeSet::new(),
            llama_instances: Vec::new(),
            multi_model_mode: None,
            passive_mode: None,
            running_models: Vec::new(),
            webserver: None,
            api: None,
            mesh_events: VecDeque::new(),
            mesh_event_limit: DEFAULT_PRETTY_DASHBOARD_EVENT_HISTORY_LIMIT,
            panel_focus: DashboardPanel::Events,
            panel_layout,
            panel_view_states: [DashboardPanelViewState::default(); PRETTY_DASHBOARD_PANEL_COUNT],
            events_follow: true,
            events_filter: DashboardEventsFilterState::default(),
            llama_process_rows: Vec::new(),
            webserver_rows: Vec::new(),
            loaded_model_rows: Vec::new(),
            request_history: DashboardRequestHistoryState::default(),
            request_window: DashboardRequestWindow::default(),
            join_token: None,
            terminal_size: None,
            model_progress: None,
            startup_progress: None,
            startup_milestones: BTreeSet::new(),
            shutdown_in_progress: false,
        };
        state.apply_layout(panel_layout);
        state
    }
}

impl DashboardState {
    fn reduce(&mut self, action: DashboardAction) {
        match action {
            DashboardAction::OutputEvent(event) => self.apply_output_event(&event),
            DashboardAction::SnapshotUpdated(snapshot) => self.apply_snapshot(&snapshot),
            DashboardAction::FocusNextPanel => {
                self.panel_focus = self.panel_focus.next();
                if self.panel_focus != DashboardPanel::Events {
                    self.events_filter.editing = false;
                }
            }
            DashboardAction::FocusPreviousPanel => {
                self.panel_focus = self.panel_focus.previous();
                if self.panel_focus != DashboardPanel::Events {
                    self.events_filter.editing = false;
                }
            }
            DashboardAction::ToggleEventsFollow => {
                self.events_follow = !self.events_follow;
                self.sync_events_panel();
            }
            DashboardAction::StartEventsFilterEdit => {
                self.panel_focus = DashboardPanel::Events;
                self.events_filter.editing = true;
                self.sync_events_panel();
            }
            DashboardAction::InsertEventsFilterChar(ch) => {
                self.panel_focus = DashboardPanel::Events;
                self.events_filter.editing = true;
                self.events_filter.query.push(ch);
                self.sync_events_panel();
            }
            DashboardAction::BackspaceEventsFilter => {
                self.panel_focus = DashboardPanel::Events;
                self.events_filter.editing = true;
                self.events_filter.query.pop();
                self.sync_events_panel();
            }
            DashboardAction::ConfirmEventsFilter => {
                self.events_filter.editing = false;
                self.sync_events_panel();
            }
            DashboardAction::CancelEventsFilter => {
                self.panel_focus = DashboardPanel::Events;
                self.events_filter.query.clear();
                self.events_filter.editing = false;
                self.sync_events_panel();
            }
            DashboardAction::SelectPreviousRequestWindow => {
                self.request_window = self.request_window.previous();
            }
            DashboardAction::SelectNextRequestWindow => {
                self.request_window = self.request_window.next();
            }
            DashboardAction::SetJoinTokenCopyStatus(copy_status) => {
                if let Some(join_token) = self.join_token.as_mut() {
                    join_token.copy_status = copy_status;
                }
            }
            #[cfg(test)]
            DashboardAction::SetPanelScroll {
                panel,
                scroll_offset,
            } => {
                self.panel_view_state_mut(panel).scroll_offset = scroll_offset;
                self.clamp_panel_view(panel);
            }
            #[cfg(test)]
            DashboardAction::SetPanelSelection {
                panel,
                selected_row,
            } => {
                self.panel_view_state_mut(panel).selected_row = selected_row;
                self.clamp_panel_view(panel);
            }
            DashboardAction::Resize(layout) => {
                self.apply_layout(layout);
            }
        }
    }

    fn apply_layout(&mut self, layout: DashboardLayoutState) {
        self.panel_layout = layout;
        for panel in DashboardPanel::ALL {
            self.panel_view_state_mut(panel).viewport_rows = if panel == DashboardPanel::JoinToken {
                self.join_token_viewport_columns()
            } else {
                tui_panel_viewport_rows(panel, self.panel_layout.rows_for(panel))
            };
            self.clamp_panel_view(panel);
        }
        self.sync_events_panel();
    }

    fn apply_snapshot(&mut self, snapshot: &DashboardSnapshot) {
        if self.shutdown_in_progress {
            self.merge_shutdown_process_snapshot(snapshot);
        } else {
            self.llama_process_rows = snapshot.llama_process_rows.clone();
            self.webserver_rows = snapshot.webserver_rows.clone();
            self.loaded_model_rows = snapshot.loaded_model_rows.clone();
        }
        self.request_history = DashboardRequestHistoryState::from_snapshot(snapshot);
        self.clamp_all_panel_views();
        self.sync_events_panel();
    }

    fn merge_shutdown_process_snapshot(&mut self, snapshot: &DashboardSnapshot) {
        for snapshot_row in &snapshot.llama_process_rows {
            if let Some(existing) = self
                .llama_process_rows
                .iter_mut()
                .find(|row| row.name == snapshot_row.name)
            {
                *existing = snapshot_row.clone();
            } else {
                self.llama_process_rows.push(snapshot_row.clone());
            }
        }
        self.llama_process_rows
            .sort_by_key(|row| row.name.to_lowercase());

        for snapshot_row in &snapshot.loaded_model_rows {
            if let Some(existing) = self
                .loaded_model_rows
                .iter_mut()
                .find(|row| row.name == snapshot_row.name)
            {
                *existing = snapshot_row.clone();
            } else {
                self.loaded_model_rows.push(snapshot_row.clone());
            }
        }
        self.loaded_model_rows
            .sort_by(|left, right| left.name.cmp(&right.name));

        for snapshot_row in &snapshot.webserver_rows {
            if let Some(existing) = self
                .webserver_rows
                .iter_mut()
                .find(|row| row.label == snapshot_row.label && row.port == snapshot_row.port)
            {
                *existing = snapshot_row.clone();
            } else {
                self.webserver_rows.push(snapshot_row.clone());
            }
        }
        self.webserver_rows
            .sort_by(|left, right| left.label.cmp(&right.label));
    }

    fn mark_runtime_shutting_down(&mut self) {
        self.shutdown_in_progress = true;
        self.runtime_ready = false;
        for instance in &mut self.llama_instances {
            instance.status = RuntimeStatus::ShuttingDown;
        }
        for model in &mut self.running_models {
            model.status = RuntimeStatus::ShuttingDown;
        }
        for row in &mut self.llama_process_rows {
            row.status = RuntimeStatus::ShuttingDown;
        }
        for row in &mut self.loaded_model_rows {
            row.status = RuntimeStatus::ShuttingDown;
        }
        for row in &mut self.webserver_rows {
            row.status = RuntimeStatus::ShuttingDown;
        }
        if let Some(webserver) = &mut self.webserver {
            webserver.status = RuntimeStatus::ShuttingDown;
        }
        if let Some(api) = &mut self.api {
            api.status = RuntimeStatus::ShuttingDown;
        }
    }

    fn is_startup_loading(&self) -> bool {
        !self.runtime_ready && self.active_loading_progress().is_some()
    }

    fn active_loading_progress(&self) -> Option<LoadingProgressState> {
        if self.runtime_ready {
            return None;
        }

        if let Some(progress) = self.model_progress.as_ref() {
            if let Some(ratio) = model_download_progress_ratio(progress) {
                return Some(LoadingProgressState {
                    ratio,
                    detail: loading_progress_detail(model_progress_detail(progress), ratio, None),
                });
            }
        }

        if let Some(progress) = self.startup_progress.as_ref() {
            let ratio = startup_progress_ratio(progress);
            return Some(LoadingProgressState {
                ratio,
                detail: loading_progress_detail(
                    progress.detail.clone(),
                    ratio,
                    Some((progress.completed_steps, progress.total_steps)),
                ),
            });
        }

        self.model_progress.as_ref().map(|progress| {
            let ratio = fallback_model_progress_ratio(progress);
            LoadingProgressState {
                ratio,
                detail: loading_progress_detail(model_progress_detail(progress), ratio, None),
            }
        })
    }

    fn apply_startup_progress_event(&mut self, event: &OutputEvent) {
        if matches!(event, OutputEvent::Startup { .. }) {
            self.startup_milestones.clear();
            self.startup_progress = None;
        }

        let Some((milestone_key, detail)) = startup_progress_event(event) else {
            return;
        };

        if let Some(key) = milestone_key {
            self.startup_milestones.insert(key);
        }

        let completed_steps = self.startup_milestones.len();
        let total_steps = if matches!(event, OutputEvent::RuntimeReady { .. }) {
            completed_steps.max(1)
        } else {
            PRETTY_TUI_STARTUP_PROGRESS_MIN_STEPS.max(completed_steps.saturating_add(1))
        };

        self.startup_progress = Some(StartupProgressState {
            completed_steps,
            total_steps,
            detail,
        });
    }

    fn apply_output_event(&mut self, event: &OutputEvent) {
        match event {
            OutputEvent::Startup { version, .. } => {
                self.version = Some(version.clone());
                self.runtime_ready = false;
            }
            OutputEvent::NodeIdentity { node_id, mesh_id } => {
                self.node_id = Some(node_id.clone());
                self.mesh_id = mesh_id.clone();
            }
            OutputEvent::ModelQueued { model } | OutputEvent::ModelLoading { model, .. } => {
                self.upsert_model(model, RuntimeStatus::Starting, None, None, None);
            }
            OutputEvent::ModelLoaded { model, .. } => {
                self.upsert_model(model, RuntimeStatus::Starting, None, None, None);
            }
            OutputEvent::MoeDetected { model, moe, .. } => {
                self.upsert_model(model, RuntimeStatus::Starting, None, None, None);
                if let Some(existing) = self
                    .running_models
                    .iter_mut()
                    .find(|candidate| candidate.model == *model)
                {
                    existing.moe = Some(moe.clone());
                }
            }
            OutputEvent::MoeDistribution {
                model,
                moe,
                distribution,
            } => {
                self.upsert_model(model, RuntimeStatus::Starting, None, None, None);
                if let Some(existing) = self
                    .running_models
                    .iter_mut()
                    .find(|candidate| candidate.model == *model)
                {
                    existing.moe = Some(moe.clone());
                    existing.moe_distribution = Some(distribution.clone());
                }
            }
            OutputEvent::MoeAnalysisProgress { .. } => {}
            OutputEvent::MoeStatus { model, .. } => {
                self.upsert_model(model, RuntimeStatus::Starting, None, None, None);
            }
            OutputEvent::ModelReady {
                model,
                internal_port,
                role,
            } => {
                self.upsert_model(
                    model,
                    RuntimeStatus::Ready,
                    *internal_port,
                    role.clone(),
                    None,
                );
            }
            OutputEvent::HostElected {
                model,
                role,
                capacity_gb,
                ..
            } => {
                self.upsert_model(
                    model,
                    RuntimeStatus::Starting,
                    None,
                    role.clone(),
                    *capacity_gb,
                );
            }
            OutputEvent::PassiveMode {
                role,
                status,
                capacity_gb,
                models_on_disk,
                detail,
            } => {
                let next_models_on_disk = models_on_disk.clone().unwrap_or_default();
                if let Some(existing) = self.passive_mode.as_mut() {
                    existing.role = role.clone();
                    existing.status = status.clone();
                    existing.capacity_gb = capacity_gb.or(existing.capacity_gb);
                    if models_on_disk.is_some() {
                        existing.models_on_disk = next_models_on_disk;
                    }
                    existing.detail = detail.clone().or_else(|| existing.detail.clone());
                } else {
                    self.passive_mode = Some(PassiveModeState {
                        role: role.clone(),
                        status: status.clone(),
                        capacity_gb: *capacity_gb,
                        models_on_disk: next_models_on_disk,
                        detail: detail.clone(),
                    });
                }
            }
            OutputEvent::MultiModelMode { count, models } => {
                self.multi_model_mode = Some(MultiModelModeState {
                    count: *count,
                    models: models.clone(),
                });
            }
            OutputEvent::RpcServerStarting {
                port,
                device,
                log_path,
            } => self.upsert_llama_instance(LlamaInstanceState {
                kind: LlamaInstanceKind::RpcServer,
                port: *port,
                status: RuntimeStatus::Starting,
                device: Some(device.clone()),
                model: None,
                ctx_size: None,
                log_path: log_path.clone(),
            }),
            OutputEvent::RpcReady {
                port,
                device,
                log_path,
            } => self.upsert_llama_instance(LlamaInstanceState {
                kind: LlamaInstanceKind::RpcServer,
                port: *port,
                status: RuntimeStatus::Ready,
                device: Some(device.clone()),
                model: None,
                ctx_size: None,
                log_path: log_path.clone(),
            }),
            OutputEvent::LlamaStarting {
                model,
                http_port,
                ctx_size,
                log_path,
            } => self.upsert_llama_instance(LlamaInstanceState {
                kind: LlamaInstanceKind::LlamaServer,
                port: *http_port,
                status: RuntimeStatus::Starting,
                device: None,
                model: model.clone(),
                ctx_size: *ctx_size,
                log_path: log_path.clone(),
            }),
            OutputEvent::LlamaReady {
                model,
                port,
                ctx_size,
                log_path,
            } => self.upsert_llama_instance(LlamaInstanceState {
                kind: LlamaInstanceKind::LlamaServer,
                port: *port,
                status: RuntimeStatus::Ready,
                device: None,
                model: model.clone(),
                ctx_size: *ctx_size,
                log_path: log_path.clone(),
            }),
            OutputEvent::WebserverStarting { url } => {
                self.webserver = Some(EndpointState {
                    label: "Console".to_string(),
                    status: RuntimeStatus::Starting,
                    url: url.clone(),
                    details: Vec::new(),
                });
            }
            OutputEvent::WebserverReady { url } => {
                self.webserver = Some(EndpointState {
                    label: "Console".to_string(),
                    status: RuntimeStatus::Ready,
                    url: url.clone(),
                    details: Vec::new(),
                });
            }
            OutputEvent::ApiStarting { url } => {
                self.api = Some(EndpointState {
                    label: "OpenAI-compatible API".to_string(),
                    status: RuntimeStatus::Starting,
                    url: url.clone(),
                    details: Vec::new(),
                });
            }
            OutputEvent::ApiReady { url } => {
                self.api = Some(EndpointState {
                    label: "OpenAI-compatible API".to_string(),
                    status: RuntimeStatus::Ready,
                    url: url.clone(),
                    details: Vec::new(),
                });
            }
            OutputEvent::RuntimeReady {
                api_url,
                console_url,
                api_port: _api_port,
                console_port: _console_port,
                pi_command,
                goose_command,
                ..
            } => {
                self.runtime_ready = true;
                self.model_progress = None;
                if let Some(console_url) = console_url.clone() {
                    self.webserver = Some(EndpointState {
                        label: "Console".to_string(),
                        status: RuntimeStatus::Ready,
                        url: console_url,
                        details: Vec::new(),
                    });
                }
                let mut details = Vec::new();
                if let Some(pi_command) = pi_command.clone() {
                    details.push(format!("pi:    {pi_command}"));
                }
                if let Some(goose_command) = goose_command.clone() {
                    details.push(format!("goose: {goose_command}"));
                }
                self.api = Some(EndpointState {
                    label: "OpenAI-compatible API".to_string(),
                    status: RuntimeStatus::Ready,
                    url: api_url.clone(),
                    details,
                });
            }
            OutputEvent::ModelDownloadProgress {
                label,
                file,
                downloaded_bytes,
                total_bytes,
                status,
            } => {
                self.model_progress = Some(ModelProgressState {
                    label: label.clone(),
                    file: file.clone(),
                    downloaded_bytes: *downloaded_bytes,
                    total_bytes: *total_bytes,
                    status: status.clone(),
                });
            }
            OutputEvent::Shutdown { .. } => {
                self.mark_runtime_shutting_down();
            }
            OutputEvent::Error { .. } => {
                if let Some(model) = self.running_models.last_mut() {
                    model.status = RuntimeStatus::Error;
                }
            }
            OutputEvent::InviteToken {
                token,
                mesh_id,
                mesh_name,
            } => {
                self.join_token = Some(DashboardJoinTokenState::new(
                    token.clone(),
                    mesh_id.clone(),
                    mesh_name.clone(),
                ));
                let join_token_view = self.panel_view_state_mut(DashboardPanel::JoinToken);
                join_token_view.scroll_offset = 0;
                join_token_view.selected_row = None;
            }
            OutputEvent::Info { .. }
            | OutputEvent::Warning { .. }
            | OutputEvent::DiscoveryStarting { .. }
            | OutputEvent::MeshFound { .. }
            | OutputEvent::DiscoveryJoined { .. }
            | OutputEvent::DiscoveryFailed { .. }
            | OutputEvent::WaitingForPeers { .. }
            | OutputEvent::RequestRouted { .. } => {}
            OutputEvent::PeerJoined { peer_id, .. } => {
                self.peer_ids.insert(peer_id.clone());
            }
            OutputEvent::PeerLeft { peer_id, .. } => {
                self.peer_ids.remove(peer_id);
            }
        }

        self.apply_startup_progress_event(event);
        self.record_mesh_event(event);
        self.clamp_all_panel_views();
        self.sync_events_panel();
    }

    fn panel_view_state(&self, panel: DashboardPanel) -> DashboardPanelViewState {
        self.panel_view_states[panel.index()]
    }

    fn panel_view_state_mut(&mut self, panel: DashboardPanel) -> &mut DashboardPanelViewState {
        &mut self.panel_view_states[panel.index()]
    }

    fn filtered_mesh_events(&self) -> Vec<&MeshEventState> {
        if !self.events_filter.is_active() {
            return self.mesh_events.iter().collect();
        }

        let needle = self.events_filter.query.to_lowercase();
        self.mesh_events
            .iter()
            .filter(|event| event_matches_filter(event, &needle))
            .collect()
    }

    fn row_count_for_panel(&self, panel: DashboardPanel) -> usize {
        match panel {
            DashboardPanel::JoinToken => self
                .join_token
                .as_ref()
                .map(|join_token| join_token_char_count(&join_token.token))
                .unwrap_or(0),
            DashboardPanel::Events => self.filtered_mesh_events().len(),
            DashboardPanel::LlamaCpp => self.llama_process_rows.len(),
            DashboardPanel::Webserver => self.webserver_rows.len(),
            DashboardPanel::Models => self.loaded_model_rows.len(),
            DashboardPanel::Requests => {
                usize::from(!self.request_history.accepted_request_buckets.is_empty())
            }
        }
    }

    fn rows_are_selectable_for_panel(&self, panel: DashboardPanel) -> bool {
        self.panel_layout.rows_are_selectable_for(panel)
    }

    fn clamp_all_panel_views(&mut self) {
        for panel in DashboardPanel::ALL {
            self.clamp_panel_view(panel);
        }
    }

    fn clamp_panel_view(&mut self, panel: DashboardPanel) {
        let row_count = self.row_count_for_panel(panel);
        let rows_are_selectable = self.rows_are_selectable_for_panel(panel);
        let panel_view = self.panel_view_state_mut(panel);
        let viewport_rows = panel_view.viewport_rows.max(1);

        if row_count == 0 {
            panel_view.scroll_offset = 0;
            panel_view.selected_row = None;
            return;
        }

        let max_scroll_offset = row_count.saturating_sub(viewport_rows);
        panel_view.scroll_offset = panel_view.scroll_offset.min(max_scroll_offset);
        if !rows_are_selectable {
            panel_view.selected_row = None;
            return;
        }
        panel_view.selected_row = panel_view
            .selected_row
            .map(|selected| selected.min(row_count - 1));

        if panel == DashboardPanel::Events
            && TuiEventListRenderer::ACTIVE == TuiEventListRenderer::Scrollbar
        {
            return;
        }

        if let Some(selected_row) = panel_view.selected_row {
            if selected_row < panel_view.scroll_offset {
                panel_view.scroll_offset = selected_row;
            }
            let visible_end = panel_view.scroll_offset + viewport_rows;
            if selected_row >= visible_end {
                panel_view.scroll_offset = selected_row + 1 - viewport_rows;
            }
            panel_view.scroll_offset = panel_view.scroll_offset.min(max_scroll_offset);
        }
    }

    fn sync_events_panel(&mut self) {
        if !self.events_follow {
            self.clamp_panel_view(DashboardPanel::Events);
            return;
        }

        let row_count = self.filtered_mesh_events().len();
        let events_view = self.panel_view_state_mut(DashboardPanel::Events);
        if row_count == 0 {
            events_view.scroll_offset = 0;
            events_view.selected_row = None;
            return;
        }

        let viewport_rows = events_view.viewport_rows.max(1);
        events_view.selected_row = Some(row_count - 1);
        events_view.scroll_offset = row_count.saturating_sub(viewport_rows);
    }

    fn event_scroll_bounds(&self) -> (usize, usize, usize) {
        let row_count = self.row_count_for_panel(DashboardPanel::Events);
        let viewport_rows = self
            .panel_view_state(DashboardPanel::Events)
            .viewport_rows
            .max(1);
        let max_scroll_offset = row_count.saturating_sub(viewport_rows);
        (row_count, viewport_rows, max_scroll_offset)
    }

    fn scroll_events_by(&mut self, delta: isize) {
        let (row_count, _viewport_rows, max_scroll_offset) = self.event_scroll_bounds();
        let was_following = self.events_follow;
        let current_scroll = if was_following {
            max_scroll_offset
        } else {
            self.panel_view_state(DashboardPanel::Events)
                .scroll_offset
                .min(max_scroll_offset)
        };
        let events_view = self.panel_view_state_mut(DashboardPanel::Events);
        if row_count == 0 {
            events_view.scroll_offset = 0;
            events_view.selected_row = None;
            self.events_follow = true;
            return;
        }

        let next_scroll = current_scroll
            .saturating_add_signed(delta)
            .min(max_scroll_offset);
        events_view.scroll_offset = next_scroll;
        events_view.selected_row = row_count.checked_sub(1);
        self.events_follow = next_scroll == max_scroll_offset;
    }

    fn page_events_by(&mut self, direction: isize) {
        let (_row_count, viewport_rows, _max_scroll_offset) = self.event_scroll_bounds();
        let step = viewport_rows.saturating_sub(1).max(1) as isize;
        self.scroll_events_by(direction.saturating_mul(step));
    }

    fn jump_events_to_start(&mut self) {
        let (row_count, _viewport_rows, _max_scroll_offset) = self.event_scroll_bounds();
        let events_view = self.panel_view_state_mut(DashboardPanel::Events);
        if row_count == 0 {
            events_view.scroll_offset = 0;
            events_view.selected_row = None;
            self.events_follow = true;
        } else {
            events_view.scroll_offset = 0;
            events_view.selected_row = row_count.checked_sub(1);
            self.events_follow = false;
        }
    }

    fn jump_events_to_end(&mut self) {
        self.events_follow = true;
        self.sync_events_panel();
    }

    fn move_panel_selection(&mut self, panel: DashboardPanel, delta: isize) {
        let row_count = self.row_count_for_panel(panel);
        if row_count == 0 {
            return;
        }

        if !self.rows_are_selectable_for_panel(panel) {
            self.scroll_panel_rows_by(panel, delta);
            return;
        }

        let current = self
            .panel_view_state(panel)
            .selected_row
            .unwrap_or_else(|| {
                if delta.is_negative() || (panel == DashboardPanel::Events && self.events_follow) {
                    row_count - 1
                } else {
                    0
                }
            });

        let next = current.saturating_add_signed(delta).min(row_count - 1);
        self.panel_view_state_mut(panel).selected_row = Some(next);
        self.clamp_panel_view(panel);
        self.sync_follow_with_events_view(panel);
    }

    fn page_panel_selection(&mut self, panel: DashboardPanel, direction: isize) {
        let row_count = self.row_count_for_panel(panel);
        if row_count == 0 {
            return;
        }

        let current_view = self.panel_view_state(panel);
        let step = self
            .panel_view_state(panel)
            .viewport_rows
            .saturating_sub(1)
            .max(1) as isize;
        let delta = direction.saturating_mul(step);
        if !self.rows_are_selectable_for_panel(panel) {
            self.scroll_panel_rows_by(panel, delta);
            return;
        }
        let current_selection = current_view.selected_row.unwrap_or_else(|| {
            if direction.is_negative() || (panel == DashboardPanel::Events && self.events_follow) {
                row_count - 1
            } else {
                0
            }
        });
        let next_selection = current_selection
            .saturating_add_signed(delta)
            .min(row_count - 1);
        let next_scroll = current_view.scroll_offset.saturating_add_signed(delta);
        let panel_view = self.panel_view_state_mut(panel);
        panel_view.selected_row = Some(next_selection);
        panel_view.scroll_offset = next_scroll;
        self.clamp_panel_view(panel);
        self.sync_follow_with_events_view(panel);
    }

    fn jump_panel_selection_to_start(&mut self, panel: DashboardPanel) {
        if self.row_count_for_panel(panel) == 0 {
            return;
        }
        if !self.rows_are_selectable_for_panel(panel) {
            let panel_view = self.panel_view_state_mut(panel);
            panel_view.scroll_offset = 0;
            panel_view.selected_row = None;
            return;
        }
        self.panel_view_state_mut(panel).selected_row = Some(0);
        self.clamp_panel_view(panel);
        self.sync_follow_with_events_view(panel);
    }

    fn jump_panel_selection_to_end(&mut self, panel: DashboardPanel) {
        let row_count = self.row_count_for_panel(panel);
        if row_count == 0 {
            return;
        }
        if !self.rows_are_selectable_for_panel(panel) {
            let viewport_rows = self.panel_view_state(panel).viewport_rows.max(1);
            let panel_view = self.panel_view_state_mut(panel);
            panel_view.scroll_offset = row_count.saturating_sub(viewport_rows);
            panel_view.selected_row = None;
            return;
        }
        self.panel_view_state_mut(panel).selected_row = Some(row_count - 1);
        self.clamp_panel_view(panel);
        self.sync_follow_with_events_view(panel);
    }

    fn scroll_panel_rows_by(&mut self, panel: DashboardPanel, delta: isize) {
        let row_count = self.row_count_for_panel(panel);
        if row_count == 0 {
            return;
        }
        let current_view = self.panel_view_state(panel);
        let max_scroll_offset = row_count.saturating_sub(current_view.viewport_rows.max(1));
        let next_scroll = current_view
            .scroll_offset
            .saturating_add_signed(delta)
            .min(max_scroll_offset);
        let panel_view = self.panel_view_state_mut(panel);
        panel_view.scroll_offset = next_scroll;
        panel_view.selected_row = None;
    }

    fn join_token_viewport_columns(&self) -> usize {
        let Some((columns, rows)) = self.terminal_size else {
            return 1;
        };
        let areas = tui_layout(
            Rect {
                x: 0,
                y: 0,
                width: columns,
                height: rows,
            },
            self,
        );
        usize::from(join_token_content_width(
            areas.join_token_panel,
            areas.join_token_copy_button,
        ))
        .max(1)
    }

    fn scroll_join_token_by(&mut self, delta: isize) {
        let row_count = self.row_count_for_panel(DashboardPanel::JoinToken);
        if row_count == 0 {
            return;
        }
        let viewport_columns = self.join_token_viewport_columns();
        let max_scroll_offset = row_count.saturating_sub(viewport_columns);
        let current = self
            .panel_view_state(DashboardPanel::JoinToken)
            .scroll_offset
            .min(max_scroll_offset);
        let next = current.saturating_add_signed(delta).min(max_scroll_offset);
        let join_token_view = self.panel_view_state_mut(DashboardPanel::JoinToken);
        join_token_view.viewport_rows = viewport_columns.max(1);
        join_token_view.scroll_offset = next;
        join_token_view.selected_row = None;
    }

    fn jump_join_token_to_start(&mut self) {
        self.panel_view_state_mut(DashboardPanel::JoinToken)
            .scroll_offset = 0;
    }

    fn jump_join_token_to_end(&mut self) {
        let row_count = self.row_count_for_panel(DashboardPanel::JoinToken);
        let viewport_columns = self.join_token_viewport_columns();
        let max_scroll_offset = row_count.saturating_sub(viewport_columns);
        let join_token_view = self.panel_view_state_mut(DashboardPanel::JoinToken);
        join_token_view.viewport_rows = viewport_columns.max(1);
        join_token_view.scroll_offset = max_scroll_offset;
        join_token_view.selected_row = None;
    }

    fn sync_follow_with_events_view(&mut self, panel: DashboardPanel) {
        if panel != DashboardPanel::Events {
            return;
        }

        let row_count = self.row_count_for_panel(DashboardPanel::Events);
        if row_count == 0 {
            self.events_follow = true;
            return;
        }

        let view = self.panel_view_state(DashboardPanel::Events);
        let viewport_rows = view.viewport_rows.max(1);
        if row_count <= viewport_rows {
            if view.selected_row != Some(row_count - 1) {
                self.events_follow = false;
            }
            return;
        }

        let last_row = row_count - 1;
        let at_bottom =
            view.selected_row == Some(last_row) && view.scroll_offset + viewport_rows >= row_count;
        self.events_follow = at_bottom;
        if self.events_follow {
            self.sync_events_panel();
        }
    }

    fn upsert_llama_instance(&mut self, next: LlamaInstanceState) {
        if let Some(existing) = self
            .llama_instances
            .iter_mut()
            .find(|candidate| candidate.kind == next.kind && candidate.port == next.port)
        {
            *existing = next;
        } else {
            self.llama_instances.push(next);
        }

        self.llama_instances
            .sort_by_key(|instance| (instance.kind.sort_key(), instance.port));
    }

    fn upsert_model(
        &mut self,
        model: &str,
        status: RuntimeStatus,
        internal_port: Option<u16>,
        role: Option<String>,
        capacity_gb: Option<f64>,
    ) {
        if let Some(existing) = self
            .running_models
            .iter_mut()
            .find(|candidate| candidate.model == model)
        {
            if !matches!(existing.status, RuntimeStatus::Ready)
                || matches!(status, RuntimeStatus::Ready)
            {
                existing.status = status;
            }
            existing.internal_port = internal_port.or(existing.internal_port);
            existing.role = role.or_else(|| existing.role.clone());
            existing.capacity_gb = capacity_gb.or(existing.capacity_gb);
        } else {
            self.running_models.push(RunningModelState {
                model: model.to_string(),
                status,
                internal_port,
                role,
                capacity_gb,
                moe: None,
                moe_distribution: None,
            });
        }

        self.running_models
            .sort_by(|left, right| left.model.cmp(&right.model));
    }

    fn record_mesh_event(&mut self, event: &OutputEvent) {
        self.mesh_events.push_back(MeshEventState {
            timestamp: Local::now().format("%H:%M:%S").to_string(),
            level: event.level(),
            summary: event.summary_line(),
        });
        while self.mesh_events.len() > self.mesh_event_limit {
            self.mesh_events.pop_front();
        }
    }

    fn copy_join_token(&mut self) {
        let Some(token) = self
            .join_token
            .as_ref()
            .map(|join_token| join_token.token.clone())
        else {
            return;
        };
        let copy_status = match copy_join_token_to_clipboard(&token) {
            Ok(()) => DashboardJoinTokenCopyStatus::Copied,
            Err(message) => DashboardJoinTokenCopyStatus::Failed(message),
        };
        self.reduce(DashboardAction::SetJoinTokenCopyStatus(copy_status));
    }

    fn join_token_copy_button_contains(&self, column: u16, row: u16) -> bool {
        let Some((columns, rows)) = self.terminal_size else {
            return false;
        };
        let areas = tui_layout(
            Rect {
                x: 0,
                y: 0,
                width: columns,
                height: rows,
            },
            self,
        );
        point_in_rect(column, row, areas.join_token_copy_button)
    }

    fn join_token_panel_contains(&self, column: u16, row: u16) -> bool {
        let Some((columns, rows)) = self.terminal_size else {
            return false;
        };
        let areas = tui_layout(
            Rect {
                x: 0,
                y: 0,
                width: columns,
                height: rows,
            },
            self,
        );
        point_in_rect(column, row, areas.join_token_panel)
    }

    fn apply_tui_event(&mut self, event: TuiEvent) -> TuiControlFlow {
        match event {
            TuiEvent::Resize { columns, rows } => {
                self.terminal_size = Some((columns, rows));
                self.reduce(DashboardAction::Resize(dashboard_layout_for_terminal_size(
                    columns, rows,
                )));
                TuiControlFlow::Continue
            }
            TuiEvent::MouseDown { column, row }
                if self.join_token_copy_button_contains(column, row) =>
            {
                self.panel_focus = DashboardPanel::JoinToken;
                self.copy_join_token();
                TuiControlFlow::Continue
            }
            TuiEvent::MouseDown { column, row } if self.join_token_panel_contains(column, row) => {
                self.panel_focus = DashboardPanel::JoinToken;
                self.events_filter.editing = false;
                TuiControlFlow::Continue
            }
            TuiEvent::Key(TuiKeyEvent::Interrupt) => {
                self.mark_runtime_shutting_down();
                TuiControlFlow::Quit
            }
            TuiEvent::Key(TuiKeyEvent::Char('q')) if !self.events_filter.editing => {
                self.mark_runtime_shutting_down();
                TuiControlFlow::Quit
            }
            TuiEvent::Key(TuiKeyEvent::Tab) => {
                self.reduce(DashboardAction::FocusNextPanel);
                TuiControlFlow::Continue
            }
            TuiEvent::Key(TuiKeyEvent::BackTab) => {
                self.reduce(DashboardAction::FocusPreviousPanel);
                TuiControlFlow::Continue
            }
            TuiEvent::Key(TuiKeyEvent::Char('/')) if !self.events_filter.editing => {
                self.reduce(DashboardAction::StartEventsFilterEdit);
                TuiControlFlow::Continue
            }
            TuiEvent::Key(TuiKeyEvent::Char('f')) if !self.events_filter.editing => {
                self.reduce(DashboardAction::ToggleEventsFollow);
                TuiControlFlow::Continue
            }
            TuiEvent::Key(TuiKeyEvent::Char('c'))
                if !self.events_filter.editing
                    && self.panel_focus == DashboardPanel::JoinToken
                    && self.join_token.is_some() =>
            {
                self.copy_join_token();
                TuiControlFlow::Continue
            }
            TuiEvent::Key(TuiKeyEvent::Left) | TuiEvent::Key(TuiKeyEvent::Char('h'))
                if !self.events_filter.editing && self.panel_focus == DashboardPanel::JoinToken =>
            {
                self.scroll_join_token_by(-1);
                TuiControlFlow::Continue
            }
            TuiEvent::Key(TuiKeyEvent::Right) | TuiEvent::Key(TuiKeyEvent::Char('l'))
                if !self.events_filter.editing && self.panel_focus == DashboardPanel::JoinToken =>
            {
                self.scroll_join_token_by(1);
                TuiControlFlow::Continue
            }
            TuiEvent::Key(TuiKeyEvent::Char('g'))
                if !self.events_filter.editing && self.panel_focus == DashboardPanel::JoinToken =>
            {
                self.jump_join_token_to_start();
                TuiControlFlow::Continue
            }
            TuiEvent::Key(TuiKeyEvent::Char('G'))
                if !self.events_filter.editing && self.panel_focus == DashboardPanel::JoinToken =>
            {
                self.jump_join_token_to_end();
                TuiControlFlow::Continue
            }
            TuiEvent::Key(TuiKeyEvent::Up)
            | TuiEvent::Key(TuiKeyEvent::Char('k'))
            | TuiEvent::Key(TuiKeyEvent::Down)
            | TuiEvent::Key(TuiKeyEvent::Char('j'))
            | TuiEvent::Key(TuiKeyEvent::PageUp)
            | TuiEvent::Key(TuiKeyEvent::PageDown)
                if !self.events_filter.editing && self.panel_focus == DashboardPanel::JoinToken =>
            {
                TuiControlFlow::Continue
            }
            TuiEvent::Key(TuiKeyEvent::Up)
                if !self.events_filter.editing && self.panel_focus == DashboardPanel::Requests =>
            {
                self.reduce(DashboardAction::SelectPreviousRequestWindow);
                TuiControlFlow::Continue
            }
            TuiEvent::Key(TuiKeyEvent::Down)
                if !self.events_filter.editing && self.panel_focus == DashboardPanel::Requests =>
            {
                self.reduce(DashboardAction::SelectNextRequestWindow);
                TuiControlFlow::Continue
            }
            TuiEvent::Key(TuiKeyEvent::Up) | TuiEvent::Key(TuiKeyEvent::Char('k'))
                if !self.events_filter.editing
                    && self.panel_focus == DashboardPanel::Events
                    && TuiEventListRenderer::ACTIVE == TuiEventListRenderer::Scrollbar =>
            {
                self.scroll_events_by(-1);
                TuiControlFlow::Continue
            }
            TuiEvent::Key(TuiKeyEvent::Down) | TuiEvent::Key(TuiKeyEvent::Char('j'))
                if !self.events_filter.editing
                    && self.panel_focus == DashboardPanel::Events
                    && TuiEventListRenderer::ACTIVE == TuiEventListRenderer::Scrollbar =>
            {
                self.scroll_events_by(1);
                TuiControlFlow::Continue
            }
            TuiEvent::Key(TuiKeyEvent::PageUp)
                if !self.events_filter.editing
                    && self.panel_focus == DashboardPanel::Events
                    && TuiEventListRenderer::ACTIVE == TuiEventListRenderer::Scrollbar =>
            {
                self.page_events_by(-1);
                TuiControlFlow::Continue
            }
            TuiEvent::Key(TuiKeyEvent::PageDown)
                if !self.events_filter.editing
                    && self.panel_focus == DashboardPanel::Events
                    && TuiEventListRenderer::ACTIVE == TuiEventListRenderer::Scrollbar =>
            {
                self.page_events_by(1);
                TuiControlFlow::Continue
            }
            TuiEvent::Key(TuiKeyEvent::Char('g'))
                if !self.events_filter.editing
                    && self.panel_focus == DashboardPanel::Events
                    && TuiEventListRenderer::ACTIVE == TuiEventListRenderer::Scrollbar =>
            {
                self.jump_events_to_start();
                TuiControlFlow::Continue
            }
            TuiEvent::Key(TuiKeyEvent::Char('G'))
                if !self.events_filter.editing
                    && self.panel_focus == DashboardPanel::Events
                    && TuiEventListRenderer::ACTIVE == TuiEventListRenderer::Scrollbar =>
            {
                self.jump_events_to_end();
                TuiControlFlow::Continue
            }
            TuiEvent::Key(TuiKeyEvent::Left)
            | TuiEvent::Key(TuiKeyEvent::Char('h'))
            | TuiEvent::Key(TuiKeyEvent::Up)
            | TuiEvent::Key(TuiKeyEvent::Char('k'))
                if !self.events_filter.editing =>
            {
                self.move_panel_selection(self.panel_focus, -1);
                TuiControlFlow::Continue
            }
            TuiEvent::Key(TuiKeyEvent::Right)
            | TuiEvent::Key(TuiKeyEvent::Char('l'))
            | TuiEvent::Key(TuiKeyEvent::Down)
            | TuiEvent::Key(TuiKeyEvent::Char('j'))
                if !self.events_filter.editing =>
            {
                self.move_panel_selection(self.panel_focus, 1);
                TuiControlFlow::Continue
            }
            TuiEvent::Key(TuiKeyEvent::PageUp) if !self.events_filter.editing => {
                self.page_panel_selection(self.panel_focus, -1);
                TuiControlFlow::Continue
            }
            TuiEvent::Key(TuiKeyEvent::PageDown) if !self.events_filter.editing => {
                self.page_panel_selection(self.panel_focus, 1);
                TuiControlFlow::Continue
            }
            TuiEvent::Key(TuiKeyEvent::Char('g')) if !self.events_filter.editing => {
                self.jump_panel_selection_to_start(self.panel_focus);
                TuiControlFlow::Continue
            }
            TuiEvent::Key(TuiKeyEvent::Char('G')) if !self.events_filter.editing => {
                self.jump_panel_selection_to_end(self.panel_focus);
                TuiControlFlow::Continue
            }
            TuiEvent::Key(TuiKeyEvent::Backspace) if self.events_filter.editing => {
                self.reduce(DashboardAction::BackspaceEventsFilter);
                TuiControlFlow::Continue
            }
            TuiEvent::Key(TuiKeyEvent::Enter) if self.events_filter.editing => {
                self.reduce(DashboardAction::ConfirmEventsFilter);
                TuiControlFlow::Continue
            }
            TuiEvent::Key(TuiKeyEvent::Escape) if self.events_filter.editing => {
                self.reduce(DashboardAction::CancelEventsFilter);
                TuiControlFlow::Continue
            }
            TuiEvent::Key(TuiKeyEvent::Char(ch))
                if self.events_filter.editing && !ch.is_control() =>
            {
                self.reduce(DashboardAction::InsertEventsFilterChar(ch));
                TuiControlFlow::Continue
            }
            _ => TuiControlFlow::Continue,
        }
    }
}

fn render_dashboard_text(state: &DashboardState) -> String {
    let mut output = String::new();
    let mut header = String::from("mesh-llm");
    if let Some(version) = &state.version {
        header.push(' ');
        header.push_str(version);
    }
    if let Some(node_id) = &state.node_id {
        header.push_str(&format!("  node={node_id}"));
    }
    if let Some(mesh_id) = &state.mesh_id {
        header.push_str(&format!("  mesh={mesh_id}"));
    }
    let _ = writeln!(&mut output, "{header}");
    let _ = writeln!(&mut output);

    write_dashboard_section(
        &mut output,
        "Running llama.cpp instances",
        &render_llama_instances(state),
    );
    let _ = writeln!(&mut output);
    write_dashboard_section(&mut output, "Running models", &render_models(state));
    let _ = writeln!(&mut output);
    write_dashboard_section(&mut output, "Running webserver", &render_webserver(state));
    let _ = writeln!(&mut output);
    write_dashboard_section(&mut output, "Running API", &render_api(state));
    let _ = writeln!(&mut output);
    write_dashboard_section(
        &mut output,
        &format!("Mesh events (latest {})", state.mesh_event_limit),
        &render_mesh_events(state),
    );
    output
}

fn write_dashboard_section(output: &mut String, title: &str, lines: &[String]) {
    let _ = writeln!(
        output,
        "┌ {title} ────────────────────────────────────────────────────────────"
    );
    if lines.is_empty() {
        let _ = writeln!(output, "│ (none)");
    } else {
        for line in lines {
            let _ = writeln!(output, "│ {line}");
        }
    }
    let _ = writeln!(
        output,
        "└────────────────────────────────────────────────────────────────────"
    );
}

fn render_llama_instances(state: &DashboardState) -> Vec<String> {
    let mut lines = Vec::new();
    for instance in &state.llama_instances {
        let mut line = format!(
            "{}   {}   port={} ",
            instance.kind.as_str(),
            instance.status.as_str(),
            instance.port
        );
        if let Some(device) = &instance.device {
            line.push_str(&format!("  device={device}"));
        }
        if let Some(model) = &instance.model {
            line.push_str(&format!("  model={model}"));
        }
        if let Some(ctx_size) = instance.ctx_size {
            line.push_str(&format!("  ctx={ctx_size}"));
        }
        lines.push(line.trim_end().to_string());
        if let Some(log_path) = &instance.log_path {
            lines.push(format!("             logs={log_path}"));
        }
    }
    lines
}

fn render_models(state: &DashboardState) -> Vec<String> {
    let mut lines = Vec::new();
    if let Some(passive_mode) = &state.passive_mode {
        let mut line = format!("{}   {}", passive_mode.role, passive_mode.status.as_str());
        if let Some(capacity_gb) = passive_mode.capacity_gb {
            line.push_str(&format!("   capacity={capacity_gb:.1}GB"));
        }
        if !passive_mode.models_on_disk.is_empty() {
            line.push_str(&format!(
                "   models={}",
                passive_mode.models_on_disk.join(", ")
            ));
        }
        if let Some(detail) = &passive_mode.detail {
            line.push_str(&format!("   {detail}"));
        }
        lines.push(line);
    }
    if let Some(multi_model_mode) = &state.multi_model_mode {
        let models = if multi_model_mode.models.is_empty() {
            "(none)".to_string()
        } else {
            multi_model_mode.models.join(", ")
        };
        lines.push(format!(
            "multi-model mode   {} model(s)   models={models}",
            multi_model_mode.count
        ));
    }

    lines.extend(state.running_models.iter().map(|model| {
        let mut line = format!("{}   {}", model.model, model.status.as_str());
        if let Some(port) = model.internal_port {
            line.push_str(&format!("   port={port}"));
        }
        if let Some(role) = &model.role {
            line.push_str(&format!("   role={role}"));
        }
        if let Some(capacity_gb) = model.capacity_gb {
            line.push_str(&format!("   capacity={capacity_gb:.1}GB"));
        }
        if let Some(moe) = &model.moe {
            line.push_str(&format!("   MoE: {} experts, top-{}", moe.experts, moe.top_k));
        }
        if let Some(distribution) = &model.moe_distribution {
            line.push_str(&format!(
                "   split leader={} active={} fallback={} shard={}/{} ranking={} origin={} overlap={} shared={} unique={}",
                distribution.leader,
                distribution.active_nodes,
                distribution.fallback_nodes,
                distribution.shard_index,
                distribution.shard_count,
                distribution.ranking_source,
                distribution.ranking_origin,
                distribution.overlap,
                distribution.shared_experts,
                distribution.unique_experts,
            ));
        }
        line
    }));

    lines
}

fn render_webserver(state: &DashboardState) -> Vec<String> {
    render_endpoint(&state.webserver)
}

fn render_api(state: &DashboardState) -> Vec<String> {
    render_endpoint(&state.api)
}

fn render_endpoint(endpoint: &Option<EndpointState>) -> Vec<String> {
    endpoint
        .iter()
        .flat_map(|endpoint| {
            let mut lines = vec![format!(
                "{}   {}   {}",
                endpoint.label,
                endpoint.status.as_str(),
                endpoint.url
            )];
            lines.extend(
                endpoint
                    .details
                    .iter()
                    .map(|detail| format!("    {detail}")),
            );
            lines
        })
        .collect()
}

fn render_mesh_events(state: &DashboardState) -> Vec<String> {
    state
        .mesh_events
        .iter()
        .map(|event| {
            let (badge_text, _) = event_severity_badge(event);
            format!(
                "{} {:<PRETTY_TUI_EVENT_LEVEL_WIDTH$} {}",
                event.timestamp,
                badge_text,
                sanitize_mesh_event_message(&event.summary)
            )
        })
        .collect()
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct TuiListScrollbarLayout {
    list_area: Rect,
    scrollbar_area: Option<Rect>,
}

fn tui_list_scrollbar_layout(
    inner_area: Rect,
    row_count: usize,
    viewport_rows: usize,
) -> TuiListScrollbarLayout {
    let show_scrollbar = row_count > viewport_rows && inner_area.width > 1;
    let list_area = if show_scrollbar {
        Rect {
            width: inner_area.width.saturating_sub(1),
            ..inner_area
        }
    } else {
        inner_area
    };
    let scrollbar_area = show_scrollbar.then_some(Rect {
        x: inner_area.right().saturating_sub(1),
        y: inner_area.y,
        width: 1,
        height: inner_area.height,
    });
    TuiListScrollbarLayout {
        list_area,
        scrollbar_area,
    }
}

fn tui_list_scrollbar_state(
    row_count: usize,
    viewport_rows: usize,
    scroll_offset: usize,
) -> ScrollbarState {
    let visible_rows = viewport_rows.min(row_count);
    let scroll_positions = row_count.saturating_sub(visible_rows).saturating_add(1);
    ScrollbarState::new(scroll_positions)
        .position(scroll_offset.min(scroll_positions.saturating_sub(1)))
        .viewport_content_length(visible_rows)
}

#[cfg(test)]
fn render_tui_events_snapshot(state: &DashboardState, columns: u16, rows: u16) -> String {
    let width = usize::from(columns.max(40));
    let max_lines = usize::from(rows.max(3));
    let mut output = String::new();
    let _ = writeln!(&mut output, "{}", truncate_with_ellipsis("mesh-llm", width));
    let _ = writeln!(
        &mut output,
        "{}",
        truncate_with_ellipsis(
            &spans_plain_text(&dashboard_status_line(state, columns).spans),
            width
        )
    );
    let _ = writeln!(
        &mut output,
        "{}",
        truncate_with_ellipsis(
            &format_tui_panel_title(state, DashboardPanel::Events),
            width,
        )
    );

    for row in visible_event_rows(state, state.panel_layout.rows_for(DashboardPanel::Events)) {
        match row {
            TuiEventRow::Event { event, .. } => {
                let _ = writeln!(&mut output, "{}", format_event_row(event, width));
            }
            TuiEventRow::Message(message) => {
                let _ = writeln!(&mut output, "{}", truncate_with_ellipsis(message, width));
            }
            TuiEventRow::Padding => {
                let _ = writeln!(&mut output);
            }
        }
    }

    let mut lines: Vec<&str> = output.lines().collect();
    if lines.len() > max_lines {
        lines.truncate(max_lines);
        let mut truncated = lines.join("\n");
        truncated.push('\n');
        return truncated;
    }

    output
}

#[derive(Clone, Copy)]
enum TuiEventRow<'a> {
    Event {
        absolute_index: usize,
        event: &'a MeshEventState,
    },
    Message(&'static str),
    Padding,
}

type TuiTerminal = Terminal<CrosstermBackend<io::Stderr>>;

fn draw_tui_dashboard_with_terminal(
    terminal: &mut TuiTerminal,
    state: &DashboardState,
) -> io::Result<()> {
    terminal.hide_cursor().map_err(io::Error::other)?;
    terminal
        .set_cursor_position((0, 0))
        .map_err(io::Error::other)?;
    terminal
        .draw(|frame| render_tui_frame(frame, state))
        .map(|_| ())
        .map_err(io::Error::other)
}

fn render_tui_text_snapshot(state: &DashboardState, width: u16, height: u16) -> io::Result<String> {
    let width = width.max(1);
    let height = height.max(1);
    let backend = TestBackend::new(width, height);
    let mut terminal = Terminal::new(backend).map_err(io::Error::other)?;
    terminal
        .draw(|frame| render_tui_frame(frame, state))
        .map_err(io::Error::other)?;
    let buffer = terminal.backend().buffer();
    let mut lines = Vec::with_capacity(usize::from(height));
    for y in 0..height {
        let mut line = String::new();
        for x in 0..width {
            line.push_str(buffer[(x, y)].symbol());
        }
        lines.push(line.trim_end().to_string());
    }
    Ok(lines.join("\n"))
}

fn render_tui_frame(frame: &mut Frame, state: &DashboardState) {
    frame.render_widget(RatatuiClear, frame.area());

    if frame.area().width < PRETTY_TUI_MIN_DASHBOARD_WIDTH {
        render_tui_too_narrow_message(frame, frame.area());
        return;
    }

    let areas = tui_layout(frame.area(), state);
    let _main_body = areas.main_body;
    let full_screen_loading = state.is_startup_loading();

    if let Some(loading_area) = areas.loading.filter(|_| full_screen_loading) {
        render_model_progress_loader(frame, state, loading_area);
        return;
    }

    if let Some(logo_area) = areas.logo {
        render_tui_logo(frame, logo_area, true);
    }

    render_join_token_panel(
        frame,
        state,
        areas.join_token_panel,
        areas.join_token_copy_button,
    );

    frame.render_widget(
        Paragraph::new(dashboard_status_line(state, areas.status_bar.width))
            .style(tui_theme().status_bar),
        areas.status_bar,
    );

    render_events_panel(frame, state, areas.events.0, areas.events.1);
    render_processes_panel(
        frame,
        state,
        areas.processes,
        areas.llama_processes,
        areas.webserver_processes,
    );
    render_models_panel(frame, state, areas.models.0, areas.models.1);
    render_requests_panel(frame, state, areas.requests.0, areas.requests.1);
}

fn render_tui_too_narrow_message(frame: &mut Frame, area: Rect) {
    if area.width == 0 || area.height == 0 {
        return;
    }

    let message = Line::from(vec![
        Span::styled(
            "mesh-llm dashboard needs ",
            Style::default().fg(tui_theme().muted),
        ),
        Span::styled(
            format!(">= {PRETTY_TUI_MIN_DASHBOARD_WIDTH} columns"),
            Style::default().fg(tui_theme().warning),
        ),
        Span::styled(
            ". Resize or use line-oriented pretty output.",
            Style::default().fg(tui_theme().muted),
        ),
    ]);
    frame.render_widget(
        Paragraph::new(message)
            .alignment(Alignment::Center)
            .block(Block::bordered().border_type(BorderType::Rounded)),
        area,
    );
}

#[derive(Clone, Copy)]
struct TuiFrameAreas {
    loading: Option<Rect>,
    logo: Option<Rect>,
    join_token_panel: Rect,
    join_token_copy_button: Rect,
    main_body: Rect,
    requests: (Rect, Rect),
    status_bar: Rect,
    events: (Rect, Rect),
    processes: Rect,
    llama_processes: (Rect, Rect),
    webserver_processes: (Rect, Rect),
    models: (Rect, Rect),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct TuiBandHeights {
    join_token: u16,
    main_body: u16,
    requests: u16,
    status: u16,
}

fn tui_layout(area: Rect, state: &DashboardState) -> TuiFrameAreas {
    let zero = Rect {
        x: area.x,
        y: area.y,
        width: 0,
        height: 0,
    };

    if state.is_startup_loading() {
        return TuiFrameAreas {
            loading: Some(area),
            logo: None,
            join_token_panel: zero,
            join_token_copy_button: zero,
            main_body: zero,
            requests: (zero, zero),
            status_bar: zero,
            events: (zero, zero),
            processes: zero,
            llama_processes: (zero, zero),
            webserver_processes: (zero, zero),
            models: (zero, zero),
        };
    }

    let band_heights = tui_band_heights(area, state);
    let content_height = band_heights
        .main_body
        .saturating_add(band_heights.join_token)
        .saturating_add(band_heights.requests);
    let dashboard_height = content_height
        .saturating_add(band_heights.status)
        .min(area.height);
    let [slack_area, dashboard_area] = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(0), Constraint::Length(dashboard_height)])
        .areas(area);
    let loading = (slack_area.height > 0).then_some(slack_area);
    let logo = (state.runtime_ready && slack_area.height > 0)
        .then(|| tui_centered_logo_area(slack_area))
        .flatten();

    let [content_area, status_band] = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(content_height),
            Constraint::Length(band_heights.status),
        ])
        .areas(dashboard_area);

    let [join_token_panel, main_body, requests_band] = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(band_heights.join_token),
            Constraint::Length(band_heights.main_body),
            Constraint::Length(band_heights.requests),
        ])
        .areas(content_area);

    let join_token_copy_button = tui_join_token_copy_button_area(join_token_panel);

    let [events_column, processes_column, models_column] = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(PRETTY_TUI_EVENTS_COLUMN_PERCENT),
            Constraint::Fill(PRETTY_TUI_REMAINING_COLUMN_WEIGHT),
            Constraint::Fill(PRETTY_TUI_REMAINING_COLUMN_WEIGHT),
        ])
        .areas(main_body);
    let [events_title, events_body] = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(1), Constraint::Min(0)])
        .areas(events_column);
    let [models_title, models_body] = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(1), Constraint::Min(0)])
        .areas(models_column);

    let processes_block = tui_processes_block(state);
    let processes_inner = processes_block.inner(processes_column);
    let (llama_panel_height, webserver_panel_height) = tui_process_panel_heights(
        processes_inner.height,
        state.panel_layout.rows_for(DashboardPanel::LlamaCpp),
        state.panel_layout.rows_for(DashboardPanel::Webserver),
    );
    let [llama_panel, webserver_panel] = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(llama_panel_height),
            Constraint::Length(webserver_panel_height),
        ])
        .areas(processes_inner);
    let [llama_title, llama_body] = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(1), Constraint::Min(0)])
        .areas(llama_panel);
    let [webserver_title, webserver_body] = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(1), Constraint::Min(0)])
        .areas(webserver_panel);
    let [requests_title, requests_body] = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(1), Constraint::Min(0)])
        .areas(requests_band);
    TuiFrameAreas {
        loading,
        logo,
        join_token_panel,
        join_token_copy_button,
        main_body,
        requests: (requests_title, requests_body),
        status_bar: status_band,
        events: (events_title, events_body),
        processes: processes_column,
        llama_processes: (llama_title, llama_body),
        webserver_processes: (webserver_title, webserver_body),
        models: (models_title, models_body),
    }
}

fn tui_ready_logo_height(area: Rect) -> u16 {
    if area.height == 0 {
        return 0;
    }
    let desired = tui_ready_logo_text()
        .map(|text| u16::try_from(text.lines.len()).unwrap_or(u16::MAX))
        .unwrap_or_else(|| (area.height / 4).max(3));
    desired.min(area.height)
}

fn tui_ready_logo_width(area: Rect) -> u16 {
    if area.width == 0 {
        return 0;
    }
    tui_ready_logo_text()
        .map(|text| {
            text.lines
                .iter()
                .map(tui_logo_line_width)
                .max()
                .and_then(|width| u16::try_from(width).ok())
                .unwrap_or(area.width)
                .min(area.width)
        })
        .unwrap_or(area.width)
}

fn tui_centered_logo_area(area: Rect) -> Option<Rect> {
    let logo_width = tui_ready_logo_width(area);
    let logo_height = tui_ready_logo_height(area);
    if logo_width == 0 || logo_height == 0 {
        return None;
    }

    let [vertical] = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(logo_height)])
        .flex(Flex::Center)
        .areas(area);
    let [centered] = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Length(logo_width)])
        .flex(Flex::Center)
        .areas(vertical);
    Some(centered)
}

fn tui_desired_main_body_height(state: &DashboardState) -> u16 {
    u16::try_from(
        state
            .panel_layout
            .rows_for(DashboardPanel::Events)
            .saturating_add(2)
            .max(
                state
                    .panel_layout
                    .rows_for(DashboardPanel::Models)
                    .saturating_add(2),
            )
            .max(
                state
                    .panel_layout
                    .rows_for(DashboardPanel::LlamaCpp)
                    .saturating_add(state.panel_layout.rows_for(DashboardPanel::Webserver))
                    .saturating_add(5),
            ),
    )
    .unwrap_or(u16::MAX)
}

fn tui_desired_requests_band_height(state: &DashboardState) -> u16 {
    u16::try_from(
        state
            .panel_layout
            .rows_for(DashboardPanel::Requests)
            .saturating_add(2),
    )
    .unwrap_or(u16::MAX)
}

fn tui_band_heights(area: Rect, state: &DashboardState) -> TuiBandHeights {
    let status = area.height.min(1);
    let remaining_after_status = area.height.saturating_sub(status);
    let join_token = PRETTY_TUI_JOIN_TOKEN_PANEL_HEIGHT.min(remaining_after_status);
    let remaining_after_join_token = remaining_after_status.saturating_sub(join_token);
    let main_body_desired = tui_desired_main_body_height(state);
    let requests_desired = tui_desired_requests_band_height(state);
    let requests_min = remaining_after_join_token.min(5);
    let requests = requests_desired
        .min(remaining_after_join_token)
        .max(requests_min);
    let main_body = remaining_after_join_token
        .saturating_sub(requests)
        .min(main_body_desired);

    TuiBandHeights {
        join_token,
        main_body,
        requests,
        status,
    }
}

fn tui_process_panel_heights(
    available_height: u16,
    desired_llama_rows: usize,
    desired_webserver_rows: usize,
) -> (u16, u16) {
    if available_height == 0 {
        return (0, 0);
    }

    let desired_llama_block =
        u16::try_from(desired_llama_rows.saturating_add(2)).unwrap_or(u16::MAX);
    let desired_webserver_block =
        u16::try_from(desired_webserver_rows.saturating_add(2)).unwrap_or(u16::MAX);
    let desired_total = desired_llama_block.saturating_add(desired_webserver_block);

    if available_height == 1 {
        return (1, 0);
    }

    if desired_total == 0 {
        let llama_block = available_height / 2;
        return (llama_block, available_height.saturating_sub(llama_block));
    }

    let layout_height = available_height;
    let minimum_llama = 2.min(layout_height);
    let minimum_webserver = u16::from(layout_height > minimum_llama);
    let flexible_height = layout_height
        .saturating_sub(minimum_llama)
        .saturating_sub(minimum_webserver);
    let desired_flexible = desired_total
        .saturating_sub(minimum_llama)
        .saturating_sub(minimum_webserver);
    let llama_flexible = flexible_height
        .saturating_mul(desired_llama_block.saturating_sub(minimum_llama))
        .checked_div(desired_flexible)
        .unwrap_or(flexible_height / 2);
    let llama_block = minimum_llama + llama_flexible;
    let webserver_block = layout_height.saturating_sub(llama_block);

    (llama_block, webserver_block)
}

fn render_join_token_panel(
    frame: &mut Frame,
    state: &DashboardState,
    panel_area: Rect,
    copy_button_area: Rect,
) {
    if panel_area.width == 0 || panel_area.height == 0 {
        return;
    }

    let theme = tui_theme();
    let block = tui_panel_block(state, DashboardPanel::JoinToken).padding(Padding::horizontal(
        PRETTY_TUI_JOIN_TOKEN_HORIZONTAL_PADDING,
    ));
    let inner_area = block.inner(panel_area);
    frame.render_widget(block, panel_area);
    render_join_token_title_status(frame, state, panel_area);

    if inner_area.height == 0 || inner_area.width == 0 {
        return;
    }

    let token_area = join_token_text_area(panel_area, copy_button_area);

    let token_line = join_token_line(state, usize::from(token_area.width));
    frame.render_widget(
        Paragraph::new(token_line).style(Style::default().fg(theme.text)),
        token_area,
    );

    if copy_button_area.width > 0 && copy_button_area.height > 0 {
        let copy_enabled = state.join_token.is_some();
        let button_style = if copy_enabled {
            Style::default()
                .fg(theme.surface)
                .bg(theme.accent)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(theme.dim).bg(theme.surface_raised)
        };
        frame.render_widget(
            Paragraph::new(PRETTY_TUI_JOIN_TOKEN_COPY_BUTTON_LABEL)
                .style(button_style)
                .alignment(Alignment::Center),
            copy_button_area,
        );
    }
}

fn render_join_token_title_status(frame: &mut Frame, state: &DashboardState, panel_area: Rect) {
    if panel_area.width <= 4 || panel_area.height == 0 {
        return;
    }

    let theme = tui_theme();
    let left_title_width = format_tui_panel_title(state, DashboardPanel::JoinToken)
        .chars()
        .count();
    let max_status_width = usize::from(panel_area.width)
        .saturating_sub(left_title_width.saturating_add(5))
        .max(1);
    let status = truncate_with_ellipsis(&join_token_panel_right_title(state), max_status_width);
    let title = format!(" {status} ");
    let title_width = u16::try_from(title.chars().count())
        .unwrap_or(u16::MAX)
        .min(panel_area.width.saturating_sub(2));
    if title_width == 0 {
        return;
    }

    let title_area = Rect {
        x: panel_area
            .right()
            .saturating_sub(title_width)
            .saturating_sub(1),
        y: panel_area.y,
        width: title_width,
        height: 1,
    };
    frame.render_widget(
        Paragraph::new(Line::styled(
            title,
            Style::default()
                .fg(theme.muted)
                .bg(theme.surface_raised)
                .add_modifier(Modifier::BOLD),
        )),
        title_area,
    );
}

fn join_token_panel_left_title(state: &DashboardState, focus_marker: char) -> String {
    let mut title = format!("{focus_marker} Join Token");
    if let Some(join_token) = &state.join_token {
        title.push_str("  mesh=");
        title.push_str(&join_token.mesh_label());
    }
    title
}

fn join_token_panel_right_title(state: &DashboardState) -> String {
    let Some(join_token) = &state.join_token else {
        return "waiting for cluster invite".to_string();
    };
    match &join_token.copy_status {
        DashboardJoinTokenCopyStatus::Idle => "click Copy or c".to_string(),
        DashboardJoinTokenCopyStatus::Copied => "copied".to_string(),
        DashboardJoinTokenCopyStatus::Failed(message) => {
            format!("copy failed: {}", truncate_with_ellipsis(message, 40))
        }
    }
}

fn join_token_line(state: &DashboardState, width: usize) -> Line<'static> {
    let theme = tui_theme();
    if let Some(join_token) = &state.join_token {
        let token_width = width.saturating_sub(6).max(1);
        let scroll_offset = state
            .panel_view_state(DashboardPanel::JoinToken)
            .scroll_offset;
        Line::from(vec![
            Span::styled("token ", Style::default().fg(theme.muted)),
            Span::styled(
                join_token_visible_slice(&join_token.token, scroll_offset, token_width),
                Style::default().fg(theme.text).add_modifier(Modifier::BOLD),
            ),
        ])
    } else {
        Line::styled(
            "join token will appear here when the mesh invite is ready",
            Style::default().fg(theme.muted),
        )
    }
}

fn join_token_text_area(panel_area: Rect, copy_button_area: Rect) -> Rect {
    if panel_area.width == 0 || panel_area.height < 3 {
        return Rect {
            x: panel_area.x,
            y: panel_area.y,
            width: 0,
            height: 0,
        };
    }

    let inner_x = panel_area
        .x
        .saturating_add(1)
        .saturating_add(PRETTY_TUI_JOIN_TOKEN_HORIZONTAL_PADDING);
    let inner_y = panel_area.y.saturating_add(panel_area.height / 2);
    let inner_right = panel_area
        .right()
        .saturating_sub(1)
        .saturating_sub(PRETTY_TUI_JOIN_TOKEN_HORIZONTAL_PADDING);
    let token_right = if copy_button_area.width > 0 {
        copy_button_area.x.saturating_sub(1).min(inner_right)
    } else {
        inner_right
    };
    Rect {
        x: inner_x,
        y: inner_y,
        width: token_right.saturating_sub(inner_x),
        height: 1,
    }
}

fn join_token_content_width(panel_area: Rect, copy_button_area: Rect) -> u16 {
    join_token_text_area(panel_area, copy_button_area)
        .width
        .saturating_sub(6)
}

fn join_token_char_count(token: &str) -> usize {
    token.chars().count()
}

fn join_token_visible_slice(token: &str, scroll_offset: usize, width: usize) -> String {
    token.chars().skip(scroll_offset).take(width).collect()
}

fn tui_join_token_copy_button_area(panel_area: Rect) -> Rect {
    if panel_area.width == 0 || panel_area.height < 3 {
        return Rect {
            x: panel_area.x,
            y: panel_area.y,
            width: 0,
            height: 0,
        };
    }
    let button_width = u16::try_from(PRETTY_TUI_JOIN_TOKEN_COPY_BUTTON_LABEL.chars().count())
        .unwrap_or(u16::MAX)
        .saturating_add(2)
        .min(panel_area.width.saturating_sub(2));
    Rect {
        x: panel_area
            .right()
            .saturating_sub(button_width)
            .saturating_sub(1)
            .saturating_sub(PRETTY_TUI_JOIN_TOKEN_HORIZONTAL_PADDING),
        y: panel_area.y.saturating_add(panel_area.height / 2),
        width: button_width,
        height: 1,
    }
}

fn point_in_rect(column: u16, row: u16, rect: Rect) -> bool {
    rect.width > 0
        && rect.height > 0
        && column >= rect.left()
        && column < rect.right()
        && row >= rect.top()
        && row < rect.bottom()
}

fn copy_join_token_to_clipboard(token: &str) -> Result<(), String> {
    let mut clipboard = arboard::Clipboard::new().map_err(|err| err.to_string())?;
    clipboard
        .set_text(token.to_string())
        .map_err(|err| err.to_string())
}

fn render_requests_panel(
    frame: &mut Frame,
    state: &DashboardState,
    title_area: Rect,
    body_area: Rect,
) {
    let panel_area = combine_panel_rect(title_area, body_area);
    let block = tui_panel_block(state, DashboardPanel::Requests);
    frame.render_widget(block.clone(), panel_area);
    let inner_area = block.inner(panel_area);
    if inner_area.height == 0 {
        return;
    }

    let is_focused = state.panel_focus == DashboardPanel::Requests;
    let [summary_area, graph_slot] = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(1), Constraint::Min(0)])
        .areas(inner_area);

    frame.render_widget(
        Paragraph::new(tui_requests_summary_line(
            &state.request_history,
            state.request_window,
        ))
        .style(if is_focused {
            Style::default().add_modifier(Modifier::BOLD)
        } else {
            Style::default()
        }),
        summary_area,
    );

    if graph_slot.width == 0 || graph_slot.height == 0 {
        return;
    }

    let chart_spec = tui_request_chart_spec(
        &state.request_history,
        state.request_window,
        graph_slot.width,
    );
    frame.render_widget(
        TuiRequestChartWidget {
            chart_spec,
            is_focused,
        },
        graph_slot,
    );
}

#[derive(Clone, Copy, Default)]
struct TuiModelGaugeScale {
    max_file_size_gb: f64,
}

fn render_models_panel(
    frame: &mut Frame,
    state: &DashboardState,
    title_area: Rect,
    body_area: Rect,
) {
    let panel_area = combine_panel_rect(title_area, body_area);
    let block = tui_panel_block(state, DashboardPanel::Models);
    frame.render_widget(block.clone(), panel_area);
    let inner_area = block.inner(panel_area);
    if inner_area.height == 0 {
        return;
    }

    if state.loaded_model_rows.is_empty() {
        frame.render_widget(
            Paragraph::new(empty_panel_message(state, DashboardPanel::Models))
                .style(Style::default().fg(Color::DarkGray)),
            inner_area,
        );
        return;
    }

    let view = state.panel_view_state(DashboardPanel::Models);
    let is_focused = state.panel_focus == DashboardPanel::Models;
    let viewport_rows =
        tui_panel_viewport_rows(DashboardPanel::Models, usize::from(inner_area.height));
    let row_count = state.row_count_for_panel(DashboardPanel::Models);
    let show_scrollbar = row_count > viewport_rows && inner_area.width > 1;
    let list_area = if show_scrollbar {
        Rect {
            width: inner_area.width.saturating_sub(1),
            ..inner_area
        }
    } else {
        inner_area
    };
    let content_width = usize::from(list_area.width.max(1));
    let scale = tui_model_gauge_scale(&state.loaded_model_rows);
    for (local_index, (row_index, row)) in state
        .loaded_model_rows
        .iter()
        .enumerate()
        .skip(view.scroll_offset)
        .take(viewport_rows)
        .enumerate()
    {
        let card_y = list_area.y.saturating_add(
            u16::try_from(local_index.saturating_mul(PRETTY_TUI_MODEL_CARD_HEIGHT)).unwrap_or(0),
        );
        if card_y >= list_area.bottom() {
            break;
        }

        let row_area = Rect {
            x: list_area.x,
            y: card_y,
            width: list_area.width,
            height: PRETTY_TUI_MODEL_CARD_HEIGHT as u16,
        };
        let is_selected = view.selected_row == Some(row_index);

        frame.render_widget(
            TuiModelCardWidget {
                row,
                scale,
                content_width,
                is_selected,
                is_focused,
            },
            row_area,
        );
    }

    if show_scrollbar {
        let scrollbar_area = Rect {
            x: inner_area.right().saturating_sub(1),
            y: inner_area.y,
            width: 1,
            height: inner_area.height,
        };
        let mut scrollbar_state = ScrollbarState::new(row_count)
            .position(view.scroll_offset)
            .viewport_content_length(viewport_rows.min(row_count));
        let scrollbar = Scrollbar::new(ScrollbarOrientation::VerticalRight)
            .begin_symbol(None)
            .end_symbol(None)
            .track_symbol(Some("│"));
        frame.render_stateful_widget(scrollbar, scrollbar_area, &mut scrollbar_state);
    }
}

fn tui_panel_viewport_rows(panel: DashboardPanel, visible_rows: usize) -> usize {
    match panel {
        DashboardPanel::Models => tui_models_viewport_rows(visible_rows as u16),
        _ => visible_rows.max(1),
    }
}

fn tui_models_viewport_rows(visible_height: u16) -> usize {
    let visible_height = usize::from(visible_height);
    if visible_height == 0 {
        return 0;
    }
    (visible_height / PRETTY_TUI_MODEL_CARD_HEIGHT).max(1)
}

struct TuiModelCardWidget<'a> {
    row: &'a DashboardModelRow,
    scale: TuiModelGaugeScale,
    content_width: usize,
    is_selected: bool,
    is_focused: bool,
}

impl Widget for TuiModelCardWidget<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        if area.width == 0 || area.height == 0 {
            return;
        }

        let theme = tui_theme();
        let card_bg = if self.is_selected {
            theme.selection_bg
        } else {
            theme.surface_raised
        };
        let border_fg = if self.is_selected && self.is_focused {
            theme.accent
        } else {
            theme.dim
        };
        let block = Block::bordered()
            .border_type(BorderType::Rounded)
            .style(Style::default().bg(card_bg))
            .border_style(Style::default().fg(border_fg).bg(card_bg))
            .title(Line::styled(
                truncate_with_ellipsis(&self.row.name, self.content_width.saturating_sub(6).max(1)),
                Style::default()
                    .fg(theme.text)
                    .bg(card_bg)
                    .add_modifier(Modifier::BOLD),
            ));
        let inner = block.inner(area);
        block.render(area, buf);
        if inner.height == 0 || inner.width == 0 {
            return;
        }

        let [summary_top, summary_bottom, divider, ctx_row, file_row] = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(1),
                Constraint::Length(1),
                Constraint::Length(1),
                Constraint::Length(1),
                Constraint::Length(1),
            ])
            .areas(inner);

        render_tui_model_summary_cells(
            buf,
            summary_top,
            card_bg,
            vec![
                (
                    "PORT",
                    self.row
                        .port
                        .map(|port| port.to_string())
                        .unwrap_or_else(|| "n/a".to_string()),
                    Style::default().fg(theme.text).bg(card_bg),
                ),
                (
                    "DEVICE",
                    "n/a".to_string(),
                    Style::default().fg(theme.text).bg(card_bg),
                ),
                (
                    "STATUS",
                    self.row.status.as_str().to_string(),
                    tui_model_status_style(&self.row.status).bg(card_bg),
                ),
            ],
        );

        render_tui_model_summary_cells(
            buf,
            summary_bottom,
            card_bg,
            vec![
                (
                    "SLOTS",
                    self.row
                        .slots
                        .map(|slots| slots.to_string())
                        .unwrap_or_else(|| "n/a".to_string()),
                    Style::default().fg(theme.warning).bg(card_bg),
                ),
                (
                    "QUANT",
                    self.row
                        .quantization
                        .as_deref()
                        .unwrap_or("n/a")
                        .to_string(),
                    Style::default().fg(theme.text).bg(card_bg),
                ),
                (
                    "CTX",
                    "n/a".to_string(),
                    Style::default().fg(theme.text).bg(card_bg),
                ),
            ],
        );

        Paragraph::new(tui_model_card_divider(usize::from(inner.width)))
            .style(Style::default().fg(theme.dim).bg(card_bg))
            .render(divider, buf);

        let ctx_value = "n/a".to_string();
        let ctx_max = "n/a".to_string();
        let file_value = self
            .row
            .file_size_gb
            .map(|file_size_gb| format!("{file_size_gb:.1}"))
            .unwrap_or_else(|| "n/a".to_string());
        let file_max = if self.scale.max_file_size_gb > 0.0 {
            format!("{:.1} GB", self.scale.max_file_size_gb)
        } else {
            "n/a".to_string()
        };

        render_tui_model_metric_row(
            buf,
            ctx_row,
            card_bg,
            "CTX",
            format!("{ctx_value} / {ctx_max}"),
            0.0,
        );
        render_tui_model_metric_row(
            buf,
            file_row,
            card_bg,
            "FILE",
            format!("{file_value} / {file_max}"),
            tui_model_gauge_ratio(self.row.file_size_gb, self.scale.max_file_size_gb),
        );
    }
}

fn render_tui_model_metric_row(
    buf: &mut Buffer,
    area: Rect,
    card_bg: Color,
    label: &'static str,
    value_label: String,
    ratio: f64,
) {
    if area.width == 0 || area.height == 0 {
        return;
    }

    let theme = tui_theme();
    let value_width = u16::try_from(value_label.chars().count().clamp(8, 20)).unwrap_or(20);
    let [label_area, bar_area, value_area] = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Length(5),
            Constraint::Min(1),
            Constraint::Length(value_width),
        ])
        .areas(area);

    Paragraph::new(label)
        .style(
            Style::default()
                .fg(theme.muted)
                .bg(card_bg)
                .add_modifier(Modifier::BOLD),
        )
        .render(label_area, buf);
    render_tui_model_usage_bar(buf, bar_area, card_bg, ratio);
    Paragraph::new(truncate_with_ellipsis(
        &value_label,
        usize::from(value_area.width),
    ))
    .style(Style::default().fg(theme.text).bg(card_bg))
    .alignment(Alignment::Right)
    .render(value_area, buf);
}

fn render_tui_model_summary_cells(
    buf: &mut Buffer,
    area: Rect,
    card_bg: Color,
    entries: Vec<(&'static str, String, Style)>,
) {
    if area.width == 0 || area.height == 0 || entries.is_empty() {
        return;
    }

    let columns = entries.len();
    let gap_width = u16::from(columns > 1);
    let mut constraints = Vec::with_capacity(columns.saturating_mul(2).saturating_sub(1));
    for index in 0..columns {
        constraints.push(Constraint::Fill(1));
        if index + 1 < columns {
            constraints.push(Constraint::Length(gap_width));
        }
    }
    let cells = Layout::default()
        .direction(Direction::Horizontal)
        .constraints(constraints)
        .split(area);

    for (index, (label, value, value_style)) in entries.into_iter().enumerate() {
        let cell_index = index.saturating_mul(2);
        let Some(cell_area) = cells.get(cell_index).copied() else {
            continue;
        };
        if cell_area.width == 0 {
            continue;
        }

        let label_text = format!("{label}: ");
        let label_width = label_text.chars().count();
        let value_width = usize::from(cell_area.width)
            .saturating_sub(label_width)
            .max(1);
        let value = truncate_with_ellipsis(&value, value_width);
        let line = Line::from(vec![
            Span::styled(
                label_text,
                Style::default()
                    .fg(tui_theme().dim)
                    .bg(card_bg)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(value, value_style.bg(card_bg)),
        ]);
        Paragraph::new(line)
            .style(Style::default().bg(card_bg))
            .alignment(Alignment::Left)
            .render(cell_area, buf);
    }
}

fn render_tui_model_usage_bar(buf: &mut Buffer, area: Rect, card_bg: Color, ratio: f64) {
    if area.width == 0 || area.height == 0 {
        return;
    }

    let theme = tui_theme();
    let ratio = ratio.clamp(0.0, 1.0);
    let filled_width = (ratio * f64::from(area.width)).round() as u16;
    let fill_color = tui_model_usage_color(ratio);
    let empty_style = Style::default().fg(theme.dim).bg(card_bg);
    let fill_style = Style::default().fg(fill_color).bg(card_bg);

    for y in area.top()..area.bottom() {
        for x in area.left()..area.right() {
            let filled = x.saturating_sub(area.left()) < filled_width;
            buf[(x, y)]
                .set_symbol("█")
                .set_style(if filled { fill_style } else { empty_style });
        }
    }
}

fn tui_model_usage_color(ratio: f64) -> Color {
    let theme = tui_theme();
    let ratio = ratio.clamp(0.0, 1.0);
    if ratio <= 0.5 {
        tui_lerp_rgb(theme.success, theme.warning, ratio / 0.5)
    } else {
        tui_lerp_rgb(theme.warning, theme.error, (ratio - 0.5) / 0.5)
    }
}

fn tui_lerp_rgb(start: Color, end: Color, t: f64) -> Color {
    let Color::Rgb(start_r, start_g, start_b) = start else {
        return end;
    };
    let Color::Rgb(end_r, end_g, end_b) = end else {
        return start;
    };
    let t = t.clamp(0.0, 1.0);
    Color::Rgb(
        (f64::from(start_r) + (f64::from(end_r) - f64::from(start_r)) * t).round() as u8,
        (f64::from(start_g) + (f64::from(end_g) - f64::from(start_g)) * t).round() as u8,
        (f64::from(start_b) + (f64::from(end_b) - f64::from(start_b)) * t).round() as u8,
    )
}

fn normalize_request_buckets(
    buckets: &[DashboardAcceptedRequestBucket],
) -> Vec<DashboardAcceptedRequestBucket> {
    let mut counts_by_offset = vec![0_u64; PRETTY_DASHBOARD_REQUEST_MAX_WINDOW_SECS as usize];
    for bucket in buckets {
        let offset = bucket.second_offset as usize;
        if offset < counts_by_offset.len() {
            counts_by_offset[offset] = bucket.accepted_count;
        }
    }

    (0..PRETTY_DASHBOARD_REQUEST_MAX_WINDOW_SECS as usize)
        .map(|index| {
            let second_offset =
                (PRETTY_DASHBOARD_REQUEST_MAX_WINDOW_SECS as usize - 1 - index) as u32;
            DashboardAcceptedRequestBucket {
                second_offset,
                accepted_count: counts_by_offset[second_offset as usize],
            }
        })
        .collect()
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct TuiRequestChartSpec {
    bucket_values: Vec<u64>,
    bar_width: u16,
    bar_gap: u16,
    visible_bucket_start: usize,
    visible_bucket_count: usize,
    scale_max: u64,
    scale_width: u16,
}

struct TuiRequestChartWidget {
    chart_spec: TuiRequestChartSpec,
    is_focused: bool,
}

impl Widget for TuiRequestChartWidget {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let (scale_area, plot_area) = tui_request_chart_areas(area, &self.chart_spec);
        tui_clear_request_chart_area(area, buf);
        tui_render_request_chart_guides(plot_area, buf, self.is_focused);
        tui_render_request_scale(scale_area, buf, &self.chart_spec, self.is_focused);
        tui_render_request_chart_braille(plot_area, buf, &self.chart_spec, self.is_focused);
    }
}

fn tui_current_rps(history: &DashboardRequestHistoryState) -> u64 {
    history
        .accepted_request_buckets
        .last()
        .map(|bucket| bucket.accepted_count)
        .unwrap_or(0)
}

fn tui_requests_summary_line(
    history: &DashboardRequestHistoryState,
    request_window: DashboardRequestWindow,
) -> Line<'static> {
    let label_style = Style::default().fg(Color::DarkGray);
    let value_style = Style::default()
        .fg(Color::White)
        .add_modifier(Modifier::BOLD);
    let p50 = tui_p50_latency_ms(&history.latency_samples_ms)
        .map(|latency_ms| format!("{latency_ms}ms"))
        .unwrap_or_else(|| "n/a".to_string());

    Line::from(vec![
        Span::styled("RPS ", label_style),
        Span::styled(tui_current_rps(history).to_string(), value_style),
        Span::raw("  "),
        Span::styled("inflight ", label_style),
        Span::styled(history.current_inflight_requests.to_string(), value_style),
        Span::raw("  "),
        Span::styled("p50 ", label_style),
        Span::styled(p50, value_style),
        Span::raw("  "),
        Span::styled("window ", label_style),
        Span::styled(request_window.label(), value_style),
        Span::raw("  "),
        Span::styled(request_window.bucket_label(), label_style),
    ])
}

fn tui_request_chart_spec(
    history: &DashboardRequestHistoryState,
    request_window: DashboardRequestWindow,
    graph_width: u16,
) -> TuiRequestChartSpec {
    let mut bucket_values = vec![0_u64; PRETTY_DASHBOARD_REQUEST_WINDOW_BUCKETS];
    let bucket_seconds = request_window.bucket_seconds().max(1);
    let window_seconds = request_window.seconds();
    for bucket in &history.accepted_request_buckets {
        if bucket.second_offset >= window_seconds {
            continue;
        }
        let age_bucket = bucket.second_offset / bucket_seconds;
        let Some(visual_index) =
            PRETTY_DASHBOARD_REQUEST_WINDOW_BUCKETS.checked_sub(1 + age_bucket as usize)
        else {
            continue;
        };
        if let Some(value) = bucket_values.get_mut(visual_index) {
            *value += bucket.accepted_count;
        }
    }
    let max_bucket_value = bucket_values.iter().copied().max().unwrap_or(0);
    let scale_max = tui_request_scale_ceiling(max_bucket_value);
    let scale_width = tui_request_scale_width(scale_max, graph_width);
    let plot_width = graph_width.saturating_sub(scale_width).max(1);
    let bucket_count = u16::try_from(bucket_values.len())
        .unwrap_or(u16::MAX)
        .max(1);
    let base_bar_width = if plot_width >= bucket_count {
        (plot_width / bucket_count).max(1)
    } else {
        1
    };
    let bar_width = request_window
        .bar_width_cap()
        .map(|cap| base_bar_width.min(cap))
        .unwrap_or(base_bar_width)
        .max(1);
    let remaining_width = plot_width.saturating_sub(bucket_count.saturating_mul(bar_width));
    let bar_gap = if bucket_count > 1 {
        request_window
            .preferred_bar_gap()
            .min(remaining_width / bucket_count.saturating_sub(1))
    } else {
        0
    };
    let slot_width = bar_width.saturating_add(bar_gap).max(1);
    let visible_bucket_count = usize::from(
        plot_width
            .saturating_add(bar_gap)
            .checked_div(slot_width)
            .unwrap_or(0)
            .max(1),
    )
    .min(bucket_values.len());
    TuiRequestChartSpec {
        bucket_values,
        bar_width,
        bar_gap,
        visible_bucket_start: PRETTY_DASHBOARD_REQUEST_WINDOW_BUCKETS
            .saturating_sub(visible_bucket_count),
        visible_bucket_count,
        scale_max,
        scale_width,
    }
}

fn tui_request_scale_ceiling(max_bucket_value: u64) -> u64 {
    let headroom = max_bucket_value / 5 + 1;
    tui_nice_request_scale(max_bucket_value.saturating_add(headroom))
}

fn tui_nice_request_scale(value: u64) -> u64 {
    let value = value.max(1);
    let mut magnitude = 1_u64;
    while magnitude.saturating_mul(10) <= value {
        magnitude = magnitude.saturating_mul(10);
    }

    for multiplier in [1_u64, 2, 5, 10] {
        let candidate = magnitude.saturating_mul(multiplier);
        if candidate >= value {
            return candidate;
        }
    }
    magnitude.saturating_mul(10)
}

fn tui_request_scale_width(scale_max: u64, graph_width: u16) -> u16 {
    if graph_width < 12 {
        return 0;
    }

    let label_width = u16::try_from(scale_max.to_string().chars().count())
        .unwrap_or(u16::MAX)
        .max(2);
    label_width
        .saturating_add(1)
        .min(graph_width.saturating_sub(1))
}

fn tui_request_chart_areas(area: Rect, chart_spec: &TuiRequestChartSpec) -> (Rect, Rect) {
    let scale_width = chart_spec.scale_width.min(area.width.saturating_sub(1));
    let scale_area = Rect {
        width: scale_width,
        ..area
    };
    let plot_area = Rect {
        x: area.x.saturating_add(scale_width),
        width: area.width.saturating_sub(scale_width),
        ..area
    };
    (scale_area, plot_area)
}

fn tui_clear_request_chart_area(area: Rect, buf: &mut Buffer) {
    let theme = tui_theme();
    let clear_style = Style::default().bg(theme.surface);
    for y in area.top()..area.bottom() {
        for x in area.left()..area.right() {
            buf[(x, y)].set_symbol(" ").set_style(clear_style);
        }
    }
}

fn tui_render_request_chart_braille(
    area: Rect,
    buf: &mut Buffer,
    chart_spec: &TuiRequestChartSpec,
    is_focused: bool,
) {
    if area.width == 0 || area.height == 0 || chart_spec.bucket_values.is_empty() {
        return;
    }

    let current_bar_style = Style::default().fg(if is_focused {
        Color::Cyan
    } else {
        Color::Rgb(70, 170, 220)
    });
    let history_bar_style = Style::default().fg(if is_focused {
        Color::Rgb(82, 150, 220)
    } else {
        Color::Rgb(70, 110, 170)
    });
    let vertical_units = u64::from(area.height.max(1)).saturating_mul(4);
    let visible_bucket_count = chart_spec
        .visible_bucket_count
        .min(chart_spec.bucket_values.len());
    let rendered_width = u16::try_from(visible_bucket_count)
        .unwrap_or(u16::MAX)
        .saturating_mul(chart_spec.bar_width)
        .saturating_add(
            u16::try_from(visible_bucket_count.saturating_sub(1))
                .unwrap_or(u16::MAX)
                .saturating_mul(chart_spec.bar_gap),
        );
    let x_origin = area.right().saturating_sub(rendered_width);

    for (visible_index, (index, value)) in chart_spec
        .bucket_values
        .iter()
        .enumerate()
        .skip(chart_spec.visible_bucket_start)
        .take(visible_bucket_count)
        .enumerate()
    {
        if *value == 0 {
            continue;
        }
        let filled_units = value
            .saturating_mul(vertical_units)
            .div_ceil(chart_spec.scale_max.max(1))
            .clamp(1, vertical_units);
        let Ok(visible_index_u16) = u16::try_from(visible_index) else {
            continue;
        };
        let x_start = x_origin.saturating_add(
            visible_index_u16
                .saturating_mul(chart_spec.bar_width.saturating_add(chart_spec.bar_gap)),
        );
        let style = if index + 1 == chart_spec.bucket_values.len() {
            current_bar_style
        } else {
            history_bar_style
        };

        for x_offset in 0..chart_spec.bar_width {
            let x = x_start.saturating_add(x_offset);
            if x >= area.right() {
                continue;
            }
            for row in 0..area.height {
                let y = area.bottom().saturating_sub(1 + row);
                if y < area.top() {
                    continue;
                }
                let cell_base_units = u64::from(row).saturating_mul(4);
                let filled_in_cell = filled_units.saturating_sub(cell_base_units).min(4) as u8;
                if filled_in_cell == 0 {
                    continue;
                }
                let symbol = tui_braille_bar_symbol(filled_in_cell, filled_in_cell);
                let symbol = symbol.to_string();
                buf[(x, y)].set_symbol(&symbol).set_style(style);
            }
        }
    }
}

fn tui_render_request_scale(
    area: Rect,
    buf: &mut Buffer,
    chart_spec: &TuiRequestChartSpec,
    is_focused: bool,
) {
    if area.width == 0 || area.height == 0 {
        return;
    }

    let theme = tui_theme();
    let style = Style::default()
        .fg(if is_focused { theme.muted } else { theme.dim })
        .add_modifier(Modifier::DIM);
    let labels = tui_request_scale_labels(area.height, chart_spec.scale_max);

    for (row, value) in labels {
        let y = area
            .y
            .saturating_add(row)
            .min(area.bottom().saturating_sub(1));
        let label = value.to_string();
        let label_width = u16::try_from(label.chars().count()).unwrap_or(u16::MAX);
        let x = area
            .right()
            .saturating_sub(1)
            .saturating_sub(label_width)
            .max(area.x);
        for (offset, ch) in label.chars().enumerate() {
            let Ok(offset) = u16::try_from(offset) else {
                continue;
            };
            let x = x.saturating_add(offset);
            if x >= area.right() {
                continue;
            }
            let symbol = ch.to_string();
            buf[(x, y)].set_symbol(&symbol).set_style(style);
        }
    }
}

fn tui_request_scale_labels(height: u16, scale_max: u64) -> Vec<(u16, u64)> {
    if height == 0 {
        return Vec::new();
    }

    let mut labels = vec![(0_u16, scale_max)];
    if height > 2 && scale_max > 1 {
        labels.push((height / 2, scale_max / 2));
    }
    if height > 1 {
        labels.push((height.saturating_sub(1), 0));
    }
    labels
}

fn tui_braille_bar_symbol(left_filled_dots: u8, right_filled_dots: u8) -> char {
    const LEFT_BOTTOM_TO_TOP: [u8; 4] = [0x40, 0x04, 0x02, 0x01];
    const RIGHT_BOTTOM_TO_TOP: [u8; 4] = [0x80, 0x20, 0x10, 0x08];

    let mut mask = 0_u32;
    for dot in LEFT_BOTTOM_TO_TOP
        .iter()
        .take(usize::from(left_filled_dots.min(4)))
    {
        mask |= u32::from(*dot);
    }
    for dot in RIGHT_BOTTOM_TO_TOP
        .iter()
        .take(usize::from(right_filled_dots.min(4)))
    {
        mask |= u32::from(*dot);
    }

    char::from_u32(0x2800 + mask).unwrap_or(' ')
}

fn tui_render_request_chart_guides(area: Rect, buf: &mut Buffer, is_focused: bool) {
    if area.width == 0 || area.height == 0 {
        return;
    }

    let guide_style = Style::default().fg(if is_focused {
        Color::Rgb(34, 38, 45)
    } else {
        Color::Rgb(26, 30, 36)
    });
    let baseline_style = Style::default().fg(if is_focused {
        Color::Rgb(42, 48, 56)
    } else {
        Color::Rgb(32, 36, 44)
    });

    for y in area.top()..area.bottom() {
        let is_baseline = y + 1 == area.bottom();
        for x in area.left()..area.right() {
            let cell = &mut buf[(x, y)];
            if cell.symbol() != " " {
                continue;
            }

            if is_baseline {
                cell.set_symbol(PRETTY_TUI_REQUEST_GRAPH_BASELINE_SYMBOL)
                    .set_style(baseline_style);
            } else if (x - area.left() + y - area.top()).is_multiple_of(4) {
                cell.set_symbol(PRETTY_TUI_REQUEST_GRAPH_GUIDE_SYMBOL)
                    .set_style(guide_style);
            }
        }
    }
}

fn tui_p50_latency_ms(samples_ms: &[u64]) -> Option<u64> {
    if samples_ms.is_empty() {
        return None;
    }

    let mut sorted = samples_ms.to_vec();
    sorted.sort_unstable();
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 1 {
        Some(sorted[mid])
    } else {
        Some((sorted[mid - 1] + sorted[mid]) / 2)
    }
}

fn tui_model_gauge_scale(rows: &[DashboardModelRow]) -> TuiModelGaugeScale {
    TuiModelGaugeScale {
        max_file_size_gb: rows
            .iter()
            .filter_map(|row| row.file_size_gb)
            .fold(0.0, f64::max),
    }
}

fn tui_model_card_divider(content_width: usize) -> Line<'static> {
    let theme = tui_theme();
    Line::from(Span::styled(
        "─".repeat(content_width),
        Style::default().fg(theme.dim).add_modifier(Modifier::DIM),
    ))
}

#[cfg(test)]
fn spans_plain_text(spans: &[Span<'_>]) -> String {
    let mut text = String::new();
    for span in spans {
        text.push_str(span.content.as_ref());
    }
    text
}

fn tui_model_gauge_ratio(value: Option<f64>, max_value: f64) -> f64 {
    let Some(value) = value.filter(|value| *value > 0.0) else {
        return 0.0;
    };
    if max_value <= 0.0 {
        return 0.0;
    }
    (value / max_value).clamp(0.0, 1.0)
}

fn tui_model_status_style(status: &RuntimeStatus) -> Style {
    match status {
        RuntimeStatus::Starting => Style::default().fg(Color::Yellow),
        RuntimeStatus::Ready => Style::default().fg(Color::Green),
        RuntimeStatus::ShuttingDown => Style::default().fg(Color::Yellow),
        RuntimeStatus::Stopped => Style::default().fg(Color::DarkGray),
        RuntimeStatus::Exited => Style::default().fg(Color::DarkGray),
        RuntimeStatus::Warning => Style::default().fg(Color::Yellow),
        RuntimeStatus::Error => Style::default().fg(Color::Red),
    }
}

fn render_processes_panel(
    frame: &mut Frame,
    state: &DashboardState,
    processes_area: Rect,
    llama_processes: (Rect, Rect),
    webserver_processes: (Rect, Rect),
) {
    frame.render_widget(tui_processes_block(state), processes_area);
    render_process_table(
        frame,
        state,
        DashboardPanel::LlamaCpp,
        llama_processes.0,
        llama_processes.1,
    );
    render_process_table(
        frame,
        state,
        DashboardPanel::Webserver,
        webserver_processes.0,
        webserver_processes.1,
    );
}

fn render_process_table(
    frame: &mut Frame,
    state: &DashboardState,
    panel: DashboardPanel,
    title_area: Rect,
    body_area: Rect,
) {
    let panel_area = combine_panel_rect(title_area, body_area);
    let block = tui_panel_block(state, panel);
    frame.render_widget(block.clone(), panel_area);
    let inner_area = block.inner(panel_area);
    if inner_area.height == 0 {
        return;
    }

    let view = state.panel_view_state(panel);
    let is_focused = state.panel_focus == panel;
    match panel {
        DashboardPanel::LlamaCpp => {
            if state.llama_process_rows.is_empty() {
                frame.render_widget(
                    Paragraph::new(empty_panel_message(state, panel))
                        .style(Style::default().fg(Color::DarkGray)),
                    inner_area,
                );
                return;
            }

            let [model_width, pid_width, port_width, status_width] =
                llama_process_column_widths(inner_area.width);
            let available_rows = usize::from(inner_area.height.saturating_sub(1));
            let rows = state
                .llama_process_rows
                .iter()
                .enumerate()
                .skip(view.scroll_offset)
                .take(available_rows)
                .map(|(_, row)| {
                    let model = llama_process_model_metadata(row, &state.loaded_model_rows);
                    Row::new(vec![
                        Cell::from(truncate_with_ellipsis(
                            model.map(|model| model.name.as_str()).unwrap_or(&row.name),
                            model_width,
                        )),
                        Cell::from(truncate_with_ellipsis(&row.pid.to_string(), pid_width)),
                        Cell::from(truncate_with_ellipsis(&row.port.to_string(), port_width)),
                        process_status_cell(row.status.as_str(), status_width),
                    ])
                })
                .collect::<Vec<_>>();
            let selected_local_index = view
                .selected_row
                .map(|selected| selected.saturating_sub(view.scroll_offset));
            let mut table_state = TableState::default();
            table_state.select(selected_local_index);
            let table = Table::new(
                rows,
                [
                    Constraint::Fill(1),
                    Constraint::Length(u16::try_from(pid_width).unwrap_or(u16::MAX)),
                    Constraint::Length(u16::try_from(port_width).unwrap_or(u16::MAX)),
                    Constraint::Length(u16::try_from(status_width).unwrap_or(u16::MAX)),
                ],
            )
            .header(process_table_header_row([
                "MODEL".to_string(),
                "PID".to_string(),
                "PORT".to_string(),
                right_align_text("STATE", status_width),
            ]))
            .column_spacing(1)
            .highlight_symbol(if is_focused { "› " } else { "  " })
            .highlight_spacing(HighlightSpacing::Always)
            .row_highlight_style(process_table_highlight_style(is_focused));
            frame.render_stateful_widget(table, inner_area, &mut table_state);
        }
        DashboardPanel::Webserver => {
            if state.webserver_rows.is_empty() {
                frame.render_widget(
                    Paragraph::new(empty_panel_message(state, panel))
                        .style(Style::default().fg(Color::DarkGray)),
                    inner_area,
                );
                return;
            }

            let [label_width, pid_width, port_width, status_width] =
                webserver_process_column_widths(inner_area.width);
            let available_rows = usize::from(inner_area.height.saturating_sub(1));
            let rows = state
                .webserver_rows
                .iter()
                .enumerate()
                .skip(view.scroll_offset)
                .take(available_rows)
                .map(|(_, row)| {
                    Row::new(vec![
                        Cell::from(truncate_with_ellipsis(&row.label, label_width)),
                        Cell::from(truncate_with_ellipsis(
                            &format_dashboard_pid(row.pid),
                            pid_width,
                        )),
                        Cell::from(truncate_with_ellipsis(
                            &format_dashboard_port(row.port),
                            port_width,
                        )),
                        process_status_cell(row.status.as_str(), status_width),
                    ])
                })
                .collect::<Vec<_>>();
            let selected_local_index = view
                .selected_row
                .map(|selected| selected.saturating_sub(view.scroll_offset));
            let mut table_state = TableState::default();
            table_state.select(selected_local_index);
            let table = Table::new(
                rows,
                [
                    Constraint::Fill(1),
                    Constraint::Length(u16::try_from(pid_width).unwrap_or(u16::MAX)),
                    Constraint::Length(u16::try_from(port_width).unwrap_or(u16::MAX)),
                    Constraint::Length(u16::try_from(status_width).unwrap_or(u16::MAX)),
                ],
            )
            .header(process_table_header_row([
                PRETTY_TUI_WEBSERVER_PROCESS_HEADER_LABEL.to_string(),
                "PID".to_string(),
                "PORT".to_string(),
                right_align_text("STATE", status_width),
            ]))
            .column_spacing(1)
            .highlight_symbol(if is_focused { "› " } else { "  " })
            .highlight_spacing(HighlightSpacing::Always)
            .row_highlight_style(process_table_highlight_style(is_focused));
            frame.render_stateful_widget(table, inner_area, &mut table_state);
        }
        _ => {}
    }
}

fn combine_panel_rect(title_area: Rect, body_area: Rect) -> Rect {
    Rect {
        x: title_area.x,
        y: title_area.y,
        width: title_area.width.max(body_area.width),
        height: title_area.height.saturating_add(body_area.height),
    }
}

fn tui_panel_block(state: &DashboardState, panel: DashboardPanel) -> Block<'static> {
    Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(panel_border_style(state, panel))
        .title(Line::styled(
            format_tui_panel_title(state, panel),
            panel_title_style(state, panel),
        ))
}

fn tui_processes_block(state: &DashboardState) -> Block<'static> {
    Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(processes_border_style(state))
        .title(Line::styled(" Processes", processes_title_style(state)))
}

fn processes_title_style(state: &DashboardState) -> Style {
    let theme = tui_theme();
    if matches!(
        state.panel_focus,
        DashboardPanel::LlamaCpp | DashboardPanel::Webserver
    ) {
        Style::default()
            .fg(theme.accent)
            .bg(theme.surface_raised)
            .add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(theme.dim).add_modifier(Modifier::DIM)
    }
}

fn processes_border_style(state: &DashboardState) -> Style {
    let theme = tui_theme();
    if matches!(
        state.panel_focus,
        DashboardPanel::LlamaCpp | DashboardPanel::Webserver
    ) {
        Style::default().fg(theme.accent)
    } else {
        Style::default().fg(theme.dim)
    }
}

fn process_table_highlight_style(is_focused: bool) -> Style {
    let theme = tui_theme();
    if is_focused {
        Style::default()
            .bg(theme.selection_bg)
            .add_modifier(Modifier::BOLD)
    } else {
        Style::default()
    }
}

fn process_table_header_row<const N: usize>(labels: [String; N]) -> Row<'static> {
    let theme = tui_theme();
    Row::new(labels.into_iter().map(|label| {
        Cell::from(label).style(
            Style::default()
                .fg(theme.muted)
                .add_modifier(Modifier::BOLD),
        )
    }))
    .style(Style::default().bg(theme.surface_raised))
}

fn right_align_text(value: &str, width: usize) -> String {
    let value = truncate_with_ellipsis(value, width);
    format!("{value:>width$}")
}

fn format_dashboard_pid(pid: Option<u32>) -> String {
    pid.map(|pid| pid.to_string())
        .unwrap_or_else(|| "-".to_string())
}

fn format_dashboard_port(port: u16) -> String {
    if port == 0 {
        "-".to_string()
    } else {
        port.to_string()
    }
}

fn process_status_cell(status: &str, width: usize) -> Cell<'static> {
    let theme = tui_theme();
    let style = match status {
        "ready" => Style::default().fg(theme.success),
        "starting" => Style::default().fg(theme.warning),
        "shutting down" => Style::default().fg(theme.warning),
        "warning" => Style::default().fg(theme.warning),
        "error" => Style::default().fg(theme.error),
        "stopped" => Style::default().fg(theme.dim),
        "exited" => Style::default().fg(theme.dim),
        _ => Style::default().fg(theme.text),
    };
    Cell::from(right_align_text(status, width)).style(style)
}

fn llama_process_model_metadata<'a>(
    process: &DashboardProcessRow,
    models: &'a [DashboardModelRow],
) -> Option<&'a DashboardModelRow> {
    models
        .iter()
        .find(|model| model.port == Some(process.port))
        .or_else(|| {
            models.iter().find(|model| {
                process.name.contains(&model.name) || model.name.contains(&process.name)
            })
        })
}

fn llama_process_column_widths(body_width: u16) -> [usize; 4] {
    let status_width = 5usize;
    let port_width = 5usize;
    let pid_width = 5usize;
    let reserved_width = status_width + port_width + pid_width + 3 + 2;
    let model_width = usize::from(body_width)
        .saturating_sub(reserved_width)
        .max(8);
    [model_width, pid_width, port_width, status_width]
}

fn webserver_process_column_widths(body_width: u16) -> [usize; 4] {
    let pid_width = 5usize;
    let port_width = 5usize;
    let status_width = 5usize;
    let reserved_width = pid_width + port_width + status_width + 3 + 2;
    let label_width = usize::from(body_width)
        .saturating_sub(reserved_width)
        .max(PRETTY_TUI_WEBSERVER_PROCESS_HEADER_LABEL.len());
    [label_width, pid_width, port_width, status_width]
}

fn render_events_panel(
    frame: &mut Frame,
    state: &DashboardState,
    title_area: Rect,
    body_area: Rect,
) {
    render_events_panel_with_renderer(
        frame,
        state,
        title_area,
        body_area,
        TuiEventListRenderer::ACTIVE,
    );
}

fn render_events_panel_with_renderer(
    frame: &mut Frame,
    state: &DashboardState,
    title_area: Rect,
    body_area: Rect,
    renderer: TuiEventListRenderer,
) {
    let panel_area = combine_panel_rect(title_area, body_area);
    let block = tui_panel_block(state, DashboardPanel::Events);
    frame.render_widget(block.clone(), panel_area);
    let inner_area = block.inner(panel_area);
    if inner_area.height == 0 {
        return;
    }

    match renderer {
        TuiEventListRenderer::Legacy => render_legacy_events_list(frame, state, inner_area),
        TuiEventListRenderer::Scrollbar => render_scrollbar_events_list(frame, state, inner_area),
    }
}

fn render_legacy_events_list(frame: &mut Frame, state: &DashboardState, inner_area: Rect) {
    let view = state.panel_view_state(DashboardPanel::Events);
    let row_count = state.row_count_for_panel(DashboardPanel::Events);
    let viewport_rows = usize::from(inner_area.height).max(1);
    let scroll_offset = effective_events_scroll_offset(state, row_count, viewport_rows);
    let layout = tui_list_scrollbar_layout(inner_area, row_count, viewport_rows);
    let content_width = usize::from(
        layout
            .list_area
            .width
            .saturating_sub(PRETTY_TUI_LIST_HIGHLIGHT_SYMBOL_WIDTH)
            .max(1),
    );
    let rows = visible_event_rows_from(state, viewport_rows, scroll_offset);
    let is_focused = state.panel_focus == DashboardPanel::Events;
    render_event_list_rows(
        frame,
        layout.list_area,
        &rows,
        view.selected_row,
        is_focused,
        content_width,
    );

    if let Some(scrollbar_area) = layout.scrollbar_area {
        let mut scrollbar_state = tui_list_scrollbar_state(row_count, viewport_rows, scroll_offset);
        let scrollbar = Scrollbar::new(ScrollbarOrientation::VerticalRight)
            .begin_symbol(None)
            .end_symbol(None)
            .track_symbol(Some("│"));
        frame.render_stateful_widget(scrollbar, scrollbar_area, &mut scrollbar_state);
    }
}

fn render_scrollbar_events_list(frame: &mut Frame, state: &DashboardState, inner_area: Rect) {
    let row_count = state.row_count_for_panel(DashboardPanel::Events);
    let viewport_rows = usize::from(inner_area.height).max(1);
    let scroll_offset = effective_events_scroll_offset(state, row_count, viewport_rows);
    let events = state.filtered_mesh_events();
    frame.render_widget(
        TuiScrollbarEventList {
            events: &events,
            empty_message: empty_panel_message(state, DashboardPanel::Events),
            scroll_offset,
        },
        inner_area,
    );
}

struct TuiScrollbarEventList<'a> {
    events: &'a [&'a MeshEventState],
    empty_message: &'static str,
    scroll_offset: usize,
}

impl Widget for TuiScrollbarEventList<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        Widget::render(RatatuiClear, area, buf);
        if area.height == 0 {
            return;
        }

        let row_count = self.events.len();
        let viewport_rows = usize::from(area.height).max(1);
        let layout = tui_list_scrollbar_layout(area, row_count, viewport_rows);
        let content_width = usize::from(layout.list_area.width.max(1));

        if row_count == 0 {
            let line = Line::from(Span::styled(
                self.empty_message.to_string(),
                Style::default().fg(Color::DarkGray),
            ));
            Widget::render(line, single_line_rect(layout.list_area, 0), buf);
            return;
        }

        let scroll_offset = self
            .scroll_offset
            .min(row_count.saturating_sub(viewport_rows));
        for (row_index, event) in self
            .events
            .iter()
            .skip(scroll_offset)
            .take(viewport_rows)
            .enumerate()
        {
            let row_area = single_line_rect(layout.list_area, row_index);
            if row_area.height == 0 {
                break;
            }
            Widget::render(event_line(event, content_width), row_area, buf);
        }

        if let Some(scrollbar_area) = layout.scrollbar_area {
            let mut scrollbar_state =
                tui_list_scrollbar_state(row_count, viewport_rows, scroll_offset);
            let scrollbar = Scrollbar::new(ScrollbarOrientation::VerticalRight)
                .begin_symbol(None)
                .end_symbol(None)
                .track_symbol(Some("│"));
            StatefulWidget::render(scrollbar, scrollbar_area, buf, &mut scrollbar_state);
        }
    }
}

fn single_line_rect(area: Rect, row_index: usize) -> Rect {
    let y = area
        .y
        .saturating_add(u16::try_from(row_index).unwrap_or(u16::MAX));
    if y >= area.bottom() {
        return Rect { height: 0, ..area };
    }
    Rect {
        y,
        height: 1,
        ..area
    }
}

fn effective_events_scroll_offset(
    state: &DashboardState,
    row_count: usize,
    viewport_rows: usize,
) -> usize {
    if row_count == 0 {
        return 0;
    }

    let max_scroll_offset = row_count.saturating_sub(viewport_rows);
    if state.events_follow {
        max_scroll_offset
    } else {
        state
            .panel_view_state(DashboardPanel::Events)
            .scroll_offset
            .min(max_scroll_offset)
    }
}

fn render_event_list_rows(
    frame: &mut Frame,
    area: Rect,
    rows: &[TuiEventRow<'_>],
    selected_row: Option<usize>,
    is_focused: bool,
    content_width: usize,
) {
    frame.render_widget(RatatuiClear, area);

    let reserve_highlight_column = selected_row.is_some();
    let highlight_style = process_table_highlight_style(is_focused);
    for (row_index, row) in rows.iter().take(usize::from(area.height)).enumerate() {
        let y = area
            .y
            .saturating_add(u16::try_from(row_index).unwrap_or(u16::MAX));
        if y >= area.bottom() {
            break;
        }

        let row_area = Rect {
            y,
            height: 1,
            ..area
        };
        let selected = matches!(
            row,
            TuiEventRow::Event { absolute_index, .. }
                if Some(*absolute_index) == selected_row
        );
        let line = match row {
            TuiEventRow::Event { event, .. } => event_line(event, content_width),
            TuiEventRow::Message(message) => Line::from(Span::styled(
                (*message).to_string(),
                Style::default().fg(Color::DarkGray),
            )),
            TuiEventRow::Padding => Line::raw(""),
        };
        let line = event_list_line(line, reserve_highlight_column, selected, is_focused);
        Widget::render(line, row_area, frame.buffer_mut());
        if selected {
            frame.buffer_mut().set_style(row_area, highlight_style);
        }
    }
}

fn event_list_line(
    mut line: Line<'static>,
    reserve_highlight_column: bool,
    selected: bool,
    is_focused: bool,
) -> Line<'static> {
    if reserve_highlight_column {
        let symbol = if selected && is_focused { "› " } else { "  " };
        line.spans.insert(0, Span::raw(symbol));
    }
    line
}

fn render_model_progress_loader(frame: &mut Frame, state: &DashboardState, area: Rect) {
    if area.height < 2 || area.width < 12 {
        return;
    }
    let progress = state.active_loading_progress();
    let logo_text = tui_logo_view(area, false);
    let logo_height = logo_text
        .as_ref()
        .map(|text| u16::try_from(text.lines.len()).unwrap_or(u16::MAX))
        .unwrap_or(0)
        .min(area.height);
    let has_progress = progress.is_some();
    let gap_height = u16::from(has_progress && logo_height > 0);
    let bar_height = u16::from(has_progress);
    let detail_height = u16::from(has_progress);
    let loader_height = logo_height
        .saturating_add(gap_height)
        .saturating_add(bar_height)
        .saturating_add(detail_height)
        .max(logo_height.max(1))
        .min(area.height);
    let loader_area = Rect {
        x: area.x,
        y: area.y + area.height.saturating_sub(loader_height) / 2,
        width: area.width,
        height: loader_height,
    };

    let theme = tui_theme();

    if let Some(logo_text) = logo_text {
        let logo_area = Rect {
            x: loader_area.x,
            y: loader_area.y,
            width: loader_area.width,
            height: logo_height,
        };
        frame.render_widget(
            Paragraph::new(logo_text).alignment(Alignment::Center),
            logo_area,
        );
    }

    if let Some(progress) = progress {
        let bar_y = loader_area.y + logo_height + gap_height;
        let bar_area = Rect {
            x: loader_area.x,
            y: bar_y,
            width: loader_area.width,
            height: 1,
        };
        frame.render_widget(
            Paragraph::new(Line::from(vec![Span::styled(
                loading_progress_bar(
                    progress.ratio,
                    usize::from(bar_area.width).saturating_sub(12),
                ),
                Style::default().fg(theme.accent),
            )]))
            .alignment(Alignment::Center),
            bar_area,
        );

        let detail_area = Rect {
            x: loader_area.x,
            y: bar_y.saturating_add(1),
            width: loader_area.width,
            height: 1,
        };
        frame.render_widget(
            Paragraph::new(Line::from(Span::styled(
                progress.detail,
                Style::default().fg(theme.muted),
            )))
            .alignment(Alignment::Center),
            detail_area,
        );
    }
}

fn render_tui_logo(frame: &mut Frame, area: Rect, dimmed: bool) {
    let Some(logo_text) = tui_logo_view(area, dimmed) else {
        return;
    };
    let logo_height = u16::try_from(logo_text.lines.len())
        .unwrap_or(u16::MAX)
        .min(area.height);
    let logo_y = if dimmed {
        area.y
    } else {
        area.y + area.height.saturating_sub(logo_height) / 2
    };
    let logo_area = Rect {
        x: area.x,
        y: logo_y,
        width: area.width,
        height: logo_height,
    };
    frame.render_widget(
        Paragraph::new(logo_text).alignment(if dimmed {
            Alignment::Left
        } else {
            Alignment::Center
        }),
        logo_area,
    );
}

fn tui_logo_view(area: Rect, dimmed: bool) -> Option<Text<'static>> {
    let source = if dimmed {
        tui_ready_logo_text()?
    } else {
        tui_logo_text()?
    };
    Some(tui_crop_logo_text(source, area, dimmed))
}

fn tui_logo_text() -> Option<&'static Text<'static>> {
    PRETTY_TUI_SPLASH_TEXT
        .get_or_init(|| PRETTY_TUI_SPLASH_ANSI.into_text().ok().map(tui_static_text))
        .as_ref()
}

fn tui_static_text(text: Text<'_>) -> Text<'static> {
    Text {
        alignment: text.alignment,
        style: text.style,
        lines: text
            .lines
            .into_iter()
            .map(|line| Line {
                alignment: line.alignment,
                style: line.style,
                spans: line
                    .spans
                    .into_iter()
                    .map(|span| Span {
                        content: span.content.into_owned().into(),
                        style: span.style,
                    })
                    .collect(),
            })
            .collect(),
    }
}

fn tui_ready_logo_text() -> Option<&'static Text<'static>> {
    PRETTY_TUI_READY_LOGO_TEXT
        .get_or_init(|| tui_logo_text().map(tui_trim_logo_text))
        .as_ref()
}

fn tui_trim_logo_text(source: &Text<'static>) -> Text<'static> {
    let first_visible = source
        .lines
        .iter()
        .position(tui_logo_line_has_visible_content)
        .unwrap_or(0);
    let last_visible = source
        .lines
        .iter()
        .rposition(tui_logo_line_has_visible_content)
        .map(|index| index + 1)
        .unwrap_or(source.lines.len());
    let visible_lines = &source.lines[first_visible..last_visible];
    let Some((first_column, last_column)) = tui_logo_visible_columns(visible_lines) else {
        return Text::from(visible_lines.to_vec());
    };
    Text::from(
        visible_lines
            .iter()
            .map(|line| tui_slice_logo_line(line, first_column, last_column))
            .collect::<Vec<_>>(),
    )
}

fn tui_crop_logo_text(source: &Text<'static>, area: Rect, dimmed: bool) -> Text<'static> {
    if area.width == 0 || area.height == 0 {
        return Text::default();
    }

    let visible_height = source.lines.len().min(usize::from(area.height));
    let line_start = if dimmed {
        0
    } else {
        source.lines.len().saturating_sub(visible_height) / 2
    };
    let mut lines = Vec::with_capacity(visible_height);
    let dim_patch = dimmed.then(|| Style::default().add_modifier(Modifier::DIM));

    for line in source.lines.iter().skip(line_start).take(visible_height) {
        let mut cropped = tui_crop_logo_line(line, usize::from(area.width));
        if let Some(dim_patch) = dim_patch {
            for span in &mut cropped.spans {
                span.style = span.style.patch(dim_patch);
            }
        }
        lines.push(cropped);
    }

    Text::from(lines)
}

fn tui_crop_logo_line(line: &Line<'static>, max_width: usize) -> Line<'static> {
    if max_width == 0 {
        return Line::default();
    }

    let line_width = tui_logo_line_width(line);
    if line_width <= max_width {
        return line.clone();
    }

    let crop_start = line_width.saturating_sub(max_width) / 2;
    let crop_end = crop_start + max_width;
    let mut spans = Vec::new();
    let mut offset = 0usize;

    for span in &line.spans {
        let span_width = span.content.chars().count();
        let span_start = offset;
        let span_end = offset + span_width;
        let take_start = crop_start.max(span_start);
        let take_end = crop_end.min(span_end);

        if take_start < take_end {
            let content: String = span
                .content
                .chars()
                .skip(take_start - span_start)
                .take(take_end - take_start)
                .collect();
            if !content.is_empty() {
                spans.push(Span::styled(content, span.style));
            }
        }

        offset = span_end;
        if offset >= crop_end {
            break;
        }
    }

    Line::from(spans)
}

fn tui_slice_logo_line(line: &Line<'static>, start: usize, end: usize) -> Line<'static> {
    if start >= end {
        return Line::default();
    }

    let mut spans = Vec::new();
    let mut offset = 0usize;

    for span in &line.spans {
        let span_width = span.content.chars().count();
        let span_start = offset;
        let span_end = offset + span_width;
        let take_start = start.max(span_start);
        let take_end = end.min(span_end);

        if take_start < take_end {
            let content: String = span
                .content
                .chars()
                .skip(take_start - span_start)
                .take(take_end - take_start)
                .collect();
            if !content.is_empty() {
                spans.push(Span::styled(content, span.style));
            }
        }

        offset = span_end;
        if offset >= end {
            break;
        }
    }

    Line::from(spans)
}

fn tui_logo_visible_columns(lines: &[Line<'static>]) -> Option<(usize, usize)> {
    let mut first = usize::MAX;
    let mut last = 0usize;

    for line in lines {
        let mut offset = 0usize;
        for span in &line.spans {
            for ch in span.content.chars() {
                if !ch.is_whitespace() {
                    first = first.min(offset);
                    last = last.max(offset + 1);
                }
                offset += 1;
            }
        }
    }

    (first < last).then_some((first, last))
}

fn tui_logo_line_width(line: &Line<'static>) -> usize {
    line.spans
        .iter()
        .map(|span| span.content.chars().count())
        .sum()
}

fn tui_logo_line_has_visible_content(line: &Line<'static>) -> bool {
    line.spans
        .iter()
        .any(|span| span.content.chars().any(|ch| !ch.is_whitespace()))
}

fn loading_progress_bar(ratio: f64, width: usize) -> String {
    let width = width.clamp(8, 40);
    let filled = (ratio.clamp(0.0, 1.0) * width as f64)
        .round()
        .clamp(1.0, width as f64) as usize;
    format!("{}{}", "█".repeat(filled), "░".repeat(width - filled))
}

fn model_download_progress_ratio(progress: &ModelProgressState) -> Option<f64> {
    match (progress.downloaded_bytes, progress.total_bytes) {
        (Some(downloaded), Some(total))
            if total > 0 && matches!(progress.status, ModelProgressStatus::Downloading) =>
        {
            Some(downloaded.min(total) as f64 / total as f64)
        }
        _ => None,
    }
}

fn fallback_model_progress_ratio(progress: &ModelProgressState) -> f64 {
    if let Some(ratio) = model_download_progress_ratio(progress) {
        return ratio;
    }

    match progress.status {
        ModelProgressStatus::Ready => 0.85,
        ModelProgressStatus::Downloading => 0.33,
        ModelProgressStatus::Ensuring => 0.20,
    }
}

fn startup_progress_ratio(progress: &StartupProgressState) -> f64 {
    if progress.total_steps == 0 {
        return 0.0;
    }

    progress.completed_steps.min(progress.total_steps) as f64 / progress.total_steps as f64
}

fn loading_progress_detail(detail: String, ratio: f64, steps: Option<(usize, usize)>) -> String {
    let percent = (ratio.clamp(0.0, 1.0) * 100.0).round() as usize;
    match steps {
        Some((completed, total)) => format!("{detail}  {percent}% ({completed}/{total})"),
        None => format!("{detail}  {percent}%"),
    }
}

fn startup_progress_event(event: &OutputEvent) -> Option<(Option<String>, String)> {
    match event {
        OutputEvent::Startup { version, .. } => Some((
            Some("startup".to_string()),
            format!("starting mesh-llm {version}"),
        )),
        OutputEvent::DiscoveryStarting { source } => Some((
            Some("discovery_starting".to_string()),
            format!("discovering mesh via {source}"),
        )),
        OutputEvent::MeshFound { mesh, peers, .. } => Some((
            Some("mesh_found".to_string()),
            format!("found mesh {mesh} with {peers} peer(s)"),
        )),
        OutputEvent::DiscoveryJoined { mesh } => Some((
            Some("discovery_joined".to_string()),
            format!("joined mesh {mesh}"),
        )),
        OutputEvent::WaitingForPeers { detail } => Some((
            Some("waiting_for_peers".to_string()),
            detail
                .clone()
                .unwrap_or_else(|| "waiting for peers".to_string()),
        )),
        OutputEvent::ModelQueued { model } => Some((
            Some(format!("model_queued:{model}")),
            format!("queued model {model}"),
        )),
        OutputEvent::ModelLoading { model, .. } => Some((
            Some(format!("model_loading:{model}")),
            format!("loading model {model}"),
        )),
        OutputEvent::ModelLoaded { model, .. } => Some((
            Some(format!("model_loaded:{model}")),
            format!("loaded model {model}"),
        )),
        OutputEvent::ModelDownloadProgress {
            label,
            file,
            downloaded_bytes,
            total_bytes,
            status,
        } => {
            let progress = ModelProgressState {
                label: label.clone(),
                file: file.clone(),
                downloaded_bytes: *downloaded_bytes,
                total_bytes: *total_bytes,
                status: status.clone(),
            };
            let milestone_key = matches!(status, ModelProgressStatus::Ready)
                .then(|| format!("model_download_ready:{label}"));
            Some((milestone_key, model_progress_detail(&progress)))
        }
        OutputEvent::HostElected { model, host, .. } => Some((
            Some(format!("host_elected:{model}")),
            format!("elected {host} for {model}"),
        )),
        OutputEvent::RpcServerStarting { port, device, .. } => Some((
            Some(format!("rpc_server_starting:{port}")),
            format!("starting rpc-server on {device}"),
        )),
        OutputEvent::RpcReady { port, device, .. } => Some((
            Some(format!("rpc_ready:{port}")),
            format!("rpc-server ready on {device}"),
        )),
        OutputEvent::LlamaStarting {
            model, http_port, ..
        } => Some((
            Some(format!("llama_starting:{}", model_key(model, *http_port))),
            model
                .as_ref()
                .map(|model| format!("starting llama-server for {model}"))
                .unwrap_or_else(|| format!("starting llama-server on port {http_port}")),
        )),
        OutputEvent::LlamaReady { model, port, .. } => Some((
            Some(format!("llama_ready:{}", model_key(model, *port))),
            model
                .as_ref()
                .map(|model| format!("llama-server ready for {model}"))
                .unwrap_or_else(|| format!("llama-server ready on port {port}")),
        )),
        OutputEvent::ModelReady { model, .. } => Some((
            Some(format!("model_ready:{model}")),
            format!("model {model} ready"),
        )),
        OutputEvent::WebserverStarting { url } => Some((
            Some("webserver_starting".to_string()),
            format!("starting console at {url}"),
        )),
        OutputEvent::WebserverReady { url } => Some((
            Some("webserver_ready".to_string()),
            format!("console ready at {url}"),
        )),
        OutputEvent::ApiStarting { url } => Some((
            Some("api_starting".to_string()),
            format!("starting API at {url}"),
        )),
        OutputEvent::ApiReady { url } => {
            Some((Some("api_ready".to_string()), format!("API ready at {url}")))
        }
        OutputEvent::RuntimeReady { .. } => Some((
            Some("runtime_ready".to_string()),
            "mesh-llm runtime ready".to_string(),
        )),
        _ => None,
    }
}

fn model_key(model: &Option<String>, port: u16) -> String {
    model
        .as_ref()
        .cloned()
        .unwrap_or_else(|| format!("port:{port}"))
}

fn model_progress_detail(progress: &ModelProgressState) -> String {
    let target = progress.file.as_deref().unwrap_or(&progress.label);
    format_model_download_progress_message(
        &progress.label,
        Some(target),
        progress.downloaded_bytes,
        progress.total_bytes,
        &progress.status,
    )
}

fn dashboard_status_line(state: &DashboardState, width: u16) -> Line<'static> {
    let theme = tui_theme();
    let readiness = readiness_label(state);
    let mut left_spans = vec![Span::styled(
        readiness_badge(readiness),
        readiness_badge_style(readiness),
    )];
    left_spans.push(Span::raw(" "));
    push_status_key_hint(&mut left_spans, "Q", "Quit");
    push_status_key_hint(&mut left_spans, "Tab", "Next");
    push_status_key_hint(&mut left_spans, "↑/↓", "Window");
    push_status_key_hint(&mut left_spans, "Shift-Tab", "Prev");
    push_status_key_hint(&mut left_spans, "/", "Filter");
    push_status_key_hint(&mut left_spans, "F", "Follow");
    push_status_key_hint(&mut left_spans, "R", "Refresh");

    let mut right_spans = Vec::new();
    push_status_metric(&mut right_spans, "peers", state.peer_ids.len().to_string());
    push_status_metric(
        &mut right_spans,
        "models",
        visible_model_count(state).to_string(),
    );
    push_status_metric(
        &mut right_spans,
        "processes",
        visible_process_count(state).to_string(),
    );
    push_status_metric(&mut right_spans, "uptime", dashboard_uptime_label(state));
    right_spans.push(status_separator_span());
    right_spans.push(Span::styled(
        Local::now().format("%H:%M:%S").to_string(),
        Style::default().fg(theme.muted),
    ));

    let mut spans = left_spans;
    let left_width = status_spans_width(&spans);
    let right_width = status_spans_width(&right_spans);
    let gap_width = usize::from(width)
        .saturating_sub(left_width)
        .saturating_sub(right_width)
        .max(1);
    spans.push(status_gap_span(gap_width));
    spans.extend(right_spans);

    Line::from(spans)
}

fn status_spans_width(spans: &[Span<'_>]) -> usize {
    spans.iter().map(|span| span.content.chars().count()).sum()
}

fn status_gap_span(width: usize) -> Span<'static> {
    Span::raw(" ".repeat(width))
}

fn push_status_metric(spans: &mut Vec<Span<'static>>, label: &'static str, value: String) {
    let theme = tui_theme();
    spans.push(status_separator_span());
    spans.push(Span::styled(
        format!("{label}: "),
        Style::default().fg(theme.dim),
    ));
    spans.push(Span::styled(value, Style::default().fg(theme.text)));
}

fn status_separator_span() -> Span<'static> {
    Span::styled(" | ", Style::default().fg(tui_theme().dim))
}

fn push_status_key_hint(spans: &mut Vec<Span<'static>>, key: &'static str, label: &'static str) {
    spans.push(key_hint_span(key));
    spans.push(Span::raw(" "));
    spans.push(hint_label_span(label));
    spans.push(Span::raw(" "));
}

fn readiness_badge(readiness: &str) -> String {
    format!(" {} ", readiness.to_ascii_uppercase())
}

fn readiness_badge_style(readiness: &str) -> Style {
    let theme = tui_theme();
    let color = match readiness {
        "ready" => theme.success,
        "degraded" => theme.warning,
        "starting" | "warming" => theme.accent_soft,
        "stopped" => theme.dim,
        _ => theme.muted,
    };
    Style::default()
        .fg(color)
        .bg(theme.surface)
        .add_modifier(Modifier::BOLD)
}

fn dashboard_uptime_label(state: &DashboardState) -> String {
    format_duration_compact(state.session_started_at.elapsed())
}

fn format_duration_compact(duration: Duration) -> String {
    let total_secs = duration.as_secs();
    let hours = total_secs / 3600;
    let minutes = (total_secs % 3600) / 60;
    let seconds = total_secs % 60;
    if hours > 0 {
        format!("{hours}h{minutes:02}m")
    } else if minutes > 0 {
        format!("{minutes}m{seconds:02}s")
    } else {
        format!("{seconds}s")
    }
}

fn key_hint_span(key: &'static str) -> Span<'static> {
    let theme = tui_theme();
    Span::styled(
        format!("[{key}]"),
        Style::default()
            .fg(theme.accent)
            .bg(theme.surface_raised)
            .add_modifier(Modifier::BOLD),
    )
}

fn hint_label_span(label: &'static str) -> Span<'static> {
    Span::styled(label.to_string(), Style::default().fg(tui_theme().muted))
}

fn format_tui_panel_title(state: &DashboardState, panel: DashboardPanel) -> String {
    let focus_marker = if state.panel_focus == panel {
        '▶'
    } else {
        ' '
    };
    match panel {
        DashboardPanel::JoinToken => join_token_panel_left_title(state, focus_marker),
        DashboardPanel::Events => format!(
            "{focus_marker} Mesh Events  follow={}  filter={}",
            if state.events_follow { "ON" } else { "OFF" },
            events_filter_label(&state.events_filter)
        ),
        DashboardPanel::LlamaCpp => format!("{focus_marker} llama.cpp Processes"),
        DashboardPanel::Webserver => format!("{focus_marker} mesh-llm Processes"),
        DashboardPanel::Models => format!("{focus_marker} Loaded Models"),
        DashboardPanel::Requests => format!(
            "{focus_marker} Incoming Requests  {}  {}",
            state.request_window.label(),
            state.request_window.bucket_label()
        ),
    }
}

fn panel_title_style(state: &DashboardState, panel: DashboardPanel) -> Style {
    let theme = tui_theme();
    if state.panel_focus == panel {
        Style::default()
            .fg(theme.accent)
            .bg(theme.surface_raised)
            .add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(theme.dim).add_modifier(Modifier::DIM)
    }
}

fn panel_border_style(state: &DashboardState, panel: DashboardPanel) -> Style {
    let theme = tui_theme();
    if state.panel_focus == panel {
        Style::default().fg(theme.accent)
    } else {
        Style::default().fg(theme.dim)
    }
}

#[cfg(test)]
fn visible_event_rows<'a>(state: &'a DashboardState, viewport_rows: usize) -> Vec<TuiEventRow<'a>> {
    let scroll_offset = state.panel_view_state(DashboardPanel::Events).scroll_offset;
    visible_event_rows_from(state, viewport_rows, scroll_offset)
}

fn visible_event_rows_from<'a>(
    state: &'a DashboardState,
    viewport_rows: usize,
    scroll_offset: usize,
) -> Vec<TuiEventRow<'a>> {
    let row_count = state.row_count_for_panel(DashboardPanel::Events);
    let mut rows = if row_count == 0 {
        vec![TuiEventRow::Message(empty_panel_message(
            state,
            DashboardPanel::Events,
        ))]
    } else {
        state
            .filtered_mesh_events()
            .into_iter()
            .enumerate()
            .skip(scroll_offset)
            .take(viewport_rows)
            .map(|(absolute_index, event)| TuiEventRow::Event {
                absolute_index,
                event,
            })
            .collect::<Vec<_>>()
    };

    if state.events_follow && row_count > 0 {
        let padding = viewport_rows.saturating_sub(rows.len());
        if padding > 0 {
            let mut anchored_rows = Vec::with_capacity(viewport_rows);
            anchored_rows.extend((0..padding).map(|_| TuiEventRow::Padding));
            anchored_rows.extend(rows);
            rows = anchored_rows;
        }
    }

    while rows.len() < viewport_rows.max(1) {
        rows.push(TuiEventRow::Padding);
    }

    rows
}

fn empty_panel_message(state: &DashboardState, panel: DashboardPanel) -> &'static str {
    match panel {
        DashboardPanel::JoinToken => "join token will appear here when the mesh invite is ready",
        DashboardPanel::Events if state.events_filter.is_active() => {
            "(no events match the current filter)"
        }
        DashboardPanel::Events => "(waiting for mesh events)",
        DashboardPanel::LlamaCpp => "(no llama.cpp processes yet)",
        DashboardPanel::Webserver => "(no webserver processes yet)",
        DashboardPanel::Models => "(no loaded models yet)",
        DashboardPanel::Requests => "(incoming request metrics will appear here)",
    }
}

fn event_severity_badge(event: &MeshEventState) -> (&'static str, Style) {
    let theme = tui_theme();
    let summary_lower = event.summary.to_lowercase();
    if matches!(event.level, OutputLevel::Error)
        || summary_lower.contains("err")
        || summary_lower.contains("failed")
    {
        (
            "ERR",
            Style::default()
                .fg(theme.error)
                .add_modifier(Modifier::BOLD),
        )
    } else if matches!(event.level, OutputLevel::Warn) || summary_lower.contains("warn") {
        (
            "WARN",
            Style::default()
                .fg(theme.warning)
                .add_modifier(Modifier::BOLD),
        )
    } else if summary_lower.contains("ready")
        || summary_lower.contains("elected")
        || summary_lower.contains("joined")
        || summary_lower.contains("ok")
    {
        (
            "OK",
            Style::default()
                .fg(theme.success)
                .add_modifier(Modifier::BOLD),
        )
    } else {
        (
            "INFO",
            Style::default()
                .fg(theme.accent_soft)
                .add_modifier(Modifier::BOLD),
        )
    }
}

fn event_severity_badge_span(event: &MeshEventState) -> Span<'static> {
    let (badge_text, badge_style) = event_severity_badge(event);
    Span::styled(
        format!("{badge_text:<PRETTY_TUI_EVENT_LEVEL_WIDTH$}"),
        badge_style,
    )
}

fn event_matches_filter(event: &MeshEventState, needle: &str) -> bool {
    let (badge_text, _) = event_severity_badge(event);
    let sanitized_message = sanitize_mesh_event_message(&event.summary);
    let rendered_search_text =
        format!("{} {} {}", event.timestamp, badge_text, sanitized_message).to_lowercase();
    rendered_search_text.contains(needle)
}

fn event_line(event: &MeshEventState, width: usize) -> Line<'static> {
    let theme = tui_theme();
    let (badge_text, _) = event_severity_badge(event);
    let message = sanitize_mesh_event_message(&event.summary);
    let prefix = format!(
        "{} {:<PRETTY_TUI_EVENT_LEVEL_WIDTH$}",
        event.timestamp, badge_text
    );
    let prefix_len = prefix.chars().count();
    let remaining = width.saturating_sub(prefix_len + 1);
    if remaining == 0 {
        return Line::from(vec![Span::styled(
            truncate_with_ellipsis(&prefix, width),
            Style::default().fg(theme.dim),
        )]);
    }

    Line::from(vec![
        Span::styled(event.timestamp.clone(), Style::default().fg(theme.dim)),
        Span::raw(" "),
        event_severity_badge_span(event),
        Span::raw(" "),
        Span::styled(
            truncate_with_ellipsis(&message, remaining),
            Style::default().fg(theme.text),
        ),
    ])
}

fn sanitize_mesh_event_message(message: &str) -> String {
    let mut output = String::with_capacity(message.len());
    let mut last_was_space = false;
    for ch in message.chars().filter(|ch| !is_mesh_event_emoji(*ch)) {
        if ch.is_whitespace() {
            if !last_was_space {
                output.push(' ');
            }
            last_was_space = true;
        } else {
            output.push(ch);
            last_was_space = false;
        }
    }
    output.trim().to_string()
}

fn is_mesh_event_emoji(ch: char) -> bool {
    matches!(
        ch as u32,
        0x1F300..=0x1FAFF | 0x2300..=0x23FF | 0x2600..=0x27BF | 0xFE0F
    )
}

#[cfg(test)]
fn format_event_row(event: &MeshEventState, width: usize) -> String {
    spans_plain_text(&event_line(event, width).spans)
}

fn readiness_label(state: &DashboardState) -> &'static str {
    if state.runtime_ready {
        "ready"
    } else if state.llama_instances.iter().any(|instance| {
        matches!(
            instance.status,
            RuntimeStatus::Error | RuntimeStatus::Warning
        )
    }) || state
        .running_models
        .iter()
        .any(|model| matches!(model.status, RuntimeStatus::Error | RuntimeStatus::Warning))
        || state
            .loaded_model_rows
            .iter()
            .any(|row| matches!(row.status, RuntimeStatus::Error | RuntimeStatus::Warning))
        || state
            .webserver_rows
            .iter()
            .any(|row| matches!(row.status, RuntimeStatus::Error | RuntimeStatus::Warning))
    {
        "degraded"
    } else if state
        .llama_instances
        .iter()
        .any(|instance| matches!(instance.status, RuntimeStatus::Starting))
        || state
            .running_models
            .iter()
            .any(|model| matches!(model.status, RuntimeStatus::Starting))
        || state
            .loaded_model_rows
            .iter()
            .any(|row| matches!(row.status, RuntimeStatus::Starting))
        || state
            .webserver_rows
            .iter()
            .any(|row| matches!(row.status, RuntimeStatus::Starting))
    {
        "starting"
    } else if state
        .llama_instances
        .iter()
        .all(|instance| matches!(instance.status, RuntimeStatus::Stopped))
        && state
            .running_models
            .iter()
            .all(|model| matches!(model.status, RuntimeStatus::Stopped))
        && !matches!(
            state.webserver.as_ref().map(|endpoint| &endpoint.status),
            Some(RuntimeStatus::Ready)
        )
        && !matches!(
            state.api.as_ref().map(|endpoint| &endpoint.status),
            Some(RuntimeStatus::Ready)
        )
    {
        "stopped"
    } else {
        "warming"
    }
}

fn visible_process_count(state: &DashboardState) -> usize {
    let snapshot_processes = state.llama_process_rows.len() + state.webserver_rows.len();
    if snapshot_processes > 0 {
        snapshot_processes
    } else {
        state.llama_instances.len()
            + usize::from(state.webserver.is_some())
            + usize::from(state.api.is_some())
    }
}

fn visible_model_count(state: &DashboardState) -> usize {
    if !state.loaded_model_rows.is_empty() {
        state.loaded_model_rows.len()
    } else {
        state.running_models.len()
    }
}

fn events_filter_label(filter: &DashboardEventsFilterState) -> String {
    if filter.editing {
        format!("/{query}_", query = filter.query)
    } else if filter.query.is_empty() {
        "(none)".to_string()
    } else {
        format!("/{query}", query = filter.query)
    }
}

fn truncate_with_ellipsis(text: &str, width: usize) -> String {
    if width == 0 {
        return String::new();
    }
    let count = text.chars().count();
    if count <= width {
        return text.to_string();
    }
    if width == 1 {
        return "…".to_string();
    }
    text.chars().take(width - 1).collect::<String>() + "…"
}

pub trait Formatter: Send {
    fn format(&mut self, event: &OutputEvent) -> io::Result<String>;
}

#[derive(Default)]
pub struct DashboardFormatter {
    state: DashboardState,
}

impl Formatter for DashboardFormatter {
    fn format(&mut self, event: &OutputEvent) -> io::Result<String> {
        self.state
            .reduce(DashboardAction::OutputEvent(event.clone()));
        Ok(render_dashboard_text(&self.state))
    }
}

#[derive(Default)]
pub struct InteractiveDashboardFormatter {
    state: DashboardState,
    terminal: Option<TuiTerminal>,
    terminal_active: bool,
    dirty: bool,
}

impl InteractiveDashboardFormatter {
    fn handle_output_event(&mut self, event: &OutputEvent) -> io::Result<Option<String>> {
        self.state
            .reduce(DashboardAction::OutputEvent(event.clone()));
        if self.terminal_active {
            self.dirty = true;
            Ok(None)
        } else {
            let (columns, rows) = crossterm::terminal::size().unwrap_or((120, 24));
            render_tui_text_snapshot(&self.state, columns, rows).map(Some)
        }
    }

    fn handle_snapshot(&mut self, snapshot: DashboardSnapshot) {
        self.state
            .reduce(DashboardAction::SnapshotUpdated(snapshot));
        if self.terminal_active {
            self.dirty = true;
        }
    }

    fn handle_tui_event(&mut self, event: TuiEvent) -> TuiControlFlow {
        let control = self.state.apply_tui_event(event);
        if self.terminal_active {
            self.dirty = true;
        }
        control
    }

    fn enter_terminal(&mut self) -> io::Result<()> {
        if self.terminal_active {
            return Ok(());
        }
        write_tui_enter()?;
        self.mark_terminal_escape_written();
        let backend = CrosstermBackend::new(io::stderr());
        let mut terminal = Terminal::new(backend).map_err(io::Error::other)?;
        terminal.hide_cursor().map_err(io::Error::other)?;
        self.terminal = Some(terminal);
        Ok(())
    }

    fn mark_terminal_escape_written(&mut self) {
        // From this point on, a later setup failure still needs normal TUI
        // cleanup: the terminal may already be in alternate-screen/raw-input
        // state even if ratatui terminal construction or cursor hiding fails.
        self.terminal_active = true;
        self.dirty = true;
    }

    fn exit_terminal(&mut self) -> io::Result<()> {
        if !self.terminal_active {
            return Ok(());
        }
        if let Some(mut terminal) = self.terminal.take() {
            terminal.show_cursor().map_err(io::Error::other)?;
        }
        self.terminal_active = false;
        self.dirty = false;
        write_tui_exit()
    }

    fn render_if_dirty(&mut self) -> io::Result<bool> {
        if !self.terminal_active || !self.dirty {
            return Ok(false);
        }
        let (columns, rows) = crossterm::terminal::size().unwrap_or((120, 40));
        self.state
            .reduce(DashboardAction::Resize(dashboard_layout_for_terminal_size(
                columns, rows,
            )));
        let terminal = self.terminal.as_mut().ok_or_else(|| {
            io::Error::other("pretty TUI terminal missing while terminal mode is active")
        })?;
        draw_tui_dashboard_with_terminal(terminal, &self.state)?;
        self.dirty = false;
        Ok(true)
    }
}

impl Formatter for InteractiveDashboardFormatter {
    fn format(&mut self, event: &OutputEvent) -> io::Result<String> {
        Ok(self.handle_output_event(event)?.unwrap_or_default())
    }
}

pub struct JsonFormatter;

impl Formatter for JsonFormatter {
    fn format(&mut self, event: &OutputEvent) -> io::Result<String> {
        let mut record = Map::new();
        record.insert(
            "timestamp".to_string(),
            Value::String(Utc::now().to_rfc3339_opts(SecondsFormat::Millis, true)),
        );
        record.insert(
            "level".to_string(),
            Value::String(event.level().as_str().to_string()),
        );
        record.insert(
            "event".to_string(),
            Value::String(event.event_name().to_string()),
        );
        record.extend(event.json_fields());
        record.insert("message".to_string(), Value::String(event.message()));
        serde_json::to_string(&Value::Object(record))
            .map(|line| format!("{line}\n"))
            .map_err(io::Error::other)
    }
}

enum FormatterSelection {
    InteractiveDashboard(InteractiveDashboardFormatter),
    DashboardFallback(DashboardFormatter),
    Json(JsonFormatter),
}

impl FormatterSelection {
    #[cfg(test)]
    fn kind(&self) -> &'static str {
        match self {
            Self::InteractiveDashboard(_) => "interactive_dashboard",
            Self::DashboardFallback(_) => "pretty_fallback",
            Self::Json(_) => "json",
        }
    }

    fn mode(&self) -> LogFormat {
        match self {
            Self::InteractiveDashboard(_) | Self::DashboardFallback(_) => LogFormat::Pretty,
            Self::Json(_) => LogFormat::Json,
        }
    }

    fn is_interactive_dashboard(&self) -> bool {
        matches!(self, Self::InteractiveDashboard(_))
    }

    fn handle_output_event(&mut self, event: &OutputEvent) -> io::Result<()> {
        match self {
            Self::InteractiveDashboard(formatter) => {
                if let Some(rendered) = formatter.handle_output_event(event)? {
                    write_rendered_output(LogFormat::Pretty, &rendered)?;
                }
                Ok(())
            }
            _ => {
                let rendered = self.format(event)?;
                write_rendered_output(self.mode(), &rendered)
            }
        }
    }

    fn enter_tui(&mut self) -> io::Result<()> {
        match self {
            Self::InteractiveDashboard(formatter) => formatter.enter_terminal(),
            _ => Ok(()),
        }
    }

    fn exit_tui(&mut self) -> io::Result<()> {
        match self {
            Self::InteractiveDashboard(formatter) => formatter.exit_terminal(),
            _ => Ok(()),
        }
    }

    fn handle_tui_event(&mut self, event: TuiEvent) -> TuiControlFlow {
        match self {
            Self::InteractiveDashboard(formatter) => formatter.handle_tui_event(event),
            _ => TuiControlFlow::Continue,
        }
    }

    fn handle_tui_snapshot(&mut self, snapshot: DashboardSnapshot) {
        if let Self::InteractiveDashboard(formatter) = self {
            formatter.handle_snapshot(snapshot);
        }
    }

    fn render_interactive_if_dirty(&mut self) -> io::Result<bool> {
        match self {
            Self::InteractiveDashboard(formatter) => formatter.render_if_dirty(),
            _ => Ok(false),
        }
    }

    fn writes_ready_prompt(&self) -> bool {
        matches!(self, Self::DashboardFallback(_))
    }
}

impl Formatter for FormatterSelection {
    fn format(&mut self, event: &OutputEvent) -> io::Result<String> {
        match self {
            Self::InteractiveDashboard(formatter) => formatter.format(event),
            Self::DashboardFallback(formatter) => formatter.format(event),
            Self::Json(formatter) => formatter.format(event),
        }
    }
}

fn select_formatter(
    mode: LogFormat,
    console_session_mode: ConsoleSessionMode,
) -> FormatterSelection {
    match mode {
        LogFormat::Pretty => match console_session_mode {
            ConsoleSessionMode::InteractiveDashboard => {
                FormatterSelection::InteractiveDashboard(InteractiveDashboardFormatter::default())
            }
            ConsoleSessionMode::Fallback => {
                FormatterSelection::DashboardFallback(DashboardFormatter::default())
            }
        },
        LogFormat::Json => FormatterSelection::Json(JsonFormatter),
    }
}

pub struct OutputManager {
    tx: tokio::sync::mpsc::UnboundedSender<OutputCommand>,
    ready_prompt_active: Arc<AtomicBool>,
    mode: LogFormat,
    console_session_mode: Option<ConsoleSessionMode>,
    dashboard_snapshot_provider: Arc<RwLock<Option<Arc<dyn DashboardSnapshotProvider>>>>,
}

enum OutputCommand {
    Event(OutputEvent),
    ActivateReadyPrompt,
    EnterTui(tokio::sync::oneshot::Sender<io::Result<()>>),
    ExitTui(tokio::sync::oneshot::Sender<io::Result<()>>),
    TuiEvent {
        event: TuiEvent,
        response: tokio::sync::oneshot::Sender<io::Result<TuiControlFlow>>,
    },
    RenderTui(tokio::sync::oneshot::Sender<io::Result<bool>>),
}

static GLOBAL_OUTPUT_MANAGER: OnceLock<OutputManager> = OnceLock::new();

impl OutputManager {
    pub fn init_global(
        mode: LogFormat,
        console_session_mode: ConsoleSessionMode,
    ) -> &'static OutputManager {
        GLOBAL_OUTPUT_MANAGER.get_or_init(|| {
            let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<OutputCommand>();
            let ready_prompt_active = Arc::new(AtomicBool::new(false));
            let worker_prompt_active = ready_prompt_active.clone();
            let dashboard_snapshot_provider: Arc<
                RwLock<Option<Arc<dyn DashboardSnapshotProvider>>>,
            > = Arc::new(RwLock::new(None));
            let worker_snapshot_provider = dashboard_snapshot_provider.clone();
            tokio::spawn(async move {
        let mut formatter = select_formatter(mode, console_session_mode);
                let mut redraw_tick = time::interval(PRETTY_TUI_REDRAW_INTERVAL);
                redraw_tick.set_missed_tick_behavior(MissedTickBehavior::Skip);
                let mut snapshot_tick = time::interval(PRETTY_TUI_SNAPSHOT_INTERVAL);
                snapshot_tick.set_missed_tick_behavior(MissedTickBehavior::Skip);
                let mut last_snapshot_at = Instant::now() - PRETTY_TUI_SNAPSHOT_INTERVAL;
                loop {
                    tokio::select! {
                        maybe_command = rx.recv() => {
                            let Some(command) = maybe_command else {
                                if let Err(err) = formatter.exit_tui() {
                                    tracing::warn!("interactive terminal cleanup failed: {err}");
                                }
                                break;
                            };
                            match command {
                                OutputCommand::Event(event) => {
                                    if let Err(err) = formatter.handle_output_event(&event) {
                                        tracing::warn!("output write failed: {err}");
                                    } else if matches!(mode, LogFormat::Pretty)
                                        && worker_prompt_active.load(Ordering::Acquire)
                                        && formatter.writes_ready_prompt()
                                    {
                                        if let Err(err) = write_prompt() {
                                            tracing::warn!("interactive prompt write failed: {err}");
                                        }
                                    }
                                }
                                OutputCommand::ActivateReadyPrompt => {
                                    worker_prompt_active.store(true, Ordering::Release);
                                    if matches!(mode, LogFormat::Pretty) && formatter.writes_ready_prompt() {
                                        if let Err(err) = write_prompt() {
                                            tracing::warn!("interactive prompt write failed: {err}");
                                        }
                                    }
                                }
                                OutputCommand::EnterTui(response) => {
                                    let _ = response.send(formatter.enter_tui());
                                }
                                OutputCommand::ExitTui(response) => {
                                    let _ = response.send(formatter.exit_tui());
                                }
                                OutputCommand::TuiEvent { event, response } => {
                                    let _ = response.send(Ok(formatter.handle_tui_event(event)));
                                }
                                OutputCommand::RenderTui(response) => {
                                    let _ = response.send(formatter.render_interactive_if_dirty());
                                }
                            }
                        }
                        _ = redraw_tick.tick(), if formatter.is_interactive_dashboard() => {
                            if let Err(err) = formatter.render_interactive_if_dirty() {
                                tracing::warn!("interactive dashboard redraw failed: {err}");
                            }
                        }
                        _ = snapshot_tick.tick(), if formatter.is_interactive_dashboard() => {
                            if last_snapshot_at.elapsed() < PRETTY_TUI_SNAPSHOT_INTERVAL {
                                continue;
                            }
                            let Some(provider) = worker_snapshot_provider
                                .read()
                                .ok()
                                .and_then(|slot| slot.clone()) else {
                                continue;
                            };
                            last_snapshot_at = Instant::now();
                            formatter.handle_tui_snapshot(provider.snapshot().await);
                        }
                    }
                }
            });
            Self {
                tx,
                ready_prompt_active,
                mode,
            console_session_mode: matches!(mode, LogFormat::Pretty)
                .then_some(console_session_mode),
                dashboard_snapshot_provider,
            }
        })
    }

    pub fn global() -> &'static OutputManager {
        GLOBAL_OUTPUT_MANAGER
            .get()
            .expect("OutputManager::init_global must be called before OutputManager::global")
    }

    pub fn emit_event(&self, event: OutputEvent) -> io::Result<()> {
        self.tx.send(OutputCommand::Event(event)).map_err(|_| {
            io::Error::new(
                io::ErrorKind::BrokenPipe,
                "output manager worker unavailable",
            )
        })
    }

    pub fn schedule_ready_prompt(&self) -> io::Result<()> {
        self.tx
            .send(OutputCommand::ActivateReadyPrompt)
            .map_err(|_| {
                io::Error::new(
                    io::ErrorKind::BrokenPipe,
                    "output manager worker unavailable",
                )
            })
    }

    pub fn write_ready_prompt(&self) -> io::Result<()> {
        self.ready_prompt_active.store(true, Ordering::Release);
        if matches!(self.mode, LogFormat::Pretty)
            && !matches!(
                self.console_session_mode,
                Some(ConsoleSessionMode::InteractiveDashboard)
            )
        {
            write_prompt()
        } else {
            Ok(())
        }
    }

    pub fn ready_prompt_active(&self) -> bool {
        self.ready_prompt_active.load(Ordering::Acquire)
    }

    pub fn mode(&self) -> LogFormat {
        self.mode
    }

    pub fn console_session_mode(&self) -> Option<ConsoleSessionMode> {
        self.console_session_mode
    }

    pub fn register_dashboard_snapshot_provider(
        &self,
        provider: Arc<dyn DashboardSnapshotProvider>,
    ) {
        if !matches!(self.mode, LogFormat::Pretty) {
            return;
        }

        if let Ok(mut slot) = self.dashboard_snapshot_provider.write() {
            *slot = Some(provider);
        }
    }

    #[allow(dead_code)]
    pub async fn dashboard_snapshot(&self) -> Option<DashboardSnapshot> {
        if !matches!(self.mode, LogFormat::Pretty) {
            return None;
        }

        let provider = self
            .dashboard_snapshot_provider
            .read()
            .ok()
            .and_then(|slot| slot.clone())?;
        Some(provider.snapshot().await)
    }

    pub async fn enter_tui(&self) -> io::Result<()> {
        let (response_tx, response_rx) = tokio::sync::oneshot::channel();
        self.tx
            .send(OutputCommand::EnterTui(response_tx))
            .map_err(|_| {
                io::Error::new(
                    io::ErrorKind::BrokenPipe,
                    "output manager worker unavailable",
                )
            })?;
        response_rx.await.map_err(|_| {
            io::Error::new(
                io::ErrorKind::BrokenPipe,
                "output manager worker unavailable",
            )
        })?
    }

    pub async fn exit_tui(&self) -> io::Result<()> {
        let (response_tx, response_rx) = tokio::sync::oneshot::channel();
        self.tx
            .send(OutputCommand::ExitTui(response_tx))
            .map_err(|_| {
                io::Error::new(
                    io::ErrorKind::BrokenPipe,
                    "output manager worker unavailable",
                )
            })?;
        response_rx.await.map_err(|_| {
            io::Error::new(
                io::ErrorKind::BrokenPipe,
                "output manager worker unavailable",
            )
        })?
    }

    pub async fn dispatch_tui_event(&self, event: TuiEvent) -> io::Result<TuiControlFlow> {
        let (response_tx, response_rx) = tokio::sync::oneshot::channel();
        self.tx
            .send(OutputCommand::TuiEvent {
                event,
                response: response_tx,
            })
            .map_err(|_| {
                io::Error::new(
                    io::ErrorKind::BrokenPipe,
                    "output manager worker unavailable",
                )
            })?;
        response_rx.await.map_err(|_| {
            io::Error::new(
                io::ErrorKind::BrokenPipe,
                "output manager worker unavailable",
            )
        })?
    }

    pub async fn render_tui_if_dirty(&self) -> io::Result<bool> {
        let (response_tx, response_rx) = tokio::sync::oneshot::channel();
        self.tx
            .send(OutputCommand::RenderTui(response_tx))
            .map_err(|_| {
                io::Error::new(
                    io::ErrorKind::BrokenPipe,
                    "output manager worker unavailable",
                )
            })?;
        response_rx.await.map_err(|_| {
            io::Error::new(
                io::ErrorKind::BrokenPipe,
                "output manager worker unavailable",
            )
        })?
    }
}

fn write_rendered_output(mode: LogFormat, rendered: &str) -> io::Result<()> {
    let mut stdout = io::stdout().lock();
    let mut stderr = io::stderr().lock();
    write_rendered_output_to_writers(mode, rendered, &mut stdout, &mut stderr)
}

fn write_rendered_output_to_writers<StdoutWriter, StderrWriter>(
    mode: LogFormat,
    rendered: &str,
    stdout: &mut StdoutWriter,
    stderr: &mut StderrWriter,
) -> io::Result<()>
where
    StdoutWriter: Write,
    StderrWriter: Write,
{
    match mode {
        LogFormat::Pretty => {
            stderr.write_all(rendered.as_bytes())?;
            if !rendered.ends_with('\n') {
                stderr.write_all(b"\n")?;
            }
            stderr.flush()
        }
        LogFormat::Json => {
            stdout.write_all(rendered.as_bytes())?;
            if !rendered.ends_with('\n') {
                stdout.write_all(b"\n")?;
            }
            stdout.flush()
        }
    }
}

fn classify_error_type(message: &str, context: Option<&str>) -> &'static str {
    if message.starts_with("GGUF file not found:") {
        "missing_gguf"
    } else if message.starts_with("Failed to bind to port")
        || context
            .map(|value| value.contains("Address already in use"))
            .unwrap_or(false)
    {
        "bind_failed"
    } else {
        "runtime_error"
    }
}

fn build_fatal_error_event(err: &AnyhowError) -> OutputEvent {
    let message = err.to_string();
    let context = err
        .chain()
        .skip(1)
        .map(ToString::to_string)
        .collect::<Vec<_>>();
    OutputEvent::Error {
        message,
        context: (!context.is_empty()).then(|| context.join(": ")),
    }
}

pub fn emit_fatal_error(err: &AnyhowError) -> io::Result<()> {
    emit_event(build_fatal_error_event(err))
}

pub fn json_mode_enabled() -> bool {
    GLOBAL_OUTPUT_MANAGER
        .get()
        .map(|output_manager| matches!(output_manager.mode(), LogFormat::Json))
        .unwrap_or(false)
}

fn write_prompt() -> io::Result<()> {
    let mut stderr = io::stderr().lock();
    stderr.write_all(b"> ")?;
    stderr.flush()
}

fn dashboard_layout_for_terminal_size(columns: u16, rows: u16) -> DashboardLayoutState {
    let footer_rows = 2usize;
    let join_token_rows = usize::from(PRETTY_TUI_JOIN_TOKEN_PANEL_HEIGHT);
    let requests_rows = 6usize;
    let requests_band_rows = requests_rows + 2;
    // Cap the dashboard height so it stays compact (~32 rows) instead of
    // stretching to fill the entire terminal.
    let max_dashboard_rows = usize::from(rows).min(32);
    let narrow_width_penalty = usize::from(columns < PRETTY_TUI_MIN_DASHBOARD_WIDTH);
    let main_body_rows = max_dashboard_rows
        .saturating_sub(footer_rows + join_token_rows + requests_band_rows)
        .saturating_sub(narrow_width_penalty)
        .max(5);
    let process_body_rows = main_body_rows.saturating_sub(6).max(2);
    let llama_rows = ((process_body_rows.saturating_add(1)) / 3).max(1);
    let webserver_rows = process_body_rows.saturating_sub(llama_rows).max(1);
    let events_rows = main_body_rows.saturating_sub(2).max(1);
    let models_rows = main_body_rows.saturating_sub(2).max(1);
    DashboardLayoutState::new(
        events_rows,
        llama_rows,
        webserver_rows,
        models_rows,
        requests_rows,
    )
}

fn write_tui_enter() -> io::Result<()> {
    let mut stderr = io::stderr().lock();
    write_tui_enter_to_writer(&mut stderr)
}

fn write_tui_exit() -> io::Result<()> {
    let mut stderr = io::stderr().lock();
    write_tui_exit_to_writer(&mut stderr)
}

#[cfg(test)]
fn write_tui_redraw_start_to_writer<W: Write>(writer: &mut W) -> io::Result<()> {
    execute!(writer, Hide, MoveTo(0, 0)).map_err(io::Error::other)
}

pub(crate) fn force_restore_tui_terminal() -> io::Result<()> {
    // Emergency restore path for panic/unwind and failed worker cleanup. This
    // intentionally bypasses the OutputManager so terminal recovery still has a
    // chance if its worker is wedged; SIGKILL cannot be recovered in-process.
    write_tui_exit()
}

fn write_tui_enter_to_writer<W: Write>(writer: &mut W) -> io::Result<()> {
    execute!(
        writer,
        EnterAlternateScreen,
        EnableMouseCapture,
        MoveTo(0, 0),
        Clear(ClearType::All),
        Hide
    )
    .map_err(io::Error::other)
}

fn write_tui_exit_to_writer<W: Write>(writer: &mut W) -> io::Result<()> {
    execute!(
        writer,
        Show,
        DisableMouseCapture,
        LeaveAlternateScreen,
        MoveTo(0, 0),
        Clear(ClearType::All)
    )
    .map_err(io::Error::other)
}

#[cfg(test)]
fn write_tui_frame_to_writer<W: Write>(writer: &mut W, rendered: &str) -> io::Result<()> {
    execute!(writer, MoveTo(0, 0), Clear(ClearType::All)).map_err(io::Error::other)?;
    writer.write_all(rendered.as_bytes())?;
    if !rendered.ends_with('\n') {
        writer.write_all(b"\n")?;
    }
    writer.flush()
}

pub fn emit_event(event: OutputEvent) -> io::Result<()> {
    match GLOBAL_OUTPUT_MANAGER.get() {
        Some(output_manager) => output_manager.emit_event(event),
        None => Ok(()),
    }
}

pub fn interactive_tui_active() -> bool {
    GLOBAL_OUTPUT_MANAGER.get().is_some_and(|output_manager| {
        matches!(output_manager.mode(), LogFormat::Pretty)
            && matches!(
                output_manager.console_session_mode(),
                Some(ConsoleSessionMode::InteractiveDashboard)
            )
    })
}

#[cfg(test)]
impl DashboardState {
    pub fn with_mesh_event_limit(mesh_event_limit: usize) -> Self {
        Self {
            mesh_event_limit: mesh_event_limit.max(1),
            ..Self::default()
        }
    }
}

#[cfg(test)]
impl DashboardFormatter {
    pub fn with_state(state: DashboardState) -> Self {
        Self { state }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_moe_summary() -> MoeSummary {
        MoeSummary {
            experts: 128,
            top_k: 8,
        }
    }

    fn sample_moe_distribution() -> MoeDistributionSummary {
        MoeDistributionSummary {
            leader: "node-7".to_string(),
            active_nodes: 3,
            fallback_nodes: 1,
            shard_index: 0,
            shard_count: 3,
            ranking_source: "shared".to_string(),
            ranking_origin: "cache".to_string(),
            overlap: 2,
            shared_experts: 48,
            unique_experts: 80,
        }
    }

    struct StaticDashboardSnapshotProvider {
        snapshot: DashboardSnapshot,
    }

    impl DashboardSnapshotProvider for StaticDashboardSnapshotProvider {
        fn snapshot(&self) -> DashboardSnapshotFuture<'_> {
            let snapshot = self.snapshot.clone();
            Box::pin(async move { snapshot })
        }
    }

    #[derive(Default)]
    struct DashboardReducerFixture {
        state: DashboardState,
    }

    impl DashboardReducerFixture {
        fn with_snapshot(mut self, snapshot: DashboardSnapshot) -> Self {
            self.state
                .reduce(DashboardAction::SnapshotUpdated(snapshot));
            self
        }

        fn with_events<I>(mut self, events: I) -> Self
        where
            I: IntoIterator<Item = OutputEvent>,
        {
            for event in events {
                self.state.reduce(DashboardAction::OutputEvent(event));
            }
            self
        }

        fn reduce(&mut self, action: DashboardAction) {
            self.state.reduce(action);
        }
    }

    fn sample_process_row(name: &str, port: u16) -> DashboardProcessRow {
        DashboardProcessRow {
            name: name.to_string(),
            backend: "metal".to_string(),
            status: RuntimeStatus::Ready,
            port,
            pid: u32::from(port) + 1000,
        }
    }

    fn sample_endpoint_row(label: &str, port: u16) -> DashboardEndpointRow {
        DashboardEndpointRow {
            label: label.to_string(),
            status: RuntimeStatus::Ready,
            url: format!("http://127.0.0.1:{port}"),
            port,
            pid: None,
        }
    }

    fn sample_model_row(name: &str, port: u16) -> DashboardModelRow {
        DashboardModelRow {
            name: name.to_string(),
            role: Some("host".to_string()),
            status: RuntimeStatus::Ready,
            port: Some(port),
            device: Some("GPU0".to_string()),
            slots: Some(4),
            quantization: Some("Q4_K_M".to_string()),
            ctx_size: Some(8192),
            file_size_gb: Some(24.0),
        }
    }

    fn snapshot_fixture(model_rows: usize, request_buckets: usize) -> DashboardSnapshot {
        DashboardSnapshot {
            llama_process_rows: vec![sample_process_row("llama-server", 8001)],
            webserver_rows: vec![
                sample_endpoint_row("Console", 3131),
                sample_endpoint_row("API", 9337),
            ],
            loaded_model_rows: (0..model_rows)
                .map(|index| sample_model_row(&format!("Model-{index}"), 4000 + index as u16))
                .collect(),
            current_inflight_requests: 3,
            accepted_request_buckets: (0..request_buckets)
                .map(|second_offset| DashboardAcceptedRequestBucket {
                    second_offset: second_offset as u32,
                    accepted_count: second_offset as u64,
                })
                .collect(),
            latency_samples_ms: vec![11, 17, 19, 23],
        }
    }

    fn info_event(message: impl Into<String>) -> OutputEvent {
        OutputEvent::Info {
            message: message.into(),
            context: None,
        }
    }

    fn sample_events_covering_all_variants() -> Vec<OutputEvent> {
        vec![
            OutputEvent::Info {
                message: "mesh is private by default".to_string(),
                context: Some("publish=false".to_string()),
            },
            OutputEvent::Startup {
                version: "v0.64.0".to_string(),
                message: Some("mesh-llm starting".to_string()),
            },
            OutputEvent::NodeIdentity {
                node_id: "node-123".to_string(),
                mesh_id: Some("mesh-abc".to_string()),
            },
            OutputEvent::InviteToken {
                token: "invite-token-123".to_string(),
                mesh_id: "mesh-abc".to_string(),
                mesh_name: None,
            },
            OutputEvent::DiscoveryStarting {
                source: "Nostr re-discovery".to_string(),
            },
            OutputEvent::MeshFound {
                mesh: "mesh-abc".to_string(),
                peers: 7,
                region: Some("us-west".to_string()),
            },
            OutputEvent::DiscoveryJoined {
                mesh: "mesh-abc".to_string(),
            },
            OutputEvent::DiscoveryFailed {
                message: "Could not re-join any mesh".to_string(),
                detail: Some("relay timeout".to_string()),
            },
            OutputEvent::WaitingForPeers {
                detail: Some("waiting for two more peers".to_string()),
            },
            OutputEvent::PassiveMode {
                role: "standby".to_string(),
                status: RuntimeStatus::Starting,
                capacity_gb: Some(24.0),
                models_on_disk: Some(vec!["Qwen2.5-32B".to_string(), "GLM-4.7-Flash".to_string()]),
                detail: Some("No matching model on disk — running as standby GPU node".to_string()),
            },
            OutputEvent::PeerJoined {
                peer_id: "peer-1".to_string(),
                label: Some("lab-gpu-1".to_string()),
            },
            OutputEvent::PeerLeft {
                peer_id: "peer-2".to_string(),
                reason: Some("shutdown".to_string()),
            },
            OutputEvent::ModelQueued {
                model: "Qwen3-32B".to_string(),
            },
            OutputEvent::ModelLoading {
                model: "Qwen3-32B".to_string(),
                source: Some("huggingface".to_string()),
            },
            OutputEvent::ModelLoaded {
                model: "Qwen3-32B".to_string(),
                bytes: Some(24_012_755_755),
                moe: Some(sample_moe_summary()),
            },
            OutputEvent::MoeDetected {
                model: "Qwen3-MoE".to_string(),
                moe: sample_moe_summary(),
                fits_locally: Some(false),
                capacity_gb: Some(24.0),
                model_gb: Some(36.0),
            },
            OutputEvent::MoeDistribution {
                model: "Qwen3-MoE".to_string(),
                moe: sample_moe_summary(),
                distribution: sample_moe_distribution(),
            },
            OutputEvent::MoeStatus {
                model: "Qwen3-MoE".to_string(),
                status: MoeStatusSummary {
                    phase: "standing by".to_string(),
                    detail: "outside active MoE placement".to_string(),
                },
            },
            OutputEvent::MoeAnalysisProgress {
                model: "Qwen3-MoE".to_string(),
                progress: MoeAnalysisProgressSummary {
                    mode: "ranking".to_string(),
                    spinner: "⠋".to_string(),
                    current: 3,
                    total: Some(8),
                    elapsed_secs: 12,
                },
            },
            OutputEvent::HostElected {
                model: "Qwen3-32B".to_string(),
                host: "node-7".to_string(),
                role: Some("host".to_string()),
                capacity_gb: Some(24.0),
            },
            OutputEvent::RpcServerStarting {
                port: 43683,
                device: "CUDA0".to_string(),
                log_path: Some("/tmp/rpc.log".to_string()),
            },
            OutputEvent::RpcReady {
                port: 43683,
                device: "CUDA0".to_string(),
                log_path: Some("/tmp/rpc.log".to_string()),
            },
            OutputEvent::LlamaStarting {
                model: Some("Qwen3-32B".to_string()),
                http_port: 8001,
                ctx_size: Some(8192),
                log_path: Some("/tmp/llama.log".to_string()),
            },
            OutputEvent::LlamaReady {
                model: Some("Qwen3-32B".to_string()),
                port: 8001,
                ctx_size: Some(8192),
                log_path: Some("/tmp/llama.log".to_string()),
            },
            OutputEvent::ModelReady {
                model: "Qwen3-32B".to_string(),
                internal_port: Some(38373),
                role: Some("host".to_string()),
            },
            OutputEvent::MultiModelMode {
                count: 2,
                models: vec!["Qwen3-32B".to_string(), "GLM-4.7-Flash".to_string()],
            },
            OutputEvent::WebserverStarting {
                url: "http://localhost:3131".to_string(),
            },
            OutputEvent::WebserverReady {
                url: "http://localhost:3131".to_string(),
            },
            OutputEvent::ApiStarting {
                url: "http://localhost:9337".to_string(),
            },
            OutputEvent::ApiReady {
                url: "http://localhost:9337".to_string(),
            },
            OutputEvent::RuntimeReady {
                api_url: "http://localhost:9337".to_string(),
                console_url: Some("http://localhost:3131".to_string()),
                api_port: 9337,
                console_port: Some(3131),
                models_count: Some(2),
                pi_command: Some("pi --provider mesh --model Qwen3-32B".to_string()),
                goose_command: Some("GOOSE_PROVIDER=openai OPENAI_HOST=http://localhost:9337 OPENAI_API_KEY=mesh GOOSE_MODEL=Qwen3-32B goose session".to_string()),
            },
            OutputEvent::ModelDownloadProgress {
                label: "Qwen2.5-0.5B-Instruct-Q4_K_M".to_string(),
                file: Some("qwen2.5-0.5b-instruct-q4_k_m.gguf".to_string()),
                downloaded_bytes: Some(245_500_000),
                total_bytes: Some(491_000_000),
                status: ModelProgressStatus::Downloading,
            },
            OutputEvent::RequestRouted {
                model: "Qwen3-32B".to_string(),
                target: "peer-7".to_string(),
            },
            OutputEvent::Warning {
                message: "⚠️ legacy warning prefix still present".to_string(),
                context: Some("model=Qwen3-32B".to_string()),
            },
            OutputEvent::Error {
                message: "❌ llama-server exited".to_string(),
                context: Some("model=Qwen3-32B port=9337".to_string()),
            },
            OutputEvent::Shutdown {
                reason: Some("user requested shutdown".to_string()),
            },
        ]
    }

    #[test]
    fn tui_reducer_focus_cycle_wraps_across_dashboard_panels() {
        let mut fixture = DashboardReducerFixture::default();

        assert_eq!(fixture.state.panel_focus, DashboardPanel::Events);
        assert!(fixture.state.events_follow, "follow should default to ON");

        fixture.reduce(DashboardAction::ToggleEventsFollow);
        assert!(!fixture.state.events_follow);
        fixture.reduce(DashboardAction::ToggleEventsFollow);
        assert!(fixture.state.events_follow);

        let expected_forward_order = [
            DashboardPanel::LlamaCpp,
            DashboardPanel::Webserver,
            DashboardPanel::Models,
            DashboardPanel::Requests,
            DashboardPanel::JoinToken,
            DashboardPanel::Events,
        ];
        for expected_panel in expected_forward_order {
            fixture.reduce(DashboardAction::FocusNextPanel);
            assert_eq!(fixture.state.panel_focus, expected_panel);
        }

        fixture.reduce(DashboardAction::FocusPreviousPanel);
        assert_eq!(fixture.state.panel_focus, DashboardPanel::JoinToken);
    }

    #[test]
    fn tui_reducer_filter_is_case_insensitive_substring() {
        let mut fixture = DashboardReducerFixture::default().with_events(vec![
            OutputEvent::DiscoveryJoined {
                mesh: "Poker-Night".to_string(),
            },
            info_event("background sync complete"),
            OutputEvent::Warning {
                message: "capacity estimate stale".to_string(),
                context: Some("model=Qwen3-32B".to_string()),
            },
        ]);

        fixture.reduce(DashboardAction::FocusNextPanel);
        assert_eq!(fixture.state.panel_focus, DashboardPanel::LlamaCpp);

        fixture.reduce(DashboardAction::StartEventsFilterEdit);
        assert_eq!(fixture.state.panel_focus, DashboardPanel::Events);
        assert!(fixture.state.events_filter.editing);

        for ch in "PoKeR".chars() {
            fixture.reduce(DashboardAction::InsertEventsFilterChar(ch));
        }

        let filtered_events = fixture.state.filtered_mesh_events();
        assert_eq!(filtered_events.len(), 1);
        assert!(filtered_events[0].summary.contains("Poker-Night"));

        fixture.reduce(DashboardAction::BackspaceEventsFilter);
        assert_eq!(fixture.state.events_filter.query, "PoKe");
        assert_eq!(fixture.state.filtered_mesh_events().len(), 1);

        fixture.reduce(DashboardAction::ConfirmEventsFilter);
        assert!(!fixture.state.events_filter.editing);
        assert_eq!(fixture.state.events_filter.query, "PoKe");

        fixture.reduce(DashboardAction::StartEventsFilterEdit);
        fixture.reduce(DashboardAction::CancelEventsFilter);
        assert!(!fixture.state.events_filter.editing);
        assert!(fixture.state.events_filter.query.is_empty());
        assert_eq!(fixture.state.filtered_mesh_events().len(), 3);

        fixture.reduce(DashboardAction::StartEventsFilterEdit);
        for ch in "mesh.*night".chars() {
            fixture.reduce(DashboardAction::InsertEventsFilterChar(ch));
        }
        assert_eq!(fixture.state.filtered_mesh_events().len(), 0);
    }

    #[test]
    fn tui_reducer_filter_matches_visible_event_badges() {
        let mut fixture = DashboardReducerFixture::default().with_events(vec![
            info_event("plain operational marker"),
            info_event("ok heartbeat marker"),
            OutputEvent::Warning {
                message: "capacity stale marker".to_string(),
                context: None,
            },
        ]);

        fixture.reduce(DashboardAction::StartEventsFilterEdit);
        for ch in "INFO".chars() {
            fixture.reduce(DashboardAction::InsertEventsFilterChar(ch));
        }

        let filtered_events = fixture.state.filtered_mesh_events();
        assert_eq!(filtered_events.len(), 1);
        assert_eq!(filtered_events[0].summary, "plain operational marker");
    }

    #[test]
    fn tui_reducer_preserves_scroll_on_resize() {
        let mut fixture =
            DashboardReducerFixture::default().with_snapshot(snapshot_fixture(12, 30));

        fixture.reduce(DashboardAction::FocusNextPanel);
        fixture.reduce(DashboardAction::FocusNextPanel);
        fixture.reduce(DashboardAction::FocusNextPanel);
        assert_eq!(fixture.state.panel_focus, DashboardPanel::Models);

        fixture.reduce(DashboardAction::Resize(DashboardLayoutState::new(
            4, 4, 4, 3, 2,
        )));
        fixture.reduce(DashboardAction::SetPanelSelection {
            panel: DashboardPanel::Models,
            selected_row: Some(5),
        });
        fixture.reduce(DashboardAction::SetPanelScroll {
            panel: DashboardPanel::Models,
            scroll_offset: 4,
        });

        let before_resize = fixture.state.panel_view_state(DashboardPanel::Models);
        assert_eq!(before_resize.selected_row, None);
        assert_eq!(before_resize.scroll_offset, 4);

        fixture.reduce(DashboardAction::Resize(DashboardLayoutState::new(
            6, 4, 4, 5, 2,
        )));

        let after_resize = fixture.state.panel_view_state(DashboardPanel::Models);
        assert_eq!(fixture.state.panel_focus, DashboardPanel::Models);
        assert_eq!(after_resize.selected_row, None);
        assert_eq!(after_resize.scroll_offset, 4);
        assert_eq!(
            after_resize.viewport_rows,
            tui_panel_viewport_rows(DashboardPanel::Models, 5)
        );
    }

    #[test]
    fn tui_reducer_caps_event_history_at_1000() {
        let mut fixture = DashboardReducerFixture::default().with_snapshot(snapshot_fixture(2, 35));

        for index in 0..1005 {
            fixture.reduce(DashboardAction::OutputEvent(info_event(format!(
                "event-{index}"
            ))));
        }

        assert_eq!(fixture.state.mesh_event_limit, 1000);
        assert_eq!(fixture.state.mesh_events.len(), 1000);
        assert_eq!(
            fixture.state.request_history.accepted_request_buckets.len(),
            PRETTY_DASHBOARD_REQUEST_MAX_WINDOW_SECS as usize
        );
        assert!(fixture
            .state
            .mesh_events
            .front()
            .expect("expected oldest retained event")
            .summary
            .contains("event-5"));
        assert!(fixture
            .state
            .mesh_events
            .back()
            .expect("expected newest retained event")
            .summary
            .contains("event-1004"));
    }

    #[test]
    fn tui_events_follow_mode_keeps_latest_row_visible() {
        let mut formatter = InteractiveDashboardFormatter::default();
        formatter
            .state
            .reduce(DashboardAction::Resize(DashboardLayoutState::new(
                4, 2, 2, 2, 2,
            )));

        for index in 0..8 {
            formatter
                .handle_output_event(&info_event(format!("event-{index}")))
                .expect("event render should succeed");
        }

        let before = formatter.state.panel_view_state(DashboardPanel::Events);
        assert!(formatter.state.events_follow);
        assert_eq!(before.selected_row, Some(7));
        assert_eq!(before.scroll_offset, 4);

        formatter
            .handle_output_event(&info_event("event-8"))
            .expect("event render should succeed");

        let after = formatter.state.panel_view_state(DashboardPanel::Events);
        assert!(formatter.state.events_follow);
        assert_eq!(after.selected_row, Some(8));
        assert_eq!(after.scroll_offset, 5);
    }

    #[test]
    fn tui_events_short_list_navigation_keeps_non_follow_anchor() {
        let mut formatter = InteractiveDashboardFormatter::default();
        formatter
            .state
            .reduce(DashboardAction::Resize(DashboardLayoutState::new(
                8, 2, 2, 2, 2,
            )));

        for index in 0..3 {
            formatter
                .handle_output_event(&info_event(format!("event-{index}")))
                .expect("event render should succeed");
        }

        assert!(formatter.state.events_follow);
        formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Char('f')));
        assert!(!formatter.state.events_follow);

        let viewport_rows = formatter
            .state
            .panel_view_state(DashboardPanel::Events)
            .viewport_rows;
        assert!(
            formatter.state.row_count_for_panel(DashboardPanel::Events) < viewport_rows,
            "test must exercise the short-list path"
        );
        let first_event_before = visible_event_rows(&formatter.state, viewport_rows)
            .iter()
            .position(|row| matches!(row, TuiEventRow::Event { .. }))
            .expect("expected at least one event row");

        formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Char('G')));

        let view_after = formatter.state.panel_view_state(DashboardPanel::Events);
        assert_eq!(view_after.selected_row, Some(2));
        assert_eq!(view_after.scroll_offset, 0);
        assert!(
            formatter.state.events_follow,
            "jumping to the end of a short scrollbar list should follow the newest event"
        );
        let first_event_after = visible_event_rows(&formatter.state, viewport_rows)
            .iter()
            .position(|row| matches!(row, TuiEventRow::Event { .. }))
            .expect("expected at least one event row");
        assert!(
            first_event_after >= first_event_before,
            "short scrollbar lists may bottom-anchor when follow is re-enabled, but must not scroll text out of range"
        );
    }

    #[test]
    fn tui_events_short_list_arrow_navigation_disables_follow() {
        let mut formatter = InteractiveDashboardFormatter::default();
        formatter
            .state
            .reduce(DashboardAction::Resize(DashboardLayoutState::new(
                8, 2, 2, 2, 2,
            )));

        for index in 0..3 {
            formatter
                .handle_output_event(&info_event(format!("event-{index}")))
                .expect("event render should succeed");
        }

        assert!(formatter.state.events_follow);
        assert_eq!(
            formatter
                .state
                .panel_view_state(DashboardPanel::Events)
                .selected_row,
            Some(2)
        );

        formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Up));

        assert!(formatter.state.events_follow);
        assert_eq!(
            formatter.state.panel_view_state(DashboardPanel::Events),
            DashboardPanelViewState {
                scroll_offset: 0,
                selected_row: Some(2),
                viewport_rows: 8,
            }
        );

        formatter
            .state
            .reduce(DashboardAction::Resize(DashboardLayoutState::new(
                8, 2, 2, 2, 2,
            )));

        assert_eq!(
            formatter
                .state
                .panel_view_state(DashboardPanel::Events)
                .selected_row,
            Some(2),
            "short scrollbar lists do not move a selected row; arrows only scroll text"
        );
    }

    #[test]
    fn tui_events_pgup_pgdn_and_home_end_navigation() {
        let mut formatter = InteractiveDashboardFormatter::default();
        formatter
            .state
            .reduce(DashboardAction::Resize(DashboardLayoutState::new(
                5, 2, 2, 2, 2,
            )));

        for index in 0..12 {
            formatter
                .handle_output_event(&info_event(format!("event-{index}")))
                .expect("event render should succeed");
        }

        assert!(formatter.state.events_follow);
        assert_eq!(
            formatter.state.panel_view_state(DashboardPanel::Events),
            DashboardPanelViewState {
                scroll_offset: 7,
                selected_row: Some(11),
                viewport_rows: 5,
            }
        );

        formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::PageUp));
        assert!(!formatter.state.events_follow);
        assert_eq!(
            formatter.state.panel_view_state(DashboardPanel::Events),
            DashboardPanelViewState {
                scroll_offset: 3,
                selected_row: Some(11),
                viewport_rows: 5,
            }
        );

        formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::PageDown));
        assert!(formatter.state.events_follow);
        assert_eq!(
            formatter.state.panel_view_state(DashboardPanel::Events),
            DashboardPanelViewState {
                scroll_offset: 7,
                selected_row: Some(11),
                viewport_rows: 5,
            }
        );

        formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Char('g')));
        assert!(!formatter.state.events_follow);
        assert_eq!(
            formatter.state.panel_view_state(DashboardPanel::Events),
            DashboardPanelViewState {
                scroll_offset: 0,
                selected_row: Some(11),
                viewport_rows: 5,
            }
        );

        formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Char('G')));
        assert!(formatter.state.events_follow);
        assert_eq!(
            formatter.state.panel_view_state(DashboardPanel::Events),
            DashboardPanelViewState {
                scroll_offset: 7,
                selected_row: Some(11),
                viewport_rows: 5,
            }
        );
    }

    #[test]
    fn tui_events_filter_persists_across_focus_changes() {
        let mut formatter = InteractiveDashboardFormatter::default();
        formatter
            .handle_output_event(&OutputEvent::DiscoveryJoined {
                mesh: "Poker-Night".to_string(),
            })
            .expect("event render should succeed");
        formatter
            .handle_output_event(&info_event("background sync complete"))
            .expect("event render should succeed");

        formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Tab));
        assert_eq!(formatter.state.panel_focus, DashboardPanel::LlamaCpp);

        formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Char('/')));
        for ch in "poker".chars() {
            formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Char(ch)));
        }
        formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Enter));

        assert_eq!(formatter.state.panel_focus, DashboardPanel::Events);
        assert_eq!(formatter.state.events_filter.query, "poker");
        assert_eq!(formatter.state.filtered_mesh_events().len(), 1);

        formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Tab));
        assert_eq!(formatter.state.panel_focus, DashboardPanel::LlamaCpp);
        assert!(!formatter.state.events_filter.editing);
        assert_eq!(formatter.state.events_filter.query, "poker");
        assert_eq!(formatter.state.filtered_mesh_events().len(), 1);

        formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::BackTab));
        assert_eq!(formatter.state.panel_focus, DashboardPanel::Events);
        assert_eq!(formatter.state.events_filter.query, "poker");
        assert_eq!(formatter.state.filtered_mesh_events().len(), 1);
    }

    #[test]
    fn tui_event_line_uses_compact_timestamp_level_message_layout() {
        let line = event_line(
            &MeshEventState {
                timestamp: "12:34:56".to_string(),
                level: OutputLevel::Info,
                summary: "✅   joined   mesh   poker-night".to_string(),
            },
            80,
        );

        assert_eq!(
            spans_plain_text(&line.spans),
            "12:34:56 OK   joined mesh poker-night"
        );
    }

    fn sample_mesh_event_states(count: usize) -> Vec<MeshEventState> {
        (0..count)
            .map(|index| MeshEventState {
                timestamp: format!("12:34:{index:02}"),
                level: OutputLevel::Info,
                summary: format!("event-{index:02} tdd-scroll-marker"),
            })
            .collect()
    }

    fn render_scrollbar_event_list_widget_snapshot(
        events: &[MeshEventState],
        scroll_offset: usize,
        width: u16,
        height: u16,
    ) -> String {
        let event_refs = events.iter().collect::<Vec<_>>();
        let backend = TestBackend::new(width, height);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal
            .draw(|frame| {
                frame.render_widget(
                    TuiScrollbarEventList {
                        events: &event_refs,
                        empty_message: "(waiting for mesh events)",
                        scroll_offset,
                    },
                    frame.area(),
                );
            })
            .unwrap();
        test_buffer_to_string(terminal.backend().buffer(), width, height)
    }

    fn render_events_panel_with_renderer_snapshot(
        state: &DashboardState,
        renderer: TuiEventListRenderer,
        width: u16,
        height: u16,
    ) -> String {
        let backend = TestBackend::new(width, height);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal
            .draw(|frame| {
                let title_area = Rect {
                    x: 0,
                    y: 0,
                    width,
                    height: 1,
                };
                let body_area = Rect {
                    x: 0,
                    y: 1,
                    width,
                    height: height.saturating_sub(1),
                };
                render_events_panel_with_renderer(frame, state, title_area, body_area, renderer);
            })
            .unwrap();
        test_buffer_to_string(terminal.backend().buffer(), width, height)
    }

    fn test_buffer_to_string(buffer: &ratatui::buffer::Buffer, width: u16, height: u16) -> String {
        let mut lines = Vec::with_capacity(usize::from(height));
        for y in 0..height {
            let mut line = String::new();
            for x in 0..width {
                line.push_str(buffer[(x, y)].symbol());
            }
            lines.push(line.trim_end().to_string());
        }
        lines.join("\n")
    }

    #[test]
    fn tui_scrollbar_event_list_renders_standalone_vertical_slice() {
        let events = sample_mesh_event_states(7);

        let rendered = render_scrollbar_event_list_widget_snapshot(&events, 2, 42, 3);

        assert!(rendered.contains("event-02 tdd-scroll-marker"));
        assert!(rendered.contains("event-03 tdd-scroll-marker"));
        assert!(rendered.contains("event-04 tdd-scroll-marker"));
        assert!(!rendered.contains("event-01 tdd-scroll-marker"));
        assert!(!rendered.contains("event-05 tdd-scroll-marker"));
        assert!(
            rendered.lines().all(|line| !line.contains('─')),
            "new event list should use the vertical scrollbar only: {rendered}"
        );
        assert!(
            rendered
                .lines()
                .any(|line| line.ends_with('│') || line.ends_with('█')),
            "expected a vertical scrollbar in the rightmost column: {rendered}"
        );
    }

    #[test]
    fn tui_scrollbar_event_list_reaches_bottom_at_last_slice() {
        let events = sample_mesh_event_states(7);

        let rendered = render_scrollbar_event_list_widget_snapshot(&events, 4, 42, 3);
        let scrollbar_column: String = rendered
            .lines()
            .map(|line| line.chars().last().unwrap_or(' '))
            .collect();

        assert!(rendered.contains("event-04 tdd-scroll-marker"));
        assert!(rendered.contains("event-05 tdd-scroll-marker"));
        assert!(rendered.contains("event-06 tdd-scroll-marker"));
        assert!(
            scrollbar_column.ends_with('█'),
            "expected scrollbar thumb to reach bottom for final visible slice: {rendered}"
        );
    }

    #[test]
    fn tui_events_panel_can_swap_between_scrollbar_widget_and_legacy_list() {
        let mut formatter = InteractiveDashboardFormatter::default();
        formatter
            .state
            .reduce(DashboardAction::Resize(DashboardLayoutState::new(
                4, 2, 2, 2, 2,
            )));
        for index in 0..6 {
            formatter
                .handle_output_event(&info_event(format!("event-{index:02} swap-marker")))
                .expect("event render should succeed");
        }

        let scrollbar_rendered = render_events_panel_with_renderer_snapshot(
            &formatter.state,
            TuiEventListRenderer::Scrollbar,
            72,
            8,
        );
        let legacy_rendered = render_events_panel_with_renderer_snapshot(
            &formatter.state,
            TuiEventListRenderer::Legacy,
            72,
            8,
        );

        assert!(scrollbar_rendered.contains("event-05 swap-marker"));
        assert!(legacy_rendered.contains("event-05 swap-marker"));
    }

    #[test]
    fn tui_events_scrollbar_arrows_scroll_text_line_by_line() {
        let mut formatter = InteractiveDashboardFormatter::default();
        formatter
            .state
            .reduce(DashboardAction::Resize(DashboardLayoutState::new(
                4, 2, 2, 2, 2,
            )));

        for index in 0..8 {
            formatter
                .handle_output_event(&info_event(format!("event-{index:02} line-scroll-marker")))
                .expect("event render should succeed");
        }

        assert!(formatter.state.events_follow);
        assert_eq!(
            formatter
                .state
                .panel_view_state(DashboardPanel::Events)
                .scroll_offset,
            4
        );

        formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Up));
        assert!(!formatter.state.events_follow);
        assert_eq!(
            formatter
                .state
                .panel_view_state(DashboardPanel::Events)
                .scroll_offset,
            3,
            "Up should scroll the event text up by exactly one line"
        );

        formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Up));
        assert_eq!(
            formatter
                .state
                .panel_view_state(DashboardPanel::Events)
                .scroll_offset,
            2,
            "a second Up press should scroll one more line"
        );

        formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Down));
        assert_eq!(
            formatter
                .state
                .panel_view_state(DashboardPanel::Events)
                .scroll_offset,
            3,
            "Down should scroll the event text down by exactly one line"
        );

        formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Down));
        assert_eq!(
            formatter
                .state
                .panel_view_state(DashboardPanel::Events)
                .scroll_offset,
            4
        );
        assert!(
            formatter.state.events_follow,
            "scrolling down to the newest event should re-enable follow mode"
        );
    }

    #[test]
    fn tui_events_fewer_items_than_viewport_scroll_offset_is_zero() {
        let mut formatter = InteractiveDashboardFormatter::default();
        formatter
            .state
            .reduce(DashboardAction::Resize(DashboardLayoutState::new(
                8, 2, 2, 2, 2,
            )));

        for index in 0..3 {
            formatter
                .handle_output_event(&info_event(format!("event-{index}")))
                .expect("event render should succeed");
        }

        let view = formatter.state.panel_view_state(DashboardPanel::Events);
        assert_eq!(view.scroll_offset, 0);
        assert_eq!(view.viewport_rows, 8);

        let rows = visible_event_rows(&formatter.state, view.viewport_rows);
        let event_count = rows
            .iter()
            .filter(|r| matches!(r, TuiEventRow::Event { .. }))
            .count();
        assert_eq!(event_count, 3);
    }

    #[test]
    fn tui_events_overflow_scroll_offset_tracks_last_event() {
        let mut formatter = InteractiveDashboardFormatter::default();
        formatter
            .state
            .reduce(DashboardAction::Resize(DashboardLayoutState::new(
                4, 2, 2, 2, 2,
            )));

        for index in 0..10 {
            formatter
                .handle_output_event(&info_event(format!("event-{index}")))
                .expect("event render should succeed");
        }

        assert!(formatter.state.events_follow);
        let view = formatter.state.panel_view_state(DashboardPanel::Events);
        assert_eq!(view.scroll_offset, 6);
        assert_eq!(view.selected_row, Some(9));
    }

    #[test]
    fn tui_events_manual_scroll_up_disables_follow() {
        let mut formatter = InteractiveDashboardFormatter::default();
        formatter
            .state
            .reduce(DashboardAction::Resize(DashboardLayoutState::new(
                4, 2, 2, 2, 2,
            )));

        for index in 0..8 {
            formatter
                .handle_output_event(&info_event(format!("event-{index}")))
                .expect("event render should succeed");
        }

        assert!(formatter.state.events_follow);
        let view_before = formatter.state.panel_view_state(DashboardPanel::Events);
        assert_eq!(view_before.scroll_offset, 4);
        assert_eq!(view_before.selected_row, Some(7));

        formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Up));

        assert!(!formatter.state.events_follow);
        let view_after = formatter.state.panel_view_state(DashboardPanel::Events);
        assert_eq!(
            view_after.selected_row,
            Some(7),
            "Up should not move a selected event row in scrollbar mode"
        );
        assert_eq!(
            view_after.scroll_offset, 3,
            "Up should scroll the event text by exactly one line"
        );
    }

    #[test]
    fn tui_events_up_repaints_actual_viewport_without_top_pinning() {
        let mut formatter = InteractiveDashboardFormatter::default();
        formatter
            .state
            .reduce(DashboardAction::Resize(DashboardLayoutState::new(
                5, 2, 2, 2, 2,
            )));

        for index in 0..12 {
            formatter
                .handle_output_event(&info_event(format!("event-{index:02} no-pin-marker")))
                .expect("event render should succeed");
        }

        let backend = TestBackend::new(90, 14);
        let mut terminal = Terminal::new(backend).expect("test backend should initialize");
        let title_area = Rect::new(0, 0, 90, 1);
        let body_area = Rect::new(0, 1, 90, 12);
        terminal
            .draw(|frame| render_events_panel(frame, &formatter.state, title_area, body_area))
            .expect("initial event panel render should succeed");

        formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Up));
        assert!(!formatter.state.events_follow);

        terminal
            .draw(|frame| render_events_panel(frame, &formatter.state, title_area, body_area))
            .expect("up-arrow event panel render should succeed");

        let buffer = terminal.backend().buffer();
        let rendered_lines: Vec<String> = (0..14)
            .map(|y| {
                (0..90)
                    .map(|x| buffer[(x, y)].symbol())
                    .collect::<String>()
                    .trim_end()
                    .to_string()
            })
            .collect();
        let rendered = rendered_lines.join("\n");

        assert!(
            rendered.contains("event-01 no-pin-marker"),
            "event renderer should use the actual panel height, not the stale state viewport: {rendered_lines:?}"
        );
        assert!(
            rendered.contains("event-11 no-pin-marker"),
            "latest row should remain visible after one Up press: {rendered_lines:?}"
        );
        assert!(
            !rendered.contains("event-00 no-pin-marker"),
            "top row should scroll out instead of pinning to the panel top: {rendered_lines:?}"
        );
        for index in 1..=11 {
            let marker = format!("event-{index:02} no-pin-marker");
            assert_eq!(
                rendered.matches(&marker).count(),
                1,
                "event rows should be painted exactly once after Up, without duplicated stale text: {rendered_lines:?}"
            );
        }
    }

    #[test]
    fn tui_events_scroll_repaints_long_rows_cleanly() {
        let mut formatter = InteractiveDashboardFormatter::default();
        formatter
            .state
            .reduce(DashboardAction::Resize(DashboardLayoutState::new(
                2, 1, 1, 1, 1,
            )));

        formatter
            .handle_output_event(&info_event("short pre-scroll"))
            .expect("event render should succeed");
        formatter
            .handle_output_event(&info_event(
                "this row is intentionally long so scrolling has to repaint cleanly unique-tail-marker",
            ))
            .expect("event render should succeed");
        formatter
            .handle_output_event(&info_event("short post-scroll"))
            .expect("event render should succeed");

        let initial_state = formatter.state.clone();
        let mut scrolled_state = initial_state.clone();
        scrolled_state.events_follow = false;
        let events_view = scrolled_state.panel_view_state_mut(DashboardPanel::Events);
        events_view.scroll_offset = 0;
        events_view.selected_row = Some(0);

        let backend = TestBackend::new(72, 16);
        let mut terminal = Terminal::new(backend).expect("test backend should initialize");
        terminal
            .draw(|frame| render_tui_frame(frame, &initial_state))
            .expect("initial frame render should succeed");
        terminal
            .draw(|frame| render_tui_frame(frame, &scrolled_state))
            .expect("scrolled frame render should succeed");

        let buffer = terminal.backend().buffer();
        let rendered_lines: Vec<String> = (0..16)
            .map(|y| {
                (0..72)
                    .map(|x| buffer[(x, y)].symbol())
                    .collect::<String>()
                    .trim_end()
                    .to_string()
            })
            .collect();
        let scrolled_event_line = rendered_lines
            .iter()
            .find(|line| line.contains("short pr"))
            .unwrap_or_else(|| {
                panic!("expected the scrolled short event to be visible: {rendered_lines:?}")
            });
        assert!(
            !scrolled_event_line.contains("unique-tail-marker"),
            "expected long event text to be truncated before repaint: {scrolled_event_line}"
        );
        assert!(
            rendered_lines
                .iter()
                .all(|line| !line.contains("unique-tail-marker")),
            "expected no stale long-event text after scrolling: {rendered_lines:?}"
        );
    }

    #[test]
    fn tui_events_filter_empty_state_repaints_over_previous_rows() {
        let mut formatter = InteractiveDashboardFormatter::default();
        formatter
            .state
            .reduce(DashboardAction::Resize(DashboardLayoutState::new(
                4, 2, 2, 2, 2,
            )));

        formatter
            .handle_output_event(&info_event("sticky-filter-marker before-filter"))
            .expect("event render should succeed");
        formatter
            .handle_output_event(&info_event("another visible row before-filter"))
            .expect("event render should succeed");

        let backend = TestBackend::new(80, 18);
        let mut terminal = Terminal::new(backend).expect("test backend should initialize");
        terminal
            .draw(|frame| render_tui_frame(frame, &formatter.state))
            .expect("initial frame render should succeed");

        formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Char('/')));
        for ch in "zzzz".chars() {
            formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Char(ch)));
        }

        assert_eq!(formatter.state.filtered_mesh_events().len(), 0);
        terminal
            .draw(|frame| render_tui_frame(frame, &formatter.state))
            .expect("filtered frame render should succeed");

        let buffer = terminal.backend().buffer();
        let rendered_lines: Vec<String> = (0..18)
            .map(|y| {
                (0..80)
                    .map(|x| buffer[(x, y)].symbol())
                    .collect::<String>()
                    .trim_end()
                    .to_string()
            })
            .collect();
        assert!(
            rendered_lines
                .iter()
                .any(|line| line.contains("no events match")),
            "expected filtered empty-state message: {rendered_lines:?}"
        );
        assert!(
            rendered_lines
                .iter()
                .all(|line| !line.contains("sticky-filter-marker")),
            "expected filtered empty state to repaint over stale event rows: {rendered_lines:?}"
        );
    }

    #[test]
    fn tui_events_live_filter_repaints_to_matching_badge_rows() {
        let mut formatter = InteractiveDashboardFormatter::default();
        formatter
            .state
            .reduce(DashboardAction::Resize(DashboardLayoutState::new(
                4, 2, 2, 2, 2,
            )));

        formatter
            .handle_output_event(&info_event("plain operational live-filter-marker"))
            .expect("event render should succeed");
        formatter
            .handle_output_event(&info_event("ok heartbeat stale-ok-marker"))
            .expect("event render should succeed");
        formatter
            .handle_output_event(&OutputEvent::Warning {
                message: "capacity stale-warn-marker".to_string(),
                context: None,
            })
            .expect("event render should succeed");

        let backend = TestBackend::new(80, 18);
        let mut terminal = Terminal::new(backend).expect("test backend should initialize");
        terminal
            .draw(|frame| render_tui_frame(frame, &formatter.state))
            .expect("initial frame render should succeed");

        formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Char('/')));
        for ch in "info".chars() {
            formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Char(ch)));
        }

        assert_eq!(formatter.state.filtered_mesh_events().len(), 1);
        terminal
            .draw(|frame| render_tui_frame(frame, &formatter.state))
            .expect("filtered frame render should succeed");

        let buffer = terminal.backend().buffer();
        let rendered_lines: Vec<String> = (0..18)
            .map(|y| {
                (0..80)
                    .map(|x| buffer[(x, y)].symbol())
                    .collect::<String>()
                    .trim_end()
                    .to_string()
            })
            .collect();
        assert!(
            rendered_lines
                .iter()
                .any(|line| line.contains("INFO plain operati")),
            "expected INFO badge row to remain visible: {rendered_lines:?}"
        );
        assert!(
            rendered_lines.iter().all(
                |line| !line.contains("stale-ok-marker") && !line.contains("stale-warn-marker")
            ),
            "expected non-matching rows to be repainted away: {rendered_lines:?}"
        );
    }

    #[test]
    fn tui_events_snapshot_preserves_timestamp_readability() {
        let mut formatter = InteractiveDashboardFormatter::default();
        formatter
            .state
            .reduce(DashboardAction::Resize(DashboardLayoutState::new(
                4, 2, 2, 2, 2,
            )));
        formatter
            .handle_output_event(&OutputEvent::DiscoveryJoined {
                mesh: "poker-night".to_string(),
            })
            .expect("event render should succeed");

        let rendered = render_tui_events_snapshot(&formatter.state, 48, 20);
        let event_line = rendered
            .lines()
            .find(|line| line.contains("joined mesh poker-night"))
            .expect("expected rendered mesh event line");
        let timestamp = event_line
            .split_whitespace()
            .find(|token| token.len() == 8 && token.chars().nth(2) == Some(':'))
            .expect("expected timestamp token");
        assert_hh_mm_ss(timestamp);
        assert!(
            event_line.contains(" OK   joined mesh poker-night"),
            "expected compact log row in {event_line}"
        );
        assert!(event_line.contains("joined mesh poker-night"));
        assert!(!event_line.contains("✅"));
    }

    #[test]
    fn tui_list_scrollbar_layout_reserves_one_column_gutter_on_overflow() {
        let inner_area = Rect::new(12, 4, 18, 5);

        assert_eq!(
            tui_list_scrollbar_layout(inner_area, 9, 5),
            TuiListScrollbarLayout {
                list_area: Rect::new(12, 4, 17, 5),
                scrollbar_area: Some(Rect::new(29, 4, 1, 5)),
            }
        );
        assert_eq!(
            tui_list_scrollbar_layout(inner_area, 5, 5),
            TuiListScrollbarLayout {
                list_area: inner_area,
                scrollbar_area: None,
            }
        );
    }

    fn assert_hh_mm_ss(text: &str) {
        assert_eq!(text.len(), 8, "timestamp should be HH:MM:SS, got {text}");
        for (index, ch) in text.chars().enumerate() {
            match index {
                2 | 5 => assert_eq!(ch, ':', "timestamp should use colon separators: {text}"),
                _ => assert!(
                    ch.is_ascii_digit(),
                    "timestamp should contain digits: {text}"
                ),
            }
        }
    }

    fn render_tui_frame_snapshot(state: &DashboardState, width: u16, height: u16) -> String {
        let backend = ratatui::backend::TestBackend::new(width, height);
        let mut terminal = Terminal::new(backend).expect("test backend should initialize");
        terminal
            .draw(|frame| render_tui_frame(frame, state))
            .expect("frame render should succeed");
        let buffer = terminal.backend().buffer();
        let mut lines = Vec::with_capacity(usize::from(height));
        for y in 0..height {
            let mut line = String::new();
            for x in 0..width {
                line.push_str(buffer[(x, y)].symbol());
            }
            lines.push(line.trim_end().to_string());
        }
        lines.join("\n")
    }

    fn render_tui_frame_snapshot_with_buffer(
        state: &DashboardState,
        width: u16,
        height: u16,
    ) -> (String, ratatui::buffer::Buffer) {
        let backend = ratatui::backend::TestBackend::new(width, height);
        let mut terminal = Terminal::new(backend).expect("test backend should initialize");
        terminal
            .draw(|frame| render_tui_frame(frame, state))
            .expect("frame render should succeed");
        let buffer = terminal.backend().buffer().clone();
        let mut lines = Vec::with_capacity(usize::from(height));
        for y in 0..height {
            let mut line = String::new();
            for x in 0..width {
                line.push_str(buffer[(x, y)].symbol());
            }
            lines.push(line.trim_end().to_string());
        }
        (lines.join("\n"), buffer)
    }

    fn find_rendered_line<'a>(rendered: &'a str, needle: &str) -> (usize, &'a str) {
        rendered
            .lines()
            .enumerate()
            .find(|(_, line)| line.contains(needle))
            .unwrap_or_else(|| panic!("expected rendered line containing {needle:?}\n{rendered}"))
    }

    fn find_rendered_line_after<'a>(
        rendered: &'a str,
        start_index: usize,
        needle: &str,
    ) -> (usize, &'a str) {
        rendered
            .lines()
            .enumerate()
            .skip(start_index.saturating_add(1))
            .find(|(_, line)| line.contains(needle))
            .unwrap_or_else(|| {
                panic!(
                    "expected rendered line containing {needle:?} after index {start_index}\n{rendered}"
                )
            })
    }

    fn requests_inner_area(state: &DashboardState, width: u16, height: u16) -> Rect {
        let areas = tui_layout(Rect::new(0, 0, width, height), state);
        tui_panel_block(state, DashboardPanel::Requests)
            .inner(combine_panel_rect(areas.requests.0, areas.requests.1))
    }

    fn request_graph_visible_row_count(buffer: &ratatui::buffer::Buffer, area: Rect) -> usize {
        (area.y.saturating_add(1)..area.bottom())
            .filter(|&y| {
                (area.x..area.right()).any(|x| {
                    let symbol = buffer[(x, y)].symbol().chars().next();
                    matches!(symbol, Some('·' | '─')) || symbol.is_some_and(is_braille_bar_symbol)
                })
            })
            .count()
    }

    fn request_graph_contains_bars(buffer: &ratatui::buffer::Buffer, area: Rect) -> bool {
        (area.y.saturating_add(1)..area.bottom()).any(|y| {
            (area.x..area.right()).any(|x| {
                buffer[(x, y)]
                    .symbol()
                    .chars()
                    .next()
                    .is_some_and(is_braille_bar_symbol)
            })
        })
    }

    fn is_braille_bar_symbol(ch: char) -> bool {
        matches!(ch as u32, 0x2801..=0x28ff)
    }

    fn request_graph_contains_guides(buffer: &ratatui::buffer::Buffer, area: Rect) -> bool {
        (area.y.saturating_add(1)..area.bottom()).any(|y| {
            (area.x..area.right())
                .any(|x| matches!(buffer[(x, y)].symbol().chars().next(), Some('·' | '─')))
        })
    }

    #[test]
    fn tui_layout_uses_join_token_band_with_nested_process_tables() {
        let mut state = DashboardState::default();
        state.reduce(DashboardAction::Resize(dashboard_layout_for_terminal_size(
            120, 24,
        )));

        let areas = tui_layout(Rect::new(0, 0, 120, 24), &state);

        assert_eq!(
            areas.join_token_panel.y,
            areas.loading.map_or(0, |area| area.bottom())
        );
        assert_eq!(areas.join_token_panel.width, 120);
        assert_eq!(
            areas.join_token_panel.height,
            PRETTY_TUI_JOIN_TOKEN_PANEL_HEIGHT
        );
        assert!(areas.join_token_copy_button.x > areas.join_token_panel.x);
        assert_eq!(areas.join_token_copy_button.y, areas.join_token_panel.y + 2);
        assert_eq!(
            areas.join_token_copy_button.right(),
            areas
                .join_token_panel
                .right()
                .saturating_sub(1)
                .saturating_sub(PRETTY_TUI_JOIN_TOKEN_HORIZONTAL_PADDING)
        );
        assert_eq!(
            join_token_text_area(areas.join_token_panel, areas.join_token_copy_button).x,
            areas
                .join_token_panel
                .x
                .saturating_add(1)
                .saturating_add(PRETTY_TUI_JOIN_TOKEN_HORIZONTAL_PADDING)
        );
        assert_eq!(
            areas.main_body.y,
            areas.join_token_panel.y + areas.join_token_panel.height
        );
        assert_eq!(
            areas.requests.0.y,
            areas.main_body.y + areas.main_body.height
        );
        assert_eq!(
            areas.requests.1.y,
            areas.requests.0.y + areas.requests.0.height
        );
        assert_eq!(
            areas.status_bar.y,
            areas.requests.1.y + areas.requests.1.height
        );
        assert_eq!(areas.status_bar.height, 1);
        assert_eq!(areas.events.0.y, areas.main_body.y);
        assert!(areas.processes.x > areas.events.0.x);
        assert!(areas.models.0.x > areas.processes.x);
        let events_inner = tui_panel_block(&state, DashboardPanel::Events)
            .inner(combine_panel_rect(areas.events.0, areas.events.1));
        let models_inner = tui_panel_block(&state, DashboardPanel::Models)
            .inner(combine_panel_rect(areas.models.0, areas.models.1));
        let llama_inner = tui_panel_block(&state, DashboardPanel::LlamaCpp).inner(
            combine_panel_rect(areas.llama_processes.0, areas.llama_processes.1),
        );
        let webserver_inner = tui_panel_block(&state, DashboardPanel::Webserver).inner(
            combine_panel_rect(areas.webserver_processes.0, areas.webserver_processes.1),
        );
        let requests_inner = tui_panel_block(&state, DashboardPanel::Requests)
            .inner(combine_panel_rect(areas.requests.0, areas.requests.1));
        assert_eq!(
            events_inner.height as usize,
            state.panel_layout.rows_for(DashboardPanel::Events)
        );
        assert_eq!(
            models_inner.height as usize,
            state.panel_layout.rows_for(DashboardPanel::Models)
        );
        assert_eq!(
            areas.llama_processes.0.y,
            tui_processes_block(&state).inner(areas.processes).y
        );
        assert_eq!(
            areas.llama_processes.1.y,
            areas.llama_processes.0.y + areas.llama_processes.0.height
        );
        assert_eq!(
            areas.webserver_processes.0.y,
            combine_panel_rect(areas.llama_processes.0, areas.llama_processes.1).bottom()
        );
        assert_eq!(
            areas.webserver_processes.1.y,
            areas.webserver_processes.0.y + areas.webserver_processes.0.height
        );
        assert_eq!(
            llama_inner.height as usize,
            state.panel_layout.rows_for(DashboardPanel::LlamaCpp)
        );
        assert_eq!(
            webserver_inner.height as usize,
            state.panel_layout.rows_for(DashboardPanel::Webserver)
        );
        assert_eq!(state.panel_layout.rows_for(DashboardPanel::LlamaCpp), 1);
        assert_eq!(state.panel_layout.rows_for(DashboardPanel::Webserver), 2);
        assert_eq!(
            requests_inner.height as usize,
            state.panel_layout.rows_for(DashboardPanel::Requests)
        );
    }

    #[test]
    fn tui_main_columns_pin_events_and_split_remaining_width() {
        let mut state = DashboardState::default();
        state.reduce(DashboardAction::Resize(dashboard_layout_for_terminal_size(
            121, 24,
        )));

        let areas = tui_layout(Rect::new(0, 0, 121, 24), &state);
        let events_width = combine_panel_rect(areas.events.0, areas.events.1).width;
        let processes_width = areas.processes.width;
        let models_width = combine_panel_rect(areas.models.0, areas.models.1).width;
        let expected_events_width = areas
            .main_body
            .width
            .saturating_mul(PRETTY_TUI_EVENTS_COLUMN_PERCENT)
            / 100;

        assert!(
            events_width.abs_diff(expected_events_width) <= 1,
            "Mesh Events should stay at roughly {PRETTY_TUI_EVENTS_COLUMN_PERCENT}% of the main body"
        );
        assert!(
            processes_width.abs_diff(models_width) <= 1,
            "Loaded Models and Processes should split the remaining width evenly"
        );
        assert_eq!(
            events_width
                .saturating_add(processes_width)
                .saturating_add(models_width),
            areas.main_body.width
        );
    }

    #[test]
    fn tui_layout_bottom_anchors_dashboard_with_top_slack() {
        let mut state = DashboardState::default();
        state.reduce(DashboardAction::Resize(dashboard_layout_for_terminal_size(
            120, 24,
        )));

        let area = Rect::new(0, 0, 120, 48);
        let areas = tui_layout(area, &state);

        assert!(
            areas.loading.is_some(),
            "expected unused top space above dashboard"
        );
        assert_eq!(areas.status_bar.bottom(), area.bottom());
        assert!(
            areas.main_body.y > area.y,
            "dashboard should sit at the bottom"
        );
    }

    #[test]
    fn tui_band_heights_never_exceed_terminal_budget() {
        let mut state = DashboardState::default();
        state.reduce(DashboardAction::Resize(dashboard_layout_for_terminal_size(
            120, 12,
        )));

        let area = Rect::new(0, 0, 120, 12);
        let band_heights = tui_band_heights(area, &state);
        let areas = tui_layout(area, &state);
        let requests_inner = tui_panel_block(&state, DashboardPanel::Requests)
            .inner(combine_panel_rect(areas.requests.0, areas.requests.1));

        assert_eq!(
            band_heights
                .join_token
                .saturating_add(band_heights.main_body)
                .saturating_add(band_heights.requests)
                .saturating_add(band_heights.status),
            area.height,
            "expected top-level bands to fit the frame budget without overlapping pane borders"
        );
        assert_eq!(areas.status_bar.bottom(), area.bottom());
        assert!(
            requests_inner.height >= 3,
            "expected summary + at least two graph rows in constrained layout"
        );
    }

    #[test]
    fn tui_invite_token_event_populates_join_token_panel() {
        let mut state = DashboardState::default();
        state.reduce(DashboardAction::OutputEvent(OutputEvent::InviteToken {
            token: "mesh-invite-token-123".to_string(),
            mesh_id: "mesh-alpha".to_string(),
            mesh_name: None,
        }));

        let join_token = state
            .join_token
            .as_ref()
            .expect("invite token event should populate dashboard join token state");
        assert_eq!(join_token.token, "mesh-invite-token-123");
        assert_eq!(join_token.mesh_id, "mesh-alpha");
        assert_eq!(join_token.copy_status, DashboardJoinTokenCopyStatus::Idle);

        let rendered = render_tui_frame_snapshot(&state, 120, 24);
        let (join_index, _) = find_rendered_line(&rendered, "Join Token");
        let (events_index, _) = find_rendered_line(&rendered, "Mesh Events");
        assert!(
            join_index < events_index,
            "join token panel should render above existing dashboard panels\n{rendered}"
        );
        assert!(rendered.contains("mesh-invite-token-123"));
        assert!(rendered.contains("Copy"));

        let lines: Vec<&str> = rendered.lines().collect();
        assert!(
            lines[join_index.saturating_add(1)]
                .trim_matches(|ch| ch == '│' || ch == ' ')
                .is_empty(),
            "join token panel should leave one blank body row above the token\n{rendered}"
        );
        assert!(
            lines[join_index.saturating_add(3)]
                .trim_matches(|ch| ch == '│' || ch == ' ')
                .is_empty(),
            "join token panel should leave one blank body row below the token\n{rendered}"
        );
    }

    #[test]
    fn tui_join_token_copy_button_hit_test_uses_latest_resize() {
        let mut state = DashboardState::default();
        state.apply_tui_event(TuiEvent::Resize {
            columns: 120,
            rows: 24,
        });
        state.reduce(DashboardAction::OutputEvent(OutputEvent::InviteToken {
            token: "mesh-invite-token-123".to_string(),
            mesh_id: "mesh-alpha".to_string(),
            mesh_name: None,
        }));
        let areas = tui_layout(Rect::new(0, 0, 120, 24), &state);

        assert!(state.join_token_copy_button_contains(
            areas.join_token_copy_button.x,
            areas.join_token_copy_button.y
        ));
        assert!(!state.join_token_copy_button_contains(0, 0));
    }

    #[test]
    fn tui_join_token_is_selectable_with_backtab_and_mouse() {
        let mut state = DashboardState::default();
        state.apply_tui_event(TuiEvent::Resize {
            columns: 120,
            rows: 24,
        });
        assert_eq!(state.panel_focus, DashboardPanel::Events);

        state.apply_tui_event(TuiEvent::Key(TuiKeyEvent::BackTab));
        assert_eq!(state.panel_focus, DashboardPanel::JoinToken);

        state.panel_focus = DashboardPanel::Events;
        let areas = tui_layout(Rect::new(0, 0, 120, 24), &state);
        state.apply_tui_event(TuiEvent::MouseDown {
            column: areas.join_token_panel.x.saturating_add(1),
            row: areas.join_token_panel.y.saturating_add(1),
        });
        assert_eq!(state.panel_focus, DashboardPanel::JoinToken);

        let rendered = render_tui_frame_snapshot(&state, 120, 24);
        assert!(
            rendered.contains("▶ Join Token"),
            "focused join-token panel should use the standard focus marker\n{rendered}"
        );
    }

    #[test]
    fn tui_join_token_scrolls_horizontally_with_left_right_keys() {
        let mut state = DashboardState::default();
        state.apply_tui_event(TuiEvent::Resize {
            columns: 48,
            rows: 24,
        });
        state.reduce(DashboardAction::OutputEvent(OutputEvent::InviteToken {
            token: "mesh-invite-token-abcdefghijklmnopqrstuvwxyz-0123456789".to_string(),
            mesh_id: "mesh-alpha".to_string(),
            mesh_name: None,
        }));
        state.panel_focus = DashboardPanel::JoinToken;

        assert_eq!(
            state
                .panel_view_state(DashboardPanel::JoinToken)
                .scroll_offset,
            0
        );
        state.apply_tui_event(TuiEvent::Key(TuiKeyEvent::Right));
        assert_eq!(
            state
                .panel_view_state(DashboardPanel::JoinToken)
                .scroll_offset,
            1
        );
        state.apply_tui_event(TuiEvent::Key(TuiKeyEvent::Left));
        assert_eq!(
            state
                .panel_view_state(DashboardPanel::JoinToken)
                .scroll_offset,
            0
        );

        state.apply_tui_event(TuiEvent::Key(TuiKeyEvent::Char('G')));
        let view = state.panel_view_state(DashboardPanel::JoinToken);
        assert!(
            view.scroll_offset > 0,
            "G should jump to the end of the horizontally scrollable token"
        );
        state.apply_tui_event(TuiEvent::Key(TuiKeyEvent::Char('g')));
        assert_eq!(
            state
                .panel_view_state(DashboardPanel::JoinToken)
                .scroll_offset,
            0
        );
    }

    #[test]
    fn tui_join_token_status_renders_on_right_title_bar() {
        let mut state = DashboardState::default();
        state.reduce(DashboardAction::OutputEvent(OutputEvent::InviteToken {
            token: "mesh-invite-token-123".to_string(),
            mesh_id: "mesh-alpha".to_string(),
            mesh_name: None,
        }));
        state.reduce(DashboardAction::SetJoinTokenCopyStatus(
            DashboardJoinTokenCopyStatus::Copied,
        ));

        let rendered = render_tui_frame_snapshot(&state, 120, 24);
        let (_, join_title_line) = find_rendered_line(&rendered, "Join Token");
        let mesh_index = join_title_line
            .find("mesh=mesh-alpha")
            .expect("left title should include mesh id");
        let copied_index = join_title_line
            .rfind("copied")
            .expect("right title should include copy status");
        assert!(
            mesh_index < 40,
            "mesh id should stay near the left title bar"
        );
        assert!(
            copied_index > 90,
            "copy status should be aligned toward the far right title bar: {join_title_line:?}"
        );
    }

    #[test]
    fn tui_join_token_title_includes_mesh_name_when_available() {
        let mut state = DashboardState::default();
        state.reduce(DashboardAction::OutputEvent(OutputEvent::InviteToken {
            token: "mesh-invite-token-123".to_string(),
            mesh_id: "abcd1230".to_string(),
            mesh_name: Some("mymesh".to_string()),
        }));

        let title = join_token_panel_left_title(&state, ' ');

        assert!(title.contains("mesh=mymesh (abcd1230)"));
    }

    #[test]
    fn tui_join_token_title_uses_mesh_id_without_name() {
        let mut state = DashboardState::default();
        state.reduce(DashboardAction::OutputEvent(OutputEvent::InviteToken {
            token: "mesh-invite-token-123".to_string(),
            mesh_id: "abcde1230".to_string(),
            mesh_name: None,
        }));

        let title = join_token_panel_left_title(&state, ' ');

        assert!(title.contains("mesh=abcde1230"));
        assert!(!title.contains('('));
    }

    #[test]
    fn tui_frame_clears_stale_join_token_rows_between_draws() {
        let mut state = DashboardState::default();
        state.reduce(DashboardAction::OutputEvent(OutputEvent::InviteToken {
            token: "mesh-invite-token-123".to_string(),
            mesh_id: "mesh-alpha".to_string(),
            mesh_name: None,
        }));

        let backend = ratatui::backend::TestBackend::new(120, 24);
        let mut terminal = Terminal::new(backend).expect("test backend should initialize");
        terminal
            .draw(|frame| render_tui_frame(frame, &state))
            .expect("initial frame render should succeed");

        terminal
            .draw(|frame| {
                frame.render_widget(
                    Paragraph::new(
                        "stale Join Token mesh=mesh-alpha token mesh-invite-token-123 Copy",
                    ),
                    Rect::new(0, 0, 120, 1),
                );
            })
            .expect("stale frame render should succeed");

        let loading_state = DashboardState {
            model_progress: Some(ModelProgressState {
                label: "qwen2.5".to_string(),
                file: Some("qwen.gguf".to_string()),
                downloaded_bytes: Some(1),
                total_bytes: Some(10),
                status: ModelProgressStatus::Downloading,
            }),
            ..DashboardState::default()
        };

        terminal
            .draw(|frame| render_tui_frame(frame, &loading_state))
            .expect("loading frame render should succeed");

        let buffer = terminal.backend().buffer();
        let mut rendered = String::new();
        for y in 0..24 {
            for x in 0..120 {
                rendered.push_str(buffer[(x, y)].symbol());
            }
            rendered.push('\n');
        }

        assert!(
            !rendered.contains("stale Join Token"),
            "full-frame redraw should clear stale join-token rows from previous frames\n{rendered}"
        );
        assert!(
            !rendered.contains("mesh-invite-token-123"),
            "full-frame redraw should clear stale token text from previous frames\n{rendered}"
        );
    }

    #[test]
    fn tui_process_tables_render_empty_states_without_collapsing() {
        let mut state = DashboardState::default();
        state.reduce(DashboardAction::Resize(dashboard_layout_for_terminal_size(
            120, 24,
        )));

        let areas = tui_layout(Rect::new(0, 0, 120, 24), &state);
        let llama_inner = tui_panel_block(&state, DashboardPanel::LlamaCpp).inner(
            combine_panel_rect(areas.llama_processes.0, areas.llama_processes.1),
        );
        let webserver_inner = tui_panel_block(&state, DashboardPanel::Webserver).inner(
            combine_panel_rect(areas.webserver_processes.0, areas.webserver_processes.1),
        );
        assert_eq!(
            llama_inner.height as usize,
            state.panel_layout.rows_for(DashboardPanel::LlamaCpp)
        );
        assert_eq!(
            webserver_inner.height as usize,
            state.panel_layout.rows_for(DashboardPanel::Webserver)
        );

        let rendered = render_tui_frame_snapshot(&state, 120, 24);
        assert!(rendered.contains("Processes"));
        assert!(rendered.contains("llama.cpp"));
        assert!(rendered.contains("mesh-llm"));
        assert!(rendered.contains("(no llama.cpp processes yet)"));
        assert!(rendered.contains("(no webserver processes yet)"));
    }

    #[test]
    fn tui_process_tables_render_headers_and_joined_model_metadata() {
        let mut formatter = InteractiveDashboardFormatter::default();
        let mut process_row = sample_process_row("llama-server", 8001);
        process_row.backend = "metal".to_string();
        let mut model_row = sample_model_row("Mistral-7B", 8001);
        model_row.device = Some("GPU0".to_string());
        model_row.ctx_size = Some(8192);
        formatter.handle_snapshot(DashboardSnapshot {
            llama_process_rows: vec![process_row],
            webserver_rows: vec![sample_endpoint_row("Console", 3131)],
            loaded_model_rows: vec![model_row],
            ..snapshot_fixture(0, 30)
        });
        formatter.handle_tui_event(TuiEvent::Resize {
            columns: 240,
            rows: 30,
        });

        let rendered = render_tui_frame_snapshot(&formatter.state, 240, 30);
        let (_, process_header_line) = find_rendered_line(&rendered, "MODEL");
        assert!(process_header_line.contains("PID"));
        assert!(process_header_line.contains("PORT"));
        assert!(process_header_line.contains("STATE"));
        assert!(!process_header_line.contains("SLOTS"));
        assert!(rendered.contains("Mistral-7B"));
        assert_eq!(PRETTY_TUI_WEBSERVER_PROCESS_HEADER_LABEL, "PROCESSES");
        assert!(!rendered.contains("ENDPOINT"));
        assert!(rendered.contains("PID"));
        assert!(!rendered.contains("URL"));
        assert!(rendered.contains("mesh-llm Processes"));
    }

    #[test]
    fn tui_process_table_widths_give_text_columns_leftover_space() {
        let [model_width, pid_width, port_width, status_width] = llama_process_column_widths(52);

        assert_eq!(pid_width, 5);
        assert_eq!(port_width, 5);
        assert_eq!(status_width, 5);
        assert_eq!(model_width, 32);

        let rows = vec![DashboardEndpointRow {
            label: "Plugin: browser-tools".to_string(),
            status: RuntimeStatus::Ready,
            url: "browser-tools".to_string(),
            port: 0,
            pid: Some(4321),
        }];
        let [label_width, web_pid_width, web_port_width, web_status_width] =
            webserver_process_column_widths(52);

        assert_eq!(web_pid_width, 5);
        assert_eq!(web_port_width, 5);
        assert_eq!(web_status_width, 5);
        assert_eq!(label_width, 32);
        assert!(label_width >= rows[0].label.len());
        assert!(label_width >= PRETTY_TUI_WEBSERVER_PROCESS_HEADER_LABEL.len());
    }

    #[test]
    fn tui_dashboard_process_table_renders_missing_pid_as_dash() {
        assert_eq!(format_dashboard_pid(None), "-");
        assert_eq!(format_dashboard_pid(Some(4321)), "4321");
    }

    #[test]
    fn tui_hjkl_and_arrows_navigate_focused_panel_without_changing_focus() {
        let mut formatter = InteractiveDashboardFormatter::default();
        formatter.handle_snapshot(DashboardSnapshot {
            loaded_model_rows: vec![
                sample_model_row("Model-0", 4000),
                sample_model_row("Model-1", 4001),
                sample_model_row("Model-2", 4002),
            ],
            ..snapshot_fixture(0, 30)
        });
        formatter.handle_tui_event(TuiEvent::Resize {
            columns: 140,
            rows: 18,
        });
        formatter.state.panel_layout.widgets[DashboardPanel::Models.index()].selectable = true;
        formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Tab));
        formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Tab));
        formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Tab));
        assert_eq!(formatter.state.panel_focus, DashboardPanel::Models);

        formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Char('l')));
        formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Right));
        assert_eq!(formatter.state.panel_focus, DashboardPanel::Models);
        assert_eq!(
            formatter
                .state
                .panel_view_state(DashboardPanel::Models)
                .selected_row,
            Some(2)
        );

        formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Char('h')));
        formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Left));
        assert_eq!(formatter.state.panel_focus, DashboardPanel::Models);
        assert_eq!(
            formatter
                .state
                .panel_view_state(DashboardPanel::Models)
                .selected_row,
            Some(0)
        );
    }

    #[test]
    fn tui_up_down_cycle_request_window_when_requests_panel_is_focused() {
        let mut formatter = InteractiveDashboardFormatter::default();
        formatter.handle_tui_event(TuiEvent::Resize {
            columns: 140,
            rows: 18,
        });
        for _ in 0..4 {
            formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Tab));
        }
        assert_eq!(formatter.state.panel_focus, DashboardPanel::Requests);
        assert_eq!(
            formatter.state.request_window,
            DashboardRequestWindow::SixtySeconds
        );

        formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Down));
        assert_eq!(
            formatter.state.request_window,
            DashboardRequestWindow::TenMinutes
        );
        formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Down));
        formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Down));
        formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Down));
        assert_eq!(
            formatter.state.request_window,
            DashboardRequestWindow::TwentyFourHours
        );

        formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Down));
        assert_eq!(
            formatter.state.request_window,
            DashboardRequestWindow::TwentyFourHours
        );
        formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Up));
        assert_eq!(
            formatter.state.request_window,
            DashboardRequestWindow::TwelveHours
        );

        let rendered = render_tui_frame_snapshot(&formatter.state, 140, 18);
        assert!(rendered.contains("12h"));
        assert!(rendered.contains("30m buckets"));
    }

    #[test]
    fn tui_process_tables_support_focus_and_row_navigation() {
        let mut formatter = InteractiveDashboardFormatter::default();
        formatter.handle_snapshot(DashboardSnapshot {
            llama_process_rows: vec![
                sample_process_row("llama-0", 8001),
                sample_process_row("llama-1", 8002),
                sample_process_row("llama-2", 8003),
                sample_process_row("llama-3", 8004),
            ],
            webserver_rows: vec![
                sample_endpoint_row("Console", 3131),
                sample_endpoint_row("API", 9337),
                sample_endpoint_row("Metrics", 9393),
            ],
            ..snapshot_fixture(1, 30)
        });
        formatter.handle_tui_event(TuiEvent::Resize {
            columns: 120,
            rows: 12,
        });

        formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Tab));
        assert_eq!(formatter.state.panel_focus, DashboardPanel::LlamaCpp);
        assert_eq!(
            formatter.state.panel_view_state(DashboardPanel::LlamaCpp),
            DashboardPanelViewState {
                scroll_offset: 0,
                selected_row: None,
                viewport_rows: formatter
                    .state
                    .panel_layout
                    .rows_for(DashboardPanel::LlamaCpp),
            }
        );

        formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Down));
        formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Down));
        assert_eq!(
            formatter
                .state
                .panel_view_state(DashboardPanel::LlamaCpp)
                .selected_row,
            Some(2)
        );

        formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::PageDown));
        let llama_viewport_rows = formatter
            .state
            .panel_layout
            .rows_for(DashboardPanel::LlamaCpp);
        assert_eq!(
            formatter.state.panel_view_state(DashboardPanel::LlamaCpp),
            DashboardPanelViewState {
                scroll_offset: 4usize.saturating_sub(llama_viewport_rows),
                selected_row: Some(3),
                viewport_rows: llama_viewport_rows,
            }
        );

        formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Tab));
        assert_eq!(formatter.state.panel_focus, DashboardPanel::Webserver);
        formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Char('G')));
        assert_eq!(
            formatter
                .state
                .panel_view_state(DashboardPanel::Webserver)
                .selected_row,
            Some(2)
        );
        formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Char('g')));
        assert_eq!(
            formatter.state.panel_view_state(DashboardPanel::Webserver),
            DashboardPanelViewState {
                scroll_offset: 0,
                selected_row: Some(0),
                viewport_rows: formatter
                    .state
                    .panel_layout
                    .rows_for(DashboardPanel::Webserver),
            }
        );
    }

    #[test]
    fn tui_request_chart_preserves_thirty_one_second_buckets_with_newest_last() {
        let history = DashboardRequestHistoryState::from_snapshot(&DashboardSnapshot {
            accepted_request_buckets: vec![
                DashboardAcceptedRequestBucket {
                    second_offset: 0,
                    accepted_count: 9,
                },
                DashboardAcceptedRequestBucket {
                    second_offset: 5,
                    accepted_count: 4,
                },
                DashboardAcceptedRequestBucket {
                    second_offset: 29,
                    accepted_count: 1,
                },
            ],
            ..DashboardSnapshot::default()
        });

        let chart_spec =
            tui_request_chart_spec(&history, DashboardRequestWindow::SixtySeconds, 160);

        assert_eq!(
            chart_spec.bucket_values.len(),
            PRETTY_DASHBOARD_REQUEST_WINDOW_BUCKETS,
            "expected 30 two-second buckets"
        );
        assert_eq!(chart_spec.bucket_values.get(15), Some(&1));
        assert_eq!(chart_spec.bucket_values.get(27), Some(&4));
        assert_eq!(chart_spec.bucket_values.last(), Some(&9));
    }

    #[test]
    fn tui_braille_bar_symbols_use_vertical_subcell_fill() {
        assert_eq!(tui_braille_bar_symbol(0, 0), '⠀');
        assert_eq!(tui_braille_bar_symbol(1, 1), '⣀');
        assert_eq!(tui_braille_bar_symbol(2, 2), '⣤');
        assert_eq!(tui_braille_bar_symbol(3, 3), '⣶');
        assert_eq!(tui_braille_bar_symbol(4, 4), '⣿');
        assert!(is_braille_bar_symbol(tui_braille_bar_symbol(1, 0)));
        assert_ne!(tui_braille_bar_symbol(1, 0), tui_braille_bar_symbol(0, 1));
    }

    #[test]
    fn tui_request_chart_scale_uses_bucket_max_and_headroom_for_every_window() {
        let quiet_history = DashboardRequestHistoryState::default();
        let quiet_spec =
            tui_request_chart_spec(&quiet_history, DashboardRequestWindow::TwentyFourHours, 160);
        assert_eq!(quiet_spec.scale_max, 1);
        assert!(quiet_spec.scale_width >= 3);

        let sparse_day_history = DashboardRequestHistoryState::from_snapshot(&DashboardSnapshot {
            accepted_request_buckets: vec![DashboardAcceptedRequestBucket {
                second_offset: 23 * 60 * 60,
                accepted_count: 1,
            }],
            ..DashboardSnapshot::default()
        });
        let sparse_day_spec = tui_request_chart_spec(
            &sparse_day_history,
            DashboardRequestWindow::TwentyFourHours,
            160,
        );
        assert_eq!(sparse_day_spec.scale_max, 2);

        let busy_history = DashboardRequestHistoryState::from_snapshot(&DashboardSnapshot {
            accepted_request_buckets: vec![DashboardAcceptedRequestBucket {
                second_offset: 0,
                accepted_count: 51,
            }],
            ..DashboardSnapshot::default()
        });
        let busy_spec =
            tui_request_chart_spec(&busy_history, DashboardRequestWindow::SixtySeconds, 160);
        assert!(busy_spec.scale_max > 51);
        assert_eq!(busy_spec.scale_max, 100);
    }

    #[test]
    fn tui_request_scale_omits_duplicate_midpoint_for_unit_range() {
        assert_eq!(tui_request_scale_labels(4, 1), vec![(0, 1), (3, 0)]);
        assert_eq!(tui_request_scale_labels(4, 2), vec![(0, 2), (2, 1), (3, 0)]);
    }

    #[test]
    fn tui_request_chart_uses_thirty_and_sixty_minute_long_window_buckets() {
        assert_eq!(
            DashboardRequestWindow::TwelveHours.bucket_seconds(),
            30 * 60
        );
        assert_eq!(
            DashboardRequestWindow::TwentyFourHours.bucket_seconds(),
            60 * 60
        );
        assert_eq!(
            DashboardRequestWindow::TwelveHours.bucket_label(),
            "30m buckets"
        );
        assert_eq!(
            DashboardRequestWindow::TwentyFourHours.bucket_label(),
            "60m buckets"
        );

        let history = DashboardRequestHistoryState::from_snapshot(&DashboardSnapshot {
            accepted_request_buckets: vec![
                DashboardAcceptedRequestBucket {
                    second_offset: 30 * 60 - 1,
                    accepted_count: 3,
                },
                DashboardAcceptedRequestBucket {
                    second_offset: 30 * 60,
                    accepted_count: 5,
                },
            ],
            ..DashboardSnapshot::default()
        });
        let chart_spec = tui_request_chart_spec(&history, DashboardRequestWindow::TwelveHours, 160);
        assert_eq!(chart_spec.bucket_values.last(), Some(&3));
        assert_eq!(
            chart_spec
                .bucket_values
                .get(PRETTY_DASHBOARD_REQUEST_WINDOW_BUCKETS - 2),
            Some(&5)
        );
    }

    #[test]
    fn tui_request_chart_right_aligns_newest_bucket() {
        let mut state = DashboardState::default();
        state.reduce(DashboardAction::Resize(dashboard_layout_for_terminal_size(
            160, 24,
        )));
        state.reduce(DashboardAction::SnapshotUpdated(DashboardSnapshot {
            accepted_request_buckets: vec![DashboardAcceptedRequestBucket {
                second_offset: 0,
                accepted_count: 9,
            }],
            ..snapshot_fixture(0, 0)
        }));

        let (_, buffer) = render_tui_frame_snapshot_with_buffer(&state, 160, 24);
        let requests_inner = requests_inner_area(&state, 160, 24);
        let [_, graph_slot] = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(1), Constraint::Min(0)])
            .areas(requests_inner);
        let chart_spec = tui_request_chart_spec(
            &state.request_history,
            state.request_window,
            graph_slot.width,
        );
        let (_, plot_area) = tui_request_chart_areas(graph_slot, &chart_spec);

        assert!(
            (plot_area.y..plot_area.bottom()).any(|y| {
                buffer[(plot_area.right().saturating_sub(1), y)]
                    .symbol()
                    .chars()
                    .next()
                    .is_some_and(is_braille_bar_symbol)
            }),
            "expected newest request bucket to touch the right edge of the plot area"
        );
    }

    #[test]
    fn tui_request_chart_shrinks_long_window_bars() {
        let history = DashboardRequestHistoryState::from_snapshot(&DashboardSnapshot {
            accepted_request_buckets: vec![DashboardAcceptedRequestBucket {
                second_offset: 0,
                accepted_count: 9,
            }],
            ..DashboardSnapshot::default()
        });
        let short_spec =
            tui_request_chart_spec(&history, DashboardRequestWindow::SixtySeconds, 160);
        let day_spec =
            tui_request_chart_spec(&history, DashboardRequestWindow::TwentyFourHours, 160);

        assert!(
            short_spec.bar_width > day_spec.bar_width,
            "expected longer request windows to render narrower bars"
        );
        assert_eq!(day_spec.bar_width, 1);
    }

    #[test]
    fn tui_requests_panel_renders_multi_row_barchart_and_summary_values() {
        let mut state = DashboardState::default();
        state.reduce(DashboardAction::Resize(dashboard_layout_for_terminal_size(
            160, 24,
        )));
        state.reduce(DashboardAction::SnapshotUpdated(DashboardSnapshot {
            current_inflight_requests: 7,
            accepted_request_buckets: vec![
                DashboardAcceptedRequestBucket {
                    second_offset: 0,
                    accepted_count: 9,
                },
                DashboardAcceptedRequestBucket {
                    second_offset: 1,
                    accepted_count: 4,
                },
            ],
            latency_samples_ms: vec![11, 17, 19, 23],
            ..snapshot_fixture(0, 0)
        }));

        let (rendered, buffer) = render_tui_frame_snapshot_with_buffer(&state, 160, 24);
        let requests_inner = requests_inner_area(&state, 160, 24);
        let (_, line) = find_rendered_line(&rendered, "RPS ");

        assert!(
            line.contains("RPS 9"),
            "expected current-bucket RPS in {line}"
        );
        assert!(
            line.contains("inflight 7"),
            "expected inflight count in {line}"
        );
        assert!(line.contains("p50 18ms"), "expected p50 latency in {line}");
        assert!(
            line.contains("window 60s"),
            "expected request window in {line}"
        );
        assert!(
            line.contains("2s buckets"),
            "expected bucket size in {line}"
        );
        assert!(
            !line.contains('|'),
            "expected summary row, not old sparkline strip: {line}"
        );
        assert!(
            rendered.contains("Incoming Requests  60s  2s buckets"),
            "expected request panel title to show window and bucket size in {rendered}"
        );
        assert!(
            request_graph_visible_row_count(&buffer, requests_inner) >= 2,
            "expected multi-row request graph in area {requests_inner:?}\n{rendered}"
        );
        assert!(
            request_graph_contains_bars(&buffer, requests_inner),
            "expected real bar glyphs in request graph area {requests_inner:?}\n{rendered}"
        );
        assert!(
            rendered.contains("20"),
            "expected adaptive request scale label in {rendered}"
        );
        assert!(
            !rendered.contains('•'),
            "expected Braille bar glyphs instead of dot bullets in {rendered}"
        );
    }

    #[test]
    fn tui_requests_panel_shows_na_latency_when_window_empty() {
        let mut state = DashboardState::default();
        state.reduce(DashboardAction::Resize(dashboard_layout_for_terminal_size(
            160, 24,
        )));
        state.reduce(DashboardAction::SnapshotUpdated(DashboardSnapshot {
            current_inflight_requests: 2,
            accepted_request_buckets: vec![DashboardAcceptedRequestBucket {
                second_offset: 0,
                accepted_count: 3,
            }],
            latency_samples_ms: Vec::new(),
            ..snapshot_fixture(0, 0)
        }));

        let (rendered, buffer) = render_tui_frame_snapshot_with_buffer(&state, 160, 24);
        let requests_inner = requests_inner_area(&state, 160, 24);
        let (_, line) = find_rendered_line(&rendered, "RPS ");

        assert!(
            line.contains("p50 n/a"),
            "expected empty-window latency text in {line}"
        );
        assert!(
            request_graph_visible_row_count(&buffer, requests_inner) >= 2,
            "expected visible empty-state graph guides in area {requests_inner:?}\n{rendered}"
        );
        assert!(
            request_graph_contains_guides(&buffer, requests_inner),
            "expected empty-state graph guides in area {requests_inner:?}\n{rendered}"
        );
    }

    #[test]
    fn tui_requests_panel_zero_traffic_still_renders_visible_graph_area() {
        let mut state = DashboardState::default();
        state.reduce(DashboardAction::Resize(dashboard_layout_for_terminal_size(
            160, 24,
        )));

        let (rendered, buffer) = render_tui_frame_snapshot_with_buffer(&state, 160, 24);
        let requests_inner = requests_inner_area(&state, 160, 24);
        let (_, line) = find_rendered_line(&rendered, "RPS ");

        assert!(line.contains("RPS 0"), "expected zero RPS in {line}");
        assert!(
            line.contains("inflight 0"),
            "expected zero inflight in {line}"
        );
        assert!(line.contains("p50 n/a"), "expected n/a latency in {line}");
        assert!(
            request_graph_visible_row_count(&buffer, requests_inner) >= 2,
            "expected idle graph area to stay visibly chart-like in {requests_inner:?}\n{rendered}"
        );
        assert!(
            request_graph_contains_guides(&buffer, requests_inner),
            "expected idle graph guides in area {requests_inner:?}\n{rendered}"
        );
        assert!(
            !request_graph_contains_bars(&buffer, requests_inner),
            "expected idle graph to avoid fake traffic bars in area {requests_inner:?}\n{rendered}"
        );
    }

    #[test]
    fn tui_requests_panel_clears_stale_bars_before_redraw() {
        let mut busy_state = DashboardState::default();
        busy_state.reduce(DashboardAction::Resize(dashboard_layout_for_terminal_size(
            160, 24,
        )));
        busy_state.reduce(DashboardAction::SnapshotUpdated(DashboardSnapshot {
            accepted_request_buckets: vec![
                DashboardAcceptedRequestBucket {
                    second_offset: 0,
                    accepted_count: 40,
                },
                DashboardAcceptedRequestBucket {
                    second_offset: 1,
                    accepted_count: 32,
                },
                DashboardAcceptedRequestBucket {
                    second_offset: 2,
                    accepted_count: 28,
                },
            ],
            ..snapshot_fixture(0, 0)
        }));

        let mut quiet_state = DashboardState::default();
        quiet_state.reduce(DashboardAction::Resize(dashboard_layout_for_terminal_size(
            160, 24,
        )));

        let backend = ratatui::backend::TestBackend::new(160, 24);
        let mut terminal = Terminal::new(backend).expect("test backend should initialize");
        terminal
            .draw(|frame| render_tui_frame(frame, &busy_state))
            .expect("busy frame render should succeed");
        terminal
            .draw(|frame| render_tui_frame(frame, &quiet_state))
            .expect("quiet frame render should succeed");

        let buffer = terminal.backend().buffer().clone();
        let requests_inner = requests_inner_area(&quiet_state, 160, 24);

        assert!(
            !request_graph_contains_bars(&buffer, requests_inner),
            "expected quiet redraw to clear stale Braille bars"
        );
    }

    #[test]
    fn tui_requests_panel_stays_multi_row_at_tighter_live_height() {
        let mut state = DashboardState::default();
        state.reduce(DashboardAction::Resize(dashboard_layout_for_terminal_size(
            160, 23,
        )));

        let (rendered, buffer) = render_tui_frame_snapshot_with_buffer(&state, 160, 23);
        let requests_inner = requests_inner_area(&state, 160, 23);

        assert!(
            requests_inner.height >= 3,
            "expected summary + at least two graph rows in area {requests_inner:?}\n{rendered}"
        );
        assert!(
            request_graph_visible_row_count(&buffer, requests_inner) >= 2,
            "expected visible request graph rows in area {requests_inner:?}\n{rendered}"
        );
        assert!(
            request_graph_contains_guides(&buffer, requests_inner),
            "expected chart guides in tighter live-height area {requests_inner:?}\n{rendered}"
        );
    }

    #[test]
    fn tui_status_bar_reports_focus_follow_and_filter_state() {
        let mut state = DashboardState::default();
        state.reduce(DashboardAction::Resize(dashboard_layout_for_terminal_size(
            240, 24,
        )));
        state.reduce(DashboardAction::SnapshotUpdated(DashboardSnapshot {
            llama_process_rows: vec![sample_process_row("llama-0", 8001)],
            webserver_rows: vec![
                sample_endpoint_row("Console", 3131),
                sample_endpoint_row("API", 9337),
            ],
            loaded_model_rows: vec![
                sample_model_row("Model-0", 4000),
                sample_model_row("Model-1", 4001),
            ],
            ..snapshot_fixture(0, 30)
        }));
        state.reduce(DashboardAction::OutputEvent(OutputEvent::Startup {
            version: "v0.64.0".to_string(),
            message: None,
        }));
        state.reduce(DashboardAction::OutputEvent(OutputEvent::NodeIdentity {
            node_id: "node-7".to_string(),
            mesh_id: Some("poker-night".to_string()),
        }));
        state.reduce(DashboardAction::OutputEvent(OutputEvent::PeerJoined {
            peer_id: "peer-1".to_string(),
            label: Some("alice".to_string()),
        }));
        state.reduce(DashboardAction::OutputEvent(OutputEvent::PeerJoined {
            peer_id: "peer-2".to_string(),
            label: Some("bob".to_string()),
        }));
        state.reduce(DashboardAction::OutputEvent(OutputEvent::RuntimeReady {
            api_url: "http://localhost:9337".to_string(),
            console_url: Some("http://localhost:3131".to_string()),
            api_port: 9337,
            console_port: Some(3131),
            models_count: Some(2),
            pi_command: None,
            goose_command: None,
        }));
        state.reduce(DashboardAction::FocusNextPanel);
        state.reduce(DashboardAction::FocusNextPanel);
        state.reduce(DashboardAction::FocusNextPanel);
        state.reduce(DashboardAction::SetPanelSelection {
            panel: DashboardPanel::Models,
            selected_row: Some(1),
        });
        state.reduce(DashboardAction::StartEventsFilterEdit);
        state.reduce(DashboardAction::InsertEventsFilterChar('p'));
        state.reduce(DashboardAction::InsertEventsFilterChar('o'));
        state.reduce(DashboardAction::ConfirmEventsFilter);
        state.reduce(DashboardAction::FocusNextPanel);
        state.reduce(DashboardAction::FocusNextPanel);
        state.reduce(DashboardAction::FocusNextPanel);
        state.reduce(DashboardAction::ToggleEventsFollow);

        let rendered = render_tui_frame_snapshot(&state, 240, 24);
        assert!(rendered.contains("READY"));
        assert!(rendered.contains("uptime:"));
        assert!(
            rendered.contains("peers: 2"),
            "expected peer count in {rendered}"
        );
        assert!(
            rendered.contains("models: 2"),
            "expected model count in {rendered}"
        );
        assert!(
            rendered.contains("processes: 3"),
            "expected process count in {rendered}"
        );
        assert!(rendered.contains("[Tab]"));
        assert!(rendered.contains("[Shift-Tab]"));
        assert!(rendered.contains("[/]"));
        assert!(rendered.contains("[F]"));
        assert!(rendered.contains("[↑/↓]"));
        assert!(rendered.contains("[R]"));
        assert!(rendered.contains("[Q]"));
    }

    #[test]
    fn tui_status_bar_uses_badge_uptime_and_key_hint_styles() {
        let mut state = DashboardState::default();
        state.reduce(DashboardAction::Resize(dashboard_layout_for_terminal_size(
            180, 24,
        )));
        state.reduce(DashboardAction::OutputEvent(OutputEvent::RuntimeReady {
            api_url: "http://localhost:9337".to_string(),
            console_url: None,
            api_port: 9337,
            console_port: None,
            models_count: Some(0),
            pi_command: None,
            goose_command: None,
        }));

        let (rendered, buffer) = render_tui_frame_snapshot_with_buffer(&state, 180, 24);
        let (ready_y, ready_line) = find_rendered_line(&rendered, "READY");
        let ready_x = ready_line
            .find("READY")
            .expect("expected READY badge in status line");
        let (tab_y, tab_line) = find_rendered_line(&rendered, "[Tab]");
        let tab_x = tab_line
            .find("[Tab]")
            .expect("expected bracketed Tab hint in controls line");
        let peers_x = ready_line
            .find("peers:")
            .expect("expected peer stats in status line");
        let processes_x = ready_line
            .find("processes:")
            .expect("expected process stats in status line");
        let uptime_x = ready_line
            .find("uptime:")
            .expect("expected uptime in status line");
        let theme = tui_theme();

        assert!(
            rendered.contains("uptime:"),
            "expected uptime text in {rendered}"
        );
        assert!(
            rendered.contains("[Q] Quit"),
            "expected bracketed quit hint in {rendered}"
        );
        assert!(
            rendered.contains("[↑/↓] Window"),
            "expected bracketed request-window hint in {rendered}"
        );
        assert!(
            ready_x <= 1,
            "expected READY badge at the far left of status line: {ready_line}"
        );
        assert!(
            ready_x < tab_x,
            "expected READY badge to precede hotkeys in {ready_line}"
        );
        assert!(
            peers_x > tab_x,
            "expected status stats to stay pinned after the flexible gap in {ready_line}"
        );
        assert!(
            uptime_x > processes_x,
            "expected uptime to stay near the clock at the right edge in {ready_line}"
        );
        assert_eq!(
            buffer[(ready_x as u16, ready_y as u16)].style().fg,
            Some(theme.success)
        );
        assert_eq!(
            buffer[(tab_x as u16, tab_y as u16)].style().fg,
            Some(theme.accent)
        );
        assert_eq!(
            buffer[(tab_x as u16, tab_y as u16)].style().bg,
            Some(theme.surface_raised)
        );
    }

    #[test]
    fn tui_model_progress_renders_meshllm_wordmark_and_bar() {
        let mut state = DashboardState::default();
        state.reduce(DashboardAction::Resize(dashboard_layout_for_terminal_size(
            120, 24,
        )));
        state.reduce(DashboardAction::OutputEvent(
            OutputEvent::ModelDownloadProgress {
                label: "Qwen2.5-0.5B-Instruct-Q4_K_M".to_string(),
                file: Some("qwen2.5-0.5b-instruct-q4_k_m.gguf".to_string()),
                downloaded_bytes: Some(245_500_000),
                total_bytes: Some(491_000_000),
                status: ModelProgressStatus::Downloading,
            },
        ));

        let (rendered, buffer) = render_tui_frame_snapshot_with_buffer(&state, 120, 48);
        let (detail_y, _) = find_rendered_line(
            &rendered,
            "downloading model qwen2.5-0.5b-instruct-q4_k_m.gguf",
        );

        assert!(rendered.contains("downloading model qwen2.5-0.5b-instruct-q4_k_m.gguf"));
        assert!(rendered.contains('█'));
        assert!(
            !render_tui_frame_snapshot(&state, 120, 48).contains("Mesh Events"),
            "loading splash should own the full frame"
        );
        assert!(
            (0..detail_y.saturating_sub(2)).any(|y| {
                (0..120).any(|x| {
                    let cell = &buffer[(x as u16, y as u16)];
                    cell.symbol() != " " && cell.style().fg.is_some()
                })
            }),
            "expected ANSI logo content above the progress detail\n{rendered}"
        );
    }

    #[test]
    fn tui_startup_progress_continues_after_model_download_ready() {
        let mut state = DashboardState::default();
        state.reduce(DashboardAction::Resize(dashboard_layout_for_terminal_size(
            120, 24,
        )));
        state.reduce(DashboardAction::OutputEvent(
            OutputEvent::ModelDownloadProgress {
                label: "Qwen2.5-0.5B-Instruct-Q4_K_M".to_string(),
                file: Some("qwen2.5-0.5b-instruct-q4_k_m.gguf".to_string()),
                downloaded_bytes: Some(491_000_000),
                total_bytes: Some(491_000_000),
                status: ModelProgressStatus::Ready,
            },
        ));
        state.reduce(DashboardAction::OutputEvent(OutputEvent::LlamaStarting {
            model: Some("Qwen2.5-0.5B-Instruct-Q4_K_M".to_string()),
            http_port: 9338,
            ctx_size: Some(4096),
            log_path: None,
        }));

        let progress = state
            .active_loading_progress()
            .expect("startup loading progress should remain active before runtime ready");
        let rendered = render_tui_frame_snapshot(&state, 120, 48);
        let bar = loading_progress_bar(progress.ratio, 40);

        assert!(
            progress.ratio < 1.0,
            "startup progress must not jump to 100%"
        );
        assert!(progress
            .detail
            .contains("starting llama-server for Qwen2.5"));
        assert!(rendered.contains("starting llama-server for Qwen2.5"));
        assert!(
            bar.contains('░'),
            "progress bar should still have remaining work"
        );
        assert!(!rendered.contains("Mesh Events"));
    }

    #[test]
    fn tui_startup_progress_advances_with_startup_milestones() {
        let mut state = DashboardState::default();
        state.reduce(DashboardAction::OutputEvent(
            OutputEvent::ModelDownloadProgress {
                label: "Qwen2.5-0.5B-Instruct-Q4_K_M".to_string(),
                file: Some("qwen2.5-0.5b-instruct-q4_k_m.gguf".to_string()),
                downloaded_bytes: Some(491_000_000),
                total_bytes: Some(491_000_000),
                status: ModelProgressStatus::Ready,
            },
        ));
        let after_download = state
            .active_loading_progress()
            .expect("download-ready progress should seed startup progress")
            .ratio;

        state.reduce(DashboardAction::OutputEvent(
            OutputEvent::RpcServerStarting {
                port: 50052,
                device: "CUDA0".to_string(),
                log_path: None,
            },
        ));
        let after_rpc = state
            .active_loading_progress()
            .expect("rpc startup should advance startup progress")
            .ratio;

        state.reduce(DashboardAction::OutputEvent(OutputEvent::ModelReady {
            model: "Qwen2.5-0.5B-Instruct-Q4_K_M".to_string(),
            internal_port: Some(9338),
            role: Some("host".to_string()),
        }));
        let after_model_ready = state
            .active_loading_progress()
            .expect("model ready should advance startup progress")
            .ratio;

        assert!(after_rpc > after_download);
        assert!(after_model_ready > after_rpc);
    }

    #[test]
    fn tui_runtime_ready_keeps_dimmed_logo_above_dashboard() {
        let mut state = DashboardState::default();
        state.reduce(DashboardAction::Resize(dashboard_layout_for_terminal_size(
            160, 48,
        )));
        state.reduce(DashboardAction::OutputEvent(OutputEvent::RuntimeReady {
            api_url: "http://localhost:9337".to_string(),
            console_url: Some("http://localhost:3131".to_string()),
            api_port: 9337,
            console_port: Some(3131),
            models_count: Some(0),
            pi_command: None,
            goose_command: None,
        }));

        let area = Rect::new(0, 0, 160, 48);
        let areas = tui_layout(area, &state);
        let (rendered, buffer) = render_tui_frame_snapshot_with_buffer(&state, 160, 48);
        let slack_area = areas
            .loading
            .expect("runtime-ready layout should expose slack above dashboard");
        let logo_area = areas
            .logo
            .expect("runtime-ready layout should center a logo in the slack area");
        let ready_logo_height = u16::try_from(
            tui_ready_logo_text()
                .expect("ready logo text should be available")
                .lines
                .len(),
        )
        .unwrap_or(u16::MAX);
        let ready_logo_width = tui_ready_logo_text()
            .expect("ready logo text should be available")
            .lines
            .iter()
            .map(tui_logo_line_width)
            .max()
            .and_then(|width| u16::try_from(width).ok())
            .unwrap_or(logo_area.width);
        let first_visible_logo_row = (logo_area.y..logo_area.bottom())
            .find(|&y| {
                (logo_area.x..logo_area.right()).any(|x| {
                    let cell = &buffer[(x, y)];
                    cell.symbol() != " " && cell.style().add_modifier.contains(Modifier::DIM)
                })
            })
            .expect("expected dimmed ANSI logo content in the centered slack area");

        assert!(rendered.contains("Mesh Events"));
        assert!(rendered.contains("READY"));
        assert!(
            logo_area.height > 0 && logo_area.bottom() <= areas.main_body.y,
            "expected centered logo area above dashboard"
        );
        assert_eq!(logo_area.height, ready_logo_height.min(slack_area.height));
        assert_eq!(logo_area.width, ready_logo_width.min(slack_area.width));
        assert_eq!(
            logo_area.y,
            slack_area.y + (slack_area.height - logo_area.height) / 2
        );
        assert_eq!(
            logo_area.x,
            slack_area.x + (slack_area.width - logo_area.width) / 2
        );
        assert_eq!(first_visible_logo_row, logo_area.y);
        assert!(
            (logo_area.y..logo_area.bottom()).any(|y| {
                (logo_area.x..logo_area.right()).any(|x| {
                    let cell = &buffer[(x, y)];
                    cell.symbol() != " " && cell.style().add_modifier.contains(Modifier::DIM)
                })
            }),
            "expected dimmed ANSI logo content in the centered slack area\n{rendered}"
        );
    }

    #[test]
    fn tui_snapshot_renders_full_dashboard_spec() {
        let mut state = DashboardState::default();
        state.reduce(DashboardAction::Resize(dashboard_layout_for_terminal_size(
            220, 24,
        )));
        state.reduce(DashboardAction::SnapshotUpdated(snapshot_fixture(2, 30)));
        state.reduce(DashboardAction::OutputEvent(OutputEvent::Startup {
            version: "v0.64.0".to_string(),
            message: None,
        }));
        state.reduce(DashboardAction::OutputEvent(OutputEvent::NodeIdentity {
            node_id: "node-7".to_string(),
            mesh_id: Some("poker-night".to_string()),
        }));
        state.reduce(DashboardAction::OutputEvent(OutputEvent::PeerJoined {
            peer_id: "peer-1".to_string(),
            label: Some("alice".to_string()),
        }));
        state.reduce(DashboardAction::OutputEvent(OutputEvent::RuntimeReady {
            api_url: "http://localhost:9337".to_string(),
            console_url: Some("http://localhost:3131".to_string()),
            api_port: 9337,
            console_port: Some(3131),
            models_count: Some(2),
            pi_command: None,
            goose_command: None,
        }));
        state.reduce(DashboardAction::OutputEvent(info_event(
            "mesh named poker-night is private by default",
        )));

        let areas = tui_layout(Rect::new(0, 0, 220, 24), &state);
        let (rendered, buffer) = render_tui_frame_snapshot_with_buffer(&state, 220, 24);

        assert!(rendered.contains("Mesh Events"));
        assert!(rendered.contains("Processes"));
        assert!(rendered.contains("llama.cpp"));
        assert!(rendered.contains("mesh-llm Processes"));
        assert!(rendered.contains("Loaded Models"));
        assert!(rendered.contains("Incoming Requests"));
        assert!(!rendered.contains('📋'));
        assert!(!rendered.contains('⚙'));
        assert!(!rendered.contains('🔧'));
        assert!(!rendered.contains('📊'));
        assert!(!rendered.contains('📈'));
        assert!(rendered.contains("RPS "));
        assert!(rendered.contains("READY"));
        assert!(rendered.contains("[Tab] Next"));
        assert!(rendered.contains("[Shift-Tab] Prev"));
        assert!(rendered.contains('─'));
        assert!(rendered.contains('│'));
        assert!(rendered.contains("q"));
        assert!(!rendered.contains("Running llama.cpp instances"));
        assert!(!rendered.contains("Running models"));

        for panel_area in [
            combine_panel_rect(areas.events.0, areas.events.1),
            (combine_panel_rect(areas.llama_processes.0, areas.llama_processes.1)),
            (combine_panel_rect(areas.webserver_processes.0, areas.webserver_processes.1)),
            combine_panel_rect(areas.models.0, areas.models.1),
            combine_panel_rect(areas.requests.0, areas.requests.1),
        ] {
            assert_eq!(buffer[(panel_area.x, panel_area.y)].symbol(), "╭");
            assert_eq!(
                buffer[(panel_area.right().saturating_sub(1), panel_area.y)].symbol(),
                "╮"
            );
        }
    }

    #[test]
    fn tui_terminal_setup_marks_cleanup_required_after_enter_escape() {
        let mut formatter = InteractiveDashboardFormatter::default();

        formatter.mark_terminal_escape_written();

        assert!(formatter.terminal_active);
        assert!(formatter.dirty);
        assert!(formatter.terminal.is_none());
    }

    #[test]
    fn tui_narrow_terminal_renders_resize_guidance_instead_of_dashboard() {
        let mut state = DashboardState::default();
        state.reduce(DashboardAction::Resize(dashboard_layout_for_terminal_size(
            PRETTY_TUI_MIN_DASHBOARD_WIDTH - 1,
            24,
        )));
        state.reduce(DashboardAction::SnapshotUpdated(snapshot_fixture(2, 30)));
        state.reduce(DashboardAction::OutputEvent(OutputEvent::RuntimeReady {
            api_url: "http://localhost:9337".to_string(),
            console_url: Some("http://localhost:3131".to_string()),
            api_port: 9337,
            console_port: Some(3131),
            models_count: Some(2),
            pi_command: None,
            goose_command: None,
        }));

        let rendered = render_tui_frame_snapshot(&state, PRETTY_TUI_MIN_DASHBOARD_WIDTH - 1, 12);

        assert!(rendered.contains(">= 60 columns"));
        assert!(rendered.contains("Resize"));
        assert!(!rendered.contains("Mesh Events"));
    }

    #[test]
    fn tui_survives_rapid_event_bursts_without_scroll_jump() {
        let mut formatter = InteractiveDashboardFormatter::default();
        formatter.handle_tui_event(TuiEvent::Resize {
            columns: 140,
            rows: 18,
        });

        for index in 0..40 {
            let _ = formatter.handle_output_event(&info_event(format!("seed event {index}")));
        }

        formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Up));
        formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Up));
        formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Up));
        let before = formatter.state.panel_view_state(DashboardPanel::Events);
        assert!(
            !formatter.state.events_follow,
            "manual scroll should disable follow"
        );

        for index in 0..200 {
            let _ = formatter.handle_output_event(&info_event(format!("burst event {index}")));
        }

        let after = formatter.state.panel_view_state(DashboardPanel::Events);
        assert_eq!(after.scroll_offset, before.scroll_offset);
        assert_eq!(after.selected_row, before.selected_row);
        assert!(!formatter.state.events_follow);
        let rendered = render_tui_frame_snapshot(&formatter.state, 140, 18);
        assert!(rendered.contains("seed event"));
    }

    #[test]
    fn tui_models_render_ten_cell_ctx_and_cap_segments() {
        let mut state = DashboardState::default();
        state.reduce(DashboardAction::Resize(dashboard_layout_for_terminal_size(
            260, 32,
        )));
        state.reduce(DashboardAction::SnapshotUpdated(DashboardSnapshot {
            loaded_model_rows: vec![
                sample_model_row("Segmented-Model", 4001),
                DashboardModelRow {
                    name: "Half-Scale".to_string(),
                    role: Some("host".to_string()),
                    status: RuntimeStatus::Ready,
                    port: Some(4002),
                    device: Some("CUDA0".to_string()),
                    slots: Some(8),
                    quantization: Some("Q5_K_M".to_string()),
                    ctx_size: Some(4096),
                    file_size_gb: Some(12.0),
                },
            ],
            ..snapshot_fixture(0, 30)
        }));

        let (rendered, buffer) = render_tui_frame_snapshot_with_buffer(&state, 260, 32);
        let theme = tui_theme();
        let (full_title_y, full_title_line) = find_rendered_line(&rendered, "Segmented-Model");
        assert!(
            full_title_line.contains('╭') && full_title_line.contains('╮'),
            "expected bordered card title line in {full_title_line}"
        );
        assert!(
            full_title_line.contains("│╭"),
            "expected model card to start flush against the panel content edge, without a highlight gutter, in {full_title_line}"
        );
        let (full_ctx_y, full_ctx_line) =
            find_rendered_line_after(&rendered, full_title_y, "n/a / n/a");
        let (full_cap_y, full_cap_line) =
            find_rendered_line_after(&rendered, full_ctx_y, "24.0 / 24.0 GB");
        let (_, divider_line) = find_rendered_line_after(&rendered, full_title_y, "──");
        assert!(
            !divider_line.contains('├') && !divider_line.contains('┤'),
            "expected subtle interior divider, not frame-joining divider, in {divider_line}"
        );
        assert!(
            full_ctx_line.contains("CTX") && full_ctx_line.contains("n/a / n/a"),
            "expected CTX row with right-aligned value label in {full_ctx_line}"
        );
        assert!(
            full_cap_line.contains("FILE") && full_cap_line.contains("24.0 / 24.0 GB"),
            "expected FILE row with right-aligned value label in {full_cap_line}"
        );
        let full_ctx_gauge_x = full_ctx_line
            .find('█')
            .map(|index| full_ctx_line[..index].chars().count())
            .expect("expected CTX usage bar x coordinate");
        let full_cap_gauge_x = full_cap_line
            .find('█')
            .map(|index| full_cap_line[..index].chars().count())
            .expect("expected FILE usage bar x coordinate");
        assert_eq!(
            buffer[(
                u16::try_from(full_ctx_gauge_x).unwrap(),
                u16::try_from(full_ctx_y).unwrap()
            )]
                .style()
                .fg,
            Some(theme.dim)
        );
        assert_eq!(
            buffer[(
                u16::try_from(full_cap_gauge_x).unwrap(),
                u16::try_from(full_cap_y).unwrap()
            )]
                .style()
                .fg,
            Some(tui_model_usage_color(1.0))
        );

        let (half_title_y, _) = find_rendered_line(&rendered, "Half-Scale");
        let (half_ctx_y, half_ctx_line) =
            find_rendered_line_after(&rendered, half_title_y, "n/a / n/a");
        let (half_cap_y, half_cap_line) =
            find_rendered_line_after(&rendered, half_ctx_y, "12.0 / 24.0 GB");
        let half_ctx_gauge_x = half_ctx_line
            .find('█')
            .map(|index| half_ctx_line[..index].chars().count())
            .expect("expected half-scale CTX usage bar x coordinate");
        let half_cap_gauge_x = half_cap_line
            .find('█')
            .map(|index| half_cap_line[..index].chars().count())
            .expect("expected half-scale FILE usage bar x coordinate");
        assert_eq!(
            buffer[(
                u16::try_from(half_ctx_gauge_x).unwrap(),
                u16::try_from(half_ctx_y).unwrap()
            )]
                .style()
                .fg,
            Some(theme.dim)
        );
        assert_eq!(
            buffer[(
                u16::try_from(half_cap_gauge_x).unwrap(),
                u16::try_from(half_cap_y).unwrap()
            )]
                .style()
                .fg,
            Some(tui_model_usage_color(0.5))
        );
        let ctx_value_x = half_ctx_line
            .find("n/a / n/a")
            .map(|index| half_ctx_line[..index].chars().count())
            .expect("expected half CTX value label x coordinate");
        let file_value_x = half_cap_line
            .find("12.0 / 24.0 GB")
            .map(|index| half_cap_line[..index].chars().count())
            .expect("expected half FILE value label x coordinate");
        assert!(
            ((half_ctx_gauge_x + 1)..ctx_value_x).any(|x| {
                buffer[(
                    u16::try_from(x).unwrap(),
                    u16::try_from(half_ctx_y).unwrap(),
                )]
                    .style()
                    .fg
                    == Some(theme.dim)
            }),
            "expected CTX usage bar to show grey empty track after the fill"
        );
        assert!(
            ((half_cap_gauge_x + 1)..file_value_x).any(|x| {
                buffer[(
                    u16::try_from(x).unwrap(),
                    u16::try_from(half_cap_y).unwrap(),
                )]
                    .style()
                    .fg
                    == Some(theme.dim)
            }),
            "expected FILE usage bar to show grey empty track after the fill"
        );
    }

    #[test]
    fn tui_models_snapshot_includes_quant_slots_and_status() {
        let mut state = DashboardState::default();
        state.reduce(DashboardAction::Resize(dashboard_layout_for_terminal_size(
            260, 24,
        )));
        state.reduce(DashboardAction::SnapshotUpdated(DashboardSnapshot {
            loaded_model_rows: vec![DashboardModelRow {
                name: "Metadata-Model".to_string(),
                role: Some("host".to_string()),
                status: RuntimeStatus::Warning,
                port: Some(4011),
                device: Some("CUDA0".to_string()),
                slots: Some(8),
                quantization: Some("Q8_0".to_string()),
                ctx_size: Some(8192),
                file_size_gb: Some(24.0),
            }],
            ..snapshot_fixture(0, 30)
        }));

        let (rendered, buffer) = render_tui_frame_snapshot_with_buffer(&state, 260, 24);
        let theme = tui_theme();
        let (title_y, title_line) = find_rendered_line(&rendered, "Metadata-Model");
        assert!(
            title_line.contains('╭') && title_line.contains('╮'),
            "expected bordered card title line in {title_line}"
        );
        let (meta_y, meta_line) = find_rendered_line_after(&rendered, title_y, "STATUS");
        let (_, detail_line) = find_rendered_line_after(&rendered, title_y, "QUANT");
        assert!(
            meta_line.contains("STATUS: warning"),
            "expected warning status in {meta_line}"
        );
        assert!(
            meta_line.contains("PORT: 4011"),
            "expected port in {meta_line}"
        );
        assert!(
            meta_line.contains("DEVICE: n/a"),
            "expected temporary device placeholder in {meta_line}"
        );
        assert!(
            !meta_line.contains("DEV:"),
            "expected full DEVICE label rather than DEV in {meta_line}"
        );
        let areas = tui_layout(Rect::new(0, 0, 260, 24), &state);
        let models_area = combine_panel_rect(areas.models.0, areas.models.1);
        let models_meta_line = (models_area.x..models_area.right())
            .map(|x| buffer[(x, meta_y as u16)].symbol())
            .collect::<String>();
        let port_byte = models_meta_line
            .find("PORT:")
            .expect("expected PORT label x coordinate");
        let device_byte = models_meta_line
            .find("DEVICE:")
            .expect("expected DEVICE label x coordinate");
        let status_byte = models_meta_line
            .find("STATUS:")
            .expect("expected STATUS label x coordinate");
        let port_x = models_meta_line[..port_byte].chars().count();
        let device_x = models_meta_line[..device_byte].chars().count();
        let status_x = models_meta_line[..status_byte].chars().count();
        assert!(
            (device_x - port_x).abs_diff(status_x - device_x) <= 1,
            "expected PORT, DEVICE, and STATUS to use equal three-column spacing in {models_meta_line}"
        );
        assert!(
            detail_line.contains("SLOTS: 8"),
            "expected slots in {detail_line}"
        );
        assert!(
            detail_line.contains("Q8_0"),
            "expected quantization in {detail_line}"
        );
        assert!(
            detail_line.contains("CTX: n/a"),
            "expected temporary ctx placeholder in {detail_line}"
        );
        assert!(
            !detail_line.contains("ROLE:"),
            "role should not render in model details: {detail_line}"
        );
        let (ctx_y, ctx_line) = find_rendered_line_after(&rendered, title_y, "n/a / n/a");
        let (_, divider_line) = find_rendered_line_after(&rendered, title_y, "──");
        let (file_y, file_line) = find_rendered_line_after(&rendered, title_y, "24.0 / 24.0 GB");
        assert!(
            !divider_line.contains('├') && !divider_line.contains('┤'),
            "expected subtle interior divider, not frame-joining divider, in {divider_line}"
        );
        assert!(
            ctx_line.contains("CTX") && ctx_line.contains("n/a / n/a"),
            "expected visible ctx stat with right label in {ctx_line}"
        );
        assert!(
            file_line.contains("FILE") && file_line.contains("24.0 / 24.0 GB"),
            "expected visible file size stat with right label in {file_line}"
        );
        let ctx_gauge_x = ctx_line
            .find('█')
            .map(|index| ctx_line[..index].chars().count())
            .expect("expected CTX usage bar x coordinate");
        let file_gauge_x = file_line
            .find('█')
            .map(|index| file_line[..index].chars().count())
            .expect("expected FILE usage bar x coordinate");
        assert_eq!(
            buffer[(
                u16::try_from(ctx_gauge_x).unwrap(),
                u16::try_from(ctx_y).unwrap()
            )]
                .style()
                .fg,
            Some(theme.dim)
        );
        assert_eq!(
            buffer[(
                u16::try_from(file_gauge_x).unwrap(),
                u16::try_from(file_y).unwrap()
            )]
                .style()
                .fg,
            Some(tui_model_usage_color(1.0))
        );
    }

    #[test]
    fn tui_models_truncate_long_names_without_wrapping() {
        let long_name = "Extremely-Verbose-Model-Name-That-Should-Never-Wrap-Onto-A-Second-Line";
        let mut state = DashboardState::default();
        state.reduce(DashboardAction::Resize(dashboard_layout_for_terminal_size(
            220, 24,
        )));
        state.reduce(DashboardAction::SnapshotUpdated(DashboardSnapshot {
            loaded_model_rows: vec![DashboardModelRow {
                name: long_name.to_string(),
                role: Some("host".to_string()),
                status: RuntimeStatus::Ready,
                port: Some(4022),
                device: Some("GPU0".to_string()),
                slots: Some(4),
                quantization: Some("Q4_K_M".to_string()),
                ctx_size: Some(8192),
                file_size_gb: Some(24.0),
            }],
            ..snapshot_fixture(0, 30)
        }));

        let rendered = render_tui_frame_snapshot(&state, 220, 24);
        let (title_y, title_line) = rendered
            .lines()
            .enumerate()
            .find(|(_, line)| line.contains('╭') && line.contains('…'))
            .expect("expected truncated card title line");
        let (meta_y, meta_line) = find_rendered_line_after(&rendered, title_y, "DEVICE");
        let (_, detail_line) = find_rendered_line_after(&rendered, title_y, "Q4_K_M");
        assert!(
            title_line.contains('…'),
            "expected ellipsis in truncated model title: {title_line}"
        );
        assert!(
            detail_line.contains("Q4_K_M"),
            "expected quantization to remain visible: {detail_line}"
        );
        assert!(
            meta_line.contains("n/a"),
            "expected device placeholder: {meta_line}"
        );
        assert!(
            !meta_line.contains("GPU0"),
            "raw device should not render while placeholder data is used: {meta_line}"
        );
        assert!(meta_y > title_y, "expected metadata on a later card row");
        assert!(
            !rendered.contains(long_name),
            "full long model name should not survive truncation"
        );
    }

    #[test]
    fn tui_models_cards_scroll_without_selecting_inner_cards() {
        let mut formatter = InteractiveDashboardFormatter::default();
        formatter.handle_snapshot(DashboardSnapshot {
            loaded_model_rows: (0..5)
                .map(|index| sample_model_row(&format!("Model-{index}"), 4000 + index as u16))
                .collect(),
            ..snapshot_fixture(0, 30)
        });
        formatter.handle_tui_event(TuiEvent::Resize {
            columns: 180,
            rows: 24,
        });
        formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Tab));
        formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Tab));
        formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Tab));

        let initial_view = formatter.state.panel_view_state(DashboardPanel::Models);
        assert_eq!(formatter.state.panel_focus, DashboardPanel::Models);
        assert_eq!(initial_view.viewport_rows, 1);

        formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Down));
        formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Down));
        formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Down));

        let after = formatter.state.panel_view_state(DashboardPanel::Models);
        assert_eq!(after.selected_row, None);
        assert_eq!(after.scroll_offset, 3);

        let (rendered, buffer) = render_tui_frame_snapshot_with_buffer(&formatter.state, 180, 24);
        assert!(
            rendered.contains("▶ Loaded Models"),
            "expected the outer models pane to remain focused in {rendered}"
        );
        assert!(
            rendered.contains("Model-3"),
            "expected first visible card in {rendered}"
        );
        let (model_y, _) = find_rendered_line(&rendered, "Model-3");
        let areas = tui_layout(Rect::new(0, 0, 180, 24), &formatter.state);
        let models_area = combine_panel_rect(areas.models.0, areas.models.1);
        let model_x = (models_area.x..models_area.right())
            .find(|&x| buffer[(x, model_y as u16)].symbol() == "M")
            .expect("model name should have an x coordinate inside the models panel");
        let theme = tui_theme();
        assert_ne!(
            buffer[(model_x, model_y as u16)].style().bg,
            Some(theme.selection_bg),
            "model card content should not use the selected-row background"
        );
        assert!(
            !rendered.contains("Model-2"),
            "expected previous card to be scrolled off in {rendered}"
        );
        assert!(
            !rendered.contains("Model-0"),
            "expected scrolled-off card to disappear"
        );
    }

    fn parse_json_line(rendered: &str) -> Value {
        assert!(
            rendered.ends_with('\n'),
            "json formatter should emit newline-delimited output"
        );
        serde_json::from_str(rendered.trim_end()).expect("line should parse as json")
    }

    fn assert_required_json_envelope(value: &Value, event: &OutputEvent) {
        let timestamp = value
            .get("timestamp")
            .and_then(Value::as_str)
            .expect("json output should include string timestamp");
        assert!(
            timestamp.ends_with('Z') && timestamp.contains('T'),
            "timestamp should be RFC3339 UTC, got {timestamp}"
        );
        assert_eq!(
            value.get("level").and_then(Value::as_str),
            Some(event.level().as_str()),
            "json output should include level for {event:?}"
        );
        assert_eq!(
            value.get("event").and_then(Value::as_str),
            Some(event.event_name()),
            "json output should include event name for {event:?}"
        );
        assert_eq!(
            value.get("message").and_then(Value::as_str),
            Some(event.message().as_str()),
            "json output should include message for {event:?}"
        );
    }

    #[test]
    fn json_formatter_emits_app_owned_ndjson() {
        let mut output = Vec::new();
        let mut formatter = JsonFormatter;

        output
            .write_all(
                formatter
                    .format(&OutputEvent::RpcServerStarting {
                        port: 43683,
                        device: "CUDA0".to_string(),
                        log_path: Some("/tmp/rpc.log".to_string()),
                    })
                    .expect("json emit should succeed")
                    .as_bytes(),
            )
            .expect("write should succeed");

        let rendered = String::from_utf8(output).expect("output should be utf8");
        let line = rendered.trim_end();
        let value: Value = serde_json::from_str(line).expect("line should parse as json");
        assert_eq!(value["event"], "rpc_server_starting");
        assert_eq!(value["device"], "CUDA0");
        assert_eq!(value["log_path"], "/tmp/rpc.log");
        assert!(rendered.ends_with('\n'));
    }

    #[test]
    fn json_formatter_emits_llama_server_starting_payload() {
        let mut formatter = JsonFormatter;
        let rendered = formatter
            .format(&OutputEvent::LlamaStarting {
                model: Some("Qwen3.6-35B".to_string()),
                http_port: 43683,
                ctx_size: Some(8192),
                log_path: Some("/tmp/llama.log".to_string()),
            })
            .expect("llama startup render should succeed");
        let value: Value = serde_json::from_str(rendered.trim_end()).expect("line should parse");

        assert_eq!(value["event"], "llama_starting");
        assert_eq!(value["model"], "Qwen3.6-35B");
        assert_eq!(value["http_port"], 43683);
        assert_eq!(value["ctx_size"], 8192);
        assert_eq!(value["log_path"], "/tmp/llama.log");
    }

    #[test]
    fn json_formatter_includes_invite_mesh_metadata() {
        let mut formatter = JsonFormatter;
        let rendered = formatter
            .format(&OutputEvent::InviteToken {
                token: "invite-token".to_string(),
                mesh_id: "mesh-123".to_string(),
                mesh_name: None,
            })
            .expect("invite render should succeed");
        let value: Value = serde_json::from_str(rendered.trim_end()).expect("line should parse");

        assert_eq!(value["event"], "invite_token");
        assert_eq!(value["token"], "invite-token");
        assert_eq!(value["mesh_id"], "mesh-123");
    }

    #[test]
    fn json_formatter_includes_discovery_payloads() {
        let mut formatter = JsonFormatter;

        let started = formatter
            .format(&OutputEvent::DiscoveryStarting {
                source: "Nostr re-discovery".to_string(),
            })
            .expect("discovery start render should succeed");
        let started_value: Value = serde_json::from_str(started.trim_end()).expect("json line");
        assert_eq!(started_value["event"], "discovery_starting");
        assert_eq!(started_value["source"], "Nostr re-discovery");

        let candidate = formatter
            .format(&OutputEvent::MeshFound {
                mesh: "poker-night".to_string(),
                peers: 7,
                region: None,
            })
            .expect("discovery candidate render should succeed");
        let candidate_value: Value = serde_json::from_str(candidate.trim_end()).expect("json line");
        assert_eq!(candidate_value["event"], "mesh_found");
        assert_eq!(candidate_value["mesh"], "poker-night");
        assert_eq!(candidate_value["peers"], 7);
        assert_eq!(candidate_value["region"], Value::Null);

        let joined = formatter
            .format(&OutputEvent::DiscoveryJoined {
                mesh: "poker-night".to_string(),
            })
            .expect("discovery join render should succeed");
        let joined_value: Value = serde_json::from_str(joined.trim_end()).expect("json line");
        assert_eq!(joined_value["event"], "discovery_joined");
        assert_eq!(joined_value["mesh"], "poker-night");

        let failed = formatter
            .format(&OutputEvent::DiscoveryFailed {
                message: "Could not re-join any mesh — will retry".to_string(),
                detail: None,
            })
            .expect("discovery failure render should succeed");
        let failed_value: Value = serde_json::from_str(failed.trim_end()).expect("json line");
        assert_eq!(failed_value["event"], "discovery_failed");
        assert_eq!(
            failed_value["message"],
            "Could not re-join any mesh — will retry"
        );
        assert_eq!(failed_value["detail"], Value::Null);
    }

    #[test]
    fn json_formatter_includes_moe_detection_and_distribution_payloads() {
        let mut formatter = JsonFormatter;

        let detected = formatter
            .format(&OutputEvent::MoeDetected {
                model: "Qwen3-MoE".to_string(),
                moe: MoeSummary {
                    experts: 128,
                    top_k: 8,
                },
                fits_locally: Some(true),
                capacity_gb: Some(24.0),
                model_gb: Some(18.0),
            })
            .expect("moe detection render should succeed");
        let detected_value: Value = serde_json::from_str(detected.trim_end()).expect("json line");
        assert_eq!(detected_value["event"], "moe_detected");
        assert_eq!(detected_value["model"], "Qwen3-MoE");
        assert_eq!(detected_value["moe"]["experts"], 128);
        assert_eq!(detected_value["moe"]["top_k"], 8);
        assert_eq!(detected_value["fits_locally"], true);
        assert_eq!(detected_value["capacity_gb"], serde_json::json!(24.0));
        assert_eq!(detected_value["model_gb"], serde_json::json!(18.0));

        let distributed = formatter
            .format(&OutputEvent::MoeDistribution {
                model: "Qwen3-MoE".to_string(),
                moe: MoeSummary {
                    experts: 128,
                    top_k: 8,
                },
                distribution: MoeDistributionSummary {
                    leader: "node-7".to_string(),
                    active_nodes: 3,
                    fallback_nodes: 1,
                    shard_index: 0,
                    shard_count: 3,
                    ranking_source: "shared".to_string(),
                    ranking_origin: "cache".to_string(),
                    overlap: 2,
                    shared_experts: 48,
                    unique_experts: 80,
                },
            })
            .expect("moe distribution render should succeed");
        let distributed_value: Value =
            serde_json::from_str(distributed.trim_end()).expect("json line");
        assert_eq!(distributed_value["event"], "moe_distribution");
        assert_eq!(distributed_value["model"], "Qwen3-MoE");
        assert_eq!(distributed_value["moe"]["experts"], 128);
        assert_eq!(distributed_value["moe"]["top_k"], 8);
        assert_eq!(distributed_value["distribution"]["leader"], "node-7");
        assert_eq!(distributed_value["distribution"]["active_nodes"], 3);
        assert_eq!(distributed_value["distribution"]["fallback_nodes"], 1);
        assert_eq!(distributed_value["distribution"]["shard_index"], 0);
        assert_eq!(distributed_value["distribution"]["shard_count"], 3);
        assert_eq!(
            distributed_value["distribution"]["ranking_source"],
            "shared"
        );
        assert_eq!(distributed_value["distribution"]["ranking_origin"], "cache");
        assert_eq!(distributed_value["distribution"]["overlap"], 2);
        assert_eq!(distributed_value["distribution"]["shared_experts"], 48);
        assert_eq!(distributed_value["distribution"]["unique_experts"], 80);
    }

    #[test]
    fn dashboard_formatter_renders_invite_and_waiting_events_into_mesh_history() {
        let mut formatter =
            DashboardFormatter::with_state(DashboardState::with_mesh_event_limit(4));

        let _ = formatter
            .format(&OutputEvent::InviteToken {
                token: "invite-token".to_string(),
                mesh_id: "mesh-123".to_string(),
                mesh_name: None,
            })
            .expect("invite render should succeed");
        let dashboard = formatter
            .format(&OutputEvent::WaitingForPeers { detail: None })
            .expect("waiting render should succeed");

        assert!(dashboard.contains("Mesh events (latest 4)"));
        assert!(dashboard.contains("Invite created for mesh mesh-123: invite-token"));
        assert!(dashboard.contains("Waiting for peers..."));
        assert!(!dashboard.contains('📡'));
        for line in dashboard
            .lines()
            .filter(|line| line.contains("Waiting for peers"))
        {
            assert!(
                !line.contains('⏳'),
                "mesh event line should be emoji-free: {line}"
            );
        }
    }

    #[test]
    fn tui_falls_back_to_legacy_stderr_render_when_not_tty() {
        let mut formatter = select_formatter(LogFormat::Pretty, ConsoleSessionMode::Fallback);

        assert_eq!(formatter.kind(), "pretty_fallback");

        let dashboard = formatter
            .format(&OutputEvent::RuntimeReady {
                api_url: "http://localhost:9337".to_string(),
                console_url: Some("http://localhost:3131".to_string()),
                api_port: 9337,
                console_port: Some(3131),
                models_count: Some(1),
                pi_command: None,
                goose_command: None,
            })
            .expect("fallback render should succeed");

        assert!(dashboard.contains("Running llama.cpp instances"));
        assert!(dashboard.contains("Running API"));
        assert!(dashboard.contains("OpenAI-compatible API   ready   http://localhost:9337"));
        assert!(!dashboard.contains("\u{1b}[?1049h"));
        assert!(!dashboard.contains("\u{1b}[?1049l"));
        assert!(!dashboard.contains("\u{1b}[?25l"));
        assert!(!dashboard.contains("\u{1b}[?25h"));
    }

    #[test]
    fn tui_event_loop_dispatches_quit_on_q() {
        let mut formatter = InteractiveDashboardFormatter::default();

        assert_eq!(
            formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Char('q'))),
            TuiControlFlow::Quit
        );
        assert_eq!(
            formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Interrupt)),
            TuiControlFlow::Quit
        );
    }

    #[test]
    fn interactive_preterminal_render_uses_tui_snapshot_only() {
        let mut formatter = InteractiveDashboardFormatter::default();

        let rendered = formatter
            .handle_output_event(&OutputEvent::RuntimeReady {
                api_url: "http://localhost:9337".to_string(),
                console_url: Some("http://localhost:3131".to_string()),
                api_port: 9337,
                console_port: Some(3131),
                models_count: Some(1),
                pi_command: None,
                goose_command: None,
            })
            .expect("interactive pre-terminal render should succeed")
            .expect("interactive formatter should emit initial textual snapshot");

        assert!(rendered.contains("Incoming Requests"));
        assert!(rendered.contains("READY"));
        assert!(rendered.contains('─'));
        assert!(rendered.contains('│'));
        assert!(!rendered.contains("Running llama.cpp instances"));
        assert!(!rendered.contains("Running models"));
    }

    #[test]
    fn tui_restores_terminal_state_on_exit() {
        let mut output = Vec::new();

        write_tui_enter_to_writer(&mut output).expect("enter should succeed");
        write_tui_frame_to_writer(&mut output, "dashboard").expect("frame render should succeed");
        write_tui_exit_to_writer(&mut output).expect("exit should succeed");

        let rendered = String::from_utf8(output).expect("terminal output should be utf8");
        let leave_index = rendered
            .rfind("[?1049l")
            .expect("expected leave-alternate-screen sequence in exit output");
        let clear_index = rendered
            .rfind("[2J")
            .expect("expected full-screen clear in exit output");

        assert!(rendered.contains("dashboard"));
        assert!(rendered.contains('\u{1b}'));
        assert!(
            clear_index > leave_index,
            "expected final clear after leaving alternate screen in {rendered:?}"
        );
        assert!(rendered.matches('\u{1b}').count() >= 6);
    }

    #[test]
    fn tui_redraw_start_repositions_without_physical_clear() {
        let mut output = Vec::new();

        write_tui_redraw_start_to_writer(&mut output).expect("redraw start should succeed");

        let rendered = String::from_utf8(output).expect("terminal output should be utf8");
        assert!(
            rendered.contains("[?25l"),
            "redraw start should hide the cursor before repainting: {rendered:?}"
        );
        assert!(
            rendered.contains("[H") || rendered.contains("[1;1H"),
            "redraw start should move to the top-left before repainting: {rendered:?}"
        );
        assert!(
            !rendered.contains("[2J"),
            "redraw start should avoid a physical full-screen clear that flickers between frames: {rendered:?}"
        );
    }

    #[test]
    fn tui_handles_resize_without_resetting_focus() {
        let mut formatter = InteractiveDashboardFormatter::default();
        formatter.handle_snapshot(snapshot_fixture(12, 30));

        formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Tab));
        formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Tab));
        formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Tab));
        assert_eq!(formatter.state.panel_focus, DashboardPanel::Models);

        formatter.handle_tui_event(TuiEvent::Resize {
            columns: 120,
            rows: 36,
        });

        assert_eq!(formatter.state.panel_focus, DashboardPanel::Models);
    }

    #[tokio::test]
    async fn dashboard_snapshot_registration_stays_pretty_only() {
        let dashboard_manager = OutputManager {
            tx: tokio::sync::mpsc::unbounded_channel().0,
            ready_prompt_active: Arc::new(AtomicBool::new(false)),
            mode: LogFormat::Pretty,
            console_session_mode: Some(ConsoleSessionMode::InteractiveDashboard),
            dashboard_snapshot_provider: Arc::new(RwLock::new(None)),
        };
        let json_manager = OutputManager {
            tx: tokio::sync::mpsc::unbounded_channel().0,
            ready_prompt_active: Arc::new(AtomicBool::new(false)),
            mode: LogFormat::Json,
            console_session_mode: None,
            dashboard_snapshot_provider: Arc::new(RwLock::new(None)),
        };
        let expected = DashboardSnapshot {
            current_inflight_requests: 3,
            ..DashboardSnapshot::default()
        };
        let provider = Arc::new(StaticDashboardSnapshotProvider {
            snapshot: expected.clone(),
        });

        dashboard_manager.register_dashboard_snapshot_provider(provider.clone());
        json_manager.register_dashboard_snapshot_provider(provider);

        assert_eq!(dashboard_manager.dashboard_snapshot().await, Some(expected));
        assert_eq!(json_manager.dashboard_snapshot().await, None);
    }

    #[test]
    fn json_formatter_writes_machine_output_to_stdout_only() {
        let mut stdout = Vec::new();
        let mut stderr = Vec::new();

        write_rendered_output_to_writers(
            LogFormat::Json,
            "{\"event\":\"ready\"}\n",
            &mut stdout,
            &mut stderr,
        )
        .expect("json write should succeed");

        assert_eq!(
            String::from_utf8(stdout).expect("stdout should be utf-8"),
            "{\"event\":\"ready\"}\n"
        );
        assert!(
            stderr.is_empty(),
            "json output must not be routed to stderr"
        );
    }

    #[test]
    fn dashboard_formatter_renders_discovery_events_into_mesh_history() {
        let mut formatter =
            DashboardFormatter::with_state(DashboardState::with_mesh_event_limit(4));

        formatter
            .format(&OutputEvent::DiscoveryStarting {
                source: "Nostr re-discovery".to_string(),
            })
            .expect("discovery start render should succeed");
        formatter
            .format(&OutputEvent::MeshFound {
                mesh: "poker-night".to_string(),
                peers: 7,
                region: None,
            })
            .expect("discovery candidate render should succeed");
        let dashboard = formatter
            .format(&OutputEvent::DiscoveryJoined {
                mesh: "poker-night".to_string(),
            })
            .expect("discovery join render should succeed");

        assert!(dashboard.contains("discovering mesh via Nostr re-discovery"));
        assert!(dashboard.contains("discovered mesh poker-night (7 peer(s))"));
        assert!(dashboard.contains("joined mesh poker-night"));
        assert!(!dashboard.contains('🔍'));
        assert!(!dashboard.contains('📡'));
        assert!(!dashboard.contains('✅'));
    }

    #[test]
    fn dashboard_formatter_renders_discovery_failure_in_mesh_history() {
        let mut formatter =
            DashboardFormatter::with_state(DashboardState::with_mesh_event_limit(4));

        let dashboard = formatter
            .format(&OutputEvent::DiscoveryFailed {
                message: "Nostr re-discovery failed".to_string(),
                detail: Some("relay timeout".to_string()),
            })
            .expect("discovery failure render should succeed");

        assert!(dashboard.contains("Nostr re-discovery failed: relay timeout"));
        assert!(!dashboard.contains("⚠️ Nostr re-discovery failed"));
    }

    #[test]
    fn dashboard_formatter_renders_warning_context_in_mesh_history() {
        let mut formatter =
            DashboardFormatter::with_state(DashboardState::with_mesh_event_limit(4));

        let dashboard = formatter
            .format(&OutputEvent::Warning {
                message: "llama-server process exited unexpectedly".to_string(),
                context: Some("model=Qwen3-32B port=9337".to_string()),
            })
            .expect("warning render should succeed");

        assert!(dashboard
            .contains("model=Qwen3-32B port=9337: llama-server process exited unexpectedly"));
        assert!(!dashboard.contains("⚠️ model=Qwen3-32B port=9337"));

        let dashboard = formatter
            .format(&OutputEvent::Warning {
                message: "⚠️ top-level --client now maps to `mesh-llm client`; re-running with client semantics"
                    .to_string(),
                context: None,
            })
            .expect("warning render with embedded icon should succeed");

        assert!(dashboard.contains(
            "top-level --client now maps to `mesh-llm client`; re-running with client semantics"
        ));
        assert!(!dashboard.contains(
            "⚠️ ⚠️ top-level --client now maps to `mesh-llm client`; re-running with client semantics"
        ));
    }

    #[test]
    fn dashboard_formatter_renders_info_context_in_mesh_history() {
        let mut formatter =
            DashboardFormatter::with_state(DashboardState::with_mesh_event_limit(4));

        let dashboard = formatter
            .format(&OutputEvent::Info {
                message: "mesh named poker-night is private by default".to_string(),
                context: Some("publish=false".to_string()),
            })
            .expect("info render should succeed");

        assert!(dashboard.contains("publish=false: mesh named poker-night is private by default"));
    }

    #[test]
    fn dashboard_formatter_renders_multi_model_mode_in_running_models_section() {
        let mut formatter =
            DashboardFormatter::with_state(DashboardState::with_mesh_event_limit(4));

        formatter
            .format(&OutputEvent::MultiModelMode {
                count: 3,
                models: vec![
                    "Qwen2.5-32B".to_string(),
                    "GLM-4.7-Flash".to_string(),
                    "MiniMax-M2.5".to_string(),
                ],
            })
            .expect("multi-model render should succeed");
        formatter
            .format(&OutputEvent::ModelReady {
                model: "GLM-4.7-Flash".to_string(),
                internal_port: Some(3001),
                role: Some("host".to_string()),
            })
            .expect("model render should succeed");
        let dashboard = formatter
            .format(&OutputEvent::ModelReady {
                model: "Qwen2.5-32B".to_string(),
                internal_port: Some(3002),
                role: Some("standby".to_string()),
            })
            .expect("model render should succeed");

        assert!(dashboard.contains("Running models"));
        assert!(dashboard.contains(
            "multi-model mode   3 model(s)   models=Qwen2.5-32B, GLM-4.7-Flash, MiniMax-M2.5"
        ));
        assert!(dashboard.contains("GLM-4.7-Flash   ready   port=3001   role=host"));
        assert!(dashboard.contains("Qwen2.5-32B   ready   port=3002   role=standby"));
    }

    #[test]
    fn dashboard_formatter_pins_host_elected_role_and_capacity() {
        let mut formatter =
            DashboardFormatter::with_state(DashboardState::with_mesh_event_limit(4));

        let dashboard = formatter
            .format(&OutputEvent::HostElected {
                model: "Qwen3-32B".to_string(),
                host: "node-7".to_string(),
                role: Some("host".to_string()),
                capacity_gb: Some(24.0),
            })
            .expect("host election render should succeed");

        assert!(dashboard.contains("Running models"));
        assert!(dashboard.contains("Qwen3-32B   starting   role=host   capacity=24.0GB"));
        assert!(dashboard.contains("Qwen3-32B elected node-7 as host (24.0GB capacity)"));
        assert!(!dashboard.contains('🗳'));
    }

    #[test]
    fn dashboard_formatter_pins_passive_mode_in_running_models() {
        let mut formatter =
            DashboardFormatter::with_state(DashboardState::with_mesh_event_limit(4));

        let dashboard = formatter
            .format(&OutputEvent::PassiveMode {
                role: "standby".to_string(),
                status: RuntimeStatus::Starting,
                capacity_gb: Some(24.0),
                models_on_disk: Some(vec!["Qwen2.5-32B".to_string(), "GLM-4.7-Flash".to_string()]),
                detail: Some("No matching model on disk — running as standby GPU node. Proxying requests to other nodes. Will activate when needed.".to_string()),
            })
            .expect("passive mode render should succeed");

        assert!(dashboard.contains("Running models"));
        assert!(dashboard
            .contains("standby   starting   capacity=24.0GB   models=Qwen2.5-32B, GLM-4.7-Flash"));
        assert!(dashboard.contains("No matching model on disk — running as standby GPU node."));
        assert!(dashboard.contains("No matching model on disk — running as standby GPU node. Proxying requests to other nodes. Will activate when needed. (24.0GB capacity) models=Qwen2.5-32B, GLM-4.7-Flash"));
        assert!(!dashboard.contains('💤'));
    }

    #[test]
    fn dashboard_formatter_renders_moe_detection_and_distribution_in_running_models() {
        let mut formatter =
            DashboardFormatter::with_state(DashboardState::with_mesh_event_limit(4));

        formatter
            .format(&OutputEvent::MoeDetected {
                model: "Qwen3-MoE".to_string(),
                moe: MoeSummary {
                    experts: 128,
                    top_k: 8,
                },
                fits_locally: None,
                capacity_gb: None,
                model_gb: None,
            })
            .expect("moe detection render should succeed");
        let dashboard = formatter
            .format(&OutputEvent::MoeDistribution {
                model: "Qwen3-MoE".to_string(),
                moe: MoeSummary {
                    experts: 128,
                    top_k: 8,
                },
                distribution: MoeDistributionSummary {
                    leader: "node-7".to_string(),
                    active_nodes: 3,
                    fallback_nodes: 1,
                    shard_index: 0,
                    shard_count: 3,
                    ranking_source: "shared".to_string(),
                    ranking_origin: "cache".to_string(),
                    overlap: 2,
                    shared_experts: 48,
                    unique_experts: 80,
                },
            })
            .expect("moe distribution render should succeed");

        assert!(dashboard.contains("Running models"));
        assert!(dashboard.contains("Qwen3-MoE   starting   MoE: 128 experts, top-8"));
        assert!(dashboard.contains("split leader=node-7 active=3 fallback=1 shard=0/3 ranking=shared origin=cache overlap=2 shared=48 unique=80"));
        assert!(dashboard.contains("[Qwen3-MoE] MoE model detected: 128 experts, top-8"));
        assert!(dashboard.contains("[Qwen3-MoE] MoE split mode — leader=node-7 active=3 fallback=1 I am shard 0/3 (ranking=shared origin=cache, overlap=2); My experts: 128 (48 shared + 80 unique)"));
        assert!(!dashboard.contains('🧩'));
    }

    #[test]
    fn dashboard_formatter_renders_moe_status_messages_in_mesh_history() {
        let mut formatter =
            DashboardFormatter::with_state(DashboardState::with_mesh_event_limit(4));

        let dashboard = formatter
            .format(&OutputEvent::MoeStatus {
                model: "Qwen3-MoE".to_string(),
                status: MoeStatusSummary {
                    phase: "standing by".to_string(),
                    detail: "outside active MoE placement (leader=node-7 active=3 fallback=1)"
                        .to_string(),
                },
            })
            .expect("moe status render should succeed");

        assert!(dashboard.contains("[Qwen3-MoE] standing by: outside active MoE placement (leader=node-7 active=3 fallback=1)"));
        assert!(!dashboard.contains('🧩'));
    }

    #[test]
    fn json_formatter_includes_multi_model_mode_payload() {
        let mut formatter = JsonFormatter;
        let rendered = formatter
            .format(&OutputEvent::MultiModelMode {
                count: 2,
                models: vec!["Qwen2.5-32B".to_string(), "GLM-4.7-Flash".to_string()],
            })
            .expect("multi-model render should succeed");
        let value: Value = serde_json::from_str(rendered.trim_end()).expect("line should parse");

        assert_eq!(value["event"], "multi_model_mode");
        assert_eq!(value["count"], 2);
        assert_eq!(
            value["models"],
            serde_json::json!(["Qwen2.5-32B", "GLM-4.7-Flash"])
        );
    }

    #[test]
    fn json_formatter_includes_warning_context() {
        let mut formatter = JsonFormatter;
        let rendered = formatter
            .format(&OutputEvent::Warning {
                message: "Failed to start llama-server: bind error".to_string(),
                context: Some("model=Qwen3-32B mode=dense port=9337".to_string()),
            })
            .expect("warning render should succeed");
        let value: Value = serde_json::from_str(rendered.trim_end()).expect("line should parse");

        assert_eq!(value["event"], "warning");
        assert_eq!(value["warning"], "Failed to start llama-server: bind error");
        assert_eq!(value["context"], "model=Qwen3-32B mode=dense port=9337");
    }

    #[test]
    fn json_formatter_includes_info_context() {
        let mut formatter = JsonFormatter;
        let rendered = formatter
            .format(&OutputEvent::Info {
                message: "joined mesh".to_string(),
                context: Some("mesh=mesh-123".to_string()),
            })
            .expect("info render should succeed");
        let value: Value = serde_json::from_str(rendered.trim_end()).expect("line should parse");

        assert_eq!(value["event"], "info");
        assert_eq!(value["message"], "joined mesh");
        assert_eq!(value["context"], "mesh=mesh-123");
    }

    #[test]
    fn json_formatter_includes_moe_status_payload() {
        let mut formatter = JsonFormatter;
        let rendered = formatter
            .format(&OutputEvent::MoeStatus {
                model: "Qwen3-MoE".to_string(),
                status: MoeStatusSummary {
                    phase: "re-checking deployment".to_string(),
                    detail: "mesh changed".to_string(),
                },
            })
            .expect("moe status render should succeed");
        let value: Value = serde_json::from_str(rendered.trim_end()).expect("line should parse");

        assert_eq!(value["event"], "moe_status");
        assert_eq!(value["model"], "Qwen3-MoE");
        assert_eq!(value["status"]["phase"], "re-checking deployment");
        assert_eq!(value["status"]["detail"], "mesh changed");
    }

    #[test]
    fn json_formatter_includes_model_ready_port() {
        let mut formatter = JsonFormatter;
        let rendered = formatter
            .format(&OutputEvent::ModelReady {
                model: "Qwen3-32B".to_string(),
                internal_port: Some(3002),
                role: Some("host".to_string()),
            })
            .expect("model ready render should succeed");
        let value: Value = serde_json::from_str(rendered.trim_end()).expect("line should parse");

        assert_eq!(value["event"], "model_ready");
        assert_eq!(value["model"], "Qwen3-32B");
        assert_eq!(value["port"], serde_json::json!(3002));
        assert_eq!(value["internal_port"], serde_json::json!(3002));
        assert_eq!(value["role"], "host");
    }

    #[test]
    fn json_formatter_includes_host_elected_role_and_capacity() {
        let mut formatter = JsonFormatter;
        let rendered = formatter
            .format(&OutputEvent::HostElected {
                model: "Qwen3-32B".to_string(),
                host: "node-7".to_string(),
                role: Some("host".to_string()),
                capacity_gb: Some(24.0),
            })
            .expect("host election render should succeed");
        let value: Value = serde_json::from_str(rendered.trim_end()).expect("line should parse");

        assert_eq!(value["event"], "host_elected");
        assert_eq!(value["model"], "Qwen3-32B");
        assert_eq!(value["host"], "node-7");
        assert_eq!(value["role"], "host");
        assert_eq!(value["capacity_gb"], serde_json::json!(24.0));
    }

    #[test]
    fn json_formatter_includes_passive_mode_payload() {
        let mut formatter = JsonFormatter;
        let rendered = formatter
            .format(&OutputEvent::PassiveMode {
                role: "client".to_string(),
                status: RuntimeStatus::Ready,
                capacity_gb: None,
                models_on_disk: None,
                detail: Some("Client ready".to_string()),
            })
            .expect("passive mode render should succeed");
        let value: Value = serde_json::from_str(rendered.trim_end()).expect("line should parse");

        assert_eq!(value["event"], "passive_mode");
        assert_eq!(value["role"], "client");
        assert_eq!(value["status"], "ready");
        assert_eq!(value["capacity_gb"], Value::Null);
        assert_eq!(value["models_on_disk"], Value::Null);
        assert_eq!(value["detail"], "Client ready");
    }

    #[test]
    fn dashboard_formatter_keeps_pinned_sections_and_bounds_mesh_history() {
        let mut formatter =
            DashboardFormatter::with_state(DashboardState::with_mesh_event_limit(2));

        formatter
            .format(&OutputEvent::Startup {
                version: "v0.64.0".to_string(),
                message: None,
            })
            .expect("startup render should succeed");
        formatter
            .format(&OutputEvent::LlamaStarting {
                model: Some("Qwen3.6-35B".to_string()),
                http_port: 43683,
                ctx_size: Some(8192),
                log_path: Some("/tmp/llama.log".to_string()),
            })
            .expect("llama render should succeed");
        formatter
            .format(&OutputEvent::RpcServerStarting {
                port: 43683,
                device: "CUDA0".to_string(),
                log_path: Some("/tmp/rpc.log".to_string()),
            })
            .expect("rpc render should succeed");
        formatter
            .format(&OutputEvent::ModelReady {
                model: "Qwen3.6-35B".to_string(),
                internal_port: Some(38373),
                role: Some("host".to_string()),
            })
            .expect("model render should succeed");
        formatter
            .format(&OutputEvent::RuntimeReady {
                api_url: "http://localhost:9337".to_string(),
                console_url: Some("http://localhost:3131".to_string()),
                api_port: 9337,
                console_port: Some(3131),
                models_count: Some(1),
                pi_command: Some("pi --provider mesh --model Qwen3.6-35B".to_string()),
                goose_command: Some(
                    "GOOSE_PROVIDER=openai OPENAI_HOST=http://localhost:9337 OPENAI_API_KEY=mesh GOOSE_MODEL=Qwen3.6-35B goose session"
                        .to_string(),
                ),
            })
            .expect("api render should succeed");
        formatter
            .format(&OutputEvent::PeerJoined {
                peer_id: "peer-1".to_string(),
                label: None,
            })
            .expect("peer render should succeed");
        let dashboard = formatter
            .format(&OutputEvent::PeerJoined {
                peer_id: "peer-2".to_string(),
                label: None,
            })
            .expect("peer render should succeed");

        assert!(dashboard.contains("Running llama.cpp instances"));
        assert!(dashboard.contains("Running models"));
        assert!(dashboard.contains("Running webserver"));
        assert!(dashboard.contains("Running API"));
        assert!(dashboard.contains("Mesh events (latest 2)"));
        assert!(dashboard.contains("llama-server   starting   port=43683"));
        assert!(dashboard.contains("model=Qwen3.6-35B"));
        assert!(dashboard.contains("ctx=8192"));
        assert!(dashboard.contains("logs=/tmp/llama.log"));
        assert!(dashboard.contains("OpenAI-compatible API   ready   http://localhost:9337"));
        assert!(dashboard.contains("Console   ready   http://localhost:3131"));
        assert!(dashboard.contains("pi:    pi --provider mesh --model Qwen3.6-35B"));
        assert!(dashboard.contains("goose: GOOSE_PROVIDER=openai OPENAI_HOST=http://localhost:9337 OPENAI_API_KEY=mesh GOOSE_MODEL=Qwen3.6-35B goose session"));
        assert!(dashboard.contains("peer-1"));
        assert!(dashboard.contains("peer-2"));
        assert!(!dashboard.contains("mesh-llm starting"));
    }

    #[test]
    fn dashboard_and_json_formatters_cover_all_output_variants_without_panics() {
        let events = sample_events_covering_all_variants();
        let mut pretty = DashboardFormatter::with_state(DashboardState::with_mesh_event_limit(64));
        let mut json = JsonFormatter;

        for event in &events {
            let dashboard_rendered = pretty
                .format(event)
                .expect("pretty formatter should render every event variant");
            assert!(
                dashboard_rendered.contains("Running llama.cpp instances")
                    && dashboard_rendered.contains("Running models")
                    && dashboard_rendered.contains("Running webserver")
                    && dashboard_rendered.contains("Running API")
                    && dashboard_rendered.contains("Mesh events"),
                "pretty formatter should keep pinned sections for {event:?}"
            );

            let json_rendered = json
                .format(event)
                .expect("json formatter should render every event variant");
            let value = parse_json_line(&json_rendered);
            assert_required_json_envelope(&value, event);
        }
    }

    #[test]
    fn json_formatter_includes_required_fields_for_every_output_variant() {
        let events = sample_events_covering_all_variants();
        let mut formatter = JsonFormatter;

        for event in &events {
            let rendered = formatter
                .format(event)
                .expect("json formatter should render every event variant");
            let value = parse_json_line(&rendered);
            assert_required_json_envelope(&value, event);
        }
    }

    #[test]
    fn json_formatter_preserves_representative_optional_metadata_fields() {
        let mut formatter = JsonFormatter;

        let model_ready = parse_json_line(
            &formatter
                .format(&OutputEvent::ModelReady {
                    model: "Qwen3-32B".to_string(),
                    internal_port: Some(38373),
                    role: Some("host".to_string()),
                })
                .expect("model ready render should succeed"),
        );
        assert_eq!(model_ready["model"], "Qwen3-32B");
        assert_eq!(model_ready["port"], 38373);
        assert_eq!(model_ready["internal_port"], 38373);
        assert_eq!(model_ready["role"], "host");

        let rpc_starting = parse_json_line(
            &formatter
                .format(&OutputEvent::RpcServerStarting {
                    port: 43683,
                    device: "CUDA0".to_string(),
                    log_path: Some("/tmp/rpc.log".to_string()),
                })
                .expect("rpc startup render should succeed"),
        );
        assert_eq!(rpc_starting["port"], 43683);
        assert_eq!(rpc_starting["device"], "CUDA0");
        assert_eq!(rpc_starting["log_path"], "/tmp/rpc.log");

        let llama_starting = parse_json_line(
            &formatter
                .format(&OutputEvent::LlamaStarting {
                    model: Some("Qwen3-32B".to_string()),
                    http_port: 8001,
                    ctx_size: Some(8192),
                    log_path: Some("/tmp/llama.log".to_string()),
                })
                .expect("llama startup render should succeed"),
        );
        assert_eq!(llama_starting["model"], "Qwen3-32B");
        assert_eq!(llama_starting["http_port"], 8001);
        assert_eq!(llama_starting["ctx_size"], 8192);
        assert_eq!(llama_starting["log_path"], "/tmp/llama.log");

        let info = parse_json_line(
            &formatter
                .format(&OutputEvent::Info {
                    message: "joined mesh".to_string(),
                    context: Some("mesh=mesh-123".to_string()),
                })
                .expect("info render should succeed"),
        );
        assert_eq!(info["context"], "mesh=mesh-123");

        let warning = parse_json_line(
            &formatter
                .format(&OutputEvent::Warning {
                    message: "bind warning".to_string(),
                    context: Some("model=Qwen3-32B".to_string()),
                })
                .expect("warning render should succeed"),
        );
        assert_eq!(warning["warning"], "bind warning");
        assert_eq!(warning["context"], "model=Qwen3-32B");

        let runtime_ready = parse_json_line(
            &formatter
                .format(&OutputEvent::RuntimeReady {
                    api_url: "http://localhost:9337".to_string(),
                    console_url: Some("http://localhost:3131".to_string()),
                    api_port: 9337,
                    console_port: Some(3131),
                    models_count: Some(2),
                    pi_command: Some("pi --provider mesh --model Qwen3-32B".to_string()),
                    goose_command: Some("goose session".to_string()),
                })
                .expect("runtime ready render should succeed"),
        );
        assert_eq!(runtime_ready["api_port"], 9337);
        assert_eq!(runtime_ready["console_port"], 3131);
        assert_eq!(runtime_ready["console_url"], "http://localhost:3131");
        assert_eq!(runtime_ready["models_count"], 2);
        assert_eq!(
            runtime_ready["pi_command"],
            "pi --provider mesh --model Qwen3-32B"
        );
        assert_eq!(runtime_ready["goose_command"], "goose session");
    }

    #[test]
    fn dashboard_formatter_mesh_history_keeps_timestamps_and_emoji_readable() {
        let mut formatter =
            DashboardFormatter::with_state(DashboardState::with_mesh_event_limit(8));

        formatter
            .format(&OutputEvent::InviteToken {
                token: "invite-token-1234567890".to_string(),
                mesh_id: "mesh-abc".to_string(),
                mesh_name: None,
            })
            .expect("invite render should succeed");
        formatter
            .format(&OutputEvent::DiscoveryStarting {
                source: "Nostr re-discovery".to_string(),
            })
            .expect("discovery start render should succeed");
        formatter
            .format(&OutputEvent::Warning {
                message: "legacy capacity estimate may be stale".to_string(),
                context: Some("model=Qwen3-32B".to_string()),
            })
            .expect("warning render should succeed");
        let dashboard = formatter
            .format(&OutputEvent::MoeStatus {
                model: "Qwen3-MoE".to_string(),
                status: MoeStatusSummary {
                    phase: "ranking".to_string(),
                    detail: "waiting for shared expert rankings".to_string(),
                },
            })
            .expect("moe status render should succeed");

        let mesh_lines: Vec<&str> = dashboard
            .lines()
            .filter(|line| line.starts_with("│ "))
            .filter(|line| {
                line.contains("Invite created")
                    || line.contains("discovering mesh")
                    || line.contains("legacy capacity estimate may be stale")
                    || line.contains("waiting for shared expert rankings")
            })
            .collect();

        assert_eq!(
            mesh_lines.len(),
            4,
            "expected four readable mesh history lines"
        );
        for line in &mesh_lines {
            let timestamp: String = line.chars().skip(2).take(8).collect();
            assert_hh_mm_ss(&timestamp);
        }

        assert!(dashboard.contains("Invite created for mesh mesh-abc: invite-token-1234567890"));
        assert!(dashboard.contains("discovering mesh via Nostr re-discovery"));
        assert!(dashboard.contains("model=Qwen3-32B: legacy capacity estimate may be stale"));
        assert!(dashboard.contains("[Qwen3-MoE] ranking: waiting for shared expert rankings"));
        assert!(!dashboard.contains('📡'));
        assert!(!dashboard.contains('🔍'));
        assert!(!dashboard.contains("⚠️"));
        assert!(!dashboard.contains('🧩'));
    }

    #[test]
    fn dashboard_formatter_keeps_long_names_paths_and_tokens_readable() {
        let long_model = "Qwen3.6-35B-A3B-UD-Q4_K_XL-with-extra-routing-suffix";
        let long_token =
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.super.long.mesh.invite.token.payload";
        let long_rpc_log = "/Users/ndizazzo/.mesh-llm/runtime/3845607/logs/rpc-server-43683-with-a-very-long-name.log";
        let long_llama_log = "/Users/ndizazzo/.mesh-llm/runtime/3845607/logs/llama-server-8001-with-a-very-long-name.log";
        let mut formatter =
            DashboardFormatter::with_state(DashboardState::with_mesh_event_limit(8));

        formatter
            .format(&OutputEvent::InviteToken {
                token: long_token.to_string(),
                mesh_id: "mesh-readable".to_string(),
                mesh_name: None,
            })
            .expect("invite render should succeed");
        formatter
            .format(&OutputEvent::RpcServerStarting {
                port: 43683,
                device: "CUDA0".to_string(),
                log_path: Some(long_rpc_log.to_string()),
            })
            .expect("rpc render should succeed");
        formatter
            .format(&OutputEvent::LlamaStarting {
                model: Some(long_model.to_string()),
                http_port: 8001,
                ctx_size: Some(8192),
                log_path: Some(long_llama_log.to_string()),
            })
            .expect("llama render should succeed");
        let dashboard = formatter
            .format(&OutputEvent::ModelReady {
                model: long_model.to_string(),
                internal_port: Some(38373),
                role: Some("host".to_string()),
            })
            .expect("model ready render should succeed");

        assert!(dashboard.contains(long_model));
        assert!(dashboard.contains(long_token));
        assert!(dashboard.contains(long_rpc_log));
        assert!(dashboard.contains(long_llama_log));
        assert!(dashboard.contains("Mesh events (latest 8)"));
        assert!(dashboard.contains("│ rpc-server   starting   port=43683   device=CUDA0"));
        assert!(dashboard.contains("│              logs=/Users/ndizazzo/.mesh-llm/runtime/3845607/logs/rpc-server-43683-with-a-very-long-name.log"));
        assert!(dashboard.contains("│ llama-server   starting   port=8001"));
        assert!(dashboard.contains("model=Qwen3.6-35B-A3B-UD-Q4_K_XL-with-extra-routing-suffix"));
        assert!(dashboard.contains("ctx=8192"));
        assert!(dashboard.contains("│              logs=/Users/ndizazzo/.mesh-llm/runtime/3845607/logs/llama-server-8001-with-a-very-long-name.log"));
        assert!(dashboard.contains("│ Qwen3.6-35B-A3B-UD-Q4_K_XL-with-extra-routing-suffix   ready   port=38373   role=host"));
        assert!(dashboard
            .lines()
            .any(|line| line.starts_with("┌ Running llama.cpp instances ")));
        assert!(dashboard
            .lines()
            .any(|line| line.starts_with("┌ Running models ")));
        assert!(dashboard
            .lines()
            .any(|line| line.starts_with("┌ Mesh events (latest 8) ")));
    }
}
