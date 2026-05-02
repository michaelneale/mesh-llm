use std::{
    collections::{BTreeMap, VecDeque},
    future::Future,
    io,
    net::{IpAddr, Ipv4Addr, SocketAddr, TcpListener, TcpStream, ToSocketAddrs},
    os::fd::AsRawFd,
    path::PathBuf,
    ptr,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        mpsc, Arc, Mutex,
    },
    thread,
    time::{Duration, Instant},
};

use crate::{
    cli::ServeBinaryArgs,
    config::{load_json, validate_config},
    kv_integration::KvStageIntegration,
    openai::{self, EmbeddedOpenAiArgs},
    runtime_state::{kv_desc_annotations, load_runtime, RuntimeSessionStats, RuntimeState},
    telemetry::{lifecycle_attrs, now_unix_nanos, Telemetry},
};
use anyhow::{anyhow, bail, Context, Result};
use serde_json::{json, Value};
use skippy_metrics::attr;
use skippy_protocol::{
    binary::{
        read_stage_message, recv_reply, send_ready, send_reply_ack, send_reply_ack_with_stats,
        send_reply_predicted_tokens_with_stats, send_reply_predicted_with_stats, state_flags,
        write_stage_message, StageReplyStats, StageSamplingConfig, StageWireMessage,
        WireActivationDType, WireMessageKind, WireReplyKind,
    },
    LoadMode, MessageBase, StageConfig, StageTopology, SCHEMA_VERSION,
};
use skippy_runtime::{
    ActivationDesc, ActivationFrame, LogitBias, ModelInfo, RuntimeActivationDType,
    RuntimeActivationLayout, SamplingConfig, MAX_LOGIT_BIAS,
};
use socket2::{Domain, Protocol, SockAddr, Socket, Type};

static BINARY_SESSION_COUNTER: AtomicU64 = AtomicU64::new(1);

struct BinaryKvRecordCandidate {
    attrs: BTreeMap<String, Value>,
    session_id: String,
    stage_index: u32,
    page_id: String,
    identity: crate::kv_proto::PageIdentity,
    token_start: u64,
    token_count: u64,
}

#[derive(Default)]
struct BinaryKvLookupResult {
    restored_tokens: usize,
    stats: StageReplyStats,
}

#[derive(Clone)]
pub struct BinaryStageOptions {
    pub config: StageConfig,
    pub topology: Option<StageTopology>,
    pub bind_addr: SocketAddr,
    pub activation_width: i32,
    pub wire_dtype: WireActivationDType,
    pub metrics_otlp_grpc: Option<String>,
    pub telemetry_queue_capacity: usize,
    pub telemetry_level: crate::telemetry::TelemetryLevel,
    pub max_inflight: usize,
    pub reply_credit_limit: Option<usize>,
    pub async_prefill_forward: bool,
    pub downstream_wire_condition: WireCondition,
    pub downstream_connect_timeout_secs: u64,
    pub openai: Option<EmbeddedOpenAiStageOptions>,
}

#[derive(Clone)]
pub struct EmbeddedOpenAiStageOptions {
    pub bind_addr: SocketAddr,
    pub model_id: Option<String>,
    pub default_max_tokens: u32,
    pub generation_concurrency: usize,
    pub prefill_chunk_size: usize,
    pub prefill_chunk_policy: String,
    pub prefill_chunk_schedule: Option<String>,
    pub prefill_adaptive_start: usize,
    pub prefill_adaptive_step: usize,
    pub prefill_adaptive_max: usize,
    pub draft_model_path: Option<PathBuf>,
    pub speculative_window: usize,
    pub adaptive_speculative_window: bool,
    pub draft_n_gpu_layers: Option<i32>,
}

impl BinaryStageOptions {
    pub fn from_cli_args(args: ServeBinaryArgs) -> Result<Self> {
        if args.activation_width <= 0 {
            bail!("activation_width must be greater than zero");
        }
        if args.openai_generation_concurrency == 0 {
            bail!("--openai-generation-concurrency must be greater than zero");
        }
        if args.openai_prefill_chunk_size == 0 {
            bail!("--openai-prefill-chunk-size must be greater than zero");
        }
        let wire_dtype = parse_wire_dtype(&args.activation_wire_dtype)?;
        let downstream_wire_condition =
            WireCondition::new(args.downstream_wire_delay_ms, args.downstream_wire_mbps)?;
        let config = load_json::<StageConfig>(&args.config)
            .with_context(|| format!("load stage config {}", args.config.display()))?;
        let topology = match args.topology.as_ref() {
            Some(path) => Some(
                load_json::<StageTopology>(path)
                    .with_context(|| format!("load topology {}", path.display()))?,
            ),
            None => None,
        };
        let bind_addr = args.bind_addr.unwrap_or(config.bind_addr.parse()?);
        let openai = args
            .openai_bind_addr
            .map(|bind_addr| EmbeddedOpenAiStageOptions {
                bind_addr,
                model_id: args.openai_model_id,
                default_max_tokens: args.openai_default_max_tokens,
                generation_concurrency: args.openai_generation_concurrency,
                prefill_chunk_size: args.openai_prefill_chunk_size,
                prefill_chunk_policy: args.openai_prefill_chunk_policy,
                prefill_chunk_schedule: args.openai_prefill_chunk_schedule,
                prefill_adaptive_start: args.openai_prefill_adaptive_start,
                prefill_adaptive_step: args.openai_prefill_adaptive_step,
                prefill_adaptive_max: args.openai_prefill_adaptive_max,
                draft_model_path: args.openai_draft_model_path,
                speculative_window: args.openai_speculative_window,
                adaptive_speculative_window: args.openai_adaptive_speculative_window,
                draft_n_gpu_layers: args.openai_draft_n_gpu_layers,
            });
        Ok(Self {
            config,
            topology,
            bind_addr,
            activation_width: args.activation_width,
            wire_dtype,
            metrics_otlp_grpc: args.metrics_otlp_grpc,
            telemetry_queue_capacity: args.telemetry_queue_capacity,
            telemetry_level: args.telemetry_level,
            max_inflight: args.max_inflight,
            reply_credit_limit: args.reply_credit_limit,
            async_prefill_forward: args.async_prefill_forward,
            downstream_wire_condition,
            downstream_connect_timeout_secs: args.downstream_connect_timeout_secs,
            openai,
        })
    }
}

pub async fn serve_binary(args: ServeBinaryArgs) -> Result<()> {
    serve_binary_stage(BinaryStageOptions::from_cli_args(args)?).await
}

pub async fn serve_binary_stage(options: BinaryStageOptions) -> Result<()> {
    serve_binary_stage_with_shutdown(options, std::future::pending::<()>()).await
}

pub async fn serve_binary_stage_with_shutdown(
    options: BinaryStageOptions,
    shutdown: impl Future<Output = ()> + Send + 'static,
) -> Result<()> {
    let stop = Arc::new(AtomicBool::new(false));
    let stop_task = tokio::spawn({
        let stop = stop.clone();
        async move {
            shutdown.await;
            stop.store(true, Ordering::SeqCst);
        }
    });
    let result = run_binary_stage(options, stop);
    stop_task.abort();
    result
}

fn run_binary_stage(options: BinaryStageOptions, shutdown: Arc<AtomicBool>) -> Result<()> {
    let BinaryStageOptions {
        config,
        topology,
        bind_addr,
        activation_width,
        wire_dtype,
        metrics_otlp_grpc,
        telemetry_queue_capacity,
        telemetry_level,
        max_inflight,
        reply_credit_limit,
        async_prefill_forward,
        downstream_wire_condition,
        downstream_connect_timeout_secs,
        openai,
    } = options;
    validate_config(&config, topology.as_ref())?;
    let telemetry = Telemetry::new(
        metrics_otlp_grpc,
        telemetry_queue_capacity,
        config.clone(),
        telemetry_level,
    );
    telemetry.emit("stage.binary_server_start", lifecycle_attrs(&config));
    let runtime = load_runtime(&config)?.context("binary stage server requires model_path")?;
    let kv_cache_safe = model_supports_kv_only_cache(&config);
    let kv = if kv_cache_safe {
        KvStageIntegration::from_config(&config)?.map(Arc::new)
    } else {
        let mut attrs = lifecycle_attrs(&config);
        attrs.insert(
            "skippy.kv.decision".to_string(),
            json!("disabled_recurrent_state_required"),
        );
        telemetry.emit("stage.binary_kv_disabled", attrs);
        None
    };
    let listener = TcpListener::bind(bind_addr)?;
    listener.set_nonblocking(true)?;
    if let Some(openai_options) = openai {
        if config.stage_index != 0 || config.layer_start != 0 {
            bail!("--openai-bind-addr is only supported on stage 0");
        }
        let openai_config = config.clone();
        let openai_runtime = runtime.clone();
        let openai_telemetry = telemetry.clone();
        tokio::spawn(async move {
            if let Err(error) = openai::serve_embedded_openai(EmbeddedOpenAiArgs {
                bind_addr: openai_options.bind_addr,
                config: openai_config,
                runtime: openai_runtime,
                model_id: openai_options.model_id,
                default_max_tokens: openai_options.default_max_tokens,
                generation_concurrency: openai_options.generation_concurrency,
                prefill_chunk_size: openai_options.prefill_chunk_size,
                prefill_chunk_policy: openai_options.prefill_chunk_policy,
                prefill_chunk_schedule: openai_options.prefill_chunk_schedule,
                prefill_adaptive_start: openai_options.prefill_adaptive_start,
                prefill_adaptive_step: openai_options.prefill_adaptive_step,
                prefill_adaptive_max: openai_options.prefill_adaptive_max,
                draft_model_path: openai_options.draft_model_path,
                speculative_window: openai_options.speculative_window,
                adaptive_speculative_window: openai_options.adaptive_speculative_window,
                draft_n_gpu_layers: openai_options.draft_n_gpu_layers,
                activation_width,
                wire_dtype,
                downstream_connect_timeout_secs,
                downstream_wire_condition,
                telemetry: openai_telemetry,
            })
            .await
            {
                eprintln!("embedded OpenAI server failed: {error:#}");
            }
        });
    }
    println!(
        "skippy-server listening: binary={} stage_id={} layer_range={}..{} activation_width={} dtype={:?}",
        bind_addr,
        config.stage_id,
        config.layer_start,
        config.layer_end,
        activation_width,
        wire_dtype,
    );

    while !shutdown.load(Ordering::SeqCst) {
        let (mut upstream, _) = match listener.accept() {
            Ok(conn) => conn,
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                thread::sleep(Duration::from_millis(50));
                continue;
            }
            Err(error) => return Err(error).context("accept binary stage connection"),
        };
        upstream.set_nodelay(true).ok();
        let peer_addr = upstream.peer_addr().ok();
        let config = config.clone();
        let topology = topology.clone();
        let runtime = runtime.clone();
        let kv = kv.clone();
        let telemetry = telemetry.clone();
        thread::spawn(move || {
            let connection_result = (|| -> Result<()> {
                let downstream =
                    connect_binary_downstream(&config, downstream_connect_timeout_secs)?;
                handle_binary_connection(
                    &config,
                    topology.as_ref(),
                    &runtime,
                    kv.as_ref(),
                    &telemetry,
                    &mut upstream,
                    downstream,
                    activation_width,
                    wire_dtype,
                    max_inflight,
                    reply_credit_limit,
                    async_prefill_forward,
                    downstream_wire_condition,
                )
            })()
            .context("binary stage connection failed");
            if let Err(error) = connection_result {
                let mut attrs = lifecycle_attrs(&config);
                if let Some(peer_addr) = peer_addr {
                    attrs.insert("llama_stage.peer_addr".to_string(), json!(peer_addr));
                }
                attrs.insert("llama_stage.error".to_string(), json!(error.to_string()));
                eprintln!("{error:#}");
                telemetry.emit("stage.binary_connection_error", attrs);
            }
        });
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn handle_binary_connection(
    config: &StageConfig,
    topology: Option<&StageTopology>,
    runtime: &Arc<Mutex<RuntimeState>>,
    kv: Option<&Arc<KvStageIntegration>>,
    telemetry: &Telemetry,
    upstream: &mut TcpStream,
    mut downstream: Option<TcpStream>,
    activation_width: i32,
    wire_dtype: WireActivationDType,
    max_inflight: usize,
    reply_credit_limit: Option<usize>,
    async_prefill_forward: bool,
    downstream_wire_condition: WireCondition,
) -> Result<()> {
    if let Some(downstream) = downstream.as_mut() {
        skippy_protocol::binary::recv_ready(&mut *downstream)
            .context("downstream binary stage did not become ready")?;
    }
    send_ready(&mut *upstream).context("failed to send binary ready")?;

    let connection_session_id = BINARY_SESSION_COUNTER.fetch_add(1, Ordering::Relaxed);
    let max_deferred_prefill_replies =
        reply_credit_limit.unwrap_or_else(|| max_inflight.saturating_sub(1));
    let mut pending_prefill_replies = 0usize;
    let mut pending_reply_stats = StageReplyStats::default();
    let mut request_summary = BinaryRequestSummary::default();
    let mut record_workers = Vec::new();
    let mut async_forwarder = if async_prefill_forward {
        downstream
            .as_ref()
            .map(|downstream| AsyncForwarder::new(downstream, telemetry.clone()))
            .transpose()
            .context("create async activation forwarder")?
    } else {
        None
    };

    loop {
        let message = read_stage_message(&mut *upstream, activation_width)
            .context("read binary stage message")?;
        let message_start_unix_nanos = now_unix_nanos() as u64;
        let message_started = Instant::now();
        let session_id = binary_message_session_id(connection_session_id, &message);
        let session_key = session_id.to_string();
        let mut recv_attrs = binary_message_attrs(config, session_id, &message);
        recv_attrs.insert(
            "skippy.activation_bytes".to_string(),
            json!(message.activation.len()),
        );
        telemetry.emit_debug("stage.binary_recv", recv_attrs);

        if message.kind == WireMessageKind::Stop {
            if pending_prefill_replies != 0 {
                bail!("cannot stop with {pending_prefill_replies} deferred prefill replies");
            }
            let mut stop_stats = std::mem::take(&mut pending_reply_stats);
            drain_record_workers(&mut record_workers);
            request_summary.emit(telemetry, config, session_id);
            request_summary = BinaryRequestSummary::default();
            if let Some(downstream) = downstream.as_mut() {
                if let Some(forwarder) = async_forwarder.as_mut() {
                    forwarder
                        .flush()
                        .context("flush async forwards before stop")?;
                }
                write_stage_message_conditioned(
                    &mut *downstream,
                    &message,
                    wire_dtype,
                    downstream_wire_condition,
                )
                .context("forward binary stop")?;
                let reply = recv_reply(&mut *downstream).context("stop downstream ACK")?;
                if reply.kind != WireReplyKind::Ack {
                    bail!("stop expected downstream ACK");
                }
                stop_stats.merge(reply.stats);
            }
            let reset_start_unix_nanos = now_unix_nanos() as u64;
            let reset_timer = Instant::now();
            let lock_timer = Instant::now();
            let mut runtime = runtime.lock().expect("runtime lock poisoned");
            let runtime_lock_wait_ms = elapsed_ms(lock_timer);
            let drop_stats = runtime
                .drop_session_timed(&session_key)
                .context("reset binary stage session")?;
            drop(runtime);
            let reset_end_unix_nanos = now_unix_nanos() as u64;
            let mut reset_attrs = binary_message_attrs(config, session_id, &message);
            reset_attrs.insert(
                "llama_stage.runtime_lock_wait_ms".to_string(),
                json!(runtime_lock_wait_ms),
            );
            reset_attrs.insert(
                "llama_stage.session_reset_ms".to_string(),
                json!(drop_stats.reset_ms),
            );
            reset_attrs.insert(
                "llama_stage.session_reset".to_string(),
                json!(drop_stats.reset_session),
            );
            reset_attrs.insert(
                "llama_stage.elapsed_ms".to_string(),
                json!(elapsed_ms(reset_timer)),
            );
            insert_runtime_session_stats(
                &mut reset_attrs,
                "llama_stage.runtime_sessions_after",
                drop_stats.stats_after,
            );
            telemetry.emit_debug_span(
                "stage.binary_session_stop",
                reset_attrs,
                reset_start_unix_nanos,
                reset_end_unix_nanos,
            );
            send_reply_ack_with_stats(&mut *upstream, stop_stats).context("send stop ACK")?;
            continue;
        }

        if message.kind.is_session_control() {
            let control_started = Instant::now();
            let mut control_stats = std::mem::take(&mut pending_reply_stats);
            let flush_started = Instant::now();
            if let Some(forwarder) = async_forwarder.as_mut() {
                forwarder
                    .flush()
                    .context("flush async forwards before session control")?;
            }
            let flush_us = elapsed_us(flush_started);
            let pending_prefill_before_control = pending_prefill_replies;
            let drain_started = Instant::now();
            drain_deferred_prefill_replies(
                downstream.as_mut(),
                &mut pending_prefill_replies,
                &mut control_stats,
            )
            .context("drain deferred replies before session control")?;
            let prefill_drain_us = elapsed_us(drain_started);
            let prefill_drained_replies =
                pending_prefill_before_control.saturating_sub(pending_prefill_replies);
            let local_started = Instant::now();
            {
                let mut runtime = runtime.lock().expect("runtime lock poisoned");
                match message.kind {
                    WireMessageKind::CheckpointSession => runtime
                        .checkpoint_session(&session_key)
                        .context("checkpoint binary stage session")?,
                    WireMessageKind::RestoreSession => runtime
                        .restore_session(&session_key)
                        .context("restore binary stage session")?,
                    _ => unreachable!("session control checked above"),
                }
            }
            let local_us = elapsed_us(local_started);
            let mut downstream_write_us = 0;
            let mut downstream_wait_us = 0;
            if let Some(downstream) = downstream.as_mut() {
                let downstream_write_started = Instant::now();
                write_stage_message_conditioned(
                    &mut *downstream,
                    &message,
                    wire_dtype,
                    downstream_wire_condition,
                )
                .context("forward session control")?;
                downstream_write_us = elapsed_us(downstream_write_started);
                let downstream_wait_started = Instant::now();
                let reply =
                    recv_reply(&mut *downstream).context("session control downstream ACK")?;
                downstream_wait_us = elapsed_us(downstream_wait_started);
                if reply.kind != WireReplyKind::Ack {
                    bail!("session control expected downstream ACK");
                }
                control_stats.merge(reply.stats);
            }
            record_session_control_timing(
                &mut control_stats,
                message.kind,
                SessionControlTiming {
                    flush_us,
                    prefill_drain_us,
                    local_us,
                    downstream_write_us,
                    downstream_wait_us,
                    total_us: elapsed_us(control_started),
                    prefill_drained_replies: prefill_drained_replies as i64,
                },
            );
            send_reply_ack_with_stats(&mut *upstream, control_stats)
                .context("session control ack")?;
            continue;
        }

        if message.kind == WireMessageKind::StateImport {
            {
                let mut runtime = runtime.lock().expect("runtime lock poisoned");
                if (message.state.flags & state_flags::FULL_STATE) != 0 {
                    runtime
                        .import_full_state(&session_key, &message.raw_bytes)
                        .context("import local full stage state")?;
                } else {
                    runtime
                        .import_state(&session_key, &message.raw_bytes)
                        .context("import local stage state")?;
                }
            }
            if let Some(downstream) = downstream.as_mut() {
                if let Some(forwarder) = async_forwarder.as_mut() {
                    forwarder
                        .flush()
                        .context("flush async forwards before state import")?;
                }
                write_stage_message_conditioned(
                    &mut *downstream,
                    &message,
                    wire_dtype,
                    downstream_wire_condition,
                )
                .context("forward state import")?;
                let reply =
                    recv_reply(&mut *downstream).context("state import downstream reply")?;
                if reply.kind != WireReplyKind::Ack {
                    bail!("state import expected downstream ACK");
                }
            }
            send_reply_ack(&mut *upstream).context("state import ack")?;
            continue;
        }

        if message.kind == WireMessageKind::StateExport {
            if pending_prefill_replies != 0 {
                bail!(
                    "cannot export state with {pending_prefill_replies} deferred prefill replies"
                );
            }
            if downstream.is_some() {
                bail!("binary state export is supported only on a directly addressed stage");
            }
            if let Some(forwarder) = async_forwarder.as_mut() {
                forwarder
                    .flush()
                    .context("flush async forwards before state export")?;
            }
            let state_bytes = {
                let mut runtime = runtime.lock().expect("runtime lock poisoned");
                if (message.state.flags & state_flags::FULL_STATE) != 0 {
                    runtime
                        .export_full_state(&session_key)
                        .context("export local full stage state")?
                } else {
                    runtime
                        .export_state(&session_key)
                        .context("export local stage state")?
                }
            };
            let mut state = message.state;
            state.source_stage_index = config.stage_index as i32;
            let export = StageWireMessage {
                kind: WireMessageKind::StateImport,
                pos_start: message.pos_start,
                token_count: i32::try_from(state_bytes.len())
                    .context("exported state exceeds binary wire length range")?,
                state,
                request_id: message.request_id,
                session_id: message.session_id,
                sampling: None,
                tokens: Vec::new(),
                activation: Vec::new(),
                raw_bytes: state_bytes,
            };
            write_stage_message(&mut *upstream, &export, wire_dtype)
                .context("send state export payload")?;
            continue;
        }

        if !message.state.matches_kind(message.kind) {
            bail!("binary stage state does not match message kind");
        }

        let requires_predicted = message.kind.requires_predicted_reply();
        let early_prefill_ack = message.kind.is_prefill() && !requires_predicted;
        let mut upstream_reply_start_unix_nanos = None;
        let mut upstream_reply_end_unix_nanos = None;
        let mut early_reply_ms = 0.0;
        if early_prefill_ack {
            let reply_start_unix_nanos = now_unix_nanos() as u64;
            upstream_reply_start_unix_nanos = Some(reply_start_unix_nanos);
            let reply_started = Instant::now();
            send_reply_ack(&mut *upstream).context("early prefill ack")?;
            upstream_reply_end_unix_nanos = Some(now_unix_nanos() as u64);
            early_reply_ms = elapsed_ms(reply_started);
        }

        let input_decode_started = Instant::now();
        let input = input_activation_frame(config, topology, &message, activation_width)?;
        let input_activation_decode_ms = if message.activation.is_empty() {
            0.0
        } else {
            elapsed_ms(input_decode_started)
        };
        let token_ids = token_sideband_or_fill(&message)?;
        let mut message_reply_stats = StageReplyStats::default();
        let lookup_result = maybe_lookup_binary_prefill(
            config,
            runtime,
            kv,
            telemetry,
            &session_key,
            &message,
            &token_ids,
        );
        message_reply_stats.merge(lookup_result.stats);
        let restored_prefill =
            lookup_result.restored_tokens >= token_ids.len() && !token_ids.is_empty();
        let executable_token_ids = if message.kind.is_prefill()
            && lookup_result.restored_tokens > 0
            && lookup_result.restored_tokens < token_ids.len()
            && config.layer_start == 0
        {
            &token_ids[lookup_result.restored_tokens..]
        } else {
            &token_ids
        };
        let compute_start_unix_nanos: u64;
        let compute_end_unix_nanos: u64;
        let mut runtime_lock_wait_ms = 0.0;
        let mut runtime_lock_hold_ms = 0.0;
        let mut runtime_lock_acquires = 0usize;
        let mut runtime_sessions_before = None;
        let mut runtime_sessions_after = None;
        let (predicted_token, predicted_tokens, output, compute_ms) = if restored_prefill {
            let now = now_unix_nanos() as u64;
            compute_start_unix_nanos = now;
            compute_end_unix_nanos = now;
            (
                message.state.current_token,
                Vec::new(),
                empty_activation_frame(config, &message),
                0.0,
            )
        } else {
            if message.kind == WireMessageKind::VerifySpan
                && (message.state.flags & state_flags::SKIP_VERIFY_CHECKPOINT) == 0
            {
                let checkpoint_started = Instant::now();
                {
                    let mut runtime = runtime.lock().expect("runtime lock poisoned");
                    runtime
                        .checkpoint_session(&session_key)
                        .context("checkpoint binary stage session before verify span")?;
                }
                let checkpoint_us = elapsed_us(checkpoint_started);
                record_session_control_timing(
                    &mut message_reply_stats,
                    WireMessageKind::CheckpointSession,
                    SessionControlTiming {
                        flush_us: 0,
                        prefill_drain_us: 0,
                        local_us: checkpoint_us,
                        downstream_write_us: 0,
                        downstream_wait_us: 0,
                        total_us: checkpoint_us,
                        prefill_drained_replies: 0,
                    },
                );
            }
            compute_start_unix_nanos = now_unix_nanos() as u64;
            let compute_started = Instant::now();
            let result = {
                let lock_started = Instant::now();
                let mut runtime = runtime.lock().expect("runtime lock poisoned");
                runtime_lock_wait_ms = elapsed_ms(lock_started);
                runtime_lock_acquires = 1;
                let lock_hold_started = Instant::now();
                runtime_sessions_before = Some(runtime.session_stats());
                let result = run_binary_stage_message(
                    &mut runtime,
                    &session_key,
                    &message,
                    executable_token_ids,
                    input.as_ref(),
                )
                .context("execute binary stage message")?;
                runtime_sessions_after = Some(runtime.session_stats());
                runtime_lock_hold_ms = elapsed_ms(lock_hold_started);
                result
            };
            let compute_ms = elapsed_ms(compute_started);
            compute_end_unix_nanos = now_unix_nanos() as u64;
            (result.0, result.1, result.2, compute_ms)
        };
        let mut decode_attrs = binary_message_attrs(config, session_id, &message);
        decode_attrs.insert(
            "skippy.output_activation_bytes".to_string(),
            json!(output.payload.len()),
        );
        decode_attrs.insert("skippy.compute_ms".to_string(), json!(compute_ms));
        decode_attrs.insert(
            "llama_stage.input_activation_decode_ms".to_string(),
            json!(input_activation_decode_ms),
        );
        decode_attrs.insert(
            "llama_stage.runtime_lock_wait_ms".to_string(),
            json!(runtime_lock_wait_ms),
        );
        decode_attrs.insert(
            "llama_stage.runtime_lock_hold_ms".to_string(),
            json!(runtime_lock_hold_ms),
        );
        decode_attrs.insert(
            "llama_stage.runtime_lock_acquires".to_string(),
            json!(runtime_lock_acquires),
        );
        if let Some(stats) = runtime_sessions_before {
            insert_runtime_session_stats(
                &mut decode_attrs,
                "llama_stage.runtime_sessions_before",
                stats,
            );
        }
        if let Some(stats) = runtime_sessions_after {
            insert_runtime_session_stats(
                &mut decode_attrs,
                "llama_stage.runtime_sessions_after",
                stats,
            );
        }
        decode_attrs.insert(
            "skippy.kv.restored_prefill".to_string(),
            json!(restored_prefill),
        );
        decode_attrs.insert(
            "llama_stage.compute_start_unix_nanos".to_string(),
            json!(compute_start_unix_nanos),
        );
        decode_attrs.insert(
            "llama_stage.compute_end_unix_nanos".to_string(),
            json!(compute_end_unix_nanos),
        );
        telemetry.emit_debug_span(
            "stage.binary_llama_decode",
            decode_attrs,
            compute_start_unix_nanos,
            compute_end_unix_nanos,
        );

        if message.kind.is_prefill() && !restored_prefill {
            let records = maybe_plan_record_binary_prefill(
                config,
                kv,
                telemetry,
                &session_key,
                &message,
                &token_ids,
                lookup_result.restored_tokens as u64,
            );
            if !records.is_empty() {
                message_reply_stats.kv_recorded_pages += records.len() as i64;
                message_reply_stats.kv_record_stage_mask |= stage_mask(config.stage_index);
                record_workers.push(spawn_commit_record_binary_prefill(
                    runtime.clone(),
                    kv.cloned(),
                    telemetry.clone(),
                    records,
                ));
            }
        }

        let mut forward_write_ms = 0.0;
        let mut forward_activation_encode_ms = 0.0;
        let mut downstream_wait_ms = 0.0;
        let mut upstream_reply_ms = early_reply_ms;
        let mut forward_write_start_unix_nanos = None;
        let mut forward_write_end_unix_nanos = None;
        let mut downstream_wait_start_unix_nanos = None;
        let mut downstream_wait_end_unix_nanos = None;
        let mut forward_mode = "none";
        let pending_prefill_replies_before = pending_prefill_replies;
        let mut credit_wait_count = 0usize;
        let mut deferred_prefill_replies_drained = 0usize;

        if let Some(downstream) = downstream.as_mut() {
            if output.payload.is_empty() {
                bail!("stage has downstream but produced an empty activation payload");
            }
            let forwarded = forwarded_stage_message_timed(
                config,
                &message,
                &output,
                wire_dtype,
                activation_width,
            )?;
            forward_activation_encode_ms += forwarded.activation_encode_ms;
            let mut downstream_write_attrs = binary_message_attrs(config, session_id, &message);
            downstream_write_attrs.insert(
                "llama_stage.forward_activation_bytes".to_string(),
                json!(forwarded.message.activation.len()),
            );
            downstream_write_attrs.insert(
                "llama_stage.activation_encode_ms".to_string(),
                json!(forwarded.activation_encode_ms),
            );
            downstream_write_attrs.insert(
                "llama_stage.output_activation_bytes".to_string(),
                json!(output.payload.len()),
            );
            let forward_start_unix_nanos = now_unix_nanos() as u64;
            forward_write_start_unix_nanos = Some(forward_start_unix_nanos);
            let forward_started = Instant::now();
            if async_prefill_forward && early_prefill_ack && max_deferred_prefill_replies > 0 {
                forward_mode = "async_enqueue";
                downstream_write_attrs.insert(
                    "llama_stage.forward_mode".to_string(),
                    json!("async_writer"),
                );
                let forwarder = async_forwarder
                    .as_mut()
                    .context("missing async activation forwarder")?;
                forwarder
                    .send(
                        forwarded.message,
                        wire_dtype,
                        downstream_wire_condition,
                        downstream_write_attrs,
                    )
                    .context("queue async activation frame downstream")?;
            } else {
                forward_mode = "sync_write";
                downstream_write_attrs
                    .insert("llama_stage.forward_mode".to_string(), json!("sync_write"));
                if let Some(forwarder) = async_forwarder.as_mut() {
                    forwarder.flush().context("flush async activation frames")?;
                }
                let downstream_write_start_unix_nanos = now_unix_nanos() as u64;
                let downstream_write_started = Instant::now();
                write_stage_message_conditioned(
                    &mut *downstream,
                    &forwarded.message,
                    wire_dtype,
                    downstream_wire_condition,
                )
                .context("forward activation frame downstream")?;
                let downstream_write_end_unix_nanos = now_unix_nanos() as u64;
                downstream_write_attrs.insert(
                    "llama_stage.forward_write_ms".to_string(),
                    json!(elapsed_ms(downstream_write_started)),
                );
                telemetry.emit_debug_span(
                    "stage.binary_downstream_write",
                    downstream_write_attrs,
                    downstream_write_start_unix_nanos,
                    downstream_write_end_unix_nanos,
                );
            }
            forward_write_end_unix_nanos = Some(now_unix_nanos() as u64);
            forward_write_ms += elapsed_ms(forward_started);

            if requires_predicted {
                while pending_prefill_replies > 0 {
                    let wait_start_unix_nanos = now_unix_nanos() as u64;
                    downstream_wait_start_unix_nanos.get_or_insert(wait_start_unix_nanos);
                    let wait_started = Instant::now();
                    let reply = recv_reply(&mut *downstream)
                        .context("drain deferred downstream prefill reply")?;
                    downstream_wait_ms += elapsed_ms(wait_started);
                    if reply.kind != WireReplyKind::Ack {
                        bail!("expected deferred downstream ACK");
                    }
                    pending_reply_stats.merge(reply.stats);
                    pending_prefill_replies -= 1;
                    deferred_prefill_replies_drained += 1;
                }
                let wait_start_unix_nanos = now_unix_nanos() as u64;
                downstream_wait_start_unix_nanos.get_or_insert(wait_start_unix_nanos);
                let wait_started = Instant::now();
                let reply = recv_reply(&mut *downstream).context("downstream predicted reply")?;
                downstream_wait_end_unix_nanos = Some(now_unix_nanos() as u64);
                downstream_wait_ms += elapsed_ms(wait_started);
                if message.kind == WireMessageKind::VerifySpan {
                    if reply.kind != WireReplyKind::PredictedTokens {
                        bail!("expected downstream predicted-tokens reply");
                    }
                } else if reply.kind != WireReplyKind::PredictedToken {
                    bail!("expected downstream predicted-token reply");
                }
                message_reply_stats.merge(reply.stats);
                message_reply_stats.merge(pending_reply_stats);
                pending_reply_stats = StageReplyStats::default();
                record_verify_span_timing(
                    &mut message_reply_stats,
                    &message,
                    compute_ms,
                    forward_write_ms,
                    downstream_wait_ms,
                );
                let reply_start_unix_nanos = now_unix_nanos() as u64;
                upstream_reply_start_unix_nanos.get_or_insert(reply_start_unix_nanos);
                let reply_started = Instant::now();
                if message.kind == WireMessageKind::VerifySpan {
                    send_reply_predicted_tokens_with_stats(
                        &mut *upstream,
                        &reply.predicted_tokens,
                        message_reply_stats,
                    )
                    .context("relay predicted-tokens reply")?;
                } else {
                    send_reply_predicted_with_stats(
                        &mut *upstream,
                        reply.predicted,
                        message_reply_stats,
                    )
                    .context("relay predicted-token reply")?;
                }
                upstream_reply_end_unix_nanos = Some(now_unix_nanos() as u64);
                upstream_reply_ms += elapsed_ms(reply_started);
            } else if max_deferred_prefill_replies == 0 {
                let wait_start_unix_nanos = now_unix_nanos() as u64;
                downstream_wait_start_unix_nanos.get_or_insert(wait_start_unix_nanos);
                let wait_started = Instant::now();
                let reply = recv_reply(&mut *downstream).context("downstream ACK")?;
                downstream_wait_end_unix_nanos = Some(now_unix_nanos() as u64);
                downstream_wait_ms += elapsed_ms(wait_started);
                if reply.kind != WireReplyKind::Ack {
                    bail!("expected downstream ACK");
                }
                message_reply_stats.merge(reply.stats);
                if !early_prefill_ack {
                    let reply_start_unix_nanos = now_unix_nanos() as u64;
                    upstream_reply_start_unix_nanos.get_or_insert(reply_start_unix_nanos);
                    let reply_started = Instant::now();
                    send_reply_ack_with_stats(&mut *upstream, message_reply_stats)
                        .context("relay ACK")?;
                    upstream_reply_end_unix_nanos = Some(now_unix_nanos() as u64);
                    upstream_reply_ms += elapsed_ms(reply_started);
                } else {
                    pending_reply_stats.merge(message_reply_stats);
                }
            } else {
                while pending_prefill_replies >= max_deferred_prefill_replies {
                    credit_wait_count += 1;
                    let wait_start_unix_nanos = now_unix_nanos() as u64;
                    downstream_wait_start_unix_nanos.get_or_insert(wait_start_unix_nanos);
                    let wait_started = Instant::now();
                    let reply =
                        recv_reply(&mut *downstream).context("bounded-credit downstream ACK")?;
                    downstream_wait_end_unix_nanos = Some(now_unix_nanos() as u64);
                    downstream_wait_ms += elapsed_ms(wait_started);
                    if reply.kind != WireReplyKind::Ack {
                        bail!("expected downstream ACK while enforcing credit");
                    }
                    pending_reply_stats.merge(reply.stats);
                    pending_prefill_replies -= 1;
                    deferred_prefill_replies_drained += 1;
                }
                pending_prefill_replies += 1;
                if !early_prefill_ack {
                    let reply_start_unix_nanos = now_unix_nanos() as u64;
                    upstream_reply_start_unix_nanos.get_or_insert(reply_start_unix_nanos);
                    let reply_started = Instant::now();
                    send_reply_ack_with_stats(&mut *upstream, message_reply_stats)
                        .context("deferred relay ACK")?;
                    upstream_reply_end_unix_nanos = Some(now_unix_nanos() as u64);
                    upstream_reply_ms += elapsed_ms(reply_started);
                } else {
                    pending_reply_stats.merge(message_reply_stats);
                }
            }
        } else if requires_predicted {
            message_reply_stats.merge(pending_reply_stats);
            pending_reply_stats = StageReplyStats::default();
            record_verify_span_timing(
                &mut message_reply_stats,
                &message,
                compute_ms,
                forward_write_ms,
                downstream_wait_ms,
            );
            let reply_start_unix_nanos = now_unix_nanos() as u64;
            upstream_reply_start_unix_nanos.get_or_insert(reply_start_unix_nanos);
            let reply_started = Instant::now();
            if message.kind == WireMessageKind::VerifySpan {
                send_reply_predicted_tokens_with_stats(
                    &mut *upstream,
                    &predicted_tokens,
                    message_reply_stats,
                )
                .context("send predicted tokens")?;
            } else {
                send_reply_predicted_with_stats(
                    &mut *upstream,
                    predicted_token,
                    message_reply_stats,
                )
                .context("send predicted token")?;
            }
            upstream_reply_end_unix_nanos = Some(now_unix_nanos() as u64);
            upstream_reply_ms += elapsed_ms(reply_started);
        } else if !early_prefill_ack {
            let reply_start_unix_nanos = now_unix_nanos() as u64;
            upstream_reply_start_unix_nanos.get_or_insert(reply_start_unix_nanos);
            let reply_started = Instant::now();
            send_reply_ack_with_stats(&mut *upstream, message_reply_stats).context("send ACK")?;
            upstream_reply_end_unix_nanos = Some(now_unix_nanos() as u64);
            upstream_reply_ms += elapsed_ms(reply_started);
        } else {
            pending_reply_stats.merge(message_reply_stats);
        }

        let message_end_unix_nanos = now_unix_nanos() as u64;
        let message_elapsed_ms = elapsed_ms(message_started);
        request_summary.observe(BinaryMessageObservation {
            config,
            message: &message,
            compute_ms,
            forward_write_ms,
            downstream_wait_ms,
            upstream_reply_ms,
            message_elapsed_ms,
            input_activation_bytes: message.activation.len(),
            output_activation_bytes: output.payload.len(),
            input_activation_decode_ms,
            forward_activation_encode_ms,
            runtime_lock_hold_ms,
            prefill_credit_limit: max_deferred_prefill_replies,
            pending_prefill_replies_before,
            pending_prefill_replies_after: pending_prefill_replies,
            credit_wait_count,
            deferred_prefill_replies_drained,
        });

        let mut timing_attrs = binary_message_attrs(config, session_id, &message);
        timing_attrs.insert(
            "llama_stage.message_start_unix_nanos".to_string(),
            json!(message_start_unix_nanos),
        );
        timing_attrs.insert(
            "llama_stage.message_end_unix_nanos".to_string(),
            json!(message_end_unix_nanos),
        );
        timing_attrs.insert(
            "llama_stage.compute_start_unix_nanos".to_string(),
            json!(compute_start_unix_nanos),
        );
        timing_attrs.insert(
            "llama_stage.compute_end_unix_nanos".to_string(),
            json!(compute_end_unix_nanos),
        );
        timing_attrs.insert("llama_stage.compute_ms".to_string(), json!(compute_ms));
        timing_attrs.insert(
            "llama_stage.input_activation_decode_ms".to_string(),
            json!(input_activation_decode_ms),
        );
        timing_attrs.insert(
            "llama_stage.runtime_lock_wait_ms".to_string(),
            json!(runtime_lock_wait_ms),
        );
        timing_attrs.insert(
            "llama_stage.runtime_lock_hold_ms".to_string(),
            json!(runtime_lock_hold_ms),
        );
        timing_attrs.insert(
            "llama_stage.runtime_lock_acquires".to_string(),
            json!(runtime_lock_acquires),
        );
        if let Some(stats) = runtime_sessions_before {
            insert_runtime_session_stats(
                &mut timing_attrs,
                "llama_stage.runtime_sessions_before",
                stats,
            );
        }
        if let Some(stats) = runtime_sessions_after {
            insert_runtime_session_stats(
                &mut timing_attrs,
                "llama_stage.runtime_sessions_after",
                stats,
            );
        }
        timing_attrs.insert(
            "llama_stage.forward_write_ms".to_string(),
            json!(forward_write_ms),
        );
        timing_attrs.insert(
            "llama_stage.activation_encode_ms".to_string(),
            json!(forward_activation_encode_ms),
        );
        timing_attrs.insert(
            "llama_stage.downstream_wait_ms".to_string(),
            json!(downstream_wait_ms),
        );
        timing_attrs.insert("skippy.compute_ms".to_string(), json!(compute_ms));
        timing_attrs.insert(
            "skippy.forward_write_ms".to_string(),
            json!(forward_write_ms),
        );
        timing_attrs.insert(
            "skippy.downstream_wait_ms".to_string(),
            json!(downstream_wait_ms),
        );
        timing_attrs.insert(
            "skippy.upstream_reply_ms".to_string(),
            json!(upstream_reply_ms),
        );
        timing_attrs.insert("llama_stage.forward_mode".to_string(), json!(forward_mode));
        insert_optional_unix_nanos(
            &mut timing_attrs,
            "llama_stage.forward_write_start_unix_nanos",
            forward_write_start_unix_nanos,
        );
        insert_optional_unix_nanos(
            &mut timing_attrs,
            "llama_stage.forward_write_end_unix_nanos",
            forward_write_end_unix_nanos,
        );
        insert_optional_unix_nanos(
            &mut timing_attrs,
            "llama_stage.downstream_wait_start_unix_nanos",
            downstream_wait_start_unix_nanos,
        );
        insert_optional_unix_nanos(
            &mut timing_attrs,
            "llama_stage.downstream_wait_end_unix_nanos",
            downstream_wait_end_unix_nanos,
        );
        insert_optional_unix_nanos(
            &mut timing_attrs,
            "llama_stage.upstream_reply_start_unix_nanos",
            upstream_reply_start_unix_nanos,
        );
        insert_optional_unix_nanos(
            &mut timing_attrs,
            "llama_stage.upstream_reply_end_unix_nanos",
            upstream_reply_end_unix_nanos,
        );
        timing_attrs.insert(
            "skippy.message_elapsed_ms".to_string(),
            json!(message_elapsed_ms),
        );
        timing_attrs.insert(
            "skippy.input_activation_bytes".to_string(),
            json!(message.activation.len()),
        );
        timing_attrs.insert(
            "skippy.output_activation_bytes".to_string(),
            json!(output.payload.len()),
        );
        timing_attrs.insert(
            "skippy.prefill_credit_limit".to_string(),
            json!(max_deferred_prefill_replies),
        );
        timing_attrs.insert(
            "skippy.prefill_pending_replies_before".to_string(),
            json!(pending_prefill_replies_before),
        );
        timing_attrs.insert(
            "skippy.prefill_pending_replies_after".to_string(),
            json!(pending_prefill_replies),
        );
        timing_attrs.insert(
            "skippy.prefill_credit_wait_count".to_string(),
            json!(credit_wait_count),
        );
        timing_attrs.insert(
            "skippy.prefill_deferred_replies_drained".to_string(),
            json!(deferred_prefill_replies_drained),
        );
        telemetry.emit_debug_span(
            "stage.binary_message_timing",
            timing_attrs,
            message_start_unix_nanos,
            message_end_unix_nanos,
        );
    }
}

fn insert_optional_unix_nanos(attrs: &mut BTreeMap<String, Value>, key: &str, value: Option<u64>) {
    if let Some(value) = value {
        attrs.insert(key.to_string(), json!(value));
    }
}

fn insert_runtime_session_stats(
    attrs: &mut BTreeMap<String, Value>,
    prefix: &str,
    stats: RuntimeSessionStats,
) {
    attrs.insert(
        format!("{prefix}.active_sessions"),
        json!(stats.active_sessions),
    );
    attrs.insert(
        format!("{prefix}.idle_sessions"),
        json!(stats.idle_sessions),
    );
    attrs.insert(
        format!("{prefix}.warm_kv_sessions"),
        json!(stats.warm_kv_sessions),
    );
    attrs.insert(
        format!("{prefix}.tracked_token_counts"),
        json!(stats.tracked_token_counts),
    );
    attrs.insert(format!("{prefix}.checkpoints"), json!(stats.checkpoints));
}

fn binary_message_attrs(
    config: &StageConfig,
    session_id: u64,
    message: &StageWireMessage,
) -> std::collections::BTreeMap<String, serde_json::Value> {
    let mut attrs = lifecycle_attrs(config);
    attrs.insert(attr::SESSION_ID.to_string(), json!(session_id.to_string()));
    attrs.insert(
        attr::REQUEST_ID.to_string(),
        json!(binary_message_request_id(message)),
    );
    attrs.insert(
        "skippy.prompt_index".to_string(),
        json!(message.state.seq_id),
    );
    attrs.insert(
        "skippy.message_kind".to_string(),
        json!(format!("{:?}", message.kind)),
    );
    attrs.insert("skippy.token_count".to_string(), json!(message.token_count));
    attrs.insert(
        "skippy.prompt_token_count".to_string(),
        json!(message.state.prompt_token_count),
    );
    attrs.insert(
        "skippy.decode_step".to_string(),
        json!(message.state.decode_step),
    );
    let layer_count = i64::from(config.layer_end.saturating_sub(config.layer_start));
    let kv_tokens_after = estimated_kv_tokens_after(message);
    attrs.insert("skippy.kv_tokens_after".to_string(), json!(kv_tokens_after));
    attrs.insert("skippy.kv_layer_count".to_string(), json!(layer_count));
    attrs.insert(
        "skippy.kv_token_layer_cells".to_string(),
        json!(kv_tokens_after.saturating_mul(layer_count)),
    );
    attrs
}

fn estimated_kv_tokens_after(message: &StageWireMessage) -> i64 {
    if message.kind == WireMessageKind::Stop {
        return 0;
    }
    let pos_start = i64::from(message.pos_start.max(0));
    let token_count = i64::from(message.token_count.max(0));
    pos_start.saturating_add(token_count)
}

fn drain_deferred_prefill_replies(
    downstream: Option<&mut TcpStream>,
    pending_prefill_replies: &mut usize,
    pending_reply_stats: &mut StageReplyStats,
) -> Result<()> {
    let Some(downstream) = downstream else {
        return Ok(());
    };
    while *pending_prefill_replies > 0 {
        let reply =
            recv_reply(&mut *downstream).context("drain deferred downstream prefill ACK")?;
        if reply.kind != WireReplyKind::Ack {
            bail!("expected deferred downstream ACK");
        }
        pending_reply_stats.merge(reply.stats);
        *pending_prefill_replies -= 1;
    }
    Ok(())
}

fn record_session_control_timing(
    stats: &mut StageReplyStats,
    kind: WireMessageKind,
    timing: SessionControlTiming,
) {
    match kind {
        WireMessageKind::CheckpointSession => {
            stats.checkpoint_flush_us += timing.flush_us;
            stats.checkpoint_prefill_drain_us += timing.prefill_drain_us;
            stats.checkpoint_local_us += timing.local_us;
            stats.checkpoint_downstream_write_us += timing.downstream_write_us;
            stats.checkpoint_downstream_wait_us += timing.downstream_wait_us;
            stats.checkpoint_total_us += timing.total_us;
            stats.checkpoint_prefill_drained_replies += timing.prefill_drained_replies;
        }
        WireMessageKind::RestoreSession => {
            stats.restore_flush_us += timing.flush_us;
            stats.restore_prefill_drain_us += timing.prefill_drain_us;
            stats.restore_local_us += timing.local_us;
            stats.restore_downstream_write_us += timing.downstream_write_us;
            stats.restore_downstream_wait_us += timing.downstream_wait_us;
            stats.restore_total_us += timing.total_us;
            stats.restore_prefill_drained_replies += timing.prefill_drained_replies;
        }
        _ => {}
    }
}

fn record_verify_span_timing(
    stats: &mut StageReplyStats,
    message: &StageWireMessage,
    compute_ms: f64,
    forward_write_ms: f64,
    downstream_wait_ms: f64,
) {
    if message.kind != WireMessageKind::VerifySpan {
        return;
    }
    let compute_us = ms_to_us(compute_ms);
    let forward_write_us = ms_to_us(forward_write_ms);
    let downstream_wait_us = ms_to_us(downstream_wait_ms);
    let token_count = i64::from(message.token_count.max(0));
    stats.verify_span_compute_us += compute_us;
    stats.verify_span_forward_write_us += forward_write_us;
    stats.verify_span_downstream_wait_us += downstream_wait_us;
    stats.verify_span_total_us += compute_us + forward_write_us + downstream_wait_us;
    stats.verify_span_stage_count += 1;
    stats.verify_span_request_count += 1;
    stats.verify_span_token_count += token_count;
    stats.verify_span_max_tokens = stats.verify_span_max_tokens.max(token_count);
    if (message.state.flags & state_flags::SKIP_VERIFY_CHECKPOINT) == 0 {
        stats.verify_span_checkpointed_requests += 1;
    } else {
        stats.verify_span_skip_checkpoint_requests += 1;
    }
}

fn elapsed_ms(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1000.0
}

fn elapsed_us(started: Instant) -> i64 {
    let micros = started.elapsed().as_micros();
    micros.min(i64::MAX as u128) as i64
}

fn ms_to_us(ms: f64) -> i64 {
    if !ms.is_finite() || ms <= 0.0 {
        0
    } else {
        (ms * 1000.0).round().min(i64::MAX as f64) as i64
    }
}

fn stage_mask(stage_index: u32) -> i64 {
    if stage_index < 63 {
        1_i64 << stage_index
    } else {
        0
    }
}

fn block_on_kv<F: Future>(future: F) -> F::Output {
    tokio::task::block_in_place(|| tokio::runtime::Handle::current().block_on(future))
}

fn binary_message_base(
    config: &StageConfig,
    session_id: &str,
    message: &StageWireMessage,
) -> MessageBase {
    MessageBase {
        schema_version: SCHEMA_VERSION,
        run_id: config.run_id.clone(),
        request_id: binary_message_request_id(message),
        session_id: session_id.to_string(),
        stage_id: "binary-driver".to_string(),
        stage_index: 0,
        topology_id: config.topology_id.clone(),
        model_id: Some(config.model_id.clone()),
        tokenizer_id: None,
        chat_template_id: None,
        seq: Some(message.state.seq_id.max(0) as u64),
    }
}

fn binary_message_session_id(fallback: u64, message: &StageWireMessage) -> u64 {
    if message.session_id == 0 {
        fallback
    } else {
        message.session_id
    }
}

fn binary_message_request_id(message: &StageWireMessage) -> String {
    if message.request_id == 0 {
        format!("prompt-{}", message.state.seq_id)
    } else {
        message.request_id.to_string()
    }
}

fn binary_kv_attrs(config: &StageConfig, kv: &KvStageIntegration) -> BTreeMap<String, Value> {
    let mut attrs = lifecycle_attrs(config);
    for (key, value) in kv.attrs() {
        attrs.insert(key.to_string(), value);
    }
    attrs
}

fn binary_message_kv_attrs(
    config: &StageConfig,
    kv: &KvStageIntegration,
    session_id: &str,
    message: &StageWireMessage,
    token_count: usize,
) -> BTreeMap<String, Value> {
    let mut attrs = binary_kv_attrs(config, kv);
    attrs.insert(attr::SESSION_ID.to_string(), json!(session_id));
    attrs.insert(
        attr::REQUEST_ID.to_string(),
        json!(binary_message_request_id(message)),
    );
    attrs.insert(
        "skippy.message_kind".to_string(),
        json!(format!("{:?}", message.kind)),
    );
    attrs.insert(
        "skippy.kv.token_start".to_string(),
        json!(message.pos_start.max(0)),
    );
    attrs.insert("skippy.kv.token_count".to_string(), json!(token_count));
    attrs
}

fn maybe_lookup_binary_prefill(
    config: &StageConfig,
    runtime: &Arc<Mutex<RuntimeState>>,
    kv: Option<&Arc<KvStageIntegration>>,
    telemetry: &Telemetry,
    session_id: &str,
    message: &StageWireMessage,
    token_ids: &[i32],
) -> BinaryKvLookupResult {
    let mut result = BinaryKvLookupResult::default();
    let Some(kv) = kv else {
        return result;
    };
    if !message.kind.is_prefill()
        || message.kind.requires_predicted_reply()
        || !kv.should_lookup()
        || config.downstream.is_some()
        || token_ids.is_empty()
    {
        return result;
    }
    let token_start = message.pos_start.max(0) as u64;
    let base = binary_message_base(config, session_id, message);
    let identities = kv.lookup_identities(config, &base, token_start, token_ids);
    let mut attrs = binary_message_kv_attrs(config, kv, session_id, message, token_ids.len());
    attrs.insert(
        "skippy.kv.lookup_candidates".to_string(),
        json!(identities.len()),
    );
    let started = Instant::now();
    match block_on_kv(
        kv.lookup_prefixes(
            identities
                .into_iter()
                .map(|candidate| candidate.identity)
                .collect(),
        ),
    ) {
        Ok(outcome) => {
            attrs.insert(
                "skippy.kv.lookup_ms".to_string(),
                json!(elapsed_ms(started)),
            );
            attrs.insert(
                "skippy.kv.lookup_hits".to_string(),
                json!(outcome.pages.len()),
            );
            if !outcome.errors.is_empty() {
                attrs.insert("skippy.kv.lookup_errors".to_string(), json!(outcome.errors));
            }
            if let Some(page) = outcome
                .pages
                .into_iter()
                .max_by_key(|page| page.identity.as_ref().map(|identity| identity.token_count))
            {
                let restored_tokens = page
                    .identity
                    .as_ref()
                    .map(|identity| identity.token_count as usize)
                    .unwrap_or(0)
                    .min(token_ids.len());
                attrs.insert(
                    "skippy.kv.hit_page_id".to_string(),
                    json!(page.page_id.clone()),
                );
                attrs.insert(
                    "skippy.kv.restored_tokens".to_string(),
                    json!(restored_tokens),
                );
                if restored_tokens < token_ids.len() && config.layer_start != 0 {
                    result.restored_tokens = restored_tokens;
                    result.stats.kv_lookup_hits += 1;
                    attrs.insert(
                        "skippy.kv.decision".to_string(),
                        json!("partial_hit_deferred"),
                    );
                    telemetry.emit("stage.binary_kv_lookup_decision", attrs);
                    return result;
                }
                let already_loaded = {
                    let runtime = runtime.lock().expect("runtime lock poisoned");
                    runtime.has_session_range(session_id, token_start, restored_tokens as u64)
                };
                if already_loaded {
                    result.restored_tokens = restored_tokens;
                    result.stats.kv_lookup_hits += 1;
                    result.stats.kv_imported_tokens += restored_tokens as i64;
                    result.stats.kv_hit_stage_mask |= stage_mask(config.stage_index);
                    attrs.insert(
                        "skippy.kv.decision".to_string(),
                        json!("hit_already_loaded"),
                    );
                    telemetry.emit("stage.binary_kv_lookup_decision", attrs);
                    return result;
                }
                let warmed = {
                    let mut runtime = runtime.lock().expect("runtime lock poisoned");
                    runtime.take_warm_kv_session(
                        session_id,
                        &page.page_id,
                        token_start,
                        restored_tokens as u64,
                    )
                };
                if warmed {
                    result.restored_tokens = restored_tokens;
                    result.stats.kv_lookup_hits += 1;
                    result.stats.kv_imported_tokens += restored_tokens as i64;
                    result.stats.kv_hit_stage_mask |= stage_mask(config.stage_index);
                    attrs.insert("skippy.kv.decision".to_string(), json!("hit_warm_session"));
                    telemetry.emit("stage.binary_kv_lookup_decision", attrs);
                    return result;
                }
                let attach_started = Instant::now();
                match block_on_kv(kv.attach_page(&page.page_id)) {
                    Ok(attached) => {
                        attrs.insert(
                            "skippy.kv.attach_ms".to_string(),
                            json!(elapsed_ms(attach_started)),
                        );
                        attrs.insert(
                            "skippy.kv.hit_bytes".to_string(),
                            json!(attached.bytes().len()),
                        );
                        let import_started = Instant::now();
                        let import_result = {
                            let mut runtime = runtime.lock().expect("runtime lock poisoned");
                            runtime.import_kv_page(session_id, &attached.manifest, attached.bytes())
                        };
                        attrs.insert(
                            "skippy.kv.runtime_import_ms".to_string(),
                            json!(elapsed_ms(import_started)),
                        );
                        match import_result {
                            Ok(()) => {
                                result.restored_tokens = restored_tokens;
                                result.stats.kv_lookup_hits += 1;
                                result.stats.kv_imported_pages += 1;
                                result.stats.kv_imported_tokens += restored_tokens as i64;
                                result.stats.kv_hit_stage_mask |= stage_mask(config.stage_index);
                                attrs.insert(
                                    "skippy.kv.decision".to_string(),
                                    json!("hit_imported"),
                                );
                                telemetry.emit("stage.binary_kv_lookup_decision", attrs);
                                return result;
                            }
                            Err(error) => {
                                result.stats.kv_lookup_errors += 1;
                                attrs.insert(
                                    "skippy.kv.decision".to_string(),
                                    json!("hit_import_error"),
                                );
                                attrs.insert(
                                    "skippy.kv.error".to_string(),
                                    json!(error.to_string()),
                                );
                            }
                        }
                    }
                    Err(error) => {
                        result.stats.kv_lookup_errors += 1;
                        attrs.insert(
                            "skippy.kv.attach_ms".to_string(),
                            json!(elapsed_ms(attach_started)),
                        );
                        attrs.insert("skippy.kv.decision".to_string(), json!("hit_attach_error"));
                        attrs.insert("skippy.kv.error".to_string(), json!(error.to_string()));
                    }
                }
            } else {
                result.stats.kv_lookup_misses += 1;
                attrs.insert("skippy.kv.decision".to_string(), json!("miss"));
            }
            telemetry.emit("stage.binary_kv_lookup_decision", attrs);
            result
        }
        Err(error) => {
            result.stats.kv_lookup_errors += 1;
            attrs.insert(
                "skippy.kv.lookup_ms".to_string(),
                json!(elapsed_ms(started)),
            );
            attrs.insert("skippy.kv.decision".to_string(), json!("error"));
            attrs.insert("skippy.kv.error".to_string(), json!(error.to_string()));
            telemetry.emit("stage.binary_kv_lookup_decision", attrs);
            result
        }
    }
}

fn maybe_plan_record_binary_prefill(
    config: &StageConfig,
    kv: Option<&Arc<KvStageIntegration>>,
    telemetry: &Telemetry,
    session_id: &str,
    message: &StageWireMessage,
    token_ids: &[i32],
    min_record_tokens: u64,
) -> Vec<BinaryKvRecordCandidate> {
    let Some(kv) = kv else {
        return Vec::new();
    };
    if !kv.should_record() || token_ids.is_empty() {
        return Vec::new();
    }
    let token_start = message.pos_start.max(0) as u64;
    let base = binary_message_base(config, session_id, message);
    let identities = kv
        .record_identities(config, &base, token_start, token_ids)
        .into_iter()
        .filter(|identity| identity.identity.token_count > min_record_tokens)
        .collect::<Vec<_>>();
    let mut records = Vec::with_capacity(identities.len());
    for identity in identities {
        let token_count = identity.identity.token_count;
        let mut attrs =
            binary_message_kv_attrs(config, kv, session_id, message, token_count as usize);
        attrs.insert(
            "skippy.kv.page_id".to_string(),
            json!(identity.page_id.clone()),
        );
        if !kv.try_begin_record(&identity.page_id) {
            attrs.insert(
                "skippy.kv.decision".to_string(),
                json!("record_skipped_inflight"),
            );
            telemetry.emit("stage.binary_kv_record_commit", attrs);
            continue;
        }
        attrs.insert("skippy.kv.decision".to_string(), json!("record_queued"));
        records.push(BinaryKvRecordCandidate {
            attrs,
            session_id: session_id.to_string(),
            stage_index: config.stage_index,
            page_id: identity.page_id,
            identity: identity.identity,
            token_start,
            token_count,
        });
    }
    records
}

fn spawn_commit_record_binary_prefill(
    runtime: Arc<Mutex<RuntimeState>>,
    kv: Option<Arc<KvStageIntegration>>,
    telemetry: Telemetry,
    records: Vec<BinaryKvRecordCandidate>,
) -> thread::JoinHandle<()> {
    let Some(kv) = kv else {
        return thread::spawn(|| {});
    };
    thread::spawn(move || {
        thread::sleep(Duration::from_millis(2));
        let Ok(tokio) = tokio::runtime::Builder::new_current_thread()
            .enable_io()
            .enable_time()
            .build()
        else {
            return;
        };
        tokio.block_on(async move {
            for record in records {
                commit_binary_record(&runtime, &kv, &telemetry, record).await;
            }
        });
    })
}

fn drain_record_workers(workers: &mut Vec<thread::JoinHandle<()>>) {
    for worker in workers.drain(..) {
        let _ = worker.join();
    }
}

async fn commit_binary_record(
    runtime: &Arc<Mutex<RuntimeState>>,
    kv: &KvStageIntegration,
    telemetry: &Telemetry,
    record: BinaryKvRecordCandidate,
) {
    let BinaryKvRecordCandidate {
        mut attrs,
        session_id,
        page_id,
        stage_index,
        identity,
        token_start,
        token_count,
    } = record;
    let page_id_for_finish = page_id.clone();
    let export_started = Instant::now();
    let page = match export_binary_kv_record_payload(runtime, &session_id, token_start, token_count)
    {
        Ok(page) => page,
        Err(error) => {
            let error = error.to_string();
            let decision = if error.contains("not implemented") {
                "unsupported"
            } else if error.contains("busy") {
                "record_skipped_runtime_busy"
            } else {
                "record_error"
            };
            attrs.insert(
                "skippy.kv.runtime_export_ms".to_string(),
                json!(elapsed_ms(export_started)),
            );
            attrs.insert("skippy.kv.decision".to_string(), json!(decision));
            attrs.insert("skippy.kv.error".to_string(), json!(error));
            telemetry.emit("stage.binary_kv_runtime_export", attrs);
            kv.finish_record(&page_id_for_finish);
            return;
        }
    };
    attrs.insert(
        "skippy.kv.runtime_export_ms".to_string(),
        json!(elapsed_ms(export_started)),
    );
    attrs.insert(
        "skippy.kv.export_bytes".to_string(),
        json!(page.desc.payload_bytes),
    );
    attrs.insert(
        "skippy.kv.export_layers".to_string(),
        json!(page.desc.layer_count),
    );
    attrs.insert("skippy.kv.stage_index".to_string(), json!(stage_index));

    let annotations = kv_desc_annotations(&page.desc);
    let started = Instant::now();
    match kv
        .record_page_into(
            page_id.clone(),
            identity,
            page.payload.len(),
            annotations,
            |output| {
                output.copy_from_slice(&page.payload);
                Ok(())
            },
        )
        .await
    {
        Ok(outcome) => {
            attrs.insert(
                "skippy.kv.record_ms".to_string(),
                json!(elapsed_ms(started)),
            );
            attrs.insert(
                "skippy.kv.record_write_ms".to_string(),
                json!(outcome.write_ms),
            );
            attrs.insert(
                "skippy.kv.checksum_ms".to_string(),
                json!(outcome.checksum_ms),
            );
            attrs.insert(
                "skippy.kv.recorded_bytes".to_string(),
                json!(outcome.manifest.byte_size),
            );
            attrs.insert("skippy.kv.decision".to_string(), json!("committed"));
            telemetry.emit("stage.binary_kv_record_commit", attrs);
            maybe_warm_binary_kv_session(
                runtime,
                telemetry,
                page_id,
                &outcome.manifest,
                &page.payload,
            );
        }
        Err(error) => {
            attrs.insert(
                "skippy.kv.record_ms".to_string(),
                json!(elapsed_ms(started)),
            );
            attrs.insert("skippy.kv.decision".to_string(), json!("record_error"));
            attrs.insert("skippy.kv.error".to_string(), json!(error.to_string()));
            telemetry.emit("stage.binary_kv_record_commit", attrs);
        }
    }
    kv.finish_record(&page_id_for_finish);
}

fn export_binary_kv_record_payload(
    runtime: &Arc<Mutex<RuntimeState>>,
    session_id: &str,
    token_start: u64,
    token_count: u64,
) -> Result<skippy_runtime::RuntimeKvPage> {
    let mut runtime = runtime
        .lock()
        .map_err(|_| anyhow!("runtime lock poisoned"))?;
    runtime.export_kv_page(session_id, token_start, token_count)
}

fn maybe_warm_binary_kv_session(
    runtime: &Arc<Mutex<RuntimeState>>,
    telemetry: &Telemetry,
    page_id: String,
    manifest: &crate::kv_proto::KvPageManifest,
    payload: &[u8],
) {
    let mut attrs = BTreeMap::new();
    attrs.insert("skippy.kv.page_id".to_string(), json!(page_id.clone()));
    let started = Instant::now();
    for attempt in 0..2 {
        match runtime.try_lock() {
            Ok(mut runtime) => {
                match runtime.warm_kv_session(page_id.clone(), manifest, payload) {
                    Ok(()) => {
                        attrs.insert("skippy.kv.warm_ms".to_string(), json!(elapsed_ms(started)));
                        attrs.insert("skippy.kv.decision".to_string(), json!("warmed"));
                        telemetry.emit("stage.binary_kv_warm_session", attrs);
                    }
                    Err(error) => {
                        attrs.insert("skippy.kv.warm_ms".to_string(), json!(elapsed_ms(started)));
                        attrs.insert("skippy.kv.decision".to_string(), json!("warm_error"));
                        attrs.insert("skippy.kv.error".to_string(), json!(error.to_string()));
                        telemetry.emit("stage.binary_kv_warm_session", attrs);
                    }
                }
                return;
            }
            Err(std::sync::TryLockError::WouldBlock) => {
                if attempt < 1 {
                    thread::sleep(Duration::from_millis(2));
                }
            }
            Err(std::sync::TryLockError::Poisoned(_)) => {
                attrs.insert("skippy.kv.decision".to_string(), json!("warm_error"));
                attrs.insert(
                    "skippy.kv.error".to_string(),
                    json!("runtime lock poisoned"),
                );
                telemetry.emit("stage.binary_kv_warm_session", attrs);
                return;
            }
        }
    }
    attrs.insert("skippy.kv.warm_ms".to_string(), json!(elapsed_ms(started)));
    attrs.insert(
        "skippy.kv.decision".to_string(),
        json!("warm_skipped_runtime_busy"),
    );
    telemetry.emit("stage.binary_kv_warm_session", attrs);
}

#[derive(Default)]
struct BinaryRequestSummary {
    request_id: Option<String>,
    prompt_index: i32,
    prompt_token_count: i32,
    message_count: usize,
    prefill_message_count: usize,
    decode_message_count: usize,
    prefill_token_count: i64,
    decode_token_count: i64,
    compute_ms: f64,
    forward_write_ms: f64,
    downstream_wait_ms: f64,
    upstream_reply_ms: f64,
    message_elapsed_ms: f64,
    input_activation_decode_ms: f64,
    forward_activation_encode_ms: f64,
    runtime_lock_hold_ms: f64,
    input_activation_bytes: usize,
    output_activation_bytes: usize,
    max_input_activation_bytes: usize,
    max_output_activation_bytes: usize,
    kv_tokens_after_max: i64,
    kv_token_layer_cells_max: i64,
    prefill_credit_limit: usize,
    prefill_credit_wait_count: usize,
    prefill_deferred_replies_drained: usize,
    prefill_pending_replies_max: usize,
}

struct BinaryMessageObservation<'a> {
    config: &'a StageConfig,
    message: &'a StageWireMessage,
    compute_ms: f64,
    forward_write_ms: f64,
    downstream_wait_ms: f64,
    upstream_reply_ms: f64,
    message_elapsed_ms: f64,
    input_activation_decode_ms: f64,
    forward_activation_encode_ms: f64,
    runtime_lock_hold_ms: f64,
    input_activation_bytes: usize,
    output_activation_bytes: usize,
    prefill_credit_limit: usize,
    pending_prefill_replies_before: usize,
    pending_prefill_replies_after: usize,
    credit_wait_count: usize,
    deferred_prefill_replies_drained: usize,
}

#[derive(Clone, Copy)]
struct SessionControlTiming {
    flush_us: i64,
    prefill_drain_us: i64,
    local_us: i64,
    downstream_write_us: i64,
    downstream_wait_us: i64,
    total_us: i64,
    prefill_drained_replies: i64,
}

#[derive(Clone, Copy, Debug)]
pub struct WireCondition {
    delay_ms: f64,
    mbps: Option<f64>,
}

impl WireCondition {
    pub fn new(delay_ms: f64, mbps: Option<f64>) -> Result<Self> {
        if delay_ms < 0.0 {
            bail!("downstream wire delay must not be negative");
        }
        if mbps.is_some_and(|value| value <= 0.0) {
            bail!("downstream wire mbps must be greater than zero");
        }
        Ok(Self { delay_ms, mbps })
    }

    fn sleep_for(&self, message: &StageWireMessage) {
        let delay_seconds = self.delay_ms / 1000.0;
        let bandwidth_seconds = self
            .mbps
            .map(|mbps| conditioned_wire_bytes(message) as f64 / (mbps * 125_000.0))
            .unwrap_or(0.0);
        let seconds = delay_seconds + bandwidth_seconds;
        if seconds > 0.0 {
            thread::sleep(Duration::from_secs_f64(seconds));
        }
    }
}

fn conditioned_wire_bytes(message: &StageWireMessage) -> usize {
    let header_bytes: usize = 4 * 4 + 9 * 4;
    let token_bytes = message
        .tokens
        .len()
        .saturating_mul(std::mem::size_of::<i32>());
    header_bytes
        .saturating_add(token_bytes)
        .saturating_add(message.activation.len())
        .saturating_add(message.raw_bytes.len())
}

pub(crate) fn write_stage_message_conditioned(
    writer: impl io::Write,
    message: &StageWireMessage,
    dtype: WireActivationDType,
    condition: WireCondition,
) -> io::Result<()> {
    condition.sleep_for(message);
    write_stage_message(writer, message, dtype)
}

struct AsyncForwarder {
    sender: mpsc::SyncSender<AsyncForwardJob>,
    pending: VecDeque<mpsc::Receiver<Result<()>>>,
}

struct AsyncForwardJob {
    message: StageWireMessage,
    wire_dtype: WireActivationDType,
    condition: WireCondition,
    attrs: BTreeMap<String, Value>,
    done: mpsc::Sender<Result<()>>,
}

impl AsyncForwarder {
    fn new(downstream: &TcpStream, telemetry: Telemetry) -> Result<Self> {
        let mut writer = downstream
            .try_clone()
            .context("clone downstream stream for async activation forwarding")?;
        let (sender, receiver) = mpsc::sync_channel::<AsyncForwardJob>(1);
        thread::spawn(move || {
            while let Ok(job) = receiver.recv() {
                let write_start_unix_nanos = now_unix_nanos() as u64;
                let write_started = Instant::now();
                let result = write_stage_message_conditioned(
                    &mut writer,
                    &job.message,
                    job.wire_dtype,
                    job.condition,
                )
                .context("async forward activation frame downstream");
                let write_end_unix_nanos = now_unix_nanos() as u64;
                let mut attrs = job.attrs;
                attrs.insert(
                    "llama_stage.forward_write_ms".to_string(),
                    json!(elapsed_ms(write_started)),
                );
                telemetry.emit_debug_span(
                    "stage.binary_downstream_write",
                    attrs,
                    write_start_unix_nanos,
                    write_end_unix_nanos,
                );
                let _ = job.done.send(result);
            }
        });
        Ok(Self {
            sender,
            pending: VecDeque::new(),
        })
    }

    fn send(
        &mut self,
        message: StageWireMessage,
        wire_dtype: WireActivationDType,
        condition: WireCondition,
        attrs: BTreeMap<String, Value>,
    ) -> Result<()> {
        let (done, receiver) = mpsc::channel();
        self.sender
            .send(AsyncForwardJob {
                message,
                wire_dtype,
                condition,
                attrs,
                done,
            })
            .map_err(|_| anyhow!("async activation forwarder stopped"))?;
        self.pending.push_back(receiver);
        Ok(())
    }

    fn flush(&mut self) -> Result<()> {
        while let Some(receiver) = self.pending.pop_front() {
            receiver
                .recv()
                .map_err(|_| anyhow!("async activation forwarder dropped result"))??;
        }
        Ok(())
    }
}

impl BinaryRequestSummary {
    fn observe(&mut self, observation: BinaryMessageObservation<'_>) {
        let message = observation.message;
        if self.message_count == 0 {
            self.request_id = Some(binary_message_request_id(message));
            self.prompt_index = message.state.seq_id;
            self.prompt_token_count = message.state.prompt_token_count;
        }

        self.message_count += 1;
        if message.kind.is_prefill() {
            self.prefill_message_count += 1;
            self.prefill_token_count += i64::from(message.token_count.max(0));
        } else if message.kind.requires_predicted_reply() {
            self.decode_message_count += 1;
            self.decode_token_count += i64::from(message.token_count.max(0));
        }

        self.compute_ms += observation.compute_ms;
        self.forward_write_ms += observation.forward_write_ms;
        self.downstream_wait_ms += observation.downstream_wait_ms;
        self.upstream_reply_ms += observation.upstream_reply_ms;
        self.message_elapsed_ms += observation.message_elapsed_ms;
        self.input_activation_decode_ms += observation.input_activation_decode_ms;
        self.forward_activation_encode_ms += observation.forward_activation_encode_ms;
        self.runtime_lock_hold_ms += observation.runtime_lock_hold_ms;
        self.input_activation_bytes += observation.input_activation_bytes;
        self.output_activation_bytes += observation.output_activation_bytes;
        self.max_input_activation_bytes = self
            .max_input_activation_bytes
            .max(observation.input_activation_bytes);
        self.max_output_activation_bytes = self
            .max_output_activation_bytes
            .max(observation.output_activation_bytes);

        let layer_count = i64::from(
            observation
                .config
                .layer_end
                .saturating_sub(observation.config.layer_start),
        );
        let kv_tokens_after = estimated_kv_tokens_after(message);
        self.kv_tokens_after_max = self.kv_tokens_after_max.max(kv_tokens_after);
        self.kv_token_layer_cells_max = self
            .kv_token_layer_cells_max
            .max(kv_tokens_after.saturating_mul(layer_count));
        self.prefill_credit_limit = observation.prefill_credit_limit;
        self.prefill_credit_wait_count += observation.credit_wait_count;
        self.prefill_deferred_replies_drained += observation.deferred_prefill_replies_drained;
        self.prefill_pending_replies_max = self
            .prefill_pending_replies_max
            .max(observation.pending_prefill_replies_before)
            .max(observation.pending_prefill_replies_after);
    }

    fn emit(&self, telemetry: &Telemetry, config: &StageConfig, session_id: u64) {
        if self.message_count == 0 || !telemetry.is_enabled() {
            return;
        }
        let mut attrs = lifecycle_attrs(config);
        attrs.insert(attr::SESSION_ID.to_string(), json!(session_id.to_string()));
        if let Some(request_id) = self.request_id.as_ref() {
            attrs.insert(attr::REQUEST_ID.to_string(), json!(request_id));
        }
        attrs.insert("skippy.prompt_index".to_string(), json!(self.prompt_index));
        attrs.insert(
            "skippy.prompt_token_count".to_string(),
            json!(self.prompt_token_count),
        );
        attrs.insert(
            "skippy.message_count".to_string(),
            json!(self.message_count),
        );
        attrs.insert(
            "skippy.prefill_message_count".to_string(),
            json!(self.prefill_message_count),
        );
        attrs.insert(
            "skippy.decode_message_count".to_string(),
            json!(self.decode_message_count),
        );
        attrs.insert(
            "skippy.prefill_token_count".to_string(),
            json!(self.prefill_token_count),
        );
        attrs.insert(
            "skippy.decode_token_count".to_string(),
            json!(self.decode_token_count),
        );
        attrs.insert("skippy.compute_ms".to_string(), json!(self.compute_ms));
        attrs.insert(
            "skippy.forward_write_ms".to_string(),
            json!(self.forward_write_ms),
        );
        attrs.insert(
            "skippy.downstream_wait_ms".to_string(),
            json!(self.downstream_wait_ms),
        );
        attrs.insert(
            "skippy.upstream_reply_ms".to_string(),
            json!(self.upstream_reply_ms),
        );
        attrs.insert(
            "skippy.message_elapsed_ms".to_string(),
            json!(self.message_elapsed_ms),
        );
        attrs.insert(
            "llama_stage.input_activation_decode_ms".to_string(),
            json!(self.input_activation_decode_ms),
        );
        attrs.insert(
            "llama_stage.activation_encode_ms".to_string(),
            json!(self.forward_activation_encode_ms),
        );
        attrs.insert(
            "llama_stage.runtime_lock_hold_ms".to_string(),
            json!(self.runtime_lock_hold_ms),
        );
        attrs.insert(
            "skippy.input_activation_bytes".to_string(),
            json!(self.input_activation_bytes),
        );
        attrs.insert(
            "skippy.output_activation_bytes".to_string(),
            json!(self.output_activation_bytes),
        );
        attrs.insert(
            "skippy.max_input_activation_bytes".to_string(),
            json!(self.max_input_activation_bytes),
        );
        attrs.insert(
            "skippy.max_output_activation_bytes".to_string(),
            json!(self.max_output_activation_bytes),
        );
        attrs.insert(
            "skippy.kv_tokens_after".to_string(),
            json!(self.kv_tokens_after_max),
        );
        attrs.insert(
            "skippy.kv_token_layer_cells".to_string(),
            json!(self.kv_token_layer_cells_max),
        );
        attrs.insert(
            "skippy.prefill_credit_limit".to_string(),
            json!(self.prefill_credit_limit),
        );
        attrs.insert(
            "skippy.prefill_credit_wait_count".to_string(),
            json!(self.prefill_credit_wait_count),
        );
        attrs.insert(
            "skippy.prefill_deferred_replies_drained".to_string(),
            json!(self.prefill_deferred_replies_drained),
        );
        attrs.insert(
            "skippy.prefill_pending_replies_max".to_string(),
            json!(self.prefill_pending_replies_max),
        );
        telemetry.emit("stage.binary_request_summary", attrs);
    }
}

pub(crate) fn connect_binary_downstream(
    config: &StageConfig,
    timeout_secs: u64,
) -> Result<Option<TcpStream>> {
    let Some(peer) = config.downstream.as_ref() else {
        return Ok(None);
    };
    let endpoint = peer
        .endpoint
        .strip_prefix("tcp://")
        .unwrap_or(&peer.endpoint);
    let downstream_addr = resolve_downstream_endpoint(endpoint)?;
    let source_ip = downstream_source_ip(config)?;
    let attempts = timeout_secs.saturating_mul(2).max(1);
    let mut last_error = None;
    for _ in 0..attempts {
        match connect_downstream_socket(downstream_addr, source_ip, Duration::from_secs(2)) {
            Ok(stream) => {
                stream.set_nodelay(true).ok();
                return Ok(Some(stream));
            }
            Err(error) => {
                last_error = Some(anyhow!(error));
                thread::sleep(Duration::from_millis(500));
            }
        }
    }
    Err(last_error
        .unwrap_or_else(|| anyhow!("timed out"))
        .context(format!(
            "connect downstream binary stage at {endpoint} ({downstream_addr})"
        )))
}

fn downstream_source_ip(config: &StageConfig) -> Result<Option<IpAddr>> {
    let bind_addr = config
        .bind_addr
        .parse::<SocketAddr>()
        .with_context(|| format!("parse stage bind_addr {}", config.bind_addr))?;
    let ip = bind_addr.ip();
    if ip.is_unspecified() {
        Ok(None)
    } else {
        Ok(Some(ip))
    }
}

fn resolve_downstream_endpoint(endpoint: &str) -> Result<SocketAddr> {
    endpoint
        .to_socket_addrs()
        .with_context(|| format!("resolve downstream binary stage endpoint {endpoint}"))?
        .find(SocketAddr::is_ipv4)
        .or_else(|| endpoint.to_socket_addrs().ok()?.next())
        .with_context(|| {
            format!("downstream binary stage endpoint resolved no addresses: {endpoint}")
        })
}

fn connect_downstream_socket(
    downstream_addr: SocketAddr,
    source_ip: Option<IpAddr>,
    timeout: Duration,
) -> io::Result<TcpStream> {
    let mut errors = Vec::new();

    macro_rules! try_connect {
        ($mode:literal, $connect:expr) => {
            match $connect {
                Ok(stream) => return Ok(stream),
                Err(error) => {
                    eprintln!(
                        "downstream connect retry: source={source_ip:?} remote={downstream_addr} mode={} error={error}",
                        $mode
                    );
                    errors.push(format!("{} failed: {error}", $mode));
                }
            }
        };
    }

    try_connect!(
        "route-selected",
        connect_route_selected_with_timeout(downstream_addr, source_ip, timeout)
    );
    try_connect!(
        "bound-interface",
        connect_bound_with_timeout(downstream_addr, source_ip, timeout, true)
    );
    try_connect!(
        "source-bound",
        connect_bound_with_timeout(downstream_addr, source_ip, timeout, false)
    );
    try_connect!(
        "blocking-source-bound",
        connect_blocking_with_timeout(downstream_addr, source_ip, timeout, false)
    );
    try_connect!(
        "blocking-route-selected",
        connect_route_selected_blocking_with_timeout(downstream_addr, source_ip, timeout)
    );

    Err(io::Error::new(io::ErrorKind::Other, errors.join("; ")))
}

fn connect_route_selected_with_timeout(
    downstream_addr: SocketAddr,
    source_ip: Option<IpAddr>,
    timeout: Duration,
) -> io::Result<TcpStream> {
    let stream = TcpStream::connect_timeout(&downstream_addr, timeout)?;
    if let Some(source_ip) = source_ip {
        let local_ip = stream.local_addr()?.ip();
        if local_ip != source_ip {
            return Err(io::Error::new(
                io::ErrorKind::AddrNotAvailable,
                format!("route-selected local address {local_ip} did not match {source_ip}"),
            ));
        }
    }
    eprintln!(
        "downstream connect succeeded: source={source_ip:?} remote={downstream_addr} mode=route-selected"
    );
    Ok(stream)
}

fn connect_route_selected_blocking_with_timeout(
    downstream_addr: SocketAddr,
    source_ip: Option<IpAddr>,
    timeout: Duration,
) -> io::Result<TcpStream> {
    let (tx, rx) = mpsc::sync_channel(1);
    thread::spawn(move || {
        let result = TcpStream::connect(downstream_addr)
            .and_then(|stream| validate_route_selected_stream(stream, source_ip));
        let _ = tx.send(result);
    });
    rx.recv_timeout(timeout).map_err(|_| {
        io::Error::new(
            io::ErrorKind::TimedOut,
            "blocking route-selected fallback connect timed out",
        )
    })?
}

fn validate_route_selected_stream(
    stream: TcpStream,
    source_ip: Option<IpAddr>,
) -> io::Result<TcpStream> {
    if let Some(source_ip) = source_ip {
        let local_ip = stream.local_addr()?.ip();
        if local_ip != source_ip {
            return Err(io::Error::new(
                io::ErrorKind::AddrNotAvailable,
                format!("route-selected local address {local_ip} did not match {source_ip}"),
            ));
        }
    }
    eprintln!(
        "downstream connect retry succeeded: source={source_ip:?} mode=blocking-route-selected"
    );
    Ok(stream)
}

fn connect_bound_with_timeout(
    downstream_addr: SocketAddr,
    source_ip: Option<IpAddr>,
    timeout: Duration,
    bind_interface: bool,
) -> io::Result<TcpStream> {
    let Some(source_ip) = source_ip else {
        return TcpStream::connect_timeout(&downstream_addr, timeout);
    };
    let domain = match (source_ip, downstream_addr) {
        (IpAddr::V4(_), SocketAddr::V4(_)) => Domain::IPV4,
        (IpAddr::V6(_), SocketAddr::V6(_)) => Domain::IPV6,
        _ => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("source address {source_ip} cannot connect to {downstream_addr}"),
            ));
        }
    };
    let socket = Socket::new(domain, Type::STREAM, Some(Protocol::TCP))?;
    socket.bind(&SockAddr::from(SocketAddr::new(source_ip, 0)))?;
    if bind_interface {
        bind_socket_to_source_interface(&socket, source_ip)?;
    }
    socket.connect_timeout(&SockAddr::from(downstream_addr), timeout)?;
    Ok(socket.into())
}

fn connect_blocking_with_timeout(
    downstream_addr: SocketAddr,
    source_ip: Option<IpAddr>,
    timeout: Duration,
    bind_interface: bool,
) -> io::Result<TcpStream> {
    let (tx, rx) = mpsc::sync_channel(1);
    thread::spawn(move || {
        let _ = tx.send(connect_bound_blocking(
            downstream_addr,
            source_ip,
            bind_interface,
        ));
    });
    rx.recv_timeout(timeout)
        .map_err(|_| io::Error::new(io::ErrorKind::TimedOut, "fallback connect timed out"))?
}

fn connect_bound_blocking(
    downstream_addr: SocketAddr,
    source_ip: Option<IpAddr>,
    bind_interface: bool,
) -> io::Result<TcpStream> {
    let Some(source_ip) = source_ip else {
        return TcpStream::connect(downstream_addr);
    };
    let domain = match (source_ip, downstream_addr) {
        (IpAddr::V4(_), SocketAddr::V4(_)) => Domain::IPV4,
        (IpAddr::V6(_), SocketAddr::V6(_)) => Domain::IPV6,
        _ => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("source address {source_ip} cannot connect to {downstream_addr}"),
            ));
        }
    };
    let socket = Socket::new(domain, Type::STREAM, Some(Protocol::TCP))?;
    socket.bind(&SockAddr::from(SocketAddr::new(source_ip, 0)))?;
    if bind_interface {
        bind_socket_to_source_interface(&socket, source_ip)?;
    }
    socket.connect(&SockAddr::from(downstream_addr))?;
    Ok(socket.into())
}

#[cfg(target_os = "macos")]
fn bind_socket_to_source_interface(socket: &Socket, source_ip: IpAddr) -> io::Result<()> {
    let Some(interface_index) = interface_index_for_ip(source_ip)? else {
        return Ok(());
    };
    let interface_index = interface_index as libc::c_int;
    let (level, optname) = match source_ip {
        IpAddr::V4(_) => (libc::IPPROTO_IP, 25),
        IpAddr::V6(_) => (libc::IPPROTO_IPV6, 125),
    };
    let result = unsafe {
        libc::setsockopt(
            socket.as_raw_fd(),
            level,
            optname,
            ptr::addr_of!(interface_index).cast(),
            std::mem::size_of_val(&interface_index) as libc::socklen_t,
        )
    };
    if result == -1 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn bind_socket_to_source_interface(_socket: &Socket, _source_ip: IpAddr) -> io::Result<()> {
    Ok(())
}

#[cfg(target_os = "macos")]
fn interface_index_for_ip(source_ip: IpAddr) -> io::Result<Option<u32>> {
    let mut addrs: *mut libc::ifaddrs = ptr::null_mut();
    let result = unsafe { libc::getifaddrs(ptr::addr_of_mut!(addrs)) };
    if result == -1 {
        return Err(io::Error::last_os_error());
    }
    let mut cursor = addrs;
    while !cursor.is_null() {
        let ifaddr = unsafe { &*cursor };
        if !ifaddr.ifa_addr.is_null() && sockaddr_ip(ifaddr.ifa_addr) == Some(source_ip) {
            let index = unsafe { libc::if_nametoindex(ifaddr.ifa_name) };
            unsafe { libc::freeifaddrs(addrs) };
            if index == 0 {
                return Err(io::Error::last_os_error());
            }
            return Ok(Some(index));
        }
        cursor = ifaddr.ifa_next;
    }
    unsafe { libc::freeifaddrs(addrs) };
    Ok(None)
}

#[cfg(target_os = "macos")]
fn sockaddr_ip(addr: *const libc::sockaddr) -> Option<IpAddr> {
    match unsafe { (*addr).sa_family as libc::c_int } {
        libc::AF_INET => {
            let addr = unsafe { &*(addr.cast::<libc::sockaddr_in>()) };
            Some(IpAddr::V4(Ipv4Addr::from(
                addr.sin_addr.s_addr.to_ne_bytes(),
            )))
        }
        libc::AF_INET6 => {
            let addr = unsafe { &*(addr.cast::<libc::sockaddr_in6>()) };
            Some(IpAddr::V6(addr.sin6_addr.s6_addr.into()))
        }
        _ => None,
    }
}

fn model_supports_kv_only_cache(config: &StageConfig) -> bool {
    let Some(path) = kv_cache_inspection_path(config) else {
        return true;
    };
    let Ok(info) = ModelInfo::open(path) else {
        return true;
    };
    let Ok(tensors) = info.tensors() else {
        return true;
    };
    !tensors
        .iter()
        .any(|tensor| tensor_name_requires_recurrent_state(&tensor.name))
}

fn kv_cache_inspection_path(config: &StageConfig) -> Option<PathBuf> {
    let path = config.model_path.as_deref()?;
    match config.load_mode {
        LoadMode::LayerPackage => crate::package::materialize_layer_package(config).ok(),
        LoadMode::RuntimeSlice | LoadMode::ArtifactSlice => Some(PathBuf::from(path)),
    }
}

fn tensor_name_requires_recurrent_state(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    lower.contains(".ssm")
        || lower.contains("ssm_")
        || lower.contains("time_mix")
        || lower.contains("recurrent")
        || lower.contains("rwkv")
}

pub(crate) fn run_binary_stage_message(
    runtime: &mut RuntimeState,
    session_id: &str,
    message: &StageWireMessage,
    token_ids: &[i32],
    input: Option<&ActivationFrame>,
) -> Result<(i32, Vec<i32>, ActivationFrame)> {
    match message.kind {
        WireMessageKind::PrefillEmbd | WireMessageKind::PrefillFinalEmbd => {
            let output = runtime.prefill_frame(session_id, token_ids, input)?;
            Ok((message.state.current_token, Vec::new(), output))
        }
        WireMessageKind::DecodeEmbd
        | WireMessageKind::DecodeReadout
        | WireMessageKind::DecodeLightCtx
        | WireMessageKind::DecodeReplayEmbd
        | WireMessageKind::DecodeReplayFinalEmbd => {
            let token_id = token_ids
                .first()
                .copied()
                .unwrap_or(message.state.current_token);
            let sampling = runtime_sampling_config(message.sampling.as_ref());
            let (predicted, output) =
                runtime.decode_frame_sampled(session_id, token_id, sampling.as_ref(), input)?;
            Ok((predicted, Vec::new(), output))
        }
        WireMessageKind::VerifySpan => {
            let (predicted_tokens, output) = runtime.verify_frame(session_id, token_ids, input)?;
            let predicted = predicted_tokens.first().copied().unwrap_or(0);
            Ok((predicted, predicted_tokens, output))
        }
        WireMessageKind::Stop
        | WireMessageKind::StateImport
        | WireMessageKind::StateExport
        | WireMessageKind::CheckpointSession
        | WireMessageKind::RestoreSession => {
            bail!("message kind is not executable")
        }
    }
}

fn runtime_sampling_config(sampling: Option<&StageSamplingConfig>) -> Option<SamplingConfig> {
    let sampling = sampling?;
    let mut config = SamplingConfig {
        enabled: true,
        seed: sampling.seed,
        temperature: sampling.temperature,
        top_p: sampling.top_p,
        top_k: sampling.top_k,
        presence_penalty: sampling.presence_penalty,
        frequency_penalty: sampling.frequency_penalty,
        repeat_penalty: sampling.repeat_penalty,
        penalty_last_n: sampling.penalty_last_n,
        ..SamplingConfig::default()
    };
    config.logit_bias = sampling
        .logit_bias
        .iter()
        .take(MAX_LOGIT_BIAS)
        .map(|source| LogitBias {
            token_id: source.token_id,
            bias: source.bias,
        })
        .collect();
    sampling.enabled().then_some(config)
}

fn input_activation_frame(
    config: &StageConfig,
    topology: Option<&StageTopology>,
    message: &StageWireMessage,
    activation_width: i32,
) -> Result<Option<ActivationFrame>> {
    if message.activation.is_empty() {
        return Ok(None);
    }
    let payload = message
        .activation_f32_payload(activation_width)
        .context("decode wire activation payload")?;
    let (layer_start, layer_end) = upstream_layer_range(config, topology, message);
    Ok(Some(ActivationFrame {
        desc: ActivationDesc {
            version: 1,
            dtype: RuntimeActivationDType::F32,
            layout: RuntimeActivationLayout::TokenMajor,
            producer_stage_index: message.state.source_stage_index,
            layer_start,
            layer_end,
            token_count: message.token_count.try_into().unwrap_or(0),
            sequence_count: if message.token_count > 0 { 1 } else { 0 },
            payload_bytes: payload.len() as u64,
            flags: 0,
        },
        payload,
    }))
}

fn empty_activation_frame(config: &StageConfig, message: &StageWireMessage) -> ActivationFrame {
    ActivationFrame {
        desc: ActivationDesc {
            version: 1,
            dtype: RuntimeActivationDType::F32,
            layout: RuntimeActivationLayout::TokenMajor,
            producer_stage_index: config.stage_index as i32,
            layer_start: config.layer_start as i32,
            layer_end: config.layer_end as i32,
            token_count: message.token_count.try_into().unwrap_or(0),
            sequence_count: if message.token_count > 0 { 1 } else { 0 },
            payload_bytes: 0,
            flags: 0,
        },
        payload: Vec::new(),
    }
}

fn upstream_layer_range(
    config: &StageConfig,
    topology: Option<&StageTopology>,
    message: &StageWireMessage,
) -> (i32, i32) {
    if let Some(topology) = topology {
        if let Some(stage) = topology
            .stages
            .iter()
            .find(|stage| stage.stage_index as i32 == message.state.source_stage_index)
        {
            return (stage.layer_start as i32, stage.layer_end as i32);
        }
    }
    (0, config.layer_start as i32)
}

fn token_sideband_or_fill(message: &StageWireMessage) -> Result<Vec<i32>> {
    let token_count: usize = message
        .token_count
        .try_into()
        .context("negative token_count")?;
    if message.tokens.len() == token_count {
        return Ok(message.tokens.clone());
    }
    if !message.tokens.is_empty() && token_count == 1 {
        return Ok(vec![message.tokens[0]]);
    }
    let fill = if message.state.current_token != skippy_protocol::binary::LLAMA_TOKEN_NULL {
        message.state.current_token
    } else {
        0
    };
    Ok(vec![fill; token_count])
}

pub(crate) fn forwarded_stage_message(
    config: &StageConfig,
    incoming: &StageWireMessage,
    output: &ActivationFrame,
    wire_dtype: WireActivationDType,
    activation_width: i32,
) -> Result<StageWireMessage> {
    Ok(
        forwarded_stage_message_timed(config, incoming, output, wire_dtype, activation_width)?
            .message,
    )
}

pub(crate) struct ForwardedStageMessage {
    pub message: StageWireMessage,
    pub activation_encode_ms: f64,
}

pub(crate) fn forwarded_stage_message_timed(
    config: &StageConfig,
    incoming: &StageWireMessage,
    output: &ActivationFrame,
    wire_dtype: WireActivationDType,
    activation_width: i32,
) -> Result<ForwardedStageMessage> {
    let mut state = incoming.state;
    state.source_stage_index = config.stage_index as i32;
    state.reserved = wire_dtype as i32;
    let encode_started = Instant::now();
    let activation = skippy_protocol::binary::encode_f32_activation_payload(
        wire_dtype,
        incoming.token_count,
        activation_width,
        &output.payload,
    )
    .context("encode output activation payload")?;
    let activation_encode_ms = elapsed_ms(encode_started);
    Ok(ForwardedStageMessage {
        message: StageWireMessage {
            kind: incoming.kind,
            pos_start: incoming.pos_start,
            token_count: incoming.token_count,
            state,
            request_id: incoming.request_id,
            session_id: incoming.session_id,
            sampling: incoming.sampling.clone(),
            tokens: incoming.tokens.clone(),
            activation,
            raw_bytes: Vec::new(),
        },
        activation_encode_ms,
    })
}

pub fn parse_wire_dtype(value: &str) -> Result<WireActivationDType> {
    match value {
        "fp32" | "f32" => Ok(WireActivationDType::F32),
        "fp16" | "f16" => Ok(WireActivationDType::F16),
        "q8" | "int8" | "i8" => Ok(WireActivationDType::Q8),
        _ => bail!("unsupported activation wire dtype {value}"),
    }
}

#[cfg(test)]
mod tests {
    use super::tensor_name_requires_recurrent_state;

    #[test]
    fn recurrent_tensor_names_disable_kv_only_cache() {
        assert!(tensor_name_requires_recurrent_state("blk.0.ssm_a"));
        assert!(tensor_name_requires_recurrent_state(
            "blk.0.ssm_conv1d.weight"
        ));
        assert!(tensor_name_requires_recurrent_state(
            "blk.0.time_mix_k.weight"
        ));
        assert!(tensor_name_requires_recurrent_state(
            "blk.0.rwkv_gate.weight"
        ));
        assert!(!tensor_name_requires_recurrent_state("blk.0.attn_q.weight"));
        assert!(!tensor_name_requires_recurrent_state(
            "blk.0.ffn_down.weight"
        ));
    }
}
