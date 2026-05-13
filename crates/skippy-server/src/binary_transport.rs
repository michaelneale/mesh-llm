use std::{
    collections::{BTreeMap, VecDeque},
    future::Future,
    io,
    net::{IpAddr, SocketAddr, TcpListener, TcpStream, ToSocketAddrs},
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        mpsc, Arc, Mutex,
    },
    thread,
    time::{Duration, Instant},
};

use crate::{
    cli::ServeBinaryArgs,
    config::validate_config,
    frontend::{self, EmbeddedOpenAiArgs},
    kv_integration::{KvStageIntegration, PrefillKvIdentity},
    runtime_state::{load_runtime, RuntimeSessionStats, RuntimeState},
    telemetry::{lifecycle_attrs, now_unix_nanos, Telemetry},
};
use anyhow::{anyhow, bail, Context, Result};
use serde_json::{json, Value};
use skippy_metrics::{attr, metric};
use skippy_protocol::{
    binary::{
        activation_frame_flags_from_state_flags, read_stage_message, recv_reply, send_ready,
        send_reply_ack, send_reply_ack_with_stats, send_reply_predicted_tokens_with_stats,
        send_reply_predicted_with_stats, state_flags, StageReplyStats, StageSamplingConfig,
        StageStateHeader, StageWireMessage, WireActivationDType, WireMessageKind, WireReplyKind,
        STAGE_LOGIT_BIAS_WIRE_BYTES, STAGE_SAMPLING_CONFIG_BASE_BYTES,
        STAGE_WIRE_FIXED_HEADER_BYTES,
    },
    MessageBase, StageConfig, StageTopology, SCHEMA_VERSION,
};
use skippy_runtime::{
    ActivationDesc, ActivationFrame, LogitBias, RuntimeActivationDType, RuntimeActivationLayout,
    SamplingConfig, MAX_LOGIT_BIAS,
};
use socket2::{Domain, Protocol, SockAddr, Socket, Type};

pub(crate) mod forwarding;
mod options;
mod socket;
mod wire;

pub(crate) use self::forwarding::{forwarded_stage_message, forwarded_stage_message_timed};
pub use self::options::{parse_wire_dtype, BinaryStageOptions, EmbeddedOpenAiStageOptions};
use self::socket::*;
pub(crate) use self::wire::write_stage_message_conditioned;
pub use self::wire::WireCondition;

static BINARY_SESSION_COUNTER: AtomicU64 = AtomicU64::new(1);

#[derive(Default)]
struct BinaryKvLookupResult {
    restored_tokens: usize,
    activation: Option<ActivationFrame>,
    stats: StageReplyStats,
}

#[derive(Default)]
struct BinaryKvRecordResult {
    recorded_pages: usize,
    recorded_tokens: u64,
    evicted_entries: usize,
    evicted_tokens: u64,
    recorded_activations: usize,
    recorded_activation_bytes: u64,
    evicted_activation_entries: usize,
    evicted_activation_bytes: u64,
}

#[derive(Default)]
struct BinaryPrefixCacheControlResult {
    hit: bool,
    stats: StageReplyStats,
}

struct BinaryRestoredPrefix {
    page_id: String,
    token_count: usize,
    entries: usize,
    resident_seq_id: Option<i32>,
    resident_borrowed: Option<bool>,
    exact: bool,
}

impl BinaryRestoredPrefix {
    fn exact(page_id: String, token_count: usize, entries: usize) -> Self {
        Self {
            page_id,
            token_count,
            entries,
            resident_seq_id: None,
            resident_borrowed: None,
            exact: true,
        }
    }

    fn resident(
        page_id: String,
        token_count: usize,
        seq_id: i32,
        entries: usize,
        borrowed: bool,
    ) -> Self {
        Self {
            page_id,
            token_count,
            entries,
            resident_seq_id: Some(seq_id),
            resident_borrowed: Some(borrowed),
            exact: false,
        }
    }

    fn insert_hit_attrs(&self, attrs: &mut BTreeMap<String, Value>) {
        if self.exact {
            attrs.insert(
                "skippy.exact_cache.hit_page_id".to_string(),
                json!(self.page_id),
            );
            attrs.insert(
                "skippy.exact_cache.entries".to_string(),
                json!(self.entries),
            );
        } else {
            attrs.insert("skippy.kv.hit_page_id".to_string(), json!(self.page_id));
            attrs.insert(
                "skippy.kv.resident_entries".to_string(),
                json!(self.entries),
            );
            if let Some(seq_id) = self.resident_seq_id {
                attrs.insert("skippy.kv.resident_seq_id".to_string(), json!(seq_id));
            }
            if let Some(borrowed) = self.resident_borrowed {
                attrs.insert("skippy.kv.resident_lane_hit".to_string(), json!(borrowed));
            }
        }
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
        max_inflight: _,
        reply_credit_limit,
        async_prefill_forward,
        downstream_wire_condition,
        downstream_connect_timeout_secs,
        openai,
    } = options;
    validate_config(&config, topology.as_ref())?;
    let max_inflight = config.lane_count as usize;
    let telemetry = Telemetry::new(
        metrics_otlp_grpc,
        telemetry_queue_capacity,
        config.clone(),
        telemetry_level,
    );
    telemetry.emit("stage.binary_server_start", lifecycle_attrs(&config));
    let runtime = load_runtime(&config)?.context("binary stage server requires model_path")?;
    {
        let timer = Instant::now();
        let sessions = runtime
            .lock()
            .map_err(|_| anyhow!("runtime lock poisoned"))?
            .prewarm_idle_sessions(max_inflight)
            .context("prewarm binary stage runtime sessions")?;
        let mut attrs = lifecycle_attrs(&config);
        attrs.insert("llama_stage.max_inflight".to_string(), json!(max_inflight));
        attrs.insert(
            "llama_stage.lane_count".to_string(),
            json!(sessions.lane_count),
        );
        attrs.insert(
            "llama_stage.runtime_sessions_active".to_string(),
            json!(sessions.active_sessions),
        );
        attrs.insert(
            "llama_stage.runtime_sessions_idle".to_string(),
            json!(sessions.idle_sessions),
        );
        attrs.insert(
            "llama_stage.elapsed_ms".to_string(),
            json!(timer.elapsed().as_secs_f64() * 1000.0),
        );
        telemetry.emit("stage.binary_runtime_prewarm", attrs);
    }
    let kv = KvStageIntegration::from_config(&config)?.map(Arc::new);
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
            if let Err(error) = frontend::serve_embedded_openai(EmbeddedOpenAiArgs {
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
                hook_policy: None,
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
        prepare_binary_stage_connection(&upstream)?;
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

fn prepare_binary_stage_connection(stream: &TcpStream) -> Result<()> {
    stream
        .set_nonblocking(false)
        .context("set binary stage connection blocking")?;
    stream.set_nodelay(true).ok();
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
    let mut accumulated_prefill_tokens: BTreeMap<String, Vec<i32>> = BTreeMap::new();
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
        let recv_start_unix_nanos = now_unix_nanos() as u64;
        let recv_started = Instant::now();
        let message = match read_stage_message(&mut *upstream, activation_width) {
            Ok(message) => message,
            Err(error)
                if error.kind() == io::ErrorKind::UnexpectedEof
                    && pending_prefill_replies == 0
                    && request_summary.message_count == 0 =>
            {
                return Ok(());
            }
            Err(error) => return Err(error).context("read binary stage message"),
        };
        let recv_end_unix_nanos = now_unix_nanos() as u64;
        let recv_read_ms = elapsed_ms(recv_started);
        let message_start_unix_nanos = now_unix_nanos() as u64;
        let message_started = Instant::now();
        let session_id = binary_message_session_id(connection_session_id, &message);
        let session_key = session_id.to_string();
        let mut recv_attrs = binary_message_attrs(config, session_id, &message);
        recv_attrs.insert(
            "llama_stage.recv_start_unix_nanos".to_string(),
            json!(recv_start_unix_nanos),
        );
        recv_attrs.insert(
            "llama_stage.recv_end_unix_nanos".to_string(),
            json!(recv_end_unix_nanos),
        );
        recv_attrs.insert("llama_stage.recv_read_ms".to_string(), json!(recv_read_ms));
        recv_attrs.insert(
            "llama_stage.source_stage_index".to_string(),
            json!(message.state.source_stage_index),
        );
        recv_attrs.insert(
            "llama_stage.configured_upstream_stage_index".to_string(),
            json!(config.upstream.as_ref().map(|peer| peer.stage_index)),
        );
        recv_attrs.insert(
            "llama_stage.message_wire_bytes".to_string(),
            json!(estimated_stage_message_wire_bytes(&message)),
        );
        recv_attrs.insert(
            "skippy.activation_bytes".to_string(),
            json!(message.activation.len()),
        );
        telemetry.emit_debug_span(
            "stage.binary_recv",
            recv_attrs,
            recv_start_unix_nanos,
            recv_end_unix_nanos,
        );

        if message.kind == WireMessageKind::Stop {
            if pending_prefill_replies != 0 {
                bail!("cannot stop with {pending_prefill_replies} deferred prefill replies");
            }
            let mut stop_stats = std::mem::take(&mut pending_reply_stats);
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
            let accumulated = std::mem::take(&mut accumulated_prefill_tokens);
            for (prefill_session_key, tokens) in accumulated {
                let record = maybe_record_binary_full_prefill(
                    config,
                    &mut runtime,
                    kv,
                    telemetry,
                    &prefill_session_key,
                    &message,
                    &tokens,
                );
                if record.recorded_pages > 0 {
                    stop_stats.kv_recorded_pages += record.recorded_pages as i64;
                    stop_stats.kv_record_stage_mask |= stage_mask(config.stage_index);
                }
            }
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
                &drop_stats.stats_after,
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
                    WireMessageKind::TrimSession => runtime
                        .trim_session(&session_key, message.token_count.max(0) as u64)
                        .context("trim binary stage session")?,
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

        if message.kind.is_generation_control() {
            let mut generation_stats = std::mem::take(&mut pending_reply_stats);
            if let Some(forwarder) = async_forwarder.as_mut() {
                forwarder
                    .flush()
                    .context("flush async forwards before generation config")?;
            }
            drain_deferred_prefill_replies(
                downstream.as_mut(),
                &mut pending_prefill_replies,
                &mut generation_stats,
            )
            .context("drain deferred replies before generation config")?;
            if let Some(downstream) = downstream.as_mut() {
                write_stage_message_conditioned(
                    &mut *downstream,
                    &message,
                    wire_dtype,
                    downstream_wire_condition,
                )
                .context("forward generation config")?;
                let reply =
                    recv_reply(&mut *downstream).context("generation config downstream ACK")?;
                if reply.kind != WireReplyKind::Ack {
                    bail!("generation config expected downstream ACK");
                }
                generation_stats.merge(reply.stats);
            } else if let Some(metadata) = message.chat_sampling_metadata.as_deref() {
                let sampling = runtime_sampling_config(message.sampling.as_ref());
                let mut runtime = runtime.lock().expect("runtime lock poisoned");
                runtime
                    .configure_chat_sampling(
                        &session_key,
                        metadata,
                        message.state.prompt_token_count.max(0) as u64,
                        sampling.as_ref(),
                    )
                    .context("configure binary stage generation")?;
            }
            send_reply_ack_with_stats(&mut *upstream, generation_stats)
                .context("generation config ack")?;
            continue;
        }

        if message.kind.is_prefix_cache_control() {
            let control_started = Instant::now();
            let mut control_stats = std::mem::take(&mut pending_reply_stats);
            if let Some(forwarder) = async_forwarder.as_mut() {
                forwarder
                    .flush()
                    .context("flush async forwards before prefix cache control")?;
            }
            drain_deferred_prefill_replies(
                downstream.as_mut(),
                &mut pending_prefill_replies,
                &mut control_stats,
            )
            .context("drain deferred replies before prefix cache control")?;
            if message.kind == WireMessageKind::TryRestorePrefillDecode {
                handle_binary_restore_prefill_decode_control(
                    config,
                    topology,
                    runtime,
                    kv,
                    telemetry,
                    &session_key,
                    session_id,
                    &message,
                    downstream.as_mut(),
                    upstream,
                    wire_dtype,
                    downstream_wire_condition,
                    activation_width,
                    control_started,
                    control_stats,
                )
                .context("handle restore-prefill-decode control")?;
                continue;
            }
            let token_ids = token_sideband_or_fill(&message)
                .context("read prefix cache control token sideband")?;
            let local = maybe_prefix_cache_control(
                config,
                runtime,
                kv,
                telemetry,
                &session_key,
                &message,
                &token_ids,
            );
            control_stats.merge(local.stats);
            if local.hit {
                if let Some(downstream) = downstream.as_mut() {
                    write_stage_message_conditioned(
                        &mut *downstream,
                        &message,
                        wire_dtype,
                        downstream_wire_condition,
                    )
                    .context("forward prefix cache control")?;
                    let reply =
                        recv_reply(&mut *downstream).context("prefix cache downstream ACK")?;
                    if reply.kind != WireReplyKind::Ack {
                        bail!("prefix cache control expected downstream ACK");
                    }
                    let downstream_missed = message.kind == WireMessageKind::TryRestorePrefill
                        && (reply.stats.kv_lookup_misses > 0
                            || reply.stats.kv_lookup_errors > 0
                            || reply.stats.kv_lookup_hits == 0);
                    control_stats.merge(reply.stats);
                    if downstream_missed {
                        let mut runtime = runtime.lock().expect("runtime lock poisoned");
                        let _ = runtime.drop_session_timed(&session_key);
                    }
                }
            }
            let mut attrs = binary_message_attrs(config, session_id, &message);
            attrs.insert("skippy.kv.control_hit".to_string(), json!(local.hit));
            attrs.insert(
                "llama_stage.elapsed_ms".to_string(),
                json!(elapsed_ms(control_started)),
            );
            telemetry.emit_debug("stage.binary_prefix_cache_control", attrs);
            send_reply_ack_with_stats(&mut *upstream, control_stats)
                .context("prefix cache control ack")?;
            continue;
        }

        if message.kind == WireMessageKind::StateImport {
            bail!("binary state import is no longer supported by the skippy runtime ABI");
        }

        if message.kind == WireMessageKind::StateExport {
            bail!("binary state export is no longer supported by the skippy runtime ABI");
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

        let token_ids = token_sideband_or_fill(&message)?;
        if message.kind.is_prefill() {
            accumulate_prefill_tokens(
                &mut accumulated_prefill_tokens,
                &session_key,
                message.pos_start.max(0) as usize,
                &token_ids,
            );
        }
        let mut message_reply_stats = StageReplyStats::default();
        let lookup_result = maybe_lookup_binary_prefill(
            config,
            runtime,
            kv,
            telemetry,
            &session_key,
            &message,
            &token_ids,
            activation_width,
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
        let mut input_activation_decode_ms = 0.0;
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
                lookup_result
                    .activation
                    .clone()
                    .unwrap_or_else(|| empty_activation_frame(config, &message)),
                0.0,
            )
        } else {
            let input_decode_started = Instant::now();
            let input = input_activation_frame(config, topology, &message, activation_width)?;
            input_activation_decode_ms = if message.activation.is_empty() {
                0.0
            } else {
                elapsed_ms(input_decode_started)
            };
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
                    message.kind == WireMessageKind::PrefillFinalEmbd && downstream.is_none(),
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
        if let Some(stats) = runtime_sessions_before.as_ref() {
            insert_runtime_session_stats(
                &mut decode_attrs,
                "llama_stage.runtime_sessions_before",
                stats,
            );
        }
        if let Some(stats) = runtime_sessions_after.as_ref() {
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
            let record = if let Some(tokens) = accumulated_prefill_tokens.get(&session_key).cloned()
            {
                let mut runtime = runtime.lock().expect("runtime lock poisoned");
                let mut record = maybe_record_binary_full_prefill(
                    config,
                    &mut runtime,
                    kv,
                    telemetry,
                    &session_key,
                    &message,
                    &tokens,
                );
                drop(runtime);
                if let Some(kv) = kv {
                    if config.downstream.is_some() {
                        let base = binary_message_base(config, &session_key, &message);
                        if let Some(activation) = kv.record_resident_activation(
                            config,
                            &base,
                            0,
                            &tokens,
                            activation_width,
                            &output,
                        ) {
                            record.recorded_activations =
                                record.recorded_activations.saturating_add(1);
                            record.recorded_activation_bytes = record
                                .recorded_activation_bytes
                                .saturating_add(activation.payload_bytes as u64);
                            record.evicted_activation_entries = record
                                .evicted_activation_entries
                                .saturating_add(activation.evicted_entries);
                            record.evicted_activation_bytes = record
                                .evicted_activation_bytes
                                .saturating_add(activation.evicted_bytes);
                        }
                    }
                }
                record
            } else {
                maybe_record_binary_prefill(
                    config,
                    runtime,
                    kv,
                    telemetry,
                    &session_key,
                    &message,
                    &token_ids,
                    lookup_result.restored_tokens as u64,
                    activation_width,
                    Some(&output),
                )
            };
            if record.recorded_pages > 0 {
                message_reply_stats.kv_recorded_pages += record.recorded_pages as i64;
                message_reply_stats.kv_record_stage_mask |= stage_mask(config.stage_index);
            }
            if record.recorded_activations > 0 {
                message_reply_stats.kv_recorded_bytes = message_reply_stats
                    .kv_recorded_bytes
                    .saturating_add(record.recorded_activation_bytes as i64);
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
                let reply_write_ms = elapsed_ms(reply_started);
                upstream_reply_ms += reply_write_ms;
                emit_upstream_reply_write_span(
                    telemetry,
                    config,
                    session_id,
                    &message,
                    UpstreamReplyWriteSpan {
                        reply_kind: reply.kind,
                        predicted_token_count: reply.predicted_tokens.len(),
                        start_unix_nanos: reply_start_unix_nanos,
                        end_unix_nanos: upstream_reply_end_unix_nanos
                            .unwrap_or(reply_start_unix_nanos),
                        write_ms: reply_write_ms,
                    },
                );
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
                    let reply_write_ms = elapsed_ms(reply_started);
                    upstream_reply_ms += reply_write_ms;
                    emit_upstream_reply_write_span(
                        telemetry,
                        config,
                        session_id,
                        &message,
                        UpstreamReplyWriteSpan {
                            reply_kind: WireReplyKind::Ack,
                            predicted_token_count: 0,
                            start_unix_nanos: reply_start_unix_nanos,
                            end_unix_nanos: upstream_reply_end_unix_nanos
                                .unwrap_or(reply_start_unix_nanos),
                            write_ms: reply_write_ms,
                        },
                    );
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
                    let reply_write_ms = elapsed_ms(reply_started);
                    upstream_reply_ms += reply_write_ms;
                    emit_upstream_reply_write_span(
                        telemetry,
                        config,
                        session_id,
                        &message,
                        UpstreamReplyWriteSpan {
                            reply_kind: WireReplyKind::Ack,
                            predicted_token_count: 0,
                            start_unix_nanos: reply_start_unix_nanos,
                            end_unix_nanos: upstream_reply_end_unix_nanos
                                .unwrap_or(reply_start_unix_nanos),
                            write_ms: reply_write_ms,
                        },
                    );
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
            let reply_write_ms = elapsed_ms(reply_started);
            upstream_reply_ms += reply_write_ms;
            emit_upstream_reply_write_span(
                telemetry,
                config,
                session_id,
                &message,
                UpstreamReplyWriteSpan {
                    reply_kind: if message.kind == WireMessageKind::VerifySpan {
                        WireReplyKind::PredictedTokens
                    } else {
                        WireReplyKind::PredictedToken
                    },
                    predicted_token_count: if message.kind == WireMessageKind::VerifySpan {
                        predicted_tokens.len()
                    } else {
                        1
                    },
                    start_unix_nanos: reply_start_unix_nanos,
                    end_unix_nanos: upstream_reply_end_unix_nanos.unwrap_or(reply_start_unix_nanos),
                    write_ms: reply_write_ms,
                },
            );
        } else if !early_prefill_ack {
            let reply_start_unix_nanos = now_unix_nanos() as u64;
            upstream_reply_start_unix_nanos.get_or_insert(reply_start_unix_nanos);
            let reply_started = Instant::now();
            send_reply_ack_with_stats(&mut *upstream, message_reply_stats).context("send ACK")?;
            upstream_reply_end_unix_nanos = Some(now_unix_nanos() as u64);
            let reply_write_ms = elapsed_ms(reply_started);
            upstream_reply_ms += reply_write_ms;
            emit_upstream_reply_write_span(
                telemetry,
                config,
                session_id,
                &message,
                UpstreamReplyWriteSpan {
                    reply_kind: WireReplyKind::Ack,
                    predicted_token_count: 0,
                    start_unix_nanos: reply_start_unix_nanos,
                    end_unix_nanos: upstream_reply_end_unix_nanos.unwrap_or(reply_start_unix_nanos),
                    write_ms: reply_write_ms,
                },
            );
        } else {
            pending_reply_stats.merge(message_reply_stats);
        }

        let message_end_unix_nanos = now_unix_nanos() as u64;
        let message_elapsed_ms = elapsed_ms(message_started);
        request_summary.observe(BinaryMessageObservation {
            config,
            message: &message,
            reply_stats: message_reply_stats,
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
        if let Some(stats) = runtime_sessions_before.as_ref() {
            insert_runtime_session_stats(
                &mut timing_attrs,
                "llama_stage.runtime_sessions_before",
                stats,
            );
        }
        if let Some(stats) = runtime_sessions_after.as_ref() {
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

fn estimated_stage_message_wire_bytes(message: &StageWireMessage) -> usize {
    let sampling_bytes = message.sampling.as_ref().map_or(0, |sampling| {
        STAGE_SAMPLING_CONFIG_BASE_BYTES
            + sampling
                .logit_bias
                .len()
                .min(skippy_protocol::binary::MAX_STAGE_LOGIT_BIAS)
                * STAGE_LOGIT_BIAS_WIRE_BYTES
    });
    let chat_metadata_bytes = message
        .chat_sampling_metadata
        .as_ref()
        .map_or(0, |metadata| std::mem::size_of::<u32>() + metadata.len());
    let payload_bytes = if message.kind == WireMessageKind::StateImport {
        message.raw_bytes.len()
    } else {
        message.tokens.len() * std::mem::size_of::<i32>()
            + message.positions.len() * std::mem::size_of::<i32>()
            + message.activation.len()
    };

    STAGE_WIRE_FIXED_HEADER_BYTES + sampling_bytes + chat_metadata_bytes + payload_bytes
}

fn estimated_reply_wire_bytes(reply_kind: WireReplyKind, predicted_token_count: usize) -> usize {
    const REPLY_HEADER_BYTES: usize = 3 * std::mem::size_of::<i32>();
    const REPLY_STATS_BYTES: usize = 34 * std::mem::size_of::<i64>();
    let token_count = match reply_kind {
        WireReplyKind::Ack => 0,
        WireReplyKind::PredictedToken => 1,
        WireReplyKind::PredictedTokens => predicted_token_count,
    };
    REPLY_HEADER_BYTES + token_count * std::mem::size_of::<i32>() + REPLY_STATS_BYTES
}

struct UpstreamReplyWriteSpan {
    reply_kind: WireReplyKind,
    predicted_token_count: usize,
    start_unix_nanos: u64,
    end_unix_nanos: u64,
    write_ms: f64,
}

fn emit_upstream_reply_write_span(
    telemetry: &Telemetry,
    config: &StageConfig,
    session_id: u64,
    message: &StageWireMessage,
    span: UpstreamReplyWriteSpan,
) {
    let mut attrs = binary_message_attrs(config, session_id, message);
    attrs.insert(
        "llama_stage.reply_kind".to_string(),
        json!(format!("{:?}", span.reply_kind)),
    );
    attrs.insert(
        "llama_stage.reply_predicted_token_count".to_string(),
        json!(span.predicted_token_count),
    );
    attrs.insert(
        "llama_stage.upstream_reply_ms".to_string(),
        json!(span.write_ms),
    );
    attrs.insert(
        "llama_stage.reply_wire_bytes".to_string(),
        json!(estimated_reply_wire_bytes(
            span.reply_kind,
            span.predicted_token_count
        )),
    );
    attrs.insert(
        "llama_stage.upstream_reply_start_unix_nanos".to_string(),
        json!(span.start_unix_nanos),
    );
    attrs.insert(
        "llama_stage.upstream_reply_end_unix_nanos".to_string(),
        json!(span.end_unix_nanos),
    );
    telemetry.emit_debug_span(
        "stage.binary_upstream_reply_write",
        attrs,
        span.start_unix_nanos,
        span.end_unix_nanos,
    );
}

fn insert_runtime_session_stats(
    attrs: &mut BTreeMap<String, Value>,
    prefix: &str,
    stats: &RuntimeSessionStats,
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
        format!("{prefix}.idle_resident_prefixes"),
        json!(stats.idle_resident_prefixes),
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

fn maybe_prefix_cache_control(
    config: &StageConfig,
    runtime: &Arc<Mutex<RuntimeState>>,
    kv: Option<&Arc<KvStageIntegration>>,
    telemetry: &Telemetry,
    session_id: &str,
    message: &StageWireMessage,
    token_ids: &[i32],
) -> BinaryPrefixCacheControlResult {
    let mut result = BinaryPrefixCacheControlResult::default();
    let Some(kv) = kv else {
        return result;
    };
    if !kv.should_lookup() || token_ids.is_empty() {
        return result;
    }
    let token_start = if message.kind == WireMessageKind::TryRestorePrefillDecode {
        0
    } else {
        message.pos_start.max(0) as u64
    };
    let base = binary_message_base(config, session_id, message);
    let identity = kv.prefill_identity(config, &base, token_start, token_ids);
    let mut attrs = binary_message_kv_attrs(config, kv, session_id, message, token_ids.len());
    attrs.insert("skippy.kv.lookup_candidates".to_string(), json!(1));
    let started = Instant::now();
    if token_start != 0 {
        result.stats.kv_lookup_misses += 1;
        attrs.insert(
            "skippy.kv.lookup_ms".to_string(),
            json!(elapsed_ms(started)),
        );
        attrs.insert(
            "skippy.kv.decision".to_string(),
            json!("nonzero_token_start_unsupported"),
        );
        telemetry.emit("stage.binary_kv_lookup_decision", attrs);
        return result;
    }
    match message.kind {
        WireMessageKind::ProbePrefill => {
            if let Some(probed) = kv.probe_resident_prefix(&identity) {
                result.hit = probed.token_count >= token_ids.len();
                if result.hit {
                    result.stats.kv_lookup_hits += 1;
                    result.stats.kv_hit_stage_mask |= stage_mask(config.stage_index);
                    attrs.insert("skippy.kv.decision".to_string(), json!("probe_hit"));
                    attrs.insert("skippy.kv.hit_page_id".to_string(), json!(probed.page_id));
                    attrs.insert(
                        "skippy.kv.restored_tokens".to_string(),
                        json!(probed.token_count),
                    );
                    attrs.insert(
                        "skippy.kv.resident_seq_id".to_string(),
                        json!(probed.seq_id),
                    );
                    attrs.insert(
                        "skippy.kv.resident_entries".to_string(),
                        json!(probed.entries),
                    );
                } else {
                    result.stats.kv_lookup_misses += 1;
                    attrs.insert("skippy.kv.decision".to_string(), json!("probe_short"));
                    attrs.insert(
                        "skippy.kv.restored_tokens".to_string(),
                        json!(probed.token_count),
                    );
                }
            } else {
                result.stats.kv_lookup_misses += 1;
                attrs.insert("skippy.kv.decision".to_string(), json!("probe_miss"));
            }
        }
        WireMessageKind::RestorePrefill
        | WireMessageKind::TryRestorePrefill
        | WireMessageKind::TryRestorePrefillDecode => {
            let restore = {
                let mut runtime = runtime.lock().expect("runtime lock poisoned");
                restore_binary_prefix(
                    kv,
                    &mut runtime,
                    session_id,
                    std::slice::from_ref(&identity),
                    token_ids,
                )
            };
            match restore {
                Ok(Some(restored)) if restored.token_count >= token_ids.len() => {
                    result.hit = true;
                    result.stats.kv_lookup_hits += 1;
                    result.stats.kv_imported_tokens += restored.token_count as i64;
                    result.stats.kv_imported_pages += 1;
                    result.stats.kv_hit_stage_mask |= stage_mask(config.stage_index);
                    let decision = match message.kind {
                        WireMessageKind::TryRestorePrefill => "try_restore_hit",
                        WireMessageKind::TryRestorePrefillDecode => "try_restore_decode_hit",
                        _ => "restore_hit",
                    };
                    attrs.insert("skippy.kv.decision".to_string(), json!(decision));
                    restored.insert_hit_attrs(&mut attrs);
                    attrs.insert(
                        "skippy.kv.restored_tokens".to_string(),
                        json!(restored.token_count),
                    );
                }
                Ok(Some(restored)) => {
                    result.stats.kv_lookup_misses += 1;
                    let decision = match message.kind {
                        WireMessageKind::TryRestorePrefill => "try_restore_short",
                        WireMessageKind::TryRestorePrefillDecode => "try_restore_decode_short",
                        _ => "restore_short",
                    };
                    attrs.insert("skippy.kv.decision".to_string(), json!(decision));
                    attrs.insert(
                        "skippy.kv.restored_tokens".to_string(),
                        json!(restored.token_count),
                    );
                }
                Ok(None) => {
                    result.stats.kv_lookup_misses += 1;
                    let decision = match message.kind {
                        WireMessageKind::TryRestorePrefill => "try_restore_miss",
                        WireMessageKind::TryRestorePrefillDecode => "try_restore_decode_miss",
                        _ => "restore_miss",
                    };
                    attrs.insert("skippy.kv.decision".to_string(), json!(decision));
                }
                Err(error) => {
                    result.stats.kv_lookup_errors += 1;
                    let decision = match message.kind {
                        WireMessageKind::TryRestorePrefill => "try_restore_error",
                        WireMessageKind::TryRestorePrefillDecode => "try_restore_decode_error",
                        _ => "restore_error",
                    };
                    attrs.insert("skippy.kv.decision".to_string(), json!(decision));
                    attrs.insert("skippy.kv.error".to_string(), json!(error.to_string()));
                }
            }
        }
        _ => return result,
    }
    attrs.insert(
        "skippy.kv.lookup_ms".to_string(),
        json!(elapsed_ms(started)),
    );
    telemetry.emit("stage.binary_kv_lookup_decision", attrs);
    result
}

fn restore_binary_prefix(
    kv: &KvStageIntegration,
    runtime: &mut RuntimeState,
    session_id: &str,
    identities: &[PrefillKvIdentity],
    token_ids: &[i32],
) -> Result<Option<BinaryRestoredPrefix>> {
    match kv.restore_exact_state(runtime, session_id, identities)? {
        Some(restored) => Ok(Some(BinaryRestoredPrefix::exact(
            restored.page_id,
            restored.token_count,
            restored.entries,
        ))),
        None => kv
            .restore_resident_prefix(runtime, session_id, identities, token_ids)
            .map(|restored| {
                restored.map(|restored| {
                    BinaryRestoredPrefix::resident(
                        restored.page_id,
                        restored.token_count,
                        restored.seq_id,
                        restored.entries,
                        restored.borrowed,
                    )
                })
            }),
    }
}

#[allow(clippy::too_many_arguments)]
fn handle_binary_restore_prefill_decode_control(
    config: &StageConfig,
    topology: Option<&StageTopology>,
    runtime: &Arc<Mutex<RuntimeState>>,
    kv: Option<&Arc<KvStageIntegration>>,
    telemetry: &Telemetry,
    session_id: &str,
    wire_session_id: u64,
    message: &StageWireMessage,
    downstream: Option<&mut TcpStream>,
    upstream: &mut TcpStream,
    wire_dtype: WireActivationDType,
    downstream_wire_condition: WireCondition,
    activation_width: i32,
    control_started: Instant,
    mut control_stats: StageReplyStats,
) -> Result<()> {
    let (prefix_tokens, current_token) = restore_decode_sideband(message)?;
    let local = maybe_prefix_cache_control(
        config,
        runtime,
        kv,
        telemetry,
        session_id,
        message,
        prefix_tokens,
    );
    control_stats.merge(local.stats);
    if !local.hit {
        let mut attrs = binary_message_attrs(config, wire_session_id, message);
        attrs.insert("skippy.kv.control_hit".to_string(), json!(false));
        attrs.insert(
            "llama_stage.elapsed_ms".to_string(),
            json!(elapsed_ms(control_started)),
        );
        telemetry.emit_debug("stage.binary_prefix_cache_decode_control", attrs);
        send_reply_ack_with_stats(upstream, control_stats).context("restore-decode miss ACK")?;
        return Ok(());
    }

    let input = input_activation_frame(config, topology, message, activation_width)?;
    let decode_message = restore_prefill_decode_as_decode_message(message, current_token);
    let compute_started = Instant::now();
    let (predicted_token, output, runtime_lock_wait_ms, runtime_lock_hold_ms) = {
        let lock_started = Instant::now();
        let mut runtime = runtime.lock().expect("runtime lock poisoned");
        let runtime_lock_wait_ms = elapsed_ms(lock_started);
        let lock_hold_started = Instant::now();
        if let Some(metadata) = message.chat_sampling_metadata.as_deref() {
            let sampling = runtime_sampling_config(message.sampling.as_ref());
            runtime
                .configure_chat_sampling(
                    session_id,
                    metadata,
                    message.state.prompt_token_count.max(0) as u64,
                    sampling.as_ref(),
                )
                .context("configure restore-decode chat sampling")?;
        }
        let (predicted, _, output) = run_binary_stage_message(
            &mut runtime,
            session_id,
            &decode_message,
            &[current_token],
            input.as_ref(),
            downstream.is_none(),
        )
        .context("execute restore-decode stage message")?;
        (
            predicted,
            output,
            runtime_lock_wait_ms,
            elapsed_ms(lock_hold_started),
        )
    };
    let compute_ms = elapsed_ms(compute_started);

    if let Some(downstream) = downstream {
        let forwarded =
            forwarded_stage_message_timed(config, message, &output, wire_dtype, activation_width)
                .context("forward restore-decode activation")?;
        write_stage_message_conditioned(
            &mut *downstream,
            &forwarded.message,
            wire_dtype,
            downstream_wire_condition,
        )
        .context("forward restore-decode downstream")?;
        let reply = recv_reply(&mut *downstream).context("restore-decode downstream reply")?;
        let downstream_missed = reply.kind != WireReplyKind::PredictedToken
            || reply.stats.kv_lookup_misses > 0
            || reply.stats.kv_lookup_errors > 0
            || reply.stats.kv_lookup_hits == 0;
        control_stats.merge(reply.stats);
        if downstream_missed {
            let mut runtime = runtime.lock().expect("runtime lock poisoned");
            let _ = runtime.drop_session_timed(session_id);
            let mut attrs = binary_message_attrs(config, wire_session_id, message);
            attrs.insert("skippy.kv.control_hit".to_string(), json!(false));
            attrs.insert(
                "llama_stage.elapsed_ms".to_string(),
                json!(elapsed_ms(control_started)),
            );
            attrs.insert("llama_stage.compute_ms".to_string(), json!(compute_ms));
            attrs.insert(
                "llama_stage.runtime_lock_wait_ms".to_string(),
                json!(runtime_lock_wait_ms),
            );
            attrs.insert(
                "llama_stage.runtime_lock_hold_ms".to_string(),
                json!(runtime_lock_hold_ms),
            );
            telemetry.emit_debug("stage.binary_prefix_cache_decode_control", attrs);
            send_reply_ack_with_stats(upstream, control_stats)
                .context("restore-decode downstream miss ACK")?;
            return Ok(());
        }
        let mut attrs = binary_message_attrs(config, wire_session_id, message);
        attrs.insert("skippy.kv.control_hit".to_string(), json!(true));
        attrs.insert(
            "llama_stage.elapsed_ms".to_string(),
            json!(elapsed_ms(control_started)),
        );
        attrs.insert("llama_stage.compute_ms".to_string(), json!(compute_ms));
        attrs.insert(
            "llama_stage.runtime_lock_wait_ms".to_string(),
            json!(runtime_lock_wait_ms),
        );
        attrs.insert(
            "llama_stage.runtime_lock_hold_ms".to_string(),
            json!(runtime_lock_hold_ms),
        );
        attrs.insert(
            "llama_stage.forward_activation_bytes".to_string(),
            json!(forwarded.message.activation.len()),
        );
        attrs.insert(
            "llama_stage.activation_encode_ms".to_string(),
            json!(forwarded.activation_encode_ms),
        );
        telemetry.emit_debug("stage.binary_prefix_cache_decode_control", attrs);
        send_reply_predicted_with_stats(upstream, reply.predicted, control_stats)
            .context("restore-decode predicted-token reply")?;
        return Ok(());
    }

    let mut attrs = binary_message_attrs(config, wire_session_id, message);
    attrs.insert("skippy.kv.control_hit".to_string(), json!(true));
    attrs.insert(
        "llama_stage.elapsed_ms".to_string(),
        json!(elapsed_ms(control_started)),
    );
    attrs.insert("llama_stage.compute_ms".to_string(), json!(compute_ms));
    attrs.insert(
        "llama_stage.runtime_lock_wait_ms".to_string(),
        json!(runtime_lock_wait_ms),
    );
    attrs.insert(
        "llama_stage.runtime_lock_hold_ms".to_string(),
        json!(runtime_lock_hold_ms),
    );
    telemetry.emit_debug("stage.binary_prefix_cache_decode_control", attrs);
    send_reply_predicted_with_stats(upstream, predicted_token, control_stats)
        .context("restore-decode final predicted-token reply")?;
    Ok(())
}

fn restore_decode_sideband(message: &StageWireMessage) -> Result<(&[i32], i32)> {
    let Some((&current, prefix_tokens)) = message.tokens.split_last() else {
        bail!("restore-decode message requires prefix tokens plus current token");
    };
    if prefix_tokens.is_empty() {
        bail!("restore-decode message requires non-empty prefix tokens");
    }
    Ok((prefix_tokens, current))
}

fn restore_prefill_decode_as_decode_message(
    message: &StageWireMessage,
    current_token: i32,
) -> StageWireMessage {
    let mut decode = message.clone();
    decode.kind = WireMessageKind::DecodeEmbd;
    decode.token_count = 1;
    decode.tokens = vec![current_token];
    decode.positions.clear();
    decode.activation.clear();
    decode.raw_bytes.clear();
    decode.state.phase = StageStateHeader::new(
        WireMessageKind::DecodeEmbd,
        message.state.dtype().unwrap_or(WireActivationDType::F32),
    )
    .phase;
    decode.state.current_token = current_token;
    decode
}

#[allow(clippy::too_many_arguments)]
fn maybe_lookup_binary_prefill(
    config: &StageConfig,
    runtime: &Arc<Mutex<RuntimeState>>,
    kv: Option<&Arc<KvStageIntegration>>,
    telemetry: &Telemetry,
    session_id: &str,
    message: &StageWireMessage,
    token_ids: &[i32],
    activation_width: i32,
) -> BinaryKvLookupResult {
    let mut result = BinaryKvLookupResult::default();
    let Some(kv) = kv else {
        return result;
    };
    if !message.kind.is_prefill()
        || message.kind.requires_predicted_reply()
        || !kv.should_lookup()
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
    if token_start != 0 {
        result.stats.kv_lookup_misses += 1;
        attrs.insert(
            "skippy.kv.lookup_ms".to_string(),
            json!(elapsed_ms(started)),
        );
        attrs.insert(
            "skippy.kv.decision".to_string(),
            json!("nonzero_token_start_unsupported"),
        );
        telemetry.emit("stage.binary_kv_lookup_decision", attrs);
        return result;
    }
    if config.downstream.is_some() {
        let Some(activation) =
            kv.restore_resident_activation(config, &base, token_start, token_ids, activation_width)
        else {
            result.stats.kv_lookup_misses += 1;
            attrs.insert(
                "skippy.kv.lookup_ms".to_string(),
                json!(elapsed_ms(started)),
            );
            attrs.insert(
                "skippy.kv.decision".to_string(),
                json!("activation_resident_miss"),
            );
            telemetry.emit("stage.binary_kv_lookup_decision", attrs);
            return result;
        };
        let prefix_restore = {
            let mut runtime = runtime.lock().expect("runtime lock poisoned");
            restore_binary_prefix(
                kv,
                &mut runtime,
                session_id,
                std::slice::from_ref(&activation.identity),
                token_ids,
            )
        };
        match prefix_restore {
            Ok(Some(restored)) if restored.token_count >= token_ids.len() => {
                result.restored_tokens = restored.token_count;
                result.activation = Some(activation.frame);
                result.stats.kv_lookup_hits += 1;
                result.stats.kv_imported_tokens += restored.token_count as i64;
                result.stats.kv_imported_pages += 1;
                result.stats.kv_hit_stage_mask |= stage_mask(config.stage_index);
                attrs.insert(
                    "skippy.kv.lookup_ms".to_string(),
                    json!(elapsed_ms(started)),
                );
                attrs.insert(
                    "skippy.kv.decision".to_string(),
                    json!("activation_resident_hit"),
                );
                restored.insert_hit_attrs(&mut attrs);
                attrs.insert(
                    "skippy.activation_cache.hit_page_id".to_string(),
                    json!(activation.page_id),
                );
                attrs.insert(
                    "skippy.kv.restored_tokens".to_string(),
                    json!(restored.token_count),
                );
                attrs.insert(
                    "skippy.activation_cache.payload_bytes".to_string(),
                    json!(activation.payload_bytes),
                );
                attrs.insert(
                    "skippy.activation_cache.entries".to_string(),
                    json!(activation.entries),
                );
                telemetry.emit("stage.binary_kv_lookup_decision", attrs);
                return result;
            }
            Ok(Some(restored)) => {
                result.stats.kv_lookup_misses += 1;
                attrs.insert(
                    "skippy.kv.lookup_ms".to_string(),
                    json!(elapsed_ms(started)),
                );
                attrs.insert(
                    "skippy.kv.decision".to_string(),
                    json!("activation_hit_prefix_short"),
                );
                attrs.insert(
                    "skippy.kv.restored_tokens".to_string(),
                    json!(restored.token_count),
                );
                telemetry.emit("stage.binary_kv_lookup_decision", attrs);
                return result;
            }
            Ok(None) => {
                result.stats.kv_lookup_misses += 1;
                attrs.insert(
                    "skippy.kv.lookup_ms".to_string(),
                    json!(elapsed_ms(started)),
                );
                attrs.insert(
                    "skippy.kv.decision".to_string(),
                    json!("activation_hit_kv_miss"),
                );
                attrs.insert(
                    "skippy.activation_cache.hit_page_id".to_string(),
                    json!(activation.page_id),
                );
                telemetry.emit("stage.binary_kv_lookup_decision", attrs);
                return result;
            }
            Err(error) => {
                result.stats.kv_lookup_errors += 1;
                attrs.insert(
                    "skippy.kv.lookup_ms".to_string(),
                    json!(elapsed_ms(started)),
                );
                attrs.insert(
                    "skippy.kv.decision".to_string(),
                    json!("activation_hit_kv_error"),
                );
                attrs.insert("skippy.kv.error".to_string(), json!(error.to_string()));
                telemetry.emit("stage.binary_kv_lookup_decision", attrs);
                return result;
            }
        }
    }
    let prefix_restore = {
        let mut runtime = runtime.lock().expect("runtime lock poisoned");
        restore_binary_prefix(kv, &mut runtime, session_id, &identities, token_ids)
    };
    match prefix_restore {
        Ok(Some(restored)) => {
            result.restored_tokens = restored.token_count;
            result.stats.kv_lookup_hits += 1;
            result.stats.kv_imported_tokens += restored.token_count as i64;
            result.stats.kv_imported_pages += 1;
            result.stats.kv_hit_stage_mask |= stage_mask(config.stage_index);
            attrs.insert(
                "skippy.kv.lookup_ms".to_string(),
                json!(elapsed_ms(started)),
            );
            attrs.insert("skippy.kv.decision".to_string(), json!("resident_hit"));
            restored.insert_hit_attrs(&mut attrs);
            attrs.insert(
                "skippy.kv.restored_tokens".to_string(),
                json!(restored.token_count),
            );
            attrs.insert(
                "skippy.kv.suffix_prefill_tokens".to_string(),
                json!(token_ids.len().saturating_sub(restored.token_count)),
            );
            telemetry.emit("stage.binary_kv_lookup_decision", attrs);
            return result;
        }
        Ok(None) => {}
        Err(error) => {
            result.stats.kv_lookup_errors += 1;
            attrs.insert(
                "skippy.kv.lookup_ms".to_string(),
                json!(elapsed_ms(started)),
            );
            attrs.insert("skippy.kv.decision".to_string(), json!("resident_error"));
            attrs.insert("skippy.kv.error".to_string(), json!(error.to_string()));
            telemetry.emit("stage.binary_kv_lookup_decision", attrs);
            return result;
        }
    }
    result.stats.kv_lookup_misses += 1;
    attrs.insert(
        "skippy.kv.lookup_ms".to_string(),
        json!(elapsed_ms(started)),
    );
    attrs.insert("skippy.kv.decision".to_string(), json!("resident_miss"));
    telemetry.emit("stage.binary_kv_lookup_decision", attrs);
    result
}

#[allow(clippy::too_many_arguments)]
fn maybe_record_binary_prefill(
    config: &StageConfig,
    runtime: &Arc<Mutex<RuntimeState>>,
    kv: Option<&Arc<KvStageIntegration>>,
    telemetry: &Telemetry,
    session_id: &str,
    message: &StageWireMessage,
    token_ids: &[i32],
    min_record_tokens: u64,
    activation_width: i32,
    output: Option<&ActivationFrame>,
) -> BinaryKvRecordResult {
    let mut result = BinaryKvRecordResult::default();
    let Some(kv) = kv else {
        return result;
    };
    if !message.kind.is_prefill()
        || message.kind.requires_predicted_reply()
        || !kv.should_record()
        || token_ids.is_empty()
    {
        return result;
    }
    let token_start = message.pos_start.max(0) as u64;
    let base = binary_message_base(config, session_id, message);
    let identities = kv.record_identities(config, &base, token_start, token_ids);
    let mut attrs = binary_message_kv_attrs(config, kv, session_id, message, token_ids.len());
    attrs.insert(
        "skippy.kv.record_candidates".to_string(),
        json!(identities.len()),
    );
    let started = Instant::now();
    if token_start != 0 {
        attrs.insert(
            "skippy.kv.record_ms".to_string(),
            json!(elapsed_ms(started)),
        );
        attrs.insert(
            "skippy.kv.decision".to_string(),
            json!("nonzero_token_start_unsupported"),
        );
        telemetry.emit("stage.binary_kv_record_decision", attrs);
        return result;
    }
    {
        let mut runtime = runtime.lock().expect("runtime lock poisoned");
        for identity in identities {
            let token_count = identity.identity.token_count;
            if token_count <= min_record_tokens {
                continue;
            }
            let token_count_usize = usize::try_from(token_count)
                .unwrap_or(usize::MAX)
                .min(token_ids.len());
            if token_count_usize == token_ids.len() {
                match kv.record_exact_state(&mut runtime, session_id, &identity) {
                    Ok(Some(record)) => {
                        result.recorded_pages = result.recorded_pages.saturating_add(1);
                        result.recorded_tokens = result
                            .recorded_tokens
                            .saturating_add(record.token_count as u64);
                        result.evicted_entries = result
                            .evicted_entries
                            .saturating_add(record.evicted_entries);
                        result.evicted_tokens = result
                            .evicted_tokens
                            .saturating_add(record.evicted_logical_bytes);
                        attrs.insert(
                            "skippy.exact_cache.recorded_page_id".to_string(),
                            json!(record.page_id),
                        );
                        attrs.insert(
                            "skippy.exact_cache.payload_kind".to_string(),
                            json!(record.payload_kind.to_string()),
                        );
                        attrs.insert(
                            "skippy.exact_cache.logical_bytes".to_string(),
                            json!(record.logical_bytes),
                        );
                        attrs.insert(
                            "skippy.exact_cache.physical_bytes".to_string(),
                            json!(record.physical_bytes),
                        );
                        attrs.insert(
                            "skippy.exact_cache.entries".to_string(),
                            json!(record.entries),
                        );
                        attrs.insert(
                            "skippy.exact_cache.dedupe_reused_block_count".to_string(),
                            json!(record.dedupe.reused_block_count),
                        );
                        continue;
                    }
                    Ok(None) => {}
                    Err(error) => {
                        attrs.insert(
                            "skippy.exact_cache.record_error".to_string(),
                            json!(error.to_string()),
                        );
                    }
                }
            }
            match kv.record_resident_prefix(
                &mut runtime,
                session_id,
                &identity,
                &token_ids[..token_count_usize],
            ) {
                Ok(Some(record)) => {
                    result.recorded_pages = result.recorded_pages.saturating_add(1);
                    result.recorded_tokens = result
                        .recorded_tokens
                        .saturating_add(record.token_count as u64);
                    result.evicted_entries = result
                        .evicted_entries
                        .saturating_add(record.evicted_entries);
                    result.evicted_tokens =
                        result.evicted_tokens.saturating_add(record.evicted_tokens);
                }
                Ok(None) => {}
                Err(error) => {
                    attrs.insert(
                        "skippy.kv.record_error".to_string(),
                        json!(error.to_string()),
                    );
                    break;
                }
            }
        }
    }
    if config.downstream.is_some() {
        if let Some(output) = output {
            if let Some(record) = kv.record_resident_activation(
                config,
                &base,
                token_start,
                token_ids,
                activation_width,
                output,
            ) {
                result.recorded_activations = result.recorded_activations.saturating_add(1);
                result.recorded_activation_bytes = result
                    .recorded_activation_bytes
                    .saturating_add(record.payload_bytes as u64);
                result.evicted_activation_entries = result
                    .evicted_activation_entries
                    .saturating_add(record.evicted_entries);
                result.evicted_activation_bytes = result
                    .evicted_activation_bytes
                    .saturating_add(record.evicted_bytes);
                attrs.insert(
                    "skippy.activation_cache.recorded_page_id".to_string(),
                    json!(record.page_id),
                );
                attrs.insert(
                    "skippy.activation_cache.entries".to_string(),
                    json!(record.entries),
                );
                attrs.insert(
                    "skippy.activation_cache.resident_bytes".to_string(),
                    json!(record.resident_bytes),
                );
            }
        }
    }
    attrs.insert(
        "skippy.kv.record_ms".to_string(),
        json!(elapsed_ms(started)),
    );
    attrs.insert(
        "skippy.kv.recorded_pages".to_string(),
        json!(result.recorded_pages),
    );
    attrs.insert(
        "skippy.kv.recorded_tokens".to_string(),
        json!(result.recorded_tokens),
    );
    attrs.insert(
        "skippy.kv.evicted_entries".to_string(),
        json!(result.evicted_entries),
    );
    attrs.insert(
        "skippy.kv.evicted_tokens".to_string(),
        json!(result.evicted_tokens),
    );
    attrs.insert(
        "skippy.activation_cache.recorded_frames".to_string(),
        json!(result.recorded_activations),
    );
    attrs.insert(
        "skippy.activation_cache.recorded_bytes".to_string(),
        json!(result.recorded_activation_bytes),
    );
    attrs.insert(
        "skippy.activation_cache.evicted_entries".to_string(),
        json!(result.evicted_activation_entries),
    );
    attrs.insert(
        "skippy.activation_cache.evicted_bytes".to_string(),
        json!(result.evicted_activation_bytes),
    );
    telemetry.emit("stage.binary_kv_record_decision", attrs);
    result
}

fn accumulate_prefill_tokens(
    accumulated: &mut BTreeMap<String, Vec<i32>>,
    session_id: &str,
    token_start: usize,
    token_ids: &[i32],
) {
    if token_ids.is_empty() {
        return;
    }
    let tokens = accumulated.entry(session_id.to_string()).or_default();
    if token_start == 0 {
        tokens.clear();
    }
    if token_start == tokens.len() {
        tokens.extend_from_slice(token_ids);
    }
}

fn maybe_record_binary_full_prefill(
    config: &StageConfig,
    runtime: &mut RuntimeState,
    kv: Option<&Arc<KvStageIntegration>>,
    telemetry: &Telemetry,
    session_id: &str,
    message: &StageWireMessage,
    token_ids: &[i32],
) -> BinaryKvRecordResult {
    let mut result = BinaryKvRecordResult::default();
    let Some(kv) = kv else {
        return result;
    };
    if !kv.should_record() || token_ids.is_empty() {
        return result;
    }
    let base = binary_message_base(config, session_id, message);
    let identity = kv.prefill_identity(config, &base, 0, token_ids);
    let mut attrs = binary_message_kv_attrs(config, kv, session_id, message, token_ids.len());
    attrs.insert("skippy.kv.record_candidates".to_string(), json!(1));
    attrs.insert(
        "skippy.kv.decision".to_string(),
        json!("full_prefill_record"),
    );
    let started = Instant::now();
    match kv.record_exact_state(runtime, session_id, &identity) {
        Ok(Some(record)) => {
            result.recorded_pages = 1;
            result.recorded_tokens = record.token_count as u64;
            result.evicted_entries = record.evicted_entries;
            result.evicted_tokens = record.evicted_logical_bytes;
            attrs.insert(
                "skippy.exact_cache.recorded_page_id".to_string(),
                json!(record.page_id),
            );
            attrs.insert(
                "skippy.exact_cache.payload_kind".to_string(),
                json!(record.payload_kind.to_string()),
            );
            attrs.insert(
                "skippy.exact_cache.logical_bytes".to_string(),
                json!(record.logical_bytes),
            );
            attrs.insert(
                "skippy.exact_cache.physical_bytes".to_string(),
                json!(record.physical_bytes),
            );
            attrs.insert(
                "skippy.exact_cache.entries".to_string(),
                json!(record.entries),
            );
        }
        Ok(None) => {}
        Err(error) => {
            attrs.insert(
                "skippy.exact_cache.record_error".to_string(),
                json!(error.to_string()),
            );
        }
    }
    if result.recorded_pages == 0 {
        match kv.record_resident_prefix(runtime, session_id, &identity, token_ids) {
            Ok(Some(record)) => {
                result.recorded_pages = 1;
                result.recorded_tokens = record.token_count as u64;
                result.evicted_entries = record.evicted_entries;
                result.evicted_tokens = record.evicted_tokens;
                attrs.insert(
                    "skippy.kv.recorded_page_id".to_string(),
                    json!(record.page_id),
                );
                attrs.insert(
                    "skippy.kv.resident_seq_id".to_string(),
                    json!(record.seq_id),
                );
            }
            Ok(None) => {}
            Err(error) => {
                attrs.insert(
                    "skippy.kv.record_error".to_string(),
                    json!(error.to_string()),
                );
            }
        }
    }
    attrs.insert(
        "skippy.kv.record_ms".to_string(),
        json!(elapsed_ms(started)),
    );
    attrs.insert(
        "skippy.kv.recorded_pages".to_string(),
        json!(result.recorded_pages),
    );
    attrs.insert(
        "skippy.kv.recorded_tokens".to_string(),
        json!(result.recorded_tokens),
    );
    telemetry.emit("stage.binary_kv_record_decision", attrs);
    result
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
    reply_stats: StageReplyStats,
}

struct BinaryMessageObservation<'a> {
    config: &'a StageConfig,
    message: &'a StageWireMessage,
    reply_stats: StageReplyStats,
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
        self.reply_stats.merge(observation.reply_stats);
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
        let lookups = self.reply_stats.kv_lookup_hits + self.reply_stats.kv_lookup_misses;
        let hit_rate = if lookups > 0 {
            self.reply_stats.kv_lookup_hits as f64 / lookups as f64
        } else {
            0.0
        };
        attrs.insert(
            metric::KV_LOOKUP_REQUESTS.to_string(),
            json!(lookups.max(0)),
        );
        attrs.insert(
            metric::KV_LOOKUP_HITS.to_string(),
            json!(self.reply_stats.kv_lookup_hits),
        );
        attrs.insert(
            metric::KV_LOOKUP_MISSES.to_string(),
            json!(self.reply_stats.kv_lookup_misses),
        );
        attrs.insert("skippy.kv.lookup_hit_rate".to_string(), json!(hit_rate));
        attrs.insert(
            "skippy.kv.lookup_errors".to_string(),
            json!(self.reply_stats.kv_lookup_errors),
        );
        attrs.insert(
            metric::KV_IMPORTED_PAGES.to_string(),
            json!(self.reply_stats.kv_imported_pages),
        );
        attrs.insert(
            "skippy.kv.imported_tokens".to_string(),
            json!(self.reply_stats.kv_imported_tokens),
        );
        attrs.insert(
            metric::KV_COMMITTED_PAGES.to_string(),
            json!(self.reply_stats.kv_recorded_pages),
        );
        attrs.insert(
            "skippy.kv.recorded_bytes".to_string(),
            json!(self.reply_stats.kv_recorded_bytes),
        );
        attrs.insert(
            "skippy.kv.hit_stage_mask".to_string(),
            json!(self.reply_stats.kv_hit_stage_mask),
        );
        attrs.insert(
            "skippy.kv.record_stage_mask".to_string(),
            json!(self.reply_stats.kv_record_stage_mask),
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

pub(crate) fn run_binary_stage_message(
    runtime: &mut RuntimeState,
    session_id: &str,
    message: &StageWireMessage,
    token_ids: &[i32],
    input: Option<&ActivationFrame>,
    sample_final_prefill: bool,
) -> Result<(i32, Vec<i32>, ActivationFrame)> {
    match message.kind {
        WireMessageKind::PrefillEmbd => {
            let output = runtime.prefill_frame_with_positions(
                session_id,
                token_ids,
                &message.positions,
                input,
            )?;
            Ok((message.state.current_token, Vec::new(), output))
        }
        WireMessageKind::PrefillFinalEmbd if sample_final_prefill => {
            let sampling = runtime_sampling_config(message.sampling.as_ref());
            let (predicted, output) = runtime.prefill_final_frame_sampled(
                session_id,
                token_ids,
                &message.positions,
                sampling.as_ref(),
                input,
            )?;
            Ok((predicted, Vec::new(), output))
        }
        WireMessageKind::PrefillFinalEmbd => {
            let output = runtime.prefill_frame_with_positions(
                session_id,
                token_ids,
                &message.positions,
                input,
            )?;
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
        | WireMessageKind::ConfigureGeneration
        | WireMessageKind::CheckpointSession
        | WireMessageKind::RestoreSession
        | WireMessageKind::TrimSession
        | WireMessageKind::ProbePrefill
        | WireMessageKind::RestorePrefill
        | WireMessageKind::TryRestorePrefill
        | WireMessageKind::TryRestorePrefillDecode => {
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
        min_p: sampling.min_p,
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
            flags: activation_frame_flags_from_state_flags(message.state.flags),
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

#[cfg(test)]
mod tests;
