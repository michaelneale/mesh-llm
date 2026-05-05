use std::{
    collections::HashSet,
    fs,
    net::SocketAddr,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    time::Instant,
};

use anyhow::{bail, Context, Result};
use model_artifact::ModelIdentity;
use model_hf::HfModelRepository;
use model_ref::ModelRef;
use serde::Deserialize;
use serde::Serialize;
use serde_json::json;
use sha2::{Digest, Sha256};
use skippy_protocol::binary::{
    read_stage_message, recv_reply, state_flags, write_stage_message, StageStateHeader,
    StageWireMessage, WireMessageKind, WireReplyKind,
};
use skippy_runtime::{
    package::{materialize_layer_package_details, MaterializedPackage, PackageStageRequest},
    ActivationFrame, RuntimeConfig, RuntimeKvPageDesc, RuntimeLoadMode, StageModel, StageSession,
    GGML_TYPE_F16,
};

use crate::{
    cli::{
        ChainArgs, DtypeMatrixArgs, RuntimeArgs, ServerArgs, SingleStepArgs, SplitScanArgs,
        StageLoadMode, StateHandoffArgs, StatePayloadKind,
    },
    report::{
        BaselineReport, BoundaryReport, ChainReport, ChainStageReport, DtypeMatrixReport,
        PackagePartReport, PackageStageReport, SingleStepReport, SplitReport, SplitScanReport,
        StageModelReport, StateHandoffReport, StatePayloadBlockDigestReport,
        StatePayloadDigestReport,
    },
    support::{
        activation_width, connect_ready, generate_run_id, parse_wire_dtype, temp_config_path_for,
        ChildGuard,
    },
};

struct FullModelResult {
    token_id: i32,
    predicted_token: i32,
}

struct BinarySplitConfig {
    stage_server_bin: PathBuf,
    model: PathBuf,
    stage_model: Option<PathBuf>,
    stage_load_mode: StageLoadMode,
    split_layer: u32,
    layer_end: u32,
    ctx_size: u32,
    n_gpu_layers: i32,
    prompt: String,
    stage1_bind_addr: SocketAddr,
    activation_wire_dtype: String,
    child_logs: bool,
    startup_timeout_secs: u64,
    model_identity: ModelIdentity,
}

struct BinarySplitResult {
    token_id: i32,
    predicted_token: i32,
    activation_width: i32,
    wire_dtype: String,
    boundary_producer_stage_index: i32,
    boundary_layer_start: i32,
    boundary_layer_end: i32,
    boundary_token_count: u32,
    boundary_payload_bytes: u64,
    boundary_wire_payload_bytes: usize,
    stage_models: Vec<StageModelReport>,
}

struct BinaryChainConfig {
    stage_server_bin: PathBuf,
    model: PathBuf,
    stage_model: Option<PathBuf>,
    stage_load_mode: StageLoadMode,
    split_layer_1: u32,
    split_layer_2: u32,
    layer_end: u32,
    ctx_size: u32,
    n_gpu_layers: i32,
    prompt: String,
    stage1_bind_addr: SocketAddr,
    stage2_bind_addr: SocketAddr,
    activation_wire_dtype: String,
    child_logs: bool,
    startup_timeout_secs: u64,
    model_identity: ModelIdentity,
}

struct BinaryChainResult {
    token_id: i32,
    predicted_token: i32,
    activation_width: i32,
    wire_dtype: String,
    stage0_wire_payload_bytes: usize,
    stage0_payload_bytes: u64,
    split_layer_1: u32,
    split_layer_2: u32,
    layer_end: u32,
    stage_models: Vec<StageModelReport>,
}

struct BinaryStateHandoffConfig {
    stage_server_bin: PathBuf,
    model: PathBuf,
    stage_model: Option<PathBuf>,
    stage_load_mode: StageLoadMode,
    state_layer_start: u32,
    state_layer_end: u32,
    state_stage_index: u32,
    layer_end: u32,
    ctx_size: u32,
    n_gpu_layers: i32,
    prompt: String,
    source_bind_addr: SocketAddr,
    restore_bind_addr: SocketAddr,
    activation_width: i32,
    activation_wire_dtype: String,
    state_payload_kind: StatePayloadKind,
    prefix_token_count: Option<usize>,
    cache_hit_repeats: usize,
    runtime_lane_count: Option<u32>,
    borrow_resident_hits: bool,
    cache_decoded_result_hits: bool,
    binary_control: bool,
    child_logs: bool,
    startup_timeout_secs: u64,
    model_identity: ModelIdentity,
}

struct BinaryStateHandoffResult {
    prompt_token_count: usize,
    benchmark_prompt_token_count: usize,
    benchmark_prompt_text: String,
    requested_prefix_token_count: Option<usize>,
    stage_index: u32,
    layer_start: u32,
    layer_end: u32,
    include_embeddings: bool,
    include_output: bool,
    handoff_transport: &'static str,
    state_payload_kind: StatePayloadKind,
    borrowed_resident_hits: bool,
    cached_decoded_result_hits: bool,
    activation_width: i32,
    source_predicted_token: i32,
    restored_predicted_token: i32,
    state_bytes: usize,
    cache_storage_bytes: Option<usize>,
    resident_state_bytes: Option<usize>,
    roundtrip_state_bytes: usize,
    payload_digest: StatePayloadDigestReport,
    tokenize_ms: f64,
    source_prefill_ms: f64,
    source_export_ms: f64,
    source_decode_ms: f64,
    restore_import_ms: f64,
    restore_export_ms: f64,
    restore_decode_ms: f64,
    cache_hit_import_ms: Vec<f64>,
    cache_hit_decode_ms: Vec<f64>,
    matches: bool,
    predicted_token_matches: bool,
    roundtrip_state_matches: bool,
    restored_output_matches: Option<bool>,
    cache_hit_matches: bool,
    stage_models: Vec<StageModelReport>,
}

#[derive(Clone)]
enum LocalStatePayload {
    ResidentKv {
        cache_seq_id: i32,
        token_count: u64,
    },
    FullState(Vec<u8>),
    RecurrentOnly(Vec<u8>),
    KvRecurrent {
        kv_desc: RuntimeKvPageDesc,
        kv: Vec<u8>,
        recurrent: Vec<u8>,
    },
}

impl LocalStatePayload {
    fn byte_len(&self) -> usize {
        match self {
            Self::ResidentKv { .. } => 0,
            Self::FullState(bytes) | Self::RecurrentOnly(bytes) => bytes.len(),
            Self::KvRecurrent { kv, recurrent, .. } => kv.len().saturating_add(recurrent.len()),
        }
    }

    fn same_payload(&self, other: &Self) -> bool {
        match (self, other) {
            (
                Self::ResidentKv {
                    token_count: a_count,
                    ..
                },
                Self::ResidentKv {
                    token_count: b_count,
                    ..
                },
            ) => a_count == b_count,
            (Self::FullState(a), Self::FullState(b))
            | (Self::RecurrentOnly(a), Self::RecurrentOnly(b)) => a == b,
            (
                Self::KvRecurrent {
                    kv_desc: a_desc,
                    kv: a_kv,
                    recurrent: a_recurrent,
                },
                Self::KvRecurrent {
                    kv_desc: b_desc,
                    kv: b_kv,
                    recurrent: b_recurrent,
                },
            ) => a_desc == b_desc && a_kv == b_kv && a_recurrent == b_recurrent,
            _ => false,
        }
    }

    fn digest_report(&self) -> StatePayloadDigestReport {
        match self {
            Self::ResidentKv { .. } => payload_digest_report(
                state_payload_kind_name(StatePayloadKind::ResidentKv),
                &[],
                None,
                None,
            ),
            Self::FullState(bytes) => payload_digest_report(
                state_payload_kind_name(StatePayloadKind::FullState),
                bytes,
                None,
                None,
            ),
            Self::RecurrentOnly(recurrent) => payload_digest_report(
                state_payload_kind_name(StatePayloadKind::RecurrentOnly),
                recurrent,
                None,
                Some(recurrent.as_slice()),
            ),
            Self::KvRecurrent { kv, recurrent, .. } => {
                let mut hasher = Sha256::new();
                hasher.update(b"kv-recurrent:kv:");
                hasher.update((kv.len() as u64).to_le_bytes());
                hasher.update(kv);
                hasher.update(b":recurrent:");
                hasher.update((recurrent.len() as u64).to_le_bytes());
                hasher.update(recurrent);
                let mut report = payload_digest_report(
                    state_payload_kind_name(StatePayloadKind::KvRecurrent),
                    &[],
                    Some(kv.as_slice()),
                    Some(recurrent.as_slice()),
                );
                report.payload_sha256 = hex_sha256_finish(hasher);
                report.total_bytes = kv.len().saturating_add(recurrent.len());
                report
            }
        }
    }
}

pub fn single_step(args: SingleStepArgs) -> Result<()> {
    let model_identity = runtime_model_identity(&args.runtime)?;
    let baseline = run_full_model_decode(&args.runtime)?;
    let report = run_single_step_with_baseline(
        &args.runtime,
        &args.server,
        &model_identity,
        baseline,
        SingleStepCase {
            split_layer: args.split_layer,
            stage1_bind_addr: args.stage1_bind_addr,
            activation_wire_dtype: args.activation_wire_dtype,
        },
    )?;
    emit_report(&report, args.output.report_out.as_deref())?;
    ensure_matches(report.matches, args.allow_mismatch)?;
    Ok(())
}

pub fn chain(args: ChainArgs) -> Result<()> {
    let splits = parse_chain_splits(&args.splits)?;
    let model_identity = runtime_model_identity(&args.runtime)?;
    let baseline = run_full_model_decode(&args.runtime)?;
    let chain = run_binary_chain(BinaryChainConfig {
        stage_server_bin: args.server.stage_server_bin,
        model: args.runtime.model,
        stage_model: args.runtime.stage_model,
        stage_load_mode: args.runtime.stage_load_mode,
        split_layer_1: splits.0,
        split_layer_2: splits.1,
        layer_end: args.runtime.layer_end,
        ctx_size: args.runtime.ctx_size,
        n_gpu_layers: args.runtime.n_gpu_layers,
        prompt: args.runtime.prompt,
        stage1_bind_addr: args.stage1_bind_addr,
        stage2_bind_addr: args.stage2_bind_addr,
        activation_wire_dtype: args.activation_wire_dtype,
        child_logs: args.server.child_logs,
        startup_timeout_secs: args.server.startup_timeout_secs,
        model_identity: model_identity.clone(),
    })?;
    let matches = baseline.predicted_token == chain.predicted_token;
    let report = ChainReport {
        mode: "chain",
        status: status(matches),
        model_identity,
        matches,
        baseline: baseline_report(baseline),
        token_id: chain.token_id,
        predicted_token: chain.predicted_token,
        activation_width: chain.activation_width,
        wire_dtype: chain.wire_dtype,
        stages: vec![
            ChainStageReport {
                stage_index: 0,
                layer_start: 0,
                layer_end: chain.split_layer_1,
                payload_bytes: Some(chain.stage0_payload_bytes),
                wire_payload_bytes: Some(chain.stage0_wire_payload_bytes),
                forwarded_over_binary: false,
                returned_predicted_token: false,
            },
            ChainStageReport {
                stage_index: 1,
                layer_start: chain.split_layer_1,
                layer_end: chain.split_layer_2,
                payload_bytes: None,
                wire_payload_bytes: None,
                forwarded_over_binary: true,
                returned_predicted_token: false,
            },
            ChainStageReport {
                stage_index: 2,
                layer_start: chain.split_layer_2,
                layer_end: chain.layer_end,
                payload_bytes: None,
                wire_payload_bytes: None,
                forwarded_over_binary: false,
                returned_predicted_token: true,
            },
        ],
        stage_models: chain.stage_models,
    };
    emit_report(&report, args.output.report_out.as_deref())?;
    ensure_matches(report.matches, args.allow_mismatch)?;
    Ok(())
}

pub fn split_scan(args: SplitScanArgs) -> Result<()> {
    let splits = parse_split_list(&args.splits)?;
    let model_identity = runtime_model_identity(&args.runtime)?;
    let baseline = run_full_model_decode(&args.runtime)?;
    let mut results = Vec::with_capacity(splits.len());
    for split_layer in splits {
        if split_layer == 0 || split_layer >= args.runtime.layer_end {
            bail!(
                "split layer {split_layer} must be greater than zero and less than layer_end {}",
                args.runtime.layer_end
            );
        }
        results.push(run_single_step_with_baseline(
            &args.runtime,
            &args.server,
            &model_identity,
            FullModelResult {
                token_id: baseline.token_id,
                predicted_token: baseline.predicted_token,
            },
            SingleStepCase {
                split_layer,
                stage1_bind_addr: args.stage1_bind_addr,
                activation_wire_dtype: args.activation_wire_dtype.clone(),
            },
        )?);
    }
    let mismatch_count = results.iter().filter(|result| !result.matches).count();
    let report = SplitScanReport {
        mode: "split-scan",
        status: status(mismatch_count == 0),
        model_identity,
        baseline: baseline_report(baseline),
        split_count: results.len(),
        mismatch_count,
        results,
    };
    emit_report(&report, args.output.report_out.as_deref())?;
    ensure_matches(mismatch_count == 0, args.allow_mismatch)?;
    Ok(())
}

pub fn dtype_matrix(args: DtypeMatrixArgs) -> Result<()> {
    let dtypes = parse_csv(&args.dtypes)?;
    let model_identity = runtime_model_identity(&args.runtime)?;
    let baseline = run_full_model_decode(&args.runtime)?;
    let mut results = Vec::with_capacity(dtypes.len());
    for dtype in dtypes {
        results.push(run_single_step_with_baseline(
            &args.runtime,
            &args.server,
            &model_identity,
            FullModelResult {
                token_id: baseline.token_id,
                predicted_token: baseline.predicted_token,
            },
            SingleStepCase {
                split_layer: args.split_layer,
                stage1_bind_addr: args.stage1_bind_addr,
                activation_wire_dtype: dtype,
            },
        )?);
    }
    let mismatch_count = results.iter().filter(|result| !result.matches).count();
    let report = DtypeMatrixReport {
        mode: "dtype-matrix",
        status: status(mismatch_count == 0),
        model_identity,
        baseline: baseline_report(baseline),
        dtype_count: results.len(),
        mismatch_count,
        results,
    };
    emit_report(&report, args.output.report_out.as_deref())?;
    ensure_matches(mismatch_count == 0, args.allow_mismatch)?;
    Ok(())
}

pub fn state_handoff(args: StateHandoffArgs) -> Result<()> {
    let report_out = args.output.report_out;
    let model_identity = runtime_model_identity(&args.runtime)?;
    let state_layer_end = args.state_layer_end.unwrap_or(args.runtime.layer_end);
    let state_stage_index = args.state_stage_index.unwrap_or({
        if args.state_layer_start == 0 {
            0
        } else if state_layer_end == args.runtime.layer_end {
            2
        } else {
            1
        }
    });
    let handoff = run_binary_state_handoff(BinaryStateHandoffConfig {
        stage_server_bin: args.server.stage_server_bin,
        model: args.runtime.model,
        stage_model: args.runtime.stage_model,
        stage_load_mode: args.runtime.stage_load_mode,
        state_layer_start: args.state_layer_start,
        state_layer_end,
        state_stage_index,
        layer_end: args.runtime.layer_end,
        ctx_size: args.runtime.ctx_size,
        n_gpu_layers: args.runtime.n_gpu_layers,
        prompt: args.runtime.prompt,
        source_bind_addr: args.source_bind_addr,
        restore_bind_addr: args.restore_bind_addr,
        activation_width: args.activation_width,
        activation_wire_dtype: args.activation_wire_dtype,
        state_payload_kind: args.state_payload_kind,
        prefix_token_count: args.prefix_token_count,
        cache_hit_repeats: args.cache_hit_repeats,
        runtime_lane_count: args.runtime_lane_count,
        borrow_resident_hits: args.borrow_resident_hits,
        cache_decoded_result_hits: args.cache_decoded_result_hits,
        binary_control: args.binary_control,
        child_logs: args.server.child_logs,
        startup_timeout_secs: args.server.startup_timeout_secs,
        model_identity: model_identity.clone(),
    })?;
    let report = StateHandoffReport {
        mode: "state-handoff",
        status: status(handoff.matches),
        model_identity,
        matches: handoff.matches,
        predicted_token_matches: handoff.predicted_token_matches,
        roundtrip_state_matches: handoff.roundtrip_state_matches,
        restored_output_matches: handoff.restored_output_matches,
        cache_hit_matches: handoff.cache_hit_matches,
        stage_index: handoff.stage_index,
        layer_start: handoff.layer_start,
        layer_end: handoff.layer_end,
        include_embeddings: handoff.include_embeddings,
        include_output: handoff.include_output,
        handoff_transport: handoff.handoff_transport,
        state_payload_kind: state_payload_kind_name(handoff.state_payload_kind),
        borrowed_resident_hits: handoff.borrowed_resident_hits,
        cached_decoded_result_hits: handoff.cached_decoded_result_hits,
        source_predicted_token: handoff.source_predicted_token,
        restored_predicted_token: handoff.restored_predicted_token,
        prompt_token_count: handoff.prompt_token_count,
        benchmark_prompt_token_count: handoff.benchmark_prompt_token_count,
        benchmark_prompt_text: handoff.benchmark_prompt_text,
        requested_prefix_token_count: handoff.requested_prefix_token_count,
        activation_width: handoff.activation_width,
        state_bytes: handoff.state_bytes,
        state_bytes_per_prompt_token: handoff.state_bytes as f64
            / handoff.prompt_token_count as f64,
        cache_storage_bytes: handoff.cache_storage_bytes,
        cache_storage_bytes_per_prompt_token: handoff
            .cache_storage_bytes
            .map(|bytes| bytes as f64 / handoff.prompt_token_count as f64),
        resident_state_bytes: handoff.resident_state_bytes,
        roundtrip_state_bytes: handoff.roundtrip_state_bytes,
        payload_digest: handoff.payload_digest,
        tokenize_ms: handoff.tokenize_ms,
        source_prefill_ms: handoff.source_prefill_ms,
        source_export_ms: handoff.source_export_ms,
        source_decode_ms: handoff.source_decode_ms,
        restore_import_ms: handoff.restore_import_ms,
        restore_export_ms: handoff.restore_export_ms,
        restore_decode_ms: handoff.restore_decode_ms,
        cache_hit_repeats: handoff.cache_hit_import_ms.len(),
        recompute_total_ms: handoff.source_prefill_ms + handoff.source_decode_ms,
        cache_hit_total_ms: mean_pair_sum(
            &handoff.cache_hit_import_ms,
            &handoff.cache_hit_decode_ms,
        ),
        cache_hit_speedup: speedup(
            handoff.source_prefill_ms + handoff.source_decode_ms,
            mean_pair_sum(&handoff.cache_hit_import_ms, &handoff.cache_hit_decode_ms),
        ),
        cache_hit_import_ms: handoff.cache_hit_import_ms,
        cache_hit_decode_ms: handoff.cache_hit_decode_ms,
        stage_models: handoff.stage_models,
    };
    emit_report(&report, report_out.as_deref())?;
    ensure_matches(report.matches, args.allow_mismatch)?;
    Ok(())
}

struct SingleStepCase {
    split_layer: u32,
    stage1_bind_addr: SocketAddr,
    activation_wire_dtype: String,
}

fn run_single_step_with_baseline(
    runtime: &RuntimeArgs,
    server: &ServerArgs,
    model_identity: &ModelIdentity,
    baseline: FullModelResult,
    case: SingleStepCase,
) -> Result<SingleStepReport> {
    let split = run_binary_split(BinarySplitConfig {
        stage_server_bin: server.stage_server_bin.clone(),
        model: runtime.model.clone(),
        stage_model: runtime.stage_model.clone(),
        stage_load_mode: runtime.stage_load_mode,
        split_layer: case.split_layer,
        layer_end: runtime.layer_end,
        ctx_size: runtime.ctx_size,
        n_gpu_layers: runtime.n_gpu_layers,
        prompt: runtime.prompt.clone(),
        stage1_bind_addr: case.stage1_bind_addr,
        activation_wire_dtype: case.activation_wire_dtype,
        child_logs: server.child_logs,
        startup_timeout_secs: server.startup_timeout_secs,
        model_identity: model_identity.clone(),
    })?;
    let matches = baseline.predicted_token == split.predicted_token;
    let stage_models = split.stage_models.clone();
    Ok(SingleStepReport {
        mode: "single-step",
        status: status(matches),
        model_identity: model_identity.clone(),
        matches,
        baseline: baseline_report(baseline),
        split: split_report(split),
        stage_models,
    })
}

fn run_full_model_decode(args: &RuntimeArgs) -> Result<FullModelResult> {
    let config = RuntimeConfig {
        stage_index: 0,
        layer_start: 0,
        layer_end: args.layer_end,
        ctx_size: args.ctx_size,
        lane_count: 1,
        n_batch: None,
        n_ubatch: None,
        n_threads: None,
        n_threads_batch: None,
        n_gpu_layers: args.n_gpu_layers,
        selected_backend_device: None,
        load_mode: RuntimeLoadMode::RuntimeSlice,
        projector_path: None,
        include_embeddings: true,
        include_output: true,
        filter_tensors_on_load: false,
        cache_type_k: GGML_TYPE_F16,
        cache_type_v: GGML_TYPE_F16,
        flash_attn_type: skippy_runtime::FlashAttentionType::Auto,
    };
    let model = StageModel::open(&args.model, &config).context("failed to open full model")?;
    let tokens = model
        .tokenize(&args.prompt, true)
        .context("failed to tokenize prompt with full model")?;
    let token_id = *tokens.first().context("prompt produced no tokens")?;
    let mut session = model
        .create_session()
        .context("failed to create full-model session")?;
    let predicted_token = session
        .decode_step_frame(token_id, None, 0)
        .context("full model failed to decode")?
        .0;
    Ok(FullModelResult {
        token_id,
        predicted_token,
    })
}

fn run_binary_split(args: BinarySplitConfig) -> Result<BinarySplitResult> {
    if args.split_layer == 0 || args.split_layer >= args.layer_end {
        bail!("split_layer must be greater than zero and less than layer_end");
    }
    let wire_dtype = parse_wire_dtype(&args.activation_wire_dtype)?;
    let stage0_spec = PackageStageSpec {
        topology_id: "correctness-single-step",
        stage_id: "stage-0",
        stage_index: 0,
        layer_start: 0,
        layer_end: args.split_layer,
        include_embeddings: true,
        include_output: false,
    };
    let stage1_spec = PackageStageSpec {
        topology_id: "correctness-single-step",
        stage_id: "stage-1",
        stage_index: 1,
        layer_start: args.split_layer,
        layer_end: args.layer_end,
        include_embeddings: false,
        include_output: true,
    };
    let stage0_resolution = stage_model_resolution(
        &args.model,
        args.stage_model.as_ref(),
        args.stage_load_mode,
        &args.model_identity,
        stage0_spec,
    )?;
    let stage1_resolution = stage_model_resolution(
        &args.model,
        args.stage_model.as_ref(),
        args.stage_load_mode,
        &args.model_identity,
        stage1_spec,
    )?;
    let stage0_config = RuntimeConfig {
        stage_index: 0,
        layer_start: 0,
        layer_end: args.split_layer,
        ctx_size: args.ctx_size,
        lane_count: 1,
        n_batch: None,
        n_ubatch: None,
        n_threads: None,
        n_threads_batch: None,
        n_gpu_layers: args.n_gpu_layers,
        selected_backend_device: None,
        load_mode: runtime_load_mode(args.stage_load_mode),
        projector_path: None,
        include_embeddings: true,
        include_output: false,
        filter_tensors_on_load: true,
        cache_type_k: GGML_TYPE_F16,
        cache_type_v: GGML_TYPE_F16,
        flash_attn_type: skippy_runtime::FlashAttentionType::Auto,
    };
    let stage0 = StageModel::open(&stage0_resolution.path, &stage0_config)
        .context("failed to open stage 0")?;
    let tokens = stage0
        .tokenize(&args.prompt, true)
        .context("failed to tokenize prompt")?;
    let token_id = *tokens.first().context("prompt produced no tokens")?;
    let mut session0 = stage0
        .create_session()
        .context("failed to create stage 0 session")?;
    let (_boundary_prediction, boundary) = session0
        .decode_step_frame(token_id, None, 0)
        .context("stage 0 failed to produce activation frame")?;
    if boundary.payload.is_empty() {
        bail!("stage 0 produced an empty activation frame");
    }
    let activation_width = activation_width(&boundary)?;

    let run_id = generate_run_id();
    let model_id = args.model_identity.model_id.clone();
    let config_path = temp_config_path_for(&run_id, "stage-1");
    let config = json!({
        "run_id": run_id,
        "topology_id": "correctness-single-step",
        "model_id": model_id,
        "model_path": stage_server_model_path(
            &args.model,
            args.stage_model.as_ref(),
            args.stage_load_mode,
            stage1_spec,
        )?,
        "stage_id": "stage-1",
        "stage_index": 1,
        "layer_start": args.split_layer,
        "layer_end": args.layer_end,
        "ctx_size": args.ctx_size,
        "n_gpu_layers": args.n_gpu_layers,
        "filter_tensors_on_load": true,
        "load_mode": protocol_load_mode(args.stage_load_mode),
        "bind_addr": args.stage1_bind_addr,
        "upstream": {
            "stage_id": "stage-0",
            "stage_index": 0,
            "endpoint": "driver"
        },
        "downstream": null
    });
    fs::write(&config_path, serde_json::to_vec_pretty(&config)?)
        .with_context(|| format!("failed to write {}", config_path.display()))?;

    let mut stage_command = Command::new(&args.stage_server_bin);
    stage_command.args([
        "serve-binary",
        "--config",
        config_path
            .to_str()
            .context("stage config path is not valid UTF-8")?,
        "--activation-width",
        &activation_width.to_string(),
        "--activation-wire-dtype",
        &args.activation_wire_dtype,
    ]);
    configure_child_logs(&mut stage_command, args.child_logs);
    let _stage1 = ChildGuard::spawn(stage_command)?;

    let mut stream = connect_ready(args.stage1_bind_addr, args.startup_timeout_secs)
        .context("stage 1 binary server did not become ready")?;
    let mut state = StageStateHeader::new(WireMessageKind::DecodeEmbd, wire_dtype);
    state.prompt_token_count = 0;
    state.decode_step = 0;
    state.current_token = token_id;
    state.source_stage_index = 0;
    let activation = skippy_protocol::binary::encode_f32_activation_payload(
        wire_dtype,
        1,
        activation_width,
        &boundary.payload,
    )
    .context("failed to encode boundary activation for wire")?;
    let message = StageWireMessage {
        kind: WireMessageKind::DecodeEmbd,
        pos_start: 0,
        token_count: 1,
        state,
        request_id: 1,
        session_id: 1,
        sampling: None,
        chat_sampling_metadata: None,
        tokens: vec![token_id],
        activation,
        raw_bytes: Vec::new(),
    };
    write_stage_message(&mut stream, &message, wire_dtype).context("send binary decode")?;
    let reply = recv_reply(&mut stream).context("receive binary reply")?;
    if reply.kind != WireReplyKind::PredictedToken {
        bail!("expected predicted-token reply, got {:?}", reply.kind);
    }
    write_stage_message(&mut stream, &StageWireMessage::stop(wire_dtype), wire_dtype)
        .context("send binary stop")?;

    Ok(BinarySplitResult {
        token_id,
        predicted_token: reply.predicted,
        activation_width,
        wire_dtype: args.activation_wire_dtype,
        boundary_producer_stage_index: boundary.desc.producer_stage_index,
        boundary_layer_start: boundary.desc.layer_start,
        boundary_layer_end: boundary.desc.layer_end,
        boundary_token_count: boundary.desc.token_count,
        boundary_payload_bytes: boundary.desc.payload_bytes,
        boundary_wire_payload_bytes: message.activation.len(),
        stage_models: vec![stage0_resolution.report, stage1_resolution.report],
    })
}

fn run_binary_chain(args: BinaryChainConfig) -> Result<BinaryChainResult> {
    if args.split_layer_1 == 0
        || args.split_layer_1 >= args.split_layer_2
        || args.split_layer_2 >= args.layer_end
    {
        bail!("splits must partition 0..layer_end in ascending order");
    }
    let wire_dtype = parse_wire_dtype(&args.activation_wire_dtype)?;
    let stage0_spec = PackageStageSpec {
        topology_id: "correctness-chain",
        stage_id: "stage-0",
        stage_index: 0,
        layer_start: 0,
        layer_end: args.split_layer_1,
        include_embeddings: true,
        include_output: false,
    };
    let stage1_spec = PackageStageSpec {
        topology_id: "correctness-chain",
        stage_id: "stage-1",
        stage_index: 1,
        layer_start: args.split_layer_1,
        layer_end: args.split_layer_2,
        include_embeddings: false,
        include_output: false,
    };
    let stage2_spec = PackageStageSpec {
        topology_id: "correctness-chain",
        stage_id: "stage-2",
        stage_index: 2,
        layer_start: args.split_layer_2,
        layer_end: args.layer_end,
        include_embeddings: false,
        include_output: true,
    };
    let stage0_resolution = stage_model_resolution(
        &args.model,
        args.stage_model.as_ref(),
        args.stage_load_mode,
        &args.model_identity,
        stage0_spec,
    )?;
    let stage1_resolution = stage_model_resolution(
        &args.model,
        args.stage_model.as_ref(),
        args.stage_load_mode,
        &args.model_identity,
        stage1_spec,
    )?;
    let stage2_resolution = stage_model_resolution(
        &args.model,
        args.stage_model.as_ref(),
        args.stage_load_mode,
        &args.model_identity,
        stage2_spec,
    )?;
    let stage0_config = RuntimeConfig {
        stage_index: 0,
        layer_start: 0,
        layer_end: args.split_layer_1,
        ctx_size: args.ctx_size,
        lane_count: 1,
        n_batch: None,
        n_ubatch: None,
        n_threads: None,
        n_threads_batch: None,
        n_gpu_layers: args.n_gpu_layers,
        selected_backend_device: None,
        load_mode: runtime_load_mode(args.stage_load_mode),
        projector_path: None,
        include_embeddings: true,
        include_output: false,
        filter_tensors_on_load: true,
        cache_type_k: GGML_TYPE_F16,
        cache_type_v: GGML_TYPE_F16,
        flash_attn_type: skippy_runtime::FlashAttentionType::Auto,
    };
    let stage0 = StageModel::open(&stage0_resolution.path, &stage0_config)
        .context("failed to open stage 0")?;
    let tokens = stage0
        .tokenize(&args.prompt, true)
        .context("failed to tokenize prompt")?;
    let token_id = *tokens.first().context("prompt produced no tokens")?;
    let mut session0 = stage0
        .create_session()
        .context("failed to create stage 0 session")?;
    let (_boundary_prediction, boundary) = session0
        .decode_step_frame(token_id, None, 0)
        .context("stage 0 failed to produce activation frame")?;
    if boundary.payload.is_empty() {
        bail!("stage 0 produced an empty activation frame");
    }
    let activation_width = activation_width(&boundary)?;

    let run_id = generate_run_id();
    let model_id = args.model_identity.model_id.clone();
    let stage1_config_path = temp_config_path_for(&run_id, "stage-1");
    let stage2_config_path = temp_config_path_for(&run_id, "stage-2");
    let stage2_config = json!({
        "run_id": run_id,
        "topology_id": "correctness-chain",
        "model_id": model_id,
        "model_path": stage_server_model_path(
            &args.model,
            args.stage_model.as_ref(),
            args.stage_load_mode,
            stage2_spec,
        )?,
        "stage_id": "stage-2",
        "stage_index": 2,
        "layer_start": args.split_layer_2,
        "layer_end": args.layer_end,
        "ctx_size": args.ctx_size,
        "n_gpu_layers": args.n_gpu_layers,
        "filter_tensors_on_load": true,
        "load_mode": protocol_load_mode(args.stage_load_mode),
        "bind_addr": args.stage2_bind_addr,
        "upstream": {
            "stage_id": "stage-1",
            "stage_index": 1,
            "endpoint": format!("tcp://{}", args.stage1_bind_addr)
        },
        "downstream": null
    });
    let stage1_config = json!({
        "run_id": run_id,
        "topology_id": "correctness-chain",
        "model_id": model_id,
        "model_path": stage_server_model_path(
            &args.model,
            args.stage_model.as_ref(),
            args.stage_load_mode,
            stage1_spec,
        )?,
        "stage_id": "stage-1",
        "stage_index": 1,
        "layer_start": args.split_layer_1,
        "layer_end": args.split_layer_2,
        "ctx_size": args.ctx_size,
        "n_gpu_layers": args.n_gpu_layers,
        "filter_tensors_on_load": true,
        "load_mode": protocol_load_mode(args.stage_load_mode),
        "bind_addr": args.stage1_bind_addr,
        "upstream": {
            "stage_id": "stage-0",
            "stage_index": 0,
            "endpoint": "driver"
        },
        "downstream": {
            "stage_id": "stage-2",
            "stage_index": 2,
            "endpoint": format!("tcp://{}", args.stage2_bind_addr)
        }
    });
    fs::write(
        &stage2_config_path,
        serde_json::to_vec_pretty(&stage2_config)?,
    )
    .with_context(|| format!("failed to write {}", stage2_config_path.display()))?;
    fs::write(
        &stage1_config_path,
        serde_json::to_vec_pretty(&stage1_config)?,
    )
    .with_context(|| format!("failed to write {}", stage1_config_path.display()))?;

    let mut stage2_command = Command::new(&args.stage_server_bin);
    stage2_command.args([
        "serve-binary",
        "--config",
        stage2_config_path
            .to_str()
            .context("stage 2 config path is not valid UTF-8")?,
        "--activation-width",
        &activation_width.to_string(),
        "--activation-wire-dtype",
        &args.activation_wire_dtype,
    ]);
    configure_child_logs(&mut stage2_command, args.child_logs);
    let _stage2 = ChildGuard::spawn(stage2_command)?;

    let mut stage1_command = Command::new(&args.stage_server_bin);
    stage1_command.args([
        "serve-binary",
        "--config",
        stage1_config_path
            .to_str()
            .context("stage 1 config path is not valid UTF-8")?,
        "--activation-width",
        &activation_width.to_string(),
        "--activation-wire-dtype",
        &args.activation_wire_dtype,
    ]);
    configure_child_logs(&mut stage1_command, args.child_logs);
    let _stage1 = ChildGuard::spawn(stage1_command)?;

    let mut stream = connect_ready(args.stage1_bind_addr, args.startup_timeout_secs)
        .context("stage 1 binary server did not become ready")?;
    let mut state = StageStateHeader::new(WireMessageKind::DecodeEmbd, wire_dtype);
    state.prompt_token_count = 0;
    state.decode_step = 0;
    state.current_token = token_id;
    state.source_stage_index = 0;
    let activation = skippy_protocol::binary::encode_f32_activation_payload(
        wire_dtype,
        1,
        activation_width,
        &boundary.payload,
    )
    .context("failed to encode boundary activation for wire")?;
    let message = StageWireMessage {
        kind: WireMessageKind::DecodeEmbd,
        pos_start: 0,
        token_count: 1,
        state,
        request_id: 2,
        session_id: 2,
        sampling: None,
        chat_sampling_metadata: None,
        tokens: vec![token_id],
        activation,
        raw_bytes: Vec::new(),
    };
    write_stage_message(&mut stream, &message, wire_dtype).context("send binary chain decode")?;
    let reply = recv_reply(&mut stream).context("receive binary chain reply")?;
    if reply.kind != WireReplyKind::PredictedToken {
        bail!("expected predicted-token reply, got {:?}", reply.kind);
    }
    write_stage_message(&mut stream, &StageWireMessage::stop(wire_dtype), wire_dtype)
        .context("send binary chain stop")?;

    Ok(BinaryChainResult {
        token_id,
        predicted_token: reply.predicted,
        activation_width,
        wire_dtype: args.activation_wire_dtype,
        stage0_wire_payload_bytes: message.activation.len(),
        stage0_payload_bytes: boundary.desc.payload_bytes,
        split_layer_1: args.split_layer_1,
        split_layer_2: args.split_layer_2,
        layer_end: args.layer_end,
        stage_models: vec![
            stage0_resolution.report,
            stage1_resolution.report,
            stage2_resolution.report,
        ],
    })
}

fn run_binary_state_handoff(args: BinaryStateHandoffConfig) -> Result<BinaryStateHandoffResult> {
    let tokenize_started = Instant::now();
    let wire_dtype = parse_wire_dtype(&args.activation_wire_dtype)?;
    if args.state_layer_start >= args.state_layer_end || args.state_layer_end > args.layer_end {
        bail!(
            "state handoff range {}..{} must be non-empty and within 0..{}",
            args.state_layer_start,
            args.state_layer_end,
            args.layer_end
        );
    }
    if args.cache_hit_repeats == 0 {
        bail!("cache_hit_repeats must be greater than zero");
    }
    let include_embeddings = args.state_layer_start == 0;
    let include_output = args.state_layer_end == args.layer_end;
    let stage_spec = PackageStageSpec {
        topology_id: "correctness-state-handoff",
        stage_id: stage_id_for_index(args.state_stage_index),
        stage_index: args.state_stage_index,
        layer_start: args.state_layer_start,
        layer_end: args.state_layer_end,
        include_embeddings,
        include_output,
    };
    let stage_resolution = stage_model_resolution(
        &args.model,
        args.stage_model.as_ref(),
        args.stage_load_mode,
        &args.model_identity,
        stage_spec,
    )?;
    let (tokenizer_path, tokenizer_config) = tokenizer_model_for_state_handoff(&args)?;
    let tokenizer = StageModel::open(&tokenizer_path, &tokenizer_config).with_context(|| {
        format!(
            "failed to open tokenizer model {}",
            tokenizer_path.display()
        )
    })?;
    let tokens = state_handoff_tokens(&tokenizer, &args.prompt, args.prefix_token_count)
        .context("failed to tokenize state handoff prompt")?;
    let split = args.prefix_token_count.unwrap_or(tokens.len() - 1);
    let prefix = tokens[..split].to_vec();
    let continuation = tokens[split];
    let benchmark_prompt_text = tokenizer
        .detokenize(&tokens[..=split])
        .context("failed to detokenize state handoff benchmark prompt")?;
    drop(tokenizer);
    let tokenize_ms = elapsed_ms(tokenize_started);
    let input_started = Instant::now();
    let input_resolution = if args.state_layer_start == 0 {
        None
    } else {
        Some(stage_model_resolution(
            &args.model,
            args.stage_model.as_ref(),
            args.stage_load_mode,
            &args.model_identity,
            PackageStageSpec {
                topology_id: "correctness-state-handoff",
                stage_id: stage_id_for_index(args.state_stage_index.saturating_sub(1)),
                stage_index: args.state_stage_index.saturating_sub(1),
                layer_start: 0,
                layer_end: args.state_layer_start,
                include_embeddings: true,
                include_output: false,
            },
        )?)
    };
    let (prefill_input, decode_input, stage_activation_width) =
        build_state_handoff_inputs(&args, input_resolution.as_ref(), &prefix, continuation)
            .context("build state handoff input activations")?;
    let input_build_ms = elapsed_ms(input_started);
    let use_binary_control = args.binary_control
        && include_output
        && args.state_payload_kind == StatePayloadKind::FullState;
    if !use_binary_control {
        return run_local_state_handoff(
            &args,
            stage_resolution,
            input_resolution,
            prefix,
            continuation,
            benchmark_prompt_text,
            prefill_input,
            decode_input,
            tokenize_ms + input_build_ms,
            stage_activation_width,
            include_embeddings,
            include_output,
        );
    }

    let run_id = generate_run_id();
    let model_id = args.model_identity.model_id.clone();
    let source_config_path = temp_config_path_for(&run_id, "state-source");
    let restore_config_path = temp_config_path_for(&run_id, "state-restore");
    let source_config = json!({
        "run_id": run_id,
        "topology_id": "correctness-state-handoff",
        "model_id": model_id,
        "model_path": stage_server_model_path(
            &args.model,
            args.stage_model.as_ref(),
            args.stage_load_mode,
            stage_spec,
        )?,
        "stage_id": "state-source",
        "stage_index": args.state_stage_index,
        "layer_start": args.state_layer_start,
        "layer_end": args.state_layer_end,
        "ctx_size": args.ctx_size,
        "n_gpu_layers": args.n_gpu_layers,
        "filter_tensors_on_load": should_filter_state_handoff_tensors(&args),
        "load_mode": protocol_load_mode(args.stage_load_mode),
        "bind_addr": args.source_bind_addr,
        "upstream": {
            "stage_id": "driver",
            "stage_index": 0,
            "endpoint": "driver"
        },
        "downstream": null
    });
    let restore_config = json!({
        "run_id": run_id,
        "topology_id": "correctness-state-handoff",
        "model_id": model_id,
        "model_path": stage_server_model_path(
            &args.model,
            args.stage_model.as_ref(),
            args.stage_load_mode,
            stage_spec,
        )?,
        "stage_id": "state-restore",
        "stage_index": args.state_stage_index,
        "layer_start": args.state_layer_start,
        "layer_end": args.state_layer_end,
        "ctx_size": args.ctx_size,
        "n_gpu_layers": args.n_gpu_layers,
        "filter_tensors_on_load": should_filter_state_handoff_tensors(&args),
        "load_mode": protocol_load_mode(args.stage_load_mode),
        "bind_addr": args.restore_bind_addr,
        "upstream": {
            "stage_id": "driver",
            "stage_index": 0,
            "endpoint": "driver"
        },
        "downstream": null
    });
    fs::write(
        &source_config_path,
        serde_json::to_vec_pretty(&source_config)?,
    )
    .with_context(|| format!("failed to write {}", source_config_path.display()))?;
    fs::write(
        &restore_config_path,
        serde_json::to_vec_pretty(&restore_config)?,
    )
    .with_context(|| format!("failed to write {}", restore_config_path.display()))?;

    let mut source_command = Command::new(&args.stage_server_bin);
    source_command.args([
        "serve-binary",
        "--config",
        source_config_path
            .to_str()
            .context("source config path is not valid UTF-8")?,
        "--activation-width",
        &stage_activation_width.to_string(),
        "--activation-wire-dtype",
        &args.activation_wire_dtype,
    ]);
    configure_child_logs(&mut source_command, args.child_logs);
    let _source = ChildGuard::spawn(source_command)?;

    let mut source_stream = connect_ready(args.source_bind_addr, args.startup_timeout_secs)
        .context("source binary server did not become ready")?;
    let source_prefill_started = Instant::now();
    send_prefill_for_state_handoff(
        &mut source_stream,
        &prefix,
        prefill_input.as_ref(),
        wire_dtype,
        stage_activation_width,
    )
    .context("send source prefill")?;
    let source_prefill_ms = elapsed_ms(source_prefill_started);
    let source_export_started = Instant::now();
    let state_bytes =
        export_state_over_binary(&mut source_stream, wire_dtype, args.activation_width, true)
            .context("export source state")?;
    let source_export_ms = elapsed_ms(source_export_started);
    let source_decode_started = Instant::now();
    let source_predicted_token = decode_for_state_handoff(
        &mut source_stream,
        continuation,
        prefix.len(),
        decode_input.as_ref(),
        wire_dtype,
        stage_activation_width,
    )
    .context("decode source continuation")?;
    let source_decode_ms = elapsed_ms(source_decode_started);
    write_stage_message(
        &mut source_stream,
        &StageWireMessage::stop(wire_dtype),
        wire_dtype,
    )
    .context("send source stop")?;
    drop(source_stream);
    drop(_source);

    let mut restore_command = Command::new(&args.stage_server_bin);
    restore_command.args([
        "serve-binary",
        "--config",
        restore_config_path
            .to_str()
            .context("restore config path is not valid UTF-8")?,
        "--activation-width",
        &stage_activation_width.to_string(),
        "--activation-wire-dtype",
        &args.activation_wire_dtype,
    ]);
    configure_child_logs(&mut restore_command, args.child_logs);
    let _restore = ChildGuard::spawn(restore_command)?;

    let mut restore_stream = connect_ready(args.restore_bind_addr, args.startup_timeout_secs)
        .context("restore binary server did not become ready")?;
    let restore_import_started = Instant::now();
    import_state_over_binary(&mut restore_stream, &state_bytes, wire_dtype, true)
        .context("import state into restore server")?;
    let restore_import_ms = elapsed_ms(restore_import_started);
    let restore_export_started = Instant::now();
    let roundtrip_state_bytes =
        export_state_over_binary(&mut restore_stream, wire_dtype, args.activation_width, true)
            .context("export restored state")?;
    let restore_export_ms = elapsed_ms(restore_export_started);
    let restore_decode_started = Instant::now();
    let restored_predicted_token = decode_for_state_handoff(
        &mut restore_stream,
        continuation,
        prefix.len(),
        decode_input.as_ref(),
        wire_dtype,
        stage_activation_width,
    )
    .context("decode restored continuation")?;
    let restore_decode_ms = elapsed_ms(restore_decode_started);
    let mut cache_hit_import_ms = vec![restore_import_ms];
    let mut cache_hit_decode_ms = vec![restore_decode_ms];
    let predicted_token_matches = source_predicted_token == restored_predicted_token;
    let mut cache_hit_matches = predicted_token_matches;
    for _ in 1..args.cache_hit_repeats {
        let import_started = Instant::now();
        import_state_over_binary(&mut restore_stream, &state_bytes, wire_dtype, true)
            .context("repeat import state into restore server")?;
        cache_hit_import_ms.push(elapsed_ms(import_started));
        let decode_started = Instant::now();
        let predicted = decode_for_state_handoff(
            &mut restore_stream,
            continuation,
            prefix.len(),
            decode_input.as_ref(),
            wire_dtype,
            stage_activation_width,
        )
        .context("repeat decode restored continuation")?;
        cache_hit_decode_ms.push(elapsed_ms(decode_started));
        cache_hit_matches &= predicted == source_predicted_token;
    }
    write_stage_message(
        &mut restore_stream,
        &StageWireMessage::stop(wire_dtype),
        wire_dtype,
    )
    .context("send restore stop")?;

    let mut stage_models = Vec::new();
    if let Some(input_resolution) = input_resolution {
        stage_models.push(input_resolution.report);
    }
    stage_models.push(stage_resolution.report);

    let roundtrip_state_matches = state_bytes == roundtrip_state_bytes;
    Ok(BinaryStateHandoffResult {
        prompt_token_count: prefix.len(),
        benchmark_prompt_token_count: prefix.len().saturating_add(1),
        benchmark_prompt_text: benchmark_prompt_text.clone(),
        requested_prefix_token_count: args.prefix_token_count,
        stage_index: args.state_stage_index,
        layer_start: args.state_layer_start,
        layer_end: args.state_layer_end,
        include_embeddings,
        include_output,
        handoff_transport: "binary-control",
        state_payload_kind: args.state_payload_kind,
        borrowed_resident_hits: false,
        cached_decoded_result_hits: false,
        activation_width: stage_activation_width,
        source_predicted_token,
        restored_predicted_token,
        state_bytes: state_bytes.len(),
        cache_storage_bytes: Some(state_bytes.len()),
        resident_state_bytes: None,
        roundtrip_state_bytes: roundtrip_state_bytes.len(),
        payload_digest: payload_digest_report(
            state_payload_kind_name(args.state_payload_kind),
            &state_bytes,
            None,
            None,
        ),
        tokenize_ms: tokenize_ms + input_build_ms,
        source_prefill_ms,
        source_export_ms,
        source_decode_ms,
        restore_import_ms,
        restore_export_ms,
        restore_decode_ms,
        cache_hit_import_ms,
        cache_hit_decode_ms,
        matches: predicted_token_matches && cache_hit_matches,
        predicted_token_matches,
        roundtrip_state_matches,
        restored_output_matches: None,
        cache_hit_matches,
        stage_models,
    })
}

fn stage_id_for_index(stage_index: u32) -> &'static str {
    match stage_index {
        0 => "stage-0",
        1 => "stage-1",
        2 => "stage-2",
        _ => "stage-n",
    }
}

fn elapsed_ms(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1000.0
}

fn mean_pair_sum(left: &[f64], right: &[f64]) -> f64 {
    let count = left.len().min(right.len());
    if count == 0 {
        return 0.0;
    }
    left.iter()
        .zip(right.iter())
        .take(count)
        .map(|(left, right)| left + right)
        .sum::<f64>()
        / count as f64
}

fn speedup(recompute_ms: f64, cache_ms: f64) -> f64 {
    if cache_ms <= f64::EPSILON {
        return 0.0;
    }
    recompute_ms / cache_ms
}

fn state_handoff_tokens(
    tokenizer: &StageModel,
    prompt: &str,
    prefix_token_count: Option<usize>,
) -> Result<Vec<i32>> {
    let mut text = prompt.to_string();
    let needed = prefix_token_count.unwrap_or(1).saturating_add(1);
    let mut tokens = tokenizer
        .tokenize(&text, true)
        .context("failed to tokenize prompt")?;
    if tokens.len() >= needed {
        tokens.truncate(needed);
        return Ok(tokens);
    }

    let filler = " Deterministic full-state cache economics filler sentence.";
    for _ in 0..10_000 {
        text.push_str(filler);
        tokens = tokenizer
            .tokenize(&text, true)
            .context("failed to tokenize expanded prompt")?;
        if tokens.len() >= needed {
            tokens.truncate(needed);
            return Ok(tokens);
        }
    }

    bail!(
        "could not expand prompt to {needed} tokens for state handoff prefix sweep; reached {} tokens",
        tokens.len()
    )
}

#[allow(clippy::too_many_arguments)]
fn run_local_state_handoff(
    args: &BinaryStateHandoffConfig,
    stage_resolution: StageModelResolution,
    input_resolution: Option<StageModelResolution>,
    prefix: Vec<i32>,
    continuation: i32,
    benchmark_prompt_text: String,
    prefill_input: Option<ActivationFrame>,
    decode_input: Option<ActivationFrame>,
    tokenize_ms: f64,
    activation_width: i32,
    include_embeddings: bool,
    include_output: bool,
) -> Result<BinaryStateHandoffResult> {
    let lane_count = effective_state_handoff_lane_count(args);
    let runtime_config = RuntimeConfig {
        stage_index: args.state_stage_index,
        layer_start: args.state_layer_start,
        layer_end: args.state_layer_end,
        ctx_size: args.ctx_size,
        lane_count,
        n_batch: None,
        n_ubatch: None,
        n_threads: None,
        n_threads_batch: None,
        n_gpu_layers: args.n_gpu_layers,
        selected_backend_device: None,
        load_mode: runtime_load_mode(args.stage_load_mode),
        projector_path: None,
        include_embeddings,
        include_output,
        filter_tensors_on_load: should_filter_state_handoff_tensors(args),
        cache_type_k: GGML_TYPE_F16,
        cache_type_v: GGML_TYPE_F16,
        flash_attn_type: skippy_runtime::FlashAttentionType::Auto,
    };
    let model = StageModel::open(&stage_resolution.path, &runtime_config)
        .context("failed to open local state handoff stage")?;

    if args.borrow_resident_hits && args.state_payload_kind == StatePayloadKind::ResidentKv {
        return run_local_resident_slot_handoff(
            model,
            args,
            stage_resolution,
            input_resolution,
            prefix,
            continuation,
            benchmark_prompt_text,
            prefill_input,
            decode_input,
            tokenize_ms,
            activation_width,
            include_embeddings,
            include_output,
        );
    }

    let (
        state_payload,
        resident_state_bytes,
        source_predicted_token,
        source_output,
        source_prefill_ms,
        source_export_ms,
        source_decode_ms,
    ) = {
        let mut source = model
            .create_session()
            .context("failed to create local state handoff source session")?;
        let source_prefill_started = Instant::now();
        if prefill_input.is_some() {
            source
                .prefill_chunk_frame(&prefix, prefill_input.as_ref(), 0)
                .context("local state handoff source prefill failed")?;
        } else {
            source
                .prefill_chunked(&prefix)
                .context("local state handoff source prefill failed")?;
        }
        let source_prefill_ms = elapsed_ms(source_prefill_started);

        let source_export_started = Instant::now();
        let state_payload = export_local_state_payload(&mut source, args, prefix.len() as u64)
            .context("local state handoff source export failed")?;
        let source_export_ms = elapsed_ms(source_export_started);

        let source_decode_started = Instant::now();
        let (source_predicted_token, source_output) = source
            .decode_step_frame(continuation, decode_input.as_ref(), 0)
            .context("local state handoff source decode failed")?;
        let source_decode_ms = elapsed_ms(source_decode_started);

        let resident_state_bytes =
            measure_resident_state_bytes(&mut source, args, prefix.len() as u64)
                .context("local state handoff resident KV size measurement failed")?;

        (
            state_payload,
            resident_state_bytes,
            source_predicted_token,
            source_output,
            source_prefill_ms,
            source_export_ms,
            source_decode_ms,
        )
    };

    let (
        roundtrip_state_payload,
        restored_predicted_token,
        restored_output,
        restore_import_ms,
        restore_export_ms,
        restore_decode_ms,
    ) = {
        let restore_import_started = Instant::now();
        let mut restore = create_local_cache_hit_session(
            &model,
            args,
            &state_payload,
            prefix.len() as u64,
            &prefix,
        )
        .context("local state handoff restore import failed")?;
        let restore_import_ms = elapsed_ms(restore_import_started);

        let restore_export_started = Instant::now();
        let roundtrip_state_payload = if args.borrow_resident_hits
            && args.state_payload_kind == StatePayloadKind::ResidentKv
        {
            state_payload.clone()
        } else {
            export_local_state_payload(&mut restore, args, prefix.len() as u64)
                .context("local state handoff restore export failed")?
        };
        let restore_export_ms = elapsed_ms(restore_export_started);

        let restore_decode_started = Instant::now();
        let (restored_predicted_token, restored_output) = restore
            .decode_step_frame(continuation, decode_input.as_ref(), 0)
            .context("local state handoff restore decode failed")?;
        let restore_decode_ms = elapsed_ms(restore_decode_started);
        (
            roundtrip_state_payload,
            restored_predicted_token,
            restored_output,
            restore_import_ms,
            restore_export_ms,
            restore_decode_ms,
        )
    };
    let mut cache_hit_import_ms = vec![restore_import_ms];
    let mut cache_hit_decode_ms = vec![restore_decode_ms];
    let predicted_token_matches = source_predicted_token == restored_predicted_token;
    let restored_output_matches = source_output.payload == restored_output.payload;
    let mut cache_hit_matches = predicted_token_matches && restored_output_matches;
    for _ in 1..args.cache_hit_repeats {
        let import_started = Instant::now();
        let mut hit = create_local_cache_hit_session(
            &model,
            args,
            &state_payload,
            prefix.len() as u64,
            &prefix,
        )
        .context("local state handoff repeat import failed")?;
        cache_hit_import_ms.push(elapsed_ms(import_started));
        let decode_started = Instant::now();
        let (predicted, output) = hit
            .decode_step_frame(continuation, decode_input.as_ref(), 0)
            .context("local state handoff repeat decode failed")?;
        cache_hit_decode_ms.push(elapsed_ms(decode_started));
        cache_hit_matches &=
            predicted == source_predicted_token && output.payload == source_output.payload;
    }

    let mut stage_models = Vec::new();
    if let Some(input_resolution) = input_resolution {
        stage_models.push(input_resolution.report);
    }
    stage_models.push(stage_resolution.report);

    let roundtrip_state_matches = state_payload.same_payload(&roundtrip_state_payload);
    Ok(BinaryStateHandoffResult {
        prompt_token_count: prefix.len(),
        benchmark_prompt_token_count: prefix.len().saturating_add(1),
        benchmark_prompt_text,
        requested_prefix_token_count: args.prefix_token_count,
        stage_index: args.state_stage_index,
        layer_start: args.state_layer_start,
        layer_end: args.state_layer_end,
        include_embeddings,
        include_output,
        handoff_transport: "local-runtime",
        state_payload_kind: args.state_payload_kind,
        borrowed_resident_hits: args.borrow_resident_hits
            && args.state_payload_kind == StatePayloadKind::ResidentKv,
        cached_decoded_result_hits: false,
        activation_width,
        source_predicted_token,
        restored_predicted_token,
        state_bytes: state_payload.byte_len(),
        cache_storage_bytes: match args.state_payload_kind {
            StatePayloadKind::ResidentKv => resident_state_bytes,
            _ => Some(state_payload.byte_len()),
        },
        resident_state_bytes,
        roundtrip_state_bytes: roundtrip_state_payload.byte_len(),
        payload_digest: state_payload.digest_report(),
        tokenize_ms,
        source_prefill_ms,
        source_export_ms,
        source_decode_ms,
        restore_import_ms,
        restore_export_ms,
        restore_decode_ms,
        cache_hit_import_ms,
        cache_hit_decode_ms,
        matches: predicted_token_matches && restored_output_matches && cache_hit_matches,
        predicted_token_matches,
        roundtrip_state_matches,
        restored_output_matches: Some(restored_output_matches),
        cache_hit_matches,
        stage_models,
    })
}

#[allow(clippy::too_many_arguments)]
fn run_local_resident_slot_handoff(
    model: StageModel,
    args: &BinaryStateHandoffConfig,
    stage_resolution: StageModelResolution,
    input_resolution: Option<StageModelResolution>,
    prefix: Vec<i32>,
    continuation: i32,
    benchmark_prompt_text: String,
    prefill_input: Option<ActivationFrame>,
    decode_input: Option<ActivationFrame>,
    tokenize_ms: f64,
    activation_width: i32,
    include_embeddings: bool,
    include_output: bool,
) -> Result<BinaryStateHandoffResult> {
    let mut slot = model
        .create_session()
        .context("failed to create local resident slot session")?;
    let source_prefill_started = Instant::now();
    if prefill_input.is_some() {
        slot.prefill_chunk_frame(&prefix, prefill_input.as_ref(), 0)
            .context("local resident slot source prefill failed")?;
    } else {
        slot.prefill_chunked(&prefix)
            .context("local resident slot source prefill failed")?;
    }
    let source_prefill_ms = elapsed_ms(source_prefill_started);

    let resident_state_bytes = measure_resident_state_bytes(&mut slot, args, prefix.len() as u64)
        .context("local resident slot KV size measurement failed")?;
    let cache_seq_id = resident_cache_seq_id(args);
    slot.save_prefix(cache_seq_id, prefix.len() as u64)
        .context("local resident slot save prefix failed")?;
    let state_payload = LocalStatePayload::ResidentKv {
        cache_seq_id,
        token_count: prefix.len() as u64,
    };

    let source_export_ms = 0.0;
    let source_decode_started = Instant::now();
    let (source_predicted_token, source_output) = slot
        .decode_step_frame(continuation, decode_input.as_ref(), 0)
        .context("local resident slot source decode failed")?;
    let source_decode_ms = elapsed_ms(source_decode_started);
    drop(slot);

    let restore_import_started = Instant::now();
    let mut restore = model
        .create_session_from_resident_prefix(cache_seq_id, &prefix)
        .context("local resident slot restore borrow failed")?;
    let restore_import_ms = elapsed_ms(restore_import_started);
    let restore_export_ms = 0.0;
    let restore_decode_started = Instant::now();
    let (restored_predicted_token, restored_output) =
        if args.cache_decoded_result_hits && decode_input.is_none() {
            (
                restore
                    .sample_current(None)
                    .context("local resident slot restore sample failed")?,
                ActivationFrame {
                    desc: skippy_runtime::ActivationDesc {
                        version: 0,
                        dtype: skippy_runtime::RuntimeActivationDType::Unknown,
                        layout: skippy_runtime::RuntimeActivationLayout::Opaque,
                        producer_stage_index: args.state_stage_index as i32,
                        layer_start: args.state_layer_start as i32,
                        layer_end: args.state_layer_end as i32,
                        token_count: 0,
                        sequence_count: 0,
                        payload_bytes: 0,
                        flags: 0,
                    },
                    payload: Vec::new(),
                },
            )
        } else {
            restore
                .decode_step_frame(continuation, decode_input.as_ref(), 0)
                .context("local resident slot restore decode failed")?
        };
    let restore_decode_ms = elapsed_ms(restore_decode_started);
    drop(restore);

    let mut cache_hit_import_ms = vec![restore_import_ms];
    let mut cache_hit_decode_ms = vec![restore_decode_ms];
    let predicted_token_matches = source_predicted_token == restored_predicted_token;
    let restored_output_matches = source_output.payload == restored_output.payload;
    let mut cache_hit_matches = predicted_token_matches && restored_output_matches;
    for _ in 1..args.cache_hit_repeats {
        let import_started = Instant::now();
        let mut hit = model
            .create_session_from_resident_prefix(cache_seq_id, &prefix)
            .context("local resident slot repeat borrow failed")?;
        cache_hit_import_ms.push(elapsed_ms(import_started));
        let decode_started = Instant::now();
        let (predicted, output) = if args.cache_decoded_result_hits && decode_input.is_none() {
            (
                hit.sample_current(None)
                    .context("local resident slot repeat sample failed")?,
                ActivationFrame {
                    desc: skippy_runtime::ActivationDesc {
                        version: 0,
                        dtype: skippy_runtime::RuntimeActivationDType::Unknown,
                        layout: skippy_runtime::RuntimeActivationLayout::Opaque,
                        producer_stage_index: args.state_stage_index as i32,
                        layer_start: args.state_layer_start as i32,
                        layer_end: args.state_layer_end as i32,
                        token_count: 0,
                        sequence_count: 0,
                        payload_bytes: 0,
                        flags: 0,
                    },
                    payload: Vec::new(),
                },
            )
        } else {
            hit.decode_step_frame(continuation, decode_input.as_ref(), 0)
                .context("local resident slot repeat decode failed")?
        };
        cache_hit_decode_ms.push(elapsed_ms(decode_started));
        cache_hit_matches &=
            predicted == source_predicted_token && output.payload == source_output.payload;
    }

    let mut stage_models = Vec::new();
    if let Some(input_resolution) = input_resolution {
        stage_models.push(input_resolution.report);
    }
    stage_models.push(stage_resolution.report);

    Ok(BinaryStateHandoffResult {
        prompt_token_count: prefix.len(),
        benchmark_prompt_token_count: prefix.len().saturating_add(1),
        benchmark_prompt_text,
        requested_prefix_token_count: args.prefix_token_count,
        stage_index: args.state_stage_index,
        layer_start: args.state_layer_start,
        layer_end: args.state_layer_end,
        include_embeddings,
        include_output,
        handoff_transport: "local-runtime",
        state_payload_kind: args.state_payload_kind,
        borrowed_resident_hits: true,
        cached_decoded_result_hits: args.cache_decoded_result_hits,
        activation_width,
        source_predicted_token,
        restored_predicted_token,
        state_bytes: state_payload.byte_len(),
        cache_storage_bytes: resident_state_bytes,
        resident_state_bytes,
        roundtrip_state_bytes: state_payload.byte_len(),
        payload_digest: state_payload.digest_report(),
        tokenize_ms,
        source_prefill_ms,
        source_export_ms,
        source_decode_ms,
        restore_import_ms,
        restore_export_ms,
        restore_decode_ms,
        cache_hit_import_ms,
        cache_hit_decode_ms,
        matches: predicted_token_matches && restored_output_matches && cache_hit_matches,
        predicted_token_matches,
        roundtrip_state_matches: true,
        restored_output_matches: Some(restored_output_matches),
        cache_hit_matches,
        stage_models,
    })
}

fn create_local_cache_hit_session(
    model: &StageModel,
    args: &BinaryStateHandoffConfig,
    payload: &LocalStatePayload,
    token_count: u64,
    token_ids: &[i32],
) -> Result<StageSession> {
    if args.borrow_resident_hits && args.state_payload_kind == StatePayloadKind::ResidentKv {
        let LocalStatePayload::ResidentKv {
            cache_seq_id,
            token_count: payload_token_count,
        } = payload
        else {
            bail!("borrowed resident hit requested with non-resident payload");
        };
        if *payload_token_count != token_count {
            bail!(
                "resident KV payload token count {} does not match requested import token count {token_count}",
                payload_token_count
            );
        }
        return model
            .create_session_from_resident_prefix(*cache_seq_id, token_ids)
            .context("failed to borrow resident prefix session");
    }
    let mut session = model
        .create_session()
        .context("failed to create local state handoff cache-hit session")?;
    import_local_state_payload(&mut session, args, payload, token_count, token_ids)?;
    Ok(session)
}

fn export_local_state_payload(
    session: &mut StageSession,
    args: &BinaryStateHandoffConfig,
    token_count: u64,
) -> Result<LocalStatePayload> {
    match args.state_payload_kind {
        StatePayloadKind::ResidentKv => {
            let cache_seq_id = resident_cache_seq_id(args);
            session.save_prefix(cache_seq_id, token_count)?;
            Ok(LocalStatePayload::ResidentKv {
                cache_seq_id,
                token_count,
            })
        }
        StatePayloadKind::FullState => Ok(LocalStatePayload::FullState(
            session
                .export_full_state(args.state_layer_start as i32, args.state_layer_end as i32)?,
        )),
        StatePayloadKind::RecurrentOnly => Ok(LocalStatePayload::RecurrentOnly(
            session.export_recurrent_state()?,
        )),
        StatePayloadKind::KvRecurrent => {
            let page = session.export_kv_page(
                args.state_layer_start as i32,
                args.state_layer_end as i32,
                0,
                token_count,
            )?;
            let recurrent = session.export_recurrent_state()?;
            Ok(LocalStatePayload::KvRecurrent {
                kv_desc: page.desc,
                kv: page.payload,
                recurrent,
            })
        }
    }
}

fn measure_resident_state_bytes(
    session: &mut StageSession,
    args: &BinaryStateHandoffConfig,
    token_count: u64,
) -> Result<Option<usize>> {
    if args.state_payload_kind != StatePayloadKind::ResidentKv {
        return Ok(None);
    }
    let page = match session.export_kv_page(
        args.state_layer_start as i32,
        args.state_layer_end as i32,
        0,
        token_count,
    ) {
        Ok(page) => page,
        Err(_) => return Ok(None),
    };
    Ok(Some(
        usize::try_from(page.desc.payload_bytes).unwrap_or(page.payload.len()),
    ))
}

fn import_local_state_payload(
    session: &mut StageSession,
    args: &BinaryStateHandoffConfig,
    payload: &LocalStatePayload,
    token_count: u64,
    token_ids: &[i32],
) -> Result<()> {
    match payload {
        LocalStatePayload::ResidentKv {
            cache_seq_id,
            token_count: payload_token_count,
        } => {
            if *payload_token_count != token_count {
                bail!(
                    "resident KV payload token count {} does not match requested import token count {token_count}",
                    payload_token_count
                );
            }
            session.restore_prefix(*cache_seq_id, token_ids)
        }
        LocalStatePayload::FullState(bytes) => session.import_full_state(
            args.state_layer_start as i32,
            args.state_layer_end as i32,
            bytes,
        ),
        LocalStatePayload::RecurrentOnly(bytes) => {
            session.import_recurrent_state_for_token_count(bytes, token_count)
        }
        LocalStatePayload::KvRecurrent {
            kv_desc,
            kv,
            recurrent,
        } => {
            session.import_kv_page(kv_desc, kv)?;
            session.import_recurrent_state_for_token_count(recurrent, token_count)
        }
    }
}

fn state_payload_kind_name(kind: StatePayloadKind) -> &'static str {
    match kind {
        StatePayloadKind::ResidentKv => "resident-kv",
        StatePayloadKind::FullState => "full-state",
        StatePayloadKind::RecurrentOnly => "recurrent-only",
        StatePayloadKind::KvRecurrent => "kv-recurrent",
    }
}

fn resident_cache_seq_id(args: &BinaryStateHandoffConfig) -> i32 {
    let lane_count = effective_state_handoff_lane_count(args) as i32;
    lane_count.saturating_mul(2).saturating_add(1)
}

fn effective_state_handoff_lane_count(args: &BinaryStateHandoffConfig) -> u32 {
    args.runtime_lane_count
        .unwrap_or_else(|| args.cache_hit_repeats.saturating_add(2).max(2) as u32)
        .max(1)
}

fn payload_digest_report(
    payload_kind: &'static str,
    full_payload: &[u8],
    kv: Option<&[u8]>,
    recurrent: Option<&[u8]>,
) -> StatePayloadDigestReport {
    const BLOCK_SIZE_BYTES: usize = 1024 * 1024;
    let mut blocks = Vec::new();
    if full_payload.is_empty() {
        if let Some(kv) = kv {
            blocks.extend(block_digests("kv", kv, BLOCK_SIZE_BYTES));
        }
        if let Some(recurrent) = recurrent {
            blocks.extend(block_digests("recurrent", recurrent, BLOCK_SIZE_BYTES));
        }
    } else {
        blocks.extend(block_digests("payload", full_payload, BLOCK_SIZE_BYTES));
    }
    let unique_block_count = blocks
        .iter()
        .map(|block| block.sha256.as_str())
        .collect::<HashSet<_>>()
        .len();
    let block_count = blocks.len();
    StatePayloadDigestReport {
        payload_kind,
        payload_sha256: sha256_hex(full_payload),
        total_bytes: full_payload.len(),
        kv_bytes: kv.map_or(0, <[u8]>::len),
        kv_sha256: kv.map(sha256_hex),
        recurrent_bytes: recurrent.map_or(0, <[u8]>::len),
        recurrent_sha256: recurrent.map(sha256_hex),
        block_size_bytes: BLOCK_SIZE_BYTES,
        block_count,
        unique_block_count,
        duplicate_block_count: block_count.saturating_sub(unique_block_count),
        blocks,
    }
}

fn block_digests(
    component: &'static str,
    bytes: &[u8],
    block_size: usize,
) -> Vec<StatePayloadBlockDigestReport> {
    bytes
        .chunks(block_size)
        .enumerate()
        .map(|(index, chunk)| StatePayloadBlockDigestReport {
            component,
            index,
            offset: index.saturating_mul(block_size),
            bytes: chunk.len(),
            sha256: sha256_hex(chunk),
        })
        .collect()
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hex_sha256_finish(hasher)
}

fn hex_sha256_finish(hasher: Sha256) -> String {
    let digest = hasher.finalize();
    let mut out = String::with_capacity(digest.len() * 2);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(&mut out, "{byte:02x}");
    }
    out
}

fn should_filter_state_handoff_tensors(args: &BinaryStateHandoffConfig) -> bool {
    args.stage_load_mode != StageLoadMode::RuntimeSlice
        || args.state_layer_start != 0
        || args.state_layer_end != args.layer_end
}

fn build_state_handoff_inputs(
    args: &BinaryStateHandoffConfig,
    input_resolution: Option<&StageModelResolution>,
    prefix: &[i32],
    continuation: i32,
) -> Result<(Option<ActivationFrame>, Option<ActivationFrame>, i32)> {
    let Some(input_resolution) = input_resolution else {
        return Ok((None, None, args.activation_width));
    };
    let input_config = RuntimeConfig {
        stage_index: args.state_stage_index.saturating_sub(1),
        layer_start: 0,
        layer_end: args.state_layer_start,
        ctx_size: args.ctx_size,
        lane_count: 1,
        n_batch: None,
        n_ubatch: None,
        n_threads: None,
        n_threads_batch: None,
        n_gpu_layers: args.n_gpu_layers,
        selected_backend_device: None,
        load_mode: runtime_load_mode(args.stage_load_mode),
        projector_path: None,
        include_embeddings: true,
        include_output: false,
        filter_tensors_on_load: true,
        cache_type_k: GGML_TYPE_F16,
        cache_type_v: GGML_TYPE_F16,
        flash_attn_type: skippy_runtime::FlashAttentionType::Auto,
    };
    let input_model = StageModel::open(&input_resolution.path, &input_config)
        .context("failed to open state handoff input producer")?;
    let mut input_session = input_model
        .create_session()
        .context("failed to create state handoff input producer session")?;
    let prefill_input = input_session
        .prefill_chunk_frame(prefix, None, 0)
        .context("state handoff input producer failed to prefill prefix")?;
    let (_, decode_input) = input_session
        .decode_step_frame(continuation, None, 0)
        .context("state handoff input producer failed to decode continuation")?;
    let prefill_width = activation_width(&prefill_input)?;
    let decode_width = activation_width(&decode_input)?;
    if prefill_width != decode_width {
        bail!(
            "state handoff input width changed between prefill ({prefill_width}) and decode ({decode_width})"
        );
    }
    Ok((Some(prefill_input), Some(decode_input), prefill_width))
}

fn send_prefill_for_state_handoff(
    stream: &mut std::net::TcpStream,
    tokens: &[i32],
    input: Option<&ActivationFrame>,
    wire_dtype: skippy_protocol::binary::WireActivationDType,
    activation_width: i32,
) -> Result<()> {
    let token_count = i32::try_from(tokens.len()).context("too many prompt tokens")?;
    let mut state = StageStateHeader::new(WireMessageKind::PrefillEmbd, wire_dtype);
    state.prompt_token_count = token_count;
    state.current_token = tokens
        .last()
        .copied()
        .unwrap_or(skippy_protocol::binary::LLAMA_TOKEN_NULL);
    state.source_stage_index = input
        .map(|frame| frame.desc.producer_stage_index)
        .unwrap_or(-1);
    let activation = encode_handoff_activation(input, token_count, wire_dtype, activation_width)?;
    let message = StageWireMessage {
        kind: WireMessageKind::PrefillEmbd,
        pos_start: 0,
        token_count,
        state,
        request_id: 1,
        session_id: 1,
        sampling: None,
        chat_sampling_metadata: None,
        tokens: tokens.to_vec(),
        activation,
        raw_bytes: Vec::new(),
    };
    write_stage_message(&mut *stream, &message, wire_dtype)?;
    let reply = recv_reply(&mut *stream).context("receive prefill ACK")?;
    if reply.kind != WireReplyKind::Ack {
        bail!("expected prefill ACK, got {:?}", reply.kind);
    }
    Ok(())
}

fn export_state_over_binary(
    stream: &mut std::net::TcpStream,
    wire_dtype: skippy_protocol::binary::WireActivationDType,
    activation_width: i32,
    full_state: bool,
) -> Result<Vec<u8>> {
    let mut state = StageStateHeader::new(WireMessageKind::StateExport, wire_dtype);
    if full_state {
        state.flags |= state_flags::FULL_STATE;
    }
    let message = StageWireMessage {
        kind: WireMessageKind::StateExport,
        pos_start: 0,
        token_count: 0,
        state,
        request_id: 2,
        session_id: 1,
        sampling: None,
        chat_sampling_metadata: None,
        tokens: Vec::new(),
        activation: Vec::new(),
        raw_bytes: Vec::new(),
    };
    write_stage_message(&mut *stream, &message, wire_dtype)?;
    let reply =
        read_stage_message(&mut *stream, activation_width).context("receive state export")?;
    if reply.kind != WireMessageKind::StateImport {
        bail!("expected state import payload, got {:?}", reply.kind);
    }
    if reply.raw_bytes.is_empty() {
        bail!("state export returned an empty payload");
    }
    Ok(reply.raw_bytes)
}

fn import_state_over_binary(
    stream: &mut std::net::TcpStream,
    state_bytes: &[u8],
    wire_dtype: skippy_protocol::binary::WireActivationDType,
    full_state: bool,
) -> Result<()> {
    let token_count = i32::try_from(state_bytes.len()).context("state payload is too large")?;
    let mut state = StageStateHeader::new(WireMessageKind::StateImport, wire_dtype);
    if full_state {
        state.flags |= state_flags::FULL_STATE;
    }
    let message = StageWireMessage {
        kind: WireMessageKind::StateImport,
        pos_start: 0,
        token_count,
        state,
        request_id: 3,
        session_id: 1,
        sampling: None,
        chat_sampling_metadata: None,
        tokens: Vec::new(),
        activation: Vec::new(),
        raw_bytes: state_bytes.to_vec(),
    };
    write_stage_message(&mut *stream, &message, wire_dtype)?;
    let reply = recv_reply(&mut *stream).context("receive state import ACK")?;
    if reply.kind != WireReplyKind::Ack {
        bail!("expected state import ACK, got {:?}", reply.kind);
    }
    Ok(())
}

fn decode_for_state_handoff(
    stream: &mut std::net::TcpStream,
    token_id: i32,
    pos_start: usize,
    input: Option<&ActivationFrame>,
    wire_dtype: skippy_protocol::binary::WireActivationDType,
    activation_width: i32,
) -> Result<i32> {
    let mut state = StageStateHeader::new(WireMessageKind::DecodeEmbd, wire_dtype);
    state.prompt_token_count = i32::try_from(pos_start).context("prompt position exceeds i32")?;
    state.decode_step = 0;
    state.current_token = token_id;
    state.source_stage_index = input
        .map(|frame| frame.desc.producer_stage_index)
        .unwrap_or(-1);
    let activation = encode_handoff_activation(input, 1, wire_dtype, activation_width)?;
    let message = StageWireMessage {
        kind: WireMessageKind::DecodeEmbd,
        pos_start: i32::try_from(pos_start).context("decode position exceeds i32")?,
        token_count: 1,
        state,
        request_id: 4,
        session_id: 1,
        sampling: None,
        chat_sampling_metadata: None,
        tokens: vec![token_id],
        activation,
        raw_bytes: Vec::new(),
    };
    write_stage_message(&mut *stream, &message, wire_dtype)?;
    let reply = recv_reply(&mut *stream).context("receive decode reply")?;
    if reply.kind != WireReplyKind::PredictedToken {
        bail!("expected decode predicted token, got {:?}", reply.kind);
    }
    Ok(reply.predicted)
}

fn encode_handoff_activation(
    input: Option<&ActivationFrame>,
    token_count: i32,
    wire_dtype: skippy_protocol::binary::WireActivationDType,
    activation_width: i32,
) -> Result<Vec<u8>> {
    let Some(input) = input else {
        return Ok(Vec::new());
    };
    skippy_protocol::binary::encode_f32_activation_payload(
        wire_dtype,
        token_count,
        activation_width,
        &input.payload,
    )
    .context("failed to encode state handoff input activation")
}

fn baseline_report(result: FullModelResult) -> BaselineReport {
    BaselineReport {
        token_id: result.token_id,
        predicted_token: result.predicted_token,
    }
}

fn split_report(result: BinarySplitResult) -> SplitReport {
    SplitReport {
        token_id: result.token_id,
        predicted_token: result.predicted_token,
        activation_width: result.activation_width,
        wire_dtype: result.wire_dtype,
        boundary: BoundaryReport {
            producer_stage_index: result.boundary_producer_stage_index,
            layer_start: result.boundary_layer_start,
            layer_end: result.boundary_layer_end,
            token_count: result.boundary_token_count,
            payload_bytes: result.boundary_payload_bytes,
            wire_payload_bytes: result.boundary_wire_payload_bytes,
        },
    }
}

fn emit_report<T: Serialize>(report: &T, report_out: Option<&Path>) -> Result<()> {
    let json = serde_json::to_string_pretty(report)?;
    println!("{json}");
    if let Some(path) = report_out {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)
                    .with_context(|| format!("create report directory {}", parent.display()))?;
            }
        }
        fs::write(path, format!("{json}\n"))
            .with_context(|| format!("write correctness report {}", path.display()))?;
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct PackageStageSpec {
    topology_id: &'static str,
    stage_id: &'static str,
    stage_index: u32,
    layer_start: u32,
    layer_end: u32,
    include_embeddings: bool,
    include_output: bool,
}

struct StageModelResolution {
    path: PathBuf,
    report: StageModelReport,
}

#[derive(Debug, Deserialize)]
struct SliceManifest {
    stages: Vec<SliceManifestStage>,
}

#[derive(Debug, Deserialize)]
struct SliceManifestStage {
    stage_index: usize,
    path: String,
}

fn stage_model_resolution(
    baseline_model: &Path,
    stage_model: Option<&PathBuf>,
    stage_load_mode: StageLoadMode,
    model_identity: &ModelIdentity,
    spec: PackageStageSpec,
) -> Result<StageModelResolution> {
    let (path, package) = match stage_load_mode {
        StageLoadMode::RuntimeSlice => (baseline_model.to_path_buf(), None),
        StageLoadMode::ArtifactSlice => (artifact_stage_path(stage_model, spec.stage_index)?, None),
        StageLoadMode::LayerPackage => {
            let package_ref = layer_package_ref(baseline_model, stage_model);
            let package_ref = package_ref.to_string_lossy().into_owned();
            let materialized = materialize_layer_package_details(&PackageStageRequest {
                model_id: model_identity.model_id.clone(),
                topology_id: spec.topology_id.to_string(),
                package_ref: package_ref.clone(),
                stage_id: spec.stage_id.to_string(),
                layer_start: spec.layer_start,
                layer_end: spec.layer_end,
                include_embeddings: spec.include_embeddings,
                include_output: spec.include_output,
            })?;
            let path = materialized.output_path.clone();
            (path, Some(package_stage_report(package_ref, materialized)))
        }
    };
    Ok(StageModelResolution {
        report: StageModelReport {
            stage_index: spec.stage_index,
            layer_start: spec.layer_start,
            layer_end: spec.layer_end,
            load_mode: protocol_load_mode(stage_load_mode),
            model_path: path.to_string_lossy().into_owned(),
            package,
        },
        path,
    })
}

fn package_stage_report(
    package_ref: String,
    materialized: MaterializedPackage,
) -> PackageStageReport {
    PackageStageReport {
        package_ref,
        materialized_path: materialized.output_path.to_string_lossy().into_owned(),
        manifest_sha256: materialized.manifest_sha256,
        selected_parts: materialized
            .selected_parts
            .into_iter()
            .map(|part| PackagePartReport {
                role: part.role,
                layer_index: part.layer_index,
                path: part.path.to_string_lossy().into_owned(),
                sha256: part.sha256,
                artifact_bytes: part.artifact_bytes,
            })
            .collect(),
    }
}

fn stage_server_model_path(
    baseline_model: &Path,
    stage_model: Option<&PathBuf>,
    stage_load_mode: StageLoadMode,
    spec: PackageStageSpec,
) -> Result<String> {
    match stage_load_mode {
        StageLoadMode::RuntimeSlice => Ok(baseline_model.to_string_lossy().into_owned()),
        StageLoadMode::ArtifactSlice => Ok(artifact_stage_path(stage_model, spec.stage_index)?
            .to_string_lossy()
            .into_owned()),
        StageLoadMode::LayerPackage => Ok(layer_package_ref(baseline_model, stage_model)
            .to_string_lossy()
            .into_owned()),
    }
}

fn tokenizer_model_for_state_handoff(
    args: &BinaryStateHandoffConfig,
) -> Result<(PathBuf, RuntimeConfig)> {
    let (path, layer_end, load_mode, filter_tensors_on_load) = match args.stage_load_mode {
        StageLoadMode::LayerPackage => {
            let package_ref = layer_package_ref(&args.model, args.stage_model.as_ref());
            let package_ref_string = package_ref.to_string_lossy().into_owned();
            let materialized = materialize_layer_package_details(&PackageStageRequest {
                model_id: args.model_identity.model_id.clone(),
                topology_id: "correctness-tokenizer".to_string(),
                package_ref: package_ref_string,
                stage_id: "tokenizer".to_string(),
                layer_start: 0,
                layer_end: 1,
                include_embeddings: true,
                include_output: false,
            })?;
            (
                materialized.output_path,
                1,
                RuntimeLoadMode::LayerPackage,
                true,
            )
        }
        StageLoadMode::ArtifactSlice => {
            let path = artifact_stage_path(args.stage_model.as_ref(), 0)?;
            (path, args.layer_end, RuntimeLoadMode::ArtifactSlice, true)
        }
        StageLoadMode::RuntimeSlice => (
            args.model.clone(),
            args.layer_end,
            RuntimeLoadMode::RuntimeSlice,
            false,
        ),
    };

    Ok((
        path,
        RuntimeConfig {
            stage_index: 0,
            layer_start: 0,
            layer_end,
            ctx_size: args.ctx_size,
            lane_count: 1,
            n_batch: None,
            n_ubatch: None,
            n_threads: None,
            n_threads_batch: None,
            n_gpu_layers: args.n_gpu_layers,
            selected_backend_device: None,
            load_mode,
            projector_path: None,
            include_embeddings: true,
            include_output: false,
            filter_tensors_on_load,
            cache_type_k: GGML_TYPE_F16,
            cache_type_v: GGML_TYPE_F16,
            flash_attn_type: skippy_runtime::FlashAttentionType::Auto,
        },
    ))
}

fn layer_package_ref<'a>(baseline_model: &'a Path, stage_model: Option<&'a PathBuf>) -> &'a Path {
    stage_model.map(PathBuf::as_path).unwrap_or(baseline_model)
}

fn runtime_load_mode(stage_load_mode: StageLoadMode) -> RuntimeLoadMode {
    match stage_load_mode {
        StageLoadMode::RuntimeSlice => RuntimeLoadMode::RuntimeSlice,
        StageLoadMode::ArtifactSlice => RuntimeLoadMode::ArtifactSlice,
        StageLoadMode::LayerPackage => RuntimeLoadMode::LayerPackage,
    }
}

fn protocol_load_mode(stage_load_mode: StageLoadMode) -> &'static str {
    match stage_load_mode {
        StageLoadMode::RuntimeSlice => "runtime-slice",
        StageLoadMode::ArtifactSlice => "artifact-slice",
        StageLoadMode::LayerPackage => "layer-package",
    }
}

fn artifact_stage_path(stage_model: Option<&PathBuf>, stage_index: u32) -> Result<PathBuf> {
    let stage_model =
        stage_model.context("--stage-model is required when --stage-load-mode artifact-slice")?;
    if stage_model.is_dir() {
        let manifest_path = stage_model.join("slice-manifest.json");
        if manifest_path.is_file() {
            return artifact_stage_path_from_manifest(&manifest_path, stage_index);
        }
        return Ok(stage_model.join(format!("stage-{stage_index:03}.gguf")));
    }
    if stage_model
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name == "slice-manifest.json")
    {
        return artifact_stage_path_from_manifest(stage_model, stage_index);
    }
    if stage_index == 0 {
        return Ok(stage_model.to_path_buf());
    }
    bail!(
        "artifact-slice --stage-model must be a slice directory or slice-manifest.json for multi-stage correctness"
    )
}

fn artifact_stage_path_from_manifest(manifest_path: &Path, stage_index: u32) -> Result<PathBuf> {
    let manifest: SliceManifest = serde_json::from_str(
        &fs::read_to_string(manifest_path)
            .with_context(|| format!("read slice manifest {}", manifest_path.display()))?,
    )
    .with_context(|| format!("parse slice manifest {}", manifest_path.display()))?;
    let stage_index = stage_index as usize;
    let stage = manifest
        .stages
        .iter()
        .find(|stage| stage.stage_index == stage_index)
        .with_context(|| format!("slice manifest is missing stage {stage_index}"))?;
    let path = PathBuf::from(&stage.path);
    if path.is_absolute() {
        Ok(path)
    } else {
        Ok(manifest_path
            .parent()
            .unwrap_or_else(|| Path::new(""))
            .join(path))
    }
}

fn runtime_model_identity(args: &RuntimeArgs) -> Result<ModelIdentity> {
    if let Some(model_id) = args.model_id.as_ref() {
        let model_ref = ModelRef::parse(model_id)
            .with_context(|| format!("--model-id must be a model coordinate, got {model_id:?}"))?;
        return Ok(ModelIdentity::from_model_id(model_ref.display_id()));
    }

    if let Some(identity) = HfModelRepository::from_env()
        .ok()
        .and_then(|repository| repository.identity_for_path(&args.model))
    {
        return Ok(identity.to_model_identity());
    }

    bail!(
        "--model-id is required for local model paths that are not in the Hugging Face cache; pass a coordinate like org/repo:Q4_K_M"
    )
}

fn parse_chain_splits(spec: &str) -> Result<(u32, u32)> {
    let splits = parse_csv(spec)?
        .into_iter()
        .map(|value| {
            value
                .parse::<u32>()
                .with_context(|| format!("invalid split {value}"))
        })
        .collect::<Result<Vec<_>>>()?;
    if splits.len() != 2 {
        bail!("--splits for chain must contain exactly two comma-separated layer indexes");
    }
    Ok((splits[0], splits[1]))
}

fn parse_split_list(spec: &str) -> Result<Vec<u32>> {
    if let Some((start, end)) = spec.split_once("..") {
        let start = start.parse::<u32>().context("invalid split range start")?;
        let end = end.parse::<u32>().context("invalid split range end")?;
        if start >= end {
            bail!("split range start must be less than end");
        }
        return Ok((start..end).collect());
    }
    parse_csv(spec)?
        .into_iter()
        .map(|value| {
            value
                .parse::<u32>()
                .with_context(|| format!("invalid split {value}"))
        })
        .collect()
}

fn parse_csv(spec: &str) -> Result<Vec<String>> {
    let values = spec
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .collect::<Vec<_>>();
    if values.is_empty() {
        bail!("list must not be empty");
    }
    Ok(values)
}

fn configure_child_logs(command: &mut Command, child_logs: bool) {
    if child_logs {
        command.stdout(Stdio::inherit()).stderr(Stdio::inherit());
    } else {
        command.stdout(Stdio::null()).stderr(Stdio::null());
    }
}

fn ensure_matches(matches: bool, allow_mismatch: bool) -> Result<()> {
    if !matches && !allow_mismatch {
        bail!("staged execution did not match full-model baseline");
    }
    Ok(())
}

fn status(matches: bool) -> &'static str {
    if matches {
        "pass"
    } else {
        "fail"
    }
}

#[allow(dead_code)]
fn _assert_model_path(path: &Path) -> Result<()> {
    if !path.exists() {
        bail!("model path does not exist: {}", path.display());
    }
    Ok(())
}
