use std::{
    fs,
    net::SocketAddr,
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

use anyhow::{bail, Context, Result};
use model_artifact::ModelIdentity;
use model_hf::HfModelRepository;
use model_ref::ModelRef;
use serde::Deserialize;
use serde::Serialize;
use serde_json::json;
use skippy_protocol::binary::{
    recv_reply, write_stage_message, StageStateHeader, StageWireMessage, WireMessageKind,
    WireReplyKind,
};
use skippy_runtime::{
    package::{materialize_layer_package_details, MaterializedPackage, PackageStageRequest},
    RuntimeConfig, RuntimeLoadMode, StageModel, GGML_TYPE_F16,
};

use crate::{
    cli::{
        ChainArgs, DtypeMatrixArgs, RuntimeArgs, ServerArgs, SingleStepArgs, SplitScanArgs,
        StageLoadMode,
    },
    report::{
        BaselineReport, BoundaryReport, ChainReport, ChainStageReport, DtypeMatrixReport,
        PackagePartReport, PackageStageReport, SingleStepReport, SplitReport, SplitScanReport,
        StageModelReport,
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
            let package_ref = stage_model
                .context("--stage-model is required when --stage-load-mode layer-package")?;
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
        StageLoadMode::LayerPackage => Ok(stage_model
            .context("--stage-model is required when --stage-load-mode layer-package")?
            .to_string_lossy()
            .into_owned()),
    }
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
