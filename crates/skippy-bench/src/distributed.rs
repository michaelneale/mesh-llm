use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    net::TcpStream,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use anyhow::{anyhow, bail, Context, Result};
use model_artifact::ModelIdentity;
use model_ref::ModelRef;
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use skippy_protocol::binary::{
    recv_ready, recv_reply, write_stage_message, StageStateHeader, StageWireMessage,
    WireMessageKind, WireReplyKind,
};
use skippy_runtime::{
    package::{materialize_layer_package_details, PackageStageRequest},
    write_gguf_from_parts, RuntimeConfig, RuntimeLoadMode, StageModel,
};
use skippy_topology::{
    dense_attention_layers, infer_family_capability, plan_contiguous_with_splits, BoundaryDecision,
    NodeSpec, PlannerPolicy, TopologyPlanRequest, WireValidation,
};

use crate::{
    cli::{FocusedRuntimeArgs, FocusedRuntimeScenario, RunArgs, DEFAULT_RUN_MAX_NEW_TOKENS},
    model_identity::model_identity_for_path,
    support::{parse_wire_dtype, retry, ChildGuard},
};

struct DistributedRunOutcome {
    run_id: String,
    topology_id: String,
    model_id: String,
    model_identity: ModelIdentity,
    run_dir: PathBuf,
    plan_path: PathBuf,
    report_path: PathBuf,
    execute_remote: bool,
    stage_count: usize,
    hosts: Vec<String>,
    report_counts: Value,
    remote_status_path: Option<PathBuf>,
    driver_result_path: Option<PathBuf>,
    driver_report: Option<PromptDriverReport>,
    startup_elapsed: Option<Duration>,
    run_elapsed: Duration,
}

#[derive(Debug, Serialize)]
struct FocusedRuntimeReport {
    schema_version: u32,
    scenario: String,
    mode: String,
    run_id: String,
    topology_id: String,
    model_id: String,
    model_identity: ModelIdentity,
    stage_count: usize,
    hosts: Vec<String>,
    topology: FocusedRuntimeTopology,
    model: FocusedRuntimeModel,
    latency_ms: FocusedRuntimeLatency,
    throughput_tokens_per_second: FocusedRuntimeThroughput,
    token_counts: FocusedRuntimeTokenCounts,
    preset: FocusedRuntimePreset,
    summary: FocusedRuntimeSummary,
    outputs: FocusedRuntimeOutputs,
}

#[derive(Debug, Serialize)]
struct FocusedRuntimeSummary {
    startup_elapsed_ms: Option<u128>,
    run_elapsed_ms: u128,
    prompt_count: usize,
    max_new_tokens: usize,
    prompt_tokens_total: usize,
    generated_tokens_total: usize,
    elapsed_ms_p50: u128,
    elapsed_ms_p95: u128,
    ttft_ms_p50: u128,
    ttft_ms_p95: u128,
    decode_elapsed_ms_p50: u128,
    decode_elapsed_ms_p95: u128,
    total_tokens_per_second: f64,
    generated_tokens_per_second: f64,
}

#[derive(Debug, Serialize)]
struct FocusedRuntimeTopology {
    topology_id: String,
    stage_count: usize,
    hosts: Vec<String>,
}

#[derive(Debug, Serialize)]
struct FocusedRuntimeModel {
    model_id: String,
    model_identity: ModelIdentity,
}

#[derive(Debug, Serialize)]
struct FocusedRuntimeLatency {
    startup_elapsed_ms: Option<u128>,
    run_elapsed_ms: u128,
    elapsed_ms_p50: u128,
    elapsed_ms_p95: u128,
    ttft_ms_p50: u128,
    ttft_ms_p95: u128,
    decode_elapsed_ms_p50: u128,
    decode_elapsed_ms_p95: u128,
}

#[derive(Debug, Serialize)]
struct FocusedRuntimeThroughput {
    total: f64,
    generated: f64,
}

#[derive(Debug, Serialize)]
struct FocusedRuntimeTokenCounts {
    prompt_total: usize,
    generated_total: usize,
    prompt_count: usize,
    max_new_tokens: usize,
}

#[derive(Debug, Serialize)]
struct FocusedRuntimePreset {
    scenario: String,
    description: &'static str,
    prompt_limit: Option<usize>,
    max_new_tokens: usize,
    generated_prompt_corpus: Option<PathBuf>,
}

#[derive(Debug, Serialize)]
struct FocusedRuntimeOutputs {
    report: PathBuf,
    driver_result: Option<PathBuf>,
    deployment_plan: PathBuf,
    remote_status: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize)]
struct StageAssignment {
    stage_id: String,
    stage_index: u32,
    host: String,
    local: bool,
    layer_start: u32,
    layer_end: u32,
    bind_addr: String,
    endpoint: String,
    config_path: PathBuf,
    remote_config_path: String,
    remote_log_path: String,
    remote_pid_path: String,
    remote_exit_code_path: String,
    remote_model_path: Option<String>,
    local_materialized_model_path: Option<PathBuf>,
    local_shared_model_path: Option<PathBuf>,
    selected_package_files: Vec<String>,
}

#[derive(Debug, Serialize)]
struct DeploymentPlan {
    run_id: String,
    topology_id: String,
    model_id: String,
    model_identity: ModelIdentity,
    hosts: Vec<String>,
    stage_load_mode: String,
    remote_root: String,
    remote_roots: BTreeMap<String, String>,
    remote_shared_roots: BTreeMap<String, PathBuf>,
    endpoint_hosts: BTreeMap<String, String>,
    work_dir: PathBuf,
    metrics_http: String,
    metrics_otlp_grpc: String,
    stages: Vec<StageAssignment>,
    execute_remote: bool,
    keep_remote: bool,
    rsync_model_artifacts: bool,
}

#[derive(Debug, Clone, Serialize)]
struct RemoteStageStatus {
    stage_id: String,
    host: String,
    pid: Option<u32>,
    pid_alive: bool,
    log_ready: bool,
    protocol_ready: bool,
    exit_code: Option<i32>,
    log_tail: String,
    collected_log_path: Option<PathBuf>,
    terminated: bool,
}

#[derive(Debug, Serialize)]
struct PromptDriverReport {
    first_stage_endpoint: String,
    prompt_count: usize,
    max_new_tokens: usize,
    prefill_chunk_size: Option<usize>,
    prefill_chunk_threshold: Option<usize>,
    prefill_chunk_schedule: Option<String>,
    corpus: Option<PathBuf>,
    summary: PromptDriverSummary,
    results: Vec<PromptDriverResult>,
}

#[derive(Debug, Serialize)]
struct PromptDriverSummary {
    prompt_tokens_total: usize,
    generated_tokens_total: usize,
    elapsed_ms_total: u128,
    elapsed_ms_mean: f64,
    elapsed_ms_p50: u128,
    elapsed_ms_p95: u128,
    elapsed_ms_p99: u128,
    wire_elapsed_ms_mean: f64,
    wire_elapsed_ms_p50: u128,
    wire_elapsed_ms_p95: u128,
    wire_elapsed_ms_p99: u128,
    prefill_elapsed_ms_mean: f64,
    prefill_elapsed_ms_p50: u128,
    prefill_elapsed_ms_p95: u128,
    prefill_elapsed_ms_p99: u128,
    ttft_ms_mean: f64,
    ttft_ms_p50: u128,
    ttft_ms_p95: u128,
    ttft_ms_p99: u128,
    decode_elapsed_ms_mean: f64,
    decode_elapsed_ms_p50: u128,
    decode_elapsed_ms_p95: u128,
    decode_elapsed_ms_p99: u128,
    total_tokens_per_second: f64,
    generated_tokens_per_second: f64,
}

#[derive(Debug, Serialize)]
struct PromptDriverResult {
    prompt_id: Option<String>,
    category: Option<String>,
    prompt: String,
    token_ids: Vec<i32>,
    prefill_token_count: usize,
    prefill_chunk_count: usize,
    effective_prefill_chunk_size: Option<usize>,
    predicted_tokens: Vec<i32>,
    max_new_tokens: usize,
    elapsed_ms: u128,
    wire_elapsed_ms: u128,
    prefill_elapsed_ms: u128,
    ttft_ms: u128,
    decode_elapsed_ms: u128,
}

#[derive(Debug, Clone)]
struct PromptCase {
    prompt_id: Option<String>,
    category: Option<String>,
    prompt: String,
}

struct DriverTokenizer {
    model: StageModel,
    _materialized_model_path: Option<PathBuf>,
}

#[derive(Debug, Deserialize)]
struct CreateRunResponse {
    run_id: String,
}

#[derive(Debug, Deserialize)]
struct PackageManifest {
    model_id: String,
    source_model: PackageSourceModel,
    shared: PackageShared,
    layers: Vec<PackageLayer>,
}

#[derive(Debug, Deserialize)]
struct PackageSourceModel {
    repo: Option<String>,
    revision: Option<String>,
    primary_file: Option<String>,
    canonical_ref: Option<String>,
    distribution_id: Option<String>,
}

#[derive(Debug, Deserialize)]
struct PackageShared {
    metadata: PackageArtifact,
    embeddings: PackageArtifact,
    output: PackageArtifact,
}

#[derive(Debug, Deserialize)]
struct PackageArtifact {
    path: String,
}

#[derive(Debug, Deserialize)]
struct PackageLayer {
    layer_index: u32,
    path: String,
}

pub fn run_distributed(args: RunArgs) -> Result<()> {
    let outcome = run_distributed_collect(args)?;
    print_distributed_run_outcome(&outcome)
}

fn run_distributed_collect(args: RunArgs) -> Result<DistributedRunOutcome> {
    let run_started = Instant::now();
    let hosts = parse_hosts(&args.hosts)?;
    let ranges = parse_stage_ranges(&args.splits, args.layer_end)?;
    validate_distinct_stage_hosts(&hosts, ranges.len())?;
    validate_topology_plan(&args, &hosts, &ranges)?;
    validate_balanced_stage_ranges(&ranges)?;
    let run_id = args.run_id.clone().unwrap_or_else(generate_bench_run_id);
    let run_dir = args.work_dir.join(&run_id);
    let config_dir = run_dir.join("configs");
    fs::create_dir_all(&config_dir)
        .with_context(|| format!("create benchmark work dir {}", config_dir.display()))?;

    let metrics_http = format!("http://{}", args.metrics_http_addr);
    let db = args
        .db
        .clone()
        .unwrap_or_else(|| run_dir.join("metrics.sqlite"));
    let model_ref = model_ref_for_configs(&args)?;
    let fallback_model_identity =
        model_identity_for_path(&args.model_id, args.model_path.as_deref())?;
    let plan = build_deployment_plan(
        &args,
        &run_id,
        &hosts,
        &ranges,
        &config_dir,
        &model_ref,
        fallback_model_identity,
    )?;
    write_stage_configs(&args, &plan, &model_ref)?;
    write_json_file(&run_dir.join("deployment-plan.json"), &plan)?;

    let client = Client::builder()
        .timeout(Duration::from_secs(15))
        .build()
        .context("failed to build HTTP client")?;

    let mut metrics_command = Command::new(&args.metrics_server_bin);
    metrics_command.args([
        "serve",
        "--db",
        db.to_str().context("db path is not valid UTF-8")?,
        "--http-addr",
        &args.metrics_http_addr.to_string(),
        "--otlp-grpc-addr",
        &args.metrics_otlp_grpc_addr.to_string(),
    ]);
    configure_child_logs(&mut metrics_command, args.child_logs);
    let _metrics = ChildGuard::spawn(metrics_command)?;

    let run_config = json!({
        "run_id": run_id,
        "topology_id": args.topology_id,
        "model_id": plan.model_identity.model_id,
        "model_identity": plan.model_identity,
        "mode": "distributed-run",
        "hosts": hosts,
        "stage_load_mode": args.stage_load_mode,
        "stage_count": plan.stages.len(),
        "prompt_corpus": args.prompt_corpus.clone(),
        "prompt_limit": args.prompt_limit,
        "prefill_chunk_size": args.prefill_chunk_size,
        "prefill_chunk_threshold": args.prefill_chunk_threshold,
        "prefill_chunk_schedule": args.prefill_chunk_schedule,
        "max_new_tokens": effective_run_max_new_tokens(&args),
        "stage_max_inflight": args.stage_max_inflight,
        "stage_reply_credit_limit": args.stage_reply_credit_limit,
        "stage_async_prefill_forward": args.stage_async_prefill_forward,
        "stage_downstream_wire_delay_ms": args.stage_downstream_wire_delay_ms,
        "stage_downstream_wire_mbps": args.stage_downstream_wire_mbps,
        "stage_telemetry_queue_capacity": args.stage_telemetry_queue_capacity,
        "stage_telemetry_level": args.stage_telemetry_level,
        "stages": plan
            .stages
            .iter()
            .map(|stage| {
                json!({
                    "stage_id": stage.stage_id,
                    "stage_index": stage.stage_index,
                    "host": stage.host,
                    "layer_start": stage.layer_start,
                    "layer_end": stage.layer_end,
                    "bind_addr": stage.bind_addr,
                    "endpoint": stage.endpoint,
                })
            })
            .collect::<Vec<_>>(),
        "execute_remote": args.execute_remote,
        "keep_remote": args.keep_remote,
        "rsync_model_artifacts": args.rsync_model_artifacts,
    });
    retry(args.startup_timeout_secs, || {
        let response = client
            .post(format!("{metrics_http}/v1/runs"))
            .json(&run_config)
            .send()
            .and_then(|response| response.error_for_status())?
            .json::<CreateRunResponse>()?;
        if response.run_id == run_id {
            Ok(())
        } else {
            Err(anyhow!(
                "metrics-server returned unexpected run_id {}",
                response.run_id
            ))
        }
    })
    .context("metrics-server did not become ready")?;

    let mut protocol_ready = false;
    let mut startup_elapsed = None;
    let mut remote_sessions = Vec::new();
    let run_result = (|| -> Result<(Value, PathBuf, Option<PromptDriverReport>)> {
        let mut driver_result = None;
        if args.execute_remote {
            remote_sessions = execute_remote_plan(&args, &plan)?;
            wait_remote_readiness(&args, &plan)?;
            protocol_ready = true;
            startup_elapsed = Some(run_started.elapsed());
            let result = run_remote_prompt_driver(&args, &plan)?;
            driver_result = Some(result);
        }

        thread::sleep(Duration::from_secs(1));
        client
            .post(format!("{metrics_http}/v1/runs/{run_id}/finalize"))
            .send()
            .context("failed to finalize run")?
            .error_for_status()
            .context("metrics-server rejected finalize")?;
        let report: Value = client
            .get(format!("{metrics_http}/v1/runs/{run_id}/report.json"))
            .send()
            .context("failed to fetch report")?
            .error_for_status()
            .context("metrics-server rejected report fetch")?
            .json()
            .context("failed to parse report")?;

        let output = args
            .output
            .clone()
            .unwrap_or_else(|| run_dir.join("report.json"));
        write_json_file(&output, &report)?;
        if let Some(driver_result) = driver_result.as_ref() {
            write_json_file(&run_dir.join("driver-result.json"), driver_result)?;
        }
        Ok((report, output, driver_result))
    })();

    let mut remote_status_path = None;
    if args.execute_remote {
        let cleanup_statuses = collect_and_cleanup_remote(&args, &plan, &run_dir, protocol_ready)
            .context("collect remote logs and cleanup")?;
        let path = run_dir.join("remote-status.json");
        write_json_file(&path, &cleanup_statuses)?;
        remote_status_path = Some(path);
        if args.keep_remote {
            for session in remote_sessions.drain(..) {
                session.keep_alive();
            }
        }
    }

    let (report, output, driver_report) = run_result?;
    let driver_result_path = driver_report
        .as_ref()
        .map(|_| run_dir.join("driver-result.json"));

    Ok(DistributedRunOutcome {
        run_id,
        topology_id: plan.topology_id.clone(),
        model_id: plan.model_id.clone(),
        model_identity: plan.model_identity.clone(),
        run_dir: run_dir.clone(),
        plan_path: run_dir.join("deployment-plan.json"),
        report_path: output,
        execute_remote: args.execute_remote,
        stage_count: plan.stages.len(),
        hosts: plan.hosts.clone(),
        report_counts: report["counts"].clone(),
        remote_status_path,
        driver_result_path,
        driver_report,
        startup_elapsed,
        run_elapsed: run_started.elapsed(),
    })
}

fn print_distributed_run_outcome(outcome: &DistributedRunOutcome) -> Result<()> {
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "run_id": outcome.run_id.clone(),
            "model_identity": outcome.model_identity.clone(),
            "plan": outcome.plan_path.clone(),
            "report": outcome.report_path.clone(),
            "execute_remote": outcome.execute_remote,
            "stage_count": outcome.stage_count,
            "hosts": outcome.hosts.clone(),
            "report_counts": outcome.report_counts.clone(),
            "remote_status": outcome.remote_status_path.clone(),
            "driver_result": outcome.driver_result_path.clone(),
        }))?
    );

    Ok(())
}

pub fn focused_runtime(args: FocusedRuntimeArgs) -> Result<()> {
    let mut args = apply_focused_runtime_preset(args);
    validate_focused_runtime_args(&args)?;
    if args.schema_smoke {
        let report = focused_runtime_schema_smoke_report(&args)?;
        write_or_print_focused_runtime_report(&report, args.focused_output.as_deref())?;
        return Ok(());
    }

    let scenario = args.scenario;
    let focused_output = args.focused_output.clone();
    let preset = prepare_focused_runtime_inputs(&mut args)?;
    let outcome = run_distributed_collect(args.run)?;
    let report = focused_runtime_report_from_outcome(scenario, preset, &outcome)?;
    let output =
        focused_output.unwrap_or_else(|| outcome.run_dir.join("focused-runtime-report.json"));
    write_json_file(&output, &report)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn validate_focused_runtime_args(args: &FocusedRuntimeArgs) -> Result<()> {
    validate_focused_runtime_topology(&args.run)?;
    if args.schema_smoke {
        return Ok(());
    }
    if !args.run.execute_remote {
        bail!(
            "focused-runtime requires --execute-remote so driver timing fields are produced; use --schema-smoke for CI-only schema validation"
        );
    }
    Ok(())
}

fn apply_focused_runtime_preset(mut args: FocusedRuntimeArgs) -> FocusedRuntimeArgs {
    match args.scenario {
        FocusedRuntimeScenario::ColdStartup | FocusedRuntimeScenario::FirstToken => {
            if args.run.prompt_limit.is_none() {
                args.run.prompt_limit = Some(1);
            }
            if args.run.max_new_tokens.is_none() {
                args.run.max_new_tokens = Some(DEFAULT_RUN_MAX_NEW_TOKENS);
            }
        }
        FocusedRuntimeScenario::SteadyDecode => {
            if args.run.prompt_limit.is_none() {
                args.run.prompt_limit = Some(1);
            }
            if args.run.max_new_tokens.is_none() {
                args.run.max_new_tokens = Some(128_usize);
            }
        }
        FocusedRuntimeScenario::KvWarmReuse => {
            if args.run.prompt_limit.is_none() {
                args.run.prompt_limit = Some(2);
            }
            if args.run.max_new_tokens.is_none() {
                args.run.max_new_tokens = Some(16_usize);
            }
        }
    }
    args
}

fn effective_run_max_new_tokens(args: &RunArgs) -> usize {
    args.max_new_tokens.unwrap_or(DEFAULT_RUN_MAX_NEW_TOKENS)
}

fn validate_focused_runtime_topology(run: &RunArgs) -> Result<()> {
    let hosts = parse_hosts(&run.hosts)?;
    let ranges = parse_stage_ranges(&run.splits, run.layer_end)?;
    validate_distinct_stage_hosts(&hosts, ranges.len())?;
    validate_topology_plan(run, &hosts, &ranges)?;
    validate_balanced_stage_ranges(&ranges)?;
    Ok(())
}

fn prepare_focused_runtime_inputs(args: &mut FocusedRuntimeArgs) -> Result<FocusedRuntimePreset> {
    let mut generated_prompt_corpus = None;
    if matches!(args.scenario, FocusedRuntimeScenario::KvWarmReuse)
        && args.run.prompt_corpus.is_none()
        && args.run.prompt_token_ids.is_none()
    {
        let run_id = args
            .run
            .run_id
            .clone()
            .unwrap_or_else(generate_bench_run_id);
        args.run.run_id = Some(run_id.clone());
        let path = args
            .run
            .work_dir
            .join(&run_id)
            .join("focused-kv-warm-reuse-corpus.jsonl");
        let escaped_prompt = serde_json::to_string(&args.run.prompt)?;
        let corpus = format!(
            "{{\"id\":\"kv-warm-reuse-1\",\"category\":\"kv_warm_reuse\",\"prompt\":{escaped_prompt}}}\n{{\"id\":\"kv-warm-reuse-2\",\"category\":\"kv_warm_reuse\",\"prompt\":{escaped_prompt}}}\n"
        );
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).with_context(|| {
                format!(
                    "create focused KV warm-reuse corpus dir {}",
                    parent.display()
                )
            })?;
        }
        fs::write(&path, corpus)
            .with_context(|| format!("write focused KV warm-reuse corpus {}", path.display()))?;
        args.run.prompt_corpus = Some(path.clone());
        generated_prompt_corpus = Some(path);
    }

    Ok(FocusedRuntimePreset {
        scenario: args.scenario.as_str().to_string(),
        description: focused_runtime_preset_description(args.scenario),
        prompt_limit: args.run.prompt_limit,
        max_new_tokens: effective_run_max_new_tokens(&args.run),
        generated_prompt_corpus,
    })
}

fn focused_runtime_preset_description(scenario: FocusedRuntimeScenario) -> &'static str {
    match scenario {
        FocusedRuntimeScenario::ColdStartup => {
            "one-prompt run with a default one-token decode budget; report startup readiness separately from driver latency"
        }
        FocusedRuntimeScenario::FirstToken => {
            "one-prompt run focused on existing prompt-driver TTFT percentiles"
        }
        FocusedRuntimeScenario::SteadyDecode => {
            "one-prompt run with a larger default decode budget when max-new-tokens is otherwise left at the run default"
        }
        FocusedRuntimeScenario::KvWarmReuse => {
            "two identical prompts by default so the second request can exercise warm-prefix reuse where supported"
        }
    }
}

fn focused_runtime_report_from_outcome(
    scenario: FocusedRuntimeScenario,
    preset: FocusedRuntimePreset,
    outcome: &DistributedRunOutcome,
) -> Result<FocusedRuntimeReport> {
    let driver = outcome
        .driver_report
        .as_ref()
        .context("focused-runtime requires driver-result output from an executed staged run")?;
    let summary = focused_runtime_summary(driver, outcome.startup_elapsed, outcome.run_elapsed);
    Ok(FocusedRuntimeReport {
        schema_version: 1,
        scenario: scenario.as_str().to_string(),
        mode: "executed".to_string(),
        run_id: outcome.run_id.clone(),
        topology_id: outcome.topology_id.clone(),
        model_id: outcome.model_id.clone(),
        model_identity: outcome.model_identity.clone(),
        stage_count: outcome.stage_count,
        hosts: outcome.hosts.clone(),
        topology: focused_runtime_topology(
            &outcome.topology_id,
            outcome.stage_count,
            &outcome.hosts,
        ),
        model: focused_runtime_model(&outcome.model_id, &outcome.model_identity),
        latency_ms: focused_runtime_latency(&summary),
        throughput_tokens_per_second: focused_runtime_throughput(&summary),
        token_counts: focused_runtime_token_counts(&summary),
        preset,
        summary,
        outputs: FocusedRuntimeOutputs {
            report: outcome.report_path.clone(),
            driver_result: outcome.driver_result_path.clone(),
            deployment_plan: outcome.plan_path.clone(),
            remote_status: outcome.remote_status_path.clone(),
        },
    })
}

fn focused_runtime_summary(
    driver: &PromptDriverReport,
    startup_elapsed: Option<Duration>,
    run_elapsed: Duration,
) -> FocusedRuntimeSummary {
    FocusedRuntimeSummary {
        startup_elapsed_ms: startup_elapsed.map(|elapsed| elapsed.as_millis()),
        run_elapsed_ms: run_elapsed.as_millis(),
        prompt_count: driver.prompt_count,
        max_new_tokens: driver.max_new_tokens,
        prompt_tokens_total: driver.summary.prompt_tokens_total,
        generated_tokens_total: driver.summary.generated_tokens_total,
        elapsed_ms_p50: driver.summary.elapsed_ms_p50,
        elapsed_ms_p95: driver.summary.elapsed_ms_p95,
        ttft_ms_p50: driver.summary.ttft_ms_p50,
        ttft_ms_p95: driver.summary.ttft_ms_p95,
        decode_elapsed_ms_p50: driver.summary.decode_elapsed_ms_p50,
        decode_elapsed_ms_p95: driver.summary.decode_elapsed_ms_p95,
        total_tokens_per_second: driver.summary.total_tokens_per_second,
        generated_tokens_per_second: driver.summary.generated_tokens_per_second,
    }
}

fn focused_runtime_schema_smoke_report(args: &FocusedRuntimeArgs) -> Result<FocusedRuntimeReport> {
    let hosts = parse_hosts(&args.run.hosts)?;
    let ranges = parse_stage_ranges(&args.run.splits, args.run.layer_end)?;
    validate_distinct_stage_hosts(&hosts, ranges.len())?;
    validate_topology_plan(&args.run, &hosts, &ranges)?;
    validate_balanced_stage_ranges(&ranges)?;
    let stage_count = ranges.len();
    let prompt_count = args.run.prompt_limit.unwrap_or(1);
    let model_identity = ModelIdentity::from_model_id(args.run.model_id.clone());
    let summary = FocusedRuntimeSummary {
        startup_elapsed_ms: Some(0),
        run_elapsed_ms: 0,
        prompt_count,
        max_new_tokens: effective_run_max_new_tokens(&args.run),
        prompt_tokens_total: 8 * prompt_count,
        generated_tokens_total: effective_run_max_new_tokens(&args.run) * prompt_count,
        elapsed_ms_p50: 10,
        elapsed_ms_p95: 10,
        ttft_ms_p50: 5,
        ttft_ms_p95: 5,
        decode_elapsed_ms_p50: 5,
        decode_elapsed_ms_p95: 5,
        total_tokens_per_second: 900.0,
        generated_tokens_per_second: 100.0,
    };
    Ok(FocusedRuntimeReport {
        schema_version: 1,
        scenario: args.scenario.as_str().to_string(),
        mode: "schema-smoke".to_string(),
        run_id: args
            .run
            .run_id
            .clone()
            .unwrap_or_else(|| "focused-runtime-schema-smoke".to_string()),
        topology_id: args.run.topology_id.clone(),
        model_id: args.run.model_id.clone(),
        model_identity: model_identity.clone(),
        stage_count,
        hosts: hosts.clone(),
        topology: focused_runtime_topology(&args.run.topology_id, stage_count, &hosts),
        model: focused_runtime_model(&args.run.model_id, &model_identity),
        latency_ms: focused_runtime_latency(&summary),
        throughput_tokens_per_second: focused_runtime_throughput(&summary),
        token_counts: focused_runtime_token_counts(&summary),
        preset: FocusedRuntimePreset {
            scenario: args.scenario.as_str().to_string(),
            description: focused_runtime_preset_description(args.scenario),
            prompt_limit: args.run.prompt_limit,
            max_new_tokens: effective_run_max_new_tokens(&args.run),
            generated_prompt_corpus: None,
        },
        summary,
        outputs: FocusedRuntimeOutputs {
            report: PathBuf::from("schema-smoke-report.json"),
            driver_result: Some(PathBuf::from("schema-smoke-driver-result.json")),
            deployment_plan: PathBuf::from("schema-smoke-deployment-plan.json"),
            remote_status: None,
        },
    })
}

fn focused_runtime_topology(
    topology_id: &str,
    stage_count: usize,
    hosts: &[String],
) -> FocusedRuntimeTopology {
    FocusedRuntimeTopology {
        topology_id: topology_id.to_string(),
        stage_count,
        hosts: hosts.to_vec(),
    }
}

fn focused_runtime_model(model_id: &str, model_identity: &ModelIdentity) -> FocusedRuntimeModel {
    FocusedRuntimeModel {
        model_id: model_id.to_string(),
        model_identity: model_identity.clone(),
    }
}

fn focused_runtime_latency(summary: &FocusedRuntimeSummary) -> FocusedRuntimeLatency {
    FocusedRuntimeLatency {
        startup_elapsed_ms: summary.startup_elapsed_ms,
        run_elapsed_ms: summary.run_elapsed_ms,
        elapsed_ms_p50: summary.elapsed_ms_p50,
        elapsed_ms_p95: summary.elapsed_ms_p95,
        ttft_ms_p50: summary.ttft_ms_p50,
        ttft_ms_p95: summary.ttft_ms_p95,
        decode_elapsed_ms_p50: summary.decode_elapsed_ms_p50,
        decode_elapsed_ms_p95: summary.decode_elapsed_ms_p95,
    }
}

fn focused_runtime_throughput(summary: &FocusedRuntimeSummary) -> FocusedRuntimeThroughput {
    FocusedRuntimeThroughput {
        total: summary.total_tokens_per_second,
        generated: summary.generated_tokens_per_second,
    }
}

fn focused_runtime_token_counts(summary: &FocusedRuntimeSummary) -> FocusedRuntimeTokenCounts {
    FocusedRuntimeTokenCounts {
        prompt_total: summary.prompt_tokens_total,
        generated_total: summary.generated_tokens_total,
        prompt_count: summary.prompt_count,
        max_new_tokens: summary.max_new_tokens,
    }
}

fn write_or_print_focused_runtime_report(
    report: &FocusedRuntimeReport,
    output: Option<&Path>,
) -> Result<()> {
    if let Some(output) = output {
        write_json_file(output, report)?;
    }
    println!("{}", serde_json::to_string_pretty(report)?);
    Ok(())
}

fn validate_topology_plan(args: &RunArgs, hosts: &[String], ranges: &[(u32, u32)]) -> Result<()> {
    let identity = format!(
        "{} {} {}",
        args.model_id,
        args.model_path
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_default(),
        args.stage_model
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_default()
    );
    let activation_width =
        u32::try_from(args.activation_width).context("activation_width must be non-negative")?;
    let family = infer_family_capability(&identity, args.layer_end, activation_width);
    let request = TopologyPlanRequest {
        topology_id: args.topology_id.clone(),
        model_id: args.model_id.clone(),
        layers: dense_attention_layers(args.layer_end, 0),
        nodes: hosts
            .iter()
            .map(|host| NodeSpec {
                node_id: host.clone(),
                cached_slice_bytes: 0,
                vram_bytes: 0,
            })
            .collect(),
        family: family.clone(),
        policy: PlannerPolicy::default(),
    };
    let splits = split_boundaries_from_ranges(ranges);
    let plan = plan_contiguous_with_splits(&request, &splits).context("topology planner failed")?;

    if args.activation_wire_dtype.eq_ignore_ascii_case("q8") {
        match family.as_ref().map(|family| family.q8_wire_validation) {
            Some(WireValidation::Validated) => {}
            Some(WireValidation::Rejected) => {
                bail!("topology planner rejected q8 activation wire dtype for {}; use f16 or add a passing q8 correctness record", args.model_id);
            }
            Some(WireValidation::Untested) => {
                bail!("topology planner has no q8 validation for {}; use f16 until this family/split passes correctness", args.model_id);
            }
            None => {}
        }
    }

    let rejected = plan
        .boundaries
        .iter()
        .filter(|boundary| boundary.decision == BoundaryDecision::Rejected)
        .collect::<Vec<_>>();
    if !rejected.is_empty() {
        let reasons = rejected
            .iter()
            .map(|boundary| {
                format!(
                    "layer {}: {:?}: {}",
                    boundary.layer_boundary,
                    boundary.reason_codes,
                    boundary.messages.join("; ")
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        bail!("topology planner rejected split plan:\n{reasons}");
    }

    Ok(())
}

fn split_boundaries_from_ranges(ranges: &[(u32, u32)]) -> Vec<u32> {
    ranges
        .iter()
        .take(ranges.len().saturating_sub(1))
        .map(|(_, end)| *end)
        .collect()
}

fn build_deployment_plan(
    args: &RunArgs,
    run_id: &str,
    hosts: &[String],
    ranges: &[(u32, u32)],
    config_dir: &Path,
    model_ref: &str,
    model_identity: ModelIdentity,
) -> Result<DeploymentPlan> {
    let metrics_http = format!("http://{}", args.metrics_http_addr);
    let metrics_otlp = metrics_otlp_grpc_url(args);
    let remote_root_map = parse_remote_root_map(args.remote_root_map.as_deref())?;
    let remote_shared_root_map = parse_path_map(args.remote_shared_root_map.as_deref())?;
    let endpoint_host_map = parse_remote_root_map(args.endpoint_host_map.as_deref())?;
    let package_manifest = if args.stage_load_mode == "layer-package" {
        args.stage_model
            .as_ref()
            .filter(|path| path.is_dir())
            .map(|path| load_package_manifest(path))
            .transpose()?
    } else {
        None
    };
    let coordinator_materializes =
        coordinator_materializes_layer_package(args) && package_manifest.is_some();
    let mut stages = Vec::with_capacity(ranges.len());
    for (index, (layer_start, layer_end)) in ranges.iter().copied().enumerate() {
        let stage_id = format!("stage-{index}");
        let host = hosts[index % hosts.len()].clone();
        let local = args.execute_remote && index == 0;
        let port = args
            .first_stage_port
            .checked_add(u16::try_from(index).context("stage index exceeds u16")?)
            .context("stage port overflow")?;
        let endpoint_host = endpoint_host_map
            .get(&host)
            .map(String::as_str)
            .unwrap_or(&host);
        let bind_host = endpoint_host_map
            .get(&host)
            .map(String::as_str)
            .unwrap_or(&args.remote_bind_host);
        let bind_addr = format!("{bind_host}:{port}");
        let endpoint = format!("tcp://{endpoint_host}:{port}");
        let host_remote_root = remote_root_map
            .get(&host)
            .map(String::as_str)
            .unwrap_or(&args.remote_root);
        let remote_stage_dir = format!("{host_remote_root}/{run_id}/{stage_id}");
        let selected_package_files = package_manifest
            .as_ref()
            .map(|manifest| {
                selected_package_files(
                    manifest,
                    layer_start,
                    layer_end,
                    index == 0,
                    index + 1 == ranges.len(),
                )
            })
            .transpose()?
            .unwrap_or_default();
        let stage_cache_key = if coordinator_materializes {
            Some(stage_model_cache_key(
                args,
                &stage_id,
                layer_start,
                layer_end,
            ))
        } else {
            None
        };
        let local_materialized_model_path = stage_cache_key.as_ref().map(|key| {
            args.work_dir
                .join("model-cache")
                .join(key)
                .join("stage.gguf")
        });
        let remote_model_path = stage_cache_key.as_ref().map(|key| {
            format!(
                "{host_remote_root}/model-cache/{}/stage.gguf",
                key.display()
            )
        });
        let local_shared_model_path = if let Some(key) = stage_cache_key.as_ref() {
            remote_shared_root_map
                .get(&host)
                .map(|root| root.join("model-cache").join(key).join("stage.gguf"))
        } else {
            None
        };
        stages.push(StageAssignment {
            stage_id,
            stage_index: index as u32,
            host,
            local,
            layer_start,
            layer_end,
            bind_addr,
            endpoint,
            config_path: config_dir.join(format!("stage-{index}.json")),
            remote_config_path: format!("{remote_stage_dir}/stage.json"),
            remote_log_path: format!("{remote_stage_dir}/stage.log"),
            remote_pid_path: format!("{remote_stage_dir}/stage.pid"),
            remote_exit_code_path: format!("{remote_stage_dir}/stage.exit"),
            remote_model_path,
            local_materialized_model_path,
            local_shared_model_path,
            selected_package_files,
        });
    }

    let model_identity = package_manifest
        .as_ref()
        .map(model_identity_from_package_manifest)
        .transpose()?
        .unwrap_or(model_identity);
    let plan = DeploymentPlan {
        run_id: run_id.to_string(),
        topology_id: args.topology_id.clone(),
        model_id: model_identity.model_id.clone(),
        model_identity,
        hosts: hosts.to_vec(),
        stage_load_mode: args.stage_load_mode.clone(),
        remote_root: args.remote_root.clone(),
        remote_roots: remote_root_map,
        remote_shared_roots: remote_shared_root_map,
        endpoint_hosts: endpoint_host_map,
        work_dir: args.work_dir.clone(),
        metrics_http,
        metrics_otlp_grpc: metrics_otlp,
        stages,
        execute_remote: args.execute_remote,
        keep_remote: args.keep_remote,
        rsync_model_artifacts: args.rsync_model_artifacts,
    };

    let _ = model_ref;
    Ok(plan)
}

fn model_identity_from_package_manifest(manifest: &PackageManifest) -> Result<ModelIdentity> {
    let model_ref = ModelRef::parse(&manifest.model_id).with_context(|| {
        format!(
            "package manifest model_id must be a model coordinate, got {:?}",
            manifest.model_id
        )
    })?;
    Ok(ModelIdentity {
        model_id: model_ref.display_id(),
        source_repo: manifest.source_model.repo.clone(),
        source_revision: manifest.source_model.revision.clone(),
        source_file: manifest.source_model.primary_file.clone(),
        canonical_ref: manifest.source_model.canonical_ref.clone(),
        distribution_id: manifest.source_model.distribution_id.clone(),
        selector: model_ref.selector,
    })
}

fn write_stage_configs(args: &RunArgs, plan: &DeploymentPlan, model_ref: &str) -> Result<()> {
    for stage in &plan.stages {
        let stage_model_ref = if let Some(remote_model_path) = stage.remote_model_path.as_ref() {
            if stage.local {
                stage
                    .local_materialized_model_path
                    .as_ref()
                    .map(|path| path_string(path))
                    .unwrap_or_else(|| remote_model_path.clone())
            } else {
                remote_model_path.clone()
            }
        } else if args.execute_remote
            && args.rsync_model_artifacts
            && args.stage_load_mode == "layer-package"
        {
            format!("{}/package", remote_parent(&stage.remote_config_path)?)
        } else {
            model_ref.to_string()
        };
        let config_load_mode =
            if stage.remote_model_path.is_some() && args.stage_load_mode == "layer-package" {
                "artifact-slice"
            } else {
                args.stage_load_mode.as_str()
            };
        let upstream = if stage.stage_index == 0 {
            json!(null)
        } else {
            let previous = &plan.stages[stage.stage_index as usize - 1];
            json!({
                "stage_id": previous.stage_id,
                "stage_index": previous.stage_index,
                "endpoint": if stage.stage_index == 1 { "driver".to_string() } else { previous.endpoint.clone() }
            })
        };
        let downstream = plan
            .stages
            .get(stage.stage_index as usize + 1)
            .map(|next| {
                json!({
                    "stage_id": next.stage_id,
                    "stage_index": next.stage_index,
                    "endpoint": next.endpoint,
                })
            })
            .unwrap_or_else(|| json!(null));
        let config = json!({
            "run_id": plan.run_id,
            "topology_id": plan.topology_id,
            "model_id": plan.model_id,
            "model_path": stage_model_ref,
            "stage_id": stage.stage_id,
            "stage_index": stage.stage_index,
            "layer_start": stage.layer_start,
            "layer_end": stage.layer_end,
            "ctx_size": args.ctx_size,
            "n_gpu_layers": args.n_gpu_layers,
            "cache_type_k": args.cache_type_k,
            "cache_type_v": args.cache_type_v,
            "filter_tensors_on_load": config_load_mode != "runtime-slice",
            "load_mode": config_load_mode,
            "bind_addr": stage.bind_addr,
            "upstream": upstream,
            "downstream": downstream,
        });
        write_json_file(&stage.config_path, &config)?;
    }
    Ok(())
}

fn execute_remote_plan(args: &RunArgs, plan: &DeploymentPlan) -> Result<Vec<ChildGuard>> {
    let mut sessions = Vec::with_capacity(plan.stages.len());
    let mut started_stages = Vec::with_capacity(plan.stages.len());
    for stage in plan.stages.iter().rev() {
        if stage.local {
            prepare_local_stage(args, stage)?;
            let command = local_start_command(args, plan, stage);
            let mut local = Command::new("sh");
            local
                .arg("-c")
                .arg(command)
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null());
            sessions.push(ChildGuard::spawn(local).with_context(|| {
                format!("start local stage {} on {}", stage.stage_id, stage.host)
            })?);
        } else {
            let remote_stage_dir = remote_parent(&stage.remote_config_path)?;
            run_command(
                Command::new("ssh")
                    .arg(&stage.host)
                    .arg(format!("mkdir -p {remote_stage_dir}")),
            )
            .with_context(|| format!("create remote stage dir on {}", stage.host))?;

            run_command(
                Command::new("rsync")
                    .arg("-az")
                    .arg(&args.stage_server_bin)
                    .arg(format!("{}:{remote_stage_dir}/skippy-server", stage.host)),
            )
            .with_context(|| format!("rsync stage server to {}", stage.host))?;

            run_command(
                Command::new("rsync")
                    .arg("-az")
                    .arg(&stage.config_path)
                    .arg(format!("{}:{}", stage.host, stage.remote_config_path)),
            )
            .with_context(|| format!("rsync config to {}", stage.host))?;

            if args.rsync_model_artifacts {
                rsync_model_artifacts(args, stage)?;
            }

            let remote_bin = format!("{remote_stage_dir}/skippy-server");
            let command = remote_start_command(args, plan, stage, &remote_bin);
            let mut ssh = Command::new("ssh");
            ssh.arg(&stage.host)
                .arg(command)
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null());
            sessions
                .push(ChildGuard::spawn(ssh).with_context(|| {
                    format!("start stage {} on {}", stage.stage_id, stage.host)
                })?);
        }
        started_stages.push(stage);
        if let Err(error) = wait_stage_log_ready(stage, args.startup_timeout_secs)
            .with_context(|| format!("wait for {} on {} to listen", stage.stage_id, stage.host))
        {
            for started_stage in &started_stages {
                let pid = remote_pid(started_stage).ok().flatten();
                let _ = terminate_remote_stage(started_stage, pid);
            }
            return Err(error);
        }
    }
    Ok(sessions)
}

fn wait_stage_log_ready(stage: &StageAssignment, timeout_secs: u64) -> Result<()> {
    let attempts = timeout_secs.saturating_mul(2).max(1);
    for _ in 0..attempts {
        if remote_log_ready(stage).unwrap_or(false) {
            return Ok(());
        }
        if let Some(exit_code) = remote_exit_code(stage).ok().flatten() {
            let log_tail = remote_log_tail(stage)
                .unwrap_or_else(|error| format!("failed to read remote log tail: {error:#}"));
            bail!("stage exited before listening with code {exit_code}; log tail:\n{log_tail}");
        }
        thread::sleep(Duration::from_millis(500));
    }
    bail!(
        "stage did not report listening in {}",
        stage.remote_log_path
    )
}

fn prepare_local_stage(args: &RunArgs, stage: &StageAssignment) -> Result<()> {
    let local_stage_dir = remote_parent(&stage.remote_config_path)?;
    fs::create_dir_all(&local_stage_dir)
        .with_context(|| format!("create local stage dir {local_stage_dir}"))?;
    fs::copy(&stage.config_path, &stage.remote_config_path).with_context(|| {
        format!(
            "copy local stage config {} to {}",
            stage.config_path.display(),
            stage.remote_config_path
        )
    })?;
    if args.rsync_model_artifacts {
        if let (Some(stage_model), Some(local_model)) = (
            args.stage_model.as_ref(),
            stage.local_materialized_model_path.as_ref(),
        ) {
            materialize_stage_model_on_coordinator(stage_model, stage, local_model)?;
        }
    }
    Ok(())
}

fn local_start_command(args: &RunArgs, plan: &DeploymentPlan, stage: &StageAssignment) -> String {
    stage_start_wrapper(
        args,
        plan,
        stage,
        &path_string(&args.stage_server_bin),
        &stage.remote_config_path,
    )
}

fn remote_start_command(
    args: &RunArgs,
    plan: &DeploymentPlan,
    stage: &StageAssignment,
    remote_bin: &str,
) -> String {
    stage_start_wrapper(args, plan, stage, remote_bin, &stage.remote_config_path)
}

fn stage_start_wrapper(
    args: &RunArgs,
    plan: &DeploymentPlan,
    stage: &StageAssignment,
    bin: &str,
    config_path: &str,
) -> String {
    let stage_command = stage_server_command(args, plan, bin, config_path);
    let exit_path = shell_quote(&stage.remote_exit_code_path);
    let log_path = shell_quote(&stage.remote_log_path);
    let pid_path = shell_quote(&stage.remote_pid_path);
    let wrapper = format!(
        "trap 'kill \"$child\" 2>/dev/null || true; wait \"$child\" 2>/dev/null; status=$?; printf \"%s\\n\" \"$status\" > {exit_path}; exit \"$status\"' TERM INT HUP; {stage_command} > {log_path} 2>&1 & child=$!; printf \"%s\\n\" \"$child\" > {pid_path}; wait \"$child\"; status=$?; printf \"%s\\n\" \"$status\" > {exit_path}; exit \"$status\""
    );
    format!(
        "chmod +x {} && rm -f {} {} && sh -c {}",
        shell_quote(bin),
        shell_quote(&stage.remote_exit_code_path),
        shell_quote(&stage.remote_pid_path),
        shell_quote(&wrapper),
    )
}

fn stage_server_command(
    args: &RunArgs,
    plan: &DeploymentPlan,
    bin: &str,
    config_path: &str,
) -> String {
    let reply_credit_arg = args
        .stage_reply_credit_limit
        .map(|limit| format!(" --reply-credit-limit {limit}"))
        .unwrap_or_default();
    let async_prefill_forward_arg = if args.stage_async_prefill_forward {
        " --async-prefill-forward"
    } else {
        ""
    };
    let downstream_wire_mbps_arg = args
        .stage_downstream_wire_mbps
        .map(|mbps| format!(" --downstream-wire-mbps {mbps}"))
        .unwrap_or_default();
    format!(
        "{} serve-binary --config {} --activation-width {} --activation-wire-dtype {} --metrics-otlp-grpc {} --telemetry-queue-capacity {} --telemetry-level {} --max-inflight {}{}{} --downstream-wire-delay-ms {}{}",
        shell_quote(bin),
        shell_quote(config_path),
        args.activation_width,
        shell_quote(&args.activation_wire_dtype),
        shell_quote(&plan.metrics_otlp_grpc),
        args.stage_telemetry_queue_capacity,
        shell_quote(&args.stage_telemetry_level),
        args.stage_max_inflight,
        reply_credit_arg,
        async_prefill_forward_arg,
        args.stage_downstream_wire_delay_ms,
        downstream_wire_mbps_arg,
    )
}

fn wait_remote_readiness(args: &RunArgs, plan: &DeploymentPlan) -> Result<Vec<RemoteStageStatus>> {
    let attempts = args.startup_timeout_secs.saturating_mul(2).max(1);
    let mut last_statuses = Vec::new();
    for _ in 0..attempts {
        last_statuses = plan
            .stages
            .iter()
            .map(remote_stage_status)
            .collect::<Vec<_>>();
        if last_statuses
            .iter()
            .all(|status| status.pid_alive && status.log_ready)
        {
            match probe_remote_chain_readiness(args, plan) {
                Ok(()) => {
                    if let Some(first) = last_statuses.first_mut() {
                        first.protocol_ready = true;
                    }
                    return Ok(last_statuses);
                }
                Err(error) => {
                    if let Some(first) = last_statuses.first_mut() {
                        first.log_tail = format!(
                            "{}\nprotocol readiness probe failed: {error:#}",
                            first.log_tail
                        );
                    }
                }
            }
        }
        thread::sleep(Duration::from_millis(500));
    }

    Err(anyhow!(
        "remote stages did not become ready: {}",
        serde_json::to_string(&last_statuses)?
    ))
}

fn probe_remote_chain_readiness(args: &RunArgs, plan: &DeploymentPlan) -> Result<()> {
    let first = plan
        .stages
        .first()
        .context("deployment plan has no stages")?;
    let mut stream = connect_endpoint_ready(&first.endpoint, args.startup_timeout_secs)
        .with_context(|| format!("connect to first binary stage {}", first.endpoint))?;
    let wire_dtype = parse_wire_dtype(&args.activation_wire_dtype)?;
    write_stage_message(&mut stream, &StageWireMessage::stop(wire_dtype), wire_dtype)
        .context("send readiness stop frame")?;
    Ok(())
}

fn collect_and_cleanup_remote(
    args: &RunArgs,
    plan: &DeploymentPlan,
    run_dir: &Path,
    protocol_ready: bool,
) -> Result<Vec<RemoteStageStatus>> {
    let logs_dir = run_dir.join("logs");
    fs::create_dir_all(&logs_dir)
        .with_context(|| format!("create remote log directory {}", logs_dir.display()))?;
    let mut statuses = Vec::with_capacity(plan.stages.len());
    for stage in &plan.stages {
        let mut status = remote_stage_status(stage);
        status.protocol_ready = protocol_ready && stage.stage_index == 0;
        status.collected_log_path = collect_remote_log(stage, &logs_dir);
        if !args.keep_remote {
            status.terminated = terminate_remote_stage(stage, status.pid).is_ok();
            let _ = wait_remote_exit_code(stage, Duration::from_secs(5));
            let exit_code = remote_exit_code(stage).ok().flatten();
            if status.terminated {
                status.pid_alive = remote_pid_alive_opt(stage, status.pid).unwrap_or(false);
                status.exit_code = exit_code;
            }
        }
        statuses.push(status);
    }
    Ok(statuses)
}

fn run_remote_prompt_driver(args: &RunArgs, plan: &DeploymentPlan) -> Result<PromptDriverReport> {
    let first = plan
        .stages
        .first()
        .context("deployment plan has no stages")?;
    let wire_dtype = parse_wire_dtype(&args.activation_wire_dtype)?;
    let prompt_cases = prompt_cases(args)?;
    if prompt_cases.is_empty() {
        bail!("prompt corpus is empty");
    }
    if args.prompt_corpus.is_some() && args.prompt_token_ids.is_some() {
        bail!("--prompt-token-ids cannot be used with --prompt-corpus");
    }
    let tokenizer = if args.prompt_token_ids.is_some() {
        None
    } else {
        Some(DriverTokenizer::open(args, plan)?)
    };

    let mut results = Vec::with_capacity(prompt_cases.len());
    for (index, prompt_case) in prompt_cases.iter().enumerate() {
        let started = Instant::now();
        let token_ids = if let Some(token_ids) = args.prompt_token_ids.as_ref() {
            parse_prompt_token_ids(token_ids)?
        } else {
            tokenizer
                .as_ref()
                .expect("tokenizer is present without explicit prompt tokens")
                .tokenize(&prompt_case.prompt)?
        };
        let mut result =
            run_remote_prompt_case(args, first, wire_dtype, prompt_case, token_ids, index)?;
        result.elapsed_ms = started.elapsed().as_millis();
        results.push(result);
    }

    Ok(PromptDriverReport {
        first_stage_endpoint: first.endpoint.clone(),
        prompt_count: results.len(),
        max_new_tokens: effective_run_max_new_tokens(args),
        prefill_chunk_size: args.prefill_chunk_size,
        prefill_chunk_threshold: args.prefill_chunk_threshold,
        prefill_chunk_schedule: args.prefill_chunk_schedule.clone(),
        corpus: args.prompt_corpus.clone(),
        summary: prompt_driver_summary(&results),
        results,
    })
}

fn prompt_driver_summary(results: &[PromptDriverResult]) -> PromptDriverSummary {
    let prompt_tokens_total = results.iter().map(|result| result.token_ids.len()).sum();
    let generated_tokens_total = results
        .iter()
        .map(|result| result.predicted_tokens.len())
        .sum();
    let elapsed_ms_total = results.iter().map(|result| result.elapsed_ms).sum();
    let elapsed_seconds = elapsed_ms_total as f64 / 1000.0;
    PromptDriverSummary {
        prompt_tokens_total,
        generated_tokens_total,
        elapsed_ms_total,
        elapsed_ms_mean: if results.is_empty() {
            0.0
        } else {
            elapsed_ms_total as f64 / results.len() as f64
        },
        elapsed_ms_p50: percentile_ms(results, 0.50),
        elapsed_ms_p95: percentile_ms(results, 0.95),
        elapsed_ms_p99: percentile_ms(results, 0.99),
        wire_elapsed_ms_mean: mean_ms(results, |result| result.wire_elapsed_ms),
        wire_elapsed_ms_p50: percentile_ms_by(results, 0.50, |result| result.wire_elapsed_ms),
        wire_elapsed_ms_p95: percentile_ms_by(results, 0.95, |result| result.wire_elapsed_ms),
        wire_elapsed_ms_p99: percentile_ms_by(results, 0.99, |result| result.wire_elapsed_ms),
        prefill_elapsed_ms_mean: mean_ms(results, |result| result.prefill_elapsed_ms),
        prefill_elapsed_ms_p50: percentile_ms_by(results, 0.50, |result| result.prefill_elapsed_ms),
        prefill_elapsed_ms_p95: percentile_ms_by(results, 0.95, |result| result.prefill_elapsed_ms),
        prefill_elapsed_ms_p99: percentile_ms_by(results, 0.99, |result| result.prefill_elapsed_ms),
        ttft_ms_mean: mean_ms(results, |result| result.ttft_ms),
        ttft_ms_p50: percentile_ms_by(results, 0.50, |result| result.ttft_ms),
        ttft_ms_p95: percentile_ms_by(results, 0.95, |result| result.ttft_ms),
        ttft_ms_p99: percentile_ms_by(results, 0.99, |result| result.ttft_ms),
        decode_elapsed_ms_mean: mean_ms(results, |result| result.decode_elapsed_ms),
        decode_elapsed_ms_p50: percentile_ms_by(results, 0.50, |result| result.decode_elapsed_ms),
        decode_elapsed_ms_p95: percentile_ms_by(results, 0.95, |result| result.decode_elapsed_ms),
        decode_elapsed_ms_p99: percentile_ms_by(results, 0.99, |result| result.decode_elapsed_ms),
        total_tokens_per_second: if elapsed_seconds > 0.0 {
            (prompt_tokens_total + generated_tokens_total) as f64 / elapsed_seconds
        } else {
            0.0
        },
        generated_tokens_per_second: if elapsed_seconds > 0.0 {
            generated_tokens_total as f64 / elapsed_seconds
        } else {
            0.0
        },
    }
}

fn percentile_ms(results: &[PromptDriverResult], percentile: f64) -> u128 {
    percentile_ms_by(results, percentile, |result| result.elapsed_ms)
}

fn percentile_ms_by(
    results: &[PromptDriverResult],
    percentile: f64,
    value: impl Fn(&PromptDriverResult) -> u128,
) -> u128 {
    if results.is_empty() {
        return 0;
    }
    let mut values = results.iter().map(value).collect::<Vec<_>>();
    values.sort_unstable();
    let rank = ((values.len() - 1) as f64 * percentile).ceil() as usize;
    values[rank.min(values.len() - 1)]
}

fn mean_ms(results: &[PromptDriverResult], value: impl Fn(&PromptDriverResult) -> u128) -> f64 {
    if results.is_empty() {
        return 0.0;
    }
    results.iter().map(value).sum::<u128>() as f64 / results.len() as f64
}

fn run_remote_prompt_case(
    args: &RunArgs,
    first: &StageAssignment,
    wire_dtype: skippy_protocol::binary::WireActivationDType,
    prompt_case: &PromptCase,
    token_ids: Vec<i32>,
    prompt_index: usize,
) -> Result<PromptDriverResult> {
    if token_ids.is_empty() {
        bail!("prompt produced no tokens");
    }

    let mut stream = connect_endpoint_ready(&first.endpoint, args.startup_timeout_secs)
        .with_context(|| {
            format!(
                "connect prompt {prompt_index} to first binary stage {}",
                first.endpoint
            )
        })?;

    let wire_started = Instant::now();
    let request_id = 10_000_u64 + prompt_index as u64;
    let session_id = 20_000_u64 + prompt_index as u64;
    let prefill_token_count = token_ids.len().saturating_sub(1);
    let mut prefill_chunk_count = 0usize;
    let mut effective_chunk_size = None;
    let prefill_started = Instant::now();
    if prefill_token_count > 0 {
        let prefill_tokens = token_ids[..prefill_token_count].to_vec();
        let chunk_size = effective_prefill_chunk_size(args, prefill_tokens.len());
        effective_chunk_size = Some(chunk_size);
        for (chunk_index, chunk) in prefill_tokens.chunks(chunk_size).enumerate() {
            prefill_chunk_count += 1;
            let pos_start = chunk_index
                .checked_mul(chunk_size)
                .context("prefill chunk position overflow")?;
            send_prefill_chunk(
                &mut stream,
                wire_dtype,
                PrefillChunk {
                    prompt_index,
                    request_id,
                    session_id,
                    pos_start,
                    prefill_token_count,
                    tokens: chunk,
                },
            )?;
        }
    }
    let prefill_elapsed_ms = prefill_started.elapsed().as_millis();

    let max_new_tokens = effective_run_max_new_tokens(args);
    let mut predicted_tokens = Vec::with_capacity(max_new_tokens);
    let mut current = *token_ids.last().expect("checked non-empty tokens");
    let decode_started = Instant::now();
    let mut ttft_ms = 0;
    for decode_step in 0..max_new_tokens {
        let mut state = StageStateHeader::new(WireMessageKind::DecodeEmbd, wire_dtype);
        state.seq_id = i32::try_from(prompt_index).context("prompt index exceeds i32")?;
        state.prompt_token_count =
            i32::try_from(token_ids.len()).context("prompt token count exceeds i32")?;
        state.decode_step = i32::try_from(decode_step).context("decode step exceeds i32")?;
        state.current_token = current;
        state.source_stage_index = -1;
        let message = StageWireMessage {
            kind: WireMessageKind::DecodeEmbd,
            pos_start: i32::try_from(prefill_token_count + decode_step)
                .context("decode position exceeds i32")?,
            token_count: 1,
            state,
            request_id,
            session_id,
            sampling: None,
            tokens: vec![current],
            activation: Vec::new(),
            raw_bytes: Vec::new(),
        };
        write_stage_message(&mut stream, &message, wire_dtype).with_context(|| {
            format!("send remote decode step {decode_step} for prompt {prompt_index}")
        })?;
        let reply = recv_reply(&mut stream).with_context(|| {
            format!("receive remote decode step {decode_step} reply for prompt {prompt_index}")
        })?;
        if reply.kind != WireReplyKind::PredictedToken {
            bail!("expected predicted-token reply, got {:?}", reply.kind);
        }
        if decode_step == 0 {
            ttft_ms = wire_started.elapsed().as_millis();
        }
        current = reply.predicted;
        predicted_tokens.push(reply.predicted);
    }
    let decode_elapsed_ms = decode_started.elapsed().as_millis();

    write_stage_message(
        &mut stream,
        &StageWireMessage::stop_with_identity(wire_dtype, request_id, session_id),
        wire_dtype,
    )
    .context("send remote stop")?;
    let wire_elapsed_ms = wire_started.elapsed().as_millis();

    Ok(PromptDriverResult {
        prompt_id: prompt_case.prompt_id.clone(),
        category: prompt_case.category.clone(),
        prompt: prompt_case.prompt.clone(),
        token_ids,
        prefill_token_count,
        prefill_chunk_count,
        effective_prefill_chunk_size: effective_chunk_size,
        predicted_tokens,
        max_new_tokens,
        elapsed_ms: 0,
        wire_elapsed_ms,
        prefill_elapsed_ms,
        ttft_ms,
        decode_elapsed_ms,
    })
}

fn effective_prefill_chunk_size(args: &RunArgs, prefill_token_count: usize) -> usize {
    let Some(chunk_size) = args.prefill_chunk_size else {
        return prefill_token_count.max(1);
    };
    if args
        .prefill_chunk_threshold
        .is_some_and(|threshold| prefill_token_count <= threshold)
    {
        return prefill_token_count.max(1);
    }
    adaptive_prefill_chunk_size(args, prefill_token_count)
        .unwrap_or(chunk_size)
        .max(1)
}

fn adaptive_prefill_chunk_size(args: &RunArgs, prefill_token_count: usize) -> Option<usize> {
    let schedule = args.prefill_chunk_schedule.as_deref()?;
    let mut selected = None;
    for entry in schedule.split(',') {
        let entry = entry.trim();
        if entry.is_empty() {
            continue;
        }
        let (min_tokens, chunk_size) = entry.split_once(':')?;
        let min_tokens = min_tokens.trim().parse::<usize>().ok()?;
        let chunk_size = chunk_size.trim().parse::<usize>().ok()?;
        if prefill_token_count >= min_tokens {
            selected = Some(match selected {
                Some((selected_min, selected_chunk)) if selected_min > min_tokens => {
                    (selected_min, selected_chunk)
                }
                _ => (min_tokens, chunk_size),
            });
        }
    }
    selected.map(|(_, chunk_size)| chunk_size)
}

struct PrefillChunk<'a> {
    prompt_index: usize,
    request_id: u64,
    session_id: u64,
    pos_start: usize,
    prefill_token_count: usize,
    tokens: &'a [i32],
}

fn send_prefill_chunk(
    stream: &mut TcpStream,
    wire_dtype: skippy_protocol::binary::WireActivationDType,
    chunk: PrefillChunk<'_>,
) -> Result<()> {
    let mut state = StageStateHeader::new(WireMessageKind::PrefillEmbd, wire_dtype);
    state.seq_id = i32::try_from(chunk.prompt_index).context("prompt index exceeds i32")?;
    state.prompt_token_count =
        i32::try_from(chunk.prefill_token_count).context("prompt token count exceeds i32")?;
    state.current_token = *chunk.tokens.last().context("prefill chunk is empty")?;
    state.source_stage_index = -1;
    let message = StageWireMessage {
        kind: WireMessageKind::PrefillEmbd,
        pos_start: i32::try_from(chunk.pos_start).context("prefill chunk position exceeds i32")?,
        token_count: i32::try_from(chunk.tokens.len())
            .context("prefill token count exceeds i32")?,
        state,
        request_id: chunk.request_id,
        session_id: chunk.session_id,
        sampling: None,
        tokens: chunk.tokens.to_vec(),
        activation: Vec::new(),
        raw_bytes: Vec::new(),
    };
    write_stage_message(&mut *stream, &message, wire_dtype).with_context(|| {
        format!(
            "send remote prefill chunk for prompt {}",
            chunk.prompt_index
        )
    })?;
    let reply = recv_reply(&mut *stream).with_context(|| {
        format!(
            "receive remote prefill chunk ACK for prompt {}",
            chunk.prompt_index
        )
    })?;
    if reply.kind != WireReplyKind::Ack {
        bail!("expected prefill ACK, got {:?}", reply.kind);
    }
    Ok(())
}

impl DriverTokenizer {
    fn open(args: &RunArgs, plan: &DeploymentPlan) -> Result<Self> {
        let first = plan
            .stages
            .first()
            .context("deployment plan has no stages")?;
        let mut materialized_model_path = None;
        let (model_path, load_mode) = if let Some(model_path) = args.model_path.as_ref() {
            (model_path.clone(), RuntimeLoadMode::RuntimeSlice)
        } else if args.stage_load_mode == "layer-package" {
            let missing_model =
                "--model-path is required unless --stage-model is a local layer-package directory";
            let stage_model = args
                .stage_model
                .as_ref()
                .filter(|path| path.is_dir())
                .context(missing_model)?;
            let package = materialize_layer_package_details(&PackageStageRequest {
                model_id: args.model_id.clone(),
                topology_id: args.topology_id.clone(),
                package_ref: path_string(stage_model),
                stage_id: "driver-tokenizer".to_string(),
                layer_start: first.layer_start,
                layer_end: first.layer_end,
                include_embeddings: true,
                include_output: plan.stages.len() == 1,
            })
            .context("materialize local layer-package tokenizer model")?;
            materialized_model_path = Some(package.output_path.clone());
            (package.output_path, RuntimeLoadMode::LayerPackage)
        } else {
            bail!("--model-path or a local layer-package --stage-model is required for prompt tokenization");
        };

        let model = StageModel::open(
            &model_path,
            &RuntimeConfig {
                stage_index: 0,
                layer_start: first.layer_start,
                layer_end: first.layer_end,
                ctx_size: args.ctx_size,
                n_gpu_layers: args.n_gpu_layers,
                selected_backend_device: None,
                cache_type_k: skippy_runtime::GGML_TYPE_F16,
                cache_type_v: skippy_runtime::GGML_TYPE_F16,
                load_mode,
                projector_path: None,
                include_embeddings: true,
                include_output: plan.stages.len() == 1,
                filter_tensors_on_load: args.stage_load_mode != "runtime-slice",
            },
        )
        .with_context(|| format!("open tokenizer model {}", model_path.display()))?;
        Ok(Self {
            model,
            _materialized_model_path: materialized_model_path,
        })
    }

    fn tokenize(&self, prompt: &str) -> Result<Vec<i32>> {
        self.model
            .tokenize(prompt, true)
            .with_context(|| format!("tokenize prompt {prompt:?}"))
    }
}

fn prompt_cases(args: &RunArgs) -> Result<Vec<PromptCase>> {
    if let Some(path) = args.prompt_corpus.as_ref() {
        let text = fs::read_to_string(path)
            .with_context(|| format!("read prompt corpus {}", path.display()))?;
        let mut cases = Vec::new();
        for (line_index, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let value: Value = serde_json::from_str(line).with_context(|| {
                format!(
                    "parse prompt corpus JSONL line {} in {}",
                    line_index + 1,
                    path.display()
                )
            })?;
            cases.push(prompt_case_from_value(&value).with_context(|| {
                format!(
                    "read prompt corpus line {} in {}",
                    line_index + 1,
                    path.display()
                )
            })?);
            if args.prompt_limit.is_some_and(|limit| cases.len() >= limit) {
                break;
            }
        }
        Ok(cases)
    } else {
        Ok(vec![PromptCase {
            prompt_id: None,
            category: None,
            prompt: args.prompt.clone(),
        }])
    }
}

fn prompt_case_from_value(value: &Value) -> Result<PromptCase> {
    let prompt_id = value
        .get("id")
        .or_else(|| value.get("prompt_id"))
        .and_then(|value| {
            value
                .as_str()
                .map(ToOwned::to_owned)
                .or_else(|| value.as_i64().map(|id| id.to_string()))
        });
    let category = value
        .get("category")
        .and_then(Value::as_str)
        .map(ToOwned::to_owned);
    let prompt = if let Some(prompt) = value.get("prompt").and_then(Value::as_str) {
        prompt.to_string()
    } else if let Some(turns) = value.get("turns").and_then(Value::as_array) {
        turns
            .iter()
            .find_map(Value::as_str)
            .context("turns did not contain a string prompt")?
            .to_string()
    } else if let Some(messages) = value.get("messages").and_then(Value::as_array) {
        messages
            .iter()
            .filter_map(|message| {
                let role = message.get("role").and_then(Value::as_str)?;
                let content = message.get("content").and_then(Value::as_str)?;
                (role == "user").then_some(content)
            })
            .next()
            .context("messages did not contain a user prompt")?
            .to_string()
    } else {
        bail!("prompt corpus row must include prompt, turns, or messages");
    };
    Ok(PromptCase {
        prompt_id,
        category,
        prompt,
    })
}

fn parse_prompt_token_ids(value: &str) -> Result<Vec<i32>> {
    let mut tokens = Vec::new();
    for token in value
        .split(',')
        .map(str::trim)
        .filter(|token| !token.is_empty())
    {
        tokens.push(
            token
                .parse::<i32>()
                .with_context(|| format!("invalid prompt token id {token}"))?,
        );
    }
    if tokens.is_empty() {
        bail!("--prompt-token-ids must contain at least one token id");
    }
    Ok(tokens)
}

fn remote_stage_status(stage: &StageAssignment) -> RemoteStageStatus {
    let pid = remote_pid(stage).ok().flatten();
    let pid_alive = pid
        .map(|pid| remote_pid_alive(stage, pid).unwrap_or(false))
        .unwrap_or(false);
    let log_ready = remote_log_ready(stage).unwrap_or(false);
    let exit_code = remote_exit_code(stage).ok().flatten();
    let log_tail = remote_log_tail(stage)
        .unwrap_or_else(|error| format!("failed to read remote log tail: {error:#}"));
    RemoteStageStatus {
        stage_id: stage.stage_id.clone(),
        host: stage.host.clone(),
        pid,
        pid_alive,
        log_ready,
        protocol_ready: false,
        exit_code,
        log_tail,
        collected_log_path: None,
        terminated: false,
    }
}

fn remote_pid(stage: &StageAssignment) -> Result<Option<u32>> {
    if stage.local {
        let output = fs::read_to_string(&stage.remote_pid_path).unwrap_or_default();
        let output = output.trim();
        if output.is_empty() {
            return Ok(None);
        }
        return Ok(Some(
            output
                .parse::<u32>()
                .with_context(|| format!("parse pid for {}", stage.stage_id))?,
        ));
    }
    let output = ssh_capture(
        &stage.host,
        &format!(
            "cat {} 2>/dev/null || true",
            shell_quote(&stage.remote_pid_path)
        ),
    )?;
    let output = output.trim();
    if output.is_empty() {
        return Ok(None);
    }
    Ok(Some(output.parse::<u32>().with_context(|| {
        format!("parse pid for {}", stage.stage_id)
    })?))
}

fn remote_pid_alive(stage: &StageAssignment, pid: u32) -> Result<bool> {
    if stage.local {
        return Ok(Command::new("kill")
            .arg("-0")
            .arg(pid.to_string())
            .status()
            .with_context(|| format!("check local pid {pid} for {}", stage.stage_id))?
            .success());
    }
    ssh_success(&stage.host, &format!("kill -0 {pid} 2>/dev/null"))
}

fn remote_pid_alive_opt(stage: &StageAssignment, pid: Option<u32>) -> Result<bool> {
    pid.map(|pid| remote_pid_alive(stage, pid))
        .unwrap_or(Ok(false))
}

fn remote_exit_code(stage: &StageAssignment) -> Result<Option<i32>> {
    if stage.local {
        let output = fs::read_to_string(&stage.remote_exit_code_path).unwrap_or_default();
        let output = output.trim();
        if output.is_empty() {
            return Ok(None);
        }
        return Ok(Some(output.parse::<i32>().with_context(|| {
            format!("parse exit code for {}", stage.stage_id)
        })?));
    }
    let output = ssh_capture(
        &stage.host,
        &format!(
            "cat {} 2>/dev/null || true",
            shell_quote(&stage.remote_exit_code_path)
        ),
    )?;
    let output = output.trim();
    if output.is_empty() {
        return Ok(None);
    }
    Ok(Some(output.parse::<i32>().with_context(|| {
        format!("parse exit code for {}", stage.stage_id)
    })?))
}

fn wait_remote_exit_code(stage: &StageAssignment, timeout: Duration) -> Result<()> {
    let attempts = (timeout.as_millis() / 250).max(1);
    let mut last_error = None;
    for _ in 0..attempts {
        match remote_exit_code(stage) {
            Ok(Some(_)) => return Ok(()),
            Ok(None) => {}
            Err(error) => last_error = Some(error),
        }
        thread::sleep(Duration::from_millis(250));
    }
    Err(last_error.unwrap_or_else(|| anyhow!("timed out waiting for remote exit code")))
}

fn remote_log_ready(stage: &StageAssignment) -> Result<bool> {
    if stage.local {
        let log = fs::read_to_string(&stage.remote_log_path).unwrap_or_default();
        return Ok(log.contains("skippy-server listening: binary="));
    }
    ssh_success(
        &stage.host,
        &format!(
            "test -f {} && grep -q {} {}",
            shell_quote(&stage.remote_log_path),
            shell_quote("skippy-server listening: binary="),
            shell_quote(&stage.remote_log_path)
        ),
    )
}

fn remote_log_tail(stage: &StageAssignment) -> Result<String> {
    if stage.local {
        let log = fs::read_to_string(&stage.remote_log_path).unwrap_or_default();
        let tail = log
            .lines()
            .rev()
            .take(40)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect::<Vec<_>>()
            .join("\n");
        return Ok(tail);
    }
    ssh_capture(
        &stage.host,
        &format!(
            "tail -n 40 {} 2>/dev/null || true",
            shell_quote(&stage.remote_log_path)
        ),
    )
}

fn collect_remote_log(stage: &StageAssignment, logs_dir: &Path) -> Option<PathBuf> {
    let local_path = logs_dir.join(format!("{}-{}.log", stage.stage_index, stage.stage_id));
    if stage.local {
        return fs::copy(&stage.remote_log_path, &local_path)
            .ok()
            .map(|_| local_path);
    }
    let status = Command::new("rsync")
        .arg("-az")
        .arg(format!("{}:{}", stage.host, stage.remote_log_path))
        .arg(&local_path)
        .status()
        .ok()?;
    status.success().then_some(local_path)
}

fn terminate_remote_stage(stage: &StageAssignment, pid: Option<u32>) -> Result<()> {
    let Some(pid) = pid else {
        return Ok(());
    };
    if stage.local {
        run_command(
            Command::new("sh").arg("-c").arg(format!(
                "kill -TERM {pid} 2>/dev/null || true; for i in 1 2 3 4 5; do kill -0 {pid} 2>/dev/null || exit 0; sleep 1; done; kill -KILL {pid} 2>/dev/null || true"
            )),
        )?;
        return Ok(());
    }
    ssh_success(
        &stage.host,
        &format!(
            "kill -TERM {pid} 2>/dev/null || true; for i in 1 2 3 4 5; do kill -0 {pid} 2>/dev/null || exit 0; sleep 1; done; kill -KILL {pid} 2>/dev/null || true"
        ),
    )?;
    Ok(())
}

fn connect_endpoint_ready(endpoint: &str, timeout_secs: u64) -> Result<TcpStream> {
    let endpoint = endpoint.strip_prefix("tcp://").unwrap_or(endpoint);
    let attempts = timeout_secs.saturating_mul(2).max(1);
    let mut last_error = None;
    for _ in 0..attempts {
        match TcpStream::connect(endpoint) {
            Ok(mut stream) => {
                stream.set_nodelay(true).ok();
                match recv_ready(&mut stream) {
                    Ok(()) => return Ok(stream),
                    Err(error) => {
                        last_error = Some(anyhow!(error).context("ready handshake failed"))
                    }
                }
            }
            Err(error) => last_error = Some(anyhow!(error).context("connect failed")),
        }
        thread::sleep(Duration::from_millis(500));
    }
    Err(last_error.unwrap_or_else(|| anyhow!("timed out")))
}

fn rsync_model_artifacts(args: &RunArgs, stage: &StageAssignment) -> Result<()> {
    let Some(stage_model) = args.stage_model.as_ref() else {
        return Ok(());
    };
    if let (Some(local_model), Some(remote_model)) = (
        stage.local_materialized_model_path.as_ref(),
        stage.remote_model_path.as_ref(),
    ) {
        materialize_stage_model_on_coordinator(stage_model, stage, local_model)?;
        if let Some(shared_model) = stage.local_shared_model_path.as_ref() {
            place_stage_model_on_shared_root(local_model, shared_model).with_context(|| {
                format!(
                    "place coordinator-materialized stage model for {} at {}",
                    stage.stage_id,
                    shared_model.display()
                )
            })?;
        } else {
            let remote_parent = remote_parent(remote_model)?;
            run_command(
                Command::new("ssh")
                    .arg(&stage.host)
                    .arg(format!("mkdir -p {}", shell_quote(&remote_parent))),
            )
            .with_context(|| format!("create remote model cache dir on {}", stage.host))?;
            run_command(
                Command::new("rsync")
                    .arg("-az")
                    .arg(local_model)
                    .arg(format!("{}:{}", stage.host, remote_model)),
            )
            .with_context(|| {
                format!(
                    "rsync coordinator-materialized stage model for {} to {}",
                    stage.stage_id, stage.host
                )
            })?;
        }
    } else if args.stage_load_mode == "layer-package" && stage_model.is_dir() {
        let remote_package_dir = format!("{}/package", remote_parent(&stage.remote_config_path)?);
        run_command(
            Command::new("ssh")
                .arg(&stage.host)
                .arg(format!("mkdir -p {remote_package_dir}")),
        )?;
        let mut rsync = Command::new("rsync");
        rsync.arg("-azR");
        rsync.arg("./model-package.json");
        for path in &stage.selected_package_files {
            rsync.arg(format!("./{path}"));
        }
        rsync.arg(format!("{}:{remote_package_dir}/", stage.host));
        rsync.current_dir(stage_model);
        run_command(&mut rsync).with_context(|| {
            format!(
                "rsync selected package artifacts for {} to {}",
                stage.stage_id, stage.host
            )
        })?;
    } else {
        run_command(
            Command::new("rsync")
                .arg("-az")
                .arg(stage_model)
                .arg(format!("{}:{}/model", stage.host, args.remote_root)),
        )
        .with_context(|| format!("rsync model artifact to {}", stage.host))?;
    }
    Ok(())
}

fn place_stage_model_on_shared_root(source: &Path, destination: &Path) -> Result<()> {
    let source_metadata =
        fs::metadata(source).with_context(|| format!("read stage model {}", source.display()))?;
    if destination.is_file() {
        let destination_metadata = fs::metadata(destination)
            .with_context(|| format!("read shared stage model {}", destination.display()))?;
        if destination_metadata.len() == source_metadata.len()
            && destination_metadata.modified().with_context(|| {
                format!("read shared stage model mtime {}", destination.display())
            })? >= source_metadata
                .modified()
                .with_context(|| format!("read stage model mtime {}", source.display()))?
        {
            return Ok(());
        }
    }
    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("create shared stage model dir {}", parent.display()))?;
    }
    let tmp = destination.with_extension("gguf.tmp");
    let _ = fs::remove_file(&tmp);
    match fs::hard_link(source, destination) {
        Ok(()) => Ok(()),
        Err(_) => {
            fs::copy(source, &tmp).with_context(|| {
                format!("copy stage model {} to {}", source.display(), tmp.display())
            })?;
            fs::rename(&tmp, destination).with_context(|| {
                format!(
                    "move shared stage model {} to {}",
                    tmp.display(),
                    destination.display()
                )
            })?;
            Ok(())
        }
    }
}

fn materialize_stage_model_on_coordinator(
    package_dir: &Path,
    stage: &StageAssignment,
    output: &Path,
) -> Result<()> {
    let input_paths = stage
        .selected_package_files
        .iter()
        .map(|path| package_dir.join(path))
        .collect::<Vec<_>>();
    if materialized_stage_model_is_current(output, &input_paths)? {
        return Ok(());
    }
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("create materialized stage model dir {}", parent.display()))?;
    }
    write_gguf_from_parts(&input_paths, output).with_context(|| {
        format!(
            "materialize {} package files for {} into {}",
            input_paths.len(),
            stage.stage_id,
            output.display()
        )
    })
}

fn materialized_stage_model_is_current(output: &Path, inputs: &[PathBuf]) -> Result<bool> {
    if !output.is_file() {
        return Ok(false);
    }
    let output_metadata = fs::metadata(output)
        .with_context(|| format!("read materialized stage model {}", output.display()))?;
    if output_metadata.len() == 0 {
        return Ok(false);
    }
    let output_modified = output_metadata
        .modified()
        .with_context(|| format!("read materialized stage model mtime {}", output.display()))?;
    for input in inputs {
        let input_modified = fs::metadata(input)
            .with_context(|| format!("read package part {}", input.display()))?
            .modified()
            .with_context(|| format!("read package part mtime {}", input.display()))?;
        if input_modified > output_modified {
            return Ok(false);
        }
    }
    Ok(true)
}

fn run_command(command: &mut Command) -> Result<()> {
    let status = command
        .status()
        .with_context(|| format!("failed to spawn {:?}", command))?;
    if !status.success() {
        bail!("command failed with status {status}: {:?}", command);
    }
    Ok(())
}

fn ssh_success(host: &str, remote_command: &str) -> Result<bool> {
    let status = Command::new("ssh")
        .arg(host)
        .arg(remote_command)
        .status()
        .with_context(|| format!("run ssh command on {host}"))?;
    Ok(status.success())
}

fn ssh_capture(host: &str, remote_command: &str) -> Result<String> {
    let output = Command::new("ssh")
        .arg(host)
        .arg(remote_command)
        .output()
        .with_context(|| format!("run ssh command on {host}"))?;
    if !output.status.success() {
        bail!(
            "ssh command on {host} failed with status {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        );
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

fn shell_quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\\''"))
}

fn configure_child_logs(command: &mut Command, child_logs: bool) {
    if child_logs {
        command.stdout(Stdio::inherit()).stderr(Stdio::inherit());
    } else {
        command.stdout(Stdio::null()).stderr(Stdio::null());
    }
}

fn metrics_otlp_grpc_url(args: &RunArgs) -> String {
    args.metrics_otlp_grpc_url
        .clone()
        .unwrap_or_else(|| format!("http://{}", args.metrics_otlp_grpc_addr))
}

fn coordinator_materializes_layer_package(args: &RunArgs) -> bool {
    args.execute_remote
        && args.rsync_model_artifacts
        && args.stage_load_mode == "layer-package"
        && args
            .stage_model
            .as_ref()
            .map(|path| path.is_dir())
            .unwrap_or(false)
}

fn parse_remote_root_map(value: Option<&str>) -> Result<BTreeMap<String, String>> {
    let mut roots = BTreeMap::new();
    let Some(value) = value else {
        return Ok(roots);
    };
    for entry in value
        .split(',')
        .map(str::trim)
        .filter(|entry| !entry.is_empty())
    {
        let (host, root) = entry.split_once('=').with_context(|| {
            format!("invalid remote root mapping {entry:?}; expected host=/path")
        })?;
        let host = host.trim();
        let root = root.trim().trim_end_matches('/');
        if host.is_empty() || root.is_empty() {
            bail!("invalid remote root mapping {entry:?}; expected host=/path");
        }
        roots.insert(host.to_string(), root.to_string());
    }
    Ok(roots)
}

fn parse_path_map(value: Option<&str>) -> Result<BTreeMap<String, PathBuf>> {
    Ok(parse_remote_root_map(value)?
        .into_iter()
        .map(|(host, path)| (host, PathBuf::from(path)))
        .collect())
}

fn model_ref_for_configs(args: &RunArgs) -> Result<String> {
    match args.stage_load_mode.as_str() {
        "runtime-slice" => args
            .model_path
            .as_ref()
            .map(|path| path_string(path))
            .context("--model-path is required when --stage-load-mode runtime-slice"),
        "artifact-slice" | "layer-package" => {
            let stage_model = args.stage_model.as_ref().with_context(|| {
                format!(
                    "--stage-model is required when --stage-load-mode {}",
                    args.stage_load_mode
                )
            })?;
            Ok(path_string(stage_model))
        }
        other => bail!(
            "unsupported --stage-load-mode {other}; expected runtime-slice, artifact-slice, or layer-package"
        ),
    }
}

fn parse_hosts(hosts: &str) -> Result<Vec<String>> {
    let parsed = hosts
        .split(',')
        .map(str::trim)
        .filter(|host| !host.is_empty())
        .map(ToOwned::to_owned)
        .collect::<Vec<_>>();
    if parsed.is_empty() {
        bail!("--hosts must contain at least one host");
    }
    let unique = parsed.iter().collect::<BTreeSet<_>>();
    if unique.len() != parsed.len() {
        bail!("--hosts must not repeat hosts; each stage must run on a separate node");
    }
    Ok(parsed)
}

fn validate_distinct_stage_hosts(hosts: &[String], stage_count: usize) -> Result<()> {
    if hosts.len() < stage_count {
        bail!(
            "--hosts supplies {} unique hosts for {stage_count} stages; provide one separate node per stage",
            hosts.len()
        );
    }
    Ok(())
}

fn validate_balanced_stage_ranges(ranges: &[(u32, u32)]) -> Result<()> {
    let Some(first) = ranges.first() else {
        bail!("at least one stage range is required");
    };
    let mut min_len = first.1 - first.0;
    let mut max_len = min_len;
    let mut lengths = Vec::with_capacity(ranges.len());
    for &(start, end) in ranges {
        let len = end - start;
        lengths.push(len);
        min_len = min_len.min(len);
        max_len = max_len.max(len);
    }
    if max_len - min_len > 1 {
        bail!(
            "stage layer ranges must be evenly balanced across nodes; got lengths {:?}",
            lengths
        );
    }
    Ok(())
}

fn parse_stage_ranges(splits: &str, layer_end: u32) -> Result<Vec<(u32, u32)>> {
    let mut boundaries = vec![0];
    for split in splits
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        boundaries.push(
            split
                .parse::<u32>()
                .with_context(|| format!("invalid split {split}"))?,
        );
    }
    boundaries.push(layer_end);
    for pair in boundaries.windows(2) {
        if pair[0] >= pair[1] {
            bail!("splits must be strictly ascending and less than layer_end");
        }
    }
    Ok(boundaries
        .windows(2)
        .map(|pair| (pair[0], pair[1]))
        .collect())
}

fn load_package_manifest(package_dir: &Path) -> Result<PackageManifest> {
    let path = package_dir.join("model-package.json");
    let contents = fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
    serde_json::from_str(&contents).with_context(|| format!("parse {}", path.display()))
}

fn selected_package_files(
    manifest: &PackageManifest,
    layer_start: u32,
    layer_end: u32,
    include_embeddings: bool,
    include_output: bool,
) -> Result<Vec<String>> {
    let mut files = vec![manifest.shared.metadata.path.clone()];
    if include_embeddings {
        files.push(manifest.shared.embeddings.path.clone());
    }
    for layer_index in layer_start..layer_end {
        let layer = manifest
            .layers
            .iter()
            .find(|layer| layer.layer_index == layer_index)
            .with_context(|| format!("package manifest is missing layer {layer_index}"))?;
        files.push(layer.path.clone());
    }
    if include_output {
        files.push(manifest.shared.output.path.clone());
    }
    Ok(files)
}

fn stage_model_cache_key(
    args: &RunArgs,
    stage_id: &str,
    layer_start: u32,
    layer_end: u32,
) -> PathBuf {
    PathBuf::from(safe_cache_component(&args.model_id))
        .join(safe_cache_component(&args.topology_id))
        .join(format!(
            "{}-{}-{}",
            safe_cache_component(stage_id),
            layer_start,
            layer_end
        ))
}

fn safe_cache_component(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.') {
                ch
            } else {
                '_'
            }
        })
        .collect()
}

fn remote_parent(path: &str) -> Result<String> {
    path.rsplit_once('/')
        .map(|(parent, _)| parent.to_string())
        .context("remote path has no parent")
}

fn write_json_file(path: &Path, value: &impl Serialize) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("create directory {}", parent.display()))?;
    }
    fs::write(path, format!("{}\n", serde_json::to_string_pretty(value)?))
        .with_context(|| format!("write {}", path.display()))
}

fn path_string(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

fn generate_bench_run_id() -> String {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock before Unix epoch")
        .as_millis();
    format!("run-bench-{millis}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_lab_hosts() {
        assert_eq!(
            parse_hosts("shadowfax.local,black.local,studio54.local,build.local").unwrap(),
            vec![
                "shadowfax.local",
                "black.local",
                "studio54.local",
                "build.local"
            ]
        );
    }

    #[test]
    fn rejects_duplicate_stage_hosts() {
        assert!(parse_hosts("shadowfax.local,black.local,shadowfax.local").is_err());
    }

    #[test]
    fn requires_one_host_per_stage() {
        let hosts = parse_hosts("shadowfax.local,black.local").unwrap();
        assert!(validate_distinct_stage_hosts(&hosts, 3).is_err());
        assert!(validate_distinct_stage_hosts(&hosts, 2).is_ok());
    }

    #[test]
    fn builds_stable_stage_model_cache_key() {
        let args = RunArgs {
            metrics_server_bin: PathBuf::from("metrics-server"),
            stage_server_bin: PathBuf::from("skippy-server"),
            hosts: "shadowfax.local,black.local".to_string(),
            run_id: Some("run-1".to_string()),
            topology_id: "quad/small".to_string(),
            model_id: "Qwen/Qwen3-4B:Q4_K_M".to_string(),
            model_path: None,
            stage_model: Some(PathBuf::from("model-package")),
            stage_load_mode: "layer-package".to_string(),
            splits: "1".to_string(),
            layer_end: 2,
            ctx_size: 128,
            n_gpu_layers: 0,
            selected_backend_device: None,
            cache_type_k: "f16".to_string(),
            cache_type_v: "f16".to_string(),
            activation_width: 2048,
            activation_wire_dtype: "f32".to_string(),
            prompt: "Hello".to_string(),
            prompt_corpus: None,
            prompt_limit: None,
            prompt_token_ids: None,
            max_new_tokens: Some(1),
            prefill_chunk_size: None,
            prefill_chunk_threshold: None,
            prefill_chunk_schedule: None,
            metrics_http_addr: "127.0.0.1:18080".parse().unwrap(),
            metrics_otlp_grpc_addr: "127.0.0.1:14317".parse().unwrap(),
            metrics_otlp_grpc_url: None,
            db: None,
            output: None,
            work_dir: PathBuf::from("/tmp/work"),
            remote_root: "/tmp/remote".to_string(),
            remote_root_map: None,
            remote_shared_root_map: None,
            endpoint_host_map: None,
            remote_bind_host: "0.0.0.0".to_string(),
            first_stage_port: 19031,
            execute_remote: false,
            keep_remote: false,
            rsync_model_artifacts: true,
            child_logs: false,
            startup_timeout_secs: 60,
            stage_max_inflight: 4,
            stage_reply_credit_limit: None,
            stage_async_prefill_forward: false,
            stage_downstream_wire_delay_ms: 0.0,
            stage_downstream_wire_mbps: None,
            stage_telemetry_queue_capacity: 8192,
            stage_telemetry_level: "summary".to_string(),
        };

        assert_eq!(
            stage_model_cache_key(&args, "stage-0", 0, 1),
            PathBuf::from("Qwen_Qwen3-4B_Q4_K_M")
                .join("quad_small")
                .join("stage-0-0-1")
        );
    }

    #[test]
    fn builds_stage_ranges_from_splits() {
        assert_eq!(
            parse_stage_ranges("1,4", 40).unwrap(),
            vec![(0, 1), (1, 4), (4, 40)]
        );
        assert!(parse_stage_ranges("4,1", 40).is_err());
        assert!(parse_stage_ranges("1,40", 40).is_err());
    }

    #[test]
    fn requires_balanced_stage_ranges() {
        validate_balanced_stage_ranges(&parse_stage_ranges("14,27", 40).unwrap()).unwrap();
        validate_balanced_stage_ranges(&parse_stage_ranges("13,27", 40).unwrap()).unwrap();
        assert!(
            validate_balanced_stage_ranges(&parse_stage_ranges("12,20,28", 40).unwrap()).is_err()
        );
        assert!(validate_balanced_stage_ranges(&parse_stage_ranges("1,4,7", 40).unwrap()).is_err());
    }

    #[test]
    fn selects_minimal_package_files() {
        let manifest = PackageManifest {
            model_id: "org/repo:Q4_K_M".to_string(),
            source_model: PackageSourceModel {
                repo: Some("org/repo".to_string()),
                revision: Some("abc123".to_string()),
                primary_file: Some("Model-Q4_K_M.gguf".to_string()),
                canonical_ref: Some("org/repo@abc123/Model-Q4_K_M.gguf".to_string()),
                distribution_id: Some("Model-Q4_K_M".to_string()),
            },
            shared: PackageShared {
                metadata: PackageArtifact {
                    path: "shared/metadata.gguf".to_string(),
                },
                embeddings: PackageArtifact {
                    path: "shared/embeddings.gguf".to_string(),
                },
                output: PackageArtifact {
                    path: "shared/output.gguf".to_string(),
                },
            },
            layers: vec![
                PackageLayer {
                    layer_index: 0,
                    path: "layers/layer-000.gguf".to_string(),
                },
                PackageLayer {
                    layer_index: 1,
                    path: "layers/layer-001.gguf".to_string(),
                },
            ],
        };
        let files = selected_package_files(&manifest, 0, 1, true, false).unwrap();
        assert_eq!(
            files,
            vec![
                "shared/metadata.gguf",
                "shared/embeddings.gguf",
                "layers/layer-000.gguf"
            ]
        );
    }

    #[test]
    fn reads_model_identity_from_package_manifest() {
        let manifest = PackageManifest {
            model_id: "org/repo:Q4_K_M".to_string(),
            source_model: PackageSourceModel {
                repo: Some("org/repo".to_string()),
                revision: Some("abc123".to_string()),
                primary_file: Some("Model-Q4_K_M.gguf".to_string()),
                canonical_ref: Some("org/repo@abc123/Model-Q4_K_M.gguf".to_string()),
                distribution_id: Some("Model-Q4_K_M".to_string()),
            },
            shared: PackageShared {
                metadata: PackageArtifact {
                    path: "shared/metadata.gguf".to_string(),
                },
                embeddings: PackageArtifact {
                    path: "shared/embeddings.gguf".to_string(),
                },
                output: PackageArtifact {
                    path: "shared/output.gguf".to_string(),
                },
            },
            layers: Vec::new(),
        };

        let identity = model_identity_from_package_manifest(&manifest).unwrap();
        assert_eq!(identity.model_id, "org/repo:Q4_K_M");
        assert_eq!(identity.source_repo.as_deref(), Some("org/repo"));
        assert_eq!(identity.source_revision.as_deref(), Some("abc123"));
        assert_eq!(identity.source_file.as_deref(), Some("Model-Q4_K_M.gguf"));
        assert_eq!(identity.selector.as_deref(), Some("Q4_K_M"));
    }

    #[test]
    fn parses_prompt_token_ids() {
        assert_eq!(parse_prompt_token_ids("1, 2,3").unwrap(), vec![1, 2, 3]);
        assert!(parse_prompt_token_ids("").is_err());
        assert!(parse_prompt_token_ids("1,nope").is_err());
    }

    #[test]
    fn applies_prefill_chunk_threshold() {
        let mut args = test_run_args();
        args.prefill_chunk_size = Some(128);
        assert_eq!(effective_prefill_chunk_size(&args, 64), 128);
        assert_eq!(effective_prefill_chunk_size(&args, 256), 128);

        args.prefill_chunk_threshold = Some(128);
        assert_eq!(effective_prefill_chunk_size(&args, 64), 64);
        assert_eq!(effective_prefill_chunk_size(&args, 128), 128);
        assert_eq!(effective_prefill_chunk_size(&args, 129), 128);
    }

    #[test]
    fn applies_prefill_chunk_schedule() {
        let mut args = test_run_args();
        args.prefill_chunk_size = Some(256);
        args.prefill_chunk_schedule = Some("513:512,1025:768".to_string());

        assert_eq!(effective_prefill_chunk_size(&args, 512), 256);
        assert_eq!(effective_prefill_chunk_size(&args, 513), 512);
        assert_eq!(effective_prefill_chunk_size(&args, 1024), 512);
        assert_eq!(effective_prefill_chunk_size(&args, 1025), 768);
    }

    #[test]
    fn parses_prompt_cases_from_corpus_shapes() {
        let turns = prompt_case_from_value(&json!({
            "prompt_id": 42,
            "category": "math",
            "turns": ["first turn", "second turn"]
        }))
        .unwrap();
        assert_eq!(turns.prompt_id.as_deref(), Some("42"));
        assert_eq!(turns.category.as_deref(), Some("math"));
        assert_eq!(turns.prompt, "first turn");

        let messages = prompt_case_from_value(&json!({
            "id": "mt_bench_1",
            "messages": [
                {"role": "system", "content": "ignore"},
                {"role": "user", "content": "hello"}
            ]
        }))
        .unwrap();
        assert_eq!(messages.prompt_id.as_deref(), Some("mt_bench_1"));
        assert_eq!(messages.prompt, "hello");
    }

    #[test]
    fn summarizes_prompt_driver_percentiles() {
        let results = [100_u128, 200, 300, 400]
            .into_iter()
            .map(|elapsed_ms| PromptDriverResult {
                prompt_id: None,
                category: None,
                prompt: "hello".to_string(),
                token_ids: vec![1, 2],
                prefill_token_count: 1,
                prefill_chunk_count: 1,
                effective_prefill_chunk_size: Some(1),
                predicted_tokens: vec![3],
                max_new_tokens: 1,
                elapsed_ms,
                wire_elapsed_ms: elapsed_ms - 10,
                prefill_elapsed_ms: elapsed_ms - 20,
                ttft_ms: elapsed_ms - 15,
                decode_elapsed_ms: 10,
            })
            .collect::<Vec<_>>();

        let summary = prompt_driver_summary(&results);
        assert_eq!(summary.prompt_tokens_total, 8);
        assert_eq!(summary.generated_tokens_total, 4);
        assert_eq!(summary.elapsed_ms_total, 1000);
        assert_eq!(summary.elapsed_ms_p50, 300);
        assert_eq!(summary.elapsed_ms_p95, 400);
        assert_eq!(summary.elapsed_ms_p99, 400);
        assert_eq!(summary.wire_elapsed_ms_p50, 290);
        assert_eq!(summary.prefill_elapsed_ms_p50, 280);
        assert_eq!(summary.ttft_ms_p50, 285);
        assert_eq!(summary.decode_elapsed_ms_p50, 10);
    }

    #[test]
    fn focused_runtime_schema_smoke_uses_compact_output_shape() {
        let mut run = test_run_args();
        run.hosts = "host-a,host-b".to_string();
        run.splits = "1".to_string();
        run.layer_end = 2;
        run.prompt_limit = Some(3);
        run.max_new_tokens = Some(7);
        let args = FocusedRuntimeArgs {
            scenario: FocusedRuntimeScenario::FirstToken,
            focused_output: None,
            schema_smoke: true,
            run,
        };

        let report = focused_runtime_schema_smoke_report(&args).unwrap();
        let value = serde_json::to_value(&report).unwrap();

        assert_eq!(value["schema_version"], 1);
        assert_eq!(value["scenario"], "first-token");
        assert_eq!(value["mode"], "schema-smoke");
        assert_eq!(value["stage_count"], 2);
        assert_eq!(value["hosts"], json!(["host-a", "host-b"]));
        assert_eq!(value["summary"]["prompt_count"], 3);
        assert_eq!(value["summary"]["max_new_tokens"], 7);
        assert_eq!(value["summary"]["elapsed_ms_p95"], 10);
        assert_eq!(value["summary"]["ttft_ms_p50"], 5);
        assert_eq!(value["summary"]["generated_tokens_per_second"], 100.0);
        assert_eq!(value["topology"]["topology_id"], "topology");
        assert_eq!(value["topology"]["stage_count"], 2);
        assert_eq!(
            value["model"]["model_id"],
            "test-org/bench-model-GGUF:Q4_K_M"
        );
        assert_eq!(value["latency_ms"]["elapsed_ms_p95"], 10);
        assert_eq!(value["latency_ms"]["startup_elapsed_ms"], 0);
        assert_eq!(value["throughput_tokens_per_second"]["generated"], 100.0);
        assert_eq!(value["token_counts"]["prompt_total"], 24);
        assert_eq!(value["token_counts"]["generated_total"], 21);
        assert_eq!(
            value["model_identity"]["model_id"],
            "test-org/bench-model-GGUF:Q4_K_M"
        );
        assert_eq!(
            value["outputs"]["deployment_plan"],
            "schema-smoke-deployment-plan.json"
        );
    }

    #[test]
    fn focused_runtime_preset_only_rewrites_omitted_decode_budget() {
        let mut run = test_run_args();
        run.max_new_tokens = None;
        let args = FocusedRuntimeArgs {
            scenario: FocusedRuntimeScenario::SteadyDecode,
            focused_output: None,
            schema_smoke: true,
            run,
        };

        let args = apply_focused_runtime_preset(args);
        assert_eq!(args.run.prompt_limit, Some(1));
        assert_eq!(args.run.max_new_tokens, Some(128));

        let mut run = test_run_args();
        run.max_new_tokens = Some(1);
        let args = FocusedRuntimeArgs {
            scenario: FocusedRuntimeScenario::SteadyDecode,
            focused_output: None,
            schema_smoke: true,
            run,
        };

        let args = apply_focused_runtime_preset(args);
        assert_eq!(args.run.max_new_tokens, Some(1));
    }

    #[test]
    fn focused_runtime_kv_warm_reuse_preserves_explicit_one_token_budget() {
        let mut run = test_run_args();
        run.max_new_tokens = Some(1);
        let args = FocusedRuntimeArgs {
            scenario: FocusedRuntimeScenario::KvWarmReuse,
            focused_output: None,
            schema_smoke: true,
            run,
        };

        let args = apply_focused_runtime_preset(args);
        assert_eq!(args.run.prompt_limit, Some(2));
        assert_eq!(args.run.max_new_tokens, Some(1));
    }

    #[test]
    fn focused_runtime_cold_startup_description_matches_default_decode_budget() {
        let description = focused_runtime_preset_description(FocusedRuntimeScenario::ColdStartup);

        assert!(description.contains("default one-token decode budget"));
        assert!(!description.starts_with("one-prompt, one-token run"));
    }

    #[test]
    fn run_args_default_to_one_generated_token_when_omitted() {
        let run = test_run_args();

        assert_eq!(run.max_new_tokens, None);
        assert_eq!(effective_run_max_new_tokens(&run), 1);
    }

    #[test]
    fn focused_runtime_schema_smoke_rejects_invalid_topology() {
        let mut run = test_run_args();
        run.hosts = "host-a".to_string();
        run.splits = "1".to_string();
        run.layer_end = 2;
        let args = FocusedRuntimeArgs {
            scenario: FocusedRuntimeScenario::FirstToken,
            focused_output: None,
            schema_smoke: true,
            run,
        };

        let err = focused_runtime_schema_smoke_report(&args).unwrap_err();
        assert!(err
            .to_string()
            .contains("provide one separate node per stage"));

        let mut run = test_run_args();
        run.hosts = "host-a,host-b,host-c,host-d".to_string();
        run.splits = "1,4,7".to_string();
        run.layer_end = 40;
        let args = FocusedRuntimeArgs {
            scenario: FocusedRuntimeScenario::FirstToken,
            focused_output: None,
            schema_smoke: true,
            run,
        };

        let err = focused_runtime_schema_smoke_report(&args).unwrap_err();
        assert!(err.to_string().contains("evenly balanced"));
    }

    #[test]
    fn focused_runtime_requires_executed_run_unless_schema_smoke() {
        let mut run = test_run_args();
        run.hosts = "host-a,host-b".to_string();
        let args = FocusedRuntimeArgs {
            scenario: FocusedRuntimeScenario::SteadyDecode,
            focused_output: None,
            schema_smoke: false,
            run,
        };

        let err = validate_focused_runtime_args(&args).unwrap_err();
        assert!(err.to_string().contains("requires --execute-remote"));

        let smoke_args = FocusedRuntimeArgs {
            schema_smoke: true,
            ..args
        };
        validate_focused_runtime_args(&smoke_args).unwrap();
    }

    #[test]
    fn focused_runtime_summary_reuses_prompt_driver_fields() {
        let results = [120_u128, 240, 360]
            .into_iter()
            .map(|elapsed_ms| PromptDriverResult {
                prompt_id: None,
                category: None,
                prompt: "hello".to_string(),
                token_ids: vec![1, 2, 3],
                prefill_token_count: 2,
                prefill_chunk_count: 1,
                effective_prefill_chunk_size: Some(2),
                predicted_tokens: vec![4, 5],
                max_new_tokens: 2,
                elapsed_ms,
                wire_elapsed_ms: elapsed_ms - 5,
                prefill_elapsed_ms: elapsed_ms - 10,
                ttft_ms: elapsed_ms - 20,
                decode_elapsed_ms: 20,
            })
            .collect::<Vec<_>>();
        let summary = prompt_driver_summary(&results);
        let driver = PromptDriverReport {
            first_stage_endpoint: "tcp://host-a:19031".to_string(),
            prompt_count: 3,
            max_new_tokens: 2,
            prefill_chunk_size: Some(2),
            prefill_chunk_threshold: None,
            prefill_chunk_schedule: None,
            corpus: None,
            summary,
            results,
        };

        let focused = focused_runtime_summary(
            &driver,
            Some(Duration::from_millis(1234)),
            Duration::from_millis(5678),
        );

        assert_eq!(focused.startup_elapsed_ms, Some(1234));
        assert_eq!(focused.run_elapsed_ms, 5678);
        assert_eq!(focused.prompt_count, 3);
        assert_eq!(focused.max_new_tokens, 2);
        assert_eq!(focused.prompt_tokens_total, 9);
        assert_eq!(focused.generated_tokens_total, 6);
        assert_eq!(focused.elapsed_ms_p50, 240);
        assert_eq!(focused.elapsed_ms_p95, 360);
        assert_eq!(focused.ttft_ms_p50, 220);
        assert_eq!(focused.decode_elapsed_ms_p95, 20);
        assert!(focused.total_tokens_per_second > 0.0);
        assert!(focused.generated_tokens_per_second > 0.0);
    }

    #[test]
    fn parses_remote_root_map() {
        let roots = parse_remote_root_map(Some(
            "build.local=/Users/jdumay/models/bench, black.local=/tmp/bench",
        ))
        .unwrap();
        assert_eq!(
            roots.get("build.local").map(String::as_str),
            Some("/Users/jdumay/models/bench")
        );
        assert_eq!(
            roots.get("black.local").map(String::as_str),
            Some("/tmp/bench")
        );
        assert!(parse_remote_root_map(Some("build.local")).is_err());
    }

    #[test]
    fn remote_start_command_records_exit_code() {
        let args = RunArgs {
            metrics_server_bin: PathBuf::from("metrics-server"),
            stage_server_bin: PathBuf::from("skippy-server"),
            hosts: "host.local".to_string(),
            run_id: Some("run-1".to_string()),
            topology_id: "topology".to_string(),
            model_id: "test-org/bench-model-GGUF:Q4_K_M".to_string(),
            model_path: Some(PathBuf::from("model.gguf")),
            stage_model: None,
            stage_load_mode: "runtime-slice".to_string(),
            splits: "1".to_string(),
            layer_end: 2,
            ctx_size: 128,
            n_gpu_layers: 0,
            selected_backend_device: None,
            cache_type_k: "f16".to_string(),
            cache_type_v: "f16".to_string(),
            activation_width: 2048,
            activation_wire_dtype: "f32".to_string(),
            prompt: "Hello".to_string(),
            prompt_corpus: None,
            prompt_limit: None,
            prompt_token_ids: None,
            max_new_tokens: Some(1),
            prefill_chunk_size: None,
            prefill_chunk_threshold: None,
            prefill_chunk_schedule: None,
            metrics_http_addr: "127.0.0.1:18080".parse().unwrap(),
            metrics_otlp_grpc_addr: "127.0.0.1:14317".parse().unwrap(),
            metrics_otlp_grpc_url: Some("http://coordinator.local:14317".to_string()),
            db: None,
            output: None,
            work_dir: PathBuf::from("/tmp/work"),
            remote_root: "/tmp/remote".to_string(),
            remote_root_map: None,
            remote_shared_root_map: None,
            endpoint_host_map: None,
            remote_bind_host: "0.0.0.0".to_string(),
            first_stage_port: 19031,
            execute_remote: true,
            keep_remote: false,
            rsync_model_artifacts: false,
            child_logs: false,
            startup_timeout_secs: 60,
            stage_max_inflight: 4,
            stage_reply_credit_limit: Some(2),
            stage_async_prefill_forward: true,
            stage_downstream_wire_delay_ms: 1.0,
            stage_downstream_wire_mbps: Some(1000.0),
            stage_telemetry_queue_capacity: 8192,
            stage_telemetry_level: "summary".to_string(),
        };
        let plan = DeploymentPlan {
            run_id: "run-1".to_string(),
            topology_id: "topology".to_string(),
            model_id: "test-org/bench-model-GGUF:Q4_K_M".to_string(),
            model_identity: ModelIdentity::from_model_id("test-org/bench-model-GGUF:Q4_K_M"),
            hosts: vec!["host.local".to_string()],
            stage_load_mode: "runtime-slice".to_string(),
            remote_root: "/tmp/remote".to_string(),
            remote_roots: BTreeMap::new(),
            remote_shared_roots: BTreeMap::new(),
            endpoint_hosts: BTreeMap::new(),
            work_dir: PathBuf::from("/tmp/work"),
            metrics_http: "http://127.0.0.1:18080".to_string(),
            metrics_otlp_grpc: "http://coordinator.local:14317".to_string(),
            stages: Vec::new(),
            execute_remote: true,
            keep_remote: false,
            rsync_model_artifacts: false,
        };
        let stage = StageAssignment {
            stage_id: "stage-0".to_string(),
            stage_index: 0,
            host: "host.local".to_string(),
            local: false,
            layer_start: 0,
            layer_end: 1,
            bind_addr: "0.0.0.0:19031".to_string(),
            endpoint: "tcp://host.local:19031".to_string(),
            config_path: PathBuf::from("/tmp/local/stage.json"),
            remote_config_path: "/tmp/remote/run-1/stage-0/stage.json".to_string(),
            remote_log_path: "/tmp/remote/run-1/stage-0/stage.log".to_string(),
            remote_pid_path: "/tmp/remote/run-1/stage-0/stage.pid".to_string(),
            remote_exit_code_path: "/tmp/remote/run-1/stage-0/stage.exit".to_string(),
            remote_model_path: None,
            local_materialized_model_path: None,
            local_shared_model_path: None,
            selected_package_files: Vec::new(),
        };
        let command = remote_start_command(
            &args,
            &plan,
            &stage,
            "/tmp/remote/run-1/stage-0/skippy-server",
        );
        assert!(command.contains("stage.exit"));
        assert!(command.contains("stage.pid"));
        assert!(command.contains("wait \"$child\""));
        assert!(!command.contains("nohup"));
        assert!(command.contains("--metrics-otlp-grpc"));
        assert!(command.contains("coordinator.local:14317"));
        assert!(command.contains("--max-inflight 4"));
        assert!(command.contains("--reply-credit-limit 2"));
        assert!(command.contains("--async-prefill-forward"));
        assert!(command.contains("--downstream-wire-delay-ms 1"));
        assert!(command.contains("--downstream-wire-mbps 1000"));
        assert!(command.contains("--telemetry-level"));
        assert!(command.contains("summary"));
    }

    fn test_run_args() -> RunArgs {
        RunArgs {
            metrics_server_bin: PathBuf::from("metrics-server"),
            stage_server_bin: PathBuf::from("skippy-server"),
            hosts: "host.local".to_string(),
            run_id: Some("run-1".to_string()),
            topology_id: "topology".to_string(),
            model_id: "test-org/bench-model-GGUF:Q4_K_M".to_string(),
            model_path: Some(PathBuf::from("model.gguf")),
            stage_model: None,
            stage_load_mode: "runtime-slice".to_string(),
            splits: "1".to_string(),
            layer_end: 2,
            ctx_size: 128,
            n_gpu_layers: 0,
            selected_backend_device: None,
            cache_type_k: "f16".to_string(),
            cache_type_v: "f16".to_string(),
            activation_width: 2048,
            activation_wire_dtype: "f32".to_string(),
            prompt: "Hello".to_string(),
            prompt_corpus: None,
            prompt_limit: None,
            prompt_token_ids: None,
            max_new_tokens: None,
            prefill_chunk_size: None,
            prefill_chunk_threshold: None,
            prefill_chunk_schedule: None,
            metrics_http_addr: "127.0.0.1:18080".parse().unwrap(),
            metrics_otlp_grpc_addr: "127.0.0.1:14317".parse().unwrap(),
            metrics_otlp_grpc_url: None,
            db: None,
            output: None,
            work_dir: PathBuf::from("/tmp/work"),
            remote_root: "/tmp/remote".to_string(),
            remote_root_map: None,
            remote_shared_root_map: None,
            endpoint_host_map: None,
            remote_bind_host: "0.0.0.0".to_string(),
            first_stage_port: 19031,
            execute_remote: false,
            keep_remote: false,
            rsync_model_artifacts: false,
            child_logs: false,
            startup_timeout_secs: 60,
            stage_max_inflight: 4,
            stage_reply_credit_limit: None,
            stage_async_prefill_forward: false,
            stage_downstream_wire_delay_ms: 0.0,
            stage_downstream_wire_mbps: None,
            stage_telemetry_queue_capacity: 8192,
            stage_telemetry_level: "summary".to_string(),
        }
    }

    #[test]
    fn planner_rejects_qwen_q8_before_launch() {
        let mut args = test_run_args();
        args.model_id = "Qwen/Qwen3-0.6B".to_string();
        args.hosts = "host-a,host-b".to_string();
        args.splits = "14".to_string();
        args.layer_end = 28;
        args.activation_width = 1024;
        args.activation_wire_dtype = "q8".to_string();
        let hosts = parse_hosts(&args.hosts).unwrap();
        let ranges = parse_stage_ranges(&args.splits, args.layer_end).unwrap();

        let err = validate_topology_plan(&args, &hosts, &ranges).unwrap_err();

        assert!(err.to_string().contains("rejected q8"));
    }

    #[test]
    fn planner_rejects_gemma_known_bad_split_before_launch() {
        let mut args = test_run_args();
        args.model_id = "gemma-4-e4b".to_string();
        args.hosts = "host-a,host-b,host-c".to_string();
        args.splits = "14,28".to_string();
        args.layer_end = 42;
        args.activation_width = 2560;
        args.activation_wire_dtype = "f16".to_string();
        let hosts = parse_hosts(&args.hosts).unwrap();
        let ranges = parse_stage_ranges(&args.splits, args.layer_end).unwrap();

        let err = validate_topology_plan(&args, &hosts, &ranges).unwrap_err();

        assert!(err.to_string().contains("SharedKvRegionCut"));
    }

    #[test]
    fn planner_accepts_gemma_validated_split() {
        let mut args = test_run_args();
        args.model_id = "gemma-4-e4b".to_string();
        args.hosts = "host-a,host-b".to_string();
        args.splits = "21".to_string();
        args.layer_end = 42;
        args.activation_width = 2560;
        args.activation_wire_dtype = "f16".to_string();
        let hosts = parse_hosts(&args.hosts).unwrap();
        let ranges = parse_stage_ranges(&args.splits, args.layer_end).unwrap();

        validate_topology_plan(&args, &hosts, &ranges).unwrap();
    }
}
