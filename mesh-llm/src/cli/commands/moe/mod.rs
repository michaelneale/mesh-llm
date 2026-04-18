mod formatters;
mod formatters_console;
mod formatters_json;
mod hf_jobs;

use anyhow::{bail, Context, Result};
use hf_hub::{
    AddSource, CommitInfo, CommitOperation, CreateRepoParams, FileProgress, FileStatus, HFError,
    Progress, ProgressEvent, ProgressHandler, RepoCreateBranchParams, RepoCreateCommitParams,
    RepoFileExistsParams, RepoInfo, RepoInfoParams, RepoListRefsParams, RepoType, UploadEvent,
    UploadPhase,
};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::fs;
use std::io::Write as IoWrite;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::cli::moe::{HfJobArgs, MoeAnalyzeCommand, MoeCommand};
use crate::cli::terminal_progress::{clear_stderr_line, start_spinner, SpinnerHandle};
use crate::cli::Cli;
use crate::inference::moe;
use crate::models;
use crate::system::moe_planner::{self, MoePlanArgs};

use formatters::moe_plan_formatter;

const MICRO_PROMPTS: &[&str] = &[
    "Write a concise explanation of how a rainbow forms.",
    "Summarize the causes and effects of inflation in a paragraph.",
    "Explain why distributed systems are hard to debug.",
    "Give three practical tips for writing reliable shell scripts.",
    "Describe the water cycle for a middle school student.",
    "Compare TCP and QUIC in two short paragraphs.",
    "Explain the difference between RAM and disk storage.",
    "Write a short answer on why model evaluation matters.",
];

const SHARE_UPLOAD_BATCH_MAX_FILES: usize = 8;
const SHARE_UPLOAD_BATCH_MAX_BYTES: u64 = 1_500_000_000;
const SHARE_UPLOAD_STALL_TIMEOUT: Duration = Duration::from_secs(180);
const SHARE_UPLOAD_POLL_INTERVAL: Duration = Duration::from_secs(5);
const SHARE_UPLOAD_MAX_RETRIES: usize = 3;
const SHARE_REPO_READY_TIMEOUT: Duration = Duration::from_secs(30);
const SHARE_REPO_READY_POLL_INTERVAL: Duration = Duration::from_millis(750);
const FULL_ANALYZE_CONTEXT_SIZE: u32 = 4096;
const FULL_ANALYZE_GPU_LAYERS: u32 = 0;
const FULL_ANALYZE_TOKEN_COUNT: u32 = 32;

struct TempRootGuard(PathBuf);

#[derive(Clone, Debug)]
struct SharePublishTarget {
    package_repo: String,
    publisher: String,
    trust: &'static str,
}

impl Drop for TempRootGuard {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

struct ShareUploadFileState {
    bytes_completed: u64,
    total_bytes: u64,
}

#[derive(Clone, Debug)]
struct StagedUploadFile {
    repo_path: String,
    local_path: PathBuf,
    size_bytes: u64,
}

#[derive(Clone, Debug)]
struct ShareUploadBatch {
    files: Vec<StagedUploadFile>,
    total_bytes: u64,
}

struct ShareUploadProgressState {
    spinner: Option<SpinnerHandle>,
    phase: Option<UploadPhase>,
    overall_total_files: usize,
    overall_total_bytes: u64,
    prior_completed_files: usize,
    prior_completed_bytes: u64,
    current_batch_index: usize,
    total_batches: usize,
    current_batch_total_files: usize,
    current_batch_total_bytes: u64,
    batch_bytes_completed: u64,
    bytes_per_sec: Option<f64>,
    batch_transfer_bytes_completed: u64,
    batch_transfer_bytes: u64,
    transfer_bytes_per_sec: Option<f64>,
    completed_files: BTreeSet<String>,
    active_files: BTreeMap<String, ShareUploadFileState>,
    last_draw: Option<std::time::Instant>,
    last_progress_change: std::time::Instant,
    last_progress_snapshot: (u64, u64, usize),
}

struct ShareUploadProgress {
    state: Mutex<ShareUploadProgressState>,
}

impl ShareUploadProgress {
    fn new(overall_total_files: usize, overall_total_bytes: u64) -> Self {
        Self {
            state: Mutex::new(ShareUploadProgressState {
                spinner: None,
                phase: None,
                overall_total_files,
                overall_total_bytes,
                prior_completed_files: 0,
                prior_completed_bytes: 0,
                current_batch_index: 0,
                total_batches: 0,
                current_batch_total_files: 0,
                current_batch_total_bytes: 0,
                batch_bytes_completed: 0,
                bytes_per_sec: None,
                batch_transfer_bytes_completed: 0,
                batch_transfer_bytes: 0,
                transfer_bytes_per_sec: None,
                completed_files: BTreeSet::new(),
                active_files: BTreeMap::new(),
                last_draw: None,
                last_progress_change: std::time::Instant::now(),
                last_progress_snapshot: (0, 0, 0),
            }),
        }
    }

    fn begin_batch(
        &self,
        batch_index: usize,
        total_batches: usize,
        prior_completed_files: usize,
        prior_completed_bytes: u64,
        batch: &ShareUploadBatch,
    ) {
        let Ok(mut state) = self.state.lock() else {
            return;
        };
        if let Some(mut spinner) = state.spinner.take() {
            spinner.finish();
        }
        state.phase = None;
        state.prior_completed_files = prior_completed_files;
        state.prior_completed_bytes = prior_completed_bytes;
        state.current_batch_index = batch_index;
        state.total_batches = total_batches;
        state.current_batch_total_files = batch.files.len();
        state.current_batch_total_bytes = batch.total_bytes;
        state.batch_bytes_completed = 0;
        state.bytes_per_sec = None;
        state.batch_transfer_bytes_completed = 0;
        state.batch_transfer_bytes = 0;
        state.transfer_bytes_per_sec = None;
        state.completed_files.clear();
        state.active_files.clear();
        state.last_draw = None;
        state.last_progress_change = std::time::Instant::now();
        state.last_progress_snapshot = (0, 0, 0);
    }

    fn transition_phase(state: &mut ShareUploadProgressState, phase: &UploadPhase) {
        if state.phase.as_ref() == Some(phase) {
            return;
        }
        if let Some(mut spinner) = state.spinner.take() {
            spinner.finish();
        }
        state.phase = Some(phase.clone());
        match phase {
            UploadPhase::Preparing => {
                state.spinner = Some(start_spinner(&format!(
                    "Preparing upload batch {}/{}",
                    state.current_batch_index, state.total_batches
                )));
            }
            UploadPhase::CheckingUploadMode => {
                state.spinner = Some(start_spinner(&format!(
                    "Checking upload mode for batch {}/{}",
                    state.current_batch_index, state.total_batches
                )));
            }
            UploadPhase::Uploading => {
                let _ = clear_stderr_line();
                eprintln!(
                    "⬆️ Uploading batch {}/{}...",
                    state.current_batch_index, state.total_batches
                );
            }
            UploadPhase::Committing => {
                let done = (state.prior_completed_files
                    + state
                        .current_batch_total_files
                        .min(state.completed_files.len()))
                .min(state.overall_total_files);
                state.spinner = Some(start_spinner(&format!(
                    "Creating contribution PR ({done}/{})",
                    state.overall_total_files
                )));
            }
        }
    }

    fn apply_file_progress(state: &mut ShareUploadProgressState, file: &FileProgress) {
        match file.status {
            FileStatus::Started | FileStatus::InProgress => {
                state.active_files.insert(
                    file.filename.clone(),
                    ShareUploadFileState {
                        bytes_completed: file.bytes_completed,
                        total_bytes: file.total_bytes,
                    },
                );
            }
            FileStatus::Complete => {
                state.completed_files.insert(file.filename.clone());
                state.active_files.remove(&file.filename);
            }
        }
    }

    fn draw(state: &mut ShareUploadProgressState, force: bool) {
        let now = std::time::Instant::now();
        if !force
            && state.last_draw.is_some_and(|last| {
                now.duration_since(last) < std::time::Duration::from_millis(700)
            })
        {
            return;
        }
        state.last_draw = Some(now);

        let batch_completed_files = if state.phase == Some(UploadPhase::Committing) {
            state.current_batch_total_files
        } else {
            state
                .completed_files
                .len()
                .min(state.current_batch_total_files)
        };
        let done =
            (state.prior_completed_files + batch_completed_files).min(state.overall_total_files);
        let percent = if state.overall_total_files == 0 {
            0.0
        } else {
            (done as f64 / state.overall_total_files as f64) * 100.0
        };
        let total_processed_bytes = state.prior_completed_bytes + state.batch_bytes_completed;
        let processing = if state.overall_total_bytes > 0 {
            format!(
                " processed {}/{}",
                format_share_bytes(total_processed_bytes),
                format_share_bytes(state.overall_total_bytes)
            )
        } else {
            String::new()
        };
        let transfer = if state.batch_transfer_bytes > 0 {
            format!(
                ", uploading {}/{}",
                format_share_bytes(state.batch_transfer_bytes_completed),
                format_share_bytes(state.batch_transfer_bytes)
            )
        } else {
            String::new()
        };
        let speed = state
            .transfer_bytes_per_sec
            .or(state.bytes_per_sec)
            .filter(|bytes_per_sec| *bytes_per_sec > 0.0)
            .map(|bytes_per_sec| format!(" at {}/s", format_share_bytes(bytes_per_sec as u64)))
            .unwrap_or_default();

        let _ = clear_stderr_line();
        eprintln!(
            "⬆️ Uploading batch {}/{} {:>5.1}% [{}/{} files]{}{}{}",
            state.current_batch_index,
            state.total_batches,
            percent,
            done,
            state.overall_total_files,
            processing,
            transfer,
            speed
        );

        let mut active_files: Vec<(&String, &ShareUploadFileState)> =
            state.active_files.iter().collect();
        active_files.sort_by(|(left_name, left_file), (right_name, right_file)| {
            right_file
                .bytes_completed
                .cmp(&left_file.bytes_completed)
                .then_with(|| right_file.total_bytes.cmp(&left_file.total_bytes))
                .then_with(|| left_name.cmp(right_name))
        });

        let active_count = active_files.len();
        for (name, file) in active_files.into_iter().take(8) {
            let file_percent = if file.total_bytes == 0 {
                0.0
            } else {
                (file.bytes_completed as f64 / file.total_bytes as f64) * 100.0
            };
            eprintln!(
                "   {} {:>5.1}% ({}/{})",
                display_upload_filename(name),
                file_percent,
                format_share_bytes(file.bytes_completed),
                format_share_bytes(file.total_bytes)
            );
        }
        if active_count > 8 {
            eprintln!("   … {} more tracked file(s)", active_count - 8);
        }
        let _ = std::io::stderr().flush();
    }

    fn note_progress_change(state: &mut ShareUploadProgressState) {
        let snapshot = (
            state.batch_bytes_completed,
            state.batch_transfer_bytes_completed,
            state.completed_files.len(),
        );
        if snapshot != state.last_progress_snapshot {
            state.last_progress_snapshot = snapshot;
            state.last_progress_change = std::time::Instant::now();
        }
    }

    fn stall_message(&self, timeout: Duration) -> Option<String> {
        let Ok(state) = self.state.lock() else {
            return None;
        };
        if state.phase != Some(UploadPhase::Uploading) {
            return None;
        }
        let stalled_for = std::time::Instant::now().duration_since(state.last_progress_change);
        if stalled_for < timeout {
            return None;
        }
        let active = state
            .active_files
            .keys()
            .take(3)
            .map(|path| display_upload_filename(path))
            .collect::<Vec<_>>();
        let active_suffix = if active.is_empty() {
            String::new()
        } else {
            format!(" Active files: {}.", active.join(", "))
        };
        Some(format!(
            "upload batch {}/{} stalled for {} with no byte progress.{}",
            state.current_batch_index,
            state.total_batches,
            format_duration(stalled_for),
            active_suffix
        ))
    }
}

impl ProgressHandler for ShareUploadProgress {
    fn on_progress(&self, event: &ProgressEvent) {
        let ProgressEvent::Upload(event) = event else {
            return;
        };
        let Ok(mut state) = self.state.lock() else {
            return;
        };
        match event {
            UploadEvent::Start { .. } => {}
            UploadEvent::Progress {
                phase,
                bytes_completed,
                total_bytes,
                bytes_per_sec,
                transfer_bytes_completed,
                transfer_bytes,
                transfer_bytes_per_sec,
                files,
            } => {
                Self::transition_phase(&mut state, phase);
                state.batch_bytes_completed = *bytes_completed;
                if *total_bytes > 0 {
                    state.current_batch_total_bytes = *total_bytes;
                }
                state.bytes_per_sec = *bytes_per_sec;
                state.batch_transfer_bytes_completed = *transfer_bytes_completed;
                if *transfer_bytes > 0 {
                    state.batch_transfer_bytes = *transfer_bytes;
                }
                state.transfer_bytes_per_sec = *transfer_bytes_per_sec;
                for file in files {
                    Self::apply_file_progress(&mut state, file);
                }
                Self::note_progress_change(&mut state);
                if *phase == UploadPhase::Uploading {
                    Self::draw(&mut state, false);
                }
            }
            UploadEvent::FileComplete { files, phase } => {
                Self::transition_phase(&mut state, phase);
                for name in files {
                    state.completed_files.insert(name.clone());
                    state.active_files.remove(name);
                }
                Self::note_progress_change(&mut state);
                if *phase == UploadPhase::Uploading {
                    Self::draw(&mut state, true);
                }
            }
            UploadEvent::Complete => {
                if let Some(mut spinner) = state.spinner.take() {
                    spinner.finish();
                }
                if state.current_batch_total_files > 0 {
                    let remaining: Vec<String> = state.active_files.keys().cloned().collect();
                    state.completed_files.extend(remaining);
                    state.active_files.clear();
                    state.batch_bytes_completed = state.current_batch_total_bytes;
                    state.batch_transfer_bytes_completed = state.batch_transfer_bytes;
                    Self::note_progress_change(&mut state);
                    Self::draw(&mut state, true);
                }
            }
        }
    }
}

impl Drop for ShareUploadProgress {
    fn drop(&mut self) {
        if let Ok(mut state) = self.state.lock() {
            if let Some(mut spinner) = state.spinner.take() {
                spinner.finish();
            }
        }
    }
}

fn display_upload_filename(path: &str) -> String {
    Path::new(path)
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.to_string())
}

fn format_share_bytes(bytes: u64) -> String {
    const KB: f64 = 1_000.0;
    const MB: f64 = 1_000_000.0;
    const GB: f64 = 1_000_000_000.0;

    if bytes >= 1_000_000_000 {
        format!("{:.1}GB", bytes as f64 / GB)
    } else if bytes >= 1_000_000 {
        format!("{:.0}MB", bytes as f64 / MB)
    } else if bytes >= 1_000 {
        format!("{:.0}KB", bytes as f64 / KB)
    } else {
        format!("{bytes}B")
    }
}

fn format_duration(duration: Duration) -> String {
    let secs = duration.as_secs();
    if secs >= 3600 {
        format!("{}h {}m", secs / 3600, (secs % 3600) / 60)
    } else if secs >= 60 {
        format!("{}m {}s", secs / 60, secs % 60)
    } else {
        format!("{secs}s")
    }
}

pub(crate) async fn dispatch_moe_command(command: &MoeCommand, cli: &Cli) -> Result<()> {
    match command {
        MoeCommand::Plan {
            model,
            ranking_file,
            json,
            max_vram,
            nodes,
            dataset_repo,
        } => {
            run_plan(
                model,
                ranking_file.as_deref(),
                *json,
                max_vram.or(cli.max_vram),
                *nodes,
                dataset_repo,
            )
            .await
        }
        MoeCommand::Analyze { command } => match command {
            MoeAnalyzeCommand::Full {
                model,
                context_size,
                n_gpu_layers,
            } => run_analyze_full(model, *context_size, *n_gpu_layers).await,
            MoeAnalyzeCommand::Micro {
                model,
                prompt_count,
                token_count,
                context_size,
                n_gpu_layers,
            } => {
                run_analyze_micro(
                    model,
                    *prompt_count,
                    *token_count,
                    *context_size,
                    *n_gpu_layers,
                )
                .await
            }
        },
        MoeCommand::Publish {
            model,
            catalog_repo,
            namespace,
            hf_job,
        } => run_publish(model, catalog_repo, namespace.as_deref(), hf_job).await,
    }
}

async fn run_plan(
    model: &str,
    ranking_file: Option<&Path>,
    json_output: bool,
    max_vram: Option<f64>,
    nodes: Option<usize>,
    dataset_repo: &str,
) -> Result<()> {
    if !json_output {
        eprintln!("📍 Resolving MoE model: {model}");
        if let Some(path) = ranking_file {
            eprintln!("📦 Using explicit ranking override: {}", path.display());
        } else {
            eprintln!("📦 Checking local MoE ranking cache...");
            eprintln!("☁️ Checking {dataset_repo} for published rankings...");
        }
    }
    let report = moe_planner::plan_moe(MoePlanArgs {
        model: model.to_string(),
        ranking_file: ranking_file.map(Path::to_path_buf),
        max_vram_gb: max_vram,
        nodes,
        dataset_repo: dataset_repo.to_string(),
        progress: !json_output,
    })
    .await?;
    moe_plan_formatter(json_output).render(&report)
}

async fn run_analyze_full(model: &str, context_size: u32, n_gpu_layers: u32) -> Result<()> {
    let resolved = moe_planner::resolve_model_context(model).await?;
    eprintln!("📍 Model: {}", resolved.display_name);
    eprintln!("🧠 Running full-v1 MoE analysis");
    let artifacts = run_local_full_analysis(&resolved, context_size, n_gpu_layers)?;
    println!("✅ Full MoE analysis complete");
    println!("  Ranking: {}", artifacts.ranking_path.display());
    println!("  Analysis: {}", artifacts.analysis_path.display());
    println!("  Log: {}", artifacts.log_path.display());
    println!(
        "  Package cache: {}",
        moe::package_cache_root_dir(&resolved.path).display()
    );
    println!(
        "  Variant cache: {}",
        moe::package_cache_variant_dir(&resolved.path).display()
    );
    print_submit_suggestion(&resolved.path);
    Ok(())
}

async fn run_analyze_micro(
    model: &str,
    prompt_count: usize,
    token_count: u32,
    context_size: u32,
    n_gpu_layers: u32,
) -> Result<()> {
    let resolved = moe_planner::resolve_model_context(model).await?;
    let prompt_count = prompt_count.clamp(1, MICRO_PROMPTS.len());
    let log_path = log_path_for(&resolved.path, "micro-v1");
    if let Some(parent) = log_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let binary = resolve_analyze_binary()?;
    let temp_root = std::env::temp_dir().join(format!(
        "mesh-llm-moe-micro-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
    ));
    fs::create_dir_all(&temp_root)?;
    let _temp_root_guard = TempRootGuard(temp_root.clone());

    eprintln!("📍 Model: {}", resolved.display_name);
    eprintln!(
        "🧠 Running micro-v1 MoE analysis with {} prompt(s), {} token(s)",
        prompt_count, token_count
    );
    let mut spinner = start_spinner("Running micro-v1 prompts");
    let mut logs = String::new();
    let mut totals: BTreeMap<u32, (f64, u64)> = BTreeMap::new();
    for (index, prompt) in MICRO_PROMPTS.iter().take(prompt_count).enumerate() {
        spinner.set_message(format!(
            "Running micro-v1 prompt {}/{}",
            index + 1,
            prompt_count
        ));
        let partial = temp_root.join(format!("prompt-{}.csv", index + 1));
        let command = vec![
            binary.to_string_lossy().to_string(),
            "-m".to_string(),
            resolved.path.display().to_string(),
            "--export-ranking".to_string(),
            partial.display().to_string(),
            "-n".to_string(),
            token_count.to_string(),
            "-c".to_string(),
            context_size.to_string(),
            "-ngl".to_string(),
            n_gpu_layers.to_string(),
            "--all-layers".to_string(),
            "-p".to_string(),
            (*prompt).to_string(),
        ];
        let output = Command::new(&binary)
            .args(&command[1..])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .with_context(|| {
                format!(
                    "Run micro-v1 prompt {} for {}",
                    index + 1,
                    resolved.path.display()
                )
            })?;
        writeln!(&mut logs, "$ {}", shell_join(&command)).ok();
        writeln!(&mut logs, "[prompt {}]\n{}\n", index + 1, prompt).ok();
        writeln!(
            &mut logs,
            "[stdout]\n{}\n[stderr]\n{}\n",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        )
        .ok();
        if !output.status.success() {
            spinner.finish();
            fs::write(&log_path, logs)?;
            bail!("MoE micro analysis failed. Log: {}", log_path.display());
        }
        for row in read_analyze_rows(&partial)? {
            let entry = totals.entry(row.expert_id).or_insert((0.0, 0));
            entry.0 += row.gate_mass;
            entry.1 += row.selection_count;
        }
    }
    spinner.finish();
    fs::write(&log_path, logs)?;
    let artifact = moe::SharedRankingArtifact {
        kind: moe::SharedRankingKind::MicroAnalyze,
        origin: moe::SharedRankingOrigin::LocalMicroAnalyze,
        ranking: totals.keys().copied().collect::<Vec<_>>(),
        micro_prompt_count: Some(prompt_count),
        micro_tokens: Some(token_count),
        micro_layer_scope: Some(moe::MoeMicroLayerScope::All),
    };
    let mut ranking = totals.into_iter().collect::<Vec<_>>();
    ranking.sort_by(|a, b| {
        b.1 .0
            .partial_cmp(&a.1 .0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    let artifact = moe::SharedRankingArtifact {
        ranking: ranking.iter().map(|(expert_id, _)| *expert_id).collect(),
        ..artifact
    };
    let wrote_cache = moe::cache_shared_ranking_if_stronger(&resolved.path, &artifact)?;
    let cache_path = moe::shared_ranking_cache_path(&resolved.path, &artifact);
    write_canonical_micro_ranking(
        &cache_path,
        &artifact,
        &ranking,
        ranking.iter().map(|(_, values)| values.0).sum::<f64>(),
    )?;
    let analysis_path = moe_planner::write_analysis_json(&resolved, &cache_path, "micro-v1")?;
    let ranking = moe_planner::ResolvedRanking {
        path: cache_path.clone(),
        metadata_path: None,
        analysis_path: Some(analysis_path.clone()),
        analyzer_id: "micro-v1".to_string(),
        source: moe_planner::RankingSource::LocalCache,
        reason: "local analysis artifact".to_string(),
    };
    sync_local_package_cache(&resolved, &ranking, Some(&log_path))?;
    println!("✅ Micro MoE analysis complete");
    println!("  Ranking: {}", cache_path.display());
    println!("  Analysis: {}", analysis_path.display());
    if !wrote_cache {
        println!(
            "  Note: A stronger or equivalent shared ranking already exists, so this micro-v1 result was not promoted as the preferred shared artifact."
        );
    }
    println!("  Log: {}", log_path.display());
    println!(
        "  Package cache: {}",
        moe::package_cache_root_dir(&resolved.path).display()
    );
    println!(
        "  Variant cache: {}",
        moe::package_cache_variant_dir(&resolved.path).display()
    );
    print_submit_suggestion(&resolved.path);
    Ok(())
}

fn write_canonical_micro_ranking(
    path: &Path,
    artifact: &moe::SharedRankingArtifact,
    ranking: &[(u32, (f64, u64))],
    total_mass_sum: f64,
) -> Result<()> {
    let mut output = String::new();
    writeln!(&mut output, "# mesh-llm-moe-ranking=v1").ok();
    writeln!(&mut output, "# ranking_kind={}", artifact.kind.label()).ok();
    writeln!(&mut output, "# ranking_origin={}", artifact.origin.label()).ok();
    if let Some(prompt_count) = artifact.micro_prompt_count {
        writeln!(&mut output, "# micro_prompt_count={prompt_count}").ok();
    }
    if let Some(tokens) = artifact.micro_tokens {
        writeln!(&mut output, "# micro_tokens={tokens}").ok();
    }
    if let Some(layer_scope) = artifact.micro_layer_scope {
        let scope = match layer_scope {
            moe::MoeMicroLayerScope::First => "first",
            moe::MoeMicroLayerScope::All => "all",
        };
        writeln!(&mut output, "# micro_layer_scope={scope}").ok();
    }
    writeln!(
        &mut output,
        "expert_id,total_mass,mass_fraction,selection_count"
    )
    .ok();
    for (expert_id, (gate_mass, selection_count)) in ranking {
        let mass_fraction = if total_mass_sum > 0.0 {
            gate_mass / total_mass_sum
        } else {
            0.0
        };
        writeln!(
            &mut output,
            "{expert_id},{gate_mass:.12},{mass_fraction:.12},{selection_count}"
        )
        .ok();
    }
    fs::write(path, output).with_context(|| format!("Write {}", path.display()))?;
    Ok(())
}

fn print_submit_suggestion(model_path: &Path) {
    let Some(identity) = models::huggingface_identity_for_path(model_path) else {
        return;
    };
    println!("📤 Contribute this ranking to mesh-llm so other users can reuse it:");
    println!("  mesh-llm moe publish '{}'", identity.distribution_ref());
}

struct FullAnalyzeArtifacts {
    ranking: moe_planner::ResolvedRanking,
    ranking_path: PathBuf,
    analysis_path: PathBuf,
    log_path: PathBuf,
}

fn full_analyze_artifacts(model: &moe_planner::MoeModelContext) -> FullAnalyzeArtifacts {
    let ranking_path = moe::ranking_cache_path(&model.path);
    let analysis_path = ranking_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join("analysis.json");
    let log_path = log_path_for(&model.path, "full-v1");
    FullAnalyzeArtifacts {
        ranking: moe_planner::ResolvedRanking {
            path: ranking_path.clone(),
            metadata_path: None,
            analysis_path: Some(analysis_path.clone()),
            analyzer_id: "full-v1".to_string(),
            source: moe_planner::RankingSource::LocalCache,
            reason: "local full analysis artifact".to_string(),
        },
        ranking_path,
        analysis_path,
        log_path,
    }
}

fn has_complete_full_analyze_artifacts(model: &moe_planner::MoeModelContext) -> bool {
    let artifacts = full_analyze_artifacts(model);
    artifacts.ranking_path.exists()
        && artifacts.analysis_path.exists()
        && artifacts.log_path.exists()
}

fn run_local_full_analysis(
    model: &moe_planner::MoeModelContext,
    context_size: u32,
    n_gpu_layers: u32,
) -> Result<FullAnalyzeArtifacts> {
    let artifacts = full_analyze_artifacts(model);
    let binary = resolve_analyze_binary()?;
    if let Some(parent) = artifacts.ranking_path.parent() {
        fs::create_dir_all(parent)?;
    }
    if let Some(parent) = artifacts.log_path.parent() {
        fs::create_dir_all(parent)?;
    }

    let command = vec![
        binary.to_string_lossy().to_string(),
        "-m".to_string(),
        model.path.display().to_string(),
        "--all-layers".to_string(),
        "--export-ranking".to_string(),
        artifacts.ranking_path.display().to_string(),
        "-n".to_string(),
        FULL_ANALYZE_TOKEN_COUNT.to_string(),
        "-c".to_string(),
        context_size.to_string(),
        "-ngl".to_string(),
        n_gpu_layers.to_string(),
    ];
    run_analyzer_command(&command, &artifacts.log_path, "full-v1")?;
    let analysis_path =
        moe_planner::write_analysis_json(model, &artifacts.ranking_path, "full-v1")?;
    let ranking = moe_planner::ResolvedRanking {
        analysis_path: Some(analysis_path.clone()),
        ..artifacts.ranking.clone()
    };
    sync_local_package_cache(model, &ranking, Some(&artifacts.log_path))?;
    Ok(FullAnalyzeArtifacts {
        ranking,
        ranking_path: artifacts.ranking_path,
        analysis_path,
        log_path: artifacts.log_path,
    })
}

fn sync_local_package_cache(
    model: &moe_planner::MoeModelContext,
    ranking: &moe_planner::ResolvedRanking,
    log_path: Option<&Path>,
) -> Result<moe_planner::MoeSubmitBundle> {
    let bundle = moe_planner::build_submit_bundle(model, ranking, log_path)?;

    let meshllm_path = moe::package_cache_meshllm_path(&model.path);
    let existing_meshllm = if meshllm_path.exists() {
        Some(
            fs::read_to_string(&meshllm_path)
                .with_context(|| format!("Read {}", meshllm_path.display()))?,
        )
    } else {
        None
    };
    let meshllm = moe_planner::build_meshllm_descriptor(
        existing_meshllm.as_deref(),
        model,
        &bundle.manifest_repo_path,
    )?;
    if let Some(parent) = meshllm_path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("Create {}", parent.display()))?;
    }
    fs::write(
        &meshllm_path,
        serde_json::to_string_pretty(&meshllm)? + "\n",
    )
    .with_context(|| format!("Write {}", meshllm_path.display()))?;

    let package_ranking_path = moe::package_cache_ranking_path(&model.path);
    if let Some(parent) = package_ranking_path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("Create {}", parent.display()))?;
    }
    if ranking.path != package_ranking_path {
        fs::copy(&ranking.path, &package_ranking_path).with_context(|| {
            format!(
                "Copy ranking {} to {}",
                ranking.path.display(),
                package_ranking_path.display()
            )
        })?;
    }

    let package_analysis_path = moe::package_cache_analysis_path(&model.path);
    moe_planner::write_package_analysis_json(model, ranking, &package_analysis_path)?;

    let package_log_path = moe::package_cache_run_log_path(&model.path);
    if let Some(log_path) = log_path.filter(|path| path.exists()) {
        if let Some(parent) = package_log_path.parent() {
            fs::create_dir_all(parent).with_context(|| format!("Create {}", parent.display()))?;
        }
        if log_path != package_log_path {
            fs::copy(log_path, &package_log_path).with_context(|| {
                format!(
                    "Copy run log {} to {}",
                    log_path.display(),
                    package_log_path.display()
                )
            })?;
        }
    } else if package_log_path.exists() {
        fs::remove_file(&package_log_path)
            .with_context(|| format!("Remove stale {}", package_log_path.display()))?;
    }

    Ok(bundle)
}

async fn run_publish(
    model: &str,
    catalog_repo: &str,
    namespace: Option<&str>,
    hf_job: &HfJobArgs,
) -> Result<()> {
    if hf_job.hf_job {
        return hf_jobs::submit_publish_job(model, catalog_repo, namespace, hf_job).await;
    }
    let share_error = |title: &str, detail: &str| -> anyhow::Error {
        eprintln!("❌ {title}");
        eprintln!("   {detail}");
        anyhow::anyhow!("{title}: {detail}")
    };

    let resolved = moe_planner::resolve_model_context(model).await?;
    let ranking = if has_complete_full_analyze_artifacts(&resolved) {
        eprintln!(
            "🧠 Reusing full-v1 MoE analysis from {}",
            full_analyze_artifacts(&resolved).ranking_path.display()
        );
        let artifacts = full_analyze_artifacts(&resolved);
        sync_local_package_cache(&resolved, &artifacts.ranking, Some(&artifacts.log_path))?;
        artifacts.ranking
    } else {
        eprintln!("🧠 Running full-v1 MoE analysis before publish");
        run_local_full_analysis(
            &resolved,
            FULL_ANALYZE_CONTEXT_SIZE,
            FULL_ANALYZE_GPU_LAYERS,
        )?
        .ranking
    };
    moe_planner::validate_ranking(&resolved, &ranking).with_context(|| {
        format!(
            "Validate ranking {} against model {}",
            ranking.path.display(),
            resolved.display_name
        )
    })?;
    let log_path = log_path_for(&resolved.path, &ranking.analyzer_id);
    let bundle = sync_local_package_cache(&resolved, &ranking, Some(log_path.as_path()))?;
    models::hf_token_override().ok_or_else(|| {
        share_error(
            "Missing Hugging Face token",
            "Set HF_TOKEN or HUGGING_FACE_HUB_TOKEN before running `mesh-llm moe publish`.",
        )
    })?;
    let api =
        models::build_hf_tokio_api(false).context("Build Hugging Face client for MoE publish")?;
    let publish_target = resolve_publish_target(&api, &resolved, namespace)
        .await
        .map_err(|err| {
            share_error(
                "Failed to resolve package repository target",
                &err.to_string(),
            )
        })?;
    ensure_repo_ready(&api, &publish_target.package_repo, RepoType::Model)
        .await
        .map_err(|err| {
            share_error(
                "Failed to prepare package repository",
                &format!(
                    "Ensure {} exists and is ready: {}",
                    publish_target.package_repo, err
                ),
            )
        })?;
    let (package_owner, package_name) = parse_repo_id(&publish_target.package_repo)?;
    let package_repo = api.model(package_owner, package_name);
    let package_info = match package_repo
        .info(
            &RepoInfoParams::builder()
                .revision("main".to_string())
                .build(),
        )
        .await
    {
        Ok(info) => Some(info),
        Err(HFError::RevisionNotFound { .. }) => None,
        Err(err) => {
            return Err(err).with_context(|| {
                format!(
                    "Fetch package repo info for {}",
                    publish_target.package_repo
                )
            });
        }
    };
    let (package_siblings, package_main_head) = if let Some(info) = package_info.as_ref() {
        repo_info_siblings_and_sha(info)?
    } else {
        (Vec::new(), None)
    };

    println!("📤 MoE package publish");
    println!("📦 {}", resolved.display_name);
    println!("   ranking: {}", ranking.path.display());
    println!("   source: {}", ranking.source.label());
    println!("📚 Catalog");
    println!("   repo: {catalog_repo}");
    println!("📦 Package repo");
    println!("   repo: {}", publish_target.package_repo);
    println!("   model_ref: {}", bundle.model_ref);
    println!("   trust: {}", publish_target.trust);
    println!("   variant: {}", bundle.variant);

    let temp_root = std::env::temp_dir().join(format!(
        "mesh-llm-moe-publish-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
    ));
    fs::create_dir_all(&temp_root)?;
    let _temp_root_guard = TempRootGuard(temp_root.clone());

    let existing_meshllm = if package_siblings
        .iter()
        .any(|entry| entry.rfilename == "meshllm.json")
    {
        Some(
            download_repo_text(&package_repo, "meshllm.json", "main")
                .await
                .with_context(|| {
                    format!(
                        "Download existing meshllm.json from {}",
                        publish_target.package_repo
                    )
                })?,
        )
    } else {
        None
    };
    let meshllm = moe_planner::build_meshllm_descriptor(
        existing_meshllm.as_deref(),
        &resolved,
        &bundle.manifest_repo_path,
    )?;
    let meshllm_cache_path = moe::package_cache_meshllm_path(&resolved.path);
    if let Some(parent) = meshllm_cache_path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(
        &meshllm_cache_path,
        serde_json::to_string_pretty(&meshllm)? + "\n",
    )
    .with_context(|| format!("Write {}", meshllm_cache_path.display()))?;
    let readme = build_package_readme(&meshllm, &publish_target.package_repo);

    let mut staged_files = Vec::new();
    stage_share_text(&temp_root, "README.md", &readme)?;
    staged_files.push(staged_upload_file(&temp_root, "README.md")?);
    stage_share_file(&temp_root, "meshllm.json", &meshllm_cache_path)?;
    staged_files.push(staged_upload_file(&temp_root, "meshllm.json")?);
    stage_share_file(
        &temp_root,
        &bundle.ranking_repo_path,
        &moe::package_cache_ranking_path(&resolved.path),
    )?;
    staged_files.push(staged_upload_file(&temp_root, &bundle.ranking_repo_path)?);
    stage_share_file(
        &temp_root,
        &bundle.analysis_repo_path,
        &moe::package_cache_analysis_path(&resolved.path),
    )?;
    staged_files.push(staged_upload_file(&temp_root, &bundle.analysis_repo_path)?);
    if let Some(log_repo_path) = bundle.log_repo_path.as_ref() {
        let package_log_path = moe::package_cache_run_log_path(&resolved.path);
        if package_log_path.exists() {
            stage_share_file(&temp_root, log_repo_path, &package_log_path)?;
            staged_files.push(staged_upload_file(&temp_root, log_repo_path)?);
        }
    }

    let current_exe = std::env::current_exe().context("Failed to determine own binary path")?;
    let bin_dir = current_exe
        .parent()
        .ok_or_else(|| anyhow::anyhow!("Current executable has no parent directory"))?;
    for path in stage_variant_components(&temp_root, &resolved, &ranking, &bundle, bin_dir)? {
        staged_files.push(staged_upload_file(&temp_root, &path)?);
    }

    let branch_name = share_branch_name(&format!(
        "{}/{}",
        publish_target.package_repo, bundle.variant_root
    ));
    let mut upload_branch = branch_name.clone();
    let mut use_direct_main = false;
    let branch_state = if package_main_head.is_some() {
        load_share_branch_state(&package_repo, &branch_name, &staged_files)
            .await
            .with_context(|| {
                format!(
                    "Inspect contribution branch {} for {}",
                    branch_name, publish_target.package_repo
                )
            })?
    } else {
        None
    };
    let mut branch_head = if let Some(state) = &branch_state {
        println!("🌿 Resuming contribution branch");
        println!("   branch: {branch_name}");
        println!(
            "   staged remotely: {}/{}",
            state.uploaded_paths.len(),
            staged_files.len()
        );
        state.head_commit.clone()
    } else if let Some(main_head) = package_main_head.clone() {
        println!("🌿 Creating contribution branch");
        println!("   branch: {branch_name}");
        package_repo
            .create_branch(
                &RepoCreateBranchParams::builder()
                    .branch(branch_name.clone())
                    .revision(main_head.clone())
                    .build(),
            )
            .await
            .map_err(|err| {
                share_error(
                    "Failed to create contribution branch",
                    &format!(
                        "Create branch {branch_name} on {}: {}",
                        publish_target.package_repo, err
                    ),
                )
            })?;
        main_head
    } else {
        use_direct_main = true;
        upload_branch = "main".to_string();
        println!("🌱 Initializing empty package repo");
        println!("   repo: {}", publish_target.package_repo);
        "main".to_string()
    };

    let mut completed_paths = branch_state
        .as_ref()
        .map(|state| state.uploaded_paths.clone())
        .unwrap_or_default();
    let mut completed_bytes = staged_files
        .iter()
        .filter(|file| completed_paths.contains(&file.repo_path))
        .map(|file| file.size_bytes)
        .sum::<u64>();
    let pending_files = staged_files
        .iter()
        .filter(|file| !completed_paths.contains(&file.repo_path))
        .cloned()
        .collect::<Vec<_>>();

    if pending_files.is_empty() {
        println!("✅ Contribution branch already contains all staged files");
        println!("   branch: {upload_branch}");
        return Ok(());
    }

    let batches = build_upload_batches(&pending_files);
    let progress = Arc::new(ShareUploadProgress::new(
        staged_files.len(),
        staged_files.iter().map(|file| file.size_bytes).sum(),
    ));
    let mut final_commit: Option<CommitInfo> = None;
    println!(
        "⬆️ Opening contribution PR in {} upload batch(es)...",
        batches.len()
    );
    for (index, batch) in batches.iter().enumerate() {
        progress.begin_batch(
            index + 1,
            batches.len(),
            completed_paths.len(),
            completed_bytes,
            batch,
        );
        let batch_commit = upload_share_batch_with_retry(
            &package_repo,
            &upload_branch,
            if use_direct_main && index == 0 {
                None
            } else {
                Some(branch_head.clone())
            },
            batch,
            &progress,
            batch_commit_message(&bundle.commit_message, index + 1, batches.len()),
            batch_commit_description(&bundle.commit_description, index + 1, batches.len()),
            !use_direct_main && index == 0,
        )
        .await
        .map_err(|err| {
            share_error(
                "Package upload failed",
                &format!(
                    "Upload staged files to {}: {}",
                    publish_target.package_repo, err
                ),
            )
        })?;
        for file in &batch.files {
            completed_paths.insert(file.repo_path.clone());
        }
        completed_bytes += batch.total_bytes;
        if let Some(commit_oid) = batch_commit.commit_oid.clone() {
            branch_head = commit_oid;
        }
        final_commit = Some(match (final_commit.take(), batch_commit.pr_url.clone()) {
            (None, _) => batch_commit,
            (Some(mut current), Some(pr_url)) => {
                current.pr_url = Some(pr_url);
                if current.pr_num.is_none() {
                    current.pr_num = batch_commit.pr_num;
                }
                if batch_commit.commit_oid.is_some() {
                    current.commit_oid = batch_commit.commit_oid;
                }
                if batch_commit.commit_url.is_some() {
                    current.commit_url = batch_commit.commit_url;
                }
                current
            }
            (Some(mut current), None) => {
                if batch_commit.commit_oid.is_some() {
                    current.commit_oid = batch_commit.commit_oid;
                }
                if batch_commit.commit_url.is_some() {
                    current.commit_url = batch_commit.commit_url;
                }
                current
            }
        });
    }

    let commit = final_commit.unwrap_or(CommitInfo {
        commit_url: None,
        commit_message: Some(bundle.commit_message.clone()),
        commit_description: Some(bundle.commit_description.clone()),
        commit_oid: Some(branch_head.clone()),
        pr_url: None,
        pr_num: None,
    });
    if use_direct_main {
        println!("✅ Published MoE package");
        println!("   branch: main");
    } else {
        println!("✅ Opened MoE package contribution");
        println!("   branch: {branch_name}");
    }
    if let Some(commit_oid) = commit.commit_oid.as_deref() {
        println!("   commit: {commit_oid}");
    }
    if let Some(commit_url) = commit.commit_url.as_deref() {
        println!("   url: {commit_url}");
    }
    if let Some(pr_url) = commit.pr_url.as_deref() {
        println!("   pr: {pr_url}");
    }

    let package_revision = commit
        .commit_oid
        .clone()
        .unwrap_or_else(|| branch_head.clone());
    let catalog_path = moe_planner::catalog_entry_path_for_source_repo(
        resolved.source_repo.as_deref().ok_or_else(|| {
            anyhow::anyhow!("Resolved model is missing a source repo for catalog publication")
        })?,
    )?;
    let source_file = resolved
        .path
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| anyhow::anyhow!("Resolved model path has no file name"))?;
    let catalog_source = crate::models::catalog::CatalogSource {
        repo: resolved
            .source_repo
            .clone()
            .ok_or_else(|| anyhow::anyhow!("Resolved model is missing a source repo"))?,
        revision: resolved
            .source_revision
            .clone()
            .ok_or_else(|| anyhow::anyhow!("Resolved model is missing a source revision"))?,
        file: source_file.to_string(),
    };
    let package_pointer = crate::models::catalog::CatalogPackagePointer {
        package_repo: publish_target.package_repo.clone(),
        package_revision: package_revision.clone(),
        publisher: publish_target.publisher.clone(),
        trust: publish_target.trust.to_string(),
    };
    let catalog_commit = contribute_catalog_entry(
        &api,
        catalog_repo,
        &catalog_path,
        &bundle.variant,
        catalog_source,
        package_pointer,
    )
    .await
    .map_err(|err| {
        share_error(
            "Catalog contribution failed",
            &format!("Update {}: {}", catalog_repo, err),
        )
    })?;
    println!("✅ Opened catalog contribution");
    println!("   repo: {catalog_repo}");
    println!("   entry: {catalog_path}");
    if let Some(pr_url) = catalog_commit.pr_url.as_deref() {
        println!("   pr: {pr_url}");
    } else if let Some(commit_url) = catalog_commit.commit_url.as_deref() {
        println!("   url: {commit_url}");
    }
    Ok(())
}

#[derive(Debug)]
struct ShareBranchState {
    head_commit: String,
    uploaded_paths: BTreeSet<String>,
}

async fn dataset_branch_head(
    dataset: &hf_hub::HFRepository,
    branch: &str,
) -> Result<Option<String>> {
    let refs = dataset
        .list_refs(&RepoListRefsParams::builder().build())
        .await?;
    Ok(refs
        .branches
        .into_iter()
        .find(|entry| entry.name == branch)
        .map(|entry| entry.target_commit))
}

async fn load_share_branch_state(
    dataset: &hf_hub::HFRepository,
    branch: &str,
    staged_files: &[StagedUploadFile],
) -> Result<Option<ShareBranchState>> {
    let Some(head_commit) = dataset_branch_head(dataset, branch).await? else {
        return Ok(None);
    };
    let mut spinner = start_spinner(&format!("Checking existing files on {branch}"));
    let mut uploaded_paths = BTreeSet::new();
    for (index, file) in staged_files.iter().enumerate() {
        spinner.set_message(format!(
            "Checking branch files {}/{}",
            index + 1,
            staged_files.len()
        ));
        if dataset
            .file_exists(
                &RepoFileExistsParams::builder()
                    .filename(file.repo_path.clone())
                    .revision(branch.to_string())
                    .build(),
            )
            .await?
        {
            uploaded_paths.insert(file.repo_path.clone());
        }
    }
    spinner.finish();
    Ok(Some(ShareBranchState {
        head_commit,
        uploaded_paths,
    }))
}

async fn upload_share_batch_with_retry(
    dataset: &hf_hub::HFRepository,
    branch: &str,
    parent_commit: Option<String>,
    batch: &ShareUploadBatch,
    progress: &Arc<ShareUploadProgress>,
    commit_message: String,
    commit_description: String,
    create_pr: bool,
) -> Result<CommitInfo> {
    let batch_paths = batch
        .files
        .iter()
        .map(|file| file.repo_path.clone())
        .collect::<BTreeSet<_>>();
    let operations = batch
        .files
        .iter()
        .map(|file| CommitOperation::Add {
            path_in_repo: file.repo_path.clone(),
            source: AddSource::File(file.local_path.clone()),
        })
        .collect::<Vec<_>>();
    let mut last_error = None;

    for attempt in 1..=SHARE_UPLOAD_MAX_RETRIES {
        let progress_handler: Progress = Some(progress.clone());
        let builder = RepoCreateCommitParams::builder()
            .operations(operations.clone())
            .commit_message(commit_message.clone())
            .commit_description(commit_description.clone())
            .revision(branch.to_string())
            .create_pr(create_pr)
            .progress(progress_handler);
        let params = if let Some(parent_commit) = parent_commit.clone() {
            builder.parent_commit(parent_commit).build()
        } else {
            builder.build()
        };
        match create_commit_with_stall_monitor(dataset, &params, progress).await {
            Ok(commit) => return Ok(commit),
            Err(err) => {
                let repo_not_ready = err
                    .downcast_ref::<HFError>()
                    .is_some_and(|hf| matches!(hf, HFError::RepoNotFound { .. }));
                last_error = Some(err);
                match load_share_branch_state(dataset, branch, &batch.files).await {
                    Ok(Some(state)) => {
                        if state.uploaded_paths == batch_paths {
                            return Ok(CommitInfo {
                                commit_url: None,
                                commit_message: Some(commit_message.clone()),
                                commit_description: Some(commit_description.clone()),
                                commit_oid: Some(state.head_commit),
                                pr_url: None,
                                pr_num: None,
                            });
                        }
                    }
                    Ok(None) => {}
                    Err(state_err)
                        if repo_not_ready
                            && state_err
                                .downcast_ref::<HFError>()
                                .is_some_and(|hf| matches!(hf, HFError::RepoNotFound { .. })) => {}
                    Err(state_err) => return Err(state_err),
                }
                if repo_not_ready {
                    tokio::time::sleep(SHARE_REPO_READY_POLL_INTERVAL).await;
                }
                if attempt < SHARE_UPLOAD_MAX_RETRIES {
                    eprintln!(
                        "↻ Retrying upload batch to {} (attempt {}/{})",
                        branch,
                        attempt + 1,
                        SHARE_UPLOAD_MAX_RETRIES
                    );
                    if repo_not_ready {
                        eprintln!("   repository is still propagating on the Hub");
                    }
                }
            }
        }
    }

    Err(last_error.unwrap_or_else(|| anyhow::anyhow!("upload batch failed")))
}

async fn create_commit_with_stall_monitor(
    dataset: &hf_hub::HFRepository,
    params: &RepoCreateCommitParams,
    progress: &ShareUploadProgress,
) -> Result<CommitInfo> {
    let upload = dataset.create_commit(params);
    tokio::pin!(upload);
    loop {
        tokio::select! {
            result = &mut upload => return result.map_err(Into::into),
            _ = tokio::time::sleep(SHARE_UPLOAD_POLL_INTERVAL) => {
                if let Some(message) = progress.stall_message(SHARE_UPLOAD_STALL_TIMEOUT) {
                    bail!("{message}");
                }
            }
        }
    }
}

fn build_upload_batches(files: &[StagedUploadFile]) -> Vec<ShareUploadBatch> {
    let mut batches = Vec::new();
    let mut current_files = Vec::new();
    let mut current_bytes = 0u64;

    for file in files.iter().cloned() {
        let would_overflow = !current_files.is_empty()
            && (current_files.len() >= SHARE_UPLOAD_BATCH_MAX_FILES
                || current_bytes.saturating_add(file.size_bytes) > SHARE_UPLOAD_BATCH_MAX_BYTES);
        if would_overflow {
            batches.push(ShareUploadBatch {
                files: current_files,
                total_bytes: current_bytes,
            });
            current_files = Vec::new();
            current_bytes = 0;
        }
        current_bytes = current_bytes.saturating_add(file.size_bytes);
        current_files.push(file);
        if current_files.len() >= SHARE_UPLOAD_BATCH_MAX_FILES
            || current_bytes >= SHARE_UPLOAD_BATCH_MAX_BYTES
        {
            batches.push(ShareUploadBatch {
                files: current_files,
                total_bytes: current_bytes,
            });
            current_files = Vec::new();
            current_bytes = 0;
        }
    }

    if !current_files.is_empty() {
        batches.push(ShareUploadBatch {
            files: current_files,
            total_bytes: current_bytes,
        });
    }
    batches
}

fn batch_commit_message(base: &str, batch_index: usize, total_batches: usize) -> String {
    if total_batches <= 1 {
        base.to_string()
    } else {
        format!("{base} (batch {batch_index}/{total_batches})")
    }
}

fn batch_commit_description(base: &str, batch_index: usize, total_batches: usize) -> String {
    if total_batches <= 1 {
        base.to_string()
    } else {
        format!("{base}\n\nUpload batch {batch_index}/{total_batches}.")
    }
}

fn share_branch_name(dataset_prefix: &str) -> String {
    let digest = hex::encode(Sha256::digest(dataset_prefix.as_bytes()));
    let hint = dataset_prefix
        .split('/')
        .rev()
        .take(3)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .map(sanitize_branch_component)
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join("-");
    format!("mesh-llm-moe-{}-{}", hint, &digest[..12])
}

fn sanitize_branch_component(input: &str) -> String {
    input
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
                ch
            } else {
                '-'
            }
        })
        .collect::<String>()
        .trim_matches('-')
        .to_string()
}

fn staged_upload_file(temp_root: &Path, relative_path: &str) -> Result<StagedUploadFile> {
    let local_path = temp_root.join(relative_path);
    let size_bytes = fs::metadata(&local_path)
        .with_context(|| format!("Read metadata for staged {}", relative_path))?
        .len();
    Ok(StagedUploadFile {
        repo_path: relative_path.to_string(),
        local_path,
        size_bytes,
    })
}

fn variant_component_paths(prefix: &str, expert_count: u32) -> Vec<String> {
    let mut paths = vec![
        format!("{prefix}/manifest.json"),
        format!("{prefix}/trunk.gguf"),
    ];
    for expert_id in 0..expert_count {
        paths.push(format!(
            "{prefix}/experts/{}",
            moe::expert_component_filename(expert_id, expert_count)
        ));
    }
    paths
}

fn stage_variant_components(
    temp_root: &Path,
    model: &moe_planner::MoeModelContext,
    ranking: &moe_planner::ResolvedRanking,
    bundle: &moe_planner::MoeSubmitBundle,
    bin_dir: &Path,
) -> Result<Vec<String>> {
    let cached_manifest_path = moe::package_cache_manifest_path(&model.path);
    if cached_manifest_path.exists() {
        let cached_manifest_text = fs::read_to_string(&cached_manifest_path)
            .with_context(|| format!("Read {}", cached_manifest_path.display()))?;
        let cached_manifest: moe_planner::MoePackageManifest =
            serde_json::from_str(&cached_manifest_text)
                .with_context(|| format!("Parse {}", cached_manifest_path.display()))?;
        if cached_manifest.ranking_sha256 == moe_planner::sha256_file(&ranking.path)?
            && cached_manifest.n_expert == model.expert_count
            && cached_manifest.n_expert_used == model.used_expert_count
            && moe::component_trunk_path(&model.path).exists()
            && (0..model.expert_count).all(|expert_id| {
                moe::component_expert_path(&model.path, expert_id, model.expert_count).exists()
            })
        {
            println!(
                "   reusing local package cache from {}",
                moe::package_cache_variant_dir(&model.path).display()
            );
            let manifest_repo_path = temp_root.join(&bundle.manifest_repo_path);
            if let Some(parent) = manifest_repo_path.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::copy(&cached_manifest_path, &manifest_repo_path)?;

            let trunk_repo_path = temp_root.join(format!("{}/trunk.gguf", bundle.variant_root));
            if let Some(parent) = trunk_repo_path.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::copy(moe::component_trunk_path(&model.path), &trunk_repo_path)?;

            for expert_id in 0..model.expert_count {
                let filename = moe::expert_component_filename(expert_id, model.expert_count);
                let repo_path =
                    temp_root.join(format!("{}/experts/{filename}", bundle.variant_root));
                if let Some(parent) = repo_path.parent() {
                    fs::create_dir_all(parent)?;
                }
                fs::copy(
                    moe::component_expert_path(&model.path, expert_id, model.expert_count),
                    &repo_path,
                )?;
            }

            return Ok(variant_component_paths(
                &bundle.variant_root,
                model.expert_count,
            ));
        }
    }

    println!("   extracting trunk");
    let trunk_path = moe::component_trunk_path(&model.path);
    moe::run_extract_trunk(bin_dir, &model.path, &trunk_path)?;

    let mut spinner = start_spinner("Extracting expert components");
    let mut expert_files = Vec::with_capacity(model.expert_count as usize);
    for expert_id in 0..model.expert_count {
        spinner.set_message(format!(
            "Extracting expert {}/{}",
            expert_id + 1,
            model.expert_count
        ));
        let filename = moe::expert_component_filename(expert_id, model.expert_count);
        let output_path = moe::component_expert_path(&model.path, expert_id, model.expert_count);
        moe::run_extract_expert(bin_dir, &model.path, expert_id, &output_path)?;
        expert_files.push(moe::ExpertComponentFile {
            path: format!("experts/{filename}"),
            sha256: moe_planner::sha256_file(&output_path)?,
            expert_id: Some(expert_id),
        });
    }
    spinner.finish();

    let manifest = moe_planner::MoePackageManifest {
        schema_version: 1,
        format: "meshllm-moe-components".to_string(),
        ranking_sha256: moe_planner::sha256_file(&ranking.path)?,
        n_expert: model.expert_count,
        n_expert_used: model.used_expert_count,
        min_experts_per_node: model.min_experts_per_node,
        trunk: moe::ExpertComponentFile {
            path: "trunk.gguf".to_string(),
            sha256: moe_planner::sha256_file(&trunk_path)?,
            expert_id: None,
        },
        experts: expert_files,
    };
    if let Some(parent) = cached_manifest_path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(
        &cached_manifest_path,
        serde_json::to_string_pretty(&manifest)? + "\n",
    )
    .with_context(|| format!("Write {}", cached_manifest_path.display()))?;

    let manifest_repo_path = temp_root.join(&bundle.manifest_repo_path);
    if let Some(parent) = manifest_repo_path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::copy(&cached_manifest_path, &manifest_repo_path)?;

    let trunk_repo_path = temp_root.join(format!("{}/trunk.gguf", bundle.variant_root));
    if let Some(parent) = trunk_repo_path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::copy(&trunk_path, &trunk_repo_path)?;

    for expert_id in 0..model.expert_count {
        let filename = moe::expert_component_filename(expert_id, model.expert_count);
        let repo_path = temp_root.join(format!("{}/experts/{filename}", bundle.variant_root));
        if let Some(parent) = repo_path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::copy(
            moe::component_expert_path(&model.path, expert_id, model.expert_count),
            &repo_path,
        )?;
    }
    Ok(variant_component_paths(
        &bundle.variant_root,
        model.expert_count,
    ))
}

async fn resolve_publish_target(
    api: &hf_hub::HFClient,
    model: &moe_planner::MoeModelContext,
    namespace: Option<&str>,
) -> Result<SharePublishTarget> {
    let whoami = api.whoami().await.context("Fetch Hugging Face identity")?;
    let publisher = whoami.username;
    let repo_name = moe_planner::default_package_repo_name_for_model(model)?;

    if let Some(namespace) = namespace {
        let package_repo = format!("{namespace}/{repo_name}");
        ensure_repo_exists(api, &package_repo, RepoType::Model)
            .await
            .with_context(|| {
                format!(
                    "Access or create model repo {}. Check that your Hugging Face token can create or write model repos in namespace {}.",
                    package_repo, namespace
                )
            })?;
        return Ok(SharePublishTarget {
            package_repo,
            publisher,
            trust: if namespace == "meshllm" {
                "canonical"
            } else {
                "community"
            },
        });
    }

    let canonical_repo = format!("meshllm/{repo_name}");
    match ensure_repo_exists(api, &canonical_repo, RepoType::Model).await {
        Ok(()) => Ok(SharePublishTarget {
            package_repo: canonical_repo,
            publisher,
            trust: "canonical",
        }),
        Err(err)
            if err
                .downcast_ref::<HFError>()
                .is_some_and(|hf| matches!(hf, HFError::Forbidden | HFError::AuthRequired)) =>
        {
            let package_repo = format!("{}/{}", publisher, repo_name);
            eprintln!(
                "↪ No permission to publish in meshllm. Falling back to community package repo {}",
                package_repo
            );
            ensure_repo_exists(api, &package_repo, RepoType::Model)
                .await
                .with_context(|| {
                    format!(
                        "Access or create fallback model repo {}. Check that your Hugging Face token can create or write model repos in namespace {}.",
                        package_repo, publisher
                    )
                })?;
            Ok(SharePublishTarget {
                package_repo,
                publisher,
                trust: "community",
            })
        }
        Err(err) => Err(err.into()),
    }
}

async fn ensure_repo_exists(
    api: &hf_hub::HFClient,
    repo_id: &str,
    repo_type: RepoType,
) -> Result<()> {
    if repo_exists(api, repo_id, repo_type).await? {
        return Ok(());
    }

    match api
        .create_repo(
            &CreateRepoParams::builder()
                .repo_id(repo_id.to_string())
                .repo_type(repo_type)
                .exist_ok(true)
                .build(),
        )
        .await
    {
        Ok(_) => Ok(()),
        Err(err @ (HFError::Forbidden | HFError::AuthRequired)) => {
            if repo_exists(api, repo_id, repo_type).await? {
                Ok(())
            } else {
                Err(anyhow::Error::from(err))
            }
        }
        Err(err) => Err(err.into()),
    }
}

async fn ensure_repo_ready(
    api: &hf_hub::HFClient,
    repo_id: &str,
    repo_type: RepoType,
) -> Result<()> {
    ensure_repo_exists(api, repo_id, repo_type).await?;
    let started = std::time::Instant::now();
    loop {
        match repo_exists(api, repo_id, repo_type).await {
            Ok(true) => return Ok(()),
            Ok(false) => {
                if started.elapsed() >= SHARE_REPO_READY_TIMEOUT {
                    bail!(
                        "Repository {} still is not visible after {:.0}s",
                        repo_id,
                        SHARE_REPO_READY_TIMEOUT.as_secs_f64()
                    );
                }
            }
            Err(err) => return Err(err),
        }
        tokio::time::sleep(SHARE_REPO_READY_POLL_INTERVAL).await;
    }
}

async fn repo_exists(api: &hf_hub::HFClient, repo_id: &str, repo_type: RepoType) -> Result<bool> {
    let (owner, name) = parse_repo_id(repo_id)?;
    let params = RepoInfoParams::builder()
        .revision("main".to_string())
        .build();
    let result = match repo_type {
        RepoType::Model => api.model(owner, name).info(&params).await,
        RepoType::Dataset => api.dataset(owner, name).info(&params).await,
        RepoType::Space => api.space(owner, name).info(&params).await,
        RepoType::Kernel => {
            return Err(anyhow::anyhow!(
                "Kernel repositories are not supported for MoE package publication"
            ));
        }
    };
    match result {
        Ok(_) => Ok(true),
        Err(HFError::RepoNotFound { .. }) => Ok(false),
        Err(HFError::RevisionNotFound { .. }) => Ok(true),
        Err(err) => Err(err.into()),
    }
}

fn repo_info_siblings_and_sha(
    info: &RepoInfo,
) -> Result<(Vec<hf_hub::RepoSibling>, Option<String>)> {
    match info {
        RepoInfo::Model(info) => Ok((info.siblings.clone().unwrap_or_default(), info.sha.clone())),
        RepoInfo::Dataset(info) => {
            Ok((info.siblings.clone().unwrap_or_default(), info.sha.clone()))
        }
        RepoInfo::Space(info) => Ok((info.siblings.clone().unwrap_or_default(), info.sha.clone())),
    }
}

async fn download_repo_text(
    repo: &hf_hub::HFRepository,
    path: &str,
    revision: &str,
) -> Result<String> {
    let downloaded = repo
        .download_file(
            &hf_hub::RepoDownloadFileParams::builder()
                .filename(path.to_string())
                .revision(revision.to_string())
                .build(),
        )
        .await
        .with_context(|| format!("Download {}", path))?;
    fs::read_to_string(&downloaded).with_context(|| format!("Read {}", downloaded.display()))
}

fn build_package_readme(meshllm: &moe_planner::MeshllmPackageJson, package_repo: &str) -> String {
    let mut out = String::new();
    let _ = writeln!(&mut out, "# Mesh-LLM MoE Package");
    let _ = writeln!(&mut out);
    let _ = writeln!(
        &mut out,
        "Derived Mesh-LLM MoE package for `{}`.",
        meshllm.source.repo
    );
    let _ = writeln!(&mut out, "Published in `{}`.", package_repo);
    let _ = writeln!(&mut out);
    let _ = writeln!(&mut out, "## Source");
    let _ = writeln!(&mut out);
    let _ = writeln!(&mut out, "- repo: `{}`", meshllm.source.repo);
    let _ = writeln!(&mut out, "- revision: `{}`", meshllm.source.revision);
    let _ = writeln!(&mut out);
    let _ = writeln!(&mut out, "## Variants");
    let _ = writeln!(&mut out);
    for (variant, entry) in &meshllm.variants {
        let _ = writeln!(
            &mut out,
            "- `{}`: `{}` (`{}`)",
            variant, entry.manifest, entry.distribution_id
        );
    }
    out
}

async fn contribute_catalog_entry(
    api: &hf_hub::HFClient,
    catalog_repo: &str,
    repo_path: &str,
    variant: &str,
    source: crate::models::catalog::CatalogSource,
    package_pointer: crate::models::catalog::CatalogPackagePointer,
) -> Result<CommitInfo> {
    let (owner, name) = parse_repo_id(catalog_repo)?;
    let dataset = api.dataset(owner, name);
    let mut spinner = start_spinner(&format!("Opening catalog PR in {}", catalog_repo));
    let source_repo = source.repo.clone();
    let info = dataset
        .info(
            &RepoInfoParams::builder()
                .revision("main".to_string())
                .build(),
        )
        .await
        .with_context(|| format!("Fetch main branch info for {}", catalog_repo))?;
    let (siblings, main_head) = repo_info_siblings_and_sha(&info)?;
    let mut entry = if siblings
        .iter()
        .any(|sibling| sibling.rfilename == repo_path)
    {
        let existing = download_repo_text(&dataset, repo_path, "main")
            .await
            .with_context(|| format!("Download existing catalog entry {}", repo_path))?;
        let parsed: crate::models::catalog::CatalogRepoEntry = serde_json::from_str(&existing)
            .with_context(|| format!("Parse existing catalog entry {}", repo_path))?;
        anyhow::ensure!(
            parsed.source_repo == source.repo,
            "Catalog entry {} belongs to {}, expected {}",
            repo_path,
            parsed.source_repo,
            source.repo
        );
        parsed
    } else {
        crate::models::catalog::CatalogRepoEntry {
            schema_version: 1,
            source_repo: source.repo.clone(),
            variants: BTreeMap::new(),
        }
    };
    let variant_entry = entry
        .variants
        .entry(variant.to_string())
        .or_insert_with(|| crate::models::catalog::CatalogVariantEntry {
            source: source.clone(),
            curated: None,
            packages: Vec::new(),
        });
    variant_entry.source = source;
    variant_entry
        .packages
        .retain(|existing| existing.package_repo != package_pointer.package_repo);
    variant_entry.packages.push(package_pointer.clone());
    moe_planner::sort_catalog_package_pointers(&mut variant_entry.packages);
    let temp_root = TempRootGuard(make_temp_root("mesh-llm-catalog")?);
    stage_share_text(
        &temp_root.0,
        repo_path,
        &(serde_json::to_string_pretty(&entry)? + "\n"),
    )?;
    let builder = RepoCreateCommitParams::builder()
        .operations(vec![CommitOperation::Add {
            path_in_repo: repo_path.to_string(),
            source: AddSource::File(temp_root.0.join(repo_path)),
        }])
        .commit_message(format!(
            "Register {} package for {}:{}",
            package_pointer.trust, source_repo, variant
        ))
        .commit_description(format!(
            "Register `{}` as the {} package for `{}` variant `{}`.",
            package_pointer.package_repo, package_pointer.trust, source_repo, variant
        ))
        .revision("main".to_string())
        .create_pr(true);
    let params = if let Some(main_head) = main_head {
        builder.parent_commit(main_head).build()
    } else {
        builder.build()
    };
    let result = dataset
        .create_commit(&params)
        .await
        .map_err(anyhow::Error::from);
    spinner.finish();
    result
}

fn parse_repo_id(repo_id: &str) -> Result<(&str, &str)> {
    repo_id
        .split_once('/')
        .ok_or_else(|| anyhow::anyhow!("Repository id must look like `owner/name`, got {repo_id}"))
}

fn make_temp_root(prefix: &str) -> Result<PathBuf> {
    let temp_root = std::env::temp_dir().join(format!(
        "{}-{}-{}",
        prefix,
        std::process::id(),
        SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos()
    ));
    fs::create_dir_all(&temp_root)?;
    Ok(temp_root)
}

fn stage_share_text(temp_root: &Path, relative_path: &str, content: &str) -> Result<()> {
    let target = temp_root.join(relative_path);
    if let Some(parent) = target.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(target, content).with_context(|| format!("Write staged {}", relative_path))?;
    Ok(())
}

fn stage_share_file(temp_root: &Path, relative_path: &str, source: &Path) -> Result<()> {
    let target = temp_root.join(relative_path);
    if let Some(parent) = target.parent() {
        fs::create_dir_all(parent)?;
    }
    if target.exists() {
        fs::remove_file(&target).with_context(|| format!("Remove staged {}", target.display()))?;
    }
    match fs::hard_link(source, &target) {
        Ok(()) => Ok(()),
        Err(_) => {
            fs::copy(source, &target)
                .with_context(|| format!("Copy {} to {}", source.display(), target.display()))?;
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn variant_component_paths_include_manifest_trunk_and_experts() {
        assert_eq!(
            variant_component_paths("variants/Q4_K_XL", 3),
            vec![
                "variants/Q4_K_XL/manifest.json".to_string(),
                "variants/Q4_K_XL/trunk.gguf".to_string(),
                "variants/Q4_K_XL/experts/expert-000.gguf".to_string(),
                "variants/Q4_K_XL/experts/expert-001.gguf".to_string(),
                "variants/Q4_K_XL/experts/expert-002.gguf".to_string(),
            ]
        );
    }

    #[test]
    fn catalog_entry_path_uses_source_repo() {
        assert_eq!(
            moe_planner::catalog_entry_path_for_source_repo("unsloth/Qwen3.6-35B-A3B-GGUF")
                .unwrap(),
            "entries/unsloth/Qwen3.6-35B-A3B-GGUF.json"
        );
    }

    #[test]
    fn parse_repo_id_requires_owner_and_name() {
        assert!(parse_repo_id("meshllm/catalog").is_ok());
        assert!(parse_repo_id("invalid").is_err());
    }

    #[test]
    fn build_upload_batches_splits_large_batches() {
        let files = (0..11)
            .map(|index| StagedUploadFile {
                repo_path: format!("file-{index}.gguf"),
                local_path: PathBuf::from(format!("/tmp/file-{index}.gguf")),
                size_bytes: 300_000_000,
            })
            .collect::<Vec<_>>();
        let batches = build_upload_batches(&files);
        assert_eq!(batches.len(), 3);
        assert_eq!(batches[0].files.len(), 5);
        assert_eq!(batches[1].files.len(), 5);
        assert_eq!(batches[2].files.len(), 1);
    }

    #[test]
    fn share_branch_name_is_stable_and_sanitized() {
        let branch = share_branch_name("meshllm/qwen3.6-35b-a3b-gguf-moe/variants/Q4_K_XL");
        assert!(branch.starts_with("mesh-llm-moe-"));
        assert!(!branch.contains('/'));
        assert_eq!(
            branch,
            share_branch_name("meshllm/qwen3.6-35b-a3b-gguf-moe/variants/Q4_K_XL")
        );
    }
}

fn resolve_analyze_binary() -> Result<PathBuf> {
    let exe = std::env::current_exe().context("Failed to determine own binary path")?;
    let bin_dir = exe
        .parent()
        .ok_or_else(|| anyhow::anyhow!("Current executable has no parent directory"))?;
    let candidates = [
        bin_dir.join("llama-moe-analyze"),
        bin_dir.join("../llama.cpp/build/bin/llama-moe-analyze"),
        bin_dir.join("../../llama.cpp/build/bin/llama-moe-analyze"),
        bin_dir.join("../../../llama.cpp/build/bin/llama-moe-analyze"),
    ];
    for candidate in candidates {
        if candidate.exists() {
            return Ok(candidate.canonicalize().unwrap_or(candidate));
        }
    }
    bail!(
        "llama-moe-analyze not found next to {} or nearby llama.cpp/build/bin directories",
        bin_dir.display()
    )
}

fn log_path_for(model_path: &Path, analyzer_id: &str) -> PathBuf {
    let stem = model_path
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("model");
    models::mesh_llm_cache_dir()
        .join("moe")
        .join("logs")
        .join(format!("{stem}.{analyzer_id}.log"))
}

fn run_analyzer_command(command: &[String], log_path: &Path, label: &str) -> Result<()> {
    let mut spinner = start_spinner(&format!("Running {label}"));
    let output = Command::new(&command[0])
        .args(&command[1..])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .with_context(|| format!("Run {}", command[0]))?;
    spinner.finish();
    fs::write(
        log_path,
        format!(
            "$ {}\n\n[stdout]\n{}\n[stderr]\n{}",
            shell_join(command),
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ),
    )?;
    if !output.status.success() {
        bail!(
            "MoE analysis failed. Log: {}. Cause: llama-moe-analyze exited with {}",
            log_path.display(),
            output.status
        );
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct AnalyzeRow {
    expert_id: u32,
    gate_mass: f64,
    selection_count: u64,
}

fn read_analyze_rows(path: &Path) -> Result<Vec<AnalyzeRow>> {
    let content = fs::read_to_string(path).with_context(|| format!("Read {}", path.display()))?;
    let mut rows = Vec::new();
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') || trimmed.starts_with("expert") {
            continue;
        }
        let parts = trimmed.split(',').map(str::trim).collect::<Vec<_>>();
        if parts.len() < 4 {
            continue;
        }
        rows.push(AnalyzeRow {
            expert_id: parts[0].parse()?,
            gate_mass: parts[1].parse()?,
            selection_count: parts[3].parse()?,
        });
    }
    Ok(rows)
}

fn shell_join(command: &[String]) -> String {
    command
        .iter()
        .map(|part| {
            if part.contains([' ', '\n', '\t', '"', '\'']) {
                format!("{:?}", part)
            } else {
                part.clone()
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}
