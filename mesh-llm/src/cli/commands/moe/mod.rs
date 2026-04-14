mod formatters;
mod formatters_console;
mod formatters_json;
mod hf_jobs;

use anyhow::{bail, Context, Result};
use base64::Engine as _;
use reqwest::StatusCode;
use serde::Deserialize;
use serde_json::json;
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::cli::moe::{HfJobArgs, MoeAnalyzeCommand, MoeCommand};
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

struct TempRootGuard(PathBuf);

impl Drop for TempRootGuard {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
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
                hf_job,
            } => run_analyze_full(model, *context_size, *n_gpu_layers, hf_job).await,
            MoeAnalyzeCommand::Micro {
                model,
                prompt_count,
                token_count,
                context_size,
                n_gpu_layers,
                hf_job,
            } => {
                run_analyze_micro(
                    model,
                    *prompt_count,
                    *token_count,
                    *context_size,
                    *n_gpu_layers,
                    hf_job,
                )
                .await
            }
        },
        MoeCommand::Share {
            model,
            ranking_file,
            dataset_repo,
        } => run_share(model, ranking_file.as_deref(), dataset_repo).await,
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

async fn run_analyze_full(
    model: &str,
    context_size: u32,
    n_gpu_layers: u32,
    hf_job: &HfJobArgs,
) -> Result<()> {
    if hf_job.hf_job {
        return hf_jobs::submit_full_analyze_job(model, context_size, n_gpu_layers, hf_job).await;
    }
    let resolved = moe_planner::resolve_model_context(model).await?;
    let output_path = moe::ranking_cache_path(&resolved.path);
    let log_path = log_path_for(&resolved.path, "full-v1");
    let binary = resolve_analyze_binary()?;
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)?;
    }
    if let Some(parent) = log_path.parent() {
        fs::create_dir_all(parent)?;
    }

    eprintln!("📍 Model: {}", resolved.display_name);
    eprintln!("🧠 Running full-v1 MoE analysis");
    let command = vec![
        binary.to_string_lossy().to_string(),
        "-m".to_string(),
        resolved.path.display().to_string(),
        "--all-layers".to_string(),
        "--export-ranking".to_string(),
        output_path.display().to_string(),
        "-n".to_string(),
        "32".to_string(),
        "-c".to_string(),
        context_size.to_string(),
        "-ngl".to_string(),
        n_gpu_layers.to_string(),
    ];
    run_analyzer_command(&command, &log_path, "full-v1")?;
    println!("✅ Full MoE analysis complete");
    println!("  Ranking: {}", output_path.display());
    println!("  Log: {}", log_path.display());
    print_submit_suggestion(&resolved.path);
    Ok(())
}

async fn run_analyze_micro(
    model: &str,
    prompt_count: usize,
    token_count: u32,
    context_size: u32,
    n_gpu_layers: u32,
    hf_job: &HfJobArgs,
) -> Result<()> {
    if hf_job.hf_job {
        return hf_jobs::submit_micro_analyze_job(
            model,
            prompt_count,
            token_count,
            context_size,
            n_gpu_layers,
            hf_job,
        )
        .await;
    }
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
    println!("✅ Micro MoE analysis complete");
    println!("  Ranking: {}", cache_path.display());
    if !wrote_cache {
        println!(
            "  Note: A stronger or equivalent shared ranking already exists, so this micro-v1 result was not promoted as the preferred shared artifact."
        );
    }
    println!("  Log: {}", log_path.display());
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
    println!("  mesh-llm moe share '{}'", identity.canonical_ref);
}

async fn run_share(model: &str, ranking_file: Option<&Path>, dataset_repo: &str) -> Result<()> {
    let share_error = |title: &str, detail: &str| -> anyhow::Error {
        eprintln!("❌ {title}");
        eprintln!("   {detail}");
        anyhow::anyhow!("{title}: {detail}")
    };

    let resolved = moe_planner::resolve_model_context(model).await?;
    let ranking = moe_planner::local_submit_ranking(&resolved, ranking_file)?;
    moe_planner::validate_ranking(&resolved, &ranking).with_context(|| {
        format!(
            "Validate ranking {} against model {}",
            ranking.path.display(),
            resolved.display_name
        )
    })?;
    let log_path = log_path_for(&resolved.path, &ranking.analyzer_id);
    let bundle = moe_planner::build_submit_bundle(&resolved, &ranking, Some(log_path.as_path()))?;
    let api = models::build_hf_api(false).context("Build Hugging Face client for MoE share")?;
    let dataset = api.repo(hf_hub::Repo::with_revision(
        dataset_repo.to_string(),
        hf_hub::RepoType::Dataset,
        "main".to_string(),
    ));
    let info = dataset
        .info()
        .with_context(|| format!("Fetch dataset info for {}", dataset_repo))?;
    let existing = bundle
        .dataset_paths
        .iter()
        .filter(|path| info.siblings.iter().any(|entry| &entry.rfilename == *path))
        .cloned()
        .collect::<Vec<_>>();

    println!("📤 MoE ranking share");
    println!("📦 {}", resolved.display_name);
    println!("   ranking: {}", ranking.path.display());
    println!("   source: {}", ranking.source.label());
    println!("☁️ Dataset contribution");
    println!("   repo: {dataset_repo}");
    println!("   prefix: {}", bundle.dataset_prefix);
    match classify_share_prefix(&bundle.dataset_paths, &existing) {
        SharePrefixState::AlreadyPublished(existing) => {
            println!("✅ Already published");
            for path in existing {
                println!("   existing: {path}");
            }
            return Ok(());
        }
        SharePrefixState::PartiallyPopulated(existing) => {
            return Err(share_error(
                "Remote artifact prefix is partially populated",
                &format!("{} already contains: {}", dataset_repo, existing.join(", ")),
            ));
        }
        SharePrefixState::New => {}
    }

    let token = models::hf_token_override().ok_or_else(|| {
        share_error(
            "Missing Hugging Face token",
            "Set HF_TOKEN or HUGGING_FACE_HUB_TOKEN before running `mesh-llm moe share`.",
        )
    })?;

    let mut operations = vec![ndjson_header(
        &bundle.commit_message,
        &bundle.commit_description,
    )];
    operations.push(ndjson_file_op(
        &format!("{}/ranking.csv", bundle.dataset_prefix),
        &fs::read(&bundle.ranking_path)
            .with_context(|| format!("Read {}", bundle.ranking_path.display()))?,
    ));
    operations.push(ndjson_file_op(
        &format!("{}/metadata.json", bundle.dataset_prefix),
        bundle.metadata_content.as_bytes(),
    ));
    if let Some(log_path) = bundle.log_path.as_ref() {
        operations.push(ndjson_file_op(
            &format!("{}/run.log", bundle.dataset_prefix),
            &fs::read(log_path).with_context(|| format!("Read {}", log_path.display()))?,
        ));
    }

    let endpoint = std::env::var("HF_ENDPOINT")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| "https://huggingface.co".to_string());
    let commit_url = format!(
        "{}/api/datasets/{}/commit/main",
        endpoint.trim_end_matches('/'),
        dataset_repo
    );
    let body = operations
        .into_iter()
        .map(|value| serde_json::to_string(&value))
        .collect::<std::result::Result<Vec<_>, _>>()?
        .join("\n")
        + "\n";

    println!("⬆️ Opening contribution PR...");
    let response = reqwest::Client::new()
        .post(&commit_url)
        .bearer_auth(token)
        .query(&[("create_pr", "1")])
        .header("Content-Type", "application/x-ndjson")
        .body(body)
        .send()
        .await
        .map_err(|err| {
            share_error(
                "Dataset contribution request failed",
                &format!("POST {}: {}", commit_url, err),
            )
        })?;
    if response.status() != StatusCode::OK {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(share_error(
            "Dataset contribution failed",
            &format!("{}: {}", status, body.trim()),
        ));
    }
    let commit: HfCommitResponse = response.json().await.map_err(|err| {
        share_error(
            "Could not decode Hugging Face response",
            &format!("{}", err),
        )
    })?;
    println!("✅ Opened MoE dataset contribution");
    println!("   commit: {}", commit.commit_oid);
    println!("   url: {}", commit.commit_url);
    if let Some(pr_url) = commit.pull_request_url.as_deref() {
        println!("   pr: {}", pr_url);
    }
    Ok(())
}

#[derive(Deserialize)]
struct HfCommitResponse {
    #[serde(rename = "commitOid")]
    commit_oid: String,
    #[serde(rename = "commitUrl")]
    commit_url: String,
    #[serde(rename = "pullRequestUrl")]
    pull_request_url: Option<String>,
}

fn ndjson_header(summary: &str, description: &str) -> serde_json::Value {
    json!({
        "key": "header",
        "value": {
            "summary": summary,
            "description": description,
        }
    })
}

fn ndjson_file_op(path_in_repo: &str, content: &[u8]) -> serde_json::Value {
    json!({
        "key": "file",
        "value": {
            "content": base64::engine::general_purpose::STANDARD.encode(content),
            "path": path_in_repo,
            "encoding": "base64",
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ndjson_header_uses_expected_shape() {
        let value = ndjson_header("summary", "description");
        assert_eq!(value["key"], "header");
        assert_eq!(value["value"]["summary"], "summary");
        assert_eq!(value["value"]["description"], "description");
    }

    #[test]
    fn ndjson_file_op_uses_base64_payload() {
        let value = ndjson_file_op("path/in/repo.txt", b"hello");
        assert_eq!(value["key"], "file");
        assert_eq!(value["value"]["path"], "path/in/repo.txt");
        assert_eq!(value["value"]["encoding"], "base64");
        assert_eq!(value["value"]["content"], "aGVsbG8=");
    }

    #[test]
    fn classify_share_prefix_distinguishes_new_existing_and_partial() {
        let all = vec![
            "a/ranking.csv".to_string(),
            "a/metadata.json".to_string(),
            "a/run.log".to_string(),
        ];
        assert_eq!(classify_share_prefix(&all, &[]), SharePrefixState::New);
        assert_eq!(
            classify_share_prefix(&all, &all),
            SharePrefixState::AlreadyPublished(all.clone())
        );
        assert_eq!(
            classify_share_prefix(&all, &all[..1]),
            SharePrefixState::PartiallyPopulated(vec!["a/ranking.csv".to_string()])
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

struct SpinnerHandle {
    done: Arc<AtomicBool>,
    message: Arc<Mutex<String>>,
    thread: Option<thread::JoinHandle<()>>,
}

impl SpinnerHandle {
    fn set_message(&self, message: impl Into<String>) {
        if let Ok(mut guard) = self.message.lock() {
            *guard = message.into();
        }
    }

    fn finish(&mut self) {
        self.done.store(true, Ordering::Relaxed);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
        eprint!("\r\x1b[2K");
        let _ = std::io::Write::flush(&mut std::io::stderr());
    }
}

impl Drop for SpinnerHandle {
    fn drop(&mut self) {
        self.finish();
    }
}

fn start_spinner(message: &str) -> SpinnerHandle {
    let done = Arc::new(AtomicBool::new(false));
    let done_thread = Arc::clone(&done);
    let message = Arc::new(Mutex::new(message.to_string()));
    let message_thread = Arc::clone(&message);
    let thread = thread::spawn(move || {
        let frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
        let mut index = 0usize;
        while !done_thread.load(Ordering::Relaxed) {
            let current = message_thread
                .lock()
                .map(|guard| guard.clone())
                .unwrap_or_else(|_| "Working".to_string());
            eprint!("\r{} {}\x1b[K", frames[index % frames.len()], current);
            let _ = std::io::Write::flush(&mut std::io::stderr());
            index += 1;
            thread::sleep(Duration::from_millis(120));
        }
    });
    SpinnerHandle {
        done,
        message,
        thread: Some(thread),
    }
}

#[derive(Debug, PartialEq, Eq)]
enum SharePrefixState {
    New,
    AlreadyPublished(Vec<String>),
    PartiallyPopulated(Vec<String>),
}

fn classify_share_prefix(dataset_paths: &[String], existing: &[String]) -> SharePrefixState {
    if existing.len() == dataset_paths.len() {
        SharePrefixState::AlreadyPublished(existing.to_vec())
    } else if !existing.is_empty() {
        SharePrefixState::PartiallyPopulated(existing.to_vec())
    } else {
        SharePrefixState::New
    }
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
