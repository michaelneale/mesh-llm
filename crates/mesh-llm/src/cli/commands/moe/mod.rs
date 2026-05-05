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
use std::time::{SystemTime, UNIX_EPOCH};

use crate::cli::moe::{HfJobArgs, MoeAnalyzeCommand, MoeCommand};
use crate::cli::terminal_progress::start_spinner;
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
                share,
                context_size,
                n_gpu_layers,
                hf_job,
            } => run_analyze_full(model, *share, *context_size, *n_gpu_layers, hf_job).await,
            MoeAnalyzeCommand::Micro {
                model,
                share,
                prompt_count,
                token_count,
                context_size,
                n_gpu_layers,
                hf_job,
            } => {
                run_analyze_micro(
                    model,
                    *share,
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
        MoeCommand::Migrate {
            source_model,
            nas_root,
            n_nodes,
            overlap,
            min_experts_per_node,
            ranking_file,
            dry_run,
            bin_dir,
        } => {
            run_migrate(MigrateArgs {
                source_model: source_model.clone(),
                nas_root: nas_root.clone(),
                n_nodes: *n_nodes,
                overlap: *overlap,
                min_experts_per_node: *min_experts_per_node,
                ranking_file: ranking_file.clone(),
                dry_run: *dry_run,
                bin_dir: bin_dir.clone(),
            })
            .await
        }
    }
}

struct MigrateArgs {
    source_model: PathBuf,
    nas_root: Option<PathBuf>,
    n_nodes: usize,
    overlap: u32,
    min_experts_per_node: Option<u32>,
    ranking_file: Option<PathBuf>,
    dry_run: bool,
    bin_dir: Option<PathBuf>,
}

async fn run_migrate(args: MigrateArgs) -> Result<()> {
    use crate::plugin::{self, MoeStorageConfig, MoeStorageMode};

    if !args.source_model.exists() {
        bail!(
            "source model {} does not exist",
            args.source_model.display()
        );
    }
    if args.n_nodes < 2 {
        bail!("--n-nodes must be >= 2 (got {})", args.n_nodes);
    }

    // Build a MoeStorageConfig for the duration of this command. Prefer the
    // explicit --nas-root; else fall back to the per-user config and env.
    let mut storage = plugin::MoeStorageConfig {
        mode: MoeStorageMode::Split,
        ..MoeStorageConfig::default()
    };
    if let Some(root) = args.nas_root.as_ref() {
        if !root.is_absolute() {
            bail!(
                "--nas-root must be an absolute path (got {})",
                root.display()
            );
        }
        storage.trunk_path = Some(root.join("mesh-llm").join("trunks"));
        storage.experts_path = Some(root.join("mesh-llm").join("experts"));
    } else {
        let cfg = plugin::load_config(None).with_context(|| "loading mesh-llm config")?;
        if matches!(cfg.moe.storage.mode, MoeStorageMode::Split) {
            storage.trunk_path = cfg.moe.storage.trunk_path.clone();
            storage.experts_path = cfg.moe.storage.experts_path.clone();
            storage.experts_local_override = cfg.moe.storage.experts_local_override.clone();
        }
        // Fall back to MESH_LLM_NAS_ROOT if config didn't provide paths.
        if storage.trunk_path.is_none() || storage.experts_path.is_none() {
            if let Some(root) = plugin::resolve_trunk_root(&storage)
                .or_else(|| plugin::resolve_experts_root(&storage))
            {
                // resolve_* already derives the `/mesh-llm/trunks` and
                // `/mesh-llm/experts` subpaths, so we re-use them directly.
                if storage.trunk_path.is_none() {
                    storage.trunk_path = plugin::resolve_trunk_root(&storage);
                }
                if storage.experts_path.is_none() {
                    storage.experts_path = plugin::resolve_experts_root(&storage);
                }
                let _ = root; // used above via resolve_*
            }
        }
    }
    // Point at exactly what is missing so users can self-heal.
    let trunk_missing = storage.trunk_path.is_none();
    let experts_missing = storage.experts_path.is_none();
    if trunk_missing || experts_missing {
        let missing = match (trunk_missing, experts_missing) {
            (true, true) => "both trunk and experts paths are unset",
            (true, false) => "trunk path is unset",
            (false, true) => "experts path is unset",
            (false, false) => unreachable!(),
        };
        bail!(
            "{missing}. Fix by ONE of:\n  \
             1. pass --nas-root <absolute-path> (derives <root>/mesh-llm/trunks and /experts),\n  \
             2. set moe.storage.trunk_path + moe.storage.experts_path in ~/.mesh-llm/config.toml,\n  \
             3. export MESH_LLM_NAS_ROOT=<absolute-path>"
        );
    }

    // Resolve a ranking + assignment plan for the source model. We cap
    // `nodes` to the requested `n_nodes` so the report matches what we
    // actually want to lay out on disk.
    let report = moe_planner::plan_moe(MoePlanArgs {
        model: args.source_model.display().to_string(),
        ranking_file: args.ranking_file.clone(),
        max_vram_gb: None,
        nodes: Some(args.n_nodes),
        dataset_repo: "meshllm/moe-rankings".to_string(),
        progress: true,
        min_experts_per_node_override: args.min_experts_per_node,
    })
    .await
    .with_context(|| "planning migration")?;

    if report.assignments.len() != args.n_nodes {
        bail!(
            "planner returned {} assignments for {} nodes; aborting",
            report.assignments.len(),
            args.n_nodes
        );
    }

    // Build the (node_index -> experts) map the splitter needs.
    let mut assignments_map = BTreeMap::new();
    for (i, a) in report.assignments.iter().enumerate() {
        assignments_map.insert(i, a.experts.clone());
    }

    // Report what we plan to do.
    let source_size = fs::metadata(&args.source_model)?.len();
    eprintln!("📍 Source model: {}", args.source_model.display());
    eprintln!(
        "📐 Mesh: {} nodes, overlap={}, experts={}, used={}",
        args.n_nodes,
        args.overlap,
        report.model.expert_count,
        report.model.used_expert_count
    );
    eprintln!(
        "🗂  Trunk root:   {}",
        storage.trunk_path.as_ref().unwrap().display()
    );
    eprintln!(
        "🗂  Experts root: {}",
        storage.experts_path.as_ref().unwrap().display()
    );

    // Rough size estimate: assume trunk fraction ≈ 1 - (expert_bytes /
    // total). We can't know the exact split without running the splitter;
    // report source size as an upper bound for the dry-run.
    let est_per_node = (source_size / args.n_nodes as u64).saturating_add(
        // trunk duplication cost eliminated by split, but we don't know
        // trunk size until after the split. Use 5% of source as a crude
        // placeholder so users see a meaningful number.
        source_size / 20,
    );
    eprintln!(
        "🧮 Expected per-node experts shard: ~{:.1} GB (upper bound; real size shown after split)",
        est_per_node as f64 / 1e9
    );
    eprintln!(
        "🧮 Expected monolithic per-node total: ~{:.1} GB × {} = ~{:.1} GB",
        source_size as f64 / 1e9,
        args.n_nodes,
        source_size as f64 * args.n_nodes as f64 / 1e9
    );
    if args.dry_run {
        eprintln!("(--dry-run: no files written)");
        return Ok(());
    }

    // Resolve the splitter binary dir.
    let bin_dir = args
        .bin_dir
        .clone()
        .unwrap_or_else(|| default_bin_dir().unwrap_or_else(|| PathBuf::from(".")));

    // Drive the splitter node-by-node. `ensure_trunk_and_expert` is
    // idempotent and holds the trunk build lock, so sequential calls are
    // safe AND cheap (trunk is written on the first pass; subsequent
    // passes detect the existing hash and skip).
    eprintln!("🚧 Migrating (node-by-node)...");
    let mut trunk_path_final = None;
    for node_idx in 0..args.n_nodes {
        let opts = moe::EnsureOpts {
            bin_dir: bin_dir.clone(),
            model_path: args.source_model.clone(),
            assignments: assignments_map.clone(),
            node_index: node_idx,
            // We always drive the migrate ourselves, so every node invocation
            // is allowed to build (is_leader=true). The flock inside
            // `ensure_trunk_and_expert` serializes concurrent writers.
            is_leader: true,
            follower_timeout: std::time::Duration::from_secs(5),
        };
        let resolved = moe::ensure_trunk_and_expert(&storage, &opts)
            .with_context(|| format!("migrating node {node_idx}"))?;
        let e_size = fs::metadata(&resolved.experts).map(|m| m.len()).unwrap_or(0);
        eprintln!(
            "  ✓ node {}: experts = {} ({:.2} GB)",
            node_idx,
            resolved.experts.display(),
            e_size as f64 / 1e9
        );
        trunk_path_final = Some(resolved.trunk);
    }

    if let Some(trunk) = trunk_path_final.as_ref() {
        let t_size = fs::metadata(trunk).map(|m| m.len()).unwrap_or(0);
        eprintln!("✅ Trunk: {} ({:.2} GB)", trunk.display(), t_size as f64 / 1e9);
    }

    // Item #2: compute the full source-model SHA-256 and store it in the
    // manifest. Migrate is an offline batch operation so the one-time cost
    // (~10-30 min on 600 GB at NAS speeds) is acceptable; at runtime we
    // stay on the cheap `source_fingerprint` for version derivation and
    // only rely on trunk/experts hashes for integrity. Populate the
    // manifest after the split so we don't re-hash on idempotent re-runs.
    if let Some(trunk) = trunk_path_final.as_ref() {
        // trunk path layout: <root>/trunks/<stem>/<version>/trunk.gguf
        // — the parent directory name is the manifest version id.
        let version = trunk
            .parent()
            .and_then(|p| p.file_name())
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        let manifest_path = moe::manifest_file_path(&storage, &args.source_model, &version);
        if let Some(mp) = manifest_path {
            if let Ok(mut manifest) = moe::load_manifest(&mp) {
                if manifest.source_model_sha256.is_empty() {
                    eprintln!(
                        "🔎 Computing source model SHA-256 ({:.1} GB, one-time; stored in manifest)...",
                        source_size as f64 / 1e9
                    );
                    let t0 = std::time::Instant::now();
                    let sha = moe::sha256_file(&args.source_model)
                        .with_context(|| "hashing source model")?;
                    eprintln!(
                        "   source_model_sha256 = {}… ({:.1}s)",
                        &sha[..16],
                        t0.elapsed().as_secs_f64()
                    );
                    manifest.source_model_sha256 = sha;
                    moe::write_manifest_atomic(&mp, &manifest)
                        .with_context(|| "re-writing manifest with source_model_sha256")?;
                }
            }
        }
    }

    // Legacy cleanup hint: if the old per-node cache has entries for this
    // model, tell the user where they are. Never delete automatically.
    let mono_cache = dirs::home_dir()
        .map(|h| h.join(".cache").join("mesh-llm").join("splits"))
        .filter(|p| p.exists());
    if let Some(root) = mono_cache {
        let stem = args
            .source_model
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        let legacy_dir = root.join(&stem);
        if legacy_dir.exists() {
            eprintln!(
                "💡 Legacy monolithic shards still present at {}. Delete them once you've verified split mode works end-to-end.",
                legacy_dir.display()
            );
        }
    }
    Ok(())
}

/// Default bin-dir fallback: the directory containing the mesh-llm binary.
/// mesh-llm is shipped alongside llama-moe-split under
/// `/opt/mesh-bundle/` in the standard CUDA image.
fn default_bin_dir() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    exe.parent().map(|p| p.to_path_buf())
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
        min_experts_per_node_override: None,
    })
    .await?;
    moe_plan_formatter(json_output).render(&report)
}

async fn run_analyze_full(
    model: &str,
    share: bool,
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
    let analysis_path = moe_planner::write_analysis_json(&resolved, &output_path, "full-v1")?;
    println!("✅ Full MoE analysis complete");
    println!("  Ranking: {}", output_path.display());
    println!("  Analysis: {}", analysis_path.display());
    println!("  Log: {}", log_path.display());
    if share {
        auto_share_ranking(&resolved, &output_path, &hf_job.dataset_repo).await?;
    } else {
        print_submit_suggestion(&resolved.path);
    }
    Ok(())
}

async fn run_analyze_micro(
    model: &str,
    share: bool,
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
    let analysis_path = moe_planner::write_analysis_json(&resolved, &cache_path, "micro-v1")?;
    println!("✅ Micro MoE analysis complete");
    println!("  Ranking: {}", cache_path.display());
    println!("  Analysis: {}", analysis_path.display());
    if !wrote_cache {
        println!(
            "  Note: A stronger or equivalent shared ranking already exists, so this micro-v1 result was not promoted as the preferred shared artifact."
        );
    }
    println!("  Log: {}", log_path.display());
    if share {
        auto_share_ranking(&resolved, cache_path.as_path(), &hf_job.dataset_repo).await?;
    } else {
        print_submit_suggestion(&resolved.path);
    }
    Ok(())
}

async fn auto_share_ranking(
    resolved: &moe_planner::MoeModelContext,
    ranking_path: &Path,
    dataset_repo: &str,
) -> Result<()> {
    println!("📤 Auto-sharing ranking...");
    run_share_resolved(resolved, Some(ranking_path), dataset_repo)
        .await
        .with_context(|| {
            format!(
                "Automatic share failed after analyze completed for {}",
                resolved.display_name
            )
        })
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
    println!("  mesh-llm moe share '{}'", identity.distribution_ref());
}

async fn run_share(model: &str, ranking_file: Option<&Path>, dataset_repo: &str) -> Result<()> {
    let resolved = moe_planner::resolve_model_context(model).await?;
    run_share_resolved(&resolved, ranking_file, dataset_repo).await
}

async fn run_share_resolved(
    resolved: &moe_planner::MoeModelContext,
    ranking_file: Option<&Path>,
    dataset_repo: &str,
) -> Result<()> {
    let share_error = |title: &str, detail: &str| -> anyhow::Error {
        eprintln!("❌ {title}");
        eprintln!("   {detail}");
        anyhow::anyhow!("{title}: {detail}")
    };

    let ranking = moe_planner::local_submit_ranking(resolved, ranking_file)?;
    moe_planner::validate_ranking(resolved, &ranking).with_context(|| {
        format!(
            "Validate ranking {} against model {}",
            ranking.path.display(),
            resolved.display_name
        )
    })?;
    let log_path = log_path_for(&resolved.path, &ranking.analyzer_id);
    let bundle = moe_planner::build_submit_bundle(resolved, &ranking, Some(log_path.as_path()))?;
    let api =
        models::build_hf_tokio_api(false).context("Build Hugging Face client for MoE share")?;
    let (owner, name) = dataset_repo.split_once('/').unwrap_or(("", dataset_repo));
    let dataset = api.dataset(owner, name);
    let info = dataset
        .info(
            &hf_hub::RepoInfoParams::builder()
                .revision("main".to_string())
                .build(),
        )
        .await
        .with_context(|| format!("Fetch dataset info for {}", dataset_repo))?;
    let hf_hub::RepoInfo::Dataset(info) = info else {
        anyhow::bail!("Expected dataset repo info for {}", dataset_repo);
    };
    let existing = bundle
        .dataset_paths
        .iter()
        .filter(|path| {
            info.siblings
                .as_ref()
                .is_some_and(|siblings| siblings.iter().any(|entry| &entry.rfilename == *path))
        })
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
    operations.push(ndjson_file_op(
        &format!("{}/analysis.json", bundle.dataset_prefix),
        bundle.analysis_content.as_bytes(),
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
            "a/analysis.json".to_string(),
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
