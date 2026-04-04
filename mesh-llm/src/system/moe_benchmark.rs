use anyhow::{anyhow, bail, Context, Result};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use crate::inference::moe;
use crate::models::{self, catalog};
use crate::network::router;
use crate::system::benchmark_prompts::{self, PromptCorpusEntry, PromptCorpusSummary};

#[derive(Clone, Debug)]
pub(crate) struct MoeRankingBenchmarkArgs {
    pub model: String,
    pub nodes: usize,
    pub overlap: usize,
    pub min_experts: Option<u32>,
    pub variants: Vec<BenchmarkVariant>,
    pub analyze_ranking: Option<PathBuf>,
    pub prompts: Option<PathBuf>,
    pub output: Option<PathBuf>,
}

#[derive(Clone, Debug)]
pub(crate) struct MoeMicroAnalyzeBenchmarkArgs {
    pub model: String,
    pub min_experts: Option<u32>,
    pub analyze_ranking: Option<PathBuf>,
    pub prompts: Option<PathBuf>,
    pub output: Option<PathBuf>,
}

#[derive(Clone, Debug)]
pub(crate) struct MoeGroupingBenchmarkArgs {
    pub model: String,
    pub nodes: usize,
    pub overlap: usize,
    pub min_experts: Option<u32>,
    pub analyze_ranking: Option<PathBuf>,
    pub output: Option<PathBuf>,
}

#[derive(Clone, Debug)]
pub(crate) struct MoeModelMatrixBenchmarkArgs {
    pub models: Vec<String>,
    pub nodes: usize,
    pub overlap: usize,
    pub min_experts: Option<u32>,
    pub prompts: Option<PathBuf>,
    pub analyze_ranking_dir: Option<PathBuf>,
    pub output: Option<PathBuf>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum BenchmarkVariant {
    Analyze,
}

#[derive(Clone)]
struct ResolvedBenchmarkModel {
    input: String,
    path: PathBuf,
    name: String,
    architecture: String,
    info: moe::GgufMoeInfo,
    min_experts: u32,
}

#[derive(Clone, Debug)]
struct AnalyzeExpertMass {
    expert_id: u32,
    gate_mass: f64,
    mass_pct: f64,
    selection_count: u64,
}

#[derive(Clone, Debug)]
struct AnalyzeMassProfile {
    entries: Vec<AnalyzeExpertMass>,
    mass_by_expert: HashMap<u32, f64>,
    total_mass: f64,
}

#[derive(Debug, Serialize)]
struct BenchmarkModelInfo {
    input: String,
    resolved_path: String,
    name: String,
    architecture: String,
    expert_count: u32,
    expert_used_count: u32,
}

#[derive(Debug, Serialize)]
struct BenchmarkConfig {
    nodes: usize,
    overlap: usize,
    min_experts: u32,
}

#[derive(Debug, Serialize)]
struct AssignmentReport {
    node: usize,
    expert_count: usize,
    shared: usize,
    unique: usize,
    experts: Vec<u32>,
}

#[derive(Debug, Serialize)]
struct VariantReport {
    name: &'static str,
    ranking_source: String,
    ranking_len: usize,
    assignments: Vec<AssignmentReport>,
}

#[derive(Debug, Serialize)]
struct MoeRankingBenchmarkReport {
    benchmark: &'static str,
    model: BenchmarkModelInfo,
    config: BenchmarkConfig,
    prompt_corpus: Option<PromptCorpusSummary>,
    variants: Vec<VariantReport>,
}

#[derive(Debug, Serialize)]
struct GroupingStrategyReport {
    name: &'static str,
    ranking_source: String,
    grouping_mode: &'static str,
    replicated_experts: usize,
    shared_mass_pct: f64,
    mean_node_mass_pct: f64,
    min_node_mass_pct: f64,
    max_node_mass_pct: f64,
    node_mass_imbalance_pct: f64,
    assignments: Vec<AssignmentReport>,
}

#[derive(Debug, Serialize)]
struct MoeGroupingBenchmarkReport {
    benchmark: &'static str,
    model: BenchmarkModelInfo,
    config: BenchmarkConfig,
    analyze_ranking_source: String,
    strategies: Vec<GroupingStrategyReport>,
}

#[derive(Debug, Serialize)]
struct MicroAnalyzeExperimentReport {
    name: String,
    prompt_count: usize,
    tokens: u32,
    all_layers: bool,
    runtime_seconds: f64,
    spearman_rank_correlation: f64,
    recall_at_min_experts: f64,
    weighted_recall_at_min_experts: f64,
    captures_top_truth_expert: bool,
    ranking_preview: Vec<u32>,
}

#[derive(Debug, Serialize)]
struct MoeMicroAnalyzeBenchmarkReport {
    benchmark: &'static str,
    model: BenchmarkModelInfo,
    min_experts: u32,
    analyze_ranking_source: String,
    prompt_corpus: Option<PromptCorpusSummary>,
    experiments: Vec<MicroAnalyzeExperimentReport>,
}

#[derive(Debug, Serialize)]
struct MoeModelMatrixModelReport {
    model: BenchmarkModelInfo,
    ranking: MoeRankingBenchmarkReport,
    grouping: MoeGroupingBenchmarkReport,
    micro_analyze: MoeMicroAnalyzeBenchmarkReport,
}

#[derive(Debug, Serialize)]
struct MoeModelMatrixReport {
    benchmark: &'static str,
    prompt_corpus: Option<PromptCorpusSummary>,
    models: Vec<MoeModelMatrixModelReport>,
}

pub(crate) async fn run_moe_ranking_benchmark(args: MoeRankingBenchmarkArgs) -> Result<()> {
    validate_nodes(args.nodes)?;
    if args.variants.is_empty() {
        bail!("--variants must include at least one ranking source");
    }

    let model = resolve_benchmark_model(&args.model, args.min_experts).await?;
    let prompt_corpus = load_prompt_summary(args.prompts.as_deref())?;
    let report = build_ranking_report(
        &model,
        args.nodes,
        args.overlap,
        &args.variants,
        args.analyze_ranking.as_deref(),
        prompt_corpus,
    )?;
    write_json_report(&report, args.output.as_deref(), "MoE ranking benchmark")?;
    Ok(())
}

pub(crate) async fn run_moe_micro_analyze_benchmark(
    args: MoeMicroAnalyzeBenchmarkArgs,
) -> Result<()> {
    let model = resolve_benchmark_model(&args.model, args.min_experts).await?;
    let prompt_corpus = load_prompt_summary(args.prompts.as_deref())?;
    let report =
        build_micro_analyze_report(&model, args.analyze_ranking.as_deref(), prompt_corpus)?;
    write_json_report(
        &report,
        args.output.as_deref(),
        "MoE micro-analyze benchmark",
    )?;
    Ok(())
}

pub(crate) async fn run_moe_grouping_benchmark(args: MoeGroupingBenchmarkArgs) -> Result<()> {
    validate_nodes(args.nodes)?;
    let model = resolve_benchmark_model(&args.model, args.min_experts).await?;
    let report = build_grouping_report(
        &model,
        args.nodes,
        args.overlap,
        args.analyze_ranking.as_deref(),
    )?;
    write_json_report(&report, args.output.as_deref(), "MoE grouping benchmark")?;
    Ok(())
}

pub(crate) async fn run_moe_model_matrix_benchmark(
    args: MoeModelMatrixBenchmarkArgs,
) -> Result<()> {
    validate_nodes(args.nodes)?;
    if args.models.is_empty() {
        bail!("--model must be provided at least once");
    }

    let prompt_corpus = load_prompt_summary(args.prompts.as_deref())?;
    let mut reports = Vec::with_capacity(args.models.len());
    for model_spec in &args.models {
        let model = resolve_benchmark_model(model_spec, args.min_experts).await?;
        let explicit_analyze = explicit_analyze_path(&model, args.analyze_ranking_dir.as_deref());
        let ensured_analyze = ensure_full_analyze_ranking(&model, explicit_analyze.as_deref())?;
        let ranking = build_ranking_report(
            &model,
            args.nodes,
            args.overlap,
            &[BenchmarkVariant::Analyze],
            Some(ensured_analyze.as_path()),
            prompt_corpus.clone(),
        )?;
        let grouping = build_grouping_report(
            &model,
            args.nodes,
            args.overlap,
            Some(ensured_analyze.as_path()),
        )?;
        let micro_analyze = build_micro_analyze_report(
            &model,
            Some(ensured_analyze.as_path()),
            prompt_corpus.clone(),
        )?;
        reports.push(MoeModelMatrixModelReport {
            model: benchmark_model_info(&model),
            ranking,
            grouping,
            micro_analyze,
        });
    }

    let report = MoeModelMatrixReport {
        benchmark: "moe-model-matrix",
        prompt_corpus,
        models: reports,
    };
    write_json_report(
        &report,
        args.output.as_deref(),
        "MoE model matrix benchmark",
    )?;
    Ok(())
}

fn build_ranking_report(
    model: &ResolvedBenchmarkModel,
    nodes: usize,
    overlap: usize,
    variants: &[BenchmarkVariant],
    analyze_ranking: Option<&Path>,
    prompt_corpus: Option<PromptCorpusSummary>,
) -> Result<MoeRankingBenchmarkReport> {
    let mut reports = Vec::with_capacity(variants.len());
    for &variant in variants {
        let ranking = resolve_variant_ranking(variant, model, analyze_ranking)?;
        let assignments =
            moe::compute_assignments_with_overlap(&ranking, nodes, model.min_experts, overlap);
        reports.push(VariantReport {
            name: variant_name(variant),
            ranking_source: variant_source_label(variant, model, analyze_ranking),
            ranking_len: ranking.len(),
            assignments: assignment_reports(assignments),
        });
    }

    Ok(MoeRankingBenchmarkReport {
        benchmark: "moe-ranking",
        model: benchmark_model_info(model),
        config: BenchmarkConfig {
            nodes,
            overlap,
            min_experts: model.min_experts,
        },
        prompt_corpus,
        variants: reports,
    })
}

fn build_grouping_report(
    model: &ResolvedBenchmarkModel,
    nodes: usize,
    overlap: usize,
    analyze_ranking: Option<&Path>,
) -> Result<MoeGroupingBenchmarkReport> {
    let analyze_path = ensure_full_analyze_ranking(model, analyze_ranking)?;
    let profile = load_analyze_mass_profile(&analyze_path)?;
    let analyze_ranking_vec = profile.ranking().to_vec();
    let sequential: Vec<u32> = (0..model.info.expert_count).collect();
    let replicate = model.min_experts as usize;

    let strategies = vec![
        (
            "current-sequential",
            "shared-core-overlap",
            "sequential-fallback".to_string(),
            sequential.clone(),
            moe::compute_assignments_with_overlap(&sequential, nodes, model.min_experts, overlap),
            replicate,
        ),
        (
            "current-analyze",
            "shared-core-overlap",
            analyze_path.display().to_string(),
            analyze_ranking_vec.clone(),
            moe::compute_assignments_with_overlap(
                &analyze_ranking_vec,
                nodes,
                model.min_experts,
                overlap,
            ),
            replicate,
        ),
        (
            "snake-analyze-replicated",
            "snake-draft",
            analyze_path.display().to_string(),
            analyze_ranking_vec.clone(),
            moe::compute_snake_draft_assignments(&analyze_ranking_vec, nodes, replicate),
            replicate,
        ),
    ];

    let reports = strategies
        .into_iter()
        .map(
            |(name, grouping_mode, ranking_source, ranking, assignments, replicated_experts)| {
                let node_mass_pct = assignments
                    .iter()
                    .map(|assignment| mass_pct_for_experts(&assignment.experts, &profile))
                    .collect::<Vec<_>>();
                let shared_mass_pct = mass_pct_for_experts(
                    &ranking[..replicated_experts.min(ranking.len())],
                    &profile,
                );
                let mean_node_mass_pct =
                    node_mass_pct.iter().sum::<f64>() / node_mass_pct.len().max(1) as f64;
                let min_node_mass_pct = node_mass_pct.iter().copied().fold(f64::INFINITY, f64::min);
                let max_node_mass_pct = node_mass_pct
                    .iter()
                    .copied()
                    .fold(f64::NEG_INFINITY, f64::max);
                GroupingStrategyReport {
                    name,
                    ranking_source,
                    grouping_mode,
                    replicated_experts,
                    shared_mass_pct,
                    mean_node_mass_pct,
                    min_node_mass_pct,
                    max_node_mass_pct,
                    node_mass_imbalance_pct: max_node_mass_pct - min_node_mass_pct,
                    assignments: assignment_reports(assignments),
                }
            },
        )
        .collect();

    Ok(MoeGroupingBenchmarkReport {
        benchmark: "moe-grouping",
        model: benchmark_model_info(model),
        config: BenchmarkConfig {
            nodes,
            overlap,
            min_experts: model.min_experts,
        },
        analyze_ranking_source: analyze_path.display().to_string(),
        strategies: reports,
    })
}

fn build_micro_analyze_report(
    model: &ResolvedBenchmarkModel,
    analyze_ranking: Option<&Path>,
    prompt_corpus: Option<PromptCorpusSummary>,
) -> Result<MoeMicroAnalyzeBenchmarkReport> {
    let analyze_path = ensure_full_analyze_ranking(model, analyze_ranking)?;
    let profile = load_analyze_mass_profile(&analyze_path)?;
    let prompts = load_or_default_prompts(prompt_corpus.as_ref())?;
    let prompt_count = prompts.len();
    let experiments = micro_experiment_configs(prompt_count)
        .into_iter()
        .map(|config| run_micro_experiment(model, &profile, &prompts, config))
        .collect::<Result<Vec<_>>>()?;

    Ok(MoeMicroAnalyzeBenchmarkReport {
        benchmark: "moe-micro-analyze",
        model: benchmark_model_info(model),
        min_experts: model.min_experts,
        analyze_ranking_source: analyze_path.display().to_string(),
        prompt_corpus,
        experiments,
    })
}

async fn resolve_benchmark_model(
    model_spec: &str,
    min_experts_override: Option<u32>,
) -> Result<ResolvedBenchmarkModel> {
    let path = models::resolve_model_spec(Path::new(model_spec)).await?;
    let info = moe::detect_moe(&path).with_context(|| {
        format!(
            "Model is not auto-detected as MoE from GGUF header: {}",
            path.display()
        )
    })?;
    let name = model_display_name(&path);
    let bundled = bundled_moe_config(&name);
    let min_experts = min_experts_override
        .or_else(|| bundled.as_ref().map(|cfg| cfg.min_experts_per_node))
        .unwrap_or_else(|| ((info.expert_count as f64) * 0.5).ceil() as u32);
    let architecture = moe::scan_gguf_compact_meta(&path)
        .map(|meta| meta.architecture)
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "unknown".to_string());

    Ok(ResolvedBenchmarkModel {
        input: model_spec.to_string(),
        path,
        name,
        architecture,
        info,
        min_experts,
    })
}

fn explicit_analyze_path(
    model: &ResolvedBenchmarkModel,
    analyze_ranking_dir: Option<&Path>,
) -> Option<PathBuf> {
    let dir = analyze_ranking_dir?;
    let path = dir.join(format!("{}.csv", model.path.file_stem()?.to_string_lossy()));
    path.exists().then_some(path)
}

fn benchmark_model_info(model: &ResolvedBenchmarkModel) -> BenchmarkModelInfo {
    BenchmarkModelInfo {
        input: model.input.clone(),
        resolved_path: model.path.display().to_string(),
        name: model.name.clone(),
        architecture: model.architecture.clone(),
        expert_count: model.info.expert_count,
        expert_used_count: model.info.expert_used_count,
    }
}

fn model_display_name(model_path: &Path) -> String {
    model_path
        .file_stem()
        .and_then(|value| value.to_str())
        .map(router::strip_split_suffix_owned)
        .unwrap_or_else(|| model_path.display().to_string())
}

fn bundled_moe_config(model_name: &str) -> Option<catalog::MoeConfig> {
    let q = model_name.to_lowercase();
    catalog::MODEL_CATALOG
        .iter()
        .find(|m| m.name.to_lowercase() == q || m.file.to_lowercase().contains(&q))
        .and_then(|m| m.moe.clone())
}

fn validate_nodes(nodes: usize) -> Result<()> {
    if nodes == 0 {
        bail!("--nodes must be at least 1");
    }
    Ok(())
}

fn load_prompt_summary(path: Option<&Path>) -> Result<Option<PromptCorpusSummary>> {
    path.map(benchmark_prompts::summarize_prompt_corpus)
        .transpose()
}

fn load_or_default_prompts(prompt_corpus: Option<&PromptCorpusSummary>) -> Result<Vec<String>> {
    let Some(summary) = prompt_corpus else {
        return Ok(vec![
            "User: Explain how MoE expert routing works in a large language model.\nAssistant:"
                .to_string(),
        ]);
    };

    let prompts = benchmark_prompts::load_prompt_corpus(Path::new(&summary.path))?;
    Ok(prompts.into_iter().map(render_prompt).collect())
}

fn render_prompt(entry: PromptCorpusEntry) -> String {
    let mut rendered = String::new();
    for message in entry.messages {
        let _ = writeln!(
            rendered,
            "{}: {}\n",
            capitalize_role(&message.role),
            message.content.trim()
        );
    }
    rendered.trim().to_string()
}

fn capitalize_role(role: &str) -> String {
    let mut chars = role.chars();
    match chars.next() {
        Some(first) => format!("{}{}", first.to_ascii_uppercase(), chars.as_str()),
        None => "User".to_string(),
    }
}

fn assignment_reports(assignments: Vec<moe::NodeAssignment>) -> Vec<AssignmentReport> {
    assignments
        .into_iter()
        .enumerate()
        .map(|(node, assignment)| AssignmentReport {
            node,
            expert_count: assignment.experts.len(),
            shared: assignment.n_shared,
            unique: assignment.n_unique,
            experts: assignment.experts,
        })
        .collect()
}

fn resolve_variant_ranking(
    variant: BenchmarkVariant,
    model: &ResolvedBenchmarkModel,
    analyze_ranking: Option<&Path>,
) -> Result<Vec<u32>> {
    match variant {
        BenchmarkVariant::Analyze => {
            if let Some(path) = analyze_ranking {
                return moe::load_cached_ranking(path)
                    .with_context(|| format!("Load moe-analyze ranking from {}", path.display()));
            }

            let cached_path = moe::ranking_cache_path(&model.path);
            if let Some(ranking) = moe::load_cached_ranking(&cached_path) {
                return Ok(ranking);
            }

            bail!(
                "No moe-analyze ranking found for {}. Provide --analyze-ranking or cache a ranking at {}",
                model.name,
                cached_path.display()
            )
        }
    }
}

fn variant_name(variant: BenchmarkVariant) -> &'static str {
    match variant {
        BenchmarkVariant::Analyze => "analyze",
    }
}

fn variant_source_label(
    variant: BenchmarkVariant,
    model: &ResolvedBenchmarkModel,
    analyze_ranking: Option<&Path>,
) -> String {
    match variant {
        BenchmarkVariant::Analyze => {
            if let Some(path) = analyze_ranking {
                return path.display().to_string();
            }
            let cached_path = moe::ranking_cache_path(&model.path);
            if cached_path.exists() {
                cached_path.display().to_string()
            } else {
                "missing".to_string()
            }
        }
    }
}

fn ensure_full_analyze_ranking(
    model: &ResolvedBenchmarkModel,
    explicit: Option<&Path>,
) -> Result<PathBuf> {
    if let Some(path) = explicit {
        if path.exists() {
            return Ok(path.to_path_buf());
        }
        bail!("Explicit analyze ranking not found: {}", path.display());
    }

    let cached = moe::ranking_cache_path(&model.path);
    if cached.exists() {
        return Ok(cached);
    }

    if let Some(parent) = cached.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("Create analyze ranking directory {}", parent.display()))?;
    }

    let analyze_bin = resolve_analyze_binary()?;
    let output = Command::new(&analyze_bin)
        .args([
            "-m",
            &model.path.to_string_lossy(),
            "--all-layers",
            "--export-ranking",
            &cached.to_string_lossy(),
            "-n",
            "32",
            "-c",
            "4096",
            "-ngl",
            "99",
        ])
        .output()
        .with_context(|| format!("Run {} for {}", analyze_bin.display(), model.path.display()))?;

    if !output.status.success() {
        bail!(
            "llama-moe-analyze failed for {}: {}",
            model.path.display(),
            String::from_utf8_lossy(&output.stderr)
        );
    }

    Ok(cached)
}

fn resolve_analyze_binary() -> Result<PathBuf> {
    let exe = std::env::current_exe().context("Failed to determine own binary path")?;
    let bin_dir = exe
        .parent()
        .ok_or_else(|| anyhow!("Current executable has no parent directory"))?;
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

fn load_analyze_mass_profile(path: &Path) -> Result<AnalyzeMassProfile> {
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("Read analyze ranking {}", path.display()))?;
    let mut entries = Vec::new();
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') || trimmed.starts_with("expert") {
            continue;
        }
        let parts = trimmed.split(',').map(str::trim).collect::<Vec<_>>();
        if parts.len() < 4 {
            continue;
        }
        entries.push(AnalyzeExpertMass {
            expert_id: parts[0].parse().with_context(|| {
                format!("Parse expert id from {} in {}", trimmed, path.display())
            })?,
            gate_mass: parts[1].parse().with_context(|| {
                format!("Parse gate mass from {} in {}", trimmed, path.display())
            })?,
            mass_pct: parts[2].parse().with_context(|| {
                format!("Parse mass pct from {} in {}", trimmed, path.display())
            })?,
            selection_count: parts[3].parse().with_context(|| {
                format!(
                    "Parse selection count from {} in {}",
                    trimmed,
                    path.display()
                )
            })?,
        });
    }

    if entries.is_empty() {
        bail!(
            "Analyze ranking was empty or unreadable: {}",
            path.display()
        );
    }

    let mut mass_by_expert = HashMap::new();
    let total_mass = entries.iter().map(|entry| entry.gate_mass).sum::<f64>();
    for entry in &entries {
        mass_by_expert.insert(entry.expert_id, entry.gate_mass);
    }

    Ok(AnalyzeMassProfile {
        entries,
        mass_by_expert,
        total_mass,
    })
}

impl AnalyzeMassProfile {
    fn ranking(&self) -> Vec<u32> {
        self.entries.iter().map(|entry| entry.expert_id).collect()
    }
}

fn recall_at_top_n(candidate: &[u32], truth: &[u32], n: usize) -> f64 {
    let n = n.min(candidate.len()).min(truth.len());
    if n == 0 {
        return 0.0;
    }
    let candidate_set: BTreeSet<u32> = candidate.iter().take(n).copied().collect();
    let truth_set: BTreeSet<u32> = truth.iter().take(n).copied().collect();
    candidate_set.intersection(&truth_set).count() as f64 / n as f64
}

fn weighted_recall_at_top_n(candidate: &[u32], truth: &AnalyzeMassProfile, n: usize) -> f64 {
    let truth_top = truth.entries.iter().take(n).collect::<Vec<_>>();
    if truth_top.is_empty() {
        return 0.0;
    }
    let denominator = truth_top.iter().map(|entry| entry.gate_mass).sum::<f64>();
    if denominator <= f64::EPSILON {
        return 0.0;
    }
    let candidate_set: BTreeSet<u32> = candidate.iter().take(n).copied().collect();
    let numerator = truth_top
        .iter()
        .filter(|entry| candidate_set.contains(&entry.expert_id))
        .map(|entry| entry.gate_mass)
        .sum::<f64>();
    numerator / denominator
}

fn spearman_rank_correlation(candidate: &[u32], truth: &AnalyzeMassProfile) -> f64 {
    let n = candidate.len().min(truth.entries.len());
    if n < 2 {
        return 1.0;
    }
    let mut candidate_rank = HashMap::new();
    for (idx, expert) in candidate.iter().enumerate() {
        candidate_rank.insert(*expert, idx as f64);
    }
    let sum_d2 = truth
        .entries
        .iter()
        .take(n)
        .enumerate()
        .filter_map(|(idx, entry)| {
            candidate_rank
                .get(&entry.expert_id)
                .map(|cand| (*cand - idx as f64).powi(2))
        })
        .sum::<f64>();
    let n = n as f64;
    1.0 - (6.0 * sum_d2) / (n * (n * n - 1.0))
}

fn mass_pct_for_experts(experts: &[u32], profile: &AnalyzeMassProfile) -> f64 {
    if profile.total_mass <= f64::EPSILON {
        return 0.0;
    }
    let numerator = experts
        .iter()
        .filter_map(|expert| profile.mass_by_expert.get(expert).copied())
        .sum::<f64>();
    100.0 * numerator / profile.total_mass
}

#[derive(Clone, Copy)]
struct MicroAnalyzeConfig {
    name: &'static str,
    prompt_count: usize,
    tokens: u32,
    all_layers: bool,
}

fn micro_experiment_configs(prompt_count: usize) -> Vec<MicroAnalyzeConfig> {
    let mut configs = vec![
        MicroAnalyzeConfig {
            name: "micro-1p-8t-first-layer",
            prompt_count: 1,
            tokens: 8,
            all_layers: false,
        },
        MicroAnalyzeConfig {
            name: "micro-1p-8t-all-layers",
            prompt_count: 1,
            tokens: 8,
            all_layers: true,
        },
    ];
    if prompt_count >= 4 {
        configs.push(MicroAnalyzeConfig {
            name: "micro-4p-8t-all-layers",
            prompt_count: 4,
            tokens: 8,
            all_layers: true,
        });
        configs.push(MicroAnalyzeConfig {
            name: "micro-4p-32t-all-layers",
            prompt_count: 4,
            tokens: 32,
            all_layers: true,
        });
    } else if prompt_count >= 2 {
        configs.push(MicroAnalyzeConfig {
            name: "micro-2p-32t-all-layers",
            prompt_count: 2,
            tokens: 32,
            all_layers: true,
        });
    }
    configs
}

fn run_micro_experiment(
    model: &ResolvedBenchmarkModel,
    truth: &AnalyzeMassProfile,
    prompts: &[String],
    config: MicroAnalyzeConfig,
) -> Result<MicroAnalyzeExperimentReport> {
    let selected_prompts = prompts.iter().take(config.prompt_count).collect::<Vec<_>>();
    let start = Instant::now();
    let temp_root = std::env::temp_dir().join(format!(
        "mesh-llm-micro-analyze-{}-{}",
        std::process::id(),
        start.elapsed().as_nanos()
    ));
    std::fs::create_dir_all(&temp_root)
        .with_context(|| format!("Create temp dir {}", temp_root.display()))?;

    let mut mass_by_expert: BTreeMap<u32, f64> = BTreeMap::new();
    let mut selection_count_by_expert: BTreeMap<u32, u64> = BTreeMap::new();

    for (idx, prompt) in selected_prompts.iter().enumerate() {
        let output_path = temp_root.join(format!("prompt-{idx}.csv"));
        run_micro_analyze_export(
            model,
            prompt,
            &output_path,
            config.tokens,
            config.all_layers,
        )?;
        let partial = load_analyze_mass_profile(&output_path)?;
        for entry in partial.entries {
            *mass_by_expert.entry(entry.expert_id).or_insert(0.0) += entry.gate_mass;
            *selection_count_by_expert
                .entry(entry.expert_id)
                .or_insert(0) += entry.selection_count;
        }
    }

    let mut entries = mass_by_expert
        .into_iter()
        .map(|(expert_id, gate_mass)| AnalyzeExpertMass {
            expert_id,
            gate_mass,
            mass_pct: 0.0,
            selection_count: selection_count_by_expert
                .get(&expert_id)
                .copied()
                .unwrap_or(0),
        })
        .collect::<Vec<_>>();
    entries.sort_by(|a, b| {
        b.gate_mass
            .partial_cmp(&a.gate_mass)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.expert_id.cmp(&b.expert_id))
    });
    let total_mass = entries.iter().map(|entry| entry.gate_mass).sum::<f64>();
    for entry in &mut entries {
        entry.mass_pct = if total_mass <= f64::EPSILON {
            0.0
        } else {
            100.0 * entry.gate_mass / total_mass
        };
    }
    let ranking = entries
        .iter()
        .map(|entry| entry.expert_id)
        .collect::<Vec<_>>();
    let elapsed = start.elapsed();
    let _ = std::fs::remove_dir_all(&temp_root);

    Ok(MicroAnalyzeExperimentReport {
        name: config.name.to_string(),
        prompt_count: selected_prompts.len(),
        tokens: config.tokens,
        all_layers: config.all_layers,
        runtime_seconds: elapsed.as_secs_f64(),
        spearman_rank_correlation: spearman_rank_correlation(&ranking, truth),
        recall_at_min_experts: recall_at_top_n(
            &ranking,
            &truth.ranking(),
            model.min_experts as usize,
        ),
        weighted_recall_at_min_experts: weighted_recall_at_top_n(
            &ranking,
            truth,
            model.min_experts as usize,
        ),
        captures_top_truth_expert: truth
            .entries
            .first()
            .map(|entry| {
                ranking
                    .iter()
                    .take(model.min_experts as usize)
                    .any(|expert| *expert == entry.expert_id)
            })
            .unwrap_or(false),
        ranking_preview: ranking.iter().take(16).copied().collect(),
    })
}

fn run_micro_analyze_export(
    model: &ResolvedBenchmarkModel,
    prompt: &str,
    output_path: &Path,
    tokens: u32,
    all_layers: bool,
) -> Result<()> {
    let analyze_bin = resolve_analyze_binary()?;
    let mut command = Command::new(&analyze_bin);
    command.args([
        "-m",
        &model.path.to_string_lossy(),
        "--export-ranking",
        &output_path.to_string_lossy(),
        "-n",
        &tokens.to_string(),
        "-c",
        "4096",
        "-ngl",
        "99",
        "-p",
        prompt,
    ]);
    if all_layers {
        command.arg("--all-layers");
    }

    let output = command
        .output()
        .with_context(|| format!("Run micro analyze for {}", model.path.display()))?;
    if !output.status.success() {
        bail!(
            "llama-moe-analyze micro run failed for {}: {}",
            model.path.display(),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    Ok(())
}

fn write_json_report<T: Serialize>(report: &T, output: Option<&Path>, label: &str) -> Result<()> {
    let json = serde_json::to_string_pretty(report)?;
    if let Some(path) = output {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).with_context(|| {
                format!("Create benchmark output directory {}", parent.display())
            })?;
        }
        std::fs::write(path, json)
            .with_context(|| format!("Write benchmark report to {}", path.display()))?;
        eprintln!("📝 Wrote {label} report to {}", path.display());
    } else {
        println!("{json}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_profile() -> AnalyzeMassProfile {
        let entries = vec![
            AnalyzeExpertMass {
                expert_id: 0,
                gate_mass: 50.0,
                mass_pct: 50.0,
                selection_count: 10,
            },
            AnalyzeExpertMass {
                expert_id: 1,
                gate_mass: 30.0,
                mass_pct: 30.0,
                selection_count: 7,
            },
            AnalyzeExpertMass {
                expert_id: 2,
                gate_mass: 20.0,
                mass_pct: 20.0,
                selection_count: 5,
            },
        ];
        let mut mass_by_expert = HashMap::new();
        for entry in &entries {
            mass_by_expert.insert(entry.expert_id, entry.gate_mass);
        }
        AnalyzeMassProfile {
            entries,
            mass_by_expert,
            total_mass: 100.0,
        }
    }

    #[test]
    fn weighted_recall_prefers_hot_experts() {
        let profile = fixture_profile();
        let candidate = vec![0, 2, 1];
        assert!((weighted_recall_at_top_n(&candidate, &profile, 2) - 0.625).abs() < 1e-9);
    }

    #[test]
    fn spearman_is_one_for_identical_ranking() {
        let profile = fixture_profile();
        assert!((spearman_rank_correlation(&[0, 1, 2], &profile) - 1.0).abs() < 1e-9);
    }
}
