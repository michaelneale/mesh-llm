use anyhow::{bail, Context, Result};
use hf_hub::{RepoDownloadFileParams, RepoInfoParams};
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::cli::terminal_progress::start_spinner;
use crate::inference::{election, moe};
use crate::models::{
    build_hf_api, catalog, gguf::GgufTensorByteProfile, huggingface_identity_for_path,
    resolve_model_spec, resolve_model_spec_with_progress,
};

const DEFAULT_DATASET_REVISION: &str = "main";
pub(crate) const DEFAULT_MOE_RANKINGS_DATASET: &str = "meshllm/moe-rankings";
const ANALYSIS_JSON_FILENAME: &str = "analysis.json";
const DEFAULT_MASS_CHECKPOINTS: [usize; 8] = [1, 2, 4, 8, 16, 32, 64, 128];
const MODEL_LOAD_HEADROOM_NUMERATOR: u64 = 11;
const MODEL_LOAD_HEADROOM_DENOMINATOR: u64 = 10;

#[derive(Clone, Debug)]
pub(crate) struct MoePlanArgs {
    pub model: String,
    pub ranking_file: Option<PathBuf>,
    pub max_vram_gb: Option<f64>,
    pub nodes: Option<usize>,
    pub dataset_repo: String,
    pub progress: bool,
}

#[derive(Clone, Debug)]
pub(crate) struct MoePlanReport {
    pub model: MoeModelContext,
    pub ranking: ResolvedRanking,
    pub target_vram_bytes: u64,
    pub recommended_nodes: usize,
    pub max_supported_nodes: usize,
    pub feasible: bool,
    pub assumptions: Vec<String>,
    pub assignments: Vec<moe::NodeAssignment>,
    pub shared_mass_pct: Option<f64>,
    pub max_node_mass_pct: Option<f64>,
    pub min_node_mass_pct: Option<f64>,
}

#[derive(Clone, Debug)]
pub(crate) struct MoeModelContext {
    pub input: String,
    pub path: PathBuf,
    pub display_name: String,
    pub source_repo: Option<String>,
    pub source_revision: Option<String>,
    pub distribution_id: String,
    pub expert_count: u32,
    pub used_expert_count: u32,
    pub min_experts_per_node: u32,
    pub total_model_bytes: u64,
}

#[derive(Clone, Debug)]
pub(crate) struct ResolvedRanking {
    pub path: PathBuf,
    pub metadata_path: Option<PathBuf>,
    pub analysis_path: Option<PathBuf>,
    pub analyzer_id: String,
    pub source: RankingSource,
    pub reason: String,
}

#[derive(Clone, Debug)]
pub(crate) struct MoeSubmitBundle {
    pub dataset_prefix: String,
    pub dataset_paths: Vec<String>,
    pub ranking_path: PathBuf,
    pub metadata_content: String,
    pub analysis_content: String,
    pub log_path: Option<PathBuf>,
    pub commit_message: String,
    pub commit_description: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub(crate) struct MoeAnalysisJson {
    pub schema_version: u32,
    pub ranking: MoeAnalysisRanking,
    pub model: MoeAnalysisModel,
    pub memory: MoeAnalysisMemory,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub(crate) struct MoeAnalysisRanking {
    pub sha256: String,
    pub rows: usize,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub(crate) struct MoeAnalysisModel {
    pub expert_count: u32,
    pub expert_used_count: u32,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub(crate) struct MoeAnalysisMemory {
    pub full_model_bytes: u64,
    pub base_resident_bytes: u64,
    pub shard_file_overhead_bytes: u64,
    pub expert_tensor_bytes_total: u64,
    pub expert_bytes: MoeAnalysisExpertBytes,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(crate) enum MoeAnalysisExpertBytes {
    Uniform { bytes_per_expert: u64 },
    DensePerExpert { values: Vec<u64> },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct MoeStartupFitEstimate {
    pub analyzer_id: String,
    pub ranking_source: &'static str,
    pub target_vram_bytes: u64,
    pub required_experts_per_node: u32,
    pub full_model_bytes: u64,
    pub full_model_launch_bytes: u64,
    pub recommended_nodes: Option<usize>,
    pub predicted_max_shard_bytes: Option<u64>,
    pub predicted_max_shard_launch_bytes: Option<u64>,
}

impl MoeStartupFitEstimate {
    pub(crate) fn full_model_fits(&self) -> bool {
        self.full_model_launch_bytes <= self.target_vram_bytes
    }

    pub(crate) fn shard_plan_fits(&self) -> bool {
        self.predicted_max_shard_launch_bytes
            .is_some_and(|bytes| bytes <= self.target_vram_bytes)
    }

    pub(crate) fn any_fit_exists(&self) -> bool {
        self.full_model_fits() || self.shard_plan_fits()
    }
}

#[derive(Clone, Debug)]
pub(crate) enum RankingSource {
    Override,
    LocalCache,
    HuggingFaceDataset,
}

impl RankingSource {
    pub(crate) fn label(&self) -> &'static str {
        match self {
            Self::Override => "override",
            Self::LocalCache => "local cache",
            Self::HuggingFaceDataset => "Hugging Face dataset",
        }
    }
}

#[derive(Debug)]
struct AnalyzeMassProfile {
    ranking: Vec<u32>,
    masses: Vec<(u32, f64)>,
    total_mass: f64,
}

#[derive(Clone, Debug, Default)]
struct AggregatedTensorByteProfile {
    full_model_bytes: u64,
    base_resident_bytes: u64,
    expert_tensor_bytes: u64,
    file_overhead_bytes: u64,
}

pub(crate) async fn plan_moe(args: MoePlanArgs) -> Result<MoePlanReport> {
    if let Some(nodes) = args.nodes {
        if nodes == 0 {
            bail!("--nodes must be at least 1");
        }
    }
    let mut model = resolve_model_context_with_progress(&args.model, args.progress).await?;
    let ranking = resolve_best_ranking(&model, &args).await?;
    let target_vram_bytes = resolve_target_vram_bytes(args.max_vram_gb);
    if target_vram_bytes == 0 {
        bail!(
            "No VRAM target available. Pass --max-vram <GB> or run on a machine with detectable GPU memory."
        );
    }

    let ranking_vec = moe::load_cached_ranking(&ranking.path)
        .ok_or_else(|| anyhow::anyhow!("cached ranking not found: {}", ranking.path.display()))
        .with_context(|| format!("Load ranking {}", ranking.path.display()))?;
    let heuristic_recommended_nodes = ((model.total_model_bytes as f64) / target_vram_bytes as f64)
        .ceil()
        .max(1.0) as usize;
    let analysis = ranking
        .analysis_path
        .as_deref()
        .map(read_analysis_json)
        .transpose()?;

    let (recommended_nodes, max_supported_nodes, feasible, mut assumptions) = if let Some(
        ref analysis,
    ) = analysis
    {
        let fit = estimate_startup_fit_from_analysis(
            &ranking.path,
            analysis,
            target_vram_bytes,
            &ranking.analyzer_id,
            ranking.source.label(),
        )?;
        let required_experts_per_node = fit.required_experts_per_node;
        model.min_experts_per_node = required_experts_per_node;
        let max_supported_nodes =
            (model.expert_count / required_experts_per_node.max(1)).max(1) as usize;
        let recommended_nodes = args
            .nodes
            .unwrap_or_else(|| fit.recommended_nodes.unwrap_or(max_supported_nodes));
        let feasible = if recommended_nodes == 1 {
            fit.full_model_fits()
        } else if recommended_nodes > max_supported_nodes {
            false
        } else {
            let (_, _, predicted_max_shard_launch_bytes) = predict_plan_fit_for_nodes(
                &ranking_vec,
                analysis,
                recommended_nodes,
                required_experts_per_node,
            )?;
            predicted_max_shard_launch_bytes <= target_vram_bytes
        };

        let mut assumptions = vec![
            format!(
                "Full-model launch estimate from analysis.json: {:.1}GB against {:.1}GB target",
                fit.full_model_launch_bytes as f64 / 1e9,
                target_vram_bytes as f64 / 1e9
            ),
            format!(
                "Shared core uses the current 50% planning heuristic: {} / {} experts per node",
                required_experts_per_node, model.expert_count
            ),
        ];
        if recommended_nodes > 1 {
            let (_, predicted_max_shard_bytes, predicted_max_shard_launch_bytes) =
                predict_plan_fit_for_nodes(
                    &ranking_vec,
                    analysis,
                    recommended_nodes,
                    required_experts_per_node,
                )?;
            assumptions.push(format!(
                    "Estimated shard launch for {} nodes from analysis.json: {:.1}GB max per node ({:.1}GB raw shard bytes)",
                    recommended_nodes,
                    predicted_max_shard_launch_bytes as f64 / 1e9,
                    predicted_max_shard_bytes as f64 / 1e9
                ));
        }
        if args.nodes.is_none() && !fit.any_fit_exists() {
            assumptions.push(
                    "No node count up to the current max useful node count fits this VRAM target under the conservative shared-core heuristic."
                        .to_string(),
                );
        } else if args.nodes.is_some() && !feasible {
            assumptions.push(format!(
                    "Requested node count {} does not fit this VRAM target under the current shared-core heuristic.",
                    recommended_nodes
                ));
        }

        (
            recommended_nodes,
            max_supported_nodes,
            feasible,
            assumptions,
        )
    } else {
        let recommended_nodes = args.nodes.unwrap_or(heuristic_recommended_nodes);
        let max_supported_nodes =
            (model.expert_count / model.min_experts_per_node.max(1)).max(1) as usize;
        let feasible = recommended_nodes <= max_supported_nodes;
        let assumptions = vec![
            format!(
                "Minimum nodes estimated from total model bytes / target VRAM: {:.1}GB / {:.1}GB",
                model.total_model_bytes as f64 / 1e9,
                target_vram_bytes as f64 / 1e9
            ),
            format!(
                "Minimum experts per node falls back to local planner state: {}",
                model.min_experts_per_node
            ),
            "No analysis.json was available, so shard byte fit uses the legacy heuristic."
                .to_string(),
        ];
        (
            recommended_nodes,
            max_supported_nodes,
            feasible,
            assumptions,
        )
    };

    let assignments = moe::compute_assignments_with_overlap(
        &ranking_vec,
        recommended_nodes,
        model.min_experts_per_node,
        1,
    );
    let profile = load_analyze_mass_profile(&ranking.path).ok();
    let (shared_mass_pct, max_node_mass_pct, min_node_mass_pct) = if let Some(ref profile) = profile
    {
        let shared = assignments
            .first()
            .map(|assignment| assignment.n_shared)
            .unwrap_or_default();
        let shared_mass_pct = mass_pct_for_experts(
            profile,
            &profile.ranking[..shared.min(profile.ranking.len())],
        );
        let node_mass: Vec<f64> = assignments
            .iter()
            .map(|assignment| mass_pct_for_experts(profile, &assignment.experts))
            .collect();
        (
            Some(shared_mass_pct),
            node_mass.iter().copied().reduce(f64::max),
            node_mass.iter().copied().reduce(f64::min),
        )
    } else {
        (None, None, None)
    };
    if profile.is_none() {
        assumptions.push(
            "Ranking file does not include gate-mass columns, so shared/node mass percentages are unavailable."
                .to_string(),
        );
    }

    Ok(MoePlanReport {
        model,
        ranking,
        target_vram_bytes,
        recommended_nodes,
        max_supported_nodes,
        feasible,
        assumptions,
        assignments,
        shared_mass_pct,
        max_node_mass_pct,
        min_node_mass_pct,
    })
}

pub(crate) async fn resolve_model_context(model_spec: &str) -> Result<MoeModelContext> {
    resolve_model_context_with_progress(model_spec, true).await
}

pub(crate) async fn resolve_model_context_with_progress(
    model_spec: &str,
    progress: bool,
) -> Result<MoeModelContext> {
    let path = if progress {
        resolve_model_spec(Path::new(model_spec)).await?
    } else {
        resolve_model_spec_with_progress(Path::new(model_spec), false).await?
    };
    let info = moe::detect_moe(&path).with_context(|| {
        format!(
            "Model is not auto-detected as MoE from the GGUF header: {}",
            path.display()
        )
    })?;
    let display_name = model_display_name(&path);
    let identity = huggingface_identity_for_path(&path);
    let source_repo = identity.as_ref().map(|identity| identity.repo_id.clone());
    let source_revision = identity.as_ref().map(|identity| identity.revision.clone());
    let distribution_id = identity
        .as_ref()
        .map(|identity| normalize_distribution_id(&identity.local_file_name))
        .unwrap_or_else(|| normalize_distribution_id(&display_name));
    let min_experts_per_node = bundled_min_experts(&display_name)
        .unwrap_or_else(|| ((info.expert_count as f64) * 0.5).ceil() as u32);
    Ok(MoeModelContext {
        input: model_spec.to_string(),
        total_model_bytes: election::total_model_bytes(&path),
        path,
        display_name,
        source_repo,
        source_revision,
        distribution_id,
        expert_count: info.expert_count,
        used_expert_count: info.expert_used_count,
        min_experts_per_node,
    })
}

pub(crate) fn resolve_target_vram_bytes(max_vram_gb: Option<f64>) -> u64 {
    crate::mesh::detect_vram_bytes_capped(max_vram_gb)
}

pub(crate) fn normalize_distribution_id(name: &str) -> String {
    let stem = Path::new(name)
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or(name)
        .trim_end_matches(".gguf");
    if let Some((prefix, suffix)) = stem.rsplit_once("-of-") {
        let has_numeric_suffix = suffix.len() == 5 && suffix.chars().all(|ch| ch.is_ascii_digit());
        let has_numeric_prefix = prefix.len() > 6
            && prefix[prefix.len() - 6..].starts_with('-')
            && prefix[prefix.len() - 5..]
                .chars()
                .all(|ch| ch.is_ascii_digit());
        if has_numeric_suffix && has_numeric_prefix {
            return prefix[..prefix.len() - 6].to_string();
        }
    }
    stem.to_string()
}

pub(crate) fn local_submit_ranking(
    model: &MoeModelContext,
    ranking_file: Option<&Path>,
) -> Result<ResolvedRanking> {
    if let Some(path) = ranking_file {
        if !path.exists() {
            bail!("Ranking file not found: {}", path.display());
        }
        let inferred = infer_analyzer_from_ranking_path(path).ok_or_else(|| {
            anyhow::anyhow!(
                "Could not infer analyzer id from {}. Use a ranking CSV produced by `mesh-llm moe analyze` or a path containing `micro-v1` or `full-v1`.",
                path.display()
            )
        })?;
        let metadata_path = sibling_metadata_path(path);
        let analysis_path = sibling_analysis_path(path);
        return Ok(ResolvedRanking {
            path: path.to_path_buf(),
            metadata_path,
            analysis_path,
            analyzer_id: inferred.to_string(),
            source: RankingSource::Override,
            reason: "explicit --ranking-file override".to_string(),
        });
    }

    let Some(artifact) = moe::best_shared_ranking_artifact(&model.path) else {
        bail!(
                "No local ranking artifact found for {}. Run `mesh-llm moe analyze full {}` or `mesh-llm moe analyze micro {}` first.",
                model.display_name,
                model.input,
                model.input
            );
    };
    let path = moe::shared_ranking_cache_path(&model.path, &artifact);
    let analyzer_id = match artifact.kind {
        moe::SharedRankingKind::Analyze => "full-v1",
        moe::SharedRankingKind::MicroAnalyze => "micro-v1",
    };
    Ok(ResolvedRanking {
        analysis_path: sibling_analysis_path(&path),
        path,
        metadata_path: None,
        analyzer_id: analyzer_id.to_string(),
        source: RankingSource::LocalCache,
        reason: "best local cached ranking artifact".to_string(),
    })
}

pub(crate) fn validate_ranking(model: &MoeModelContext, ranking: &ResolvedRanking) -> Result<()> {
    let loaded = moe::load_cached_ranking(&ranking.path)
        .ok_or_else(|| anyhow::anyhow!("Could not parse ranking {}", ranking.path.display()))?;
    let artifact = moe::SharedRankingArtifact {
        kind: ranking_kind_for_analyzer(&ranking.analyzer_id),
        origin: moe::SharedRankingOrigin::LegacyCache,
        ranking: loaded,
        micro_prompt_count: None,
        micro_tokens: None,
        micro_layer_scope: None,
    };
    moe::validate_shared_ranking_artifact(&model.path, &artifact)?;
    load_analyze_mass_profile(&ranking.path).with_context(|| {
        format!(
            "Ranking {} must include gate-mass columns",
            ranking.path.display()
        )
    })?;
    Ok(())
}

fn infer_analyzer_from_ranking_path(path: &Path) -> Option<&'static str> {
    let text = path.to_string_lossy().to_ascii_lowercase();
    if text.contains("/full-v1/") || text.contains("\\full-v1\\") {
        return Some("full-v1");
    }
    if text.contains("/micro-v1/") || text.contains("\\micro-v1\\") {
        return Some("micro-v1");
    }

    let file_name = path.file_name()?.to_string_lossy().to_ascii_lowercase();
    if file_name.contains("micro-v1") {
        return Some("micro-v1");
    }
    if file_name.contains("full-v1") {
        return Some("full-v1");
    }
    if file_name.starts_with("local-")
        && file_name.contains(".micro-p")
        && file_name.ends_with(".csv")
    {
        return Some("micro-v1");
    }
    if file_name.starts_with("local-") && file_name.ends_with(".csv") {
        return Some("full-v1");
    }

    None
}

fn ranking_kind_for_analyzer(analyzer_id: &str) -> moe::SharedRankingKind {
    if analyzer_id.starts_with("micro") {
        moe::SharedRankingKind::MicroAnalyze
    } else {
        moe::SharedRankingKind::Analyze
    }
}

fn sibling_metadata_path(path: &Path) -> Option<PathBuf> {
    let parent = path.parent()?;
    let metadata = parent.join("metadata.json");
    metadata.exists().then_some(metadata)
}

fn sibling_analysis_path(path: &Path) -> Option<PathBuf> {
    let parent = path.parent()?;
    let analysis = parent.join(ANALYSIS_JSON_FILENAME);
    analysis.exists().then_some(analysis)
}

pub(crate) fn resolve_runtime_ranking(
    model_path: &Path,
    dataset_repo_name: &str,
) -> Result<Option<ResolvedRanking>> {
    let local_legacy = resolve_local_runtime_ranking(model_path);

    let Some(identity) = huggingface_identity_for_path(model_path) else {
        return Ok(local_legacy);
    };

    if local_legacy
        .as_ref()
        .is_some_and(|ranking| ranking_method_priority(ranking) >= 2)
    {
        // A local full-v1 ranking is treated as the current best tie-break because
        // we do not have a comparable freshness signal for local cache entries.
        return Ok(local_legacy);
    }

    let remote = match fetch_remote_ranking(
        dataset_repo_name,
        &identity.repo_id,
        &identity.revision,
        &normalize_distribution_id(&identity.local_file_name),
        false,
    ) {
        Ok(remote) => remote,
        Err(error) => return local_legacy.ok_or(error).map(Some),
    };
    Ok(select_preferred_ranking(local_legacy, remote))
}

type AnalysisDetails = (Option<&'static str>, Option<usize>, u32, u32, bool);

fn resolve_local_runtime_ranking(model_path: &Path) -> Option<ResolvedRanking> {
    moe::best_shared_ranking_artifact(model_path).map(|artifact| {
        let analyzer_id = match artifact.kind {
            moe::SharedRankingKind::Analyze => "full-v1",
            moe::SharedRankingKind::MicroAnalyze => "micro-v1",
        };
        ResolvedRanking {
            path: moe::shared_ranking_cache_path(model_path, &artifact),
            metadata_path: None,
            analysis_path: sibling_analysis_path(&moe::shared_ranking_cache_path(
                model_path, &artifact,
            )),
            analyzer_id: analyzer_id.to_string(),
            source: RankingSource::LocalCache,
            reason: format!(
                "local {} ranking cache",
                if artifact.kind == moe::SharedRankingKind::Analyze {
                    "full"
                } else {
                    "micro"
                }
            ),
        }
    })
}

async fn resolve_best_ranking(
    model: &MoeModelContext,
    args: &MoePlanArgs,
) -> Result<ResolvedRanking> {
    if let Some(path) = args.ranking_file.as_deref() {
        if !path.exists() {
            bail!("Ranking file not found: {}", path.display());
        }
        let metadata_path = sibling_metadata_path(path);
        let analysis_path = sibling_analysis_path(path);
        let analyzer_id = infer_analyzer_from_ranking_path(path)
            .unwrap_or("override")
            .to_string();
        return Ok(ResolvedRanking {
            path: path.to_path_buf(),
            metadata_path,
            analysis_path,
            analyzer_id,
            source: RankingSource::Override,
            reason: "explicit --ranking-file override".to_string(),
        });
    }

    let local_legacy = resolve_local_runtime_ranking(&model.path);

    let Some(source_repo) = model.source_repo.as_ref() else {
        return local_legacy.ok_or_else(|| {
            anyhow::anyhow!(
                "No published ranking lookup is possible for non-HF model {} and no local ranking cache exists.",
                model.display_name
            )
        });
    };
    let Some(source_revision) = model.source_revision.as_ref() else {
        return local_legacy.ok_or_else(|| {
            anyhow::anyhow!(
                "No published ranking lookup is possible without a resolved source revision for {}.",
                model.display_name
            )
        });
    };

    if local_legacy
        .as_ref()
        .is_some_and(|ranking| ranking_method_priority(ranking) >= 2)
    {
        // Prefer local full-v1 on tie: remote is only selected when it is stronger.
        return Ok(local_legacy.expect("checked is_some above"));
    }

    let dataset_repo = args.dataset_repo.clone();
    let source_repo = source_repo.clone();
    let source_revision = source_revision.clone();
    let distribution_id = model.distribution_id.clone();
    let local_fallback = local_legacy.clone();
    let remote_dataset_repo = dataset_repo.clone();
    let remote_source_repo = source_repo.clone();
    let remote_source_revision = source_revision.clone();
    let remote_distribution_id = distribution_id.clone();
    let remote_progress = args.progress;
    let mut remote_spinner = args.progress.then(|| {
        start_spinner(&format!(
            "Checking published MoE data for {}",
            model.display_name
        ))
    });
    if let Some(spinner) = remote_spinner.as_mut() {
        spinner.set_message(format!(
            "Fetching published MoE ranking and analysis for {}",
            model.display_name
        ));
    }
    let remote_lookup = tokio::task::spawn_blocking(move || {
        fetch_remote_ranking(
            &remote_dataset_repo,
            &remote_source_repo,
            &remote_source_revision,
            &remote_distribution_id,
            remote_progress,
        )
    })
    .await;
    if let Some(spinner) = remote_spinner.as_mut() {
        spinner.finish();
    }
    let remote_lookup =
        remote_lookup.context("Join blocking Hugging Face MoE ranking lookup task")?;

    let remote = match remote_lookup {
        Ok(remote) => remote,
        Err(error) => {
            return local_fallback.ok_or(error);
        }
    };
    select_preferred_ranking(local_legacy, remote).ok_or_else(|| {
        anyhow::anyhow!(
            "No published ranking found in {} for {}@{} {} and no local cache exists.",
            args.dataset_repo,
            source_repo,
            source_revision,
            model.distribution_id
        )
    })
}

fn ranking_method_priority(ranking: &ResolvedRanking) -> u8 {
    if ranking.analyzer_id.starts_with("full") {
        2
    } else if ranking.analyzer_id.starts_with("micro") {
        1
    } else {
        0
    }
}

fn select_preferred_ranking(
    local: Option<ResolvedRanking>,
    remote: Option<ResolvedRanking>,
) -> Option<ResolvedRanking> {
    match (local, remote) {
        (Some(local), Some(remote)) => {
            if ranking_method_priority(&local) >= ranking_method_priority(&remote) {
                Some(local)
            } else {
                Some(remote)
            }
        }
        (Some(local), None) => Some(local),
        (None, Some(remote)) => Some(remote),
        (None, None) => None,
    }
}

fn fetch_remote_ranking(
    dataset_repo_name: &str,
    source_repo: &str,
    source_revision: &str,
    distribution_id: &str,
    progress: bool,
) -> Result<Option<ResolvedRanking>> {
    let api = build_hf_api(progress).context("Build Hugging Face client for MoE ranking lookup")?;
    let (owner, name) = dataset_repo_name
        .split_once('/')
        .unwrap_or(("", dataset_repo_name));
    let dataset_repo = api.dataset(owner, name);
    let info = dataset_repo.info(
        &RepoInfoParams::builder()
            .revision(DEFAULT_DATASET_REVISION.to_string())
            .build(),
    )?;
    let hf_hub::RepoInfo::Dataset(info) = info else {
        bail!("Expected dataset repo info for {}", dataset_repo_name);
    };
    find_remote_ranking(
        &api,
        dataset_repo_name,
        info.sha.as_deref().unwrap_or(DEFAULT_DATASET_REVISION),
        info.siblings.as_deref().unwrap_or(&[]),
        source_repo,
        source_revision,
        distribution_id,
    )
}

fn find_remote_ranking(
    api: &hf_hub::HFClientSync,
    dataset_repo: &str,
    dataset_revision: &str,
    siblings: &[hf_hub::RepoSibling],
    source_repo: &str,
    source_revision: &str,
    distribution_id: &str,
) -> Result<Option<ResolvedRanking>> {
    let prefix = canonical_dataset_prefix(source_repo, source_revision, distribution_id);
    for analyzer_id in ["full-v1", "micro-v1"] {
        let metadata_rel = format!("{prefix}/{analyzer_id}/metadata.json");
        let ranking_rel = format!("{prefix}/{analyzer_id}/ranking.csv");
        let analysis_rel = format!("{prefix}/{analyzer_id}/{ANALYSIS_JSON_FILENAME}");
        let has_metadata = siblings.iter().any(|entry| entry.rfilename == metadata_rel);
        let has_ranking = siblings.iter().any(|entry| entry.rfilename == ranking_rel);
        let has_analysis = siblings.iter().any(|entry| entry.rfilename == analysis_rel);
        if !(has_metadata && has_ranking && has_analysis) {
            continue;
        }

        let (owner, name) = dataset_repo.split_once('/').unwrap_or(("", dataset_repo));
        let pinned = api.dataset(owner, name);
        let metadata_path = pinned
            .download_file(
                &RepoDownloadFileParams::builder()
                    .filename(metadata_rel.clone())
                    .revision(dataset_revision.to_string())
                    .build(),
            )
            .with_context(|| format!("Download {}", metadata_rel))?;
        let ranking_path = pinned
            .download_file(
                &RepoDownloadFileParams::builder()
                    .filename(ranking_rel.clone())
                    .revision(dataset_revision.to_string())
                    .build(),
            )
            .with_context(|| format!("Download {}", ranking_rel))?;
        let analysis_path = pinned
            .download_file(
                &RepoDownloadFileParams::builder()
                    .filename(analysis_rel.clone())
                    .revision(dataset_revision.to_string())
                    .build(),
            )
            .with_context(|| format!("Download {}", analysis_rel))?;
        return Ok(Some(ResolvedRanking {
            analysis_path: Some(analysis_path),
            path: ranking_path,
            metadata_path: Some(metadata_path),
            analyzer_id: analyzer_id.to_string(),
            source: RankingSource::HuggingFaceDataset,
            reason: format!("published {} ranking in {}", analyzer_id, dataset_repo),
        }));
    }

    Ok(None)
}

fn canonical_dataset_prefix(
    source_repo: &str,
    source_revision: &str,
    distribution_id: &str,
) -> String {
    let (namespace, repo) = source_repo
        .split_once('/')
        .unwrap_or(("unknown", source_repo));
    format!("data/{namespace}/{repo}/{source_revision}/gguf/{distribution_id}")
}

pub(crate) fn canonical_dataset_prefix_for_model(model: &MoeModelContext) -> Result<String> {
    let Some(source_repo) = model.source_repo.as_ref() else {
        bail!("A Hugging Face-backed model is required to derive the canonical dataset path.");
    };
    let Some(source_revision) = model.source_revision.as_ref() else {
        bail!("A resolved source revision is required to derive the canonical dataset path.");
    };
    Ok(canonical_dataset_prefix(
        source_repo,
        source_revision,
        &model.distribution_id,
    ))
}

pub(crate) fn build_submit_bundle(
    model: &MoeModelContext,
    ranking: &ResolvedRanking,
    log_path: Option<&Path>,
) -> Result<MoeSubmitBundle> {
    let dataset_prefix = format!(
        "{}/{}",
        canonical_dataset_prefix_for_model(model)?,
        ranking.analyzer_id
    );
    let ranking_rel = format!("{dataset_prefix}/ranking.csv");
    let metadata_rel = format!("{dataset_prefix}/metadata.json");
    let analysis_rel = format!("{dataset_prefix}/{ANALYSIS_JSON_FILENAME}");
    let log_rel = format!("{dataset_prefix}/run.log");

    let model_files = discover_distribution_files(model)?;
    let metadata_content = if let Some(existing) = ranking.metadata_path.as_ref() {
        fs::read_to_string(existing)
            .with_context(|| format!("Read existing metadata {}", existing.display()))?
    } else {
        serde_json::to_string_pretty(&build_metadata_json(model, ranking, &model_files)?)? + "\n"
    };
    let analysis_content = if let Some(existing) = ranking.analysis_path.as_ref() {
        fs::read_to_string(existing)
            .with_context(|| format!("Read existing analysis {}", existing.display()))?
    } else {
        serde_json::to_string_pretty(&build_analysis_json(model, ranking, &model_files)?)? + "\n"
    };

    let mut dataset_paths = vec![ranking_rel, metadata_rel, analysis_rel];
    let log_path = log_path.filter(|path| path.exists()).map(Path::to_path_buf);
    if log_path.is_some() {
        dataset_paths.push(log_rel);
    }

    let source_repo = model.source_repo.clone().unwrap_or_default();
    let source_revision = model.source_revision.clone().unwrap_or_default();
    Ok(MoeSubmitBundle {
        dataset_prefix,
        dataset_paths,
        ranking_path: ranking.path.clone(),
        metadata_content,
        analysis_content,
        log_path,
        commit_message: format!(
            "Add {} {} for {}@{}",
            model.distribution_id,
            ranking.analyzer_id,
            source_repo,
            short_revision(&source_revision)
        ),
        commit_description: format!(
            "Publish {} ranking artifacts for {} ({})",
            ranking.analyzer_id, model.display_name, model.input
        ),
    })
}

fn short_revision(revision: &str) -> &str {
    if revision.len() <= 12 {
        revision
    } else {
        &revision[..12]
    }
}

fn build_metadata_json(
    model: &MoeModelContext,
    ranking: &ResolvedRanking,
    model_files: &[(String, PathBuf)],
) -> Result<Value> {
    let Some(source_repo) = model.source_repo.as_ref() else {
        bail!("A Hugging Face-backed model is required to generate metadata.");
    };
    let Some(source_revision) = model.source_revision.as_ref() else {
        bail!("A resolved source revision is required to generate metadata.");
    };
    let primary_file = model_files
        .first()
        .map(|(repo_path, _)| repo_path.clone())
        .unwrap_or_else(|| model.distribution_id.clone());
    let all_files = model_files
        .iter()
        .map(|(repo_path, _)| repo_path.clone())
        .collect::<Vec<_>>();
    let total_files = model_files.len();
    if total_files > 0 {
        eprintln!(
            "📦 Computing SHA-256 hashes for {} GGUF file(s) while building metadata...",
            total_files
        );
    }
    let mut file_hashes = serde_json::Map::with_capacity(total_files);
    for (index, (repo_path, path)) in model_files.iter().enumerate() {
        eprintln!("   [{}/{}] Hashing {}", index + 1, total_files, repo_path);
        file_hashes.insert(repo_path.clone(), Value::String(sha256_file(path)?));
    }

    let (prompt_set, prompt_count, token_count, context_size, all_layers) =
        infer_analysis_details(ranking, model)?;

    Ok(json!({
        "schema_version": 1,
        "source_repo": source_repo,
        "source_revision": source_revision,
        "format": "gguf",
        "distribution_id": model.distribution_id,
        "analyzer_id": ranking.analyzer_id,
        "analysis_tool": "llama-moe-analyze",
        "ranking_path": "ranking.csv",
        "primary_file": primary_file,
        "all_files": all_files,
        "file_hashes": file_hashes,
        "prompt_set": prompt_set,
        "prompt_count": prompt_count,
        "token_count": token_count,
        "all_layers": all_layers,
        "command": {
            "context_size": context_size,
            "token_count": token_count,
            "analyzer_id": ranking.analyzer_id,
        },
        "created_at": iso8601_now(),
        "status": "complete",
    }))
}

fn build_analysis_json(
    model: &MoeModelContext,
    ranking: &ResolvedRanking,
    model_files: &[(String, PathBuf)],
) -> Result<Value> {
    let mass_profile = load_analyze_mass_profile(&ranking.path)?;
    let tensor_profile = aggregate_tensor_byte_profile(model_files)?;
    Ok(json!({
        "schema_version": 1,
        "ranking": {
            "sha256": sha256_file(&ranking.path)?,
            "rows": mass_profile.ranking.len(),
            "mass_checkpoints": build_mass_checkpoints(&mass_profile),
        },
        "model": {
            "expert_count": model.expert_count,
            "expert_used_count": model.used_expert_count,
        },
        "memory": {
            "full_model_bytes": tensor_profile.full_model_bytes,
            "base_resident_bytes": tensor_profile.base_resident_bytes,
            "shard_file_overhead_bytes": tensor_profile.file_overhead_bytes,
            "expert_tensor_bytes_total": tensor_profile.expert_tensor_bytes,
            "expert_bytes": expert_bytes_json(tensor_profile.expert_tensor_bytes, model.expert_count)?,
        }
    }))
}

pub(crate) fn read_analysis_json(path: &Path) -> Result<MoeAnalysisJson> {
    let content =
        fs::read_to_string(path).with_context(|| format!("Read analysis {}", path.display()))?;
    let parsed: MoeAnalysisJson = serde_json::from_str(&content)
        .with_context(|| format!("Parse analysis {}", path.display()))?;
    if parsed.schema_version != 1 {
        bail!(
            "Unsupported analysis schema {} in {}",
            parsed.schema_version,
            path.display()
        );
    }
    Ok(parsed)
}

pub(crate) fn estimate_startup_fit_from_analysis(
    ranking_path: &Path,
    analysis: &MoeAnalysisJson,
    target_vram_bytes: u64,
    analyzer_id: &str,
    ranking_source: &'static str,
) -> Result<MoeStartupFitEstimate> {
    let ranking = moe::load_cached_ranking(ranking_path)
        .ok_or_else(|| anyhow::anyhow!("Could not parse ranking {}", ranking_path.display()))?;
    if analysis.ranking.rows != ranking.len() {
        bail!(
            "Ranking row count mismatch for {}: analysis says {}, CSV has {}",
            ranking_path.display(),
            analysis.ranking.rows,
            ranking.len()
        );
    }

    let required_experts_per_node = default_required_experts_per_node(analysis.model.expert_count);
    let full_model_launch_bytes = apply_model_load_headroom(analysis.memory.full_model_bytes);
    let mut estimate = MoeStartupFitEstimate {
        analyzer_id: analyzer_id.to_string(),
        ranking_source,
        target_vram_bytes,
        required_experts_per_node,
        full_model_bytes: analysis.memory.full_model_bytes,
        full_model_launch_bytes,
        recommended_nodes: None,
        predicted_max_shard_bytes: None,
        predicted_max_shard_launch_bytes: None,
    };

    if estimate.full_model_fits() {
        estimate.recommended_nodes = Some(1);
        return Ok(estimate);
    }

    let max_supported_nodes =
        (analysis.model.expert_count / required_experts_per_node.max(1)).max(1) as usize;
    let mut best_failed_shard: Option<(u64, u64)> = None;
    for nodes in 2..=max_supported_nodes {
        let assignments =
            moe::compute_assignments_with_overlap(&ranking, nodes, required_experts_per_node, 1);
        let max_shard_bytes = assignments
            .iter()
            .map(|assignment| predict_shard_bytes(&analysis.memory, &assignment.experts))
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .max()
            .unwrap_or_default();
        let max_shard_launch_bytes = apply_model_load_headroom(max_shard_bytes);
        if max_shard_launch_bytes <= target_vram_bytes {
            estimate.recommended_nodes = Some(nodes);
            estimate.predicted_max_shard_bytes = Some(max_shard_bytes);
            estimate.predicted_max_shard_launch_bytes = Some(max_shard_launch_bytes);
            return Ok(estimate);
        }
        best_failed_shard = Some((max_shard_bytes, max_shard_launch_bytes));
    }

    if let Some((max_shard_bytes, max_shard_launch_bytes)) = best_failed_shard {
        estimate.predicted_max_shard_bytes = Some(max_shard_bytes);
        estimate.predicted_max_shard_launch_bytes = Some(max_shard_launch_bytes);
    } else if max_supported_nodes == 1 {
        let max_shard_bytes = predict_shard_bytes(&analysis.memory, &ranking)?;
        estimate.predicted_max_shard_bytes = Some(max_shard_bytes);
        estimate.predicted_max_shard_launch_bytes =
            Some(apply_model_load_headroom(max_shard_bytes));
    }

    Ok(estimate)
}

pub(crate) fn fetch_remote_startup_fit(
    dataset_repo_name: &str,
    source_repo: &str,
    source_revision: &str,
    distribution_id: &str,
    target_vram_bytes: u64,
    progress: bool,
) -> Result<Option<MoeStartupFitEstimate>> {
    let Some(ranking) = fetch_remote_ranking(
        dataset_repo_name,
        source_repo,
        source_revision,
        distribution_id,
        progress,
    )?
    else {
        return Ok(None);
    };
    let Some(analysis_path) = ranking.analysis_path.as_ref() else {
        return Ok(None);
    };
    let analysis = read_analysis_json(analysis_path)?;
    let estimate = estimate_startup_fit_from_analysis(
        &ranking.path,
        &analysis,
        target_vram_bytes,
        &ranking.analyzer_id,
        ranking.source.label(),
    )?;
    Ok(Some(estimate))
}

fn build_mass_checkpoints(profile: &AnalyzeMassProfile) -> Vec<Value> {
    let mut checkpoints = DEFAULT_MASS_CHECKPOINTS
        .into_iter()
        .filter(|top_n| *top_n <= profile.masses.len())
        .collect::<Vec<_>>();
    if !checkpoints.contains(&profile.masses.len()) {
        checkpoints.push(profile.masses.len());
    }
    checkpoints
        .into_iter()
        .map(|top_n| {
            let captured_mass = profile
                .masses
                .iter()
                .take(top_n)
                .map(|(_, mass)| *mass)
                .sum::<f64>();
            json!({
                "top_n": top_n,
                "mass_fraction": if profile.total_mass > f64::EPSILON {
                    captured_mass / profile.total_mass
                } else {
                    0.0
                }
            })
        })
        .collect()
}

fn aggregate_tensor_byte_profile(
    model_files: &[(String, PathBuf)],
) -> Result<AggregatedTensorByteProfile> {
    let mut aggregated = AggregatedTensorByteProfile::default();
    for (_, path) in model_files {
        let profile: GgufTensorByteProfile =
            crate::models::gguf::scan_gguf_tensor_byte_profile(path).ok_or_else(|| {
                anyhow::anyhow!("Could not derive GGUF tensor bytes for {}", path.display())
            })?;
        aggregated.full_model_bytes = aggregated
            .full_model_bytes
            .saturating_add(profile.full_model_bytes);
        aggregated.base_resident_bytes = aggregated
            .base_resident_bytes
            .saturating_add(profile.base_resident_bytes);
        aggregated.expert_tensor_bytes = aggregated
            .expert_tensor_bytes
            .saturating_add(profile.expert_tensor_bytes);
        aggregated.file_overhead_bytes = aggregated
            .file_overhead_bytes
            .saturating_add(profile.file_overhead_bytes);
    }
    Ok(aggregated)
}

fn expert_bytes_json(expert_tensor_bytes: u64, expert_count: u32) -> Result<Value> {
    if expert_count == 0 {
        bail!("cannot derive expert bytes for expert_count=0");
    }
    let expert_count_u64 = u64::from(expert_count);
    if expert_tensor_bytes.is_multiple_of(expert_count_u64) {
        return Ok(json!({
            "kind": "uniform",
            "bytes_per_expert": expert_tensor_bytes / expert_count_u64,
        }));
    }

    let base = expert_tensor_bytes / expert_count_u64;
    let remainder = (expert_tensor_bytes % expert_count_u64) as usize;
    let mut values = vec![base; expert_count as usize];
    for value in values.iter_mut().take(remainder) {
        *value += 1;
    }
    Ok(json!({
        "kind": "dense_per_expert",
        "values": values,
    }))
}

fn default_required_experts_per_node(expert_count: u32) -> u32 {
    ((expert_count as f64) * 0.5).ceil() as u32
}

fn apply_model_load_headroom(bytes: u64) -> u64 {
    bytes
        .saturating_mul(MODEL_LOAD_HEADROOM_NUMERATOR)
        .div_ceil(MODEL_LOAD_HEADROOM_DENOMINATOR)
}

fn predict_plan_fit_for_nodes(
    ranking: &[u32],
    analysis: &MoeAnalysisJson,
    nodes: usize,
    required_experts_per_node: u32,
) -> Result<(Vec<moe::NodeAssignment>, u64, u64)> {
    let assignments =
        moe::compute_assignments_with_overlap(ranking, nodes, required_experts_per_node, 1);
    let max_shard_bytes = assignments
        .iter()
        .map(|assignment| predict_shard_bytes(&analysis.memory, &assignment.experts))
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .max()
        .unwrap_or_default();
    Ok((
        assignments,
        max_shard_bytes,
        apply_model_load_headroom(max_shard_bytes),
    ))
}

fn predict_shard_bytes(memory: &MoeAnalysisMemory, experts: &[u32]) -> Result<u64> {
    let mut total = memory
        .base_resident_bytes
        .saturating_add(memory.shard_file_overhead_bytes);
    for expert_id in experts {
        total = total
            .checked_add(memory.expert_bytes.bytes_for(*expert_id)?)
            .ok_or_else(|| anyhow::anyhow!("predicted shard bytes overflow"))?;
    }
    Ok(total)
}

impl MoeAnalysisExpertBytes {
    fn bytes_for(&self, expert_id: u32) -> Result<u64> {
        match self {
            Self::Uniform { bytes_per_expert } => Ok(*bytes_per_expert),
            Self::DensePerExpert { values } => values
                .get(expert_id as usize)
                .copied()
                .ok_or_else(|| anyhow::anyhow!("expert id {} missing from analysis", expert_id)),
        }
    }
}

pub(crate) fn write_analysis_json(
    model: &MoeModelContext,
    ranking_path: &Path,
    analyzer_id: &str,
) -> Result<PathBuf> {
    let ranking = ResolvedRanking {
        path: ranking_path.to_path_buf(),
        metadata_path: None,
        analysis_path: None,
        analyzer_id: analyzer_id.to_string(),
        source: RankingSource::LocalCache,
        reason: "local analysis artifact".to_string(),
    };
    let model_files = discover_distribution_files(model)?;
    let content =
        serde_json::to_string_pretty(&build_analysis_json(model, &ranking, &model_files)?)? + "\n";
    let analysis_path = ranking_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(ANALYSIS_JSON_FILENAME);
    if let Some(parent) = analysis_path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("Create {}", parent.display()))?;
    }
    fs::write(&analysis_path, content)
        .with_context(|| format!("Write {}", analysis_path.display()))?;
    Ok(analysis_path)
}

fn infer_analysis_details(
    ranking: &ResolvedRanking,
    model: &MoeModelContext,
) -> Result<AnalysisDetails> {
    let context_size_default = 4096;
    let log_path = analysis_log_path(model, &ranking.analyzer_id);
    let log_text = log_path
        .as_ref()
        .filter(|path| path.exists())
        .map(fs::read_to_string)
        .transpose()
        .with_context(|| "Read local MoE analysis log".to_string())?;

    let token_count = extract_first_arg_value(log_text.as_deref(), "-n")
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or_else(|| {
            if ranking.analyzer_id.starts_with("micro") {
                128
            } else {
                32
            }
        });
    let context_size = extract_first_arg_value(log_text.as_deref(), "-c")
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(context_size_default);
    let all_layers = log_text
        .as_deref()
        .map(|text| text.contains("--all-layers"))
        .unwrap_or(true);

    if ranking.analyzer_id.starts_with("micro") {
        let prompt_count = log_text
            .as_deref()
            .map(|text| text.matches("[prompt ").count())
            .filter(|count| *count > 0)
            .unwrap_or(8);
        Ok((
            Some("meshllm-micro-v1"),
            Some(prompt_count),
            token_count,
            context_size,
            all_layers,
        ))
    } else {
        Ok((None, None, token_count, context_size, all_layers))
    }
}

fn extract_first_arg_value<'a>(text: Option<&'a str>, flag: &str) -> Option<&'a str> {
    let text = text?;
    let parts = text.split_whitespace().collect::<Vec<_>>();
    parts
        .windows(2)
        .find_map(|window| (window[0] == flag).then_some(window[1]))
}

fn analysis_log_path(model: &MoeModelContext, analyzer_id: &str) -> Option<PathBuf> {
    let stem = model
        .path
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("model");
    Some(
        crate::models::mesh_llm_cache_dir()
            .join("moe")
            .join("logs")
            .join(format!("{stem}.{analyzer_id}.log")),
    )
}

fn discover_distribution_files(model: &MoeModelContext) -> Result<Vec<(String, PathBuf)>> {
    let Some(identity) = huggingface_identity_for_path(&model.path) else {
        return Ok(vec![(
            model
                .path
                .file_name()
                .and_then(|value| value.to_str())
                .unwrap_or("model.gguf")
                .to_string(),
            model.path.clone(),
        )]);
    };

    let snapshot_root = snapshot_root_for_relative_file(&model.path, &identity.file)
        .unwrap_or_else(|| {
            model
                .path
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .to_path_buf()
        });
    let mut files = Vec::new();
    collect_distribution_files(
        &snapshot_root,
        &snapshot_root,
        &model.distribution_id,
        &mut files,
    )?;
    if files.is_empty() {
        files.push((identity.file.clone(), model.path.clone()));
    }
    files.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(files)
}

fn snapshot_root_for_relative_file(path: &Path, relative_file: &str) -> Option<PathBuf> {
    let mut root = path.to_path_buf();
    for _ in Path::new(relative_file).components() {
        root = root.parent()?.to_path_buf();
    }
    Some(root)
}

fn collect_distribution_files(
    snapshot_root: &Path,
    current: &Path,
    distribution_id: &str,
    files: &mut Vec<(String, PathBuf)>,
) -> Result<()> {
    for entry in fs::read_dir(current).with_context(|| format!("Read {}", current.display()))? {
        let entry = entry?;
        let path = entry.path();
        if entry.file_type()?.is_dir() {
            collect_distribution_files(snapshot_root, &path, distribution_id, files)?;
            continue;
        }
        if path.extension().and_then(|ext| ext.to_str()) != Some("gguf") {
            continue;
        }
        let relative = path
            .strip_prefix(snapshot_root)
            .unwrap_or(&path)
            .to_string_lossy()
            .replace('\\', "/");
        if normalize_distribution_id(&relative) == distribution_id {
            files.push((relative, path));
        }
    }
    Ok(())
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut digest = Sha256::new();
    let mut file = fs::File::open(path).with_context(|| format!("Open {}", path.display()))?;
    let mut buf = [0u8; 1024 * 1024];
    loop {
        let read = std::io::Read::read(&mut file, &mut buf)
            .with_context(|| format!("Hash {}", path.display()))?;
        if read == 0 {
            break;
        }
        digest.update(&buf[..read]);
    }
    Ok(format!("sha256:{:x}", digest.finalize()))
}

fn iso8601_now() -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    chrono::DateTime::<chrono::Utc>::from_timestamp(now as i64, 0)
        .unwrap_or_else(chrono::Utc::now)
        .to_rfc3339()
}

fn model_display_name(model_path: &Path) -> String {
    if let Some(value) = model_path.file_stem().and_then(|value| value.to_str()) {
        value.to_string()
    } else {
        model_path.to_string_lossy().to_string()
    }
}

fn bundled_min_experts(model_name: &str) -> Option<u32> {
    let q = model_name.to_lowercase();
    catalog::MODEL_CATALOG
        .iter()
        .find(|model| model.name.to_lowercase() == q || model.file.to_lowercase().contains(&q))
        .and_then(|model| model.moe.as_ref().map(|cfg| cfg.min_experts_per_node))
}

fn load_analyze_mass_profile(path: &Path) -> Result<AnalyzeMassProfile> {
    let content =
        fs::read_to_string(path).with_context(|| format!("Read ranking {}", path.display()))?;
    let mut ranking = Vec::new();
    let mut masses = Vec::new();
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') || trimmed.starts_with("expert") {
            continue;
        }
        let parts = trimmed.split(',').map(str::trim).collect::<Vec<_>>();
        if parts.len() < 2 {
            continue;
        }
        let expert_id: u32 = parts[0]
            .parse()
            .with_context(|| format!("Parse expert id from {}", path.display()))?;
        let gate_mass: f64 = parts[1]
            .parse()
            .with_context(|| format!("Parse gate mass from {}", path.display()))?;
        ranking.push(expert_id);
        masses.push((expert_id, gate_mass));
    }
    if masses.is_empty() {
        bail!("ranking does not include gate-mass rows");
    }
    let total_mass = masses.iter().map(|(_, mass)| *mass).sum::<f64>();
    Ok(AnalyzeMassProfile {
        ranking,
        masses,
        total_mass,
    })
}

fn mass_pct_for_experts(profile: &AnalyzeMassProfile, experts: &[u32]) -> f64 {
    if profile.total_mass <= f64::EPSILON {
        return 0.0;
    }
    let mut total = 0.0;
    for expert in experts {
        if let Some((_, mass)) = profile
            .masses
            .iter()
            .find(|(candidate, _)| candidate == expert)
        {
            total += *mass;
        }
    }
    (total / profile.total_mass) * 100.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_case_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "mesh-llm-moe-planner-{name}-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn fake_ranking(path: &str, analyzer_id: &str, source: RankingSource) -> ResolvedRanking {
        ResolvedRanking {
            path: PathBuf::from(path),
            metadata_path: None,
            analysis_path: None,
            analyzer_id: analyzer_id.to_string(),
            source,
            reason: "test fixture".to_string(),
        }
    }

    #[test]
    fn normalize_distribution_id_strips_split_suffix() {
        assert_eq!(
            normalize_distribution_id("GLM-5.1-UD-IQ2_M-00001-of-00006.gguf"),
            "GLM-5.1-UD-IQ2_M"
        );
    }

    #[test]
    fn normalize_distribution_id_keeps_unsplit_name() {
        assert_eq!(
            normalize_distribution_id("gemma-4-26B-A4B-it-UD-Q4_K_S.gguf"),
            "gemma-4-26B-A4B-it-UD-Q4_K_S"
        );
    }

    #[test]
    fn infer_analyzer_from_ranking_path_supports_micro_and_full() {
        assert_eq!(
            infer_analyzer_from_ranking_path(Path::new("/tmp/a/micro-v1/ranking.csv")),
            Some("micro-v1")
        );
        assert_eq!(
            infer_analyzer_from_ranking_path(Path::new("/tmp/a/full-v1/ranking.csv")),
            Some("full-v1")
        );
        assert_eq!(
            infer_analyzer_from_ranking_path(Path::new("/tmp/local-demo.micro-p8-t128-all.csv")),
            Some("micro-v1")
        );
        assert_eq!(
            infer_analyzer_from_ranking_path(Path::new("/tmp/local-demo.csv")),
            Some("full-v1")
        );
    }

    #[test]
    fn local_submit_ranking_override_infers_analyzer_and_metadata() {
        let dir = temp_case_dir("submit-override");
        let analyzer_dir = dir.join("micro-v1");
        let ranking_path = analyzer_dir.join("ranking.csv");
        let metadata_path = analyzer_dir.join("metadata.json");
        let analysis_path = analyzer_dir.join("analysis.json");
        fs::create_dir_all(&analyzer_dir).unwrap();
        fs::write(&ranking_path, "0\n1\n").unwrap();
        fs::write(&metadata_path, "{}\n").unwrap();
        fs::write(&analysis_path, "{}\n").unwrap();

        let model = MoeModelContext {
            input: "demo".to_string(),
            path: PathBuf::from("/tmp/demo.gguf"),
            display_name: "demo.gguf".to_string(),
            source_repo: Some("unsloth/demo".to_string()),
            source_revision: Some("abcdef123456".to_string()),
            distribution_id: "demo".to_string(),
            expert_count: 8,
            used_expert_count: 2,
            min_experts_per_node: 4,
            total_model_bytes: 1024,
        };

        let resolved = local_submit_ranking(&model, Some(&ranking_path)).unwrap();
        assert_eq!(resolved.analyzer_id, "micro-v1");
        assert_eq!(
            resolved.metadata_path.as_deref(),
            Some(metadata_path.as_path())
        );
        assert_eq!(
            resolved.analysis_path.as_deref(),
            Some(analysis_path.as_path())
        );
        let _ = fs::remove_dir_all(&dir);
    }

    fn write_test_ranking(path: &Path, rows: &[(u32, f64)]) {
        let mut content = String::from("expert_id,total_mass,mass_fraction,selection_count\n");
        for (expert_id, mass) in rows {
            content.push_str(&format!("{expert_id},{mass},0.0,1\n"));
        }
        fs::write(path, content).unwrap();
    }

    fn test_analysis(full_model_bytes: u64, base_resident_bytes: u64) -> MoeAnalysisJson {
        MoeAnalysisJson {
            schema_version: 1,
            ranking: MoeAnalysisRanking {
                sha256: "sha256:test".to_string(),
                rows: 4,
            },
            model: MoeAnalysisModel {
                expert_count: 4,
                expert_used_count: 2,
            },
            memory: MoeAnalysisMemory {
                full_model_bytes,
                base_resident_bytes,
                shard_file_overhead_bytes: 100_000_000,
                expert_tensor_bytes_total: 4_000_000_000,
                expert_bytes: MoeAnalysisExpertBytes::Uniform {
                    bytes_per_expert: 1_000_000_000,
                },
            },
        }
    }

    #[test]
    fn estimate_startup_fit_reports_full_model_fit() {
        let dir = temp_case_dir("startup-fit-full");
        let ranking_path = dir.join("ranking.csv");
        write_test_ranking(&ranking_path, &[(0, 4.0), (1, 3.0), (2, 2.0), (3, 1.0)]);

        let estimate = estimate_startup_fit_from_analysis(
            &ranking_path,
            &test_analysis(8_000_000_000, 2_000_000_000),
            9_000_000_000,
            "full-v1",
            "Hugging Face dataset",
        )
        .unwrap();

        assert!(estimate.full_model_fits());
        assert_eq!(estimate.recommended_nodes, Some(1));
        assert_eq!(estimate.predicted_max_shard_bytes, None);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn estimate_startup_fit_reports_split_fit_when_full_model_is_too_large() {
        let dir = temp_case_dir("startup-fit-split");
        let ranking_path = dir.join("ranking.csv");
        write_test_ranking(&ranking_path, &[(0, 4.0), (1, 3.0), (2, 2.0), (3, 1.0)]);

        let estimate = estimate_startup_fit_from_analysis(
            &ranking_path,
            &test_analysis(10_000_000_000, 500_000_000),
            4_000_000_000,
            "full-v1",
            "Hugging Face dataset",
        )
        .unwrap();

        assert!(!estimate.full_model_fits());
        assert!(estimate.shard_plan_fits());
        assert_eq!(estimate.recommended_nodes, Some(2));
        assert_eq!(estimate.predicted_max_shard_bytes, Some(3_600_000_000));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn estimate_startup_fit_reports_no_viable_fit() {
        let dir = temp_case_dir("startup-fit-none");
        let ranking_path = dir.join("ranking.csv");
        write_test_ranking(&ranking_path, &[(0, 4.0), (1, 3.0), (2, 2.0), (3, 1.0)]);

        let estimate = estimate_startup_fit_from_analysis(
            &ranking_path,
            &test_analysis(10_000_000_000, 500_000_000),
            3_800_000_000,
            "full-v1",
            "Hugging Face dataset",
        )
        .unwrap();

        assert!(!estimate.any_fit_exists());
        assert_eq!(estimate.recommended_nodes, None);
        assert_eq!(
            estimate.predicted_max_shard_launch_bytes,
            Some(3_960_000_000)
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn select_preferred_ranking_prefers_full_over_micro() {
        let local = fake_ranking("/tmp/local.csv", "micro-v1", RankingSource::LocalCache);
        let remote = fake_ranking(
            "/tmp/remote.csv",
            "full-v1",
            RankingSource::HuggingFaceDataset,
        );

        let selected = select_preferred_ranking(Some(local), Some(remote)).unwrap();
        assert_eq!(selected.analyzer_id, "full-v1");
        assert!(matches!(selected.source, RankingSource::HuggingFaceDataset));
    }

    #[test]
    fn select_preferred_ranking_prefers_local_when_methods_match() {
        let local = fake_ranking("/tmp/local.csv", "micro-v1", RankingSource::LocalCache);
        let remote = fake_ranking(
            "/tmp/remote.csv",
            "micro-v1",
            RankingSource::HuggingFaceDataset,
        );

        let selected = select_preferred_ranking(Some(local), Some(remote)).unwrap();
        assert!(matches!(selected.source, RankingSource::LocalCache));
    }

    #[test]
    fn build_submit_bundle_uses_canonical_dataset_layout() {
        let dir = temp_case_dir("submit-bundle");
        let analyzer_dir = dir.join("full-v1");
        let ranking_path = analyzer_dir.join("ranking.csv");
        let analysis_path = analyzer_dir.join("analysis.json");
        let log_path = analyzer_dir.join("run.log");
        fs::create_dir_all(&analyzer_dir).unwrap();
        fs::write(&ranking_path, "0\n1\n").unwrap();
        fs::write(&analysis_path, "{}\n").unwrap();
        fs::write(&log_path, "ok\n").unwrap();

        let model_file = dir.join("gemma-4-26B-A4B-it-UD-Q4_K_S.gguf");
        fs::write(&model_file, b"fake").unwrap();
        let model = MoeModelContext {
            input: "unsloth/gemma".to_string(),
            path: model_file,
            display_name: "gemma-4-26B-A4B-it-UD-Q4_K_S.gguf".to_string(),
            source_repo: Some("unsloth/gemma-4-26B-A4B-it-GGUF".to_string()),
            source_revision: Some("9c718328e1620e7280a93e1a809e805e0f3e4839".to_string()),
            distribution_id: "gemma-4-26B-A4B-it-UD-Q4_K_S".to_string(),
            expert_count: 64,
            used_expert_count: 4,
            min_experts_per_node: 32,
            total_model_bytes: 123,
        };
        let ranking = ResolvedRanking {
            path: ranking_path,
            metadata_path: None,
            analysis_path: Some(analysis_path),
            analyzer_id: "full-v1".to_string(),
            source: RankingSource::LocalCache,
            reason: "test".to_string(),
        };

        let bundle = build_submit_bundle(&model, &ranking, Some(&log_path)).unwrap();
        assert_eq!(
            bundle.dataset_prefix,
            "data/unsloth/gemma-4-26B-A4B-it-GGUF/9c718328e1620e7280a93e1a809e805e0f3e4839/gguf/gemma-4-26B-A4B-it-UD-Q4_K_S/full-v1"
        );
        assert!(bundle
            .dataset_paths
            .contains(&format!("{}/ranking.csv", bundle.dataset_prefix)));
        assert!(bundle
            .dataset_paths
            .contains(&format!("{}/metadata.json", bundle.dataset_prefix)));
        assert!(bundle
            .dataset_paths
            .contains(&format!("{}/analysis.json", bundle.dataset_prefix)));
        assert!(bundle
            .dataset_paths
            .contains(&format!("{}/run.log", bundle.dataset_prefix)));
        let _ = fs::remove_dir_all(&dir);
    }
}
