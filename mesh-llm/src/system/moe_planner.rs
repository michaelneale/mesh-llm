use anyhow::{bail, Context, Result};
use hf_hub::{Repo, RepoType};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::inference::{election, moe};
use crate::models::{
    build_hf_api, catalog, huggingface_identity_for_path, resolve_model_spec,
    resolve_model_spec_with_progress,
};

const DEFAULT_DATASET_REVISION: &str = "main";
pub(crate) const DEFAULT_MOE_RANKINGS_DATASET: &str = "meshllm/moe-rankings";

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
    pub log_path: Option<PathBuf>,
    pub commit_message: String,
    pub commit_description: String,
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

pub(crate) async fn plan_moe(args: MoePlanArgs) -> Result<MoePlanReport> {
    if let Some(nodes) = args.nodes {
        if nodes == 0 {
            bail!("--nodes must be at least 1");
        }
    }
    let model = resolve_model_context_with_progress(&args.model, args.progress).await?;
    let ranking = resolve_best_ranking(&model, &args).await?;
    let target_vram_bytes = resolve_target_vram_bytes(args.max_vram_gb);
    if target_vram_bytes == 0 {
        bail!(
            "No VRAM target available. Pass --max-vram <GB> or run on a machine with detectable GPU memory."
        );
    }

    let recommended_nodes = args.nodes.unwrap_or_else(|| {
        ((model.total_model_bytes as f64) / target_vram_bytes as f64)
            .ceil()
            .max(1.0) as usize
    });
    let max_supported_nodes = (model.expert_count / model.min_experts_per_node.max(1)) as usize;
    let max_supported_nodes = max_supported_nodes.max(1);
    let feasible = recommended_nodes <= max_supported_nodes;

    let ranking_vec = moe::load_cached_ranking(&ranking.path)
        .ok_or_else(|| anyhow::anyhow!("cached ranking not found: {}", ranking.path.display()))
        .with_context(|| format!("Load ranking {}", ranking.path.display()))?;
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
            &profile,
            &profile.ranking[..shared.min(profile.ranking.len())],
        );
        let node_mass: Vec<f64> = assignments
            .iter()
            .map(|assignment| mass_pct_for_experts(&profile, &assignment.experts))
            .collect();
        (
            Some(shared_mass_pct),
            node_mass.iter().copied().reduce(f64::max),
            node_mass.iter().copied().reduce(f64::min),
        )
    } else {
        (None, None, None)
    };

    let mut assumptions = vec![
        format!(
            "Minimum nodes estimated from total model bytes / target VRAM: {:.1}GB / {:.1}GB",
            model.total_model_bytes as f64 / 1e9,
            target_vram_bytes as f64 / 1e9
        ),
        format!(
            "Minimum experts per node uses catalog or 50% fallback: {}",
            model.min_experts_per_node
        ),
    ];
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
        return Ok(ResolvedRanking {
            path: path.to_path_buf(),
            metadata_path,
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
        Err(error) => return local_legacy.ok_or_else(|| error).map(Some),
    };
    Ok(select_preferred_ranking(local_legacy, remote))
}

fn resolve_local_runtime_ranking(model_path: &Path) -> Option<ResolvedRanking> {
    moe::best_shared_ranking_artifact(model_path).map(|artifact| {
        let analyzer_id = match artifact.kind {
            moe::SharedRankingKind::Analyze => "full-v1",
            moe::SharedRankingKind::MicroAnalyze => "micro-v1",
        };
        ResolvedRanking {
            path: moe::shared_ranking_cache_path(model_path, &artifact),
            metadata_path: None,
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
        let analyzer_id = infer_analyzer_from_ranking_path(path)
            .unwrap_or("override")
            .to_string();
        return Ok(ResolvedRanking {
            path: path.to_path_buf(),
            metadata_path,
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
    let remote_lookup = tokio::task::spawn_blocking(move || {
        fetch_remote_ranking(
            &remote_dataset_repo,
            &remote_source_repo,
            &remote_source_revision,
            &remote_distribution_id,
            remote_progress,
        )
    })
    .await
    .context("Join blocking Hugging Face MoE ranking lookup task")?;

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
    let dataset_repo = api.repo(Repo::with_revision(
        dataset_repo_name.to_string(),
        RepoType::Dataset,
        DEFAULT_DATASET_REVISION.to_string(),
    ));
    let info = dataset_repo.info()?;
    find_remote_ranking(
        &api,
        dataset_repo_name,
        &info.sha,
        &info.siblings,
        source_repo,
        source_revision,
        distribution_id,
    )
}

fn find_remote_ranking(
    api: &hf_hub::api::sync::Api,
    dataset_repo: &str,
    dataset_revision: &str,
    siblings: &[hf_hub::api::Siblings],
    source_repo: &str,
    source_revision: &str,
    distribution_id: &str,
) -> Result<Option<ResolvedRanking>> {
    let prefix = canonical_dataset_prefix(source_repo, source_revision, distribution_id);
    for analyzer_id in ["full-v1", "micro-v1"] {
        let metadata_rel = format!("{prefix}/{analyzer_id}/metadata.json");
        let ranking_rel = format!("{prefix}/{analyzer_id}/ranking.csv");
        let has_metadata = siblings.iter().any(|entry| entry.rfilename == metadata_rel);
        let has_ranking = siblings.iter().any(|entry| entry.rfilename == ranking_rel);
        if !(has_metadata && has_ranking) {
            continue;
        }

        let pinned = api.repo(Repo::with_revision(
            dataset_repo.to_string(),
            RepoType::Dataset,
            dataset_revision.to_string(),
        ));
        let metadata_path = pinned
            .download(&metadata_rel)
            .with_context(|| format!("Download {}", metadata_rel))?;
        let ranking_path = pinned
            .download(&ranking_rel)
            .with_context(|| format!("Download {}", ranking_rel))?;
        return Ok(Some(ResolvedRanking {
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
    let log_rel = format!("{dataset_prefix}/run.log");

    let model_files = discover_distribution_files(model)?;
    let metadata_content = if let Some(existing) = ranking.metadata_path.as_ref() {
        fs::read_to_string(existing)
            .with_context(|| format!("Read existing metadata {}", existing.display()))?
    } else {
        serde_json::to_string_pretty(&build_metadata_json(model, ranking, &model_files)?)? + "\n"
    };

    let mut dataset_paths = vec![ranking_rel, metadata_rel];
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

fn infer_analysis_details(
    ranking: &ResolvedRanking,
    model: &MoeModelContext,
) -> Result<(Option<&'static str>, Option<usize>, u32, u32, bool)> {
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
        fs::create_dir_all(&analyzer_dir).unwrap();
        fs::write(&ranking_path, "0\n1\n").unwrap();
        fs::write(&metadata_path, "{}\n").unwrap();

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
        let log_path = analyzer_dir.join("run.log");
        fs::create_dir_all(&analyzer_dir).unwrap();
        fs::write(&ranking_path, "0\n1\n").unwrap();
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
            .contains(&format!("{}/run.log", bundle.dataset_prefix)));
        let _ = fs::remove_dir_all(&dir);
    }
}
