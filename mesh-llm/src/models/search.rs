use super::resolve::{
    file_preference_score, is_known_gguf_sidecar, matching_catalog_model_for_huggingface,
    merge_capabilities, quant_selector_from_gguf_file, remote_hf_size_label_with_api,
};
use super::ModelCapabilities;
use super::{build_hf_tokio_api, capabilities, catalog};
use crate::system::hardware;
use anyhow::{Context, Result};
use hf_hub::{ListModelsParams, ModelInfo};
use regex_lite::Regex;
use serde_json::{json, Value};
use std::collections::HashSet;
use std::sync::LazyLock;
use tokio::task::JoinSet;
use tokio_stream::StreamExt;

#[derive(Clone, Debug)]
pub struct SearchHit {
    pub repo_id: String,
    pub kind: &'static str,
    pub exact_ref: String,
    pub variant_count: Option<usize>,
    pub size_label: Option<String>,
    pub downloads: Option<u64>,
    pub likes: Option<u64>,
    pub catalog: Option<&'static catalog::CatalogModel>,
    pub capabilities: ModelCapabilities,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SearchProgress {
    SearchingHub,
    InspectingRepos { completed: usize, total: usize },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SearchArtifactFilter {
    Gguf,
    Mlx,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SearchSort {
    Trending,
    Downloads,
    Likes,
    Created,
    Updated,
    ParametersDesc,
    ParametersAsc,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RepoArtifactKind {
    Gguf,
    Mlx,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct RepoArtifactCandidate {
    kind: RepoArtifactKind,
    file: String,
}

pub fn search_catalog_models(query: &str) -> Vec<&'static catalog::CatalogModel> {
    let q = query.to_lowercase();
    let mut results: Vec<_> = catalog::MODEL_CATALOG
        .iter()
        .filter(|model| {
            model.name.to_lowercase().contains(&q)
                || model.file.to_lowercase().contains(&q)
                || model.description.to_lowercase().contains(&q)
        })
        .collect();
    results.sort_by(|left, right| left.name.cmp(&right.name));
    results
}

pub fn search_catalog_json_payload(
    query: &str,
    filter: SearchArtifactFilter,
    sort: SearchSort,
    results: &[&'static catalog::CatalogModel],
    limit: usize,
) -> Value {
    let payload_results: Vec<Value> = results
        .iter()
        .take(limit)
        .map(|model| {
            json!({
                "name": model.name,
                "repo_id": model.source_repo(),
                "type": catalog_model_kind_code(model),
                "size": model.size,
                "description": model.description,
                "fit": fit_code_for_size_label(&model.size),
                "ref": model.name,
                "show": format!("mesh-llm models show {}", model.name),
                "download": format!("mesh-llm models download {}", model.name),
                "draft": model.draft,
                "capabilities": capabilities_json(capabilities::infer_catalog_capabilities(model)),
            })
        })
        .collect();
    json!({
        "query": query,
        "filter": search_filter_name(filter),
        "sort": search_sort_name(sort),
        "source": "catalog",
        "machine": local_capacity_json(),
        "results": payload_results,
    })
}

pub fn search_huggingface_json_payload(
    query: &str,
    filter: SearchArtifactFilter,
    sort: SearchSort,
    results: &[SearchHit],
) -> Value {
    let payload_results: Vec<Value> = results
        .iter()
        .map(|result| {
            json!({
                "repo_id": result.repo_id,
                "repo_url": huggingface_repo_url(&result.repo_id),
                "type": model_kind_code(result.kind),
                "variant_count": result.variant_count,
                "size": result.size_label,
                "downloads": result.downloads,
                "likes": result.likes,
                "fit": result
                    .size_label
                    .as_deref()
                    .and_then(fit_code_for_size_label),
                "ref": result.exact_ref,
                "show": format!("mesh-llm models show {}", result.exact_ref),
                "download": format!("mesh-llm models download {}", result.exact_ref),
                "capabilities": capabilities_json(result.capabilities),
                "catalog": result.catalog.map(|model| {
                    json!({
                        "name": model.name,
                        "size": model.size,
                        "description": model.description,
                    })
                }),
            })
        })
        .collect();
    json!({
        "query": query,
        "filter": search_filter_name(filter),
        "sort": search_sort_name(sort),
        "source": "huggingface",
        "machine": local_capacity_json(),
        "results": payload_results,
    })
}

// Keep search custom for now. `hf-hub` handles cache and file transport well,
// but it does not expose a Hub search surface in this crate version.
pub async fn search_huggingface<F>(
    query: &str,
    limit: usize,
    filter: SearchArtifactFilter,
    sort: SearchSort,
    mut progress: F,
) -> Result<Vec<SearchHit>>
where
    F: FnMut(SearchProgress),
{
    const SEARCH_CONCURRENCY: usize = 10;

    let repo_limit = match sort {
        SearchSort::ParametersDesc | SearchSort::ParametersAsc => {
            (limit.saturating_mul(5)).clamp(1, 100)
        }
        _ => limit.clamp(1, 100),
    };
    progress(SearchProgress::SearchingHub);
    let api = build_hf_tokio_api(false)?;
    let base_builder = ListModelsParams::builder()
        .search(query.to_string())
        .filter(
            match filter {
                SearchArtifactFilter::Gguf => "gguf",
                SearchArtifactFilter::Mlx => "mlx",
            }
            .to_string(),
        )
        .full(true)
        .limit(repo_limit);
    let params = match api_sort_key(sort) {
        Some(api_sort) => base_builder.sort(api_sort.to_string()).build(),
        None => base_builder.build(),
    };
    let stream = api.list_models(&params).context("Search Hugging Face")?;
    tokio::pin!(stream);
    let mut repos = Vec::new();
    while let Some(repo) = stream.next().await {
        repos.push(repo.context("Search Hugging Face repo summary")?);
    }

    let total = repos.len();
    progress(SearchProgress::InspectingRepos {
        completed: 0,
        total,
    });

    let mut pending = repos.into_iter().enumerate();
    let mut join_set = JoinSet::new();
    for _ in 0..SEARCH_CONCURRENCY.min(total.max(1)) {
        if let Some((index, repo)) = pending.next() {
            let api = api.clone();
            join_set.spawn(async move { (index, build_search_hit(api, repo, filter).await) });
        }
    }

    let mut completed = 0usize;
    let mut indexed_hits = Vec::new();
    while let Some(joined) = join_set.join_next().await {
        let (index, result) = joined.context("Join Hugging Face repo inspection task")?;
        completed += 1;
        progress(SearchProgress::InspectingRepos { completed, total });
        match result {
            Ok(Some(hit)) => {
                indexed_hits.push((index, hit));
            }
            Ok(None) => {}
            Err(err) => {
                eprintln!("⚠️  Failed to inspect Hugging Face repo: {err:#}");
            }
        }
        if let Some((next_index, repo)) = pending.next() {
            let api = api.clone();
            join_set.spawn(async move { (next_index, build_search_hit(api, repo, filter).await) });
        }
    }

    indexed_hits.sort_by_key(|(index, _)| *index);
    let mut hits: Vec<SearchHit> = indexed_hits.into_iter().map(|(_, hit)| hit).collect();
    apply_local_search_sort(&mut hits, sort);
    hits.truncate(limit);
    Ok(hits)
}

async fn build_search_hit(
    api: hf_hub::HFClient,
    repo: ModelInfo,
    filter: SearchArtifactFilter,
) -> Result<Option<SearchHit>> {
    let repo_downloads = repo.downloads;
    let repo_likes = repo.likes;
    let detail = repo;
    let repo_id = detail.model_id.clone().unwrap_or(detail.id.clone());
    let siblings = detail.siblings.clone().unwrap_or_default();
    let sibling_names: Vec<String> = siblings
        .iter()
        .map(|sibling| sibling.rfilename.clone())
        .collect();
    let sibling_size_entries: Vec<(String, Option<u64>)> = siblings
        .iter()
        .map(|sibling| (sibling.rfilename.clone(), sibling.size))
        .collect();
    let repo_has_mlx_weights = sibling_names.iter().any(|file| is_mlx_weight_file(file));
    let candidates = collect_repo_artifact_candidates(&sibling_names);
    if candidates.is_empty() {
        return Ok(None);
    }

    let matching_candidates: Vec<_> = candidates
        .into_iter()
        .filter(|candidate| match filter {
            SearchArtifactFilter::Gguf => candidate.kind == RepoArtifactKind::Gguf,
            SearchArtifactFilter::Mlx => {
                candidate.kind == RepoArtifactKind::Mlx && repo_has_mlx_weights
            }
        })
        .collect();
    if matching_candidates.is_empty() {
        return Ok(None);
    }

    let candidate = &matching_candidates[0];
    let variant_count = search_hit_variant_count(filter, &repo_id, &matching_candidates);
    let remote_metadata = capabilities::fetch_remote_hf_metadata_jsons(&repo_id, None).await;
    let catalog = matching_catalog_model_for_huggingface(&repo_id, None, &candidate.file);
    let size_label = match catalog {
        Some(model) => Some(model.size.to_string()),
        None => match size_label_from_sibling_entries(&candidate.file, &sibling_size_entries) {
            Some(size_label) => Some(size_label),
            None => remote_hf_size_label_with_api(&api, &repo_id, None, &candidate.file).await,
        },
    };
    let remote_caps = capabilities::infer_remote_hf_capabilities_with_metadata(
        &repo_id,
        &candidate.file,
        Some(&sibling_names),
        &remote_metadata,
    );
    let capabilities = match catalog {
        Some(model) => {
            let base = capabilities::infer_catalog_capabilities(model);
            merge_capabilities(base, remote_caps)
        }
        None => remote_caps,
    };
    Ok(Some(SearchHit {
        repo_id: repo_id.clone(),
        kind: repo_artifact_kind_label(candidate.kind),
        exact_ref: display_exact_ref(&repo_id, candidate.kind, &candidate.file),
        variant_count,
        size_label,
        downloads: detail.downloads.or(repo_downloads),
        likes: detail.likes.or(repo_likes),
        catalog,
        capabilities,
    }))
}

fn api_sort_key(sort: SearchSort) -> Option<&'static str> {
    match sort {
        SearchSort::Trending => Some("trendingScore"),
        SearchSort::Downloads => Some("downloads"),
        SearchSort::Likes => Some("likes"),
        SearchSort::Created => Some("createdAt"),
        SearchSort::Updated => Some("lastModified"),
        SearchSort::ParametersDesc | SearchSort::ParametersAsc => None,
    }
}

fn search_filter_name(filter: SearchArtifactFilter) -> &'static str {
    match filter {
        SearchArtifactFilter::Gguf => "gguf",
        SearchArtifactFilter::Mlx => "mlx",
    }
}

fn search_sort_name(sort: SearchSort) -> &'static str {
    match sort {
        SearchSort::Trending => "trending",
        SearchSort::Downloads => "downloads",
        SearchSort::Likes => "likes",
        SearchSort::Created => "created",
        SearchSort::Updated => "updated",
        SearchSort::ParametersDesc => "parameters-desc",
        SearchSort::ParametersAsc => "parameters-asc",
    }
}

fn local_capacity_json() -> Value {
    let vram_bytes = hardware::survey().vram_bytes;
    let vram_gb = vram_bytes as f64 / 1e9;
    json!({
        "vram_bytes": vram_bytes,
        "vram_gb": vram_gb,
    })
}

fn capabilities_json(caps: ModelCapabilities) -> Value {
    json!({
        "text": true,
        "multimodal": caps.multimodal_status(),
        "vision": caps.vision_status(),
        "audio": caps.audio_status(),
        "reasoning": caps.reasoning_status(),
        "tool_use": caps.tool_use_status(),
        "moe": caps.moe,
    })
}

fn fit_code_for_size_label(size_label: &str) -> Option<&'static str> {
    let model_gb = catalog::parse_size_gb(size_label);
    let vram_gb = hardware::survey().vram_bytes as f64 / 1e9;
    if model_gb <= 0.0 || vram_gb <= 0.0 {
        return None;
    }

    let code = if model_gb <= vram_gb * 0.6 {
        "comfortable"
    } else if model_gb <= vram_gb * 0.9 {
        "tight"
    } else if model_gb <= vram_gb * 1.1 {
        "tradeoff"
    } else {
        "too_large"
    };
    Some(code)
}

fn huggingface_repo_url(repo_id: &str) -> String {
    format!("https://huggingface.co/{repo_id}")
}

fn model_kind_code(kind: &str) -> &'static str {
    if kind.to_ascii_lowercase().contains("mlx") {
        "mlx"
    } else {
        "gguf"
    }
}

fn catalog_model_kind_code(model: &catalog::CatalogModel) -> &'static str {
    if model
        .source_file()
        .map(|file| {
            file.ends_with("model.safetensors") || file.ends_with("model.safetensors.index.json")
        })
        .unwrap_or(false)
        || model.url.contains("model.safetensors")
    {
        "mlx"
    } else {
        "gguf"
    }
}

fn apply_local_search_sort(hits: &mut [SearchHit], sort: SearchSort) {
    match sort {
        SearchSort::ParametersDesc => {
            hits.sort_by(|left, right| {
                approx_parameter_count_b(right)
                    .partial_cmp(&approx_parameter_count_b(left))
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| left.repo_id.cmp(&right.repo_id))
            });
        }
        SearchSort::ParametersAsc => {
            hits.sort_by(|left, right| {
                approx_parameter_count_b(left)
                    .partial_cmp(&approx_parameter_count_b(right))
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| left.repo_id.cmp(&right.repo_id))
            });
        }
        _ => {}
    }
}

fn approx_parameter_count_b(hit: &SearchHit) -> f64 {
    approximate_parameter_count_b_from_text(&format!("{} {}", hit.repo_id, hit.exact_ref))
        .unwrap_or(-1.0)
}

fn approximate_parameter_count_b_from_text(text: &str) -> Option<f64> {
    static MULTIPLIED_RE: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"(?i)(\d+(?:\.\d+)?)x(\d+(?:\.\d+)?)([bm])").unwrap());
    static SIMPLE_RE: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"(?i)(\d+(?:\.\d+)?)([bm])").unwrap());

    let mut best: Option<f64> = None;
    for captures in MULTIPLIED_RE.captures_iter(text) {
        let Some(left) = captures.get(1).and_then(|m| m.as_str().parse::<f64>().ok()) else {
            continue;
        };
        let Some(right) = captures.get(2).and_then(|m| m.as_str().parse::<f64>().ok()) else {
            continue;
        };
        let Some(unit) = captures.get(3).map(|m| m.as_str().to_ascii_lowercase()) else {
            continue;
        };
        let value = match unit.as_str() {
            "b" => left * right,
            "m" => (left * right) / 1000.0,
            _ => continue,
        };
        best = Some(best.map_or(value, |current| current.max(value)));
    }
    for captures in SIMPLE_RE.captures_iter(text) {
        let Some(number) = captures.get(1).and_then(|m| m.as_str().parse::<f64>().ok()) else {
            continue;
        };
        let Some(unit) = captures.get(2).map(|m| m.as_str().to_ascii_lowercase()) else {
            continue;
        };
        let value = match unit.as_str() {
            "b" => number,
            "m" => number / 1000.0,
            _ => continue,
        };
        best = Some(best.map_or(value, |current| current.max(value)));
    }
    best
}

fn repo_artifact_kind_label(kind: RepoArtifactKind) -> &'static str {
    match kind {
        RepoArtifactKind::Gguf => "🦙 GGUF",
        RepoArtifactKind::Mlx => "🍎 MLX",
    }
}

fn display_exact_ref(repo: &str, kind: RepoArtifactKind, file: &str) -> String {
    match kind {
        RepoArtifactKind::Gguf => match quant_selector_from_gguf_file(file) {
            Some(selector) => format!("{repo}:{selector}"),
            None => format!("{repo}/{}", display_ref_file(file)),
        },
        RepoArtifactKind::Mlx => repo.to_string(),
    }
}

fn display_ref_file(file: &str) -> String {
    if let Some(without_ext) = file.strip_suffix(".gguf") {
        if !without_ext.contains("-00001-of-") {
            return without_ext.to_string();
        }
        let Some((prefix, suffix)) = without_ext.rsplit_once("-00001-of-") else {
            return without_ext.to_string();
        };
        if suffix.len() == 5 && suffix.chars().all(|ch| ch.is_ascii_digit()) {
            return prefix.to_string();
        }
        return without_ext.to_string();
    }

    if file == "model.safetensors" {
        return "model".to_string();
    }
    if is_split_mlx_first_shard(file) {
        return "model".to_string();
    }
    file.to_string()
}

fn size_label_from_sibling_entries(
    file: &str,
    siblings: &[(String, Option<u64>)],
) -> Option<String> {
    siblings
        .iter()
        .find_map(|(name, size)| (name == file).then_some(*size).flatten())
        .map(super::format_size_bytes)
}

fn collect_repo_artifact_candidates(siblings: &[String]) -> Vec<RepoArtifactCandidate> {
    let mut gguf = Vec::new();
    let mut mlx = Vec::new();
    for sibling in siblings {
        if sibling.ends_with(".gguf") {
            if is_known_gguf_sidecar(sibling) {
                continue;
            }
            if sibling.contains("-000") && !sibling.contains("-00001-of-") {
                continue;
            }
            gguf.push(RepoArtifactCandidate {
                kind: RepoArtifactKind::Gguf,
                file: sibling.clone(),
            });
            continue;
        }
        if sibling == "model.safetensors.index.json" || sibling == "model.safetensors" {
            if sibling == "model.safetensors.index.json" {
                continue;
            }
            mlx.push(RepoArtifactCandidate {
                kind: RepoArtifactKind::Mlx,
                file: sibling.clone(),
            });
            continue;
        }
        if is_split_mlx_weight_file(sibling) {
            if !is_split_mlx_first_shard(sibling) {
                continue;
            }
            mlx.push(RepoArtifactCandidate {
                kind: RepoArtifactKind::Mlx,
                file: sibling.clone(),
            });
        }
    }
    gguf.sort_by(|left, right| {
        file_preference_score(&left.file)
            .cmp(&file_preference_score(&right.file))
            .then_with(|| left.file.cmp(&right.file))
    });
    mlx.sort_by(|left, right| {
        mlx_candidate_rank(&left.file)
            .cmp(&mlx_candidate_rank(&right.file))
            .then_with(|| left.file.cmp(&right.file))
    });
    if !mlx.is_empty() {
        let best_rank = mlx_candidate_rank(&mlx[0].file);
        mlx.retain(|candidate| mlx_candidate_rank(&candidate.file) == best_rank);
    }
    gguf.extend(mlx);
    gguf
}

fn gguf_variant_count_from_candidates(repo: &str, candidates: &[RepoArtifactCandidate]) -> usize {
    candidates
        .iter()
        .filter(|candidate| candidate.kind == RepoArtifactKind::Gguf)
        .filter_map(|candidate| {
            quant_selector_from_gguf_file(&candidate.file)
                .map(|_| display_exact_ref(repo, RepoArtifactKind::Gguf, &candidate.file))
        })
        .collect::<HashSet<_>>()
        .len()
}

fn search_hit_variant_count(
    filter: SearchArtifactFilter,
    repo: &str,
    candidates: &[RepoArtifactCandidate],
) -> Option<usize> {
    match filter {
        SearchArtifactFilter::Gguf => Some(gguf_variant_count_from_candidates(repo, candidates)),
        SearchArtifactFilter::Mlx => None,
    }
}

fn is_split_mlx_weight_file(file: &str) -> bool {
    let Some(rest) = file.strip_prefix("model-") else {
        return false;
    };
    let Some(rest) = rest.strip_suffix(".safetensors") else {
        return false;
    };
    let Some((left, right)) = rest.split_once("-of-") else {
        return false;
    };
    left.len() == 5
        && right.len() == 5
        && left.bytes().all(|b| b.is_ascii_digit())
        && right.bytes().all(|b| b.is_ascii_digit())
}

fn is_split_mlx_first_shard(file: &str) -> bool {
    is_split_mlx_weight_file(file) && file.starts_with("model-00001-of-")
}

fn is_mlx_weight_file(file: &str) -> bool {
    file == "model.safetensors" || is_split_mlx_weight_file(file)
}

fn mlx_candidate_rank(file: &str) -> usize {
    if file == "model.safetensors" {
        0
    } else if is_split_mlx_first_shard(file) {
        1
    } else if file == "model.safetensors.index.json" {
        2
    } else {
        3
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::find_catalog_model_exact;
    use std::sync::{Arc, Mutex};

    fn assert_progress_sequence(events: &[SearchProgress]) {
        assert!(
            events
                .first()
                .is_some_and(|event| matches!(event, SearchProgress::SearchingHub)),
            "expected initial SearchingHub event, got {events:?}"
        );

        let mut last_completed = 0usize;
        let mut last_total = None;
        for event in events {
            if let SearchProgress::InspectingRepos { completed, total } = *event {
                assert!(
                    completed <= total,
                    "completed {completed} exceeded total {total}"
                );
                assert!(
                    completed >= last_completed,
                    "progress regressed from {last_completed} to {completed}"
                );
                if let Some(previous_total) = last_total {
                    assert_eq!(
                        total, previous_total,
                        "repo inspection total changed from {previous_total} to {total}"
                    );
                }
                last_completed = completed;
                last_total = Some(total);
            }
        }

        if let Some(total) = last_total {
            assert_eq!(
                last_completed, total,
                "expected final inspection progress to reach total repos"
            );
        }
    }

    #[test]
    fn collect_repo_artifact_candidates_prefers_model_safetensors_over_index() {
        let siblings = vec![
            "model.safetensors".to_string(),
            "model.safetensors.index.json".to_string(),
        ];
        let candidates = collect_repo_artifact_candidates(&siblings);
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].kind, RepoArtifactKind::Mlx);
        assert_eq!(candidates[0].file, "model.safetensors");
    }

    #[test]
    fn collect_repo_artifact_candidates_keeps_gguf_first_split_only() {
        let siblings = vec![
            "GLM-5.1-UD-Q5_K_XL-00002-of-00013.gguf".to_string(),
            "GLM-5.1-UD-Q5_K_XL-00001-of-00013.gguf".to_string(),
            "GLM-5.1-UD-Q4_K_M.gguf".to_string(),
        ];
        let candidates = collect_repo_artifact_candidates(&siblings);
        let files: Vec<_> = candidates.into_iter().map(|c| c.file).collect();
        assert_eq!(
            files,
            vec![
                "GLM-5.1-UD-Q5_K_XL-00001-of-00013.gguf".to_string(),
                "GLM-5.1-UD-Q4_K_M.gguf".to_string(),
            ]
        );
    }

    #[test]
    fn collect_repo_artifact_candidates_excludes_mmproj_gguf_sidecars() {
        let siblings = vec![
            "mmproj-BF16.gguf".to_string(),
            "vision/mmproj-F16.gguf".to_string(),
            "gemma-4-26B-A4B-it-UD-Q3_K_S.gguf".to_string(),
        ];
        let candidates = collect_repo_artifact_candidates(&siblings);
        let files: Vec<_> = candidates.into_iter().map(|c| c.file).collect();
        assert_eq!(files, vec!["gemma-4-26B-A4B-it-UD-Q3_K_S.gguf".to_string()]);
    }

    #[test]
    fn gguf_variant_count_from_candidates_counts_selectable_variants() {
        let siblings = vec![
            "mmproj-BF16.gguf".to_string(),
            "BF16/Qwen3.6-35B-A3B-BF16-00001-of-00002.gguf".to_string(),
            "BF16/Qwen3.6-35B-A3B-BF16-00002-of-00002.gguf".to_string(),
            "Qwen3.6-35B-A3B-Q8_0.gguf".to_string(),
            "Qwen3.6-35B-A3B-Q4_K_M.gguf".to_string(),
        ];
        let candidates = collect_repo_artifact_candidates(&siblings);
        assert_eq!(
            gguf_variant_count_from_candidates("unsloth/Qwen3.6-35B-A3B-GGUF", &candidates),
            3
        );
    }

    #[test]
    fn search_hit_variant_count_only_applies_to_gguf_results() {
        let gguf_candidates = collect_repo_artifact_candidates(&[
            "BF16/Qwen3.6-35B-A3B-BF16-00001-of-00002.gguf".to_string(),
            "BF16/Qwen3.6-35B-A3B-BF16-00002-of-00002.gguf".to_string(),
            "Qwen3.6-35B-A3B-Q4_K_M.gguf".to_string(),
        ]);
        assert_eq!(
            search_hit_variant_count(
                SearchArtifactFilter::Gguf,
                "unsloth/Qwen3.6-35B-A3B-GGUF",
                &gguf_candidates
            ),
            Some(2)
        );

        let mlx_candidates = collect_repo_artifact_candidates(&[
            "model.safetensors".to_string(),
            "model.safetensors.index.json".to_string(),
        ]);
        assert_eq!(
            search_hit_variant_count(
                SearchArtifactFilter::Mlx,
                "mlx-community/Foo-4bit",
                &mlx_candidates
            ),
            None
        );
    }

    #[test]
    fn size_label_from_sibling_entries_prefers_repo_metadata_size() {
        let siblings = vec![
            ("model-q4.gguf".to_string(), Some(16_900_000_000)),
            ("model-q5.gguf".to_string(), Some(18_800_000_000)),
        ];
        assert_eq!(
            size_label_from_sibling_entries("model-q4.gguf", &siblings).as_deref(),
            Some("16.9GB")
        );
    }

    #[test]
    fn size_label_from_sibling_entries_returns_none_when_missing() {
        let siblings = vec![("model-q4.gguf".to_string(), None)];
        assert_eq!(
            size_label_from_sibling_entries("model-q4.gguf", &siblings),
            None
        );
        assert_eq!(
            size_label_from_sibling_entries("model-q5.gguf", &siblings),
            None
        );
    }

    #[test]
    fn display_ref_file_uses_gguf_and_mlx_stems() {
        assert_eq!(display_ref_file("Qwen3-8B-Q4_K_M.gguf"), "Qwen3-8B-Q4_K_M");
        assert_eq!(
            display_ref_file("GLM-5.1-UD-Q5_K_XL-00001-of-00013.gguf"),
            "GLM-5.1-UD-Q5_K_XL"
        );
        assert_eq!(display_ref_file("model.safetensors"), "model");
        assert_eq!(
            display_ref_file("model-00001-of-00048.safetensors"),
            "model"
        );
    }

    #[test]
    fn display_exact_ref_uses_short_quant_for_gguf() {
        assert_eq!(
            display_exact_ref(
                "unsloth/gemma-4-26B-A4B-it-GGUF",
                RepoArtifactKind::Gguf,
                "gemma-4-26B-A4B-it-UD-Q3_K_S-00001-of-00009.gguf"
            ),
            "unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q3_K_S"
        );
        assert_eq!(
            display_exact_ref(
                "QuantFactory/Meta-Llama-3.1-8B-Instruct-GGUF",
                RepoArtifactKind::Gguf,
                "Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf"
            ),
            "QuantFactory/Meta-Llama-3.1-8B-Instruct-GGUF:Q4_K_M"
        );
    }

    #[test]
    fn display_exact_ref_prefers_repo_ref_for_mlx() {
        assert_eq!(
            display_exact_ref(
                "mlx-community/Llama-3.2-3B-Instruct-4bit",
                RepoArtifactKind::Mlx,
                "model.safetensors"
            ),
            "mlx-community/Llama-3.2-3B-Instruct-4bit"
        );
    }

    #[test]
    fn mlx_identification_requires_weight_files() {
        assert!(is_mlx_weight_file("model.safetensors"));
        assert!(is_mlx_weight_file("model-00001-of-00008.safetensors"));
        assert!(is_mlx_weight_file("model-00008-of-00008.safetensors"));
        assert!(!is_mlx_weight_file("model.safetensors.index.json"));
    }

    #[test]
    fn split_mlx_candidates_emit_first_shard() {
        let siblings = vec![
            "model-00002-of-00004.safetensors".to_string(),
            "model-00001-of-00004.safetensors".to_string(),
            "model.safetensors.index.json".to_string(),
        ];
        let candidates = collect_repo_artifact_candidates(&siblings);
        let files: Vec<_> = candidates.into_iter().map(|c| c.file).collect();
        assert_eq!(files, vec!["model-00001-of-00004.safetensors".to_string()]);
    }

    #[test]
    fn mlx_candidates_only_include_model_safetensors() {
        let siblings = vec![
            "model.safetensors".to_string(),
            "model.safetensors.index.json".to_string(),
        ];
        let candidates = collect_repo_artifact_candidates(&siblings);
        let files: Vec<_> = candidates.into_iter().map(|c| c.file).collect();
        assert_eq!(files, vec!["model.safetensors".to_string()]);
    }

    #[test]
    fn search_catalog_json_payload_uses_cli_contract_fields() {
        let model = find_catalog_model_exact("Qwen3-Coder-Next-Q4_K_M").expect("catalog model");
        let payload = search_catalog_json_payload(
            "qwen",
            SearchArtifactFilter::Gguf,
            SearchSort::ParametersDesc,
            &[model],
            1,
        );

        assert_eq!(payload["filter"], serde_json::json!("gguf"));
        assert_eq!(payload["sort"], serde_json::json!("parameters-desc"));
        assert!(payload.get("machine").is_some());
        let result = &payload["results"][0];
        assert_eq!(result["ref"], serde_json::json!("Qwen3-Coder-Next-Q4_K_M"));
        assert_eq!(result["type"], serde_json::json!("gguf"));
        assert_eq!(
            result["show"],
            serde_json::json!("mesh-llm models show Qwen3-Coder-Next-Q4_K_M")
        );
    }

    #[test]
    fn search_huggingface_json_payload_uses_cli_contract_fields() {
        let model = find_catalog_model_exact("Qwen3-Coder-Next-Q4_K_M").expect("catalog model");
        let payload = search_huggingface_json_payload(
            "qwen",
            SearchArtifactFilter::Gguf,
            SearchSort::ParametersAsc,
            &[SearchHit {
                repo_id: "Qwen/Qwen3-Coder-Next-GGUF".to_string(),
                kind: "🦙 GGUF",
                exact_ref: "Qwen3-Coder-Next-Q4_K_M".to_string(),
                variant_count: Some(3),
                size_label: Some(model.size.clone()),
                downloads: Some(42),
                likes: Some(7),
                catalog: Some(model),
                capabilities: capabilities::infer_catalog_capabilities(model),
            }],
        );

        assert_eq!(payload["filter"], serde_json::json!("gguf"));
        assert_eq!(payload["sort"], serde_json::json!("parameters-asc"));
        let result = &payload["results"][0];
        assert_eq!(result["ref"], serde_json::json!("Qwen3-Coder-Next-Q4_K_M"));
        assert_eq!(result["type"], serde_json::json!("gguf"));
        assert_eq!(
            result["repo_url"],
            serde_json::json!("https://huggingface.co/Qwen/Qwen3-Coder-Next-GGUF")
        );
        assert_eq!(
            result["catalog"]["name"],
            serde_json::json!("Qwen3-Coder-Next-Q4_K_M")
        );
    }

    #[tokio::test]
    #[ignore = "live Hugging Face search; run explicitly when validating hub integration"]
    async fn live_search_huggingface_gguf_returns_results_and_reports_progress() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let results = search_huggingface(
            "llama",
            5,
            SearchArtifactFilter::Gguf,
            SearchSort::Trending,
            {
                let events = Arc::clone(&events);
                move |progress| events.lock().unwrap().push(progress)
            },
        )
        .await
        .expect("live gguf search should succeed");

        assert!(
            !results.is_empty(),
            "expected at least one live gguf result"
        );
        assert!(
            results.iter().all(|hit| hit.kind == "🦙 GGUF"),
            "expected only GGUF hits, got {results:?}"
        );
        assert!(
            results
                .iter()
                .all(|hit| !hit.repo_id.is_empty() && !hit.exact_ref.is_empty()),
            "expected populated repo ids and refs, got {results:?}"
        );

        let events = events.lock().unwrap().clone();
        assert_progress_sequence(&events);
    }

    #[tokio::test]
    #[ignore = "live Hugging Face search; run explicitly when validating hub integration"]
    async fn live_search_huggingface_mlx_returns_results_and_reports_progress() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let results = search_huggingface(
            "llama",
            5,
            SearchArtifactFilter::Mlx,
            SearchSort::Trending,
            {
                let events = Arc::clone(&events);
                move |progress| events.lock().unwrap().push(progress)
            },
        )
        .await
        .expect("live mlx search should succeed");

        assert!(!results.is_empty(), "expected at least one live mlx result");
        assert!(
            results.iter().all(|hit| hit.kind == "🍎 MLX"),
            "expected only MLX hits, got {results:?}"
        );
        assert!(
            results
                .iter()
                .all(|hit| hit.repo_id == hit.exact_ref || hit.exact_ref.starts_with(&hit.repo_id)),
            "expected mlx refs to stay repo-shaped, got {results:?}"
        );

        let events = events.lock().unwrap().clone();
        assert_progress_sequence(&events);
    }
}
