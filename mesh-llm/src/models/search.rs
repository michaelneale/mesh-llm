use super::resolve::{
    file_preference_score, is_known_gguf_sidecar, matching_catalog_model_for_huggingface,
    merge_capabilities, quant_selector_from_gguf_file, remote_hf_size_label_with_api,
};
use super::ModelCapabilities;
use super::{build_hf_tokio_api, capabilities, catalog};
use anyhow::{Context, Result};
use hf_hub::api::tokio::Api as TokioApi;
use hf_hub::api::RepoSummary;
use hf_hub::RepoType;
use tokio::task::JoinSet;

#[derive(Clone, Debug)]
pub struct SearchHit {
    pub repo_id: String,
    pub kind: &'static str,
    pub exact_ref: String,
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

// Keep search custom for now. `hf-hub` handles cache and file transport well,
// but it does not expose a Hub search surface in this crate version.
pub async fn search_huggingface<F>(
    query: &str,
    limit: usize,
    filter: SearchArtifactFilter,
    mut progress: F,
) -> Result<Vec<SearchHit>>
where
    F: FnMut(SearchProgress),
{
    const SEARCH_CONCURRENCY: usize = 6;

    let repo_limit = limit.clamp(1, 100);
    progress(SearchProgress::SearchingHub);
    let api = build_hf_tokio_api(false)?;
    let mut search = api.search(RepoType::Model).with_query(query);
    search = match filter {
        SearchArtifactFilter::Gguf => search.with_filter("gguf"),
        SearchArtifactFilter::Mlx => search.with_filter("mlx"),
    };
    let repos = search
        .with_limit(repo_limit)
        .run()
        .await
        .context("Search Hugging Face")?;

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
            Ok(hits) => {
                for (rank, hit) in hits.into_iter().enumerate() {
                    indexed_hits.push((index, rank, hit));
                }
            }
            Err(err) => {
                eprintln!("⚠️  Failed to inspect Hugging Face repo: {err:#}");
            }
        }
        if let Some((next_index, repo)) = pending.next() {
            let api = api.clone();
            join_set.spawn(async move { (next_index, build_search_hit(api, repo, filter).await) });
        }
    }

    indexed_hits.sort_by_key(|(index, rank, _)| (*index, *rank));
    let mut hits: Vec<SearchHit> = indexed_hits
        .into_iter()
        .map(|(_, _, hit)| hit)
        .take(limit)
        .collect();
    if hits.len() > limit {
        hits.truncate(limit);
    }
    Ok(hits)
}

async fn build_search_hit(
    api: TokioApi,
    repo: RepoSummary,
    filter: SearchArtifactFilter,
) -> Result<Vec<SearchHit>> {
    let detail = api
        .repo(repo.repo())
        .info()
        .await
        .with_context(|| format!("Fetch Hugging Face repo {}", repo.id))?;

    let repo_id = detail
        .id
        .clone()
        .or(detail.model_id.clone())
        .unwrap_or(repo.id.clone());
    let sibling_names: Vec<String> = detail
        .siblings
        .iter()
        .map(|sibling| sibling.rfilename.clone())
        .collect();
    let sibling_size_entries: Vec<(String, Option<u64>)> = detail
        .siblings
        .iter()
        .map(|sibling| (sibling.rfilename.clone(), sibling.size))
        .collect();
    let repo_has_mlx_weights = sibling_names.iter().any(|file| is_mlx_weight_file(file));
    let candidates = collect_repo_artifact_candidates(&sibling_names);
    if candidates.is_empty() {
        return Ok(Vec::new());
    }

    let mut hits = Vec::new();
    for candidate in candidates {
        let matches_filter = match filter {
            SearchArtifactFilter::Gguf => candidate.kind == RepoArtifactKind::Gguf,
            SearchArtifactFilter::Mlx => {
                candidate.kind == RepoArtifactKind::Mlx && repo_has_mlx_weights
            }
        };
        if !matches_filter {
            continue;
        }
        let catalog = matching_catalog_model_for_huggingface(&repo_id, None, &candidate.file);
        let size_label = match catalog {
            Some(model) => Some(model.size.to_string()),
            None => match size_label_from_sibling_entries(&candidate.file, &sibling_size_entries) {
                Some(size_label) => Some(size_label),
                None => remote_hf_size_label_with_api(&api, &repo_id, None, &candidate.file).await,
            },
        };
        let remote_caps = capabilities::infer_remote_hf_capabilities(
            &repo_id,
            None,
            &candidate.file,
            Some(&sibling_names),
        )
        .await;
        let capabilities = match catalog {
            Some(model) => {
                let base = capabilities::infer_catalog_capabilities(model);
                merge_capabilities(base, remote_caps)
            }
            None => remote_caps,
        };
        hits.push(SearchHit {
            repo_id: repo_id.clone(),
            kind: repo_artifact_kind_label(candidate.kind),
            exact_ref: display_exact_ref(&repo_id, candidate.kind, &candidate.file),
            size_label,
            downloads: detail.downloads.or(repo.downloads),
            likes: detail.likes.or(repo.likes),
            catalog,
            capabilities,
        });
    }
    Ok(hits)
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
}
