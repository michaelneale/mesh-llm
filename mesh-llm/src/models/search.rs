use super::resolve::{
    artifact_kind_label, collect_repo_artifact_candidates, matching_catalog_model_for_huggingface,
    merge_capabilities, remote_hf_size_label_with_api, RepoArtifactKind,
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
    pub file: String,
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
    let repos = api
        .search(RepoType::Model)
        .with_query(query)
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
        for hit in result? {
            indexed_hits.push((index, hit));
        }
        if let Some((next_index, repo)) = pending.next() {
            let api = api.clone();
            join_set.spawn(async move { (next_index, build_search_hit(api, repo, filter).await) });
        }
    }

    indexed_hits.sort_by_key(|(index, _)| *index);
    let mut hits: Vec<SearchHit> = indexed_hits
        .into_iter()
        .map(|(_, hit)| hit)
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
    let candidates = collect_repo_artifact_candidates(&sibling_names);
    if candidates.is_empty() {
        return Ok(Vec::new());
    }

    let mut hits = Vec::new();
    for candidate in candidates {
        let matches_filter = match filter {
            SearchArtifactFilter::Gguf => candidate.kind == RepoArtifactKind::Gguf,
            SearchArtifactFilter::Mlx => candidate.kind == RepoArtifactKind::Mlx,
        };
        if !matches_filter {
            continue;
        }
        let catalog = matching_catalog_model_for_huggingface(&repo_id, None, &candidate.file);
        let size_label = match catalog {
            Some(model) => Some(model.size.to_string()),
            None => remote_hf_size_label_with_api(&api, &repo_id, None, &candidate.file).await,
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
            file: candidate.file.clone(),
            kind: artifact_kind_label(candidate.kind),
            exact_ref: format!("{repo_id}/{}", candidate.file),
            size_label,
            downloads: detail.downloads.or(repo.downloads),
            likes: detail.likes.or(repo.likes),
            catalog,
            capabilities,
        });
    }
    Ok(hits)
}
