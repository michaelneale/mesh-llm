use super::resolve::{
    canonical_hf_ref_file_component, file_preference_score, matching_catalog_model_for_huggingface,
    merge_capabilities, remote_hf_size_label_with_api,
};
use super::ModelCapabilities;
use super::{access, build_hf_tokio_api, capabilities, catalog};
use anyhow::{Context, Result};
use hf_hub::api::tokio::Api as TokioApi;
use hf_hub::api::tokio::ApiError;
use hf_hub::api::RepoSummary;
use hf_hub::RepoType;
use std::collections::BTreeMap;
use tokio::task::JoinSet;

#[derive(Clone, Debug)]
pub struct SearchHit {
    pub repo_id: String,
    pub repo_url: String,
    pub description: Option<String>,
    pub recommended_ref: Option<String>,
    pub highest_quality_ref: Option<String>,
    pub metadata_notice: Option<String>,
    pub gguf_files: usize,
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
        .with_filter("gguf")
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
            join_set.spawn(async move { (index, build_search_hit(api, repo).await) });
        }
    }

    let mut completed = 0usize;
    let mut indexed_hits = Vec::new();
    while let Some(joined) = join_set.join_next().await {
        let (index, result) = joined.context("Join Hugging Face repo inspection task")?;
        completed += 1;
        progress(SearchProgress::InspectingRepos { completed, total });
        if let Some(hit) = result? {
            indexed_hits.push((index, hit));
        }
        if let Some((next_index, repo)) = pending.next() {
            let api = api.clone();
            join_set.spawn(async move { (next_index, build_search_hit(api, repo).await) });
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

async fn build_search_hit(api: TokioApi, repo: RepoSummary) -> Result<Option<SearchHit>> {
    let detail = match api.repo(repo.repo()).info().await {
        Ok(detail) => detail,
        Err(err) => {
            if let Some(hit) = build_gated_search_hit(&api, &repo, &err).await? {
                return Ok(Some(hit));
            }
            return Err(err).with_context(|| format!("Fetch Hugging Face repo {}", repo.id));
        }
    };

    let repo_id = detail
        .id
        .clone()
        .or(detail.model_id.clone())
        .unwrap_or(repo.id.clone());
    if detail.gated == Some(true) {
        return Ok(Some(SearchHit {
            repo_id: repo_id.clone(),
            repo_url: access::repo_url(&repo_id),
            description: detail.description.clone().or(repo.description.clone()),
            recommended_ref: None,
            highest_quality_ref: None,
            metadata_notice: Some(access::gated_access_message(&repo_id)),
            gguf_files: 0,
            size_label: None,
            downloads: detail.downloads.or(repo.downloads),
            likes: detail.likes.or(repo.likes),
            catalog: None,
            capabilities: ModelCapabilities::default(),
        }));
    }
    let sibling_names: Vec<String> = detail
        .siblings
        .iter()
        .map(|sibling| sibling.rfilename.clone())
        .collect();
    let files: Vec<String> = detail
        .siblings
        .into_iter()
        .map(|sibling| sibling.rfilename)
        .filter(|file| is_primary_model_gguf(file))
        .collect();
    if files.is_empty() {
        return Ok(None);
    }
    let mut stems: BTreeMap<String, String> = BTreeMap::new();
    for file in files {
        let stem = canonical_hf_ref_file_component(&file);
        let update = match stems.get(&stem) {
            Some(existing) => file_preference_score(&file) < file_preference_score(existing),
            None => true,
        };
        if update {
            stems.insert(stem, file);
        }
    }
    let gguf_files = stems.len();
    let Some((recommended_stem, file)) = stems
        .iter()
        .min_by(|left, right| {
            file_preference_score(left.1)
                .cmp(&file_preference_score(right.1))
                .then_with(|| left.0.cmp(right.0))
        })
        .map(|(stem, file)| (stem.to_string(), file.to_string()))
    else {
        return Ok(None);
    };
    let highest_quality_ref = stems
        .iter()
        .min_by(|left, right| {
            file_quality_score(left.1)
                .cmp(&file_quality_score(right.1))
                .then_with(|| file_preference_score(left.1).cmp(&file_preference_score(right.1)))
                .then_with(|| left.0.cmp(right.0))
        })
        .map(|(stem, _)| format!("{repo_id}/{stem}"));
    let catalog = matching_catalog_model_for_huggingface(&repo_id, None, &file);
    let size_label = match catalog {
        Some(model) => Some(model.size.to_string()),
        None => remote_hf_size_label_with_api(&api, &repo_id, None, &file).await,
    };
    let remote_caps =
        capabilities::infer_remote_hf_capabilities(&repo_id, None, &file, Some(&sibling_names))
            .await;
    let capabilities = match catalog {
        Some(model) => {
            let base = capabilities::infer_catalog_capabilities(model);
            merge_capabilities(base, remote_caps)
        }
        None => remote_caps,
    };
    Ok(Some(SearchHit {
        repo_id: repo_id.clone(),
        repo_url: access::repo_url(&repo_id),
        description: detail.description.clone().or(repo.description.clone()),
        recommended_ref: Some(format!("{repo_id}/{recommended_stem}")),
        highest_quality_ref,
        metadata_notice: None,
        gguf_files,
        size_label,
        downloads: detail.downloads.or(repo.downloads),
        likes: detail.likes.or(repo.likes),
        catalog,
        capabilities,
    }))
}

async fn build_gated_search_hit(
    api: &TokioApi,
    repo: &RepoSummary,
    err: &ApiError,
) -> Result<Option<SearchHit>> {
    let repo_id = repo.id.clone();
    let is_gated = match err {
        ApiError::RequestError(request_error) => {
            if access::reqwest_error_indicates_gated(request_error) {
                true
            } else {
                matches!(
                    access::probe_repo_access(api, &repo_id, None).await?,
                    access::RepoAccess::Gated
                )
            }
        }
        _ => false,
    };

    if !is_gated {
        return Ok(None);
    }

    Ok(Some(SearchHit {
        repo_id: repo_id.clone(),
        repo_url: access::repo_url(&repo_id),
        description: repo.description.clone(),
        recommended_ref: None,
        highest_quality_ref: None,
        metadata_notice: Some(access::gated_access_message(&repo_id)),
        gguf_files: 0,
        size_label: None,
        downloads: repo.downloads,
        likes: repo.likes,
        catalog: None,
        capabilities: ModelCapabilities::default(),
    }))
}

fn file_quality_score(file: &str) -> usize {
    let upper = file.to_ascii_uppercase();
    if upper.contains("BF16") || upper.contains("F16") {
        return 0;
    }
    if upper.contains("Q8_0") {
        return 1;
    }
    if upper.contains("Q6_K") {
        return 2;
    }
    if upper.contains("Q5_") {
        return 3;
    }
    if upper.contains("Q4_") {
        return 4;
    }
    if upper.contains("Q3_") {
        return 5;
    }
    if upper.contains("Q2_") || upper.contains("IQ") {
        return 6;
    }
    7
}

fn is_primary_model_gguf(file: &str) -> bool {
    let lower = file.to_ascii_lowercase();
    lower.ends_with(".gguf") && !lower.contains("mmproj") && !lower.contains("imatrix")
}

#[cfg(test)]
mod tests {
    use super::is_primary_model_gguf;

    #[test]
    fn primary_model_filter_excludes_imatrix_and_mmproj() {
        assert!(is_primary_model_gguf("MiniMax-M2-BF16.i1-IQ1_S.gguf"));
        assert!(!is_primary_model_gguf("MiniMax-M2-BF16.imatrix.gguf"));
        assert!(!is_primary_model_gguf("mmproj-BF16.gguf"));
    }
}
