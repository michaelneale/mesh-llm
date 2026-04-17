use super::local::HuggingFaceModelIdentity;
use super::ModelCapabilities;
use super::{
    capabilities, catalog, find_model_path, format_size_bytes, huggingface_identity_for_path,
    track_model_usage,
};
use crate::cli::terminal_progress::start_spinner;
use anyhow::{anyhow, bail, Context, Result};
use hf_hub::{ListModelsParams, RepoInfo, RepoInfoParams};
use std::cmp::Ordering;
use std::collections::HashSet;
use std::path::{Path, PathBuf};
#[cfg(test)]
use std::sync::{Arc, LazyLock, Mutex};
use tokio_stream::StreamExt;

#[derive(Clone, Debug)]
pub struct ModelDetails {
    pub display_name: String,
    pub exact_ref: String,
    pub source: &'static str,
    pub kind: &'static str,
    pub download_url: String,
    pub size_label: Option<String>,
    pub description: Option<String>,
    pub draft: Option<String>,
    pub capabilities: ModelCapabilities,
    pub moe: Option<catalog::MoeConfig>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ShowVariantsProgress {
    Inspecting { completed: usize, total: usize },
}

#[derive(Clone, Debug)]
enum ExactModelRef {
    Catalog(&'static catalog::CatalogModel),
    HuggingFace {
        repo: String,
        revision: Option<String>,
        file: String,
    },
    Url {
        url: String,
        filename: String,
    },
}

pub(super) fn merge_capabilities(
    left: ModelCapabilities,
    right: ModelCapabilities,
) -> ModelCapabilities {
    ModelCapabilities {
        multimodal: left.multimodal || right.multimodal,
        vision: left.vision.max(right.vision),
        audio: left.audio.max(right.audio),
        reasoning: left.reasoning.max(right.reasoning),
        tool_use: left.tool_use.max(right.tool_use),
        moe: left.moe || right.moe,
    }
}

pub fn find_catalog_model_exact(query: &str) -> Option<&'static catalog::CatalogModel> {
    let q = query.to_lowercase();
    catalog::MODEL_CATALOG.iter().find(|model| {
        model.name.to_lowercase() == q
            || model.file.to_lowercase() == q
            || model.file.trim_end_matches(".gguf").to_lowercase() == q
    })
}

pub async fn download_model_ref_with_progress_details(
    input: &str,
    progress: bool,
) -> Result<(PathBuf, Option<ModelDetails>)> {
    let details = if progress {
        let mut spinner = start_spinner(&format!("Resolving {input}"));
        let details = show_exact_model(input).await.ok();
        spinner.finish();
        details
    } else {
        show_exact_model(input).await.ok()
    };
    let download_ref = details
        .as_ref()
        .map(|detail| detail.download_url.as_str())
        .unwrap_or(input);
    let path = download_exact_ref_with_progress(download_ref, progress).await?;
    Ok((path, details))
}

pub async fn download_exact_ref_with_progress(input: &str, progress: bool) -> Result<PathBuf> {
    let input = canonicalize_model_ref_input(input).await?;
    match parse_exact_model_ref(&input)? {
        ExactModelRef::Catalog(model) => {
            catalog::download_model_with_progress(model, progress).await
        }
        ExactModelRef::HuggingFace {
            repo,
            revision,
            file,
        } => {
            let file = resolve_huggingface_file(&repo, revision.as_deref(), &file).await?;
            if let Some(model) =
                matching_catalog_primary_for_huggingface(&repo, revision.as_deref(), &file)
            {
                return catalog::download_model_with_progress(model, progress).await;
            }
            catalog::download_hf_repo_file_with_progress_label(
                &repo,
                revision.as_deref(),
                &file,
                &input,
                progress,
            )
            .await
        }
        ExactModelRef::Url { url, filename } => {
            if let Some(model) = matching_catalog_primary_for_url(&url) {
                return catalog::download_model_with_progress(model, progress).await;
            }
            let dest = catalog::models_dir().join(&filename);
            if existing_download(&dest).await {
                return Ok(dest);
            }
            if progress {
                eprintln!("📥 Downloading {}...", dest.display());
            }
            catalog::download_hf_split_gguf(&url, &filename).await
        }
    }
}

pub async fn resolve_model_spec(input: &Path) -> Result<PathBuf> {
    resolve_model_spec_with_progress(input, true).await
}

pub async fn resolve_model_spec_with_progress(input: &Path, progress: bool) -> Result<PathBuf> {
    let raw = input.to_string_lossy();

    if input.exists() {
        record_resolved_model_usage(input, Some(raw.as_ref()));
        return Ok(input.to_path_buf());
    }

    if !raw.contains('/') {
        let installed_name = raw.strip_suffix(".gguf").unwrap_or(&raw);
        let installed_path = find_model_path(installed_name);
        if installed_path.exists() {
            let model_ref = huggingface_identity_for_path(&installed_path)
                .map(|identity| identity.canonical_ref)
                .unwrap_or_else(|| installed_name.to_string());
            record_resolved_model_usage(&installed_path, Some(&model_ref));
            return Ok(installed_path);
        }
        if let Some(entry) = catalog::find_model(&raw) {
            return catalog::download_model_with_progress(entry, progress).await;
        }
        if let Ok(canonical) = canonicalize_model_ref_input(&raw).await {
            if canonical != raw {
                return download_exact_ref_with_progress(&canonical, progress)
                    .await
                    .with_context(|| format!("Resolve model spec {raw}"));
            }
        }
        bail!(
            "Model not found: {raw}\nNot a local file, not in the Hugging Face cache, not in catalog.\n\
             Use a path, a catalog name (run `mesh-llm download` to list), or a Hugging Face exact ref/URL."
        );
    }

    let (path, _) = download_model_ref_with_progress_details(&raw, progress)
        .await
        .with_context(|| format!("Resolve model spec {raw}"))?;
    Ok(path)
}

fn record_resolved_model_usage(path: &Path, model_ref: Option<&str>) {
    if let Err(err) = track_model_usage(path, None, model_ref, Some("resolve")) {
        tracing::warn!("failed to record model usage for {}: {err}", path.display());
    }
}

pub async fn show_exact_model(input: &str) -> Result<ModelDetails> {
    let input = canonicalize_model_ref_input(input).await?;
    match parse_exact_model_ref(&input)? {
        ExactModelRef::Catalog(model) => Ok(ModelDetails {
            display_name: model.name.to_string(),
            exact_ref: model.name.to_string(),
            source: "catalog",
            kind: catalog_model_kind(model),
            download_url: match (
                model.source_repo(),
                model.source_revision(),
                model.source_file(),
            ) {
                (Some(repo), revision, Some(file)) => huggingface_resolve_url(repo, revision, file),
                _ => model.url.to_string(),
            },
            size_label: Some(model.size.to_string()),
            description: Some(model.description.to_string()),
            draft: model.draft.clone(),
            capabilities: capabilities::infer_catalog_capabilities(model),
            moe: model.moe.clone(),
        }),
        ExactModelRef::HuggingFace {
            repo,
            revision,
            file,
        } => {
            let file = resolve_huggingface_file(&repo, revision.as_deref(), &file).await?;
            let exact_ref = format_huggingface_display_ref(&repo, revision.as_deref(), &file);
            let catalog = matching_catalog_model_for_huggingface(&repo, revision.as_deref(), &file);
            let download_url = huggingface_resolve_url(&repo, revision.as_deref(), &file);
            let size_label = match catalog {
                Some(model) => Some(model.size.to_string()),
                None => remote_size_label(&download_url).await,
            };
            let capabilities = match catalog {
                Some(model) => {
                    let base = capabilities::infer_catalog_capabilities(model);
                    let remote = capabilities::infer_remote_hf_capabilities(
                        &repo,
                        revision.as_deref(),
                        &file,
                        None,
                    )
                    .await;
                    merge_capabilities(base, remote)
                }
                None => {
                    capabilities::infer_remote_hf_capabilities(
                        &repo,
                        revision.as_deref(),
                        &file,
                        None,
                    )
                    .await
                }
            };
            Ok(ModelDetails {
                display_name: Path::new(&file)
                    .file_name()
                    .and_then(|value| value.to_str())
                    .unwrap_or(&file)
                    .to_string(),
                exact_ref,
                source: "huggingface",
                kind: artifact_kind_for_file(&file),
                download_url,
                size_label,
                description: catalog.map(|model| model.description.to_string()),
                draft: catalog.and_then(|model| model.draft.clone()),
                capabilities,
                moe: catalog.and_then(|model| model.moe.clone()),
            })
        }
        ExactModelRef::Url { url, filename } => {
            let catalog = matching_catalog_model_for_url(&url);
            let size_label = match catalog {
                Some(model) => Some(model.size.to_string()),
                None => remote_size_label(&url).await,
            };
            Ok(ModelDetails {
                display_name: filename,
                exact_ref: url.clone(),
                source: "url",
                kind: artifact_kind_for_file(&url),
                download_url: url,
                size_label,
                description: catalog.map(|model| model.description.to_string()),
                draft: catalog.and_then(|model| model.draft.clone()),
                capabilities: catalog
                    .map(capabilities::infer_catalog_capabilities)
                    .unwrap_or_default(),
                moe: catalog.and_then(|model| model.moe.clone()),
            })
        }
    }
}

pub async fn resolve_huggingface_model_identity(
    input: &str,
) -> Result<Option<HuggingFaceModelIdentity>> {
    let input = canonicalize_model_ref_input(input).await?;
    match parse_exact_model_ref(&input)? {
        ExactModelRef::Catalog(model) => {
            let (Some(repo), revision, Some(file)) = (
                model.source_repo(),
                model.source_revision(),
                model.source_file(),
            ) else {
                return Ok(None);
            };
            let revision = revision.unwrap_or("main");
            let api = super::build_hf_tokio_api(false)?;
            let (owner, name) = repo.split_once('/').unwrap_or(("", repo));
            let detail = api
                .model(owner, name)
                .info(
                    &RepoInfoParams::builder()
                        .revision(revision.to_string())
                        .build(),
                )
                .await
                .with_context(|| format!("Fetch Hugging Face repo {repo}@{revision}"))?;
            let RepoInfo::Model(detail) = detail else {
                return Ok(None);
            };
            Ok(Some(HuggingFaceModelIdentity {
                repo_id: repo.to_string(),
                revision: detail.sha.clone().unwrap_or_else(|| revision.to_string()),
                file: file.to_string(),
                canonical_ref: format!(
                    "{repo}@{}/{file}",
                    detail.sha.as_deref().unwrap_or(revision)
                ),
                local_file_name: Path::new(file)
                    .file_name()
                    .and_then(|value| value.to_str())
                    .unwrap_or(file)
                    .to_string(),
            }))
        }
        ExactModelRef::HuggingFace {
            repo,
            revision,
            file,
        } => {
            let resolved_file = resolve_huggingface_file(&repo, revision.as_deref(), &file).await?;
            let revision_ref = revision.as_deref().unwrap_or("main");
            let api = super::build_hf_tokio_api(false)?;
            let (owner, name) = repo.split_once('/').unwrap_or(("", repo.as_str()));
            let detail = api
                .model(owner, name)
                .info(
                    &RepoInfoParams::builder()
                        .revision(revision_ref.to_string())
                        .build(),
                )
                .await
                .with_context(|| format!("Fetch Hugging Face repo {repo}@{revision_ref}"))?;
            let RepoInfo::Model(detail) = detail else {
                return Ok(None);
            };
            let resolved_revision = detail
                .sha
                .clone()
                .unwrap_or_else(|| revision_ref.to_string());
            Ok(Some(HuggingFaceModelIdentity {
                repo_id: repo.clone(),
                revision: resolved_revision.clone(),
                canonical_ref: format!("{}@{}/{}", repo, resolved_revision, resolved_file),
                local_file_name: Path::new(&resolved_file)
                    .file_name()
                    .and_then(|value| value.to_str())
                    .unwrap_or(&resolved_file)
                    .to_string(),
                file: resolved_file,
            }))
        }
        ExactModelRef::Url { url, .. } => {
            if let Some((repo, revision, file)) = parse_hf_resolve_url(&url) {
                let revision_ref = revision.as_deref().unwrap_or("main");
                let api = super::build_hf_tokio_api(false)?;
                let (owner, name) = repo.split_once('/').unwrap_or(("", repo.as_str()));
                let detail = api
                    .model(owner, name)
                    .info(
                        &RepoInfoParams::builder()
                            .revision(revision_ref.to_string())
                            .build(),
                    )
                    .await
                    .with_context(|| format!("Fetch Hugging Face repo {repo}@{revision_ref}"))?;
                let RepoInfo::Model(detail) = detail else {
                    return Ok(None);
                };
                let resolved_revision = detail
                    .sha
                    .clone()
                    .unwrap_or_else(|| revision_ref.to_string());
                return Ok(Some(HuggingFaceModelIdentity {
                    repo_id: repo.clone(),
                    revision: resolved_revision.clone(),
                    canonical_ref: format!("{}@{}/{}", repo, resolved_revision, file),
                    local_file_name: Path::new(&file)
                        .file_name()
                        .and_then(|value| value.to_str())
                        .unwrap_or(&file)
                        .to_string(),
                    file,
                }));
            }
            Ok(None)
        }
    }
}

pub async fn show_model_variants_with_progress<F>(
    input: &str,
    mut progress: F,
) -> Result<Option<Vec<ModelDetails>>>
where
    F: FnMut(ShowVariantsProgress),
{
    let input = canonicalize_model_ref_input(input).await?;
    let parsed = parse_huggingface_repo_ref(&input).or_else(|| parse_huggingface_repo_url(&input));
    let Some((repo, revision, selector)) = parsed else {
        return Ok(None);
    };
    if selector.is_some() {
        return Ok(None);
    }

    let api = super::build_hf_tokio_api(false)?;
    let revision_ref = revision.as_deref().unwrap_or("main");
    let (owner, name) = repo.split_once('/').unwrap_or(("", repo.as_str()));
    let detail = api
        .model(owner, name)
        .info(
            &RepoInfoParams::builder()
                .revision(revision_ref.to_string())
                .build(),
        )
        .await
        .with_context(|| format!("Fetch Hugging Face repo {repo}"))?;
    let RepoInfo::Model(detail) = detail else {
        return Ok(Some(Vec::new()));
    };

    let sibling_entries: Vec<(String, Option<u64>)> = detail
        .siblings
        .clone()
        .unwrap_or_default()
        .iter()
        .map(|sibling| (sibling.rfilename.clone(), sibling.size))
        .collect();
    let available_bytes = crate::system::hardware::survey().vram_bytes;
    let variants = collect_show_gguf_variants_from_siblings(&sibling_entries, available_bytes);
    if variants.is_empty() {
        return Ok(Some(Vec::new()));
    }

    let quant_variants: Vec<_> = variants
        .into_iter()
        .filter(|(file, _)| quant_selector_from_gguf_file(file).is_some())
        .collect();
    let total = quant_variants.len();
    progress(ShowVariantsProgress::Inspecting {
        completed: 0,
        total,
    });

    let mut seen_refs = HashSet::new();
    let mut out = Vec::new();
    for (idx, (file, size_bytes)) in quant_variants.into_iter().enumerate() {
        let exact_ref = format_huggingface_display_ref(&repo, revision.as_deref(), &file);
        if !seen_refs.insert(exact_ref.clone()) {
            progress(ShowVariantsProgress::Inspecting {
                completed: idx + 1,
                total,
            });
            continue;
        }
        let size_label = match size_bytes {
            Some(bytes) => Some(format_size_bytes(bytes)),
            None => remote_hf_size_label_with_api(&api, &repo, revision.as_deref(), &file).await,
        };
        out.push(ModelDetails {
            display_name: Path::new(&file)
                .file_name()
                .and_then(|value| value.to_str())
                .unwrap_or(&file)
                .to_string(),
            exact_ref,
            source: "huggingface",
            kind: artifact_kind_for_file(&file),
            download_url: huggingface_resolve_url(&repo, revision.as_deref(), &file),
            size_label,
            description: None,
            draft: None,
            capabilities: ModelCapabilities::default(),
            moe: None,
        });
        progress(ShowVariantsProgress::Inspecting {
            completed: idx + 1,
            total,
        });
    }

    Ok(Some(out))
}

pub(super) fn quant_selector_from_gguf_file(file: &str) -> Option<String> {
    if !file.ends_with(".gguf") {
        return None;
    }

    if let Some((prefix, _)) = file.split_once('/') {
        if is_quant_like_selector(prefix) {
            return Some(prefix.to_string());
        }
    }

    let basename = Path::new(file).file_name()?.to_str()?;
    let mut stem = basename.strip_suffix(".gguf")?;
    if let Some((base, shard)) = stem.rsplit_once("-00001-of-") {
        if shard.len() == 5 && shard.chars().all(|ch| ch.is_ascii_digit()) {
            stem = base;
        }
    }

    for marker in [
        "-UD-", ".UD-", "-IQ", ".IQ", "-Q", ".Q", "-BF16", ".BF16", "-F16", ".F16", "-F32", ".F32",
    ] {
        if let Some(pos) = stem.rfind(marker) {
            return Some(stem[pos + 1..].to_string());
        }
    }
    None
}

fn is_quant_like_selector(value: &str) -> bool {
    let upper = value.to_ascii_uppercase();
    upper.starts_with("UD-")
        || upper.starts_with("Q")
        || upper.starts_with("IQ")
        || upper == "BF16"
        || upper == "F16"
        || upper == "F32"
}

fn format_repo_selector_ref(repo: &str, revision: Option<&str>, selector: &str) -> String {
    match revision {
        Some(revision) => format!("{repo}:{selector}@{revision}"),
        None => format!("{repo}:{selector}"),
    }
}

fn format_huggingface_display_ref(repo: &str, revision: Option<&str>, file: &str) -> String {
    if let Some(selector) = quant_selector_from_gguf_file(file) {
        return format_repo_selector_ref(repo, revision, &selector);
    }
    if is_primary_mlx_weight_file(file) {
        return match revision {
            Some(revision) => format!("{repo}@{revision}"),
            None => repo.to_string(),
        };
    }
    format_huggingface_exact_ref(repo, revision, file)
}

fn artifact_kind_for_file(file: &str) -> &'static str {
    if file.ends_with(".safetensors") || file.ends_with(".safetensors.index.json") {
        "🍎 MLX"
    } else {
        "🦙 GGUF"
    }
}

fn catalog_model_kind(model: &catalog::CatalogModel) -> &'static str {
    if model
        .source_file()
        .map(|file| {
            file.ends_with("model.safetensors") || file.ends_with("model.safetensors.index.json")
        })
        .unwrap_or(false)
        || model.url.contains("model.safetensors")
    {
        "🍎 MLX"
    } else {
        "🦙 GGUF"
    }
}

pub fn installed_model_capabilities(model_name: &str) -> ModelCapabilities {
    let path = find_model_path(model_name);
    let catalog = find_catalog_model_exact(model_name);
    capabilities::infer_local_model_capabilities(model_name, &path, catalog)
}

pub fn installed_model_display_name(model_name: &str) -> String {
    find_catalog_model_exact(model_name)
        .map(|model| model.name.clone())
        .unwrap_or_else(|| model_name.to_string())
}

pub fn installed_model_huggingface_ref(identity: &HuggingFaceModelIdentity) -> String {
    format_huggingface_display_ref(&identity.repo_id, None, &identity.file)
}

pub(super) fn catalog_hf_asset_ref(
    model: &'static catalog::CatalogModel,
    file_name: &str,
) -> Option<(String, Option<String>, String)> {
    if model.file == file_name {
        return Some((
            model.source_repo()?.to_string(),
            model.source_revision().map(str::to_string),
            model.source_file()?.to_string(),
        ));
    }

    let source_url = if let Some(asset) = model
        .extra_files
        .iter()
        .find(|asset| asset.file == file_name)
    {
        asset.url.as_str()
    } else if let Some(asset) = model.mmproj.as_ref() {
        if asset.file == file_name {
            asset.url.as_str()
        } else {
            return None;
        }
    } else {
        return None;
    };

    parse_hf_resolve_url(source_url)
}

pub(super) fn matching_catalog_model_for_huggingface(
    repo: &str,
    revision: Option<&str>,
    file: &str,
) -> Option<&'static catalog::CatalogModel> {
    let repo = repo.to_lowercase();
    let revision = revision.map(|value| value.to_lowercase());
    let file = file.to_lowercase();

    catalog::MODEL_CATALOG
        .iter()
        .find(|model| {
            std::iter::once(model.file.as_str())
                .chain(model.extra_files.iter().map(|asset| asset.file.as_str()))
                .chain(model.mmproj.iter().map(|asset| asset.file.as_str()))
                .any(|asset_name| {
                    let Some((asset_repo, asset_revision, asset_file)) =
                        catalog_hf_asset_ref(model, asset_name)
                    else {
                        return false;
                    };
                    if asset_repo.to_lowercase() != repo || asset_file.to_lowercase() != file {
                        return false;
                    }
                    match &revision {
                        Some(revision) => {
                            asset_revision.map(|value| value.to_lowercase())
                                == Some(revision.clone())
                        }
                        None => true,
                    }
                })
        })
        .or_else(|| {
            if revision.is_some() {
                None
            } else {
                matching_catalog_model_by_basename(file.as_str())
            }
        })
}

fn matching_catalog_model_for_url(url: &str) -> Option<&'static catalog::CatalogModel> {
    catalog::MODEL_CATALOG
        .iter()
        .find(|model| model.url.eq_ignore_ascii_case(url))
        .or_else(|| matching_catalog_model_by_basename(url))
}

fn matching_catalog_primary_for_huggingface(
    repo: &str,
    revision: Option<&str>,
    file: &str,
) -> Option<&'static catalog::CatalogModel> {
    let model = matching_catalog_model_for_huggingface(repo, revision, file)?;
    match catalog_hf_asset_ref(model, model.file.as_str()) {
        Some((asset_repo, asset_revision, asset_file))
            if asset_repo.eq_ignore_ascii_case(repo)
                && asset_file.eq_ignore_ascii_case(file)
                && match revision {
                    Some(revision) => asset_revision
                        .as_deref()
                        .map(|value| value.eq_ignore_ascii_case(revision))
                        .unwrap_or(false),
                    None => true,
                } =>
        {
            Some(model)
        }
        _ => None,
    }
}

fn matching_catalog_primary_for_url(url: &str) -> Option<&'static catalog::CatalogModel> {
    let model = matching_catalog_model_for_url(url)?;
    if model.url.eq_ignore_ascii_case(url) {
        Some(model)
    } else {
        None
    }
}

fn matching_catalog_model_by_basename(repo_file: &str) -> Option<&'static catalog::CatalogModel> {
    let basename = Path::new(repo_file)
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or(repo_file)
        .to_lowercase();
    catalog::MODEL_CATALOG.iter().find(|model| {
        model.file.to_lowercase() == basename
            || model.file.trim_end_matches(".gguf").to_lowercase()
                == basename.trim_end_matches(".gguf")
    })
}

pub(super) fn parse_hf_resolve_url(url: &str) -> Option<(String, Option<String>, String)> {
    let tail = url
        .strip_prefix("https://huggingface.co/")
        .or_else(|| url.strip_prefix("http://huggingface.co/"))?;
    let parts: Vec<&str> = tail.split('/').collect();
    if parts.len() < 5 || parts.get(2) != Some(&"resolve") {
        return None;
    }
    Some((
        format!("{}/{}", parts[0], parts[1]),
        parts.get(3).map(|value| value.to_string()),
        parts[4..].join("/"),
    ))
}

pub(super) fn parse_huggingface_ref(input: &str) -> Option<(String, Option<String>, String)> {
    if let Some(parsed) = parse_hf_resolve_url(input) {
        return Some(parsed);
    }

    let parts: Vec<&str> = input.splitn(3, '/').collect();
    if parts.len() != 3 {
        return None;
    }
    if parts[0].is_empty() || parts[1].is_empty() || parts[0].contains(':') {
        return None;
    }
    let (repo_tail, revision) = match parts[1].split_once('@') {
        Some((repo, revision)) => (repo, Some(revision.to_string())),
        None => (parts[1], None),
    };
    if repo_tail.is_empty() {
        return None;
    }
    Some((
        format!("{}/{}", parts[0], repo_tail),
        revision,
        parts[2].to_string(),
    ))
}

fn parse_repo_tail_selector_and_revision(
    tail: &str,
) -> Option<(String, Option<String>, Option<String>)> {
    let (with_selector, revision) = match tail.split_once('@') {
        Some((repo, revision)) => {
            if repo.is_empty() || revision.is_empty() {
                return None;
            }
            (repo, Some(revision.to_string()))
        }
        None => (tail, None),
    };
    let (repo_tail, selector) = match with_selector.split_once(':') {
        Some((repo, selector)) => {
            if repo.is_empty() || selector.is_empty() {
                return None;
            }
            (repo, Some(selector.to_string()))
        }
        None => (with_selector, None),
    };
    Some((repo_tail.to_string(), revision, selector))
}

fn parse_huggingface_repo_ref(input: &str) -> Option<(String, Option<String>, Option<String>)> {
    let parts: Vec<&str> = input.splitn(2, '/').collect();
    if parts.len() != 2 {
        return None;
    }
    if parts[0].is_empty() || parts[1].is_empty() || parts[0].contains(':') {
        return None;
    }
    let (repo_tail, revision, selector) = parse_repo_tail_selector_and_revision(parts[1])?;
    Some((format!("{}/{}", parts[0], repo_tail), revision, selector))
}

fn parse_huggingface_repo_url(input: &str) -> Option<(String, Option<String>, Option<String>)> {
    let tail = input
        .strip_prefix("https://huggingface.co/")
        .or_else(|| input.strip_prefix("http://huggingface.co/"))?;
    let clean = tail
        .split_once('?')
        .map(|(left, _)| left)
        .unwrap_or(tail)
        .split_once('#')
        .map(|(left, _)| left)
        .unwrap_or(tail)
        .trim_matches('/');
    let parts: Vec<&str> = clean.split('/').collect();
    if parts.len() < 2 || parts[0].is_empty() || parts[1].is_empty() {
        return None;
    }
    let (repo_tail, selector_revision, selector) = parse_repo_tail_selector_and_revision(parts[1])?;
    let repo = format!("{}/{}", parts[0], repo_tail);
    if parts.len() >= 4 && parts[2] == "tree" && !parts[3].is_empty() {
        return Some((repo, Some(parts[3].to_string()), selector));
    }
    if parts.len() == 2 {
        return Some((repo, selector_revision, selector));
    }
    None
}

fn parse_exact_model_ref(input: &str) -> Result<ExactModelRef> {
    if let Some(model) = find_catalog_model_exact(input) {
        return Ok(ExactModelRef::Catalog(model));
    }
    if let Some((repo, revision, file)) = parse_huggingface_ref(input) {
        return Ok(ExactModelRef::HuggingFace {
            repo,
            revision,
            file,
        });
    }
    if let Some((repo, revision, selector)) = parse_huggingface_repo_ref(input) {
        return Ok(ExactModelRef::HuggingFace {
            repo,
            revision,
            file: selector.unwrap_or_default(),
        });
    }
    if let Some((repo, revision, selector)) = parse_huggingface_repo_url(input) {
        return Ok(ExactModelRef::HuggingFace {
            repo,
            revision,
            file: selector.unwrap_or_default(),
        });
    }
    if input.starts_with("http://") || input.starts_with("https://") {
        return Ok(ExactModelRef::Url {
            url: input.to_string(),
            filename: remote_filename(input)?,
        });
    }
    bail!(
        "Expected an exact model ref. Use a catalog id, a Hugging Face ref like org/repo, org/repo:QUANT, org/repo/file.gguf, org/repo/file-stem for split GGUFs, org/repo/model.safetensors, or org/repo/model-00001-of-00048.safetensors, or a direct URL."
    )
}

fn split_bare_name_selector(input: &str) -> (&str, Option<&str>) {
    match input.split_once(':') {
        Some((name, selector))
            if !name.is_empty()
                && !selector.is_empty()
                && !name.contains('/')
                && !name.contains('@')
                && !name.contains("://") =>
        {
            (name, Some(selector))
        }
        _ => (input, None),
    }
}

fn normalize_repo_leaf_name(value: &str) -> String {
    value
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .map(|ch| ch.to_ascii_lowercase())
        .collect()
}

fn select_strong_repo_hit(query: &str, repo_ids: &[String]) -> Option<String> {
    let query_norm = normalize_repo_leaf_name(query);
    if query_norm.is_empty() {
        return None;
    }
    let mut exact = Vec::new();
    for repo_id in repo_ids {
        let leaf = repo_id.rsplit('/').next().unwrap_or(repo_id);
        if normalize_repo_leaf_name(leaf) == query_norm {
            exact.push(repo_id.clone());
        }
    }
    if let Some(first) = exact.into_iter().next() {
        return Some(first);
    }
    None
}

async fn discover_hf_repo_for_bare_name(name: &str) -> Result<Option<String>> {
    let api = super::build_hf_tokio_api(false)?;
    let params = ListModelsParams::builder()
        .search(name.to_string())
        .filter("gguf".to_string())
        .limit(20_usize)
        .build();
    let stream = api
        .list_models(&params)
        .with_context(|| format!("Search Hugging Face for '{name}'"))?;
    tokio::pin!(stream);
    let mut repo_ids = Vec::new();
    while let Some(repo) = stream.next().await {
        repo_ids.push(repo?.id);
    }
    Ok(select_strong_repo_hit(name, &repo_ids))
}

async fn canonicalize_model_ref_input(input: &str) -> Result<String> {
    if parse_exact_model_ref(input).is_ok() || find_catalog_model_exact(input).is_some() {
        return Ok(input.to_string());
    }
    if input.contains('/') || input.starts_with("http://") || input.starts_with("https://") {
        return Ok(input.to_string());
    }

    let (name, selector) = split_bare_name_selector(input);
    if let Some(repo) = discover_hf_repo_for_bare_name(name).await? {
        if let Some(selector) = selector {
            return Ok(format!("{repo}:{selector}"));
        }
        return Ok(repo);
    }
    Ok(input.to_string())
}

fn is_split_mlx_first_shard(file: &str) -> bool {
    let Some(rest) = file.strip_prefix("model-") else {
        return false;
    };
    let Some(rest) = rest.strip_suffix(".safetensors") else {
        return false;
    };
    let Some((left, right)) = rest.split_once("-of-") else {
        return false;
    };
    left == "00001" && right.len() == 5 && right.bytes().all(|byte| byte.is_ascii_digit())
}

fn is_primary_mlx_weight_file(file: &str) -> bool {
    let basename = file.rsplit('/').next().unwrap_or(file);
    basename.eq_ignore_ascii_case("model.safetensors") || is_split_mlx_first_shard(basename)
}

fn select_default_hf_file_from_siblings(siblings: &[String]) -> Option<String> {
    siblings
        .iter()
        .filter_map(|file| {
            let lower = file.to_lowercase();
            let rank = if lower == "model.safetensors" {
                0
            } else if is_split_mlx_first_shard(&lower) {
                1
            } else if lower.ends_with(".gguf") {
                if is_known_gguf_sidecar(file) {
                    return None;
                }
                if lower.contains("-000") && !lower.contains("-00001-of-") {
                    return None;
                }
                if lower.contains("-00001-of-") {
                    2
                } else {
                    3
                }
            } else {
                return None;
            };
            Some((rank, file_preference_score(file), file.clone()))
        })
        .min_by(|left, right| left.cmp(right))
        .map(|(_, _, file)| file)
}

fn resolve_hf_file_from_siblings(requested: &str, siblings: &[String]) -> Option<String> {
    if requested.is_empty() {
        return select_default_hf_file_from_siblings(siblings);
    }

    if requested.ends_with(".gguf")
        || requested.ends_with(".safetensors")
        || requested.ends_with(".safetensors.index.json")
    {
        return Some(requested.to_string());
    }

    let requested_lower = requested.to_lowercase();
    let gguf_exact = format!("{requested}.gguf").to_lowercase();
    let gguf_split_prefix = format!("{requested}-00001-of-").to_lowercase();
    let safetensors_exact = format!("{requested}.safetensors").to_lowercase();
    let safetensors_split_prefix = format!("{requested}-00001-of-").to_lowercase();

    siblings
        .iter()
        .filter_map(|file| {
            let lower = file.to_lowercase();
            let rank = if lower == requested_lower {
                0
            } else if gguf_matches_quant_selector(&lower, &requested_lower) {
                1
            } else if lower == safetensors_exact {
                2
            } else if lower.starts_with(&safetensors_split_prefix)
                && lower.ends_with(".safetensors")
            {
                3
            } else if lower == gguf_exact {
                4
            } else if lower.starts_with(&gguf_split_prefix) && lower.ends_with(".gguf") {
                5
            } else {
                return None;
            };
            Some((rank, file_preference_score(file), file.clone()))
        })
        .min_by(|left, right| left.cmp(right))
        .map(|(_, _, file)| file)
}

fn gguf_matches_quant_selector(file_lower: &str, selector_lower: &str) -> bool {
    if !file_lower.ends_with(".gguf") || selector_lower.is_empty() {
        return false;
    }
    file_lower.contains(&format!("/{selector_lower}/"))
        || file_lower.contains(&format!("-{selector_lower}-"))
        || file_lower.ends_with(&format!("-{selector_lower}.gguf"))
        || file_lower.ends_with(&format!("/{selector_lower}.gguf"))
}

pub(super) fn is_known_gguf_sidecar(file: &str) -> bool {
    let basename = file.rsplit('/').next().unwrap_or(file);
    basename.to_ascii_lowercase().starts_with("mmproj")
}

fn collect_show_gguf_variants_from_siblings(
    siblings: &[(String, Option<u64>)],
    available_bytes: u64,
) -> Vec<(String, Option<u64>)> {
    let mut gguf_candidates: Vec<(String, Option<u64>)> = siblings
        .iter()
        .filter_map(|(file, size)| {
            let lower = file.to_lowercase();
            if !lower.ends_with(".gguf") {
                return None;
            }
            if is_known_gguf_sidecar(file) {
                return None;
            }
            if lower.contains("-000") && !lower.contains("-00001-of-") {
                return None;
            }
            Some((file.clone(), *size))
        })
        .collect();

    if available_bytes == 0 {
        gguf_candidates.sort_by(|left, right| {
            file_preference_score(&left.0)
                .cmp(&file_preference_score(&right.0))
                .then_with(|| left.0.cmp(&right.0))
        });
        return gguf_candidates;
    }

    gguf_candidates.sort_by(|left, right| {
        compare_gguf_candidates_by_fit(&left.0, left.1, &right.0, right.1, available_bytes)
    });
    gguf_candidates
}

fn fit_bucket(size_bytes: u64, available_bytes: u64) -> u8 {
    if size_bytes.saturating_mul(10) <= available_bytes.saturating_mul(9) {
        0
    } else if size_bytes.saturating_mul(10) <= available_bytes.saturating_mul(11) {
        1
    } else {
        2
    }
}

fn compare_gguf_candidates_by_fit(
    left_file: &str,
    left_size: Option<u64>,
    right_file: &str,
    right_size: Option<u64>,
    available_bytes: u64,
) -> Ordering {
    match (left_size, right_size) {
        (Some(left), Some(right)) => {
            let left_bucket = fit_bucket(left, available_bytes);
            let right_bucket = fit_bucket(right, available_bytes);
            if left_bucket != right_bucket {
                return left_bucket.cmp(&right_bucket);
            }
            let size_order = if left_bucket <= 1 {
                right.cmp(&left)
            } else {
                left.cmp(&right)
            };
            if size_order != Ordering::Equal {
                return size_order;
            }
        }
        (Some(_), None) => return Ordering::Less,
        (None, Some(_)) => return Ordering::Greater,
        (None, None) => {}
    }

    file_preference_score(left_file)
        .cmp(&file_preference_score(right_file))
        .then_with(|| left_file.cmp(right_file))
}

async fn remote_size_bytes(url: &str) -> Option<u64> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(300))
        .connect_timeout(std::time::Duration::from_secs(30))
        .user_agent(format!("mesh-llm/{}", crate::VERSION))
        .build()
        .ok()?;
    let response = client
        .head(url)
        .send()
        .await
        .ok()?
        .error_for_status()
        .ok()?;
    response
        .headers()
        .get(reqwest::header::CONTENT_LENGTH)?
        .to_str()
        .ok()?
        .parse::<u64>()
        .ok()
}

async fn select_default_hf_file_fit_aware(
    repo: &str,
    revision: Option<&str>,
    siblings: &[(String, Option<u64>)],
) -> Option<String> {
    let mut gguf_candidates: Vec<(String, Option<u64>)> = Vec::new();
    for (file, api_size) in siblings {
        let lower = file.to_lowercase();
        if !lower.ends_with(".gguf") {
            continue;
        }
        if is_known_gguf_sidecar(file) {
            continue;
        }
        if lower.contains("-000") && !lower.contains("-00001-of-") {
            continue;
        }
        gguf_candidates.push((file.clone(), *api_size));
    }
    if gguf_candidates.is_empty() {
        return None;
    }

    let available_bytes = crate::system::hardware::survey().vram_bytes;
    if available_bytes == 0 {
        gguf_candidates.sort_by(|left, right| {
            file_preference_score(&left.0)
                .cmp(&file_preference_score(&right.0))
                .then_with(|| left.0.cmp(&right.0))
        });
        return gguf_candidates.first().map(|(f, _)| f.clone());
    }

    // Prefer API-provided sizes; only fall back to HEAD for files missing a size.
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(300))
        .connect_timeout(std::time::Duration::from_secs(30))
        .user_agent(format!("mesh-llm/{}", crate::VERSION))
        .build()
        .ok();
    let mut scored: Vec<(String, Option<u64>)> = Vec::with_capacity(gguf_candidates.len());
    for (file, api_size) in gguf_candidates {
        let size = if api_size.is_some() {
            api_size
        } else if let Some(ref c) = client {
            let url = huggingface_resolve_url(repo, revision, &file);
            c.head(&url)
                .send()
                .await
                .ok()
                .and_then(|r| r.error_for_status().ok())
                .and_then(|r| {
                    r.headers()
                        .get(reqwest::header::CONTENT_LENGTH)
                        .and_then(|v| v.to_str().ok())
                        .and_then(|s| s.parse::<u64>().ok())
                })
        } else {
            None
        };
        scored.push((file, size));
    }
    scored.sort_by(|left, right| {
        compare_gguf_candidates_by_fit(&left.0, left.1, &right.0, right.1, available_bytes)
    });
    scored.first().map(|(file, _)| file.clone())
}

fn repo_prefers_gguf_only(repo: &str) -> bool {
    repo.to_ascii_lowercase().contains("gguf")
}

#[cfg(test)]
type RepoSiblingEntriesOverrideFn =
    Arc<dyn Fn(&str, &str) -> Option<Vec<(String, Option<u64>)>> + Send + Sync>;

#[cfg(test)]
static REPO_SIBLING_ENTRIES_OVERRIDE: LazyLock<Mutex<Option<RepoSiblingEntriesOverrideFn>>> =
    LazyLock::new(|| Mutex::new(None));

#[cfg(test)]
struct RepoSiblingEntriesOverrideGuard;

#[cfg(test)]
impl RepoSiblingEntriesOverrideGuard {
    fn set(func: RepoSiblingEntriesOverrideFn) -> Self {
        let mut slot = REPO_SIBLING_ENTRIES_OVERRIDE.lock().unwrap();
        *slot = Some(func);
        Self
    }
}

#[cfg(test)]
impl Drop for RepoSiblingEntriesOverrideGuard {
    fn drop(&mut self) {
        let mut slot = REPO_SIBLING_ENTRIES_OVERRIDE.lock().unwrap();
        *slot = None;
    }
}

async fn fetch_repo_sibling_entries(
    repo: &str,
    revision: &str,
) -> Result<Vec<(String, Option<u64>)>> {
    #[cfg(test)]
    {
        let func = REPO_SIBLING_ENTRIES_OVERRIDE.lock().unwrap().clone();
        if let Some(func) = func {
            if let Some(entries) = func(repo, revision) {
                return Ok(entries);
            }
        }
    }

    let api = super::build_hf_tokio_api(false)?;
    let (owner, name) = repo.split_once('/').unwrap_or(("", repo));
    let detail = api
        .model(owner, name)
        .info(
            &RepoInfoParams::builder()
                .revision(revision.to_string())
                .build(),
        )
        .await
        .with_context(|| format!("Fetch Hugging Face repo {repo}@{revision}"))?;
    let RepoInfo::Model(detail) = detail else {
        bail!("Expected model repo info for {repo}@{revision}");
    };
    Ok(detail
        .siblings
        .unwrap_or_default()
        .iter()
        .map(|sibling| (sibling.rfilename.clone(), sibling.size))
        .collect())
}

async fn resolve_huggingface_file(
    repo: &str,
    revision: Option<&str>,
    file: &str,
) -> Result<String> {
    if file.ends_with(".gguf")
        || file.ends_with(".safetensors")
        || file.ends_with(".safetensors.index.json")
    {
        return Ok(file.to_string());
    }

    let revision = revision.unwrap_or("main");
    let sibling_entries = fetch_repo_sibling_entries(repo, revision).await?;
    let siblings: Vec<String> = sibling_entries.iter().map(|(f, _)| f.clone()).collect();
    let has_mlx_weights = siblings
        .iter()
        .any(|entry| entry == "model.safetensors" || is_split_mlx_first_shard(entry));

    if file.is_empty() {
        let gguf_only = repo_prefers_gguf_only(repo);
        if gguf_only {
            if let Some(resolved) =
                select_default_hf_file_fit_aware(repo, Some(revision), &sibling_entries).await
            {
                return Ok(resolved);
            }
            bail!("No GGUF model files found in {repo}@{revision}.");
        }

        if let Some(resolved) = select_default_hf_file_from_siblings(&siblings) {
            return Ok(resolved);
        }

        if let Some(resolved) =
            select_default_hf_file_fit_aware(repo, Some(revision), &sibling_entries).await
        {
            return Ok(resolved);
        }
    }
    if file == "model" && has_mlx_weights {
        bail!(
            "MLX shorthand '/model' is not supported. Use '{repo}' or a full file ref like '{repo}/model.safetensors'."
        );
    }

    if let Some(resolved) = resolve_hf_file_from_siblings(file, &siblings) {
        return Ok(resolved);
    }

    bail!(
        "No model file matching stem '{file}' in {repo}@{revision}. Use a full ref like org/repo/file.gguf or org/repo/model.safetensors."
    )
}

pub(super) fn huggingface_resolve_url(repo: &str, revision: Option<&str>, file: &str) -> String {
    let revision = revision.unwrap_or("main");
    format!("https://huggingface.co/{repo}/resolve/{revision}/{file}")
}

fn format_huggingface_exact_ref(repo: &str, revision: Option<&str>, file: &str) -> String {
    match revision {
        Some(revision) => format!("{repo}@{revision}/{file}"),
        None => format!("{repo}/{file}"),
    }
}

fn remote_filename(input: &str) -> Result<String> {
    input
        .rsplit('/')
        .next()
        .filter(|name| !name.is_empty())
        .map(str::to_string)
        .ok_or_else(|| anyhow!("Cannot extract filename from URL: {input}"))
}

async fn existing_download(path: &Path) -> bool {
    tokio::fs::metadata(path)
        .await
        .map(|meta| meta.len() > 1_000_000)
        .unwrap_or(false)
}

pub(super) fn file_preference_score(file: &str) -> usize {
    if file.contains("-00001-of-") {
        return 0;
    }
    const PREFERRED: &[&str] = &[
        "Q4_K_M", "Q4_K_S", "Q4_1", "Q5_K_M", "Q5_K_S", "Q8_0", "BF16",
    ];
    PREFERRED
        .iter()
        .position(|needle| file.contains(needle))
        .map(|pos| pos + 1)
        .unwrap_or(PREFERRED.len() + 2)
}

async fn remote_size_label(url: &str) -> Option<String> {
    let size = remote_size_bytes(url).await?;
    Some(format_size_bytes(size))
}

pub(super) async fn remote_hf_size_label_with_api(
    _api: &hf_hub::HFClient,
    repo: &str,
    revision: Option<&str>,
    file: &str,
) -> Option<String> {
    let url = huggingface_resolve_url(repo, revision, file);
    remote_size_label(&url).await
}

#[cfg(test)]
mod tests;
