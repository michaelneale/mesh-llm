use super::ModelCapabilities;
use super::{capabilities, catalog, find_model_path, format_size_bytes};
use anyhow::{anyhow, bail, Context, Result};
use std::path::{Path, PathBuf};

#[derive(Clone, Debug)]
pub struct ModelDetails {
    pub display_name: String,
    pub exact_ref: String,
    pub source: &'static str,
    pub download_url: String,
    pub size_label: Option<String>,
    pub description: Option<String>,
    pub draft: Option<String>,
    pub capabilities: ModelCapabilities,
    pub moe: Option<catalog::MoeConfig>,
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

pub async fn download_exact_ref(input: &str) -> Result<PathBuf> {
    match parse_exact_model_ref(input)? {
        ExactModelRef::Catalog(model) => catalog::download_model(model).await,
        ExactModelRef::HuggingFace {
            repo,
            revision,
            file,
        } => {
            if let Some(model) =
                matching_catalog_primary_for_huggingface(&repo, revision.as_deref(), &file)
            {
                return catalog::download_model(model).await;
            }
            catalog::download_hf_repo_file(&repo, revision.as_deref(), &file).await
        }
        ExactModelRef::Url { url, filename } => {
            if let Some(model) = matching_catalog_primary_for_url(&url) {
                return catalog::download_model(model).await;
            }
            let dest = catalog::models_dir().join(&filename);
            if existing_download(&dest).await {
                return Ok(dest);
            }
            eprintln!("📥 Downloading {}...", dest.display());
            catalog::download_hf_split_gguf(&url, &filename).await
        }
    }
}

pub async fn resolve_model_spec(input: &Path) -> Result<PathBuf> {
    let raw = input.to_string_lossy();

    if input.exists() {
        return Ok(input.to_path_buf());
    }

    if !raw.contains('/') {
        for dir in super::model_dirs() {
            let candidate = dir.join(input);
            if candidate.exists() {
                return Ok(candidate);
            }
        }
        let installed_name = raw.strip_suffix(".gguf").unwrap_or(&raw);
        let installed_path = find_model_path(installed_name);
        if installed_path.exists() {
            return Ok(installed_path);
        }
        if let Some(entry) = catalog::find_model(&raw) {
            return catalog::download_model(entry).await;
        }
        bail!(
            "Model not found: {raw}\nNot a local file, not in the Hugging Face cache or legacy ~/.models, not in catalog.\n\
             Use a path, a catalog name (run `mesh-llm download` to list), or a Hugging Face exact ref/URL."
        );
    }

    download_exact_ref(&raw)
        .await
        .with_context(|| format!("Resolve model spec {raw}"))
}

pub async fn show_exact_model(input: &str) -> Result<ModelDetails> {
    match parse_exact_model_ref(input)? {
        ExactModelRef::Catalog(model) => Ok(ModelDetails {
            display_name: model.name.to_string(),
            exact_ref: model.name.to_string(),
            source: "catalog",
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
            let exact_ref = format_huggingface_exact_ref(&repo, revision.as_deref(), &file);
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

pub(super) fn catalog_hf_match(
    path: &Path,
) -> Option<(
    &'static catalog::CatalogModel,
    String,
    Option<String>,
    String,
)> {
    let model = catalog_match(path)?;
    let file_name = path.file_name()?.to_str()?;
    if model.file == file_name {
        if let (Some(repo), revision, Some(file)) = (
            model.source_repo(),
            model.source_revision(),
            model.source_file(),
        ) {
            return Some((
                model,
                repo.to_string(),
                revision.map(str::to_string),
                file.to_string(),
            ));
        }
        return None;
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

    let (repo, revision, file) = parse_hf_resolve_url(source_url)?;
    Some((model, repo, revision, file))
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

pub(super) fn catalog_match(path: &Path) -> Option<&'static catalog::CatalogModel> {
    let file_name = path.file_name()?.to_str()?;
    catalog::MODEL_CATALOG.iter().find(|model| {
        model.file == file_name
            || model
                .extra_files
                .iter()
                .any(|asset| asset.file == file_name)
            || model
                .mmproj
                .as_ref()
                .map(|asset| asset.file == file_name)
                .unwrap_or(false)
    })
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn primary_hf_ref_maps_to_full_catalog_download() {
        let model = matching_catalog_primary_for_huggingface(
            "unsloth/Qwen3.5-0.8B-GGUF",
            Some("main"),
            "Qwen3.5-0.8B-Q4_K_M.gguf",
        )
        .expect("primary model file should map to catalog download");
        assert_eq!(model.name, "Qwen3.5-0.8B-Vision-Q4_K_M");
        assert!(model.mmproj.is_some());
    }

    #[test]
    fn mmproj_hf_ref_does_not_expand_to_full_catalog_download() {
        assert!(matching_catalog_primary_for_huggingface(
            "unsloth/Qwen3.5-0.8B-GGUF",
            Some("main"),
            "mmproj-BF16.gguf",
        )
        .is_none());
    }

    #[test]
    fn primary_url_maps_to_full_catalog_download() {
        let model = matching_catalog_primary_for_url(
            "https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-Q4_K_M.gguf",
        )
        .expect("primary model url should map to catalog download");
        assert_eq!(model.name, "Qwen3.5-0.8B-Vision-Q4_K_M");
        assert!(model.mmproj.is_some());
    }

    #[test]
    fn mmproj_url_does_not_expand_to_full_catalog_download() {
        assert!(matching_catalog_primary_for_url(
            "https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/mmproj-BF16.gguf",
        )
        .is_none());
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
    if !input.ends_with(".gguf") {
        return None;
    }

    let parts: Vec<&str> = input.splitn(3, '/').collect();
    if parts.len() != 3 {
        return None;
    }
    let (repo_tail, revision) = match parts[1].split_once('@') {
        Some((repo, revision)) => (repo, Some(revision.to_string())),
        None => (parts[1], None),
    };
    Some((
        format!("{}/{}", parts[0], repo_tail),
        revision,
        parts[2].to_string(),
    ))
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
    if input.starts_with("http://") || input.starts_with("https://") {
        return Ok(ExactModelRef::Url {
            url: input.to_string(),
            filename: remote_filename(input)?,
        });
    }
    bail!(
        "Expected an exact model ref. Use a catalog id, a Hugging Face ref like org/repo/file.gguf, or a direct URL."
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
        .unwrap_or(PREFERRED.len() + 1)
}

async fn remote_size_label(url: &str) -> Option<String> {
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
    let size = response
        .headers()
        .get(reqwest::header::CONTENT_LENGTH)?
        .to_str()
        .ok()?
        .parse::<u64>()
        .ok()?;
    Some(format_size_bytes(size))
}

pub(super) async fn remote_hf_size_label_with_api(
    _api: &hf_hub::api::tokio::Api,
    repo: &str,
    revision: Option<&str>,
    file: &str,
) -> Option<String> {
    let url = huggingface_resolve_url(repo, revision, file);
    remote_size_label(&url).await
}
