use super::ModelCapabilities;
use super::{capabilities, catalog, find_model_path, format_size_bytes};
use anyhow::{anyhow, bail, Context, Result};
use hf_hub::{Repo, RepoType};
use std::cmp::Ordering;
use std::path::{Path, PathBuf};

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
            let file = resolve_huggingface_file(&repo, revision.as_deref(), &file).await?;
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
        let installed_name = raw.strip_suffix(".gguf").unwrap_or(&raw);
        let installed_path = find_model_path(installed_name);
        if installed_path.exists() {
            return Ok(installed_path);
        }
        if let Some(entry) = catalog::find_model(&raw) {
            return catalog::download_model(entry).await;
        }
        bail!(
            "Model not found: {raw}\nNot a local file, not in the Hugging Face cache, not in catalog.\n\
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

    #[test]
    fn split_stem_resolves_to_first_part() {
        let siblings = vec![
            "zai-org.GLM-5.1.Q2_K-00002-of-00018.gguf".to_string(),
            "zai-org.GLM-5.1.Q2_K-00001-of-00018.gguf".to_string(),
        ];
        let resolved = resolve_hf_file_from_siblings("zai-org.GLM-5.1.Q2_K", &siblings).unwrap();
        assert_eq!(resolved, "zai-org.GLM-5.1.Q2_K-00001-of-00018.gguf");
    }

    #[test]
    fn stem_without_split_resolves_to_gguf() {
        let siblings = vec![
            "Qwen3-8B-Q4_K_M.gguf".to_string(),
            "Qwen3-8B-Q8_0.gguf".to_string(),
        ];
        let resolved = resolve_hf_file_from_siblings("Qwen3-8B-Q4_K_M", &siblings).unwrap();
        assert_eq!(resolved, "Qwen3-8B-Q4_K_M.gguf");
    }

    #[test]
    fn mlx_stem_resolves_to_model_safetensors() {
        let siblings = vec![
            "model.safetensors.index.json".to_string(),
            "model.safetensors".to_string(),
        ];
        let resolved = resolve_hf_file_from_siblings("model", &siblings).unwrap();
        assert_eq!(resolved, "model.safetensors");
    }

    #[test]
    fn mlx_stem_resolves_to_first_split_shard() {
        let siblings = vec![
            "model-00002-of-00048.safetensors".to_string(),
            "model-00001-of-00048.safetensors".to_string(),
            "model.safetensors.index.json".to_string(),
        ];
        let resolved = resolve_hf_file_from_siblings("model", &siblings).unwrap();
        assert_eq!(resolved, "model-00001-of-00048.safetensors");
    }

    #[test]
    fn repo_only_resolution_prefers_mlx_model_safetensors() {
        let siblings = vec![
            "Qwen3-8B-Q4_K_M.gguf".to_string(),
            "model.safetensors".to_string(),
            "model.safetensors.index.json".to_string(),
        ];
        let resolved = resolve_hf_file_from_siblings("", &siblings).unwrap();
        assert_eq!(resolved, "model.safetensors");
    }

    #[test]
    fn repo_only_resolution_falls_back_to_gguf_when_no_mlx_weights() {
        let siblings = vec![
            "Qwen3-8B-Q8_0.gguf".to_string(),
            "Qwen3-8B-Q4_K_M.gguf".to_string(),
        ];
        let resolved = resolve_hf_file_from_siblings("", &siblings).unwrap();
        assert_eq!(resolved, "Qwen3-8B-Q4_K_M.gguf");
    }

    #[test]
    fn parse_huggingface_ref_rejects_http_url() {
        assert!(parse_huggingface_ref("https://example.com/model.gguf").is_none());
    }

    #[test]
    fn parse_huggingface_repo_ref_parses_repo_only() {
        let parsed = parse_huggingface_repo_ref("GreenBitAI/Llama-2-7B-layer-mix-bpw-2.2-mlx");
        assert_eq!(
            parsed,
            Some((
                "GreenBitAI/Llama-2-7B-layer-mix-bpw-2.2-mlx".to_string(),
                None,
                None
            ))
        );
    }

    #[test]
    fn parse_huggingface_repo_ref_parses_quant_selector() {
        let parsed = parse_huggingface_repo_ref("unsloth/gemma-4-31B-it-GGUF:UD-Q4_K_XL");
        assert_eq!(
            parsed,
            Some((
                "unsloth/gemma-4-31B-it-GGUF".to_string(),
                None,
                Some("UD-Q4_K_XL".to_string())
            ))
        );
    }

    #[test]
    fn parse_huggingface_repo_url_parses_repo_only() {
        let parsed =
            parse_huggingface_repo_url("https://huggingface.co/unsloth/gemma-4-31B-it-GGUF");
        assert_eq!(
            parsed,
            Some(("unsloth/gemma-4-31B-it-GGUF".to_string(), None, None))
        );
    }

    #[test]
    fn parse_huggingface_repo_url_parses_tree_revision() {
        let parsed = parse_huggingface_repo_url(
            "https://huggingface.co/unsloth/gemma-4-31B-it-GGUF/tree/main",
        );
        assert_eq!(
            parsed,
            Some((
                "unsloth/gemma-4-31B-it-GGUF".to_string(),
                Some("main".to_string()),
                None
            ))
        );
    }

    #[test]
    fn quant_selector_resolves_to_first_matching_split_gguf() {
        let siblings = vec![
            "UD-Q5_K_XL/gemma-4-31B-it-UD-Q5_K_XL-00001-of-00009.gguf".to_string(),
            "UD-Q4_K_XL/gemma-4-31B-it-UD-Q4_K_XL-00002-of-00009.gguf".to_string(),
            "UD-Q4_K_XL/gemma-4-31B-it-UD-Q4_K_XL-00001-of-00009.gguf".to_string(),
        ];
        let resolved = resolve_hf_file_from_siblings("UD-Q4_K_XL", &siblings).unwrap();
        assert_eq!(
            resolved,
            "UD-Q4_K_XL/gemma-4-31B-it-UD-Q4_K_XL-00001-of-00009.gguf"
        );
    }

    #[test]
    fn fit_aware_gguf_prefers_largest_comfortable_candidate() {
        let available = 20_000_000_000u64;
        let ordering = compare_gguf_candidates_by_fit(
            "repo/model-q4.gguf",
            Some(12_000_000_000),
            "repo/model-q5.gguf",
            Some(17_000_000_000),
            available,
        );
        assert_eq!(ordering, Ordering::Greater);
    }

    #[test]
    fn fit_aware_gguf_prefers_smaller_when_both_too_large() {
        let available = 20_000_000_000u64;
        let ordering = compare_gguf_candidates_by_fit(
            "repo/model-q8.gguf",
            Some(29_000_000_000),
            "repo/model-bf16.gguf",
            Some(35_000_000_000),
            available,
        );
        assert_eq!(ordering, Ordering::Less);
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

async fn remote_hf_size_bytes(repo: &str, revision: Option<&str>, file: &str) -> Option<u64> {
    let url = huggingface_resolve_url(repo, revision, file);
    remote_size_bytes(&url).await
}

async fn select_default_hf_file_fit_aware(
    repo: &str,
    revision: Option<&str>,
    siblings: &[String],
) -> Option<String> {
    let mut gguf_candidates = Vec::new();
    for file in siblings {
        let lower = file.to_lowercase();
        if !lower.ends_with(".gguf") {
            continue;
        }
        if lower.contains("-000") && !lower.contains("-00001-of-") {
            continue;
        }
        gguf_candidates.push(file.clone());
    }
    if gguf_candidates.is_empty() {
        return select_default_hf_file_from_siblings(siblings);
    }

    let available_bytes = crate::system::hardware::survey().vram_bytes;
    if available_bytes == 0 {
        return select_default_hf_file_from_siblings(siblings);
    }

    let mut scored: Vec<(String, Option<u64>)> = Vec::with_capacity(gguf_candidates.len());
    for file in gguf_candidates {
        let size = remote_hf_size_bytes(repo, revision, &file).await;
        scored.push((file, size));
    }
    scored.sort_by(|left, right| {
        compare_gguf_candidates_by_fit(&left.0, left.1, &right.0, right.1, available_bytes)
    });
    scored
        .first()
        .map(|(file, _)| file.clone())
        .or_else(|| select_default_hf_file_from_siblings(siblings))
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
    let api = super::build_hf_tokio_api(false)?;
    let detail = api
        .repo(Repo::with_revision(
            repo.to_string(),
            RepoType::Model,
            revision.to_string(),
        ))
        .info()
        .await
        .with_context(|| format!("Fetch Hugging Face repo {repo}@{revision}"))?;
    let siblings: Vec<String> = detail
        .siblings
        .iter()
        .map(|sibling| sibling.rfilename.clone())
        .collect();

    if file.is_empty() {
        if let Some(resolved) =
            select_default_hf_file_fit_aware(repo, Some(revision), &siblings).await
        {
            return Ok(resolved);
        }
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
    _api: &hf_hub::api::tokio::Api,
    repo: &str,
    revision: Option<&str>,
    file: &str,
) -> Option<String> {
    let url = huggingface_resolve_url(repo, revision, file);
    remote_size_label(&url).await
}
