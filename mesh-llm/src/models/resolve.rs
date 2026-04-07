use super::ModelCapabilities;
use super::{capabilities, catalog, find_model_path, format_size_bytes};
use crate::system::hardware;
use anyhow::{anyhow, bail, Context, Result};
use hf_hub::api::{RepoInfo, RepoSummary, Siblings};
use hf_hub::{Repo, RepoType};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
#[cfg(test)]
use std::sync::{Arc, LazyLock, Mutex};

#[derive(Clone, Debug)]
pub struct ModelDetails {
    pub display_name: String,
    pub exact_ref: String,
    pub source: &'static str,
    pub download_url: String,
    pub size_label: Option<String>,
    pub total_size_bytes: Option<u64>,
    pub quant: Option<String>,
    pub fit: Option<bool>,
    pub resolved_files: Vec<String>,
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CapabilityProfile {
    Text,
    Vision,
    Audio,
    Multimodal,
}

impl CapabilityProfile {
    pub fn from_env() -> Self {
        match std::env::var("MESH_LLM_CAPABILITY_PROFILE")
            .ok()
            .as_deref()
            .map(|value| value.to_ascii_lowercase())
            .as_deref()
        {
            Some("vision") => Self::Vision,
            Some("audio") => Self::Audio,
            Some("multimodal") => Self::Multimodal,
            _ => Self::Text,
        }
    }

    pub fn as_label(self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Vision => "vision",
            Self::Audio => "audio",
            Self::Multimodal => "multimodal",
        }
    }
}

#[derive(Clone, Debug)]
struct VariantCandidate {
    stem: String,
    file: String,
    capabilities: ModelCapabilities,
    size_bytes: Option<u64>,
}

#[derive(Clone, Debug)]
struct RepoSelection {
    repo: String,
    selected: VariantCandidate,
    fit: Option<bool>,
}

#[derive(Clone, Debug)]
pub struct RepoInspection {
    pub repo: String,
    pub description: Option<String>,
    pub downloads: Option<u64>,
    pub likes: Option<u64>,
    pub recommended_ref: String,
    pub highest_quality_ref: String,
    pub recommended_fit: Option<bool>,
    pub recommended_size_bytes: Option<u64>,
    pub highest_quality_size_bytes: Option<u64>,
    pub variant_count: usize,
    pub variants: Vec<RepoVariantInspection>,
}

#[derive(Clone, Debug)]
pub struct RepoVariantInspection {
    pub reference: String,
    pub quant: String,
    pub capability: String,
    pub size_bytes: Option<u64>,
    pub fit: Option<bool>,
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
    download_exact_ref_with_profile(input, CapabilityProfile::from_env()).await
}

pub async fn download_exact_ref_with_profile(
    input: &str,
    profile: CapabilityProfile,
) -> Result<PathBuf> {
    if is_imatrix_artifact(input) {
        bail!(
            "🟡 '{}' is an imatrix artifact, not a runnable model. Choose a .gguf model variant instead.",
            input
        );
    }

    #[cfg(test)]
    {
        let override_fn = DOWNLOAD_EXACT_REF_OVERRIDE.lock().unwrap().clone();
        if let Some(override_fn) = override_fn {
            return override_fn(input, profile);
        }
    }

    if let Some(selection) =
        resolve_repo_or_family_ref(input, profile, true).await?
    {
        if selection.fit == Some(false) {
            eprintln!("🟡 This model is likely too large for local serving on this machine.");
        }
        return catalog::download_hf_repo_file(&selection.repo, None, &selection.selected.file).await;
    }

    match parse_exact_model_ref(input)? {
        ExactModelRef::Catalog(model) => catalog::download_model(model).await,
        ExactModelRef::HuggingFace {
            repo,
            revision,
            file,
        } => {
            let resolved_file =
                resolve_huggingface_file_selector(&repo, revision.as_deref(), &file).await?;
            if let Some(model) =
                matching_catalog_primary_for_huggingface(&repo, revision.as_deref(), &resolved_file)
            {
                return catalog::download_model(model).await;
            }
            catalog::download_hf_repo_file(&repo, revision.as_deref(), &resolved_file).await
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

#[cfg(test)]
type DownloadExactRefOverrideFn =
    Arc<dyn Fn(&str, CapabilityProfile) -> Result<PathBuf> + Send + Sync>;

#[cfg(test)]
static DOWNLOAD_EXACT_REF_OVERRIDE: LazyLock<Mutex<Option<DownloadExactRefOverrideFn>>> =
    LazyLock::new(|| Mutex::new(None));

#[cfg(test)]
pub(crate) struct DownloadExactRefOverrideGuard;

#[cfg(test)]
impl DownloadExactRefOverrideGuard {
    pub(crate) fn set(func: DownloadExactRefOverrideFn) -> Self {
        *DOWNLOAD_EXACT_REF_OVERRIDE.lock().unwrap() = Some(func);
        Self
    }
}

#[cfg(test)]
impl Drop for DownloadExactRefOverrideGuard {
    fn drop(&mut self) {
        *DOWNLOAD_EXACT_REF_OVERRIDE.lock().unwrap() = None;
    }
}

pub async fn resolve_model_spec(input: &Path) -> Result<PathBuf> {
    let raw = input.to_string_lossy();

    if input.exists() {
        if is_imatrix_artifact(&raw) {
            bail!(
                "🟡 '{}' is an imatrix artifact, not a runnable model. Choose a .gguf model variant instead.",
                raw
            );
        }
        return Ok(input.to_path_buf());
    }

    if is_imatrix_artifact(&raw) {
        bail!(
            "🟡 '{}' is an imatrix artifact, not a runnable model. Choose a .gguf model variant instead.",
            raw
        );
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
        if let Some(selection) =
            resolve_repo_or_family_ref(&raw, CapabilityProfile::from_env(), true).await?
        {
            return catalog::download_hf_repo_file(&selection.repo, None, &selection.selected.file)
                .await
                .with_context(|| format!("Resolve model spec {raw}"));
        }
        bail!(
            "Model not found: {raw}\nNot a local file, not in the Hugging Face cache, not in catalog.\n\
             Use a path, a catalog name (run `mesh-llm download` to list), or a Hugging Face exact ref/URL."
        );
    }

    if let Some(selection) =
        resolve_repo_or_family_ref(&raw, CapabilityProfile::from_env(), true).await?
    {
        return catalog::download_hf_repo_file(&selection.repo, None, &selection.selected.file)
            .await
            .with_context(|| format!("Resolve model spec {raw}"));
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
            total_size_bytes: None,
            quant: None,
            fit: None,
            resolved_files: Vec::new(),
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
            let info = repo_info_with_blobs(
                &super::build_hf_tokio_api(false)?,
                Repo::with_revision(
                    repo.clone(),
                    RepoType::Model,
                    revision.clone().unwrap_or_else(|| "main".to_string()),
                ),
            )
            .await
            .with_context(|| format!("Fetch Hugging Face repo {}", repo))?;
            let selector_key = normalize_selector_key(&file);
            let mut resolved_files: Vec<String> = info
                .siblings
                .iter()
                .map(|sibling| sibling.rfilename.clone())
                .filter(|name| is_primary_model_gguf(name))
                .filter(|name| normalize_selector_key(name) == selector_key)
                .collect();
            resolved_files.sort_by(|left, right| {
                file_preference_score(left)
                    .cmp(&file_preference_score(right))
                    .then_with(|| left.cmp(right))
            });
            let resolved_file = resolved_files
                .first()
                .cloned()
                .or_else(|| None)
                .unwrap_or_else(|| file.clone());
            let exact_ref = format_huggingface_exact_ref(
                &repo,
                revision.as_deref(),
                &canonical_hf_ref_file_component(&resolved_file),
            );
            let catalog =
                matching_catalog_model_for_huggingface(&repo, revision.as_deref(), &resolved_file);
            let download_url = huggingface_resolve_url(&repo, revision.as_deref(), &resolved_file);
            let total_size_bytes = {
                let mut all_known = true;
                let mut total = 0u64;
                for name in &resolved_files {
                    if let Some(sibling) = info.siblings.iter().find(|value| value.rfilename == *name) {
                        if let Some(size) = sibling.size {
                            total = total.saturating_add(size);
                        } else {
                            all_known = false;
                            break;
                        }
                    } else {
                        all_known = false;
                        break;
                    }
                }
                if all_known && !resolved_files.is_empty() {
                    Some(total)
                } else {
                    None
                }
            };
            let size_label = match (catalog, total_size_bytes) {
                (_, Some(total)) => Some(format_size_bytes(total)),
                (Some(model), None) => Some(model.size.to_string()),
                (None, None) => remote_size_label(&download_url).await,
            };
            let quant = exact_ref
                .rsplit('/')
                .next()
                .and_then(|value| value.split('-').next_back())
                .map(|value| value.to_string());
            let fit = match total_size_bytes {
                Some(total) => {
                    let vram = hardware::survey().vram_bytes;
                    if vram > 0 {
                        Some((total as f64 * 1.1) as u64 <= vram)
                    } else {
                        None
                    }
                }
                None => None,
            };
            let capabilities = match catalog {
                Some(model) => {
                    let base = capabilities::infer_catalog_capabilities(model);
                    let remote = capabilities::infer_remote_hf_capabilities(
                        &repo,
                        revision.as_deref(),
                        &resolved_file,
                        None,
                    )
                    .await;
                    merge_capabilities(base, remote)
                }
                None => {
                    capabilities::infer_remote_hf_capabilities(
                        &repo,
                        revision.as_deref(),
                        &resolved_file,
                        None,
                    )
                    .await
                }
            };
            Ok(ModelDetails {
                display_name: Path::new(&resolved_file)
                    .file_name()
                    .and_then(|value| value.to_str())
                    .unwrap_or(&resolved_file)
                    .to_string(),
                exact_ref,
                source: "huggingface",
                download_url,
                size_label,
                total_size_bytes,
                quant,
                fit,
                resolved_files,
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
                total_size_bytes: None,
                quant: None,
                fit: None,
                resolved_files: Vec::new(),
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
mod selector_tests {
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

pub(super) fn canonical_hf_ref_file_component(file: &str) -> String {
    split_gguf_stem(file)
        .unwrap_or_else(|| file.to_string())
        .trim_end_matches(".gguf")
        .to_string()
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

async fn resolve_huggingface_file_selector(
    repo: &str,
    revision: Option<&str>,
    selector: &str,
) -> Result<String> {
    if selector.ends_with(".gguf") {
        return Ok(selector.to_string());
    }

    let target = normalize_selector_key(selector);
    let api = super::build_hf_tokio_api(false)?;
    let revision = revision.unwrap_or("main").to_string();
    let repo_handle = Repo::with_revision(repo.to_string(), RepoType::Model, revision);
    let detail = api
        .repo(repo_handle)
        .info()
        .await
        .with_context(|| format!("Resolve Hugging Face model selector {repo}/{selector}"))?;

    let files: Vec<String> = detail
        .siblings
        .into_iter()
        .map(|sibling| sibling.rfilename)
        .collect();

    choose_hf_file_for_selector(&target, &files)
        .ok_or_else(|| anyhow::anyhow!("Model file selector not found in repo: {repo}/{selector}"))
}

fn normalize_selector_key(value: &str) -> String {
    let basename = Path::new(value)
        .file_name()
        .and_then(|part| part.to_str())
        .unwrap_or(value);
    split_gguf_stem(basename)
        .unwrap_or_else(|| basename.trim_end_matches(".gguf").to_string())
        .to_lowercase()
}

fn choose_hf_file_for_selector(selector_key: &str, files: &[String]) -> Option<String> {
    let mut matches: Vec<String> = files
        .iter()
        .filter(|file| file.ends_with(".gguf"))
        .filter(|file| normalize_selector_key(file) == selector_key)
        .cloned()
        .collect();
    if matches.is_empty() {
        return None;
    }
    matches.sort_by(|left, right| {
        file_preference_score(left)
            .cmp(&file_preference_score(right))
            .then_with(|| left.cmp(right))
    });
    matches.into_iter().next()
}

fn split_gguf_stem(value: &str) -> Option<String> {
    let stem = value.trim_end_matches(".gguf");
    let of_index = stem.rfind("-of-")?;
    if !stem[of_index + 4..].chars().all(|ch| ch.is_ascii_digit()) {
        return None;
    }
    let prefix = &stem[..of_index];
    let dash = prefix.rfind('-')?;
    if !prefix[dash + 1..].chars().all(|ch| ch.is_ascii_digit()) {
        return None;
    }
    Some(prefix[..dash].to_string())
}

pub(super) fn file_preference_score(file: &str) -> usize {
    const PREFERRED: &[&str] = &[
        "Q4_K_M", "Q4_K_S", "Q4_1", "Q5_K_M", "Q5_K_S", "Q8_0", "BF16",
    ];
    let quant_rank = PREFERRED
        .iter()
        .position(|needle| file.contains(needle))
        .unwrap_or(PREFERRED.len() + 1);

    // Prefer single-file variants over split shards when both exist.
    // If only split files exist, still prefer shard 1.
    let split_penalty = if file.contains("-of-") { 100 } else { 0 };
    let split_part_penalty = split_part_number(file)
        .map(|part| if part <= 1 { 0 } else { 1_000 + part })
        .unwrap_or(0);

    quant_rank + split_penalty + split_part_penalty
}

fn split_part_number(file: &str) -> Option<usize> {
    let marker = file.find("-of-")?;
    let prefix = &file[..marker];
    let digits: String = prefix
        .chars()
        .rev()
        .take_while(|ch| ch.is_ascii_digit())
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    if digits.is_empty() {
        return None;
    }
    digits.parse::<usize>().ok()
}

fn is_primary_model_gguf(file: &str) -> bool {
    let lower = file.to_ascii_lowercase();
    lower.ends_with(".gguf") && !lower.contains("mmproj") && !lower.contains("imatrix")
}

fn is_imatrix_artifact(value: &str) -> bool {
    value.to_ascii_lowercase().contains("imatrix")
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
    if upper.contains("Q5_K_M") || upper.contains("Q5_K_S") {
        return 3;
    }
    if upper.contains("Q4_K_M") || upper.contains("Q4_K_S") {
        return 4;
    }
    if upper.contains("Q4_1") || upper.contains("Q4_0") {
        return 5;
    }
    if upper.contains("Q3_") {
        return 6;
    }
    if upper.contains("Q2_") || upper.contains("IQ") {
        return 7;
    }
    8
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

fn aggregate_variants_from_siblings(siblings: &[Siblings]) -> Vec<VariantCandidate> {
    #[derive(Clone, Debug)]
    struct VariantAggregate {
        stem: String,
        selected_file: String,
        selected_score: usize,
        total_size: u64,
        missing_size: bool,
    }

    let mut by_stem: BTreeMap<String, VariantAggregate> = BTreeMap::new();
    for sibling in siblings {
        let file = sibling.rfilename.as_str();
        if !is_primary_model_gguf(file) {
            continue;
        }
        let stem = canonical_hf_ref_file_component(file);
        let score = file_preference_score(file);
        let entry = by_stem.entry(stem.clone()).or_insert_with(|| VariantAggregate {
            stem: stem.clone(),
            selected_file: file.to_string(),
            selected_score: score,
            total_size: 0,
            missing_size: false,
        });
        if score < entry.selected_score {
            entry.selected_score = score;
            entry.selected_file = file.to_string();
        }
        if let Some(size) = sibling.size {
            entry.total_size = entry.total_size.saturating_add(size);
        } else {
            entry.missing_size = true;
        }
    }

    by_stem
        .into_values()
        .map(|value| VariantCandidate {
            stem: value.stem.clone(),
            file: value.selected_file,
            capabilities: capabilities::infer_filename_capabilities(&value.stem),
            size_bytes: if value.missing_size {
                None
            } else {
                Some(value.total_size)
            },
        })
        .collect()
}

async fn repo_info_with_blobs(api: &hf_hub::api::tokio::Api, repo: Repo) -> Result<RepoInfo> {
    let response = api
        .repo(repo)
        .info_request()
        .query(&[("blobs", "true")])
        .send()
        .await
        .context("Fetch Hugging Face repo metadata request")?
        .error_for_status()
        .context("Fetch Hugging Face repo metadata status")?;
    response
        .json::<RepoInfo>()
        .await
        .context("Decode Hugging Face repo metadata")
}

async fn resolve_repo_id_for_query(
    api: &hf_hub::api::tokio::Api,
    query: &str,
) -> Result<Option<String>> {
    if query.contains('/') && !query.contains(' ') {
        return Ok(Some(query.to_string()));
    }
    let repos = api
        .search(RepoType::Model)
        .with_query(query)
        .with_filter("gguf")
        .with_limit(100)
        .run()
        .await
        .context("Search Hugging Face for repo family")?;
    Ok(pick_repo_summary(query, &repos).map(|repo| repo.id.clone()))
}

fn parse_repo_query(input: &str) -> Option<String> {
    let trimmed = input.trim();
    if trimmed.is_empty()
        || trimmed.starts_with("http://")
        || trimmed.starts_with("https://")
        || trimmed.ends_with(".gguf")
    {
        return None;
    }
    let slash_count = trimmed.matches('/').count();
    if slash_count >= 2 {
        return None;
    }
    Some(trimmed.to_string())
}

fn capability_matches(profile: CapabilityProfile, caps: ModelCapabilities) -> bool {
    match profile {
        CapabilityProfile::Text => true,
        CapabilityProfile::Vision => caps.vision != capabilities::CapabilityLevel::None,
        CapabilityProfile::Audio => caps.audio != capabilities::CapabilityLevel::None,
        CapabilityProfile::Multimodal => caps.multimodal || caps.supports_multimodal_runtime(),
    }
}

fn capability_richness(caps: ModelCapabilities) -> usize {
    if caps.multimodal || caps.supports_multimodal_runtime() {
        3
    } else if caps.vision != capabilities::CapabilityLevel::None
        || caps.audio != capabilities::CapabilityLevel::None
    {
        2
    } else {
        1
    }
}

fn pick_repo_summary<'a>(query: &str, repos: &'a [RepoSummary]) -> Option<&'a RepoSummary> {
    let query = query.to_ascii_lowercase();
    let mut candidates: Vec<&RepoSummary> = repos
        .iter()
        .filter(|repo| {
            let id = repo.id.to_ascii_lowercase();
            id == query || id.ends_with(&format!("/{query}"))
        })
        .collect();
    if candidates.is_empty() {
        return None;
    }
    candidates.sort_by(|left, right| {
        let l_exact = left.id.eq_ignore_ascii_case(&query);
        let r_exact = right.id.eq_ignore_ascii_case(&query);
        r_exact
            .cmp(&l_exact)
            .then_with(|| right.downloads.unwrap_or(0).cmp(&left.downloads.unwrap_or(0)))
            .then_with(|| left.id.cmp(&right.id))
    });
    candidates.into_iter().next()
}

fn choose_variant(
    variants: &[VariantCandidate],
    profile: CapabilityProfile,
    vram_bytes: Option<u64>,
    prefer_fit: bool,
) -> Option<(VariantCandidate, Option<bool>)> {
    let mut eligible: Vec<VariantCandidate> = variants
        .iter()
        .filter(|variant| capability_matches(profile, variant.capabilities))
        .cloned()
        .collect();
    if eligible.is_empty() {
        return None;
    }

    let mut scored: Vec<(VariantCandidate, Option<bool>, usize, usize, usize, u64)> = eligible
        .drain(..)
        .map(|variant| {
            let fit = match (vram_bytes, variant.size_bytes) {
                (Some(vram), Some(size)) if vram > 0 => Some((size as f64 * 1.1) as u64 <= vram),
                _ => None,
            };
            (
                variant.clone(),
                fit,
                capability_richness(variant.capabilities),
                file_quality_score(&variant.file),
                file_preference_score(&variant.file),
                variant.size_bytes.unwrap_or(u64::MAX),
            )
        })
        .collect();
    let any_fit = scored.iter().any(|value| value.1 == Some(true));
    scored.sort_by(|left, right| {
        let by_fit = right.1.unwrap_or(false).cmp(&left.1.unwrap_or(false));
        let by_richness = right.2.cmp(&left.2);
        let by_quality = left.3.cmp(&right.3);
        let by_preference = left.4.cmp(&right.4);
        let by_size = left.5.cmp(&right.5);
        let by_file = left.0.file.cmp(&right.0.file);
        if prefer_fit {
            if any_fit {
                by_fit
                    .then_with(|| by_richness)
                    .then_with(|| by_quality)
                    .then_with(|| by_preference)
                    .then_with(|| by_size)
                    .then_with(|| by_file)
            } else {
                by_richness
                    .then_with(|| by_size)
                    .then_with(|| by_quality)
                    .then_with(|| by_preference)
                    .then_with(|| by_file)
            }
        } else {
            by_richness
                .then_with(|| by_quality)
                .then_with(|| by_preference)
                .then_with(|| by_size)
                .then_with(|| by_file)
        }
    });
    let (selected, fit, _, _, _, _) = scored.into_iter().next()?;
    Some((selected, fit))
}

async fn resolve_repo_or_family_ref(
    input: &str,
    profile: CapabilityProfile,
    announce: bool,
) -> Result<Option<RepoSelection>> {
    let Some(query) = parse_repo_query(input) else {
        return Ok(None);
    };
    let api = super::build_hf_tokio_api(false)?;
    let Some(repo_id) = resolve_repo_id_for_query(&api, &query).await? else {
        return Ok(None);
    };

    let info = repo_info_with_blobs(&api, Repo::new(repo_id.clone(), RepoType::Model))
        .await
        .with_context(|| format!("Fetch Hugging Face repo {}", repo_id))?;
    if info.gated == Some(true) {
        return Err(anyhow!(super::access::gated_access_message(&repo_id)));
    }
    let repo_id = info.id.clone().or(info.model_id.clone()).unwrap_or(repo_id);
    let variants = aggregate_variants_from_siblings(&info.siblings);
    if variants.is_empty() {
        return Ok(None);
    }
    let vram_bytes = {
        let detected = hardware::survey().vram_bytes;
        if detected > 0 { Some(detected) } else { None }
    };
    let Some((selected, fit)) = choose_variant(&variants, profile, vram_bytes, true) else {
        let label = profile.as_label();
        return Err(anyhow!(
            "🟡 No {label}-capable variants found in {repo_id}. Run 'mesh-llm models show {repo_id}' to inspect available variants."
        ));
    };
    let highest_quality = choose_variant(&variants, profile, None, false).map(|(value, _)| value);

    if announce {
        eprintln!("🔎 Resolving model ref: {query}");
        eprintln!("📦 Repo: {repo_id}");
        eprintln!("🧾 Found {} GGUF variants", variants.len());
        eprintln!("🎯 capability: {}", profile.as_label());
        if let Some(vram) = vram_bytes {
            eprintln!(
                "💻 Local capacity: {:.1} GB VRAM (fit threshold includes 10% headroom)",
                vram as f64 / 1e9
            );
        }
        eprintln!();
        if fit == Some(true) {
            eprintln!("✅ Picked variant that fits your machine:");
        } else {
            eprintln!("🏆 Picked highest quality variant:");
        }
        eprintln!("   🔗 {repo_id}/{}", selected.stem);
        eprintln!(
            "   ⚖️ quant: {}",
            selected
                .stem
                .split('-')
                .next_back()
                .unwrap_or(selected.stem.as_str())
        );
        if let Some(size) = selected.size_bytes {
            eprintln!("   📏 size: {}", format_size_bytes(size));
        }
        if let Some(fit) = fit {
            eprintln!("   💻 fit: {}", if fit { "✅" } else { "❌" });
        }
        if let Some(best) = highest_quality {
            eprintln!();
            eprintln!("🏆 Highest quality:");
            eprintln!("   🔗 {repo_id}/{}", best.stem);
        }
        eprintln!();
    }

    Ok(Some(RepoSelection {
        repo: repo_id,
        selected,
        fit,
    }))
}

pub async fn inspect_repo_ref(input: &str, profile: CapabilityProfile) -> Result<Option<RepoInspection>> {
    let Some(query) = parse_repo_query(input) else {
        return Ok(None);
    };

    let api = super::build_hf_tokio_api(false)?;
    let Some(repo_id) = resolve_repo_id_for_query(&api, &query).await? else {
        return Ok(None);
    };

    let info = repo_info_with_blobs(&api, Repo::new(repo_id.clone(), RepoType::Model))
        .await
        .with_context(|| format!("Fetch Hugging Face repo {}", repo_id))?;
    if info.gated == Some(true) {
        return Err(anyhow!(super::access::gated_access_message(&repo_id)));
    }

    let repo_id = info.id.clone().or(info.model_id.clone()).unwrap_or(repo_id);
    let variants = aggregate_variants_from_siblings(&info.siblings);
    if variants.is_empty() {
        return Ok(None);
    }

    let vram_bytes = {
        let detected = hardware::survey().vram_bytes;
        if detected > 0 { Some(detected) } else { None }
    };
    let Some((recommended, fit)) = choose_variant(&variants, profile, vram_bytes, true) else {
        let label = profile.as_label();
        return Err(anyhow!(
            "🟡 No {label}-capable variants found in {repo_id}. Run 'mesh-llm models show {repo_id}' to inspect available variants."
        ));
    };
    let Some((highest, _)) = choose_variant(&variants, profile, None, false) else {
        return Ok(None);
    };

    let mut table_variants: Vec<RepoVariantInspection> = variants
        .iter()
        .map(|variant| RepoVariantInspection {
            reference: format!("{repo_id}/{}", variant.stem),
            quant: variant
                .stem
                .split('-')
                .next_back()
                .unwrap_or(variant.stem.as_str())
                .to_string(),
            capability: if variant.capabilities.multimodal
                || variant.capabilities.supports_multimodal_runtime()
            {
                "multimodal".to_string()
            } else if variant.capabilities.vision != capabilities::CapabilityLevel::None {
                "vision".to_string()
            } else if variant.capabilities.audio != capabilities::CapabilityLevel::None {
                "audio".to_string()
            } else {
                "text".to_string()
            },
            size_bytes: variant.size_bytes,
            fit: match (vram_bytes, variant.size_bytes) {
                (Some(vram), Some(size)) if vram > 0 => Some((size as f64 * 1.1) as u64 <= vram),
                _ => None,
            },
        })
        .collect();
    table_variants.sort_by(|left, right| {
        left.size_bytes
            .unwrap_or(u64::MAX)
            .cmp(&right.size_bytes.unwrap_or(u64::MAX))
            .then_with(|| left.reference.cmp(&right.reference))
    });

    Ok(Some(RepoInspection {
        repo: repo_id.clone(),
        description: info.description.clone(),
        downloads: info.downloads,
        likes: info.likes,
        recommended_ref: format!("{repo_id}/{}", recommended.stem),
        highest_quality_ref: format!("{repo_id}/{}", highest.stem),
        recommended_fit: fit,
        recommended_size_bytes: recommended.size_bytes,
        highest_quality_size_bytes: highest.size_bytes,
        variant_count: variants.len(),
        variants: table_variants,
    }))
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

#[cfg(test)]
mod tests {
    use super::capability_matches;
    use super::choose_variant;
    use super::aggregate_variants_from_siblings;
    use super::CapabilityProfile;
    use super::VariantCandidate;
    use super::choose_hf_file_for_selector;
    use super::canonical_hf_ref_file_component;
    use super::file_preference_score;
    use super::is_imatrix_artifact;
    use super::normalize_selector_key;
    use super::resolve_model_spec;
    use crate::models::CapabilityLevel;
    use crate::models::ModelCapabilities;
    use std::path::Path;
    use hf_hub::api::Siblings;

    #[test]
    fn file_preference_prefers_single_file_over_split() {
        let single = "MiniMax-M2.5-Q4_K_M.gguf";
        let split = "Q4_K_M/MiniMax-M2.5-Q4_K_M-00001-of-00004.gguf";
        assert!(file_preference_score(single) < file_preference_score(split));
    }

    #[test]
    fn file_preference_prefers_first_split_part() {
        let first = "Q4_K_M/MiniMax-M2.5-Q4_K_M-00001-of-00004.gguf";
        let second = "Q4_K_M/MiniMax-M2.5-Q4_K_M-00002-of-00004.gguf";
        assert!(file_preference_score(first) < file_preference_score(second));
    }

    #[test]
    fn canonical_ref_component_collapses_split_suffix() {
        assert_eq!(
            canonical_hf_ref_file_component("MiniMax-M2-HQ4_K-00001-of-00004.gguf"),
            "MiniMax-M2-HQ4_K".to_string()
        );
        assert_eq!(
            canonical_hf_ref_file_component("MiniMax-M2-Q4_K_M.gguf"),
            "MiniMax-M2-Q4_K_M".to_string()
        );
    }

    #[test]
    fn selector_chooses_first_split_part_for_stem() {
        let selector = normalize_selector_key("MiniMax-M2-HQ4_K");
        let files = vec![
            "MiniMax-M2-HQ4_K-00003-of-00004.gguf".to_string(),
            "MiniMax-M2-HQ4_K-00001-of-00004.gguf".to_string(),
            "MiniMax-M2-HQ4_K-00002-of-00004.gguf".to_string(),
        ];
        let chosen = choose_hf_file_for_selector(&selector, &files).unwrap();
        assert_eq!(chosen, "MiniMax-M2-HQ4_K-00001-of-00004.gguf");
    }

    #[test]
    fn capability_profile_matches_expected_signals() {
        let caps = ModelCapabilities {
            multimodal: true,
            vision: CapabilityLevel::Supported,
            audio: CapabilityLevel::Likely,
            ..ModelCapabilities::default()
        };
        assert!(capability_matches(CapabilityProfile::Text, caps));
        assert!(capability_matches(CapabilityProfile::Vision, caps));
        assert!(capability_matches(CapabilityProfile::Audio, caps));
        assert!(capability_matches(CapabilityProfile::Multimodal, caps));
    }

    #[test]
    fn choose_variant_prefers_fitting_candidate() {
        let text_caps = ModelCapabilities::default();
        let variants = vec![
            VariantCandidate {
                stem: "MiniMax-M2-Q4_K_M".to_string(),
                file: "Q4_K_M/MiniMax-M2-Q4_K_M-00001-of-00003.gguf".to_string(),
                capabilities: text_caps,
                size_bytes: Some(120_000_000_000),
            },
            VariantCandidate {
                stem: "MiniMax-M2-Q2_K_L".to_string(),
                file: "Q2_K_L/MiniMax-M2-Q2_K_L-00001-of-00002.gguf".to_string(),
                capabilities: text_caps,
                size_bytes: Some(70_000_000_000),
            },
        ];

        let (picked, fit) = choose_variant(
            &variants,
            CapabilityProfile::Text,
            Some(96_000_000_000),
            true,
        )
            .expect("should pick one variant");
        assert_eq!(picked.stem, "MiniMax-M2-Q2_K_L");
        assert_eq!(fit, Some(true));
    }

    #[test]
    fn choose_variant_prefers_richer_capability_within_fit() {
        let text_caps = ModelCapabilities::default();
        let vision_caps = ModelCapabilities {
            multimodal: true,
            vision: CapabilityLevel::Supported,
            ..ModelCapabilities::default()
        };
        let variants = vec![
            VariantCandidate {
                stem: "Model-text-Q4_K_M".to_string(),
                file: "Model-text-Q4_K_M.gguf".to_string(),
                capabilities: text_caps,
                size_bytes: Some(20_000_000_000),
            },
            VariantCandidate {
                stem: "Model-vision-Q4_K_M".to_string(),
                file: "Model-vision-Q4_K_M.gguf".to_string(),
                capabilities: vision_caps,
                size_bytes: Some(25_000_000_000),
            },
        ];
        let (picked, fit) = choose_variant(
            &variants,
            CapabilityProfile::Text,
            Some(96_000_000_000),
            true,
        )
            .expect("should pick one variant");
        assert_eq!(picked.stem, "Model-vision-Q4_K_M");
        assert_eq!(fit, Some(true));
    }

    #[test]
    fn aggregate_variants_sums_split_sizes_per_stem() {
        let siblings = vec![
            Siblings {
                rfilename: "Q4/model-Q4_K_M-00001-of-00003.gguf".to_string(),
                size: Some(10),
            },
            Siblings {
                rfilename: "Q4/model-Q4_K_M-00002-of-00003.gguf".to_string(),
                size: Some(20),
            },
            Siblings {
                rfilename: "Q4/model-Q4_K_M-00003-of-00003.gguf".to_string(),
                size: Some(30),
            },
        ];
        let variants = aggregate_variants_from_siblings(&siblings);
        assert_eq!(variants.len(), 1);
        assert_eq!(variants[0].stem, "Q4/model-Q4_K_M");
        assert_eq!(variants[0].size_bytes, Some(60));
    }

    #[test]
    fn choose_variant_prefers_smallest_when_none_fit() {
        let text_caps = ModelCapabilities::default();
        let variants = vec![
            VariantCandidate {
                stem: "Model-Q8_0".to_string(),
                file: "Model-Q8_0.gguf".to_string(),
                capabilities: text_caps,
                size_bytes: Some(200_000_000_000),
            },
            VariantCandidate {
                stem: "Model-IQ2_XXS".to_string(),
                file: "Model-IQ2_XXS.gguf".to_string(),
                capabilities: text_caps,
                size_bytes: Some(80_000_000_000),
            },
        ];
        let (picked, fit) = choose_variant(&variants, CapabilityProfile::Text, Some(64_000_000_000), true)
            .expect("should pick one variant");
        assert_eq!(picked.stem, "Model-IQ2_XXS");
        assert_eq!(fit, Some(false));
    }

    #[test]
    fn aggregate_variants_ignores_imatrix_artifacts() {
        let siblings = vec![
            Siblings {
                rfilename: "MiniMax-M2-BF16.imatrix.gguf".to_string(),
                size: Some(492_000_000),
            },
            Siblings {
                rfilename: "MiniMax-M2-BF16.i1-IQ1_S.gguf".to_string(),
                size: Some(46_500_000_000),
            },
        ];
        let variants = aggregate_variants_from_siblings(&siblings);
        assert_eq!(variants.len(), 1);
        assert_eq!(variants[0].stem, "MiniMax-M2-BF16.i1-IQ1_S");
        assert_eq!(variants[0].size_bytes, Some(46_500_000_000));
    }

    #[test]
    fn imatrix_detector_flags_expected_values() {
        assert!(is_imatrix_artifact("MiniMax-M2-BF16.imatrix.gguf"));
        assert!(is_imatrix_artifact("org/repo/MiniMax-M2-BF16.imatrix"));
        assert!(!is_imatrix_artifact("MiniMax-M2-BF16.i1-IQ1_S.gguf"));
    }

    #[tokio::test]
    async fn resolve_model_spec_rejects_imatrix_refs() {
        let err = resolve_model_spec(Path::new("org/repo/MiniMax-M2-BF16.imatrix.gguf"))
            .await
            .expect_err("imatrix refs should be rejected");
        assert!(err.to_string().contains("imatrix artifact"));
    }
}
