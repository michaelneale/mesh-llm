use super::ModelCapabilities;
use super::{build_hf_tokio_api, capabilities, catalog, find_model_path, format_size_bytes};
use anyhow::{anyhow, bail, Context, Result};
use hf_hub::{Repo, RepoType};
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

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ResolveFormatPreference {
    Auto,
    Gguf,
    Mlx,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum MlxSelectionPolicy {
    AllowImplicit,
    RequireExplicitFlag,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(super) enum RepoArtifactKind {
    Gguf,
    Mlx,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct RepoArtifactCandidate {
    pub kind: RepoArtifactKind,
    pub file: String,
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

pub async fn download_exact_ref(
    input: &str,
    preference: ResolveFormatPreference,
    command_prefix: &str,
    mlx_policy: MlxSelectionPolicy,
) -> Result<PathBuf> {
    match parse_exact_model_ref(input, preference, command_prefix, mlx_policy).await? {
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

    download_exact_ref(
        &raw,
        ResolveFormatPreference::Auto,
        "mesh-llm models download",
        MlxSelectionPolicy::AllowImplicit,
    )
    .await
    .with_context(|| format!("Resolve model spec {raw}"))
}

pub async fn show_exact_model(input: &str) -> Result<ModelDetails> {
    match parse_exact_model_ref(
        input,
        ResolveFormatPreference::Auto,
        "mesh-llm models show",
        MlxSelectionPolicy::AllowImplicit,
    )
    .await?
    {
        ExactModelRef::Catalog(model) => Ok(ModelDetails {
            display_name: model.name.to_string(),
            exact_ref: model.name.to_string(),
            source: "catalog",
            kind: catalog_model_kind_label(model),
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
                kind: artifact_kind_label_for_file(&file),
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
                kind: artifact_kind_label_for_file(&url),
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
mod repo_artifact_tests {
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
    if !is_supported_huggingface_file_ref(input) {
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

pub(super) fn parse_huggingface_repo_ref(input: &str) -> Option<(String, Option<String>)> {
    if input.starts_with("http://") || input.starts_with("https://") {
        return None;
    }
    if input.contains("/resolve/") || is_supported_huggingface_file_ref(input) {
        return None;
    }
    let parts: Vec<&str> = input.split('/').collect();
    if parts.len() != 2 {
        return None;
    }
    let (repo_tail, revision) = match parts[1].split_once('@') {
        Some((repo, revision)) if !repo.is_empty() && !revision.is_empty() => {
            (repo, Some(revision.to_string()))
        }
        Some(_) => return None,
        None => (parts[1], None),
    };
    if parts[0].is_empty() || repo_tail.is_empty() {
        return None;
    }
    Some((format!("{}/{}", parts[0], repo_tail), revision))
}

fn is_supported_huggingface_file_ref(input: &str) -> bool {
    input.ends_with(".gguf")
        || input.ends_with("model.safetensors")
        || input.ends_with("model.safetensors.index.json")
}

pub(super) fn artifact_kind_for_file(file: &str) -> Option<RepoArtifactKind> {
    if file.ends_with(".gguf") {
        return Some(RepoArtifactKind::Gguf);
    }
    if file.ends_with("model.safetensors") || file.ends_with("model.safetensors.index.json") {
        return Some(RepoArtifactKind::Mlx);
    }
    None
}

fn validate_preference_matches_file(file: &str, preference: ResolveFormatPreference) -> Result<()> {
    match preference {
        ResolveFormatPreference::Auto => Ok(()),
        ResolveFormatPreference::Gguf => {
            if artifact_kind_for_file(file) == Some(RepoArtifactKind::Gguf) {
                Ok(())
            } else {
                bail!("`--gguf` requires a GGUF artifact");
            }
        }
        ResolveFormatPreference::Mlx => {
            if artifact_kind_for_file(file) == Some(RepoArtifactKind::Mlx) {
                Ok(())
            } else {
                bail!("`--mlx` requires an MLX artifact");
            }
        }
    }
}

fn ensure_explicit_mlx_selection(
    kind: Option<RepoArtifactKind>,
    preference: ResolveFormatPreference,
    mlx_policy: MlxSelectionPolicy,
    command_prefix: &str,
    target: &str,
) -> Result<()> {
    if kind == Some(RepoArtifactKind::Mlx)
        && mlx_policy == MlxSelectionPolicy::RequireExplicitFlag
        && preference != ResolveFormatPreference::Mlx
    {
        bail!(
            "MLX selection requires explicit `--mlx`.\nRetry with:\n  {command_prefix} {target} --mlx"
        );
    }
    Ok(())
}

pub(super) fn collect_repo_artifact_candidates(siblings: &[String]) -> Vec<RepoArtifactCandidate> {
    let mut gguf = Vec::new();
    let mut mlx = Vec::new();
    for sibling in siblings {
        if sibling.ends_with(".gguf") {
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
        right
            .file
            .ends_with("model.safetensors.index.json")
            .cmp(&left.file.ends_with("model.safetensors.index.json"))
            .then_with(|| left.file.cmp(&right.file))
    });
    if mlx
        .iter()
        .any(|candidate| candidate.file.ends_with("model.safetensors.index.json"))
    {
        mlx.retain(|candidate| candidate.file.ends_with("model.safetensors.index.json"));
    }
    gguf.extend(mlx);
    gguf
}

pub(super) fn artifact_kind_label(kind: RepoArtifactKind) -> &'static str {
    match kind {
        RepoArtifactKind::Gguf => "🦙 gguf",
        RepoArtifactKind::Mlx => "🍎 mlx",
    }
}

pub(super) fn artifact_kind_label_for_file(file: &str) -> &'static str {
    artifact_kind_for_file(file)
        .map(artifact_kind_label)
        .unwrap_or("unknown")
}

pub fn catalog_model_kind_label(model: &catalog::CatalogModel) -> &'static str {
    if model
        .source_file()
        .map(|file| {
            file.ends_with("model.safetensors") || file.ends_with("model.safetensors.index.json")
        })
        .unwrap_or(false)
        || model.url.contains("model.safetensors")
    {
        "🍎 mlx"
    } else {
        "🦙 gguf"
    }
}

fn format_repo_artifact_suggestions(
    repo: &str,
    revision: Option<&str>,
    candidates: &[RepoArtifactCandidate],
    command_prefix: &str,
) -> String {
    let mut lines = vec![format!("🤔 Multiple artifacts found in {repo}")];
    if let Some(revision) = revision {
        lines[0].push('@');
        lines[0].push_str(revision);
    }
    let gguf: Vec<_> = candidates
        .iter()
        .filter(|candidate| candidate.kind == RepoArtifactKind::Gguf)
        .collect();
    let mlx: Vec<_> = candidates
        .iter()
        .filter(|candidate| candidate.kind == RepoArtifactKind::Mlx)
        .collect();
    if !gguf.is_empty() {
        lines.push(String::new());
        lines.push("🦙 GGUF:".to_string());
        for candidate in gguf {
            lines.push(format!(
                "  {command_prefix} {}",
                format_huggingface_exact_ref(repo, revision, &candidate.file)
            ));
        }
    }
    if !mlx.is_empty() {
        lines.push(String::new());
        lines.push("🍎 MLX:".to_string());
        for candidate in mlx {
            lines.push(format!(
                "  {command_prefix} {}",
                format_huggingface_exact_ref(repo, revision, &candidate.file)
            ));
        }
    }
    lines.join("\n")
}

fn choose_repo_artifact_candidate(
    repo: &str,
    revision: Option<&str>,
    candidates: &[RepoArtifactCandidate],
    preference: ResolveFormatPreference,
    command_prefix: &str,
    mlx_policy: MlxSelectionPolicy,
) -> Result<RepoArtifactCandidate> {
    let filtered: Vec<_> = candidates
        .iter()
        .filter(|candidate| match preference {
            ResolveFormatPreference::Auto => true,
            ResolveFormatPreference::Gguf => candidate.kind == RepoArtifactKind::Gguf,
            ResolveFormatPreference::Mlx => candidate.kind == RepoArtifactKind::Mlx,
        })
        .cloned()
        .collect();

    match filtered.len() {
        0 => match preference {
            ResolveFormatPreference::Auto => {
                bail!("No downloadable model artifacts found in {repo}")
            }
            ResolveFormatPreference::Gguf => bail!("No GGUF artifacts found in {repo}"),
            ResolveFormatPreference::Mlx => bail!("No MLX artifacts found in {repo}"),
        },
        1 => {
            let candidate = filtered[0].clone();
            ensure_explicit_mlx_selection(
                Some(candidate.kind),
                preference,
                mlx_policy,
                command_prefix,
                &repo.to_string(),
            )?;
            Ok(candidate)
        }
        _ => bail!(
            "{}",
            format_repo_artifact_suggestions(repo, revision, &filtered, command_prefix)
        ),
    }
}

async fn resolve_repo_artifact_ref(
    repo: &str,
    revision: Option<&str>,
    preference: ResolveFormatPreference,
    command_prefix: &str,
    mlx_policy: MlxSelectionPolicy,
) -> Result<(String, Option<String>, String)> {
    let api = build_hf_tokio_api(false)?;
    let repo_handle = match revision {
        Some(rev) => Repo::with_revision(repo.to_string(), RepoType::Model, rev.to_string()),
        None => Repo::new(repo.to_string(), RepoType::Model),
    };
    let detail = api
        .repo(repo_handle)
        .info()
        .await
        .map_err(|err| anyhow!("Fetch Hugging Face repo {repo}: {err}"))?;
    let siblings: Vec<String> = detail
        .siblings
        .into_iter()
        .map(|sibling| sibling.rfilename)
        .collect();
    let candidates = collect_repo_artifact_candidates(&siblings);
    let choice = choose_repo_artifact_candidate(
        repo,
        revision,
        &candidates,
        preference,
        command_prefix,
        mlx_policy,
    )?;
    Ok((repo.to_string(), revision.map(str::to_string), choice.file))
}

async fn parse_exact_model_ref(
    input: &str,
    preference: ResolveFormatPreference,
    command_prefix: &str,
    mlx_policy: MlxSelectionPolicy,
) -> Result<ExactModelRef> {
    if let Some(model) = find_catalog_model_exact(input) {
        ensure_explicit_mlx_selection(
            artifact_kind_for_file(&model.file),
            preference,
            mlx_policy,
            command_prefix,
            input,
        )?;
        return Ok(ExactModelRef::Catalog(model));
    }
    if let Some((repo, revision, file)) = parse_huggingface_ref(input) {
        validate_preference_matches_file(&file, preference)?;
        ensure_explicit_mlx_selection(
            artifact_kind_for_file(&file),
            preference,
            mlx_policy,
            command_prefix,
            input,
        )?;
        return Ok(ExactModelRef::HuggingFace {
            repo,
            revision,
            file,
        });
    }
    if let Some((repo, revision)) = parse_huggingface_repo_ref(input) {
        let (repo, revision, file) = resolve_repo_artifact_ref(
            &repo,
            revision.as_deref(),
            preference,
            command_prefix,
            mlx_policy,
        )
        .await?;
        return Ok(ExactModelRef::HuggingFace {
            repo,
            revision,
            file,
        });
    }
    if input.starts_with("http://") || input.starts_with("https://") {
        let filename = remote_filename(input)?;
        validate_preference_matches_file(&filename, preference)?;
        ensure_explicit_mlx_selection(
            artifact_kind_for_file(&filename),
            preference,
            mlx_policy,
            command_prefix,
            input,
        )?;
        return Ok(ExactModelRef::Url {
            url: input.to_string(),
            filename,
        });
    }
    bail!(
        "Expected a model ref. Use a catalog id, a Hugging Face ref like org/repo/file.gguf, org/repo/model.safetensors, or org/repo, or a direct URL."
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_huggingface_repo_ref_accepts_repo_and_revision_shorthand() {
        let (repo, revision) =
            parse_huggingface_repo_ref("mlx-community/Qwen2.5-0.5B-Instruct@main").unwrap();
        assert_eq!(repo, "mlx-community/Qwen2.5-0.5B-Instruct");
        assert_eq!(revision.as_deref(), Some("main"));
    }

    #[test]
    fn collect_repo_artifact_candidates_prefers_mlx_index_and_split_gguf_primary() {
        let siblings = vec![
            "Qwen3-8B-Q4_K_M-00002-of-00004.gguf".to_string(),
            "Qwen3-8B-Q4_K_M-00001-of-00004.gguf".to_string(),
            "Qwen3-8B-Q6_K.gguf".to_string(),
            "model.safetensors".to_string(),
            "model.safetensors.index.json".to_string(),
        ];
        let candidates = collect_repo_artifact_candidates(&siblings);
        let files: Vec<_> = candidates
            .into_iter()
            .map(|candidate| candidate.file)
            .collect();
        assert_eq!(
            files,
            vec![
                "Qwen3-8B-Q4_K_M-00001-of-00004.gguf".to_string(),
                "Qwen3-8B-Q6_K.gguf".to_string(),
                "model.safetensors.index.json".to_string(),
            ]
        );
    }

    #[test]
    fn format_repo_artifact_suggestions_groups_by_backend() {
        let text = format_repo_artifact_suggestions(
            "some-org/repo",
            None,
            &[
                RepoArtifactCandidate {
                    kind: RepoArtifactKind::Gguf,
                    file: "Qwen3-8B-Q4_K_M.gguf".to_string(),
                },
                RepoArtifactCandidate {
                    kind: RepoArtifactKind::Mlx,
                    file: "model.safetensors".to_string(),
                },
            ],
            "mesh-llm models download",
        );
        assert!(text.contains("🤔 Multiple artifacts found in some-org/repo"));
        assert!(text.contains("🦙 GGUF:"));
        assert!(text.contains("🍎 MLX:"));
        assert!(text.contains("mesh-llm models download some-org/repo/Qwen3-8B-Q4_K_M.gguf"));
        assert!(text.contains("mesh-llm models download some-org/repo/model.safetensors"));
    }

    #[test]
    fn explicit_mlx_flag_is_required_for_exact_mlx_artifacts_when_requested() {
        let err = ensure_explicit_mlx_selection(
            Some(RepoArtifactKind::Mlx),
            ResolveFormatPreference::Auto,
            MlxSelectionPolicy::RequireExplicitFlag,
            "mesh-llm --model",
            "mlx-community/Qwen3-0.6B-4bit/model.safetensors.index.json",
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("MLX selection requires explicit `--mlx`"));
        assert!(err.contains(
            "mesh-llm --model mlx-community/Qwen3-0.6B-4bit/model.safetensors.index.json --mlx"
        ));
    }

    #[test]
    fn explicit_mlx_flag_allows_exact_mlx_artifacts() {
        ensure_explicit_mlx_selection(
            Some(RepoArtifactKind::Mlx),
            ResolveFormatPreference::Mlx,
            MlxSelectionPolicy::RequireExplicitFlag,
            "mesh-llm --model",
            "mlx-community/Qwen3-0.6B-4bit/model.safetensors.index.json",
        )
        .unwrap();
    }

    #[test]
    fn choose_repo_artifact_candidate_requires_mlx_flag_for_mlx_only_repo() {
        let err = choose_repo_artifact_candidate(
            "mlx-community/Qwen3-0.6B-4bit",
            None,
            &[RepoArtifactCandidate {
                kind: RepoArtifactKind::Mlx,
                file: "model.safetensors.index.json".to_string(),
            }],
            ResolveFormatPreference::Auto,
            "mesh-llm --model",
            MlxSelectionPolicy::RequireExplicitFlag,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("MLX selection requires explicit `--mlx`"));
    }
}
