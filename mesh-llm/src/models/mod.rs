pub mod capabilities;
pub mod catalog;
pub mod cli;
pub mod inventory;
pub mod local;
pub mod topology;

use anyhow::{anyhow, bail, Context, Result};
use hf_hub::api::sync::{Api, ApiBuilder};
use hf_hub::api::RepoInfo;
use hf_hub::{Repo, RepoType};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

pub use capabilities::{CapabilityLevel, ModelCapabilities};
pub use cli::{
    print_legacy_storage_warning, run_model_download, run_model_installed, run_model_recommended,
    run_model_search, run_model_show, warn_about_legacy_model_usage,
};
pub use inventory::{
    scan_all_model_metadata, scan_local_inventory_snapshot, scan_local_model_sizes,
};
pub use local::{
    exact_model_source_for_path, find_model_path, huggingface_hub_cache, huggingface_hub_cache_dir,
    huggingface_identity_for_path, legacy_models_dir, legacy_models_present, model_dirs,
    path_is_in_legacy_models_dir, scan_installed_models, scan_local_models,
};
pub use topology::{infer_local_model_topology, ModelMoeInfo, ModelTopology};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MigrationStatus {
    Rehydratable,
    LegacyOnly,
}

struct MigrationEntry {
    path: PathBuf,
    status: MigrationStatus,
    detail: String,
    catalog: Option<&'static catalog::CatalogModel>,
}

impl MigrationEntry {
    fn file_name(&self) -> String {
        self.path
            .file_name()
            .and_then(|value| value.to_str())
            .map(str::to_string)
            .unwrap_or_else(|| self.path.display().to_string())
    }
}

struct CachedRepo {
    repo_id: String,
    ref_name: String,
    local_revision: String,
}

#[derive(Default)]
struct MigrationCounts {
    adopted: usize,
    downloaded: usize,
}

#[derive(Default)]
struct UpdateCounts {
    refreshed: usize,
    missing_meta: usize,
}

#[derive(Clone, Debug)]
pub struct SearchHit {
    pub repo_id: String,
    pub file: String,
    pub exact_ref: String,
    pub size_label: Option<String>,
    pub downloads: Option<u64>,
    pub likes: Option<u64>,
    pub catalog: Option<&'static catalog::CatalogModel>,
    pub capabilities: ModelCapabilities,
}

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

#[derive(Debug, Deserialize)]
struct HuggingFaceRepoSummary {
    id: String,
    downloads: Option<u64>,
    likes: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct HuggingFaceRepoDetail {
    #[serde(default)]
    id: Option<String>,
    #[serde(default, rename = "modelId")]
    model_id: Option<String>,
    #[serde(default)]
    siblings: Vec<HuggingFaceSibling>,
}

#[derive(Debug, Deserialize)]
struct HuggingFaceSibling {
    rfilename: String,
}

fn merge_capabilities(left: ModelCapabilities, right: ModelCapabilities) -> ModelCapabilities {
    ModelCapabilities {
        vision: left.vision.max(right.vision),
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

pub async fn download_exact_ref(input: &str) -> Result<PathBuf> {
    match parse_exact_model_ref(input)? {
        ExactModelRef::Catalog(model) => catalog::download_model(model).await,
        ExactModelRef::HuggingFace {
            repo,
            revision,
            file,
        } => catalog::download_hf_repo_file(&repo, revision.as_deref(), &file).await,
        ExactModelRef::Url { url, filename } => {
            let dest = catalog::models_dir().join(&filename);
            if existing_download(&dest).await {
                return Ok(dest);
            }
            eprintln!("📥 Downloading {}...", dest.display());
            catalog::download_url(&url, &dest).await?;
            Ok(dest)
        }
    }
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

// Keep search custom for now. `hf-hub` handles cache and file transport well,
// but it does not expose a Hub search surface in this crate version.
pub async fn search_huggingface(query: &str, limit: usize) -> Result<Vec<SearchHit>> {
    let repo_limit = limit.clamp(1, 100);
    let client = http_client()?;
    let mut request = client.get("https://huggingface.co/api/models").query(&[
        ("search", query),
        ("filter", "gguf"),
        ("limit", &repo_limit.to_string()),
    ]);
    if let Some(token) = hf_token_override() {
        request = request.bearer_auth(token);
    }
    let repos: Vec<HuggingFaceRepoSummary> = request
        .send()
        .await
        .context("Search Hugging Face")?
        .error_for_status()
        .context("Hugging Face search failed")?
        .json()
        .await
        .context("Parse Hugging Face search response")?;

    let mut hits = Vec::new();
    for repo in repos {
        let mut detail_request =
            client.get(format!("https://huggingface.co/api/models/{}", repo.id));
        if let Some(token) = hf_token_override() {
            detail_request = detail_request.bearer_auth(token);
        }
        let detail: HuggingFaceRepoDetail = detail_request
            .send()
            .await
            .with_context(|| format!("Fetch Hugging Face repo {}", repo.id))?
            .error_for_status()
            .with_context(|| format!("Hugging Face repo {} returned an error", repo.id))?
            .json()
            .await
            .with_context(|| format!("Parse Hugging Face repo {}", repo.id))?;

        let repo_id = detail.id.or(detail.model_id).unwrap_or(repo.id.clone());
        let sibling_names: Vec<String> = detail
            .siblings
            .iter()
            .map(|sibling| sibling.rfilename.clone())
            .collect();
        let mut files: Vec<String> = detail
            .siblings
            .into_iter()
            .map(|sibling| sibling.rfilename)
            .filter(|file| file.ends_with(".gguf"))
            .collect();
        if files.is_empty() {
            continue;
        }
        files.sort_by(|left, right| {
            file_preference_score(left)
                .cmp(&file_preference_score(right))
                .then_with(|| left.cmp(right))
        });
        if let Some(file) = files.into_iter().next() {
            let catalog = matching_catalog_model_for_huggingface(&repo_id, None, &file);
            let download_url = huggingface_resolve_url(&repo_id, None, &file);
            let size_label = match catalog {
                Some(model) => Some(model.size.to_string()),
                None => remote_size_label(&download_url).await,
            };
            let remote_caps = capabilities::infer_remote_hf_capabilities(
                &repo_id,
                None,
                &file,
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
                file: file.clone(),
                exact_ref: format!("{repo_id}/{file}"),
                size_label,
                downloads: repo.downloads,
                likes: repo.likes,
                catalog,
                capabilities,
            });
            if hits.len() >= limit {
                return Ok(hits);
            }
        }
    }
    Ok(hits)
}

pub fn run_migrate(apply: bool, prune: bool) -> Result<()> {
    let entries = migration_entries();
    let legacy_dir = legacy_models_dir();
    if entries.is_empty() {
        eprintln!("📦 No legacy GGUF files found");
        eprintln!("   {}", legacy_dir.display());
        return Ok(());
    }

    eprintln!("🧳 Legacy model scan");
    eprintln!("📁 Source: {}", legacy_dir.display());
    eprintln!();
    for entry in &entries {
        let (label, icon) = match entry.status {
            MigrationStatus::Rehydratable => ("Rehydratable", "✅"),
            MigrationStatus::LegacyOnly => ("Legacy-only", "⚠️"),
        };
        eprintln!("{icon} {label}: {}", entry.file_name());
        eprintln!("   path: {}", entry.path.display());
        eprintln!("   info: {}", entry.detail);
    }
    eprintln!();

    let rehydratable = entries
        .iter()
        .filter(|entry| entry.status == MigrationStatus::Rehydratable)
        .count();
    let legacy_only = entries.len() - rehydratable;
    eprintln!("📊 Summary");
    eprintln!("   ✅ rehydratable: {rehydratable}");
    eprintln!("   ⚠️ legacy-only: {legacy_only}");

    if !apply && !prune {
        eprintln!();
        eprintln!("➡️ Next steps");
        if rehydratable > 0 {
            eprintln!("   mesh-llm models migrate --apply");
            eprintln!(
                "   Rehydrate recognized Hugging Face-backed models into {}",
                huggingface_hub_cache_dir().display()
            );
            eprintln!("   mesh-llm models migrate --prune");
            eprintln!("   Remove rehydrated legacy GGUF files after you verify the HF cache copy");
        }
        if legacy_only > 0 {
            eprintln!("   mesh-llm --gguf /path/to/model.gguf");
            eprintln!("   Keep using custom local GGUF files explicitly");
        }
        return Ok(());
    }

    if prune && !apply {
        return run_prune(&entries);
    }

    let api = build_hf_api(true)?;
    let mut migrated = 0usize;
    let mut totals = MigrationCounts::default();
    let mut grouped = BTreeMap::<String, Vec<&MigrationEntry>>::new();
    eprintln!("🚚 Migrating recognized models");
    eprintln!("📁 Destination: {}", huggingface_hub_cache_dir().display());
    eprintln!();
    for entry in entries
        .iter()
        .filter(|entry| entry.status == MigrationStatus::Rehydratable)
    {
        let Some(model) = entry.catalog else {
            continue;
        };
        grouped.entry(model.name.clone()).or_default().push(entry);
    }
    let total_groups = grouped.len();
    for (index, grouped_entries) in grouped.values().enumerate() {
        let Some(model) = grouped_entries.first().and_then(|entry| entry.catalog) else {
            continue;
        };
        eprintln!("🧭 [{}/{}] {}", index + 1, total_groups, model.name);
        let counts = migrate_catalog_model(&api, model, grouped_entries)?;
        totals.adopted += counts.adopted;
        totals.downloaded += counts.downloaded;
        migrated += 1;
    }

    eprintln!();
    eprintln!("✅ Migration complete");
    eprintln!("   model groups migrated: {migrated}");
    eprintln!("   adopted local files: {}", totals.adopted);
    eprintln!("   downloaded files: {}", totals.downloaded);
    eprintln!("   destination: {}", huggingface_hub_cache_dir().display());
    eprintln!("   legacy files were left in place");
    eprintln!("   next: mesh-llm models migrate --prune");
    eprintln!("   custom local GGUFs still work via `mesh-llm --gguf /path/to/model.gguf`");
    Ok(())
}

pub fn run_update(repo: Option<&str>, all: bool, check: bool) -> Result<()> {
    let api = build_hf_api(!check)?;
    let repos = cached_repos()?;
    if repos.is_empty() {
        eprintln!("📦 No cached Hugging Face model repos found");
        eprintln!("   {}", huggingface_hub_cache_dir().display());
        return Ok(());
    }

    let selected: Vec<CachedRepo> = if check {
        if all {
            repos
        } else if let Some(repo_id) = repo {
            let repo_id = repo_id.trim();
            let Some(found) = repos.into_iter().find(|entry| entry.repo_id == repo_id) else {
                anyhow::bail!("Cached repo not found: {repo_id}");
            };
            vec![found]
        } else {
            repos
        }
    } else if all {
        repos
    } else {
        let Some(repo_id) = repo else {
            anyhow::bail!("Pass a repo id or --all. Use `mesh-llm models updates --check` to inspect updates without downloading.");
        };
        let repo_id = repo_id.trim();
        let Some(found) = repos.into_iter().find(|entry| entry.repo_id == repo_id) else {
            anyhow::bail!("Cached repo not found: {repo_id}");
        };
        vec![found]
    };

    if !check {
        eprintln!("🔄 Updating cached Hugging Face repos");
        eprintln!("📁 Cache: {}", huggingface_hub_cache_dir().display());
        eprintln!("📦 Selected: {}", selected.len());
        eprintln!();
    }
    let mut updates = 0usize;
    let total_selected = selected.len();
    let mut refresh_totals = UpdateCounts::default();
    for (index, repo) in selected.into_iter().enumerate() {
        if check {
            print_update_check_progress(index + 1, total_selected, &repo.repo_id)?;
            if let Some(remote_revision) = check_repo_update(&api, &repo)? {
                updates += 1;
                clear_progress_line()?;
                eprintln!("🆕 [{}/{}] {}", index + 1, total_selected, repo.repo_id);
                eprintln!("   ref: {}", repo.ref_name);
                eprintln!("   local: {}", short_revision(&repo.local_revision));
                eprintln!("   latest: {}", short_revision(&remote_revision));
                eprintln!("   update: mesh-llm models updates {}", repo.repo_id);
            }
        } else {
            eprintln!("🧭 [{}/{}] {}", index + 1, total_selected, repo.repo_id);
            let counts = update_cached_repo(&api, &repo)?;
            refresh_totals.refreshed += counts.refreshed;
            refresh_totals.missing_meta += counts.missing_meta;
        }
    }
    if check {
        clear_progress_line()?;
        if updates > 0 {
            eprintln!("📬 Update summary");
            eprintln!("   repos with updates: {updates}");
            eprintln!("   update one: mesh-llm models updates <repo>");
            eprintln!("   update all: mesh-llm models updates --all");
        }
    } else {
        eprintln!();
        eprintln!("✅ Update complete");
        eprintln!("   refreshed files: {}", refresh_totals.refreshed);
        if refresh_totals.missing_meta > 0 {
            eprintln!("   missing config.json: {}", refresh_totals.missing_meta);
        }
    }
    Ok(())
}

pub fn warn_about_updates_for_paths(paths: &[PathBuf]) {
    let mut cache_models = Vec::new();
    let mut seen = BTreeSet::new();
    for path in paths {
        let Some(repo) = (match cached_repo_for_path(path) {
            Ok(repo) => repo,
            Err(err) => {
                eprintln!(
                    "Warning: could not inspect cached Hugging Face repo for {}: {err}",
                    path.display()
                );
                continue;
            }
        }) else {
            continue;
        };
        if seen.insert((repo.repo_id.clone(), repo.local_revision.clone())) {
            cache_models.push(repo);
        }
    }
    if cache_models.is_empty() {
        return;
    }

    let api = match build_hf_api(false) {
        Ok(api) => api,
        Err(err) => {
            eprintln!("Warning: could not initialize Hugging Face update checks: {err}");
            return;
        }
    };
    for repo in cache_models {
        match check_repo_update(&api, &repo) {
            Ok(Some(remote_revision)) => {
                eprintln!("🆕 Update available for {}", repo.repo_id);
                eprintln!("   local: {}", short_revision(&repo.local_revision));
                eprintln!("   latest: {}", short_revision(&remote_revision));
                eprintln!("   continuing with pinned local snapshot");
                eprintln!("   update: mesh-llm models updates {}", repo.repo_id);
            }
            Ok(None) => {}
            Err(err) => {
                eprintln!(
                    "Warning: could not check for updates for {}: {err}",
                    repo.repo_id
                );
            }
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

fn build_hf_api(progress: bool) -> Result<Api> {
    let mut builder = ApiBuilder::from_cache(huggingface_hub_cache()).with_progress(progress);
    if let Ok(endpoint) = std::env::var("HF_ENDPOINT") {
        let endpoint = endpoint.trim();
        if !endpoint.is_empty() {
            builder = builder.with_endpoint(endpoint.to_string());
        }
    }
    builder = builder.with_token(hf_token_override());
    builder.build().context("Build Hugging Face API client")
}

fn hf_token_override() -> Option<String> {
    for key in ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"] {
        if let Ok(token) = std::env::var(key) {
            let token = token.trim();
            if !token.is_empty() {
                return Some(token.to_string());
            }
        }
    }
    None
}

fn migration_entries() -> Vec<MigrationEntry> {
    let mut paths = legacy_gguf_files(legacy_models_dir());
    paths.sort();
    paths.into_iter().map(classify_legacy_path).collect()
}

fn legacy_gguf_files(root: PathBuf) -> Vec<PathBuf> {
    let mut files = Vec::new();
    if !root.exists() {
        return files;
    }
    let mut stack = vec![root];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let Ok(file_type) = entry.file_type() else {
                continue;
            };
            if file_type.is_dir() {
                stack.push(path);
                continue;
            }
            if path.extension().and_then(|ext| ext.to_str()) == Some("gguf") {
                files.push(path);
            }
        }
    }
    files
}

fn classify_legacy_path(path: PathBuf) -> MigrationEntry {
    if let Some((model, repo, revision, file)) = catalog_hf_match(&path) {
        let detail = match revision {
            Some(revision) => format!(
                "recognized as {} from Hugging Face repo {} at revision {} ({})",
                model.name, repo, revision, file
            ),
            None => format!(
                "recognized as {} from Hugging Face repo {} ({})",
                model.name, repo, file
            ),
        };
        return MigrationEntry {
            path,
            status: MigrationStatus::Rehydratable,
            detail,
            catalog: Some(model),
        };
    }

    if let Some(model) = catalog_match(&path) {
        return MigrationEntry {
            path,
            status: MigrationStatus::LegacyOnly,
            detail: format!(
                "recognized as {}, but its catalog source is not a Hugging Face repo",
                model.name
            ),
            catalog: None,
        };
    }

    MigrationEntry {
        path,
        status: MigrationStatus::LegacyOnly,
        detail: "no canonical Hugging Face source is known for this GGUF".to_string(),
        catalog: None,
    }
}

fn catalog_hf_match(
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

fn catalog_hf_asset_ref(
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

fn catalog_match(path: &Path) -> Option<&'static catalog::CatalogModel> {
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

fn matching_catalog_model_for_huggingface(
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

fn parse_hf_resolve_url(url: &str) -> Option<(String, Option<String>, String)> {
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

fn parse_huggingface_ref(input: &str) -> Option<(String, Option<String>, String)> {
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

fn huggingface_resolve_url(repo: &str, revision: Option<&str>, file: &str) -> String {
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

fn file_preference_score(file: &str) -> usize {
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

fn http_client() -> Result<reqwest::Client> {
    reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(300))
        .connect_timeout(std::time::Duration::from_secs(30))
        .user_agent(format!("mesh-llm/{}", crate::VERSION))
        .build()
        .context("Build HTTP client")
}

async fn remote_size_label(url: &str) -> Option<String> {
    let client = http_client().ok()?;
    let mut request = client.head(url);
    if url.contains("huggingface.co/") {
        if let Some(token) = hf_token_override() {
            request = request.bearer_auth(token);
        }
    }
    let response = request.send().await.ok()?.error_for_status().ok()?;
    let size = response
        .headers()
        .get(reqwest::header::CONTENT_LENGTH)?
        .to_str()
        .ok()?
        .parse::<u64>()
        .ok()?;
    Some(format_size_bytes(size))
}

fn format_size_bytes(bytes: u64) -> String {
    if bytes >= 1_000_000_000 {
        format!("{:.1}GB", bytes as f64 / 1e9)
    } else {
        format!("{:.0}MB", bytes as f64 / 1e6)
    }
}

fn short_revision(revision: &str) -> String {
    if revision.len() <= 12 {
        revision.to_string()
    } else {
        revision[..12].to_string()
    }
}

fn migrate_catalog_model(
    api: &Api,
    model: &catalog::CatalogModel,
    entries: &[&MigrationEntry],
) -> Result<MigrationCounts> {
    let mut counts = MigrationCounts::default();
    let legacy_files: BTreeMap<String, &PathBuf> = entries
        .iter()
        .map(|entry| (entry.file_name().to_lowercase(), &entry.path))
        .collect();

    let mut config_downloaded = BTreeSet::new();
    for url in model
        .extra_files
        .iter()
        .map(|asset| asset.url.as_str())
        .chain(std::iter::once(model.url.as_str()))
        .chain(model.mmproj.iter().map(|asset| asset.url.as_str()))
    {
        let Some((repo_id, revision, file)) = parse_hf_resolve_url(url) else {
            continue;
        };
        let repo = Repo::with_revision(
            repo_id.clone(),
            RepoType::Model,
            revision.clone().unwrap_or_else(|| "main".to_string()),
        );
        let api_repo = api.repo(repo.clone());
        if let Some(legacy_path) = legacy_files.get(&file.to_lowercase()) {
            match adopt_legacy_asset_into_hf_cache(api, &repo, &file, legacy_path)? {
                Some(path) => {
                    eprintln!("   🔁 adopted {}", path.display());
                    counts.adopted += 1;
                }
                None => {
                    let path = api_repo
                        .download(&file)
                        .with_context(|| format!("Download {repo_id}/{file}"))?;
                    eprintln!("   ✅ downloaded {}", path.display());
                    counts.downloaded += 1;
                }
            }
        } else {
            let path = api_repo
                .download(&file)
                .with_context(|| format!("Download {repo_id}/{file}"))?;
            eprintln!("   ✅ downloaded {}", path.display());
            counts.downloaded += 1;
        }

        if config_downloaded.insert((repo_id.clone(), revision.clone())) {
            match api_repo.get("config.json") {
                Ok(config_path) => eprintln!("   🧾 config {}", config_path.display()),
                Err(err) => {
                    if is_not_found_error(&err.to_string()) {
                        eprintln!("   ℹ️ no config.json published for {repo_id}");
                    } else {
                        eprintln!("   ⚠️ config {repo_id}: {err}");
                    }
                }
            }
        }
    }

    Ok(counts)
}

fn run_prune(entries: &[MigrationEntry]) -> Result<()> {
    eprintln!();
    eprintln!("🧹 Pruning migrated legacy files");
    eprintln!("📁 Source: {}", legacy_models_dir().display());
    eprintln!();

    let mut pruned = 0usize;
    let mut kept = 0usize;
    for entry in entries
        .iter()
        .filter(|entry| entry.status == MigrationStatus::Rehydratable)
    {
        if pruneable_legacy_path(entry)? {
            std::fs::remove_file(&entry.path)
                .with_context(|| format!("Remove {}", entry.path.display()))?;
            eprintln!("   🗑️ {}", entry.path.display());
            pruned += 1;
        } else {
            eprintln!("   ⏭️ {}", entry.path.display());
            kept += 1;
        }
    }

    eprintln!();
    eprintln!("✅ Prune complete");
    eprintln!("   removed legacy files: {pruned}");
    eprintln!("   kept legacy files: {kept}");
    Ok(())
}

fn pruneable_legacy_path(entry: &MigrationEntry) -> Result<bool> {
    let Some(model) = entry.catalog else {
        return Ok(false);
    };
    let file_name = entry.file_name();
    let Some((repo_id, revision, cached_file)) = catalog_hf_asset_ref(model, &file_name) else {
        return Ok(false);
    };
    let cache = huggingface_hub_cache();
    let repo = Repo::with_revision(
        repo_id,
        RepoType::Model,
        revision.unwrap_or_else(|| "main".to_string()),
    );
    let cache_repo = cache.repo(repo);
    Ok(cache_repo.get(&cached_file).is_some())
}

fn adopt_legacy_asset_into_hf_cache(
    api: &Api,
    repo: &Repo,
    file: &str,
    legacy_path: &Path,
) -> Result<Option<PathBuf>> {
    let url = api.repo(repo.clone()).url(file);
    let metadata = match api.metadata(&url) {
        Ok(metadata) => metadata,
        Err(err) => {
            eprintln!("   ⚠️ could not verify {file} for adoption: {err}");
            return Ok(None);
        }
    };

    let cache = huggingface_hub_cache();
    let cache_repo = cache.repo(repo.clone());
    let etag = metadata.etag();
    let blob_path = cache_repo.blob_path(etag);
    if blob_path.exists() {
        return materialize_cached_snapshot_pointer(
            &cache_repo,
            metadata.commit_hash(),
            file,
            &blob_path,
        );
    }

    let legacy_size = std::fs::metadata(legacy_path)
        .with_context(|| format!("Read {}", legacy_path.display()))?
        .len();
    if legacy_size != metadata.size() as u64 {
        eprintln!(
            "   ⚠️ size mismatch for {} (legacy {}, remote {})",
            legacy_path.display(),
            format_size_bytes(legacy_size),
            format_size_bytes(metadata.size() as u64)
        );
        return Ok(None);
    }

    if etag.len() == 64 && etag.chars().all(|ch| ch.is_ascii_hexdigit()) {
        let display_name = legacy_path
            .file_name()
            .and_then(|value| value.to_str())
            .unwrap_or(file);
        let digest = sha256_file_hex(legacy_path, display_name, legacy_size)?;
        eprintln!(
            "   ✅ verified {} ({})",
            display_name,
            format_size_bytes(legacy_size)
        );
        if !digest.eq_ignore_ascii_case(etag) {
            eprintln!("   ⚠️ checksum mismatch for {}", legacy_path.display());
            return Ok(None);
        }
    }

    if let Some(parent) = blob_path.parent() {
        std::fs::create_dir_all(parent).with_context(|| format!("Create {}", parent.display()))?;
    }
    link_or_copy_file(legacy_path, &blob_path)?;

    materialize_cached_snapshot_pointer(&cache_repo, metadata.commit_hash(), file, &blob_path)
}

fn materialize_cached_snapshot_pointer(
    cache_repo: &hf_hub::CacheRepo,
    commit_hash: &str,
    file: &str,
    blob_path: &Path,
) -> Result<Option<PathBuf>> {
    let mut pointer_path = cache_repo.pointer_path(commit_hash);
    pointer_path.push(file);
    if let Some(parent) = pointer_path.parent() {
        std::fs::create_dir_all(parent).with_context(|| format!("Create {}", parent.display()))?;
    }
    link_or_copy_file(blob_path, &pointer_path)?;
    cache_repo
        .create_ref(commit_hash)
        .context("Write cache ref")?;

    Ok(Some(pointer_path))
}

fn link_or_copy_file(src: &Path, dst: &Path) -> Result<()> {
    if dst.exists() {
        return Ok(());
    }
    match std::fs::hard_link(src, dst) {
        Ok(()) => Ok(()),
        Err(_) => {
            std::fs::copy(src, dst)
                .with_context(|| format!("Copy {} -> {}", src.display(), dst.display()))?;
            Ok(())
        }
    }
}

fn sha256_file_hex(path: &Path, label: &str, total_bytes: u64) -> Result<String> {
    let file = std::fs::File::open(path).with_context(|| format!("Open {}", path.display()))?;
    let mut reader = std::io::BufReader::new(file);
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 1024 * 1024];
    let mut processed = 0u64;
    let mut last_progress = std::time::Instant::now();
    print_verify_progress(label, processed, total_bytes)?;
    loop {
        let read = reader
            .read(&mut buffer)
            .with_context(|| format!("Read {}", path.display()))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
        processed += read as u64;
        if total_bytes > 0 && last_progress.elapsed() >= std::time::Duration::from_millis(500) {
            print_verify_progress(label, processed, total_bytes)?;
            last_progress = std::time::Instant::now();
        }
    }
    print_verify_progress(label, total_bytes, total_bytes)?;
    eprintln!();
    Ok(format!("{:x}", hasher.finalize()))
}

fn print_verify_progress(label: &str, processed: u64, total_bytes: u64) -> Result<()> {
    let pct = if total_bytes > 0 {
        (processed as f64 / total_bytes as f64) * 100.0
    } else {
        0.0
    };
    eprint!(
        "\r   🔍 Verifying {}  {:>5.1}%  {}/{}",
        label,
        pct,
        format_size_bytes(processed),
        format_size_bytes(total_bytes)
    );
    std::io::stderr().flush().context("Flush verify progress")?;
    Ok(())
}

fn print_update_check_progress(current: usize, total: usize, repo_id: &str) -> Result<()> {
    let pct = if total > 0 {
        (current as f64 / total as f64) * 100.0
    } else {
        100.0
    };
    eprint!(
        "\r🔄 Checking updates {:>5.1}%  [{}/{}] {}",
        pct, current, total, repo_id
    );
    std::io::stderr()
        .flush()
        .context("Flush update check progress")?;
    Ok(())
}

fn clear_progress_line() -> Result<()> {
    eprint!("\r{: <140}\r", "");
    std::io::stderr().flush().context("Flush progress clear")?;
    Ok(())
}

fn cached_repos() -> Result<Vec<CachedRepo>> {
    let root = huggingface_hub_cache_dir();
    let mut repos = Vec::new();
    if !root.exists() {
        return Ok(repos);
    }

    for entry in std::fs::read_dir(&root).with_context(|| format!("Read {}", root.display()))? {
        let entry = entry?;
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|value| value.to_str()) else {
            continue;
        };
        if !name.starts_with("models--") {
            continue;
        }
        let Some(repo_id) = cache_repo_id_from_dir(name) else {
            continue;
        };
        // `hf-hub` does not currently expose cached ref enumeration, so we
        // still inspect the repo directory to discover installed revisions.
        let refs_dir = path.join("refs");
        if !refs_dir.is_dir() {
            continue;
        }
        if let Some((ref_name, local_revision)) = first_cache_ref(&refs_dir)? {
            repos.push(CachedRepo {
                repo_id,
                ref_name,
                local_revision,
            });
        }
    }

    repos.sort_by(|left, right| left.repo_id.cmp(&right.repo_id));
    Ok(repos)
}

fn cached_repo_for_path(path: &Path) -> Result<Option<CachedRepo>> {
    let root = huggingface_hub_cache_dir();
    let rel = match path.strip_prefix(&root) {
        Ok(rel) => rel,
        Err(_) => return Ok(None),
    };
    let mut components = rel.components();
    let Some(repo_component) = components.next() else {
        return Ok(None);
    };
    let Some(repo_dir_name) = repo_component.as_os_str().to_str() else {
        return Ok(None);
    };
    if !repo_dir_name.starts_with("models--") {
        return Ok(None);
    }
    let Some(snapshot_component) = components.next() else {
        return Ok(None);
    };
    if snapshot_component.as_os_str() != "snapshots" {
        return Ok(None);
    }
    let Some(revision_component) = components.next() else {
        return Ok(None);
    };
    let Some(local_revision) = revision_component.as_os_str().to_str() else {
        return Ok(None);
    };
    let Some(repo_id) = cache_repo_id_from_dir(repo_dir_name) else {
        return Ok(None);
    };
    let repo_dir = root.join(repo_dir_name);
    let ref_name =
        matching_ref_name(&repo_dir, local_revision)?.unwrap_or_else(|| "main".to_string());
    Ok(Some(CachedRepo {
        repo_id,
        ref_name,
        local_revision: local_revision.to_string(),
    }))
}

fn cache_repo_id_from_dir(name: &str) -> Option<String> {
    Some(name.strip_prefix("models--")?.replace("--", "/"))
}

fn first_cache_ref(refs_dir: &Path) -> Result<Option<(String, String)>> {
    let main = refs_dir.join("main");
    if main.is_file() {
        let value = std::fs::read_to_string(&main)
            .with_context(|| format!("Read {}", main.display()))?
            .trim()
            .to_string();
        if !value.is_empty() {
            return Ok(Some(("main".to_string(), value)));
        }
    }

    let mut refs = Vec::new();
    collect_ref_files(refs_dir, refs_dir, &mut refs)?;
    refs.sort_by(|left, right| left.0.cmp(&right.0));
    Ok(refs.into_iter().next())
}

fn matching_ref_name(repo_dir: &Path, revision: &str) -> Result<Option<String>> {
    let refs_dir = repo_dir.join("refs");
    if !refs_dir.is_dir() {
        return Ok(None);
    }
    let mut refs = Vec::new();
    collect_ref_files(&refs_dir, &refs_dir, &mut refs)?;
    refs.sort_by(|left, right| left.0.cmp(&right.0));
    Ok(refs
        .into_iter()
        .find(|(_, value)| value == revision)
        .map(|(name, _)| name))
}

// This remains a small compatibility layer around the on-disk cache because
// `hf-hub` does not expose "list all refs for this cached repo" yet.
fn collect_ref_files(root: &Path, dir: &Path, refs: &mut Vec<(String, String)>) -> Result<()> {
    for entry in std::fs::read_dir(dir).with_context(|| format!("Read {}", dir.display()))? {
        let entry = entry?;
        let path = entry.path();
        let file_type = entry.file_type()?;
        if file_type.is_dir() {
            collect_ref_files(root, &path, refs)?;
            continue;
        }
        if !file_type.is_file() {
            continue;
        }
        let ref_name = path
            .strip_prefix(root)
            .unwrap_or(&path)
            .to_string_lossy()
            .replace('\\', "/");
        let revision = std::fs::read_to_string(&path)
            .with_context(|| format!("Read {}", path.display()))?
            .trim()
            .to_string();
        if !revision.is_empty() {
            refs.push((ref_name, revision));
        }
    }
    Ok(())
}

fn remote_repo_info(api: &Api, repo_id: &str, ref_name: &str) -> Result<RepoInfo> {
    let repo = Repo::with_revision(repo_id.to_string(), RepoType::Model, ref_name.to_string());
    api.repo(repo)
        .info()
        .with_context(|| format!("Fetch repo info for {repo_id}@{ref_name}"))
}

fn repo_info_sha(info: &RepoInfo) -> String {
    info.sha.clone()
}

fn check_repo_update(api: &Api, repo: &CachedRepo) -> Result<Option<String>> {
    let remote = remote_repo_info(api, &repo.repo_id, &repo.ref_name)?;
    let remote_revision = repo_info_sha(&remote);
    if remote_revision == repo.local_revision {
        Ok(None)
    } else {
        Ok(Some(remote_revision))
    }
}

fn update_cached_repo(api: &Api, repo: &CachedRepo) -> Result<UpdateCounts> {
    let repo_handle =
        Repo::with_revision(repo.repo_id.clone(), RepoType::Model, repo.ref_name.clone());
    let api_repo = api.repo(repo_handle);
    let files = cached_repo_files(repo)?;
    if files.is_empty() {
        eprintln!("⚠️ {} has no cached files to refresh", repo.repo_id);
        return Ok(UpdateCounts::default());
    }

    eprintln!("   ref: {}", repo.ref_name);
    eprintln!("   current: {}", short_revision(&repo.local_revision));
    let mut counts = UpdateCounts::default();
    let mut downloaded = BTreeSet::new();
    let total_files = files.len() + 1;
    let mut position = 0usize;
    for file in files
        .into_iter()
        .chain(std::iter::once("config.json".to_string()))
    {
        if !downloaded.insert(file.clone()) {
            continue;
        }
        position += 1;
        eprintln!("   ↻ [{}/{}] {}", position, total_files, file);
        match api_repo.download(&file) {
            Ok(path) => {
                eprintln!("   ✅ {}", path.display());
                counts.refreshed += 1;
            }
            Err(err) if file == "config.json" => {
                if is_not_found_error(&err.to_string()) {
                    eprintln!("   ℹ️ no config.json published for {}", repo.repo_id);
                } else {
                    eprintln!("   ⚠️ config.json: {err}");
                }
                counts.missing_meta += 1;
            }
            Err(err) => {
                return Err(err).with_context(|| format!("Download {}/{}", repo.repo_id, file))
            }
        }
    }

    Ok(counts)
}

fn is_not_found_error(message: &str) -> bool {
    let message = message.to_ascii_lowercase();
    message.contains("404") || message.contains("not found")
}

fn cached_repo_files(repo: &CachedRepo) -> Result<Vec<String>> {
    let cache = huggingface_hub_cache();
    let repo_handle =
        Repo::with_revision(repo.repo_id.clone(), RepoType::Model, repo.ref_name.clone());
    let root = cache.repo(repo_handle).pointer_path(&repo.local_revision);
    if !root.is_dir() {
        return Ok(Vec::new());
    }

    let mut files = Vec::new();
    collect_snapshot_files(&root, &root, &mut files)?;
    files.sort();
    Ok(files)
}

fn collect_snapshot_files(root: &Path, dir: &Path, files: &mut Vec<String>) -> Result<()> {
    for entry in std::fs::read_dir(dir).with_context(|| format!("Read {}", dir.display()))? {
        let entry = entry?;
        let path = entry.path();
        let file_type = entry.file_type()?;
        if file_type.is_dir() {
            collect_snapshot_files(root, &path, files)?;
            continue;
        }
        if !file_type.is_file() && !file_type.is_symlink() {
            continue;
        }
        let rel = path
            .strip_prefix(root)
            .unwrap_or(&path)
            .to_string_lossy()
            .replace('\\', "/");
        files.push(rel);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_hf_resolve_url_extracts_repo_revision_and_file() {
        let (repo, revision, file) = parse_hf_resolve_url(
            "https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf",
        )
        .unwrap();
        assert_eq!(repo, "Qwen/Qwen3-8B-GGUF");
        assert_eq!(revision.as_deref(), Some("main"));
        assert_eq!(file, "Qwen3-8B-Q4_K_M.gguf");
    }

    #[test]
    fn cache_repo_id_from_dir_decodes_hf_cache_names() {
        assert_eq!(
            cache_repo_id_from_dir("models--Qwen--Qwen3-8B-GGUF"),
            Some("Qwen/Qwen3-8B-GGUF".to_string())
        );
    }

    #[test]
    fn parse_huggingface_ref_accepts_revision_shorthand() {
        let (repo, revision, file) =
            parse_huggingface_ref("Qwen/Qwen3-8B-GGUF@main/Qwen3-8B-Q4_K_M.gguf").unwrap();
        assert_eq!(repo, "Qwen/Qwen3-8B-GGUF");
        assert_eq!(revision.as_deref(), Some("main"));
        assert_eq!(file, "Qwen3-8B-Q4_K_M.gguf");
    }

    #[test]
    fn find_catalog_model_exact_matches_filename_stem() {
        let model = find_catalog_model_exact("Qwen3-8B-Q4_K_M").unwrap();
        assert_eq!(model.name, "Qwen3-8B-Q4_K_M");
    }
}
