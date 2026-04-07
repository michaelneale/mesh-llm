//! Built-in model catalog plus managed acquisition helpers.

use super::access;
use anyhow::{Context, Result};
use hf_hub::api::sync::ApiError as SyncApiError;
use hf_hub::api::Progress as HfProgress;
use hf_hub::{Repo, RepoType};
use serde::Deserialize;
use std::collections::BTreeSet;
#[cfg(test)]
use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};
#[cfg(test)]
use std::sync::Arc;
use std::sync::LazyLock;
#[cfg(test)]
use std::sync::Mutex;
use std::time::{Duration, Instant};
use tokio::io::AsyncWriteExt;
#[derive(Clone, Debug, Deserialize)]
pub struct CatalogAsset {
    pub file: String,
    pub url: String,
}

#[derive(Clone, Debug)]
pub struct CatalogModel {
    pub name: String,
    pub file: String,
    /// Legacy transport field. Prefer `source_repo()`, `source_revision()`,
    /// and `source_file()` for curated model identity.
    pub url: String,
    pub size: String,
    pub description: String,
    /// If set, this model has a recommended draft model for speculative decoding.
    pub draft: Option<String>,
    /// MoE expert routing config. If set, this model supports expert sharding.
    /// Pre-computed from `moe-analyze --export-ranking --all-layers`.
    pub moe: Option<MoeConfig>,
    /// Additional split GGUF files (for models too large for a single file).
    /// llama.cpp auto-discovers splits from the first file, but all parts
    /// must be present in the same directory.
    pub extra_files: Vec<CatalogAsset>,
    /// Multimodal projector for vision models.
    /// When set, llama-server is launched with `--mmproj <file>`.
    pub mmproj: Option<CatalogAsset>,
}

impl CatalogModel {
    pub fn source_repo(&self) -> Option<&str> {
        parse_hf_resolve_url_parts(&self.url).map(|(repo, _, _)| repo)
    }

    pub fn source_revision(&self) -> Option<&str> {
        parse_hf_resolve_url_parts(&self.url).and_then(|(_, revision, _)| revision)
    }

    pub fn source_file(&self) -> Option<&str> {
        parse_hf_resolve_url_parts(&self.url).map(|(_, _, file)| file)
    }
}

/// Pre-computed MoE expert sharding configuration for a model.
/// Derived from router gate mass analysis — determines which experts go where.
#[derive(Clone, Debug)]
pub struct MoeConfig {
    /// Total number of experts in the model
    pub n_expert: u32,
    /// Number of experts selected per token (top-k)
    pub n_expert_used: u32,
    /// Minimum number of experts per node for coherent output.
    /// Determined experimentally per model (~36% for Qwen3-30B-A3B).
    pub min_experts_per_node: u32,
    /// Expert IDs sorted by gate mass descending (hottest first).
    pub ranking: Vec<u32>,
}

#[derive(Debug, Deserialize)]
struct CatalogModelJson {
    name: String,
    file: String,
    url: String,
    size: String,
    description: String,
    draft: Option<String>,
    moe: Option<MoeConfigJson>,
    #[serde(default)]
    extra_files: Vec<CatalogAsset>,
    mmproj: Option<CatalogAsset>,
}

#[derive(Clone, Debug, Deserialize)]
struct MoeConfigJson {
    n_expert: u32,
    n_expert_used: u32,
    min_experts_per_node: u32,
}

pub static MODEL_CATALOG: LazyLock<Vec<CatalogModel>> = LazyLock::new(load_catalog);

fn load_catalog() -> Vec<CatalogModel> {
    let raw: Vec<CatalogModelJson> =
        serde_json::from_str(include_str!("catalog.json")).expect("parse bundled catalog.json");
    raw.into_iter().map(CatalogModel::from_json).collect()
}

impl CatalogModel {
    fn from_json(raw: CatalogModelJson) -> Self {
        Self {
            name: raw.name,
            file: raw.file,
            url: raw.url,
            size: raw.size,
            description: raw.description,
            draft: raw.draft,
            moe: raw.moe.map(MoeConfig::from_json),
            extra_files: raw.extra_files,
            mmproj: raw.mmproj,
        }
    }
}

impl MoeConfig {
    fn from_json(raw: MoeConfigJson) -> Self {
        Self {
            n_expert: raw.n_expert,
            n_expert_used: raw.n_expert_used,
            min_experts_per_node: raw.min_experts_per_node,
            ranking: Vec::new(),
        }
    }
}

/// Get the canonical managed model root (the Hugging Face hub cache).
pub fn models_dir() -> PathBuf {
    crate::models::huggingface_hub_cache_dir()
}

/// Find a catalog model by name (case-insensitive partial match)
/// Parse a size string like "20GB", "4.4GB", "491MB" into GB as f64.
pub fn parse_size_gb(s: &str) -> f64 {
    let s = s.trim();
    if let Some(gb) = s.strip_suffix("GB") {
        gb.trim().parse().unwrap_or(0.0)
    } else if let Some(mb) = s.strip_suffix("MB") {
        mb.trim().parse::<f64>().unwrap_or(0.0) / 1000.0
    } else {
        0.0
    }
}

pub fn find_model(query: &str) -> Option<&'static CatalogModel> {
    let q = query.to_lowercase();
    MODEL_CATALOG
        .iter()
        .find(|m| m.name.to_lowercase() == q)
        .or_else(|| {
            MODEL_CATALOG
                .iter()
                .find(|m| m.name.to_lowercase().contains(&q))
        })
}

fn parse_hf_resolve_url_parts(url: &str) -> Option<(&str, Option<&str>, &str)> {
    let tail = url
        .strip_prefix("https://huggingface.co/")
        .or_else(|| url.strip_prefix("http://huggingface.co/"))?;
    let (repo, rest) = tail.split_once("/resolve/")?;
    if !repo.contains('/') {
        return None;
    }
    let (revision, file) = rest.split_once('/')?;
    if file.is_empty() {
        return None;
    }
    Some((repo, Some(revision), file))
}

pub fn huggingface_repo_url(url: &str) -> Option<String> {
    let (repo, _, _) = parse_hf_resolve_url_parts(url)?;
    Some(format!("https://huggingface.co/{repo}"))
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
struct HfAsset {
    repo: String,
    revision: String,
    file: String,
}

impl HfAsset {
    fn repo_handle(&self) -> Repo {
        Repo::with_revision(self.repo.clone(), RepoType::Model, self.revision.clone())
    }
}

fn hf_asset_from_url(url: &str) -> Option<HfAsset> {
    let (repo, revision, file) = parse_hf_resolve_url_parts(url)?;
    Some(HfAsset {
        repo: repo.to_string(),
        revision: revision.unwrap_or("main").to_string(),
        file: file.to_string(),
    })
}

fn expand_split_asset(asset: &HfAsset) -> Result<Vec<HfAsset>> {
    let re = regex_lite::Regex::new(r"-00001-of-(\d{5})\.gguf$").unwrap();
    let Some(caps) = re.captures(&asset.file) else {
        return Ok(vec![asset.clone()]);
    };
    let count: u32 = caps[1].parse()?;
    Ok((1..=count)
        .map(|index| HfAsset {
            repo: asset.repo.clone(),
            revision: asset.revision.clone(),
            file: asset
                .file
                .replace("-00001-of-", &format!("-{index:05}-of-")),
        })
        .collect())
}

#[cfg(test)]
type DownloadHfAssetsOverrideFn =
    Arc<dyn Fn(&str, Vec<HfAsset>) -> Result<Vec<PathBuf>> + Send + Sync>;

#[cfg(test)]
static DOWNLOAD_HF_ASSETS_OVERRIDE: LazyLock<Mutex<HashMap<String, DownloadHfAssetsOverrideFn>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

#[cfg(test)]
struct DownloadHfAssetsOverrideGuard(String);

#[cfg(test)]
impl DownloadHfAssetsOverrideGuard {
    fn set(label: String, func: DownloadHfAssetsOverrideFn) -> Self {
        let mut map = DOWNLOAD_HF_ASSETS_OVERRIDE.lock().unwrap();
        map.insert(label.clone(), func);
        DownloadHfAssetsOverrideGuard(label)
    }
}

#[cfg(test)]
impl Drop for DownloadHfAssetsOverrideGuard {
    fn drop(&mut self) {
        let mut map = DOWNLOAD_HF_ASSETS_OVERRIDE.lock().unwrap();
        map.remove(&self.0);
    }
}

async fn download_hf_assets(label: &str, assets: Vec<HfAsset>) -> Result<Vec<PathBuf>> {
    let label = label.to_string();
    #[cfg(test)]
    {
        let func = DOWNLOAD_HF_ASSETS_OVERRIDE
            .lock()
            .unwrap()
            .get(&label)
            .cloned();
        if let Some(func) = func {
            return func(&label, assets);
        }
    }
    ensure_hf_repo_access(&assets).await?;
    tokio::task::spawn_blocking(move || download_hf_assets_blocking(&label, assets))
        .await
        .context("Join Hugging Face download task")?
}

async fn ensure_hf_repo_access(assets: &[HfAsset]) -> Result<()> {
    let repos: BTreeSet<(String, String)> = assets
        .iter()
        .map(|asset| (asset.repo.clone(), asset.revision.clone()))
        .collect();
    if repos.is_empty() {
        return Ok(());
    }
    let api = super::build_hf_tokio_api(false)?;
    for (repo, revision) in repos {
        if matches!(
            access::probe_repo_access(&api, &repo, Some(revision.as_str())).await?,
            access::RepoAccess::Gated
        ) {
            return Err(anyhow::anyhow!(access::gated_access_message(&repo)));
        }
    }
    Ok(())
}

struct MeshDownloadProgress {
    filename: String,
    total: usize,
    downloaded: usize,
    last_draw: Option<Instant>,
}

impl MeshDownloadProgress {
    fn new() -> Self {
        Self {
            filename: String::new(),
            total: 0,
            downloaded: 0,
            last_draw: None,
        }
    }

    fn draw(&mut self, force: bool) {
        let now = Instant::now();
        if !force
            && self
                .last_draw
                .is_some_and(|last| now.duration_since(last) < Duration::from_millis(150))
        {
            return;
        }
        self.last_draw = Some(now);
        let percent = if self.total == 0 {
            0
        } else {
            ((self.downloaded as f64 / self.total as f64) * 100.0).round() as usize
        };
        eprint!(
            "\r   ⏬ {} {:>3}% ({}/{})",
            self.filename,
            percent.min(100),
            format_download_bytes(self.downloaded as u64),
            format_download_bytes(self.total as u64)
        );
        let _ = std::io::stderr().flush();
        if force {
            eprintln!();
        }
    }
}

impl HfProgress for MeshDownloadProgress {
    fn init(&mut self, size: usize, filename: &str) {
        self.filename = filename.to_string();
        self.total = size;
        self.downloaded = 0;
        self.last_draw = None;
        self.draw(false);
    }

    fn update(&mut self, size: usize) {
        self.downloaded = self.downloaded.saturating_add(size);
        self.draw(false);
    }

    fn finish(&mut self) {
        self.downloaded = self.total;
        self.draw(true);
    }
}

fn format_download_bytes(bytes: u64) -> String {
    if bytes >= 1_000_000_000 {
        format!("{:.1}GB", bytes as f64 / 1e9)
    } else if bytes >= 1_000_000 {
        format!("{:.0}MB", bytes as f64 / 1e6)
    } else if bytes >= 1_000 {
        format!("{:.0}KB", bytes as f64 / 1e3)
    } else {
        format!("{bytes}B")
    }
}

fn download_hf_assets_blocking(label: &str, assets: Vec<HfAsset>) -> Result<Vec<PathBuf>> {
    let api = super::build_hf_api(false)?;
    let cache = crate::models::huggingface_hub_cache();
    let download_plan = build_download_plan(assets)?;

    eprintln!("📥 Ensuring {} is available locally...", label);

    let mut primary_paths = Vec::new();
    for (required, asset) in download_plan {
        let repo_handle = asset.repo_handle();
        let cache_repo = cache.repo(repo_handle.clone());
        let api_repo = api.repo(repo_handle);
        let path = match cache_repo.get(&asset.file) {
            Some(path) => {
                if required {
                    eprintln!("   ✅ Using cached model {}", asset.file);
                } else {
                    eprintln!("   🧾 Using cached model metadata");
                }
                path
            }
            None => {
                if required {
                    eprintln!("   📥 Downloading model {}", asset.file);
                }
                match if required {
                    api_repo.download_with_progress(&asset.file, MeshDownloadProgress::new())
                } else {
                    api_repo.download(&asset.file)
                } {
                    Ok(path) => {
                        if required {
                            eprintln!("   ✅ Ready {}", asset.file);
                        } else {
                            eprintln!("   🧾 Downloaded model metadata");
                        }
                        path
                    }
                    Err(_) if !required && asset.file == "config.json" => {
                        continue;
                    }
                    Err(err) => {
                        if hf_download_error_looks_access_denied(&err) {
                            return Err(anyhow::anyhow!(
                                "Cannot access Hugging Face asset {}/{}@{} ({}). {}",
                                asset.repo,
                                asset.file,
                                asset.revision,
                                err,
                                access::gated_access_message(&asset.repo)
                            ));
                        }
                        return Err(err).with_context(|| {
                            format!(
                                "Cache Hugging Face asset {}/{}@{}",
                                asset.repo, asset.file, asset.revision
                            )
                        });
                    }
                }
            }
        };
        if required && asset.file != "config.json" {
            primary_paths.push(path);
        }
    }

    Ok(primary_paths)
}

fn build_download_plan(
    assets: Vec<HfAsset>,
) -> Result<std::collections::BTreeSet<(bool, HfAsset)>> {
    let mut download_plan = std::collections::BTreeSet::new();
    let mut config_repos = std::collections::BTreeSet::new();

    for asset in assets {
        for expanded in expand_split_asset(&asset)? {
            config_repos.insert((expanded.repo.clone(), expanded.revision.clone()));
            download_plan.insert((true, expanded));
        }
    }
    for (repo, revision) in config_repos {
        download_plan.insert((
            false,
            HfAsset {
                repo,
                revision,
                file: "config.json".to_string(),
            },
        ));
    }
    Ok(download_plan)
}

fn hf_download_error_looks_access_denied(err: &SyncApiError) -> bool {
    let text = err.to_string().to_ascii_lowercase();
    text.contains("http status: 401") || text.contains("http status: 403") || text.contains("gated")
}

pub async fn download_hf_repo_file(
    repo: &str,
    revision: Option<&str>,
    file: &str,
) -> Result<PathBuf> {
    let revision = revision.unwrap_or("main").to_string();
    let asset = HfAsset {
        repo: repo.to_string(),
        revision: revision.clone(),
        file: file.to_string(),
    };
    let mut paths =
        download_hf_assets(&format!("{repo}/{file}@{revision}"), vec![asset.clone()]).await?;
    paths.sort();
    paths
        .into_iter()
        .find(|path| path_suffix_matches_ignore_case(path, &asset.file))
        .ok_or_else(|| {
            anyhow::anyhow!(
                "Downloaded Hugging Face asset not found in cache: {repo}/{file}@{revision}"
            )
        })
}

/// Download a model into the managed model store with resume support.
/// Returns the path to the primary downloaded file.
/// For split GGUFs (extra_files), downloads all parts to the same directory.
pub async fn download_model(model: &CatalogModel) -> Result<PathBuf> {
    let hf_assets: Option<Vec<HfAsset>> = std::iter::once(model.url.as_str())
        .chain(model.extra_files.iter().map(|asset| asset.url.as_str()))
        .chain(model.mmproj.iter().map(|asset| asset.url.as_str()))
        .map(hf_asset_from_url)
        .collect();
    if let Some(assets) = hf_assets {
        let source = model.source_file().unwrap_or(model.file.as_str());
        let mut paths = download_hf_assets(&model.name, assets).await?;
        paths.sort();
        if let Some(path) = paths
            .iter()
            .find(|path| path_suffix_matches_ignore_case(path, source))
            .cloned()
        {
            return Ok(path);
        }
        return paths
            .into_iter()
            .find(|path| path_suffix_matches_ignore_case(path, &model.file))
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Downloaded model path not found in cache for {}",
                    model.name
                )
            });
    }

    let dir = models_dir();
    tokio::fs::create_dir_all(&dir).await?;
    let dest = dir.join(&model.file);

    // Collect all files to download: primary + extra splits + mmproj
    let mut files: Vec<(&str, &str)> = vec![(model.file.as_str(), model.url.as_str())];
    for asset in &model.extra_files {
        files.push((asset.file.as_str(), asset.url.as_str()));
    }
    if let Some(asset) = &model.mmproj {
        files.push((asset.file.as_str(), asset.url.as_str()));
    }

    let mut all_present = true;
    let mut total_size: u64 = 0;
    for (file, _) in &files {
        let path = dir.join(file);
        let size = tokio::fs::metadata(&path)
            .await
            .map(|m| m.len())
            .unwrap_or(0);
        if size < 1_000_000 {
            all_present = false;
            break;
        }
        total_size += size;
    }

    if all_present {
        eprintln!(
            "✅ {} already exists ({:.1}GB, {} file{})",
            model.name,
            total_size as f64 / 1e9,
            files.len(),
            if files.len() > 1 { "s" } else { "" },
        );
        return Ok(dest);
    }

    eprintln!("📥 Downloading {} ({})...", model.name, model.size);

    // Collect files that still need downloading
    let mut needed: Vec<(String, String)> = Vec::new();
    for (file, url) in &files {
        let path = dir.join(file);
        if path.exists() {
            let size = tokio::fs::metadata(&path)
                .await
                .map(|m| m.len())
                .unwrap_or(0);
            if size > 1_000_000 {
                eprintln!("  ✅ {file} already exists ({:.1}GB)", size as f64 / 1e9);
                continue;
            }
        }
        needed.push((file.to_string(), url.to_string()));
    }

    if needed.len() > 1 {
        // Parallel download of split files
        eprintln!("  ⚡ Downloading {} files in parallel...", needed.len());
        let total = needed.len();
        let completed = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut handles = Vec::new();
        for (file, url) in needed {
            let path = dir.join(&file);
            let completed = completed.clone();
            handles.push(tokio::spawn(async move {
                download_with_resume(&path, &url).await?;
                let done = completed.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                eprintln!("  ✅ {file} [{done}/{total}]");
                Ok::<(), anyhow::Error>(())
            }));
        }
        for handle in handles {
            handle.await??;
        }
    } else if let Some((file, url)) = needed.into_iter().next() {
        // Single file — just download it
        let path = dir.join(&file);
        download_with_resume(&path, &url).await?;
    }

    eprintln!("✅ Downloaded {} to {}", model.name, dir.display());
    Ok(dest)
}

/// Download any URL to a destination path with resume support.
pub async fn download_url(url: &str, dest: &Path) -> Result<()> {
    download_with_resume(dest, url).await
}

fn path_suffix_matches_ignore_case(path: &Path, expected: &str) -> bool {
    let expected_parts = expected
        .split(['/', '\\'])
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>();

    if expected_parts.is_empty() {
        return false;
    }

    let mut path_parts = path.iter().rev();

    for expected_part in expected_parts.iter().rev() {
        let Some(path_part) = path_parts.next() else {
            return false;
        };

        let Some(path_part) = path_part.to_str() else {
            return false;
        };

        if !path_part.eq_ignore_ascii_case(expected_part) {
            return false;
        }
    }

    true
}

/// Download a HuggingFace GGUF URL, auto-detecting split files (-00001-of-NNNNN.gguf).
/// If the filename matches the split pattern, discovers and downloads all parts in parallel.
/// Returns the path to the first part (or the single file).
pub async fn download_hf_split_gguf(url: &str, filename: &str) -> Result<PathBuf> {
    if let Some(asset) = hf_asset_from_url(url) {
        return download_hf_repo_file(&asset.repo, Some(&asset.revision), &asset.file).await;
    }

    let dir = models_dir();
    tokio::fs::create_dir_all(&dir).await?;

    let re = regex_lite::Regex::new(r"-00001-of-(\d{5})\.gguf$").unwrap();
    if let Some(caps) = re.captures(filename) {
        let n: u32 = caps[1].parse()?;
        eprintln!("📥 Detected split GGUF: {n} parts");

        // Build list of (part_filename, part_url) for all parts
        let mut files: Vec<(String, String)> = Vec::new();
        for i in 1..=n {
            let part_filename = filename.replace("-00001-of-", &format!("-{i:05}-of-"));
            let part_url = url.replace("-00001-of-", &format!("-{i:05}-of-"));
            files.push((part_filename, part_url));
        }

        // Filter to parts that still need downloading
        let mut needed: Vec<(String, String)> = Vec::new();
        for (f, u) in &files {
            let path = dir.join(f);
            if path.exists() {
                let size = tokio::fs::metadata(&path)
                    .await
                    .map(|m| m.len())
                    .unwrap_or(0);
                if size > 1_000_000 {
                    eprintln!("  ✅ {f} already exists ({:.1}GB)", size as f64 / 1e9);
                    continue;
                }
            }
            needed.push((f.clone(), u.clone()));
        }

        if !needed.is_empty() {
            eprintln!("  ⚡ Downloading {} file(s) in parallel...", needed.len());
            let total = needed.len();
            let completed = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let mut handles = Vec::new();
            for (file, url) in needed {
                let path = dir.join(&file);
                let completed = completed.clone();
                handles.push(tokio::spawn(async move {
                    download_with_resume(&path, &url).await?;
                    let done = completed.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                    eprintln!("  ✅ {file} [{done}/{total}]");
                    Ok::<(), anyhow::Error>(())
                }));
            }
            for handle in handles {
                handle.await??;
            }
        }

        return Ok(dir.join(filename));
    }

    // Not a split file — single download
    let dest = dir.join(filename);
    if dest.exists() {
        let size = tokio::fs::metadata(&dest).await?.len();
        if size > 1_000_000 {
            eprintln!(
                "✅ {} already exists ({:.1}GB)",
                filename,
                size as f64 / 1e9
            );
            return Ok(dest);
        }
    }
    eprintln!("📥 Downloading {filename}...");
    download_with_resume(&dest, url).await?;
    Ok(dest)
}

/// Download with resume support and retries using reqwest.
async fn download_with_resume(dest: &Path, url: &str) -> Result<()> {
    use tokio_stream::StreamExt;

    const MAX_ATTEMPTS: u64 = 6;
    let tmp = dest.with_extension("gguf.part");
    let label = dest
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("download");
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(3600)) // 1h overall timeout
        .connect_timeout(std::time::Duration::from_secs(30))
        .build()?;

    let mut attempt: u64 = 1;
    loop {
        // Check how much we already have (for resume)
        let existing_bytes = if tmp.exists() {
            tokio::fs::metadata(&tmp).await?.len()
        } else {
            0
        };

        print_transfer_status(label, existing_bytes, None, attempt, "connecting")?;

        let mut request = client.get(url);
        if existing_bytes > 0 {
            request = request.header("Range", format!("bytes={existing_bytes}-"));
        }

        // Exponential backoff: 1s, 2s, 4s, ... capped at 30s
        let backoff_secs = std::cmp::min(1 * (1u64 << (attempt - 1).min(4)), 30);

        let response = match request.send().await {
            Ok(r) => r,
            Err(e) => {
                eprintln!();
                if attempt >= MAX_ATTEMPTS {
                    anyhow::bail!("Download failed after {MAX_ATTEMPTS} attempts: {e}");
                }
                eprintln!(
                    "  🟡 Download failed ({}). Retrying in {backoff_secs}s (attempt {}/{MAX_ATTEMPTS})",
                    e
                , attempt + 1);
                tokio::time::sleep(std::time::Duration::from_secs(backoff_secs)).await;
                attempt += 1;
                continue;
            }
        };

        let status = response.status();

        // Resume was requested but server ignored Range. Reset and restart from zero.
        if existing_bytes > 0 && status == reqwest::StatusCode::OK {
            let _ = tokio::fs::remove_file(&tmp).await;
            eprintln!();
            eprintln!(
                "  🟡 Existing partial download is not recoverable; clearing and restarting."
            );
            if attempt >= MAX_ATTEMPTS {
                anyhow::bail!(
                    "Download failed after {MAX_ATTEMPTS} attempts: server ignored ranged resume"
                );
            }
            tokio::time::sleep(std::time::Duration::from_secs(backoff_secs)).await;
            attempt += 1;
            continue;
        }

        if status == reqwest::StatusCode::RANGE_NOT_SATISFIABLE {
            let _ = tokio::fs::remove_file(&tmp).await;
            eprintln!();
            eprintln!(
                "  🟡 Existing partial download is not recoverable; clearing and restarting."
            );
            if attempt >= MAX_ATTEMPTS {
                anyhow::bail!(
                    "Download failed after {MAX_ATTEMPTS} attempts: ranged resume rejected"
                );
            }
            tokio::time::sleep(std::time::Duration::from_secs(backoff_secs)).await;
            attempt += 1;
            continue;
        }

        if !status.is_success() && status != reqwest::StatusCode::PARTIAL_CONTENT {
            let retryable = matches!(
                status,
                reqwest::StatusCode::TOO_MANY_REQUESTS
                    | reqwest::StatusCode::INTERNAL_SERVER_ERROR
                    | reqwest::StatusCode::BAD_GATEWAY
                    | reqwest::StatusCode::SERVICE_UNAVAILABLE
                    | reqwest::StatusCode::GATEWAY_TIMEOUT
            );
            if !retryable {
                anyhow::bail!("Download failed: HTTP {}", status);
            }
            if attempt >= MAX_ATTEMPTS {
                anyhow::bail!(
                    "Download failed after {MAX_ATTEMPTS} attempts: HTTP {}",
                    status
                );
            }
            eprintln!();
            eprintln!(
                "  🟡 Download failed (HTTP {status}). Retrying in {backoff_secs}s (attempt {}/{MAX_ATTEMPTS})",
                attempt + 1
            );
            tokio::time::sleep(std::time::Duration::from_secs(backoff_secs)).await;
            attempt += 1;
            continue;
        }

        // Total size from Content-Length (or Content-Range)
        let total_bytes = if status == reqwest::StatusCode::PARTIAL_CONTENT {
            // Content-Range: bytes 1234-5678/9999
            response
                .headers()
                .get("content-range")
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.rsplit('/').next())
                .and_then(|s| s.parse::<u64>().ok())
        } else {
            response.content_length().map(|cl| cl + existing_bytes)
        };

        // On first successful response, check disk space
        if attempt == 1 || existing_bytes == 0 {
            if let Some(total) = total_bytes {
                let remaining = total.saturating_sub(existing_bytes);
                if let Some(free) = free_disk_space(dest) {
                    // Need remaining bytes + 1GB headroom
                    let needed = remaining + 1_000_000_000;
                    if free < needed {
                        anyhow::bail!(
                            "Not enough disk space: need {:.1}GB but only {:.1}GB free on {}",
                            needed as f64 / 1e9,
                            free as f64 / 1e9,
                            dest.parent().unwrap_or(dest).display()
                        );
                    }
                }
            }
        }

        // Open file for append (resume) or create
        let mut file = tokio::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&tmp)
            .await
            .context("Failed to open temp file")?;

        let mut stream = response.bytes_stream();
        let mut downloaded = existing_bytes;
        let mut last_progress = std::time::Instant::now();

        // Print initial progress
        print_transfer_status(label, downloaded, total_bytes, attempt, "downloading")?;

        loop {
            match stream.next().await {
                Some(Ok(chunk)) => {
                    file.write_all(&chunk)
                        .await
                        .context("Failed to write chunk")?;
                    downloaded += chunk.len() as u64;

                    // Update progress every 500ms
                    if last_progress.elapsed() >= std::time::Duration::from_millis(500) {
                        print_transfer_status(
                            label,
                            downloaded,
                            total_bytes,
                            attempt,
                            "downloading",
                        )?;
                        last_progress = std::time::Instant::now();
                    }
                }
                Some(Err(e)) => {
                    file.flush().await.ok();
                    eprintln!();
                    if attempt >= MAX_ATTEMPTS {
                        anyhow::bail!(
                            "Download failed after {MAX_ATTEMPTS} attempts: interrupted at {} ({e})",
                            format_size_bytes(downloaded)
                        );
                    }
                    let retry_secs = std::cmp::min(1 * (1u64 << (attempt - 1).min(4)), 30);
                    eprintln!(
                        "  🟡 Download failed ({}). Retrying in {retry_secs}s (attempt {}/{MAX_ATTEMPTS})",
                        e,
                        attempt + 1
                    );
                    tokio::time::sleep(std::time::Duration::from_secs(retry_secs)).await;
                    attempt += 1;
                    break;
                }
                None => {
                    // Stream complete
                    file.flush().await?;
                    print_transfer_status(label, downloaded, total_bytes, attempt, "complete")?;
                    eprintln!();
                    tokio::fs::rename(&tmp, dest)
                        .await
                        .context("Failed to move downloaded file")?;
                    return Ok(());
                }
            }
        }
    }
}

/// Check free disk space on the filesystem containing `path`.
/// Returns None if the check fails (e.g. path doesn't exist yet).
fn free_disk_space(path: &Path) -> Option<u64> {
    // Walk up to find an existing directory for statvfs
    let mut check = path.to_path_buf();
    loop {
        if check.exists() {
            break;
        }
        if !check.pop() {
            return None;
        }
    }
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        let c_path = std::ffi::CString::new(check.as_os_str().as_bytes()).ok()?;
        let mut stat: libc::statvfs = unsafe { std::mem::zeroed() };
        let ret = unsafe { libc::statvfs(c_path.as_ptr(), &mut stat) };
        if ret == 0 {
            // f_bavail = blocks available to unprivileged users
            Some(stat.f_bavail as u64 * stat.f_frsize as u64)
        } else {
            None
        }
    }
    #[cfg(not(unix))]
    {
        None
    }
}

fn print_transfer_status(
    label: &str,
    downloaded: u64,
    total: Option<u64>,
    attempt: u64,
    phase: &str,
) -> Result<()> {
    let progress = match total {
        Some(total) if total > 0 => format!(
            "{:>5.1}%  {}/{}",
            (downloaded as f64 / total as f64) * 100.0,
            format_size_bytes(downloaded),
            format_size_bytes(total)
        ),
        _ => format!("      {}", format_size_bytes(downloaded)),
    };
    let resume = if attempt > 1 {
        format!("  attempt {}", attempt)
    } else {
        String::new()
    };
    eprint!("\r  📥 {}  {:<11} {}{}", label, phase, progress, resume);
    std::io::stderr()
        .flush()
        .context("Flush transfer progress")?;
    Ok(())
}

fn format_size_bytes(bytes: u64) -> String {
    if bytes >= 1_000_000_000 {
        format!("{:.1}GB", bytes as f64 / 1e9)
    } else {
        format!("{:.0}MB", bytes as f64 / 1e6)
    }
}

/// List available models
pub fn list_models() {
    eprintln!("Available models:");
    eprintln!();
    for m in MODEL_CATALOG.iter() {
        let draft_info = if let Some(d) = m.draft.as_deref() {
            format!(" (draft: {})", d)
        } else {
            String::new()
        };
        eprintln!(
            "  {:40} {:>6}  {}{}",
            m.name, m.size, m.description, draft_info
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;
    use tokio::sync::Mutex;

    struct TestRequest {
        range: Option<String>,
    }

    async fn read_http_request(stream: &mut tokio::net::TcpStream) -> anyhow::Result<TestRequest> {
        let mut buf = vec![0u8; 4096];
        let mut read = 0usize;
        loop {
            if read == buf.len() {
                buf.resize(buf.len() * 2, 0);
            }
            let n = stream.read(&mut buf[read..]).await?;
            if n == 0 {
                break;
            }
            read += n;
            if read >= 4 && buf[..read].windows(4).any(|w| w == b"\r\n\r\n") {
                break;
            }
        }
        let head = String::from_utf8_lossy(&buf[..read]);
        let mut range = None;
        for line in head.lines() {
            if let Some((name, value)) = line.split_once(':') {
                if name.eq_ignore_ascii_case("range") {
                    range = Some(value.trim().to_string());
                    break;
                }
            }
        }
        Ok(TestRequest { range })
    }

    fn temp_download_dest(prefix: &str) -> PathBuf {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir()
            .join(format!("mesh-llm-download-test-{prefix}-{unique}"))
            .join("model.gguf")
    }

    #[test]
    fn source_identity_is_exposed_for_hf_catalog_entries() {
        let model = find_model("Qwen3-8B-Q4_K_M").unwrap();
        assert_eq!(model.source_repo(), Some("unsloth/Qwen3-8B-GGUF"));
        assert_eq!(model.source_revision(), Some("main"));
        assert_eq!(model.source_file(), Some("Qwen3-8B-Q4_K_M.gguf"));
        assert!(model.source_repo().is_some());
    }

    #[test]
    fn source_identity_is_absent_for_direct_url_entries() {
        let model = find_model("Qwen3.5-27B-Q4_K_M").unwrap();
        assert_eq!(model.source_repo(), None);
        assert_eq!(model.source_revision(), None);
        assert_eq!(model.source_file(), None);
        assert!(model.source_repo().is_none());
    }

    #[test]
    fn test_free_disk_space() {
        let path = std::env::temp_dir().join("test_file.gguf");
        let free = free_disk_space(&path);

        #[cfg(unix)]
        {
            assert!(
                free.is_some(),
                "should get free space for {}",
                path.display()
            );
            let bytes = free.unwrap();
            assert!(bytes > 1_000_000_000, "should have >1GB free, got {bytes}");
        }

        #[cfg(not(unix))]
        {
            assert!(free.is_none(), "non-unix builds should skip statvfs checks");
        }
    }

    #[test]
    fn test_split_gguf_detection() {
        let re = regex_lite::Regex::new(r"-00001-of-(\d{5})\.gguf$").unwrap();

        // Should match split GGUFs
        let caps = re.captures("Model-Q4_K_M-00001-of-00004.gguf");
        assert!(caps.is_some());
        assert_eq!(&caps.unwrap()[1], "00004");

        let caps = re.captures("Qwen3-Coder-Next-Q4_K_M-00001-of-00004.gguf");
        assert!(caps.is_some());

        let caps = re.captures("MiniMax-M2.5-Q4_K_M-00001-of-00004.gguf");
        assert!(caps.is_some());

        // Should NOT match non-split or other parts
        assert!(re.captures("Model-Q4_K_M.gguf").is_none());
        assert!(re.captures("Model-Q4_K_M-00002-of-00004.gguf").is_none());
        assert!(re.captures("Model-Q4_K_M-00001-of-00004.bin").is_none());
    }

    #[test]
    fn test_split_url_generation() {
        let filename = "Model-Q4_K_M-00001-of-00003.gguf";
        let url = "https://huggingface.co/org/repo/resolve/main/Model-Q4_K_M-00001-of-00003.gguf";

        let mut files = Vec::new();
        for i in 1..=3u32 {
            let part_filename = filename.replace("-00001-of-", &format!("-{i:05}-of-"));
            let part_url = url.replace("-00001-of-", &format!("-{i:05}-of-"));
            files.push((part_filename, part_url));
        }

        assert_eq!(files.len(), 3);
        assert_eq!(files[0].0, "Model-Q4_K_M-00001-of-00003.gguf");
        assert_eq!(files[1].0, "Model-Q4_K_M-00002-of-00003.gguf");
        assert_eq!(files[2].0, "Model-Q4_K_M-00003-of-00003.gguf");
        assert!(files[0].1.contains("-00001-of-"));
        assert!(files[1].1.contains("-00002-of-"));
        assert!(files[2].1.contains("-00003-of-"));
    }

    #[test]
    fn expand_split_asset_expands_all_parts_from_first_shard() {
        let asset = HfAsset {
            repo: "org/repo".to_string(),
            revision: "main".to_string(),
            file: "MiniMax-M2-HQ4_K-00001-of-00004.gguf".to_string(),
        };
        let expanded = expand_split_asset(&asset).unwrap();
        assert_eq!(expanded.len(), 4);
        assert_eq!(expanded[0].file, "MiniMax-M2-HQ4_K-00001-of-00004.gguf");
        assert_eq!(expanded[3].file, "MiniMax-M2-HQ4_K-00004-of-00004.gguf");
    }

    #[test]
    fn download_plan_expands_split_shorthand_to_all_shards() {
        let plan = build_download_plan(vec![HfAsset {
            repo: "anikifoss/MiniMax-M2-HQ4_K".to_string(),
            revision: "main".to_string(),
            file: "MiniMax-M2-HQ4_K-00001-of-00004.gguf".to_string(),
        }])
        .unwrap();
        let required_files: Vec<String> = plan
            .iter()
            .filter_map(|(required, asset)| {
                if *required {
                    Some(asset.file.clone())
                } else {
                    None
                }
            })
            .collect();

        assert_eq!(required_files.len(), 4);
        assert!(required_files.contains(&"MiniMax-M2-HQ4_K-00001-of-00004.gguf".to_string()));
        assert!(required_files.contains(&"MiniMax-M2-HQ4_K-00002-of-00004.gguf".to_string()));
        assert!(required_files.contains(&"MiniMax-M2-HQ4_K-00003-of-00004.gguf".to_string()));
        assert!(required_files.contains(&"MiniMax-M2-HQ4_K-00004-of-00004.gguf".to_string()));
        assert!(plan
            .iter()
            .any(|(required, asset)| !required && asset.file == "config.json"));
    }

    #[test]
    fn path_file_name_matches_nested_path_ignore_case() {
        let path = Path::new("/tmp/cache/Subdir/Model.Q4_K_M.gguf");
        assert!(path_suffix_matches_ignore_case(
            path,
            "subdir/model.q4_k_m.gguf"
        ));
    }

    #[test]
    fn path_file_name_matches_rejects_wrong_suffix() {
        let path = Path::new("/tmp/cache/other/Model.Q4_K_M.gguf");
        assert!(!path_suffix_matches_ignore_case(
            path,
            "subdir/model.q4_k_m.gguf"
        ));
    }

    #[tokio::test]
    async fn download_hf_repo_file_matches_cache_file_case_insensitively() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let cached_file = std::env::temp_dir()
            .join(format!("mesh-llm-hf-case-repo-{unique}"))
            .join("qwen2.5-coder-7b-instruct-q4_k_m.gguf");
        std::fs::create_dir_all(cached_file.parent().unwrap()).unwrap();
        std::fs::write(&cached_file, b"gguf").unwrap();

        {
            let label =
                "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF/Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf@main"
                    .to_string();
            let _guard = DownloadHfAssetsOverrideGuard::set(
                label,
                Arc::new({
                    let cached = cached_file.clone();
                    move |_, _| Ok(vec![cached.clone()])
                }),
            );
            let resolved = download_hf_repo_file(
                "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
                Some("main"),
                "Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf",
            )
            .await
            .unwrap();
            assert_eq!(resolved, cached_file);
        }

        {
            let label =
                "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF/Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf@main"
                    .to_string();
            let _guard = DownloadHfAssetsOverrideGuard::set(label, Arc::new(|_, _| Ok(Vec::new())));
            assert!(download_hf_repo_file(
                "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
                Some("main"),
                "Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf",
            )
            .await
            .is_err());
        }
    }

    #[tokio::test]
    async fn download_hf_repo_file_matches_nested_cache_path_case_insensitively() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let cached_file = std::env::temp_dir()
            .join(format!("mesh-llm-hf-nested-repo-{unique}"))
            .join("nested")
            .join("Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf");
        std::fs::create_dir_all(cached_file.parent().unwrap()).unwrap();
        std::fs::write(&cached_file, b"gguf").unwrap();

        let label =
            "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF/nested/qwen2.5-coder-7b-instruct-q4_k_m.gguf@main"
                .to_string();
        let _guard = DownloadHfAssetsOverrideGuard::set(
            label,
            Arc::new({
                let cached = cached_file.clone();
                move |_, _| Ok(vec![cached.clone()])
            }),
        );

        let resolved = download_hf_repo_file(
            "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
            Some("main"),
            "nested/qwen2.5-coder-7b-instruct-q4_k_m.gguf",
        )
        .await
        .unwrap();
        assert_eq!(resolved, cached_file);
    }

    #[tokio::test]
    async fn download_model_matches_hf_cache_file_case_insensitively() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let cached_file = std::env::temp_dir()
            .join(format!("mesh-llm-hf-case-model-{unique}"))
            .join("qwen2.5-coder-7b-instruct-q4_k_m.gguf");
        std::fs::create_dir_all(cached_file.parent().unwrap()).unwrap();
        std::fs::write(&cached_file, b"gguf").unwrap();

        let model = CatalogModel {
            name: "Qwen2.5-Coder-7B-Instruct-Q4_K_M".to_string(),
            file: "Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf".to_string(),
            url: "https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF/resolve/main/qwen2.5-coder-7b-instruct-q4_k_m.gguf".to_string(),
            size: "4.4GB".to_string(),
            description: "".to_string(),
            draft: None,
            moe: None,
            extra_files: Vec::new(),
            mmproj: None,
        };

        {
            let label = model.name.clone();
            let _guard = DownloadHfAssetsOverrideGuard::set(
                label,
                Arc::new({
                    let cached = cached_file.clone();
                    move |_, _| Ok(vec![cached.clone()])
                }),
            );
            let resolved = download_model(&model).await.unwrap();
            assert_eq!(resolved, cached_file);
        }

        {
            let label = model.name.clone();
            let _guard = DownloadHfAssetsOverrideGuard::set(label, Arc::new(|_, _| Ok(Vec::new())));
            assert!(download_model(&model).await.is_err());
        }
    }

    #[tokio::test]
    async fn download_model_prefers_nested_source_path_over_same_basename() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("mesh-llm-hf-nested-model-{unique}"));
        let wrong_file = root.join("another").join("model.gguf");
        let expected_file = root.join("nested").join("model.gguf");
        std::fs::create_dir_all(wrong_file.parent().unwrap()).unwrap();
        std::fs::create_dir_all(expected_file.parent().unwrap()).unwrap();
        std::fs::write(&wrong_file, b"wrong").unwrap();
        std::fs::write(&expected_file, b"right").unwrap();

        let model = CatalogModel {
            name: "Nested-Path-Model".to_string(),
            file: "model.gguf".to_string(),
            url: "https://huggingface.co/org/repo/resolve/main/nested/MODEL.gguf".to_string(),
            size: "1GB".to_string(),
            description: "".to_string(),
            draft: None,
            moe: None,
            extra_files: Vec::new(),
            mmproj: None,
        };

        let label = model.name.clone();
        let _guard = DownloadHfAssetsOverrideGuard::set(
            label,
            Arc::new({
                let wrong = wrong_file.clone();
                let expected = expected_file.clone();
                move |_, _| Ok(vec![wrong.clone(), expected.clone()])
            }),
        );

        let resolved = download_model(&model).await.unwrap();
        assert_eq!(resolved, expected_file);
    }

    #[tokio::test]
    async fn download_with_resume_retries_transient_status_and_succeeds() {
        let payload = Arc::new(vec![b'x'; 32 * 1024]);
        let request_count = Arc::new(AtomicUsize::new(0));
        let seen_ranges = Arc::new(Mutex::new(Vec::<Option<String>>::new()));

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let payload_for_task = payload.clone();
        let request_count_for_task = request_count.clone();
        let seen_ranges_for_task = seen_ranges.clone();
        let server = tokio::spawn(async move {
            for _ in 0..2 {
                let (mut stream, _) = listener.accept().await.unwrap();
                let req = read_http_request(&mut stream).await.unwrap();
                seen_ranges_for_task.lock().await.push(req.range);
                let n = request_count_for_task.fetch_add(1, Ordering::SeqCst);
                if n == 0 {
                    stream
                        .write_all(b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\n\r\n")
                        .await
                        .unwrap();
                } else {
                    let header = format!(
                        "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nAccept-Ranges: bytes\r\n\r\n",
                        payload_for_task.len()
                    );
                    stream.write_all(header.as_bytes()).await.unwrap();
                    stream.write_all(&payload_for_task).await.unwrap();
                }
            }
        });

        let dest = temp_download_dest("retry-status");
        std::fs::create_dir_all(dest.parent().unwrap()).unwrap();
        let url = format!("http://{addr}/model.gguf");
        download_with_resume(&dest, &url).await.unwrap();
        server.await.unwrap();

        let downloaded = std::fs::read(&dest).unwrap();
        assert_eq!(downloaded, *payload);
        assert_eq!(request_count.load(Ordering::SeqCst), 2);
        let ranges = seen_ranges.lock().await.clone();
        assert_eq!(ranges, vec![None, None]);
    }

    #[tokio::test]
    async fn download_with_resume_recovers_from_interrupted_stream() {
        let payload = Arc::new((0..200_000u32).map(|v| (v % 251) as u8).collect::<Vec<_>>());
        let cut = 80_000usize;
        let request_count = Arc::new(AtomicUsize::new(0));
        let seen_ranges = Arc::new(Mutex::new(Vec::<Option<String>>::new()));

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let payload_for_task = payload.clone();
        let request_count_for_task = request_count.clone();
        let seen_ranges_for_task = seen_ranges.clone();
        let server = tokio::spawn(async move {
            for _ in 0..2 {
                let (mut stream, _) = listener.accept().await.unwrap();
                let req = read_http_request(&mut stream).await.unwrap();
                seen_ranges_for_task.lock().await.push(req.range.clone());
                let n = request_count_for_task.fetch_add(1, Ordering::SeqCst);
                if n == 0 {
                    let header = format!(
                        "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nAccept-Ranges: bytes\r\n\r\n",
                        payload_for_task.len()
                    );
                    stream.write_all(header.as_bytes()).await.unwrap();
                    stream.write_all(&payload_for_task[..cut]).await.unwrap();
                    let _ = stream.shutdown().await;
                } else {
                    let expected_range = format!("bytes={cut}-");
                    assert_eq!(req.range.as_deref(), Some(expected_range.as_str()));
                    let total = payload_for_task.len();
                    let rest = &payload_for_task[cut..];
                    let header = format!(
                        "HTTP/1.1 206 Partial Content\r\nContent-Length: {}\r\nContent-Range: bytes {}-{}/{}\r\nAccept-Ranges: bytes\r\n\r\n",
                        rest.len(),
                        cut,
                        total - 1,
                        total
                    );
                    stream.write_all(header.as_bytes()).await.unwrap();
                    stream.write_all(rest).await.unwrap();
                }
            }
        });

        let dest = temp_download_dest("resume-interrupt");
        std::fs::create_dir_all(dest.parent().unwrap()).unwrap();
        let url = format!("http://{addr}/model.gguf");
        download_with_resume(&dest, &url).await.unwrap();
        server.await.unwrap();

        let downloaded = std::fs::read(&dest).unwrap();
        assert_eq!(downloaded, *payload);
        assert_eq!(request_count.load(Ordering::SeqCst), 2);
        let ranges = seen_ranges.lock().await.clone();
        assert_eq!(ranges, vec![None, Some(format!("bytes={cut}-"))]);
    }

    #[tokio::test]
    async fn download_with_resume_clears_unrecoverable_partial_and_restarts() {
        let payload = Arc::new(vec![b'z'; 48 * 1024]);
        let stale_partial = vec![b'p'; 12 * 1024];
        let request_count = Arc::new(AtomicUsize::new(0));
        let seen_ranges = Arc::new(Mutex::new(Vec::<Option<String>>::new()));

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let payload_for_task = payload.clone();
        let request_count_for_task = request_count.clone();
        let seen_ranges_for_task = seen_ranges.clone();
        let server = tokio::spawn(async move {
            for _ in 0..2 {
                let (mut stream, _) = listener.accept().await.unwrap();
                let req = read_http_request(&mut stream).await.unwrap();
                seen_ranges_for_task.lock().await.push(req.range.clone());
                let n = request_count_for_task.fetch_add(1, Ordering::SeqCst);
                if n == 0 {
                    assert!(req.range.is_some());
                }
                let header = format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nAccept-Ranges: bytes\r\n\r\n",
                    payload_for_task.len()
                );
                stream.write_all(header.as_bytes()).await.unwrap();
                stream.write_all(&payload_for_task).await.unwrap();
            }
        });

        let dest = temp_download_dest("reset-unrecoverable");
        std::fs::create_dir_all(dest.parent().unwrap()).unwrap();
        let tmp = dest.with_extension("gguf.part");
        std::fs::write(&tmp, &stale_partial).unwrap();

        let url = format!("http://{addr}/model.gguf");
        download_with_resume(&dest, &url).await.unwrap();
        server.await.unwrap();

        let downloaded = std::fs::read(&dest).unwrap();
        assert_eq!(downloaded, *payload);
        assert_eq!(request_count.load(Ordering::SeqCst), 2);
        let ranges = seen_ranges.lock().await.clone();
        assert!(ranges[0].is_some());
        assert_eq!(ranges[1], None);
    }
}
