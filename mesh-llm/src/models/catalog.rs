//! Built-in model catalog plus managed acquisition helpers.

use anyhow::{Context, Result};
use hf_hub::api::Progress as HfProgress;
use hf_hub::{Repo, RepoType};
use serde::Deserialize;
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

fn is_mlx_primary_asset(file: &str) -> bool {
    matches!(file, "model.safetensors" | "model.safetensors.index.json")
        || is_split_mlx_first_shard_file(file)
}

/// Returns true if `file` is the first shard of a sharded MLX safetensors set,
/// i.e. `model-00001-of-NNNNN.safetensors`.
fn is_split_mlx_first_shard_file(file: &str) -> bool {
    let Some(rest) = file.strip_prefix("model-") else {
        return false;
    };
    let Some(rest) = rest.strip_suffix(".safetensors") else {
        return false;
    };
    let Some((left, right)) = rest.split_once("-of-") else {
        return false;
    };
    left == "00001" && right.len() == 5 && right.bytes().all(|b| b.is_ascii_digit())
}

/// Expands a first-shard MLX ref (`model-00001-of-NNNNN.safetensors`) into the
/// full list of shard assets without needing to download the index.
fn expand_split_mlx_first_shard(asset: &HfAsset) -> Vec<HfAsset> {
    let Some(rest) = asset.file.strip_prefix("model-00001-of-") else {
        return Vec::new();
    };
    let Some(total_str) = rest.strip_suffix(".safetensors") else {
        return Vec::new();
    };
    if total_str.len() != 5 || !total_str.bytes().all(|b| b.is_ascii_digit()) {
        return Vec::new();
    }
    let Ok(count): Result<u32> = total_str.parse().map_err(anyhow::Error::from) else {
        return Vec::new();
    };
    (1..=count)
        .map(|index| HfAsset {
            repo: asset.repo.clone(),
            revision: asset.revision.clone(),
            file: format!("model-{index:05}-of-{total_str}.safetensors"),
        })
        .collect()
}

fn mlx_sidecar_assets(asset: &HfAsset) -> Vec<(bool, HfAsset)> {
    [
        (true, "tokenizer.json"),
        (false, "tokenizer_config.json"),
        (false, "chat_template.jinja"),
        (false, "chat_template.json"),
    ]
    .into_iter()
    .map(|(required, file)| {
        (
            required,
            HfAsset {
                repo: asset.repo.clone(),
                revision: asset.revision.clone(),
                file: file.to_string(),
            },
        )
    })
    .collect()
}

fn is_optional_metadata(required: bool, _asset: &HfAsset) -> bool {
    !required
}

fn parse_safetensors_index_shards(index: &serde_json::Value) -> Result<Vec<String>> {
    let weight_map = index["weight_map"]
        .as_object()
        .context("missing weight_map in safetensors index")?;
    let mut shards = std::collections::BTreeSet::new();
    for file in weight_map.values() {
        let file = file
            .as_str()
            .context("weight_map value in safetensors index is not a string")?;
        shards.insert(file.to_string());
    }
    Ok(shards.into_iter().collect())
}

fn ensure_cached_hf_asset(
    api: &hf_hub::api::sync::Api,
    cache: &hf_hub::Cache,
    asset: &HfAsset,
) -> Result<PathBuf> {
    let repo_handle = asset.repo_handle();
    let cache_repo = cache.repo(repo_handle.clone());
    if let Some(path) = cache_repo.get(&asset.file) {
        return Ok(path);
    }
    api.repo(repo_handle)
        .download(&asset.file)
        .with_context(|| {
            format!(
                "Cache Hugging Face asset {}/{}@{}",
                asset.repo, asset.file, asset.revision
            )
        })
}

fn mlx_sharded_weight_assets(
    api: &hf_hub::api::sync::Api,
    cache: &hf_hub::Cache,
    asset: &HfAsset,
) -> Result<Vec<HfAsset>> {
    if asset.file != "model.safetensors.index.json" {
        return Ok(Vec::new());
    }
    let index_path = ensure_cached_hf_asset(api, cache, asset)?;
    let index_text = std::fs::read_to_string(&index_path)
        .with_context(|| format!("Read {}", index_path.display()))?;
    let index: serde_json::Value = serde_json::from_str(&index_text)
        .with_context(|| format!("Parse {}", index_path.display()))?;
    Ok(parse_safetensors_index_shards(&index)?
        .into_iter()
        .map(|file| HfAsset {
            repo: asset.repo.clone(),
            revision: asset.revision.clone(),
            file,
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

async fn download_hf_assets(
    label: &str,
    assets: Vec<HfAsset>,
    progress: bool,
) -> Result<Vec<PathBuf>> {
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
    tokio::task::spawn_blocking(move || download_hf_assets_blocking(&label, assets, progress))
        .await
        .context("Join Hugging Face download task")?
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

fn download_hf_assets_blocking(
    label: &str,
    assets: Vec<HfAsset>,
    progress: bool,
) -> Result<Vec<PathBuf>> {
    let api = super::build_hf_api(false)?;
    let cache = crate::models::huggingface_hub_cache();
    let mut download_plan = std::collections::BTreeSet::new();
    let mut config_repos = std::collections::BTreeSet::new();

    for asset in assets {
        for expanded in expand_split_asset(&asset)? {
            config_repos.insert((expanded.repo.clone(), expanded.revision.clone()));
            download_plan.insert((true, expanded));
        }
    }
    let current_plan: Vec<(bool, HfAsset)> = download_plan.iter().cloned().collect();
    for (_, asset) in current_plan {
        if !is_mlx_primary_asset(&asset.file) {
            continue;
        }
        for sidecar in mlx_sidecar_assets(&asset) {
            download_plan.insert(sidecar);
        }
        // Expand shards from an index file (downloads index to discover shard names)
        for shard in mlx_sharded_weight_assets(&api, &cache, &asset)? {
            download_plan.insert((true, shard));
        }
        // Expand shards from a first-shard ref without needing to download the index
        for shard in expand_split_mlx_first_shard(&asset) {
            download_plan.insert((true, shard));
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

    if progress {
        eprintln!("📥 Ensuring {} is available locally...", label);
    }

    let mut primary_paths = Vec::new();
    for (required, asset) in download_plan {
        let repo_handle = asset.repo_handle();
        let cache_repo = cache.repo(repo_handle.clone());
        let api_repo = api.repo(repo_handle);
        let path = match cache_repo.get(&asset.file) {
            Some(path) => {
                if progress {
                    if required {
                        eprintln!("   ✅ Using cached model {}", asset.file);
                    } else {
                        eprintln!("   🧾 Using cached model metadata");
                    }
                }
                path
            }
            None => {
                if progress && required {
                    eprintln!("   📥 Downloading model {}", asset.file);
                }
                match if required {
                    if progress {
                        api_repo.download_with_progress(&asset.file, MeshDownloadProgress::new())
                    } else {
                        api_repo.download(&asset.file)
                    }
                } else {
                    api_repo.download(&asset.file)
                } {
                    Ok(path) => {
                        if progress {
                            if required {
                                eprintln!("   ✅ Ready {}", asset.file);
                            } else {
                                eprintln!("   🧾 Downloaded model metadata");
                            }
                        }
                        path
                    }
                    Err(_) if is_optional_metadata(required, &asset) => {
                        continue;
                    }
                    Err(err) => {
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

pub async fn download_hf_repo_file(
    repo: &str,
    revision: Option<&str>,
    file: &str,
) -> Result<PathBuf> {
    download_hf_repo_file_with_progress(repo, revision, file, true).await
}

pub async fn download_hf_repo_file_with_progress(
    repo: &str,
    revision: Option<&str>,
    file: &str,
    progress: bool,
) -> Result<PathBuf> {
    let revision = revision.unwrap_or("main").to_string();
    let asset = HfAsset {
        repo: repo.to_string(),
        revision: revision.clone(),
        file: file.to_string(),
    };
    let mut paths = download_hf_assets(
        &format!("{repo}/{file}@{revision}"),
        vec![asset.clone()],
        progress,
    )
    .await?;
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
    download_model_with_progress(model, true).await
}

pub async fn download_model_with_progress(model: &CatalogModel, progress: bool) -> Result<PathBuf> {
    let hf_assets: Option<Vec<HfAsset>> = std::iter::once(model.url.as_str())
        .chain(model.extra_files.iter().map(|asset| asset.url.as_str()))
        .chain(model.mmproj.iter().map(|asset| asset.url.as_str()))
        .map(hf_asset_from_url)
        .collect();
    if let Some(assets) = hf_assets {
        let source = model.source_file().unwrap_or(model.file.as_str());
        let mut paths = download_hf_assets(&model.name, assets, progress).await?;
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
        if progress {
            eprintln!(
                "✅ {} already exists ({:.1}GB, {} file{})",
                model.name,
                total_size as f64 / 1e9,
                files.len(),
                if files.len() > 1 { "s" } else { "" },
            );
        }
        return Ok(dest);
    }

    if progress {
        eprintln!("📥 Downloading {} ({})...", model.name, model.size);
    }

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
                if progress {
                    eprintln!("  ✅ {file} already exists ({:.1}GB)", size as f64 / 1e9);
                }
                continue;
            }
        }
        needed.push((file.to_string(), url.to_string()));
    }

    if needed.len() > 1 {
        // Parallel download of split files
        if progress {
            eprintln!("  ⚡ Downloading {} files in parallel...", needed.len());
        }
        let total = needed.len();
        let completed = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut handles = Vec::new();
        for (file, url) in needed {
            let path = dir.join(&file);
            let progress = progress;
            let completed = completed.clone();
            handles.push(tokio::spawn(async move {
                download_with_resume(&path, &url).await?;
                let done = completed.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                if progress {
                    eprintln!("  ✅ {file} [{done}/{total}]");
                }
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

    if progress {
        eprintln!("✅ Downloaded {} to {}", model.name, dir.display());
    }
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

    let tmp = dest.with_extension("gguf.part");
    let label = dest
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("download");
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(3600)) // 1h overall timeout
        .connect_timeout(std::time::Duration::from_secs(30))
        .build()?;

    let mut attempt: u64 = 0;
    loop {
        attempt += 1;
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

        // Exponential backoff: 3s, 6s, 12s, ... capped at 60s
        let backoff_secs = std::cmp::min(3 * (1u64 << (attempt - 1).min(4)), 60);

        let response = match request.send().await {
            Ok(r) => r,
            Err(e) => {
                eprintln!();
                eprintln!("  ⚠️ {label}: connection failed: {e}");
                eprintln!("  retrying in {backoff_secs}s...");
                tokio::time::sleep(std::time::Duration::from_secs(backoff_secs)).await;
                continue;
            }
        };

        let status = response.status();
        if !status.is_success() && status != reqwest::StatusCode::PARTIAL_CONTENT {
            // If server doesn't support resume (416 Range Not Satisfiable), start fresh
            if status == reqwest::StatusCode::RANGE_NOT_SATISFIABLE {
                let _ = tokio::fs::remove_file(&tmp).await;
                eprintln!();
                eprintln!("  ⚠️ {label}: server rejected resume, starting fresh...");
                continue;
            }
            eprintln!();
            eprintln!("  ⚠️ {label}: HTTP {status}, retrying in {backoff_secs}s...");
            tokio::time::sleep(std::time::Duration::from_secs(backoff_secs)).await;
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

        // Reset backoff on successful data receipt
        let mut got_data = false;

        loop {
            match stream.next().await {
                Some(Ok(chunk)) => {
                    file.write_all(&chunk)
                        .await
                        .context("Failed to write chunk")?;
                    downloaded += chunk.len() as u64;
                    got_data = true;

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
                    eprintln!(
                        "  ⚠️ {label}: interrupted at {}: {e}",
                        format_size_bytes(downloaded)
                    );
                    // If we got data, reset attempt counter (connection was working)
                    if got_data {
                        attempt = 0;
                    }
                    let retry_secs = std::cmp::min(3 * (1u64 << attempt.min(4)), 60);
                    eprintln!("  retrying in {retry_secs}s (will resume)...");
                    tokio::time::sleep(std::time::Duration::from_secs(retry_secs)).await;
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
            Some(stat.f_bavail as u64 * stat.f_frsize)
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

    #[test]
    fn mlx_sidecars_include_required_tokenizer_and_optional_templates() {
        let asset = HfAsset {
            repo: "mlx-community/qwen2.5-0.5b-instruct-q2".to_string(),
            revision: "main".to_string(),
            file: "model.safetensors".to_string(),
        };
        let sidecars = mlx_sidecar_assets(&asset);
        assert_eq!(sidecars.len(), 4);
        assert!(sidecars[0].0);
        assert_eq!(sidecars[0].1.file, "tokenizer.json");
        assert!(sidecars
            .iter()
            .any(|(_, a)| a.file == "tokenizer_config.json"));
        assert!(sidecars
            .iter()
            .any(|(_, a)| a.file == "chat_template.jinja"));
        assert!(sidecars.iter().any(|(_, a)| a.file == "chat_template.json"));
    }

    #[test]
    fn parse_safetensors_index_shards_extracts_unique_shards() {
        let index = serde_json::json!({
            "weight_map": {
                "layer.0.q": "model-00001-of-00002.safetensors",
                "layer.0.k": "model-00001-of-00002.safetensors",
                "layer.1.q": "model-00002-of-00002.safetensors"
            }
        });
        let shards = parse_safetensors_index_shards(&index).unwrap();
        assert_eq!(
            shards,
            vec![
                "model-00001-of-00002.safetensors".to_string(),
                "model-00002-of-00002.safetensors".to_string()
            ]
        );
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

    #[test]
    fn is_split_mlx_first_shard_file_identifies_correct_patterns() {
        assert!(is_split_mlx_first_shard_file(
            "model-00001-of-00004.safetensors"
        ));
        assert!(is_split_mlx_first_shard_file(
            "model-00001-of-00048.safetensors"
        ));
        assert!(!is_split_mlx_first_shard_file(
            "model-00002-of-00004.safetensors"
        ));
        assert!(!is_split_mlx_first_shard_file("model.safetensors"));
        assert!(!is_split_mlx_first_shard_file(
            "model.safetensors.index.json"
        ));
        assert!(!is_split_mlx_first_shard_file("model-00001-of-00004.gguf"));
    }

    #[test]
    fn expand_split_mlx_first_shard_generates_all_shards() {
        let asset = HfAsset {
            repo: "org/repo".to_string(),
            revision: "main".to_string(),
            file: "model-00001-of-00003.safetensors".to_string(),
        };
        let shards = expand_split_mlx_first_shard(&asset);
        assert_eq!(shards.len(), 3);
        assert_eq!(shards[0].file, "model-00001-of-00003.safetensors");
        assert_eq!(shards[1].file, "model-00002-of-00003.safetensors");
        assert_eq!(shards[2].file, "model-00003-of-00003.safetensors");
        for shard in &shards {
            assert_eq!(shard.repo, "org/repo");
            assert_eq!(shard.revision, "main");
        }
    }

    #[test]
    fn expand_split_mlx_first_shard_returns_empty_for_non_first_shard() {
        let asset = HfAsset {
            repo: "org/repo".to_string(),
            revision: "main".to_string(),
            file: "model-00002-of-00003.safetensors".to_string(),
        };
        let shards = expand_split_mlx_first_shard(&asset);
        assert!(shards.is_empty());
    }

    #[test]
    fn is_mlx_primary_asset_includes_first_shard() {
        assert!(is_mlx_primary_asset("model.safetensors"));
        assert!(is_mlx_primary_asset("model.safetensors.index.json"));
        assert!(is_mlx_primary_asset("model-00001-of-00048.safetensors"));
        assert!(!is_mlx_primary_asset("model-00002-of-00048.safetensors"));
        assert!(!is_mlx_primary_asset("model-00048-of-00048.safetensors"));
    }
}
