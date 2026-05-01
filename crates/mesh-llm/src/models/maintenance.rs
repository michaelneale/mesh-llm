use super::{build_hf_api, huggingface_hub_cache_dir, short_revision};
use crate::cli::terminal_progress::{clear_stderr_line, DeterminateProgressLine};
use anyhow::{Context, Result};
use hf_hub::{RepoDownloadFileParams, RepoInfo, RepoInfoParams, RepoType};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

struct CachedRepo {
    repo_id: String,
    ref_name: String,
    local_revision: String,
}

#[derive(Default)]
struct UpdateCounts {
    refreshed: usize,
    missing_meta: usize,
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
                eprintln!();
            }
        } else {
            eprintln!("🧭 [{}/{}] {}", index + 1, total_selected, repo.repo_id);
            let counts = update_cached_repo(&api, &repo)?;
            refresh_totals.refreshed += counts.refreshed;
            refresh_totals.missing_meta += counts.missing_meta;
            eprintln!();
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

fn print_update_check_progress(current: usize, total: usize, repo_id: &str) -> Result<()> {
    DeterminateProgressLine::new("🔄").draw_counts(
        "Checking updates",
        current,
        total,
        Some(&format!(" {repo_id}")),
    )
}

fn clear_progress_line() -> Result<()> {
    clear_stderr_line()
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

pub(super) fn cache_repo_id_from_dir(name: &str) -> Option<String> {
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

fn remote_repo_info(api: &hf_hub::HFClientSync, repo_id: &str, ref_name: &str) -> Result<RepoInfo> {
    let (owner, name) = repo_id.split_once('/').unwrap_or(("", repo_id));
    api.model(owner, name)
        .info(
            &RepoInfoParams::builder()
                .revision(ref_name.to_string())
                .build(),
        )
        .with_context(|| format!("Fetch repo info for {repo_id}@{ref_name}"))
}

fn repo_info_sha(info: &RepoInfo) -> String {
    match info {
        RepoInfo::Model(info) => info.sha.clone().unwrap_or_default(),
        RepoInfo::Dataset(info) => info.sha.clone().unwrap_or_default(),
        RepoInfo::Space(info) => info.sha.clone().unwrap_or_default(),
    }
}

fn check_repo_update(api: &hf_hub::HFClientSync, repo: &CachedRepo) -> Result<Option<String>> {
    let remote = remote_repo_info(api, &repo.repo_id, &repo.ref_name)?;
    let remote_revision = repo_info_sha(&remote);
    if remote_revision == repo.local_revision {
        Ok(None)
    } else {
        Ok(Some(remote_revision))
    }
}

fn update_cached_repo(api: &hf_hub::HFClientSync, repo: &CachedRepo) -> Result<UpdateCounts> {
    let (owner, name) = repo
        .repo_id
        .split_once('/')
        .unwrap_or(("", repo.repo_id.as_str()));
    let api_repo = api.model(owner, name);
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
        match api_repo.download_file(
            &RepoDownloadFileParams::builder()
                .filename(file.clone())
                .revision(repo.ref_name.clone())
                .build(),
        ) {
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
    let snapshots_dir = huggingface_hub_cache_dir()
        .join(super::local::huggingface_repo_folder_name(
            &repo.repo_id,
            RepoType::Model,
        ))
        .join("snapshots");
    if !snapshots_dir.is_dir() {
        return Ok(Vec::new());
    }

    let mut snapshot_entries = Vec::new();
    for entry in std::fs::read_dir(&snapshots_dir)
        .with_context(|| format!("Read {}", snapshots_dir.display()))?
    {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }
        snapshot_entries.push((
            entry.file_name().to_string_lossy().to_string(),
            entry.path(),
        ));
    }

    let mut snapshot_roots = Vec::new();
    let exact = snapshots_dir.join(&repo.local_revision);
    if exact.is_dir() {
        snapshot_roots.push(exact);
    } else {
        let mut prefix_matches: Vec<PathBuf> = snapshot_entries
            .iter()
            .filter(|(name, _)| {
                name.starts_with(&repo.local_revision) || repo.local_revision.starts_with(name)
            })
            .map(|(_, path)| path.clone())
            .collect();
        prefix_matches.sort();
        snapshot_roots.extend(prefix_matches);
    }

    if snapshot_roots.is_empty() {
        let mut all: Vec<PathBuf> = snapshot_entries.into_iter().map(|(_, path)| path).collect();
        all.sort();
        snapshot_roots = all;
    }

    let mut files = BTreeSet::new();
    for root in snapshot_roots {
        let mut collected = Vec::new();
        collect_snapshot_files(&root, &root, &mut collected)?;
        files.extend(collected);
    }
    Ok(files.into_iter().collect())
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
    use serial_test::serial;
    use std::ffi::OsString;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn restore_env(key: &str, value: Option<OsString>) {
        if let Some(value) = value {
            std::env::set_var(key, value);
        } else {
            std::env::remove_var(key);
        }
    }

    #[test]
    #[serial]
    fn cached_repo_files_falls_back_to_matching_snapshot_prefix() {
        let prev_hub_cache = std::env::var_os("HF_HUB_CACHE");
        let prev_hf_home = std::env::var_os("HF_HOME");
        let prev_xdg = std::env::var_os("XDG_CACHE_HOME");

        let base = std::env::temp_dir().join(format!(
            "mesh-llm-maintenance-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        let snapshot = base
            .join("models--unsloth--Qwen3.6-35B-A3B-GGUF")
            .join("snapshots")
            .join("9280dd353ab5cafebabedeadbeef123456789abc");
        std::fs::create_dir_all(snapshot.join("BF16")).unwrap();
        std::fs::write(snapshot.join("BF16/model.gguf"), b"gguf").unwrap();

        std::env::set_var("HF_HUB_CACHE", &base);
        std::env::remove_var("HF_HOME");
        std::env::remove_var("XDG_CACHE_HOME");

        let repo = CachedRepo {
            repo_id: "unsloth/Qwen3.6-35B-A3B-GGUF".to_string(),
            ref_name: "main".to_string(),
            local_revision: "9280dd353ab5".to_string(),
        };
        let files = cached_repo_files(&repo).expect("should collect snapshot files");
        assert_eq!(files, vec!["BF16/model.gguf".to_string()]);

        let _ = std::fs::remove_dir_all(&base);
        restore_env("HF_HUB_CACHE", prev_hub_cache);
        restore_env("HF_HOME", prev_hf_home);
        restore_env("XDG_CACHE_HOME", prev_xdg);
    }
}
