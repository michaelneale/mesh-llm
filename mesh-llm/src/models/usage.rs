use super::local::{
    gguf_metadata_cache_path, huggingface_hub_cache_dir, huggingface_identity_for_path,
};
use anyhow::{Context, Result};
use chrono::{DateTime, Duration as ChronoDuration, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::Duration;

#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub struct ModelUsageRecord {
    pub lookup_key: String,
    pub display_name: String,
    pub model_ref: Option<String>,
    pub source: String,
    pub mesh_managed: bool,
    pub primary_path: PathBuf,
    pub managed_paths: Vec<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hf_repo_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hf_revision: Option<String>,
    pub first_seen_at: String,
    pub last_used_at: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct ModelCleanupCandidate {
    pub display_name: String,
    pub model_ref: Option<String>,
    pub source: String,
    pub primary_path: PathBuf,
    pub mesh_managed: bool,
    pub last_used_at: String,
    pub file_count: usize,
    pub total_bytes: u64,
    pub stale_record_only: bool,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct ModelCleanupPlan {
    pub candidates: Vec<ModelCleanupCandidate>,
    pub total_files: usize,
    pub total_bytes: u64,
    pub skipped_recent: usize,
    pub stale_record_only: usize,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct ModelCleanupResult {
    pub removed_candidates: usize,
    pub removed_files: usize,
    pub removed_records: usize,
    pub removed_metadata_files: usize,
    pub reclaimed_bytes: u64,
}

#[derive(Clone, Debug)]
struct CleanupEntry {
    record: ModelUsageRecord,
    record_path: PathBuf,
    removable_paths: Vec<PathBuf>,
    total_bytes: u64,
    stale_record_only: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct RecordLocation {
    lookup_key: String,
    record_path: PathBuf,
    record: Option<ModelUsageRecord>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct HuggingFaceRecordIdentity {
    repo_id: String,
    revision: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct PathHuggingFaceIdentity {
    repo_id: String,
    revision: String,
    canonical_ref: String,
}

pub fn model_usage_cache_dir() -> PathBuf {
    super::mesh_llm_cache_dir().join("model-usage")
}

pub fn load_model_usage_record_for_path(path: &Path) -> Option<ModelUsageRecord> {
    let usage_dir = model_usage_cache_dir();
    let root = huggingface_hub_cache_dir();
    let lookup_key = usage_lookup_key(path, &root)?;
    resolve_record_location(&usage_dir, &lookup_key, &[normalize_path(path)]).record
}

pub fn track_model_usage(
    path: &Path,
    display_name: Option<&str>,
    model_ref: Option<&str>,
    source: Option<&str>,
) -> Result<()> {
    let usage_dir = model_usage_cache_dir();
    let root = huggingface_hub_cache_dir();
    record_model_usage_in_dir(
        &usage_dir,
        &root,
        path,
        &[],
        display_name,
        model_ref,
        source,
        false,
    )
}

pub fn track_managed_model_usage(
    primary_path: &Path,
    managed_paths: &[PathBuf],
    display_name: &str,
    model_ref: Option<&str>,
    source: &str,
) -> Result<()> {
    let usage_dir = model_usage_cache_dir();
    let root = huggingface_hub_cache_dir();
    record_model_usage_in_dir(
        &usage_dir,
        &root,
        primary_path,
        managed_paths,
        Some(display_name),
        model_ref,
        Some(source),
        true,
    )
}

pub fn plan_model_cleanup(unused_since: Option<Duration>) -> Result<ModelCleanupPlan> {
    let usage_dir = model_usage_cache_dir();
    let root = huggingface_hub_cache_dir();
    plan_model_cleanup_in_dir(&usage_dir, &root, unused_since)
}

pub fn execute_model_cleanup(unused_since: Option<Duration>) -> Result<ModelCleanupResult> {
    let usage_dir = model_usage_cache_dir();
    let root = huggingface_hub_cache_dir();
    let records = load_model_usage_records_from_dir(&usage_dir);
    let cutoff = unused_since
        .map(ChronoDuration::from_std)
        .transpose()?
        .map(|age| Utc::now() - age);
    let mut skipped_recent = 0usize;
    let entries = plan_cleanup_entries(records, &usage_dir, &root, cutoff, &mut skipped_recent);
    execute_model_cleanup_entries(entries)
}

fn load_model_usage_records_from_dir(dir: &Path) -> Vec<ModelUsageRecord> {
    let mut records = Vec::new();
    let Ok(entries) = std::fs::read_dir(dir) else {
        return records;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|value| value.to_str()) != Some("json") {
            continue;
        }
        if let Some(record) = read_usage_record(&path) {
            records.push(record);
        }
    }
    records
}

fn read_usage_record(path: &Path) -> Option<ModelUsageRecord> {
    let bytes = std::fs::read(path).ok()?;
    serde_json::from_slice(&bytes).ok()
}

#[allow(clippy::too_many_arguments)]
fn record_model_usage_in_dir(
    usage_dir: &Path,
    track_root: &Path,
    path: &Path,
    managed_paths: &[PathBuf],
    display_name: Option<&str>,
    model_ref: Option<&str>,
    source: Option<&str>,
    mesh_managed: bool,
) -> Result<()> {
    let Some(lookup_key) = usage_lookup_key(path, track_root) else {
        return Ok(());
    };
    let now = Utc::now().to_rfc3339();
    let primary_path = normalize_path(path);
    let normalized_managed_paths = unique_paths(managed_paths.to_vec());
    let candidate_paths = usage_record_candidate_paths(&primary_path, &normalized_managed_paths);
    let location = resolve_record_location(usage_dir, &lookup_key, &candidate_paths);
    let record_path = location.record_path;
    let existing = location.record;
    let existing_display_name = existing
        .as_ref()
        .map(|record| record.display_name.as_str())
        .filter(|value| !value.is_empty());
    let existing_source = existing
        .as_ref()
        .map(|record| record.source.as_str())
        .filter(|value| !value.is_empty());
    let existing_model_ref = existing
        .as_ref()
        .and_then(|record| record.model_ref.as_deref());

    let mut merged_paths = existing
        .as_ref()
        .map(|record| record.managed_paths.clone())
        .unwrap_or_default();
    if mesh_managed {
        if normalized_managed_paths.is_empty() {
            merged_paths.push(primary_path.clone());
        } else {
            merged_paths.extend(normalized_managed_paths.iter().cloned());
        }
    }
    merged_paths = unique_paths(merged_paths);
    let hf_identity = infer_record_hf_identity(&primary_path, &merged_paths, track_root)
        .or_else(|| existing.as_ref().and_then(record_hf_identity));

    let record = ModelUsageRecord {
        lookup_key: location.lookup_key,
        display_name: display_name
            .or(existing_display_name)
            .map(str::to_string)
            .unwrap_or_else(|| default_display_name(&primary_path)),
        model_ref: model_ref
            .or(existing_model_ref)
            .map(str::to_string)
            .or_else(|| default_model_ref(&primary_path)),
        source: source
            .or(existing_source)
            .map(str::to_string)
            .unwrap_or_else(|| default_source(&primary_path)),
        mesh_managed: mesh_managed || existing.as_ref().is_some_and(|record| record.mesh_managed),
        primary_path,
        managed_paths: merged_paths,
        hf_repo_id: hf_identity
            .as_ref()
            .map(|identity| identity.repo_id.clone()),
        hf_revision: hf_identity
            .as_ref()
            .map(|identity| identity.revision.clone()),
        first_seen_at: existing
            .as_ref()
            .map(|record| record.first_seen_at.clone())
            .unwrap_or_else(|| now.clone()),
        last_used_at: now,
    };

    std::fs::create_dir_all(usage_dir)
        .with_context(|| format!("Create {}", usage_dir.display()))?;
    let bytes = serde_json::to_vec_pretty(&record)?;
    std::fs::write(&record_path, bytes)
        .with_context(|| format!("Write {}", record_path.display()))?;
    Ok(())
}

fn plan_model_cleanup_in_dir(
    usage_dir: &Path,
    track_root: &Path,
    unused_since: Option<Duration>,
) -> Result<ModelCleanupPlan> {
    let records = load_model_usage_records_from_dir(usage_dir);
    let cutoff = unused_since
        .map(ChronoDuration::from_std)
        .transpose()?
        .map(|age| Utc::now() - age);
    let mut skipped_recent = 0usize;
    let entries = plan_cleanup_entries(records, usage_dir, track_root, cutoff, &mut skipped_recent);
    let mut plan = ModelCleanupPlan {
        skipped_recent,
        ..Default::default()
    };
    for entry in entries {
        if entry.stale_record_only {
            plan.stale_record_only += 1;
        }
        plan.total_files += entry.removable_paths.len();
        plan.total_bytes += entry.total_bytes;
        plan.candidates.push(ModelCleanupCandidate {
            display_name: entry.record.display_name,
            model_ref: entry.record.model_ref,
            source: entry.record.source,
            primary_path: entry.record.primary_path,
            mesh_managed: entry.record.mesh_managed,
            last_used_at: entry.record.last_used_at,
            file_count: entry.removable_paths.len(),
            total_bytes: entry.total_bytes,
            stale_record_only: entry.stale_record_only,
        });
    }
    plan.candidates.sort_by(|left, right| {
        left.last_used_at
            .cmp(&right.last_used_at)
            .then_with(|| left.display_name.cmp(&right.display_name))
    });
    Ok(plan)
}

fn plan_cleanup_entries(
    records: Vec<ModelUsageRecord>,
    usage_dir: &Path,
    track_root: &Path,
    cutoff: Option<DateTime<Utc>>,
    skipped_recent: &mut usize,
) -> Vec<CleanupEntry> {
    let mut entries = Vec::new();

    for record in records {
        if !record.mesh_managed {
            continue;
        }
        let last_used =
            parse_timestamp(&record.last_used_at).unwrap_or(DateTime::<Utc>::UNIX_EPOCH);
        if let Some(cutoff) = cutoff {
            if last_used > cutoff {
                *skipped_recent += 1;
                continue;
            }
        }

        let removable_paths: Vec<PathBuf> = unique_paths(record.managed_paths.clone())
            .into_iter()
            .filter(|path| is_trackable_path(path, track_root))
            .filter(|path| path_matches_record_identity(&record, path, track_root))
            .filter(|path| path.exists())
            .collect();
        let total_bytes = removable_paths
            .iter()
            .filter_map(|path| std::fs::metadata(path).ok().map(|meta| meta.len()))
            .sum();
        let stale_record_only = removable_paths.is_empty();
        let record_path = usage_record_path(usage_dir, &record.lookup_key);

        entries.push(CleanupEntry {
            record,
            record_path,
            removable_paths,
            total_bytes,
            stale_record_only,
        });
    }

    entries.sort_by(|left, right| {
        left.record
            .last_used_at
            .cmp(&right.record.last_used_at)
            .then_with(|| left.record.display_name.cmp(&right.record.display_name))
    });
    entries
}

fn execute_model_cleanup_entries(entries: Vec<CleanupEntry>) -> Result<ModelCleanupResult> {
    let mut result = ModelCleanupResult::default();
    for entry in entries {
        for path in &entry.removable_paths {
            if let Ok(meta) = std::fs::metadata(path) {
                result.reclaimed_bytes += meta.len();
            }
            if path.exists() {
                std::fs::remove_file(path).with_context(|| format!("Remove {}", path.display()))?;
                result.removed_files += 1;
            }
            if let Some(cache_path) = gguf_metadata_cache_path(path) {
                if cache_path.exists() {
                    std::fs::remove_file(&cache_path).with_context(|| {
                        format!("Remove metadata cache {}", cache_path.display())
                    })?;
                    result.removed_metadata_files += 1;
                }
            }
            prune_empty_ancestors(path, &huggingface_hub_cache_dir());
        }
        if entry.record_path.exists() {
            std::fs::remove_file(&entry.record_path)
                .with_context(|| format!("Remove {}", entry.record_path.display()))?;
            result.removed_records += 1;
        }
        result.removed_candidates += 1;
    }
    Ok(result)
}

fn usage_lookup_key(path: &Path, track_root: &Path) -> Option<String> {
    if !is_trackable_path(path, track_root) {
        return None;
    }
    if let Some(identity) = hf_identity_for_path_in_root(path, track_root) {
        return Some(format!("hf:{}", identity.canonical_ref));
    }
    if let Some(identity) = huggingface_identity_for_path(path) {
        return Some(format!("hf:{}", identity.canonical_ref));
    }
    Some(format!(
        "path:{}",
        normalize_path(path).to_string_lossy().replace('\\', "/")
    ))
}

fn resolve_record_location(
    usage_dir: &Path,
    direct_lookup_key: &str,
    candidate_paths: &[PathBuf],
) -> RecordLocation {
    let direct_record_path = usage_record_path(usage_dir, direct_lookup_key);
    if let Some(record) = read_usage_record(&direct_record_path) {
        return RecordLocation {
            lookup_key: direct_lookup_key.to_string(),
            record_path: direct_record_path,
            record: Some(record),
        };
    }

    let Some(existing) = find_usage_record_by_paths(usage_dir, candidate_paths) else {
        return RecordLocation {
            lookup_key: direct_lookup_key.to_string(),
            record_path: direct_record_path,
            record: None,
        };
    };

    let record_path = usage_record_path(usage_dir, &existing.lookup_key);
    RecordLocation {
        lookup_key: existing.lookup_key.clone(),
        record_path,
        record: Some(existing),
    }
}

fn find_usage_record_by_paths(
    usage_dir: &Path,
    candidate_paths: &[PathBuf],
) -> Option<ModelUsageRecord> {
    let candidate_paths = unique_paths(candidate_paths.to_vec());
    if candidate_paths.is_empty() {
        return None;
    }
    let candidate_set: HashSet<PathBuf> = candidate_paths.iter().cloned().collect();
    load_model_usage_records_from_dir(usage_dir)
        .into_iter()
        .find(|record| record_matches_any_path(record, &candidate_set))
}

fn record_matches_any_path(record: &ModelUsageRecord, candidate_paths: &HashSet<PathBuf>) -> bool {
    let primary_path = normalize_path(&record.primary_path);
    if candidate_paths.contains(&primary_path) {
        return true;
    }
    if record
        .managed_paths
        .iter()
        .map(|path| normalize_path(path))
        .any(|path| candidate_paths.contains(&path))
    {
        return true;
    }
    false
}

fn usage_record_candidate_paths(primary_path: &Path, managed_paths: &[PathBuf]) -> Vec<PathBuf> {
    let mut paths = vec![primary_path.to_path_buf()];
    paths.extend(managed_paths.iter().cloned());
    unique_paths(paths)
}

fn infer_record_hf_identity(
    primary_path: &Path,
    managed_paths: &[PathBuf],
    track_root: &Path,
) -> Option<HuggingFaceRecordIdentity> {
    let mut paths = usage_record_candidate_paths(primary_path, managed_paths).into_iter();
    let first = paths
        .find_map(|path| hf_identity_for_path_in_root(&path, track_root))
        .map(|identity| HuggingFaceRecordIdentity {
            repo_id: identity.repo_id,
            revision: identity.revision,
        })?;

    for path in usage_record_candidate_paths(primary_path, managed_paths) {
        let Some(identity) = hf_identity_for_path_in_root(&path, track_root) else {
            continue;
        };
        if identity.repo_id != first.repo_id || identity.revision != first.revision {
            return None;
        }
    }
    Some(first)
}

fn record_hf_identity(record: &ModelUsageRecord) -> Option<HuggingFaceRecordIdentity> {
    Some(HuggingFaceRecordIdentity {
        repo_id: record.hf_repo_id.clone()?,
        revision: record.hf_revision.clone()?,
    })
}

fn path_matches_record_identity(record: &ModelUsageRecord, path: &Path, track_root: &Path) -> bool {
    let Some(record_identity) = record_hf_identity(record) else {
        return true;
    };
    hf_identity_for_path_in_root(path, track_root).is_some_and(|identity| {
        identity.repo_id == record_identity.repo_id && identity.revision == record_identity.revision
    })
}

fn hf_identity_for_path_in_root(path: &Path, track_root: &Path) -> Option<PathHuggingFaceIdentity> {
    let path = normalize_path(path);
    let root = normalize_path(track_root);
    let relative = path.strip_prefix(&root).ok()?;
    let mut components = relative.components();
    let repo_dir = components.next()?.as_os_str().to_str()?;
    let repo_id = repo_dir.strip_prefix("models--")?.replace("--", "/");
    if components.next()?.as_os_str() != "snapshots" {
        return None;
    }
    let revision = components.next()?.as_os_str().to_str()?.to_string();
    let relative_file = components
        .map(|component| component.as_os_str().to_str())
        .collect::<Option<Vec<_>>>()?
        .join("/");
    if relative_file.is_empty() {
        return None;
    }
    Some(PathHuggingFaceIdentity {
        repo_id: repo_id.clone(),
        revision: revision.clone(),
        canonical_ref: format!("{repo_id}@{revision}/{relative_file}"),
    })
}

pub(crate) fn usage_record_path(usage_dir: &Path, lookup_key: &str) -> PathBuf {
    let digest = Sha256::digest(lookup_key.as_bytes());
    usage_dir.join(format!("{digest:x}.json"))
}

fn normalize_path(path: &Path) -> PathBuf {
    path.canonicalize().unwrap_or_else(|_| path.to_path_buf())
}

fn unique_paths(paths: Vec<PathBuf>) -> Vec<PathBuf> {
    let mut seen = HashSet::new();
    let mut unique = Vec::new();
    for path in paths {
        let normalized = normalize_path(&path);
        if seen.insert(normalized.clone()) {
            unique.push(normalized);
        }
    }
    unique.sort();
    unique
}

fn parse_timestamp(value: &str) -> Option<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(value)
        .ok()
        .map(|parsed| parsed.with_timezone(&Utc))
}

fn default_display_name(path: &Path) -> String {
    path.file_stem()
        .and_then(|value| value.to_str())
        .or_else(|| path.file_name().and_then(|value| value.to_str()))
        .unwrap_or("model")
        .to_string()
}

fn default_model_ref(path: &Path) -> Option<String> {
    huggingface_identity_for_path(path).map(|identity| identity.canonical_ref)
}

fn default_source(path: &Path) -> String {
    if huggingface_identity_for_path(path).is_some() {
        "huggingface-cache".to_string()
    } else {
        "local-cache".to_string()
    }
}

fn is_trackable_path(path: &Path, track_root: &Path) -> bool {
    let path = normalize_path(path);
    let root = normalize_path(track_root);
    path.starts_with(&root)
}

fn prune_empty_ancestors(path: &Path, stop_at: &Path) {
    let stop_at = normalize_path(stop_at);
    let mut current = path.parent().map(normalize_path);
    while let Some(dir) = current {
        if dir == stop_at {
            break;
        }
        let Ok(mut entries) = std::fs::read_dir(&dir) else {
            break;
        };
        if entries.next().is_some() {
            break;
        }
        if std::fs::remove_dir(&dir).is_err() {
            break;
        }
        current = dir.parent().map(normalize_path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEMP_DIR_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn temp_dir(prefix: &str) -> PathBuf {
        let sequence = TEMP_DIR_COUNTER.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "{prefix}-{}-{}-{}",
            std::process::id(),
            sequence,
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system time should be after epoch")
                .as_nanos()
        ))
    }

    fn write_record(dir: &Path, record: &ModelUsageRecord) {
        std::fs::create_dir_all(dir).expect("usage dir should be created");
        let path = usage_record_path(dir, &record.lookup_key);
        std::fs::write(
            path,
            serde_json::to_vec_pretty(record).expect("record JSON should serialize"),
        )
        .expect("record should be written");
    }

    #[test]
    fn record_model_usage_merges_managed_paths() {
        let usage_dir = temp_dir("mesh-llm-usage-dir");
        let cache_root = temp_dir("mesh-llm-hf-cache");
        let primary = cache_root
            .join("models--Org--Demo")
            .join("snapshots")
            .join("rev1")
            .join("Demo-Q4_K_M.gguf");
        let shard = cache_root
            .join("models--Org--Demo")
            .join("snapshots")
            .join("rev1")
            .join("Demo-Q4_K_M-00002-of-00002.gguf");
        std::fs::create_dir_all(primary.parent().expect("primary path should have parent"))
            .expect("primary parent should exist");
        std::fs::write(&primary, b"primary").expect("primary model should be written");
        std::fs::write(&shard, b"shard").expect("shard model should be written");

        record_model_usage_in_dir(
            &usage_dir,
            &cache_root,
            &primary,
            &[primary.clone(), shard.clone()],
            Some("Demo-Q4_K_M"),
            Some("Org/Demo@rev1/Demo-Q4_K_M.gguf"),
            Some("catalog"),
            true,
        )
        .expect("managed usage should be recorded");

        let records = load_model_usage_records_from_dir(&usage_dir);
        assert_eq!(records.len(), 1);
        assert!(records[0].mesh_managed);
        assert_eq!(records[0].managed_paths.len(), 2);
        assert_eq!(records[0].display_name, "Demo-Q4_K_M");
        assert_eq!(records[0].hf_repo_id.as_deref(), Some("Org/Demo"));
        assert_eq!(records[0].hf_revision.as_deref(), Some("rev1"));

        let _ = std::fs::remove_dir_all(&usage_dir);
        let _ = std::fs::remove_dir_all(&cache_root);
    }

    #[test]
    fn load_model_usage_record_for_split_shard_returns_bundle_record() {
        let usage_dir = temp_dir("mesh-llm-usage-dir");
        let cache_root = temp_dir("mesh-llm-hf-cache");
        let primary = cache_root
            .join("models--Org--Bundle")
            .join("snapshots")
            .join("rev1")
            .join("Bundle-Q4_K_M-00001-of-00002.gguf");
        let shard = cache_root
            .join("models--Org--Bundle")
            .join("snapshots")
            .join("rev1")
            .join("Bundle-Q4_K_M-00002-of-00002.gguf");
        std::fs::create_dir_all(primary.parent().expect("primary path should have parent"))
            .expect("primary parent should exist");
        std::fs::write(&primary, b"primary").expect("primary model should be written");
        std::fs::write(&shard, b"shard").expect("shard model should be written");

        record_model_usage_in_dir(
            &usage_dir,
            &cache_root,
            &primary,
            &[primary.clone(), shard.clone()],
            Some("Bundle-Q4_K_M"),
            Some("Org/Bundle@rev1/Bundle-Q4_K_M-00001-of-00002.gguf"),
            Some("catalog"),
            true,
        )
        .expect("managed usage should be recorded");

        let record = find_usage_record_by_paths(&usage_dir, &[shard.clone()])
            .expect("split shard should resolve back to the bundle record");
        assert_eq!(
            record.lookup_key,
            usage_lookup_key(&primary, &cache_root).expect("primary path should key")
        );
        assert_eq!(record.managed_paths.len(), 2);

        let _ = std::fs::remove_dir_all(&usage_dir);
        let _ = std::fs::remove_dir_all(&cache_root);
    }

    #[test]
    fn record_model_usage_updates_last_used_at_by_managed_identity() {
        let usage_dir = temp_dir("mesh-llm-usage-dir");
        let cache_root = temp_dir("mesh-llm-hf-cache");
        let primary = cache_root
            .join("models--Org--Bundle")
            .join("snapshots")
            .join("rev1")
            .join("Bundle-Q4_K_M-00001-of-00002.gguf");
        let shard = cache_root
            .join("models--Org--Bundle")
            .join("snapshots")
            .join("rev1")
            .join("Bundle-Q4_K_M-00002-of-00002.gguf");
        std::fs::create_dir_all(primary.parent().expect("primary path should have parent"))
            .expect("primary parent should exist");
        std::fs::write(&primary, b"primary").expect("primary model should be written");
        std::fs::write(&shard, b"shard").expect("shard model should be written");

        let lookup_key = usage_lookup_key(&primary, &cache_root).expect("primary path should key");
        write_record(
            &usage_dir,
            &ModelUsageRecord {
                lookup_key: lookup_key.clone(),
                display_name: "Bundle-Q4_K_M".to_string(),
                model_ref: Some("Org/Bundle@rev1/Bundle-Q4_K_M-00001-of-00002.gguf".to_string()),
                source: "catalog".to_string(),
                mesh_managed: true,
                primary_path: primary.clone(),
                managed_paths: vec![primary.clone(), shard.clone()],
                hf_repo_id: Some("Org/Bundle".to_string()),
                hf_revision: Some("rev1".to_string()),
                first_seen_at: "2026-04-01T00:00:00Z".to_string(),
                last_used_at: "2026-04-01T00:00:00Z".to_string(),
            },
        );

        record_model_usage_in_dir(
            &usage_dir,
            &cache_root,
            &shard,
            &[],
            None,
            None,
            Some("resolve"),
            false,
        )
        .expect("usage refresh should succeed");

        let records = load_model_usage_records_from_dir(&usage_dir);
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].lookup_key, lookup_key);
        assert_eq!(records[0].managed_paths.len(), 2);
        assert_ne!(records[0].last_used_at, "2026-04-01T00:00:00Z");

        let _ = std::fs::remove_dir_all(&usage_dir);
        let _ = std::fs::remove_dir_all(&cache_root);
    }

    #[test]
    fn cleanup_plan_filters_recent_and_external_records() {
        let usage_dir = temp_dir("mesh-llm-usage-dir");
        let cache_root = temp_dir("mesh-llm-hf-cache");
        let old_path = cache_root
            .join("models--Org--Old")
            .join("snapshots")
            .join("rev1")
            .join("Old-Q4_K_M.gguf");
        let recent_path = cache_root
            .join("models--Org--Recent")
            .join("snapshots")
            .join("rev1")
            .join("Recent-Q4_K_M.gguf");
        let external_path = cache_root
            .join("models--Org--External")
            .join("snapshots")
            .join("rev1")
            .join("External-Q4_K_M.gguf");

        for path in [&old_path, &recent_path, &external_path] {
            std::fs::create_dir_all(path.parent().expect("test path should have parent"))
                .expect("test parent should exist");
            std::fs::write(path, vec![0_u8; 16]).expect("test model should be written");
        }

        write_record(
            &usage_dir,
            &ModelUsageRecord {
                lookup_key: usage_lookup_key(&old_path, &cache_root).expect("old path should key"),
                display_name: "Old".to_string(),
                model_ref: None,
                source: "catalog".to_string(),
                mesh_managed: true,
                primary_path: old_path.clone(),
                managed_paths: vec![old_path.clone()],
                hf_repo_id: Some("Org/Old".to_string()),
                hf_revision: Some("rev1".to_string()),
                first_seen_at: "2026-04-01T00:00:00Z".to_string(),
                last_used_at: "2026-04-01T00:00:00Z".to_string(),
            },
        );
        write_record(
            &usage_dir,
            &ModelUsageRecord {
                lookup_key: usage_lookup_key(&recent_path, &cache_root)
                    .expect("recent path should key"),
                display_name: "Recent".to_string(),
                model_ref: None,
                source: "catalog".to_string(),
                mesh_managed: true,
                primary_path: recent_path.clone(),
                managed_paths: vec![recent_path.clone()],
                hf_repo_id: Some("Org/Recent".to_string()),
                hf_revision: Some("rev1".to_string()),
                first_seen_at: Utc::now().to_rfc3339(),
                last_used_at: Utc::now().to_rfc3339(),
            },
        );
        write_record(
            &usage_dir,
            &ModelUsageRecord {
                lookup_key: usage_lookup_key(&external_path, &cache_root)
                    .expect("external path should key"),
                display_name: "External".to_string(),
                model_ref: None,
                source: "local-cache".to_string(),
                mesh_managed: false,
                primary_path: external_path.clone(),
                managed_paths: vec![],
                hf_repo_id: Some("Org/External".to_string()),
                hf_revision: Some("rev1".to_string()),
                first_seen_at: "2026-04-01T00:00:00Z".to_string(),
                last_used_at: "2026-04-01T00:00:00Z".to_string(),
            },
        );

        let plan =
            plan_model_cleanup_in_dir(&usage_dir, &cache_root, Some(Duration::from_secs(60)))
                .expect("cleanup plan should succeed");
        assert_eq!(plan.candidates.len(), 1);
        assert_eq!(plan.candidates[0].display_name, "Old");
        assert_eq!(plan.skipped_recent, 1);

        let _ = std::fs::remove_dir_all(&usage_dir);
        let _ = std::fs::remove_dir_all(&cache_root);
    }

    #[test]
    fn execute_cleanup_removes_files_and_records() {
        let usage_dir = temp_dir("mesh-llm-usage-dir");
        let cache_root = temp_dir("mesh-llm-hf-cache");
        let primary = cache_root
            .join("models--Org--Cleanup")
            .join("snapshots")
            .join("rev1")
            .join("Cleanup-Q4_K_M.gguf");
        std::fs::create_dir_all(primary.parent().expect("cleanup path should have parent"))
            .expect("cleanup parent should exist");
        std::fs::write(&primary, vec![0_u8; 32]).expect("cleanup model should be written");

        let record = ModelUsageRecord {
            lookup_key: usage_lookup_key(&primary, &cache_root).expect("cleanup path should key"),
            display_name: "Cleanup".to_string(),
            model_ref: Some("Org/Cleanup@rev1/Cleanup-Q4_K_M.gguf".to_string()),
            source: "catalog".to_string(),
            mesh_managed: true,
            primary_path: primary.clone(),
            managed_paths: vec![primary.clone()],
            hf_repo_id: Some("Org/Cleanup".to_string()),
            hf_revision: Some("rev1".to_string()),
            first_seen_at: "2026-04-01T00:00:00Z".to_string(),
            last_used_at: "2026-04-01T00:00:00Z".to_string(),
        };
        write_record(&usage_dir, &record);

        let mut skipped_recent = 0usize;
        let entries = plan_cleanup_entries(
            vec![record],
            &usage_dir,
            &cache_root,
            None,
            &mut skipped_recent,
        );
        let result = execute_model_cleanup_entries(entries).expect("cleanup should succeed");
        assert_eq!(result.removed_candidates, 1);
        assert_eq!(result.removed_files, 1);
        assert_eq!(result.removed_records, 1);
        assert!(!primary.exists());
        assert!(load_model_usage_records_from_dir(&usage_dir).is_empty());

        let _ = std::fs::remove_dir_all(&usage_dir);
        let _ = std::fs::remove_dir_all(&cache_root);
    }

    #[test]
    fn cleanup_skips_paths_that_no_longer_match_record_identity() {
        let usage_dir = temp_dir("mesh-llm-usage-dir");
        let cache_root = temp_dir("mesh-llm-hf-cache");
        let old_path = cache_root
            .join("models--Org--Actual")
            .join("snapshots")
            .join("rev2")
            .join("Actual-Q4_K_M.gguf");
        std::fs::create_dir_all(old_path.parent().expect("cleanup path should have parent"))
            .expect("cleanup parent should exist");
        std::fs::write(&old_path, vec![0_u8; 32]).expect("cleanup model should be written");

        let record = ModelUsageRecord {
            lookup_key: "hf:Org/Expected@rev1/Expected-Q4_K_M.gguf".to_string(),
            display_name: "Expected".to_string(),
            model_ref: Some("Org/Expected@rev1/Expected-Q4_K_M.gguf".to_string()),
            source: "catalog".to_string(),
            mesh_managed: true,
            primary_path: old_path.clone(),
            managed_paths: vec![old_path.clone()],
            hf_repo_id: Some("Org/Expected".to_string()),
            hf_revision: Some("rev1".to_string()),
            first_seen_at: "2026-04-01T00:00:00Z".to_string(),
            last_used_at: "2026-04-01T00:00:00Z".to_string(),
        };
        write_record(&usage_dir, &record);

        let mut skipped_recent = 0usize;
        let entries = plan_cleanup_entries(
            vec![record],
            &usage_dir,
            &cache_root,
            None,
            &mut skipped_recent,
        );
        assert_eq!(entries.len(), 1);
        assert!(entries[0].removable_paths.is_empty());
        assert!(entries[0].stale_record_only);
        assert!(old_path.exists());

        let _ = std::fs::remove_dir_all(&usage_dir);
        let _ = std::fs::remove_dir_all(&cache_root);
    }
}
