//! MoE expert sharding: split models across mesh nodes by expert assignment.
//!
//! Each node gets a GGUF with the full trunk (attention, norms, embeddings, head)
//! plus a subset of experts. The shared core (hottest experts by gate mass) is
//! replicated to every node. Remaining experts are distributed uniquely.
//!
//! No cross-node traffic during inference — each node runs independently.
//!
//! # Storage modes
//! - **Monolithic** (default): each node holds a self-contained GGUF with trunk
//!   + its expert subset.
//! - **Split** (`moe.storage.mode = "split"`): one shared trunk GGUF lives on
//!   NAS, and each node has a small per-node experts GGUF. Handled by the
//!   `ensure_trunk_and_expert` + `SplitManifest` machinery below.

use chrono::{DateTime, Utc};
use clap::ValueEnum;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

pub use crate::models::gguf::{detect_moe, scan_gguf_compact_meta, GgufMoeInfo};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, ValueEnum)]
pub enum MoeRankingStrategy {
    #[default]
    Auto,
    Analyze,
    MicroAnalyze,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, ValueEnum)]
pub enum MoeMicroLayerScope {
    First,
    #[default]
    All,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SharedRankingKind {
    Analyze,
    MicroAnalyze,
}

impl SharedRankingKind {
    pub fn label(self) -> &'static str {
        match self {
            Self::Analyze => "analyze",
            Self::MicroAnalyze => "micro-analyze",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SharedRankingOrigin {
    LocalFullAnalyze,
    LocalMicroAnalyze,
    PeerImport,
    LegacyCache,
}

impl SharedRankingOrigin {
    pub fn label(self) -> &'static str {
        match self {
            Self::LocalFullAnalyze => "local-full-analyze",
            Self::LocalMicroAnalyze => "local-micro-analyze",
            Self::PeerImport => "peer-import",
            Self::LegacyCache => "legacy-cache",
        }
    }

    pub fn from_label(value: &str) -> Option<Self> {
        match value {
            "local-full-analyze" => Some(Self::LocalFullAnalyze),
            "local-micro-analyze" => Some(Self::LocalMicroAnalyze),
            "peer-import" => Some(Self::PeerImport),
            "legacy-cache" => Some(Self::LegacyCache),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SharedRankingArtifact {
    pub kind: SharedRankingKind,
    pub origin: SharedRankingOrigin,
    pub ranking: Vec<u32>,
    pub micro_prompt_count: Option<usize>,
    pub micro_tokens: Option<u32>,
    pub micro_layer_scope: Option<MoeMicroLayerScope>,
}

#[derive(Clone, Debug)]
pub struct MoeRuntimeOptions {
    pub ranking_strategy: MoeRankingStrategy,
    pub micro_prompt_count: usize,
    pub micro_tokens: u32,
    pub micro_layer_scope: MoeMicroLayerScope,
}

impl Default for MoeRuntimeOptions {
    fn default() -> Self {
        Self {
            ranking_strategy: MoeRankingStrategy::Auto,
            micro_prompt_count: 1,
            micro_tokens: 8,
            micro_layer_scope: MoeMicroLayerScope::All,
        }
    }
}
// ── GGUF assembler: combine trunk + expert files into a shard ──

// ── Ranking cache ──

fn mesh_cache_dir() -> PathBuf {
    if let Ok(path) = std::env::var("XDG_CACHE_HOME") {
        let trimmed = path.trim();
        if !trimmed.is_empty() {
            return PathBuf::from(trimmed).join("mesh-llm");
        }
    }

    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".cache")
        .join("mesh-llm")
}

fn split_cache_root() -> PathBuf {
    mesh_cache_dir().join("splits")
}

fn ranking_cache_root() -> PathBuf {
    mesh_cache_dir().join("moe-rankings")
}

fn ranking_strength_key(artifact: &SharedRankingArtifact) -> (u8, u8, usize, u32) {
    match artifact.kind {
        SharedRankingKind::Analyze => (2, 0, 0, 0),
        SharedRankingKind::MicroAnalyze => (
            1,
            match artifact
                .micro_layer_scope
                .unwrap_or(MoeMicroLayerScope::First)
            {
                MoeMicroLayerScope::All => 1,
                MoeMicroLayerScope::First => 0,
            },
            artifact.micro_prompt_count.unwrap_or(0),
            artifact.micro_tokens.unwrap_or(0),
        ),
    }
}

fn better_shared_ranking(
    candidate: &SharedRankingArtifact,
    current: &SharedRankingArtifact,
) -> bool {
    ranking_strength_key(candidate) > ranking_strength_key(current)
}

fn sanitize_cache_component(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for ch in input.chars() {
        if ch.is_ascii_alphanumeric() || matches!(ch, '.' | '_' | '-') {
            out.push(ch);
        } else {
            out.push('_');
        }
    }
    out.trim_matches('_').to_string()
}

fn ranking_cache_stem(model_path: &Path) -> String {
    if let Some(identity) = crate::models::huggingface_identity_for_path(model_path) {
        let repo = identity.repo_id.replace('/', "--");
        let revision = sanitize_cache_component(&identity.revision);
        let file = sanitize_cache_component(&identity.local_file_name);
        return format!("hf-{repo}-{revision}-{file}");
    }

    let metadata_key = std::fs::metadata(model_path)
        .ok()
        .and_then(|metadata| {
            let modified = metadata.modified().ok()?;
            let modified = modified
                .duration_since(std::time::UNIX_EPOCH)
                .ok()?
                .as_nanos();
            Some(format!(
                "{}:{}:{}",
                model_path.to_string_lossy(),
                metadata.len(),
                modified
            ))
        })
        .unwrap_or_else(|| model_path.to_string_lossy().to_string());
    let digest = Sha256::digest(metadata_key.as_bytes());
    let stem = model_path
        .file_stem()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();
    let stem = sanitize_cache_component(&stem);
    format!("local-{stem}-{:x}", digest)
}

/// Path to cached ranking CSV for a model.
/// Stored under the mesh-llm cache root so local sidecar data never pollutes the model directory.
pub fn ranking_cache_path(model_path: &Path) -> PathBuf {
    ranking_cache_root().join(format!("{}.csv", ranking_cache_stem(model_path)))
}

pub fn micro_ranking_cache_path(
    model_path: &Path,
    prompt_count: usize,
    tokens: u32,
    layer_scope: MoeMicroLayerScope,
) -> PathBuf {
    let stem = ranking_cache_stem(model_path);
    let layer_suffix = match layer_scope {
        MoeMicroLayerScope::First => "first",
        MoeMicroLayerScope::All => "all",
    };
    ranking_cache_root().join(format!(
        "{stem}.micro-p{prompt_count}-t{tokens}-{layer_suffix}.csv"
    ))
}

#[derive(Default)]
struct CachedRankingMetadata {
    ranking_origin: Option<SharedRankingOrigin>,
    micro_prompt_count: Option<usize>,
    micro_tokens: Option<u32>,
    micro_layer_scope: Option<MoeMicroLayerScope>,
}

struct CachedRankingFile {
    ranking: Vec<u32>,
    metadata: CachedRankingMetadata,
}

fn parse_cached_ranking_metadata(line: &str, metadata: &mut CachedRankingMetadata) {
    let Some(rest) = line.strip_prefix('#') else {
        return;
    };
    let rest = rest.trim();
    let Some((key, value)) = rest.split_once('=') else {
        return;
    };
    let value = value.trim();
    match key.trim() {
        "ranking_origin" => metadata.ranking_origin = SharedRankingOrigin::from_label(value),
        "micro_prompt_count" => metadata.micro_prompt_count = value.parse().ok(),
        "micro_tokens" => metadata.micro_tokens = value.parse().ok(),
        "micro_layer_scope" => {
            metadata.micro_layer_scope = match value {
                "all" => Some(MoeMicroLayerScope::All),
                "first" => Some(MoeMicroLayerScope::First),
                _ => None,
            }
        }
        _ => {}
    }
}

fn load_cached_ranking_file(path: &Path) -> Option<CachedRankingFile> {
    let content = std::fs::read_to_string(path).ok()?;
    let mut metadata = CachedRankingMetadata::default();
    let ranking: Vec<u32> = content
        .lines()
        .filter_map(|line| {
            if line.is_empty() {
                return None;
            }
            if line.starts_with('#') {
                parse_cached_ranking_metadata(line, &mut metadata);
                return None;
            }
            if line.starts_with("expert") {
                return None;
            }
            line.split(',').next()?.trim().parse().ok()
        })
        .collect();
    if ranking.is_empty() {
        None
    } else {
        Some(CachedRankingFile { ranking, metadata })
    }
}

pub fn load_shared_ranking_artifact(
    path: &Path,
    kind: SharedRankingKind,
    fallback_origin: SharedRankingOrigin,
    micro_prompt_count: Option<usize>,
    micro_tokens: Option<u32>,
    micro_layer_scope: Option<MoeMicroLayerScope>,
) -> Option<SharedRankingArtifact> {
    let file = load_cached_ranking_file(path)?;
    Some(SharedRankingArtifact {
        kind,
        origin: file.metadata.ranking_origin.unwrap_or(fallback_origin),
        ranking: file.ranking,
        micro_prompt_count: file.metadata.micro_prompt_count.or(micro_prompt_count),
        micro_tokens: file.metadata.micro_tokens.or(micro_tokens),
        micro_layer_scope: file.metadata.micro_layer_scope.or(micro_layer_scope),
    })
}

/// Load a cached ranking CSV. Format: one expert_id per line, sorted by gate mass descending.
/// Also supports the full CSV format from moe-analyze: expert_id,total_mass,mass_fraction,selection_count
pub fn load_cached_ranking(path: &Path) -> Option<Vec<u32>> {
    load_cached_ranking_file(path).map(|file| file.ranking)
}

fn write_shared_ranking_artifact(
    path: &Path,
    artifact: &SharedRankingArtifact,
) -> anyhow::Result<()> {
    if artifact.ranking.is_empty() {
        anyhow::bail!("cannot write empty ranking to {}", path.display());
    }

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let mut lines = vec![
        "# mesh-llm-moe-ranking=v1".to_string(),
        format!("# ranking_kind={}", artifact.kind.label()),
        format!("# ranking_origin={}", artifact.origin.label()),
    ];
    if let Some(prompt_count) = artifact.micro_prompt_count {
        lines.push(format!("# micro_prompt_count={prompt_count}"));
    }
    if let Some(tokens) = artifact.micro_tokens {
        lines.push(format!("# micro_tokens={tokens}"));
    }
    if let Some(layer_scope) = artifact.micro_layer_scope {
        let scope = match layer_scope {
            MoeMicroLayerScope::First => "first",
            MoeMicroLayerScope::All => "all",
        };
        lines.push(format!("# micro_layer_scope={scope}"));
    }
    lines.extend(artifact.ranking.iter().map(u32::to_string));
    std::fs::write(path, format!("{}\n", lines.join("\n")))?;
    Ok(())
}

fn parse_micro_cache_filename(
    model_path: &Path,
    file_name: &str,
) -> Option<(usize, u32, MoeMicroLayerScope)> {
    let stem = ranking_cache_stem(model_path);
    let prefix = format!("{stem}.micro-p");
    let rest = file_name.strip_prefix(&prefix)?.strip_suffix(".csv")?;
    let (prompt_count, rest) = rest.split_once("-t")?;
    let (tokens, layer_scope) = rest.split_once('-')?;
    let layer_scope = match layer_scope {
        "all" => MoeMicroLayerScope::All,
        "first" => MoeMicroLayerScope::First,
        _ => return None,
    };
    Some((
        prompt_count.parse().ok()?,
        tokens.parse().ok()?,
        layer_scope,
    ))
}

pub fn best_shared_ranking_artifact(model_path: &Path) -> Option<SharedRankingArtifact> {
    if let Some(artifact) = load_shared_ranking_artifact(
        &ranking_cache_path(model_path),
        SharedRankingKind::Analyze,
        SharedRankingOrigin::LegacyCache,
        None,
        None,
        None,
    ) {
        return Some(artifact);
    }

    let mut best: Option<SharedRankingArtifact> = None;
    let root = ranking_cache_root();
    let entries = std::fs::read_dir(root).ok()?;
    for entry in entries.flatten() {
        let file_name = match entry.file_name().into_string() {
            Ok(value) => value,
            Err(_) => continue,
        };
        let Some((prompt_count, tokens, layer_scope)) =
            parse_micro_cache_filename(model_path, &file_name)
        else {
            continue;
        };
        let path = entry.path();
        let Some(candidate) = load_shared_ranking_artifact(
            &path,
            SharedRankingKind::MicroAnalyze,
            SharedRankingOrigin::LegacyCache,
            Some(prompt_count),
            Some(tokens),
            Some(layer_scope),
        ) else {
            continue;
        };
        if best
            .as_ref()
            .map(|current| better_shared_ranking(&candidate, current))
            .unwrap_or(true)
        {
            best = Some(candidate);
        }
    }
    best
}

pub fn cache_shared_ranking_if_stronger(
    model_path: &Path,
    artifact: &SharedRankingArtifact,
) -> anyhow::Result<bool> {
    validate_shared_ranking_artifact(model_path, artifact)?;

    if let Some(current) = best_shared_ranking_artifact(model_path) {
        let stronger = better_shared_ranking(artifact, &current);
        let upgrades_legacy_metadata = !stronger
            && ranking_strength_key(artifact) == ranking_strength_key(&current)
            && current.origin == SharedRankingOrigin::LegacyCache
            && artifact.origin != SharedRankingOrigin::LegacyCache;
        if !stronger && !upgrades_legacy_metadata {
            return Ok(false);
        }
    }

    let path = shared_ranking_cache_path(model_path, artifact);
    write_shared_ranking_artifact(&path, artifact)?;
    Ok(true)
}

pub fn validate_shared_ranking_artifact(
    model_path: &Path,
    artifact: &SharedRankingArtifact,
) -> anyhow::Result<()> {
    if artifact.ranking.is_empty() {
        anyhow::bail!("cannot cache empty shared ranking");
    }

    let expected = detect_moe(model_path).ok_or_else(|| {
        anyhow::anyhow!(
            "cannot validate MoE ranking for non-MoE or unreadable model {}",
            model_path.display()
        )
    })?;
    let expected_len = expected.expert_count as usize;
    if artifact.ranking.len() != expected_len {
        anyhow::bail!(
            "invalid ranking length for {}: expected {}, got {}",
            model_path.display(),
            expected_len,
            artifact.ranking.len()
        );
    }

    let mut seen = vec![false; expected_len];
    for &expert_id in &artifact.ranking {
        let index = expert_id as usize;
        if index >= expected_len {
            anyhow::bail!(
                "invalid expert id {} for {} experts in {}",
                expert_id,
                expected_len,
                model_path.display()
            );
        }
        if seen[index] {
            anyhow::bail!(
                "duplicate expert id {} in ranking for {}",
                expert_id,
                model_path.display()
            );
        }
        seen[index] = true;
    }

    Ok(())
}

pub fn shared_ranking_cache_path(model_path: &Path, artifact: &SharedRankingArtifact) -> PathBuf {
    match artifact.kind {
        SharedRankingKind::Analyze => ranking_cache_path(model_path),
        SharedRankingKind::MicroAnalyze => micro_ranking_cache_path(
            model_path,
            artifact.micro_prompt_count.unwrap_or(1),
            artifact.micro_tokens.unwrap_or(8),
            artifact.micro_layer_scope.unwrap_or_default(),
        ),
    }
}

// ── Expert assignment ──

/// Expert assignment for a single node: which expert IDs it should hold.
#[derive(Clone, Debug)]
pub struct NodeAssignment {
    /// All expert IDs for this node (shared core + unique shard), sorted.
    pub experts: Vec<u32>,
    /// How many of these are shared (replicated to every node).
    pub n_shared: usize,
    /// How many are unique to this node.
    pub n_unique: usize,
}

/// Compute expert assignments for N nodes using the overlap strategy.
///
/// - `ranking`: expert IDs sorted by gate mass descending (hottest first)
/// - `n_nodes`: number of mesh nodes to split across
/// - `min_experts`: minimum experts per node for coherent output
///
/// Returns one NodeAssignment per node. Every expert appears in at least one node.
/// Convenience wrapper for compute_assignments_with_overlap with overlap=1.
#[cfg(test)]
pub fn compute_assignments(
    ranking: &[u32],
    n_nodes: usize,
    min_experts: u32,
) -> Vec<NodeAssignment> {
    compute_assignments_with_overlap(ranking, n_nodes, min_experts, 1)
}

/// Compute expert assignments with a configurable overlap factor.
///
/// - `overlap`: how many nodes each expert should live on (1 = no redundancy,
///   2 = every expert on at least 2 nodes, etc.). Capped at n_nodes.
///
/// Strategy:
/// 1. Shared core = top `min_experts` by gate mass → replicated to every node
/// 2. Remaining experts distributed with `overlap` copies each
///
/// With overlap=2, losing any single node doesn't orphan any expert —
/// at least one other node still has it.
pub fn compute_assignments_with_overlap(
    ranking: &[u32],
    n_nodes: usize,
    min_experts: u32,
    overlap: usize,
) -> Vec<NodeAssignment> {
    let n_expert = ranking.len();
    let min_exp = min_experts as usize;
    let overlap = overlap.min(n_nodes).max(1);

    if n_nodes <= 1 || min_exp >= n_expert {
        // Single node or core covers everything — give everyone all experts
        return vec![
            NodeAssignment {
                experts: ranking.to_vec(),
                n_shared: n_expert,
                n_unique: 0,
            };
            n_nodes.max(1)
        ];
    }

    // Shared core = top min_experts by gate mass (replicated to every node)
    let shared_core: Vec<u32> = ranking[..min_exp].to_vec();

    // Remaining experts to distribute with overlap
    let remaining: Vec<u32> = ranking[min_exp..].to_vec();

    // With overlap, each expert goes to `overlap` nodes.
    // Total expert-slots = remaining.len() * overlap, distributed round-robin.
    let mut node_experts: Vec<Vec<u32>> = vec![Vec::new(); n_nodes];

    for (i, &expert_id) in remaining.iter().enumerate() {
        // Assign to `overlap` consecutive nodes (wrapping)
        for j in 0..overlap {
            let node = (i + j) % n_nodes;
            node_experts[node].push(expert_id);
        }
    }

    let mut assignments = Vec::with_capacity(n_nodes);
    for node_exps in node_experts {
        let n_unique = node_exps.len();
        let mut experts = shared_core.clone();
        experts.extend_from_slice(&node_exps);
        experts.sort();
        experts.dedup(); // in case overlap wraps and duplicates with shared core

        assignments.push(NodeAssignment {
            experts,
            n_shared: min_exp,
            n_unique,
        });
    }

    assignments
}

/// Compute expert assignments by snake-drafting the ranking across nodes.
///
/// The first `replicate` experts are replicated to every node. Remaining experts
/// are assigned in snake order to balance hot and cold experts across nodes.
pub fn compute_snake_draft_assignments(
    ranking: &[u32],
    n_nodes: usize,
    replicate: usize,
) -> Vec<NodeAssignment> {
    let n_expert = ranking.len();
    if n_nodes <= 1 || replicate >= n_expert {
        return vec![
            NodeAssignment {
                experts: ranking.to_vec(),
                n_shared: n_expert,
                n_unique: 0,
            };
            n_nodes.max(1)
        ];
    }

    let shared_core: Vec<u32> = ranking[..replicate].to_vec();
    let remaining = &ranking[replicate..];
    let mut node_experts: Vec<Vec<u32>> = vec![Vec::new(); n_nodes];

    for (i, &expert_id) in remaining.iter().enumerate() {
        let round = i / n_nodes;
        let pos = i % n_nodes;
        let node = if round.is_multiple_of(2) {
            pos
        } else {
            n_nodes - 1 - pos
        };
        node_experts[node].push(expert_id);
    }

    node_experts
        .into_iter()
        .map(|node_unique| {
            let n_unique = node_unique.len();
            let mut experts = shared_core.clone();
            experts.extend(node_unique);
            experts.sort();
            NodeAssignment {
                experts,
                n_shared: shared_core.len(),
                n_unique,
            }
        })
        .collect()
}

/// Format expert list as comma-separated string for moe-split --expert-list.
pub fn expert_list_arg(assignment: &NodeAssignment) -> String {
    assignment
        .experts
        .iter()
        .map(|e| e.to_string())
        .collect::<Vec<_>>()
        .join(",")
}

/// Strip the `-NNNNN-of-NNNNN` split-file suffix from a GGUF stem so that the
/// cache directory name doesn't accidentally trigger llama.cpp's split-file
/// detection when loading per-node MoE shards. Multi-file GGUFs from
/// HuggingFace (e.g. `Model-00001-of-00013.gguf`) include this suffix in the
/// filename; without stripping it the resulting cache path
/// `.../splits/Model-00001-of-00013/N-nodes/node-K.gguf` causes
/// `invalid split file name` at load time.
fn strip_split_suffix(stem: &str) -> String {
    split_stem_prefix(stem).unwrap_or(stem).to_string()
}

/// Match llama.cpp split naming exactly: `-NNNNN-of-NNNNN` at the end.
fn split_stem_prefix(stem: &str) -> Option<&str> {
    let suffix = stem.rfind("-of-")?;
    let split_count = &stem[suffix + 4..];
    if split_count.len() != 5 || !split_count.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }

    let dash = stem[..suffix].rfind('-')?;
    let split_index = &stem[dash + 1..suffix];
    if split_index.len() != 5 || !split_index.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }

    Some(&stem[..dash])
}

/// Path to the cached split GGUF for a given model + node count + node index.
pub fn split_path(model_path: &Path, n_nodes: usize, node_index: usize) -> PathBuf {
    let stem = model_path.file_stem().unwrap_or_default().to_string_lossy();
    let clean_stem = strip_split_suffix(&stem);
    let new_path = split_cache_root()
        .join(&clean_stem)
        .join(format!("{n_nodes}-nodes"))
        .join(format!("node-{node_index}.gguf"));
    migrate_legacy_split_if_present(model_path, n_nodes, node_index, &new_path);
    new_path
}

fn legacy_split_path_for_stem(
    model_path: &Path,
    stem: &str,
    n_nodes: usize,
    node_index: usize,
) -> PathBuf {
    let dir = model_path.parent().unwrap_or(Path::new("."));
    dir.join("moe-splits")
        .join(stem)
        .join(format!("{n_nodes}-nodes"))
        .join(format!("node-{node_index}.gguf"))
}

fn legacy_split_paths(model_path: &Path, n_nodes: usize, node_index: usize) -> Vec<PathBuf> {
    let stem = model_path.file_stem().unwrap_or_default().to_string_lossy();
    let stem = stem.as_ref();
    let stripped_stem = strip_split_suffix(stem);
    // Backward compatibility for v0.63.0: migrate legacy MoE shards from both
    // the old raw-stem directory and the new stripped-stem directory. This can
    // be removed in a future version once older cache layouts no longer matter.
    let mut paths = vec![legacy_split_path_for_stem(
        model_path, stem, n_nodes, node_index,
    )];
    if stripped_stem != stem {
        paths.push(legacy_split_path_for_stem(
            model_path,
            &stripped_stem,
            n_nodes,
            node_index,
        ));
    }
    paths
}

fn migrate_legacy_split_if_present(
    model_path: &Path,
    n_nodes: usize,
    node_index: usize,
    new_path: &Path,
) {
    if new_path.exists() {
        return;
    }

    let Some(legacy_path) = legacy_split_paths(model_path, n_nodes, node_index)
        .into_iter()
        .find(|path| path.exists())
    else {
        return;
    };

    if let Some(parent) = new_path.parent() {
        if std::fs::create_dir_all(parent).is_err() {
            return;
        }
    }

    if std::fs::rename(&legacy_path, new_path).is_ok() {
        cleanup_empty_legacy_split_dirs(&legacy_path);
        return;
    }

    if std::fs::copy(&legacy_path, new_path).is_ok() {
        let _ = std::fs::remove_file(&legacy_path);
        cleanup_empty_legacy_split_dirs(&legacy_path);
    }
}

fn cleanup_empty_legacy_split_dirs(legacy_path: &Path) {
    let Some(node_dir) = legacy_path.parent() else {
        return;
    };
    let Some(model_dir) = node_dir.parent() else {
        return;
    };
    let Some(root_dir) = model_dir.parent() else {
        return;
    };

    for dir in [node_dir, model_dir, root_dir] {
        let Ok(mut entries) = std::fs::read_dir(dir) else {
            break;
        };
        if entries.next().is_none() {
            let _ = std::fs::remove_dir(dir);
        } else {
            break;
        }
    }
}

fn resolve_split_binary(bin_dir: &Path) -> anyhow::Result<PathBuf> {
    let candidates = [
        bin_dir.join("llama-moe-split"),
        bin_dir.join("../llama.cpp/build/bin/llama-moe-split"),
        bin_dir.join("../../llama.cpp/build/bin/llama-moe-split"),
        bin_dir.join("../../../llama.cpp/build/bin/llama-moe-split"),
    ];

    for candidate in candidates {
        if candidate.exists() {
            return Ok(candidate.canonicalize().unwrap_or(candidate));
        }
    }

    anyhow::bail!(
        "llama-moe-split not found in {} or nearby llama.cpp/build/bin directories",
        bin_dir.display()
    );
}

/// Run llama-moe-split to produce a split GGUF for one node.
pub fn run_split(
    bin_dir: &Path,
    model_path: &Path,
    assignment: &NodeAssignment,
    output_path: &Path,
) -> anyhow::Result<()> {
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let expert_list = expert_list_arg(assignment);
    let split_bin = resolve_split_binary(bin_dir)?;
    let status = std::process::Command::new(&split_bin)
        .args([
            "-m",
            &model_path.to_string_lossy(),
            "--expert-list",
            &expert_list,
            "-o",
            &output_path.to_string_lossy(),
        ])
        .status()
        .map_err(|e| anyhow::anyhow!("Failed to run {}: {e}", split_bin.display()))?;

    anyhow::ensure!(status.success(), "llama-moe-split exited with {status}");
    Ok(())
}

// ────────────────────────────────────────────────────────────────────────────
// Trunk/experts split storage (moe.storage.mode = "split")
// ────────────────────────────────────────────────────────────────────────────

/// Splitter format we require when building trunk/experts shards. Bumped in
/// lockstep with the llama.cpp fork's `MOE_SPLIT_SPLITTER_VERSION`. When the
/// installed splitter reports a different string, we refuse to emit split
/// shards rather than silently produce shards the loader won't accept.
pub const REQUIRED_SPLITTER_VERSION: &str = "trunk-experts/v1";

/// Typed failure modes for trunk/experts storage. Distinct variants so the
/// orchestrator (election / launcher) can react differently — e.g. a
/// `NasUnreachable` should never fall back to monolithic local; a
/// `SplitterTooOld` should prompt the user to rebuild the image.
#[derive(thiserror::Error, Debug)]
pub enum MoeStorageError {
    #[error("NAS/storage unreachable at {path}: {source}")]
    NasUnreachable {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error(
        "llama-moe-split reports version {found:?} but mesh-llm requires {expected:?}; \
         rebuild llama.cpp on the current fork to get trunk/experts support"
    )]
    SplitterTooOld {
        found: String,
        expected: &'static str,
    },
    #[error("trunk hash mismatch for {path}: manifest {expected}, file {found}")]
    TrunkHashMismatch {
        path: PathBuf,
        expected: String,
        found: String,
    },
    #[error("shard size mismatch for {path}: manifest {expected} bytes, file {found} bytes")]
    ShardSizeMismatch {
        path: PathBuf,
        expected: u64,
        found: u64,
    },
    #[error("manifest not found at {0}; expected leader to publish it first")]
    ManifestMissing(PathBuf),
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct TrunkEntry {
    pub path: PathBuf,
    pub sha256: String,
    pub size_bytes: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ExpertEntry {
    pub path: PathBuf,
    pub sha256: String,
    pub size_bytes: u64,
}

/// Manifest describing a complete trunk/experts split. Lives at
/// `<trunk_root>/manifests/<model_stem>/<version>.json`. Published atomically
/// (tmp + fsync + rename) by the elected leader after it finishes producing
/// trunk + every node's experts shard. Followers poll for its presence.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct SplitManifest {
    /// `manifest.version`: stable derived ID (see `derive_manifest_version`).
    /// New plan ⇒ new version directory; old manifests remain for forensics.
    pub version: String,
    /// SHA-256 of the source monolithic GGUF at split time.
    pub source_model_sha256: String,
    /// Value of `llama-moe-split --splitter-version` used to build these
    /// shards. Tells a future loader which splitter produced the shards
    /// without re-opening the GGUF.
    pub splitter_version: String,
    /// Mesh sizing at plan time.
    pub n_nodes: usize,
    /// Per-node expert assignment exactly as passed to the splitter.
    pub assignments: BTreeMap<usize, Vec<u32>>,
    pub trunk: TrunkEntry,
    pub experts: BTreeMap<usize, ExpertEntry>,
    pub created_at: DateTime<Utc>,
}

/// Paths + version id resolved for one node's view of a trunk/experts split.
/// This is what the launcher hands to llama-server as
/// `--model-trunk <trunk>` + `--model-experts <experts>`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolvedShards {
    pub trunk: PathBuf,
    pub experts: PathBuf,
    pub manifest_version: String,
}

/// Directory layout helpers. All paths are derived from the `MoeStorageConfig`
/// roots so nodes that mount the NAS at different points still agree on
/// structure.
pub fn trunk_file_path(
    cfg: &crate::plugin::MoeStorageConfig,
    model_path: &Path,
    version: &str,
) -> Option<PathBuf> {
    let root = crate::plugin::resolve_trunk_root(cfg)?;
    let stem = model_stem_for_split(model_path);
    Some(root.join(&stem).join(version).join("trunk.gguf"))
}

pub fn experts_file_path(
    cfg: &crate::plugin::MoeStorageConfig,
    model_path: &Path,
    n_nodes: usize,
    node_index: usize,
    version: &str,
) -> Option<PathBuf> {
    let root = crate::plugin::resolve_experts_root(cfg)?;
    let stem = model_stem_for_split(model_path);
    Some(
        root.join(&stem)
            .join(format!("{n_nodes}-nodes"))
            .join(version)
            .join(format!("node-{node_index}-experts.gguf")),
    )
}

pub fn manifest_file_path(
    cfg: &crate::plugin::MoeStorageConfig,
    model_path: &Path,
    version: &str,
) -> Option<PathBuf> {
    let root = crate::plugin::resolve_trunk_root(cfg)?;
    let stem = model_stem_for_split(model_path);
    Some(
        root.join("manifests")
            .join(&stem)
            .join(format!("{version}.json")),
    )
}

/// Reuse the same stem-stripping rule as monolithic `split_path` so that
/// multi-file source GGUFs (e.g. `Kimi-K2.5-Q4_K_M-00001-of-00013.gguf`) map
/// to a single stem (`Kimi-K2.5-Q4_K_M`).
fn model_stem_for_split(model_path: &Path) -> String {
    let stem = model_path.file_stem().unwrap_or_default().to_string_lossy();
    // strip_split_suffix already returns an owned String.
    strip_split_suffix(&stem)
}

/// Derive a short, stable manifest version ID from the inputs that
/// determine the split output bit-for-bit. Any change in source model,
/// splitter, assignments, or node count produces a new version directory;
/// identical inputs share the directory so re-runs skip redundant writes.
pub fn derive_manifest_version(
    source_model_sha256: &str,
    splitter_version: &str,
    n_nodes: usize,
    assignments: &BTreeMap<usize, Vec<u32>>,
) -> String {
    let mut h = Sha256::new();
    h.update(source_model_sha256.as_bytes());
    h.update(b"|");
    h.update(splitter_version.as_bytes());
    h.update(b"|");
    h.update(n_nodes.to_le_bytes());
    for (idx, experts) in assignments {
        h.update(b"|node=");
        h.update(idx.to_le_bytes());
        for e in experts {
            h.update(b",");
            h.update(e.to_le_bytes());
        }
    }
    let digest = h.finalize();
    hex_lower(&digest[..6])
}

fn hex_lower(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0x0f) as usize] as char);
    }
    out
}

/// Streaming SHA-256 of a file. Used to verify shards match the manifest.
pub fn sha256_file(path: &Path) -> anyhow::Result<String> {
    use std::io::Read as _;
    let mut f = std::fs::File::open(path)
        .map_err(|e| anyhow::anyhow!("opening {} for hashing: {e}", path.display()))?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 4 * 1024 * 1024];
    loop {
        let n = f
            .read(&mut buf)
            .map_err(|e| anyhow::anyhow!("reading {} for hashing: {e}", path.display()))?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(hex_lower(&h.finalize()))
}

/// Probe `llama-moe-split --splitter-version`. Returns the reported string,
/// or `SplitterTooOld` when it does not match `REQUIRED_SPLITTER_VERSION`.
pub fn probe_splitter_version(bin_dir: &Path) -> anyhow::Result<String> {
    let split_bin = resolve_split_binary(bin_dir)?;
    let output = std::process::Command::new(&split_bin)
        .arg("--splitter-version")
        .output()
        .map_err(|e| anyhow::anyhow!("probing {} version: {e}", split_bin.display()))?;
    if !output.status.success() {
        anyhow::bail!(
            "llama-moe-split --splitter-version exited {} (stderr={})",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    let version = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if version != REQUIRED_SPLITTER_VERSION {
        return Err(MoeStorageError::SplitterTooOld {
            found: version,
            expected: REQUIRED_SPLITTER_VERSION,
        }
        .into());
    }
    Ok(version)
}

/// Run the splitter in trunk/experts mode for one node. Writes both
/// `trunk_out` and `experts_out` atomically (the C++ side uses .tmp+rename).
/// `reuse_trunk` tells the splitter to keep an existing trunk when its hash
/// still matches; set to true for any invocation past the first node.
pub fn run_split_trunk_experts(
    bin_dir: &Path,
    model_path: &Path,
    assignment: &NodeAssignment,
    trunk_out: &Path,
    experts_out: &Path,
    reuse_trunk: bool,
) -> anyhow::Result<()> {
    if let Some(p) = trunk_out.parent() {
        std::fs::create_dir_all(p)?;
    }
    if let Some(p) = experts_out.parent() {
        std::fs::create_dir_all(p)?;
    }

    let expert_list = expert_list_arg(assignment);
    let split_bin = resolve_split_binary(bin_dir)?;
    let mut cmd = std::process::Command::new(&split_bin);
    cmd.args([
        "-m",
        &model_path.to_string_lossy(),
        "--expert-list",
        &expert_list,
        "--trunk-out",
        &trunk_out.to_string_lossy(),
        "--experts-out",
        &experts_out.to_string_lossy(),
    ]);
    if reuse_trunk {
        cmd.arg("--reuse-trunk");
    }
    let status = cmd
        .status()
        .map_err(|e| anyhow::anyhow!("failed to run {}: {e}", split_bin.display()))?;
    anyhow::ensure!(
        status.success(),
        "llama-moe-split (trunk/experts) exited with {status}"
    );
    Ok(())
}

/// Acquire an exclusive advisory lock on a lockfile sibling to the trunk path.
/// Used so concurrent splitter invocations across nodes don't race when
/// producing the shared trunk for the first time. The lock is held for the
/// duration of `body` and released when the File drops at the end.
pub fn with_trunk_build_lock<T>(
    lock_path: &Path,
    body: impl FnOnce() -> anyhow::Result<T>,
) -> anyhow::Result<T> {
    if let Some(p) = lock_path.parent() {
        std::fs::create_dir_all(p).map_err(|e| {
            anyhow::anyhow!("creating lock parent {}: {e}", p.display())
        })?;
    }
    let lock_file = std::fs::OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .truncate(false)
        .open(lock_path)
        .map_err(|e| anyhow::anyhow!("opening lock {}: {e}", lock_path.display()))?;
    // SAFETY: flock is async-signal-safe; we block until we get the lock.
    use std::os::unix::io::AsRawFd as _;
    let fd = lock_file.as_raw_fd();
    let rc = unsafe { libc::flock(fd, libc::LOCK_EX) };
    if rc != 0 {
        let err = std::io::Error::last_os_error();
        anyhow::bail!("flock({}): {err}", lock_path.display());
    }
    let result = body();
    // LOCK_UN is implicit on close; we don't rely on it here since dropping
    // lock_file on return will unlock. Keep lock_file alive through body.
    drop(lock_file);
    result
}

/// Write a JSON manifest atomically: tmp + fsync + rename, so readers never
/// see a half-written file.
pub fn write_manifest_atomic(path: &Path, manifest: &SplitManifest) -> anyhow::Result<()> {
    use std::io::Write as _;
    if let Some(p) = path.parent() {
        std::fs::create_dir_all(p)?;
    }
    let tmp = path.with_extension("json.tmp");
    {
        let mut f = std::fs::File::create(&tmp)
            .map_err(|e| anyhow::anyhow!("create {}: {e}", tmp.display()))?;
        let body = serde_json::to_vec_pretty(manifest)
            .map_err(|e| anyhow::anyhow!("serializing manifest: {e}"))?;
        f.write_all(&body)
            .map_err(|e| anyhow::anyhow!("writing {}: {e}", tmp.display()))?;
        f.sync_all()
            .map_err(|e| anyhow::anyhow!("fsync {}: {e}", tmp.display()))?;
    }
    std::fs::rename(&tmp, path)
        .map_err(|e| anyhow::anyhow!("rename {} -> {}: {e}", tmp.display(), path.display()))?;
    Ok(())
}

pub fn load_manifest(path: &Path) -> anyhow::Result<SplitManifest> {
    let data = std::fs::read(path)
        .map_err(|e| anyhow::anyhow!("reading manifest {}: {e}", path.display()))?;
    let m: SplitManifest = serde_json::from_slice(&data)
        .map_err(|e| anyhow::anyhow!("parsing manifest {}: {e}", path.display()))?;
    Ok(m)
}

/// Pre-launch integrity check. Compares this node's resolved trunk + experts
/// files against what the manifest declared. Cheap-ish: size always, SHA256
/// only when `full` is true (too expensive per-launch on Kimi K2.5 scale —
/// callers should use `full=false` for hot path and `true` for migration/
/// audit).
pub fn verify_shards(
    resolved: &ResolvedShards,
    manifest: &SplitManifest,
    node_index: usize,
    full: bool,
) -> anyhow::Result<()> {
    let expert = manifest.experts.get(&node_index).ok_or_else(|| {
        anyhow::anyhow!(
            "manifest has no expert entry for node {node_index} (has nodes {:?})",
            manifest.experts.keys().collect::<Vec<_>>()
        )
    })?;

    for (path, expected_size, expected_hash) in [
        (
            resolved.trunk.as_path(),
            manifest.trunk.size_bytes,
            &manifest.trunk.sha256,
        ),
        (resolved.experts.as_path(), expert.size_bytes, &expert.sha256),
    ] {
        let meta = std::fs::metadata(path).map_err(|e| MoeStorageError::NasUnreachable {
            path: path.to_path_buf(),
            source: e,
        })?;
        if meta.len() != expected_size {
            return Err(MoeStorageError::ShardSizeMismatch {
                path: path.to_path_buf(),
                expected: expected_size,
                found: meta.len(),
            }
            .into());
        }
        if full {
            let got = sha256_file(path)?;
            if &got != expected_hash {
                return Err(MoeStorageError::TrunkHashMismatch {
                    path: path.to_path_buf(),
                    expected: expected_hash.clone(),
                    found: got,
                }
                .into());
            }
        }
    }
    Ok(())
}

/// Cheap, stable identity for a source GGUF: `sha256(filename | size)[..6]`.
/// Used for manifest version derivation so we don't have to hash hundreds of
/// GB on every launch. Full integrity comes from `source_model_sha256` stored
/// in the manifest, computed exactly once at split time.
pub fn source_fingerprint(model_path: &Path) -> anyhow::Result<String> {
    let meta = std::fs::metadata(model_path).map_err(|e| {
        anyhow::anyhow!("stat {} for fingerprint: {e}", model_path.display())
    })?;
    let mut h = Sha256::new();
    let name = model_path
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    h.update(name.as_bytes());
    h.update(b"|");
    h.update(meta.len().to_le_bytes());
    Ok(hex_lower(&h.finalize()[..6]))
}

/// Side-input for `ensure_trunk_and_expert`. The caller (election/planner)
/// passes its own view of model + assignments + node index. `is_leader`
/// gates whether we are allowed to *build* the trunk+experts — a false value
/// means "only poll for an existing manifest; fail if not present yet".
#[derive(Clone, Debug)]
pub struct EnsureOpts {
    pub bin_dir: PathBuf,
    pub model_path: PathBuf,
    pub assignments: BTreeMap<usize, Vec<u32>>,
    pub node_index: usize,
    pub is_leader: bool,
    /// Max wall time a non-leader will wait for the leader's manifest to
    /// appear (e.g. while the leader is still producing trunk+experts).
    pub follower_timeout: std::time::Duration,
}

/// Top-level orchestrator: return the (trunk, experts) paths the launcher
/// should hand to llama-server, after ensuring the on-disk split exists and
/// matches the manifest.
///
/// Leader path: produce trunk + this node's experts + manifest (atomic).
/// Follower path: poll for the manifest; when it appears, verify shards.
pub fn ensure_trunk_and_expert(
    cfg: &crate::plugin::MoeStorageConfig,
    opts: &EnsureOpts,
) -> anyhow::Result<ResolvedShards> {
    if !matches!(
        cfg.mode,
        crate::plugin::MoeStorageMode::Split
    ) {
        anyhow::bail!("ensure_trunk_and_expert called with storage.mode != split");
    }

    let splitter_version = probe_splitter_version(&opts.bin_dir)?;
    let fingerprint = source_fingerprint(&opts.model_path)?;
    // Version is derived purely from inputs that determine output bits.
    // We prefix the cheap fingerprint so two distinct models with the same
    // assignments map to different version dirs.
    let mut assignments_sorted: BTreeMap<usize, Vec<u32>> = opts.assignments.clone();
    for v in assignments_sorted.values_mut() {
        v.sort_unstable();
        v.dedup();
    }
    let n_nodes = assignments_sorted.len();
    let version_seed = format!("{fingerprint}:{splitter_version}");
    let version =
        derive_manifest_version(&version_seed, &splitter_version, n_nodes, &assignments_sorted);

    let trunk = trunk_file_path(cfg, &opts.model_path, &version)
        .ok_or_else(|| anyhow::anyhow!("moe.storage.trunk_path not configured"))?;
    let experts = experts_file_path(
        cfg,
        &opts.model_path,
        n_nodes,
        opts.node_index,
        &version,
    )
    .ok_or_else(|| anyhow::anyhow!("moe.storage.experts_path not configured"))?;
    let manifest_path = manifest_file_path(cfg, &opts.model_path, &version)
        .ok_or_else(|| anyhow::anyhow!("moe.storage.trunk_path not configured"))?;

    // Fast path: manifest exists AND already contains our node's expert
    // entry → size-verify and return. If the manifest exists but is missing
    // our entry (e.g. a sibling node published first), fall through so a
    // leader run can *extend* it.
    if let Ok(manifest) = load_manifest(&manifest_path) {
        if manifest.experts.contains_key(&opts.node_index) {
            let resolved = ResolvedShards {
                trunk: trunk.clone(),
                experts: experts.clone(),
                manifest_version: version.clone(),
            };
            verify_shards(&resolved, &manifest, opts.node_index, /*full=*/ false)?;
            return Ok(resolved);
        }
    }

    if !opts.is_leader {
        // Followers never build — they wait for the leader to publish a
        // manifest that includes this node's expert entry.
        let deadline = std::time::Instant::now() + opts.follower_timeout;
        loop {
            if let Ok(manifest) = load_manifest(&manifest_path) {
                if manifest.experts.contains_key(&opts.node_index) {
                    let resolved = ResolvedShards {
                        trunk: trunk.clone(),
                        experts: experts.clone(),
                        manifest_version: version.clone(),
                    };
                    verify_shards(&resolved, &manifest, opts.node_index, /*full=*/ false)?;
                    return Ok(resolved);
                }
            }
            if std::time::Instant::now() >= deadline {
                return Err(MoeStorageError::ManifestMissing(manifest_path.clone()).into());
            }
            std::thread::sleep(std::time::Duration::from_secs(2));
        }
    }

    // Leader path: hold the trunk build lock, produce THIS node's shards
    // (other nodes produce their own experts shards under the same trunk
    // path with --reuse-trunk), and publish (or extend) the manifest.
    let lock_path = trunk
        .parent()
        .map(|p| p.join(".trunk.build.lock"))
        .unwrap_or_else(|| PathBuf::from("/tmp/.mesh-llm-trunk.build.lock"));

    with_trunk_build_lock(&lock_path, || -> anyhow::Result<()> {
        // Re-check under the lock: another process may have added our entry
        // while we waited. Idempotent early-exit.
        if let Ok(m) = load_manifest(&manifest_path) {
            if m.experts.contains_key(&opts.node_index) {
                return Ok(());
            }
        }

        // Produce trunk + experts for this node.
        let this_assignment = opts
            .assignments
            .get(&opts.node_index)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "leader node index {} missing from assignments (have {:?})",
                    opts.node_index,
                    opts.assignments.keys().collect::<Vec<_>>()
                )
            })?;
        let assignment_struct = NodeAssignment {
            experts: this_assignment.clone(),
            n_shared: 0,
            n_unique: this_assignment.len(),
        };
        let reuse = trunk.exists();
        run_split_trunk_experts(
            &opts.bin_dir,
            &opts.model_path,
            &assignment_struct,
            &trunk,
            &experts,
            reuse,
        )?;

        // If a manifest already exists for this version, append our node
        // to its experts map; otherwise create a fresh manifest.
        let trunk_meta = std::fs::metadata(&trunk)?;
        let experts_meta = std::fs::metadata(&experts)?;
        let experts_sha = sha256_file(&experts)?;
        let this_entry = ExpertEntry {
            path: experts.clone(),
            sha256: experts_sha,
            size_bytes: experts_meta.len(),
        };

        // If there's already a manifest, inherit its trunk entry (we keep
        // the first leader's trunk hash — trunk is shared and identical
        // across nodes by design, so every leader writes the same bits).
        // Otherwise compute the trunk hash ourselves. This keeps the
        // append-a-node case cheap: no 150 GB re-hash per late joiner.
        let (experts_map_base, trunk_entry) = match load_manifest(&manifest_path) {
            Ok(existing) => (existing.experts, existing.trunk),
            Err(_) => {
                let trunk_sha = sha256_file(&trunk)?;
                (
                    BTreeMap::new(),
                    TrunkEntry {
                        path: trunk.clone(),
                        sha256: trunk_sha,
                        size_bytes: trunk_meta.len(),
                    },
                )
            }
        };
        let mut experts_map = experts_map_base;
        experts_map.insert(opts.node_index, this_entry);

        // Full source hash is expensive on 600 GB models; we compute it
        // lazily only when the caller has a cached value or asks for it.
        // For MVP we skip it (empty string means "not computed") — the
        // fingerprint already handled version identity, and integrity
        // comes from trunk/experts hashes which ARE computed.
        let manifest = SplitManifest {
            version: version.clone(),
            source_model_sha256: String::new(),
            splitter_version: splitter_version.clone(),
            n_nodes,
            assignments: assignments_sorted.clone(),
            trunk: trunk_entry,
            experts: experts_map,
            created_at: Utc::now(),
        };
        write_manifest_atomic(&manifest_path, &manifest)?;
        Ok(())
    })?;

    // Re-load + verify.
    let manifest = load_manifest(&manifest_path)?;
    let resolved = ResolvedShards {
        trunk,
        experts,
        manifest_version: version,
    };
    verify_shards(&resolved, &manifest, opts.node_index, /*full=*/ false)?;
    Ok(resolved)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::gguf::detect_moe;
    use serial_test::serial;

    #[test]
    fn test_assignments_2_nodes() {
        // 10 experts, min 4, 2 nodes
        let ranking: Vec<u32> = (0..10).collect();
        let assignments = compute_assignments(&ranking, 2, 4);

        assert_eq!(assignments.len(), 2);
        // Each node: 4 shared + 3 unique = 7 experts
        assert_eq!(assignments[0].experts.len(), 7);
        assert_eq!(assignments[1].experts.len(), 7);
        assert_eq!(assignments[0].n_shared, 4);
        assert_eq!(assignments[0].n_unique, 3);

        // Shared core (0-3) in both
        for e in 0..4 {
            assert!(assignments[0].experts.contains(&e));
            assert!(assignments[1].experts.contains(&e));
        }

        // Full coverage
        let mut all: Vec<u32> = assignments[0].experts.clone();
        all.extend(&assignments[1].experts);
        all.sort();
        all.dedup();
        assert_eq!(all, (0..10).collect::<Vec<u32>>());
    }

    #[test]
    fn test_assignments_3_nodes() {
        // 128 experts, min 46, 3 nodes
        let ranking: Vec<u32> = (0..128).collect();
        let assignments = compute_assignments(&ranking, 3, 46);

        assert_eq!(assignments.len(), 3);
        // 82 remaining / 3 = 27 each + 1 leftover
        // Nodes 0: 46+28=74, Node 1: 46+27=73, Node 2: 46+27=73
        assert_eq!(assignments[0].experts.len(), 74);
        assert_eq!(assignments[1].experts.len(), 73);
        assert_eq!(assignments[2].experts.len(), 73);

        // Full coverage
        let mut all: Vec<u32> = Vec::new();
        for a in &assignments {
            all.extend(&a.experts);
        }
        all.sort();
        all.dedup();
        assert_eq!(all, (0..128).collect::<Vec<u32>>());
    }

    #[test]
    fn test_ranking_cache_roundtrip() {
        let dir = std::env::temp_dir().join("moe-test-ranking");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.csv");

        let ranking: Vec<u32> = vec![0, 26, 41, 69, 104, 3, 7, 99];
        let content: String = ranking
            .iter()
            .map(|e| e.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(&path, content).unwrap();

        let loaded = load_cached_ranking(&path).unwrap();
        assert_eq!(loaded, ranking);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_shared_ranking_metadata_roundtrip() {
        let dir = std::env::temp_dir().join("moe-test-shared-ranking");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("shared.csv");

        let artifact = SharedRankingArtifact {
            kind: SharedRankingKind::MicroAnalyze,
            origin: SharedRankingOrigin::PeerImport,
            ranking: vec![3, 1, 2, 0],
            micro_prompt_count: Some(2),
            micro_tokens: Some(16),
            micro_layer_scope: Some(MoeMicroLayerScope::All),
        };
        write_shared_ranking_artifact(&path, &artifact).unwrap();

        let loaded = load_shared_ranking_artifact(
            &path,
            SharedRankingKind::MicroAnalyze,
            SharedRankingOrigin::LegacyCache,
            Some(1),
            Some(8),
            Some(MoeMicroLayerScope::First),
        )
        .unwrap();
        assert_eq!(loaded, artifact);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_load_moe_analyze_csv() {
        // The CSV format from moe-analyze: expert_id,total_mass,mass_fraction,selection_count
        let dir = std::env::temp_dir().join("moe-test-csv");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("ranking.csv");
        std::fs::write(
            &path,
            "expert_id,total_mass,mass_fraction,selection_count\n\
            0,8365.69,0.250,15680\n\
            26,267.43,0.008,4800\n\
            41,250.11,0.007,4600\n",
        )
        .unwrap();

        let loaded = load_cached_ranking(&path).unwrap();
        assert_eq!(loaded, vec![0, 26, 41]);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    #[serial]
    fn split_path_uses_mesh_cache_root() {
        let prev_xdg = std::env::var_os("XDG_CACHE_HOME");
        std::env::set_var("XDG_CACHE_HOME", "/tmp/mesh-llm-cache-root");

        let model = Path::new("/tmp/models/Qwen3-30B-A3B-Q4_K_M.gguf");
        let path = split_path(model, 3, 1);

        assert_eq!(
            path,
            PathBuf::from("/tmp/mesh-llm-cache-root")
                .join("mesh-llm")
                .join("splits")
                .join("Qwen3-30B-A3B-Q4_K_M")
                .join("3-nodes")
                .join("node-1.gguf")
        );

        restore_env("XDG_CACHE_HOME", prev_xdg);
    }

    #[test]
    fn strip_split_suffix_removes_huggingface_shard_pattern() {
        assert_eq!(
            strip_split_suffix("Kimi-K2.5-Q4_K_M-00001-of-00013"),
            "Kimi-K2.5-Q4_K_M"
        );
        assert_eq!(
            strip_split_suffix("DeepSeek-V3-BF16-00001-of-00163"),
            "DeepSeek-V3-BF16"
        );
        assert_eq!(strip_split_suffix("model-1-of-2"), "model-1-of-2");
        assert_eq!(strip_split_suffix("model-001-of-003"), "model-001-of-003");
        // Single-file GGUFs should pass through unchanged
        assert_eq!(
            strip_split_suffix("Qwen3-30B-A3B-Q4_K_M"),
            "Qwen3-30B-A3B-Q4_K_M"
        );
        // Don't strip if the pattern doesn't look like digits
        assert_eq!(
            strip_split_suffix("model-name-of-something"),
            "model-name-of-something"
        );
    }

    #[test]
    #[serial]
    fn split_path_strips_multifile_suffix() {
        let prev_xdg = std::env::var_os("XDG_CACHE_HOME");
        std::env::set_var("XDG_CACHE_HOME", "/tmp/mesh-llm-cache-root");

        // Multi-file GGUF: the -00001-of-00013 should be stripped from the cache dir
        let model = Path::new("/models/Kimi-K2.5-Q4_K_M-00001-of-00013.gguf");
        let path = split_path(model, 29, 22);

        assert_eq!(
            path,
            PathBuf::from("/tmp/mesh-llm-cache-root")
                .join("mesh-llm")
                .join("splits")
                .join("Kimi-K2.5-Q4_K_M")
                .join("29-nodes")
                .join("node-22.gguf")
        );

        restore_env("XDG_CACHE_HOME", prev_xdg);
    }

    #[test]
    #[serial]
    fn split_path_migrates_legacy_split_if_present() {
        let prev_xdg = std::env::var_os("XDG_CACHE_HOME");
        let base =
            std::env::temp_dir().join(format!("mesh-llm-moe-migrate-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let cache_root = base.join("cache");
        let model_root = base.join("models");
        let model_path = model_root.join("demo.gguf");
        let legacy = model_root
            .join("moe-splits")
            .join("demo")
            .join("2-nodes")
            .join("node-0.gguf");
        std::fs::create_dir_all(legacy.parent().unwrap()).unwrap();
        std::fs::create_dir_all(&model_root).unwrap();
        std::fs::write(&model_path, b"model").unwrap();
        std::fs::write(&legacy, b"legacy-split").unwrap();
        std::env::set_var("XDG_CACHE_HOME", &cache_root);

        let new_path = split_path(&model_path, 2, 0);

        assert!(new_path.exists());
        assert_eq!(std::fs::read(&new_path).unwrap(), b"legacy-split");
        assert!(!legacy.exists());

        restore_env("XDG_CACHE_HOME", prev_xdg);
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    #[serial]
    fn split_path_migrates_legacy_split_from_unstripped_multifile_stem() {
        let prev_xdg = std::env::var_os("XDG_CACHE_HOME");
        let base = std::env::temp_dir().join(format!(
            "mesh-llm-moe-migrate-multifile-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&base);
        let cache_root = base.join("cache");
        let model_root = base.join("models");
        let model_path = model_root.join("demo-00001-of-00013.gguf");
        let legacy = model_root
            .join("moe-splits")
            .join("demo-00001-of-00013")
            .join("2-nodes")
            .join("node-0.gguf");
        std::fs::create_dir_all(legacy.parent().unwrap()).unwrap();
        std::fs::create_dir_all(&model_root).unwrap();
        std::fs::write(&model_path, b"model").unwrap();
        std::fs::write(&legacy, b"legacy-split").unwrap();
        std::env::set_var("XDG_CACHE_HOME", &cache_root);

        let new_path = split_path(&model_path, 2, 0);

        assert!(new_path.exists());
        assert_eq!(std::fs::read(&new_path).unwrap(), b"legacy-split");
        assert!(!legacy.exists());
        assert_eq!(
            new_path,
            cache_root
                .join("mesh-llm")
                .join("splits")
                .join("demo")
                .join("2-nodes")
                .join("node-0.gguf")
        );

        restore_env("XDG_CACHE_HOME", prev_xdg);
        let _ = std::fs::remove_dir_all(&base);
    }

    fn restore_env(key: &str, value: Option<std::ffi::OsString>) {
        if let Some(value) = value {
            std::env::set_var(key, value);
        } else {
            std::env::remove_var(key);
        }
    }

    #[test]
    fn test_detect_moe_qwen3() {
        let hf_cache = crate::models::huggingface_hub_cache_dir();
        let path = hf_cache.join("Qwen3-30B-A3B-Q4_K_M.gguf");
        if !path.exists() {
            eprintln!("Skipping: model file not found");
            return;
        }
        let info = detect_moe(&path).expect("Should detect MoE");
        assert_eq!(info.expert_count, 128);
        assert_eq!(info.expert_used_count, 8);
    }

    #[test]
    fn test_detect_moe_olmoe() {
        let hf_cache = crate::models::huggingface_hub_cache_dir();
        let path = hf_cache.join("olmoe-1b-7b-0924-instruct-q4_k_m.gguf");
        if !path.exists() {
            eprintln!("Skipping: OLMoE model file not found");
            return;
        }
        let info = detect_moe(&path).expect("Should detect MoE");
        assert_eq!(info.expert_count, 64);
        assert_eq!(info.expert_used_count, 8);
    }

    #[test]
    fn test_detect_moe_dense_model() {
        // Qwen2.5-3B is dense (no experts) — should return None
        let hf_cache = crate::models::huggingface_hub_cache_dir();
        let path = hf_cache.join("Qwen2.5-3B-Instruct-Q4_K_M.gguf");
        if !path.exists() {
            eprintln!("Skipping: dense model file not found");
            return;
        }
        assert!(
            detect_moe(&path).is_none(),
            "Dense model should not be detected as MoE"
        );
    }

    #[test]
    fn test_single_node() {
        let ranking: Vec<u32> = (0..8).collect();
        let assignments = compute_assignments(&ranking, 1, 4);
        assert_eq!(assignments.len(), 1);
        assert_eq!(assignments[0].experts.len(), 8); // gets everything
    }

    // ── Overlap tests ──

    #[test]
    fn test_overlap_2x_3_nodes() {
        // 128 experts, min 46, 3 nodes, 2× overlap
        let ranking: Vec<u32> = (0..128).collect();
        let assignments = compute_assignments_with_overlap(&ranking, 3, 46, 2);

        assert_eq!(assignments.len(), 3);

        // Every expert should appear in at least 2 nodes
        let mut expert_count: std::collections::HashMap<u32, usize> =
            std::collections::HashMap::new();
        for a in &assignments {
            for &e in &a.experts {
                *expert_count.entry(e).or_default() += 1;
            }
        }

        // Shared core (0..46) in all 3 nodes
        for e in 0..46 {
            assert!(
                *expert_count.get(&e).unwrap() >= 3,
                "Shared expert {e} should be in all nodes"
            );
        }
        // Remaining experts (46..128) in at least 2 nodes
        for e in 46..128 {
            assert!(
                *expert_count.get(&e).unwrap() >= 2,
                "Expert {e} should be in at least 2 nodes, got {}",
                expert_count[&e]
            );
        }
        // Full coverage
        assert_eq!(expert_count.len(), 128);
    }

    #[test]
    fn test_overlap_2x_2_nodes() {
        // With 2 nodes and 2× overlap, every remaining expert is on both nodes
        let ranking: Vec<u32> = (0..10).collect();
        let assignments = compute_assignments_with_overlap(&ranking, 2, 4, 2);

        assert_eq!(assignments.len(), 2);
        // Both nodes should have all 10 experts (4 shared + 6 remaining × 2× = both)
        assert_eq!(assignments[0].experts.len(), 10);
        assert_eq!(assignments[1].experts.len(), 10);
    }

    #[test]
    fn test_overlap_1x_same_as_original() {
        // overlap=1 should give same results as compute_assignments
        let ranking: Vec<u32> = (0..128).collect();
        let a1 = compute_assignments(&ranking, 3, 46);
        let a2 = compute_assignments_with_overlap(&ranking, 3, 46, 1);

        for i in 0..3 {
            assert_eq!(a1[i].experts, a2[i].experts);
        }
    }

    #[test]
    fn test_overlap_capped_at_n_nodes() {
        // overlap=10 with 3 nodes should cap to 3 (every expert on every node)
        let ranking: Vec<u32> = (0..20).collect();
        let assignments = compute_assignments_with_overlap(&ranking, 3, 5, 10);

        // All 3 nodes should have all 20 experts
        for a in &assignments {
            assert_eq!(a.experts.len(), 20);
        }
    }

    #[test]
    fn test_overlap_glm5_10_nodes() {
        // GLM-5: 256 experts, min 96, 10 nodes, 2× overlap
        let ranking: Vec<u32> = (0..256).collect();
        let assignments = compute_assignments_with_overlap(&ranking, 10, 96, 2);

        assert_eq!(assignments.len(), 10);

        // Full coverage
        let mut all: std::collections::HashSet<u32> = std::collections::HashSet::new();
        for a in &assignments {
            all.extend(&a.experts);
        }
        assert_eq!(all.len(), 256);

        // Every remaining expert on at least 2 nodes
        let mut expert_count: std::collections::HashMap<u32, usize> =
            std::collections::HashMap::new();
        for a in &assignments {
            for &e in &a.experts {
                *expert_count.entry(e).or_default() += 1;
            }
        }
        for e in 96..256 {
            assert!(*expert_count.get(&e).unwrap() >= 2);
        }

        // Print sizes for verification
        for (i, a) in assignments.iter().enumerate() {
            eprintln!(
                "  Node {i}: {} experts ({} shared + {} unique)",
                a.experts.len(),
                a.n_shared,
                a.n_unique
            );
        }
    }

    // ── trunk/experts split storage tests ────────────────────────────────

    fn mk_assignments(n_nodes: usize) -> BTreeMap<usize, Vec<u32>> {
        let mut m = BTreeMap::new();
        for i in 0..n_nodes {
            m.insert(i, vec![i as u32, (i + n_nodes) as u32]);
        }
        m
    }

    #[test]
    fn derive_manifest_version_is_deterministic_and_short() {
        let a = mk_assignments(2);
        let v1 = derive_manifest_version("abc:sha", "trunk-experts/v1", 2, &a);
        let v2 = derive_manifest_version("abc:sha", "trunk-experts/v1", 2, &a);
        assert_eq!(v1, v2);
        assert_eq!(v1.len(), 12); // 6 bytes hex = 12 chars
    }

    #[test]
    fn derive_manifest_version_changes_with_each_input() {
        let a = mk_assignments(2);
        let base = derive_manifest_version("abc:sha", "trunk-experts/v1", 2, &a);
        // Change each input independently; all should produce different versions.
        assert_ne!(base, derive_manifest_version("xyz:sha", "trunk-experts/v1", 2, &a));
        assert_ne!(base, derive_manifest_version("abc:sha", "trunk-experts/v2", 2, &a));
        assert_ne!(base, derive_manifest_version("abc:sha", "trunk-experts/v1", 3, &a));
        let mut a2 = a.clone();
        a2.get_mut(&0).unwrap().push(99);
        assert_ne!(base, derive_manifest_version("abc:sha", "trunk-experts/v1", 2, &a2));
    }

    #[test]
    fn sha256_file_is_deterministic_content_sensitive_and_matches_in_memory() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let content = b"mesh-llm trunk test\n";
        std::fs::write(tmp.path(), content).unwrap();
        // Must match the in-memory hash over the same bytes.
        let mut h = sha2::Sha256::new();
        sha2::Digest::update(&mut h, content);
        let expected = hex_lower(&sha2::Digest::finalize(h));
        assert_eq!(sha256_file(tmp.path()).unwrap(), expected);
        // Stable under repeated reads.
        assert_eq!(sha256_file(tmp.path()).unwrap(), expected);
        // Sensitive to content change.
        std::fs::write(tmp.path(), b"different\n").unwrap();
        assert_ne!(sha256_file(tmp.path()).unwrap(), expected);
    }

    #[test]
    fn source_fingerprint_is_stable_and_size_sensitive() {
        let a = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(a.path(), b"0123456789").unwrap();
        let f1 = source_fingerprint(a.path()).unwrap();
        let f2 = source_fingerprint(a.path()).unwrap();
        assert_eq!(f1, f2);
        // Same name, different size → different fingerprint.
        std::fs::write(a.path(), b"differentdatasizeisnotsame").unwrap();
        let f3 = source_fingerprint(a.path()).unwrap();
        assert_ne!(f1, f3);
    }

    #[test]
    fn manifest_round_trips_via_atomic_write() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("0abc.json");
        let original = SplitManifest {
            version: "0abc".into(),
            source_model_sha256: String::new(),
            splitter_version: "trunk-experts/v1".into(),
            n_nodes: 2,
            assignments: mk_assignments(2),
            trunk: TrunkEntry {
                path: dir.path().join("trunk.gguf"),
                sha256: "deadbeef".into(),
                size_bytes: 100,
            },
            experts: {
                let mut m = BTreeMap::new();
                m.insert(
                    0,
                    ExpertEntry {
                        path: dir.path().join("e0.gguf"),
                        sha256: "feedface".into(),
                        size_bytes: 50,
                    },
                );
                m
            },
            created_at: Utc::now(),
        };
        write_manifest_atomic(&path, &original).unwrap();
        // Atomic: tmp must not be left behind.
        assert!(!path.with_extension("json.tmp").exists());
        let loaded = load_manifest(&path).unwrap();
        // created_at serializes with microsecond precision; compare field-by-field
        // with a tolerance if needed. Here serde uses RFC3339 which preserves.
        assert_eq!(loaded.version, original.version);
        assert_eq!(loaded.trunk, original.trunk);
        assert_eq!(loaded.experts, original.experts);
        assert_eq!(loaded.assignments, original.assignments);
    }

    #[test]
    fn verify_shards_detects_size_mismatch() {
        let dir = tempfile::tempdir().unwrap();
        let trunk = dir.path().join("trunk.gguf");
        let experts = dir.path().join("e0.gguf");
        std::fs::write(&trunk, b"trunk-bytes-here").unwrap();
        std::fs::write(&experts, b"expert-bytes").unwrap();
        let manifest = SplitManifest {
            version: "v".into(),
            source_model_sha256: String::new(),
            splitter_version: "trunk-experts/v1".into(),
            n_nodes: 1,
            assignments: mk_assignments(1),
            trunk: TrunkEntry {
                path: trunk.clone(),
                sha256: sha256_file(&trunk).unwrap(),
                size_bytes: 999, // WRONG
            },
            experts: {
                let mut m = BTreeMap::new();
                m.insert(
                    0,
                    ExpertEntry {
                        path: experts.clone(),
                        sha256: sha256_file(&experts).unwrap(),
                        size_bytes: std::fs::metadata(&experts).unwrap().len(),
                    },
                );
                m
            },
            created_at: Utc::now(),
        };
        let resolved = ResolvedShards {
            trunk: trunk.clone(),
            experts: experts.clone(),
            manifest_version: "v".into(),
        };
        let err = verify_shards(&resolved, &manifest, 0, false).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("size mismatch"), "unexpected error: {msg}");
    }

    #[test]
    fn verify_shards_detects_hash_mismatch_when_full() {
        let dir = tempfile::tempdir().unwrap();
        let trunk = dir.path().join("trunk.gguf");
        let experts = dir.path().join("e0.gguf");
        std::fs::write(&trunk, b"trunk-v1").unwrap();
        std::fs::write(&experts, b"expert-v1").unwrap();
        let manifest = SplitManifest {
            version: "v".into(),
            source_model_sha256: String::new(),
            splitter_version: "trunk-experts/v1".into(),
            n_nodes: 1,
            assignments: mk_assignments(1),
            trunk: TrunkEntry {
                path: trunk.clone(),
                // Wrong hash (but correct size) → size-only check passes, full fails.
                sha256: "0".repeat(64),
                size_bytes: std::fs::metadata(&trunk).unwrap().len(),
            },
            experts: {
                let mut m = BTreeMap::new();
                m.insert(
                    0,
                    ExpertEntry {
                        path: experts.clone(),
                        sha256: sha256_file(&experts).unwrap(),
                        size_bytes: std::fs::metadata(&experts).unwrap().len(),
                    },
                );
                m
            },
            created_at: Utc::now(),
        };
        let resolved = ResolvedShards {
            trunk: trunk.clone(),
            experts: experts.clone(),
            manifest_version: "v".into(),
        };
        // size-only passes
        verify_shards(&resolved, &manifest, 0, false).unwrap();
        // full catches the hash mismatch
        let err = verify_shards(&resolved, &manifest, 0, true).unwrap_err();
        assert!(err.to_string().contains("hash mismatch"), "got: {err}");
    }

    #[test]
    fn trunk_build_lock_is_exclusive() {
        use std::sync::{Arc, Barrier};
        use std::thread;
        use std::time::{Duration, Instant};
        let dir = tempfile::tempdir().unwrap();
        let lock = dir.path().join(".trunk.build.lock");
        let barrier = Arc::new(Barrier::new(2));
        let lock_a = lock.clone();
        let bar_a = barrier.clone();
        let hold_ms = 300u64;
        let t1 = thread::spawn(move || {
            with_trunk_build_lock(&lock_a, || {
                bar_a.wait();
                thread::sleep(Duration::from_millis(hold_ms));
                Ok(Instant::now())
            })
        });
        barrier.wait();
        let started = Instant::now();
        let second = with_trunk_build_lock(&lock, || Ok(Instant::now())).unwrap();
        let first_done = t1.join().unwrap().unwrap();
        // Second acquire must not start until the first released.
        assert!(
            second >= first_done,
            "second acquired at {:?} before first released at {:?}",
            second.duration_since(started),
            first_done.duration_since(started)
        );
    }
}

