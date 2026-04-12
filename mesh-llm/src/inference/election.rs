//! Automatic host election and dynamic mesh management.
//!
//! Per-model election: nodes serving the same model form a group.
//! The highest-VRAM node in each group becomes its host and runs llama-server.
//! Every mesh change: kill llama-server, re-elect, winner starts fresh.
//! mesh-llm owns :api_port and proxies to the right host by model name.

use crate::inference::{launch, moe};
use crate::mesh;
use crate::models;
use crate::network::tunnel;
use crate::system::hardware;
use launch::{BinaryFlavor, SplitMode};
use mesh::NodeRole;
use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use tokio::sync::watch;

/// Returns `true` when `flavor` and `gpu_count` together call for row-split
/// tensor parallelism.
///
/// Row split requires a backend that implements `ggml_backend_split_buffer_type`
/// (CUDA and ROCm).  When no flavor is specified the binary may still be a CUDA
/// or ROCm build discovered automatically, so `None` is treated as potentially
/// supported; if the binary turns out to be CPU/Metal/Vulkan, llama.cpp falls
/// back safely.
fn should_use_row_split(flavor: Option<BinaryFlavor>, gpu_count: usize) -> bool {
    let backend_supported = matches!(
        flavor,
        Some(BinaryFlavor::Cuda) | Some(BinaryFlavor::Rocm) | None
    );
    backend_supported && gpu_count > 1
}

/// Returns `Some(SplitMode::Row)` when the local machine has multiple GPUs and
/// the llama.cpp backend supports row-level tensor parallelism (CUDA, ROCm).
///
/// Row split shards weight matrices across local GPUs so all GPUs are active on
/// every token — faster than layer (pipeline) split where GPUs take turns.
/// This does NOT work over RPC (network) — only for GPUs on the same machine.
///
/// When no explicit flavor is provided the resolved binary may still be CUDA/ROCm
/// (auto-detected from the binary name), so `None` is treated as potentially
/// supported.
pub(crate) fn local_multi_gpu_split_mode(flavor: Option<BinaryFlavor>) -> Option<SplitMode> {
    let hw = hardware::query(&[hardware::Metric::GpuCount]);
    let gpu_count = usize::from(hw.gpu_count);
    if should_use_row_split(flavor, gpu_count) {
        tracing::info!(
            "Local multi-GPU detected ({} GPUs) — using row split for tensor parallelism",
            gpu_count
        );
        Some(SplitMode::Row)
    } else {
        None
    }
}

/// Calculate total model size, summing all split files if present.
/// Split files follow the pattern: name-00001-of-00004.gguf
pub fn total_model_bytes(model: &Path) -> u64 {
    let name = model.to_string_lossy();
    // Check for split pattern: *-00001-of-NNNNN.gguf
    if let Some(pos) = name.find("-00001-of-") {
        let of_pos = pos + 10;
        if let Some(ext_pos) = name[of_pos..].find(".gguf") {
            if let Ok(n_split) = name[of_pos..of_pos + ext_pos].parse::<u32>() {
                let prefix = &name[..pos + 1];
                let suffix = &name[of_pos + ext_pos..];
                let mut total: u64 = 0;
                for i in 1..=n_split {
                    let split_name = format!("{}{:05}-of-{:05}{}", prefix, i, n_split, suffix);
                    total += std::fs::metadata(&split_name).map(|m| m.len()).unwrap_or(0);
                }
                return total;
            }
        }
    }
    std::fs::metadata(model).map(|m| m.len()).unwrap_or(0)
}

/// Determine if this node should be host for its model group.
/// Only considers peers serving the same model.
/// Deterministic: highest VRAM wins, tie-break by node ID.
pub fn should_be_host_for_model(
    my_id: iroh::EndpointId,
    my_vram: u64,
    model_peers: &[mesh::PeerInfo],
) -> bool {
    for peer in model_peers {
        if matches!(peer.role, NodeRole::Client) {
            continue;
        }
        if peer.vram_bytes > my_vram {
            return false;
        }
        if peer.vram_bytes == my_vram && peer.id > my_id {
            return false;
        }
    }
    true
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum DenseLaunchPlan {
    Solo,
    Split {
        worker_ids: Vec<iroh::EndpointId>,
        total_group_vram: u64,
    },
    WaitingForCapacity {
        worker_ids: Vec<iroh::EndpointId>,
        total_group_vram: u64,
        min_vram: u64,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum DenseRunningPlan {
    Solo,
    Split { worker_ids: Vec<iroh::EndpointId> },
}

impl DenseLaunchPlan {
    fn running_plan(&self) -> Option<DenseRunningPlan> {
        match self {
            DenseLaunchPlan::Solo => Some(DenseRunningPlan::Solo),
            DenseLaunchPlan::Split { worker_ids, .. } => Some(DenseRunningPlan::Split {
                worker_ids: worker_ids.clone(),
            }),
            DenseLaunchPlan::WaitingForCapacity { .. } => None,
        }
    }
}

fn split_peer_vram_bytes(peer: &mesh::PeerInfo, my_vram: u64) -> u64 {
    if peer.vram_bytes > 0 {
        peer.vram_bytes
    } else {
        my_vram
    }
}

fn build_dense_launch_plan(
    my_vram: u64,
    model_bytes: u64,
    force_split: bool,
    model_name: &str,
    model_peers: &[mesh::PeerInfo],
) -> DenseLaunchPlan {
    let min_vram = (model_bytes as f64 * 1.1) as u64;
    if !force_split && my_vram >= min_vram {
        return DenseLaunchPlan::Solo;
    }

    let mut candidates: Vec<_> = model_peers
        .iter()
        .filter(|p| matches!(p.role, NodeRole::Worker) || p.is_assigned_model(model_name))
        .filter(|p| !matches!(p.role, NodeRole::Client))
        .filter(|p| !matches!(p.rtt_ms, Some(rtt) if rtt > mesh::MAX_SPLIT_RTT_MS))
        .collect();
    candidates.sort_by_key(|p| (p.rtt_ms.unwrap_or(u32::MAX), p.id));

    let mut total_group_vram = my_vram;
    let mut worker_ids = Vec::new();
    for peer in candidates {
        if total_group_vram >= min_vram && !(force_split && worker_ids.is_empty()) {
            break;
        }
        total_group_vram += split_peer_vram_bytes(peer, my_vram);
        worker_ids.push(peer.id);
    }

    if total_group_vram >= min_vram && (!force_split || !worker_ids.is_empty()) {
        DenseLaunchPlan::Split {
            worker_ids,
            total_group_vram,
        }
    } else {
        DenseLaunchPlan::WaitingForCapacity {
            worker_ids,
            total_group_vram,
            min_vram,
        }
    }
}

fn rpc_ports_for_worker_ids(
    all_ports: &HashMap<iroh::EndpointId, u16>,
    worker_ids: &[iroh::EndpointId],
) -> Option<Vec<u16>> {
    worker_ids
        .iter()
        .map(|id| all_ports.get(id).copied())
        .collect()
}

/// The current state of llama-server as managed by the election loop.
/// The API proxy reads this to know where to forward requests.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum InferenceTarget {
    /// No llama-server running anywhere (election in progress, mesh empty, etc.)
    None,
    /// We are host — llama-server is on this local port.
    Local(u16),
    /// Another node is host — proxy via QUIC to this peer.
    Remote(iroh::EndpointId),
    /// MoE mode — this node runs its own llama-server with its expert shard.
    /// All MoE nodes are independent; the proxy picks one per session.
    MoeLocal(u16),
    /// MoE mode — another node is running its shard; proxy via QUIC.
    MoeRemote(iroh::EndpointId),
}

/// MoE deployment state shared between election and proxy.
/// The proxy uses this to route sessions to MoE nodes.
#[derive(Clone, Debug, Default)]
pub struct MoeState {
    /// All MoE node targets (local + remote), in stable order.
    pub nodes: Vec<InferenceTarget>,
    /// Full-coverage targets that can serve the whole model if the active shard set fails.
    pub fallbacks: Vec<InferenceTarget>,
}

/// Per-model routing table. The API proxy uses this to route by model name.
#[derive(Clone, Debug, Default)]
pub struct ModelTargets {
    /// model_name → list of inference targets (multiple hosts = load balancing)
    pub targets: HashMap<String, Vec<InferenceTarget>>,
    /// MoE state — if set, this model uses MoE expert sharding.
    /// The proxy uses this for session-sticky routing across MoE nodes.
    pub moe: Option<MoeState>,
    /// Round-robin counter for load balancing, shared across clones via Arc<AtomicU64>
    /// so that all ModelTargets clones (including per-request proxy clones) share a sequence.
    counter: Arc<AtomicU64>,
}

#[derive(Clone, Debug)]
pub struct LocalProcessInfo {
    pub backend: String,
    pub pid: u32,
    pub port: u16,
    pub context_length: u32,
}

fn stop_requested(stop_rx: &watch::Receiver<bool>) -> bool {
    *stop_rx.borrow()
}

async fn wait_for_peer_moe_ranking(
    model_name: &str,
    model_path: &Path,
    peer_rx: &mut watch::Receiver<usize>,
    stop_rx: &mut watch::Receiver<bool>,
    timeout: std::time::Duration,
) {
    if moe::best_shared_ranking_artifact(model_path).is_some() {
        return;
    }

    eprintln!(
        "🧩 [{model_name}] Waiting up to {:.0}s for peer MoE ranking before local analysis",
        timeout.as_secs_f64()
    );

    let deadline = tokio::time::Instant::now() + timeout;
    loop {
        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        if remaining.is_zero() {
            eprintln!("  No peer MoE ranking arrived in time — continuing with local analysis");
            return;
        }

        tokio::select! {
            _ = tokio::time::sleep(remaining) => {
                eprintln!("  No peer MoE ranking arrived in time — continuing with local analysis");
                return;
            }
            res = peer_rx.changed() => {
                if res.is_err() {
                    return;
                }
                if let Some(artifact) = moe::best_shared_ranking_artifact(model_path) {
                    eprintln!(
                        "  Using imported peer MoE ranking mode={} origin={}",
                        artifact.kind.label(),
                        artifact.origin.label()
                    );
                    return;
                }
            }
            res = stop_rx.changed() => {
                if res.is_err() || stop_requested(stop_rx) {
                    return;
                }
            }
        }
    }
}

impl ModelTargets {
    /// Get target for a specific model. Round-robins across multiple hosts.
    pub fn get(&self, model: &str) -> InferenceTarget {
        match self.targets.get(model) {
            Some(targets) if !targets.is_empty() => {
                let idx = self.counter.fetch_add(1, Ordering::Relaxed) as usize % targets.len();
                targets[idx].clone()
            }
            _ => InferenceTarget::None,
        }
    }

    /// All candidate targets for a model, preserving their current order.
    pub fn candidates(&self, model: &str) -> Vec<InferenceTarget> {
        self.targets.get(model).cloned().unwrap_or_default()
    }

    /// Round-robin pick from a caller-supplied candidate slice.
    pub fn pick_from(&self, candidates: &[InferenceTarget]) -> InferenceTarget {
        if candidates.is_empty() {
            InferenceTarget::None
        } else {
            let idx = self.counter.fetch_add(1, Ordering::Relaxed) as usize % candidates.len();
            candidates[idx].clone()
        }
    }

    /// Sticky pick from a caller-supplied candidate slice.
    pub fn pick_sticky_from(candidates: &[InferenceTarget], sticky_key: u64) -> InferenceTarget {
        if candidates.is_empty() {
            InferenceTarget::None
        } else {
            let idx = sticky_key as usize % candidates.len();
            candidates[idx].clone()
        }
    }

    /// Get MoE target for a session (hash-based routing).
    /// Returns None if not in MoE mode.
    pub fn get_moe_target(&self, session_hint: &str) -> Option<InferenceTarget> {
        let moe = self.moe.as_ref()?;
        if moe.nodes.is_empty() {
            return None;
        }
        // Simple hash routing: hash the session hint, pick a node
        let hash = session_hint
            .bytes()
            .fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));
        let idx = (hash as usize) % moe.nodes.len();
        Some(moe.nodes[idx].clone())
    }

    pub fn get_moe_failover_targets(&self, session_hint: &str) -> Vec<InferenceTarget> {
        let Some(primary) = self.get_moe_target(session_hint) else {
            return Vec::new();
        };
        let mut ordered = vec![primary.clone()];
        if let Some(moe) = self.moe.as_ref() {
            for fallback in &moe.fallbacks {
                if fallback != &primary {
                    ordered.push(fallback.clone());
                }
            }
        }
        ordered
    }
}

/// Compute shard index for a node given all node IDs in the MoE group.
/// Nodes are sorted by ID to ensure all nodes agree on the ordering.
/// Returns (sorted_ids, my_index).
#[cfg(test)]
pub fn moe_shard_index(
    my_id: iroh::EndpointId,
    peer_ids: &[iroh::EndpointId],
) -> (Vec<iroh::EndpointId>, usize) {
    let mut all_ids: Vec<iroh::EndpointId> = peer_ids.to_vec();
    if !all_ids.contains(&my_id) {
        all_ids.push(my_id);
    }
    all_ids.sort();
    let idx = all_ids.iter().position(|id| *id == my_id).unwrap_or(0);
    (all_ids, idx)
}

/// Build the MoE target map from sorted node IDs.
/// The caller's own node gets MoeLocal(port), others get MoeRemote(id).
pub fn build_moe_targets(
    sorted_ids: &[iroh::EndpointId],
    fallback_ids: &[iroh::EndpointId],
    my_id: iroh::EndpointId,
    active_local_port: Option<u16>,
    fallback_local_port: Option<u16>,
    model_name: &str,
) -> ModelTargets {
    let mut moe_state = MoeState::default();
    for &id in sorted_ids {
        if id == my_id {
            if let Some(port) = active_local_port {
                moe_state.nodes.push(InferenceTarget::MoeLocal(port));
            }
        } else {
            moe_state.nodes.push(InferenceTarget::MoeRemote(id));
        }
    }
    for &id in fallback_ids {
        if id == my_id {
            if let Some(port) = fallback_local_port {
                moe_state.fallbacks.push(InferenceTarget::Local(port));
            }
        } else {
            moe_state.fallbacks.push(InferenceTarget::Remote(id));
        }
    }
    let mut targets = ModelTargets::default();
    let primary_targets = if let Some(port) = active_local_port {
        vec![InferenceTarget::MoeLocal(port)]
    } else if let Some(port) = fallback_local_port {
        vec![InferenceTarget::Local(port)]
    } else {
        Vec::new()
    };
    if !primary_targets.is_empty() {
        targets
            .targets
            .insert(model_name.to_string(), primary_targets);
    }
    targets.moe = Some(moe_state);
    targets
}

#[derive(Clone, Debug)]
struct ResolvedMoeConfig {
    config: crate::models::catalog::MoeConfig,
    ranking_strategy: moe::MoeRankingStrategy,
    ranking_source: String,
    ranking_origin: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MoePlacementRole {
    SplitShard,
    FullFallback,
    Standby,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct MoePlacementPlan {
    leader_id: iroh::EndpointId,
    active_ids: Vec<iroh::EndpointId>,
    fallback_ids: Vec<iroh::EndpointId>,
    overlap: usize,
}

const MOE_SCALE_UP_QUIET_SECS: u64 = 45;

#[derive(Clone, Copy, Debug)]
struct MoePlacementCandidate {
    id: iroh::EndpointId,
    vram_bytes: u64,
    full_coverage: bool,
}

impl MoePlacementPlan {
    fn role_for(&self, my_id: iroh::EndpointId) -> MoePlacementRole {
        if self.active_ids.contains(&my_id) {
            MoePlacementRole::SplitShard
        } else if self.fallback_ids.contains(&my_id) {
            MoePlacementRole::FullFallback
        } else {
            MoePlacementRole::Standby
        }
    }

    fn shard_index_for(&self, my_id: iroh::EndpointId) -> Option<usize> {
        self.active_ids.iter().position(|id| *id == my_id)
    }

    fn materially_improves_upon(&self, current: &Self) -> bool {
        let improves_fallback = self.fallback_ids.len() > current.fallback_ids.len()
            && self.active_ids.len() >= current.active_ids.len();
        let improves_active_count = self.active_ids.len() > current.active_ids.len()
            && self.fallback_ids.len() >= current.fallback_ids.len();
        let improves_overlap = self.overlap > current.overlap
            && self.active_ids.len() >= current.active_ids.len()
            && self.fallback_ids.len() >= current.fallback_ids.len();

        improves_fallback || improves_active_count || improves_overlap
    }
}

fn running_plan_state(
    last_plan: Option<&MoePlacementPlan>,
    currently_running: bool,
) -> (&[iroh::EndpointId], &[iroh::EndpointId]) {
    if currently_running {
        let active_ids = last_plan
            .map(|plan| plan.active_ids.as_slice())
            .unwrap_or(&[]);
        let fallback_ids = last_plan
            .map(|plan| plan.fallback_ids.as_slice())
            .unwrap_or(&[]);
        (active_ids, fallback_ids)
    } else {
        (&[], &[])
    }
}

fn compute_best_moe_placement(
    mut candidates: Vec<MoePlacementCandidate>,
) -> Option<MoePlacementPlan> {
    if candidates.is_empty() {
        return None;
    }

    candidates.sort_by(|a, b| {
        b.vram_bytes
            .cmp(&a.vram_bytes)
            .then_with(|| a.id.cmp(&b.id))
    });
    let leader_id = candidates[0].id;
    let mut active_ids: Vec<iroh::EndpointId> =
        candidates.iter().map(|candidate| candidate.id).collect();
    active_ids.sort();
    active_ids.dedup();

    let mut fallback_ids = Vec::new();
    if active_ids.len() >= 3 {
        if let Some(fallback_candidate) =
            candidates.iter().find(|candidate| candidate.full_coverage)
        {
            active_ids.retain(|id| *id != fallback_candidate.id);
            fallback_ids.push(fallback_candidate.id);
        }
    }

    fallback_ids.sort();
    fallback_ids.dedup();

    let overlap = if active_ids.len() >= 3 { 2 } else { 1 };

    Some(MoePlacementPlan {
        leader_id,
        active_ids,
        fallback_ids,
        overlap,
    })
}

fn plan_moe_placement(
    candidates: Vec<MoePlacementCandidate>,
    current_active_ids: &[iroh::EndpointId],
    current_fallback_ids: &[iroh::EndpointId],
    allow_scale_up: bool,
) -> Option<MoePlacementPlan> {
    let candidate_ids: HashSet<_> = candidates.iter().map(|candidate| candidate.id).collect();
    let keep_current_active = !current_active_ids.is_empty()
        && current_active_ids
            .iter()
            .all(|id| candidate_ids.contains(id));

    let best = compute_best_moe_placement(candidates.clone())?;
    if !keep_current_active {
        return Some(best);
    }

    let mut stable = MoePlacementPlan {
        leader_id: best.leader_id,
        active_ids: current_active_ids.to_vec(),
        fallback_ids: current_fallback_ids
            .iter()
            .copied()
            .filter(|id| candidate_ids.contains(id) && !current_active_ids.contains(id))
            .collect(),
        overlap: if current_active_ids.len() >= 3 { 2 } else { 1 },
    };
    stable.active_ids.sort();
    stable.active_ids.dedup();
    stable.fallback_ids.sort();
    stable.fallback_ids.dedup();

    if allow_scale_up && best.materially_improves_upon(&stable) {
        Some(best)
    } else {
        Some(stable)
    }
}

/// Look up base MoE config for a model.
/// 1. Catalog provides MoE shape hints when available.
/// 2. GGUF header detection fills in the rest with conservative defaults.
fn lookup_moe_config(
    model_name: &str,
    model_path: &Path,
) -> Option<crate::models::catalog::MoeConfig> {
    // Tier 1: catalog lookup (shape hints only; runtime ranking is resolved later)
    let q = model_name.to_lowercase();
    if let Some(cfg) = crate::models::catalog::MODEL_CATALOG
        .iter()
        .find(|m| m.name.to_lowercase() == q || m.file.to_lowercase().contains(&q))
        .and_then(|m| m.moe.clone())
    {
        if !cfg.ranking.is_empty() {
            return Some(cfg);
        }
        // Catalog says MoE but no ranking — fall through to GGUF detect + sequential fallback
        // (keeps n_expert/n_expert_used/min_experts from catalog)
    }

    // Tier 2: auto-detect from GGUF header
    let info = models::gguf::detect_moe(model_path)?;
    eprintln!(
        "🔍 Auto-detected MoE from GGUF: {} experts, top-{}",
        info.expert_count, info.expert_used_count
    );

    // Conservative default: 50% shared core (safe floor for quality).
    // Without a ranking, we use sequential expert IDs (0..N).
    let min_experts = (info.expert_count as f64 * 0.5).ceil() as u32;

    // Check for cached ranking on disk
    let ranking_path = moe::ranking_cache_path(model_path);
    if let Some(ranking) = moe::load_cached_ranking(&ranking_path) {
        eprintln!("  Using cached ranking from {}", ranking_path.display());
        return Some(crate::models::catalog::MoeConfig {
            n_expert: info.expert_count,
            n_expert_used: info.expert_used_count,
            min_experts_per_node: min_experts,
            ranking,
        });
    }

    // No ranking available — use sequential (0, 1, 2, ...) as fallback.
    // The election loop can run moe-analyze to compute a proper ranking.
    let sequential: Vec<u32> = (0..info.expert_count).collect();
    Some(crate::models::catalog::MoeConfig {
        n_expert: info.expert_count,
        n_expert_used: info.expert_used_count,
        min_experts_per_node: min_experts,
        ranking: sequential,
    })
}

fn should_attempt_local_micro_analyze(
    model_path: &Path,
    model_name: &str,
    local_vram_budget: u64,
) -> bool {
    let model_bytes = total_model_bytes(model_path);
    // Require roughly the same headroom we already use for "fits locally" checks.
    let fits_with_headroom = local_vram_budget >= (model_bytes as f64 * 1.1) as u64;
    if !fits_with_headroom {
        eprintln!(
            "🧩 [{model_name}] Skipping local micro-analyze: model needs about {:.1}GB with headroom, local capacity is {:.1}GB",
            model_bytes as f64 * 1.1 / 1e9,
            local_vram_budget as f64 / 1e9
        );
    }
    fits_with_headroom
}

fn resolve_runtime_moe_config(
    model_name: &str,
    model_path: &Path,
    bin_dir: &Path,
    local_vram_budget: u64,
    options: &moe::MoeRuntimeOptions,
) -> anyhow::Result<Option<ResolvedMoeConfig>> {
    let base = match lookup_moe_config(model_name, model_path) {
        Some(cfg) => cfg,
        None => return Ok(None),
    };

    let started = std::time::Instant::now();
    let (ranking, ranking_source, ranking_origin) = match options.ranking_strategy {
        moe::MoeRankingStrategy::Auto => {
            let model_path_for_ranking = model_path.to_path_buf();
            let resolved_ranking_result: anyhow::Result<
                Option<crate::system::moe_planner::ResolvedRanking>,
            > = match tokio::runtime::Handle::try_current() {
                Ok(handle) => match tokio::task::block_in_place(|| {
                    handle.block_on(tokio::task::spawn_blocking(move || {
                        crate::system::moe_planner::resolve_runtime_ranking(
                            &model_path_for_ranking,
                            crate::system::moe_planner::DEFAULT_MOE_RANKINGS_DATASET,
                        )
                    }))
                }) {
                    Ok(Ok(resolved)) => Ok(resolved),
                    Ok(Err(err)) => {
                        eprintln!(
                            "⚠ [{model_name}] Failed to resolve shared MoE ranking ({err}); falling back to local analysis or sequential expert order"
                        );
                        Ok(None)
                    }
                    Err(err) => {
                        eprintln!(
                            "⚠ [{model_name}] Failed to join shared MoE ranking resolver ({err}); falling back to local analysis or sequential expert order"
                        );
                        Ok(None)
                    }
                },
                Err(_) => crate::system::moe_planner::resolve_runtime_ranking(
                    model_path,
                    crate::system::moe_planner::DEFAULT_MOE_RANKINGS_DATASET,
                ),
            };
            let resolved_ranking = match resolved_ranking_result {
                Ok(resolved) => resolved,
                Err(err) => {
                    eprintln!(
                        "⚠ [{model_name}] Failed to resolve shared MoE ranking ({err}); falling back to local analysis or sequential expert order"
                    );
                    None
                }
            };
            if let Some(resolved) = resolved_ranking {
                eprintln!(
                    "🧩 [{model_name}] Using {} MoE ranking mode={} path={}",
                    resolved.source.label(),
                    resolved.analyzer_id,
                    resolved.path.display()
                );
                (
                    moe::load_cached_ranking(&resolved.path).ok_or_else(|| {
                        anyhow::anyhow!(
                            "Failed to load resolved ranking {}",
                            resolved.path.display()
                        )
                    })?,
                    resolved.analyzer_id,
                    resolved.source.label().to_string(),
                )
            } else {
                if should_attempt_local_micro_analyze(model_path, model_name, local_vram_budget) {
                    match ensure_micro_analyze_ranking(bin_dir, model_name, model_path, options) {
                        Ok(artifact) => (
                            artifact.ranking,
                            "micro-v1".to_string(),
                            artifact.origin.label().to_string(),
                        ),
                        Err(err) => {
                            eprintln!(
                                "⚠ [{model_name}] micro-analyze failed ({err}); falling back to sequential expert order"
                            );
                            (
                                (0..base.n_expert).collect(),
                                "sequential-fallback".to_string(),
                                "fallback".to_string(),
                            )
                        }
                    }
                } else {
                    eprintln!(
                        "🧩 [{model_name}] Waiting for peer MoE ranking or using sequential fallback on this node"
                    );
                    (
                        (0..base.n_expert).collect(),
                        "sequential-fallback".to_string(),
                        "fallback".to_string(),
                    )
                }
            }
        }
        moe::MoeRankingStrategy::Analyze => {
            let cached = moe::ranking_cache_path(model_path);
            let artifact = ensure_full_analyze_ranking(bin_dir, model_name, model_path, &cached)?;
            (
                artifact.ranking,
                "full-v1".to_string(),
                artifact.origin.label().to_string(),
            )
        }
        moe::MoeRankingStrategy::MicroAnalyze => {
            let artifact = ensure_micro_analyze_ranking(bin_dir, model_name, model_path, options)?;
            (
                artifact.ranking,
                "micro-v1".to_string(),
                artifact.origin.label().to_string(),
            )
        }
    };

    eprintln!(
        "🧩 [{}] MoE ranking={} origin={} resolved in {:.1}s",
        model_name,
        ranking_source,
        ranking_origin,
        started.elapsed().as_secs_f64()
    );

    Ok(Some(ResolvedMoeConfig {
        config: crate::models::catalog::MoeConfig { ranking, ..base },
        ranking_strategy: options.ranking_strategy,
        ranking_source,
        ranking_origin,
    }))
}

fn refresh_auto_moe_config_from_cache(
    model_name: &str,
    model_path: &Path,
    cfg: &mut ResolvedMoeConfig,
) -> bool {
    if !matches!(cfg.ranking_strategy, moe::MoeRankingStrategy::Auto) {
        return false;
    }
    let Some(artifact) = moe::best_shared_ranking_artifact(model_path) else {
        return false;
    };
    let resolved = crate::system::moe_planner::ResolvedRanking {
        path: moe::shared_ranking_cache_path(model_path, &artifact),
        metadata_path: None,
        analyzer_id: match artifact.kind {
            moe::SharedRankingKind::Analyze => "full-v1",
            moe::SharedRankingKind::MicroAnalyze => "micro-v1",
        }
        .to_string(),
        source: crate::system::moe_planner::RankingSource::LocalCache,
        reason: "local ranking refresh".to_string(),
    };
    let Some(ranking) = moe::load_cached_ranking(&resolved.path) else {
        return false;
    };
    if cfg.config.ranking == ranking
        && cfg.ranking_source == resolved.analyzer_id
        && cfg.ranking_origin == resolved.source.label()
    {
        return false;
    }

    eprintln!(
        "🧩 [{model_name}] Switching to better {} MoE ranking mode={}",
        resolved.source.label(),
        resolved.analyzer_id
    );
    cfg.config.ranking = ranking;
    cfg.ranking_source = resolved.analyzer_id;
    cfg.ranking_origin = resolved.source.label().to_string();
    true
}

fn print_runtime_submit_suggestion(model_name: &str, model_path: &Path, ranking_path: &Path) {
    let Some(identity) = crate::models::huggingface_identity_for_path(model_path) else {
        return;
    };
    eprintln!("🧠 [{model_name}] Generated a local MoE ranking");
    eprintln!(
        "📍 [{model_name}] Ranking cache: {}",
        ranking_path.display()
    );
    eprintln!(
        "☁️  [{model_name}] Published source: {}",
        crate::system::moe_planner::DEFAULT_MOE_RANKINGS_DATASET
    );
    eprintln!("⚠️  [{model_name}] This run did not use a published ranking.");
    eprintln!(
        "📤 [{model_name}] Contribute it with: mesh-llm moe share '{}'",
        identity.canonical_ref
    );
}

fn resolve_analyze_binary(bin_dir: &Path) -> anyhow::Result<std::path::PathBuf> {
    let candidates = [
        bin_dir.join("llama-moe-analyze"),
        bin_dir.join("../llama.cpp/build/bin/llama-moe-analyze"),
        bin_dir.join("../../llama.cpp/build/bin/llama-moe-analyze"),
        bin_dir.join("../../../llama.cpp/build/bin/llama-moe-analyze"),
    ];
    for candidate in candidates {
        if candidate.exists() {
            return Ok(candidate.canonicalize().unwrap_or(candidate));
        }
    }
    anyhow::bail!(
        "llama-moe-analyze not found in {} or nearby llama.cpp/build/bin directories",
        bin_dir.display()
    )
}

fn should_suppress_moe_analyze_line(line: &str) -> bool {
    let trimmed = line.trim();
    trimmed.is_empty() || trimmed.starts_with("print_info:")
}

fn should_relay_moe_analyze_warning(line: &str) -> bool {
    let trimmed = line.trim();
    if should_suppress_moe_analyze_line(trimmed) {
        return false;
    }

    trimmed.starts_with("W ")
        || trimmed.starts_with("E ")
        || trimmed.to_ascii_lowercase().contains("failed")
        || trimmed.to_ascii_lowercase().contains("error")
}

#[derive(Default)]
struct MoeAnalyzeProgressState {
    current_prompt: usize,
    total_prompts: Option<usize>,
    done: bool,
}

fn format_moe_analysis_progress_line(
    model_name: &str,
    mode: &str,
    spinner: &str,
    current: usize,
    total: Option<usize>,
    elapsed: std::time::Duration,
) -> String {
    let mode_label = format!("MoE {mode}");
    let progress = match total {
        Some(total) if total > 0 => format!(
            "{:>5.1}%  {}/{}",
            (current as f64 / total as f64) * 100.0,
            current,
            total
        ),
        Some(total) => format!("       0/{}", total),
        None => "starting".to_string(),
    };
    let spinner_progress = format!("{spinner} {progress}");
    format!(
        "🧩 [{}] {:<17} {}  {:>3}s",
        model_name,
        mode_label,
        spinner_progress,
        elapsed.as_secs()
    )
}

struct MoeElectionParams {
    runtime: Arc<crate::runtime::instance::InstanceRuntime>,
    node: mesh::Node,
    tunnel_mgr: tunnel::Manager,
    ingress_http_port: u16,
    bin_dir: std::path::PathBuf,
    model: std::path::PathBuf,
    model_name: String,
    moe_cfg: ResolvedMoeConfig,
    my_vram: u64,
    model_bytes: u64,
    binary_flavor: Option<launch::BinaryFlavor>,
    ctx_size_override: Option<u32>,
    target_tx: Arc<watch::Sender<ModelTargets>>,
    stop_rx: watch::Receiver<bool>,
}

struct StartLlamaParams<'a> {
    runtime: &'a crate::runtime::instance::InstanceRuntime,
    node: &'a mesh::Node,
    tunnel_mgr: &'a tunnel::Manager,
    bin_dir: &'a Path,
    model: &'a Path,
    model_name: &'a str,
    model_peers: &'a [mesh::PeerInfo],
    explicit_mmproj: Option<&'a Path>,
    draft: Option<&'a Path>,
    draft_max: u16,
    force_split: bool,
    binary_flavor: Option<launch::BinaryFlavor>,
    ctx_size_override: Option<u32>,
}

pub struct ElectionLoopParams {
    pub runtime: Arc<crate::runtime::instance::InstanceRuntime>,
    pub node: mesh::Node,
    pub tunnel_mgr: tunnel::Manager,
    pub ingress_http_port: u16,
    pub rpc_port: u16,
    pub bin_dir: std::path::PathBuf,
    pub model: std::path::PathBuf,
    pub model_name: String,
    pub explicit_mmproj: Option<std::path::PathBuf>,
    pub draft: Option<std::path::PathBuf>,
    pub draft_max: u16,
    pub force_split: bool,
    pub binary_flavor: Option<launch::BinaryFlavor>,
    pub ctx_size_override: Option<u32>,
    pub moe_runtime_options: moe::MoeRuntimeOptions,
    pub target_tx: Arc<watch::Sender<ModelTargets>>,
    pub stop_rx: watch::Receiver<bool>,
}

fn spawn_moe_analysis_spinner(
    model_name: String,
    mode: &'static str,
    progress: Arc<Mutex<MoeAnalyzeProgressState>>,
    started: std::time::Instant,
) -> thread::JoinHandle<()> {
    const FRAMES: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
    thread::spawn(move || {
        let mut frame_idx = 0usize;
        loop {
            let (current, total, done) = progress
                .lock()
                .map(|state| (state.current_prompt, state.total_prompts, state.done))
                .unwrap_or((0, None, true));
            let spinner = if done {
                "✓"
            } else {
                FRAMES[frame_idx % FRAMES.len()]
            };
            let line = format_moe_analysis_progress_line(
                &model_name,
                mode,
                spinner,
                current,
                total,
                started.elapsed(),
            );
            eprint!("\r\x1b[2K{line}");
            let _ = std::io::stderr().flush();
            if done {
                eprintln!();
                break;
            }
            frame_idx += 1;
            thread::sleep(std::time::Duration::from_millis(125));
        }
    })
}

fn parse_moe_analyze_prompt_total(line: &str) -> Option<usize> {
    let trimmed = line.trim();
    let rest = trimmed.strip_prefix("Running ")?;
    let prompt_count = rest.split_whitespace().next()?;
    prompt_count.parse::<usize>().ok()
}

fn parse_moe_analyze_prompt_progress(line: &str) -> Option<(usize, usize)> {
    let trimmed = line.trim();
    let rest = trimmed.strip_prefix("Prompt ")?;
    let progress = rest.split(':').next()?.trim();
    let (current, total) = progress.split_once('/')?;
    Some((current.parse::<usize>().ok()?, total.parse::<usize>().ok()?))
}

fn spawn_moe_analyze_log_relay<R: std::io::Read + Send + 'static>(
    reader: R,
    model_name: String,
    progress: Arc<Mutex<MoeAnalyzeProgressState>>,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        let reader = BufReader::new(reader);
        for line in reader.lines().map_while(Result::ok) {
            if let Some(total) = parse_moe_analyze_prompt_total(&line) {
                if let Ok(mut state) = progress.lock() {
                    state.total_prompts = Some(total);
                }
                continue;
            }
            if let Some((current, total)) = parse_moe_analyze_prompt_progress(&line) {
                if let Ok(mut state) = progress.lock() {
                    state.total_prompts = Some(total);
                    state.current_prompt = current.saturating_sub(1);
                }
                continue;
            }
            if should_relay_moe_analyze_warning(&line) {
                eprint!("\r\x1b[2K");
                eprintln!("  [{model_name}] {line}");
            }
        }
    })
}

fn ensure_full_analyze_ranking(
    bin_dir: &Path,
    model_name: &str,
    model_path: &Path,
    cached_path: &Path,
) -> anyhow::Result<moe::SharedRankingArtifact> {
    if let Some(artifact) = moe::load_shared_ranking_artifact(
        cached_path,
        moe::SharedRankingKind::Analyze,
        moe::SharedRankingOrigin::LegacyCache,
        None,
        None,
        None,
    ) {
        eprintln!(
            "🧩 [{model_name}] Using cached MoE ranking mode=full-analyze origin={} cache={}",
            artifact.origin.label(),
            cached_path.display()
        );
        return Ok(artifact);
    }
    if let Some(parent) = cached_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let analyze_bin = resolve_analyze_binary(bin_dir)?;
    let started = std::time::Instant::now();
    let temp_output = std::env::temp_dir().join(format!(
        "mesh-llm-full-live-{}-{}.csv",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|duration| duration.as_nanos())
            .unwrap_or(0)
    ));
    eprintln!(
        "🧩 [{model_name}] MoE analysis mode=full-analyze cache={}",
        cached_path.display()
    );
    let progress = Arc::new(Mutex::new(MoeAnalyzeProgressState::default()));
    let spinner = spawn_moe_analysis_spinner(
        model_name.to_string(),
        "full-analyze",
        Arc::clone(&progress),
        started,
    );
    let mut child = Command::new(&analyze_bin)
        .args([
            "-m",
            &model_path.to_string_lossy(),
            "--all-layers",
            "--export-ranking",
            &temp_output.to_string_lossy(),
            "-n",
            "32",
            "-c",
            "4096",
            "-ngl",
            "99",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    let stdout_relay = child.stdout.take().map(|stdout| {
        spawn_moe_analyze_log_relay(stdout, model_name.to_string(), Arc::clone(&progress))
    });
    let stderr_relay = child.stderr.take().map(|stderr| {
        spawn_moe_analyze_log_relay(stderr, model_name.to_string(), Arc::clone(&progress))
    });
    let status = child.wait()?;
    if let Some(handle) = stdout_relay {
        let _ = handle.join();
    }
    if let Some(handle) = stderr_relay {
        let _ = handle.join();
    }
    if let Ok(mut state) = progress.lock() {
        if let Some(total) = state.total_prompts {
            state.current_prompt = total;
        }
        state.done = true;
    }
    let _ = spinner.join();
    anyhow::ensure!(status.success(), "llama-moe-analyze exited with {status}");
    let ranking = moe::load_cached_ranking(&temp_output).ok_or_else(|| {
        anyhow::anyhow!(
            "No ranking produced by full analyze at {}",
            temp_output.display()
        )
    })?;
    let artifact = moe::SharedRankingArtifact {
        kind: moe::SharedRankingKind::Analyze,
        origin: moe::SharedRankingOrigin::LocalFullAnalyze,
        ranking,
        micro_prompt_count: None,
        micro_tokens: None,
        micro_layer_scope: None,
    };
    let wrote_cache = moe::cache_shared_ranking_if_stronger(model_path, &artifact)?;
    std::fs::copy(&temp_output, cached_path)?;
    let _ = std::fs::remove_file(&temp_output);
    eprintln!(
        "  Full moe-analyze cached at {} in {:.1}s (origin={})",
        cached_path.display(),
        started.elapsed().as_secs_f64(),
        artifact.origin.label()
    );
    if !wrote_cache {
        eprintln!(
            "  A stronger or equivalent shared ranking already exists, so this full-v1 result was not promoted as the preferred shared artifact"
        );
    }
    print_runtime_submit_suggestion(model_name, model_path, cached_path);
    Ok(artifact)
}

fn ensure_micro_analyze_ranking(
    bin_dir: &Path,
    model_name: &str,
    model_path: &Path,
    options: &moe::MoeRuntimeOptions,
) -> anyhow::Result<moe::SharedRankingArtifact> {
    let cached_path = moe::micro_ranking_cache_path(
        model_path,
        options.micro_prompt_count,
        options.micro_tokens,
        options.micro_layer_scope,
    );
    if let Some(artifact) = moe::load_shared_ranking_artifact(
        &cached_path,
        moe::SharedRankingKind::MicroAnalyze,
        moe::SharedRankingOrigin::LegacyCache,
        Some(options.micro_prompt_count),
        Some(options.micro_tokens),
        Some(options.micro_layer_scope),
    ) {
        eprintln!(
            "🧩 [{model_name}] Using cached MoE ranking mode=micro-analyze origin={} cache={}",
            artifact.origin.label(),
            cached_path.display()
        );
        return Ok(artifact);
    }
    let analyze = run_micro_analyze_ranking(bin_dir, model_name, model_path, options)?;
    let artifact = moe::SharedRankingArtifact {
        kind: moe::SharedRankingKind::MicroAnalyze,
        origin: moe::SharedRankingOrigin::LocalMicroAnalyze,
        ranking: analyze.ranking,
        micro_prompt_count: Some(options.micro_prompt_count),
        micro_tokens: Some(options.micro_tokens),
        micro_layer_scope: Some(options.micro_layer_scope),
    };
    let wrote_cache = moe::cache_shared_ranking_if_stronger(model_path, &artifact)?;
    write_runtime_canonical_micro_ranking(
        &cached_path,
        &artifact,
        &analyze.rows,
        analyze.rows.iter().map(|(_, values)| values.0).sum::<f64>(),
    )?;
    eprintln!(
        "  Micro moe-analyze cached at {} (origin={})",
        cached_path.display(),
        artifact.origin.label()
    );
    if !wrote_cache {
        eprintln!(
            "  A stronger or equivalent shared ranking already exists, so this micro-v1 result was not promoted as the preferred shared artifact"
        );
    }
    print_runtime_submit_suggestion(model_name, model_path, &cached_path);
    Ok(artifact)
}

#[derive(Clone, Copy)]
struct AnalyzeMassRow {
    expert_id: u32,
    gate_mass: f64,
    selection_count: u64,
}

struct RuntimeMicroAnalyzeResult {
    ranking: Vec<u32>,
    rows: Vec<(u32, (f64, u64))>,
}

fn run_micro_analyze_ranking(
    bin_dir: &Path,
    model_name: &str,
    model_path: &Path,
    options: &moe::MoeRuntimeOptions,
) -> anyhow::Result<RuntimeMicroAnalyzeResult> {
    let prompts = default_micro_prompts();
    let prompt_count = options.micro_prompt_count.max(1).min(prompts.len());
    let analyze_bin = resolve_analyze_binary(bin_dir)?;
    let timestamp_nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    let tmp_dir = std::env::temp_dir().join(format!(
        "mesh-llm-micro-live-{}-{}",
        std::process::id(),
        timestamp_nanos
    ));
    std::fs::create_dir_all(&tmp_dir)?;
    let started = std::time::Instant::now();
    let mut mass_by_expert: HashMap<u32, (f64, u64)> = HashMap::new();
    eprintln!(
        "🧩 [{model_name}] MoE analysis mode=micro-analyze prompts={} tokens={} layers={} cache=pending",
        prompt_count,
        options.micro_tokens,
        match options.micro_layer_scope {
            moe::MoeMicroLayerScope::All => "all",
            moe::MoeMicroLayerScope::First => "first",
        }
    );
    let progress = Arc::new(Mutex::new(MoeAnalyzeProgressState {
        current_prompt: 0,
        total_prompts: Some(prompt_count),
        done: false,
    }));
    let spinner = spawn_moe_analysis_spinner(
        model_name.to_string(),
        "micro-analyze",
        Arc::clone(&progress),
        started,
    );

    for (idx, prompt) in prompts.iter().take(prompt_count).enumerate() {
        let output_path = tmp_dir.join(format!("prompt-{idx}.csv"));
        let mut command = Command::new(&analyze_bin);
        command.args([
            "-m",
            &model_path.to_string_lossy(),
            "--export-ranking",
            &output_path.to_string_lossy(),
            "-n",
            &options.micro_tokens.to_string(),
            "-c",
            "4096",
            "-ngl",
            "99",
            "-p",
            prompt,
        ]);
        if matches!(options.micro_layer_scope, moe::MoeMicroLayerScope::All) {
            command.arg("--all-layers");
        }
        let output = command.output()?;
        if !output.status.success() {
            if let Ok(mut state) = progress.lock() {
                state.done = true;
            }
            let _ = spinner.join();
            let stderr = String::from_utf8_lossy(&output.stderr);
            let stdout = String::from_utf8_lossy(&output.stdout);
            let mut details = stderr
                .lines()
                .chain(stdout.lines())
                .filter(|line| !should_suppress_moe_analyze_line(line))
                .collect::<Vec<_>>();
            if details.len() > 20 {
                details.truncate(20);
            }
            let detail_text = if details.is_empty() {
                String::new()
            } else {
                format!(": {}", details.join(" | "))
            };
            anyhow::bail!(
                "llama-moe-analyze exited with {}{}",
                output.status,
                detail_text
            );
        }
        for row in load_analyze_mass_rows(&output_path)? {
            let entry = mass_by_expert.entry(row.expert_id).or_insert((0.0, 0));
            entry.0 += row.gate_mass;
            entry.1 += row.selection_count;
        }
        if let Ok(mut state) = progress.lock() {
            state.current_prompt = idx + 1;
        }
    }
    if let Ok(mut state) = progress.lock() {
        state.current_prompt = prompt_count;
        state.done = true;
    }
    let _ = spinner.join();

    let mut rows = mass_by_expert.into_iter().collect::<Vec<_>>();
    rows.sort_by(|a, b| {
        b.1 .0
            .partial_cmp(&a.1 .0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    let ranking = rows.iter().map(|(expert_id, _)| *expert_id).collect();
    let _ = std::fs::remove_dir_all(&tmp_dir);
    eprintln!(
        "  Micro moe-analyze used {} prompt(s), {} token(s), {} in {:.1}s",
        prompt_count,
        options.micro_tokens,
        match options.micro_layer_scope {
            moe::MoeMicroLayerScope::All => "all layers",
            moe::MoeMicroLayerScope::First => "first layer",
        },
        started.elapsed().as_secs_f64()
    );
    Ok(RuntimeMicroAnalyzeResult { ranking, rows })
}

fn load_analyze_mass_rows(path: &Path) -> anyhow::Result<Vec<AnalyzeMassRow>> {
    let content = std::fs::read_to_string(path)?;
    let mut rows = Vec::new();
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') || trimmed.starts_with("expert") {
            continue;
        }
        let parts = trimmed.split(',').map(str::trim).collect::<Vec<_>>();
        if parts.len() < 2 {
            continue;
        }
        rows.push(AnalyzeMassRow {
            expert_id: parts[0].parse()?,
            gate_mass: parts[1].parse()?,
            selection_count: parts[3].parse()?,
        });
    }
    Ok(rows)
}

fn write_runtime_canonical_micro_ranking(
    path: &Path,
    artifact: &moe::SharedRankingArtifact,
    ranking: &[(u32, (f64, u64))],
    total_mass_sum: f64,
) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut output = String::new();
    writeln!(&mut output, "# mesh-llm-moe-ranking=v1").ok();
    writeln!(&mut output, "# ranking_kind={}", artifact.kind.label()).ok();
    writeln!(&mut output, "# ranking_origin={}", artifact.origin.label()).ok();
    if let Some(prompt_count) = artifact.micro_prompt_count {
        writeln!(&mut output, "# micro_prompt_count={prompt_count}").ok();
    }
    if let Some(tokens) = artifact.micro_tokens {
        writeln!(&mut output, "# micro_tokens={tokens}").ok();
    }
    if let Some(layer_scope) = artifact.micro_layer_scope {
        let scope = match layer_scope {
            moe::MoeMicroLayerScope::All => "all",
            moe::MoeMicroLayerScope::First => "first",
        };
        writeln!(&mut output, "# micro_layer_scope={scope}").ok();
    }
    writeln!(
        &mut output,
        "expert_id,total_mass,mass_fraction,selection_count"
    )
    .ok();
    for (expert_id, (gate_mass, selection_count)) in ranking {
        let mass_fraction = if total_mass_sum > 0.0 {
            gate_mass / total_mass_sum
        } else {
            0.0
        };
        writeln!(
            &mut output,
            "{expert_id},{gate_mass:.12},{mass_fraction:.12},{selection_count}"
        )
        .ok();
    }
    std::fs::write(path, output)?;
    Ok(())
}

fn default_micro_prompts() -> &'static [&'static str] {
    &[
        "User: Explain how mixture-of-experts routing works in a language model.\nAssistant:",
        "User: Write a short professional email asking for feedback on a technical design.\nAssistant:",
        "User: Outline a debugging plan for a flaky distributed systems test.\nAssistant:",
        "User: Summarize the tradeoffs between latency and quality in MoE inference.\nAssistant:",
    ]
}

/// Background election loop for a single model.
/// This node serves `model` — it only cares about peers also serving `model`.
///
/// On every mesh change:
/// 1. Kill llama-server (if we're running it)
/// 2. Re-elect within the model group
/// 3. Winner starts llama-server with --rpc pointing at group nodes
///
/// Publishes the current ModelTargets via the watch channel so the
/// API proxy knows where to forward requests.
#[allow(clippy::too_many_arguments)]
pub async fn election_loop(
    params: ElectionLoopParams,
    mut on_change: impl FnMut(bool, bool) + Send,
    mut on_process: impl FnMut(Option<LocalProcessInfo>) + Send,
) {
    let ElectionLoopParams {
        runtime,
        node,
        tunnel_mgr,
        ingress_http_port,
        rpc_port: _rpc_port,
        bin_dir,
        model,
        model_name,
        explicit_mmproj,
        draft,
        draft_max,
        force_split,
        binary_flavor,
        ctx_size_override,
        moe_runtime_options,
        target_tx,
        mut stop_rx,
    } = params;
    let mut peer_rx = node.peer_change_rx.clone();

    // Track the actual running launch topology so we only restart on real split changes.
    let mut last_running_plan: Option<DenseRunningPlan> = None;
    let mut currently_host = false;
    let mut current_local_port: Option<u16> = None;
    let mut llama_process: Option<launch::InferenceServerProcess> = None;

    // Initial settle
    tokio::time::sleep(std::time::Duration::from_secs(2)).await;

    let model_bytes = total_model_bytes(&model);
    let my_vram = node.vram_bytes();
    let model_fits_locally = my_vram >= (model_bytes as f64 * 1.1) as u64;

    // Check if this is a MoE model with enough metadata to plan expert routing.
    let moe_config = lookup_moe_config(&model_name, &model);
    if let Some(moe_config) = &moe_config {
        eprintln!(
            "🧩 [{}] MoE model detected ({} experts, top-{})",
            model_name, moe_config.n_expert, moe_config.n_expert_used
        );
    }

    // MoE mode: each node runs its own llama-server with its expert shard.
    // Only enter MoE split mode if the model doesn't fit locally or --split is forced.
    // Otherwise, just run the full model — every node is independent.
    if moe_config.is_some() {
        let need_moe_split = force_split || !model_fits_locally;
        if need_moe_split {
            if matches!(
                moe_runtime_options.ranking_strategy,
                moe::MoeRankingStrategy::Auto
            ) && moe::best_shared_ranking_artifact(&model).is_none()
            {
                wait_for_peer_moe_ranking(
                    &model_name,
                    &model,
                    &mut peer_rx,
                    &mut stop_rx,
                    std::time::Duration::from_secs(8),
                )
                .await;
            }
            let resolved_moe_cfg = match resolve_runtime_moe_config(
                &model_name,
                &model,
                &bin_dir,
                my_vram,
                &moe_runtime_options,
            ) {
                Ok(Some(cfg)) => cfg,
                Ok(None) => {
                    eprintln!("⚠️  [{}] Failed to resolve MoE split config", model_name);
                    return;
                }
                Err(e) => {
                    eprintln!(
                        "⚠️  [{}] Failed to resolve MoE ranking/grouping: {e}",
                        model_name
                    );
                    return;
                }
            };
            moe_election_loop(
                MoeElectionParams {
                    runtime: runtime.clone(),
                    node,
                    tunnel_mgr,
                    ingress_http_port,
                    bin_dir,
                    model,
                    model_name,
                    moe_cfg: resolved_moe_cfg,
                    my_vram,
                    model_bytes,
                    binary_flavor,
                    ctx_size_override,
                    target_tx,
                    stop_rx,
                },
                &mut on_change,
                &mut on_process,
            )
            .await;
            return;
        } else {
            eprintln!(
                "🧩 [{}] MoE model fits locally ({:.1}GB capacity for {:.1}GB model) — no split needed",
                model_name,
                my_vram as f64 / 1e9,
                model_bytes as f64 / 1e9
            );
            // Fall through to normal election loop — each node runs full model independently
        }
    }

    loop {
        if stop_requested(&stop_rx) {
            break;
        }
        // Collect our model group (peers also serving this model)
        let peers = node.peers().await;
        let model_peers: Vec<mesh::PeerInfo> = peers
            .iter()
            .filter(|p| p.is_assigned_model(&model_name))
            .cloned()
            .collect();
        let desired_launch =
            build_dense_launch_plan(my_vram, model_bytes, force_split, &model_name, &model_peers);

        // Splitting decision: only split when forced OR when the model
        // genuinely doesn't fit on this node alone. If it fits, every
        // node serving this model runs its own independent llama-server
        // (no election needed — everyone is a host).
        let requires_split = force_split || !model_fits_locally;

        let i_am_host = if requires_split {
            // Distributed mode: elect one host from the model group
            should_be_host_for_model(node.id(), my_vram, &model_peers)
        } else if model_peers.is_empty() {
            // No other node serving this model — we must host
            true
        } else if currently_host {
            // Already running — don't tear down
            true
        } else {
            // Another node is already serving this model.
            // Only spin up a duplicate if there's enough demand:
            //   - 2+ clients connected, OR
            //   - 10+ requests in the demand tracker for this model
            let n_clients = peers
                .iter()
                .filter(|p| matches!(p.role, mesh::NodeRole::Client))
                .count();
            let demand = node.get_demand();
            let req_count = demand
                .get(&model_name)
                .map(|d| d.request_count)
                .unwrap_or(0);
            let force_duplicate_host = std::env::var("MESH_LLM_FORCE_DUPLICATE_HOSTS")
                .ok()
                .as_deref()
                == Some("1");
            let should_dup = force_duplicate_host || n_clients >= 2 || req_count >= 10;
            if !should_dup {
                eprintln!(
                    "💤 [{}] Peer already serving — standby (clients: {}, requests: {})",
                    model_name, n_clients, req_count
                );
            } else if force_duplicate_host {
                eprintln!(
                    "🧪 [{}] Forcing duplicate host for benchmark topology",
                    model_name
                );
            }
            should_dup
        };

        // If we're already host and nothing changed, skip restart
        if currently_host && i_am_host && desired_launch.running_plan() == last_running_plan {
            // Just update the target map (in case other models' hosts changed)
            if let Some(local_port) = current_local_port {
                update_targets(
                    &node,
                    &model_name,
                    InferenceTarget::Local(local_port),
                    &target_tx,
                )
                .await;
            }
            // Wait for next change OR llama-server death
            tokio::select! {
                res = peer_rx.changed() => {
                    if res.is_err() { break; }
                    eprintln!("⚡ Mesh changed — re-checking... (still host, no restart needed)");
                    tokio::time::sleep(std::time::Duration::from_secs(3)).await;
                    continue;
                }
                _ = async {
                    if let Some(ref mut process) = llama_process {
                        let _ = (&mut process.death_rx).await;
                    } else {
                        std::future::pending::<()>().await;
                    }
                } => {
                    eprintln!("🔄 [{}] llama-server died — restarting...", model_name);
                    llama_process = None;
                    currently_host = false;
                    current_local_port = None;
                    last_running_plan = None;
                    update_targets(&node, &model_name, InferenceTarget::None, &target_tx).await;
                    on_process(None);
                    on_change(false, false);
                    tokio::time::sleep(std::time::Duration::from_secs(2)).await;
                    // Fall through to restart
                }
                res = stop_rx.changed() => {
                    if res.is_err() || stop_requested(&stop_rx) {
                        break;
                    }
                }
            }
        }

        // Something changed — kill llama-server if we were running it
        if currently_host {
            if let Some(process) = llama_process.take() {
                process.handle.shutdown().await;
            }
            tunnel_mgr.set_http_port(0);
            node.set_role(NodeRole::Worker).await;
            current_local_port = None;
            last_running_plan = None;
            update_targets(&node, &model_name, InferenceTarget::None, &target_tx).await;
            on_process(None);
            on_change(false, false);
            currently_host = false;
        }

        if stop_requested(&stop_rx) {
            break;
        }

        if i_am_host {
            match &desired_launch {
                DenseLaunchPlan::WaitingForCapacity {
                    total_group_vram,
                    min_vram,
                    ..
                } => {
                    eprintln!(
                        "⏳ [{}] Waiting for more peers — need {:.1}GB capacity, have {:.1}GB across eligible split workers",
                        model_name,
                        *min_vram as f64 / 1e9,
                        *total_group_vram as f64 / 1e9
                    );
                    update_targets(&node, &model_name, InferenceTarget::None, &target_tx).await;
                    on_change(false, false);
                    if peer_rx.changed().await.is_err() {
                        break;
                    }
                    tokio::time::sleep(std::time::Duration::from_secs(3)).await;
                    continue;
                }
                DenseLaunchPlan::Split {
                    total_group_vram,
                    worker_ids,
                } => {
                    eprintln!(
                        "🗳 [{}] Elected as host ({:.1}GB capacity for {:.1}GB model, {} node(s), split)",
                        model_name,
                        *total_group_vram as f64 / 1e9,
                        model_bytes as f64 / 1e9,
                        worker_ids.len() + 1
                    );
                }
                DenseLaunchPlan::Solo => {
                    eprintln!(
                        "🗳 [{}] Running as host ({:.1}GB capacity for {:.1}GB model, serving entirely)",
                        model_name,
                        my_vram as f64 / 1e9,
                        model_bytes as f64 / 1e9
                    );
                }
            }
            on_change(true, false);

            // In solo mode, pass empty model_peers so start_llama won't use any workers
            let peers_for_launch = if matches!(desired_launch, DenseLaunchPlan::Split { .. }) {
                &model_peers[..]
            } else {
                &[]
            };
            let (llama_port, process) = match start_llama(StartLlamaParams {
                runtime: &runtime,
                node: &node,
                tunnel_mgr: &tunnel_mgr,
                bin_dir: &bin_dir,
                model: &model,
                model_name: &model_name,
                model_peers: peers_for_launch,
                explicit_mmproj: explicit_mmproj.as_deref(),
                draft: draft.as_deref(),
                draft_max,
                force_split,
                binary_flavor,
                ctx_size_override,
            })
            .await
            {
                Some((port, death_rx)) => (port, death_rx),
                None => {
                    on_change(true, false);
                    let _ = peer_rx.changed().await;
                    tokio::time::sleep(std::time::Duration::from_secs(3)).await;
                    continue;
                }
            };

            node.set_role(NodeRole::Host {
                http_port: ingress_http_port,
            })
            .await;
            // Point tunnel directly at llama-server, bypassing the API proxy.
            // The client proxy has already normalized the request; the host
            // doesn't need to re-parse or re-route it.
            tunnel_mgr.set_http_port(llama_port);
            currently_host = true;
            current_local_port = Some(llama_port);
            last_running_plan = desired_launch.running_plan();
            // Re-gossip so peers learn we're the host for this model
            node.regossip().await;
            update_targets(
                &node,
                &model_name,
                InferenceTarget::Local(llama_port),
                &target_tx,
            )
            .await;
            llama_process = Some(process);
            if let Some(ref process) = llama_process {
                on_process(Some(LocalProcessInfo {
                    backend: "llama".into(),
                    pid: process.handle.pid(),
                    port: llama_port,
                    context_length: process.context_length,
                }));
            }
            on_change(true, true);
            eprintln!(
                "✅ [{}] llama-server ready on internal port {llama_port}",
                model_name
            );
        } else {
            // We're a worker in split mode. Find who the host is.
            node.set_role(NodeRole::Worker).await;
            currently_host = false;
            last_running_plan = None;

            let host_peer = model_peers
                .iter()
                .filter(|p| !matches!(p.role, NodeRole::Client))
                .max_by_key(|p| (p.vram_bytes, p.id));

            if let Some(host) = host_peer {
                if should_be_host_for_model(host.id, host.vram_bytes, &model_peers) {
                    update_targets(
                        &node,
                        &model_name,
                        InferenceTarget::Remote(host.id),
                        &target_tx,
                    )
                    .await;
                    eprintln!(
                        "📡 [{}] Worker — host is {} (split mode)",
                        model_name,
                        host.id.fmt_short()
                    );
                } else {
                    update_targets(&node, &model_name, InferenceTarget::None, &target_tx).await;
                }
            } else {
                update_targets(&node, &model_name, InferenceTarget::None, &target_tx).await;
            }
            on_change(false, false);
        }

        // Wait for next peer change OR llama-server death
        tokio::select! {
            res = peer_rx.changed() => {
                if res.is_err() { break; }
                eprintln!("⚡ Mesh changed — re-electing...");
            }
            _ = async {
                if let Some(ref mut process) = llama_process {
                    let _ = (&mut process.death_rx).await;
                } else {
                    std::future::pending::<()>().await;
                }
            } => {
                eprintln!("🔄 [{}] llama-server died — restarting...", model_name);
                llama_process = None;
                currently_host = false;
                current_local_port = None;
                last_running_plan = None;
                update_targets(&node, &model_name, InferenceTarget::None, &target_tx).await;
                on_change(false, false);
            }
            res = stop_rx.changed() => {
                if res.is_err() || stop_requested(&stop_rx) {
                    break;
                }
            }
        }
        if stop_requested(&stop_rx) {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_secs(3)).await;
    }

    if currently_host {
        if let Some(process) = llama_process.take() {
            process.handle.shutdown().await;
        }
        tunnel_mgr.set_http_port(0);
        node.set_role(NodeRole::Worker).await;
        update_targets(&node, &model_name, InferenceTarget::None, &target_tx).await;
        on_process(None);
        on_change(false, false);
    }
}

/// MoE election loop: every node runs its own llama-server with its expert shard.
///
/// Unlike tensor-split mode (one host + RPC workers), MoE mode means:
/// - Every node is independent — no host/worker distinction for this model
/// - Each node runs moe-split locally to produce its shard (cached)
/// - Each node starts its own llama-server with its shard GGUF
/// - The proxy routes sessions to nodes via hash-based affinity
#[allow(clippy::too_many_arguments)]
async fn moe_election_loop(
    params: MoeElectionParams,
    on_change: &mut impl FnMut(bool, bool),
    on_process: &mut impl FnMut(Option<LocalProcessInfo>),
) {
    let MoeElectionParams {
        runtime,
        node,
        tunnel_mgr,
        ingress_http_port,
        bin_dir,
        model,
        model_name,
        mut moe_cfg,
        my_vram,
        model_bytes,
        binary_flavor,
        ctx_size_override,
        target_tx,
        mut stop_rx,
    } = params;
    let mut peer_rx = node.peer_change_rx.clone();
    let mut currently_running = false;
    let mut last_plan: Option<MoePlacementPlan> = None;
    let mut llama_process: Option<launch::InferenceServerProcess> = None;
    let mut current_local_port: Option<u16> = None;
    let mut last_plan_change_at = tokio::time::Instant::now();

    loop {
        if stop_requested(&stop_rx) {
            break;
        }

        if !currently_running {
            let _ = refresh_auto_moe_config_from_cache(&model_name, &model, &mut moe_cfg);
        }

        let peers = node.peers().await;
        let local_descriptors = node.served_model_descriptors().await;
        let declared_model_peers: Vec<mesh::PeerInfo> = peers
            .iter()
            .filter(|p| !matches!(p.role, NodeRole::Client))
            .filter(|peer| {
                peer.is_assigned_model(&model_name)
                    || peer
                        .requested_models
                        .iter()
                        .any(|requested| requested == &model_name)
                    || peer.models.iter().any(|model| model == &model_name)
            })
            .cloned()
            .collect();
        let eligible_model_peers: Vec<mesh::PeerInfo> = declared_model_peers
            .iter()
            .filter_map(|peer| {
                mesh::peer_is_eligible_for_active_moe(&local_descriptors, peer, &model_name)
                    .then_some(peer.clone())
            })
            .collect();
        let model_fits = my_vram >= (model_bytes as f64 * 1.1) as u64;
        let placement_peers: Vec<mesh::PeerInfo> = if !currently_running
            && !model_fits
            && eligible_model_peers.is_empty()
        {
            if !declared_model_peers.is_empty() {
                eprintln!(
                        "🧩 [{model_name}] Bootstrapping MoE placement with {} declared peer(s) while active eligibility catches up",
                        declared_model_peers.len()
                    );
            }
            declared_model_peers.clone()
        } else {
            eligible_model_peers.clone()
        };
        let recovering_peer_count = peers
            .iter()
            .filter(|p| p.is_assigned_model(&model_name))
            .filter(|p| !matches!(p.role, NodeRole::Client))
            .filter(|peer| !peer.moe_recovery_ready())
            .count();
        if recovering_peer_count > 0 {
            eprintln!(
                "🧩 [{model_name}] Holding {} recovered peer(s) out of active MoE placement until stable",
                recovering_peer_count
            );
        }

        let my_id = node.id();
        let mut candidates = vec![MoePlacementCandidate {
            id: my_id,
            vram_bytes: my_vram,
            full_coverage: model_fits,
        }];
        candidates.extend(placement_peers.iter().map(|peer| MoePlacementCandidate {
            id: peer.id,
            vram_bytes: peer.vram_bytes,
            full_coverage: peer.vram_bytes >= (model_bytes as f64 * 1.1) as u64,
        }));
        let (current_active_ids, current_fallback_ids) =
            running_plan_state(last_plan.as_ref(), currently_running);
        let provisional_best = compute_best_moe_placement(candidates.clone());
        let allow_scale_up = currently_running
            && last_plan_change_at.elapsed()
                >= std::time::Duration::from_secs(MOE_SCALE_UP_QUIET_SECS);
        let Some(plan) = plan_moe_placement(
            candidates,
            current_active_ids,
            current_fallback_ids,
            allow_scale_up,
        ) else {
            tokio::select! {
                res = peer_rx.changed() => {
                    if res.is_err() { break; }
                }
                res = stop_rx.changed() => {
                    if res.is_err() || stop_requested(&stop_rx) {
                        break;
                    }
                }
            }
            continue;
        };
        let role = plan.role_for(my_id);
        let healthy_reserve_count = placement_peers
            .iter()
            .filter(|peer| {
                !plan.active_ids.contains(&peer.id) && !plan.fallback_ids.contains(&peer.id)
            })
            .count();
        if healthy_reserve_count > 0 && currently_running {
            if !allow_scale_up {
                let remaining = std::time::Duration::from_secs(MOE_SCALE_UP_QUIET_SECS)
                    .saturating_sub(last_plan_change_at.elapsed())
                    .as_secs();
                eprintln!(
                    "🧩 [{model_name}] Keeping {} healthy peer(s) in reserve for {}s before considering MoE scale-up",
                    healthy_reserve_count,
                    remaining
                );
            } else if provisional_best
                .as_ref()
                .filter(|best| {
                    last_plan
                        .as_ref()
                        .is_some_and(|current| best.materially_improves_upon(current))
                })
                .is_none()
            {
                eprintln!(
                    "🧩 [{model_name}] Keeping {} healthy peer(s) in reserve; the current MoE plan is still preferred",
                    healthy_reserve_count
                );
            }
        }

        if currently_running && last_plan.as_ref() == Some(&plan) {
            tokio::select! {
                res = peer_rx.changed() => {
                    if res.is_err() { break; }
                }
                res = stop_rx.changed() => {
                    if res.is_err() || stop_requested(&stop_rx) {
                        break;
                    }
                }
            }
            if stop_requested(&stop_rx) {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_secs(3)).await;
            continue;
        }

        if currently_running {
            if let Some(previous_plan) = last_plan.as_ref() {
                let previous_role = previous_plan.role_for(my_id);
                let same_local_deployment = previous_role == role
                    && previous_plan.active_ids == plan.active_ids
                    && previous_plan.overlap == plan.overlap;
                if same_local_deployment && previous_plan.fallback_ids != plan.fallback_ids {
                    let targets = build_moe_targets(
                        &plan.active_ids,
                        &plan.fallback_ids,
                        my_id,
                        matches!(role, MoePlacementRole::SplitShard).then_some(
                            current_local_port.expect("running MoE shard should have a local port"),
                        ),
                        matches!(role, MoePlacementRole::FullFallback).then_some(
                            current_local_port
                                .expect("running MoE fallback should have a local port"),
                        ),
                        &model_name,
                    );
                    target_tx.send_replace(targets);
                    last_plan = Some(plan);
                    last_plan_change_at = tokio::time::Instant::now();
                    continue;
                }
            }
        }

        // Something changed — kill existing llama-server
        if currently_running {
            if let Some(process) = llama_process.take() {
                process.handle.shutdown().await;
            }
            tunnel_mgr.set_http_port(0);
            currently_running = false;
            current_local_port = None;
            on_process(None);
            on_change(false, false);
        }

        last_plan = Some(plan.clone());
        last_plan_change_at = tokio::time::Instant::now();

        if matches!(role, MoePlacementRole::Standby) {
            node.set_model_runtime_context_length(&model_name, None)
                .await;
            node.regossip().await;
            eprintln!(
                "🧩 [{}] Standing by outside active MoE placement (leader={} active={} fallback={})",
                model_name,
                plan.leader_id.fmt_short(),
                plan.active_ids.len(),
                plan.fallback_ids.len()
            );
            node.set_role(NodeRole::Worker).await;
            update_targets(&node, &model_name, InferenceTarget::None, &target_tx).await;
            on_change(false, false);
        } else if matches!(role, MoePlacementRole::FullFallback) {
            eprintln!(
                "🧩 [{}] MoE full-coverage fallback — leader={} active-shards={} fallback-nodes={}",
                model_name,
                plan.leader_id.fmt_short(),
                plan.active_ids.len(),
                plan.fallback_ids.len()
            );
            on_change(true, false);

            let llama_port = match find_free_port().await {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("  Failed to find free port: {e}");
                    if peer_rx.changed().await.is_err() {
                        break;
                    }
                    tokio::time::sleep(std::time::Duration::from_secs(3)).await;
                    continue;
                }
            };

            match launch::start_llama_server(
                &runtime,
                &bin_dir,
                binary_flavor,
                launch::ModelLaunchSpec {
                    model: &model,
                    http_port: llama_port,
                    tunnel_ports: &[],
                    tensor_split: None,
                    split_mode: None,
                    draft: None,
                    draft_max: 0,
                    model_bytes,
                    my_vram,
                    mmproj: None,
                    ctx_size_override,
                    total_group_vram: None,
                },
            )
            .await
            {
                Ok(process) => {
                    node.set_role(NodeRole::Host {
                        http_port: ingress_http_port,
                    })
                    .await;
                    tunnel_mgr.set_http_port(llama_port);
                    currently_running = true;
                    current_local_port = Some(llama_port);
                    llama_process = Some(process);
                    if let Some(ref process) = llama_process {
                        on_process(Some(LocalProcessInfo {
                            backend: "llama".into(),
                            pid: process.handle.pid(),
                            port: llama_port,
                            context_length: process.context_length,
                        }));
                    }
                    node.regossip().await;
                    let targets = build_moe_targets(
                        &plan.active_ids,
                        &plan.fallback_ids,
                        my_id,
                        None,
                        Some(llama_port),
                        &model_name,
                    );
                    target_tx.send_replace(targets);
                    on_change(true, true);
                    eprintln!(
                        "✅ [{}] MoE fallback replica ready on port {llama_port}",
                        model_name
                    );
                }
                Err(e) => {
                    eprintln!("  Failed to start fallback llama-server: {e}");
                }
            }
        } else if plan.active_ids.len() == 1 {
            if model_fits {
                node.set_model_runtime_context_length(&model_name, None)
                    .await;
                node.regossip().await;
                eprintln!(
                    "🧩 [{}] MoE model — serving entirely ({:.1}GB fits in {:.1}GB capacity)",
                    model_name,
                    model_bytes as f64 / 1e9,
                    my_vram as f64 / 1e9
                );
                on_change(true, false);

                let llama_port = match find_free_port().await {
                    Ok(p) => p,
                    Err(e) => {
                        eprintln!("  Failed to find free port: {e}");
                        if peer_rx.changed().await.is_err() {
                            break;
                        }
                        tokio::time::sleep(std::time::Duration::from_secs(3)).await;
                        continue;
                    }
                };

                let mb = total_model_bytes(&model);
                match launch::start_llama_server(
                    &runtime,
                    &bin_dir,
                    binary_flavor,
                    launch::ModelLaunchSpec {
                        model: &model,
                        http_port: llama_port,
                        tunnel_ports: &[],
                        tensor_split: None,
                        split_mode: local_multi_gpu_split_mode(binary_flavor),
                        draft: None,
                        draft_max: 0,
                        model_bytes: mb,
                        my_vram,
                        mmproj: None,
                        ctx_size_override,
                        total_group_vram: None,
                    },
                )
                .await
                {
                    Ok(process) => {
                        node.set_role(NodeRole::Host {
                            http_port: ingress_http_port,
                        })
                        .await;
                        tunnel_mgr.set_http_port(llama_port);
                        currently_running = true;
                        current_local_port = Some(llama_port);
                        llama_process = Some(process);
                        if let Some(ref process) = llama_process {
                            on_process(Some(LocalProcessInfo {
                                backend: "llama".into(),
                                pid: process.handle.pid(),
                                port: llama_port,
                                context_length: process.context_length,
                            }));
                        }
                        update_targets(
                            &node,
                            &model_name,
                            InferenceTarget::Local(llama_port),
                            &target_tx,
                        )
                        .await;
                        on_change(true, true);
                        eprintln!(
                            "✅ [{}] MoE — llama-server ready on port {llama_port}",
                            model_name
                        );
                    }
                    Err(e) => {
                        eprintln!("  Failed to start llama-server: {e}");
                    }
                }
            } else {
                node.set_model_runtime_context_length(&model_name, None)
                    .await;
                node.regossip().await;
                eprintln!("⚠️  [{}] MoE model too large to serve entirely ({:.1}GB model, {:.1}GB capacity) — waiting for peers",
                    model_name, model_bytes as f64 / 1e9, my_vram as f64 / 1e9);
                on_change(false, false);
            }
        } else {
            let my_shard_index = plan.shard_index_for(my_id).unwrap_or(0);
            eprintln!(
                "🧩 [{}] MoE split mode — leader={} active={} fallback={} I am shard {}/{} (ranking={} origin={}, overlap={})",
                model_name,
                plan.leader_id.fmt_short(),
                plan.active_ids.len(),
                plan.fallback_ids.len(),
                my_shard_index,
                plan.active_ids.len(),
                moe_cfg.ranking_source,
                moe_cfg.ranking_origin,
                plan.overlap
            );
            on_change(true, false);

            let assignments = moe::compute_assignments_with_overlap(
                &moe_cfg.config.ranking,
                plan.active_ids.len(),
                moe_cfg.config.min_experts_per_node,
                plan.overlap,
            );
            let my_assignment = &assignments[my_shard_index];
            eprintln!(
                "  My experts: {} ({} shared + {} unique)",
                my_assignment.experts.len(),
                my_assignment.n_shared,
                my_assignment.n_unique
            );

            // Advertise a non-ready local runtime before split generation / load so
            // peer liveness stays conservative during MoE convergence.
            node.set_model_runtime_starting(&model_name).await;
            node.regossip().await;

            let shard_path = moe::split_path(&model, plan.active_ids.len(), my_shard_index);

            if !shard_path.exists() {
                eprintln!("  Splitting GGUF → {} ...", shard_path.display());
                match moe::run_split(&bin_dir, &model, my_assignment, &shard_path) {
                    Ok(()) => {
                        let size = std::fs::metadata(&shard_path).map(|m| m.len()).unwrap_or(0);
                        eprintln!("  Split complete: {:.1} GB", size as f64 / 1e9);
                    }
                    Err(e) => {
                        eprintln!("  ❌ moe-split failed: {e}");
                        node.set_model_runtime_context_length(&model_name, None)
                            .await;
                        node.regossip().await;
                        if peer_rx.changed().await.is_err() {
                            break;
                        }
                        tokio::time::sleep(std::time::Duration::from_secs(10)).await;
                        continue;
                    }
                }
            } else {
                let size = std::fs::metadata(&shard_path).map(|m| m.len()).unwrap_or(0);
                eprintln!(
                    "  Using cached shard: {} ({:.1} GB)",
                    shard_path.display(),
                    size as f64 / 1e9
                );
            }

            // Start llama-server with our shard
            let llama_port = match find_free_port().await {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("  Failed to find free port: {e}");
                    if peer_rx.changed().await.is_err() {
                        break;
                    }
                    tokio::time::sleep(std::time::Duration::from_secs(3)).await;
                    continue;
                }
            };

            let shard_bytes = std::fs::metadata(&shard_path).map(|m| m.len()).unwrap_or(0);
            match launch::start_llama_server(
                &runtime,
                &bin_dir,
                binary_flavor,
                launch::ModelLaunchSpec {
                    model: &shard_path,
                    http_port: llama_port,
                    tunnel_ports: &[],
                    tensor_split: None,
                    split_mode: local_multi_gpu_split_mode(binary_flavor),
                    draft: None,
                    draft_max: 0,
                    model_bytes: shard_bytes,
                    my_vram,
                    mmproj: None,
                    ctx_size_override,
                    total_group_vram: None,
                },
            )
            .await
            {
                Ok(process) => {
                    node.set_role(NodeRole::Host {
                        http_port: ingress_http_port,
                    })
                    .await;
                    tunnel_mgr.set_http_port(llama_port);
                    currently_running = true;
                    current_local_port = Some(llama_port);
                    llama_process = Some(process);
                    if let Some(ref process) = llama_process {
                        on_process(Some(LocalProcessInfo {
                            backend: "llama".into(),
                            pid: process.handle.pid(),
                            port: llama_port,
                            context_length: process.context_length,
                        }));
                    }
                    node.regossip().await;

                    let targets = build_moe_targets(
                        &plan.active_ids,
                        &plan.fallback_ids,
                        my_id,
                        Some(llama_port),
                        None,
                        &model_name,
                    );
                    target_tx.send_replace(targets);

                    on_change(true, true);
                    eprintln!(
                        "✅ [{}] MoE shard {} ready on port {llama_port} ({} experts)",
                        model_name,
                        my_shard_index,
                        my_assignment.experts.len()
                    );
                }
                Err(e) => {
                    eprintln!(
                        "  ❌ MoE split validation failed for shard {}: {e}",
                        shard_path.display()
                    );
                    eprintln!(
                        "  ⚠️  [{}] Refusing to enter MoE split mode on this node until the shard validates",
                        model_name
                    );
                    node.set_model_runtime_context_length(&model_name, None)
                        .await;
                    node.regossip().await;
                }
            }
        }

        // Wait for next peer change
        tokio::select! {
            res = peer_rx.changed() => {
                if res.is_err() { break; }
            }
            res = stop_rx.changed() => {
                if res.is_err() || stop_requested(&stop_rx) {
                    break;
                }
            }
        }
        if stop_requested(&stop_rx) {
            break;
        }
        eprintln!(
            "⚡ [{}] Mesh changed — re-checking MoE deployment...",
            model_name
        );
        tokio::time::sleep(std::time::Duration::from_secs(3)).await;
    }

    if currently_running {
        if let Some(process) = llama_process.take() {
            process.handle.shutdown().await;
        }
        tunnel_mgr.set_http_port(0);
        node.set_role(NodeRole::Worker).await;
        update_targets(&node, &model_name, InferenceTarget::None, &target_tx).await;
        on_process(None);
        on_change(false, false);
    }
}

/// Update the model targets map — sets our model's target and includes
/// targets for other models we know about from peers.
/// When multiple nodes serve the same model, all are included for load balancing.
fn extend_targets_from_peer(
    targets: &mut HashMap<String, Vec<InferenceTarget>>,
    peer_models: &[String],
    role: &NodeRole,
    peer_id: iroh::EndpointId,
) {
    // Only confirmed hosts can serve HTTP inference traffic.
    // Split workers may advertise the model they're helping serve, but they
    // only run rpc-server and will drop tunneled chat requests.
    if !matches!(role, NodeRole::Host { .. }) {
        return;
    }

    for serving in peer_models {
        targets
            .entry(serving.clone())
            .or_default()
            .push(InferenceTarget::Remote(peer_id));
    }
}

async fn update_targets(
    node: &mesh::Node,
    my_model: &str,
    my_target: InferenceTarget,
    target_tx: &Arc<watch::Sender<ModelTargets>>,
) {
    let peers = node.peers().await;
    let mut targets: HashMap<String, Vec<InferenceTarget>> = HashMap::new();

    // Start from the current targets — preserve local targets set by other election loops
    // (multi-model per node: each loop manages its own model's entry)
    {
        let current = target_tx.borrow();
        for (model, model_targets) in &current.targets {
            if model != my_model {
                // Keep only Local targets from other loops — remote targets get rebuilt below
                let locals: Vec<_> = model_targets
                    .iter()
                    .filter(|t| {
                        matches!(t, InferenceTarget::Local(_) | InferenceTarget::MoeLocal(_))
                    })
                    .cloned()
                    .collect();
                if !locals.is_empty() {
                    targets.insert(model.clone(), locals);
                }
            }
        }
    }

    // Our model — we're always first in the list
    if !matches!(my_target, InferenceTarget::None) {
        targets
            .entry(my_model.to_string())
            .or_default()
            .push(my_target);
    }

    // All peers — group by model (multi-model aware)
    for p in &peers {
        let peer_models = p.routable_models();
        extend_targets_from_peer(&mut targets, &peer_models, &p.role, p.id);
    }

    let count: usize = targets.values().map(|v| v.len()).sum();
    if count > 1 {
        for (model, hosts) in &targets {
            if hosts.len() > 1 {
                eprintln!(
                    "⚡ [{}] {} hosts available (load balancing)",
                    model,
                    hosts.len()
                );
            }
        }
    }

    target_tx.send_replace(ModelTargets {
        targets,
        moe: None,
        counter: Default::default(),
    });
}

/// Start llama-server with --rpc pointing at model-group nodes (self + workers).
/// Returns the ephemeral port and a death notification receiver, or None on failure.
#[allow(clippy::too_many_arguments)]
async fn start_llama(
    params: StartLlamaParams<'_>,
) -> Option<(u16, launch::InferenceServerProcess)> {
    let StartLlamaParams {
        runtime,
        node,
        tunnel_mgr,
        bin_dir,
        model,
        model_name,
        model_peers,
        explicit_mmproj,
        draft,
        draft_max,
        force_split,
        binary_flavor,
        ctx_size_override,
    } = params;
    let my_vram = node.vram_bytes();
    let model_bytes = total_model_bytes(model);
    let launch_plan =
        build_dense_launch_plan(my_vram, model_bytes, force_split, model_name, model_peers);
    let worker_ids = match launch_plan {
        DenseLaunchPlan::Solo => {
            let worker_count = model_peers
                .iter()
                .filter(|p| !matches!(p.role, NodeRole::Client))
                .count();
            if worker_count > 0 {
                eprintln!(
                    "  Model fits on host ({:.1}GB capacity for {:.1}GB model) — serving entirely",
                    my_vram as f64 / 1e9,
                    model_bytes as f64 / 1e9
                );
                eprintln!("  Use --split to force distributed mode");
            }
            Vec::new()
        }
        DenseLaunchPlan::Split { worker_ids, .. } => {
            for id in &worker_ids {
                if let Some(peer) = model_peers.iter().find(|peer| peer.id == *id) {
                    let rtt_str = peer
                        .rtt_ms
                        .map(|r| format!("{}ms", r))
                        .unwrap_or("?ms".to_string());
                    eprintln!(
                        "  ✓ Adding {} — {:.1}GB capacity, RTT {rtt_str}",
                        peer.id.fmt_short(),
                        split_peer_vram_bytes(peer, my_vram) as f64 / 1e9
                    );
                }
            }
            worker_ids.clone()
        }
        DenseLaunchPlan::WaitingForCapacity { .. } => {
            return None;
        }
    };

    // Wait for tunnels to workers
    if !worker_ids.is_empty() {
        eprintln!("  Waiting for tunnels to {} worker(s)...", worker_ids.len());
        let _ = tokio::time::timeout(
            std::time::Duration::from_secs(10),
            tunnel_mgr.wait_for_peers(worker_ids.len()),
        )
        .await;
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;

        // B2B tunnel map exchange
        let my_map = tunnel_mgr.peer_ports_map().await;
        let _ = node.broadcast_tunnel_map(my_map).await;
        let _ = node
            .wait_for_tunnel_maps(worker_ids.len(), std::time::Duration::from_secs(10))
            .await;
        let remote_maps = node.all_remote_tunnel_maps().await;
        tunnel_mgr.update_rewrite_map(&remote_maps).await;
    }

    // Build --rpc list: only remote workers.
    // The host's own GPU is used directly on the local backend — no need to route
    // through the local rpc-server (which would add unnecessary TCP round trips).
    let all_ports = tunnel_mgr.peer_ports_map().await;
    let Some(rpc_ports) = rpc_ports_for_worker_ids(&all_ports, &worker_ids) else {
        eprintln!(
            "  Waiting for selected worker tunnels ({}/{} ready)",
            all_ports
                .keys()
                .filter(|id| worker_ids.contains(id))
                .count(),
            worker_ids.len()
        );
        return None;
    };

    // Calculate tensor split from VRAM.
    // Device order: RPC workers first (matching --rpc order), then the local host device last.
    let my_vram_f = my_vram as f64;
    let mut all_vrams: Vec<f64> = Vec::new();
    for id in &worker_ids {
        if let Some(peer) = model_peers.iter().find(|p| p.id == *id) {
            all_vrams.push(split_peer_vram_bytes(peer, my_vram) as f64);
        }
    }
    all_vrams.push(my_vram_f); // Host device is last
    let total: f64 = all_vrams.iter().sum();
    let split = if total > 0.0 && !rpc_ports.is_empty() {
        let s: Vec<String> = all_vrams
            .iter()
            .map(|v| format!("{:.2}", v / total))
            .collect();
        let split_str = s.join(",");
        eprintln!(
            "  Tensor split: {split_str} ({} node(s), {:.0}GB total)",
            rpc_ports.len() + 1,
            total / 1e9
        );
        Some(split_str)
    } else {
        eprintln!("  Serving entirely ({:.0}GB capacity)", my_vram_f / 1e9);
        None
    };

    // Launch on ephemeral port
    let llama_port = match find_free_port().await {
        Ok(p) => p,
        Err(e) => {
            eprintln!("  Failed to find free port: {e}");
            return None;
        }
    };

    // Look up mmproj for vision models
    let mmproj_path = crate::models::resolve_mmproj_path(model_name, model, explicit_mmproj);

    // In split mode (pipeline parallel), pass total group VRAM so context size
    // accounts for the host only holding its share of layers. KV cache is also
    // distributed — each node holds KV for its own layers.
    let group_vram = if !rpc_ports.is_empty() {
        Some(total as u64)
    } else {
        None
    };

    match launch::start_llama_server(
        runtime,
        bin_dir,
        binary_flavor,
        launch::ModelLaunchSpec {
            model,
            http_port: llama_port,
            tunnel_ports: &rpc_ports,
            tensor_split: split.as_deref(),
            // Row split only works for local multi-GPU — not over RPC.
            // When we have RPC workers, llama.cpp uses layer (pipeline) split.
            split_mode: if rpc_ports.is_empty() {
                local_multi_gpu_split_mode(binary_flavor)
            } else {
                None
            },
            draft,
            draft_max,
            model_bytes,
            my_vram,
            mmproj: mmproj_path.as_deref(),
            ctx_size_override,
            total_group_vram: group_vram,
        },
    )
    .await
    {
        Ok(process) => Some((llama_port, process)),
        Err(e) => {
            eprintln!("  Failed to start llama-server: {e}");
            None
        }
    }
}

async fn find_free_port() -> anyhow::Result<u16> {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let port = listener.local_addr()?.port();
    drop(listener);
    Ok(port)
}

#[cfg(test)]
mod tests {
    use super::*;
    use iroh::EndpointAddr;
    use iroh::SecretKey;

    /// Create a deterministic EndpointId from a byte seed.
    fn make_id(seed: u8) -> iroh::EndpointId {
        let mut bytes = [0u8; 32];
        bytes[0] = seed;
        SecretKey::from_bytes(&bytes).public()
    }

    fn make_dense_peer(
        id: iroh::EndpointId,
        vram_bytes: u64,
        rtt_ms: Option<u32>,
        serving_model: &str,
    ) -> mesh::PeerInfo {
        mesh::PeerInfo {
            id,
            addr: EndpointAddr {
                id,
                addrs: Default::default(),
            },
            tunnel_port: None,
            role: NodeRole::Worker,
            models: vec![],
            vram_bytes,
            rtt_ms,
            model_source: None,
            serving_models: vec![serving_model.to_string()],
            hosted_models: vec![],
            hosted_models_known: false,
            available_models: vec![],
            requested_models: vec![],
            last_seen: std::time::Instant::now(),
            last_mentioned: std::time::Instant::now(),
            moe_recovered_at: None,
            version: None,
            gpu_name: None,
            hostname: None,
            is_soc: None,
            gpu_vram: None,
            gpu_reserved_bytes: None,
            gpu_mem_bandwidth_gbps: None,
            gpu_compute_tflops_fp32: None,
            gpu_compute_tflops_fp16: None,
            available_model_metadata: vec![],
            experts_summary: None,
            available_model_sizes: HashMap::new(),
            served_model_descriptors: vec![],
            served_model_runtime: vec![],
            owner_attestation: None,
            owner_summary: crate::crypto::OwnershipSummary::default(),
        }
    }

    #[test]
    fn dense_launch_plan_prefers_lowest_rtt_workers_needed_for_capacity() {
        let model = "dense";
        let id_a = make_id(1);
        let id_b = make_id(2);
        let id_c = make_id(3);
        let id_d = make_id(4);
        let peers = vec![
            make_dense_peer(id_b, 30, Some(60), model),
            make_dense_peer(id_c, 30, Some(20), model),
            make_dense_peer(id_d, 30, Some(40), model),
        ];

        let plan = build_dense_launch_plan(60, 100, false, model, &peers);
        assert_eq!(
            plan,
            DenseLaunchPlan::Split {
                worker_ids: vec![id_c, id_d],
                total_group_vram: 120,
            }
        );

        assert!(should_be_host_for_model(id_a, 60, &peers));
    }

    #[test]
    fn dense_launch_plan_ignores_unselected_spare_worker_churn() {
        let model = "dense";
        let id_b = make_id(2);
        let id_c = make_id(3);
        let id_d = make_id(4);
        let base = vec![
            make_dense_peer(id_b, 30, Some(10), model),
            make_dense_peer(id_c, 30, Some(20), model),
        ];
        let mut with_spare = base.clone();
        with_spare.push(make_dense_peer(id_d, 50, Some(70), model));

        let base_plan = build_dense_launch_plan(60, 100, false, model, &base);
        let spare_plan = build_dense_launch_plan(60, 100, false, model, &with_spare);

        assert_eq!(base_plan.running_plan(), spare_plan.running_plan());
        assert_eq!(
            base_plan.running_plan(),
            Some(DenseRunningPlan::Split {
                worker_ids: vec![id_b, id_c],
            })
        );
    }

    #[test]
    fn dense_launch_plan_replans_across_surviving_workers_after_peer_loss() {
        let model = "dense";
        let id_b = make_id(2);
        let id_c = make_id(3);
        let id_d = make_id(4);
        let initial = vec![
            make_dense_peer(id_b, 30, Some(10), model),
            make_dense_peer(id_c, 30, Some(20), model),
            make_dense_peer(id_d, 30, Some(30), model),
        ];
        let survivors = vec![
            make_dense_peer(id_c, 30, Some(20), model),
            make_dense_peer(id_d, 30, Some(30), model),
        ];

        let initial_plan = build_dense_launch_plan(50, 100, false, model, &initial);
        let survivor_plan = build_dense_launch_plan(50, 100, false, model, &survivors);

        assert_eq!(
            initial_plan.running_plan(),
            Some(DenseRunningPlan::Split {
                worker_ids: vec![id_b, id_c],
            })
        );
        assert_eq!(
            survivor_plan.running_plan(),
            Some(DenseRunningPlan::Split {
                worker_ids: vec![id_c, id_d],
            })
        );
    }

    #[test]
    fn dense_launch_plan_waits_when_only_ineligible_capacity_remains() {
        let model = "dense";
        let id_b = make_id(2);
        let id_c = make_id(3);
        let peers = vec![
            make_dense_peer(id_b, 30, Some(10), model),
            make_dense_peer(id_c, 40, Some(mesh::MAX_SPLIT_RTT_MS + 1), model),
        ];

        let plan = build_dense_launch_plan(50, 100, false, model, &peers);
        assert_eq!(
            plan,
            DenseLaunchPlan::WaitingForCapacity {
                worker_ids: vec![id_b],
                total_group_vram: 80,
                min_vram: 110,
            }
        );
    }

    #[test]
    fn selected_worker_ids_require_complete_rpc_port_map() {
        let id_b = make_id(2);
        let id_c = make_id(3);
        let mut complete = HashMap::new();
        complete.insert(id_b, 9001);
        complete.insert(id_c, 9002);

        let ports =
            rpc_ports_for_worker_ids(&complete, &[id_b, id_c]).expect("all selected workers ready");
        assert_eq!(ports, vec![9001, 9002]);

        complete.remove(&id_c);
        assert!(
            rpc_ports_for_worker_ids(&complete, &[id_b, id_c]).is_none(),
            "launch must wait until every selected worker has a resolved RPC port"
        );
    }

    // ── Shard index computation ──

    #[test]
    fn test_shard_index_2_nodes() {
        let id_a = make_id(1);
        let id_b = make_id(2);

        let (all_a, idx_a) = moe_shard_index(id_a, &[id_b]);
        let (all_b, idx_b) = moe_shard_index(id_b, &[id_a]);

        // Both should see the same sorted order
        assert_eq!(all_a, all_b);
        // They should have different indices
        assert_ne!(idx_a, idx_b);
        // Indices should cover 0..2
        let mut indices = vec![idx_a, idx_b];
        indices.sort();
        assert_eq!(indices, vec![0, 1]);
    }

    #[test]
    fn test_shard_index_3_nodes() {
        let id_a = make_id(1);
        let id_b = make_id(2);
        let id_c = make_id(3);

        let (_, idx_a) = moe_shard_index(id_a, &[id_b, id_c]);
        let (_, idx_b) = moe_shard_index(id_b, &[id_a, id_c]);
        let (_, idx_c) = moe_shard_index(id_c, &[id_a, id_b]);

        let mut indices = vec![idx_a, idx_b, idx_c];
        indices.sort();
        assert_eq!(indices, vec![0, 1, 2]);
    }

    #[test]
    fn test_shard_index_solo() {
        let id = make_id(42);
        let (all, idx) = moe_shard_index(id, &[]);
        assert_eq!(all.len(), 1);
        assert_eq!(idx, 0);
    }

    #[test]
    fn test_shard_index_stable_across_calls() {
        // Same inputs should always give same outputs
        let id_a = make_id(10);
        let id_b = make_id(20);
        let id_c = make_id(30);

        let (order1, idx1) = moe_shard_index(id_a, &[id_b, id_c]);
        let (order2, idx2) = moe_shard_index(id_a, &[id_c, id_b]); // different peer order
        assert_eq!(order1, order2); // sorted, so same
        assert_eq!(idx1, idx2);
    }

    #[test]
    fn test_shard_index_my_id_already_in_peers() {
        // Edge case: what if peers list already contains my ID?
        let id_a = make_id(1);
        let id_b = make_id(2);
        let (all, idx) = moe_shard_index(id_a, &[id_a, id_b]);
        // Should not duplicate
        assert_eq!(all.len(), 2);
        assert!(idx < 2);
    }

    // ── MoE target map construction ──

    #[test]
    fn test_build_moe_targets_2_nodes() {
        let id_a = make_id(1);
        let id_b = make_id(2);
        let (sorted, _) = moe_shard_index(id_a, &[id_b]);

        let targets = build_moe_targets(&sorted, &[], id_a, Some(8080), None, "test-model");

        // Should have MoE state
        let moe = targets.moe.as_ref().unwrap();
        assert_eq!(moe.nodes.len(), 2);

        // Model should be in targets
        assert!(matches!(
            targets.get("test-model"),
            InferenceTarget::MoeLocal(8080)
        ));

        // One should be local, one remote
        let local_count = moe
            .nodes
            .iter()
            .filter(|t| matches!(t, InferenceTarget::MoeLocal(_)))
            .count();
        let remote_count = moe
            .nodes
            .iter()
            .filter(|t| matches!(t, InferenceTarget::MoeRemote(_)))
            .count();
        assert_eq!(local_count, 1);
        assert_eq!(remote_count, 1);
    }

    #[test]
    fn test_build_moe_targets_local_port_correct() {
        let id_a = make_id(1);
        let id_b = make_id(2);
        let (sorted, idx_a) = moe_shard_index(id_a, &[id_b]);

        let targets = build_moe_targets(&sorted, &[], id_a, Some(9999), None, "m");
        let moe = targets.moe.as_ref().unwrap();

        // Our index in the MoE state should have our port
        match &moe.nodes[idx_a] {
            InferenceTarget::MoeLocal(port) => assert_eq!(*port, 9999),
            other => panic!("Expected MoeLocal(9999), got {:?}", other),
        }
    }

    #[test]
    fn test_build_moe_targets_reconfigures_when_third_node_drops() {
        let id_a = make_id(1);
        let id_b = make_id(2);
        let id_c = make_id(3);

        let (sorted_three, _) = moe_shard_index(id_a, &[id_b, id_c]);
        let targets_three = build_moe_targets(&sorted_three, &[], id_a, Some(8080), None, "m");
        let moe_three = targets_three.moe.as_ref().unwrap();
        assert_eq!(moe_three.nodes.len(), 3);
        assert!(moe_three
            .nodes
            .iter()
            .any(|target| matches!(target, InferenceTarget::MoeRemote(id) if *id == id_c)));

        let (sorted_two, _) = moe_shard_index(id_a, &[id_b]);
        let targets_two = build_moe_targets(&sorted_two, &[], id_a, Some(8080), None, "m");
        let moe_two = targets_two.moe.as_ref().unwrap();
        assert_eq!(moe_two.nodes.len(), 2);
        assert!(!moe_two
            .nodes
            .iter()
            .any(|target| matches!(target, InferenceTarget::MoeRemote(id) if *id == id_c)));

        // The survivor should still route locally, but only across the 2 remaining shards.
        assert!(matches!(
            targets_two.get("m"),
            InferenceTarget::MoeLocal(8080)
        ));
    }

    #[test]
    fn test_build_moe_targets_collapse_to_single_node_after_peer_loss() {
        let id_a = make_id(1);
        let id_b = make_id(2);

        let (sorted_two, _) = moe_shard_index(id_a, &[id_b]);
        let targets_two = build_moe_targets(&sorted_two, &[], id_a, Some(8080), None, "m");
        let moe_two = targets_two.moe.as_ref().unwrap();
        assert_eq!(moe_two.nodes.len(), 2);

        let targets_one = build_moe_targets(&[id_a], &[], id_a, Some(8080), None, "m");
        let moe_one = targets_one.moe.as_ref().unwrap();
        assert_eq!(moe_one.nodes.len(), 1);
        assert!(matches!(moe_one.nodes[0], InferenceTarget::MoeLocal(8080)));

        for i in 0..20 {
            match targets_one.get_moe_target(&format!("after-drop-{i}")) {
                Some(InferenceTarget::MoeLocal(8080)) => {}
                other => panic!("Expected MoeLocal(8080) after collapse, got {:?}", other),
            }
        }
    }

    #[test]
    fn test_build_moe_targets_include_full_fallback_candidates() {
        let id_a = make_id(1);
        let id_b = make_id(2);
        let id_c = make_id(3);
        let targets = build_moe_targets(&[id_a, id_b], &[id_c], id_a, Some(8080), None, "m");
        let moe = targets.moe.as_ref().unwrap();
        assert_eq!(moe.nodes.len(), 2);
        assert_eq!(moe.fallbacks.len(), 1);
        assert!(matches!(moe.fallbacks[0], InferenceTarget::Remote(id) if id == id_c));

        let candidates = targets.get_moe_failover_targets("session");
        assert_eq!(candidates.len(), 2);
        assert!(matches!(candidates[1], InferenceTarget::Remote(id) if id == id_c));
    }

    #[test]
    fn test_plan_moe_placement_reserves_full_fallback_when_spare_node_exists() {
        let id_a = make_id(1);
        let id_b = make_id(2);
        let id_c = make_id(3);
        let id_d = make_id(4);

        let plan = plan_moe_placement(
            vec![
                MoePlacementCandidate {
                    id: id_a,
                    vram_bytes: 40,
                    full_coverage: true,
                },
                MoePlacementCandidate {
                    id: id_b,
                    vram_bytes: 24,
                    full_coverage: false,
                },
                MoePlacementCandidate {
                    id: id_c,
                    vram_bytes: 24,
                    full_coverage: false,
                },
                MoePlacementCandidate {
                    id: id_d,
                    vram_bytes: 24,
                    full_coverage: false,
                },
            ],
            &[],
            &[],
            true,
        )
        .unwrap();

        assert_eq!(plan.leader_id, id_a);
        assert_eq!(plan.active_ids.len(), 3);
        assert_eq!(plan.fallback_ids, vec![id_a]);
        assert_eq!(plan.overlap, 2);
    }

    #[test]
    fn test_plan_moe_placement_keeps_current_active_set_during_recovery() {
        let id_a = make_id(1);
        let id_b = make_id(2);
        let id_c = make_id(3);

        let plan = plan_moe_placement(
            vec![
                MoePlacementCandidate {
                    id: id_a,
                    vram_bytes: 48,
                    full_coverage: true,
                },
                MoePlacementCandidate {
                    id: id_b,
                    vram_bytes: 24,
                    full_coverage: false,
                },
                MoePlacementCandidate {
                    id: id_c,
                    vram_bytes: 24,
                    full_coverage: false,
                },
            ],
            &[id_b, id_c],
            &[],
            false,
        )
        .unwrap();

        assert_eq!(plan.active_ids, vec![id_b, id_c]);
        assert_eq!(plan.fallback_ids, Vec::<iroh::EndpointId>::new());
        assert_eq!(plan.overlap, 1);
    }

    #[test]
    fn test_plan_moe_placement_scales_up_after_quiet_window_when_materially_better() {
        let id_a = make_id(1);
        let id_b = make_id(2);
        let id_c = make_id(3);

        let plan = plan_moe_placement(
            vec![
                MoePlacementCandidate {
                    id: id_a,
                    vram_bytes: 48,
                    full_coverage: true,
                },
                MoePlacementCandidate {
                    id: id_b,
                    vram_bytes: 24,
                    full_coverage: false,
                },
                MoePlacementCandidate {
                    id: id_c,
                    vram_bytes: 24,
                    full_coverage: false,
                },
            ],
            &[id_b, id_c],
            &[],
            true,
        )
        .unwrap();

        assert_eq!(plan.active_ids, vec![id_b, id_c]);
        assert_eq!(plan.fallback_ids, vec![id_a]);
        assert_eq!(plan.overlap, 1);
    }

    #[test]
    fn test_running_plan_state_ignores_stale_plan_when_not_running() {
        let id_a = make_id(1);
        let id_b = make_id(2);
        let stale = MoePlacementPlan {
            leader_id: id_a,
            active_ids: vec![id_a],
            fallback_ids: vec![id_b],
            overlap: 1,
        };

        let (active_ids, fallback_ids) = running_plan_state(Some(&stale), false);
        assert!(active_ids.is_empty());
        assert!(fallback_ids.is_empty());

        let (active_ids, fallback_ids) = running_plan_state(Some(&stale), true);
        assert_eq!(active_ids, &[id_a]);
        assert_eq!(fallback_ids, &[id_b]);
    }

    #[test]
    fn test_extend_targets_ignores_non_host_peer() {
        let mut targets = HashMap::new();
        let worker_id = make_id(7);
        let models = vec!["Qwen3-Coder-Next-Q4_K_M".to_string()];

        extend_targets_from_peer(&mut targets, &models, &NodeRole::Worker, worker_id);

        assert!(targets.is_empty());
    }

    #[test]
    fn test_extend_targets_worker_before_host_only_keeps_host() {
        let mut targets = HashMap::new();
        let worker_id = make_id(7);
        let host_id = make_id(8);
        let models = vec!["Qwen3-Coder-Next-Q4_K_M".to_string()];

        extend_targets_from_peer(&mut targets, &models, &NodeRole::Worker, worker_id);
        extend_targets_from_peer(
            &mut targets,
            &models,
            &NodeRole::Host { http_port: 8080 },
            host_id,
        );

        let model_targets = targets.get("Qwen3-Coder-Next-Q4_K_M").unwrap();
        assert_eq!(model_targets.len(), 1);
        assert!(matches!(model_targets[0], InferenceTarget::Remote(id) if id == host_id));
    }

    #[test]
    fn test_extend_targets_keeps_multiple_hosts_for_load_balancing() {
        let mut targets = HashMap::new();
        let host_a = make_id(8);
        let host_b = make_id(9);
        let models = vec!["Qwen3-8B-Q4_K_M".to_string()];

        extend_targets_from_peer(
            &mut targets,
            &models,
            &NodeRole::Host { http_port: 8080 },
            host_a,
        );
        extend_targets_from_peer(
            &mut targets,
            &models,
            &NodeRole::Host { http_port: 8081 },
            host_b,
        );

        let model_targets = targets.get("Qwen3-8B-Q4_K_M").unwrap();
        assert_eq!(model_targets.len(), 2);
        assert!(matches!(model_targets[0], InferenceTarget::Remote(id) if id == host_a));
        assert!(matches!(model_targets[1], InferenceTarget::Remote(id) if id == host_b));
    }

    #[test]
    fn test_model_targets_round_robin_multiple_hosts() {
        let mut targets = ModelTargets::default();
        targets.targets.insert(
            "m".to_string(),
            vec![
                InferenceTarget::Local(7001),
                InferenceTarget::Local(7002),
                InferenceTarget::Local(7003),
            ],
        );

        assert!(matches!(targets.get("m"), InferenceTarget::Local(7001)));
        assert!(matches!(targets.get("m"), InferenceTarget::Local(7002)));
        assert!(matches!(targets.get("m"), InferenceTarget::Local(7003)));
        assert!(matches!(targets.get("m"), InferenceTarget::Local(7001)));
    }

    #[test]
    fn test_model_targets_round_robin_shared_across_clones() {
        let mut targets = ModelTargets::default();
        targets.targets.insert(
            "m".to_string(),
            vec![InferenceTarget::Local(8001), InferenceTarget::Local(8002)],
        );

        let clone = targets.clone();

        assert!(matches!(targets.get("m"), InferenceTarget::Local(8001)));
        assert!(matches!(clone.get("m"), InferenceTarget::Local(8002)));
        assert!(matches!(targets.get("m"), InferenceTarget::Local(8001)));
    }

    // ── Session hash routing ──

    #[test]
    fn test_session_routing_sticky() {
        let id_a = make_id(1);
        let id_b = make_id(2);
        let (sorted, _) = moe_shard_index(id_a, &[id_b]);
        let targets = build_moe_targets(&sorted, &[], id_a, Some(8080), None, "m");

        // Same session hint should always route to same node
        let t1 = targets.get_moe_target("user-123");
        let t2 = targets.get_moe_target("user-123");
        assert_eq!(format!("{:?}", t1), format!("{:?}", t2));
    }

    #[test]
    fn test_session_routing_distributes() {
        let id_a = make_id(1);
        let id_b = make_id(2);
        let (sorted, _) = moe_shard_index(id_a, &[id_b]);
        let targets = build_moe_targets(&sorted, &[], id_a, Some(8080), None, "m");

        // With enough different sessions, both nodes should get traffic
        let mut hit_local = false;
        let mut hit_remote = false;
        for i in 0..100 {
            let hint = format!("session-{i}");
            match targets.get_moe_target(&hint) {
                Some(InferenceTarget::MoeLocal(_)) => hit_local = true,
                Some(InferenceTarget::MoeRemote(_)) => hit_remote = true,
                _ => {}
            }
        }
        assert!(hit_local, "Should route some sessions locally");
        assert!(hit_remote, "Should route some sessions to remote");
    }

    #[test]
    fn test_session_routing_empty_moe() {
        let targets = ModelTargets::default();
        assert!(targets.get_moe_target("anything").is_none());
    }

    #[test]
    fn test_session_routing_single_node() {
        let id_a = make_id(1);
        let targets = build_moe_targets(&[id_a], &[], id_a, Some(8080), None, "m");

        // All sessions should go to the single node
        for i in 0..20 {
            match targets.get_moe_target(&format!("s{i}")) {
                Some(InferenceTarget::MoeLocal(8080)) => {}
                other => panic!("Expected MoeLocal(8080), got {:?}", other),
            }
        }
    }

    // ── Both nodes agree on the same assignments ──

    #[test]
    fn test_both_nodes_get_consistent_view() {
        // If node A and B both compute assignments for 2 nodes,
        // they should get the same expert lists (just different shard indices)
        let id_a = make_id(1);
        let id_b = make_id(2);

        let (_, idx_a) = moe_shard_index(id_a, &[id_b]);
        let (_, idx_b) = moe_shard_index(id_b, &[id_a]);

        let ranking: Vec<u32> = (0..128).collect();
        let assignments = crate::inference::moe::compute_assignments(&ranking, 2, 46);

        // Node A picks assignment[idx_a], Node B picks assignment[idx_b]
        // They should be different shards
        assert_ne!(idx_a, idx_b);
        // Their unique experts should not overlap
        let a_experts: std::collections::HashSet<u32> =
            assignments[idx_a].experts.iter().cloned().collect();
        let b_experts: std::collections::HashSet<u32> =
            assignments[idx_b].experts.iter().cloned().collect();
        let shared: Vec<u32> = a_experts.intersection(&b_experts).cloned().collect();
        // Shared should be exactly the core (first 46)
        assert_eq!(shared.len(), 46);
        // Union should cover all 128
        let union: std::collections::HashSet<u32> = a_experts.union(&b_experts).cloned().collect();
        assert_eq!(union.len(), 128);
    }

    #[test]
    fn test_pick_sticky_from_consistent() {
        let id_a = make_id(1);
        let id_b = make_id(2);
        let candidates = vec![InferenceTarget::Remote(id_a), InferenceTarget::Remote(id_b)];

        let first = ModelTargets::pick_sticky_from(&candidates, 42);
        let second = ModelTargets::pick_sticky_from(&candidates, 42);
        assert_eq!(first, second);
    }

    #[test]
    fn test_pick_sticky_from_empty_returns_none() {
        let result = ModelTargets::pick_sticky_from(&[], 42);
        assert_eq!(result, InferenceTarget::None);
    }

    #[test]
    fn test_pick_from_round_robins() {
        let id_a = make_id(1);
        let id_b = make_id(2);
        let targets = ModelTargets::default();
        let candidates = vec![InferenceTarget::Remote(id_a), InferenceTarget::Remote(id_b)];

        let first = targets.pick_from(&candidates);
        let second = targets.pick_from(&candidates);
        assert_ne!(first, second);
    }

    #[test]
    fn test_pick_from_empty_returns_none() {
        let targets = ModelTargets::default();
        let result = targets.pick_from(&[]);
        assert_eq!(result, InferenceTarget::None);
    }

    // ── Row-split / tensor-parallelism selection ──

    #[test]
    fn row_split_enabled_for_cuda_multi_gpu() {
        assert!(should_use_row_split(Some(BinaryFlavor::Cuda), 2));
        assert!(should_use_row_split(Some(BinaryFlavor::Cuda), 8));
    }

    #[test]
    fn row_split_enabled_for_rocm_multi_gpu() {
        assert!(should_use_row_split(Some(BinaryFlavor::Rocm), 2));
    }

    #[test]
    fn row_split_enabled_for_unknown_flavor_multi_gpu() {
        // None means auto-detected; the resolved binary may still be CUDA/ROCm.
        assert!(should_use_row_split(None, 2));
        assert!(should_use_row_split(None, 4));
    }

    #[test]
    fn row_split_disabled_for_single_gpu() {
        assert!(!should_use_row_split(Some(BinaryFlavor::Cuda), 1));
        assert!(!should_use_row_split(Some(BinaryFlavor::Rocm), 1));
        assert!(!should_use_row_split(None, 1));
    }

    #[test]
    fn row_split_disabled_for_zero_gpus() {
        assert!(!should_use_row_split(Some(BinaryFlavor::Cuda), 0));
        assert!(!should_use_row_split(None, 0));
    }

    #[test]
    fn row_split_disabled_for_non_cuda_backends() {
        // Metal, Vulkan, CPU don't support ggml_backend_split_buffer_type.
        assert!(!should_use_row_split(Some(BinaryFlavor::Metal), 8));
        assert!(!should_use_row_split(Some(BinaryFlavor::Vulkan), 4));
        assert!(!should_use_row_split(Some(BinaryFlavor::Cpu), 4));
    }
}
