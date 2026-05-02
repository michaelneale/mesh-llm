//! Automatic host election and dynamic mesh management.
//!
//! Per-model election: nodes serving the same model form a group.
//! The highest-VRAM node in each group becomes its host and runs llama-server.
//! Every mesh change: kill llama-server, re-elect, winner starts fresh.
//! mesh-llm owns :api_port and proxies to the right host by model name.

use crate::cli::output::{
    emit_event, MoeAnalysisProgressSummary, MoeDistributionSummary, MoeStatusSummary, MoeSummary,
    OutputEvent,
};
use crate::inference::{launch, moe, skippy};
use crate::mesh;
use crate::models;
use crate::network::tunnel;
use crate::system::hardware;
use launch::{BinaryFlavor, SplitMode};
use mesh::NodeRole;
use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};
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

fn split_mode_for_local_launch(
    flavor: Option<BinaryFlavor>,
    pinned_gpu: Option<&crate::runtime::StartupPinnedGpuTarget>,
) -> Option<SplitMode> {
    if pinned_gpu.is_some() {
        return None;
    }
    local_multi_gpu_split_mode(flavor)
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

fn effective_local_launch_vram(
    my_vram: u64,
    pinned_gpu: Option<&crate::runtime::StartupPinnedGpuTarget>,
    binary_flavor: Option<launch::BinaryFlavor>,
    gpu_vram: Option<&str>,
) -> u64 {
    if let Some(gpu) = pinned_gpu {
        return gpu.vram_bytes;
    }

    match binary_flavor {
        Some(launch::BinaryFlavor::Vulkan) | Some(launch::BinaryFlavor::Metal) => {
            primary_gpu_vram_bytes(gpu_vram)
                .map(|bytes| bytes.min(my_vram))
                .unwrap_or(my_vram)
        }
        Some(launch::BinaryFlavor::Cpu)
        | Some(launch::BinaryFlavor::Cuda)
        | Some(launch::BinaryFlavor::Rocm)
        | None => my_vram,
    }
}

fn primary_gpu_vram_bytes(gpu_vram: Option<&str>) -> Option<u64> {
    gpu_vram?
        .split(',')
        .next()
        .and_then(|value| value.trim().parse::<u64>().ok())
        .filter(|bytes| *bytes > 0)
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

/// The current inference target selected by the election loop.
/// The API proxy reads this to know where to forward requests.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum InferenceTarget {
    /// No backend running anywhere (election in progress, mesh empty, etc.)
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

fn emit_ready_events(
    model_name: &str,
    llama_port: u16,
    model_port: u16,
    ctx_size: u32,
    log_path: Option<String>,
) {
    let _ = emit_event(OutputEvent::LlamaReady {
        model: Some(model_name.to_string()),
        port: llama_port,
        ctx_size: Some(ctx_size),
        log_path,
    });
    let _ = emit_event(OutputEvent::ModelReady {
        model: model_name.to_string(),
        internal_port: Some(model_port),
        role: None,
    });
}

fn emit_moe_status(model_name: &str, phase: &str, detail: impl Into<String>) {
    let _ = emit_event(OutputEvent::MoeStatus {
        model: model_name.to_string(),
        status: MoeStatusSummary {
            phase: phase.to_string(),
            detail: detail.into(),
        },
    });
}

fn emit_warning(message: impl Into<String>, context: Option<String>) {
    let _ = emit_event(OutputEvent::Warning {
        message: message.into(),
        context,
    });
}

fn emit_error(message: impl Into<String>, context: Option<String>) {
    let _ = emit_event(OutputEvent::Error {
        message: message.into(),
        context,
    });
}

fn emit_info(message: impl Into<String>, context: Option<String>) {
    let _ = emit_event(OutputEvent::Info {
        message: message.into(),
        context,
    });
}

fn emit_moe_analysis_progress(
    model_name: &str,
    mode: &str,
    spinner: &str,
    current: usize,
    total: Option<usize>,
    elapsed_secs: u64,
) {
    let _ = emit_event(OutputEvent::MoeAnalysisProgress {
        model: model_name.to_string(),
        progress: MoeAnalysisProgressSummary {
            mode: mode.to_string(),
            spinner: spinner.to_string(),
            current,
            total,
            elapsed_secs,
        },
    });
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

    emit_moe_status(
        model_name,
        "waiting for peer ranking",
        format!("up to {:.0}s before local analysis", timeout.as_secs_f64()),
    );

    let deadline = tokio::time::Instant::now() + timeout;
    loop {
        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        if remaining.is_zero() {
            emit_moe_status(
                model_name,
                "peer ranking timeout",
                "continuing with local analysis",
            );
            return;
        }

        tokio::select! {
            _ = tokio::time::sleep(remaining) => {
                emit_moe_status(
                    model_name,
                    "peer ranking timeout",
                    "continuing with local analysis",
                );
                return;
            }
            res = peer_rx.changed() => {
                if res.is_err() {
                    return;
                }
                if let Some(artifact) = moe::best_shared_ranking_artifact(model_path) {
                    emit_moe_status(
                        model_name,
                        "using imported peer ranking",
                        format!(
                            "mode={} origin={}",
                            artifact.kind.label(),
                            artifact.origin.label()
                        ),
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
const SKIPPY_STAGE_FAILURE_QUARANTINE: Duration = Duration::from_secs(60);

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
    emit_moe_status(
        model_name,
        "auto-detected MoE",
        format!(
            "{} experts, top-{}",
            info.expert_count, info.expert_used_count
        ),
    );

    // Conservative default: 50% shared core (safe floor for quality).
    // Without a ranking, we use sequential expert IDs (0..N).
    let min_experts = (info.expert_count as f64 * 0.5).ceil() as u32;

    // Check for cached ranking on disk
    let ranking_path = moe::ranking_cache_path(model_path);
    if let Some(ranking) = moe::load_cached_ranking(&ranking_path) {
        emit_moe_status(
            model_name,
            "using cached ranking",
            format!("{}", ranking_path.display()),
        );
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
        emit_moe_status(
            model_name,
            "skipping local micro-analyze",
            format!(
                "model needs about {:.1}GB with headroom, local capacity is {:.1}GB",
                model_bytes as f64 * 1.1 / 1e9,
                local_vram_budget as f64 / 1e9
            ),
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
                        emit_moe_status(
                            model_name,
                            "shared ranking resolve failed",
                            format!(
                                "falling back to local analysis or sequential expert order ({err})"
                            ),
                        );
                        Ok(None)
                    }
                    Err(err) => {
                        emit_moe_status(
                            model_name,
                            "shared ranking resolver join failed",
                            format!(
                                "falling back to local analysis or sequential expert order ({err})"
                            ),
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
                    emit_moe_status(
                        model_name,
                        "shared ranking resolve failed",
                        format!(
                            "falling back to local analysis or sequential expert order ({err})"
                        ),
                    );
                    None
                }
            };
            if let Some(resolved) = resolved_ranking {
                emit_moe_status(
                    model_name,
                    "using shared ranking",
                    format!(
                        "mode={} path={} source={}",
                        resolved.analyzer_id,
                        resolved.path.display(),
                        resolved.source.label()
                    ),
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
                            emit_moe_status(
                                model_name,
                                "micro-analyze failed",
                                format!("falling back to sequential expert order ({err})"),
                            );
                            (
                                (0..base.n_expert).collect(),
                                "sequential-fallback".to_string(),
                                "fallback".to_string(),
                            )
                        }
                    }
                } else {
                    emit_moe_status(
                        model_name,
                        "waiting for peer ranking",
                        "or using sequential fallback on this node",
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

    emit_moe_status(
        model_name,
        "ranking resolved",
        format!(
            "ranking={} origin={} in {:.1}s",
            ranking_source,
            ranking_origin,
            started.elapsed().as_secs_f64()
        ),
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
        analysis_path: None,
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

    emit_moe_status(
        model_name,
        "switching to better ranking",
        format!(
            "mode={} source={}",
            resolved.analyzer_id,
            resolved.source.label()
        ),
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
    emit_moe_status(model_name, "generated local ranking", "ready to share");
    emit_moe_status(
        model_name,
        "ranking cache",
        format!("{}", ranking_path.display()),
    );
    emit_moe_status(
        model_name,
        "published source",
        crate::system::moe_planner::DEFAULT_MOE_RANKINGS_DATASET.to_string(),
    );
    emit_moe_status(model_name, "published ranking", "not used on this run");
    emit_moe_status(
        model_name,
        "contribute ranking",
        format!("mesh-llm moe share '{}'", identity.distribution_ref()),
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

struct MoeElectionParams {
    runtime: Arc<crate::runtime::instance::InstanceRuntime>,
    node: mesh::Node,
    tunnel_mgr: tunnel::Manager,
    ingress_http_port: u16,
    bin_dir: std::path::PathBuf,
    model: std::path::PathBuf,
    model_name: String,
    moe_cfg: ResolvedMoeConfig,
    moe_summary: MoeSummary,
    my_vram: u64,
    model_bytes: u64,
    binary_flavor: Option<launch::BinaryFlavor>,
    ctx_size_override: Option<u32>,
    pinned_gpu: Option<crate::runtime::StartupPinnedGpuTarget>,
    target_tx: Arc<watch::Sender<ModelTargets>>,
    stop_rx: watch::Receiver<bool>,
    slots: usize,
}

struct StartSkippyLocalParams<'a> {
    node: &'a mesh::Node,
    model: &'a Path,
    model_name: &'a str,
    explicit_mmproj: Option<&'a Path>,
    ctx_size_override: Option<u32>,
    pinned_gpu: Option<&'a crate::runtime::StartupPinnedGpuTarget>,
    slots: usize,
}

struct StartSkippySplitParams<'a> {
    node: &'a mesh::Node,
    model: &'a Path,
    model_name: &'a str,
    package_info: Option<skippy::StagePackageInfo>,
    explicit_mmproj: Option<&'a Path>,
    model_peers: &'a [mesh::PeerInfo],
    worker_ids: &'a [iroh::EndpointId],
    ctx_size_override: Option<u32>,
    pinned_gpu: Option<&'a crate::runtime::StartupPinnedGpuTarget>,
    slots: usize,
}

struct SkippySplitDeployment {
    topology_id: String,
    run_id: String,
    context_length: u32,
    stage0_status: skippy::StageStatusSnapshot,
    stage0: skippy::SkippyModelHandle,
    http: skippy::SkippyHttpHandle,
    remote_stops: Vec<(iroh::EndpointId, skippy::StageStopRequest)>,
    remote_statuses: Vec<(iroh::EndpointId, skippy::StageStatusSnapshot)>,
}

struct SkippyLocalDeployment {
    context_length: u32,
    model: skippy::SkippyModelHandle,
    http: skippy::SkippyHttpHandle,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct SkippyStageFailure {
    stage_id: String,
    peer_id: Option<iroh::EndpointId>,
    reason: String,
}

impl SkippyLocalDeployment {
    fn http_port(&self) -> u16 {
        self.http.port()
    }

    async fn wait_for_failure(&self, node: &mesh::Node) -> SkippyStageFailure {
        loop {
            let status = self.model.status();
            if matches!(
                status.state,
                skippy::SkippyModelState::Failed | skippy::SkippyModelState::Stopped
            ) {
                return SkippyStageFailure {
                    stage_id: "stage-0".to_string(),
                    peer_id: Some(node.id()),
                    reason: status
                        .last_error
                        .unwrap_or_else(|| format!("local skippy model is {:?}", status.state)),
                };
            }
            tokio::time::sleep(std::time::Duration::from_secs(1)).await;
        }
    }

    async fn shutdown(self) {
        let _ = self.http.shutdown().await;
        self.model.shutdown();
    }
}

impl SkippySplitDeployment {
    fn http_port(&self) -> u16 {
        self.http.port()
    }

    async fn wait_for_failure(&self, node: &mesh::Node) -> SkippyStageFailure {
        loop {
            if let Some(failure) = self.check_failure(node).await {
                return failure;
            }
            tokio::time::sleep(std::time::Duration::from_secs(1)).await;
        }
    }

    async fn check_failure(&self, node: &mesh::Node) -> Option<SkippyStageFailure> {
        let stage0 = self.stage0.status();
        if matches!(
            stage0.state,
            skippy::SkippyModelState::Failed | skippy::SkippyModelState::Stopped
        ) {
            return Some(SkippyStageFailure {
                stage_id: self.stage0_status.stage_id.clone(),
                peer_id: Some(node.id()),
                reason: stage0
                    .last_error
                    .unwrap_or_else(|| format!("local stage 0 is {:?}", stage0.state)),
            });
        }

        for (peer_id, expected) in &self.remote_statuses {
            let filter = skippy::StageStatusFilter {
                topology_id: Some(self.topology_id.clone()),
                run_id: Some(self.run_id.clone()),
                stage_id: Some(expected.stage_id.clone()),
            };
            let response = node
                .send_stage_control(*peer_id, skippy::StageControlRequest::Status(filter))
                .await;
            match response {
                Ok(skippy::StageControlResponse::Status(statuses)) => {
                    let current = statuses.iter().find(|status| {
                        status.topology_id == self.topology_id
                            && status.run_id == self.run_id
                            && status.stage_id == expected.stage_id
                    });
                    if let Some(failure) =
                        active_stage_failure_from_status(Some(*peer_id), expected, current)
                    {
                        return Some(failure);
                    }
                }
                Ok(skippy::StageControlResponse::Ready(ready)) => {
                    if let Some(failure) = active_stage_failure_from_status(
                        Some(*peer_id),
                        expected,
                        Some(&ready.status),
                    ) {
                        return Some(failure);
                    }
                }
                Err(error) => {
                    return Some(SkippyStageFailure {
                        stage_id: expected.stage_id.clone(),
                        peer_id: Some(*peer_id),
                        reason: format!("stage status refresh failed: {error}"),
                    });
                }
            }
        }
        None
    }

    async fn shutdown(self, node: &mesh::Node) {
        self.shutdown_marking_failure(node, None).await;
    }

    async fn shutdown_marking_failure(
        self,
        node: &mesh::Node,
        failure: Option<&SkippyStageFailure>,
    ) {
        let Self {
            topology_id,
            run_id,
            stage0_status,
            stage0,
            http,
            remote_stops,
            remote_statuses,
            ..
        } = self;
        let failed_stage_id = failure.map(|failure| failure.stage_id.as_str());
        let failed_reason = failure.map(|failure| failure.reason.clone());
        let _ = http.shutdown().await;
        stage0.shutdown();
        let mut stage0_shutdown_status = stage0_status;
        if failed_stage_id == Some(stage0_shutdown_status.stage_id.as_str()) {
            stage0_shutdown_status.state = skippy::StageRuntimeState::Failed;
            stage0_shutdown_status.error = failed_reason.clone();
        } else {
            stage0_shutdown_status.state = skippy::StageRuntimeState::Stopped;
        }
        stage0_shutdown_status.shutdown_generation =
            stage0_shutdown_status.shutdown_generation.saturating_add(1);
        node.record_stage_status(Some(node.id()), stage0_shutdown_status)
            .await;
        node.stop_stage_transport_bridge(&topology_id, &run_id, "stage-1")
            .await;
        let remote_status_by_stage: HashMap<_, _> = remote_statuses
            .into_iter()
            .map(|(peer_id, status)| (status.stage_id.clone(), (peer_id, status)))
            .collect();
        if let Some(failure) = failure {
            if let Some((peer_id, status)) = remote_status_by_stage.get(&failure.stage_id) {
                let mut failed = status.clone();
                failed.state = skippy::StageRuntimeState::Failed;
                failed.error = Some(failure.reason.clone());
                failed.shutdown_generation = failed.shutdown_generation.saturating_add(1);
                node.record_stage_status(Some(*peer_id), failed).await;
            }
        }
        for (peer_id, stop) in remote_stops.into_iter().rev() {
            let stage_id = stop.stage_id.clone();
            if let Err(error) = node
                .send_stage_control(peer_id, skippy::StageControlRequest::Stop(stop))
                .await
            {
                if failed_stage_id == Some(stage_id.as_str()) {
                    continue;
                }
                if let Some((_, status)) = remote_status_by_stage.get(&stage_id) {
                    let mut failed = status.clone();
                    failed.state = skippy::StageRuntimeState::Failed;
                    failed.error = Some(format!("stage stop failed: {error}"));
                    node.record_stage_status(Some(peer_id), failed).await;
                }
            }
            node.stop_stage_transport_bridge(&topology_id, &run_id, &stage_id)
                .await;
        }
    }
}

fn prune_skippy_stage_quarantine(
    quarantined_peers: &mut HashMap<iroh::EndpointId, Instant>,
    now: Instant,
) {
    quarantined_peers.retain(|_, until| *until > now);
}

fn quarantine_skippy_stage_failure(
    quarantined_peers: &mut HashMap<iroh::EndpointId, Instant>,
    failure: &SkippyStageFailure,
    now: Instant,
) -> Option<iroh::EndpointId> {
    let peer_id = failure.peer_id?;
    quarantined_peers.insert(peer_id, now + SKIPPY_STAGE_FAILURE_QUARANTINE);
    Some(peer_id)
}

fn model_peers_for_election(
    peers: &[mesh::PeerInfo],
    model_name: &str,
    quarantined_peers: &HashMap<iroh::EndpointId, Instant>,
) -> Vec<mesh::PeerInfo> {
    peers
        .iter()
        .filter(|peer| peer.is_assigned_model(model_name))
        .filter(|peer| !quarantined_peers.contains_key(&peer.id))
        .cloned()
        .collect()
}

fn active_stage_failure_from_status(
    peer_id: Option<iroh::EndpointId>,
    expected: &skippy::StageStatusSnapshot,
    status: Option<&skippy::StageStatusSnapshot>,
) -> Option<SkippyStageFailure> {
    let Some(status) = status else {
        return Some(SkippyStageFailure {
            stage_id: expected.stage_id.clone(),
            peer_id,
            reason: "stage status missing from runtime".to_string(),
        });
    };
    if status.topology_id != expected.topology_id
        || status.run_id != expected.run_id
        || status.stage_id != expected.stage_id
    {
        return None;
    }
    match status.state {
        skippy::StageRuntimeState::Failed => Some(SkippyStageFailure {
            stage_id: status.stage_id.clone(),
            peer_id,
            reason: status
                .error
                .clone()
                .unwrap_or_else(|| "stage entered failed state".to_string()),
        }),
        skippy::StageRuntimeState::Stopping => Some(SkippyStageFailure {
            stage_id: status.stage_id.clone(),
            peer_id,
            reason: "stage started stopping during active topology".to_string(),
        }),
        skippy::StageRuntimeState::Stopped => Some(SkippyStageFailure {
            stage_id: status.stage_id.clone(),
            peer_id,
            reason: "stage stopped during active topology".to_string(),
        }),
        skippy::StageRuntimeState::Starting | skippy::StageRuntimeState::Ready => None,
    }
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
    pub pinned_gpu: Option<crate::runtime::StartupPinnedGpuTarget>,
    pub moe_runtime_options: moe::MoeRuntimeOptions,
    pub target_tx: Arc<watch::Sender<ModelTargets>>,
    pub stop_rx: watch::Receiver<bool>,
    pub slots: usize,
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
            emit_moe_analysis_progress(
                &model_name,
                mode,
                spinner,
                current,
                total,
                started.elapsed().as_secs(),
            );
            if done {
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
                emit_moe_status(&model_name, "moe-analyze warning", line);
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
        emit_moe_status(
            model_name,
            "using cached ranking",
            format!(
                "mode=full-analyze origin={} cache={}",
                artifact.origin.label(),
                cached_path.display()
            ),
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
    emit_moe_status(
        model_name,
        "MoE analysis",
        format!("mode=full-analyze cache={}", cached_path.display()),
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
    emit_moe_status(
        model_name,
        "full-analyze cached",
        format!(
            "{} in {:.1}s (origin={})",
            cached_path.display(),
            started.elapsed().as_secs_f64(),
            artifact.origin.label()
        ),
    );
    if !wrote_cache {
        emit_moe_status(
            model_name,
            "shared ranking already preferred",
            "full-v1 result was not promoted as the preferred shared artifact",
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
        emit_moe_status(
            model_name,
            "using cached ranking",
            format!(
                "mode=micro-analyze origin={} cache={}",
                artifact.origin.label(),
                cached_path.display()
            ),
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
    emit_moe_status(
        model_name,
        "micro-analyze cached",
        format!(
            "{} (origin={})",
            cached_path.display(),
            artifact.origin.label()
        ),
    );
    if !wrote_cache {
        emit_moe_status(
            model_name,
            "shared ranking already preferred",
            "micro-v1 result was not promoted as the preferred shared artifact",
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
    emit_moe_status(
        model_name,
        "MoE analysis",
        format!(
            "mode=micro-analyze prompts={} tokens={} layers={} cache=pending",
            prompt_count,
            options.micro_tokens,
            match options.micro_layer_scope {
                moe::MoeMicroLayerScope::All => "all",
                moe::MoeMicroLayerScope::First => "first",
            }
        ),
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
    emit_moe_status(
        model_name,
        "micro-analyze complete",
        format!(
            "{} prompt(s), {} token(s), {} in {:.1}s",
            prompt_count,
            options.micro_tokens,
            match options.micro_layer_scope {
                moe::MoeMicroLayerScope::All => "all layers",
                moe::MoeMicroLayerScope::First => "first layer",
            },
            started.elapsed().as_secs_f64()
        ),
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
/// 1. Stop any active local runtime
/// 2. Re-elect within the model group
/// 3. Winner starts embedded skippy locally or as stage 0 of a staged topology
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
        draft: _draft,
        draft_max: _draft_max,
        force_split,
        binary_flavor,
        ctx_size_override,
        pinned_gpu,
        moe_runtime_options,
        target_tx,
        mut stop_rx,
        slots,
    } = params;
    let mut peer_rx = node.peer_change_rx.clone();

    // Track the actual running launch topology so we only restart on real split changes.
    let mut last_running_plan: Option<DenseRunningPlan> = None;
    let mut currently_host = false;
    let mut current_local_port: Option<u16> = None;
    let mut skippy_local: Option<SkippyLocalDeployment> = None;
    let mut skippy_split: Option<SkippySplitDeployment> = None;
    let mut skippy_stage_quarantine: HashMap<iroh::EndpointId, Instant> = HashMap::new();

    // Initial settle
    tokio::time::sleep(std::time::Duration::from_secs(2)).await;

    let package_source_info = skippy::is_layer_package_ref(&model.to_string_lossy())
        .then(|| skippy::inspect_stage_package(&model.to_string_lossy()))
        .transpose()
        .map_err(|error| {
            emit_warning(
                format!("Failed to inspect skippy package: {error}"),
                Some(format!("model={model_name} path={}", model.display())),
            );
            error
        })
        .ok()
        .flatten();
    let model_bytes = package_source_info
        .as_ref()
        .and_then(|info| info.source_model_bytes)
        .unwrap_or_else(|| total_model_bytes(&model));
    let my_vram = node.vram_bytes();
    let local_launch_vram = effective_local_launch_vram(
        my_vram,
        pinned_gpu.as_ref(),
        binary_flavor,
        node.gpu_vram.as_deref(),
    );
    let model_fits_locally = local_launch_vram >= (model_bytes as f64 * 1.1) as u64;

    // Check if this is a MoE model with enough metadata to plan expert routing.
    let moe_config = lookup_moe_config(&model_name, &model);
    let moe_summary = moe_config.as_ref().map(|moe_config| MoeSummary {
        experts: moe_config.n_expert,
        top_k: moe_config.n_expert_used,
    });
    if moe_summary.is_some() {
        if let Some(moe_summary) = &moe_summary {
            let _ = emit_event(OutputEvent::MoeDetected {
                model: model_name.clone(),
                moe: moe_summary.clone(),
                fits_locally: None,
                capacity_gb: None,
                model_gb: None,
            });
        }
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
                    emit_warning(
                        "Failed to resolve MoE split config",
                        Some(format!("model={model_name}")),
                    );
                    return;
                }
                Err(e) => {
                    emit_warning(
                        format!("Failed to resolve MoE ranking/grouping: {e}"),
                        Some(format!("model={model_name}")),
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
                    moe_summary: moe_summary
                        .clone()
                        .expect("MoE summary should exist when entering MoE mode"),
                    my_vram,
                    model_bytes,
                    binary_flavor,
                    ctx_size_override,
                    pinned_gpu: pinned_gpu.clone(),
                    target_tx,
                    stop_rx,
                    slots,
                },
                &mut on_change,
                &mut on_process,
            )
            .await;
            return;
        } else {
            if let Some(moe_summary) = &moe_summary {
                let _ = emit_event(OutputEvent::MoeDetected {
                    model: model_name.clone(),
                    moe: moe_summary.clone(),
                    fits_locally: Some(true),
                    capacity_gb: Some(my_vram as f64 / 1e9),
                    model_gb: Some(model_bytes as f64 / 1e9),
                });
            }
            // Fall through to normal election loop — each node runs full model independently
        }
    }

    loop {
        if stop_requested(&stop_rx) {
            break;
        }
        prune_skippy_stage_quarantine(&mut skippy_stage_quarantine, Instant::now());
        // Collect our model group (peers also serving this model)
        let peers = node.peers().await;
        let model_peers = model_peers_for_election(&peers, &model_name, &skippy_stage_quarantine);
        let desired_launch = build_dense_launch_plan(
            local_launch_vram,
            model_bytes,
            force_split,
            &model_name,
            &model_peers,
        );

        // Splitting decision: only split when forced OR when the model
        // genuinely doesn't fit on this node alone. If it fits, every
        // node serving this model runs its own independent llama-server
        // (no election needed — everyone is a host).
        let requires_split = force_split || !model_fits_locally;
        let local_stage_quarantined = skippy_stage_quarantine.contains_key(&node.id());

        let i_am_host = if local_stage_quarantined {
            false
        } else if requires_split {
            // Distributed mode: elect one host from the model group using the
            // same advertised node capacity every peer observes through gossip.
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
                emit_info(
                    format!(
                        "[{model_name}] Peer already serving — standby (clients: {n_clients}, requests: {req_count})"
                    ),
                    None,
                );
            } else if force_duplicate_host {
                emit_info(
                    format!("[{model_name}] Forcing duplicate host for benchmark topology"),
                    None,
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
            // Wait for next change or active runtime failure.
            tokio::select! {
                res = peer_rx.changed() => {
                    if res.is_err() { break; }
                    emit_info(
                        "Mesh changed — re-checking... (still host, no restart needed)",
                        Some(format!("model={model_name}")),
                    );
                    tokio::time::sleep(std::time::Duration::from_secs(3)).await;
                    continue;
                }
                failure = async {
                    if let Some(deployment) = skippy_local.as_ref() {
                        deployment.wait_for_failure(&node).await
                    } else if let Some(deployment) = skippy_split.as_ref() {
                        deployment.wait_for_failure(&node).await
                    } else {
                        std::future::pending::<SkippyStageFailure>().await
                    }
                } => {
                    if stop_requested(&stop_rx) || launch::runtime_shutting_down() {
                        break;
                    }
                    if let Some(peer_id) = quarantine_skippy_stage_failure(
                        &mut skippy_stage_quarantine,
                        &failure,
                        Instant::now(),
                    ) {
                        emit_info(
                            format!(
                                "[{model_name}] Temporarily excluding failed stage peer {} from replanning",
                                peer_id.fmt_short()
                            ),
                            Some(format!("stage={}", failure.stage_id)),
                        );
                    }
                    if let Some(deployment) = skippy_split.take() {
                        withdraw_failed_skippy_split(
                            &node,
                            &tunnel_mgr,
                            &target_tx,
                            &model_name,
                            failure,
                            deployment,
                        )
                        .await;
                    }
                    if let Some(deployment) = skippy_local.take() {
                        deployment.shutdown().await;
                    }
                    currently_host = false;
                    current_local_port = None;
                    last_running_plan = None;
                    node.set_role(NodeRole::Worker).await;
                    on_process(None);
                    on_change(false, false);
                    tokio::time::sleep(std::time::Duration::from_secs(2)).await;
                    continue;
                }
                res = stop_rx.changed() => {
                    if res.is_err() || stop_requested(&stop_rx) {
                        break;
                    }
                }
            }
        }

        // Something changed — stop the active runtime if we were hosting.
        if currently_host {
            if let Some(deployment) = skippy_split.take() {
                deployment.shutdown(&node).await;
            }
            if let Some(deployment) = skippy_local.take() {
                deployment.shutdown().await;
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
                    let _ = emit_event(OutputEvent::WaitingForPeers {
                        detail: Some(format!(
                            "[{}] Waiting for more peers — need {:.1}GB capacity, have {:.1}GB across eligible split workers",
                            model_name,
                            *min_vram as f64 / 1e9,
                            *total_group_vram as f64 / 1e9
                        )),
                    });
                    update_targets(&node, &model_name, InferenceTarget::None, &target_tx).await;
                    on_change(false, false);
                    tokio::select! {
                        res = peer_rx.changed() => {
                            if res.is_err() {
                                break;
                            }
                        }
                        _ = tokio::time::sleep(std::time::Duration::from_secs(3)) => {}
                    }
                    tokio::time::sleep(std::time::Duration::from_secs(3)).await;
                    continue;
                }
                DenseLaunchPlan::Split {
                    total_group_vram,
                    worker_ids: _worker_ids,
                } => {
                    let _ = emit_event(OutputEvent::HostElected {
                        model: model_name.clone(),
                        host: node.id().fmt_short().to_string(),
                        role: Some("host".to_string()),
                        capacity_gb: Some(*total_group_vram as f64 / 1e9),
                    });
                }
                DenseLaunchPlan::Solo => {
                    let _ = emit_event(OutputEvent::HostElected {
                        model: model_name.clone(),
                        host: node.id().fmt_short().to_string(),
                        role: Some("host".to_string()),
                        capacity_gb: Some(local_launch_vram as f64 / 1e9),
                    });
                }
            }
            on_change(true, false);

            let split_worker_ids = match &desired_launch {
                DenseLaunchPlan::Split { worker_ids, .. } => Some(worker_ids.clone()),
                _ => None,
            };

            let local_port = if let Some(worker_ids) = split_worker_ids {
                match start_skippy_split(StartSkippySplitParams {
                    node: &node,
                    model: &model,
                    model_name: &model_name,
                    package_info: package_source_info.clone(),
                    explicit_mmproj: explicit_mmproj.as_deref(),
                    model_peers: &model_peers,
                    worker_ids: &worker_ids,
                    ctx_size_override,
                    pinned_gpu: pinned_gpu.as_ref(),
                    slots,
                })
                .await
                {
                    Some(deployment) => {
                        let port = deployment.http_port();
                        skippy_split = Some(deployment);
                        port
                    }
                    None => {
                        on_change(true, false);
                        let _ = peer_rx.changed().await;
                        tokio::time::sleep(std::time::Duration::from_secs(3)).await;
                        continue;
                    }
                }
            } else {
                match start_skippy_local(StartSkippyLocalParams {
                    node: &node,
                    model: &model,
                    model_name: &model_name,
                    explicit_mmproj: explicit_mmproj.as_deref(),
                    ctx_size_override,
                    pinned_gpu: pinned_gpu.as_ref(),
                    slots,
                })
                .await
                {
                    Some(deployment) => {
                        let port = deployment.http_port();
                        skippy_local = Some(deployment);
                        port
                    }
                    None => {
                        on_change(true, false);
                        let _ = peer_rx.changed().await;
                        tokio::time::sleep(std::time::Duration::from_secs(3)).await;
                        continue;
                    }
                }
            };

            node.set_role(NodeRole::Host {
                http_port: ingress_http_port,
            })
            .await;
            tunnel_mgr.set_http_port(local_port);
            currently_host = true;
            current_local_port = Some(local_port);
            last_running_plan = desired_launch.running_plan();
            // Re-gossip so peers learn we're the host for this model
            node.regossip().await;
            update_targets(
                &node,
                &model_name,
                InferenceTarget::Local(local_port),
                &target_tx,
            )
            .await;
            let ctx_size = skippy_split
                .as_ref()
                .map(|deployment| deployment.context_length)
                .or_else(|| {
                    skippy_local
                        .as_ref()
                        .map(|deployment| deployment.context_length)
                })
                .unwrap_or_else(|| ctx_size_override.unwrap_or(4096));
            if skippy_local.is_some() || skippy_split.is_some() {
                on_process(Some(LocalProcessInfo {
                    backend: "skippy".into(),
                    pid: std::process::id(),
                    port: local_port,
                    context_length: ctx_size,
                }));
            }
            emit_ready_events(&model_name, local_port, local_port, ctx_size, None);
            on_change(true, true);
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
                    let _ = emit_event(OutputEvent::WaitingForPeers {
                        detail: Some(format!(
                            "[{}] Worker — host is {} (split mode)",
                            model_name,
                            host.id.fmt_short()
                        )),
                    });
                } else {
                    update_targets(&node, &model_name, InferenceTarget::None, &target_tx).await;
                }
            } else {
                update_targets(&node, &model_name, InferenceTarget::None, &target_tx).await;
            }
            on_change(false, false);
        }

        // Wait for next peer change or active runtime failure.
        let quarantine_recheck = !skippy_stage_quarantine.is_empty();
        tokio::select! {
            res = peer_rx.changed() => {
                if res.is_err() { break; }
                emit_info(
                    "Mesh changed — re-electing...",
                    Some(format!("model={model_name}")),
                );
            }
            failure = async {
                if let Some(deployment) = skippy_local.as_ref() {
                    deployment.wait_for_failure(&node).await
                } else if let Some(deployment) = skippy_split.as_ref() {
                    deployment.wait_for_failure(&node).await
                } else {
                    std::future::pending::<SkippyStageFailure>().await
                }
            } => {
                if stop_requested(&stop_rx) || launch::runtime_shutting_down() {
                    break;
                }
                if let Some(peer_id) = quarantine_skippy_stage_failure(
                    &mut skippy_stage_quarantine,
                    &failure,
                    Instant::now(),
                ) {
                    emit_info(
                        format!(
                            "[{model_name}] Temporarily excluding failed stage peer {} from replanning",
                            peer_id.fmt_short()
                        ),
                        Some(format!("stage={}", failure.stage_id)),
                    );
                }
                if let Some(deployment) = skippy_split.take() {
                    withdraw_failed_skippy_split(
                        &node,
                        &tunnel_mgr,
                        &target_tx,
                        &model_name,
                        failure,
                        deployment,
                    )
                    .await;
                }
                if let Some(deployment) = skippy_local.take() {
                    deployment.shutdown().await;
                }
                currently_host = false;
                current_local_port = None;
                last_running_plan = None;
                node.set_role(NodeRole::Worker).await;
                on_process(None);
                on_change(false, false);
            }
            res = stop_rx.changed() => {
                if res.is_err() || stop_requested(&stop_rx) {
                    break;
                }
            }
            _ = async {
                if !quarantine_recheck {
                    std::future::pending::<()>().await;
                } else {
                    tokio::time::sleep(std::time::Duration::from_secs(3)).await;
                }
            } => {}
        }
        if stop_requested(&stop_rx) {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_secs(3)).await;
    }

    if currently_host {
        if let Some(deployment) = skippy_split.take() {
            deployment.shutdown(&node).await;
        }
        if let Some(deployment) = skippy_local.take() {
            deployment.shutdown().await;
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
        moe_summary,
        my_vram,
        model_bytes,
        binary_flavor,
        ctx_size_override,
        pinned_gpu,
        target_tx,
        mut stop_rx,
        slots,
    } = params;
    let mut peer_rx = node.peer_change_rx.clone();
    let mut currently_running = false;
    let mut last_plan: Option<MoePlacementPlan> = None;
    let mut llama_process: Option<launch::InferenceServerProcess> = None;
    let mut backend_proxy: Option<crate::network::openai::backend::BackendProxyHandle> = None;
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
        let placement_peers: Vec<mesh::PeerInfo> =
            if !currently_running && !model_fits && eligible_model_peers.is_empty() {
                if !declared_model_peers.is_empty() {
                    emit_moe_status(
                        &model_name,
                        "bootstrapping placement",
                        format!(
                            "{} declared peer(s) while active eligibility catches up",
                            declared_model_peers.len()
                        ),
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
            emit_moe_status(
                &model_name,
                "holding recovered peers",
                format!(
                    "{} recovered peer(s) out of active MoE placement until stable",
                    recovering_peer_count
                ),
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
                emit_moe_status(
                    &model_name,
                    "holding reserve peers",
                    format!(
                        "{} healthy peer(s) in reserve for {}s before considering MoE scale-up",
                        healthy_reserve_count, remaining
                    ),
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
                emit_moe_status(
                    &model_name,
                    "holding reserve peers",
                    format!(
                        "{} healthy peer(s) in reserve; the current MoE plan is still preferred",
                        healthy_reserve_count
                    ),
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
            if let Some(proxy) = backend_proxy.take() {
                proxy.shutdown().await;
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
            emit_moe_status(
                &model_name,
                "standing by",
                format!(
                    "outside active MoE placement (leader={} active={} fallback={})",
                    plan.leader_id.fmt_short(),
                    plan.active_ids.len(),
                    plan.fallback_ids.len()
                ),
            );
            node.set_role(NodeRole::Worker).await;
            update_targets(&node, &model_name, InferenceTarget::None, &target_tx).await;
            on_change(false, false);
        } else if matches!(role, MoePlacementRole::FullFallback) {
            emit_moe_status(
                &model_name,
                "full-coverage fallback",
                format!(
                    "leader={} active-shards={} fallback-nodes={}",
                    plan.leader_id.fmt_short(),
                    plan.active_ids.len(),
                    plan.fallback_ids.len()
                ),
            );
            on_change(true, false);

            let llama_port = match find_free_port().await {
                Ok(p) => p,
                Err(e) => {
                    emit_error(
                        format!("Failed to find free port: {e}"),
                        Some(format!("model={model_name} mode=moe-fallback")),
                    );
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
                    my_vram: pinned_gpu
                        .as_ref()
                        .map(|gpu| gpu.vram_bytes)
                        .unwrap_or(my_vram),
                    mmproj: None,
                    ctx_size_override,
                    total_group_vram: None,
                    selected_gpu: pinned_gpu.as_ref(),
                    slots,
                    runtime_data_producer: Some(node.runtime_data_collector().producer(
                        crate::runtime_data::RuntimeDataSource {
                            scope: "runtime",
                            plugin_data_key: None,
                            plugin_endpoint_key: None,
                        },
                    )),
                },
            )
            .await
            {
                Ok(process) => {
                    let proxy = match crate::network::openai::backend::start_backend_proxy(
                        llama_port,
                    )
                    .await
                    {
                        Ok(proxy) => proxy,
                        Err(err) => {
                            emit_error(
                                format!("Failed to start local OpenAI backend proxy: {err}"),
                                Some(format!("model={model_name} port={llama_port}")),
                            );
                            process.handle.shutdown().await;
                            if peer_rx.changed().await.is_err() {
                                break;
                            }
                            tokio::time::sleep(std::time::Duration::from_secs(3)).await;
                            continue;
                        }
                    };
                    let local_proxy_port = proxy.port();
                    backend_proxy = Some(proxy);

                    node.set_role(NodeRole::Host {
                        http_port: ingress_http_port,
                    })
                    .await;
                    tunnel_mgr.set_http_port(local_proxy_port);
                    currently_running = true;
                    current_local_port = Some(local_proxy_port);
                    let ctx_size = process.context_length;
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
                        Some(local_proxy_port),
                        &model_name,
                    );
                    target_tx.send_replace(targets);
                    let lp = runtime.log_path(&format!("llama-server-{}", llama_port));
                    emit_ready_events(
                        &model_name,
                        llama_port,
                        local_proxy_port,
                        ctx_size,
                        Some(lp.display().to_string()),
                    );
                    on_change(true, true);
                }
                Err(e) => {
                    emit_error(
                        format!("Failed to start fallback llama-server: {e}"),
                        Some(format!("model={model_name}")),
                    );
                }
            }
        } else if plan.active_ids.len() == 1 {
            if model_fits {
                node.set_model_runtime_context_length(&model_name, None)
                    .await;
                node.regossip().await;
                let _ = emit_event(OutputEvent::MoeDetected {
                    model: model_name.clone(),
                    moe: moe_summary.clone(),
                    fits_locally: Some(true),
                    capacity_gb: Some(my_vram as f64 / 1e9),
                    model_gb: Some(model_bytes as f64 / 1e9),
                });
                on_change(true, false);

                let llama_port = match find_free_port().await {
                    Ok(p) => p,
                    Err(e) => {
                        emit_error(
                            format!("Failed to find free port: {e}"),
                            Some(format!("model={model_name} mode=moe-solo")),
                        );
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
                        split_mode: split_mode_for_local_launch(binary_flavor, pinned_gpu.as_ref()),
                        draft: None,
                        draft_max: 0,
                        model_bytes: mb,
                        my_vram: pinned_gpu
                            .as_ref()
                            .map(|gpu| gpu.vram_bytes)
                            .unwrap_or(my_vram),
                        mmproj: None,
                        ctx_size_override,
                        total_group_vram: None,
                        selected_gpu: pinned_gpu.as_ref(),
                        slots,
                        runtime_data_producer: Some(node.runtime_data_collector().producer(
                            crate::runtime_data::RuntimeDataSource {
                                scope: "runtime",
                                plugin_data_key: None,
                                plugin_endpoint_key: None,
                            },
                        )),
                    },
                )
                .await
                {
                    Ok(process) => {
                        let proxy =
                            match crate::network::openai::backend::start_backend_proxy(llama_port)
                                .await
                            {
                                Ok(proxy) => proxy,
                                Err(err) => {
                                    emit_error(
                                        format!(
                                            "Failed to start local OpenAI backend proxy: {err}"
                                        ),
                                        Some(format!("model={model_name} port={llama_port}")),
                                    );
                                    process.handle.shutdown().await;
                                    continue;
                                }
                            };
                        let local_proxy_port = proxy.port();
                        backend_proxy = Some(proxy);

                        node.set_role(NodeRole::Host {
                            http_port: ingress_http_port,
                        })
                        .await;
                        tunnel_mgr.set_http_port(local_proxy_port);
                        currently_running = true;
                        current_local_port = Some(local_proxy_port);
                        let ctx_size = process.context_length;
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
                            InferenceTarget::Local(local_proxy_port),
                            &target_tx,
                        )
                        .await;
                        let lp = runtime.log_path(&format!("llama-server-{}", llama_port));
                        emit_ready_events(
                            &model_name,
                            llama_port,
                            local_proxy_port,
                            ctx_size,
                            Some(lp.display().to_string()),
                        );
                        on_change(true, true);
                    }
                    Err(e) => {
                        emit_error(
                            format!("Failed to start llama-server: {e}"),
                            Some(format!(
                                "model={model_name} mode=moe-solo port={llama_port}"
                            )),
                        );
                    }
                }
            } else {
                node.set_model_runtime_context_length(&model_name, None)
                    .await;
                node.regossip().await;
                let _ = emit_event(OutputEvent::MoeDetected {
                    model: model_name.clone(),
                    moe: moe_summary.clone(),
                    fits_locally: Some(false),
                    capacity_gb: Some(my_vram as f64 / 1e9),
                    model_gb: Some(model_bytes as f64 / 1e9),
                });
                on_change(false, false);
            }
        } else {
            let my_shard_index = plan.shard_index_for(my_id).unwrap_or(0);
            on_change(true, false);

            let assignments = moe::compute_assignments_with_overlap(
                &moe_cfg.config.ranking,
                plan.active_ids.len(),
                moe_cfg.config.min_experts_per_node,
                plan.overlap,
            );
            let my_assignment = &assignments[my_shard_index];
            let _ = emit_event(OutputEvent::MoeDistribution {
                model: model_name.clone(),
                moe: moe_summary.clone(),
                distribution: MoeDistributionSummary {
                    leader: plan.leader_id.fmt_short().to_string(),
                    active_nodes: plan.active_ids.len(),
                    fallback_nodes: plan.fallback_ids.len(),
                    shard_index: my_shard_index,
                    shard_count: plan.active_ids.len(),
                    ranking_source: moe_cfg.ranking_source.clone(),
                    ranking_origin: moe_cfg.ranking_origin.clone(),
                    overlap: plan.overlap,
                    shared_experts: my_assignment.n_shared,
                    unique_experts: my_assignment.n_unique,
                },
            });

            // Advertise a non-ready local runtime before split generation / load so
            // peer liveness stays conservative during MoE convergence.
            node.set_model_runtime_starting(&model_name).await;
            node.regossip().await;

            let shard_path = moe::split_path(&model, plan.active_ids.len(), my_shard_index);

            if !shard_path.exists() {
                emit_warning(
                    format!("Splitting GGUF → {} ...", shard_path.display()),
                    Some(format!(
                        "model={model_name} shard={}/{}",
                        my_shard_index + 1,
                        plan.active_ids.len()
                    )),
                );
                match moe::run_split(&bin_dir, &model, my_assignment, &shard_path) {
                    Ok(()) => {
                        let size = std::fs::metadata(&shard_path).map(|m| m.len()).unwrap_or(0);
                        emit_warning(
                            format!("Split complete: {:.1} GB", size as f64 / 1e9),
                            Some(format!(
                                "model={model_name} shard_path={}",
                                shard_path.display()
                            )),
                        );
                    }
                    Err(e) => {
                        emit_error(
                            format!("moe-split failed: {e}"),
                            Some(format!(
                                "model={model_name} shard_path={}",
                                shard_path.display()
                            )),
                        );
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
                emit_moe_status(
                    &model_name,
                    "using cached shard",
                    format!("{} ({:.1} GB)", shard_path.display(), size as f64 / 1e9),
                );
            }

            // Start llama-server with our shard
            let llama_port = match find_free_port().await {
                Ok(p) => p,
                Err(e) => {
                    emit_error(
                        format!("Failed to find free port: {e}"),
                        Some(format!("model={model_name} mode=moe-split")),
                    );
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
                    split_mode: split_mode_for_local_launch(binary_flavor, pinned_gpu.as_ref()),
                    draft: None,
                    draft_max: 0,
                    model_bytes: shard_bytes,
                    my_vram: pinned_gpu
                        .as_ref()
                        .map(|gpu| gpu.vram_bytes)
                        .unwrap_or(my_vram),
                    mmproj: None,
                    ctx_size_override,
                    total_group_vram: None,
                    selected_gpu: pinned_gpu.as_ref(),
                    slots,
                    runtime_data_producer: Some(node.runtime_data_collector().producer(
                        crate::runtime_data::RuntimeDataSource {
                            scope: "runtime",
                            plugin_data_key: None,
                            plugin_endpoint_key: None,
                        },
                    )),
                },
            )
            .await
            {
                Ok(process) => {
                    let proxy = match crate::network::openai::backend::start_backend_proxy(
                        llama_port,
                    )
                    .await
                    {
                        Ok(proxy) => proxy,
                        Err(err) => {
                            emit_error(
                                format!("Failed to start local OpenAI backend proxy: {err}"),
                                Some(format!("model={model_name} port={llama_port}")),
                            );
                            process.handle.shutdown().await;
                            continue;
                        }
                    };
                    let local_proxy_port = proxy.port();
                    backend_proxy = Some(proxy);

                    node.set_role(NodeRole::Host {
                        http_port: ingress_http_port,
                    })
                    .await;
                    tunnel_mgr.set_http_port(local_proxy_port);
                    currently_running = true;
                    current_local_port = Some(local_proxy_port);
                    let ctx_size = process.context_length;
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
                        Some(local_proxy_port),
                        None,
                        &model_name,
                    );
                    target_tx.send_replace(targets);

                    let lp = runtime.log_path(&format!("llama-server-{}", llama_port));
                    emit_ready_events(
                        &model_name,
                        llama_port,
                        local_proxy_port,
                        ctx_size,
                        Some(lp.display().to_string()),
                    );
                    on_change(true, true);
                }
                Err(e) => {
                    emit_error(
                        format!(
                            "MoE split validation failed for shard {}: {e}",
                            shard_path.display()
                        ),
                        Some(format!("model={model_name}")),
                    );
                    emit_warning(
                        "Refusing to enter MoE split mode on this node until the shard validates",
                        Some(format!(
                            "model={model_name} shard_path={}",
                            shard_path.display()
                        )),
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
        emit_moe_status(&model_name, "re-checking deployment", "mesh changed");
        tokio::time::sleep(std::time::Duration::from_secs(3)).await;
    }

    if currently_running {
        if let Some(process) = llama_process.take() {
            process.handle.shutdown().await;
        }
        if let Some(proxy) = backend_proxy.take() {
            proxy.shutdown().await;
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
                emit_info(
                    format!("[{model}] {} hosts available (load balancing)", hosts.len()),
                    None,
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

async fn start_skippy_local(params: StartSkippyLocalParams<'_>) -> Option<SkippyLocalDeployment> {
    let StartSkippyLocalParams {
        node,
        model,
        model_name,
        explicit_mmproj,
        ctx_size_override,
        pinned_gpu,
        slots,
    } = params;
    let context_length = ctx_size_override.unwrap_or(4096);
    let projector_path = crate::models::resolve_mmproj_path(model_name, model, explicit_mmproj)
        .filter(|path| path.exists());
    let mut options = skippy::SkippyModelLoadOptions::for_direct_gguf(model_name, model)
        .with_ctx_size(context_length)
        .with_generation_concurrency(slots.max(1));
    if let Some(projector_path) = projector_path {
        options = options.with_projector_path(projector_path);
    }
    if let Some(gpu) = pinned_gpu {
        options = options.with_selected_device(skippy::SkippyDeviceDescriptor {
            backend_device: gpu.backend_device.clone(),
            stable_id: Some(gpu.stable_id.clone()),
            index: Some(gpu.index),
            vram_bytes: Some(gpu.vram_bytes),
        });
    }

    let model = match skippy::SkippyModelHandle::load_with_hooks(
        options,
        Some(skippy::MeshAutoHookPolicy::new(node.clone())),
    ) {
        Ok(model) => model,
        Err(error) => {
            emit_error(
                format!("Failed to load local model runtime: {error}"),
                Some(format!("model={model_name} path={}", model.display())),
            );
            return None;
        }
    };
    let http_port = match find_free_port().await {
        Ok(port) => port,
        Err(error) => {
            emit_error(
                format!("Failed to find local model HTTP port: {error}"),
                Some(format!("model={model_name}")),
            );
            model.shutdown();
            return None;
        }
    };
    let http = model.start_http(http_port);
    Some(SkippyLocalDeployment {
        context_length,
        model,
        http,
    })
}

async fn start_skippy_split(params: StartSkippySplitParams<'_>) -> Option<SkippySplitDeployment> {
    let StartSkippySplitParams {
        node,
        model,
        model_name,
        package_info,
        explicit_mmproj,
        model_peers,
        worker_ids,
        ctx_size_override,
        pinned_gpu,
        slots,
    } = params;

    let raw_package_ref = model.to_string_lossy().to_string();
    let package_info = match package_info {
        Some(info) => info,
        None => match skippy::StagePackageRef::parse(&raw_package_ref) {
            Ok(parsed) if parsed.is_distributable_package() => {
                let package_ref = parsed.as_package_ref().unwrap_or(raw_package_ref);
                match skippy::inspect_stage_package(&package_ref) {
                    Ok(info) => info,
                    Err(error) => {
                        emit_error(
                            format!("Failed to inspect skippy layer package: {error}"),
                            Some(format!("model={model_name} package={package_ref}")),
                        );
                        return None;
                    }
                }
            }
            Ok(skippy::StagePackageRef::SyntheticDirectGguf(path)) => {
                emit_error(
                    "Skippy staged serving requires a package-backed model source; direct --gguf paths only support local skippy serving for now",
                    Some(format!("model={model_name} path={}", path.display())),
                );
                return None;
            }
            Ok(_) => {
                emit_error(
                    "Skippy staged serving requires a distributable layer package",
                    Some(format!("model={model_name} package={raw_package_ref}")),
                );
                return None;
            }
            Err(error) => {
                emit_error(
                    format!(
                        "Skippy staged serving requires a package-backed model source: {error}"
                    ),
                    Some(format!("model={model_name} path={}", model.display())),
                );
                return None;
            }
        },
    };
    let activation_width = match i32::try_from(package_info.activation_width) {
        Ok(width) if width > 0 => width,
        _ => {
            emit_error(
                "Invalid package activation width for staged serving",
                Some(format!(
                    "model={model_name} activation_width={}",
                    package_info.activation_width
                )),
            );
            return None;
        }
    };
    let ctx_size = ctx_size_override.unwrap_or(4096);
    let mut participants = vec![skippy::StageTopologyParticipant {
        node_id: node.id(),
        vram_bytes: effective_local_launch_vram(
            node.vram_bytes(),
            pinned_gpu,
            None,
            node.gpu_vram.as_deref(),
        ),
    }];
    for worker_id in worker_ids {
        let Some(peer) = model_peers.iter().find(|peer| peer.id == *worker_id) else {
            continue;
        };
        participants.push(skippy::StageTopologyParticipant {
            node_id: peer.id,
            vram_bytes: split_peer_vram_bytes(peer, node.vram_bytes()),
        });
    }
    let run_id = format!("mesh-stage-{}", now_unix_nanos());
    let topology_id = format!("topology-{run_id}");
    let topology_plan =
        match skippy::plan_package_topology(&topology_id, &package_info, &participants) {
            Ok(plan) => plan,
            Err(error) => {
                emit_error(
                    format!("Failed to plan skippy stage topology: {error}"),
                    Some(format!(
                        "model={model_name} package={}",
                        package_info.package_ref
                    )),
                );
                return None;
            }
        };
    if let Some(family_id) = topology_plan.family_id.as_deref() {
        emit_info(
            format!("[{model_name}] skippy topology uses family capability {family_id}"),
            None,
        );
    }
    for diagnostic in &topology_plan.diagnostics {
        emit_info(
            format!("[{model_name}] skippy topology: {diagnostic}"),
            None,
        );
    }
    let plans = topology_plan.stages;
    if plans.len() < 2 {
        emit_error(
            "Staged serving needs at least two non-empty stage ranges",
            Some(format!(
                "model={model_name} layers={}",
                package_info.layer_count
            )),
        );
        return None;
    }
    let context = skippy::StageDeploymentContext {
        topology_id: &topology_id,
        run_id: &run_id,
        model_id: model_name,
        package: &package_info,
        activation_width,
        ctx_size,
        projector_path: crate::models::resolve_mmproj_path(model_name, model, explicit_mmproj)
            .filter(|path| path.exists())
            .map(|path| path.to_string_lossy().to_string()),
    };
    let package_ref = package_info.package_ref.clone();
    let manifest_sha256 = package_info.manifest_sha256.clone();
    let mut remote_stops = Vec::new();
    let mut ready_statuses: HashMap<String, skippy::StageStatusSnapshot> = HashMap::new();

    for index in (1..plans.len()).rev() {
        let plan = &plans[index];
        let downstream = plans.get(index + 1).map(|next| {
            let endpoint = ready_statuses
                .get(&next.stage_id)
                .map(|status| status.bind_addr.clone())
                .unwrap_or_default();
            skippy::StagePeerDescriptor {
                stage_id: next.stage_id.clone(),
                stage_index: next.stage_index,
                endpoint,
                node_id: Some(next.node_id),
            }
        });
        let request = skippy::remote_stage_load_request(&context, plan, downstream);
        emit_info(
            format!(
                "[{model_name}] Loading stage {} on {} layers {}..{}",
                plan.stage_id,
                plan.node_id.fmt_short(),
                plan.layer_start,
                plan.layer_end
            ),
            None,
        );
        match node
            .send_stage_control(plan.node_id, skippy::StageControlRequest::Load(request))
            .await
        {
            Ok(skippy::StageControlResponse::Ready(ready)) if ready.accepted => {
                remote_stops.push((plan.node_id, skippy::stage_stop_request(&context, plan, 2)));
            }
            Ok(skippy::StageControlResponse::Ready(ready)) => {
                emit_error(
                    format!(
                        "Remote stage rejected load: {}",
                        ready.error.unwrap_or_else(|| "unknown error".to_string())
                    ),
                    Some(format!("model={model_name} stage={}", plan.stage_id)),
                );
                cleanup_loaded_remote_stages(node, &remote_stops).await;
                return None;
            }
            Ok(skippy::StageControlResponse::Status(_)) => {
                emit_error(
                    "Remote stage returned unexpected status response to load",
                    Some(format!("model={model_name} stage={}", plan.stage_id)),
                );
                cleanup_loaded_remote_stages(node, &remote_stops).await;
                return None;
            }
            Err(error) => {
                emit_error(
                    format!("Failed to load remote stage: {error}"),
                    Some(format!(
                        "model={model_name} stage={} peer={}",
                        plan.stage_id,
                        plan.node_id.fmt_short()
                    )),
                );
                cleanup_loaded_remote_stages(node, &remote_stops).await;
                return None;
            }
        }
        let Some(status) = wait_for_stage_ready(
            node,
            plan.node_id,
            &topology_id,
            &run_id,
            &plan.stage_id,
            std::time::Duration::from_secs(120),
        )
        .await
        else {
            cleanup_loaded_remote_stages(node, &remote_stops).await;
            return None;
        };
        ready_statuses.insert(plan.stage_id.clone(), status);
    }

    let stage0 = &plans[0];
    let stage1 = &plans[1];
    let stage1_endpoint = match node
        .ensure_stage_transport_bridge(
            stage1.node_id,
            topology_id.clone(),
            run_id.clone(),
            stage1.stage_id.clone(),
        )
        .await
    {
        Ok(endpoint) => endpoint,
        Err(error) => {
            emit_error(
                format!("Failed to create stage transport bridge: {error}"),
                Some(format!("model={model_name} stage={}", stage1.stage_id)),
            );
            cleanup_loaded_remote_stages(node, &remote_stops).await;
            return None;
        }
    };

    let selected_device = skippy::pinned_stage_device(pinned_gpu);
    let stage0_config = skippy::stage0_config(
        &context,
        stage0,
        stage1,
        stage1_endpoint,
        selected_device.clone(),
    );
    let stage0_handle = match skippy::SkippyModelHandle::load_stage0_config(
        stage0_config,
        activation_width,
        slots.max(1),
        256,
        Some(skippy::MeshAutoHookPolicy::new(node.clone())),
    ) {
        Ok(handle) => handle,
        Err(error) => {
            emit_error(
                format!("Failed to load local stage 0: {error}"),
                Some(format!("model={model_name}")),
            );
            cleanup_loaded_remote_stages(node, &remote_stops).await;
            return None;
        }
    };
    let http_port = match find_free_port().await {
        Ok(port) => port,
        Err(error) => {
            emit_error(
                format!("Failed to find local stage 0 HTTP port: {error}"),
                Some(format!("model={model_name}")),
            );
            cleanup_loaded_remote_stages(node, &remote_stops).await;
            return None;
        }
    };
    let http = stage0_handle.start_http(http_port);
    let stage0_bind_addr = format!("127.0.0.1:{http_port}");
    node.record_stage_topology(skippy::stage_topology_instance(
        &context,
        &plans,
        &ready_statuses,
        stage0_bind_addr.clone(),
    ))
    .await;
    let stage0_model_status = stage0_handle.status();
    let stage0_status = skippy::StageStatusSnapshot {
        topology_id: topology_id.clone(),
        run_id: run_id.clone(),
        model_id: model_name.to_string(),
        backend: "skippy".to_string(),
        stage_id: stage0.stage_id.clone(),
        stage_index: stage0.stage_index,
        layer_start: stage0.layer_start,
        layer_end: stage0.layer_end,
        state: skippy::StageRuntimeState::Ready,
        bind_addr: stage0_bind_addr,
        activation_width: activation_width as u32,
        wire_dtype: skippy::StageWireDType::F16,
        selected_device,
        package_ref: Some(package_ref.clone()),
        manifest_sha256: Some(manifest_sha256.clone()),
        source_model_path: Some(package_info.source_model_path.clone()),
        source_model_sha256: Some(package_info.source_model_sha256.clone()),
        source_model_bytes: package_info.source_model_bytes,
        materialized_path: stage0_model_status.materialized_path,
        materialized_pinned: stage0_model_status.materialized_pinned,
        projector_path: stage0_model_status.projector_path,
        ctx_size,
        error: None,
        shutdown_generation: 1,
    };
    node.record_stage_status(Some(node.id()), stage0_status.clone())
        .await;

    Some(SkippySplitDeployment {
        topology_id,
        run_id,
        context_length: ctx_size,
        stage0_status,
        stage0: stage0_handle,
        http,
        remote_stops,
        remote_statuses: ready_statuses
            .into_values()
            .map(|status| {
                let peer_id = plans
                    .iter()
                    .find(|plan| plan.stage_id == status.stage_id)
                    .map(|plan| plan.node_id)
                    .unwrap_or_else(|| node.id());
                (peer_id, status)
            })
            .collect(),
    })
}

async fn cleanup_loaded_remote_stages(
    node: &mesh::Node,
    remote_stops: &[(iroh::EndpointId, skippy::StageStopRequest)],
) {
    for (peer_id, stop) in remote_stops.iter().rev() {
        let _ = node
            .send_stage_control(*peer_id, skippy::StageControlRequest::Stop(stop.clone()))
            .await;
    }
}

async fn withdraw_failed_skippy_split(
    node: &mesh::Node,
    tunnel_mgr: &tunnel::Manager,
    target_tx: &Arc<watch::Sender<ModelTargets>>,
    model_name: &str,
    failure: SkippyStageFailure,
    deployment: SkippySplitDeployment,
) {
    emit_warning(
        format!("stage topology failed — replanning: {}", failure.reason),
        Some(format!(
            "model={model_name} stage={} peer={}",
            failure.stage_id,
            failure
                .peer_id
                .map(|peer_id| peer_id.fmt_short().to_string())
                .unwrap_or_else(|| "unknown".to_string())
        )),
    );
    update_targets(node, model_name, InferenceTarget::None, target_tx).await;
    tunnel_mgr.set_http_port(0);
    deployment
        .shutdown_marking_failure(node, Some(&failure))
        .await;
}

async fn wait_for_stage_ready(
    node: &mesh::Node,
    peer_id: iroh::EndpointId,
    topology_id: &str,
    run_id: &str,
    stage_id: &str,
    timeout: std::time::Duration,
) -> Option<skippy::StageStatusSnapshot> {
    let deadline = tokio::time::Instant::now() + timeout;
    loop {
        let response = node
            .send_stage_control(
                peer_id,
                skippy::StageControlRequest::Status(skippy::StageStatusFilter {
                    topology_id: Some(topology_id.to_string()),
                    run_id: Some(run_id.to_string()),
                    stage_id: Some(stage_id.to_string()),
                }),
            )
            .await;
        match response {
            Ok(skippy::StageControlResponse::Status(statuses)) => {
                if let Some(status) = statuses.into_iter().next() {
                    if status.state == skippy::StageRuntimeState::Ready {
                        return Some(status);
                    }
                    if status.state == skippy::StageRuntimeState::Failed {
                        emit_error(
                            format!(
                                "Stage failed while loading: {}",
                                status.error.unwrap_or_else(|| "unknown error".to_string())
                            ),
                            Some(format!("stage={stage_id}")),
                        );
                        return None;
                    }
                }
            }
            Ok(skippy::StageControlResponse::Ready(ready)) => {
                if ready.status.state == skippy::StageRuntimeState::Ready {
                    return Some(ready.status);
                }
            }
            Err(error) => {
                emit_warning(
                    format!("Stage readiness check failed: {error}"),
                    Some(format!("stage={stage_id}")),
                );
            }
        }
        if tokio::time::Instant::now() >= deadline {
            emit_error(
                "Timed out waiting for stage readiness",
                Some(format!("stage={stage_id}")),
            );
            return None;
        }
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    }
}

async fn find_free_port() -> anyhow::Result<u16> {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let port = listener.local_addr()?.port();
    drop(listener);
    Ok(port)
}

fn now_unix_nanos() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_nanos().min(i64::MAX as u128) as i64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests;
