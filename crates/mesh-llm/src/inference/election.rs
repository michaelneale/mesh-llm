//! Model routing targets shared by runtime and the OpenAI surface.

use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Calculate total model size, summing all split files if present.
/// Split files follow the pattern: name-00001-of-00004.gguf.
pub fn total_model_bytes(model: &Path) -> u64 {
    let name = model.to_string_lossy();
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

/// The current inference target selected by the runtime planner.
/// The OpenAI surface reads this to know where to forward requests.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum InferenceTarget {
    /// No backend running anywhere.
    None,
    /// This node serves the model on the given local HTTP port.
    Local(u16),
    /// Another node serves the model; proxy via QUIC to this peer.
    Remote(iroh::EndpointId),
    /// MoE mode: this node serves an expert shard on the given local HTTP port.
    MoeLocal(u16),
    /// MoE mode: another node serves an expert shard.
    MoeRemote(iroh::EndpointId),
}

/// MoE deployment state shared between planning and proxying.
/// The proxy uses this to route sessions to MoE nodes.
#[derive(Clone, Debug, Default)]
pub struct MoeState {
    /// All MoE node targets, in stable order.
    pub nodes: Vec<InferenceTarget>,
    /// Full-coverage targets that can serve the whole model if the active shard set fails.
    pub fallbacks: Vec<InferenceTarget>,
}

/// Per-model routing table. The API proxy uses this to route by model name.
#[derive(Clone, Debug, Default)]
pub struct ModelTargets {
    /// model_name -> list of inference targets.
    pub targets: HashMap<String, Vec<InferenceTarget>>,
    /// MoE state if this model uses expert sharding.
    pub moe: Option<MoeState>,
    /// Shared round-robin counter across clones.
    counter: Arc<AtomicU64>,
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

    /// Get MoE target for a session using simple hash-based routing.
    pub fn get_moe_target(&self, session_hint: &str) -> Option<InferenceTarget> {
        let moe = self.moe.as_ref()?;
        if moe.nodes.is_empty() {
            return None;
        }
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
