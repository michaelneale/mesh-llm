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

/// The current inference target selected by runtime planning.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum InferenceTarget {
    /// No backend running anywhere.
    None,
    /// This node serves the model on the given local HTTP port.
    Local(u16),
    /// Another node serves the model; proxy via QUIC to this peer.
    Remote(iroh::EndpointId),
}

/// Per-model routing table.
#[derive(Clone, Debug, Default)]
pub struct ModelTargets {
    /// model_name -> list of inference targets.
    pub targets: HashMap<String, Vec<InferenceTarget>>,
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
}

/// Return true if `name` advertises a single-digit billion-parameter
/// count, e.g. `"Qwen3.5-2B-Q4_K_M"` or `"llama-3-7b-instruct"`.
///
/// Accepts: a standalone digit 1-9 immediately followed by `b` or `B`,
/// at a word boundary (not part of a multi-digit number, decimal, or
/// alphanumeric run like `"BF16"` or `"A3B"`).
///
/// This is the "small tier" gate shared by the MoA worker, the
/// guardrails backend decorator, and the main router. Keep one
/// implementation here so they cannot drift.
pub fn is_small_tier_name(name: &str) -> bool {
    let bytes = name.as_bytes();
    for i in 0..bytes.len() {
        let c = bytes[i];
        if !c.is_ascii_digit() {
            continue;
        }
        // Must be a single digit at a word boundary: previous char must
        // not be another digit, a '.', or an ASCII letter.
        if i > 0 {
            let prev = bytes[i - 1];
            if prev.is_ascii_digit() || prev == b'.' || prev.is_ascii_alphabetic() {
                continue;
            }
        }
        // Digit must be 1-9
        if c == b'0' {
            continue;
        }
        // Next byte must be b or B
        let Some(&next) = bytes.get(i + 1) else {
            continue;
        };
        if next != b'b' && next != b'B' {
            continue;
        }
        // Byte after must not be another digit (avoid BF16-like continuations)
        if let Some(&after) = bytes.get(i + 2) {
            if after.is_ascii_digit() {
                continue;
            }
        }
        return true;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::is_small_tier_name;

    #[test]
    fn small_tier_matches_single_digit_b() {
        assert!(is_small_tier_name("Qwen3.5-2B-Q4_K_M"));
        assert!(is_small_tier_name("llama-3-7b-instruct"));
        assert!(is_small_tier_name("Qwen3-8B"));
        assert!(is_small_tier_name("Ministral-3B"));
    }

    #[test]
    fn small_tier_rejects_big_and_unsized() {
        assert!(!is_small_tier_name("MiniMax-M2.5"));
        assert!(!is_small_tier_name("Qwen3-70B"));
        assert!(!is_small_tier_name("Qwen3-31B"));
        assert!(!is_small_tier_name("Coder-Next"));
        // BF16-style continuations must not match.
        assert!(!is_small_tier_name("model-BF16"));
        // A3B-style suffixes (e.g. activation count) must not match.
        assert!(!is_small_tier_name("Qwen3-Coder-480B-A35B-Instruct"));
        // Multi-digit B should not match as small tier.
        assert!(!is_small_tier_name("llama-13b"));
    }
}
