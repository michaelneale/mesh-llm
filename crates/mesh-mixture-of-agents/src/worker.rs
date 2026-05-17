//! Worker role assignment and text extraction helpers.

use crate::ModelEntry;

/// Worker role determines the context shape and depth.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerRole {
    /// Fast small model — classify, quick proposal.
    Fast,
    /// Specialist — code, domain knowledge.
    Specialist,
    /// Strong reasoner — deeper analysis.
    Strong,
    /// General-purpose worker.
    Generalist,
    /// Reducer/finalizer — only invoked for arbitration.
    Reducer,
}

impl WorkerRole {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Fast => "fast",
            Self::Specialist => "specialist",
            Self::Strong => "strong",
            Self::Generalist => "generalist",
            Self::Reducer => "reducer",
        }
    }
}

/// A worker assignment: which model plays which role.
pub struct Assignment {
    pub model_name: String,
    pub backend_index: usize,
    pub role: WorkerRole,
}

/// Assign roles to models.
///
/// Heuristic: more models = more specialization.
/// With 2: fast + strong.  With 3+: fast + specialist(s) + strong.
pub fn assign_roles(models: &[ModelEntry]) -> Vec<Assignment> {
    if models.is_empty() {
        return vec![];
    }
    if models.len() == 1 {
        return vec![Assignment {
            model_name: models[0].name.clone(),
            backend_index: models[0].backend_index,
            role: WorkerRole::Generalist,
        }];
    }

    // Reorder by capacity tier so role assignment lines up with model
    // capability instead of arbitrary list order:
    //   - "small tier"  = names advertising a single-digit billion-param
    //                     count (1B-9B), e.g. Qwen3-8B, Qwen2.5-3B
    //   - "big tier"    = everything else: multi-digit B (31B, 70B) or
    //                     names that don't encode a size at all
    //                     (MiniMax-M2.5, Coder-Next, fine-tune tags)
    //
    // This mirrors the same heuristic `pick_model_classified` uses in the
    // main router so MoA's reducer/strong worker matches what `auto` would
    // pick.
    let mut sorted: Vec<ModelEntry> = models.to_vec();
    sorted.sort_by_key(|m| is_single_digit_b_name(&m.name) == false);
    // After sort: small-tier first, big-tier last. That way:
    //   first  = fast       (smallest model)
    //   middle = specialist
    //   last   = strong     (biggest model — also used as reducer)

    let mut assignments = Vec::new();

    // First = fast
    assignments.push(Assignment {
        model_name: sorted[0].name.clone(),
        backend_index: sorted[0].backend_index,
        role: WorkerRole::Fast,
    });

    // Middle = specialist(s)
    for m in &sorted[1..sorted.len() - 1] {
        assignments.push(Assignment {
            model_name: m.name.clone(),
            backend_index: m.backend_index,
            role: WorkerRole::Specialist,
        });
    }

    // Last = strong
    let last = sorted.last().unwrap();
    assignments.push(Assignment {
        model_name: last.name.clone(),
        backend_index: last.backend_index,
        role: WorkerRole::Strong,
    });

    assignments
}

/// Return true if `name` advertises a single-digit billion-parameter
/// count, e.g. "Qwen3.5-2B-Q4_K_M" or "llama-3-7b-instruct".
///
/// Accepts: a standalone digit 1-9 immediately followed by `b` or `B`,
/// at a word boundary (not part of a multi-digit number, decimal, or
/// alphanumeric run like "BF16" or "A3B").
///
/// Mirrors `pick_model_classified`'s sizing heuristic in the main
/// router so MoA picks the same "strong" model as `auto` would.
pub(crate) fn is_single_digit_b_name(name: &str) -> bool {
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

/// Strip `<think>...</think>` tags, return the remaining content.
pub fn strip_thinking(text: &str) -> String {
    let mut result = text.to_string();
    while let Some(start) = result.find("<think>") {
        if let Some(end) = result[start..].find("</think>") {
            result = format!(
                "{}{}",
                &result[..start],
                &result[start + end + "</think>".len()..]
            );
        } else {
            result = result[..start].to_string();
            break;
        }
    }
    result = result.replace("</think>", "");
    result.trim().to_string()
}

/// Extract content inside `<think>` tags.
pub fn extract_thinking(text: &str) -> String {
    if let Some(start) = text.find("<think>") {
        let after = &text[start + "<think>".len()..];
        if let Some(end) = after.find("</think>") {
            return after[..end].trim().to_string();
        }
        return after.trim().to_string();
    }
    String::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn assign_two_models() {
        let models = vec![
            ModelEntry {
                name: "small".into(),
                backend_index: 0,
            },
            ModelEntry {
                name: "big".into(),
                backend_index: 1,
            },
        ];
        let assignments = assign_roles(&models);
        assert_eq!(assignments.len(), 2);
        assert_eq!(assignments[0].role, WorkerRole::Fast);
        assert_eq!(assignments[1].role, WorkerRole::Strong);
    }

    #[test]
    fn assign_three_models() {
        let models = vec![
            ModelEntry {
                name: "small".into(),
                backend_index: 0,
            },
            ModelEntry {
                name: "mid".into(),
                backend_index: 1,
            },
            ModelEntry {
                name: "big".into(),
                backend_index: 2,
            },
        ];
        let assignments = assign_roles(&models);
        assert_eq!(assignments.len(), 3);
        assert_eq!(assignments[0].role, WorkerRole::Fast);
        assert_eq!(assignments[1].role, WorkerRole::Specialist);
        assert_eq!(assignments[2].role, WorkerRole::Strong);
    }

    #[test]
    fn assign_roles_sorts_by_size_tier() {
        // 3B is last in list-order, but should NOT end up as Strong —
        // MiniMax (no digit) and Qwen3-32B (multi-digit) belong in the
        // big tier; Qwen2.5-3B and Qwen3-8B belong in the small tier.
        let models = vec![
            ModelEntry {
                name: "MiniMax-M2.5".into(),
                backend_index: 0,
            },
            ModelEntry {
                name: "unsloth/Qwen3-32B-GGUF:Q4_K_M".into(),
                backend_index: 1,
            },
            ModelEntry {
                name: "Qwen3-8B".into(),
                backend_index: 2,
            },
            ModelEntry {
                name: "Qwen2.5-3B".into(),
                backend_index: 3,
            },
        ];
        let assignments = assign_roles(&models);
        assert_eq!(assignments.len(), 4);
        // Fast = a small-tier model (3B or 8B)
        assert_eq!(assignments[0].role, WorkerRole::Fast);
        assert!(
            is_single_digit_b_name(&assignments[0].model_name),
            "fast should be small-tier, got {}",
            assignments[0].model_name
        );
        // Strong = a big-tier model (MiniMax or 32B)
        assert_eq!(assignments[3].role, WorkerRole::Strong);
        assert!(
            !is_single_digit_b_name(&assignments[3].model_name),
            "strong should be big-tier, got {}",
            assignments[3].model_name
        );
    }

    #[test]
    fn size_heuristic_classifies_known_models() {
        // Single-digit B → small tier
        assert!(is_single_digit_b_name("Qwen3-8B"));
        assert!(is_single_digit_b_name("Qwen2.5-3B"));
        assert!(is_single_digit_b_name("Qwen3.5-9B-Q4_K_M"));
        assert!(is_single_digit_b_name("llama-3-7b-instruct"));

        // Multi-digit B → big tier
        assert!(!is_single_digit_b_name("Qwen3-32B"));
        assert!(!is_single_digit_b_name("llama-3-70b"));

        // No size in name → big tier
        assert!(!is_single_digit_b_name("MiniMax-M2.5"));
        assert!(!is_single_digit_b_name("Coder-Next"));

        // Active-params subset (A3B inside larger name) → big tier
        assert!(!is_single_digit_b_name("Qwen3.6-35B-A3B"));

        // BF16-style continuation → not a single-digit-B match
        assert!(!is_single_digit_b_name("model-bf16"));
    }

    #[test]
    fn strip_thinking_tags() {
        assert_eq!(strip_thinking("<think>foo</think>bar"), "bar");
        assert_eq!(
            strip_thinking("before<think>mid</think>after"),
            "beforeafter"
        );
        assert_eq!(strip_thinking("<think>only thinking"), "");
        assert_eq!(strip_thinking("no tags here"), "no tags here");
    }
}
