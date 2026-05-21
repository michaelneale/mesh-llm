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
    sorted.sort_by_key(|m| !is_single_digit_b_name(&m.name));
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

/// Truncate `text` so the returned slice is at most `max_bytes` long,
/// honouring UTF-8 char boundaries (never panics, unlike `&text[..N]`).
pub fn truncate_chars(text: &str, max_bytes: usize) -> &str {
    if text.len() <= max_bytes {
        return text;
    }
    let mut idx = max_bytes;
    while idx > 0 && !text.is_char_boundary(idx) {
        idx -= 1;
    }
    &text[..idx]
}

/// Strip `<think>...</think>` tags, return the remaining content.
///
/// Single linear scan over the input: skips think blocks (matched or
/// unclosed) and removes orphan `</think>` closers. The earlier shape
/// rebuilt the whole string on every block (`format!`/`replace` in a
/// loop) which is O(n*k) on long outputs with many tags.
pub fn strip_thinking(text: &str) -> String {
    const OPEN: &str = "<think>";
    const CLOSE: &str = "</think>";
    let bytes = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < bytes.len() {
        // Match a full <think>...</think> block.
        if bytes[i..].starts_with(OPEN.as_bytes()) {
            match text[i + OPEN.len()..].find(CLOSE) {
                Some(rel_end) => {
                    i += OPEN.len() + rel_end + CLOSE.len();
                    continue;
                }
                // Unclosed <think> — drop the rest of the string.
                None => break,
            }
        }
        // Drop an orphan </think> (a closer with no matching opener).
        if bytes[i..].starts_with(CLOSE.as_bytes()) {
            i += CLOSE.len();
            continue;
        }
        // Otherwise copy one char (UTF-8 safe: walk by character).
        let ch = text[i..].chars().next().expect("char boundary");
        out.push(ch);
        i += ch.len_utf8();
    }
    out.trim().to_string()
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
    fn truncate_chars_shorter_than_limit_is_passthrough() {
        assert_eq!(truncate_chars("hello", 100), "hello");
    }

    #[test]
    fn truncate_chars_exact_limit_is_passthrough() {
        assert_eq!(truncate_chars("hello", 5), "hello");
    }

    #[test]
    fn truncate_chars_respects_utf8_boundary() {
        // "café!" is 6 bytes: c a f 0xC3 0xA9 !  (é is 2 bytes).
        let s = "café!";
        assert_eq!(s.len(), 6);
        // Byte 4 is mid-codepoint (between 0xC3 and 0xA9). Naive `&s[..4]`
        // would panic; truncate_chars must walk back to byte 3 ("caf").
        assert_eq!(truncate_chars(s, 4), "caf");
        // Byte 5 IS a valid boundary (between é and !).
        assert_eq!(truncate_chars(s, 5), "café");
        // Within limit ⇒ passthrough.
        assert_eq!(truncate_chars(s, 6), "café!");
    }

    #[test]
    fn truncate_chars_handles_multibyte_only() {
        let s = "日本語"; // each char is 3 bytes ⇒ 9 bytes total
                          // Byte 4 lands mid-char ⇒ walks back to 3 (first char only).
        assert_eq!(truncate_chars(s, 4), "日");
        // Byte 0 is always safe.
        assert_eq!(truncate_chars(s, 0), "");
    }

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

    #[test]
    fn strip_thinking_drops_orphan_close() {
        // Orphan </think> with no matching opener: drop the closer,
        // keep surrounding content.
        assert_eq!(strip_thinking("stuff</think>answer"), "stuffanswer");
    }

    #[test]
    fn strip_thinking_handles_multiple_blocks_in_linear_time() {
        // Regression for PR #566 review item #5b: the previous shape
        // rebuilt the whole string on every think block (`format!` /
        // `replace` in a loop), which is O(n*k). Verify the new linear
        // implementation produces the same output for many blocks.
        let mut input = String::new();
        for i in 0..50 {
            input.push_str(&format!("<think>think-{i}</think>seg{i} "));
        }
        let stripped = strip_thinking(&input);
        let mut expected = String::new();
        for i in 0..50 {
            expected.push_str(&format!("seg{i} "));
        }
        assert_eq!(stripped, expected.trim());
    }

    #[test]
    fn strip_thinking_preserves_utf8() {
        // Multibyte characters outside think blocks must survive intact.
        assert_eq!(strip_thinking("<think>思</think>答案"), "答案");
        assert_eq!(strip_thinking("前置</think>中间<think>隐"), "前置中间");
    }
}
