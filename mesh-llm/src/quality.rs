/// Response quality checks — cheap confidence scoring.
///
/// These checks run on completed responses to detect obvious failures
/// that should trigger a retry with a different model.

/// Confidence level from quality checks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Confidence {
    Accept,  // >0.8 — clearly good, return immediately
    Medium,  // 0.4-0.8 — probably fine, could be better
    Weak,    // 0.2-0.4 — likely bad, consider retry
    Reject,  // <0.2 — definitely bad, retry or escalate
}

/// Result of quality checks on a response.
#[derive(Debug)]
pub struct QualityReport {
    pub confidence: Confidence,
    pub issues: Vec<&'static str>,
}

/// Check response quality from cheap signals. No LLM involved.
pub fn check_response(response_text: &str, finish_reason: Option<&str>) -> QualityReport {
    let mut issues = Vec::new();

    // 1. Empty response
    let trimmed = response_text.trim();
    if trimmed.is_empty() {
        return QualityReport {
            confidence: Confidence::Reject,
            issues: vec!["empty response"],
        };
    }

    // 2. Truncated (hit max tokens)
    if finish_reason == Some("length") {
        issues.push("truncated (hit token limit)");
    }

    // 3. Repetition detection — same line/phrase repeated 5+ times
    if detect_repetition(trimmed) {
        issues.push("repetitive output (stuck in loop)");
    }

    // 4. Very short response to a non-trivial question
    // (caller should pass expected_min_length for better checks)
    if trimmed.len() < 10 && finish_reason != Some("stop") {
        issues.push("suspiciously short");
    }

    // 5. Common failure patterns
    if trimmed.contains("I cannot") && trimmed.contains("I'm sorry") && trimmed.len() < 200 {
        issues.push("refusal response");
    }

    // Score
    let confidence = if issues.is_empty() {
        Confidence::Accept
    } else if issues.contains(&"empty response") || issues.contains(&"repetitive output (stuck in loop)") {
        Confidence::Reject
    } else if issues.contains(&"truncated (hit token limit)") {
        Confidence::Medium
    } else if issues.len() >= 2 {
        Confidence::Weak
    } else {
        Confidence::Medium
    };

    QualityReport { confidence, issues }
}

/// Detect if text has excessive repetition (same line repeated 5+ times).
fn detect_repetition(text: &str) -> bool {
    let lines: Vec<&str> = text.lines()
        .map(|l| l.trim())
        .filter(|l| l.len() > 5) // skip trivial lines
        .collect();

    if lines.len() < 5 {
        return false;
    }

    // Check if any line appears 5+ times
    let mut counts = std::collections::HashMap::new();
    for line in &lines {
        let count = counts.entry(*line).or_insert(0u32);
        *count += 1;
        if *count >= 5 {
            return true;
        }
    }

    // Check for repeating pattern (e.g., "AB AB AB AB AB")
    // Look at last 10 lines — if 80%+ are identical, it's a loop
    let tail: Vec<&str> = lines.iter().rev().take(10).copied().collect();
    if tail.len() >= 5 {
        let most_common = tail.iter()
            .max_by_key(|l| tail.iter().filter(|t| t == l).count())
            .unwrap();
        let count = tail.iter().filter(|l| *l == most_common).count();
        if count as f64 / tail.len() as f64 >= 0.8 {
            return true;
        }
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_good_response() {
        let r = check_response("Paris is the capital of France.", Some("stop"));
        assert_eq!(r.confidence, Confidence::Accept);
        assert!(r.issues.is_empty());
    }

    #[test]
    fn test_empty_response() {
        let r = check_response("", Some("stop"));
        assert_eq!(r.confidence, Confidence::Reject);
    }

    #[test]
    fn test_truncated() {
        let r = check_response("This is a long response that got cut", Some("length"));
        assert_eq!(r.confidence, Confidence::Medium);
        assert!(r.issues.contains(&"truncated (hit token limit)"));
    }

    #[test]
    fn test_repetitive() {
        let repeated = "The answer is 42.\n".repeat(10);
        let r = check_response(&repeated, Some("stop"));
        assert_eq!(r.confidence, Confidence::Reject);
        assert!(r.issues.contains(&"repetitive output (stuck in loop)"));
    }

    #[test]
    fn test_whitespace_only() {
        let r = check_response("   \n\n  ", Some("stop"));
        assert_eq!(r.confidence, Confidence::Reject);
    }

    #[test]
    fn test_normal_code_response() {
        let code = "```python\ndef binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1\n```";
        let r = check_response(code, Some("stop"));
        assert_eq!(r.confidence, Confidence::Accept);
    }

    #[test]
    fn test_refusal() {
        let r = check_response("I'm sorry, I cannot help with that request.", Some("stop"));
        assert_eq!(r.confidence, Confidence::Medium);
    }
}
