//! Retry / tool-error budgets. Ported from forge
//! `src/forge/guardrails/error_tracker.py` (v0.6.0).

#[derive(Debug, Clone)]
pub struct ErrorTracker {
    pub max_retries: u32,
    pub max_tool_errors: u32,
    consecutive_retries: u32,
    consecutive_tool_errors: u32,
}

impl ErrorTracker {
    pub fn new(max_retries: u32, max_tool_errors: u32) -> Self {
        Self {
            max_retries,
            max_tool_errors,
            consecutive_retries: 0,
            consecutive_tool_errors: 0,
        }
    }

    /// Record a validation failure (bare text response or unknown tool).
    pub fn record_retry(&mut self) {
        self.consecutive_retries += 1;
    }

    /// Reset retry counter (call on successful validation).
    pub fn reset_retries(&mut self) {
        self.consecutive_retries = 0;
    }

    /// Record the outcome of a tool execution.
    ///
    /// Soft errors (e.g. unresolved tool name) do not count against
    /// the budget. Individual success does NOT reset — only a fully
    /// clean batch does (via [`reset_errors`]).
    pub fn record_result(&mut self, success: bool, is_soft_error: bool) {
        if success {
            return;
        }
        if !is_soft_error {
            self.consecutive_tool_errors += 1;
        }
    }

    /// Reset tool error counter (call after a fully clean batch).
    pub fn reset_errors(&mut self) {
        self.consecutive_tool_errors = 0;
    }

    pub fn retries_exhausted(&self) -> bool {
        self.consecutive_retries > self.max_retries
    }

    pub fn tool_errors_exhausted(&self) -> bool {
        self.consecutive_tool_errors > self.max_tool_errors
    }

    pub fn consecutive_retries(&self) -> u32 {
        self.consecutive_retries
    }

    pub fn consecutive_tool_errors(&self) -> u32 {
        self.consecutive_tool_errors
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retries_exhaust_strictly_greater_than_max() {
        let mut t = ErrorTracker::new(3, 2);
        for _ in 0..3 {
            t.record_retry();
        }
        assert!(
            !t.retries_exhausted(),
            "exactly max_retries is not exhausted"
        );
        t.record_retry();
        assert!(t.retries_exhausted());
    }

    #[test]
    fn reset_clears_retries() {
        let mut t = ErrorTracker::new(3, 2);
        t.record_retry();
        t.record_retry();
        t.reset_retries();
        assert_eq!(t.consecutive_retries(), 0);
    }

    #[test]
    fn soft_errors_do_not_count() {
        let mut t = ErrorTracker::new(3, 2);
        t.record_result(false, true);
        t.record_result(false, true);
        t.record_result(false, true);
        assert_eq!(t.consecutive_tool_errors(), 0);
        assert!(!t.tool_errors_exhausted());
    }

    #[test]
    fn hard_errors_exhaust() {
        let mut t = ErrorTracker::new(3, 2);
        t.record_result(false, false);
        t.record_result(false, false);
        assert!(!t.tool_errors_exhausted());
        t.record_result(false, false);
        assert!(t.tool_errors_exhausted());
    }

    #[test]
    fn success_does_not_reset_errors() {
        let mut t = ErrorTracker::new(3, 2);
        t.record_result(false, false);
        t.record_result(true, false);
        assert_eq!(t.consecutive_tool_errors(), 1);
        t.reset_errors();
        assert_eq!(t.consecutive_tool_errors(), 0);
    }
}
