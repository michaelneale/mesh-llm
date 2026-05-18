//! Local outcome-aware target health for routing decisions.
//!
//! This state is intentionally process-local. It helps the local proxy avoid a
//! target that just timed out or returned unavailable, but it is not a mesh
//! protocol signal and should not be gossiped.

use crate::inference::election::InferenceTarget;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const DEFAULT_BASE_COOLDOWN: Duration = Duration::from_secs(30);
const DEFAULT_MAX_COOLDOWN: Duration = Duration::from_secs(5 * 60);
const DEFAULT_MAX_ENTRIES: usize = 2048;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TargetHealthOutcome {
    Success,
    Timeout,
    Unavailable,
    ContextOverflow,
    Rejected,
    ClientDisconnected,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct TargetKey {
    model: String,
    target: InferenceTarget,
}

#[derive(Clone, Debug)]
struct TargetEntry {
    failures: u32,
    cool_until: Instant,
}

#[derive(Clone, Copy, Debug)]
struct TargetHealthConfig {
    base_cooldown: Duration,
    max_cooldown: Duration,
    max_entries: usize,
}

impl Default for TargetHealthConfig {
    fn default() -> Self {
        Self {
            base_cooldown: DEFAULT_BASE_COOLDOWN,
            max_cooldown: DEFAULT_MAX_COOLDOWN,
            max_entries: DEFAULT_MAX_ENTRIES,
        }
    }
}

#[derive(Default)]
struct TargetHealthState {
    entries: HashMap<TargetKey, TargetEntry>,
    lru: VecDeque<TargetKey>,
    routes_avoided: u64,
}

#[cfg(test)]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct TargetHealthSnapshot {
    pub cooling_targets: usize,
    pub routes_avoided: u64,
}

#[derive(Clone)]
pub(crate) struct TargetHealth {
    inner: Arc<Mutex<TargetHealthState>>,
    config: TargetHealthConfig,
}

impl Default for TargetHealth {
    fn default() -> Self {
        Self::new()
    }
}

impl TargetHealth {
    pub(crate) fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(TargetHealthState::default())),
            config: TargetHealthConfig::default(),
        }
    }

    #[cfg(test)]
    fn with_config(base_cooldown: Duration, max_cooldown: Duration, max_entries: usize) -> Self {
        Self {
            inner: Arc::new(Mutex::new(TargetHealthState::default())),
            config: TargetHealthConfig {
                base_cooldown,
                max_cooldown,
                max_entries,
            },
        }
    }

    pub(crate) fn record_outcome(
        &self,
        model: Option<&str>,
        target: &InferenceTarget,
        outcome: TargetHealthOutcome,
    ) {
        if matches!(target, InferenceTarget::None) {
            return;
        }
        let Some(model) = normalized_model(model) else {
            return;
        };
        let key = TargetKey {
            model,
            target: target.clone(),
        };
        let mut state = self.inner.lock().unwrap();
        state.prune_expired(Instant::now());

        match outcome {
            TargetHealthOutcome::Success => state.remove_key(&key),
            TargetHealthOutcome::Timeout | TargetHealthOutcome::Unavailable => {
                state.record_failure(key, Instant::now(), self.config);
            }
            TargetHealthOutcome::ContextOverflow
            | TargetHealthOutcome::Rejected
            | TargetHealthOutcome::ClientDisconnected => {}
        }
    }

    pub(crate) fn eligible_candidates(
        &self,
        model: &str,
        candidates: &[InferenceTarget],
    ) -> Vec<InferenceTarget> {
        if candidates.len() <= 1 {
            return candidates.to_vec();
        }
        let Some(model) = normalized_model(Some(model)) else {
            return candidates.to_vec();
        };
        let now = Instant::now();
        let mut state = self.inner.lock().unwrap();
        state.prune_expired(now);

        let mut eligible = Vec::with_capacity(candidates.len());
        let mut cooling = 0usize;
        for candidate in candidates {
            let key = TargetKey {
                model: model.clone(),
                target: candidate.clone(),
            };
            if state.is_cooling(&key, now) {
                cooling += 1;
            } else {
                eligible.push(candidate.clone());
            }
        }

        if cooling == 0 || !has_routable_candidate(&eligible) {
            candidates.to_vec()
        } else {
            state.routes_avoided = state.routes_avoided.saturating_add(cooling as u64);
            eligible
        }
    }

    #[cfg(test)]
    pub(crate) fn snapshot(&self) -> TargetHealthSnapshot {
        let mut state = self.inner.lock().unwrap();
        state.prune_expired(Instant::now());
        TargetHealthSnapshot {
            cooling_targets: state.entries.len(),
            routes_avoided: state.routes_avoided,
        }
    }
}

impl TargetHealthState {
    fn record_failure(&mut self, key: TargetKey, now: Instant, config: TargetHealthConfig) {
        let failures = self
            .entries
            .get(&key)
            .map(|entry| entry.failures.saturating_add(1))
            .unwrap_or(1);
        let cooldown = cooldown_for_failure(failures, config);
        self.entries.insert(
            key.clone(),
            TargetEntry {
                failures,
                cool_until: now + cooldown,
            },
        );
        self.touch_key(&key);
        self.prune_over_capacity(config.max_entries);
    }

    fn is_cooling(&self, key: &TargetKey, now: Instant) -> bool {
        self.entries
            .get(key)
            .map(|entry| entry.cool_until > now)
            .unwrap_or(false)
    }

    fn prune_expired(&mut self, now: Instant) {
        let expired: Vec<TargetKey> = self
            .entries
            .iter()
            .filter_map(|(key, entry)| (entry.cool_until <= now).then_some(key.clone()))
            .collect();
        for key in expired {
            self.remove_key(&key);
        }
    }

    fn prune_over_capacity(&mut self, max_entries: usize) {
        while self.entries.len() > max_entries {
            let Some(key) = self.lru.pop_front() else {
                break;
            };
            self.entries.remove(&key);
        }
    }

    fn touch_key(&mut self, key: &TargetKey) {
        self.lru.retain(|existing| existing != key);
        self.lru.push_back(key.clone());
    }

    fn remove_key(&mut self, key: &TargetKey) {
        self.entries.remove(key);
        self.lru.retain(|existing| existing != key);
    }
}

fn normalized_model(model: Option<&str>) -> Option<String> {
    model
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .map(ToOwned::to_owned)
}

fn has_routable_candidate(candidates: &[InferenceTarget]) -> bool {
    candidates
        .iter()
        .any(|candidate| !matches!(candidate, InferenceTarget::None))
}

fn cooldown_for_failure(failures: u32, config: TargetHealthConfig) -> Duration {
    let multiplier = 1u32
        .checked_shl(failures.saturating_sub(1).min(6))
        .unwrap_or(64);
    config
        .base_cooldown
        .saturating_mul(multiplier)
        .min(config.max_cooldown)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn local(port: u16) -> InferenceTarget {
        InferenceTarget::Local(port)
    }

    #[test]
    fn retryable_failure_cools_target_when_alternatives_exist() {
        let health = TargetHealth::default();
        let candidates = vec![local(9001), local(9002)];

        health.record_outcome(Some("qwen"), &local(9001), TargetHealthOutcome::Unavailable);

        assert_eq!(
            health.eligible_candidates("qwen", &candidates),
            vec![local(9002)]
        );
        assert_eq!(
            health.snapshot(),
            TargetHealthSnapshot {
                cooling_targets: 1,
                routes_avoided: 1,
            }
        );
    }

    #[test]
    fn success_clears_target_cooldown() {
        let health = TargetHealth::default();
        let candidates = vec![local(9001), local(9002)];

        health.record_outcome(Some("qwen"), &local(9001), TargetHealthOutcome::Timeout);
        health.record_outcome(Some("qwen"), &local(9001), TargetHealthOutcome::Success);

        assert_eq!(health.eligible_candidates("qwen", &candidates), candidates);
        assert_eq!(health.snapshot().cooling_targets, 0);
    }

    #[test]
    fn context_overflow_and_rejected_do_not_cool_target() {
        let health = TargetHealth::default();
        let candidates = vec![local(9001), local(9002)];

        health.record_outcome(
            Some("qwen"),
            &local(9001),
            TargetHealthOutcome::ContextOverflow,
        );
        health.record_outcome(Some("qwen"), &local(9001), TargetHealthOutcome::Rejected);

        assert_eq!(health.eligible_candidates("qwen", &candidates), candidates);
        assert_eq!(health.snapshot().cooling_targets, 0);
    }

    #[test]
    fn all_cooling_candidates_remain_eligible_to_preserve_availability() {
        let health = TargetHealth::default();
        let candidates = vec![local(9001), local(9002)];

        health.record_outcome(Some("qwen"), &local(9001), TargetHealthOutcome::Timeout);
        health.record_outcome(Some("qwen"), &local(9002), TargetHealthOutcome::Unavailable);

        assert_eq!(health.eligible_candidates("qwen", &candidates), candidates);
        assert_eq!(health.snapshot().cooling_targets, 2);
    }

    #[test]
    fn none_does_not_count_as_a_routable_cooldown_alternative() {
        let health = TargetHealth::default();
        let candidates = vec![local(9001), InferenceTarget::None];

        health.record_outcome(Some("qwen"), &local(9001), TargetHealthOutcome::Timeout);

        assert_eq!(health.eligible_candidates("qwen", &candidates), candidates);
        assert_eq!(
            health.snapshot(),
            TargetHealthSnapshot {
                cooling_targets: 1,
                routes_avoided: 0,
            }
        );
    }

    #[test]
    fn cooldowns_are_scoped_by_model_and_target() {
        let health = TargetHealth::default();
        let candidates = vec![local(9001), local(9002)];

        health.record_outcome(Some("qwen"), &local(9001), TargetHealthOutcome::Timeout);

        assert_eq!(
            health.eligible_candidates("qwen", &candidates),
            vec![local(9002)]
        );
        assert_eq!(health.eligible_candidates("llama", &candidates), candidates);
    }

    #[test]
    fn expired_cooldowns_are_pruned() {
        let health = TargetHealth::with_config(
            Duration::from_millis(0),
            Duration::from_millis(0),
            DEFAULT_MAX_ENTRIES,
        );
        let candidates = vec![local(9001), local(9002)];

        health.record_outcome(Some("qwen"), &local(9001), TargetHealthOutcome::Timeout);

        assert_eq!(health.eligible_candidates("qwen", &candidates), candidates);
        assert_eq!(health.snapshot().cooling_targets, 0);
    }

    #[test]
    fn entry_limit_evicts_oldest_cooldown() {
        let health = TargetHealth::with_config(Duration::from_secs(60), Duration::from_secs(60), 1);
        let candidates = vec![local(9001), local(9002)];

        health.record_outcome(Some("qwen"), &local(9001), TargetHealthOutcome::Timeout);
        health.record_outcome(Some("qwen"), &local(9002), TargetHealthOutcome::Timeout);

        assert_eq!(
            health.eligible_candidates("qwen", &candidates),
            vec![local(9001)]
        );
        assert_eq!(health.snapshot().cooling_targets, 1);
    }
}
