use crate::api::status::ModelTargetCapacityAdviceState;
use crate::mesh::NodeRole;
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ModelTargetReconciliationPolicy {
    pub(crate) enabled: bool,
    pub(crate) max_loads_per_tick: usize,
    pub(crate) failure_cooldown_secs: u64,
    pub(crate) manual_unload_cooldown_secs: u64,
}

impl Default for ModelTargetReconciliationPolicy {
    fn default() -> Self {
        Self {
            enabled: false,
            max_loads_per_tick: 1,
            failure_cooldown_secs: 5 * 60,
            manual_unload_cooldown_secs: 5 * 60,
        }
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct ModelTargetReconciliationState {
    in_flight_model_refs: BTreeSet<String>,
    failed_until_secs: BTreeMap<String, u64>,
    manual_unload_until_secs: BTreeMap<String, u64>,
}

impl ModelTargetReconciliationState {
    pub(crate) fn mark_load_started(&mut self, model_ref: &str) {
        self.in_flight_model_refs.insert(model_ref.to_string());
    }

    pub(crate) fn record_load_success(&mut self, model_ref: &str) {
        self.in_flight_model_refs.remove(model_ref);
        self.failed_until_secs.remove(model_ref);
    }

    pub(crate) fn record_load_failure(
        &mut self,
        model_ref: &str,
        now_secs: u64,
        policy: &ModelTargetReconciliationPolicy,
    ) {
        self.in_flight_model_refs.remove(model_ref);
        if policy.failure_cooldown_secs > 0 {
            self.failed_until_secs.insert(
                model_ref.to_string(),
                now_secs.saturating_add(policy.failure_cooldown_secs),
            );
        }
    }

    pub(crate) fn record_manual_unload(
        &mut self,
        model_ref: &str,
        now_secs: u64,
        policy: &ModelTargetReconciliationPolicy,
    ) {
        self.in_flight_model_refs.remove(model_ref);
        if policy.manual_unload_cooldown_secs > 0 {
            self.manual_unload_until_secs.insert(
                model_ref.to_string(),
                now_secs.saturating_add(policy.manual_unload_cooldown_secs),
            );
        }
    }

    pub(crate) fn prune_expired(&mut self, now_secs: u64) {
        self.failed_until_secs.retain(|_, until| *until > now_secs);
        self.manual_unload_until_secs
            .retain(|_, until| *until > now_secs);
    }

    fn suppressed(&self, model_ref: &str, model_name: Option<&str>, now_secs: u64) -> bool {
        self.in_flight_model_refs.contains(model_ref)
            || self.cooldown_active(&self.failed_until_secs, model_ref, model_name, now_secs)
            || self.cooldown_active(
                &self.manual_unload_until_secs,
                model_ref,
                model_name,
                now_secs,
            )
    }

    fn cooldown_active(
        &self,
        cooldowns: &BTreeMap<String, u64>,
        model_ref: &str,
        model_name: Option<&str>,
        now_secs: u64,
    ) -> bool {
        cooldowns
            .get(model_ref)
            .is_some_and(|until| *until > now_secs)
            || model_name
                .and_then(|name| cooldowns.get(name))
                .is_some_and(|until| *until > now_secs)
    }
}

#[derive(Clone, Debug)]
pub(crate) struct ModelTargetReconciliationInput<'a> {
    pub(crate) now_secs: u64,
    pub(crate) local_role: NodeRole,
    pub(crate) local_interest_model_refs: &'a BTreeSet<String>,
    pub(crate) loaded_model_refs: &'a BTreeSet<String>,
    pub(crate) targets: &'a [ModelTargetReconciliationCandidate],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ModelTargetReconciliationCandidate {
    pub(crate) model_ref: String,
    pub(crate) model_name: Option<String>,
    pub(crate) wanted: bool,
    pub(crate) serving_node_count: usize,
    pub(crate) capacity_state: ModelTargetReconciliationCapacityState,
    pub(crate) local_path: Option<PathBuf>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ModelTargetReconciliationCapacityState {
    AlreadyServing,
    SingleNodeFit,
    SplitCandidate,
    InsufficientCapacity,
    UnknownModelSize,
    UnknownCapacity,
    NoEligibleHosts,
}

impl From<ModelTargetCapacityAdviceState> for ModelTargetReconciliationCapacityState {
    fn from(value: ModelTargetCapacityAdviceState) -> Self {
        match value {
            ModelTargetCapacityAdviceState::AlreadyServing => Self::AlreadyServing,
            ModelTargetCapacityAdviceState::SingleNodeFit => Self::SingleNodeFit,
            ModelTargetCapacityAdviceState::SplitCandidate => Self::SplitCandidate,
            ModelTargetCapacityAdviceState::InsufficientCapacity => Self::InsufficientCapacity,
            ModelTargetCapacityAdviceState::UnknownModelSize => Self::UnknownModelSize,
            ModelTargetCapacityAdviceState::UnknownCapacity => Self::UnknownCapacity,
            ModelTargetCapacityAdviceState::NoEligibleHosts => Self::NoEligibleHosts,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ModelTargetReconciliationAction {
    pub(crate) model_ref: String,
    pub(crate) model_name: Option<String>,
    pub(crate) load_spec: PathBuf,
}

pub(crate) fn plan_model_target_reconciliation(
    policy: &ModelTargetReconciliationPolicy,
    state: &mut ModelTargetReconciliationState,
    input: ModelTargetReconciliationInput<'_>,
) -> Vec<ModelTargetReconciliationAction> {
    state.prune_expired(input.now_secs);
    if !policy.enabled
        || policy.max_loads_per_tick == 0
        || matches!(input.local_role, NodeRole::Client)
    {
        return Vec::new();
    }

    let mut actions = Vec::new();
    for target in input.targets {
        if actions.len() >= policy.max_loads_per_tick {
            break;
        }
        let Some(load_spec) = target.local_path.clone() else {
            continue;
        };
        if !target.wanted
            || target.serving_node_count > 0
            || target.capacity_state != ModelTargetReconciliationCapacityState::SingleNodeFit
            || !input.local_interest_model_refs.contains(&target.model_ref)
            || loaded_target(input.loaded_model_refs, target)
            || state.suppressed(
                &target.model_ref,
                target.model_name.as_deref(),
                input.now_secs,
            )
        {
            continue;
        }

        actions.push(ModelTargetReconciliationAction {
            model_ref: target.model_ref.clone(),
            model_name: target.model_name.clone(),
            load_spec,
        });
    }
    actions
}

fn loaded_target(
    loaded_model_refs: &BTreeSet<String>,
    target: &ModelTargetReconciliationCandidate,
) -> bool {
    loaded_model_refs.contains(&target.model_ref)
        || target
            .model_name
            .as_deref()
            .is_some_and(|name| loaded_model_refs.contains(name))
}

#[cfg(test)]
mod tests {
    use super::*;

    const NOW: u64 = 1_764_000_000;

    fn enabled_policy() -> ModelTargetReconciliationPolicy {
        ModelTargetReconciliationPolicy {
            enabled: true,
            ..ModelTargetReconciliationPolicy::default()
        }
    }

    fn target(model_ref: &str) -> ModelTargetReconciliationCandidate {
        ModelTargetReconciliationCandidate {
            model_ref: model_ref.to_string(),
            model_name: Some("Qwen3-8B-Q4_K_M".to_string()),
            wanted: true,
            serving_node_count: 0,
            capacity_state: ModelTargetReconciliationCapacityState::SingleNodeFit,
            local_path: Some(PathBuf::from("/models/qwen.gguf")),
        }
    }

    fn input<'a>(
        local_interests: &'a BTreeSet<String>,
        loaded: &'a BTreeSet<String>,
        targets: &'a [ModelTargetReconciliationCandidate],
    ) -> ModelTargetReconciliationInput<'a> {
        ModelTargetReconciliationInput {
            now_secs: NOW,
            local_role: NodeRole::Host { http_port: 9337 },
            local_interest_model_refs: local_interests,
            loaded_model_refs: loaded,
            targets,
        }
    }

    #[test]
    fn planner_is_disabled_by_default() {
        let targets = vec![target("org/model@main:file.gguf")];
        let local_interests = BTreeSet::from(["org/model@main:file.gguf".to_string()]);
        let loaded = BTreeSet::new();
        let mut state = ModelTargetReconciliationState::default();

        let actions = plan_model_target_reconciliation(
            &ModelTargetReconciliationPolicy::default(),
            &mut state,
            input(&local_interests, &loaded, &targets),
        );

        assert!(actions.is_empty());
    }

    #[test]
    fn plans_single_local_load_for_wanted_single_node_fit_interest() {
        let targets = vec![target("org/model@main:file.gguf")];
        let local_interests = BTreeSet::from(["org/model@main:file.gguf".to_string()]);
        let loaded = BTreeSet::new();
        let mut state = ModelTargetReconciliationState::default();

        let actions = plan_model_target_reconciliation(
            &enabled_policy(),
            &mut state,
            input(&local_interests, &loaded, &targets),
        );

        assert_eq!(
            actions,
            vec![ModelTargetReconciliationAction {
                model_ref: "org/model@main:file.gguf".to_string(),
                model_name: Some("Qwen3-8B-Q4_K_M".to_string()),
                load_spec: PathBuf::from("/models/qwen.gguf"),
            }]
        );
    }

    #[test]
    fn skips_peer_only_or_requested_targets_without_local_interest() {
        let targets = vec![target("org/model@main:file.gguf")];
        let local_interests = BTreeSet::new();
        let loaded = BTreeSet::new();
        let mut state = ModelTargetReconciliationState::default();

        let actions = plan_model_target_reconciliation(
            &enabled_policy(),
            &mut state,
            input(&local_interests, &loaded, &targets),
        );

        assert!(actions.is_empty());
    }

    #[test]
    fn skips_non_single_node_or_already_available_targets() {
        let mut split = target("org/split@main:file.gguf");
        split.capacity_state = ModelTargetReconciliationCapacityState::SplitCandidate;
        let mut served = target("org/served@main:file.gguf");
        served.serving_node_count = 1;
        let mut missing_path = target("org/missing@main:file.gguf");
        missing_path.local_path = None;
        let targets = vec![split, served, missing_path];
        let local_interests = BTreeSet::from([
            "org/split@main:file.gguf".to_string(),
            "org/served@main:file.gguf".to_string(),
            "org/missing@main:file.gguf".to_string(),
        ]);
        let loaded = BTreeSet::new();
        let mut state = ModelTargetReconciliationState::default();

        let actions = plan_model_target_reconciliation(
            &enabled_policy(),
            &mut state,
            input(&local_interests, &loaded, &targets),
        );

        assert!(actions.is_empty());
    }

    #[test]
    fn cooldowns_and_in_flight_entries_suppress_until_expired() {
        let targets = vec![target("org/model@main:file.gguf")];
        let local_interests = BTreeSet::from(["org/model@main:file.gguf".to_string()]);
        let loaded = BTreeSet::new();
        let policy = enabled_policy();
        let mut state = ModelTargetReconciliationState::default();
        state.record_load_failure("org/model@main:file.gguf", NOW, &policy);

        let actions = plan_model_target_reconciliation(
            &policy,
            &mut state,
            input(&local_interests, &loaded, &targets),
        );
        assert!(actions.is_empty());

        let actions = plan_model_target_reconciliation(
            &policy,
            &mut state,
            ModelTargetReconciliationInput {
                now_secs: NOW + policy.failure_cooldown_secs + 1,
                ..input(&local_interests, &loaded, &targets)
            },
        );
        assert_eq!(actions.len(), 1);
    }

    #[test]
    fn loaded_model_name_suppresses_duplicate_action() {
        let targets = vec![target("org/model@main:file.gguf")];
        let local_interests = BTreeSet::from(["org/model@main:file.gguf".to_string()]);
        let loaded = BTreeSet::from(["Qwen3-8B-Q4_K_M".to_string()]);
        let mut state = ModelTargetReconciliationState::default();

        let actions = plan_model_target_reconciliation(
            &enabled_policy(),
            &mut state,
            input(&local_interests, &loaded, &targets),
        );

        assert!(actions.is_empty());
    }

    #[test]
    fn client_role_never_reconciles_local_loads() {
        let targets = vec![target("org/model@main:file.gguf")];
        let local_interests = BTreeSet::from(["org/model@main:file.gguf".to_string()]);
        let loaded = BTreeSet::new();
        let mut state = ModelTargetReconciliationState::default();

        let actions = plan_model_target_reconciliation(
            &enabled_policy(),
            &mut state,
            ModelTargetReconciliationInput {
                local_role: NodeRole::Client,
                ..input(&local_interests, &loaded, &targets)
            },
        );

        assert!(actions.is_empty());
    }

    #[test]
    fn max_loads_per_tick_caps_eligible_targets() {
        let mut first = target("org/first@main:file.gguf");
        first.model_name = Some("First".to_string());
        let mut second = target("org/second@main:file.gguf");
        second.model_name = Some("Second".to_string());
        let targets = vec![first, second];
        let local_interests = BTreeSet::from([
            "org/first@main:file.gguf".to_string(),
            "org/second@main:file.gguf".to_string(),
        ]);
        let loaded = BTreeSet::new();
        let mut state = ModelTargetReconciliationState::default();

        let actions = plan_model_target_reconciliation(
            &enabled_policy(),
            &mut state,
            input(&local_interests, &loaded, &targets),
        );

        assert_eq!(actions.len(), 1);
        assert_eq!(actions[0].model_ref, "org/first@main:file.gguf");
    }

    #[test]
    fn loaded_model_ref_suppresses_duplicate_action() {
        let targets = vec![target("org/model@main:file.gguf")];
        let local_interests = BTreeSet::from(["org/model@main:file.gguf".to_string()]);
        let loaded = BTreeSet::from(["org/model@main:file.gguf".to_string()]);
        let mut state = ModelTargetReconciliationState::default();

        let actions = plan_model_target_reconciliation(
            &enabled_policy(),
            &mut state,
            input(&local_interests, &loaded, &targets),
        );

        assert!(actions.is_empty());
    }

    #[test]
    fn in_flight_load_suppresses_until_completion() {
        let targets = vec![target("org/model@main:file.gguf")];
        let local_interests = BTreeSet::from(["org/model@main:file.gguf".to_string()]);
        let loaded = BTreeSet::new();
        let policy = enabled_policy();
        let mut state = ModelTargetReconciliationState::default();
        state.mark_load_started("org/model@main:file.gguf");

        let actions = plan_model_target_reconciliation(
            &policy,
            &mut state,
            input(&local_interests, &loaded, &targets),
        );
        assert!(actions.is_empty());

        state.record_load_success("org/model@main:file.gguf");
        let actions = plan_model_target_reconciliation(
            &policy,
            &mut state,
            input(&local_interests, &loaded, &targets),
        );
        assert_eq!(actions.len(), 1);
    }

    #[test]
    fn manual_unload_cooldown_suppresses_by_model_ref_or_name() {
        let targets = vec![target("org/model@main:file.gguf")];
        let local_interests = BTreeSet::from(["org/model@main:file.gguf".to_string()]);
        let loaded = BTreeSet::new();
        let policy = enabled_policy();
        let mut state = ModelTargetReconciliationState::default();
        state.record_manual_unload("Qwen3-8B-Q4_K_M", NOW, &policy);

        let actions = plan_model_target_reconciliation(
            &policy,
            &mut state,
            input(&local_interests, &loaded, &targets),
        );
        assert!(actions.is_empty());

        let actions = plan_model_target_reconciliation(
            &policy,
            &mut state,
            ModelTargetReconciliationInput {
                now_secs: NOW + policy.manual_unload_cooldown_secs + 1,
                ..input(&local_interests, &loaded, &targets)
            },
        );
        assert_eq!(actions.len(), 1);
    }
}
