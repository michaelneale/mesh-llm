//! Ranked model-target aggregation for the management API.
//!
//! This module keeps raw mesh signals separate from the derived ranking and
//! wanted hints that API handlers expose to operators.

use super::{status::ModelTargetPayload, LocalModelInterest, MeshApi};
use crate::mesh;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

#[derive(Clone, Debug)]
struct ModelTargetAccumulator {
    model_ref: String,
    display_name: String,
    model_name: Option<String>,
    explicit_interest_count: usize,
    request_count: u64,
    last_active_secs_ago: Option<u64>,
    serving_node_count: usize,
    requested: bool,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct ModelTargetLookup {
    pub(crate) targets: Vec<ModelTargetPayload>,
    pub(crate) by_model_name: HashMap<String, ModelTargetPayload>,
    pub(crate) by_model_ref: HashMap<String, ModelTargetPayload>,
    pub(crate) wanted_model_refs: Vec<String>,
}

#[derive(Debug, Default)]
struct CatalogTargetIndex {
    canonical_ref_by_model_name: HashMap<String, String>,
    model_name_by_ref: HashMap<String, String>,
    display_name_by_ref: HashMap<String, String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum WantedReason {
    ExplicitInterest,
    ActiveDemand,
    Requested,
}

impl WantedReason {
    const fn as_str(self) -> &'static str {
        match self {
            Self::ExplicitInterest => "explicit_interest",
            Self::ActiveDemand => "active_demand",
            Self::Requested => "requested",
        }
    }
}

impl MeshApi {
    pub(crate) async fn model_targets(&self) -> Vec<ModelTargetPayload> {
        self.model_target_lookup().await.targets
    }

    pub(crate) async fn wanted_model_refs(&self) -> Vec<String> {
        self.model_target_lookup().await.wanted_model_refs
    }

    pub(crate) async fn model_target_lookup(&self) -> ModelTargetLookup {
        let (node, local_interests) = {
            let inner = self.inner.lock().await;
            (
                inner.node.clone(),
                inner
                    .model_interests
                    .values()
                    .cloned()
                    .collect::<Vec<LocalModelInterest>>(),
            )
        };

        let peers = node.peers().await;
        let catalog = node.mesh_catalog_entries().await;
        let active_demand = node.active_demand().await;
        let requested_models = node.requested_models().await;
        let node_explicit_model_interests = node.explicit_model_interests().await;
        let my_hosted_models = node.hosted_models().await;

        build_model_target_lookup(ModelTargetSource {
            local_interests,
            node_explicit_model_interests,
            peers,
            catalog,
            active_demand,
            requested_models,
            my_hosted_models,
            now: current_unix_secs(),
        })
    }
}

struct ModelTargetSource {
    local_interests: Vec<LocalModelInterest>,
    node_explicit_model_interests: Vec<String>,
    peers: Vec<mesh::PeerInfo>,
    catalog: Vec<mesh::MeshCatalogEntry>,
    active_demand: HashMap<String, mesh::ModelDemand>,
    requested_models: Vec<String>,
    my_hosted_models: Vec<String>,
    now: u64,
}

fn build_model_target_lookup(source: ModelTargetSource) -> ModelTargetLookup {
    let index = build_catalog_target_index(&source.catalog);
    let serving_count_by_ref =
        collect_serving_counts(&source.my_hosted_models, &source.peers, &index);
    let mut targets = HashMap::<String, ModelTargetAccumulator>::new();

    apply_explicit_interest_signals(
        &mut targets,
        source.local_interests,
        source.node_explicit_model_interests,
        &source.peers,
        &index,
    );
    apply_active_demand_signals(&mut targets, source.active_demand, source.now, &index);
    apply_requested_model_signals(&mut targets, source.requested_models, &index);
    apply_serving_signals(&mut targets, serving_count_by_ref);

    let mut targets = targets.into_values().collect::<Vec<_>>();
    sort_model_targets(&mut targets);
    let payloads = build_target_payloads(targets);
    build_target_lookup(payloads)
}

fn build_catalog_target_index(catalog: &[mesh::MeshCatalogEntry]) -> CatalogTargetIndex {
    let mut index = CatalogTargetIndex::default();
    for entry in catalog {
        let model_ref = model_ref_for_catalog_entry(entry);
        let display_name = crate::models::installed_model_display_name(&entry.model_name);
        index
            .canonical_ref_by_model_name
            .insert(entry.model_name.clone(), model_ref.clone());
        index
            .model_name_by_ref
            .insert(model_ref.clone(), entry.model_name.clone());
        index
            .model_name_by_ref
            .insert(entry.model_name.clone(), entry.model_name.clone());
        index
            .display_name_by_ref
            .insert(model_ref.clone(), display_name.clone());
        index
            .display_name_by_ref
            .insert(entry.model_name.clone(), display_name);
    }
    index
}

fn collect_serving_counts(
    my_hosted_models: &[String],
    peers: &[mesh::PeerInfo],
    index: &CatalogTargetIndex,
) -> HashMap<String, usize> {
    let mut serving_count_by_ref = HashMap::new();
    for model_name in my_hosted_models {
        record_serving_model(model_name, index, &mut serving_count_by_ref);
    }
    for peer in peers {
        for model_name in peer.http_routable_models() {
            record_serving_model(&model_name, index, &mut serving_count_by_ref);
        }
    }
    serving_count_by_ref
}

fn record_serving_model(
    model_name: &str,
    index: &CatalogTargetIndex,
    serving_count_by_ref: &mut HashMap<String, usize>,
) {
    let model_ref = index
        .canonical_ref_by_model_name
        .get(model_name)
        .cloned()
        .unwrap_or_else(|| model_name.to_string());
    let tracks_canonical_alias = model_ref != model_name;
    *serving_count_by_ref.entry(model_ref).or_insert(0usize) += 1;
    if tracks_canonical_alias {
        *serving_count_by_ref
            .entry(model_name.to_string())
            .or_insert(0usize) += 1;
    }
}

fn apply_explicit_interest_signals(
    targets: &mut HashMap<String, ModelTargetAccumulator>,
    local_interests: Vec<LocalModelInterest>,
    node_explicit_model_interests: Vec<String>,
    peers: &[mesh::PeerInfo],
    index: &CatalogTargetIndex,
) {
    let mut local_explicit_refs = HashSet::new();
    for interest in local_interests {
        let model_ref = interest.model_ref;
        local_explicit_refs.insert(model_ref.clone());
        increment_explicit_interest(targets, model_ref, index);
    }
    for model_ref in node_explicit_model_interests {
        if local_explicit_refs.insert(model_ref.clone()) {
            increment_explicit_interest(targets, model_ref, index);
        }
    }

    for peer in peers {
        let mut peer_interests = HashSet::new();
        for model_ref in &peer.explicit_model_interests {
            if peer_interests.insert(model_ref.clone()) {
                increment_explicit_interest(targets, model_ref.clone(), index);
            }
        }
    }
}

fn increment_explicit_interest(
    targets: &mut HashMap<String, ModelTargetAccumulator>,
    model_ref: String,
    index: &CatalogTargetIndex,
) {
    let model_name = model_name_for_model_ref(&model_ref, index);
    let display_name = display_name_for_model_ref(&model_ref, index);
    ensure_model_target(targets, model_ref, model_name, display_name).explicit_interest_count += 1;
}

fn apply_active_demand_signals(
    targets: &mut HashMap<String, ModelTargetAccumulator>,
    active_demand: HashMap<String, mesh::ModelDemand>,
    now: u64,
    index: &CatalogTargetIndex,
) {
    for (model_name, demand) in active_demand {
        let model_ref = preferred_target_ref_for_model_name(&model_name, index, targets);
        let model_name =
            model_name_for_model_ref(&model_ref, index).or_else(|| Some(model_name.clone()));
        let display_name = display_name_for_model_ref(&model_ref, index);
        let target = ensure_model_target(targets, model_ref, model_name, display_name);
        target.request_count = target.request_count.max(demand.request_count);
        target.last_active_secs_ago = Some(now.saturating_sub(demand.last_active));
    }
}

fn apply_requested_model_signals(
    targets: &mut HashMap<String, ModelTargetAccumulator>,
    requested_models: Vec<String>,
    index: &CatalogTargetIndex,
) {
    for requested_model in requested_models {
        let model_ref = preferred_target_ref_for_model_name(&requested_model, index, targets);
        let model_name =
            model_name_for_model_ref(&model_ref, index).or_else(|| Some(requested_model.clone()));
        let display_name = display_name_for_model_ref(&model_ref, index);
        ensure_model_target(targets, model_ref, model_name, display_name).requested = true;
    }
}

fn apply_serving_signals(
    targets: &mut HashMap<String, ModelTargetAccumulator>,
    serving_count_by_ref: HashMap<String, usize>,
) {
    for target in targets.values_mut() {
        target.serving_node_count = serving_count_by_ref
            .get(&target.model_ref)
            .copied()
            .unwrap_or_default();
    }
}

fn build_target_payloads(targets: Vec<ModelTargetAccumulator>) -> Vec<ModelTargetPayload> {
    targets
        .into_iter()
        .enumerate()
        .map(|(index, target)| {
            let wanted_reason = wanted_reason(&target);
            ModelTargetPayload {
                rank: index + 1,
                model_ref: target.model_ref,
                display_name: target.display_name,
                model_name: target.model_name,
                explicit_interest_count: target.explicit_interest_count,
                request_count: target.request_count,
                last_active_secs_ago: target.last_active_secs_ago,
                serving_node_count: target.serving_node_count,
                requested: target.requested,
                wanted: wanted_reason.is_some(),
                wanted_reason: wanted_reason.map(WantedReason::as_str),
            }
        })
        .collect()
}

fn build_target_lookup(mut payloads: Vec<ModelTargetPayload>) -> ModelTargetLookup {
    let wanted_model_refs = payloads
        .iter()
        .filter(|target| target.wanted)
        .map(|target| target.model_ref.clone())
        .collect::<Vec<_>>();
    let mut by_model_name = HashMap::new();
    let mut by_model_ref = HashMap::new();
    for payload in &payloads {
        by_model_ref.insert(payload.model_ref.clone(), payload.clone());
        if let Some(model_name) = &payload.model_name {
            by_model_name.insert(model_name.clone(), payload.clone());
        }
    }
    payloads.shrink_to_fit();

    ModelTargetLookup {
        targets: payloads,
        by_model_name,
        by_model_ref,
        wanted_model_refs,
    }
}

fn sort_model_targets(targets: &mut [ModelTargetAccumulator]) {
    targets.sort_by(compare_model_targets);
}

fn compare_model_targets(
    left: &ModelTargetAccumulator,
    right: &ModelTargetAccumulator,
) -> Ordering {
    right
        .explicit_interest_count
        .cmp(&left.explicit_interest_count)
        .then_with(|| right.request_count.cmp(&left.request_count))
        .then_with(|| requested_only_priority(right).cmp(&requested_only_priority(left)))
        .then_with(|| {
            left.last_active_secs_ago
                .unwrap_or(u64::MAX)
                .cmp(&right.last_active_secs_ago.unwrap_or(u64::MAX))
        })
        .then_with(|| left.display_name.cmp(&right.display_name))
        .then_with(|| left.model_ref.cmp(&right.model_ref))
}

fn requested_only_priority(target: &ModelTargetAccumulator) -> bool {
    target.serving_node_count == 0
        && target.requested
        && target.explicit_interest_count == 0
        && target.request_count == 0
}

fn wanted_reason(target: &ModelTargetAccumulator) -> Option<WantedReason> {
    if target.serving_node_count > 0 {
        return None;
    }
    if target.explicit_interest_count > 0 {
        return Some(WantedReason::ExplicitInterest);
    }
    if target.request_count > 0 {
        return Some(WantedReason::ActiveDemand);
    }
    if target.requested {
        return Some(WantedReason::Requested);
    }
    None
}

fn model_ref_for_catalog_entry(entry: &mesh::MeshCatalogEntry) -> String {
    entry
        .descriptor
        .as_ref()
        .and_then(|descriptor| descriptor.identity.canonical_ref.clone())
        .unwrap_or_else(|| entry.model_name.clone())
}

fn display_name_for_model_ref(model_ref: &str, index: &CatalogTargetIndex) -> String {
    index
        .display_name_by_ref
        .get(model_ref)
        .cloned()
        .unwrap_or_else(|| crate::models::installed_model_display_name(model_ref))
}

fn model_name_for_model_ref(model_ref: &str, index: &CatalogTargetIndex) -> Option<String> {
    index.model_name_by_ref.get(model_ref).cloned()
}

fn ensure_model_target(
    targets: &mut HashMap<String, ModelTargetAccumulator>,
    model_ref: String,
    model_name: Option<String>,
    display_name: String,
) -> &mut ModelTargetAccumulator {
    targets
        .entry(model_ref.clone())
        .or_insert_with(|| ModelTargetAccumulator {
            model_ref,
            display_name,
            model_name,
            explicit_interest_count: 0,
            request_count: 0,
            last_active_secs_ago: None,
            serving_node_count: 0,
            requested: false,
        })
}

fn preferred_target_ref_for_model_name(
    model_name: &str,
    index: &CatalogTargetIndex,
    targets: &HashMap<String, ModelTargetAccumulator>,
) -> String {
    if targets.contains_key(model_name) {
        return model_name.to_string();
    }

    let canonical_ref = index
        .canonical_ref_by_model_name
        .get(model_name)
        .cloned()
        .unwrap_or_else(|| model_name.to_string());
    if targets.contains_key(&canonical_ref) {
        return canonical_ref;
    }

    canonical_ref
}

fn current_unix_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn target(model_ref: &str) -> ModelTargetAccumulator {
        ModelTargetAccumulator {
            model_ref: model_ref.to_string(),
            display_name: model_ref.to_string(),
            model_name: Some(model_ref.to_string()),
            explicit_interest_count: 0,
            request_count: 0,
            last_active_secs_ago: None,
            serving_node_count: 0,
            requested: false,
        }
    }

    #[test]
    fn requested_signal_does_not_double_count_existing_demand() {
        let mut demand_only = target("a-demand-only");
        demand_only.request_count = 7;

        let mut requested_with_same_demand = target("z-requested-with-demand");
        requested_with_same_demand.request_count = 7;
        requested_with_same_demand.requested = true;

        let mut targets = vec![requested_with_same_demand, demand_only];
        sort_model_targets(&mut targets);

        assert_eq!(targets[0].model_ref, "a-demand-only");
        assert_eq!(targets[1].model_ref, "z-requested-with-demand");
    }

    #[test]
    fn requested_only_signal_ranks_above_inert_targets() {
        let inert = target("a-inert");
        let mut requested = target("z-requested");
        requested.requested = true;

        let mut targets = vec![inert, requested];
        sort_model_targets(&mut targets);

        assert_eq!(targets[0].model_ref, "z-requested");
        assert_eq!(wanted_reason(&targets[0]), Some(WantedReason::Requested));
        assert_eq!(wanted_reason(&targets[1]), None);
    }

    #[test]
    fn served_targets_are_not_wanted_even_with_interest() {
        let mut interested = target("interested-served");
        interested.explicit_interest_count = 3;
        interested.serving_node_count = 1;

        assert_eq!(wanted_reason(&interested), None);
    }
}
