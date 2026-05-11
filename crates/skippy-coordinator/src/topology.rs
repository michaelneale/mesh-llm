use std::cmp::Ordering;

const MAX_AUTO_PARALLEL_LANES: usize = 16;
const CONTEXT_STEPS: &[u32] = &[512, 1024, 2048, 4096, 8192, 16_384, 32_768, 65_536, 131_072];

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TopologyPlanningInput {
    pub native_context_length: u32,
    pub layer_count: u32,
    pub model_weight_bytes: u64,
    pub kv_bytes_per_token: u64,
    pub minimum_nodes: usize,
    pub nodes: Vec<TopologyNode>,
    pub context_length_override: Option<u32>,
    pub parallel_lanes_override: Option<usize>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TopologyNode {
    pub node_id: String,
    pub detected_vram_bytes: u64,
    pub max_vram_bytes: Option<u64>,
    pub runtime_headroom_bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TopologyPlan {
    pub context_length: u32,
    pub parallel_lanes: usize,
    pub stages: Vec<TopologyStagePlan>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TopologyStagePlan {
    pub stage_id: String,
    pub stage_index: u32,
    pub node_id: String,
    pub layer_start: u32,
    pub layer_end: u32,
    pub parameter_bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum TopologyPlanError {
    #[error("topology planning requires native GGUF context length")]
    MissingNativeContext,
    #[error("topology planning requires at least one model layer")]
    MissingLayers,
    #[error("topology planning requires model weight bytes")]
    MissingModelWeights,
    #[error("topology planning requires KV bytes per token")]
    MissingKvBytesPerToken,
    #[error("topology planning requires at least one node")]
    MissingNodes,
    #[error("requested context {requested} is below minimum valid context {minimum}")]
    ContextBelowMinimum { requested: u32, minimum: u32 },
    #[error("requested context {requested} exceeds native context {native}")]
    ContextExceedsNative { requested: u32, native: u32 },
    #[error("requested parallel lanes must be greater than zero")]
    ZeroParallelLanes,
    #[error("no topology can distribute all layers and keep context >= {minimum_context}")]
    NoValidTopology { minimum_context: u32 },
}

pub fn plan_topology(input: &TopologyPlanningInput) -> Result<TopologyPlan, TopologyPlanError> {
    validate_input(input)?;

    let minimum_context = minimum_valid_context(input.native_context_length);
    let context_candidates = context_candidates(
        input.native_context_length,
        minimum_context,
        input.context_length_override,
    )?;
    let lane_candidates = parallel_lane_candidates(input.parallel_lanes_override)?;
    let nodes = usable_nodes(&input.nodes);

    for context_length in context_candidates {
        for parallel_lanes in lane_candidates.iter().copied() {
            let minimum_nodes = input.minimum_nodes.max(1);
            for node_count in minimum_nodes..=nodes.len().min(input.layer_count as usize) {
                let mut best_for_count: Option<CandidatePlan> = None;
                for subset in node_subsets(&nodes, node_count) {
                    if let Some(candidate) =
                        fit_candidate(input, &subset, context_length, parallel_lanes)
                    {
                        if best_for_count
                            .as_ref()
                            .is_none_or(|current| candidate.cmp(current) == Ordering::Greater)
                        {
                            best_for_count = Some(candidate);
                        }
                    }
                }
                if let Some(candidate) = best_for_count {
                    return Ok(candidate.plan);
                }
            }
        }
    }

    Err(TopologyPlanError::NoValidTopology { minimum_context })
}

fn validate_input(input: &TopologyPlanningInput) -> Result<(), TopologyPlanError> {
    if input.native_context_length == 0 {
        return Err(TopologyPlanError::MissingNativeContext);
    }
    if input.layer_count == 0 {
        return Err(TopologyPlanError::MissingLayers);
    }
    if input.model_weight_bytes == 0 {
        return Err(TopologyPlanError::MissingModelWeights);
    }
    if input.kv_bytes_per_token == 0 {
        return Err(TopologyPlanError::MissingKvBytesPerToken);
    }
    if input.nodes.is_empty() {
        return Err(TopologyPlanError::MissingNodes);
    }
    Ok(())
}

fn context_candidates(
    native_context: u32,
    minimum_context: u32,
    override_context: Option<u32>,
) -> Result<Vec<u32>, TopologyPlanError> {
    if let Some(requested) = override_context {
        if requested > native_context {
            return Err(TopologyPlanError::ContextExceedsNative {
                requested,
                native: native_context,
            });
        }
        if requested < minimum_context {
            return Err(TopologyPlanError::ContextBelowMinimum {
                requested,
                minimum: minimum_context,
            });
        }
        return Ok(vec![requested]);
    }

    let mut candidates = CONTEXT_STEPS
        .iter()
        .copied()
        .filter(|context| *context >= minimum_context && *context <= native_context)
        .collect::<Vec<_>>();
    candidates.push(native_context);
    candidates.sort_unstable();
    candidates.dedup();
    candidates.reverse();
    Ok(candidates)
}

fn parallel_lane_candidates(
    override_lanes: Option<usize>,
) -> Result<Vec<usize>, TopologyPlanError> {
    if let Some(lanes) = override_lanes {
        if lanes == 0 {
            return Err(TopologyPlanError::ZeroParallelLanes);
        }
        return Ok(vec![lanes]);
    }
    Ok((1..=MAX_AUTO_PARALLEL_LANES).rev().collect())
}

fn minimum_valid_context(native_context: u32) -> u32 {
    native_context.div_ceil(2).max(1)
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct UsableNode {
    node_id: String,
    usable_vram_bytes: u64,
}

fn usable_nodes(nodes: &[TopologyNode]) -> Vec<UsableNode> {
    let mut nodes = nodes
        .iter()
        .map(|node| {
            let capped = node
                .max_vram_bytes
                .map(|max| node.detected_vram_bytes.min(max))
                .unwrap_or(node.detected_vram_bytes);
            UsableNode {
                node_id: node.node_id.clone(),
                usable_vram_bytes: capped.saturating_sub(node.runtime_headroom_bytes),
            }
        })
        .collect::<Vec<_>>();
    nodes.sort_by(|left, right| {
        right
            .usable_vram_bytes
            .cmp(&left.usable_vram_bytes)
            .then_with(|| left.node_id.cmp(&right.node_id))
    });
    nodes
}

fn node_subsets(nodes: &[UsableNode], count: usize) -> Vec<Vec<UsableNode>> {
    let mut subsets = Vec::new();
    let mut current = Vec::with_capacity(count);
    collect_node_subsets(nodes, count, 0, &mut current, &mut subsets);
    subsets
}

fn collect_node_subsets(
    nodes: &[UsableNode],
    count: usize,
    start: usize,
    current: &mut Vec<UsableNode>,
    subsets: &mut Vec<Vec<UsableNode>>,
) {
    if current.len() == count {
        subsets.push(current.clone());
        return;
    }
    let needed = count - current.len();
    if nodes.len().saturating_sub(start) < needed {
        return;
    }
    for index in start..=nodes.len() - needed {
        current.push(nodes[index].clone());
        collect_node_subsets(nodes, count, index + 1, current, subsets);
        current.pop();
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct CandidatePlan {
    plan: TopologyPlan,
    minimum_remaining_vram: u64,
    total_remaining_vram: u128,
}

impl Ord for CandidatePlan {
    fn cmp(&self, other: &Self) -> Ordering {
        self.minimum_remaining_vram
            .cmp(&other.minimum_remaining_vram)
            .then_with(|| self.total_remaining_vram.cmp(&other.total_remaining_vram))
            .then_with(|| {
                let left = self
                    .plan
                    .stages
                    .iter()
                    .map(|stage| stage.node_id.as_str())
                    .collect::<Vec<_>>();
                let right = other
                    .plan
                    .stages
                    .iter()
                    .map(|stage| stage.node_id.as_str())
                    .collect::<Vec<_>>();
                right.cmp(&left)
            })
    }
}

impl PartialOrd for CandidatePlan {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn fit_candidate(
    input: &TopologyPlanningInput,
    nodes: &[UsableNode],
    context_length: u32,
    parallel_lanes: usize,
) -> Option<CandidatePlan> {
    let layer_count = input.layer_count as usize;
    if nodes.len() > layer_count {
        return None;
    }

    let weight_per_layer = input
        .model_weight_bytes
        .div_ceil(u64::from(input.layer_count));
    let kv_per_layer = input
        .kv_bytes_per_token
        .div_ceil(u64::from(input.layer_count));
    let bytes_per_layer = candidate_bytes_per_layer(
        weight_per_layer,
        kv_per_layer,
        context_length,
        parallel_lanes,
    )?;

    let mut capacities = nodes
        .iter()
        .map(|node| {
            let max_layers = node.usable_vram_bytes / bytes_per_layer;
            (
                node.clone(),
                max_layers.min(u64::from(input.layer_count)) as usize,
            )
        })
        .collect::<Vec<_>>();
    capacities.sort_by(|(left_node, left_layers), (right_node, right_layers)| {
        right_layers
            .cmp(left_layers)
            .then_with(|| {
                right_node
                    .usable_vram_bytes
                    .cmp(&left_node.usable_vram_bytes)
            })
            .then_with(|| left_node.node_id.cmp(&right_node.node_id))
    });

    if capacities.iter().any(|(_, max_layers)| *max_layers == 0) {
        return None;
    }
    if capacities
        .iter()
        .map(|(_, max_layers)| *max_layers)
        .sum::<usize>()
        < layer_count
    {
        return None;
    }

    let mut next_layer = 0u32;
    let mut stages = Vec::with_capacity(capacities.len());
    let mut minimum_remaining_vram = u64::MAX;
    let mut total_remaining_vram = 0u128;

    for (stage_index, (node, max_layers)) in capacities.iter().enumerate() {
        let remaining_layers = input.layer_count - next_layer;
        let remaining_nodes = capacities.len() - stage_index;
        let min_for_later = remaining_nodes.saturating_sub(1) as u32;
        let assignable = remaining_layers.saturating_sub(min_for_later);
        let layer_span = assignable.min(*max_layers as u32);
        if layer_span == 0 {
            return None;
        }

        let layer_start = next_layer;
        let layer_end = layer_start + layer_span;
        let parameter_bytes = u64::from(layer_span).saturating_mul(weight_per_layer);
        let required_bytes = u64::from(layer_span).saturating_mul(bytes_per_layer);
        if required_bytes > node.usable_vram_bytes {
            return None;
        }
        let remaining = node.usable_vram_bytes - required_bytes;
        minimum_remaining_vram = minimum_remaining_vram.min(remaining);
        total_remaining_vram += u128::from(remaining);
        stages.push(TopologyStagePlan {
            stage_id: format!("stage-{stage_index}"),
            stage_index: stage_index as u32,
            node_id: node.node_id.clone(),
            layer_start,
            layer_end,
            parameter_bytes,
        });
        next_layer = layer_end;
    }

    (next_layer == input.layer_count).then_some(CandidatePlan {
        plan: TopologyPlan {
            context_length,
            parallel_lanes,
            stages,
        },
        minimum_remaining_vram,
        total_remaining_vram,
    })
}

fn candidate_bytes_per_layer(
    weight_per_layer: u64,
    kv_per_layer: u64,
    context_length: u32,
    parallel_lanes: usize,
) -> Option<u64> {
    let kv_bytes = u128::from(kv_per_layer)
        .checked_mul(u128::from(context_length))?
        .checked_mul(parallel_lanes as u128)?;
    let total = u128::from(weight_per_layer).checked_add(kv_bytes)?;
    total.try_into().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1024 * 1024 * 1024;
    const QWEN_CODER_480B_NATIVE_CONTEXT: u32 = 262_144;
    const QWEN_CODER_480B_LAYERS: u32 = 62;
    const QWEN_CODER_480B_WEIGHT_BYTES: u64 = 315_680_000_000;
    const QWEN_CODER_480B_Q8_KV_BYTES_PER_TOKEN: u64 = 128 * 1024;

    fn node(id: &str, gib: u64) -> TopologyNode {
        TopologyNode {
            node_id: id.to_string(),
            detected_vram_bytes: gib * GIB,
            max_vram_bytes: None,
            runtime_headroom_bytes: 0,
        }
    }

    fn input(nodes: Vec<TopologyNode>) -> TopologyPlanningInput {
        TopologyPlanningInput {
            native_context_length: 65_536,
            layer_count: 40,
            model_weight_bytes: 40 * GIB,
            kv_bytes_per_token: 64 * 1024,
            minimum_nodes: 1,
            nodes,
            context_length_override: None,
            parallel_lanes_override: None,
        }
    }

    fn qwen_coder_480b_input(nodes: Vec<TopologyNode>) -> TopologyPlanningInput {
        TopologyPlanningInput {
            native_context_length: QWEN_CODER_480B_NATIVE_CONTEXT,
            layer_count: QWEN_CODER_480B_LAYERS,
            model_weight_bytes: QWEN_CODER_480B_WEIGHT_BYTES,
            kv_bytes_per_token: QWEN_CODER_480B_Q8_KV_BYTES_PER_TOKEN,
            minimum_nodes: 2,
            nodes,
            context_length_override: None,
            parallel_lanes_override: None,
        }
    }

    fn qwen_node(index: usize, gib: u64) -> TopologyNode {
        node(&format!("qwen-node-{index:02}"), gib)
    }

    fn qwen_nodes(count: usize, gib: u64) -> Vec<TopologyNode> {
        (0..count).map(|index| qwen_node(index, gib)).collect()
    }

    #[test]
    fn chooses_highest_context_then_parallelism() {
        let plan = plan_topology(&input(vec![node("a", 23), node("b", 23)])).unwrap();

        assert_eq!(plan.context_length, 65_536);
        assert_eq!(plan.parallel_lanes, 1);
        assert_eq!(plan.stages.len(), 2);
    }

    #[test]
    fn uses_minimum_nodes_for_same_context_and_lanes() {
        let plan = plan_topology(&input(vec![
            node("a", 80),
            node("b", 80),
            node("c", 80),
            node("d", 80),
            node("e", 80),
            node("f", 80),
        ]))
        .unwrap();

        assert_eq!(plan.context_length, 65_536);
        assert_eq!(plan.parallel_lanes, 16);
        assert_eq!(plan.stages.len(), 2);
    }

    #[test]
    fn assigns_fewer_layers_to_lower_vram_node() {
        let plan = plan_topology(&input(vec![node("small", 16), node("large", 48)])).unwrap();

        assert_eq!(plan.context_length, 65_536);
        let small = plan
            .stages
            .iter()
            .find(|stage| stage.node_id == "small")
            .unwrap();
        let large = plan
            .stages
            .iter()
            .find(|stage| stage.node_id == "large")
            .unwrap();
        assert!(small.layer_end - small.layer_start < large.layer_end - large.layer_start);
    }

    #[test]
    fn applies_max_vram_and_runtime_headroom_per_node() {
        let mut capped = node("capped", 80);
        capped.max_vram_bytes = Some(24 * GIB);
        capped.runtime_headroom_bytes = 8 * GIB;
        let plan = plan_topology(&input(vec![capped, node("peer", 48)])).unwrap();

        let capped_stage = plan
            .stages
            .iter()
            .find(|stage| stage.node_id == "capped")
            .unwrap();
        assert!(capped_stage.layer_end - capped_stage.layer_start < 20);
    }

    #[test]
    fn rejects_below_half_native_context() {
        let err = plan_topology(&input(vec![node("tiny-a", 8), node("tiny-b", 8)]))
            .expect_err("context below half native should be rejected");

        assert_eq!(
            err,
            TopologyPlanError::NoValidTopology {
                minimum_context: 32_768
            }
        );
    }

    #[test]
    fn rejects_context_override_above_native() {
        let mut request = input(vec![node("a", 80)]);
        request.context_length_override = Some(131_072);

        assert_eq!(
            plan_topology(&request),
            Err(TopologyPlanError::ContextExceedsNative {
                requested: 131_072,
                native: 65_536,
            })
        );
    }

    #[test]
    fn qwen_coder_480b_rejects_when_layers_cannot_fit_above_half_context() {
        // Simulation: 4 x 70 GiB nodes.
        //
        // Expected topology: none.
        //
        // Why: the planner may degrade context only to half native
        // (131_072). At this machine size the full 62-layer package plus
        // half-context KV cannot be distributed, so launching would silently
        // produce an under-resourced split.
        let err = plan_topology(&qwen_coder_480b_input(qwen_nodes(4, 70)))
            .expect_err("four 70 GiB nodes cannot hold this layer package above half context");

        assert_eq!(
            err,
            TopologyPlanError::NoValidTopology {
                minimum_context: 131_072
            }
        );
    }

    #[test]
    fn qwen_coder_480b_uses_half_context_when_native_context_does_not_fit() {
        // Simulation: 4 x 82 GiB nodes.
        //
        // Expected topology: 4 stages, 131_072 context, 1 lane.
        //
        // Why: native 262_144 context does not fit across these nodes, but
        // exactly half native does. The planner protects context first, so it
        // keeps the highest valid context and accepts single-lane serving.
        let plan = plan_topology(&qwen_coder_480b_input(qwen_nodes(4, 82))).unwrap();

        assert_eq!(plan.context_length, 131_072);
        assert_eq!(plan.parallel_lanes, 1);
        assert_eq!(plan.stages.len(), 4);
        assert_eq!(plan.stages.first().unwrap().layer_start, 0);
        assert_eq!(
            plan.stages.last().unwrap().layer_end,
            QWEN_CODER_480B_LAYERS
        );
    }

    #[test]
    fn qwen_coder_480b_prefers_native_context_then_parallelism() {
        // Simulation: 5 x 80 GiB nodes.
        //
        // Expected topology: 5 stages, native 262_144 context, 2 lanes.
        //
        // Why: adding the fifth node makes native context fit. After native
        // context is preserved, the planner spends remaining memory on
        // parallel lanes and finds that two lanes fit but three do not.
        let plan = plan_topology(&qwen_coder_480b_input(qwen_nodes(5, 80))).unwrap();

        assert_eq!(plan.context_length, QWEN_CODER_480B_NATIVE_CONTEXT);
        assert_eq!(plan.parallel_lanes, 2);
        assert_eq!(plan.stages.len(), 5);
    }

    #[test]
    fn qwen_coder_480b_uses_extra_nodes_only_when_they_increase_lanes() {
        // Simulation: 10 x 80 GiB nodes.
        //
        // Expected topology: 9 stages, native 262_144 context, 12 lanes.
        //
        // Why: context is already at native, so the planner maximizes lanes
        // next. Twelve lanes require nine of these nodes; the tenth node does
        // not improve context or lanes and is therefore omitted for speed.
        let plan = plan_topology(&qwen_coder_480b_input(qwen_nodes(10, 80))).unwrap();

        assert_eq!(plan.context_length, QWEN_CODER_480B_NATIVE_CONTEXT);
        assert_eq!(plan.parallel_lanes, 12);
        assert_eq!(plan.stages.len(), 9);
    }

    #[test]
    fn qwen_coder_480b_does_not_use_extra_nodes_for_the_same_shape() {
        // Simulation: 7 x 80 GiB nodes plus 3 x 1 GiB bystanders.
        //
        // Expected topology: 7 stages, native 262_144 context, 8 lanes.
        //
        // Why: the small bystander nodes cannot carry even one useful layer
        // at the chosen context/lane shape. The planner keeps them out rather
        // than adding split boundaries that would reduce decode speed.
        let mut nodes = qwen_nodes(7, 80);
        nodes.extend((7..10).map(|index| qwen_node(index, 1)));
        let plan = plan_topology(&qwen_coder_480b_input(nodes)).unwrap();

        assert_eq!(plan.context_length, QWEN_CODER_480B_NATIVE_CONTEXT);
        assert_eq!(plan.parallel_lanes, 8);
        assert_eq!(plan.stages.len(), 7);
        assert!(plan
            .stages
            .iter()
            .all(|stage| !stage.node_id.ends_with("07")
                && !stage.node_id.ends_with("08")
                && !stage.node_id.ends_with("09")));
    }

    #[test]
    fn qwen_coder_480b_assigns_less_work_to_smaller_nodes() {
        // Simulation: 1 x 64 GiB node and 5 x 80 GiB nodes.
        //
        // Expected topology: native context with the 64 GiB node assigned
        // fewer layers than the largest stage.
        //
        // Why: KV and weights are layer-local. Assigning fewer layers to the
        // smaller node prevents it from forcing down the cluster-wide context.
        let mut nodes = vec![qwen_node(0, 64)];
        nodes.extend(qwen_nodes(5, 80).into_iter().skip(1));
        let plan = plan_topology(&qwen_coder_480b_input(nodes)).unwrap();

        let smallest_stage = plan
            .stages
            .iter()
            .find(|stage| stage.node_id == "qwen-node-00")
            .unwrap();
        let max_layers = plan
            .stages
            .iter()
            .map(|stage| stage.layer_end - stage.layer_start)
            .max()
            .unwrap();
        assert!(smallest_stage.layer_end - smallest_stage.layer_start < max_layers);
    }

    #[test]
    fn qwen_coder_480b_applies_max_vram_and_headroom_in_simulation() {
        // Simulation: one physically larger 120 GiB node capped to 80 GiB
        // with 16 GiB runtime headroom, plus 5 x 80 GiB nodes.
        //
        // Expected topology: the capped node receives fewer layers than the
        // largest stage, despite having 120 GiB physically detected.
        //
        // Why: planning must apply max-vram and local runtime headroom per
        // node before assigning layers. The capped node's usable budget is
        // 64 GiB, so it should be treated as smaller than the uncapped peers.
        let mut capped = qwen_node(0, 120);
        capped.max_vram_bytes = Some(80 * GIB);
        capped.runtime_headroom_bytes = 16 * GIB;
        let mut nodes = vec![capped];
        nodes.extend(qwen_nodes(5, 80).into_iter().skip(1));
        let plan = plan_topology(&qwen_coder_480b_input(nodes)).unwrap();

        let capped_stage = plan
            .stages
            .iter()
            .find(|stage| stage.node_id == "qwen-node-00")
            .unwrap();
        let max_layers = plan
            .stages
            .iter()
            .map(|stage| stage.layer_end - stage.layer_start)
            .max()
            .unwrap();
        assert!(capped_stage.layer_end - capped_stage.layer_start < max_layers);
    }
}
