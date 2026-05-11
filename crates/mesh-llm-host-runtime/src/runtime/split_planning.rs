use anyhow::{Context, Result};
use skippy_coordinator::topology::{
    plan_topology, TopologyNode, TopologyPlanningInput, TopologyStagePlan,
};

const RUNTIME_NODE_HEADROOM_NUMERATOR: u64 = 1;
const RUNTIME_NODE_HEADROOM_DENOMINATOR: u64 = 10;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct SplitTopologyPlanInput {
    pub(super) native_context_length: u32,
    pub(super) layer_count: u32,
    pub(super) model_weight_bytes: u64,
    pub(super) kv_bytes_per_token: u64,
    pub(super) context_length_override: Option<u32>,
    pub(super) parallel_lanes_override: Option<usize>,
    pub(super) minimum_nodes: usize,
    pub(super) nodes: Vec<SplitTopologyPlanNode>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct SplitTopologyPlanNode {
    pub(super) node_id: String,
    pub(super) detected_vram_bytes: u64,
    pub(super) max_vram_bytes: Option<u64>,
    pub(super) runtime_headroom_bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct SplitTopologyPlan {
    pub(super) context_length: u32,
    pub(super) parallel_lanes: usize,
    pub(super) stages: Vec<TopologyStagePlan>,
}

pub(super) fn plan_split_topology(input: SplitTopologyPlanInput) -> Result<SplitTopologyPlan> {
    let plan = plan_topology(&TopologyPlanningInput {
        native_context_length: input.native_context_length,
        layer_count: input.layer_count,
        model_weight_bytes: input.model_weight_bytes,
        kv_bytes_per_token: input.kv_bytes_per_token,
        minimum_nodes: input.minimum_nodes,
        nodes: input
            .nodes
            .into_iter()
            .map(|node| TopologyNode {
                node_id: node.node_id,
                detected_vram_bytes: node.detected_vram_bytes,
                max_vram_bytes: node.max_vram_bytes,
                runtime_headroom_bytes: node.runtime_headroom_bytes,
            })
            .collect(),
        context_length_override: input.context_length_override,
        parallel_lanes_override: input.parallel_lanes_override,
    })
    .context("plan skippy split topology")?;

    Ok(SplitTopologyPlan {
        context_length: plan.context_length,
        parallel_lanes: plan.parallel_lanes,
        stages: plan.stages,
    })
}

pub(super) fn default_runtime_headroom_bytes(vram_bytes: u64) -> u64 {
    vram_bytes
        .saturating_mul(RUNTIME_NODE_HEADROOM_NUMERATOR)
        .div_ceil(RUNTIME_NODE_HEADROOM_DENOMINATOR)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_runtime_headroom_is_per_node() {
        assert_eq!(default_runtime_headroom_bytes(100), 10);
        assert_eq!(default_runtime_headroom_bytes(101), 11);
    }
}
