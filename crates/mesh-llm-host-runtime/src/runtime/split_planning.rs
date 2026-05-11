use crate::inference::skippy;
use anyhow::{Context, Result};
use skippy_coordinator::topology::{
    plan_topology, TopologyNode, TopologyPlanningInput, TopologyStagePlan,
};
use std::collections::HashMap;

use super::local::{runtime_model_required_bytes, SplitParticipant, SplitParticipantExclusion};

// VRAM budget already accounts for OS/runtime reservations (e.g. Metal's
// recommendedMaxWorkingSetSize on macOS).  No additional headroom deduction.
const RUNTIME_NODE_HEADROOM_NUMERATOR: u64 = 0;
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

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct RuntimeSliceStagePlan {
    pub(super) stage_id: String,
    pub(super) stage_index: u32,
    pub(super) node_id: iroh::EndpointId,
    pub(super) layer_start: u32,
    pub(super) layer_end: u32,
    pub(super) parameter_bytes: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct SplitTopologyResourceInputs {
    pub(super) native_context_length: u32,
    pub(super) kv_bytes_per_token: u64,
    pub(super) ctx_size_override: Option<u32>,
    pub(super) parallel_override: Option<usize>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct PlannedRuntimeSliceTopology {
    pub(super) stages: Vec<RuntimeSliceStagePlan>,
    pub(super) context_length: u32,
    pub(super) slots: usize,
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

pub(super) fn split_participants_for_stages(
    participants: &[SplitParticipant],
    stages: &[RuntimeSliceStagePlan],
) -> Vec<SplitParticipant> {
    let participant_by_node = participants
        .iter()
        .copied()
        .map(|participant| (participant.node_id, participant))
        .collect::<HashMap<_, _>>();
    stages
        .iter()
        .filter_map(|stage| participant_by_node.get(&stage.node_id).copied())
        .collect()
}

pub(super) fn plan_runtime_slice_topology_with_resources(
    topology_id: &str,
    model_ref: &str,
    package: &skippy::SkippyPackageIdentity,
    participants: &[SplitParticipant],
    excluded: &[SplitParticipantExclusion],
    resources: SplitTopologyResourceInputs,
) -> Result<PlannedRuntimeSliceTopology> {
    tracing::info!(
        topology_id,
        model_ref,
        participants = ?split_participant_labels(participants),
        layer_count = package.layer_count,
        native_context_length = resources.native_context_length,
        "planning resource-aware split runtime topology"
    );

    let participant_by_id = participants
        .iter()
        .copied()
        .map(|participant| (participant.node_id.to_string(), participant))
        .collect::<HashMap<_, _>>();
    let plan = plan_split_topology(SplitTopologyPlanInput {
        native_context_length: resources.native_context_length,
        layer_count: package.layer_count,
        model_weight_bytes: package.source_model_bytes,
        kv_bytes_per_token: resources.kv_bytes_per_token,
        context_length_override: resources.ctx_size_override,
        parallel_lanes_override: resources.parallel_override,
        minimum_nodes: super::local::SPLIT_DEFAULT_MIN_PARTICIPANTS,
        nodes: participants
            .iter()
            .map(|participant| SplitTopologyPlanNode {
                node_id: participant.node_id.to_string(),
                detected_vram_bytes: participant.vram_bytes,
                max_vram_bytes: Some(participant.vram_bytes),
                runtime_headroom_bytes: default_runtime_headroom_bytes(participant.vram_bytes),
            })
            .collect(),
    })?;

    let mut stages = plan
        .stages
        .into_iter()
        .map(|stage| {
            let participant = participant_by_id.get(&stage.node_id).ok_or_else(|| {
                anyhow::anyhow!("topology planner returned unknown node {}", stage.node_id)
            })?;
            Ok(RuntimeSliceStagePlan {
                stage_id: stage.stage_id,
                stage_index: stage.stage_index,
                node_id: participant.node_id,
                layer_start: stage.layer_start,
                layer_end: stage.layer_end,
                parameter_bytes: stage.parameter_bytes,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    stages.sort_by_key(|stage| stage.stage_index);
    validate_split_capacity(model_ref, package, participants, &stages, excluded)?;
    tracing::info!(
        topology_id,
        model_ref,
        context_length = plan.context_length,
        slots = plan.parallel_lanes,
        stages = ?split_stage_plan_labels(&stages),
        "planned resource-aware split runtime topology"
    );
    Ok(PlannedRuntimeSliceTopology {
        stages,
        context_length: plan.context_length,
        slots: plan.parallel_lanes,
    })
}

pub(super) fn split_participant_labels(participants: &[SplitParticipant]) -> Vec<String> {
    participants
        .iter()
        .map(|participant| {
            format!(
                "{}:{} cached={} missing={} rtt={}ms transfer={}",
                participant.node_id.fmt_short(),
                format_gb(participant.vram_bytes),
                format_gb(participant.cached_slice_bytes),
                format_gb(participant.missing_artifact_bytes),
                participant.rtt_ms.unwrap_or_default(),
                participant.artifact_transfer_supported
            )
        })
        .collect()
}

pub(super) fn split_participant_exclusion_labels(
    excluded: &[SplitParticipantExclusion],
) -> Vec<String> {
    excluded
        .iter()
        .map(|exclusion| {
            format!(
                "{}:{}",
                exclusion.node_id.fmt_short(),
                exclusion.reason.as_str()
            )
        })
        .collect()
}

pub(super) fn validate_split_capacity(
    model_ref: &str,
    package: &skippy::SkippyPackageIdentity,
    participants: &[SplitParticipant],
    stages: &[RuntimeSliceStagePlan],
    excluded: &[SplitParticipantExclusion],
) -> Result<()> {
    let total_vram_bytes = participants
        .iter()
        .map(|participant| participant.vram_bytes)
        .sum::<u64>();
    let required_total_bytes = runtime_model_required_bytes(package.source_model_bytes);
    anyhow::ensure!(
        total_vram_bytes >= required_total_bytes,
        "{}",
        format_aggregate_split_capacity_error(
            model_ref,
            required_total_bytes,
            total_vram_bytes,
            participants,
            excluded
        )
    );

    let vram_by_node = participants
        .iter()
        .map(|participant| (participant.node_id, participant.vram_bytes))
        .collect::<HashMap<_, _>>();
    for stage in stages {
        let node_vram = vram_by_node
            .get(&stage.node_id)
            .copied()
            .unwrap_or_default();
        let required_stage_bytes = runtime_model_required_bytes(stage.parameter_bytes);
        anyhow::ensure!(
            node_vram >= required_stage_bytes,
            "{} assigned to {} for {model_ref} requires {}, which exceeds node capacity {}",
            stage.stage_id,
            stage.node_id.fmt_short(),
            format_gb(required_stage_bytes),
            format_gb(node_vram)
        );
    }
    Ok(())
}

pub(super) fn format_aggregate_split_capacity_error(
    model_ref: &str,
    required_bytes: u64,
    available_bytes: u64,
    participants: &[SplitParticipant],
    excluded: &[SplitParticipantExclusion],
) -> String {
    SplitCapacityReadinessReport::new(required_bytes, available_bytes, participants, excluded)
        .error_message(model_ref)
}

pub(super) fn format_gb(bytes: u64) -> String {
    format!("{:.1}GB", bytes as f64 / 1e9)
}

pub(super) fn split_stage_plan_labels(stages: &[RuntimeSliceStagePlan]) -> Vec<String> {
    stages
        .iter()
        .map(|stage| {
            format!(
                "{}:{}:{}..{}",
                stage.stage_id,
                stage.node_id.fmt_short(),
                stage.layer_start,
                stage.layer_end
            )
        })
        .collect()
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct SplitCapacityReadinessReport {
    required_bytes: u64,
    available_bytes: u64,
    missing_bytes: u64,
    participants: Vec<SplitParticipant>,
    excluded: Vec<SplitParticipantExclusion>,
}

impl SplitCapacityReadinessReport {
    fn new(
        required_bytes: u64,
        available_bytes: u64,
        participants: &[SplitParticipant],
        excluded: &[SplitParticipantExclusion],
    ) -> Self {
        Self {
            required_bytes,
            available_bytes,
            missing_bytes: required_bytes.saturating_sub(available_bytes),
            participants: participants.to_vec(),
            excluded: excluded.to_vec(),
        }
    }

    fn error_message(&self, model_ref: &str) -> String {
        let mut message = format!(
            "aggregate split capacity for {model_ref} requires {}, mesh has {} across {} participant(s), short by {}",
            format_gb(self.required_bytes),
            format_gb(self.available_bytes),
            self.participants.len(),
            format_gb(self.missing_bytes)
        );
        if !self.participants.is_empty() {
            message.push_str("; participants [");
            message.push_str(&split_participant_labels(&self.participants).join(", "));
            message.push(']');
        }
        if !self.excluded.is_empty() {
            message.push_str("; excluded [");
            message.push_str(&split_participant_exclusion_labels(&self.excluded).join(", "));
            message.push(']');
        }
        message
    }
}

#[cfg(test)]
mod tests {
    use super::super::local::SplitParticipantExclusionReason;
    use super::*;
    use iroh::SecretKey;
    use std::path::PathBuf;

    fn make_id(seed: u8) -> iroh::EndpointId {
        let mut bytes = [0u8; 32];
        bytes[0] = seed;
        SecretKey::from_bytes(&bytes).public()
    }

    fn package(layer_count: u32, source_model_bytes: u64) -> skippy::SkippyPackageIdentity {
        skippy::SkippyPackageIdentity {
            package_ref: "gguf:///models/qwen.gguf".to_string(),
            manifest_sha256: "manifest".to_string(),
            source_model_path: PathBuf::from("/models/qwen.gguf"),
            source_model_sha256: "source".to_string(),
            source_model_bytes,
            source_files: Vec::new(),
            layer_count,
            activation_width: 896,
            tensor_count: 100,
        }
    }

    fn participant(seed: u8, vram_bytes: u64) -> SplitParticipant {
        SplitParticipant::new(make_id(seed), vram_bytes, None)
    }

    #[test]
    fn default_runtime_headroom_is_zero() {
        assert_eq!(default_runtime_headroom_bytes(100), 0);
        assert_eq!(default_runtime_headroom_bytes(101), 0);
    }

    #[test]
    fn selects_participants_in_stage_order() {
        let a = participant(1, 24_000_000_000);
        let b = participant(2, 24_000_000_000);
        let stages = vec![
            RuntimeSliceStagePlan {
                stage_id: "stage-0".to_string(),
                stage_index: 0,
                node_id: b.node_id,
                layer_start: 0,
                layer_end: 10,
                parameter_bytes: 10_000_000,
            },
            RuntimeSliceStagePlan {
                stage_id: "stage-1".to_string(),
                stage_index: 1,
                node_id: a.node_id,
                layer_start: 10,
                layer_end: 20,
                parameter_bytes: 10_000_000,
            },
        ];

        let selected = split_participants_for_stages(&[a, b], &stages);

        assert_eq!(
            selected
                .iter()
                .map(|participant| participant.node_id)
                .collect::<Vec<_>>(),
            vec![b.node_id, a.node_id]
        );
    }

    #[test]
    fn resource_planner_returns_runtime_stage_shape() {
        let participants = vec![
            participant(1, 42_000_000_000),
            participant(2, 42_000_000_000),
            participant(3, 42_000_000_000),
        ];

        let plan = plan_runtime_slice_topology_with_resources(
            "topology-test",
            "model-a",
            &package(30, 60_000_000_000),
            &participants,
            &[],
            SplitTopologyResourceInputs {
                native_context_length: 65_536,
                kv_bytes_per_token: 16 * 1024,
                ctx_size_override: None,
                parallel_override: None,
            },
        )
        .expect("resource-aware topology");

        assert_eq!(plan.context_length, 65_536);
        assert_eq!(plan.stages.len(), 2);
        assert!(plan.slots > 0);
        assert_eq!(plan.stages.first().unwrap().layer_start, 0);
        assert_eq!(plan.stages.last().unwrap().layer_end, 30);
    }

    #[test]
    fn capacity_report_includes_participants_and_exclusions() {
        let participants = vec![participant(1, 40_000_000_000)];
        let excluded = vec![SplitParticipantExclusion {
            node_id: make_id(2),
            reason: SplitParticipantExclusionReason::MissingVram,
        }];

        let message = format_aggregate_split_capacity_error(
            "model-a",
            100_000_000_000,
            40_000_000_000,
            &participants,
            &excluded,
        );

        assert!(message.contains("model-a"));
        assert!(message.contains("short by 60.0GB"));
        assert!(message.contains("participants ["));
        assert!(message.contains("excluded ["));
        assert!(message.contains("missing_vram"));
    }
}
