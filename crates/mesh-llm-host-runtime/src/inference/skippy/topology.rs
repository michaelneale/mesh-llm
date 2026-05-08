use std::collections::HashMap;

use anyhow::{anyhow, bail, Result};
use skippy_topology::{
    infer_family_capability, plan_package_aware_contiguous_with_signals, BoundaryDecision,
    DiagnosticSeverity, LayerSpec, NodePlacementSignal, NodeSpec, PlannerPolicy,
    TopologyPlanRequest,
};

use super::{materialization::StagePackageInfo, package::SkippyPackageIdentity};

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct StageTopologyParticipant {
    pub(crate) node_id: iroh::EndpointId,
    pub(crate) vram_bytes: u64,
    pub(crate) cached_slice_bytes: u64,
    pub(crate) missing_artifact_bytes: u64,
    pub(crate) rtt_ms: Option<u32>,
    pub(crate) artifact_transfer_supported: bool,
    pub(crate) availability_score: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct MeshStagePlan {
    pub(crate) stage_id: String,
    pub(crate) stage_index: u32,
    pub(crate) node_id: iroh::EndpointId,
    pub(crate) layer_start: u32,
    pub(crate) layer_end: u32,
    pub(crate) parameter_bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct MeshTopologyPlan {
    pub(crate) stages: Vec<MeshStagePlan>,
    pub(crate) family_id: Option<String>,
    pub(crate) diagnostics: Vec<String>,
}

pub(crate) fn plan_package_topology(
    topology_id: &str,
    package: &StagePackageInfo,
    participants: &[StageTopologyParticipant],
) -> Result<MeshTopologyPlan> {
    if package.layer_count == 0 {
        bail!("stage topology requires at least one package layer");
    }
    if participants.is_empty() {
        bail!("stage topology requires at least one participant");
    }

    let node_by_id = participants
        .iter()
        .map(|participant| (participant.node_id.to_string(), participant.node_id))
        .collect::<HashMap<_, _>>();
    let layers = layer_specs(package);
    let family = infer_family_capability(
        &package.model_id,
        package.layer_count,
        package.activation_width,
    );
    let request = TopologyPlanRequest {
        topology_id: topology_id.to_string(),
        model_id: package.model_id.clone(),
        layers,
        nodes: participants
            .iter()
            .map(|participant| NodeSpec {
                node_id: participant.node_id.to_string(),
                cached_slice_bytes: participant.cached_slice_bytes,
                vram_bytes: participant.vram_bytes,
            })
            .collect(),
        family,
        policy: PlannerPolicy::default(),
    };
    let placement_signals = participants
        .iter()
        .map(|participant| NodePlacementSignal {
            node_id: participant.node_id.to_string(),
            cached_slice_bytes: participant.cached_slice_bytes,
            missing_artifact_bytes: participant.missing_artifact_bytes,
            rtt_ms: participant.rtt_ms,
            artifact_transfer_supported: participant.artifact_transfer_supported,
            availability_score: participant.availability_score,
        })
        .collect::<Vec<_>>();
    let plan = plan_package_aware_contiguous_with_signals(&request, &placement_signals)?;

    let rejected = plan
        .boundaries
        .iter()
        .filter(|boundary| boundary.decision == BoundaryDecision::Rejected)
        .map(|boundary| {
            format!(
                "rejected boundary at layer {}: {}",
                boundary.layer_boundary,
                boundary.messages.join("; ")
            )
        })
        .collect::<Vec<_>>();
    if !rejected.is_empty() {
        bail!("{}", rejected.join("; "));
    }
    let errors = plan
        .diagnostics
        .iter()
        .filter(|diagnostic| diagnostic.severity == DiagnosticSeverity::Error)
        .map(|diagnostic| diagnostic.message.clone())
        .collect::<Vec<_>>();
    if !errors.is_empty() {
        bail!("{}", errors.join("; "));
    }

    let mut stages = plan
        .stages
        .into_iter()
        .map(|stage| {
            let node_id = node_by_id.get(&stage.node_id).copied().ok_or_else(|| {
                anyhow!("topology planner returned unknown node {}", stage.node_id)
            })?;
            Ok(MeshStagePlan {
                stage_id: stage.stage_id,
                stage_index: stage.stage_index,
                node_id,
                layer_start: stage.layer_start,
                layer_end: stage.layer_end,
                parameter_bytes: stage.parameter_bytes,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    stages.sort_by_key(|stage| stage.stage_index);

    Ok(MeshTopologyPlan {
        stages,
        family_id: plan.family_id,
        diagnostics: plan
            .diagnostics
            .into_iter()
            .map(|diagnostic| diagnostic.message)
            .collect(),
    })
}

pub(crate) fn plan_package_identity_topology(
    topology_id: &str,
    model_id: &str,
    package: &SkippyPackageIdentity,
    participants: &[StageTopologyParticipant],
) -> Result<MeshTopologyPlan> {
    let package_dir = package
        .source_model_path
        .parent()
        .map(ToOwned::to_owned)
        .unwrap_or_default();
    let package = StagePackageInfo {
        package_ref: package.package_ref.clone(),
        package_dir,
        manifest_sha256: package.manifest_sha256.clone(),
        model_id: model_id.to_string(),
        source_model_path: package.source_model_path.to_string_lossy().to_string(),
        source_model_sha256: package.source_model_sha256.clone(),
        source_model_bytes: Some(package.source_model_bytes),
        layer_count: package.layer_count,
        activation_width: package.activation_width,
        projector_path: None,
        layers: Vec::new(),
    };
    plan_package_topology(topology_id, &package, participants)
}

fn layer_specs(package: &StagePackageInfo) -> Vec<LayerSpec> {
    let fallback_parameter_bytes = package
        .source_model_bytes
        .map(|bytes| bytes / u64::from(package.layer_count.max(1)))
        .unwrap_or_default();
    let layer_bytes = package
        .layers
        .iter()
        .map(|layer| {
            (
                layer.layer_index,
                layer.tensor_bytes.max(layer.artifact_bytes),
            )
        })
        .collect::<HashMap<_, _>>();

    (0..package.layer_count)
        .map(|index| LayerSpec {
            index,
            attention: true,
            recurrent: false,
            parameter_bytes: layer_bytes
                .get(&index)
                .copied()
                .unwrap_or(fallback_parameter_bytes),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::skippy::materialization::StagePackageLayerInfo;
    use iroh::SecretKey;
    use std::path::PathBuf;

    fn make_id(seed: u8) -> iroh::EndpointId {
        let mut bytes = [0u8; 32];
        bytes[0] = seed;
        SecretKey::from_bytes(&bytes).public()
    }

    fn participant(node_id: iroh::EndpointId, vram_bytes: u64) -> StageTopologyParticipant {
        StageTopologyParticipant {
            node_id,
            vram_bytes,
            cached_slice_bytes: 0,
            missing_artifact_bytes: 0,
            rtt_ms: None,
            artifact_transfer_supported: false,
            availability_score: 0,
        }
    }

    fn package(layer_count: u32) -> StagePackageInfo {
        StagePackageInfo {
            package_ref: "hf://Mesh-LLM/demo-package".to_string(),
            package_dir: PathBuf::from("/tmp/package"),
            manifest_sha256: "manifest".to_string(),
            model_id: "Qwen/Qwen3-0.6B".to_string(),
            source_model_path: "model.gguf".to_string(),
            source_model_sha256: "source".to_string(),
            source_model_bytes: Some(120),
            layer_count,
            activation_width: 1024,
            projector_path: None,
            layers: (0..layer_count)
                .map(|layer_index| StagePackageLayerInfo {
                    layer_index,
                    tensor_count: 1,
                    tensor_bytes: 10,
                    artifact_bytes: 12,
                })
                .collect(),
        }
    }

    #[test]
    fn topology_adapter_preserves_weighted_stage_order() {
        let id_a = make_id(1);
        let id_b = make_id(2);
        let id_c = make_id(3);

        let plan = plan_package_topology(
            "topology-a",
            &package(12),
            &[
                participant(id_a, 60),
                participant(id_b, 30),
                participant(id_c, 30),
            ],
        )
        .unwrap();

        assert_eq!(plan.stages.len(), 3);
        assert_eq!(
            (
                plan.stages[0].node_id,
                plan.stages[0].layer_start,
                plan.stages[0].layer_end
            ),
            (id_a, 0, 6)
        );
        assert_eq!(
            (
                plan.stages[1].node_id,
                plan.stages[1].layer_start,
                plan.stages[1].layer_end
            ),
            (id_b, 6, 9)
        );
        assert_eq!(
            (
                plan.stages[2].node_id,
                plan.stages[2].layer_start,
                plan.stages[2].layer_end
            ),
            (id_c, 9, 12)
        );
    }

    #[test]
    fn topology_adapter_drops_extra_participants_without_empty_ranges() {
        let id_a = make_id(1);
        let id_b = make_id(2);
        let id_c = make_id(3);

        let plan = plan_package_topology(
            "topology-a",
            &package(2),
            &[
                participant(id_a, 10),
                participant(id_b, 10),
                participant(id_c, 10),
            ],
        )
        .unwrap();

        assert_eq!(plan.stages.len(), 2);
        assert_eq!(
            (
                plan.stages[0].node_id,
                plan.stages[0].layer_start,
                plan.stages[0].layer_end
            ),
            (id_a, 0, 1)
        );
        assert_eq!(
            (
                plan.stages[1].node_id,
                plan.stages[1].layer_start,
                plan.stages[1].layer_end
            ),
            (id_b, 1, 2)
        );
        assert!(plan
            .stages
            .iter()
            .all(|stage| stage.layer_start < stage.layer_end));
    }

    #[test]
    fn topology_adapter_prefers_cached_peer_for_equal_capacity() {
        let id_a = make_id(1);
        let id_b = make_id(2);

        let mut cold = participant(id_a, 40);
        cold.missing_artifact_bytes = 32;
        let mut warm = participant(id_b, 40);
        warm.cached_slice_bytes = 64;
        warm.artifact_transfer_supported = true;

        let plan = plan_package_topology("topology-a", &package(8), &[cold, warm]).unwrap();

        assert_eq!(plan.stages.len(), 2);
        assert_eq!(plan.stages[0].node_id, id_b);
        assert_eq!(
            (plan.stages[0].layer_start, plan.stages[0].layer_end),
            (0, 4)
        );
        assert_eq!(plan.stages[1].node_id, id_a);
    }
}
