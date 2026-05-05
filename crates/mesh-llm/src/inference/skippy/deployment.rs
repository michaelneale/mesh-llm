use std::collections::HashMap;

use skippy_protocol::{FlashAttentionType, LoadMode, PeerConfig, StageConfig, StageDevice};

use super::family_policy::FamilyPolicy;
use super::materialization::StagePackageInfo;
use super::topology::MeshStagePlan;
use super::{
    KvCachePolicy, StageLoadRequest, StagePeerDescriptor, StageStatusSnapshot, StageStopRequest,
};
use crate::mesh;

pub(crate) struct StageDeploymentContext<'a> {
    pub(crate) topology_id: &'a str,
    pub(crate) run_id: &'a str,
    pub(crate) model_id: &'a str,
    pub(crate) package: &'a StagePackageInfo,
    pub(crate) family_policy: &'a FamilyPolicy,
    pub(crate) activation_width: i32,
    pub(crate) ctx_size: u32,
    pub(crate) lane_count: u32,
    pub(crate) n_batch: Option<u32>,
    pub(crate) n_ubatch: Option<u32>,
    pub(crate) kv_cache: KvCachePolicy,
    pub(crate) flash_attn_type: FlashAttentionType,
    pub(crate) projector_path: Option<String>,
}

pub(crate) fn remote_stage_load_request(
    context: &StageDeploymentContext<'_>,
    stage: &MeshStagePlan,
    downstream: Option<StagePeerDescriptor>,
) -> StageLoadRequest {
    StageLoadRequest {
        topology_id: context.topology_id.to_string(),
        run_id: context.run_id.to_string(),
        model_id: context.model_id.to_string(),
        backend: "skippy".to_string(),
        package_ref: context.package.package_ref.clone(),
        manifest_sha256: context.package.manifest_sha256.clone(),
        stage_id: stage.stage_id.clone(),
        stage_index: stage.stage_index,
        layer_start: stage.layer_start,
        layer_end: stage.layer_end,
        model_path: Some(context.package.package_ref.clone()),
        projector_path: None,
        selected_device: None,
        bind_addr: "127.0.0.1:0".to_string(),
        activation_width: context.activation_width,
        wire_dtype: context.family_policy.activation_wire_dtype,
        ctx_size: context.ctx_size,
        lane_count: context.lane_count,
        n_batch: context.n_batch,
        n_ubatch: context.n_ubatch,
        n_gpu_layers: -1,
        cache_type_k: context.kv_cache.cache_type_k().to_string(),
        cache_type_v: context.kv_cache.cache_type_v().to_string(),
        flash_attn_type: context.flash_attn_type,
        shutdown_generation: 1,
        load_mode: LoadMode::LayerPackage,
        upstream: None,
        downstream,
    }
}

pub(crate) fn stage0_config(
    context: &StageDeploymentContext<'_>,
    stage0: &MeshStagePlan,
    downstream_stage: &MeshStagePlan,
    downstream_endpoint: String,
    selected_device: Option<StageDevice>,
) -> StageConfig {
    StageConfig {
        run_id: context.run_id.to_string(),
        topology_id: context.topology_id.to_string(),
        model_id: context.model_id.to_string(),
        package_ref: Some(context.package.package_ref.clone()),
        manifest_sha256: Some(context.package.manifest_sha256.clone()),
        source_model_path: Some(context.package.source_model_path.clone()),
        source_model_sha256: Some(context.package.source_model_sha256.clone()),
        source_model_bytes: context.package.source_model_bytes,
        materialized_path: None,
        materialized_pinned: false,
        model_path: Some(context.package.package_ref.clone()),
        projector_path: context.projector_path.clone(),
        stage_id: stage0.stage_id.clone(),
        stage_index: stage0.stage_index,
        layer_start: stage0.layer_start,
        layer_end: stage0.layer_end,
        ctx_size: context.ctx_size,
        lane_count: context.lane_count,
        n_batch: context.n_batch,
        n_ubatch: context.n_ubatch,
        n_gpu_layers: -1,
        cache_type_k: context.kv_cache.cache_type_k().to_string(),
        cache_type_v: context.kv_cache.cache_type_v().to_string(),
        flash_attn_type: context.flash_attn_type,
        filter_tensors_on_load: true,
        selected_device,
        kv_cache: None,
        load_mode: LoadMode::LayerPackage,
        bind_addr: "127.0.0.1:0".to_string(),
        upstream: None,
        downstream: Some(PeerConfig {
            stage_id: downstream_stage.stage_id.clone(),
            stage_index: downstream_stage.stage_index,
            endpoint: downstream_endpoint,
        }),
    }
}

pub(crate) fn stage_stop_request(
    context: &StageDeploymentContext<'_>,
    stage: &MeshStagePlan,
    shutdown_generation: u64,
) -> StageStopRequest {
    StageStopRequest {
        topology_id: context.topology_id.to_string(),
        run_id: context.run_id.to_string(),
        stage_id: stage.stage_id.clone(),
        shutdown_generation,
    }
}

pub(crate) fn stage_topology_instance(
    context: &StageDeploymentContext<'_>,
    stages: &[MeshStagePlan],
    ready_statuses: &HashMap<String, StageStatusSnapshot>,
    stage0_bind_addr: String,
) -> mesh::StageTopologyInstance {
    mesh::StageTopologyInstance {
        topology_id: context.topology_id.to_string(),
        run_id: context.run_id.to_string(),
        model_id: context.model_id.to_string(),
        package_ref: context.package.package_ref.clone(),
        manifest_sha256: context.package.manifest_sha256.clone(),
        stages: stages
            .iter()
            .map(|stage| mesh::StageAssignment {
                stage_id: stage.stage_id.clone(),
                stage_index: stage.stage_index,
                node_id: stage.node_id,
                layer_start: stage.layer_start,
                layer_end: stage.layer_end,
                endpoint: mesh::StageEndpoint {
                    bind_addr: ready_statuses
                        .get(&stage.stage_id)
                        .map(|status| status.bind_addr.clone())
                        .unwrap_or_else(|| stage0_bind_addr.clone()),
                },
            })
            .collect(),
    }
}

pub(crate) fn pinned_stage_device(
    pinned_gpu: Option<&crate::runtime::StartupPinnedGpuTarget>,
) -> Option<StageDevice> {
    pinned_gpu.map(|gpu| StageDevice {
        backend_device: gpu.backend_device.clone(),
        stable_id: Some(gpu.stable_id.clone()),
        index: Some(gpu.index),
        vram_bytes: Some(gpu.vram_bytes),
    })
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

    fn package() -> StagePackageInfo {
        StagePackageInfo {
            package_ref: "hf://Mesh-LLM/demo-package".to_string(),
            package_dir: PathBuf::from("/tmp/package"),
            manifest_sha256: "manifest".to_string(),
            model_id: "model".to_string(),
            source_model_path: "model.gguf".to_string(),
            source_model_sha256: "source".to_string(),
            source_model_bytes: Some(100),
            layer_count: 4,
            activation_width: 1024,
            layers: vec![StagePackageLayerInfo {
                layer_index: 0,
                tensor_count: 1,
                tensor_bytes: 10,
                artifact_bytes: 12,
            }],
        }
    }

    #[test]
    fn remote_load_request_uses_package_identity_and_layer_mode() {
        let package = package();
        let context = StageDeploymentContext {
            topology_id: "topology-a",
            run_id: "run-a",
            model_id: "model-a",
            package: &package,
            family_policy: &crate::inference::skippy::family_policy::family_policy_for_model_path(
                "model.gguf",
                Some("Qwen/Qwen3-0.6B:Q8_0"),
            ),
            activation_width: 1024,
            ctx_size: 8192,
            lane_count: 2,
            n_batch: None,
            n_ubatch: None,
            kv_cache: KvCachePolicy::for_model_size(100),
            flash_attn_type: FlashAttentionType::Auto,
            projector_path: Some("/models/mmproj.gguf".to_string()),
        };
        let request = remote_stage_load_request(
            &context,
            &MeshStagePlan {
                stage_id: "stage-1".to_string(),
                stage_index: 1,
                node_id: make_id(1),
                layer_start: 4,
                layer_end: 8,
            },
            None,
        );

        assert_eq!(request.package_ref, "hf://Mesh-LLM/demo-package");
        assert_eq!(request.manifest_sha256, "manifest");
        assert_eq!(request.load_mode, LoadMode::LayerPackage);
        assert_eq!(
            request.model_path.as_deref(),
            Some("hf://Mesh-LLM/demo-package")
        );
        assert_eq!((request.layer_start, request.layer_end), (4, 8));
        assert!(request.projector_path.is_none());

        let stage0 = stage0_config(
            &context,
            &MeshStagePlan {
                stage_id: "stage-0".to_string(),
                stage_index: 0,
                node_id: make_id(0),
                layer_start: 0,
                layer_end: 4,
            },
            &MeshStagePlan {
                stage_id: "stage-1".to_string(),
                stage_index: 1,
                node_id: make_id(1),
                layer_start: 4,
                layer_end: 8,
            },
            "127.0.0.1:9001".to_string(),
            None,
        );
        assert_eq!(
            stage0.projector_path.as_deref(),
            Some("/models/mmproj.gguf")
        );
    }
}
