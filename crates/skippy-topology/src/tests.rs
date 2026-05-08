use super::*;
use serde::Deserialize;

fn nodes(count: u32) -> Vec<NodeSpec> {
    (0..count)
        .map(|index| NodeSpec {
            node_id: format!("node-{index}"),
            cached_slice_bytes: 0,
            vram_bytes: 0,
        })
        .collect()
}

fn compact_identity(value: &str) -> String {
    value.to_ascii_lowercase().replace(['_', '-', '/', ' '], "")
}

fn weighted_node(node_id: &str, vram_bytes: u64) -> NodeSpec {
    NodeSpec {
        node_id: node_id.to_string(),
        cached_slice_bytes: 0,
        vram_bytes,
    }
}

fn placement_signal(node_id: &str) -> NodePlacementSignal {
    NodePlacementSignal {
        node_id: node_id.to_string(),
        cached_slice_bytes: 0,
        missing_artifact_bytes: 0,
        rtt_ms: None,
        artifact_transfer_supported: false,
        availability_score: 0,
    }
}

fn stage_layout(plan: &TopologyPlan) -> Vec<(&str, u32, u32)> {
    plan.stages
        .iter()
        .map(|stage| (stage.node_id.as_str(), stage.layer_start, stage.layer_end))
        .collect()
}

#[test]
fn dense_attention_plan_allows_costed_kv_migration() {
    let request = TopologyPlanRequest {
        topology_id: "dense".to_string(),
        model_id: "qwen3".to_string(),
        layers: dense_attention_layers(6, 10),
        nodes: nodes(3),
        family: None,
        policy: PlannerPolicy::default(),
    };

    let plan = plan_even_contiguous(&request).expect("plan");

    assert_eq!(plan.stages.len(), 3);
    assert!(plan
        .stages
        .iter()
        .all(|stage| stage.state_affinity == StateAffinity::AttentionKv));
    assert!(plan
        .stages
        .iter()
        .all(|stage| stage.migration_policy == MigrationPolicy::CostedKv));
    assert!(plan.diagnostics.is_empty());
}

#[test]
fn weighted_contiguous_plan_uses_node_vram_for_layer_spans() {
    let request = TopologyPlanRequest {
        topology_id: "topology-a".into(),
        model_id: "model-a".into(),
        layers: dense_attention_layers(12, 10),
        nodes: vec![
            weighted_node("node-a", 60),
            weighted_node("node-b", 30),
            weighted_node("node-c", 30),
        ],
        family: None,
        policy: PlannerPolicy::default(),
    };

    let plan = plan_weighted_contiguous(&request).expect("plan");

    assert_eq!(
        plan.stages
            .iter()
            .map(|stage| (stage.node_id.as_str(), stage.layer_start, stage.layer_end))
            .collect::<Vec<_>>(),
        vec![("node-a", 0, 6), ("node-b", 6, 9), ("node-c", 9, 12)]
    );
}

#[test]
fn weighted_contiguous_plan_falls_back_to_even_without_weights() {
    let request = TopologyPlanRequest {
        topology_id: "topology-a".into(),
        model_id: "model-a".into(),
        layers: dense_attention_layers(6, 10),
        nodes: vec![weighted_node("node-a", 0), weighted_node("node-b", 0)],
        family: None,
        policy: PlannerPolicy::default(),
    };

    let plan = plan_weighted_contiguous(&request).expect("plan");

    assert_eq!(
        plan.stages
            .iter()
            .map(|stage| (stage.node_id.as_str(), stage.layer_start, stage.layer_end))
            .collect::<Vec<_>>(),
        vec![("node-a", 0, 3), ("node-b", 3, 6)]
    );
}

#[test]
fn package_aware_plan_matches_weighted_without_package_signals() {
    let request = TopologyPlanRequest {
        topology_id: "topology-a".into(),
        model_id: "model-a".into(),
        layers: dense_attention_layers(12, 10),
        nodes: vec![
            weighted_node("node-a", 60),
            weighted_node("node-b", 30),
            weighted_node("node-c", 30),
        ],
        family: None,
        policy: PlannerPolicy::default(),
    };

    let weighted_plan = plan_weighted_contiguous(&request).expect("weighted plan");
    let package_plan =
        plan_package_aware_contiguous_with_signals(&request, &[]).expect("package-aware plan");

    assert_eq!(stage_layout(&package_plan), stage_layout(&weighted_plan));
}

#[test]
fn package_aware_plan_prefers_cached_peer_for_equal_capacity() {
    let mut cold = placement_signal("cold");
    cold.missing_artifact_bytes = 32;
    let mut warm = placement_signal("warm");
    warm.cached_slice_bytes = 64;
    warm.artifact_transfer_supported = true;
    let request = TopologyPlanRequest {
        topology_id: "topology-a".into(),
        model_id: "model-a".into(),
        layers: dense_attention_layers(8, 10),
        nodes: vec![weighted_node("cold", 40), weighted_node("warm", 40)],
        family: None,
        policy: PlannerPolicy::default(),
    };

    let plan = plan_package_aware_contiguous_with_signals(&request, &[cold, warm]).expect("plan");

    assert_eq!(stage_layout(&plan), vec![("warm", 0, 4), ("cold", 4, 8)]);
    assert!(plan.stages[0]
        .reason_codes
        .contains(&PlanReasonCode::CacheLocalityPreferred));
    assert!(plan.stages[1]
        .reason_codes
        .contains(&PlanReasonCode::ArtifactTransferPenalty));
}

#[test]
fn package_aware_plan_penalizes_missing_untransferable_artifacts() {
    let mut cold = placement_signal("cold-high-vram");
    cold.missing_artifact_bytes = 64;
    cold.artifact_transfer_supported = false;
    let mut ready = placement_signal("ready-lower-vram");
    ready.cached_slice_bytes = 16;
    ready.artifact_transfer_supported = true;
    let request = TopologyPlanRequest {
        topology_id: "topology-a".into(),
        model_id: "model-a".into(),
        layers: dense_attention_layers(8, 10),
        nodes: vec![
            weighted_node("cold-high-vram", 100),
            weighted_node("ready-lower-vram", 80),
        ],
        family: None,
        policy: PlannerPolicy::default(),
    };

    let plan = plan_package_aware_contiguous_with_signals(&request, &[cold, ready]).expect("plan");

    assert_eq!(plan.stages[0].node_id, "ready-lower-vram");
    assert!(plan.stages[1]
        .reason_codes
        .contains(&PlanReasonCode::ArtifactTransferPenalty));
}

#[test]
fn package_aware_plan_treats_high_rtt_as_cost_not_exclusion() {
    let mut distant = placement_signal("distant");
    distant.rtt_ms = Some(250);
    let request = TopologyPlanRequest {
        topology_id: "topology-a".into(),
        model_id: "model-a".into(),
        layers: dense_attention_layers(8, 10),
        nodes: vec![weighted_node("distant", 100), weighted_node("nearby", 80)],
        family: None,
        policy: PlannerPolicy::default(),
    };

    let plan = plan_package_aware_contiguous_with_signals(&request, &[distant]).expect("plan");

    assert!(plan.stages.iter().any(|stage| stage.node_id == "distant"));
    let distant_stage = plan
        .stages
        .iter()
        .find(|stage| stage.node_id == "distant")
        .expect("distant stage");
    assert!(distant_stage
        .reason_codes
        .contains(&PlanReasonCode::NetworkPipelineCost));
}

#[test]
fn package_aware_plan_can_promote_extra_cached_peer() {
    let mut cached_extra = placement_signal("cached-extra");
    cached_extra.cached_slice_bytes = 100;
    let request = TopologyPlanRequest {
        topology_id: "topology-a".into(),
        model_id: "model-a".into(),
        layers: dense_attention_layers(2, 10),
        nodes: vec![
            weighted_node("cold-a", 40),
            weighted_node("cold-b", 40),
            weighted_node("cached-extra", 40),
        ],
        family: None,
        policy: PlannerPolicy::default(),
    };

    let plan = plan_package_aware_contiguous_with_signals(&request, &[cached_extra]).expect("plan");

    assert_eq!(plan.stages.len(), 2);
    assert_eq!(plan.stages[0].node_id, "cached-extra");
    assert!(plan
        .stages
        .iter()
        .any(|stage| stage.node_id == "cold-a" || stage.node_id == "cold-b"));
}

#[test]
fn falcon_h1_marks_every_stage_as_sticky() {
    let request = TopologyPlanRequest {
        topology_id: "falcon".to_string(),
        model_id: "falcon-h1".to_string(),
        layers: falcon_h1_layers(6, 10),
        nodes: nodes(3),
        family: None,
        policy: PlannerPolicy::default(),
    };

    let plan = plan_even_contiguous(&request).expect("plan");

    assert_eq!(
        plan.stages
            .iter()
            .map(|stage| (stage.layer_start, stage.layer_end))
            .collect::<Vec<_>>(),
        vec![(0, 2), (2, 4), (4, 6)]
    );
    assert!(plan
        .stages
        .iter()
        .all(|stage| stage.state_affinity == StateAffinity::Mixed));
    assert!(plan
        .stages
        .iter()
        .all(|stage| stage.migration_policy == MigrationPolicy::StickyRecurrentOwner));
    assert_eq!(plan.diagnostics.len(), 3);
}

#[test]
fn qwen3next_mixed_layers_only_make_recurrent_ranges_sticky() {
    let request = TopologyPlanRequest {
        topology_id: "qwen3next".to_string(),
        model_id: "qwen3next".to_string(),
        layers: qwen3next_layers(8, [2, 3, 6], 10),
        nodes: nodes(4),
        family: None,
        policy: PlannerPolicy::default(),
    };

    let plan = plan_even_contiguous(&request).expect("plan");

    assert_eq!(
        plan.stages
            .iter()
            .map(|stage| stage.state_affinity)
            .collect::<Vec<_>>(),
        vec![
            StateAffinity::AttentionKv,
            StateAffinity::Recurrent,
            StateAffinity::AttentionKv,
            StateAffinity::Mixed
        ]
    );
    assert_eq!(
        plan.stages
            .iter()
            .map(|stage| stage.migration_policy)
            .collect::<Vec<_>>(),
        vec![
            MigrationPolicy::CostedKv,
            MigrationPolicy::StickyRecurrentOwner,
            MigrationPolicy::CostedKv,
            MigrationPolicy::StickyRecurrentOwner
        ]
    );
    assert_eq!(plan.diagnostics.len(), 2);
}

#[test]
fn explicit_recurrent_transfer_policy_is_loud() {
    let request = TopologyPlanRequest {
        topology_id: "transfer".to_string(),
        model_id: "falcon-h1".to_string(),
        layers: falcon_h1_layers(2, 10),
        nodes: nodes(1),
        family: None,
        policy: PlannerPolicy {
            allow_recurrent_state_transfer: true,
        },
    };

    let plan = plan_even_contiguous(&request).expect("plan");

    assert_eq!(
        plan.stages[0].migration_policy,
        MigrationPolicy::RecurrentStateTransferAllowed
    );
    assert_eq!(plan.diagnostics[0].severity, DiagnosticSeverity::Warning);
}

#[test]
fn qwen3_family_defaults_to_f16_and_records_q8_rejection() {
    let request = TopologyPlanRequest {
        topology_id: "qwen3-wire".to_string(),
        model_id: "qwen3".to_string(),
        layers: dense_attention_layers(28, 10),
        nodes: nodes(2),
        family: Some(qwen3_dense_capability(28, 1024)),
        policy: PlannerPolicy::default(),
    };

    let plan = plan_even_contiguous(&request).expect("plan");

    assert_eq!(plan.family_id.as_deref(), Some("qwen3_dense"));
    assert_eq!(plan.boundaries.len(), 1);
    assert_eq!(plan.boundaries[0].decision, BoundaryDecision::Accepted);
    assert_eq!(plan.boundaries[0].wire_dtype, WireDType::F16);
    assert_eq!(plan.boundaries[0].raw_activation_bytes_per_token, 4096);
    assert_eq!(plan.boundaries[0].wire_payload_bytes_per_token, 2048);
    assert!(plan.boundaries[0]
        .reason_codes
        .contains(&PlanReasonCode::Q8WireRejected));
}

#[test]
fn accepted_dense_families_emit_exact_state_mobility_reason() {
    let request = TopologyPlanRequest {
        topology_id: "gemma3".to_string(),
        model_id: "gemma3".to_string(),
        layers: dense_attention_layers(26, 10),
        nodes: nodes(2),
        family: Some(gemma3_capability(26, 1152)),
        policy: PlannerPolicy::default(),
    };

    let plan = plan_even_contiguous(&request).expect("plan");

    assert_eq!(plan.family_id.as_deref(), Some("gemma3"));
    assert!(plan.stages.iter().all(|stage| {
        stage
            .reason_codes
            .contains(&PlanReasonCode::ExactStateMobilityAccepted)
    }));
    assert!(plan
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == PlanReasonCode::ExactStateMobilityAccepted));
}

#[test]
fn untested_dense_family_blocks_q8_but_has_no_split_constraints() {
    let request = TopologyPlanRequest {
        topology_id: "olmo".to_string(),
        model_id: "olmo".to_string(),
        layers: dense_attention_layers(32, 10),
        nodes: nodes(2),
        family: Some(olmo_capability(32, 4096)),
        policy: PlannerPolicy::default(),
    };

    let plan = plan_even_contiguous(&request).expect("plan");

    assert_eq!(plan.family_id.as_deref(), Some("olmo"));
    assert_eq!(plan.boundaries[0].decision, BoundaryDecision::Accepted);
    assert_eq!(plan.boundaries[0].wire_dtype, WireDType::F16);
    assert!(plan.boundaries[0]
        .reason_codes
        .contains(&PlanReasonCode::DefaultWireDtypeF16));
    assert!(!plan.boundaries[0]
        .reason_codes
        .contains(&PlanReasonCode::Q8WireValidated));
    assert!(!plan.boundaries[0]
        .reason_codes
        .contains(&PlanReasonCode::Q8WireRejected));
}

#[test]
fn measured_dense_family_q8_policy_is_recorded() {
    let families = [
        (
            gemma2_capability(26, 2304),
            PlanReasonCode::Q8WireValidated,
            4608,
        ),
        (
            gemma3_capability(26, 1152),
            PlanReasonCode::Q8WireRejected,
            2304,
        ),
        (
            glm4_capability(40, 4096),
            PlanReasonCode::Q8WireRejected,
            8192,
        ),
    ];

    for (family, expected_reason, expected_f16_wire_bytes) in families {
        let request = TopologyPlanRequest {
            topology_id: family.family_id.clone(),
            model_id: family.family_id.clone(),
            layers: dense_attention_layers(family.layer_count, 10),
            nodes: nodes(2),
            family: Some(family),
            policy: PlannerPolicy::default(),
        };

        let plan = plan_even_contiguous(&request).expect("plan");

        assert_eq!(plan.boundaries[0].wire_dtype, WireDType::F16);
        assert_eq!(
            plan.boundaries[0].wire_payload_bytes_per_token,
            expected_f16_wire_bytes
        );
        assert!(plan.boundaries[0].reason_codes.contains(&expected_reason));
    }
}

#[test]
fn falcon_family_capability_marks_attention_layers_sticky() {
    let request = TopologyPlanRequest {
        topology_id: "falcon-family".to_string(),
        model_id: "falcon-h1".to_string(),
        layers: dense_attention_layers(24, 10),
        nodes: nodes(2),
        family: Some(falcon_h1_capability(24, 2048)),
        policy: PlannerPolicy::default(),
    };

    let plan = plan_even_contiguous(&request).expect("plan");

    assert!(plan
        .stages
        .iter()
        .all(|stage| stage.state_affinity == StateAffinity::Mixed));
    assert!(plan
        .stages
        .iter()
        .all(|stage| stage.migration_policy == MigrationPolicy::StickyRecurrentOwner));
    assert!(plan
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == PlanReasonCode::ExactStateMobilityRejected));
    assert!(plan.boundaries[0]
        .reason_codes
        .contains(&PlanReasonCode::RecurrentOwnerSticky));
}

#[test]
fn gemma4_e4b_accepts_validated_boundary_with_sideband() {
    let request = TopologyPlanRequest {
        topology_id: "gemma-valid".to_string(),
        model_id: "gemma4-e4b".to_string(),
        layers: dense_attention_layers(42, 10),
        nodes: nodes(2),
        family: Some(gemma4_e4b_capability(42, 2560)),
        policy: PlannerPolicy::default(),
    };

    let plan = plan_even_contiguous(&request).expect("plan");

    assert_eq!(plan.boundaries[0].layer_boundary, 21);
    assert_eq!(plan.boundaries[0].decision, BoundaryDecision::Accepted);
    assert_eq!(plan.boundaries[0].wire_dtype, WireDType::F16);
    assert_eq!(plan.boundaries[0].raw_activation_bytes_per_token, 10240);
    assert_eq!(plan.boundaries[0].wire_payload_bytes_per_token, 5120);
    assert!(plan.boundaries[0]
        .reason_codes
        .contains(&PlanReasonCode::TokenSidebandRequired));
    assert!(plan.boundaries[0]
        .reason_codes
        .contains(&PlanReasonCode::Q8WireRejected));
}

#[test]
fn rwkv7_boundary_accounts_for_v_first_sideband() {
    let request = TopologyPlanRequest {
        topology_id: "rwkv7-sideband".to_string(),
        model_id: "rwkv7-191m".to_string(),
        layers: falcon_h1_layers(12, 4),
        nodes: nodes(3),
        family: Some(rwkv7_capability(12, 768)),
        policy: PlannerPolicy::default(),
    };

    let plan = plan_even_contiguous(&request).expect("plan");

    assert_eq!(plan.boundaries[0].layer_boundary, 4);
    assert_eq!(plan.boundaries[0].wire_dtype, WireDType::F16);
    assert_eq!(plan.boundaries[0].raw_activation_bytes_per_token, 6144);
    assert_eq!(plan.boundaries[0].wire_payload_bytes_per_token, 3072);
    assert!(plan.boundaries[0]
        .reason_codes
        .contains(&PlanReasonCode::ActivationSidebandRequired));
    assert!(plan.boundaries[0]
        .reason_codes
        .contains(&PlanReasonCode::RecurrentOwnerSticky));
}

#[test]
fn gemma4_e4b_rejects_known_bad_shared_kv_boundaries() {
    let request = TopologyPlanRequest {
        topology_id: "gemma-invalid".to_string(),
        model_id: "gemma4-e4b".to_string(),
        layers: dense_attention_layers(42, 10),
        nodes: nodes(3),
        family: Some(gemma4_e4b_capability(42, 2560)),
        policy: PlannerPolicy::default(),
    };

    let plan = plan_even_contiguous(&request).expect("plan");

    assert_eq!(
        plan.boundaries
            .iter()
            .map(|boundary| (boundary.layer_boundary, boundary.decision))
            .collect::<Vec<_>>(),
        vec![
            (14, BoundaryDecision::Rejected),
            (28, BoundaryDecision::Rejected)
        ]
    );
    assert!(plan
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == PlanReasonCode::SharedKvRegionCut));
}

#[test]
fn gemma3n_requires_altup_sideband_and_reviewed_kv_boundary() {
    let request = TopologyPlanRequest {
        topology_id: "gemma3n".to_string(),
        model_id: "gemma3n".to_string(),
        layers: dense_attention_layers(30, 10),
        nodes: nodes(3),
        family: Some(gemma3n_capability(30, 2048)),
        policy: PlannerPolicy::default(),
    };

    let even_plan = plan_even_contiguous(&request).expect("even plan");
    assert_eq!(
        even_plan
            .boundaries
            .iter()
            .map(|boundary| {
                (
                    boundary.layer_boundary,
                    boundary.decision,
                    boundary.raw_activation_bytes_per_token,
                    boundary.wire_payload_bytes_per_token,
                )
            })
            .collect::<Vec<_>>(),
        vec![
            (10, BoundaryDecision::Accepted, 32768, 16384),
            (20, BoundaryDecision::Rejected, 32768, 16384)
        ]
    );
    assert!(even_plan.boundaries.iter().all(|boundary| boundary
        .reason_codes
        .contains(&PlanReasonCode::ActivationSidebandRequired)));

    let reviewed_plan = plan_contiguous_with_splits(&request, &[10, 15]).expect("reviewed plan");
    assert_eq!(
        reviewed_plan
            .boundaries
            .iter()
            .map(|boundary| (boundary.layer_boundary, boundary.decision))
            .collect::<Vec<_>>(),
        vec![
            (10, BoundaryDecision::Accepted),
            (15, BoundaryDecision::Accepted)
        ]
    );
}

#[test]
fn explicit_splits_return_reasoned_boundary_decisions() {
    let request = TopologyPlanRequest {
        topology_id: "gemma-explicit".to_string(),
        model_id: "gemma4-e4b".to_string(),
        layers: dense_attention_layers(42, 10),
        nodes: nodes(3),
        family: Some(gemma4_e4b_capability(42, 2560)),
        policy: PlannerPolicy::default(),
    };

    let plan = plan_contiguous_with_splits(&request, &[12, 24]).expect("plan");

    assert_eq!(
        plan.stages
            .iter()
            .map(|stage| (stage.layer_start, stage.layer_end))
            .collect::<Vec<_>>(),
        vec![(0, 12), (12, 24), (24, 42)]
    );
    assert_eq!(
        plan.boundaries
            .iter()
            .map(|boundary| (boundary.layer_boundary, boundary.decision))
            .collect::<Vec<_>>(),
        vec![
            (12, BoundaryDecision::Rejected),
            (24, BoundaryDecision::Rejected)
        ]
    );
}

#[test]
fn infers_known_family_capabilities_from_model_identity() {
    let reviewed = reviewed_capability_records();
    assert!(reviewed.len() >= 13);

    let llama = infer_family_capability(
        "/Volumes/External/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        16,
        2048,
    )
    .expect("reviewed llama");
    assert_eq!(llama.family_id, "llama");
    assert_eq!(llama.q8_wire_validation, WireValidation::Validated);
    assert_eq!(llama.exact_state_mobility, ExactStateMobility::Accepted);

    let gemma4_e4b = infer_family_capability(
            "unsloth/gemma-4-E4B-it-GGUF@315e03409eb1cdde302488d66e586dea1e82aad1/gemma-4-E4B-it-Q4_K_M.gguf",
            42,
            2560,
        )
        .expect("reviewed gemma4 e4b");
    assert_eq!(gemma4_e4b.family_id, "gemma4_e4b");
    assert_eq!(gemma4_e4b.q8_wire_validation, WireValidation::Rejected);
    assert!(!gemma4_e4b.split_constraints.is_empty());
    assert!(!gemma4_e4b.sidebands.is_empty());

    assert_eq!(
        infer_family_capability("meshllm/gemma-4-e4b-it", 42, 2560)
            .expect("gemma")
            .family_id,
        "gemma4_e4b"
    );
    assert_eq!(
        infer_family_capability("tiiuae/Falcon-H1-1.5B", 24, 2048)
            .expect("falcon")
            .family_id,
        "falcon_h1"
    );
    assert_eq!(
        infer_family_capability("Qwen/Qwen3-Coder-Next", 48, 2048)
            .expect("qwen3next")
            .family_id,
        "qwen3next"
    );
    let rwkv6 =
        infer_family_capability("latestissue/rwkv-6-finch-1b6-gguf:Q4_K", 24, 2048).expect("rwkv6");
    assert_eq!(rwkv6.family_id, "rwkv6");
    assert_eq!(rwkv6.q8_wire_validation, WireValidation::Rejected);
    assert_eq!(
        rwkv6.exact_state_mobility,
        ExactStateMobility::RejectedTooLarge
    );
    for (identity, expected_family) in [
        ("bartowski/ai21labs_AI21-Jamba2-3B-GGUF:Q4_K_M", "jamba"),
        ("meshllm/lfm2-350m-parity-q4_k_m-gguf:Q4_K_M", "lfm2"),
        ("mradermacher/mamba-130m-hf-GGUF:Q4_K_M", "mamba"),
        ("mradermacher/mamba-2.8b-hf-GGUF:Q4_K_M", "mamba2"),
        ("Mungert/rwkv7-191M-world-GGUF:Q4_K", "rwkv7"),
    ] {
        let capability = infer_family_capability(identity, 12, 768)
            .unwrap_or_else(|| panic!("failed to infer {identity}"));
        assert_eq!(capability.family_id, expected_family, "{identity}");
        assert!(
            !capability.recurrent_ranges.is_empty(),
            "{identity} should be treated as recurrent"
        );
    }
    for (identity, expected_family) in [
        ("mradermacher/Maincoder-1B-GGUF:Q2_K", "maincoder"),
        ("LiteLLMs/OpenELM-270M-GGUF:Q2_K", "openelm"),
        (
            "RichardErkhov/smallcloudai_-_Refact-1_6B-fim-gguf:Q2_K",
            "refact",
        ),
        ("s3nh/MiniCPM-2B-dpo-fp32-GGUF:Q3_K_S", "minicpm"),
        (
            "duyntnet/MiniCPM-3B-OpenHermes-2.5-v2-imatrix-GGUF:IQ2_XXS",
            "minicpm3",
        ),
        ("mmnga-o/plamo-3-nict-2b-base-gguf:IQ3_M", "plamo3"),
        ("StatPan/42dot_LLM-PLM-1.3B_GGUF:q3_k_m", "plm"),
        (
            "bartowski/SmallThinker-3B-Preview-GGUF:IQ2_M",
            "smallthinker",
        ),
        ("bartowski/HuggingFaceTB_SmolLM3-3B-GGUF:IQ2_M", "smollm3"),
    ] {
        let capability = infer_family_capability(identity, 24, 2048)
            .unwrap_or_else(|| panic!("failed to infer {identity}"));
        assert_eq!(capability.family_id, expected_family, "{identity}");
        assert!(
            capability.recurrent_ranges.is_empty(),
            "{identity} should be treated as dense"
        );
    }
    assert_eq!(
        infer_family_capability("Qwen/Qwen3-0.6B", 28, 1024)
            .expect("qwen3")
            .family_id,
        "qwen3_dense"
    );
    let qwen2moe = infer_family_capability("mradermacher/Qwen2-1.5B-2x-MoE-GGUF:Q4_K_S", 28, 1536)
        .expect("qwen2moe");
    assert_eq!(qwen2moe.family_id, "qwen2moe");
    assert_eq!(qwen2moe.q8_wire_validation, WireValidation::Rejected);
    let qwen3moe = infer_family_capability(
        "mradermacher/Qwen3-MOE-4x0.6B-2.4B-Writing-Thunder-GGUF:Q4_K_M",
        28,
        1024,
    )
    .expect("qwen3moe");
    assert_eq!(qwen3moe.family_id, "qwen3moe");
    assert_eq!(qwen3moe.q8_wire_validation, WireValidation::Validated);
    let openai_moe =
        infer_family_capability("ggml-org/gpt-oss-20b-GGUF:gpt-oss-20b-mxfp4", 24, 2880)
            .expect("openai_moe/gpt-oss");
    assert_eq!(openai_moe.family_id, "openai_moe");
    assert_eq!(openai_moe.q8_wire_validation, WireValidation::Rejected);
    assert_eq!(
        openai_moe.exact_state_mobility,
        ExactStateMobility::Accepted
    );
    let llama4 = infer_family_capability(
        "ggml-org/Llama-4-Scout-17B-16E-Instruct-GGUF:Q4_K_M",
        48,
        5120,
    )
    .expect("llama4 package");
    assert_eq!(llama4.family_id, "llama4");
    assert_eq!(llama4.q8_wire_validation, WireValidation::Untested);
    assert_eq!(llama4.exact_state_mobility, ExactStateMobility::Untested);
    let mistral4 = infer_family_capability(
        "bartowski/mistralai_Mistral-Small-4-119B-2603-GGUF:IQ2_XXS",
        36,
        4096,
    )
    .expect("mistral4 package");
    assert_eq!(mistral4.family_id, "mistral4");
    assert_eq!(mistral4.q8_wire_validation, WireValidation::Untested);
    assert_eq!(mistral4.exact_state_mobility, ExactStateMobility::Untested);
    let qwen3_coder_package = infer_family_capability(
        "unsloth/Qwen3-Coder-480B-A35B-Instruct-GGUF:UD-Q4_K_XL",
        62,
        6144,
    )
    .expect("qwen3 coder package");
    assert_eq!(qwen3_coder_package.family_id, "qwen3moe");
    assert_eq!(
        qwen3_coder_package.q8_wire_validation,
        WireValidation::Untested
    );
    let qwen3_coder_30b =
        infer_family_capability("unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:Q4_K_M", 48, 2048)
            .expect("qwen3 coder 30b");
    assert_eq!(qwen3_coder_30b.family_id, "qwen3moe");
    let exaone_moe_package =
        infer_family_capability("LGAI-EXAONE/K-EXAONE-236B-A23B-GGUF:Q4_K_M", 49, 6144)
            .expect("exaone-moe package");
    assert_eq!(exaone_moe_package.family_id, "exaone_moe");
    assert_eq!(
        exaone_moe_package.q8_wire_validation,
        WireValidation::Untested
    );
    assert_eq!(
        exaone_moe_package.exact_state_mobility,
        ExactStateMobility::RejectedTooLarge
    );
    let gemma3n =
        infer_family_capability("lmstudio-community/gemma-3n-E2B-it-GGUF:Q4_K_M", 30, 2048)
            .expect("gemma3n");
    assert_eq!(gemma3n.family_id, "gemma3n");
    assert_eq!(gemma3n.q8_wire_validation, WireValidation::Validated);
    assert_eq!(gemma3n.exact_state_mobility, ExactStateMobility::Accepted);
    assert_eq!(gemma3n.sidebands[0].kind, SidebandKind::Gemma3nAltup);
    let qwen2vl = infer_family_capability("bartowski/Qwen2-VL-2B-Instruct-GGUF:Q4_K_M", 28, 1536)
        .expect("qwen2vl");
    assert_eq!(qwen2vl.family_id, "qwen2vl");
    assert_eq!(qwen2vl.q8_wire_validation, WireValidation::Rejected);
    assert_eq!(qwen2vl.exact_state_mobility, ExactStateMobility::Untested);
    let qwen3vl = infer_family_capability("Qwen/Qwen3-VL-2B-Instruct-GGUF:Q4_K_M", 28, 2048)
        .expect("qwen3vl");
    assert_eq!(qwen3vl.family_id, "qwen3vl");
    assert_eq!(qwen3vl.q8_wire_validation, WireValidation::Validated);
    assert_eq!(qwen3vl.exact_state_mobility, ExactStateMobility::Untested);
    let deepseek2ocr =
        infer_family_capability("ggml-org/DeepSeek-OCR-GGUF:Q8_0", 12, 1280).expect("deepseek2ocr");
    assert_eq!(deepseek2ocr.family_id, "deepseek2ocr");
    assert_eq!(deepseek2ocr.q8_wire_validation, WireValidation::Rejected);
    assert_eq!(
        deepseek2ocr.exact_state_mobility,
        ExactStateMobility::Accepted
    );
    let hunyuan_vl =
        infer_family_capability("ggml-org/HunyuanOCR-GGUF:Q8_0", 24, 1024).expect("hunyuan_vl");
    assert_eq!(hunyuan_vl.family_id, "hunyuan_vl");
    assert_eq!(hunyuan_vl.q8_wire_validation, WireValidation::Untested);
    assert_eq!(
        hunyuan_vl.exact_state_mobility,
        ExactStateMobility::Untested
    );
    let qwen3vlmoe = infer_family_capability(
        "noctrex/Qwen3-VL-30B-A3B-Instruct-1M-MXFP4_MOE-GGUF:MXFP4_MOE",
        48,
        2048,
    )
    .expect("qwen3vlmoe");
    assert_eq!(qwen3vlmoe.family_id, "qwen3vlmoe");
    assert_eq!(qwen3vlmoe.q8_wire_validation, WireValidation::Rejected);
    assert_eq!(
        qwen3vlmoe.exact_state_mobility,
        ExactStateMobility::Accepted
    );
    let apertus =
        infer_family_capability("unsloth/Apertus-8B-Instruct-2509-GGUF:UD-IQ2_M", 32, 4096)
            .expect("apertus");
    assert_eq!(apertus.family_id, "apertus");
    assert_eq!(apertus.default_wire_dtype, WireDType::F32);
    assert_eq!(apertus.q8_wire_validation, WireValidation::Rejected);
    let bitnet =
        infer_family_capability("Sarverott/bitnet_b1_58-large-Q4_K_M-GGUF:Q4_K_M", 24, 1536)
            .expect("bitnet");
    assert_eq!(bitnet.family_id, "bitnet");
    assert_eq!(bitnet.q8_wire_validation, WireValidation::Validated);
    assert_eq!(bitnet.exact_state_mobility, ExactStateMobility::Accepted);
    let plamo =
        infer_family_capability("QuantFactory/plamo-13b-GGUF:Q2_K", 40, 5120).expect("plamo");
    assert_eq!(plamo.family_id, "plamo");
    assert_eq!(plamo.q8_wire_validation, WireValidation::Validated);
    assert_eq!(plamo.exact_state_mobility, ExactStateMobility::Accepted);
    let starcoder = infer_family_capability(
        "RichardErkhov/bigcode_-_tiny_starcoder_py-gguf:Q2_K",
        20,
        768,
    )
    .expect("starcoder");
    assert_eq!(starcoder.family_id, "starcoder");
    assert_eq!(starcoder.q8_wire_validation, WireValidation::Validated);
    assert_eq!(starcoder.exact_state_mobility, ExactStateMobility::Accepted);
    let llada =
        infer_family_capability("mradermacher/LLaDA-1.5-Tiny-GGUF:Q2_K", 6, 512).expect("llada");
    assert_eq!(llada.family_id, "llada");
    assert_eq!(llada.q8_wire_validation, WireValidation::Validated);
    assert_eq!(llada.exact_state_mobility, ExactStateMobility::Untested);
    let plamo2 = infer_family_capability("mmnga/plamo-2-1b-gguf:Q4_K_M", 16, 2048).expect("plamo2");
    assert_eq!(plamo2.family_id, "plamo2");
    assert_eq!(plamo2.q8_wire_validation, WireValidation::Validated);
    assert_eq!(plamo2.exact_state_mobility, ExactStateMobility::Accepted);
    assert_eq!(plamo2.recurrent_ranges.len(), 1);
    let ernie4_5 =
        infer_family_capability("lmstudio-community/ERNIE-4.5-0.3B-GGUF:Q4_K_M", 18, 1024)
            .expect("ernie4_5");
    assert_eq!(ernie4_5.family_id, "ernie4_5");
    assert_eq!(ernie4_5.q8_wire_validation, WireValidation::Validated);
    assert_eq!(ernie4_5.exact_state_mobility, ExactStateMobility::Accepted);
    let ernie4_5_moe = infer_family_capability(
        "lmstudio-community/ERNIE-4.5-21B-A3B-PT-GGUF:Q4_K_M",
        28,
        2560,
    )
    .expect("ernie4_5_moe");
    assert_eq!(ernie4_5_moe.family_id, "ernie4_5_moe");
    assert_eq!(ernie4_5_moe.q8_wire_validation, WireValidation::Validated);
    assert_eq!(
        ernie4_5_moe.exact_state_mobility,
        ExactStateMobility::Accepted
    );
    let qwen =
        infer_family_capability("zhangtao103239/Qwen-1.8B-GGUF:q5_k_m", 24, 2048).expect("qwen");
    assert_eq!(qwen.family_id, "qwen");
    assert_eq!(qwen.q8_wire_validation, WireValidation::Rejected);
    assert_eq!(qwen.exact_state_mobility, ExactStateMobility::Accepted);
    let jais = infer_family_capability("mradermacher/Jais-family-256m-GGUF:Q4_K_M", 14, 1088)
        .expect("jais");
    assert_eq!(jais.family_id, "jais");
    assert_eq!(jais.q8_wire_validation, WireValidation::Rejected);
    assert_eq!(jais.exact_state_mobility, ExactStateMobility::Accepted);
    let jais2 =
        infer_family_capability("mradermacher/JAIS2-IT-0.3-GGUF:Q4_K_M", 32, 3328).expect("jais2");
    assert_eq!(jais2.family_id, "jais2");
    assert_eq!(jais2.q8_wire_validation, WireValidation::Rejected);
    assert_eq!(jais2.exact_state_mobility, ExactStateMobility::Accepted);
    let nemotron_h =
        infer_family_capability("nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF:Q4_K_M", 42, 3136)
            .expect("nemotron_h");
    assert_eq!(nemotron_h.family_id, "nemotron_h");
    assert_eq!(nemotron_h.q8_wire_validation, WireValidation::Rejected);
    assert_eq!(
        nemotron_h.exact_state_mobility,
        ExactStateMobility::RejectedTooLarge
    );
    assert_eq!(nemotron_h.recurrent_ranges.len(), 1);
    let nemotron_h_moe = infer_family_capability(
        "lmstudio-community/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-GGUF:Q4_K_M",
        52,
        2688,
    )
    .expect("nemotron_h_moe package");
    assert_eq!(nemotron_h_moe.family_id, "nemotron_h_moe");
    assert_eq!(nemotron_h_moe.q8_wire_validation, WireValidation::Untested);
    assert_eq!(
        nemotron_h_moe.exact_state_mobility,
        ExactStateMobility::RejectedTooLarge
    );
    assert_eq!(nemotron_h_moe.recurrent_ranges.len(), 1);
    let llada_moe = infer_family_capability(
        "mradermacher/LLaDA-MoE-7B-A1B-Instruct-i1-GGUF:IQ2_XS",
        16,
        2048,
    )
    .expect("llada_moe");
    assert_eq!(llada_moe.family_id, "llada_moe");
    assert_eq!(llada_moe.q8_wire_validation, WireValidation::Rejected);
    assert_eq!(llada_moe.exact_state_mobility, ExactStateMobility::Untested);
    let dream = infer_family_capability("mradermacher/DreamOn-v0-7B-i1-GGUF:IQ2_XS", 28, 3584)
        .expect("dream");
    assert_eq!(dream.family_id, "dream");
    assert_eq!(dream.q8_wire_validation, WireValidation::Validated);
    assert_eq!(dream.exact_state_mobility, ExactStateMobility::Untested);
    let nemotron = infer_family_capability(
        "mradermacher/nemotron-3-8b-chat-4k-sft-hf-i1-GGUF:IQ2_XS",
        32,
        4096,
    )
    .expect("nemotron");
    assert_eq!(nemotron.family_id, "nemotron");
    assert_eq!(nemotron.q8_wire_validation, WireValidation::Rejected);
    assert_eq!(nemotron.exact_state_mobility, ExactStateMobility::Accepted);
    let seed_oss = infer_family_capability(
        "lmstudio-community/Seed-OSS-36B-Instruct-GGUF:Q4_K_M",
        64,
        5120,
    )
    .expect("seed_oss package");
    assert_eq!(seed_oss.family_id, "seed_oss");
    assert_eq!(seed_oss.q8_wire_validation, WireValidation::Untested);
    assert_eq!(seed_oss.exact_state_mobility, ExactStateMobility::Untested);
    let lfm2moe =
        infer_family_capability("noctrex/LFM2-8B-A1B-MXFP4_MOE-GGUF", 24, 2048).expect("lfm2moe");
    assert_eq!(lfm2moe.family_id, "lfm2moe");
    assert_eq!(lfm2moe.q8_wire_validation, WireValidation::Validated);
    assert_eq!(lfm2moe.exact_state_mobility, ExactStateMobility::Accepted);
    assert_eq!(lfm2moe.recurrent_ranges.len(), 1);
    let kimi = infer_family_capability(
        "bartowski/moonshotai_Kimi-Linear-48B-A3B-Instruct-GGUF:IQ2_XXS",
        27,
        2304,
    )
    .expect("kimi_linear");
    assert_eq!(kimi.family_id, "kimi_linear");
    assert_eq!(kimi.q8_wire_validation, WireValidation::Validated);
    assert_eq!(
        kimi.exact_state_mobility,
        ExactStateMobility::RejectedTooLarge
    );
    assert_eq!(
        kimi.recurrent_ranges,
        vec![
            LayerRange { start: 0, end: 3 },
            LayerRange { start: 4, end: 7 },
            LayerRange { start: 8, end: 11 },
            LayerRange { start: 12, end: 15 },
            LayerRange { start: 16, end: 19 },
            LayerRange { start: 20, end: 23 },
            LayerRange { start: 24, end: 26 },
        ]
    );
    assert_eq!(
        infer_family_capability("meta/Llama-3.2-1B-Instruct", 16, 2048)
            .expect("llama")
            .family_id,
        "llama"
    );
    assert_eq!(
        infer_family_capability("DeepSeek-Coder-V2-Lite-Instruct", 27, 2048)
            .expect("deepseek")
            .family_id,
        "deepseek2"
    );
    let deepseek3 = infer_family_capability("unsloth/DeepSeek-V3.2-GGUF:UD-Q4_K_XL", 61, 7168)
        .expect("reviewed deepseek3");
    assert_eq!(deepseek3.family_id, "deepseek3");
    assert_eq!(deepseek3.q8_wire_validation, WireValidation::Untested);
    assert_eq!(deepseek3.exact_state_mobility, ExactStateMobility::Accepted);
    let qwen35moe = infer_family_capability("unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_XL", 40, 2048)
        .expect("reviewed qwen35moe");
    assert_eq!(qwen35moe.family_id, "qwen35moe");
    assert_eq!(qwen35moe.q8_wire_validation, WireValidation::Untested);
    assert_eq!(
        qwen35moe.exact_state_mobility,
        ExactStateMobility::RejectedTooLarge
    );
    assert_eq!(qwen35moe.recurrent_ranges.len(), 1);
    let generic_deepseek3 = infer_family_capability("unsloth/DeepSeek-V3.2-GGUF:Q4_K_M", 61, 7168)
        .expect("generic deepseek3");
    assert_eq!(generic_deepseek3.family_id, "deepseek3");
    assert_eq!(
        generic_deepseek3.exact_state_mobility,
        ExactStateMobility::Untested
    );
    assert_eq!(
        infer_family_capability("unsloth/GLM-4.7-Flash-GGUF", 47, 2048)
            .expect("glm47")
            .family_id,
        "glm47_flash"
    );
    assert_eq!(
        infer_family_capability("meshllm/glm-4-9b-0414", 40, 4096)
            .expect("glm4")
            .family_id,
        "glm4"
    );
    assert_eq!(
        infer_family_capability("bartowski/gemma-2-2b-it", 26, 2304)
            .expect("gemma2")
            .family_id,
        "gemma2"
    );
    assert_eq!(
        infer_family_capability("ggml-org/gemma-3-1b-it", 26, 1152)
            .expect("gemma3")
            .family_id,
        "gemma3"
    );
    let gemma =
        infer_family_capability("ggml-org/gemma-3-270m-it-GGUF:Q8_0", 18, 640).expect("gemma");
    assert_eq!(gemma.family_id, "gemma");
    assert_eq!(gemma.default_wire_dtype, WireDType::F32);
    assert_eq!(gemma.q8_wire_validation, WireValidation::Rejected);
    assert_eq!(
        infer_family_capability("google-gemma-4-26B-A4B-it", 30, 2816)
            .expect("gemma4a4b")
            .family_id,
        "gemma4_a4b"
    );
    assert_eq!(
        infer_family_capability("meshllm/olmo-7b-instruct", 32, 4096)
            .expect("olmo")
            .family_id,
        "olmo"
    );
    assert_eq!(
        infer_family_capability("unsloth/MiniMax-M2.7-GGUF", 62, 3072)
            .expect("minimax")
            .family_id,
        "minimax_m27"
    );
    assert!(infer_family_capability("unknown", 1, 1).is_none());
}

#[test]
fn every_stage_runtime_llama_architecture_has_family_inference() {
    for expected in STAGE_RUNTIME_LLAMA_FAMILY_EXPECTATIONS {
        let capability = infer_family_capability(expected.llama_architecture, 12, 768)
            .unwrap_or_else(|| {
                panic!(
                    "missing family inference for {}",
                    expected.llama_architecture
                )
            });

        assert_eq!(
            capability.family_id, expected.family_id,
            "{}",
            expected.llama_architecture
        );
        assert_eq!(
            !capability.recurrent_ranges.is_empty(),
            expected.recurrent_or_hybrid,
            "{}",
            expected.llama_architecture
        );
    }
}

#[derive(Debug, Deserialize)]
struct ParityCandidateManifest {
    candidates: Vec<ParityCandidate>,
}

#[derive(Debug, Deserialize)]
struct ParityCandidate {
    llama_model: String,
    status: String,
}

#[test]
fn parity_candidate_manifest_covers_stage_runtime_architectures() {
    let manifest: ParityCandidateManifest = serde_json::from_str(include_str!(
        "../../../docs/skippy/llama-parity-candidates.json"
    ))
    .expect("parity candidate manifest must parse");
    let candidates: Vec<String> = manifest
        .candidates
        .into_iter()
        .map(|candidate| compact_identity(&candidate.llama_model))
        .collect();

    for expected in STAGE_RUNTIME_LLAMA_FAMILY_EXPECTATIONS {
        assert!(
            candidates.contains(&compact_identity(expected.llama_architecture)),
            "missing parity candidate row for {}",
            expected.llama_architecture
        );
    }
}

#[test]
fn parity_candidate_manifest_uses_known_statuses() {
    let manifest: ParityCandidateManifest = serde_json::from_str(include_str!(
        "../../../docs/skippy/llama-parity-candidates.json"
    ))
    .expect("parity candidate manifest must parse");

    for candidate in manifest.candidates {
        assert!(
            matches!(
                candidate.status.as_str(),
                "candidate"
                    | "candidate_stateful"
                    | "candidate_multimodal"
                    | "certified"
                    | "certified_package_only"
                    | "implementation_base"
                    | "needs_candidate"
                    | "needs_runtime_slice_support"
                    | "no_public_gguf_candidate"
                    | "non_causal_aux"
                    | "package_or_remote_only"
            ),
            "{} has unknown status {}",
            candidate.llama_model,
            candidate.status
        );
    }
}

#[test]
fn reviewed_supported_families_smoke_plan_with_expected_policy_signals() {
    let reviewed = reviewed_capability_records();
    assert!(reviewed.len() >= 13);

    for record in reviewed {
        let identity = record
            .model_id
            .as_deref()
            .or(record.canonical_ref.as_deref())
            .or(record.source_repo.as_deref())
            .or(record.distribution_id.as_deref())
            .expect("reviewed family record has an identity");
        let expected = record.capability;
        let family =
            infer_family_capability(identity, expected.layer_count, expected.activation_width)
                .unwrap_or_else(|| panic!("failed to infer reviewed family for {identity}"));

        assert_eq!(
            family.family_id, expected.family_id,
            "family id mismatch for {identity}"
        );
        assert_eq!(
            family.layer_count, expected.layer_count,
            "layer count mismatch for {identity}"
        );
        assert_eq!(
            family.activation_width, expected.activation_width,
            "activation width mismatch for {identity}"
        );

        let request = TopologyPlanRequest {
            topology_id: format!("smoke-{}", family.family_id),
            model_id: identity.to_string(),
            layers: dense_attention_layers(family.layer_count, 10),
            nodes: nodes(2),
            family: Some(family.clone()),
            policy: PlannerPolicy::default(),
        };
        let plan = plan_even_contiguous(&request)
            .unwrap_or_else(|error| panic!("failed to plan {identity}: {error}"));

        assert_eq!(plan.family_id.as_deref(), Some(family.family_id.as_str()));
        assert_eq!(
            plan.stages.len(),
            2,
            "unexpected stage count for {identity}"
        );
        assert_eq!(
            plan.boundaries.len(),
            1,
            "unexpected boundary count for {identity}"
        );
        assert_eq!(
            plan.boundaries[0].wire_dtype, family.default_wire_dtype,
            "supported family default wire mismatch for {identity}"
        );
        if family.default_wire_dtype == WireDType::F16 {
            assert!(
                plan.boundaries[0]
                    .reason_codes
                    .contains(&PlanReasonCode::DefaultWireDtypeF16),
                "missing f16 reason for {identity}"
            );
        }

        match family.q8_wire_validation {
            WireValidation::Validated => assert!(
                plan.boundaries[0]
                    .reason_codes
                    .contains(&PlanReasonCode::Q8WireValidated),
                "missing q8 validated signal for {identity}"
            ),
            WireValidation::Rejected => assert!(
                plan.boundaries[0]
                    .reason_codes
                    .contains(&PlanReasonCode::Q8WireRejected),
                "missing q8 rejected signal for {identity}"
            ),
            WireValidation::Untested => {}
        }

        if family.recurrent_ranges.is_empty() {
            assert!(
                plan.stages.iter().all(|stage| {
                    stage.migration_policy != MigrationPolicy::StickyRecurrentOwner
                }),
                "dense family unexpectedly sticky: {identity}"
            );
        } else {
            assert!(
                plan.stages.iter().any(|stage| {
                    stage.migration_policy == MigrationPolicy::StickyRecurrentOwner
                }),
                "recurrent family was not sticky: {identity}"
            );
            assert!(
                plan.boundaries[0]
                    .reason_codes
                    .contains(&PlanReasonCode::RecurrentOwnerSticky),
                "missing recurrent boundary signal for {identity}"
            );
            match family.exact_state_mobility {
                ExactStateMobility::Accepted => assert!(
                    plan.diagnostics.iter().any(|diagnostic| {
                        diagnostic.code == PlanReasonCode::ExactStateMobilityAccepted
                    }),
                    "missing accepted recurrent state mobility diagnostic for {identity}"
                ),
                ExactStateMobility::RejectedTooLarge => assert!(
                    plan.diagnostics.iter().any(|diagnostic| {
                        diagnostic.code == PlanReasonCode::ExactStateMobilityRejected
                    }),
                    "missing rejected recurrent state mobility diagnostic for {identity}"
                ),
                ExactStateMobility::Untested => {}
            }
        }

        if !family.sidebands.is_empty() {
            let sideband_reason_codes: Vec<_> = family
                .sidebands
                .iter()
                .map(|sideband| match sideband.kind {
                    SidebandKind::TokenIds => PlanReasonCode::TokenSidebandRequired,
                    SidebandKind::Rwkv7VFirst | SidebandKind::Gemma3nAltup => {
                        PlanReasonCode::ActivationSidebandRequired
                    }
                })
                .collect();
            assert!(
                sideband_reason_codes
                    .iter()
                    .any(|code| plan.boundaries[0].reason_codes.contains(code)),
                "missing sideband signal for {identity}"
            );
        }
    }
}
