use super::*;

fn nodes(count: u32) -> Vec<NodeSpec> {
    (0..count)
        .map(|index| NodeSpec {
            node_id: format!("node-{index}"),
            cached_slice_bytes: 0,
            vram_bytes: 0,
        })
        .collect()
}

fn weighted_node(node_id: &str, vram_bytes: u64) -> NodeSpec {
    NodeSpec {
        node_id: node_id.to_string(),
        cached_slice_bytes: 0,
        vram_bytes,
    }
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
            assert!(
                plan.diagnostics.iter().any(|diagnostic| {
                    diagnostic.code == PlanReasonCode::ExactStateMobilityRejected
                }),
                "missing recurrent state mobility diagnostic for {identity}"
            );
        }

        if !family.sidebands.is_empty() {
            let sideband_reason_codes: Vec<_> = family
                .sidebands
                .iter()
                .map(|sideband| match sideband.kind {
                    SidebandKind::TokenIds => PlanReasonCode::TokenSidebandRequired,
                    SidebandKind::Rwkv7VFirst => PlanReasonCode::ActivationSidebandRequired,
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
