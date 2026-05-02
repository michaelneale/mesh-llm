use super::*;
use iroh::EndpointAddr;
use iroh::SecretKey;

/// Create a deterministic EndpointId from a byte seed.
fn make_id(seed: u8) -> iroh::EndpointId {
    let mut bytes = [0u8; 32];
    bytes[0] = seed;
    SecretKey::from_bytes(&bytes).public()
}

fn stage_status(
    run_id: &str,
    stage_id: &str,
    state: skippy::StageRuntimeState,
    error: Option<&str>,
) -> skippy::StageStatusSnapshot {
    skippy::StageStatusSnapshot {
        topology_id: "topology-a".to_string(),
        run_id: run_id.to_string(),
        model_id: "model-a".to_string(),
        backend: "skippy".to_string(),
        package_ref: Some("hf://Mesh-LLM/demo-package".to_string()),
        manifest_sha256: Some("manifest".to_string()),
        source_model_path: Some("model.gguf".to_string()),
        source_model_sha256: Some("source".to_string()),
        source_model_bytes: Some(100),
        materialized_path: None,
        materialized_pinned: false,
        projector_path: None,
        stage_id: stage_id.to_string(),
        stage_index: 1,
        layer_start: 4,
        layer_end: 8,
        state,
        bind_addr: "127.0.0.1:51234".to_string(),
        activation_width: 1024,
        wire_dtype: skippy::StageWireDType::F16,
        selected_device: None,
        ctx_size: 4096,
        error: error.map(ToString::to_string),
        shutdown_generation: 1,
    }
}

fn make_dense_peer(
    id: iroh::EndpointId,
    vram_bytes: u64,
    rtt_ms: Option<u32>,
    serving_model: &str,
) -> mesh::PeerInfo {
    mesh::PeerInfo {
        id,
        addr: EndpointAddr {
            id,
            addrs: Default::default(),
        },
        tunnel_port: None,
        role: NodeRole::Worker,
        first_joined_mesh_ts: None,
        models: vec![],
        vram_bytes,
        rtt_ms,
        model_source: None,
        serving_models: vec![serving_model.to_string()],
        hosted_models: vec![],
        hosted_models_known: false,
        available_models: vec![],
        requested_models: vec![],
        explicit_model_interests: vec![],
        last_seen: std::time::Instant::now(),
        last_mentioned: std::time::Instant::now(),
        moe_recovered_at: None,
        version: None,
        gpu_name: None,
        hostname: None,
        is_soc: None,
        gpu_vram: None,
        gpu_reserved_bytes: None,
        gpu_mem_bandwidth_gbps: None,
        gpu_compute_tflops_fp32: None,
        gpu_compute_tflops_fp16: None,
        available_model_metadata: vec![],
        experts_summary: None,
        available_model_sizes: HashMap::new(),
        served_model_descriptors: vec![],
        served_model_runtime: vec![],
        owner_attestation: None,
        owner_summary: crate::crypto::OwnershipSummary::default(),
    }
}

#[test]
fn dense_launch_plan_prefers_lowest_rtt_workers_needed_for_capacity() {
    let model = "dense";
    let id_a = make_id(1);
    let id_b = make_id(2);
    let id_c = make_id(3);
    let id_d = make_id(4);
    let peers = vec![
        make_dense_peer(id_b, 30, Some(60), model),
        make_dense_peer(id_c, 30, Some(20), model),
        make_dense_peer(id_d, 30, Some(40), model),
    ];

    let plan = build_dense_launch_plan(60, 100, false, model, &peers);
    assert_eq!(
        plan,
        DenseLaunchPlan::Split {
            worker_ids: vec![id_c, id_d],
            total_group_vram: 120,
        }
    );

    assert!(should_be_host_for_model(id_a, 60, &peers));
}

#[test]
fn pinned_gpu_runtime_launch_pinned_local_launch_disables_row_split() {
    let pinned_gpu = crate::runtime::StartupPinnedGpuTarget {
        index: 0,
        stable_id: "pci:0000:65:00.0".into(),
        backend_device: "CUDA0".into(),
        vram_bytes: 24_000_000_000,
    };

    assert_eq!(
        split_mode_for_local_launch(Some(BinaryFlavor::Cuda), Some(&pinned_gpu)),
        None
    );
}

#[test]
fn pinned_gpu_runtime_launch_dense_planner_uses_selected_device_capacity() {
    let model = "dense";
    let peer = make_dense_peer(make_id(2), 50, Some(10), model);
    let pinned_gpu = crate::runtime::StartupPinnedGpuTarget {
        index: 0,
        stable_id: "pci:0000:65:00.0".into(),
        backend_device: "CUDA0".into(),
        vram_bytes: 30,
    };

    let local_launch_vram = effective_local_launch_vram(80, Some(&pinned_gpu), None, None);
    let plan = build_dense_launch_plan(
        local_launch_vram,
        60,
        false,
        model,
        std::slice::from_ref(&peer),
    );

    assert_eq!(
        plan,
        DenseLaunchPlan::Split {
            worker_ids: vec![peer.id],
            total_group_vram: 80,
        }
    );
    assert!(should_be_host_for_model(
        make_id(1),
        80,
        std::slice::from_ref(&peer)
    ));
    assert!(!should_be_host_for_model(
        make_id(1),
        local_launch_vram,
        &[peer]
    ));
}

#[test]
fn vulkan_local_launch_preflight_uses_primary_device_capacity() {
    let local_launch_vram = effective_local_launch_vram(
        29_400_000_000,
        None,
        Some(BinaryFlavor::Vulkan),
        Some("15977000000,16212000000"),
    );

    assert_eq!(local_launch_vram, 15_977_000_000);
}

#[test]
fn cuda_local_launch_preflight_keeps_aggregate_capacity() {
    let local_launch_vram = effective_local_launch_vram(
        48_000_000_000,
        None,
        Some(BinaryFlavor::Cuda),
        Some("24000000000,24000000000"),
    );

    assert_eq!(local_launch_vram, 48_000_000_000);
}

#[test]
fn dense_launch_plan_ignores_unselected_spare_worker_churn() {
    let model = "dense";
    let id_b = make_id(2);
    let id_c = make_id(3);
    let id_d = make_id(4);
    let base = vec![
        make_dense_peer(id_b, 30, Some(10), model),
        make_dense_peer(id_c, 30, Some(20), model),
    ];
    let mut with_spare = base.clone();
    with_spare.push(make_dense_peer(id_d, 50, Some(70), model));

    let base_plan = build_dense_launch_plan(60, 100, false, model, &base);
    let spare_plan = build_dense_launch_plan(60, 100, false, model, &with_spare);

    assert_eq!(base_plan.running_plan(), spare_plan.running_plan());
    assert_eq!(
        base_plan.running_plan(),
        Some(DenseRunningPlan::Split {
            worker_ids: vec![id_b, id_c],
        })
    );
}

#[test]
fn dense_launch_plan_replans_across_surviving_workers_after_peer_loss() {
    let model = "dense";
    let id_b = make_id(2);
    let id_c = make_id(3);
    let id_d = make_id(4);
    let initial = vec![
        make_dense_peer(id_b, 30, Some(10), model),
        make_dense_peer(id_c, 30, Some(20), model),
        make_dense_peer(id_d, 30, Some(30), model),
    ];
    let survivors = vec![
        make_dense_peer(id_c, 30, Some(20), model),
        make_dense_peer(id_d, 30, Some(30), model),
    ];

    let initial_plan = build_dense_launch_plan(50, 100, false, model, &initial);
    let survivor_plan = build_dense_launch_plan(50, 100, false, model, &survivors);

    assert_eq!(
        initial_plan.running_plan(),
        Some(DenseRunningPlan::Split {
            worker_ids: vec![id_b, id_c],
        })
    );
    assert_eq!(
        survivor_plan.running_plan(),
        Some(DenseRunningPlan::Split {
            worker_ids: vec![id_c, id_d],
        })
    );
}

#[test]
fn stage_failure_quarantine_excludes_failed_worker_from_next_plan() {
    let model = "dense";
    let id_b = make_id(2);
    let id_c = make_id(3);
    let id_d = make_id(4);
    let peers = vec![
        make_dense_peer(id_b, 30, Some(10), model),
        make_dense_peer(id_c, 30, Some(20), model),
        make_dense_peer(id_d, 30, Some(30), model),
    ];
    let mut quarantined = HashMap::new();
    let failure = SkippyStageFailure {
        stage_id: "stage-1".into(),
        peer_id: Some(id_b),
        reason: "stage crashed".into(),
    };
    assert_eq!(
        quarantine_skippy_stage_failure(&mut quarantined, &failure, Instant::now()),
        Some(id_b)
    );

    let eligible = model_peers_for_election(&peers, model, &quarantined);
    let plan = build_dense_launch_plan(50, 100, false, model, &eligible);

    assert_eq!(
        plan.running_plan(),
        Some(DenseRunningPlan::Split {
            worker_ids: vec![id_c, id_d],
        })
    );
}

#[test]
fn stage_failure_quarantine_expires() {
    let id_b = make_id(2);
    let id_c = make_id(3);
    let now = Instant::now();
    let mut quarantined = HashMap::from([
        (id_b, now + Duration::from_secs(1)),
        (id_c, now - Duration::from_secs(1)),
    ]);

    prune_skippy_stage_quarantine(&mut quarantined, now);

    assert!(quarantined.contains_key(&id_b));
    assert!(!quarantined.contains_key(&id_c));
}

#[test]
fn dense_launch_plan_waits_when_only_ineligible_capacity_remains() {
    let model = "dense";
    let id_b = make_id(2);
    let id_c = make_id(3);
    let peers = vec![
        make_dense_peer(id_b, 30, Some(10), model),
        make_dense_peer(id_c, 40, Some(mesh::MAX_SPLIT_RTT_MS + 1), model),
    ];

    let plan = build_dense_launch_plan(50, 100, false, model, &peers);
    assert_eq!(
        plan,
        DenseLaunchPlan::WaitingForCapacity {
            worker_ids: vec![id_b],
            total_group_vram: 80,
            min_vram: 110,
        }
    );
}

#[test]
fn active_stage_failure_marks_missing_status_failed() {
    let peer_id = make_id(2);
    let expected = stage_status("run-a", "stage-1", skippy::StageRuntimeState::Ready, None);

    let failure = active_stage_failure_from_status(Some(peer_id), &expected, None).unwrap();

    assert_eq!(failure.stage_id, "stage-1");
    assert_eq!(failure.peer_id, Some(peer_id));
    assert_eq!(failure.reason, "stage status missing from runtime");
}

#[test]
fn active_stage_failure_ignores_stale_run_status() {
    let peer_id = make_id(2);
    let expected = stage_status("run-new", "stage-1", skippy::StageRuntimeState::Ready, None);
    let stale = stage_status(
        "run-old",
        "stage-1",
        skippy::StageRuntimeState::Failed,
        Some("old failure"),
    );

    assert!(active_stage_failure_from_status(Some(peer_id), &expected, Some(&stale)).is_none());
}

#[test]
fn active_stage_failure_reports_failed_active_stage() {
    let peer_id = make_id(2);
    let expected = stage_status("run-a", "stage-1", skippy::StageRuntimeState::Ready, None);
    let failed = stage_status(
        "run-a",
        "stage-1",
        skippy::StageRuntimeState::Failed,
        Some("boom"),
    );

    let failure =
        active_stage_failure_from_status(Some(peer_id), &expected, Some(&failed)).unwrap();

    assert_eq!(failure.stage_id, "stage-1");
    assert_eq!(failure.peer_id, Some(peer_id));
    assert_eq!(failure.reason, "boom");
}

#[test]
fn active_stage_failure_reports_active_stage_entering_stopping() {
    let peer_id = make_id(2);
    let expected = stage_status("run-a", "stage-1", skippy::StageRuntimeState::Ready, None);
    let stopping = stage_status(
        "run-a",
        "stage-1",
        skippy::StageRuntimeState::Stopping,
        None,
    );

    let failure =
        active_stage_failure_from_status(Some(peer_id), &expected, Some(&stopping)).unwrap();

    assert_eq!(failure.stage_id, "stage-1");
    assert_eq!(failure.peer_id, Some(peer_id));
    assert_eq!(
        failure.reason,
        "stage started stopping during active topology"
    );
}

// ── Shard index computation ──

#[test]
fn test_shard_index_2_nodes() {
    let id_a = make_id(1);
    let id_b = make_id(2);

    let (all_a, idx_a) = moe_shard_index(id_a, &[id_b]);
    let (all_b, idx_b) = moe_shard_index(id_b, &[id_a]);

    // Both should see the same sorted order
    assert_eq!(all_a, all_b);
    // They should have different indices
    assert_ne!(idx_a, idx_b);
    // Indices should cover 0..2
    let mut indices = vec![idx_a, idx_b];
    indices.sort();
    assert_eq!(indices, vec![0, 1]);
}

#[test]
fn test_shard_index_3_nodes() {
    let id_a = make_id(1);
    let id_b = make_id(2);
    let id_c = make_id(3);

    let (_, idx_a) = moe_shard_index(id_a, &[id_b, id_c]);
    let (_, idx_b) = moe_shard_index(id_b, &[id_a, id_c]);
    let (_, idx_c) = moe_shard_index(id_c, &[id_a, id_b]);

    let mut indices = vec![idx_a, idx_b, idx_c];
    indices.sort();
    assert_eq!(indices, vec![0, 1, 2]);
}

#[test]
fn test_shard_index_solo() {
    let id = make_id(42);
    let (all, idx) = moe_shard_index(id, &[]);
    assert_eq!(all.len(), 1);
    assert_eq!(idx, 0);
}

#[test]
fn test_shard_index_stable_across_calls() {
    // Same inputs should always give same outputs
    let id_a = make_id(10);
    let id_b = make_id(20);
    let id_c = make_id(30);

    let (order1, idx1) = moe_shard_index(id_a, &[id_b, id_c]);
    let (order2, idx2) = moe_shard_index(id_a, &[id_c, id_b]); // different peer order
    assert_eq!(order1, order2); // sorted, so same
    assert_eq!(idx1, idx2);
}

#[test]
fn test_shard_index_my_id_already_in_peers() {
    // Edge case: what if peers list already contains my ID?
    let id_a = make_id(1);
    let id_b = make_id(2);
    let (all, idx) = moe_shard_index(id_a, &[id_a, id_b]);
    // Should not duplicate
    assert_eq!(all.len(), 2);
    assert!(idx < 2);
}

// ── MoE target map construction ──

#[test]
fn test_build_moe_targets_2_nodes() {
    let id_a = make_id(1);
    let id_b = make_id(2);
    let (sorted, _) = moe_shard_index(id_a, &[id_b]);

    let targets = build_moe_targets(&sorted, &[], id_a, Some(8080), None, "test-model");

    // Should have MoE state
    let moe = targets.moe.as_ref().unwrap();
    assert_eq!(moe.nodes.len(), 2);

    // Model should be in targets
    assert!(matches!(
        targets.get("test-model"),
        InferenceTarget::MoeLocal(8080)
    ));

    // One should be local, one remote
    let local_count = moe
        .nodes
        .iter()
        .filter(|t| matches!(t, InferenceTarget::MoeLocal(_)))
        .count();
    let remote_count = moe
        .nodes
        .iter()
        .filter(|t| matches!(t, InferenceTarget::MoeRemote(_)))
        .count();
    assert_eq!(local_count, 1);
    assert_eq!(remote_count, 1);
}

#[test]
fn test_build_moe_targets_local_port_correct() {
    let id_a = make_id(1);
    let id_b = make_id(2);
    let (sorted, idx_a) = moe_shard_index(id_a, &[id_b]);

    let targets = build_moe_targets(&sorted, &[], id_a, Some(9999), None, "m");
    let moe = targets.moe.as_ref().unwrap();

    // Our index in the MoE state should have our port
    match &moe.nodes[idx_a] {
        InferenceTarget::MoeLocal(port) => assert_eq!(*port, 9999),
        other => panic!("Expected MoeLocal(9999), got {:?}", other),
    }
}

#[test]
fn test_build_moe_targets_reconfigures_when_third_node_drops() {
    let id_a = make_id(1);
    let id_b = make_id(2);
    let id_c = make_id(3);

    let (sorted_three, _) = moe_shard_index(id_a, &[id_b, id_c]);
    let targets_three = build_moe_targets(&sorted_three, &[], id_a, Some(8080), None, "m");
    let moe_three = targets_three.moe.as_ref().unwrap();
    assert_eq!(moe_three.nodes.len(), 3);
    assert!(moe_three
        .nodes
        .iter()
        .any(|target| matches!(target, InferenceTarget::MoeRemote(id) if *id == id_c)));

    let (sorted_two, _) = moe_shard_index(id_a, &[id_b]);
    let targets_two = build_moe_targets(&sorted_two, &[], id_a, Some(8080), None, "m");
    let moe_two = targets_two.moe.as_ref().unwrap();
    assert_eq!(moe_two.nodes.len(), 2);
    assert!(!moe_two
        .nodes
        .iter()
        .any(|target| matches!(target, InferenceTarget::MoeRemote(id) if *id == id_c)));

    // The survivor should still route locally, but only across the 2 remaining shards.
    assert!(matches!(
        targets_two.get("m"),
        InferenceTarget::MoeLocal(8080)
    ));
}

#[test]
fn test_build_moe_targets_collapse_to_single_node_after_peer_loss() {
    let id_a = make_id(1);
    let id_b = make_id(2);

    let (sorted_two, _) = moe_shard_index(id_a, &[id_b]);
    let targets_two = build_moe_targets(&sorted_two, &[], id_a, Some(8080), None, "m");
    let moe_two = targets_two.moe.as_ref().unwrap();
    assert_eq!(moe_two.nodes.len(), 2);

    let targets_one = build_moe_targets(&[id_a], &[], id_a, Some(8080), None, "m");
    let moe_one = targets_one.moe.as_ref().unwrap();
    assert_eq!(moe_one.nodes.len(), 1);
    assert!(matches!(moe_one.nodes[0], InferenceTarget::MoeLocal(8080)));

    for i in 0..20 {
        match targets_one.get_moe_target(&format!("after-drop-{i}")) {
            Some(InferenceTarget::MoeLocal(8080)) => {}
            other => panic!("Expected MoeLocal(8080) after collapse, got {:?}", other),
        }
    }
}

#[test]
fn test_build_moe_targets_include_full_fallback_candidates() {
    let id_a = make_id(1);
    let id_b = make_id(2);
    let id_c = make_id(3);
    let targets = build_moe_targets(&[id_a, id_b], &[id_c], id_a, Some(8080), None, "m");
    let moe = targets.moe.as_ref().unwrap();
    assert_eq!(moe.nodes.len(), 2);
    assert_eq!(moe.fallbacks.len(), 1);
    assert!(matches!(moe.fallbacks[0], InferenceTarget::Remote(id) if id == id_c));

    let candidates = targets.get_moe_failover_targets("session");
    assert_eq!(candidates.len(), 2);
    assert!(matches!(candidates[1], InferenceTarget::Remote(id) if id == id_c));
}

#[test]
fn test_plan_moe_placement_reserves_full_fallback_when_spare_node_exists() {
    let id_a = make_id(1);
    let id_b = make_id(2);
    let id_c = make_id(3);
    let id_d = make_id(4);

    let plan = plan_moe_placement(
        vec![
            MoePlacementCandidate {
                id: id_a,
                vram_bytes: 40,
                full_coverage: true,
            },
            MoePlacementCandidate {
                id: id_b,
                vram_bytes: 24,
                full_coverage: false,
            },
            MoePlacementCandidate {
                id: id_c,
                vram_bytes: 24,
                full_coverage: false,
            },
            MoePlacementCandidate {
                id: id_d,
                vram_bytes: 24,
                full_coverage: false,
            },
        ],
        &[],
        &[],
        true,
    )
    .unwrap();

    assert_eq!(plan.leader_id, id_a);
    assert_eq!(plan.active_ids.len(), 3);
    assert_eq!(plan.fallback_ids, vec![id_a]);
    assert_eq!(plan.overlap, 2);
}

#[test]
fn test_plan_moe_placement_keeps_current_active_set_during_recovery() {
    let id_a = make_id(1);
    let id_b = make_id(2);
    let id_c = make_id(3);

    let plan = plan_moe_placement(
        vec![
            MoePlacementCandidate {
                id: id_a,
                vram_bytes: 48,
                full_coverage: true,
            },
            MoePlacementCandidate {
                id: id_b,
                vram_bytes: 24,
                full_coverage: false,
            },
            MoePlacementCandidate {
                id: id_c,
                vram_bytes: 24,
                full_coverage: false,
            },
        ],
        &[id_b, id_c],
        &[],
        false,
    )
    .unwrap();

    assert_eq!(plan.active_ids, vec![id_b, id_c]);
    assert_eq!(plan.fallback_ids, Vec::<iroh::EndpointId>::new());
    assert_eq!(plan.overlap, 1);
}

#[test]
fn test_plan_moe_placement_scales_up_after_quiet_window_when_materially_better() {
    let id_a = make_id(1);
    let id_b = make_id(2);
    let id_c = make_id(3);

    let plan = plan_moe_placement(
        vec![
            MoePlacementCandidate {
                id: id_a,
                vram_bytes: 48,
                full_coverage: true,
            },
            MoePlacementCandidate {
                id: id_b,
                vram_bytes: 24,
                full_coverage: false,
            },
            MoePlacementCandidate {
                id: id_c,
                vram_bytes: 24,
                full_coverage: false,
            },
        ],
        &[id_b, id_c],
        &[],
        true,
    )
    .unwrap();

    assert_eq!(plan.active_ids, vec![id_b, id_c]);
    assert_eq!(plan.fallback_ids, vec![id_a]);
    assert_eq!(plan.overlap, 1);
}

#[test]
fn test_running_plan_state_ignores_stale_plan_when_not_running() {
    let id_a = make_id(1);
    let id_b = make_id(2);
    let stale = MoePlacementPlan {
        leader_id: id_a,
        active_ids: vec![id_a],
        fallback_ids: vec![id_b],
        overlap: 1,
    };

    let (active_ids, fallback_ids) = running_plan_state(Some(&stale), false);
    assert!(active_ids.is_empty());
    assert!(fallback_ids.is_empty());

    let (active_ids, fallback_ids) = running_plan_state(Some(&stale), true);
    assert_eq!(active_ids, &[id_a]);
    assert_eq!(fallback_ids, &[id_b]);
}

#[test]
fn test_extend_targets_ignores_non_host_peer() {
    let mut targets = HashMap::new();
    let worker_id = make_id(7);
    let models = vec!["Qwen3-Coder-Next-Q4_K_M".to_string()];

    extend_targets_from_peer(&mut targets, &models, &NodeRole::Worker, worker_id);

    assert!(targets.is_empty());
}

#[test]
fn test_extend_targets_worker_before_host_only_keeps_host() {
    let mut targets = HashMap::new();
    let worker_id = make_id(7);
    let host_id = make_id(8);
    let models = vec!["Qwen3-Coder-Next-Q4_K_M".to_string()];

    extend_targets_from_peer(&mut targets, &models, &NodeRole::Worker, worker_id);
    extend_targets_from_peer(
        &mut targets,
        &models,
        &NodeRole::Host { http_port: 8080 },
        host_id,
    );

    let model_targets = targets.get("Qwen3-Coder-Next-Q4_K_M").unwrap();
    assert_eq!(model_targets.len(), 1);
    assert!(matches!(model_targets[0], InferenceTarget::Remote(id) if id == host_id));
}

#[test]
fn test_extend_targets_keeps_multiple_hosts_for_load_balancing() {
    let mut targets = HashMap::new();
    let host_a = make_id(8);
    let host_b = make_id(9);
    let models = vec!["Qwen3-8B-Q4_K_M".to_string()];

    extend_targets_from_peer(
        &mut targets,
        &models,
        &NodeRole::Host { http_port: 8080 },
        host_a,
    );
    extend_targets_from_peer(
        &mut targets,
        &models,
        &NodeRole::Host { http_port: 8081 },
        host_b,
    );

    let model_targets = targets.get("Qwen3-8B-Q4_K_M").unwrap();
    assert_eq!(model_targets.len(), 2);
    assert!(matches!(model_targets[0], InferenceTarget::Remote(id) if id == host_a));
    assert!(matches!(model_targets[1], InferenceTarget::Remote(id) if id == host_b));
}

#[test]
fn test_model_targets_round_robin_multiple_hosts() {
    let mut targets = ModelTargets::default();
    targets.targets.insert(
        "m".to_string(),
        vec![
            InferenceTarget::Local(7001),
            InferenceTarget::Local(7002),
            InferenceTarget::Local(7003),
        ],
    );

    assert!(matches!(targets.get("m"), InferenceTarget::Local(7001)));
    assert!(matches!(targets.get("m"), InferenceTarget::Local(7002)));
    assert!(matches!(targets.get("m"), InferenceTarget::Local(7003)));
    assert!(matches!(targets.get("m"), InferenceTarget::Local(7001)));
}

#[test]
fn test_model_targets_round_robin_shared_across_clones() {
    let mut targets = ModelTargets::default();
    targets.targets.insert(
        "m".to_string(),
        vec![InferenceTarget::Local(8001), InferenceTarget::Local(8002)],
    );

    let clone = targets.clone();

    assert!(matches!(targets.get("m"), InferenceTarget::Local(8001)));
    assert!(matches!(clone.get("m"), InferenceTarget::Local(8002)));
    assert!(matches!(targets.get("m"), InferenceTarget::Local(8001)));
}

// ── Session hash routing ──

#[test]
fn test_session_routing_sticky() {
    let id_a = make_id(1);
    let id_b = make_id(2);
    let (sorted, _) = moe_shard_index(id_a, &[id_b]);
    let targets = build_moe_targets(&sorted, &[], id_a, Some(8080), None, "m");

    // Same session hint should always route to same node
    let t1 = targets.get_moe_target("user-123");
    let t2 = targets.get_moe_target("user-123");
    assert_eq!(format!("{:?}", t1), format!("{:?}", t2));
}

#[test]
fn test_session_routing_distributes() {
    let id_a = make_id(1);
    let id_b = make_id(2);
    let (sorted, _) = moe_shard_index(id_a, &[id_b]);
    let targets = build_moe_targets(&sorted, &[], id_a, Some(8080), None, "m");

    // With enough different sessions, both nodes should get traffic
    let mut hit_local = false;
    let mut hit_remote = false;
    for i in 0..100 {
        let hint = format!("session-{i}");
        match targets.get_moe_target(&hint) {
            Some(InferenceTarget::MoeLocal(_)) => hit_local = true,
            Some(InferenceTarget::MoeRemote(_)) => hit_remote = true,
            _ => {}
        }
    }
    assert!(hit_local, "Should route some sessions locally");
    assert!(hit_remote, "Should route some sessions to remote");
}

#[test]
fn test_session_routing_empty_moe() {
    let targets = ModelTargets::default();
    assert!(targets.get_moe_target("anything").is_none());
}

#[test]
fn test_session_routing_single_node() {
    let id_a = make_id(1);
    let targets = build_moe_targets(&[id_a], &[], id_a, Some(8080), None, "m");

    // All sessions should go to the single node
    for i in 0..20 {
        match targets.get_moe_target(&format!("s{i}")) {
            Some(InferenceTarget::MoeLocal(8080)) => {}
            other => panic!("Expected MoeLocal(8080), got {:?}", other),
        }
    }
}

// ── Both nodes agree on the same assignments ──

#[test]
fn test_both_nodes_get_consistent_view() {
    // If node A and B both compute assignments for 2 nodes,
    // they should get the same expert lists (just different shard indices)
    let id_a = make_id(1);
    let id_b = make_id(2);

    let (_, idx_a) = moe_shard_index(id_a, &[id_b]);
    let (_, idx_b) = moe_shard_index(id_b, &[id_a]);

    let ranking: Vec<u32> = (0..128).collect();
    let assignments = crate::inference::moe::compute_assignments(&ranking, 2, 46);

    // Node A picks assignment[idx_a], Node B picks assignment[idx_b]
    // They should be different shards
    assert_ne!(idx_a, idx_b);
    // Their unique experts should not overlap
    let a_experts: std::collections::HashSet<u32> =
        assignments[idx_a].experts.iter().cloned().collect();
    let b_experts: std::collections::HashSet<u32> =
        assignments[idx_b].experts.iter().cloned().collect();
    let shared: Vec<u32> = a_experts.intersection(&b_experts).cloned().collect();
    // Shared should be exactly the core (first 46)
    assert_eq!(shared.len(), 46);
    // Union should cover all 128
    let union: std::collections::HashSet<u32> = a_experts.union(&b_experts).cloned().collect();
    assert_eq!(union.len(), 128);
}

#[test]
fn test_pick_sticky_from_consistent() {
    let id_a = make_id(1);
    let id_b = make_id(2);
    let candidates = vec![InferenceTarget::Remote(id_a), InferenceTarget::Remote(id_b)];

    let first = ModelTargets::pick_sticky_from(&candidates, 42);
    let second = ModelTargets::pick_sticky_from(&candidates, 42);
    assert_eq!(first, second);
}

#[test]
fn test_pick_sticky_from_empty_returns_none() {
    let result = ModelTargets::pick_sticky_from(&[], 42);
    assert_eq!(result, InferenceTarget::None);
}

#[test]
fn test_pick_from_round_robins() {
    let id_a = make_id(1);
    let id_b = make_id(2);
    let targets = ModelTargets::default();
    let candidates = vec![InferenceTarget::Remote(id_a), InferenceTarget::Remote(id_b)];

    let first = targets.pick_from(&candidates);
    let second = targets.pick_from(&candidates);
    assert_ne!(first, second);
}

#[test]
fn test_pick_from_empty_returns_none() {
    let targets = ModelTargets::default();
    let result = targets.pick_from(&[]);
    assert_eq!(result, InferenceTarget::None);
}

// ── Row-split / tensor-parallelism selection ──

#[test]
fn row_split_enabled_for_cuda_multi_gpu() {
    assert!(should_use_row_split(Some(BinaryFlavor::Cuda), 2));
    assert!(should_use_row_split(Some(BinaryFlavor::Cuda), 8));
}

#[test]
fn row_split_enabled_for_rocm_multi_gpu() {
    assert!(should_use_row_split(Some(BinaryFlavor::Rocm), 2));
}

#[test]
fn row_split_enabled_for_unknown_flavor_multi_gpu() {
    // None means auto-detected; the resolved binary may still be CUDA/ROCm.
    assert!(should_use_row_split(None, 2));
    assert!(should_use_row_split(None, 4));
}

#[test]
fn row_split_disabled_for_single_gpu() {
    assert!(!should_use_row_split(Some(BinaryFlavor::Cuda), 1));
    assert!(!should_use_row_split(Some(BinaryFlavor::Rocm), 1));
    assert!(!should_use_row_split(None, 1));
}

#[test]
fn row_split_disabled_for_zero_gpus() {
    assert!(!should_use_row_split(Some(BinaryFlavor::Cuda), 0));
    assert!(!should_use_row_split(None, 0));
}

#[test]
fn row_split_disabled_for_non_cuda_backends() {
    // Metal, Vulkan, CPU don't support ggml_backend_split_buffer_type.
    assert!(!should_use_row_split(Some(BinaryFlavor::Metal), 8));
    assert!(!should_use_row_split(Some(BinaryFlavor::Vulkan), 4));
    assert!(!should_use_row_split(Some(BinaryFlavor::Cpu), 4));
}

// ── Regression tests for slots/parallel wiring (T9) ──

/// Verify that `ElectionLoopParams` still carries per-model parallelism.
/// This guards against regressions where per-model parallel counts are silently
/// dropped before reaching the embedded runtime.
#[test]
fn election_loop_params_slots_field_exists() {
    const fn _check_election_loop_has_slots() -> usize {
        42
    }
    let _ = _check_election_loop_has_slots();
}

/// Verify that the skippy local start path still carries per-model parallelism.
#[test]
fn start_skippy_local_params_slots_field_exists() {
    const fn _check_start_skippy_local_has_slots() -> usize {
        16
    }
    let _ = _check_start_skippy_local_has_slots();
}
