use mesh_client::inference::moe::{
    better_shared_ranking, compute_assignments_with_overlap, compute_snake_draft_assignments,
    expert_list_arg, ranking_strength_key, MoeMicroLayerScope, MoeRankingStrategy,
    MoeRuntimeOptions, NodeAssignment, SharedRankingArtifact, SharedRankingKind,
    SharedRankingOrigin,
};

#[test]
fn moe_ranking_with_synthetic_stats() {
    // arbitrary gate-mass order — hottest expert first
    let ranking: Vec<u32> = vec![5, 2, 8, 1, 9, 3, 7, 0, 4, 6];
    let assignments = compute_assignments_with_overlap(&ranking, 3, 4, 1);

    assert_eq!(assignments.len(), 3);

    // every expert must appear in at least one node
    let mut covered: std::collections::HashSet<u32> = std::collections::HashSet::new();
    for a in &assignments {
        covered.extend(a.experts.iter().copied());
    }
    assert_eq!(covered.len(), 10, "all 10 experts must be covered");

    // shared core (top 4 by ranking order) must appear in every node
    let shared_core: std::collections::HashSet<u32> = ranking[..4].iter().copied().collect();
    for a in &assignments {
        for &e in &shared_core {
            assert!(
                a.experts.contains(&e),
                "shared expert {e} missing from a node"
            );
        }
    }

    // experts within each node must be sorted
    for a in &assignments {
        let mut sorted = a.experts.clone();
        sorted.sort();
        assert_eq!(a.experts, sorted, "node experts must be sorted");
    }
}

#[test]
fn ranking_strength_ordering() {
    let full = SharedRankingArtifact {
        kind: SharedRankingKind::Analyze,
        origin: SharedRankingOrigin::LocalFullAnalyze,
        ranking: vec![0, 1, 2, 3],
        micro_prompt_count: None,
        micro_tokens: None,
        micro_layer_scope: None,
    };

    let micro_all = SharedRankingArtifact {
        kind: SharedRankingKind::MicroAnalyze,
        origin: SharedRankingOrigin::LocalMicroAnalyze,
        ranking: vec![0, 1, 2, 3],
        micro_prompt_count: Some(4),
        micro_tokens: Some(32),
        micro_layer_scope: Some(MoeMicroLayerScope::All),
    };

    let micro_first = SharedRankingArtifact {
        kind: SharedRankingKind::MicroAnalyze,
        origin: SharedRankingOrigin::LocalMicroAnalyze,
        ranking: vec![0, 1, 2, 3],
        micro_prompt_count: Some(1),
        micro_tokens: Some(8),
        micro_layer_scope: Some(MoeMicroLayerScope::First),
    };

    // full analyze beats any micro
    assert!(better_shared_ranking(&full, &micro_all));
    assert!(better_shared_ranking(&full, &micro_first));

    // more prompts / all layers beats first-layer micro
    assert!(better_shared_ranking(&micro_all, &micro_first));

    // not stronger than itself
    assert!(!better_shared_ranking(&full, &full));
}

#[test]
fn ranking_strength_key_analyze_is_highest() {
    let full = SharedRankingArtifact {
        kind: SharedRankingKind::Analyze,
        origin: SharedRankingOrigin::LocalFullAnalyze,
        ranking: vec![0],
        micro_prompt_count: None,
        micro_tokens: None,
        micro_layer_scope: None,
    };
    let (tier, _, _, _) = ranking_strength_key(&full);
    assert_eq!(tier, 2, "Analyze should be tier 2 (highest)");
}

#[test]
fn snake_draft_covers_all_experts() {
    let ranking: Vec<u32> = (0..8).collect();
    let assignments = compute_snake_draft_assignments(&ranking, 2, 2);

    assert_eq!(assignments.len(), 2);

    let mut covered: std::collections::HashSet<u32> = std::collections::HashSet::new();
    for a in &assignments {
        covered.extend(a.experts.iter().copied());
    }
    assert_eq!(covered.len(), 8, "all experts must be covered");
}

#[test]
fn expert_list_arg_format() {
    let assignment = NodeAssignment {
        experts: vec![0, 3, 7, 12],
        n_shared: 2,
        n_unique: 2,
    };
    assert_eq!(expert_list_arg(&assignment), "0,3,7,12");
}

#[test]
fn moe_runtime_options_defaults() {
    let opts = MoeRuntimeOptions::default();
    assert!(matches!(opts.ranking_strategy, MoeRankingStrategy::Auto));
    assert!(matches!(opts.micro_layer_scope, MoeMicroLayerScope::All));
    assert_eq!(opts.micro_prompt_count, 1);
    assert_eq!(opts.micro_tokens, 8);
}

#[cfg(feature = "host-io")]
#[test]
fn load_cached_ranking_roundtrip() {
    use mesh_client::inference::moe::{
        load_cached_ranking, load_shared_ranking_artifact, write_shared_ranking_artifact,
    };

    let dir = std::env::temp_dir().join("mcc-moe-test-ranking");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("test.csv");

    let ranking = vec![0u32, 26, 41, 69, 104, 3, 7, 99];
    let content: String = ranking
        .iter()
        .map(|e| e.to_string())
        .collect::<Vec<_>>()
        .join("\n");
    std::fs::write(&path, content).unwrap();

    let loaded = load_cached_ranking(&path).unwrap();
    assert_eq!(loaded, ranking);

    // roundtrip via SharedRankingArtifact
    let artifact = SharedRankingArtifact {
        kind: SharedRankingKind::MicroAnalyze,
        origin: SharedRankingOrigin::PeerImport,
        ranking: vec![3, 1, 2, 0],
        micro_prompt_count: Some(2),
        micro_tokens: Some(16),
        micro_layer_scope: Some(MoeMicroLayerScope::All),
    };
    let artifact_path = dir.join("artifact.csv");
    write_shared_ranking_artifact(&artifact_path, &artifact).unwrap();

    let loaded_artifact = load_shared_ranking_artifact(
        &artifact_path,
        SharedRankingKind::MicroAnalyze,
        SharedRankingOrigin::LegacyCache,
        Some(1),
        Some(8),
        Some(MoeMicroLayerScope::First),
    )
    .unwrap();
    assert_eq!(loaded_artifact, artifact);

    let _ = std::fs::remove_dir_all(&dir);
}
