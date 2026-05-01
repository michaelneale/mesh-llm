use mesh_client::network::nostr::{
    score_mesh, smart_auto, AutoDecision, DiscoveredMesh, MeshFilter, MeshListing,
};

fn make_listing(
    name: Option<&str>,
    mesh_id: Option<&str>,
    serving: &[&str],
    node_count: usize,
    vram: u64,
    clients: usize,
    max_clients: usize,
) -> DiscoveredMesh {
    DiscoveredMesh {
        listing: MeshListing {
            invite_token: format!("invite-{}", mesh_id.unwrap_or("test")),
            serving: serving.iter().map(|s| s.to_string()).collect(),
            wanted: vec![],
            on_disk: vec![],
            total_vram_bytes: vram,
            node_count,
            client_count: clients,
            max_clients,
            name: name.map(|s| s.to_string()),
            region: None,
            mesh_id: mesh_id.map(|s| s.to_string()),
        },
        publisher_npub: format!("npub-{}", mesh_id.unwrap_or("test")),
        published_at: 1000,
        expires_at: Some(9999999999),
    }
}

#[test]
fn score_mesh_with_synthetic_listing() {
    let mesh = make_listing(
        Some("mesh-llm"),
        Some("abc"),
        &["Qwen3-8B-Q4_K_M"],
        3,
        48_000_000_000,
        1,
        10,
    );
    let score = score_mesh(&mesh, 1500, None);
    assert!(
        score > 400,
        "community mesh should score > 400, got {score}"
    );
}

#[test]
fn score_mesh_private_mesh_penalty() {
    let mesh = make_listing(
        Some("bobs-cluster"),
        Some("xyz"),
        &["Qwen3-8B-Q4_K_M"],
        3,
        48_000_000_000,
        0,
        0,
    );
    let score = score_mesh(&mesh, 1500, None);
    assert!(score < 100, "private mesh should score < 100, got {score}");
}

#[test]
fn score_mesh_full_penalty() {
    let mesh = make_listing(
        None,
        Some("full"),
        &["Qwen3-8B-Q4_K_M"],
        2,
        16_000_000_000,
        5,
        5,
    );
    let score = score_mesh(&mesh, 1500, None);
    assert!(score < 0, "full mesh should score negative, got {score}");
}

#[test]
fn score_mesh_sticky_bonus() {
    let mesh = make_listing(
        None,
        Some("my-mesh"),
        &["Qwen3-8B-Q4_K_M"],
        2,
        16_000_000_000,
        0,
        0,
    );
    let sticky = score_mesh(&mesh, 1500, Some("my-mesh"));
    let fresh = score_mesh(&mesh, 1500, None);
    assert!(
        sticky > fresh + 400,
        "sticky bonus should be >400, sticky={sticky} fresh={fresh}"
    );
}

#[test]
fn mesh_filter_model_match() {
    let mesh = make_listing(None, None, &["Qwen3-8B-Q4_K_M"], 1, 8_000_000_000, 0, 0);
    let f = MeshFilter {
        model: Some("qwen3-8b".into()),
        ..Default::default()
    };
    assert!(f.matches(&mesh));
}

#[test]
fn mesh_filter_model_no_match() {
    let mesh = make_listing(None, None, &["Qwen3-8B-Q4_K_M"], 1, 8_000_000_000, 0, 0);
    let f = MeshFilter {
        model: Some("llama".into()),
        ..Default::default()
    };
    assert!(!f.matches(&mesh));
}

#[test]
fn smart_auto_start_new_when_empty() {
    match smart_auto(&[], 8.0, None, None) {
        AutoDecision::StartNew { models } => {
            assert!(!models.is_empty(), "should suggest models");
        }
        AutoDecision::Join { .. } => panic!("empty list should produce StartNew"),
    }
}

#[test]
fn smart_auto_join_community_mesh() {
    let meshes = vec![
        make_listing(
            Some("mesh-llm"),
            Some("aaa"),
            &["Qwen3-8B-Q4_K_M"],
            3,
            48_000_000_000,
            1,
            10,
        ),
        make_listing(
            Some("bobs-cluster"),
            Some("bbb"),
            &["Qwen3-8B-Q4_K_M"],
            5,
            80_000_000_000,
            0,
            0,
        ),
    ];
    match smart_auto(&meshes, 8.0, None, None) {
        AutoDecision::Join { candidates } => {
            assert!(!candidates.is_empty());
            assert_eq!(candidates[0].0, "invite-aaa", "community mesh first");
        }
        AutoDecision::StartNew { .. } => panic!("should join"),
    }
}

#[test]
fn smart_auto_sticky_prefers_last_mesh() {
    let meshes = vec![
        make_listing(
            None,
            Some("other"),
            &["Qwen3-8B-Q4_K_M"],
            3,
            24_000_000_000,
            0,
            0,
        ),
        make_listing(
            None,
            Some("sticky-mesh"),
            &["Qwen3-8B-Q4_K_M"],
            2,
            16_000_000_000,
            0,
            0,
        ),
    ];
    match smart_auto(&meshes, 8.0, None, Some("sticky-mesh")) {
        AutoDecision::Join { candidates } => {
            assert!(!candidates.is_empty());
            assert_eq!(candidates[0].0, "invite-sticky-mesh", "sticky mesh first");
        }
        AutoDecision::StartNew { .. } => panic!("should join"),
    }
}
