use iroh::SecretKey;
use mesh_client::inference::election::InferenceTarget;
use mesh_client::network::affinity::AffinityRouter;

fn make_id(seed: u8) -> iroh::EndpointId {
    let mut bytes = [0u8; 32];
    bytes[0] = seed;
    SecretKey::from_bytes(&bytes).public()
}

#[test]
fn affinity_sticky_routing_consistent() {
    let router = AffinityRouter::new();
    let id_a = make_id(1);
    let id_b = make_id(2);

    let model = "test-model";
    let prefix_hash = 42u64;
    let target = InferenceTarget::Remote(id_a);
    let candidates = vec![InferenceTarget::Remote(id_a), InferenceTarget::Remote(id_b)];

    router.learn_target(model, prefix_hash, &target);

    let result = router.lookup_target(model, prefix_hash, &candidates);
    assert_eq!(result, Some(InferenceTarget::Remote(id_a)));
}

#[test]
fn affinity_forget_removes_entry() {
    let router = AffinityRouter::new();
    let id_a = make_id(3);

    let model = "qwen";
    let prefix_hash = 99u64;
    let target = InferenceTarget::Remote(id_a);
    let candidates = vec![InferenceTarget::Remote(id_a)];

    router.learn_target(model, prefix_hash, &target);
    assert!(router
        .lookup_target(model, prefix_hash, &candidates)
        .is_some());

    router.forget_target(model, prefix_hash, &target);
    assert!(router
        .lookup_target(model, prefix_hash, &candidates)
        .is_none());
}

#[test]
fn affinity_stats_snapshot_initial() {
    let router = AffinityRouter::new();
    let stats = router.stats_snapshot();
    assert_eq!(stats.prefix_entries, 0);
    assert_eq!(stats.learned, 0);
    assert_eq!(stats.prefix_hits, 0);
}
