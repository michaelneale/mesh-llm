use mesh_llm_test_harness::FixtureMesh;

#[test]
#[ignore] // GREEN in Wave 10 when mesh-llm binary + Qwen2.5-0.5B-Q4 are available
fn fixture_launches_and_captures_invite() {
    let fixture = FixtureMesh::new("Qwen2.5-0.5B-Q4").expect("fixture startup");
    let token = fixture.invite_token();
    assert!(!token.is_empty(), "invite token should be non-empty");
    assert!(
        token.len() > 20,
        "invite token should look like a real token"
    );
    drop(fixture);
}
