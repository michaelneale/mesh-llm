use mesh_llm_test_harness::FixtureMesh;
use std::io::{self, BufRead};

fn main() {
    let model = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "Qwen2.5-0.5B-Q4".to_string());
    let fixture = FixtureMesh::new(&model).expect("fixture startup failed");
    println!("INVITE_TOKEN={}", fixture.invite_token());
    let stdin = io::stdin();
    for _ in stdin.lock().lines() {}
}
