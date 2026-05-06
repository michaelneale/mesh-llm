#![recursion_limit = "256"]

#[tokio::main]
async fn main() {
    std::process::exit(mesh_llm::run_main().await);
}
