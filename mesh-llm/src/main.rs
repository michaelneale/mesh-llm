#[tokio::main]
async fn main() {
    if let Err(err) = mesh_llm::run().await {
        let text = err.to_string();
        if text.starts_with('🟡') {
            eprintln!("{text}");
        } else {
            eprintln!("Error: {err:#}");
        }
        std::process::exit(1);
    }
}
