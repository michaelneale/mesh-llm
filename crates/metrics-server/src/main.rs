use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    metrics_server::run().await
}
