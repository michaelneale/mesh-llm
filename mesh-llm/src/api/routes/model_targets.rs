use super::super::{http::respond_json, status::ModelTargetPayload, MeshApi};
use serde::Serialize;
use tokio::net::TcpStream;

#[derive(Debug, Serialize)]
struct ModelTargetListResponse {
    model_targets: Vec<ModelTargetPayload>,
}

pub(super) async fn handle(stream: &mut TcpStream, state: &MeshApi) -> anyhow::Result<()> {
    respond_json(
        stream,
        200,
        &ModelTargetListResponse {
            model_targets: state.model_targets().await,
        },
    )
    .await
}
