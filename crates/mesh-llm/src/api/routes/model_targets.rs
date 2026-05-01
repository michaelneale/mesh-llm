use super::super::{http::respond_json, status::ModelTargetPayload, MeshApi};
use serde::Serialize;
use tokio::net::TcpStream;

#[derive(Debug, Serialize)]
struct ModelTargetListResponse {
    model_targets: Vec<ModelTargetResponseItem>,
}

#[derive(Debug, Serialize)]
struct ModelTargetResponseItem {
    model_ref: String,
    display_name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    model_name: Option<String>,
    signals: ModelTargetSignals,
    derived: ModelTargetDerived,
}

#[derive(Debug, Serialize)]
struct ModelTargetSignals {
    explicit_interest_count: usize,
    request_count: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    last_active_secs_ago: Option<u64>,
    serving_node_count: usize,
    requested: bool,
}

#[derive(Debug, Serialize)]
struct ModelTargetDerived {
    target_rank: usize,
    wanted: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    wanted_reason: Option<&'static str>,
}

impl From<ModelTargetPayload> for ModelTargetResponseItem {
    fn from(target: ModelTargetPayload) -> Self {
        Self {
            model_ref: target.model_ref,
            display_name: target.display_name,
            model_name: target.model_name,
            signals: ModelTargetSignals {
                explicit_interest_count: target.explicit_interest_count,
                request_count: target.request_count,
                last_active_secs_ago: target.last_active_secs_ago,
                serving_node_count: target.serving_node_count,
                requested: target.requested,
            },
            derived: ModelTargetDerived {
                target_rank: target.rank,
                wanted: target.wanted,
                wanted_reason: target.wanted_reason,
            },
        }
    }
}

pub(super) async fn handle(stream: &mut TcpStream, state: &MeshApi) -> anyhow::Result<()> {
    let model_targets = state
        .model_targets()
        .await
        .into_iter()
        .map(ModelTargetResponseItem::from)
        .collect();

    respond_json(stream, 200, &ModelTargetListResponse { model_targets }).await
}
