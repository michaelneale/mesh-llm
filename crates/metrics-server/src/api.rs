use anyhow::Context;
use axum::{
    body::Bytes,
    extract::{Path, State},
    http::header,
    response::IntoResponse,
    Json,
};
use opentelemetry_proto::tonic::collector::{
    logs::v1::{ExportLogsServiceRequest, ExportLogsServiceResponse},
    metrics::v1::{ExportMetricsServiceRequest, ExportMetricsServiceResponse},
    trace::v1::{ExportTraceServiceRequest, ExportTraceServiceResponse},
};
use prost::Message;
use serde_json::Value;

use crate::{
    model::{
        Artifact, ArtifactsResponse, CreateRunRequest, CreateRunResponse, Report, RunStatusResponse,
    },
    server::{AppError, AppState},
    util::generate_run_id,
};

pub(crate) async fn create_run(
    State(state): State<AppState>,
    Json(request): Json<CreateRunRequest>,
) -> Result<Json<CreateRunResponse>, AppError> {
    let run_id = request.run_id.unwrap_or_else(generate_run_id);
    let config = Value::Object(request.config);
    state.store.create_run(&run_id, &config)?;
    Ok(Json(CreateRunResponse {
        run_id,
        status: "running".to_string(),
    }))
}

pub(crate) async fn run_status(
    State(state): State<AppState>,
    Path(run_id): Path<String>,
) -> Result<Json<RunStatusResponse>, AppError> {
    Ok(Json(state.store.run_status(&run_id)?))
}

pub(crate) async fn finalize_run(
    State(state): State<AppState>,
    Path(run_id): Path<String>,
) -> Result<Json<RunStatusResponse>, AppError> {
    state.store.finalize_run(&run_id)?;
    Ok(Json(state.store.run_status(&run_id)?))
}

pub(crate) async fn report_json(
    State(state): State<AppState>,
    Path(run_id): Path<String>,
) -> Result<Json<Report>, AppError> {
    Ok(Json(state.store.report(&run_id)?))
}

pub(crate) async fn artifacts(Path(run_id): Path<String>) -> Json<ArtifactsResponse> {
    Json(ArtifactsResponse {
        artifacts: vec![Artifact {
            name: "report.json".to_string(),
            url: format!("/v1/runs/{run_id}/report.json"),
            content_type: "application/json".to_string(),
        }],
    })
}

pub(crate) async fn otlp_http_traces(
    State(state): State<AppState>,
    body: Bytes,
) -> Result<impl IntoResponse, AppError> {
    let request = ExportTraceServiceRequest::decode(body.as_ref())
        .context("decode OTLP/HTTP trace export request")?;
    state.store.ingest_traces(request)?;
    Ok(protobuf_response(ExportTraceServiceResponse {
        partial_success: None,
    }))
}

pub(crate) async fn otlp_http_metrics(
    State(state): State<AppState>,
    body: Bytes,
) -> Result<impl IntoResponse, AppError> {
    let request = ExportMetricsServiceRequest::decode(body.as_ref())
        .context("decode OTLP/HTTP metric export request")?;
    state.store.ingest_metrics(request)?;
    Ok(protobuf_response(ExportMetricsServiceResponse {
        partial_success: None,
    }))
}

pub(crate) async fn otlp_http_logs(
    State(state): State<AppState>,
    body: Bytes,
) -> Result<impl IntoResponse, AppError> {
    let request = ExportLogsServiceRequest::decode(body.as_ref())
        .context("decode OTLP/HTTP log export request")?;
    state.store.ingest_logs(request)?;
    Ok(protobuf_response(ExportLogsServiceResponse {
        partial_success: None,
    }))
}

fn protobuf_response<M: Message>(message: M) -> impl IntoResponse {
    (
        [(header::CONTENT_TYPE, "application/x-protobuf")],
        message.encode_to_vec(),
    )
}
