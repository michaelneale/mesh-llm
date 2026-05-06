use std::net::SocketAddr;

use anyhow::Result;
use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use opentelemetry_proto::tonic::collector::{
    logs::v1::logs_service_server::LogsServiceServer,
    metrics::v1::metrics_service_server::MetricsServiceServer,
    trace::v1::trace_service_server::TraceServiceServer,
};
use serde_json::json;
use tokio::net::TcpListener;

use crate::{
    api::{
        artifacts, create_run, finalize_run, otlp_http_logs, otlp_http_metrics, otlp_http_traces,
        report_json, run_status,
    },
    cli::ServeArgs,
    otlp::OtlpIngest,
    store::Store,
};

#[derive(Clone)]
pub(crate) struct AppState {
    pub(crate) store: Store,
}

#[derive(Debug)]
pub(crate) struct AppError(anyhow::Error);

impl<E> From<E> for AppError
where
    E: Into<anyhow::Error>,
{
    fn from(error: E) -> Self {
        Self(error.into())
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": self.0.to_string() })),
        )
            .into_response()
    }
}

pub(crate) async fn serve(args: ServeArgs) -> Result<()> {
    let store = Store::open(&args.db, args.debug_retain_raw_otlp)?;
    let state = AppState { store };

    let http = serve_http(state.clone(), args.http_addr);
    let otlp = serve_otlp_grpc(state, args.otlp_grpc_addr);

    println!(
        "metrics-server listening: http={} otlp_grpc={} db={} debug_retain_raw_otlp={}",
        args.http_addr,
        args.otlp_grpc_addr,
        args.db.display(),
        args.debug_retain_raw_otlp
    );

    tokio::try_join!(http, otlp)?;
    Ok(())
}

pub(crate) async fn serve_http(state: AppState, addr: SocketAddr) -> Result<()> {
    let app = Router::new()
        .route("/v1/runs", post(create_run))
        .route("/v1/runs/{run_id}/status", get(run_status))
        .route("/v1/runs/{run_id}/finalize", post(finalize_run))
        .route("/v1/runs/{run_id}/report.json", get(report_json))
        .route("/v1/runs/{run_id}/artifacts", get(artifacts))
        .route("/v1/traces", post(otlp_http_traces))
        .route("/v1/metrics", post(otlp_http_metrics))
        .route("/v1/logs", post(otlp_http_logs))
        .with_state(state);

    let listener = TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

pub(crate) async fn serve_otlp_grpc(state: AppState, addr: SocketAddr) -> Result<()> {
    let ingest = OtlpIngest::new(state.store.clone());
    tonic::transport::Server::builder()
        .add_service(TraceServiceServer::new(ingest.clone()))
        .add_service(MetricsServiceServer::new(ingest.clone()))
        .add_service(LogsServiceServer::new(ingest))
        .serve(addr)
        .await?;
    Ok(())
}
