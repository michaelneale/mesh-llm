use opentelemetry_proto::tonic::collector::{
    logs::v1::{
        logs_service_server::LogsService, ExportLogsServiceRequest, ExportLogsServiceResponse,
    },
    metrics::v1::{
        metrics_service_server::MetricsService, ExportMetricsServiceRequest,
        ExportMetricsServiceResponse,
    },
    trace::v1::{
        trace_service_server::TraceService, ExportTraceServiceRequest, ExportTraceServiceResponse,
    },
};
use tonic::{Request, Response as TonicResponse, Status};

use crate::store::Store;

#[derive(Clone)]
pub(crate) struct OtlpIngest {
    store: Store,
}

impl OtlpIngest {
    pub(crate) fn new(store: Store) -> Self {
        Self { store }
    }
}

#[tonic::async_trait]
impl TraceService for OtlpIngest {
    async fn export(
        &self,
        request: Request<ExportTraceServiceRequest>,
    ) -> Result<TonicResponse<ExportTraceServiceResponse>, Status> {
        self.store
            .ingest_traces(request.into_inner())
            .map_err(|error| Status::internal(error.to_string()))?;
        Ok(TonicResponse::new(ExportTraceServiceResponse {
            partial_success: None,
        }))
    }
}

#[tonic::async_trait]
impl MetricsService for OtlpIngest {
    async fn export(
        &self,
        request: Request<ExportMetricsServiceRequest>,
    ) -> Result<TonicResponse<ExportMetricsServiceResponse>, Status> {
        self.store
            .ingest_metrics(request.into_inner())
            .map_err(|error| Status::internal(error.to_string()))?;
        Ok(TonicResponse::new(ExportMetricsServiceResponse {
            partial_success: None,
        }))
    }
}

#[tonic::async_trait]
impl LogsService for OtlpIngest {
    async fn export(
        &self,
        request: Request<ExportLogsServiceRequest>,
    ) -> Result<TonicResponse<ExportLogsServiceResponse>, Status> {
        self.store
            .ingest_logs(request.into_inner())
            .map_err(|error| Status::internal(error.to_string()))?;
        Ok(TonicResponse::new(ExportLogsServiceResponse {
            partial_success: None,
        }))
    }
}
