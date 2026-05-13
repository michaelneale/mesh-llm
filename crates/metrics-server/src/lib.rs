mod api;
mod cli;
mod fixture;
mod model;
mod otlp;
mod otlp_value;
mod server;
mod store;
mod util;

use anyhow::Result;
use clap::Parser;

use cli::{Cli, Command};

pub async fn run() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Serve(args) => server::serve(args).await,
        Command::EmitFixture(args) => fixture::emit_fixture(args).await,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        api::otlp_http_traces,
        otlp_value::{kv_i64, kv_string},
        server::AppState,
        store::Store,
        util::now_unix_nanos,
    };
    use axum::{body::Bytes, extract::State, http::StatusCode, response::IntoResponse};
    use opentelemetry_proto::tonic::metrics::v1::{
        Gauge, ResourceMetrics, ScopeMetrics as MetricScopeMetrics,
    };
    use opentelemetry_proto::tonic::{
        collector::{
            metrics::v1::ExportMetricsServiceRequest, trace::v1::ExportTraceServiceRequest,
        },
        common::v1::InstrumentationScope,
        metrics::v1::{metric as otlp_metric, number_data_point, Metric, NumberDataPoint},
        resource::v1::Resource,
        trace::v1::{span, ResourceSpans, ScopeSpans, Span},
    };
    use prost::Message;
    use rusqlite::{params, Connection};
    use serde_json::json;
    use skippy_metrics::{attr, metric};

    fn fixture_request(run_id: &str) -> ExportTraceServiceRequest {
        let start = now_unix_nanos() as u64;
        ExportTraceServiceRequest {
            resource_spans: vec![ResourceSpans {
                resource: Some(Resource {
                    attributes: vec![kv_string(attr::RUN_ID, run_id)],
                    dropped_attributes_count: 0,
                    entity_refs: vec![],
                }),
                scope_spans: vec![ScopeSpans {
                    scope: Some(InstrumentationScope {
                        name: "metrics-server-test".to_string(),
                        version: "test".to_string(),
                        attributes: vec![],
                        dropped_attributes_count: 0,
                    }),
                    spans: vec![Span {
                        trace_id: vec![7; 16],
                        span_id: vec![9; 8],
                        trace_state: String::new(),
                        parent_span_id: vec![],
                        flags: 1,
                        name: "stage.test".to_string(),
                        kind: span::SpanKind::Internal as i32,
                        start_time_unix_nano: start,
                        end_time_unix_nano: start + 10,
                        attributes: vec![
                            kv_string(attr::RUN_ID, run_id),
                            kv_string(attr::REQUEST_ID, "request-1"),
                            kv_string(attr::SESSION_ID, "session-1"),
                            kv_string(attr::STAGE_ID, "stage-0"),
                            kv_i64(metric::OTEL_DROPPED_EVENTS, 2),
                            kv_i64(metric::OTEL_EXPORT_ERRORS, 1),
                        ],
                        dropped_attributes_count: 0,
                        events: vec![],
                        dropped_events_count: 0,
                        links: vec![],
                        dropped_links_count: 0,
                        status: None,
                    }],
                    schema_url: String::new(),
                }],
                schema_url: String::new(),
            }],
        }
    }

    fn in_memory_store(retain_raw_otlp: bool) -> Store {
        Store::from_connection(Connection::open_in_memory().unwrap(), retain_raw_otlp).unwrap()
    }

    fn fixture_metrics_request(run_id: &str) -> ExportMetricsServiceRequest {
        ExportMetricsServiceRequest {
            resource_metrics: vec![ResourceMetrics {
                resource: Some(Resource {
                    attributes: vec![kv_string(attr::RUN_ID, run_id)],
                    dropped_attributes_count: 0,
                    entity_refs: vec![],
                }),
                scope_metrics: vec![MetricScopeMetrics {
                    scope: Some(InstrumentationScope {
                        name: "skippy-server".to_string(),
                        version: "test".to_string(),
                        attributes: vec![],
                        dropped_attributes_count: 0,
                    }),
                    metrics: vec![Metric {
                        name: metric::OTEL_QUEUE_DEPTH.to_string(),
                        description: String::new(),
                        unit: "{page}".to_string(),
                        metadata: vec![],
                        data: Some(otlp_metric::Data::Gauge(Gauge {
                            data_points: vec![NumberDataPoint {
                                attributes: vec![kv_string(attr::NODE_ID, "node-a")],
                                start_time_unix_nano: 0,
                                time_unix_nano: 42,
                                exemplars: vec![],
                                flags: 0,
                                value: Some(number_data_point::Value::AsInt(7)),
                            }],
                        })),
                    }],
                    schema_url: String::new(),
                }],
                schema_url: String::new(),
            }],
        }
    }

    #[test]
    fn grpc_trace_ingest_populates_report_without_raw_by_default() {
        let store = in_memory_store(false);
        store.ingest_traces(fixture_request("run-test")).unwrap();

        let report = store.report("run-test").unwrap();
        assert_eq!(report.counts["spans"], 1);
        assert_eq!(report.counts["requests"], 1);
        assert_eq!(report.counts["stages"], 1);
        assert_eq!(report.telemetry_loss.dropped_events, 2);
        assert_eq!(report.telemetry_loss.export_errors, 1);

        let raw_json: String = store
            .conn
            .lock()
            .unwrap()
            .query_row(
                "SELECT raw_json FROM spans WHERE run_id = ?",
                params!["run-test"],
                |row| row.get(0),
            )
            .unwrap();
        assert!(raw_json.is_empty());
    }

    #[test]
    fn create_run_registers_planned_stages_before_telemetry_arrives() {
        let store = in_memory_store(false);
        store
            .create_run(
                "run-planned",
                &json!({
                    "stages": [
                        {
                            "stage_id": "stage-0",
                            "stage_index": 0,
                            "host": "shadowfax.local",
                            "layer_start": 0,
                            "layer_end": 1,
                            "endpoint": "tcp://192.168.86.250:19031"
                        },
                        {
                            "stage_id": "stage-1",
                            "stage_index": 1,
                            "host": "black.local",
                            "layer_start": 1,
                            "layer_end": 4,
                            "endpoint": "tcp://192.168.86.24:19032"
                        }
                    ]
                }),
            )
            .unwrap();

        let report = store.report("run-planned").unwrap();
        assert_eq!(report.counts["stages"], 2);
        assert_eq!(report.counts["spans"], 0);
        assert_eq!(report.stages[0].stage_id, "stage-0");
        assert_eq!(report.stages[1].stage_id, "stage-1");
        assert_eq!(report.stages[1].attributes["host"], "black.local");
    }

    #[test]
    fn raw_otlp_retention_is_debug_only() {
        let store = in_memory_store(true);
        store.ingest_traces(fixture_request("run-raw")).unwrap();
        let raw_json: String = store
            .conn
            .lock()
            .unwrap()
            .query_row(
                "SELECT raw_json FROM spans WHERE run_id = ?",
                params!["run-raw"],
                |row| row.get(0),
            )
            .unwrap();
        assert!(raw_json.contains("stage.test"));
    }

    #[test]
    fn metric_ingest_persists_scalar_points_without_raw() {
        let store = in_memory_store(false);
        store
            .ingest_metrics(fixture_metrics_request("run-metrics"))
            .unwrap();

        let (name, value, raw_json): (String, i64, String) = store
            .conn
            .lock()
            .unwrap()
            .query_row(
                "SELECT metric_name, int_value, raw_json
                 FROM metric_points JOIN metrics USING (ingest_id)
                 WHERE metric_points.run_id = ?",
                params!["run-metrics"],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .unwrap();
        assert_eq!(name, metric::OTEL_QUEUE_DEPTH);
        assert_eq!(value, 7);
        assert!(raw_json.is_empty());
    }

    #[tokio::test]
    async fn otlp_http_trace_ingest_accepts_protobuf() {
        let store = in_memory_store(false);
        let state = AppState {
            store: store.clone(),
        };
        let body = Bytes::from(fixture_request("run-http").encode_to_vec());

        let response = otlp_http_traces(State(state), body)
            .await
            .unwrap()
            .into_response();
        assert_eq!(response.status(), StatusCode::OK);

        let report = store.report("run-http").unwrap();
        assert_eq!(report.counts["spans"], 1);
    }
}
