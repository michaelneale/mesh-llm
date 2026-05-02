use anyhow::Result;
use opentelemetry_proto::tonic::{
    collector::trace::v1::{trace_service_client::TraceServiceClient, ExportTraceServiceRequest},
    common::v1::InstrumentationScope,
    resource::v1::Resource,
    trace::v1::{span, ResourceSpans, ScopeSpans, Span},
};
use skippy_metrics::attr;

use crate::{
    cli::EmitFixtureArgs,
    otlp_value::{kv_i64, kv_string},
    util::now_unix_nanos,
};

pub(crate) async fn emit_fixture(args: EmitFixtureArgs) -> Result<()> {
    let mut client = TraceServiceClient::connect(args.otlp_grpc_addr).await?;
    let start = now_unix_nanos() as u64;
    let end = start + 5_000_000;
    let request = ExportTraceServiceRequest {
        resource_spans: vec![ResourceSpans {
            resource: Some(Resource {
                attributes: vec![kv_string(attr::RUN_ID, &args.run_id)],
                dropped_attributes_count: 0,
                entity_refs: vec![],
            }),
            scope_spans: vec![ScopeSpans {
                scope: Some(InstrumentationScope {
                    name: "skippy-server-fixture".to_string(),
                    version: env!("CARGO_PKG_VERSION").to_string(),
                    attributes: vec![],
                    dropped_attributes_count: 0,
                }),
                spans: vec![Span {
                    trace_id: vec![1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    span_id: vec![2, 2, 2, 2, 2, 2, 2, 2],
                    trace_state: String::new(),
                    parent_span_id: vec![],
                    flags: 1,
                    name: "fixture.decode".to_string(),
                    kind: span::SpanKind::Internal as i32,
                    start_time_unix_nano: start,
                    end_time_unix_nano: end,
                    attributes: vec![
                        kv_string(attr::RUN_ID, &args.run_id),
                        kv_string(attr::REQUEST_ID, &args.request_id),
                        kv_string(attr::SESSION_ID, &args.session_id),
                        kv_string(attr::STAGE_ID, &args.stage_id),
                        kv_i64(attr::STAGE_INDEX, 0),
                    ],
                    dropped_attributes_count: 0,
                    events: vec![span::Event {
                        time_unix_nano: end,
                        name: "fixture.tokens".to_string(),
                        attributes: vec![kv_i64("skippy.output_tokens", 16)],
                        dropped_attributes_count: 0,
                    }],
                    dropped_events_count: 0,
                    links: vec![],
                    dropped_links_count: 0,
                    status: None,
                }],
                schema_url: String::new(),
            }],
            schema_url: String::new(),
        }],
    };

    client.export(request).await?;
    println!("emitted fixture span for run_id={}", args.run_id);
    Ok(())
}
