use std::{
    collections::BTreeMap,
    path::PathBuf,
    sync::{Arc, Mutex},
};

use anyhow::{Context, Result};
use opentelemetry_proto::tonic::{
    collector::{
        logs::v1::ExportLogsServiceRequest, metrics::v1::ExportMetricsServiceRequest,
        trace::v1::ExportTraceServiceRequest,
    },
    common::v1::InstrumentationScope,
    metrics::v1::{metric as otlp_metric, number_data_point, Metric, NumberDataPoint},
    trace::v1::{ResourceSpans, Span},
};
use rusqlite::{params, Connection};
use serde_json::{Map, Value};
use skippy_metrics::{attr, metric};

use crate::{
    model::{
        Report, RequestRecord, RunRecord, RunStatusResponse, SpanRecord, StageRecord,
        StageRequestSummary, TelemetryLossReport,
    },
    otlp_value::{
        attribute_string, attribute_string_from_value, attributes_to_json, bytes_to_hex,
        empty_string_to_none,
    },
    util::{generate_ingest_id, now_unix_nanos},
};

#[derive(Clone)]
pub(crate) struct Store {
    pub(crate) conn: Arc<Mutex<Connection>>,
    retain_raw_otlp: bool,
}

impl Store {
    pub(crate) fn open(path: &PathBuf, retain_raw_otlp: bool) -> Result<Self> {
        let conn = Connection::open(path)
            .with_context(|| format!("open SQLite database {}", path.display()))?;
        Self::from_connection(conn, retain_raw_otlp)
    }

    pub(crate) fn from_connection(conn: Connection, retain_raw_otlp: bool) -> Result<Self> {
        let store = Self {
            conn: Arc::new(Mutex::new(conn)),
            retain_raw_otlp,
        };
        store.init_schema()?;
        Ok(store)
    }

    fn init_schema(&self) -> Result<()> {
        let conn = self.conn.lock().expect("metrics db lock poisoned");
        conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS runs (
                run_id VARCHAR PRIMARY KEY,
                started_at_unix_nanos BIGINT NOT NULL,
                finished_at_unix_nanos BIGINT,
                status VARCHAR NOT NULL,
                config_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS requests (
                run_id VARCHAR NOT NULL,
                request_id VARCHAR NOT NULL,
                session_id VARCHAR,
                first_seen_unix_nanos BIGINT NOT NULL,
                PRIMARY KEY (run_id, request_id)
            );

            CREATE TABLE IF NOT EXISTS stages (
                run_id VARCHAR NOT NULL,
                stage_id VARCHAR NOT NULL,
                first_seen_unix_nanos BIGINT NOT NULL,
                attributes_json TEXT NOT NULL,
                PRIMARY KEY (run_id, stage_id)
            );

            CREATE TABLE IF NOT EXISTS stage_request_summaries (
                run_id VARCHAR NOT NULL,
                request_id VARCHAR NOT NULL,
                stage_id VARCHAR NOT NULL,
                span_count BIGINT NOT NULL,
                first_start_unix_nanos BIGINT NOT NULL,
                last_end_unix_nanos BIGINT NOT NULL,
                PRIMARY KEY (run_id, request_id, stage_id)
            );

            CREATE TABLE IF NOT EXISTS spans (
                run_id VARCHAR NOT NULL,
                request_id VARCHAR,
                session_id VARCHAR,
                stage_id VARCHAR,
                trace_id VARCHAR NOT NULL,
                span_id VARCHAR NOT NULL,
                parent_span_id VARCHAR,
                name VARCHAR NOT NULL,
                kind INTEGER NOT NULL,
                start_time_unix_nanos BIGINT NOT NULL,
                end_time_unix_nanos BIGINT NOT NULL,
                attributes_json TEXT NOT NULL,
                resource_attributes_json TEXT NOT NULL,
                scope_name VARCHAR,
                scope_version VARCHAR,
                raw_json TEXT NOT NULL,
                PRIMARY KEY (trace_id, span_id)
            );

            CREATE TABLE IF NOT EXISTS span_events (
                run_id VARCHAR NOT NULL,
                trace_id VARCHAR NOT NULL,
                span_id VARCHAR NOT NULL,
                name VARCHAR NOT NULL,
                time_unix_nanos BIGINT NOT NULL,
                attributes_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS metrics (
                ingest_id VARCHAR PRIMARY KEY,
                run_id VARCHAR NOT NULL,
                received_at_unix_nanos BIGINT NOT NULL,
                resource_metric_count BIGINT NOT NULL,
                raw_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS metric_points (
                ingest_id VARCHAR NOT NULL,
                run_id VARCHAR NOT NULL,
                resource_attributes_json TEXT NOT NULL,
                scope_name VARCHAR,
                scope_version VARCHAR,
                metric_name VARCHAR NOT NULL,
                metric_unit VARCHAR NOT NULL,
                metric_kind VARCHAR NOT NULL,
                time_unix_nanos BIGINT NOT NULL,
                start_time_unix_nanos BIGINT NOT NULL,
                int_value BIGINT,
                double_value DOUBLE,
                attributes_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS logs (
                ingest_id VARCHAR PRIMARY KEY,
                run_id VARCHAR NOT NULL,
                received_at_unix_nanos BIGINT NOT NULL,
                resource_log_count BIGINT NOT NULL,
                raw_json TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_requests_run_session
                ON requests(run_id, session_id);
            CREATE INDEX IF NOT EXISTS idx_spans_run_request
                ON spans(run_id, request_id);
            CREATE INDEX IF NOT EXISTS idx_spans_run_session
                ON spans(run_id, session_id);
            CREATE INDEX IF NOT EXISTS idx_spans_run_stage
                ON spans(run_id, stage_id);
            CREATE INDEX IF NOT EXISTS idx_span_events_run_span
                ON span_events(run_id, trace_id, span_id);
            CREATE INDEX IF NOT EXISTS idx_metrics_run
                ON metrics(run_id);
            CREATE INDEX IF NOT EXISTS idx_metric_points_run_name
                ON metric_points(run_id, metric_name);
            CREATE INDEX IF NOT EXISTS idx_logs_run
                ON logs(run_id);
            "#,
        )?;
        Ok(())
    }

    pub(crate) fn create_run(&self, run_id: &str, config: &Value) -> Result<()> {
        let conn = self.conn.lock().expect("metrics db lock poisoned");
        conn.execute(
            "INSERT INTO runs
             (run_id, started_at_unix_nanos, finished_at_unix_nanos, status, config_json)
             VALUES (?, ?, NULL, 'running', ?)",
            params![run_id, now_unix_nanos(), config.to_string()],
        )?;
        insert_planned_stages_locked(&conn, run_id, config)?;
        Ok(())
    }

    fn ensure_run(&self, run_id: &str) -> Result<()> {
        let conn = self.conn.lock().expect("metrics db lock poisoned");
        ensure_run_locked(&conn, run_id)
    }

    pub(crate) fn run_status(&self, run_id: &str) -> Result<RunStatusResponse> {
        let conn = self.conn.lock().expect("metrics db lock poisoned");
        let (status, started_at_unix_nanos, finished_at_unix_nanos) = conn.query_row(
            "SELECT status, started_at_unix_nanos, finished_at_unix_nanos
             FROM runs WHERE run_id = ?",
            params![run_id],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, Option<i64>>(2)?,
                ))
            },
        )?;

        Ok(RunStatusResponse {
            run_id: run_id.to_string(),
            status,
            started_at_unix_nanos,
            finished_at_unix_nanos,
            request_count: count_locked(
                &conn,
                "SELECT COUNT(*) FROM requests WHERE run_id = ?",
                run_id,
            )?,
            span_count: count_locked(&conn, "SELECT COUNT(*) FROM spans WHERE run_id = ?", run_id)?,
            metric_export_count: count_locked(
                &conn,
                "SELECT COUNT(*) FROM metrics WHERE run_id = ?",
                run_id,
            )?,
            log_export_count: count_locked(
                &conn,
                "SELECT COUNT(*) FROM logs WHERE run_id = ?",
                run_id,
            )?,
        })
    }

    pub(crate) fn finalize_run(&self, run_id: &str) -> Result<()> {
        self.refresh_stage_request_summaries(run_id)?;
        let conn = self.conn.lock().expect("metrics db lock poisoned");
        conn.execute(
            "UPDATE runs
             SET status = 'completed', finished_at_unix_nanos = ?
             WHERE run_id = ?",
            params![now_unix_nanos(), run_id],
        )?;
        Ok(())
    }

    pub(crate) fn report(&self, run_id: &str) -> Result<Report> {
        self.refresh_stage_request_summaries(run_id)?;
        let conn = self.conn.lock().expect("metrics db lock poisoned");
        let run = conn.query_row(
            "SELECT run_id, status, started_at_unix_nanos, finished_at_unix_nanos, config_json
             FROM runs WHERE run_id = ?",
            params![run_id],
            |row| {
                let config_json: String = row.get(4)?;
                Ok(RunRecord {
                    run_id: row.get(0)?,
                    status: row.get(1)?,
                    started_at_unix_nanos: row.get(2)?,
                    finished_at_unix_nanos: row.get(3)?,
                    config: serde_json::from_str(&config_json).unwrap_or(Value::Null),
                })
            },
        )?;

        let mut counts = BTreeMap::new();
        for (name, query) in [
            ("requests", "SELECT COUNT(*) FROM requests WHERE run_id = ?"),
            ("stages", "SELECT COUNT(*) FROM stages WHERE run_id = ?"),
            ("spans", "SELECT COUNT(*) FROM spans WHERE run_id = ?"),
            (
                "span_events",
                "SELECT COUNT(*) FROM span_events WHERE run_id = ?",
            ),
            (
                "metric_exports",
                "SELECT COUNT(*) FROM metrics WHERE run_id = ?",
            ),
            ("log_exports", "SELECT COUNT(*) FROM logs WHERE run_id = ?"),
        ] {
            counts.insert(name.to_string(), count_locked(&conn, query, run_id)?);
        }

        Ok(Report {
            run,
            counts,
            telemetry_loss: telemetry_loss_report(&conn, run_id)?,
            requests: request_records(&conn, run_id)?,
            stages: stage_records(&conn, run_id)?,
            stage_request_summaries: stage_request_summary_records(&conn, run_id)?,
            spans: span_records(&conn, run_id)?,
        })
    }

    pub(crate) fn ingest_traces(&self, request: ExportTraceServiceRequest) -> Result<()> {
        for resource_spans in request.resource_spans {
            self.ingest_resource_spans(resource_spans)?;
        }
        Ok(())
    }

    fn ingest_resource_spans(&self, resource_spans: ResourceSpans) -> Result<()> {
        let resource_attributes = resource_spans
            .resource
            .as_ref()
            .map(|resource| attributes_to_json(&resource.attributes))
            .unwrap_or_else(|| Value::Object(Map::new()));

        let conn = self.conn.lock().expect("metrics db lock poisoned");
        conn.execute_batch("BEGIN TRANSACTION")?;
        let result = (|| -> Result<()> {
            for scope_spans in resource_spans.scope_spans {
                let scope = scope_spans.scope.clone();
                for span in scope_spans.spans {
                    let span_attributes = attributes_to_json(&span.attributes);
                    let run_id = attribute_string(&span.attributes, attr::RUN_ID)
                        .or_else(|| attribute_string_from_value(&resource_attributes, attr::RUN_ID))
                        .unwrap_or_else(|| "unknown".to_string());
                    let request_id =
                        attribute_string(&span.attributes, attr::REQUEST_ID).or_else(|| {
                            attribute_string_from_value(&resource_attributes, attr::REQUEST_ID)
                        });
                    let session_id =
                        attribute_string(&span.attributes, attr::SESSION_ID).or_else(|| {
                            attribute_string_from_value(&resource_attributes, attr::SESSION_ID)
                        });
                    let stage_id =
                        attribute_string(&span.attributes, attr::STAGE_ID).or_else(|| {
                            attribute_string_from_value(&resource_attributes, attr::STAGE_ID)
                        });

                    ensure_run_locked(&conn, &run_id)?;
                    insert_request_if_needed_locked(
                        &conn,
                        &run_id,
                        request_id.as_deref(),
                        session_id.as_deref(),
                    )?;
                    insert_stage_if_needed_locked(
                        &conn,
                        &run_id,
                        stage_id.as_deref(),
                        &span_attributes,
                    )?;
                    insert_span_locked(
                        &conn,
                        self.retain_raw_otlp,
                        &run_id,
                        request_id.as_deref(),
                        session_id.as_deref(),
                        stage_id.as_deref(),
                        &span,
                        &span_attributes,
                        &resource_attributes,
                        scope.as_ref(),
                    )?;
                }
            }
            Ok(())
        })();
        match result {
            Ok(()) => {
                conn.execute_batch("COMMIT")?;
                Ok(())
            }
            Err(error) => {
                let _ = conn.execute_batch("ROLLBACK");
                Err(error)
            }
        }
    }
}

fn insert_request_if_needed_locked(
    conn: &Connection,
    run_id: &str,
    request_id: Option<&str>,
    session_id: Option<&str>,
) -> Result<()> {
    let Some(request_id) = request_id else {
        return Ok(());
    };
    conn.execute(
        "INSERT INTO requests
             SELECT ?, ?, ?, ?
             WHERE NOT EXISTS (
                 SELECT 1 FROM requests WHERE run_id = ? AND request_id = ?
             )",
        params![
            run_id,
            request_id,
            session_id,
            now_unix_nanos(),
            run_id,
            request_id
        ],
    )?;
    Ok(())
}

fn insert_stage_if_needed_locked(
    conn: &Connection,
    run_id: &str,
    stage_id: Option<&str>,
    attributes: &Value,
) -> Result<()> {
    let Some(stage_id) = stage_id else {
        return Ok(());
    };
    conn.execute(
        "INSERT INTO stages
             SELECT ?, ?, ?, ?
             WHERE NOT EXISTS (
                 SELECT 1 FROM stages WHERE run_id = ? AND stage_id = ?
             )",
        params![
            run_id,
            stage_id,
            now_unix_nanos(),
            attributes.to_string(),
            run_id,
            stage_id
        ],
    )?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn insert_span_locked(
    conn: &Connection,
    retain_raw_otlp: bool,
    run_id: &str,
    request_id: Option<&str>,
    session_id: Option<&str>,
    stage_id: Option<&str>,
    span: &Span,
    span_attributes: &Value,
    resource_attributes: &Value,
    scope: Option<&InstrumentationScope>,
) -> Result<()> {
    let trace_id = bytes_to_hex(&span.trace_id);
    let span_id = bytes_to_hex(&span.span_id);
    let parent_span_id = if span.parent_span_id.is_empty() {
        None
    } else {
        Some(bytes_to_hex(&span.parent_span_id))
    };
    let (scope_name, scope_version) = scope
        .map(|scope| (Some(scope.name.as_str()), Some(scope.version.as_str())))
        .unwrap_or((None, None));
    let raw_json = if retain_raw_otlp {
        serde_json::to_string(span)?
    } else {
        String::new()
    };

    conn.execute(
        "INSERT OR REPLACE INTO spans
             (run_id, request_id, session_id, stage_id, trace_id, span_id, parent_span_id,
              name, kind, start_time_unix_nanos, end_time_unix_nanos, attributes_json,
              resource_attributes_json, scope_name, scope_version, raw_json)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        params![
            run_id,
            request_id,
            session_id,
            stage_id,
            trace_id,
            span_id,
            parent_span_id,
            span.name,
            span.kind,
            span.start_time_unix_nano as i64,
            span.end_time_unix_nano as i64,
            span_attributes.to_string(),
            resource_attributes.to_string(),
            scope_name,
            scope_version,
            raw_json,
        ],
    )?;

    for event in &span.events {
        conn.execute(
            "INSERT INTO span_events
                 (run_id, trace_id, span_id, name, time_unix_nanos, attributes_json)
                 VALUES (?, ?, ?, ?, ?, ?)",
            params![
                run_id,
                bytes_to_hex(&span.trace_id),
                bytes_to_hex(&span.span_id),
                event.name,
                event.time_unix_nano as i64,
                attributes_to_json(&event.attributes).to_string(),
            ],
        )?;
    }

    Ok(())
}

impl Store {
    pub(crate) fn ingest_metrics(&self, request: ExportMetricsServiceRequest) -> Result<()> {
        let run_id = request
            .resource_metrics
            .first()
            .and_then(|resource_metrics| resource_metrics.resource.as_ref())
            .and_then(|resource| attribute_string(&resource.attributes, attr::RUN_ID))
            .unwrap_or_else(|| "unknown".to_string());
        self.ensure_run(&run_id)?;
        let raw_json = if self.retain_raw_otlp {
            serde_json::to_string(&request)?
        } else {
            String::new()
        };
        let ingest_id = generate_ingest_id("metric");
        let conn = self.conn.lock().expect("metrics db lock poisoned");
        conn.execute(
            "INSERT INTO metrics
             (ingest_id, run_id, received_at_unix_nanos, resource_metric_count, raw_json)
             VALUES (?, ?, ?, ?, ?)",
            params![
                ingest_id,
                run_id,
                now_unix_nanos(),
                request.resource_metrics.len() as i64,
                raw_json,
            ],
        )?;
        insert_metric_points_locked(&conn, &ingest_id, &run_id, &request)?;
        Ok(())
    }

    pub(crate) fn ingest_logs(&self, request: ExportLogsServiceRequest) -> Result<()> {
        let run_id = request
            .resource_logs
            .first()
            .and_then(|resource_logs| resource_logs.resource.as_ref())
            .and_then(|resource| attribute_string(&resource.attributes, attr::RUN_ID))
            .unwrap_or_else(|| "unknown".to_string());
        self.ensure_run(&run_id)?;
        let raw_json = if self.retain_raw_otlp {
            serde_json::to_string(&request)?
        } else {
            String::new()
        };
        let conn = self.conn.lock().expect("metrics db lock poisoned");
        conn.execute(
            "INSERT INTO logs
             (ingest_id, run_id, received_at_unix_nanos, resource_log_count, raw_json)
             VALUES (?, ?, ?, ?, ?)",
            params![
                generate_ingest_id("log"),
                run_id,
                now_unix_nanos(),
                request.resource_logs.len() as i64,
                raw_json,
            ],
        )?;
        Ok(())
    }

    fn refresh_stage_request_summaries(&self, run_id: &str) -> Result<()> {
        let conn = self.conn.lock().expect("metrics db lock poisoned");
        conn.execute(
            "DELETE FROM stage_request_summaries WHERE run_id = ?",
            params![run_id],
        )?;
        conn.execute(
            "INSERT INTO stage_request_summaries
             SELECT run_id, request_id, stage_id, COUNT(*) AS span_count,
                    MIN(start_time_unix_nanos) AS first_start_unix_nanos,
                    MAX(end_time_unix_nanos) AS last_end_unix_nanos
             FROM spans
             WHERE run_id = ? AND request_id IS NOT NULL AND stage_id IS NOT NULL
             GROUP BY run_id, request_id, stage_id",
            params![run_id],
        )?;
        Ok(())
    }
}

fn ensure_run_locked(conn: &Connection, run_id: &str) -> Result<()> {
    conn.execute(
        "INSERT INTO runs
         SELECT ?, ?, NULL, 'implicit', '{}'
         WHERE NOT EXISTS (SELECT 1 FROM runs WHERE run_id = ?)",
        params![run_id, now_unix_nanos(), run_id],
    )?;
    Ok(())
}

fn count_locked(conn: &Connection, query: &str, run_id: &str) -> Result<i64> {
    Ok(conn.query_row(query, params![run_id], |row| row.get(0))?)
}

fn request_records(conn: &Connection, run_id: &str) -> Result<Vec<RequestRecord>> {
    let mut stmt = conn.prepare(
        "SELECT run_id, request_id, session_id, first_seen_unix_nanos
         FROM requests WHERE run_id = ? ORDER BY first_seen_unix_nanos, request_id",
    )?;
    let rows = stmt.query_map(params![run_id], |row| {
        Ok(RequestRecord {
            run_id: row.get(0)?,
            request_id: row.get(1)?,
            session_id: row.get(2)?,
            first_seen_unix_nanos: row.get(3)?,
        })
    })?;
    collect_rows(rows)
}

fn stage_records(conn: &Connection, run_id: &str) -> Result<Vec<StageRecord>> {
    let mut stmt = conn.prepare(
        "SELECT run_id, stage_id, first_seen_unix_nanos, attributes_json
         FROM stages WHERE run_id = ? ORDER BY first_seen_unix_nanos, stage_id",
    )?;
    let rows = stmt.query_map(params![run_id], |row| {
        let attributes_json: String = row.get(3)?;
        Ok(StageRecord {
            run_id: row.get(0)?,
            stage_id: row.get(1)?,
            first_seen_unix_nanos: row.get(2)?,
            attributes: serde_json::from_str(&attributes_json).unwrap_or(Value::Null),
        })
    })?;
    collect_rows(rows)
}

fn insert_planned_stages_locked(conn: &Connection, run_id: &str, config: &Value) -> Result<()> {
    let Some(stages) = config.get("stages").and_then(Value::as_array) else {
        return Ok(());
    };

    let first_seen_unix_nanos = now_unix_nanos();
    for (index, stage) in stages.iter().enumerate() {
        let Some(stage_id) = stage.get("stage_id").and_then(Value::as_str) else {
            continue;
        };
        conn.execute(
            "INSERT INTO stages
             SELECT ?, ?, ?, ?
             WHERE NOT EXISTS (
                 SELECT 1 FROM stages WHERE run_id = ? AND stage_id = ?
             )",
            params![
                run_id,
                stage_id,
                first_seen_unix_nanos + index as i64,
                stage.to_string(),
                run_id,
                stage_id
            ],
        )?;
    }

    Ok(())
}

fn stage_request_summary_records(
    conn: &Connection,
    run_id: &str,
) -> Result<Vec<StageRequestSummary>> {
    let mut stmt = conn.prepare(
        "SELECT run_id, request_id, stage_id, span_count,
                first_start_unix_nanos, last_end_unix_nanos
         FROM stage_request_summaries
         WHERE run_id = ?
         ORDER BY first_start_unix_nanos, request_id, stage_id",
    )?;
    let rows = stmt.query_map(params![run_id], |row| {
        Ok(StageRequestSummary {
            run_id: row.get(0)?,
            request_id: row.get(1)?,
            stage_id: row.get(2)?,
            span_count: row.get(3)?,
            first_start_unix_nanos: row.get(4)?,
            last_end_unix_nanos: row.get(5)?,
        })
    })?;
    collect_rows(rows)
}

fn span_records(conn: &Connection, run_id: &str) -> Result<Vec<SpanRecord>> {
    let mut stmt = conn.prepare(
        "SELECT run_id, request_id, session_id, stage_id, trace_id, span_id,
                parent_span_id, name, kind, start_time_unix_nanos,
                end_time_unix_nanos, attributes_json
         FROM spans WHERE run_id = ? ORDER BY start_time_unix_nanos, trace_id, span_id",
    )?;
    let rows = stmt.query_map(params![run_id], |row| {
        let attributes_json: String = row.get(11)?;
        Ok(SpanRecord {
            run_id: row.get(0)?,
            request_id: row.get(1)?,
            session_id: row.get(2)?,
            stage_id: row.get(3)?,
            trace_id: row.get(4)?,
            span_id: row.get(5)?,
            parent_span_id: row.get(6)?,
            name: row.get(7)?,
            kind: row.get(8)?,
            start_time_unix_nanos: row.get(9)?,
            end_time_unix_nanos: row.get(10)?,
            attributes: serde_json::from_str(&attributes_json).unwrap_or(Value::Null),
        })
    })?;
    collect_rows(rows)
}

fn telemetry_loss_report(conn: &Connection, run_id: &str) -> Result<TelemetryLossReport> {
    Ok(TelemetryLossReport {
        dropped_events: max_span_i64_attribute(conn, run_id, metric::OTEL_DROPPED_EVENTS)?,
        export_errors: max_span_i64_attribute(conn, run_id, metric::OTEL_EXPORT_ERRORS)?,
    })
}

fn max_span_i64_attribute(conn: &Connection, run_id: &str, key: &str) -> Result<i64> {
    let mut stmt = conn.prepare("SELECT attributes_json FROM spans WHERE run_id = ?")?;
    let rows = stmt.query_map(params![run_id], |row| row.get::<_, String>(0))?;
    let mut max_value = 0_i64;
    for row in rows {
        let attributes_json = row?;
        let attributes: Value = serde_json::from_str(&attributes_json).unwrap_or(Value::Null);
        if let Some(value) = attributes.get(key).and_then(Value::as_i64) {
            max_value = max_value.max(value);
        }
    }
    Ok(max_value)
}

fn collect_rows<T>(
    rows: rusqlite::MappedRows<'_, impl FnMut(&rusqlite::Row<'_>) -> rusqlite::Result<T>>,
) -> Result<Vec<T>> {
    let mut out = Vec::new();
    for row in rows {
        out.push(row?);
    }
    Ok(out)
}

fn insert_metric_points_locked(
    conn: &Connection,
    ingest_id: &str,
    run_id: &str,
    request: &ExportMetricsServiceRequest,
) -> Result<()> {
    for resource_metrics in &request.resource_metrics {
        let resource_attributes = resource_metrics
            .resource
            .as_ref()
            .map(|resource| attributes_to_json(&resource.attributes))
            .unwrap_or(Value::Null);
        for scope_metrics in &resource_metrics.scope_metrics {
            let (scope_name, scope_version) = scope_metrics
                .scope
                .as_ref()
                .map(|scope| (scope.name.clone(), scope.version.clone()))
                .unwrap_or_default();
            for metric in &scope_metrics.metrics {
                insert_metric_points_for_metric_locked(
                    conn,
                    ingest_id,
                    run_id,
                    &resource_attributes,
                    &scope_name,
                    &scope_version,
                    metric,
                )?;
            }
        }
    }
    Ok(())
}

fn insert_metric_points_for_metric_locked(
    conn: &Connection,
    ingest_id: &str,
    run_id: &str,
    resource_attributes: &Value,
    scope_name: &str,
    scope_version: &str,
    metric: &Metric,
) -> Result<()> {
    let context = MetricPointContext {
        ingest_id,
        run_id,
        resource_attributes,
        scope_name,
        scope_version,
        metric,
    };
    match metric.data.as_ref() {
        Some(otlp_metric::Data::Gauge(gauge)) => {
            for point in &gauge.data_points {
                insert_number_point_locked(conn, &context, "gauge", point)?;
            }
        }
        Some(otlp_metric::Data::Sum(sum)) => {
            for point in &sum.data_points {
                insert_number_point_locked(conn, &context, "sum", point)?;
            }
        }
        _ => {}
    }
    Ok(())
}

struct MetricPointContext<'a> {
    ingest_id: &'a str,
    run_id: &'a str,
    resource_attributes: &'a Value,
    scope_name: &'a str,
    scope_version: &'a str,
    metric: &'a Metric,
}

fn insert_number_point_locked(
    conn: &Connection,
    context: &MetricPointContext<'_>,
    metric_kind: &str,
    point: &NumberDataPoint,
) -> Result<()> {
    let (int_value, double_value) = match point.value.as_ref() {
        Some(number_data_point::Value::AsInt(value)) => (Some(*value), None),
        Some(number_data_point::Value::AsDouble(value)) => (None, Some(*value)),
        None => (None, None),
    };
    conn.execute(
        "INSERT INTO metric_points
         (ingest_id, run_id, resource_attributes_json, scope_name, scope_version,
          metric_name, metric_unit, metric_kind, time_unix_nanos, start_time_unix_nanos,
          int_value, double_value, attributes_json)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        params![
            context.ingest_id,
            context.run_id,
            context.resource_attributes.to_string(),
            empty_string_to_none(context.scope_name),
            empty_string_to_none(context.scope_version),
            context.metric.name,
            context.metric.unit,
            metric_kind,
            point.time_unix_nano as i64,
            point.start_time_unix_nano as i64,
            int_value,
            double_value,
            attributes_to_json(&point.attributes).to_string(),
        ],
    )?;
    Ok(())
}
