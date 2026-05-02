use std::{
    sync::atomic::{AtomicU64, Ordering},
    time::{SystemTime, UNIX_EPOCH},
};

static RUN_COUNTER: AtomicU64 = AtomicU64::new(1);

pub(crate) fn generate_run_id() -> String {
    format!(
        "run-{}-{}",
        now_unix_nanos(),
        RUN_COUNTER.fetch_add(1, Ordering::Relaxed)
    )
}

pub(crate) fn generate_ingest_id(prefix: &str) -> String {
    format!(
        "{}-{}-{}",
        prefix,
        now_unix_nanos(),
        RUN_COUNTER.fetch_add(1, Ordering::Relaxed)
    )
}

pub(crate) fn now_unix_nanos() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock before Unix epoch")
        .as_nanos() as i64
}
