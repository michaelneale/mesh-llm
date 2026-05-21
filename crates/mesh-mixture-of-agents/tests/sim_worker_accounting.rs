//! Pin the worker-accounting contract for `TurnResult.worker_summaries`.
//!
//! Background — PR #566 review feedback (Apr 2026):
//!
//! > Worker accounting was inconsistent:
//! > - Similar requests reported different `x-moa-workers` values.
//! > - Similar requests reported different `x-moa-workers-ok` values.
//! > - Some successful responses used fewer workers than expected.
//! > - Some churn responses still appeared to report stale worker counts.
//! > - One concurrency response reported zero successful workers.
//!
//! `worker_summaries.len()` is what the `x-moa-workers` header reports.
//! If the gateway dispatches N workers, the summaries must reflect all
//! N of them once the turn finishes — including ones that were aborted
//! by early-exit consensus. Otherwise a client cannot tell whether the
//! response came from 2 workers because we only dispatched 2 or because
//! 2 more were cancelled mid-flight.
//!
//! Two regressions this file pins:
//!
//! 1. **Early-exit aborts must still be accounted for.** When the
//!    arbiter decides from the first two workers and aborts the rest,
//!    the cancelled workers' summaries are silently dropped today
//!    because `gather_workers_incremental` drains `join_next()` with
//!    `if let Ok(...)` and `JoinSet::abort_all` returns
//!    `Err(JoinError::cancelled)` for the aborted tasks.
//!
//! 2. **All-fail must still attribute every worker.** When every
//!    backend errors, `worker_summaries.len()` must still equal the
//!    number of workers dispatched. (This one passes today because the
//!    `Err` branch in `gather_workers_incremental` pushes a summary
//!    regardless — `sim_all_workers_fail.rs` already covers it.)

use async_trait::async_trait;
use mesh_mixture_of_agents as moa;
use serde_json::{json, Value};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

/// Backend that returns a deterministic answer text after a configurable
/// delay. Used to set up early-exit consensus.
struct DelayedAnswerBackend {
    text: String,
    delay: Duration,
    calls: AtomicUsize,
}

impl DelayedAnswerBackend {
    fn new(text: impl Into<String>, delay: Duration) -> Arc<Self> {
        Arc::new(Self {
            text: text.into(),
            delay,
            calls: AtomicUsize::new(0),
        })
    }
}

#[async_trait]
impl moa::ModelBackend for DelayedAnswerBackend {
    async fn chat_completion(
        &self,
        _model: &str,
        _messages: &[Value],
        _tools: Option<&Value>,
        _max_tokens: u32,
        _timeout: Duration,
        _sampling: moa::SamplingParams,
    ) -> Result<Value, String> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        tokio::time::sleep(self.delay).await;
        Ok(json!({
            "choices": [{"message": {"content": self.text}}],
        }))
    }
}

fn four_workers_two_fast_consensus() -> moa::GatewayConfig {
    // Two fast workers that agree on the same answer ("Tokyo") → arbiter
    // should early-exit on consensus and abort the two slow workers.
    let fast_a = DelayedAnswerBackend::new("Tokyo", Duration::from_millis(20));
    let fast_b = DelayedAnswerBackend::new("Tokyo", Duration::from_millis(40));
    let slow_a = DelayedAnswerBackend::new("Tokyo", Duration::from_secs(5));
    let slow_b = DelayedAnswerBackend::new("Tokyo", Duration::from_secs(5));

    let backends: Vec<Arc<dyn moa::ModelBackend>> = vec![fast_a, fast_b, slow_a, slow_b];
    let models = vec![
        moa::ModelEntry {
            name: "fast-a-3b".into(),
            backend_index: 0,
        },
        moa::ModelEntry {
            name: "fast-b-3b".into(),
            backend_index: 1,
        },
        // The two big-tier models — one of them will be picked as
        // `Strong`. They are deliberately slow so the early-exit path
        // is hit before they finish.
        moa::ModelEntry {
            name: "slow-a-32b".into(),
            backend_index: 2,
        },
        moa::ModelEntry {
            name: "slow-b-32b".into(),
            backend_index: 3,
        },
    ];

    moa::GatewayConfig {
        backends,
        models,
        worker_timeout: Duration::from_secs(10),
        hedge_delay: Duration::from_millis(200),
        reducer_timeout: Duration::from_secs(2),
    }
}

fn user_turn(content: &str) -> Value {
    json!({
        "model": "mesh",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 32,
    })
}

#[tokio::test]
async fn early_exit_summaries_account_for_aborted_workers() {
    // Bias the runtime to advance through sleeps as fast as possible —
    // we just want the order: fast_a finishes, fast_b finishes, consensus
    // → abort the two slow workers.
    let config = four_workers_two_fast_consensus();
    let body = user_turn("What is the capital of Japan? One word only.");

    let result = moa::handle_turn(&config, &body).await;

    // Sanity: this should be the early-exit path.
    assert_eq!(
        result.turn_kind,
        moa::TurnKind::EarlyExit,
        "two fast agreeing workers should produce TurnKind::EarlyExit; got {:?}",
        result.turn_kind
    );

    // The contract: we dispatched 4 workers. Every dispatched worker
    // must appear in `worker_summaries`, with `succeeded` reflecting
    // its true fate (succeeded / failed / aborted). The header
    // `x-moa-workers` is built from `worker_summaries.len()`; if it is
    // less than the dispatched count, the client cannot tell whether
    // we ran 2 workers or 4-with-2-aborted.
    let dispatched = 4;
    let summary_count = result.worker_summaries.len();
    let summary_models: Vec<&str> = result
        .worker_summaries
        .iter()
        .map(|w| w.model.as_str())
        .collect();
    assert_eq!(
        summary_count, dispatched,
        "worker_summaries.len() must equal the dispatched worker count even when \
         early-exit aborts cancelled the remaining workers. Got {summary_count} \
         summaries: {summary_models:?}; expected {dispatched}."
    );

    // Every dispatched model must be named.
    for name in ["fast-a-3b", "fast-b-3b", "slow-a-32b", "slow-b-32b"] {
        assert!(
            summary_models.contains(&name),
            "worker_summaries is missing model {name:?}; got {summary_models:?}"
        );
    }

    // `succeeded` should be true for the workers that produced output
    // before consensus, and false (or otherwise "not succeeded") for
    // the ones we cancelled. We do not assert exactly which two — the
    // contract is just "the count is honest."
    let succeeded_count = result
        .worker_summaries
        .iter()
        .filter(|w| w.succeeded)
        .count();
    assert!(
        succeeded_count >= 2,
        "at least the two fast workers must be marked succeeded; got {succeeded_count}"
    );
    assert!(
        succeeded_count <= dispatched,
        "succeeded_count {succeeded_count} cannot exceed dispatched {dispatched}"
    );
}
