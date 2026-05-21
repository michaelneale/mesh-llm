//! Worker fan-out and incremental gathering.
//!
//! Workers run in parallel via [`tokio::task::JoinSet`]. As each one
//! completes, the result is normalized, allowed-tool filtered, and fed
//! to the arbiter's early-exit check. If the arbiter can already decide
//! from the responses collected so far (consensus, sole survivor, etc.)
//! the remaining workers are cancelled and the decision is returned
//! immediately.

use std::time::{Duration, Instant};

use crate::enforce_allowed_tools;
use crate::worker::WorkerRole;
use crate::{arbiter, normalize, WorkerSummary};
use normalize::WorkerOutput;

/// Min confidence for the time-based grace path; matches the consensus rule.
const GRACE_MIN_CONFIDENCE: f32 = 0.5;

/// Identifier for a worker we dispatched. Used to reconcile the
/// per-worker accounting at the end of fan-out so the — possibly
/// aborted or panicked — task's existence still shows up in
/// `worker_summaries`.
pub(crate) struct DispatchedWorker {
    pub model: String,
    pub role: WorkerRole,
}

pub(crate) async fn gather_workers_incremental(
    join_set: &mut tokio::task::JoinSet<(String, WorkerRole, Result<String, String>, u64)>,
    dispatched: &[DispatchedWorker],
    has_tools: bool,
    allowed_tools: &[String],
    first_answer_grace: Duration,
) -> (
    Vec<WorkerOutput>,
    Vec<WorkerSummary>,
    Option<arbiter::Decision>,
) {
    let total_workers = dispatched.len();
    let mut outputs = Vec::new();
    let mut summaries = Vec::new();
    let mut total_finished: usize = 0;
    let dispatched_at = Instant::now();
    let grace_enabled = !has_tools && !first_answer_grace.is_zero();

    let grace_eligible = |outs: &[WorkerOutput]| -> bool {
        if !grace_enabled {
            return false;
        }
        let answers: Vec<&WorkerOutput> = outs
            .iter()
            .filter(|o| o.kind == normalize::OutputKind::Answer)
            .collect();
        answers.len() == 1 && answers[0].confidence >= GRACE_MIN_CONFIDENCE
    };

    loop {
        let grace_remaining = if grace_eligible(&outputs) {
            first_answer_grace.saturating_sub(dispatched_at.elapsed())
        } else {
            Duration::from_secs(60 * 60)
        };
        let armed = grace_eligible(&outputs);

        let join_result = tokio::select! {
            biased;
            join = join_set.join_next() => join,
            _ = tokio::time::sleep(grace_remaining), if armed => {
                tracing::info!(
                    "moa: grace early-exit — sole answer after {}ms (grace={}ms), {} pending",
                    dispatched_at.elapsed().as_millis(),
                    first_answer_grace.as_millis(),
                    total_workers.saturating_sub(total_finished),
                );
                drain_after_early_exit(join_set, &mut summaries).await;
                reconcile_dispatched(dispatched, &mut summaries);
                let answer = outputs
                    .iter()
                    .find(|o| o.kind == normalize::OutputKind::Answer)
                    .expect("grace_eligible guaranteed exactly one Answer")
                    .payload
                    .clone();
                return (outputs, summaries, Some(arbiter::Decision::Answer(answer)));
            }
        };

        let Some(join_result) = join_result else {
            break;
        };

        match join_result {
            Ok((model, role, Ok(text), elapsed)) => {
                total_finished += 1;
                let mut normalized =
                    normalize::normalize_worker_output(&text, &model, role, elapsed);
                enforce_allowed_tools(&mut normalized, allowed_tools, &model);
                tracing::info!(
                    "moa: worker {} ({}) → {:?} conf={:.2} ({}ms, {} chars)",
                    model,
                    role.label(),
                    normalized.kind,
                    normalized.confidence,
                    elapsed,
                    text.len(),
                );
                summaries.push(WorkerSummary {
                    model: model.clone(),
                    role,
                    succeeded: true,
                    elapsed_ms: elapsed,
                    output_kind: Some(normalized.kind),
                    confidence: Some(normalized.confidence),
                });
                outputs.push(normalized);

                if let Some(decision) =
                    arbiter::try_early_decision(&outputs, total_workers, total_finished, has_tools)
                {
                    drain_after_early_exit(join_set, &mut summaries).await;
                    reconcile_dispatched(dispatched, &mut summaries);
                    return (outputs, summaries, Some(decision));
                }
            }
            Ok((model, role, Err(e), elapsed)) => {
                total_finished += 1;
                tracing::warn!(
                    "moa: worker {} ({}) failed after {}ms: {}",
                    model,
                    role.label(),
                    elapsed,
                    e,
                );
                summaries.push(WorkerSummary {
                    model,
                    role,
                    succeeded: false,
                    elapsed_ms: elapsed,
                    output_kind: None,
                    confidence: None,
                });

                if let Some(decision) =
                    arbiter::try_early_decision(&outputs, total_workers, total_finished, has_tools)
                {
                    drain_after_early_exit(join_set, &mut summaries).await;
                    reconcile_dispatched(dispatched, &mut summaries);
                    return (outputs, summaries, Some(decision));
                }
            }
            Err(e) => {
                total_finished += 1;
                tracing::warn!("moa: worker task panicked or was cancelled: {e}");
                // No (model, role) payload available from a JoinError, so
                // we cannot attribute this slot here. `reconcile_dispatched`
                // at the end picks up any dispatched worker that has not
                // produced a summary by name.
            }
        }
    }

    reconcile_dispatched(dispatched, &mut summaries);
    (outputs, summaries, None)
}

/// After `abort_all`, drain any tasks that did finish before the abort
/// reached them, recording each as a summary. Aborted tasks produce a
/// `JoinError::cancelled` which carries no `(model, role)` payload —
/// those are reconciled by [`reconcile_dispatched`] using the dispatch
/// list.
async fn drain_after_early_exit(
    join_set: &mut tokio::task::JoinSet<(String, WorkerRole, Result<String, String>, u64)>,
    summaries: &mut Vec<WorkerSummary>,
) {
    join_set.abort_all();
    while let Some(leftover) = join_set.join_next().await {
        if let Ok((m, r, result, el)) = leftover {
            summaries.push(WorkerSummary {
                model: m,
                role: r,
                succeeded: result.is_ok(),
                elapsed_ms: el,
                output_kind: None,
                confidence: None,
            });
        }
    }
}

/// Ensure every dispatched worker appears in `summaries`. Anything we
/// dispatched that didn't produce a summary by name (aborted by
/// early-exit, panicked, or otherwise lost) gets a synthesized
/// `succeeded: false` entry so the `x-moa-workers` header faithfully
/// reflects the dispatched count.
fn reconcile_dispatched(dispatched: &[DispatchedWorker], summaries: &mut Vec<WorkerSummary>) {
    for w in dispatched {
        if summaries.iter().any(|s| s.model == w.model) {
            continue;
        }
        summaries.push(WorkerSummary {
            model: w.model.clone(),
            role: w.role,
            succeeded: false,
            elapsed_ms: 0,
            output_kind: None,
            confidence: None,
        });
    }
}
