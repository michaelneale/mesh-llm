//! Worker fan-out and incremental gathering.
//!
//! Workers run in parallel via [`tokio::task::JoinSet`]. As each one
//! completes, the result is normalized, allowed-tool filtered, and fed
//! to the arbiter's early-exit check. If the arbiter can already decide
//! from the responses collected so far (consensus, sole survivor, etc.)
//! the remaining workers are cancelled and the decision is returned
//! immediately.

use crate::enforce_allowed_tools;
use crate::worker::WorkerRole;
use crate::{arbiter, normalize, WorkerSummary};
use normalize::WorkerOutput;

pub(crate) async fn gather_workers_incremental(
    join_set: &mut tokio::task::JoinSet<(String, WorkerRole, Result<String, String>, u64)>,
    total_workers: usize,
    has_tools: bool,
    allowed_tools: &[String],
) -> (
    Vec<WorkerOutput>,
    Vec<WorkerSummary>,
    Option<arbiter::Decision>,
) {
    let mut outputs = Vec::new();
    let mut summaries = Vec::new();
    let mut total_finished: usize = 0;

    while let Some(join_result) = join_set.join_next().await {
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
                    return (outputs, summaries, Some(decision));
                }
            }
            Err(e) => {
                total_finished += 1;
                tracing::warn!("moa: worker task panicked: {e}");
            }
        }
    }

    (outputs, summaries, None)
}
