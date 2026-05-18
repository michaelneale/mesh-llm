//! Reducer candidate selection + hedged call ladder.
//!
//! The reducer is invoked when arbitration can't reach a decision from
//! worker outputs alone. Rather than picking a single reducer model and
//! eating its timeout when it's slow/broken, we keep an ordered ladder
//! of candidates (big-tier first, small-tier as last resort) and call
//! them with hedging: start the first, hedge to the next after
//! `hedge_delay`, race for the first OK. On fast errors, jump to the
//! next candidate immediately.

use crate::backend::{call_backend, ModelBackend, SamplingParams};
use crate::worker;
use crate::GatewayConfig;
use serde_json::Value;
use std::sync::Arc;
use std::time::Duration;

/// Pick the reducer — prefers first model (typically local, zero RTT).
/// Reducer candidates in priority order: big-tier models first (multi-
/// digit B, or names with no size like MiniMax), then small-tier models
/// as last-resort fallback. Callers should try each in order and stop
/// on the first that succeeds, so a broken big-tier peer (e.g. a peer
/// running a stale binary that 502s on tool calls) doesn't take down
/// the whole reducer step.
pub(crate) fn reducer_candidates(config: &GatewayConfig) -> Vec<(String, usize)> {
    let mut big = Vec::new();
    let mut small = Vec::new();
    for m in &config.models {
        let entry = (m.name.clone(), m.backend_index);
        if worker::is_single_digit_b_name(&m.name) {
            small.push(entry);
        } else {
            big.push(entry);
        }
    }
    big.extend(small);
    if big.is_empty() {
        big.push(("unknown".into(), 0));
    }
    big
}

/// Successful hedged-reducer outcome.
///
/// `attempts` reports how many candidates were actually spawned:
/// `1` = clean happy path (cand 0 returned before hedge fired),
/// `≥2` = hedge fired or a fast-fail cascaded to the next candidate.
#[derive(Debug)]
pub(crate) struct HedgedReducerOk {
    pub winner: String,
    pub text: String,
    pub attempts: u32,
}

/// Call the ordered reducer candidates with hedging.
///
/// Starts the first candidate immediately. If it hasn't returned within
/// `hedge_delay`, the next candidate is started in parallel without
/// cancelling the in-flight one — we race for the first OK. If a candidate
/// errors, the next one is started immediately (no hedge wait).
///
/// Returns the first successful [`HedgedReducerOk`]. If every candidate
/// fails, returns the last error encountered.
///
/// Cost shape:
/// - Happy path (cand 0 OK in <hedge_delay): exactly 1 backend call.
/// - Slow happy path (cand 0 OK in hedge_delay..reducer_timeout): up to 2
///   overlapping calls, accept whichever wins, cancel the loser.
/// - Fast-fail (cand 0 errors quickly): immediate move to cand 1, 1 call.
/// - All fail: at most N calls, capped at reducer_timeout + (N-1)·hedge_delay
///   end-to-end (vs N·reducer_timeout sequentially).
pub(crate) async fn hedged_reducer_call(
    backends: &[Arc<dyn ModelBackend>],
    candidates: Vec<(String, usize)>,
    messages: Vec<Value>,
    tools: Option<Value>,
    timeout: Duration,
    hedge_delay: Duration,
) -> Result<HedgedReducerOk, String> {
    use tokio::task::JoinSet;

    if candidates.is_empty() {
        return Err("no reducer candidates".into());
    }

    let mut join_set: JoinSet<(String, Result<String, String>)> = JoinSet::new();
    let mut remaining = candidates.into_iter();
    let mut last_err: Option<String> = None;
    let mut attempts: u32 = 0;

    // Spawn a single candidate.
    fn spawn(
        join_set: &mut JoinSet<(String, Result<String, String>)>,
        backends: &[Arc<dyn ModelBackend>],
        name: String,
        backend_idx: usize,
        messages: Vec<Value>,
        tools: Option<Value>,
        timeout: Duration,
    ) {
        let backend = backends[backend_idx].clone();
        tracing::info!("moa: reducer hedge → {name}");
        join_set.spawn(async move {
            let result = call_backend(
                &*backend,
                &name,
                &messages,
                tools.as_ref(),
                2048,
                timeout,
                SamplingParams::reducer(),
            )
            .await;
            (name, result)
        });
    }

    // Start candidate 0.
    if let Some((name, idx)) = remaining.next() {
        spawn(
            &mut join_set,
            backends,
            name,
            idx,
            messages.clone(),
            tools.clone(),
            timeout,
        );
        attempts += 1;
    }

    // Race in-flight calls against a hedge timer.
    while !join_set.is_empty() {
        let hedge_sleep = tokio::time::sleep(hedge_delay);
        tokio::pin!(hedge_sleep);

        tokio::select! {
            // A candidate finished.
            joined = join_set.join_next() => {
                match joined {
                    Some(Ok((name, Ok(text)))) => {
                        // First success wins. Cancel the rest.
                        join_set.abort_all();
                        // Drain so cancellations complete cleanly.
                        while join_set.join_next().await.is_some() {}
                        return Ok(HedgedReducerOk {
                            winner: name,
                            text,
                            attempts,
                        });
                    }
                    Some(Ok((name, Err(e)))) => {
                        tracing::warn!(
                            "moa: reducer {name} failed: {e}, trying next candidate"
                        );
                        last_err = Some(e);
                        // Start the next candidate immediately on failure.
                        if let Some((next_name, next_idx)) = remaining.next() {
                            spawn(
                                &mut join_set,
                                backends,
                                next_name,
                                next_idx,
                                messages.clone(),
                                tools.clone(),
                                timeout,
                            );
                            attempts += 1;
                        }
                    }
                    Some(Err(join_err)) => {
                        tracing::warn!("moa: reducer task join error: {join_err}");
                        if let Some((next_name, next_idx)) = remaining.next() {
                            spawn(
                                &mut join_set,
                                backends,
                                next_name,
                                next_idx,
                                messages.clone(),
                                tools.clone(),
                                timeout,
                            );
                            attempts += 1;
                        }
                    }
                    None => break,
                }
            }
            // Hedge timer fires: start another candidate alongside in-flight ones.
            _ = &mut hedge_sleep => {
                if let Some((next_name, next_idx)) = remaining.next() {
                    spawn(
                        &mut join_set,
                        backends,
                        next_name,
                        next_idx,
                        messages.clone(),
                        tools.clone(),
                        timeout,
                    );
                    attempts += 1;
                }
                // If no more to start, just wait on the JoinSet without the
                // hedge timer racing again (next loop iteration's sleep will
                // simply never fire because we'll take the join branch).
            }
        }
    }

    Err(last_err.unwrap_or_else(|| "all reducer candidates failed".into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use serde_json::json;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[derive(Clone)]
    enum FakeBehavior {
        OkAfter(Duration, String),
        ErrAfter(Duration, String),
    }

    struct FakeBackend {
        behaviors: std::sync::Mutex<std::collections::HashMap<String, FakeBehavior>>,
        calls: AtomicUsize,
    }

    impl FakeBackend {
        fn new(behaviors: Vec<(&str, FakeBehavior)>) -> Arc<Self> {
            let mut map = std::collections::HashMap::new();
            for (n, b) in behaviors {
                map.insert(n.to_string(), b);
            }
            Arc::new(FakeBackend {
                behaviors: std::sync::Mutex::new(map),
                calls: AtomicUsize::new(0),
            })
        }
        fn calls(&self) -> usize {
            self.calls.load(Ordering::SeqCst)
        }
    }

    #[async_trait]
    impl ModelBackend for FakeBackend {
        async fn chat_completion(
            &self,
            model: &str,
            _messages: &[Value],
            _tools: Option<&Value>,
            _max_tokens: u32,
            _timeout: Duration,
            _sampling: SamplingParams,
        ) -> Result<Value, String> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let behavior = self.behaviors.lock().unwrap().get(model).cloned();
            match behavior {
                Some(FakeBehavior::OkAfter(d, body)) => {
                    tokio::time::sleep(d).await;
                    Ok(json!({
                        "choices": [{"message": {"content": body}}],
                    }))
                }
                Some(FakeBehavior::ErrAfter(d, msg)) => {
                    tokio::time::sleep(d).await;
                    Err(msg)
                }
                None => Err(format!("unconfigured model: {model}")),
            }
        }
    }

    #[tokio::test]
    async fn hedged_reducer_happy_path_calls_only_first() {
        let fake = FakeBackend::new(vec![
            (
                "alpha",
                FakeBehavior::OkAfter(Duration::from_millis(50), "alpha-resp".into()),
            ),
            (
                "beta",
                FakeBehavior::OkAfter(Duration::from_millis(50), "beta-resp".into()),
            ),
        ]);
        let backends: Vec<Arc<dyn ModelBackend>> = vec![fake.clone(), fake.clone()];
        let candidates = vec![("alpha".into(), 0), ("beta".into(), 1)];

        let res = hedged_reducer_call(
            &backends,
            candidates,
            vec![],
            None,
            Duration::from_secs(15),
            Duration::from_secs(5),
        )
        .await;

        let ok = res.expect("happy path returns Ok");
        assert_eq!(ok.winner, "alpha", "first candidate should win");
        assert_eq!(ok.attempts, 1, "happy path spawns exactly one candidate");
        assert_eq!(fake.calls(), 1, "only one backend call on happy path");
    }

    #[tokio::test]
    async fn hedged_reducer_slow_first_hedges_to_second() {
        let fake = FakeBackend::new(vec![
            // alpha takes longer than hedge_delay; beta is fast.
            (
                "alpha",
                FakeBehavior::OkAfter(Duration::from_millis(800), "alpha-late".into()),
            ),
            (
                "beta",
                FakeBehavior::OkAfter(Duration::from_millis(100), "beta-fast".into()),
            ),
        ]);
        let backends: Vec<Arc<dyn ModelBackend>> = vec![fake.clone(), fake.clone()];
        let candidates = vec![("alpha".into(), 0), ("beta".into(), 1)];

        let res = hedged_reducer_call(
            &backends,
            candidates,
            vec![],
            None,
            Duration::from_secs(15),
            Duration::from_millis(100),
        )
        .await;

        let ok = res.expect("hedge returns Ok");
        assert_eq!(
            ok.winner, "beta",
            "hedge winner should be the faster second candidate"
        );
        assert_eq!(ok.text, "beta-fast");
        assert_eq!(ok.attempts, 2, "hedge fires the second candidate");
        assert_eq!(fake.calls(), 2, "both candidates should have been issued");
    }

    #[tokio::test]
    async fn hedged_reducer_fast_fail_starts_next_immediately() {
        let fake = FakeBackend::new(vec![
            (
                "alpha",
                FakeBehavior::ErrAfter(Duration::from_millis(50), "boom".into()),
            ),
            (
                "beta",
                FakeBehavior::OkAfter(Duration::from_millis(100), "beta-ok".into()),
            ),
        ]);
        let backends: Vec<Arc<dyn ModelBackend>> = vec![fake.clone(), fake.clone()];
        let candidates = vec![("alpha".into(), 0), ("beta".into(), 1)];

        let start = tokio::time::Instant::now();
        let res = hedged_reducer_call(
            &backends,
            candidates,
            vec![],
            None,
            Duration::from_secs(15),
            // Large hedge_delay — the fast-fail path must not wait for it.
            Duration::from_secs(60),
        )
        .await;

        let ok = res.expect("fail-then-recover returns Ok");
        let elapsed = start.elapsed();
        assert_eq!(ok.winner, "beta");
        assert_eq!(ok.text, "beta-ok");
        assert_eq!(
            ok.attempts, 2,
            "fast-fail should cascade to a second attempt"
        );
        assert_eq!(fake.calls(), 2);
        assert!(
            elapsed < Duration::from_secs(10),
            "fast-fail should not wait for hedge_delay; took {elapsed:?}"
        );
    }

    #[tokio::test]
    async fn hedged_reducer_all_fail_returns_last_err() {
        let fake = FakeBackend::new(vec![
            (
                "alpha",
                FakeBehavior::ErrAfter(Duration::from_millis(10), "alpha-boom".into()),
            ),
            (
                "beta",
                FakeBehavior::ErrAfter(Duration::from_millis(10), "beta-boom".into()),
            ),
        ]);
        let backends: Vec<Arc<dyn ModelBackend>> = vec![fake.clone(), fake.clone()];
        let candidates = vec![("alpha".into(), 0), ("beta".into(), 1)];

        let res = hedged_reducer_call(
            &backends,
            candidates,
            vec![],
            None,
            Duration::from_secs(15),
            Duration::from_millis(200),
        )
        .await;

        let err = res.expect_err("all-fail returns Err");
        assert!(
            err.contains("boom"),
            "should surface a backend error: {err}"
        );
        assert_eq!(fake.calls(), 2);
    }
}
