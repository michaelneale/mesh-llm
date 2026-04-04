//! CommitLLM challenge-response verification for remote mesh peers.
//!
//! When a proxy starts routing inference to a remote host, it can run a
//! challenge to verify the host is computing matmuls honestly. The flow:
//!
//! 1. POST /commitllm/challenge/begin  — enables capture on remote llama-server
//! 2. POST /completion {prompt, n_predict:1} — runs one token of inference
//! 3. POST /commitllm/challenge/end    — returns per-matmul INT32 accumulators
//! 4. Verify captures structurally (Freivalds algebraic check planned)
//!
//! After verification, capture turns off and inference runs at full speed.
//!
//! Triggered by the election loop when setting `InferenceTarget::Remote` or
//! `InferenceTarget::MoeRemote`. Local targets are not challenged — we trust
//! our own llama-server process.
//!
//! For pipeline-split (RPC) scenarios, only the host is challenged. RPC workers
//! do delegated matrix math that the host orchestrates — per-worker verification
//! is not possible through this mechanism.

use crate::mesh;
use anyhow::{bail, Context, Result};
use serde::Deserialize;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

/// Per-matmul captures returned by llama-server's challenge/end endpoint.
#[derive(Debug, Deserialize)]
pub struct ChallengeResponse {
    pub n_matmuls: usize,
    pub matmuls: Vec<MatmulCapture>,
}

#[derive(Debug, Deserialize)]
pub struct MatmulCapture {
    pub n_rows: usize,
    pub n_blocks: usize,
    pub z: Vec<i32>,
}

/// Verification state for a peer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VerifyState {
    /// Challenge passed — peer is trusted until TTL expires.
    Verified { at_ms: u64 },
    /// Challenge was attempted and failed — peer is untrusted.
    Failed { at_ms: u64 },
    /// Peer doesn't support challenge endpoints (e.g. Metal build, old llama.cpp).
    Unsupported,
}

/// Tracks which remote peers have been challenged and their results.
#[derive(Clone)]
pub struct VerificationTracker {
    inner: Arc<Mutex<TrackerInner>>,
    /// When true, only Verified peers are routable.
    pub require_verification: bool,
}

struct TrackerInner {
    peers: HashMap<iroh::EndpointId, VerifyState>,
}

/// How long a verification stays valid before re-challenge (10 minutes).
const VERIFY_TTL_MS: u64 = 10 * 60 * 1000;

impl VerificationTracker {
    pub fn new(require_verification: bool) -> Self {
        Self {
            inner: Arc::new(Mutex::new(TrackerInner {
                peers: HashMap::new(),
            })),
            require_verification,
        }
    }

    /// Returns true if this peer has been verified recently (within TTL).
    pub async fn is_verified(&self, peer_id: iroh::EndpointId) -> bool {
        let inner = self.inner.lock().await;
        matches!(
            inner.peers.get(&peer_id),
            Some(VerifyState::Verified { at_ms }) if now_millis().saturating_sub(*at_ms) < VERIFY_TTL_MS
        )
    }

    /// Returns true if the proxy should skip this peer.
    /// Only returns true when `require_verification` is enabled.
    pub async fn should_skip(&self, peer_id: iroh::EndpointId) -> bool {
        if !self.require_verification {
            return false;
        }
        !self.is_verified(peer_id).await
    }

    /// Check if a challenge needs to run (not yet verified, or TTL expired).
    pub async fn needs_challenge(&self, peer_id: iroh::EndpointId) -> bool {
        let inner = self.inner.lock().await;
        match inner.peers.get(&peer_id) {
            Some(VerifyState::Verified { at_ms }) => {
                now_millis().saturating_sub(*at_ms) >= VERIFY_TTL_MS
            }
            Some(VerifyState::Unsupported) => false, // no point re-trying
            _ => true,
        }
    }

    pub async fn mark_verified(&self, peer_id: iroh::EndpointId) {
        self.inner.lock().await.peers.insert(
            peer_id,
            VerifyState::Verified {
                at_ms: now_millis(),
            },
        );
    }

    pub async fn mark_failed(&self, peer_id: iroh::EndpointId) {
        self.inner.lock().await.peers.insert(
            peer_id,
            VerifyState::Failed {
                at_ms: now_millis(),
            },
        );
    }

    pub async fn mark_unsupported(&self, peer_id: iroh::EndpointId) {
        self.inner
            .lock()
            .await
            .peers
            .insert(peer_id, VerifyState::Unsupported);
    }
}

fn now_millis() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

/// Run a challenge against a remote peer's llama-server via QUIC tunnel.
///
/// Sends begin → inference → end through the mesh tunnel, then returns
/// the captured matmul data for structural verification.
pub async fn run_challenge_remote(
    node: &mesh::Node,
    peer_id: iroh::EndpointId,
) -> Result<ChallengeResponse> {
    // Step 1: Begin challenge
    let begin_resp = tunnel_post(node, peer_id, "/commitllm/challenge/begin", None)
        .await
        .context("challenge/begin via tunnel")?;
    if !begin_resp.status.starts_with('2') {
        bail!(
            "remote challenge/begin returned status {}",
            begin_resp.status
        );
    }

    // Step 2: Run minimal inference (1 token).
    // Always attempt step 3 even if this fails, so capture gets disabled.
    let inference_result: Result<()> = async {
        let resp = tunnel_post(
            node,
            peer_id,
            "/completion",
            Some(r#"{"prompt":"a","n_predict":1}"#),
        )
        .await
        .context("challenge inference via tunnel")?;
        if !resp.status.starts_with('2') {
            bail!("remote challenge inference returned status {}", resp.status);
        }
        Ok(())
    }
    .await;

    // Step 3: End challenge (always attempted — capture is already enabled)
    let end_result = tunnel_post(node, peer_id, "/commitllm/challenge/end", None).await;

    if let Err(err) = inference_result {
        if let Err(end_err) = &end_result {
            tracing::warn!(
                "failed to POST challenge/end during cleanup on {}: {end_err}",
                peer_id.fmt_short()
            );
        }
        return Err(err);
    }

    let end_resp = end_result.context("challenge/end via tunnel")?;
    if !end_resp.status.starts_with('2') {
        bail!("remote challenge/end returned status {}", end_resp.status);
    }

    let captures: ChallengeResponse = serde_json::from_str(&end_resp.body)
        .context("failed to parse remote challenge/end JSON")?;

    tracing::info!(
        "remote challenge complete ({}): {} matmuls, {} total z values",
        peer_id.fmt_short(),
        captures.n_matmuls,
        captures.matmuls.iter().map(|m| m.z.len()).sum::<usize>()
    );

    Ok(captures)
}

/// Spawn a background challenge against a remote peer, updating the tracker
/// on success/failure. Non-blocking — the caller doesn't wait.
pub fn spawn_challenge(
    node: mesh::Node,
    peer_id: iroh::EndpointId,
    tracker: VerificationTracker,
    model_name: String,
) {
    tokio::spawn(async move {
        if !tracker.needs_challenge(peer_id).await {
            return;
        }
        tracing::info!(
            "🔍 [{model_name}] Running verification challenge on {}...",
            peer_id.fmt_short()
        );
        match run_challenge_remote(&node, peer_id).await {
            Ok(captures) => match verify_captures(&captures) {
                Ok(result) => {
                    tracing::info!(
                        "✅ [{model_name}] Peer {} verified: {result}",
                        peer_id.fmt_short()
                    );
                    tracker.mark_verified(peer_id).await;
                }
                Err(err) => {
                    tracing::warn!(
                        "❌ [{model_name}] Peer {} failed verification: {err}",
                        peer_id.fmt_short()
                    );
                    tracker.mark_failed(peer_id).await;
                }
            },
            Err(err) => {
                // Challenge endpoints not available — peer is running a llama.cpp
                // build without CommitLLM capture (e.g. Metal, old version).
                // Mark as unsupported so we don't retry, but distinguish from failure.
                tracing::debug!(
                    "[{model_name}] Peer {} does not support verification challenges: {err}",
                    peer_id.fmt_short()
                );
                tracker.mark_unsupported(peer_id).await;
            }
        }
    });
}

// ---------------------------------------------------------------------------
// QUIC tunnel HTTP helper
// ---------------------------------------------------------------------------

/// Minimal parsed HTTP response from a tunnel POST.
struct TunnelResponse {
    status: String,
    body: String,
}

/// Send a POST request through the QUIC tunnel to a peer's llama-server.
async fn tunnel_post(
    node: &mesh::Node,
    peer_id: iroh::EndpointId,
    path: &str,
    body: Option<&str>,
) -> Result<TunnelResponse> {
    let (mut send, mut recv) = node.open_http_tunnel(peer_id).await?;

    let content = body.unwrap_or("");
    let content_type = if body.is_some() {
        "Content-Type: application/json\r\n"
    } else {
        ""
    };
    let request = format!(
        "POST {path} HTTP/1.1\r\n\
         Host: localhost\r\n\
         {content_type}\
         Content-Length: {}\r\n\
         Connection: close\r\n\
         \r\n\
         {content}",
        content.len()
    );
    send.write_all(request.as_bytes()).await?;
    send.finish()?;

    let response_bytes = recv.read_to_end(64 * 1024 * 1024).await?;
    let response_str = String::from_utf8_lossy(&response_bytes);

    let status_line = response_str.lines().next().unwrap_or("").to_string();
    let status = status_line
        .split_whitespace()
        .nth(1)
        .unwrap_or("0")
        .to_string();

    let body_str = if let Some(pos) = response_str.find("\r\n\r\n") {
        response_str[pos + 4..].to_string()
    } else {
        String::new()
    };

    Ok(TunnelResponse {
        status,
        body: body_str,
    })
}

// ---------------------------------------------------------------------------
// Structural verification
// ---------------------------------------------------------------------------

/// Verify challenge captures using structural checks.
///
/// Currently validates:
/// - The challenge reported at least one matmul.
/// - `n_matmuls` matches the actual number of captures returned.
/// - Each capture has `z.len() == n_rows`.
/// - Not more than half of the captures have all-zero accumulators.
///
/// Does not yet validate expected matmul counts per model architecture or
/// `n_blocks` ranges. Full algebraic verification (`v · x == r · z`) requires
/// a verifier key generated from the model weights — will be added when
/// verilm-core keygen is integrated.
pub fn verify_captures(captures: &ChallengeResponse) -> Result<VerifyResult> {
    if captures.matmuls.is_empty() {
        bail!("challenge returned zero matmuls — capture may not be working");
    }

    if captures.n_matmuls != captures.matmuls.len() {
        bail!(
            "n_matmuls={} but matmuls.len()={} — inconsistent response",
            captures.n_matmuls,
            captures.matmuls.len()
        );
    }

    let n_matmuls = captures.matmuls.len();
    let mut total_z = 0usize;
    let mut all_zero_count = 0usize;

    for (i, m) in captures.matmuls.iter().enumerate() {
        if m.z.len() != m.n_rows {
            bail!("matmul {i}: z.len()={} != n_rows={}", m.z.len(), m.n_rows);
        }
        total_z += m.z.len();

        if m.z.iter().all(|&v| v == 0) {
            all_zero_count += 1;
        }
    }

    if all_zero_count > n_matmuls / 2 {
        bail!("{all_zero_count}/{n_matmuls} matmuls have all-zero accumulators — suspicious");
    }

    Ok(VerifyResult {
        n_matmuls: captures.n_matmuls,
        total_z_values: total_z,
        passed: true,
    })
}

#[derive(Debug)]
pub struct VerifyResult {
    pub n_matmuls: usize,
    pub total_z_values: usize,
    pub passed: bool,
}

impl std::fmt::Display for VerifyResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "verify: {} matmuls, {} z values, {}",
            self.n_matmuls,
            self.total_z_values,
            if self.passed { "PASS" } else { "FAIL" }
        )
    }
}
