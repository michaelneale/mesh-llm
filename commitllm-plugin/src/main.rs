//! CommitLLM verified inference plugin for mesh-llm.
//!
//! Provides cryptographic verification of LLM inference receipts within the mesh.
//! Runs as an external plugin process, communicating with the mesh-llm host via
//! the standard plugin protocol.
//!
//! ## Tools
//!
//! - `commitllm_verify_receipt` — verify a binary receipt against a verifier key
//! - `commitllm_hash_gguf` — compute model identity hash (R_W) from a GGUF file
//! - `commitllm_peer_trust` — show trust scores for mesh peers
//! - `commitllm_status` — show plugin status and verification statistics
//!
//! ## Channel Messages
//!
//! The plugin uses the `commitllm.v1` channel for mesh-wide receipt and
//! challenge traffic between peers.

use anyhow::Result;
use mesh_llm_plugin::{
    json_schema_for,
    plugin_server_info, proto, structured_tool_result, tool_error, tool_with_schema,
    PluginMetadata, PluginRuntime, SimplePlugin, ToolRouter,
};
use rmcp::model::ServerInfo;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use tokio::sync::Mutex;

const PLUGIN_ID: &str = "commitllm";
const CHANNEL: &str = "commitllm.v1";

// ---------------------------------------------------------------------------
// Tool argument / result types
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize, JsonSchema)]
struct VerifyReceiptArgs {
    /// Base64-encoded binary receipt (V4 audit response, bincode+zstd).
    receipt_b64: String,
    /// Base64-encoded verifier key. When absent, uses the cached key for
    /// the model identified in the receipt (if available).
    #[serde(default)]
    verifier_key_b64: Option<String>,
}

#[derive(Debug, Serialize)]
struct VerifyReceiptResult {
    verdict: String,
    audit_coverage: String,
    failures: Vec<FailureEntry>,
    timings_ms: TimingsMs,
    details: serde_json::Value,
}

#[derive(Debug, Serialize)]
struct FailureEntry {
    code: String,
    message: String,
    category: String,
}

#[derive(Debug, Serialize)]
struct TimingsMs {
    structural: f64,
    embedding: f64,
    bridge: f64,
    specs: f64,
    total: f64,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct HashGgufArgs {
    /// Path to the GGUF file on disk.
    path: String,
}

#[derive(Debug, Serialize)]
struct HashGgufResult {
    /// SHA-256 of the entire GGUF file.
    file_hash: String,
    /// Size in bytes.
    file_size: u64,
    /// Detected model name from metadata, if available.
    model_name: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema, Default)]
struct PeerTrustArgs {
    /// Optional peer ID to filter to.
    #[serde(default)]
    peer_id: Option<String>,
}

#[derive(Debug, Serialize)]
struct PeerTrustEntry {
    peer_id: String,
    verifications_passed: u64,
    verifications_failed: u64,
    trust_score: f64,
    last_verified_at: Option<String>,
    models: Vec<String>,
}

#[derive(Debug, Deserialize, JsonSchema, Default)]
struct StatusArgs {}

#[derive(Debug, Serialize)]
struct StatusResult {
    plugin: String,
    local_peer_id: String,
    mesh_id: String,
    total_verifications: u64,
    total_passed: u64,
    total_failed: u64,
    cached_keys: usize,
    known_peers: usize,
    channel: String,
}

// ---------------------------------------------------------------------------
// Receipt message types (for mesh channel transport)
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize, Deserialize)]
struct ReceiptMessage {
    /// The request ID this receipt is for.
    request_id: String,
    /// Model identity claimed by the provider.
    model_id: String,
    /// The binary receipt (base64).
    receipt_b64: String,
    /// Source peer ID of the provider.
    provider_peer_id: String,
}

#[derive(Debug, Serialize, Deserialize)]
#[allow(dead_code)]
struct ChallengeMessage {
    /// The request ID to challenge.
    request_id: String,
    /// Token index to audit.
    token_index: u32,
    /// Layer indices to audit.
    layer_indices: Vec<usize>,
    /// Audit tier: "routine" or "deep".
    tier: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct VerificationResult {
    request_id: String,
    provider_peer_id: String,
    verdict: String,
    model_id: String,
    failures: Vec<String>,
}

// ---------------------------------------------------------------------------
// Plugin state
// ---------------------------------------------------------------------------

struct PluginState {
    local_peer_id: String,
    mesh_id: String,
    /// Known mesh peers from mesh events.
    known_peers: BTreeMap<String, PeerInfo>,
    /// Per-peer trust tracking.
    peer_trust: HashMap<String, TrustScore>,
    /// Cached verifier keys by model identity hash.
    verifier_keys: HashMap<String, Vec<u8>>,
    /// Verification statistics.
    total_verifications: u64,
    total_passed: u64,
    total_failed: u64,
    /// Recent verification results.
    recent_results: Vec<VerificationResult>,
    /// Received receipts awaiting verification.
    pending_receipts: HashMap<String, ReceiptMessage>,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct PeerInfo {
    peer_id: String,
    models: Vec<String>,
    serving_models: Vec<String>,
}

#[derive(Debug, Clone, Default)]
struct TrustScore {
    passed: u64,
    failed: u64,
    last_verified_epoch_ms: u64,
}

impl TrustScore {
    fn score(&self) -> f64 {
        let total = self.passed + self.failed;
        if total == 0 {
            return 0.5; // neutral — no data
        }
        self.passed as f64 / total as f64
    }
}

impl Default for PluginState {
    fn default() -> Self {
        Self {
            local_peer_id: String::new(),
            mesh_id: String::new(),
            known_peers: BTreeMap::new(),
            peer_trust: HashMap::new(),
            verifier_keys: HashMap::new(),
            total_verifications: 0,
            total_passed: 0,
            total_failed: 0,
            recent_results: Vec::new(),
            pending_receipts: HashMap::new(),
        }
    }
}

impl PluginState {
    fn record_mesh_event(&mut self, event: &proto::MeshEvent) {
        if !event.local_peer_id.is_empty() {
            self.local_peer_id = event.local_peer_id.clone();
        }
        if !event.mesh_id.is_empty() {
            self.mesh_id = event.mesh_id.clone();
        }
        if let Some(peer) = &event.peer {
            let peer_id = peer.peer_id.clone();
            match proto::mesh_event::Kind::try_from(event.kind).ok() {
                Some(proto::mesh_event::Kind::PeerDown) => {
                    self.known_peers.remove(&peer_id);
                }
                _ => {
                    self.known_peers.insert(
                        peer_id,
                        PeerInfo {
                            peer_id: peer.peer_id.clone(),
                            models: peer.models.clone(),
                            serving_models: peer.serving_models.clone(),
                        },
                    );
                }
            }
        }
    }

    fn record_verification(&mut self, result: &VerificationResult) {
        self.total_verifications += 1;
        if result.verdict == "pass" {
            self.total_passed += 1;
            let entry = self
                .peer_trust
                .entry(result.provider_peer_id.clone())
                .or_default();
            entry.passed += 1;
            entry.last_verified_epoch_ms = now_millis();
        } else {
            self.total_failed += 1;
            let entry = self
                .peer_trust
                .entry(result.provider_peer_id.clone())
                .or_default();
            entry.failed += 1;
            entry.last_verified_epoch_ms = now_millis();
        }

        // Keep bounded history.
        if self.recent_results.len() >= 1000 {
            self.recent_results.remove(0);
        }
        self.recent_results.push(result.clone());
    }
}

fn now_millis() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

// ---------------------------------------------------------------------------
// Verification logic
// ---------------------------------------------------------------------------

/// Verify a binary receipt using the verilm-verify canonical verifier.
///
/// The receipt is a V4 audit response in bincode+zstd format.
/// The verifier key is deserialized from bincode.
fn verify_receipt_binary(
    receipt_bytes: &[u8],
    key_bytes: &[u8],
) -> Result<verilm_verify::V4VerifyReport> {
    // Deserialize the verifier key.
    let key: verilm_core::types::VerifierKey =
        bincode::deserialize(key_bytes).map_err(|e| anyhow::anyhow!("bad verifier key: {e}"))?;

    // Run canonical verification (no tokenizer/detokenizer — those are optional).
    let report = verilm_verify::canonical::verify_binary(&key, receipt_bytes, None, None)
        .map_err(|e| anyhow::anyhow!("verification failed: {e}"))?;

    Ok(report)
}

/// Compute SHA-256 hash of a GGUF file (model identity).
///
/// This is the simplest form of model identity — just hash the whole file.
/// For witness-mode verification, this is sufficient to detect model
/// substitution. Full Freivalds verification needs weight-level hashing
/// (future work: parse GGUF tensor data and compute R_W).
fn hash_gguf_file(path: &str) -> Result<(String, u64, Option<String>)> {
    use std::io::Read;

    let mut file = std::fs::File::open(path)
        .map_err(|e| anyhow::anyhow!("cannot open {path}: {e}"))?;

    let metadata = file.metadata()?;
    let file_size = metadata.len();

    // Hash the entire file.
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1024 * 1024]; // 1MB buffer
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    let hash = hex::encode(hasher.finalize());

    // Try to extract model name from GGUF metadata.
    // GGUF header: magic (4) + version (4) + tensor_count (8) + metadata_kv_count (8)
    // We don't do a full parse here — just return None for now.
    // A proper implementation would read the metadata KV pairs.
    let model_name: Option<String> = None;

    Ok((hash, file_size, model_name))
}

// ---------------------------------------------------------------------------
// Plugin construction
// ---------------------------------------------------------------------------

fn server_info() -> ServerInfo {
    plugin_server_info(
        PLUGIN_ID,
        env!("CARGO_PKG_VERSION"),
        "CommitLLM Verified Inference",
        "Cryptographic verification of LLM inference receipts within the mesh",
        None::<String>,
    )
}

fn build_plugin(state: Arc<Mutex<PluginState>>) -> SimplePlugin {
    let mesh_event_state = state.clone();
    let channel_state = state.clone();
    let health_state = state.clone();

    SimplePlugin::new(
        PluginMetadata::new(PLUGIN_ID, env!("CARGO_PKG_VERSION"), server_info())
            .with_capabilities(vec![
                format!("channel:{CHANNEL}"),
                "mesh-events".into(),
            ]),
    )
    .with_tool_router(tool_router(state.clone()))
    .with_health(move |_context| {
        let state = health_state.clone();
        Box::pin(async move {
            let state = state.lock().await;
            Ok(format!(
                "verified={}/{} passed={} failed={} peers={} keys={}",
                state.total_verifications,
                state.total_passed + state.total_failed,
                state.total_passed,
                state.total_failed,
                state.known_peers.len(),
                state.verifier_keys.len(),
            ))
        })
    })
    .on_mesh_event(move |event, _context| {
        let state = mesh_event_state.clone();
        Box::pin(async move {
            state.lock().await.record_mesh_event(&event);
            Ok(())
        })
    })
    .on_channel_message(move |message, _context| {
        let state = channel_state.clone();
        Box::pin(async move {
            if message.channel != CHANNEL {
                return Ok(());
            }

            // Parse the message body as JSON.
            let body: serde_json::Value = match serde_json::from_slice(&message.body) {
                Ok(v) => v,
                Err(_) => return Ok(()),
            };

            let kind = message.message_kind.as_str();
            match kind {
                "receipt" => {
                    // A peer is sharing a receipt for us to verify.
                    if let Ok(receipt_msg) = serde_json::from_value::<ReceiptMessage>(body) {
                        let request_id = receipt_msg.request_id.clone();
                        let mut state = state.lock().await;
                        state
                            .pending_receipts
                            .insert(request_id.clone(), receipt_msg);

                        // Trim pending if too many.
                        while state.pending_receipts.len() > 10_000 {
                            if let Some(oldest) =
                                state.pending_receipts.keys().next().cloned()
                            {
                                state.pending_receipts.remove(&oldest);
                            }
                        }
                    }
                }
                "verification_result" => {
                    // A peer is sharing their verification result.
                    if let Ok(vr) = serde_json::from_value::<VerificationResult>(body) {
                        let mut state = state.lock().await;
                        state.record_verification(&vr);
                    }
                }
                _ => {}
            }

            Ok(())
        })
    })
}

fn tool_router(state: Arc<Mutex<PluginState>>) -> ToolRouter {
    let mut router = ToolRouter::new();

    // --- commitllm_verify_receipt ---
    let verify_state = state.clone();
    router.add_raw(
        tool_with_schema(
            "commitllm_verify_receipt",
            "Verify a CommitLLM binary receipt against a verifier key. \
             Returns the verification verdict (pass/fail), audit coverage, \
             any failure details, and timing information.",
            json_schema_for::<VerifyReceiptArgs>(),
        ),
        move |request, _context| {
            let state = verify_state.clone();
            Box::pin(async move {
                let args: VerifyReceiptArgs = request.arguments()?;

                // Decode the receipt.
                let receipt_bytes = base64_decode(&args.receipt_b64)
                    .map_err(|e| mesh_llm_plugin::PluginError::invalid_params(
                        format!("bad receipt_b64: {e}"),
                    ))?;

                // Get the verifier key.
                let key_bytes = if let Some(ref key_b64) = args.verifier_key_b64 {
                    base64_decode(key_b64)
                        .map_err(|e| mesh_llm_plugin::PluginError::invalid_params(
                            format!("bad verifier_key_b64: {e}"),
                        ))?
                } else {
                    // Try cached keys (future: auto-detect model from receipt).
                    let state = state.lock().await;
                    if state.verifier_keys.is_empty() {
                        return Ok(tool_error(
                            "No verifier key provided and no cached keys available. \
                             Pass verifier_key_b64 or load a key first.",
                        ));
                    }
                    // For now, just use the first cached key.
                    state.verifier_keys.values().next().unwrap().clone()
                };

                // Run verification.
                match verify_receipt_binary(&receipt_bytes, &key_bytes) {
                    Ok(report) => {
                        let verdict = format!("{:?}", report.verdict).to_lowercase();
                        let coverage = format!("{}", report.coverage);

                        let failures: Vec<FailureEntry> = report
                            .failures
                            .iter()
                            .map(|f| FailureEntry {
                                code: format!("{:?}", f.code),
                                message: f.message.clone(),
                                category: format!("{:?}", f.category),
                            })
                            .collect();

                        let total_ms = report.duration.as_secs_f64() * 1000.0;
                        let timings = TimingsMs {
                            structural: 0.0,
                            embedding: 0.0,
                            bridge: 0.0,
                            specs: 0.0,
                            total: total_ms,
                        };

                        // Record the result for trust scoring.
                        // (In a full implementation, we'd extract the provider peer ID
                        //  from the receipt metadata.)
                        let mut state_guard = state.lock().await;
                        state_guard.total_verifications += 1;
                        if verdict == "pass" {
                            state_guard.total_passed += 1;
                        } else {
                            state_guard.total_failed += 1;
                        }

                        let result = VerifyReceiptResult {
                            verdict,
                            audit_coverage: coverage,
                            failures,
                            timings_ms: timings,
                            details: json!({
                                "checks_run": report.checks_run,
                                "checks_passed": report.checks_passed,
                                "n_failures": report.failures.len(),
                                "token_index": report.token_index,
                            }),
                        };

                        structured_tool_result(result)
                    }
                    Err(e) => Ok(tool_error(format!("Verification error: {e}"))),
                }
            })
        },
    );

    // --- commitllm_hash_gguf ---
    router.add_raw(
        tool_with_schema(
            "commitllm_hash_gguf",
            "Compute the SHA-256 identity hash of a GGUF model file. \
             This is the simplest model identity check — if two nodes claim \
             to serve the same model, their GGUF hashes must match.",
            json_schema_for::<HashGgufArgs>(),
        ),
        move |request, _context| {
            Box::pin(async move {
                let args: HashGgufArgs = request.arguments()?;
                match hash_gguf_file(&args.path) {
                    Ok((hash, size, name)) => {
                        structured_tool_result(HashGgufResult {
                            file_hash: hash,
                            file_size: size,
                            model_name: name,
                        })
                    }
                    Err(e) => Ok(tool_error(format!("Hash error: {e}"))),
                }
            })
        },
    );

    // --- commitllm_peer_trust ---
    let trust_state = state.clone();
    router.add_raw(
        tool_with_schema(
            "commitllm_peer_trust",
            "Show trust scores for mesh peers based on verification history. \
             Peers with more passed verifications get higher trust scores. \
             Scores range from 0.0 (all failed) to 1.0 (all passed), \
             with 0.5 meaning no verification data.",
            json_schema_for::<PeerTrustArgs>(),
        ),
        move |request, _context| {
            let state = trust_state.clone();
            Box::pin(async move {
                let args: PeerTrustArgs = request.arguments_or_default()?;
                let state = state.lock().await;

                let mut entries: Vec<PeerTrustEntry> = Vec::new();

                for (peer_id, info) in &state.known_peers {
                    if let Some(ref filter) = args.peer_id {
                        if peer_id != filter {
                            continue;
                        }
                    }

                    let trust = state
                        .peer_trust
                        .get(peer_id)
                        .cloned()
                        .unwrap_or_default();

                    entries.push(PeerTrustEntry {
                        peer_id: peer_id.clone(),
                        verifications_passed: trust.passed,
                        verifications_failed: trust.failed,
                        trust_score: trust.score(),
                        last_verified_at: if trust.last_verified_epoch_ms > 0 {
                            Some(format!("{}ms", trust.last_verified_epoch_ms))
                        } else {
                            None
                        },
                        models: info.serving_models.clone(),
                    });
                }

                // Sort by trust score descending (most trusted first).
                entries.sort_by(|a, b| b.trust_score.partial_cmp(&a.trust_score).unwrap());

                structured_tool_result(json!({
                    "peers": entries,
                    "total_known_peers": state.known_peers.len(),
                    "total_with_verification_data": state.peer_trust.len(),
                }))
            })
        },
    );

    // --- commitllm_status ---
    let status_state = state.clone();
    router.add_raw(
        tool_with_schema(
            "commitllm_status",
            "Show CommitLLM plugin status including verification statistics, \
             cached keys, and mesh connectivity.",
            json_schema_for::<StatusArgs>(),
        ),
        move |request, _context| {
            let state = status_state.clone();
            Box::pin(async move {
                let _args: StatusArgs = request.arguments_or_default()?;
                let state = state.lock().await;

                structured_tool_result(StatusResult {
                    plugin: PLUGIN_ID.to_string(),
                    local_peer_id: state.local_peer_id.clone(),
                    mesh_id: state.mesh_id.clone(),
                    total_verifications: state.total_verifications,
                    total_passed: state.total_passed,
                    total_failed: state.total_failed,
                    cached_keys: state.verifier_keys.len(),
                    known_peers: state.known_peers.len(),
                    channel: CHANNEL.to_string(),
                })
            })
        },
    );

    router
}

fn base64_decode(s: &str) -> Result<Vec<u8>> {
    use base64::Engine;
    base64::engine::general_purpose::STANDARD
        .decode(s)
        .map_err(|e| anyhow::anyhow!("base64 decode: {e}"))
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> Result<()> {
    let state = Arc::new(Mutex::new(PluginState::default()));
    PluginRuntime::run(build_plugin(state)).await
}
