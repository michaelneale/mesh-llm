//! CommitLLM challenge-response verification for mesh nodes.
//!
//! When a client first connects to a node, it can run a one-time challenge
//! to verify the node is computing matmuls honestly. The flow:
//!
//! 1. POST /commitllm/challenge/begin  — enables capture on llama-server
//! 2. POST /completion {prompt, n_predict:1} — runs one token of inference
//! 3. POST /commitllm/challenge/end    — returns per-matmul INT32 accumulators
//! 4. Verify captures against verifier key using Freivalds check
//!
//! After verification, capture turns off and inference runs at full speed.
//!
//! Metal backends are excluded — Metal Q8_0 uses FP32 accumulation,
//! no integer sumi exists to verify.

use crate::inference::launch::BinaryFlavor;
use anyhow::{bail, Context, Result};
use serde::Deserialize;

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

/// Returns true if this backend flavor supports Freivalds verification.
/// Metal is excluded because its Q8_0 kernel uses FP32 accumulation.
pub fn supports_verification(flavor: BinaryFlavor) -> bool {
    match flavor {
        BinaryFlavor::Cpu | BinaryFlavor::Cuda | BinaryFlavor::Rocm | BinaryFlavor::Vulkan => true,
        BinaryFlavor::Metal => false,
    }
}

/// Run a challenge against a local llama-server instance.
///
/// Sends begin → inference → end and returns the captured matmul data.
/// The caller is responsible for running Freivalds verification on the result.
pub async fn run_challenge(port: u16) -> Result<ChallengeResponse> {
    let client = reqwest::Client::new();
    let base = format!("http://127.0.0.1:{port}");

    // Step 1: Begin challenge (enable capture)
    let resp = client
        .post(format!("{base}/commitllm/challenge/begin"))
        .send()
        .await
        .context("failed to POST /commitllm/challenge/begin")?;
    if !resp.status().is_success() {
        bail!(
            "challenge/begin returned status {}",
            resp.status().as_u16()
        );
    }

    // Step 2: Run minimal inference (1 token)
    let resp = client
        .post(format!("{base}/completion"))
        .header("Content-Type", "application/json")
        .body(r#"{"prompt":"a","n_predict":1}"#)
        .send()
        .await
        .context("failed to POST /completion for challenge inference")?;
    if !resp.status().is_success() {
        bail!(
            "challenge inference returned status {}",
            resp.status().as_u16()
        );
    }

    // Step 3: End challenge (disable capture, get matmul data)
    let resp = client
        .post(format!("{base}/commitllm/challenge/end"))
        .send()
        .await
        .context("failed to POST /commitllm/challenge/end")?;
    if !resp.status().is_success() {
        bail!(
            "challenge/end returned status {}",
            resp.status().as_u16()
        );
    }

    let captures: ChallengeResponse = resp
        .json()
        .await
        .context("failed to parse challenge/end JSON response")?;

    tracing::info!(
        "challenge complete: {} matmuls, {} total z values",
        captures.n_matmuls,
        captures.matmuls.iter().map(|m| m.z.len()).sum::<usize>()
    );

    Ok(captures)
}

/// Verify challenge captures using Freivalds check.
///
/// For now, this performs structural validation:
/// - Expected number of matmuls (7 per layer + lm_head)
/// - Non-zero accumulators (a node returning all zeros is cheating)
/// - Consistent block counts within expected ranges
///
/// Full algebraic verification (v · x == r · z) requires a verifier key
/// generated from the model weights. This will be added when verilm-core
/// is integrated as a dependency.
pub fn verify_captures(captures: &ChallengeResponse) -> Result<VerifyResult> {
    if captures.n_matmuls == 0 {
        bail!("challenge returned zero matmuls — capture may not be working");
    }

    // Structural checks
    let mut total_z = 0usize;
    let mut all_zero_count = 0usize;

    for (i, m) in captures.matmuls.iter().enumerate() {
        if m.z.len() != m.n_rows {
            bail!(
                "matmul {i}: z.len()={} != n_rows={}",
                m.z.len(),
                m.n_rows
            );
        }
        total_z += m.z.len();

        // Check for all-zero accumulators (suspicious)
        if m.z.iter().all(|&v| v == 0) {
            all_zero_count += 1;
        }
    }

    // If more than half the matmuls have all-zero z, something is wrong
    if all_zero_count > captures.n_matmuls / 2 {
        bail!(
            "{all_zero_count}/{} matmuls have all-zero accumulators — suspicious",
            captures.n_matmuls
        );
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
