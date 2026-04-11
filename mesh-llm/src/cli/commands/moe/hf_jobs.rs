use anyhow::{bail, Context, Result};
use reqwest::StatusCode;
use serde::Deserialize;
use serde_json::json;
use std::path::Path;

use crate::cli::moe::{HfJobArgs, HfJobReleaseTarget};
use crate::models;

const DEFAULT_HF_ENDPOINT: &str = "https://huggingface.co";
const CPU_JOB_IMAGE: &str = "ghcr.io/astral-sh/uv:python3.12-bookworm";
const CUDA_JOB_IMAGE: &str = "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel";
const ROCM_JOB_IMAGE: &str = "rocm/pytorch:rocm6.3_ubuntu24.04_py3.12_pytorch_release_2.4.0";
const VULKAN_JOB_IMAGE: &str = "ghcr.io/astral-sh/uv:python3.12-bookworm";
const HF_JOB_WRAPPER_TEMPLATE: &str = include_str!("hf_job_wrapper.sh");

pub(crate) async fn submit_full_analyze_job(
    model_spec: &str,
    context_size: u32,
    n_gpu_layers: u32,
    options: &HfJobArgs,
) -> Result<()> {
    let identity = remote_identity(model_spec).await?;
    let spec = JobSubmissionSpec {
        model_ref: identity.canonical_ref.clone(),
        analyzer: RemoteAnalyzer::Full {
            context_size,
            n_gpu_layers,
        },
        dataset_repo: options.dataset_repo.clone(),
        flavor: options.hf_job_flavor.clone(),
        timeout: options.hf_job_timeout.clone(),
        namespace: options.hf_job_namespace.clone(),
        release_repo: options.hf_job_release_repo.clone(),
        release_tag: options.hf_job_release_tag.clone(),
        release_target: options.hf_job_release_target,
        source_repo: identity.repo_id.clone(),
        source_revision: identity.revision.clone(),
        distribution_id: crate::system::moe_planner::normalize_distribution_id(
            &identity.local_file_name,
        ),
    };
    submit_job(spec).await
}

pub(crate) async fn submit_micro_analyze_job(
    model_spec: &str,
    prompt_count: usize,
    token_count: u32,
    context_size: u32,
    n_gpu_layers: u32,
    options: &HfJobArgs,
) -> Result<()> {
    let identity = remote_identity(model_spec).await?;
    let spec = JobSubmissionSpec {
        model_ref: identity.canonical_ref.clone(),
        analyzer: RemoteAnalyzer::Micro {
            prompt_count,
            token_count,
            context_size,
            n_gpu_layers,
        },
        dataset_repo: options.dataset_repo.clone(),
        flavor: options.hf_job_flavor.clone(),
        timeout: options.hf_job_timeout.clone(),
        namespace: options.hf_job_namespace.clone(),
        release_repo: options.hf_job_release_repo.clone(),
        release_tag: options.hf_job_release_tag.clone(),
        release_target: options.hf_job_release_target,
        source_repo: identity.repo_id.clone(),
        source_revision: identity.revision.clone(),
        distribution_id: crate::system::moe_planner::normalize_distribution_id(
            &identity.local_file_name,
        ),
    };
    submit_job(spec).await
}

#[derive(Clone)]
struct JobSubmissionSpec {
    model_ref: String,
    analyzer: RemoteAnalyzer,
    dataset_repo: String,
    flavor: String,
    timeout: String,
    namespace: Option<String>,
    release_repo: String,
    release_tag: String,
    release_target: HfJobReleaseTarget,
    source_repo: String,
    source_revision: String,
    distribution_id: String,
}

#[derive(Clone)]
enum RemoteAnalyzer {
    Full {
        context_size: u32,
        n_gpu_layers: u32,
    },
    Micro {
        prompt_count: usize,
        token_count: u32,
        context_size: u32,
        n_gpu_layers: u32,
    },
}

impl RemoteAnalyzer {
    fn analyzer_id(&self) -> &'static str {
        match self {
            Self::Full { .. } => "full-v1",
            Self::Micro { .. } => "micro-v1",
        }
    }

    fn analyze_command_with_env_ref(&self) -> String {
        match self {
            Self::Full {
                context_size,
                n_gpu_layers,
            } => format!(
                "./mesh-llm moe analyze full \"$MODEL_REF\" --context-size {} --n-gpu-layers {}",
                context_size, n_gpu_layers
            ),
            Self::Micro {
                prompt_count,
                token_count,
                context_size,
                n_gpu_layers,
            } => format!(
                "./mesh-llm moe analyze micro \"$MODEL_REF\" --prompt-count {} --token-count {} --context-size {} --n-gpu-layers {}",
                prompt_count,
                token_count,
                context_size,
                n_gpu_layers
            ),
        }
    }
}

#[derive(Deserialize)]
struct WhoAmIResponse {
    name: String,
}

#[derive(Deserialize)]
struct JobCreateResponse {
    id: String,
    owner: JobOwner,
    status: JobStatus,
}

#[derive(Deserialize)]
struct JobOwner {
    name: String,
}

#[derive(Deserialize)]
struct JobStatus {
    stage: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct HardwareFlavor {
    name: String,
    pretty_name: String,
    unit_cost_usd: f64,
    unit_label: String,
}

#[derive(Clone, Debug)]
struct PricingEstimate {
    flavor: HardwareFlavor,
    max_cost_usd: f64,
}

async fn submit_job(spec: JobSubmissionSpec) -> Result<()> {
    let token = models::hf_token_override().ok_or_else(|| {
        anyhow::anyhow!(
            "Missing Hugging Face token. Set HF_TOKEN or HUGGING_FACE_HUB_TOKEN before using --hf-job."
        )
    })?;
    let endpoint = hf_endpoint();
    let namespace = match spec.namespace.clone() {
        Some(namespace) => namespace,
        None => resolve_namespace(&endpoint, &token).await?,
    };

    let release_url = release_download_url(
        &spec.release_repo,
        &spec.release_tag,
        &release_asset_name(&spec.release_tag, spec.release_target),
    );
    let timeout_seconds = parse_timeout_seconds(&spec.timeout)?;
    let pricing = fetch_pricing_estimate(&endpoint, &spec.flavor, timeout_seconds).await?;
    let script = remote_bash_script(&spec);
    let payload = json!({
        "dockerImage": job_image(spec.release_target),
        "command": ["bash", "-lc", script],
        "arguments": [],
        "environment": {
            "MESH_LLM_RELEASE_URL": release_url,
            "MODEL_REF": spec.model_ref,
            "DATASET_REPO": spec.dataset_repo,
            "HF_JOB_FLAVOR": pricing.flavor.name,
            "HF_JOB_FLAVOR_PRETTY": pricing.flavor.pretty_name,
            "HF_JOB_UNIT_COST_USD": format!("{:.6}", pricing.flavor.unit_cost_usd),
            "HF_JOB_UNIT_LABEL": pricing.flavor.unit_label,
            "HF_JOB_TIMEOUT_SECONDS": timeout_seconds.to_string(),
            "HF_JOB_MAX_COST_USD": format!("{:.2}", pricing.max_cost_usd),
        },
        "secrets": {
            "HF_TOKEN": token,
        },
        "flavor": spec.flavor,
        "timeoutSeconds": timeout_seconds,
        "labels": {
            "app": "mesh-llm",
            "workflow": "moe-analyze-share",
            "analyzer_id": spec.analyzer.analyzer_id(),
            "source_repo": sanitize_label(&spec.source_repo),
            "source_revision": sanitize_label(&spec.source_revision),
            "distribution_id": sanitize_label(&spec.distribution_id),
            "dataset_repo": sanitize_label(&spec.dataset_repo),
        }
    });

    println!("☁️ Hugging Face Job submission");
    println!("📍 Analyzer: {}", spec.analyzer.analyzer_id());
    println!("📦 Model: {}", spec.model_ref);
    println!("🗂️ Dataset: {}", spec.dataset_repo);
    println!("🖥️ Flavor: {}", spec.flavor);
    println!("⏱️ Timeout: {}", spec.timeout);
    println!("📥 Release: {} {}", spec.release_repo, spec.release_tag);
    println!(
        "💵 Pricing: {} @ ${:.6}/{}",
        pricing.flavor.pretty_name, pricing.flavor.unit_cost_usd, pricing.flavor.unit_label
    );
    println!("🧮 Max cost: ${:.2} USD", pricing.max_cost_usd);

    let url = format!("{}/api/jobs/{}", endpoint.trim_end_matches('/'), namespace);
    let response = reqwest::Client::new()
        .post(&url)
        .bearer_auth(&token)
        .json(&payload)
        .send()
        .await
        .with_context(|| format!("POST {}", url))?;
    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        bail!("HF Job submission failed: {}: {}", status, body.trim());
    }
    let job: JobCreateResponse = response
        .json()
        .await
        .context("Decode Hugging Face Jobs response")?;
    println!("✅ Submitted Hugging Face Job");
    println!("🆔 Job: {}", job.id);
    println!("📡 Status: {}", job.status.stage);
    println!(
        "🔗 URL: {}/jobs/{}/{}",
        endpoint.trim_end_matches('/'),
        job.owner.name,
        job.id
    );
    Ok(())
}

async fn fetch_pricing_estimate(
    endpoint: &str,
    flavor: &str,
    timeout_seconds: u64,
) -> Result<PricingEstimate> {
    let url = format!("{}/api/jobs/hardware", endpoint.trim_end_matches('/'));
    let response = reqwest::Client::new()
        .get(&url)
        .send()
        .await
        .with_context(|| format!("GET {}", url))?;
    if response.status() != StatusCode::OK {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        bail!(
            "Failed to resolve Hugging Face Jobs pricing: {}: {}",
            status,
            body.trim()
        );
    }
    let hardware: Vec<HardwareFlavor> = response
        .json()
        .await
        .context("Decode Hugging Face hardware pricing response")?;
    let flavor = hardware
        .into_iter()
        .find(|candidate| candidate.name == flavor)
        .ok_or_else(|| anyhow::anyhow!("Unknown Hugging Face Jobs flavor: {flavor}"))?;
    let max_cost_usd =
        estimate_cost_usd(flavor.unit_cost_usd, &flavor.unit_label, timeout_seconds)?;
    Ok(PricingEstimate {
        flavor,
        max_cost_usd,
    })
}

async fn remote_identity(model_spec: &str) -> Result<models::local::HuggingFaceModelIdentity> {
    let path = Path::new(model_spec);
    if path.exists() {
        return models::huggingface_identity_for_path(path).ok_or_else(|| {
            anyhow::anyhow!(
                "--hf-job requires a Hugging Face-backed local model path or a Hugging Face model ref."
            )
        });
    }
    models::resolve_huggingface_model_identity(model_spec)
        .await?
        .ok_or_else(|| {
            anyhow::anyhow!(
                "--hf-job requires a Hugging Face-backed model ref, selector, URL, or cached HF local path."
            )
        })
}

async fn resolve_namespace(endpoint: &str, token: &str) -> Result<String> {
    let url = format!("{}/api/whoami-v2", endpoint.trim_end_matches('/'));
    let response = reqwest::Client::new()
        .get(&url)
        .bearer_auth(token)
        .send()
        .await
        .with_context(|| format!("GET {}", url))?;
    if response.status() != StatusCode::OK {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        bail!(
            "Failed to resolve Hugging Face namespace: {}: {}",
            status,
            body.trim()
        );
    }
    let whoami: WhoAmIResponse = response.json().await.context("Decode whoami response")?;
    Ok(whoami.name)
}

fn hf_endpoint() -> String {
    std::env::var("HF_ENDPOINT")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| DEFAULT_HF_ENDPOINT.to_string())
}

fn remote_bash_script(spec: &JobSubmissionSpec) -> String {
    let analyze_command = spec.analyzer.analyze_command_with_env_ref();
    let share_command =
        format!("./mesh-llm moe share \"$MODEL_REF\" --dataset-repo \"$DATASET_REPO\"");
    HF_JOB_WRAPPER_TEMPLATE
        .replace("__ANALYZE_COMMAND__", &shell_single_quote(&analyze_command))
        .replace("__SHARE_COMMAND__", &shell_single_quote(&share_command))
}

fn parse_timeout_seconds(input: &str) -> Result<u64> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        bail!("HF job timeout must not be empty");
    }
    let split_at = trimmed
        .find(|ch: char| !ch.is_ascii_digit() && ch != '.')
        .unwrap_or(trimmed.len());
    let (number, suffix) = trimmed.split_at(split_at);
    let value: f64 = number
        .parse()
        .with_context(|| format!("Parse timeout value from {}", input))?;
    let factor = match suffix {
        "" | "s" => 1.0,
        "m" => 60.0,
        "h" => 3600.0,
        "d" => 86400.0,
        _ => bail!("Unsupported timeout unit in {}. Use s, m, h, or d.", input),
    };
    Ok((value * factor).round() as u64)
}

fn estimate_cost_usd(unit_cost_usd: f64, unit_label: &str, timeout_seconds: u64) -> Result<f64> {
    let timeout_seconds = timeout_seconds as f64;
    let max_cost = match unit_label {
        "second" => unit_cost_usd * timeout_seconds,
        "minute" => unit_cost_usd * (timeout_seconds / 60.0),
        "hour" => unit_cost_usd * (timeout_seconds / 3600.0),
        "day" => unit_cost_usd * (timeout_seconds / 86_400.0),
        other => bail!("Unsupported Hugging Face pricing unit: {other}"),
    };
    Ok(max_cost)
}

fn release_asset_name(release_tag: &str, release_target: HfJobReleaseTarget) -> String {
    let target_suffix = match release_target {
        HfJobReleaseTarget::Cpu => "x86_64-unknown-linux-gnu".to_string(),
        HfJobReleaseTarget::Cuda => "x86_64-unknown-linux-gnu-cuda".to_string(),
        HfJobReleaseTarget::Rocm => "x86_64-unknown-linux-gnu-rocm".to_string(),
        HfJobReleaseTarget::Vulkan => "x86_64-unknown-linux-gnu-vulkan".to_string(),
    };
    if release_tag == "latest" {
        format!("mesh-llm-{target_suffix}.tar.gz")
    } else {
        format!("mesh-llm-{release_tag}-{target_suffix}.tar.gz")
    }
}

fn release_download_url(release_repo: &str, release_tag: &str, asset_name: &str) -> String {
    if release_tag == "latest" {
        format!("https://github.com/{release_repo}/releases/latest/download/{asset_name}")
    } else {
        format!("https://github.com/{release_repo}/releases/download/{release_tag}/{asset_name}")
    }
}

fn job_image(release_target: HfJobReleaseTarget) -> &'static str {
    match release_target {
        HfJobReleaseTarget::Cpu => CPU_JOB_IMAGE,
        HfJobReleaseTarget::Cuda => CUDA_JOB_IMAGE,
        HfJobReleaseTarget::Rocm => ROCM_JOB_IMAGE,
        HfJobReleaseTarget::Vulkan => VULKAN_JOB_IMAGE,
    }
}

fn sanitize_label(value: &str) -> String {
    let sanitized = value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '_' | '=' | '-') {
                ch
            } else {
                '-'
            }
        })
        .collect::<String>()
        .trim_matches('-')
        .to_string();
    if sanitized.is_empty() {
        "unknown".to_string()
    } else {
        sanitized
    }
}

fn shell_single_quote(value: &str) -> String {
    value.replace('\'', "'\"'\"'")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_timeout_accepts_supported_units() {
        assert_eq!(parse_timeout_seconds("90").unwrap(), 90);
        assert_eq!(parse_timeout_seconds("1.5h").unwrap(), 5400);
        assert_eq!(parse_timeout_seconds("30m").unwrap(), 1800);
    }

    #[test]
    fn release_assets_match_expected_linux_names() {
        assert_eq!(
            release_asset_name("latest", HfJobReleaseTarget::Cpu),
            "mesh-llm-x86_64-unknown-linux-gnu.tar.gz"
        );
        assert_eq!(
            release_asset_name("v0.1.0", HfJobReleaseTarget::Cuda),
            "mesh-llm-v0.1.0-x86_64-unknown-linux-gnu-cuda.tar.gz"
        );
    }

    #[test]
    fn job_images_match_runtime_targets() {
        assert_eq!(
            job_image(HfJobReleaseTarget::Cpu),
            "ghcr.io/astral-sh/uv:python3.12-bookworm"
        );
        assert_eq!(
            job_image(HfJobReleaseTarget::Cuda),
            "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel"
        );
    }

    #[test]
    fn sanitize_label_replaces_unsupported_characters() {
        assert_eq!(
            sanitize_label("meshllm/moe-rankings"),
            "meshllm-moe-rankings"
        );
        assert_eq!(sanitize_label("Qwen/Qwen3@abc"), "Qwen-Qwen3-abc");
    }

    #[test]
    fn shell_single_quote_escapes_embedded_quotes() {
        assert_eq!(shell_single_quote("a'b"), "a'\"'\"'b");
    }

    #[test]
    fn estimate_cost_supports_minute_pricing() {
        let cost = estimate_cost_usd(0.03, "minute", 7200).unwrap();
        assert!((cost - 3.60).abs() < 1e-9);
    }

    #[test]
    fn estimate_cost_supports_hour_pricing() {
        let cost = estimate_cost_usd(1.80, "hour", 7200).unwrap();
        assert!((cost - 3.60).abs() < 1e-9);
    }
}
