use anyhow::{bail, Context, Result};
use reqwest::StatusCode;
use serde::Deserialize;
use serde_json::json;
use std::path::Path;

use crate::cli::moe::{HfJobArgs, HfJobReleaseTarget};
use crate::inference::launch::BinaryFlavor;
use crate::models;
use crate::system::release_target::{CanonicalArch, CanonicalOs, ReleaseTarget};

const DEFAULT_HF_ENDPOINT: &str = "https://huggingface.co";
const CPU_JOB_IMAGE: &str = "ghcr.io/astral-sh/uv:python3.12-bookworm";
const CUDA_JOB_IMAGE: &str = "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel";
const ROCM_JOB_IMAGE: &str = "rocm/pytorch:rocm6.3_ubuntu24.04_py3.12_pytorch_release_2.4.0";
const VULKAN_JOB_IMAGE: &str = "ghcr.io/astral-sh/uv:python3.12-bookworm";
const HF_JOB_WRAPPER_TEMPLATE: &str = include_str!("hf_job_wrapper.sh");

pub(crate) async fn submit_publish_job(
    model_spec: &str,
    catalog_repo: &str,
    package_namespace: Option<&str>,
    options: &HfJobArgs,
) -> Result<()> {
    let identity = remote_identity(model_spec).await?;
    let spec = JobSubmissionSpec {
        model_ref: identity.distribution_ref(),
        catalog_repo: catalog_repo.to_string(),
        package_namespace: package_namespace.map(ToOwned::to_owned),
        flavor: options.hf_job_flavor.clone(),
        timeout: options.hf_job_timeout.clone(),
        job_namespace: options.hf_job_namespace.clone(),
        release_repo: options.hf_job_release_repo.clone(),
        release_tag: options.hf_job_release_tag.clone(),
        release_target: options.hf_job_release_target,
        source_repo: identity.repo_id.clone(),
        source_revision: identity.revision.clone(),
        source_file: identity.file.clone(),
        distribution_id: crate::system::moe_planner::normalize_distribution_id(
            &identity.local_file_name,
        ),
    };
    submit_job(spec).await
}

#[derive(Clone)]
struct JobSubmissionSpec {
    model_ref: String,
    catalog_repo: String,
    package_namespace: Option<String>,
    flavor: String,
    timeout: String,
    job_namespace: Option<String>,
    release_repo: String,
    release_tag: String,
    release_target: HfJobReleaseTarget,
    source_repo: String,
    source_revision: String,
    source_file: String,
    distribution_id: String,
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
    #[serde(default)]
    pretty_name: Option<String>,
    #[serde(default, rename = "unitCostUSD")]
    unit_cost_usd: Option<f64>,
    #[serde(default, rename = "unitCostMicroUSD")]
    unit_cost_micro_usd: Option<u64>,
    #[serde(default)]
    unit_label: Option<String>,
}

impl HardwareFlavor {
    fn pretty_name(&self) -> &str {
        self.pretty_name.as_deref().unwrap_or(&self.name)
    }

    fn unit_label(&self) -> &str {
        self.unit_label.as_deref().unwrap_or("minute")
    }

    fn resolved_unit_cost_usd(&self) -> Result<f64> {
        if let Some(unit_cost_usd) = self.unit_cost_usd {
            return Ok(unit_cost_usd);
        }
        if let Some(unit_cost_micro_usd) = self.unit_cost_micro_usd {
            return Ok(unit_cost_micro_usd as f64 / 1_000_000.0);
        }
        bail!(
            "Hugging Face hardware flavor {} is missing both unitCostUSD and unitCostMicroUSD",
            self.name
        );
    }
}

#[derive(Clone, Debug)]
struct PricingEstimate {
    flavor: HardwareFlavor,
    unit_cost_usd: f64,
    max_cost_usd: f64,
}

async fn submit_job(spec: JobSubmissionSpec) -> Result<()> {
    let token = models::hf_token_override().ok_or_else(|| {
        anyhow::anyhow!(
            "Missing Hugging Face token. Set HF_TOKEN or HUGGING_FACE_HUB_TOKEN before using --hf-job."
        )
    })?;
    let endpoint = hf_endpoint();
    let namespace = match spec.job_namespace.clone() {
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
            "SOURCE_REPO": spec.source_repo,
            "SOURCE_REVISION": spec.source_revision,
            "SOURCE_FILE": spec.source_file,
            "CATALOG_REPO": spec.catalog_repo,
            "HF_JOB_FLAVOR": pricing.flavor.name,
            "HF_JOB_FLAVOR_PRETTY": pricing.flavor.pretty_name(),
            "HF_JOB_UNIT_COST_USD": format!("{:.6}", pricing.unit_cost_usd),
            "HF_JOB_UNIT_LABEL": pricing.flavor.unit_label(),
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
            "workflow": "moe-publish",
            "source_repo": sanitize_label(&spec.source_repo),
            "source_revision": sanitize_label(&spec.source_revision),
            "distribution_id": sanitize_label(&spec.distribution_id),
            "catalog_repo": sanitize_label(&spec.catalog_repo),
        }
    });

    println!("☁️ Hugging Face Job submission");
    println!("📦 Model: {}", spec.model_ref);
    println!("📤 Workflow: moe publish");
    println!("🗂️ Catalog: {}", spec.catalog_repo);
    if let Some(package_namespace) = &spec.package_namespace {
        println!("📦 Package namespace: {}", package_namespace);
    }
    println!("🖥️ Flavor: {}", spec.flavor);
    println!("⏱️ Timeout: {}", spec.timeout);
    println!("📥 Release: {} {}", spec.release_repo, spec.release_tag);
    println!(
        "💵 Pricing: {} @ ${:.6}/{}",
        pricing.flavor.pretty_name(),
        pricing.unit_cost_usd,
        pricing.flavor.unit_label()
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
    let unit_cost_usd = flavor.resolved_unit_cost_usd()?;
    let max_cost_usd = estimate_cost_usd(unit_cost_usd, flavor.unit_label(), timeout_seconds)?;
    Ok(PricingEstimate {
        flavor,
        unit_cost_usd,
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
    let mut publish_command =
        "./mesh-llm moe publish \"$MODEL_REF\" --catalog-repo \"$CATALOG_REPO\"".to_string();
    if let Some(namespace) = &spec.package_namespace {
        publish_command.push_str(" --namespace ");
        publish_command.push_str(&shell_single_quote(namespace));
    }
    HF_JOB_WRAPPER_TEMPLATE.replace("__PUBLISH_COMMAND__", &shell_single_quote(&publish_command))
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

fn hf_job_binary_flavor(release_target: HfJobReleaseTarget) -> BinaryFlavor {
    match release_target {
        HfJobReleaseTarget::Cpu => BinaryFlavor::Cpu,
        HfJobReleaseTarget::Cuda => BinaryFlavor::Cuda,
        HfJobReleaseTarget::Rocm => BinaryFlavor::Rocm,
        HfJobReleaseTarget::Vulkan => BinaryFlavor::Vulkan,
    }
}

fn linux_x86_64_release_target(flavor: BinaryFlavor) -> ReleaseTarget {
    ReleaseTarget::new(CanonicalOs::Linux, CanonicalArch::X86_64, flavor)
}

fn release_asset_name(release_tag: &str, release_target: HfJobReleaseTarget) -> String {
    let target = linux_x86_64_release_target(hf_job_binary_flavor(release_target));
    if release_tag == "latest" {
        target
            .stable_asset_name()
            .expect("linux x86_64 HF job targets must have a stable release asset")
    } else {
        target
            .versioned_asset_name(release_tag)
            .expect("linux x86_64 HF job targets must have a versioned release asset")
    }
}

#[cfg(test)]
pub(crate) fn release_target_versioned_linux_asset_name(
    release_tag: &str,
    flavor: BinaryFlavor,
) -> Option<String> {
    linux_x86_64_release_target(flavor).versioned_asset_name(release_tag)
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
    fn estimate_cost_supports_minute_pricing() {
        let cost = estimate_cost_usd(2.0, "minute", 1800).unwrap();
        assert!((cost - 60.0).abs() < f64::EPSILON);
    }

    #[test]
    fn estimate_cost_supports_hour_pricing() {
        let cost = estimate_cost_usd(12.0, "hour", 1800).unwrap();
        assert!((cost - 6.0).abs() < f64::EPSILON);
    }

    #[test]
    fn sanitize_label_replaces_unsupported_characters() {
        assert_eq!(sanitize_label("meshllm/catalog"), "meshllm-catalog");
        assert_eq!(sanitize_label("  weird value  "), "weird-value");
    }

    #[test]
    fn shell_single_quote_escapes_embedded_quotes() {
        assert_eq!(shell_single_quote("a'b"), "a'\"'\"'b");
    }

    #[test]
    fn job_images_match_runtime_targets() {
        assert_eq!(job_image(HfJobReleaseTarget::Cpu), CPU_JOB_IMAGE);
        assert_eq!(job_image(HfJobReleaseTarget::Cuda), CUDA_JOB_IMAGE);
        assert_eq!(job_image(HfJobReleaseTarget::Rocm), ROCM_JOB_IMAGE);
        assert_eq!(job_image(HfJobReleaseTarget::Vulkan), VULKAN_JOB_IMAGE);
    }

    #[test]
    fn release_assets_match_expected_linux_names() {
        let versioned = release_target_versioned_linux_asset_name("v0.1.2", BinaryFlavor::Cpu)
            .expect("linux x86_64 cpu release asset");
        assert_eq!(
            release_asset_name("v0.1.2", HfJobReleaseTarget::Cpu),
            versioned
        );
    }

    #[test]
    fn release_target_hf_jobs_parity() {
        for target in [
            HfJobReleaseTarget::Cpu,
            HfJobReleaseTarget::Cuda,
            HfJobReleaseTarget::Rocm,
            HfJobReleaseTarget::Vulkan,
        ] {
            let _ = release_asset_name("latest", target);
        }
    }

    #[test]
    fn remote_publish_script_includes_namespace_when_requested() {
        let spec = JobSubmissionSpec {
            model_ref: "mesh/model:Q4".to_string(),
            catalog_repo: "meshllm/catalog".to_string(),
            package_namespace: Some("meshllm".to_string()),
            flavor: "cpu-xl".to_string(),
            timeout: "1h".to_string(),
            job_namespace: None,
            release_repo: "Mesh-LLM/mesh-llm".to_string(),
            release_tag: "latest".to_string(),
            release_target: HfJobReleaseTarget::Cpu,
            source_repo: "unsloth/demo".to_string(),
            source_revision: "deadbeef".to_string(),
            source_file: "model.gguf".to_string(),
            distribution_id: "Demo-Q4".to_string(),
        };
        let script = remote_bash_script(&spec);
        assert!(script.contains("__PUBLISH_COMMAND__") == false);
        assert!(script.contains("moe publish"));
        assert!(script.contains("--namespace"));
    }
}
