use std::collections::HashMap;

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

/// Client for the HuggingFace Jobs REST API.
///
/// The Rust `hf_hub` crate has no Jobs API support, so we call the 5 simple
/// endpoints directly with `reqwest`.
pub struct HfJobsClient {
    http: reqwest::Client,
    endpoint: String,
    token: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct JobSpec {
    #[serde(rename = "dockerImage")]
    pub docker_image: String,
    pub command: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub arguments: Vec<String>,
    pub environment: HashMap<String, String>,
    /// Secret environment variables to inject into the job container.
    ///
    /// The HF Jobs REST API accepts the secret values here and stores them as
    /// job secrets server-side; callers must redact this field before printing.
    pub secrets: HashMap<String, String>,
    pub flavor: String,
    #[serde(rename = "timeoutSeconds")]
    pub timeout_seconds: u64,
    pub volumes: Vec<JobVolume>,
}

#[derive(Debug, Clone, Serialize)]
pub struct JobVolume {
    #[serde(rename = "type")]
    pub volume_type: String,
    pub source: String,
    #[serde(rename = "mountPath")]
    pub mount_path: String,
    #[serde(rename = "readOnly", skip_serializing_if = "Option::is_none")]
    pub read_only: Option<bool>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct HardwareFlavor {
    pub name: String,
    #[serde(default)]
    pub pretty_name: Option<String>,
    #[serde(default)]
    pub cpu: Option<String>,
    #[serde(default)]
    pub ram: Option<String>,
    #[serde(default)]
    pub accelerator: Option<serde_json::Value>,
    #[serde(default, rename = "unitCostUSD")]
    pub unit_cost_usd: Option<f64>,
    #[serde(default, rename = "unitCostMicroUSD")]
    pub unit_cost_micro_usd: Option<u64>,
    #[serde(default)]
    pub unit_label: Option<String>,
}

impl HardwareFlavor {
    pub fn pretty_name(&self) -> &str {
        self.pretty_name.as_deref().unwrap_or(&self.name)
    }

    pub fn unit_label(&self) -> &str {
        self.unit_label.as_deref().unwrap_or("minute")
    }

    pub fn ram_bytes(&self) -> Option<u64> {
        self.ram.as_deref().and_then(parse_size_bytes)
    }

    pub fn resolved_unit_cost_usd(&self) -> Result<f64> {
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

#[derive(Debug, Clone, Serialize)]
pub struct CpuJobPlan {
    pub flavor: String,
    pub pretty_name: String,
    pub cpu: Option<String>,
    pub ram: Option<String>,
    pub unit_cost_usd: f64,
    pub unit_label: String,
    pub max_cost_usd: f64,
    pub timeout_seconds: u64,
    pub minimum_timeout_seconds: u64,
    pub requested_timeout_seconds: u64,
    pub timeout_bumped_to_minimum: bool,
    pub auto_selected_hardware: bool,
    pub selection_reason: String,
    pub model_size_bytes: u64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct JobInfo {
    pub id: String,
    pub status: JobStatus,
    pub created_at: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct JobStatus {
    pub stage: JobStage,
    pub message: Option<String>,
    #[serde(default)]
    pub failure_count: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum JobStage {
    Pending,
    Running,
    Completed,
    Error,
    Canceled,
    Deleted,
    /// Catch-all for stages added after this code was written.
    #[serde(other)]
    Unknown,
}

impl std::fmt::Display for JobStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            JobStage::Pending => write!(f, "PENDING"),
            JobStage::Running => write!(f, "RUNNING"),
            JobStage::Completed => write!(f, "COMPLETED"),
            JobStage::Error => write!(f, "ERROR"),
            JobStage::Canceled => write!(f, "CANCELED"),
            JobStage::Deleted => write!(f, "DELETED"),
            JobStage::Unknown => write!(f, "UNKNOWN"),
        }
    }
}

impl JobStage {
    pub fn is_success(self) -> bool {
        matches!(self, JobStage::Completed)
    }

    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            JobStage::Completed | JobStage::Error | JobStage::Canceled | JobStage::Deleted
        )
    }
}

/// A single log line from the SSE stream.
#[derive(Debug, Deserialize)]
pub struct LogEntry {
    pub data: String,
    pub timestamp: Option<String>,
}

impl HfJobsClient {
    /// Build a client from environment.
    ///
    /// Requires `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN` to be set.
    pub fn from_env() -> Result<Self> {
        let token = model_hf::hf_token_override()
            .context("HF_TOKEN not set. Export a Hugging Face token with write access.")?;

        let endpoint =
            std::env::var("HF_ENDPOINT").unwrap_or_else(|_| "https://huggingface.co".to_string());

        Ok(Self {
            http: reqwest::Client::new(),
            endpoint,
            token,
        })
    }

    /// Submit a new job.
    pub async fn submit(&self, namespace: &str, spec: &JobSpec) -> Result<JobInfo> {
        let url = format!("{}/api/jobs/{}", self.endpoint, namespace);
        let resp = self
            .http
            .post(&url)
            .bearer_auth(&self.token)
            .json(spec)
            .send()
            .await
            .context("submit HF job")?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("HF Jobs API returned {status}: {body}");
        }
        resp.json().await.context("parse job submit response")
    }

    /// Inspect a job's current status.
    pub async fn inspect(&self, namespace: &str, job_id: &str) -> Result<JobInfo> {
        let url = format!("{}/api/jobs/{}/{}", self.endpoint, namespace, job_id);
        let resp = self
            .http
            .get(&url)
            .bearer_auth(&self.token)
            .send()
            .await
            .context("inspect HF job")?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("HF Jobs API returned {status}: {body}");
        }
        resp.json().await.context("parse job inspect response")
    }

    /// Stream job logs via SSE. Returns lines as they arrive.
    ///
    /// Each SSE `data:` line is a JSON object with `data` and `timestamp` fields.
    pub async fn stream_logs(
        &self,
        namespace: &str,
        job_id: &str,
    ) -> Result<impl futures::Stream<Item = Result<String>>> {
        let url = format!("{}/api/jobs/{}/{}/logs", self.endpoint, namespace, job_id);
        let resp = self
            .http
            .get(&url)
            .bearer_auth(&self.token)
            .send()
            .await
            .context("fetch HF job logs")?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("HF Jobs API returned {status}: {body}");
        }

        // Buffer partial lines across chunk boundaries so SSE `data:` lines
        // that span two HTTP chunks are reconstructed before parsing.
        use futures::StreamExt;
        let stream = resp.bytes_stream();
        let mut partial = String::new();
        let lines = stream.flat_map(
            move |chunk: std::result::Result<bytes::Bytes, reqwest::Error>| {
                let lines: Vec<Result<String>> = match chunk {
                    Ok(bytes) => {
                        partial.push_str(&String::from_utf8_lossy(&bytes));
                        let mut results = Vec::new();

                        // Process all complete lines; keep the last fragment.
                        while let Some(newline_pos) = partial.find('\n') {
                            let line = partial[..newline_pos].trim().to_string();
                            partial = partial[newline_pos + 1..].to_string();

                            if let Some(json_str) = line.strip_prefix("data: ") {
                                match serde_json::from_str::<LogEntry>(json_str) {
                                    Ok(entry) => results.push(Ok(entry.data)),
                                    Err(_) => results.push(Ok(json_str.to_string())),
                                }
                            }
                        }
                        results
                    }
                    Err(e) => vec![Err(anyhow::anyhow!("log stream error: {e}"))],
                };
                futures::stream::iter(lines)
            },
        );

        Ok(lines)
    }

    /// Cancel a running job.
    pub async fn cancel(&self, namespace: &str, job_id: &str) -> Result<()> {
        let url = format!("{}/api/jobs/{}/{}/cancel", self.endpoint, namespace, job_id);
        let resp = self
            .http
            .post(&url)
            .bearer_auth(&self.token)
            .send()
            .await
            .context("cancel HF job")?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("HF Jobs API returned {status}: {body}");
        }
        Ok(())
    }

    /// List recent jobs in a namespace.
    pub async fn list(&self, namespace: &str) -> Result<Vec<JobInfo>> {
        let url = format!("{}/api/jobs/{}", self.endpoint, namespace);
        let resp = self
            .http
            .get(&url)
            .bearer_auth(&self.token)
            .send()
            .await
            .context("list HF jobs")?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("HF Jobs API returned {status}: {body}");
        }
        resp.json().await.context("parse job list response")
    }

    /// The token this client was constructed with.
    pub fn token(&self) -> &str {
        &self.token
    }

    /// The HF endpoint this client targets.
    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }
}

pub fn hf_endpoint() -> String {
    std::env::var("HF_ENDPOINT")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| "https://huggingface.co".to_string())
}

pub async fn fetch_hardware(endpoint: &str) -> Result<Vec<HardwareFlavor>> {
    let url = format!("{}/api/jobs/hardware", endpoint.trim_end_matches('/'));
    let resp = reqwest::Client::new()
        .get(&url)
        .send()
        .await
        .with_context(|| format!("GET {url}"))?;
    let status = resp.status();
    if !status.is_success() {
        let body = resp.text().await.unwrap_or_default();
        bail!("Failed to resolve Hugging Face Jobs pricing: {status}: {body}");
    }
    resp.json()
        .await
        .context("decode Hugging Face Jobs hardware pricing response")
}

pub async fn plan_cpu_job(
    endpoint: &str,
    requested_flavor: &str,
    requested_timeout_seconds: u64,
    model_size_bytes: u64,
) -> Result<CpuJobPlan> {
    let hardware = fetch_hardware(endpoint).await?;
    plan_cpu_job_from_hardware(
        &hardware,
        requested_flavor,
        requested_timeout_seconds,
        model_size_bytes,
    )
}

pub fn plan_cpu_job_from_hardware(
    hardware: &[HardwareFlavor],
    requested_flavor: &str,
    requested_timeout_seconds: u64,
    model_size_bytes: u64,
) -> Result<CpuJobPlan> {
    let auto_selected_hardware = requested_flavor == "auto";
    let minimum_timeout_seconds = recommended_cpu_timeout_seconds(model_size_bytes);
    let timeout_seconds = requested_timeout_seconds.max(minimum_timeout_seconds);
    let timeout_bumped_to_minimum = timeout_seconds != requested_timeout_seconds;

    let cpu_flavors = hardware
        .iter()
        .filter(|flavor| flavor.accelerator.is_none())
        .cloned()
        .collect::<Vec<_>>();
    if cpu_flavors.is_empty() {
        bail!("Hugging Face Jobs hardware list did not include any CPU flavors");
    }

    let (flavor, selection_reason) = if auto_selected_hardware {
        select_cpu_flavor(&cpu_flavors)?
    } else {
        let flavor = cpu_flavors
            .into_iter()
            .find(|candidate| candidate.name == requested_flavor)
            .ok_or_else(|| {
                anyhow::anyhow!("Unknown CPU Hugging Face Jobs flavor: {requested_flavor}")
            })?;
        (
            flavor,
            "requested explicitly; CPU hardware only needs to run the splitter/build, not hold the whole model in RAM"
                .to_string(),
        )
    };

    let unit_cost_usd = flavor.resolved_unit_cost_usd()?;
    let unit_label = flavor.unit_label().to_string();
    let max_cost_usd = estimate_cost_usd(unit_cost_usd, &unit_label, timeout_seconds)?;
    Ok(CpuJobPlan {
        flavor: flavor.name.clone(),
        pretty_name: flavor.pretty_name().to_string(),
        cpu: flavor.cpu.clone(),
        ram: flavor.ram.clone(),
        unit_cost_usd,
        unit_label,
        max_cost_usd,
        timeout_seconds,
        minimum_timeout_seconds,
        requested_timeout_seconds,
        timeout_bumped_to_minimum,
        auto_selected_hardware,
        selection_reason,
        model_size_bytes,
    })
}

fn select_cpu_flavor(flavors: &[HardwareFlavor]) -> Result<(HardwareFlavor, String)> {
    if let Some(flavor) = flavors.iter().find(|flavor| flavor.name == "cpu-upgrade") {
        return Ok((
            flavor.clone(),
            "auto-selected cpu-upgrade as the splitter-friendly baseline: 8 vCPU / 32 GB, CPU and I/O bound"
                .to_string(),
        ));
    }

    let mut baseline = flavors
        .iter()
        .filter(|flavor| {
            parse_cpu_count(flavor.cpu.as_deref()).unwrap_or_default() >= 8
                && flavor.ram_bytes().unwrap_or_default() >= 32 * 1024 * 1024 * 1024
        })
        .cloned()
        .collect::<Vec<_>>();
    baseline.sort_by(|a, b| {
        a.resolved_unit_cost_usd()
            .unwrap_or(f64::INFINITY)
            .partial_cmp(&b.resolved_unit_cost_usd().unwrap_or(f64::INFINITY))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    if let Some(flavor) = baseline.into_iter().next() {
        return Ok((
            flavor,
            "auto-selected cheapest CPU flavor meeting the splitter baseline: at least 8 vCPU and 32 GB RAM"
                .to_string(),
        ));
    }

    let fallback = flavors
        .iter()
        .min_by(|a, b| {
            a.resolved_unit_cost_usd()
                .unwrap_or(f64::INFINITY)
                .partial_cmp(&b.resolved_unit_cost_usd().unwrap_or(f64::INFINITY))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .cloned()
        .context("choose fallback CPU Hugging Face Jobs flavor")?;
    Ok((
        fallback,
        "auto-selected cheapest CPU flavor because no advertised CPU flavor met the splitter baseline"
            .to_string(),
    ))
}

fn recommended_cpu_timeout_seconds(model_size_bytes: u64) -> u64 {
    let gib = model_size_bytes as f64 / 1024_f64.powi(3);
    if gib <= 8.0 {
        2 * 60 * 60
    } else if gib <= 32.0 {
        4 * 60 * 60
    } else if gib <= 128.0 {
        8 * 60 * 60
    } else {
        12 * 60 * 60
    }
}

pub fn estimate_cost_usd(
    unit_cost_usd: f64,
    unit_label: &str,
    timeout_seconds: u64,
) -> Result<f64> {
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

fn parse_size_bytes(input: &str) -> Option<u64> {
    let mut parts = input.split_whitespace();
    let value = parts.next()?.parse::<f64>().ok()?;
    let unit = parts.next().unwrap_or("B").to_ascii_lowercase();
    let multiplier = match unit.as_str() {
        "b" | "byte" | "bytes" => 1.0,
        "kb" => 1024.0,
        "mb" => 1024.0 * 1024.0,
        "gb" => 1024.0 * 1024.0 * 1024.0,
        "tb" => 1024.0 * 1024.0 * 1024.0 * 1024.0,
        _ => return None,
    };
    Some((value * multiplier).round() as u64)
}

fn parse_cpu_count(input: Option<&str>) -> Option<u64> {
    input?.split_whitespace().next()?.parse::<u64>().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cpu(name: &str, cpu_count: &str, ram: &str, cost: f64) -> HardwareFlavor {
        HardwareFlavor {
            name: name.to_string(),
            pretty_name: Some(name.to_string()),
            cpu: Some(cpu_count.to_string()),
            ram: Some(ram.to_string()),
            accelerator: None,
            unit_cost_usd: Some(cost),
            unit_cost_micro_usd: None,
            unit_label: Some("minute".to_string()),
        }
    }

    #[test]
    fn estimates_minute_pricing() {
        let cost = estimate_cost_usd(2.0, "minute", 1800).unwrap();
        assert!((cost - 60.0).abs() < f64::EPSILON);
    }

    #[test]
    fn cpu_plan_prefers_upgrade_baseline() {
        let hardware = vec![
            cpu("cpu-basic", "2 vCPU", "16 GB", 0.01),
            cpu("cpu-upgrade", "8 vCPU", "32 GB", 0.05),
            cpu("cpu-expensive", "32 vCPU", "256 GB", 0.10),
        ];
        let plan =
            plan_cpu_job_from_hardware(&hardware, "auto", 60, 200 * 1024 * 1024 * 1024).unwrap();
        assert_eq!(plan.flavor, "cpu-upgrade");
        assert!(plan.auto_selected_hardware);
        assert!(plan.timeout_bumped_to_minimum);
    }

    #[test]
    fn cpu_plan_selects_cheapest_splitter_baseline_without_upgrade() {
        let hardware = vec![
            cpu("cpu-small", "2 vCPU", "16 GB", 0.01),
            cpu("cpu-mid", "16 vCPU", "64 GB", 0.10),
            cpu("cpu-large", "32 vCPU", "256 GB", 0.20),
        ];
        let plan =
            plan_cpu_job_from_hardware(&hardware, "auto", 60, 100 * 1024 * 1024 * 1024).unwrap();
        assert_eq!(plan.flavor, "cpu-mid");
    }
}
