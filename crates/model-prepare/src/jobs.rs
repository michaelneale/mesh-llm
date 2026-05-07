use std::collections::HashMap;

use anyhow::{Context, Result};
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

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct JobInfo {
    pub id: String,
    pub status: JobStatus,
    pub created_at: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct JobStatus {
    pub stage: JobStage,
    pub message: Option<String>,
    #[serde(default)]
    pub failure_count: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum JobStage {
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
            JobStage::Running => write!(f, "RUNNING"),
            JobStage::Completed => write!(f, "COMPLETED"),
            JobStage::Error => write!(f, "ERROR"),
            JobStage::Canceled => write!(f, "CANCELED"),
            JobStage::Deleted => write!(f, "DELETED"),
            JobStage::Unknown => write!(f, "UNKNOWN"),
        }
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
    /// Requires `HF_TOKEN` to be set.
    pub fn from_env() -> Result<Self> {
        let token = std::env::var("HF_TOKEN")
            .ok()
            .filter(|t| !t.is_empty())
            .context("HF_TOKEN not set. Export a HuggingFace token with write access.")?;

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
