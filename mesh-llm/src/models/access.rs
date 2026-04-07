use anyhow::{Context, Result};
use hf_hub::api::tokio::Api as TokioApi;
use hf_hub::{Repo, RepoType};
use reqwest::StatusCode;
use serde_json::Value;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RepoAccess {
    Open,
    Gated,
}

pub fn repo_url(repo: &str) -> String {
    format!("https://huggingface.co/{repo}")
}

pub fn gated_access_message(repo: &str) -> String {
    format!(
        "🟡 This Hugging Face repo is gated and cannot be downloaded until terms are accepted at {}",
        repo_url(repo)
    )
}

pub fn reqwest_error_indicates_gated(err: &reqwest::Error) -> bool {
    if err.is_decode() {
        let text = err.to_string().to_ascii_lowercase();
        if text.contains("manual") && text.contains("expected a boolean") {
            return true;
        }
    }

    if let Some(status) = err.status() {
        if status == StatusCode::FORBIDDEN || status == StatusCode::UNAUTHORIZED {
            let text = err.to_string().to_ascii_lowercase();
            if text.contains("gated") {
                return true;
            }
        }
    }

    false
}

pub async fn probe_repo_access(
    api: &TokioApi,
    repo: &str,
    revision: Option<&str>,
) -> Result<RepoAccess> {
    let revision = revision.unwrap_or("main").to_string();
    let repo_handle = Repo::with_revision(repo.to_string(), RepoType::Model, revision);
    let response = api
        .repo(repo_handle)
        .info_request()
        .send()
        .await
        .with_context(|| format!("Probe Hugging Face repo metadata {repo}"))?;

    let headers = response.headers().clone();
    let status = response.status();
    let body = response.text().await.unwrap_or_default();

    if headers
        .get("x-error-code")
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value.eq_ignore_ascii_case("GatedRepo"))
    {
        return Ok(RepoAccess::Gated);
    }

    if let Ok(json) = serde_json::from_str::<Value>(&body) {
        if repo_info_json_is_gated(&json) {
            return Ok(RepoAccess::Gated);
        }
    }

    if status == StatusCode::FORBIDDEN || status == StatusCode::UNAUTHORIZED {
        let lowered = body.to_ascii_lowercase();
        if lowered.contains("gated") || lowered.contains("accept") {
            return Ok(RepoAccess::Gated);
        }
    }

    Ok(RepoAccess::Open)
}

fn repo_info_json_is_gated(json: &Value) -> bool {
    match json.get("gated") {
        Some(Value::Bool(value)) => *value,
        Some(Value::String(value)) => {
            let normalized = value.trim().to_ascii_lowercase();
            !normalized.is_empty() && normalized != "false" && normalized != "0"
        }
        _ => false,
    }
}
