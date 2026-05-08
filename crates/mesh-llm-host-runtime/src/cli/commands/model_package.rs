use anyhow::{bail, Context, Result};
use tokio_stream::StreamExt;

use model_package::jobs::HfJobsClient;
use model_package::permissions;
use model_package::prepare::{self, DiscoveredQuant, PrepareParams};
use model_package::script;
use serde_json::json;

/// All CLI arguments for `model-package`, bundled to avoid too-many-arguments.
pub(crate) struct ModelPrepareArgs<'a> {
    pub source_repo: Option<&'a str>,
    pub quant: Option<&'a str>,
    pub target: Option<&'a str>,
    pub model_id: Option<&'a str>,
    pub flavor: &'a str,
    pub timeout: &'a str,
    pub mesh_llm_ref: &'a str,
    pub dry_run: bool,
    pub confirm: bool,
    pub follow: bool,
    pub json: bool,
    pub status: Option<&'a str>,
    pub logs: Option<&'a str>,
    pub cancel: Option<&'a str>,
    pub list: bool,
    pub update_script: bool,
}

/// Dispatch the model-package command.
pub(crate) async fn dispatch_model_package(args: ModelPrepareArgs<'_>) -> Result<()> {
    let ModelPrepareArgs {
        source_repo,
        quant,
        target,
        model_id,
        flavor,
        timeout,
        mesh_llm_ref,
        dry_run,
        confirm,
        follow,
        json,
        status,
        logs,
        cancel,
        list,
        update_script,
    } = args;
    // ── Management subcommands (no source_repo needed) ───────────────
    if update_script {
        return run_update_script().await;
    }

    if let Some(job_id) = status {
        let jobs_client = HfJobsClient::from_env()?;
        return run_status(&jobs_client, job_id, json).await;
    }
    if let Some(job_id) = logs {
        let jobs_client = HfJobsClient::from_env()?;
        return run_logs(&jobs_client, job_id, json).await;
    }
    if let Some(job_id) = cancel {
        let jobs_client = HfJobsClient::from_env()?;
        return run_cancel(&jobs_client, job_id, json).await;
    }
    if list {
        let jobs_client = HfJobsClient::from_env()?;
        return run_list(&jobs_client, json).await;
    }

    // ── Submit flow (source ref required) ────────────────────────────
    let source_ref = source_repo.context(
        "Source repo is required for job submission.\n\
         Usage: mesh-llm models package <source_repo>:<quant>",
    )?;
    let source_model_ref = model_ref::ModelRef::parse(source_ref)
        .with_context(|| format!("invalid source model ref: {source_ref}"))?;
    let source_repo = source_model_ref.repo.as_str();
    let source_quant = match (source_model_ref.selector.as_deref(), quant) {
        (Some(selector), Some(quant)) if selector != quant => {
            bail!(
                "source ref selector '{selector}' conflicts with --quant '{quant}'. \
                 Use `mesh-llm models package {source_repo}:{selector}`."
            );
        }
        (Some(selector), _) => Some(selector),
        (None, Some(quant)) => Some(quant),
        (None, None) => None,
    };

    // Build HF client for API calls.
    let hf_client = model_package::build_hf_client()?;

    // If no quant specified, list available quants and exit.
    // This path doesn't need HF_TOKEN — works for public repos.
    if source_quant.is_none() {
        return run_list_quants(&hf_client, source_repo, json).await;
    }

    let submitting = confirm && !dry_run;
    let jobs_client = if submitting {
        Some(HfJobsClient::from_env()?)
    } else {
        None
    };

    // Resolve permissions.
    eprintln!("🔑 Checking permissions...");
    let perms = permissions::check_permissions(&hf_client).await?;

    // Parse timeout.
    let timeout_seconds = parse_timeout(timeout)?;

    // Resolve source, target, and build job spec.
    eprintln!("🔍 Resolving source...");
    let params = PrepareParams {
        source_repo: source_repo.to_string(),
        quant: source_quant.map(|s| s.to_string()),
        target: target.map(|s| s.to_string()),
        model_id: model_id.map(|s| s.to_string()),
        flavor: flavor.to_string(),
        timeout_seconds,
        mesh_llm_ref: mesh_llm_ref.to_string(),
        hf_token: jobs_client
            .as_ref()
            .map(|client| client.token().to_string()),
    };

    let job = prepare::resolve(&hf_client, params, &perms).await?;

    // Print resolved info.
    let shard_info = model_ref::split_gguf_shard_info(&job.source_file);
    let shard_str = if let Some(shard) = shard_info {
        format!(" ({} shards)", shard.total)
    } else {
        String::new()
    };

    eprintln!("   Repo:   {}", job.source_repo);
    eprintln!("   File:   {}{}", job.source_file, shard_str);
    eprintln!();
    eprintln!(
        "🔑 Permissions: {} ({})",
        perms.username,
        if perms.is_meshllm_member {
            "meshllm org member"
        } else {
            "not in meshllm org"
        }
    );
    eprintln!("   Target:  {}", job.target_repo);
    eprintln!(
        "   Catalog: meshllm/catalog ({})",
        if job.catalog_create_pr {
            "will open PR"
        } else {
            "direct commit"
        }
    );
    eprintln!();
    eprintln!(
        "📋 Job: {}, timeout {}, mesh-llm@{}",
        job.spec.flavor,
        format_timeout(job.spec.timeout_seconds),
        job.spec
            .environment
            .get("MESH_LLM_REF")
            .map(|s| s.as_str())
            .unwrap_or("main")
    );
    eprintln!(
        "   Hardware: {} {} ({})",
        job.job_plan.pretty_name,
        hardware_label(job.job_plan.cpu.as_deref(), job.job_plan.ram.as_deref()),
        job.job_plan.selection_reason
    );
    eprintln!(
        "   Pricing:  ${:.6}/{}, max {}",
        job.job_plan.unit_cost_usd,
        job.job_plan.unit_label,
        format_cost(job.job_plan.max_cost_usd)
    );

    if !submitting {
        let redacted = redacted_spec(&job.spec);
        if json {
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "dryRun": true,
                    "confirmRequired": true,
                    "sourceRepo": job.source_repo,
                    "sourceFile": job.source_file,
                    "targetRepo": job.target_repo,
                    "modelId": job.model_id,
                    "jobPlan": job.job_plan,
                    "spec": redacted,
                }))?
            );
        } else {
            eprintln!();
            eprintln!("🔍 Dry run — no HF Job was submitted. Add --confirm to submit.");
            println!("{}", serde_json::to_string_pretty(&redacted)?);
        }
        return Ok(());
    }

    ensure_bucket_script_current(&hf_client).await?;

    // Submit.
    eprintln!();
    let jobs_client = jobs_client.as_ref().expect("jobs client initialized");
    let info = jobs_client.submit(&job.namespace, &job.spec).await?;
    let job_url = format!(
        "{}/jobs/{}/{}",
        jobs_client.endpoint(),
        job.namespace,
        info.id
    );
    eprintln!("🚀 Submitted: {}", info.id);
    eprintln!("   Console: {job_url}");
    eprintln!("   Status:  mesh-llm models package --status {}", info.id);
    eprintln!("   Logs:    mesh-llm models package --logs {}", info.id);

    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "submitted": true,
                "job": info,
                "jobUrl": job_url,
                "namespace": job.namespace,
                "sourceRepo": job.source_repo,
                "sourceFile": job.source_file,
                "targetRepo": job.target_repo,
                "modelId": job.model_id,
                "jobPlan": job.job_plan,
            }))?
        );
    }

    // Follow logs if requested.
    if follow {
        eprintln!();
        eprintln!("📜 Following logs...");
        eprintln!();
        follow_until_done(jobs_client, &job.namespace, &info.id).await?;
    }

    Ok(())
}

async fn run_list_quants(
    client: &hf_hub::HFClient,
    source_repo: &str,
    json_output: bool,
) -> Result<()> {
    let quants = prepare::list_quants(client, source_repo).await?;

    if json_output {
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "sourceRepo": source_repo,
                "quants": quants,
            }))?
        );
        return Ok(());
    }

    if quants.is_empty() {
        eprintln!("No GGUF files found in {source_repo}");
        return Ok(());
    }

    eprintln!("📦 Available quants in {source_repo}:");
    eprintln!();
    print_quant_table(&quants);
    eprintln!();
    eprintln!("Specify one as a model ref, e.g.:");
    eprintln!(
        "   mesh-llm models package {}:{}",
        source_repo, quants[0].name
    );

    Ok(())
}

fn print_quant_table(quants: &[DiscoveredQuant]) {
    // Find the longest name for alignment.
    let max_name = quants.iter().map(|q| q.name.len()).max().unwrap_or(0);

    for q in quants {
        let shard_str = if q.shard_count == 1 {
            "1 file".to_string()
        } else {
            format!("{} shards", q.shard_count)
        };
        eprintln!(
            "   {:<width$}   {:>9}, {}",
            q.name,
            shard_str,
            prepare::format_size(q.total_bytes),
            width = max_name
        );
    }
}

async fn run_update_script() -> Result<()> {
    eprintln!("📤 Uploading embedded script to meshllm/layer-split-output bucket...");
    let client = model_package::build_hf_client()?;

    // Check permissions first.
    let perms = permissions::check_permissions(&client).await?;
    if !perms.is_meshllm_member {
        anyhow::bail!(
            "Only meshllm org members can update the bucket script.\n\
             You are logged in as '{}' which is not in the meshllm org.",
            perms.username
        );
    }

    script::update_bucket_script(&client).await?;
    eprintln!(
        "✅ Bucket script updated ({} bytes)",
        script::EMBEDDED_SCRIPT_SIZE
    );
    Ok(())
}

async fn run_status(client: &HfJobsClient, job_id: &str, json_output: bool) -> Result<()> {
    let (namespace, id) = parse_job_id(job_id).await?;
    let info = client.inspect(&namespace, &id).await?;
    if json_output {
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "namespace": namespace,
                "job": info,
            }))?
        );
        return Ok(());
    }
    eprintln!("Job:     {}", info.id);
    eprintln!("Status:  {}", info.status.stage);
    if let Some(msg) = &info.status.message {
        eprintln!("Message: {msg}");
    }
    if let Some(created) = &info.created_at {
        eprintln!("Created: {created}");
    }
    Ok(())
}

async fn run_logs(client: &HfJobsClient, job_id: &str, json_output: bool) -> Result<()> {
    use model_package::jobs::JobStage;

    let (namespace, id) = parse_job_id(job_id).await?;

    let info = client.inspect(&namespace, &id).await?;
    if matches!(info.status.stage, JobStage::Running) && !json_output {
        eprintln!("Job is still running; draining currently buffered logs only.");
        eprintln!("Use --follow when submitting to stream until completion.");
        eprintln!();
    }

    let mut stream = std::pin::pin!(client.stream_logs(&namespace, &id).await?);
    loop {
        match tokio::time::timeout(std::time::Duration::from_secs(5), stream.next()).await {
            Ok(Some(Ok(text))) if json_output => {
                println!("{}", serde_json::to_string(&json!({ "data": text }))?);
            }
            Ok(Some(Ok(text))) => println!("{text}"),
            Ok(Some(Err(e))) => {
                eprintln!("Log stream error: {e}");
                break;
            }
            Ok(None) => break,
            Err(_) => break,
        }
    }
    Ok(())
}

async fn run_cancel(client: &HfJobsClient, job_id: &str, json_output: bool) -> Result<()> {
    let (namespace, id) = parse_job_id(job_id).await?;
    client.cancel(&namespace, &id).await?;
    if json_output {
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "namespace": namespace,
                "jobId": id,
                "canceled": true,
            }))?
        );
    } else {
        eprintln!("✅ Job {id} canceled");
    }
    Ok(())
}

async fn run_list(client: &HfJobsClient, json_output: bool) -> Result<()> {
    // We need to know the namespace — resolve via whoami.
    let hf_client = model_package::build_hf_client()?;
    let perms = permissions::check_permissions(&hf_client).await?;

    let jobs = client.list(&perms.namespace).await?;
    if json_output {
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "namespace": perms.namespace,
                "jobs": jobs,
            }))?
        );
        return Ok(());
    }
    if jobs.is_empty() {
        eprintln!("No jobs found in namespace '{}'", perms.namespace);
        return Ok(());
    }

    eprintln!("Recent jobs in '{}':", perms.namespace);
    eprintln!();
    for job in &jobs {
        let created = job.created_at.as_deref().unwrap_or("?");
        eprintln!("  {} {} {}", job.id, job.status.stage, created);
    }
    Ok(())
}

/// Follow job logs until the job reaches a terminal state.
async fn follow_until_done(client: &HfJobsClient, namespace: &str, job_id: &str) -> Result<()> {
    use model_package::jobs::JobStage;

    loop {
        loop {
            let info = client.inspect(namespace, job_id).await?;
            match info.status.stage {
                JobStage::Running => break,
                JobStage::Completed => {
                    eprintln!("Job {} finished: {}", job_id, info.status.stage);
                    return Ok(());
                }
                JobStage::Error | JobStage::Canceled | JobStage::Deleted => {
                    if let Some(msg) = &info.status.message {
                        eprintln!("Message: {msg}");
                    }
                    anyhow::bail!(
                        "Job {} finished unsuccessfully: {}",
                        job_id,
                        info.status.stage
                    );
                }
                _ => tokio::time::sleep(std::time::Duration::from_secs(3)).await,
            }
        }

        let mut stream = std::pin::pin!(client.stream_logs(namespace, job_id).await?);
        while let Some(line) = stream.next().await {
            match line {
                Ok(text) => println!("{text}"),
                Err(e) => {
                    eprintln!("Log stream error: {e}");
                    break;
                }
            }
        }

        let info = client.inspect(namespace, job_id).await?;
        match info.status.stage {
            JobStage::Completed => {
                eprintln!();
                eprintln!("Job {} finished: {}", job_id, info.status.stage);
                return Ok(());
            }
            JobStage::Error | JobStage::Canceled | JobStage::Deleted => {
                if let Some(msg) = &info.status.message {
                    eprintln!("Message: {msg}");
                }
                anyhow::bail!(
                    "Job {} finished unsuccessfully: {}",
                    job_id,
                    info.status.stage
                );
            }
            _ => {
                eprintln!(
                    "Log stream ended while job is still {}; reconnecting...",
                    info.status.stage
                );
                tokio::time::sleep(std::time::Duration::from_secs(3)).await;
            }
        }
    }
}

async fn ensure_bucket_script_current(client: &hf_hub::HFClient) -> Result<()> {
    match script::check_bucket_script(client).await {
        Ok(freshness) if freshness.is_current => Ok(()),
        Ok(freshness) => {
            eprintln!(
                "Bucket script is out of date ({}); updating it now...",
                freshness
                    .mismatch_reason
                    .as_deref()
                    .unwrap_or("embedded script differs from bucket script")
            );
            script::update_bucket_script(client).await?;
            eprintln!("Bucket script updated.");
            Ok(())
        }
        Err(err) => {
            eprintln!(
                "Could not check bucket script freshness ({err:#}); uploading current script..."
            );
            script::update_bucket_script(client).await?;
            eprintln!("Bucket script updated.");
            Ok(())
        }
    }
}

fn redacted_spec(spec: &model_package::jobs::JobSpec) -> model_package::jobs::JobSpec {
    let mut redacted = spec.clone();
    for value in redacted.secrets.values_mut() {
        if value.len() > 8 {
            *value = format!("{}...{}", &value[..4], &value[value.len() - 4..]);
        } else {
            *value = "****".to_string();
        }
    }
    redacted
}

fn hardware_label(cpu: Option<&str>, ram: Option<&str>) -> String {
    match (cpu, ram) {
        (Some(cpu), Some(ram)) => format!("({cpu}, {ram})"),
        (Some(cpu), None) => format!("({cpu})"),
        (None, Some(ram)) => format!("({ram})"),
        (None, None) => String::new(),
    }
}

fn format_cost(value: f64) -> String {
    format!("${value:.2} USD")
}

/// Parse a job ID that may or may not include a namespace prefix.
///
/// If the job ID contains a `/`, treat the first part as the namespace.
/// Otherwise, resolve the namespace via whoami.
async fn parse_job_id(job_id: &str) -> Result<(String, String)> {
    if let Some((ns, id)) = job_id.split_once('/') {
        Ok((ns.to_string(), id.to_string()))
    } else {
        // Need to figure out namespace from the user's identity.
        let hf_client = model_package::build_hf_client()?;
        let perms = permissions::check_permissions(&hf_client).await?;
        Ok((perms.namespace, job_id.to_string()))
    }
}

/// Parse a human-readable timeout string like "3h", "2h30m", "7200" into seconds.
fn parse_timeout(s: &str) -> Result<u64> {
    let s = s.trim();

    // Pure number → seconds.
    if let Ok(secs) = s.parse::<u64>() {
        return Ok(secs);
    }

    let mut total: u64 = 0;
    let mut current = String::new();

    for ch in s.chars() {
        if ch.is_ascii_digit() {
            current.push(ch);
        } else {
            let val: u64 = current
                .parse()
                .with_context(|| format!("invalid timeout: '{s}'"))?;
            current.clear();

            match ch {
                'h' | 'H' => total += val * 3600,
                'm' | 'M' => total += val * 60,
                's' | 'S' => total += val,
                _ => anyhow::bail!("invalid timeout unit '{ch}' in '{s}'"),
            }
        }
    }

    // Handle trailing number without unit (treat as seconds).
    if !current.is_empty() {
        let val: u64 = current.parse()?;
        total += val;
    }

    if total == 0 {
        anyhow::bail!("timeout must be > 0: '{s}'");
    }

    Ok(total)
}

fn format_timeout(seconds: u64) -> String {
    let hours = seconds / 3600;
    let minutes = (seconds % 3600) / 60;
    if minutes > 0 {
        format!("{hours}h{minutes}m")
    } else {
        format!("{hours}h")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_timeout_hours() {
        assert_eq!(parse_timeout("3h").unwrap(), 10800);
    }

    #[test]
    fn parse_timeout_hours_minutes() {
        assert_eq!(parse_timeout("2h30m").unwrap(), 9000);
    }

    #[test]
    fn parse_timeout_plain_seconds() {
        assert_eq!(parse_timeout("7200").unwrap(), 7200);
    }

    #[test]
    fn parse_timeout_mixed() {
        assert_eq!(parse_timeout("1h30m45s").unwrap(), 5445);
    }
}
