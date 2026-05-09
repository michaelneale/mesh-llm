use anyhow::{bail, Context, Result};
use model_package::certify::{self, CertifyOptions, CertifyParams};
use model_package::jobs::HfJobsClient;
use model_package::{permissions, script};
use serde_json::json;

use crate::cli::commands::model_package as package_command;

/// All remote HF Job arguments for `models certify --hf-job`, bundled to avoid too-many-arguments.
pub(crate) struct ModelCertifyArgs<'a> {
    pub source_repo: Option<&'a str>,
    pub family: Option<&'a str>,
    pub quant: Option<&'a str>,
    pub artifact_repo: Option<&'a str>,
    pub model_id: Option<&'a str>,
    pub flavor: &'a str,
    pub timeout: &'a str,
    pub mesh_llm_ref: &'a str,
    pub job_image: &'a str,
    pub dry_run: bool,
    pub confirm: bool,
    pub follow: bool,
    pub json: bool,
    pub status: Option<&'a str>,
    pub logs: Option<&'a str>,
    pub cancel: Option<&'a str>,
    pub list: bool,
    pub update_script: bool,
    pub options: CertifyOptions,
}

/// Dispatch the model-certification command.
pub(crate) async fn dispatch_model_certify(args: ModelCertifyArgs<'_>) -> Result<()> {
    let ModelCertifyArgs {
        source_repo,
        family,
        quant,
        artifact_repo,
        model_id,
        flavor,
        timeout,
        mesh_llm_ref,
        job_image,
        dry_run,
        confirm,
        follow,
        json,
        status,
        logs,
        cancel,
        list,
        update_script,
        options,
    } = args;

    if update_script {
        return run_update_script().await;
    }

    if let Some(job_id) = status {
        let jobs_client = HfJobsClient::from_env()?;
        return package_command::run_status(&jobs_client, job_id, json).await;
    }
    if let Some(job_id) = logs {
        let jobs_client = HfJobsClient::from_env()?;
        return package_command::run_logs(&jobs_client, job_id, json).await;
    }
    if let Some(job_id) = cancel {
        let jobs_client = HfJobsClient::from_env()?;
        return package_command::run_cancel(&jobs_client, job_id, json).await;
    }
    if list {
        let jobs_client = HfJobsClient::from_env()?;
        return package_command::run_list(&jobs_client, json).await;
    }

    let source_ref = source_repo.context(
        "Source repo is required for certification job submission.\n\
         Usage: mesh-llm models certify <source_repo>:<quant> --hf-job --family <family>",
    )?;
    let source_model_ref = model_ref::ModelRef::parse(source_ref)
        .with_context(|| format!("invalid source model ref: {source_ref}"))?;
    let source_repo = source_model_ref.repo.as_str();
    let source_quant = match (source_model_ref.selector.as_deref(), quant) {
        (Some(selector), Some(quant)) if selector != quant => {
            bail!(
                "source ref selector '{selector}' conflicts with requested quant '{quant}'. \
                 Use `mesh-llm models certify {source_repo}:{selector} --hf-job --family ...`."
            );
        }
        (Some(selector), _) => Some(selector),
        (None, Some(quant)) => Some(quant),
        (None, None) => None,
    };
    let source_quant = source_quant.with_context(|| {
        format!(
            "source ref must include a quant selector, for example `mesh-llm models certify {source_repo}:Q4_K_M --hf-job --family ...`"
        )
    })?;
    let family = family.context("--family is required for certification jobs")?;

    let submitting = confirm && !dry_run;
    let jobs_client = if submitting {
        Some(HfJobsClient::from_env()?)
    } else {
        None
    };

    let hf_client = model_package::build_hf_client()?;
    eprintln!("🔑 Checking permissions...");
    let perms = permissions::check_permissions(&hf_client).await?;

    let timeout_seconds = package_command::parse_timeout(timeout)?;

    eprintln!("🔍 Resolving certification source...");
    let params = CertifyParams {
        source_repo: source_repo.to_string(),
        quant: Some(source_quant.to_string()),
        family: family.to_string(),
        artifact_repo: artifact_repo.map(|s| s.to_string()),
        model_id: model_id.map(|s| s.to_string()),
        flavor: flavor.to_string(),
        timeout_seconds,
        mesh_llm_ref: mesh_llm_ref.to_string(),
        job_image: Some(job_image.to_string()),
        hf_token: jobs_client
            .as_ref()
            .map(|client| client.token().to_string()),
        options,
    };

    let job = certify::resolve(&hf_client, params, &perms).await?;
    let redacted = package_command::redacted_spec(&job.spec);

    eprintln!("   Repo:   {}", job.source_repo);
    eprintln!("   File:   {}", job.source_file);
    eprintln!("   Family: {}", job.family);
    eprintln!("   Model:  {}", job.model_id);
    if let Some(artifact_repo) = &job.artifact_repo {
        eprintln!("   Artifacts: {artifact_repo}");
    } else {
        eprintln!("   Artifacts: not published unless --artifact-repo is set");
    }
    eprintln!();
    eprintln!(
        "📋 Job: {}, timeout {}, mesh-llm@{}",
        job.job_plan.flavor,
        package_command::format_timeout(job.job_plan.timeout_seconds),
        job.spec
            .environment
            .get("MESH_LLM_REF")
            .map(|s| s.as_str())
            .unwrap_or("main")
    );
    eprintln!(
        "   Hardware: {} {} ({})",
        job.job_plan.pretty_name,
        package_command::hardware_label(job.job_plan.cpu.as_deref(), job.job_plan.ram.as_deref()),
        job.job_plan.selection_reason
    );
    eprintln!(
        "   Pricing:  ${:.6}/{}, max {}",
        job.job_plan.unit_cost_usd,
        job.job_plan.unit_label,
        package_command::format_cost(job.job_plan.max_cost_usd)
    );

    if !submitting {
        if json {
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "dryRun": true,
                    "confirmRequired": true,
                    "sourceRepo": job.source_repo,
                    "sourceFile": job.source_file,
                    "family": job.family,
                    "artifactRepo": job.artifact_repo,
                    "modelId": job.model_id,
                    "resolvedQuant": job.resolved_quant,
                    "jobPlan": job.job_plan,
                    "spec": redacted,
                }))?
            );
        } else {
            eprintln!();
            eprintln!("🔍 Dry run — no HF Job was submitted.");
            eprintln!(
                "Add --confirm to submit. Max cost: {}.",
                package_command::format_cost(job.job_plan.max_cost_usd)
            );
            println!("{}", serde_json::to_string_pretty(&redacted)?);
        }
        return Ok(());
    }

    package_command::ensure_bucket_job_script_current(&hf_client, script::JobScript::CertifyModel)
        .await?;

    eprintln!();
    eprintln!(
        "💸 Confirmed HF Jobs max cost: {}",
        package_command::format_cost(job.job_plan.max_cost_usd)
    );
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
    eprintln!("   Status:  mesh-llm models certify --status {}", info.id);
    eprintln!("   Logs:    mesh-llm models certify --logs {}", info.id);

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
                "family": job.family,
                "artifactRepo": job.artifact_repo,
                "modelId": job.model_id,
                "jobPlan": job.job_plan,
            }))?
        );
    }

    if follow {
        eprintln!();
        package_command::follow_until_done(jobs_client, &job.namespace, &info.id).await?;
    }

    Ok(())
}

async fn run_update_script() -> Result<()> {
    eprintln!("📤 Uploading embedded certification script to meshllm/layer-split-output bucket...");
    let client = model_package::build_hf_client()?;

    let perms = permissions::check_permissions(&client).await?;
    if !perms.is_meshllm_member {
        bail!(
            "Only meshllm org members can update the bucket script.\n\
             You are logged in as '{}' which is not in the meshllm org.",
            perms.username
        );
    }

    script::update_bucket_job_script(&client, script::JobScript::CertifyModel).await?;
    eprintln!(
        "✅ Certification script updated ({} bytes)",
        script::JobScript::CertifyModel.expected_size()
    );
    Ok(())
}
