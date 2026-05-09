use std::collections::HashMap;

use anyhow::{Context, Result};
use hf_hub::HFClient;
use serde::Serialize;

use crate::jobs::{CpuJobPlan, JobSpec, JobVolume};
use crate::permissions::PermissionCheck;
use crate::prepare::{self, ResolvedSourceQuant};

/// Parameters for a family-certification HF Job.
pub struct CertifyParams {
    pub source_repo: String,
    pub quant: Option<String>,
    pub family: String,
    pub artifact_repo: Option<String>,
    pub model_id: Option<String>,
    pub flavor: String,
    pub timeout_seconds: u64,
    pub mesh_llm_ref: String,
    pub job_image: Option<String>,
    pub hf_token: Option<String>,
    pub options: CertifyOptions,
}

/// Optional arguments forwarded to scripts/family-certify.sh.
#[derive(Debug, Clone, Default, Serialize)]
pub struct CertifyOptions {
    pub run_id: Option<String>,
    pub layer_end: Option<String>,
    pub split_layer: Option<String>,
    pub splits: Option<String>,
    pub activation_width: Option<String>,
    pub prompt: Option<String>,
    pub ctx_size: Option<String>,
    pub n_gpu_layers: Option<String>,
    pub startup_timeout_secs: Option<String>,
    pub wire_dtype: Option<String>,
    pub wire_dtypes: Option<String>,
    pub prefix_token_count: Option<String>,
    pub cache_hit_repeats: Option<String>,
    pub allow_mismatch: bool,
    pub strict_dtype: bool,
    pub skip_correctness: bool,
    pub skip_dtype: bool,
    pub skip_state: bool,
    pub borrow_resident_hits: bool,
}

/// A resolved certification job, ready to submit.
pub struct CertifyJob {
    pub source_repo: String,
    pub source_file: String,
    pub family: String,
    pub artifact_repo: Option<String>,
    pub model_id: String,
    pub namespace: String,
    pub resolved_quant: ResolvedSourceQuant,
    pub job_plan: CpuJobPlan,
    pub spec: JobSpec,
}

/// Resolve source files and build the certification HF Job spec.
pub async fn resolve(
    client: &HFClient,
    params: CertifyParams,
    permissions: &PermissionCheck,
) -> Result<CertifyJob> {
    let quant = params
        .quant
        .as_deref()
        .context("--quant is required when submitting a certification job")?;

    let resolved =
        prepare::resolve_source_quant(client, &params.source_repo, quant, params.model_id.clone())
            .await?;

    let job_plan = crate::jobs::plan_certification_cpu_job(
        &crate::jobs::hf_endpoint(),
        &params.flavor,
        params.timeout_seconds,
        resolved.quant.total_bytes,
    )
    .await?;

    let mut environment = HashMap::new();
    environment.insert("SOURCE_REPO".into(), params.source_repo.clone());
    environment.insert("SOURCE_FILE".into(), resolved.source_file.clone());
    environment.insert("MODEL_ID".into(), resolved.model_id.clone());
    environment.insert("FAMILY".into(), params.family.clone());
    environment.insert("SOURCE_REVISION".into(), "main".into());
    environment.insert("MESH_LLM_REF".into(), params.mesh_llm_ref.clone());

    if let Some(artifact_repo) = &params.artifact_repo {
        environment.insert("ARTIFACT_REPO".into(), artifact_repo.clone());
    }
    insert_option(&mut environment, "RUN_ID", &params.options.run_id);
    insert_option(&mut environment, "LAYER_END", &params.options.layer_end);
    insert_option(&mut environment, "SPLIT_LAYER", &params.options.split_layer);
    insert_option(&mut environment, "SPLITS", &params.options.splits);
    insert_option(
        &mut environment,
        "ACTIVATION_WIDTH",
        &params.options.activation_width,
    );
    insert_option(&mut environment, "PROMPT", &params.options.prompt);
    insert_option(&mut environment, "CTX_SIZE", &params.options.ctx_size);
    insert_option(
        &mut environment,
        "N_GPU_LAYERS",
        &params.options.n_gpu_layers,
    );
    insert_option(
        &mut environment,
        "STARTUP_TIMEOUT_SECS",
        &params.options.startup_timeout_secs,
    );
    insert_option(&mut environment, "WIRE_DTYPE", &params.options.wire_dtype);
    insert_option(&mut environment, "WIRE_DTYPES", &params.options.wire_dtypes);
    insert_option(
        &mut environment,
        "PREFIX_TOKEN_COUNT",
        &params.options.prefix_token_count,
    );
    insert_option(
        &mut environment,
        "CACHE_HIT_REPEATS",
        &params.options.cache_hit_repeats,
    );
    insert_bool(
        &mut environment,
        "ALLOW_MISMATCH",
        params.options.allow_mismatch,
    );
    insert_bool(
        &mut environment,
        "STRICT_DTYPE",
        params.options.strict_dtype,
    );
    insert_bool(
        &mut environment,
        "SKIP_CORRECTNESS",
        params.options.skip_correctness,
    );
    insert_bool(&mut environment, "SKIP_DTYPE", params.options.skip_dtype);
    insert_bool(&mut environment, "SKIP_STATE", params.options.skip_state);
    insert_bool(
        &mut environment,
        "BORROW_RESIDENT_HITS",
        params.options.borrow_resident_hits,
    );

    // The HF Jobs API passes secrets as env vars inside the container.
    // Dry runs intentionally omit secrets so users can inspect cost/spec first.
    let mut secrets = HashMap::new();
    if params.artifact_repo.is_some() {
        if let Some(hf_token) = params.hf_token {
            secrets.insert("HF_TOKEN".into(), hf_token);
        }
    }

    let volumes = vec![
        JobVolume {
            volume_type: "model".into(),
            source: params.source_repo.clone(),
            mount_path: "/source".into(),
            read_only: Some(true),
        },
        JobVolume {
            volume_type: "bucket".into(),
            source: "meshllm/layer-split-output".into(),
            mount_path: "/bucket".into(),
            read_only: None,
        },
    ];

    let spec = JobSpec {
        docker_image: crate::jobs::resolve_hf_jobs_image(
            &params.mesh_llm_ref,
            params.job_image.as_deref(),
        ),
        command: vec!["bash".into(), "/bucket/certify-model-job.sh".into()],
        arguments: vec![],
        environment,
        secrets,
        flavor: job_plan.flavor.clone(),
        timeout_seconds: job_plan.timeout_seconds,
        volumes,
    };

    Ok(CertifyJob {
        source_repo: params.source_repo,
        source_file: resolved.source_file.clone(),
        family: params.family,
        artifact_repo: params.artifact_repo,
        model_id: resolved.model_id.clone(),
        namespace: permissions.namespace.clone(),
        resolved_quant: resolved,
        job_plan,
        spec,
    })
}

fn insert_option(environment: &mut HashMap<String, String>, name: &str, value: &Option<String>) {
    if let Some(value) = value.as_ref().filter(|value| !value.trim().is_empty()) {
        environment.insert(name.to_string(), value.clone());
    }
}

fn insert_bool(environment: &mut HashMap<String, String>, name: &str, value: bool) {
    if value {
        environment.insert(name.to_string(), "true".to_string());
    }
}
