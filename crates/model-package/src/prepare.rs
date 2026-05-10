use std::collections::HashMap;

use anyhow::{Context, Result};
use futures::StreamExt;
use hf_hub::types::{RepoListTreeParams, RepoTreeEntry};
use hf_hub::HFClient;
use model_ref::{
    gguf_matches_quant_selector, normalize_gguf_distribution_id, quant_selector_from_gguf_file,
    split_gguf_shard_info,
};
use serde::Serialize;

use crate::jobs::{CpuJobPlan, JobSpec, JobVolume};
use crate::permissions::PermissionCheck;

/// Parameters for a model-package job.
pub struct PrepareParams {
    pub source_repo: String,
    pub quant: Option<String>,
    pub target: Option<String>,
    pub model_id: Option<String>,
    pub flavor: String,
    pub timeout_seconds: u64,
    pub mesh_llm_ref: String,
    pub hf_token: Option<String>,
}

/// A fully resolved model-package job, ready to submit.
pub struct PrepareJob {
    pub source_repo: String,
    pub source_file: String,
    pub target_repo: String,
    pub model_id: String,
    pub namespace: String,
    pub catalog_create_pr: bool,
    pub job_plan: CpuJobPlan,
    pub spec: JobSpec,
}

/// A discovered quant variant in a HF model repo.
#[derive(Debug, Clone, Serialize)]
pub struct DiscoveredQuant {
    /// The quant selector name (e.g. "Q4_K_M", "UD-Q4_K_XL").
    pub name: String,
    /// Number of GGUF files (shards) for this quant.
    pub shard_count: usize,
    /// Total size in bytes across all shards.
    pub total_bytes: u64,
    /// The first shard file path (or single file path).
    pub first_file: String,
}

/// List all available GGUF quant variants in a HF model repo.
pub async fn list_quants(client: &HFClient, repo: &str) -> Result<Vec<DiscoveredQuant>> {
    let (owner, name) = parse_repo(repo)?;
    let hf_repo = client.model(&owner, &name);

    let params = RepoListTreeParams::builder().recursive(true).build();

    let stream = hf_repo.list_tree(&params).context("list repo tree")?;

    futures::pin_mut!(stream);

    // Collect all GGUF files with their sizes.
    let mut gguf_files: Vec<(String, u64)> = Vec::new();
    while let Some(entry) = stream.next().await {
        let entry = entry.context("read repo tree entry")?;
        if let RepoTreeEntry::File { path, size, .. } = entry {
            if path.ends_with(".gguf") {
                gguf_files.push((path, size));
            }
        }
    }

    Ok(discover_quants_from_gguf_files(gguf_files))
}

/// Group GGUF files into quant variants.
pub fn discover_quants_from_gguf_files(gguf_files: Vec<(String, u64)>) -> Vec<DiscoveredQuant> {
    let mut quant_map: HashMap<String, Vec<(String, u64)>> = HashMap::new();
    for (path, size) in gguf_files {
        if let Some(selector) = quant_selector_from_gguf_file(&path) {
            quant_map.entry(selector).or_default().push((path, size));
        } else if let Some(dist_id) = normalize_gguf_distribution_id(&path) {
            // Fallback: use the distribution ID as the selector.
            quant_map.entry(dist_id).or_default().push((path, size));
        }
    }

    let mut quants: Vec<DiscoveredQuant> = quant_map
        .into_iter()
        .map(|(name, mut files)| {
            // Sort so shard -00001 comes first.
            files.sort_by(|a, b| a.0.cmp(&b.0));
            let total_bytes = files.iter().map(|(_, size)| size).sum();
            let shard_count = files.len();
            let first_file = files[0].0.clone();
            DiscoveredQuant {
                name,
                shard_count,
                total_bytes,
                first_file,
            }
        })
        .collect();

    // Sort by name for stable output.
    quants.sort_by(|a, b| a.name.cmp(&b.name));
    quants
}

/// Resolve source files, permissions, target repo, and build the job spec.
pub async fn resolve(
    client: &HFClient,
    params: PrepareParams,
    permissions: &PermissionCheck,
) -> Result<PrepareJob> {
    let quant = params
        .quant
        .as_deref()
        .context("--quant is required when submitting a job")?;

    let quants = list_quants(client, &params.source_repo).await?;

    if quants.is_empty() {
        anyhow::bail!("No GGUF files found in {}", params.source_repo);
    }

    // Find matching quant.
    let matched = quants
        .iter()
        .find(|q| q.name.eq_ignore_ascii_case(quant))
        .or_else(|| {
            // Fall back to gguf_matches_quant_selector on the first file.
            quants
                .iter()
                .find(|q| gguf_matches_quant_selector(&q.first_file, quant))
        })
        .with_context(|| {
            let available: Vec<&str> = quants.iter().map(|q| q.name.as_str()).collect();
            format!(
                "No quant matching '{}' in {}.\nAvailable: {}",
                quant,
                params.source_repo,
                available.join(", ")
            )
        })?;

    // For sharded models, ensure we have the first shard.
    let source_file = if let Some(shard) = split_gguf_shard_info(&matched.first_file) {
        // Verify it's shard 00001.
        if shard.part != "00001" {
            // Reconstruct the -00001- path.
            let first = matched
                .first_file
                .replace(&format!("-{}-of-", shard.part), "-00001-of-");
            first
        } else {
            matched.first_file.clone()
        }
    } else {
        matched.first_file.clone()
    };

    // Derive distribution ID and target repo.
    let dist_id =
        normalize_gguf_distribution_id(&source_file).unwrap_or_else(|| matched.name.clone());

    let target_repo = params
        .target
        .unwrap_or_else(|| format!("{}/{}-layers", permissions.namespace, dist_id));

    let model_id = params.model_id.unwrap_or_else(|| {
        model_ref::format_gguf_selection_ref(&params.source_repo, &source_file, &matched.name)
    });

    let job_plan = crate::jobs::plan_cpu_job(
        &crate::jobs::hf_endpoint(),
        &params.flavor,
        params.timeout_seconds,
        matched.total_bytes,
    )
    .await?;

    // Build environment variables.
    let mut environment = HashMap::new();
    environment.insert("SOURCE_REPO".into(), params.source_repo.clone());
    environment.insert("SOURCE_FILE".into(), source_file.clone());
    environment.insert("SOURCE_QUANT".into(), matched.name.clone());
    environment.insert("SOURCE_TOTAL_BYTES".into(), matched.total_bytes.to_string());
    environment.insert("TARGET_REPO".into(), target_repo.clone());
    environment.insert("MODEL_ID".into(), model_id.clone());
    environment.insert("SOURCE_REVISION".into(), "main".into());
    environment.insert("MESH_LLM_REF".into(), params.mesh_llm_ref.clone());
    environment.insert(
        "CATALOG_CREATE_PR".into(),
        if permissions.catalog_create_pr {
            "true"
        } else {
            "false"
        }
        .into(),
    );

    // The HF Jobs API passes secrets as env vars inside the container.
    // Dry runs intentionally omit secrets so users can inspect cost/spec first.
    let mut secrets = HashMap::new();
    if let Some(hf_token) = params.hf_token {
        secrets.insert("HF_TOKEN".into(), hf_token);
    }

    let volumes = vec![JobVolume {
        volume_type: "bucket".into(),
        source: "meshllm/layer-split-output".into(),
        mount_path: "/bucket".into(),
        read_only: None,
    }];

    let spec = JobSpec {
        docker_image: "ubuntu:22.04".into(),
        command: vec!["bash".into(), "/bucket/split-model-job.sh".into()],
        arguments: vec![],
        environment,
        secrets,
        flavor: job_plan.flavor.clone(),
        timeout_seconds: job_plan.timeout_seconds,
        volumes,
    };

    Ok(PrepareJob {
        source_repo: params.source_repo,
        source_file,
        target_repo,
        model_id,
        namespace: permissions.namespace.clone(),
        catalog_create_pr: permissions.catalog_create_pr,
        job_plan,
        spec,
    })
}

/// Format a byte count as a human-readable size.
pub fn format_size(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = 1024 * KB;
    const GB: u64 = 1024 * MB;
    const TB: u64 = 1024 * GB;

    if bytes >= TB {
        format!("{:.1} TB", bytes as f64 / TB as f64)
    } else if bytes >= GB {
        format!("{:.1} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.1} KB", bytes as f64 / KB as f64)
    } else {
        format!("{bytes} B")
    }
}

fn parse_repo(repo: &str) -> Result<(String, String)> {
    let parts: Vec<&str> = repo.splitn(2, '/').collect();
    if parts.len() != 2 {
        anyhow::bail!("Invalid repo format: '{}'. Expected 'owner/name'.", repo);
    }
    Ok((parts[0].to_string(), parts[1].to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn groups_sharded_quant_files() {
        let quants = discover_quants_from_gguf_files(vec![
            (
                "UD-Q4_K_XL/Qwen3-32B-UD-Q4_K_XL-00002-of-00002.gguf".to_string(),
                20,
            ),
            (
                "UD-Q4_K_XL/Qwen3-32B-UD-Q4_K_XL-00001-of-00002.gguf".to_string(),
                10,
            ),
        ]);

        assert_eq!(quants.len(), 1);
        assert_eq!(quants[0].name, "UD-Q4_K_XL");
        assert_eq!(quants[0].shard_count, 2);
        assert_eq!(quants[0].total_bytes, 30);
        assert!(quants[0].first_file.ends_with("00001-of-00002.gguf"));
    }

    #[test]
    fn groups_root_quant_files_by_selector() {
        let quants = discover_quants_from_gguf_files(vec![
            ("Qwen3-8B-Q4_K_M.gguf".to_string(), 5),
            ("Qwen3-8B-Q8_0.gguf".to_string(), 9),
        ]);

        let names = quants
            .iter()
            .map(|quant| quant.name.as_str())
            .collect::<Vec<_>>();
        assert_eq!(names, vec!["Q4_K_M", "Q8_0"]);
    }
}
