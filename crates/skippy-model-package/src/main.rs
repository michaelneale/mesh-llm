use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process::Command as ProcessCommand;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{bail, ensure, Context, Result};
use clap::{Parser, Subcommand};
use model_artifact::{ModelArtifactFile, ResolvedModelArtifact};
use model_hf::HfModelRepository;
use model_ref::{
    format_canonical_ref, normalize_gguf_distribution_id, parse_model_ref, split_gguf_shard_info,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use skippy_ffi::TensorRole;
use skippy_runtime::{write_gguf_from_parts, ModelInfo, TensorInfo};

#[derive(Debug, Parser)]
#[command(name = "skippy-model-package")]
#[command(about = "Inspect, plan, write, and validate skippy model packages")]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Inspect {
        model: PathBuf,
    },
    Plan {
        model: PathBuf,
        #[arg(long)]
        stages: usize,
    },
    Write {
        model: PathBuf,
        #[arg(long)]
        layers: String,
        #[arg(long)]
        out: PathBuf,
        #[arg(long)]
        stage_index: Option<u32>,
        #[arg(long)]
        include_embeddings: bool,
        #[arg(long)]
        include_output: bool,
        #[arg(long)]
        manifest: Option<PathBuf>,
    },
    WriteStages {
        model: PathBuf,
        #[arg(long)]
        stages: usize,
        #[arg(long)]
        out_dir: PathBuf,
    },
    WritePackage {
        model: String,
        #[arg(long)]
        out_dir: PathBuf,
        #[arg(long = "projector")]
        projectors: Vec<PathBuf>,
        #[arg(long)]
        after_artifact_command: Option<PathBuf>,
        #[arg(long)]
        model_id: Option<String>,
        #[arg(long)]
        source_repo: Option<String>,
        #[arg(long)]
        source_revision: Option<String>,
        #[arg(long)]
        source_file: Option<String>,
    },
    Validate {
        full: PathBuf,
        slices: Vec<PathBuf>,
    },
    ValidatePackage {
        full: PathBuf,
        package: PathBuf,
    },
}

#[derive(Debug, Serialize)]
struct InspectOutput {
    tensor_count: usize,
    tensors: Vec<TensorOutput>,
}

#[derive(Debug, Serialize)]
struct TensorOutput {
    name: String,
    layer_index: Option<u32>,
    role: String,
    ggml_type: u32,
    byte_size: u64,
}

#[derive(Debug, Serialize)]
struct PlanOutput {
    schema_version: u32,
    stage_count: usize,
    layer_count: u32,
    stages: Vec<StagePlan>,
}

#[derive(Debug, Clone, Serialize)]
struct StagePlan {
    stage_index: usize,
    layer_start: u32,
    layer_end: u32,
    includes_embeddings: bool,
    includes_output: bool,
    tensor_count: usize,
    tensor_bytes: u64,
}

#[derive(Debug, Serialize)]
struct SliceManifest {
    schema_version: u32,
    source_model: String,
    source_sha256: String,
    stage_count: usize,
    layer_count: u32,
    stages: Vec<SliceManifestStage>,
}

#[derive(Debug, Serialize)]
struct SliceManifestStage {
    stage_index: usize,
    layer_start: u32,
    layer_end: u32,
    includes_embeddings: bool,
    includes_output: bool,
    path: String,
    tensor_count: usize,
    tensor_bytes: u64,
    artifact_bytes: u64,
    sha256: String,
}

#[derive(Debug, Serialize)]
struct ValidateOutput {
    valid: bool,
    full_tensor_count: usize,
    required_owned_tensor_count: usize,
    missing_owned_tensors: Vec<String>,
    duplicate_owned_tensors: Vec<String>,
    slices: Vec<ValidateSlice>,
}

#[derive(Debug, Serialize)]
struct ValidateSlice {
    path: String,
    tensor_count: usize,
    owned_tensor_count: usize,
    tensor_bytes: u64,
    missing_from_full: Vec<String>,
    sha256: String,
}

#[derive(Debug, Deserialize, Serialize)]
struct PackageManifest {
    schema_version: u32,
    model_id: String,
    source_model: PackageSourceModel,
    format: String,
    layer_count: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    activation_width: Option<u32>,
    shared: PackageShared,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    projectors: Vec<PackageProjector>,
    layers: Vec<PackageLayer>,
    skippy_abi_version: String,
    created_at_unix_secs: u64,
}

#[derive(Debug, Deserialize, Serialize)]
struct PackageSourceModel {
    path: String,
    sha256: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    repo: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    revision: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    primary_file: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    canonical_ref: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    distribution_id: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    files: Vec<ModelArtifactFile>,
}

#[derive(Debug, Deserialize, Serialize)]
struct PackageShared {
    metadata: PackageArtifact,
    embeddings: PackageArtifact,
    output: PackageArtifact,
}

#[derive(Debug, Deserialize, Serialize)]
struct PackageLayer {
    layer_index: u32,
    path: String,
    tensor_count: usize,
    tensor_bytes: u64,
    artifact_bytes: u64,
    sha256: String,
}

#[derive(Debug, Deserialize, Serialize)]
struct PackageArtifact {
    path: String,
    tensor_count: usize,
    tensor_bytes: u64,
    artifact_bytes: u64,
    sha256: String,
}

#[derive(Debug, Deserialize, Serialize)]
struct PackageProjector {
    kind: String,
    path: String,
    tensor_count: usize,
    tensor_bytes: u64,
    artifact_bytes: u64,
    sha256: String,
}

#[derive(Debug, Serialize)]
struct PackageValidateOutput {
    valid: bool,
    full_tensor_count: usize,
    layer_count: u32,
    manifest_layer_count_matches_model: bool,
    activation_width_matches_model: bool,
    expected_activation_width: u32,
    manifest_activation_width: Option<u32>,
    source_sha256_matches_manifest: bool,
    required_owned_tensor_count: usize,
    missing_owned_tensors: Vec<String>,
    duplicate_owned_tensors: Vec<String>,
    checked_artifact_count: usize,
    artifacts: Vec<PackageValidateArtifact>,
    checked_projector_count: usize,
    projectors: Vec<PackageValidateProjector>,
    missing_layers: Vec<u32>,
    duplicate_layers: Vec<u32>,
}

#[derive(Debug, Serialize)]
struct PackageValidateArtifact {
    path: String,
    tensor_count: usize,
    owned_tensor_count: usize,
    tensor_bytes: u64,
    artifact_bytes: u64,
    sha256_matches_manifest: bool,
    tensor_count_matches_manifest: bool,
    tensor_bytes_matches_manifest: bool,
    artifact_bytes_matches_manifest: bool,
    missing_from_full: Vec<String>,
}

#[derive(Debug, Serialize)]
struct PackageValidateProjector {
    path: String,
    kind: String,
    tensor_count: usize,
    tensor_bytes: u64,
    artifact_bytes: u64,
    sha256_matches_manifest: bool,
    tensor_count_matches_manifest: bool,
    tensor_bytes_matches_manifest: bool,
    artifact_bytes_matches_manifest: bool,
}

#[derive(Debug, Clone)]
struct PackageArtifactSpec {
    stage_index: u32,
    layer_start: u32,
    layer_end: u32,
    includes_embeddings: bool,
    includes_output: bool,
    relative_path: PathBuf,
}

#[derive(Debug, Clone)]
struct ArtifactHook {
    command: Option<PathBuf>,
}

#[derive(Debug, Default)]
struct ExplicitSourceIdentity {
    model_id: Option<String>,
    source_repo: Option<String>,
    source_revision: Option<String>,
    source_file: Option<String>,
}

#[derive(Debug)]
struct PackageInput {
    model_path: PathBuf,
    model_id: String,
    source_identity: PackageSourceIdentity,
}

#[derive(Debug)]
struct PackageSourceIdentity {
    repo: Option<String>,
    revision: Option<String>,
    primary_file: Option<String>,
    canonical_ref: Option<String>,
    distribution_id: Option<String>,
    files: Vec<ModelArtifactFile>,
}

struct ModelSource {
    paths: Vec<PathBuf>,
    infos: Vec<ModelInfo>,
    tensors: Vec<TensorInfo>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    match args.command {
        Command::Inspect { model } => inspect(model),
        Command::Plan { model, stages } => plan(model, stages),
        Command::Write {
            model,
            layers,
            out,
            stage_index,
            include_embeddings,
            include_output,
            manifest,
        } => write_one(
            model,
            layers,
            out,
            stage_index,
            include_embeddings,
            include_output,
            manifest,
        ),
        Command::WriteStages {
            model,
            stages,
            out_dir,
        } => write_stages(model, stages, out_dir),
        Command::WritePackage {
            model,
            out_dir,
            projectors,
            after_artifact_command,
            model_id,
            source_repo,
            source_revision,
            source_file,
        } => write_package(
            model,
            out_dir,
            projectors,
            ArtifactHook {
                command: after_artifact_command,
            },
            ExplicitSourceIdentity {
                model_id,
                source_repo,
                source_revision,
                source_file,
            },
        ),
        Command::Validate { full, slices } => validate(full, slices),
        Command::ValidatePackage { full, package } => validate_package(full, package),
    }
}

fn inspect(model: PathBuf) -> Result<()> {
    let source = ModelSource::open(&model)?;
    let tensors = source.tensors;
    let output = InspectOutput {
        tensor_count: tensors.len(),
        tensors: tensors.into_iter().map(tensor_output).collect(),
    };
    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(())
}

fn plan(model: PathBuf, stages: usize) -> Result<()> {
    let output = build_plan(&model, stages)?;
    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(())
}

fn write_one(
    model: PathBuf,
    layers: String,
    out: PathBuf,
    stage_index: Option<u32>,
    include_embeddings: bool,
    include_output: bool,
    manifest: Option<PathBuf>,
) -> Result<()> {
    let source = ModelSource::open(&model)?;
    let tensors = &source.tensors;
    let layer_count = layer_count(tensors)?;
    let (layer_start, layer_end) = parse_layer_range(&layers)?;
    if layer_end > layer_count {
        bail!("layer range end exceeds model layer count {layer_count}");
    }

    let stage_index = stage_index.unwrap_or(0);
    let includes_embeddings = include_embeddings || layer_start == 0;
    let includes_output = include_output || layer_end == layer_count;
    let stage = stage_plan_from_tensors(
        stage_index as usize,
        layer_start,
        layer_end,
        includes_embeddings,
        includes_output,
        tensors,
    );

    write_stage_artifact(&source, &stage, &out)?;

    if let Some(path) = manifest {
        let manifest = build_manifest(&model, layer_count, vec![(stage, out)])?;
        write_json_file(&path, &manifest)?;
    }
    Ok(())
}

fn write_stages(model: PathBuf, stages: usize, out_dir: PathBuf) -> Result<()> {
    if stages == 0 {
        bail!("--stages must be greater than zero");
    }
    fs::create_dir_all(&out_dir)
        .with_context(|| format!("create output directory {}", out_dir.display()))?;

    let source = ModelSource::open(&model)?;
    let tensors = &source.tensors;
    let plan = build_plan_from_tensors(stages, tensors)?;
    let mut written = Vec::new();
    for stage in plan.stages {
        let path = out_dir.join(format!("stage-{:03}.gguf", stage.stage_index));
        write_stage_artifact(&source, &stage, &path)?;
        written.push((stage, path));
    }

    let manifest = build_manifest(&model, plan.layer_count, written)?;
    let manifest_path = out_dir.join("slice-manifest.json");
    write_json_file(&manifest_path, &manifest)?;
    println!("{}", serde_json::to_string_pretty(&manifest)?);
    Ok(())
}

fn write_package(
    model: String,
    out_dir: PathBuf,
    projectors: Vec<PathBuf>,
    artifact_hook: ArtifactHook,
    explicit: ExplicitSourceIdentity,
) -> Result<()> {
    let input = resolve_package_input(model, explicit)?;
    fs::create_dir_all(&out_dir)
        .with_context(|| format!("create output directory {}", out_dir.display()))?;
    fs::create_dir_all(out_dir.join("shared"))
        .with_context(|| format!("create shared directory {}", out_dir.display()))?;
    fs::create_dir_all(out_dir.join("layers"))
        .with_context(|| format!("create layers directory {}", out_dir.display()))?;
    if !projectors.is_empty() {
        fs::create_dir_all(out_dir.join("projectors"))
            .with_context(|| format!("create projectors directory {}", out_dir.display()))?;
    }

    let source = ModelSource::open(&input.model_path)?;
    let tensors = &source.tensors;
    let layer_count = layer_count(tensors)?;
    let activation_width = activation_width(&input.model_path)?;
    let source_sha256 = file_sha256(&input.model_path)?;

    let metadata = write_package_artifact(
        &source,
        tensors,
        PackageArtifactSpec {
            stage_index: 0,
            layer_start: 0,
            layer_end: 0,
            includes_embeddings: false,
            includes_output: false,
            relative_path: PathBuf::from("shared/metadata.gguf"),
        },
        &out_dir,
        &artifact_hook,
    )?;
    let embeddings = write_package_artifact(
        &source,
        tensors,
        PackageArtifactSpec {
            stage_index: 1,
            layer_start: 0,
            layer_end: 0,
            includes_embeddings: true,
            includes_output: false,
            relative_path: PathBuf::from("shared/embeddings.gguf"),
        },
        &out_dir,
        &artifact_hook,
    )?;
    let output = write_package_artifact(
        &source,
        tensors,
        PackageArtifactSpec {
            stage_index: 2,
            layer_start: layer_count,
            layer_end: layer_count,
            includes_embeddings: false,
            includes_output: true,
            relative_path: PathBuf::from("shared/output.gguf"),
        },
        &out_dir,
        &artifact_hook,
    )?;

    let mut layers = Vec::new();
    for layer_index in 0..layer_count {
        let relative = PathBuf::from(format!("layers/layer-{layer_index:03}.gguf"));
        let artifact = write_package_artifact(
            &source,
            tensors,
            PackageArtifactSpec {
                stage_index: 1000 + layer_index,
                layer_start: layer_index,
                layer_end: layer_index + 1,
                includes_embeddings: false,
                includes_output: false,
                relative_path: relative,
            },
            &out_dir,
            &artifact_hook,
        )?;
        layers.push(PackageLayer {
            layer_index,
            path: artifact.path,
            tensor_count: artifact.tensor_count,
            tensor_bytes: artifact.tensor_bytes,
            artifact_bytes: artifact.artifact_bytes,
            sha256: artifact.sha256,
        });
    }

    let projectors = projectors
        .iter()
        .enumerate()
        .map(|(index, projector)| {
            copy_projector_artifact(projector, index, &out_dir, &artifact_hook)
        })
        .collect::<Result<Vec<_>>>()?;

    let manifest = PackageManifest {
        schema_version: 1,
        model_id: input.model_id,
        source_model: PackageSourceModel {
            path: input.model_path.display().to_string(),
            sha256: source_sha256,
            repo: input.source_identity.repo,
            revision: input.source_identity.revision,
            primary_file: input.source_identity.primary_file,
            canonical_ref: input.source_identity.canonical_ref,
            distribution_id: input.source_identity.distribution_id,
            files: input.source_identity.files,
        },
        format: "layer-package".to_string(),
        layer_count,
        activation_width: Some(activation_width),
        shared: PackageShared {
            metadata,
            embeddings,
            output,
        },
        projectors,
        layers,
        skippy_abi_version: format!(
            "{}.{}.{}",
            skippy_ffi::ABI_VERSION_MAJOR,
            skippy_ffi::ABI_VERSION_MINOR,
            skippy_ffi::ABI_VERSION_PATCH
        ),
        created_at_unix_secs: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .context("system clock before Unix epoch")?
            .as_secs(),
    };

    let manifest_path = out_dir.join("model-package.json");
    write_json_file(&manifest_path, &manifest)?;
    println!("{}", serde_json::to_string_pretty(&manifest)?);
    Ok(())
}

fn resolve_package_input(model: String, explicit: ExplicitSourceIdentity) -> Result<PackageInput> {
    let path = PathBuf::from(&model);
    if path.exists() {
        return resolve_local_package_input(path, explicit);
    }

    if explicit.model_id.is_some()
        || explicit.source_repo.is_some()
        || explicit.source_revision.is_some()
        || explicit.source_file.is_some()
    {
        bail!("explicit source identity flags are only valid when write-package input is a local path");
    }

    parse_model_ref(&model).with_context(|| {
        format!(
            "write-package input must be a model coordinate like org/repo:Q4_K_M, not {model:?}"
        )
    })?;

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("build async runtime for Hugging Face model resolution")?;

    runtime.block_on(async {
        let repository = HfModelRepository::from_env()?;
        let artifact = model_artifact::resolve_model_artifact_ref(&model, &repository).await?;
        let paths = repository.download_artifact_files(&artifact).await?;
        let primary_index = artifact
            .files
            .iter()
            .position(|file| file.path == artifact.primary_file)
            .context("resolved artifact file list did not include primary file")?;
        let model_path = paths
            .get(primary_index)
            .cloned()
            .context("downloaded artifact path list did not include primary file")?;
        Ok(package_input_from_resolved_artifact(model_path, artifact))
    })
}

fn resolve_local_package_input(
    model_path: PathBuf,
    explicit: ExplicitSourceIdentity,
) -> Result<PackageInput> {
    let model_id = explicit.model_id.context(
        "local write-package input requires --model-id; prefer passing a coordinate like org/repo:Q4_K_M",
    )?;
    let parsed_model_id = parse_model_ref(&model_id)
        .with_context(|| format!("--model-id must be a model coordinate, got {model_id:?}"))?;
    let cache_identity = if explicit.source_revision.is_none() || explicit.source_file.is_none() {
        HfModelRepository::from_env()
            .ok()
            .and_then(|repository| repository.identity_for_path(&model_path))
    } else {
        None
    };

    let repo = explicit
        .source_repo
        .or_else(|| {
            cache_identity
                .as_ref()
                .map(|identity| identity.repo_id.clone())
        })
        .unwrap_or_else(|| parsed_model_id.repo.clone());
    let revision = explicit
        .source_revision
        .or_else(|| cache_identity.as_ref().map(|identity| identity.revision.clone()))
        .context("local write-package input requires --source-revision for paths outside the Hugging Face cache")?;
    let primary_file = explicit
        .source_file
        .or_else(|| cache_identity.as_ref().map(|identity| identity.file.clone()))
        .context("local write-package input requires --source-file for paths outside the Hugging Face cache")?;
    let canonical_ref = format_canonical_ref(&repo, &revision, &primary_file);
    let distribution_id = normalize_gguf_distribution_id(&primary_file);
    let files = local_artifact_files(&model_path, &primary_file)?;

    Ok(PackageInput {
        model_path,
        model_id: parsed_model_id.display_id(),
        source_identity: PackageSourceIdentity {
            repo: Some(repo),
            revision: Some(revision),
            primary_file: Some(primary_file.clone()),
            canonical_ref: Some(canonical_ref),
            distribution_id,
            files,
        },
    })
}

fn package_input_from_resolved_artifact(
    model_path: PathBuf,
    artifact: ResolvedModelArtifact,
) -> PackageInput {
    PackageInput {
        model_path,
        model_id: artifact.model_id,
        source_identity: PackageSourceIdentity {
            repo: Some(artifact.source_repo),
            revision: Some(artifact.source_revision),
            primary_file: Some(artifact.primary_file),
            canonical_ref: Some(artifact.canonical_ref),
            distribution_id: Some(artifact.distribution_id),
            files: artifact.files,
        },
    }
}

#[cfg(test)]
fn model_distribution_id(model: &Path) -> Option<String> {
    model
        .to_str()
        .and_then(normalize_gguf_distribution_id)
        .or_else(|| {
            model
                .file_name()
                .and_then(|name| name.to_str())
                .and_then(normalize_gguf_distribution_id)
        })
}

fn validate(full: PathBuf, slices: Vec<PathBuf>) -> Result<()> {
    if slices.is_empty() {
        bail!("at least one slice path is required");
    }

    let full_source = ModelSource::open(&full)?;
    let full_tensors = full_source.tensors;
    let full_names: BTreeSet<_> = full_tensors
        .iter()
        .map(|tensor| tensor.name.as_str())
        .collect();
    let required_owned_tensors = full_tensors
        .iter()
        .filter(|tensor| is_owned_slice_tensor(tensor))
        .map(|tensor| tensor.name.clone())
        .collect::<BTreeSet<_>>();
    let mut owned_occurrences = BTreeMap::<String, usize>::new();
    let mut output = ValidateOutput {
        valid: true,
        full_tensor_count: full_tensors.len(),
        required_owned_tensor_count: required_owned_tensors.len(),
        missing_owned_tensors: Vec::new(),
        duplicate_owned_tensors: Vec::new(),
        slices: Vec::new(),
    };

    for path in slices {
        let source = ModelSource::open(&path)?;
        let tensors = source.tensors;
        let missing_from_full = tensors
            .iter()
            .filter(|tensor| !full_names.contains(tensor.name.as_str()))
            .map(|tensor| tensor.name.clone())
            .collect::<Vec<_>>();
        let owned_tensor_count = tensors
            .iter()
            .filter(|tensor| is_owned_slice_tensor(tensor))
            .inspect(|tensor| {
                *owned_occurrences.entry(tensor.name.clone()).or_default() += 1;
            })
            .count();
        if tensors.is_empty() || !missing_from_full.is_empty() {
            output.valid = false;
        }
        output.slices.push(ValidateSlice {
            path: path.display().to_string(),
            tensor_count: tensors.len(),
            owned_tensor_count,
            tensor_bytes: tensors.iter().map(|tensor| tensor.byte_size).sum(),
            missing_from_full,
            sha256: file_sha256(&path)?,
        });
    }

    output.missing_owned_tensors = required_owned_tensors
        .iter()
        .filter(|name| !owned_occurrences.contains_key(*name))
        .cloned()
        .collect();
    output.duplicate_owned_tensors = owned_occurrences
        .into_iter()
        .filter_map(|(name, count)| (count > 1).then_some(name))
        .collect();
    if !output.missing_owned_tensors.is_empty() || !output.duplicate_owned_tensors.is_empty() {
        output.valid = false;
    }

    println!("{}", serde_json::to_string_pretty(&output)?);
    if !output.valid {
        bail!("slice validation failed");
    }
    Ok(())
}

fn is_owned_slice_tensor(tensor: &TensorInfo) -> bool {
    matches!(
        tensor.role,
        TensorRole::Embedding | TensorRole::Layer | TensorRole::FinalNorm | TensorRole::Output
    )
}

fn validate_package(full: PathBuf, package: PathBuf) -> Result<()> {
    let full_source = ModelSource::open(&full)?;
    let full_tensors = full_source.tensors;
    let full_names: BTreeSet<_> = full_tensors
        .iter()
        .map(|tensor| tensor.name.as_str())
        .collect();
    let required_owned_tensors = full_tensors
        .iter()
        .filter(|tensor| is_owned_slice_tensor(tensor))
        .map(|tensor| tensor.name.clone())
        .collect::<BTreeSet<_>>();
    let mut owned_occurrences = BTreeMap::<String, usize>::new();
    let full_layer_count = layer_count(&full_tensors)?;
    let manifest_path = package.join("model-package.json");
    let manifest: PackageManifest = serde_json::from_str(
        &fs::read_to_string(&manifest_path)
            .with_context(|| format!("read {}", manifest_path.display()))?,
    )
    .with_context(|| format!("parse {}", manifest_path.display()))?;
    let source_sha256_matches_manifest = file_sha256(&full)? == manifest.source_model.sha256;
    let manifest_layer_count_matches_model = manifest.layer_count == full_layer_count;
    let expected_activation_width = activation_width(&full)?;
    let manifest_activation_width = manifest.activation_width;
    let activation_width_matches_model =
        manifest_activation_width == Some(expected_activation_width);

    let expected_layers = (0..manifest.layer_count).collect::<BTreeSet<_>>();
    let mut layer_occurrences = BTreeMap::<u32, usize>::new();
    for layer in &manifest.layers {
        *layer_occurrences.entry(layer.layer_index).or_default() += 1;
    }
    let actual_layers = layer_occurrences.keys().copied().collect::<BTreeSet<_>>();
    let missing_layers = expected_layers
        .difference(&actual_layers)
        .copied()
        .collect::<Vec<_>>();
    let duplicate_layers = layer_occurrences
        .into_iter()
        .filter_map(|(layer, count)| (count > 1).then_some(layer))
        .collect::<Vec<_>>();

    let mut artifacts = Vec::new();
    artifacts.push(validate_package_artifact(
        &package,
        &manifest.shared.metadata,
        &full_names,
        &mut owned_occurrences,
    )?);
    artifacts.push(validate_package_artifact(
        &package,
        &manifest.shared.embeddings,
        &full_names,
        &mut owned_occurrences,
    )?);
    artifacts.push(validate_package_artifact(
        &package,
        &manifest.shared.output,
        &full_names,
        &mut owned_occurrences,
    )?);
    for layer in &manifest.layers {
        artifacts.push(validate_package_artifact(
            &package,
            &PackageArtifact {
                path: layer.path.clone(),
                tensor_count: layer.tensor_count,
                tensor_bytes: layer.tensor_bytes,
                artifact_bytes: layer.artifact_bytes,
                sha256: layer.sha256.clone(),
            },
            &full_names,
            &mut owned_occurrences,
        )?);
    }
    let projectors = manifest
        .projectors
        .iter()
        .map(|projector| validate_package_projector(&package, projector))
        .collect::<Result<Vec<_>>>()?;

    let missing_owned_tensors = required_owned_tensors
        .iter()
        .filter(|name| !owned_occurrences.contains_key(*name))
        .cloned()
        .collect::<Vec<_>>();
    let duplicate_owned_tensors = owned_occurrences
        .into_iter()
        .filter_map(|(name, count)| (count > 1).then_some(name))
        .collect::<Vec<_>>();
    let valid = source_sha256_matches_manifest
        && manifest_layer_count_matches_model
        && activation_width_matches_model
        && missing_layers.is_empty()
        && duplicate_layers.is_empty()
        && missing_owned_tensors.is_empty()
        && duplicate_owned_tensors.is_empty()
        && artifacts.iter().all(|artifact| {
            artifact.sha256_matches_manifest
                && artifact.tensor_count_matches_manifest
                && artifact.tensor_bytes_matches_manifest
                && artifact.artifact_bytes_matches_manifest
                && artifact.missing_from_full.is_empty()
        })
        && projectors.iter().all(|projector| {
            projector.sha256_matches_manifest
                && projector.tensor_count_matches_manifest
                && projector.tensor_bytes_matches_manifest
                && projector.artifact_bytes_matches_manifest
        });
    let output = PackageValidateOutput {
        valid,
        full_tensor_count: full_tensors.len(),
        layer_count: manifest.layer_count,
        manifest_layer_count_matches_model,
        activation_width_matches_model,
        expected_activation_width,
        manifest_activation_width,
        source_sha256_matches_manifest,
        required_owned_tensor_count: required_owned_tensors.len(),
        missing_owned_tensors,
        duplicate_owned_tensors,
        checked_artifact_count: artifacts.len(),
        artifacts,
        checked_projector_count: projectors.len(),
        projectors,
        missing_layers,
        duplicate_layers,
    };
    println!("{}", serde_json::to_string_pretty(&output)?);
    if !valid {
        bail!("package validation failed");
    }
    Ok(())
}

fn validate_package_artifact(
    package: &Path,
    artifact: &PackageArtifact,
    full_names: &BTreeSet<&str>,
    owned_occurrences: &mut BTreeMap<String, usize>,
) -> Result<PackageValidateArtifact> {
    let path = package.join(&artifact.path);
    let info = ModelInfo::open(&path)?;
    let tensors = info.tensors()?;
    let missing_from_full = tensors
        .iter()
        .filter(|tensor| !full_names.contains(tensor.name.as_str()))
        .map(|tensor| tensor.name.clone())
        .collect::<Vec<_>>();
    let owned_tensor_count = tensors
        .iter()
        .filter(|tensor| is_owned_slice_tensor(tensor))
        .inspect(|tensor| {
            *owned_occurrences.entry(tensor.name.clone()).or_default() += 1;
        })
        .count();
    let tensor_bytes = tensors.iter().map(|tensor| tensor.byte_size).sum();
    let artifact_bytes = fs::metadata(&path)
        .with_context(|| format!("read artifact metadata {}", path.display()))?
        .len();
    Ok(PackageValidateArtifact {
        path: artifact.path.clone(),
        tensor_count: tensors.len(),
        owned_tensor_count,
        tensor_bytes,
        artifact_bytes,
        sha256_matches_manifest: file_sha256(&path)? == artifact.sha256,
        tensor_count_matches_manifest: tensors.len() == artifact.tensor_count,
        tensor_bytes_matches_manifest: tensor_bytes == artifact.tensor_bytes,
        artifact_bytes_matches_manifest: artifact_bytes == artifact.artifact_bytes,
        missing_from_full,
    })
}

fn validate_package_projector(
    package: &Path,
    projector: &PackageProjector,
) -> Result<PackageValidateProjector> {
    let path = package.join(&projector.path);
    let info = ModelInfo::open(&path)
        .with_context(|| format!("open package projector {}", path.display()))?;
    let tensors = info
        .tensors()
        .with_context(|| format!("read package projector tensors {}", path.display()))?;
    let tensor_bytes = tensors.iter().map(|tensor| tensor.byte_size).sum();
    let artifact_bytes = fs::metadata(&path)
        .with_context(|| format!("read projector metadata {}", path.display()))?
        .len();
    Ok(PackageValidateProjector {
        path: projector.path.clone(),
        kind: projector.kind.clone(),
        tensor_count: tensors.len(),
        tensor_bytes,
        artifact_bytes,
        sha256_matches_manifest: file_sha256(&path)? == projector.sha256,
        tensor_count_matches_manifest: tensors.len() == projector.tensor_count,
        tensor_bytes_matches_manifest: tensor_bytes == projector.tensor_bytes,
        artifact_bytes_matches_manifest: artifact_bytes == projector.artifact_bytes,
    })
}

fn build_plan(model: &Path, stages: usize) -> Result<PlanOutput> {
    if stages == 0 {
        bail!("--stages must be greater than zero");
    }
    let source = ModelSource::open(model)?;
    build_plan_from_tensors(stages, &source.tensors)
}

fn build_plan_from_tensors(stages: usize, tensors: &[TensorInfo]) -> Result<PlanOutput> {
    let layer_count = layer_count(tensors)?;
    if stages as u32 > layer_count {
        bail!("--stages must not exceed model layer count {layer_count}");
    }
    let ranges = partition_layers(layer_count, stages);
    let mut stage_tensors: BTreeMap<usize, Vec<&TensorInfo>> = BTreeMap::new();
    for (stage_index, (layer_start, layer_end)) in ranges.iter().copied().enumerate() {
        let tensors_for_stage = tensors
            .iter()
            .filter(|tensor| tensor_in_stage(tensor, stage_index, stages, layer_start, layer_end))
            .collect();
        stage_tensors.insert(stage_index, tensors_for_stage);
    }

    Ok(PlanOutput {
        schema_version: 1,
        stage_count: stages,
        layer_count,
        stages: ranges
            .into_iter()
            .enumerate()
            .map(|(stage_index, (layer_start, layer_end))| {
                let tensors = stage_tensors.remove(&stage_index).unwrap_or_default();
                StagePlan {
                    stage_index,
                    layer_start,
                    layer_end,
                    includes_embeddings: stage_index == 0,
                    includes_output: stage_index + 1 == stages,
                    tensor_count: tensors.len(),
                    tensor_bytes: tensors.iter().map(|tensor| tensor.byte_size).sum(),
                }
            })
            .collect(),
    })
}

fn write_stage_artifact(source: &ModelSource, stage: &StagePlan, out: &Path) -> Result<()> {
    create_parent_dir(out)?;

    if source.infos.len() == 1 {
        write_single_source_stage_artifact(&source.infos[0], stage, out)
    } else {
        write_sharded_stage_artifact(source, stage, out)
    }
}

fn write_package_artifact(
    source: &ModelSource,
    tensors: &[TensorInfo],
    spec: PackageArtifactSpec,
    out_dir: &Path,
    artifact_hook: &ArtifactHook,
) -> Result<PackageArtifact> {
    let stage = stage_plan_from_tensors(
        spec.stage_index as usize,
        spec.layer_start,
        spec.layer_end,
        spec.includes_embeddings,
        spec.includes_output,
        tensors,
    );
    let path = out_dir.join(&spec.relative_path);
    write_stage_artifact(source, &stage, &path)?;
    let metadata = fs::metadata(&path)
        .with_context(|| format!("read artifact metadata {}", path.display()))?;
    let artifact = PackageArtifact {
        path: spec.relative_path.display().to_string(),
        tensor_count: stage.tensor_count,
        tensor_bytes: stage.tensor_bytes,
        artifact_bytes: metadata.len(),
        sha256: file_sha256(&path)?,
    };
    run_artifact_hook(artifact_hook, &path, &artifact.path)?;
    Ok(artifact)
}

fn copy_projector_artifact(
    projector: &Path,
    index: usize,
    out_dir: &Path,
    artifact_hook: &ArtifactHook,
) -> Result<PackageProjector> {
    if !projector.is_file() {
        bail!("projector is not a file: {}", projector.display());
    }
    let info = ModelInfo::open(projector)
        .with_context(|| format!("open multimodal projector GGUF {}", projector.display()))?;
    let tensors = info
        .tensors()
        .with_context(|| format!("read multimodal projector tensors {}", projector.display()))?;
    let file_name = projector
        .file_name()
        .and_then(|name| name.to_str())
        .filter(|name| !name.trim().is_empty())
        .map(ToString::to_string)
        .unwrap_or_else(|| format!("mmproj-{index:03}.gguf"));
    let relative_path = PathBuf::from("projectors").join(file_name);
    let output_path = out_dir.join(&relative_path);
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("create projector directory {}", parent.display()))?;
    }
    fs::copy(projector, &output_path).with_context(|| {
        format!(
            "copy multimodal projector {} to {}",
            projector.display(),
            output_path.display()
        )
    })?;
    let metadata = fs::metadata(&output_path)
        .with_context(|| format!("read projector metadata {}", output_path.display()))?;

    let package_projector = PackageProjector {
        kind: "mmproj".to_string(),
        path: relative_path.to_string_lossy().replace('\\', "/"),
        tensor_count: tensors.len(),
        tensor_bytes: tensors.iter().map(|tensor| tensor.byte_size).sum(),
        artifact_bytes: metadata.len(),
        sha256: file_sha256(&output_path)?,
    };
    run_artifact_hook(artifact_hook, &output_path, &package_projector.path)?;
    Ok(package_projector)
}

fn run_artifact_hook(
    artifact_hook: &ArtifactHook,
    absolute_path: &Path,
    relative_path: &str,
) -> Result<()> {
    let Some(command) = &artifact_hook.command else {
        return Ok(());
    };
    let status = ProcessCommand::new(command)
        .env("SKIPPY_PACKAGE_ARTIFACT_PATH", absolute_path)
        .env("SKIPPY_PACKAGE_ARTIFACT_RELATIVE_PATH", relative_path)
        .status()
        .with_context(|| format!("run artifact hook {}", command.display()))?;
    if !status.success() {
        bail!(
            "artifact hook {} failed for {} with status {status}",
            command.display(),
            relative_path
        );
    }
    Ok(())
}

fn build_manifest(
    model: &Path,
    layer_count: u32,
    written: Vec<(StagePlan, PathBuf)>,
) -> Result<SliceManifest> {
    let mut stages = Vec::new();
    for (stage, path) in written {
        let metadata = fs::metadata(&path)
            .with_context(|| format!("read artifact metadata {}", path.display()))?;
        stages.push(SliceManifestStage {
            stage_index: stage.stage_index,
            layer_start: stage.layer_start,
            layer_end: stage.layer_end,
            includes_embeddings: stage.includes_embeddings,
            includes_output: stage.includes_output,
            path: path.display().to_string(),
            tensor_count: stage.tensor_count,
            tensor_bytes: stage.tensor_bytes,
            artifact_bytes: metadata.len(),
            sha256: file_sha256(&path)?,
        });
    }

    Ok(SliceManifest {
        schema_version: 1,
        source_model: model.display().to_string(),
        source_sha256: file_sha256(model)?,
        stage_count: stages.len(),
        layer_count,
        stages,
    })
}

fn write_json_file<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    create_parent_dir(path)?;
    let json = serde_json::to_vec_pretty(value)?;
    let mut file = File::create(path).with_context(|| format!("create {}", path.display()))?;
    file.write_all(&json)
        .with_context(|| format!("write {}", path.display()))?;
    file.write_all(b"\n")
        .with_context(|| format!("write {}", path.display()))?;
    Ok(())
}

fn create_parent_dir(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .with_context(|| format!("create output directory {}", parent.display()))?;
        }
    }
    Ok(())
}

impl ModelSource {
    fn open(path: &Path) -> Result<Self> {
        let paths = resolve_gguf_shard_paths(path)?;
        let mut infos = Vec::with_capacity(paths.len());
        let mut tensors = Vec::new();
        for path in &paths {
            let info = ModelInfo::open(path)
                .with_context(|| format!("open GGUF metadata {}", path.display()))?;
            tensors.extend(
                info.tensors()
                    .with_context(|| format!("read GGUF tensors {}", path.display()))?,
            );
            infos.push(info);
        }
        Ok(Self {
            paths,
            infos,
            tensors,
        })
    }
}

fn resolve_gguf_shard_paths(path: &Path) -> Result<Vec<PathBuf>> {
    let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
        return Ok(vec![path.to_path_buf()]);
    };
    let Some(shard) = split_gguf_shard_info(file_name) else {
        return Ok(vec![path.to_path_buf()]);
    };
    let total = shard
        .total
        .parse::<usize>()
        .with_context(|| format!("parse GGUF shard total from {file_name}"))?;
    if total <= 1 {
        return Ok(vec![path.to_path_buf()]);
    }

    let parent = path.parent().unwrap_or_else(|| Path::new(""));
    let mut paths = Vec::with_capacity(total);
    for part in 1..=total {
        let shard_name = format!("{}-{part:05}-of-{}.gguf", shard.prefix, shard.total);
        let shard_path = parent.join(shard_name);
        if !shard_path.exists() {
            bail!(
                "split GGUF shard {} is missing sibling {}",
                path.display(),
                shard_path.display()
            );
        }
        paths.push(shard_path);
    }
    Ok(paths)
}

fn local_artifact_files(model_path: &Path, primary_file: &str) -> Result<Vec<ModelArtifactFile>> {
    let shard_paths = resolve_gguf_shard_paths(model_path)?;
    if shard_paths.len() <= 1 {
        return Ok(vec![ModelArtifactFile::new(primary_file.to_string())]);
    }

    let primary_path = Path::new(primary_file);
    let primary_parent = primary_path.parent();
    let files = shard_paths
        .into_iter()
        .map(|path| {
            let file_name = path
                .file_name()
                .and_then(|name| name.to_str())
                .context("split GGUF shard path has no file name")?;
            let relative = primary_parent
                .map(|parent| parent.join(file_name))
                .unwrap_or_else(|| PathBuf::from(file_name));
            Ok(ModelArtifactFile::new(
                relative.to_string_lossy().replace('\\', "/"),
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(files)
}

fn write_single_source_stage_artifact(
    info: &ModelInfo,
    stage: &StagePlan,
    out: &Path,
) -> Result<()> {
    let mut plan = info.create_slice_plan()?;
    plan.add_layer_range(
        stage.stage_index as u32,
        stage.layer_start,
        stage.layer_end,
        stage.includes_embeddings,
        stage.includes_output,
    )?;
    info.write_slice_gguf(&plan, stage.stage_index as u32, out)
        .with_context(|| format!("write GGUF slice {}", out.display()))
}

fn write_sharded_stage_artifact(source: &ModelSource, stage: &StagePlan, out: &Path) -> Result<()> {
    let parent = out.parent().unwrap_or_else(|| Path::new("."));
    let stem = out
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("stage");
    let pid = std::process::id();
    let scratch = parent.join(format!(".{stem}.shard-parts-{pid}"));
    if scratch.exists() {
        fs::remove_dir_all(&scratch)
            .with_context(|| format!("remove stale shard scratch {}", scratch.display()))?;
    }
    fs::create_dir_all(&scratch)
        .with_context(|| format!("create shard scratch {}", scratch.display()))?;

    let result = (|| {
        let mut parts = Vec::with_capacity(source.infos.len());
        for (index, info) in source.infos.iter().enumerate() {
            let part_path = scratch.join(format!("part-{index:05}.gguf"));
            write_single_source_stage_artifact(info, stage, &part_path).with_context(|| {
                format!(
                    "write shard-local GGUF slice from {}",
                    source.paths[index].display()
                )
            })?;
            parts.push(part_path);
        }
        write_gguf_from_parts(&parts, out)
            .with_context(|| format!("merge split-GGUF shard slices into {}", out.display()))
    })();

    let cleanup = fs::remove_dir_all(&scratch)
        .with_context(|| format!("remove shard scratch {}", scratch.display()));
    result.and(cleanup)
}

const MAX_GGUF_STRING_BYTES: u64 = 1_000_000;
const MAX_GGUF_ARRAY_ELEMENTS: u64 = 1_000_000;
const MAX_GGUF_ARRAY_DEPTH: usize = 64;
const MAX_GGUF_HEADER_KV_COUNT: u64 = 1_000_000;
const MAX_GGUF_TENSOR_COUNT: u64 = 1_000_000;

fn activation_width(model_path: &Path) -> Result<u32> {
    let mut file = File::open(model_path)
        .with_context(|| format!("open GGUF metadata source {}", model_path.display()))?;
    let mut magic = [0u8; 4];
    file.read_exact(&mut magic)
        .with_context(|| format!("read GGUF magic from {}", model_path.display()))?;
    anyhow::ensure!(
        &magic == b"GGUF",
        "not a GGUF file: {}",
        model_path.display()
    );

    let version = read_gguf_u32(&mut file)?;
    anyhow::ensure!(
        version >= 2,
        "unsupported GGUF version {version} in {}",
        model_path.display()
    );
    let _tensor_count = read_gguf_header_count(&mut file, MAX_GGUF_TENSOR_COUNT, "tensor")?;
    let kv_count = read_gguf_header_count(&mut file, MAX_GGUF_HEADER_KV_COUNT, "metadata")?;

    let mut architecture = None;
    let mut embedding_lengths = BTreeMap::<String, u32>::new();
    for _ in 0..kv_count {
        let key = read_gguf_string(&mut file)?;
        let value_type = GgufValueType::from_u32(read_gguf_u32(&mut file)?)?;
        if key == "general.architecture" {
            architecture = read_gguf_string_value(&mut file, value_type)?;
        } else if let Some(arch) = key.strip_suffix(".embedding_length") {
            if let Some(value) = read_gguf_u32_value(&mut file, value_type)? {
                embedding_lengths.insert(arch.to_string(), value);
            }
        } else {
            skip_gguf_value(&mut file, value_type)?;
        }
    }

    let architecture = architecture.with_context(|| {
        format!(
            "GGUF metadata for {} does not contain general.architecture",
            model_path.display()
        )
    })?;
    let width = embedding_lengths.remove(&architecture).with_context(|| {
        format!(
            "GGUF metadata for {} does not contain {}.embedding_length",
            model_path.display(),
            architecture
        )
    })?;
    anyhow::ensure!(
        width > 0,
        "GGUF metadata for {} has invalid {}.embedding_length 0",
        model_path.display(),
        architecture
    );
    Ok(width)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GgufValueType {
    Uint8,
    Int8,
    Uint16,
    Int16,
    Uint32,
    Int32,
    Float32,
    Bool,
    String,
    Array,
    Uint64,
    Int64,
    Float64,
}

impl GgufValueType {
    fn from_u32(value: u32) -> Result<Self> {
        Ok(match value {
            0 => Self::Uint8,
            1 => Self::Int8,
            2 => Self::Uint16,
            3 => Self::Int16,
            4 => Self::Uint32,
            5 => Self::Int32,
            6 => Self::Float32,
            7 => Self::Bool,
            8 => Self::String,
            9 => Self::Array,
            10 => Self::Uint64,
            11 => Self::Int64,
            12 => Self::Float64,
            other => bail!("unsupported GGUF metadata value type {other}"),
        })
    }

    fn fixed_width(self) -> Option<u64> {
        match self {
            Self::Uint8 | Self::Int8 | Self::Bool => Some(1),
            Self::Uint16 | Self::Int16 => Some(2),
            Self::Uint32 | Self::Int32 | Self::Float32 => Some(4),
            Self::Uint64 | Self::Int64 | Self::Float64 => Some(8),
            Self::String | Self::Array => None,
        }
    }
}

fn read_gguf_u32(reader: &mut impl Read) -> Result<u32> {
    let mut bytes = [0u8; 4];
    reader.read_exact(&mut bytes).context("read GGUF u32")?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_gguf_i32(reader: &mut impl Read) -> Result<i32> {
    let mut bytes = [0u8; 4];
    reader.read_exact(&mut bytes).context("read GGUF i32")?;
    Ok(i32::from_le_bytes(bytes))
}

fn read_gguf_u64(reader: &mut impl Read) -> Result<u64> {
    let mut bytes = [0u8; 8];
    reader.read_exact(&mut bytes).context("read GGUF u64")?;
    Ok(u64::from_le_bytes(bytes))
}

fn read_gguf_i64(reader: &mut impl Read) -> Result<i64> {
    let mut bytes = [0u8; 8];
    reader.read_exact(&mut bytes).context("read GGUF i64")?;
    Ok(i64::from_le_bytes(bytes))
}

fn read_gguf_u16(reader: &mut impl Read) -> Result<u16> {
    let mut bytes = [0u8; 2];
    reader.read_exact(&mut bytes).context("read GGUF u16")?;
    Ok(u16::from_le_bytes(bytes))
}

fn read_gguf_u8(reader: &mut impl Read) -> Result<u8> {
    let mut bytes = [0u8; 1];
    reader.read_exact(&mut bytes).context("read GGUF u8")?;
    Ok(bytes[0])
}

fn read_gguf_header_count(reader: &mut impl Read, max: u64, label: &str) -> Result<u64> {
    let count = read_gguf_i64(reader)?;
    ensure!(count >= 0, "GGUF {label} count is negative: {count}");
    let count = u64::try_from(count).context("GGUF header count does not fit u64")?;
    ensure!(
        count <= max,
        "GGUF {label} count {count} exceeds safety limit {max}"
    );
    Ok(count)
}

fn read_gguf_string(reader: &mut impl Read) -> Result<String> {
    let len = read_gguf_u64(reader)?;
    ensure!(
        len <= MAX_GGUF_STRING_BYTES,
        "GGUF string length {len} exceeds safety limit {MAX_GGUF_STRING_BYTES}"
    );
    let len = usize::try_from(len).context("GGUF string length does not fit usize")?;
    let mut bytes = vec![0u8; len];
    reader
        .read_exact(&mut bytes)
        .context("read GGUF string bytes")?;
    String::from_utf8(bytes).context("GGUF string is not valid UTF-8")
}

fn read_gguf_string_value(
    reader: &mut (impl Read + Seek),
    value_type: GgufValueType,
) -> Result<Option<String>> {
    if value_type == GgufValueType::String {
        return Ok(Some(read_gguf_string(reader)?));
    }
    skip_gguf_value(reader, value_type)?;
    Ok(None)
}

fn read_gguf_u32_value(
    reader: &mut (impl Read + Seek),
    value_type: GgufValueType,
) -> Result<Option<u32>> {
    Ok(match value_type {
        GgufValueType::Uint32 => Some(read_gguf_u32(reader)?),
        GgufValueType::Int32 => {
            let value = read_gguf_i32(reader)?;
            Some(u32::try_from(value).context("GGUF embedding_length is negative")?)
        }
        GgufValueType::Uint16 => Some(u32::from(read_gguf_u16(reader)?)),
        GgufValueType::Uint8 => Some(u32::from(read_gguf_u8(reader)?)),
        _ => {
            skip_gguf_value(reader, value_type)?;
            None
        }
    })
}

fn skip_gguf_value(reader: &mut (impl Read + Seek), value_type: GgufValueType) -> Result<()> {
    skip_gguf_value_with_depth(reader, value_type, 0)
}

fn skip_gguf_value_with_depth(
    reader: &mut (impl Read + Seek),
    value_type: GgufValueType,
    depth: usize,
) -> Result<()> {
    ensure!(
        depth <= MAX_GGUF_ARRAY_DEPTH,
        "GGUF array nesting exceeds safety limit {MAX_GGUF_ARRAY_DEPTH}"
    );
    if let Some(width) = value_type.fixed_width() {
        skip_gguf_bytes(reader, width)
    } else if value_type == GgufValueType::String {
        let len = read_gguf_u64(reader)?;
        ensure!(
            len <= MAX_GGUF_STRING_BYTES,
            "GGUF string length {len} exceeds safety limit {MAX_GGUF_STRING_BYTES}"
        );
        skip_gguf_bytes(reader, len)
    } else {
        let item_type = GgufValueType::from_u32(read_gguf_u32(reader)?)?;
        let len = read_gguf_u64(reader)?;
        ensure!(
            len <= MAX_GGUF_ARRAY_ELEMENTS,
            "GGUF array length {len} exceeds safety limit {MAX_GGUF_ARRAY_ELEMENTS}"
        );
        if let Some(width) = item_type.fixed_width() {
            let bytes = width
                .checked_mul(len)
                .context("GGUF array byte size overflows u64")?;
            skip_gguf_bytes(reader, bytes)
        } else {
            for _ in 0..len {
                skip_gguf_value_with_depth(reader, item_type, depth + 1)?;
            }
            Ok(())
        }
    }
}

fn skip_gguf_bytes(reader: &mut impl Seek, len: u64) -> Result<()> {
    let offset = i64::try_from(len).context("GGUF value is too large to seek over")?;
    reader
        .seek(SeekFrom::Current(offset))
        .context("skip GGUF metadata value")?;
    Ok(())
}

fn layer_count(tensors: &[TensorInfo]) -> Result<u32> {
    tensors
        .iter()
        .filter_map(|tensor| tensor.layer_index)
        .max()
        .map(|max_layer| max_layer + 1)
        .context("model has no layer tensors")
}

fn stage_plan_from_tensors(
    stage_index: usize,
    layer_start: u32,
    layer_end: u32,
    includes_embeddings: bool,
    includes_output: bool,
    tensors: &[TensorInfo],
) -> StagePlan {
    let selected = tensors
        .iter()
        .filter(|tensor| {
            tensor_in_explicit_stage(
                tensor,
                layer_start,
                layer_end,
                includes_embeddings,
                includes_output,
            )
        })
        .collect::<Vec<_>>();
    StagePlan {
        stage_index,
        layer_start,
        layer_end,
        includes_embeddings,
        includes_output,
        tensor_count: selected.len(),
        tensor_bytes: selected.iter().map(|tensor| tensor.byte_size).sum(),
    }
}

fn tensor_in_stage(
    tensor: &TensorInfo,
    stage_index: usize,
    stages: usize,
    layer_start: u32,
    layer_end: u32,
) -> bool {
    tensor_in_explicit_stage(
        tensor,
        layer_start,
        layer_end,
        stage_index == 0,
        stage_index + 1 == stages,
    )
}

fn tensor_in_explicit_stage(
    tensor: &TensorInfo,
    layer_start: u32,
    layer_end: u32,
    includes_embeddings: bool,
    includes_output: bool,
) -> bool {
    matches!(
        tensor.layer_index,
        Some(layer) if layer >= layer_start && layer < layer_end
    ) || (includes_embeddings && tensor.role == TensorRole::Embedding)
        || (includes_output && matches!(tensor.role, TensorRole::FinalNorm | TensorRole::Output))
        || matches!(
            tensor.role,
            TensorRole::Metadata | TensorRole::Tokenizer | TensorRole::Unknown
        )
}

fn parse_layer_range(layers: &str) -> Result<(u32, u32)> {
    let Some((start, end)) = layers.split_once("..") else {
        bail!("--layers must use START..END syntax");
    };
    let start = start.parse::<u32>().context("parse layer range start")?;
    let end = end.parse::<u32>().context("parse layer range end")?;
    if start >= end {
        bail!("layer range start must be less than end");
    }
    Ok((start, end))
}

fn partition_layers(layer_count: u32, stages: usize) -> Vec<(u32, u32)> {
    let base = layer_count / stages as u32;
    let extra = layer_count % stages as u32;
    let mut start = 0;
    (0..stages)
        .map(|stage_index| {
            let width = base + u32::from((stage_index as u32) < extra);
            let end = start + width;
            let range = (start, end);
            start = end;
            range
        })
        .collect()
}

fn tensor_output(tensor: TensorInfo) -> TensorOutput {
    TensorOutput {
        name: tensor.name,
        layer_index: tensor.layer_index,
        role: role_name(tensor.role).to_string(),
        ggml_type: tensor.ggml_type,
        byte_size: tensor.byte_size,
    }
}

fn role_name(role: TensorRole) -> &'static str {
    match role {
        TensorRole::Unknown => "unknown",
        TensorRole::Metadata => "metadata",
        TensorRole::Tokenizer => "tokenizer",
        TensorRole::Embedding => "embedding",
        TensorRole::Layer => "layer",
        TensorRole::FinalNorm => "final_norm",
        TensorRole::Output => "output",
    }
}

fn file_sha256(path: &Path) -> Result<String> {
    if let Some(hash) = file_sha256_openssl(path)? {
        return Ok(hash);
    }

    let mut file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .with_context(|| format!("read {}", path.display()))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hex_lower(&hasher.finalize()))
}

fn file_sha256_openssl(path: &Path) -> Result<Option<String>> {
    let output = match ProcessCommand::new("openssl")
        .arg("dgst")
        .arg("-sha256")
        .arg("-r")
        .arg(path)
        .output()
    {
        Ok(output) => output,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error).with_context(|| format!("hash {}", path.display())),
    };
    if !output.status.success() {
        return Ok(None);
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let Some(hash) = stdout.split_whitespace().next() else {
        return Ok(None);
    };
    if hash.len() == 64 && hash.chars().all(|ch| ch.is_ascii_hexdigit()) {
        Ok(Some(hash.to_ascii_lowercase()))
    } else {
        Ok(None)
    }
}

fn hex_lower(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}

#[cfg(test)]
mod tests {
    use super::{
        activation_width, local_artifact_files, model_distribution_id, resolve_gguf_shard_paths,
        resolve_local_package_input, ExplicitSourceIdentity,
    };
    use std::path::{Path, PathBuf};

    #[test]
    fn model_distribution_id_uses_shared_gguf_stem_normalization() {
        assert_eq!(
            model_distribution_id(Path::new("UD-IQ2_M/GLM-5.1-UD-IQ2_M-00001-of-00006.gguf")),
            Some("GLM-5.1-UD-IQ2_M".to_string())
        );
        assert_eq!(
            model_distribution_id(Path::new("Qwen3-8B-Q4_K_M.gguf")),
            Some("Qwen3-8B-Q4_K_M".to_string())
        );
        assert_eq!(model_distribution_id(Path::new("README.md")), None);
    }

    #[test]
    fn local_package_input_requires_explicit_identity() {
        let error =
            resolve_local_package_input("model.gguf".into(), ExplicitSourceIdentity::default())
                .unwrap_err();

        assert!(error.to_string().contains("requires --model-id"));
    }

    #[test]
    fn local_package_input_uses_explicit_coordinate_identity() {
        let input = resolve_local_package_input(
            "local.gguf".into(),
            ExplicitSourceIdentity {
                model_id: Some("org/repo:Q4_K_M".to_string()),
                source_repo: None,
                source_revision: Some("abc123".to_string()),
                source_file: Some("Qwen3-8B-Q4_K_M.gguf".to_string()),
            },
        )
        .unwrap();

        assert_eq!(input.model_id, "org/repo:Q4_K_M");
        assert_eq!(input.source_identity.repo.as_deref(), Some("org/repo"));
        assert_eq!(input.source_identity.revision.as_deref(), Some("abc123"));
        assert_eq!(
            input.source_identity.canonical_ref.as_deref(),
            Some("org/repo@abc123/Qwen3-8B-Q4_K_M.gguf")
        );
        assert_eq!(
            input.source_identity.distribution_id.as_deref(),
            Some("Qwen3-8B-Q4_K_M")
        );
    }

    #[test]
    fn split_gguf_path_resolves_sibling_shards() {
        let dir = unique_test_dir("split-gguf-path");
        std::fs::create_dir_all(&dir).unwrap();
        for part in 1..=3 {
            std::fs::write(
                dir.join(format!("MiniMax-M2.7-UD-Q2_K_XL-{part:05}-of-00003.gguf")),
                b"",
            )
            .unwrap();
        }

        let input = dir.join("MiniMax-M2.7-UD-Q2_K_XL-00002-of-00003.gguf");
        let paths = resolve_gguf_shard_paths(&input).unwrap();
        let names = paths
            .iter()
            .map(|path| path.file_name().unwrap().to_string_lossy().to_string())
            .collect::<Vec<_>>();

        assert_eq!(
            names,
            vec![
                "MiniMax-M2.7-UD-Q2_K_XL-00001-of-00003.gguf",
                "MiniMax-M2.7-UD-Q2_K_XL-00002-of-00003.gguf",
                "MiniMax-M2.7-UD-Q2_K_XL-00003-of-00003.gguf",
            ]
        );
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn local_artifact_files_preserve_shard_subdirectory() {
        let dir = unique_test_dir("split-gguf-files");
        let shard_dir = dir.join("UD-Q2_K_XL");
        std::fs::create_dir_all(&shard_dir).unwrap();
        for part in 1..=2 {
            std::fs::write(
                shard_dir.join(format!("MiniMax-M2.7-UD-Q2_K_XL-{part:05}-of-00002.gguf")),
                b"",
            )
            .unwrap();
        }

        let input = shard_dir.join("MiniMax-M2.7-UD-Q2_K_XL-00001-of-00002.gguf");
        let files = local_artifact_files(
            &input,
            "UD-Q2_K_XL/MiniMax-M2.7-UD-Q2_K_XL-00001-of-00002.gguf",
        )
        .unwrap()
        .into_iter()
        .map(|file| file.path)
        .collect::<Vec<_>>();

        assert_eq!(
            files,
            vec![
                "UD-Q2_K_XL/MiniMax-M2.7-UD-Q2_K_XL-00001-of-00002.gguf",
                "UD-Q2_K_XL/MiniMax-M2.7-UD-Q2_K_XL-00002-of-00002.gguf",
            ]
        );
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn activation_width_reads_arch_embedding_length_from_gguf_metadata() {
        let dir = unique_test_dir("activation-width");
        std::fs::create_dir_all(&dir).unwrap();
        let model = dir.join("model.gguf");
        let mut bytes = gguf_header(2);
        push_string_kv(&mut bytes, "general.architecture", "qwen2");
        push_u32_kv(&mut bytes, "qwen2.embedding_length", 3584);
        std::fs::write(&model, bytes).unwrap();

        assert_eq!(activation_width(&model).unwrap(), 3584);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn activation_width_accepts_smaller_and_signed_integer_metadata() {
        let dir = unique_test_dir("activation-width-int-forms");
        std::fs::create_dir_all(&dir).unwrap();
        let u16_model = dir.join("u16.gguf");
        let i32_model = dir.join("i32.gguf");

        let mut u16_bytes = gguf_header(2);
        push_string_kv(&mut u16_bytes, "general.architecture", "tiny");
        push_u16_kv(&mut u16_bytes, "tiny.embedding_length", 1024);
        std::fs::write(&u16_model, u16_bytes).unwrap();

        let mut i32_bytes = gguf_header(2);
        push_string_kv(&mut i32_bytes, "general.architecture", "qwen2");
        push_i32_kv(&mut i32_bytes, "qwen2.embedding_length", 4096);
        std::fs::write(&i32_model, i32_bytes).unwrap();

        assert_eq!(activation_width(&u16_model).unwrap(), 1024);
        assert_eq!(activation_width(&i32_model).unwrap(), 4096);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn activation_width_rejects_zero_embedding_length() {
        let dir = unique_test_dir("activation-width-zero");
        std::fs::create_dir_all(&dir).unwrap();
        let model = dir.join("model.gguf");
        let mut bytes = gguf_header(2);
        push_string_kv(&mut bytes, "general.architecture", "qwen2");
        push_u32_kv(&mut bytes, "qwen2.embedding_length", 0);
        std::fs::write(&model, bytes).unwrap();

        let error = activation_width(&model).unwrap_err().to_string();
        assert!(error.contains("embedding_length 0"), "{error}");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn activation_width_rejects_oversized_metadata_string() {
        let dir = unique_test_dir("activation-width-big-string");
        std::fs::create_dir_all(&dir).unwrap();
        let model = dir.join("model.gguf");
        let mut bytes = gguf_header(3);
        push_string_kv(&mut bytes, "general.architecture", "qwen2");
        push_oversized_string_kv(&mut bytes, "junk");
        push_u32_kv(&mut bytes, "qwen2.embedding_length", 3584);
        std::fs::write(&model, bytes).unwrap();

        let error = activation_width(&model).unwrap_err().to_string();
        assert!(error.contains("exceeds safety limit"), "{error}");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn activation_width_rejects_too_deep_metadata_arrays() {
        let dir = unique_test_dir("activation-width-deep-array");
        std::fs::create_dir_all(&dir).unwrap();
        let model = dir.join("model.gguf");
        let mut bytes = gguf_header(3);
        push_string_kv(&mut bytes, "general.architecture", "qwen2");
        push_deep_array_kv(&mut bytes, "junk", 65);
        push_u32_kv(&mut bytes, "qwen2.embedding_length", 3584);
        std::fs::write(&model, bytes).unwrap();

        let error = activation_width(&model).unwrap_err().to_string();
        assert!(error.contains("array nesting exceeds"), "{error}");
        std::fs::remove_dir_all(dir).unwrap();
    }

    fn gguf_header(kv_count: u64) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&2_u32.to_le_bytes());
        bytes.extend_from_slice(&0_i64.to_le_bytes());
        bytes.extend_from_slice(&(kv_count as i64).to_le_bytes());
        bytes
    }

    fn push_gguf_string(bytes: &mut Vec<u8>, value: &str) {
        bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
        bytes.extend_from_slice(value.as_bytes());
    }

    fn push_string_kv(bytes: &mut Vec<u8>, key: &str, value: &str) {
        push_gguf_string(bytes, key);
        bytes.extend_from_slice(&8_u32.to_le_bytes());
        push_gguf_string(bytes, value);
    }

    fn push_u32_kv(bytes: &mut Vec<u8>, key: &str, value: u32) {
        push_gguf_string(bytes, key);
        bytes.extend_from_slice(&4_u32.to_le_bytes());
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn push_i32_kv(bytes: &mut Vec<u8>, key: &str, value: i32) {
        push_gguf_string(bytes, key);
        bytes.extend_from_slice(&5_u32.to_le_bytes());
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn push_u16_kv(bytes: &mut Vec<u8>, key: &str, value: u16) {
        push_gguf_string(bytes, key);
        bytes.extend_from_slice(&2_u32.to_le_bytes());
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn push_oversized_string_kv(bytes: &mut Vec<u8>, key: &str) {
        push_gguf_string(bytes, key);
        bytes.extend_from_slice(&8_u32.to_le_bytes());
        bytes.extend_from_slice(&(super::MAX_GGUF_STRING_BYTES + 1).to_le_bytes());
    }

    fn push_deep_array_kv(bytes: &mut Vec<u8>, key: &str, depth: usize) {
        push_gguf_string(bytes, key);
        bytes.extend_from_slice(&9_u32.to_le_bytes());
        for _ in 0..depth {
            bytes.extend_from_slice(&9_u32.to_le_bytes());
            bytes.extend_from_slice(&1_u64.to_le_bytes());
        }
        bytes.extend_from_slice(&4_u32.to_le_bytes());
        bytes.extend_from_slice(&0_u64.to_le_bytes());
    }

    fn unique_test_dir(name: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "skippy-model-package-{name}-{}-{nanos}",
            std::process::id()
        ))
    }
}
