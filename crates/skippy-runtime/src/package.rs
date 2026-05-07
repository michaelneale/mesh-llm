use std::{
    collections::BTreeMap,
    fs,
    io::Read,
    path::{Path, PathBuf},
    time::UNIX_EPOCH,
};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::write_gguf_from_parts;

#[derive(Debug, Clone)]
pub struct PackageStageRequest {
    pub model_id: String,
    pub topology_id: String,
    pub package_ref: String,
    pub stage_id: String,
    pub layer_start: u32,
    pub layer_end: u32,
    pub include_embeddings: bool,
    pub include_output: bool,
}

#[derive(Debug, Clone)]
pub struct MaterializedPackage {
    pub output_path: PathBuf,
    pub manifest_sha256: String,
    pub selected_parts: Vec<PackagePart>,
}

#[derive(Debug, Clone)]
pub struct SelectedPackageParts {
    pub package_dir: PathBuf,
    pub manifest_sha256: String,
    pub selected_parts: Vec<PackagePart>,
    pub absolute_paths: Vec<PathBuf>,
    pub projector_paths: Vec<PathBuf>,
    pub integrity: PackageIntegrityReport,
}

#[derive(Debug, Clone)]
pub struct LayerPackageInfo {
    pub package_dir: PathBuf,
    pub manifest_sha256: String,
    pub model_id: String,
    pub source_model_path: String,
    pub source_model_sha256: String,
    pub source_model_bytes: Option<u64>,
    pub layer_count: u32,
    pub activation_width: Option<u32>,
    pub projectors: Vec<PackageProjectorInfo>,
    pub layers: Vec<LayerPackageLayerInfo>,
}

#[derive(Debug, Clone)]
pub struct PackageProjectorInfo {
    pub kind: String,
    pub path: PathBuf,
    pub artifact_bytes: u64,
}

#[derive(Debug, Clone)]
pub struct LayerPackageLayerInfo {
    pub layer_index: u32,
    pub tensor_count: usize,
    pub tensor_bytes: u64,
    pub artifact_bytes: u64,
}

#[derive(Debug, Clone)]
pub struct PackagePart {
    pub role: String,
    pub layer_index: Option<u32>,
    pub path: PathBuf,
    pub sha256: String,
    pub artifact_bytes: u64,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct PackageIntegrityOptions {
    verify_sha256: bool,
    cache_dir: Option<PathBuf>,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct PackageIntegrityReport {
    pub manifest_sha256: String,
    pub artifacts: usize,
    pub verified_artifacts: usize,
    pub cached_artifacts: usize,
}

impl PackageIntegrityOptions {
    pub fn manifest_only() -> Self {
        Self {
            verify_sha256: false,
            cache_dir: None,
        }
    }

    pub fn verify_without_cache() -> Self {
        Self {
            verify_sha256: true,
            cache_dir: None,
        }
    }

    pub fn verify_with_cache(cache_dir: impl AsRef<Path>) -> Self {
        Self {
            verify_sha256: true,
            cache_dir: Some(cache_dir.as_ref().to_path_buf()),
        }
    }

    fn from_env() -> Self {
        let mut options = if env_flag("SKIPPY_VERIFY_PACKAGE_SHA") {
            Self::verify_without_cache()
        } else {
            Self::manifest_only()
        };
        if let Some(cache_dir) = std::env::var_os("SKIPPY_PACKAGE_VERIFY_CACHE_DIR") {
            options.cache_dir = Some(PathBuf::from(cache_dir));
        }
        options
    }
}

#[derive(Debug, Deserialize)]
struct PackageManifest {
    schema_version: u32,
    model_id: String,
    source_model: PackageSourceModel,
    format: String,
    layer_count: u32,
    #[serde(default)]
    activation_width: Option<u32>,
    shared: PackageShared,
    #[serde(default)]
    projectors: Vec<PackageProjector>,
    layers: Vec<PackageLayer>,
    skippy_abi_version: String,
}

#[derive(Debug, Deserialize)]
struct PackageSourceModel {
    path: String,
    sha256: String,
    repo: Option<String>,
    revision: Option<String>,
    primary_file: Option<String>,
    canonical_ref: Option<String>,
    distribution_id: Option<String>,
    #[serde(default)]
    files: Vec<PackageSourceFile>,
}

#[derive(Debug, Deserialize)]
struct PackageSourceFile {
    path: String,
    size_bytes: Option<u64>,
    sha256: Option<String>,
}

#[derive(Debug, Deserialize)]
struct PackageShared {
    metadata: PackageArtifact,
    embeddings: PackageArtifact,
    output: PackageArtifact,
}

#[derive(Debug, Deserialize)]
struct PackageArtifact {
    path: String,
    tensor_count: usize,
    tensor_bytes: u64,
    artifact_bytes: u64,
    sha256: String,
}

#[derive(Debug, Deserialize)]
struct PackageProjector {
    kind: String,
    path: String,
    tensor_count: usize,
    tensor_bytes: u64,
    artifact_bytes: u64,
    sha256: String,
}

#[derive(Debug, Deserialize)]
struct PackageLayer {
    layer_index: u32,
    path: String,
    tensor_count: usize,
    tensor_bytes: u64,
    artifact_bytes: u64,
    sha256: String,
}

pub fn materialize_layer_package(request: &PackageStageRequest) -> Result<PathBuf> {
    Ok(materialize_layer_package_details(request)?.output_path)
}

pub fn materialize_layer_package_details(
    request: &PackageStageRequest,
) -> Result<MaterializedPackage> {
    let selection = select_layer_package_parts(request)?;
    let output = materialized_path(
        request,
        &selection.package_dir,
        &selection.manifest_sha256,
        &selection.selected_parts,
    );
    if output.is_file()
        && fs::metadata(&output)
            .with_context(|| format!("read materialized model {}", output.display()))?
            .len()
            > 0
        && !env_flag("SKIPPY_FORCE_MATERIALIZE")
    {
        return Ok(MaterializedPackage {
            output_path: output,
            manifest_sha256: selection.manifest_sha256,
            selected_parts: selection.selected_parts,
        });
    }

    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("create materialization directory {}", parent.display()))?;
    }
    write_gguf_from_parts(&selection.absolute_paths, &output).with_context(|| {
        format!(
            "materialize layer package {}",
            selection.package_dir.display()
        )
    })?;

    Ok(MaterializedPackage {
        output_path: output,
        manifest_sha256: selection.manifest_sha256,
        selected_parts: selection.selected_parts,
    })
}

pub fn select_layer_package_parts(request: &PackageStageRequest) -> Result<SelectedPackageParts> {
    select_layer_package_parts_with_integrity(request, &PackageIntegrityOptions::from_env())
}

pub fn select_layer_package_parts_with_integrity(
    request: &PackageStageRequest,
    integrity_options: &PackageIntegrityOptions,
) -> Result<SelectedPackageParts> {
    let package_dir = resolve_package_dir(&request.package_ref)?;
    let manifest_path = package_dir.join("model-package.json");
    let manifest_contents = fs::read(&manifest_path)
        .with_context(|| format!("read package manifest {}", manifest_path.display()))?;
    let manifest_sha256 = sha256_bytes(&manifest_contents);
    let manifest = load_manifest(&manifest_path, &manifest_contents)?;
    validate_manifest(&manifest, request)?;

    let layer_by_index = manifest
        .layers
        .iter()
        .map(|layer| {
            (
                layer.layer_index,
                PackageArtifact {
                    path: layer.path.clone(),
                    tensor_count: layer.tensor_count,
                    tensor_bytes: layer.tensor_bytes,
                    artifact_bytes: layer.artifact_bytes,
                    sha256: layer.sha256.clone(),
                },
            )
        })
        .collect::<BTreeMap<_, _>>();

    let mut parts = Vec::new();
    push_part(
        &mut parts,
        "metadata",
        None,
        &manifest.shared.metadata,
        &package_dir,
    )?;
    if request.include_embeddings {
        push_part(
            &mut parts,
            "embeddings",
            None,
            &manifest.shared.embeddings,
            &package_dir,
        )?;
    }
    for layer_index in request.layer_start..request.layer_end {
        let artifact = layer_by_index
            .get(&layer_index)
            .with_context(|| format!("package is missing layer {layer_index}"))?;
        push_part(
            &mut parts,
            "layer",
            Some(layer_index),
            artifact,
            &package_dir,
        )?;
    }
    if request.include_output {
        push_part(
            &mut parts,
            "output",
            None,
            &manifest.shared.output,
            &package_dir,
        )?;
    }

    let absolute_paths = parts
        .iter()
        .map(|part| package_dir.join(&part.path))
        .collect::<Vec<_>>();
    let projector_paths = manifest
        .projectors
        .iter()
        .map(|projector| projector_path(projector, &package_dir))
        .collect::<Result<Vec<_>>>()?;
    let integrity = verify_package_artifacts(
        &package_dir,
        &manifest_sha256,
        &parts,
        &manifest.projectors,
        integrity_options,
    )?;

    Ok(SelectedPackageParts {
        package_dir,
        manifest_sha256,
        selected_parts: parts,
        absolute_paths,
        projector_paths,
        integrity,
    })
}

pub fn verify_layer_package_integrity(
    request: &PackageStageRequest,
    integrity_options: &PackageIntegrityOptions,
) -> Result<PackageIntegrityReport> {
    Ok(select_layer_package_parts_with_integrity(request, integrity_options)?.integrity)
}

pub fn verify_layer_package_metadata_integrity(
    package_ref: &str,
    integrity_options: &PackageIntegrityOptions,
) -> Result<PackageIntegrityReport> {
    let package_dir = resolve_package_dir(package_ref)?;
    let manifest_path = package_dir.join("model-package.json");
    let manifest_contents = fs::read(&manifest_path)
        .with_context(|| format!("read package manifest {}", manifest_path.display()))?;
    let manifest_sha256 = sha256_bytes(&manifest_contents);
    let manifest = load_manifest(&manifest_path, &manifest_contents)?;
    validate_manifest_identity(&manifest)?;
    validate_layer_manifest(&manifest)?;

    let mut parts = Vec::new();
    push_part(
        &mut parts,
        "metadata",
        None,
        &manifest.shared.metadata,
        &package_dir,
    )?;
    verify_package_artifacts(
        &package_dir,
        &manifest_sha256,
        &parts,
        &[],
        integrity_options,
    )
}

pub fn inspect_layer_package(package_ref: &str) -> Result<LayerPackageInfo> {
    let package_dir = resolve_package_dir(package_ref)?;
    let manifest_path = package_dir.join("model-package.json");
    let manifest_contents = fs::read(&manifest_path)
        .with_context(|| format!("read package manifest {}", manifest_path.display()))?;
    let manifest_sha256 = sha256_bytes(&manifest_contents);
    let manifest = load_manifest(&manifest_path, &manifest_contents)?;
    validate_manifest_identity(&manifest)?;
    validate_layer_manifest(&manifest)?;
    let projectors = manifest
        .projectors
        .into_iter()
        .map(|projector| {
            let path = safe_relative_manifest_path(&projector.path)?;
            Ok(PackageProjectorInfo {
                kind: projector.kind,
                path: package_dir.join(path),
                artifact_bytes: projector.artifact_bytes,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let activation_width = match manifest.activation_width {
        Some(width) => Some(width),
        None => infer_activation_width_from_layers(&package_dir, &manifest.layers)?,
    };

    Ok(LayerPackageInfo {
        package_dir,
        manifest_sha256,
        model_id: manifest.model_id,
        source_model_path: manifest.source_model.path,
        source_model_sha256: manifest.source_model.sha256,
        source_model_bytes: (!manifest.source_model.files.is_empty())
            .then(|| {
                manifest
                    .source_model
                    .files
                    .iter()
                    .try_fold(0u64, |total, file| {
                        file.size_bytes
                            .map(|bytes| total.saturating_add(bytes))
                            .ok_or(())
                    })
                    .ok()
            })
            .flatten(),
        layer_count: manifest.layer_count,
        activation_width,
        projectors,
        layers: manifest
            .layers
            .into_iter()
            .map(|layer| LayerPackageLayerInfo {
                layer_index: layer.layer_index,
                tensor_count: layer.tensor_count,
                tensor_bytes: layer.tensor_bytes,
                artifact_bytes: layer.artifact_bytes,
            })
            .collect(),
    })
}

fn infer_activation_width_from_layers(
    package_dir: &Path,
    layers: &[PackageLayer],
) -> Result<Option<u32>> {
    let Some(layer) = layers.first() else {
        return Ok(None);
    };
    let layer_path = package_dir.join(safe_relative_manifest_path(&layer.path)?);
    let info = match crate::ModelInfo::open(&layer_path) {
        Ok(info) => info,
        Err(_) => return Ok(None),
    };
    let count = match info.tensor_count() {
        Ok(count) => count,
        Err(_) => return Ok(None),
    };
    for i in 0..count {
        let Ok(tensor) = info.tensor_at(i) else {
            continue;
        };
        if tensor.name.contains("attn_norm.weight") {
            let width = activation_width_from_tensor_count(
                &tensor.name,
                &layer_path,
                tensor.element_count,
            )?;
            return Ok(Some(width));
        }
    }
    Ok(None)
}

fn activation_width_from_tensor_count(
    name: &str,
    layer_path: &Path,
    element_count: u64,
) -> Result<u32> {
    u32::try_from(element_count).with_context(|| {
        format!(
            "activation width tensor {name} in {} has {element_count} elements, which exceeds u32::MAX",
            layer_path.display()
        )
    })
}

pub fn is_hf_package_ref(value: &str) -> bool {
    value.starts_with("hf://")
}

fn resolve_package_dir(package_ref: &str) -> Result<PathBuf> {
    if is_hf_package_ref(package_ref) {
        bail!(
            "hf:// package refs must be resolved to a local path before calling skippy-runtime. \
             Use the mesh-llm layer package resolver to download first. Got: {package_ref}"
        );
    }
    Ok(PathBuf::from(package_ref))
}

#[cfg(test)]
#[derive(Debug, PartialEq, Eq)]
struct HfPackageRef {
    repo_id: String,
    revision: Option<String>,
}

#[cfg(test)]
fn parse_hf_package_ref(value: &str) -> Result<HfPackageRef> {
    let Some(rest) = value.strip_prefix("hf://") else {
        bail!("HF package references must start with hf://");
    };
    if rest.is_empty() {
        bail!("HF package reference is missing a repo id");
    }

    let (repo_id, revision) = if let Some((repo_id, revision)) = rest.split_once('@') {
        (repo_id, Some(revision))
    } else if let Some(index) = rest.rfind(':') {
        (&rest[..index], Some(&rest[index + 1..]))
    } else {
        (rest, None)
    };

    if repo_id.split('/').count() != 2 || repo_id.contains(':') || repo_id.contains('@') {
        bail!("HF package repo id must look like namespace/repo");
    }
    if let Some(revision) = revision {
        if revision.is_empty() {
            bail!("HF package revision is empty");
        }
    }

    Ok(HfPackageRef {
        repo_id: repo_id.to_string(),
        revision: revision.map(ToString::to_string),
    })
}

fn load_manifest(path: &Path, contents: &[u8]) -> Result<PackageManifest> {
    serde_json::from_slice(contents)
        .with_context(|| format!("parse package manifest {}", path.display()))
}

fn validate_manifest(manifest: &PackageManifest, request: &PackageStageRequest) -> Result<()> {
    validate_manifest_identity(manifest)?;
    let layer_counts = validate_layer_manifest(manifest)?;
    if request.layer_start >= request.layer_end {
        bail!("stage layer_start must be less than layer_end");
    }
    if request.layer_end > manifest.layer_count {
        bail!(
            "stage layer_end {} exceeds package layer_count {}",
            request.layer_end,
            manifest.layer_count
        );
    }

    for layer_index in request.layer_start..request.layer_end {
        if !layer_counts.contains_key(&layer_index) {
            bail!("package is missing layer {layer_index}");
        }
    }
    Ok(())
}

fn validate_layer_manifest(manifest: &PackageManifest) -> Result<BTreeMap<u32, usize>> {
    let mut layer_counts = BTreeMap::<u32, usize>::new();
    for layer in &manifest.layers {
        *layer_counts.entry(layer.layer_index).or_default() += 1;
        if layer.layer_index >= manifest.layer_count {
            bail!(
                "package layer index {} exceeds layer_count {}",
                layer.layer_index,
                manifest.layer_count
            );
        }
        validate_artifact_manifest(
            &format!("layer {}", layer.layer_index),
            &PackageArtifact {
                path: layer.path.clone(),
                tensor_count: layer.tensor_count,
                tensor_bytes: layer.tensor_bytes,
                artifact_bytes: layer.artifact_bytes,
                sha256: layer.sha256.clone(),
            },
        )?;
    }
    let duplicates = layer_counts
        .iter()
        .filter_map(|(layer, count)| (*count > 1).then_some(*layer))
        .collect::<Vec<_>>();
    if !duplicates.is_empty() {
        bail!("package manifest contains duplicate layers: {duplicates:?}");
    }
    Ok(layer_counts)
}

fn validate_manifest_identity(manifest: &PackageManifest) -> Result<()> {
    if manifest.schema_version != 1 {
        bail!(
            "unsupported package manifest schema_version {}",
            manifest.schema_version
        );
    }
    if manifest.format != "layer-package" {
        bail!("package manifest format must be layer-package");
    }
    if !abi_version_supported(&manifest.skippy_abi_version)? {
        bail!(
            "package ABI version {} is not compatible with runtime ABI {}.{}.{}",
            manifest.skippy_abi_version,
            skippy_ffi::ABI_VERSION_MAJOR,
            skippy_ffi::ABI_VERSION_MINOR,
            skippy_ffi::ABI_VERSION_PATCH
        );
    }
    if manifest.model_id.trim().is_empty() {
        bail!("package manifest model_id must not be empty");
    }
    if manifest.source_model.path.trim().is_empty()
        || manifest
            .source_model
            .repo
            .as_deref()
            .is_some_and(|repo| repo.trim().is_empty())
        || manifest
            .source_model
            .revision
            .as_deref()
            .is_some_and(|revision| revision.trim().is_empty())
        || manifest
            .source_model
            .primary_file
            .as_deref()
            .is_some_and(|primary_file| primary_file.trim().is_empty())
        || manifest
            .source_model
            .canonical_ref
            .as_deref()
            .is_some_and(|canonical_ref| canonical_ref.trim().is_empty())
        || manifest
            .source_model
            .distribution_id
            .as_deref()
            .is_some_and(|distribution_id| distribution_id.trim().is_empty())
    {
        bail!("package manifest source_model identity fields must not be empty");
    }
    validate_sha256_digest("source_model sha256", &manifest.source_model.sha256)?;
    for file in &manifest.source_model.files {
        if file.path.trim().is_empty()
            || file
                .sha256
                .as_deref()
                .is_some_and(|sha256| sha256.trim().is_empty())
        {
            bail!("package manifest source_model files must not contain empty path or sha256");
        }
        if let Some(sha256) = &file.sha256 {
            validate_sha256_digest("source_model file sha256", sha256)?;
        }
        let _ = file.size_bytes;
    }
    validate_artifact_manifest("metadata", &manifest.shared.metadata)?;
    validate_artifact_manifest("embeddings", &manifest.shared.embeddings)?;
    validate_artifact_manifest("output", &manifest.shared.output)?;
    for projector in &manifest.projectors {
        validate_projector_manifest(projector)?;
    }
    Ok(())
}

fn validate_artifact_manifest(role: &str, artifact: &PackageArtifact) -> Result<()> {
    if artifact.path.trim().is_empty() {
        bail!("package {role} artifact path must not be empty");
    }
    let _ = safe_relative_manifest_path(&artifact.path)
        .with_context(|| format!("package {role} artifact path must be a safe relative path"))?;
    validate_sha256_digest(&format!("package {role} artifact sha256"), &artifact.sha256)?;
    if artifact.artifact_bytes == 0 {
        bail!("package {role} artifact_bytes must be greater than zero");
    }
    if artifact.tensor_count == 0 && artifact.tensor_bytes > 0 {
        bail!("package {role} tensor_bytes must be zero when tensor_count is zero");
    }
    if artifact.tensor_count > 0 && artifact.tensor_bytes == 0 {
        bail!("package {role} tensor_bytes must be greater than zero when tensors are present");
    }
    Ok(())
}

fn validate_projector_manifest(projector: &PackageProjector) -> Result<()> {
    if projector.kind.trim().is_empty() {
        bail!("package projector kind must not be empty");
    }
    if projector.kind != "mmproj" {
        bail!("unsupported package projector kind {}", projector.kind);
    }
    validate_artifact_manifest(
        &format!("{} projector", projector.kind),
        &PackageArtifact {
            path: projector.path.clone(),
            tensor_count: projector.tensor_count,
            tensor_bytes: projector.tensor_bytes,
            artifact_bytes: projector.artifact_bytes,
            sha256: projector.sha256.clone(),
        },
    )
}

fn push_part(
    parts: &mut Vec<PackagePart>,
    role: &str,
    layer_index: Option<u32>,
    artifact: &PackageArtifact,
    package_dir: &Path,
) -> Result<()> {
    let path = safe_relative_manifest_path(&artifact.path)?;
    let absolute = package_dir.join(&path);
    let metadata = fs::metadata(&absolute)
        .with_context(|| format!("read package part metadata {}", path.display()))?;
    if !metadata.is_file() {
        bail!("package part is not a file: {}", path.display());
    }
    if metadata.len() != artifact.artifact_bytes {
        bail!(
            "package part size mismatch for {}: expected {}, got {}",
            path.display(),
            artifact.artifact_bytes,
            metadata.len()
        );
    }
    parts.push(PackagePart {
        role: role.to_string(),
        layer_index,
        path,
        sha256: artifact.sha256.to_ascii_lowercase(),
        artifact_bytes: artifact.artifact_bytes,
    });
    Ok(())
}

fn projector_path(projector: &PackageProjector, package_dir: &Path) -> Result<PathBuf> {
    let path = safe_relative_manifest_path(&projector.path)?;
    let absolute = package_dir.join(&path);
    let metadata = fs::metadata(&absolute)
        .with_context(|| format!("read package projector metadata {}", path.display()))?;
    if !metadata.is_file() {
        bail!("package projector is not a file: {}", path.display());
    }
    if metadata.len() != projector.artifact_bytes {
        bail!(
            "package projector size mismatch for {}: expected {}, got {}",
            path.display(),
            projector.artifact_bytes,
            metadata.len()
        );
    }
    Ok(absolute)
}

#[derive(Debug)]
struct ArtifactVerification<'a> {
    role: &'a str,
    layer_index: Option<u32>,
    path: PathBuf,
    sha256: String,
    artifact_bytes: u64,
}

#[derive(Debug, Deserialize, Serialize)]
struct IntegrityCacheRecord {
    schema_version: u32,
    manifest_sha256: String,
    artifact_sha256: String,
    artifact_bytes: u64,
    file_len: u64,
    modified_unix_nanos: Option<u128>,
}

fn verify_package_artifacts(
    package_dir: &Path,
    manifest_sha256: &str,
    parts: &[PackagePart],
    projectors: &[PackageProjector],
    options: &PackageIntegrityOptions,
) -> Result<PackageIntegrityReport> {
    let artifacts = parts
        .iter()
        .map(|part| ArtifactVerification {
            role: &part.role,
            layer_index: part.layer_index,
            path: part.path.clone(),
            sha256: part.sha256.clone(),
            artifact_bytes: part.artifact_bytes,
        })
        .chain(projectors.iter().map(|projector| ArtifactVerification {
            role: "projector",
            layer_index: None,
            path: PathBuf::from(&projector.path),
            sha256: projector.sha256.to_ascii_lowercase(),
            artifact_bytes: projector.artifact_bytes,
        }))
        .collect::<Vec<_>>();

    let mut report = PackageIntegrityReport {
        manifest_sha256: manifest_sha256.to_string(),
        artifacts: artifacts.len(),
        verified_artifacts: 0,
        cached_artifacts: 0,
    };

    if !options.verify_sha256 {
        return Ok(report);
    }

    for artifact in artifacts {
        let relative_path = safe_relative_manifest_path(&artifact.path)?;
        let absolute = package_dir.join(&relative_path);
        let metadata = fs::metadata(&absolute).with_context(|| {
            format!("read package artifact metadata {}", relative_path.display())
        })?;
        let fingerprint = file_fingerprint(&metadata);
        if let Some(cache_dir) = &options.cache_dir {
            if integrity_cache_hit(
                cache_dir,
                manifest_sha256,
                &artifact,
                metadata.len(),
                fingerprint,
            )? {
                report.cached_artifacts += 1;
                continue;
            }
        }

        let actual = file_sha256(&absolute)?;
        if actual != artifact.sha256 {
            bail!(
                "package artifact checksum mismatch for {}: expected {}, got {}",
                relative_path.display(),
                artifact.sha256,
                actual
            );
        }
        report.verified_artifacts += 1;
        if let Some(cache_dir) = &options.cache_dir {
            write_integrity_cache_record(
                cache_dir,
                manifest_sha256,
                &artifact,
                metadata.len(),
                fingerprint,
            )?;
        }
    }

    Ok(report)
}

fn integrity_cache_hit(
    cache_dir: &Path,
    manifest_sha256: &str,
    artifact: &ArtifactVerification<'_>,
    file_len: u64,
    modified_unix_nanos: Option<u128>,
) -> Result<bool> {
    let path = integrity_cache_path(cache_dir, manifest_sha256, artifact);
    let Ok(bytes) = fs::read(&path) else {
        return Ok(false);
    };
    let Ok(record) = serde_json::from_slice::<IntegrityCacheRecord>(&bytes) else {
        return Ok(false);
    };
    Ok(record.schema_version == 1
        && record.manifest_sha256 == manifest_sha256
        && record.artifact_sha256 == artifact.sha256
        && record.artifact_bytes == artifact.artifact_bytes
        && record.file_len == file_len
        && record.modified_unix_nanos == modified_unix_nanos)
}

fn write_integrity_cache_record(
    cache_dir: &Path,
    manifest_sha256: &str,
    artifact: &ArtifactVerification<'_>,
    file_len: u64,
    modified_unix_nanos: Option<u128>,
) -> Result<()> {
    fs::create_dir_all(cache_dir)
        .with_context(|| format!("create package integrity cache {}", cache_dir.display()))?;
    let record = IntegrityCacheRecord {
        schema_version: 1,
        manifest_sha256: manifest_sha256.to_string(),
        artifact_sha256: artifact.sha256.clone(),
        artifact_bytes: artifact.artifact_bytes,
        file_len,
        modified_unix_nanos,
    };
    fs::write(
        integrity_cache_path(cache_dir, manifest_sha256, artifact),
        serde_json::to_vec_pretty(&record)?,
    )
    .with_context(|| format!("write package integrity cache {}", cache_dir.display()))
}

fn integrity_cache_path(
    cache_dir: &Path,
    manifest_sha256: &str,
    artifact: &ArtifactVerification<'_>,
) -> PathBuf {
    let mut hasher = Sha256::new();
    hasher.update(b"skippy-package-integrity-cache-v1\0");
    hasher.update(manifest_sha256.as_bytes());
    hasher.update(b"\0");
    hasher.update(artifact.role.as_bytes());
    hasher.update(b"\0");
    hasher.update(artifact.layer_index.unwrap_or(u32::MAX).to_le_bytes());
    hasher.update(b"\0");
    hasher.update(artifact.path.to_string_lossy().as_bytes());
    hasher.update(b"\0");
    hasher.update(artifact.sha256.as_bytes());
    hasher.update(b"\0");
    hasher.update(artifact.artifact_bytes.to_le_bytes());
    cache_dir.join(format!("{}.json", hex_lower(&hasher.finalize())))
}

fn file_fingerprint(metadata: &fs::Metadata) -> Option<u128> {
    metadata
        .modified()
        .ok()
        .and_then(|modified| modified.duration_since(UNIX_EPOCH).ok())
        .map(|duration| duration.as_nanos())
}

fn safe_relative_manifest_path(path: impl AsRef<Path>) -> Result<PathBuf> {
    let path = path.as_ref();
    let mut components = path.components();
    let Some(first) = components.next() else {
        bail!("manifest file path is empty");
    };
    anyhow::ensure!(
        matches!(first, std::path::Component::Normal(_))
            && components.all(|component| matches!(component, std::path::Component::Normal(_))),
        "manifest file path must be a safe relative path: {}",
        path.display()
    );
    Ok(path.to_path_buf())
}

fn validate_sha256_digest(label: &str, value: &str) -> Result<()> {
    if value.len() != 64 || !value.chars().all(|ch| ch.is_ascii_hexdigit()) {
        bail!("{label} must be a hex SHA-256 digest");
    }
    Ok(())
}

fn materialized_path(
    request: &PackageStageRequest,
    package_dir: &Path,
    manifest_sha256: &str,
    parts: &[PackagePart],
) -> PathBuf {
    let root = std::env::var_os("SKIPPY_MATERIALIZED_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("skippy-runtime/materialized"));
    let stable_package_dir = fs::canonicalize(package_dir).unwrap_or_else(|_| package_dir.into());
    let mut hasher = Sha256::new();
    hasher.update(stable_package_dir.to_string_lossy().as_bytes());
    hasher.update(b"\0");
    hasher.update(request.model_id.as_bytes());
    hasher.update(b"\0");
    hasher.update(request.topology_id.as_bytes());
    hasher.update(b"\0");
    hasher.update(request.stage_id.as_bytes());
    hasher.update(b"\0");
    hasher.update(request.layer_start.to_le_bytes());
    hasher.update(request.layer_end.to_le_bytes());
    hasher.update([
        u8::from(request.include_embeddings),
        u8::from(request.include_output),
    ]);
    hasher.update(manifest_sha256.as_bytes());
    for part in parts {
        hasher.update(b"\0");
        hasher.update(part.role.as_bytes());
        hasher.update(b"\0");
        hasher.update(part.layer_index.unwrap_or(u32::MAX).to_le_bytes());
        hasher.update(part.path.to_string_lossy().as_bytes());
        hasher.update(b"\0");
        hasher.update(part.sha256.as_bytes());
    }
    let digest = hex_lower(&hasher.finalize());
    let cache_key = &digest[..24];
    root.join(format!(
        "{}-{}-{}-{}-{}.gguf",
        sanitize(&request.model_id),
        sanitize(&request.stage_id),
        request.layer_start,
        request.layer_end,
        cache_key
    ))
}

fn abi_version_supported(version: &str) -> Result<bool> {
    let mut parts = version.split('.');
    let major = parts
        .next()
        .context("package ABI version is missing a major version")?
        .parse::<u32>()
        .context("parse package ABI major version")?;
    let minor = parts
        .next()
        .context("package ABI version is missing a minor version")?
        .parse::<u32>()
        .context("parse package ABI minor version")?;
    let _patch = parts
        .next()
        .unwrap_or("0")
        .parse::<u32>()
        .context("parse package ABI patch version")?;
    Ok(major == skippy_ffi::ABI_VERSION_MAJOR && minor <= skippy_ffi::ABI_VERSION_MINOR)
}

fn env_flag(name: &str) -> bool {
    std::env::var_os(name).is_some_and(|value| {
        let value = value.to_string_lossy();
        value != "0" && !value.eq_ignore_ascii_case("false")
    })
}

fn file_sha256(path: &Path) -> Result<String> {
    let mut file = fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
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

fn sha256_bytes(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hex_lower(&hasher.finalize())
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

fn sanitize(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
                ch
            } else {
                '_'
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_package_fixture(dir: &Path) -> serde_json::Value {
        fs::create_dir_all(dir.join("layers")).unwrap();
        fs::create_dir_all(dir.join("projectors")).unwrap();
        fs::write(dir.join("metadata.gguf"), b"metadata").unwrap();
        fs::write(dir.join("embeddings.gguf"), b"embeddings").unwrap();
        fs::write(dir.join("output.gguf"), b"output").unwrap();
        fs::write(dir.join("layers/00000.gguf"), b"layer0").unwrap();
        fs::write(dir.join("projectors/mmproj.gguf"), b"projector").unwrap();
        let manifest = serde_json::json!({
            "schema_version": 1,
            "model_id": "model-a",
            "source_model": {
                "path": "model-a.gguf",
                "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "files": [
                    {
                        "path": "model-a.gguf",
                        "size_bytes": 123,
                        "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                    }
                ]
            },
            "format": "layer-package",
            "layer_count": 1,
            "activation_width": 4096,
            "shared": {
                "metadata": {
                    "path": "metadata.gguf",
                    "tensor_count": 1,
                    "tensor_bytes": 1,
                    "artifact_bytes": 8,
                    "sha256": sha256_bytes(b"metadata")
                },
                "embeddings": {
                    "path": "embeddings.gguf",
                    "tensor_count": 1,
                    "tensor_bytes": 1,
                    "artifact_bytes": 10,
                    "sha256": sha256_bytes(b"embeddings")
                },
                "output": {
                    "path": "output.gguf",
                    "tensor_count": 1,
                    "tensor_bytes": 1,
                    "artifact_bytes": 6,
                    "sha256": sha256_bytes(b"output")
                }
            },
            "projectors": [
                {
                    "kind": "mmproj",
                    "path": "projectors/mmproj.gguf",
                    "tensor_count": 1,
                    "tensor_bytes": 1,
                    "artifact_bytes": 9,
                    "sha256": sha256_bytes(b"projector")
                }
            ],
            "layers": [
                {
                    "layer_index": 0,
                    "path": "layers/00000.gguf",
                    "tensor_count": 1,
                    "tensor_bytes": 1,
                    "artifact_bytes": 6,
                    "sha256": sha256_bytes(b"layer0")
                }
            ],
            "skippy_abi_version": format!(
                "{}.{}.{}",
                skippy_ffi::ABI_VERSION_MAJOR,
                skippy_ffi::ABI_VERSION_MINOR,
                skippy_ffi::ABI_VERSION_PATCH
            ),
        });
        fs::write(
            dir.join("model-package.json"),
            serde_json::to_vec_pretty(&manifest).unwrap(),
        )
        .unwrap();
        manifest
    }

    fn package_stage_request(package_ref: &Path) -> PackageStageRequest {
        PackageStageRequest {
            model_id: "model-a".to_string(),
            topology_id: "topology-a".to_string(),
            package_ref: package_ref.to_string_lossy().to_string(),
            stage_id: "stage-0".to_string(),
            layer_start: 0,
            layer_end: 1,
            include_embeddings: true,
            include_output: true,
        }
    }

    #[test]
    fn parses_hf_package_refs() {
        assert_eq!(
            parse_hf_package_ref("hf://Mesh-LLM/Qwen3.6-package").unwrap(),
            HfPackageRef {
                repo_id: "Mesh-LLM/Qwen3.6-package".to_string(),
                revision: None,
            }
        );
        assert_eq!(
            parse_hf_package_ref("hf://Mesh-LLM/Qwen3.6-package:abc123").unwrap(),
            HfPackageRef {
                repo_id: "Mesh-LLM/Qwen3.6-package".to_string(),
                revision: Some("abc123".to_string()),
            }
        );
        assert_eq!(
            parse_hf_package_ref("hf://Mesh-LLM/Qwen3.6-package@branch-name").unwrap(),
            HfPackageRef {
                repo_id: "Mesh-LLM/Qwen3.6-package".to_string(),
                revision: Some("branch-name".to_string()),
            }
        );
    }

    #[test]
    fn rejects_invalid_hf_package_refs() {
        assert!(parse_hf_package_ref("hf://").is_err());
        assert!(parse_hf_package_ref("hf://namespace-only").is_err());
        assert!(parse_hf_package_ref("hf://namespace/repo@").is_err());
        assert!(parse_hf_package_ref("hf://namespace/repo:").is_err());
    }

    #[test]
    fn checks_abi_version_compatibility() {
        assert!(abi_version_supported(&format!(
            "{}.{}.0",
            skippy_ffi::ABI_VERSION_MAJOR,
            skippy_ffi::ABI_VERSION_MINOR
        ))
        .unwrap());
        assert!(
            !abi_version_supported(&format!("{}.{}.0", skippy_ffi::ABI_VERSION_MAJOR + 1, 0))
                .unwrap()
        );
        assert!(!abi_version_supported(&format!(
            "{}.{}.0",
            skippy_ffi::ABI_VERSION_MAJOR,
            skippy_ffi::ABI_VERSION_MINOR + 1
        ))
        .unwrap());
    }

    #[test]
    fn inspect_layer_package_returns_manifest_identity() {
        let dir = tempfile::tempdir().unwrap();
        write_package_fixture(dir.path());

        let info = inspect_layer_package(&dir.path().to_string_lossy()).unwrap();

        assert_eq!(info.model_id, "model-a");
        assert_eq!(info.layer_count, 1);
        assert_eq!(info.activation_width, Some(4096));
        assert_eq!(info.source_model_bytes, Some(123));
        assert_eq!(info.projectors.len(), 1);
        assert_eq!(info.projectors[0].kind, "mmproj");
        assert_eq!(
            info.projectors[0].path,
            dir.path().join("projectors/mmproj.gguf")
        );
        assert_eq!(info.manifest_sha256.len(), 64);
    }

    #[test]
    fn verify_layer_package_integrity_rejects_selected_artifact_sha_mismatch() {
        let dir = tempfile::tempdir().unwrap();
        write_package_fixture(dir.path());
        fs::write(dir.path().join("layers/00000.gguf"), b"wrong0").unwrap();

        let error = verify_layer_package_integrity(
            &package_stage_request(dir.path()),
            &PackageIntegrityOptions::verify_without_cache(),
        )
        .unwrap_err()
        .to_string();

        assert!(error.contains("checksum mismatch"), "{error}");
        assert!(error.contains("layers/00000.gguf"), "{error}");
    }

    #[test]
    fn verify_layer_package_metadata_integrity_allows_metadata_only_scope() {
        let dir = tempfile::tempdir().unwrap();
        write_package_fixture(dir.path());
        fs::write(dir.path().join("layers/00000.gguf"), b"wrong0").unwrap();
        fs::write(
            dir.path().join("projectors/mmproj.gguf"),
            b"wrong-projector",
        )
        .unwrap();

        let report = verify_layer_package_metadata_integrity(
            &dir.path().to_string_lossy(),
            &PackageIntegrityOptions::verify_without_cache(),
        )
        .expect("metadata-only verification should not require a stage layer range");

        assert_eq!(report.artifacts, 1);
        assert_eq!(report.verified_artifacts, 1);
        assert_eq!(report.cached_artifacts, 0);
    }

    #[test]
    fn verify_layer_package_metadata_integrity_checks_metadata_sha() {
        let dir = tempfile::tempdir().unwrap();
        write_package_fixture(dir.path());
        fs::write(dir.path().join("metadata.gguf"), b"metadota").unwrap();

        let error = verify_layer_package_metadata_integrity(
            &dir.path().to_string_lossy(),
            &PackageIntegrityOptions::verify_without_cache(),
        )
        .unwrap_err()
        .to_string();

        assert!(error.contains("checksum mismatch"), "{error}");
        assert!(error.contains("metadata.gguf"), "{error}");
    }

    #[test]
    fn verify_layer_package_integrity_uses_private_cache_records() {
        let dir = tempfile::tempdir().unwrap();
        let cache_dir = tempfile::tempdir().unwrap();
        write_package_fixture(dir.path());

        let options = PackageIntegrityOptions::verify_with_cache(cache_dir.path());
        let first = verify_layer_package_integrity(&package_stage_request(dir.path()), &options)
            .expect("first verification should hash artifacts");
        assert_eq!(first.verified_artifacts, 5);
        assert_eq!(first.cached_artifacts, 0);

        let second = verify_layer_package_integrity(&package_stage_request(dir.path()), &options)
            .expect("second verification should reuse cache");
        assert_eq!(second.verified_artifacts, 0);
        assert_eq!(second.cached_artifacts, 5);

        let cache_blob = fs::read_to_string(
            fs::read_dir(cache_dir.path())
                .unwrap()
                .next()
                .unwrap()
                .unwrap()
                .path(),
        )
        .unwrap();
        assert!(
            !cache_blob.contains(&dir.path().to_string_lossy().to_string()),
            "cache records must not store raw local package paths"
        );
    }

    #[test]
    fn validates_source_model_sha256_as_hex_digest() {
        let dir = tempfile::tempdir().unwrap();
        let mut manifest = write_package_fixture(dir.path());
        manifest["source_model"]["sha256"] = serde_json::Value::String("not-a-sha".to_string());
        fs::write(
            dir.path().join("model-package.json"),
            serde_json::to_vec_pretty(&manifest).unwrap(),
        )
        .unwrap();

        let error = inspect_layer_package(&dir.path().to_string_lossy())
            .unwrap_err()
            .to_string();

        assert!(error.contains("source_model sha256"), "{error}");
    }

    #[test]
    fn rejects_package_artifact_paths_that_escape_package_dir() {
        let dir = tempfile::tempdir().unwrap();
        let mut manifest = write_package_fixture(dir.path());
        manifest["layers"][0]["path"] = serde_json::Value::String("../outside.gguf".to_string());
        fs::write(
            dir.path().join("model-package.json"),
            serde_json::to_vec_pretty(&manifest).unwrap(),
        )
        .unwrap();

        let error = select_layer_package_parts(&package_stage_request(dir.path()))
            .unwrap_err()
            .to_string();

        assert!(error.contains("safe relative"), "{error}");
    }

    #[test]
    fn inspect_layer_package_rejects_unsafe_layer_paths() {
        let dir = tempfile::tempdir().unwrap();
        let mut manifest = write_package_fixture(dir.path());
        manifest["layers"][0]["path"] = serde_json::Value::String("../outside.gguf".to_string());
        fs::write(
            dir.path().join("model-package.json"),
            serde_json::to_vec_pretty(&manifest).unwrap(),
        )
        .unwrap();

        let error = inspect_layer_package(&dir.path().to_string_lossy())
            .unwrap_err()
            .to_string();

        assert!(error.contains("safe relative"), "{error}");
    }

    #[test]
    fn activation_width_inference_rejects_u32_overflow() {
        let path = Path::new("/package/layers/00000.gguf");

        let error = activation_width_from_tensor_count(
            "blk.0.attn_norm.weight",
            path,
            u64::from(u32::MAX) + 1,
        )
        .unwrap_err()
        .to_string();

        assert!(error.contains("exceeds u32::MAX"));
    }
}
