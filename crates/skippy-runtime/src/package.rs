use std::{
    collections::BTreeMap,
    ffi::OsString,
    fs,
    io::Read,
    path::{Path, PathBuf},
    process::Command,
};

use anyhow::{bail, Context, Result};
use serde::Deserialize;
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
    pub layers: Vec<LayerPackageLayerInfo>,
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

    // Download only the layer files needed for this stage (selective fetch).
    ensure_layer_files(&request.package_ref, &package_dir, &parts)?;

    if env_flag("SKIPPY_VERIFY_PACKAGE_SHA") {
        for part in &parts {
            let absolute = package_dir.join(&part.path);
            let actual = file_sha256(&absolute)?;
            if actual != part.sha256 {
                bail!(
                    "package part checksum mismatch for {}: expected {}, got {}",
                    part.path.display(),
                    part.sha256,
                    actual
                );
            }
        }
    }

    let absolute_paths = parts
        .iter()
        .map(|part| package_dir.join(&part.path))
        .collect::<Vec<_>>();

    Ok(SelectedPackageParts {
        package_dir,
        manifest_sha256,
        selected_parts: parts,
        absolute_paths,
    })
}

pub fn inspect_layer_package(package_ref: &str) -> Result<LayerPackageInfo> {
    let package_dir = resolve_package_dir(package_ref)?;
    let manifest_path = package_dir.join("model-package.json");
    let manifest_contents = fs::read(&manifest_path)
        .with_context(|| format!("read package manifest {}", manifest_path.display()))?;
    let manifest_sha256 = sha256_bytes(&manifest_contents);
    let manifest = load_manifest(&manifest_path, &manifest_contents)?;
    validate_manifest_identity(&manifest)?;
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
        activation_width: manifest.activation_width,
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

pub fn is_hf_package_ref(value: &str) -> bool {
    value.starts_with("hf://")
}

fn resolve_package_dir(package_ref: &str) -> Result<PathBuf> {
    if is_hf_package_ref(package_ref) {
        download_hf_package_manifest(package_ref)
    } else {
        Ok(PathBuf::from(package_ref))
    }
}

/// Download only the manifest from an HF layer package repo.
/// Layer files are fetched on-demand by `ensure_layer_files`.
fn download_hf_package_manifest(package_ref: &str) -> Result<PathBuf> {
    let reference = parse_hf_package_ref(package_ref)?;
    let package_dir = hf_cache_root().join(sanitize(&format!(
        "{}-{}",
        reference.repo_id,
        reference.revision.as_deref().unwrap_or("main")
    )));
    if package_dir.join("model-package.json").is_file() {
        return Ok(package_dir);
    }

    fs::create_dir_all(&package_dir)
        .with_context(|| format!("create HF package cache {}", package_dir.display()))?;

    // Download only the manifest initially
    let mut args = vec![
        OsString::from("download"),
        OsString::from(reference.repo_id.as_str()),
        OsString::from("--local-dir"),
        package_dir.as_os_str().to_os_string(),
        OsString::from("--quiet"),
        OsString::from("--include"),
        OsString::from("model-package.json"),
    ];
    if let Some(revision) = reference.revision.as_ref() {
        args.push(OsString::from("--revision"));
        args.push(OsString::from(revision));
    }

    let status = Command::new("hf")
        .args(&args)
        .status()
        .context("run hf download for layer package manifest")?;
    if !status.success() {
        bail!("hf download failed for {}", reference.repo_id);
    }
    if !package_dir.join("model-package.json").is_file() {
        bail!(
            "downloaded HF package is missing model-package.json: {}",
            package_dir.display()
        );
    }
    Ok(package_dir)
}

/// Download specific layer files and shared artifacts for a stage assignment.
/// Only fetches files that aren't already present on disk.
fn ensure_layer_files(
    package_ref: &str,
    package_dir: &Path,
    parts: &[PackagePart],
) -> Result<()> {
    if !is_hf_package_ref(package_ref) {
        return Ok(()); // local package, files already present
    }

    // Check which files are missing
    let missing: Vec<&PackagePart> = parts
        .iter()
        .filter(|part| {
            let path = package_dir.join(&part.path);
            !path.is_file()
                || fs::metadata(&path)
                    .map(|m| m.len() != part.artifact_bytes)
                    .unwrap_or(true)
        })
        .collect();

    if missing.is_empty() {
        return Ok(());
    }

    let reference = parse_hf_package_ref(package_ref)?;
    let mut args = vec![
        OsString::from("download"),
        OsString::from(reference.repo_id.as_str()),
        OsString::from("--local-dir"),
        package_dir.as_os_str().to_os_string(),
        OsString::from("--quiet"),
    ];
    if let Some(revision) = reference.revision.as_ref() {
        args.push(OsString::from("--revision"));
        args.push(OsString::from(revision));
    }

    // Add --include for each missing file
    for part in &missing {
        args.push(OsString::from("--include"));
        args.push(OsString::from(part.path.to_string_lossy().as_ref()));
    }

    eprintln!(
        "skippy: downloading {} layer files from {}",
        missing.len(),
        reference.repo_id
    );

    let status = Command::new("hf")
        .args(&args)
        .status()
        .context("run hf download for layer package files")?;
    if !status.success() {
        bail!(
            "hf download failed for layer files from {}",
            reference.repo_id
        );
    }
    Ok(())
}

#[derive(Debug, PartialEq, Eq)]
struct HfPackageRef {
    repo_id: String,
    revision: Option<String>,
}

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

fn hf_cache_root() -> PathBuf {
    std::env::var_os("SKIPPY_HF_PACKAGE_CACHE")
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os("HF_HOME")
                .map(PathBuf::from)
                .map(|path| path.join("skippy-runtime/packages"))
        })
        .or_else(|| {
            std::env::var_os("HOME")
                .map(PathBuf::from)
                .map(|path| path.join(".cache/skippy-runtime/hf-packages"))
        })
        .unwrap_or_else(|| std::env::temp_dir().join("skippy-runtime/hf-packages"))
}

fn load_manifest(path: &Path, contents: &[u8]) -> Result<PackageManifest> {
    serde_json::from_slice(contents)
        .with_context(|| format!("parse package manifest {}", path.display()))
}

fn validate_manifest(manifest: &PackageManifest, request: &PackageStageRequest) -> Result<()> {
    validate_manifest_identity(manifest)?;
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
    for layer_index in request.layer_start..request.layer_end {
        if !layer_counts.contains_key(&layer_index) {
            bail!("package is missing layer {layer_index}");
        }
    }
    Ok(())
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
        || manifest.source_model.sha256.trim().is_empty()
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
    for file in &manifest.source_model.files {
        if file.path.trim().is_empty()
            || file
                .sha256
                .as_deref()
                .is_some_and(|sha256| sha256.trim().is_empty())
        {
            bail!("package manifest source_model files must not contain empty path or sha256");
        }
        let _ = file.size_bytes;
    }
    validate_artifact_manifest("metadata", &manifest.shared.metadata)?;
    validate_artifact_manifest("embeddings", &manifest.shared.embeddings)?;
    validate_artifact_manifest("output", &manifest.shared.output)?;
    Ok(())
}

fn validate_artifact_manifest(role: &str, artifact: &PackageArtifact) -> Result<()> {
    if artifact.path.trim().is_empty() {
        bail!("package {role} artifact path must not be empty");
    }
    if artifact.sha256.len() != 64 || !artifact.sha256.chars().all(|ch| ch.is_ascii_hexdigit()) {
        bail!("package {role} artifact sha256 must be a hex SHA-256 digest");
    }
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

fn push_part(
    parts: &mut Vec<PackagePart>,
    role: &str,
    layer_index: Option<u32>,
    artifact: &PackageArtifact,
    package_dir: &Path,
) -> Result<()> {
    let path = PathBuf::from(&artifact.path);
    if path.is_absolute() {
        bail!("package part paths must be relative: {}", path.display());
    }
    let absolute = package_dir.join(&path);
    let metadata = fs::metadata(&absolute)
        .with_context(|| format!("read package part metadata {}", absolute.display()))?;
    if !metadata.is_file() {
        bail!("package part is not a file: {}", absolute.display());
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
        fs::create_dir_all(dir.path().join("layers")).unwrap();
        fs::write(dir.path().join("metadata.gguf"), b"metadata").unwrap();
        fs::write(dir.path().join("embeddings.gguf"), b"embeddings").unwrap();
        fs::write(dir.path().join("output.gguf"), b"output").unwrap();
        fs::write(dir.path().join("layers/00000.gguf"), b"layer0").unwrap();
        let manifest = serde_json::json!({
            "schema_version": 1,
            "model_id": "model-a",
            "source_model": {
                "path": "/models/model-a.gguf",
                "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "files": [
                    {
                        "path": "/models/model-a.gguf",
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
            dir.path().join("model-package.json"),
            serde_json::to_vec_pretty(&manifest).unwrap(),
        )
        .unwrap();

        let info = inspect_layer_package(&dir.path().to_string_lossy()).unwrap();

        assert_eq!(info.model_id, "model-a");
        assert_eq!(info.layer_count, 1);
        assert_eq!(info.activation_width, Some(4096));
        assert_eq!(info.source_model_bytes, Some(123));
        assert_eq!(info.manifest_sha256.len(), 64);
    }
}
