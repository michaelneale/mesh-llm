use std::{
    collections::BTreeSet,
    fs,
    path::{Path, PathBuf},
    process::Command,
    sync::OnceLock,
};

use anyhow::{anyhow, bail, Context, Result};
use model_ref::split_gguf_shard_info;
use serde::Deserialize;
use serde_json::Value;
use skippy_runtime::{
    package::{inspect_layer_package, materialize_layer_package_details, PackageStageRequest},
    redirect_native_logs_to_file, ActivationFrame, RuntimeConfig, RuntimeKvPage, RuntimeLoadMode,
    StageModel, StageSession, GGML_TYPE_F16,
};

use crate::FamilySpec;

const MANIFEST_JSON: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../docs/skippy/llama-parity-candidates.json"
));

const PROMPT: &str = "Hello from the Skippy llama parity harness.";
const CTX_SIZE: u32 = 128;
const CACHE_SEQ_ID: i32 = 17;

pub(crate) fn p0_p1_manifest_rows() -> BTreeSet<(String, String, String)> {
    let manifest = manifest();
    manifest
        .rows_by_priority(["p0", "p1"])
        .into_iter()
        .map(|row| {
            (
                manifest.priority_for(row).to_string(),
                row.llama_model.clone(),
                row.family.clone(),
            )
        })
        .collect()
}

pub(crate) fn assert_manifest_row_complete(spec: FamilySpec) -> Result<()> {
    let row = manifest_row(spec)?;
    if !matches!(row.status.as_str(), "certified" | "certified_package_only") {
        bail!(
            "{} / {} is {}, but P0/P1 rows must be certified before getting a parity module",
            row.llama_model,
            row.family,
            row.status
        );
    }
    if row.repo.trim().is_empty() {
        bail!("{} / {} is missing repo", row.llama_model, row.family);
    }
    if row.include()?.files().is_empty() {
        bail!(
            "{} / {} is missing include files",
            row.llama_model,
            row.family
        );
    }
    Ok(())
}

pub(crate) fn activation_handoff_matches_full_model(spec: FamilySpec) -> Result<()> {
    prepare_native_logs()?;
    let Some(case) = resolve_case_for_ignored_test(spec)? else {
        return Ok(());
    };
    let layout = case.layout()?;
    if case.row.is_package_only() {
        return run_lightweight_activation_smoke(&layout, spec);
    }
    let splits = split_layers_for(case.row, layout.layer_count)?;
    run_correctness_chain(&layout, spec, splits)
}

pub(crate) fn cache_state_restore_matches_recompute(spec: FamilySpec) -> Result<()> {
    prepare_native_logs()?;
    let Some(case) = resolve_case_for_ignored_test(spec)? else {
        return Ok(());
    };
    let layout = case.layout()?;
    let (stage_start, stage_end) = cache_stage_range(case.row, &layout)?;
    let state_kind = if case.row.is_recurrent() {
        StateKind::KvRecurrent
    } else {
        StateKind::ResidentKv
    };
    run_stage_state_restore(&layout, spec, stage_start, stage_end, state_kind)
}

fn resolve_case_for_ignored_test(spec: FamilySpec) -> Result<Option<ResolvedCase>> {
    match ResolvedCase::resolve(spec) {
        Ok(case) => Ok(Some(case)),
        Err(error)
            if manifest_row(spec)?.is_package_only()
                && !env_flag("SKIPPY_PARITY_REQUIRE_PACKAGE_ONLY", false) =>
        {
            eprintln!(
                "skipping package-only parity artifact for {} / {}: {error:#}",
                spec.llama_model, spec.family
            );
            Ok(None)
        }
        Err(error) => Err(error),
    }
}

fn prepare_native_logs() -> Result<()> {
    static LOG_PATH: OnceLock<PathBuf> = OnceLock::new();

    if env_flag("SKIPPY_PARITY_NATIVE_LOGS", false) {
        return Ok(());
    }

    let path = LOG_PATH.get_or_init(|| {
        std::env::temp_dir().join(format!("skippy-parity-native-{}.log", std::process::id()))
    });
    redirect_native_logs_to_file(path)
}

#[derive(Debug, Deserialize)]
struct Manifest {
    support_priority: SupportPriority,
    candidates: Vec<CandidateRow>,
}

#[derive(Debug, Deserialize)]
struct SupportPriority {
    p0: PriorityRows,
    p1: PriorityRows,
}

#[derive(Debug, Deserialize)]
struct PriorityRows {
    llama_models: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct CandidateRow {
    llama_model: String,
    family: String,
    #[serde(default)]
    status: String,
    #[serde(default)]
    repo: String,
    #[serde(default)]
    include: Option<IncludeSpec>,
    #[serde(default)]
    recurrent: Option<String>,
    #[serde(default)]
    recurrent_ranges: Option<Vec<String>>,
    #[serde(default)]
    splits: Option<String>,
    #[serde(default)]
    prompt: Option<String>,
    #[serde(default)]
    wire_dtype: Option<String>,
    #[serde(default)]
    n_gpu_layers: Option<i32>,
}

impl CandidateRow {
    fn include(&self) -> Result<&IncludeSpec> {
        self.include
            .as_ref()
            .with_context(|| format!("{} / {} is missing include", self.llama_model, self.family))
    }

    fn is_package_only(&self) -> bool {
        self.status == "certified_package_only"
    }

    fn is_recurrent(&self) -> bool {
        self.recurrent.is_some()
            || self
                .recurrent_ranges
                .as_ref()
                .is_some_and(|ranges| !ranges.is_empty())
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum IncludeSpec {
    One(String),
    Many(Vec<String>),
}

impl IncludeSpec {
    fn files(&self) -> Vec<&str> {
        match self {
            Self::One(file) => vec![file.as_str()],
            Self::Many(files) => files.iter().map(String::as_str).collect(),
        }
    }

    fn is_package_manifest_only(&self) -> bool {
        matches!(self, Self::One(value) if value == "model-package.json")
    }
}

impl Manifest {
    fn priority_for(&self, row: &CandidateRow) -> &'static str {
        if self
            .support_priority
            .p0
            .llama_models
            .iter()
            .any(|model| model == &row.llama_model)
        {
            "p0"
        } else if self
            .support_priority
            .p1
            .llama_models
            .iter()
            .any(|model| model == &row.llama_model)
        {
            "p1"
        } else {
            "p2"
        }
    }

    fn rows_by_priority<const N: usize>(&self, priorities: [&str; N]) -> Vec<&CandidateRow> {
        self.candidates
            .iter()
            .filter(|row| priorities.contains(&self.priority_for(row)))
            .collect()
    }
}

static MANIFEST: OnceLock<Manifest> = OnceLock::new();

fn manifest() -> &'static Manifest {
    MANIFEST.get_or_init(|| {
        serde_json::from_str(MANIFEST_JSON).expect("parity candidate manifest must parse")
    })
}

fn manifest_row(spec: FamilySpec) -> Result<&'static CandidateRow> {
    let row = manifest()
        .candidates
        .iter()
        .find(|row| row.llama_model == spec.llama_model && row.family == spec.family)
        .with_context(|| {
            format!(
                "missing manifest row for {} / {}",
                spec.llama_model, spec.family
            )
        })?;
    let actual_priority = manifest().priority_for(row);
    if actual_priority != spec.priority {
        bail!(
            "{} / {} declared as {}, manifest says {}",
            spec.llama_model,
            spec.family,
            spec.priority,
            actual_priority
        );
    }
    Ok(row)
}

struct ResolvedCase {
    row: &'static CandidateRow,
    artifact: ResolvedArtifact,
}

enum ResolvedArtifact {
    Gguf(PathBuf),
    LayerPackage(PathBuf),
}

struct TestLayout {
    layer_count: u32,
    full_model: StagePath,
    package_dir: Option<PathBuf>,
}

#[derive(Clone)]
struct StagePath {
    path: PathBuf,
    load_mode: RuntimeLoadMode,
    filter_tensors_on_load: bool,
}

impl ResolvedCase {
    fn resolve(spec: FamilySpec) -> Result<Self> {
        assert_manifest_row_complete(spec)?;
        let row = manifest_row(spec)?;
        download_if_requested(row)?;
        let artifact = if row.include()?.is_package_manifest_only() {
            ResolvedArtifact::LayerPackage(resolve_package_dir(row)?)
        } else {
            ResolvedArtifact::Gguf(resolve_primary_gguf(row)?)
        };
        Ok(Self { row, artifact })
    }

    fn layout(&self) -> Result<TestLayout> {
        match &self.artifact {
            ResolvedArtifact::Gguf(path) => {
                let layer_count = layer_count_for_gguf(path)?;
                Ok(TestLayout {
                    layer_count,
                    full_model: StagePath {
                        path: path.clone(),
                        load_mode: RuntimeLoadMode::RuntimeSlice,
                        filter_tensors_on_load: false,
                    },
                    package_dir: None,
                })
            }
            ResolvedArtifact::LayerPackage(package_dir) => {
                let package_ref = package_dir.to_string_lossy();
                let info = inspect_layer_package(&package_ref)
                    .with_context(|| format!("inspect layer package {}", package_dir.display()))?;
                Ok(TestLayout {
                    layer_count: info.layer_count,
                    full_model: StagePath {
                        path: package_dir.clone(),
                        load_mode: RuntimeLoadMode::LayerPackage,
                        filter_tensors_on_load: true,
                    },
                    package_dir: Some(package_dir.clone()),
                })
            }
        }
    }
}

fn download_if_requested(row: &CandidateRow) -> Result<()> {
    if env_flag("SKIPPY_PARITY_DOWNLOAD", true) {
        if row.is_package_only()
            && !row.include()?.is_package_manifest_only()
            && !env_flag("SKIPPY_PARITY_DOWNLOAD_PACKAGE_ONLY", false)
        {
            return Ok(());
        }
        let mut command = Command::new("hf");
        command.args(["download", &row.repo]);
        if !row.include()?.is_package_manifest_only() {
            for include in row.include()?.files() {
                command.args(["--include", include]);
            }
        }
        let status = command
            .status()
            .with_context(|| format!("run hf download for {}", row.repo))?;
        if !status.success() {
            bail!("hf download failed for {} with status {status}", row.repo);
        }
    }
    Ok(())
}

fn resolve_primary_gguf(row: &CandidateRow) -> Result<PathBuf> {
    let candidates = repo_snapshot_files(&row.repo)?
        .into_iter()
        .filter(|path| {
            let rel = repo_relative_path(&row.repo, path)
                .map(|path| path.to_string_lossy().replace('\\', "/"))
                .unwrap_or_default();
            row.include()
                .map(IncludeSpec::files)
                .unwrap_or_default()
                .iter()
                .any(|pattern| wildcard_match(pattern, &rel))
                && path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| {
                        let lower = name.to_ascii_lowercase();
                        lower.ends_with(".gguf") && !lower.starts_with("mmproj")
                    })
        })
        .collect::<Vec<_>>();
    choose_primary_gguf(row, candidates)
}

fn resolve_package_dir(row: &CandidateRow) -> Result<PathBuf> {
    repo_snapshot_dirs(&row.repo)?
        .into_iter()
        .find(|dir| dir.join("model-package.json").is_file())
        .with_context(|| {
            format!(
                "no downloaded model-package.json for {}; run SKIPPY_PARITY_DOWNLOAD=1 cargo test -p skippy-correctness --test parity_models -- --ignored",
                row.repo
            )
        })
}

fn choose_primary_gguf(row: &CandidateRow, mut candidates: Vec<PathBuf>) -> Result<PathBuf> {
    candidates.sort_by_key(|path| {
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or_default()
            .to_ascii_lowercase();
        (
            !name.contains("-00001-of-"),
            name.contains("-000") && !name.contains("-00001-of-"),
            name,
        )
    });
    candidates.into_iter().next().with_context(|| {
        format!(
            "no downloaded GGUF for {} include {:?}; run SKIPPY_PARITY_DOWNLOAD=1 cargo test -p skippy-correctness --test parity_models -- --ignored",
            row.repo,
            row.include().map(IncludeSpec::files).unwrap_or_default()
        )
    })
}

fn repo_snapshot_files(repo: &str) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for dir in repo_snapshot_dirs(repo)? {
        collect_files(&dir, &mut files)?;
    }
    Ok(files)
}

fn repo_snapshot_dirs(repo: &str) -> Result<Vec<PathBuf>> {
    let cache_dir = model_hf::huggingface_hub_cache_dir();
    let repo_dir = cache_dir.join(format!("models--{}", repo.replace('/', "--")));
    let snapshots = repo_dir.join("snapshots");
    if !snapshots.is_dir() {
        bail!(
            "Hugging Face cache has no snapshots for {} at {}; run with SKIPPY_PARITY_DOWNLOAD=1",
            repo,
            snapshots.display()
        );
    }
    let mut dirs = fs::read_dir(&snapshots)
        .with_context(|| format!("read {}", snapshots.display()))?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.is_dir())
        .collect::<Vec<_>>();
    dirs.sort();
    dirs.reverse();
    Ok(dirs)
}

fn repo_relative_path(repo: &str, path: &Path) -> Option<PathBuf> {
    let cache_dir = model_hf::huggingface_hub_cache_dir();
    let snapshots = cache_dir
        .join(format!("models--{}", repo.replace('/', "--")))
        .join("snapshots");
    for snapshot in fs::read_dir(snapshots).ok()? {
        let snapshot = snapshot.ok()?.path();
        if let Ok(relative) = path.strip_prefix(snapshot) {
            return Some(relative.to_path_buf());
        }
    }
    None
}

fn collect_files(dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    for entry in fs::read_dir(dir).with_context(|| format!("read {}", dir.display()))? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect_files(&path, out)?;
        } else if path.is_file() {
            out.push(path);
        }
    }
    Ok(())
}

fn wildcard_match(pattern: &str, value: &str) -> bool {
    let pattern = pattern.replace('\\', "/");
    if pattern == value {
        return true;
    }
    let parts = pattern.split('*').collect::<Vec<_>>();
    if parts.len() == 1 {
        return pattern == value;
    }
    let mut remainder = value;
    if let Some(first) = parts.first().filter(|part| !part.is_empty()) {
        let Some(stripped) = remainder.strip_prefix(first) else {
            return false;
        };
        remainder = stripped;
    }
    for part in parts.iter().skip(1).take(parts.len().saturating_sub(2)) {
        if part.is_empty() {
            continue;
        }
        let Some(index) = remainder.find(part) else {
            return false;
        };
        remainder = &remainder[index + part.len()..];
    }
    if let Some(last) = parts.last().filter(|part| !part.is_empty()) {
        return remainder.ends_with(last);
    }
    true
}

fn layer_count_for_gguf(path: &Path) -> Result<u32> {
    let max_layer = tensors_for_gguf(path)?
        .into_iter()
        .filter_map(|tensor| tensor.layer_index)
        .max()
        .with_context(|| format!("{} has no layer-indexed tensors", path.display()))?;
    Ok(max_layer.saturating_add(1))
}

fn tensors_for_gguf(path: &Path) -> Result<Vec<skippy_runtime::TensorInfo>> {
    let info = skippy_runtime::ModelInfo::open(path)
        .with_context(|| format!("inspect {}", path.display()))?;
    let tensors = info.tensors()?;
    if tensors.iter().any(|tensor| tensor.layer_index.is_some()) {
        return Ok(tensors);
    }

    let shard_paths = split_gguf_sibling_paths(path)?;
    if shard_paths.len() <= 1 {
        return Ok(tensors);
    }

    let mut all_tensors = tensors;
    for shard in shard_paths.into_iter().filter(|shard| shard != path) {
        let info = skippy_runtime::ModelInfo::open(&shard)
            .with_context(|| format!("inspect split GGUF shard {}", shard.display()))?;
        all_tensors.extend(info.tensors()?);
    }
    Ok(all_tensors)
}

fn split_gguf_sibling_paths(path: &Path) -> Result<Vec<PathBuf>> {
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .context("split GGUF path has no UTF-8 file name")?;
    let Some(shard) = split_gguf_shard_info(file_name) else {
        return Ok(vec![path.to_path_buf()]);
    };
    let total = shard
        .total
        .parse::<u32>()
        .context("parse split GGUF shard total")?;
    let dir = path
        .parent()
        .ok_or_else(|| anyhow!("split GGUF path has no parent directory"))?;
    let mut paths = Vec::new();
    for part in 1..=total {
        let shard_name = format!("{}-{part:05}-of-{}.gguf", shard.prefix, shard.total);
        paths.push(dir.join(shard_name));
    }
    Ok(paths)
}

fn split_layers_for(row: &CandidateRow, layer_count: u32) -> Result<(u32, u32)> {
    if let Some(splits) = row.splits.as_deref() {
        return parse_reviewed_splits(row, splits, layer_count);
    }
    split_layers(layer_count)
}

fn parse_reviewed_splits(row: &CandidateRow, splits: &str, layer_count: u32) -> Result<(u32, u32)> {
    let values = splits
        .split(',')
        .map(|value| {
            value.trim().parse::<u32>().with_context(|| {
                format!(
                    "parse reviewed splits for {} / {}",
                    row.llama_model, row.family
                )
            })
        })
        .collect::<Result<Vec<_>>>()?;
    match values.as_slice() {
        [split_1, split_2]
            if *split_1 > 0 && split_1 < split_2 && *split_2 < layer_count =>
        {
            Ok((*split_1, *split_2))
        }
        _ => bail!(
            "{} / {} reviewed splits {splits:?} must contain two increasing boundaries inside 0..{layer_count}",
            row.llama_model,
            row.family
        ),
    }
}

fn split_layers(layer_count: u32) -> Result<(u32, u32)> {
    if layer_count < 3 {
        bail!("parity activation handoff requires at least three layers, got {layer_count}");
    }
    let split_1 = (layer_count / 3).max(1);
    let split_2 = ((layer_count * 2) / 3)
        .max(split_1 + 1)
        .min(layer_count - 1);
    Ok((split_1, split_2))
}

fn cache_stage_range(row: &CandidateRow, layout: &TestLayout) -> Result<(u32, u32)> {
    if row.is_package_only() {
        if layout.layer_count == 0 {
            bail!("package-only cache smoke requires at least one layer");
        }
        return Ok((0, 1));
    }
    let (split_1, split_2) = split_layers_for(row, layout.layer_count)?;
    Ok((split_1, split_2))
}

fn run_lightweight_activation_smoke(layout: &TestLayout, spec: FamilySpec) -> Result<()> {
    let n_gpu_layers = case_n_gpu_layers(manifest_row(spec)?);
    if layout.layer_count < 2 {
        bail!("package activation smoke requires at least two layers");
    }
    let stage0_shape = StageShape {
        stage_index: 0,
        layer_start: 0,
        layer_end: 1,
        include_embeddings: true,
        include_output: false,
    };
    let stage1_shape = StageShape {
        stage_index: 1,
        layer_start: 1,
        layer_end: 2,
        include_embeddings: false,
        include_output: false,
    };
    let stage0 = open_stage_model(
        &stage_path(layout, spec, stage0_shape)?,
        stage0_shape,
        n_gpu_layers,
    )?;
    let stage1 = open_stage_model(
        &stage_path(layout, spec, stage1_shape)?,
        stage1_shape,
        n_gpu_layers,
    )?;
    let token = first_prompt_token(&stage0)?;
    let mut session0 = stage0.create_session()?;
    let (_, activation0) = session0.decode_step_frame(token, None, 0)?;
    ensure_activation_payload("package stage 0", &activation0)?;
    let mut session1 = stage1.create_session()?;
    let (_, activation1) = session1.decode_step_frame(token, Some(&activation0), 0)?;
    ensure_activation_payload("package stage 1", &activation1)?;
    Ok(())
}

fn run_correctness_chain(layout: &TestLayout, spec: FamilySpec, splits: (u32, u32)) -> Result<()> {
    let binary = std::env::var("CARGO_BIN_EXE_skippy-correctness")
        .unwrap_or_else(|_| "skippy-correctness".to_string());
    let stage_server_bin = std::env::var_os("SKIPPY_STAGE_SERVER_BIN")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/debug/skippy-server")
        });
    if !stage_server_bin.is_file() {
        bail!(
            "skippy-server binary is required for activation chain tests; build it with `cargo build -p skippy-server` or set SKIPPY_STAGE_SERVER_BIN"
        );
    }
    let model_id = format!(
        "meshllm/{}",
        sanitize_model_id_part(&format!("{}-{}", spec.llama_model, spec.family))
    );
    let output = Command::new(&binary)
        .args([
            "chain",
            "--model",
            &layout.full_model.path.to_string_lossy(),
            "--model-id",
            &model_id,
            "--splits",
            &format!("{},{}", splits.0, splits.1),
            "--layer-end",
            &layout.layer_count.to_string(),
            "--n-gpu-layers",
            &case_n_gpu_layers(manifest_row(spec)?).to_string(),
            "--stage-server-bin",
            &stage_server_bin.to_string_lossy(),
            "--prompt",
            case_prompt(manifest_row(spec)?),
            "--activation-wire-dtype",
            case_wire_dtype(manifest_row(spec)?),
        ])
        .output()
        .with_context(|| {
            format!(
                "run {binary} chain for {} / {}",
                spec.llama_model, spec.family
            )
        })?;
    if !output.status.success() {
        bail!(
            "{} / {} correctness chain failed with status {}\nstdout:\n{}\nstderr:\n{}",
            spec.llama_model,
            spec.family,
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    let report: Value = serde_json::from_slice(&output.stdout).with_context(|| {
        format!(
            "parse correctness chain report for {} / {}",
            spec.llama_model, spec.family
        )
    })?;
    if report.get("matches").and_then(Value::as_bool) != Some(true) {
        bail!(
            "{} / {} correctness chain did not match baseline: {}",
            spec.llama_model,
            spec.family,
            report
        );
    }
    Ok(())
}

fn case_prompt(row: &CandidateRow) -> &str {
    row.prompt.as_deref().unwrap_or("Hello")
}

fn case_wire_dtype(row: &CandidateRow) -> &str {
    row.wire_dtype.as_deref().unwrap_or("f16")
}

fn case_n_gpu_layers(row: &CandidateRow) -> i32 {
    row.n_gpu_layers.unwrap_or_else(parity_n_gpu_layers)
}

fn sanitize_model_id_part(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.') {
                ch
            } else {
                '-'
            }
        })
        .collect()
}

#[derive(Clone, Copy)]
struct StageShape {
    stage_index: u32,
    layer_start: u32,
    layer_end: u32,
    include_embeddings: bool,
    include_output: bool,
}

fn stage_path(layout: &TestLayout, spec: FamilySpec, shape: StageShape) -> Result<StagePath> {
    if matches!(layout.full_model.load_mode, RuntimeLoadMode::RuntimeSlice) {
        return Ok(StagePath {
            path: layout.full_model.path.clone(),
            load_mode: RuntimeLoadMode::RuntimeSlice,
            filter_tensors_on_load: true,
        });
    }
    let package_dir = layout
        .package_dir
        .as_deref()
        .ok_or_else(|| anyhow!("layer-package stage requested without a package dir"))?;
    materialize_stage(
        package_dir,
        manifest_row(spec)?,
        shape.layer_start,
        shape.layer_end,
        shape.include_embeddings,
        shape.include_output,
    )
}

fn materialize_stage(
    package_dir: &Path,
    row: &CandidateRow,
    layer_start: u32,
    layer_end: u32,
    include_embeddings: bool,
    include_output: bool,
) -> Result<StagePath> {
    let materialized = materialize_layer_package_details(&PackageStageRequest {
        model_id: format!("{}:{}", row.repo, row.family),
        topology_id: "parity-model-tests".to_string(),
        package_ref: package_dir.to_string_lossy().into_owned(),
        stage_id: format!("stage-{layer_start}-{layer_end}"),
        layer_start,
        layer_end,
        include_embeddings,
        include_output,
    })?;
    Ok(StagePath {
        path: materialized.output_path,
        load_mode: RuntimeLoadMode::LayerPackage,
        filter_tensors_on_load: true,
    })
}

fn run_stage_state_restore(
    layout: &TestLayout,
    spec: FamilySpec,
    stage_start: u32,
    stage_end: u32,
    state_kind: StateKind,
) -> Result<()> {
    let n_gpu_layers = case_n_gpu_layers(manifest_row(spec)?);
    let input_shape = StageShape {
        stage_index: 0,
        layer_start: 0,
        layer_end: stage_start,
        include_embeddings: true,
        include_output: false,
    };
    let target_shape = StageShape {
        stage_index: 1,
        layer_start: stage_start,
        layer_end: stage_end,
        include_embeddings: stage_start == 0,
        include_output: stage_end == layout.layer_count,
    };
    let target = open_stage_model(
        &stage_path(layout, spec, target_shape)?,
        target_shape,
        n_gpu_layers,
    )?;
    let token_source = if stage_start == 0 {
        &target
    } else {
        &open_stage_model(
            &stage_path(layout, spec, input_shape)?,
            input_shape,
            n_gpu_layers,
        )?
    };
    let mut tokens = token_source.tokenize(PROMPT, true)?;
    if tokens.len() < 2 {
        bail!("prompt produced fewer than two tokens");
    }
    tokens.truncate(2);
    let prefix = vec![tokens[0]];
    let continuation = tokens[1];
    let (prefill_input, decode_input) = if stage_start == 0 {
        (None, None)
    } else {
        let input = open_stage_model(
            &stage_path(layout, spec, input_shape)?,
            input_shape,
            n_gpu_layers,
        )?;
        let mut input_session = input.create_session()?;
        let prefill = input_session.prefill_chunk_frame(&prefix, None, 0)?;
        let (_, decode) = input_session.decode_step_frame(continuation, None, 0)?;
        (Some(prefill), Some(decode))
    };

    let mut source = target.create_session()?;
    source.prefill_chunk_frame(&prefix, prefill_input.as_ref(), 0)?;
    let payload = export_state_payload(&mut source, state_kind, stage_start, stage_end, 1)?;
    let (source_predicted, source_frame) =
        source.decode_step_frame(continuation, decode_input.as_ref(), 0)?;

    let mut restored = target.create_session()?;
    import_state_payload(
        &mut restored,
        &payload,
        state_kind,
        stage_start,
        stage_end,
        1,
        &prefix,
    )?;
    let (restored_predicted, restored_frame) =
        restored.decode_step_frame(continuation, decode_input.as_ref(), 0)?;
    if source_predicted != restored_predicted {
        bail!(
            "{} / {} restored token {restored_predicted} did not match source {source_predicted}",
            spec.llama_model,
            spec.family
        );
    }
    if !activation_frames_match(&source_frame, &restored_frame) {
        bail!(
            "{} / {} restored activation payload did not match recompute",
            spec.llama_model,
            spec.family
        );
    }
    let suffix_matches = verify_suffix_prefill_after_restore(
        &target,
        &payload,
        state_kind,
        stage_start,
        stage_end,
        &prefix,
        continuation,
        prefill_input.as_ref(),
        decode_input.as_ref(),
        target_shape.include_output,
    )?;
    if !suffix_matches {
        bail!(
            "{} / {} suffix prefill after cache restore did not match recompute",
            spec.llama_model,
            spec.family
        );
    }
    Ok(())
}

#[derive(Clone, Copy)]
enum StateKind {
    ResidentKv,
    KvRecurrent,
}

#[derive(Clone)]
enum StatePayload {
    ResidentKv,
    KvRecurrent {
        kv: Option<RuntimeKvPage>,
        recurrent: Vec<u8>,
    },
}

#[allow(clippy::too_many_arguments)]
fn verify_suffix_prefill_after_restore(
    target: &StageModel,
    payload: &StatePayload,
    state_kind: StateKind,
    stage_start: u32,
    stage_end: u32,
    prefix: &[i32],
    continuation: i32,
    prefill_input: Option<&ActivationFrame>,
    decode_input: Option<&ActivationFrame>,
    include_output: bool,
) -> Result<bool> {
    let mut source = target.create_session()?;
    source.prefill_chunk_frame(prefix, prefill_input, 0)?;
    let (source_predicted, source_frame) = if include_output {
        let (predicted, frame) = source.verify_tokens_frame(&[continuation], decode_input, 0)?;
        (Some(predicted), frame)
    } else {
        (
            None,
            source.prefill_chunk_frame(&[continuation], decode_input, 0)?,
        )
    };

    let mut restored = target.create_session()?;
    import_state_payload(
        &mut restored,
        payload,
        state_kind,
        stage_start,
        stage_end,
        1,
        prefix,
    )?;
    let (restored_predicted, restored_frame) = if include_output {
        let (predicted, frame) = restored.verify_tokens_frame(&[continuation], decode_input, 0)?;
        (Some(predicted), frame)
    } else {
        (
            None,
            restored.prefill_chunk_frame(&[continuation], decode_input, 0)?,
        )
    };

    Ok(source_predicted == restored_predicted
        && activation_frames_match(&source_frame, &restored_frame))
}

fn activation_frames_match(source: &ActivationFrame, restored: &ActivationFrame) -> bool {
    if source.desc.token_count != restored.desc.token_count
        || source.desc.flags != restored.desc.flags
        || source.payload.len() != restored.payload.len()
    {
        return false;
    }
    if source.payload == restored.payload {
        return true;
    }
    if source.payload.len() % 4 != 0 {
        return false;
    }

    source
        .payload
        .chunks_exact(4)
        .zip(restored.payload.chunks_exact(4))
        .all(|(lhs, rhs)| {
            let lhs = f32::from_le_bytes(lhs.try_into().expect("chunks_exact"));
            let rhs = f32::from_le_bytes(rhs.try_into().expect("chunks_exact"));
            if lhs.to_bits() == rhs.to_bits() {
                return true;
            }
            if !lhs.is_finite() || !rhs.is_finite() {
                return false;
            }
            let diff = (lhs - rhs).abs();
            let scale = lhs.abs().max(rhs.abs()).max(1.0);
            diff <= 1.0e-3 || diff / scale <= 1.0e-3
        })
}

fn export_state_payload(
    session: &mut StageSession,
    state_kind: StateKind,
    layer_start: u32,
    layer_end: u32,
    token_count: u64,
) -> Result<StatePayload> {
    match state_kind {
        StateKind::ResidentKv => {
            session.save_prefix(CACHE_SEQ_ID, token_count)?;
            Ok(StatePayload::ResidentKv)
        }
        StateKind::KvRecurrent => {
            let kv = match session.export_kv_page(
                layer_start as i32,
                layer_end as i32,
                0,
                token_count,
            ) {
                Ok(page) => Some(page),
                Err(error) if native_kv_unavailable(&error) => None,
                Err(error) => return Err(error),
            };
            let recurrent = session.export_recurrent_state()?;
            if recurrent.is_empty() {
                bail!("KvRecurrent family exported empty recurrent state");
            }
            Ok(StatePayload::KvRecurrent { kv, recurrent })
        }
    }
}

fn import_state_payload(
    session: &mut StageSession,
    payload: &StatePayload,
    state_kind: StateKind,
    layer_start: u32,
    layer_end: u32,
    token_count: u64,
    token_ids: &[i32],
) -> Result<()> {
    match (state_kind, payload) {
        (StateKind::ResidentKv, StatePayload::ResidentKv) => {
            session.restore_prefix(CACHE_SEQ_ID, token_ids)
        }
        (StateKind::KvRecurrent, StatePayload::KvRecurrent { kv, recurrent }) => {
            if let Some(kv) = kv {
                session.import_kv_page(&kv.desc, &kv.payload)?;
            }
            session.import_recurrent_state_for_token_count(recurrent, token_count)
        }
        _ => bail!("state payload kind mismatch"),
    }
    .with_context(|| {
        format!(
            "import state payload for layers {layer_start}..{layer_end} and {token_count} tokens"
        )
    })
}

fn native_kv_unavailable(error: &anyhow::Error) -> bool {
    error.chain().any(|cause| {
        let message = cause.to_string();
        message.contains("runtime memory type is not supported for native KV pages")
            || message.contains("runtime has no attention KV cache")
            || message.contains("no KV cache layers selected by layer range")
    })
}

fn open_stage_model(path: &StagePath, shape: StageShape, n_gpu_layers: i32) -> Result<StageModel> {
    StageModel::open(
        &path.path,
        &RuntimeConfig {
            stage_index: shape.stage_index,
            layer_start: shape.layer_start,
            layer_end: shape.layer_end,
            ctx_size: CTX_SIZE,
            lane_count: 4,
            n_batch: None,
            n_ubatch: None,
            n_threads: None,
            n_threads_batch: None,
            n_gpu_layers,
            selected_backend_device: None,
            cache_type_k: GGML_TYPE_F16,
            cache_type_v: GGML_TYPE_F16,
            flash_attn_type: skippy_runtime::FlashAttentionType::Auto,
            load_mode: path.load_mode,
            projector_path: None,
            include_embeddings: shape.include_embeddings,
            include_output: shape.include_output,
            filter_tensors_on_load: path.filter_tensors_on_load,
        },
    )
    .with_context(|| {
        format!(
            "open {} layers {}..{}",
            path.path.display(),
            shape.layer_start,
            shape.layer_end
        )
    })
}

fn first_prompt_token(model: &StageModel) -> Result<i32> {
    model
        .tokenize(PROMPT, true)?
        .into_iter()
        .next()
        .context("prompt produced no tokens")
}

fn ensure_activation_payload(label: &str, frame: &ActivationFrame) -> Result<()> {
    if frame.payload.is_empty() {
        bail!("{label} produced empty activation payload");
    }
    Ok(())
}

fn env_flag(name: &str, default: bool) -> bool {
    std::env::var(name)
        .ok()
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(default)
}

fn env_i32(name: &str) -> Option<i32> {
    std::env::var(name).ok()?.parse().ok()
}

fn parity_n_gpu_layers() -> i32 {
    env_i32("SKIPPY_PARITY_N_GPU_LAYERS").unwrap_or(999)
}
