use anyhow::{anyhow, bail, Context, Result};
pub use mesh_llm_gpu_bench::BenchmarkOutput;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use crate::hardware::HardwareSurvey;

#[cfg(test)]
use crate::hardware::GpuFacts;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GpuBandwidth {
    pub name: String,
    pub vram_bytes: u64,
    pub p50_gbps: f64,
    pub p90_gbps: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compute_tflops_fp32: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compute_tflops_fp16: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkFingerprint {
    pub gpus: Vec<GpuBandwidth>, // per-GPU identity + bandwidth, in device order
    pub is_soc: bool,
    pub timestamp_secs: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BenchmarkResult {
    pub mem_bandwidth_gbps: Vec<f64>,
    pub compute_tflops_fp32: Option<Vec<f64>>,
    pub compute_tflops_fp16: Option<Vec<f64>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SavedBenchmark {
    pub path: PathBuf,
    pub result: BenchmarkResult,
}

pub const BENCHMARK_TIMEOUT: Duration = Duration::from_secs(25);

const BENCHMARK_CHILD_ENV: &str = "MESH_LLM_BENCHMARK_CHILD";

fn benchmark_backend_name(backend: mesh_llm_gpu_bench::BenchmarkBackend) -> &'static str {
    match backend {
        mesh_llm_gpu_bench::BenchmarkBackend::Metal => "metal",
        mesh_llm_gpu_bench::BenchmarkBackend::Cuda => "cuda",
        mesh_llm_gpu_bench::BenchmarkBackend::Hip => "hip",
        mesh_llm_gpu_bench::BenchmarkBackend::Intel => "intel",
    }
}

fn parse_benchmark_backend(name: &str) -> Option<mesh_llm_gpu_bench::BenchmarkBackend> {
    if name.eq_ignore_ascii_case("metal") {
        Some(mesh_llm_gpu_bench::BenchmarkBackend::Metal)
    } else if name.eq_ignore_ascii_case("cuda") {
        Some(mesh_llm_gpu_bench::BenchmarkBackend::Cuda)
    } else if name.eq_ignore_ascii_case("hip") {
        Some(mesh_llm_gpu_bench::BenchmarkBackend::Hip)
    } else if name.eq_ignore_ascii_case("intel") {
        Some(mesh_llm_gpu_bench::BenchmarkBackend::Intel)
    } else {
        None
    }
}

fn benchmark_marker_name(backend: mesh_llm_gpu_bench::BenchmarkBackend) -> String {
    format!("mesh-llm-benchmark-{}", benchmark_backend_name(backend))
}

fn parse_benchmark_backend_from_path(
    binary: &Path,
) -> Option<mesh_llm_gpu_bench::BenchmarkBackend> {
    let raw = binary.file_name()?.to_string_lossy();
    if let Some(name) = raw.strip_prefix("mesh-llm-benchmark-") {
        return parse_benchmark_backend(name);
    }

    let raw = binary.to_string_lossy();
    if let Some(name) = raw.strip_prefix("in-process:") {
        return parse_benchmark_backend(name);
    }

    None
}

fn benchmark_child_path(bin_dir: &Path) -> PathBuf {
    if let Some(path) = std::env::var_os(BENCHMARK_CHILD_ENV) {
        return PathBuf::from(path);
    }

    let mesh_binary = if cfg!(windows) {
        "mesh-llm.exe"
    } else {
        "mesh-llm"
    };
    bin_dir.join(mesh_binary)
}

fn run_benchmark_subprocess(binary: &Path, timeout: Duration) -> Result<Vec<BenchmarkOutput>> {
    let backend = parse_benchmark_backend_from_path(binary)
        .with_context(|| format!("unknown benchmark runner marker {}", binary.display()))?;
    let backend_name = benchmark_backend_name(backend);
    let child_path = benchmark_child_path(binary.parent().unwrap_or_else(|| Path::new(".")));

    let mut child = Command::new(&child_path)
        .args(["benchmark", "run-gpu", "--backend", backend_name])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| format!("failed to start benchmark child {}", child_path.display()))?;

    let started = Instant::now();
    while child.try_wait()?.is_none() {
        if started.elapsed() >= timeout {
            let _ = child.kill();
            let output = child.wait_with_output()?;
            let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
            if stderr.is_empty() {
                bail!("benchmark timed out after {:.1}s", timeout.as_secs_f64());
            }
            bail!(
                "benchmark timed out after {:.1}s: {stderr}",
                timeout.as_secs_f64()
            );
        }
        thread::sleep(Duration::from_millis(25));
    }

    let output = child.wait_with_output()?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        if stderr.is_empty() {
            bail!("benchmark child exited with status {}", output.status);
        }
        bail!("benchmark child failed: {stderr}");
    }

    parse_benchmark_output(&output.stdout)
        .ok_or_else(|| anyhow!("benchmark child returned invalid output"))
}

pub fn run_backend_by_name(backend: &str) -> Result<Vec<BenchmarkOutput>> {
    let backend = parse_benchmark_backend(backend)
        .with_context(|| format!("unsupported benchmark backend {backend}"))?;
    mesh_llm_gpu_bench::run_benchmark(
        mesh_llm_gpu_bench::BenchmarkRunner { backend },
        BENCHMARK_TIMEOUT,
    )
}

/// Normalize `HardwareSurvey.gpu_name` into a per-GPU list of names.
/// - Splits on ',' and trims whitespace for robustness.
/// - Expands summarized forms like "8× NVIDIA A100" into 8 identical entries.
/// - If the expanded list length does not match `gpu_vram.len()` but `gpu_vram` is
///   non-empty, falls back to assuming all GPUs share the same summarized name and
///   returns `gpu_vram.len()` copies of it.
fn per_gpu_names(hw: &HardwareSurvey) -> Vec<String> {
    let raw = match hw.gpu_name.as_deref() {
        Some(s) => s.trim(),
        None => return Vec::new(),
    };

    if raw.is_empty() {
        return Vec::new();
    }

    let mut names: Vec<String> = Vec::new();

    for part in raw.split(',') {
        let part_trimmed = part.trim();
        if part_trimmed.is_empty() {
            continue;
        }

        // Handle summarized "N× name" form (e.g., "8× NVIDIA A100").
        if let Some((count_str, name)) = part_trimmed.split_once('×') {
            if let Ok(count) = count_str.trim().parse::<usize>() {
                let name_trimmed = name.trim();
                for _ in 0..count {
                    names.push(name_trimmed.to_string());
                }
                continue;
            }
        }

        // Fallback: treat as a single GPU name.
        names.push(part_trimmed.to_string());
    }

    if names.len() == hw.gpu_vram.len() || hw.gpu_vram.is_empty() {
        return names;
    }

    // As a last resort, assume all GPUs share the same summarized name.
    let gpu_count = hw.gpu_vram.len();
    vec![raw.to_string(); gpu_count]
}

/// Returns true if the current hardware differs from the fingerprint's recorded hardware.
/// Compares GPU names, VRAM sizes (by index), and the is_soc flag.
pub fn hardware_changed(fingerprint: &BenchmarkFingerprint, hw: &HardwareSurvey) -> bool {
    if fingerprint.is_soc != hw.is_soc {
        return true;
    }

    let hw_names: Vec<String> = per_gpu_names(hw);

    if fingerprint.gpus.len() != hw_names.len() || fingerprint.gpus.len() != hw.gpu_vram.len() {
        return true;
    }

    for (i, cached) in fingerprint.gpus.iter().enumerate() {
        if cached.name != hw_names[i] || cached.vram_bytes != hw.gpu_vram[i] {
            return true;
        }
    }
    false
}

/// Returns the cache-backed benchmark fingerprint path, usually
/// `~/.cache/mesh-llm/benchmark-fingerprint.json`.
/// Falls back to `~/.cache` and then the platform temp directory if needed.
pub fn fingerprint_path() -> PathBuf {
    dirs::cache_dir()
        .or_else(|| dirs::home_dir().map(|home| home.join(".cache")))
        .unwrap_or_else(std::env::temp_dir)
        .join("mesh-llm")
        .join("benchmark-fingerprint.json")
}

/// Load a `BenchmarkFingerprint` from disk.  Returns `None` on any error.
pub fn load_fingerprint(path: &Path) -> Option<BenchmarkFingerprint> {
    let content = std::fs::read_to_string(path).ok()?;
    serde_json::from_str(&content).ok()
}

/// Atomically write a `BenchmarkFingerprint` to disk.
/// Uses a `.json.tmp` staging file + rename for crash safety.
/// Logs a warning on failure — never panics.
pub fn save_fingerprint(path: &Path, fp: &BenchmarkFingerprint) {
    if let Err(err) = try_save_fingerprint(path, fp) {
        tracing::warn!("benchmark: failed to persist fingerprint: {err}");
    }
}

pub fn try_save_fingerprint(path: &Path, fp: &BenchmarkFingerprint) -> Result<()> {
    let tmp = path.with_extension("json.tmp");

    std::fs::create_dir_all(path.parent().unwrap_or_else(|| Path::new(".")))
        .with_context(|| format!("failed to create cache dir for {}", path.display()))?;

    let json =
        serde_json::to_string_pretty(fp).context("failed to serialize benchmark fingerprint")?;

    std::fs::write(&tmp, &json)
        .with_context(|| format!("failed to write temporary fingerprint {}", tmp.display()))?;

    // On Windows, `rename` fails if the destination already exists.
    // Remove the destination first there; on Unix the rename stays atomic.
    #[cfg(windows)]
    if path.exists() {
        std::fs::remove_file(path)
            .with_context(|| format!("failed to remove existing fingerprint {}", path.display()))?;
    }

    if let Err(e) = std::fs::rename(&tmp, path) {
        let _ = std::fs::remove_file(&tmp);
        return Err(e).with_context(|| {
            format!(
                "failed to rename fingerprint into place at {}",
                path.display()
            )
        });
    }

    Ok(())
}

/// Determine whether this hardware maps to a benchmark backend.
pub fn detect_benchmark_binary(hw: &HardwareSurvey, bin_dir: &Path) -> Option<PathBuf> {
    let runner = mesh_llm_gpu_bench::runner_for(
        std::env::consts::OS,
        hw.gpu_count,
        hw.gpu_name.as_deref(),
        hw.is_soc,
    )?;
    Some(bin_dir.join(benchmark_marker_name(runner.backend)))
}

/// Parse raw stdout bytes from a benchmark run into a vec of per-device outputs.
///
/// Expects a JSON array of [`BenchmarkOutput`].  Returns `None` on any parse
/// failure or if the device list is empty.
pub fn parse_benchmark_output(stdout: &[u8]) -> Option<Vec<BenchmarkOutput>> {
    mesh_llm_gpu_bench::parse_benchmark_output(stdout)
}

/// Run an in-process benchmark backend and return per-device outputs.
pub fn run_benchmark(binary: &Path, timeout: Duration) -> Option<Vec<BenchmarkOutput>> {
    run_benchmark_subprocess(binary, timeout)
        .map_err(|err| tracing::warn!("benchmark failed: {err:#}"))
        .ok()
}

fn run_backend_for_hardware(
    hw: &HardwareSurvey,
    bin_dir: &Path,
    timeout: Duration,
) -> Result<Vec<BenchmarkOutput>> {
    let runner = detect_benchmark_binary(hw, bin_dir).with_context(|| {
        format!(
            "no supported benchmark backend found for detected GPU platform {:?}",
            hw.gpu_name
        )
    })?;

    run_benchmark_subprocess(&runner, timeout)
}

/// Load a cached fingerprint if hardware is unchanged, otherwise run the
/// compiled benchmark backend and persist the result.
///
/// Not `async` — intended for use inside `tokio::task::spawn_blocking`.
pub fn run_or_load(
    hw: &HardwareSurvey,
    bin_dir: &Path,
    timeout: Duration,
) -> Option<BenchmarkResult> {
    let path = fingerprint_path();

    // Cache-hit path
    if let Some(ref cached) = load_fingerprint(&path) {
        if !hardware_changed(cached, hw) {
            let mem_bandwidth: Vec<f64> = cached.gpus.iter().map(|g| g.p90_gbps).collect();
            let compute_tflops_fp32 = cached
                .gpus
                .iter()
                .map(|g| g.compute_tflops_fp32)
                .collect::<Option<Vec<f64>>>();
            let compute_tflops_fp16 = cached
                .gpus
                .iter()
                .map(|g| g.compute_tflops_fp16)
                .collect::<Option<Vec<f64>>>();
            let result = BenchmarkResult {
                mem_bandwidth_gbps: mem_bandwidth,
                compute_tflops_fp32,
                compute_tflops_fp16,
            };
            tracing::info!(
                "Using cached bandwidth fingerprint: {} GPUs",
                result.mem_bandwidth_gbps.len()
            );
            return Some(result);
        }
    }

    tracing::info!("Hardware changed or no cache — running memory bandwidth benchmark");

    let outputs = run_backend_for_hardware(hw, bin_dir, timeout)
        .map_err(|err| tracing::warn!("benchmark failed: {err:#}"))
        .ok()?;

    let (gpus, result) = build_benchmark_result(hw, &outputs);

    let fingerprint = BenchmarkFingerprint {
        gpus,
        is_soc: hw.is_soc,
        timestamp_secs: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
    };

    save_fingerprint(&path, &fingerprint);
    Some(result)
}

pub fn run_and_save(
    hw: &HardwareSurvey,
    bin_dir: &Path,
    timeout: Duration,
) -> Result<SavedBenchmark> {
    run_and_save_to_path(hw, bin_dir, timeout, &fingerprint_path())
}

fn run_and_save_to_path(
    hw: &HardwareSurvey,
    bin_dir: &Path,
    timeout: Duration,
    path: &Path,
) -> Result<SavedBenchmark> {
    if hw.gpu_count == 0 {
        bail!("no GPUs detected on this node");
    }

    let outputs = run_backend_for_hardware(hw, bin_dir, timeout)?;

    let result = save_result_from_outputs(path, hw, &outputs)?;
    Ok(SavedBenchmark {
        path: path.to_path_buf(),
        result,
    })
}

fn save_result_from_outputs(
    path: &Path,
    hw: &HardwareSurvey,
    outputs: &[BenchmarkOutput],
) -> Result<BenchmarkResult> {
    let (gpus, result) = build_benchmark_result(hw, outputs);

    let fingerprint = BenchmarkFingerprint {
        gpus,
        is_soc: hw.is_soc,
        timestamp_secs: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
    };

    try_save_fingerprint(path, &fingerprint)?;
    Ok(result)
}

fn build_benchmark_result(
    hw: &HardwareSurvey,
    outputs: &[BenchmarkOutput],
) -> (Vec<GpuBandwidth>, BenchmarkResult) {
    let hw_names = per_gpu_names(hw);

    let count = outputs
        .len()
        .min(hw.gpu_vram.len())
        .min(if hw_names.is_empty() {
            usize::MAX
        } else {
            hw_names.len()
        });

    let gpus: Vec<GpuBandwidth> = (0..count)
        .map(|i| GpuBandwidth {
            name: hw_names.get(i).cloned().unwrap_or_default(),
            vram_bytes: hw.gpu_vram.get(i).copied().unwrap_or(0),
            p50_gbps: outputs[i].p50_gbps,
            p90_gbps: outputs[i].p90_gbps,
            compute_tflops_fp32: outputs[i].compute_tflops_fp32,
            compute_tflops_fp16: outputs[i].compute_tflops_fp16,
        })
        .collect();

    let mem_bandwidth_gbps = gpus.iter().map(|g| g.p90_gbps).collect();
    let compute_tflops_fp32 = gpus
        .iter()
        .map(|g| g.compute_tflops_fp32)
        .collect::<Option<Vec<f64>>>();
    let compute_tflops_fp16 = gpus
        .iter()
        .map(|g| g.compute_tflops_fp16)
        .collect::<Option<Vec<f64>>>();

    (
        gpus,
        BenchmarkResult {
            mem_bandwidth_gbps,
            compute_tflops_fp32,
            compute_tflops_fp16,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    #[cfg(unix)]
    use std::os::unix::fs::PermissionsExt;

    fn make_survey(
        gpu_count: u8,
        gpu_vram: Vec<u64>,
        gpu_name: Option<&str>,
        is_soc: bool,
    ) -> HardwareSurvey {
        HardwareSurvey {
            gpu_count,
            gpu_vram,
            gpu_name: gpu_name.map(str::to_owned),
            is_soc,
            ..Default::default()
        }
    }

    fn make_fingerprint(gpus: Vec<GpuBandwidth>, is_soc: bool) -> BenchmarkFingerprint {
        BenchmarkFingerprint {
            gpus,
            is_soc,
            timestamp_secs: 0,
        }
    }

    fn build_output(fp32: Option<f64>, fp16: Option<f64>) -> BenchmarkOutput {
        BenchmarkOutput {
            device: "Test GPU".into(),
            buffer_mb: 0,
            runs: 0,
            p50_gbps: 1.0,
            p90_gbps: 2.0,
            compute_tflops_fp32: fp32,
            compute_tflops_fp16: fp16,
            noise_pct: 0.0,
            runtime_s: 0.0,
            rated_gbps: None,
            rated_estimated: None,
            efficiency_pct: None,
            bus_width_bits: None,
            mem_clock_mhz: None,
            gcn_arch: None,
            hbm: None,
        }
    }

    fn with_benchmark_child_override<T>(path: &Path, f: impl FnOnce() -> T) -> T {
        std::env::set_var(BENCHMARK_CHILD_ENV, path);
        let result = f();
        std::env::remove_var(BENCHMARK_CHILD_ENV);
        result
    }

    #[cfg(unix)]
    fn write_test_child(root: &Path, name: &str, body: &str) -> PathBuf {
        let path = root.join(name);
        let script = format!("#!/bin/sh\nset -eu\n{body}\n");
        std::fs::write(&path, script).expect("write test child");
        let mut perms = std::fs::metadata(&path).expect("metadata").permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&path, perms).expect("chmod test child");
        path
    }

    #[cfg(windows)]
    fn write_test_child(root: &Path, name: &str, body: &str) -> PathBuf {
        let path = root.join(name);
        let script = format!("@echo off\r\n{body}\r\n");
        std::fs::write(&path, script).expect("write test child");
        path
    }

    fn make_hw_with_gpus() -> HardwareSurvey {
        HardwareSurvey {
            gpu_vram: vec![64_000_000_000],
            gpu_name: Some("Test GPU".into()),
            gpu_count: 1,
            is_soc: false,
            gpus: vec![GpuFacts {
                index: 0,
                display_name: "Test GPU".into(),
                backend_device: None,
                vram_bytes: 64_000_000_000,
                reserved_bytes: None,
                mem_bandwidth_gbps: None,
                compute_tflops_fp32: None,
                compute_tflops_fp16: None,
                unified_memory: false,
                stable_id: None,
                pci_bdf: None,
                vendor_uuid: None,
                metal_registry_id: None,
                dxgi_luid: None,
                pnp_instance_id: None,
            }],
            ..Default::default()
        }
    }

    // 1. Same hardware → false
    #[test]
    fn test_hardware_changed_same() {
        let hw = make_survey(1, vec![80_000_000_000], Some("A100"), false);
        let fp = make_fingerprint(
            vec![GpuBandwidth {
                name: "A100".into(),
                vram_bytes: 80_000_000_000,
                p50_gbps: 1935.0,
                p90_gbps: 1948.7,
                compute_tflops_fp32: None,
                compute_tflops_fp16: None,
            }],
            false,
        );
        assert!(!hardware_changed(&fp, &hw));
    }

    // 2. VRAM differs → true
    #[test]
    fn test_hardware_changed_vram() {
        let hw = make_survey(1, vec![40_000_000_000], Some("A100"), false);
        let fp = make_fingerprint(
            vec![GpuBandwidth {
                name: "A100".into(),
                vram_bytes: 80_000_000_000,
                p50_gbps: 1935.0,
                p90_gbps: 1948.7,
                compute_tflops_fp32: None,
                compute_tflops_fp16: None,
            }],
            false,
        );
        assert!(hardware_changed(&fp, &hw));
    }

    // 3. GPU count differs → true
    #[test]
    fn test_hardware_changed_gpu_count() {
        let hw = make_survey(
            2,
            vec![80_000_000_000, 80_000_000_000],
            Some("A100, A100"),
            false,
        );
        let fp = make_fingerprint(
            vec![GpuBandwidth {
                name: "A100".into(),
                vram_bytes: 80_000_000_000,
                p50_gbps: 1935.0,
                p90_gbps: 1948.7,
                compute_tflops_fp32: None,
                compute_tflops_fp16: None,
            }],
            false,
        );
        assert!(hardware_changed(&fp, &hw));
    }

    // 4. is_soc differs → true
    #[test]
    fn test_hardware_changed_soc_flag() {
        let hw = make_survey(1, vec![16_000_000_000], None, false);
        let fp = make_fingerprint(vec![], true); // is_soc: true vs false
        assert!(hardware_changed(&fp, &hw));
    }

    // 5. Parse single CUDA GPU JSON — assert p90_gbps == 1948.7
    #[test]
    fn test_benchmark_output_deserialize_cuda_single() {
        let json_str = r#"[{"device":"NVIDIA A100-SXM4-80GB","buffer_mb":512,"runs":20,"p50_gbps":1935.2,"p90_gbps":1948.7,"compute_tflops_fp32":19.5,"compute_tflops_fp16":312.0,"noise_pct":0.4,"runtime_s":1.23,"rated_gbps":2000,"rated_estimated":false,"efficiency_pct":96.8,"bus_width_bits":5120,"mem_clock_mhz":1215}]"#;
        let outputs: Vec<BenchmarkOutput> = serde_json::from_str(json_str).expect("should parse");
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].p90_gbps, 1948.7);
        assert_eq!(outputs[0].compute_tflops_fp32, Some(19.5));
        assert_eq!(outputs[0].compute_tflops_fp16, Some(312.0));
    }

    // 6. Parse 2-device JSON — assert both entries deserialize
    #[test]
    fn test_benchmark_output_deserialize_multi_gpu() {
        let json_str = r#"[{"device":"NVIDIA A100","buffer_mb":512,"runs":20,"p50_gbps":1935.2,"p90_gbps":1948.7,"compute_tflops_fp32":19.5,"compute_tflops_fp16":312.0,"noise_pct":0.4,"runtime_s":1.23,"rated_gbps":2000,"rated_estimated":false,"efficiency_pct":96.8,"bus_width_bits":5120,"mem_clock_mhz":1215},{"device":"NVIDIA A6000","buffer_mb":512,"runs":20,"p50_gbps":768.0,"p90_gbps":780.1,"compute_tflops_fp32":38.7,"compute_tflops_fp16":77.4,"noise_pct":0.6,"runtime_s":1.15,"rated_gbps":768,"rated_estimated":false,"efficiency_pct":100.0,"bus_width_bits":384,"mem_clock_mhz":2000}]"#;
        let outputs: Vec<BenchmarkOutput> = serde_json::from_str(json_str).expect("should parse");
        assert_eq!(outputs.len(), 2);
    }

    // 7. Error JSON (object, not array) → Err, no panic
    #[test]
    fn test_benchmark_output_deserialize_error_json() {
        let json_str = r#"{"error":"No CUDA-capable device found"}"#;
        let result = serde_json::from_str::<Vec<BenchmarkOutput>>(json_str);
        assert!(result.is_err(), "expected Err, got Ok");
    }

    // 8. parse_benchmark_output: single GPU → Some(vec with 1 entry, p90 == 1948.7)
    #[test]
    fn test_parse_benchmark_output_single_gpu() {
        let json = r#"[{"device":"NVIDIA A100-SXM4-80GB","buffer_mb":512,"runs":20,"p50_gbps":1935.2,"p90_gbps":1948.7,"compute_tflops_fp32":19.5,"compute_tflops_fp16":312.0,"noise_pct":0.4,"runtime_s":1.23,"rated_gbps":2000,"rated_estimated":false,"efficiency_pct":96.8,"bus_width_bits":5120,"mem_clock_mhz":1215}]"#;
        let result = parse_benchmark_output(json.as_bytes()).expect("should return Some");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].p90_gbps, 1948.7);
    }

    // 9. parse_benchmark_output: two GPUs → Some(vec with 2 entries), sum ~2728.8
    #[test]
    fn test_parse_benchmark_output_multi_gpu_sum() {
        let json = r#"[{"device":"NVIDIA A100","buffer_mb":512,"runs":20,"p50_gbps":1935.2,"p90_gbps":1948.7,"compute_tflops_fp32":19.5,"compute_tflops_fp16":312.0,"noise_pct":0.4,"runtime_s":1.23,"rated_gbps":2000,"rated_estimated":false,"efficiency_pct":96.8,"bus_width_bits":5120,"mem_clock_mhz":1215},{"device":"NVIDIA A6000","buffer_mb":512,"runs":20,"p50_gbps":768.0,"p90_gbps":780.1,"compute_tflops_fp32":38.7,"compute_tflops_fp16":77.4,"noise_pct":0.6,"runtime_s":1.15,"rated_gbps":768,"rated_estimated":false,"efficiency_pct":100.0,"bus_width_bits":384,"mem_clock_mhz":2000}]"#;
        let outputs = parse_benchmark_output(json.as_bytes()).expect("should return Some");
        assert_eq!(outputs.len(), 2);
        let sum: f64 = outputs.iter().map(|o| o.p90_gbps).sum();
        assert!(
            (sum - 2728.8_f64).abs() < 0.01,
            "expected ~2728.8, got {sum}"
        );
    }

    // 10. parse_benchmark_output: error object → None
    #[test]
    fn test_parse_benchmark_output_error_json() {
        let json = r#"{"error": "No CUDA devices found"}"#;
        let result = parse_benchmark_output(json.as_bytes());
        assert!(result.is_none());
    }

    // 11. parse_benchmark_output: empty array → None
    #[test]
    fn test_parse_benchmark_output_empty_array() {
        let result = parse_benchmark_output(b"[]");
        assert!(result.is_none());
    }

    // 12. detect_benchmark_binary: gpu_count == 0 → None (no process spawned)
    #[test]
    fn test_detect_benchmark_binary_gpu_count_zero() {
        let hw = HardwareSurvey {
            gpu_count: 0,
            ..Default::default()
        };
        let result = detect_benchmark_binary(&hw, Path::new("/tmp"));
        assert!(result.is_none());
    }

    #[test]
    fn test_runner_for_windows_cuda() {
        let hw = make_survey(1, vec![24_000_000_000], Some("NVIDIA RTX 4090"), false);
        let runner = mesh_llm_gpu_bench::runner_for(
            "windows",
            hw.gpu_count,
            hw.gpu_name.as_deref(),
            hw.is_soc,
        )
        .expect("CUDA runner");
        assert_eq!(runner.backend, mesh_llm_gpu_bench::BenchmarkBackend::Cuda);
    }

    #[test]
    fn test_runner_for_windows_hip() {
        let hw = make_survey(
            1,
            vec![24_000_000_000],
            Some("AMD Radeon RX 7900 XTX"),
            false,
        );
        let runner = mesh_llm_gpu_bench::runner_for(
            "windows",
            hw.gpu_count,
            hw.gpu_name.as_deref(),
            hw.is_soc,
        )
        .expect("HIP runner");
        assert_eq!(runner.backend, mesh_llm_gpu_bench::BenchmarkBackend::Hip);
    }

    #[test]
    fn test_runner_for_windows_intel() {
        let hw = make_survey(1, vec![16_000_000_000], Some("Intel Arc A770"), false);
        let runner = mesh_llm_gpu_bench::runner_for(
            "windows",
            hw.gpu_count,
            hw.gpu_name.as_deref(),
            hw.is_soc,
        );
        assert!(runner.is_none(), "Intel runner should be de-advertised");
    }

    #[test]
    fn test_runner_for_linux_cuda() {
        let hw = make_survey(1, vec![24_000_000_000], Some("NVIDIA RTX 4090"), false);
        let runner = mesh_llm_gpu_bench::runner_for(
            "linux",
            hw.gpu_count,
            hw.gpu_name.as_deref(),
            hw.is_soc,
        )
        .expect("CUDA runner");
        assert_eq!(runner.backend, mesh_llm_gpu_bench::BenchmarkBackend::Cuda);
    }

    #[test]
    fn test_runner_for_macos_soc() {
        let hw = make_survey(1, vec![24_000_000_000], Some("Apple M4 Pro"), true);
        let runner = mesh_llm_gpu_bench::runner_for(
            "macos",
            hw.gpu_count,
            hw.gpu_name.as_deref(),
            hw.is_soc,
        )
        .expect("Metal runner");
        assert_eq!(runner.backend, mesh_llm_gpu_bench::BenchmarkBackend::Metal);
    }

    // 13. hardware_changed: same VRAM, different GPU name → true
    #[test]
    fn test_hardware_changed_gpu_name() {
        let hw = make_survey(1, vec![80_000_000_000], Some("NVIDIA A6000"), false);
        let fp = make_fingerprint(
            vec![GpuBandwidth {
                name: "NVIDIA A100".into(),
                vram_bytes: 80_000_000_000,
                p50_gbps: 1935.0,
                p90_gbps: 1948.7,
                compute_tflops_fp32: None,
                compute_tflops_fp16: None,
            }],
            false,
        );
        assert!(
            hardware_changed(&fp, &hw),
            "name change should trigger hardware_changed"
        );
    }

    // 14. Cache round-trip: save → load → hardware_changed returns false for same hw
    #[test]
    fn test_fingerprint_cache_roundtrip() {
        let path = std::env::temp_dir().join("mesh-llm-test-fingerprint-roundtrip.json");
        let fp = make_fingerprint(
            vec![GpuBandwidth {
                name: "NVIDIA A100".into(),
                vram_bytes: 80_000_000_000,
                p50_gbps: 1935.2,
                p90_gbps: 1948.7,
                compute_tflops_fp32: Some(19.5),
                compute_tflops_fp16: Some(312.0),
            }],
            false,
        );
        save_fingerprint(&path, &fp);
        let loaded = load_fingerprint(&path).expect("fingerprint should round-trip");
        let _ = std::fs::remove_file(&path);

        let hw = make_survey(1, vec![80_000_000_000], Some("NVIDIA A100"), false);
        assert!(
            !hardware_changed(&loaded, &hw),
            "same hardware should not trigger hardware_changed after round-trip"
        );
    }

    #[test]
    fn test_try_save_fingerprint_overwrites_existing_cache() {
        let path = std::env::temp_dir().join("mesh-llm-test-fingerprint-overwrite.json");
        std::fs::write(&path, "stale").expect("seed existing cache");

        let fp = make_fingerprint(
            vec![GpuBandwidth {
                name: "NVIDIA A100".into(),
                vram_bytes: 80_000_000_000,
                p50_gbps: 1935.2,
                p90_gbps: 1948.7,
                compute_tflops_fp32: Some(19.5),
                compute_tflops_fp16: Some(312.0),
            }],
            false,
        );

        try_save_fingerprint(&path, &fp).expect("fingerprint should overwrite existing cache");
        let loaded = load_fingerprint(&path).expect("fingerprint should load after overwrite");
        let _ = std::fs::remove_file(&path);

        assert_eq!(loaded.gpus[0].p90_gbps, 1948.7);
    }

    #[test]
    fn test_save_result_from_outputs_rewrites_existing_cache() {
        let root = std::env::temp_dir().join(format!(
            "mesh-llm-run-and-save-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system time")
                .as_nanos()
        ));
        let path = root.join("benchmark-fingerprint.json");
        std::fs::create_dir_all(&root).expect("create test dir");

        let old = make_fingerprint(
            vec![GpuBandwidth {
                name: "Test GPU".into(),
                vram_bytes: 64_000_000_000,
                p50_gbps: 1.0,
                p90_gbps: 2.0,
                compute_tflops_fp32: None,
                compute_tflops_fp16: None,
            }],
            cfg!(target_os = "macos"),
        );
        try_save_fingerprint(&path, &old).expect("seed fingerprint cache");

        let hw = HardwareSurvey {
            gpu_count: 1,
            gpu_vram: vec![64_000_000_000],
            gpu_name: Some("NVIDIA RTX 4090".into()),
            is_soc: false,
            ..Default::default()
        };

        let saved = save_result_from_outputs(
            &path,
            &hw,
            &[BenchmarkOutput {
                device: "Test GPU".into(),
                buffer_mb: 512,
                runs: 2,
                p50_gbps: 111.0,
                p90_gbps: 222.0,
                compute_tflops_fp32: None,
                compute_tflops_fp16: None,
                noise_pct: 0.1,
                runtime_s: 0.5,
                rated_gbps: None,
                rated_estimated: None,
                efficiency_pct: None,
                bus_width_bits: None,
                mem_clock_mhz: None,
                gcn_arch: None,
                hbm: None,
            }],
        )
        .expect("save should succeed");
        let loaded = load_fingerprint(&path).expect("fingerprint should exist");
        let _ = std::fs::remove_dir_all(&root);

        assert_eq!(saved.mem_bandwidth_gbps, vec![222.0]);
        assert_eq!(loaded.gpus[0].p90_gbps, 222.0);
    }

    #[test]
    #[serial]
    fn test_run_and_save_backend_not_compiled_fails_cleanly() {
        let root = std::env::temp_dir().join(format!(
            "mesh-llm-run-and-save-missing-{}",
            std::process::id()
        ));
        let bin_dir = root.join("bin");
        let path = root.join("benchmark-fingerprint.json");
        std::fs::create_dir_all(&bin_dir).expect("create bin dir");

        #[cfg(unix)]
        let child = write_test_child(
            &root,
            "mesh-llm-child",
            "echo 'CUDA benchmark backend was not compiled into this mesh-llm binary' >&2\nexit 1",
        );
        #[cfg(windows)]
        let child = write_test_child(
            &root,
            "mesh-llm-child.cmd",
            "echo CUDA benchmark backend was not compiled into this mesh-llm binary 1>&2\r\nexit /b 1",
        );

        let hw = HardwareSurvey {
            gpu_count: 1,
            gpu_vram: vec![64_000_000_000],
            gpu_name: Some("NVIDIA RTX 4090".into()),
            is_soc: false,
            ..Default::default()
        };

        let err = with_benchmark_child_override(&child, || {
            run_and_save_to_path(&hw, &bin_dir, Duration::from_secs(1), &path)
                .expect_err("uncompiled benchmark backend should fail")
        });
        let _ = std::fs::remove_dir_all(&root);

        assert!(
            err.to_string().contains("not compiled")
                || err.to_string().contains("benchmark backend")
        );
    }

    #[test]
    #[serial]
    fn test_run_benchmark_times_out_child_process() {
        let root = std::env::temp_dir().join(format!(
            "mesh-llm-benchmark-timeout-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system time")
                .as_nanos()
        ));
        std::fs::create_dir_all(&root).expect("create timeout dir");
        #[cfg(unix)]
        let child = write_test_child(&root, "mesh-llm-child", "sleep 5");
        #[cfg(windows)]
        let child = write_test_child(&root, "mesh-llm-child.cmd", "timeout /t 5 >NUL");
        let marker = root.join("mesh-llm-benchmark-cuda");

        let started = Instant::now();
        let result = with_benchmark_child_override(&child, || {
            run_benchmark(&marker, Duration::from_millis(100))
        });
        let elapsed = started.elapsed();
        let _ = std::fs::remove_dir_all(&root);

        assert!(result.is_none(), "timed out benchmark should fail");
        assert!(
            elapsed < Duration::from_secs(2),
            "timeout should be bounded"
        );
    }

    // 15. Old cache format (hardware_key field) fails to parse → load_fingerprint returns None
    #[test]
    fn test_old_cache_format_fails_parse() {
        let old_json = r#"{
            "hardware_key": {
                "gpu_count": 1,
                "gpu_vram": [80000000000],
                "gpu_name": "NVIDIA A100",
                "is_soc": false
            },
            "mem_bandwidth_gbps": 1948.7,
            "p50_gbps": 1935.2,
            "timestamp_secs": 1700000000
        }"#;
        let path = std::env::temp_dir().join("mesh-llm-test-fingerprint-old-format.json");
        std::fs::write(&path, old_json).expect("write should succeed");
        let result = load_fingerprint(&path);
        let _ = std::fs::remove_file(&path);
        assert!(
            result.is_none(),
            "old cache format should fail to parse and return None"
        );
    }

    #[test]
    fn test_benchmark_output_deserializes_without_tflops_fields() {
        let json = r#"[{"device":"NVIDIA A100","buffer_mb":512,"runs":20,"p50_gbps":1935.2,"p90_gbps":1948.7,"noise_pct":0.4,"runtime_s":1.23,"rated_gbps":2000,"rated_estimated":false,"efficiency_pct":96.8,"bus_width_bits":5120,"mem_clock_mhz":1215}]"#;
        let outputs: Vec<BenchmarkOutput> = serde_json::from_str(json).expect("should parse");

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].compute_tflops_fp32, None);
        assert_eq!(outputs[0].compute_tflops_fp16, None);
    }

    #[test]
    fn test_benchmark_output_deserializes_with_tflops_fields() {
        let json = r#"[{"device":"NVIDIA A100","buffer_mb":512,"runs":20,"p50_gbps":1935.2,"p90_gbps":1948.7,"compute_tflops_fp32":19.5,"compute_tflops_fp16":312.0,"noise_pct":0.4,"runtime_s":1.23,"rated_gbps":2000,"rated_estimated":false,"efficiency_pct":96.8,"bus_width_bits":5120,"mem_clock_mhz":1215}]"#;
        let outputs: Vec<BenchmarkOutput> = serde_json::from_str(json).expect("should parse");

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].compute_tflops_fp32, Some(19.5));
        assert_eq!(outputs[0].compute_tflops_fp16, Some(312.0));
    }

    #[test]
    fn test_benchmark_output_deserializes_fp32_only() {
        let json = r#"[{"device":"NVIDIA A100","buffer_mb":512,"runs":20,"p50_gbps":1935.2,"p90_gbps":1948.7,"compute_tflops_fp32":19.5,"noise_pct":0.4,"runtime_s":1.23,"rated_gbps":2000,"rated_estimated":false,"efficiency_pct":96.8,"bus_width_bits":5120,"mem_clock_mhz":1215}]"#;
        let outputs: Vec<BenchmarkOutput> = serde_json::from_str(json).expect("should parse");

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].compute_tflops_fp32, Some(19.5));
        assert_eq!(outputs[0].compute_tflops_fp16, None);
    }

    #[test]
    fn test_gpu_bandwidth_serde_round_trip_with_tflops() {
        let gpu = GpuBandwidth {
            name: "NVIDIA A100".into(),
            vram_bytes: 80_000_000_000,
            p50_gbps: 1935.2,
            p90_gbps: 1948.7,
            compute_tflops_fp32: Some(19.5),
            compute_tflops_fp16: Some(312.0),
        };

        let json = serde_json::to_string(&gpu).expect("should serialize");
        let round_trip: GpuBandwidth = serde_json::from_str(&json).expect("should deserialize");

        assert_eq!(round_trip, gpu);
    }

    #[test]
    fn test_gpu_bandwidth_omits_missing_tflops_fields_when_serializing() {
        let gpu = GpuBandwidth {
            name: "NVIDIA A100".into(),
            vram_bytes: 80_000_000_000,
            p50_gbps: 1935.2,
            p90_gbps: 1948.7,
            compute_tflops_fp32: None,
            compute_tflops_fp16: None,
        };

        let value = serde_json::to_value(&gpu).expect("should serialize");
        let object = value
            .as_object()
            .expect("GpuBandwidth should serialize as an object");

        assert!(!object.contains_key("compute_tflops_fp32"));
        assert!(!object.contains_key("compute_tflops_fp16"));
    }

    #[test]
    fn test_benchmark_result_tflops_none_when_binary_has_no_tflops() {
        let hw = make_hw_with_gpus();
        let output = build_output(None, None);
        let (_, result) = build_benchmark_result(&hw, &[output]);

        assert!(result.compute_tflops_fp32.is_none());
        assert!(result.compute_tflops_fp16.is_none());
    }

    #[test]
    fn test_benchmark_result_fp16_not_derived_when_fp32_available() {
        let hw = make_hw_with_gpus();
        let output = build_output(Some(19.5), None);
        let (_, result) = build_benchmark_result(&hw, &[output]);

        assert_eq!(result.compute_tflops_fp32, Some(vec![19.5]));
        assert!(result.compute_tflops_fp16.is_none());
    }

    #[test]
    fn test_benchmark_result_does_not_backfill_hardware_tflops() {
        let mut hw = make_hw_with_gpus();
        hw.gpus[0].compute_tflops_fp32 = Some(123.0);
        hw.gpus[0].compute_tflops_fp16 = Some(456.0);
        let output = build_output(None, None);
        let (_, result) = build_benchmark_result(&hw, &[output]);

        assert!(result.compute_tflops_fp32.is_none());
        assert!(result.compute_tflops_fp16.is_none());
    }

    #[test]
    fn test_build_benchmark_result_expands_identical_multi_gpu_names() {
        let hw = make_survey(
            2,
            vec![80_000_000_000, 80_000_000_000],
            Some("2× NVIDIA A100"),
            false,
        );
        let outputs = vec![
            BenchmarkOutput {
                device: "GPU 0".into(),
                buffer_mb: 512,
                runs: 2,
                p50_gbps: 100.0,
                p90_gbps: 110.0,
                compute_tflops_fp32: None,
                compute_tflops_fp16: None,
                noise_pct: 0.0,
                runtime_s: 0.0,
                rated_gbps: None,
                rated_estimated: None,
                efficiency_pct: None,
                bus_width_bits: None,
                mem_clock_mhz: None,
                gcn_arch: None,
                hbm: None,
            },
            BenchmarkOutput {
                device: "GPU 1".into(),
                buffer_mb: 512,
                runs: 2,
                p50_gbps: 120.0,
                p90_gbps: 130.0,
                compute_tflops_fp32: None,
                compute_tflops_fp16: None,
                noise_pct: 0.0,
                runtime_s: 0.0,
                rated_gbps: None,
                rated_estimated: None,
                efficiency_pct: None,
                bus_width_bits: None,
                mem_clock_mhz: None,
                gcn_arch: None,
                hbm: None,
            },
        ];

        let (gpus, result) = build_benchmark_result(&hw, &outputs);
        let fingerprint = make_fingerprint(gpus.clone(), false);

        assert_eq!(gpus.len(), 2);
        assert_eq!(gpus[0].name, "NVIDIA A100");
        assert_eq!(gpus[1].name, "NVIDIA A100");
        assert_eq!(result.mem_bandwidth_gbps, vec![110.0, 130.0]);
        assert!(!hardware_changed(&fingerprint, &hw));
    }

    #[test]
    fn test_old_fingerprint_cache_loads_without_tflops() {
        let json = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/pre-tops-fingerprint.json"
        ));
        let path = std::env::temp_dir().join("mesh-llm-test-fingerprint-pre-tops.json");
        std::fs::write(&path, json).expect("write should succeed");

        let fingerprint = load_fingerprint(&path).expect("old-format fingerprint should parse");
        let _ = std::fs::remove_file(&path);

        assert_eq!(fingerprint.gpus.len(), 1);
        assert_eq!(fingerprint.gpus[0].name, "NVIDIA A100");
        assert_eq!(fingerprint.gpus[0].compute_tflops_fp32, None);
        assert_eq!(fingerprint.gpus[0].compute_tflops_fp16, None);
    }

    #[test]
    fn test_fingerprint_path_filename() {
        let path = fingerprint_path();
        assert!(
            path.ends_with("benchmark-fingerprint.json"),
            "fingerprint_path() should use 'benchmark-fingerprint.json', got {:?}",
            path.file_name()
        );
        let parent = path.parent().expect("path should have parent");
        assert!(
            parent.ends_with("mesh-llm"),
            "fingerprint should be under mesh-llm cache directory, got {:?}",
            parent
        );
    }

    #[test]
    fn test_run_benchmark_rejects_unknown_in_process_runner() {
        let result = run_benchmark(Path::new("not-a-runner"), Duration::from_secs(1));

        assert!(result.is_none(), "unknown benchmark runner should fail");
    }
}
