use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use crate::system::hardware::HardwareSurvey;

#[cfg(test)]
use crate::system::hardware::GpuFacts;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BenchmarkOutput {
    pub device: String,
    pub buffer_mb: u32,
    pub runs: u32,
    pub p50_gbps: f64,
    pub p90_gbps: f64,
    pub compute_tflops_fp32: Option<f64>,
    pub compute_tflops_fp16: Option<f64>,
    pub noise_pct: f64,
    pub runtime_s: f64,
    pub rated_gbps: Option<f64>,
    pub rated_estimated: Option<bool>,
    pub efficiency_pct: Option<f64>,
    pub bus_width_bits: Option<u32>,
    pub mem_clock_mhz: Option<u64>,
    pub gcn_arch: Option<String>,
    pub hbm: Option<bool>,
}

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

fn benchmark_binary_name_for(os: &str, base: &str) -> String {
    if os == "windows" {
        format!("{base}.exe")
    } else {
        base.to_string()
    }
}

fn push_search_dir(dirs: &mut Vec<PathBuf>, dir: PathBuf) {
    let normalized = dir.canonicalize().unwrap_or(dir);
    if !dirs.iter().any(|existing| existing == &normalized) {
        dirs.push(normalized);
    }
}

fn benchmark_search_dirs(bin_dir: &Path, exe_dir: Option<&Path>) -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    push_search_dir(&mut dirs, bin_dir.to_path_buf());

    if let Some(exe_dir) = exe_dir {
        push_search_dir(&mut dirs, exe_dir.to_path_buf());
        push_search_dir(&mut dirs, exe_dir.join("../../mesh-llm/target/release"));
    }

    push_search_dir(&mut dirs, bin_dir.join("../../../mesh-llm/target/release"));
    dirs
}

fn detect_benchmark_binary_for_with_exe_dir(
    os: &str,
    hw: &HardwareSurvey,
    bin_dir: &Path,
    exe_dir: Option<&Path>,
) -> Option<PathBuf> {
    if hw.gpu_count == 0 {
        tracing::debug!("no GPUs detected — skipping benchmark");
        return None;
    }

    let gpu_upper = hw.gpu_name.as_deref().unwrap_or("").to_uppercase();

    let candidate_name = if os == "macos" && hw.is_soc {
        benchmark_binary_name_for(os, "membench-fingerprint")
    } else if os == "linux" || os == "windows" {
        if gpu_upper.contains("NVIDIA") {
            benchmark_binary_name_for(os, "membench-fingerprint-cuda")
        } else if gpu_upper.contains("AMD") || gpu_upper.contains("RADEON") {
            benchmark_binary_name_for(os, "membench-fingerprint-hip")
        } else if gpu_upper.contains("INTEL") || gpu_upper.contains("ARC") {
            tracing::info!("Intel Arc benchmark is unvalidated — results may be inaccurate");
            benchmark_binary_name_for(os, "membench-fingerprint-intel")
        } else if os == "linux" && hw.is_soc {
            tracing::warn!("Jetson benchmark is unvalidated for ARM CUDA — attempting");
            benchmark_binary_name_for(os, "membench-fingerprint-cuda")
        } else {
            tracing::warn!(
                "could not identify benchmark binary for this GPU platform: {:?}",
                hw.gpu_name
            );
            return None;
        }
    } else {
        tracing::warn!(
            "could not identify benchmark binary for this GPU platform: {:?}",
            hw.gpu_name
        );
        return None;
    };

    for search_dir in benchmark_search_dirs(bin_dir, exe_dir) {
        let candidate = search_dir.join(&candidate_name);
        if candidate.exists() {
            return Some(candidate);
        }
    }

    tracing::warn!(
        "{candidate_name} not found in benchmark search dirs: {:?}",
        benchmark_search_dirs(bin_dir, exe_dir)
    );
    None
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

/// Determine which benchmark binary to use for the current hardware platform.
///
/// Returns `None` (soft failure) if:
/// - No GPUs are present
/// - The binary is not found on disk
/// - The platform/GPU combination is unrecognised
///
/// Never panics or hard-fails with `ensure!`.
pub fn detect_benchmark_binary(hw: &HardwareSurvey, bin_dir: &Path) -> Option<PathBuf> {
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|exe_path| exe_path.parent().map(Path::to_path_buf));
    detect_benchmark_binary_for_with_exe_dir(std::env::consts::OS, hw, bin_dir, exe_dir.as_deref())
}

/// Parse raw stdout bytes from a benchmark run into a vec of per-device outputs.
///
/// Expects a JSON array of [`BenchmarkOutput`].  Returns `None` on any parse
/// failure or if the device list is empty.
pub fn parse_benchmark_output(stdout: &[u8]) -> Option<Vec<BenchmarkOutput>> {
    match serde_json::from_slice::<Vec<BenchmarkOutput>>(stdout) {
        Ok(outputs) if !outputs.is_empty() => Some(outputs),
        Ok(_) => {
            tracing::debug!("benchmark returned empty device list");
            None
        }
        Err(err) => {
            if let Ok(val) = serde_json::from_slice::<serde_json::Value>(stdout) {
                if let Some(msg) = val.get("error").and_then(|v| v.as_str()) {
                    tracing::warn!("benchmark reported error: {msg}");
                    return None;
                }
            }
            tracing::warn!("failed to parse benchmark output: {err}");
            None
        }
    }
}

/// Run the benchmark binary synchronously and return per-device outputs.
///
/// Spawns the binary as a subprocess and polls for completion up to `timeout`.
/// If the process exceeds the timeout, it is killed to avoid zombie processes.
///
/// Designed to be called inside `tokio::task::spawn_blocking` — never `async`.
pub fn run_benchmark(binary: &Path, timeout: Duration) -> Option<Vec<BenchmarkOutput>> {
    use std::io::Read;

    let mut child = match std::process::Command::new(binary)
        .arg("--json")
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            tracing::error!("failed to spawn {binary:?}: {e}");
            return None;
        }
    };

    let deadline = Instant::now() + timeout;
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) => {
                if Instant::now() >= deadline {
                    tracing::warn!("benchmark timed out after {timeout:?}, killing subprocess");
                    let _ = child.kill();
                    let _ = child.wait();
                    return None;
                }
                std::thread::sleep(Duration::from_millis(100));
            }
            Err(e) => {
                tracing::error!("error waiting for benchmark: {e}");
                let _ = child.kill();
                let _ = child.wait();
                return None;
            }
        }
    };

    if !status.success() {
        tracing::warn!("benchmark exited with {:?}", status);
        return None;
    }

    let mut stdout_bytes = Vec::new();
    if let Some(mut pipe) = child.stdout.take() {
        let _ = pipe.read_to_end(&mut stdout_bytes);
    }
    parse_benchmark_output(&stdout_bytes)
}

/// Load a cached fingerprint if hardware is unchanged, otherwise run the
/// benchmark binary and persist the result.
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

    let binary = detect_benchmark_binary(hw, bin_dir)?;
    let outputs = run_benchmark(&binary, timeout)?;

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

    let binary = detect_benchmark_binary(hw, bin_dir).with_context(|| {
        format!(
            "no supported benchmark binary found for detected GPU platform {:?}",
            hw.gpu_name
        )
    })?;

    let outputs = run_benchmark(&binary, timeout)
        .with_context(|| format!("benchmark run failed for {}", binary.display()))?;

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
    fn test_benchmark_binary_name_for_windows() {
        assert_eq!(
            benchmark_binary_name_for("windows", "membench-fingerprint-cuda"),
            "membench-fingerprint-cuda.exe"
        );
        assert_eq!(
            benchmark_binary_name_for("linux", "membench-fingerprint-cuda"),
            "membench-fingerprint-cuda"
        );
    }

    #[test]
    fn test_detect_benchmark_binary_windows_cuda_missing_is_soft_failure() {
        let hw = make_survey(1, vec![24_000_000_000], Some("NVIDIA RTX 4090"), false);
        assert!(detect_benchmark_binary_for_with_exe_dir(
            "windows",
            &hw,
            Path::new("C:\\bench"),
            None,
        )
        .is_none());
    }

    #[test]
    fn test_detect_benchmark_binary_windows_hip_missing_is_soft_failure() {
        let hw = make_survey(
            1,
            vec![24_000_000_000],
            Some("AMD Radeon RX 7900 XTX"),
            false,
        );
        assert!(detect_benchmark_binary_for_with_exe_dir(
            "windows",
            &hw,
            Path::new("C:\\bench"),
            None,
        )
        .is_none());
    }

    #[test]
    fn test_detect_benchmark_binary_windows_intel_missing_is_soft_failure() {
        let hw = make_survey(1, vec![16_000_000_000], Some("Intel Arc A770"), false);
        assert!(detect_benchmark_binary_for_with_exe_dir(
            "windows",
            &hw,
            Path::new("C:\\bench"),
            None,
        )
        .is_none());
    }

    #[test]
    fn test_detect_benchmark_binary_linux_cuda_finds_crate_release_helper() {
        let root =
            std::env::temp_dir().join(format!("mesh-llm-benchmark-lookup-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let bin_dir = root.join("llama.cpp/build/bin");
        let exe_dir = root.join("target/release");
        let crate_release = root.join("mesh-llm/target/release");
        std::fs::create_dir_all(&bin_dir).expect("create bin dir");
        std::fs::create_dir_all(&exe_dir).expect("create exe dir");
        std::fs::create_dir_all(&crate_release).expect("create crate release dir");

        let helper = crate_release.join("membench-fingerprint-cuda");
        std::fs::write(&helper, b"").expect("write helper");

        let hw = make_survey(1, vec![24_000_000_000], Some("NVIDIA RTX 4090"), false);
        let result = detect_benchmark_binary_for_with_exe_dir(
            "linux",
            &hw,
            &bin_dir,
            Some(exe_dir.as_path()),
        );
        let expected = helper.canonicalize().expect("canonical helper path");

        let _ = std::fs::remove_dir_all(&root);

        assert_eq!(result.as_deref(), Some(expected.as_path()));
    }

    #[test]
    fn test_detect_benchmark_binary_macos_soc_finds_crate_release_helper() {
        let root = std::env::temp_dir().join(format!(
            "mesh-llm-benchmark-lookup-macos-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        let bin_dir = root.join("llama.cpp/build/bin");
        let exe_dir = root.join("target/release");
        let crate_release = root.join("mesh-llm/target/release");
        std::fs::create_dir_all(&bin_dir).expect("create bin dir");
        std::fs::create_dir_all(&exe_dir).expect("create exe dir");
        std::fs::create_dir_all(&crate_release).expect("create crate release dir");

        let helper = crate_release.join("membench-fingerprint");
        std::fs::write(&helper, b"").expect("write helper");

        let hw = make_survey(1, vec![24_000_000_000], Some("Apple M4 Pro"), true);
        let result = detect_benchmark_binary_for_with_exe_dir(
            "macos",
            &hw,
            &bin_dir,
            Some(exe_dir.as_path()),
        );
        let expected = helper.canonicalize().expect("canonical helper path");

        let _ = std::fs::remove_dir_all(&root);

        assert_eq!(result.as_deref(), Some(expected.as_path()));
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

    #[cfg(unix)]
    #[test]
    fn test_run_and_save_rewrites_existing_cache() {
        use std::os::unix::fs::PermissionsExt;

        let root = std::env::temp_dir().join(format!(
            "mesh-llm-run-and-save-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system time")
                .as_nanos()
        ));
        let bin_dir = root.join("bin");
        let path = root.join("benchmark-fingerprint.json");
        std::fs::create_dir_all(&bin_dir).expect("create bin dir");

        let binary_name = if cfg!(target_os = "macos") {
            "membench-fingerprint"
        } else {
            "membench-fingerprint-cuda"
        };
        let binary = bin_dir.join(binary_name);
        std::fs::write(
            &binary,
            "#!/bin/sh\nprintf '%s\n' '[{\"device\":\"Test GPU\",\"buffer_mb\":512,\"runs\":2,\"p50_gbps\":111.0,\"p90_gbps\":222.0,\"noise_pct\":0.1,\"runtime_s\":0.5}]'\n",
        )
        .expect("write fake benchmark");
        std::fs::set_permissions(&binary, std::fs::Permissions::from_mode(0o755))
            .expect("chmod fake benchmark");

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
            gpu_name: Some(if cfg!(target_os = "macos") {
                "Apple M4 Pro".into()
            } else {
                "NVIDIA RTX 4090".into()
            }),
            is_soc: cfg!(target_os = "macos"),
            ..Default::default()
        };

        let saved = run_and_save_to_path(&hw, &bin_dir, Duration::from_secs(2), &path)
            .expect("forced benchmark should succeed");
        let loaded = load_fingerprint(&path).expect("fingerprint should exist");
        let _ = std::fs::remove_dir_all(&root);

        assert_eq!(saved.result.mem_bandwidth_gbps, vec![222.0]);
        assert_eq!(loaded.gpus[0].p90_gbps, 222.0);
    }

    #[test]
    fn test_run_and_save_missing_binary_fails_cleanly() {
        let root = std::env::temp_dir().join(format!(
            "mesh-llm-run-and-save-missing-{}",
            std::process::id()
        ));
        let bin_dir = root.join("bin");
        let path = root.join("benchmark-fingerprint.json");
        std::fs::create_dir_all(&bin_dir).expect("create bin dir");

        let hw = HardwareSurvey {
            gpu_count: 1,
            gpu_vram: vec![64_000_000_000],
            gpu_name: Some(if cfg!(target_os = "macos") {
                "Apple M4 Pro".into()
            } else {
                "NVIDIA RTX 4090".into()
            }),
            is_soc: cfg!(target_os = "macos"),
            ..Default::default()
        };

        let err = run_and_save_to_path(&hw, &bin_dir, Duration::from_secs(1), &path)
            .expect_err("missing benchmark binary should fail");
        let _ = std::fs::remove_dir_all(&root);

        assert!(err.to_string().contains("benchmark binary"));
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

    #[cfg(unix)]
    #[test]
    fn test_run_benchmark_kills_on_timeout() {
        use std::os::unix::fs::PermissionsExt;

        let dir = std::env::temp_dir().join("mesh-llm-test-bm-timeout");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create test dir");

        let script = dir.join("hang.sh");
        std::fs::write(&script, "#!/bin/sh\nsleep 999\n").expect("write script");
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755))
            .expect("set permissions");

        let start = std::time::Instant::now();
        let result = run_benchmark(&script, Duration::from_secs(1));
        let elapsed = start.elapsed();

        let _ = std::fs::remove_dir_all(&dir);

        assert!(
            result.is_none(),
            "hanging benchmark should return None on timeout"
        );
        assert!(
            elapsed < Duration::from_secs(5),
            "should kill subprocess promptly, took {:?}",
            elapsed
        );
    }
}
