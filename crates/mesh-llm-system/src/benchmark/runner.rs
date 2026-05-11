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
