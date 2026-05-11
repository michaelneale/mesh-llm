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
        push_search_dir(&mut dirs, exe_dir.join("../../target/release"));
    }

    push_search_dir(&mut dirs, bin_dir.join("../../../target/release"));
    push_search_dir(&mut dirs, bin_dir.join("../../../../target/release"));
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
