use crate::BenchmarkOutput;
use anyhow::{anyhow, Result};
use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BenchmarkBackend {
    Metal,
    Cuda,
    Hip,
    Intel,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BenchmarkRunner {
    pub backend: BenchmarkBackend,
}

pub fn runner_for(
    os: &str,
    gpu_count: u8,
    gpu_name: Option<&str>,
    is_soc: bool,
) -> Option<BenchmarkRunner> {
    if gpu_count == 0 {
        tracing::debug!("no GPUs detected; skipping benchmark");
        return None;
    }

    let gpu_upper = gpu_name.unwrap_or("").to_uppercase();

    if os == "macos" && is_soc {
        return Some(BenchmarkRunner {
            backend: BenchmarkBackend::Metal,
        });
    }

    if os == "linux" || os == "windows" {
        if gpu_upper.contains("NVIDIA") {
            return Some(BenchmarkRunner {
                backend: BenchmarkBackend::Cuda,
            });
        }

        if gpu_upper.contains("AMD") || gpu_upper.contains("RADEON") {
            return Some(BenchmarkRunner {
                backend: BenchmarkBackend::Hip,
            });
        }

        if gpu_upper.contains("INTEL") || gpu_upper.contains("ARC") {
            tracing::info!(
                "Intel GPU benchmark is not supported in standard mesh-llm builds; skipping"
            );
            return None;
        }

        if os == "linux" && is_soc {
            tracing::warn!("Jetson benchmark is unvalidated for ARM CUDA; attempting");
            return Some(BenchmarkRunner {
                backend: BenchmarkBackend::Cuda,
            });
        }
    }

    tracing::warn!("could not identify benchmark runner for GPU platform: {gpu_name:?}");
    None
}

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

pub fn run_benchmark(runner: BenchmarkRunner, _timeout: Duration) -> Result<Vec<BenchmarkOutput>> {
    match runner.backend {
        BenchmarkBackend::Metal => run_metal_benchmark(),
        BenchmarkBackend::Cuda => run_cuda_benchmark(),
        BenchmarkBackend::Hip => run_hip_benchmark(),
        BenchmarkBackend::Intel => run_intel_benchmark(),
    }
}

#[cfg(target_os = "macos")]
fn run_metal_benchmark() -> Result<Vec<BenchmarkOutput>> {
    crate::metal::run()
}

#[cfg(not(target_os = "macos"))]
fn run_metal_benchmark() -> Result<Vec<BenchmarkOutput>> {
    Err(anyhow!(
        "Metal benchmark backend was not compiled into this mesh-llm binary"
    ))
}

#[cfg(feature = "cuda")]
fn run_cuda_benchmark() -> Result<Vec<BenchmarkOutput>> {
    crate::cuda::run()
}

#[cfg(not(feature = "cuda"))]
fn run_cuda_benchmark() -> Result<Vec<BenchmarkOutput>> {
    Err(anyhow!(
        "CUDA benchmark backend was not compiled into this mesh-llm binary"
    ))
}

#[cfg(feature = "hip")]
fn run_hip_benchmark() -> Result<Vec<BenchmarkOutput>> {
    crate::hip::run()
}

#[cfg(not(feature = "hip"))]
fn run_hip_benchmark() -> Result<Vec<BenchmarkOutput>> {
    Err(anyhow!(
        "HIP benchmark backend was not compiled into this mesh-llm binary"
    ))
}

#[cfg(feature = "intel")]
fn run_intel_benchmark() -> Result<Vec<BenchmarkOutput>> {
    crate::intel::run()
}

#[cfg(not(feature = "intel"))]
fn run_intel_benchmark() -> Result<Vec<BenchmarkOutput>> {
    Err(anyhow!(
        "Intel benchmark backend was not compiled into this mesh-llm binary"
    ))
}
