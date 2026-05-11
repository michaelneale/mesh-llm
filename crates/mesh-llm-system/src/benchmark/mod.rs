use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use crate::hardware::HardwareSurvey;

#[cfg(test)]
use crate::hardware::GpuFacts;

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

include!("fingerprint.rs");
include!("discovery.rs");
include!("persistence.rs");
include!("runner.rs");
include!("tests.rs");
