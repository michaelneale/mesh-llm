use crate::{capture::capture_stdout, parse_benchmark_output, BenchmarkOutput};
use anyhow::{Context, Result};
use std::ffi::c_int;

extern "C" {
    fn mesh_llm_gpu_bench_intel_main() -> c_int;
}

pub fn run() -> Result<Vec<BenchmarkOutput>> {
    let stdout = capture_stdout(mesh_llm_gpu_bench_intel_main)?;
    parse_benchmark_output(&stdout).context("Intel benchmark returned invalid output")
}
