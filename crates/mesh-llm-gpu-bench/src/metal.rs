use crate::{parse_benchmark_output, BenchmarkOutput};
use anyhow::{anyhow, Context, Result};
use std::ffi::{c_char, c_void, CStr};

extern "C" {
    fn mesh_llm_gpu_bench_metal_json(error_out: *mut *mut c_char) -> *mut c_char;
    fn mesh_llm_gpu_bench_free(ptr: *mut c_void);
}

pub fn run() -> Result<Vec<BenchmarkOutput>> {
    let mut error: *mut c_char = std::ptr::null_mut();
    let json = unsafe { mesh_llm_gpu_bench_metal_json(&mut error) };

    if json.is_null() {
        let message = if error.is_null() {
            "Metal benchmark failed".to_string()
        } else {
            let message = unsafe { CStr::from_ptr(error) }
                .to_string_lossy()
                .into_owned();
            unsafe { mesh_llm_gpu_bench_free(error.cast()) };
            message
        };
        return Err(anyhow!(message));
    }

    let bytes = unsafe { CStr::from_ptr(json) }.to_bytes().to_vec();
    unsafe { mesh_llm_gpu_bench_free(json.cast()) };

    parse_benchmark_output(&bytes).context("Metal benchmark returned invalid output")
}
