use anyhow::{Context, Result};
use serde_json::{json, Value};

use crate::cli::GpuCommand;

use crate::system::{
    benchmark::{self, SavedBenchmark},
    hardware::{self, GpuFacts, HardwareSurvey},
};

pub(crate) fn dispatch_gpu_command(json_output: bool, command: Option<&GpuCommand>) -> Result<()> {
    match command {
        Some(GpuCommand::Benchmark { json }) => run_gpu_benchmark(json_output || *json),
        None => run_gpus(json_output),
    }
}

pub(crate) fn run_gpus(json_output: bool) -> Result<()> {
    let mut hw = hardware::survey();
    attach_cached_bandwidth(&mut hw);

    if json_output {
        return print_json(gpus_json(&hw));
    }

    if hw.gpus.is_empty() {
        println!("⚠️ No GPUs detected on this node.");
        return Ok(());
    }

    for (index, gpu) in hw.gpus.iter().enumerate() {
        if index > 0 {
            println!();
        }
        print_gpu(gpu);
    }

    Ok(())
}

fn run_gpu_benchmark(json_output: bool) -> Result<()> {
    let hw = hardware::survey();
    if hw.gpus.is_empty() {
        if json_output {
            return print_json(gpu_benchmark_empty_json());
        }
        println!("⚠️ No GPUs detected on this node. Nothing to benchmark.");
        return Ok(());
    }

    let bin_dir = std::env::current_exe()
        .context("failed to resolve mesh-llm binary path")?
        .parent()
        .context("mesh-llm binary path has no parent directory")?
        .to_path_buf();

    let saved = benchmark::run_and_save(&hw, &bin_dir, benchmark::BENCHMARK_TIMEOUT)?;
    let total_bandwidth: f64 = saved.result.mem_bandwidth_gbps.iter().sum();

    if json_output {
        return print_json(gpu_benchmark_json(&hw, &saved));
    }

    println!("✅ Refreshed GPU benchmark fingerprint.");
    println!(
        "  GPUs benchmarked: {}",
        saved.result.mem_bandwidth_gbps.len()
    );
    println!("  Total bandwidth: {}", format_bandwidth(total_bandwidth));
    println!("  Cache path: {}", saved.path.display());

    Ok(())
}

fn gpus_json(hw: &HardwareSurvey) -> Value {
    json!({
        "gpu_count": hw.gpus.len(),
        "gpus": hw.gpus.iter().map(gpu_json).collect::<Vec<_>>(),
    })
}

fn gpu_json(gpu: &GpuFacts) -> Value {
    json!({
        "index": gpu.index,
        "name": gpu.display_name,
        "stable_id": gpu.stable_id,
        "backend_device": gpu.backend_device,
        "vram_bytes": gpu.vram_bytes,
        "reserved_bytes": gpu.reserved_bytes,
        "mem_bandwidth_gbps": gpu.mem_bandwidth_gbps,
        "compute_tflops_fp32": gpu.compute_tflops_fp32,
        "compute_tflops_fp16": gpu.compute_tflops_fp16,
        "unified_memory": gpu.unified_memory,
        "pci_bdf": gpu.pci_bdf,
        "vendor_uuid": gpu.vendor_uuid,
        "metal_registry_id": gpu.metal_registry_id,
        "dxgi_luid": gpu.dxgi_luid,
        "pnp_instance_id": gpu.pnp_instance_id,
    })
}

fn gpu_benchmark_empty_json() -> Value {
    json!({
        "refreshed": false,
        "reason": "no_gpus_detected",
        "gpu_count": 0,
        "detected_gpu_count": 0,
        "total_bandwidth_gbps": 0.0,
        "cache_path": Value::Null,
        "gpus": [],
    })
}

fn gpu_benchmark_json(hw: &HardwareSurvey, saved: &SavedBenchmark) -> Value {
    let benchmarked_gpu_count = saved.result.mem_bandwidth_gbps.len();
    let gpus = hw
        .gpus
        .iter()
        .take(benchmarked_gpu_count)
        .enumerate()
        .map(|(index, gpu)| {
            json!({
                "index": gpu.index,
                "name": gpu.display_name,
                "stable_id": gpu.stable_id,
                "backend_device": gpu.backend_device,
                "vram_bytes": gpu.vram_bytes,
                "reserved_bytes": gpu.reserved_bytes,
                "unified_memory": gpu.unified_memory,
                "pci_bdf": gpu.pci_bdf,
                "vendor_uuid": gpu.vendor_uuid,
                "metal_registry_id": gpu.metal_registry_id,
                "dxgi_luid": gpu.dxgi_luid,
                "pnp_instance_id": gpu.pnp_instance_id,
                "mem_bandwidth_gbps": saved.result.mem_bandwidth_gbps.get(index),
                "compute_tflops_fp32": saved
                    .result
                    .compute_tflops_fp32
                    .as_ref()
                    .and_then(|values| values.get(index)),
                "compute_tflops_fp16": saved
                    .result
                    .compute_tflops_fp16
                    .as_ref()
                    .and_then(|values| values.get(index)),
            })
        })
        .collect::<Vec<_>>();

    json!({
        "refreshed": true,
        "gpu_count": benchmarked_gpu_count,
        "detected_gpu_count": hw.gpus.len(),
        "total_bandwidth_gbps": saved.result.mem_bandwidth_gbps.iter().sum::<f64>(),
        "cache_path": saved.path,
        "gpus": gpus,
    })
}

fn print_json(value: Value) -> Result<()> {
    println!("{}", serde_json::to_string_pretty(&value)?);
    Ok(())
}

fn attach_cached_bandwidth(hw: &mut HardwareSurvey) {
    let path = benchmark::fingerprint_path();
    let Some(fingerprint) = benchmark::load_fingerprint(&path) else {
        return;
    };
    if benchmark::hardware_changed(&fingerprint, hw) {
        return;
    }

    for (gpu, cached) in hw.gpus.iter_mut().zip(fingerprint.gpus.iter()) {
        gpu.mem_bandwidth_gbps = Some(cached.p90_gbps);
    }
}

fn print_gpu(gpu: &GpuFacts) {
    println!("🖥️ GPU {}", gpu.index);
    println!("  Name: {}", gpu.display_name);
    if let Some(stable_id) = gpu.stable_id.as_deref() {
        println!("  Stable ID: {stable_id}");
    }
    if let Some(backend_device) = gpu.backend_device.as_deref() {
        println!("  Backend device: {backend_device}");
    }
    println!("  VRAM: {}", format_vram(gpu.vram_bytes));
    println!(
        "  Bandwidth: {}",
        gpu.mem_bandwidth_gbps
            .map(format_bandwidth)
            .unwrap_or_else(|| "unavailable".to_string())
    );
    println!(
        "  Unified memory: {}",
        if gpu.unified_memory { "yes" } else { "no" }
    );
    if let Some(pci_bdf) = gpu.pci_bdf.as_deref() {
        println!("  PCI BDF: {pci_bdf}");
    }
    if let Some(vendor_uuid) = gpu.vendor_uuid.as_deref() {
        println!("  Vendor UUID: {vendor_uuid}");
    }
    if let Some(metal_registry_id) = gpu.metal_registry_id.as_deref() {
        println!("  Metal registry ID: {metal_registry_id}");
    }
    if let Some(dxgi_luid) = gpu.dxgi_luid.as_deref() {
        println!("  DXGI LUID: {dxgi_luid}");
    }
    if let Some(pnp_instance_id) = gpu.pnp_instance_id.as_deref() {
        println!("  PnP instance ID: {pnp_instance_id}");
    }
}

fn format_vram(bytes: u64) -> String {
    if bytes == 0 {
        "unknown".to_string()
    } else {
        format!("{:.1} GB", bytes as f64 / 1e9)
    }
}

fn format_bandwidth(gbps: f64) -> String {
    format!("{gbps:.1} GB/s")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn sample_gpu(index: usize) -> GpuFacts {
        GpuFacts {
            index,
            display_name: format!("GPU {index}"),
            backend_device: Some(format!("CUDA{index}")),
            vram_bytes: 24_000_000_000,
            reserved_bytes: Some(1_000_000_000),
            mem_bandwidth_gbps: Some(1008.0),
            compute_tflops_fp32: Some(82.4),
            compute_tflops_fp16: Some(164.8),
            unified_memory: false,
            stable_id: Some(format!("stable-{index}")),
            pci_bdf: Some(format!("0000:{index:02x}:00.0")),
            vendor_uuid: Some(format!("uuid-{index}")),
            metal_registry_id: None,
            dxgi_luid: None,
            pnp_instance_id: None,
        }
    }

    #[test]
    fn test_format_vram_unknown() {
        assert_eq!(format_vram(0), "unknown");
    }

    #[test]
    fn test_format_vram_gb() {
        assert_eq!(format_vram(24_000_000_000), "24.0 GB");
    }

    #[test]
    fn test_format_bandwidth() {
        assert_eq!(format_bandwidth(1008.04), "1008.0 GB/s");
    }

    #[test]
    fn gpus_json_includes_gpu_fields() {
        let hw = HardwareSurvey {
            gpus: vec![sample_gpu(0)],
            ..HardwareSurvey::default()
        };

        let value = gpus_json(&hw);

        assert_eq!(value["gpu_count"], json!(1));
        assert_eq!(value["gpus"][0]["name"], json!("GPU 0"));
        assert_eq!(value["gpus"][0]["mem_bandwidth_gbps"], json!(1008.0));
        assert_eq!(value["gpus"][0]["stable_id"], json!("stable-0"));
    }

    #[test]
    fn gpus_json_handles_no_gpus() {
        let value = gpus_json(&HardwareSurvey::default());

        assert_eq!(
            value,
            json!({
                "gpu_count": 0,
                "gpus": [],
            })
        );
    }

    #[test]
    fn gpu_benchmark_json_includes_summary_and_gpu_metrics() {
        let hw = HardwareSurvey {
            gpus: vec![sample_gpu(0), sample_gpu(1)],
            ..HardwareSurvey::default()
        };
        let saved = SavedBenchmark {
            path: PathBuf::from("/tmp/benchmark-fingerprint.json"),
            result: benchmark::BenchmarkResult {
                mem_bandwidth_gbps: vec![1008.0, 912.5],
                compute_tflops_fp32: Some(vec![82.4, 70.2]),
                compute_tflops_fp16: Some(vec![164.8, 140.4]),
            },
        };

        let value = gpu_benchmark_json(&hw, &saved);

        assert_eq!(value["refreshed"], json!(true));
        assert_eq!(value["gpu_count"], json!(2));
        assert_eq!(value["detected_gpu_count"], json!(2));
        assert_eq!(value["total_bandwidth_gbps"], json!(1920.5));
        assert_eq!(
            value["cache_path"],
            json!("/tmp/benchmark-fingerprint.json")
        );
        assert_eq!(value["gpus"][1]["mem_bandwidth_gbps"], json!(912.5));
        assert_eq!(value["gpus"][1]["compute_tflops_fp16"], json!(140.4));
    }

    #[test]
    fn gpu_benchmark_json_truncates_gpu_entries_to_benchmarked_count() {
        let hw = HardwareSurvey {
            gpus: vec![sample_gpu(0), sample_gpu(1)],
            ..HardwareSurvey::default()
        };
        let saved = SavedBenchmark {
            path: PathBuf::from("/tmp/benchmark-fingerprint.json"),
            result: benchmark::BenchmarkResult {
                mem_bandwidth_gbps: vec![1008.0],
                compute_tflops_fp32: Some(vec![82.4]),
                compute_tflops_fp16: Some(vec![164.8]),
            },
        };

        let value = gpu_benchmark_json(&hw, &saved);

        assert_eq!(value["gpu_count"], json!(1));
        assert_eq!(value["detected_gpu_count"], json!(2));
        assert_eq!(value["gpus"].as_array().map(Vec::len), Some(1));
        assert_eq!(value["gpus"][0]["name"], json!("GPU 0"));
    }

    #[test]
    fn gpu_benchmark_empty_json_is_machine_readable() {
        assert_eq!(
            gpu_benchmark_empty_json(),
            json!({
                "refreshed": false,
                "reason": "no_gpus_detected",
                "gpu_count": 0,
                "detected_gpu_count": 0,
                "total_bandwidth_gbps": 0.0,
                "cache_path": Value::Null,
                "gpus": [],
            })
        );
    }
}
