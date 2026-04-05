use anyhow::Result;

use crate::system::{
    benchmark,
    hardware::{self, GpuFacts, HardwareSurvey},
};

pub(crate) fn run_gpus() -> Result<()> {
    let mut hw = hardware::survey();
    attach_cached_bandwidth(&mut hw);

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

fn attach_cached_bandwidth(hw: &mut HardwareSurvey) {
    let path = benchmark::fingerprint_path();
    let Some(fingerprint) = benchmark::load_fingerprint(&path) else {
        return;
    };
    if benchmark::hardware_changed(&fingerprint, hw) {
        return;
    }

    for (gpu, cached) in hw.gpus.iter_mut().zip(fingerprint.gpus.iter()) {
        gpu.bandwidth_gbps = Some(cached.p90_gbps);
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
        gpu.bandwidth_gbps
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
}
