use super::GpuFacts;

pub fn gpu_facts() -> anyhow::Result<Vec<GpuFacts>> {
    let mut accelerator_index = 0usize;
    let mut facts = Vec::new();

    for device in skippy_runtime::backend_devices()? {
        if !matches!(
            device.device_type,
            skippy_runtime::BackendDeviceType::Gpu
                | skippy_runtime::BackendDeviceType::IntegratedGpu
        ) {
            continue;
        }

        let index = accelerator_index;
        accelerator_index += 1;

        let backend_device = Some(device.name.clone());
        let pci_bdf = device.device_id.clone();
        let unified_memory = device.device_type == skippy_runtime::BackendDeviceType::IntegratedGpu
            || (cfg!(target_os = "macos") && device.name.starts_with("MTL"));
        let stable_id = if unified_memory && cfg!(target_os = "macos") {
            Some(format!("metal:{index}"))
        } else {
            pci_bdf
                .as_ref()
                .map(|id| format!("pci:{id}"))
                .or_else(|| Some(device.name.to_ascii_lowercase()))
        };
        let (vram_bytes, reserved_bytes) = if unified_memory && cfg!(target_os = "macos") {
            #[cfg(target_os = "macos")]
            {
                super::macos_metal_gpu_budget(super::query_metal_recommended_working_set_bytes())
                    .unwrap_or((device.memory_total, None))
            }
            #[cfg(not(target_os = "macos"))]
            {
                (device.memory_total, None)
            }
        } else {
            (device.memory_total, None)
        };

        facts.push(GpuFacts {
            index,
            display_name: device.description.unwrap_or(device.name),
            backend_device,
            vram_bytes,
            reserved_bytes,
            mem_bandwidth_gbps: None,
            compute_tflops_fp32: None,
            compute_tflops_fp16: None,
            unified_memory,
            stable_id,
            pci_bdf,
            vendor_uuid: None,
            metal_registry_id: None,
            dxgi_luid: None,
            pnp_instance_id: None,
        });
    }

    if facts.is_empty() {
        facts = super::enrichers::discover_gpu_facts();
    }
    super::enrichers::enrich_gpu_facts(&mut facts);

    Ok(facts)
}
