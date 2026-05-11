fn parse_vulkan_gpu_header(line: &str) -> Option<usize> {
    let trimmed = line.trim();
    let suffix = trimmed.strip_prefix("GPU")?.strip_suffix(':')?;
    suffix.parse().ok()
}

/// Parse `vulkaninfo --summary` device sections.
pub fn parse_vulkaninfo_summary_devices(output: &str) -> Vec<VulkanGpuFacts> {
    let mut devices = Vec::new();
    let mut current: Option<VulkanGpuFacts> = None;

    for line in output.lines() {
        if let Some(index) = parse_vulkan_gpu_header(line) {
            if let Some(device) = current.take() {
                devices.push(device);
            }
            current = Some(VulkanGpuFacts {
                index,
                ..Default::default()
            });
            continue;
        }

        let Some(device) = current.as_mut() else {
            continue;
        };
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        let key = key.trim();
        let value = value.trim().to_string();
        match key {
            "vendorID" => device.vendor_id = Some(value),
            "deviceID" => device.device_id = Some(value),
            "deviceType" => device.device_type = value,
            "deviceName" => device.display_name = value,
            "deviceUUID" => device.device_uuid = Some(value),
            _ => {}
        }
    }

    if let Some(device) = current {
        devices.push(device);
    }
    devices
}

fn pci_bdf_from_radv_vulkan_uuid(uuid: &str) -> Option<String> {
    let parts = uuid.split('-').collect::<Vec<_>>();
    if parts.len() != 5 {
        return None;
    }
    if parts[0].len() != 8
        || parts[1].len() != 4
        || parts[2].len() != 4
        || !parts
            .iter()
            .all(|part| part.chars().all(|ch| ch.is_ascii_hexdigit()))
    {
        return None;
    }
    if parts[2] != "0000" || parts[3] != "0000" || parts[4] != "000000000000" {
        return None;
    }

    let domain = parts[0];
    let bus = &parts[1][..2];
    let device = &parts[1][2..4];
    let function = &parts[2][..1];
    Some(format!("{domain}:{bus}:{device}.{function}"))
}

fn vendor_uuid_matches_vulkan_device(gpu: &GpuFacts, device: &VulkanGpuFacts) -> bool {
    let Some(vulkan_uuid) = device.device_uuid.as_deref() else {
        return false;
    };
    let Some(vendor_uuid) = gpu.vendor_uuid.as_deref() else {
        return false;
    };
    vendor_uuid
        .strip_prefix("GPU-")
        .unwrap_or(vendor_uuid)
        .eq_ignore_ascii_case(vulkan_uuid)
}

pub fn merge_vulkan_devices_into_gpu_facts(
    gpus: &mut Vec<GpuFacts>,
    devices: &[VulkanGpuFacts],
    unified_memory_bytes: u64,
) {
    for device in devices {
        if device.device_type.contains("CPU") {
            continue;
        }

        if let Some(existing) = gpus
            .iter_mut()
            .find(|gpu| vendor_uuid_matches_vulkan_device(gpu, device))
        {
            existing.index = device.index;
            existing.backend_device = Some(format!("Vulkan{}", device.index));
            continue;
        }

        let pci_bdf = device
            .device_uuid
            .as_deref()
            .and_then(pci_bdf_from_radv_vulkan_uuid);
        let Some(pci_bdf) = pci_bdf else {
            continue;
        };
        let stable_id = format!("pci:{pci_bdf}");
        if gpus
            .iter()
            .any(|gpu| gpu.stable_id.as_deref() == Some(stable_id.as_str()))
        {
            continue;
        }

        let unified_memory = device.device_type.contains("INTEGRATED");
        let vram_bytes = if unified_memory {
            (unified_memory_bytes as f64 * 0.75) as u64
        } else {
            0
        };
        gpus.push(GpuFacts {
            index: device.index,
            display_name: device.display_name.clone(),
            backend_device: Some(format!("Vulkan{}", device.index)),
            vram_bytes,
            reserved_bytes: None,
            mem_bandwidth_gbps: None,
            compute_tflops_fp32: None,
            compute_tflops_fp16: None,
            unified_memory,
            stable_id: Some(stable_id),
            pci_bdf: Some(pci_bdf),
            vendor_uuid: device.device_uuid.clone(),
            metal_registry_id: None,
            dxgi_luid: None,
            pnp_instance_id: None,
        });
    }
}

pub fn augment_gpu_facts_with_vulkan_devices(gpus: &mut Vec<GpuFacts>) {
    let Ok(output) = std::process::Command::new("vulkaninfo")
        .arg("--summary")
        .output()
    else {
        return;
    };
    if !output.status.success() {
        return;
    }
    let Ok(stdout) = String::from_utf8(output.stdout) else {
        return;
    };
    let devices = parse_vulkaninfo_summary_devices(&stdout);
    #[cfg(target_os = "linux")]
    let unified_memory_bytes = read_system_ram_bytes();
    #[cfg(not(target_os = "linux"))]
    let unified_memory_bytes = 0;
    merge_vulkan_devices_into_gpu_facts(gpus, &devices, unified_memory_bytes);
}
