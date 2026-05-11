fn backend_device_for_name(name: &str, index: usize, is_soc: bool) -> Option<String> {
    backend_device_for_name_for_platform(name, index, is_soc, cfg!(target_os = "macos"))
}

fn backend_device_for_name_for_platform(
    name: &str,
    index: usize,
    is_soc: bool,
    soc_backend_is_metal: bool,
) -> Option<String> {
    if soc_backend_is_metal && is_soc {
        return Some(format!("MTL{index}"));
    }
    let upper = name.to_ascii_uppercase();
    if upper.contains("NVIDIA")
        || (is_soc
            && (upper.contains("JETSON")
                || upper.contains("TEGRA")
                || upper.contains("NVGPU")
                || upper.contains("ORIN")))
    {
        Some(format!("CUDA{index}"))
    } else if upper.contains("AMD")
        || upper.contains("RADEON")
        || upper.contains("INSTINCT")
        || upper.starts_with("MI")
    {
        Some(format!("ROCm{index}"))
    } else {
        None
    }
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
fn detect_nvidia_identities() -> Vec<(Option<String>, Option<String>)> {
    let out = match std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=pci.bus_id,uuid", "--format=csv,noheader"])
        .output()
    {
        Ok(out) if out.status.success() => out,
        _ => return Vec::new(),
    };
    let Ok(stdout) = String::from_utf8(out.stdout) else {
        return Vec::new();
    };
    parse_nvidia_gpu_identity(&stdout)
}

#[cfg(not(any(target_os = "linux", target_os = "windows")))]
fn detect_nvidia_identities() -> Vec<(Option<String>, Option<String>)> {
    Vec::new()
}

fn inferred_gpu_name_count(name: Option<&str>) -> usize {
    let Some(name) = name.map(str::trim).filter(|name| !name.is_empty()) else {
        return 0;
    };

    name.split_once('×')
        .or_else(|| name.split_once('x'))
        .or_else(|| name.split_once('X'))
        .and_then(|(count, _)| count.trim().parse::<usize>().ok())
        .filter(|&count| count > 0)
        .unwrap_or(1)
}

fn is_pinnable_gpu_stable_id(stable_id: &str) -> bool {
    stable_id.starts_with("pci:")
        || stable_id.starts_with("uuid:")
        || stable_id.starts_with("metal:")
}

pub fn pinnable_gpu_stable_ids(gpus: &[GpuFacts]) -> Vec<String> {
    gpus.iter()
        .filter_map(|gpu| gpu.stable_id.as_deref())
        .filter(|stable_id| is_pinnable_gpu_stable_id(stable_id))
        .map(str::to_string)
        .collect()
}

fn format_pinnable_gpu_ids(ids: &[String]) -> String {
    if ids.is_empty() {
        "none".to_string()
    } else {
        ids.join(", ")
    }
}

pub fn resolve_pinned_gpu<'a>(
    configured_id: Option<&str>,
    gpus: &'a [GpuFacts],
) -> Result<&'a GpuFacts, PinnedGpuResolverError> {
    let available_pinnable_ids = pinnable_gpu_stable_ids(gpus);
    let Some(configured_id) = configured_id.map(str::trim).filter(|id| !id.is_empty()) else {
        return Err(PinnedGpuResolverError::MissingConfiguredId {
            available_pinnable_ids,
        });
    };
    let configured_id = configured_id.to_string();

    if !is_pinnable_gpu_stable_id(&configured_id) {
        return Err(PinnedGpuResolverError::NonPinnableConfiguredId {
            configured_id,
            available_pinnable_ids,
        });
    }

    if available_pinnable_ids.is_empty() {
        return Err(PinnedGpuResolverError::NoPinnableGpus {
            configured_id,
            available_pinnable_ids,
        });
    }

    let matches = gpus
        .iter()
        .enumerate()
        .filter(|(_, gpu)| gpu.stable_id.as_deref() == Some(configured_id.as_str()))
        .filter(|(_, gpu)| {
            gpu.stable_id
                .as_deref()
                .is_some_and(is_pinnable_gpu_stable_id)
        })
        .collect::<Vec<_>>();

    match matches.as_slice() {
        [(_, gpu)] => Ok(*gpu),
        [] => Err(PinnedGpuResolverError::NoMatch {
            configured_id,
            available_pinnable_ids,
        }),
        _ => Err(PinnedGpuResolverError::AmbiguousMatch {
            configured_id,
            available_pinnable_ids,
            match_indexes: matches.iter().map(|(index, _)| *index).collect(),
        }),
    }
}

fn hydrate_gpu_facts(survey: &mut HardwareSurvey, metrics: &[Metric]) {
    let expected_count = survey
        .gpu_vram
        .len()
        .max(usize::from(survey.gpu_count))
        .max(inferred_gpu_name_count(survey.gpu_name.as_deref()));
    let mut names = expand_gpu_names(survey.gpu_name.as_deref(), expected_count);
    if names.is_empty() && expected_count > 0 {
        names = (0..expected_count)
            .map(|index| format!("GPU {index}"))
            .collect();
    }

    let needs_nvidia_identities = metrics.contains(&Metric::GpuName);
    let nvidia_identities = if needs_nvidia_identities {
        detect_nvidia_identities()
    } else {
        Vec::new()
    };
    hydrate_gpu_facts_with_identities(
        survey,
        metrics,
        &nvidia_identities,
        names,
        expected_count,
        cfg!(target_os = "macos"),
    );
}

fn hydrate_gpu_facts_with_identities(
    survey: &mut HardwareSurvey,
    metrics: &[Metric],
    nvidia_identities: &[(Option<String>, Option<String>)],
    names: Vec<String>,
    expected_count: usize,
    soc_backend_is_metal: bool,
) {
    let count = expected_count.max(names.len());
    survey.gpus = (0..count)
        .map(|index| {
            let display_name = names
                .get(index)
                .cloned()
                .unwrap_or_else(|| format!("GPU {index}"));
            let backend_device = if soc_backend_is_metal == cfg!(target_os = "macos") {
                backend_device_for_name(&display_name, index, survey.is_soc)
            } else {
                backend_device_for_name_for_platform(
                    &display_name,
                    index,
                    survey.is_soc,
                    soc_backend_is_metal,
                )
            };
            let (pci_bdf, vendor_uuid) = nvidia_identities.get(index).cloned().unwrap_or_default();
            let stable_id = if survey.is_soc && soc_backend_is_metal {
                Some(format!("metal:{index}"))
            } else if let Some(ref pci_bdf) = pci_bdf {
                Some(format!("pci:{pci_bdf}"))
            } else if let Some(ref vendor_uuid) = vendor_uuid {
                Some(format!("uuid:{vendor_uuid}"))
            } else if let Some(ref backend_device) = backend_device {
                Some(backend_device.to_ascii_lowercase())
            } else {
                Some(format!("index:{index}"))
            };

            GpuFacts {
                index,
                display_name,
                backend_device,
                vram_bytes: survey.gpu_vram.get(index).copied().unwrap_or(0),
                reserved_bytes: survey.gpu_reserved.get(index).cloned().flatten(),
                mem_bandwidth_gbps: None,
                compute_tflops_fp32: None,
                compute_tflops_fp16: None,
                unified_memory: survey.is_soc,
                stable_id,
                pci_bdf,
                vendor_uuid,
                metal_registry_id: None,
                dxgi_luid: None,
                pnp_instance_id: None,
            }
        })
        .collect();

    debug_assert!(pinnable_gpu_stable_ids(&survey.gpus)
        .into_iter()
        .all(|stable_id| resolve_pinned_gpu(Some(&stable_id), &survey.gpus).is_ok()));

    if metrics.contains(&Metric::GpuCount) && survey.gpu_count == 0 {
        survey.gpu_count = u8::try_from(survey.gpus.len()).unwrap_or(u8::MAX);
    }
    if metrics.contains(&Metric::GpuName) && survey.gpu_name.is_none() {
        let names: Vec<String> = survey
            .gpus
            .iter()
            .map(|gpu| gpu.display_name.clone())
            .collect();
        survey.gpu_name = summarize_gpu_name(&names);
    }
}
