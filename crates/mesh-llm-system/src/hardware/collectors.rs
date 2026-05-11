fn detect_hostname() -> Option<String> {
    let out = std::process::Command::new("hostname").output().ok()?;
    if !out.status.success() {
        return None;
    }
    parse_hostname(&String::from_utf8(out.stdout).ok()?)
}

#[cfg(target_os = "linux")]
fn read_system_ram_bytes() -> u64 {
    (|| -> Option<u64> {
        let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
        for line in meminfo.lines() {
            if line.starts_with("MemTotal:") {
                let kb = line.split_whitespace().nth(1)?.parse::<u64>().ok()?;
                return Some(kb * 1024);
            }
        }
        None
    })()
    .unwrap_or(0)
}

#[cfg(target_os = "linux")]
fn try_tegrastats_ram() -> Option<u64> {
    use std::io::BufRead;
    let mut child = std::process::Command::new("tegrastats")
        .stdout(std::process::Stdio::piped())
        .spawn()
        .ok()?;
    let stdout = child.stdout.take()?;
    let line = std::io::BufReader::new(stdout).lines().next()?.ok()?;
    let _ = child.kill();
    let _ = child.wait();
    parse_tegrastats_ram(&line)
}

#[cfg(target_os = "windows")]
fn powershell_output(script: &str) -> Option<String> {
    let output = std::process::Command::new("powershell")
        .args(["-NoProfile", "-Command", script])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8(output.stdout).ok()
}

#[cfg(target_os = "windows")]
fn read_windows_total_ram_bytes() -> Option<u64> {
    let output = powershell_output(
        "Get-CimInstance Win32_ComputerSystem | Select-Object -ExpandProperty TotalPhysicalMemory",
    )?;
    parse_windows_total_physical_memory(&output)
}

#[cfg(target_os = "windows")]
fn read_windows_video_controllers() -> Vec<(String, u64)> {
    let Some(output) = powershell_output(
        "Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM | ConvertTo-Json -Compress",
    ) else {
        return Vec::new();
    };
    parse_windows_video_controller_json(&output)
}

impl Collector for DefaultCollector {
    fn collect(&self, metrics: &[Metric]) -> HardwareSurvey {
        let mut survey = HardwareSurvey::default();

        #[cfg(target_os = "macos")]
        {
            if metrics.contains(&Metric::IsSoc) {
                survey.is_soc = true;
            }
            if metrics.contains(&Metric::VramBytes) {
                let out = std::process::Command::new("sysctl")
                    .args(["-n", "hw.memsize"])
                    .output()
                    .ok();
                if let Some(out) = out {
                    if let Ok(s) = String::from_utf8(out.stdout) {
                        if let Ok(bytes) = s.trim().parse::<u64>() {
                            let iogpu_output = std::process::Command::new("sysctl")
                                .arg("iogpu")
                                .output()
                                .ok()
                                .filter(|out| out.status.success())
                                .and_then(|out| String::from_utf8(out.stdout).ok());
                            let (usable_bytes, reserved_bytes) =
                                derive_macos_gpu_budget(bytes, iogpu_output.as_deref());
                            survey.vram_bytes = usable_bytes;
                            survey.gpu_vram = vec![bytes];
                            survey.gpu_reserved = vec![reserved_bytes];
                        }
                    }
                }
            }
            if metrics.contains(&Metric::GpuName) {
                let out = std::process::Command::new("sysctl")
                    .args(["-n", "machdep.cpu.brand_string"])
                    .output()
                    .ok();
                if let Some(out) = out {
                    if let Ok(s) = String::from_utf8(out.stdout) {
                        survey.gpu_name = parse_macos_cpu_brand(&s);
                    }
                }
            }
            if metrics.contains(&Metric::GpuCount) {
                survey.gpu_count = 1;
            }
        }

        #[cfg(target_os = "linux")]
        {
            let system_ram = read_system_ram_bytes();

            if metrics.contains(&Metric::VramBytes) {
                // Try NVIDIA (mesh.rs:284-316)
                let nvidia_vram: Option<(u64, Vec<u64>)> = (|| {
                    let out = std::process::Command::new("nvidia-smi")
                        .args([
                            "--query-gpu=memory.total,memory.reserved",
                            "--format=csv,noheader,nounits",
                        ])
                        .output()
                        .ok();
                    if let Some(out) = out {
                        if out.status.success() {
                            let s = String::from_utf8(out.stdout).ok()?;
                            let parsed = parse_nvidia_gpu_memory_and_reserved(&s);
                            if !parsed.is_empty() {
                                survey.gpu_reserved =
                                    parsed.iter().map(|(_, reserved)| *reserved).collect();
                                let per_gpu: Vec<u64> =
                                    parsed.iter().map(|(total, _)| *total).collect();
                                let total: u64 = per_gpu.iter().sum();
                                if total > 0 {
                                    return Some((total, per_gpu));
                                }
                            }
                        }
                    }
                    let out = std::process::Command::new("nvidia-smi")
                        .args(["--query-gpu=memory.total", "--format=csv,noheader,nounits"])
                        .output()
                        .ok()?;
                    if !out.status.success() {
                        return None;
                    }
                    let s = String::from_utf8(out.stdout).ok()?;
                    let per_gpu: Vec<u64> = s
                        .lines()
                        .filter_map(|line| {
                            let mib = line.trim().parse::<u64>().ok()?;
                            Some(mib * 1024 * 1024)
                        })
                        .collect();
                    let total: u64 = per_gpu.iter().sum();
                    if total > 0 {
                        survey.gpu_reserved = vec![None; per_gpu.len()];
                        Some((total, per_gpu))
                    } else {
                        None
                    }
                })();

                if let Some((vram, per_gpu)) = nvidia_vram {
                    survey.gpu_vram = per_gpu;
                    let ram_offload = system_ram.saturating_sub(vram);
                    survey.vram_bytes = vram + (ram_offload as f64 * 0.75) as u64;
                } else {
                    // Try AMD ROCm (mesh.rs:295-316)
                    let rocm_vram: Option<(Vec<u64>, bool)> = (|| {
                        let out = std::process::Command::new("rocm-smi")
                            .args(["--showmeminfo", "vram", "--csv"])
                            .output()
                            .ok()?;
                        if !out.status.success() {
                            return None;
                        }
                        let s = String::from_utf8(out.stdout).ok()?;
                        let parsed = parse_rocm_gpu_memory_and_used(&s);
                        // ROCm exposes total and live used VRAM here, not a
                        // true reserved/unavailable metric, so leave
                        // reserved_bytes unavailable for this backend.
                        survey.gpu_reserved = vec![None; parsed.len()];
                        let vrams: Vec<u64> = parsed.iter().map(|(total, _)| *total).collect();
                        if vrams.is_empty() {
                            None
                        } else {
                            let gtt_totals = std::process::Command::new("rocm-smi")
                                .args(["--showmeminfo", "gtt", "--csv"])
                                .output()
                                .ok()
                                .filter(|out| out.status.success())
                                .and_then(|out| String::from_utf8(out.stdout).ok())
                                .map(|stdout| {
                                    parse_rocm_gpu_memory_and_used(&stdout)
                                        .into_iter()
                                        .map(|(total, _)| total)
                                        .collect::<Vec<_>>()
                                })
                                .unwrap_or_default();
                            if let Some(usable_bytes) =
                                rocm_unified_memory_usable_bytes(&vrams, &gtt_totals, system_ram)
                            {
                                Some((vec![usable_bytes], true))
                            } else {
                                Some((vrams, false))
                            }
                        }
                    })();

                    if let Some((per_gpu, unified_memory)) = rocm_vram {
                        let vram: u64 = per_gpu.iter().sum();
                        survey.gpu_vram = per_gpu;
                        if unified_memory {
                            survey.is_soc = true;
                            survey.vram_bytes = vram;
                        } else {
                            let ram_offload = system_ram.saturating_sub(vram);
                            survey.vram_bytes = vram + (ram_offload as f64 * 0.75) as u64;
                        }
                    } else {
                        let intel_gpus: Option<Vec<XpuSmiGpuInfo>> = (|| {
                            for args in [["discovery", "--json"], ["discovery", "-j"]] {
                                let out = std::process::Command::new("xpu-smi")
                                    .args(args)
                                    .output()
                                    .ok()?;
                                if !out.status.success() {
                                    continue;
                                }
                                let stdout = String::from_utf8(out.stdout).ok()?;
                                let gpus = parse_xpu_smi_discovery_json(&stdout);
                                if !gpus.is_empty() {
                                    return Some(gpus);
                                }
                            }
                            None
                        })();

                        if let Some(intel_gpus) = intel_gpus {
                            // xpu-smi discovery reports capacity plus used
                            // bytes, but not a true reserved/unavailable
                            // metric, so leave reserved_bytes unavailable.
                            survey.gpu_reserved = vec![None; intel_gpus.len()];
                            let per_gpu: Vec<u64> = intel_gpus
                                .iter()
                                .map(|gpu| gpu.total_bytes.unwrap_or(0))
                                .collect();
                            let total: u64 = per_gpu.iter().sum();
                            survey.gpu_vram = per_gpu;
                            if total > 0 {
                                let ram_offload = system_ram.saturating_sub(total);
                                survey.vram_bytes = total + (ram_offload as f64 * 0.75) as u64;
                            } else if system_ram > 0 {
                                survey.vram_bytes = (system_ram as f64 * 0.75) as u64;
                            }
                        } else if system_ram > 0 {
                            // CPU-only (mesh.rs:320-322)
                            survey.vram_bytes = (system_ram as f64 * 0.75) as u64;
                        }
                    }
                }
            }

            if metrics.contains(&Metric::GpuName) || metrics.contains(&Metric::GpuCount) {
                let nvidia_names: Option<Vec<String>> = (|| {
                    let out = std::process::Command::new("nvidia-smi")
                        .args(["--query-gpu=name", "--format=csv,noheader"])
                        .output()
                        .ok()?;
                    if !out.status.success() {
                        return None;
                    }
                    let s = String::from_utf8(out.stdout).ok()?;
                    let names = parse_nvidia_gpu_names(&s);
                    if names.is_empty() {
                        None
                    } else {
                        Some(names)
                    }
                })();

                if let Some(ref names) = nvidia_names {
                    if metrics.contains(&Metric::GpuName) {
                        survey.gpu_name = summarize_gpu_name(names);
                    }
                    if metrics.contains(&Metric::GpuCount) {
                        survey.gpu_count = u8::try_from(names.len()).unwrap_or(u8::MAX);
                    }
                } else {
                    let out = std::process::Command::new("rocm-smi")
                        .args(["--showproductname"])
                        .output()
                        .ok();
                    if let Some(out) = out {
                        if out.status.success() {
                            if let Ok(s) = String::from_utf8(out.stdout) {
                                let names = parse_rocm_gpu_names(&s);
                                if metrics.contains(&Metric::GpuName) {
                                    survey.gpu_name = summarize_gpu_name(&names);
                                }
                                if metrics.contains(&Metric::GpuCount) {
                                    survey.gpu_count = u8::try_from(names.len()).unwrap_or(u8::MAX);
                                }
                            }
                        }
                    } else {
                        for args in [["discovery", "--json"], ["discovery", "-j"]] {
                            let out = std::process::Command::new("xpu-smi")
                                .args(args)
                                .output()
                                .ok();
                            if let Some(out) = out {
                                if out.status.success() {
                                    if let Ok(stdout) = String::from_utf8(out.stdout) {
                                        let gpus = parse_xpu_smi_discovery_json(&stdout);
                                        if !gpus.is_empty() {
                                            let names: Vec<String> =
                                                gpus.iter().map(|gpu| gpu.name.clone()).collect();
                                            if metrics.contains(&Metric::GpuName) {
                                                survey.gpu_name = summarize_gpu_name(&names);
                                            }
                                            if metrics.contains(&Metric::GpuCount) {
                                                survey.gpu_count =
                                                    u8::try_from(names.len()).unwrap_or(u8::MAX);
                                            }
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        #[cfg(target_os = "windows")]
        {
            let system_ram = read_windows_total_ram_bytes().unwrap_or(0);
            let want_gpu_info =
                metrics.contains(&Metric::GpuName) || metrics.contains(&Metric::GpuCount);
            let want_vram = metrics.contains(&Metric::VramBytes);

            let nvidia_names = if want_gpu_info {
                std::process::Command::new("nvidia-smi")
                    .args(["--query-gpu=name", "--format=csv,noheader"])
                    .output()
                    .ok()
                    .and_then(|out| {
                        if !out.status.success() {
                            return None;
                        }
                        let s = String::from_utf8(out.stdout).ok()?;
                        let names = parse_nvidia_gpu_names(&s);
                        if names.is_empty() {
                            None
                        } else {
                            Some(names)
                        }
                    })
            } else {
                None
            };

            let nvidia_vram = if want_vram {
                std::process::Command::new("nvidia-smi")
                    .args(["--query-gpu=memory.total", "--format=csv,noheader,nounits"])
                    .output()
                    .ok()
                    .and_then(|out| {
                        if !out.status.success() {
                            return None;
                        }
                        let s = String::from_utf8(out.stdout).ok()?;
                        let per_gpu = parse_nvidia_gpu_memory(&s);
                        if per_gpu.is_empty() {
                            None
                        } else {
                            Some(per_gpu)
                        }
                    })
            } else {
                None
            };

            let windows_gpus = if want_gpu_info || want_vram {
                read_windows_video_controllers()
            } else {
                Vec::new()
            };

            if want_vram {
                if let Some(per_gpu) = nvidia_vram {
                    let total: u64 = per_gpu.iter().sum();
                    if total > 0 {
                        survey.gpu_vram = per_gpu;
                        let ram_offload = system_ram.saturating_sub(total);
                        survey.vram_bytes = total + (ram_offload as f64 * 0.75) as u64;
                    }
                } else {
                    let per_gpu: Vec<u64> = windows_gpus
                        .iter()
                        .map(|(_, ram)| *ram)
                        .filter(|ram| *ram > 0)
                        .collect();
                    let total: u64 = per_gpu.iter().sum();
                    if total > 0 {
                        survey.gpu_vram = per_gpu;
                        let ram_offload = system_ram.saturating_sub(total);
                        survey.vram_bytes = total + (ram_offload as f64 * 0.75) as u64;
                    } else if system_ram > 0 {
                        survey.vram_bytes = (system_ram as f64 * 0.75) as u64;
                    }
                }
            }

            if want_gpu_info {
                if let Some(ref names) = nvidia_names {
                    if metrics.contains(&Metric::GpuName) {
                        survey.gpu_name = summarize_gpu_name(names);
                    }
                    if metrics.contains(&Metric::GpuCount) {
                        survey.gpu_count = u8::try_from(names.len()).unwrap_or(u8::MAX);
                    }
                } else {
                    let names: Vec<String> =
                        windows_gpus.iter().map(|(name, _)| name.clone()).collect();
                    if metrics.contains(&Metric::GpuName) {
                        survey.gpu_name = summarize_gpu_name(&names);
                    }
                    if metrics.contains(&Metric::GpuCount) {
                        survey.gpu_count = u8::try_from(names.len()).unwrap_or(u8::MAX);
                    }
                }
            }
        }

        survey
    }
}

#[cfg(target_os = "linux")]
impl Collector for TegraCollector {
    fn collect(&self, metrics: &[Metric]) -> HardwareSurvey {
        let mut survey = HardwareSurvey::default();

        if metrics.contains(&Metric::IsSoc) {
            survey.is_soc = true;
        }

        if metrics.contains(&Metric::GpuName) {
            if let Ok(model) = std::fs::read_to_string("/sys/firmware/devicetree/base/model") {
                survey.gpu_name = parse_tegra_model_name(&model);
            }
        }

        if metrics.contains(&Metric::VramBytes) {
            let total_ram = (|| -> Option<u64> {
                let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
                for line in meminfo.lines() {
                    if line.starts_with("MemTotal:") {
                        let kb = line.split_whitespace().nth(1)?.parse::<u64>().ok()?;
                        return Some(kb * 1024);
                    }
                }
                None
            })()
            .or_else(try_tegrastats_ram);
            if let Some(ram) = total_ram {
                survey.vram_bytes = (ram as f64 * 0.75) as u64;
                survey.gpu_vram = vec![ram];
            }
        }

        if metrics.contains(&Metric::GpuCount) {
            survey.gpu_count = 1;
        }

        survey
    }
}

#[cfg(target_os = "macos")]
fn detect_collector_impl() -> Box<dyn Collector> {
    Box::new(DefaultCollector)
}

#[cfg(target_os = "linux")]
fn detect_collector_impl() -> Box<dyn Collector> {
    if cfg!(target_arch = "aarch64") {
        if let Ok(compat) = std::fs::read_to_string("/proc/device-tree/compatible") {
            if is_tegra(&compat) {
                return Box::new(TegraCollector);
            }
        }
    }
    Box::new(DefaultCollector)
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn detect_collector_impl() -> Box<dyn Collector> {
    Box::new(DefaultCollector)
}

fn detect_collector() -> Box<dyn Collector> {
    detect_collector_impl()
}
