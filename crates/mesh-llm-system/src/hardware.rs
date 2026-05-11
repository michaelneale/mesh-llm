//! Hardware detection via Collector trait pattern.
//! VRAM formula preserved byte-identical from mesh.rs:detect_vram_bytes().

#[cfg(any(target_os = "windows", target_os = "linux", test))]
use serde_json::Value;

#[derive(Default, Debug, Clone, PartialEq)]
pub struct GpuFacts {
    pub index: usize,
    pub display_name: String,
    pub backend_device: Option<String>,
    pub vram_bytes: u64,
    pub reserved_bytes: Option<u64>,
    pub mem_bandwidth_gbps: Option<f64>,
    pub compute_tflops_fp32: Option<f64>,
    pub compute_tflops_fp16: Option<f64>,
    pub unified_memory: bool,
    pub stable_id: Option<String>,
    pub pci_bdf: Option<String>,
    pub vendor_uuid: Option<String>,
    pub metal_registry_id: Option<String>,
    pub dxgi_luid: Option<String>,
    pub pnp_instance_id: Option<String>,
}

#[derive(Default, Debug, Clone, PartialEq, Eq)]
pub struct VulkanGpuFacts {
    pub index: usize,
    pub display_name: String,
    pub device_type: String,
    pub vendor_id: Option<String>,
    pub device_id: Option<String>,
    pub device_uuid: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PinnedGpuResolverError {
    MissingConfiguredId {
        available_pinnable_ids: Vec<String>,
    },
    NonPinnableConfiguredId {
        configured_id: String,
        available_pinnable_ids: Vec<String>,
    },
    NoPinnableGpus {
        configured_id: String,
        available_pinnable_ids: Vec<String>,
    },
    NoMatch {
        configured_id: String,
        available_pinnable_ids: Vec<String>,
    },
    AmbiguousMatch {
        configured_id: String,
        available_pinnable_ids: Vec<String>,
        match_indexes: Vec<usize>,
    },
}

impl std::fmt::Display for PinnedGpuResolverError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingConfiguredId {
                available_pinnable_ids,
            } => write!(
                f,
                "missing configured gpu_id; available pinnable GPU IDs: {}",
                format_pinnable_gpu_ids(available_pinnable_ids)
            ),
            Self::NonPinnableConfiguredId {
                configured_id,
                available_pinnable_ids,
            } => write!(
                f,
                "configured gpu_id '{}' is not pinnable; available pinnable GPU IDs: {}",
                configured_id,
                format_pinnable_gpu_ids(available_pinnable_ids)
            ),
            Self::NoPinnableGpus {
                configured_id,
                available_pinnable_ids,
            } => write!(
                f,
                "configured gpu_id '{}' could not be resolved because this host has no pinnable GPUs; available pinnable GPU IDs: {}",
                configured_id,
                format_pinnable_gpu_ids(available_pinnable_ids)
            ),
            Self::NoMatch {
                configured_id,
                available_pinnable_ids,
            } => write!(
                f,
                "configured gpu_id '{}' did not match any available pinnable GPU; available pinnable GPU IDs: {}",
                configured_id,
                format_pinnable_gpu_ids(available_pinnable_ids)
            ),
            Self::AmbiguousMatch {
                configured_id,
                available_pinnable_ids,
                match_indexes,
            } => write!(
                f,
                "configured gpu_id '{}' matched multiple GPUs at indexes {:?}; available pinnable GPU IDs: {}",
                configured_id,
                match_indexes,
                format_pinnable_gpu_ids(available_pinnable_ids)
            ),
        }
    }
}

impl std::error::Error for PinnedGpuResolverError {}

#[derive(Default, Debug, Clone, PartialEq)]
pub struct HardwareSurvey {
    pub vram_bytes: u64,
    pub gpu_name: Option<String>,
    pub gpu_count: u8,
    pub hostname: Option<String>,
    pub is_soc: bool,
    /// Per-GPU VRAM in bytes, same order as gpu_name list.
    /// Unified-memory SoCs report a single entry.
    pub gpu_vram: Vec<u64>,
    /// Per-GPU reserved or otherwise unavailable bytes when the platform
    /// reports a true reserved/unavailable value. Do not populate this from
    /// live used-memory counters.
    pub gpu_reserved: Vec<Option<u64>>,
    /// Per-GPU facts in device-enumeration order.
    pub gpus: Vec<GpuFacts>,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum Metric {
    GpuName,
    VramBytes,
    GpuCount,
    Hostname,
    IsSoc,
    GpuFacts,
}

pub trait Collector {
    fn collect(&self, metrics: &[Metric]) -> HardwareSurvey;
}

struct DefaultCollector;

#[cfg(target_os = "linux")]
struct TegraCollector;

/// Parse `nvidia-smi --query-gpu=name --format=csv,noheader` output → GPU name list.
#[cfg(any(target_os = "linux", target_os = "windows", test))]
pub fn parse_nvidia_gpu_names(output: &str) -> Vec<String> {
    output
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .map(|l| l.to_string())
        .collect()
}

/// Parse `nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits` → per-GPU VRAM bytes.
#[cfg(any(target_os = "windows", test))]
pub fn parse_nvidia_gpu_memory(output: &str) -> Vec<u64> {
    output
        .lines()
        .filter_map(|line| {
            let mib = line.trim().parse::<u64>().ok()?;
            Some(mib * 1024 * 1024)
        })
        .collect()
}

#[cfg(any(target_os = "linux", target_os = "windows", test))]
pub fn parse_nvidia_gpu_memory_and_reserved(output: &str) -> Vec<(u64, Option<u64>)> {
    output
        .lines()
        .filter_map(|line| {
            let mut parts = line.split(',').map(str::trim);
            let total_mib = parts.next()?.parse::<u64>().ok()?;
            let reserved_mib = parts.next().and_then(|value| value.parse::<u64>().ok());
            Some((
                total_mib * 1024 * 1024,
                reserved_mib.map(|mib| mib * 1024 * 1024),
            ))
        })
        .collect()
}

/// Parse `sysctl -n machdep.cpu.brand_string` output → CPU brand string.
#[cfg(any(target_os = "macos", test))]
pub fn parse_macos_cpu_brand(output: &str) -> Option<String> {
    let s = output.trim();
    if s.is_empty() {
        None
    } else {
        Some(s.to_string())
    }
}

#[cfg(any(target_os = "macos", test))]
fn macos_metal_gpu_budget(metal_recommended_bytes: Option<u64>) -> Option<(u64, Option<u64>)> {
    metal_recommended_bytes
        .filter(|bytes| *bytes > 0)
        .map(|bytes| (bytes, None))
}

#[cfg(target_os = "macos")]
fn query_metal_recommended_working_set_bytes() -> Option<u64> {
    use std::ffi::{c_char, c_void, CStr};

    #[link(name = "Metal", kind = "framework")]
    extern "C" {
        fn MTLCreateSystemDefaultDevice() -> *mut c_void;
    }

    #[link(name = "objc")]
    extern "C" {
        fn sel_registerName(name: *const c_char) -> *mut c_void;
        fn objc_msgSend(receiver: *mut c_void, selector: *mut c_void, ...) -> usize;
    }

    unsafe {
        let device = MTLCreateSystemDefaultDevice();
        if device.is_null() {
            return None;
        }
        let selector = CStr::from_bytes_with_nul_unchecked(b"recommendedMaxWorkingSetSize\0");
        let selector = sel_registerName(selector.as_ptr());
        if selector.is_null() {
            return None;
        }
        let bytes = objc_msgSend(device, selector) as u64;
        (bytes > 0).then_some(bytes)
    }
}

/// Parse `rocm-smi --showproductname` output → GPU names from "Card series:" lines.
#[cfg(any(target_os = "linux", test))]
pub fn parse_rocm_gpu_names(output: &str) -> Vec<String> {
    let mut names = Vec::new();
    for line in output.lines() {
        let lower = line.to_ascii_lowercase();
        if let Some(pos) = lower.find("card series:") {
            let val = line[pos + "card series:".len()..].trim();
            if !val.is_empty() {
                names.push(val.to_string());
            }
        }
    }
    names
}

/// Parse `rocm-smi --showmeminfo vram --csv` output into per-GPU total bytes
/// and live used bytes. The used column is a utilization metric, not a
/// reserved/system-memory metric, so callers must not surface it as
/// `reserved_bytes`.
#[cfg(any(target_os = "linux", test))]
pub fn parse_rocm_gpu_memory_and_used(output: &str) -> Vec<(u64, Option<u64>)> {
    output
        .lines()
        .skip(1)
        .filter_map(|line| {
            let mut columns = line.split(',').map(str::trim);
            let _device = columns.next()?;
            let total = columns.next()?.parse::<u64>().ok()?;
            let used = columns.next().and_then(|value| value.parse::<u64>().ok());
            Some((total, used))
        })
        .collect()
}

#[cfg(any(target_os = "linux", test))]
const ROCM_UNIFIED_VRAM_MAX_BYTES: u64 = 2 * 1024 * 1024 * 1024;
#[cfg(any(target_os = "linux", test))]
const ROCM_UNIFIED_MIN_GTT_BYTES: u64 = 8 * 1024 * 1024 * 1024;

#[cfg(any(target_os = "linux", test))]
fn rocm_unified_memory_usable_bytes(
    vram_totals: &[u64],
    gtt_totals: &[u64],
    system_ram: u64,
) -> Option<u64> {
    if vram_totals.len() != 1 || gtt_totals.len() != 1 {
        return None;
    }
    let vram = vram_totals[0];
    let gtt = gtt_totals[0];
    if vram == 0
        || vram > ROCM_UNIFIED_VRAM_MAX_BYTES
        || gtt < ROCM_UNIFIED_MIN_GTT_BYTES
        || gtt < vram.saturating_mul(8)
    {
        return None;
    }

    let unified_total = if system_ram > 0 {
        gtt.min(system_ram)
    } else {
        gtt
    };
    Some((unified_total as f64 * 0.90) as u64)
}

#[cfg(any(target_os = "linux", test))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct XpuSmiGpuInfo {
    pub name: String,
    pub total_bytes: Option<u64>,
    pub used_bytes: Option<u64>,
}

#[cfg(any(target_os = "linux", test))]
fn xpu_json_string(map: &serde_json::Map<String, Value>, keys: &[&str]) -> Option<String> {
    keys.iter().find_map(|key| match map.get(*key) {
        Some(Value::String(value)) if !value.trim().is_empty() => Some(value.trim().to_string()),
        _ => None,
    })
}

#[cfg(any(target_os = "linux", test))]
fn xpu_json_u64(map: &serde_json::Map<String, Value>, keys: &[&str]) -> Option<u64> {
    keys.iter().find_map(|key| match map.get(*key) {
        Some(Value::Number(value)) => value.as_u64(),
        Some(Value::String(value)) => value.trim().parse::<u64>().ok(),
        _ => None,
    })
}

#[cfg(any(target_os = "linux", test))]
fn collect_xpu_smi_devices(value: &Value, devices: &mut Vec<XpuSmiGpuInfo>) {
    match value {
        Value::Object(map) => {
            let name = xpu_json_string(map, &["device_name", "deviceName", "name"]);
            let total_bytes = xpu_json_u64(
                map,
                &[
                    "memory_physical_size_byte",
                    "memoryPhysicalSizeByte",
                    "memory_total_bytes",
                    "memoryTotalBytes",
                    "memory_size_byte",
                    "memorySizeByte",
                    "lmem_total_bytes",
                    "lmemTotalBytes",
                ],
            );
            let used_bytes = xpu_json_u64(
                map,
                &[
                    "memory_used_byte",
                    "memoryUsedByte",
                    "memory_used_bytes",
                    "memoryUsedBytes",
                    "lmem_used_bytes",
                    "lmemUsedBytes",
                ],
            );
            if let Some(name) = name.filter(|_| total_bytes.is_some() || used_bytes.is_some()) {
                devices.push(XpuSmiGpuInfo {
                    name,
                    total_bytes,
                    used_bytes,
                });
            }
            for child in map.values() {
                collect_xpu_smi_devices(child, devices);
            }
        }
        Value::Array(values) => {
            for value in values {
                collect_xpu_smi_devices(value, devices);
            }
        }
        _ => {}
    }
}

#[cfg(any(target_os = "linux", test))]
pub fn parse_xpu_smi_discovery_json(output: &str) -> Vec<XpuSmiGpuInfo> {
    let Ok(value) = serde_json::from_str::<Value>(output) else {
        return Vec::new();
    };
    let mut devices = Vec::new();
    collect_xpu_smi_devices(&value, &mut devices);
    devices
}

/// Summarize GPU names: empty→None, 1→name, N identical→"N× name", N mixed→"a, b".
pub fn summarize_gpu_name(names: &[String]) -> Option<String> {
    match names.len() {
        0 => None,
        1 => Some(names[0].clone()),
        n => {
            let first = &names[0];
            if names.iter().all(|name| name == first) {
                Some(format!("{}× {}", n, first))
            } else {
                Some(names.join(", "))
            }
        }
    }
}

/// Expand a summarized GPU name string into per-device names.
/// - Splits comma-separated mixed GPU names.
/// - Expands summarized forms like `2× NVIDIA A100`.
/// - Falls back to repeating the raw summary to match `expected_count`.
pub fn expand_gpu_names(summary: Option<&str>, expected_count: usize) -> Vec<String> {
    let Some(raw) = summary.map(str::trim) else {
        return Vec::new();
    };
    if raw.is_empty() {
        return Vec::new();
    }

    let mut names = Vec::new();
    for part in raw.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        if let Some((count_str, name)) = part.split_once('×') {
            if let Ok(count) = count_str.trim().parse::<usize>() {
                let name = name.trim();
                if !name.is_empty() {
                    for _ in 0..count {
                        names.push(name.to_string());
                    }
                    continue;
                }
            }
        }
        names.push(part.to_string());
    }

    if expected_count > 0 && names.len() != expected_count {
        return vec![raw.to_string(); expected_count];
    }
    names
}

#[cfg(any(target_os = "linux", target_os = "windows", test))]
pub fn parse_nvidia_gpu_identity(output: &str) -> Vec<(Option<String>, Option<String>)> {
    fn normalize_identity_field(part: &str) -> Option<&str> {
        let part = part.trim();
        if part.is_empty() || part.eq_ignore_ascii_case("n/a") || part == "[N/A]" {
            None
        } else {
            Some(part)
        }
    }

    output
        .lines()
        .map(|line| {
            let mut parts = line.split(',').map(str::trim);
            let pci_bdf = parts
                .next()
                .and_then(normalize_identity_field)
                .map(|part| part.to_ascii_lowercase());
            let vendor_uuid = parts
                .next()
                .and_then(normalize_identity_field)
                .map(str::to_string);
            (pci_bdf, vendor_uuid)
        })
        .collect()
}

/// Check if a null-separated `/proc/device-tree/compatible` string contains a Tegra entry.
#[cfg(any(target_os = "linux", test))]
pub fn is_tegra(compatible: &str) -> bool {
    compatible.split('\0').any(|entry| entry.contains("tegra"))
}

/// Parse `/sys/firmware/devicetree/base/model` (null-terminated) → clean Jetson name.
/// Strips "NVIDIA " prefix and " Developer Kit" suffix.
#[cfg(any(target_os = "linux", test))]
pub fn parse_tegra_model_name(model: &str) -> Option<String> {
    let s = model.trim_matches('\0').trim();
    if s.is_empty() {
        return None;
    }
    let s = s.strip_prefix("NVIDIA ").unwrap_or(s);
    let s = s.strip_suffix(" Developer Kit").unwrap_or(s);
    Some(s.to_string())
}

/// Parse a `tegrastats` output line → total RAM bytes.
/// Handles optional timestamp prefix. No regex crate — plain string search.
#[cfg(any(target_os = "linux", test))]
pub fn parse_tegrastats_ram(output: &str) -> Option<u64> {
    let ram_pos = output.find("RAM ")?;
    let after_ram = &output[ram_pos + 4..];
    let slash_pos = after_ram.find('/')?;
    let after_slash = &after_ram[slash_pos + 1..];
    let mb_end = after_slash.find('M')?;
    let mb: u64 = after_slash[..mb_end].trim().parse().ok()?;
    Some(mb * 1024 * 1024)
}

/// Parse `hostname` command output → trimmed hostname string.
pub fn parse_hostname(output: &str) -> Option<String> {
    let s = output.trim();
    if s.is_empty() {
        None
    } else {
        Some(s.to_string())
    }
}

/// Parse PowerShell `Win32_VideoController | ConvertTo-Json` output → `(name, adapter_ram_bytes)`.
#[cfg(any(target_os = "windows", test))]
pub fn parse_windows_video_controller_json(output: &str) -> Vec<(String, u64)> {
    fn parse_u64(value: &Value) -> Option<u64> {
        match value {
            Value::Number(n) => n.as_u64(),
            Value::String(s) => s.trim().parse::<u64>().ok(),
            _ => None,
        }
    }

    fn parse_entry(value: &Value) -> Option<(String, u64)> {
        let name = value.get("Name")?.as_str()?.trim();
        if name.is_empty() {
            return None;
        }
        let adapter_ram = value.get("AdapterRAM").and_then(parse_u64).unwrap_or(0);
        Some((name.to_string(), adapter_ram))
    }

    let Ok(value) = serde_json::from_str::<Value>(output) else {
        return Vec::new();
    };

    match value {
        Value::Array(values) => values.iter().filter_map(parse_entry).collect(),
        Value::Object(_) => parse_entry(&value).into_iter().collect(),
        _ => Vec::new(),
    }
}

/// Parse `TotalPhysicalMemory` output from PowerShell/CIM.
#[cfg(any(target_os = "windows", test))]
pub fn parse_windows_total_physical_memory(output: &str) -> Option<u64> {
    output.trim().parse::<u64>().ok()
}

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
                if let Some((vram_bytes, reserved_bytes)) =
                    macos_metal_gpu_budget(query_metal_recommended_working_set_bytes())
                {
                    survey.vram_bytes = vram_bytes;
                    survey.gpu_vram = vec![vram_bytes];
                    survey.gpu_reserved = vec![reserved_bytes];
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
                    survey.vram_bytes = vram + (ram_offload as f64 * 0.90) as u64;
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
                            survey.vram_bytes = vram + (ram_offload as f64 * 0.90) as u64;
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
                                survey.vram_bytes = total + (ram_offload as f64 * 0.90) as u64;
                            } else if system_ram > 0 {
                                survey.vram_bytes = (system_ram as f64 * 0.90) as u64;
                            }
                        } else if system_ram > 0 {
                            // CPU-only (mesh.rs:320-322)
                            survey.vram_bytes = (system_ram as f64 * 0.90) as u64;
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
                        survey.vram_bytes = total + (ram_offload as f64 * 0.90) as u64;
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
                        survey.vram_bytes = total + (ram_offload as f64 * 0.90) as u64;
                    } else if system_ram > 0 {
                        survey.vram_bytes = (system_ram as f64 * 0.90) as u64;
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
                survey.vram_bytes = (ram as f64 * 0.90) as u64;
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
            (unified_memory_bytes as f64 * 0.90) as u64
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

/// Collect only the requested hardware metrics.
pub fn query(metrics: &[Metric]) -> HardwareSurvey {
    let collector = detect_collector();
    let mut survey = collector.collect(metrics);
    if metrics.contains(&Metric::Hostname) {
        survey.hostname = detect_hostname();
    }
    if metrics.contains(&Metric::GpuFacts) {
        hydrate_gpu_facts(&mut survey, metrics);
    }
    survey
}

pub fn survey() -> HardwareSurvey {
    query(&[
        Metric::GpuName,
        Metric::VramBytes,
        Metric::GpuCount,
        Metric::Hostname,
        Metric::IsSoc,
        Metric::GpuFacts,
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_gpu(index: usize, stable_id: Option<&str>) -> GpuFacts {
        GpuFacts {
            index,
            display_name: format!("GPU {index}"),
            backend_device: Some(format!("CUDA{index}")),
            vram_bytes: 24_000_000_000,
            reserved_bytes: None,
            mem_bandwidth_gbps: None,
            compute_tflops_fp32: None,
            compute_tflops_fp16: None,
            unified_memory: false,
            stable_id: stable_id.map(str::to_string),
            pci_bdf: None,
            vendor_uuid: None,
            metal_registry_id: None,
            dxgi_luid: None,
            pnp_instance_id: None,
        }
    }

    #[test]
    fn test_parse_vulkaninfo_summary_devices() {
        let fixture = r#"
Devices:
========
GPU0:
    vendorID           = 0x10de
    deviceID           = 0x2c02
    deviceType         = PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
    deviceName         = NVIDIA GeForce RTX 5080
    deviceUUID         = 459fc93e-aae5-c491-2080-5b93901cdcce
GPU1:
    vendorID           = 0x1002
    deviceID           = 0x164e
    deviceType         = PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU
    deviceName         = AMD Ryzen 7 7800X3D 8-Core Processor (RADV RAPHAEL_MENDOCINO)
    deviceUUID         = 00000000-0c00-0000-0000-000000000000
GPU2:
    deviceType         = PHYSICAL_DEVICE_TYPE_CPU
    deviceName         = llvmpipe
"#;

        let devices = parse_vulkaninfo_summary_devices(fixture);

        assert_eq!(devices.len(), 3);
        assert_eq!(devices[0].index, 0);
        assert_eq!(devices[0].display_name, "NVIDIA GeForce RTX 5080");
        assert_eq!(devices[1].index, 1);
        assert_eq!(
            devices[1].device_uuid.as_deref(),
            Some("00000000-0c00-0000-0000-000000000000")
        );
    }

    #[test]
    fn test_merge_vulkan_devices_adds_integrated_pci_device() {
        let mut gpus = vec![GpuFacts {
            vendor_uuid: Some("GPU-459fc93e-aae5-c491-2080-5b93901cdcce".to_string()),
            stable_id: Some("pci:00000000:01:00.0".to_string()),
            pci_bdf: Some("00000000:01:00.0".to_string()),
            ..synthetic_gpu(0, Some("pci:00000000:01:00.0"))
        }];
        let devices = vec![
            VulkanGpuFacts {
                index: 0,
                display_name: "NVIDIA GeForce RTX 5080".to_string(),
                device_type: "PHYSICAL_DEVICE_TYPE_DISCRETE_GPU".to_string(),
                device_uuid: Some("459fc93e-aae5-c491-2080-5b93901cdcce".to_string()),
                ..Default::default()
            },
            VulkanGpuFacts {
                index: 1,
                display_name: "AMD Ryzen 7 7800X3D 8-Core Processor (RADV RAPHAEL_MENDOCINO)"
                    .to_string(),
                device_type: "PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU".to_string(),
                device_uuid: Some("00000000-0c00-0000-0000-000000000000".to_string()),
                ..Default::default()
            },
            VulkanGpuFacts {
                index: 2,
                display_name: "llvmpipe".to_string(),
                device_type: "PHYSICAL_DEVICE_TYPE_CPU".to_string(),
                ..Default::default()
            },
        ];

        merge_vulkan_devices_into_gpu_facts(&mut gpus, &devices, 64 * 1024 * 1024 * 1024);

        assert_eq!(gpus.len(), 2);
        assert_eq!(gpus[0].backend_device.as_deref(), Some("Vulkan0"));
        assert_eq!(gpus[1].stable_id.as_deref(), Some("pci:00000000:0c:00.0"));
        assert_eq!(gpus[1].backend_device.as_deref(), Some("Vulkan1"));
        assert_eq!(gpus[1].pci_bdf.as_deref(), Some("00000000:0c:00.0"));
        assert!(gpus[1].unified_memory);
        assert!(gpus[1].vram_bytes > 0);
    }

    #[test]
    fn test_parse_nvidia_gpu_name_single() {
        let names = parse_nvidia_gpu_names("NVIDIA A100-SXM4-80GB\n");
        assert_eq!(names, vec!["NVIDIA A100-SXM4-80GB"]);
    }

    #[test]
    fn test_parse_nvidia_gpu_name_multi_identical() {
        let names = parse_nvidia_gpu_names("NVIDIA A100\nNVIDIA A100\n");
        assert_eq!(names.len(), 2);
        assert_eq!(names[0], "NVIDIA A100");
        assert_eq!(names[1], "NVIDIA A100");
    }

    #[test]
    fn test_parse_nvidia_gpu_name_multi_mixed() {
        let names = parse_nvidia_gpu_names("NVIDIA A100\nNVIDIA RTX 4090\n");
        assert_eq!(names.len(), 2);
        assert_eq!(names[0], "NVIDIA A100");
        assert_eq!(names[1], "NVIDIA RTX 4090");
    }

    #[test]
    fn test_parse_nvidia_gpu_name_empty() {
        assert!(parse_nvidia_gpu_names("").is_empty());
    }

    #[test]
    fn test_parse_nvidia_gpu_memory() {
        assert_eq!(
            parse_nvidia_gpu_memory("81920\n24576\n"),
            vec![81_920u64 * 1024 * 1024, 24_576u64 * 1024 * 1024]
        );
    }

    #[test]
    fn test_parse_nvidia_gpu_memory_and_reserved() {
        assert_eq!(
            parse_nvidia_gpu_memory_and_reserved("81920,1024\n24576,0\n"),
            vec![
                (81_920u64 * 1024 * 1024, Some(1_024u64 * 1024 * 1024)),
                (24_576u64 * 1024 * 1024, Some(0)),
            ]
        );
    }

    #[test]
    fn test_parse_macos_cpu_brand() {
        assert_eq!(
            parse_macos_cpu_brand("Apple M4 Max\n"),
            Some("Apple M4 Max".to_string())
        );
    }

    #[test]
    fn test_parse_macos_cpu_brand_empty() {
        assert_eq!(parse_macos_cpu_brand(""), None);
    }

    #[test]
    fn test_macos_gpu_budget_uses_metal_working_set_as_vram() {
        let metal_bytes = 107_u64 * 1024 * 1024 * 1024;

        assert_eq!(
            macos_metal_gpu_budget(Some(metal_bytes)),
            Some((metal_bytes, None))
        );
    }

    #[test]
    fn test_macos_gpu_budget_is_unavailable_without_metal_working_set() {
        assert_eq!(macos_metal_gpu_budget(Some(0)), None);
        assert_eq!(macos_metal_gpu_budget(None), None);
    }

    #[test]
    fn test_parse_rocm_gpu_names_single() {
        let fixture = "\
======================= ROCm System Management Interface =======================
================================= Product Info =================================
GPU[0]\t\t: Card Series:\t\t\tNavi31 [Radeon RX 7900 XTX]
================================================================================";
        assert_eq!(
            parse_rocm_gpu_names(fixture),
            vec!["Navi31 [Radeon RX 7900 XTX]".to_string()]
        );
    }

    #[test]
    fn test_parse_rocm_gpu_names_multi() {
        let fixture = "\
======================= ROCm System Management Interface =======================
================================= Product Info =================================
GPU[0]\t\t: Card series:\t\t\tAMD Instinct MI300X
GPU[1]\t\t: Card series:\t\t\tAMD Instinct MI300X
================================================================================";
        assert_eq!(
            parse_rocm_gpu_names(fixture),
            vec![
                "AMD Instinct MI300X".to_string(),
                "AMD Instinct MI300X".to_string()
            ]
        );
    }

    #[test]
    fn test_rocm_tiny_vram_large_gtt_is_unified_memory() {
        let vram = vec![536_870_912];
        let gtt = vec![137_438_953_472];
        let system_ram = 137_438_953_472;

        assert_eq!(
            rocm_unified_memory_usable_bytes(&vram, &gtt, system_ram),
            Some(123_695_058_124)
        );
    }

    #[test]
    fn test_rocm_discrete_vram_large_gtt_is_not_unified_memory() {
        let vram = vec![24 * 1024 * 1024 * 1024];
        let gtt = vec![137_438_953_472];
        let system_ram = 137_438_953_472;

        assert_eq!(
            rocm_unified_memory_usable_bytes(&vram, &gtt, system_ram),
            None
        );
    }

    #[test]
    fn test_parse_rocm_gpu_memory_and_used() {
        let fixture = "\
device,VRAM Total Memory (B),VRAM Total Used Memory (B)
card0,25753026560,416378880
card1,25753026560,512000000";
        assert_eq!(
            parse_rocm_gpu_memory_and_used(fixture),
            vec![
                (25_753_026_560, Some(416_378_880)),
                (25_753_026_560, Some(512_000_000)),
            ]
        );
    }

    #[test]
    fn test_parse_xpu_smi_discovery_json() {
        let fixture = r#"{
          "devices": [
            {
              "device_name": "Intel Arc A770",
              "memory_physical_size_byte": 17179869184,
              "memory_used_byte": 536870912
            },
            {
              "device_name": "Intel Arc B580",
              "memory_physical_size_byte": "12884901888",
              "memory_used_byte": "268435456"
            }
          ]
        }"#;
        assert_eq!(
            parse_xpu_smi_discovery_json(fixture),
            vec![
                XpuSmiGpuInfo {
                    name: "Intel Arc A770".to_string(),
                    total_bytes: Some(17_179_869_184),
                    used_bytes: Some(536_870_912),
                },
                XpuSmiGpuInfo {
                    name: "Intel Arc B580".to_string(),
                    total_bytes: Some(12_884_901_888),
                    used_bytes: Some(268_435_456),
                },
            ]
        );
    }

    #[test]
    fn test_rocm_used_memory_does_not_surface_as_reserved_bytes() {
        let fixture = "\
device,VRAM Total Memory (B),VRAM Total Used Memory (B)
card0,25753026560,416378880
card1,25753026560,512000000";
        let parsed = parse_rocm_gpu_memory_and_used(fixture);
        let mut survey = HardwareSurvey {
            gpu_vram: parsed.iter().map(|(total, _)| *total).collect(),
            gpu_reserved: vec![None; parsed.len()],
            ..Default::default()
        };

        hydrate_gpu_facts(&mut survey, &[Metric::GpuFacts]);

        assert_eq!(survey.gpus.len(), 2);
        assert!(survey.gpus.iter().all(|gpu| gpu.reserved_bytes.is_none()));
    }

    #[test]
    fn test_xpu_used_memory_does_not_surface_as_reserved_bytes() {
        let fixture = r#"{
          "devices": [
            {
              "device_name": "Intel Arc A770",
              "memory_physical_size_byte": 17179869184,
              "memory_used_byte": 536870912
            },
            {
              "device_name": "Intel Arc B580",
              "memory_physical_size_byte": "12884901888",
              "memory_used_byte": "268435456"
            }
          ]
        }"#;
        let gpus = parse_xpu_smi_discovery_json(fixture);
        let mut survey = HardwareSurvey {
            gpu_vram: gpus
                .iter()
                .map(|gpu| gpu.total_bytes.unwrap_or(0))
                .collect(),
            gpu_reserved: vec![None; gpus.len()],
            ..Default::default()
        };

        hydrate_gpu_facts(&mut survey, &[Metric::GpuFacts]);

        assert_eq!(survey.gpus.len(), 2);
        assert!(survey.gpus.iter().all(|gpu| gpu.reserved_bytes.is_none()));
    }

    #[test]
    fn test_hydrate_gpu_facts_uses_uuid_and_cuda_for_tegra_soc() {
        let mut survey = HardwareSurvey {
            gpu_name: Some("Jetson AGX Orin".to_string()),
            gpu_count: 1,
            gpu_vram: vec![65_890_271_232],
            is_soc: true,
            ..Default::default()
        };
        let identities = vec![(
            None,
            Some("ddae9891-aaa8-5edd-bbf3-3a33c5adc75f".to_string()),
        )];
        let expected_count = survey
            .gpu_vram
            .len()
            .max(usize::from(survey.gpu_count))
            .max(inferred_gpu_name_count(survey.gpu_name.as_deref()));
        let names = expand_gpu_names(survey.gpu_name.as_deref(), expected_count);

        hydrate_gpu_facts_with_identities(
            &mut survey,
            &[Metric::GpuFacts],
            &identities,
            names,
            expected_count,
            false,
        );

        assert_eq!(survey.gpus.len(), 1);
        assert_eq!(survey.gpus[0].display_name, "Jetson AGX Orin");
        assert_eq!(survey.gpus[0].backend_device.as_deref(), Some("CUDA0"));
        assert_eq!(
            survey.gpus[0].stable_id.as_deref(),
            Some("uuid:ddae9891-aaa8-5edd-bbf3-3a33c5adc75f")
        );
        assert_eq!(survey.gpus[0].pci_bdf, None);
        assert_eq!(
            survey.gpus[0].vendor_uuid.as_deref(),
            Some("ddae9891-aaa8-5edd-bbf3-3a33c5adc75f")
        );
        assert!(survey.gpus[0].unified_memory);
    }

    #[test]
    fn test_summarize_gpu_name_single() {
        assert_eq!(
            summarize_gpu_name(&["A100".to_string()]),
            Some("A100".to_string())
        );
    }

    #[test]
    fn test_summarize_gpu_name_identical() {
        assert_eq!(
            summarize_gpu_name(&["A100".to_string(), "A100".to_string()]),
            Some("2\u{00D7} A100".to_string())
        );
    }

    #[test]
    fn test_summarize_gpu_name_mixed() {
        assert_eq!(
            summarize_gpu_name(&["A100".to_string(), "RTX 4090".to_string()]),
            Some("A100, RTX 4090".to_string())
        );
    }

    #[test]
    fn test_summarize_gpu_name_empty() {
        assert_eq!(summarize_gpu_name(&[]), None);
    }

    #[test]
    fn test_expand_gpu_names_identical_summary() {
        assert_eq!(
            expand_gpu_names(Some("2× NVIDIA A100"), 2),
            vec!["NVIDIA A100".to_string(), "NVIDIA A100".to_string()]
        );
    }

    #[test]
    fn test_expand_gpu_names_mixed_summary() {
        assert_eq!(
            expand_gpu_names(Some("NVIDIA A100, NVIDIA RTX 4090"), 2),
            vec!["NVIDIA A100".to_string(), "NVIDIA RTX 4090".to_string()]
        );
    }

    #[test]
    fn test_parse_nvidia_gpu_identity_rows() {
        let identities =
            parse_nvidia_gpu_identity("00000000:65:00.0, GPU-abc\n00000000:b3:00.0, GPU-def\n");
        assert_eq!(
            identities,
            vec![
                (
                    Some("00000000:65:00.0".to_string()),
                    Some("GPU-abc".to_string())
                ),
                (
                    Some("00000000:b3:00.0".to_string()),
                    Some("GPU-def".to_string())
                )
            ]
        );
    }

    #[test]
    fn test_parse_nvidia_gpu_identity_ignores_not_available_placeholders() {
        let identities = parse_nvidia_gpu_identity("[N/A], ddae9891-aaa8-5edd-bbf3-3a33c5adc75f\n");
        assert_eq!(
            identities,
            vec![(
                None,
                Some("ddae9891-aaa8-5edd-bbf3-3a33c5adc75f".to_string())
            )]
        );
    }

    #[test]
    fn test_backend_device_for_name_recognizes_jetson_soc_names() {
        assert_eq!(
            backend_device_for_name_for_platform("Jetson AGX Orin", 0, true, false),
            Some("CUDA0".to_string())
        );
    }

    #[test]
    fn test_backend_device_for_name_recognizes_nvgpu_soc_names() {
        assert_eq!(
            backend_device_for_name_for_platform("Orin (nvgpu)", 1, true, false),
            Some("CUDA1".to_string())
        );
    }

    #[test]
    fn pinned_gpu_runtime_resolver_accepts_single_match() {
        let gpus = vec![
            synthetic_gpu(0, Some("pci:0000:65:00.0")),
            synthetic_gpu(1, Some("uuid:GPU-def")),
        ];

        let resolved = resolve_pinned_gpu(Some("pci:0000:65:00.0"), &gpus).unwrap();

        assert_eq!(resolved.index, 0);
        assert_eq!(resolved.stable_id.as_deref(), Some("pci:0000:65:00.0"));
    }

    #[test]
    fn pinned_gpu_runtime_resolver_missing_configured_id_fails() {
        let gpus = vec![synthetic_gpu(0, Some("pci:0000:65:00.0"))];

        let err = resolve_pinned_gpu(None, &gpus).unwrap_err();

        assert_eq!(
            err,
            PinnedGpuResolverError::MissingConfiguredId {
                available_pinnable_ids: vec!["pci:0000:65:00.0".to_string()],
            }
        );
        assert!(err
            .to_string()
            .contains("available pinnable GPU IDs: pci:0000:65:00.0"));
    }

    #[test]
    fn pinned_gpu_runtime_resolver_no_match_lists_available_ids() {
        let gpus = vec![
            synthetic_gpu(0, Some("pci:0000:65:00.0")),
            synthetic_gpu(1, Some("uuid:GPU-def")),
        ];

        let err = resolve_pinned_gpu(Some("pci:0000:b3:00.0"), &gpus).unwrap_err();

        assert_eq!(
            err,
            PinnedGpuResolverError::NoMatch {
                configured_id: "pci:0000:b3:00.0".to_string(),
                available_pinnable_ids: vec![
                    "pci:0000:65:00.0".to_string(),
                    "uuid:GPU-def".to_string(),
                ],
            }
        );
        assert!(err.to_string().contains("pci:0000:b3:00.0"));
        assert!(err.to_string().contains("pci:0000:65:00.0, uuid:GPU-def"));
    }

    #[test]
    fn pinned_gpu_runtime_resolver_duplicate_match_fails() {
        let gpus = vec![
            synthetic_gpu(0, Some("uuid:GPU-shared")),
            synthetic_gpu(1, Some("uuid:GPU-shared")),
        ];

        let err = resolve_pinned_gpu(Some("uuid:GPU-shared"), &gpus).unwrap_err();

        assert_eq!(
            err,
            PinnedGpuResolverError::AmbiguousMatch {
                configured_id: "uuid:GPU-shared".to_string(),
                available_pinnable_ids: vec![
                    "uuid:GPU-shared".to_string(),
                    "uuid:GPU-shared".to_string(),
                ],
                match_indexes: vec![0, 1],
            }
        );
        assert!(err.to_string().contains("indexes [0, 1]"));
    }

    #[test]
    fn pinned_gpu_runtime_resolver_rejects_index_fallback_ids() {
        let gpus = vec![synthetic_gpu(0, Some("pci:0000:65:00.0"))];

        let err = resolve_pinned_gpu(Some("index:0"), &gpus).unwrap_err();

        assert_eq!(
            err,
            PinnedGpuResolverError::NonPinnableConfiguredId {
                configured_id: "index:0".to_string(),
                available_pinnable_ids: vec!["pci:0000:65:00.0".to_string()],
            }
        );
        assert!(err.to_string().contains("not pinnable"));
    }

    #[test]
    fn pinned_gpu_runtime_resolver_rejects_backend_device_fallback_ids() {
        let gpus = vec![synthetic_gpu(0, Some("pci:0000:65:00.0"))];

        let err = resolve_pinned_gpu(Some("cuda0"), &gpus).unwrap_err();

        assert_eq!(
            err,
            PinnedGpuResolverError::NonPinnableConfiguredId {
                configured_id: "cuda0".to_string(),
                available_pinnable_ids: vec!["pci:0000:65:00.0".to_string()],
            }
        );
    }

    #[test]
    fn pinned_gpu_runtime_resolver_fails_when_host_has_no_pinnable_gpus() {
        let gpus = vec![
            synthetic_gpu(0, Some("cuda0")),
            synthetic_gpu(1, Some("index:1")),
        ];

        let err = resolve_pinned_gpu(Some("pci:0000:65:00.0"), &gpus).unwrap_err();

        assert_eq!(
            err,
            PinnedGpuResolverError::NoPinnableGpus {
                configured_id: "pci:0000:65:00.0".to_string(),
                available_pinnable_ids: vec![],
            }
        );
        assert!(err.to_string().contains("available pinnable GPU IDs: none"));
    }

    #[test]
    fn test_hardware_survey_default() {
        let s = HardwareSurvey::default();
        assert_eq!(s.vram_bytes, 0);
        assert_eq!(s.gpu_name, None);
        assert_eq!(s.gpu_count, 0);
        assert_eq!(s.hostname, None);
        assert!(s.gpu_vram.is_empty());
        assert!(s.gpu_reserved.is_empty());
        assert!(s.gpus.is_empty());
    }

    #[test]
    fn test_query_gpu_name_only() {
        let result = query(&[Metric::GpuName]);
        assert_eq!(result.vram_bytes, 0);
        assert_eq!(result.hostname, None);
    }

    #[test]
    fn test_query_vram_only() {
        let result = query(&[Metric::VramBytes]);
        assert_eq!(result.gpu_name, None);
        assert_eq!(result.hostname, None);
    }

    #[test]
    fn test_query_multiple_metrics() {
        let result = query(&[Metric::GpuName, Metric::VramBytes]);
        assert_eq!(result.hostname, None);
        assert_eq!(result.gpu_count, 0);
    }

    #[test]
    fn test_survey_returns_all_metrics() {
        let s = survey();
        let q = query(&[
            Metric::GpuName,
            Metric::VramBytes,
            Metric::GpuCount,
            Metric::Hostname,
        ]);
        assert_eq!(s.vram_bytes, q.vram_bytes);
        assert_eq!(s.gpu_name, q.gpu_name);
        assert_eq!(s.gpu_count, q.gpu_count);
        assert_eq!(s.hostname.is_some(), q.hostname.is_some());
    }

    #[test]
    fn test_is_tegra_positive() {
        assert!(is_tegra("nvidia,p3737-0000+p3701-0005\0nvidia,tegra234\0"));
    }

    #[test]
    fn test_is_tegra_negative_arm() {
        assert!(!is_tegra("raspberrypi,4-model-b\0"));
    }

    #[test]
    fn test_parse_tegra_model_name() {
        assert_eq!(
            parse_tegra_model_name("NVIDIA Jetson AGX Orin Developer Kit\0"),
            Some("Jetson AGX Orin".to_string())
        );
    }

    #[test]
    fn test_parse_tegra_model_name_nano() {
        assert_eq!(
            parse_tegra_model_name("NVIDIA Jetson Orin Nano Developer Kit\0"),
            Some("Jetson Orin Nano".to_string())
        );
    }

    #[test]
    fn test_parse_tegra_model_name_no_prefix() {
        assert_eq!(
            parse_tegra_model_name("Jetson Xavier NX\0"),
            Some("Jetson Xavier NX".to_string())
        );
    }

    #[test]
    fn test_parse_tegrastats_ram() {
        let line = "RAM 14640/62838MB (lfb 11x4MB) CPU [0%@729,off,off,off,0%@729,off,off,off]";
        assert_eq!(parse_tegrastats_ram(line), Some(62838u64 * 1024 * 1024));
    }

    #[test]
    fn test_parse_tegrastats_ram_with_timestamp() {
        let line = "12-27-2022 13:48:01 RAM 14640/62838MB (lfb 11x4MB)";
        assert_eq!(parse_tegrastats_ram(line), Some(62838u64 * 1024 * 1024));
    }

    #[test]
    fn test_parse_tegrastats_ram_empty() {
        assert_eq!(parse_tegrastats_ram(""), None);
    }

    #[test]
    fn test_parse_hostname() {
        assert_eq!(parse_hostname("lemony-28\n"), Some("lemony-28".to_string()));
    }

    #[test]
    fn test_parse_hostname_empty() {
        assert_eq!(parse_hostname(""), None);
    }

    #[test]
    fn test_parse_hostname_whitespace() {
        assert_eq!(parse_hostname("  carrack  \n"), Some("carrack".to_string()));
    }

    #[test]
    fn test_parse_windows_video_controller_json_array() {
        let json = r#"[{"Name":"NVIDIA RTX 4090","AdapterRAM":25769803776},{"Name":"AMD Radeon PRO","AdapterRAM":"8589934592"}]"#;
        assert_eq!(
            parse_windows_video_controller_json(json),
            vec![
                ("NVIDIA RTX 4090".to_string(), 25_769_803_776),
                ("AMD Radeon PRO".to_string(), 8_589_934_592),
            ]
        );
    }

    #[test]
    fn test_parse_windows_video_controller_json_single_object() {
        let json = r#"{"Name":"NVIDIA RTX 5090","AdapterRAM":34359738368}"#;
        assert_eq!(
            parse_windows_video_controller_json(json),
            vec![("NVIDIA RTX 5090".to_string(), 34_359_738_368)]
        );
    }

    #[test]
    fn test_parse_windows_total_physical_memory() {
        assert_eq!(
            parse_windows_total_physical_memory("68719476736\r\n"),
            Some(68_719_476_736)
        );
    }

    #[test]
    fn test_is_tegra_negative_x86() {
        assert!(!is_tegra(""));
    }

    #[test]
    fn test_query_hostname_only() {
        let result = query(&[Metric::Hostname]);
        assert_eq!(result.gpu_name, None);
        assert_eq!(result.gpu_count, 0);
        assert_eq!(result.vram_bytes, 0);
    }

    #[test]
    fn test_detect_collector_returns_default_on_non_tegra() {
        let collector = detect_collector();
        let s = collector.collect(&[Metric::VramBytes]);
        let _ = s.vram_bytes;
    }

    #[test]
    fn test_query_is_soc_only() {
        let result = query(&[Metric::IsSoc]);
        assert_eq!(result.vram_bytes, 0);
        assert_eq!(result.gpu_name, None);
        assert_eq!(result.gpu_count, 0);
        assert_eq!(result.hostname, None);
        let _ = result.is_soc;
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn test_macos_is_soc_true() {
        let result = DefaultCollector.collect(&[Metric::IsSoc]);
        assert!(
            result.is_soc,
            "macOS DefaultCollector must report is_soc=true"
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_tegra_is_soc_true() {
        let result = TegraCollector.collect(&[Metric::IsSoc]);
        assert!(result.is_soc, "TegraCollector must report is_soc=true");
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_linux_discrete_is_soc_false() {
        let result = DefaultCollector.collect(&[Metric::IsSoc]);
        assert!(
            !result.is_soc,
            "Linux DefaultCollector must report is_soc=false"
        );
    }

    #[test]
    fn test_default_collector_nvidia_fixture() {
        let names = parse_nvidia_gpu_names("NVIDIA A100\n");
        assert_eq!(names, vec!["NVIDIA A100"]);
        assert_eq!(
            summarize_gpu_name(&["NVIDIA A100".to_string()]),
            Some("NVIDIA A100".to_string())
        );
    }

    #[test]
    fn test_tegra_collector_sysfs_fixture() {
        assert_eq!(
            parse_tegra_model_name("NVIDIA Jetson AGX Orin Developer Kit\0"),
            Some("Jetson AGX Orin".to_string())
        );
    }
}
