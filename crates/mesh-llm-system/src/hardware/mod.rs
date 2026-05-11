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

include!("parsers.rs");
include!("collectors.rs");
include!("gpu_facts.rs");
include!("vulkan.rs");
include!("query.rs");
include!("tests.rs");
