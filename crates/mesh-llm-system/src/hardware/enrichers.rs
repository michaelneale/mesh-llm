use super::GpuFacts;

#[cfg(target_os = "linux")]
mod linux {
    use super::GpuFacts;
    use libc::{c_char, c_int, c_uint, c_void};
    use std::ffi::CStr;

    #[derive(Clone, Debug, Default)]
    struct NvidiaDeviceInfo {
        name: Option<String>,
        pci_bdf: Option<String>,
        total_bytes: Option<u64>,
        reserved_bytes: Option<u64>,
        uuid: Option<String>,
    }

    struct DlLibrary(*mut c_void);

    impl DlLibrary {
        fn open(name: &'static [u8]) -> Option<Self> {
            let handle = unsafe { libc::dlopen(name.as_ptr().cast(), libc::RTLD_LAZY) };
            if handle.is_null() {
                None
            } else {
                Some(Self(handle))
            }
        }

        unsafe fn symbol<T: Copy>(&self, name: &'static [u8]) -> Option<T> {
            let symbol = libc::dlsym(self.0, name.as_ptr().cast());
            if symbol.is_null() {
                None
            } else {
                Some(std::mem::transmute_copy(&symbol))
            }
        }
    }

    impl Drop for DlLibrary {
        fn drop(&mut self) {
            unsafe {
                libc::dlclose(self.0);
            }
        }
    }

    type CuDevice = c_int;
    type CuResult = c_int;
    type CuInit = unsafe extern "C" fn(c_uint) -> CuResult;
    type CuDeviceGetCount = unsafe extern "C" fn(*mut c_int) -> CuResult;
    type CuDeviceGet = unsafe extern "C" fn(*mut CuDevice, c_int) -> CuResult;
    type CuDeviceGetName = unsafe extern "C" fn(*mut c_char, c_int, CuDevice) -> CuResult;
    type CuDeviceTotalMem = unsafe extern "C" fn(*mut usize, CuDevice) -> CuResult;
    type CuDeviceGetPciBusId = unsafe extern "C" fn(*mut c_char, c_int, CuDevice) -> CuResult;

    type NvmlDevice = *mut c_void;
    type NvmlReturn = c_int;
    type NvmlInit = unsafe extern "C" fn() -> NvmlReturn;
    type NvmlShutdown = unsafe extern "C" fn() -> NvmlReturn;
    type NvmlDeviceGetCount = unsafe extern "C" fn(*mut c_uint) -> NvmlReturn;
    type NvmlDeviceGetHandleByIndex = unsafe extern "C" fn(c_uint, *mut NvmlDevice) -> NvmlReturn;
    type NvmlDeviceGetUuid = unsafe extern "C" fn(NvmlDevice, *mut c_char, c_uint) -> NvmlReturn;
    type NvmlDeviceGetMemoryInfo = unsafe extern "C" fn(NvmlDevice, *mut NvmlMemory) -> NvmlReturn;
    type NvmlDeviceGetMemoryInfoV2 =
        unsafe extern "C" fn(NvmlDevice, *mut NvmlMemoryV2) -> NvmlReturn;

    #[repr(C)]
    #[derive(Default)]
    struct NvmlMemory {
        total: u64,
        free: u64,
        used: u64,
    }

    #[repr(C)]
    #[derive(Default)]
    struct NvmlMemoryV2 {
        version: c_uint,
        total: u64,
        reserved: u64,
        free: u64,
        used: u64,
    }

    const NVML_SUCCESS: NvmlReturn = 0;
    const CUDA_SUCCESS: CuResult = 0;

    pub(crate) fn enrich_gpu_facts(gpus: &mut [GpuFacts]) {
        let mut infos = cuda_device_infos();
        merge_nvml_device_infos(&mut infos);
        if infos.is_empty() {
            return;
        }

        for gpu in gpus {
            let Some(info) = match_nvidia_device(gpu, &infos) else {
                continue;
            };
            if let Some(total_bytes) = info.total_bytes {
                gpu.vram_bytes = total_bytes;
            }
            if info.reserved_bytes.is_some() {
                gpu.reserved_bytes = info.reserved_bytes;
            }
            if let Some(uuid) = &info.uuid {
                gpu.vendor_uuid = Some(uuid.clone());
                if gpu.stable_id.as_deref().is_none_or(|stable_id| {
                    stable_id.starts_with("index:")
                        || stable_id.starts_with("cuda")
                        || stable_id.starts_with("vulkan")
                }) {
                    gpu.stable_id = Some(format!("uuid:{uuid}"));
                }
            }
            if let Some(pci_bdf) = &info.pci_bdf {
                gpu.pci_bdf = Some(pci_bdf.clone());
                if !super::super::is_placeholder_pci_bdf(pci_bdf) {
                    gpu.stable_id = Some(format!("pci:{pci_bdf}"));
                }
            }
        }
    }

    pub(crate) fn discover_gpu_facts() -> Vec<GpuFacts> {
        let mut infos = cuda_device_infos();
        merge_nvml_device_infos(&mut infos);

        infos
            .into_iter()
            .enumerate()
            .filter_map(|(index, info)| gpu_fact_from_nvidia_info(index, info))
            .collect()
    }

    fn match_nvidia_device<'a>(
        gpu: &GpuFacts,
        infos: &'a [NvidiaDeviceInfo],
    ) -> Option<&'a NvidiaDeviceInfo> {
        if !gpu.display_name.to_ascii_lowercase().contains("nvidia")
            && gpu.vendor_uuid.is_none()
            && !gpu
                .backend_device
                .as_deref()
                .is_some_and(|name| name.starts_with("CUDA") || name.starts_with("Vulkan"))
        {
            return None;
        }

        if let Some(pci_bdf) = gpu.pci_bdf.as_deref().and_then(normalize_pci_bdf) {
            if let Some(info) = infos
                .iter()
                .find(|info| info.pci_bdf.as_deref() == Some(pci_bdf.as_str()))
            {
                return Some(info);
            }
        }

        infos.get(gpu.index).or_else(|| {
            if infos.len() == 1 {
                infos.first()
            } else {
                None
            }
        })
    }

    fn cuda_device_infos() -> Vec<NvidiaDeviceInfo> {
        let Some(lib) = DlLibrary::open(b"libcuda.so.1\0") else {
            return Vec::new();
        };
        let Some(cu_init) = (unsafe { lib.symbol::<CuInit>(b"cuInit\0") }) else {
            return Vec::new();
        };
        let Some(cu_device_get_count) =
            (unsafe { lib.symbol::<CuDeviceGetCount>(b"cuDeviceGetCount\0") })
        else {
            return Vec::new();
        };
        let Some(cu_device_get) = (unsafe { lib.symbol::<CuDeviceGet>(b"cuDeviceGet\0") }) else {
            return Vec::new();
        };
        let cu_device_total_mem =
            unsafe { lib.symbol::<CuDeviceTotalMem>(b"cuDeviceTotalMem_v2\0") };
        let cu_device_get_name = unsafe { lib.symbol::<CuDeviceGetName>(b"cuDeviceGetName\0") };
        let cu_device_get_pci_bus_id =
            unsafe { lib.symbol::<CuDeviceGetPciBusId>(b"cuDeviceGetPCIBusId\0") };

        if unsafe { cu_init(0) } != CUDA_SUCCESS {
            return Vec::new();
        }

        let mut count = 0;
        if unsafe { cu_device_get_count(&mut count) } != CUDA_SUCCESS || count <= 0 {
            return Vec::new();
        }

        let mut infos = Vec::new();
        for index in 0..count {
            let mut device = 0;
            if unsafe { cu_device_get(&mut device, index) } != CUDA_SUCCESS {
                continue;
            }

            let mut info = NvidiaDeviceInfo::default();
            if let Some(device_name) = cu_device_get_name {
                let mut buf = [0 as c_char; 256];
                if unsafe { device_name(buf.as_mut_ptr(), buf.len() as c_int, device) }
                    == CUDA_SUCCESS
                {
                    info.name = unsafe { c_string(buf.as_ptr()) };
                }
            }
            if let Some(total_mem) = cu_device_total_mem {
                let mut total = 0usize;
                if unsafe { total_mem(&mut total, device) } == CUDA_SUCCESS {
                    info.total_bytes = Some(total as u64);
                }
            }
            if let Some(pci_bus_id) = cu_device_get_pci_bus_id {
                let mut buf = [0 as c_char; 32];
                if unsafe { pci_bus_id(buf.as_mut_ptr(), buf.len() as c_int, device) }
                    == CUDA_SUCCESS
                {
                    info.pci_bdf = unsafe { c_string(buf.as_ptr()) }
                        .as_deref()
                        .and_then(normalize_pci_bdf);
                }
            }
            infos.push(info);
        }

        infos
    }

    fn merge_nvml_device_infos(infos: &mut Vec<NvidiaDeviceInfo>) {
        let Some(lib) = DlLibrary::open(b"libnvidia-ml.so.1\0") else {
            return;
        };
        let Some(nvml_init) = (unsafe { lib.symbol::<NvmlInit>(b"nvmlInit_v2\0") }) else {
            return;
        };
        let Some(nvml_device_get_count) =
            (unsafe { lib.symbol::<NvmlDeviceGetCount>(b"nvmlDeviceGetCount_v2\0") })
        else {
            return;
        };
        let Some(nvml_device_get_handle_by_index) = (unsafe {
            lib.symbol::<NvmlDeviceGetHandleByIndex>(b"nvmlDeviceGetHandleByIndex_v2\0")
        }) else {
            return;
        };
        let nvml_shutdown = unsafe { lib.symbol::<NvmlShutdown>(b"nvmlShutdown\0") };
        let nvml_device_get_uuid =
            unsafe { lib.symbol::<NvmlDeviceGetUuid>(b"nvmlDeviceGetUUID\0") };
        let nvml_device_get_memory_info =
            unsafe { lib.symbol::<NvmlDeviceGetMemoryInfo>(b"nvmlDeviceGetMemoryInfo\0") };
        let nvml_device_get_memory_info_v2 =
            unsafe { lib.symbol::<NvmlDeviceGetMemoryInfoV2>(b"nvmlDeviceGetMemoryInfo_v2\0") };

        if unsafe { nvml_init() } != NVML_SUCCESS {
            return;
        }

        let mut count = 0;
        if unsafe { nvml_device_get_count(&mut count) } == NVML_SUCCESS {
            for index in 0..count {
                let mut device = std::ptr::null_mut();
                if unsafe { nvml_device_get_handle_by_index(index, &mut device) } != NVML_SUCCESS {
                    continue;
                }

                let mut info = infos
                    .get(index as usize)
                    .cloned()
                    .unwrap_or_else(NvidiaDeviceInfo::default);
                if let Some(get_uuid) = nvml_device_get_uuid {
                    let mut buf = [0 as c_char; 96];
                    if unsafe { get_uuid(device, buf.as_mut_ptr(), buf.len() as c_uint) }
                        == NVML_SUCCESS
                    {
                        info.uuid = unsafe { c_string(buf.as_ptr()) };
                    }
                }
                if let Some(get_memory_v2) = nvml_device_get_memory_info_v2 {
                    let mut memory = NvmlMemoryV2 {
                        version: (std::mem::size_of::<NvmlMemoryV2>() as c_uint) | (2 << 24),
                        ..NvmlMemoryV2::default()
                    };
                    if unsafe { get_memory_v2(device, &mut memory) } == NVML_SUCCESS {
                        info.total_bytes = Some(memory.total);
                        info.reserved_bytes = Some(round_up_to_mib(memory.reserved));
                    }
                } else if let Some(get_memory) = nvml_device_get_memory_info {
                    let mut memory = NvmlMemory::default();
                    if unsafe { get_memory(device, &mut memory) } == NVML_SUCCESS {
                        info.total_bytes = Some(memory.total);
                    }
                }

                if index as usize >= infos.len() {
                    infos.push(info);
                } else {
                    infos[index as usize] = info;
                }
            }
        }

        if let Some(shutdown) = nvml_shutdown {
            unsafe {
                shutdown();
            }
        }
    }

    unsafe fn c_string(ptr: *const c_char) -> Option<String> {
        if ptr.is_null() {
            return None;
        }
        let value = CStr::from_ptr(ptr).to_string_lossy().trim().to_string();
        if value.is_empty() {
            None
        } else {
            Some(value)
        }
    }

    fn normalize_pci_bdf(value: &str) -> Option<String> {
        let trimmed = value.trim();
        let (domain, rest) = trimmed.split_once(':')?;
        if domain.len() == 4 && rest.contains(':') && rest.contains('.') {
            Some(format!("0000{domain}:{rest}"))
        } else if domain.len() == 8 && rest.contains(':') && rest.contains('.') {
            Some(trimmed.to_string())
        } else {
            None
        }
    }

    fn round_up_to_mib(bytes: u64) -> u64 {
        const MIB: u64 = 1024 * 1024;
        bytes.div_ceil(MIB) * MIB
    }

    fn gpu_fact_from_nvidia_info(index: usize, info: NvidiaDeviceInfo) -> Option<GpuFacts> {
        let display_name = info.name?;
        let stable_id = info
            .pci_bdf
            .as_ref()
            .filter(|pci_bdf| !super::super::is_placeholder_pci_bdf(pci_bdf))
            .map(|pci_bdf| format!("pci:{pci_bdf}"));
        Some(GpuFacts {
            index,
            display_name,
            backend_device: Some(format!("CUDA{index}")),
            vram_bytes: info.total_bytes.unwrap_or(0),
            reserved_bytes: info.reserved_bytes,
            mem_bandwidth_gbps: None,
            compute_tflops_fp32: None,
            compute_tflops_fp16: None,
            unified_memory: false,
            stable_id,
            pci_bdf: info.pci_bdf,
            vendor_uuid: info.uuid,
            metal_registry_id: None,
            dxgi_luid: None,
            pnp_instance_id: None,
        })
    }
}

#[cfg(target_os = "linux")]
pub(super) use linux::{discover_gpu_facts, enrich_gpu_facts};

#[cfg(not(target_os = "linux"))]
pub(super) fn enrich_gpu_facts(_gpus: &mut [GpuFacts]) {}

#[cfg(not(target_os = "linux"))]
pub(super) fn discover_gpu_facts() -> Vec<GpuFacts> {
    Vec::new()
}
