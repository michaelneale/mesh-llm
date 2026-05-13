use std::ffi::{c_char, CStr};
use std::ptr;

use anyhow::{anyhow, Result};
use skippy_ffi::{BackendDevice as RawBackendDevice, BackendDeviceType as RawBackendDeviceType};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendDeviceType {
    Cpu,
    Gpu,
    IntegratedGpu,
    Accelerator,
    Meta,
}

impl From<RawBackendDeviceType> for BackendDeviceType {
    fn from(value: RawBackendDeviceType) -> Self {
        match value {
            RawBackendDeviceType::Cpu => Self::Cpu,
            RawBackendDeviceType::Gpu => Self::Gpu,
            RawBackendDeviceType::IGpu => Self::IntegratedGpu,
            RawBackendDeviceType::Accel => Self::Accelerator,
            RawBackendDeviceType::Meta => Self::Meta,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendDevice {
    pub name: String,
    pub description: Option<String>,
    pub device_id: Option<String>,
    pub memory_free: u64,
    pub memory_total: u64,
    pub device_type: BackendDeviceType,
    pub caps: u64,
}

pub fn backend_devices() -> Result<Vec<BackendDevice>> {
    let mut error = ptr::null_mut();
    let mut count = 0usize;
    let status = unsafe { skippy_ffi::skippy_backend_device_count(&mut count, &mut error) };
    super::ensure_ok(status, error)?;

    let mut devices = Vec::with_capacity(count);
    for index in 0..count {
        let mut raw = RawBackendDevice {
            version: 0,
            name: ptr::null(),
            description: ptr::null(),
            device_id: ptr::null(),
            memory_free: 0,
            memory_total: 0,
            device_type: RawBackendDeviceType::Cpu,
            caps: 0,
        };
        let mut error = ptr::null_mut();
        let status = unsafe { skippy_ffi::skippy_backend_device_at(index, &mut raw, &mut error) };
        super::ensure_ok(status, error)?;
        devices.push(backend_device_from_raw(raw)?);
    }

    Ok(devices)
}

fn backend_device_from_raw(raw: RawBackendDevice) -> Result<BackendDevice> {
    Ok(BackendDevice {
        name: c_string_required(raw.name, "backend device name")?,
        description: c_string_optional(raw.description)?,
        device_id: c_string_optional(raw.device_id)?,
        memory_free: raw.memory_free,
        memory_total: raw.memory_total,
        device_type: raw.device_type.into(),
        caps: raw.caps,
    })
}

fn c_string_required(ptr: *const c_char, field: &str) -> Result<String> {
    if ptr.is_null() {
        return Err(anyhow!("{field} is null"));
    }
    Ok(unsafe { CStr::from_ptr(ptr) }
        .to_string_lossy()
        .into_owned())
}

fn c_string_optional(ptr: *const c_char) -> Result<Option<String>> {
    if ptr.is_null() {
        return Ok(None);
    }
    let value = unsafe { CStr::from_ptr(ptr) }
        .to_string_lossy()
        .into_owned();
    Ok((!value.is_empty()).then_some(value))
}
