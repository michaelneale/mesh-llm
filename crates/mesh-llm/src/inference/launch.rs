//! Shared backend-adjacent helpers that are still needed outside model serving.

use anyhow::Result;
use clap::ValueEnum;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum BinaryFlavor {
    Cpu,
    Cuda,
    Rocm,
    Vulkan,
    Metal,
}

impl BinaryFlavor {
    pub const ALL: [BinaryFlavor; 5] = [
        BinaryFlavor::Cpu,
        BinaryFlavor::Cuda,
        BinaryFlavor::Rocm,
        BinaryFlavor::Vulkan,
        BinaryFlavor::Metal,
    ];

    pub fn suffix(self) -> &'static str {
        match self {
            BinaryFlavor::Cpu => "cpu",
            BinaryFlavor::Cuda => "cuda",
            BinaryFlavor::Rocm => "rocm",
            BinaryFlavor::Vulkan => "vulkan",
            BinaryFlavor::Metal => "metal",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct BinaryBackendDeviceProbe {
    pub(crate) path: PathBuf,
    pub(crate) flavor: Option<BinaryFlavor>,
    pub(crate) available_devices: Vec<String>,
}

static RUNTIME_SHUTTING_DOWN: AtomicBool = AtomicBool::new(false);

pub fn mark_runtime_shutting_down() {
    RUNTIME_SHUTTING_DOWN.store(true, Ordering::SeqCst);
}

pub fn clear_runtime_shutting_down() {
    RUNTIME_SHUTTING_DOWN.store(false, Ordering::SeqCst);
}

pub(crate) fn platform_bin_name(name: &str) -> String {
    #[cfg(windows)]
    {
        if Path::new(name)
            .extension()
            .and_then(|ext| ext.to_str())
            .is_some_and(|ext| ext.eq_ignore_ascii_case("exe"))
        {
            name.to_string()
        } else {
            format!("{name}.exe")
        }
    }

    #[cfg(not(windows))]
    {
        name.to_string()
    }
}

pub(crate) fn backend_device_for_flavor(
    index: usize,
    binary_flavor: BinaryFlavor,
) -> Option<String> {
    match binary_flavor {
        BinaryFlavor::Cpu => None,
        BinaryFlavor::Cuda => Some(format!("CUDA{index}")),
        BinaryFlavor::Rocm => Some(format!("ROCm{index}")),
        BinaryFlavor::Vulkan => Some(format!("Vulkan{index}")),
        BinaryFlavor::Metal => Some(format!("MTL{index}")),
    }
}

pub(crate) fn resolve_requested_device_from_available(
    available: &[String],
    binary: &Path,
    requested: &str,
) -> Result<String> {
    if !available.is_empty() {
        if available.iter().any(|candidate| candidate == requested) {
            return Ok(requested.to_string());
        }

        let is_amd_requested = requested.starts_with("ROCm") || requested.starts_with("HIP");
        if is_amd_requested {
            let alt_device = if requested.starts_with("ROCm") {
                requested.replace("ROCm", "HIP")
            } else {
                requested.replace("HIP", "ROCm")
            };
            if available.iter().any(|candidate| candidate == &alt_device) {
                return Ok(alt_device);
            }
        }

        anyhow::bail!(
            "requested device {requested} is not supported by {}. Available devices: {}",
            binary.display(),
            available.join(", ")
        );
    }

    Ok(requested.to_string())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ProcessSignal {
    Terminate,
    Kill,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SignalOutcome {
    Sent,
    AlreadyDead,
    Skipped,
    Failed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TerminationOutcome {
    NotRunning,
    Graceful,
    Killed,
    Failed,
}

impl TerminationOutcome {
    pub(crate) fn is_success(self) -> bool {
        !matches!(self, TerminationOutcome::Failed)
    }
}

pub fn is_safe_kill_target(pid: u32) -> bool {
    pid > 1 && pid <= i32::MAX as u32
}

pub(crate) fn terminate_process_blocking(
    pid: u32,
    expected_comm: &str,
    expected_start_time: Option<i64>,
) -> TerminationOutcome {
    match send_signal_if_matches(
        pid,
        expected_comm,
        expected_start_time,
        ProcessSignal::Terminate,
    ) {
        SignalOutcome::Sent => {}
        SignalOutcome::AlreadyDead => return TerminationOutcome::NotRunning,
        SignalOutcome::Skipped | SignalOutcome::Failed => return TerminationOutcome::Failed,
    }

    for _ in 0..20 {
        std::thread::sleep(Duration::from_millis(250));
        if crate::runtime::instance::validate::process_liveness(pid)
            == crate::runtime::instance::validate::Liveness::Dead
        {
            return TerminationOutcome::Graceful;
        }
    }

    match send_signal_if_matches(pid, expected_comm, expected_start_time, ProcessSignal::Kill) {
        SignalOutcome::Sent => TerminationOutcome::Killed,
        SignalOutcome::AlreadyDead => TerminationOutcome::Graceful,
        SignalOutcome::Skipped | SignalOutcome::Failed => TerminationOutcome::Failed,
    }
}

pub async fn terminate_process(
    pid: u32,
    expected_comm: &str,
    expected_start_time: Option<i64>,
) -> bool {
    !matches!(
        send_signal_if_matches(
            pid,
            expected_comm,
            expected_start_time,
            ProcessSignal::Terminate
        ),
        SignalOutcome::Failed | SignalOutcome::Skipped
    )
}

pub async fn force_kill_process(
    pid: u32,
    expected_comm: &str,
    expected_start_time: Option<i64>,
) -> bool {
    !matches!(
        send_signal_if_matches(pid, expected_comm, expected_start_time, ProcessSignal::Kill),
        SignalOutcome::Failed | SignalOutcome::Skipped
    )
}

pub async fn wait_for_exit(pid: u32, timeout_ms: u64) -> bool {
    if crate::runtime::instance::validate::process_liveness(pid)
        == crate::runtime::instance::validate::Liveness::Dead
    {
        return true;
    }
    let steps = timeout_ms.div_ceil(100);
    for _ in 0..steps {
        tokio::time::sleep(Duration::from_millis(100)).await;
        if crate::runtime::instance::validate::process_liveness(pid)
            == crate::runtime::instance::validate::Liveness::Dead
        {
            return true;
        }
    }
    false
}

fn send_signal_if_matches(
    pid: u32,
    expected_comm: &str,
    expected_start_time: Option<i64>,
    signal: ProcessSignal,
) -> SignalOutcome {
    if !is_safe_kill_target(pid) {
        tracing::error!("BUG: attempted to signal unsafe pid {pid} - refusing");
        return SignalOutcome::Failed;
    }

    #[cfg(not(windows))]
    {
        let matches = if let Some(expected_t) = expected_start_time {
            crate::runtime::instance::validate::validate_pid_matches(pid, expected_comm, expected_t)
        } else {
            crate::runtime::instance::validate::process_name_matches(pid, expected_comm)
        };
        if !matches {
            if crate::runtime::instance::validate::process_liveness(pid)
                == crate::runtime::instance::validate::Liveness::Dead
            {
                return SignalOutcome::AlreadyDead;
            }
            tracing::warn!("pid {pid} no longer matches {expected_comm}, skipping signal");
            return SignalOutcome::Skipped;
        }
    }

    #[cfg(windows)]
    {
        let _ = (expected_comm, expected_start_time);
    }

    #[cfg(unix)]
    unsafe {
        let ret = libc::kill(
            pid as libc::pid_t,
            match signal {
                ProcessSignal::Terminate => libc::SIGTERM,
                ProcessSignal::Kill => libc::SIGKILL,
            },
        );
        if ret == 0 {
            return SignalOutcome::Sent;
        }

        let err = std::io::Error::last_os_error();
        if err.raw_os_error() == Some(libc::ESRCH) {
            return SignalOutcome::AlreadyDead;
        }

        tracing::warn!(pid, error = %err, ?signal, "failed to signal process");
        SignalOutcome::Failed
    }

    #[cfg(windows)]
    {
        let pid_str = pid.to_string();
        let mut command = std::process::Command::new("taskkill");
        command.args(["/PID", &pid_str, "/T"]);
        if signal == ProcessSignal::Kill {
            command.arg("/F");
        }
        match command
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
        {
            Ok(status) if status.success() => SignalOutcome::Sent,
            Ok(status) => {
                tracing::warn!(pid, exit_code = status.code(), ?signal, "taskkill failed");
                SignalOutcome::Failed
            }
            Err(err) => {
                tracing::warn!(pid, error = %err, ?signal, "failed to run taskkill");
                SignalOutcome::Failed
            }
        }
    }
}
