//! Runtime security hardening for inference privacy.
//!
//! Implements OS-level protections that prevent the machine operator from
//! inspecting inference data at runtime. On macOS with SIP enabled, these
//! create a strong barrier against memory inspection:
//!
//!   - **PT_DENY_ATTACH**: Blocks all debugger attachment (lldb, dtrace).
//!     Even root cannot override while SIP is enabled.
//!   - **Core dump disable**: Prevents plaintext prompts from appearing
//!     in crash dumps.
//!   - **Environment scrubbing**: Removes DYLD_INSERT_LIBRARIES and
//!     similar injection vectors.
//!   - **SIP verification**: Confirms System Integrity Protection is on.
//!   - **RDMA check**: Verifies Thunderbolt 5 RDMA is disabled (RDMA
//!     can bypass all software protections at 80 Gb/s).
//!   - **Binary self-hash**: SHA-256 of the running binary for attestation.
//!
//! These protections are complementary. SIP ensures PT_DENY_ATTACH and
//! Hardened Runtime are enforced by the kernel. SIP cannot be disabled
//! at runtime — it requires rebooting into Recovery Mode, which kills
//! the process.

use sha2::{Digest, Sha256};
use std::io::Read;

/// Security posture of this node, reported in gossip so peers can
/// make trust decisions.
#[derive(Debug, Clone, Default)]
pub struct SecurityPosture {
    /// macOS System Integrity Protection is enabled.
    pub sip_enabled: bool,
    /// Thunderbolt 5 RDMA is disabled (or not supported).
    pub rdma_disabled: bool,
    /// PT_DENY_ATTACH succeeded — no debugger can attach.
    pub debugger_blocked: bool,
    /// RLIMIT_CORE set to zero — no core dumps.
    pub core_dumps_disabled: bool,
    /// Dangerous environment variables removed.
    pub env_scrubbed: bool,
    /// SHA-256 hash of the running binary.
    pub binary_hash: Option<String>,
}

impl SecurityPosture {
    /// True if all critical checks passed (suitable for verified meshes).
    pub fn is_fully_hardened(&self) -> bool {
        self.sip_enabled
            && self.rdma_disabled
            && self.debugger_blocked
            && self.core_dumps_disabled
            && self.env_scrubbed
            && self.binary_hash.is_some()
    }
}

/// Apply all runtime hardening checks and return the security posture.
///
/// Call this at process startup before joining a mesh or accepting
/// any inference work. Each check is independent — failures in one
/// don't prevent others from running.
///
/// If `require_sip` is true, returns Err when SIP is disabled.
/// If `require_rdma_off` is true, returns Err when RDMA is enabled.
pub fn harden_runtime(
    require_sip: bool,
    require_rdma_off: bool,
) -> Result<SecurityPosture, String> {
    let debugger_blocked = deny_debugger_attachment();
    let core_dumps_disabled = disable_core_dumps();
    scrub_dangerous_env();
    let sip_enabled = check_sip_enabled();
    let rdma_disabled = check_rdma_disabled();
    let binary_hash = self_binary_hash();

    if require_sip && !sip_enabled {
        return Err("SIP is disabled — cannot safely serve inference.\n\n\
             To enable SIP:\n\
             1. Shut down your Mac completely\n\
             2. Press and hold the power button until \"Loading startup options\" appears\n\
             3. Select Options → Continue\n\
             4. Utilities → Terminal → type: csrutil enable\n\
             5. Restart\n\n\
             Then retry."
            .to_string());
    }

    if require_rdma_off && !rdma_disabled {
        return Err("RDMA is enabled — remote memory access possible.\n\n\
             To disable RDMA:\n\
             Open System Settings → Sharing → disable Remote Direct Memory Access\n\n\
             Then retry."
            .to_string());
    }

    Ok(SecurityPosture {
        sip_enabled,
        rdma_disabled,
        debugger_blocked,
        core_dumps_disabled,
        env_scrubbed: true,
        binary_hash,
    })
}

/// Prevent debugger attachment using ptrace(PT_DENY_ATTACH).
///
/// Once called, the kernel denies all future ptrace requests against
/// this process. Combined with SIP + Hardened Runtime, this makes the
/// process memory unreadable by any other process.
fn deny_debugger_attachment() -> bool {
    #[cfg(target_os = "macos")]
    {
        const PT_DENY_ATTACH: libc::c_int = 31;
        let result =
            unsafe { libc::ptrace(PT_DENY_ATTACH, 0, std::ptr::null_mut::<libc::c_char>(), 0) };
        if result == 0 {
            tracing::info!("security: PT_DENY_ATTACH enabled — debugger attachment blocked");
            true
        } else {
            tracing::warn!(
                "security: PT_DENY_ATTACH failed ({})",
                std::io::Error::last_os_error()
            );
            false
        }
    }

    #[cfg(not(target_os = "macos"))]
    {
        tracing::debug!("security: PT_DENY_ATTACH not available on this platform");
        false
    }
}

/// Disable core dumps by setting RLIMIT_CORE to zero.
///
/// Core dumps can contain plaintext prompts, model weights, and private
/// keys. Setting the limit to zero prevents the kernel from writing core
/// files even on crash.
fn disable_core_dumps() -> bool {
    #[cfg(unix)]
    {
        let zero = libc::rlimit {
            rlim_cur: 0,
            rlim_max: 0,
        };
        let ret = unsafe { libc::setrlimit(libc::RLIMIT_CORE, &zero) };
        if ret == 0 {
            tracing::info!("security: core dumps disabled (RLIMIT_CORE = 0)");
            true
        } else {
            tracing::warn!(
                "security: failed to disable core dumps ({})",
                std::io::Error::last_os_error()
            );
            false
        }
    }

    #[cfg(not(unix))]
    {
        tracing::debug!("security: core dump disable not available on this platform");
        false
    }
}

/// Environment variables that could be used to inject code into the
/// process or its children.
const DANGEROUS_ENV_VARS: &[&str] = &[
    "DYLD_INSERT_LIBRARIES",
    "DYLD_LIBRARY_PATH",
    "DYLD_FRAMEWORK_PATH",
    "LD_PRELOAD",
    "LD_LIBRARY_PATH",
    "PYTHONPATH",
    "PYTHONSTARTUP",
    "PYTHONHOME",
];

/// Remove environment variables that could be used for code injection.
fn scrub_dangerous_env() {
    for var in DANGEROUS_ENV_VARS {
        if std::env::var_os(var).is_some() {
            tracing::warn!("security: scrubbing dangerous env var: {var}");
            // SAFETY: single-threaded at startup, before any threads spawned.
            unsafe {
                std::env::remove_var(var);
            }
        }
    }
    tracing::info!("security: environment scrubbed");
}

/// Check if System Integrity Protection is enabled.
///
/// SIP is the foundation of the macOS security model. With SIP enabled:
///   - Hardened Runtime protections are kernel-enforced
///   - Unsigned kernel extensions cannot load
///   - Root cannot modify /System or attach to protected processes
///   - SIP cannot be disabled at runtime (requires Recovery Mode reboot)
pub fn check_sip_enabled() -> bool {
    #[cfg(target_os = "macos")]
    {
        match std::process::Command::new("/usr/bin/csrutil")
            .arg("status")
            .output()
        {
            Ok(output) => {
                let stdout = String::from_utf8_lossy(&output.stdout);
                let enabled = stdout.contains("enabled");
                if enabled {
                    tracing::info!("security: SIP is enabled");
                } else {
                    tracing::error!("security: SIP is DISABLED");
                }
                enabled
            }
            Err(e) => {
                tracing::error!("security: failed to check SIP: {e}");
                false
            }
        }
    }

    #[cfg(not(target_os = "macos"))]
    {
        tracing::debug!("security: SIP check not applicable on this platform");
        false
    }
}

/// Check if RDMA (Remote Direct Memory Access) over Thunderbolt 5 is disabled.
///
/// RDMA allows another Mac to directly read this process's memory at
/// 80 Gb/s, bypassing PT_DENY_ATTACH, Hardened Runtime, and SIP entirely.
/// RDMA is disabled by default on macOS; enabling requires Recovery OS.
///
/// Returns true if RDMA is disabled or the hardware doesn't support it.
pub fn check_rdma_disabled() -> bool {
    #[cfg(target_os = "macos")]
    {
        match std::process::Command::new("/usr/bin/rdma_ctl")
            .arg("status")
            .output()
        {
            Ok(output) => {
                let stdout = String::from_utf8_lossy(&output.stdout);
                let disabled = stdout.trim() == "disabled";
                if disabled {
                    tracing::debug!("security: RDMA is disabled");
                } else {
                    tracing::warn!("security: RDMA is ENABLED");
                }
                disabled
            }
            Err(_) => {
                // rdma_ctl not found → RDMA not supported on this hardware
                tracing::debug!("security: rdma_ctl not available, assuming RDMA not supported");
                true
            }
        }
    }

    #[cfg(not(target_os = "macos"))]
    {
        tracing::debug!("security: RDMA check not applicable on this platform");
        true
    }
}

/// Compute the SHA-256 hash of the currently running binary.
///
/// Included in attestation blobs so peers can verify the node is running
/// expected (blessed) software.
pub fn self_binary_hash() -> Option<String> {
    let exe_path = std::env::current_exe().ok()?;
    let hash = hash_file(&exe_path)?;
    tracing::info!(
        "security: binary self-hash ({}): {}…",
        exe_path.display(),
        &hash[..16]
    );
    Some(hash)
}

/// Compute the SHA-256 hash of a file, reading in 64 KB chunks.
pub fn hash_file(path: &std::path::Path) -> Option<String> {
    let mut file = std::fs::File::open(path).ok()?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 65536];
    loop {
        let n = file.read(&mut buf).ok()?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Some(format!("{:x}", hasher.finalize()))
}

/// Zero a byte buffer using volatile writes to prevent dead-store
/// elimination by the compiler.
pub fn secure_zero(buf: &mut [u8]) {
    for byte in buf.iter_mut() {
        unsafe {
            std::ptr::write_volatile(byte, 0);
        }
    }
    std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_security_posture_default_not_hardened() {
        let posture = SecurityPosture::default();
        assert!(!posture.is_fully_hardened());
    }

    #[test]
    fn test_security_posture_fully_hardened() {
        let posture = SecurityPosture {
            sip_enabled: true,
            rdma_disabled: true,
            debugger_blocked: true,
            core_dumps_disabled: true,
            env_scrubbed: true,
            binary_hash: Some("abc123".to_string()),
        };
        assert!(posture.is_fully_hardened());
    }

    #[test]
    fn test_security_posture_missing_hash_not_hardened() {
        let posture = SecurityPosture {
            sip_enabled: true,
            rdma_disabled: true,
            debugger_blocked: true,
            core_dumps_disabled: true,
            env_scrubbed: true,
            binary_hash: None,
        };
        assert!(!posture.is_fully_hardened());
    }

    #[test]
    fn test_self_binary_hash_returns_some() {
        // The test binary itself should be hashable.
        let hash = self_binary_hash();
        assert!(hash.is_some());
        let h = hash.unwrap();
        assert_eq!(h.len(), 64); // SHA-256 hex = 64 chars
    }

    #[test]
    fn test_hash_file_nonexistent() {
        let result = hash_file(std::path::Path::new("/nonexistent/path"));
        assert!(result.is_none());
    }

    #[test]
    fn test_secure_zero() {
        let mut buf = vec![0xFFu8; 64];
        secure_zero(&mut buf);
        assert!(buf.iter().all(|&b| b == 0));
    }

    #[test]
    fn test_dangerous_env_vars_list() {
        // Ensure the list is non-empty and contains key entries.
        assert!(DANGEROUS_ENV_VARS.contains(&"DYLD_INSERT_LIBRARIES"));
        assert!(DANGEROUS_ENV_VARS.contains(&"LD_PRELOAD"));
    }
}
