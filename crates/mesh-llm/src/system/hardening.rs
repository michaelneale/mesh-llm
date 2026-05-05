//! Runtime security hardening for inference privacy.
//!
//! Best-effort OS-level protections that make it harder for the machine
//! operator to inspect inference data at runtime. On macOS with SIP enabled,
//! these create a strong barrier against memory inspection.

use sha2::{Digest, Sha256};
use std::io::Read;

/// Security posture of this node, reported in gossip.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct SecurityPosture {
    pub sip_enabled: bool,
    pub rdma_disabled: bool,
    pub debugger_blocked: bool,
    pub core_dumps_disabled: bool,
    pub binary_hash: Option<String>,
}

impl SecurityPosture {
    /// True if all critical checks passed. Used by attestation verification
    /// and will be exposed in /api/status.
    #[allow(dead_code)]
    pub fn is_hardened(&self) -> bool {
        self.sip_enabled
            && self.rdma_disabled
            && self.debugger_blocked
            && self.core_dumps_disabled
            && self.binary_hash.is_some()
    }
}

/// Scrub environment variables that could be used for code injection (e.g.
/// `DYLD_INSERT_LIBRARIES`, `LD_PRELOAD`).
///
/// **Must be called before any worker threads are spawned** — `std::env::remove_var`
/// is `unsafe` because it mutates global process state and is UB when another
/// thread may be reading the environment concurrently.
///
/// In practice this means calling it from `main()` before `#[tokio::main]`
/// enters the async runtime.
pub fn scrub_env_pre_thread() {
    scrub_dangerous_env();
}

/// Apply all runtime hardening and return the security posture.
///
/// Env scrubbing is intentionally **not** included here — it must happen
/// before threads exist (see [`scrub_env_pre_thread`]).
pub fn harden_runtime() -> SecurityPosture {
    let debugger_blocked = deny_debugger_attachment();
    let core_dumps_disabled = disable_core_dumps();
    let sip_enabled = check_sip_enabled();
    let rdma_disabled = check_rdma_disabled();
    let binary_hash = self_binary_hash();

    SecurityPosture {
        sip_enabled,
        rdma_disabled,
        debugger_blocked,
        core_dumps_disabled,
        binary_hash,
    }
}

fn deny_debugger_attachment() -> bool {
    #[cfg(target_os = "macos")]
    {
        const PT_DENY_ATTACH: libc::c_int = 31;
        let result =
            unsafe { libc::ptrace(PT_DENY_ATTACH, 0, std::ptr::null_mut::<libc::c_char>(), 0) };
        if result == 0 {
            tracing::info!("security: PT_DENY_ATTACH enabled");
            true
        } else {
            tracing::warn!("security: PT_DENY_ATTACH failed");
            false
        }
    }

    #[cfg(not(target_os = "macos"))]
    {
        false
    }
}

fn disable_core_dumps() -> bool {
    #[cfg(unix)]
    {
        let zero = libc::rlimit {
            rlim_cur: 0,
            rlim_max: 0,
        };
        let ret = unsafe { libc::setrlimit(libc::RLIMIT_CORE, &zero) };
        if ret == 0 {
            tracing::info!("security: core dumps disabled");
            true
        } else {
            tracing::warn!("security: failed to disable core dumps");
            false
        }
    }

    #[cfg(not(unix))]
    {
        false
    }
}

const DANGEROUS_ENV_VARS: &[&str] = &[
    "DYLD_INSERT_LIBRARIES",
    "DYLD_LIBRARY_PATH",
    "DYLD_FRAMEWORK_PATH",
    "LD_PRELOAD",
    "LD_LIBRARY_PATH",
];

fn scrub_dangerous_env() {
    for var in DANGEROUS_ENV_VARS {
        if std::env::var_os(var).is_some() {
            // Note: tracing may not be initialized yet (called pre-tokio),
            // so this warn may go to stderr via the default subscriber.
            tracing::warn!("security: scrubbing {var}");
            // SAFETY: called from main() before #[tokio::main] spawns threads.
            // The caller (scrub_env_pre_thread / main.rs) guarantees single-threaded context.
            unsafe { std::env::remove_var(var) };
        }
    }
}

pub fn check_sip_enabled() -> bool {
    #[cfg(target_os = "macos")]
    {
        std::process::Command::new("/usr/bin/csrutil")
            .arg("status")
            .output()
            .map(|o| String::from_utf8_lossy(&o.stdout).contains("enabled"))
            .unwrap_or(false)
    }

    #[cfg(not(target_os = "macos"))]
    {
        false
    }
}

pub fn check_rdma_disabled() -> bool {
    #[cfg(target_os = "macos")]
    {
        match std::process::Command::new("/usr/bin/rdma_ctl")
            .arg("status")
            .output()
        {
            Ok(output) => {
                let stdout = String::from_utf8_lossy(&output.stdout);
                stdout.trim() == "disabled"
            }
            Err(_) => true, // rdma_ctl not found → RDMA not supported
        }
    }

    #[cfg(not(target_os = "macos"))]
    {
        true
    }
}

pub fn self_binary_hash() -> Option<String> {
    let exe_path = std::env::current_exe().ok()?;
    let mut file = std::fs::File::open(exe_path).ok()?;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_not_hardened() {
        assert!(!SecurityPosture::default().is_hardened());
    }

    #[test]
    fn binary_hash_works() {
        let hash = self_binary_hash();
        assert!(hash.is_some());
        assert_eq!(hash.unwrap().len(), 64);
    }
}
