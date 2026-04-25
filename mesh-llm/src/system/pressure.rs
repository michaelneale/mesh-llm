//! System memory-pressure detection for yield-on-pressure mode.
//!
//! Answers one question on a tick: "Is this machine under memory pressure?"
//!
//! Design notes:
//! - macOS reads `sysctl vm.memory_pressure` which returns the kernel's
//!   composite memory-pressure level. On Apple Silicon unified memory this
//!   covers both CPU and GPU memory contention. No privileges required.
//!   Importantly, llama-server allocates its memory up-front at load time, so
//!   the pressure level already accounts for mesh-llm's own footprint — if
//!   pressure rises, it's because something *else* is competing.
//! - All other platforms return `Unknown`. This is a fail-open signal: the
//!   yield controller treats it as "keep serving" so a broken probe never
//!   takes a node offline silently.
//!
//! All platform implementations are in-crate so we stay in a single module
//! and avoid leaking platform crates into the wider build.

/// System memory-pressure level consumed by the yield controller.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Pressure {
    /// Memory is plentiful — no reason to yield.
    Normal,
    /// The system is under memory pressure (warn or critical). Other
    /// workloads are competing for the memory our model is sitting on.
    Pressured,
    /// Probe failed or platform is not supported. Treated as "keep serving".
    Unknown,
}

/// A synchronous pressure probe. Implementations must be cheap — we call
/// `sample` on a timer from a normal tokio task.
pub trait PressureProbe: Send + Sync + 'static {
    fn sample(&self) -> Pressure;
}

/// Build the default probe for this platform. Returns a probe that always
/// reports `Unknown` on platforms without a real implementation.
pub fn default_probe() -> Box<dyn PressureProbe> {
    #[cfg(target_os = "macos")]
    {
        return Box::new(macos::MacMemoryPressureProbe);
    }
    #[cfg(not(target_os = "macos"))]
    {
        Box::new(UnknownProbe)
    }
}

/// Probe that always reports `Unknown`. Used on unsupported platforms and in
/// tests.
#[allow(dead_code)]
pub struct UnknownProbe;

impl PressureProbe for UnknownProbe {
    fn sample(&self) -> Pressure {
        Pressure::Unknown
    }
}

#[cfg(target_os = "macos")]
mod macos {
    use super::{Pressure, PressureProbe};

    /// Reads `vm.memory_pressure` via sysctl(3).
    ///
    /// Kernel values (from osfmk/vm/vm_pageout.h):
    ///   0 = kVMPressureNormal
    ///   1 = kVMPressureWarning
    ///   2 = kVMPressureCritical  (sometimes reported as 4; both > 0)
    ///
    /// Any non-zero value means the kernel considers memory scarce.
    pub struct MacMemoryPressureProbe;

    impl PressureProbe for MacMemoryPressureProbe {
        fn sample(&self) -> Pressure {
            match read_memory_pressure() {
                Some(0) => Pressure::Normal,
                Some(_) => Pressure::Pressured,
                None => Pressure::Unknown,
            }
        }
    }

    fn read_memory_pressure() -> Option<i32> {
        // sysctl name: "vm.memory_pressure"
        let name = c"vm.memory_pressure";
        let mut value: i32 = 0;
        let mut size = std::mem::size_of::<i32>();
        // SAFETY: `sysctlbyname` reads a known integer sysctl. `value` is a
        // valid i32 pointer with correct `size`. The name is a static
        // null-terminated string.
        let ret = unsafe {
            libc::sysctlbyname(
                name.as_ptr(),
                &mut value as *mut i32 as *mut libc::c_void,
                &mut size,
                std::ptr::null_mut(),
                0,
            )
        };
        if ret == 0 {
            Some(value)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_probe_always_returns_unknown() {
        let probe = UnknownProbe;
        assert_eq!(probe.sample(), Pressure::Unknown);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn mac_probe_returns_valid_state() {
        let probe = macos::MacMemoryPressureProbe;
        let s = probe.sample();
        // Under normal CI conditions we expect Normal, but Pressured is also
        // valid if the machine is loaded. Unknown would indicate sysctl broke.
        assert!(
            matches!(s, Pressure::Normal | Pressure::Pressured),
            "expected Normal or Pressured on macOS, got {s:?}"
        );
    }

    #[test]
    fn default_probe_does_not_panic() {
        let probe = default_probe();
        let s = probe.sample();
        assert!(matches!(
            s,
            Pressure::Normal | Pressure::Pressured | Pressure::Unknown
        ));
    }
}
