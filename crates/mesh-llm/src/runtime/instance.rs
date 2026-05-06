//! Per-instance runtime directory management.
//!
//! Each non-client mesh-llm invocation acquires an `InstanceRuntime` under
//! `~/.mesh-llm/runtime/{pid}/` (overridable via env vars). The directory
//! holds an advisory `flock(2)` lock for the instance's lifetime and an
//! `owner.json` record for local status and `mesh-llm stop`.
//!
//! # Runtime directory layout
//!
//! **ALLOWED** under `runtime_dir/`:
//! - `lock` — `flock(2)` advisory lock file held by the owning mesh-llm
//! - `owner.json` — metadata about the owning instance (pid, version, api_port, started_at)
//! - `logs/` — process-local runtime logs, including embedded skippy/llama.cpp native logs
//!
//! **FORBIDDEN** under `runtime_dir/`:
//! - Application state, configuration, or catalog caches (live elsewhere under `~/.mesh-llm/`)
//! - Unix domain sockets (out of scope — use the API port)
//! - Downloaded model files (live under `~/.mesh-llm/models/`)
//! - Any new file type not explicitly listed above — update this list first
//!
//! # Runtime root resolution
//!
//! The root directory (containing per-instance subdirectories) is resolved via
//! this precedence:
//!
//! 1. `MESH_LLM_RUNTIME_ROOT` environment variable (highest; used by tests)
//! 2. `$XDG_RUNTIME_DIR/mesh-llm/runtime` (systemd services, rootless containers)
//! 3. Platform home directory (`$HOME` on Unix, Windows profile directory on Windows)
//! 4. Fails fast with a clear error if none of the above are set
//!
//! # Liveness detection
//!
//! Primary mechanism: `libc::flock(LOCK_EX | LOCK_NB)` on the `lock` file.
//! Released automatically by the kernel when the owning fd closes (including
//! on `SIGKILL`). Race-free and survives all abnormal terminations.
//!
//! Secondary (PID validation before stopping an instance):
//! - `/proc/{pid}/comm` on Linux (no shell spawn)
//! - `ps -p {pid} -o comm=` on macOS
//! - `start_time` tolerance ±2 seconds
//!
//! # Known limitations
//!
//! - **NFS-mounted `$HOME`**: advisory `flock` is unreliable on NFS. Override
//!   `MESH_LLM_RUNTIME_ROOT` to a local path in NFS environments.
//! - **Symlinked `~/.mesh-llm`**: two mesh-llm instances started via different
//!   symlink paths to the same physical directory will still see each other
//!   correctly via `flock`, but may appear as "different" dirs when listed.
//! - **Windows**: `flock` is a no-op. Runtime dirs are still created and
//!   process liveness falls back to best-effort PID checks.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
#[cfg(unix)]
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};

/// Write UTF-8 text atomically to `path` using a sibling `*.tmp` file.
///
/// Writes to `{path}.tmp`, calls `sync_all()`, then renames to `path`.
/// If writing or renaming fails, removes the tmp file before returning the error.
pub fn write_text_file_atomic(path: &Path, contents: &str) -> Result<()> {
    let tmp_path = tmp_path_for(path);

    let write_result = (|| -> Result<()> {
        let mut opts = fs::OpenOptions::new();
        opts.write(true).create(true).truncate(true);
        #[cfg(unix)]
        opts.mode(0o600);

        let mut file = opts
            .open(&tmp_path)
            .with_context(|| format!("failed to create tmp file: {}", tmp_path.display()))?;

        use std::io::Write;
        file.write_all(contents.as_bytes())
            .with_context(|| format!("failed to write tmp file: {}", tmp_path.display()))?;
        file.sync_all()
            .with_context(|| format!("failed to sync tmp file: {}", tmp_path.display()))?;

        fs::rename(&tmp_path, path).with_context(|| {
            format!(
                "failed to rename tmp file from {} to {}",
                tmp_path.display(),
                path.display()
            )
        })?;

        Ok(())
    })();

    if write_result.is_err() {
        let _ = fs::remove_file(&tmp_path);
    }

    write_result
}

fn tmp_path_for(path: &Path) -> PathBuf {
    let extension = path
        .extension()
        .map(|ext| format!("{}.tmp", ext.to_string_lossy()))
        .unwrap_or_else(|| "tmp".to_string());
    path.with_extension(extension)
}

/// Resolve the runtime root directory for this mesh-llm installation.
///
/// Precedence:
/// 1. `MESH_LLM_RUNTIME_ROOT` environment variable (test override / custom deployment)
/// 2. `$XDG_RUNTIME_DIR/mesh-llm/runtime`
/// 3. The platform home directory from [`dirs::home_dir`]
/// 4. [`anyhow::bail!`] - at least one of the above must be set
pub fn runtime_root() -> Result<PathBuf> {
    runtime_root_with_home(dirs::home_dir())
}

fn runtime_root_with_home(home: Option<PathBuf>) -> Result<PathBuf> {
    // 1. Explicit override — always wins (also used by tests to avoid touching ~)
    if let Ok(root) = std::env::var("MESH_LLM_RUNTIME_ROOT") {
        return Ok(PathBuf::from(root));
    }

    // 2. XDG_RUNTIME_DIR (standard on modern Linux)
    if let Ok(xdg) = std::env::var("XDG_RUNTIME_DIR") {
        return Ok(PathBuf::from(xdg).join("mesh-llm").join("runtime"));
    }

    // 3. Platform home directory. On Windows this can be available even when
    // HOME is unset in the launching shell.
    if let Some(home) = home {
        return Ok(home.join(".mesh-llm").join("runtime"));
    }

    // 4. Nothing usable - fail fast with a clear message.
    anyhow::bail!(
        "mesh-llm requires a home directory, XDG_RUNTIME_DIR, or MESH_LLM_RUNTIME_ROOT to be set"
    )
}

/// A scoped runtime directory for a single mesh-llm process instance.
///
/// Holds an exclusive `flock(2)` advisory lock on `{dir}/lock` for the duration
/// of the process lifetime. The lock is released automatically when this struct
/// is dropped — the `File` field's `Drop` closes the fd, and the kernel then
/// releases the associated flock.
///
/// Construct via [`InstanceRuntime::acquire`].
#[derive(Debug)]
pub struct InstanceRuntime {
    dir: PathBuf,
    pid: u32,
    _lock_file: File,
}

impl InstanceRuntime {
    /// Acquire a scoped runtime directory for `pid`.
    ///
    /// Creates the following directories (idempotent):
    /// - `{root}/{pid}/`
    ///
    /// Then opens `{root}/{pid}/lock` and acquires a **non-blocking exclusive
    /// flock**. Returns `Err` if the lock cannot be obtained (i.e. another live
    /// process already holds it).
    ///
    /// # Platform notes
    ///
    /// On non-Unix platforms the directories are created and the lock file is
    /// opened, but no flock is attempted (best-effort degraded mode).
    pub fn acquire(pid: u32) -> Result<Self> {
        let root = runtime_root()?;
        fs::create_dir_all(&root).context("failed to create runtime root")?;

        let dir = root.join(pid.to_string());
        fs::create_dir_all(&dir).context("failed to create runtime directory")?;

        // On Unix, harden permissions on the runtime directories so instance
        // metadata is only readable by the owning user.
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let private = std::fs::Permissions::from_mode(0o700);
            for d in [&root, &dir] {
                // Best-effort: log but don't fail if we can't set permissions
                // (e.g. on a read-only or network filesystem).
                if let Err(e) = std::fs::set_permissions(d, private.clone()) {
                    tracing::debug!(
                        path = %d.display(),
                        error = %e,
                        "could not set restrictive permissions on runtime directory"
                    );
                }
            }
        }

        let lock_path = dir.join("lock");
        // The lock file is opened only to hold a flock — we never write to it.
        // `truncate(false)` is the safe choice: existing locks must not be wiped.
        let lock_file = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&lock_path)
            .with_context(|| format!("failed to open lock file: {}", lock_path.display()))?;

        #[cfg(unix)]
        {
            use std::os::unix::io::AsRawFd;

            let fd = lock_file.as_raw_fd();
            // SAFETY: flock is safe to call with a valid fd
            let ret = unsafe { libc::flock(fd, libc::LOCK_EX | libc::LOCK_NB) };
            if ret != 0 {
                let err = std::io::Error::last_os_error();
                if err.raw_os_error() == Some(libc::EWOULDBLOCK) {
                    anyhow::bail!(
                        "runtime directory for pid {pid} is already locked \
                         (another live process owns this slot)"
                    );
                }
                return Err(anyhow::Error::from(err)).context("flock failed on runtime lock file");
            }
        }

        Ok(Self {
            dir,
            pid,
            _lock_file: lock_file,
        })
    }

    /// Returns the runtime directory path (`{root}/{pid}/`).
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// The PID this runtime slot was acquired for.
    #[allow(dead_code)]
    pub fn pid(&self) -> u32 {
        self.pid
    }
}

/// Probe whether the flock at `lock_path` is currently held by a live process.
///
/// Opens the file and attempts a non-blocking exclusive flock:
/// - Returns `true` if the lock is held (`EWOULDBLOCK`) — the slot is live.
/// - Returns `false` if the lock was acquired — no live holder; probe lock is
///   released immediately before returning.
/// - Returns `false` on any other error (file missing, permission denied, etc.)
///   to treat unknown states as "not locked" (callers must validate independently).
///
/// Only Unix currently supports probing the runtime flock.
#[cfg(all(test, unix))]
pub fn is_locked(lock_path: &Path) -> bool {
    use std::os::unix::io::AsRawFd;

    let file = match fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(lock_path)
    {
        Ok(f) => f,
        Err(_) => return false,
    };

    let fd = file.as_raw_fd();
    // SAFETY: flock is safe to call with a valid fd.
    let ret = unsafe { libc::flock(fd, libc::LOCK_EX | libc::LOCK_NB) };
    if ret != 0 {
        let err = std::io::Error::last_os_error();
        return err.raw_os_error() == Some(libc::EWOULDBLOCK);
    }

    // The probe acquired the lock, so release it explicitly. Dropping the file
    // would also close the fd, but an explicit unlock keeps immediate follow-up
    // probes deterministic in high-parallelism test runs.
    // SAFETY: flock is safe to call with a valid fd.
    let _ = unsafe { libc::flock(fd, libc::LOCK_UN) };
    drop(file);
    false
}

/// Portable process identity validation.
///
/// Reads a process's command name (`comm`) and start time so callers can
/// confirm that a recorded PID still refers to the same process that wrote it
/// (guard against PID reuse).
///
/// # Platform support
///
/// | Platform | `process_comm`           | `process_started_at_unix`              |
/// |----------|--------------------------|----------------------------------------|
/// | Linux    | `/proc/{pid}/comm`       | `/proc/{pid}/stat` field 22 + btime    |
/// | macOS    | `ps -p {pid} -o comm=`   | `ps -p {pid} -o lstart=`               |
/// | Other    | `Ok(None)`               | `Ok(None)`                             |
pub mod validate {
    /// Tolerance (in seconds) when comparing a recorded start time against the
    /// live process start time.  A difference of up to this many seconds is
    /// treated as "same process".
    pub const START_TIME_TOLERANCE_SECS: i64 = 2;

    /// Liveness state inferred from whether the process comm is readable.
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub enum Liveness {
        /// Process is alive — comm was readable.
        Alive,
        /// Process is gone — PID not found.
        Dead,
        /// Cannot determine liveness (permission error or unsupported platform).
        Unknown,
    }

    #[cfg(target_os = "linux")]
    mod platform {
        use std::sync::OnceLock;

        static BTIME: OnceLock<i64> = OnceLock::new();

        fn btime() -> i64 {
            *BTIME.get_or_init(|| {
                (|| -> anyhow::Result<i64> {
                    let content = std::fs::read_to_string("/proc/stat")?;
                    for line in content.lines() {
                        if let Some(rest) = line.strip_prefix("btime ") {
                            return Ok(rest.trim().parse()?);
                        }
                    }
                    anyhow::bail!("btime line not found in /proc/stat")
                })()
                .unwrap_or(0)
            })
        }

        pub fn process_comm(pid: u32) -> anyhow::Result<Option<String>> {
            let path = format!("/proc/{pid}/comm");
            match std::fs::read_to_string(&path) {
                Ok(s) => Ok(Some(s.trim().to_string())),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
                Err(e) if e.kind() == std::io::ErrorKind::PermissionDenied => {
                    tracing::debug!(pid, path, "permission denied reading process comm");
                    Ok(None)
                }
                Err(e) => Err(e.into()),
            }
        }

        pub fn process_executable_name(pid: u32) -> anyhow::Result<Option<String>> {
            let path = format!("/proc/{pid}/exe");
            match std::fs::read_link(&path) {
                Ok(target) => Ok(target
                    .file_name()
                    .map(|name| name.to_string_lossy().into_owned())),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
                Err(e) if e.kind() == std::io::ErrorKind::PermissionDenied => {
                    tracing::debug!(
                        pid,
                        path,
                        "permission denied reading process executable path"
                    );
                    Ok(None)
                }
                Err(e) => Err(e.into()),
            }
        }

        pub fn process_started_at_unix(pid: u32) -> anyhow::Result<Option<i64>> {
            let path = format!("/proc/{pid}/stat");
            let content = match std::fs::read_to_string(&path) {
                Ok(s) => s,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
                Err(e) if e.kind() == std::io::ErrorKind::PermissionDenied => {
                    tracing::debug!(pid, "permission denied reading /proc/{pid}/stat");
                    return Ok(None);
                }
                Err(e) => return Err(e.into()),
            };

            // `comm` may contain spaces and parentheses; skip past the last ')'.
            let rparen = content.rfind(')').ok_or_else(|| {
                anyhow::anyhow!("malformed /proc/{pid}/stat: no closing ')' found")
            })?;
            let after_comm = content.get(rparen + 2..).unwrap_or("");
            let fields: Vec<&str> = after_comm.split_whitespace().collect();

            // Field 22 overall (1-indexed) = index 19 (0-indexed) after the closing ')'.
            let starttime_ticks: u64 = fields
                .get(19)
                .ok_or_else(|| anyhow::anyhow!("starttime field missing in /proc/{pid}/stat"))?
                .parse()
                .map_err(|e| {
                    anyhow::anyhow!("failed to parse starttime in /proc/{pid}/stat: {e}")
                })?;

            // SAFETY: sysconf is safe to call with a valid constant.
            let clk_tck = unsafe { libc::sysconf(libc::_SC_CLK_TCK) };
            if clk_tck <= 0 {
                anyhow::bail!("sysconf(_SC_CLK_TCK) returned {clk_tck}");
            }

            let bt = btime();
            if bt == 0 {
                anyhow::bail!("could not determine boot time from /proc/stat");
            }

            Ok(Some(bt + (starttime_ticks as i64 / clk_tck)))
        }
    }

    #[cfg(target_os = "macos")]
    mod platform {
        pub fn process_comm(pid: u32) -> anyhow::Result<Option<String>> {
            let output = std::process::Command::new("ps")
                .args(["-p", &pid.to_string(), "-o", "comm="])
                .output()?;
            let s = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if s.is_empty() {
                return Ok(None);
            }
            // macOS `ps -o comm=` returns the full executable path. Normalize
            // to a basename so stop-time validation matches Linux comm behavior.
            let basename = std::path::Path::new(&s)
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or(s);
            Ok(Some(basename))
        }

        pub fn process_started_at_unix(pid: u32) -> anyhow::Result<Option<i64>> {
            // Force C locale so `ps -o lstart=` always emits English month
            // abbreviations (Jan/Feb/…) regardless of the system locale.
            let output = std::process::Command::new("ps")
                .args(["-p", &pid.to_string(), "-o", "lstart="])
                .env("LANG", "C")
                .env("LC_ALL", "C")
                .output()?;
            let s = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if s.is_empty() {
                return Ok(None);
            }
            parse_lstart(&s)
        }

        fn parse_lstart(s: &str) -> anyhow::Result<Option<i64>> {
            use chrono::{Local, NaiveDate, NaiveDateTime, NaiveTime, TimeZone};

            // split_whitespace collapses double-spaces (e.g. single-digit day padding).
            // macOS `ps -o lstart=` format: "DoW DD Mon HH:MM:SS YYYY"
            // e.g. "Tue  7 Apr 22:53:35 2026"
            let parts: Vec<&str> = s.split_whitespace().collect();
            if parts.len() != 5 {
                return Ok(None);
            }

            // parts[0]=DoW, parts[1]=day, parts[2]=month, parts[3]=HH:MM:SS, parts[4]=year
            let day: u32 = match parts[1].parse() {
                Ok(d) => d,
                Err(_) => return Ok(None),
            };
            let month: u32 = match parts[2] {
                "Jan" => 1,
                "Feb" => 2,
                "Mar" => 3,
                "Apr" => 4,
                "May" => 5,
                "Jun" => 6,
                "Jul" => 7,
                "Aug" => 8,
                "Sep" => 9,
                "Oct" => 10,
                "Nov" => 11,
                "Dec" => 12,
                _ => return Ok(None),
            };
            let year: i32 = match parts[4].parse() {
                Ok(y) => y,
                Err(_) => return Ok(None),
            };

            let time_parts: Vec<&str> = parts[3].split(':').collect();
            if time_parts.len() != 3 {
                return Ok(None);
            }
            let (hour, min, sec): (u32, u32, u32) = match (
                time_parts[0].parse(),
                time_parts[1].parse(),
                time_parts[2].parse(),
            ) {
                (Ok(h), Ok(m), Ok(s)) => (h, m, s),
                _ => return Ok(None),
            };

            let date = match NaiveDate::from_ymd_opt(year, month, day) {
                Some(d) => d,
                None => return Ok(None),
            };
            let time = match NaiveTime::from_hms_opt(hour, min, sec) {
                Some(t) => t,
                None => return Ok(None),
            };
            let naive_dt = NaiveDateTime::new(date, time);

            let local_dt = match Local.from_local_datetime(&naive_dt).single() {
                Some(dt) => dt,
                None => return Ok(None),
            };

            Ok(Some(local_dt.timestamp()))
        }

        pub fn process_executable_name(pid: u32) -> anyhow::Result<Option<String>> {
            process_comm(pid)
        }
    }

    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    mod platform {
        pub fn process_comm(_pid: u32) -> anyhow::Result<Option<String>> {
            Ok(None)
        }

        pub fn process_started_at_unix(_pid: u32) -> anyhow::Result<Option<i64>> {
            Ok(None)
        }

        pub fn process_executable_name(_pid: u32) -> anyhow::Result<Option<String>> {
            Ok(None)
        }
    }

    /// Read the command name (`comm`) of the given process.
    ///
    /// Returns `Ok(Some(name))` if the process exists, `Ok(None)` if it does
    /// not, and `Err` only for unexpected I/O errors.
    pub fn process_comm(pid: u32) -> anyhow::Result<Option<String>> {
        platform::process_comm(pid)
    }

    /// Read the Unix start time (seconds since epoch) of the given process.
    ///
    /// Returns `Ok(Some(t))` on success, `Ok(None)` if the process is gone,
    /// and `Err` for unexpected errors.
    pub fn process_started_at_unix(pid: u32) -> anyhow::Result<Option<i64>> {
        platform::process_started_at_unix(pid)
    }

    /// Read the executable basename for the given process when available.
    pub fn process_executable_name(pid: u32) -> anyhow::Result<Option<String>> {
        platform::process_executable_name(pid)
    }

    /// Determine liveness of a process by attempting to read its comm.
    ///
    /// - [`Liveness::Alive`]   — comm is readable.
    /// - [`Liveness::Dead`]    — PID not found.
    /// - [`Liveness::Unknown`] — unexpected error or unsupported platform.
    pub fn process_liveness(pid: u32) -> Liveness {
        match process_comm(pid) {
            Ok(Some(_)) => Liveness::Alive,
            Ok(None) => Liveness::Dead,
            Err(_) => Liveness::Unknown,
        }
    }

    /// Returns `true` iff the live process name matches the expected spawned binary.
    ///
    /// Linux prefers `/proc/{pid}/exe` because `/proc/{pid}/comm` truncates names to
    /// 15 bytes, which breaks flavored binaries like `llama-server-vulkan`.
    pub fn process_name_matches(pid: u32, expected_comm: &str) -> bool {
        process_executable_name(pid)
            .ok()
            .flatten()
            .is_some_and(|name| name == expected_comm)
            || process_comm(pid)
                .ok()
                .flatten()
                .is_some_and(|name| name == expected_comm)
    }

    /// Returns `true` iff the live process identified by `pid` has:
    /// 1. A comm that equals `expected_comm`, **and**
    /// 2. A start time within [`START_TIME_TOLERANCE_SECS`] of `expected_started_at_unix`.
    ///
    /// Returns `false` on any error or mismatch.
    #[cfg(not(windows))]
    pub fn validate_pid_matches(
        pid: u32,
        expected_comm: &str,
        expected_started_at_unix: i64,
    ) -> bool {
        match process_started_at_unix(pid) {
            Ok(Some(t)) => {
                process_name_matches(pid, expected_comm)
                    && (t - expected_started_at_unix).abs() <= START_TIME_TOLERANCE_SECS
            }
            _ => false,
        }
    }

    /// Returns the Unix start time of the current process.
    pub fn current_process_start_time_unix() -> anyhow::Result<i64> {
        process_started_at_unix(std::process::id())?
            .ok_or_else(|| anyhow::anyhow!("could not determine start time of current process"))
    }
}

/// Snapshot of a co-located mesh-llm instance discovered via the runtime root.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct LocalInstanceSnapshot {
    /// PID of the mesh-llm process that owns this runtime directory.
    pub pid: u32,
    /// Console/management API port reported in owner.json, if present.
    pub api_port: Option<u16>,
    /// Version string from owner.json, if present.
    pub version: Option<String>,
    /// Unix timestamp (seconds) when the owner process started.
    pub started_at_unix: i64,
    /// Absolute path to the runtime directory (`{root}/{pid}/`).
    pub runtime_dir: PathBuf,
    /// True iff this snapshot refers to the calling process itself.
    pub is_self: bool,
}

/// Deserialisation target for `owner.json` written by each instance on startup.
#[derive(Deserialize)]
struct OwnerMetadata {
    pid: u32,
    api_port: Option<u16>,
    version: Option<String>,
    started_at_unix: Option<i64>,
    mesh_llm_binary: Option<String>,
}

#[derive(Debug, Clone)]
pub(crate) struct RuntimeProcessTarget {
    pub label: String,
    pub pid: u32,
    pub expected_comm: String,
    pub expected_start_time: Option<i64>,
}

fn binary_process_name(binary: &str) -> Option<String> {
    let path = Path::new(binary);

    #[cfg(windows)]
    {
        path.file_stem()
            .map(|name| name.to_string_lossy().into_owned())
    }

    #[cfg(not(windows))]
    {
        path.file_name()
            .map(|name| name.to_string_lossy().into_owned())
    }
}

pub(crate) fn collect_runtime_stop_targets(
    root: &Path,
) -> anyhow::Result<Vec<RuntimeProcessTarget>> {
    if !root.exists() {
        return Ok(Vec::new());
    }

    let mut targets = Vec::new();

    for entry in fs::read_dir(root)
        .with_context(|| format!("failed to read runtime root: {}", root.display()))?
        .flatten()
    {
        let entry_path = entry.path();
        if !entry_path.is_dir() {
            continue;
        }

        let owner_path = entry_path.join("owner.json");
        if !owner_path.exists() {
            continue;
        }

        let owner_json = match fs::read_to_string(&owner_path) {
            Ok(owner_json) => owner_json,
            Err(err) => {
                tracing::warn!(
                    path = %owner_path.display(),
                    error = %err,
                    "failed to read owner.json while collecting stop targets"
                );
                continue;
            }
        };

        let owner: OwnerMetadata = match serde_json::from_str(&owner_json) {
            Ok(owner) => owner,
            Err(err) => {
                tracing::warn!(
                    path = %owner_path.display(),
                    error = %err,
                    "failed to parse owner.json while collecting stop targets"
                );
                continue;
            }
        };

        let expected_comm = owner
            .mesh_llm_binary
            .as_deref()
            .and_then(binary_process_name)
            .unwrap_or_else(|| "mesh-llm".to_string());

        targets.push(RuntimeProcessTarget {
            label: expected_comm.clone(),
            pid: owner.pid,
            expected_comm,
            expected_start_time: owner.started_at_unix,
        });
    }

    Ok(targets)
}

/// Scan `root` for live co-located mesh-llm instances.
///
/// Each subdirectory under `root` represents one instance slot (`{root}/{pid}/`).
/// An instance is considered live if its PID is still alive according to
/// [`validate::process_liveness`]. Stale directories (dead owner) are skipped.
///
/// Returns `Ok(vec![])` immediately if `root` does not exist (first run).
///
/// All blocking filesystem I/O is delegated to [`tokio::task::spawn_blocking`].
pub async fn scan_local_instances(
    root: &Path,
    my_pid: u32,
) -> anyhow::Result<Vec<LocalInstanceSnapshot>> {
    if !root.exists() {
        return Ok(vec![]);
    }

    let root_owned = root.to_owned();
    let snapshots =
        tokio::task::spawn_blocking(move || -> anyhow::Result<Vec<LocalInstanceSnapshot>> {
            let mut snapshots = Vec::new();

            for entry in fs::read_dir(&root_owned)
                .with_context(|| format!("failed to read runtime root: {}", root_owned.display()))?
                .flatten()
            {
                let entry_path = entry.path();
                if !entry_path.is_dir() {
                    continue;
                }

                let owner_path = entry_path.join("owner.json");
                if !owner_path.exists() {
                    continue;
                }

                let json = match fs::read_to_string(&owner_path) {
                    Ok(s) => s,
                    Err(e) => {
                        tracing::warn!(
                            path = %owner_path.display(),
                            error = %e,
                            "failed to read owner.json — skipping"
                        );
                        continue;
                    }
                };

                let meta: OwnerMetadata = match serde_json::from_str(&json) {
                    Ok(m) => m,
                    Err(e) => {
                        tracing::warn!(
                            path = %owner_path.display(),
                            error = %e,
                            "failed to parse owner.json — skipping"
                        );
                        continue;
                    }
                };

                if validate::process_liveness(meta.pid) == validate::Liveness::Dead {
                    continue;
                }

                snapshots.push(LocalInstanceSnapshot {
                    pid: meta.pid,
                    api_port: meta.api_port,
                    version: meta.version,
                    started_at_unix: meta.started_at_unix.unwrap_or(0),
                    runtime_dir: entry_path,
                    is_self: meta.pid == my_pid,
                });
            }

            Ok(snapshots)
        })
        .await
        .context("scan_local_instances task panicked")??;

    Ok(snapshots)
}

/// Spawn a background task that refreshes `shared` every 5 seconds.
///
/// On each iteration the task calls [`scan_local_instances`] and, on success,
/// replaces the shared state atomically (short lock hold — never held across an
/// await point). Errors are logged with [`tracing::warn!`] and the loop continues.
///
/// The returned [`tokio::task::JoinHandle`] may be dropped; the task runs until
/// the process exits.
pub fn spawn_local_instance_scanner(
    root: PathBuf,
    my_pid: u32,
    runtime_data_producer: crate::runtime_data::RuntimeDataProducer,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(std::time::Duration::from_secs(5)).await;
            match scan_local_instances(&root, my_pid).await {
                Ok(instances) => {
                    publish_local_instance_scan_results(&runtime_data_producer, instances);
                }
                Err(e) => {
                    tracing::warn!("local instance scan failed: {e}");
                }
            }
        }
    })
}

pub(crate) fn publish_local_instance_scan_results(
    runtime_data_producer: &crate::runtime_data::RuntimeDataProducer,
    instances: Vec<LocalInstanceSnapshot>,
) -> bool {
    runtime_data_producer.replace_local_instances_snapshot(instances)
}

#[cfg(test)]
mod scan_tests {
    use super::*;
    use serial_test::serial;
    use tempfile::tempdir;

    fn write_owner_json(
        dir: &Path,
        pid: u32,
        api_port: Option<u16>,
        version: &str,
        started_at: i64,
    ) {
        let meta = serde_json::json!({
            "pid": pid,
            "api_port": api_port,
            "version": version,
            "started_at_unix": started_at,
            "mesh_llm_binary": "/usr/bin/mesh-llm",
        });
        let json = serde_json::to_string_pretty(&meta).expect("serialise owner meta");
        write_text_file_atomic(&dir.join("owner.json"), &json).expect("write owner.json");
    }

    #[tokio::test]
    #[serial]
    async fn scan_returns_empty_when_root_missing() {
        let tmp = tempdir().unwrap();
        let missing = tmp.path().join("nonexistent-runtime-root");
        let result = scan_local_instances(&missing, 1000)
            .await
            .expect("scan should not error for missing root");
        assert!(result.is_empty(), "missing root must yield empty result");
    }

    #[tokio::test]
    #[serial]
    async fn scan_includes_self() {
        let root = tempdir().unwrap();
        let my_pid = std::process::id();
        let instance_dir = root.path().join(my_pid.to_string());
        fs::create_dir_all(&instance_dir).unwrap();
        write_owner_json(&instance_dir, my_pid, Some(3131), "0.99.0-test", 1700000000);

        let result = scan_local_instances(root.path(), my_pid)
            .await
            .expect("scan should succeed");
        assert_eq!(result.len(), 1, "own instance must appear in results");
        assert!(
            result[0].is_self,
            "entry for own pid must have is_self=true"
        );
        assert_eq!(result[0].pid, my_pid);
    }

    #[tokio::test]
    #[serial]
    async fn scan_skips_dead_owners() {
        let root = tempdir().unwrap();
        // PID 999999 is almost certainly dead on any test machine.
        let dead_pid: u32 = 999_999;
        let instance_dir = root.path().join(dead_pid.to_string());
        fs::create_dir_all(&instance_dir).unwrap();
        write_owner_json(&instance_dir, dead_pid, None, "0.99.0-test", 1700000000);

        let result = scan_local_instances(root.path(), std::process::id())
            .await
            .expect("scan should succeed");
        assert!(
            result.is_empty(),
            "dead-owner entry must be skipped, got: {result:?}"
        );
    }

    #[tokio::test]
    #[serial]
    async fn scan_reads_all_fields() {
        let root = tempdir().unwrap();
        let my_pid = std::process::id();
        let instance_dir = root.path().join(my_pid.to_string());
        fs::create_dir_all(&instance_dir).unwrap();
        write_owner_json(&instance_dir, my_pid, Some(3131), "0.42.0", 1700000000);

        let result = scan_local_instances(root.path(), my_pid)
            .await
            .expect("scan should succeed");
        assert_eq!(result.len(), 1);
        let snap = &result[0];
        assert_eq!(snap.pid, my_pid);
        assert_eq!(snap.api_port, Some(3131));
        assert_eq!(snap.version.as_deref(), Some("0.42.0"));
        assert_eq!(snap.started_at_unix, 1700000000);
        assert_eq!(snap.runtime_dir, instance_dir);
        assert!(snap.is_self);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;
    use tempfile::tempdir;

    struct EnvGuard {
        key: String,
        original: Option<String>,
    }

    impl EnvGuard {
        fn save_and_remove(key: &str) -> Self {
            let original = std::env::var(key).ok();
            #[allow(deprecated)]
            std::env::remove_var(key);
            Self {
                key: key.to_string(),
                original,
            }
        }

        fn save_and_set(key: &str, value: &str) -> Self {
            let original = std::env::var(key).ok();
            #[allow(deprecated)]
            std::env::set_var(key, value);
            Self {
                key: key.to_string(),
                original,
            }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            match &self.original {
                #[allow(deprecated)]
                Some(v) => std::env::set_var(&self.key, v),
                #[allow(deprecated)]
                None => std::env::remove_var(&self.key),
            }
        }
    }

    #[cfg(unix)]
    fn wait_until_unlocked(lock_path: &Path) -> bool {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(1);
        while std::time::Instant::now() < deadline {
            if !is_locked(lock_path) {
                return true;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        !is_locked(lock_path)
    }

    #[test]
    #[serial]
    fn runtime_root_respects_env_override() {
        let dir = tempdir().unwrap();
        let _g = EnvGuard::save_and_set("MESH_LLM_RUNTIME_ROOT", dir.path().to_str().unwrap());

        let root = runtime_root().expect("runtime_root should succeed");
        assert_eq!(root, dir.path());
    }

    #[test]
    #[serial]
    fn runtime_root_falls_back_to_xdg() {
        let dir = tempdir().unwrap();
        let _g_mesh = EnvGuard::save_and_remove("MESH_LLM_RUNTIME_ROOT");
        let _g_xdg = EnvGuard::save_and_set("XDG_RUNTIME_DIR", dir.path().to_str().unwrap());

        let root = runtime_root().expect("runtime_root should succeed with XDG");
        assert_eq!(root, dir.path().join("mesh-llm").join("runtime"));
    }

    #[cfg(not(windows))]
    #[test]
    #[serial]
    fn runtime_root_falls_back_to_home() {
        let dir = tempdir().unwrap();
        let _g_mesh = EnvGuard::save_and_remove("MESH_LLM_RUNTIME_ROOT");
        let _g_xdg = EnvGuard::save_and_remove("XDG_RUNTIME_DIR");
        let _g_home = EnvGuard::save_and_set("HOME", dir.path().to_str().unwrap());

        let root = runtime_root().expect("runtime_root should succeed with HOME");
        assert_eq!(root, dir.path().join(".mesh-llm").join("runtime"));
    }

    #[cfg(windows)]
    #[test]
    #[serial]
    fn runtime_root_falls_back_to_windows_profile_without_home() {
        let _g_mesh = EnvGuard::save_and_remove("MESH_LLM_RUNTIME_ROOT");
        let _g_xdg = EnvGuard::save_and_remove("XDG_RUNTIME_DIR");
        let _g_home = EnvGuard::save_and_remove("HOME");

        let home = dirs::home_dir().expect("Windows profile directory should be available");
        let root = runtime_root().expect("runtime_root should succeed without HOME on Windows");
        assert_eq!(root, home.join(".mesh-llm").join("runtime"));
    }

    #[test]
    #[serial]
    fn runtime_root_bails_when_unset() {
        let _g_mesh = EnvGuard::save_and_remove("MESH_LLM_RUNTIME_ROOT");
        let _g_xdg = EnvGuard::save_and_remove("XDG_RUNTIME_DIR");
        let _g_home = EnvGuard::save_and_remove("HOME");

        let result = runtime_root_with_home(None);
        assert!(
            result.is_err(),
            "runtime_root must bail when no path source is set"
        );
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("HOME")
                || msg.contains("XDG_RUNTIME_DIR")
                || msg.contains("MESH_LLM_RUNTIME_ROOT"),
            "error message should name the missing env vars, got: {msg}"
        );
    }

    #[test]
    #[serial]
    fn acquire_creates_directories() {
        let dir = tempdir().unwrap();
        let _g = EnvGuard::save_and_set("MESH_LLM_RUNTIME_ROOT", dir.path().to_str().unwrap());

        let rt = InstanceRuntime::acquire(1001).expect("acquire should succeed");

        assert!(rt.dir().exists(), "runtime dir must be created");
        assert!(rt.dir().join("lock").exists(), "lock file must be created");
    }

    #[test]
    #[serial]
    #[cfg(unix)]
    fn acquire_holds_flock() {
        let dir = tempdir().unwrap();
        let _g = EnvGuard::save_and_set("MESH_LLM_RUNTIME_ROOT", dir.path().to_str().unwrap());

        let rt = InstanceRuntime::acquire(1002).expect("acquire should succeed");
        let lock_path = rt.dir().join("lock");

        assert!(
            is_locked(&lock_path),
            "lock file must be held while InstanceRuntime is live"
        );

        drop(rt);

        assert!(
            wait_until_unlocked(&lock_path),
            "lock file must be released after InstanceRuntime is dropped"
        );
    }

    #[test]
    #[serial]
    fn acquire_second_time_fails() {
        let dir = tempdir().unwrap();
        let _g = EnvGuard::save_and_set("MESH_LLM_RUNTIME_ROOT", dir.path().to_str().unwrap());

        let _rt = InstanceRuntime::acquire(1003).expect("first acquire should succeed");
        let result = InstanceRuntime::acquire(1003);
        assert!(
            result.is_err(),
            "second acquire of same pid slot must fail while first is held"
        );
    }

    #[test]
    #[serial]
    #[cfg(unix)]
    fn is_locked_returns_true_while_held() {
        let dir = tempdir().unwrap();
        let _g = EnvGuard::save_and_set("MESH_LLM_RUNTIME_ROOT", dir.path().to_str().unwrap());

        let rt = InstanceRuntime::acquire(1004).expect("acquire should succeed");
        let lock_path = rt.dir().join("lock");

        assert!(
            is_locked(&lock_path),
            "is_locked must return true while InstanceRuntime holds the flock"
        );
    }

    #[test]
    #[serial]
    #[cfg(unix)]
    fn is_locked_returns_false_after_drop() {
        let dir = tempdir().unwrap();
        let _g = EnvGuard::save_and_set("MESH_LLM_RUNTIME_ROOT", dir.path().to_str().unwrap());

        let rt = InstanceRuntime::acquire(1005).expect("acquire should succeed");
        let lock_path = rt.dir().join("lock");

        drop(rt);

        assert!(
            wait_until_unlocked(&lock_path),
            "is_locked must return false after InstanceRuntime is dropped"
        );
    }

    #[test]
    fn write_text_file_atomic_cleans_up_tmp_file_on_error() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("owner.json");
        fs::create_dir_all(&path).unwrap();

        let result = write_text_file_atomic(&path, "{}{}");
        assert!(result.is_err(), "rename into directory should fail");

        let tmp_path = super::tmp_path_for(&path);
        assert!(
            !tmp_path.exists(),
            "tmp file should be removed when the atomic write fails"
        );
    }

    #[test]
    fn validate_self_process_comm_returns_something() {
        let pid = std::process::id();
        let result = validate::process_comm(pid).expect("process_comm should not error for self");
        let comm = result.expect("process_comm should return Some for self process");
        assert!(!comm.is_empty(), "comm for self process must be non-empty");
    }

    #[test]
    fn validate_self_process_start_time_is_recent() {
        let pid = std::process::id();
        let t = match validate::process_started_at_unix(pid)
            .expect("process_started_at_unix should not error for self")
        {
            Some(t) => t,
            None => return,
        };
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;
        assert!(t > 0, "start time must be positive");
        assert!(
            now - t < 3600,
            "process must have started within the last hour, got t={t}, now={now}"
        );
    }

    #[test]
    fn validate_nonexistent_pid_is_dead() {
        assert_eq!(
            validate::process_liveness(999999),
            validate::Liveness::Dead,
            "PID 999999 must report Dead liveness"
        );
    }

    #[test]
    #[cfg(not(windows))]
    fn validate_pid_matches_rejects_wrong_comm() {
        let pid = std::process::id();
        assert!(
            !validate::validate_pid_matches(pid, "definitely-not-this-comm-string", 0),
            "wrong comm must cause validate_pid_matches to return false"
        );
    }

    #[test]
    #[cfg(not(windows))]
    fn validate_pid_matches_rejects_wrong_start_time() {
        let pid = std::process::id();
        let comm = match validate::process_comm(pid).ok().flatten() {
            Some(c) => c,
            None => return,
        };
        let t = match validate::process_started_at_unix(pid).ok().flatten() {
            Some(t) => t,
            None => return,
        };
        assert!(
            !validate::validate_pid_matches(pid, &comm, t + 60),
            "start time off by 60s must be rejected"
        );
    }

    #[test]
    fn process_name_matches_accepts_comm_match_for_self() {
        let pid = std::process::id();
        let comm = match validate::process_comm(pid).ok().flatten() {
            Some(c) => c,
            None => return,
        };
        assert!(
            validate::process_name_matches(pid, &comm),
            "a matching comm must be accepted even when executable basename differs"
        );
    }

    #[test]
    #[cfg(not(windows))]
    fn validate_pid_matches_accepts_within_tolerance() {
        let pid = std::process::id();
        let comm = match validate::process_comm(pid).ok().flatten() {
            Some(c) => c,
            None => return,
        };
        let t = match validate::process_started_at_unix(pid).ok().flatten() {
            Some(t) => t,
            None => return,
        };
        assert!(
            validate::validate_pid_matches(pid, &comm, t + 1),
            "start time off by 1s must be accepted (tolerance is {}s)",
            validate::START_TIME_TOLERANCE_SECS
        );
    }

    #[test]
    #[cfg(not(windows))]
    fn validate_pid_matches_rejects_outside_tolerance() {
        let pid = std::process::id();
        let comm = match validate::process_comm(pid).ok().flatten() {
            Some(c) => c,
            None => return,
        };
        let t = match validate::process_started_at_unix(pid).ok().flatten() {
            Some(t) => t,
            None => return,
        };
        assert!(
            !validate::validate_pid_matches(pid, &comm, t + 3),
            "start time off by 3s must be rejected (tolerance is {}s)",
            validate::START_TIME_TOLERANCE_SECS
        );
    }

    #[test]
    fn validate_current_process_start_time_is_positive() {
        if let Ok(t) = validate::current_process_start_time_unix() {
            assert!(
                t > 0,
                "current process start time must be positive, got {t}"
            );
        }
    }

    #[test]
    fn validate_liveness_dead_for_nonexistent_pid() {
        assert_eq!(
            validate::process_liveness(999999),
            validate::Liveness::Dead,
            "liveness for nonexistent PID 999999 must be Dead"
        );
    }
}
