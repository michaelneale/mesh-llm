//! Per-instance runtime directory management with scoped child-process cleanup.
//!
//! Each non-client mesh-llm invocation acquires an `InstanceRuntime` under
//! `~/.mesh-llm/runtime/{pid}/` (overridable via env vars). The directory
//! holds an advisory `flock(2)` lock for the instance's lifetime, plus
//! pidfiles for every child process that mesh-llm spawns. On startup,
//! other mesh-llm instances scan the root for runtime dirs whose locks
//! are not held (their owners are dead) and reap any orphaned children
//! listed in their pidfiles.
//!
//! # Runtime directory layout
//!
//! **ALLOWED** under `runtime_dir/`:
//! - `lock` — `flock(2)` advisory lock file held by the owning mesh-llm
//! - `owner.json` — metadata about the owning instance (pid, version, api_port, started_at)
//! - `pidfiles/` — JSON pidfiles for child processes (`llama-server.json`, `rpc-server-{port}.json`)
//! - `logs/` — stdout/stderr logs for each child (`llama-server-{port}.log`, `rpc-server-{port}.log`)
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
//! 3. `$HOME/.mesh-llm/runtime` (default interactive use)
//! 4. Fails fast with a clear error if none of the above are set
//!
//! # Liveness detection
//!
//! Primary mechanism: `libc::flock(LOCK_EX | LOCK_NB)` on the `lock` file.
//! Released automatically by the kernel when the owning fd closes (including
//! on `SIGKILL`). Race-free and survives all abnormal terminations.
//!
//! Secondary (PID validation before killing a child):
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
//! - **Windows**: orphan detection is best-effort. `flock` is a no-op.
//!   Runtime dirs are still created but cleanup falls back to current behavior.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
#[cfg(unix)]
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Maximum bytes for argv snippet in pidfile metadata.
pub const ARGV_SNIPPET_MAX_BYTES: usize = 256;

/// Metadata for a child process pidfile (JSON format).
///
/// Written atomically to `{runtime_dir}/pidfiles/{name}.json` and read back
/// for liveness validation and process reaping.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PidfileMetadata {
    /// Command name (e.g., "llama-server", "rpc-server").
    pub cmd_name: String,
    /// Child process PID.
    pub child_pid: u32,
    /// Child process start time (Unix timestamp in seconds).
    pub child_started_at_unix: i64,
    /// Owner process PID (the mesh-llm instance that spawned the child).
    pub owner_pid: u32,
    /// Owner process start time (Unix timestamp in seconds).
    pub owner_started_at_unix: i64,
    /// Truncated argv snippet (capped at ARGV_SNIPPET_MAX_BYTES).
    pub argv_snippet: String,
    /// Runtime directory path where this pidfile is stored.
    pub runtime_dir: PathBuf,
}

impl PidfileMetadata {
    /// Truncate argv to max_bytes at a valid UTF-8 boundary, appending "…" if truncated.
    pub fn cap_argv(argv: &[String], max_bytes: usize) -> String {
        let joined = argv.join(" ");
        if joined.len() <= max_bytes {
            return joined;
        }

        if max_bytes == 0 {
            return String::new();
        }

        let ellipsis = '…';
        let ellipsis_len = ellipsis.len_utf8();
        let mut cutoff = if max_bytes < ellipsis_len {
            max_bytes.min(joined.len())
        } else {
            max_bytes.saturating_sub(ellipsis_len).min(joined.len())
        };
        while cutoff > 0 && !joined.is_char_boundary(cutoff) {
            cutoff -= 1;
        }

        let mut truncated = joined[..cutoff].to_string();
        if ellipsis_len > max_bytes {
            return truncated;
        }

        truncated.push(ellipsis);
        truncated
    }

    /// Write this metadata to a pidfile atomically.
    ///
    /// Writes to `{path}.tmp`, calls `sync_all()`, then renames to `path`.
    /// If writing or renaming fails, removes the tmp file before returning the error.
    pub fn write_atomic(&self, path: &Path) -> Result<()> {
        if self.child_pid == 0 {
            anyhow::bail!("refusing to write pidfile with child_pid=0");
        }
        let json =
            serde_json::to_string_pretty(self).context("failed to serialize pidfile metadata")?;

        write_text_file_atomic(path, &json).context("failed to write pidfile atomically")
    }

    /// Read and deserialize a pidfile from disk.
    ///
    /// Returns an error (never panics) if the file is corrupt or missing.
    pub fn read(path: &Path) -> Result<Self> {
        let content = fs::read_to_string(path)
            .with_context(|| format!("failed to read pidfile: {}", path.display()))?;

        serde_json::from_str(&content)
            .map_err(|err| anyhow::anyhow!("corrupt pidfile at {}: {}", path.display(), err))
    }
}

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

/// RAII guard that removes a pidfile when dropped.
///
/// Logs via `tracing::debug!` if removal fails; never panics.
#[derive(Debug)]
pub struct PidfileGuard {
    path: PathBuf,
}

impl PidfileGuard {
    /// Create a new guard for the given pidfile path.
    pub fn new(path: PathBuf) -> Self {
        Self { path }
    }
}

impl Drop for PidfileGuard {
    fn drop(&mut self) {
        if let Err(e) = fs::remove_file(&self.path) {
            if e.kind() != std::io::ErrorKind::NotFound {
                tracing::debug!(
                    path = %self.path.display(),
                    error = %e,
                    "failed to remove pidfile on drop"
                );
            }
        }
    }
}

/// Resolve the runtime root directory for this mesh-llm installation.
///
/// Precedence:
/// 1. `MESH_LLM_RUNTIME_ROOT` environment variable (test override / custom deployment)
/// 2. `$XDG_RUNTIME_DIR/mesh-llm/runtime`
/// 3. `$HOME/.mesh-llm/runtime` (via [`dirs::home_dir`])
/// 4. [`anyhow::bail!`] — at least one of the above must be set
pub fn runtime_root() -> Result<PathBuf> {
    // 1. Explicit override — always wins (also used by tests to avoid touching ~)
    if let Ok(root) = std::env::var("MESH_LLM_RUNTIME_ROOT") {
        return Ok(PathBuf::from(root));
    }

    // 2. XDG_RUNTIME_DIR (standard on modern Linux)
    if let Ok(xdg) = std::env::var("XDG_RUNTIME_DIR") {
        return Ok(PathBuf::from(xdg).join("mesh-llm").join("runtime"));
    }

    // 3. $HOME/.mesh-llm/runtime — only when HOME env var is explicitly set
    if std::env::var_os("HOME").is_some_and(|h| !h.is_empty()) {
        if let Some(home) = dirs::home_dir() {
            return Ok(home.join(".mesh-llm").join("runtime"));
        }
    }

    // 4. Nothing usable — fail fast with a clear message
    anyhow::bail!("mesh-llm requires HOME, XDG_RUNTIME_DIR, or MESH_LLM_RUNTIME_ROOT to be set")
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
    /// - `{root}/{pid}/pidfiles/`
    /// - `{root}/{pid}/logs/`
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
        fs::create_dir_all(dir.join("pidfiles")).context("failed to create pidfiles directory")?;
        fs::create_dir_all(dir.join("logs")).context("failed to create logs directory")?;

        // On Unix, harden permissions on the runtime directories so that
        // pidfiles and logs are only readable by the owning user.
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let private = std::fs::Permissions::from_mode(0o700);
            for d in [&root, &dir, &dir.join("pidfiles"), &dir.join("logs")] {
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

    /// Returns the path where the named child's pidfile should be written.
    ///
    /// Conventionally `{dir}/pidfiles/{name}.json`.
    pub fn pidfile_path(&self, name: &str) -> PathBuf {
        self.dir.join("pidfiles").join(format!("{name}.json"))
    }

    /// Returns the path where the named child's log file should be written.
    ///
    /// Conventionally `{dir}/logs/{name}.log`.
    pub fn log_path(&self, name: &str) -> PathBuf {
        self.dir.join("logs").join(format!("{name}.log"))
    }

    /// The PID this runtime slot was acquired for.
    #[allow(dead_code)]
    pub fn pid(&self) -> u32 {
        self.pid
    }

    /// Write a pidfile atomically and return an RAII guard that removes it on drop.
    pub fn write_pidfile(&self, name: &str, metadata: &PidfileMetadata) -> Result<PidfileGuard> {
        let path = self.pidfile_path(name);
        metadata.write_atomic(&path)?;
        Ok(PidfileGuard::new(path))
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
/// On non-Unix platforms always returns `false`.
pub fn is_locked(lock_path: &Path) -> bool {
    #[cfg(unix)]
    {
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
        // SAFETY: flock is safe to call with a valid fd
        let ret = unsafe { libc::flock(fd, libc::LOCK_EX | libc::LOCK_NB) };
        if ret != 0 {
            let err = std::io::Error::last_os_error();
            return err.raw_os_error() == Some(libc::EWOULDBLOCK);
        }

        drop(file);
        false
    }

    #[cfg(not(unix))]
    {
        let _ = lock_path;
        false
    }
}

/// Portable process identity validation.
///
/// Reads a process's command name (`comm`) and start time so callers can
/// confirm that a PID in a pidfile still refers to the same process that wrote
/// it (guard against PID reuse).
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
            // macOS `ps -o comm=` returns the full executable path (e.g.
            // "/.../llama.cpp/build/bin/llama-server"). Pidfile metadata stores
            // just the basename so the orphan reaper can match against it on
            // either Linux (`/proc/{pid}/comm` is the basename) or macOS.
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

/// Pure reap decision logic for scoped runtime cleanup.
///
/// Determines what action to take for a pidfile entry based on liveness
/// observations already collected by the caller. No IO is performed here.
pub mod reap {
    use super::validate;
    use super::PidfileMetadata;
    use anyhow::Context;
    use std::path::{Path, PathBuf};

    #[derive(Debug)]
    enum PendingActionKind {
        KillChildAndRemovePidfile {
            child_pid: u32,
            cmd_name: String,
            child_started_at_unix: i64,
        },
        RemovePidfileOnly,
    }

    #[derive(Debug)]
    struct PendingAction {
        pidfile_path: PathBuf,
        kind: PendingActionKind,
    }

    #[derive(Debug, Default)]
    struct ScanResult {
        summary: ReapSummary,
        pending_actions: Vec<PendingAction>,
        stale_runtime_dirs: Vec<PathBuf>,
    }

    /// Decision returned by [`decide_action`].
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub enum Action {
        /// Owner process is alive — hands off, do not touch.
        Keep,
        /// Orphaned child we own — kill the child and remove the pidfile.
        KillChildAndRemovePidfile,
        /// Child is dead or PID was recycled — remove the pidfile only.
        RemovePidfileOnly,
        /// Cannot determine liveness — leave for next scan.
        Skip,
    }

    /// Pure decision function — NO IO inside this function.
    ///
    /// Determines what action to take for a pidfile entry given the liveness
    /// observations already collected by the caller.
    ///
    /// # Decision rules
    ///
    /// 1. If `owner_dir_is_locked` → [`Action::Keep`] (owner is alive, never touch).
    /// 2. Owner is dead. Match on `child_liveness`:
    ///    - [`validate::Liveness::Dead`] → [`Action::RemovePidfileOnly`]
    ///    - [`validate::Liveness::Alive`] with `child_comm_matches && child_start_time_matches`
    ///      → [`Action::KillChildAndRemovePidfile`]
    ///    - [`validate::Liveness::Alive`] with any mismatch (PID recycled — defensive)
    ///      → [`Action::RemovePidfileOnly`]
    ///    - [`validate::Liveness::Unknown`] → [`Action::Skip`]
    pub fn decide_action(
        _metadata: &PidfileMetadata,
        child_liveness: validate::Liveness,
        child_comm_matches: bool,
        child_start_time_matches: bool,
        owner_dir_is_locked: bool,
    ) -> Action {
        if owner_dir_is_locked {
            return Action::Keep;
        }

        // Owner is dead.
        match child_liveness {
            validate::Liveness::Dead => Action::RemovePidfileOnly,
            validate::Liveness::Alive => {
                if child_comm_matches && child_start_time_matches {
                    Action::KillChildAndRemovePidfile
                } else {
                    // PID recycled (defensive) — just remove the stale pidfile.
                    Action::RemovePidfileOnly
                }
            }
            validate::Liveness::Unknown => Action::Skip,
        }
    }

    /// Summary of a completed reap scan.
    #[derive(Debug, Clone, Copy, Default)]
    pub struct ReapSummary {
        /// Number of runtime directories scanned.
        pub dirs_scanned: usize,
        /// Number of directories skipped because the owner is still alive.
        pub dirs_skipped_alive: usize,
        /// Number of directories that were garbage collected.
        pub dirs_gc_d: usize,
        /// Number of child processes killed.
        pub children_killed: usize,
        /// Number of pidfiles removed.
        pub pidfiles_removed: usize,
        /// Number of entries skipped due to errors.
        pub skipped_errors: usize,
    }

    /// Scan all peer runtime directories under `root` and reap orphaned children
    /// whose owner process is dead.
    ///
    /// Skips `my_runtime_dir` (own slot by path comparison), directories with live
    /// owners (flock still held), and entries that cannot be parsed. Removes stale
    /// directories older than 1 hour that have no live owner.
    pub async fn reap_cross_runtime_orphans(
        root: &std::path::Path,
        my_runtime_dir: &std::path::Path,
    ) -> anyhow::Result<ReapSummary> {
        #[cfg(not(unix))]
        {
            let _ = root;
            let _ = my_runtime_dir;
            return Ok(ReapSummary::default());
        }

        if !root.exists() {
            return Ok(ReapSummary::default());
        }

        let root = root.to_path_buf();
        let my_runtime_dir = my_runtime_dir.to_path_buf();
        let scan =
            tokio::task::spawn_blocking(move || scan_cross_runtime_orphans(&root, &my_runtime_dir))
                .await
                .context("cross-runtime reap scan task panicked")??;

        let mut summary = scan.summary;
        let pending_actions = scan.pending_actions;
        let stale_runtime_dirs = scan.stale_runtime_dirs;
        let mut pidfiles_to_remove = Vec::new();

        for action in &pending_actions {
            match &action.kind {
                PendingActionKind::KillChildAndRemovePidfile {
                    child_pid,
                    cmd_name,
                    child_started_at_unix,
                } => {
                    // Treat 0 as "unknown start time" (detection failed at pidfile creation).
                    let start_time_hint = if *child_started_at_unix == 0 {
                        None
                    } else {
                        Some(*child_started_at_unix)
                    };
                    let terminated = crate::inference::launch::terminate_process(
                        *child_pid,
                        cmd_name,
                        start_time_hint,
                    )
                    .await;
                    let exited = crate::inference::launch::wait_for_exit(*child_pid, 5000).await;
                    let force_killed = if exited {
                        false
                    } else {
                        crate::inference::launch::force_kill_process(
                            *child_pid,
                            cmd_name,
                            start_time_hint,
                        )
                        .await
                    };
                    let exited_after_force = if exited {
                        true
                    } else {
                        crate::inference::launch::wait_for_exit(*child_pid, 5000).await
                    };

                    if (terminated || force_killed) && exited_after_force {
                        summary.children_killed += 1;
                        summary.pidfiles_removed += 1;
                        pidfiles_to_remove.push(action.pidfile_path.clone());
                    } else {
                        tracing::warn!(
                            pid = *child_pid,
                            cmd_name,
                            terminated,
                            exited,
                            force_killed,
                            exited_after_force,
                            "failed to reap orphan child; keeping pidfile"
                        );
                    }
                }
                PendingActionKind::RemovePidfileOnly => {
                    summary.pidfiles_removed += 1;
                    pidfiles_to_remove.push(action.pidfile_path.clone());
                }
            }
        }

        tokio::task::spawn_blocking(move || {
            for path in pidfiles_to_remove {
                let _ = std::fs::remove_file(path);
            }
            for path in stale_runtime_dirs {
                let _ = std::fs::remove_dir_all(path);
            }
        })
        .await
        .context("cross-runtime reap cleanup task panicked")?;

        Ok(summary)
    }

    /// Drain stale pidfiles from our own runtime directory.
    /// Used before retrying a spawn (e.g., pre-rpc-server start).
    /// Unlike reap_cross_runtime_orphans, this operates on OUR OWN dir only.
    /// If a child is alive and matches, it is killed (retry drain).
    /// If dead or mismatched, the pidfile is removed defensively.
    pub async fn reap_own_stale_pidfiles(my_dir: &std::path::Path) -> anyhow::Result<ReapSummary> {
        let my_dir = my_dir.to_path_buf();
        if !my_dir.join("pidfiles").exists() {
            return Ok(ReapSummary::default());
        }

        let scan = tokio::task::spawn_blocking(move || scan_own_stale_pidfiles(&my_dir))
            .await
            .context("own-runtime reap scan task panicked")??;

        let mut summary = scan.summary;
        let pending_actions = scan.pending_actions;
        let mut pidfiles_to_remove = Vec::new();

        for action in &pending_actions {
            match &action.kind {
                PendingActionKind::KillChildAndRemovePidfile {
                    child_pid,
                    cmd_name,
                    child_started_at_unix,
                } => {
                    // Treat 0 as "unknown start time" (detection failed at pidfile creation).
                    let start_time_hint = if *child_started_at_unix == 0 {
                        None
                    } else {
                        Some(*child_started_at_unix)
                    };
                    let terminated = crate::inference::launch::terminate_process(
                        *child_pid,
                        cmd_name,
                        start_time_hint,
                    )
                    .await;
                    let exited = crate::inference::launch::wait_for_exit(*child_pid, 5000).await;
                    let force_killed = if exited {
                        false
                    } else {
                        crate::inference::launch::force_kill_process(
                            *child_pid,
                            cmd_name,
                            start_time_hint,
                        )
                        .await
                    };
                    let exited_after_force = if exited {
                        true
                    } else {
                        crate::inference::launch::wait_for_exit(*child_pid, 5000).await
                    };

                    if (terminated || force_killed) && exited_after_force {
                        summary.children_killed += 1;
                        summary.pidfiles_removed += 1;
                        pidfiles_to_remove.push(action.pidfile_path.clone());
                    } else {
                        tracing::warn!(
                            pid = *child_pid,
                            cmd_name,
                            terminated,
                            exited,
                            force_killed,
                            exited_after_force,
                            "failed to reap local stale child; keeping pidfile"
                        );
                    }
                }
                PendingActionKind::RemovePidfileOnly => {
                    summary.pidfiles_removed += 1;
                    pidfiles_to_remove.push(action.pidfile_path.clone());
                }
            }
        }

        tokio::task::spawn_blocking(move || {
            for path in pidfiles_to_remove {
                let _ = std::fs::remove_file(path);
            }
        })
        .await
        .context("own-runtime reap cleanup task panicked")?;

        Ok(summary)
    }

    fn scan_cross_runtime_orphans(
        root: &Path,
        my_runtime_dir: &Path,
    ) -> anyhow::Result<ScanResult> {
        let mut result = ScanResult::default();
        let entries = std::fs::read_dir(root)
            .with_context(|| format!("failed to read runtime root: {}", root.display()))?;

        for entry in entries.flatten() {
            let entry_path = entry.path();

            if entry_path == my_runtime_dir {
                continue;
            }

            if !entry_path.is_dir() {
                continue;
            }

            if super::is_locked(&entry_path.join("lock")) {
                result.summary.dirs_skipped_alive += 1;
                continue;
            }

            scan_pidfiles_dir(&entry_path.join("pidfiles"), false, &mut result);

            if std::fs::metadata(&entry_path)
                .ok()
                .and_then(|m| m.modified().ok())
                .map(|t| t.elapsed().unwrap_or_default().as_secs() > 3600)
                .unwrap_or(false)
            {
                result.stale_runtime_dirs.push(entry_path);
                result.summary.dirs_gc_d += 1;
            }

            result.summary.dirs_scanned += 1;
        }

        Ok(result)
    }

    fn scan_own_stale_pidfiles(my_dir: &Path) -> anyhow::Result<ScanResult> {
        let mut result = ScanResult::default();
        scan_pidfiles_dir(&my_dir.join("pidfiles"), true, &mut result);
        Ok(result)
    }

    fn scan_pidfiles_dir(pidfiles_dir: &Path, own_runtime: bool, result: &mut ScanResult) {
        let Ok(pidfiles) = std::fs::read_dir(pidfiles_dir) else {
            return;
        };

        for pidfile_entry in pidfiles.flatten() {
            let path = pidfile_entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("json") {
                continue;
            }

            let metadata = match super::PidfileMetadata::read(&path) {
                Ok(m) => m,
                Err(_) => {
                    result.summary.skipped_errors += 1;
                    continue;
                }
            };

            if metadata.child_pid == 0 || metadata.child_pid == 1 {
                tracing::warn!(
                    "skipping pidfile with invalid child_pid={}",
                    metadata.child_pid
                );
                result.pending_actions.push(PendingAction {
                    pidfile_path: path,
                    kind: PendingActionKind::RemovePidfileOnly,
                });
                continue;
            }

            let child_liveness = super::validate::process_liveness(metadata.child_pid);
            let child_comm_matches =
                super::validate::process_name_matches(metadata.child_pid, &metadata.cmd_name);
            let child_start_matches = if metadata.child_started_at_unix == 0 {
                true
            } else {
                super::validate::process_started_at_unix(metadata.child_pid)
                    .ok()
                    .flatten()
                    .map(|t| {
                        (t - metadata.child_started_at_unix).abs()
                            <= super::validate::START_TIME_TOLERANCE_SECS
                    })
                    .unwrap_or(false)
            };

            let action = if own_runtime {
                if child_liveness == super::validate::Liveness::Alive
                    && child_comm_matches
                    && child_start_matches
                {
                    Action::KillChildAndRemovePidfile
                } else {
                    Action::RemovePidfileOnly
                }
            } else {
                decide_action(
                    &metadata,
                    child_liveness,
                    child_comm_matches,
                    child_start_matches,
                    false,
                )
            };

            match action {
                Action::KillChildAndRemovePidfile => result.pending_actions.push(PendingAction {
                    pidfile_path: path,
                    kind: PendingActionKind::KillChildAndRemovePidfile {
                        child_pid: metadata.child_pid,
                        cmd_name: metadata.cmd_name,
                        child_started_at_unix: metadata.child_started_at_unix,
                    },
                }),
                Action::RemovePidfileOnly => result.pending_actions.push(PendingAction {
                    pidfile_path: path,
                    kind: PendingActionKind::RemovePidfileOnly,
                }),
                Action::Skip => {
                    result.summary.skipped_errors += 1;
                }
                Action::Keep => {}
            }
        }
    }
}

/// Snapshot of a co-located mesh-llm instance discovered via the runtime root.
#[derive(Debug, Clone, Serialize)]
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

        let pidfiles_dir = entry_path.join("pidfiles");
        if let Ok(pidfiles) = fs::read_dir(&pidfiles_dir) {
            for pidfile_entry in pidfiles.flatten() {
                let pidfile_path = pidfile_entry.path();
                if pidfile_path.extension().and_then(|ext| ext.to_str()) != Some("json") {
                    continue;
                }

                let metadata = match PidfileMetadata::read(&pidfile_path) {
                    Ok(metadata) => metadata,
                    Err(err) => {
                        tracing::warn!(
                            path = %pidfile_path.display(),
                            error = %err,
                            "failed to parse pidfile while collecting stop targets"
                        );
                        continue;
                    }
                };

                targets.push(RuntimeProcessTarget {
                    label: metadata.cmd_name.clone(),
                    pid: metadata.child_pid,
                    expected_comm: metadata.cmd_name,
                    // Treat 0 as "unknown" — start-time detection failed at pidfile
                    // creation, so omit the hint rather than passing a bogus timestamp.
                    expected_start_time: if metadata.child_started_at_unix == 0 {
                        None
                    } else {
                        Some(metadata.child_started_at_unix)
                    },
                });
            }
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
/// [`validate::process_liveness`]. Stale directories (dead owner) are skipped;
/// they will be garbage-collected by the reaper (T12).
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
    shared: Arc<tokio::sync::Mutex<Vec<LocalInstanceSnapshot>>>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(std::time::Duration::from_secs(5)).await;
            match scan_local_instances(&root, my_pid).await {
                Ok(instances) => {
                    // Short lock hold — never across an await point.
                    *shared.lock().await = instances;
                }
                Err(e) => {
                    tracing::warn!("local instance scan failed: {e}");
                }
            }
        }
    })
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

    #[test]
    #[serial]
    fn runtime_root_bails_when_unset() {
        // dirs 6.x only checks the HOME env var — no passwd fallback — so
        // removing HOME is sufficient to make dirs::home_dir() return None.
        let _g_mesh = EnvGuard::save_and_remove("MESH_LLM_RUNTIME_ROOT");
        let _g_xdg = EnvGuard::save_and_remove("XDG_RUNTIME_DIR");
        let _g_home = EnvGuard::save_and_remove("HOME");

        let result = runtime_root();
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
        assert!(
            rt.dir().join("pidfiles").exists(),
            "pidfiles subdir must be created"
        );
        assert!(
            rt.dir().join("logs").exists(),
            "logs subdir must be created"
        );
    }

    #[test]
    #[serial]
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
            !is_locked(&lock_path),
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
    fn is_locked_returns_false_after_drop() {
        let dir = tempdir().unwrap();
        let _g = EnvGuard::save_and_set("MESH_LLM_RUNTIME_ROOT", dir.path().to_str().unwrap());

        let rt = InstanceRuntime::acquire(1005).expect("acquire should succeed");
        let lock_path = rt.dir().join("lock");

        drop(rt);

        assert!(
            !is_locked(&lock_path),
            "is_locked must return false after InstanceRuntime is dropped"
        );
    }

    #[test]
    fn pidfile_roundtrip() {
        let metadata = PidfileMetadata {
            cmd_name: "llama-server".to_string(),
            child_pid: 1234,
            child_started_at_unix: 1700000000,
            owner_pid: 1000,
            owner_started_at_unix: 1699999900,
            argv_snippet: "llama-server -m model.gguf".to_string(),
            runtime_dir: PathBuf::from("/tmp/runtime"),
        };

        let dir = tempdir().unwrap();
        let pidfile_path = dir.path().join("test.json");

        metadata
            .write_atomic(&pidfile_path)
            .expect("write_atomic should succeed");

        let read_back = PidfileMetadata::read(&pidfile_path).expect("read should succeed");
        assert_eq!(read_back, metadata, "roundtrip must preserve metadata");
    }

    #[test]
    fn pidfile_atomic_write_no_partial() {
        let metadata = PidfileMetadata {
            cmd_name: "rpc-server".to_string(),
            child_pid: 5678,
            child_started_at_unix: 1700000100,
            owner_pid: 1000,
            owner_started_at_unix: 1699999900,
            argv_snippet: "rpc-server --port 9999".to_string(),
            runtime_dir: PathBuf::from("/tmp/runtime"),
        };

        let dir = tempdir().unwrap();
        let pidfile_path = dir.path().join("atomic.json");

        metadata
            .write_atomic(&pidfile_path)
            .expect("write_atomic should succeed");

        // Verify the main file exists and tmp file does not
        assert!(
            pidfile_path.exists(),
            "pidfile must exist after atomic write"
        );
        let tmp_path = pidfile_path.with_extension("json.tmp");
        assert!(
            !tmp_path.exists(),
            "tmp file must not exist after successful atomic write"
        );
    }

    #[test]
    fn pidfile_corrupt_returns_err_not_panic() {
        let dir = tempdir().unwrap();
        let pidfile_path = dir.path().join("corrupt.json");

        // Write invalid JSON
        fs::write(&pidfile_path, "{ invalid json }").expect("write should succeed");

        let result = PidfileMetadata::read(&pidfile_path);
        assert!(
            result.is_err(),
            "read must return Err for corrupt pidfile, not panic"
        );

        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("corrupt pidfile"),
            "error message must mention corrupt pidfile"
        );
    }

    #[test]
    fn pidfile_guard_removes_on_drop() {
        let dir = tempdir().unwrap();
        let pidfile_path = dir.path().join("guard_test.json");

        // Create a dummy pidfile
        fs::write(&pidfile_path, "{}").expect("write should succeed");
        assert!(
            pidfile_path.exists(),
            "pidfile must exist before guard drop"
        );

        {
            let _guard = PidfileGuard::new(pidfile_path.clone());
            assert!(
                pidfile_path.exists(),
                "pidfile must exist while guard is held"
            );
        }

        assert!(
            !pidfile_path.exists(),
            "pidfile must be removed after guard is dropped"
        );
    }

    #[test]
    fn pidfile_guard_survives_missing_file_on_drop() {
        let dir = tempdir().unwrap();
        let pidfile_path = dir.path().join("missing.json");

        // Guard for a file that doesn't exist
        let _guard = PidfileGuard::new(pidfile_path.clone());

        // Drop should not panic even though the file is missing
        drop(_guard);

        // Test passes if we reach here without panicking
        assert!(!pidfile_path.exists(), "file should still not exist");
    }

    #[test]
    fn cap_argv_truncates_at_byte_boundary() {
        let argv = vec![
            "llama-server".to_string(),
            "-m".to_string(),
            "model.gguf".to_string(),
            "--port".to_string(),
            "9999".to_string(),
        ];

        let capped = PidfileMetadata::cap_argv(&argv, 20);
        assert!(
            capped.ends_with("…"),
            "truncated argv must end with ellipsis"
        );
        assert!(
            capped.is_char_boundary(capped.len()),
            "capped argv must end at a valid UTF-8 boundary"
        );
    }

    #[test]
    fn cap_argv_preserves_utf8_multibyte_boundary() {
        // Create argv with multibyte UTF-8 characters
        let argv = vec![
            "test".to_string(),
            "café".to_string(), // é is 2 bytes in UTF-8
        ];

        let capped = PidfileMetadata::cap_argv(&argv, 8);
        // Should truncate before the multibyte character, not in the middle of it
        assert!(
            capped.is_char_boundary(capped.len()),
            "truncated argv must end at a valid UTF-8 boundary"
        );
        assert!(
            capped.len() <= 8,
            "truncated argv must respect max_bytes including the ellipsis"
        );
    }

    #[test]
    fn cap_argv_handles_zero_max_bytes() {
        let argv = vec!["llama-server".to_string(), "--model".to_string()];

        let capped = PidfileMetadata::cap_argv(&argv, 0);
        assert_eq!(capped, "", "zero-byte budget should yield an empty string");
    }

    #[test]
    fn cap_argv_preserves_small_byte_budgets_without_ellipsis() {
        let argv = vec!["abc".to_string()];

        assert_eq!(PidfileMetadata::cap_argv(&argv, 1), "a");
        assert_eq!(PidfileMetadata::cap_argv(&argv, 2), "ab");
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
    #[serial]
    fn write_pidfile_creates_file_and_guard() {
        let dir = tempdir().unwrap();
        let _g = EnvGuard::save_and_set("MESH_LLM_RUNTIME_ROOT", dir.path().to_str().unwrap());

        let rt = InstanceRuntime::acquire(2000).expect("acquire should succeed");

        let metadata = PidfileMetadata {
            cmd_name: "test-server".to_string(),
            child_pid: 9999,
            child_started_at_unix: 1700000000,
            owner_pid: 2000,
            owner_started_at_unix: 1699999900,
            argv_snippet: "test-server".to_string(),
            runtime_dir: rt.dir().to_path_buf(),
        };

        let pidfile_path = rt.pidfile_path("test-server");
        assert!(
            !pidfile_path.exists(),
            "pidfile must not exist before write_pidfile"
        );

        let _guard = rt
            .write_pidfile("test-server", &metadata)
            .expect("write_pidfile should succeed");

        assert!(
            pidfile_path.exists(),
            "pidfile must exist after write_pidfile"
        );

        let read_back = PidfileMetadata::read(&pidfile_path).expect("read should succeed");
        assert_eq!(read_back, metadata, "written metadata must match");

        drop(_guard);

        assert!(
            !pidfile_path.exists(),
            "pidfile must be removed after guard is dropped"
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
    fn validate_pid_matches_rejects_wrong_comm() {
        let pid = std::process::id();
        assert!(
            !validate::validate_pid_matches(pid, "definitely-not-this-comm-string", 0),
            "wrong comm must cause validate_pid_matches to return false"
        );
    }

    #[test]
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

    fn dummy_metadata() -> PidfileMetadata {
        PidfileMetadata {
            cmd_name: "test-server".to_string(),
            child_pid: 1234,
            child_started_at_unix: 1700000000,
            owner_pid: 5678,
            owner_started_at_unix: 1699999900,
            argv_snippet: "test-server --port 9999".to_string(),
            runtime_dir: PathBuf::from("/tmp/runtime"),
        }
    }

    #[test]
    fn decide_keep_when_owner_locked() {
        let meta = dummy_metadata();
        let action = reap::decide_action(&meta, validate::Liveness::Alive, true, true, true);
        assert_eq!(action, reap::Action::Keep);
    }

    #[test]
    fn decide_remove_pidfile_when_child_dead() {
        let meta = dummy_metadata();
        let action = reap::decide_action(&meta, validate::Liveness::Dead, false, false, false);
        assert_eq!(action, reap::Action::RemovePidfileOnly);
    }

    #[test]
    fn decide_kill_and_remove_when_child_alive_and_matches() {
        let meta = dummy_metadata();
        let action = reap::decide_action(&meta, validate::Liveness::Alive, true, true, false);
        assert_eq!(action, reap::Action::KillChildAndRemovePidfile);
    }

    #[test]
    fn decide_remove_only_when_child_pid_recycled_comm_mismatch() {
        let meta = dummy_metadata();
        let action = reap::decide_action(&meta, validate::Liveness::Alive, false, true, false);
        assert_eq!(action, reap::Action::RemovePidfileOnly);
    }

    #[test]
    fn decide_remove_only_when_child_start_time_drifted() {
        let meta = dummy_metadata();
        let action = reap::decide_action(&meta, validate::Liveness::Alive, true, false, false);
        assert_eq!(action, reap::Action::RemovePidfileOnly);
    }

    #[test]
    fn decide_skip_when_child_liveness_unknown() {
        let meta = dummy_metadata();
        let action = reap::decide_action(&meta, validate::Liveness::Unknown, false, false, false);
        assert_eq!(action, reap::Action::Skip);
    }

    #[test]
    fn decide_keep_trumps_everything_when_owner_locked() {
        let meta = dummy_metadata();
        let action = reap::decide_action(&meta, validate::Liveness::Dead, false, false, true);
        assert_eq!(action, reap::Action::Keep);
    }

    #[tokio::test]
    #[serial]
    async fn reap_skips_own_dir() {
        let root = tempdir().unwrap();
        let dir_a = root.path().join("1001");
        let dir_b = root.path().join("1002");
        fs::create_dir_all(dir_a.join("pidfiles")).unwrap();
        fs::create_dir_all(dir_b.join("pidfiles")).unwrap();

        let summary = reap::reap_cross_runtime_orphans(root.path(), &dir_a)
            .await
            .unwrap();

        assert_eq!(
            summary.dirs_scanned, 1,
            "only the peer dir should be scanned, not own dir"
        );
        assert_eq!(summary.dirs_skipped_alive, 0);
    }

    #[tokio::test]
    #[serial]
    async fn reap_skips_alive_owner() {
        let root = tempdir().unwrap();
        let _g = EnvGuard::save_and_set("MESH_LLM_RUNTIME_ROOT", root.path().to_str().unwrap());

        let rt = InstanceRuntime::acquire(9100).expect("acquire should succeed");

        let my_dir = root.path().join("99999");
        let summary = reap::reap_cross_runtime_orphans(root.path(), &my_dir)
            .await
            .unwrap();

        assert_eq!(
            summary.dirs_skipped_alive, 1,
            "live-owner dir must be skipped"
        );
        assert_eq!(summary.dirs_scanned, 0);

        drop(rt);
    }

    #[tokio::test]
    #[serial]
    async fn reap_removes_dead_child_pidfile() {
        let root = tempdir().unwrap();
        let peer_dir = root.path().join("50001");
        let pidfiles_dir = peer_dir.join("pidfiles");
        fs::create_dir_all(&pidfiles_dir).unwrap();

        let metadata = PidfileMetadata {
            cmd_name: "llama-server".to_string(),
            child_pid: 999999,
            child_started_at_unix: 1700000000,
            owner_pid: 50001,
            owner_started_at_unix: 1699999900,
            argv_snippet: "llama-server".to_string(),
            runtime_dir: peer_dir.clone(),
        };
        let pidfile_path = pidfiles_dir.join("llama-server.json");
        metadata.write_atomic(&pidfile_path).unwrap();

        let my_dir = root.path().join("99999");
        let summary = reap::reap_cross_runtime_orphans(root.path(), &my_dir)
            .await
            .unwrap();

        assert_eq!(
            summary.pidfiles_removed, 1,
            "dead-child pidfile must be removed"
        );
        assert!(!pidfile_path.exists(), "pidfile must be gone from disk");
        assert_eq!(summary.children_killed, 0, "no kill for a dead PID");
    }

    #[tokio::test]
    #[serial]
    async fn reap_preserves_external_pid_via_comm_mismatch() {
        let root = tempdir().unwrap();
        let peer_dir = root.path().join("60001");
        let pidfiles_dir = peer_dir.join("pidfiles");
        fs::create_dir_all(&pidfiles_dir).unwrap();

        let self_pid = std::process::id();
        let metadata = PidfileMetadata {
            cmd_name: "definitely-not-this-binary".to_string(),
            child_pid: self_pid,
            child_started_at_unix: 1700000000,
            owner_pid: 60001,
            owner_started_at_unix: 1699999900,
            argv_snippet: "fake".to_string(),
            runtime_dir: peer_dir.clone(),
        };
        let pidfile_path = pidfiles_dir.join("fake-server.json");
        metadata.write_atomic(&pidfile_path).unwrap();

        let my_dir = root.path().join("99999");
        let summary = reap::reap_cross_runtime_orphans(root.path(), &my_dir)
            .await
            .unwrap();

        assert_eq!(
            summary.pidfiles_removed, 1,
            "pidfile removed on comm mismatch (RemovePidfileOnly path)"
        );
        assert!(!pidfile_path.exists(), "pidfile must be removed from disk");
        assert_eq!(
            summary.children_killed, 0,
            "process must not be killed on comm mismatch"
        );
    }

    #[tokio::test]
    #[serial]
    async fn own_drain_removes_dead_pidfile() {
        let dir = tempdir().unwrap();
        let my_dir = dir.path().join("my_runtime");
        let pidfiles_dir = my_dir.join("pidfiles");
        fs::create_dir_all(&pidfiles_dir).unwrap();

        let metadata = PidfileMetadata {
            cmd_name: "llama-server".to_string(),
            child_pid: 999999,
            child_started_at_unix: 1700000000,
            owner_pid: 1234,
            owner_started_at_unix: 1699999900,
            argv_snippet: "llama-server".to_string(),
            runtime_dir: my_dir.clone(),
        };
        let pidfile_path = pidfiles_dir.join("llama-server.json");
        metadata.write_atomic(&pidfile_path).unwrap();

        let summary = reap::reap_own_stale_pidfiles(&my_dir).await.unwrap();

        assert_eq!(summary.pidfiles_removed, 1, "dead pidfile must be removed");
        assert_eq!(summary.children_killed, 0, "dead PID must not be killed");
        assert!(!pidfile_path.exists(), "pidfile must be gone from disk");
    }

    #[tokio::test]
    #[serial]
    async fn cross_runtime_reaper_ignores_non_directories() {
        let root = tempdir().unwrap();
        let my_dir = root.path().join("self");
        fs::create_dir_all(&my_dir).unwrap();

        let stray_file = root.path().join("README.txt");
        fs::write(&stray_file, "not a runtime directory").unwrap();

        let result = reap::reap_cross_runtime_orphans(root.path(), &my_dir)
            .await
            .unwrap();

        assert_eq!(result.dirs_scanned, 0);
        assert_eq!(result.dirs_gc_d, 0);
        assert!(
            stray_file.exists(),
            "non-directory entries must not be collected"
        );
    }

    #[tokio::test]
    #[serial]
    async fn own_drain_removes_mismatched_pidfile_defensively() {
        let dir = tempdir().unwrap();
        let my_dir = dir.path().join("my_runtime");
        let pidfiles_dir = my_dir.join("pidfiles");
        fs::create_dir_all(&pidfiles_dir).unwrap();

        let self_pid = std::process::id();
        let metadata = PidfileMetadata {
            cmd_name: "definitely-not-this-binary".to_string(),
            child_pid: self_pid,
            child_started_at_unix: 1700000000,
            owner_pid: 1234,
            owner_started_at_unix: 1699999900,
            argv_snippet: "fake".to_string(),
            runtime_dir: my_dir.clone(),
        };
        let pidfile_path = pidfiles_dir.join("fake-server.json");
        metadata.write_atomic(&pidfile_path).unwrap();

        let summary = reap::reap_own_stale_pidfiles(&my_dir).await.unwrap();

        assert_eq!(
            summary.pidfiles_removed, 1,
            "mismatched pidfile must be removed defensively"
        );
        assert_eq!(
            summary.children_killed, 0,
            "bystander process must not be killed on comm mismatch"
        );
        assert!(!pidfile_path.exists(), "pidfile must be gone from disk");
    }

    #[tokio::test]
    #[serial]
    async fn own_drain_returns_empty_when_no_pidfiles_dir() {
        let dir = tempdir().unwrap();
        let my_dir = dir.path().join("my_runtime_no_pidfiles");
        fs::create_dir_all(&my_dir).unwrap();

        let summary = reap::reap_own_stale_pidfiles(&my_dir).await.unwrap();

        assert_eq!(
            summary.pidfiles_removed, 0,
            "no pidfiles dir — nothing to remove"
        );
        assert_eq!(
            summary.children_killed, 0,
            "no pidfiles dir — nothing to kill"
        );
        assert_eq!(summary.skipped_errors, 0, "no pidfiles dir — no errors");
    }
}
