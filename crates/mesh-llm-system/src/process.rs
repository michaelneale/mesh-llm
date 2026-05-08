/// Tolerance (in seconds) when comparing a recorded start time against the
/// live process start time. A difference of up to this many seconds is treated
/// as the same process.
pub const START_TIME_TOLERANCE_SECS: i64 = 2;

/// Liveness state inferred from whether the process comm is readable.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Liveness {
    /// Process is alive because comm was readable.
    Alive,
    /// Process is gone because the PID was not found.
    Dead,
    /// Liveness could not be determined.
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

        let rparen = content
            .rfind(')')
            .ok_or_else(|| anyhow::anyhow!("malformed /proc/{pid}/stat: no closing ')' found"))?;
        let after_comm = content.get(rparen + 2..).unwrap_or("");
        let fields: Vec<&str> = after_comm.split_whitespace().collect();

        let starttime_ticks: u64 = fields
            .get(19)
            .ok_or_else(|| anyhow::anyhow!("starttime field missing in /proc/{pid}/stat"))?
            .parse()
            .map_err(|e| anyhow::anyhow!("failed to parse starttime in /proc/{pid}/stat: {e}"))?;

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
        let basename = std::path::Path::new(&s)
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or(s);
        Ok(Some(basename))
    }

    pub fn process_started_at_unix(pid: u32) -> anyhow::Result<Option<i64>> {
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

        let parts: Vec<&str> = s.split_whitespace().collect();
        if parts.len() != 5 {
            return Ok(None);
        }

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

/// Read the command name of the given process.
pub fn process_comm(pid: u32) -> anyhow::Result<Option<String>> {
    platform::process_comm(pid)
}

/// Read the Unix start time of the given process.
pub fn process_started_at_unix(pid: u32) -> anyhow::Result<Option<i64>> {
    platform::process_started_at_unix(pid)
}

/// Read the executable basename for the given process when available.
pub fn process_executable_name(pid: u32) -> anyhow::Result<Option<String>> {
    platform::process_executable_name(pid)
}

/// Determine liveness of a process by attempting to read its comm.
pub fn process_liveness(pid: u32) -> Liveness {
    match process_comm(pid) {
        Ok(Some(_)) => Liveness::Alive,
        Ok(None) => Liveness::Dead,
        Err(_) => Liveness::Unknown,
    }
}

/// Returns true iff the live process name matches the expected spawned binary.
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

/// Returns true iff the live process matches the expected name and start time.
#[cfg(not(windows))]
pub fn validate_pid_matches(pid: u32, expected_comm: &str, expected_started_at_unix: i64) -> bool {
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
