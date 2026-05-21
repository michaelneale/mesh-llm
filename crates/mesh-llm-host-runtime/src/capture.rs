use anyhow::{Context, Result};
use serde_json::{json, Value};
use std::fs::{self, File, OpenOptions};
use std::io::Write;
#[cfg(unix)]
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::sync::{
    mpsc::{self, SyncSender, TrySendError},
    Arc,
};

pub(crate) const SWARM_CAPTURE_ENV: &str = "MESH_LLM_SWARM_CAPTURE";
pub(crate) const SWARM_CAPTURE_FILE: &str = "swarm-capture.jsonl";
const SWARM_CAPTURE_QUEUE_CAPACITY: usize = 8192;

#[derive(Clone, Debug)]
pub(crate) struct SwarmCaptureRecorder {
    writer: SyncSender<Vec<u8>>,
    path: Arc<PathBuf>,
}

impl SwarmCaptureRecorder {
    pub(crate) fn from_cli_or_env(cli_dir: Option<&Path>) -> Result<Option<Self>> {
        if let Some(dir) = cli_dir {
            return Self::new(dir).map(Some);
        }

        let Some(raw_dir) = std::env::var_os(SWARM_CAPTURE_ENV) else {
            return Ok(None);
        };
        if raw_dir.is_empty() {
            return Ok(None);
        }

        Self::new(PathBuf::from(raw_dir)).map(Some)
    }

    pub(crate) fn new(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        prepare_capture_dir(dir)?;

        let path = dir.join(SWARM_CAPTURE_FILE);
        let mut options = OpenOptions::new();
        options.create(true).append(true);
        #[cfg(unix)]
        options.mode(0o600).custom_flags(libc::O_NOFOLLOW);
        let file = options
            .open(&path)
            .with_context(|| format!("open swarm capture log {}", path.to_string_lossy()))?;
        set_private_file_permissions(&file);

        let (writer, reader) = mpsc::sync_channel::<Vec<u8>>(SWARM_CAPTURE_QUEUE_CAPACITY);
        std::thread::Builder::new()
            .name("mesh-swarm-capture-writer".to_string())
            .spawn(move || run_writer(file, reader))
            .context("start swarm capture writer thread")?;

        Ok(Self {
            writer,
            path: Arc::new(path),
        })
    }

    pub(crate) fn path(&self) -> &Path {
        &self.path
    }

    pub(crate) fn record_event(&self, event: &str, fields: Value) {
        let Some(line) = serialize_event_record(event, fields) else {
            return;
        };

        queue_event_record(&self.writer, event, line);
    }
}

fn serialize_event_record(event: &str, fields: Value) -> Option<Vec<u8>> {
    let record = json!({
        "ts_unix_ms": current_time_unix_ms(),
        "event": event,
        "fields": fields,
    });

    match serde_json::to_vec(&record) {
        Ok(mut line) => {
            line.push(b'\n');
            Some(line)
        }
        Err(_) => {
            tracing::debug!(event, "failed to serialize swarm capture event");
            None
        }
    }
}

fn queue_event_record(writer: &SyncSender<Vec<u8>>, event: &str, line: Vec<u8>) {
    match writer.try_send(line) {
        Ok(()) => {}
        Err(TrySendError::Full(_)) => {
            tracing::debug!(event, "swarm capture queue full; dropping event");
        }
        Err(TrySendError::Disconnected(_)) => {
            tracing::debug!(event, "swarm capture writer stopped; dropping event");
        }
    }
}

pub(crate) fn http_path_without_query(path: &str) -> &str {
    path.split('?').next().unwrap_or(path)
}

fn current_time_unix_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

fn prepare_capture_dir(path: &Path) -> Result<()> {
    match fs::symlink_metadata(path) {
        Ok(metadata) => validate_existing_capture_dir(path, &metadata),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            fs::create_dir_all(path).with_context(|| {
                format!("create swarm capture directory {}", path.to_string_lossy())
            })?;
            let metadata = fs::symlink_metadata(path).with_context(|| {
                format!("inspect swarm capture directory {}", path.to_string_lossy())
            })?;
            validate_capture_dir_type(path, &metadata)?;
            set_private_dir_permissions(path)?;
            Ok(())
        }
        Err(error) => Err(error)
            .with_context(|| format!("inspect swarm capture directory {}", path.to_string_lossy())),
    }
}

fn validate_existing_capture_dir(path: &Path, metadata: &fs::Metadata) -> Result<()> {
    validate_capture_dir_type(path, metadata)?;
    ensure_existing_capture_dir_private(path, metadata)
}

fn validate_capture_dir_type(path: &Path, metadata: &fs::Metadata) -> Result<()> {
    if metadata.file_type().is_symlink() {
        anyhow::bail!(
            "swarm capture directory {} must not be a symlink",
            path.to_string_lossy()
        );
    }
    if !metadata.is_dir() {
        anyhow::bail!(
            "swarm capture path {} is not a directory",
            path.to_string_lossy()
        );
    }
    Ok(())
}

fn ensure_existing_capture_dir_private(path: &Path, metadata: &fs::Metadata) -> Result<()> {
    #[cfg(unix)]
    {
        let mode = metadata.permissions().mode() & 0o777;
        if mode & 0o077 != 0 {
            anyhow::bail!(
                "existing swarm capture directory {} must be private (mode 0700 or stricter); current mode is {:03o}",
                path.to_string_lossy(),
                mode
            );
        }
    }

    #[cfg(not(unix))]
    {
        let _ = (path, metadata);
    }

    Ok(())
}

fn set_private_dir_permissions(path: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        let dir = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW)
            .open(path)
            .with_context(|| {
                format!(
                    "open swarm capture directory {} without following symlinks",
                    path.to_string_lossy()
                )
            })?;
        dir.set_permissions(fs::Permissions::from_mode(0o700))
            .with_context(|| {
                format!(
                    "set private permissions on swarm capture directory {}",
                    path.to_string_lossy()
                )
            })?;
    }

    #[cfg(not(unix))]
    {
        let _ = path;
    }

    Ok(())
}

fn set_private_file_permissions(file: &File) {
    #[cfg(unix)]
    {
        let _ = file.set_permissions(fs::Permissions::from_mode(0o600));
    }

    #[cfg(not(unix))]
    {
        let _ = file;
    }
}

fn run_writer(mut file: File, reader: mpsc::Receiver<Vec<u8>>) {
    for line in reader {
        if let Err(error) = file.write_all(&line).and_then(|_| file.flush()) {
            tracing::debug!(%error, "failed to append swarm capture event");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;
    use std::io::{BufRead, BufReader};

    struct EnvVarGuard {
        key: &'static str,
        previous: Option<std::ffi::OsString>,
    }

    impl EnvVarGuard {
        fn capture(key: &'static str) -> Self {
            Self {
                key,
                previous: std::env::var_os(key),
            }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            match &self.previous {
                Some(value) => std::env::set_var(self.key, value),
                None => std::env::remove_var(self.key),
            }
        }
    }

    #[cfg(unix)]
    use std::os::unix::fs::{symlink, PermissionsExt};

    #[test]
    fn recorder_appends_jsonl_events() {
        let temp = tempfile::tempdir().expect("tempdir");
        let capture_dir = temp.path().join("capture");
        let recorder = SwarmCaptureRecorder::new(&capture_dir).expect("recorder");

        recorder.record_event("peer_seen", json!({"peer_id_short": "abc123"}));

        wait_for_capture_bytes(recorder.path());
        let file = File::open(recorder.path()).expect("open capture log");
        let lines = BufReader::new(file)
            .lines()
            .collect::<Result<Vec<_>, _>>()
            .expect("read lines");
        assert_eq!(lines.len(), 1);

        let parsed: Value = serde_json::from_str(&lines[0]).expect("json line");
        assert_eq!(parsed["event"], "peer_seen");
        assert_eq!(parsed["fields"]["peer_id_short"], "abc123");
        assert!(parsed["ts_unix_ms"].as_u64().is_some());
    }

    #[test]
    fn http_path_without_query_omits_invite_like_values() {
        let path = http_path_without_query("/api/discover?invite_token=secret-token&foo=bar");

        assert_eq!(path, "/api/discover");
        assert!(!path.contains("secret-token"));
    }

    #[test]
    #[serial]
    fn empty_env_disables_capture() {
        let _env_guard = EnvVarGuard::capture(SWARM_CAPTURE_ENV);
        std::env::set_var(SWARM_CAPTURE_ENV, "");

        let recorder = SwarmCaptureRecorder::from_cli_or_env(None).expect("env resolution");

        assert!(recorder.is_none());
    }

    #[test]
    fn recorder_creates_nested_directory() {
        let temp = tempfile::tempdir().expect("tempdir");
        let nested = temp.path().join("deep").join("nested").join("capture");

        let recorder = SwarmCaptureRecorder::new(&nested).expect("nested recorder");
        recorder.record_event("test_event", json!({"k": "v"}));

        wait_for_capture_bytes(recorder.path());
        assert!(nested.join(SWARM_CAPTURE_FILE).exists());
    }

    #[test]
    fn multiple_events_produce_separate_lines() {
        let temp = tempfile::tempdir().expect("tempdir");
        let capture_dir = temp.path().join("capture");
        let recorder = SwarmCaptureRecorder::new(&capture_dir).expect("recorder");

        for i in 0..10 {
            recorder.record_event("batch", json!({"seq": i}));
        }

        wait_for_lines(recorder.path(), 10);
        let file = File::open(recorder.path()).expect("open");
        let lines: Vec<_> = BufReader::new(file)
            .lines()
            .collect::<Result<Vec<_>, _>>()
            .expect("lines");
        assert_eq!(lines.len(), 10);
        for (i, line) in lines.iter().enumerate() {
            let parsed: Value = serde_json::from_str(line).expect("json");
            assert_eq!(parsed["fields"]["seq"], i as u64);
        }
    }

    #[test]
    #[serial]
    fn cli_env_precedence_cli_wins() {
        let temp = tempfile::tempdir().expect("tempdir");
        let env_dir = temp.path().join("env_dir");
        let cli_dir = temp.path().join("cli_dir");
        let _env_guard = EnvVarGuard::capture(SWARM_CAPTURE_ENV);
        std::env::set_var(SWARM_CAPTURE_ENV, env_dir.to_str().unwrap());

        let recorder =
            SwarmCaptureRecorder::from_cli_or_env(Some(&cli_dir)).expect("cli takes precedence");

        assert!(recorder.is_some());
        assert!(recorder.unwrap().path().starts_with(&cli_dir));
    }

    #[cfg(unix)]
    #[test]
    fn existing_permissive_directory_is_rejected_and_preserved() {
        let temp = tempfile::tempdir().expect("tempdir");
        let dir = temp.path().join("shared-capture");
        fs::create_dir(&dir).expect("create dir");
        fs::set_permissions(&dir, fs::Permissions::from_mode(0o755)).expect("set permissive mode");

        let error = SwarmCaptureRecorder::new(&dir).expect_err("permissive dir rejected");

        assert!(
            error.to_string().contains("must be private"),
            "unexpected error: {error:#}"
        );
        let mode = fs::symlink_metadata(&dir)
            .expect("metadata")
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(mode, 0o755);
    }

    #[cfg(unix)]
    #[test]
    fn existing_capture_directory_symlink_is_rejected() {
        let temp = tempfile::tempdir().expect("tempdir");
        let target = temp.path().join("target-capture");
        fs::create_dir(&target).expect("create target dir");
        fs::set_permissions(&target, fs::Permissions::from_mode(0o700)).expect("set private mode");
        let dir = temp.path().join("capture-link");
        symlink(&target, &dir).expect("symlink capture dir");

        let error = SwarmCaptureRecorder::new(&dir).expect_err("symlink dir rejected");

        assert!(
            error.to_string().contains("must not be a symlink"),
            "unexpected error: {error:#}"
        );
        assert!(!target.join(SWARM_CAPTURE_FILE).exists());
    }

    #[cfg(unix)]
    #[test]
    fn existing_capture_log_symlink_is_rejected() {
        let temp = tempfile::tempdir().expect("tempdir");
        let dir = temp.path().join("capture");
        fs::create_dir(&dir).expect("create capture dir");
        fs::set_permissions(&dir, fs::Permissions::from_mode(0o700)).expect("set private mode");
        let target = temp.path().join("target.jsonl");
        fs::write(&target, b"before\n").expect("write target");
        symlink(&target, dir.join(SWARM_CAPTURE_FILE)).expect("symlink capture log");

        let error = SwarmCaptureRecorder::new(&dir).expect_err("symlink log rejected");

        assert!(
            error.to_string().contains("open swarm capture log"),
            "unexpected error: {error:#}"
        );
        assert_eq!(
            fs::read_to_string(&target).expect("read target"),
            "before\n"
        );
    }

    fn wait_for_capture_bytes(path: &Path) {
        for _ in 0..100 {
            if path
                .metadata()
                .map(|metadata| metadata.len() > 0)
                .unwrap_or(false)
            {
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
    }

    fn wait_for_lines(path: &Path, expected: usize) {
        for _ in 0..200 {
            if let Ok(file) = File::open(path) {
                let count = BufReader::new(file).lines().count();
                if count >= expected {
                    return;
                }
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
    }
}
