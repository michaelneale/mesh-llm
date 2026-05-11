enum RemoteSyncEvent {
    HostStarted {
        host: String,
        stages: Vec<String>,
    },
    StepStarted {
        host: String,
        label: String,
    },
    StepProgress {
        host: String,
        label: String,
        detail: String,
        elapsed: Duration,
    },
    StepFinished {
        host: String,
        label: String,
        detail: String,
        elapsed: Duration,
    },
    HostFinished {
        host: String,
        elapsed: Duration,
    },
    HostFailed {
        host: String,
        error: String,
    },
}

fn rsync_remote_stage_inputs(
    args: &PromptArgs,
    stages: &[LocalStage],
    model_package_dir: &Path,
    hf_package_ref: bool,
) -> Result<()> {
    let mut stages_by_host: BTreeMap<String, Vec<&LocalStage>> = BTreeMap::new();
    for stage in stages {
        let remote = stage
            .remote
            .as_ref()
            .context("remote stage missing placement")?;
        stages_by_host
            .entry(remote.host.clone())
            .or_default()
            .push(stage);
    }

    eprintln!(
        "remote sync: preparing inputs for {} stages on {} hosts in parallel",
        stages.len(),
        stages_by_host.len()
    );
    let (tx, rx) = mpsc::channel::<RemoteSyncEvent>();
    thread::scope(|scope| -> Result<()> {
        let mut handles = Vec::new();
        for (host, host_stages) in stages_by_host {
            let tx = tx.clone();
            handles.push(scope.spawn(move || {
                let started = Instant::now();
                send_remote_sync_event(
                    &tx,
                    RemoteSyncEvent::HostStarted {
                        host: host.clone(),
                        stages: host_stages
                            .iter()
                            .map(|stage| {
                                format!(
                                    "{}:{}..{}",
                                    stage.stage_id, stage.layer_start, stage.layer_end
                                )
                            })
                            .collect(),
                    },
                );
                let result = sync_remote_host_inputs(
                    args,
                    &host,
                    &host_stages,
                    model_package_dir,
                    hf_package_ref,
                    &tx,
                );
                match &result {
                    Ok(()) => send_remote_sync_event(
                        &tx,
                        RemoteSyncEvent::HostFinished {
                            host,
                            elapsed: started.elapsed(),
                        },
                    ),
                    Err(error) => send_remote_sync_event(
                        &tx,
                        RemoteSyncEvent::HostFailed {
                            host,
                            error: format!("{error:#}"),
                        },
                    ),
                }
                result
            }));
        }
        drop(tx);

        for event in rx {
            log_remote_sync_event(event);
        }

        for handle in handles {
            handle
                .join()
                .map_err(|_| anyhow!("remote sync worker panicked"))??;
        }
        Ok(())
    })?;
    eprintln!("remote sync: all remote inputs are ready");
    Ok(())
}

fn sync_remote_host_inputs(
    args: &PromptArgs,
    host: &str,
    stages: &[&LocalStage],
    model_package_dir: &Path,
    hf_package_ref: bool,
    tx: &mpsc::Sender<RemoteSyncEvent>,
) -> Result<()> {
    let first_remote = stages
        .first()
        .and_then(|stage| stage.remote.as_ref())
        .context("remote host has no stages")?;
    let mut mkdir_paths = BTreeSet::new();
    mkdir_paths.insert(first_remote.stage_dir.clone());
    mkdir_paths.insert(
        Path::new(&first_remote.stage_server_bin)
            .parent()
            .and_then(Path::to_str)
            .context("remote stage binary path has no parent")?
            .to_string(),
    );
    mkdir_paths.insert(
        Path::new(&first_remote.model_path)
            .parent()
            .and_then(Path::to_str)
            .context("remote model package path has no parent")?
            .to_string(),
    );
    for stage in stages.iter().skip(1) {
        let remote = stage
            .remote
            .as_ref()
            .context("remote stage missing placement")?;
        mkdir_paths.insert(remote.stage_dir.clone());
    }

    let label = format!("create {} remote directories", mkdir_paths.len());
    let started = Instant::now();
    send_remote_sync_event(
        tx,
        RemoteSyncEvent::StepStarted {
            host: host.to_string(),
            label: label.clone(),
        },
    );
    let mkdir_command = format!(
        "mkdir -p {}",
        mkdir_paths
            .iter()
            .map(|path| shell_quote(path))
            .collect::<Vec<_>>()
            .join(" ")
    );
    run_status(
        Command::new("ssh").arg("-n").arg(host).arg(mkdir_command),
        &format!("create remote dirs on {host}"),
    )?;
    send_remote_sync_event(
        tx,
        RemoteSyncEvent::StepFinished {
            host: host.to_string(),
            label,
            detail: "done".to_string(),
            elapsed: started.elapsed(),
        },
    );

    rsync_to_host_cached(
        &args.stage_server_bin,
        host,
        &first_remote.stage_server_bin,
        &args.remote_root,
        tx,
    )?;
    if !hf_package_ref {
        rsync_dir_to_host_cached(model_package_dir, host, &first_remote.model_path, tx)?;
    }

    for stage in stages {
        let remote = stage
            .remote
            .as_ref()
            .context("remote stage missing placement")?;
        rsync_to_host_with_progress(
            &stage.config_path,
            host,
            &remote.config_path,
            &format!("{} config", stage.stage_id),
            tx,
        )?;
    }

    Ok(())
}

fn send_remote_sync_event(tx: &mpsc::Sender<RemoteSyncEvent>, event: RemoteSyncEvent) {
    let _ = tx.send(event);
}

fn log_remote_sync_event(event: RemoteSyncEvent) {
    match event {
        RemoteSyncEvent::HostStarted { host, stages } => {
            eprintln!("remote sync [{host}]: start {}", stages.join(", "));
        }
        RemoteSyncEvent::StepStarted { host, label } => {
            eprintln!("remote sync [{host}]: {label} ...");
        }
        RemoteSyncEvent::StepProgress {
            host,
            label,
            detail,
            elapsed,
        } => {
            eprintln!(
                "remote sync [{host}]: {label} still running ({:.1}s) {}",
                elapsed.as_secs_f64(),
                truncate_label(&detail, 96)
            );
        }
        RemoteSyncEvent::StepFinished {
            host,
            label,
            detail,
            elapsed,
        } => {
            eprintln!(
                "remote sync [{host}]: {label} {} ({:.1}s)",
                detail,
                elapsed.as_secs_f64()
            );
        }
        RemoteSyncEvent::HostFinished { host, elapsed } => {
            eprintln!(
                "remote sync [{host}]: complete ({:.1}s)",
                elapsed.as_secs_f64()
            );
        }
        RemoteSyncEvent::HostFailed { host, error } => {
            eprintln!("remote sync [{host}]: failed: {error}");
        }
    }
}

fn start_remote_stages(
    args: &PromptArgs,
    stages: &[LocalStage],
    metrics_otlp_url: &str,
    children: &mut Vec<ChildGuard>,
) -> Result<()> {
    for stage in stages {
        stop_remote_stage_listener(stage)?;
    }

    for stage in stages.iter().rev() {
        let remote = stage
            .remote
            .as_ref()
            .context("remote stage missing placement")?;
        let command_text = remote_stage_command(args, remote, metrics_otlp_url)?;
        let mut ssh = Command::new("ssh");
        ssh.arg("-n").arg(&remote.host).arg(command_text);
        ssh.stdin(Stdio::null());
        ssh.stdout(Stdio::null()).stderr(Stdio::null());
        children.push(ChildGuard::spawn(ssh)?);
    }

    Ok(())
}

fn stop_remote_stage_listener(stage: &LocalStage) -> Result<()> {
    let Some(remote) = stage.remote.as_ref() else {
        return Ok(());
    };
    let command = format!(
        concat!(
            "pids=$(lsof -tiTCP:{port} -sTCP:LISTEN 2>/dev/null || true); ",
            "if [ -n \"$pids\" ]; then ",
            "echo stopping stale listener on :{port} >&2; ",
            "kill $pids 2>/dev/null || true; ",
            "sleep 0.5; ",
            "kill -9 $pids 2>/dev/null || true; ",
            "fi"
        ),
        port = stage.port
    );
    run_status(
        Command::new("ssh").arg("-n").arg(&remote.host).arg(command),
        &format!("stop stale stage listener {}:{}", remote.host, stage.port),
    )
}

fn add_stage_server_args(
    command: &mut Command,
    args: &PromptArgs,
    stage: &LocalStage,
    metrics_otlp_url: &str,
) -> Result<()> {
    command.args([
        "serve-binary",
        "--config",
        path_str(&stage.config_path)?,
        "--activation-width",
        &args.activation_width.to_string(),
        "--activation-wire-dtype",
        &args.activation_wire_dtype,
        "--metrics-otlp-grpc",
        metrics_otlp_url,
        "--telemetry-queue-capacity",
        &args.stage_telemetry_queue_capacity.to_string(),
        "--telemetry-level",
        &args.stage_telemetry_level,
        "--max-inflight",
        &args.stage_max_inflight.to_string(),
        "--reply-credit-limit",
        &args.stage_reply_credit_limit.to_string(),
    ]);
    if !args.no_stage_async_prefill_forward {
        command.arg("--async-prefill-forward");
    }
    Ok(())
}

fn remote_stage_command(
    args: &PromptArgs,
    remote: &RemoteStage,
    metrics_otlp_url: &str,
) -> Result<String> {
    let mut stage_args = vec![
        shell_quote(&remote.stage_server_bin),
        "serve-binary".to_string(),
        "--config".to_string(),
        shell_quote(&remote.config_path),
        "--activation-width".to_string(),
        shell_quote(&args.activation_width.to_string()),
        "--activation-wire-dtype".to_string(),
        shell_quote(&args.activation_wire_dtype),
        "--metrics-otlp-grpc".to_string(),
        shell_quote(metrics_otlp_url),
        "--telemetry-queue-capacity".to_string(),
        shell_quote(&args.stage_telemetry_queue_capacity.to_string()),
        "--telemetry-level".to_string(),
        shell_quote(&args.stage_telemetry_level),
        "--max-inflight".to_string(),
        shell_quote(&args.stage_max_inflight.to_string()),
        "--reply-credit-limit".to_string(),
        shell_quote(&args.stage_reply_credit_limit.to_string()),
    ];
    if !args.no_stage_async_prefill_forward {
        stage_args.push("--async-prefill-forward".to_string());
    }

    Ok(format!(
        concat!(
            "set -e; ",
            "cd {stage_dir}; ",
            "chmod +x {stage_bin}; ",
            "rm -f stage.pid {stage_exit}; ",
            "{stage_command} > {stage_log} 2>&1 & ",
            "stage_pid=$!; echo $stage_pid > stage.pid; ",
            "trap 'kill $stage_pid 2>/dev/null || true; wait $stage_pid 2>/dev/null || true' INT TERM HUP EXIT; ",
            "set +e; ",
            "wait $stage_pid; stage_status=$?; ",
            "echo $stage_status > {stage_exit}; ",
            "exit $stage_status"
        ),
        stage_dir = shell_quote(&remote.stage_dir),
        stage_bin = shell_quote(&remote.stage_server_bin),
        stage_command = stage_args.join(" "),
        stage_log = shell_quote(&remote.stage_log_path),
        stage_exit = shell_quote(&remote.stage_exit_path),
    ))
}

fn rsync_to_host_with_progress(
    local: &Path,
    host: &str,
    remote_path: &str,
    label: &str,
    tx: &mpsc::Sender<RemoteSyncEvent>,
) -> Result<()> {
    let file_name = local
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("input");
    let size = fs::metadata(local).ok().map(|metadata| metadata.len());
    let label = match size {
        Some(size) => format!("{label} ({file_name}, {})", format_bytes(size)),
        None => format!("{label} ({file_name})"),
    };
    let started = Instant::now();
    send_remote_sync_event(
        tx,
        RemoteSyncEvent::StepStarted {
            host: host.to_string(),
            label: label.clone(),
        },
    );
    let mut command = Command::new("rsync");
    command
        .args(["-az", "--progress", "--chmod=ugo=rwX"])
        .arg(local)
        .arg(format!("{host}:{remote_path}"))
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = command
        .spawn()
        .with_context(|| format!("failed to spawn {:?}", command))?;
    let (progress_tx, progress_rx) = std::sync::mpsc::channel::<String>();
    if let Some(stdout) = child.stdout.take() {
        spawn_rsync_progress_reader(stdout, progress_tx.clone());
    }
    if let Some(stderr) = child.stderr.take() {
        spawn_rsync_progress_reader(stderr, progress_tx);
    }
    let mut rsync_progress = String::new();
    let mut last_report = Instant::now();
    loop {
        while let Ok(line) = progress_rx.try_recv() {
            rsync_progress = line;
        }
        if last_report.elapsed() >= Duration::from_secs(5) {
            send_remote_sync_event(
                tx,
                RemoteSyncEvent::StepProgress {
                    host: host.to_string(),
                    label: label.clone(),
                    detail: if rsync_progress.is_empty() {
                        "waiting for rsync progress".to_string()
                    } else {
                        rsync_progress.clone()
                    },
                    elapsed: started.elapsed(),
                },
            );
            last_report = Instant::now();
        }
        if let Some(status) = child.try_wait()? {
            if !status.success() {
                bail!(
                    "rsync {} to {host}:{remote_path} failed with status {status}",
                    local.display()
                );
            }
            break;
        }
        thread::sleep(Duration::from_millis(500));
    }
    send_remote_sync_event(
        tx,
        RemoteSyncEvent::StepFinished {
            host: host.to_string(),
            label,
            detail: "copied".to_string(),
            elapsed: started.elapsed(),
        },
    );
    Ok(())
}

fn rsync_dir_to_host_with_progress(
    local: &Path,
    host: &str,
    remote_path: &str,
    tx: &mpsc::Sender<RemoteSyncEvent>,
) -> Result<()> {
    let dir_name = local
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("input-dir");
    let label = format!("model package ({dir_name}/)");
    let started = Instant::now();
    send_remote_sync_event(
        tx,
        RemoteSyncEvent::StepStarted {
            host: host.to_string(),
            label: label.clone(),
        },
    );

    run_status(
        Command::new("ssh")
            .arg("-n")
            .arg(host)
            .arg(format!("mkdir -p {}", shell_quote(remote_path))),
        &format!("create remote package dir {host}:{remote_path}"),
    )?;

    let mut command = Command::new("rsync");
    command
        .args(["-az", "--delete", "--progress", "--chmod=ugo=rwX"])
        .arg(format!("{}/", local.display()))
        .arg(format!("{host}:{}/", remote_path))
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = command
        .spawn()
        .with_context(|| format!("failed to spawn {:?}", command))?;
    let (progress_tx, progress_rx) = std::sync::mpsc::channel::<String>();
    if let Some(stdout) = child.stdout.take() {
        spawn_rsync_progress_reader(stdout, progress_tx.clone());
    }
    if let Some(stderr) = child.stderr.take() {
        spawn_rsync_progress_reader(stderr, progress_tx);
    }
    let mut rsync_progress = String::new();
    let mut last_report = Instant::now();
    loop {
        while let Ok(line) = progress_rx.try_recv() {
            rsync_progress = line;
        }
        if last_report.elapsed() >= Duration::from_secs(5) {
            send_remote_sync_event(
                tx,
                RemoteSyncEvent::StepProgress {
                    host: host.to_string(),
                    label: label.clone(),
                    detail: if rsync_progress.is_empty() {
                        "waiting for rsync progress".to_string()
                    } else {
                        rsync_progress.clone()
                    },
                    elapsed: started.elapsed(),
                },
            );
            last_report = Instant::now();
        }
        if let Some(status) = child.try_wait()? {
            if !status.success() {
                bail!(
                    "rsync directory {} to {host}:{remote_path} failed with status {status}",
                    local.display()
                );
            }
            break;
        }
        thread::sleep(Duration::from_millis(500));
    }
    send_remote_sync_event(
        tx,
        RemoteSyncEvent::StepFinished {
            host: host.to_string(),
            label,
            detail: "copied".to_string(),
            elapsed: started.elapsed(),
        },
    );
    Ok(())
}

fn rsync_to_host_cached(
    local: &Path,
    host: &str,
    remote_path: &str,
    remote_root: &str,
    tx: &mpsc::Sender<RemoteSyncEvent>,
) -> Result<()> {
    let file_name = local
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("input");
    let label = format!("binary {file_name}");
    let started = Instant::now();
    if remote_file_available(host, remote_path)? {
        send_remote_sync_event(
            tx,
            RemoteSyncEvent::StepFinished {
                host: host.to_string(),
                label,
                detail: "cached".to_string(),
                elapsed: started.elapsed(),
            },
        );
        return Ok(());
    }
    if promote_remote_artifact(local, host, remote_root, remote_path)? {
        send_remote_sync_event(
            tx,
            RemoteSyncEvent::StepFinished {
                host: host.to_string(),
                label,
                detail: "moved remote artifact into cache".to_string(),
                elapsed: started.elapsed(),
            },
        );
        return Ok(());
    }
    rsync_to_host_with_progress(local, host, remote_path, &label, tx)
}

fn rsync_dir_to_host_cached(
    local: &Path,
    host: &str,
    remote_path: &str,
    tx: &mpsc::Sender<RemoteSyncEvent>,
) -> Result<()> {
    let dir_name = local
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("input-dir");
    let label = format!("model package ({dir_name}/)");
    let started = Instant::now();
    if remote_package_available(local, host, remote_path)? {
        send_remote_sync_event(
            tx,
            RemoteSyncEvent::StepFinished {
                host: host.to_string(),
                label,
                detail: "cached".to_string(),
                elapsed: started.elapsed(),
            },
        );
        return Ok(());
    }
    rsync_dir_to_host_with_progress(local, host, remote_path, tx)
}

fn promote_remote_artifact(
    local: &Path,
    host: &str,
    remote_root: &str,
    remote_path: &str,
) -> Result<bool> {
    let Some(file_name) = local.file_name().and_then(|value| value.to_str()) else {
        return Ok(false);
    };
    let local_size = fs::metadata(local)
        .with_context(|| format!("stat local artifact {}", local.display()))?
        .len();
    let remote_parent = Path::new(remote_path)
        .parent()
        .and_then(Path::to_str)
        .context("remote artifact path has no parent")?;
    let command = format!(
        concat!(
            "candidate=$(find {root} -type f -name {file_name} -size {size}c ! -path {target} -print -quit 2>/dev/null); ",
            "if [ -n \"$candidate\" ]; then ",
            "mkdir -p {parent}; ",
            "mv \"$candidate\" {target}; ",
            "fi; ",
            "test -s {target}"
        ),
        root = shell_quote(remote_root),
        file_name = shell_quote(file_name),
        size = local_size,
        target = shell_quote(remote_path),
        parent = shell_quote(remote_parent),
    );
    let status = Command::new("ssh")
        .arg("-n")
        .arg(host)
        .arg(command)
        .status()
        .with_context(|| format!("promote remote artifact on {host}"))?;
    Ok(status.success())
}

struct PackageArtifactCheck {
    path: String,
    artifact_bytes: u64,
}

fn remote_package_available(local: &Path, host: &str, remote_path: &str) -> Result<bool> {
    let artifacts = package_artifact_checks(local)?;
    let manifest = format!("{remote_path}/model-package.json");
    let marker = format!("{remote_path}/.complete");
    let mut checks = vec![format!(
        "test -s {} -a -s {}",
        shell_quote(&manifest),
        shell_quote(&marker)
    )];
    for artifact in artifacts {
        let remote_artifact = format!("{remote_path}/{}", artifact.path);
        checks.push(format!(
            concat!(
                "(test -f {path} && ",
                "actual=$(wc -c < {path} 2>/dev/null | tr -d '[:space:]') && ",
                "test \"$actual\" = {expected})"
            ),
            path = shell_quote(&remote_artifact),
            expected = shell_quote(&artifact.artifact_bytes.to_string())
        ));
    }
    let status = Command::new("ssh")
        .arg("-n")
        .arg(host)
        .arg(checks.join(" && "))
        .status()
        .with_context(|| format!("check remote package {host}:{remote_path}"))?;
    Ok(status.success())
}

fn package_artifact_checks(package_dir: &Path) -> Result<Vec<PackageArtifactCheck>> {
    let manifest_path = package_dir.join("model-package.json");
    let manifest: Value = serde_json::from_slice(
        &fs::read(&manifest_path).with_context(|| format!("read {}", manifest_path.display()))?,
    )
    .with_context(|| format!("parse {}", manifest_path.display()))?;
    let mut artifacts = Vec::new();

    let shared = manifest
        .get("shared")
        .and_then(Value::as_object)
        .context("package manifest missing shared artifact map")?;
    for key in ["metadata", "embeddings", "output"] {
        artifacts.push(package_artifact_check(
            shared
                .get(key)
                .with_context(|| format!("package manifest missing shared.{key}"))?,
            &format!("shared.{key}"),
        )?);
    }

    let layers = manifest
        .get("layers")
        .and_then(Value::as_array)
        .context("package manifest missing layers array")?;
    for (index, layer) in layers.iter().enumerate() {
        artifacts.push(package_artifact_check(layer, &format!("layers[{index}]"))?);
    }

    Ok(artifacts)
}

fn package_artifact_check(value: &Value, label: &str) -> Result<PackageArtifactCheck> {
    let path = value
        .get("path")
        .and_then(Value::as_str)
        .with_context(|| format!("package manifest artifact {label} missing path"))?;
    validate_package_relative_path(path, label)?;
    let artifact_bytes = value
        .get("artifact_bytes")
        .and_then(Value::as_u64)
        .with_context(|| format!("package manifest artifact {label} missing artifact_bytes"))?;
    Ok(PackageArtifactCheck {
        path: path.to_string(),
        artifact_bytes,
    })
}

fn validate_package_relative_path(path: &str, label: &str) -> Result<()> {
    let relative = Path::new(path);
    if !path.is_empty()
        && relative.components().all(|component| match component {
            Component::Normal(_) => true,
            Component::CurDir => true,
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => false,
        })
    {
        Ok(())
    } else {
        bail!("package manifest artifact {label} has unsafe path {path:?}")
    }
}

fn remote_file_available(host: &str, remote_path: &str) -> Result<bool> {
    let status = Command::new("ssh")
        .arg("-n")
        .arg(host)
        .arg(format!("test -s {}", shell_quote(remote_path)))
        .status()
        .with_context(|| format!("check remote artifact {host}:{remote_path}"))?;
    Ok(status.success())
}

fn spawn_rsync_progress_reader<R>(reader: R, tx: std::sync::mpsc::Sender<String>)
where
    R: std::io::Read + Send + 'static,
{
    thread::spawn(move || {
        let mut reader = BufReader::new(reader);
        let mut buffer = Vec::new();
        loop {
            buffer.clear();
            match reader.read_until(b'\r', &mut buffer) {
                Ok(0) => break,
                Ok(_) => {
                    let line = String::from_utf8_lossy(&buffer)
                        .trim_matches(|ch| ch == '\r' || ch == '\n')
                        .trim()
                        .to_string();
                    if !line.is_empty() {
                        let _ = tx.send(line);
                    }
                }
                Err(_) => break,
            }
        }
    });
}
