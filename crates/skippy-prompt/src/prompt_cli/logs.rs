fn handle_prompt_error(
    error: anyhow::Error,
    interrupt: &Arc<PromptInterruptState>,
    prompt_index: usize,
) -> Result<()> {
    if interrupt.take_interrupt() {
        eprintln!();
        eprintln!("request {prompt_index}: interrupted");
        Ok(())
    } else {
        Err(error)
    }
}

fn stage_diagnostics_hint(run_dir: &Path, stages: &[LocalStage]) -> String {
    let mut lines = vec![
        format!("run_dir={}", run_dir.display()),
        "stage diagnostics:".to_string(),
    ];
    for stage in stages {
        if let Some(remote) = stage.remote.as_ref() {
            lines.push(format!(
                "  {} {}:{}..{} addr={} log={}:{} exit={}:{}",
                stage.stage_id,
                remote.host,
                stage.layer_start,
                stage.layer_end,
                stage.endpoint_addr,
                remote.host,
                remote.stage_log_path,
                remote.host,
                remote.stage_exit_path
            ));
        } else {
            lines.push(format!(
                "  {} layers {}..{} addr={} log={}",
                stage.stage_id,
                stage.layer_start,
                stage.layer_end,
                stage.endpoint_addr,
                run_dir
                    .join(format!("stage-{}.log", stage.stage_index))
                    .display()
            ));
        }
    }
    lines.join("\n")
}

fn prompt_log_context(
    run_dir: &Path,
    stages: &[LocalStage],
    default_lines: usize,
) -> PromptLogContext {
    let mut entries = vec![PromptLogEntry {
        label: "metrics-server".to_string(),
        target: PromptLogTarget::Local(run_dir.join("metrics-server.log")),
    }];
    for stage in stages {
        if let Some(remote) = stage.remote.as_ref() {
            entries.push(PromptLogEntry {
                label: format!("stage-{}", stage.stage_index),
                target: PromptLogTarget::Remote {
                    host: remote.host.clone(),
                    path: remote.stage_log_path.clone(),
                },
            });
        } else {
            entries.push(PromptLogEntry {
                label: format!("stage-{}", stage.stage_index),
                target: PromptLogTarget::Local(
                    run_dir.join(format!("stage-{}.log", stage.stage_index)),
                ),
            });
        }
    }
    PromptLogContext {
        entries,
        default_lines,
    }
}

fn show_prompt_logs(context: Option<&PromptLogContext>, spec: &str) -> Result<()> {
    let Some(context) = context else {
        eprintln!("no prompt-managed logs are attached to this REPL");
        return Ok(());
    };
    let mut lines = context.default_lines.max(1);
    let mut filters = Vec::new();
    for part in spec.split_whitespace() {
        if let Ok(value) = part.parse::<usize>() {
            lines = value.max(1);
        } else {
            filters.push(part);
        }
    }
    let exact_entries = context
        .entries
        .iter()
        .filter(|entry| filters.iter().all(|filter| entry.label == *filter))
        .collect::<Vec<_>>();
    let entries = if exact_entries.is_empty() && !filters.is_empty() {
        context
            .entries
            .iter()
            .filter(|entry| filters.iter().all(|filter| entry.label.contains(filter)))
            .collect::<Vec<_>>()
    } else {
        exact_entries
    };
    if entries.is_empty() {
        let available = context
            .entries
            .iter()
            .map(|entry| entry.label.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        eprintln!("no logs matched; available logs: {available}");
        return Ok(());
    }

    for entry in entries {
        eprintln!(
            "==> {} ({}) <==",
            entry.label,
            describe_log_target(&entry.target)
        );
        match tail_prompt_log(&entry.target, lines) {
            Ok(tail) if tail.is_empty() => eprintln!("  <empty>"),
            Ok(tail) => {
                for line in tail {
                    eprintln!("{line}");
                }
            }
            Err(error) => eprintln!("  <failed to read log: {error:#}>"),
        }
    }
    Ok(())
}

fn describe_log_target(target: &PromptLogTarget) -> String {
    match target {
        PromptLogTarget::Local(path) => path.display().to_string(),
        PromptLogTarget::Remote { host, path } => format!("{host}:{path}"),
    }
}

fn tail_prompt_log(target: &PromptLogTarget, lines: usize) -> Result<Vec<String>> {
    match target {
        PromptLogTarget::Local(path) => tail_local_log(path, lines),
        PromptLogTarget::Remote { host, path } => tail_remote_log(host, path, lines),
    }
}

fn tail_local_log(path: &Path, lines: usize) -> Result<Vec<String>> {
    let file = fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let reader = BufReader::new(file);
    let mut tail = VecDeque::with_capacity(lines);
    for line in reader.lines() {
        if tail.len() == lines {
            tail.pop_front();
        }
        tail.push_back(line.with_context(|| format!("read {}", path.display()))?);
    }
    Ok(tail.into_iter().collect())
}

fn tail_remote_log(host: &str, path: &str, lines: usize) -> Result<Vec<String>> {
    let output = Command::new("ssh")
        .arg("-n")
        .arg(host)
        .arg(format!("tail -n {lines} {}", shell_quote(path)))
        .output()
        .with_context(|| format!("tail remote log {host}:{path}"))?;
    if !output.status.success() {
        bail!(
            "tail remote log {host}:{path} failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(str::to_string)
        .collect())
}
