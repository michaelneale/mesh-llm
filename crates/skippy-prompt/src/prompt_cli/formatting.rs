fn format_bytes(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut value = bytes as f64;
    let mut unit = 0usize;
    while value >= 1024.0 && unit + 1 < UNITS.len() {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} {}", UNITS[unit])
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}

fn format_stage_mask(mask: i64) -> String {
    if mask <= 0 {
        return "-".to_string();
    }
    let stages = (0..63)
        .filter(|index| (mask & (1_i64 << index)) != 0)
        .map(|index| index.to_string())
        .collect::<Vec<_>>();
    if stages.is_empty() {
        "-".to_string()
    } else {
        stages.join(",")
    }
}

fn stage_mask_count(mask: i64) -> u64 {
    if mask <= 0 {
        return 0;
    }
    (mask as u64).count_ones() as u64
}

fn run_status(command: &mut Command, description: &str) -> Result<()> {
    let status = command
        .status()
        .with_context(|| format!("{description}: failed to spawn {:?}", command))?;
    if !status.success() {
        bail!("{description} failed with status {status}");
    }
    Ok(())
}

fn metrics_otlp_url(args: &PromptArgs, stages: &[LocalStage]) -> Result<String> {
    if let Some(url) = args.metrics_otlp_grpc_url.clone() {
        return Ok(url);
    }

    let Some(first_remote) = stages.iter().find_map(|stage| stage.remote.as_ref()) else {
        return Ok(format!("http://{}", args.metrics_otlp_grpc_addr));
    };

    let launcher_host = infer_launcher_host_from_ssh(&first_remote.host)?;
    Ok(format!(
        "http://{}:{}",
        launcher_host,
        args.metrics_otlp_grpc_addr.port()
    ))
}

fn metrics_otlp_bind_addr(args: &PromptArgs, remote: bool) -> String {
    if remote && args.metrics_otlp_grpc_addr.ip().is_loopback() {
        format!("0.0.0.0:{}", args.metrics_otlp_grpc_addr.port())
    } else {
        args.metrics_otlp_grpc_addr.to_string()
    }
}

fn stage_model_location(stage: &LocalStage) -> String {
    stage
        .remote
        .as_ref()
        .map(|remote| remote.model_path.clone())
        .unwrap_or_else(|| stage.model_path.display().to_string())
}

fn infer_launcher_host_from_ssh(host: &str) -> Result<String> {
    let output = Command::new("ssh")
        .arg("-n")
        .arg(host)
        .arg("sh -lc 'set -- $SSH_CONNECTION; printf %s \"$1\"'")
        .output()
        .with_context(|| format!("infer launcher host via ssh {host}"))?;
    if !output.status.success() {
        bail!(
            "infer launcher host via ssh {host} failed with status {}",
            output.status
        );
    }
    let value = String::from_utf8(output.stdout)
        .context("ssh launcher host output was not UTF-8")?
        .trim()
        .to_string();
    if value.is_empty() {
        bail!("ssh {host} did not report SSH_CONNECTION client address");
    }
    Ok(value)
}

fn model_cache_key(model_path: &Path, ranges: &[(u32, u32)]) -> Result<String> {
    let canonical = model_path
        .canonicalize()
        .with_context(|| format!("canonicalize model path {}", model_path.display()))?;
    let metadata =
        fs::metadata(&canonical).with_context(|| format!("stat model {}", canonical.display()))?;
    let modified = metadata
        .modified()
        .ok()
        .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok());

    let mut hasher = DefaultHasher::new();
    canonical.hash(&mut hasher);
    metadata.len().hash(&mut hasher);
    if let Some(modified) = modified {
        modified.as_secs().hash(&mut hasher);
        modified.subsec_nanos().hash(&mut hasher);
    }
    ranges.hash(&mut hasher);

    let stem = canonical
        .file_stem()
        .and_then(|value| value.to_str())
        .map(sanitize_cache_name)
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "model".to_string());
    Ok(format!("{stem}-{:016x}", hasher.finish()))
}

fn model_package_cache_key(model_path: &Path) -> Result<String> {
    let canonical = model_path
        .canonicalize()
        .with_context(|| format!("canonicalize model path {}", model_path.display()))?;
    let metadata =
        fs::metadata(&canonical).with_context(|| format!("stat model {}", canonical.display()))?;
    let modified = metadata
        .modified()
        .ok()
        .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok());

    let mut hasher = DefaultHasher::new();
    canonical.hash(&mut hasher);
    metadata.len().hash(&mut hasher);
    if let Some(modified) = modified {
        modified.as_secs().hash(&mut hasher);
        modified.subsec_nanos().hash(&mut hasher);
    }

    let stem = canonical
        .file_stem()
        .and_then(|value| value.to_str())
        .map(sanitize_cache_name)
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "model".to_string());
    Ok(format!("{stem}-package-{:016x}", hasher.finish()))
}

fn binary_cache_key(stage_server_bin: &Path) -> Result<String> {
    let mut hasher = DefaultHasher::new();
    hash_file_identity(stage_server_bin, &mut hasher)?;
    Ok(format!("{:016x}", hasher.finish()))
}

fn hash_file_identity(path: &Path, hasher: &mut DefaultHasher) -> Result<()> {
    let mut file =
        fs::File::open(path).with_context(|| format!("open binary {}", path.display()))?;
    let metadata = file
        .metadata()
        .with_context(|| format!("stat binary {}", path.display()))?;
    metadata.len().hash(hasher);
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .with_context(|| format!("read binary {}", path.display()))?;
        if read == 0 {
            break;
        }
        buffer[..read].hash(hasher);
    }
    Ok(())
}

fn sanitize_cache_name(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '.' | '-' | '_') {
                ch
            } else {
                '-'
            }
        })
        .collect()
}

fn shell_quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\\''"))
}
