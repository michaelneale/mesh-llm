pub fn load_fingerprint(path: &Path) -> Option<BenchmarkFingerprint> {
    let content = std::fs::read_to_string(path).ok()?;
    serde_json::from_str(&content).ok()
}

/// Atomically write a `BenchmarkFingerprint` to disk.
/// Uses a `.json.tmp` staging file + rename for crash safety.
/// Logs a warning on failure — never panics.
pub fn save_fingerprint(path: &Path, fp: &BenchmarkFingerprint) {
    if let Err(err) = try_save_fingerprint(path, fp) {
        tracing::warn!("benchmark: failed to persist fingerprint: {err}");
    }
}

pub fn try_save_fingerprint(path: &Path, fp: &BenchmarkFingerprint) -> Result<()> {
    let tmp = path.with_extension("json.tmp");

    std::fs::create_dir_all(path.parent().unwrap_or_else(|| Path::new(".")))
        .with_context(|| format!("failed to create cache dir for {}", path.display()))?;

    let json =
        serde_json::to_string_pretty(fp).context("failed to serialize benchmark fingerprint")?;

    std::fs::write(&tmp, &json)
        .with_context(|| format!("failed to write temporary fingerprint {}", tmp.display()))?;

    // On Windows, `rename` fails if the destination already exists.
    // Remove the destination first there; on Unix the rename stays atomic.
    #[cfg(windows)]
    if path.exists() {
        std::fs::remove_file(path)
            .with_context(|| format!("failed to remove existing fingerprint {}", path.display()))?;
    }

    if let Err(e) = std::fs::rename(&tmp, path) {
        let _ = std::fs::remove_file(&tmp);
        return Err(e).with_context(|| {
            format!(
                "failed to rename fingerprint into place at {}",
                path.display()
            )
        });
    }

    Ok(())
}
