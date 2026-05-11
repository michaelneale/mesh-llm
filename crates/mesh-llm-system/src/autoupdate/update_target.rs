async fn resolve_release_info(requested_version: Option<&str>) -> Result<Option<ReleaseInfo>> {
    let Some(requested_version) = requested_version else {
        return Ok(latest_release_info().await);
    };
    let tag = normalize_release_tag(requested_version)?;
    Ok(release_info_for_tag(&tag).await)
}

fn normalize_release_tag(raw: &str) -> Result<String> {
    let trimmed = raw.trim();
    anyhow::ensure!(!trimmed.is_empty(), "release version must not be empty");
    let version = trimmed.trim_start_matches('v');
    semver::Version::parse(version).with_context(|| format!("Invalid release version: {raw}"))?;
    Ok(format!("v{version}"))
}

fn describe_requested_update(
    target_version: &str,
    current_version: &str,
    exact: bool,
) -> &'static str {
    if !exact {
        return "Updating";
    }

    match (
        semver::Version::parse(target_version),
        semver::Version::parse(current_version),
    ) {
        (Ok(target), Ok(current)) if target < current => "Downgrading",
        (Ok(target), Ok(current)) if target == current => "Reinstalling",
        _ => "Installing",
    }
}

fn path_is_writable(path: &Path) -> bool {
    #[cfg(unix)]
    {
        let Ok(c_path) = CString::new(path.as_os_str().as_bytes()) else {
            return false;
        };
        unsafe { libc::access(c_path.as_ptr(), libc::W_OK) == 0 }
    }

    #[cfg(not(unix))]
    {
        std::fs::metadata(path)
            .map(|meta| !meta.permissions().readonly())
            .unwrap_or(false)
    }
}

fn bundle_install_dir(
    exe: &Path,
    requested_flavor: Option<backend::BinaryFlavor>,
) -> Option<(PathBuf, backend::BinaryFlavor)> {
    let dir = exe.parent()?;
    let file_name = exe.file_name()?.to_str()?;
    #[cfg(windows)]
    {
        if !file_name.eq_ignore_ascii_case(&mesh_binary_name()) {
            return None;
        }
    }
    #[cfg(not(windows))]
    {
        if file_name != mesh_binary_name() {
            return None;
        }
    }
    let flavor = installed_bundle_flavor(dir, requested_flavor)?;
    Some((dir.to_path_buf(), flavor))
}
