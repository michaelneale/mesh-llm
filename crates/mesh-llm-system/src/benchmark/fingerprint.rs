/// Normalize `HardwareSurvey.gpu_name` into a per-GPU list of names.
/// - Splits on ',' and trims whitespace for robustness.
/// - Expands summarized forms like "8× NVIDIA A100" into 8 identical entries.
/// - If the expanded list length does not match `gpu_vram.len()` but `gpu_vram` is
///   non-empty, falls back to assuming all GPUs share the same summarized name and
///   returns `gpu_vram.len()` copies of it.
fn per_gpu_names(hw: &HardwareSurvey) -> Vec<String> {
    let raw = match hw.gpu_name.as_deref() {
        Some(s) => s.trim(),
        None => return Vec::new(),
    };

    if raw.is_empty() {
        return Vec::new();
    }

    let mut names: Vec<String> = Vec::new();

    for part in raw.split(',') {
        let part_trimmed = part.trim();
        if part_trimmed.is_empty() {
            continue;
        }

        // Handle summarized "N× name" form (e.g., "8× NVIDIA A100").
        if let Some((count_str, name)) = part_trimmed.split_once('×') {
            if let Ok(count) = count_str.trim().parse::<usize>() {
                let name_trimmed = name.trim();
                for _ in 0..count {
                    names.push(name_trimmed.to_string());
                }
                continue;
            }
        }

        // Fallback: treat as a single GPU name.
        names.push(part_trimmed.to_string());
    }

    if names.len() == hw.gpu_vram.len() || hw.gpu_vram.is_empty() {
        return names;
    }

    // As a last resort, assume all GPUs share the same summarized name.
    let gpu_count = hw.gpu_vram.len();
    vec![raw.to_string(); gpu_count]
}

/// Returns true if the current hardware differs from the fingerprint's recorded hardware.
/// Compares GPU names, VRAM sizes (by index), and the is_soc flag.
pub fn hardware_changed(fingerprint: &BenchmarkFingerprint, hw: &HardwareSurvey) -> bool {
    if fingerprint.is_soc != hw.is_soc {
        return true;
    }

    let hw_names: Vec<String> = per_gpu_names(hw);

    if fingerprint.gpus.len() != hw_names.len() || fingerprint.gpus.len() != hw.gpu_vram.len() {
        return true;
    }

    for (i, cached) in fingerprint.gpus.iter().enumerate() {
        if cached.name != hw_names[i] || cached.vram_bytes != hw.gpu_vram[i] {
            return true;
        }
    }
    false
}
