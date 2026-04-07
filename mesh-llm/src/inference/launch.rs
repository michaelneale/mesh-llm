//! Process management for llama.cpp binaries.
//!
//! Starts rpc-server and llama-server wired up to the mesh tunnel ports.

use anyhow::{Context, Result};
use clap::ValueEnum;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::net::TcpListener;
use tokio::process::Command;

/// llama.cpp split mode for distributing tensors across devices.
///
/// - `Layer` (default): each device gets a contiguous range of layers.
///   Works over RPC (network) and local multi-GPU.
/// - `Row`: weight matrices are sharded across devices (true tensor parallelism).
///   Only works for local multi-GPU (CUDA, ROCm) — NOT over RPC.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[allow(dead_code)] // Layer is available for explicit CLI override
pub enum SplitMode {
    /// Pipeline parallelism — split by layers (default, works everywhere).
    Layer,
    /// Tensor parallelism — split weight rows across local GPUs.
    Row,
}

impl SplitMode {
    fn as_arg(self) -> &'static str {
        match self {
            SplitMode::Layer => "layer",
            SplitMode::Row => "row",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum BinaryFlavor {
    Cpu,
    Cuda,
    Rocm,
    Vulkan,
    Metal,
}

impl BinaryFlavor {
    pub const ALL: [BinaryFlavor; 5] = [
        BinaryFlavor::Cpu,
        BinaryFlavor::Cuda,
        BinaryFlavor::Rocm,
        BinaryFlavor::Vulkan,
        BinaryFlavor::Metal,
    ];

    pub fn suffix(self) -> &'static str {
        match self {
            BinaryFlavor::Cpu => "cpu",
            BinaryFlavor::Cuda => "cuda",
            BinaryFlavor::Rocm => "rocm",
            BinaryFlavor::Vulkan => "vulkan",
            BinaryFlavor::Metal => "metal",
        }
    }

    fn preferred_devices(self) -> &'static [&'static str] {
        match self {
            BinaryFlavor::Cpu => &["CPU"],
            BinaryFlavor::Cuda => &["CUDA0", "CPU"],
            BinaryFlavor::Rocm => &["HIP0", "CPU"],
            BinaryFlavor::Vulkan => &["Vulkan0", "CPU"],
            BinaryFlavor::Metal => &["MTL0", "CPU"],
        }
    }

    fn primary_device(self) -> &'static str {
        self.preferred_devices()[0]
    }
}

#[derive(Clone, Debug)]
struct ResolvedBinary {
    path: PathBuf,
    flavor: Option<BinaryFlavor>,
}

pub(crate) fn platform_bin_name(name: &str) -> String {
    #[cfg(windows)]
    {
        if Path::new(name)
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.eq_ignore_ascii_case("exe"))
            .unwrap_or(false)
        {
            name.to_string()
        } else {
            format!("{name}.exe")
        }
    }

    #[cfg(not(windows))]
    {
        name.to_string()
    }
}

fn flavored_bin_name(name: &str, flavor: BinaryFlavor) -> String {
    platform_bin_name(&format!("{name}-{}", flavor.suffix()))
}

fn bare_bin_name(path: &Path) -> Option<String> {
    let file_name = path.file_name()?.to_string_lossy();
    #[cfg(windows)]
    {
        // On Windows, strip a `.exe` extension in a case-insensitive way.
        if path
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.eq_ignore_ascii_case("exe"))
            .unwrap_or(false)
        {
            Some(path.file_stem()?.to_string_lossy().to_string())
        } else {
            Some(file_name.to_string())
        }
    }

    #[cfg(not(windows))]
    {
        Some(file_name.to_string())
    }
}

fn infer_binary_flavor(name: &str, path: &Path) -> Option<BinaryFlavor> {
    let file_name = bare_bin_name(path)?;
    for flavor in BinaryFlavor::ALL {
        if file_name == format!("{name}-{}", flavor.suffix()) {
            return Some(flavor);
        }
    }
    None
}

fn resolve_binary_path(
    bin_dir: &Path,
    name: &str,
    requested_flavor: Option<BinaryFlavor>,
) -> Result<ResolvedBinary> {
    if let Some(flavor) = requested_flavor {
        let flavored = bin_dir.join(flavored_bin_name(name, flavor));
        if flavored.exists() {
            return Ok(ResolvedBinary {
                path: flavored,
                flavor: Some(flavor),
            });
        }

        let generic = bin_dir.join(platform_bin_name(name));
        if generic.exists() {
            return Ok(ResolvedBinary {
                path: generic,
                flavor: Some(flavor),
            });
        }

        anyhow::bail!(
            "{} not found in {} for requested flavor '{}'",
            flavored.display(),
            bin_dir.display(),
            flavor.suffix()
        );
    }

    let generic = bin_dir.join(platform_bin_name(name));
    if generic.exists() {
        let flavor = infer_binary_flavor(name, &generic);
        return Ok(ResolvedBinary {
            path: generic,
            flavor,
        });
    }

    let matches: Vec<ResolvedBinary> = BinaryFlavor::ALL
        .into_iter()
        .map(|flavor| ResolvedBinary {
            path: bin_dir.join(flavored_bin_name(name, flavor)),
            flavor: Some(flavor),
        })
        .filter(|candidate| candidate.path.exists())
        .collect();

    match matches.len() {
        1 => Ok(matches.into_iter().next().unwrap()),
        0 => anyhow::bail!(
            "{} not found in {}",
            bin_dir.join(platform_bin_name(name)).display(),
            bin_dir.display()
        ),
        _ => {
            let options = matches
                .iter()
                .filter_map(|candidate| candidate.flavor.map(|flavor| flavor.suffix()))
                .collect::<Vec<_>>()
                .join(", ");
            anyhow::bail!(
                "multiple {} flavors found in {} ({options}). Pass --llama-flavor to choose one.",
                name,
                bin_dir.display()
            );
        }
    }
}

fn temp_log_path(name: &str) -> PathBuf {
    let mesh_pid = std::process::id();
    std::env::temp_dir().join(format!("mesh-llm-{mesh_pid}-{name}"))
}

#[derive(Clone, Debug)]
pub struct InferenceServerHandle {
    pid: u32,
    expected_exit: Arc<AtomicBool>,
}

impl InferenceServerHandle {
    pub fn pid(&self) -> u32 {
        self.pid
    }

    pub async fn shutdown(&self) {
        self.expected_exit.store(true, Ordering::Relaxed);
        terminate_process(self.pid).await;
    }
}

#[derive(Debug)]
pub struct InferenceServerProcess {
    pub handle: InferenceServerHandle,
    pub death_rx: tokio::sync::oneshot::Receiver<()>,
    pub context_length: u32,
}

pub struct ModelLaunchSpec<'a> {
    pub model: &'a Path,
    pub http_port: u16,
    pub tunnel_ports: &'a [u16],
    pub tensor_split: Option<&'a str>,
    pub split_mode: Option<SplitMode>,
    pub draft: Option<&'a Path>,
    pub draft_max: u16,
    pub model_bytes: u64,
    pub my_vram: u64,
    pub mmproj: Option<&'a Path>,
    pub ctx_size_override: Option<u32>,
    pub total_group_vram: Option<u64>,
}

pub(crate) const GB: u64 = 1_000_000_000;

fn compute_context_size(
    ctx_size_override: Option<u32>,
    model_bytes: u64,
    my_vram: u64,
    total_group_vram: Option<u64>,
) -> u32 {
    let host_model_bytes = if let Some(group_vram) = total_group_vram {
        if group_vram > 0 {
            let host_fraction = my_vram as f64 / group_vram as f64;
            (model_bytes as f64 * host_fraction) as u64
        } else {
            model_bytes
        }
    } else {
        model_bytes
    };
    let vram_after_model = my_vram.saturating_sub(host_model_bytes);
    if let Some(override_ctx) = ctx_size_override {
        override_ctx
    } else if vram_after_model >= 30 * GB {
        65536
    } else if vram_after_model >= 12 * GB {
        32768
    } else if vram_after_model >= 6 * GB {
        16384
    } else if vram_after_model >= 3 * GB {
        8192
    } else {
        4096
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum KvType {
    F16,
    Q8_0,
    Q4_0,
}

impl KvType {
    fn as_arg(&self) -> &'static str {
        match self {
            KvType::F16 => "f16",
            KvType::Q8_0 => "q8_0",
            KvType::Q4_0 => "q4_0",
        }
    }

    fn is_quantized(&self) -> bool {
        !matches!(self, KvType::F16)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct KvCacheQuant {
    pub k_type: KvType,
    pub v_type: KvType,
}

/// Tracks a known open upstream llama.cpp bug that constrains what KV cache
/// configurations are actually safe to run. Used by `validation_warnings()`
/// so call sites and tests can assert on specific bugs by ID.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum KvCacheWarning {
    /// ggml-org/llama.cpp#20866 — asymmetric quantized K/V types hit
    /// BEST_FATTN_KERNEL_NONE in the CUDA FA kernel selector unless llama.cpp
    /// is built with -DGGML_CUDA_FA_ALL_QUANTS=ON. Our own CUDA build sets
    /// this flag (see scripts/build-linux.sh), but standard Homebrew and
    /// official release binaries will crash on FA ops.
    MismatchedQuantNeedsCudaFaAllQuants,
    /// ggml-org/llama.cpp#21450 — on Metal (Apple Silicon), when Flash
    /// Attention falls back to CPU, a quantized V cache crashes with
    /// "quantized V cache requires Flash Attention".
    QuantizedVBreaksMetalFaFallback,
}

impl KvCacheQuant {
    /// Thresholds in bytes for the tier boundaries below. Named constants so
    /// the tests can assert exact boundary behavior.
    pub const MEDIUM_TIER_MIN_BYTES: u64 = 5 * GB;
    pub const LARGE_TIER_MIN_BYTES: u64 = 50 * GB;

    /// Choose a KV cache quantization pair for the given model size.
    ///
    /// The small and large tiers are safe on all supported backends without
    /// extra build flags. The medium tier (5-50GB) opts into Q8_0/Q4_0 for
    /// ~25% additional KV savings and relies on GGML_CUDA_FA_ALL_QUANTS in
    /// our CUDA build (enforced by scripts/build-linux.sh) and on Metal
    /// Flash Attention being available on the host (true for all M1+ Macs).
    /// `validation_warnings()` returns the open upstream bugs this tier
    /// currently trips (`#20866`, `#21450`), and
    /// `detect_known_crash_signature` attributes the matching crash in the
    /// llama-server failure path. See the tier comment block in
    /// `build_llama_server_args` for the full rationale.
    pub fn for_model_size(model_bytes: u64) -> Self {
        if model_bytes >= Self::LARGE_TIER_MIN_BYTES {
            Self {
                k_type: KvType::Q4_0,
                v_type: KvType::Q4_0,
            }
        } else if model_bytes >= Self::MEDIUM_TIER_MIN_BYTES {
            // Aggressive asymmetric: ~25% less KV memory than Q8_0/Q8_0 with
            // minimal quality impact. Known risks:
            // - ggml-org/llama.cpp#20866: requires CUDA build flag (we set it)
            // - ggml-org/llama.cpp#21450: crashes if Metal FA unavailable.
            //   Safe on M1+ Macs (all support Metal FA); rare CPU-FA fallback
            //   on older Intel Macs is detected by `detect_known_crash_signature`.
            Self {
                k_type: KvType::Q8_0,
                v_type: KvType::Q4_0,
            }
        } else {
            Self {
                k_type: KvType::F16,
                v_type: KvType::F16,
            }
        }
    }

    /// Tier-specific human-readable label used in the startup log line.
    pub fn label(&self, model_bytes: u64) -> String {
        let tier = if model_bytes >= Self::LARGE_TIER_MIN_BYTES {
            "model > 50GB"
        } else if model_bytes >= Self::MEDIUM_TIER_MIN_BYTES {
            "model 5-50GB, aggressive asymmetric"
        } else {
            "model < 5GB, no quantization"
        };
        format!(
            "{} K + {} V ({tier})",
            self.k_type.as_arg().to_uppercase(),
            self.v_type.as_arg().to_uppercase()
        )
    }

    /// Return the set of known open upstream bugs this configuration would
    /// trip over. Empty means safe to ship with default llama.cpp builds.
    pub fn validation_warnings(&self) -> Vec<KvCacheWarning> {
        let mut warnings = Vec::new();
        let mismatched = self.k_type != self.v_type;

        if self.v_type.is_quantized() && mismatched {
            warnings.push(KvCacheWarning::QuantizedVBreaksMetalFaFallback);
        }

        if self.k_type.is_quantized() && self.v_type.is_quantized() && mismatched {
            warnings.push(KvCacheWarning::MismatchedQuantNeedsCudaFaAllQuants);
        }

        warnings
    }

    /// Emit `--cache-type-k`/`--cache-type-v` args (skipped for f16/f16 which
    /// is the llama-server default), log validation warnings for any known
    /// upstream bugs, and log the tier info line.
    pub fn append_args(&self, args: &mut Vec<String>, model_bytes: u64) {
        for warning in self.validation_warnings() {
            emit_kv_cache_warning(warning, self.k_type, self.v_type);
        }

        if self.k_type == KvType::F16 && self.v_type == KvType::F16 {
            tracing::info!("KV cache: {}", self.label(model_bytes));
            return;
        }

        args.extend_from_slice(&[
            "--cache-type-k".to_string(),
            self.k_type.as_arg().to_string(),
            "--cache-type-v".to_string(),
            self.v_type.as_arg().to_string(),
        ]);
        tracing::info!("KV cache: {}", self.label(model_bytes));
    }
}

impl KvCacheWarning {
    /// Substrings that uniquely identify this bug's crash in a llama.cpp log
    /// tail. `detect_known_crash_signature` scans for any of these.
    fn crash_signatures(&self) -> &'static [&'static str] {
        match self {
            KvCacheWarning::MismatchedQuantNeedsCudaFaAllQuants => &[
                "fatal error",
                "BEST_FATTN_KERNEL_NONE",
                "ggml-cuda/fattn.cu",
            ],
            KvCacheWarning::QuantizedVBreaksMetalFaFallback => &[
                "quantized V cache requires Flash Attention",
                "V cache quantization requires flash_attn",
            ],
        }
    }

    /// Stable, user-facing description of the upstream bug this warning
    /// corresponds to. Used in post-mortem messages.
    fn post_mortem_hint(&self) -> &'static str {
        match self {
            KvCacheWarning::MismatchedQuantNeedsCudaFaAllQuants =>
                "Known upstream bug: CUDA FA kernel selector rejects mismatched K/V \
                 quantization types without GGML_CUDA_FA_ALL_QUANTS. Our custom CUDA \
                 build sets this flag (see scripts/build-linux.sh); if you are running \
                 a Homebrew or official release llama.cpp binary this will crash every \
                 time. Track ggml-org/llama.cpp#20866.",
            KvCacheWarning::QuantizedVBreaksMetalFaFallback =>
                "Known upstream bug: Metal crashes on quantized V cache when Flash \
                 Attention falls back to CPU. All Apple Silicon (M1+) supports Metal FA \
                 and is not affected. If you are seeing this on Apple Silicon, please \
                 file a mesh-llm bug — it should not happen. Older Intel Macs or any \
                 host without Metal FA will trip this. Track ggml-org/llama.cpp#21450.",
        }
    }
}

/// Scan a log tail for a known upstream crash signature and return the
/// matching warning, if any. Used for post-mortem diagnostics when
/// llama-server exits before becoming healthy.
pub(crate) fn detect_known_crash_signature(log_tail: &str) -> Option<KvCacheWarning> {
    let candidates = [
        KvCacheWarning::QuantizedVBreaksMetalFaFallback,
        KvCacheWarning::MismatchedQuantNeedsCudaFaAllQuants,
    ];
    candidates
        .into_iter()
        .find(|w| w.crash_signatures().iter().any(|sig| log_tail.contains(sig)))
}

fn emit_kv_cache_warning(warning: KvCacheWarning, k: KvType, v: KvType) {
    match warning {
        KvCacheWarning::MismatchedQuantNeedsCudaFaAllQuants => tracing::warn!(
            "KV cache K/V types mismatched and both quantized ({}/{}); the CUDA \
             FA kernel selector returns BEST_FATTN_KERNEL_NONE unless llama.cpp is \
             built with -DGGML_CUDA_FA_ALL_QUANTS=ON. Our own build sets this flag \
             (see scripts/build-linux.sh), but standard Homebrew or release binaries \
             will crash on FA ops. Track ggml-org/llama.cpp#20866 for the upstream fix.",
            k.as_arg(),
            v.as_arg()
        ),
        KvCacheWarning::QuantizedVBreaksMetalFaFallback => tracing::warn!(
            "KV cache V is quantized ({}); requires -fa on at runtime. On Apple \
             Silicon, if Metal Flash Attention is unavailable, llama-server will \
             crash with 'quantized V cache requires Flash Attention'. Track \
             ggml-org/llama.cpp#21450 for the Metal fix.",
            v.as_arg()
        ),
    }
}

fn log_tail(path: &Path, max_lines: usize) -> String {
    let Ok(contents) = std::fs::read_to_string(path) else {
        return String::new();
    };

    let lines: Vec<&str> = contents.lines().collect();
    let start = lines.len().saturating_sub(max_lines);
    lines[start..].join("\n")
}

fn log_tail_message(path: &Path, max_lines: usize) -> String {
    let tail = log_tail(path, max_lines);
    if tail.is_empty() {
        format!("See {}", path.display())
    } else {
        format!("See {}:\n{}", path.display(), tail)
    }
}

fn parse_available_devices(output: &str) -> Vec<String> {
    let mut devices = Vec::new();
    let mut in_devices = false;

    for line in output.lines() {
        let trimmed = line.trim();
        if trimmed == "available devices:" {
            in_devices = true;
            continue;
        }
        if !in_devices || trimmed.is_empty() {
            continue;
        }
        let Some((name, _rest)) = trimmed.split_once(':') else {
            continue;
        };
        if !name.chars().all(|c| c.is_ascii_alphanumeric()) {
            continue;
        }
        devices.push(name.to_string());
    }

    devices
}

fn probe_available_devices(binary: &Path) -> Vec<String> {
    let Ok(output) = std::process::Command::new(binary)
        .args(["-d", "__mesh_llm_probe_invalid__", "-p", "0"])
        .output()
    else {
        return Vec::new();
    };

    let mut combined = String::from_utf8_lossy(&output.stdout).to_string();
    if !combined.is_empty() && !output.stderr.is_empty() {
        combined.push('\n');
    }
    combined.push_str(&String::from_utf8_lossy(&output.stderr));
    parse_available_devices(&combined)
}

fn preferred_device(available: &[String], flavor: Option<BinaryFlavor>) -> Option<String> {
    let candidates: &[&str] = if let Some(flavor) = flavor {
        flavor.preferred_devices()
    } else {
        &["MTL0", "CUDA0", "HIP0", "Vulkan0", "CPU"]
    };

    for candidate in candidates {
        if available.iter().any(|device| device == candidate) {
            return Some((*candidate).to_string());
        }
    }
    available.first().cloned()
}

fn resolve_device_for_binary(
    binary: &Path,
    flavor: Option<BinaryFlavor>,
    requested: Option<&str>,
) -> Result<String> {
    let available = probe_available_devices(binary);

    if let Some(device) = requested {
        if !available.is_empty() && !available.iter().any(|candidate| candidate == device) {
            anyhow::bail!(
                "requested device {device} is not supported by {}. Available devices: {}",
                binary.display(),
                available.join(", ")
            );
        }
        return Ok(device.to_string());
    }

    if let Some(selected) = preferred_device(&available, flavor) {
        return Ok(selected);
    }

    if let Some(flavor) = flavor {
        return Ok(flavor.primary_device().to_string());
    }

    Ok(detect_device())
}

fn command_has_output(command: &str, args: &[&str]) -> bool {
    let Ok(output) = std::process::Command::new(command).args(args).output() else {
        return false;
    };
    output.status.success()
        && String::from_utf8_lossy(&output.stdout)
            .lines()
            .any(|line| !line.trim().is_empty())
}

/// Start a local rpc-server and return the port it's listening on.
/// Picks an available port automatically.
/// If `gguf_path` is provided, passes `--gguf` so the server loads weights from the local file.
pub async fn start_rpc_server(
    bin_dir: &Path,
    binary_flavor: Option<BinaryFlavor>,
    device: Option<&str>,
    gguf_path: Option<&Path>,
) -> Result<u16> {
    let rpc_server = resolve_binary_path(bin_dir, "rpc-server", binary_flavor)?;

    // Find a free port
    let port = find_free_port().await?;

    let device = resolve_device_for_binary(&rpc_server.path, rpc_server.flavor, device)?;
    let startup_timeout = if device.starts_with("Vulkan") {
        std::time::Duration::from_secs(90)
    } else {
        std::time::Duration::from_secs(15)
    };
    let startup_polls = (startup_timeout.as_millis() / 500) as usize;

    tracing::info!("Starting rpc-server on :{port} (device: {device})");

    let rpc_log = temp_log_path(&format!("rpc-server-{port}.log"));
    let rpc_log_file = std::fs::File::create(&rpc_log)
        .with_context(|| format!("Failed to create rpc-server log file {}", rpc_log.display()))?;
    let rpc_log_file2 = rpc_log_file.try_clone()?;

    let mut args = vec![
        "-d".to_string(),
        device.clone(),
        "-p".to_string(),
        port.to_string(),
    ];
    if let Some(path) = gguf_path {
        args.push("--gguf".to_string());
        args.push(path.to_string_lossy().to_string());
        tracing::info!(
            "rpc-server will load weights from local GGUF: {}",
            path.display()
        );
    }

    let mut child = Command::new(&rpc_server.path)
        .args(&args)
        .stdout(std::process::Stdio::from(rpc_log_file))
        .stderr(std::process::Stdio::from(rpc_log_file2))
        .spawn()
        .with_context(|| {
            format!(
                "Failed to start rpc-server at {}",
                rpc_server.path.display()
            )
        })?;

    // Wait for it to be listening
    for _ in 0..startup_polls {
        if is_port_open(port).await {
            // Detach — let it run in the background
            tokio::spawn(async move {
                let _ = child.wait().await;
            });
            return Ok(port);
        }
        if let Some(status) = child.try_wait().with_context(|| {
            format!(
                "Failed to poll rpc-server status for {}",
                rpc_server.path.display()
            )
        })? {
            let tail = log_tail(&rpc_log, 40);
            let tail_msg = if tail.is_empty() {
                format!("See {}", rpc_log.display())
            } else {
                format!("See {}:\n{}", rpc_log.display(), tail)
            };
            anyhow::bail!(
                "rpc-server exited before listening on port {port} (device: {device}, status: {status}). {tail_msg}"
            );
        }
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    }

    let tail = log_tail(&rpc_log, 40);
    let tail_msg = if tail.is_empty() {
        format!("See {}", rpc_log.display())
    } else {
        format!("See {}:\n{}", rpc_log.display(), tail)
    };
    anyhow::bail!(
        "rpc-server failed to start on port {port} within {}s (device: {device}). {tail_msg}",
        startup_timeout.as_secs()
    );
}

/// Kill orphan rpc-server processes from previous mesh-llm runs.
/// Only kills rpc-servers with PPID 1 (parent died, adopted by init).
/// Safe to call while a live mesh-llm has its own rpc-server child.
pub async fn kill_orphan_rpc_servers() {
    #[cfg(windows)]
    {
        return;
    }

    #[cfg(not(windows))]
    if let Ok(output) = std::process::Command::new("ps")
        .args(["-eo", "pid,ppid,comm"])
        .output()
    {
        let stdout = String::from_utf8_lossy(&output.stdout);
        let mut killed = 0;
        for line in stdout.lines() {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 3 && parts[2].contains("rpc-server") && parts[1] == "1" {
                if let Ok(pid) = parts[0].parse::<u32>() {
                    let _ = std::process::Command::new("kill")
                        .arg(pid.to_string())
                        .status();
                    killed += 1;
                }
            }
        }
        if killed > 0 {
            eprintln!("🧹 Cleaned up {killed} orphan rpc-server process(es)");
            tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        }
    }
}

/// Kill all running llama-server processes.
pub async fn kill_llama_server() {
    let _ = terminate_process_by_name("llama-server");
    // Wait for the process to actually exit and release the port
    for _ in 0..20 {
        tokio::time::sleep(std::time::Duration::from_millis(250)).await;
        if !is_process_running("llama-server") {
            return;
        }
    }
    // Force kill if still alive after 5s
    let _ = force_kill_process_by_name("llama-server");
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
}

async fn terminate_process(pid: u32) {
    let pid_str = pid.to_string();

    #[cfg(not(windows))]
    {
        let _ = std::process::Command::new("kill")
            .args(["-TERM", &pid_str])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status();
        for _ in 0..20 {
            tokio::time::sleep(std::time::Duration::from_millis(250)).await;
            let alive = std::process::Command::new("kill")
                .args(["-0", &pid_str])
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .status()
                .map(|s| s.success())
                .unwrap_or(false);
            if !alive {
                return;
            }
        }
        let _ = std::process::Command::new("kill")
            .args(["-9", &pid_str])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status();
        tokio::time::sleep(std::time::Duration::from_millis(250)).await;
    }

    #[cfg(windows)]
    {
        // Graceful termination via taskkill (sends WM_CLOSE to the process tree)
        let _ = std::process::Command::new("taskkill")
            .args(["/PID", &pid_str, "/T"])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status();
        for _ in 0..20 {
            tokio::time::sleep(std::time::Duration::from_millis(250)).await;
            let alive = std::process::Command::new("tasklist")
                .args(["/FI", &format!("PID eq {pid_str}"), "/NH"])
                .output()
                .map(|o| {
                    o.status.success() && String::from_utf8_lossy(&o.stdout).contains(&pid_str)
                })
                .unwrap_or(false);
            if !alive {
                return;
            }
        }
        // Force-kill if still alive
        let _ = std::process::Command::new("taskkill")
            .args(["/PID", &pid_str, "/T", "/F"])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status();
        tokio::time::sleep(std::time::Duration::from_millis(250)).await;
    }
}

/// Start llama-server with the given model, HTTP port, and RPC tunnel ports.
/// Returns a oneshot receiver that fires when the process exits.
/// `model_bytes` is the total GGUF file size, used to select KV cache quantization:
///   - < 5GB: FP16 (default) — small models, KV cache is tiny
///   - 5-50GB: Q8_0 K + Q4_0 V — keeps attention routing precise (K dominates
///     quality via softmax), compresses values aggressively (~25% less KV memory
///     than Q8_0/Q8_0 with minimal quality impact)
///   - > 50GB: Q4_0 — maximum compression, these models need every byte
pub async fn start_llama_server(
    bin_dir: &Path,
    binary_flavor: Option<BinaryFlavor>,
    spec: ModelLaunchSpec<'_>,
) -> Result<InferenceServerProcess> {
    let model = spec.model;
    let http_port = spec.http_port;
    let tunnel_ports = spec.tunnel_ports;
    let tensor_split = spec.tensor_split;
    let split_mode = spec.split_mode;
    let draft = spec.draft;
    let draft_max = spec.draft_max;
    let model_bytes = spec.model_bytes;
    let my_vram = spec.my_vram;
    let mmproj = spec.mmproj;
    let ctx_size_override = spec.ctx_size_override;
    let total_group_vram = spec.total_group_vram;
    let llama_server = resolve_binary_path(bin_dir, "llama-server", binary_flavor)?;

    anyhow::ensure!(model.exists(), "Model not found at {}", model.display());

    // Build --rpc argument: all tunnel ports as localhost endpoints
    let rpc_endpoints: Vec<String> = tunnel_ports
        .iter()
        .map(|p| format!("127.0.0.1:{p}"))
        .collect();
    let rpc_arg = rpc_endpoints.join(",");

    tracing::info!(
        "Starting llama-server on :{http_port} with model {} and --rpc {}",
        model.display(),
        rpc_arg
    );

    let llama_log = temp_log_path("llama-server.log");
    let log_file = std::fs::File::create(&llama_log).with_context(|| {
        format!(
            "Failed to create llama-server log file {}",
            llama_log.display()
        )
    })?;
    let log_file2 = log_file.try_clone()?;

    // llama-server uses --rpc only for remote workers.
    // Context size: scale to available VRAM on the host node.
    // In split mode (pipeline parallel), each node holds a range of layers
    // and the KV cache for those layers is allocated on the same device.
    // So both weights and KV are distributed. The host only needs VRAM for
    // its share of weights + its share of KV. We estimate the host's weight
    // share proportionally and let llama-server pick the largest -c that fits.
    let host_model_bytes = if let Some(group_vram) = total_group_vram {
        // Split mode: host holds its share of the weights
        if group_vram > 0 {
            let host_fraction = my_vram as f64 / group_vram as f64;
            (model_bytes as f64 * host_fraction) as u64
        } else {
            model_bytes
        }
    } else {
        // Local mode: host holds all weights
        model_bytes
    };
    let vram_after_model = my_vram.saturating_sub(host_model_bytes);
    let ctx_size = compute_context_size(ctx_size_override, model_bytes, my_vram, total_group_vram);
    tracing::info!(
        "Context size: {ctx_size} tokens (model {:.1}GB, host weights ~{:.1}GB, {:.0}GB VRAM, {:.1}GB free{})",
        model_bytes as f64 / GB as f64,
        host_model_bytes as f64 / GB as f64,
        my_vram as f64 / GB as f64,
        vram_after_model as f64 / GB as f64,
        if total_group_vram.is_some() {
            " [split]"
        } else {
            ""
        }
    );

    let mut args = vec!["-m".to_string(), model.to_string_lossy().to_string()];
    if !tunnel_ports.is_empty() {
        args.push("--rpc".to_string());
        args.push(rpc_arg);
    }
    args.extend_from_slice(&[
        "-ngl".to_string(),
        "99".to_string(),
        "-fa".to_string(),
        "on".to_string(),
        "-fit".to_string(),
        "off".to_string(),
        "--no-mmap".to_string(),
        "--host".to_string(),
        "0.0.0.0".to_string(),
        "--port".to_string(),
        http_port.to_string(),
        "-c".to_string(),
        ctx_size.to_string(),
        // Use deepseek format: thinking goes into reasoning_content field.
        // Goose/OpenAI clients parse this correctly. "none" leaks raw <think>
        // tags into content which is worse.
        "--reasoning-format".to_string(),
        "deepseek".to_string(),
        // Disable thinking by default. Thinking models (Qwen3, MiniMax) burn
        // 15-80s on hidden reasoning for no quality gain on most tasks, and
        // Qwen3.5-9B is completely broken (reasoning consumes all max_tokens).
        // API users can opt-in per-request with:
        //   "chat_template_kwargs": {"enable_thinking": true}
        "--reasoning-budget".to_string(),
        "0".to_string(),
    ]);
    // KV cache quantization — asymmetric K/V strategy.
    //
    // K precision dominates quality: K controls attention routing via softmax,
    // where small errors get exponentially amplified. V errors scale linearly
    // in the weighted sum and are far more tolerant of compression.
    // (See TurboQuant ICLR 2026 / asymmetric K/V findings.)
    //
    // Current tiers:
    //   < 5GB:  leave default (FP16) — small models, KV cache is negligible
    //   5-50GB: K=Q8_0, V=Q4_0 — aggressive asymmetric, ~25% less KV memory
    //           than Q8_0/Q8_0 with minimal quality impact. Relies on two
    //           open upstream bugs being worked around (see caveats below).
    //   > 50GB: Q4_0/Q4_0 — maximum compression, matched quantized types so
    //           no GGML_CUDA_FA_ALL_QUANTS needed. Requires -fa on.
    //
    // Caveats for the 5-50GB K=Q8_0/V=Q4_0 tier as of 2026-04:
    //   - ggml-org/llama.cpp#20866 — asymmetric K/V types require rebuilding
    //     llama.cpp with -DGGML_CUDA_FA_ALL_QUANTS=ON. Standard Homebrew and
    //     release binaries crash with BEST_FATTN_KERNEL_NONE → GGML_ABORT.
    //     Our own CUDA build sets that flag and the post-cmake assertion in
    //     scripts/build-linux.sh keeps it from being dropped silently. Users
    //     pointing mesh-llm at an external llama.cpp binary can still trip
    //     this; `detect_known_crash_signature` attributes the failure.
    //   - ggml-org/llama.cpp#21450 — Metal crashes on mixed quantized KV when
    //     Flash Attention falls back to CPU ("quantized V cache requires Flash
    //     Attention"). All Apple Silicon (M1+) supports Metal FA and is not
    //     affected in practice. Older Intel Macs or any host without Metal
    //     FA trip this; `detect_known_crash_signature` attributes the
    //     failure when it matches.
    //
    // TODO(ggml-org/llama.cpp#20866, ggml-org/llama.cpp#21450): once both are
    // closed upstream and our fork is rebased past the fixes, remove the
    // corresponding KvCacheWarning variants, the build assertion, and this
    // caveat block.
    KvCacheQuant::for_model_size(model_bytes).append_args(&mut args, model_bytes);
    if let Some(ts) = tensor_split {
        args.push("--tensor-split".to_string());
        args.push(ts.to_string());
    }
    if let Some(mode) = split_mode {
        args.push("--split-mode".to_string());
        args.push(mode.as_arg().to_string());
        match mode {
            SplitMode::Layer => {
                tracing::info!(
                    "Split mode: {} (layer-based / pipeline parallelism)",
                    mode.as_arg()
                );
            }
            SplitMode::Row => {
                tracing::info!(
                    "Split mode: {} (tensor parallelism across local GPUs)",
                    mode.as_arg()
                );
            }
        }
    }
    let local_device = resolve_device_for_binary(&llama_server.path, llama_server.flavor, None)?;
    if let Some(draft_path) = draft {
        if draft_path.exists() {
            if local_device != "CPU" {
                args.push("-md".to_string());
                args.push(draft_path.to_string_lossy().to_string());
                args.push("-ngld".to_string());
                args.push("99".to_string());
                args.push("--device-draft".to_string());
                args.push(local_device.clone());
                args.push("--draft-max".to_string());
                args.push(draft_max.to_string());
                tracing::info!(
                    "Speculative decoding: draft={}, draft-max={}, device={}",
                    draft_path.display(),
                    draft_max,
                    local_device
                );
            } else {
                tracing::warn!(
                    "Draft model present at {} but no GPU backend detected, skipping speculative decoding",
                    draft_path.display()
                );
            }
        } else {
            tracing::warn!(
                "Draft model not found at {}, skipping speculative decoding",
                draft_path.display()
            );
        }
    }
    if let Some(proj) = mmproj {
        if proj.exists() {
            args.push("--mmproj".to_string());
            args.push(proj.to_string_lossy().to_string());
            // Vision images can produce large token batches — need ubatch >= 2048
            args.push("--ubatch-size".to_string());
            args.push("2048".to_string());
            tracing::info!("Vision: mmproj={}", proj.display());
        } else {
            tracing::warn!("mmproj not found at {}, skipping vision", proj.display());
        }
    }
    let mut child = Command::new(&llama_server.path)
        .args(&args)
        .stdout(std::process::Stdio::from(log_file))
        .stderr(std::process::Stdio::from(log_file2))
        .spawn()
        .with_context(|| {
            format!(
                "Failed to start llama-server at {}",
                llama_server.path.display()
            )
        })?;

    // Wait for health check
    let url = format!("http://127.0.0.1:{http_port}/health");
    for i in 0..600 {
        if i > 0 && i % 10 == 0 {
            let bytes = crate::network::tunnel::bytes_transferred();
            let kb = bytes as f64 / 1024.0;
            let mb = bytes as f64 / (1024.0 * 1024.0);
            let gb = bytes as f64 / (1024.0 * 1024.0 * 1024.0);
            let transferred = if gb >= 1.0 {
                format!("{gb:.1} GB")
            } else if mb >= 1.0 {
                format!("{mb:.1} MB")
            } else {
                format!("{kb:.0} KB")
            };
            tracing::info!(
                "Still waiting for llama-server to load model... ({i}s, {transferred} transferred)"
            );
        }
        if let Some(status) = child.try_wait().with_context(|| {
            format!(
                "Failed to poll llama-server status for {}",
                llama_server.path.display()
            )
        })? {
            let tail = log_tail(&llama_log, 80);
            let hint = detect_known_crash_signature(&tail)
                .map(|w| format!("\n\n{}", w.post_mortem_hint()))
                .unwrap_or_default();
            let tail_msg = if tail.is_empty() {
                format!("See {}", llama_log.display())
            } else {
                format!("See {}:\n{}", llama_log.display(), tail)
            };
            anyhow::bail!(
                "llama-server exited before becoming healthy on port {http_port} (status: {status}). {}{}",
                tail_msg,
                hint
            );
        }
        if reqwest_health_check(&url).await {
            let pid = child
                .id()
                .context("llama-server started but did not expose a PID")?;
            let expected_exit = Arc::new(AtomicBool::new(false));
            let handle = InferenceServerHandle {
                pid,
                expected_exit: expected_exit.clone(),
            };
            let (death_tx, death_rx) = tokio::sync::oneshot::channel();
            tokio::spawn(async move {
                let _ = child.wait().await;
                if !expected_exit.load(Ordering::Relaxed) {
                    eprintln!("⚠️  llama-server process exited unexpectedly");
                }
                let _ = death_tx.send(());
            });
            return Ok(InferenceServerProcess {
                handle,
                death_rx,
                context_length: ctx_size,
            });
        }
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
    }

    anyhow::bail!(
        "llama-server failed to become healthy within 600s. {}",
        log_tail_message(&llama_log, 80)
    );
}

/// Find an available TCP port
async fn find_free_port() -> Result<u16> {
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let port = listener.local_addr()?.port();
    drop(listener);
    Ok(port)
}

/// Check if a port is accepting connections
async fn is_port_open(port: u16) -> bool {
    tokio::net::TcpStream::connect(format!("127.0.0.1:{port}"))
        .await
        .is_ok()
}

pub fn terminate_process_by_name(name: &str) -> bool {
    kill_process_by_name(name, false)
}

pub fn force_kill_process_by_name(name: &str) -> bool {
    kill_process_by_name(name, true)
}

fn kill_process_by_name(name: &str, force: bool) -> bool {
    #[cfg(windows)]
    {
        let image = platform_bin_name(name);
        let mut cmd = std::process::Command::new("taskkill");
        if force {
            cmd.arg("/F");
        }
        cmd.args(["/IM", &image]);
        cmd.status().is_ok_and(|status| status.success())
    }

    #[cfg(not(windows))]
    {
        let mut cmd = std::process::Command::new("pkill");
        if force {
            cmd.arg("-9");
        }
        cmd.args(["-f", name]);
        cmd.status().is_ok_and(|status| status.success())
    }
}

fn is_process_running(name: &str) -> bool {
    #[cfg(windows)]
    {
        let image = platform_bin_name(name);
        std::process::Command::new("tasklist")
            .args(["/FI", &format!("IMAGENAME eq {image}")])
            .output()
            .map(|output| {
                output.status.success()
                    && String::from_utf8_lossy(&output.stdout)
                        .to_ascii_lowercase()
                        .contains(&image.to_ascii_lowercase())
            })
            .unwrap_or(false)
    }

    #[cfg(not(windows))]
    {
        std::process::Command::new("pgrep")
            .args(["-f", name])
            .output()
            .map(|output| output.status.success() && !output.stdout.is_empty())
            .unwrap_or(false)
    }
}

/// Detect the best available compute device
fn detect_device() -> String {
    if cfg!(target_os = "macos") {
        return "MTL0".to_string();
    }

    // Linux: check for NVIDIA CUDA
    if command_has_output("nvidia-smi", &["--query-gpu=name", "--format=csv,noheader"]) {
        return "CUDA0".to_string();
    }

    // Linux: check for NVIDIA Tegra/Jetson (tegrastats — Jetson AGX/NX devices support CUDA)
    // nvidia-smi is absent on Tegra; tegrastats is the canonical hardware stats tool.
    if let Ok(mut child) = std::process::Command::new("tegrastats")
        .args(["--interval", "1"])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
    {
        let _ = child.kill();
        let _ = child.wait();
        return "CUDA0".to_string();
    }

    // ROCm/HIP
    if has_rocm_backend() {
        return "HIP0".to_string();
    }

    // Vulkan
    if command_succeeds("vulkaninfo", &["--summary"]) {
        return "Vulkan0".to_string();
    }

    "CPU".to_string()
}

fn has_rocm_backend() -> bool {
    #[cfg(windows)]
    {
        if std::env::var_os("ROCM_PATH").is_some() || std::env::var_os("HIP_PATH").is_some() {
            return true;
        }
        if let Some(program_files) = std::env::var_os("ProgramFiles") {
            let base = PathBuf::from(program_files).join("AMD");
            if base.join("ROCm").exists() || base.join("HIP").exists() {
                return true;
            }
        }
        command_has_output("hipInfo", &[]) || command_has_output("hipconfig", &[])
    }

    #[cfg(not(windows))]
    {
        command_has_output("rocm-smi", &["--showproductname"])
            || command_has_output("rocminfo", &[])
    }
}

fn command_succeeds(command: &str, args: &[&str]) -> bool {
    std::process::Command::new(command)
        .args(args)
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

/// Simple HTTP health check (avoid adding reqwest as a dep — just use TCP + raw HTTP)
async fn reqwest_health_check(url: &str) -> bool {
    // Parse host:port from URL
    let url = url.strip_prefix("http://").unwrap_or(url);
    let (host_port, path) = url.split_once('/').unwrap_or((url, ""));
    let path = format!("/{path}");

    let Ok(mut stream) = tokio::net::TcpStream::connect(host_port).await else {
        return false;
    };

    let request = format!("GET {path} HTTP/1.1\r\nHost: {host_port}\r\nConnection: close\r\n\r\n");
    if stream.write_all(request.as_bytes()).await.is_err() {
        return false;
    }

    let mut response = vec![0u8; 1024];
    let Ok(n) = stream.read(&mut response).await else {
        return false;
    };

    let response = String::from_utf8_lossy(&response[..n]);
    response.contains("200 OK")
}

use tokio::io::{AsyncReadExt, AsyncWriteExt};

#[cfg(test)]
mod tests {
    use super::{
        compute_context_size, parse_available_devices, preferred_device, temp_log_path,
        BinaryFlavor, KvCacheQuant, KvCacheWarning, KvType, SplitMode, GB,
    };
    use std::path::Path;

    #[test]
    fn kv_quant_small_model_is_plain_f16() {
        let quant = KvCacheQuant::for_model_size(1 * GB);
        assert_eq!(quant.k_type, KvType::F16);
        assert_eq!(quant.v_type, KvType::F16);
        assert!(
            quant.validation_warnings().is_empty(),
            "small-model default should not trigger any upstream bug warnings"
        );
    }

    #[test]
    fn kv_quant_medium_model_is_aggressive_asymmetric() {
        let quant = KvCacheQuant::for_model_size(20 * GB);
        assert_eq!(
            quant.k_type,
            KvType::Q8_0,
            "medium-tier K should be Q8_0 for attention routing precision"
        );
        assert_eq!(
            quant.v_type,
            KvType::Q4_0,
            "medium-tier V is Q4_0 for 25% memory savings over Q8_0/Q8_0. \
             This intentionally opts into two open upstream bugs (#20866 on CUDA, \
             #21450 on Metal FA fallback); our own CUDA build sets GGML_CUDA_FA_ALL_QUANTS \
             and Metal FA is available on all M1+ Macs. If you change this, also update \
             kv_quant_medium_model_emits_expected_warnings and the detect_known_crash_signature \
             wiring."
        );
    }

    #[test]
    fn kv_quant_medium_model_emits_expected_warnings() {
        let warnings = KvCacheQuant::for_model_size(20 * GB).validation_warnings();
        assert!(
            warnings.contains(&KvCacheWarning::MismatchedQuantNeedsCudaFaAllQuants),
            "medium tier must flag #20866 so the startup log is clear about the CUDA \
             build requirement, got {warnings:?}"
        );
        assert!(
            warnings.contains(&KvCacheWarning::QuantizedVBreaksMetalFaFallback),
            "medium tier must flag #21450 so a user hitting the rare Metal-CPU-FA-fallback \
             crash can connect their failure to the open upstream issue, got {warnings:?}"
        );
    }

    #[test]
    fn kv_quant_large_model_is_matched_q4_0() {
        let quant = KvCacheQuant::for_model_size(100 * GB);
        assert_eq!(quant.k_type, KvType::Q4_0);
        assert_eq!(
            quant.v_type,
            KvType::Q4_0,
            "large-tier K and V must be the same quantized type to avoid #20866"
        );
    }

    #[test]
    fn kv_quant_default_tier_warnings_are_intentional() {
        let expected: &[(u64, &[KvCacheWarning])] = &[
            (1, &[]),
            (4, &[]),
            (5, &[
                KvCacheWarning::QuantizedVBreaksMetalFaFallback,
                KvCacheWarning::MismatchedQuantNeedsCudaFaAllQuants,
            ]),
            (20, &[
                KvCacheWarning::QuantizedVBreaksMetalFaFallback,
                KvCacheWarning::MismatchedQuantNeedsCudaFaAllQuants,
            ]),
            (49, &[
                KvCacheWarning::QuantizedVBreaksMetalFaFallback,
                KvCacheWarning::MismatchedQuantNeedsCudaFaAllQuants,
            ]),
            (50, &[]),
            (100, &[]),
        ];

        for (size_gb, expected_warnings) in expected {
            let actual = KvCacheQuant::for_model_size(size_gb * GB).validation_warnings();
            assert_eq!(
                actual, *expected_warnings,
                "tier for {size_gb}GB drifted from the documented warning set. \
                 If you changed the defaults, update this test to the new expected \
                 warnings and verify detect_known_crash_signature still maps them."
            );
        }
    }

    #[test]
    fn kv_quant_mismatched_quant_flags_cuda_bug() {
        let quant = KvCacheQuant {
            k_type: KvType::Q8_0,
            v_type: KvType::Q4_0,
        };
        let warnings = quant.validation_warnings();
        assert!(
            warnings.contains(&KvCacheWarning::MismatchedQuantNeedsCudaFaAllQuants),
            "K=Q8_0/V=Q4_0 must flag #20866, got {warnings:?}"
        );
    }

    #[test]
    fn kv_quant_quantized_v_flags_metal_bug() {
        let quant = KvCacheQuant {
            k_type: KvType::F16,
            v_type: KvType::Q4_0,
        };
        let warnings = quant.validation_warnings();
        assert!(
            warnings.contains(&KvCacheWarning::QuantizedVBreaksMetalFaFallback),
            "any quantized V must flag #21450, got {warnings:?}"
        );
    }

    #[test]
    fn kv_quant_tier_boundaries_are_exact() {
        let just_below_medium = KvCacheQuant::for_model_size(KvCacheQuant::MEDIUM_TIER_MIN_BYTES - 1);
        assert_eq!(just_below_medium.k_type, KvType::F16);
        assert_eq!(just_below_medium.v_type, KvType::F16);
        let at_medium = KvCacheQuant::for_model_size(KvCacheQuant::MEDIUM_TIER_MIN_BYTES);
        assert_eq!(at_medium.k_type, KvType::Q8_0);
        assert_eq!(at_medium.v_type, KvType::Q4_0);
        let just_below_large = KvCacheQuant::for_model_size(KvCacheQuant::LARGE_TIER_MIN_BYTES - 1);
        assert_eq!(just_below_large.k_type, KvType::Q8_0);
        assert_eq!(just_below_large.v_type, KvType::Q4_0);
        let at_large = KvCacheQuant::for_model_size(KvCacheQuant::LARGE_TIER_MIN_BYTES);
        assert_eq!(at_large.k_type, KvType::Q4_0);
        assert_eq!(at_large.v_type, KvType::Q4_0);
    }

    #[test]
    fn kv_quant_append_args_emits_cache_flags_for_quantized_tiers() {
        let mut args: Vec<String> = Vec::new();
        KvCacheQuant::for_model_size(20 * GB).append_args(&mut args, 20 * GB);
        assert_eq!(
            args,
            vec![
                "--cache-type-k".to_string(),
                "q8_0".to_string(),
                "--cache-type-v".to_string(),
                "q4_0".to_string(),
            ]
        );
    }

    #[test]
    fn kv_quant_append_args_is_silent_for_f16_default() {
        let mut args: Vec<String> = Vec::new();
        KvCacheQuant::for_model_size(1 * GB).append_args(&mut args, 1 * GB);
        assert!(
            args.is_empty(),
            "f16/f16 must not emit --cache-type-* flags (it's the llama-server default): {args:?}"
        );
    }

    #[test]
    fn detect_crash_signature_matches_cuda_fa_abort() {
        let log = "ggml_backend_cuda_flash_attn_ext\n\
                   /llama.cpp/ggml/src/ggml-cuda/fattn.cu:504: fatal error\n\
                   BEST_FATTN_KERNEL_NONE";
        assert_eq!(
            super::detect_known_crash_signature(log),
            Some(KvCacheWarning::MismatchedQuantNeedsCudaFaAllQuants)
        );
    }

    #[test]
    fn detect_crash_signature_matches_metal_v_cache_error_both_phrasings() {
        let new_phrasing = "common_init_from_params: quantized V cache requires Flash Attention";
        let old_phrasing =
            "llama_init_from_model: V cache quantization requires flash_attn";
        assert_eq!(
            super::detect_known_crash_signature(new_phrasing),
            Some(KvCacheWarning::QuantizedVBreaksMetalFaFallback)
        );
        assert_eq!(
            super::detect_known_crash_signature(old_phrasing),
            Some(KvCacheWarning::QuantizedVBreaksMetalFaFallback)
        );
    }

    #[test]
    fn detect_crash_signature_returns_none_for_unrelated_logs() {
        let log = "llama_context: n_ctx = 65536\n\
                   sched_reserve: graph nodes = 3849\n\
                   common_init_from_params: warming up the model with an empty run";
        assert_eq!(super::detect_known_crash_signature(log), None);
    }

    #[test]
    fn post_mortem_hint_references_issue_number() {
        assert!(KvCacheWarning::MismatchedQuantNeedsCudaFaAllQuants
            .post_mortem_hint()
            .contains("#20866"));
        assert!(KvCacheWarning::QuantizedVBreaksMetalFaFallback
            .post_mortem_hint()
            .contains("#21450"));
    }

    #[test]
    fn parse_available_devices_ignores_non_device_lines() {
        let output = r#"
error: unknown device: HIP0
available devices:
No devices found
  Vulkan0: AMD Radeon RX 9070 XT (16304 MiB, 13737 MiB free)
  CPU: AMD Ryzen 7 7800X3D 8-Core Processor (192857 MiB, 192857 MiB free)
"#;

        assert_eq!(
            parse_available_devices(output),
            vec!["Vulkan0".to_string(), "CPU".to_string()]
        );
    }

    #[test]
    fn preferred_device_picks_vulkan_when_that_is_all_binary_supports() {
        let available = vec!["Vulkan0".to_string(), "CPU".to_string()];
        assert_eq!(
            preferred_device(&available, Some(BinaryFlavor::Vulkan)),
            Some("Vulkan0".to_string())
        );
    }

    #[test]
    fn infer_binary_flavor_from_filename() {
        assert_eq!(
            super::infer_binary_flavor("rpc-server", Path::new("rpc-server-vulkan")),
            Some(BinaryFlavor::Vulkan)
        );
        #[cfg(windows)]
        assert_eq!(
            super::infer_binary_flavor("rpc-server", Path::new("rpc-server-vulkan.exe")),
            Some(BinaryFlavor::Vulkan)
        );
        assert_eq!(
            super::infer_binary_flavor("rpc-server", Path::new("rpc-server")),
            None
        );
    }

    #[cfg(windows)]
    #[test]
    fn platform_bin_name_preserves_existing_exe_suffix_case_insensitively() {
        assert_eq!(super::platform_bin_name("rpc-server.EXE"), "rpc-server.EXE");
    }

    #[test]
    fn compute_context_size_prefers_explicit_override() {
        assert_eq!(
            compute_context_size(Some(24576), 8_000_000_000, 48_000_000_000, None),
            24576
        );
    }

    #[test]
    fn compute_context_size_uses_full_model_bytes_in_local_mode() {
        assert_eq!(
            compute_context_size(None, 10_000_000_000, 22_000_000_000, None),
            32768
        );
        assert_eq!(
            compute_context_size(None, 10_000_000_000, 13_000_000_000, None),
            8192
        );
    }

    #[test]
    fn compute_context_size_accounts_for_split_host_weight_share() {
        let model_bytes = 40_000_000_000;
        let my_vram = 20_000_000_000;
        let total_group_vram = Some(80_000_000_000);

        assert_eq!(
            compute_context_size(None, model_bytes, my_vram, total_group_vram),
            16384
        );
    }

    #[test]
    fn temp_log_path_includes_pid_and_suffix() {
        let suffix = "rpc-server.log";
        let path = temp_log_path(suffix);
        let file_name = path
            .file_name()
            .expect("temp_log_path should produce a filename")
            .to_string_lossy();
        let pid = std::process::id().to_string();

        assert!(
            file_name.contains(&pid),
            "expected filename '{file_name}' to contain current pid '{pid}'"
        );
        assert!(
            file_name.contains(suffix),
            "expected filename '{file_name}' to contain suffix '{suffix}'"
        );
    }

    // ── SplitMode ──

    #[test]
    fn split_mode_layer_arg() {
        assert_eq!(SplitMode::Layer.as_arg(), "layer");
    }

    #[test]
    fn split_mode_row_arg() {
        assert_eq!(SplitMode::Row.as_arg(), "row");
    }
}
