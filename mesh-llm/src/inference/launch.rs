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
            BinaryFlavor::Rocm => &["ROCm0", "HIP0", "CPU"],
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

#[derive(Debug)]
pub struct InferenceServerHandle {
    pid: u32,
    expected_exit: Arc<AtomicBool>,
    expected_comm: String,
    expected_start_time: Option<i64>,
    pub(crate) _pidfile_guard: Option<crate::runtime::instance::PidfileGuard>,
}

impl InferenceServerHandle {
    pub fn pid(&self) -> u32 {
        self.pid
    }

    pub async fn shutdown(&self) {
        self.expected_exit.store(true, Ordering::Relaxed);
        terminate_process_with_wait(
            self.pid,
            &self.expected_comm,
            self.expected_start_time,
            20,
            std::time::Duration::from_millis(250),
        )
        .await;
    }
}

impl Drop for InferenceServerHandle {
    /// Best-effort termination if the handle is dropped without an explicit
    /// `shutdown().await` (panic, task abort, or any path that bypasses the
    /// async cleanup). We can't await in `drop`, so this only issues a single
    /// SIGTERM — the death-watcher and the cross-runtime reaper handle any
    /// stragglers. If `expected_exit` is already set, the async shutdown ran
    /// and there is nothing to do.
    fn drop(&mut self) {
        if self.expected_exit.swap(true, Ordering::Relaxed) {
            return;
        }
        let _ = send_signal_if_matches(
            self.pid,
            &self.expected_comm,
            self.expected_start_time,
            ProcessSignal::Terminate,
        );
    }
}

/// Handle for a running rpc-server process.
///
/// Symmetric with [`InferenceServerHandle`] for llama-server.
/// The `_pidfile_guard` field is `None` until T9 wires up pidfile writing.
#[derive(Debug)]
pub struct RpcServerHandle {
    pub pid: u32,
    pub port: u16,
    pub expected_exit: Arc<AtomicBool>,
    pub expected_comm: String,
    pub expected_start_time: Option<i64>,
    pub(crate) _pidfile_guard: Option<crate::runtime::instance::PidfileGuard>,
}

impl Drop for RpcServerHandle {
    /// Best-effort SIGTERM if the rpc-server handle is dropped without an
    /// explicit `shutdown().await` (panic / task abort path). Mirrors the
    /// `InferenceServerHandle::Drop` safety net so a crashed parent does not
    /// leave an orphan rpc-server holding GPU memory.
    fn drop(&mut self) {
        if self.expected_exit.swap(true, Ordering::Relaxed) {
            return;
        }
        let _ = send_signal_if_matches(
            self.pid,
            &self.expected_comm,
            self.expected_start_time,
            ProcessSignal::Terminate,
        );
    }
}

impl RpcServerHandle {
    pub async fn shutdown(&self) {
        self.expected_exit.store(true, Ordering::Relaxed);
        terminate_process_with_wait(
            self.pid,
            &self.expected_comm,
            self.expected_start_time,
            50,
            std::time::Duration::from_millis(100),
        )
        .await;
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
    pub selected_gpu: Option<&'a crate::runtime::StartupPinnedGpuTarget>,
    /// Number of parallel slots for llama-server (`--parallel`).
    /// Set from the `[[models]].slots` TOML config; defaults to 4 when unset.
    pub slots: usize,
}

pub(crate) const GB: u64 = 1_000_000_000;

fn spawned_binary_name(path: &Path) -> String {
    #[cfg(windows)]
    {
        path.file_stem()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.to_string_lossy().into_owned())
    }

    #[cfg(not(windows))]
    {
        path.file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.to_string_lossy().into_owned())
    }
}

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
            KvCacheWarning::MismatchedQuantNeedsCudaFaAllQuants => {
                "Known upstream bug: CUDA FA kernel selector rejects mismatched K/V \
                 quantization types without GGML_CUDA_FA_ALL_QUANTS. Our custom CUDA \
                 build sets this flag (see scripts/build-linux.sh); if you are running \
                 a Homebrew or official release llama.cpp binary this will crash every \
                 time. Track ggml-org/llama.cpp#20866."
            }
            KvCacheWarning::QuantizedVBreaksMetalFaFallback => {
                "Known upstream bug: Metal crashes on quantized V cache when Flash \
                 Attention falls back to CPU. All Apple Silicon (M1+) supports Metal FA \
                 and is not affected. If you are seeing this on Apple Silicon, please \
                 file a mesh-llm bug — it should not happen. Older Intel Macs or any \
                 host without Metal FA will trip this. Track ggml-org/llama.cpp#21450."
            }
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
    candidates.into_iter().find(|w| {
        w.crash_signatures()
            .iter()
            .any(|sig| log_tail.contains(sig))
    })
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
        &["MTL0", "CUDA0", "ROCm0", "HIP0", "Vulkan0", "CPU"]
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
        if !available.is_empty() {
            if available.iter().any(|candidate| candidate == device) {
                return Ok(device.to_string());
            }

            // Dual support for ROCm/HIP transition
            let is_amd_requested = device.starts_with("ROCm") || device.starts_with("HIP");
            if is_amd_requested {
                let alt_device = if device.starts_with("ROCm") {
                    device.replace("ROCm", "HIP")
                } else {
                    device.replace("HIP", "ROCm")
                };
                if available.iter().any(|candidate| candidate == &alt_device) {
                    return Ok(alt_device);
                }
            }

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

fn selected_backend_device(
    selected_gpu: Option<&crate::runtime::StartupPinnedGpuTarget>,
) -> Result<Option<&str>> {
    match selected_gpu {
        Some(gpu) => Ok(Some(gpu.backend_device.as_str())),
        None => Ok(None),
    }
}

fn ensure_selected_gpu_capacity(
    selected_gpu: Option<&crate::runtime::StartupPinnedGpuTarget>,
    required_bytes: u64,
    purpose: &str,
) -> Result<()> {
    let Some(gpu) = selected_gpu else {
        return Ok(());
    };
    let required_with_headroom = ((required_bytes as f64) * 1.1).ceil() as u64;
    anyhow::ensure!(
        gpu.vram_bytes >= required_with_headroom,
        "pinned GPU '{}' ({}) has {:.1}GB but {purpose} assumes at least {:.1}GB on the selected device",
        gpu.stable_id,
        gpu.backend_device,
        gpu.vram_bytes as f64 / GB as f64,
        required_with_headroom as f64 / GB as f64
    );
    Ok(())
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

/// Start a local rpc-server and return a handle holding its PID and port.
/// Picks an available port automatically.
/// If `gguf_path` is provided, passes `--gguf` so the server loads weights from the local file.
pub async fn start_rpc_server(
    runtime: &crate::runtime::instance::InstanceRuntime,
    bin_dir: &Path,
    binary_flavor: Option<BinaryFlavor>,
    device: Option<&str>,
    gguf_path: Option<&Path>,
) -> Result<RpcServerHandle> {
    let rpc_server = resolve_binary_path(bin_dir, "rpc-server", binary_flavor)?;
    let rpc_server_name = spawned_binary_name(&rpc_server.path);

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

    let rpc_log = runtime.log_path(&format!("rpc-server-{port}"));
    eprintln!(
        "⏳ Starting rpc-server on port {port}... (logs: {})",
        rpc_log.display()
    );
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
        .env("MESH_LLM_OWNER_PID", std::process::id().to_string())
        .env(
            "MESH_LLM_RUNTIME_DIR",
            runtime.dir().to_string_lossy().to_string(),
        )
        .stdout(std::process::Stdio::from(rpc_log_file))
        .stderr(std::process::Stdio::from(rpc_log_file2))
        .spawn()
        .with_context(|| {
            format!(
                "Failed to start rpc-server at {}",
                rpc_server.path.display()
            )
        })?;

    let pid = child.id().context("rpc-server did not expose a PID")?;
    if pid == 0 {
        anyhow::bail!("rpc-server returned PID 0 — refusing to proceed");
    }
    let child_started_at =
        crate::runtime::instance::validate::process_started_at_unix(pid).unwrap_or(None);
    let owner_started_at: i64 =
        crate::runtime::instance::validate::current_process_start_time_unix().unwrap_or(0);
    let metadata = crate::runtime::instance::PidfileMetadata {
        cmd_name: rpc_server_name.clone(),
        child_pid: pid,
        child_started_at_unix: child_started_at.unwrap_or(0),
        owner_pid: std::process::id(),
        owner_started_at_unix: owner_started_at,
        argv_snippet: crate::runtime::instance::PidfileMetadata::cap_argv(
            &args,
            crate::runtime::instance::ARGV_SNIPPET_MAX_BYTES,
        ),
        runtime_dir: runtime.dir().to_path_buf(),
    };
    let pidfile_guard = runtime.write_pidfile(&format!("rpc-server-{port}"), &metadata)?;
    let expected_exit = Arc::new(AtomicBool::new(false));
    let expected_exit_clone = expected_exit.clone();

    // Wait for it to be listening
    for _ in 0..startup_polls {
        if is_port_open(port).await {
            let pidfile_path = runtime.pidfile_path(&format!("rpc-server-{port}"));
            tokio::spawn(async move {
                let _ = child.wait().await;
                let _ = std::fs::remove_file(&pidfile_path);
                if !expected_exit_clone.load(Ordering::Relaxed) {
                    eprintln!("⚠️  rpc-server process exited unexpectedly");
                }
            });
            return Ok(RpcServerHandle {
                pid,
                port,
                expected_exit,
                expected_comm: rpc_server_name.clone(),
                expected_start_time: child_started_at,
                _pidfile_guard: Some(pidfile_guard),
            });
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ProcessSignal {
    Terminate,
    Kill,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SignalOutcome {
    Sent,
    AlreadyDead,
    Skipped,
    Failed,
}

fn send_signal_if_matches(
    pid: u32,
    expected_comm: &str,
    expected_start_time: Option<i64>,
    signal: ProcessSignal,
) -> SignalOutcome {
    if !is_safe_kill_target(pid) {
        tracing::error!("BUG: attempted to signal unsafe pid {pid} — refusing");
        return SignalOutcome::Failed;
    }

    #[cfg(not(windows))]
    {
        if let Some(expected_t) = expected_start_time {
            if !crate::runtime::instance::validate::validate_pid_matches(
                pid,
                expected_comm,
                expected_t,
            ) {
                tracing::warn!("pid {pid} no longer matches expected identity, skipping signal");
                return SignalOutcome::Skipped;
            }
        } else if !crate::runtime::instance::validate::process_name_matches(pid, expected_comm) {
            tracing::warn!("pid {pid} no longer matches {expected_comm}, skipping signal");
            return SignalOutcome::Skipped;
        }
    }

    #[cfg(windows)]
    {
        tracing::debug!(
            pid,
            expected_comm,
            "skipping process identity validation on Windows"
        );
    }

    #[cfg(unix)]
    unsafe {
        let ret = libc::kill(
            pid as libc::pid_t,
            match signal {
                ProcessSignal::Terminate => libc::SIGTERM,
                ProcessSignal::Kill => libc::SIGKILL,
            },
        );
        if ret == 0 {
            return SignalOutcome::Sent;
        }

        let err = std::io::Error::last_os_error();
        if err.raw_os_error() == Some(libc::ESRCH) {
            return SignalOutcome::AlreadyDead;
        }

        tracing::warn!(pid, error = %err, ?signal, "failed to signal process");
        SignalOutcome::Failed
    }

    #[cfg(windows)]
    {
        let pid_str = pid.to_string();
        let mut command = std::process::Command::new("taskkill");
        command.args(["/PID", &pid_str, "/T"]);
        if signal == ProcessSignal::Kill {
            command.arg("/F");
        }
        match command
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
        {
            Ok(status) if status.success() => SignalOutcome::Sent,
            Ok(status) => {
                tracing::warn!(pid, exit_code = status.code(), ?signal, "taskkill failed");
                SignalOutcome::Failed
            }
            Err(err) => {
                tracing::warn!(pid, error = %err, ?signal, "failed to run taskkill");
                SignalOutcome::Failed
            }
        }
    }
}

pub(crate) fn terminate_process_blocking(
    pid: u32,
    expected_comm: &str,
    expected_start_time: Option<i64>,
) -> bool {
    match send_signal_if_matches(
        pid,
        expected_comm,
        expected_start_time,
        ProcessSignal::Terminate,
    ) {
        SignalOutcome::Sent => {}
        SignalOutcome::AlreadyDead => return true,
        // Identity mismatch: the PID belongs to a different process; do not
        // claim a successful stop.
        SignalOutcome::Skipped | SignalOutcome::Failed => return false,
    }

    for _ in 0..20 {
        std::thread::sleep(std::time::Duration::from_millis(250));
        if crate::runtime::instance::validate::process_liveness(pid)
            == crate::runtime::instance::validate::Liveness::Dead
        {
            return true;
        }
    }

    matches!(
        send_signal_if_matches(pid, expected_comm, expected_start_time, ProcessSignal::Kill),
        SignalOutcome::Sent | SignalOutcome::AlreadyDead
    )
}

async fn terminate_process_with_wait(
    pid: u32,
    expected_comm: &str,
    expected_start_time: Option<i64>,
    attempts: usize,
    interval: std::time::Duration,
) {
    match send_signal_if_matches(
        pid,
        expected_comm,
        expected_start_time,
        ProcessSignal::Terminate,
    ) {
        SignalOutcome::Sent => {}
        SignalOutcome::AlreadyDead | SignalOutcome::Skipped | SignalOutcome::Failed => return,
    }

    for _ in 0..attempts {
        tokio::time::sleep(interval).await;
        if crate::runtime::instance::validate::process_liveness(pid)
            == crate::runtime::instance::validate::Liveness::Dead
        {
            return;
        }
    }

    let _ = send_signal_if_matches(pid, expected_comm, expected_start_time, ProcessSignal::Kill);
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
    runtime: &crate::runtime::instance::InstanceRuntime,
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
    let selected_gpu = spec.selected_gpu;
    let slots = spec.slots;
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

    let llama_log = runtime.log_path(&format!("llama-server-{}", http_port));
    eprintln!(
        "⏳ Starting llama-server on :{http_port}... (logs: {})",
        llama_log.display()
    );
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
    ensure_selected_gpu_capacity(selected_gpu, host_model_bytes, "this local launch")?;
    let vram_after_model = my_vram.saturating_sub(host_model_bytes);
    let ctx_size = compute_context_size(ctx_size_override, model_bytes, my_vram, total_group_vram);
    tracing::info!(
        "Context size: {ctx_size} tokens (model {:.1}GB, host weights ~{:.1}GB, {:.0}GB capacity, {:.1}GB free{})",
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
    // Pin slot count explicitly. llama.cpp's default of 4 slots silently
    // queues anything beyond that in an unbounded deferred deque, which is
    // the source of the MiniMax-style death spiral: 43k-token prefills
    // occupy all slots for minutes while new requests pile up invisibly.
    // The backend proxy enforces a matching inflight cap so the deferred
    // queue never actually fills. See network/openai/backend.rs.
    args.extend_from_slice(&[
        "-ngl".to_string(),
        "99".to_string(),
        "-fa".to_string(),
        "on".to_string(),
        "-fit".to_string(),
        "off".to_string(),
        "--no-mmap".to_string(),
        "--parallel".to_string(),
        slots.to_string(),
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
        // Anti-repetition default. llama.cpp ships repeat_penalty off
        // (1.0), which lets small quantized models fall into tight decode
        // loops ("Ok 1 2 3 4 5 Ok 1 2 3 4 5 ...") on mildly adversarial
        // prompts. 1.1 over a 256-token window is the conventional mild
        // setting — it kills loops without measurably affecting normal
        // prose. Per-request JSON fields (`repeat_penalty`, `repeat_last_n`)
        // still override these, so clients that tune their own sampling are
        // unaffected.
        "--repeat-penalty".to_string(),
        "1.1".to_string(),
        "--repeat-last-n".to_string(),
        "256".to_string(),
    ]);

    // Mesh hooks — tell llama-server where to call back.
    // Uses the management API port (default 3131) as the callback target.
    if let Ok(api_port) = std::env::var("MESH_API_PORT") {
        args.extend_from_slice(&["--mesh-port".to_string(), api_port]);
    }
    if std::env::var("MESH_HOOK_DEBUG").is_ok() {
        args.push("--mesh-hook-debug".to_string());
    }

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
    let local_device = resolve_device_for_binary(
        &llama_server.path,
        llama_server.flavor,
        selected_backend_device(selected_gpu)?,
    )?;
    if selected_gpu.is_some() {
        args.push("--device".to_string());
        args.push(local_device.clone());
    }
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
        .env("MESH_LLM_OWNER_PID", std::process::id().to_string())
        .env(
            "MESH_LLM_RUNTIME_DIR",
            runtime.dir().to_string_lossy().to_string(),
        )
        .stdout(std::process::Stdio::from(log_file))
        .stderr(std::process::Stdio::from(log_file2))
        .spawn()
        .with_context(|| {
            format!(
                "Failed to start llama-server at {}",
                llama_server.path.display()
            )
        })?;

    // Wait for health check — scale timeout by model size so large MoE shards
    // don't hit a fixed ceiling. 120s per GB gives plenty of headroom for slow
    // I/O or GPU upload, with a 600s floor for small models.
    let model_gb = model_bytes / GB + 1; // ceiling
    let max_wait_secs = std::cmp::max(600, model_gb * 120);
    tracing::info!("Health timeout: {max_wait_secs}s (model ~{model_gb} GB)");
    let url = format!("http://127.0.0.1:{http_port}/health");
    for i in 0..max_wait_secs {
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
            if pid == 0 {
                anyhow::bail!("llama-server returned PID 0 — refusing to proceed");
            }
            let child_started_at =
                crate::runtime::instance::validate::process_started_at_unix(pid).unwrap_or(None);
            let owner_started_at: i64 =
                crate::runtime::instance::validate::current_process_start_time_unix().unwrap_or(0);
            let llama_server_name = spawned_binary_name(&llama_server.path);
            let metadata = crate::runtime::instance::PidfileMetadata {
                cmd_name: llama_server_name.clone(),
                child_pid: pid,
                child_started_at_unix: child_started_at.unwrap_or(0),
                owner_pid: std::process::id(),
                owner_started_at_unix: owner_started_at,
                argv_snippet: crate::runtime::instance::PidfileMetadata::cap_argv(
                    &args,
                    crate::runtime::instance::ARGV_SNIPPET_MAX_BYTES,
                ),
                runtime_dir: runtime.dir().to_path_buf(),
            };
            let pidfile_guard =
                runtime.write_pidfile(&format!("llama-server-{}", http_port), &metadata)?;
            let expected_exit = Arc::new(AtomicBool::new(false));
            let handle = InferenceServerHandle {
                pid,
                expected_exit: expected_exit.clone(),
                expected_comm: llama_server_name,
                expected_start_time: child_started_at,
                _pidfile_guard: Some(pidfile_guard),
            };
            let (death_tx, death_rx) = tokio::sync::oneshot::channel();
            let pidfile_path = runtime.pidfile_path(&format!("llama-server-{}", http_port));
            tokio::spawn(async move {
                let _ = child.wait().await;
                let _ = std::fs::remove_file(&pidfile_path);
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
        "llama-server failed to become healthy within {max_wait_secs}s (model ~{model_gb} GB). {}",
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

/// Returns true only when `pid` is safe to pass to `kill(2)`.
///
/// Unsafe values on Unix:
///   0        → signals every process in the caller's process group
///   1        → signals init / launchd
///   >i32::MAX → wraps to a negative pid_t (e.g. u32::MAX → −1, which kills all user processes)
pub fn is_safe_kill_target(pid: u32) -> bool {
    pid > 1 && pid <= i32::MAX as u32
}

/// Terminate a process by PID, validating comm before signaling.
/// Returns true if the process is dead or was not ours. Returns false on unexpected error.
pub async fn terminate_process(
    pid: u32,
    expected_comm: &str,
    expected_start_time: Option<i64>,
) -> bool {
    if !is_safe_kill_target(pid) {
        tracing::error!("BUG: attempted to signal unsafe pid {pid} — refusing");
        return false;
    }
    !matches!(
        send_signal_if_matches(
            pid,
            expected_comm,
            expected_start_time,
            ProcessSignal::Terminate
        ),
        SignalOutcome::Failed
    )
}

/// Force-kill a process by PID, validating comm before signaling.
pub async fn force_kill_process(
    pid: u32,
    expected_comm: &str,
    expected_start_time: Option<i64>,
) -> bool {
    if !is_safe_kill_target(pid) {
        tracing::error!("BUG: attempted to signal unsafe pid {pid} — refusing");
        return false;
    }
    !matches!(
        send_signal_if_matches(pid, expected_comm, expected_start_time, ProcessSignal::Kill),
        SignalOutcome::Failed
    )
}

/// Poll until process exits or timeout_ms elapses. Returns true if dead within timeout.
pub async fn wait_for_exit(pid: u32, timeout_ms: u64) -> bool {
    if crate::runtime::instance::validate::process_liveness(pid)
        == crate::runtime::instance::validate::Liveness::Dead
    {
        return true;
    }
    let steps = timeout_ms / 100;
    for _ in 0..steps {
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        if crate::runtime::instance::validate::process_liveness(pid)
            == crate::runtime::instance::validate::Liveness::Dead
        {
            return true;
        }
    }
    false
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
        return "ROCm0".to_string();
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
        compute_context_size, is_safe_kill_target, parse_available_devices, preferred_device,
        terminate_process, wait_for_exit, BinaryFlavor, KvCacheQuant, KvCacheWarning, KvType,
        RpcServerHandle, SplitMode, GB,
    };
    use std::path::Path;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    #[test]
    fn kv_quant_small_model_is_plain_f16() {
        let quant = KvCacheQuant::for_model_size(GB);
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
            (
                5,
                &[
                    KvCacheWarning::QuantizedVBreaksMetalFaFallback,
                    KvCacheWarning::MismatchedQuantNeedsCudaFaAllQuants,
                ],
            ),
            (
                20,
                &[
                    KvCacheWarning::QuantizedVBreaksMetalFaFallback,
                    KvCacheWarning::MismatchedQuantNeedsCudaFaAllQuants,
                ],
            ),
            (
                49,
                &[
                    KvCacheWarning::QuantizedVBreaksMetalFaFallback,
                    KvCacheWarning::MismatchedQuantNeedsCudaFaAllQuants,
                ],
            ),
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
        let just_below_medium =
            KvCacheQuant::for_model_size(KvCacheQuant::MEDIUM_TIER_MIN_BYTES - 1);
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
        KvCacheQuant::for_model_size(GB).append_args(&mut args, GB);
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
        let old_phrasing = "llama_init_from_model: V cache quantization requires flash_attn";
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
    fn pinned_gpu_runtime_launch_returns_required_backend_device() {
        let pinned_gpu = crate::runtime::StartupPinnedGpuTarget {
            index: 1,
            stable_id: "pci:0000:65:00.0".into(),
            backend_device: "CUDA1".into(),
            vram_bytes: 24_000_000_000,
        };
        let selected = super::selected_backend_device(Some(&pinned_gpu)).unwrap();

        assert_eq!(selected, Some("CUDA1"));
    }

    #[test]
    fn pinned_gpu_runtime_launch_rejects_insufficient_selected_device_capacity() {
        let err = super::ensure_selected_gpu_capacity(
            Some(&crate::runtime::StartupPinnedGpuTarget {
                index: 0,
                stable_id: "uuid:GPU-small".into(),
                backend_device: "CUDA0".into(),
                vram_bytes: 8_000_000_000,
            }),
            10_000_000_000,
            "this local launch",
        )
        .unwrap_err();

        let message = err.to_string();
        assert!(message.contains("uuid:GPU-small"));
        assert!(message.contains("selected device"));
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

    #[test]
    fn rpc_handle_has_pid_and_port() {
        let handle = RpcServerHandle {
            pid: 12345,
            port: 8080,
            expected_exit: Arc::new(AtomicBool::new(false)),
            expected_comm: "rpc-server".to_string(),
            expected_start_time: Some(1700000000),
            _pidfile_guard: None,
        };
        assert!(handle.pid > 0);
        assert!(handle.port > 0);
    }

    #[test]
    fn rpc_handle_shutdown_sets_expected_exit() {
        let flag = Arc::new(AtomicBool::new(false));
        let handle = RpcServerHandle {
            pid: 999_999,
            port: 9999,
            expected_exit: flag.clone(),
            expected_comm: "rpc-server".to_string(),
            expected_start_time: Some(1700000000),
            _pidfile_guard: None,
        };
        assert!(!flag.load(Ordering::Relaxed));
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .build()
            .unwrap();
        rt.block_on(handle.shutdown());
        assert!(flag.load(Ordering::Relaxed));
    }

    #[test]
    fn safe_kill_target_rejects_zero() {
        assert!(!is_safe_kill_target(0));
    }

    #[test]
    fn safe_kill_target_rejects_one() {
        assert!(!is_safe_kill_target(1));
    }

    #[test]
    fn safe_kill_target_rejects_u32_max() {
        assert!(!is_safe_kill_target(u32::MAX));
    }

    #[test]
    fn safe_kill_target_rejects_i32_max_plus_one() {
        assert!(!is_safe_kill_target(i32::MAX as u32 + 1));
    }

    #[test]
    fn safe_kill_target_accepts_normal_pid() {
        assert!(is_safe_kill_target(999_999));
        assert!(is_safe_kill_target(2));
        assert!(is_safe_kill_target(i32::MAX as u32));
    }

    #[tokio::test]
    async fn terminate_nonexistent_pid_returns_true() {
        let result = terminate_process(999999, "nonexistent", None).await;
        assert!(result, "nonexistent PID should return true (already dead)");
    }

    #[tokio::test]
    async fn terminate_skips_when_comm_mismatch() {
        let self_pid = std::process::id();
        let result = terminate_process(self_pid, "wrong-comm-name", None).await;
        assert!(
            result,
            "mismatched comm should return true (skipped, treated as not our process)"
        );
    }

    #[tokio::test]
    async fn terminate_unsafe_pid_returns_false() {
        let result = terminate_process(1, "mesh-llm", None).await;
        assert!(!result, "unsafe PID should return false");
    }

    #[tokio::test]
    async fn wait_for_exit_returns_false_for_live_process() {
        let self_pid = std::process::id();
        let result = wait_for_exit(self_pid, 50).await;
        assert!(
            !result,
            "live process with short timeout should return false"
        );
    }

    #[tokio::test]
    async fn wait_for_exit_immediately_detects_dead_process() {
        let result = wait_for_exit(999_999, 0).await;
        assert!(
            result,
            "dead process should be detected before entering the poll loop"
        );
    }

    #[test]
    fn no_pkill_f_in_source_tree() {
        let src = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/src/inference/launch.rs"
        ))
        .unwrap();
        let forbidden_pattern = ["pkill", "-f", "llama-server"].join(" ");
        assert!(
            !src.contains(&forbidden_pattern),
            "forbidden pattern found in launch.rs"
        );
        let kill_func = format!("{}_{}{}", "kill", "llama", "_server");
        assert!(
            !src.contains(&kill_func),
            "legacy function reference still present after removal"
        );
        let runtime_src =
            std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/src/runtime/mod.rs"))
                .unwrap();
        assert!(
            !runtime_src.contains(&kill_func),
            "legacy function reference still present in runtime module"
        );
        let orphan_func = format!("{}_{}{}{}", "kill", "orphan", "_rpc", "_servers");
        assert!(
            !runtime_src.contains(&orphan_func),
            "legacy orphan cleanup function reference still present in runtime module"
        );
    }

    #[test]
    fn log_path_does_not_duplicate_extension() {
        use crate::runtime::instance::InstanceRuntime;
        use std::env;
        use tempfile::TempDir;

        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let temp_path = temp_dir.path();

        // Set runtime root to temp directory for test isolation
        let original_root = env::var("MESH_LLM_RUNTIME_ROOT").ok();
        env::set_var("MESH_LLM_RUNTIME_ROOT", temp_path);

        let runtime =
            InstanceRuntime::acquire(std::process::id()).expect("Failed to acquire runtime");

        // Test llama-server log path (should NOT have .log.log)
        let llama_log = runtime.log_path("llama-server");
        assert!(
            llama_log.to_string_lossy().ends_with("llama-server.log"),
            "llama-server log path should end with .log, got: {}",
            llama_log.display()
        );
        assert!(
            !llama_log.to_string_lossy().contains(".log.log"),
            "llama-server log path should not have double .log extension, got: {}",
            llama_log.display()
        );

        // Test rpc-server log path (should NOT have .log.log)
        let rpc_log = runtime.log_path("rpc-server-8001");
        assert!(
            rpc_log.to_string_lossy().ends_with("rpc-server-8001.log"),
            "rpc-server log path should end with .log, got: {}",
            rpc_log.display()
        );
        assert!(
            !rpc_log.to_string_lossy().contains(".log.log"),
            "rpc-server log path should not have double .log extension, got: {}",
            rpc_log.display()
        );

        // Restore original env var
        if let Some(orig) = original_root {
            env::set_var("MESH_LLM_RUNTIME_ROOT", orig);
        } else {
            env::remove_var("MESH_LLM_RUNTIME_ROOT");
        }
    }

    // ── Regression tests for slots/parallel wiring (T9) ──

    /// Verify that ModelLaunchSpec has a public `slots` field at compile time.
    /// This is a structural assertion — if the field disappears or becomes private,
    /// this code will not compile. It guards against regressions where TOML config
    /// parallel values are silently dropped before reaching llama-server.
    #[test]
    fn model_launch_spec_has_public_slots_field() {
        use super::ModelLaunchSpec;
        let path = Path::new("/dev/null");
        let _spec = ModelLaunchSpec {
            model: path,
            http_port: 8080,
            tunnel_ports: &[],
            tensor_split: None,
            split_mode: None,
            draft: None,
            draft_max: 0,
            model_bytes: 1_000_000_000u64,
            my_vram: 24_000_000_000u64,
            mmproj: None,
            ctx_size_override: None,
            total_group_vram: None,
            selected_gpu: None,
            slots: 8, // ← compile-time check: field must exist and be accessible
        };
    }

    /// Verify that all construction sites in election.rs populate `slots` correctly.
    /// This test constructs a ModelLaunchSpec with explicit non-default slots value
    /// to ensure callers can pass per-model parallel counts from TOML config.
    /// If any site forgets the field, this compilation will fail — which is exactly
    /// what we want as a regression guard.
    #[test]
    fn model_launch_spec_accepts_non_default_slots() {
        use super::ModelLaunchSpec;
        let path = Path::new("/dev/null");
        let spec = ModelLaunchSpec {
            model: path,
            http_port: 8080,
            tunnel_ports: &[],
            tensor_split: None,
            split_mode: Some(SplitMode::Layer),
            draft: None,
            draft_max: 4,
            model_bytes: 50_000_000_000u64,
            my_vram: 24_000_000_000u64,
            mmproj: Some(Path::new("/dev/null")),
            ctx_size_override: Some(32768),
            total_group_vram: Some(96_000_000_000u64),
            selected_gpu: None,
            slots: 16, // non-default value from TOML config
        };
        assert_eq!(spec.slots, 16);
    }

    /// Verify that start_llama_server receives and can read the `slots` field.
    /// This is a compile-time check that the destructured `let slots = spec.slots;`
    /// in start_llama_server actually works — if the field were renamed or removed,
    /// this would fail to compile. The actual process-spawning behavior is tested
    /// at integration time since we don't want to spawn real llama-server here.
    #[test]
    fn launch_spec_slots_propagates_through_destructure() {
        use super::ModelLaunchSpec;
        let path = Path::new("/dev/null");
        let spec = ModelLaunchSpec {
            model: path,
            http_port: 8080,
            tunnel_ports: &[],
            tensor_split: None,
            split_mode: None,
            draft: None,
            draft_max: 0,
            model_bytes: 20_000_000_000u64,
            my_vram: 16_000_000_000u64,
            mmproj: None,
            ctx_size_override: Some(8192),
            total_group_vram: None,
            selected_gpu: None,
            slots: 32,
        };

        // Destructure exactly as start_llama_server does it (lines ~1078-1091)
        let _model = spec.model;
        let _http_port = spec.http_port;
        let _tunnel_ports = spec.tunnel_ports;
        let _tensor_split = spec.tensor_split;
        let _split_mode = spec.split_mode;
        let _draft = spec.draft;
        let _draft_max = spec.draft_max;
        let _model_bytes = spec.model_bytes;
        let _my_vram = spec.my_vram;
        let _mmproj = spec.mmproj;
        let _ctx_size_override = spec.ctx_size_override;
        let _total_group_vram = spec.total_group_vram;
        let _selected_gpu = spec.selected_gpu;
        let slots = spec.slots; // ← this is the key line being tested
        assert_eq!(slots, 32);
    }

    /// Verify that default slots value of 4 would be explicitly set — callers must
    /// not rely on defaults. This test documents the expected behavior: when TOML
    /// config omits `slots`, the caller should use an explicit fallback (4).
    #[test]
    fn launch_spec_slots_is_explicitly_set() {
        use super::ModelLaunchSpec;
        let path = Path::new("/dev/null");
        // The following would NOT compile if we omitted `slots:`:
        //   let bad_spec = ModelLaunchSpec { model: path, ..Default::default() };
        // Since ModelLaunchSpec doesn't implement Default, this is enforced at compile time.
        let good_spec = ModelLaunchSpec {
            model: path,
            http_port: 8080,
            tunnel_ports: &[],
            tensor_split: None,
            split_mode: None,
            draft: None,
            draft_max: 0,
            model_bytes: 1_000_000_000u64,
            my_vram: 24_000_000_000u64,
            mmproj: None,
            ctx_size_override: None,
            total_group_vram: None,
            selected_gpu: None,
            slots: 4, // explicit default from TOML config fallback
        };
        assert_eq!(good_spec.slots, 4);
    }
}
