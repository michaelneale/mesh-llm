use std::collections::BTreeSet;
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::fs::{File, OpenOptions};
use std::io::{LineWriter, Write};
use std::path::Path;
use std::ptr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};

#[cfg(unix)]
use std::os::unix::fs::OpenOptionsExt;

use anyhow::{anyhow, Context, Result};
use serde_json::Value;
use skippy_ffi::{
    ActivationDType, ActivationDesc as RawActivationDesc, ActivationLayout,
    ChatMessage as RawChatMessage, Error as RawError,
    GenerationSignalWindow as RawGenerationSignalWindow, KvPageDesc as RawKvPageDesc, LoadMode,
    LogitBias as RawLogitBias, Model as RawModel, ModelInfo as RawModelInfo,
    RuntimeConfig as RawRuntimeConfig, SamplingConfig as RawSamplingConfig, Session as RawSession,
    SlicePlan as RawSlicePlan, Status, TensorInfo as RawTensorInfo, TensorRole,
    TokenSignal as RawTokenSignal,
};
use tokio::sync::mpsc;

mod devices;
pub mod package;

pub const MAX_LOGIT_BIAS: usize = 256;
pub const GGML_TYPE_F16: u32 = 1;
pub const GGML_TYPE_Q4_0: u32 = 2;
pub const GGML_TYPE_Q8_0: u32 = 8;
pub const LLAMA_SERVER_DEFAULT_N_BATCH: u32 = 2048;
pub const LLAMA_SERVER_DEFAULT_N_UBATCH: u32 = 512;

/// GGML_LLAMA_LOG_LEVEL values (set before llama_backend_init).
/// 0=silent, 1=error, 2=warn, 3=info (default), 4=debug.
pub const LLAMA_LOG_LEVEL_DEBUG: &str = "4";

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[repr(i32)]
pub enum FlashAttentionType {
    #[default]
    Auto = -1,
    Disabled = 0,
    Enabled = 1,
}

pub use devices::{backend_devices, BackendDevice, BackendDeviceType};
pub use skippy_ffi::LoadMode as RuntimeLoadMode;
pub use skippy_ffi::{
    ActivationDType as RuntimeActivationDType, ActivationLayout as RuntimeActivationLayout,
};

static NATIVE_LOG_FILE: OnceLock<Mutex<Option<LineWriter<File>>>> = OnceLock::new();

/// Channel sender for filtered native log messages.
/// Messages matching key patterns (backend init, model load, VRAM, KV cache, tokenizer) are sent here.
static NATIVE_LOG_FILTERED_TX: OnceLock<Mutex<Option<mpsc::UnboundedSender<NativeLogEvent>>>> =
    OnceLock::new();

static NATIVE_LOG_AGGREGATOR: OnceLock<Mutex<NativeLogAggregator>> = OnceLock::new();
static NATIVE_LOG_FORWARDING_ENABLED: AtomicBool = AtomicBool::new(false);

#[derive(Debug, Clone, PartialEq)]
pub struct NativeLogEvent {
    pub message: String,
    pub category: &'static str,
    pub params: Vec<(String, Value)>,
}

#[derive(Debug, Default)]
struct ProgressTracker {
    total: Option<usize>,
    completed: usize,
    next_percent: usize,
}

impl ProgressTracker {
    fn reset(&mut self) {
        *self = Self::default();
    }

    fn set_total(&mut self, total: usize) {
        self.total = Some(total);
        self.completed = 0;
        self.next_percent = 10;
    }

    fn advance(
        &mut self,
        delta: usize,
        category: &'static str,
        label: &'static str,
        unit: &'static str,
    ) -> Vec<NativeLogEvent> {
        let Some(total) = self.total else {
            return Vec::new();
        };
        if total == 0 {
            return Vec::new();
        }

        self.completed = self.completed.saturating_add(delta).min(total);
        let mut events = Vec::new();
        while self.next_percent <= 100 && self.completed * 100 >= total * self.next_percent {
            events.push(NativeLogEvent {
                message: format!(
                    "{label} {}% ({}/{} {unit})",
                    self.next_percent, self.completed, total
                ),
                category,
                params: Vec::new(),
            });
            self.next_percent += 10;
        }
        events
    }

    fn is_complete(&self) -> bool {
        matches!(self.total, Some(total) if total > 0 && self.completed >= total)
    }
}

#[derive(Debug, Default)]
struct ModelMetadataHighlights {
    architecture: Option<String>,
    name: Option<String>,
    model_type: Option<String>,
    size_label: Option<String>,
    context_length: Option<String>,
    block_count: Option<String>,
    embedding_length: Option<String>,
    feed_forward_length: Option<String>,
    attention_heads: Option<String>,
    attention_heads_kv: Option<String>,
    tokenizer_model: Option<String>,
    tokenizer_pre: Option<String>,
}

impl ModelMetadataHighlights {
    fn apply(&mut self, key: &str, value: &str) {
        let value = value.trim().trim_matches('"').to_string();
        if value.is_empty() {
            return;
        }

        match key {
            "general.architecture" => self.architecture = Some(value),
            "general.name" => self.name = Some(value),
            "general.type" => self.model_type = Some(value),
            "general.size_label" => self.size_label = Some(value),
            "tokenizer.ggml.model" => self.tokenizer_model = Some(value),
            "tokenizer.ggml.pre" => self.tokenizer_pre = Some(value),
            _ if key.ends_with(".context_length") => self.context_length = Some(value),
            _ if key.ends_with(".block_count") => self.block_count = Some(value),
            _ if key.ends_with(".embedding_length") => self.embedding_length = Some(value),
            _ if key.ends_with(".feed_forward_length") => self.feed_forward_length = Some(value),
            _ if key.ends_with(".attention.head_count") => self.attention_heads = Some(value),
            _ if key.ends_with(".attention.head_count_kv") => self.attention_heads_kv = Some(value),
            _ => {}
        }
    }

    fn summary_params(&self) -> Vec<(String, Value)> {
        let mut params = Vec::new();
        if let Some(value) = &self.architecture {
            params.push(("architecture".to_string(), Value::String(value.clone())));
        }
        if let Some(value) = &self.name {
            params.push(("name".to_string(), Value::String(value.clone())));
        }
        if let Some(value) = &self.model_type {
            params.push(("type".to_string(), Value::String(value.clone())));
        }
        if let Some(value) = &self.size_label {
            params.push(("size".to_string(), Value::String(value.clone())));
        }
        if let Some(value) = &self.context_length {
            params.push(("ctx".to_string(), json_value_from_text(value)));
        }
        if let Some(value) = &self.block_count {
            params.push(("blocks".to_string(), json_value_from_text(value)));
        }
        if let Some(value) = &self.embedding_length {
            params.push(("embed".to_string(), json_value_from_text(value)));
        }
        if let Some(value) = &self.feed_forward_length {
            params.push(("ffn".to_string(), json_value_from_text(value)));
        }
        if let Some(value) = &self.attention_heads {
            params.push(("heads".to_string(), json_value_from_text(value)));
        }
        if let Some(value) = &self.attention_heads_kv {
            params.push(("kv_heads".to_string(), json_value_from_text(value)));
        }
        if let Some(value) = &self.tokenizer_model {
            params.push(("tokenizer".to_string(), Value::String(value.clone())));
        }
        if let Some(value) = &self.tokenizer_pre {
            params.push(("tokenizer_pre".to_string(), Value::String(value.clone())));
        }
        params
    }
}

#[derive(Debug, Default)]
struct NativeLogAggregator {
    metadata_progress: ProgressTracker,
    tensor_progress: ProgressTracker,
    layer_assign_progress: ProgressTracker,
    kv_cache_progress: ProgressTracker,
    metadata_in_dump: bool,
    metadata_summary_emitted: bool,
    metadata_highlights: ModelMetadataHighlights,
    tensor_groups: Vec<(String, usize)>,
    tensor_groups_emitted: bool,
    kv_layers_seen: BTreeSet<usize>,
}

fn native_log_file() -> &'static Mutex<Option<LineWriter<File>>> {
    NATIVE_LOG_FILE.get_or_init(|| Mutex::new(None))
}

fn native_log_aggregator() -> &'static Mutex<NativeLogAggregator> {
    NATIVE_LOG_AGGREGATOR.get_or_init(|| Mutex::new(NativeLogAggregator::default()))
}

/// Register a channel receiver for filtered native log messages.
/// Returns the receiver end; call this once before model loading begins.
pub fn register_filtered_native_logs() -> mpsc::UnboundedReceiver<NativeLogEvent> {
    let (tx, rx) = mpsc::unbounded_channel();
    NATIVE_LOG_FILTERED_TX
        .get_or_init(|| Mutex::new(None))
        .lock()
        .unwrap()
        .replace(tx);
    if let Ok(mut aggregator) = native_log_aggregator().lock() {
        aggregator.reset();
    }
    rx
}

pub fn unregister_filtered_native_logs() {
    if let Some(sender) = NATIVE_LOG_FILTERED_TX.get() {
        sender.lock().unwrap().take();
    }
    if let Ok(mut aggregator) = native_log_aggregator().lock() {
        aggregator.reset();
    }
}

pub fn set_filtered_native_logs_enabled(enabled: bool) {
    NATIVE_LOG_FORWARDING_ENABLED.store(enabled, Ordering::Relaxed);
}

impl NativeLogAggregator {
    fn reset(&mut self) {
        *self = Self::default();
    }

    fn reset_model_loading_state(&mut self) {
        self.metadata_progress.reset();
        self.tensor_progress.reset();
        self.layer_assign_progress.reset();
        self.kv_cache_progress.reset();
        self.metadata_in_dump = false;
        self.metadata_summary_emitted = false;
        self.metadata_highlights = ModelMetadataHighlights::default();
        self.tensor_groups.clear();
        self.tensor_groups_emitted = false;
        self.kv_layers_seen.clear();
    }

    fn process_line(&mut self, line: &str) -> Vec<NativeLogEvent> {
        let s = line.trim();
        if s.is_empty() {
            return Vec::new();
        }

        let mut events = Vec::new();
        let metadata_kv = parse_metadata_kv_line(s);
        let tensor_summary = parse_tensor_type_summary(s);
        if metadata_kv.is_none() {
            events.extend(self.flush_metadata_summary());
        }
        if tensor_summary.is_none() {
            events.extend(self.flush_tensor_group_summary());
        }

        if let Some((metadata_rows, tensor_rows)) = parse_loaded_metadata_counts(s) {
            self.reset_model_loading_state();
            self.metadata_progress.set_total(metadata_rows);
            self.tensor_progress.set_total(tensor_rows);
            events.push(NativeLogEvent {
                message: format!(
                    "model load plan: metadata rows={metadata_rows}, tensor rows={tensor_rows}"
                ),
                category: "model",
                params: Vec::new(),
            });
            return events;
        }

        if let Some((key, value)) = metadata_kv {
            self.metadata_in_dump = true;
            self.metadata_highlights.apply(key, value);
            events.extend(
                self.metadata_progress
                    .advance(1, "model", "metadata", "rows"),
            );
            return events;
        }

        if let Some((tensor_type, count)) = tensor_summary {
            self.record_tensor_group(tensor_type, count);
            events.extend(
                self.tensor_progress
                    .advance(count, "model", "tensors", "tensors"),
            );
            if self.tensor_progress.is_complete() {
                events.extend(self.flush_tensor_group_summary());
            }
            return events;
        }

        if let Some(layers) = parse_kv_cache_layers_total(s) {
            self.kv_cache_progress.set_total(layers);
            self.kv_layers_seen.clear();
            events.push(NativeLogEvent {
                message: format!("kv cache plan: layer rows={layers}"),
                category: "kv_cache",
                params: Vec::new(),
            });
            return events;
        }

        if let Some(layer_index) = parse_kv_cache_layer_index(s) {
            if self.kv_layers_seen.insert(layer_index) {
                events.extend(
                    self.kv_cache_progress
                        .advance(1, "kv_cache", "kv cache", "layers"),
                );
            }
            return events;
        }

        if let Some(layer_index) = parse_layer_assign_index(s) {
            if self.layer_assign_progress.total.is_none() {
                if let Some(total) = self
                    .metadata_highlights
                    .block_count
                    .as_deref()
                    .and_then(|s| s.parse::<usize>().ok())
                {
                    self.layer_assign_progress.set_total(total);
                }
            }
            let new_completed = layer_index + 1;
            if new_completed > self.layer_assign_progress.completed {
                let delta = new_completed - self.layer_assign_progress.completed;
                events.extend(
                    self.layer_assign_progress
                        .advance(delta, "model", "layers", "layers"),
                );
            }
            return events;
        }

        if should_suppress_native_log_line(s) {
            return events;
        }

        if let Some(event) = summarize_native_log_line(s) {
            events.push(event);
        }

        events
    }

    fn flush_metadata_summary(&mut self) -> Vec<NativeLogEvent> {
        if !self.metadata_in_dump || self.metadata_summary_emitted {
            return Vec::new();
        }
        self.metadata_in_dump = false;
        self.metadata_summary_emitted = true;
        let params = self.metadata_highlights.summary_params();
        if params.is_empty() {
            Vec::new()
        } else {
            vec![NativeLogEvent {
                message: "Reading model metadata...".to_string(),
                category: "model",
                params,
            }]
        }
    }

    fn record_tensor_group(&mut self, tensor_type: &str, count: usize) {
        let tensor_type = canonical_tensor_group_key(tensor_type);
        if let Some((_, existing_count)) = self
            .tensor_groups
            .iter_mut()
            .find(|(existing_type, _)| existing_type == &tensor_type)
        {
            *existing_count = count;
        } else {
            self.tensor_groups.push((tensor_type, count));
        }
        self.tensor_groups_emitted = false;
    }

    fn flush_tensor_group_summary(&mut self) -> Vec<NativeLogEvent> {
        if self.tensor_groups.is_empty() || self.tensor_groups_emitted {
            return Vec::new();
        }
        self.tensor_groups_emitted = true;
        vec![NativeLogEvent {
            message: "Reading tensor groups...".to_string(),
            category: "model",
            params: self
                .tensor_groups
                .iter()
                .map(|(tensor_type, count)| (tensor_type.clone(), Value::from(*count as u64)))
                .collect(),
        }]
    }
}

fn json_value_from_text(value: &str) -> Value {
    value
        .parse::<u64>()
        .map(Value::from)
        .unwrap_or_else(|_| Value::String(value.to_string()))
}

fn canonical_tensor_group_key(tensor_type: &str) -> String {
    let trimmed = tensor_type.trim();
    if trimmed.eq_ignore_ascii_case("q4_k") {
        "q4_K".to_string()
    } else if trimmed.eq_ignore_ascii_case("q5_k") {
        "q5_K".to_string()
    } else {
        trimmed.to_string()
    }
}

fn should_suppress_native_log_line(line: &str) -> bool {
    line.starts_with("llama_model_loader:") && (line.contains(": - kv") || line.contains("- kv"))
        || (line.starts_with("clip_model_loader:") && line.contains(": tensor["))
        || line.contains("tokenizer.ggml.tokens arr")
        || line.contains("tokenizer.ggml.merges arr")
        || line.contains("tokenizer.ggml.token_type arr")
        || line.starts_with("print_info:")
        || (line.starts_with("llama_kv_cache:")
            && (line.contains(": filtered") || line.contains(": dev =")))
}

fn summarize_native_log_line(line: &str) -> Option<NativeLogEvent> {
    if line.contains("backend_init")
        || line.contains("llama_backend_init")
        || line.contains("GGML_CUDA")
        || (line.contains("CUDA") && (line.contains("init") || line.contains("device")))
        || (line.contains("metal") && (line.contains("init") || line.contains("device")))
    {
        return Some(NativeLogEvent {
            message: line.to_string(),
            category: "backend",
            params: Vec::new(),
        });
    }

    if line.contains(".gguf loaded")
        || line.starts_with("llm_load_print_meta")
        || line.starts_with("llm_load_tensors")
        || (line.contains("loading model") && !line.contains("clip_model"))
        || (line.contains("loaded model") && !line.starts_with("llama_model_loader:"))
    {
        return Some(NativeLogEvent {
            message: line.to_string(),
            category: "model",
            params: Vec::new(),
        });
    }

    if line.contains("VRAM")
        || line.contains("vram")
        || line.contains("mem_alloc")
        || line.contains("_Mapped model buffer size")
        || (line.contains("GPU") && line.contains("memory"))
        || line.contains("compute buffer size")
        || line.contains("scratch buffer")
    {
        return Some(NativeLogEvent {
            message: line.to_string(),
            category: "memory",
            params: Vec::new(),
        });
    }

    if line.starts_with("llama_kv_cache:")
        && (line.contains("buffer size") || line.contains("size = ") || line.contains("attn_rot"))
    {
        return Some(NativeLogEvent {
            message: line.to_string(),
            category: "kv_cache",
            params: Vec::new(),
        });
    }

    if line.starts_with("init_tokenizer:")
        || line.starts_with("load: special tokens cache size")
        || line.starts_with("load: token to piece cache size")
    {
        return Some(NativeLogEvent {
            message: line.to_string(),
            category: "tokenizer",
            params: Vec::new(),
        });
    }

    None
}

fn parse_loaded_metadata_counts(line: &str) -> Option<(usize, usize)> {
    let (_, remainder) = line.split_once("loaded meta data with ")?;
    let (metadata_rows, remainder) = remainder.split_once(" key-value pairs and ")?;
    let metadata_rows = metadata_rows.trim().parse().ok()?;
    let (tensor_rows, _) = remainder.split_once(" tensors")?;
    let tensor_rows = tensor_rows.trim().parse().ok()?;
    Some((metadata_rows, tensor_rows))
}

fn parse_metadata_kv_line(line: &str) -> Option<(&str, &str)> {
    if !line.starts_with("llama_model_loader:") || !line.contains("- kv") {
        return None;
    }
    let (_, remainder) = line.split_once(": - kv")?;
    let (_, remainder) = remainder.split_once(':')?;
    let remainder = remainder.trim();
    let (lhs, value) = remainder.split_once(" = ")?;
    let key = lhs.split_whitespace().next()?;
    Some((key, value.trim()))
}

fn parse_tensor_type_summary(line: &str) -> Option<(&str, usize)> {
    if !line.starts_with("llama_model_loader:") || !line.contains("- type") {
        return None;
    }
    let (_, remainder) = line.split_once("- type")?;
    let remainder = remainder.trim();
    let (tensor_type, count_and_suffix) = remainder.split_once(':')?;
    let count = count_and_suffix.split_whitespace().next()?.parse().ok()?;
    Some((tensor_type.trim(), count))
}

fn parse_layer_assign_index(line: &str) -> Option<usize> {
    if !line.starts_with("load_tensors: layer") || !line.contains("assigned to device") {
        return None;
    }
    let (_, remainder) = line.split_once("load_tensors: layer")?;
    let digits = remainder
        .trim_start()
        .chars()
        .take_while(|ch| ch.is_ascii_digit())
        .collect::<String>();
    (!digits.is_empty()).then(|| digits.parse().ok()).flatten()
}

fn parse_kv_cache_layers_total(line: &str) -> Option<usize> {
    if !line.starts_with("llama_kv_cache:") || !line.contains(" layers") {
        return None;
    }
    let prefix = line.split_once(" layers")?.0;
    let digits = prefix
        .chars()
        .rev()
        .skip_while(|ch| ch.is_whitespace())
        .take_while(|ch| ch.is_ascii_digit())
        .collect::<String>()
        .chars()
        .rev()
        .collect::<String>();
    (!digits.is_empty()).then(|| digits.parse().ok()).flatten()
}

fn parse_kv_cache_layer_index(line: &str) -> Option<usize> {
    if !line.starts_with("llama_kv_cache: layer") {
        return None;
    }
    let (_, remainder) = line.split_once("layer")?;
    let digits = remainder
        .trim_start()
        .chars()
        .take_while(|ch| ch.is_ascii_digit())
        .collect::<String>();
    (!digits.is_empty()).then(|| digits.parse().ok()).flatten()
}

fn flush_native_log_writer<W: Write>(writer: &mut Option<LineWriter<W>>) {
    if let Some(writer) = writer.as_mut() {
        let _ = writer.flush();
    }
}

fn clear_native_log_file() {
    if let Ok(mut guard) = native_log_file().lock() {
        flush_native_log_writer(&mut guard);
        *guard = None;
    }
}

fn set_native_log_callback(callback: skippy_ffi::LlamaLogCallback) {
    unsafe {
        skippy_ffi::llama_log_set(callback, ptr::null_mut());
        skippy_ffi::mtmd_helper_log_set(callback, ptr::null_mut());
    }
}

pub fn redirect_native_logs_to_file(path: impl AsRef<Path>) -> Result<()> {
    let path = path.as_ref();
    let mut options = OpenOptions::new();
    options.create(true).append(true);
    #[cfg(unix)]
    options.mode(0o600);

    let file = options
        .open(path)
        .with_context(|| format!("open skippy native log file {}", path.display()))?;
    let mut guard = native_log_file()
        .lock()
        .map_err(|_| anyhow!("native log file mutex poisoned"))?;
    flush_native_log_writer(&mut guard);
    *guard = Some(LineWriter::new(file));
    drop(guard);

    set_native_log_callback(Some(write_native_log));

    Ok(())
}

pub fn suppress_native_logs() {
    clear_native_log_file();
    set_native_log_callback(Some(discard_native_log));
}

pub fn restore_native_logs() {
    clear_native_log_file();
    set_native_log_callback(None);
}

/// Enable verbose llama.cpp logging. Call before `llama_backend_init()` / model loading.
/// Sets GGML_LLAMA_LOG_LEVEL=4 so LLAMA_LOG_DEBUG macros produce output.
pub fn enable_verbose_native_logs() {
    std::env::set_var("GGML_LLAMA_LOG_LEVEL", LLAMA_LOG_LEVEL_DEBUG);
}

/// Disable verbose llama.cpp logging (restore default level).
pub fn disable_verbose_native_logs() {
    std::env::remove_var("GGML_LLAMA_LOG_LEVEL");
}

unsafe extern "C" fn write_native_log(_level: c_int, text: *const c_char, _user_data: *mut c_void) {
    if text.is_null() {
        return;
    }

    let bytes = unsafe { CStr::from_ptr(text) }.to_bytes();
    if let Ok(mut guard) = native_log_file().lock() {
        if let Some(writer) = guard.as_mut() {
            let _ = writer.write_all(bytes);
        }
    }

    // Also send aggregated messages through the channel when runtime forwarding is enabled.
    if !NATIVE_LOG_FORWARDING_ENABLED.load(Ordering::Relaxed) {
        return;
    }

    if let Ok(text_str) = core::str::from_utf8(bytes) {
        let events = if let Ok(mut aggregator) = native_log_aggregator().lock() {
            aggregator.process_line(text_str.trim())
        } else {
            Vec::new()
        };
        if let Some(tx) = NATIVE_LOG_FILTERED_TX.get() {
            if let Ok(guard) = tx.lock() {
                if let Some(ref sender) = *guard {
                    for event in events {
                        let _ = sender.send(event);
                    }
                }
            }
        }
    }
}

unsafe extern "C" fn discard_native_log(
    _level: c_int,
    _text: *const c_char,
    _user_data: *mut c_void,
) {
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeConfig {
    pub stage_index: u32,
    pub layer_start: u32,
    pub layer_end: u32,
    pub ctx_size: u32,
    pub lane_count: u32,
    pub n_batch: Option<u32>,
    pub n_ubatch: Option<u32>,
    pub n_threads: Option<u32>,
    pub n_threads_batch: Option<u32>,
    pub n_gpu_layers: i32,
    pub selected_backend_device: Option<String>,
    pub cache_type_k: u32,
    pub cache_type_v: u32,
    pub flash_attn_type: FlashAttentionType,
    pub load_mode: LoadMode,
    pub projector_path: Option<String>,
    pub include_embeddings: bool,
    pub include_output: bool,
    pub filter_tensors_on_load: bool,
}

impl RuntimeConfig {
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.layer_start >= self.layer_end {
            return Err("layer_start must be less than layer_end");
        }
        if self
            .selected_backend_device
            .as_deref()
            .is_some_and(str::is_empty)
        {
            return Err("selected_backend_device must not be empty");
        }
        if self.projector_path.as_deref().is_some_and(str::is_empty) {
            return Err("projector_path must not be empty");
        }
        if self.n_batch == Some(0) {
            return Err("n_batch must be greater than zero when provided");
        }
        if self.n_ubatch == Some(0) {
            return Err("n_ubatch must be greater than zero when provided");
        }
        if self.n_threads == Some(0) {
            return Err("n_threads must be greater than zero when provided");
        }
        if self.n_threads_batch == Some(0) {
            return Err("n_threads_batch must be greater than zero when provided");
        }
        Ok(())
    }

    fn as_raw(&self) -> Result<RawRuntimeConfigParts> {
        self.validate().map_err(anyhow::Error::msg)?;
        let selected_backend_device = self
            .selected_backend_device
            .as_ref()
            .map(|device| {
                CString::new(device.as_bytes())
                    .context("selected_backend_device contains an interior NUL byte")
            })
            .transpose()?;
        let selected_backend_device_ptr = selected_backend_device
            .as_ref()
            .map(|device| device.as_ptr())
            .unwrap_or(ptr::null());
        Ok(RawRuntimeConfigParts {
            raw: RawRuntimeConfig {
                stage_index: i32::try_from(self.stage_index).context("stage_index exceeds i32")?,
                layer_start: i32::try_from(self.layer_start).context("layer_start exceeds i32")?,
                layer_end: i32::try_from(self.layer_end).context("layer_end exceeds i32")?,
                ctx_size: i32::try_from(self.ctx_size).context("ctx_size exceeds i32")?,
                lane_count: i32::try_from(self.lane_count).context("lane_count exceeds i32")?,
                n_batch: self
                    .n_batch
                    .or(Some(LLAMA_SERVER_DEFAULT_N_BATCH))
                    .map(i32::try_from)
                    .transpose()
                    .context("n_batch exceeds i32")?
                    .unwrap_or(0),
                n_ubatch: self
                    .n_ubatch
                    .or(Some(LLAMA_SERVER_DEFAULT_N_UBATCH))
                    .map(i32::try_from)
                    .transpose()
                    .context("n_ubatch exceeds i32")?
                    .unwrap_or(0),
                n_threads: self
                    .n_threads
                    .map(i32::try_from)
                    .transpose()
                    .context("n_threads exceeds i32")?
                    .unwrap_or(0),
                n_threads_batch: self
                    .n_threads_batch
                    .or(self.n_threads)
                    .map(i32::try_from)
                    .transpose()
                    .context("n_threads_batch exceeds i32")?
                    .unwrap_or(0),
                n_gpu_layers: self.n_gpu_layers,
                cache_type_k: i32::try_from(self.cache_type_k)
                    .context("cache_type_k exceeds i32")?,
                cache_type_v: i32::try_from(self.cache_type_v)
                    .context("cache_type_v exceeds i32")?,
                flash_attn_type: self.flash_attn_type as i32,
                load_mode: self.load_mode,
                disable_repack: false,
                filter_tensors_on_load: self.filter_tensors_on_load,
                include_embeddings: self.include_embeddings,
                include_output: self.include_output,
                selected_backend_device: selected_backend_device_ptr,
            },
            _selected_backend_device: selected_backend_device,
        })
    }
}

struct RawRuntimeConfigParts {
    raw: RawRuntimeConfig,
    _selected_backend_device: Option<CString>,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            stage_index: 0,
            layer_start: 0,
            layer_end: 1,
            ctx_size: 512,
            lane_count: 1,
            n_batch: Some(LLAMA_SERVER_DEFAULT_N_BATCH),
            n_ubatch: Some(LLAMA_SERVER_DEFAULT_N_UBATCH),
            n_threads: None,
            n_threads_batch: None,
            n_gpu_layers: 0,
            selected_backend_device: None,
            cache_type_k: GGML_TYPE_F16,
            cache_type_v: GGML_TYPE_F16,
            flash_attn_type: FlashAttentionType::Auto,
            load_mode: LoadMode::RuntimeSlice,
            projector_path: None,
            include_embeddings: true,
            include_output: true,
            filter_tensors_on_load: false,
        }
    }
}

pub fn parse_cache_type(value: &str) -> Result<u32> {
    let normalized = value.trim().to_ascii_lowercase().replace('-', "_");
    match normalized.as_str() {
        "" | "f16" => Ok(GGML_TYPE_F16),
        "q4" | "q4_0" => Ok(GGML_TYPE_Q4_0),
        "q8" | "q8_0" => Ok(GGML_TYPE_Q8_0),
        _ => Err(anyhow!("unsupported KV cache type {value:?}")),
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo {
    pub name: String,
    pub layer_index: Option<u32>,
    pub role: TensorRole,
    pub ggml_type: u32,
    pub byte_size: u64,
    pub element_count: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivationDesc {
    pub version: u32,
    pub dtype: ActivationDType,
    pub layout: ActivationLayout,
    pub producer_stage_index: i32,
    pub layer_start: i32,
    pub layer_end: i32,
    pub token_count: u32,
    pub sequence_count: u32,
    pub payload_bytes: u64,
    pub flags: u64,
}

impl ActivationDesc {
    fn as_raw(&self) -> RawActivationDesc {
        RawActivationDesc {
            version: self.version,
            dtype: self.dtype,
            layout: self.layout,
            producer_stage_index: self.producer_stage_index,
            layer_start: self.layer_start,
            layer_end: self.layer_end,
            token_count: self.token_count,
            sequence_count: self.sequence_count,
            payload_bytes: self.payload_bytes,
            flags: self.flags,
        }
    }
}

impl From<RawActivationDesc> for ActivationDesc {
    fn from(raw: RawActivationDesc) -> Self {
        Self {
            version: raw.version,
            dtype: raw.dtype,
            layout: raw.layout,
            producer_stage_index: raw.producer_stage_index,
            layer_start: raw.layer_start,
            layer_end: raw.layer_end,
            token_count: raw.token_count,
            sequence_count: raw.sequence_count,
            payload_bytes: raw.payload_bytes,
            flags: raw.flags,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActivationFrame {
    pub desc: ActivationDesc,
    pub payload: Vec<u8>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RuntimeKvPageDesc {
    pub version: u32,
    pub layer_start: i32,
    pub layer_end: i32,
    pub token_start: u64,
    pub token_count: u64,
    pub layer_count: u32,
    pub k_type: u32,
    pub v_type: u32,
    pub k_row_bytes: u32,
    pub v_row_bytes: u32,
    pub v_element_bytes: u32,
    pub payload_bytes: u64,
    pub flags: u64,
}

impl RuntimeKvPageDesc {
    fn as_raw(&self) -> RawKvPageDesc {
        RawKvPageDesc {
            version: self.version,
            layer_start: self.layer_start,
            layer_end: self.layer_end,
            token_start: self.token_start,
            token_count: self.token_count,
            layer_count: self.layer_count,
            k_type: self.k_type,
            v_type: self.v_type,
            k_row_bytes: self.k_row_bytes,
            v_row_bytes: self.v_row_bytes,
            v_element_bytes: self.v_element_bytes,
            payload_bytes: self.payload_bytes,
            flags: self.flags,
        }
    }
}

impl From<RawKvPageDesc> for RuntimeKvPageDesc {
    fn from(raw: RawKvPageDesc) -> Self {
        Self {
            version: raw.version,
            layer_start: raw.layer_start,
            layer_end: raw.layer_end,
            token_start: raw.token_start,
            token_count: raw.token_count,
            layer_count: raw.layer_count,
            k_type: raw.k_type,
            v_type: raw.v_type,
            k_row_bytes: raw.k_row_bytes,
            v_row_bytes: raw.v_row_bytes,
            v_element_bytes: raw.v_element_bytes,
            payload_bytes: raw.payload_bytes,
            flags: raw.flags,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeKvPage {
    pub desc: RuntimeKvPageDesc,
    pub payload: Vec<u8>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct TokenSignal {
    pub entropy: f32,
    pub top_logprob: f32,
    pub second_logprob: f32,
    pub margin: f32,
    pub top_token: i32,
    pub second_token: i32,
}

impl From<RawTokenSignal> for TokenSignal {
    fn from(raw: RawTokenSignal) -> Self {
        Self {
            entropy: raw.entropy,
            top_logprob: raw.top_logprob,
            second_logprob: raw.second_logprob,
            margin: raw.margin,
            top_token: raw.top_token,
            second_token: raw.second_token,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct GenerationSignalWindow {
    pub token_count: u32,
    pub mean_entropy: f32,
    pub max_entropy: f32,
    pub mean_margin: f32,
    pub min_margin: f32,
    pub high_entropy_count: u32,
    pub repetition_count: u32,
}

impl From<RawGenerationSignalWindow> for GenerationSignalWindow {
    fn from(raw: RawGenerationSignalWindow) -> Self {
        Self {
            token_count: raw.token_count,
            mean_entropy: raw.mean_entropy,
            max_entropy: raw.max_entropy,
            mean_margin: raw.mean_margin,
            min_margin: raw.min_margin,
            high_entropy_count: raw.high_entropy_count,
            repetition_count: raw.repetition_count,
        }
    }
}

pub struct ModelInfo {
    raw: *mut RawModelInfo,
}

pub struct SlicePlan {
    raw: *mut RawSlicePlan,
}

struct MediaProjector {
    raw: *mut skippy_ffi::MtmdContext,
}

pub struct StageModel {
    raw: *mut RawModel,
    media: Option<MediaProjector>,
}

pub struct StageSession {
    raw: *mut RawSession,
    token_count: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MediaInput {
    pub bytes: Vec<u8>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MediaPrefill {
    pub token_count: usize,
    pub position: u64,
    pub first_token: i32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MediaPrefillChunkFrame {
    pub token_count: usize,
    pub positions: Vec<i32>,
    pub output: ActivationFrame,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MediaPrefillFrame {
    pub token_count: usize,
    pub position: u64,
    pub positions: Vec<i32>,
    pub output: ActivationFrame,
    pub chunks: Vec<MediaPrefillChunkFrame>,
}

type MediaFrameEval = (
    usize,
    u64,
    Vec<i32>,
    ActivationFrame,
    Vec<MediaPrefillChunkFrame>,
);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StageSessionCheckpoint {
    token_count: u64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct LogitBias {
    pub token_id: i32,
    pub bias: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SamplingConfig {
    pub enabled: bool,
    pub seed: u32,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: i32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub repeat_penalty: f32,
    pub penalty_last_n: i32,
    pub logit_bias: Vec<LogitBias>,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            seed: 0,
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            repeat_penalty: 1.0,
            penalty_last_n: -1,
            logit_bias: Vec::new(),
        }
    }
}

impl SamplingConfig {
    fn as_raw(&self) -> RawSamplingConfig {
        let mut logit_bias = [RawLogitBias {
            token_id: 0,
            bias: 0.0,
        }; MAX_LOGIT_BIAS];
        for (target, source) in logit_bias.iter_mut().zip(
            self.logit_bias
                .iter()
                .take(self.logit_bias.len().min(MAX_LOGIT_BIAS)),
        ) {
            *target = RawLogitBias {
                token_id: source.token_id,
                bias: source.bias,
            };
        }
        RawSamplingConfig {
            version: 1,
            flags: u32::from(self.enabled),
            seed: self.seed,
            top_k: self.top_k,
            penalty_last_n: self.penalty_last_n,
            temperature: self.temperature,
            top_p: self.top_p,
            presence_penalty: self.presence_penalty,
            frequency_penalty: self.frequency_penalty,
            repeat_penalty: self.repeat_penalty,
            logit_bias_count: self.logit_bias.len().min(MAX_LOGIT_BIAS) as u32,
            reserved: 0,
            logit_bias,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatTemplateMessage {
    pub role: String,
    pub content: String,
}

impl ChatTemplateMessage {
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChatTemplateOptions {
    pub add_assistant: bool,
    pub enable_thinking: Option<bool>,
}

impl Default for ChatTemplateOptions {
    fn default() -> Self {
        Self {
            add_assistant: true,
            enable_thinking: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatTemplateJsonOptions {
    pub add_assistant: bool,
    pub enable_thinking: Option<bool>,
    pub tools_json: Option<String>,
    pub tool_choice_json: Option<String>,
    pub parallel_tool_calls: bool,
}

impl Default for ChatTemplateJsonOptions {
    fn default() -> Self {
        Self {
            add_assistant: true,
            enable_thinking: None,
            tools_json: None,
            tool_choice_json: None,
            parallel_tool_calls: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatTemplateJsonResult {
    pub prompt: String,
    pub metadata_json: String,
}

// The experimental C ABI owns synchronization internally for model/session use.
// Rust stage-server access is additionally serialized behind a Mutex.
unsafe impl Send for StageModel {}
unsafe impl Send for StageSession {}
unsafe impl Send for MediaProjector {}

impl MediaProjector {
    fn open(path: &str, model: *mut RawModel) -> Result<Self> {
        let path = CString::new(path.as_bytes())
            .context("projector path contains an interior NUL byte")?;
        let raw_model = unsafe { skippy_ffi::skippy_model_llama_model(model) };
        if raw_model.is_null() {
            return Err(anyhow!("model did not expose a llama_model handle"));
        }
        let mut params = unsafe { skippy_ffi::mtmd_context_params_default() };
        params.use_gpu = true;
        let raw = unsafe { skippy_ffi::mtmd_init_from_file(path.as_ptr(), raw_model, params) };
        if raw.is_null() {
            return Err(anyhow!("failed to load multimodal projector {path:?}"));
        }
        Ok(Self { raw })
    }

    fn marker() -> String {
        let marker = unsafe { skippy_ffi::mtmd_default_marker() };
        if marker.is_null() {
            "<__media__>".to_string()
        } else {
            unsafe { CStr::from_ptr(marker) }
                .to_string_lossy()
                .into_owned()
        }
    }
}

impl Drop for MediaProjector {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe {
                skippy_ffi::mtmd_free(self.raw);
            }
        }
    }
}

impl StageModel {
    pub fn new_dummy() -> Self {
        Self {
            raw: std::ptr::null_mut(),
            media: None,
        }
    }

    pub fn open(path: impl AsRef<Path>, config: &RuntimeConfig) -> Result<Self> {
        let path = path.as_ref();
        let path = CString::new(path.to_string_lossy().as_bytes())
            .context("model path contains an interior NUL byte")?;
        let raw_config = config.as_raw()?;
        let mut raw = ptr::null_mut();
        let mut error = ptr::null_mut();
        let status = unsafe {
            skippy_ffi::skippy_model_open(path.as_ptr(), &raw_config.raw, &mut raw, &mut error)
        };
        ensure_ok(status, error)?;
        if raw.is_null() {
            return Err(anyhow!("skippy_model_open returned a null handle"));
        }
        let media = config
            .projector_path
            .as_deref()
            .map(|projector_path| MediaProjector::open(projector_path, raw))
            .transpose()?;
        Ok(Self { raw, media })
    }

    pub fn open_from_parts(paths: &[impl AsRef<Path>], config: &RuntimeConfig) -> Result<Self> {
        if paths.is_empty() {
            return Err(anyhow!("at least one GGUF part path is required"));
        }
        let paths = paths
            .iter()
            .map(|path| {
                CString::new(path.as_ref().to_string_lossy().as_bytes())
                    .context("part path contains an interior NUL byte")
            })
            .collect::<Result<Vec<_>>>()?;
        let path_ptrs = paths.iter().map(|path| path.as_ptr()).collect::<Vec<_>>();
        let raw_config = config.as_raw()?;
        let mut raw = ptr::null_mut();
        let mut error = ptr::null_mut();
        let status = unsafe {
            skippy_ffi::skippy_model_open_from_parts(
                path_ptrs.as_ptr(),
                path_ptrs.len(),
                &raw_config.raw,
                &mut raw,
                &mut error,
            )
        };
        ensure_ok(status, error)?;
        if raw.is_null() {
            return Err(anyhow!(
                "skippy_model_open_from_parts returned a null handle"
            ));
        }
        let media = config
            .projector_path
            .as_deref()
            .map(|projector_path| MediaProjector::open(projector_path, raw))
            .transpose()?;
        Ok(Self { raw, media })
    }

    pub fn create_session(&self) -> Result<StageSession> {
        let mut raw = ptr::null_mut();
        let mut error = ptr::null_mut();
        let status = unsafe { skippy_ffi::skippy_session_create(self.raw, &mut raw, &mut error) };
        ensure_ok(status, error)?;
        if raw.is_null() {
            return Err(anyhow!("skippy_session_create returned a null handle"));
        }
        Ok(StageSession {
            raw,
            token_count: 0,
        })
    }

    pub fn create_session_from_resident_prefix(
        &self,
        cache_seq_id: i32,
        token_ids: &[i32],
    ) -> Result<StageSession> {
        let mut raw = ptr::null_mut();
        let mut error = ptr::null_mut();
        let status = unsafe {
            skippy_ffi::skippy_session_create_from_resident_prefix(
                self.raw,
                cache_seq_id,
                token_ids.as_ptr(),
                token_ids.len(),
                &mut raw,
                &mut error,
            )
        };
        ensure_ok(status, error)?;
        if raw.is_null() {
            return Err(anyhow!(
                "skippy_session_create_from_resident_prefix returned a null handle"
            ));
        }
        Ok(StageSession {
            raw,
            token_count: u64::try_from(token_ids.len()).context("token count exceeds u64")?,
        })
    }

    pub fn media_marker(&self) -> String {
        MediaProjector::marker()
    }

    pub fn has_media_projector(&self) -> bool {
        self.media.is_some()
    }

    fn eval_media(
        &self,
        session: &mut StageSession,
        prompt: &str,
        media: &[MediaInput],
    ) -> Result<(usize, u64)> {
        let projector = self
            .media
            .as_ref()
            .ok_or_else(|| anyhow!("model was not loaded with a multimodal projector"))?;
        if media.is_empty() {
            return Err(anyhow!("media prefill requires at least one media item"));
        }
        if prompt.is_empty() {
            return Err(anyhow!("media prompt must not be empty"));
        }

        struct Bitmap {
            raw: *mut skippy_ffi::MtmdBitmap,
        }
        impl Drop for Bitmap {
            fn drop(&mut self) {
                if !self.raw.is_null() {
                    unsafe {
                        skippy_ffi::mtmd_bitmap_free(self.raw);
                    }
                }
            }
        }
        struct Chunks {
            raw: *mut skippy_ffi::MtmdInputChunks,
        }
        impl Drop for Chunks {
            fn drop(&mut self) {
                if !self.raw.is_null() {
                    unsafe {
                        skippy_ffi::mtmd_input_chunks_free(self.raw);
                    }
                }
            }
        }

        let mut bitmaps = Vec::with_capacity(media.len());
        for item in media {
            if item.bytes.is_empty() {
                return Err(anyhow!("media item must not be empty"));
            }
            let raw = unsafe {
                skippy_ffi::mtmd_helper_bitmap_init_from_buf(
                    projector.raw,
                    item.bytes.as_ptr(),
                    item.bytes.len(),
                )
            };
            if raw.is_null() {
                return Err(anyhow!("failed to decode media item for projector"));
            }
            bitmaps.push(Bitmap { raw });
        }

        let chunks = Chunks {
            raw: unsafe { skippy_ffi::mtmd_input_chunks_init() },
        };
        if chunks.raw.is_null() {
            return Err(anyhow!("failed to allocate multimodal input chunks"));
        }
        let prompt = CString::new(prompt.as_bytes())
            .context("multimodal prompt contains an interior NUL byte")?;
        let input_text = skippy_ffi::MtmdInputText {
            text: prompt.as_ptr(),
            add_special: true,
            parse_special: true,
        };
        let bitmap_ptrs = bitmaps
            .iter()
            .map(|bitmap| bitmap.raw.cast_const())
            .collect::<Vec<_>>();
        let tokenize_status = unsafe {
            skippy_ffi::mtmd_tokenize(
                projector.raw,
                chunks.raw,
                &input_text,
                bitmap_ptrs.as_ptr(),
                bitmap_ptrs.len(),
            )
        };
        if tokenize_status != 0 {
            return Err(anyhow!(
                "multimodal tokenization failed with status {tokenize_status}"
            ));
        }

        let token_count = unsafe { skippy_ffi::mtmd_helper_get_n_tokens(chunks.raw) };
        if token_count == 0 {
            return Err(anyhow!("multimodal prompt produced no tokens"));
        }
        let n_past = unsafe { skippy_ffi::skippy_session_position(session.raw) };
        if n_past < 0 {
            return Err(anyhow!("skippy session is not initialized"));
        }
        let n_batch = unsafe { skippy_ffi::skippy_session_batch_size(session.raw) };
        if n_batch <= 0 {
            return Err(anyhow!("skippy session has no valid batch size"));
        }
        let lctx = unsafe { skippy_ffi::skippy_session_llama_context(session.raw) };
        if lctx.is_null() {
            return Err(anyhow!(
                "skippy session did not expose a llama_context handle"
            ));
        }
        let mut guard_error = ptr::null_mut();
        let guard_status = unsafe {
            skippy_ffi::skippy_session_begin_external_decode(session.raw, &mut guard_error)
        };
        ensure_ok(guard_status, guard_error)?;

        struct ExternalDecodeGuard(*mut skippy_ffi::Session);

        impl Drop for ExternalDecodeGuard {
            fn drop(&mut self) {
                let mut error = ptr::null_mut();
                unsafe {
                    let _ = skippy_ffi::skippy_session_end_external_decode(self.0, &mut error);
                }
                free_error(error);
            }
        }

        let _external_decode_guard = ExternalDecodeGuard(session.raw);

        let mut new_n_past = 0_i32;
        let eval_status = unsafe {
            skippy_ffi::mtmd_helper_eval_chunks(
                projector.raw,
                lctx,
                chunks.raw,
                n_past,
                0,
                n_batch,
                true,
                &mut new_n_past,
            )
        };
        if eval_status != 0 {
            return Err(anyhow!(
                "multimodal prompt evaluation failed with status {eval_status}"
            ));
        }

        let mut error = ptr::null_mut();
        let status =
            unsafe { skippy_ffi::skippy_session_set_position(session.raw, new_n_past, &mut error) };
        ensure_ok(status, error)?;
        session.token_count =
            u64::try_from(new_n_past).context("multimodal position is negative")?;

        Ok((token_count, session.token_count))
    }

    fn eval_media_frame(
        &self,
        session: &mut StageSession,
        prompt: &str,
        media: &[MediaInput],
    ) -> Result<MediaFrameEval> {
        let projector = self
            .media
            .as_ref()
            .ok_or_else(|| anyhow!("model was not loaded with a multimodal projector"))?;
        if media.is_empty() {
            return Err(anyhow!("media prefill requires at least one media item"));
        }
        if prompt.is_empty() {
            return Err(anyhow!("media prompt must not be empty"));
        }

        struct Bitmap {
            raw: *mut skippy_ffi::MtmdBitmap,
        }
        impl Drop for Bitmap {
            fn drop(&mut self) {
                if !self.raw.is_null() {
                    unsafe {
                        skippy_ffi::mtmd_bitmap_free(self.raw);
                    }
                }
            }
        }
        struct Chunks {
            raw: *mut skippy_ffi::MtmdInputChunks,
        }
        impl Drop for Chunks {
            fn drop(&mut self) {
                if !self.raw.is_null() {
                    unsafe {
                        skippy_ffi::mtmd_input_chunks_free(self.raw);
                    }
                }
            }
        }
        struct ExternalDecodeGuard(*mut skippy_ffi::Session);
        impl Drop for ExternalDecodeGuard {
            fn drop(&mut self) {
                let mut error = ptr::null_mut();
                unsafe {
                    let _ = skippy_ffi::skippy_session_end_external_decode(self.0, &mut error);
                }
                free_error(error);
            }
        }

        let mut bitmaps = Vec::with_capacity(media.len());
        for item in media {
            if item.bytes.is_empty() {
                return Err(anyhow!("media item must not be empty"));
            }
            let raw = unsafe {
                skippy_ffi::mtmd_helper_bitmap_init_from_buf(
                    projector.raw,
                    item.bytes.as_ptr(),
                    item.bytes.len(),
                )
            };
            if raw.is_null() {
                return Err(anyhow!("failed to decode media item for projector"));
            }
            bitmaps.push(Bitmap { raw });
        }

        let chunks = Chunks {
            raw: unsafe { skippy_ffi::mtmd_input_chunks_init() },
        };
        if chunks.raw.is_null() {
            return Err(anyhow!("failed to allocate multimodal input chunks"));
        }
        let prompt = CString::new(prompt.as_bytes())
            .context("multimodal prompt contains an interior NUL byte")?;
        let input_text = skippy_ffi::MtmdInputText {
            text: prompt.as_ptr(),
            add_special: true,
            parse_special: true,
        };
        let bitmap_ptrs = bitmaps
            .iter()
            .map(|bitmap| bitmap.raw.cast_const())
            .collect::<Vec<_>>();
        let tokenize_status = unsafe {
            skippy_ffi::mtmd_tokenize(
                projector.raw,
                chunks.raw,
                &input_text,
                bitmap_ptrs.as_ptr(),
                bitmap_ptrs.len(),
            )
        };
        if tokenize_status != 0 {
            return Err(anyhow!(
                "multimodal tokenization failed with status {tokenize_status}"
            ));
        }

        let token_count = unsafe { skippy_ffi::mtmd_helper_get_n_tokens(chunks.raw) };
        if token_count == 0 {
            return Err(anyhow!("multimodal prompt produced no tokens"));
        }
        let mut n_past = unsafe { skippy_ffi::skippy_session_position(session.raw) };
        if n_past < 0 {
            return Err(anyhow!("skippy session is not initialized"));
        }
        let n_batch = unsafe { skippy_ffi::skippy_session_batch_size(session.raw) };
        if n_batch <= 0 {
            return Err(anyhow!("skippy session has no valid batch size"));
        }
        let lctx = unsafe { skippy_ffi::skippy_session_llama_context(session.raw) };
        if lctx.is_null() {
            return Err(anyhow!(
                "skippy session did not expose a llama_context handle"
            ));
        }

        let mut guard_error = ptr::null_mut();
        let guard_status = unsafe {
            skippy_ffi::skippy_session_begin_external_decode(session.raw, &mut guard_error)
        };
        ensure_ok(guard_status, guard_error)?;
        let _external_decode_guard = ExternalDecodeGuard(session.raw);

        let chunk_count = unsafe { skippy_ffi::mtmd_input_chunks_size(chunks.raw) };
        let use_mrope = unsafe { skippy_ffi::mtmd_decode_use_mrope(projector.raw) };
        let mut token_positions = Vec::<[i32; 4]>::new();
        let mut output_desc: Option<ActivationDesc> = None;
        let mut output_payload = Vec::new();
        let mut chunk_frames = Vec::new();
        let mut copied_tokens = 0usize;
        for index in 0..chunk_count {
            let chunk = unsafe { skippy_ffi::mtmd_input_chunks_get(chunks.raw, index) };
            if chunk.is_null() {
                return Err(anyhow!("multimodal chunk {index} is null"));
            }
            let chunk_type = unsafe { skippy_ffi::mtmd_input_chunk_get_type(chunk) };
            let chunk_tokens = unsafe { skippy_ffi::mtmd_input_chunk_get_n_tokens(chunk) };
            if chunk_tokens == 0 {
                continue;
            }
            if chunk_tokens > n_batch as usize {
                return Err(anyhow!(
                    "multimodal chunk {index} has {chunk_tokens} tokens, exceeding n_batch {n_batch}; increase n_batch for staged media prefill"
                ));
            }
            let chunk_positions = if use_mrope {
                let chunk_positions = match chunk_type {
                    skippy_ffi::MtmdInputChunkType::Image => {
                        let image_tokens =
                            unsafe { skippy_ffi::mtmd_input_chunk_get_tokens_image(chunk) };
                        if image_tokens.is_null() {
                            return Err(anyhow!(
                                "multimodal image chunk {index} has no image tokens"
                            ));
                        }
                        let mut positions = vec![
                            skippy_ffi::MtmdDecoderPos {
                                t: 0,
                                x: 0,
                                y: 0,
                                z: 0,
                            };
                            chunk_tokens
                        ];
                        unsafe {
                            skippy_ffi::mtmd_helper_image_get_decoder_pos(
                                image_tokens,
                                n_past,
                                positions.as_mut_ptr(),
                            );
                        }
                        positions
                            .into_iter()
                            .map(|position| {
                                [
                                    i32::try_from(position.t).unwrap_or(i32::MAX),
                                    i32::try_from(position.y).unwrap_or(i32::MAX),
                                    i32::try_from(position.x).unwrap_or(i32::MAX),
                                    i32::try_from(position.z).unwrap_or(i32::MAX),
                                ]
                            })
                            .collect::<Vec<_>>()
                    }
                    _ => (0..chunk_tokens)
                        .map(|offset| {
                            let position = n_past.saturating_add(offset as i32);
                            [position, position, position, 0]
                        })
                        .collect::<Vec<_>>(),
                };
                token_positions.extend(chunk_positions.iter().copied());
                let mut flattened = Vec::with_capacity(chunk_tokens * 4);
                for dim in 0..4 {
                    flattened.extend(chunk_positions.iter().map(|position| position[dim]));
                }
                flattened
            } else {
                Vec::new()
            };
            let mut new_n_past = n_past;
            let eval_status = unsafe {
                skippy_ffi::mtmd_helper_eval_chunk_single(
                    projector.raw,
                    lctx,
                    chunk,
                    n_past,
                    0,
                    n_batch,
                    false,
                    &mut new_n_past,
                )
            };
            if eval_status != 0 {
                return Err(anyhow!(
                    "multimodal chunk {index} evaluation failed with status {eval_status}"
                ));
            }
            let frame = session.copy_output_activation_frame(chunk_tokens, 0)?;
            if let Some(desc) = output_desc.as_ref() {
                if desc.version != frame.desc.version
                    || desc.dtype != frame.desc.dtype
                    || desc.layout != frame.desc.layout
                    || desc.producer_stage_index != frame.desc.producer_stage_index
                    || desc.layer_start != frame.desc.layer_start
                    || desc.layer_end != frame.desc.layer_end
                    || desc.sequence_count != frame.desc.sequence_count
                    || desc.flags != frame.desc.flags
                {
                    return Err(anyhow!(
                        "multimodal chunk {index} produced incompatible activation descriptor"
                    ));
                }
            } else {
                output_desc = Some(frame.desc);
            }
            copied_tokens = copied_tokens
                .checked_add(chunk_tokens)
                .context("multimodal activation token count overflow")?;
            output_payload.extend_from_slice(&frame.payload);
            chunk_frames.push(MediaPrefillChunkFrame {
                token_count: chunk_tokens,
                positions: chunk_positions,
                output: frame,
            });
            n_past = new_n_past;
        }

        let mut error = ptr::null_mut();
        let status =
            unsafe { skippy_ffi::skippy_session_set_position(session.raw, n_past, &mut error) };
        ensure_ok(status, error)?;
        session.token_count = u64::try_from(n_past).context("multimodal position is negative")?;

        if copied_tokens != token_count {
            return Err(anyhow!(
                "multimodal activation tokens copied {copied_tokens} did not match prompt tokens {token_count}"
            ));
        }
        let mut desc = output_desc
            .ok_or_else(|| anyhow!("multimodal prefill produced no activation output"))?;
        desc.token_count =
            u32::try_from(copied_tokens).context("multimodal token count exceeds u32")?;
        desc.payload_bytes = u64::try_from(output_payload.len())
            .context("multimodal activation payload length exceeds u64")?;
        let positions = if use_mrope {
            let mut positions = Vec::with_capacity(copied_tokens * 4);
            for dim in 0..4 {
                positions.extend(token_positions.iter().map(|position| position[dim]));
            }
            positions
        } else {
            Vec::new()
        };
        Ok((
            token_count,
            session.token_count,
            positions,
            ActivationFrame {
                desc,
                payload: output_payload,
            },
            chunk_frames,
        ))
    }

    pub fn prefill_media(
        &self,
        session: &mut StageSession,
        prompt: &str,
        media: &[MediaInput],
        sampling: Option<&SamplingConfig>,
    ) -> Result<MediaPrefill> {
        let (token_count, position) = self.eval_media(session, prompt, media)?;

        let first_token = session.sample_current(sampling)?;

        Ok(MediaPrefill {
            token_count,
            position,
            first_token,
        })
    }

    pub fn prefill_media_frame(
        &self,
        session: &mut StageSession,
        prompt: &str,
        media: &[MediaInput],
    ) -> Result<MediaPrefillFrame> {
        let (token_count, position, positions, output, chunks) =
            self.eval_media_frame(session, prompt, media)?;
        Ok(MediaPrefillFrame {
            token_count,
            position,
            positions,
            output,
            chunks,
        })
    }

    pub fn tokenize(&self, text: &str, add_special: bool) -> Result<Vec<i32>> {
        let text = CString::new(text).context("text contains an interior NUL byte")?;
        let mut count = 0usize;
        let mut error = ptr::null_mut();
        let status = unsafe {
            skippy_ffi::skippy_tokenize(
                self.raw,
                text.as_ptr(),
                add_special,
                ptr::null_mut(),
                0,
                &mut count,
                &mut error,
            )
        };
        if status != Status::BufferTooSmall && status != Status::Ok {
            ensure_ok(status, error)?;
        } else {
            free_error(error);
        }

        let mut tokens = vec![0_i32; count];
        let mut error = ptr::null_mut();
        let status = unsafe {
            skippy_ffi::skippy_tokenize(
                self.raw,
                text.as_ptr(),
                add_special,
                tokens.as_mut_ptr(),
                tokens.len(),
                &mut count,
                &mut error,
            )
        };
        ensure_ok(status, error)?;
        tokens.truncate(count);
        Ok(tokens)
    }

    pub fn detokenize(&self, tokens: &[i32]) -> Result<String> {
        Ok(String::from_utf8_lossy(&self.detokenize_bytes(tokens)?).into_owned())
    }

    pub fn detokenize_bytes(&self, tokens: &[i32]) -> Result<Vec<u8>> {
        let mut bytes = 0usize;
        let mut error = ptr::null_mut();
        let status = unsafe {
            skippy_ffi::skippy_detokenize(
                self.raw,
                tokens.as_ptr(),
                tokens.len(),
                ptr::null_mut(),
                0,
                &mut bytes,
                &mut error,
            )
        };
        if status != Status::BufferTooSmall && status != Status::Ok {
            ensure_ok(status, error)?;
        } else {
            free_error(error);
        }

        let mut output = vec![0_u8; bytes.max(1)];
        let mut error = ptr::null_mut();
        let status = unsafe {
            skippy_ffi::skippy_detokenize(
                self.raw,
                tokens.as_ptr(),
                tokens.len(),
                output.as_mut_ptr().cast(),
                output.len(),
                &mut bytes,
                &mut error,
            )
        };
        ensure_ok(status, error)?;
        output.truncate(bytes);
        Ok(output)
    }

    pub fn token_is_eog(&self, token: i32) -> Result<bool> {
        let mut is_eog = false;
        let mut error = ptr::null_mut();
        let status =
            unsafe { skippy_ffi::skippy_token_is_eog(self.raw, token, &mut is_eog, &mut error) };
        ensure_ok(status, error)?;
        Ok(is_eog)
    }

    pub fn apply_chat_template(
        &self,
        messages: &[ChatTemplateMessage],
        add_assistant: bool,
    ) -> Result<String> {
        self.apply_chat_template_with_options(
            messages,
            ChatTemplateOptions {
                add_assistant,
                enable_thinking: None,
            },
        )
    }

    pub fn apply_chat_template_with_options(
        &self,
        messages: &[ChatTemplateMessage],
        options: ChatTemplateOptions,
    ) -> Result<String> {
        let roles = messages
            .iter()
            .map(|message| {
                CString::new(message.role.as_str())
                    .context("message role contains an interior NUL byte")
            })
            .collect::<Result<Vec<_>>>()?;
        let contents = messages
            .iter()
            .map(|message| {
                CString::new(message.content.as_str())
                    .context("message content contains an interior NUL byte")
            })
            .collect::<Result<Vec<_>>>()?;
        let raw_messages = roles
            .iter()
            .zip(contents.iter())
            .map(|(role, content)| RawChatMessage {
                role: role.as_ptr(),
                content: content.as_ptr(),
            })
            .collect::<Vec<_>>();

        let mut bytes = 0usize;
        let mut error = ptr::null_mut();
        let status = unsafe {
            skippy_ffi::skippy_apply_chat_template(
                self.raw,
                raw_messages.as_ptr(),
                raw_messages.len(),
                options.add_assistant,
                options.enable_thinking.is_some(),
                options.enable_thinking.unwrap_or(true),
                ptr::null_mut(),
                0,
                &mut bytes,
                &mut error,
            )
        };
        if status != Status::BufferTooSmall && status != Status::Ok {
            ensure_ok(status, error)?;
        } else {
            free_error(error);
        }

        let mut output = vec![0_u8; bytes.max(1)];
        let mut error = ptr::null_mut();
        let status = unsafe {
            skippy_ffi::skippy_apply_chat_template(
                self.raw,
                raw_messages.as_ptr(),
                raw_messages.len(),
                options.add_assistant,
                options.enable_thinking.is_some(),
                options.enable_thinking.unwrap_or(true),
                output.as_mut_ptr().cast(),
                output.len(),
                &mut bytes,
                &mut error,
            )
        };
        ensure_ok(status, error)?;
        output.truncate(bytes);
        String::from_utf8(output).context("chat template output is not valid UTF-8")
    }

    pub fn apply_chat_template_json(
        &self,
        messages_json: &str,
        options: ChatTemplateJsonOptions,
    ) -> Result<ChatTemplateJsonResult> {
        let messages_json =
            CString::new(messages_json).context("messages JSON contains an interior NUL byte")?;
        let tools_json = options
            .tools_json
            .as_deref()
            .map(CString::new)
            .transpose()
            .context("tools JSON contains an interior NUL byte")?;
        let tool_choice_json = options
            .tool_choice_json
            .as_deref()
            .map(CString::new)
            .transpose()
            .context("tool choice JSON contains an interior NUL byte")?;
        let tools_ptr = tools_json
            .as_ref()
            .map(|value| value.as_ptr())
            .unwrap_or(ptr::null());
        let tool_choice_ptr = tool_choice_json
            .as_ref()
            .map(|value| value.as_ptr())
            .unwrap_or(ptr::null());

        let mut prompt_bytes = 0usize;
        let mut metadata_bytes = 0usize;
        let mut error = ptr::null_mut();
        let status = unsafe {
            skippy_ffi::skippy_apply_chat_template_json(
                self.raw,
                messages_json.as_ptr(),
                tools_ptr,
                tool_choice_ptr,
                options.add_assistant,
                options.enable_thinking.is_some(),
                options.enable_thinking.unwrap_or(true),
                options.parallel_tool_calls,
                ptr::null_mut(),
                0,
                &mut prompt_bytes,
                ptr::null_mut(),
                0,
                &mut metadata_bytes,
                &mut error,
            )
        };
        if status != Status::BufferTooSmall && status != Status::Ok {
            ensure_ok(status, error)?;
        } else {
            free_error(error);
        }

        let mut prompt = vec![0_u8; prompt_bytes.max(1)];
        let mut metadata = vec![0_u8; metadata_bytes.max(1)];
        let mut error = ptr::null_mut();
        let status = unsafe {
            skippy_ffi::skippy_apply_chat_template_json(
                self.raw,
                messages_json.as_ptr(),
                tools_ptr,
                tool_choice_ptr,
                options.add_assistant,
                options.enable_thinking.is_some(),
                options.enable_thinking.unwrap_or(true),
                options.parallel_tool_calls,
                prompt.as_mut_ptr().cast(),
                prompt.len(),
                &mut prompt_bytes,
                metadata.as_mut_ptr().cast(),
                metadata.len(),
                &mut metadata_bytes,
                &mut error,
            )
        };
        ensure_ok(status, error)?;
        prompt.truncate(prompt_bytes);
        metadata.truncate(metadata_bytes);
        Ok(ChatTemplateJsonResult {
            prompt: String::from_utf8(prompt).context("chat template output is not valid UTF-8")?,
            metadata_json: String::from_utf8(metadata)
                .context("chat template metadata is not valid UTF-8")?,
        })
    }

    pub fn parse_chat_response_json(
        &self,
        generated_text: &str,
        metadata_json: &str,
        is_partial: bool,
    ) -> Result<String> {
        let generated_text =
            CString::new(generated_text).context("generated text contains an interior NUL byte")?;
        let metadata_json = CString::new(metadata_json)
            .context("chat template metadata contains an interior NUL byte")?;

        let mut bytes = 0usize;
        let mut error = ptr::null_mut();
        let status = unsafe {
            skippy_ffi::skippy_parse_chat_response_json(
                generated_text.as_ptr(),
                metadata_json.as_ptr(),
                is_partial,
                ptr::null_mut(),
                0,
                &mut bytes,
                &mut error,
            )
        };
        if status != Status::BufferTooSmall && status != Status::Ok {
            ensure_ok(status, error)?;
        } else {
            free_error(error);
        }

        let mut output = vec![0_u8; bytes.max(1)];
        let mut error = ptr::null_mut();
        let status = unsafe {
            skippy_ffi::skippy_parse_chat_response_json(
                generated_text.as_ptr(),
                metadata_json.as_ptr(),
                is_partial,
                output.as_mut_ptr().cast(),
                output.len(),
                &mut bytes,
                &mut error,
            )
        };
        ensure_ok(status, error)?;
        output.truncate(bytes);
        String::from_utf8(output).context("parsed chat response is not valid UTF-8")
    }
}

impl Drop for StageModel {
    fn drop(&mut self) {
        self.media.take();
        if !self.raw.is_null() {
            unsafe {
                let _ = skippy_ffi::skippy_model_free(self.raw, ptr::null_mut());
            }
        }
    }
}

impl StageSession {
    pub fn token_count(&self) -> u64 {
        self.token_count
    }

    pub fn batch_size(&self) -> Result<usize> {
        let n_batch = unsafe { skippy_ffi::skippy_session_batch_size(self.raw) };
        if n_batch <= 0 {
            return Err(anyhow!("skippy session has no valid batch size"));
        }
        usize::try_from(n_batch).context("session batch size exceeds usize")
    }

    /// Captures the current position and asks the native runtime to keep an
    /// in-session recurrent checkpoint. Attention KV is restored by trimming
    /// the speculative suffix back to this position.
    pub fn checkpoint(&mut self) -> Result<StageSessionCheckpoint> {
        let mut token_count = 0u64;
        let mut error = ptr::null_mut();
        let status = unsafe {
            skippy_ffi::skippy_checkpoint_session(self.raw, &mut token_count, &mut error)
        };
        ensure_ok(status, error)?;
        self.token_count = token_count;
        Ok(StageSessionCheckpoint { token_count })
    }

    pub fn restore_checkpoint(&mut self, checkpoint: &StageSessionCheckpoint) -> Result<()> {
        let mut error = ptr::null_mut();
        let status = unsafe {
            skippy_ffi::skippy_restore_session_checkpoint(
                self.raw,
                checkpoint.token_count,
                &mut error,
            )
        };
        ensure_ok(status, error)?;
        self.token_count = checkpoint.token_count;
        Ok(())
    }

    pub fn reset(&mut self) -> Result<()> {
        let mut error = ptr::null_mut();
        let status = unsafe { skippy_ffi::skippy_session_reset(self.raw, &mut error) };
        ensure_ok(status, error)?;
        self.token_count = 0;
        Ok(())
    }

    pub fn configure_chat_sampling(
        &mut self,
        metadata_json: &str,
        prompt_token_count: u64,
        sampling: Option<&SamplingConfig>,
    ) -> Result<()> {
        let metadata_json = CString::new(metadata_json)
            .context("chat sampling metadata contains an interior NUL byte")?;
        let raw_sampling = sampling.map(SamplingConfig::as_raw);
        let sampling_ptr = raw_sampling
            .as_ref()
            .map_or(ptr::null(), |sampling| sampling as *const RawSamplingConfig);
        let mut error = ptr::null_mut();
        let status = unsafe {
            skippy_ffi::skippy_session_configure_chat_sampling(
                self.raw,
                sampling_ptr,
                metadata_json.as_ptr(),
                prompt_token_count,
                &mut error,
            )
        };
        ensure_ok(status, error)
    }

    pub fn trim_session(&mut self, token_count: u64) -> Result<()> {
        let mut error = ptr::null_mut();
        let status = unsafe { skippy_ffi::skippy_trim_session(self.raw, token_count, &mut error) };
        ensure_ok(status, error)?;
        self.token_count = token_count;
        Ok(())
    }

    pub fn set_position(&mut self, token_count: u64) -> Result<()> {
        let n_past = i32::try_from(token_count).context("session position exceeds i32")?;
        let mut error = ptr::null_mut();
        let status =
            unsafe { skippy_ffi::skippy_session_set_position(self.raw, n_past, &mut error) };
        ensure_ok(status, error)?;
        self.token_count = token_count;
        Ok(())
    }

    pub fn save_prefix(&mut self, cache_seq_id: i32, token_count: u64) -> Result<()> {
        let mut error = ptr::null_mut();
        let status = unsafe {
            skippy_ffi::skippy_session_save_prefix(self.raw, cache_seq_id, token_count, &mut error)
        };
        ensure_ok(status, error)
    }

    pub fn restore_prefix(&mut self, cache_seq_id: i32, token_ids: &[i32]) -> Result<()> {
        let mut error = ptr::null_mut();
        let status = unsafe {
            skippy_ffi::skippy_session_restore_prefix(
                self.raw,
                cache_seq_id,
                token_ids.as_ptr(),
                token_ids.len(),
                &mut error,
            )
        };
        ensure_ok(status, error)?;
        self.token_count = u64::try_from(token_ids.len()).context("token count exceeds u64")?;
        Ok(())
    }

    pub fn drop_sequence(&mut self, seq_id: i32) -> Result<()> {
        let mut error = ptr::null_mut();
        let status =
            unsafe { skippy_ffi::skippy_session_drop_sequence(self.raw, seq_id, &mut error) };
        ensure_ok(status, error)
    }

    pub fn prefill_chunk(&mut self, token_ids: &[i32]) -> Result<()> {
        let mut output_bytes = 0usize;
        let mut error = ptr::null_mut();
        let status = unsafe {
            skippy_ffi::skippy_prefill_chunk(
                self.raw,
                token_ids.as_ptr(),
                token_ids.len(),
                ptr::null(),
                0,
                ptr::null_mut(),
                0,
                &mut output_bytes,
                &mut error,
            )
        };
        ensure_ok(status, error)?;
        self.token_count = self
            .token_count
            .checked_add(u64::try_from(token_ids.len()).context("token count exceeds u64")?)
            .context("session token count overflow")?;
        Ok(())
    }

    pub fn prefill_chunked(&mut self, token_ids: &[i32]) -> Result<()> {
        if token_ids.is_empty() {
            return Ok(());
        }
        let batch_size = self.batch_size()?.max(1);
        for chunk in token_ids.chunks(batch_size) {
            self.prefill_chunk(chunk)?;
        }
        Ok(())
    }

    pub fn decode_step(&mut self, token_id: i32) -> Result<i32> {
        self.decode_step_sampled(token_id, None)
    }

    pub fn decode_step_sampled(
        &mut self,
        token_id: i32,
        sampling: Option<&SamplingConfig>,
    ) -> Result<i32> {
        let mut output_bytes = 0usize;
        let mut predicted_token = 0_i32;
        let mut error = ptr::null_mut();
        let raw_sampling = sampling.map(SamplingConfig::as_raw);
        let sampling_ptr = raw_sampling
            .as_ref()
            .map_or(ptr::null(), |sampling| sampling as *const RawSamplingConfig);
        let status = unsafe {
            skippy_ffi::skippy_decode_step_sampled(
                self.raw,
                token_id,
                sampling_ptr,
                ptr::null(),
                0,
                ptr::null_mut(),
                0,
                &mut output_bytes,
                &mut predicted_token,
                &mut error,
            )
        };
        ensure_ok(status, error)?;
        self.token_count = self
            .token_count
            .checked_add(1)
            .context("session token count overflow")?;
        Ok(predicted_token)
    }

    pub fn last_token_signal(&mut self) -> Result<TokenSignal> {
        let mut signal = RawTokenSignal::default();
        let mut error = ptr::null_mut();
        let status = unsafe {
            skippy_ffi::skippy_session_last_token_signal(self.raw, &mut signal, &mut error)
        };
        ensure_ok(status, error)?;
        Ok(signal.into())
    }

    pub fn signal_window(&mut self, window_tokens: u32) -> Result<GenerationSignalWindow> {
        let mut window = RawGenerationSignalWindow::default();
        let mut error = ptr::null_mut();
        let status = unsafe {
            skippy_ffi::skippy_session_signal_window(
                self.raw,
                window_tokens,
                &mut window,
                &mut error,
            )
        };
        ensure_ok(status, error)?;
        Ok(window.into())
    }

    pub fn verify_tokens(&mut self, token_ids: &[i32]) -> Result<Vec<i32>> {
        if token_ids.is_empty() {
            return Ok(Vec::new());
        }
        let mut predicted = vec![0_i32; token_ids.len()];
        let mut output_count = 0usize;
        let mut error = ptr::null_mut();
        let status = unsafe {
            skippy_ffi::skippy_verify_tokens(
                self.raw,
                token_ids.as_ptr(),
                token_ids.len(),
                predicted.as_mut_ptr(),
                predicted.len(),
                &mut output_count,
                &mut error,
            )
        };
        ensure_ok(status, error)?;
        self.token_count = self
            .token_count
            .checked_add(u64::try_from(token_ids.len()).context("token count exceeds u64")?)
            .context("session token count overflow")?;
        predicted.truncate(output_count);
        Ok(predicted)
    }

    /// Runs batched verification and restores the prior checkpoint.
    pub fn verify_tokens_rewound(&mut self, token_ids: &[i32]) -> Result<Vec<i32>> {
        if token_ids.is_empty() {
            return Ok(Vec::new());
        }
        let checkpoint = self.checkpoint()?;
        match self.verify_tokens(token_ids) {
            Ok(predicted) => {
                self.restore_checkpoint(&checkpoint)?;
                Ok(predicted)
            }
            Err(error) => {
                let _ = self.restore_checkpoint(&checkpoint);
                Err(error)
            }
        }
    }

    pub fn prefill_chunk_frame(
        &mut self,
        token_ids: &[i32],
        input: Option<&ActivationFrame>,
        output_capacity: usize,
    ) -> Result<ActivationFrame> {
        let (output_desc, output_payload) =
            self.prefill_chunk_frame_raw(token_ids, &[], input, output_capacity)?;
        Ok(ActivationFrame {
            desc: output_desc.into(),
            payload: output_payload,
        })
    }

    pub fn prefill_chunk_frame_with_positions(
        &mut self,
        token_ids: &[i32],
        positions: &[i32],
        input: Option<&ActivationFrame>,
        output_capacity: usize,
    ) -> Result<ActivationFrame> {
        let (output_desc, output_payload) =
            self.prefill_chunk_frame_raw(token_ids, positions, input, output_capacity)?;
        Ok(ActivationFrame {
            desc: output_desc.into(),
            payload: output_payload,
        })
    }

    fn prefill_chunk_frame_raw(
        &mut self,
        token_ids: &[i32],
        positions: &[i32],
        input: Option<&ActivationFrame>,
        output_capacity: usize,
    ) -> Result<(RawActivationDesc, Vec<u8>)> {
        let input_desc = input.map(|frame| frame.desc.as_raw());
        let input_desc_ptr = input_desc
            .as_ref()
            .map_or(ptr::null(), |desc| desc as *const RawActivationDesc);
        let input_payload_ptr = input.map_or(ptr::null(), |frame| frame.payload.as_ptr().cast());
        let mut output_desc = RawActivationDesc {
            version: 0,
            dtype: ActivationDType::Unknown,
            layout: ActivationLayout::Opaque,
            producer_stage_index: -1,
            layer_start: 0,
            layer_end: 0,
            token_count: 0,
            sequence_count: 0,
            payload_bytes: 0,
            flags: 0,
        };
        let mut output_payload = vec![0_u8; output_capacity];
        let mut output_bytes = 0usize;
        let mut error = ptr::null_mut();
        let status = unsafe {
            if positions.is_empty() {
                skippy_ffi::skippy_prefill_chunk_frame(
                    self.raw,
                    token_ids.as_ptr(),
                    token_ids.len(),
                    input_desc_ptr,
                    input_payload_ptr,
                    &mut output_desc,
                    output_payload.as_mut_ptr().cast(),
                    output_payload.len(),
                    &mut output_bytes,
                    &mut error,
                )
            } else {
                skippy_ffi::skippy_prefill_chunk_frame_with_positions(
                    self.raw,
                    token_ids.as_ptr(),
                    token_ids.len(),
                    positions.as_ptr(),
                    positions.len(),
                    input_desc_ptr,
                    input_payload_ptr,
                    &mut output_desc,
                    output_payload.as_mut_ptr().cast(),
                    output_payload.len(),
                    &mut output_bytes,
                    &mut error,
                )
            }
        };
        if status == Status::BufferTooSmall && output_bytes > output_payload.len() {
            free_error(error);
            return self.prefill_chunk_frame_raw(token_ids, positions, input, output_bytes);
        }
        ensure_ok(status, error)?;
        output_payload.truncate(output_bytes);
        self.token_count = self
            .token_count
            .checked_add(u64::try_from(token_ids.len()).context("token count exceeds u64")?)
            .context("session token count overflow")?;
        Ok((output_desc, output_payload))
    }

    pub fn prefill_chunk_frame_sampled(
        &mut self,
        token_ids: &[i32],
        sampling: Option<&SamplingConfig>,
        input: Option<&ActivationFrame>,
        output_capacity: usize,
    ) -> Result<(i32, ActivationFrame)> {
        let (predicted_token, output_desc, output_payload) =
            self.prefill_chunk_frame_sampled_raw(token_ids, &[], sampling, input, output_capacity)?;
        Ok((
            predicted_token,
            ActivationFrame {
                desc: output_desc.into(),
                payload: output_payload,
            },
        ))
    }

    pub fn prefill_chunk_frame_sampled_with_positions(
        &mut self,
        token_ids: &[i32],
        positions: &[i32],
        sampling: Option<&SamplingConfig>,
        input: Option<&ActivationFrame>,
        output_capacity: usize,
    ) -> Result<(i32, ActivationFrame)> {
        let (predicted_token, output_desc, output_payload) = self.prefill_chunk_frame_sampled_raw(
            token_ids,
            positions,
            sampling,
            input,
            output_capacity,
        )?;
        Ok((
            predicted_token,
            ActivationFrame {
                desc: output_desc.into(),
                payload: output_payload,
            },
        ))
    }

    fn prefill_chunk_frame_sampled_raw(
        &mut self,
        token_ids: &[i32],
        positions: &[i32],
        sampling: Option<&SamplingConfig>,
        input: Option<&ActivationFrame>,
        output_capacity: usize,
    ) -> Result<(i32, RawActivationDesc, Vec<u8>)> {
        let input_desc = input.map(|frame| frame.desc.as_raw());
        let input_desc_ptr = input_desc
            .as_ref()
            .map_or(ptr::null(), |desc| desc as *const RawActivationDesc);
        let input_payload_ptr = input.map_or(ptr::null(), |frame| frame.payload.as_ptr().cast());
        let raw_sampling = sampling.map(SamplingConfig::as_raw);
        let sampling_ptr = raw_sampling
            .as_ref()
            .map_or(ptr::null(), |sampling| sampling as *const RawSamplingConfig);
        let mut output_desc = RawActivationDesc {
            version: 0,
            dtype: ActivationDType::Unknown,
            layout: ActivationLayout::Opaque,
            producer_stage_index: -1,
            layer_start: 0,
            layer_end: 0,
            token_count: 0,
            sequence_count: 0,
            payload_bytes: 0,
            flags: 0,
        };
        let mut output_payload = vec![0_u8; output_capacity];
        let mut output_bytes = 0usize;
        let mut predicted_token = 0_i32;
        let mut error = ptr::null_mut();
        let status = unsafe {
            if positions.is_empty() {
                skippy_ffi::skippy_prefill_chunk_frame_sampled(
                    self.raw,
                    token_ids.as_ptr(),
                    token_ids.len(),
                    sampling_ptr,
                    input_desc_ptr,
                    input_payload_ptr,
                    &mut output_desc,
                    output_payload.as_mut_ptr().cast(),
                    output_payload.len(),
                    &mut output_bytes,
                    &mut predicted_token,
                    &mut error,
                )
            } else {
                skippy_ffi::skippy_prefill_chunk_frame_sampled_with_positions(
                    self.raw,
                    token_ids.as_ptr(),
                    token_ids.len(),
                    positions.as_ptr(),
                    positions.len(),
                    sampling_ptr,
                    input_desc_ptr,
                    input_payload_ptr,
                    &mut output_desc,
                    output_payload.as_mut_ptr().cast(),
                    output_payload.len(),
                    &mut output_bytes,
                    &mut predicted_token,
                    &mut error,
                )
            }
        };
        if status == Status::BufferTooSmall && output_bytes > output_payload.len() {
            free_error(error);
            return self.prefill_chunk_frame_sampled_raw(
                token_ids,
                positions,
                sampling,
                input,
                output_bytes,
            );
        }
        ensure_ok(status, error)?;
        output_payload.truncate(output_bytes);
        self.token_count = self
            .token_count
            .checked_add(u64::try_from(token_ids.len()).context("token count exceeds u64")?)
            .context("session token count overflow")?;
        Ok((predicted_token, output_desc, output_payload))
    }

    pub fn decode_step_frame(
        &mut self,
        token_id: i32,
        input: Option<&ActivationFrame>,
        output_capacity: usize,
    ) -> Result<(i32, ActivationFrame)> {
        self.decode_step_frame_sampled(token_id, None, input, output_capacity)
    }

    pub fn decode_step_frame_sampled(
        &mut self,
        token_id: i32,
        sampling: Option<&SamplingConfig>,
        input: Option<&ActivationFrame>,
        output_capacity: usize,
    ) -> Result<(i32, ActivationFrame)> {
        let (predicted_token, output_desc, output_payload) =
            self.decode_step_frame_raw(token_id, sampling, input, output_capacity)?;
        Ok((
            predicted_token,
            ActivationFrame {
                desc: output_desc.into(),
                payload: output_payload,
            },
        ))
    }

    fn decode_step_frame_raw(
        &mut self,
        token_id: i32,
        sampling: Option<&SamplingConfig>,
        input: Option<&ActivationFrame>,
        output_capacity: usize,
    ) -> Result<(i32, RawActivationDesc, Vec<u8>)> {
        let input_desc = input.map(|frame| frame.desc.as_raw());
        let input_desc_ptr = input_desc
            .as_ref()
            .map_or(ptr::null(), |desc| desc as *const RawActivationDesc);
        let input_payload_ptr = input.map_or(ptr::null(), |frame| frame.payload.as_ptr().cast());
        let mut output_desc = RawActivationDesc {
            version: 0,
            dtype: ActivationDType::Unknown,
            layout: ActivationLayout::Opaque,
            producer_stage_index: -1,
            layer_start: 0,
            layer_end: 0,
            token_count: 0,
            sequence_count: 0,
            payload_bytes: 0,
            flags: 0,
        };
        let mut output_payload = vec![0_u8; output_capacity];
        let mut output_bytes = 0usize;
        let mut predicted_token = 0_i32;
        let mut error = ptr::null_mut();
        let raw_sampling = sampling.map(SamplingConfig::as_raw);
        let sampling_ptr = raw_sampling
            .as_ref()
            .map_or(ptr::null(), |sampling| sampling as *const RawSamplingConfig);
        let status = unsafe {
            skippy_ffi::skippy_decode_step_frame_sampled(
                self.raw,
                token_id,
                sampling_ptr,
                input_desc_ptr,
                input_payload_ptr,
                &mut output_desc,
                output_payload.as_mut_ptr().cast(),
                output_payload.len(),
                &mut output_bytes,
                &mut predicted_token,
                &mut error,
            )
        };
        if status == Status::BufferTooSmall && output_bytes > output_payload.len() {
            free_error(error);
            return self.decode_step_frame_raw(token_id, sampling, input, output_bytes);
        }
        ensure_ok(status, error)?;
        output_payload.truncate(output_bytes);
        self.token_count = self
            .token_count
            .checked_add(1)
            .context("session token count overflow")?;
        Ok((predicted_token, output_desc, output_payload))
    }

    pub fn verify_tokens_frame(
        &mut self,
        token_ids: &[i32],
        input: Option<&ActivationFrame>,
        output_capacity: usize,
    ) -> Result<(Vec<i32>, ActivationFrame)> {
        if token_ids.is_empty() {
            return Err(anyhow!("verify_tokens_frame requires at least one token"));
        }
        let (predicted_tokens, output_desc, output_payload) =
            self.verify_tokens_frame_raw(token_ids, input, output_capacity)?;
        Ok((
            predicted_tokens,
            ActivationFrame {
                desc: output_desc.into(),
                payload: output_payload,
            },
        ))
    }

    fn verify_tokens_frame_raw(
        &mut self,
        token_ids: &[i32],
        input: Option<&ActivationFrame>,
        output_capacity: usize,
    ) -> Result<(Vec<i32>, RawActivationDesc, Vec<u8>)> {
        let input_desc = input.map(|frame| frame.desc.as_raw());
        let input_desc_ptr = input_desc
            .as_ref()
            .map_or(ptr::null(), |desc| desc as *const RawActivationDesc);
        let input_payload_ptr = input.map_or(ptr::null(), |frame| frame.payload.as_ptr().cast());
        let mut output_desc = RawActivationDesc {
            version: 0,
            dtype: ActivationDType::Unknown,
            layout: ActivationLayout::Opaque,
            producer_stage_index: -1,
            layer_start: 0,
            layer_end: 0,
            token_count: 0,
            sequence_count: 0,
            payload_bytes: 0,
            flags: 0,
        };
        let mut output_payload = vec![0_u8; output_capacity];
        let mut output_bytes = 0usize;
        let mut predicted = vec![0_i32; token_ids.len()];
        let mut output_token_count = 0usize;
        let mut error = ptr::null_mut();
        let status = unsafe {
            skippy_ffi::skippy_verify_tokens_frame(
                self.raw,
                token_ids.as_ptr(),
                token_ids.len(),
                input_desc_ptr,
                input_payload_ptr,
                &mut output_desc,
                output_payload.as_mut_ptr().cast(),
                output_payload.len(),
                &mut output_bytes,
                predicted.as_mut_ptr(),
                predicted.len(),
                &mut output_token_count,
                &mut error,
            )
        };
        if status == Status::BufferTooSmall && output_bytes > output_payload.len() {
            free_error(error);
            return self.verify_tokens_frame_raw(token_ids, input, output_bytes);
        }
        ensure_ok(status, error)?;
        predicted.truncate(output_token_count);
        output_payload.truncate(output_bytes);
        self.token_count = self
            .token_count
            .checked_add(u64::try_from(token_ids.len()).context("token count exceeds u64")?)
            .context("session token count overflow")?;
        Ok((predicted, output_desc, output_payload))
    }

    pub fn copy_output_activation_frame(
        &mut self,
        token_count: usize,
        output_capacity: usize,
    ) -> Result<ActivationFrame> {
        let (output_desc, output_payload) =
            self.copy_output_activation_frame_raw(token_count, output_capacity)?;
        Ok(ActivationFrame {
            desc: output_desc.into(),
            payload: output_payload,
        })
    }

    fn copy_output_activation_frame_raw(
        &mut self,
        token_count: usize,
        output_capacity: usize,
    ) -> Result<(RawActivationDesc, Vec<u8>)> {
        if token_count == 0 {
            return Err(anyhow!(
                "copy_output_activation_frame requires at least one token"
            ));
        }
        let mut output_desc = RawActivationDesc {
            version: 0,
            dtype: ActivationDType::Unknown,
            layout: ActivationLayout::Opaque,
            producer_stage_index: -1,
            layer_start: 0,
            layer_end: 0,
            token_count: 0,
            sequence_count: 0,
            payload_bytes: 0,
            flags: 0,
        };
        let mut output_payload = vec![0_u8; output_capacity];
        let mut output_bytes = 0usize;
        let mut error = ptr::null_mut();
        let status = unsafe {
            skippy_ffi::skippy_session_copy_output_activation_frame(
                self.raw,
                token_count,
                &mut output_desc,
                output_payload.as_mut_ptr().cast(),
                output_payload.len(),
                &mut output_bytes,
                &mut error,
            )
        };
        if status == Status::BufferTooSmall && output_bytes > output_payload.len() {
            free_error(error);
            return self.copy_output_activation_frame_raw(token_count, output_bytes);
        }
        ensure_ok(status, error)?;
        output_payload.truncate(output_bytes);
        Ok((output_desc, output_payload))
    }

    pub fn sample_current(&mut self, sampling: Option<&SamplingConfig>) -> Result<i32> {
        let raw_sampling = sampling.map(SamplingConfig::as_raw);
        let sampling_ptr = raw_sampling
            .as_ref()
            .map_or(ptr::null(), |sampling| sampling as *const RawSamplingConfig);
        let mut predicted = 0_i32;
        let mut error = ptr::null_mut();
        let status = unsafe {
            skippy_ffi::skippy_session_sample_current(
                self.raw,
                sampling_ptr,
                &mut predicted,
                &mut error,
            )
        };
        ensure_ok(status, error)?;
        Ok(predicted)
    }

    pub fn export_state(&mut self, layer_start: i32, layer_end: i32) -> Result<Vec<u8>> {
        let mut bytes = 0usize;
        let mut error = ptr::null_mut();
        let status = unsafe {
            skippy_ffi::skippy_export_state(
                self.raw,
                layer_start,
                layer_end,
                ptr::null_mut(),
                0,
                &mut bytes,
                &mut error,
            )
        };
        if status != Status::BufferTooSmall && status != Status::Ok {
            ensure_ok(status, error)?;
        } else {
            free_error(error);
        }

        let mut payload = vec![0_u8; bytes];
        let mut written = 0usize;
        let mut error = ptr::null_mut();
        let status = unsafe {
            skippy_ffi::skippy_export_state(
                self.raw,
                layer_start,
                layer_end,
                payload.as_mut_ptr().cast(),
                payload.len(),
                &mut written,
                &mut error,
            )
        };
        ensure_ok(status, error)?;
        payload.truncate(written);
        Ok(payload)
    }

    pub fn import_state(&mut self, layer_start: i32, layer_end: i32, input: &[u8]) -> Result<()> {
        let mut error = ptr::null_mut();
        let status = unsafe {
            skippy_ffi::skippy_import_state(
                self.raw,
                layer_start,
                layer_end,
                input.as_ptr().cast(),
                input.len(),
                &mut error,
            )
        };
        ensure_ok(status, error)
    }

    pub fn import_state_for_token_count(
        &mut self,
        layer_start: i32,
        layer_end: i32,
        input: &[u8],
        token_count: u64,
    ) -> Result<()> {
        self.import_state(layer_start, layer_end, input)?;
        self.token_count = self.token_count.max(token_count);
        Ok(())
    }

    pub fn export_full_state(&mut self, layer_start: i32, layer_end: i32) -> Result<Vec<u8>> {
        let mut bytes = 0usize;
        let mut error = ptr::null_mut();
        let status = unsafe {
            skippy_ffi::skippy_export_full_state(
                self.raw,
                layer_start,
                layer_end,
                ptr::null_mut(),
                0,
                &mut bytes,
                &mut error,
            )
        };
        if status != Status::BufferTooSmall && status != Status::Ok {
            ensure_ok(status, error)?;
        } else {
            free_error(error);
        }

        let mut payload = vec![0_u8; bytes];
        let mut written = 0usize;
        let mut error = ptr::null_mut();
        let status = unsafe {
            skippy_ffi::skippy_export_full_state(
                self.raw,
                layer_start,
                layer_end,
                payload.as_mut_ptr().cast(),
                payload.len(),
                &mut written,
                &mut error,
            )
        };
        ensure_ok(status, error)?;
        payload.truncate(written);
        Ok(payload)
    }

    pub fn import_full_state(
        &mut self,
        layer_start: i32,
        layer_end: i32,
        input: &[u8],
    ) -> Result<()> {
        let mut error = ptr::null_mut();
        let status = unsafe {
            skippy_ffi::skippy_import_full_state(
                self.raw,
                layer_start,
                layer_end,
                input.as_ptr().cast(),
                input.len(),
                &mut error,
            )
        };
        ensure_ok(status, error)
    }

    pub fn import_full_state_for_token_count(
        &mut self,
        layer_start: i32,
        layer_end: i32,
        input: &[u8],
        token_count: u64,
    ) -> Result<()> {
        self.import_full_state(layer_start, layer_end, input)?;
        self.token_count = self.token_count.max(token_count);
        Ok(())
    }

    pub fn export_kv_page(
        &mut self,
        layer_start: i32,
        layer_end: i32,
        token_start: u64,
        token_count: u64,
    ) -> Result<RuntimeKvPage> {
        let mut desc = RawKvPageDesc::default();
        let mut bytes = 0usize;
        let mut error = ptr::null_mut();
        let status = unsafe {
            skippy_ffi::skippy_export_kv_page(
                self.raw,
                layer_start,
                layer_end,
                token_start,
                token_count,
                &mut desc,
                ptr::null_mut(),
                0,
                &mut bytes,
                &mut error,
            )
        };
        if status != Status::BufferTooSmall && status != Status::Ok {
            ensure_ok(status, error)?;
        } else {
            free_error(error);
        }

        let mut payload = vec![0_u8; bytes];
        let mut written = 0usize;
        let mut error = ptr::null_mut();
        let status = unsafe {
            skippy_ffi::skippy_export_kv_page(
                self.raw,
                layer_start,
                layer_end,
                token_start,
                token_count,
                &mut desc,
                payload.as_mut_ptr().cast(),
                payload.len(),
                &mut written,
                &mut error,
            )
        };
        ensure_ok(status, error)?;
        payload.truncate(written);
        Ok(RuntimeKvPage {
            desc: desc.into(),
            payload,
        })
    }

    pub fn export_kv_page_into(
        &mut self,
        layer_start: i32,
        layer_end: i32,
        token_start: u64,
        token_count: u64,
        output: &mut [u8],
    ) -> Result<RuntimeKvPageDesc> {
        let mut desc = RawKvPageDesc::default();
        let mut written = 0usize;
        let mut error = ptr::null_mut();
        let status = unsafe {
            skippy_ffi::skippy_export_kv_page(
                self.raw,
                layer_start,
                layer_end,
                token_start,
                token_count,
                &mut desc,
                output.as_mut_ptr().cast(),
                output.len(),
                &mut written,
                &mut error,
            )
        };
        ensure_ok(status, error)?;
        if written != output.len() {
            anyhow::bail!(
                "KV page export wrote {written} bytes into {} byte output buffer",
                output.len()
            );
        }
        Ok(desc.into())
    }

    pub fn import_kv_page(&mut self, desc: &RuntimeKvPageDesc, payload: &[u8]) -> Result<()> {
        let raw = desc.as_raw();
        let mut error = ptr::null_mut();
        let status = unsafe {
            skippy_ffi::skippy_import_kv_page(
                self.raw,
                &raw,
                payload.as_ptr().cast(),
                payload.len(),
                &mut error,
            )
        };
        ensure_ok(status, error)?;
        self.token_count = self
            .token_count
            .max(desc.token_start.saturating_add(desc.token_count));
        Ok(())
    }

    pub fn export_recurrent_state(&mut self) -> Result<Vec<u8>> {
        let mut bytes = 0usize;
        let mut error = ptr::null_mut();
        let status = unsafe {
            skippy_ffi::skippy_export_recurrent_state(
                self.raw,
                ptr::null_mut(),
                0,
                &mut bytes,
                &mut error,
            )
        };
        if status != Status::BufferTooSmall && status != Status::Ok {
            ensure_ok(status, error)?;
        } else {
            free_error(error);
        }

        let mut payload = vec![0_u8; bytes];
        let mut written = 0usize;
        let mut error = ptr::null_mut();
        let status = unsafe {
            skippy_ffi::skippy_export_recurrent_state(
                self.raw,
                payload.as_mut_ptr().cast(),
                payload.len(),
                &mut written,
                &mut error,
            )
        };
        ensure_ok(status, error)?;
        payload.truncate(written);
        Ok(payload)
    }

    pub fn import_recurrent_state(&mut self, input: &[u8]) -> Result<()> {
        let mut error = ptr::null_mut();
        let status = unsafe {
            skippy_ffi::skippy_import_recurrent_state(
                self.raw,
                input.as_ptr().cast(),
                input.len(),
                &mut error,
            )
        };
        ensure_ok(status, error)
    }

    pub fn import_recurrent_state_for_token_count(
        &mut self,
        input: &[u8],
        token_count: u64,
    ) -> Result<()> {
        self.import_recurrent_state(input)?;
        self.set_position(token_count)
    }
}

impl Drop for StageSession {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe {
                let _ = skippy_ffi::skippy_session_free(self.raw, ptr::null_mut());
            }
        }
    }
}

impl ModelInfo {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let path = CString::new(path.to_string_lossy().as_bytes())
            .context("model path contains an interior NUL byte")?;
        let mut raw = ptr::null_mut();
        let mut error = ptr::null_mut();
        let status =
            unsafe { skippy_ffi::skippy_model_info_open(path.as_ptr(), &mut raw, &mut error) };
        ensure_ok(status, error)?;
        if raw.is_null() {
            return Err(anyhow!("skippy_model_info_open returned a null handle"));
        }
        Ok(Self { raw })
    }

    pub fn tensor_count(&self) -> Result<usize> {
        let mut count = 0usize;
        let mut error = ptr::null_mut();
        let status =
            unsafe { skippy_ffi::skippy_model_info_tensor_count(self.raw, &mut count, &mut error) };
        ensure_ok(status, error)?;
        Ok(count)
    }

    pub fn tensor_at(&self, index: usize) -> Result<TensorInfo> {
        let mut raw = RawTensorInfo {
            name: ptr::null(),
            layer_index: -1,
            role: TensorRole::Unknown,
            ggml_type: 0,
            byte_size: 0,
            element_count: 0,
        };
        let mut error = ptr::null_mut();
        let status = unsafe {
            skippy_ffi::skippy_model_info_tensor_at(self.raw, index, &mut raw, &mut error)
        };
        ensure_ok(status, error)?;

        let name = if raw.name.is_null() {
            String::new()
        } else {
            unsafe { CStr::from_ptr(raw.name) }
                .to_string_lossy()
                .into_owned()
        };

        Ok(TensorInfo {
            name,
            layer_index: u32::try_from(raw.layer_index).ok(),
            role: raw.role,
            ggml_type: raw.ggml_type,
            byte_size: raw.byte_size,
            element_count: raw.element_count,
        })
    }

    pub fn tensors(&self) -> Result<Vec<TensorInfo>> {
        let count = self.tensor_count()?;
        (0..count).map(|index| self.tensor_at(index)).collect()
    }

    pub fn create_slice_plan(&self) -> Result<SlicePlan> {
        let mut raw = ptr::null_mut();
        let mut error = ptr::null_mut();
        let status =
            unsafe { skippy_ffi::skippy_slice_plan_create(self.raw, &mut raw, &mut error) };
        ensure_ok(status, error)?;
        if raw.is_null() {
            return Err(anyhow!("skippy_slice_plan_create returned a null handle"));
        }
        Ok(SlicePlan { raw })
    }

    pub fn write_slice_gguf(
        &self,
        plan: &SlicePlan,
        stage_index: u32,
        output_path: impl AsRef<Path>,
    ) -> Result<()> {
        let stage_index = i32::try_from(stage_index).context("stage_index exceeds i32")?;
        let output_path = output_path.as_ref();
        let output_path = CString::new(output_path.to_string_lossy().as_bytes())
            .context("output path contains an interior NUL byte")?;
        let mut error = ptr::null_mut();
        let status = unsafe {
            skippy_ffi::skippy_write_slice_gguf(
                self.raw,
                plan.raw,
                stage_index,
                output_path.as_ptr(),
                &mut error,
            )
        };
        ensure_ok(status, error)
    }
}

impl Drop for ModelInfo {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe {
                let _ = skippy_ffi::skippy_model_info_free(self.raw, ptr::null_mut());
            }
        }
    }
}

impl SlicePlan {
    pub fn add_layer_range(
        &mut self,
        stage_index: u32,
        layer_start: u32,
        layer_end: u32,
        include_embeddings: bool,
        include_output: bool,
    ) -> Result<()> {
        let mut error = ptr::null_mut();
        let status = unsafe {
            skippy_ffi::skippy_slice_plan_add_layer_range(
                self.raw,
                i32::try_from(stage_index).context("stage_index exceeds i32")?,
                i32::try_from(layer_start).context("layer_start exceeds i32")?,
                i32::try_from(layer_end).context("layer_end exceeds i32")?,
                include_embeddings,
                include_output,
                &mut error,
            )
        };
        ensure_ok(status, error)
    }
}

impl Drop for SlicePlan {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe {
                let _ = skippy_ffi::skippy_slice_plan_free(self.raw, ptr::null_mut());
            }
        }
    }
}

pub fn write_gguf_from_parts(
    input_paths: &[impl AsRef<Path>],
    output_path: impl AsRef<Path>,
) -> Result<()> {
    if input_paths.is_empty() {
        return Err(anyhow!("at least one GGUF part path is required"));
    }

    let input_paths = input_paths
        .iter()
        .map(|path| {
            CString::new(path.as_ref().to_string_lossy().as_bytes())
                .context("input path contains an interior NUL byte")
        })
        .collect::<Result<Vec<_>>>()?;
    let input_ptrs = input_paths
        .iter()
        .map(|path| path.as_ptr())
        .collect::<Vec<_>>();
    let output_path = CString::new(output_path.as_ref().to_string_lossy().as_bytes())
        .context("output path contains an interior NUL byte")?;
    let mut error = ptr::null_mut();
    let status = unsafe {
        skippy_ffi::skippy_write_gguf_from_parts(
            input_ptrs.as_ptr(),
            input_ptrs.len(),
            output_path.as_ptr(),
            &mut error,
        )
    };
    ensure_ok(status, error)
}

fn ensure_ok(status: Status, error: *mut RawError) -> Result<()> {
    if status == Status::Ok {
        free_error(error);
        Ok(())
    } else {
        let message = error_message(error);
        free_error(error);
        if message.is_empty() {
            Err(anyhow!("skippy ABI call failed: {:?}", status))
        } else {
            Err(anyhow!("skippy ABI call failed: {:?}: {}", status, message))
        }
    }
}

fn error_message(error: *mut RawError) -> String {
    if error.is_null() {
        return String::new();
    }

    let message = unsafe { (*error).message };
    if message.is_null() {
        String::new()
    } else {
        unsafe { CStr::from_ptr(message) }
            .to_string_lossy()
            .into_owned()
    }
}

fn free_error(error: *mut RawError) {
    if !error.is_null() {
        unsafe {
            skippy_ffi::skippy_error_free(error);
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::Value;

    use std::{
        env,
        ffi::CString,
        fs,
        io::{LineWriter, Write},
        path::PathBuf,
        ptr,
        sync::{
            atomic::{AtomicUsize, Ordering},
            Arc,
        },
        time::{SystemTime, UNIX_EPOCH},
    };
    use tokio::sync::mpsc::error::TryRecvError;

    use super::{
        flush_native_log_writer, parse_cache_type, parse_layer_assign_index,
        redirect_native_logs_to_file, register_filtered_native_logs, restore_native_logs,
        set_filtered_native_logs_enabled, unregister_filtered_native_logs, write_native_log,
        ChatTemplateMessage, FlashAttentionType, ModelInfo, NativeLogAggregator, NativeLogEvent,
        RuntimeConfig, RuntimeLoadMode, StageModel, TensorRole, GGML_TYPE_F16, GGML_TYPE_Q4_0,
        GGML_TYPE_Q8_0,
    };

    static NATIVE_LOG_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn native_log_test_guard() -> std::sync::MutexGuard<'static, ()> {
        NATIVE_LOG_TEST_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn correctness_model() -> Option<PathBuf> {
        env::var_os("SKIPPY_CORRECTNESS_MODEL").map(PathBuf::from)
    }

    fn infer_layer_end(path: &PathBuf) -> anyhow::Result<u32> {
        let info = ModelInfo::open(path)?;
        let layer_end = info
            .tensors()?
            .into_iter()
            .filter(|tensor| tensor.role == TensorRole::Layer)
            .filter_map(|tensor| tensor.layer_index)
            .max()
            .map(|layer| layer + 1)
            .unwrap_or(1);
        Ok(layer_end)
    }

    #[test]
    fn runtime_config_rejects_empty_selected_backend_device() {
        let config = RuntimeConfig {
            selected_backend_device: Some(String::new()),
            ..RuntimeConfig::default()
        };

        assert_eq!(
            config.validate(),
            Err("selected_backend_device must not be empty")
        );
    }

    #[test]
    fn parse_cache_type_accepts_legacy_mesh_kv_defaults() -> anyhow::Result<()> {
        assert_eq!(parse_cache_type("f16")?, GGML_TYPE_F16);
        assert_eq!(parse_cache_type("q8_0")?, GGML_TYPE_Q8_0);
        assert_eq!(parse_cache_type("q4_0")?, GGML_TYPE_Q4_0);
        Ok(())
    }

    struct FlushCountingWriter {
        flush_count: Arc<AtomicUsize>,
    }

    impl Write for FlushCountingWriter {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            Ok(bytes.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            self.flush_count.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }

    #[test]
    fn native_log_writer_flush_helper_explicitly_flushes_line_writer() {
        let flush_count = Arc::new(AtomicUsize::new(0));
        let writer = FlushCountingWriter {
            flush_count: flush_count.clone(),
        };
        let mut writer = Some(LineWriter::new(writer));
        writer
            .as_mut()
            .expect("writer should exist")
            .write_all(b"buffered native log line\n")
            .expect("write to buffered test writer should succeed");

        flush_native_log_writer(&mut writer);

        assert_eq!(flush_count.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn native_log_writer_flushes_newline_and_partial_line() -> anyhow::Result<()> {
        let _native_log_guard = native_log_test_guard();

        struct RestoreNativeLogs;

        impl Drop for RestoreNativeLogs {
            fn drop(&mut self) {
                restore_native_logs();
            }
        }

        let nanos = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let path = env::temp_dir().join(format!(
            "skippy-native-log-buffer-test-{}-{nanos}.log",
            std::process::id()
        ));
        let _guard = RestoreNativeLogs;
        redirect_native_logs_to_file(&path)?;

        let message = CString::new("buffered native log line\n")?;
        unsafe {
            write_native_log(0, message.as_ptr(), ptr::null_mut());
        }

        let contents = fs::read_to_string(&path)?;
        restore_native_logs();

        fs::remove_file(&path)?;
        assert_eq!(contents, "buffered native log line\n");

        let nanos = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let path = env::temp_dir().join(format!(
            "skippy-native-log-partial-line-test-{}-{nanos}.log",
            std::process::id()
        ));
        let _guard = RestoreNativeLogs;
        redirect_native_logs_to_file(&path)?;

        let message = CString::new("partial native log line")?;
        unsafe {
            write_native_log(0, message.as_ptr(), ptr::null_mut());
        }
        restore_native_logs();

        let contents = fs::read_to_string(&path)?;
        fs::remove_file(&path)?;
        assert_eq!(contents, "partial native log line");
        Ok(())
    }

    #[test]
    fn runtime_config_raw_preserves_selected_backend_device() -> anyhow::Result<()> {
        let config = RuntimeConfig {
            selected_backend_device: Some("MTL0".to_string()),
            ..RuntimeConfig::default()
        };

        let raw = config.as_raw()?;
        let device =
            unsafe { std::ffi::CStr::from_ptr(raw.raw.selected_backend_device).to_string_lossy() };

        assert_eq!(device, "MTL0");
        Ok(())
    }

    #[test]
    fn invalid_selected_backend_device_fails_before_model_open() {
        let config = RuntimeConfig {
            selected_backend_device: Some("definitely-not-a-device".to_string()),
            ..RuntimeConfig::default()
        };

        let error = match StageModel::open("/definitely/missing/model.gguf", &config) {
            Ok(_) => panic!("invalid device should fail before model load"),
            Err(error) => error.to_string(),
        };

        assert!(
            error.contains("unknown selected backend device: definitely-not-a-device"),
            "unexpected error: {error}"
        );
    }

    fn open_correctness_model(model_path: &PathBuf) -> anyhow::Result<StageModel> {
        let layer_end = infer_layer_end(model_path)?;
        let config = RuntimeConfig {
            stage_index: 0,
            layer_start: 0,
            layer_end,
            ctx_size: 256,
            lane_count: 1,
            n_batch: None,
            n_ubatch: None,
            n_threads: None,
            n_threads_batch: None,
            n_gpu_layers: 0,
            selected_backend_device: None,
            cache_type_k: GGML_TYPE_F16,
            cache_type_v: GGML_TYPE_F16,
            flash_attn_type: FlashAttentionType::Auto,
            load_mode: RuntimeLoadMode::RuntimeSlice,
            projector_path: None,
            include_embeddings: true,
            include_output: true,
            filter_tensors_on_load: false,
        };
        StageModel::open(model_path, &config)
    }

    #[test]
    fn chat_template_applies_when_model_is_configured() -> anyhow::Result<()> {
        let Some(model_path) = correctness_model() else {
            eprintln!("skipping chat template smoke: SKIPPY_CORRECTNESS_MODEL is not set");
            return Ok(());
        };
        let model = open_correctness_model(&model_path)?;
        let prompt = model.apply_chat_template(
            &[
                ChatTemplateMessage::new("system", "You are concise."),
                ChatTemplateMessage::new("user", "Template smoke prompt."),
            ],
            true,
        )?;
        assert!(prompt.contains("Template smoke prompt."));
        assert!(prompt.len() >= "Template smoke prompt.".len());
        Ok(())
    }

    #[test]
    fn aggregator_preserves_backend_summary_lines() {
        let mut aggregator = NativeLogAggregator::default();
        assert_eq!(
            aggregator.process_line("backend_init succeeded"),
            vec![NativeLogEvent {
                message: "backend_init succeeded".to_string(),
                category: "backend",
                params: Vec::new(),
            }]
        );
        assert_eq!(
            aggregator.process_line("llama_backend_init: GGML_CUDA"),
            vec![NativeLogEvent {
                message: "llama_backend_init: GGML_CUDA".to_string(),
                category: "backend",
                params: Vec::new(),
            }]
        );
    }

    #[test]
    fn aggregator_ignores_non_backend_cuda_mentions() {
        let mut aggregator = NativeLogAggregator::default();
        assert!(aggregator
            .process_line("CUDA kernel launch for attention")
            .is_empty());
        assert!(aggregator.process_line("offloading to CUDA").is_empty());
    }

    #[test]
    fn aggregator_builds_metadata_summary_and_progress() {
        let mut aggregator = NativeLogAggregator::default();
        assert_eq!(
            aggregator.process_line(
                "llama_model_loader: loaded meta data with 10 key-value pairs and 100 tensors from model.gguf (version GGUF V3)"
            ),
            vec![NativeLogEvent {
                message: "model load plan: metadata rows=10, tensor rows=100".to_string(),
                category: "model",
                params: Vec::new(),
            }]
        );

        for (idx, line) in [
            "llama_model_loader: - kv   0: general.architecture str = qwen35",
            "llama_model_loader: - kv   1: general.name str = Qwen 3.5 4B",
            "llama_model_loader: - kv   2: general.type str = model",
            "llama_model_loader: - kv   3: general.size_label str = 4B",
            "llama_model_loader: - kv   4: qwen35.context_length u32 = 40960",
            "llama_model_loader: - kv   5: qwen35.block_count u32 = 36",
            "llama_model_loader: - kv   6: qwen35.embedding_length u32 = 2560",
            "llama_model_loader: - kv   7: qwen35.feed_forward_length u32 = 9728",
            "llama_model_loader: - kv   8: qwen35.attention.head_count u32 = 32",
            "llama_model_loader: - kv   9: qwen35.attention.head_count_kv u32 = 8",
        ]
        .iter()
        .enumerate()
        {
            let events = aggregator.process_line(line);
            assert!(
                events
                    .iter()
                    .any(|event| event.message.contains(&format!("{}%", (idx + 1) * 10))),
                "expected {}% metadata progress in {:?}",
                (idx + 1) * 10,
                events
            );
        }

        let flush_events = aggregator.process_line("llm_load_print_meta: version = 3");
        assert!(flush_events
            .iter()
            .any(|event| event.message == "llm_load_print_meta: version = 3"));
        assert!(flush_events
            .iter()
            .any(|event| event.message == "Reading model metadata..."));
        assert!(flush_events.iter().any(|event| event
            .params
            .iter()
            .any(|(key, value)| key == "architecture"
                && value == &Value::String("qwen35".to_string()))));
    }

    #[test]
    fn aggregator_emits_tensor_progress_from_type_summaries() {
        let mut aggregator = NativeLogAggregator::default();
        aggregator.process_line(
            "llama_model_loader: loaded meta data with 46 key-value pairs and 100 tensors from model.gguf (version GGUF V3)",
        );

        let first = aggregator.process_line("llama_model_loader: - type  f32:  30 tensors");
        assert!(first
            .iter()
            .any(|event| event.message.contains("tensors 10%")));
        assert!(first
            .iter()
            .any(|event| event.message.contains("tensors 30%")));

        let second = aggregator.process_line("llama_model_loader: - type q4_k:  70 tensors");
        assert!(second
            .iter()
            .any(|event| event.message.contains("tensors 100%")));
        assert!(second.iter().any(|event| {
            event.message == "Reading tensor groups..."
                && event
                    .params
                    .iter()
                    .any(|(key, value)| key == "f32" && value == &Value::from(30_u64))
                && event
                    .params
                    .iter()
                    .any(|(key, value)| key == "q4_K" && value == &Value::from(70_u64))
        }));
    }

    #[test]
    fn aggregator_preserves_memory_summary_lines() {
        let mut aggregator = NativeLogAggregator::default();
        assert_eq!(
            aggregator.process_line("VRAM used: 12.4 GB"),
            vec![NativeLogEvent {
                message: "VRAM used: 12.4 GB".to_string(),
                category: "memory",
                params: Vec::new(),
            }]
        );
    }

    #[test]
    fn aggregator_tracks_kv_cache_layer_progress_without_double_counting() {
        let mut aggregator = NativeLogAggregator::default();
        let plan = aggregator.process_line(
            "llama_kv_cache: size = 4096.00 MiB (131072 cells,   8 layers,  2/1 seqs), K (f16): 2048.00 MiB, V (f16): 2048.00 MiB",
        );
        assert_eq!(
            plan,
            vec![NativeLogEvent {
                message: "kv cache plan: layer rows=8".to_string(),
                category: "kv_cache",
                params: Vec::new(),
            }]
        );

        let first = aggregator.process_line("llama_kv_cache: layer   0: filtered");
        assert!(first
            .iter()
            .any(|event| event.message.contains("kv cache 10%")));

        let duplicate = aggregator.process_line("llama_kv_cache: layer   0: dev = MTL0");
        assert!(duplicate.is_empty());

        for layer in 1..8 {
            aggregator.process_line(&format!("llama_kv_cache: layer   {layer}: filtered"));
        }

        let summary = aggregator.process_line("llama_kv_cache: attn_rot = 128");
        assert_eq!(
            summary,
            vec![NativeLogEvent {
                message: "llama_kv_cache: attn_rot = 128".to_string(),
                category: "kv_cache",
                params: Vec::new(),
            }]
        );
    }

    #[test]
    fn aggregator_preserves_tokenizer_summary_lines() {
        let mut aggregator = NativeLogAggregator::default();
        assert_eq!(
            aggregator.process_line("init_tokenizer: initializing tokenizer for type 2"),
            vec![NativeLogEvent {
                message: "init_tokenizer: initializing tokenizer for type 2".to_string(),
                category: "tokenizer",
                params: Vec::new(),
            }]
        );
    }

    #[test]
    fn aggregator_suppresses_print_info_lines() {
        let mut aggregator = NativeLogAggregator::default();
        assert!(aggregator
            .process_line("print_info: n_vocab               = 248320")
            .is_empty());
    }

    #[test]
    fn aggregator_rejects_empty_and_whitespace_lines() {
        let mut aggregator = NativeLogAggregator::default();
        assert!(aggregator.process_line("").is_empty());
        assert!(aggregator.process_line("   ").is_empty());
    }

    #[test]
    fn aggregator_suppresses_raw_noise_lines() {
        let mut aggregator = NativeLogAggregator::default();
        assert!(aggregator
            .process_line("clip_model_loader: tensor[0]: n_dims = 1, name = v.blk.0.attn_out.bias")
            .is_empty());
        assert!(aggregator
            .process_line("tokenizer.ggml.tokens arr[str,248320] = [\"!\", ...]")
            .is_empty());
    }

    #[test]
    fn write_native_log_respects_forwarding_flag() {
        let _native_log_guard = native_log_test_guard();

        struct ResetNativeLogForwarding;

        impl Drop for ResetNativeLogForwarding {
            fn drop(&mut self) {
                unregister_filtered_native_logs();
                set_filtered_native_logs_enabled(false);
            }
        }

        let _reset = ResetNativeLogForwarding;
        unregister_filtered_native_logs();
        let mut rx = register_filtered_native_logs();

        let line =
            CString::new("init_tokenizer: initializing tokenizer for type 2\n").expect("cstring");

        set_filtered_native_logs_enabled(false);
        unsafe { write_native_log(0, line.as_ptr(), ptr::null_mut()) };
        assert_eq!(rx.try_recv(), Err(TryRecvError::Empty));

        set_filtered_native_logs_enabled(true);
        unsafe { write_native_log(0, line.as_ptr(), ptr::null_mut()) };
        assert_eq!(
            rx.blocking_recv(),
            Some(NativeLogEvent {
                message: "init_tokenizer: initializing tokenizer for type 2".to_string(),
                category: "tokenizer",
                params: Vec::new(),
            })
        );
    }

    #[test]
    fn register_filtered_native_logs_replaces_receiver_cleanly() {
        let _native_log_guard = native_log_test_guard();

        struct ResetNativeLogForwarding;

        impl Drop for ResetNativeLogForwarding {
            fn drop(&mut self) {
                unregister_filtered_native_logs();
                set_filtered_native_logs_enabled(false);
            }
        }

        let _reset = ResetNativeLogForwarding;
        unregister_filtered_native_logs();
        set_filtered_native_logs_enabled(true);

        let first = register_filtered_native_logs();
        drop(first);
        let mut second = register_filtered_native_logs();

        let line =
            CString::new("init_tokenizer: initializing tokenizer for type 2\n").expect("cstring");
        unsafe { write_native_log(0, line.as_ptr(), ptr::null_mut()) };

        assert_eq!(
            second.blocking_recv(),
            Some(NativeLogEvent {
                message: "init_tokenizer: initializing tokenizer for type 2".to_string(),
                category: "tokenizer",
                params: Vec::new(),
            })
        );
    }

    #[test]
    fn parse_layer_assign_index_extracts_layer_number() {
        assert_eq!(
            parse_layer_assign_index("load_tensors: layer   0 assigned to device CUDA0"),
            Some(0)
        );
        assert_eq!(
            parse_layer_assign_index("load_tensors: layer  63 assigned to device CUDA0"),
            Some(63)
        );
        assert_eq!(
            parse_layer_assign_index("load_tensors: layer   5 assigned to device CPU"),
            Some(5)
        );
        assert_eq!(
            parse_layer_assign_index("llm_load_tensors: offloaded 64/65 layers"),
            None
        );
        assert_eq!(
            parse_layer_assign_index("load_tensors: layer   0 computation graph"),
            None
        );
    }

    #[test]
    fn aggregator_tracks_layer_assign_progress_using_block_count() {
        let mut aggregator = NativeLogAggregator::default();

        aggregator.process_line(
            "llama_model_loader: loaded meta data with 10 key-value pairs and 100 tensors from model.gguf (version GGUF V3)"
        );
        for line in [
            "llama_model_loader: - kv   0: general.architecture str = qwen35",
            "llama_model_loader: - kv   1: general.name str = Qwen 3.5 4B",
            "llama_model_loader: - kv   2: general.type str = model",
            "llama_model_loader: - kv   3: general.size_label str = 4B",
            "llama_model_loader: - kv   4: qwen35.context_length u32 = 40960",
            "llama_model_loader: - kv   5: qwen35.block_count u32 = 4",
            "llama_model_loader: - kv   6: qwen35.embedding_length u32 = 2560",
            "llama_model_loader: - kv   7: qwen35.feed_forward_length u32 = 9728",
            "llama_model_loader: - kv   8: qwen35.attention.head_count u32 = 32",
            "llama_model_loader: - kv   9: qwen35.attention.head_count_kv u32 = 8",
        ] {
            aggregator.process_line(line);
        }
        aggregator.process_line("llm_load_print_meta: version = 3");

        let e0 = aggregator.process_line("load_tensors: layer   0 assigned to device CUDA0");
        assert!(e0.iter().any(|event| event.message.contains("layers 10%")));
        assert!(e0.iter().any(|event| event.message.contains("layers 20%")));

        let e1 = aggregator.process_line("load_tensors: layer   1 assigned to device CUDA0");
        assert!(e1.iter().any(|event| event.message.contains("layers 50%")));

        let e2 = aggregator.process_line("load_tensors: layer   2 assigned to device CUDA0");
        assert!(e2.iter().any(|event| event.message.contains("layers 70%")));

        let e3 = aggregator.process_line("load_tensors: layer   3 assigned to device CUDA0");
        let pcts: Vec<&str> = e3
            .iter()
            .filter_map(|ev| {
                if ev.message.contains("layers") && ev.message.contains('%') {
                    Some(ev.message.as_str())
                } else {
                    None
                }
            })
            .collect();
        assert!(
            pcts.iter().any(|m| m.contains("100%")),
            "expected layers 100% at final layer, got {:?}",
            pcts
        );
    }
}
