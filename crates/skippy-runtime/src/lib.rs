use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::path::Path;
use std::ptr;

use anyhow::{anyhow, Context, Result};
use skippy_ffi::{
    ActivationDType, ActivationDesc as RawActivationDesc, ActivationLayout,
    ChatMessage as RawChatMessage, Error as RawError,
    GenerationSignalWindow as RawGenerationSignalWindow, LoadMode, LogitBias as RawLogitBias,
    Model as RawModel, ModelInfo as RawModelInfo, RuntimeConfig as RawRuntimeConfig,
    SamplingConfig as RawSamplingConfig, Session as RawSession, SlicePlan as RawSlicePlan, Status,
    TensorInfo as RawTensorInfo, TensorRole, TokenSignal as RawTokenSignal,
};

pub mod package;

pub const MAX_LOGIT_BIAS: usize = 256;
pub const GGML_TYPE_F16: u32 = 1;
pub const GGML_TYPE_Q4_0: u32 = 2;
pub const GGML_TYPE_Q8_0: u32 = 8;

pub use skippy_ffi::LoadMode as RuntimeLoadMode;
pub use skippy_ffi::{
    ActivationDType as RuntimeActivationDType, ActivationLayout as RuntimeActivationLayout,
};

pub fn suppress_native_logs() {
    unsafe {
        skippy_ffi::llama_log_set(Some(discard_native_log), ptr::null_mut());
    }
}

pub fn restore_native_logs() {
    unsafe {
        skippy_ffi::llama_log_set(None, ptr::null_mut());
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
    pub n_gpu_layers: i32,
    pub selected_backend_device: Option<String>,
    pub cache_type_k: u32,
    pub cache_type_v: u32,
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
                n_gpu_layers: self.n_gpu_layers,
                cache_type_k: i32::try_from(self.cache_type_k)
                    .context("cache_type_k exceeds i32")?,
                cache_type_v: i32::try_from(self.cache_type_v)
                    .context("cache_type_v exceeds i32")?,
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
            n_gpu_layers: 0,
            selected_backend_device: None,
            cache_type_k: GGML_TYPE_F16,
            cache_type_v: GGML_TYPE_F16,
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
pub struct MediaPrefillFrame {
    pub token_count: usize,
    pub position: u64,
    pub output: ActivationFrame,
}

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
            return Err(anyhow!("skippy model did not expose a llama_model handle"));
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
        let (token_count, position) = self.eval_media(session, prompt, media)?;
        let output = session.copy_output_activation_frame(token_count, 0)?;
        Ok(MediaPrefillFrame {
            token_count,
            position,
            output,
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

    pub fn trim_session(&mut self, token_count: u64) -> Result<()> {
        let mut error = ptr::null_mut();
        let status = unsafe { skippy_ffi::skippy_trim_session(self.raw, token_count, &mut error) };
        ensure_ok(status, error)?;
        self.token_count = token_count;
        Ok(())
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
            self.prefill_chunk_frame_raw(token_ids, input, output_capacity)?;
        Ok(ActivationFrame {
            desc: output_desc.into(),
            payload: output_payload,
        })
    }

    fn prefill_chunk_frame_raw(
        &mut self,
        token_ids: &[i32],
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
        };
        if status == Status::BufferTooSmall && output_bytes > output_payload.len() {
            free_error(error);
            return self.prefill_chunk_frame_raw(token_ids, input, output_bytes);
        }
        ensure_ok(status, error)?;
        output_payload.truncate(output_bytes);
        self.token_count = self
            .token_count
            .checked_add(u64::try_from(token_ids.len()).context("token count exceeds u64")?)
            .context("session token count overflow")?;
        Ok((output_desc, output_payload))
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
    use std::{env, path::PathBuf};

    use super::{
        parse_cache_type, ChatTemplateMessage, ModelInfo, RuntimeConfig, RuntimeLoadMode,
        StageModel, TensorRole, GGML_TYPE_F16, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0,
    };

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
            n_gpu_layers: 0,
            selected_backend_device: None,
            cache_type_k: GGML_TYPE_F16,
            cache_type_v: GGML_TYPE_F16,
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
}
