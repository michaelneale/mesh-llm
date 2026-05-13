pub const ABI_VERSION_MAJOR: u32 = 0;
pub const ABI_VERSION_MINOR: u32 = 1;
pub const ABI_VERSION_PATCH: u32 = 23;

use std::ffi::{c_char, c_int, c_void};

pub type LlamaLogCallback =
    Option<unsafe extern "C" fn(level: c_int, text: *const c_char, user_data: *mut c_void)>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum Status {
    Ok = 0,
    Error = 1,
    InvalidArgument = 2,
    Unsupported = 3,
    BufferTooSmall = 4,
    IoError = 5,
    ModelError = 6,
    RuntimeError = 7,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum LoadMode {
    RuntimeSlice = 0,
    LayerPackage = 1,
    ArtifactSlice = 2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum TensorRole {
    Unknown = 0,
    Metadata = 1,
    Tokenizer = 2,
    Embedding = 3,
    Layer = 4,
    FinalNorm = 5,
    Output = 6,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum ActivationDType {
    Unknown = 0,
    F32 = 1,
    F16 = 2,
    Bf16 = 3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum ActivationLayout {
    Opaque = 0,
    TokenMajor = 1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum BackendDeviceType {
    Cpu = 0,
    Gpu = 1,
    IGpu = 2,
    Accel = 3,
    Meta = 4,
}

pub const BACKEND_DEVICE_CAP_ASYNC: u64 = 1 << 0;
pub const BACKEND_DEVICE_CAP_HOST_BUFFER: u64 = 1 << 1;
pub const BACKEND_DEVICE_CAP_BUFFER_FROM_HOST_PTR: u64 = 1 << 2;
pub const BACKEND_DEVICE_CAP_EVENTS: u64 = 1 << 3;

#[repr(C)]
pub struct Error {
    pub status: Status,
    pub message: *const c_char,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RuntimeConfig {
    pub stage_index: i32,
    pub layer_start: i32,
    pub layer_end: i32,
    pub ctx_size: i32,
    pub lane_count: i32,
    pub n_batch: i32,
    pub n_ubatch: i32,
    pub n_threads: i32,
    pub n_threads_batch: i32,
    pub n_gpu_layers: i32,
    pub cache_type_k: i32,
    pub cache_type_v: i32,
    pub flash_attn_type: i32,
    pub load_mode: LoadMode,
    pub disable_repack: bool,
    pub filter_tensors_on_load: bool,
    pub include_embeddings: bool,
    pub include_output: bool,
    pub selected_backend_device: *const c_char,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BackendDevice {
    pub version: u32,
    pub name: *const c_char,
    pub description: *const c_char,
    pub device_id: *const c_char,
    pub memory_free: u64,
    pub memory_total: u64,
    pub device_type: BackendDeviceType,
    pub caps: u64,
}

#[repr(C)]
pub struct Model {
    _private: [u8; 0],
}

#[repr(C)]
pub struct Session {
    _private: [u8; 0],
}

#[repr(C)]
pub struct ModelInfo {
    _private: [u8; 0],
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ChatMessage {
    pub role: *const c_char,
    pub content: *const c_char,
}

#[repr(C)]
pub struct SlicePlan {
    _private: [u8; 0],
}

#[repr(C)]
pub struct MtmdContext {
    _private: [u8; 0],
}

#[repr(C)]
pub struct MtmdBitmap {
    _private: [u8; 0],
}

#[repr(C)]
pub struct MtmdInputChunks {
    _private: [u8; 0],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MtmdInputChunkType {
    Text = 0,
    Image = 1,
    Audio = 2,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MtmdDecoderPos {
    pub t: u32,
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MtmdInputText {
    pub text: *const c_char,
    pub add_special: bool,
    pub parse_special: bool,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MtmdContextParams {
    pub use_gpu: bool,
    pub print_timings: bool,
    pub n_threads: c_int,
    pub image_marker: *const c_char,
    pub media_marker: *const c_char,
    pub flash_attn_type: c_int,
    pub warmup: bool,
    pub image_min_tokens: c_int,
    pub image_max_tokens: c_int,
    pub cb_eval: *mut c_void,
    pub cb_eval_user_data: *mut c_void,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TensorInfo {
    pub name: *const c_char,
    pub layer_index: i32,
    pub role: TensorRole,
    pub ggml_type: u32,
    pub byte_size: u64,
    pub element_count: u64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
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

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct LogitBias {
    pub token_id: i32,
    pub bias: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SamplingConfig {
    pub version: u32,
    pub flags: u32,
    pub seed: u32,
    pub top_k: i32,
    pub penalty_last_n: i32,
    pub temperature: f32,
    pub top_p: f32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub repeat_penalty: f32,
    pub logit_bias_count: u32,
    pub reserved: u32,
    pub logit_bias: [LogitBias; 256],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct KvPageDesc {
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

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct TokenSignal {
    pub entropy: f32,
    pub top_logprob: f32,
    pub second_logprob: f32,
    pub margin: f32,
    pub top_token: i32,
    pub second_token: i32,
}

#[repr(C)]
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

extern "C" {
    pub fn llama_log_set(log_callback: LlamaLogCallback, user_data: *mut c_void);

    pub fn skippy_status_string(status: Status) -> *const c_char;
    pub fn skippy_error_free(error: *mut Error);

    pub fn skippy_backend_device_count(out_count: *mut usize, out_error: *mut *mut Error)
        -> Status;

    pub fn skippy_backend_device_at(
        index: usize,
        out_device: *mut BackendDevice,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_model_open(
        path: *const c_char,
        config: *const RuntimeConfig,
        out_model: *mut *mut Model,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_model_open_from_parts(
        paths: *const *const c_char,
        path_count: usize,
        config: *const RuntimeConfig,
        out_model: *mut *mut Model,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_model_free(model: *mut Model, out_error: *mut *mut Error) -> Status;

    pub fn skippy_model_llama_model(model: *const Model) -> *const Opaque;

    pub fn skippy_session_create(
        model: *mut Model,
        out_session: *mut *mut Session,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_session_create_from_resident_prefix(
        model: *mut Model,
        cache_seq_id: i32,
        token_ids: *const i32,
        token_count: usize,
        out_session: *mut *mut Session,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_session_llama_context(session: *mut Session) -> *mut Opaque;

    pub fn skippy_session_position(session: *const Session) -> i32;

    pub fn skippy_session_native_seq_id(session: *const Session) -> i32;

    pub fn skippy_session_batch_size(session: *const Session) -> i32;

    pub fn skippy_session_begin_external_decode(
        session: *mut Session,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_session_end_external_decode(
        session: *mut Session,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_session_set_position(
        session: *mut Session,
        n_past: i32,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_session_sample_current(
        session: *mut Session,
        sampling: *const SamplingConfig,
        out_predicted_token: *mut i32,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_session_configure_chat_sampling(
        session: *mut Session,
        sampling: *const SamplingConfig,
        metadata_json: *const c_char,
        prompt_token_count: u64,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_session_reset(session: *mut Session, out_error: *mut *mut Error) -> Status;

    pub fn skippy_checkpoint_session(
        session: *mut Session,
        out_token_count: *mut u64,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_restore_session_checkpoint(
        session: *mut Session,
        token_count: u64,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_session_free(session: *mut Session, out_error: *mut *mut Error) -> Status;

    pub fn skippy_prefill_chunk(
        session: *mut Session,
        token_ids: *const i32,
        token_count: usize,
        input_activations: *const c_void,
        input_activation_bytes: usize,
        output_activations: *mut c_void,
        output_activation_capacity: usize,
        out_output_activation_bytes: *mut usize,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_decode_step(
        session: *mut Session,
        token_id: i32,
        input_activation: *const c_void,
        input_activation_bytes: usize,
        output_activation: *mut c_void,
        output_activation_capacity: usize,
        out_output_activation_bytes: *mut usize,
        out_predicted_token: *mut i32,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_verify_tokens(
        session: *mut Session,
        token_ids: *const i32,
        token_count: usize,
        output_tokens: *mut i32,
        output_token_capacity: usize,
        out_token_count: *mut usize,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_decode_step_sampled(
        session: *mut Session,
        token_id: i32,
        sampling: *const SamplingConfig,
        input_activation: *const c_void,
        input_activation_bytes: usize,
        output_activation: *mut c_void,
        output_activation_capacity: usize,
        out_output_activation_bytes: *mut usize,
        out_predicted_token: *mut i32,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_prefill_chunk_frame(
        session: *mut Session,
        token_ids: *const i32,
        token_count: usize,
        input_desc: *const ActivationDesc,
        input_payload: *const c_void,
        output_desc: *mut ActivationDesc,
        output_payload: *mut c_void,
        output_payload_capacity: usize,
        out_output_payload_bytes: *mut usize,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_prefill_chunk_frame_sampled(
        session: *mut Session,
        token_ids: *const i32,
        token_count: usize,
        sampling: *const SamplingConfig,
        input_desc: *const ActivationDesc,
        input_payload: *const c_void,
        output_desc: *mut ActivationDesc,
        output_payload: *mut c_void,
        output_payload_capacity: usize,
        out_output_payload_bytes: *mut usize,
        out_predicted_token: *mut i32,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_prefill_chunk_frame_with_positions(
        session: *mut Session,
        token_ids: *const i32,
        token_count: usize,
        positions: *const i32,
        position_count: usize,
        input_desc: *const ActivationDesc,
        input_payload: *const c_void,
        output_desc: *mut ActivationDesc,
        output_payload: *mut c_void,
        output_payload_capacity: usize,
        out_output_payload_bytes: *mut usize,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_prefill_chunk_frame_sampled_with_positions(
        session: *mut Session,
        token_ids: *const i32,
        token_count: usize,
        positions: *const i32,
        position_count: usize,
        sampling: *const SamplingConfig,
        input_desc: *const ActivationDesc,
        input_payload: *const c_void,
        output_desc: *mut ActivationDesc,
        output_payload: *mut c_void,
        output_payload_capacity: usize,
        out_output_payload_bytes: *mut usize,
        out_predicted_token: *mut i32,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_decode_step_frame(
        session: *mut Session,
        token_id: i32,
        input_desc: *const ActivationDesc,
        input_payload: *const c_void,
        output_desc: *mut ActivationDesc,
        output_payload: *mut c_void,
        output_payload_capacity: usize,
        out_output_payload_bytes: *mut usize,
        out_predicted_token: *mut i32,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_verify_tokens_frame(
        session: *mut Session,
        token_ids: *const i32,
        token_count: usize,
        input_desc: *const ActivationDesc,
        input_payload: *const c_void,
        output_desc: *mut ActivationDesc,
        output_payload: *mut c_void,
        output_payload_capacity: usize,
        out_output_payload_bytes: *mut usize,
        output_tokens: *mut i32,
        output_token_capacity: usize,
        out_token_count: *mut usize,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_decode_step_frame_sampled(
        session: *mut Session,
        token_id: i32,
        sampling: *const SamplingConfig,
        input_desc: *const ActivationDesc,
        input_payload: *const c_void,
        output_desc: *mut ActivationDesc,
        output_payload: *mut c_void,
        output_payload_capacity: usize,
        out_output_payload_bytes: *mut usize,
        out_predicted_token: *mut i32,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_session_copy_output_activation_frame(
        session: *mut Session,
        token_count: usize,
        output_desc: *mut ActivationDesc,
        output_payload: *mut c_void,
        output_payload_capacity: usize,
        out_output_payload_bytes: *mut usize,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_session_last_token_signal(
        session: *mut Session,
        out_signal: *mut TokenSignal,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_session_signal_window(
        session: *mut Session,
        window_tokens: u32,
        out_window: *mut GenerationSignalWindow,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_trim_session(
        session: *mut Session,
        token_count: u64,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_export_state(
        session: *mut Session,
        layer_start: i32,
        layer_end: i32,
        output: *mut c_void,
        output_capacity: usize,
        out_bytes: *mut usize,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_import_state(
        session: *mut Session,
        layer_start: i32,
        layer_end: i32,
        input: *const c_void,
        input_bytes: usize,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_export_full_state(
        session: *mut Session,
        layer_start: i32,
        layer_end: i32,
        output: *mut c_void,
        output_capacity: usize,
        out_bytes: *mut usize,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_import_full_state(
        session: *mut Session,
        layer_start: i32,
        layer_end: i32,
        input: *const c_void,
        input_bytes: usize,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_export_kv_page(
        session: *mut Session,
        layer_start: i32,
        layer_end: i32,
        token_start: u64,
        token_count: u64,
        out_desc: *mut KvPageDesc,
        output: *mut c_void,
        output_capacity: usize,
        out_bytes: *mut usize,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_import_kv_page(
        session: *mut Session,
        desc: *const KvPageDesc,
        input: *const c_void,
        input_bytes: usize,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_export_recurrent_state(
        session: *mut Session,
        output: *mut c_void,
        output_capacity: usize,
        out_bytes: *mut usize,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_import_recurrent_state(
        session: *mut Session,
        input: *const c_void,
        input_bytes: usize,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_session_save_prefix(
        session: *mut Session,
        cache_seq_id: i32,
        token_count: u64,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_session_restore_prefix(
        session: *mut Session,
        cache_seq_id: i32,
        token_ids: *const i32,
        token_count: usize,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_session_drop_sequence(
        session: *mut Session,
        seq_id: i32,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_tokenize(
        model: *mut Model,
        text: *const c_char,
        add_special: bool,
        output_tokens: *mut i32,
        output_token_capacity: usize,
        out_token_count: *mut usize,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_detokenize(
        model: *mut Model,
        tokens: *const i32,
        token_count: usize,
        output_text: *mut c_char,
        output_text_capacity: usize,
        out_text_bytes: *mut usize,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_token_is_eog(
        model: *mut Model,
        token_id: i32,
        out_is_eog: *mut bool,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_apply_chat_template(
        model: *mut Model,
        messages: *const ChatMessage,
        message_count: usize,
        add_assistant: bool,
        override_enable_thinking: bool,
        enable_thinking: bool,
        output_text: *mut c_char,
        output_text_capacity: usize,
        out_text_bytes: *mut usize,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_apply_chat_template_json(
        model: *mut Model,
        messages_json: *const c_char,
        tools_json: *const c_char,
        tool_choice_json: *const c_char,
        add_assistant: bool,
        override_enable_thinking: bool,
        enable_thinking: bool,
        parallel_tool_calls: bool,
        output_text: *mut c_char,
        output_text_capacity: usize,
        out_text_bytes: *mut usize,
        output_metadata_json: *mut c_char,
        output_metadata_json_capacity: usize,
        out_metadata_json_bytes: *mut usize,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_parse_chat_response_json(
        generated_text: *const c_char,
        metadata_json: *const c_char,
        is_partial: bool,
        output_message_json: *mut c_char,
        output_message_json_capacity: usize,
        out_message_json_bytes: *mut usize,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_model_info_open(
        path: *const c_char,
        out_info: *mut *mut ModelInfo,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_model_info_free(info: *mut ModelInfo, out_error: *mut *mut Error) -> Status;

    pub fn skippy_model_info_tensor_count(
        info: *mut ModelInfo,
        out_count: *mut usize,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_model_info_tensor_at(
        info: *mut ModelInfo,
        index: usize,
        out_tensor: *mut TensorInfo,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_slice_plan_create(
        info: *mut ModelInfo,
        out_plan: *mut *mut SlicePlan,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_slice_plan_free(plan: *mut SlicePlan, out_error: *mut *mut Error) -> Status;

    pub fn skippy_slice_plan_add_layer_range(
        plan: *mut SlicePlan,
        stage_index: i32,
        layer_start: i32,
        layer_end: i32,
        include_embeddings: bool,
        include_output: bool,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_write_slice_gguf(
        info: *mut ModelInfo,
        plan: *const SlicePlan,
        stage_index: i32,
        output_path: *const c_char,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_write_gguf_from_parts(
        input_paths: *const *const c_char,
        input_count: usize,
        output_path: *const c_char,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn mtmd_default_marker() -> *const c_char;

    pub fn mtmd_helper_log_set(log_callback: LlamaLogCallback, user_data: *mut c_void);

    pub fn mtmd_context_params_default() -> MtmdContextParams;

    pub fn mtmd_init_from_file(
        mmproj_fname: *const c_char,
        text_model: *const Opaque,
        ctx_params: MtmdContextParams,
    ) -> *mut MtmdContext;

    pub fn mtmd_free(ctx: *mut MtmdContext);

    pub fn mtmd_helper_bitmap_init_from_buf(
        ctx: *mut MtmdContext,
        buf: *const u8,
        len: usize,
    ) -> *mut MtmdBitmap;

    pub fn mtmd_bitmap_free(bitmap: *mut MtmdBitmap);

    pub fn mtmd_input_chunks_init() -> *mut MtmdInputChunks;

    pub fn mtmd_input_chunks_free(chunks: *mut MtmdInputChunks);

    pub fn mtmd_tokenize(
        ctx: *mut MtmdContext,
        output: *mut MtmdInputChunks,
        text: *const MtmdInputText,
        bitmaps: *const *const MtmdBitmap,
        n_bitmaps: usize,
    ) -> c_int;

    pub fn mtmd_helper_get_n_tokens(chunks: *const MtmdInputChunks) -> usize;

    pub fn mtmd_helper_get_n_pos(chunks: *const MtmdInputChunks) -> i32;

    pub fn mtmd_input_chunks_size(chunks: *const MtmdInputChunks) -> usize;

    pub fn mtmd_input_chunks_get(chunks: *const MtmdInputChunks, index: usize) -> *const Opaque;

    pub fn mtmd_decode_use_mrope(ctx: *const MtmdContext) -> bool;

    pub fn mtmd_input_chunk_get_type(chunk: *const Opaque) -> MtmdInputChunkType;

    pub fn mtmd_input_chunk_get_n_tokens(chunk: *const Opaque) -> usize;

    pub fn mtmd_input_chunk_get_tokens_image(chunk: *const Opaque) -> *const Opaque;

    pub fn mtmd_helper_image_get_decoder_pos(
        image: *const Opaque,
        pos_0: i32,
        out_pos: *mut MtmdDecoderPos,
    );

    pub fn mtmd_helper_eval_chunks(
        ctx: *mut MtmdContext,
        lctx: *mut Opaque,
        chunks: *const MtmdInputChunks,
        n_past: i32,
        seq_id: i32,
        n_batch: i32,
        logits_last: bool,
        new_n_past: *mut i32,
    ) -> c_int;

    pub fn mtmd_helper_eval_chunk_single(
        ctx: *mut MtmdContext,
        lctx: *mut Opaque,
        chunk: *const Opaque,
        n_past: i32,
        seq_id: i32,
        n_batch: i32,
        logits_last: bool,
        new_n_past: *mut i32,
    ) -> c_int;
}

pub type Opaque = c_void;
