pub mod backend;
pub mod chat;
pub mod common;
pub mod completions;
pub mod errors;
pub mod hooks;
pub mod models;
pub mod responses;
pub mod router;
pub mod sse;

pub use backend::{
    CancellationToken, ChatCompletionStream, CompletionStream, OpenAiBackend, OpenAiRequestContext,
    OpenAiResult,
};
pub use chat::{
    message_content_to_text, messages_to_plain_prompt, AssistantMessage, ChatCompletionChoice,
    ChatCompletionChunk, ChatCompletionChunkChoice, ChatCompletionDelta, ChatCompletionRequest,
    ChatCompletionResponse, ChatMessage, MessageContent, MessageContentPart,
};
pub use common::{
    completion_id, normalize_reasoning_template_options, now_unix_secs, FinishReason,
    PromptCacheRetention, ReasoningConfig, ReasoningEffort, ReasoningTemplateOptions, StopSequence,
    StreamOptions, Usage, THINKING_BOOLEAN_ALIASES,
};
pub use completions::{
    CompletionChoice, CompletionChunk, CompletionChunkChoice, CompletionPrompt, CompletionRequest,
    CompletionResponse,
};
pub use errors::{already_openai_error, map_upstream_error_body, OpenAiError, OpenAiErrorKind};
pub use hooks::{
    chat_mesh_hooks_enabled, first_chat_media, inject_text_into_chat_messages,
    set_chat_mesh_hooks_enabled, ChatHookAction, ChatHookOutcome, ChatMediaKind, ChatMediaRef,
    GenerationHookSignals, HookedOpenAiBackend, OpenAiHookPolicy, PrefillHookSignals,
    MESH_HOOKS_FIELD,
};
pub use models::{ModelId, ModelIdError, ModelObject, ModelsResponse};
pub use responses::{
    chat_usage_to_responses_usage, normalize_openai_compat_request, parse_chat_stream_chunk,
    responses_stream_completed_event, responses_stream_completed_event_with_sequence,
    responses_stream_content_part_added_event, responses_stream_content_part_done_event,
    responses_stream_created_event, responses_stream_created_event_with_sequence,
    responses_stream_delta_event, responses_stream_delta_event_with_logprobs,
    responses_stream_delta_event_with_logprobs_and_sequence,
    responses_stream_output_item_added_event, responses_stream_output_item_done_event,
    responses_stream_text_done_event, responses_stream_text_done_event_with_sequence,
    stream_usage_to_responses_usage, translate_chat_completion_response_to_responses,
    translate_chat_completion_to_responses, NormalizationOutcome, ResponseAdapterMode,
    ResponsesRequest, StreamUsage,
};
pub use router::{
    router, router_for, router_for_with_config, router_with_config, OpenAiFrontendConfig,
};
