use super::*;
use std::{env, fs, net::SocketAddr};

use serde_json::json;

const MM_MODEL_ENV: &str = "SKIPPY_MM_MODEL";
const MM_PROJECTOR_ENV: &str = "SKIPPY_MM_PROJECTOR";
const MM_IMAGE_ENV: &str = "SKIPPY_MM_IMAGE";
const MM_ACTIVATION_WIDTH_ENV: &str = "SKIPPY_MM_ACTIVATION_WIDTH";
const MM_SPLIT_LAYER_ENV: &str = "SKIPPY_MM_SPLIT_LAYER";
const MM_CTX_SIZE_ENV: &str = "SKIPPY_MM_CTX_SIZE";
const MM_MAX_TOKENS_ENV: &str = "SKIPPY_MM_MAX_TOKENS";

struct MultimodalSmokeFixture {
    model_path: PathBuf,
    projector_path: PathBuf,
    image_path: PathBuf,
    layer_end: u32,
    activation_width: i32,
    ctx_size: u32,
    max_tokens: u32,
}

fn multimodal_smoke_fixture() -> Result<Option<MultimodalSmokeFixture>> {
    let model_path = match env::var_os(MM_MODEL_ENV) {
        Some(path) => PathBuf::from(path),
        None => {
            eprintln!(
                    "skipping real multimodal smoke: set {MM_MODEL_ENV}, {MM_PROJECTOR_ENV}, and {MM_IMAGE_ENV}"
                );
            return Ok(None);
        }
    };
    let projector_path = match env::var_os(MM_PROJECTOR_ENV) {
        Some(path) => PathBuf::from(path),
        None => {
            eprintln!("skipping real multimodal smoke: set {MM_PROJECTOR_ENV}");
            return Ok(None);
        }
    };
    let image_path = match env::var_os(MM_IMAGE_ENV) {
        Some(path) => PathBuf::from(path),
        None => {
            eprintln!("skipping real multimodal smoke: set {MM_IMAGE_ENV}");
            return Ok(None);
        }
    };
    if !model_path.is_file() {
        bail!(
            "{MM_MODEL_ENV} does not point at a file: {}",
            model_path.display()
        );
    }
    if !projector_path.is_file() {
        bail!(
            "{MM_PROJECTOR_ENV} does not point at a file: {}",
            projector_path.display()
        );
    }
    if !image_path.is_file() {
        bail!(
            "{MM_IMAGE_ENV} does not point at a file: {}",
            image_path.display()
        );
    }
    let layer_end = model_layer_count(&model_path)?;
    let activation_width = env_i32(MM_ACTIVATION_WIDTH_ENV)?
        .map(Ok)
        .unwrap_or_else(|| infer_activation_width(&model_path))?;
    let ctx_size = env_u32(MM_CTX_SIZE_ENV)?.unwrap_or(2048);
    let max_tokens = env_u32(MM_MAX_TOKENS_ENV)?.unwrap_or(16);
    Ok(Some(MultimodalSmokeFixture {
        model_path,
        projector_path,
        image_path,
        layer_end,
        activation_width,
        ctx_size,
        max_tokens,
    }))
}

fn env_i32(name: &str) -> Result<Option<i32>> {
    env::var(name)
        .ok()
        .map(|value| {
            value
                .parse::<i32>()
                .with_context(|| format!("parse {name}={value:?} as i32"))
        })
        .transpose()
}

fn env_u32(name: &str) -> Result<Option<u32>> {
    env::var(name)
        .ok()
        .map(|value| {
            value
                .parse::<u32>()
                .with_context(|| format!("parse {name}={value:?} as u32"))
        })
        .transpose()
}

fn infer_activation_width(path: &Path) -> Result<i32> {
    let info =
        ModelInfo::open(path).with_context(|| format!("open model info {}", path.display()))?;
    let candidates = [
        "attn_norm.weight",
        "attention_norm.weight",
        "input_layernorm.weight",
        "ln_1.weight",
    ];
    let width = info
        .tensors()?
        .into_iter()
        .filter(|tensor| tensor.layer_index == Some(0))
        .find(|tensor| {
            candidates
                .iter()
                .any(|suffix| tensor.name.ends_with(suffix))
        })
        .map(|tensor| tensor.element_count)
        .ok_or_else(|| {
            anyhow!(
                "could not infer activation width for {}; set {MM_ACTIVATION_WIDTH_ENV}",
                path.display()
            )
        })?;
    i32::try_from(width).context("activation width exceeds i32")
}

fn unsupported_code(error: OpenAiError) -> Option<String> {
    error.body().error.code
}

#[test]
fn chat_runtime_feature_guard_allows_noop_parity_fields() {
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "test",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [],
        "tool_choice": null,
        "parallel_tool_calls": false,
        "response_format": {"type": "text"}
    }))
    .unwrap();

    ensure_chat_runtime_features_supported(&request).unwrap();
}

#[test]
fn chat_runtime_feature_guard_rejects_structured_output() {
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "test",
        "messages": [{"role": "user", "content": "hi"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "answer", "schema": {"type": "object"}}
        }
    }))
    .unwrap();

    let error = ensure_chat_runtime_features_supported(&request).unwrap_err();
    assert_eq!(
        unsupported_code(error),
        Some("unsupported_model_feature".to_string())
    );
}

#[test]
fn chat_runtime_feature_guard_allows_tool_calls() {
    for payload in [
        json!({
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "lookup"}}]
        }),
        json!({
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
            "tool_choice": "auto"
        }),
        json!({
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
            "parallel_tool_calls": true
        }),
    ] {
        let request: ChatCompletionRequest = serde_json::from_value(payload).unwrap();
        ensure_chat_runtime_features_supported(&request).unwrap();
    }
}

fn tool_request() -> ChatCompletionRequest {
    serde_json::from_value(json!({
        "model": "test",
        "messages": [{"role": "user", "content": "look this up"}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Look up a value",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"}
                    },
                    "required": ["city"]
                }
            }
        }]
    }))
    .unwrap()
}

#[test]
fn parses_llama_message_tool_calls() {
    let request = tool_request();
    let parsed = parsed_tool_calls_from_message_json(
        r#"{"role":"assistant","content":null,"tool_calls":[{"id":"call_123","type":"function","function":{"name":"lookup","arguments":"{\"city\":\"Sydney\"}"}}]}"#,
        &request,
    )
    .expect("tool call");

    assert_eq!(parsed.content, None);
    assert_eq!(parsed.tool_calls[0]["id"], "call_123");
    assert_eq!(parsed.tool_calls[0]["function"]["name"], "lookup");
    assert_eq!(
        parsed.tool_calls[0]["function"]["arguments"],
        "{\"city\":\"Sydney\"}"
    );
}

#[test]
fn llama_message_tool_parser_rejects_unknown_tool() {
    let request = tool_request();
    let parsed = parsed_tool_calls_from_message_json(
        r#"{"role":"assistant","content":null,"tool_calls":[{"id":"call_123","type":"function","function":{"name":"lookup","arguments":"{\"city\":\"Sydney\"}"}}]}"#,
        &request,
    )
    .expect("tool call");
    assert_eq!(parsed.tool_calls[0]["function"]["name"], "lookup");

    assert!(parsed_tool_calls_from_message_json(
        r#"{"role":"assistant","content":null,"tool_calls":[{"id":"call_123","type":"function","function":{"name":"shell","arguments":"{}"}}]}"#,
        &request
    )
    .is_none());
}

#[test]
fn tool_choice_limits_allowed_tool_name() {
    let mut request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "test",
        "messages": [{"role": "user", "content": "look this up"}],
        "tools": [
            {"type": "function", "function": {"name": "lookup"}},
            {"type": "function", "function": {"name": "search"}}
        ],
        "tool_choice": {"type": "function", "function": {"name": "lookup"}}
    }))
    .unwrap();

    assert!(parsed_tool_calls_from_message_json(
        r#"{"role":"assistant","content":null,"tool_calls":[{"id":"call_123","type":"function","function":{"name":"search","arguments":"{}"}}]}"#,
        &request
    )
    .is_none());

    request.tool_choice = Some(json!("search"));
    let parsed = parsed_tool_calls_from_message_json(
        r#"{"role":"assistant","content":null,"tool_calls":[{"id":"call_123","type":"function","function":{"name":"search","arguments":"{}"}}]}"#,
        &request,
    )
    .expect("selected tool call");
    assert_eq!(parsed.tool_calls[0]["function"]["name"], "search");
}

#[test]
fn parallel_tool_calls_false_keeps_first_call() {
    let mut request = tool_request();
    request.parallel_tool_calls = Some(false);
    let parsed = parsed_tool_calls_from_message_json(
        r#"{"role":"assistant","content":null,"tool_calls":[
            {"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{\"city\":\"Sydney\"}"}},
            {"id":"call_2","type":"function","function":{"name":"lookup","arguments":"{\"city\":\"Melbourne\"}"}}
        ]}"#,
        &request,
    )
    .expect("tool calls");

    assert_eq!(parsed.tool_calls.as_array().unwrap().len(), 1);
    assert_eq!(
        parsed.tool_calls[0]["function"]["arguments"],
        "{\"city\":\"Sydney\"}"
    );
}

#[test]
fn tool_call_stream_delta_adds_indexes() {
    let delta = tool_calls_stream_delta(json!([
        {"id":"call_a","type":"function","function":{"name":"lookup","arguments":"{}"}},
        {"id":"call_b","type":"function","function":{"name":"lookup","arguments":"{}"}}
    ]));

    assert_eq!(delta[0]["index"], 0);
    assert_eq!(delta[1]["index"], 1);
}

#[test]
fn chat_message_generation_value_preserves_tool_history() {
    let message: openai_frontend::ChatMessage = serde_json::from_value(json!({
        "role": "assistant",
        "content": null,
        "tool_calls": [{
            "id": "call_123",
            "type": "function",
            "function": {"name": "lookup", "arguments": "{\"city\":\"Sydney\"}"}
        }]
    }))
    .unwrap();
    let mut media = Vec::new();

    let value = chat_message_generation_value(&message, "<__media__>", &mut media).unwrap();

    assert_eq!(value["content"], Value::Null);
    assert_eq!(value["tool_calls"][0]["id"], "call_123");
    assert_eq!(value["tool_calls"][0]["function"]["name"], "lookup");
}

#[test]
fn chat_runtime_feature_guard_rejects_logprobs() {
    for payload in [
        json!({
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
            "logprobs": true
        }),
        json!({
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
            "logprobs": false,
            "top_logprobs": 1
        }),
    ] {
        let request: ChatCompletionRequest = serde_json::from_value(payload).unwrap();
        let error = ensure_chat_runtime_features_supported(&request).unwrap_err();
        assert_eq!(
            unsupported_code(error),
            Some("unsupported_model_feature".to_string())
        );
    }
}

#[test]
fn completion_runtime_feature_guard_rejects_logprobs() {
    let request: CompletionRequest = serde_json::from_value(json!({
        "model": "test",
        "prompt": "hi",
        "logprobs": 2
    }))
    .unwrap();

    let error = ensure_completion_runtime_features_supported(&request).unwrap_err();
    assert_eq!(
        unsupported_code(error),
        Some("unsupported_model_feature".to_string())
    );
}

fn multimodal_stage_config(
    fixture: &MultimodalSmokeFixture,
    stage_id: &str,
    stage_index: u32,
    layer_start: u32,
    layer_end: u32,
    bind_addr: SocketAddr,
) -> StageConfig {
    StageConfig {
        run_id: "mm-smoke-run".to_string(),
        topology_id: "mm-smoke-topology".to_string(),
        model_id: "mm-smoke".to_string(),
        package_ref: None,
        manifest_sha256: None,
        source_model_path: None,
        source_model_sha256: None,
        source_model_bytes: None,
        materialized_path: None,
        materialized_pinned: false,
        model_path: Some(fixture.model_path.to_string_lossy().to_string()),
        projector_path: (stage_index == 0)
            .then(|| fixture.projector_path.to_string_lossy().to_string()),
        stage_id: stage_id.to_string(),
        stage_index,
        layer_start,
        layer_end,
        ctx_size: fixture.ctx_size,
        lane_count: 1,
        n_batch: None,
        n_ubatch: None,
        n_gpu_layers: 999,
        cache_type_k: "f16".to_string(),
        cache_type_v: "f16".to_string(),
        flash_attn_type: skippy_protocol::FlashAttentionType::Auto,
        filter_tensors_on_load: false,
        selected_device: None,
        load_mode: skippy_protocol::LoadMode::RuntimeSlice,
        bind_addr: bind_addr.to_string(),
        upstream: None,
        downstream: None,
    }
}

fn local_openai_backend(config: StageConfig) -> Result<StageOpenAiBackend> {
    let runtime = load_runtime(&config)?.context("load smoke runtime")?;
    let ctx_size = usize::try_from(config.ctx_size).unwrap_or(usize::MAX);
    Ok(StageOpenAiBackend {
        runtime,
        telemetry: Telemetry::new(
            None,
            1,
            config.clone(),
            crate::telemetry::TelemetryLevel::Off,
        ),
        config,
        model_id: "mm-smoke".to_string(),
        default_max_tokens: 16,
        ctx_size,
        mode: OpenAiBackendMode::LocalRuntime,
        draft: None,
        speculative_window: 0,
        adaptive_speculative_window: false,
        generation_limit: Arc::new(Semaphore::new(1)),
        hook_policy: None,
    })
}

fn multimodal_chat_request(fixture: &MultimodalSmokeFixture) -> Result<ChatCompletionRequest> {
    let image = fs::read(&fixture.image_path)
        .with_context(|| format!("read smoke image {}", fixture.image_path.display()))?;
    let mime_type = match fixture
        .image_path
        .extension()
        .and_then(|extension| extension.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase()
        .as_str()
    {
        "jpg" | "jpeg" => "image/jpeg",
        "webp" => "image/webp",
        _ => "image/png",
    };
    let encoded = base64::engine::general_purpose::STANDARD.encode(image);
    serde_json::from_value(json!({
            "model": "mm-smoke",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image briefly."},
                    {"type": "image_url", "image_url": {"url": format!("data:{mime_type};base64,{encoded}")}}
                ]
            }],
            "max_tokens": fixture.max_tokens,
            "temperature": 0.0
        }))
        .context("build multimodal smoke request")
}

fn assert_nonempty_chat_response(response: ChatCompletionResponse) {
    let content = response
        .choices
        .first()
        .and_then(|choice| choice.message.content.as_deref())
        .unwrap_or_default()
        .trim();
    assert!(
        !content.is_empty(),
        "expected non-empty multimodal response"
    );
}

fn available_loopback_addr() -> Result<SocketAddr> {
    let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
    Ok(listener.local_addr()?)
}

fn split_layer_for_fixture(fixture: &MultimodalSmokeFixture) -> Result<Option<u32>> {
    if fixture.layer_end < 2 {
        eprintln!("skipping split multimodal smoke: model has fewer than two layers");
        return Ok(None);
    }
    let split = env_u32(MM_SPLIT_LAYER_ENV)?.unwrap_or(fixture.layer_end / 2);
    if split == 0 || split >= fixture.layer_end {
        bail!(
            "{MM_SPLIT_LAYER_ENV} must be in 1..{} for this model",
            fixture.layer_end
        );
    }
    Ok(Some(split))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn real_multimodal_local_smoke_when_fixture_is_set() -> Result<()> {
    let Some(fixture) = multimodal_smoke_fixture()? else {
        return Ok(());
    };
    let config = multimodal_stage_config(
        &fixture,
        "stage-0",
        0,
        0,
        fixture.layer_end,
        available_loopback_addr()?,
    );
    let backend = local_openai_backend(config)?;
    let response = backend
        .chat_completion(multimodal_chat_request(&fixture)?)
        .await?;

    assert_nonempty_chat_response(response);
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn real_multimodal_split_smoke_when_fixture_is_set() -> Result<()> {
    let Some(fixture) = multimodal_smoke_fixture()? else {
        return Ok(());
    };
    let Some(split_layer) = split_layer_for_fixture(&fixture)? else {
        return Ok(());
    };
    let stage1_addr = available_loopback_addr()?;
    let stage0_addr = available_loopback_addr()?;
    let mut stage1_config = multimodal_stage_config(
        &fixture,
        "stage-1",
        1,
        split_layer,
        fixture.layer_end,
        stage1_addr,
    );
    stage1_config.upstream = Some(skippy_protocol::PeerConfig {
        stage_id: "stage-0".to_string(),
        stage_index: 0,
        endpoint: stage0_addr.to_string(),
    });
    let mut stage0_config =
        multimodal_stage_config(&fixture, "stage-0", 0, 0, split_layer, stage0_addr);
    stage0_config.downstream = Some(skippy_protocol::PeerConfig {
        stage_id: "stage-1".to_string(),
        stage_index: 1,
        endpoint: stage1_addr.to_string(),
    });

    let stage1_handle =
        crate::embedded::start_binary_stage(crate::binary_transport::BinaryStageOptions {
            config: stage1_config,
            topology: None,
            bind_addr: stage1_addr,
            activation_width: fixture.activation_width,
            wire_dtype: WireActivationDType::F16,
            metrics_otlp_grpc: None,
            telemetry_queue_capacity: 1,
            telemetry_level: crate::telemetry::TelemetryLevel::Off,
            max_inflight: 4,
            reply_credit_limit: None,
            async_prefill_forward: false,
            downstream_wire_condition: WireCondition::new(0.0, None)?,
            downstream_connect_timeout_secs: 5,
            openai: None,
        });
    let ready = connect_endpoint_ready(&stage1_addr.to_string(), 10);
    if let Err(error) = ready {
        stage1_handle.abort();
        return Err(error.context("wait for stage-1 binary server"));
    }

    let telemetry = Telemetry::new(
        None,
        1,
        stage0_config.clone(),
        crate::telemetry::TelemetryLevel::Off,
    );
    let lane_pool = PersistentStageLanePool::new(&stage0_config, 1, 5, telemetry.clone())?
        .context("create split smoke lane pool")?;
    let runtime = load_runtime(&stage0_config)?.context("load stage-0 smoke runtime")?;
    let ctx_size = usize::try_from(stage0_config.ctx_size).unwrap_or(usize::MAX);
    let backend = StageOpenAiBackend {
        runtime,
        telemetry,
        config: stage0_config.clone(),
        model_id: "mm-smoke".to_string(),
        default_max_tokens: 16,
        ctx_size,
        mode: OpenAiBackendMode::EmbeddedStageZero {
            config: stage0_config,
            wire_dtype: WireActivationDType::F16,
            prefill_chunk_policy: PrefillChunkPolicy::Fixed { chunk_size: 64 },
            activation_width: fixture.activation_width,
            downstream_wire_condition: WireCondition::new(0.0, None)?,
            lane_pool: Some(lane_pool),
        },
        draft: None,
        speculative_window: 0,
        adaptive_speculative_window: false,
        generation_limit: Arc::new(Semaphore::new(1)),
        hook_policy: None,
    };
    let response = backend
        .chat_completion(multimodal_chat_request(&fixture)?)
        .await;
    stage1_handle.shutdown().await?;

    assert_nonempty_chat_response(response?);
    Ok(())
}

#[test]
fn trims_at_first_stop_sequence() {
    assert_eq!(trim_at_stop("hello END world", &["END"]), "hello ");
    assert_eq!(trim_at_stop("abc xyz def", &["def", "xyz"]), "abc ");
    assert_eq!(trim_at_stop("abc", &[""]), "abc");
}

#[test]
fn valid_utf8_prefix_skips_incomplete_suffix() {
    assert_eq!(valid_utf8_prefix_len("hello".as_bytes()), 5);
    assert_eq!(valid_utf8_prefix_len(&[b'h', b'i', 0xE2, 0x82]), 2);
    assert_eq!(valid_utf8_prefix_len(&[0xF0, 0x9F, 0x98]), 0);
}

#[test]
fn message_content_to_generation_text_inserts_media_markers() {
    let content: MessageContent = serde_json::from_value(json!([
        {"type": "text", "text": "what is this?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8="}}
    ]))
    .unwrap();
    let mut media = Vec::new();

    let text = message_content_to_generation_text(&content, "<__media__>", &mut media)
        .expect("media text");

    assert_eq!(text, "what is this?\n<__media__>");
    assert_eq!(media.len(), 1);
    assert_eq!(media[0].bytes, b"hello");
}

#[test]
fn message_content_to_generation_text_rejects_remote_media_urls() {
    let content: MessageContent = serde_json::from_value(json!([
        {"type": "input_image", "image_url": "https://example.com/image.png"}
    ]))
    .unwrap();
    let mut media = Vec::new();

    let error =
        message_content_to_generation_text(&content, "<__media__>", &mut media).unwrap_err();

    assert_eq!(
        error.body().error.code.as_deref(),
        Some("unsupported_model_feature")
    );
}

#[test]
fn multimodal_final_prefill_message_requests_downstream_prediction() {
    let sampling = WireSamplingConfig {
        flags: 1,
        seed: 7,
        ..WireSamplingConfig::default()
    };

    let message = multimodal_final_prefill_message(
        WireActivationDType::F16,
        MultimodalFinalPrefillArgs {
            request_id: 11,
            session_id: 13,
            prompt_token_count: 17,
            sampling: Some(sampling.clone()),
        },
    )
    .unwrap();

    assert_eq!(message.kind, WireMessageKind::PrefillFinalEmbd);
    assert!(message.kind.requires_predicted_reply());
    assert_eq!(message.token_count, 17);
    assert_eq!(message.state.current_token, LLAMA_TOKEN_NULL);
    assert_eq!(message.sampling, Some(sampling));
}

#[test]
fn hook_injected_text_concatenates_injection_actions() {
    let outcome = ChatHookOutcome {
        actions: vec![
            ChatHookAction::InjectText {
                text: "[first]\n".to_string(),
            },
            ChatHookAction::None,
            ChatHookAction::InjectText {
                text: "[second]\n".to_string(),
            },
        ],
    };

    assert_eq!(
        hook_injected_text(&outcome),
        Some("[first]\n[second]\n".to_string())
    );
}

#[test]
fn mid_generation_window_requires_minimum_tokens_and_cooldown() {
    let window = GenerationSignalWindow {
        token_count: 16,
        mean_entropy: 4.5,
        max_entropy: 5.0,
        mean_margin: 0.02,
        min_margin: 0.01,
        high_entropy_count: 12,
        repetition_count: 0,
    };

    assert!(!mid_generation_window_should_fire(11, &None, &window));
    assert!(!mid_generation_window_should_fire(20, &Some(0), &window));
    assert!(mid_generation_window_should_fire(33, &Some(0), &window));
}

#[test]
fn mid_generation_window_fires_on_repetition_even_with_low_entropy() {
    let window = GenerationSignalWindow {
        token_count: 16,
        mean_entropy: 0.3,
        max_entropy: 0.7,
        mean_margin: 0.7,
        min_margin: 0.4,
        high_entropy_count: 0,
        repetition_count: 3,
    };

    assert!(mid_generation_window_should_fire(16, &None, &window));
}

#[test]
fn default_sampling_controls_are_allowed() {
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "jc-builds/SmolLM2-135M-Instruct-Q4_K_M-GGUF:Q4_K_M",
        "messages": [{"role": "user", "content": "hello"}],
        "temperature": 1.0,
        "top_p": 1.0
    }))
    .unwrap();

    let sampling = chat_sampling_config(&request).unwrap();
    assert!(!sampling.enabled);
}

#[test]
fn non_default_sampling_controls_are_enabled() {
    let request: CompletionRequest = serde_json::from_value(json!({
        "model": "jc-builds/SmolLM2-135M-Instruct-Q4_K_M-GGUF:Q4_K_M",
        "prompt": "hello",
        "temperature": 0.7,
        "top_p": 0.9,
        "seed": 42
    }))
    .unwrap();

    let sampling = completion_sampling_config(&request).unwrap();
    assert!(sampling.enabled);
    assert_eq!(sampling.seed, 42);
    assert_eq!(sampling.temperature, 0.7);
}

#[test]
fn typed_sampling_penalties_are_enabled() {
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "jc-builds/SmolLM2-135M-Instruct-Q4_K_M-GGUF:Q4_K_M",
        "messages": [{"role": "user", "content": "hello"}],
        "presence_penalty": 1.0
    }))
    .unwrap();

    let sampling = chat_sampling_config(&request).unwrap();
    assert!(sampling.enabled);
    assert_eq!(sampling.presence_penalty, 1.0);
}

#[test]
fn extra_sampling_fields_are_enabled() {
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "jc-builds/SmolLM2-135M-Instruct-Q4_K_M-GGUF:Q4_K_M",
        "messages": [{"role": "user", "content": "hello"}],
        "top_k": 40
    }))
    .unwrap();

    let sampling = chat_sampling_config(&request).unwrap();
    assert!(sampling.enabled);
    assert_eq!(sampling.top_k, 40);
}

#[test]
fn canonical_reasoning_disabled_turns_off_chat_template_thinking() {
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "jc-builds/SmolLM2-135M-Instruct-Q4_K_M-GGUF:Q4_K_M",
        "messages": [{"role": "user", "content": "hello"}],
        "reasoning": {"enabled": false}
    }))
    .unwrap();

    let options = chat_template_options(&request).unwrap();
    assert_eq!(options.enable_thinking, Some(false));
}

#[test]
fn reasoning_effort_none_turns_off_chat_template_thinking() {
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "jc-builds/SmolLM2-135M-Instruct-Q4_K_M-GGUF:Q4_K_M",
        "messages": [{"role": "user", "content": "hello"}],
        "reasoning": {"effort": "none"}
    }))
    .unwrap();

    let options = chat_template_options(&request).unwrap();
    assert_eq!(options.enable_thinking, Some(false));
}

#[test]
fn provider_enable_thinking_overrides_canonical_reasoning() {
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "jc-builds/SmolLM2-135M-Instruct-Q4_K_M-GGUF:Q4_K_M",
        "messages": [{"role": "user", "content": "hello"}],
        "reasoning": {"enabled": false},
        "enable_thinking": true
    }))
    .unwrap();

    let options = chat_template_options(&request).unwrap();
    assert_eq!(options.enable_thinking, Some(true));
}

#[test]
fn chat_template_kwargs_enable_thinking_is_supported() {
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "jc-builds/SmolLM2-135M-Instruct-Q4_K_M-GGUF:Q4_K_M",
        "messages": [{"role": "user", "content": "hello"}],
        "chat_template_kwargs": {"enable_thinking": false}
    }))
    .unwrap();

    let options = chat_template_options(&request).unwrap();
    assert_eq!(options.enable_thinking, Some(false));
}

#[test]
fn thinking_boolean_aliases_are_supported() {
    for field in openai_frontend::THINKING_BOOLEAN_ALIASES {
        let request: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "jc-builds/SmolLM2-135M-Instruct-Q4_K_M-GGUF:Q4_K_M",
            "messages": [{"role": "user", "content": "hello"}],
            (*field): false
        }))
        .unwrap();
        assert_eq!(
            chat_template_options(&request).unwrap().enable_thinking,
            Some(false),
            "top-level alias {field}"
        );

        let request: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "jc-builds/SmolLM2-135M-Instruct-Q4_K_M-GGUF:Q4_K_M",
            "messages": [{"role": "user", "content": "hello"}],
            "chat_template_kwargs": {(*field): false}
        }))
        .unwrap();
        assert_eq!(
            chat_template_options(&request).unwrap().enable_thinking,
            Some(false),
            "chat_template_kwargs alias {field}"
        );
    }
}

#[test]
fn reasoning_max_tokens_enables_and_zero_budget_disables_thinking() {
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "jc-builds/SmolLM2-135M-Instruct-Q4_K_M-GGUF:Q4_K_M",
        "messages": [{"role": "user", "content": "hello"}],
        "reasoning": {"max_tokens": 1024}
    }))
    .unwrap();
    assert_eq!(
        chat_template_options(&request).unwrap().enable_thinking,
        Some(true)
    );

    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "jc-builds/SmolLM2-135M-Instruct-Q4_K_M-GGUF:Q4_K_M",
        "messages": [{"role": "user", "content": "hello"}],
        "reasoning": {"enabled": true},
        "thinking_budget": 0
    }))
    .unwrap();
    assert_eq!(
        chat_template_options(&request).unwrap().enable_thinking,
        Some(false)
    );
}

#[test]
fn logit_bias_is_enabled() {
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "jc-builds/SmolLM2-135M-Instruct-Q4_K_M-GGUF:Q4_K_M",
        "messages": [{"role": "user", "content": "hello"}],
        "logit_bias": {"123": -50.0, "456": 12.5}
    }))
    .unwrap();

    let sampling = chat_sampling_config(&request).unwrap();
    assert!(sampling.enabled);
    assert_eq!(sampling.logit_bias.len(), 2);
    assert_eq!(sampling.logit_bias[0].token_id, 123);
    assert_eq!(sampling.logit_bias[0].bias, -50.0);
    assert_eq!(sampling.logit_bias[1].token_id, 456);
    assert_eq!(sampling.logit_bias[1].bias, 12.5);
}

#[test]
fn invalid_logit_bias_returns_openai_error() {
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "jc-builds/SmolLM2-135M-Instruct-Q4_K_M-GGUF:Q4_K_M",
        "messages": [{"role": "user", "content": "hello"}],
        "logit_bias": {"not-a-token": 1.0}
    }))
    .unwrap();

    let error = chat_sampling_config(&request).unwrap_err();
    assert_eq!(error.body().error.code.as_deref(), Some("invalid_value"));
}

#[test]
fn unsupported_extra_generation_fields_return_openai_error() {
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "jc-builds/SmolLM2-135M-Instruct-Q4_K_M-GGUF:Q4_K_M",
        "messages": [{"role": "user", "content": "hello"}],
        "min_p": 0.1
    }))
    .unwrap();

    let error = chat_sampling_config(&request).unwrap_err();
    assert_eq!(
        error.body().error.code.as_deref(),
        Some("unsupported_model_feature")
    );
}

#[test]
fn maps_generation_exhaustion_to_length_finish_reason() {
    assert_eq!(finish_reason_for_generation(true), FinishReason::Length);
    assert_eq!(finish_reason_for_generation(false), FinishReason::Stop);
}

#[test]
fn generation_ids_are_unique_under_fast_creation() {
    let ids = (0..1024)
        .map(|_| OpenAiGenerationIds::new())
        .collect::<Vec<_>>();
    let mut sessions = std::collections::BTreeSet::new();
    let mut requests = std::collections::BTreeSet::new();
    for id in ids {
        assert!(sessions.insert(id.session_id));
        assert!(requests.insert(id.request_id));
    }
}

#[test]
fn prefill_chunk_schedule_parses_and_repeats_last_size() {
    let schedule = PrefillChunkSchedule::parse(Some("128, 256,512"))
        .unwrap()
        .unwrap();
    assert_eq!(schedule.label(), "128,256,512");
    assert_eq!(schedule.chunk_size_for(0), 128);
    assert_eq!(schedule.chunk_size_for(1), 256);
    assert_eq!(schedule.chunk_size_for(2), 512);
    assert_eq!(schedule.chunk_size_for(3), 512);
}

#[test]
fn prefill_chunk_schedule_rejects_bad_sizes() {
    assert!(PrefillChunkSchedule::parse(Some("128,0")).is_err());
    assert!(PrefillChunkSchedule::parse(Some("128,,256")).is_err());
    assert!(PrefillChunkSchedule::parse(Some("abc")).is_err());
}

#[test]
fn prefill_chunk_policy_keeps_legacy_schedule_behavior() {
    let policy = PrefillChunkPolicy::parse(PrefillChunkPolicyArgs {
        policy: "fixed",
        schedule: Some("128,256,384"),
        fixed_chunk_size: 256,
        adaptive_start: 128,
        adaptive_step: 128,
        adaptive_max: 384,
        schedule_arg: "--prefill-chunk-schedule",
        policy_arg: "--prefill-chunk-policy",
    })
    .unwrap();
    let mut planner = policy.planner();
    assert_eq!(planner.chunk_size_for(0), 128);
    assert_eq!(planner.chunk_size_for(1), 256);
    assert_eq!(planner.chunk_size_for(2), 384);
    assert_eq!(planner.chunk_size_for(3), 384);
}

#[test]
fn prefill_adaptive_ramp_grows_when_downstream_wait_is_hidden() {
    let policy = PrefillChunkPolicy::parse(PrefillChunkPolicyArgs {
        policy: "adaptive-ramp",
        schedule: None,
        fixed_chunk_size: 256,
        adaptive_start: 128,
        adaptive_step: 128,
        adaptive_max: 384,
        schedule_arg: "--prefill-chunk-schedule",
        policy_arg: "--prefill-chunk-policy",
    })
    .unwrap();
    let mut planner = policy.planner();
    assert_eq!(planner.chunk_size_for(0), 128);
    planner.observe(PrefillChunkObservation {
        compute_ms: 100.0,
        forward_write_ms: 5.0,
        downstream_wait_ms: 20.0,
    });
    assert_eq!(planner.chunk_size_for(1), 256);
    planner.observe(PrefillChunkObservation {
        compute_ms: 100.0,
        forward_write_ms: 5.0,
        downstream_wait_ms: 20.0,
    });
    assert_eq!(planner.chunk_size_for(2), 384);
}

#[test]
fn prefill_adaptive_ramp_can_advance_without_observations() {
    let policy = PrefillChunkPolicy::parse(PrefillChunkPolicyArgs {
        policy: "adaptive-ramp",
        schedule: None,
        fixed_chunk_size: 256,
        adaptive_start: 128,
        adaptive_step: 128,
        adaptive_max: 384,
        schedule_arg: "--prefill-chunk-schedule",
        policy_arg: "--prefill-chunk-policy",
    })
    .unwrap();
    let mut planner = policy.planner();
    assert_eq!(planner.chunk_size_for(0), 128);
    planner.advance_without_observation();
    assert_eq!(planner.chunk_size_for(1), 256);
    planner.advance_without_observation();
    assert_eq!(planner.chunk_size_for(2), 384);
    planner.advance_without_observation();
    assert_eq!(planner.chunk_size_for(3), 384);
}

#[test]
fn prefill_adaptive_ramp_backs_off_when_wait_is_exposed() {
    let policy = PrefillChunkPolicy::parse(PrefillChunkPolicyArgs {
        policy: "adaptive-ramp",
        schedule: None,
        fixed_chunk_size: 256,
        adaptive_start: 128,
        adaptive_step: 128,
        adaptive_max: 384,
        schedule_arg: "--prefill-chunk-schedule",
        policy_arg: "--prefill-chunk-policy",
    })
    .unwrap();
    let mut planner = policy.planner();
    planner.observe(PrefillChunkObservation {
        compute_ms: 100.0,
        forward_write_ms: 5.0,
        downstream_wait_ms: 10.0,
    });
    assert_eq!(planner.chunk_size_for(1), 256);
    planner.observe(PrefillChunkObservation {
        compute_ms: 100.0,
        forward_write_ms: 5.0,
        downstream_wait_ms: 150.0,
    });
    assert_eq!(planner.chunk_size_for(2), 128);
}

#[test]
fn model_matching_is_exact_for_mesh_style_ids() {
    ensure_requested_model(
        "jc-builds/SmolLM2-135M-Instruct-Q4_K_M-GGUF:Q4_K_M",
        "jc-builds/SmolLM2-135M-Instruct-Q4_K_M-GGUF:Q4_K_M",
    )
    .unwrap();

    let error = ensure_requested_model(
        "jc-builds/SmolLM2-135M-Instruct-Q4_K_M-GGUF:Q4_K_M",
        "org/repo:Q5_K_M",
    )
    .unwrap_err();
    assert_eq!(error.body().error.code.as_deref(), Some("model_not_found"));
}

#[test]
fn rejects_requests_that_exceed_context_window() {
    ensure_context_capacity(4, 4, 8).unwrap();

    let error = ensure_context_capacity(5, 4, 8).unwrap_err();
    assert_eq!(
        error.body().error.code.as_deref(),
        Some("context_length_exceeded")
    );
}

#[test]
fn omitted_max_tokens_can_use_remaining_context_budget() {
    let limit = GenerationTokenLimit::from_request(None, CONTEXT_BUDGET_MAX_TOKENS);
    assert_eq!(limit.resolve(5, 8).unwrap(), 3);
}

#[test]
fn configured_default_max_tokens_remains_explicit() {
    let limit = GenerationTokenLimit::from_request(None, 4);
    assert_eq!(limit.resolve(4, 8).unwrap(), 4);

    let error = limit.resolve(5, 8).unwrap_err();
    assert_eq!(
        error.body().error.code.as_deref(),
        Some("context_length_exceeded")
    );
}
