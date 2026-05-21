use super::*;

struct SharedRequestFields<'a> {
    presence_penalty: &'a mut Option<f32>,
    frequency_penalty: &'a mut Option<f32>,
    seed: &'a mut Option<u64>,
    logit_bias: &'a mut Option<std::collections::BTreeMap<String, serde_json::Value>>,
    temperature: &'a mut Option<f32>,
    top_p: &'a mut Option<f32>,
    stop: &'a mut Option<openai_frontend::StopSequence>,
    reasoning: &'a mut Option<openai_frontend::ReasoningConfig>,
    reasoning_effort: &'a mut Option<openai_frontend::ReasoningEffort>,
    extra: &'a mut std::collections::BTreeMap<String, serde_json::Value>,
}

pub(super) fn apply_chat_request_defaults(
    request: &mut ChatCompletionRequest,
    defaults: &EmbeddedOpenAiRequestDefaults,
) {
    apply_shared_request_defaults(
        SharedRequestFields {
            presence_penalty: &mut request.presence_penalty,
            frequency_penalty: &mut request.frequency_penalty,
            seed: &mut request.seed,
            logit_bias: &mut request.logit_bias,
            temperature: &mut request.temperature,
            top_p: &mut request.top_p,
            stop: &mut request.stop,
            reasoning: &mut request.reasoning,
            reasoning_effort: &mut request.reasoning_effort,
            extra: &mut request.extra,
        },
        defaults,
    );
}

pub(super) fn apply_completion_request_defaults(
    request: &mut CompletionRequest,
    defaults: &EmbeddedOpenAiRequestDefaults,
) {
    apply_shared_request_defaults(
        SharedRequestFields {
            presence_penalty: &mut request.presence_penalty,
            frequency_penalty: &mut request.frequency_penalty,
            seed: &mut request.seed,
            logit_bias: &mut request.logit_bias,
            temperature: &mut request.temperature,
            top_p: &mut request.top_p,
            stop: &mut request.stop,
            reasoning: &mut request.reasoning,
            reasoning_effort: &mut request.reasoning_effort,
            extra: &mut request.extra,
        },
        defaults,
    );
}

pub(super) fn message_content_to_generation_text(
    content: &MessageContent,
    marker: &str,
    media: &mut Vec<MediaInput>,
) -> OpenAiResult<String> {
    match content {
        MessageContent::Text(text) => Ok(text.clone()),
        MessageContent::Parts(parts) => {
            let mut chunks = Vec::new();
            for part in parts {
                if part.content_type == "text" {
                    if let Some(text) = part.text.as_deref() {
                        chunks.push(text.to_string());
                    }
                    continue;
                }
                if let Some(bytes) = media_bytes_from_part(part)? {
                    media.push(MediaInput { bytes });
                    chunks.push(marker.to_string());
                }
            }
            Ok(chunks.join("\n"))
        }
        MessageContent::Other(_) => Ok(String::new()),
    }
}

pub(super) fn media_bytes_from_part(part: &MessageContentPart) -> OpenAiResult<Option<Vec<u8>>> {
    let is_media = matches!(
        part.content_type.as_str(),
        "image_url" | "input_image" | "image" | "input_audio" | "audio" | "audio_url"
    );
    if !is_media {
        return Ok(None);
    }
    if let Some(url) = media_url(part) {
        return decode_media_url(&url).map(Some);
    }
    if let Some(data) = media_data(part) {
        return decode_base64_payload(&data).map(Some);
    }
    Err(OpenAiError::invalid_request(format!(
        "media content block '{}' is missing url or data",
        part.content_type
    )))
}

pub(super) fn media_url(part: &MessageContentPart) -> Option<String> {
    for key in [
        "image_url",
        "input_image",
        "image",
        "input_audio",
        "audio",
        "audio_url",
        "url",
    ] {
        if let Some(value) = part.extra.get(key) {
            if let Some(url) = value.as_str() {
                return Some(url.to_string());
            }
            if let Some(url) = value.get("url").and_then(Value::as_str) {
                return Some(url.to_string());
            }
        }
    }
    None
}

pub(super) fn media_data(part: &MessageContentPart) -> Option<String> {
    for key in ["input_audio", "audio", "image", "input_image", "image_url"] {
        if let Some(value) = part.extra.get(key) {
            if let Some(data) = value.get("data").and_then(Value::as_str) {
                return Some(data.to_string());
            }
        }
    }
    None
}

pub(super) fn decode_media_url(url: &str) -> OpenAiResult<Vec<u8>> {
    if let Some((prefix, payload)) = url.split_once(',') {
        if prefix.starts_with("data:") && prefix.contains(";base64") {
            return decode_base64_payload(payload);
        }
    }
    if url.starts_with("http://") || url.starts_with("https://") {
        return Err(OpenAiError::unsupported(
            "remote multimodal URLs must be fetched by mesh before reaching skippy",
        ));
    }
    decode_base64_payload(url)
}

pub(super) fn decode_base64_payload(payload: &str) -> OpenAiResult<Vec<u8>> {
    base64::engine::general_purpose::STANDARD
        .decode(payload.as_bytes())
        .or_else(|_| base64::engine::general_purpose::URL_SAFE_NO_PAD.decode(payload.as_bytes()))
        .map_err(|error| OpenAiError::invalid_request(format!("invalid media base64: {error}")))
}

fn apply_shared_request_defaults(
    fields: SharedRequestFields<'_>,
    defaults: &EmbeddedOpenAiRequestDefaults,
) {
    let SharedRequestFields {
        presence_penalty,
        frequency_penalty,
        seed,
        logit_bias,
        temperature,
        top_p,
        stop,
        reasoning,
        reasoning_effort,
        extra,
    } = fields;
    if presence_penalty.is_none() {
        *presence_penalty = defaults.presence_penalty;
    }
    if frequency_penalty.is_none() {
        *frequency_penalty = defaults.frequency_penalty;
    }
    if seed.is_none() {
        *seed = defaults.seed;
    }
    if logit_bias.is_none() {
        *logit_bias = defaults.logit_bias.clone();
    }
    if temperature.is_none() {
        *temperature = defaults.temperature;
    }
    if top_p.is_none() {
        *top_p = defaults.top_p;
    }
    if stop.is_none() {
        *stop = defaults
            .stop
            .as_ref()
            .map(|values| stop_sequence_from_defaults(values.clone()));
    }
    if extra_value_is_omitted(extra, "top_k") {
        if let Some(value) = defaults.top_k {
            extra.insert("top_k".to_string(), serde_json::json!(value));
        }
    }
    if extra_value_is_omitted(extra, "min_p") {
        if let Some(value) = defaults.min_p {
            extra.insert("min_p".to_string(), serde_json::json!(value));
        }
    }
    if extra_value_is_omitted(extra, "repeat_penalty")
        && extra_value_is_omitted(extra, "repetition_penalty")
    {
        if let Some(value) = defaults.repeat_penalty {
            extra.insert("repeat_penalty".to_string(), serde_json::json!(value));
        }
    }
    if extra_value_is_omitted(extra, "repeat_last_n") {
        if let Some(value) = defaults.repeat_last_n {
            extra.insert("repeat_last_n".to_string(), serde_json::json!(value));
        }
    }
    apply_reasoning_defaults(reasoning, reasoning_effort, extra, defaults);
}

fn apply_reasoning_defaults(
    reasoning: &mut Option<openai_frontend::ReasoningConfig>,
    reasoning_effort: &mut Option<openai_frontend::ReasoningEffort>,
    extra: &mut std::collections::BTreeMap<String, serde_json::Value>,
    defaults: &EmbeddedOpenAiRequestDefaults,
) {
    if explicit_reasoning_toggle_present(reasoning.as_ref(), *reasoning_effort, extra) {
        return;
    }

    match defaults.reasoning_enabled {
        Some(EmbeddedReasoningEnabled::Disabled) => {
            reasoning.get_or_insert_with(Default::default).enabled = Some(false);
            return;
        }
        Some(EmbeddedReasoningEnabled::Enabled) => {
            reasoning.get_or_insert_with(Default::default).enabled = Some(true);
        }
        Some(EmbeddedReasoningEnabled::Auto) | None => {}
    }

    match defaults.reasoning_format {
        Some(EmbeddedReasoningFormat::None) => {
            reasoning.get_or_insert_with(Default::default).enabled = Some(false);
            return;
        }
        Some(EmbeddedReasoningFormat::Deepseek)
        | Some(EmbeddedReasoningFormat::DeepseekLegacy)
        | Some(EmbeddedReasoningFormat::Hidden) => {
            reasoning.get_or_insert_with(Default::default).enabled = Some(true);
        }
        Some(EmbeddedReasoningFormat::Auto) | None => {}
    }

    if explicit_reasoning_budget_present(reasoning.as_ref(), *reasoning_effort, extra) {
        return;
    }

    match defaults.reasoning_budget {
        Some(EmbeddedReasoningBudget::Tokens(value)) => {
            reasoning.get_or_insert_with(Default::default).max_tokens = Some(value);
        }
        Some(EmbeddedReasoningBudget::Effort(value)) => {
            reasoning.get_or_insert_with(Default::default).effort = Some(value);
        }
        Some(EmbeddedReasoningBudget::Auto) | None => {}
    }
}

fn explicit_reasoning_toggle_present(
    reasoning: Option<&openai_frontend::ReasoningConfig>,
    reasoning_effort: Option<openai_frontend::ReasoningEffort>,
    extra: &std::collections::BTreeMap<String, serde_json::Value>,
) -> bool {
    reasoning.is_some_and(|value| value.enabled.is_some() || value.effort.is_some())
        || reasoning_effort.is_some()
        || openai_frontend::THINKING_BOOLEAN_ALIASES
            .iter()
            .any(|field| !extra_value_is_omitted(extra, field))
        || extra
            .get("chat_template_kwargs")
            .and_then(Value::as_object)
            .is_some_and(|object| {
                openai_frontend::THINKING_BOOLEAN_ALIASES
                    .iter()
                    .any(|field| object.get(*field).is_some_and(|value| !value.is_null()))
            })
}

fn explicit_reasoning_budget_present(
    reasoning: Option<&openai_frontend::ReasoningConfig>,
    reasoning_effort: Option<openai_frontend::ReasoningEffort>,
    extra: &std::collections::BTreeMap<String, serde_json::Value>,
) -> bool {
    reasoning.is_some_and(|value| value.max_tokens.is_some() || value.effort.is_some())
        || reasoning_effort.is_some()
        || !extra_value_is_omitted(extra, "thinking_budget")
}

fn extra_value_is_omitted(
    extra: &std::collections::BTreeMap<String, serde_json::Value>,
    field: &str,
) -> bool {
    extra.get(field).is_none_or(Value::is_null)
}

fn stop_sequence_from_defaults(values: Vec<String>) -> openai_frontend::StopSequence {
    if values.len() == 1 {
        openai_frontend::StopSequence::One(values.into_iter().next().unwrap_or_default())
    } else {
        openai_frontend::StopSequence::Many(values)
    }
}

pub(super) fn chat_sampling_config(
    request: &ChatCompletionRequest,
) -> OpenAiResult<SamplingConfig> {
    sampling_config(
        request.temperature,
        request.top_p,
        request.presence_penalty,
        request.frequency_penalty,
        request.seed,
        request.logit_bias.as_ref(),
        &request.extra,
    )
}

pub(super) fn completion_sampling_config(
    request: &CompletionRequest,
) -> OpenAiResult<SamplingConfig> {
    sampling_config(
        request.temperature,
        request.top_p,
        request.presence_penalty,
        request.frequency_penalty,
        request.seed,
        request.logit_bias.as_ref(),
        &request.extra,
    )
}

pub(super) fn chat_template_options(
    request: &ChatCompletionRequest,
) -> OpenAiResult<ChatTemplateOptions> {
    let reasoning_options = normalize_reasoning_template_options(
        request.reasoning.as_ref(),
        request.reasoning_effort,
        &request.extra,
    )?;
    Ok(ChatTemplateOptions {
        enable_thinking: reasoning_options.enable_thinking,
        ..ChatTemplateOptions::default()
    })
}

pub(super) fn ensure_chat_runtime_features_supported(
    request: &ChatCompletionRequest,
) -> OpenAiResult<()> {
    if request.logprobs.unwrap_or(false) || request.top_logprobs.is_some() {
        return Err(OpenAiError::unsupported(
            "chat logprobs are parsed by openai-frontend but not yet implemented by skippy runtime",
        ));
    }
    if request
        .response_format
        .as_ref()
        .is_some_and(requires_structured_output)
    {
        return Err(OpenAiError::unsupported(
            "structured output is parsed by openai-frontend but not yet implemented by skippy runtime",
        ));
    }
    Ok(())
}

pub(super) fn ensure_completion_runtime_features_supported(
    request: &CompletionRequest,
) -> OpenAiResult<()> {
    if request.logprobs.is_some() {
        return Err(OpenAiError::unsupported(
            "completion logprobs are parsed by openai-frontend but not yet implemented by skippy runtime",
        ));
    }
    Ok(())
}

pub(super) fn has_requested_tools(value: &Value) -> bool {
    !matches!(value, Value::Array(items) if items.is_empty())
}

pub(super) fn requires_structured_output(value: &Value) -> bool {
    value
        .as_object()
        .and_then(|object| object.get("type"))
        .and_then(Value::as_str)
        .is_some_and(|format_type| format_type != "text")
}

pub(super) fn ensure_extra_generation_fields_absent(
    extra: &std::collections::BTreeMap<String, serde_json::Value>,
) -> OpenAiResult<()> {
    const UNSUPPORTED_FIELDS: &[&str] = &[
        "typical_p",
        "top_nsigma",
        "dynatemp_range",
        "dynatemp_exponent",
        "dry",
        "xtc",
        "adaptive",
        "mirostat_mode",
        "mirostat_entropy",
        "mirostat_learning_rate",
        "samplers",
        "sampler_sequence",
        "ignore_eos",
    ];

    for field in UNSUPPORTED_FIELDS {
        if extra.get(*field).is_some_and(|value| !value.is_null()) {
            return Err(OpenAiError::unsupported(format!(
                "{field} is parsed but not yet implemented"
            )));
        }
    }
    Ok(())
}

pub(super) fn sampling_config(
    temperature: Option<f32>,
    top_p: Option<f32>,
    presence_penalty: Option<f32>,
    frequency_penalty: Option<f32>,
    seed: Option<u64>,
    logit_bias: Option<&std::collections::BTreeMap<String, serde_json::Value>>,
    extra: &std::collections::BTreeMap<String, serde_json::Value>,
) -> OpenAiResult<SamplingConfig> {
    ensure_extra_generation_fields_absent(extra)?;
    let temperature = temperature.unwrap_or(0.8);
    let top_p = top_p.unwrap_or(0.95);
    let presence_penalty = presence_penalty.unwrap_or(0.0);
    let frequency_penalty = frequency_penalty.unwrap_or(0.0);
    let top_k = optional_i32_extra(extra, "top_k")?.unwrap_or(40);
    let min_p = optional_f32_extra(extra, "min_p")?.unwrap_or(0.05);
    let repeat_penalty = optional_f32_extra(extra, "repeat_penalty")?
        .or(optional_f32_extra(extra, "repetition_penalty")?)
        .unwrap_or(1.0);
    let penalty_last_n = optional_i32_extra(extra, "repeat_last_n")?.unwrap_or(-1);
    validate_sampling_range("temperature", temperature, 0.0..=100.0)?;
    validate_sampling_range("top_p", top_p, 0.0..=1.0)?;
    validate_sampling_range("presence_penalty", presence_penalty, -2.0..=2.0)?;
    validate_sampling_range("frequency_penalty", frequency_penalty, -2.0..=2.0)?;
    validate_sampling_range("min_p", min_p, 0.0..=1.0)?;
    validate_sampling_range("repeat_penalty", repeat_penalty, 0.0..=100.0)?;
    if top_k < 0 {
        return Err(OpenAiError::invalid_request(
            "top_k must be greater than or equal to zero",
        ));
    }
    if penalty_last_n < -1 {
        return Err(OpenAiError::invalid_request(
            "repeat_last_n must be greater than or equal to -1",
        ));
    }
    let seed = match seed {
        Some(seed) => u32::try_from(seed)
            .map_err(|_| OpenAiError::invalid_request("seed exceeds u32 range"))?,
        None => 0,
    };
    let logit_bias = parse_logit_bias(logit_bias)?;
    let enabled = seed != 0
        || temperature <= 0.0
        || (temperature - 1.0).abs() > f32::EPSILON
        || (top_p - 1.0).abs() > f32::EPSILON
        || top_k > 0
        || min_p > 0.0
        || presence_penalty.abs() > f32::EPSILON
        || frequency_penalty.abs() > f32::EPSILON
        || (repeat_penalty - 1.0).abs() > f32::EPSILON
        || penalty_last_n != -1
        || !logit_bias.is_empty();
    Ok(SamplingConfig {
        enabled,
        seed,
        temperature,
        top_p,
        top_k,
        min_p,
        presence_penalty,
        frequency_penalty,
        repeat_penalty,
        penalty_last_n,
        logit_bias,
    })
}

pub(super) fn parse_logit_bias(
    logit_bias: Option<&std::collections::BTreeMap<String, serde_json::Value>>,
) -> OpenAiResult<Vec<RuntimeLogitBias>> {
    let Some(logit_bias) = logit_bias else {
        return Ok(Vec::new());
    };
    if logit_bias.len() > MAX_LOGIT_BIAS {
        return Err(OpenAiError::invalid_request(format!(
            "logit_bias supports at most {MAX_LOGIT_BIAS} entries"
        )));
    }
    let mut parsed = Vec::with_capacity(logit_bias.len());
    for (token_id, bias) in logit_bias {
        let token_id = token_id
            .parse::<i32>()
            .map_err(|_| OpenAiError::invalid_request("logit_bias token IDs must be integers"))?;
        if token_id < 0 {
            return Err(OpenAiError::invalid_request(
                "logit_bias token IDs must be greater than or equal to zero",
            ));
        }
        let bias = serde_json::from_value::<f32>(bias.clone())
            .map_err(|_| OpenAiError::invalid_request("logit_bias values must be numbers"))?;
        validate_sampling_range("logit_bias", bias, -100.0..=100.0)?;
        parsed.push(RuntimeLogitBias { token_id, bias });
    }
    Ok(parsed)
}

pub(super) fn validate_sampling_range(
    name: &str,
    value: f32,
    range: std::ops::RangeInclusive<f32>,
) -> OpenAiResult<()> {
    if !value.is_finite() || !range.contains(&value) {
        return Err(OpenAiError::invalid_request(format!(
            "{name} is outside the supported range"
        )));
    }
    Ok(())
}

pub(super) fn optional_f32_extra(
    extra: &std::collections::BTreeMap<String, serde_json::Value>,
    field: &str,
) -> OpenAiResult<Option<f32>> {
    extra
        .get(field)
        .filter(|value| !value.is_null())
        .map(|value| {
            serde_json::from_value::<f32>(value.clone())
                .map_err(|_| OpenAiError::invalid_request(format!("{field} must be a number")))
        })
        .transpose()
}

pub(super) fn optional_i32_extra(
    extra: &std::collections::BTreeMap<String, serde_json::Value>,
    field: &str,
) -> OpenAiResult<Option<i32>> {
    extra
        .get(field)
        .filter(|value| !value.is_null())
        .map(|value| {
            serde_json::from_value::<i32>(value.clone())
                .map_err(|_| OpenAiError::invalid_request(format!("{field} must be an integer")))
        })
        .transpose()
}

pub(super) fn wire_sampling_config(sampling: &SamplingConfig) -> Option<WireSamplingConfig> {
    if !sampling.enabled {
        return None;
    }
    let mut wire = WireSamplingConfig {
        flags: u32::from(sampling.enabled),
        seed: sampling.seed,
        temperature: sampling.temperature,
        top_p: sampling.top_p,
        top_k: sampling.top_k,
        min_p: sampling.min_p,
        presence_penalty: sampling.presence_penalty,
        frequency_penalty: sampling.frequency_penalty,
        repeat_penalty: sampling.repeat_penalty,
        penalty_last_n: sampling.penalty_last_n,
        ..WireSamplingConfig::default()
    };
    wire.logit_bias = sampling
        .logit_bias
        .iter()
        .take(MAX_STAGE_LOGIT_BIAS)
        .map(|source| WireLogitBias {
            token_id: source.token_id,
            bias: source.bias,
        })
        .collect();
    Some(wire)
}
