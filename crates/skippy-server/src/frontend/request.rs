use super::*;

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
    const UNSUPPORTED_FIELDS: &[&str] = &["typical_p"];

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
        penalty_last_n: -1,
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
