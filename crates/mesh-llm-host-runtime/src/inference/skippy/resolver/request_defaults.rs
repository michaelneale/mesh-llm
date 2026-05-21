use anyhow::{bail, Result};
use openai_frontend::ReasoningEffort;
use skippy_server::{
    EmbeddedReasoningBudget, EmbeddedReasoningEnabled, EmbeddedReasoningFormat,
    CONTEXT_BUDGET_MAX_TOKENS,
};

use super::support::string_list_value;
use super::types::ResolvedRequestDefaultsConfig;
use crate::plugin::{
    ModelConfigDefaults, ModelConfigEntry, ReasoningBudget, ReasoningEnabled, RequestDefaultsConfig,
};

pub(super) fn resolve_request_defaults(
    defaults: Option<&ModelConfigDefaults>,
    model_entry: Option<&ModelConfigEntry>,
    request_defaults: Option<&RequestDefaultsConfig>,
) -> Result<ResolvedRequestDefaultsConfig> {
    let model = model_entry.and_then(|entry| entry.request_defaults.as_ref());
    let global = defaults.and_then(|value| value.request_defaults.as_ref());

    reject_unsupported_request_defaults(request_defaults, "request_defaults")?;
    reject_unsupported_request_defaults(model, "models[].request_defaults")?;
    reject_unsupported_request_defaults(global, "defaults.request_defaults")?;

    Ok(ResolvedRequestDefaultsConfig {
        max_tokens: request_defaults
            .and_then(|value| value.max_tokens)
            .or_else(|| model.and_then(|value| value.max_tokens))
            .or_else(|| global.and_then(|value| value.max_tokens))
            .unwrap_or(CONTEXT_BUDGET_MAX_TOKENS),
        temperature: request_defaults
            .and_then(|value| value.temperature)
            .or_else(|| model.and_then(|value| value.temperature))
            .or_else(|| global.and_then(|value| value.temperature)),
        top_p: request_defaults
            .and_then(|value| value.top_p)
            .or_else(|| model.and_then(|value| value.top_p))
            .or_else(|| global.and_then(|value| value.top_p)),
        presence_penalty: request_defaults
            .and_then(|value| value.presence_penalty)
            .or_else(|| model.and_then(|value| value.presence_penalty))
            .or_else(|| global.and_then(|value| value.presence_penalty)),
        frequency_penalty: request_defaults
            .and_then(|value| value.frequency_penalty)
            .or_else(|| model.and_then(|value| value.frequency_penalty))
            .or_else(|| global.and_then(|value| value.frequency_penalty)),
        seed: request_defaults
            .and_then(|value| value.seed)
            .or_else(|| model.and_then(|value| value.seed))
            .or_else(|| global.and_then(|value| value.seed)),
        logit_bias: request_defaults
            .and_then(|value| value.logit_bias.clone())
            .or_else(|| model.and_then(|value| value.logit_bias.clone()))
            .or_else(|| global.and_then(|value| value.logit_bias.clone())),
        top_k: request_defaults
            .and_then(|value| value.top_k)
            .or_else(|| model.and_then(|value| value.top_k))
            .or_else(|| global.and_then(|value| value.top_k)),
        min_p: request_defaults
            .and_then(|value| value.min_p)
            .or_else(|| model.and_then(|value| value.min_p))
            .or_else(|| global.and_then(|value| value.min_p)),
        repeat_penalty: request_defaults
            .and_then(|value| value.repeat_penalty)
            .or_else(|| model.and_then(|value| value.repeat_penalty))
            .or_else(|| global.and_then(|value| value.repeat_penalty)),
        repeat_last_n: request_defaults
            .and_then(|value| value.repeat_last_n)
            .or_else(|| model.and_then(|value| value.repeat_last_n))
            .or_else(|| global.and_then(|value| value.repeat_last_n)),
        stop: request_defaults
            .and_then(|value| value.stop.as_ref())
            .or_else(|| model.and_then(|value| value.stop.as_ref()))
            .or_else(|| global.and_then(|value| value.stop.as_ref()))
            .map(string_list_value),
        reasoning_format: request_defaults
            .and_then(|value| value.reasoning_format.clone())
            .or_else(|| model.and_then(|value| value.reasoning_format.clone()))
            .or_else(|| global.and_then(|value| value.reasoning_format.clone())),
        reasoning_enabled: request_defaults
            .and_then(|value| value.reasoning_enabled.clone())
            .or_else(|| model.and_then(|value| value.reasoning_enabled.clone()))
            .or_else(|| global.and_then(|value| value.reasoning_enabled.clone())),
        reasoning_budget: request_defaults
            .and_then(|value| value.reasoning_budget.clone())
            .or_else(|| model.and_then(|value| value.reasoning_budget.clone()))
            .or_else(|| global.and_then(|value| value.reasoning_budget.clone())),
    })
}

pub(super) fn resolve_reasoning_format(value: &str) -> Option<EmbeddedReasoningFormat> {
    match value {
        "auto" => Some(EmbeddedReasoningFormat::Auto),
        "none" => Some(EmbeddedReasoningFormat::None),
        "deepseek" => Some(EmbeddedReasoningFormat::Deepseek),
        "deepseek-legacy" => Some(EmbeddedReasoningFormat::DeepseekLegacy),
        "hidden" => Some(EmbeddedReasoningFormat::Hidden),
        _ => None,
    }
}

pub(super) fn resolve_reasoning_budget(value: &ReasoningBudget) -> Option<EmbeddedReasoningBudget> {
    match value {
        ReasoningBudget::Integer(tokens) => Some(EmbeddedReasoningBudget::Tokens(*tokens)),
        ReasoningBudget::String(value) => match value.as_str() {
            "auto" => Some(EmbeddedReasoningBudget::Auto),
            "low" => Some(EmbeddedReasoningBudget::Effort(ReasoningEffort::Low)),
            "medium" => Some(EmbeddedReasoningBudget::Effort(ReasoningEffort::Medium)),
            "high" => Some(EmbeddedReasoningBudget::Effort(ReasoningEffort::High)),
            _ => None,
        },
    }
}

pub(super) fn resolve_reasoning_enabled(
    value: &ReasoningEnabled,
) -> Option<EmbeddedReasoningEnabled> {
    match value {
        ReasoningEnabled::Bool(true) => Some(EmbeddedReasoningEnabled::Enabled),
        ReasoningEnabled::Bool(false) => Some(EmbeddedReasoningEnabled::Disabled),
        ReasoningEnabled::String(value) => match value.as_str() {
            "auto" => Some(EmbeddedReasoningEnabled::Auto),
            "off" => Some(EmbeddedReasoningEnabled::Disabled),
            "on" => Some(EmbeddedReasoningEnabled::Enabled),
            _ => None,
        },
    }
}

pub(super) fn resolve_request_seed(value: i64) -> Result<u64> {
    u64::try_from(value)
        .map_err(|_| anyhow::anyhow!("request_defaults.seed must be greater than or equal to 0"))
}

pub(super) fn resolve_request_top_k(value: i64) -> Result<i32> {
    i32::try_from(value)
        .map_err(|_| anyhow::anyhow!("request_defaults.top_k exceeds supported i32 range"))
}

pub(super) fn resolve_request_repeat_last_n(value: i64) -> Result<i32> {
    i32::try_from(value)
        .map_err(|_| anyhow::anyhow!("request_defaults.repeat_last_n exceeds supported i32 range"))
}

pub(super) fn resolve_request_logit_bias(
    value: &toml::Value,
) -> Result<std::collections::BTreeMap<String, serde_json::Value>> {
    let json = serde_json::to_value(value).map_err(|error| {
        anyhow::anyhow!("request_defaults.logit_bias could not be converted to JSON: {error}")
    })?;
    serde_json::from_value::<std::collections::BTreeMap<String, serde_json::Value>>(json).map_err(
        |_| anyhow::anyhow!("request_defaults.logit_bias must be an object keyed by token id"),
    )
}

fn reject_unsupported_request_defaults(
    config: Option<&RequestDefaultsConfig>,
    base_path: &str,
) -> Result<()> {
    let Some(config) = config else {
        return Ok(());
    };

    for (field, present) in [
        ("typical_p", config.typical_p.is_some()),
        ("top_nsigma", config.top_nsigma.is_some()),
        ("dynatemp_range", config.dynatemp_range.is_some()),
        ("dynatemp_exponent", config.dynatemp_exponent.is_some()),
        ("dry", config.dry.is_some()),
        ("xtc", config.xtc.is_some()),
        ("adaptive", config.adaptive.is_some()),
        ("mirostat_mode", config.mirostat_mode.is_some()),
        ("mirostat_entropy", config.mirostat_entropy.is_some()),
        (
            "mirostat_learning_rate",
            config.mirostat_learning_rate.is_some(),
        ),
        ("samplers", config.samplers.is_some()),
        ("sampler_sequence", config.sampler_sequence.is_some()),
        ("ignore_eos", config.ignore_eos.is_some()),
        ("backend_sampling", config.backend_sampling.is_some()),
        ("chat_template", config.chat_template.is_some()),
        ("chat_template_file", config.chat_template_file.is_some()),
        ("jinja", config.jinja.is_some()),
        (
            "chat_template_kwargs",
            config.chat_template_kwargs.is_some(),
        ),
        ("skip_chat_parsing", config.skip_chat_parsing.is_some()),
        ("prefill_assistant", config.prefill_assistant.is_some()),
        ("system_prompt", config.system_prompt.is_some()),
        ("grammar", config.grammar.is_some()),
        ("json_schema", config.json_schema.is_some()),
        ("logprobs", config.logprobs.is_some()),
    ] {
        if present {
            bail!(
                "{base_path}.{field} is accepted by config schema but not supported by the skippy OpenAI frontend/runtime"
            );
        }
    }

    Ok(())
}
