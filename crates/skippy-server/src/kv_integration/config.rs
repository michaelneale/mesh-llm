use std::{
    collections::BTreeSet,
    sync::{Arc, Mutex},
};

use anyhow::Result;
use skippy_cache::{
    ExactStateCache, PrefixCandidatePolicy, ResidentActivationCache, ResidentCacheConfig,
    ResidentPrefixCache,
};
use skippy_protocol::{StageConfig, StageKvCacheConfig, StageKvCacheMode, StageKvCachePayload};

use super::{ExactStateExtra, KvStageIntegration, StageKvMode, StagePrefixCachePayload};

impl KvStageIntegration {
    pub fn from_config(config: &StageConfig) -> Result<Option<Self>> {
        let Some(cache_config) = effective_cache_config(config) else {
            return Ok(None);
        };
        let mode = match cache_config.mode {
            StageKvCacheMode::Disabled | StageKvCacheMode::Auto => StageKvMode::Disabled,
            StageKvCacheMode::Record => StageKvMode::Record,
            StageKvCacheMode::LookupRecord => StageKvMode::LookupRecord,
        };
        if mode == StageKvMode::Disabled {
            return Ok(None);
        }
        let payload = effective_cache_payload(config, cache_config.payload);
        if payload == StagePrefixCachePayload::Disabled {
            return Ok(None);
        }
        let candidate_policy = PrefixCandidatePolicy::from_cache(&cache_config);
        let resident_config = ResidentCacheConfig::from_stage(config, &cache_config);
        Ok(Some(Self {
            mode,
            payload,
            correctness_mode: false,
            trust_local_writes: true,
            candidate_policy,
            inflight_records: Arc::new(Mutex::new(BTreeSet::new())),
            resident: Arc::new(Mutex::new(ResidentPrefixCache::new(resident_config))),
            activations: Arc::new(Mutex::new(ResidentActivationCache::new(resident_config))),
            exact_states: Arc::new(Mutex::new(ExactStateCache::<ExactStateExtra>::new(
                cache_config.max_entries.max(1).min(512),
                cache_config.max_bytes,
            ))),
        }))
    }
}

fn effective_cache_payload(
    config: &StageConfig,
    requested: StageKvCachePayload,
) -> StagePrefixCachePayload {
    match requested {
        StageKvCachePayload::ResidentKv => StagePrefixCachePayload::ResidentKv,
        StageKvCachePayload::KvRecurrent => StagePrefixCachePayload::KvRecurrent,
        StageKvCachePayload::FullState => StagePrefixCachePayload::FullState,
        StageKvCachePayload::Auto => infer_cache_payload(config),
    }
}

fn infer_cache_payload(config: &StageConfig) -> StagePrefixCachePayload {
    let identity = format!(
        "{} {}",
        config.model_id,
        config.model_path.as_deref().unwrap_or_default()
    )
    .to_ascii_lowercase();

    if identity.contains("falcon-h1")
        || identity.contains("qwen3next")
        || identity.contains("qwen3-next")
        || identity.contains("qwen3.6")
        || identity.contains("qwen3_6")
    {
        return StagePrefixCachePayload::KvRecurrent;
    }
    if identity.contains("gemma")
        || identity.contains("glm-4.7")
        || identity.contains("glm47")
        || identity.contains("glm4.7")
    {
        return StagePrefixCachePayload::FullState;
    }
    if identity.contains("llama")
        || identity.contains("qwen3")
        || identity.contains("deepseek")
        || identity.contains("glm4")
        || identity.contains("olmo")
        || identity.contains("minimax")
    {
        return StagePrefixCachePayload::ResidentKv;
    }
    StagePrefixCachePayload::Disabled
}

fn effective_cache_config(config: &StageConfig) -> Option<StageKvCacheConfig> {
    if let Some(cache) = config.kv_cache.clone() {
        return Some(cache);
    }
    let mode = std::env::var("SKIPPY_KV_CACHE")
        .or_else(|_| std::env::var("SKIPPY_PREFIX_CACHE"))
        .ok()
        .and_then(|value| parse_cache_mode(&value));
    let mode = mode?;
    let max_entries = std::env::var("SKIPPY_KV_CACHE_MAX_ENTRIES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(64);
    let max_bytes = std::env::var("SKIPPY_KV_CACHE_MAX_BYTES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0);
    let min_tokens = std::env::var("SKIPPY_KV_CACHE_MIN_TOKENS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(64);
    let shared_prefix_stride_tokens = std::env::var("SKIPPY_KV_CACHE_SHARED_STRIDE_TOKENS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(128);
    let shared_prefix_record_limit = std::env::var("SKIPPY_KV_CACHE_SHARED_RECORD_LIMIT")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(2);
    let payload = std::env::var("SKIPPY_KV_CACHE_PAYLOAD")
        .ok()
        .and_then(|value| parse_cache_payload(&value))
        .unwrap_or(StageKvCachePayload::Auto);
    Some(StageKvCacheConfig {
        mode,
        payload,
        max_entries,
        max_bytes,
        min_tokens,
        shared_prefix_stride_tokens,
        shared_prefix_record_limit,
    })
}

fn parse_cache_payload(value: &str) -> Option<StageKvCachePayload> {
    match value.trim().to_ascii_lowercase().replace('_', "-").as_str() {
        "" | "auto" => Some(StageKvCachePayload::Auto),
        "resident" | "resident-kv" | "kv" => Some(StageKvCachePayload::ResidentKv),
        "kv-recurrent" | "kvrecurrent" => Some(StageKvCachePayload::KvRecurrent),
        "full" | "full-state" | "fullstate" => Some(StageKvCachePayload::FullState),
        _ => None,
    }
}

fn parse_cache_mode(value: &str) -> Option<StageKvCacheMode> {
    match value.trim().to_ascii_lowercase().replace('_', "-").as_str() {
        "" | "auto" => Some(StageKvCacheMode::Auto),
        "0" | "off" | "false" | "disabled" | "disable" => Some(StageKvCacheMode::Disabled),
        "record" => Some(StageKvCacheMode::Record),
        "1" | "on" | "true" | "lookup-record" | "lookuprecord" | "exact" => {
            Some(StageKvCacheMode::LookupRecord)
        }
        _ => None,
    }
}
