use std::path::PathBuf;

use anyhow::Result;

use super::super::{family_policy_for_model_path, KvCachePolicy, StageWireDType};
use super::request_defaults::resolve_request_defaults;
use super::speculative::resolve_speculative_config;
use super::support::{
    bool_or_auto_value, derive_fit_target_mib, effective_flash_attention,
    has_explicit_prefill_controls, kv_macro_defaults, parse_gpu_layers, pick_owned, pick_string,
    pick_string_owned, pick_value, reject_unsupported_model_fit_controls, resolve_field_string,
    resolve_field_value, resolve_prefix_cache, resolve_wire_dtype, throughput_macro_defaults,
    KvMacroDefaults, ThroughputMacroDefaults,
};
use super::types::{
    ResolvedHardwareConfig, ResolvedModelFitConfig, ResolvedSkippyConfig,
    ResolvedSkippyExecutionConfig, ResolvedThroughputConfig, SkippyConfigResolveRequest,
    BUILTIN_BATCH, BUILTIN_CTX_SIZE, BUILTIN_PARALLEL, BUILTIN_PREFILL_CHUNK_SIZE,
    BUILTIN_SAFETY_MARGIN_GB, BUILTIN_UBATCH,
};
use crate::plugin::{ModelConfigDefaults, ModelConfigEntry, ModelFitConfig, ThroughputConfig};

pub(crate) fn resolve_skippy_config(
    request: SkippyConfigResolveRequest<'_>,
) -> Result<ResolvedSkippyConfig> {
    let context = ResolverContext::new(request);
    validate_supported_model_fit_controls(&context)?;

    let family_policy =
        family_policy_for_model_path(context.request.model_path, Some(context.request.model_id));
    let kv_policy = KvCachePolicy::for_model_size(context.request.model_bytes);

    let model_fit = resolve_model_fit_config(&context, kv_policy)?;
    let hardware = resolve_hardware_config(&context);
    let throughput = resolve_throughput_config(&context);
    let skippy = resolve_execution_config(&context, family_policy.activation_wire_dtype);
    let speculative = resolve_speculative_config(
        context
            .model_entry
            .and_then(|entry| entry.speculative.as_ref()),
        context
            .defaults
            .and_then(|value| value.speculative.as_ref()),
        context.request.model_id,
        context.request.model_path,
    )?;
    let resolved_request = resolve_request_defaults(
        context.defaults,
        context.model_entry,
        context.request.request_defaults,
    )?;

    Ok(ResolvedSkippyConfig {
        model_id: context.request.model_id.to_string(),
        model_path: context.request.model_path.to_path_buf(),
        model_fit,
        hardware,
        throughput,
        skippy,
        speculative,
        request_defaults: resolved_request,
    })
}

struct ResolverContext<'a> {
    request: SkippyConfigResolveRequest<'a>,
    model_entry: Option<&'a ModelConfigEntry>,
    defaults: Option<&'a ModelConfigDefaults>,
    model_fit: Option<&'a ModelFitConfig>,
    global_model_fit: Option<&'a ModelFitConfig>,
    model_throughput: Option<&'a ThroughputConfig>,
    global_throughput: Option<&'a ThroughputConfig>,
}

impl<'a> ResolverContext<'a> {
    fn new(request: SkippyConfigResolveRequest<'a>) -> Self {
        let mesh_config = request.mesh_config;
        let model_entry = mesh_config
            .models
            .iter()
            .find(|entry| entry.model == request.model_id);
        let defaults = mesh_config.defaults.as_ref();
        let model_fit = model_entry.and_then(|entry| entry.model_fit.as_ref());
        let global_model_fit = defaults.and_then(|value| value.model_fit.as_ref());
        let model_throughput = model_entry.and_then(|entry| entry.throughput.as_ref());
        let global_throughput = defaults.and_then(|value| value.throughput.as_ref());

        Self {
            request,
            model_entry,
            defaults,
            model_fit,
            global_model_fit,
            model_throughput,
            global_throughput,
        }
    }
}

fn validate_supported_model_fit_controls(context: &ResolverContext<'_>) -> Result<()> {
    reject_unsupported_model_fit_controls(context.model_fit, "models[].model_fit")?;
    reject_unsupported_model_fit_controls(context.global_model_fit, "defaults.model_fit")
}

fn resolve_model_fit_config(
    context: &ResolverContext<'_>,
    kv_policy: KvCachePolicy,
) -> Result<ResolvedModelFitConfig> {
    let kv = resolve_kv_defaults(context, kv_policy);
    let throughput = resolve_throughput_defaults(context);

    let ctx_size = pick_value(
        context.model_fit.and_then(|fit| fit.ctx_size),
        context.global_model_fit.and_then(|fit| fit.ctx_size),
        BUILTIN_CTX_SIZE,
    );
    let batch = resolve_field_value(
        context.model_fit.and_then(|fit| fit.batch),
        throughput
            .model_macro
            .as_ref()
            .and_then(|defaults| defaults.batch),
        context.global_model_fit.and_then(|fit| fit.batch),
        throughput
            .global_macro
            .as_ref()
            .and_then(|defaults| defaults.batch),
        BUILTIN_BATCH,
    );
    let ubatch = resolve_field_value(
        context.model_fit.and_then(|fit| fit.ubatch),
        throughput
            .model_macro
            .as_ref()
            .and_then(|defaults| defaults.ubatch),
        context.global_model_fit.and_then(|fit| fit.ubatch),
        throughput
            .global_macro
            .as_ref()
            .and_then(|defaults| defaults.ubatch),
        BUILTIN_UBATCH,
    );
    let cache_type_k = resolve_cache_type_k(context, &kv, kv_policy);
    let cache_type_v = resolve_cache_type_v(context, &kv, kv_policy);
    let kv_offload = resolve_kv_offload(context, &kv);
    let flash_attention = context
        .model_fit
        .and_then(|fit| fit.flash_attention)
        .or(context.global_model_fit.and_then(|fit| fit.flash_attention))
        .unwrap_or_else(|| effective_flash_attention(&cache_type_v));
    let prefix_cache = resolve_prefix_cache(context.model_fit, context.global_model_fit)?;

    Ok(ResolvedModelFitConfig {
        ctx_size,
        batch,
        ubatch,
        cache_type_k,
        cache_type_v,
        kv_cache_policy: kv.effective_policy,
        prefix_cache,
        kv_offload,
        flash_attention,
    })
}

struct KvDefaults {
    effective_policy: String,
    model_macro: Option<KvMacroDefaults>,
    global_macro: Option<KvMacroDefaults>,
}

fn resolve_kv_defaults(context: &ResolverContext<'_>, kv_policy: KvCachePolicy) -> KvDefaults {
    let model_policy = context
        .model_fit
        .and_then(|fit| fit.kv_cache_policy.as_deref());
    let global_policy = context
        .global_model_fit
        .and_then(|fit| fit.kv_cache_policy.as_deref());
    let effective_policy = pick_string(model_policy, global_policy, Some("balanced"));

    KvDefaults {
        effective_policy: effective_policy.to_string(),
        model_macro: model_policy.map(|policy| kv_macro_defaults(policy, kv_policy)),
        global_macro: global_policy.map(|policy| kv_macro_defaults(policy, kv_policy)),
    }
}

fn resolve_cache_type_k(
    context: &ResolverContext<'_>,
    kv: &KvDefaults,
    kv_policy: KvCachePolicy,
) -> String {
    resolve_field_string(
        context
            .model_fit
            .and_then(|fit| fit.cache_type_k.as_deref()),
        kv.model_macro
            .as_ref()
            .and_then(|defaults| defaults.cache_type_k.as_deref()),
        context
            .global_model_fit
            .and_then(|fit| fit.cache_type_k.as_deref()),
        kv.global_macro
            .as_ref()
            .and_then(|defaults| defaults.cache_type_k.as_deref()),
        kv_policy.cache_type_k(),
    )
}

fn resolve_cache_type_v(
    context: &ResolverContext<'_>,
    kv: &KvDefaults,
    kv_policy: KvCachePolicy,
) -> String {
    resolve_field_string(
        context
            .model_fit
            .and_then(|fit| fit.cache_type_v.as_deref()),
        kv.model_macro
            .as_ref()
            .and_then(|defaults| defaults.cache_type_v.as_deref()),
        context
            .global_model_fit
            .and_then(|fit| fit.cache_type_v.as_deref()),
        kv.global_macro
            .as_ref()
            .and_then(|defaults| defaults.cache_type_v.as_deref()),
        kv_policy.cache_type_v(),
    )
}

fn resolve_kv_offload(context: &ResolverContext<'_>, kv: &KvDefaults) -> String {
    let model_kv_offload = context
        .model_fit
        .and_then(|fit| fit.kv_offload.as_ref())
        .map(bool_or_auto_value);
    let global_kv_offload = context
        .global_model_fit
        .and_then(|fit| fit.kv_offload.as_ref())
        .map(bool_or_auto_value);

    resolve_field_string(
        model_kv_offload.as_deref(),
        kv.model_macro
            .as_ref()
            .and_then(|defaults| defaults.kv_offload.as_deref()),
        global_kv_offload.as_deref(),
        kv.global_macro
            .as_ref()
            .and_then(|defaults| defaults.kv_offload.as_deref()),
        "auto",
    )
}

fn resolve_hardware_config(context: &ResolverContext<'_>) -> ResolvedHardwareConfig {
    let model_hardware = context
        .model_entry
        .and_then(|entry| entry.hardware.as_ref());
    let global_hardware = context.defaults.and_then(|value| value.hardware.as_ref());

    let device = pick_owned(
        model_hardware.and_then(|hardware| hardware.device.clone()),
        global_hardware.and_then(|hardware| hardware.device.clone()),
    );
    let gpu_layers = parse_gpu_layers(
        model_hardware.and_then(|hardware| hardware.gpu_layers.as_ref()),
        global_hardware.and_then(|hardware| hardware.gpu_layers.as_ref()),
    )
    .unwrap_or(-1);
    let safety_margin_gb = pick_owned(
        model_hardware.and_then(|hardware| hardware.safety_margin_gb),
        global_hardware.and_then(|hardware| hardware.safety_margin_gb),
    )
    .unwrap_or(BUILTIN_SAFETY_MARGIN_GB);
    let fit_target_mib = pick_owned(
        model_hardware.and_then(|hardware| hardware.fit_target_mib),
        global_hardware.and_then(|hardware| hardware.fit_target_mib),
    )
    .or_else(|| derive_fit_target_mib(context.request.allocatable_memory_bytes, safety_margin_gb));
    let resolved_model_path = pick_owned(
        model_hardware.and_then(|hardware| hardware.model_path.clone()),
        global_hardware.and_then(|hardware| hardware.model_path.clone()),
    )
    .map(PathBuf::from)
    .unwrap_or_else(|| context.request.model_path.to_path_buf());
    let projector_path = resolve_projector_path(context);
    let stage_layer_start = pick_owned(
        model_hardware.and_then(|hardware| hardware.stage_layer_start),
        global_hardware.and_then(|hardware| hardware.stage_layer_start),
    );
    let stage_layer_end = pick_owned(
        model_hardware.and_then(|hardware| hardware.stage_layer_end),
        global_hardware.and_then(|hardware| hardware.stage_layer_end),
    );

    ResolvedHardwareConfig {
        device,
        gpu_layers,
        fit_target_mib,
        resolved_model_path,
        projector_path,
        stage_layer_start,
        stage_layer_end,
    }
}

fn resolve_projector_path(context: &ResolverContext<'_>) -> Option<PathBuf> {
    pick_owned(
        context
            .model_entry
            .and_then(|entry| entry.multimodal.as_ref())
            .and_then(|multimodal| multimodal.mmproj.clone())
            .or_else(|| {
                context
                    .model_entry
                    .and_then(|entry| entry.hardware.as_ref())
                    .and_then(|hardware| hardware.mmproj.clone())
            }),
        context
            .defaults
            .and_then(|value| value.multimodal.as_ref())
            .and_then(|multimodal| multimodal.mmproj.clone())
            .or_else(|| {
                context
                    .defaults
                    .and_then(|value| value.hardware.as_ref())
                    .and_then(|hardware| hardware.mmproj.clone())
            }),
    )
    .map(PathBuf::from)
}

struct ThroughputDefaults {
    effective_profile: String,
    model_macro: Option<ThroughputMacroDefaults>,
    global_macro: Option<ThroughputMacroDefaults>,
}

fn resolve_throughput_defaults(context: &ResolverContext<'_>) -> ThroughputDefaults {
    let model_profile = context
        .model_throughput
        .and_then(|throughput| throughput.tuning_profile.as_deref());
    let global_profile = context
        .global_throughput
        .and_then(|throughput| throughput.tuning_profile.as_deref());
    let effective_profile = pick_string(model_profile, global_profile, Some("balanced"));

    ThroughputDefaults {
        effective_profile: effective_profile.to_string(),
        model_macro: model_profile.map(throughput_macro_defaults),
        global_macro: global_profile.map(throughput_macro_defaults),
    }
}

fn resolve_throughput_config(context: &ResolverContext<'_>) -> ResolvedThroughputConfig {
    let throughput = resolve_throughput_defaults(context);
    let parallel = resolve_field_value(
        context
            .model_throughput
            .and_then(|throughput| throughput.parallel),
        throughput
            .model_macro
            .as_ref()
            .and_then(|defaults| defaults.parallel),
        context
            .global_throughput
            .and_then(|throughput| throughput.parallel),
        throughput
            .global_macro
            .as_ref()
            .and_then(|defaults| defaults.parallel),
        BUILTIN_PARALLEL,
    );
    let continuous_batching = resolve_continuous_batching(context, &throughput);
    let threads = pick_owned(
        context
            .model_throughput
            .and_then(|throughput| throughput.threads),
        context
            .global_throughput
            .and_then(|throughput| throughput.threads),
    );
    let threads_batch = pick_owned(
        context
            .model_throughput
            .and_then(|throughput| throughput.threads_batch),
        context
            .global_throughput
            .and_then(|throughput| throughput.threads_batch),
    );

    ResolvedThroughputConfig {
        parallel,
        continuous_batching,
        threads,
        threads_batch,
        tuning_profile: throughput.effective_profile,
    }
}

fn resolve_continuous_batching(
    context: &ResolverContext<'_>,
    throughput: &ThroughputDefaults,
) -> String {
    let model_continuous_batching = context
        .model_throughput
        .and_then(|throughput| throughput.continuous_batching.as_ref())
        .map(bool_or_auto_value);
    let global_continuous_batching = context
        .global_throughput
        .and_then(|throughput| throughput.continuous_batching.as_ref())
        .map(bool_or_auto_value);

    resolve_field_string(
        model_continuous_batching.as_deref(),
        throughput
            .model_macro
            .as_ref()
            .and_then(|defaults| defaults.continuous_batching.as_deref()),
        global_continuous_batching.as_deref(),
        throughput
            .global_macro
            .as_ref()
            .and_then(|defaults| defaults.continuous_batching.as_deref()),
        "auto",
    )
}

fn resolve_execution_config(
    context: &ResolverContext<'_>,
    family_wire_dtype: StageWireDType,
) -> ResolvedSkippyExecutionConfig {
    let model_skippy = context.model_entry.and_then(|entry| entry.skippy.as_ref());
    let global_skippy = context.defaults.and_then(|value| value.skippy.as_ref());

    let activation_wire_dtype = resolve_wire_dtype(
        model_skippy.and_then(|skippy| skippy.activation_wire_dtype.as_deref()),
        global_skippy.and_then(|skippy| skippy.activation_wire_dtype.as_deref()),
        family_wire_dtype,
    );
    let binary_stage_transport = pick_string_owned(
        model_skippy.and_then(|skippy| skippy.binary_stage_transport.as_deref()),
        global_skippy.and_then(|skippy| skippy.binary_stage_transport.as_deref()),
        Some("auto"),
    );
    let prefill_chunking = pick_string_owned(
        model_skippy.and_then(|skippy| skippy.prefill_chunking.as_deref()),
        global_skippy.and_then(|skippy| skippy.prefill_chunking.as_deref()),
        Some("fixed"),
    );
    let prefill_chunk_size = pick_owned(
        model_skippy.and_then(|skippy| skippy.prefill_chunk_size),
        global_skippy.and_then(|skippy| skippy.prefill_chunk_size),
    )
    .map(|value| value as usize)
    .unwrap_or(BUILTIN_PREFILL_CHUNK_SIZE);
    let prefill_chunk_schedule = pick_owned(
        model_skippy.and_then(|skippy| skippy.prefill_chunk_schedule.clone()),
        global_skippy.and_then(|skippy| skippy.prefill_chunk_schedule.clone()),
    );
    let activation_wire_dtype_explicit = model_skippy
        .and_then(|skippy| skippy.activation_wire_dtype.as_deref())
        .or_else(|| global_skippy.and_then(|skippy| skippy.activation_wire_dtype.as_deref()))
        .is_some_and(|value| !value.eq_ignore_ascii_case("auto"));
    let prefill_controls_explicit = model_skippy.is_some_and(has_explicit_prefill_controls)
        || global_skippy.is_some_and(has_explicit_prefill_controls);

    ResolvedSkippyExecutionConfig {
        activation_wire_dtype,
        activation_wire_dtype_explicit,
        binary_stage_transport,
        prefill_chunking,
        prefill_chunk_size,
        prefill_chunk_schedule,
        prefill_controls_explicit,
        lifecycle_startup_timeout_ms: pick_owned(
            model_skippy.and_then(|skippy| skippy.lifecycle_startup_timeout_ms),
            global_skippy.and_then(|skippy| skippy.lifecycle_startup_timeout_ms),
        ),
        lifecycle_readiness_interval_ms: pick_owned(
            model_skippy.and_then(|skippy| skippy.lifecycle_readiness_interval_ms),
            global_skippy.and_then(|skippy| skippy.lifecycle_readiness_interval_ms),
        ),
        lifecycle_health_interval_ms: pick_owned(
            model_skippy.and_then(|skippy| skippy.lifecycle_health_interval_ms),
            global_skippy.and_then(|skippy| skippy.lifecycle_health_interval_ms),
        ),
    }
}
