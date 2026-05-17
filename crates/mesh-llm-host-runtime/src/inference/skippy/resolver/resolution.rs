use std::path::PathBuf;

use anyhow::Result;

use super::super::{family_policy_for_model_path, KvCachePolicy};
use super::request_defaults::resolve_request_defaults;
use super::speculative::resolve_speculative_config;
use super::support::{
    bool_or_auto_value, derive_fit_target_mib, effective_flash_attention,
    has_explicit_prefill_controls, kv_macro_defaults, parse_gpu_layers, pick_owned, pick_string,
    pick_string_owned, pick_value, reject_unsupported_model_fit_controls, resolve_field_string,
    resolve_field_value, resolve_prefix_cache, resolve_wire_dtype, throughput_macro_defaults,
};
use super::types::{
    ResolvedHardwareConfig, ResolvedModelFitConfig, ResolvedSkippyConfig,
    ResolvedSkippyExecutionConfig, ResolvedThroughputConfig, SkippyConfigResolveRequest,
    BUILTIN_BATCH, BUILTIN_CTX_SIZE, BUILTIN_PARALLEL, BUILTIN_PREFILL_CHUNK_SIZE,
    BUILTIN_SAFETY_MARGIN_GB, BUILTIN_UBATCH,
};

pub(crate) fn resolve_skippy_config(
    request: SkippyConfigResolveRequest<'_>,
) -> Result<ResolvedSkippyConfig> {
    let model_entry = request
        .mesh_config
        .models
        .iter()
        .find(|entry| entry.model == request.model_id);
    let defaults = request.mesh_config.defaults.as_ref();
    let model_fit = model_entry.and_then(|entry| entry.model_fit.as_ref());
    let global_model_fit = defaults.and_then(|value| value.model_fit.as_ref());
    reject_unsupported_model_fit_controls(model_fit, "models[].model_fit")?;
    reject_unsupported_model_fit_controls(global_model_fit, "defaults.model_fit")?;
    let model_throughput = model_entry.and_then(|entry| entry.throughput.as_ref());
    let global_throughput = defaults.and_then(|value| value.throughput.as_ref());
    let family_policy = family_policy_for_model_path(request.model_path, Some(request.model_id));
    let kv_policy = KvCachePolicy::for_model_size(request.model_bytes);

    let model_kv_cache_policy = model_fit.and_then(|fit| fit.kv_cache_policy.as_deref());
    let global_kv_cache_policy = global_model_fit.and_then(|fit| fit.kv_cache_policy.as_deref());
    let effective_kv_cache_policy = pick_string(
        model_kv_cache_policy,
        global_kv_cache_policy,
        Some("balanced"),
    );
    let model_kv_macro = model_kv_cache_policy.map(|policy| kv_macro_defaults(policy, kv_policy));
    let global_kv_macro = global_kv_cache_policy.map(|policy| kv_macro_defaults(policy, kv_policy));

    let model_tuning_profile =
        model_throughput.and_then(|throughput| throughput.tuning_profile.as_deref());
    let global_tuning_profile =
        global_throughput.and_then(|throughput| throughput.tuning_profile.as_deref());
    let effective_tuning_profile = pick_string(
        model_tuning_profile,
        global_tuning_profile,
        Some("balanced"),
    );
    let model_throughput_macro = model_tuning_profile.map(throughput_macro_defaults);
    let global_throughput_macro = global_tuning_profile.map(throughput_macro_defaults);

    let ctx_size = pick_value(
        model_fit.and_then(|fit| fit.ctx_size),
        global_model_fit.and_then(|fit| fit.ctx_size),
        BUILTIN_CTX_SIZE,
    );
    let batch = resolve_field_value(
        model_fit.and_then(|fit| fit.batch),
        model_throughput_macro
            .as_ref()
            .and_then(|defaults| defaults.batch),
        global_model_fit.and_then(|fit| fit.batch),
        global_throughput_macro
            .as_ref()
            .and_then(|defaults| defaults.batch),
        BUILTIN_BATCH,
    );
    let ubatch = resolve_field_value(
        model_fit.and_then(|fit| fit.ubatch),
        model_throughput_macro
            .as_ref()
            .and_then(|defaults| defaults.ubatch),
        global_model_fit.and_then(|fit| fit.ubatch),
        global_throughput_macro
            .as_ref()
            .and_then(|defaults| defaults.ubatch),
        BUILTIN_UBATCH,
    );
    let cache_type_k = resolve_field_string(
        model_fit.and_then(|fit| fit.cache_type_k.as_deref()),
        model_kv_macro
            .as_ref()
            .and_then(|defaults| defaults.cache_type_k.as_deref()),
        global_model_fit.and_then(|fit| fit.cache_type_k.as_deref()),
        global_kv_macro
            .as_ref()
            .and_then(|defaults| defaults.cache_type_k.as_deref()),
        kv_policy.cache_type_k(),
    );
    let cache_type_v = resolve_field_string(
        model_fit.and_then(|fit| fit.cache_type_v.as_deref()),
        model_kv_macro
            .as_ref()
            .and_then(|defaults| defaults.cache_type_v.as_deref()),
        global_model_fit.and_then(|fit| fit.cache_type_v.as_deref()),
        global_kv_macro
            .as_ref()
            .and_then(|defaults| defaults.cache_type_v.as_deref()),
        kv_policy.cache_type_v(),
    );
    let model_kv_offload = model_fit
        .and_then(|fit| fit.kv_offload.as_ref())
        .map(bool_or_auto_value);
    let global_kv_offload = global_model_fit
        .and_then(|fit| fit.kv_offload.as_ref())
        .map(bool_or_auto_value);
    let kv_offload = resolve_field_string(
        model_kv_offload.as_deref(),
        model_kv_macro
            .as_ref()
            .and_then(|defaults| defaults.kv_offload.as_deref()),
        global_kv_offload.as_deref(),
        global_kv_macro
            .as_ref()
            .and_then(|defaults| defaults.kv_offload.as_deref()),
        "auto",
    );
    let flash_attention = model_fit
        .and_then(|fit| fit.flash_attention)
        .or(global_model_fit.and_then(|fit| fit.flash_attention))
        .unwrap_or_else(|| effective_flash_attention(&cache_type_v));
    let prefix_cache = resolve_prefix_cache(model_fit, global_model_fit)?;

    let device = pick_owned(
        model_entry
            .and_then(|entry| entry.hardware.as_ref())
            .and_then(|hardware| hardware.device.clone()),
        defaults
            .and_then(|value| value.hardware.as_ref())
            .and_then(|hardware| hardware.device.clone()),
    );
    let gpu_layers = parse_gpu_layers(
        model_entry
            .and_then(|entry| entry.hardware.as_ref())
            .and_then(|hardware| hardware.gpu_layers.as_ref()),
        defaults
            .and_then(|value| value.hardware.as_ref())
            .and_then(|hardware| hardware.gpu_layers.as_ref()),
    )
    .unwrap_or(-1);
    let safety_margin_gb = pick_owned(
        model_entry
            .and_then(|entry| entry.hardware.as_ref())
            .and_then(|hardware| hardware.safety_margin_gb),
        defaults
            .and_then(|value| value.hardware.as_ref())
            .and_then(|hardware| hardware.safety_margin_gb),
    )
    .unwrap_or(BUILTIN_SAFETY_MARGIN_GB);
    let fit_target_mib = pick_owned(
        model_entry
            .and_then(|entry| entry.hardware.as_ref())
            .and_then(|hardware| hardware.fit_target_mib),
        defaults
            .and_then(|value| value.hardware.as_ref())
            .and_then(|hardware| hardware.fit_target_mib),
    )
    .or_else(|| derive_fit_target_mib(request.allocatable_memory_bytes, safety_margin_gb));
    let configured_model_path = pick_owned(
        model_entry
            .and_then(|entry| entry.hardware.as_ref())
            .and_then(|hardware| hardware.model_path.clone()),
        defaults
            .and_then(|value| value.hardware.as_ref())
            .and_then(|hardware| hardware.model_path.clone()),
    )
    .map(PathBuf::from)
    .unwrap_or_else(|| request.model_path.to_path_buf());
    let projector_path = pick_owned(
        model_entry
            .and_then(|entry| entry.multimodal.as_ref())
            .and_then(|multimodal| multimodal.mmproj.clone())
            .or_else(|| {
                model_entry
                    .and_then(|entry| entry.hardware.as_ref())
                    .and_then(|hardware| hardware.mmproj.clone())
            }),
        defaults
            .and_then(|value| value.multimodal.as_ref())
            .and_then(|multimodal| multimodal.mmproj.clone())
            .or_else(|| {
                defaults
                    .and_then(|value| value.hardware.as_ref())
                    .and_then(|hardware| hardware.mmproj.clone())
            }),
    )
    .map(PathBuf::from);
    let stage_layer_start = pick_owned(
        model_entry
            .and_then(|entry| entry.hardware.as_ref())
            .and_then(|hardware| hardware.stage_layer_start),
        defaults
            .and_then(|value| value.hardware.as_ref())
            .and_then(|hardware| hardware.stage_layer_start),
    );
    let stage_layer_end = pick_owned(
        model_entry
            .and_then(|entry| entry.hardware.as_ref())
            .and_then(|hardware| hardware.stage_layer_end),
        defaults
            .and_then(|value| value.hardware.as_ref())
            .and_then(|hardware| hardware.stage_layer_end),
    );

    let parallel = resolve_field_value(
        model_throughput.and_then(|throughput| throughput.parallel),
        model_throughput_macro
            .as_ref()
            .and_then(|defaults| defaults.parallel),
        global_throughput.and_then(|throughput| throughput.parallel),
        global_throughput_macro
            .as_ref()
            .and_then(|defaults| defaults.parallel),
        BUILTIN_PARALLEL,
    );
    let model_continuous_batching = model_throughput
        .and_then(|throughput| throughput.continuous_batching.as_ref())
        .map(bool_or_auto_value);
    let global_continuous_batching = global_throughput
        .and_then(|throughput| throughput.continuous_batching.as_ref())
        .map(bool_or_auto_value);
    let continuous_batching = resolve_field_string(
        model_continuous_batching.as_deref(),
        model_throughput_macro
            .as_ref()
            .and_then(|defaults| defaults.continuous_batching.as_deref()),
        global_continuous_batching.as_deref(),
        global_throughput_macro
            .as_ref()
            .and_then(|defaults| defaults.continuous_batching.as_deref()),
        "auto",
    );
    let threads = pick_owned(
        model_throughput.and_then(|throughput| throughput.threads),
        global_throughput.and_then(|throughput| throughput.threads),
    );
    let threads_batch = pick_owned(
        model_throughput.and_then(|throughput| throughput.threads_batch),
        global_throughput.and_then(|throughput| throughput.threads_batch),
    );

    let activation_wire_dtype = resolve_wire_dtype(
        model_entry
            .and_then(|entry| entry.skippy.as_ref())
            .and_then(|skippy| skippy.activation_wire_dtype.as_deref()),
        defaults
            .and_then(|value| value.skippy.as_ref())
            .and_then(|skippy| skippy.activation_wire_dtype.as_deref()),
        family_policy.activation_wire_dtype,
    );
    let binary_stage_transport = pick_string_owned(
        model_entry
            .and_then(|entry| entry.skippy.as_ref())
            .and_then(|skippy| skippy.binary_stage_transport.as_deref()),
        defaults
            .and_then(|value| value.skippy.as_ref())
            .and_then(|skippy| skippy.binary_stage_transport.as_deref()),
        Some("auto"),
    );
    let prefill_chunking = pick_string_owned(
        model_entry
            .and_then(|entry| entry.skippy.as_ref())
            .and_then(|skippy| skippy.prefill_chunking.as_deref()),
        defaults
            .and_then(|value| value.skippy.as_ref())
            .and_then(|skippy| skippy.prefill_chunking.as_deref()),
        Some("fixed"),
    );
    let prefill_chunk_size = pick_owned(
        model_entry
            .and_then(|entry| entry.skippy.as_ref())
            .and_then(|skippy| skippy.prefill_chunk_size),
        defaults
            .and_then(|value| value.skippy.as_ref())
            .and_then(|skippy| skippy.prefill_chunk_size),
    )
    .map(|value| value as usize)
    .unwrap_or(BUILTIN_PREFILL_CHUNK_SIZE);
    let prefill_chunk_schedule = pick_owned(
        model_entry
            .and_then(|entry| entry.skippy.as_ref())
            .and_then(|skippy| skippy.prefill_chunk_schedule.clone()),
        defaults
            .and_then(|value| value.skippy.as_ref())
            .and_then(|skippy| skippy.prefill_chunk_schedule.clone()),
    );
    let activation_wire_dtype_explicit = model_entry
        .and_then(|entry| entry.skippy.as_ref())
        .and_then(|skippy| skippy.activation_wire_dtype.as_deref())
        .or_else(|| {
            defaults
                .and_then(|value| value.skippy.as_ref())
                .and_then(|skippy| skippy.activation_wire_dtype.as_deref())
        })
        .is_some_and(|value| !value.eq_ignore_ascii_case("auto"));
    let prefill_controls_explicit = model_entry
        .and_then(|entry| entry.skippy.as_ref())
        .is_some_and(has_explicit_prefill_controls)
        || defaults
            .and_then(|value| value.skippy.as_ref())
            .is_some_and(has_explicit_prefill_controls);
    let lifecycle_startup_timeout_ms = pick_owned(
        model_entry
            .and_then(|entry| entry.skippy.as_ref())
            .and_then(|skippy| skippy.lifecycle_startup_timeout_ms),
        defaults
            .and_then(|value| value.skippy.as_ref())
            .and_then(|skippy| skippy.lifecycle_startup_timeout_ms),
    );
    let lifecycle_readiness_interval_ms = pick_owned(
        model_entry
            .and_then(|entry| entry.skippy.as_ref())
            .and_then(|skippy| skippy.lifecycle_readiness_interval_ms),
        defaults
            .and_then(|value| value.skippy.as_ref())
            .and_then(|skippy| skippy.lifecycle_readiness_interval_ms),
    );
    let lifecycle_health_interval_ms = pick_owned(
        model_entry
            .and_then(|entry| entry.skippy.as_ref())
            .and_then(|skippy| skippy.lifecycle_health_interval_ms),
        defaults
            .and_then(|value| value.skippy.as_ref())
            .and_then(|skippy| skippy.lifecycle_health_interval_ms),
    );

    let speculative = resolve_speculative_config(
        model_entry.and_then(|entry| entry.speculative.as_ref()),
        defaults.and_then(|value| value.speculative.as_ref()),
        request.model_id,
        request.model_path,
    )?;

    let resolved_request =
        resolve_request_defaults(defaults, model_entry, request.request_defaults)?;

    Ok(ResolvedSkippyConfig {
        model_id: request.model_id.to_string(),
        model_path: request.model_path.to_path_buf(),
        model_fit: ResolvedModelFitConfig {
            ctx_size,
            batch,
            ubatch,
            cache_type_k,
            cache_type_v,
            kv_cache_policy: effective_kv_cache_policy.to_string(),
            prefix_cache,
            kv_offload,
            flash_attention,
        },
        hardware: ResolvedHardwareConfig {
            device,
            gpu_layers,
            fit_target_mib,
            resolved_model_path: configured_model_path,
            projector_path,
            stage_layer_start,
            stage_layer_end,
        },
        throughput: ResolvedThroughputConfig {
            parallel,
            continuous_batching,
            threads,
            threads_batch,
            tuning_profile: effective_tuning_profile.to_string(),
        },
        skippy: ResolvedSkippyExecutionConfig {
            activation_wire_dtype,
            activation_wire_dtype_explicit,
            binary_stage_transport,
            prefill_chunking,
            prefill_chunk_size,
            prefill_chunk_schedule,
            prefill_controls_explicit,
            lifecycle_startup_timeout_ms,
            lifecycle_readiness_interval_ms,
            lifecycle_health_interval_ms,
        },
        speculative,
        request_defaults: resolved_request,
    })
}
