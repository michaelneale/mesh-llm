use std::path::{Path, PathBuf};

use anyhow::{bail, Result};
use skippy_topology::infer_family_capability;

use super::support::{pick_owned, pick_string, pick_string_owned};
use super::types::ResolvedSpeculativeConfig;
use crate::plugin::{BoolOrAuto, SpeculativeConfig};

pub(super) fn resolve_speculative_config(
    model_config: Option<&SpeculativeConfig>,
    global_config: Option<&SpeculativeConfig>,
    model_id: &str,
    model_path: &Path,
) -> Result<ResolvedSpeculativeConfig> {
    let mode = pick_string_owned(
        model_config.and_then(|config| config.mode.as_deref()),
        global_config.and_then(|config| config.mode.as_deref()),
        Some("auto"),
    );
    if mode == "ngram" {
        bail!("skippy speculative.mode = \"ngram\" is not supported by the embedded runtime");
    }
    if pick_owned(
        model_config.and_then(|config| config.draft_hf_repo.clone()),
        global_config.and_then(|config| config.draft_hf_repo.clone()),
    )
    .is_some()
        || pick_owned(
            model_config.and_then(|config| config.draft_hf_file.clone()),
            global_config.and_then(|config| config.draft_hf_file.clone()),
        )
        .is_some()
    {
        bail!(
            "skippy speculative Hugging Face draft sources are not supported by the embedded runtime"
        );
    }
    if pick_string(
        model_config.and_then(|config| config.draft_selection_policy.as_deref()),
        global_config.and_then(|config| config.draft_selection_policy.as_deref()),
        None,
    )
    .eq_ignore_ascii_case("auto")
    {
        bail!(
            "skippy speculative.draft_selection_policy = \"auto\" is not supported by the embedded runtime"
        );
    }
    if pick_owned(
        model_config.and_then(|config| config.draft_device.clone()),
        global_config.and_then(|config| config.draft_device.clone()),
    )
    .is_some()
        || pick_owned(
            model_config.and_then(|config| config.draft_threads),
            global_config.and_then(|config| config.draft_threads),
        )
        .is_some()
        || pick_owned(
            model_config.and_then(|config| config.draft_cache_type_k.clone()),
            global_config.and_then(|config| config.draft_cache_type_k.clone()),
        )
        .is_some()
        || pick_owned(
            model_config.and_then(|config| config.draft_cache_type_v.clone()),
            global_config.and_then(|config| config.draft_cache_type_v.clone()),
        )
        .is_some()
    {
        bail!("skippy explicit draft runtime overrides are not supported by the embedded runtime");
    }
    let draft_min_tokens = pick_owned(
        model_config.and_then(|config| config.draft_min_tokens),
        global_config.and_then(|config| config.draft_min_tokens),
    )
    .unwrap_or(0);
    if draft_min_tokens > 0 {
        bail!("skippy speculative.draft_min_tokens is not supported by the embedded runtime");
    }
    let draft_acceptance_threshold = pick_owned(
        model_config.and_then(|config| config.draft_acceptance_threshold),
        global_config.and_then(|config| config.draft_acceptance_threshold),
    )
    .unwrap_or(0.0);
    if draft_acceptance_threshold > 0.0 {
        bail!(
            "skippy speculative.draft_acceptance_threshold is not supported by the embedded runtime"
        );
    }
    let draft_split_probability = pick_owned(
        model_config.and_then(|config| config.draft_split_probability),
        global_config.and_then(|config| config.draft_split_probability),
    )
    .unwrap_or(0.0);
    if draft_split_probability > 0.0 {
        bail!(
            "skippy speculative.draft_split_probability is not supported by the embedded runtime"
        );
    }
    if let Some(BoolOrAuto::Bool(true)) = pick_owned(
        model_config.and_then(|config| config.spec_default.as_ref()),
        global_config.and_then(|config| config.spec_default.as_ref()),
    ) {
        bail!("skippy speculative.spec_default = true is not supported by the embedded runtime");
    }

    let mut mode = mode;
    let mut draft_model_path = pick_owned(
        model_config.and_then(|config| config.draft_model_path.clone()),
        global_config.and_then(|config| config.draft_model_path.clone()),
    )
    .map(PathBuf::from);
    let draft_max_tokens = super::support::pick_value(
        model_config.and_then(|config| config.draft_max_tokens),
        global_config.and_then(|config| config.draft_max_tokens),
        0,
    );
    let draft_n_gpu_layers = pick_owned(
        model_config.and_then(|config| config.draft_gpu_layers),
        global_config.and_then(|config| config.draft_gpu_layers),
    );
    let pairing_fault = normalize_pairing_fault(pick_string(
        model_config.and_then(|config| config.pairing_fault.as_deref()),
        global_config.and_then(|config| config.pairing_fault.as_deref()),
        Some("warn_disable"),
    ));
    let explicit = mode != "auto"
        || draft_model_path.is_some()
        || draft_max_tokens > 0
        || draft_n_gpu_layers.is_some();
    if mode == "disabled" && draft_model_path.is_some() {
        bail!("skippy speculative draft source cannot be set when speculative.mode = \"disabled\"");
    }
    if mode == "draft" || (mode == "auto" && draft_model_path.is_some()) {
        if draft_model_path.is_none() {
            bail!("skippy speculative draft mode requires an explicit draft_model_path");
        }
        if draft_max_tokens == 0 {
            bail!("skippy speculative draft mode requires draft_max_tokens > 0");
        }
        mode = "draft".to_string();
        let draft_path = draft_model_path.as_ref().expect("checked above");
        if let Some(reason) = incompatible_draft_pair_reason(model_id, model_path, draft_path) {
            match pairing_fault.as_str() {
                "warn_disable" => {
                    mode = "disabled".to_string();
                    draft_model_path = None;
                }
                "fail_open" => {}
                "fail_closed" => bail!("skippy incompatible speculative draft pairing: {reason}"),
                _ => unreachable!(),
            }
        }
    } else {
        mode = "disabled".to_string();
        draft_model_path = None;
    }
    Ok(ResolvedSpeculativeConfig {
        mode,
        draft_model_path,
        pairing_fault,
        draft_max_tokens,
        explicit,
        draft_n_gpu_layers,
    })
}

fn normalize_pairing_fault(value: &str) -> String {
    value.replace('-', "_")
}

fn incompatible_draft_pair_reason(
    model_id: &str,
    model_path: &Path,
    draft_model_path: &Path,
) -> Option<String> {
    let target_family = infer_family_capability(model_id, 0, 0)
        .map(|capability| capability.family_id.to_string())
        .or_else(|| infer_family_from_path_string(model_path));
    let draft_family = infer_family_from_path_string(draft_model_path);
    match (target_family, draft_family) {
        (Some(target_family), Some(draft_family)) if target_family != draft_family => Some(
            format!("target family {target_family} does not match draft family {draft_family}"),
        ),
        _ => None,
    }
}

fn infer_family_from_path_string(path: &Path) -> Option<String> {
    infer_family_capability(&path.display().to_string(), 0, 0)
        .map(|capability| capability.family_id.to_string())
}
