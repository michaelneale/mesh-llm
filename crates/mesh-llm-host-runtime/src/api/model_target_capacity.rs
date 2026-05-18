//! Advisory model-target capacity evaluation for management API responses.
//!
//! This is intentionally local to the API layer: it derives operator hints from
//! existing mesh/catalog signals and does not affect routing, startup, gossip,
//! or protocol compatibility.

use super::status::{ModelTargetCapacityAdvicePayload, ModelTargetCapacityAdviceState};
use crate::mesh::{NodeRole, PeerInfo};
use crate::models;
use crate::runtime;
use std::collections::HashMap;

#[derive(Clone, Copy, Debug)]
pub(crate) struct ModelTargetCapacityInput<'a> {
    pub(crate) model_ref: &'a str,
    pub(crate) model_name: Option<&'a str>,
    pub(crate) serving_node_count: usize,
    pub(crate) local_role: &'a NodeRole,
    pub(crate) local_vram_bytes: u64,
    pub(crate) peers: &'a [PeerInfo],
    pub(crate) size_lookup: &'a ModelTargetSizeLookup,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ModelSizeHint {
    model_bytes: u64,
    split_capable: bool,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct ModelTargetSizeLookup {
    hints_by_key: HashMap<String, ModelSizeHint>,
}

impl ModelTargetSizeLookup {
    pub(crate) fn load() -> Self {
        models::remote_catalog::catalog_entries()
            .map(Self::from_entries)
            .unwrap_or_default()
    }

    fn from_entries(entries: Vec<models::remote_catalog::CatalogEntry>) -> Self {
        let mut lookup = Self::default();
        for entry in entries {
            let mut variants = entry.variants.iter().collect::<Vec<_>>();
            variants.sort_by(|left, right| left.0.cmp(right.0));
            for (variant_name, variant) in variants {
                let source_file = variant
                    .source
                    .file
                    .as_deref()
                    .unwrap_or(variant_name.as_str());
                let split_capable = variant
                    .packages
                    .iter()
                    .any(|package| package.package_type == "layer-package");
                if let Some(model_bytes) = variant
                    .curated
                    .size
                    .as_deref()
                    .and_then(parse_size_label_bytes)
                {
                    lookup.insert_model_aliases(
                        variant_name,
                        &variant.curated.name,
                        &variant.source.repo,
                        variant.source.revision.as_deref(),
                        source_file,
                        ModelSizeHint {
                            model_bytes,
                            split_capable,
                        },
                    );
                }

                for package in variant
                    .packages
                    .iter()
                    .filter(|package| package.package_type == "layer-package")
                {
                    if let Some(model_bytes) = package.total_bytes {
                        lookup.insert_package_alias(
                            &package.repo,
                            ModelSizeHint {
                                model_bytes,
                                split_capable: true,
                            },
                        );
                    }
                }
            }
        }
        lookup
    }

    fn find(&self, query: &str) -> Option<ModelSizeHint> {
        self.hints_by_key.get(&normalize_match_key(query)).copied()
    }

    fn insert_model_aliases(
        &mut self,
        variant_name: &str,
        curated_name: &str,
        repo: &str,
        revision: Option<&str>,
        source_file: &str,
        hint: ModelSizeHint,
    ) {
        let basename = source_file.rsplit('/').next().unwrap_or(source_file);
        let selector = model_ref::quant_selector_from_gguf_file(source_file);
        let model_ref_with_revision =
            model_ref::format_model_ref(repo, revision, selector.as_deref());
        let model_ref_without_revision =
            model_ref::format_model_ref(repo, None, selector.as_deref());
        let canonical_ref =
            revision.map(|revision| model_ref::format_canonical_ref(repo, revision, source_file));

        for alias in [
            variant_name,
            curated_name,
            repo,
            source_file,
            basename,
            basename.trim_end_matches(".gguf"),
            model_ref_with_revision.as_str(),
            model_ref_without_revision.as_str(),
        ] {
            self.insert_model_alias(alias, hint);
        }
        if let Some(canonical_ref) = canonical_ref {
            self.insert_model_alias(&canonical_ref, hint);
        }
    }

    fn insert_model_alias(&mut self, alias: &str, hint: ModelSizeHint) {
        self.hints_by_key
            .entry(normalize_match_key(alias))
            .or_insert(hint);
    }

    fn insert_package_alias(&mut self, alias: &str, hint: ModelSizeHint) {
        self.hints_by_key.insert(normalize_match_key(alias), hint);
    }
}

pub(crate) fn evaluate_model_target_capacity(
    input: ModelTargetCapacityInput<'_>,
) -> ModelTargetCapacityAdvicePayload {
    let capacity = collect_capacity(input.local_role, input.local_vram_bytes, input.peers);
    let size_hint = input.size_lookup.find(input.model_ref).or_else(|| {
        input
            .model_name
            .and_then(|name| input.size_lookup.find(name))
    });
    let required_bytes = size_hint
        .map(|hint| runtime::runtime_model_required_bytes(hint.model_bytes))
        .filter(|required| *required > 0);
    let split_capable = size_hint.map(|hint| hint.split_capable).unwrap_or(false);

    if input.serving_node_count > 0 {
        return advice(
            ModelTargetCapacityAdviceState::AlreadyServing,
            "already_serving",
            capacity,
            AdviceDetails {
                required_bytes,
                shortfall_bytes: None,
                split_capable,
            },
        );
    }

    let Some(required_bytes) = required_bytes else {
        return advice(
            ModelTargetCapacityAdviceState::UnknownModelSize,
            "model_size_unknown",
            capacity,
            AdviceDetails {
                required_bytes: None,
                shortfall_bytes: None,
                split_capable,
            },
        );
    };

    if capacity.missing_capacity_node_count > 0 {
        return advice(
            ModelTargetCapacityAdviceState::UnknownCapacity,
            "eligible_nodes_missing_capacity",
            capacity,
            AdviceDetails {
                required_bytes: Some(required_bytes),
                shortfall_bytes: None,
                split_capable,
            },
        );
    }

    if capacity.eligible_node_count == 0 {
        return advice(
            ModelTargetCapacityAdviceState::NoEligibleHosts,
            "no_worker_or_host_capacity",
            capacity,
            AdviceDetails {
                required_bytes: Some(required_bytes),
                shortfall_bytes: None,
                split_capable,
            },
        );
    }

    if capacity
        .best_single_node_capacity_bytes
        .is_some_and(|best| best >= required_bytes)
    {
        return advice(
            ModelTargetCapacityAdviceState::SingleNodeFit,
            "single_node_capacity_available",
            capacity,
            AdviceDetails {
                required_bytes: Some(required_bytes),
                shortfall_bytes: None,
                split_capable,
            },
        );
    }

    if split_capable
        && capacity.eligible_node_count >= 2
        && capacity.aggregate_capacity_bytes >= required_bytes
    {
        return advice(
            ModelTargetCapacityAdviceState::SplitCandidate,
            "aggregate_split_capacity_available",
            capacity,
            AdviceDetails {
                required_bytes: Some(required_bytes),
                shortfall_bytes: None,
                split_capable,
            },
        );
    }

    let comparable_capacity = if split_capable && capacity.eligible_node_count >= 2 {
        capacity.aggregate_capacity_bytes
    } else {
        capacity.best_single_node_capacity_bytes.unwrap_or_default()
    };
    debug_assert_eq!(capacity.missing_capacity_node_count, 0);
    let shortfall_bytes = required_bytes.saturating_sub(comparable_capacity);
    advice(
        ModelTargetCapacityAdviceState::InsufficientCapacity,
        "capacity_shortfall",
        capacity,
        AdviceDetails {
            required_bytes: Some(required_bytes),
            shortfall_bytes: Some(shortfall_bytes),
            split_capable,
        },
    )
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct CapacitySummary {
    best_single_node_capacity_bytes: Option<u64>,
    aggregate_capacity_bytes: u64,
    eligible_node_count: usize,
    missing_capacity_node_count: usize,
    excluded_client_node_count: usize,
}

fn collect_capacity(
    local_role: &NodeRole,
    local_vram_bytes: u64,
    peers: &[PeerInfo],
) -> CapacitySummary {
    let mut summary = CapacitySummary::default();
    record_node_capacity(&mut summary, local_role, local_vram_bytes);
    for peer in peers {
        record_node_capacity(&mut summary, &peer.role, peer.vram_bytes);
    }
    summary
}

fn record_node_capacity(summary: &mut CapacitySummary, role: &NodeRole, vram_bytes: u64) {
    if matches!(role, NodeRole::Client) {
        summary.excluded_client_node_count += 1;
        return;
    }
    if vram_bytes == 0 {
        summary.missing_capacity_node_count += 1;
        return;
    }

    summary.eligible_node_count += 1;
    summary.aggregate_capacity_bytes = summary.aggregate_capacity_bytes.saturating_add(vram_bytes);
    summary.best_single_node_capacity_bytes = Some(
        summary
            .best_single_node_capacity_bytes
            .map(|best| best.max(vram_bytes))
            .unwrap_or(vram_bytes),
    );
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct AdviceDetails {
    required_bytes: Option<u64>,
    shortfall_bytes: Option<u64>,
    split_capable: bool,
}

fn advice(
    state: ModelTargetCapacityAdviceState,
    reason: &'static str,
    capacity: CapacitySummary,
    details: AdviceDetails,
) -> ModelTargetCapacityAdvicePayload {
    ModelTargetCapacityAdvicePayload {
        state,
        reason,
        required_bytes: details.required_bytes,
        best_single_node_capacity_bytes: capacity.best_single_node_capacity_bytes,
        aggregate_capacity_bytes: capacity.aggregate_capacity_bytes,
        shortfall_bytes: details.shortfall_bytes,
        eligible_node_count: capacity.eligible_node_count,
        missing_capacity_node_count: capacity.missing_capacity_node_count,
        excluded_client_node_count: capacity.excluded_client_node_count,
        split_capable: details.split_capable,
    }
}

fn normalize_match_key(value: &str) -> String {
    value.trim().trim_start_matches("hf://").to_lowercase()
}

fn parse_size_label_bytes(label: &str) -> Option<u64> {
    let compact = label.trim().replace(' ', "");
    if compact.is_empty() {
        return None;
    }

    let split_at = compact
        .find(|ch: char| !(ch.is_ascii_digit() || ch == '.'))
        .unwrap_or(compact.len());
    if split_at == 0 {
        return None;
    }
    let value = compact[..split_at].parse::<f64>().ok()?;
    if !value.is_finite() || value < 0.0 {
        return None;
    }

    let unit = compact[split_at..].to_ascii_lowercase();
    let multiplier = match unit.as_str() {
        "" | "b" => 1.0,
        "kb" => 1e3,
        "mb" => 1e6,
        "gb" => 1e9,
        "tb" => 1e12,
        "kib" => 1024.0,
        "mib" => 1024.0_f64.powi(2),
        "gib" => 1024.0_f64.powi(3),
        "tib" => 1024.0_f64.powi(4),
        _ => return None,
    };

    let bytes = value * multiplier;
    if bytes > u64::MAX as f64 {
        return None;
    }
    Some(bytes as u64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::remote_catalog::{
        CatalogCurated, CatalogEntry, CatalogPackage, CatalogSource, CatalogVariant,
    };

    #[test]
    fn parse_size_label_bytes_supports_decimal_and_binary_units() {
        assert_eq!(parse_size_label_bytes("20GB"), Some(20_000_000_000));
        assert_eq!(parse_size_label_bytes("1.5 GB"), Some(1_500_000_000));
        assert_eq!(parse_size_label_bytes("2MiB"), Some(2 * 1024 * 1024));
        assert_eq!(parse_size_label_bytes("bad"), None);
    }

    #[test]
    fn size_lookup_matches_model_and_layer_package_aliases() {
        let lookup = ModelTargetSizeLookup::from_entries(vec![catalog_entry()]);

        assert_eq!(
            lookup.find("hf://example/source@rev-a:Q4_K_M"),
            Some(ModelSizeHint {
                model_bytes: 20_000_000_000,
                split_capable: true,
            })
        );
        assert_eq!(
            lookup.find("Model-Q4_K_M.gguf"),
            Some(ModelSizeHint {
                model_bytes: 20_000_000_000,
                split_capable: true,
            })
        );
        assert_eq!(
            lookup.find("meshllm/model-q4_k_m-layers"),
            Some(ModelSizeHint {
                model_bytes: 24_000_000_000,
                split_capable: true,
            })
        );
    }

    fn catalog_entry() -> CatalogEntry {
        CatalogEntry {
            schema_version: 1,
            source_repo: "example/source".to_string(),
            variants: HashMap::from([(
                "Model-Q4_K_M".to_string(),
                CatalogVariant {
                    source: CatalogSource {
                        repo: "example/source".to_string(),
                        revision: Some("rev-a".to_string()),
                        file: Some("nested/Model-Q4_K_M.gguf".to_string()),
                    },
                    curated: CatalogCurated {
                        name: "Example Model Q4".to_string(),
                        size: Some("20GB".to_string()),
                        description: None,
                        draft: None,
                        moe: None,
                        extra_files: Vec::new(),
                        mmproj: None,
                    },
                    packages: vec![CatalogPackage {
                        package_type: "layer-package".to_string(),
                        repo: "meshllm/model-q4_k_m-layers".to_string(),
                        layer_count: Some(32),
                        total_bytes: Some(24_000_000_000),
                    }],
                },
            )]),
        }
    }
}
