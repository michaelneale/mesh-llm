use super::formatters::{
    capabilities_json, catalog_model_capabilities, catalog_model_kind_code,
    fit_code_for_size_label, format_installed_size, huggingface_cache_dir,
    installed_model_kind_code, local_capacity_json, model_kind_code, moe_json, print_json,
    InstalledRow, JsonFormatter, ModelsFormatter, SearchFormatter,
};
use crate::models::{
    catalog, search_catalog_json_payload, search_huggingface_json_payload, ModelDetails,
    SearchArtifactFilter, SearchHit, SearchSort,
};
use crate::models::{DeleteResult as CliDeleteResult, ResolvedModel as CliResolvedModel};
use anyhow::Result;
use serde_json::{json, Value};
use std::path::Path;

fn show_payload(details: &ModelDetails, variants: Option<&[ModelDetails]>) -> Value {
    json!({
        "display_name": details.exact_ref,
        "ref": details.exact_ref,
        "type": model_kind_code(details.kind),
        "source": details.source,
        "size": details.size_label,
        "fit": details
            .size_label
            .as_deref()
            .and_then(fit_code_for_size_label),
        "description": details.description,
        "draft": details.draft,
        "capabilities": capabilities_json(details.capabilities),
        "moe": moe_json(details.moe.as_ref()),
        "download_url": details.download_url,
        "machine": local_capacity_json(),
        "variants": variants
            .unwrap_or_default()
            .iter()
            .map(|variant| {
                json!({
                    "display_name": variant.exact_ref,
                    "ref": variant.exact_ref,
                    "type": model_kind_code(variant.kind),
                    "source": variant.source,
                    "size": variant.size_label,
                    "fit": variant
                        .size_label
                        .as_deref()
                        .and_then(fit_code_for_size_label),
                    "download_url": variant.download_url,
                })
            })
            .collect::<Vec<_>>(),
    })
}

impl SearchFormatter for JsonFormatter {
    fn is_json(&self) -> bool {
        true
    }

    fn render_catalog_empty(
        &self,
        query: &str,
        filter: SearchArtifactFilter,
        sort: SearchSort,
    ) -> Result<()> {
        print_json(search_catalog_json_payload(query, filter, sort, &[], 0))
    }

    fn render_catalog_results(
        &self,
        query: &str,
        filter: SearchArtifactFilter,
        results: &[&'static catalog::CatalogModel],
        limit: usize,
        sort: SearchSort,
    ) -> Result<()> {
        print_json(search_catalog_json_payload(
            query, filter, sort, results, limit,
        ))
    }

    fn render_hf_empty(
        &self,
        query: &str,
        filter: SearchArtifactFilter,
        sort: SearchSort,
    ) -> Result<()> {
        print_json(search_huggingface_json_payload(query, filter, sort, &[]))
    }

    fn render_hf_results(
        &self,
        query: &str,
        filter: SearchArtifactFilter,
        sort: SearchSort,
        results: &[SearchHit],
    ) -> Result<()> {
        print_json(search_huggingface_json_payload(
            query, filter, sort, results,
        ))
    }
}

impl ModelsFormatter for JsonFormatter {
    fn render_recommended(&self, models: &[&'static catalog::CatalogModel]) -> Result<()> {
        let results: Vec<Value> = models
            .iter()
            .map(|model| {
                let model_capabilities = catalog_model_capabilities(model);
                json!({
                    "name": model.name,
                    "size": model.size,
                    "description": model.description,
                    "draft": model.draft,
                    "type": catalog_model_kind_code(model),
                    "ref": model.name,
                    "show": format!("mesh-llm models show {}", model.name),
                    "download": format!("mesh-llm models download {}", model.name),
                    "capabilities": capabilities_json(model_capabilities),
                    "moe": moe_json(model.moe.as_ref()),
                })
            })
            .collect();
        print_json(json!({
            "source": "catalog",
            "results": results,
        }))
    }

    fn render_installed(&self, rows: &[InstalledRow]) -> Result<()> {
        let models: Vec<Value> = rows
            .iter()
            .map(|row| {
                json!({
                    "name": row.name,
                    "type": installed_model_kind_code(&row.path),
                    "size_bytes": row.size,
                    "size": row.size.map(super::formatters::format_installed_size),
                    "mesh_managed": row.managed_by_mesh,
                    "last_used_at": row.last_used_at,
                    "capabilities": capabilities_json(row.capabilities),
                    "ref": row.model_ref,
                    "show": format!("mesh-llm models show {}", row.model_ref),
                    "download": format!("mesh-llm models download {}", row.model_ref),
                    "delete": format!("mesh-llm models delete {}", row.model_ref),
                    "path": row.path,
                    "about": row.catalog_model.map(|m| m.description.clone()),
                    "draft": row.catalog_model.and_then(|m| m.draft.clone()),
                    "moe": moe_json(row.catalog_model.and_then(|m| m.moe.as_ref())),
                })
            })
            .collect();
        print_json(json!({
            "cache_dir": huggingface_cache_dir(),
            "delete_example": rows
                .first()
                .map(|row| format!("mesh-llm models delete {}", row.model_ref)),
            "results": models,
        }))
    }

    fn render_show(&self, details: &ModelDetails, variants: Option<&[ModelDetails]>) -> Result<()> {
        print_json(show_payload(details, variants))
    }

    fn render_download(
        &self,
        model_ref: &str,
        path: &Path,
        details: Option<&ModelDetails>,
        include_draft: bool,
        draft: Option<(&str, &Path)>,
    ) -> Result<()> {
        let mut payload = json!({
            "requested_ref": model_ref,
            "path": path,
            "type": details.as_ref().map(|d| model_kind_code(d.kind)),
            "resolved_ref": details.as_ref().map(|d| d.exact_ref.clone()),
        });
        if include_draft {
            payload["draft"] = match draft {
                Some((name, draft_path)) => json!({
                    "name": name,
                    "path": draft_path,
                }),
                None => Value::Null,
            };
        }
        print_json(payload)
    }

    fn render_updates_status(&self, repo: Option<&str>, all: bool, check: bool) -> Result<()> {
        print_json(json!({
            "status": "ok",
            "mode": if check { "check" } else { "update" },
            "target": {
                "repo": repo,
                "all": all,
            },
        }))
    }

    fn render_delete_preview(&self, resolved: &CliResolvedModel) -> Result<()> {
        let file_size = std::fs::metadata(&resolved.path)
            .map(|m| m.len())
            .unwrap_or(0);
        print_json(json!({
            "display_name": resolved.display_name,
            "path": resolved.path,
            "is_exact_path": resolved.is_exact_path,
            "file_size_bytes": file_size,
            "file_size_human": format_installed_size(file_size),
            "matched_records": resolved.matched_records.iter().map(|r| json!({
                "lookup_key": r.lookup_key,
                "display_name": r.display_name,
                "last_used_at": r.last_used_at,
            })).collect::<Vec<_>>(),
            "dry_run": true,
        }))
    }

    fn render_delete_result(&self, result: &CliDeleteResult) -> Result<()> {
        print_json(json!({
            "deleted_paths": result.deleted_paths.iter().map(|p| p.to_string_lossy().to_string()).collect::<Vec<_>>(),
            "reclaimed_bytes": result.reclaimed_bytes,
            "reclaimed_bytes_human": format_installed_size(result.reclaimed_bytes),
            "removed_metadata_files": result.removed_metadata_files,
            "removed_usage_records": result.removed_usage_records,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::ModelCapabilities;

    #[test]
    fn show_payload_includes_variants_for_selected_gguf_ref() {
        let details = ModelDetails {
            display_name: "Qwen3.6-35B-A3B-BF16.gguf".to_string(),
            exact_ref: "unsloth/Qwen3.6-35B-A3B-GGUF:BF16".to_string(),
            source: "huggingface",
            kind: "🦙 GGUF",
            download_url: "https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/resolve/main/BF16/Qwen3.6-35B-A3B-BF16-00001-of-00002.gguf".to_string(),
            size_label: Some("49.9GB".to_string()),
            description: None,
            draft: None,
            capabilities: ModelCapabilities::default(),
            moe: None,
        };
        let variants = vec![
            ModelDetails {
                display_name: "Qwen3.6-35B-A3B-BF16.gguf".to_string(),
                exact_ref: "unsloth/Qwen3.6-35B-A3B-GGUF:BF16".to_string(),
                source: "huggingface",
                kind: "🦙 GGUF",
                download_url: "https://example.invalid/bf16.gguf".to_string(),
                size_label: Some("49.9GB".to_string()),
                description: None,
                draft: None,
                capabilities: ModelCapabilities::default(),
                moe: None,
            },
            ModelDetails {
                display_name: "Qwen3.6-35B-A3B-Q4_K_M.gguf".to_string(),
                exact_ref: "unsloth/Qwen3.6-35B-A3B-GGUF:Q4_K_M".to_string(),
                source: "huggingface",
                kind: "🦙 GGUF",
                download_url: "https://example.invalid/q4_k_m.gguf".to_string(),
                size_label: Some("21.3GB".to_string()),
                description: None,
                draft: None,
                capabilities: ModelCapabilities::default(),
                moe: None,
            },
        ];

        let payload = show_payload(&details, Some(&variants));
        let emitted_variants = payload["variants"].as_array().expect("variants array");

        assert_eq!(payload["ref"], "unsloth/Qwen3.6-35B-A3B-GGUF:BF16");
        assert_eq!(emitted_variants.len(), 2);
        assert_eq!(
            emitted_variants[0]["ref"],
            "unsloth/Qwen3.6-35B-A3B-GGUF:BF16"
        );
        assert_eq!(
            emitted_variants[1]["ref"],
            "unsloth/Qwen3.6-35B-A3B-GGUF:Q4_K_M"
        );
    }
}
