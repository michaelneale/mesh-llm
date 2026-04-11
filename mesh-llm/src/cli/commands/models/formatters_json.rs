use super::formatters::{
    capabilities_json, catalog_model_capabilities, catalog_model_kind_code, filter_name,
    fit_code_for_size_label, huggingface_cache_dir, installed_model_kind_code, local_capacity_json,
    model_kind_code, moe_json, print_json, InstalledRow, JsonFormatter, ModelsFormatter,
    SearchFormatter,
};
use crate::models::{catalog, ModelDetails, SearchArtifactFilter, SearchHit};
use anyhow::Result;
use serde_json::{json, Value};
use std::path::Path;

impl SearchFormatter for JsonFormatter {
    fn is_json(&self) -> bool {
        true
    }

    fn render_catalog_empty(&self, query: &str, filter: SearchArtifactFilter) -> Result<()> {
        print_json(json!({
            "query": query,
            "filter": filter_name(filter),
            "source": "catalog",
            "machine": local_capacity_json(),
            "results": [],
        }))
    }

    fn render_catalog_results(
        &self,
        query: &str,
        filter: SearchArtifactFilter,
        results: &[&'static catalog::CatalogModel],
        limit: usize,
    ) -> Result<()> {
        let payload_results: Vec<Value> = results
            .iter()
            .take(limit)
            .map(|model| {
                json!({
                    "name": model.name,
                    "repo_id": model.source_repo(),
                    "type": catalog_model_kind_code(model),
                    "size": model.size,
                    "description": model.description,
                    "fit": fit_code_for_size_label(&model.size),
                    "ref": model.name,
                    "show": format!("mesh-llm models show {}", model.name),
                    "download": format!("mesh-llm models download {}", model.name),
                    "draft": model.draft,
                    "capabilities": capabilities_json(catalog_model_capabilities(model)),
                })
            })
            .collect();
        print_json(json!({
            "query": query,
            "filter": filter_name(filter),
            "source": "catalog",
            "machine": local_capacity_json(),
            "results": payload_results,
        }))
    }

    fn render_hf_empty(&self, query: &str, filter: SearchArtifactFilter) -> Result<()> {
        print_json(json!({
            "query": query,
            "filter": filter_name(filter),
            "source": "huggingface",
            "machine": local_capacity_json(),
            "results": [],
        }))
    }

    fn render_hf_results(
        &self,
        query: &str,
        filter: SearchArtifactFilter,
        results: &[SearchHit],
    ) -> Result<()> {
        let payload_results: Vec<Value> = results
            .iter()
            .map(|result| {
                json!({
                    "repo_id": result.repo_id,
                    "type": model_kind_code(result.kind),
                    "size": result.size_label,
                    "downloads": result.downloads,
                    "likes": result.likes,
                    "fit": result
                        .size_label
                        .as_deref()
                        .and_then(fit_code_for_size_label),
                    "ref": result.exact_ref,
                    "show": format!("mesh-llm models show {}", result.exact_ref),
                    "download": format!("mesh-llm models download {}", result.exact_ref),
                    "capabilities": capabilities_json(result.capabilities),
                    "catalog": result.catalog.map(|model| {
                        json!({
                            "name": model.name,
                            "size": model.size,
                            "description": model.description,
                        })
                    }),
                })
            })
            .collect();
        print_json(json!({
            "query": query,
            "filter": filter_name(filter),
            "source": "huggingface",
            "machine": local_capacity_json(),
            "results": payload_results,
        }))
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
                    "capabilities": capabilities_json(row.capabilities),
                    "ref": row.name,
                    "show": format!("mesh-llm models show {}", row.name),
                    "download": format!("mesh-llm models download {}", row.name),
                    "path": row.path,
                    "about": row.catalog_model.map(|m| m.description.clone()),
                    "draft": row.catalog_model.and_then(|m| m.draft.clone()),
                    "moe": moe_json(row.catalog_model.and_then(|m| m.moe.as_ref())),
                })
            })
            .collect();
        print_json(json!({
            "cache_dir": huggingface_cache_dir(),
            "results": models,
        }))
    }

    fn render_show(&self, details: &ModelDetails, variants: Option<&[ModelDetails]>) -> Result<()> {
        print_json(json!({
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
        }))
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
}
