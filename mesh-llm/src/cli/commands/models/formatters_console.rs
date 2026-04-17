use super::formatters::{
    catalog_model_capabilities, filter_label, fit_hint_for_size_label, format_count,
    format_installed_size, format_relative_timestamp, format_source_label, huggingface_cache_dir,
    huggingface_repo_url, installed_model_kind, model_kind_code, sort_label,
    variant_selector_label, ConsoleFormatter, InstalledRow, ModelsFormatter, SearchFormatter,
};
use crate::models::{catalog, ModelDetails, SearchArtifactFilter, SearchHit, SearchSort};
use anyhow::Result;
use std::io::Write;
use tabwriter::TabWriter;

impl SearchFormatter for ConsoleFormatter {
    fn is_json(&self) -> bool {
        false
    }

    fn render_catalog_empty(
        &self,
        query: &str,
        filter: SearchArtifactFilter,
        sort: SearchSort,
    ) -> Result<()> {
        eprintln!(
            "🔎 No {} catalog models matched '{}' (sorted by {}).",
            filter_label(filter),
            query,
            sort_label(sort)
        );
        Ok(())
    }

    fn render_catalog_results(
        &self,
        query: &str,
        filter: SearchArtifactFilter,
        results: &[&'static catalog::CatalogModel],
        limit: usize,
        sort: SearchSort,
    ) -> Result<()> {
        println!(
            "📚 {} catalog matches for '{}' ({})",
            filter_label(filter),
            query,
            sort_label(sort)
        );
        if let Some(summary) = super::formatters::local_capacity_summary() {
            println!("{}", summary);
        }
        println!();
        for model in results.iter().take(limit) {
            println!("• {}  {}", model.name, model.size);
            println!("  {}", model.description);
            if let Some(fit) = fit_hint_for_size_label(&model.size) {
                println!("  {}", fit);
            }
        }
        Ok(())
    }

    fn render_hf_empty(
        &self,
        query: &str,
        filter: SearchArtifactFilter,
        sort: SearchSort,
    ) -> Result<()> {
        eprintln!(
            "🔎 No Hugging Face {} matches for '{}' (sorted by {}).",
            filter_label(filter),
            query,
            sort_label(sort)
        );
        Ok(())
    }

    fn render_hf_results(
        &self,
        query: &str,
        filter: SearchArtifactFilter,
        sort: SearchSort,
        results: &[SearchHit],
    ) -> Result<()> {
        println!(
            "🔎 Hugging Face {} matches for '{}' ({})",
            filter_label(filter),
            query,
            sort_label(sort)
        );
        if let Some(summary) = super::formatters::local_capacity_summary() {
            println!("{}", summary);
        }
        println!();
        for (index, result) in results.iter().enumerate() {
            println!("{}. 📦 {}", index + 1, result.repo_id);
            println!("   type: {}", result.kind);
            let mut stats = Vec::new();
            if let Some(size) = &result.size_label {
                stats.push(format!("size: {} 📏", size));
            }
            if let Some(downloads) = result.downloads {
                stats.push(format!("⬇️ {}", format_count(downloads)));
            }
            if let Some(likes) = result.likes {
                stats.push(format!("❤️ {}", format_count(likes)));
            }
            if !stats.is_empty() {
                println!("   {}", stats.join("  "));
            }
            let mut caps = vec!["💬 text".to_string()];
            if result.capabilities.multimodal_label().is_some() {
                caps.push("🎛️ multimodal".to_string());
            }
            if let Some(label) = result.capabilities.vision_label() {
                caps.push(format!("👁️ vision ({label})"));
            }
            if let Some(label) = result.capabilities.audio_label() {
                caps.push(format!("🔊 audio ({label})"));
            }
            if let Some(label) = result.capabilities.reasoning_label() {
                caps.push(format!("🧠 reasoning ({label})"));
            }
            if let Some(label) = result.capabilities.tool_use_label() {
                caps.push(format!("🛠️ tool use ({label})"));
            }
            println!("   capabilities: {}", caps.join("  "));
            println!("   repo: {}", huggingface_repo_url(&result.repo_id));
            println!("   ref: {}", result.exact_ref);
            println!("   show: mesh-llm models show {}", result.exact_ref);
            println!("   download: mesh-llm models download {}", result.exact_ref);
            if let Some(size) = &result.size_label {
                if let Some(fit) = fit_hint_for_size_label(size) {
                    println!("   {}", fit);
                }
            }
            if let Some(model) = result.catalog {
                println!("   ⭐ Recommended: {} ({})", model.name, model.size);
                println!("   {}", model.description);
            }
            println!();
        }
        Ok(())
    }
}

impl ModelsFormatter for ConsoleFormatter {
    fn render_recommended(&self, models: &[&'static catalog::CatalogModel]) -> Result<()> {
        println!("📚 Recommended models");
        println!();
        for model in models {
            let model_capabilities = catalog_model_capabilities(model);
            println!("• {}  {}", model.name, model.size);
            println!("  {}", model.description);
            if let Some(draft) = model.draft.as_deref() {
                println!("  🧠 Draft: {}", draft);
            }
            if let Some(label) = model_capabilities.vision_label() {
                println!("  👁️ Vision: {}", label);
            }
            if let Some(label) = model_capabilities.audio_label() {
                println!("  🔊 Audio: {}", label);
            }
            if let Some(label) = model_capabilities.reasoning_label() {
                println!("  🧠 Reasoning: {}", label);
            }
            if model.moe.is_some() {
                println!("  🧩 MoE: yes");
            }
            println!();
        }
        Ok(())
    }

    fn render_installed(&self, rows: &[InstalledRow]) -> Result<()> {
        if rows.is_empty() {
            println!("📦 No installed models found");
            println!("   HF cache: {}", huggingface_cache_dir().display());
            return Ok(());
        }

        println!("💾 Installed models");
        println!("📁 HF cache: {}", huggingface_cache_dir().display());
        println!();
        for row in rows {
            println!("📦 {}", row.name);
            println!("   type: {}", installed_model_kind(&row.path));
            if let Some(bytes) = row.size {
                println!("   size: {} 📏", format_installed_size(bytes));
            }
            println!(
                "   owner: {}",
                if row.managed_by_mesh {
                    "mesh-managed"
                } else {
                    "external"
                }
            );
            if let Some(last_used_at) = row.last_used_at.as_deref() {
                if let Some(label) = format_relative_timestamp(last_used_at) {
                    println!("   last used: {}", label);
                }
            }
            let mut caps = vec!["💬 text".to_string()];
            if row.capabilities.multimodal_label().is_some() {
                caps.push("🎛️ multimodal".to_string());
            }
            if let Some(label) = row.capabilities.vision_label() {
                caps.push(format!("👁️ vision ({label})"));
            }
            if let Some(label) = row.capabilities.audio_label() {
                caps.push(format!("🔊 audio ({label})"));
            }
            if let Some(label) = row.capabilities.reasoning_label() {
                caps.push(format!("🧠 reasoning ({label})"));
            }
            if let Some(label) = row.capabilities.tool_use_label() {
                caps.push(format!("🛠️ tool use ({label})"));
            }
            println!("   capabilities: {}", caps.join("  "));
            println!("   ref: {}", row.model_ref);
            println!("   show: mesh-llm models show {}", row.model_ref);
            println!("   download: mesh-llm models download {}", row.model_ref);
            println!("   path: {}", row.path.display());
            if let Some(model) = row.catalog_model {
                println!("   about: {}", model.description);
                if let Some(draft) = model.draft.as_deref() {
                    println!("   🧠 draft: {}", draft);
                }
                if model.moe.is_some() {
                    println!("   🧩 MoE: yes");
                }
            }
            println!();
        }
        Ok(())
    }

    fn render_show(&self, details: &ModelDetails, variants: Option<&[ModelDetails]>) -> Result<()> {
        if model_kind_code(details.kind) == "mlx" {
            println!("🔎 {}", details.exact_ref);
        } else {
            println!("🔎 {}", details.display_name);
        }
        if let Some(summary) = super::formatters::local_capacity_summary() {
            println!("{}", summary);
        }
        println!();
        println!("Ref: {}", details.exact_ref);
        println!("Type: {}", details.kind);
        println!("Source: {}", format_source_label(details.source));
        if let Some(size) = details.size_label.as_deref() {
            println!("Size: {size}");
            if let Some(fit) = fit_hint_for_size_label(size) {
                println!("Fit: {}", fit);
            }
        }
        if let Some(description) = details.description.as_deref() {
            println!("About: {description}");
        }
        if let Some(draft) = details.draft.as_deref() {
            println!("🧠 Draft: {draft}");
        }
        println!("Capabilities:");
        println!("  💬 text");
        if details.capabilities.multimodal_label().is_some() {
            println!("  🎛️ multimodal");
        }
        if let Some(label) = details.capabilities.vision_label() {
            println!("  👁️ vision ({label})");
        }
        if let Some(label) = details.capabilities.audio_label() {
            println!("  🔊 audio ({label})");
        }
        if let Some(label) = details.capabilities.reasoning_label() {
            println!("  🧠 reasoning ({label})");
        }
        if let Some(moe) = details.moe.clone() {
            println!(
                "🧩 MoE: {} experts, top-{}, min per node {}{}",
                moe.n_expert,
                moe.n_expert_used,
                moe.min_experts_per_node,
                if moe.ranking.is_empty() {
                    ", no embedded ranking".to_string()
                } else {
                    format!(", ranking {}", moe.ranking.len())
                }
            );
        }
        println!("📥 Download:");
        if model_kind_code(details.kind) == "mlx" {
            println!("   mesh-llm models download {}", details.exact_ref);
        } else {
            println!("   {}", details.download_url);
        }

        if let Some(variants) = variants {
            if !variants.is_empty() {
                println!();
                println!("Variants:");
                let mut rows = Vec::new();
                for variant in variants {
                    let size = variant.size_label.as_deref().unwrap_or("-");
                    let fit = variant
                        .size_label
                        .as_deref()
                        .and_then(fit_hint_for_size_label)
                        .unwrap_or_else(|| "-".to_string());
                    let selected = variant.exact_ref == details.exact_ref;
                    rows.push((
                        variant_selector_label(&variant.exact_ref),
                        size.to_string(),
                        fit,
                        variant.exact_ref.clone(),
                        selected,
                    ));
                }
                let mut table = TabWriter::new(Vec::new()).padding(2);
                writeln!(&mut table, "sel\tquant\tsize\tfit\tref")?;
                writeln!(&mut table, "---\t-----\t----\t---\t---")?;
                for (quant, size, fit, r#ref, selected) in rows {
                    writeln!(
                        &mut table,
                        "{}\t{}\t{}\t{}\t{}",
                        if selected { "*" } else { " " },
                        quant,
                        size,
                        fit,
                        r#ref
                    )?;
                }
                table.flush()?;
                print!("{}", String::from_utf8_lossy(&table.into_inner()?));
            }
        }
        Ok(())
    }

    fn render_download(
        &self,
        _model_ref: &str,
        path: &std::path::Path,
        details: Option<&ModelDetails>,
        _include_draft: bool,
        draft: Option<(&str, &std::path::Path)>,
    ) -> Result<()> {
        println!("✅ Downloaded model");
        if let Some(details) = details {
            println!("   type: {}", details.kind);
        }
        println!("   {}", path.display());
        if let Some((_draft_name, draft_path)) = draft {
            println!("🧠 Downloaded draft");
            println!("   {}", draft_path.display());
        }
        Ok(())
    }

    fn render_updates_status(&self, _repo: Option<&str>, _all: bool, _check: bool) -> Result<()> {
        Ok(())
    }
}
