use super::formatters::{
    catalog_model_capabilities, display_width, filter_label, fit_hint_for_size_label, format_count,
    format_installed_size, format_source_label, huggingface_cache_dir, installed_model_kind,
    model_kind_code, pad_left_display, pad_right_display, variant_selector_label, ConsoleFormatter,
    InstalledRow, ModelsFormatter, SearchFormatter,
};
use crate::models::{catalog, ModelDetails, SearchArtifactFilter, SearchHit};
use anyhow::Result;

impl SearchFormatter for ConsoleFormatter {
    fn is_json(&self) -> bool {
        false
    }

    fn render_catalog_empty(&self, query: &str, filter: SearchArtifactFilter) -> Result<()> {
        eprintln!(
            "🔎 No {} catalog models matched '{}'.",
            filter_label(filter),
            query
        );
        Ok(())
    }

    fn render_catalog_results(
        &self,
        query: &str,
        filter: SearchArtifactFilter,
        results: &[&'static catalog::CatalogModel],
        limit: usize,
    ) -> Result<()> {
        println!(
            "📚 {} catalog matches for '{}'",
            filter_label(filter),
            query
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

    fn render_hf_empty(&self, query: &str, filter: SearchArtifactFilter) -> Result<()> {
        eprintln!(
            "🔎 No Hugging Face {} matches for '{}'.",
            filter_label(filter),
            query
        );
        Ok(())
    }

    fn render_hf_results(
        &self,
        query: &str,
        filter: SearchArtifactFilter,
        results: &[SearchHit],
    ) -> Result<()> {
        println!(
            "🔎 Hugging Face {} matches for '{}'",
            filter_label(filter),
            query
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
            println!("   ref: {}", row.name);
            println!("   show: mesh-llm models show {}", row.name);
            println!("   download: mesh-llm models download {}", row.name);
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
                let sel_width = 3usize;
                let quant_width = rows
                    .iter()
                    .map(|(quant, _, _, _, _)| display_width(quant))
                    .max()
                    .unwrap_or(5)
                    .max(display_width("quant"));
                let size_width = rows
                    .iter()
                    .map(|(_, size, _, _, _)| display_width(size))
                    .max()
                    .unwrap_or(4)
                    .max(display_width("size"));
                let fit_width = rows
                    .iter()
                    .map(|(_, _, fit, _, _)| display_width(fit))
                    .max()
                    .unwrap_or(3)
                    .max(display_width("fit"));
                println!(
                    "{}  {}  {}  {}  ref",
                    pad_right_display("sel", sel_width),
                    pad_right_display("quant", quant_width),
                    pad_left_display("size", size_width),
                    pad_right_display("fit", fit_width)
                );
                println!(
                    "{}  {}  {}  {}  ---",
                    "-".repeat(sel_width),
                    "-".repeat(quant_width),
                    "-".repeat(size_width),
                    "-".repeat(fit_width)
                );
                for (quant, size, fit, r#ref, selected) in rows {
                    println!(
                        "{}  {}  {}  {}  {}",
                        pad_right_display(if selected { "*" } else { " " }, sel_width),
                        pad_right_display(&quant, quant_width),
                        pad_left_display(&size, size_width),
                        pad_right_display(&fit, fit_width),
                        r#ref
                    );
                }
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
