use super::formatters::{
    catalog_model_capabilities, filter_label, fit_hint_for_size_label, format_count,
    format_installed_size, format_relative_timestamp, format_source_label, huggingface_cache_dir,
    huggingface_repo_url, installed_model_kind, model_kind_code, sort_label,
    variant_selector_label, ConsoleFormatter, InstalledRow, ModelsFormatter, SearchFormatter,
};
use crate::models::{
    remote_catalog, remote_catalog_model_draft_ref, remote_catalog_model_ref,
    DeleteResult as CliDeleteResult, ModelDetails, ResolvedModel as CliResolvedModel,
    SearchArtifactFilter, SearchHit, SearchSort,
};
use anyhow::Result;
use std::fmt::Write as FmtWrite;
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
        results: &[remote_catalog::RemoteCatalogModel],
        limit: usize,
        sort: SearchSort,
    ) -> Result<()> {
        let mut output = String::new();
        writeln!(
            &mut output,
            "📚 {} catalog matches for '{}' ({})",
            filter_label(filter),
            query,
            sort_label(sort)
        )?;
        if let Some(summary) = super::formatters::local_capacity_summary() {
            writeln!(&mut output, "{}", summary)?;
        }
        writeln!(&mut output)?;
        for model in results.iter().take(limit) {
            let model_ref = remote_catalog_model_ref(model);
            let size = model.size.as_deref().unwrap_or("unknown size");
            writeln!(&mut output, "• {}  {}", model.name, size)?;
            writeln!(&mut output, "  ref: {}", model_ref)?;
            if let Some(description) = model.description.as_deref() {
                writeln!(&mut output, "  {}", description)?;
            }
            if let Some(size) = model.size.as_deref() {
                if let Some(fit) = fit_hint_for_size_label(size) {
                    writeln!(&mut output, "  {}", fit)?;
                }
            }
            writeln!(&mut output)?;
        }
        crate::cli::pager::print_or_page(&output)
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
        let mut output = String::new();
        writeln!(
            &mut output,
            "🔎 Hugging Face {} matches for '{}' ({})",
            filter_label(filter),
            query,
            sort_label(sort)
        )?;
        if let Some(summary) = super::formatters::local_capacity_summary() {
            writeln!(&mut output, "{}", summary)?;
        }
        writeln!(&mut output)?;
        for (index, result) in results.iter().enumerate() {
            writeln!(&mut output, "{}. 📦 {}", index + 1, result.repo_id)?;
            writeln!(&mut output, "   type: {}", result.kind)?;
            if let Some(variant_count) = result.variant_count {
                writeln!(&mut output, "   🧬 variants: {} available", variant_count)?;
            }
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
                writeln!(&mut output, "   {}", stats.join("  "))?;
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
            writeln!(&mut output, "   capabilities: {}", caps.join("  "))?;
            writeln!(
                &mut output,
                "   repo: {}",
                huggingface_repo_url(&result.repo_id)
            )?;
            writeln!(&mut output, "   ref: {}", result.exact_ref)?;
            writeln!(
                &mut output,
                "   show: mesh-llm models show {}",
                result.exact_ref
            )?;
            writeln!(
                &mut output,
                "   download: mesh-llm models download {}",
                result.exact_ref
            )?;
            if let Some(size) = &result.size_label {
                if let Some(fit) = fit_hint_for_size_label(size) {
                    writeln!(&mut output, "   {}", fit)?;
                }
            }
            if let Some(model) = result.catalog.as_ref() {
                match model.size.as_deref() {
                    Some(size) => {
                        writeln!(&mut output, "   ⭐ Recommended: {} ({})", model.name, size)?
                    }
                    None => writeln!(&mut output, "   ⭐ Recommended: {}", model.name)?,
                }
                if let Some(description) = model.description.as_deref() {
                    writeln!(&mut output, "   {}", description)?;
                }
            }
            writeln!(&mut output)?;
        }
        crate::cli::pager::print_or_page(&output)
    }
}

impl ModelsFormatter for ConsoleFormatter {
    fn render_recommended(&self, models: &[remote_catalog::RemoteCatalogModel]) -> Result<()> {
        let mut output = String::new();
        writeln!(&mut output, "📚 Recommended models")?;
        writeln!(&mut output)?;
        for model in models {
            let model_capabilities = catalog_model_capabilities(model);
            let model_ref = remote_catalog_model_ref(model);
            let size = model.size.as_deref().unwrap_or("unknown size");
            writeln!(&mut output, "• {}  {}", model.name, size)?;
            writeln!(&mut output, "  ref: {}", model_ref)?;
            if let Some(description) = model.description.as_deref() {
                writeln!(&mut output, "  {}", description)?;
            }
            if let Some(draft) = remote_catalog_model_draft_ref(model) {
                writeln!(&mut output, "  🧠 Draft: {}", draft)?;
            }
            if let Some(label) = model_capabilities.vision_label() {
                writeln!(&mut output, "  👁️ Vision: {}", label)?;
            }
            if let Some(label) = model_capabilities.audio_label() {
                writeln!(&mut output, "  🔊 Audio: {}", label)?;
            }
            if let Some(label) = model_capabilities.reasoning_label() {
                writeln!(&mut output, "  🧠 Reasoning: {}", label)?;
            }
            writeln!(&mut output)?;
        }
        crate::cli::pager::print_or_page(&output)
    }

    fn render_installed(&self, rows: &[InstalledRow]) -> Result<()> {
        if rows.is_empty() {
            println!("📦 No installed models found");
            println!("   HF cache: {}", huggingface_cache_dir().display());
            return Ok(());
        }

        let mut output = String::new();
        writeln!(&mut output, "💾 Installed models")?;
        writeln!(
            &mut output,
            "📁 HF cache: {}",
            huggingface_cache_dir().display()
        )?;
        writeln!(&mut output)?;
        writeln!(&mut output, "🗑️ Delete example: {}", rows[0].delete_command)?;
        writeln!(&mut output)?;
        for row in rows {
            writeln!(&mut output, "📦 {}", row.name)?;
            writeln!(&mut output, "   type: {}", installed_model_kind(&row.path))?;
            if let Some(layer_count) = row.layer_count {
                writeln!(&mut output, "   layers: {} 🧩", layer_count)?;
            }
            if let Some(bytes) = row.size {
                writeln!(&mut output, "   size: {} 📏", format_installed_size(bytes))?;
            }
            writeln!(
                &mut output,
                "   owner: {}",
                if row.managed_by_mesh {
                    "mesh-managed"
                } else {
                    "external"
                }
            )?;
            if let Some(last_used_at) = row.last_used_at.as_deref() {
                if let Some(label) = format_relative_timestamp(last_used_at) {
                    writeln!(&mut output, "   last used: {}", label)?;
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
            writeln!(&mut output, "   capabilities: {}", caps.join("  "))?;
            writeln!(&mut output, "   ref: {}", row.model_ref)?;
            if let Some(command) = row.show_command.as_deref() {
                writeln!(&mut output, "   show: {command}")?;
            } else {
                writeln!(&mut output, "   show: not available for layered packages")?;
            }
            if let Some(command) = row.download_command.as_deref() {
                writeln!(&mut output, "   download: {command}")?;
            } else {
                writeln!(
                    &mut output,
                    "   download: not available for layered packages"
                )?;
            }
            writeln!(&mut output, "   delete: {}", row.delete_command)?;
            writeln!(&mut output, "   path: {}", row.path.display())?;
            if let Some(model) = row.catalog_model.as_ref() {
                if let Some(description) = model.description.as_deref() {
                    writeln!(&mut output, "   about: {}", description)?;
                }
                if let Some(draft) = model.draft.as_deref() {
                    writeln!(&mut output, "   🧠 draft: {}", draft)?;
                }
            }
            writeln!(&mut output)?;
        }
        crate::cli::pager::print_or_page(&output)
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

    fn render_delete_preview(&self, resolved: &CliResolvedModel) -> Result<()> {
        println!("🗑️ Model delete preview");
        println!();
        println!("Name: {}", resolved.display_name);
        if resolved.paths.len() > 1 {
            println!("Paths ({}):", resolved.paths.len());
            for path in &resolved.paths {
                println!("  {}", path.display());
            }
        } else {
            println!("Path: {}", resolved.path.display());
        }
        println!("Mode: installed model ref resolution");
        let file_size = resolved
            .paths
            .iter()
            .map(|path| std::fs::metadata(path).map(|m| m.len()).unwrap_or(0))
            .sum();
        println!("Size: {}", format_installed_size(file_size));
        if resolved.derived_stage_paths.is_empty() {
            println!("Derived stage cache files: 0");
        } else {
            println!(
                "Derived stage cache files ({}):",
                resolved.derived_stage_paths.len()
            );
            for path in &resolved.derived_stage_paths {
                println!("  {}", path.display());
            }
        }
        if !resolved.matched_records.is_empty() {
            println!();
            println!("{} usage record(s) found:", resolved.matched_records.len());
            for record in &resolved.matched_records {
                println!(
                    "  - {} (last used: {})",
                    record.lookup_key, record.last_used_at
                );
            }
        }
        println!();
        println!("To confirm deletion, run with --yes flag.");
        Ok(())
    }

    fn render_delete_result(&self, result: &CliDeleteResult) -> Result<()> {
        println!("✅ Model deleted successfully");
        println!();
        println!("Deleted paths:");
        for p in &result.deleted_paths {
            println!("  {}", p.display());
        }
        println!();
        println!(
            "Reclaimed: {}",
            format_installed_size(result.reclaimed_bytes)
        );
        println!("Metadata files removed: {}", result.removed_metadata_files);
        println!("Usage records purged: {}", result.removed_usage_records);
        println!(
            "Derived stage cache files removed: {}",
            result.removed_derived_cache_files
        );
        Ok(())
    }
}
