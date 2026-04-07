use crate::cli::models::ModelsCommand;
use crate::models::{
    capabilities, catalog, download_exact_ref_with_profile, find_catalog_model_exact,
    huggingface_hub_cache_dir, inspect_repo_ref, installed_model_capabilities,
    scan_installed_models, search_catalog_models, search_huggingface, show_exact_model,
    CapabilityProfile, RepoVariantInspection, SearchProgress,
};
use crate::system::hardware;
use anyhow::{anyhow, Result};
use std::io::Write;

pub async fn run_model_search(
    query: &[String],
    catalog_only: bool,
    limit: usize,
    profile: CapabilityProfile,
) -> Result<()> {
    let query = query.join(" ");
    if catalog_only {
        let results = search_catalog_models(&query);
        if results.is_empty() {
            eprintln!("🔎 No catalog models matched '{query}'.");
            return Ok(());
        }
        println!("📚 Catalog matches for '{query}'");
        if let Some(summary) = local_capacity_summary() {
            println!("{}", summary);
        }
        println!();
        for model in results.into_iter().take(limit) {
            println!("• {}  {}", model.name, model.size);
            println!("  {}", model.description);
            if let Some(fit) = fit_hint_for_size_label(&model.size) {
                println!("  {}", fit);
            }
        }
        return Ok(());
    }

    eprintln!("🔎 Searching Hugging Face GGUF repos for '{query}'...");
    let mut announced_repo_scan = false;
    let results = search_huggingface(&query, limit, |progress| match progress {
        SearchProgress::SearchingHub => {}
        SearchProgress::InspectingRepos { completed, total } => {
            if total == 0 {
                return;
            }
            if !announced_repo_scan {
                announced_repo_scan = true;
                eprintln!("   Inspecting {total} candidate repos...");
            }
            if completed == 0 {
                return;
            }
            eprint!("\r   Inspected {completed}/{total} candidate repos...");
            let _ = std::io::stderr().flush();
            if completed == total {
                eprintln!();
            }
        }
    })
    .await?;
    if results.is_empty() {
        eprintln!("🔎 No Hugging Face GGUF matches for '{query}'.");
        return Ok(());
    }

    println!("🔎 Hugging Face GGUF matches for '{query}'");
    if let Some(summary) = local_capacity_summary() {
        println!("{}", summary);
    }
    if !profile.is_text_only() {
        println!("🎯 capability: {}", profile.as_label());
    }
    println!();
    for (index, result) in results.iter().enumerate() {
        println!("{}. 📦 {}", index + 1, result.repo_id);
        println!("   🔗 {}", result.repo_url);
        let mut stats = Vec::new();
        if result.gguf_files > 0 {
            stats.push(format!(
                "📦 {} variants",
                format_count(result.gguf_files as u64)
            ));
        }
        if let Some(size) = &result.size_label {
            stats.push(format!("📏 {}", size));
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
        if let Some(description) = result.description.as_deref() {
            println!("   📝 {}", trim_ellipsis(description, 88));
        }
        if let Some(notice) = &result.metadata_notice {
            let _ = notice;
            println!(
                "   🟡 gated: additional info and downloads are unavailable until terms are accepted"
            );
            println!("   🔎 mesh-llm models show {}", result.repo_id);
            println!();
            continue;
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
        println!("   {}", caps.join("  "));
        if let Some(recommended_ref) = result.recommended_ref.as_deref() {
            println!("   ✅ recommended for this machine: {recommended_ref}");
            if let Some(highest_quality_ref) = result.highest_quality_ref.as_deref() {
                println!("   🏆 highest quality: {highest_quality_ref}");
            }
            println!("   🔎 mesh-llm models show {}", result.repo_id);
            println!("   ⬇️ mesh-llm models download {}", recommended_ref);
        }
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

pub fn run_model_recommended() {
    println!("📚 Recommended models");
    println!();
    for model in catalog::MODEL_CATALOG.iter() {
        let model_capabilities = capabilities::infer_catalog_capabilities(model);
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
}

pub fn run_model_installed() {
    let installed = scan_installed_models();
    if installed.is_empty() {
        println!("📦 No installed models found");
        println!("   HF cache: {}", huggingface_hub_cache_dir().display());
        return;
    }

    println!("💾 Installed models");
    println!("📁 HF cache: {}", huggingface_hub_cache_dir().display());
    println!();
    for name in installed {
        let path = crate::models::find_model_path(&name);
        let size = std::fs::metadata(&path).map(|meta| meta.len()).ok();
        let catalog_model = find_catalog_model_exact(&name);
        let model_capabilities = installed_model_capabilities(&name);

        match size {
            Some(bytes) => println!("• {}  {}", name, format_installed_size(bytes)),
            None => println!("• {}", name),
        }
        println!("  🤗 HF cache");
        println!("  {}", path.display());
        if let Some(model) = catalog_model {
            println!("  {}", model.description);
            if let Some(draft) = model.draft.as_deref() {
                println!("  🧠 Draft: {}", draft);
            }
            if model.moe.is_some() {
                println!("  🧩 MoE: yes");
            }
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
        println!();
    }
}

pub async fn run_model_show(model_ref: &str, profile: CapabilityProfile) -> Result<()> {
    if model_ref.matches('/').count() <= 1
        && !model_ref.starts_with("http://")
        && !model_ref.starts_with("https://")
    {
        if let Some(repo) = inspect_repo_ref(model_ref, profile).await? {
            print_repo_show(repo);
            return Ok(());
        }
    }

    let details = show_exact_model(model_ref).await?;
    if details.source == "huggingface" && !details.resolved_files.is_empty() {
        let repo = details
            .exact_ref
            .split('/')
            .take(2)
            .collect::<Vec<_>>()
            .join("/");
        println!("📦 {}", repo);
        println!("🔗 https://huggingface.co/{}", repo);
        println!("🔗 ref: {}", details.exact_ref);
        if let Some(quant) = details.quant.as_deref() {
            println!("⚖️ quant: {}", quant);
        }
        if let Some(total) = details.total_size_bytes {
            println!("📏 total size: {}", format_installed_size(total));
        } else if let Some(size) = details.size_label.as_deref() {
            println!("📏 total size: {}", size);
        }
        if let Some(fit) = details.fit {
            println!("💻 fit: {}", if fit { "✅" } else { "❌" });
        }
        println!();
        println!("Resolved files:");
        for file in &details.resolved_files {
            println!("- {}", file);
        }
        return Ok(());
    }

    println!("🔎 {}", details.display_name);
    if let Some(summary) = local_capacity_summary() {
        println!("{}", summary);
    }
    println!();
    println!("Ref: {}", details.exact_ref);
    println!("Source: {}", format_source_label(details.source));
    if let Some(size) = details.size_label {
        println!("Size: {size}");
        if let Some(fit) = fit_hint_for_size_label(&size) {
            println!("Fit: {}", fit);
        }
    }
    if let Some(description) = details.description {
        println!("About: {description}");
    }
    if let Some(draft) = details.draft {
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
    if let Some(moe) = details.moe {
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
    println!("   {}", details.download_url);
    Ok(())
}

fn print_repo_show(repo: crate::models::RepoInspection) {
    println!("📦 {}", repo.repo);
    println!("🔗 https://huggingface.co/{}", repo.repo);
    if let Some(description) = repo.description.as_deref() {
        println!("📝 {}", trim_ellipsis(description, 120));
    }
    let mut stats = Vec::new();
    stats.push(format!(
        "📦 {} variants",
        format_count(repo.variant_count as u64)
    ));
    if let Some(downloads) = repo.downloads {
        stats.push(format!("⬇️ {}", format_count(downloads)));
    }
    if let Some(likes) = repo.likes {
        stats.push(format!("❤️ {}", format_count(likes)));
    }
    println!("{}", stats.join("  "));
    if let Some(summary) = local_capacity_summary() {
        println!("{}", summary);
    }
    println!();
    println!("✅ recommended for this machine");
    println!("   🔗 {}", repo.recommended_ref);
    println!("   ⬇️ mesh-llm models download {}", repo.recommended_ref);
    if let Some(fit) = repo.recommended_fit {
        println!("   💻 fit: {}", if fit { "✅" } else { "❌" });
    }
    if let Some(size) = repo.recommended_size_bytes {
        println!("   📏 size: {}", format_installed_size(size));
    }
    println!();
    println!("🏆 highest quality");
    println!("   🔗 {}", repo.highest_quality_ref);
    println!(
        "   ⬇️ mesh-llm models download {}",
        repo.highest_quality_ref
    );
    if let Some(size) = repo.highest_quality_size_bytes {
        println!("   📏 size: {}", format_installed_size(size));
    }
    println!();
    println!("🧾 Other variants (Found {})", repo.variant_count);
    println!("#  ⚖️ quant      🎯 cap        📏 size     💻 fit  🔗 ref");
    for (index, variant) in repo.variants.iter().enumerate() {
        println!("{}", format_variant_row(index + 1, variant));
    }
}

pub async fn run_model_download(
    model_ref: &str,
    include_draft: bool,
    profile: CapabilityProfile,
) -> Result<()> {
    let path = download_exact_ref_with_profile(model_ref, profile).await?;
    println!("✅ Downloaded model");
    println!("   {}", path.display());

    if !include_draft {
        return Ok(());
    }

    let Some(details) = show_exact_model(model_ref).await.ok() else {
        return Ok(());
    };
    let Some(draft) = details.draft else {
        eprintln!("⚠ No draft model available for {}", details.display_name);
        return Ok(());
    };
    let draft_model = find_catalog_model_exact(&draft)
        .ok_or_else(|| anyhow!("Draft model '{}' not found in catalog", draft))?;
    let draft_path = catalog::download_model(draft_model).await?;
    println!("🧠 Downloaded draft");
    println!("   {}", draft_path.display());
    Ok(())
}

fn format_variant_row(index: usize, variant: &RepoVariantInspection) -> String {
    let size = variant
        .size_bytes
        .map(format_installed_size)
        .unwrap_or_else(|| "?".to_string());
    let fit = match variant.fit {
        Some(true) => "✅",
        Some(false) => "❌",
        None => "❔",
    };
    format!(
        "{:<2} {:<11} {:<12} {:<9} {:<5} {}",
        index, variant.quant, variant.capability, size, fit, variant.reference
    )
}

fn capability_profile_from_flags(
    vision: bool,
    audio: bool,
    multimodal: bool,
) -> Result<CapabilityProfile> {
    Ok(CapabilityProfile {
        vision,
        audio,
        multimodal,
    })
}

fn format_installed_size(bytes: u64) -> String {
    if bytes >= 1_000_000_000 {
        format!("{:.1}GB", bytes as f64 / 1e9)
    } else if bytes >= 1_000_000 {
        format!("{:.0}MB", bytes as f64 / 1e6)
    } else {
        format!("{}B", bytes)
    }
}

fn format_count(value: u64) -> String {
    let text = value.to_string();
    let mut out = String::with_capacity(text.len() + text.len() / 3);
    for (index, ch) in text.chars().enumerate() {
        if index > 0 && (text.len() - index) % 3 == 0 {
            out.push(',');
        }
        out.push(ch);
    }
    out
}

fn format_source_label(source: &str) -> &'static str {
    match source {
        "catalog" => "Catalog",
        "huggingface" => "Hugging Face",
        "url" => "Direct URL",
        _ => "Unknown",
    }
}

fn trim_ellipsis(text: &str, max_chars: usize) -> String {
    if max_chars == 0 {
        return String::new();
    }
    let mut out = String::new();
    for (index, ch) in text.chars().enumerate() {
        if index >= max_chars {
            out.push_str("...");
            return out;
        }
        out.push(ch);
    }
    out
}

fn local_capacity_summary() -> Option<String> {
    let vram_gb = hardware::survey().vram_bytes as f64 / 1e9;
    if vram_gb <= 0.0 {
        None
    } else {
        Some(format!("🖥️ This machine: ~{vram_gb:.1}GB available"))
    }
}

fn fit_hint_for_size_label(size_label: &str) -> Option<String> {
    let model_gb = catalog::parse_size_gb(size_label);
    let vram_gb = hardware::survey().vram_bytes as f64 / 1e9;
    if model_gb <= 0.0 || vram_gb <= 0.0 {
        return None;
    }

    let hint = if model_gb <= vram_gb * 0.6 {
        "✅ likely comfortable here"
    } else if model_gb <= vram_gb * 0.9 {
        "⚠️ likely fits, but tight"
    } else if model_gb <= vram_gb * 1.1 {
        "🟡 may load, but expect tradeoffs"
    } else {
        "⛔ likely too large for local serve"
    };
    Some(hint.to_string())
}

pub async fn dispatch_models_command(command: &ModelsCommand) -> Result<()> {
    match command {
        ModelsCommand::Recommended | ModelsCommand::List => run_model_recommended(),
        ModelsCommand::Installed => run_model_installed(),
        ModelsCommand::Search {
            query,
            catalog,
            limit,
            text: _,
            vision,
            audio,
            multimodal,
        } => {
            let profile = capability_profile_from_flags(*vision, *audio, *multimodal)?;
            run_model_search(query, *catalog, *limit, profile).await?
        }
        ModelsCommand::Show {
            model,
            text: _,
            vision,
            audio,
            multimodal,
        } => {
            let profile = capability_profile_from_flags(*vision, *audio, *multimodal)?;
            run_model_show(model, profile).await?
        }
        ModelsCommand::Download {
            model,
            draft,
            text: _,
            vision,
            audio,
            multimodal,
        } => {
            let profile = capability_profile_from_flags(*vision, *audio, *multimodal)?;
            run_model_download(model, *draft, profile).await?
        }
        ModelsCommand::Updates { repo, all, check } => {
            crate::models::run_update(repo.as_deref(), *all, *check)?
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{capability_profile_from_flags, dispatch_models_command, format_variant_row};
    use crate::cli::models::ModelsCommand;
    use crate::models::{CapabilityProfile, DownloadExactRefOverrideGuard, RepoVariantInspection};
    use std::sync::{Arc, Mutex};

    #[test]
    fn capability_flags_select_profile() {
        assert_eq!(
            capability_profile_from_flags(true, false, false).unwrap(),
            CapabilityProfile {
                vision: true,
                ..Default::default()
            }
        );
        assert_eq!(
            capability_profile_from_flags(false, true, false).unwrap(),
            CapabilityProfile {
                audio: true,
                ..Default::default()
            }
        );
        assert_eq!(
            capability_profile_from_flags(false, false, true).unwrap(),
            CapabilityProfile {
                multimodal: true,
                ..Default::default()
            }
        );
        assert_eq!(
            capability_profile_from_flags(false, false, false).unwrap(),
            CapabilityProfile::default()
        );
    }

    #[test]
    fn capability_flags_support_intersection() {
        let profile = capability_profile_from_flags(true, true, false).unwrap();
        assert!(profile.vision);
        assert!(profile.audio);
        assert!(!profile.multimodal);
    }

    #[test]
    fn variant_row_is_pasteable_and_annotated() {
        let row = format_variant_row(
            3,
            &RepoVariantInspection {
                reference: "unsloth/MiniMax-M2-GGUF/MiniMax-M2-Q4_K_M".to_string(),
                quant: "Q4_K_M".to_string(),
                capability: "text".to_string(),
                size_bytes: Some(128_800_000_000),
                fit: Some(false),
            },
        );
        assert!(row.starts_with("3"));
        assert!(row.contains("unsloth/MiniMax-M2-GGUF/MiniMax-M2-Q4_K_M"));
        assert!(row.contains("Q4_K_M"));
        assert!(row.contains("text"));
        assert!(row.contains("128.8GB"));
        assert!(row.contains("❌"));
    }

    #[tokio::test]
    async fn dispatch_download_preserves_shorthand_stem_and_profile() {
        let captured = Arc::new(Mutex::new(Vec::<(String, CapabilityProfile)>::new()));
        let seen = captured.clone();
        let _guard = DownloadExactRefOverrideGuard::set(Arc::new(move |input, profile| {
            seen.lock().unwrap().push((input.to_string(), profile));
            Ok(std::env::temp_dir().join("mesh-llm-test.gguf"))
        }));

        let command = ModelsCommand::Download {
            model: "anikifoss/MiniMax-M2-HQ4_K/MiniMax-M2-HQ4_K".to_string(),
            draft: false,
            text: false,
            vision: true,
            audio: false,
            multimodal: false,
        };

        dispatch_models_command(&command).await.unwrap();

        let captured = captured.lock().unwrap();
        assert_eq!(captured.len(), 1);
        assert_eq!(captured[0].0, "anikifoss/MiniMax-M2-HQ4_K/MiniMax-M2-HQ4_K");
        assert_eq!(
            captured[0].1,
            CapabilityProfile {
                vision: true,
                ..Default::default()
            }
        );
    }
}
