use crate::cli::models::ModelsCommand;
use crate::models::{
    capabilities, catalog, catalog_model_kind_label, download_exact_ref, find_catalog_model_exact,
    huggingface_hub_cache_dir, legacy_models_dir, legacy_models_present,
    path_is_in_legacy_models_dir, scan_installed_model_entries, search_catalog_models,
    search_huggingface, show_exact_model, InstalledModelEntry, InstalledModelKind,
    ResolveFormatPreference, SearchProgress,
};
use crate::system::hardware;
use anyhow::{anyhow, Result};
use std::io::Write;

pub async fn run_model_search(query: &[String], catalog_only: bool, limit: usize) -> Result<()> {
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
        for (index, model) in results.into_iter().take(limit).enumerate() {
            println!("{}. 📦 {}  {}", index + 1, model.name, model.size);
            println!("   type: {}", catalog_model_kind_label(model));
            println!("   {}", model.description);
            if let Some(fit) = fit_hint_for_size_label(&model.size) {
                println!("   {}", fit);
            }
            println!();
        }
        return Ok(());
    }

    eprintln!("🔎 Searching Hugging Face model repos for '{query}'...");
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
        eprintln!("🔎 No Hugging Face model matches for '{query}'.");
        return Ok(());
    }

    println!("🔎 Hugging Face matches for '{query}'");
    if let Some(summary) = local_capacity_summary() {
        println!("{}", summary);
    }
    println!();
    for (index, result) in results.iter().enumerate() {
        println!("{}. 📦 {}", index + 1, result.file);
        println!("   repo: {}", result.repo_id);
        println!("   type: {}", result.kind);
        let mut stats = Vec::new();
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
        let mut caps = vec!["💬 text".to_string()];
        if let Some(label) = result.capabilities.vision_label() {
            caps.push(format!("👁️ vision ({label})"));
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

pub fn run_model_recommended() {
    println!("📚 Recommended models");
    println!();
    for (index, model) in catalog::MODEL_CATALOG.iter().enumerate() {
        let model_capabilities = capabilities::infer_catalog_capabilities(model);
        println!("{}. 📦 {}  {}", index + 1, model.name, model.size);
        println!("   type: {}", catalog_model_kind_label(model));
        println!("   {}", model.description);
        if let Some(draft) = model.draft.as_deref() {
            println!("   🧠 Draft: {}", draft);
        }
        if let Some(label) = model_capabilities.vision_label() {
            println!("   👁️ Vision: {}", label);
        }
        if let Some(label) = model_capabilities.reasoning_label() {
            println!("   🧠 Reasoning: {}", label);
        }
        if model.moe.is_some() {
            println!("   🧩 MoE: yes");
        }
        println!();
    }
}

pub fn run_model_installed() {
    let installed = scan_installed_model_entries();
    if installed.is_empty() {
        println!("📦 No installed models found");
        println!("   HF cache: {}", huggingface_hub_cache_dir().display());
        let legacy = legacy_models_dir();
        if legacy.exists() {
            println!("   legacy: {}", legacy.display());
        }
        return;
    }

    println!("💾 Installed models");
    println!("📁 HF cache: {}", huggingface_hub_cache_dir().display());
    if legacy_models_present() {
        println!(
            "⚠️ Legacy storage detected: {}",
            legacy_models_dir().display()
        );
    }
    println!();
    for (index, entry) in installed.iter().enumerate() {
        let display_name = installed_entry_display_name(entry);
        let size = installed_entry_size_bytes(entry);
        println!(
            "{}. 📦 {}",
            index + 1,
            match size {
                Some(bytes) => format!("{display_name}  {}", format_installed_size(bytes)),
                None => display_name,
            }
        );
        println!("   type: {}", installed_entry_type_label(entry));
        println!("   source: {}", installed_entry_source_label(entry));
        println!("   path: {}", entry.path.display());

        let capabilities = capabilities::infer_local_model_capabilities(
            &entry.name,
            &entry.path,
            installed_catalog_model(entry),
        );
        let mut caps = vec!["💬 text".to_string()];
        if let Some(label) = capabilities.vision_label() {
            caps.push(format!("👁️ vision ({label})"));
        }
        if let Some(label) = capabilities.reasoning_label() {
            caps.push(format!("🧠 reasoning ({label})"));
        }
        if let Some(label) = capabilities.tool_use_label() {
            caps.push(format!("🛠️ tool use ({label})"));
        }
        if capabilities.moe {
            caps.push("🧩 moe".to_string());
        }
        println!("   capabilities: {}", caps.join("  "));

        if let Some(exact_ref) = installed_entry_exact_ref(entry) {
            println!("   ref: {}", exact_ref);
            println!("   show: mesh-llm models show {}", exact_ref);
        }

        if let Some(run_command) = installed_entry_run_command(entry) {
            println!("   run: {}", run_command);
        }

        if let Some(model) = installed_catalog_model(entry) {
            println!("   ⭐ Recommended: {} ({})", model.name, model.size);
            println!("   {}", model.description);
        }
        println!();
    }
}

pub async fn run_model_show(model_ref: &str) -> Result<()> {
    let details = show_exact_model(model_ref).await?;
    println!("🔎 {}", details.display_name);
    if let Some(summary) = local_capacity_summary() {
        println!("{}", summary);
    }
    println!();
    println!("Ref: {}", details.exact_ref);
    println!("Type: {}", details.kind);
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
    if let Some(label) = details.capabilities.vision_label() {
        println!("  👁️ vision ({label})");
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

pub async fn run_model_download(
    model_ref: &str,
    preference: ResolveFormatPreference,
    include_draft: bool,
) -> Result<()> {
    let path = download_exact_ref(
        model_ref,
        preference,
        "mesh-llm models download",
        crate::models::MlxSelectionPolicy::AllowImplicit,
    )
    .await?;
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

fn format_installed_size(bytes: u64) -> String {
    if bytes >= 1_000_000_000 {
        format!("{:.1}GB", bytes as f64 / 1e9)
    } else if bytes >= 1_000_000 {
        format!("{:.0}MB", bytes as f64 / 1e6)
    } else {
        format!("{}B", bytes)
    }
}

fn installed_entry_type_label(entry: &InstalledModelEntry) -> &'static str {
    match entry.kind {
        InstalledModelKind::Gguf => "🦙 gguf",
        InstalledModelKind::Mlx => "🍎 mlx",
    }
}

fn installed_entry_source_label(entry: &InstalledModelEntry) -> &'static str {
    if path_is_in_legacy_models_dir(entry.path.as_path()) {
        "⚠️ legacy"
    } else {
        "🤗 HF cache"
    }
}

fn installed_entry_size_bytes(entry: &InstalledModelEntry) -> Option<u64> {
    match entry.kind {
        InstalledModelKind::Gguf => std::fs::metadata(&entry.path).map(|meta| meta.len()).ok(),
        InstalledModelKind::Mlx => Some(dir_size_bytes(entry.path.as_path())),
    }
}

fn dir_size_bytes(root: &std::path::Path) -> u64 {
    let mut total = 0u64;
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let Ok(file_type) = entry.file_type() else {
                continue;
            };
            if file_type.is_dir() {
                stack.push(path);
            } else if file_type.is_file() || file_type.is_symlink() {
                total =
                    total.saturating_add(std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0));
            }
        }
    }
    total
}

fn installed_entry_exact_ref(entry: &InstalledModelEntry) -> Option<String> {
    match entry.kind {
        InstalledModelKind::Gguf => crate::models::huggingface_identity_for_path(&entry.path)
            .map(|identity| identity.canonical_ref),
        InstalledModelKind::Mlx => {
            let identity =
                crate::models::huggingface_identity_for_path(&entry.path.join("config.json"))?;
            let artifact = if entry.path.join("model.safetensors.index.json").exists() {
                "model.safetensors.index.json"
            } else if entry.path.join("model.safetensors").exists() {
                "model.safetensors"
            } else {
                return None;
            };
            Some(format!(
                "{}@{}/{}",
                identity.repo_id, identity.revision, artifact
            ))
        }
    }
}

fn installed_entry_run_command(entry: &InstalledModelEntry) -> Option<String> {
    let path = entry.path.to_str()?;
    match entry.kind {
        InstalledModelKind::Gguf => Some(format!("mesh-llm --gguf-file {}", shell_escape(path))),
        InstalledModelKind::Mlx => Some(format!("mesh-llm --mlx-file {}", shell_escape(path))),
    }
}

fn shell_escape(value: &str) -> String {
    if value
        .chars()
        .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '/' | '.' | '_' | '-'))
    {
        value.to_string()
    } else {
        format!("'{}'", value.replace('\'', r"'\''"))
    }
}

fn installed_catalog_model(entry: &InstalledModelEntry) -> Option<&'static catalog::CatalogModel> {
    find_catalog_model_exact(&entry.name).or_else(|| match entry.kind {
        InstalledModelKind::Gguf => {
            let file_name = entry.path.file_name()?.to_str()?;
            catalog::MODEL_CATALOG
                .iter()
                .find(|model| model.file == file_name)
        }
        InstalledModelKind::Mlx => {
            let identity =
                crate::models::huggingface_identity_for_path(&entry.path.join("config.json"))?;
            catalog::MODEL_CATALOG.iter().find(|model| {
                model.source_repo() == Some(identity.repo_id.as_str())
                    && matches!(
                        model.source_file(),
                        Some("model.safetensors") | Some("model.safetensors.index.json")
                    )
            })
        }
    })
}

fn installed_entry_display_name(entry: &InstalledModelEntry) -> String {
    installed_catalog_model(entry)
        .map(|model| model.name.clone())
        .unwrap_or_else(|| entry.name.clone())
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
        } => run_model_search(query, *catalog, *limit).await?,
        ModelsCommand::Show { model } => run_model_show(model).await?,
        ModelsCommand::Download {
            model,
            gguf,
            mlx,
            draft,
        } => {
            let preference = if *gguf {
                ResolveFormatPreference::Gguf
            } else if *mlx {
                ResolveFormatPreference::Mlx
            } else {
                ResolveFormatPreference::Auto
            };
            run_model_download(model, preference, *draft).await?
        }
        ModelsCommand::Migrate { apply, prune } => crate::models::run_migrate(*apply, *prune)?,
        ModelsCommand::Updates { repo, all, check } => {
            crate::models::run_update(repo.as_deref(), *all, *check)?
        }
    }
    Ok(())
}
