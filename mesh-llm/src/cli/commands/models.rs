use crate::cli::models::ModelsCommand;
use crate::models::{
    capabilities, catalog, download_exact_ref, find_catalog_model_exact, huggingface_hub_cache_dir,
    installed_model_capabilities, scan_installed_models, search_catalog_models, search_huggingface,
    show_exact_model, show_model_variants_with_progress, ModelCapabilities, SearchArtifactFilter,
    SearchProgress, ShowVariantsProgress,
};
use crate::system::hardware;
use anyhow::{anyhow, Result};
use serde_json::{json, Value};
use std::io::{IsTerminal, Write};
use std::path::Path;
use std::time::Instant;
use unicode_width::UnicodeWidthStr;

pub async fn run_model_search(
    query: &[String],
    prefer_gguf: bool,
    prefer_mlx: bool,
    catalog_only: bool,
    limit: usize,
    json_output: bool,
) -> Result<()> {
    let query = query.join(" ");
    let filter = if prefer_mlx {
        SearchArtifactFilter::Mlx
    } else if prefer_gguf {
        SearchArtifactFilter::Gguf
    } else {
        SearchArtifactFilter::Gguf
    };
    let filter_label = match filter {
        SearchArtifactFilter::Gguf => "GGUF",
        SearchArtifactFilter::Mlx => "MLX",
    };

    if catalog_only {
        let results: Vec<_> = search_catalog_models(&query)
            .into_iter()
            .filter(|model| match filter {
                SearchArtifactFilter::Gguf => !catalog_model_is_mlx(model),
                SearchArtifactFilter::Mlx => catalog_model_is_mlx(model),
            })
            .collect();
        if results.is_empty() {
            if json_output {
                print_json(json!({
                    "query": query,
                    "filter": filter_name(filter),
                    "source": "catalog",
                    "results": [],
                }))?;
                return Ok(());
            }
            eprintln!("🔎 No {filter_label} catalog models matched '{query}'.");
            return Ok(());
        }
        if json_output {
            let payload_results: Vec<Value> = results
                .into_iter()
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
                        "capabilities": capabilities_json(capabilities::infer_catalog_capabilities(model)),
                    })
                })
                .collect();
            print_json(json!({
                "query": query,
                "filter": filter_name(filter),
                "source": "catalog",
                "machine": local_capacity_json(),
                "results": payload_results,
            }))?;
            return Ok(());
        }
        println!("📚 {filter_label} catalog matches for '{query}'");
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

    if !json_output {
        eprintln!("🔎 Searching Hugging Face {filter_label} repos for '{query}'...");
    }
    let mut announced_repo_scan = false;
    let results = search_huggingface(&query, limit, filter, |progress| match progress {
        SearchProgress::SearchingHub => {}
        SearchProgress::InspectingRepos { completed, total } => {
            if json_output {
                return;
            }
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
        if json_output {
            print_json(json!({
                "query": query,
                "filter": filter_name(filter),
                "source": "huggingface",
                "results": [],
            }))?;
            return Ok(());
        }
        eprintln!("🔎 No Hugging Face {filter_label} matches for '{query}'.");
        return Ok(());
    }
    if json_output {
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
        }))?;
        return Ok(());
    }

    println!("🔎 Hugging Face {filter_label} matches for '{query}'");
    if let Some(summary) = local_capacity_summary() {
        println!("{}", summary);
    }
    println!();
    for (index, result) in results.iter().enumerate() {
        println!("{}. 📦 {}", index + 1, result.repo_id);
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

pub fn run_model_recommended(json_output: bool) -> Result<()> {
    if json_output {
        let models: Vec<Value> = catalog::MODEL_CATALOG
            .iter()
            .map(|model| {
                let model_capabilities = capabilities::infer_catalog_capabilities(model);
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
        return print_json(json!({
            "source": "catalog",
            "results": models,
        }));
    }
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
    Ok(())
}

pub fn run_model_installed(json_output: bool) -> Result<()> {
    let installed = scan_installed_models();
    if json_output {
        let models: Vec<Value> = installed
            .iter()
            .map(|name| {
                let path = crate::models::find_model_path(name);
                let size = std::fs::metadata(&path).map(|meta| meta.len()).ok();
                let catalog_model = find_catalog_model_exact(name);
                let model_capabilities = installed_model_capabilities(name);
                json!({
                    "name": name,
                    "type": installed_model_kind_code(&path),
                    "size_bytes": size,
                    "size": size.map(format_installed_size),
                    "capabilities": capabilities_json(model_capabilities),
                    "ref": name,
                    "show": format!("mesh-llm models show {}", name),
                    "download": format!("mesh-llm models download {}", name),
                    "path": path,
                    "about": catalog_model.map(|m| m.description.clone()),
                    "draft": catalog_model.and_then(|m| m.draft.clone()),
                    "moe": moe_json(catalog_model.and_then(|m| m.moe.as_ref())),
                })
            })
            .collect();
        return print_json(json!({
            "cache_dir": huggingface_hub_cache_dir(),
            "results": models,
        }));
    }
    if installed.is_empty() {
        println!("📦 No installed models found");
        println!("   HF cache: {}", huggingface_hub_cache_dir().display());
        return Ok(());
    }

    println!("💾 Installed models");
    println!("📁 HF cache: {}", huggingface_hub_cache_dir().display());
    println!();
    for name in installed.iter() {
        let path = crate::models::find_model_path(&name);
        let size = std::fs::metadata(&path).map(|meta| meta.len()).ok();
        let catalog_model = find_catalog_model_exact(&name);
        let model_capabilities = installed_model_capabilities(&name);

        println!("📦 {}", name);
        println!("   type: {}", installed_model_kind(&path));
        if let Some(bytes) = size {
            println!("   📏 {}", format_installed_size(bytes));
        }
        let mut caps = vec!["💬 text".to_string()];
        if model_capabilities.multimodal_label().is_some() {
            caps.push("🎛️ multimodal".to_string());
        }
        if let Some(label) = model_capabilities.vision_label() {
            caps.push(format!("👁️ vision ({label})"));
        }
        if let Some(label) = model_capabilities.audio_label() {
            caps.push(format!("🔊 audio ({label})"));
        }
        if let Some(label) = model_capabilities.reasoning_label() {
            caps.push(format!("🧠 reasoning ({label})"));
        }
        if let Some(label) = model_capabilities.tool_use_label() {
            caps.push(format!("🛠️ tool use ({label})"));
        }
        println!("   capabilities: {}", caps.join("  "));
        println!("   ref: {}", name);
        println!("   show: mesh-llm models show {}", name);
        println!("   download: mesh-llm models download {}", name);
        println!("   path: {}", path.display());
        if let Some(model) = catalog_model {
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

pub async fn run_model_show(model_ref: &str, json_output: bool) -> Result<()> {
    let interactive = !json_output && std::io::stdout().is_terminal();
    let detail_started = Instant::now();
    if interactive {
        eprintln!("🔎 Resolving model details from Hugging Face...");
    }
    let details = show_exact_model(model_ref).await?;
    if interactive {
        eprintln!(
            "✅ Resolved model details ({:.1}s)",
            detail_started.elapsed().as_secs_f32()
        );
    }
    if !json_output {
        if model_kind_code(details.kind) == "mlx" {
            println!("🔎 {}", details.exact_ref);
        } else {
            println!("🔎 {}", details.display_name);
        }
        if let Some(summary) = local_capacity_summary() {
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
    }

    let is_gguf = model_kind_code(details.kind) == "gguf";
    let variants = if is_gguf {
        let variants_started = Instant::now();
        if interactive {
            eprintln!("🔎 Fetching GGUF variants from Hugging Face...");
        }
        let variants = show_model_variants_with_progress(model_ref, |progress| {
            if !interactive {
                return;
            }
            match progress {
                ShowVariantsProgress::Inspecting { completed, total } => {
                    if total == 0 {
                        return;
                    }
                    eprint!("\r   Inspecting variant sizes {completed}/{total}...");
                    let _ = std::io::stderr().flush();
                    if completed == total {
                        eprintln!();
                    }
                }
            }
        })
        .await?;
        if let Some(variants) = &variants {
            if interactive {
                eprintln!(
                    "✅ Fetched {} GGUF variants ({:.1}s)",
                    variants.len(),
                    variants_started.elapsed().as_secs_f32()
                );
            }
        } else if interactive {
            eprintln!(
                "✅ No GGUF variants for this ref ({:.1}s)",
                variants_started.elapsed().as_secs_f32()
            );
        }
        variants
    } else {
        None
    };
    if let Some(variants) = &variants {
        if !variants.is_empty() && !json_output {
            println!();
            println!("Variants:");
            let mut rows = Vec::new();
            for variant in variants.iter() {
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
    if json_output {
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
                .into_iter()
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
        }))?;
    }
    Ok(())
}

pub async fn run_model_download(
    model_ref: &str,
    include_draft: bool,
    json_output: bool,
) -> Result<()> {
    let details = show_exact_model(model_ref).await.ok();
    let path = download_exact_ref(model_ref).await?;
    if json_output {
        let mut payload = json!({
            "requested_ref": model_ref,
            "path": path,
            "type": details.as_ref().map(|d| model_kind_code(d.kind)),
            "resolved_ref": details.as_ref().map(|d| d.exact_ref.clone()),
        });
        if !include_draft {
            return print_json(payload);
        }
        if let Some(details) = details {
            if let Some(draft) = details.draft {
                let draft_model = find_catalog_model_exact(&draft)
                    .ok_or_else(|| anyhow!("Draft model '{}' not found in catalog", draft))?;
                let draft_path = catalog::download_model(draft_model).await?;
                payload["draft"] = json!({
                    "name": draft,
                    "path": draft_path,
                });
            } else {
                payload["draft"] = Value::Null;
            }
        }
        return print_json(payload);
    }
    println!("✅ Downloaded model");
    if let Some(details) = &details {
        println!("   type: {}", details.kind);
    }
    println!("   {}", path.display());

    if !include_draft {
        return Ok(());
    }

    let Some(details) = details else {
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

fn installed_model_kind(path: &Path) -> &'static str {
    let text = path.to_string_lossy().to_ascii_lowercase();
    if text.ends_with(".safetensors")
        || text.ends_with(".safetensors.index.json")
        || text.contains("model.safetensors")
    {
        "🍎 MLX"
    } else {
        "🦙 GGUF"
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

fn local_capacity_summary() -> Option<String> {
    let vram_gb = hardware::survey().vram_bytes as f64 / 1e9;
    if vram_gb <= 0.0 {
        None
    } else {
        Some(format!("🖥️ This machine: ~{vram_gb:.1}GB available"))
    }
}

fn local_capacity_json() -> Value {
    let vram_bytes = hardware::survey().vram_bytes;
    let vram_gb = vram_bytes as f64 / 1e9;
    json!({
        "vram_bytes": if vram_bytes > 0 { Some(vram_bytes) } else { None::<u64> },
        "vram_gb": if vram_gb > 0.0 { Some(vram_gb) } else { None::<f64> },
    })
}

fn capabilities_json(caps: ModelCapabilities) -> Value {
    json!({
        "text": true,
        "multimodal": caps.multimodal_status(),
        "vision": caps.vision_status(),
        "audio": caps.audio_status(),
        "reasoning": caps.reasoning_status(),
        "tool_use": caps.tool_use_status(),
        "moe": caps.moe,
    })
}

fn moe_json(moe: Option<&catalog::MoeConfig>) -> Value {
    match moe {
        Some(moe) => json!({
            "n_expert": moe.n_expert,
            "n_expert_used": moe.n_expert_used,
            "min_experts_per_node": moe.min_experts_per_node,
            "ranking_len": moe.ranking.len(),
        }),
        None => Value::Null,
    }
}

fn filter_name(filter: SearchArtifactFilter) -> &'static str {
    match filter {
        SearchArtifactFilter::Gguf => "gguf",
        SearchArtifactFilter::Mlx => "mlx",
    }
}

fn print_json(value: Value) -> Result<()> {
    println!("{}", serde_json::to_string_pretty(&value)?);
    Ok(())
}

fn fit_code_for_size_label(size_label: &str) -> Option<&'static str> {
    let model_gb = catalog::parse_size_gb(size_label);
    let vram_gb = hardware::survey().vram_bytes as f64 / 1e9;
    if model_gb <= 0.0 || vram_gb <= 0.0 {
        return None;
    }

    let code = if model_gb <= vram_gb * 0.6 {
        "comfortable"
    } else if model_gb <= vram_gb * 0.9 {
        "tight"
    } else if model_gb <= vram_gb * 1.1 {
        "tradeoff"
    } else {
        "too_large"
    };
    Some(code)
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

fn variant_selector_label(exact_ref: &str) -> String {
    if let Some((_, selector)) = exact_ref.split_once(':') {
        return selector
            .split_once('@')
            .map(|(value, _)| value)
            .unwrap_or(selector)
            .to_string();
    }
    Path::new(exact_ref)
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or(exact_ref)
        .to_string()
}

fn display_width(value: &str) -> usize {
    UnicodeWidthStr::width(value)
}

fn pad_right_display(value: &str, width: usize) -> String {
    let pad = width.saturating_sub(display_width(value));
    format!("{value}{}", " ".repeat(pad))
}

fn pad_left_display(value: &str, width: usize) -> String {
    let pad = width.saturating_sub(display_width(value));
    format!("{}{value}", " ".repeat(pad))
}

pub async fn dispatch_models_command(command: &ModelsCommand) -> Result<()> {
    match command {
        ModelsCommand::Recommended { json } | ModelsCommand::List { json } => {
            run_model_recommended(*json)?
        }
        ModelsCommand::Installed { json } => run_model_installed(*json)?,
        ModelsCommand::Search {
            query,
            gguf,
            mlx,
            catalog,
            limit,
            json,
        } => run_model_search(query, *gguf, *mlx, *catalog, *limit, *json).await?,
        ModelsCommand::Show { model, json } => run_model_show(model, *json).await?,
        ModelsCommand::Download { model, draft, json } => {
            run_model_download(model, *draft, *json).await?
        }
        ModelsCommand::Updates {
            repo,
            all,
            check,
            json,
        } => {
            crate::models::run_update(repo.as_deref(), *all, *check)?;
            if *json {
                print_json(json!({
                    "repo": repo,
                    "all": all,
                    "check": check,
                    "status": "ok",
                }))?;
            }
        }
    }
    Ok(())
}

fn catalog_model_is_mlx(model: &catalog::CatalogModel) -> bool {
    model
        .source_file()
        .map(|file| {
            file.ends_with("model.safetensors") || file.ends_with("model.safetensors.index.json")
        })
        .unwrap_or(false)
        || model.url.contains("model.safetensors")
}

fn catalog_model_kind(model: &catalog::CatalogModel) -> &'static str {
    if catalog_model_is_mlx(model) {
        "🍎 MLX"
    } else {
        "🦙 GGUF"
    }
}

fn model_kind_code(kind: &str) -> &'static str {
    if kind.to_ascii_lowercase().contains("mlx") {
        "mlx"
    } else {
        "gguf"
    }
}

fn installed_model_kind_code(path: &Path) -> &'static str {
    model_kind_code(installed_model_kind(path))
}

fn catalog_model_kind_code(model: &catalog::CatalogModel) -> &'static str {
    model_kind_code(catalog_model_kind(model))
}
