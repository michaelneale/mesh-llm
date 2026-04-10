mod formatters;
mod formatters_console;
mod formatters_json;

use crate::cli::models::ModelsCommand;
use crate::models::{
    catalog, download_exact_ref, find_catalog_model_exact, installed_model_capabilities,
    scan_installed_models, search_catalog_models, search_huggingface, show_exact_model,
    show_model_variants_with_progress, SearchArtifactFilter, SearchProgress, ShowVariantsProgress,
};
use anyhow::{anyhow, Result};
use std::io::{IsTerminal, Write};
use std::time::Instant;

use formatters::{
    catalog_model_is_mlx, model_kind_code, models_formatter, search_formatter, InstalledRow,
};

pub async fn run_model_search(
    query: &[String],
    prefer_gguf: bool,
    prefer_mlx: bool,
    catalog_only: bool,
    limit: usize,
    json_output: bool,
) -> Result<()> {
    let formatter = search_formatter(json_output);
    let query = query.join(" ");
    let filter = if prefer_mlx {
        SearchArtifactFilter::Mlx
    } else if prefer_gguf {
        SearchArtifactFilter::Gguf
    } else {
        SearchArtifactFilter::Gguf
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
            return formatter.render_catalog_empty(&query, filter);
        }
        return formatter.render_catalog_results(&query, filter, &results, limit);
    }

    if !formatter.is_json() {
        eprintln!(
            "🔎 Searching Hugging Face {} repos for '{}'...",
            formatters::filter_label(filter),
            query
        );
    }
    let mut announced_repo_scan = false;
    let results = search_huggingface(&query, limit, filter, |progress| match progress {
        SearchProgress::SearchingHub => {}
        SearchProgress::InspectingRepos { completed, total } => {
            if formatter.is_json() {
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
        return formatter.render_hf_empty(&query, filter);
    }
    formatter.render_hf_results(&query, filter, &results)
}

pub fn run_model_recommended(json_output: bool) -> Result<()> {
    let formatter = models_formatter(json_output);
    let models: Vec<_> = catalog::MODEL_CATALOG.iter().collect();
    formatter.render_recommended(&models)
}

pub fn run_model_installed(json_output: bool) -> Result<()> {
    let formatter = models_formatter(json_output);
    let rows: Vec<InstalledRow> = scan_installed_models()
        .into_iter()
        .map(|name| {
            let path = crate::models::find_model_path(&name);
            let size = std::fs::metadata(&path).map(|meta| meta.len()).ok();
            let catalog_model = find_catalog_model_exact(&name);
            let capabilities = installed_model_capabilities(&name);
            InstalledRow {
                name,
                path,
                size,
                catalog_model,
                capabilities,
            }
        })
        .collect();
    formatter.render_installed(&rows)
}

pub async fn run_model_show(model_ref: &str, json_output: bool) -> Result<()> {
    let formatter = models_formatter(json_output);
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
    let is_gguf = model_kind_code(details.kind) == "gguf";
    let variants = if is_gguf {
        let variants_started = Instant::now();
        if interactive {
            eprintln!("🔎 Fetching GGUF variants from Hugging Face...");
        }
        let variants = show_model_variants_with_progress(&details.exact_ref, |progress| {
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
    formatter.render_show(&details, variants.as_deref())
}

pub async fn run_model_download(
    model_ref: &str,
    include_draft: bool,
    json_output: bool,
) -> Result<()> {
    let formatter = models_formatter(json_output);
    let details = show_exact_model(model_ref).await.ok();
    let download_ref = details
        .as_ref()
        .map(|d| d.exact_ref.as_str())
        .unwrap_or(model_ref);
    let path = download_exact_ref(download_ref).await?;
    if !include_draft {
        return formatter.render_download(model_ref, &path, details.as_ref(), false, None);
    }

    let mut draft_out: Option<(String, std::path::PathBuf)> = None;
    if let Some(details_ref) = details.as_ref() {
        if let Some(draft_name) = details_ref.draft.as_deref() {
            let draft_model = find_catalog_model_exact(draft_name)
                .ok_or_else(|| anyhow!("Draft model '{}' not found in catalog", draft_name))?;
            let draft_path = catalog::download_model(draft_model).await?;
            draft_out = Some((draft_name.to_string(), draft_path));
        } else if !json_output {
            eprintln!(
                "⚠ No draft model available for {}",
                details_ref.display_name
            );
        }
    }
    formatter.render_download(
        model_ref,
        &path,
        details.as_ref(),
        true,
        draft_out.as_ref().map(|(n, p)| (n.as_str(), p.as_path())),
    )
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
                let formatter = models_formatter(*json);
                formatter.render_updates_status(repo.as_deref(), *all, *check)?;
            }
        }
    }
    Ok(())
}
