mod formatters;
mod formatters_console;
mod formatters_json;

use crate::cli::models::ModelSearchSort;
use crate::cli::models::ModelsCommand;
use crate::cli::terminal_progress::{clear_stderr_line, start_spinner, DeterminateProgressLine};
use crate::models::{
    catalog, catalog_model_ref, delete, download_model_ref_with_progress_details,
    find_catalog_model_exact, installed_model_capabilities, load_model_usage_record_for_path,
    model_usage_cache_dir, plan_model_cleanup, scan_installed_models, search_catalog_models,
    search_huggingface, show_exact_model, show_model_variants_with_progress, ModelCleanupPlan,
    ModelCleanupResult, SearchArtifactFilter, SearchProgress, SearchSort, ShowVariantsProgress,
};
use anyhow::{anyhow, bail, Result};
use serde_json::json;
use std::io::IsTerminal;
use std::time::Duration;
use std::time::Instant;

use formatters::{
    catalog_model_is_mlx, format_installed_size, format_relative_timestamp, model_kind_code,
    models_formatter, search_formatter, InstalledRow,
};

pub async fn run_model_search(
    query: &[String],
    _prefer_gguf: bool,
    prefer_mlx: bool,
    catalog_only: bool,
    limit: usize,
    sort: ModelSearchSort,
    json_output: bool,
) -> Result<()> {
    let formatter = search_formatter(json_output);
    let query = query.join(" ");
    let filter = if prefer_mlx {
        SearchArtifactFilter::Mlx
    } else {
        SearchArtifactFilter::Gguf
    };
    let search_sort = map_search_sort(sort);

    if catalog_only {
        let results: Vec<_> = search_catalog_models(&query)
            .into_iter()
            .filter(|model| match filter {
                SearchArtifactFilter::Gguf => !catalog_model_is_mlx(model),
                SearchArtifactFilter::Mlx => catalog_model_is_mlx(model),
            })
            .collect();
        if results.is_empty() {
            return formatter.render_catalog_empty(&query, filter, search_sort);
        }
        return formatter.render_catalog_results(&query, filter, &results, limit, search_sort);
    }

    let mut announced_repo_scan = false;
    let mut last_reported_completed = 0usize;
    let mut search_spinner = if formatter.is_json() {
        None
    } else {
        Some(start_spinner(&format!(
            "Searching Hugging Face {} repos for '{}'",
            formatters::filter_label(filter),
            query
        )))
    };
    let mut repo_spinner = None;
    let repo_progress = DeterminateProgressLine::new("🔎");
    let results = search_huggingface(
        &query,
        limit,
        filter,
        search_sort,
        |progress| match progress {
            SearchProgress::SearchingHub => {}
            SearchProgress::InspectingRepos { completed, total } => {
                if formatter.is_json() {
                    return;
                }
                if let Some(mut spinner) = search_spinner.take() {
                    spinner.finish();
                }
                if total == 0 {
                    return;
                }
                if !announced_repo_scan {
                    announced_repo_scan = true;
                    repo_spinner = Some(start_spinner(&format!(
                        "Inspecting {total} candidate repos..."
                    )));
                }
                if completed == 0 {
                    return;
                }
                if let Some(mut spinner) = repo_spinner.take() {
                    spinner.finish();
                }
                if completed < total && completed < last_reported_completed.saturating_add(5) {
                    return;
                }
                last_reported_completed = completed;
                let _ = repo_progress.draw_counts(
                    "Inspecting repos",
                    completed,
                    total,
                    Some(" candidate repos"),
                );
                if completed == total {
                    let _ = clear_stderr_line();
                    eprintln!("   Inspected {completed}/{total} candidate repos...");
                }
            }
        },
    )
    .await?;
    if let Some(mut spinner) = search_spinner.take() {
        spinner.finish();
    }
    if let Some(mut spinner) = repo_spinner.take() {
        spinner.finish();
    }
    if results.is_empty() {
        return formatter.render_hf_empty(&query, filter, search_sort);
    }
    formatter.render_hf_results(&query, filter, search_sort, &results)
}

pub fn run_model_recommended(json_output: bool) -> Result<()> {
    let formatter = models_formatter(json_output);
    let models: Vec<_> = catalog::MODEL_CATALOG.iter().collect();
    formatter.render_recommended(&models)
}

fn build_installed_rows() -> Vec<InstalledRow> {
    scan_installed_models()
        .into_iter()
        .map(|name| {
            let path = crate::models::find_model_path(&name);
            let display_name = crate::models::installed_model_display_name(&name);
            let catalog_model = find_catalog_model_exact(&name);
            let layer_count = crate::models::layered_package_layer_count_for_path(&path);
            let model_ref = if layer_count.is_some() {
                name.clone()
            } else if let Some(model) = catalog_model {
                catalog_model_ref(model)
            } else if let Some(identity) = crate::models::huggingface_identity_for_path(&path) {
                crate::models::installed_model_huggingface_ref(&identity)
            } else {
                name.clone()
            };
            let show_command = layer_count
                .is_none()
                .then(|| format!("mesh-llm models show {model_ref}"));
            let download_command = layer_count
                .is_none()
                .then(|| format!("mesh-llm models download {model_ref}"));
            let delete_command = format!("mesh-llm models delete {model_ref}");
            let size =
                if let Some(bytes) = crate::models::layered_package_total_bytes_for_path(&path) {
                    Some(bytes)
                } else if path
                    .extension()
                    .and_then(|ext| ext.to_str())
                    .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"))
                {
                    Some(crate::inference::election::total_model_bytes(&path))
                } else {
                    std::fs::metadata(&path).map(|meta| meta.len()).ok()
                };
            let capabilities = installed_model_capabilities(&name);
            let usage = load_model_usage_record_for_path(&path);
            InstalledRow {
                name: display_name,
                model_ref,
                show_command,
                download_command,
                delete_command,
                path,
                size,
                layer_count,
                catalog_model,
                capabilities,
                managed_by_mesh: usage.as_ref().is_some_and(|record| record.mesh_managed),
                last_used_at: usage.map(|record| record.last_used_at),
            }
        })
        .collect()
}

pub fn run_model_installed(json_output: bool) -> Result<()> {
    let formatter = models_formatter(json_output);
    let rows = build_installed_rows();
    formatter.render_installed(&rows)
}

pub fn run_model_cleanup(unused_since: Option<&str>, yes: bool, json_output: bool) -> Result<()> {
    let unused_duration = unused_since.map(parse_cleanup_age).transpose()?;
    let plan = plan_model_cleanup(unused_duration)?;
    if yes {
        let result = crate::models::execute_model_cleanup(unused_duration)?;
        if json_output {
            render_cleanup_json(unused_since, &plan, Some(&result))?;
        } else {
            render_cleanup_console(unused_since, &plan, Some(&result))?;
        }
    } else if json_output {
        render_cleanup_json(unused_since, &plan, None)?;
    } else {
        render_cleanup_console(unused_since, &plan, None)?;
    }
    Ok(())
}

// Delete command integration will be implemented in Task 2.

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
        let variants_progress = DeterminateProgressLine::new("🔎");
        let variants = show_model_variants_with_progress(&details.exact_ref, |progress| {
            if !interactive {
                return;
            }
            match progress {
                ShowVariantsProgress::Inspecting { completed, total } => {
                    if total == 0 {
                        return;
                    }
                    let _ = variants_progress.draw_counts(
                        "Inspecting variant sizes",
                        completed,
                        total,
                        None,
                    );
                    if completed == total {
                        let _ = clear_stderr_line();
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
    let (path, details) = download_model_ref_with_progress_details(model_ref, !json_output).await?;
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
        ModelsCommand::Cleanup {
            unused_since,
            yes,
            json,
        } => run_model_cleanup(unused_since.as_deref(), *yes, *json)?,
        ModelsCommand::Prune { yes, json } => run_model_prune(*yes, *json)?,
        ModelsCommand::Search {
            query,
            gguf,
            mlx,
            catalog,
            limit,
            sort,
            json,
        } => run_model_search(query, *gguf, *mlx, *catalog, *limit, *sort, *json).await?,
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
            let repo_for_update = repo.clone();
            let repo_for_render = repo.clone();
            let all = *all;
            let check = *check;
            tokio::task::spawn_blocking(move || {
                crate::models::run_update(repo_for_update.as_deref(), all, check)
            })
            .await
            .map_err(anyhow::Error::from)??;
            if *json {
                let formatter = models_formatter(*json);
                formatter.render_updates_status(repo_for_render.as_deref(), all, check)?;
            }
        }
        ModelsCommand::Delete { model, yes, json } => {
            run_model_delete(model.as_str(), *yes, *json).await?
        }
    }
    Ok(())
}

fn run_model_prune(yes: bool, json_output: bool) -> Result<()> {
    let cache_dir = crate::inference::skippy::materialized_stage_cache_dir();
    if !yes {
        if json_output {
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "dry_run": true,
                    "cache_dir": cache_dir,
                    "apply": "mesh-llm models prune --yes",
                }))?
            );
        } else {
            println!("🧹 Derived stage cache prune preview");
            println!("📁 Cache: {}", cache_dir.display());
            println!("Apply with:");
            println!("  mesh-llm models prune --yes");
        }
        return Ok(());
    }
    let removed = crate::inference::skippy::prune_unpinned_materialized_stages()?;
    if json_output {
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "dry_run": false,
                "cache_dir": cache_dir,
                "removed_files": removed,
            }))?
        );
    } else {
        println!("✅ Derived stage cache pruned");
        println!("Removed files: {}", removed);
    }
    Ok(())
}

fn parse_cleanup_age(value: &str) -> Result<Duration> {
    let value = value.trim();
    if value.is_empty() {
        bail!("Cleanup age must not be empty");
    }
    let split_index = value
        .find(|ch: char| !ch.is_ascii_digit())
        .unwrap_or(value.len());
    if split_index == 0 || split_index == value.len() {
        bail!("Use a cleanup age like 12h, 7d, or 30m");
    }
    let amount: u64 = value[..split_index]
        .parse()
        .map_err(|_| anyhow!("Invalid cleanup age: {value}"))?;
    let unit = value[split_index..].to_ascii_lowercase();
    let seconds = match unit.as_str() {
        "m" | "min" | "mins" | "minute" | "minutes" => amount.saturating_mul(60),
        "h" | "hr" | "hrs" | "hour" | "hours" => amount.saturating_mul(60 * 60),
        "d" | "day" | "days" => amount.saturating_mul(60 * 60 * 24),
        "w" | "week" | "weeks" => amount.saturating_mul(60 * 60 * 24 * 7),
        _ => bail!("Unsupported cleanup age unit '{unit}'. Use m, h, d, or w."),
    };
    Ok(Duration::from_secs(seconds))
}

fn render_cleanup_console(
    unused_since: Option<&str>,
    plan: &ModelCleanupPlan,
    result: Option<&ModelCleanupResult>,
) -> Result<()> {
    let executed = result.is_some();
    if executed {
        println!("✅ Model cleanup complete");
    } else {
        println!("🧹 Model cleanup preview");
    }
    println!(
        "📁 HF cache: {}",
        crate::models::huggingface_hub_cache_dir().display()
    );
    println!("📁 Mesh cache: {}", model_usage_cache_dir().display());
    println!("🛡️ Scope: mesh-managed records only");
    if let Some(unused_since) = unused_since {
        println!("⏱️ Filter: unused for at least {}", unused_since);
    }
    println!();

    if plan.candidates.is_empty() {
        println!("No mesh-managed models matched the cleanup filters.");
    } else {
        for candidate in &plan.candidates {
            println!("📦 {}", candidate.display_name);
            if candidate.stale_record_only {
                println!("   would remove: stale usage record only");
            } else {
                println!(
                    "   would remove: {} across {} file{}",
                    format_installed_size(candidate.total_bytes),
                    candidate.file_count,
                    if candidate.file_count == 1 { "" } else { "s" }
                );
            }
            if let Some(model_ref) = candidate.model_ref.as_deref() {
                println!("   ref: {}", model_ref);
            }
            println!("   source: {}", candidate.source);
            if let Some(label) = format_relative_timestamp(&candidate.last_used_at) {
                println!("   last used: {}", label);
            }
            println!("   path: {}", candidate.primary_path.display());
            if candidate.stale_record_only {
                println!("   note: no managed files remain on disk; cleanup only removes the usage record");
            }
            println!();
        }
    }

    if let Some(result) = result {
        println!("Removed model records: {}", result.removed_candidates);
        println!("Removed files: {}", result.removed_files);
        println!(
            "Removed metadata cache files: {}",
            result.removed_metadata_files
        );
        println!("Removed usage records: {}", result.removed_records);
        println!(
            "Reclaimed: {}",
            format_installed_size(result.reclaimed_bytes)
        );
    } else {
        println!(
            "Would remove: {} across {} file{}",
            format_installed_size(plan.total_bytes),
            plan.total_files,
            if plan.total_files == 1 { "" } else { "s" }
        );
        if plan.stale_record_only > 0 {
            println!(
                "Would also clear {} stale usage record{}",
                plan.stale_record_only,
                if plan.stale_record_only == 1 { "" } else { "s" }
            );
        }
        if plan.skipped_recent > 0 {
            println!(
                "Skipped recent mesh-managed record{}: {}",
                if plan.skipped_recent == 1 { "" } else { "s" },
                plan.skipped_recent
            );
        }
        println!();
        println!("Apply with:");
        print!("  mesh-llm models cleanup");
        if let Some(unused_since) = unused_since {
            print!(" --unused-since {}", unused_since);
        }
        println!(" --yes");
    }
    Ok(())
}

fn render_cleanup_json(
    unused_since: Option<&str>,
    plan: &ModelCleanupPlan,
    result: Option<&ModelCleanupResult>,
) -> Result<()> {
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "hf_cache_dir": crate::models::huggingface_hub_cache_dir(),
            "mesh_cache_dir": model_usage_cache_dir(),
            "mesh_managed_only": true,
            "unused_since": unused_since,
            "dry_run": result.is_none(),
            "plan": plan,
            "result": result,
        }))?
    );
    Ok(())
}

pub async fn run_model_delete(model: &str, yes: bool, json_output: bool) -> Result<()> {
    let paths = match delete::resolve_model_identifier(model).await {
        Ok(p) => p,
        Err(e) => bail!("{e}"),
    };

    if paths.is_empty() {
        bail!("Model not found: {}", model);
    }

    if !yes {
        let derived_stage_paths =
            crate::inference::skippy::materialized_stages_for_sources(&paths)?;
        let resolved = build_delete_preview_model(model, &paths, derived_stage_paths);
        let formatter = models_formatter(json_output);
        return formatter.render_delete_preview(&resolved);
    }

    let removed_derived_cache_files =
        crate::inference::skippy::remove_materialized_stages_for_sources(&paths)?;
    let mut result = delete::delete_model_by_identifier(model).await?;
    result.removed_derived_cache_files = removed_derived_cache_files;
    let formatter = models_formatter(json_output);
    formatter.render_delete_result(&result)
}

fn build_delete_preview_model(
    model: &str,
    paths: &[std::path::PathBuf],
    derived_stage_paths: Vec<std::path::PathBuf>,
) -> crate::models::ResolvedModel {
    let primary_path = paths[0].clone();
    let display_name = if paths.len() > 1 {
        format!("{model} ({} files)", paths.len())
    } else {
        primary_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string()
    };

    crate::models::ResolvedModel {
        path: primary_path,
        paths: paths.to_vec(),
        derived_stage_paths,
        display_name,
        is_exact_path: false,
        matched_records: vec![],
    }
}

fn map_search_sort(sort: ModelSearchSort) -> SearchSort {
    match sort {
        ModelSearchSort::Trending => SearchSort::Trending,
        ModelSearchSort::Downloads => SearchSort::Downloads,
        ModelSearchSort::Likes => SearchSort::Likes,
        ModelSearchSort::Created => SearchSort::Created,
        ModelSearchSort::Updated => SearchSort::Updated,
        ModelSearchSort::ParametersDesc => SearchSort::ParametersDesc,
        ModelSearchSort::ParametersAsc => SearchSort::ParametersAsc,
    }
}

#[cfg(test)]
mod tests {
    use super::{build_delete_preview_model, build_installed_rows, parse_cleanup_age};
    use serial_test::serial;
    use std::ffi::OsString;
    use std::path::{Path, PathBuf};

    fn unique_temp_dir(prefix: &str) -> PathBuf {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("mesh-llm-{prefix}-{stamp}"))
    }

    fn restore_env(key: &str, previous: Option<OsString>) {
        if let Some(value) = previous {
            std::env::set_var(key, value);
        } else {
            std::env::remove_var(key);
        }
    }

    fn create_cache_repo_file(
        root: &Path,
        repo_id: &str,
        revision: &str,
        relative_file: &str,
        size_bytes: usize,
    ) -> PathBuf {
        let repo_dir = root.join(format!("models--{}", repo_id.replace('/', "--")));
        let refs_dir = repo_dir.join("refs");
        let snapshot_dir = repo_dir.join("snapshots").join(revision);
        std::fs::create_dir_all(&refs_dir).unwrap();
        std::fs::create_dir_all(
            snapshot_dir.join(Path::new(relative_file).parent().unwrap_or(Path::new(""))),
        )
        .unwrap();
        std::fs::write(refs_dir.join("main"), revision).unwrap();

        let path = snapshot_dir.join(relative_file);
        std::fs::write(&path, vec![0u8; size_bytes]).unwrap();
        path
    }

    #[test]
    fn cleanup_age_parser_accepts_common_units() {
        assert_eq!(
            parse_cleanup_age("30m").expect("minutes should parse"),
            std::time::Duration::from_secs(30 * 60)
        );
        assert_eq!(
            parse_cleanup_age("12h").expect("hours should parse"),
            std::time::Duration::from_secs(12 * 60 * 60)
        );
        assert_eq!(
            parse_cleanup_age("7d").expect("days should parse"),
            std::time::Duration::from_secs(7 * 24 * 60 * 60)
        );
    }

    #[test]
    fn cleanup_age_parser_rejects_missing_or_unknown_units() {
        assert!(parse_cleanup_age("30").is_err());
        assert!(parse_cleanup_age("2months").is_err());
        assert!(parse_cleanup_age("").is_err());
    }

    #[test]
    #[serial]
    fn installed_rows_keep_layered_package_ref_and_safe_commands() {
        let prev_hub_cache = std::env::var_os("HF_HUB_CACHE");
        let prev_hf_home = std::env::var_os("HF_HOME");
        let prev_xdg = std::env::var_os("XDG_CACHE_HOME");

        let temp = unique_temp_dir("installed-layered-row");
        create_cache_repo_file(
            &temp,
            "meshllm/DeepSeek-V3.2-UD-Q4_K_XL-layers",
            "abcdef1234567890",
            "shared/embeddings.gguf",
            6,
        );
        create_cache_repo_file(
            &temp,
            "meshllm/DeepSeek-V3.2-UD-Q4_K_XL-layers",
            "abcdef1234567890",
            "layers/layer-000.gguf",
            9,
        );
        create_cache_repo_file(
            &temp,
            "meshllm/DeepSeek-V3.2-UD-Q4_K_XL-layers",
            "abcdef1234567890",
            "layers/layer-001.gguf",
            9,
        );

        std::env::set_var("HF_HUB_CACHE", &temp);
        std::env::remove_var("HF_HOME");
        std::env::remove_var("XDG_CACHE_HOME");

        let rows = build_installed_rows();
        assert_eq!(rows.len(), 1);
        let row = &rows[0];
        assert_eq!(row.model_ref, "meshllm/DeepSeek-V3.2-UD-Q4_K_XL-layers");
        assert_eq!(row.layer_count, Some(2));
        assert_eq!(row.show_command, None);
        assert_eq!(row.download_command, None);
        assert_eq!(
            row.delete_command,
            "mesh-llm models delete meshllm/DeepSeek-V3.2-UD-Q4_K_XL-layers"
        );

        let _ = std::fs::remove_dir_all(&temp);
        restore_env("HF_HUB_CACHE", prev_hub_cache);
        restore_env("HF_HOME", prev_hf_home);
        restore_env("XDG_CACHE_HOME", prev_xdg);
    }

    #[test]
    fn delete_preview_model_retains_all_resolved_package_paths() {
        let paths = vec![
            PathBuf::from("/tmp/demo-layers/layers/layer-000.gguf"),
            PathBuf::from("/tmp/demo-layers/layers/layer-001.gguf"),
            PathBuf::from("/tmp/demo-layers/shared/embeddings.gguf"),
        ];

        let derived_stage_paths =
            vec![PathBuf::from("/tmp/mesh-llm/skippy-stages/demo-stage.gguf")];
        let resolved =
            build_delete_preview_model("meshllm/Demo-layers", &paths, derived_stage_paths.clone());

        assert_eq!(resolved.path, paths[0]);
        assert_eq!(resolved.paths, paths);
        assert_eq!(resolved.derived_stage_paths, derived_stage_paths);
        assert_eq!(resolved.display_name, "meshllm/Demo-layers (3 files)");
    }
}
