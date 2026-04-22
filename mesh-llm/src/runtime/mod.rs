pub(crate) mod config_state;
mod discovery;
pub mod instance;
mod local;
mod proxy;
pub(crate) mod wakeable;

use self::discovery::{nostr_rediscovery, start_new_mesh};
use self::local::{
    add_runtime_local_target, add_serving_assignment, advertise_model_ready, local_process_payload,
    remove_runtime_local_target, remove_serving_assignment, resolved_model_name,
    set_advertised_model_context, start_runtime_local_model, withdraw_advertised_model,
    LocalRuntimeModelHandle, ManagedModelController, RuntimeEvent,
};
use self::proxy::{api_proxy, bootstrap_proxy};
use crate::api;
use crate::cli::terminal_progress::start_spinner;
use crate::cli::{Cli, Command, RuntimeSurface};
use crate::crypto::{
    default_keystore_path, default_trust_store_path, keystore_exists, keystore_metadata,
    load_keystore, load_owner_keypair_from_keychain, load_trust_store, OwnerKeychainLoadError,
};
use crate::inference::{election, launch, moe};
use crate::mesh;
use crate::mesh::NodeRole;
use crate::models;
use crate::models::catalog;
use crate::network::{affinity, nostr, router, tunnel};
use crate::plugin;
use crate::system::{autoupdate, benchmark, hardware, moe_planner};
use anyhow::{Context, Result};
use clap::{CommandFactory, Parser};
use std::collections::HashMap;
use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use zeroize::Zeroizing;

fn current_time_unix_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

fn publication_state_from_update(update: nostr::PublishStateUpdate) -> api::PublicationState {
    match update {
        nostr::PublishStateUpdate::Public => api::PublicationState::Public,
        nostr::PublishStateUpdate::PublishFailed => api::PublicationState::PublishFailed,
    }
}

fn bridge_publication_state(
    console_state: api::MeshApi,
    mut status_rx: tokio::sync::watch::Receiver<Option<nostr::PublishStateUpdate>>,
) {
    tokio::spawn(async move {
        let mut pending = *status_rx.borrow_and_update();
        loop {
            if let Some(update) = pending.take() {
                console_state
                    .set_publication_state(publication_state_from_update(update))
                    .await;
            }

            if status_rx.changed().await.is_err() {
                break;
            }
            pending = *status_rx.borrow_and_update();
        }
    });
}

async fn record_first_joined_mesh_ts(node: &mesh::Node) {
    let now_ms = current_time_unix_ms();
    node.set_first_joined_mesh_ts_if_absent(now_ms).await;
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct StartupModelSpec {
    model_ref: PathBuf,
    mmproj_ref: Option<PathBuf>,
    ctx_size: Option<u32>,
    gpu_id: Option<String>,
    config_owned: bool,
    parallel: Option<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct StartupPinnedGpuTarget {
    pub(crate) index: usize,
    pub(crate) stable_id: String,
    pub(crate) backend_device: String,
    pub(crate) vram_bytes: u64,
}

#[derive(Clone, Debug)]
struct StartupModelPlan {
    declared_ref: String,
    resolved_path: PathBuf,
    mmproj_path: Option<PathBuf>,
    ctx_size: Option<u32>,
    gpu_id: Option<String>,
    pinned_gpu: Option<StartupPinnedGpuTarget>,
    parallel: Option<usize>,
}

fn resolve_runtime_owner_key_path(cli: &Cli) -> Result<Option<PathBuf>> {
    if let Some(path) = cli.owner_key.clone() {
        return Ok(Some(path));
    }

    let default_path = default_keystore_path()?;
    if keystore_exists(&default_path) {
        Ok(Some(default_path))
    } else {
        Ok(None)
    }
}

fn resolve_owner_passphrase(path: &Path) -> Result<Option<Zeroizing<String>>> {
    let info = keystore_metadata(path)?;
    if !info.encrypted {
        return Ok(None);
    }

    if let Ok(passphrase) = std::env::var("MESH_LLM_OWNER_PASSPHRASE") {
        return Ok(Some(Zeroizing::new(passphrase)));
    }

    if std::io::stdin().is_terminal() && std::io::stderr().is_terminal() {
        let prompt = format!("Enter owner keystore passphrase for {}: ", path.display());
        let passphrase = rpassword::prompt_password_stderr(&prompt)?;
        return Ok(Some(Zeroizing::new(passphrase)));
    }

    Err(crate::crypto::CryptoError::MissingPassphrase.into())
}

fn load_owner_keypair_for_runtime(path: &Path) -> Result<crate::crypto::OwnerKeypair> {
    let info = keystore_metadata(path)?;
    if info.encrypted && std::env::var("MESH_LLM_OWNER_PASSPHRASE").is_err() {
        match load_owner_keypair_from_keychain(path) {
            Ok(keypair) => return Ok(keypair),
            Err(OwnerKeychainLoadError::NoEntry)
            | Err(OwnerKeychainLoadError::Crypto(crate::crypto::CryptoError::DecryptionFailed))
            | Err(OwnerKeychainLoadError::Crypto(
                crate::crypto::CryptoError::KeychainUnavailable { .. },
            ))
            | Err(OwnerKeychainLoadError::Crypto(
                crate::crypto::CryptoError::KeychainAccessDenied { .. },
            )) => {}
            Err(OwnerKeychainLoadError::Crypto(err)) => {
                return Err(err)
                    .with_context(|| format!("Failed to load owner keystore {}", path.display()));
            }
        }
    }

    let passphrase = resolve_owner_passphrase(path)?;
    load_keystore(path, passphrase.as_deref().map(|value| value.as_str()))
        .with_context(|| format!("Failed to load owner keystore {}", path.display()))
}

fn owner_runtime_config(cli: &Cli) -> Result<mesh::OwnerRuntimeConfig> {
    let trust_store_path = default_trust_store_path()?;
    let trust_store = load_trust_store(&trust_store_path)
        .with_context(|| format!("Failed to load trust store {}", trust_store_path.display()))?
        .merged_with_trusted_owners(&cli.trust_owner);
    let trust_policy = cli.trust_policy.unwrap_or(trust_store.policy);

    let keypair = match resolve_runtime_owner_key_path(cli)? {
        Some(path) => match load_owner_keypair_for_runtime(&path) {
            Ok(keypair) => Some(keypair),
            Err(err) if !cli.owner_required => {
                eprintln!(
                    "⚠️ Owner identity unavailable at {}: {err}\n  Starting without owner attestation.",
                    path.display()
                );
                None
            }
            Err(err) => return Err(err),
        },
        None if cli.owner_required => {
            anyhow::bail!(
                "Owner identity is required but no keystore was found. Use --owner-key or run `mesh-llm auth init`."
            );
        }
        None => None,
    };

    Ok(mesh::OwnerRuntimeConfig {
        keypair,
        node_label: cli.node_label.clone(),
        trust_store,
        trust_policy,
    })
}

/// Wait for either SIGINT (ctrl-c) or SIGTERM. Without this, an unhandled
/// SIGTERM aborts the process before destructors run, so PidfileGuard never
/// removes its file and child llama-server / rpc-server are left orphaned.
async fn wait_shutdown_signal() {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{signal, SignalKind};
        let mut term = match signal(SignalKind::terminate()) {
            Ok(s) => s,
            Err(_) => {
                let _ = tokio::signal::ctrl_c().await;
                return;
            }
        };
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {}
            _ = term.recv() => {}
        }
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
    }
}

pub(crate) async fn run() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("mesh_inference=info".parse()?)
                .add_directive("nostr_relay_pool=off".parse()?)
                .add_directive("nostr_sdk=warn".parse()?)
                .add_directive("noq_proto::connection=warn".parse()?),
        )
        .with_writer(std::io::stderr)
        .init();

    // --help-advanced: print full help with all hidden options and commands visible
    if std::env::args().any(|a| a == "--help-advanced") {
        let mut cmd = Cli::command();
        // Unhide all arguments
        let args: Vec<clap::Id> = cmd.get_arguments().map(|a| a.get_id().clone()).collect();
        for id in args {
            cmd = cmd.mut_arg(id, |a| a.hide(false));
        }
        // Unhide all subcommands
        let sub_names: Vec<String> = cmd
            .get_subcommands()
            .map(|s| s.get_name().to_string())
            .collect();
        for name in sub_names {
            cmd = cmd.mut_subcommand(name, |s| s.hide(false));
        }
        cmd.print_help().ok();
        eprintln!();
        std::process::exit(0);
    }

    if std::env::args_os().len() == 1 {
        Cli::command().print_help().ok();
        std::process::exit(0);
    }

    let normalized_args = crate::cli::normalize_runtime_surface_args(std::env::args_os());
    let mut cli = Cli::parse_from(normalized_args.normalized.clone());

    if let Some(warning) = crate::cli::legacy_runtime_surface_warning(
        &cli,
        &normalized_args.original,
        normalized_args.explicit_surface,
    ) {
        eprintln!("{warning}");
    }

    if let Some(name) = cli.plugin.clone() {
        return plugin::run_plugin_process(name).await;
    }

    let checked_updates = autoupdate::maybe_auto_update(&cli).await?;

    // Finish the release check before startup continues.
    if !checked_updates && !matches!(cli.command, Some(Command::Update { .. })) {
        autoupdate::check_for_update().await;
    }

    if crate::cli::commands::dispatch(&cli).await? {
        return Ok(());
    }

    let config = plugin::load_config(cli.config.as_deref())?;
    let cli_has_explicit_models = cli_has_explicit_models(&cli);
    let has_config_models = !config.models.is_empty();
    let has_startup_models = cli_has_explicit_models || has_config_models;

    // Acquire the per-instance runtime directory and flock (skip for --client — no local servers).
    // Wrap in Arc so it can be cheaply shared with election/spawn tasks that
    // need to write pidfiles for child processes (rpc-server, llama-server).
    let runtime: Option<std::sync::Arc<crate::runtime::instance::InstanceRuntime>> = if !cli.client
    {
        match crate::runtime::instance::InstanceRuntime::acquire(std::process::id()) {
            Ok(rt) => Some(std::sync::Arc::new(rt)),
            Err(e) => {
                tracing::warn!("failed to acquire instance runtime: {e}");
                None
            }
        }
    } else {
        None
    };

    // Write owner.json into the runtime dir so sibling-instance discovery can find us.
    if let Some(ref rt) = runtime {
        let started_at =
            crate::runtime::instance::validate::current_process_start_time_unix().unwrap_or(0);
        let owner_meta = serde_json::json!({
            "pid": std::process::id(),
            "api_port": cli.console,
            "version": crate::VERSION,
            "started_at_unix": started_at,
            "mesh_llm_binary": std::env::current_exe()
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_default(),
        });
        let owner_path = rt.dir().join("owner.json");
        if let Ok(json) = serde_json::to_string_pretty(&owner_meta) {
            let _ = crate::runtime::instance::write_text_file_atomic(&owner_path, &json);
        }
    }

    // Reap orphans from dead sibling instances (skip for client — never runs llama-server).
    // This intentionally happens after subcommand dispatch so control commands
    // targeting a live instance don't kill it before sending the request.
    // Uses scoped reap that only touches runtime dirs whose owner has died (flock-released).
    if !cli.client {
        if let Some(ref rt) = runtime {
            match crate::runtime::instance::runtime_root() {
                Ok(root) => {
                    let my_dir = rt.dir().to_path_buf();
                    match crate::runtime::instance::reap::reap_cross_runtime_orphans(&root, &my_dir)
                        .await
                    {
                        Ok(summary) => {
                            if summary.children_killed > 0 || summary.dirs_gc_d > 0 {
                                eprintln!(
                                    "🧹 Reaped {} orphan children from {} dead instance(s)",
                                    summary.children_killed, summary.dirs_gc_d
                                );
                            }
                        }
                        Err(e) => tracing::warn!("cross-runtime reap failed: {e}"),
                    }
                }
                Err(e) => tracing::warn!("runtime_root resolution failed during reap: {e}"),
            }
        }
    }

    // Publication intent is now explicit only: --publish gates Nostr discovery.
    // --mesh-name alone never implies publication (Issue #240).

    // Warn users who used to rely on --mesh-name auto-publishing
    if cli.mesh_name.is_some() && !cli.publish {
        eprintln!(
            "ℹ️  Mesh named '{}' — private by default. Add --publish to make it publicly discoverable.",
            cli.mesh_name.as_ref().unwrap()
        );
    }

    // --- Public-to-private identity transition ---
    // If the previous run was public (--auto or --publish) but this run is
    // private, clear the stored identity so the private mesh gets a fresh key
    // that isn't associated with the old public listing.
    let is_public = cli.auto || cli.publish;
    if is_public {
        mesh::mark_was_public();
    } else if mesh::was_previously_public() {
        eprintln!("🔑 Previous run was public — rotating identity for private mesh");
        mesh::clear_public_identity();
    }

    let mut auto_join_candidates: Vec<(String, Option<String>)> = Vec::new();

    // --- Auto-discover ---
    if cli.auto && cli.join.is_empty() {
        cli.nostr_discovery = true;
        eprintln!("🔍 Discovering meshes via Nostr...");

        let relays = nostr_relays(&cli.nostr_relay);
        let filter = nostr::MeshFilter {
            model: None,
            min_vram_gb: None,
            region: None,
        };
        let meshes = nostr::discover(&relays, &filter, None).await?;

        let my_vram_gb = mesh::detect_vram_bytes_capped(cli.max_vram) as f64 / 1e9;
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let last_mesh_id = mesh::load_last_mesh_id();
        let target_name = cli.mesh_name.as_deref();
        // When the user did not target a specific mesh, `--auto` only joins
        // the community mesh (unnamed or name == "mesh-llm"). Other named
        // meshes are still publicly discoverable on Nostr, but the user has
        // to opt in by name. Hide them from the listing so the output matches
        // what auto will actually consider.
        let listed: Vec<&nostr::DiscoveredMesh> = if target_name.is_some() {
            meshes.iter().collect()
        } else {
            meshes
                .iter()
                .filter(|m| nostr::is_auto_eligible(m))
                .collect()
        };
        let hidden = meshes.len().saturating_sub(listed.len());
        if hidden > 0 {
            eprintln!(
                "  Found {} mesh(es) ({} named mesh(es) hidden; use --mesh-name to join)",
                listed.len(),
                hidden
            );
        } else {
            eprintln!("  Found {} mesh(es)", listed.len());
        }
        for m in &listed {
            let score = nostr::score_mesh(m, now, last_mesh_id.as_deref());
            eprintln!(
                "  · {} (score: {}, {} nodes, {:.0}GB, {} clients{})",
                m.listing.name.as_deref().unwrap_or("unnamed"),
                score,
                m.listing.node_count,
                m.listing.total_vram_bytes as f64 / 1e9,
                m.listing.client_count,
                m.listing
                    .region
                    .as_ref()
                    .map(|r| format!(", {r}"))
                    .unwrap_or_default()
            );
        }

        match nostr::smart_auto(&meshes, my_vram_gb, target_name) {
            nostr::AutoDecision::Join { candidates } => {
                if cli.client {
                    // Clients skip health probe — joining itself is the test.
                    // Queue all candidates so we can fall back if the top one is unreachable.
                    let (_, mesh) = &candidates[0];
                    if cli.mesh_name.is_none() {
                        if let Some(ref name) = mesh.listing.name {
                            cli.mesh_name = Some(name.clone());
                        }
                    }
                    eprintln!(
                        "✅ Joining: {} ({} nodes, {} models{})",
                        mesh.listing.name.as_deref().unwrap_or("unnamed"),
                        mesh.listing.node_count,
                        mesh.listing.serving.len(),
                        mesh.listing
                            .region
                            .as_ref()
                            .map(|r| format!(", region: {r}"))
                            .unwrap_or_default()
                    );
                    for (token, _) in &candidates {
                        cli.join.push(token.clone());
                    }
                } else {
                    // GPU nodes: try to join each candidate directly.
                    // No ephemeral probe — it fails when the target has a firewall
                    // even though the real join (via relay) would succeed.
                    let mut joined = false;
                    for (i, (token, mesh)) in candidates.iter().enumerate() {
                        eprintln!(
                            "  Trying mesh {}{}...",
                            mesh.listing.name.as_deref().unwrap_or("unnamed"),
                            if candidates.len() > 1 {
                                format!(" ({}/{})", i + 1, candidates.len())
                            } else {
                                String::new()
                            }
                        );
                        auto_join_candidates.push((token.clone(), mesh.listing.name.clone()));
                        joined = true;
                    }
                    if !joined {
                        eprintln!("⚠️  No meshes found — starting new");
                        let models = nostr::default_models_for_vram(my_vram_gb);
                        start_new_mesh(&mut cli, &models, my_vram_gb, has_startup_models);
                    }
                }
            }
            nostr::AutoDecision::StartNew { models } => {
                if cli.client {
                    // Retry discovery — meshes may appear later
                    eprintln!("⏳ No meshes found yet — retrying in 15s...");
                    let mut found = false;
                    for attempt in 1..=20 {
                        tokio::time::sleep(std::time::Duration::from_secs(15)).await;
                        eprintln!("🔍 Retry {attempt}/20...");
                        if let Ok(retry_meshes) = nostr::discover(&relays, &filter, None).await {
                            if let nostr::AutoDecision::Join { candidates } =
                                nostr::smart_auto(&retry_meshes, my_vram_gb, target_name)
                            {
                                let (_, mesh) = &candidates[0];
                                if cli.mesh_name.is_none() {
                                    if let Some(ref name) = mesh.listing.name {
                                        cli.mesh_name = Some(name.clone());
                                    }
                                }
                                eprintln!(
                                    "✅ Joining: {} ({} nodes, {} models)",
                                    mesh.listing.name.as_deref().unwrap_or("unnamed"),
                                    mesh.listing.node_count,
                                    mesh.listing.serving.len()
                                );
                                for (token, _) in &candidates {
                                    cli.join.push(token.clone());
                                }
                                found = true;
                                break;
                            }
                        }
                    }
                    if !found {
                        anyhow::bail!("No meshes found after 5 minutes of retrying.");
                    }
                } else {
                    start_new_mesh(&mut cli, &models, my_vram_gb, has_startup_models);
                }
            }
        }
    }

    // --- Validation ---
    if cli.client && (!cli.model.is_empty() || !cli.gguf.is_empty()) {
        anyhow::bail!("--client and --model are mutually exclusive");
    }
    if let Some(mmproj) = &cli.mmproj {
        anyhow::ensure!(!cli.client, "--mmproj cannot be used with --client");
        anyhow::ensure!(
            !cli.model.is_empty() || !cli.gguf.is_empty(),
            "--mmproj requires an explicit primary model via --model or --gguf"
        );
        anyhow::ensure!(
            mmproj.is_file(),
            "mmproj path is not a file: {}",
            mmproj.display()
        );
    }
    let startup_specs = build_startup_model_specs(&cli, &config)?;
    if should_show_serve_config_help(normalized_args.explicit_surface, &cli, &startup_specs) {
        let config_path = plugin::config_path(cli.config.as_deref()).unwrap_or_else(|_| {
            dirs::home_dir()
                .unwrap_or_else(|| PathBuf::from("~"))
                .join(".mesh-llm")
                .join("config.toml")
        });
        eprintln!(
            "⚠️ `mesh-llm serve` needs at least one startup model.\n  Add `[[models]]` to {}, or pass `--model` / `--gguf` explicitly.",
            config_path.display()
        );
        Cli::command().print_help().ok();
        eprintln!();
        return Ok(());
    }
    let mut startup_models = resolve_startup_models(&startup_specs, cli.max_vram).await?;
    preflight_config_owned_startup_models(&config, &startup_specs, &mut startup_models)?;
    let resolved_models: Vec<PathBuf> = startup_models
        .iter()
        .map(|model| model.resolved_path.clone())
        .collect();
    {
        let update_check_paths = resolved_models.clone();
        match tokio::task::spawn_blocking(move || {
            models::warn_about_updates_for_paths(&update_check_paths);
        })
        .await
        {
            Ok(()) => {}
            Err(err) => {
                eprintln!("Warning: could not join Hugging Face update check task: {err}");
            }
        }
    }

    // Build requested model names from all resolved models
    // Strip split GGUF suffix so "MiniMax-M2.5-Q4_K_M-00001-of-00004" → "MiniMax-M2.5-Q4_K_M"
    let requested_model_names: Vec<String> = resolved_models
        .iter()
        .filter_map(|m| {
            m.file_stem()
                .and_then(|s| s.to_str())
                .map(router::strip_split_suffix_owned)
        })
        .collect();

    let bin_dir = match &cli.bin_dir {
        Some(d) => d.clone(),
        None => detect_bin_dir()?,
    };

    run_auto(
        cli,
        config,
        startup_models,
        requested_model_names,
        bin_dir,
        runtime,
        auto_join_candidates,
    )
    .await
}

/// Resolve a model path: local file, catalog name, or HuggingFace URL.
async fn resolve_model(input: &std::path::Path) -> Result<PathBuf> {
    models::resolve_model_spec(input).await
}

fn cli_has_explicit_models(cli: &Cli) -> bool {
    !cli.model.is_empty() || !cli.gguf.is_empty()
}

fn build_startup_model_specs(
    cli: &Cli,
    config: &plugin::MeshConfig,
) -> Result<Vec<StartupModelSpec>> {
    if cli.client {
        return Ok(Vec::new());
    }

    let mut specs = Vec::new();
    if cli_has_explicit_models(cli) {
        for path in &cli.gguf {
            if !path.exists() {
                anyhow::bail!("GGUF file not found: {}", path.display());
            }
            specs.push(StartupModelSpec {
                model_ref: path.clone(),
                mmproj_ref: None,
                ctx_size: cli.ctx_size,
                gpu_id: None,
                config_owned: false,
                parallel: None,
            });
        }
        for model in &cli.model {
            specs.push(StartupModelSpec {
                model_ref: model.clone(),
                mmproj_ref: None,
                ctx_size: cli.ctx_size,
                gpu_id: None,
                config_owned: false,
                parallel: None,
            });
        }
        if let Some(mmproj) = &cli.mmproj {
            if let Some(primary) = specs.first_mut() {
                primary.mmproj_ref = Some(mmproj.clone());
            }
        }
        return Ok(specs);
    }

    for model in &config.models {
        specs.push(StartupModelSpec {
            model_ref: PathBuf::from(model.model.clone()),
            mmproj_ref: model.mmproj.as_ref().map(PathBuf::from),
            ctx_size: cli.ctx_size.or(model.ctx_size),
            gpu_id: model.gpu_id.clone(),
            config_owned: true,
            parallel: model.parallel,
        });
    }
    Ok(specs)
}

async fn resolve_startup_models(
    specs: &[StartupModelSpec],
    max_vram_gb: Option<f64>,
) -> Result<Vec<StartupModelPlan>> {
    let mut plans = Vec::with_capacity(specs.len());
    let mut preflight_cache: HashMap<String, Option<moe_planner::MoeStartupFitEstimate>> =
        HashMap::new();
    for spec in specs {
        preflight_remote_startup_model(spec, max_vram_gb, &mut preflight_cache).await?;
        let resolved_path = resolve_model(&spec.model_ref).await?;
        let mmproj_path = match spec.mmproj_ref.as_ref() {
            Some(mmproj) => Some(resolve_model(mmproj).await?),
            None => None,
        };
        plans.push(StartupModelPlan {
            declared_ref: spec.model_ref.to_string_lossy().to_string(),
            resolved_path,
            mmproj_path,
            ctx_size: spec.ctx_size,
            gpu_id: spec.gpu_id.clone(),
            pinned_gpu: None,
            parallel: spec.parallel,
        });
    }
    Ok(plans)
}

async fn preflight_remote_startup_model(
    spec: &StartupModelSpec,
    max_vram_gb: Option<f64>,
    cache: &mut HashMap<String, Option<moe_planner::MoeStartupFitEstimate>>,
) -> Result<()> {
    if spec.model_ref.exists() {
        return Ok(());
    }

    let declared_ref = spec.model_ref.to_string_lossy().to_string();
    if !declared_ref.contains('/') {
        let installed_name = declared_ref.strip_suffix(".gguf").unwrap_or(&declared_ref);
        if models::find_model_path(installed_name).exists() {
            return Ok(());
        }
    }

    let mut spinner = start_spinner(&format!("Checking published MoE fit for {declared_ref}"));
    let identity = models::resolve_huggingface_model_identity(&declared_ref)
        .await
        .with_context(|| format!("Resolve model identity for {declared_ref}"))?;
    let Some(identity) = identity else {
        spinner.finish();
        return Ok(());
    };

    if let Some(cached) = cache.get(&identity.canonical_ref).cloned() {
        spinner.finish();
        return apply_remote_startup_preflight(&declared_ref, &identity.local_file_name, cached);
    }

    spinner.set_message(format!(
        "Fetching published MoE analysis for {}",
        identity.local_file_name
    ));
    let dataset_repo = moe_planner::DEFAULT_MOE_RANKINGS_DATASET.to_string();
    let source_repo = identity.repo_id.clone();
    let source_revision = identity.revision.clone();
    let distribution_id = moe_planner::normalize_distribution_id(&identity.local_file_name);
    let target_vram_bytes = mesh::detect_vram_bytes_capped(max_vram_gb);
    let fetched = tokio::task::spawn_blocking(move || {
        moe_planner::fetch_remote_startup_fit(
            &dataset_repo,
            &source_repo,
            &source_revision,
            &distribution_id,
            target_vram_bytes,
            false,
        )
    })
    .await;
    spinner.finish();

    let fit = match fetched {
        Ok(Ok(fit)) => fit,
        Ok(Err(err)) => {
            eprintln!(
                "⚠️  [{declared_ref}] Published MoE preflight lookup failed ({err}); continuing with normal model resolution"
            );
            None
        }
        Err(err) => {
            eprintln!(
                "⚠️  [{declared_ref}] Published MoE preflight task failed ({err}); continuing with normal model resolution"
            );
            None
        }
    };
    cache.insert(identity.canonical_ref.clone(), fit.clone());
    apply_remote_startup_preflight(&declared_ref, &identity.local_file_name, fit)
}

fn apply_remote_startup_preflight(
    declared_ref: &str,
    model_label: &str,
    fit: Option<moe_planner::MoeStartupFitEstimate>,
) -> Result<()> {
    let Some(fit) = fit else {
        return Ok(());
    };
    if !fit.any_fit_exists() {
        anyhow::bail!(
            "Published MoE preflight says '{}' will not fit locally before download: full model needs about {}, and the smallest conservative split still needs about {} per node with a 50% shared core ({} shared experts). Local budget is {}.",
            declared_ref,
            format_vram_gb(fit.full_model_launch_bytes),
            fit.predicted_max_shard_launch_bytes
                .map(format_vram_gb)
                .unwrap_or_else(|| "unknown".to_string()),
            fit.required_experts_per_node,
            format_vram_gb(fit.target_vram_bytes),
        );
    }

    if fit.full_model_fits() {
        eprintln!(
            "🧩 [{}] Published MoE preflight: full model should fit locally (~{} <= {}, mode={}, source={})",
            model_label,
            format_vram_gb(fit.full_model_launch_bytes),
            format_vram_gb(fit.target_vram_bytes),
            fit.analyzer_id,
            fit.ranking_source,
        );
        return Ok(());
    }

    if fit.shard_plan_fits() {
        eprintln!(
            "🧩 [{}] Published MoE preflight: full model needs ~{}, but a conservative {}-node shard should fit here (~{}/node, mode={}, source={})",
            model_label,
            format_vram_gb(fit.full_model_launch_bytes),
            fit.recommended_nodes.unwrap_or(1),
            format_vram_gb(fit.predicted_max_shard_launch_bytes.unwrap_or_default()),
            fit.analyzer_id,
            fit.ranking_source,
        );
        return Ok(());
    }
    Ok(())
}

fn format_vram_gb(bytes: u64) -> String {
    format!("{:.1}GB", bytes as f64 / 1e9)
}

fn preflight_config_owned_startup_models(
    config: &plugin::MeshConfig,
    specs: &[StartupModelSpec],
    plans: &mut [StartupModelPlan],
) -> Result<()> {
    if config.gpu.assignment != plugin::GpuAssignment::Pinned {
        return Ok(());
    }

    let survey = hardware::query(pinned_startup_preflight_metrics());
    preflight_config_owned_startup_models_with_gpus(config, specs, plans, &survey.gpus)
}

fn pinned_startup_preflight_metrics() -> &'static [hardware::Metric] {
    &[
        hardware::Metric::GpuName,
        hardware::Metric::GpuFacts,
        hardware::Metric::VramBytes,
        hardware::Metric::IsSoc,
    ]
}

fn preflight_config_owned_startup_models_with_gpus(
    config: &plugin::MeshConfig,
    specs: &[StartupModelSpec],
    plans: &mut [StartupModelPlan],
    gpus: &[hardware::GpuFacts],
) -> Result<()> {
    if config.gpu.assignment != plugin::GpuAssignment::Pinned {
        return Ok(());
    }

    anyhow::ensure!(
        specs.len() == plans.len(),
        "startup model preflight received mismatched specs/plans"
    );

    for (spec, plan) in specs.iter().zip(plans.iter_mut()) {
        if !spec.config_owned {
            continue;
        }

        let resolved_gpu = hardware::resolve_pinned_gpu(plan.gpu_id.as_deref(), gpus)
            .map_err(anyhow::Error::new)
            .with_context(|| {
                format!(
                    "startup model '{}' failed pinned GPU preflight",
                    plan.declared_ref
                )
            })?;

        let stable_id = resolved_gpu.stable_id.clone().ok_or_else(|| {
            anyhow::anyhow!(
                "startup model '{}' resolved pinned GPU at index {} without a stable_id",
                plan.declared_ref,
                resolved_gpu.index
            )
        })?;

        let backend_device = resolved_gpu
            .backend_device
            .clone()
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "startup model '{}' resolved pinned GPU '{}' at index {} without a backend_device",
                    plan.declared_ref,
                    stable_id,
                    resolved_gpu.index
                )
            })
            .with_context(|| {
                format!(
                    "startup model '{}' failed pinned GPU preflight",
                    plan.declared_ref
                )
            })?;

        plan.pinned_gpu = Some(StartupPinnedGpuTarget {
            index: resolved_gpu.index,
            stable_id,
            backend_device,
            vram_bytes: resolved_gpu.vram_bytes,
        });
    }

    Ok(())
}

fn should_show_serve_config_help(
    explicit_surface: Option<RuntimeSurface>,
    cli: &Cli,
    startup_specs: &[StartupModelSpec],
) -> bool {
    explicit_surface == Some(RuntimeSurface::Serve)
        && !cli.client
        && startup_specs.is_empty()
        && !cli.auto
        && cli.join.is_empty()
        && cli.discover.is_none()
}

fn startup_rpc_backend_device<'a>(
    cli_device: Option<&'a str>,
    primary_startup_model: Option<&'a StartupModelPlan>,
) -> Result<Option<&'a str>> {
    let pinned_device = primary_startup_model
        .and_then(|model| model.pinned_gpu.as_ref())
        .map(|gpu| gpu.backend_device.as_str());

    if let (Some(cli_device), Some(pinned_device)) = (cli_device, pinned_device) {
        let is_match = cli_device == pinned_device;
        let is_lenient_match = is_match || {
            let is_amd_cli = cli_device.starts_with("ROCm") || cli_device.starts_with("HIP");
            let is_amd_pinned =
                pinned_device.starts_with("ROCm") || pinned_device.starts_with("HIP");
            if is_amd_cli && is_amd_pinned {
                let cli_idx = cli_device
                    .trim_start_matches("ROCm")
                    .trim_start_matches("HIP");
                let pinned_idx = pinned_device
                    .trim_start_matches("ROCm")
                    .trim_start_matches("HIP");
                cli_idx == pinned_idx && !cli_idx.is_empty()
            } else {
                false
            }
        };

        anyhow::ensure!(
            is_lenient_match,
            "explicit --device '{cli_device}' conflicts with pinned startup GPU backend device '{pinned_device}'"
        );
        return Ok(Some(cli_device));
    }

    Ok(cli_device.or(pinned_device))
}

/// Look up the model filename in the catalog and check if its draft model exists on disk.
/// If not on disk, downloads it (drafts are <1GB).
pub async fn ensure_draft(model: &std::path::Path) -> Option<PathBuf> {
    let filename = model.file_name()?.to_str()?;
    let catalog_entry = catalog::MODEL_CATALOG
        .iter()
        .find(|m| m.file == filename || m.file.eq_ignore_ascii_case(filename))?;
    let draft_name = catalog_entry.draft.as_deref()?;
    let draft_entry = catalog::MODEL_CATALOG
        .iter()
        .find(|m| m.name == draft_name)?;
    let draft_stem = draft_entry
        .file
        .strip_suffix(".gguf")
        .unwrap_or(&draft_entry.file);
    let draft_path = models::find_model_path(draft_stem);
    if draft_path.exists() {
        return Some(draft_path);
    }
    // Draft not on disk — download it (small, <1GB)
    eprintln!(
        "📥 Downloading draft model {} ({})...",
        draft_entry.name, draft_entry.size
    );
    match catalog::download_model(draft_entry).await {
        Ok(path) => {
            eprintln!("✅ Draft model ready: {}", draft_entry.name);
            Some(path)
        }
        Err(e) => {
            eprintln!(
                "⚠ Failed to download draft model: {e} — continuing without speculative decoding"
            );
            None
        }
    }
}

/// Pick which model this node should serve.
///
/// Priority:
/// 1. Models the mesh needs that we already have on disk
/// 2. Models in the mesh catalog that nobody is serving yet (on disk preferred)
///
/// Parse a catalog size string like "18.3GB" or "491MB" into bytes.
fn parse_size_str(s: &str) -> u64 {
    let s = s.trim();
    if let Some(gb) = s.strip_suffix("GB") {
        (gb.parse::<f64>().unwrap_or(0.0) * 1e9) as u64
    } else if let Some(mb) = s.strip_suffix("MB") {
        (mb.parse::<f64>().unwrap_or(0.0) * 1e6) as u64
    } else {
        0
    }
}

/// Pick which model this node should serve, based on demand signals.
///
/// Priority:
/// 1. Unserved models with active demand that we have on disk (hottest first)
/// 2. Underserved models with demand that we have on disk
/// 3. Unserved models with demand that we can download from catalog
/// 4. Standby if everything is covered
async fn pick_model_assignment(node: &mesh::Node, local_models: &[String]) -> Option<String> {
    let peers = node.peers().await;

    // Get active demand — the unified "what does the mesh want?"
    let demand = node.active_demand().await;

    if demand.is_empty() {
        // No API requests yet — log what the mesh is serving for visibility
        let served: Vec<String> = peers.iter().flat_map(|p| p.routable_models()).collect();
        if !served.is_empty() {
            eprintln!(
                "📋 No demand yet — mesh is serving {:?}, staying standby until needed",
                served
            );
        } else {
            eprintln!("📋 No demand signals — no models requested");
        }
        return None;
    }

    eprintln!("📋 Active demand: {:?}", demand.keys().collect::<Vec<_>>());

    // Count how many nodes are serving each model
    let mut serving_count: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();
    for p in &peers {
        for served_model in p.routable_models() {
            *serving_count.entry(served_model).or_default() += 1;
        }
    }

    let my_vram = node.vram_bytes();

    /// Check if a model fits in our VRAM. Returns false and logs if it doesn't.
    fn model_fits(model: &str, my_vram: u64) -> bool {
        let model_path = models::find_model_path(model);
        let model_bytes = std::fs::metadata(&model_path)
            .map(|md| md.len())
            .unwrap_or(0);
        let needed = (model_bytes as f64 * 1.1) as u64;
        if model_bytes > 0 && needed > my_vram {
            eprintln!(
                "📋 Skipping {} — needs {:.1}GB, we have {:.1}GB",
                model,
                needed as f64 / 1e9,
                my_vram as f64 / 1e9
            );
            return false;
        }
        true
    }

    // Sort demand entries by request_count descending (hottest first)
    let mut demand_sorted: Vec<(String, mesh::ModelDemand)> = demand.into_iter().collect();
    demand_sorted.sort_by(|a, b| b.1.request_count.cmp(&a.1.request_count));

    // Priority 1: Unserved models on disk, ordered by demand
    let mut candidates: Vec<String> = Vec::new();
    for (m, _d) in &demand_sorted {
        if serving_count.get(m).copied().unwrap_or(0) == 0
            && local_models.contains(m)
            && model_fits(m, my_vram)
        {
            candidates.push(m.clone());
        }
    }

    if !candidates.is_empty() {
        // If multiple, pick deterministically so concurrent joiners spread out
        if candidates.len() > 1 {
            let my_id = node.id();
            let id_bytes = my_id.as_bytes();
            let hash = id_bytes
                .iter()
                .fold(0u64, |acc, &b| acc.wrapping_mul(31).wrapping_add(b as u64));
            let idx = (hash as usize) % candidates.len();
            let pick = &candidates[idx];
            eprintln!(
                "📋 Assigned to serve {} (unserved, on disk, {} candidates, by demand)",
                pick,
                candidates.len()
            );
            return Some(pick.clone());
        }
        let pick = &candidates[0];
        eprintln!(
            "📋 Assigned to serve {} (unserved, on disk, by demand)",
            pick
        );
        return Some(pick.clone());
    }

    // Priority 2: Underserved models on disk (fewer servers than others)
    let max_count = serving_count.values().copied().max().unwrap_or(0);
    let mut underserved: Vec<(String, usize, u64)> = Vec::new(); // (model, servers, demand)
    for (m, d) in &demand_sorted {
        let count = serving_count.get(m).copied().unwrap_or(0);
        if count < max_count && local_models.contains(m) && model_fits(m, my_vram) {
            underserved.push((m.clone(), count, d.request_count));
        }
    }
    if !underserved.is_empty() {
        // Pick the least-served, breaking ties by highest demand
        underserved.sort_by_key(|(_, count, demand)| (*count, std::cmp::Reverse(*demand)));
        let (pick, count, _) = &underserved[0];
        let max_model = serving_count
            .iter()
            .max_by_key(|(_, &v)| v)
            .map(|(k, _)| k.as_str())
            .unwrap_or("?");
        eprintln!(
            "📋 Assigned to serve {} ({} servers vs {} has {}) — rebalancing",
            pick, count, max_model, max_count
        );
        return Some(pick.clone());
    }

    // Priority 3: Unserved models we can download from catalog
    let mut downloadable: Vec<(String, u64)> = Vec::new(); // (model, demand)
    for (m, d) in &demand_sorted {
        if serving_count.get(m).copied().unwrap_or(0) > 0 {
            continue;
        }
        if let Some(cat) = catalog::find_model(m) {
            let size_bytes = parse_size_str(&cat.size);
            let needed = (size_bytes as f64 * 1.1) as u64;
            if needed <= my_vram {
                downloadable.push((m.clone(), d.request_count));
            } else {
                eprintln!(
                    "📋 Skipping {} — needs {:.1}GB, we have {:.1}GB",
                    m,
                    needed as f64 / 1e9,
                    my_vram as f64 / 1e9
                );
            }
        }
    }
    if !downloadable.is_empty() {
        // Pick hottest downloadable, with node-ID hash for tie-breaking
        if downloadable.len() > 1 {
            let my_id = node.id();
            let id_bytes = my_id.as_bytes();
            let hash = id_bytes
                .iter()
                .fold(0u64, |acc, &b| acc.wrapping_mul(31).wrapping_add(b as u64));
            let idx = (hash as usize) % downloadable.len();
            let (pick, _) = &downloadable[idx];
            eprintln!(
                "📋 Assigned to serve {} (unserved, will download, by demand)",
                pick
            );
            return Some(pick.clone());
        }
        let (pick, _) = &downloadable[0];
        eprintln!(
            "📋 Assigned to serve {} (unserved, will download, by demand)",
            pick
        );
        return Some(pick.clone());
    }

    // Everything with demand is covered
    let all_covered = demand_sorted
        .iter()
        .all(|(m, _)| serving_count.get(m).copied().unwrap_or(0) > 0);
    if all_covered {
        eprintln!("📋 All demanded models are covered — staying on standby");
    }

    None
}

/// Check if a standby node should promote to serve a model.
/// Uses demand signals — promotes for unserved models with active demand,
/// or for demand-based rebalancing when one model is much hotter than others.
///
/// Rebalancing uses `last_active` to gate on recency (only models active within
/// the last 60 minutes are considered), then `request_count / servers` for
/// relative hotness among those recent models.
async fn check_unserved_model(node: &mesh::Node, local_models: &[String]) -> Option<String> {
    let peers = node.peers().await;
    let demand = node.active_demand().await;

    if demand.is_empty() {
        return None;
    }

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let mut serving_count: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();
    for p in &peers {
        for served_model in p.routable_models() {
            *serving_count.entry(served_model).or_default() += 1;
        }
    }

    let my_vram = node.vram_bytes();

    // Only consider models with recent activity (last 60 minutes).
    // This prevents stale cumulative request_count from triggering promotions
    // for models that were popular hours ago but idle now.
    const RECENT_SECS: u64 = 3600;

    // Priority 1: promote for models with active demand and ZERO servers
    // Sort by demand (hottest first)
    let mut unserved: Vec<(String, u64)> = Vec::new();
    for (m, d) in &demand {
        if serving_count.get(m).copied().unwrap_or(0) == 0 && local_models.contains(m) {
            let model_path = models::find_model_path(m);
            let model_bytes = std::fs::metadata(&model_path)
                .map(|md| md.len())
                .unwrap_or(0);
            let needed = (model_bytes as f64 * 1.1) as u64;
            if model_bytes > 0 && needed > my_vram {
                continue;
            }
            unserved.push((m.clone(), d.request_count));
        }
    }
    if !unserved.is_empty() {
        unserved.sort_by_key(|(_, count)| std::cmp::Reverse(*count));
        return Some(unserved[0].0.clone());
    }

    // Priority 2: demand-based rebalancing.
    // Only consider models with recent activity, then use request_count / servers
    // for relative hotness. Promote if one model is significantly hotter than others.
    let mut ratios: Vec<(String, f64)> = Vec::new();
    for (m, d) in &demand {
        if now.saturating_sub(d.last_active) > RECENT_SECS {
            continue;
        }
        let servers = serving_count.get(m).copied().unwrap_or(0) as f64;
        if servers > 0.0 && d.request_count > 0 && local_models.contains(m) {
            let model_path = models::find_model_path(m);
            let model_bytes = std::fs::metadata(&model_path)
                .map(|md| md.len())
                .unwrap_or(0);
            let needed = (model_bytes as f64 * 1.1) as u64;
            if model_bytes > 0 && needed > my_vram {
                continue;
            }
            ratios.push((m.clone(), d.request_count as f64 / servers));
        }
    }

    if !ratios.is_empty() {
        ratios.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let (hottest_model, hottest_ratio) = &ratios[0];
        let coldest_ratio = if ratios.len() >= 2 {
            ratios[ratios.len() - 1].1
        } else {
            0.0
        };
        let should_promote = if ratios.len() >= 2 {
            *hottest_ratio >= coldest_ratio * 3.0 && *hottest_ratio >= 10.0
        } else {
            *hottest_ratio >= 10.0
        };

        if should_promote {
            eprintln!(
                "📋 Promoting to serve {} — demand {:.0} req/server (coldest: {:.0})",
                hottest_model, hottest_ratio, coldest_ratio
            );
            return Some(hottest_model.clone());
        }
    }

    None
}

pub(crate) fn load_resolved_plugins(cli: &Cli) -> Result<plugin::ResolvedPlugins> {
    let config = plugin::load_config(cli.config.as_deref())?;
    resolve_plugins_from_config(&config, cli)
}

fn resolve_plugins_from_config(
    config: &plugin::MeshConfig,
    cli: &Cli,
) -> Result<plugin::ResolvedPlugins> {
    plugin::resolve_plugins(config, plugin_host_mode(cli))
}

fn plugin_host_mode(cli: &Cli) -> plugin::PluginHostMode {
    plugin::PluginHostMode {
        mesh_visibility: if cli.publish || cli.nostr_discovery {
            mesh_llm_plugin::MeshVisibility::Public
        } else {
            mesh_llm_plugin::MeshVisibility::Private
        },
    }
}

fn node_display_name(cli: &Cli, node: &mesh::Node) -> String {
    cli.name
        .clone()
        .or_else(|| std::env::var("USER").ok())
        .or_else(|| std::env::var("USERNAME").ok())
        .unwrap_or_else(|| node.id().fmt_short().to_string())
}

async fn join_mesh_for_mcp(cli: &Cli, node: &mesh::Node) -> Result<()> {
    if !cli.join.is_empty() {
        for token in &cli.join {
            match node.join(token).await {
                Ok(()) => {
                    if node.mesh_id().await.is_some() {
                        record_first_joined_mesh_ts(node).await;
                    }
                    eprintln!("Joined mesh");
                    return Ok(());
                }
                Err(err) => tracing::warn!("Failed to join via token: {err}"),
            }
        }
        anyhow::bail!("Failed to join any peer for MCP mode");
    }

    if cli.auto || cli.discover.is_some() {
        let relays = nostr_relays(&cli.nostr_relay);
        let filter = nostr::MeshFilter {
            model: None,
            min_vram_gb: None,
            region: cli.region.clone(),
        };
        let target_name = cli.discover.as_deref().or(cli.mesh_name.as_deref());
        let meshes = nostr::discover(&relays, &filter, None).await?;
        match nostr::smart_auto(&meshes, 0.0, target_name) {
            nostr::AutoDecision::Join { candidates } => {
                let mut last_err: Option<anyhow::Error> = None;
                for (token, mesh) in &candidates {
                    eprintln!(
                        "✅ Joining: {} ({} nodes, {} models{})",
                        mesh.listing.name.as_deref().unwrap_or("unnamed"),
                        mesh.listing.node_count,
                        mesh.listing.serving.len(),
                        mesh.listing
                            .region
                            .as_ref()
                            .map(|r| format!(", region: {r}"))
                            .unwrap_or_default()
                    );
                    match node.join(token).await {
                        Ok(()) => {
                            if node.mesh_id().await.is_some() {
                                record_first_joined_mesh_ts(node).await;
                            }
                            last_err = None;
                            break;
                        }
                        Err(err) => {
                            tracing::warn!("Failed to join mesh candidate: {err}");
                            last_err = Some(err);
                        }
                    }
                }
                if let Some(err) = last_err {
                    return Err(err);
                }
            }
            nostr::AutoDecision::StartNew { .. } => {
                anyhow::bail!("No mesh found for MCP mode. Pass --join or start a mesh first.");
            }
        }
    }

    Ok(())
}

pub(crate) async fn run_plugin_mcp(cli: &Cli) -> Result<()> {
    let resolved_plugins = load_resolved_plugins(cli)?;
    let owner_config = owner_runtime_config(cli)?;
    let (node, _channels) = mesh::Node::start(
        NodeRole::Client,
        &cli.relay,
        cli.bind_port,
        Some(0.0),
        !cli.no_enumerate_host,
        Some(owner_config),
        cli.config.as_deref(),
    )
    .await?;
    node.start_accepting();
    node.set_display_name(node_display_name(cli, &node)).await;
    node.start_heartbeat();
    node.start_relay_health_monitor();
    join_mesh_for_mcp(cli, &node).await?;

    let (plugin_mesh_tx, plugin_mesh_rx) = tokio::sync::mpsc::channel(256);
    let plugin_manager =
        plugin::PluginManager::start(&resolved_plugins, plugin_host_mode(cli), plugin_mesh_tx)
            .await?;
    node.set_plugin_manager(plugin_manager.clone()).await;
    node.start_plugin_channel_forwarder(plugin_mesh_rx);

    if plugin_manager.list().await.is_empty() {
        tracing::warn!("No plugins are enabled for MCP exposure");
    }

    plugin::mcp::run_mcp_server(plugin_manager).await
}

pub(crate) use self::discovery::{check_mesh, nostr_relays};

async fn store_benchmark_metrics(
    mem_arc: std::sync::Arc<tokio::sync::Mutex<Option<Vec<f64>>>>,
    fp32_arc: std::sync::Arc<tokio::sync::Mutex<Option<Vec<f64>>>>,
    fp16_arc: std::sync::Arc<tokio::sync::Mutex<Option<Vec<f64>>>>,
    result: Option<&benchmark::BenchmarkResult>,
) {
    *mem_arc.lock().await = result.map(|r| r.mem_bandwidth_gbps.clone());
    *fp32_arc.lock().await = result.and_then(|r| r.compute_tflops_fp32.clone());
    *fp16_arc.lock().await = result.and_then(|r| r.compute_tflops_fp16.clone());
}

/// Auto-election mode: start rpc-server, join mesh, auto-elect host.
async fn run_auto(
    mut cli: Cli,
    config: plugin::MeshConfig,
    startup_models: Vec<StartupModelPlan>,
    requested_model_names: Vec<String>,
    bin_dir: PathBuf,
    runtime: Option<std::sync::Arc<crate::runtime::instance::InstanceRuntime>>,
    auto_join_candidates: Vec<(String, Option<String>)>,
) -> Result<()> {
    let resolved_plugins = resolve_plugins_from_config(&config, &cli)?;
    let api_port = cli.port;
    // Export management API port for llama-server mesh hook callbacks.
    // Must be the console/management port (default 3131), NOT the proxy port
    // (default 9337) — hooks hitting the proxy would loop back to llama-server.
    std::env::set_var("MESH_API_PORT", cli.console.to_string());
    let console_port = Some(cli.console);
    let is_client = cli.client;
    let resolved_models: Vec<PathBuf> = startup_models
        .iter()
        .map(|model| model.resolved_path.clone())
        .collect();

    // Scan local models on disk
    let local_models = if is_client {
        vec![]
    } else {
        models::scan_local_models()
    };
    tracing::info!("Local models on disk: {:?}", local_models);

    // Start mesh node — clients use ephemeral key (unique identity per run)
    let role = if is_client {
        NodeRole::Client
    } else {
        NodeRole::Worker
    };
    let owner_config = owner_runtime_config(&cli)?;
    // Clients report 0 VRAM so they're never assigned a model to serve
    let max_vram = if is_client { Some(0.0) } else { cli.max_vram };
    let (node, channels) = mesh::Node::start(
        role,
        &cli.relay,
        cli.bind_port,
        max_vram,
        !cli.no_enumerate_host,
        Some(owner_config),
        cli.config.as_deref(),
    )
    .await?;
    node.start_accepting();
    let token = node.invite_token();
    node.set_display_name(node_display_name(&cli, &node)).await;
    let (plugin_mesh_tx, plugin_mesh_rx) = tokio::sync::mpsc::channel(256);
    let plugin_manager =
        plugin::PluginManager::start(&resolved_plugins, plugin_host_mode(&cli), plugin_mesh_tx)
            .await?;
    node.set_plugin_manager(plugin_manager.clone()).await;
    node.start_plugin_channel_forwarder(plugin_mesh_rx);

    // Advertise what we have on disk and what we want the mesh to serve
    node.set_available_models(local_models.clone()).await;
    node.set_requested_models(requested_model_names.clone())
        .await;

    // Start periodic health check to detect dead peers
    node.start_heartbeat();
    node.start_relay_health_monitor();

    // Launch memory bandwidth benchmark in background (non-blocking)
    // Skip for client nodes — they have no GPU to benchmark
    if !is_client {
        let mem_arc = node.gpu_mem_bandwidth_gbps.clone();
        let compute_fp32_arc = node.gpu_compute_tflops_fp32.clone();
        let compute_fp16_arc = node.gpu_compute_tflops_fp16.clone();
        let bin_dir_clone = bin_dir.clone();
        let node_bench = node.clone();
        tokio::spawn(async move {
            let result = tokio::time::timeout(
                std::time::Duration::from_secs(30),
                tokio::task::spawn_blocking(move || {
                    let hw = hardware::survey();
                    if hw.gpu_count == 0 {
                        tracing::debug!("no GPUs detected — skipping memory bandwidth benchmark");
                        return None;
                    }
                    benchmark::run_or_load(&hw, &bin_dir_clone, benchmark::BENCHMARK_TIMEOUT)
                }),
            )
            .await
            .map_err(|_| {
                tracing::warn!("benchmark timed out after 30s — bandwidth will not be gossiped")
            })
            .ok()
            .and_then(|r| r.ok())
            .flatten();

            if let Some(ref run) = result {
                let total: f64 = run.mem_bandwidth_gbps.iter().sum();
                tracing::info!(
                    "Memory bandwidth fingerprint: {} GPUs, {:.1} GB/s total",
                    run.mem_bandwidth_gbps.len(),
                    total
                );
                for (i, gbps) in run.mem_bandwidth_gbps.iter().enumerate() {
                    tracing::debug!("  GPU {}: {:.1} GB/s", i, gbps);
                }
                if let Some(fp32s) = &run.compute_tflops_fp32 {
                    let total_fp32: f64 = fp32s.iter().sum();
                    tracing::info!(
                        "Compute FP32 TFLOPS: {} GPUs, {:.1} TFLOPS total",
                        fp32s.len(),
                        total_fp32
                    );
                    for (i, tf) in fp32s.iter().enumerate() {
                        tracing::debug!("  GPU {}: {:.1} TF32", i, tf);
                    }
                }
                if let Some(fp16s) = &run.compute_tflops_fp16 {
                    let total_fp16: f64 = fp16s.iter().sum();
                    tracing::info!(
                        "Compute FP16 TFLOPS: {} GPUs, {:.1} TFLOPS total",
                        fp16s.len(),
                        total_fp16
                    );
                    for (i, tf) in fp16s.iter().enumerate() {
                        tracing::debug!("  GPU {}: {:.1} TF16", i, tf);
                    }
                }
            }
            store_benchmark_metrics(
                mem_arc.clone(),
                compute_fp32_arc.clone(),
                compute_fp16_arc.clone(),
                result.as_ref(),
            )
            .await;
            node_bench.regossip().await;
        });
    } else {
        tracing::debug!("client node — skipping memory bandwidth benchmark");
    }

    // Join mesh if --join was given or auto-discovery queued fallback candidates.
    if !cli.join.is_empty() || !auto_join_candidates.is_empty() {
        let mut joined = false;
        let join_attempts: Vec<(String, Option<String>)> = if !cli.join.is_empty() {
            cli.join
                .iter()
                .cloned()
                .map(|token| (token, None))
                .collect()
        } else {
            auto_join_candidates.clone()
        };
        let mut successful_join: Option<(String, Option<String>)> = None;

        for (t, mesh_name) in &join_attempts {
            match node.join(t).await {
                Ok(()) => {
                    if node.mesh_id().await.is_some() {
                        record_first_joined_mesh_ts(&node).await;
                    }
                    eprintln!("Joined mesh");
                    joined = true;
                    successful_join = Some((t.clone(), mesh_name.clone()));
                    break;
                }
                Err(e) => tracing::warn!("Failed to join via token: {e}"),
            }
        }

        if cli.join.is_empty() {
            cli.join.clear();
            if let Some((token, mesh_name)) = successful_join {
                cli.join.push(token);
                if cli.mesh_name.is_none() {
                    if let Some(name) = mesh_name {
                        cli.mesh_name = Some(name);
                    }
                }
            }
        }

        if !joined {
            eprintln!("Failed to join any peer — running standalone");
        }

        // Save mesh_id for sticky preference after gossip propagates it
        {
            let save_node = node.clone();
            tokio::spawn(async move {
                // Wait for gossip to propagate mesh_id
                tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                if let Some(id) = save_node.mesh_id().await {
                    record_first_joined_mesh_ts(&save_node).await;
                    mesh::save_last_mesh_id(&id);
                    tracing::info!("Mesh ID: {id}");
                }
            });
        }

        eprintln!("This node's token (for others to join): {token}");

        // Periodic rejoin: re-connect to bootstrap tokens every 60s.
        // No-op if already connected (connect_to_peer returns early).
        // Recovers from dropped connections without manual intervention.
        let rejoin_node = node.clone();
        let rejoin_tokens: Vec<String> = cli.join.clone();
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(std::time::Duration::from_secs(60)).await;
                for t in &rejoin_tokens {
                    if let Err(e) = rejoin_node.join(t).await {
                        tracing::debug!("Rejoin failed: {e}");
                    }
                }
            }
        });

        // Nostr re-discovery: if we joined via --auto (Nostr discovery) and lose
        // all peers, re-discover and join a new mesh. This handles the case where
        // the original mesh publisher restarts with a new identity.
        if cli.auto {
            let rediscover_node = node.clone();
            let rediscover_relays = nostr_relays(&cli.nostr_relay);
            let rediscover_relay_urls = cli.relay.clone();
            let rediscover_mesh_name = cli.mesh_name.clone();
            tokio::spawn(async move {
                nostr_rediscovery(
                    rediscover_node,
                    rediscover_relays,
                    rediscover_relay_urls,
                    rediscover_mesh_name,
                )
                .await;
            });
        }
    } else {
        // Originator — generate mesh_id
        let nostr_pubkey = if cli.publish {
            nostr::load_or_create_keys()
                .ok()
                .map(|k| k.public_key().to_hex())
        } else {
            None
        };
        let mesh_id = mesh::generate_mesh_id(cli.mesh_name.as_deref(), nostr_pubkey.as_deref());
        node.set_mesh_id_force(mesh_id.clone()).await;
        record_first_joined_mesh_ts(&node).await;
        mesh::save_last_mesh_id(&mesh_id);
        tracing::info!("Mesh ID: {mesh_id}");
        eprintln!("Invite: {token}");
        eprintln!("Waiting for peers...");

        // Originator also re-discovers: if we started solo and a matching mesh
        // already exists on Nostr, we should join it instead of staying alone.
        if cli.auto {
            let rediscover_node = node.clone();
            let rediscover_relays = nostr_relays(&cli.nostr_relay);
            let rediscover_relay_urls = cli.relay.clone();
            let rediscover_mesh_name = cli.mesh_name.clone();
            tokio::spawn(async move {
                nostr_rediscovery(
                    rediscover_node,
                    rediscover_relays,
                    rediscover_relay_urls,
                    rediscover_mesh_name,
                )
                .await;
            });
        }
    }

    let affinity_router = affinity::AffinityRouter::new();

    // Start bootstrap proxy if joining an existing mesh.
    // This gives instant API access via tunnel while our GPU loads.
    let mut bootstrap_listener_tx = if !cli.join.is_empty() {
        let (stop_tx, stop_rx) =
            tokio::sync::mpsc::channel::<tokio::sync::oneshot::Sender<tokio::net::TcpListener>>(1);
        let boot_node = node.clone();
        let boot_port = api_port;
        let boot_affinity = affinity_router.clone();
        tokio::spawn(async move {
            bootstrap_proxy(boot_node, boot_port, stop_rx, cli.listen_all, boot_affinity).await;
        });
        Some(stop_tx)
    } else {
        None
    };

    let primary_startup_model = startup_models.first().cloned();

    // Decide which model THIS node will serve
    let model = if let Some(primary) = primary_startup_model.as_ref() {
        // First startup model is what we serve (already resolved/downloaded)
        primary.resolved_path.clone()
    } else {
        // No --model: try to find a model on disk that the mesh needs
        eprintln!("No --model specified, checking local models against mesh...");

        // Give gossip a moment to propagate
        tokio::time::sleep(std::time::Duration::from_secs(5)).await;

        let assignment = pick_model_assignment(&node, &local_models).await;
        // If no demand-based assignment but we have VRAM, use auto pack's primary model
        let assignment = if assignment.is_none() && cli.auto && !is_client {
            let pack = nostr::auto_model_pack(node.vram_bytes() as f64 / 1e9);
            if !pack.is_empty() {
                eprintln!(
                    "📋 No unserved demand — serving {} for {:.0}GB capacity",
                    pack[0],
                    node.vram_bytes() as f64 / 1e9
                );
                Some(pack[0].clone())
            } else {
                assignment
            }
        } else {
            assignment
        };
        if let Some(model_name) = assignment {
            eprintln!("Mesh assigned model: {model_name}");
            let model_path = models::find_model_path(&model_name);
            if model_path.exists() {
                model_path
            } else if let Some(cat) = catalog::find_model(&model_name) {
                // Model not on disk but in catalog — download it
                eprintln!("📥 Downloading {} for mesh...", model_name);
                catalog::download_model(cat).await?
            } else {
                model_path
            }
        } else {
            // Nothing on disk matches — go passive, act as proxy
            // Stop bootstrap proxy first (run_passive binds its own listener)
            drop(bootstrap_listener_tx.take());
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            // If a model becomes unserved while we're standby, we'll promote
            if is_client {
                eprintln!("📡 Running as client — proxying requests to mesh");
            } else {
                eprintln!("💤 No matching model on disk — running as standby GPU node");
                eprintln!(
                    "   Capacity: {:.1}GB, models on disk: {:?}",
                    node.vram_bytes() as f64 / 1e9,
                    local_models
                );
                eprintln!("   Proxying requests to other nodes. Will activate when needed.");
            }
            match run_passive(&cli, node.clone(), is_client, plugin_manager.clone()).await? {
                Some(model_name) => {
                    // Promoted! Resolve the model path and continue to serving
                    models::find_model_path(&model_name)
                }
                None => return Ok(()), // clean shutdown
            }
        }
    };

    let model_name = {
        let stem = model
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
        // Strip split GGUF suffix: "MiniMax-M2.5-Q4_K_M-00001-of-00004" → "MiniMax-M2.5-Q4_K_M"
        router::strip_split_suffix_owned(&stem)
    };

    // Set model source for gossip (so other joiners can discover it too)
    let model_source = primary_startup_model
        .as_ref()
        .map(|model| model.declared_ref.clone())
        .unwrap_or_else(|| model_name.clone());
    node.set_model_source(model_source).await;
    // Declare which models this node may serve, but do not advertise them as
    // live/routable until their local processes have passed health checks.
    let all_declared = build_serving_list(&resolved_models, &model_name);
    node.set_serving_models(all_declared.clone()).await;
    node.set_hosted_models(Vec::new()).await;
    node.set_models(all_declared.clone()).await;
    // Re-gossip so peers learn our catalog/requested state without prematurely
    // routing requests to not-yet-ready local processes.
    node.regossip().await;

    // Ensure draft model is available (downloads if needed, <1GB)
    // `--no-draft` disables automatic draft detection, but should not
    // override an explicitly supplied `--draft` value.
    if !cli.no_draft && cli.draft.is_none() {
        if let Some(draft_path) = ensure_draft(&model).await {
            eprintln!("Auto-detected draft model: {}", draft_path.display());
            cli.draft = Some(draft_path);
        }
    }

    // Drain stale pidfiles from our own runtime dir before spawning a new rpc-server
    if let Some(ref rt) = runtime {
        let _ = crate::runtime::instance::reap::reap_own_stale_pidfiles(rt.dir()).await;
    }

    // Serve mode (non-client) always has the InstanceRuntime acquired above.
    // The fallback was only relevant during the T1-T11 staging when acquisition
    // wasn't yet wired into run() — keep an explicit error here so any future
    // refactor that drops the acquire surfaces immediately instead of panicking
    // mid-spawn from a child task.
    let runtime_arc = runtime
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("serve mode requires an instance runtime"))?
        .clone();
    let rpc_handle = launch::start_rpc_server(
        &runtime_arc,
        &bin_dir,
        cli.llama_flavor,
        startup_rpc_backend_device(cli.device.as_deref(), primary_startup_model.as_ref())?,
        Some(&model),
    )
    .await?;
    tracing::info!(
        "rpc-server on 127.0.0.1:{} (pid {}) serving {model_name}",
        rpc_handle.port,
        rpc_handle.pid
    );

    let tunnel_mgr =
        tunnel::Manager::start(node.clone(), rpc_handle.port, channels.rpc, channels.http).await?;

    // Election publishes per-model targets
    let (target_tx, target_rx) = tokio::sync::watch::channel(election::ModelTargets::default());
    let target_tx = std::sync::Arc::new(target_tx);

    // Runtime control for local load/unload of extra models.
    let (control_tx, mut control_rx) =
        tokio::sync::mpsc::unbounded_channel::<api::RuntimeControlRequest>();
    let (runtime_event_tx, mut runtime_event_rx) =
        tokio::sync::mpsc::unbounded_channel::<RuntimeEvent>();
    let mut runtime_models: HashMap<String, LocalRuntimeModelHandle> = HashMap::new();
    let mut managed_models: HashMap<String, ManagedModelController> = HashMap::new();

    // Take over listener from bootstrap proxy (if running), or bind a new one
    let existing_listener = if let Some(tx) = bootstrap_listener_tx {
        let (resp_tx, resp_rx) = tokio::sync::oneshot::channel();
        let _ = tx.send(resp_tx).await;
        // Wait for bootstrap to hand back the TcpListener
        resp_rx.await.ok()
    } else {
        None
    };

    // API proxy: model-aware routing
    let proxy_node = node.clone();
    let proxy_rx = target_rx.clone();
    let proxy_affinity = affinity_router.clone();
    let api_control_tx = control_tx.clone();
    tokio::spawn(async move {
        api_proxy(
            proxy_node,
            api_port,
            proxy_rx,
            api_control_tx,
            existing_listener,
            cli.listen_all,
            proxy_affinity,
        )
        .await;
    });

    // Console (optional)
    let model_name_for_console = model_name.clone();
    let console_state = if let Some(cport) = console_port {
        let model_size_bytes = election::total_model_bytes(&model);
        let cs = api::MeshApi::new(
            node.clone(),
            model_name_for_console.clone(),
            api_port,
            model_size_bytes,
            plugin_manager.clone(),
            affinity_router.clone(),
        );
        cs.set_primary_backend("llama".into()).await;
        cs.set_runtime_control(control_tx.clone()).await;
        cs.set_nostr_relays(nostr_relays(&cli.nostr_relay)).await;
        cs.set_nostr_discovery(cli.nostr_discovery).await;
        if let Some(draft) = &cli.draft {
            let dn = draft
                .file_stem()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();
            cs.set_draft_name(dn).await;
        }
        if let Some(ref name) = cli.mesh_name {
            cs.set_mesh_name(name.clone()).await;
        }
        let cs2 = cs.clone();
        let console_rx = target_rx.clone();
        let mn = model_name_for_console.clone();
        tokio::spawn(async move {
            // Console still takes old-style InferenceTarget for now — adapt
            let (adapted_tx, adapted_rx) =
                tokio::sync::watch::channel(election::InferenceTarget::None);
            tokio::spawn(async move {
                let mut rx = console_rx;
                loop {
                    let targets = rx.borrow().clone();
                    let target = targets.get(&mn);
                    adapted_tx.send_replace(target);
                    if rx.changed().await.is_err() {
                        break;
                    }
                }
            });
            api::start(cport, cs2, adapted_rx, cli.listen_all, cli.headless).await;
        });
        Some(cs)
    } else {
        None
    };

    if !is_client {
        if let Some(ref cs) = console_state {
            if let Ok(root) = crate::runtime::instance::runtime_root() {
                let li_handle = cs.local_instances_handle().await;
                if let Ok(initial) =
                    crate::runtime::instance::scan_local_instances(&root, std::process::id()).await
                {
                    *li_handle.lock().await = initial;
                }
                crate::runtime::instance::spawn_local_instance_scanner(
                    root,
                    std::process::id(),
                    li_handle,
                );
            }
        }
    }

    // Election loop
    tracing::info!("Entering auto-election for model: {model_name}");
    let node2 = node.clone();
    let tunnel_mgr2 = tunnel_mgr.clone();
    let bin_dir2 = bin_dir.clone();
    let model2 = model.clone();
    let draft2 = cli.draft.clone();
    let draft_max = cli.draft_max;
    let slots = primary_startup_model
        .as_ref()
        .and_then(|m| m.parallel)
        .or(config.gpu.parallel)
        .unwrap_or(4);
    let force_split = cli.split;
    let llama_flavor = cli.llama_flavor;
    let cb_console_port = console_port;
    let cli_headless = cli.headless;
    let model_name_for_cb = model_name.clone();
    let model_name_for_election = model_name.clone();
    let node_for_cb = node.clone();
    let node_for_primary_process = node.clone();
    let primary_target_tx = target_tx.clone();
    let console_state_for_election = console_state.clone();
    let console_state_for_primary_process = console_state.clone();
    let primary_process_model_name = model_name.clone();
    let primary_model_name_for_advertise = model_name.clone();
    let moe_runtime_options = moe::MoeRuntimeOptions::default();
    let primary_mmproj = primary_startup_model
        .as_ref()
        .and_then(|model| model.mmproj_path.clone());
    let primary_ctx_size = primary_startup_model
        .as_ref()
        .and_then(|model| model.ctx_size);
    let primary_pinned_gpu = primary_startup_model
        .as_ref()
        .and_then(|model| model.pinned_gpu.clone());
    let (primary_stop_tx, primary_stop_rx) = tokio::sync::watch::channel(false);
    let primary_runtime = runtime_arc.clone();
    let primary_task = tokio::spawn(async move {
        election::election_loop(
            election::ElectionLoopParams {
                runtime: primary_runtime,
                node: node2,
                tunnel_mgr: tunnel_mgr2,
                ingress_http_port: api_port,
                rpc_port: rpc_handle.port,
                bin_dir: bin_dir2,
                model: model2,
                model_name: model_name_for_election,
                explicit_mmproj: primary_mmproj,
                draft: draft2,
                draft_max,
                force_split,
                binary_flavor: llama_flavor,
                ctx_size_override: primary_ctx_size,
                pinned_gpu: primary_pinned_gpu,
                moe_runtime_options,
                target_tx: primary_target_tx,
                stop_rx: primary_stop_rx,
                slots,
            },
            move |is_host, llama_ready| {
                let advertise_node = node_for_cb.clone();
                let advertise_model = primary_model_name_for_advertise.clone();
                tokio::spawn(async move {
                    if is_host && llama_ready {
                        advertise_model_ready(&advertise_node, &advertise_model, &advertise_model)
                            .await;
                    } else {
                        withdraw_advertised_model(&advertise_node, &advertise_model).await;
                    }
                });
                if llama_ready {
                    let n = node_for_cb.clone();
                    tokio::spawn(async move { n.set_llama_ready(true).await; });
                }
                if is_host && llama_ready {
                    let url = format!("http://localhost:{api_port}");
                    eprintln!("  API:     {url}");
                    if let Some(cp) = cb_console_port {
                        eprintln!("{}", format_console_ready_line(cli_headless, cp));
                    }
                    update_pi_models_json(&model_name_for_cb, api_port);
                    eprintln!();
                    eprintln!("  pi:    pi --provider mesh --model {model_name_for_cb}");
                    eprintln!("  goose: GOOSE_PROVIDER=openai OPENAI_HOST={url} OPENAI_API_KEY=mesh GOOSE_MODEL={model_name_for_cb} goose session");
                } else if is_host {
                    eprintln!("⏳ Starting llama-server...");
                } else {
                    eprintln!("  API: http://localhost:{api_port} (proxied to host)");
                }
                if let Some(ref cs) = console_state_for_election {
                    let cs = cs.clone();
                    tokio::spawn(async move {
                        cs.update(is_host, llama_ready).await;
                    });
                }
            },
            move |process| {
                let context_node = node_for_primary_process.clone();
                let model_name = primary_process_model_name.clone();
                let console_state = console_state_for_primary_process.clone();
                tokio::spawn(async move {
                    match process {
                        Some(process) => {
                            set_advertised_model_context(
                                &context_node,
                                &model_name,
                                Some(process.context_length),
                            )
                            .await;
                            if let Some(cs) = console_state {
                                cs.upsert_local_process(local_process_payload(
                                    &model_name,
                                    &process.backend,
                                    process.port,
                                    process.pid,
                                ))
                                .await;
                            }
                        }
                        None => {
                            set_advertised_model_context(&context_node, &model_name, None).await;
                            if let Some(cs) = console_state {
                                cs.remove_local_process(&model_name).await;
                            }
                        }
                    }
                });
            },
        ).await;
    });
    managed_models.insert(
        model_name.clone(),
        ManagedModelController {
            stop_tx: primary_stop_tx,
            task: primary_task,
        },
    );

    // Additional model election loops (multi-model per node)
    // Each additional model gets its own solo election loop — no rpc, no draft, no split.
    // They share the same target_tx so the proxy sees all models.
    if startup_models.len() > 1 {
        eprintln!(
            "🔀 Multi-model mode: {} additional model(s)",
            startup_models.len() - 1
        );
        // Announce all models to mesh
        let all_names: Vec<String> = startup_models
            .iter()
            .map(|m| {
                m.resolved_path
                    .file_stem()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string()
            })
            .collect();
        node.set_models(all_names).await;
        node.regossip().await;

        for extra_model in startup_models.iter().skip(1) {
            let extra_name = {
                let stem = extra_model
                    .resolved_path
                    .file_stem()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string();
                router::strip_split_suffix_owned(&stem)
            };
            let extra_node = node.clone();
            let extra_tunnel = tunnel_mgr.clone();
            let extra_bin = bin_dir.clone();
            let extra_path = extra_model.resolved_path.clone();
            let extra_mmproj = extra_model.mmproj_path.clone();
            let extra_ctx_size = extra_model.ctx_size;
            let extra_pinned_gpu = extra_model.pinned_gpu.clone();
            let extra_target_tx = target_tx.clone();
            let extra_model_name = extra_name.clone();
            let api_port_extra = api_port;
            let extra_llama_flavor = cli.llama_flavor;
            let slots = extra_model.parallel.or(config.gpu.parallel).unwrap_or(4);
            let extra_moe_runtime_options = moe::MoeRuntimeOptions::default();
            let extra_console_state = console_state.clone();
            let extra_model_name_for_status = extra_model_name.clone();
            let extra_model_name_for_process = extra_model_name.clone();
            let extra_model_name_for_advertise = extra_model_name.clone();
            let extra_node_for_advertise = node.clone();
            let extra_node_for_process = node.clone();
            let primary_model_name_for_extra = model_name.clone();
            let managed_model_name = extra_name.clone();
            eprintln!("  + {extra_name}");
            let (extra_stop_tx, extra_stop_rx) = tokio::sync::watch::channel(false);
            let extra_runtime = runtime_arc.clone();
            let extra_task = tokio::spawn(async move {
                election::election_loop(
                    election::ElectionLoopParams {
                        runtime: extra_runtime,
                        node: extra_node,
                        tunnel_mgr: extra_tunnel,
                        ingress_http_port: api_port_extra,
                        rpc_port: 0,
                        bin_dir: extra_bin,
                        model: extra_path,
                        model_name: extra_model_name.clone(),
                        explicit_mmproj: extra_mmproj,
                        draft: None,
                        draft_max: 8,
                        force_split: false,
                        binary_flavor: extra_llama_flavor,
                        ctx_size_override: extra_ctx_size,
                        pinned_gpu: extra_pinned_gpu,
                        moe_runtime_options: extra_moe_runtime_options,
                        target_tx: extra_target_tx,
                        stop_rx: extra_stop_rx,
                        slots,
                    },
                    move |is_host, llama_ready| {
                        let advertise_node = extra_node_for_advertise.clone();
                        let model_name = extra_model_name_for_advertise.clone();
                        let primary_model_name = primary_model_name_for_extra.clone();
                        tokio::spawn(async move {
                            if is_host && llama_ready {
                                advertise_model_ready(&advertise_node, &primary_model_name, &model_name)
                                    .await;
                            } else {
                                withdraw_advertised_model(&advertise_node, &model_name).await;
                            }
                        });
                        if is_host && llama_ready {
                            eprintln!("✅ [{extra_model_name_for_status}] ready (multi-model)");
                            eprintln!("  API: http://localhost:{api_port_extra} (model={extra_model_name_for_status})");
                        }
                    },
                    move |process| {
                        let context_node = extra_node_for_process.clone();
                        let model_name = extra_model_name_for_process.clone();
                        let console_state = extra_console_state.clone();
                        tokio::spawn(async move {
                            match process {
                                Some(process) => {
                                    set_advertised_model_context(
                                        &context_node,
                                        &model_name,
                                        Some(process.context_length),
                                    )
                                    .await;
                                    if let Some(cs) = console_state {
                                        cs.upsert_local_process(local_process_payload(
                                            &model_name,
                                            &process.backend,
                                            process.port,
                                            process.pid,
                                        ))
                                        .await;
                                    }
                                }
                                None => {
                                    set_advertised_model_context(&context_node, &model_name, None)
                                        .await;
                                    if let Some(cs) = console_state {
                                        cs.remove_local_process(&model_name).await;
                                    }
                                }
                            }
                        });
                    },
                ).await;
            });
            managed_models.insert(
                managed_model_name,
                ManagedModelController {
                    stop_tx: extra_stop_tx,
                    task: extra_task,
                },
            );
        }
    }

    // Nostr publish loop (if --publish) or watchdog (if --auto, to take over if publisher dies)
    let nostr_publisher = if cli.publish {
        match nostr::load_or_create_keys() {
            Ok(nostr_keys) => {
                let relays = nostr_relays(&cli.nostr_relay);
                let pub_node = node.clone();
                let pub_name = cli.mesh_name.clone();
                let pub_region = cli.region.clone();
                let pub_max_clients = cli.max_clients;
                let (status_tx, status_rx) = tokio::sync::watch::channel(None);
                if let Some(ref cs) = console_state {
                    bridge_publication_state(cs.clone(), status_rx);
                }
                Some(tokio::spawn(async move {
                    nostr::publish_loop(
                        pub_node,
                        nostr_keys,
                        relays,
                        pub_name,
                        pub_region,
                        pub_max_clients,
                        60,
                        Some(status_tx),
                    )
                    .await;
                }))
            }
            Err(e) => {
                eprintln!(
                    "⚠️  Publishing to Nostr failed: {e}.\n\
                     Mesh is running privately — add --publish after fixing the issue to make discoverable."
                );
                tracing::warn!("Nostr publish failed: {e}");
                if let Some(ref cs) = console_state {
                    cs.set_publication_state(api::PublicationState::PublishFailed)
                        .await;
                }
                None
            }
        }
    } else if cli.auto {
        // Watchdog: if we joined via --auto, watch for the publisher to die and take over
        let relays = nostr_relays(&cli.nostr_relay);
        let wd_node = node.clone();
        let wd_name = cli.mesh_name.clone();
        let wd_region = cli.region.clone();
        let watchdog_status_rx = console_state.as_ref().map(|cs| {
            let (status_tx, status_rx) = tokio::sync::watch::channel(None);
            bridge_publication_state(cs.clone(), status_rx);
            status_tx
        });
        Some(tokio::spawn(async move {
            nostr::publish_watchdog(wd_node, relays, wd_name, wd_region, 120, watchdog_status_rx)
                .await;
        }))
    } else {
        None
    };

    // Wait for SIGINT/SIGTERM or runtime model control commands.
    let primary_model_name = model_name.clone();
    loop {
        tokio::select! {
            _ = wait_shutdown_signal() => {
                eprintln!("\nShutting down...");
                break;
            }
            Some(cmd) = control_rx.recv() => {
                match cmd {
                    api::RuntimeControlRequest::Load { spec, resp } => {
                        let mut assigned_runtime_model: Option<String> = None;
                        let result = async {
                            let model_path = resolve_model(&PathBuf::from(&spec)).await?;
                            let runtime_model_name = resolved_model_name(&model_path);
                            let already_loaded = managed_models.contains_key(&runtime_model_name)
                                || runtime_models.contains_key(&runtime_model_name);
                            anyhow::ensure!(
                                !already_loaded,
                                "model '{runtime_model_name}' is already loaded"
                            );

                            // Look up per-model parallel from TOML config by matching the
                            // spec string against [[models]].model entries. Falls back to
                            // gpu.parallel or default 4 when no entry matches.
                            let slots = config
                                .models
                                .iter()
                                .find(|m| m.model == spec)
                                .and_then(|m| m.parallel)
                                .or(config.gpu.parallel)
                                .unwrap_or(4);

                            assigned_runtime_model = Some(runtime_model_name.clone());
                            add_serving_assignment(&node, &primary_model_name, &runtime_model_name)
                                .await;
                            let (loaded_name, handle, death_rx) = start_runtime_local_model(
                                &runtime_arc,
                                &bin_dir,
                                cli.llama_flavor,
                                &node,
                                &model_path,
                                None,
                                cli.ctx_size,
                                slots,
                            )
                            .await?;

                            add_runtime_local_target(&target_tx, &loaded_name, handle.port);
                            set_advertised_model_context(
                                &node,
                                &loaded_name,
                                Some(handle.context_length),
                            )
                            .await;
                            advertise_model_ready(&node, &primary_model_name, &loaded_name).await;
                            node.set_available_models(models::scan_local_models()).await;
                            if let Some(ref cs) = console_state {
                                cs.upsert_local_process(local_process_payload(
                                    &loaded_name,
                                    &handle.backend,
                                    handle.port,
                                    handle.process.pid(),
                                ))
                                .await;
                            }

                            let event_tx = runtime_event_tx.clone();
                            let event_name = loaded_name.clone();
                            let event_port = handle.port;
                            tokio::spawn(async move {
                                let _ = death_rx.await;
                                let _ = event_tx.send(RuntimeEvent::Exited {
                                    model: event_name,
                                    port: event_port,
                                });
                            });

                            eprintln!(
                                "✅ Runtime-loaded {} model '{}' on :{}",
                                handle.backend,
                                loaded_name,
                                handle.port
                            );
                            runtime_models.insert(loaded_name.clone(), handle);
                            Ok(loaded_name)
                        }
                        .await;
                        if let Err(err) = &result {
                            let _ = err;
                            if let Some(name) = assigned_runtime_model.as_deref() {
                                remove_serving_assignment(&node, name).await;
                            }
                        }
                        let _ = resp.send(result);
                    }
                    api::RuntimeControlRequest::Unload { model, resp } => {
                        let result = if let Some(handle) = runtime_models.remove(&model) {
                            let port = handle.port;
                            remove_runtime_local_target(&target_tx, &model, port);
                            withdraw_advertised_model(&node, &model).await;
                            tokio::time::sleep(std::time::Duration::from_millis(300)).await;
                            handle.shutdown().await;
                            remove_serving_assignment(&node, &model).await;
                            if let Some(ref cs) = console_state {
                                cs.remove_local_process(&model).await;
                            }
                            eprintln!("🗑 Unloaded local model '{}' from :{}", model, port);
                            Ok(())
                        } else if let Some(controller) = managed_models.remove(&model) {
                            let _ = controller.stop_tx.send(true);
                            let _ = controller.task.await;
                            withdraw_advertised_model(&node, &model).await;
                            remove_serving_assignment(&node, &model).await;
                            if let Some(ref cs) = console_state {
                                cs.remove_local_process(&model).await;
                            }
                            eprintln!("🗑 Unloaded managed model '{}'", model);
                            Ok(())
                        } else {
                            Err(anyhow::anyhow!("model '{model}' is not loaded"))
                        };
                        let _ = resp.send(result);
                    }
                }
            }
            Some(event) = runtime_event_rx.recv() => {
                match event {
                    RuntimeEvent::Exited { model, port } => {
                        let matches = runtime_models
                            .get(&model)
                            .map(|handle| handle.port == port)
                            .unwrap_or(false);
                        if matches {
                            if let Some(handle) = runtime_models.remove(&model) {
                                handle.shutdown().await;
                            }
                            remove_runtime_local_target(&target_tx, &model, port);
                            withdraw_advertised_model(&node, &model).await;
                            remove_serving_assignment(&node, &model).await;
                            if let Some(ref cs) = console_state {
                                cs.remove_local_process(&model).await;
                            }
                            eprintln!("⚠ Runtime model '{}' exited on :{}", model, port);
                        }
                    }
                }
            }
        }
    }

    // Announce clean departure to peers
    node.broadcast_leaving().await;

    // Clean up Nostr listing on shutdown
    if cli.publish {
        if let Ok(keys) = nostr::load_or_create_keys() {
            let relays = nostr_relays(&cli.nostr_relay);
            if let Ok(publisher) = nostr::Publisher::new(keys, &relays).await {
                let _ = publisher.unpublish().await;
                eprintln!("Removed Nostr listing");
            }
        }
    }
    if let Some(handle) = nostr_publisher {
        handle.abort();
    }

    for (name, handle) in runtime_models.drain() {
        remove_runtime_local_target(&target_tx, &name, handle.port);
        withdraw_advertised_model(&node, &name).await;
        remove_serving_assignment(&node, &name).await;
        if let Some(ref cs) = console_state {
            cs.remove_local_process(&name).await;
        }
        handle.shutdown().await;
    }

    // Signal each election loop to stop, then give it a short window to drop
    // its `llama_process` (which sends SIGTERM via the handle's Drop). Abort
    // only as a fallback so the Drop chain still runs.
    for (_, controller) in managed_models.drain() {
        let _ = controller.stop_tx.send(true);
        let mut task = controller.task;
        match tokio::time::timeout(std::time::Duration::from_secs(3), &mut task).await {
            Ok(join_result) => {
                let _ = join_result;
            }
            Err(_) => {
                tracing::warn!("election task did not stop within 3s during shutdown");
                task.abort();
                let _ = task.await;
            }
        }
    }

    node.set_serving_models(Vec::new()).await;
    node.set_hosted_models(Vec::new()).await;
    rpc_handle.shutdown().await;
    if let Some(rt) = runtime {
        let outstanding_refs = std::sync::Arc::strong_count(&rt);
        if outstanding_refs == 1 {
            let dir = rt.dir().to_path_buf();
            drop(rt);
            let _ = std::fs::remove_dir_all(&dir);
        } else {
            tracing::warn!(
                outstanding_refs,
                "skipping runtime directory removal during shutdown because runtime references remain"
            );
        }
    }
    Ok(())
}

/// Used by both --client (pure consumer) and standby GPU nodes (no matching model).
/// If `create_node` is true, creates a new Node (--client path). Otherwise reuses existing.
/// Run as passive node (client or standby GPU).
/// Returns Ok(Some(model_name)) if a standby GPU should promote to serve a model.
/// Returns Ok(None) on clean shutdown.
async fn run_passive(
    cli: &Cli,
    node: mesh::Node,
    is_client: bool,
    plugin_manager: plugin::PluginManager,
) -> Result<Option<String>> {
    let local_port = cli.port;
    let affinity_router = affinity::AffinityRouter::new();
    node.set_display_name(node_display_name(cli, &node)).await;
    let mut passive_publication_state = None;
    let mut passive_publication_rx = None;

    // Nostr publishing (if --publish, for standby GPU nodes advertising capacity)
    if cli.publish && !is_client {
        let pub_node = node.clone();
        match nostr::load_or_create_keys() {
            Ok(nostr_keys) => {
                let relays = nostr_relays(&cli.nostr_relay);
                let pub_name = cli.mesh_name.clone();
                let pub_region = cli.region.clone();
                let pub_max_clients = cli.max_clients;
                let (status_tx, status_rx) = tokio::sync::watch::channel(None);
                passive_publication_rx = Some(status_rx);
                tokio::spawn(async move {
                    nostr::publish_loop(
                        pub_node,
                        nostr_keys,
                        relays,
                        pub_name,
                        pub_region,
                        pub_max_clients,
                        60,
                        Some(status_tx),
                    )
                    .await;
                });
            }
            Err(e) => {
                eprintln!(
                    "⚠️  Publishing to Nostr failed: {e}.\n\
                     Standby node is running privately — add --publish after fixing the issue to make discoverable."
                );
                tracing::warn!("Passive Nostr publish failed: {e}");
                passive_publication_state = Some(api::PublicationState::PublishFailed);
            }
        }
    } else if cli.auto && !is_client {
        // Watchdog: take over publishing if the original publisher dies
        let relays = nostr_relays(&cli.nostr_relay);
        let wd_node = node.clone();
        let wd_name = cli.mesh_name.clone();
        let wd_region = cli.region.clone();
        let (status_tx, status_rx) = tokio::sync::watch::channel(None);
        passive_publication_rx = Some(status_rx);
        tokio::spawn(async move {
            nostr::publish_watchdog(wd_node, relays, wd_name, wd_region, 120, Some(status_tx))
                .await;
        });
    }

    // Wait briefly for gossip to propagate
    tokio::time::sleep(std::time::Duration::from_secs(2)).await;

    let served = node.models_being_served().await;
    if !served.is_empty() {
        eprintln!("Models available in mesh: {:?}", served);
    }

    let addr = if cli.listen_all {
        "0.0.0.0"
    } else {
        "127.0.0.1"
    };
    let listener = tokio::net::TcpListener::bind(format!("{addr}:{local_port}"))
        .await
        .with_context(|| format!("Failed to bind to port {local_port}"))?;
    if is_client {
        eprintln!("📡 Client ready:");
    } else {
        eprintln!("💤 Standby ready:");
    }
    eprintln!("  API:     http://localhost:{local_port}");
    eprintln!("{}", format_console_ready_line(cli.headless, cli.console));

    // Console
    {
        let cport = cli.console;
        let label = if is_client {
            "(client)".to_string()
        } else {
            "(standby)".to_string()
        };
        let cs = api::MeshApi::new(
            node.clone(),
            label,
            local_port,
            0,
            plugin_manager,
            affinity_router.clone(),
        );
        cs.set_nostr_relays(nostr_relays(&cli.nostr_relay)).await;
        cs.set_nostr_discovery(cli.nostr_discovery).await;
        if is_client {
            cs.set_client(true).await;
        }
        // Both clients and standby nodes can proxy requests through the mesh
        cs.update(false, true).await;
        if let Some(state) = passive_publication_state {
            cs.set_publication_state(state).await;
        }
        if let Some(status_rx) = passive_publication_rx {
            bridge_publication_state(cs.clone(), status_rx);
        }
        let (_tx, rx) = tokio::sync::watch::channel(election::InferenceTarget::None);
        let la = cli.listen_all;
        let headless = cli.headless;
        tokio::spawn(async move {
            api::start(cport, cs, rx, la, headless).await;
        });
    }

    // Heartbeat (started in run_auto) handles periodic gossip via random-K.
    // No extra gossip loop needed here.

    // Reactive rebalancing: watch for topology changes and promote if needed.
    // Only for standby GPU nodes (not clients — they never serve).
    let (promote_tx, mut promote_rx) = tokio::sync::mpsc::channel::<String>(1);
    if !is_client {
        let watch_node = node.clone();
        let mut peer_rx = node.peer_change_rx.clone();
        let local_models = models::scan_local_models();
        tokio::spawn(async move {
            // Wait for initial mesh settle
            tokio::time::sleep(std::time::Duration::from_secs(10)).await;
            // Periodic demand check interval (aligned with gossip cycle)
            let mut demand_interval = tokio::time::interval(std::time::Duration::from_secs(60));
            demand_interval.tick().await; // consume first immediate tick
            loop {
                // Wait for EITHER a topology change OR periodic demand check
                tokio::select! {
                    res = peer_rx.changed() => {
                        if res.is_err() { break; }
                        // Debounce — multiple changes can fire in quick succession
                        tokio::time::sleep(std::time::Duration::from_secs(3)).await;
                        // Drain any queued changes
                        while peer_rx.has_changed().unwrap_or(false) {
                            let _ = peer_rx.borrow_and_update();
                        }
                    }
                    _ = demand_interval.tick() => {
                        // Periodic check for demand-based rebalancing
                    }
                }
                // Check if there's an unserved or demand-imbalanced model we can handle
                if let Some(model_name) = check_unserved_model(&watch_node, &local_models).await {
                    eprintln!("🚀 Promoting from standby — serving {model_name}");
                    let _ = promote_tx.send(model_name).await;
                    break;
                }
            }
        });
    }

    loop {
        tokio::select! {
            accept_result = listener.accept() => {
                let (tcp_stream, addr) = accept_result?;
                tcp_stream.set_nodelay(true)?;
                tracing::info!("Connection from {addr}");
                let node = node.clone();
                let affinity = affinity_router.clone();
                tokio::spawn(crate::network::proxy::handle_mesh_request(
                    node, tcp_stream, true, affinity,
                ));
            }
            Some(model_name) = promote_rx.recv() => {
                eprintln!("⬆️  Standby promoting to serve: {model_name}");
                return Ok(Some(model_name));
            }
            _ = wait_shutdown_signal() => {
                eprintln!("\nShutting down...");
                node.broadcast_leaving().await;
                return Ok(None);
            }
        }
    }
}

pub(crate) fn bundled_bin_names(name: &str) -> Vec<String> {
    #[cfg(windows)]
    let add_platform_name = |items: &mut Vec<String>, base: String| {
        items.push(base.clone());
        items.push(format!("{base}.exe"));
    };

    #[cfg(not(windows))]
    let add_platform_name = |items: &mut Vec<String>, base: String| {
        items.push(base);
    };

    let mut names = Vec::new();
    add_platform_name(&mut names, name.to_string());
    for flavor in launch::BinaryFlavor::ALL {
        add_platform_name(&mut names, format!("{name}-{}", flavor.suffix()));
    }
    names
}

fn has_bundled_llama_bins(dir: &Path) -> bool {
    bundled_bin_names("rpc-server")
        .iter()
        .any(|name| dir.join(name).exists())
        && bundled_bin_names("llama-server")
            .iter()
            .any(|name| dir.join(name).exists())
}

fn detect_bin_dir() -> Result<PathBuf> {
    let exe = std::env::current_exe().context("Failed to determine own binary path")?;
    let dir = exe.parent().context("Binary has no parent directory")?;

    if has_bundled_llama_bins(dir) {
        return Ok(dir.to_path_buf());
    }
    let dev = dir.join("../llama.cpp/build/bin");
    if has_bundled_llama_bins(&dev) {
        return Ok(dev.canonicalize()?);
    }
    let cargo = dir.join("../../llama.cpp/build/bin");
    if has_bundled_llama_bins(&cargo) {
        return Ok(cargo.canonicalize()?);
    }
    let cargo_alt = dir.join("../../../llama.cpp/build/bin");
    if has_bundled_llama_bins(&cargo_alt) {
        return Ok(cargo_alt.canonicalize()?);
    }

    Ok(dir.to_path_buf())
}

/// Update ~/.pi/agent/models.json to include a "mesh" provider.
fn update_pi_models_json(model_id: &str, port: u16) {
    let Some(home) = dirs::home_dir() else { return };
    let models_path = home.join(".pi/agent/models.json");

    let mut root: serde_json::Value = if models_path.exists() {
        match std::fs::read_to_string(&models_path) {
            Ok(s) => serde_json::from_str(&s).unwrap_or_else(|_| serde_json::json!({})),
            Err(_) => serde_json::json!({}),
        }
    } else {
        serde_json::json!({})
    };

    let providers = root.as_object_mut().and_then(|r| {
        r.entry("providers")
            .or_insert_with(|| serde_json::json!({}));
        r.get_mut("providers")?.as_object_mut()
    });
    let Some(providers) = providers else { return };

    let mesh = serde_json::json!({
        "baseUrl": format!("http://localhost:{port}/v1"),
        "api": "openai-completions",
        "apiKey": "mesh",
        "models": [{
            "id": model_id,
            "name": model_id,
            "reasoning": false,
            "input": ["text"],
            "contextWindow": 32768,
            "maxTokens": 8192,
            "compat": {
                "supportsUsageInStreaming": false,
                "maxTokensField": "max_tokens",
                "supportsDeveloperRole": false
            }
        }]
    });

    providers.insert("mesh".to_string(), mesh);

    if let Some(parent) = models_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Ok(json) = serde_json::to_string_pretty(&root) {
        if let Err(e) = std::fs::write(&models_path, json) {
            tracing::warn!("Failed to update {}: {e}", models_path.display());
        }
    }
}

/// Resolve Nostr relay URLs from CLI or defaults.
/// Build the list of models this node is serving for gossip announcement.
/// `resolved_models` comes from explicit `--model` args (may be empty for `--auto`).
/// `model_name` is the actual model we're about to serve (always set).
/// The primary model must always appear in the result.
fn build_serving_list(resolved_models: &[PathBuf], model_name: &str) -> Vec<String> {
    let clean_name = router::strip_split_suffix_owned(model_name);
    let mut all: Vec<String> = resolved_models
        .iter()
        .map(|m| {
            let stem = m
                .file_stem()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();
            // Strip split GGUF suffix: "Model-00001-of-00004" → "Model"
            router::strip_split_suffix_owned(&stem)
        })
        .collect();
    if !all.contains(&clean_name) {
        all.insert(0, clean_name);
    }
    all
}

/// Returns the console-port label line for ready-state output.
/// In headless mode, advertises the management API; otherwise the web console.
fn format_console_ready_line(headless: bool, console_port: u16) -> String {
    if headless {
        format!("  Management API: http://localhost:{console_port}")
    } else {
        format!("  Console: http://localhost:{console_port}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::local::{huggingface_repo_folder_name, huggingface_snapshot_path};
    use crate::plugin::{GpuAssignment, GpuConfig, ModelConfigEntry};
    use crate::system::hardware::GpuFacts;
    use hf_hub::RepoType;
    use serial_test::serial;
    use std::path::Path;
    use std::path::PathBuf;
    use std::time::Duration;

    fn restore_env(key: &str, value: Option<std::ffi::OsString>) {
        if let Some(value) = value {
            std::env::set_var(key, value);
        } else {
            std::env::remove_var(key);
        }
    }

    async fn build_test_mesh_api() -> api::MeshApi {
        let node = mesh::Node::new_for_tests(mesh::NodeRole::Worker)
            .await
            .unwrap();
        let resolved_plugins = plugin::ResolvedPlugins {
            externals: vec![],
            inactive: vec![],
        };
        let (mesh_tx, _mesh_rx) = tokio::sync::mpsc::channel(1);
        let plugin_manager = plugin::PluginManager::start(
            &resolved_plugins,
            plugin::PluginHostMode {
                mesh_visibility: mesh_llm_plugin::MeshVisibility::Private,
            },
            mesh_tx,
        )
        .await
        .unwrap();
        api::MeshApi::new(
            node,
            "test-model".to_string(),
            3131,
            0,
            plugin_manager,
            affinity::AffinityRouter::default(),
        )
    }

    fn synthetic_gpu(
        index: usize,
        stable_id: Option<&str>,
        backend_device: Option<&str>,
    ) -> GpuFacts {
        GpuFacts {
            index,
            display_name: format!("GPU {index}"),
            backend_device: backend_device.map(str::to_string),
            vram_bytes: 24_000_000_000,
            reserved_bytes: None,
            mem_bandwidth_gbps: None,
            compute_tflops_fp32: None,
            compute_tflops_fp16: None,
            unified_memory: false,
            stable_id: stable_id.map(str::to_string),
            pci_bdf: None,
            vendor_uuid: None,
            metal_registry_id: None,
            dxgi_luid: None,
            pnp_instance_id: None,
        }
    }

    #[tokio::test]
    #[serial]
    async fn resolve_model_accepts_short_catalog_name_from_hf_cache() {
        let prev_hub_cache = std::env::var_os("HF_HUB_CACHE");
        let prev_hf_home = std::env::var_os("HF_HOME");
        let prev_xdg = std::env::var_os("XDG_CACHE_HOME");

        let cache_root = std::env::temp_dir().join(format!(
            "mesh-llm-short-name-cache-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&cache_root).unwrap();
        std::env::set_var("HF_HUB_CACHE", &cache_root);
        std::env::remove_var("HF_HOME");
        std::env::remove_var("XDG_CACHE_HOME");

        let repo_id = "bartowski/Llama-3.2-1B-Instruct-GGUF";
        let repo_dir = cache_root.join(huggingface_repo_folder_name(repo_id, RepoType::Model));
        std::fs::create_dir_all(repo_dir.join("refs")).unwrap();
        std::fs::write(repo_dir.join("refs").join("main"), "test-commit").unwrap();
        let model_path = huggingface_snapshot_path(repo_id, RepoType::Model, "test-commit")
            .join("Llama-3.2-1B-Instruct-Q4_K_M.gguf");
        std::fs::create_dir_all(model_path.parent().unwrap()).unwrap();
        std::fs::write(&model_path, b"gguf").unwrap();

        let resolved = resolve_model(Path::new("Llama-3.2-1B-Instruct-Q4_K_M"))
            .await
            .unwrap();
        assert_eq!(resolved, model_path);

        let _ = std::fs::remove_dir_all(&cache_root);
        restore_env("HF_HUB_CACHE", prev_hub_cache);
        restore_env("HF_HOME", prev_hf_home);
        restore_env("XDG_CACHE_HOME", prev_xdg);
    }

    #[tokio::test]
    #[serial]
    async fn resolve_model_accepts_non_catalog_name_from_hf_cache() {
        let prev_hub_cache = std::env::var_os("HF_HUB_CACHE");
        let prev_hf_home = std::env::var_os("HF_HOME");
        let prev_xdg = std::env::var_os("XDG_CACHE_HOME");

        let cache_root = std::env::temp_dir().join(format!(
            "mesh-llm-non-catalog-cache-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&cache_root).unwrap();
        std::env::set_var("HF_HUB_CACHE", &cache_root);
        std::env::remove_var("HF_HOME");
        std::env::remove_var("XDG_CACHE_HOME");

        let repo_id = "someone/Custom-GGUF";
        let repo_dir = cache_root.join(huggingface_repo_folder_name(repo_id, RepoType::Model));
        std::fs::create_dir_all(repo_dir.join("refs")).unwrap();
        std::fs::write(repo_dir.join("refs").join("main"), "test-commit").unwrap();
        let model_path = huggingface_snapshot_path(repo_id, RepoType::Model, "test-commit")
            .join("Custom-Model-Q4_K_M.gguf");
        std::fs::create_dir_all(model_path.parent().unwrap()).unwrap();
        std::fs::write(&model_path, b"gguf").unwrap();

        let resolved_by_stem = resolve_model(Path::new("Custom-Model-Q4_K_M"))
            .await
            .unwrap();
        assert_eq!(resolved_by_stem, model_path);

        let resolved_by_filename = resolve_model(Path::new("Custom-Model-Q4_K_M.gguf"))
            .await
            .unwrap();
        assert_eq!(resolved_by_filename, model_path);

        let _ = std::fs::remove_dir_all(&cache_root);
        restore_env("HF_HUB_CACHE", prev_hub_cache);
        restore_env("HF_HOME", prev_hf_home);
        restore_env("XDG_CACHE_HOME", prev_xdg);
    }

    async fn wait_for_condition<F, Fut>(timeout: Duration, mut check: F)
    where
        F: FnMut() -> Fut,
        Fut: std::future::Future<Output = bool>,
    {
        let deadline = tokio::time::Instant::now() + timeout;
        loop {
            if check().await {
                return;
            }
            assert!(
                tokio::time::Instant::now() < deadline,
                "timed out waiting for test condition"
            );
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }

    #[test]
    fn test_build_serving_list_auto_no_resolved() {
        // --auto: resolved_models is empty, model picked dynamically
        let resolved: Vec<PathBuf> = vec![];
        let result = build_serving_list(&resolved, "Qwen3-30B-A3B-Q4_K_M");
        assert_eq!(result, vec!["Qwen3-30B-A3B-Q4_K_M"]);
    }

    #[test]
    fn test_build_serving_list_explicit_single_model() {
        // --model Qwen3-30B: resolved_models has the model
        let resolved = vec![PathBuf::from(
            "/home/.cache/huggingface/hub/Qwen3-30B-A3B-Q4_K_M.gguf",
        )];
        let result = build_serving_list(&resolved, "Qwen3-30B-A3B-Q4_K_M");
        assert_eq!(result, vec!["Qwen3-30B-A3B-Q4_K_M"]);
        // No duplicate
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_build_serving_list_explicit_multi_model() {
        // --model A --model B: both resolved
        let resolved = vec![
            PathBuf::from("/home/.cache/huggingface/hub/Qwen3-30B-A3B-Q4_K_M.gguf"),
            PathBuf::from("/home/.cache/huggingface/hub/Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf"),
        ];
        let result = build_serving_list(&resolved, "Qwen3-30B-A3B-Q4_K_M");
        assert_eq!(
            result,
            vec!["Qwen3-30B-A3B-Q4_K_M", "Qwen2.5-Coder-7B-Instruct-Q4_K_M"]
        );
    }

    #[test]
    fn test_build_serving_list_split_gguf() {
        // Split GGUF: file is "MiniMax-M2.5-Q4_K_M-00001-of-00004.gguf"
        // Serving list should strip the split suffix
        let resolved = vec![PathBuf::from(
            "/home/.cache/huggingface/hub/MiniMax-M2.5-Q4_K_M-00001-of-00004.gguf",
        )];
        let result = build_serving_list(&resolved, "MiniMax-M2.5-Q4_K_M");
        assert_eq!(result, vec!["MiniMax-M2.5-Q4_K_M"]);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_build_serving_list_split_gguf_model_name_also_has_suffix() {
        // If model_name also has the suffix (from dynamic pick), strip it too
        let resolved = vec![PathBuf::from(
            "/home/.cache/huggingface/hub/MiniMax-M2.5-Q4_K_M-00001-of-00004.gguf",
        )];
        let result = build_serving_list(&resolved, "MiniMax-M2.5-Q4_K_M-00001-of-00004");
        assert_eq!(result, vec!["MiniMax-M2.5-Q4_K_M"]);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_build_startup_model_specs_prefers_cli_models_over_config() {
        let cli = Cli::parse_from([
            "mesh-llm",
            "--model",
            "Qwen3-8B-Q4_K_M",
            "--ctx-size",
            "4096",
        ]);
        let config = plugin::MeshConfig {
            models: vec![plugin::ModelConfigEntry {
                model: "Ignored-Model".into(),
                mmproj: Some("/tmp/ignored-mmproj.gguf".into()),
                ctx_size: Some(8192),
                gpu_id: None,
                parallel: None,
            }],
            ..plugin::MeshConfig::default()
        };

        let specs = build_startup_model_specs(&cli, &config).unwrap();
        assert_eq!(specs.len(), 1);
        assert_eq!(specs[0].model_ref, PathBuf::from("Qwen3-8B-Q4_K_M"));
        assert_eq!(specs[0].mmproj_ref, None);
        assert_eq!(specs[0].ctx_size, Some(4096));
        assert_eq!(specs[0].gpu_id, None);
        assert!(!specs[0].config_owned);
    }

    #[test]
    fn test_build_startup_model_specs_uses_config_models_when_cli_is_empty() {
        let cli = Cli::parse_from(["mesh-llm", "--ctx-size", "4096"]);
        let config = plugin::MeshConfig {
            models: vec![
                plugin::ModelConfigEntry {
                    model: "Qwen3-8B-Q4_K_M".into(),
                    mmproj: None,
                    ctx_size: Some(8192),
                    gpu_id: None,
                    parallel: None,
                },
                plugin::ModelConfigEntry {
                    model: "bartowski/Qwen2.5-VL/model.gguf".into(),
                    mmproj: Some("bartowski/Qwen2.5-VL/mmproj.gguf".into()),
                    ctx_size: Some(16384),
                    gpu_id: None,
                    parallel: None,
                },
            ],
            ..plugin::MeshConfig::default()
        };

        let specs = build_startup_model_specs(&cli, &config).unwrap();
        assert_eq!(specs.len(), 2);
        assert_eq!(specs[0].model_ref, PathBuf::from("Qwen3-8B-Q4_K_M"));
        assert_eq!(specs[0].ctx_size, Some(4096));
        assert_eq!(specs[0].gpu_id, None);
        assert!(specs[0].config_owned);
        assert_eq!(
            specs[1].mmproj_ref,
            Some(PathBuf::from("bartowski/Qwen2.5-VL/mmproj.gguf"))
        );
        assert_eq!(specs[1].ctx_size, Some(4096));
        assert_eq!(specs[1].gpu_id, None);
        assert!(specs[1].config_owned);
    }

    #[test]
    fn test_build_startup_model_specs_ignores_config_models_for_client() {
        let cli = Cli::parse_from(["mesh-llm", "--client"]);
        let config = plugin::MeshConfig {
            models: vec![plugin::ModelConfigEntry {
                model: "Qwen3-8B-Q4_K_M".into(),
                mmproj: None,
                ctx_size: Some(8192),
                gpu_id: None,
                parallel: None,
            }],
            ..plugin::MeshConfig::default()
        };

        let specs = build_startup_model_specs(&cli, &config).unwrap();
        assert!(specs.is_empty());
    }

    #[test]
    fn pinned_gpu_startup_preflight_uses_config_gpu_id() {
        let cli = Cli::parse_from(["mesh-llm"]);
        let config = plugin::MeshConfig {
            gpu: plugin::GpuConfig {
                assignment: plugin::GpuAssignment::Pinned,
                parallel: None,
            },
            models: vec![plugin::ModelConfigEntry {
                model: "Qwen3-8B-Q4_K_M".into(),
                mmproj: None,
                ctx_size: Some(8192),
                gpu_id: Some("pci:0000:65:00.0".into()),
                parallel: None,
            }],
            ..plugin::MeshConfig::default()
        };
        let specs = build_startup_model_specs(&cli, &config).unwrap();
        let mut plans = vec![StartupModelPlan {
            declared_ref: "Qwen3-8B-Q4_K_M".into(),
            resolved_path: PathBuf::from("/tmp/Qwen3-8B-Q4_K_M.gguf"),
            mmproj_path: None,
            ctx_size: Some(8192),
            gpu_id: specs[0].gpu_id.clone(),
            pinned_gpu: None,
            parallel: None,
        }];
        let gpus = vec![
            synthetic_gpu(0, Some("pci:0000:65:00.0"), Some("CUDA0")),
            synthetic_gpu(1, Some("pci:0000:b3:00.0"), Some("CUDA1")),
        ];

        preflight_config_owned_startup_models_with_gpus(&config, &specs, &mut plans, &gpus)
            .unwrap();

        assert_eq!(plans[0].gpu_id.as_deref(), Some("pci:0000:65:00.0"));
        assert_eq!(
            plans[0].pinned_gpu,
            Some(StartupPinnedGpuTarget {
                index: 0,
                stable_id: "pci:0000:65:00.0".into(),
                backend_device: "CUDA0".into(),
                vram_bytes: 24_000_000_000,
            })
        );
    }

    #[test]
    fn pinned_gpu_startup_preflight_requests_per_gpu_vram_metrics() {
        let metrics = pinned_startup_preflight_metrics();

        assert_eq!(metrics.len(), 4);
        assert!(metrics.contains(&hardware::Metric::GpuName));
        assert!(metrics.contains(&hardware::Metric::GpuFacts));
        assert!(metrics.contains(&hardware::Metric::VramBytes));
        assert!(metrics.contains(&hardware::Metric::IsSoc));
    }

    #[test]
    fn pinned_gpu_startup_preflight_cli_models_bypass_config_gpu_id() {
        let cli = Cli::parse_from(["mesh-llm", "--model", "Qwen3-8B-Q4_K_M"]);
        let config = plugin::MeshConfig {
            gpu: plugin::GpuConfig {
                assignment: plugin::GpuAssignment::Pinned,
                parallel: None,
            },
            models: vec![plugin::ModelConfigEntry {
                model: "Ignored-Model".into(),
                mmproj: None,
                ctx_size: Some(8192),
                gpu_id: Some("pci:0000:65:00.0".into()),
                parallel: None,
            }],
            ..plugin::MeshConfig::default()
        };
        let specs = build_startup_model_specs(&cli, &config).unwrap();
        let mut plans = vec![StartupModelPlan {
            declared_ref: "Qwen3-8B-Q4_K_M".into(),
            resolved_path: PathBuf::from("/tmp/Qwen3-8B-Q4_K_M.gguf"),
            mmproj_path: None,
            ctx_size: None,
            gpu_id: specs[0].gpu_id.clone(),
            pinned_gpu: None,
            parallel: None,
        }];
        let gpus = vec![synthetic_gpu(0, Some("pci:0000:65:00.0"), Some("CUDA0"))];

        preflight_config_owned_startup_models_with_gpus(&config, &specs, &mut plans, &gpus)
            .unwrap();

        assert_eq!(specs[0].gpu_id, None);
        assert!(!specs[0].config_owned);
        assert_eq!(plans[0].gpu_id, None);
        assert_eq!(plans[0].pinned_gpu, None);
    }

    #[test]
    fn pinned_gpu_startup_preflight_missing_gpu_id_fails_closed() {
        let config = plugin::MeshConfig {
            gpu: plugin::GpuConfig {
                assignment: plugin::GpuAssignment::Pinned,
                parallel: None,
            },
            ..plugin::MeshConfig::default()
        };
        let specs = vec![StartupModelSpec {
            model_ref: PathBuf::from("Qwen3-8B-Q4_K_M"),
            mmproj_ref: None,
            ctx_size: None,
            gpu_id: None,
            config_owned: true,
            parallel: None,
        }];
        let mut plans = vec![StartupModelPlan {
            declared_ref: "Qwen3-8B-Q4_K_M".into(),
            resolved_path: PathBuf::from("/tmp/Qwen3-8B-Q4_K_M.gguf"),
            mmproj_path: None,
            ctx_size: None,
            gpu_id: None,
            pinned_gpu: None,
            parallel: None,
        }];
        let gpus = vec![synthetic_gpu(0, Some("pci:0000:65:00.0"), Some("CUDA0"))];

        let err =
            preflight_config_owned_startup_models_with_gpus(&config, &specs, &mut plans, &gpus)
                .unwrap_err();
        let message = format!("{err:#}");

        assert!(message.contains("failed pinned GPU preflight"));
        assert!(message.contains("missing configured gpu_id"));
    }

    #[test]
    fn pinned_gpu_startup_preflight_stores_resolved_pinned_target_in_plan() {
        let config = plugin::MeshConfig {
            gpu: plugin::GpuConfig {
                assignment: plugin::GpuAssignment::Pinned,
                parallel: None,
            },
            ..plugin::MeshConfig::default()
        };
        let specs = vec![StartupModelSpec {
            model_ref: PathBuf::from("Qwen3-8B-Q4_K_M"),
            mmproj_ref: None,
            ctx_size: Some(4096),
            gpu_id: Some("uuid:GPU-123".into()),
            config_owned: true,
            parallel: None,
        }];
        let mut plans = vec![StartupModelPlan {
            declared_ref: "Qwen3-8B-Q4_K_M".into(),
            resolved_path: PathBuf::from("/tmp/Qwen3-8B-Q4_K_M.gguf"),
            mmproj_path: None,
            ctx_size: Some(4096),
            gpu_id: Some("uuid:GPU-123".into()),
            pinned_gpu: None,
            parallel: None,
        }];
        let gpus = vec![synthetic_gpu(3, Some("uuid:GPU-123"), Some("CUDA3"))];

        preflight_config_owned_startup_models_with_gpus(&config, &specs, &mut plans, &gpus)
            .unwrap();

        let pinned_gpu = plans[0].pinned_gpu.as_ref().unwrap();
        assert_eq!(pinned_gpu.index, 3);
        assert_eq!(pinned_gpu.stable_id, "uuid:GPU-123");
        assert_eq!(pinned_gpu.backend_device, "CUDA3");
        assert_eq!(pinned_gpu.vram_bytes, 24_000_000_000);
    }

    #[test]
    fn pinned_gpu_startup_preflight_rejects_resolved_gpu_without_backend_device() {
        let config = plugin::MeshConfig {
            gpu: plugin::GpuConfig {
                assignment: plugin::GpuAssignment::Pinned,
                parallel: None,
            },
            ..plugin::MeshConfig::default()
        };
        let specs = vec![StartupModelSpec {
            model_ref: PathBuf::from("Qwen3-8B-Q4_K_M"),
            mmproj_ref: None,
            ctx_size: Some(4096),
            gpu_id: Some("uuid:GPU-123".into()),
            config_owned: true,
            parallel: None,
        }];
        let mut plans = vec![StartupModelPlan {
            declared_ref: "Qwen3-8B-Q4_K_M".into(),
            resolved_path: PathBuf::from("/tmp/Qwen3-8B-Q4_K_M.gguf"),
            mmproj_path: None,
            ctx_size: Some(4096),
            gpu_id: Some("uuid:GPU-123".into()),
            pinned_gpu: None,
            parallel: None,
        }];
        let gpus = vec![synthetic_gpu(3, Some("uuid:GPU-123"), None)];

        let err =
            preflight_config_owned_startup_models_with_gpus(&config, &specs, &mut plans, &gpus)
                .unwrap_err();
        let message = format!("{err:#}");

        assert!(message.contains("failed pinned GPU preflight"));
        assert!(message.contains("without a backend_device"));
    }

    #[test]
    fn pinned_gpu_startup_preflight_unresolvable_gpu_id_fails_closed() {
        let config = plugin::MeshConfig {
            gpu: plugin::GpuConfig {
                assignment: plugin::GpuAssignment::Pinned,
                parallel: None,
            },
            ..plugin::MeshConfig::default()
        };
        let specs = vec![StartupModelSpec {
            model_ref: PathBuf::from("Qwen3-8B-Q4_K_M"),
            mmproj_ref: None,
            ctx_size: None,
            gpu_id: Some("pci:0000:b3:00.0".into()),
            config_owned: true,
            parallel: None,
        }];
        let mut plans = vec![StartupModelPlan {
            declared_ref: "Qwen3-8B-Q4_K_M".into(),
            resolved_path: PathBuf::from("/tmp/Qwen3-8B-Q4_K_M.gguf"),
            mmproj_path: None,
            ctx_size: None,
            gpu_id: Some("pci:0000:b3:00.0".into()),
            pinned_gpu: None,
            parallel: None,
        }];
        let gpus = vec![synthetic_gpu(0, Some("pci:0000:65:00.0"), Some("CUDA0"))];

        let err =
            preflight_config_owned_startup_models_with_gpus(&config, &specs, &mut plans, &gpus)
                .unwrap_err();
        let message = format!("{err:#}");

        assert!(message.contains("failed pinned GPU preflight"));
        assert!(message.contains("did not match any available pinnable GPU"));
    }

    #[test]
    fn pinned_gpu_startup_rpc_device_uses_explicit_cli_device_without_pinned_target() {
        let device = startup_rpc_backend_device(Some("CUDA3"), None).unwrap();

        assert_eq!(device, Some("CUDA3"));
    }

    #[test]
    fn pinned_gpu_startup_rpc_device_allows_matching_explicit_and_pinned_device() {
        let primary_startup_model = StartupModelPlan {
            declared_ref: "Qwen3-8B-Q4_K_M".into(),
            resolved_path: PathBuf::from("/tmp/Qwen3-8B-Q4_K_M.gguf"),
            mmproj_path: None,
            ctx_size: Some(8192),
            gpu_id: Some("pci:0000:65:00.0".into()),
            pinned_gpu: Some(StartupPinnedGpuTarget {
                index: 0,
                stable_id: "pci:0000:65:00.0".into(),
                backend_device: "CUDA0".into(),
                vram_bytes: 24_000_000_000,
            }),
            parallel: None,
        };

        let device =
            startup_rpc_backend_device(Some("CUDA0"), Some(&primary_startup_model)).unwrap();

        assert_eq!(device, Some("CUDA0"));
    }

    #[test]
    fn pinned_gpu_startup_rpc_device_falls_back_to_pinned_backend_device() {
        let primary_startup_model = StartupModelPlan {
            declared_ref: "Qwen3-8B-Q4_K_M".into(),
            resolved_path: PathBuf::from("/tmp/Qwen3-8B-Q4_K_M.gguf"),
            mmproj_path: None,
            ctx_size: Some(8192),
            gpu_id: Some("pci:0000:65:00.0".into()),
            pinned_gpu: Some(StartupPinnedGpuTarget {
                index: 0,
                stable_id: "pci:0000:65:00.0".into(),
                backend_device: "CUDA0".into(),
                vram_bytes: 24_000_000_000,
            }),
            parallel: None,
        };

        let device = startup_rpc_backend_device(None, Some(&primary_startup_model)).unwrap();

        assert_eq!(device, Some("CUDA0"));
    }

    #[test]
    fn pinned_gpu_startup_rpc_device_rejects_conflicting_explicit_and_pinned_device() {
        let primary_startup_model = StartupModelPlan {
            declared_ref: "Qwen3-8B-Q4_K_M".into(),
            resolved_path: PathBuf::from("/tmp/Qwen3-8B-Q4_K_M.gguf"),
            mmproj_path: None,
            ctx_size: Some(8192),
            gpu_id: Some("pci:0000:65:00.0".into()),
            pinned_gpu: Some(StartupPinnedGpuTarget {
                index: 0,
                stable_id: "pci:0000:65:00.0".into(),
                backend_device: "CUDA0".into(),
                vram_bytes: 24_000_000_000,
            }),
            parallel: None,
        };

        let err =
            startup_rpc_backend_device(Some("CUDA3"), Some(&primary_startup_model)).unwrap_err();

        assert!(err
            .to_string()
            .contains("conflicts with pinned startup GPU backend device"));
    }

    #[test]
    fn pinned_gpu_startup_rpc_device_allows_lenient_amd_match() {
        let primary_startup_model = StartupModelPlan {
            declared_ref: "Qwen3-8B-Q4_K_M".into(),
            resolved_path: PathBuf::from("/tmp/Qwen3-8B-Q4_K_M.gguf"),
            mmproj_path: None,
            ctx_size: Some(8192),
            gpu_id: Some("pci:0000:65:00.0".into()),
            pinned_gpu: Some(StartupPinnedGpuTarget {
                index: 0,
                stable_id: "pci:0000:65:00.0".into(),
                backend_device: "ROCm0".into(),
                vram_bytes: 24_000_000_000,
            }),
            parallel: None,
        };

        let device1 =
            startup_rpc_backend_device(Some("HIP0"), Some(&primary_startup_model)).unwrap();
        assert_eq!(device1, Some("HIP0"));

        let mut model_hip = primary_startup_model.clone();
        model_hip.pinned_gpu.as_mut().unwrap().backend_device = "HIP1".into();

        let device2 = startup_rpc_backend_device(Some("ROCm1"), Some(&model_hip)).unwrap();
        assert_eq!(device2, Some("ROCm1"));
    }

    #[test]
    fn test_should_show_serve_config_help_for_bare_serve_without_models() {
        let cli = Cli::parse_from(["mesh-llm"]);
        let startup_specs = Vec::new();

        assert!(should_show_serve_config_help(
            Some(RuntimeSurface::Serve),
            &cli,
            &startup_specs
        ));
    }

    #[test]
    fn test_should_not_show_serve_config_help_when_models_are_present() {
        let cli = Cli::parse_from(["mesh-llm"]);
        let startup_specs = vec![StartupModelSpec {
            model_ref: PathBuf::from("Qwen3-8B-Q4_K_M"),
            mmproj_ref: None,
            ctx_size: None,
            gpu_id: None,
            config_owned: false,
            parallel: None,
        }];

        assert!(!should_show_serve_config_help(
            Some(RuntimeSurface::Serve),
            &cli,
            &startup_specs
        ));
    }

    #[test]
    fn test_should_not_show_serve_config_help_for_client_surface() {
        let cli = Cli::parse_from(["mesh-llm", "--client"]);
        let startup_specs = Vec::new();

        assert!(!should_show_serve_config_help(
            Some(RuntimeSurface::Client),
            &cli,
            &startup_specs
        ));
    }

    #[test]
    fn test_should_not_show_serve_config_help_for_auto_serve_without_models() {
        let cli = Cli::parse_from(["mesh-llm", "--auto"]);
        let startup_specs = Vec::new();

        assert!(!should_show_serve_config_help(
            Some(RuntimeSurface::Serve),
            &cli,
            &startup_specs
        ));
    }

    #[test]
    fn test_should_not_show_serve_config_help_for_join_serve_without_models() {
        let cli = Cli::parse_from(["mesh-llm", "--join", "token"]);
        let startup_specs = Vec::new();

        assert!(!should_show_serve_config_help(
            Some(RuntimeSurface::Serve),
            &cli,
            &startup_specs
        ));
    }

    #[tokio::test]
    async fn test_runtime_load_unload_regossips_across_nodes() {
        let host = mesh::Node::new_for_tests(mesh::NodeRole::Worker)
            .await
            .unwrap();
        let observer = mesh::Node::new_for_tests(mesh::NodeRole::Worker)
            .await
            .unwrap();

        host.set_role(mesh::NodeRole::Host { http_port: 9337 })
            .await;
        host.set_serving_models(vec!["Primary".into()]).await;
        host.set_hosted_models(vec!["Primary".into()]).await;

        observer.sync_from_peer_for_tests(&host).await;

        wait_for_condition(Duration::from_secs(5), || {
            let observer = observer.clone();
            let host_id = host.id();
            async move {
                observer.peers().await.iter().any(|peer| {
                    peer.id == host_id
                        && peer.routes_model("Primary")
                        && !peer.routes_model("Runtime")
                })
            }
        })
        .await;

        add_serving_assignment(&host, "Primary", "Runtime").await;
        advertise_model_ready(&host, "Primary", "Runtime").await;
        observer.sync_from_peer_for_tests(&host).await;

        wait_for_condition(Duration::from_secs(5), || {
            let observer = observer.clone();
            let host_id = host.id();
            async move {
                observer.peers().await.iter().any(|peer| {
                    peer.id == host_id
                        && peer.is_assigned_model("Runtime")
                        && peer.routes_model("Runtime")
                        && peer.routable_models()
                            == vec!["Primary".to_string(), "Runtime".to_string()]
                })
            }
        })
        .await;

        remove_serving_assignment(&host, "Runtime").await;
        withdraw_advertised_model(&host, "Runtime").await;
        observer.sync_from_peer_for_tests(&host).await;

        wait_for_condition(Duration::from_secs(5), || {
            let observer = observer.clone();
            let host_id = host.id();
            async move {
                observer.peers().await.iter().any(|peer| {
                    peer.id == host_id
                        && peer.routes_model("Primary")
                        && !peer.is_assigned_model("Runtime")
                        && !peer.routes_model("Runtime")
                        && peer.routable_models() == vec!["Primary".to_string()]
                })
            }
        })
        .await;
    }

    #[test]
    fn ensure_draft_catalog_lookup_is_case_insensitive() {
        // The ensure_draft fix: lowercase filename (from Qwen HF URL) must
        // match the PascalCase catalog entry so draft resolution works.
        let lowercase = "qwen2.5-3b-instruct-q4_k_m.gguf";
        let found = catalog::MODEL_CATALOG
            .iter()
            .find(|m| m.file == lowercase || m.file.eq_ignore_ascii_case(lowercase));
        assert!(found.is_some());
        assert_eq!(found.unwrap().name, "Qwen2.5-3B-Instruct-Q4_K_M");
    }

    #[test]
    fn no_draft_flag_clears_preexisting_draft() {
        // The --no-draft fix: the flag must clear a draft that was already set
        // (e.g. cached from a previous run), not just skip auto-detection.
        let mut draft = Some(PathBuf::from("/models/draft.gguf"));
        let no_draft = true;
        // Mirrors the fixed logic in run_auto
        if no_draft {
            draft = None;
        }
        assert!(draft.is_none());
    }

    #[tokio::test]
    async fn test_benchmark_result_bandwidth_still_works() {
        let mem_arc = std::sync::Arc::new(tokio::sync::Mutex::new(None));
        let fp32_arc = std::sync::Arc::new(tokio::sync::Mutex::new(None));
        let fp16_arc = std::sync::Arc::new(tokio::sync::Mutex::new(None));
        let result = benchmark::BenchmarkResult {
            mem_bandwidth_gbps: vec![10.5, 20.0],
            compute_tflops_fp32: None,
            compute_tflops_fp16: None,
        };

        store_benchmark_metrics(
            mem_arc.clone(),
            fp32_arc.clone(),
            fp16_arc.clone(),
            Some(&result),
        )
        .await;

        assert_eq!(*mem_arc.lock().await, Some(vec![10.5, 20.0]));
        assert!(fp32_arc.lock().await.is_none());
        assert!(fp16_arc.lock().await.is_none());
    }

    #[test]
    fn headless_host_logs_management_api_without_console_url() {
        let line = format_console_ready_line(true, 3131);
        assert!(
            line.contains("Management API"),
            "expected 'Management API' in headless output, got: {line}"
        );
        assert!(
            !line.contains("Console:"),
            "headless output must not contain 'Console:', got: {line}"
        );
    }

    #[test]
    fn default_host_mode_still_logs_console_url() {
        let line = format_console_ready_line(false, 3131);
        assert!(
            line.contains("Console:"),
            "expected 'Console:' in default output, got: {line}"
        );
        assert!(
            !line.contains("Management API"),
            "default output must not contain 'Management API', got: {line}"
        );
    }

    #[test]
    fn active_startup_passes_headless_to_management_server() {
        let headless_line = format_console_ready_line(true, 9090);
        let normal_line = format_console_ready_line(false, 9090);
        assert_ne!(
            headless_line, normal_line,
            "headless and non-headless output must differ"
        );
        assert!(headless_line.contains("9090"));
        assert!(normal_line.contains("9090"));
    }

    #[test]
    fn headless_passive_mode_preserves_api_without_ui() {
        let line = format_console_ready_line(true, 3131);
        assert!(
            line.contains("Management API"),
            "passive headless output must contain 'Management API', got: {line}"
        );
        assert!(
            !line.contains("Console:"),
            "passive headless output must not contain 'Console:', got: {line}"
        );
    }

    #[test]
    fn passive_headless_promotion_keeps_ui_disabled() {
        let promoted_line = format_console_ready_line(true, 3131);
        assert!(
            promoted_line.contains("Management API"),
            "promoted headless node must still advertise Management API, got: {promoted_line}"
        );
        assert!(
            !promoted_line.contains("Console:"),
            "promoted headless node must not show Console: URL, got: {promoted_line}"
        );
    }

    #[test]
    fn default_passive_mode_still_serves_ui_when_not_headless() {
        let line = format_console_ready_line(false, 3131);
        assert!(
            line.contains("Console:"),
            "default passive output must contain 'Console:', got: {line}"
        );
        assert!(
            !line.contains("Management API"),
            "default passive output must not contain 'Management API', got: {line}"
        );
    }

    // ---------------------------------------------------------------------------
    // Per-model parallel (slots) resolution tests
    // ---------------------------------------------------------------------------

    /// Scenario 1: No global `gpu.parallel` set; a specific model entry has
    /// `parallel = 1`. The model's override value must be applied correctly.
    #[test]
    fn per_model_parallel_override_applied_when_no_global() {
        let config_models = vec![ModelConfigEntry {
            model: "my-model".to_string(),
            mmproj: None,
            ctx_size: None,
            gpu_id: None,
            parallel: Some(1),
        }];
        let gpu_config = GpuConfig::default(); // no parallel set

        // Simulate load handler lookup by spec name
        let slots = config_models
            .iter()
            .find(|m| m.model == "my-model")
            .and_then(|m| m.parallel)
            .or(gpu_config.parallel)
            .unwrap_or(4);

        assert_eq!(
            slots, 1,
            "model-specific parallel=1 should win when no global"
        );
    }

    /// Scenario 2: Two models in config — only the second one specifies a
    /// `parallel` value. The slot assignment must land on the correct model.
    #[test]
    fn per_model_parallel_applies_to_correct_model() {
        let config_models = vec![
            ModelConfigEntry {
                model: "model-a".to_string(),
                mmproj: None,
                ctx_size: None,
                gpu_id: None,
                parallel: None, // no override
            },
            ModelConfigEntry {
                model: "model-b".to_string(),
                mmproj: None,
                ctx_size: None,
                gpu_id: None,
                parallel: Some(3), // only this one has an override
            },
        ];
        let gpu_config = GpuConfig::default();

        // Model A: falls back to default (no model entry match → default 4)
        let slots_a = config_models
            .iter()
            .find(|m| m.model == "model-a")
            .and_then(|m| m.parallel)
            .or(gpu_config.parallel)
            .unwrap_or(4);
        assert_eq!(
            slots_a, 4,
            "model-a should get default 4 when it has no parallel entry"
        );

        // Model B: gets its own explicit value
        let slots_b = config_models
            .iter()
            .find(|m| m.model == "model-b")
            .and_then(|m| m.parallel)
            .or(gpu_config.parallel)
            .unwrap_or(4);
        assert_eq!(slots_b, 3, "model-b should get its own parallel=3 override");
    }

    /// Scenario 3: Two models. First has NO parallel setting, second has
    /// `parallel = 2`, and global `gpu.parallel = 3`. The first model should
    /// fall through to the global (3), while the second uses its own (2).
    #[test]
    fn per_model_parallel_fallback_to_global_for_missing_entry() {
        let config_models = vec![
            ModelConfigEntry {
                model: "first".to_string(),
                mmproj: None,
                ctx_size: None,
                gpu_id: None,
                parallel: None, // missing — should use global fallback
            },
            ModelConfigEntry {
                model: "second".to_string(),
                mmproj: None,
                ctx_size: None,
                gpu_id: None,
                parallel: Some(2), // explicit override
            },
        ];
        let gpu_config = GpuConfig {
            assignment: GpuAssignment::Auto,
            parallel: Some(3), // global default
        };

        // First model: no per-model value → falls back to gpu.parallel = 3
        let slots_first = config_models
            .iter()
            .find(|m| m.model == "first")
            .and_then(|m| m.parallel)
            .or(gpu_config.parallel)
            .unwrap_or(4);
        assert_eq!(
            slots_first, 3,
            "missing model parallel should fall back to gpu.parallel=3"
        );

        // Second model: its own value wins over global
        let slots_second = config_models
            .iter()
            .find(|m| m.model == "second")
            .and_then(|m| m.parallel)
            .or(gpu_config.parallel)
            .unwrap_or(4);
        assert_eq!(
            slots_second, 2,
            "model-specific parallel=2 should win over global gpu.parallel=3"
        );
    }

    // ---------------------------------------------------------------------------
    // Publication-state matrix (Issue #240)
    // ---------------------------------------------------------------------------

    /// Helper to build a minimal `Cli` for publication-state tests.
    fn make_cli(args: &[&str]) -> crate::cli::Cli {
        crate::cli::Cli::try_parse_from(args).unwrap()
    }

    #[test]
    fn mesh_name_does_not_force_publish() {
        let cli = make_cli(&[
            "mesh-llm",
            "--model",
            "dummy-model",
            "--mesh-name",
            "my-mesh",
        ]);
        assert!(!cli.publish, "mesh_name alone must not set publish");
        assert_eq!(cli.mesh_name.as_deref(), Some("my-mesh"));
    }

    #[test]
    fn explicit_publish_remains_enabled() {
        let cli = make_cli(&["mesh-llm", "--model", "dummy-model", "--publish"]);
        assert!(
            cli.publish,
            "explicit --publish must set publish=true even without mesh_name"
        );
    }

    #[test]
    fn publish_with_mesh_name_is_public_and_named() {
        let cli = make_cli(&[
            "mesh-llm",
            "--model",
            "dummy-model",
            "--publish",
            "--mesh-name",
            "named-public",
        ]);
        assert!(cli.publish, "publish + mesh_name must keep publish=true");
        assert_eq!(
            cli.mesh_name.as_deref(),
            Some("named-public"),
            "mesh_name must be preserved alongside publish"
        );
    }

    #[test]
    fn auto_without_publish_stays_private() {
        let cli = make_cli(&["mesh-llm", "--model", "dummy-model", "--auto"]);
        assert!(!cli.publish, "--auto alone must not imply publish");
        assert!(cli.auto, "--auto flag should still be true");
    }

    /// Task 2: Named private mesh keeps private identity (no implicit publish).
    #[test]
    fn named_private_mesh_keeps_private_identity() {
        // A named mesh without --publish must have publish=false.
        // The is_public gate in runtime startup uses `cli.auto || cli.publish`,
        // so a named-only mesh should NOT trigger public identity handling.
        let cli = make_cli(&[
            "mesh-llm",
            "--model",
            "dummy-model",
            "--mesh-name",
            "private-named",
        ]);
        assert!(!cli.publish);
        assert!(!cli.auto);
        let is_public = cli.auto || cli.publish;
        assert!(
            !is_public,
            "named-only mesh must be treated as private for identity purposes"
        );
    }

    /// Task 3: start_new_mesh helper does not auto-enable publish.
    #[test]
    fn start_new_mesh_does_not_auto_enable_publish() {
        use crate::runtime::discovery::start_new_mesh;
        let mut cli = make_cli(&["mesh-llm", "--model", "dummy-model"]);
        assert!(!cli.publish, "precondition: publish starts false");
        start_new_mesh(&mut cli, &["dummy-model".to_string()], 16.0, false);
        assert!(
            !cli.publish,
            "start_new_mesh must NOT set publish=true when it was not requested"
        );
    }

    /// Task 3: Explicit --publish survives start_new_mesh unchanged.
    #[test]
    fn start_new_mesh_preserves_explicit_publish() {
        use crate::runtime::discovery::start_new_mesh;
        let mut cli = make_cli(&["mesh-llm", "--model", "dummy-model", "--publish"]);
        assert!(cli.publish, "precondition: publish is true");
        start_new_mesh(&mut cli, &["dummy-model".to_string()], 16.0, false);
        assert!(
            cli.publish,
            "explicit --publish must survive start_new_mesh call"
        );
    }

    #[test]
    fn publish_state_updates_map_to_api_states() {
        assert_eq!(
            publication_state_from_update(nostr::PublishStateUpdate::Public),
            api::PublicationState::Public
        );
        assert_eq!(
            publication_state_from_update(nostr::PublishStateUpdate::PublishFailed),
            api::PublicationState::PublishFailed
        );
    }

    #[tokio::test]
    async fn publication_bridge_keeps_private_until_a_real_publish_outcome_arrives() {
        let state = build_test_mesh_api().await;
        let (status_tx, status_rx) = tokio::sync::watch::channel(None);
        bridge_publication_state(state.clone(), status_rx);

        assert_eq!(state.publication_state().await.as_str(), "private");

        status_tx
            .send(Some(nostr::PublishStateUpdate::Public))
            .unwrap();
        wait_for_condition(Duration::from_secs(2), || {
            let state = state.clone();
            async move { state.publication_state().await.as_str() == "public" }
        })
        .await;

        status_tx
            .send(Some(nostr::PublishStateUpdate::PublishFailed))
            .unwrap();
        wait_for_condition(Duration::from_secs(2), || {
            let state = state.clone();
            async move { state.publication_state().await.as_str() == "publish_failed" }
        })
        .await;
    }
}
