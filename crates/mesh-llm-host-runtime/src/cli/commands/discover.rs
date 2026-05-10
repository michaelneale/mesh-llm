use anyhow::Result;

use crate::mesh;
use crate::network::{discovery, nostr};
use crate::runtime;
use crate::system::backend;

pub(crate) struct DiscoverOptions {
    pub(crate) name: Option<String>,
    pub(crate) model: Option<String>,
    pub(crate) min_vram_gb: Option<f64>,
    pub(crate) region: Option<String>,
    pub(crate) auto_join: bool,
    pub(crate) relays: Vec<String>,
    pub(crate) discovery_mode: discovery::MeshDiscoveryMode,
    pub(crate) supplied_join_tokens: Vec<String>,
}

impl DiscoverOptions {
    fn filter(&self) -> nostr::MeshFilter {
        nostr::MeshFilter {
            name: self.name.clone(),
            model: self.model.clone(),
            min_vram_gb: self.min_vram_gb,
            region: self.region.clone(),
        }
    }
}

pub(crate) async fn run_discover(options: DiscoverOptions) -> Result<()> {
    let filter = options.filter();

    match options.discovery_mode {
        discovery::MeshDiscoveryMode::Nostr => {
            run_nostr_discover(filter, options.auto_join, options.relays).await
        }
        discovery::MeshDiscoveryMode::Mdns => {
            run_lan_discover(filter, options.auto_join, options.supplied_join_tokens).await
        }
    }
}

async fn run_nostr_discover(
    filter: nostr::MeshFilter,
    auto_join: bool,
    relays: Vec<String>,
) -> Result<()> {
    let relays = runtime::nostr_relays(&relays);

    eprintln!("🔍 Searching Nostr relays for mesh-llm meshes...");
    let meshes = nostr::discover(&relays, &filter, None).await?;

    if meshes.is_empty() {
        eprintln!("No meshes found.");
        if filter.name.is_some()
            || filter.model.is_some()
            || filter.min_vram_gb.is_some()
            || filter.region.is_some()
        {
            eprintln!("Try broader filters or check relays.");
        }
        return Ok(());
    }

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let last_mesh_id = mesh::load_last_mesh_id();
    eprintln!("Found {} mesh(es):\n", meshes.len());
    for (i, mesh) in meshes.iter().enumerate() {
        let score = nostr::score_mesh(mesh, now, last_mesh_id.as_deref());
        let age = now.saturating_sub(mesh.published_at);
        let freshness = if age < 120 {
            "fresh"
        } else if age < 300 {
            "ok"
        } else {
            "stale"
        };
        let capacity = if mesh.listing.max_clients > 0 {
            format!(
                "{}/{} clients",
                mesh.listing.client_count, mesh.listing.max_clients
            )
        } else {
            format!("{} clients", mesh.listing.client_count)
        };
        eprintln!(
            "  [{}] {} (score: {}, {}, {})",
            i + 1,
            mesh,
            score,
            freshness,
            capacity
        );
        let token = &mesh.listing.invite_token;
        let display_token = if token.len() > 40 {
            format!("{}...{}", &token[..20], &token[token.len() - 12..])
        } else {
            token.clone()
        };
        if !mesh.listing.on_disk.is_empty() {
            eprintln!("      on disk: {}", mesh.listing.on_disk.join(", "));
        }
        eprintln!("      token: {}", display_token);
        eprintln!();
    }

    if auto_join {
        let best = &meshes[0];
        eprintln!("Auto-joining best match: {}", best);
        eprintln!("\nRun:");
        eprintln!("  mesh-llm --join {}", best.listing.invite_token);
        println!("{}", best.listing.invite_token);
    } else {
        eprintln!("To join a mesh:");
        eprintln!("  mesh-llm --join <token>");
        eprintln!("  mesh-llm --discover <name>       # join by mesh name");
        eprintln!("  mesh-llm client --discover <name> # join as client by mesh name");
    }

    Ok(())
}

async fn run_lan_discover(
    filter: nostr::MeshFilter,
    auto_join: bool,
    supplied_join_tokens: Vec<String>,
) -> Result<()> {
    let supplied_join_token = supplied_join_tokens.first().map(String::as_str);
    eprintln!(
        "Searching local LAN for mesh-llm meshes via {}...",
        discovery::LAN_SERVICE_TYPE
    );
    let meshes = discovery::discover_lan(
        &filter,
        supplied_join_token,
        std::time::Duration::from_secs(5),
    )
    .await?;

    if meshes.is_empty() {
        eprintln!("No LAN meshes found.");
        if supplied_join_token.is_none() {
            eprintln!("mDNS advertisements do not include reusable invite tokens.");
            eprintln!("Pass --join <token> to verify a LAN advertisement by token fingerprint.");
        }
        return Ok(());
    }

    eprintln!("Found {} LAN mesh(es):\n", meshes.len());
    for (i, mesh) in meshes.iter().enumerate() {
        let vram_gb = mesh.listing.total_vram_bytes as f64 / 1e9;
        let models = if mesh.listing.serving.is_empty() {
            "(no models loaded)".to_string()
        } else {
            mesh.listing.serving.join(", ")
        };
        let join_state = if mesh.joinable_with_supplied_token {
            "token fingerprint matched"
        } else {
            "requires supplied token"
        };
        eprintln!(
            "  [{}] {}  {} node(s), {:.0}GB capacity  serving: {}",
            i + 1,
            mesh.listing.name.as_deref().unwrap_or("(unnamed)"),
            mesh.listing.node_count,
            vram_gb,
            models
        );
        eprintln!(
            "      instance: {}  host: {}:{}  {}",
            mesh.instance_name, mesh.host, mesh.port, join_state
        );
        if let Some(version) = &mesh.published_version {
            eprintln!("      version: {version}");
        }
        if !mesh.listing.on_disk.is_empty() {
            eprintln!("      on disk: {}", mesh.listing.on_disk.join(", "));
        }
        eprintln!();
    }

    if auto_join {
        if let Some(token) = meshes.iter().find_map(|mesh| mesh.join_token()) {
            println!("{token}");
        } else {
            eprintln!("No LAN mesh matched the supplied token fingerprint.");
            eprintln!("mDNS intentionally does not advertise raw invite tokens.");
        }
    } else {
        eprintln!("To join a LAN mesh:");
        eprintln!("  mesh-llm --join <token>");
        eprintln!("  mesh-llm --join <token> discover --mesh-discovery-mode mdns --auto");
    }

    Ok(())
}

/// Stop all mesh-llm instances tracked in the runtime root.
pub(crate) fn run_stop() -> Result<()> {
    let root = match crate::runtime::instance::runtime_root() {
        Ok(root) => root,
        Err(_) => {
            eprintln!("Nothing running.");
            return Ok(());
        }
    };

    let targets = crate::runtime::instance::collect_runtime_stop_targets(&root)?;
    let mut killed = 0u32;
    for target in targets {
        let outcome = backend::terminate_process_blocking(
            target.pid,
            &target.expected_comm,
            target.expected_start_time,
        );
        if outcome.is_success() {
            match outcome {
                backend::TerminationOutcome::Graceful => {
                    eprintln!(
                        "🧹 Terminated owner pid={} gracefully ({})",
                        target.pid, target.label
                    );
                }
                backend::TerminationOutcome::Killed => {
                    eprintln!(
                        "🧹 Force-killed owner pid={} ({})",
                        target.pid, target.label
                    );
                }
                backend::TerminationOutcome::NotRunning => {
                    eprintln!(
                        "🧹 Owner pid={} was already stopped ({})",
                        target.pid, target.label
                    );
                }
                backend::TerminationOutcome::Failed => unreachable!(),
            }
            killed += 1;
        }
    }

    if killed == 0 {
        eprintln!("Nothing running.");
    }
    Ok(())
}
