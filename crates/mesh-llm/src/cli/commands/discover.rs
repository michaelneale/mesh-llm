use anyhow::Result;

use crate::mesh;
use crate::network::nostr;
use crate::runtime;
use crate::system::backend;

pub(crate) async fn run_discover(
    model: Option<String>,
    min_vram: Option<f64>,
    region: Option<String>,
    auto_join: bool,
    relays: Vec<String>,
) -> Result<()> {
    let relays = runtime::nostr_relays(&relays);

    let filter = nostr::MeshFilter {
        model,
        min_vram_gb: min_vram,
        region,
    };

    eprintln!("🔍 Searching Nostr relays for mesh-llm meshes...");
    let meshes = nostr::discover(&relays, &filter, None).await?;

    if meshes.is_empty() {
        eprintln!("No meshes found.");
        if filter.model.is_some() || filter.min_vram_gb.is_some() || filter.region.is_some() {
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
        eprintln!("\nOr use `mesh-llm discover --join` to auto-join the best match.");
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
