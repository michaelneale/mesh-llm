use anyhow::Result;

use crate::inference::launch;
use crate::mesh;
use crate::network::nostr;
use crate::runtime;

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

/// Returns `true` when the owner's termination outcome indicates the runtime
/// had a chance to clean up its own children (graceful SIGTERM exit).
///
/// Only [`launch::TerminationOutcome::Graceful`] qualifies — a `Killed`,
/// `NotRunning`, or `Failed` owner may leave orphan child servers.
pub(crate) fn should_protect_children(outcome: launch::TerminationOutcome) -> bool {
    matches!(outcome, launch::TerminationOutcome::Graceful)
}

/// Stop all mesh-llm instances and their child servers.
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
    let mut live_owner_runtime_dirs = std::collections::HashSet::new();
    for target in targets.iter().filter(|target| target.is_owner) {
        let outcome = launch::terminate_process_blocking(
            target.pid,
            &target.expected_comm,
            target.expected_start_time,
        );
        if outcome.is_success() {
            match outcome {
                launch::TerminationOutcome::Graceful => {
                    eprintln!(
                        "🧹 Terminated owner pid={} gracefully ({})",
                        target.pid, target.label
                    );
                }
                launch::TerminationOutcome::Killed => {
                    eprintln!(
                        "🧹 Force-killed owner pid={}; will reap its children ({})",
                        target.pid, target.label
                    );
                }
                launch::TerminationOutcome::NotRunning => {
                    eprintln!(
                        "🧹 Owner pid={} was already stopped ({})",
                        target.pid, target.label
                    );
                }
                launch::TerminationOutcome::Failed => unreachable!(),
            }
            killed += 1;
        }
        if should_protect_children(outcome) {
            live_owner_runtime_dirs.insert(target.runtime_dir.clone());
        }
    }

    for target in targets.into_iter().filter(|target| !target.is_owner) {
        if live_owner_runtime_dirs.contains(&target.runtime_dir) {
            continue;
        }

        if crate::runtime::instance::validate::process_liveness(target.pid)
            == crate::runtime::instance::validate::Liveness::Dead
        {
            continue;
        }

        if launch::terminate_process_blocking(
            target.pid,
            &target.expected_comm,
            target.expected_start_time,
        )
        .is_success()
        {
            eprintln!("🧹 Stopped {}", target.label);
            killed += 1;
        }
    }

    if killed == 0 {
        eprintln!("Nothing running.");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::launch::TerminationOutcome;

    #[test]
    fn should_protect_children_only_on_graceful() {
        assert!(
            should_protect_children(TerminationOutcome::Graceful),
            "Graceful shutdown lets the runtime clean up its own children"
        );

        assert!(
            !should_protect_children(TerminationOutcome::Killed),
            "SIGKILL bypasses the runtime's graceful shutdown path"
        );

        assert!(
            !should_protect_children(TerminationOutcome::NotRunning),
            "Owner was already dead — no guarantee children were cleaned up"
        );

        assert!(
            !should_protect_children(TerminationOutcome::Failed),
            "Could not signal the owner — no cleanup occurred"
        );
    }
}
