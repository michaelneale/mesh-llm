//! Publish and discover mesh-llm meshes via Nostr relays.
//!
//! A running mesh publishes a replaceable event (kind 31990, d-tag "mesh-llm")
//! containing its invite token, served models, VRAM, node count, etc.
//! Other nodes can discover available meshes and auto-join.

use anyhow::Result;
use nostr_sdk::prelude::*;
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// NIP-89 "Application Handler" kind — used for service advertisements.
pub const MESH_SERVICE_KIND: u16 = 31990;

/// Default public relays.
pub const DEFAULT_RELAYS: &[&str] = &[
    "wss://relay.damus.io",
    "wss://nos.lol",
    "wss://relay.nostr.band",
    "wss://nostr.land",
    "wss://nostr.wine",
];

/// What we publish about a mesh.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeshListing {
    /// Base64 invite token (others use this to --join)
    pub invite_token: String,
    /// Models currently loaded and serving inference
    pub serving: Vec<String>,
    /// Models the mesh wants but nobody is serving yet (need more GPUs)
    #[serde(default)]
    pub wanted: Vec<String>,
    /// Models on disk across the mesh (could be loaded if a GPU becomes free)
    #[serde(default)]
    pub on_disk: Vec<String>,
    /// Total VRAM across all GPU nodes (bytes)
    pub total_vram_bytes: u64,
    /// Number of GPU nodes in the mesh
    pub node_count: usize,
    /// Number of connected clients (API-only nodes)
    #[serde(default)]
    pub client_count: usize,
    /// Maximum clients this mesh accepts (0 = unlimited)
    #[serde(default)]
    pub max_clients: usize,
    /// Optional human-readable name for the mesh
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Optional geographic region
    #[serde(skip_serializing_if = "Option::is_none")]
    pub region: Option<String>,
    /// Stable mesh identity — all nodes in the same mesh share this ID.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mesh_id: Option<String>,
}

/// Discovered mesh from Nostr.
#[derive(Debug, Clone, serde::Serialize)]
pub struct DiscoveredMesh {
    pub listing: MeshListing,
    pub publisher_npub: String,
    pub published_at: u64,
    pub expires_at: Option<u64>,
}

impl std::fmt::Display for DiscoveredMesh {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let vram_gb = self.listing.total_vram_bytes as f64 / 1e9;
        let models = if self.listing.serving.is_empty() {
            "(no models loaded)".to_string()
        } else {
            self.listing.serving.join(", ")
        };
        write!(
            f,
            "{}  {} node(s), {:.0}GB capacity  serving: {}",
            self.listing.name.as_deref().unwrap_or("(unnamed)"),
            self.listing.node_count,
            vram_gb,
            models,
        )?;
        if let Some(ref region) = self.listing.region {
            write!(f, "  region: {}", region)?;
        }
        if !self.listing.wanted.is_empty() {
            write!(f, "  wanted: {}", self.listing.wanted.join(", "))?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Keys — stored in ~/.mesh-llm/nostr.nsec
// ---------------------------------------------------------------------------

fn nostr_key_path() -> Result<std::path::PathBuf> {
    let home =
        dirs::home_dir().ok_or_else(|| anyhow::anyhow!("Cannot determine home directory"))?;
    Ok(home.join(".mesh-llm").join("nostr.nsec"))
}

/// Load or generate a Nostr keypair for publishing.
pub fn load_or_create_keys() -> Result<Keys> {
    load_or_create_keys_at(&nostr_key_path()?)
}

fn load_or_create_keys_at(path: &std::path::Path) -> Result<Keys> {
    if let Some(parent) = path.parent() {
        ensure_private_nostr_dir(parent)?;
    }

    if path.exists() {
        ensure_private_nostr_key_file(path)?;
        let nsec = std::fs::read_to_string(path)?;
        let sk = SecretKey::from_bech32(nsec.trim())?;
        Ok(Keys::new(sk))
    } else {
        let keys = Keys::generate();
        let nsec = keys.secret_key().to_bech32()?;
        crate::crypto::write_keystore_bytes_atomically(path, nsec.as_bytes())?;
        tracing::info!("Generated new Nostr key, saved to {}", path.display());
        Ok(keys)
    }
}

#[cfg(unix)]
fn ensure_private_nostr_dir(dir: &std::path::Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;

    std::fs::create_dir_all(dir)?;
    let metadata = std::fs::metadata(dir)?;
    let mut perms = metadata.permissions();
    if perms.mode() & 0o077 != 0 {
        perms.set_mode(0o700);
        std::fs::set_permissions(dir, perms)?;
    }
    Ok(())
}

#[cfg(not(unix))]
fn ensure_private_nostr_dir(dir: &std::path::Path) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    Ok(())
}

#[cfg(unix)]
fn ensure_private_nostr_key_file(path: &std::path::Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;

    let metadata = std::fs::symlink_metadata(path)?;
    if !metadata.file_type().is_file() {
        anyhow::bail!("Nostr key path is not a regular file");
    }
    let mut perms = metadata.permissions();
    if perms.mode() & 0o077 != 0 {
        perms.set_mode(0o600);
        std::fs::set_permissions(path, perms)?;
    }
    Ok(())
}

#[cfg(not(unix))]
fn ensure_private_nostr_key_file(_path: &std::path::Path) -> Result<()> {
    Ok(())
}

/// Delete the Nostr key and node identity key.  After rotation the
/// node gets a fresh identity on next start.
pub fn rotate_keys() -> Result<()> {
    let home =
        dirs::home_dir().ok_or_else(|| anyhow::anyhow!("Cannot determine home directory"))?;
    let mesh_dir = home.join(".mesh-llm");

    let nostr_path = nostr_key_path()?;
    if nostr_path.exists() {
        std::fs::remove_file(&nostr_path)?;
        eprintln!("🔑 Deleted {}", nostr_path.display());
    } else {
        eprintln!("No Nostr key to rotate (none exists yet).");
    }

    let node_key_path = mesh_dir.join("key");
    if node_key_path.exists() {
        std::fs::remove_file(&node_key_path)?;
        eprintln!("🔑 Deleted {}", node_key_path.display());
    } else {
        eprintln!("No node key to rotate (none exists yet).");
    }

    eprintln!();
    eprintln!("✅ Keys rotated. New identities will be generated on next start.");
    Ok(())
}

// ---------------------------------------------------------------------------
// Publisher — background task that keeps the listing fresh
// ---------------------------------------------------------------------------

pub struct Publisher {
    client: Client,
    keys: Keys,
}

impl Publisher {
    pub async fn new(keys: Keys, relays: &[String]) -> Result<Self> {
        let _ = rustls::crypto::ring::default_provider().install_default();
        let client = Client::new(keys.clone());
        for relay in relays {
            client.add_relay(relay).await?;
        }
        client.connect().await;
        Ok(Self { client, keys })
    }

    pub fn npub(&self) -> String {
        self.keys.public_key().to_bech32().unwrap_or_default()
    }

    /// Publish (or replace) the mesh listing. Uses a replaceable event
    /// (kind 31990 + d-tag) so each publisher has exactly one listing.
    pub async fn publish(&self, listing: &MeshListing, ttl_secs: u64) -> Result<()> {
        let expiration = Timestamp::now().as_secs() + ttl_secs;
        let content = serde_json::to_string(listing)?;

        let tags = vec![
            Tag::custom(TagKind::Custom("d".into()), vec!["mesh-llm".to_string()]),
            Tag::custom(TagKind::Custom("k".into()), vec!["mesh-llm".to_string()]),
            Tag::custom(
                TagKind::Custom("expiration".into()),
                vec![expiration.to_string()],
            ),
        ];

        let builder = EventBuilder::new(Kind::Custom(MESH_SERVICE_KIND), content).tags(tags);
        self.client.send_event_builder(builder).await?;
        Ok(())
    }

    /// Delete our listing (e.g. on shutdown).
    pub async fn unpublish(&self) -> Result<()> {
        // Fetch our own events
        let filter = Filter::new()
            .kind(Kind::Custom(MESH_SERVICE_KIND))
            .author(self.keys.public_key())
            .limit(10);
        let events = self
            .client
            .fetch_events(filter, Duration::from_secs(5))
            .await?;
        for event in events.iter() {
            let request = EventDeletionRequest::new().id(event.id);
            let _ = self
                .client
                .send_event_builder(EventBuilder::delete(request))
                .await;
        }
        Ok(())
    }
}

/// Background publish loop. Republishes every `interval` seconds using
/// fresh data from the mesh node.
///
/// If `max_clients` is set, delists when that many clients are connected
/// and re-publishes when clients drop below the cap.
pub async fn publish_loop(
    node: crate::mesh::Node,
    keys: Keys,
    relays: Vec<String>,
    name: Option<String>,
    region: Option<String>,
    max_clients: Option<usize>,
    interval_secs: u64,
) {
    let publisher = match Publisher::new(keys.clone(), &relays).await {
        Ok(p) => p,
        Err(e) => {
            tracing::error!("Failed to create Nostr publisher: {e}");
            return;
        }
    };

    let npub = publisher.npub();
    if let Some(cap) = max_clients {
        eprintln!("   Will delist when {} clients connected", cap);
    }

    // Wait for llama-server to be ready before first publish (up to 60s).
    for _ in 0..120 {
        if node.is_llama_ready().await {
            break;
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
    eprintln!(
        "📡 Publishing mesh to Nostr (npub: {}...{})",
        &npub[..12],
        &npub[npub.len() - 8..]
    );

    let mut delisted = false;

    // Reusable client for solo-convergence discovery checks.
    let disco = DiscoveryClient::new(&relays).await.ok();

    loop {
        let invite_token = node.invite_token();
        let peers = node.peers().await;

        // Count clients
        let client_count = peers
            .iter()
            .filter(|p| matches!(p.role, crate::mesh::NodeRole::Client))
            .count();

        // Check max-clients cap
        if let Some(cap) = max_clients {
            if client_count >= cap && !delisted {
                if let Err(e) = publisher.unpublish().await {
                    tracing::warn!("Failed to unpublish from Nostr: {e}");
                }
                eprintln!(
                    "📡 Delisted from Nostr ({} clients, cap is {})",
                    client_count, cap
                );
                delisted = true;
                tokio::time::sleep(Duration::from_secs(interval_secs)).await;
                continue;
            } else if client_count < cap && delisted {
                eprintln!(
                    "📡 Re-publishing to Nostr ({} clients, cap is {})",
                    client_count, cap
                );
                delisted = false;
            }
        }

        if delisted {
            tokio::time::sleep(Duration::from_secs(interval_secs)).await;
            continue;
        }

        // ── Solo convergence: if we have no GPU peers, look for a mesh to join ──
        // First try split-heal (same mesh_id, different publisher, more nodes).
        // Then try merging with any other mesh (different mesh, unnamed only).
        // Only merge into meshes strictly larger than us to avoid two solo nodes
        // endlessly unpublishing and trying to join each other.
        let gpu_peers = peers
            .iter()
            .filter(|p| !matches!(p.role, crate::mesh::NodeRole::Client))
            .count();
        let my_node_count = gpu_peers + 1; // peers + self
        if gpu_peers == 0 {
            let filter = MeshFilter::default();
            if let Ok(listings) = discover(&relays, &filter, disco.as_ref()).await {
                let my_npub = publisher.npub();
                let my_mesh_id = node.mesh_id().await;

                // 1. Split-heal: same mesh_id from a different publisher with more nodes than us
                let split_target = my_mesh_id.as_ref().and_then(|mid| {
                    listings.iter().find(|m| {
                        m.listing.mesh_id.as_deref() == Some(mid.as_str())
                            && m.publisher_npub != my_npub
                            && m.listing.node_count > my_node_count
                    })
                });

                // 2. Merge: any unnamed mesh from a different publisher that is
                //    strictly larger than us. Two solo nodes (both node_count=1)
                //    must NOT try to merge — that creates a storm where both
                //    unpublish and race to join each other.
                let merge_target = if split_target.is_none() && name.is_none() {
                    listings.iter().find(|m| {
                        m.publisher_npub != my_npub
                            && m.listing.name.is_none()
                            && m.listing.node_count > my_node_count
                    })
                } else {
                    None
                };

                if let Some(target) = split_target.or(merge_target) {
                    eprintln!(
                        "📡 Found larger mesh '{}' ({} nodes vs our {}) — rejoining",
                        target.listing.name.as_deref().unwrap_or("unnamed"),
                        target.listing.node_count,
                        my_node_count
                    );
                    if let Err(e) = publisher.unpublish().await {
                        tracing::warn!("Failed to unpublish solo listing: {e}");
                    }
                    if let Err(e) = node.join(&target.listing.invite_token).await {
                        tracing::warn!("Merge/rejoin failed: {e}");
                        tokio::time::sleep(Duration::from_secs(interval_secs)).await;
                        continue;
                    }
                    eprintln!("📡 Merged into mesh — resuming publish as member");
                    // Cooldown: give the mesh time to stabilize before checking again
                    tokio::time::sleep(Duration::from_secs(30)).await;
                    continue;
                }
            }
        }

        // "Actually serving" = a Host node has llama-server running for this model.
        let my_role = node.role().await;
        let mut actually_serving: Vec<String> = Vec::new();
        if matches!(my_role, crate::mesh::NodeRole::Host { .. }) {
            for model in node.hosted_models().await {
                if !actually_serving.contains(&model) {
                    actually_serving.push(model);
                }
            }
        }
        for p in &peers {
            if matches!(p.role, crate::mesh::NodeRole::Host { .. }) {
                for model in p.routable_models() {
                    if !actually_serving.contains(&model) {
                        actually_serving.push(model);
                    }
                }
            }
        }

        let served_set: std::collections::HashSet<&str> =
            actually_serving.iter().map(|s| s.as_str()).collect();

        // Wanted = models with active demand but not currently served by a host
        let active_demand = node.active_demand().await;
        let mut wanted: Vec<String> = Vec::new();
        for m in active_demand.keys() {
            if !served_set.contains(m.as_str()) && !wanted.contains(m) {
                wanted.push(m.clone());
            }
        }

        // Available = all GGUFs on disk across mesh, minus what's already warm
        let mut available: Vec<String> = Vec::new();
        let my_available = node.available_models().await;
        for m in &my_available {
            if !served_set.contains(m.as_str()) && !available.contains(m) {
                available.push(m.clone());
            }
        }
        for p in &peers {
            for m in &p.available_models {
                if !served_set.contains(m.as_str()) && !available.contains(m) {
                    available.push(m.clone());
                }
            }
        }

        let total_vram: u64 = peers
            .iter()
            .filter(|p| !matches!(p.role, crate::mesh::NodeRole::Client))
            .map(|p| p.vram_bytes)
            .sum::<u64>()
            + node.vram_bytes();

        let node_count = peers
            .iter()
            .filter(|p| !matches!(p.role, crate::mesh::NodeRole::Client))
            .count()
            + 1; // +1 for self

        let mesh_id = node.mesh_id().await;

        let listing = MeshListing {
            invite_token,
            serving: actually_serving,
            wanted,
            on_disk: available,
            total_vram_bytes: total_vram,
            node_count,
            client_count,
            max_clients: max_clients.unwrap_or(0),
            name: name.clone(),
            region: region.clone(),
            mesh_id,
        };

        let ttl = interval_secs * 2;
        match publisher.publish(&listing, ttl).await {
            Ok(()) => tracing::debug!(
                "Published mesh listing ({} models, {} nodes, {} clients)",
                listing.serving.len(),
                listing.node_count,
                client_count
            ),
            Err(e) => tracing::warn!("Failed to publish to Nostr: {e}"),
        }

        tokio::time::sleep(Duration::from_secs(interval_secs)).await;
    }
}

// ---------------------------------------------------------------------------
// Publish watchdog — take over publishing if the original publisher dies
// ---------------------------------------------------------------------------

/// Watch for our mesh's Nostr listing to disappear, then start publishing.
/// Multiple nodes may start publishing simultaneously — that's fine, each
/// publishes with their own key and invite token, giving discoverers
/// multiple entry points to the same mesh.
///
/// Only runs on active (non-client) nodes that joined via `--auto`.
pub async fn publish_watchdog(
    node: crate::mesh::Node,
    relays: Vec<String>,
    mesh_name: Option<String>,
    region: Option<String>,
    check_interval_secs: u64,
) {
    // Short initial wait with jitter (10-30s) — start watching quickly
    let jitter = (rand::random::<u64>() % 20) + 10;
    tokio::time::sleep(Duration::from_secs(jitter)).await;

    // Reusable client for repeated discovery checks.
    let disco = DiscoveryClient::new(&relays).await.ok();

    loop {
        // Check if any listing for our mesh exists on Nostr
        let filter = MeshFilter::default();
        match discover(&relays, &filter, disco.as_ref()).await {
            Ok(meshes) => {
                let our_peers = node.peers().await;
                let served = node.models_being_served().await;
                let our_mesh_id = node.mesh_id().await;

                // Our mesh is "listed" if any Nostr listing carries our mesh_id.
                // Fall back to model overlap only if we don't have a mesh_id yet.
                let mesh_listed = if let Some(ref mid) = our_mesh_id {
                    meshes
                        .iter()
                        .any(|m| m.listing.mesh_id.as_deref() == Some(mid.as_str()))
                } else if !served.is_empty() {
                    meshes
                        .iter()
                        .any(|m| served.iter().any(|s| m.listing.serving.contains(s)))
                } else {
                    false
                };

                if !mesh_listed && (!our_peers.is_empty() || !served.is_empty()) {
                    // Brief backoff with jitter to avoid stampede (3-10s)
                    let backoff = (rand::random::<u64>() % 7) + 3;
                    eprintln!("📡 Mesh listing missing from Nostr — waiting {backoff}s before taking over...");
                    tokio::time::sleep(Duration::from_secs(backoff)).await;

                    // Re-check — maybe another watchdog already took over
                    if let Ok(recheck) = discover(&relays, &filter, disco.as_ref()).await {
                        let still_missing = if let Some(ref mid) = our_mesh_id {
                            !recheck
                                .iter()
                                .any(|m| m.listing.mesh_id.as_deref() == Some(mid.as_str()))
                        } else if !served.is_empty() {
                            !recheck
                                .iter()
                                .any(|m| served.iter().any(|s| m.listing.serving.contains(s)))
                        } else {
                            true
                        };
                        if !still_missing {
                            eprintln!("📡 Someone else took over publishing — standing down");
                            tokio::time::sleep(Duration::from_secs(check_interval_secs)).await;
                            continue;
                        }
                    }

                    eprintln!("📡 Taking over Nostr publishing for the mesh");
                    let keys = match load_or_create_keys() {
                        Ok(k) => k,
                        Err(e) => {
                            tracing::warn!("Failed to load Nostr keys for publish takeover: {e}");
                            tokio::time::sleep(Duration::from_secs(check_interval_secs)).await;
                            continue;
                        }
                    };
                    // Start publish loop (blocks forever)
                    publish_loop(node, keys, relays, mesh_name, region, None, 60).await;
                    return;
                }
            }
            Err(e) => {
                tracing::debug!("Publish watchdog: Nostr check failed: {e}");
            }
        }

        // Check frequently so we catch gaps fast
        let next_check = (rand::random::<u64>() % 15) + 20; // 20-35s
        tokio::time::sleep(Duration::from_secs(next_check)).await;
    }
}

// ---------------------------------------------------------------------------
// Discovery — find meshes on Nostr
// ---------------------------------------------------------------------------

/// Criteria for filtering discovered meshes.
#[derive(Debug, Clone, Default)]
pub struct MeshFilter {
    /// Match meshes serving (or wanting) this model name (substring match)
    pub model: Option<String>,
    /// Minimum total VRAM in GB
    pub min_vram_gb: Option<f64>,
    /// Geographic region
    pub region: Option<String>,
}

impl MeshFilter {
    pub fn matches(&self, mesh: &DiscoveredMesh) -> bool {
        if let Some(ref model) = self.model {
            let model_lower = model.to_lowercase();
            let has_model = mesh
                .listing
                .serving
                .iter()
                .any(|m| m.to_lowercase().contains(&model_lower))
                || mesh
                    .listing
                    .wanted
                    .iter()
                    .any(|m| m.to_lowercase().contains(&model_lower))
                || mesh
                    .listing
                    .on_disk
                    .iter()
                    .any(|m| m.to_lowercase().contains(&model_lower));
            if !has_model {
                return false;
            }
        }
        if let Some(min_gb) = self.min_vram_gb {
            let vram_gb = mesh.listing.total_vram_bytes as f64 / 1e9;
            if vram_gb < min_gb {
                return false;
            }
        }
        if let Some(ref region) = self.region {
            match &mesh.listing.region {
                Some(r) if r.eq_ignore_ascii_case(region) => {}
                _ => return false,
            }
        }
        true
    }
}

/// A reusable read-only Nostr client for discovery.
/// Create once, pass to repeated `discover()` calls to avoid opening
/// new websocket connections and generating throwaway keys every time.
pub struct DiscoveryClient {
    client: Client,
}

impl DiscoveryClient {
    pub async fn new(relays: &[String]) -> Result<Self> {
        let _ = rustls::crypto::ring::default_provider().install_default();
        let keys = Keys::generate();
        let client = Client::new(keys);
        let mut added = 0;
        for relay in relays {
            match client.add_relay(relay).await {
                Ok(_) => added += 1,
                Err(e) => tracing::warn!("Nostr relay {relay}: {e}"),
            }
        }
        if added == 0 {
            anyhow::bail!(
                "Could not connect to any Nostr relay (tried {})",
                relays.len()
            );
        }
        client.connect().await;
        Ok(Self { client })
    }
}

/// Discover meshes from Nostr relays.
///
/// If `cached_client` is provided, reuses its connections.  Otherwise
/// creates (and drops) a one-shot client — fine for the initial
/// `--auto` join but wasteful in tight loops.
pub async fn discover(
    relays: &[String],
    filter: &MeshFilter,
    cached_client: Option<&DiscoveryClient>,
) -> Result<Vec<DiscoveredMesh>> {
    // Build a temporary client only when no cached one is supplied.
    let _tmp;
    let client: &Client = if let Some(cc) = cached_client {
        &cc.client
    } else {
        let _ = rustls::crypto::ring::default_provider().install_default();
        let keys = Keys::generate();
        let c = Client::new(keys);
        let mut added = 0;
        for relay in relays {
            match c.add_relay(relay).await {
                Ok(_) => added += 1,
                Err(e) => tracing::warn!("Nostr relay {relay}: {e}"),
            }
        }
        if added == 0 {
            anyhow::bail!(
                "Could not connect to any Nostr relay (tried {})",
                relays.len()
            );
        }
        c.connect().await;
        _tmp = c;
        &_tmp
    };

    let nostr_filter = Filter::new()
        .kind(Kind::Custom(MESH_SERVICE_KIND))
        .custom_tag(
            SingleLetterTag::lowercase(Alphabet::K),
            "mesh-llm".to_string(),
        )
        .limit(100);

    let events = match client
        .fetch_events(nostr_filter, Duration::from_secs(5))
        .await
    {
        Ok(e) => e,
        Err(e) => {
            tracing::warn!("Nostr fetch failed: {e}");
            return Ok(Vec::new()); // No results rather than hard error
        }
    };

    let now = Timestamp::now().as_secs();

    // Dedupe by publisher (keep latest per pubkey, using replaceable event semantics)
    let mut latest: std::collections::HashMap<String, &Event> = std::collections::HashMap::new();
    for event in events.iter() {
        let pubkey = event.pubkey.to_hex();
        if let Some(existing) = latest.get(&pubkey) {
            if event.created_at.as_secs() > existing.created_at.as_secs() {
                latest.insert(pubkey, event);
            }
        } else {
            latest.insert(pubkey, event);
        }
    }

    let mut meshes = Vec::new();
    for event in latest.values() {
        // Check expiration
        let expires_at = event
            .tags
            .iter()
            .find(|t| t.as_slice().first().map(|s| s.as_str()) == Some("expiration"))
            .and_then(|t| t.as_slice().get(1))
            .and_then(|s| s.parse::<u64>().ok());

        if let Some(exp) = expires_at {
            if exp < now {
                continue; // expired
            }
        }

        let listing: MeshListing = match serde_json::from_str(&event.content) {
            Ok(l) => l,
            Err(_) => continue,
        };

        let publisher_npub = event.pubkey.to_bech32().unwrap_or_default();
        let discovered = DiscoveredMesh {
            listing,
            publisher_npub,
            published_at: event.created_at.as_secs(),
            expires_at,
        };

        if filter.matches(&discovered) {
            meshes.push(discovered);
        }
    }

    // Sort by node count (bigger meshes first), then VRAM
    meshes.sort_by(|a, b| {
        b.listing
            .node_count
            .cmp(&a.listing.node_count)
            .then(b.listing.total_vram_bytes.cmp(&a.listing.total_vram_bytes))
    });

    Ok(meshes)
}

// ---------------------------------------------------------------------------
// Smart auto-join: score meshes, detect staleness, prefer geo match
// ---------------------------------------------------------------------------

/// Score a mesh for auto-join. Higher = better.
/// Considers region match, capacity, and model availability.
/// Freshness is mostly irrelevant since Nostr listings expire at 120s (TTL=2×60s),
/// so anything we see from discover() is already reasonably fresh.
pub fn score_mesh(mesh: &DiscoveredMesh, _now_secs: u64, last_mesh_id: Option<&str>) -> i64 {
    let mut score: i64 = 100; // base score — if we can see it, it's alive

    // Target mesh name: very strong bonus if user asked for a specific mesh
    // Default "mesh-llm" name: moderate bonus for the community mesh
    // Other named meshes: penalty (they're someone's private group)
    if let Some(ref name) = mesh.listing.name {
        if name.eq_ignore_ascii_case("mesh-llm") {
            score += 300; // prefer the default community mesh
        } else {
            score -= 200; // named mesh — probably someone's private group
        }
    }

    // Sticky preference: strong bonus for the mesh we were last on
    if let (Some(last_id), Some(mesh_id)) = (last_mesh_id, &mesh.listing.mesh_id) {
        if last_id == mesh_id {
            score += 500; // strong preference, not infinite — dead/degraded mesh loses on other factors
        }
    }

    // Capacity: prefer meshes that aren't full
    if mesh.listing.max_clients > 0 {
        if mesh.listing.client_count >= mesh.listing.max_clients {
            score -= 1000; // full — don't join
        } else {
            let headroom = mesh.listing.max_clients - mesh.listing.client_count;
            score += (headroom as i64).min(20); // some capacity bonus
        }
    }

    // Size: prefer meshes with more nodes (more resilient)
    score += (mesh.listing.node_count as i64).min(10) * 5;

    // Models: prefer meshes with more warm models
    score += (mesh.listing.serving.len() as i64) * 10;

    // Wanted models: mesh needs help — bonus if we'd be useful
    score += (mesh.listing.wanted.len() as i64) * 15;

    score
}

/// Decision from smart auto-join.
#[derive(Debug)]
pub enum AutoDecision {
    /// Ranked list of meshes to try joining (best first)
    Join {
        candidates: Vec<(String, DiscoveredMesh)>,
    },
    /// No suitable mesh found — start a new one with these models
    StartNew { models: Vec<String> },
}

/// Pick meshes to join, ranked by score, or decide to start a new one.
///
/// - Scores all discovered meshes (freshness, region, capacity)
/// - Filters out stale/full meshes
/// - Returns all viable candidates ranked by score so the caller
///   can probe each in order and fall back to the next on failure
pub fn smart_auto(
    meshes: &[DiscoveredMesh],
    my_vram_gb: f64,
    target_name: Option<&str>,
) -> AutoDecision {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let last_mesh_id = crate::mesh::load_last_mesh_id();

    // If target name is set, only consider meshes with that exact name
    let candidates: Vec<&DiscoveredMesh> = if let Some(target) = target_name {
        meshes
            .iter()
            .filter(|m| {
                m.listing
                    .name
                    .as_ref()
                    .map(|n| n.eq_ignore_ascii_case(target))
                    .unwrap_or(false)
            })
            .collect()
    } else {
        meshes.iter().collect()
    };

    // Score and rank
    let mut scored: Vec<(&DiscoveredMesh, i64)> = candidates
        .iter()
        .map(|m| (*m, score_mesh(m, now, last_mesh_id.as_deref())))
        .collect();
    scored.sort_by(|a, b| b.1.cmp(&a.1));

    // Collect viable candidates.
    // If the user specified --mesh-name, take all candidates (they already
    // filtered by name above — the user explicitly asked for this mesh).
    // Otherwise, require positive score to filter out stale/private meshes.
    let viable: Vec<(String, DiscoveredMesh)> = scored
        .iter()
        .filter(|(_, score)| target_name.is_some() || *score > 0)
        .map(|(m, _)| (m.listing.invite_token.clone(), (*m).clone()))
        .collect();

    if !viable.is_empty() {
        return AutoDecision::Join { candidates: viable };
    }

    // No suitable mesh — recommend models for a new one based on VRAM
    let models = default_models_for_vram(my_vram_gb);
    AutoDecision::StartNew { models }
}

/// Model tiers by VRAM requirement (approximate loaded size × 1.1 headroom).
/// Model tiers for auto-selection, ordered largest-first.
/// min_vram = file_size * 1.1 rounded up. Prefer Qwen3 over 2.5 at same tier.
/// Parse a size string like "2.5GB" to GB as f64.
fn parse_size_gb(s: &str) -> f64 {
    s.trim_end_matches("GB").parse::<f64>().unwrap_or(0.0)
}

/// Build model tiers from the catalog, sorted largest first.
/// Each entry is (model_name, min_vram_gb) where min_vram = file_size * 1.1.
/// Excludes draft models (< 1GB).
fn model_tiers() -> Vec<(String, f64)> {
    let mut tiers: Vec<_> = crate::models::catalog::MODEL_CATALOG
        .iter()
        .filter(|m| parse_size_gb(&m.size) >= 1.0) // skip drafts
        .map(|m| (m.name.clone(), parse_size_gb(&m.size) * 1.1))
        .collect();
    tiers.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    tiers
}

/// Pick the model to SERVE for `--auto` based on VRAM.
/// Returns a single-element vec (the model this node should load).
///
/// One model per node. Biggest model that fits with 15% KV cache headroom.
///
/// Tiers:
///   <8GB:    Qwen3-4B (2.5G)
///   8-24GB:  Qwen3-8B (5G)
///   24-50GB: Qwen3.5-27B (17G) — vision + text
///   50-63GB: GLM-4.7-Flash (18G) — 30B MoE, fast, tool calling
///   63-179GB: Qwen3-Coder-Next (48G) — frontier coder ~85B
///   179GB+:  MiniMax-M2.5 (138G) — 456B MoE flagship
pub fn auto_model_pack(vram_gb: f64) -> Vec<String> {
    let local_models = crate::models::scan_local_models();
    let tiers = model_tiers();

    // Helper: check if a model is on disk
    let on_disk = |name: &str| local_models.contains(&name.to_string());
    // Helper: model size from tiers
    let size_of = |name: &str| -> f64 {
        tiers
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, s)| *s)
            .unwrap_or(f64::MAX)
    };
    let usable = vram_gb * 0.85; // 15% headroom for KV cache

    // Opinionated packs — each is (generalist, optional specialist(s))
    // The order within a tier prefers: on-disk first, then opinionated default.
    struct Pack {
        min_vram: f64,
        models: &'static [&'static str],
    }
    let packs: &[Pack] = &[
        // One model per tier. Node serves one model at a time.
        Pack {
            min_vram: 179.0,
            models: &["MiniMax-M2.5-Q4_K_M"],
        },
        Pack {
            min_vram: 63.0,
            models: &["Qwen3-Coder-Next-Q4_K_M"],
        },
        Pack {
            min_vram: 50.0,
            models: &["GLM-4.7-Flash-Q4_K_M"],
        },
        Pack {
            min_vram: 24.0,
            models: &["Qwen3.5-27B-Q4_K_M"],
        },
        Pack {
            min_vram: 8.0,
            models: &["Qwen3-8B-Q4_K_M"],
        },
        Pack {
            min_vram: 0.0,
            models: &["Qwen3-4B-Q4_K_M"],
        },
    ];

    // Find the best pack that fits
    for pack in packs {
        if vram_gb < pack.min_vram {
            continue;
        }
        // Check all models in the pack actually fit within usable VRAM
        let total: f64 = pack.models.iter().map(|m| size_of(m)).sum();
        if total <= usable {
            return pack.models.iter().map(|m| m.to_string()).collect();
        }
    }

    // Fallback: largest single model that fits, prefer on-disk
    let on_disk_fit = tiers
        .iter()
        .find(|(name, min_vram)| *min_vram <= usable && on_disk(name));
    let any_fit = tiers.iter().find(|(_, min_vram)| *min_vram <= usable);

    let primary = on_disk_fit
        .or(any_fit)
        .map(|(name, _)| name.to_string())
        .unwrap_or_else(|| "Qwen3-4B-Q4_K_M".into());

    vec![primary]
}

/// Models to advertise as "wanted" for demand seeding.
/// These tell other nodes what the mesh could use, covering every VRAM tier.
/// NOT served by this node — just demand hints for the mesh.
pub fn demand_seed_models() -> Vec<String> {
    vec![
        "Qwen3-Coder-Next-Q4_K_M".into(),
        "Qwen3.5-27B-Q4_K_M".into(),
        "GLM-4.7-Flash-Q4_K_M".into(),
        "Qwen3-8B-Q4_K_M".into(),
        "Qwen3-4B-Q4_K_M".into(),
        "Qwen3-0.6B-Q4_K_M".into(),
    ]
}

/// Legacy wrapper — returns serving models + demand seeds combined.
/// Used by `smart_auto` for the StartNew decision.
pub fn default_models_for_vram(vram_gb: f64) -> Vec<String> {
    let mut models = auto_model_pack(vram_gb);
    for m in demand_seed_models() {
        if !models.contains(&m) {
            models.push(m);
        }
    }
    models
}

#[cfg(test)]
mod auto_pack_tests {
    use super::*;

    #[test]
    fn pack_4gb_starter() {
        let pack = auto_model_pack(4.0);
        assert_eq!(pack, vec!["Qwen3-4B-Q4_K_M"]);
    }

    #[test]
    fn pack_8gb_single_model() {
        let pack = auto_model_pack(8.0);
        assert_eq!(pack, vec!["Qwen3-8B-Q4_K_M"]);
    }

    #[test]
    fn pack_16gb_single() {
        let pack = auto_model_pack(16.0);
        assert_eq!(pack, vec!["Qwen3-8B-Q4_K_M"]);
    }

    #[test]
    fn pack_24gb_vision() {
        let pack = auto_model_pack(24.0);
        assert_eq!(pack, vec!["Qwen3.5-27B-Q4_K_M"]);
    }

    #[test]
    fn pack_50gb_glm_flash() {
        let pack = auto_model_pack(50.0);
        assert_eq!(pack, vec!["GLM-4.7-Flash-Q4_K_M"]);
    }

    #[test]
    fn pack_63gb_frontier_coder() {
        let pack = auto_model_pack(63.0);
        assert_eq!(pack, vec!["Qwen3-Coder-Next-Q4_K_M"]);
    }

    #[test]
    fn pack_85gb_frontier_coder() {
        let pack = auto_model_pack(85.0);
        assert_eq!(pack, vec!["Qwen3-Coder-Next-Q4_K_M"]);
    }

    #[test]
    fn pack_206gb_minimax() {
        let pack = auto_model_pack(206.0);
        assert_eq!(pack, vec!["MiniMax-M2.5-Q4_K_M"]);
    }

    #[test]
    fn pack_between_tiers_falls_through() {
        // 40GB: below 50GB tier, falls to 24GB tier (Qwen3.5-27B)
        let pack = auto_model_pack(40.0);
        assert_eq!(pack, vec!["Qwen3.5-27B-Q4_K_M"]);
    }

    #[test]
    fn demand_seeds_are_separate() {
        let seeds = demand_seed_models();
        assert!(seeds.len() >= 4);
        assert!(seeds.contains(&"Qwen3-0.6B-Q4_K_M".to_string()));
        assert!(seeds.contains(&"Qwen3-Coder-Next-Q4_K_M".to_string()));
    }

    #[test]
    fn default_models_includes_both() {
        let all = default_models_for_vram(30.0);
        let pack = auto_model_pack(30.0);
        let seeds = demand_seed_models();
        // Pack models come first
        for m in &pack {
            assert!(
                all.contains(m),
                "pack model {m} missing from default_models"
            );
        }
        // Seeds are also present
        for m in &seeds {
            assert!(
                all.contains(m),
                "seed model {m} missing from default_models"
            );
        }
        // No duplicates
        let mut deduped = all.clone();
        deduped.sort();
        deduped.dedup();
        assert_eq!(all.len(), deduped.len());
    }
}

// ---------------------------------------------------------------------------
// Unit tests: score_mesh, smart_auto, MeshFilter
// ---------------------------------------------------------------------------
#[cfg(test)]
mod scoring_tests {
    use super::*;

    fn make_mesh(
        name: Option<&str>,
        mesh_id: Option<&str>,
        serving: &[&str],
        node_count: usize,
        vram: u64,
        clients: usize,
        max_clients: usize,
    ) -> DiscoveredMesh {
        DiscoveredMesh {
            listing: MeshListing {
                invite_token: format!("invite-{}", mesh_id.unwrap_or("test")),
                serving: serving.iter().map(|s| s.to_string()).collect(),
                wanted: vec![],
                on_disk: vec![],
                total_vram_bytes: vram,
                node_count,
                client_count: clients,
                max_clients,
                name: name.map(|s| s.to_string()),
                region: None,
                mesh_id: mesh_id.map(|s| s.to_string()),
            },
            publisher_npub: format!("npub-{}", mesh_id.unwrap_or("test")),
            published_at: 1000,
            expires_at: Some(2000),
        }
    }

    #[test]
    fn score_community_mesh_bonus() {
        let mesh = make_mesh(
            Some("mesh-llm"),
            Some("abc"),
            &["Qwen3-8B-Q4_K_M"],
            3,
            48_000_000_000,
            1,
            10,
        );
        let score = score_mesh(&mesh, 1500, None);
        // base(100) + community(300) + headroom + nodes(15) + models(10)
        assert!(score > 400, "community mesh should score high, got {score}");
    }

    #[test]
    fn score_private_mesh_penalty() {
        let mesh = make_mesh(
            Some("bobs-cluster"),
            Some("xyz"),
            &["Qwen3-8B-Q4_K_M"],
            3,
            48_000_000_000,
            0,
            0,
        );
        let score = score_mesh(&mesh, 1500, None);
        // base(100) - private(200) + nodes + models = low
        assert!(score < 100, "private mesh should score low, got {score}");
    }

    #[test]
    fn score_full_mesh_penalty() {
        let mesh = make_mesh(
            None,
            Some("full"),
            &["Qwen3-8B-Q4_K_M"],
            2,
            16_000_000_000,
            5,
            5,
        );
        let score = score_mesh(&mesh, 1500, None);
        assert!(score < 0, "full mesh should score negative, got {score}");
    }

    #[test]
    fn score_sticky_mesh_bonus() {
        let mesh = make_mesh(
            None,
            Some("my-mesh"),
            &["Qwen3-8B-Q4_K_M"],
            2,
            16_000_000_000,
            0,
            0,
        );
        let score_sticky = score_mesh(&mesh, 1500, Some("my-mesh"));
        let score_fresh = score_mesh(&mesh, 1500, None);
        assert!(
            score_sticky > score_fresh + 400,
            "sticky bonus should be large, sticky={score_sticky} fresh={score_fresh}"
        );
    }

    #[test]
    fn score_more_nodes_better() {
        let small = make_mesh(
            None,
            Some("s"),
            &["Qwen3-8B-Q4_K_M"],
            1,
            8_000_000_000,
            0,
            0,
        );
        let big = make_mesh(
            None,
            Some("b"),
            &["Qwen3-8B-Q4_K_M"],
            5,
            40_000_000_000,
            0,
            0,
        );
        assert!(score_mesh(&big, 1500, None) > score_mesh(&small, 1500, None));
    }

    #[test]
    fn score_more_models_better() {
        let one = make_mesh(
            None,
            Some("1"),
            &["Qwen3-8B-Q4_K_M"],
            2,
            16_000_000_000,
            0,
            0,
        );
        let two = make_mesh(
            None,
            Some("2"),
            &["Qwen3-8B-Q4_K_M", "Qwen3-32B-Q4_K_M"],
            2,
            40_000_000_000,
            0,
            0,
        );
        assert!(score_mesh(&two, 1500, None) > score_mesh(&one, 1500, None));
    }
}

#[cfg(test)]
mod filter_tests {
    use super::*;

    fn make_mesh_for_filter(
        serving: &[&str],
        wanted: &[&str],
        on_disk: &[&str],
        vram: u64,
        region: Option<&str>,
    ) -> DiscoveredMesh {
        DiscoveredMesh {
            listing: MeshListing {
                invite_token: "tok".into(),
                serving: serving.iter().map(|s| s.to_string()).collect(),
                wanted: wanted.iter().map(|s| s.to_string()).collect(),
                on_disk: on_disk.iter().map(|s| s.to_string()).collect(),
                total_vram_bytes: vram,
                node_count: 1,
                client_count: 0,
                max_clients: 0,
                name: None,
                region: region.map(|s| s.to_string()),
                mesh_id: None,
            },
            publisher_npub: "npub-test".into(),
            published_at: 1000,
            expires_at: Some(2000),
        }
    }

    #[test]
    fn filter_default_matches_all() {
        let m = make_mesh_for_filter(&["Qwen3-8B-Q4_K_M"], &[], &[], 8_000_000_000, None);
        assert!(MeshFilter::default().matches(&m));
    }

    #[test]
    fn filter_model_serving() {
        let m = make_mesh_for_filter(&["Qwen3-8B-Q4_K_M"], &[], &[], 8_000_000_000, None);
        let f = MeshFilter {
            model: Some("qwen3-8b".into()),
            ..Default::default()
        };
        assert!(f.matches(&m));
    }

    #[test]
    fn filter_model_wanted() {
        let m = make_mesh_for_filter(&[], &["Qwen3-32B-Q4_K_M"], &[], 8_000_000_000, None);
        let f = MeshFilter {
            model: Some("32b".into()),
            ..Default::default()
        };
        assert!(f.matches(&m));
    }

    #[test]
    fn filter_model_on_disk() {
        let m = make_mesh_for_filter(&[], &[], &["MiniMax-M2.5-Q4_K_M"], 8_000_000_000, None);
        let f = MeshFilter {
            model: Some("minimax".into()),
            ..Default::default()
        };
        assert!(f.matches(&m));
    }

    #[test]
    fn filter_model_no_match() {
        let m = make_mesh_for_filter(&["Qwen3-8B-Q4_K_M"], &[], &[], 8_000_000_000, None);
        let f = MeshFilter {
            model: Some("llama".into()),
            ..Default::default()
        };
        assert!(!f.matches(&m));
    }

    #[test]
    fn filter_min_vram() {
        let m = make_mesh_for_filter(&[], &[], &[], 8_000_000_000, None);
        let pass = MeshFilter {
            min_vram_gb: Some(5.0),
            ..Default::default()
        };
        let fail = MeshFilter {
            min_vram_gb: Some(16.0),
            ..Default::default()
        };
        assert!(pass.matches(&m));
        assert!(!fail.matches(&m));
    }

    #[test]
    fn filter_region() {
        let m = make_mesh_for_filter(&[], &[], &[], 8_000_000_000, Some("us-east"));
        let pass = MeshFilter {
            region: Some("us-east".into()),
            ..Default::default()
        };
        let fail = MeshFilter {
            region: Some("eu-west".into()),
            ..Default::default()
        };
        assert!(pass.matches(&m));
        assert!(!fail.matches(&m));
    }

    #[test]
    fn filter_region_case_insensitive() {
        let m = make_mesh_for_filter(&[], &[], &[], 8_000_000_000, Some("US-East"));
        let f = MeshFilter {
            region: Some("us-east".into()),
            ..Default::default()
        };
        assert!(f.matches(&m));
    }

    #[test]
    fn filter_combined() {
        let m = make_mesh_for_filter(
            &["Qwen3-8B-Q4_K_M"],
            &[],
            &[],
            16_000_000_000,
            Some("us-east"),
        );
        let pass = MeshFilter {
            model: Some("qwen3".into()),
            min_vram_gb: Some(10.0),
            region: Some("us-east".into()),
        };
        let fail_model = MeshFilter {
            model: Some("llama".into()),
            min_vram_gb: Some(10.0),
            region: Some("us-east".into()),
        };
        assert!(pass.matches(&m));
        assert!(!fail_model.matches(&m));
    }
}

#[cfg(test)]
mod smart_auto_tests {
    use super::*;

    fn make_mesh(
        name: Option<&str>,
        mesh_id: &str,
        serving: &[&str],
        node_count: usize,
        vram: u64,
        clients: usize,
        max_clients: usize,
    ) -> DiscoveredMesh {
        DiscoveredMesh {
            listing: MeshListing {
                invite_token: format!("invite-{mesh_id}"),
                serving: serving.iter().map(|s| s.to_string()).collect(),
                wanted: vec![],
                on_disk: vec![],
                total_vram_bytes: vram,
                node_count,
                client_count: clients,
                max_clients,
                name: name.map(|s| s.to_string()),
                region: None,
                mesh_id: Some(mesh_id.to_string()),
            },
            publisher_npub: format!("npub-{mesh_id}"),
            published_at: 1000,
            expires_at: Some(2000),
        }
    }

    #[test]
    fn smart_auto_prefers_community_mesh() {
        let meshes = vec![
            make_mesh(
                Some("mesh-llm"),
                "aaa",
                &["Qwen3-8B-Q4_K_M"],
                3,
                48_000_000_000,
                1,
                10,
            ),
            make_mesh(
                Some("bobs-cluster"),
                "bbb",
                &["Qwen3-8B-Q4_K_M"],
                5,
                80_000_000_000,
                0,
                0,
            ),
        ];
        match smart_auto(&meshes, 8.0, None) {
            AutoDecision::Join { candidates } => {
                assert!(!candidates.is_empty());
                // Community mesh should be first
                assert_eq!(candidates[0].0, "invite-aaa");
            }
            AutoDecision::StartNew { .. } => panic!("should join, not start new"),
        }
    }

    #[test]
    fn smart_auto_filters_full_mesh() {
        let meshes = vec![make_mesh(
            None,
            "full",
            &["Qwen3-8B-Q4_K_M"],
            2,
            16_000_000_000,
            10,
            10,
        )];
        match smart_auto(&meshes, 8.0, None) {
            AutoDecision::Join { candidates } => {
                // Full mesh should still appear (score might be negative but target_name is None
                // so it filters on score > 0)
                assert!(candidates.is_empty(), "full mesh should be filtered out");
            }
            AutoDecision::StartNew { models } => {
                assert!(!models.is_empty());
            }
        }
    }

    #[test]
    fn smart_auto_target_name_filters() {
        let meshes = vec![
            make_mesh(
                Some("mesh-llm"),
                "aaa",
                &["Qwen3-8B-Q4_K_M"],
                3,
                48_000_000_000,
                1,
                10,
            ),
            make_mesh(
                Some("private"),
                "bbb",
                &["Qwen3-32B-Q4_K_M"],
                2,
                40_000_000_000,
                0,
                0,
            ),
        ];
        match smart_auto(&meshes, 8.0, Some("private")) {
            AutoDecision::Join { candidates } => {
                assert!(!candidates.is_empty());
                // Only "private" mesh should match
                for (token, _) in &candidates {
                    assert_eq!(token, "invite-bbb");
                }
            }
            AutoDecision::StartNew { .. } => panic!("should find the named mesh"),
        }
    }

    #[test]
    fn smart_auto_empty_starts_new() {
        match smart_auto(&[], 24.0, None) {
            AutoDecision::StartNew { models } => {
                assert!(!models.is_empty());
            }
            AutoDecision::Join { .. } => panic!("no meshes should mean start new"),
        }
    }

    #[test]
    fn smart_auto_sticky_preference() {
        // Save a fake last-mesh
        let dir = dirs::home_dir().unwrap().join(".mesh-llm");
        let path = dir.join("last-mesh");
        let had_file = path.exists();
        let old_content = if had_file {
            std::fs::read_to_string(&path).ok()
        } else {
            None
        };

        // Write our test mesh_id
        std::fs::create_dir_all(&dir).ok();
        std::fs::write(&path, "sticky-mesh").ok();

        let meshes = vec![
            make_mesh(None, "other", &["Qwen3-8B-Q4_K_M"], 3, 24_000_000_000, 0, 0),
            make_mesh(
                None,
                "sticky-mesh",
                &["Qwen3-8B-Q4_K_M"],
                2,
                16_000_000_000,
                0,
                0,
            ),
        ];
        let result = smart_auto(&meshes, 8.0, None);

        // Restore
        if let Some(old) = old_content {
            std::fs::write(&path, old).ok();
        } else if had_file {
            // shouldn't happen but be safe
        } else {
            std::fs::remove_file(&path).ok();
        }

        match result {
            AutoDecision::Join { candidates } => {
                assert!(!candidates.is_empty());
                // Sticky mesh should be first despite fewer nodes
                assert_eq!(candidates[0].0, "invite-sticky-mesh");
            }
            AutoDecision::StartNew { .. } => panic!("should join"),
        }
    }
}

#[cfg(test)]
mod rotate_key_tests {
    use super::*;
    use serial_test::serial;
    use std::fs;

    // rotate_keys uses hardcoded paths (~/.mesh-llm/), so we test the logic
    // by verifying files are created/deleted in the real location.
    // This is safe because rotate_keys only deletes key and nostr.nsec.
    //
    // Both scenarios (keys present + keys missing) run in a single test to
    // avoid a race — Rust runs tests in parallel and both would touch the
    // same files.

    #[test]
    #[serial]
    fn rotate_deletes_both_keys_and_handles_missing() {
        let dir = dirs::home_dir().unwrap().join(".mesh-llm");
        fs::create_dir_all(&dir).ok();

        let key_path = dir.join("key");
        let nsec_path = dir.join("nostr.nsec");

        // Save originals so we can restore after the test.
        let orig_key = if key_path.exists() {
            Some(fs::read(&key_path).unwrap())
        } else {
            None
        };
        let orig_nsec = if nsec_path.exists() {
            Some(fs::read(&nsec_path).unwrap())
        } else {
            None
        };

        // --- Scenario 1: both keys exist → rotate deletes them ---
        fs::write(&key_path, b"test-node-key").unwrap();
        fs::write(&nsec_path, b"test-nostr-nsec").unwrap();

        let result = rotate_keys();
        assert!(result.is_ok(), "rotate should succeed when keys exist");
        assert!(!key_path.exists(), "node key should be deleted");
        assert!(!nsec_path.exists(), "nostr key should be deleted");

        // --- Scenario 2: no keys on disk → rotate still succeeds ---
        // (files were just deleted above, so the directory is clean)
        let result = rotate_keys();
        assert!(result.is_ok(), "rotate should succeed even with no keys");

        // Restore originals.
        if let Some(k) = orig_key {
            fs::write(&key_path, k).ok();
        }
        if let Some(n) = orig_nsec {
            fs::write(&nsec_path, n).ok();
        }
    }
}

#[cfg(test)]
mod key_file_tests {
    use super::*;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_key_path(prefix: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir()
            .join(format!("{prefix}-{unique}"))
            .join("nostr.nsec")
    }

    #[test]
    fn load_or_create_keys_at_round_trips() {
        let path = temp_key_path("mesh-llm-nostr-key");
        let first = load_or_create_keys_at(&path).unwrap();
        let second = load_or_create_keys_at(&path).unwrap();
        assert_eq!(
            first.secret_key().to_bech32().unwrap(),
            second.secret_key().to_bech32().unwrap()
        );
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[cfg(unix)]
    #[test]
    fn load_or_create_keys_at_hardens_existing_permissions() {
        use std::os::unix::fs::PermissionsExt;

        let path = temp_key_path("mesh-llm-nostr-key-perms");
        let dir = path.parent().unwrap();
        std::fs::create_dir_all(dir).unwrap();
        std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o755)).unwrap();

        let keys = Keys::generate();
        let nsec = keys.secret_key().to_bech32().unwrap();
        std::fs::write(&path, &nsec).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).unwrap();

        let loaded = load_or_create_keys_at(&path).unwrap();
        assert_eq!(
            loaded.secret_key().to_bech32().unwrap(),
            keys.secret_key().to_bech32().unwrap()
        );
        assert_eq!(
            std::fs::metadata(dir).unwrap().permissions().mode() & 0o777,
            0o700
        );
        assert_eq!(
            std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o600
        );

        let _ = std::fs::remove_dir_all(dir);
    }

    #[cfg(unix)]
    #[test]
    fn load_or_create_keys_at_rejects_symlink_key() {
        use std::os::unix::fs::PermissionsExt;

        let path = temp_key_path("mesh-llm-nostr-key-symlink");
        let dir = path.parent().unwrap();
        std::fs::create_dir_all(dir).unwrap();
        let real_file = dir.join("nostr.real");
        let keys = Keys::generate();
        let nsec = keys.secret_key().to_bech32().unwrap();
        std::fs::write(&real_file, &nsec).unwrap();
        std::fs::set_permissions(&real_file, std::fs::Permissions::from_mode(0o600)).unwrap();
        std::os::unix::fs::symlink(&real_file, &path).unwrap();

        let result = load_or_create_keys_at(&path);
        assert!(result.is_err(), "expected error for symlinked nostr key");

        let _ = std::fs::remove_dir_all(dir);
    }
}

// ---------------------------------------------------------------------------
// Integration test — publish/discover against real Nostr relays
// ---------------------------------------------------------------------------
#[cfg(test)]
mod integration_tests {
    use super::*;

    /// End-to-end: two publishers advertise the same mesh, a reusable
    /// DiscoveryClient finds both listings, and fields round-trip correctly.
    /// Covers publish, discover, multi-publisher, and client reuse in one test.
    #[tokio::test]
    async fn publish_discover_round_trip() {
        let relays: Vec<String> = DEFAULT_RELAYS.iter().map(|s| s.to_string()).collect();
        let mesh_name = format!("mesh-llm-test-{}", rand::random::<u32>());
        let mesh_id = format!("test-id-{}", rand::random::<u32>());

        // Publisher A
        let keys_a = Keys::generate();
        let pub_a = Publisher::new(keys_a.clone(), &relays)
            .await
            .expect("pub_a");
        let listing_a = MeshListing {
            invite_token: "invite-a".into(),
            serving: vec!["Qwen3-8B-Q4_K_M".into()],
            wanted: vec![],
            on_disk: vec![],
            total_vram_bytes: 16_000_000_000,
            node_count: 2,
            client_count: 0,
            max_clients: 0,
            name: Some(mesh_name.clone()),
            region: Some("test-region".into()),
            mesh_id: Some(mesh_id.clone()),
        };
        pub_a.publish(&listing_a, 120).await.expect("publish A");

        // Publisher B — same mesh, different invite token
        let keys_b = Keys::generate();
        let pub_b = Publisher::new(keys_b.clone(), &relays)
            .await
            .expect("pub_b");
        let mut listing_b = listing_a.clone();
        listing_b.invite_token = "invite-b".into();
        pub_b.publish(&listing_b, 120).await.expect("publish B");

        tokio::time::sleep(Duration::from_secs(3)).await;

        // Discover with reusable client (tests DiscoveryClient + discover)
        let dc = DiscoveryClient::new(&relays).await.expect("dc");
        let meshes = discover(&relays, &MeshFilter::default(), Some(&dc))
            .await
            .expect("discover");

        let found: Vec<_> = meshes
            .iter()
            .filter(|m| m.listing.mesh_id.as_deref() == Some(mesh_id.as_str()))
            .collect();
        assert!(
            found.len() >= 2,
            "should find both publishers for mesh_id={mesh_id}, found {}",
            found.len()
        );

        // Verify fields round-tripped
        let m = &found[0];
        assert_eq!(m.listing.name.as_deref(), Some(mesh_name.as_str()));
        assert_eq!(m.listing.serving, vec!["Qwen3-8B-Q4_K_M"]);
        assert_eq!(m.listing.node_count, 2);
        assert_eq!(m.listing.total_vram_bytes, 16_000_000_000);

        // Both invite tokens present
        let tokens: Vec<_> = found
            .iter()
            .map(|m| m.listing.invite_token.as_str())
            .collect();
        assert!(
            tokens.contains(&"invite-a"),
            "missing invite-a in {tokens:?}"
        );
        assert!(
            tokens.contains(&"invite-b"),
            "missing invite-b in {tokens:?}"
        );

        // Second discover with same client still works
        let r2 = discover(&relays, &MeshFilter::default(), Some(&dc))
            .await
            .expect("second discover");
        let found2: Vec<_> = r2
            .iter()
            .filter(|m| m.listing.mesh_id.as_deref() == Some(mesh_id.as_str()))
            .collect();
        assert!(found2.len() >= 2, "reused client should still find both");

        // Cleanup
        pub_a.unpublish().await.ok();
        pub_b.unpublish().await.ok();
    }
}
