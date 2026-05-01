use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::{Ipv4Addr, SocketAddrV4};
use std::time::{Duration, Instant};

const LAN_DISCOVERY_PORT: u16 = 19337;
const LAN_DISCOVERY_GROUP: Ipv4Addr = Ipv4Addr::new(239, 255, 71, 77);
const ANNOUNCE_INTERVAL: Duration = Duration::from_secs(3);

pub const DEFAULT_DISCOVERY_WAIT_SECS: u64 = 6;

#[derive(Debug, Clone, Default)]
pub struct MeshFilter {
    pub model: Option<String>,
    pub min_vram_gb: Option<f64>,
    pub region: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeshListing {
    pub invite_token: String,
    pub serving: Vec<String>,
    #[serde(default)]
    pub on_disk: Vec<String>,
    pub total_vram_bytes: u64,
    pub node_count: usize,
    #[serde(default)]
    pub client_count: usize,
    #[serde(default)]
    pub max_clients: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub region: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mesh_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MeshAnnouncement {
    listing: MeshListing,
    published_at: u64,
    announcer: String,
}

#[derive(Debug, Clone)]
pub struct DiscoveredMesh {
    pub listing: MeshListing,
    pub published_at: u64,
    pub announcer: String,
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
        Ok(())
    }
}

fn unix_now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn matches_filter(listing: &MeshListing, filter: &MeshFilter) -> bool {
    if let Some(ref needle) = filter.model {
        let needle = needle.to_lowercase();
        let has_model = listing
            .serving
            .iter()
            .chain(listing.on_disk.iter())
            .any(|m| m.to_lowercase().contains(&needle));
        if !has_model {
            return false;
        }
    }

    if let Some(min_vram_gb) = filter.min_vram_gb {
        let vram_gb = listing.total_vram_bytes as f64 / 1e9;
        if vram_gb < min_vram_gb {
            return false;
        }
    }

    if let Some(ref region) = filter.region {
        let region = region.to_lowercase();
        if listing.region.as_ref().map(|r| r.to_lowercase()).as_deref() != Some(region.as_str()) {
            return false;
        }
    }

    true
}

async fn current_listing(
    node: &crate::mesh::Node,
    mesh_name: Option<String>,
    region: Option<String>,
    max_clients: Option<usize>,
) -> MeshListing {
    let peers = node.peers().await;
    let is_self_client = matches!(node.role().await, crate::mesh::NodeRole::Client);
    let gpu_peer_count = peers
        .iter()
        .filter(|peer| !matches!(peer.role, crate::mesh::NodeRole::Client))
        .count();
    let client_count = peers
        .iter()
        .filter(|peer| matches!(peer.role, crate::mesh::NodeRole::Client))
        .count();

    let mut serving = node.serving_models().await;
    if serving.is_empty() {
        serving = node.models().await;
    }

    MeshListing {
        invite_token: node.invite_token(),
        serving,
        on_disk: node.available_models().await,
        total_vram_bytes: node.vram_bytes(),
        node_count: gpu_peer_count + usize::from(!is_self_client),
        client_count,
        max_clients: max_clients.unwrap_or(0),
        name: mesh_name,
        region,
        mesh_id: node.mesh_id().await,
    }
}

pub fn start_announce_loop(
    node: crate::mesh::Node,
    mesh_name: Option<String>,
    region: Option<String>,
    max_clients: Option<usize>,
) {
    tokio::spawn(async move {
        let socket = match tokio::net::UdpSocket::bind((Ipv4Addr::UNSPECIFIED, 0)).await {
            Ok(s) => s,
            Err(err) => {
                tracing::warn!("LAN announce bind failed: {err}");
                return;
            }
        };

        if let Err(err) = socket.set_multicast_loop_v4(true) {
            tracing::warn!("LAN announce multicast loop setup failed: {err}");
        }
        if let Err(err) = socket.set_multicast_ttl_v4(1) {
            tracing::warn!("LAN announce multicast TTL setup failed: {err}");
        }

        let destination = SocketAddrV4::new(LAN_DISCOVERY_GROUP, LAN_DISCOVERY_PORT);
        loop {
            let announcement = MeshAnnouncement {
                listing: current_listing(&node, mesh_name.clone(), region.clone(), max_clients)
                    .await,
                published_at: unix_now_secs(),
                announcer: node.id().fmt_short().to_string(),
            };

            if let Ok(payload) = serde_json::to_vec(&announcement) {
                if let Err(err) = socket.send_to(&payload, destination).await {
                    tracing::debug!("LAN announce send failed: {err}");
                }
            }
            tokio::time::sleep(ANNOUNCE_INTERVAL).await;
        }
    });
}

pub async fn discover(filter: &MeshFilter, wait_secs: u64) -> Result<Vec<DiscoveredMesh>> {
    let bind_addr = SocketAddrV4::new(Ipv4Addr::UNSPECIFIED, LAN_DISCOVERY_PORT);
    let std_socket = std::net::UdpSocket::bind(bind_addr)?;
    std_socket.set_nonblocking(true)?;
    std_socket.join_multicast_v4(&LAN_DISCOVERY_GROUP, &Ipv4Addr::UNSPECIFIED)?;
    let socket = tokio::net::UdpSocket::from_std(std_socket)?;

    let mut buffer = [0u8; 65_536];
    let mut by_token: HashMap<String, DiscoveredMesh> = HashMap::new();
    let deadline = Instant::now() + Duration::from_secs(wait_secs.max(1));

    while Instant::now() < deadline {
        let remaining = deadline.saturating_duration_since(Instant::now());
        let recv = tokio::time::timeout(
            remaining.min(Duration::from_millis(400)),
            socket.recv_from(&mut buffer),
        )
        .await;
        let Ok(Ok((size, _from))) = recv else {
            continue;
        };

        let Ok(announcement) = serde_json::from_slice::<MeshAnnouncement>(&buffer[..size]) else {
            continue;
        };

        if crate::mesh::Node::decode_invite_token(&announcement.listing.invite_token).is_err() {
            continue;
        }
        if !matches_filter(&announcement.listing, filter) {
            continue;
        }

        let token = announcement.listing.invite_token.clone();
        let discovered = DiscoveredMesh {
            listing: announcement.listing,
            published_at: announcement.published_at,
            announcer: announcement.announcer,
        };

        match by_token.get(&token) {
            Some(existing) if existing.published_at >= discovered.published_at => {}
            _ => {
                by_token.insert(token, discovered);
            }
        }
    }

    let mut discovered: Vec<DiscoveredMesh> = by_token.into_values().collect();
    discovered.sort_by_key(|mesh| std::cmp::Reverse(mesh.published_at));
    Ok(discovered)
}
