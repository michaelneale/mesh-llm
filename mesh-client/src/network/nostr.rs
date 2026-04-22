use anyhow::Result;
use nostr_sdk::prelude::*;
use serde::{Deserialize, Serialize};
use std::time::Duration;

pub const MESH_SERVICE_KIND: u16 = 31990;

pub const DEFAULT_RELAYS: &[&str] = &[
    "wss://relay.damus.io",
    "wss://nos.lol",
    "wss://relay.nostr.band",
    "wss://nostr.land",
    "wss://nostr.wine",
];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeshListing {
    pub invite_token: String,
    pub serving: Vec<String>,
    #[serde(default)]
    pub wanted: Vec<String>,
    #[serde(default)]
    pub on_disk: Vec<String>,
    pub total_vram_bytes: u64,
    pub node_count: usize,
    #[serde(default)]
    pub client_count: usize,
    #[serde(default)]
    pub max_clients: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub region: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mesh_id: Option<String>,
}

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
            "{}  {} node(s), {:.0}GB VRAM  serving: {}",
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

    pub async fn unpublish(&self) -> Result<()> {
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

#[derive(Debug, Clone, Default)]
pub struct MeshFilter {
    pub model: Option<String>,
    pub min_vram_gb: Option<f64>,
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

pub struct DiscoveryClient {
    client: Client,
}

impl DiscoveryClient {
    pub async fn new(keys: Keys, relays: &[String]) -> Result<Self> {
        let _ = rustls::crypto::ring::default_provider().install_default();
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

pub async fn discover(
    relays: &[String],
    filter: &MeshFilter,
    cached_client: Option<&DiscoveryClient>,
) -> Result<Vec<DiscoveredMesh>> {
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
            return Ok(Vec::new());
        }
    };

    let now = Timestamp::now().as_secs();

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
        let expires_at = event
            .tags
            .iter()
            .find(|t| t.as_slice().first().map(|s| s.as_str()) == Some("expiration"))
            .and_then(|t| t.as_slice().get(1))
            .and_then(|s| s.parse::<u64>().ok());

        if let Some(exp) = expires_at {
            if exp < now {
                continue;
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

    meshes.sort_by(|a, b| {
        b.listing
            .node_count
            .cmp(&a.listing.node_count)
            .then(b.listing.total_vram_bytes.cmp(&a.listing.total_vram_bytes))
    });

    Ok(meshes)
}

pub fn score_mesh(mesh: &DiscoveredMesh, _now_secs: u64, last_mesh_id: Option<&str>) -> i64 {
    let mut score: i64 = 100;

    if let Some(ref name) = mesh.listing.name {
        if name.eq_ignore_ascii_case("mesh-llm") {
            score += 300;
        } else {
            score -= 200;
        }
    }

    if let (Some(last_id), Some(mesh_id)) = (last_mesh_id, &mesh.listing.mesh_id) {
        if last_id == mesh_id {
            score += 500;
        }
    }

    if mesh.listing.max_clients > 0 {
        if mesh.listing.client_count >= mesh.listing.max_clients {
            score -= 1000;
        } else {
            let headroom = mesh.listing.max_clients - mesh.listing.client_count;
            score += (headroom as i64).min(20);
        }
    }

    score += (mesh.listing.node_count as i64).min(10) * 5;
    score += (mesh.listing.serving.len() as i64) * 10;
    score += (mesh.listing.wanted.len() as i64) * 15;

    score
}

#[derive(Debug)]
pub enum AutoDecision {
    Join {
        candidates: Vec<(String, DiscoveredMesh)>,
    },
    StartNew {
        models: Vec<String>,
    },
}

pub fn smart_auto(
    meshes: &[DiscoveredMesh],
    my_vram_gb: f64,
    target_name: Option<&str>,
    last_mesh_id: Option<&str>,
) -> AutoDecision {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

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

    let mut scored: Vec<(&DiscoveredMesh, i64)> = candidates
        .iter()
        .map(|m| (*m, score_mesh(m, now, last_mesh_id)))
        .collect();
    scored.sort_by_key(|entry| std::cmp::Reverse(entry.1));

    let viable: Vec<(String, DiscoveredMesh)> = scored
        .iter()
        .filter(|(_, score)| target_name.is_some() || *score > 0)
        .map(|(m, _)| (m.listing.invite_token.clone(), (*m).clone()))
        .collect();

    if !viable.is_empty() {
        return AutoDecision::Join { candidates: viable };
    }

    let models = default_models_for_vram(my_vram_gb);
    AutoDecision::StartNew { models }
}

fn parse_size_gb(s: &str) -> f64 {
    s.trim_end_matches("GB").parse::<f64>().unwrap_or(0.0)
}

fn model_tiers() -> Vec<(String, f64)> {
    let mut tiers: Vec<_> = crate::models::catalog::MODEL_CATALOG
        .iter()
        .filter(|m| parse_size_gb(&m.size) >= 1.0)
        .map(|m| (m.name.clone(), parse_size_gb(&m.size) * 1.1))
        .collect();
    tiers.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    tiers
}

pub fn auto_model_pack(vram_gb: f64) -> Vec<String> {
    let local_models: Vec<String> = Vec::new();
    let tiers = model_tiers();

    let on_disk = |name: &str| local_models.contains(&name.to_string());
    let size_of = |name: &str| -> f64 {
        tiers
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, s)| *s)
            .unwrap_or(f64::MAX)
    };
    let usable = vram_gb * 0.85;

    struct Pack {
        min_vram: f64,
        models: &'static [&'static str],
    }
    let packs: &[Pack] = &[
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

    for pack in packs {
        if vram_gb < pack.min_vram {
            continue;
        }
        let total: f64 = pack.models.iter().map(|m| size_of(m)).sum();
        if total <= usable {
            return pack.models.iter().map(|m| m.to_string()).collect();
        }
    }

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
        for m in &pack {
            assert!(
                all.contains(m),
                "pack model {m} missing from default_models"
            );
        }
        for m in &seeds {
            assert!(
                all.contains(m),
                "seed model {m} missing from default_models"
            );
        }
        let mut deduped = all.clone();
        deduped.sort();
        deduped.dedup();
        assert_eq!(all.len(), deduped.len());
    }
}

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
        match smart_auto(&meshes, 8.0, None, None) {
            AutoDecision::Join { candidates } => {
                assert!(!candidates.is_empty());
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
        match smart_auto(&meshes, 8.0, None, None) {
            AutoDecision::Join { candidates } => {
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
        match smart_auto(&meshes, 8.0, Some("private"), None) {
            AutoDecision::Join { candidates } => {
                assert!(!candidates.is_empty());
                for (token, _) in &candidates {
                    assert_eq!(token, "invite-bbb");
                }
            }
            AutoDecision::StartNew { .. } => panic!("should find the named mesh"),
        }
    }

    #[test]
    fn smart_auto_empty_starts_new() {
        match smart_auto(&[], 24.0, None, None) {
            AutoDecision::StartNew { models } => {
                assert!(!models.is_empty());
            }
            AutoDecision::Join { .. } => panic!("no meshes should mean start new"),
        }
    }

    #[test]
    fn smart_auto_sticky_preference() {
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
        match smart_auto(&meshes, 8.0, None, Some("sticky-mesh")) {
            AutoDecision::Join { candidates } => {
                assert!(!candidates.is_empty());
                assert_eq!(candidates[0].0, "invite-sticky-mesh");
            }
            AutoDecision::StartNew { .. } => panic!("should join"),
        }
    }
}
