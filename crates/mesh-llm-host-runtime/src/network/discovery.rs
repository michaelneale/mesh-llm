use anyhow::{Context, Result};
use clap::ValueEnum;
use mdns_sd::{DaemonStatus, ResolvedService, ServiceDaemon, ServiceEvent, ServiceInfo};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::time::Duration;

use crate::network::nostr;

pub(crate) const LAN_SERVICE_TYPE: &str = "_mesh-llm._tcp.local.";
const TXT_SCHEMA_VERSION: u8 = 1;
const TXT_LIST_SEPARATOR: char = '|';
const TXT_VALUE_LIMIT: usize = 220;
const DAEMON_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(2);

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, ValueEnum)]
pub(crate) enum MeshDiscoveryMode {
    #[default]
    Nostr,
    Mdns,
}

impl MeshDiscoveryMode {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Nostr => "nostr",
            Self::Mdns => "mdns",
        }
    }

    pub(crate) fn source(self) -> &'static str {
        match self {
            Self::Nostr => "nostr-relay",
            Self::Mdns => "mdns-sd",
        }
    }

    pub(crate) fn scope(self) -> DiscoveryScope {
        match self {
            Self::Nostr => DiscoveryScope::Public,
            Self::Mdns => DiscoveryScope::Lan,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum DiscoveryScope {
    Public,
    Lan,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum LanJoinMaterial {
    RequiresSuppliedToken,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub(crate) struct LanMeshAdvertisement {
    pub(crate) mesh_id: Option<String>,
    pub(crate) mesh_name: Option<String>,
    pub(crate) region: Option<String>,
    pub(crate) serving_summary: Vec<String>,
    pub(crate) wanted_summary: Vec<String>,
    pub(crate) on_disk_summary: Vec<String>,
    pub(crate) total_vram_bytes: u64,
    pub(crate) node_count: usize,
    pub(crate) client_count: usize,
    pub(crate) max_clients: usize,
    pub(crate) token_fingerprint: Option<String>,
    pub(crate) app_version: Option<String>,
    pub(crate) join_material: LanJoinMaterial,
}

impl LanMeshAdvertisement {
    pub(crate) fn from_listing(
        listing: &nostr::MeshListing,
        supplied_invite_token: Option<&str>,
        app_version: Option<&str>,
    ) -> Self {
        let token_fingerprint = supplied_invite_token
            .filter(|token| !token.trim().is_empty())
            .map(lan_token_fingerprint)
            .or_else(|| {
                (!listing.invite_token.trim().is_empty())
                    .then(|| lan_token_fingerprint(&listing.invite_token))
            });

        Self {
            mesh_id: listing.mesh_id.clone(),
            mesh_name: listing.name.clone(),
            region: listing.region.clone(),
            serving_summary: bounded_list(&listing.serving),
            wanted_summary: bounded_list(&listing.wanted),
            on_disk_summary: bounded_list(&listing.on_disk),
            total_vram_bytes: listing.total_vram_bytes,
            node_count: listing.node_count,
            client_count: listing.client_count,
            max_clients: listing.max_clients,
            token_fingerprint,
            app_version: app_version.map(str::to_owned),
            join_material: LanJoinMaterial::RequiresSuppliedToken,
        }
    }

    pub(crate) fn matches_supplied_token(&self, supplied_invite_token: Option<&str>) -> bool {
        let Some(expected) = self.token_fingerprint.as_deref() else {
            return false;
        };
        supplied_invite_token
            .filter(|token| !token.trim().is_empty())
            .map(lan_token_fingerprint)
            .as_deref()
            == Some(expected)
    }

    pub(crate) fn to_txt_properties(&self) -> Result<Vec<(String, String)>> {
        let mut txt = vec![
            ("svc".to_string(), "mesh-llm".to_string()),
            ("schema".to_string(), TXT_SCHEMA_VERSION.to_string()),
            ("join".to_string(), "token-fingerprint".to_string()),
            ("nodes".to_string(), self.node_count.to_string()),
            ("clients".to_string(), self.client_count.to_string()),
            ("max_clients".to_string(), self.max_clients.to_string()),
            ("vram".to_string(), self.total_vram_bytes.to_string()),
            ("serving".to_string(), pack_txt_list(&self.serving_summary)),
            ("wanted".to_string(), pack_txt_list(&self.wanted_summary)),
            ("on_disk".to_string(), pack_txt_list(&self.on_disk_summary)),
        ];
        push_optional_txt(&mut txt, "mesh_id", self.mesh_id.as_deref());
        push_optional_txt(&mut txt, "name", self.mesh_name.as_deref());
        push_optional_txt(&mut txt, "region", self.region.as_deref());
        push_optional_txt(&mut txt, "tok_fp", self.token_fingerprint.as_deref());
        push_optional_txt(&mut txt, "version", self.app_version.as_deref());

        for (key, value) in &txt {
            anyhow::ensure!(
                key.len() + value.len() < u8::MAX as usize,
                "mDNS TXT property '{key}' exceeds DNS-SD length limit"
            );
        }
        Ok(txt)
    }

    #[cfg(test)]
    pub(crate) fn from_txt_properties(properties: &[(String, String)]) -> Result<Self> {
        let props: HashMap<&str, &str> = properties
            .iter()
            .map(|(key, value)| (key.as_str(), value.as_str()))
            .collect();

        parse_txt_properties(&props)
    }

    fn from_resolved_service(service: &ResolvedService) -> Result<Self> {
        let props = [
            "svc",
            "schema",
            "join",
            "nodes",
            "clients",
            "max_clients",
            "vram",
            "serving",
            "wanted",
            "on_disk",
            "mesh_id",
            "name",
            "region",
            "tok_fp",
            "version",
        ]
        .into_iter()
        .filter_map(|key| service.get_property_val_str(key).map(|value| (key, value)))
        .collect::<HashMap<_, _>>();

        parse_txt_properties(&props)
    }

    fn sanitized_listing(&self) -> nostr::MeshListing {
        nostr::MeshListing {
            invite_token: String::new(),
            serving: self.serving_summary.clone(),
            wanted: self.wanted_summary.clone(),
            on_disk: self.on_disk_summary.clone(),
            total_vram_bytes: self.total_vram_bytes,
            node_count: self.node_count,
            client_count: self.client_count,
            max_clients: self.max_clients,
            name: self.mesh_name.clone(),
            region: self.region.clone(),
            mesh_id: self.mesh_id.clone(),
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct LanDiscoveredMesh {
    pub(crate) mode: &'static str,
    pub(crate) scope: DiscoveryScope,
    pub(crate) source: &'static str,
    pub(crate) service_type: &'static str,
    pub(crate) instance_name: String,
    pub(crate) host: String,
    pub(crate) port: u16,
    pub(crate) addresses: Vec<String>,
    pub(crate) listing: nostr::MeshListing,
    pub(crate) token_fingerprint: Option<String>,
    pub(crate) join_material: LanJoinMaterial,
    pub(crate) joinable_with_supplied_token: bool,
    pub(crate) published_version: Option<String>,
    pub(crate) discovered_at: u64,
    #[serde(skip)]
    join_token: Option<String>,
}

impl LanDiscoveredMesh {
    pub(crate) fn join_token(&self) -> Option<&str> {
        self.join_token.as_deref()
    }

    pub(crate) fn to_join_candidate(&self) -> Option<(String, nostr::DiscoveredMesh)> {
        let token = self.join_token.clone()?;
        let mut listing = self.listing.clone();
        listing.invite_token = token.clone();
        Some((
            token,
            nostr::DiscoveredMesh {
                listing,
                publisher_npub: format!("mdns:{}", self.instance_name),
                published_at: self.discovered_at,
                expires_at: None,
            },
        ))
    }
}

pub(crate) struct LanPublishConfig {
    pub(crate) name: Option<String>,
    pub(crate) region: Option<String>,
    pub(crate) max_clients: Option<usize>,
    pub(crate) api_port: u16,
    pub(crate) interval_secs: u64,
    pub(crate) status_tx: Option<tokio::sync::watch::Sender<Option<nostr::PublishStateUpdate>>>,
}

pub(crate) async fn publish_lan_loop(node: crate::mesh::Node, config: LanPublishConfig) {
    let daemon = match ServiceDaemon::new() {
        Ok(daemon) => daemon,
        Err(err) => {
            tracing::warn!("Failed to create mDNS daemon: {err}");
            let _ = send_publish_state(&config.status_tx, nostr::PublishStateUpdate::PublishFailed);
            return;
        }
    };

    let instance_name = lan_instance_name(&node).await;
    let host_name = format!("{instance_name}.local.");
    eprintln!("Publishing mesh on local LAN via mDNS ({LAN_SERVICE_TYPE})");

    let mut last_reported = None;
    loop {
        let listing = build_local_mesh_listing(
            &node,
            config.name.clone(),
            config.region.clone(),
            config.max_clients,
        )
        .await;
        let advert = LanMeshAdvertisement::from_listing(
            &listing,
            Some(&listing.invite_token),
            Some(crate::VERSION),
        );

        let service_info = match service_info_for_advertisement(
            &advert,
            &instance_name,
            &host_name,
            config.api_port,
        ) {
            Ok(info) => info,
            Err(err) => {
                tracing::warn!("Failed to encode mDNS mesh advertisement: {err}");
                report_publish_state(
                    &config.status_tx,
                    &mut last_reported,
                    nostr::PublishStateUpdate::PublishFailed,
                );
                tokio::time::sleep(Duration::from_secs(config.interval_secs)).await;
                continue;
            }
        };

        match daemon.register(service_info) {
            Ok(()) => {
                report_publish_state(
                    &config.status_tx,
                    &mut last_reported,
                    nostr::PublishStateUpdate::Public,
                );
            }
            Err(err) => {
                tracing::warn!("Failed to register mDNS mesh advertisement: {err}");
                report_publish_state(
                    &config.status_tx,
                    &mut last_reported,
                    nostr::PublishStateUpdate::PublishFailed,
                );
            }
        }

        tokio::time::sleep(Duration::from_secs(config.interval_secs)).await;
    }
}

pub(crate) async fn discover_lan(
    filter: &nostr::MeshFilter,
    supplied_invite_token: Option<&str>,
    timeout: Duration,
) -> Result<Vec<LanDiscoveredMesh>> {
    let daemon = ServiceDaemon::new().context("create mDNS daemon")?;
    let receiver = match daemon.browse(LAN_SERVICE_TYPE) {
        Ok(receiver) => receiver,
        Err(err) => {
            shutdown_lan_daemon(daemon).await;
            return Err(anyhow::Error::new(err).context(format!("browse {LAN_SERVICE_TYPE}")));
        }
    };
    let deadline = tokio::time::Instant::now() + timeout;
    let mut by_instance: HashMap<String, LanDiscoveredMesh> = HashMap::new();

    while tokio::time::Instant::now() < deadline {
        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        if remaining.is_zero() {
            break;
        }
        let event = match tokio::time::timeout(remaining, receiver.recv_async()).await {
            Ok(Ok(event)) => event,
            Ok(Err(_)) => break,
            Err(_) => break,
        };

        let ServiceEvent::ServiceResolved(service) = event else {
            continue;
        };
        if !service.is_valid() {
            continue;
        }

        let advert = match LanMeshAdvertisement::from_resolved_service(service.as_ref()) {
            Ok(advert) => advert,
            Err(err) => {
                tracing::debug!(
                    "Skipping malformed mDNS mesh advertisement {}: {err}",
                    service.get_fullname()
                );
                continue;
            }
        };
        let listing = advert.sanitized_listing();
        let discovered = nostr::DiscoveredMesh {
            listing: listing.clone(),
            publisher_npub: format!("mdns:{}", service.get_fullname()),
            published_at: current_unix_secs(),
            expires_at: None,
        };
        if !filter.matches(&discovered) {
            continue;
        }

        let joinable = advert.matches_supplied_token(supplied_invite_token);
        by_instance.insert(
            service.get_fullname().to_string(),
            LanDiscoveredMesh {
                mode: MeshDiscoveryMode::Mdns.as_str(),
                scope: MeshDiscoveryMode::Mdns.scope(),
                source: MeshDiscoveryMode::Mdns.source(),
                service_type: LAN_SERVICE_TYPE,
                instance_name: service.get_fullname().to_string(),
                host: service.get_hostname().to_string(),
                port: service.get_port(),
                addresses: service
                    .get_addresses()
                    .iter()
                    .map(ToString::to_string)
                    .collect(),
                listing,
                token_fingerprint: advert.token_fingerprint.clone(),
                join_material: advert.join_material,
                joinable_with_supplied_token: joinable,
                published_version: advert.app_version.clone(),
                discovered_at: current_unix_secs(),
                join_token: joinable.then(|| supplied_invite_token.unwrap_or_default().to_string()),
            },
        );
    }

    drop(receiver);
    if let Err(err) = daemon.stop_browse(LAN_SERVICE_TYPE) {
        tracing::debug!("Failed to stop mDNS LAN browse before daemon shutdown: {err}");
    }
    shutdown_lan_daemon(daemon).await;
    let mut meshes = by_instance.into_values().collect::<Vec<_>>();
    meshes.sort_by(|left, right| {
        right
            .listing
            .node_count
            .cmp(&left.listing.node_count)
            .then(
                right
                    .listing
                    .total_vram_bytes
                    .cmp(&left.listing.total_vram_bytes),
            )
            .then(left.instance_name.cmp(&right.instance_name))
    });
    Ok(meshes)
}

async fn shutdown_lan_daemon(daemon: ServiceDaemon) -> bool {
    let shutdown = tokio::task::spawn_blocking(move || {
        match daemon.shutdown() {
            Ok(receiver) => receiver.recv_timeout(DAEMON_SHUTDOWN_TIMEOUT),
            Err(err) => {
                tracing::debug!("Failed to request mDNS daemon shutdown: {err}");
                return false;
            }
        }
        .map(|status| status == DaemonStatus::Shutdown)
        .unwrap_or(false)
    })
    .await
    .unwrap_or(false);

    if !shutdown {
        tracing::debug!("mDNS daemon shutdown did not report completion before timeout");
    }
    shutdown
}

pub(crate) async fn discover_lan_join_candidates(
    filter: &nostr::MeshFilter,
    supplied_invite_token: Option<&str>,
    timeout: Duration,
) -> Result<Vec<(String, nostr::DiscoveredMesh)>> {
    Ok(discover_lan(filter, supplied_invite_token, timeout)
        .await?
        .into_iter()
        .filter_map(|mesh| mesh.to_join_candidate())
        .collect())
}

pub(crate) fn lan_token_fingerprint(token: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"mesh-llm-lan-invite-token-v1\0");
    hasher.update(token.trim().as_bytes());
    let digest = hasher.finalize();
    hex::encode(&digest[..16])
}

pub(crate) fn discovery_source_label(mode: MeshDiscoveryMode, operation: &str) -> String {
    match mode {
        MeshDiscoveryMode::Nostr => format!("Nostr {operation}"),
        MeshDiscoveryMode::Mdns => format!("mDNS LAN {operation}"),
    }
}

async fn build_local_mesh_listing(
    node: &crate::mesh::Node,
    name: Option<String>,
    region: Option<String>,
    max_clients: Option<usize>,
) -> nostr::MeshListing {
    let peers = node.peers().await;
    let client_count = peers
        .iter()
        .filter(|p| matches!(p.role, crate::mesh::NodeRole::Client))
        .count();

    let my_role = node.role().await;
    let mut actually_serving: Vec<String> = Vec::new();
    if matches!(my_role, crate::mesh::NodeRole::Host { .. }) {
        for model in node.hosted_models().await {
            push_unique(&mut actually_serving, model);
        }
    }
    for peer in &peers {
        if matches!(peer.role, crate::mesh::NodeRole::Host { .. }) {
            for model in peer.routable_models() {
                push_unique(&mut actually_serving, model);
            }
        }
    }

    let served_set = actually_serving
        .iter()
        .map(String::as_str)
        .collect::<std::collections::HashSet<_>>();

    let active_demand = node.active_demand().await;
    let mut wanted = Vec::new();
    for model in active_demand.keys() {
        if !served_set.contains(model.as_str()) {
            push_unique(&mut wanted, model.clone());
        }
    }

    let mut available = Vec::new();
    for model in node.available_models().await {
        if !served_set.contains(model.as_str()) {
            push_unique(&mut available, model);
        }
    }
    for peer in &peers {
        for model in &peer.available_models {
            if !served_set.contains(model.as_str()) {
                push_unique(&mut available, model.clone());
            }
        }
    }

    let total_vram_bytes = peers
        .iter()
        .filter(|peer| !matches!(peer.role, crate::mesh::NodeRole::Client))
        .map(|peer| peer.vram_bytes)
        .sum::<u64>()
        + node.vram_bytes();
    let node_count = peers
        .iter()
        .filter(|peer| !matches!(peer.role, crate::mesh::NodeRole::Client))
        .count()
        + 1;

    nostr::MeshListing {
        invite_token: node.invite_token(),
        serving: actually_serving,
        wanted,
        on_disk: available,
        total_vram_bytes,
        node_count,
        client_count,
        max_clients: max_clients.unwrap_or(0),
        name,
        region,
        mesh_id: node.mesh_id().await,
    }
}

async fn lan_instance_name(node: &crate::mesh::Node) -> String {
    let identity = node
        .mesh_id()
        .await
        .unwrap_or_else(|| node.id().fmt_short().to_string());
    let suffix = sanitize_dns_label(&identity);
    format!("mesh-llm-{suffix}")
}

fn service_info_for_advertisement(
    advert: &LanMeshAdvertisement,
    instance_name: &str,
    host_name: &str,
    port: u16,
) -> Result<ServiceInfo> {
    let txt = advert.to_txt_properties()?;
    ServiceInfo::new(
        LAN_SERVICE_TYPE,
        instance_name,
        host_name,
        "",
        port,
        txt.as_slice(),
    )
    .map(ServiceInfo::enable_addr_auto)
    .context("create mDNS service info")
}

fn parse_txt_properties(props: &HashMap<&str, &str>) -> Result<LanMeshAdvertisement> {
    anyhow::ensure!(
        props.get("svc") == Some(&"mesh-llm"),
        "not a mesh-llm advertisement"
    );
    let schema = props
        .get("schema")
        .and_then(|value| value.parse::<u8>().ok())
        .unwrap_or(0);
    anyhow::ensure!(
        schema == TXT_SCHEMA_VERSION,
        "unsupported mDNS mesh schema version {schema}"
    );
    anyhow::ensure!(
        props.get("join") == Some(&"token-fingerprint"),
        "unsupported mDNS join material"
    );

    Ok(LanMeshAdvertisement {
        mesh_id: optional_txt(props, "mesh_id"),
        mesh_name: optional_txt(props, "name"),
        region: optional_txt(props, "region"),
        serving_summary: unpack_txt_list(props.get("serving").copied().unwrap_or_default()),
        wanted_summary: unpack_txt_list(props.get("wanted").copied().unwrap_or_default()),
        on_disk_summary: unpack_txt_list(props.get("on_disk").copied().unwrap_or_default()),
        total_vram_bytes: parse_txt_number(props, "vram")?,
        node_count: parse_txt_number(props, "nodes")?,
        client_count: parse_txt_number(props, "clients").unwrap_or(0),
        max_clients: parse_txt_number(props, "max_clients").unwrap_or(0),
        token_fingerprint: optional_txt(props, "tok_fp"),
        app_version: optional_txt(props, "version"),
        join_material: LanJoinMaterial::RequiresSuppliedToken,
    })
}

fn parse_txt_number<T>(props: &HashMap<&str, &str>, key: &str) -> Result<T>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    let value = props
        .get(key)
        .with_context(|| format!("missing mDNS TXT property '{key}'"))?;
    value
        .parse::<T>()
        .map_err(|err| anyhow::anyhow!("invalid mDNS TXT property '{key}': {err}"))
}

fn optional_txt(props: &HashMap<&str, &str>, key: &str) -> Option<String> {
    props
        .get(key)
        .map(|value| value.trim())
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

fn push_optional_txt(txt: &mut Vec<(String, String)>, key: &str, value: Option<&str>) {
    if let Some(value) = value.map(str::trim).filter(|value| !value.is_empty()) {
        txt.push((key.to_string(), truncate_txt_value(value)));
    }
}

fn bounded_list(values: &[String]) -> Vec<String> {
    values
        .iter()
        .filter_map(|value| {
            let trimmed = value.trim();
            (!trimmed.is_empty()).then(|| truncate_txt_value(trimmed))
        })
        .take(8)
        .collect()
}

fn pack_txt_list(values: &[String]) -> String {
    truncate_txt_value(
        &values
            .iter()
            .map(|value| value.replace(TXT_LIST_SEPARATOR, " "))
            .collect::<Vec<_>>()
            .join(&TXT_LIST_SEPARATOR.to_string()),
    )
}

fn unpack_txt_list(value: &str) -> Vec<String> {
    if value.trim().is_empty() {
        return Vec::new();
    }
    value
        .split(TXT_LIST_SEPARATOR)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .collect()
}

fn truncate_txt_value(value: &str) -> String {
    value.chars().take(TXT_VALUE_LIMIT).collect()
}

fn sanitize_dns_label(value: &str) -> String {
    let mut label = value
        .chars()
        .filter_map(|ch| {
            if ch.is_ascii_alphanumeric() {
                Some(ch.to_ascii_lowercase())
            } else if ch == '-' || ch == '_' {
                Some('-')
            } else {
                None
            }
        })
        .collect::<String>();
    label.truncate(48);
    let label = label.trim_matches('-');
    if label.is_empty() {
        "node".to_string()
    } else {
        label.to_string()
    }
}

fn push_unique(values: &mut Vec<String>, value: String) {
    if !values.contains(&value) {
        values.push(value);
    }
}

fn current_unix_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn report_publish_state(
    status_tx: &Option<tokio::sync::watch::Sender<Option<nostr::PublishStateUpdate>>>,
    last_reported: &mut Option<nostr::PublishStateUpdate>,
    next: nostr::PublishStateUpdate,
) {
    if *last_reported == Some(next) {
        return;
    }
    let _ = send_publish_state(status_tx, next);
    *last_reported = Some(next);
}

fn send_publish_state(
    status_tx: &Option<tokio::sync::watch::Sender<Option<nostr::PublishStateUpdate>>>,
    next: nostr::PublishStateUpdate,
) -> Result<(), tokio::sync::watch::error::SendError<Option<nostr::PublishStateUpdate>>> {
    if let Some(tx) = status_tx {
        tx.send(Some(next))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::network::nostr::MeshListing;

    fn sample_listing(invite_token: &str) -> MeshListing {
        MeshListing {
            invite_token: invite_token.to_string(),
            serving: vec!["Qwen3-8B-Q4_K_M".to_string()],
            wanted: vec!["Qwen3-32B-Q4_K_M".to_string()],
            on_disk: vec!["Qwen3-14B-Q4_K_M".to_string()],
            total_vram_bytes: 64_000_000_000,
            node_count: 2,
            client_count: 1,
            max_clients: 4,
            name: Some("lab-cluster".to_string()),
            region: Some("LAN".to_string()),
            mesh_id: Some("mesh-lab-01".to_string()),
        }
    }

    #[test]
    fn discovery_modes_have_stable_cli_names_and_metadata() {
        assert_eq!(MeshDiscoveryMode::default(), MeshDiscoveryMode::Nostr);
        assert_eq!(MeshDiscoveryMode::Nostr.as_str(), "nostr");
        assert_eq!(MeshDiscoveryMode::Nostr.source(), "nostr-relay");
        assert_eq!(MeshDiscoveryMode::Nostr.scope(), DiscoveryScope::Public);
        assert_eq!(MeshDiscoveryMode::Mdns.as_str(), "mdns");
        assert_eq!(MeshDiscoveryMode::Mdns.source(), "mdns-sd");
        assert_eq!(MeshDiscoveryMode::Mdns.scope(), DiscoveryScope::Lan);
    }

    #[test]
    fn lan_token_fingerprint_is_stable_and_does_not_expose_token() {
        let token = "very-secret-reusable-invite-token";
        let first = lan_token_fingerprint(token);
        let second = lan_token_fingerprint(token);

        assert_eq!(first, second);
        assert!(!first.contains(token));
        assert_ne!(first, lan_token_fingerprint("different-token"));
    }

    #[test]
    fn lan_advertisement_txt_round_trips_without_raw_invite_token() {
        let invite_token = "invite-token-that-must-not-leak";
        let listing = sample_listing(invite_token);
        let advert =
            LanMeshAdvertisement::from_listing(&listing, Some(invite_token), Some(crate::VERSION));

        let txt = advert.to_txt_properties().expect("txt should encode");
        let serialized = txt
            .iter()
            .map(|(key, value)| format!("{key}={value}"))
            .collect::<Vec<_>>()
            .join(";");

        assert!(!serialized.contains(invite_token));
        assert!(serialized.contains("tok_fp="));

        let decoded = LanMeshAdvertisement::from_txt_properties(&txt).expect("txt should decode");
        assert_eq!(decoded.mesh_id.as_deref(), Some("mesh-lab-01"));
        assert_eq!(decoded.mesh_name.as_deref(), Some("lab-cluster"));
        assert_eq!(decoded.serving_summary, vec!["Qwen3-8B-Q4_K_M"]);
        assert_eq!(
            decoded.token_fingerprint.as_deref(),
            Some(lan_token_fingerprint(invite_token).as_str())
        );
        assert_eq!(
            decoded.join_material,
            LanJoinMaterial::RequiresSuppliedToken
        );
    }

    #[test]
    fn lan_advertisement_requires_matching_supplied_join_token() {
        let invite_token = "invite-token-for-lab-mesh";
        let advert = LanMeshAdvertisement::from_listing(
            &sample_listing(invite_token),
            Some(invite_token),
            Some(crate::VERSION),
        );

        assert!(advert.matches_supplied_token(Some(invite_token)));
        assert!(!advert.matches_supplied_token(None));
        assert!(!advert.matches_supplied_token(Some("wrong-token")));
    }

    #[tokio::test]
    async fn shutdown_lan_daemon_reports_completion_when_available() {
        let Ok(daemon) = ServiceDaemon::new() else {
            eprintln!("mDNS daemon unavailable in this test environment");
            return;
        };

        assert!(shutdown_lan_daemon(daemon).await);
    }
}
