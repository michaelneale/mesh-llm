//! Gossip protocol: peer announcement exchange, transitive peer tracking,
//! and peer list management (add/remove/update).

use super::*;

/// Minimum peer version we accept into the local mesh table and re-broadcast.
///
/// Peers below this floor are rejected at ingest in both `add_peer`
/// (direct gossip exchange) and `update_transitive_peer` (gossip relayed
/// by a bridge peer). They do not appear in `/api/status`, do not appear
/// in the UI, and are not included in outbound gossip. A peer that updates
/// and re-announces with a version at or above the floor is accepted on
/// the next exchange.
///
/// v0.60.0 is the cut where the on-wire `hardware` block landed; peers
/// older than that predate several gossip fields the current mesh relies
/// on. Peers that don't advertise a version at all (some legacy nodes
/// leave the field unset) are conservatively accepted, on the theory that
/// a missing version is more likely to be a legitimate old node than a
/// targeted bypass.
const MIN_REBROADCAST_VERSION_MAJOR: u64 = 0;
const MIN_REBROADCAST_VERSION_MINOR: u64 = 60;

/// Returns `true` if `version` is recent enough to include in outbound
/// gossip. `None` (no advertised version) returns `true` for back-compat.
/// Build metadata after `+` is stripped before parsing.
pub(super) fn version_allowed_for_rebroadcast(version: Option<&str>) -> bool {
    let Some(raw) = version else {
        return true;
    };
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return true;
    }
    // Strip build metadata ("0.65.1+skippy.20260504.kv.2" → "0.65.1") and
    // pre-release tag ("0.63.0-rc5" → "0.63.0") so the comparison is
    // purely on the major.minor numeric pair.
    let core = trimmed
        .split('+')
        .next()
        .unwrap_or(trimmed)
        .split('-')
        .next()
        .unwrap_or(trimmed);
    let mut parts = core.split('.');
    let Some(major) = parts.next().and_then(|s| s.parse::<u64>().ok()) else {
        return true; // Unparseable — don't penalise; conservative default.
    };
    let Some(minor) = parts.next().and_then(|s| s.parse::<u64>().ok()) else {
        return true;
    };
    if major != MIN_REBROADCAST_VERSION_MAJOR {
        // Any major > floor (e.g. v1.x.y) is allowed; any major < floor is
        // refused. With MIN_REBROADCAST_VERSION_MAJOR == 0, the "less than"
        // case cannot occur, but we keep the comparison structure for the
        // day the floor bumps to a non-zero major.
        return major > MIN_REBROADCAST_VERSION_MAJOR;
    }
    minor >= MIN_REBROADCAST_VERSION_MINOR
}

/// Returns `true` if the announcement describes a peer the mesh has no
/// observable use for via transitive gossip: a `Client`-role peer that
/// advertises **no identity** (no hostname), has **never been directly
/// measured** by any peer in the mesh, and has **no model interests**
/// (no requested/serving/hosted models).
///
/// Three independent signals must all be absent before we treat a peer
/// as a gossip-only ghost:
///
/// 1. `hostname` — populated synchronously by `system::hardware::survey()`
///    at node construction. Every real client on every supported platform
///    has one from its first gossip frame.
///
/// 2. `latency_source == Direct` — set when *any* peer in the mesh has
///    measured this peer's RTT via direct contact, then propagated
///    through gossip. A peer with a direct measurement is real — someone
///    reached it on the network. The v0.57 swarm uniformly has
///    `latency_source = Unknown`; no peer has ever directly contacted
///    one.
///
/// 3. model interests (`requested`/`serving`/`hosted`) — any of these
///    being populated makes the peer useful to the mesh (demand signal
///    or routable capacity).
///
/// A peer that fails all three is invisible to routing, untraceable on
/// the network, and contributes no demand signal. Real idle clients
/// survive: they have a hostname. Real reachable clients survive: they
/// have a direct measurement. Real demand-signaling clients survive:
/// they have a requested model.
///
/// Direct ingest in `add_peer` ignores this check — a client we actually
/// connect to is admitted regardless of what they advertise.
pub(super) fn peer_is_idle_transitive_client(ann: &PeerAnnouncement) -> bool {
    let directly_measured = matches!(
        ann.latency_source,
        Some(crate::proto::node::LatencySource::Direct)
    );
    matches!(ann.role, NodeRole::Client)
        && ann.hostname.is_none()
        && !directly_measured
        && ann.requested_models.is_empty()
        && ann.serving_models.is_empty()
        && ann
            .hosted_models
            .as_ref()
            .map(|h| h.is_empty())
            .unwrap_or(true)
}

struct LocalAnnouncementData {
    role: NodeRole,
    first_joined_mesh_ts: Option<u64>,
    models: Vec<String>,
    model_source: Option<String>,
    serving_models: Vec<String>,
    hosted_models: Vec<String>,
    available_models: Vec<String>,
    requested_models: Vec<String>,
    explicit_model_interests: Vec<String>,
    model_demand: HashMap<String, ModelDemand>,
    mesh_id: Option<String>,
    available_model_metadata: Vec<crate::proto::node::CompactModelMetadata>,
    available_model_sizes: HashMap<String, u64>,
    served_model_descriptors: Vec<ServedModelDescriptor>,
    served_model_runtime: Vec<ModelRuntimeDescriptor>,
    owner_attestation: Option<SignedNodeOwnership>,
    artifact_transfer_supported: bool,
    advertised_model_throughput: Vec<crate::network::metrics::ModelThroughputHint>,
    gpu_mem_bandwidth_gbps: Option<String>,
    gpu_compute_tflops_fp32: Option<String>,
    gpu_compute_tflops_fp16: Option<String>,
}

struct RebroadcastAnnouncements {
    announcements: Vec<PeerAnnouncement>,
    filtered_old_version: usize,
}

pub fn backfill_legacy_descriptors(ann: &mut PeerAnnouncement) {
    if ann.served_model_descriptors.is_empty() {
        let primary_model_name = ann
            .serving_models
            .first()
            .map(String::as_str)
            .unwrap_or_default()
            .to_string();
        ann.served_model_descriptors = infer_remote_served_descriptors(
            &primary_model_name,
            &ann.serving_models,
            ann.model_source.as_deref(),
        );
    }
}

pub(super) fn peer_meaningfully_changed(old: &PeerInfo, new: &PeerInfo) -> bool {
    old.addr != new.addr
        || old.role != new.role
        || old.first_joined_mesh_ts != new.first_joined_mesh_ts
        || old.models != new.models
        || old.vram_bytes != new.vram_bytes
        || old.rtt_ms != new.rtt_ms
        || old.model_source != new.model_source
        || old.serving_models != new.serving_models
        || old.hosted_models_known != new.hosted_models_known
        || old.hosted_models != new.hosted_models
        || old.available_models != new.available_models
        || old.requested_models != new.requested_models
        || old.explicit_model_interests != new.explicit_model_interests
        || old.served_model_descriptors != new.served_model_descriptors
        || old.served_model_runtime != new.served_model_runtime
        || old.artifact_transfer_supported != new.artifact_transfer_supported
        || old.stage_protocol_generation_supported != new.stage_protocol_generation_supported
        || old.stage_status_list_supported != new.stage_status_list_supported
        || old.version != new.version
        || old.owner_summary != new.owner_summary
        || old.gpu_reserved_bytes != new.gpu_reserved_bytes
        || old.propagated_latency != new.propagated_latency
}

fn merge_first_joined_mesh_ts(existing: &mut Option<u64>, incoming: Option<u64>) {
    match (*existing, incoming) {
        (None, Some(v)) => *existing = Some(v),
        (Some(_), None) => {}
        (Some(a), Some(b)) => *existing = Some(a.min(b)),
        (None, None) => {}
    }
}

pub(super) fn apply_transitive_ann(
    existing: &mut PeerInfo,
    addr: &EndpointAddr,
    ann: &PeerAnnouncement,
    bridge_id: EndpointId,
) -> bool {
    let ann_hosted_models = ann.hosted_models.clone().unwrap_or_default();
    let serving_changed = existing.serving_models != ann.serving_models
        || existing.hosted_models != ann_hosted_models
        || existing.hosted_models_known != ann.hosted_models.is_some();
    existing.serving_models = ann.serving_models.clone();
    existing.hosted_models = ann_hosted_models;
    existing.hosted_models_known = ann.hosted_models.is_some();
    existing.role = ann.role.clone();
    merge_first_joined_mesh_ts(&mut existing.first_joined_mesh_ts, ann.first_joined_mesh_ts);
    existing.vram_bytes = ann.vram_bytes;
    // Only advance addr if the transitive announcement is at least as path-rich,
    // so a direct peer's richer address is not overwritten by a weaker transitive one.
    if !addr.addrs.is_empty() && addr.addrs.len() >= existing.addr.addrs.len() {
        existing.addr = addr.clone();
    }
    if ann.version.is_some() {
        existing.version = ann.version.clone();
    }
    if ann.gpu_name.is_some() {
        existing.gpu_name = ann.gpu_name.clone();
    }
    if ann.hostname.is_some() {
        existing.hostname = ann.hostname.clone();
    }
    if ann.is_soc.is_some() {
        existing.is_soc = ann.is_soc;
    }
    if ann.gpu_vram.is_some() {
        existing.gpu_vram = ann.gpu_vram.clone();
    }
    if ann.gpu_reserved_bytes.is_some() {
        existing.gpu_reserved_bytes = ann.gpu_reserved_bytes.clone();
    }
    if ann.gpu_mem_bandwidth_gbps.is_some() {
        existing.gpu_mem_bandwidth_gbps = ann.gpu_mem_bandwidth_gbps.clone();
    }
    if ann.gpu_compute_tflops_fp32.is_some() {
        existing.gpu_compute_tflops_fp32 = ann.gpu_compute_tflops_fp32.clone();
    }
    if ann.gpu_compute_tflops_fp16.is_some() {
        existing.gpu_compute_tflops_fp16 = ann.gpu_compute_tflops_fp16.clone();
    }
    existing.models = ann.models.clone();
    existing.available_models.clear();
    existing.requested_models = ann.requested_models.clone();
    existing.explicit_model_interests = ann.explicit_model_interests.clone();
    existing.owner_attestation = ann.owner_attestation.clone();
    if ann.model_source.is_some() {
        existing.model_source = ann.model_source.clone();
    }
    existing.served_model_descriptors = ann.served_model_descriptors.clone();
    existing.served_model_runtime = ann.served_model_runtime.clone();
    existing.artifact_transfer_supported = ann.artifact_transfer_supported;
    existing.stage_protocol_generation_supported = ann.stage_protocol_generation_supported;
    existing.stage_status_list_supported = ann.stage_status_list_supported;
    existing.advertised_model_throughput = ann.advertised_model_throughput.clone();
    if ann.experts_summary.is_some() {
        existing.experts_summary = ann.experts_summary.clone();
    }
    // Propagate latency from the announcement (transitive gossip).
    if let Some(latency_ms) = ann.latency_ms {
        let source = ann
            .latency_source
            .unwrap_or(crate::proto::node::LatencySource::Unspecified);
        let is_propagatable_source = matches!(
            source,
            crate::proto::node::LatencySource::Direct
                | crate::proto::node::LatencySource::Estimated
        );
        if latency_ms > 0 && is_propagatable_source {
            let observer_id = ann
                .latency_observer_id
                .as_ref()
                .and_then(|id_bytes| EndpointId::from_bytes(id_bytes).ok());
            existing.propagated_latency = Some(PropagatedLatencyObservation {
                latency_ms,
                age_ms_at_received: ann.latency_age_ms.unwrap_or(0),
                received_at: std::time::Instant::now(),
                observer_id: observer_id.or(Some(bridge_id)),
            });
        }
    }
    serving_changed
}

impl Node {
    async fn apply_announced_peer(
        &self,
        remote: EndpointId,
        peer_id: EndpointId,
        addr: &EndpointAddr,
        ann: &PeerAnnouncement,
        rtt_ms: Option<u32>,
    ) {
        if peer_id == self.endpoint.id() {
            return;
        }
        if peer_id == remote {
            if let Some(ref their_id) = ann.mesh_id {
                self.set_mesh_id(their_id.clone()).await;
            }
            self.merge_remote_demand(&ann.model_demand);
            self.add_peer(remote, addr.clone(), ann).await;
            if let Some(rtt_ms) = rtt_ms {
                self.update_peer_rtt(remote, rtt_ms).await;
            }
            return;
        }
        self.update_transitive_peer(peer_id, addr, ann, remote)
            .await;
    }

    async fn apply_announced_peers(
        &self,
        remote: EndpointId,
        their_announcements: &[(EndpointAddr, PeerAnnouncement)],
        rtt_ms: Option<u32>,
    ) {
        for (addr, ann) in their_announcements {
            self.apply_announced_peer(remote, addr.id, addr, ann, rtt_ms)
                .await;
        }
    }

    async fn refresh_gossip_path_rtt(&self, remote: EndpointId, ceiling_rtt_ms: Option<u32>) {
        let conn = self.state.lock().await.connections.get(&remote).cloned();
        let Some(conn) = conn else {
            return;
        };
        let path_list = conn.paths();
        for path_info in &path_list {
            if !path_info.is_selected() {
                continue;
            }
            let path_rtt_ms = path_info.rtt().as_millis() as u32;
            if path_rtt_ms == 0 {
                continue;
            }
            if ceiling_rtt_ms.is_some_and(|ceiling| path_rtt_ms >= ceiling) {
                break;
            }
            let path_type = if path_info.is_ip() { "direct" } else { "relay" };
            super::emit_mesh_info(format!(
                "📡 Peer {} RTT: {}ms ({}){}",
                remote.fmt_short(),
                path_rtt_ms,
                path_type,
                if ceiling_rtt_ms.is_some() {
                    " [path info]"
                } else {
                    ""
                }
            ));
            self.update_peer_rtt(remote, path_rtt_ms).await;
            break;
        }
    }

    async fn maybe_connect_discovered_peer(
        &self,
        my_role: &super::NodeRole,
        addr: EndpointAddr,
        ann: &PeerAnnouncement,
        known_peer_check_uses_connections: bool,
        log_discovery_failure_as_warning: bool,
    ) {
        let peer_id = addr.id;
        if self.should_skip_discovered_peer(my_role, peer_id, ann)
            || self
                .discovered_peer_already_known(peer_id, known_peer_check_uses_connections)
                .await
            || Self::discovered_peer_is_filtered(peer_id, ann)
        {
            return;
        }
        if let Err(error) = Box::pin(self.connect_to_peer(addr)).await {
            if log_discovery_failure_as_warning {
                tracing::warn!("Failed to discover peer: {error}");
            } else {
                tracing::debug!(
                    "Could not connect to discovered peer {}: {error}",
                    peer_id.fmt_short()
                );
            }
        }
    }

    /// Returns `true` if the announcement would be rejected by the same
    /// gates that filter ingest. Skipping the dial here avoids spending
    /// 30s per host walking through unreachable ghost addresses
    /// sequentially in the gossip exchange dial loop — the wedge that
    /// caused `--auto` startup to hang.
    fn discovered_peer_is_filtered(peer_id: EndpointId, ann: &PeerAnnouncement) -> bool {
        if !version_allowed_for_rebroadcast(ann.version.as_deref())
            || peer_is_idle_transitive_client(ann)
        {
            tracing::debug!(
                "Skipping discovered peer {} (filtered: version={:?} role={:?})",
                peer_id.fmt_short(),
                ann.version,
                ann.role
            );
            return true;
        }
        false
    }

    fn should_skip_discovered_peer(
        &self,
        my_role: &super::NodeRole,
        peer_id: EndpointId,
        ann: &PeerAnnouncement,
    ) -> bool {
        peer_id == self.endpoint.id()
            || (matches!(my_role, super::NodeRole::Client)
                && matches!(ann.role, super::NodeRole::Client))
    }

    async fn discovered_peer_already_known(
        &self,
        peer_id: EndpointId,
        use_connections: bool,
    ) -> bool {
        let state = self.state.lock().await;
        if use_connections {
            state.connections.contains_key(&peer_id)
        } else {
            state.peers.contains_key(&peer_id)
        }
    }

    fn peer_hardware_changed(old_peer: &PeerInfo, updated_peer: &PeerInfo) -> bool {
        old_peer.gpu_name != updated_peer.gpu_name
            || old_peer.hostname != updated_peer.hostname
            || old_peer.is_soc != updated_peer.is_soc
            || old_peer.gpu_vram != updated_peer.gpu_vram
            || old_peer.gpu_reserved_bytes != updated_peer.gpu_reserved_bytes
            || old_peer.gpu_mem_bandwidth_gbps != updated_peer.gpu_mem_bandwidth_gbps
            || old_peer.gpu_compute_tflops_fp32 != updated_peer.gpu_compute_tflops_fp32
            || old_peer.gpu_compute_tflops_fp16 != updated_peer.gpu_compute_tflops_fp16
    }

    fn update_existing_direct_peer(
        existing: &mut PeerInfo,
        addr: EndpointAddr,
        ann: &PeerAnnouncement,
        owner_summary: OwnershipSummary,
        now: std::time::Instant,
    ) -> (PeerInfo, bool, bool, bool) {
        let old_peer = existing.clone();
        let role_changed = existing.role != ann.role;
        let ann_hosted_models = ann.hosted_models.clone().unwrap_or_default();
        let serving_changed = existing.serving_models != ann.serving_models
            || existing.hosted_models != ann_hosted_models
            || existing.hosted_models_known != ann.hosted_models.is_some();
        if role_changed {
            tracing::info!(
                "Peer {} role updated: {:?} → {:?}",
                existing.id.fmt_short(),
                existing.role,
                ann.role
            );
            existing.role = ann.role.clone();
        }
        if !addr.addrs.is_empty() {
            existing.addr = addr;
        }
        existing.models = ann.models.clone();
        merge_first_joined_mesh_ts(&mut existing.first_joined_mesh_ts, ann.first_joined_mesh_ts);
        existing.vram_bytes = ann.vram_bytes;
        if ann.model_source.is_some() {
            existing.model_source = ann.model_source.clone();
        }
        existing.serving_models = ann.serving_models.clone();
        existing.hosted_models = ann_hosted_models;
        existing.hosted_models_known = ann.hosted_models.is_some();
        existing.available_models.clear();
        existing
            .available_models
            .extend(ann.available_models.clone());
        existing.requested_models = ann.requested_models.clone();
        existing.explicit_model_interests = ann.explicit_model_interests.clone();
        existing.last_seen = now;
        existing.owner_attestation = ann.owner_attestation.clone();
        existing.owner_summary = owner_summary;
        existing.served_model_descriptors = ann.served_model_descriptors.clone();
        existing.served_model_runtime = ann.served_model_runtime.clone();
        existing.artifact_transfer_supported = ann.artifact_transfer_supported;
        existing.stage_protocol_generation_supported = ann.stage_protocol_generation_supported;
        existing.stage_status_list_supported = ann.stage_status_list_supported;
        existing.advertised_model_throughput = ann.advertised_model_throughput.clone();
        if ann.version.is_some() {
            existing.version = ann.version.clone();
        }
        existing.gpu_name = ann.gpu_name.clone();
        existing.hostname = ann.hostname.clone();
        existing.is_soc = ann.is_soc;
        existing.gpu_vram = ann.gpu_vram.clone();
        existing.gpu_reserved_bytes = ann.gpu_reserved_bytes.clone();
        existing.gpu_mem_bandwidth_gbps = ann.gpu_mem_bandwidth_gbps.clone();
        existing.gpu_compute_tflops_fp32 = ann.gpu_compute_tflops_fp32.clone();
        existing.gpu_compute_tflops_fp16 = ann.gpu_compute_tflops_fp16.clone();
        if ann.experts_summary.is_some() {
            existing.experts_summary = ann.experts_summary.clone();
        }
        let updated_peer = existing.clone();
        let changed = peer_meaningfully_changed(&old_peer, &updated_peer)
            || Self::peer_hardware_changed(&old_peer, &updated_peer);
        (updated_peer, changed, role_changed, serving_changed)
    }

    async fn remove_disallowed_peer(&self, id: EndpointId) {
        let mut state = self.state.lock().await;
        if state.peers.remove(&id).is_some() {
            let _ = self.peer_change_tx.send(state.peers.len());
        }
    }

    async fn direct_peer_owner_summary(
        &self,
        id: EndpointId,
        ann: &PeerAnnouncement,
    ) -> OwnershipSummary {
        let trust_store = self.trust_store.lock().await.clone();
        verify_node_ownership(
            ann.owner_attestation.as_ref(),
            id.as_bytes(),
            &trust_store,
            self.trust_policy,
            current_time_unix_ms(),
        )
    }

    async fn reject_direct_peer_for_policy(
        &self,
        id: EndpointId,
        owner_summary: &OwnershipSummary,
    ) -> bool {
        if policy_accepts_peer(self.trust_policy, owner_summary) {
            return false;
        }

        let mut state = self.state.lock().await;
        let last_status = state.policy_rejected_peers.get(&id).cloned();
        if last_status.as_ref() != Some(&owner_summary.status) {
            tracing::warn!(
                "Rejecting peer {} due to owner policy: {:?}",
                id.fmt_short(),
                owner_summary.status
            );
            state
                .policy_rejected_peers
                .insert(id, owner_summary.status.clone());
        }
        if state.peers.remove(&id).is_some() {
            let _ = self.peer_change_tx.send(state.peers.len());
        }
        true
    }

    async fn publish_direct_peer_update(
        &self,
        updated_peer: PeerInfo,
        changed: bool,
        should_publish_count: bool,
        count: usize,
    ) {
        if should_publish_count {
            let _ = self.peer_change_tx.send(count);
        }
        if changed {
            self.emit_plugin_mesh_event(
                crate::plugin::proto::mesh_event::Kind::PeerUpdated,
                Some(&updated_peer),
                String::new(),
            )
            .await;
        }
    }

    async fn upsert_existing_direct_peer(
        &self,
        id: EndpointId,
        addr: EndpointAddr,
        ann: &PeerAnnouncement,
        owner_summary: OwnershipSummary,
        now: std::time::Instant,
    ) -> bool {
        let mut state = self.state.lock().await;
        state.policy_rejected_peers.remove(&id);
        let Some(existing) = state.peers.get_mut(&id) else {
            return false;
        };
        let (updated_peer, changed, role_changed, serving_changed) =
            Self::update_existing_direct_peer(existing, addr, ann, owner_summary, now);
        let count = state.peers.len();
        let should_publish_count = role_changed || serving_changed;
        drop(state);
        self.publish_direct_peer_update(updated_peer, changed, should_publish_count, count)
            .await;
        true
    }

    async fn insert_new_direct_peer(
        &self,
        id: EndpointId,
        addr: EndpointAddr,
        ann: &PeerAnnouncement,
        owner_summary: OwnershipSummary,
    ) {
        let mut state = self.state.lock().await;
        state.policy_rejected_peers.remove(&id);
        tracing::info!(
            "Peer added: {} role={:?} vram={:.1}GB assigned={:?} catalog={:?} (total: {})",
            id.fmt_short(),
            ann.role,
            ann.vram_bytes as f64 / 1e9,
            ann.serving_models.first(),
            ann.available_models,
            state.peers.len() + 1
        );
        let peer = PeerInfo::from_announcement(id, addr, ann, owner_summary);
        state.peers.insert(id, peer.clone());
        let count = state.peers.len();
        drop(state);
        let _ = self.peer_change_tx.send(count);
        self.emit_plugin_mesh_event(
            crate::plugin::proto::mesh_event::Kind::PeerUp,
            Some(&peer),
            String::new(),
        )
        .await;
    }

    async fn collect_rebroadcast_announcements(
        &self,
        stale_cutoff: std::time::Instant,
    ) -> RebroadcastAnnouncements {
        let mut filtered_old_version = 0;
        let announcements = {
            let state = self.state.lock().await;
            state
                .peers
                .values()
                .filter(|peer| {
                    peer.last_seen >= stale_cutoff || peer.last_mentioned >= stale_cutoff
                })
                .filter(|peer| {
                    let allowed = version_allowed_for_rebroadcast(peer.version.as_deref());
                    if !allowed {
                        filtered_old_version += 1;
                    }
                    allowed
                })
                .map(Self::announcement_from_peer)
                .collect()
        };
        RebroadcastAnnouncements {
            announcements,
            filtered_old_version,
        }
    }

    async fn snapshot_local_announcement_data(&self) -> LocalAnnouncementData {
        let owner_summary = self.owner_summary.lock().await.clone();
        let hosted_models = self.hosted_models.lock().await.clone();
        let advertised_model_throughput = self
            .routing_metrics
            .advertisable_model_throughput(&hosted_models);
        LocalAnnouncementData {
            role: self.role.lock().await.clone(),
            first_joined_mesh_ts: *self.first_joined_mesh_ts.lock().await,
            models: self.models.lock().await.clone(),
            model_source: self.model_source.lock().await.clone(),
            serving_models: self.serving_models.lock().await.clone(),
            hosted_models,
            available_models: self.available_models.lock().await.clone(),
            requested_models: self.requested_models.lock().await.clone(),
            explicit_model_interests: self.explicit_model_interests.lock().await.clone(),
            model_demand: self.get_demand(),
            mesh_id: self.mesh_id.lock().await.clone(),
            available_model_metadata: Vec::new(),
            available_model_sizes: HashMap::new(),
            served_model_descriptors: self.served_model_descriptors.lock().await.clone(),
            served_model_runtime: self.model_runtime_descriptors.lock().await.clone(),
            owner_attestation: self.owner_attestation.lock().await.clone(),
            artifact_transfer_supported:
                crate::models::artifact_transfer::artifact_transfer_advertised(&owner_summary),
            advertised_model_throughput,
            gpu_mem_bandwidth_gbps: Self::format_optional_locked_f32_list(
                &self.gpu_mem_bandwidth_gbps,
            )
            .await,
            gpu_compute_tflops_fp32: Self::format_optional_locked_f32_list(
                &self.gpu_compute_tflops_fp32,
            )
            .await,
            gpu_compute_tflops_fp16: Self::format_optional_locked_f32_list(
                &self.gpu_compute_tflops_fp16,
            )
            .await,
        }
    }

    fn announcement_from_peer(peer: &PeerInfo) -> PeerAnnouncement {
        let latency = peer.display_latency();
        PeerAnnouncement {
            addr: peer.addr.clone(),
            role: peer.role.clone(),
            first_joined_mesh_ts: peer.first_joined_mesh_ts,
            models: peer.models.clone(),
            vram_bytes: peer.vram_bytes,
            model_source: peer.model_source.clone(),
            serving_models: peer.serving_models.clone(),
            hosted_models: peer.hosted_models_known.then(|| peer.hosted_models.clone()),
            available_models: peer.available_models.clone(),
            requested_models: peer.requested_models.clone(),
            explicit_model_interests: peer.explicit_model_interests.clone(),
            version: peer.version.clone(),
            model_demand: HashMap::new(),
            mesh_id: None,
            gpu_name: peer.gpu_name.clone(),
            hostname: peer.hostname.clone(),
            is_soc: peer.is_soc,
            gpu_vram: peer.gpu_vram.clone(),
            gpu_reserved_bytes: peer.gpu_reserved_bytes.clone(),
            gpu_mem_bandwidth_gbps: peer.gpu_mem_bandwidth_gbps.clone(),
            gpu_compute_tflops_fp32: peer.gpu_compute_tflops_fp32.clone(),
            gpu_compute_tflops_fp16: peer.gpu_compute_tflops_fp16.clone(),
            available_model_metadata: peer.available_model_metadata.clone(),
            experts_summary: peer.experts_summary.clone(),
            available_model_sizes: peer.available_model_sizes.clone(),
            served_model_descriptors: peer.served_model_descriptors.clone(),
            served_model_runtime: peer.served_model_runtime.clone(),
            owner_attestation: peer.owner_attestation.clone(),
            artifact_transfer_supported: peer.artifact_transfer_supported,
            stage_protocol_generation_supported: peer.stage_protocol_generation_supported,
            stage_status_list_supported: peer.stage_status_list_supported,
            advertised_model_throughput: peer.advertised_model_throughput.clone(),
            latency_ms: latency.latency_ms,
            latency_source: Some(match latency.source {
                DisplayLatencySource::Direct => crate::proto::node::LatencySource::Direct,
                DisplayLatencySource::Estimated => crate::proto::node::LatencySource::Estimated,
                DisplayLatencySource::Unknown => crate::proto::node::LatencySource::Unknown,
            }),
            latency_age_ms: Some(latency.age_ms),
            latency_observer_id: latency.observer_id,
        }
    }

    async fn format_optional_locked_f32_list(
        values: &tokio::sync::Mutex<Option<Vec<f64>>>,
    ) -> Option<String> {
        values.lock().await.as_ref().map(|values| {
            values
                .iter()
                .map(|f| format!("{:.2}", f))
                .collect::<Vec<_>>()
                .join(",")
        })
    }

    fn build_local_announcement(&self, data: LocalAnnouncementData) -> PeerAnnouncement {
        PeerAnnouncement {
            addr: self.endpoint_addr_for_advertisement(),
            role: data.role,
            first_joined_mesh_ts: data.first_joined_mesh_ts,
            models: data.models,
            vram_bytes: self.vram_bytes,
            model_source: data.model_source,
            serving_models: data.serving_models,
            hosted_models: Some(data.hosted_models),
            available_models: data.available_models,
            requested_models: data.requested_models,
            explicit_model_interests: data.explicit_model_interests,
            version: Some(crate::VERSION.to_string()),
            model_demand: data.model_demand,
            mesh_id: data.mesh_id,
            gpu_name: self.enumerate_host.then(|| self.gpu_name.clone()).flatten(),
            hostname: self.enumerate_host.then(|| self.hostname.clone()).flatten(),
            is_soc: self.is_soc,
            gpu_vram: self.enumerate_host.then(|| self.gpu_vram.clone()).flatten(),
            gpu_reserved_bytes: self
                .enumerate_host
                .then(|| self.gpu_reserved_bytes.clone())
                .flatten(),
            gpu_mem_bandwidth_gbps: data.gpu_mem_bandwidth_gbps,
            gpu_compute_tflops_fp32: data.gpu_compute_tflops_fp32,
            gpu_compute_tflops_fp16: data.gpu_compute_tflops_fp16,
            available_model_metadata: data.available_model_metadata,
            experts_summary: None,
            available_model_sizes: data.available_model_sizes,
            served_model_descriptors: data.served_model_descriptors,
            served_model_runtime: data.served_model_runtime,
            owner_attestation: data.owner_attestation,
            artifact_transfer_supported: data.artifact_transfer_supported,
            stage_protocol_generation_supported: true,
            stage_status_list_supported: true,
            advertised_model_throughput: data.advertised_model_throughput,
            latency_ms: None,
            latency_source: None,
            latency_age_ms: None,
            latency_observer_id: None,
        }
    }

    /// Open a gossip stream on an existing connection to exchange peer info.
    pub(super) async fn initiate_gossip(&self, conn: Connection, remote: EndpointId) -> Result<()> {
        // Timeout only the gossip round-trip. A misbehaving peer may accept the
        // QUIC connection and even the bi-stream but never send a gossip response,
        // blocking the join path indefinitely and preventing fallback to other
        // candidates.
        match tokio::time::timeout(
            PEER_CONNECT_AND_GOSSIP_TIMEOUT,
            self.gossip_round_trip(&conn, remote),
        )
        .await
        {
            Ok(Ok((their_announcements, rtt_ms))) => {
                self.apply_gossip_announcements(remote, rtt_ms, &their_announcements, true)
                    .await
            }
            Ok(Err(e)) => Err(e),
            Err(_) => anyhow::bail!(
                "gossip exchange with {} timed out ({}s)",
                remote.fmt_short(),
                PEER_CONNECT_AND_GOSSIP_TIMEOUT.as_secs()
            ),
        }
    }

    pub(super) async fn initiate_gossip_inner(
        &self,
        conn: Connection,
        remote: EndpointId,
        discover_peers: bool,
    ) -> Result<()> {
        let (their_announcements, rtt_ms) = self.gossip_round_trip(&conn, remote).await?;
        self.apply_gossip_announcements(remote, rtt_ms, &their_announcements, discover_peers)
            .await
    }

    async fn gossip_round_trip(
        &self,
        conn: &Connection,
        remote: EndpointId,
    ) -> Result<(Vec<(EndpointAddr, PeerAnnouncement)>, u32)> {
        let protocol = connection_protocol(conn);
        let t0 = std::time::Instant::now();
        let (mut send, mut recv) = conn.open_bi().await?;
        send.write_all(&[STREAM_GOSSIP]).await?;

        let our_announcements = self.collect_announcements().await;
        write_gossip_payload(&mut send, protocol, &our_announcements, self.endpoint.id()).await?;
        send.finish()?;

        let buf = read_len_prefixed(&mut recv).await?;
        let rtt_ms = t0.elapsed().as_millis() as u32;
        let their_announcements = decode_gossip_payload(protocol, remote, &buf)?;

        let _ = recv.read_to_end(0).await;
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        Ok((their_announcements, rtt_ms))
    }

    async fn apply_gossip_announcements(
        &self,
        remote: EndpointId,
        rtt_ms: u32,
        their_announcements: &[(EndpointAddr, PeerAnnouncement)],
        discover_peers: bool,
    ) -> Result<()> {
        self.apply_announced_peers(remote, their_announcements, Some(rtt_ms))
            .await;

        // Also check the connection's actual path info — the gossip round-trip
        // time above may reflect relay latency even if a direct path is now active.
        self.refresh_gossip_path_rtt(remote, Some(rtt_ms)).await;

        if discover_peers {
            let my_role = self.role.lock().await.clone();
            for (addr, ann) in their_announcements {
                self.maybe_connect_discovered_peer(&my_role, addr.clone(), ann, true, false)
                    .await;
            }
        }

        Ok(())
    }

    pub(super) async fn handle_gossip_stream(
        &self,
        remote: EndpointId,
        protocol: ControlProtocol,
        mut send: iroh::endpoint::SendStream,
        mut recv: iroh::endpoint::RecvStream,
    ) -> Result<()> {
        tracing::info!("Inbound gossip from {}", remote.fmt_short());

        {
            let mut state = self.state.lock().await;
            if state.dead_peers.remove(&remote).is_some() {
                super::emit_mesh_info(format!(
                    "🔄 Dead peer {} is gossiping — clearing dead status",
                    remote.fmt_short()
                ));
            }
        }

        let buf = read_len_prefixed(&mut recv).await?;
        let their_announcements = decode_gossip_payload(protocol, remote, &buf)?;

        let our_announcements = self.collect_announcements().await;
        write_gossip_payload(&mut send, protocol, &our_announcements, self.endpoint.id()).await?;
        send.finish()?;

        let _ = recv.read_to_end(0).await;

        self.apply_announced_peers(remote, &their_announcements, None)
            .await;
        self.refresh_gossip_path_rtt(remote, None).await;

        let my_role = self.role.lock().await.clone();
        for (addr, ann) in their_announcements {
            self.maybe_connect_discovered_peer(&my_role, addr, &ann, false, true)
                .await;
        }

        Ok(())
    }
    pub(super) async fn remove_peer(&self, id: EndpointId) {
        let mut state = self.state.lock().await;
        // Always clear any rejection-tracking entry so the map stays bounded.
        state.policy_rejected_peers.remove(&id);
        if let Some(peer) = state.peers.remove(&id) {
            tracing::info!(
                "Peer removed: {} (total: {})",
                id.fmt_short(),
                state.peers.len()
            );
            let count = state.peers.len();
            drop(state);
            let _ = self.peer_change_tx.send(count);
            self.emit_plugin_mesh_event(
                crate::plugin::proto::mesh_event::Kind::PeerDown,
                Some(&peer),
                String::new(),
            )
            .await;
        }
    }

    pub(super) async fn add_peer(
        &self,
        id: EndpointId,
        addr: EndpointAddr,
        ann: &PeerAnnouncement,
    ) {
        // Reject ingest from peers below the supported version floor. They
        // are not added to local state, do not appear in /api/status, and
        // are not re-broadcast. A peer that updates and re-announces will
        // be accepted on the next exchange.
        if !version_allowed_for_rebroadcast(ann.version.as_deref()) {
            tracing::debug!(
                "Refusing direct peer {} below version floor (advertised {:?})",
                id.fmt_short(),
                ann.version
            );
            self.remove_disallowed_peer(id).await;
            return;
        }
        let owner_summary = self.direct_peer_owner_summary(id, ann).await;
        if self.reject_direct_peer_for_policy(id, &owner_summary).await {
            return;
        }
        let mut state = self.state.lock().await;
        state.policy_rejected_peers.remove(&id);
        if id == self.endpoint.id() {
            return;
        }
        let now = std::time::Instant::now();
        // If this peer was previously dead, clear it — add_peer is only called
        // after a successful gossip exchange, which is proof of life.
        let recovered = state.dead_peers.remove(&id).is_some();
        if recovered {
            super::emit_mesh_info(format!(
                "🔄 Peer {} back from the dead (successful gossip)",
                id.fmt_short()
            ));
        }
        let peer_exists = state.peers.contains_key(&id);
        drop(state);
        if peer_exists
            && self
                .upsert_existing_direct_peer(id, addr.clone(), ann, owner_summary.clone(), now)
                .await
        {
            return;
        }
        self.insert_new_direct_peer(id, addr, ann, owner_summary)
            .await;
    }

    /// Update a peer learned transitively through gossip (not directly connected).
    /// Updates assigned/hosted state so models_being_served() includes their models.
    /// Refreshes `last_mentioned` (not `last_seen`) so the peer survives pruning
    /// and gossip propagation as long as a bridge peer keeps mentioning it, but
    /// PeerDown silencing uses only `last_seen` (direct proof-of-life).
    /// Does NOT trigger peer_change events for new transitive peers
    /// (avoids re-election storms at scale).
    pub(super) async fn update_transitive_peer(
        &self,
        id: EndpointId,
        addr: &EndpointAddr,
        ann: &PeerAnnouncement,
        bridge_id: EndpointId,
    ) {
        // Refuse transitive ingest from peers below the supported version
        // floor. Keeps the local table free of pre-floor gossip filler;
        // /api/status, the UI, and routing all stop seeing them.
        if !version_allowed_for_rebroadcast(ann.version.as_deref()) {
            let mut state = self.state.lock().await;
            if state.peers.remove(&id).is_some() {
                let _ = self.peer_change_tx.send(state.peers.len());
            }
            return;
        }
        // Refuse transitive ingest of idle clients — clients that aren't
        // asking for any model, aren't serving anything, and aren't hosting
        // anything. They contribute nothing the mesh can use:
        //   - not routable to (no model to serve)
        //   - not findable (clients-don't-dial-clients by design)
        //   - no demand signal (empty requested_models)
        //   - not relaying for us (no connection — purely transitive)
        // The moment any of those become non-empty, this filter stops firing
        // and the peer is admitted normally. Direct connections (`add_peer`)
        // are never affected — a client that actually contacts us still
        // gets in.
        if peer_is_idle_transitive_client(ann) {
            let mut state = self.state.lock().await;
            if state.peers.remove(&id).is_some() {
                let _ = self.peer_change_tx.send(state.peers.len());
            }
            return;
        }
        let trust_store = self.trust_store.lock().await.clone();
        let owner_summary = verify_node_ownership(
            ann.owner_attestation.as_ref(),
            id.as_bytes(),
            &trust_store,
            self.trust_policy,
            current_time_unix_ms(),
        );
        if !policy_accepts_peer(self.trust_policy, &owner_summary) {
            let mut state = self.state.lock().await;
            if state.peers.remove(&id).is_some() {
                let _ = self.peer_change_tx.send(state.peers.len());
            }
            return;
        }
        let mut state = self.state.lock().await;
        if id == self.endpoint.id() {
            return;
        }
        if state
            .dead_peers
            .get(&id)
            .is_some_and(|t| t.elapsed() < DEAD_PEER_TTL)
        {
            return;
        }
        if let Some(existing) = state.peers.get_mut(&id) {
            let old_peer = existing.clone();
            let serving_changed = apply_transitive_ann(existing, addr, ann, bridge_id);
            existing.owner_summary = owner_summary;
            // Refresh last_mentioned: the bridge peer vouches for this peer
            // being alive (collect_announcements already filters stale peers).
            // We update last_mentioned (not last_seen) so that PeerDown
            // silencing and collect_announcements use only direct proof-of-life,
            // while the prune decision considers both timestamps.
            existing.last_mentioned = std::time::Instant::now();
            let updated_peer = existing.clone();
            let changed = peer_meaningfully_changed(&old_peer, &updated_peer);
            if serving_changed {
                let count = state.peers.len();
                drop(state);
                let _ = self.peer_change_tx.send(count);
                if changed {
                    self.emit_plugin_mesh_event(
                        crate::plugin::proto::mesh_event::Kind::PeerUpdated,
                        Some(&updated_peer),
                        String::new(),
                    )
                    .await;
                }
            } else {
                drop(state);
                if changed {
                    self.emit_plugin_mesh_event(
                        crate::plugin::proto::mesh_event::Kind::PeerUpdated,
                        Some(&updated_peer),
                        String::new(),
                    )
                    .await;
                }
            }
        } else {
            // New transitive peer — not directly verified, so set last_seen to
            // epoch (not "now") to avoid incorrectly silencing PeerDown reports.
            // last_mentioned = now keeps the peer alive for the prune window.
            let mut peer = PeerInfo::from_announcement(id, addr.clone(), ann, owner_summary);
            // Mark as never directly seen — only transitively mentioned.
            peer.last_seen =
                std::time::Instant::now() - std::time::Duration::from_secs(PEER_STALE_SECS * 2);
            state.peers.insert(id, peer.clone());
            drop(state);
            self.emit_plugin_mesh_event(
                crate::plugin::proto::mesh_event::Kind::PeerUp,
                Some(&peer),
                String::new(),
            )
            .await;
        }
    }

    pub(super) async fn collect_announcements(&self) -> Vec<PeerAnnouncement> {
        let stale_cutoff =
            std::time::Instant::now() - std::time::Duration::from_secs(PEER_STALE_SECS);
        let local = self.snapshot_local_announcement_data().await;
        let RebroadcastAnnouncements {
            mut announcements,
            filtered_old_version,
        } = self.collect_rebroadcast_announcements(stale_cutoff).await;
        if filtered_old_version > 0 {
            tracing::debug!(
                filtered = filtered_old_version,
                "gossip: omitting {} peer(s) below v{}.{}.0 from outbound rebroadcast",
                filtered_old_version,
                MIN_REBROADCAST_VERSION_MAJOR,
                MIN_REBROADCAST_VERSION_MINOR,
            );
        }
        announcements.push(self.build_local_announcement(local));
        announcements
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto::OwnershipSummary;
    use iroh::SecretKey;
    use std::collections::HashMap;

    fn test_endpoint_id(seed: u8) -> EndpointId {
        EndpointId::from(SecretKey::from_bytes(&[seed; 32]).public())
    }

    fn test_addr(seed: u8) -> EndpointAddr {
        EndpointAddr {
            id: test_endpoint_id(seed),
            addrs: Default::default(),
        }
    }

    fn test_announcement(ts: Option<u64>) -> PeerAnnouncement {
        PeerAnnouncement {
            addr: test_addr(0x11),
            role: NodeRole::Worker,
            first_joined_mesh_ts: ts,
            models: vec![],
            vram_bytes: 0,
            model_source: None,
            serving_models: vec![],
            hosted_models: None,
            available_models: vec![],
            requested_models: vec![],
            explicit_model_interests: vec![],
            version: None,
            model_demand: HashMap::new(),
            mesh_id: None,
            gpu_name: None,
            hostname: None,
            is_soc: None,
            gpu_vram: None,
            gpu_reserved_bytes: None,
            gpu_mem_bandwidth_gbps: None,
            gpu_compute_tflops_fp32: None,
            gpu_compute_tflops_fp16: None,
            available_model_metadata: vec![],
            experts_summary: None,
            available_model_sizes: HashMap::new(),
            served_model_descriptors: vec![],
            served_model_runtime: vec![],
            owner_attestation: None,
            artifact_transfer_supported: true,
            stage_protocol_generation_supported: true,
            stage_status_list_supported: true,
            advertised_model_throughput: vec![],
            latency_ms: None,
            latency_source: None,
            latency_age_ms: None,
            latency_observer_id: None,
        }
    }

    fn test_peer(ts: Option<u64>) -> PeerInfo {
        PeerInfo::from_announcement(
            test_endpoint_id(0x22),
            test_addr(0x22),
            &test_announcement(ts),
            OwnershipSummary::default(),
        )
    }

    #[test]
    fn test_merge_none_to_some() {
        let mut existing = test_peer(None);
        let ann = test_announcement(Some(100));

        apply_transitive_ann(
            &mut existing,
            &test_addr(0x33),
            &ann,
            test_endpoint_id(0xee),
        );

        assert_eq!(existing.first_joined_mesh_ts, Some(100));
    }

    #[test]
    fn test_merge_some_to_none_keeps_existing() {
        let mut existing = test_peer(Some(100));
        let ann = test_announcement(None);

        apply_transitive_ann(
            &mut existing,
            &test_addr(0x33),
            &ann,
            test_endpoint_id(0xee),
        );

        assert_eq!(existing.first_joined_mesh_ts, Some(100));
    }

    #[test]
    fn test_merge_earlier_incoming_wins() {
        let mut existing = test_peer(Some(200));
        let ann = test_announcement(Some(100));

        apply_transitive_ann(
            &mut existing,
            &test_addr(0x33),
            &ann,
            test_endpoint_id(0xee),
        );

        assert_eq!(existing.first_joined_mesh_ts, Some(100));
    }

    #[test]
    fn test_merge_later_incoming_loses() {
        let mut existing = test_peer(Some(100));
        let ann = test_announcement(Some(200));

        apply_transitive_ann(
            &mut existing,
            &test_addr(0x33),
            &ann,
            test_endpoint_id(0xee),
        );

        assert_eq!(existing.first_joined_mesh_ts, Some(100));
    }

    #[test]
    fn test_merge_equal_values_unchanged() {
        let mut existing = test_peer(Some(100));
        let ann = test_announcement(Some(100));

        apply_transitive_ann(
            &mut existing,
            &test_addr(0x33),
            &ann,
            test_endpoint_id(0xee),
        );

        assert_eq!(existing.first_joined_mesh_ts, Some(100));
    }

    #[test]
    fn test_meaningfully_changed_first_joined_mesh_ts() {
        let old_peer = test_peer(Some(100));
        let new_peer = test_peer(Some(200));

        assert!(peer_meaningfully_changed(&old_peer, &new_peer));
    }

    #[test]
    fn test_meaningfully_changed_explicit_model_interests() {
        let old_peer = test_peer(Some(100));
        let mut new_peer = test_peer(Some(100));
        new_peer.explicit_model_interests = vec!["Qwen/Qwen3-Coder-Next-GGUF@main:Q4_K_M".into()];

        assert!(peer_meaningfully_changed(&old_peer, &new_peer));
    }

    #[test]
    fn test_meaningfully_changed_stage_status_list_support() {
        let old_peer = test_peer(Some(100));
        let mut new_peer = test_peer(Some(100));
        new_peer.stage_status_list_supported = !old_peer.stage_status_list_supported;

        assert!(peer_meaningfully_changed(&old_peer, &new_peer));
    }

    #[test]
    fn test_meaningfully_changed_stage_protocol_generation_support() {
        let old_peer = test_peer(Some(100));
        let mut new_peer = test_peer(Some(100));
        new_peer.stage_protocol_generation_supported =
            !old_peer.stage_protocol_generation_supported;

        assert!(peer_meaningfully_changed(&old_peer, &new_peer));
    }

    #[test]
    fn test_apply_transitive_ann_refreshes_explicit_model_interests() {
        let mut existing = test_peer(Some(100));
        let mut ann = test_announcement(Some(100));
        ann.explicit_model_interests = vec!["Qwen/Qwen3-Coder-Next-GGUF@main:Q4_K_M".into()];

        apply_transitive_ann(
            &mut existing,
            &test_addr(0x33),
            &ann,
            test_endpoint_id(0xee),
        );

        assert_eq!(
            existing.explicit_model_interests,
            vec!["Qwen/Qwen3-Coder-Next-GGUF@main:Q4_K_M".to_string()]
        );
    }

    #[test]
    fn test_apply_transitive_ann_refreshes_stage_status_list_support() {
        let mut existing = test_peer(Some(100));
        existing.stage_status_list_supported = false;
        let mut ann = test_announcement(Some(100));
        ann.stage_status_list_supported = true;

        apply_transitive_ann(
            &mut existing,
            &test_addr(0x33),
            &ann,
            test_endpoint_id(0xee),
        );

        assert!(existing.stage_status_list_supported);
    }

    #[test]
    fn test_apply_transitive_ann_refreshes_stage_protocol_generation_support() {
        let mut existing = test_peer(Some(100));
        existing.stage_protocol_generation_supported = false;
        let mut ann = test_announcement(Some(100));
        ann.stage_protocol_generation_supported = true;

        apply_transitive_ann(
            &mut existing,
            &test_addr(0x33),
            &ann,
            test_endpoint_id(0xee),
        );

        assert!(existing.stage_protocol_generation_supported);
    }

    #[test]
    fn test_apply_transitive_ann_refreshes_advertised_model_throughput() {
        let mut existing = test_peer(Some(100));
        let mut ann = test_announcement(Some(100));
        ann.advertised_model_throughput = vec![crate::network::metrics::ModelThroughputHint {
            model_name: "qwen".to_string(),
            avg_tokens_per_second_milli: 35_000,
            throughput_samples: 4,
        }];

        apply_transitive_ann(
            &mut existing,
            &test_addr(0x33),
            &ann,
            test_endpoint_id(0xee),
        );

        assert_eq!(
            existing.advertised_model_throughput,
            ann.advertised_model_throughput
        );
    }

    #[tokio::test]
    async fn test_add_peer_refreshes_stage_status_list_support() {
        let node = Node::new_for_tests(NodeRole::Worker).await.unwrap();
        let peer_id = test_endpoint_id(0x44);
        let addr = test_addr(0x44);
        let mut ann = test_announcement(Some(100));
        ann.stage_status_list_supported = false;

        node.add_peer(peer_id, addr.clone(), &ann).await;
        ann.stage_status_list_supported = true;
        node.add_peer(peer_id, addr, &ann).await;

        let state = node.state.lock().await;
        let peer = state.peers.get(&peer_id).expect("peer should be tracked");
        assert!(peer.stage_status_list_supported);
    }

    #[tokio::test]
    async fn test_add_peer_refreshes_stage_protocol_generation_support() {
        let node = Node::new_for_tests(NodeRole::Worker).await.unwrap();
        let peer_id = test_endpoint_id(0x45);
        let addr = test_addr(0x45);
        let mut ann = test_announcement(Some(100));
        ann.stage_protocol_generation_supported = false;

        node.add_peer(peer_id, addr.clone(), &ann).await;
        ann.stage_protocol_generation_supported = true;
        node.add_peer(peer_id, addr, &ann).await;

        let state = node.state.lock().await;
        let peer = state.peers.get(&peer_id).expect("peer should be tracked");
        assert!(peer.stage_protocol_generation_supported);
    }

    #[tokio::test]
    async fn test_add_peer_refreshes_advertised_model_throughput() {
        let node = Node::new_for_tests(NodeRole::Worker).await.unwrap();
        let peer_id = test_endpoint_id(0x46);
        let addr = test_addr(0x46);
        let mut ann = test_announcement(Some(100));
        ann.advertised_model_throughput = vec![crate::network::metrics::ModelThroughputHint {
            model_name: "qwen".to_string(),
            avg_tokens_per_second_milli: 20_000,
            throughput_samples: 2,
        }];

        node.add_peer(peer_id, addr.clone(), &ann).await;
        ann.advertised_model_throughput[0].avg_tokens_per_second_milli = 48_000;
        ann.advertised_model_throughput[0].throughput_samples = 9;
        node.add_peer(peer_id, addr, &ann).await;

        let state = node.state.lock().await;
        let peer = state.peers.get(&peer_id).expect("peer should be tracked");
        assert_eq!(
            peer.advertised_model_throughput,
            ann.advertised_model_throughput
        );
    }

    #[tokio::test]
    async fn test_collect_announcements_includes_self_explicit_model_interests() {
        let node = Node::new_for_tests(NodeRole::Worker).await.unwrap();
        node.set_explicit_model_interests(vec![
            "Qwen/Qwen3-Coder-Next-GGUF@main:Q4_K_M".into(),
            "Qwen/Qwen3-Coder-Next-GGUF@main:Q4_K_M".into(),
        ])
        .await;

        let announcements = node.collect_announcements().await;
        let self_announcement = announcements
            .iter()
            .find(|announcement| announcement.addr.id == node.id())
            .expect("self announcement must be present");

        assert_eq!(
            self_announcement.explicit_model_interests,
            vec!["Qwen/Qwen3-Coder-Next-GGUF@main:Q4_K_M".to_string()]
        );
    }

    #[test]
    fn version_allowed_for_rebroadcast_handles_floor() {
        // At or above the floor — allowed.
        assert!(version_allowed_for_rebroadcast(Some("0.60.0")));
        assert!(version_allowed_for_rebroadcast(Some("0.60.2")));
        assert!(version_allowed_for_rebroadcast(Some("0.64.0")));
        assert!(version_allowed_for_rebroadcast(Some("0.65.1")));
        assert!(version_allowed_for_rebroadcast(Some("1.0.0")));
        // Below the floor — refused.
        assert!(!version_allowed_for_rebroadcast(Some("0.57.0")));
        assert!(!version_allowed_for_rebroadcast(Some("0.55.1")));
        assert!(!version_allowed_for_rebroadcast(Some("0.58.0")));
        assert!(!version_allowed_for_rebroadcast(Some("0.59.99")));
    }

    #[test]
    fn version_allowed_for_rebroadcast_handles_metadata_and_prerelease() {
        // Build metadata is stripped.
        assert!(version_allowed_for_rebroadcast(Some(
            "0.65.1+skippy.20260504.kv.2"
        )));
        assert!(!version_allowed_for_rebroadcast(Some("0.57.0+anything")));
        // Pre-release tags are stripped — 0.63.0-rc5 still passes.
        assert!(version_allowed_for_rebroadcast(Some("0.63.0-rc5")));
        assert!(!version_allowed_for_rebroadcast(Some("0.58.0-beta")));
    }

    #[test]
    fn version_allowed_for_rebroadcast_is_conservative_on_unknown() {
        // Unparseable / missing / empty — preserved (don't drop legacy nodes
        // that never advertised a version).
        assert!(version_allowed_for_rebroadcast(None));
        assert!(version_allowed_for_rebroadcast(Some("")));
        assert!(version_allowed_for_rebroadcast(Some("   ")));
        assert!(version_allowed_for_rebroadcast(Some("garbage")));
        assert!(version_allowed_for_rebroadcast(Some("0")));
        assert!(version_allowed_for_rebroadcast(Some("0.x")));
    }

    #[tokio::test]
    async fn transitive_ingest_rejects_below_version_floor() {
        let node = Node::new_for_tests(NodeRole::Worker).await.unwrap();

        let old_addr = test_addr(0x57);
        let new_addr = test_addr(0x65);
        let old_id = old_addr.id;
        let new_id = new_addr.id;

        let mut old_ann = test_announcement(None);
        old_ann.addr = old_addr.clone();
        old_ann.role = NodeRole::Client;
        old_ann.version = Some("0.57.0".to_string());
        let mut new_ann = test_announcement(None);
        new_ann.addr = new_addr.clone();
        new_ann.role = NodeRole::Client;
        new_ann.version = Some("0.65.0".to_string());
        // Give the v0.65.0 client a demand signal so the idle-transitive-
        // client filter (a separate gate) doesn't drop it — this test
        // exercises the version floor specifically.
        new_ann.requested_models = vec!["Qwen3-8B-Q4_K_M".to_string()];

        let bridge = test_endpoint_id(0xBB);
        node.update_transitive_peer(old_id, &old_addr, &old_ann, bridge)
            .await;
        node.update_transitive_peer(new_id, &new_addr, &new_ann, bridge)
            .await;

        // Old peer must NOT be in local state — it was rejected at ingest.
        // New peer must be present.
        {
            let state = node.state.lock().await;
            assert!(
                !state.peers.contains_key(&old_id),
                "v0.57.0 peer must be rejected at ingest, not appear in local state"
            );
            assert!(
                state.peers.contains_key(&new_id),
                "v0.65.0 peer should be added to local state"
            );
        }

        // Outbound gossip must also exclude the old peer.
        let announcements = node.collect_announcements().await;
        assert!(
            !announcements.iter().any(|a| a.addr.id == old_id),
            "v0.57.0 peer must not appear in outbound gossip"
        );
        assert!(
            announcements.iter().any(|a| a.addr.id == new_id),
            "v0.65.0 peer should appear in outbound gossip"
        );
    }

    #[test]
    fn peer_is_idle_transitive_client_basic_shapes() {
        // Empty idle client: no hostname, no direct measurement, no
        // interests → caught.
        let mut ann = test_announcement(None);
        ann.role = NodeRole::Client;
        assert!(peer_is_idle_transitive_client(&ann));

        // Real idle user with a hostname → kept.
        let mut ann = test_announcement(None);
        ann.role = NodeRole::Client;
        ann.hostname = Some("Sams-MacBook-Pro.local".into());
        assert!(!peer_is_idle_transitive_client(&ann));

        // Hostname-less client that someone directly measured → kept.
        let mut ann = test_announcement(None);
        ann.role = NodeRole::Client;
        ann.latency_source = Some(crate::proto::node::LatencySource::Direct);
        assert!(!peer_is_idle_transitive_client(&ann));

        // Estimated latency (propagated guess, not direct) — still caught;
        // only Direct counts as proof of contact.
        let mut ann = test_announcement(None);
        ann.role = NodeRole::Client;
        ann.latency_source = Some(crate::proto::node::LatencySource::Estimated);
        assert!(peer_is_idle_transitive_client(&ann));

        // Client asking for a model → kept (demand signal).
        let mut ann = test_announcement(None);
        ann.role = NodeRole::Client;
        ann.requested_models = vec!["Qwen3-8B-Q4_K_M".to_string()];
        assert!(!peer_is_idle_transitive_client(&ann));

        // Client somehow advertising serving → kept.
        let mut ann = test_announcement(None);
        ann.role = NodeRole::Client;
        ann.serving_models = vec!["Qwen3-8B-Q4_K_M".to_string()];
        assert!(!peer_is_idle_transitive_client(&ann));

        // Client advertising hosted → kept.
        let mut ann = test_announcement(None);
        ann.role = NodeRole::Client;
        ann.hosted_models = Some(vec!["Qwen3-8B-Q4_K_M".to_string()]);
        assert!(!peer_is_idle_transitive_client(&ann));

        // Host → never caught regardless of other fields.
        let mut ann = test_announcement(None);
        ann.role = NodeRole::Host { http_port: 9337 };
        assert!(!peer_is_idle_transitive_client(&ann));

        // Worker → never caught.
        let mut ann = test_announcement(None);
        ann.role = NodeRole::Worker;
        assert!(!peer_is_idle_transitive_client(&ann));
    }

    #[tokio::test]
    async fn transitive_ingest_drops_idle_clients_but_keeps_clients_with_demand() {
        let node = Node::new_for_tests(NodeRole::Worker).await.unwrap();

        let idle_addr = test_addr(0xC1);
        let demand_addr = test_addr(0xC2);
        let host_addr = test_addr(0xC3);
        let idle_id = idle_addr.id;
        let demand_id = demand_addr.id;
        let host_id = host_addr.id;

        // Idle client — should be dropped at transitive ingest.
        let mut idle = test_announcement(None);
        idle.addr = idle_addr.clone();
        idle.role = NodeRole::Client;
        idle.version = Some("0.65.1".to_string());

        // Client asking for a model — must be kept (demand signal).
        let mut with_demand = test_announcement(None);
        with_demand.addr = demand_addr.clone();
        with_demand.role = NodeRole::Client;
        with_demand.version = Some("0.65.1".to_string());
        with_demand.requested_models = vec!["Qwen3-8B-Q4_K_M".to_string()];

        // Host — must be kept (real compute).
        let mut host = test_announcement(None);
        host.addr = host_addr.clone();
        host.role = NodeRole::Host { http_port: 9337 };
        host.version = Some("0.65.1".to_string());
        host.serving_models = vec!["Qwen3-8B-Q4_K_M".to_string()];

        let bridge = test_endpoint_id(0xBB);
        node.update_transitive_peer(idle_id, &idle_addr, &idle, bridge)
            .await;
        node.update_transitive_peer(demand_id, &demand_addr, &with_demand, bridge)
            .await;
        node.update_transitive_peer(host_id, &host_addr, &host, bridge)
            .await;

        let state = node.state.lock().await;
        assert!(
            !state.peers.contains_key(&idle_id),
            "idle transitive client must be rejected"
        );
        assert!(
            state.peers.contains_key(&demand_id),
            "client with requested_models must be kept (demand signal)"
        );
        assert!(
            state.peers.contains_key(&host_id),
            "host must be kept (real compute)"
        );
    }

    #[tokio::test]
    async fn direct_add_peer_admits_idle_clients() {
        // Idle clients we actually directly contact are still admitted.
        // The predicate is for transitive ingest only — a direct connection
        // is proof of life and the peer is observable.
        let node = Node::new_for_tests(NodeRole::Worker).await.unwrap();
        let addr = test_addr(0xC4);
        let id = addr.id;

        let mut ann = test_announcement(None);
        ann.addr = addr.clone();
        ann.role = NodeRole::Client;
        ann.version = Some("0.65.1".to_string());
        // No requested, no serving, no hosted — pure idle client.

        node.add_peer(id, addr, &ann).await;

        let state = node.state.lock().await;
        assert!(
            state.peers.contains_key(&id),
            "direct idle client must be admitted (direct contact is proof of life)"
        );
    }

    #[tokio::test]
    async fn direct_add_peer_rejects_below_version_floor() {
        let node = Node::new_for_tests(NodeRole::Worker).await.unwrap();

        let addr = test_addr(0x57);
        let id = addr.id;

        let mut ann = test_announcement(None);
        ann.addr = addr.clone();
        ann.role = NodeRole::Client;
        ann.version = Some("0.57.0".to_string());

        node.add_peer(id, addr, &ann).await;

        let state = node.state.lock().await;
        assert!(
            !state.peers.contains_key(&id),
            "direct add of v0.57.0 peer must be rejected (no local state entry)"
        );
    }

    /// Regression test for the `--auto` startup wedge: when a transitive
    /// gossip payload includes peers that would be rejected at ingest
    /// (version-floor or idle-transitive-client), `maybe_connect_discovered_peer`
    /// must skip the dial. Otherwise each unreachable ghost address triggers
    /// a 30 s `connect_to_peer` timeout sequentially in the dial loop,
    /// wedging the surrounding gossip exchange (and the `attempt_run_auto_join`
    /// that initiated it) for tens of minutes.
    ///
    /// The function returns without panicking and without dialing within a
    /// generous time bound — a real dial to a fake address would block on
    /// the 30 s `PEER_CONNECT_AND_GOSSIP_TIMEOUT`. We assert the result is
    /// reached well under that bound and that no connection entry was created.
    #[tokio::test]
    async fn maybe_connect_discovered_peer_skips_filtered_announcements() {
        let node = Node::new_for_tests(NodeRole::Worker).await.unwrap();
        let my_role = NodeRole::Worker;

        // Below-floor version — must be skipped without dialing.
        let old_addr = test_addr(0x57);
        let old_id = old_addr.id;
        let mut old_ann = test_announcement(None);
        old_ann.addr = old_addr.clone();
        old_ann.role = NodeRole::Client;
        old_ann.version = Some("0.57.0".to_string());

        // Idle transitive client (matching version, but no hostname / no
        // direct measurement / no model interests) — must also be skipped.
        let idle_addr = test_addr(0xC1);
        let idle_id = idle_addr.id;
        let mut idle_ann = test_announcement(None);
        idle_ann.addr = idle_addr.clone();
        idle_ann.role = NodeRole::Client;
        idle_ann.version = Some("0.65.1".to_string());

        // Both calls together must return well under the 30 s connect
        // timeout. If the dial-loop skip is missing, each call will block
        // on PEER_CONNECT_AND_GOSSIP_TIMEOUT (30 s) attempting to dial the
        // fake test address.
        tokio::time::timeout(std::time::Duration::from_secs(5), async {
            node.maybe_connect_discovered_peer(&my_role, old_addr, &old_ann, true, false)
                .await;
            node.maybe_connect_discovered_peer(&my_role, idle_addr, &idle_ann, true, false)
                .await;
        })
        .await
        .expect("filtered peers must be skipped quickly, not dialed");

        // No connection was attempted (no entry in state.connections), and
        // no peer was added (the filtered announcements never reach add_peer
        // or update_transitive_peer through this path).
        let state = node.state.lock().await;
        assert!(
            !state.connections.contains_key(&old_id),
            "below-floor peer must not be dialed"
        );
        assert!(
            !state.connections.contains_key(&idle_id),
            "idle transitive client must not be dialed"
        );
        assert!(
            !state.peers.contains_key(&old_id),
            "below-floor peer must not be added (this path is dial-only)"
        );
        assert!(
            !state.peers.contains_key(&idle_id),
            "idle transitive client must not be added (this path is dial-only)"
        );
    }
}
