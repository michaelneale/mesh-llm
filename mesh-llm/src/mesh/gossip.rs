//! Gossip protocol: peer announcement exchange, transitive peer tracking,
//! and peer list management (add/remove/update).

use super::*;

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
        || old.models != new.models
        || old.vram_bytes != new.vram_bytes
        || old.rtt_ms != new.rtt_ms
        || old.model_source != new.model_source
        || old.serving_models != new.serving_models
        || old.hosted_models_known != new.hosted_models_known
        || old.hosted_models != new.hosted_models
        || old.available_models != new.available_models
        || old.requested_models != new.requested_models
        || old.served_model_descriptors != new.served_model_descriptors
        || old.served_model_runtime != new.served_model_runtime
        || old.version != new.version
        || old.owner_summary != new.owner_summary
        || old.gpu_reserved_bytes != new.gpu_reserved_bytes
}

pub(super) fn apply_transitive_ann(
    existing: &mut PeerInfo,
    addr: &EndpointAddr,
    ann: &PeerAnnouncement,
) -> bool {
    let ann_hosted_models = ann.hosted_models.clone().unwrap_or_default();
    let serving_changed = existing.serving_models != ann.serving_models
        || existing.hosted_models != ann_hosted_models
        || existing.hosted_models_known != ann.hosted_models.is_some();
    existing.serving_models = ann.serving_models.clone();
    existing.hosted_models = ann_hosted_models;
    existing.hosted_models_known = ann.hosted_models.is_some();
    existing.role = ann.role.clone();
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
    existing.owner_attestation = ann.owner_attestation.clone();
    if ann.model_source.is_some() {
        existing.model_source = ann.model_source.clone();
    }
    existing.served_model_descriptors = ann.served_model_descriptors.clone();
    existing.served_model_runtime = ann.served_model_runtime.clone();
    if ann.experts_summary.is_some() {
        existing.experts_summary = ann.experts_summary.clone();
    }
    serving_changed
}

impl Node {
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
        let protocol = connection_protocol(&conn);
        let t0 = std::time::Instant::now();
        let (mut send, mut recv) = conn.open_bi().await?;
        send.write_all(&[STREAM_GOSSIP]).await?;

        let our_announcements = self.collect_announcements().await;
        write_gossip_payload(&mut send, protocol, &our_announcements, self.endpoint.id()).await?;
        send.finish()?;

        let rtt_ms = t0.elapsed().as_millis() as u32;
        let buf = read_len_prefixed(&mut recv).await?;
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
        for (addr, ann) in their_announcements {
            let peer_id = addr.id;
            if peer_id == self.endpoint.id() {
                continue;
            }
            if peer_id == remote {
                if let Some(ref their_id) = ann.mesh_id {
                    self.set_mesh_id(their_id.clone()).await;
                }
                self.merge_remote_demand(&ann.model_demand);
                self.add_peer(remote, addr.clone(), ann).await;
                self.update_peer_rtt(remote, rtt_ms).await;
            } else {
                self.update_transitive_peer(peer_id, addr, ann).await;
            }
        }

        // Also check the connection's actual path info — the gossip round-trip
        // time above may reflect relay latency even if a direct path is now active.
        {
            let conn = self.state.lock().await.connections.get(&remote).cloned();
            if let Some(conn) = conn {
                let mut paths = conn.paths();
                let path_list = iroh::Watcher::get(&mut paths);
                for path_info in path_list {
                    if path_info.is_selected() {
                        let path_rtt_ms = match path_info.rtt() {
                            Some(rtt) => rtt.as_millis() as u32,
                            None => continue,
                        };
                        let path_type = if path_info.is_ip() { "direct" } else { "relay" };
                        if path_rtt_ms > 0 && path_rtt_ms < rtt_ms {
                            eprintln!(
                                "📡 Peer {} RTT: {}ms ({}) [path info]",
                                remote.fmt_short(),
                                path_rtt_ms,
                                path_type
                            );
                            self.update_peer_rtt(remote, path_rtt_ms).await;
                        }
                        break;
                    }
                }
            }
        }

        if discover_peers {
            let my_role = self.role.lock().await.clone();
            for (addr, ann) in their_announcements {
                let peer_id = addr.id;
                if peer_id == self.endpoint.id() {
                    continue;
                }
                // Clients skip connecting to other clients
                if matches!(my_role, super::NodeRole::Client)
                    && matches!(ann.role, super::NodeRole::Client)
                {
                    continue;
                }
                let has_conn = self.state.lock().await.connections.contains_key(&peer_id);
                if !has_conn {
                    if let Err(e) = Box::pin(self.connect_to_peer(addr.clone())).await {
                        tracing::debug!(
                            "Could not connect to discovered peer {}: {e}",
                            peer_id.fmt_short()
                        );
                    }
                }
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
            if state.dead_peers.remove(&remote) {
                eprintln!(
                    "🔄 Dead peer {} is gossiping — clearing dead status",
                    remote.fmt_short()
                );
            }
        }

        let buf = read_len_prefixed(&mut recv).await?;
        let their_announcements = decode_gossip_payload(protocol, remote, &buf)?;

        let our_announcements = self.collect_announcements().await;
        write_gossip_payload(&mut send, protocol, &our_announcements, self.endpoint.id()).await?;
        send.finish()?;

        let _ = recv.read_to_end(0).await;

        for (addr, ann) in &their_announcements {
            let peer_id = addr.id;
            if peer_id == self.endpoint.id() {
                continue;
            }
            if peer_id == remote {
                if let Some(ref their_id) = ann.mesh_id {
                    self.set_mesh_id(their_id.clone()).await;
                }
                self.merge_remote_demand(&ann.model_demand);
                self.add_peer(remote, addr.clone(), ann).await;
            } else {
                self.update_transitive_peer(peer_id, addr, ann).await;
            }
        }

        {
            let conn = self.state.lock().await.connections.get(&remote).cloned();
            if let Some(conn) = conn {
                let mut paths = conn.paths();
                let path_list = iroh::Watcher::get(&mut paths);
                for path_info in path_list {
                    if path_info.is_selected() {
                        let rtt_ms = match path_info.rtt() {
                            Some(rtt) => rtt.as_millis() as u32,
                            None => continue,
                        };
                        let path_type = if path_info.is_ip() { "direct" } else { "relay" };
                        if rtt_ms > 0 {
                            eprintln!(
                                "📡 Peer {} RTT: {}ms ({})",
                                remote.fmt_short(),
                                rtt_ms,
                                path_type
                            );
                            self.update_peer_rtt(remote, rtt_ms).await;
                        }
                        break;
                    }
                }
            }
        }

        let my_role = self.role.lock().await.clone();
        for (addr, ann) in their_announcements {
            let peer_id = addr.id;
            if peer_id == self.endpoint.id() {
                continue;
            }
            // Clients should only connect to hosts/workers — not other clients.
            // This avoids O(N²) client-to-client connections in large meshes.
            if matches!(my_role, super::NodeRole::Client)
                && matches!(ann.role, super::NodeRole::Client)
            {
                continue;
            }
            let already_known = self.state.lock().await.peers.contains_key(&peer_id);
            if !already_known {
                if let Err(e) = Box::pin(self.connect_to_peer(addr)).await {
                    tracing::warn!("Failed to discover peer: {e}");
                }
            }
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
        let imported_ranking = import_remote_moe_rankings(&ann.served_model_descriptors);
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
            return;
        }
        let mut state = self.state.lock().await;
        // Peer accepted — clear any prior rejection record so future rejections log again.
        state.policy_rejected_peers.remove(&id);
        if id == self.endpoint.id() {
            return;
        }
        let now = std::time::Instant::now();
        // If this peer was previously dead, clear it — add_peer is only called
        // after a successful gossip exchange, which is proof of life.
        let recovered = state.dead_peers.remove(&id);
        if recovered {
            eprintln!(
                "🔄 Peer {} back from the dead (successful gossip)",
                id.fmt_short()
            );
        }
        if let Some(existing) = state.peers.get_mut(&id) {
            let old_peer = existing.clone();
            let role_changed = existing.role != ann.role;
            let ann_hosted_models = ann.hosted_models.clone().unwrap_or_default();
            let serving_changed = existing.serving_models != ann.serving_models
                || existing.hosted_models != ann_hosted_models
                || existing.hosted_models_known != ann.hosted_models.is_some();
            if role_changed {
                tracing::info!(
                    "Peer {} role updated: {:?} → {:?}",
                    id.fmt_short(),
                    existing.role,
                    ann.role
                );
                existing.role = ann.role.clone();
            }
            // Update addr if the new one has more info
            if !addr.addrs.is_empty() {
                existing.addr = addr;
            }
            existing.models = ann.models.clone();
            existing.vram_bytes = ann.vram_bytes;
            if ann.model_source.is_some() {
                existing.model_source = ann.model_source.clone();
            }
            existing.serving_models = ann.serving_models.clone();
            existing.hosted_models = ann_hosted_models;
            existing.hosted_models_known = ann.hosted_models.is_some();
            existing.available_models.clear();
            existing.requested_models = ann.requested_models.clone();
            existing.last_seen = now;
            if recovered {
                existing.moe_recovered_at = Some(now);
            }
            existing.owner_attestation = ann.owner_attestation.clone();
            existing.owner_summary = owner_summary.clone();
            existing.served_model_descriptors = ann.served_model_descriptors.clone();
            existing.served_model_runtime = ann.served_model_runtime.clone();
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
                || old_peer.gpu_name != updated_peer.gpu_name
                || old_peer.hostname != updated_peer.hostname
                || old_peer.is_soc != updated_peer.is_soc
                || old_peer.gpu_vram != updated_peer.gpu_vram
                || old_peer.gpu_reserved_bytes != updated_peer.gpu_reserved_bytes
                || old_peer.gpu_mem_bandwidth_gbps != updated_peer.gpu_mem_bandwidth_gbps
                || old_peer.gpu_compute_tflops_fp32 != updated_peer.gpu_compute_tflops_fp32
                || old_peer.gpu_compute_tflops_fp16 != updated_peer.gpu_compute_tflops_fp16;
            if role_changed || serving_changed {
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
            if imported_ranking {
                self.refresh_served_model_descriptors().await;
            }
            return;
        }
        tracing::info!(
            "Peer added: {} role={:?} vram={:.1}GB assigned={:?} catalog={:?} (total: {})",
            id.fmt_short(),
            ann.role,
            ann.vram_bytes as f64 / 1e9,
            ann.serving_models.first(),
            ann.available_models,
            state.peers.len() + 1
        );
        let mut peer = PeerInfo::from_announcement(id, addr, ann, owner_summary);
        if recovered {
            peer.moe_recovered_at = Some(now);
        }
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
        if imported_ranking {
            self.refresh_served_model_descriptors().await;
        }
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
    ) {
        let imported_ranking = import_remote_moe_rankings(&ann.served_model_descriptors);
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
        if state.dead_peers.contains(&id) {
            return;
        }
        if let Some(existing) = state.peers.get_mut(&id) {
            let old_peer = existing.clone();
            let serving_changed = apply_transitive_ann(existing, addr, ann);
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
            if imported_ranking {
                self.refresh_served_model_descriptors().await;
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
            if imported_ranking {
                self.refresh_served_model_descriptors().await;
            }
        }
    }

    pub(super) async fn collect_announcements(&self) -> Vec<PeerAnnouncement> {
        // Snapshot all locks independently — never hold multiple locks simultaneously.
        let my_role = self.role.lock().await.clone();
        let my_models = self.models.lock().await.clone();
        let my_source = self.model_source.lock().await.clone();
        let my_serving_models = self.serving_models.lock().await.clone();
        let my_served_model_descriptors = self.served_model_descriptors.lock().await.clone();
        let my_model_runtime_descriptors = self.model_runtime_descriptors.lock().await.clone();
        let my_hosted_models = self.hosted_models.lock().await.clone();
        let my_available = self.available_models.lock().await.clone();
        let my_requested = self.requested_models.lock().await.clone();
        let my_mesh_id = self.mesh_id.lock().await.clone();
        let my_owner_attestation = self.owner_attestation.lock().await.clone();
        let my_demand = self.get_demand();
        let stale_cutoff =
            std::time::Instant::now() - std::time::Duration::from_secs(PEER_STALE_SECS);
        // Gossip wire encoding strips available_model_metadata and available_model_sizes,
        // and remote ingest ignores them. Avoid an expensive scan_local_inventory_snapshot()
        // on the hot gossip path.
        let my_model_metadata: Vec<_> = Vec::new();
        let my_model_sizes: HashMap<_, _> = HashMap::new();
        let mut announcements: Vec<PeerAnnouncement> = {
            let state = self.state.lock().await;
            state
                .peers
                .values()
                .filter(|p| p.last_seen >= stale_cutoff || p.last_mentioned >= stale_cutoff)
                .map(|p| PeerAnnouncement {
                    addr: p.addr.clone(),
                    role: p.role.clone(),
                    models: p.models.clone(),
                    vram_bytes: p.vram_bytes,
                    model_source: p.model_source.clone(),
                    serving_models: p.serving_models.clone(),
                    hosted_models: p.hosted_models_known.then(|| p.hosted_models.clone()),
                    available_models: p.available_models.clone(),
                    requested_models: p.requested_models.clone(),
                    version: p.version.clone(),
                    model_demand: HashMap::new(),
                    mesh_id: None,
                    gpu_name: p.gpu_name.clone(),
                    hostname: p.hostname.clone(),
                    is_soc: p.is_soc,
                    gpu_vram: p.gpu_vram.clone(),
                    gpu_reserved_bytes: p.gpu_reserved_bytes.clone(),
                    gpu_mem_bandwidth_gbps: p.gpu_mem_bandwidth_gbps.clone(),
                    gpu_compute_tflops_fp32: p.gpu_compute_tflops_fp32.clone(),
                    gpu_compute_tflops_fp16: p.gpu_compute_tflops_fp16.clone(),
                    available_model_metadata: p.available_model_metadata.clone(),
                    experts_summary: p.experts_summary.clone(),
                    available_model_sizes: p.available_model_sizes.clone(),
                    served_model_descriptors: p.served_model_descriptors.clone(),
                    served_model_runtime: p.served_model_runtime.clone(),
                    owner_attestation: p.owner_attestation.clone(),
                })
                .collect()
        };
        announcements.push(PeerAnnouncement {
            addr: self.endpoint.addr(),
            role: my_role,
            models: my_models,
            vram_bytes: self.vram_bytes,
            model_source: my_source,
            serving_models: my_serving_models,
            hosted_models: Some(my_hosted_models),
            available_models: my_available,
            requested_models: my_requested,
            version: Some(crate::VERSION.to_string()),
            model_demand: my_demand,
            mesh_id: my_mesh_id,
            gpu_name: if self.enumerate_host {
                self.gpu_name.clone()
            } else {
                None
            },
            hostname: if self.enumerate_host {
                self.hostname.clone()
            } else {
                None
            },
            is_soc: self.is_soc,
            gpu_vram: if self.enumerate_host {
                self.gpu_vram.clone()
            } else {
                None
            },
            gpu_reserved_bytes: if self.enumerate_host {
                self.gpu_reserved_bytes.clone()
            } else {
                None
            },
            gpu_mem_bandwidth_gbps: self.gpu_mem_bandwidth_gbps.lock().await.as_ref().map(|v| {
                v.iter()
                    .map(|f| format!("{:.2}", f))
                    .collect::<Vec<_>>()
                    .join(",")
            }),
            gpu_compute_tflops_fp32: self.gpu_compute_tflops_fp32.lock().await.as_ref().map(|v| {
                v.iter()
                    .map(|f| format!("{:.2}", f))
                    .collect::<Vec<_>>()
                    .join(",")
            }),
            gpu_compute_tflops_fp16: self.gpu_compute_tflops_fp16.lock().await.as_ref().map(|v| {
                v.iter()
                    .map(|f| format!("{:.2}", f))
                    .collect::<Vec<_>>()
                    .join(",")
            }),
            available_model_metadata: my_model_metadata,
            experts_summary: None,
            available_model_sizes: my_model_sizes,
            served_model_descriptors: my_served_model_descriptors,
            served_model_runtime: my_model_runtime_descriptors,
            owner_attestation: my_owner_attestation,
        });
        announcements
    }
}
