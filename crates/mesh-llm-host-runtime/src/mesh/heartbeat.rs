//! Heartbeat loop, peer death detection, and PeerDown handling.
//!
//! The heartbeat runs every 60s, gossips with a random subset of peers,
//! and removes peers that fail to respond after repeated attempts.
//! PeerDown messages are broadcast to the mesh when a peer is confirmed dead.

use super::*;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct HeartbeatFailurePolicy {
    pub(super) allow_recent_inbound_grace: bool,
    pub(super) failure_threshold: u32,
}

pub(super) fn heartbeat_failure_policy_for_peer(
    _local_descriptors: &[ServedModelDescriptor],
    _local_runtime: &[ModelRuntimeDescriptor],
    peer: &PeerInfo,
    is_relay_only: bool,
) -> HeartbeatFailurePolicy {
    let _ = peer;
    HeartbeatFailurePolicy {
        allow_recent_inbound_grace: true,
        // Relay-only peers are more prone to transient timeouts (relay hiccups,
        // higher base RTT). Give them an extra cycle before declaring death.
        failure_threshold: if is_relay_only { 3 } else { 2 },
    }
}

pub(super) const RELAY_HEALTH_CHECK_SECS: u64 = 300;
pub(super) const RELAY_MISSING_GRACE_SECS: u64 = 180;
pub(super) const RELAY_ONLY_RECONNECT_SECS: u64 = 1800;
pub(super) const RELAY_RECONNECT_COOLDOWN_SECS: u64 = 600;
pub(super) const RELAY_DEGRADED_RTT_MS: u32 = 1500;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) enum SelectedPathKind {
    Direct,
    Relay,
    #[default]
    Unknown,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct RelayPathSnapshot {
    pub(super) kind: SelectedPathKind,
    pub(super) rtt_ms: Option<u32>,
}

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct RelayPeerHealth {
    pub(super) relay_since: Option<std::time::Instant>,
    pub(super) last_reconnect_at: Option<std::time::Instant>,
}

impl RelayPeerHealth {
    pub(super) fn observe(&mut self, snapshot: RelayPathSnapshot, now: std::time::Instant) {
        match snapshot.kind {
            SelectedPathKind::Direct => {
                self.relay_since = None;
            }
            SelectedPathKind::Relay => {
                if self.relay_since.is_none() {
                    self.relay_since = Some(now);
                }
            }
            SelectedPathKind::Unknown => {}
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum RelayReconnectReason {
    RelayRttDegraded,
    RelayOnlyTooLong,
}

impl RelayReconnectReason {
    fn label(self) -> &'static str {
        match self {
            RelayReconnectReason::RelayRttDegraded => "relay RTT degraded",
            RelayReconnectReason::RelayOnlyTooLong => "relay path aged out",
        }
    }
}

pub(super) fn selected_path_snapshot(conn: &Connection) -> RelayPathSnapshot {
    let path_list = conn.paths();
    for path_info in &path_list {
        if path_info.is_selected() {
            let rtt = path_info.rtt();
            return RelayPathSnapshot {
                kind: if path_info.is_ip() {
                    SelectedPathKind::Direct
                } else {
                    SelectedPathKind::Relay
                },
                rtt_ms: if rtt.is_zero() {
                    None
                } else {
                    Some(rtt.as_millis() as u32)
                },
            };
        }
    }
    RelayPathSnapshot::default()
}

pub(super) fn relay_reconnect_reason(
    health: &RelayPeerHealth,
    snapshot: RelayPathSnapshot,
    now: std::time::Instant,
    inflight_requests: u64,
    has_home_relay: bool,
) -> Option<RelayReconnectReason> {
    if inflight_requests > 0 || !has_home_relay {
        return None;
    }
    if health.last_reconnect_at.is_some_and(|last| {
        now.duration_since(last) < std::time::Duration::from_secs(RELAY_RECONNECT_COOLDOWN_SECS)
    }) {
        return None;
    }
    if snapshot.kind != SelectedPathKind::Relay {
        return None;
    }
    if snapshot
        .rtt_ms
        .is_some_and(|rtt_ms| rtt_ms >= RELAY_DEGRADED_RTT_MS)
    {
        return Some(RelayReconnectReason::RelayRttDegraded);
    }
    if health.relay_since.is_some_and(|started| {
        now.duration_since(started) >= std::time::Duration::from_secs(RELAY_ONLY_RECONNECT_SECS)
    }) {
        return Some(RelayReconnectReason::RelayOnlyTooLong);
    }
    None
}

pub(super) fn should_remove_connection(
    current_stable_id: Option<usize>,
    closing_stable_id: usize,
) -> bool {
    current_stable_id == Some(closing_stable_id)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PeerDownReportDisposition {
    SuppressReporterCooldown,
    RejectRecentlySeen,
    ProbeReachability,
}

pub(crate) fn peer_down_report_disposition(
    reporter_cooled: bool,
    recently_seen: bool,
) -> PeerDownReportDisposition {
    if reporter_cooled {
        PeerDownReportDisposition::SuppressReporterCooldown
    } else if recently_seen {
        PeerDownReportDisposition::RejectRecentlySeen
    } else {
        PeerDownReportDisposition::ProbeReachability
    }
}

/// Applies the reachability-confirmation rule for a `PeerDown` claim.
/// Returns `Some(dead_id)` if `dead_id != self_id` AND `should_remove` is `true` (peer confirmed gone).
/// Returns `None` if `dead_id == self_id` (never self-evict) or `should_remove` is `false` (peer still reachable).
pub(crate) fn resolve_peer_down(
    self_id: EndpointId,
    dead_id: EndpointId,
    should_remove: bool,
) -> Option<EndpointId> {
    if dead_id == self_id {
        return None;
    }
    if should_remove {
        Some(dead_id)
    } else {
        None
    }
}

impl Node {
    const RTT_REFRESH_SECS: u64 = 15;

    pub fn start_rtt_refresh(&self) {
        let node = self.clone();
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(std::time::Duration::from_secs(Self::RTT_REFRESH_SECS)).await;

                let connections: Vec<(EndpointId, Connection)> = {
                    let state = node.state.lock().await;
                    state
                        .connections
                        .iter()
                        .map(|(id, c)| (*id, c.clone()))
                        .collect()
                };

                for (peer_id, conn) in connections {
                    let path_list = conn.paths();
                    for path_info in &path_list {
                        if path_info.is_selected() {
                            let rtt = path_info.rtt();
                            if !rtt.is_zero() {
                                let rtt_ms = rtt.as_millis() as u32;
                                node.update_peer_rtt(peer_id, rtt_ms).await;
                            }
                            break;
                        }
                    }
                }
            }
        });
    }

    /// Start a background task that watches relay-backed connections and
    /// refreshes one degraded relay path at a time.
    pub fn start_relay_health_monitor(&self) {
        let node = self.clone();
        tokio::spawn(async move {
            let mut addr_watch = node.endpoint.watch_addr();
            let mut peer_health: HashMap<EndpointId, RelayPeerHealth> = HashMap::new();
            let mut relay_missing_since: Option<std::time::Instant> = None;
            let mut relay_missing_warned = false;

            loop {
                tokio::time::sleep(std::time::Duration::from_secs(RELAY_HEALTH_CHECK_SECS)).await;

                let now = std::time::Instant::now();
                let endpoint_addr = iroh::Watcher::get(&mut addr_watch);
                let has_home_relay = endpoint_addr.relay_urls().next().is_some();

                if has_home_relay {
                    if relay_missing_since.take().is_some() {
                        tracing::info!("Relay health: home relay restored");
                    }
                    relay_missing_warned = false;
                } else {
                    let missing_since = *relay_missing_since.get_or_insert(now);
                    if !relay_missing_warned
                        && now.duration_since(missing_since)
                            >= std::time::Duration::from_secs(RELAY_MISSING_GRACE_SECS)
                    {
                        relay_missing_warned = true;
                        tracing::warn!(
                            "Relay health: no home relay for {}s",
                            now.duration_since(missing_since).as_secs()
                        );
                    }
                }

                let inflight_requests = node.inflight_requests();
                let mut connections: Vec<(EndpointId, Connection)> = {
                    let state = node.state.lock().await;
                    state
                        .peers
                        .keys()
                        .filter_map(|id| state.connections.get(id).cloned().map(|conn| (*id, conn)))
                        .collect()
                };

                if connections.is_empty() {
                    peer_health.clear();
                    continue;
                }

                connections.sort_by_key(|(peer_id, _)| endpoint_id_hex(*peer_id));
                peer_health.retain(|peer_id, _| connections.iter().any(|(id, _)| id == peer_id));

                let mut stale_candidate: Option<(EndpointId, RelayReconnectReason)> = None;
                for (peer_id, conn) in connections {
                    let snapshot = selected_path_snapshot(&conn);
                    let health = peer_health.entry(peer_id).or_default();
                    health.observe(snapshot, now);

                    let Some(reason) = relay_reconnect_reason(
                        health,
                        snapshot,
                        now,
                        inflight_requests,
                        has_home_relay,
                    ) else {
                        continue;
                    };

                    if reason == RelayReconnectReason::RelayRttDegraded {
                        stale_candidate = Some((peer_id, reason));
                        break;
                    }
                    if stale_candidate.is_none() {
                        stale_candidate = Some((peer_id, reason));
                    }
                }

                let Some((peer_id, reason)) = stale_candidate else {
                    continue;
                };

                if let Some(health) = peer_health.get_mut(&peer_id) {
                    health.last_reconnect_at = Some(now);
                }

                if node.refresh_peer_connection(peer_id, reason).await {
                    if let Some(health) = peer_health.get_mut(&peer_id) {
                        health.relay_since = Some(now);
                    }
                }
            }
        });
    }

    /// Start a background task that periodically checks peer health.
    /// Probes each peer by attempting a gossip exchange. If the probe fails
    /// (connection dead, peer unresponsive), removes the peer immediately
    /// rather than waiting for QUIC idle timeout.
    /// Start a slow heartbeat (60s) that gossips with a random subset of peers.
    /// At small mesh sizes (≤5 peers), talks to everyone. At larger sizes,
    /// picks K random peers per cycle. Information propagates infectiously —
    /// changes reach all nodes in O(log N) cycles.
    /// Death detection primarily happens on the data path (tunnel fails →
    /// broadcast_peer_down), not via heartbeat.
    pub fn start_heartbeat(&self) {
        let node = self.clone();
        tokio::spawn(async move {
            let mut fail_counts: std::collections::HashMap<EndpointId, u32> =
                std::collections::HashMap::new();

            loop {
                tokio::time::sleep(std::time::Duration::from_secs(60)).await;

                let mut peers_and_conns: Vec<(EndpointId, Option<Connection>)> = {
                    let state = node.state.lock().await;
                    state
                        .peers
                        .keys()
                        .map(|id| {
                            let conn = state.connections.get(id).cloned();
                            (*id, conn)
                        })
                        .collect()
                };
                tracing::debug!("Heartbeat tick: {} peers to check", peers_and_conns.len());

                // Random-K gossip: pick a subset at larger mesh sizes.
                // At ≤5 peers, talk to everyone (backward compat with current behavior).
                // At larger sizes, pick 5 random peers per cycle.
                const GOSSIP_K: usize = 5;
                if peers_and_conns.len() > GOSSIP_K {
                    use rand::seq::SliceRandom;
                    peers_and_conns.shuffle(&mut rand::rng());
                    peers_and_conns.truncate(GOSSIP_K);
                }

                for (peer_id, conn) in peers_and_conns {
                    let hb_start = std::time::Instant::now();
                    let alive = if let Some(conn) = conn {
                        // Gossip as heartbeat — syncs state but won't re-discover dead peers
                        let result = tokio::time::timeout(
                            std::time::Duration::from_secs(10),
                            node.initiate_gossip_inner(conn, peer_id, false),
                        )
                        .await
                        .map(|r| r.is_ok())
                        .unwrap_or(false);
                        tracing::debug!(
                            "Heartbeat gossip {} = {} ({}ms)",
                            peer_id.fmt_short(),
                            if result { "ok" } else { "fail" },
                            hb_start.elapsed().as_millis()
                        );
                        result
                    } else {
                        // No connection — try to reconnect using stored address
                        let addr = {
                            let state = node.state.lock().await;
                            state.peers.get(&peer_id).map(|p| p.addr.clone())
                        };
                        if let Some(addr) = addr {
                            match tokio::time::timeout(
                                std::time::Duration::from_secs(10),
                                connect_mesh(&node.endpoint, addr),
                            )
                            .await
                            {
                                Ok(Ok(new_conn)) => {
                                    super::emit_mesh_info(format!(
                                        "💚 Heartbeat: reconnected to {}",
                                        peer_id.fmt_short()
                                    ));
                                    node.state
                                        .lock()
                                        .await
                                        .connections
                                        .insert(peer_id, new_conn.clone());
                                    // Spawn dispatch_streams for the new connection
                                    let n2 = node.clone();
                                    let nc = new_conn.clone();
                                    tokio::spawn(async move {
                                        n2.dispatch_streams(nc, peer_id).await;
                                    });
                                    // Try gossip on the new connection
                                    tokio::time::timeout(
                                        std::time::Duration::from_secs(10),
                                        node.initiate_gossip_inner(new_conn, peer_id, false),
                                    )
                                    .await
                                    .map(|r| r.is_ok())
                                    .unwrap_or(false)
                                }
                                _ => false,
                            }
                        } else {
                            false
                        }
                    };

                    if alive {
                        if fail_counts.contains_key(&peer_id) {
                            super::emit_mesh_info(format!(
                                "💚 Heartbeat: {} recovered (was {}/2)",
                                peer_id.fmt_short(),
                                fail_counts.get(&peer_id).unwrap_or(&0)
                            ));
                            // Clear dead_peers if peer came back
                            node.state.lock().await.dead_peers.remove(&peer_id);
                        }
                        fail_counts.remove(&peer_id);
                    } else {
                        let (recently_seen, failure_policy) = {
                            let (peer, conn) = {
                                let state = node.state.lock().await;
                                (
                                    state.peers.get(&peer_id).cloned(),
                                    state.connections.get(&peer_id).cloned(),
                                )
                            };
                            let is_relay_only = conn
                                .as_ref()
                                .map(|c| selected_path_snapshot(c).kind == SelectedPathKind::Relay)
                                .unwrap_or(false);
                            let local_descriptors =
                                node.served_model_descriptors.lock().await.clone();
                            let local_runtime = node.model_runtime_descriptors.lock().await.clone();
                            let policy = peer
                                .as_ref()
                                .map(|peer| {
                                    heartbeat_failure_policy_for_peer(
                                        &local_descriptors,
                                        &local_runtime,
                                        peer,
                                        is_relay_only,
                                    )
                                })
                                .unwrap_or(HeartbeatFailurePolicy {
                                    allow_recent_inbound_grace: true,
                                    failure_threshold: 2,
                                });
                            let recently_seen = peer
                                .as_ref()
                                .map(|peer| peer.last_seen.elapsed().as_secs() < PEER_STALE_SECS)
                                .unwrap_or(false);
                            (recently_seen, policy)
                        };
                        // Check if peer has contacted US recently (inbound gossip).
                        // If so, peer is alive — we just can't reach them outbound (NAT).
                        if recently_seen && failure_policy.allow_recent_inbound_grace {
                            // Peer is alive via inbound, don't count as failure
                            if fail_counts.contains_key(&peer_id) {
                                super::emit_mesh_info(format!(
                                    "💚 Heartbeat: {} outbound failed but seen recently (inbound alive)",
                                    peer_id.fmt_short()
                                ));
                                fail_counts.remove(&peer_id);
                            }
                        } else {
                            let count = fail_counts.entry(peer_id).or_default();
                            *count += 1;
                            if *count >= failure_policy.failure_threshold {
                                // Peers require multiple misses so a single timeout doesn't evict
                                // an otherwise-alive inbound-only peer.
                                node.state
                                    .lock()
                                    .await
                                    .dead_peers
                                    .insert(peer_id, std::time::Instant::now());
                                super::emit_mesh_warning(format!(
                                    "💔 Heartbeat: {} unreachable ({} failure{}), removing + broadcasting death",
                                    peer_id.fmt_short(),
                                    count,
                                    if *count == 1 { "" } else { "s" }
                                ));
                                fail_counts.remove(&peer_id);
                                node.handle_peer_death(peer_id).await;
                            } else {
                                super::emit_mesh_warning(format!(
                                    "💛 Heartbeat: {} unreachable ({}/{}), will retry",
                                    peer_id.fmt_short(),
                                    count,
                                    failure_policy.failure_threshold
                                ));
                            }
                        }
                    }
                }

                // Prune stale peers: neither directly verified nor transitively
                // mentioned within 2× the stale window. A peer survives if
                // either last_seen (direct) or last_mentioned (transitive) is
                // fresh, but is pruned when both are stale.
                let prune_cutoff =
                    std::time::Instant::now() - std::time::Duration::from_secs(PEER_STALE_SECS * 2);
                let stale_peers: Vec<EndpointId> = {
                    let state = node.state.lock().await;
                    state
                        .peers
                        .iter()
                        .filter(|(_, p)| {
                            p.last_seen < prune_cutoff && p.last_mentioned < prune_cutoff
                        })
                        .map(|(id, _)| *id)
                        .collect()
                };
                for stale_id in stale_peers {
                    super::emit_mesh_warning(format!(
                        "🧹 Pruning stale peer {} (no direct or transitive contact in {}s)",
                        stale_id.fmt_short(),
                        PEER_STALE_SECS * 2
                    ));
                    node.remove_peer(stale_id).await;
                    // Also close any lingering connection
                    node.state.lock().await.connections.remove(&stale_id);
                }

                // GC expired dead_peers entries so recovered peers can be
                // re-learned transitively through gossip.
                {
                    let mut state = node.state.lock().await;
                    state
                        .dead_peers
                        .retain(|_, ts| ts.elapsed() < DEAD_PEER_TTL);
                    state
                        .peer_down_rejections
                        .retain(|_, ts| ts.elapsed().as_secs() < PEER_DOWN_REPORTER_COOLDOWN_SECS);
                }

                // GC expired demand entries to prevent unbounded map growth
                node.gc_demand().await;
            }
        });
    }

    /// Handle a peer death: remove from state, broadcast to all other peers.
    pub async fn handle_peer_death(&self, dead_id: EndpointId) {
        super::emit_mesh_warning(format!(
            "⚠️  Peer {} died — removing and broadcasting",
            dead_id.fmt_short()
        ));
        {
            let mut state = self.state.lock().await;
            // Keep the connection alive — if the peer recovers, their inbound
            // gossip will arrive on the existing connection and trigger recovery
            // via handle_gossip_stream → add_peer → clear dead_peers.
            // Don't remove: state.connections.remove(&dead_id);
            state.dead_peers.insert(dead_id, std::time::Instant::now());
        }
        self.remove_peer(dead_id).await;
        self.broadcast_peer_down(dead_id).await;
    }

    /// Broadcast that a peer is down to all connected peers.
    async fn broadcast_peer_down(&self, dead_id: EndpointId) {
        let conns: Vec<(EndpointId, Connection)> = {
            let state = self.state.lock().await;
            state
                .connections
                .iter()
                .filter(|(id, _)| **id != dead_id)
                .map(|(id, c)| (*id, c.clone()))
                .collect()
        };
        let dead_bytes = dead_id.as_bytes().to_vec();
        for (peer_id, conn) in conns {
            let bytes = dead_bytes.clone();
            let protocol = connection_protocol(&conn);
            tokio::spawn(async move {
                let res = async {
                    let (mut send, _recv) = conn.open_bi().await?;
                    send.write_all(&[STREAM_PEER_DOWN]).await?;
                    let _ = protocol;
                    let proto_msg = crate::proto::node::PeerDown {
                        peer_id: bytes,
                        gen: NODE_PROTOCOL_GENERATION,
                    };
                    write_len_prefixed(&mut send, &proto_msg.encode_to_vec()).await?;
                    send.finish()?;
                    Ok::<_, anyhow::Error>(())
                }
                .await;
                if let Err(e) = res {
                    tracing::debug!(
                        "Failed to broadcast peer_down to {}: {e}",
                        peer_id.fmt_short()
                    );
                }
            });
        }
    }

    /// Announce clean shutdown to all peers.
    pub async fn broadcast_leaving(&self) {
        let my_id_bytes = self.endpoint.id().as_bytes().to_vec();
        let conns: Vec<(EndpointId, Connection)> = {
            let state = self.state.lock().await;
            state
                .connections
                .iter()
                .map(|(id, c)| (*id, c.clone()))
                .collect()
        };
        for (peer_id, conn) in conns {
            let bytes = my_id_bytes.clone();
            let protocol = connection_protocol(&conn);
            tokio::spawn(async move {
                let res = async {
                    let (mut send, _recv) = conn.open_bi().await?;
                    send.write_all(&[STREAM_PEER_LEAVING]).await?;
                    let _ = protocol;
                    let proto_msg = crate::proto::node::PeerLeaving {
                        peer_id: bytes,
                        gen: NODE_PROTOCOL_GENERATION,
                    };
                    write_len_prefixed(&mut send, &proto_msg.encode_to_vec()).await?;
                    send.finish()?;
                    Ok::<_, anyhow::Error>(())
                }
                .await;
                if let Err(e) = res {
                    tracing::debug!("Failed to send leaving to {}: {e}", peer_id.fmt_short());
                }
            });
        }
        // Give broadcasts a moment to flush
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    }

    async fn refresh_peer_connection(
        &self,
        peer_id: EndpointId,
        reason: RelayReconnectReason,
    ) -> bool {
        let (addr, existing_conn) = {
            let state = self.state.lock().await;
            let Some(peer) = state.peers.get(&peer_id).cloned() else {
                return false;
            };
            let conn = state.connections.get(&peer_id).cloned();
            (peer.addr, conn)
        };

        let Some(existing_conn) = existing_conn else {
            return false;
        };

        let existing_id = existing_conn.stable_id();
        super::emit_mesh_info(format!(
            "🔄 Relay health: refreshing {} ({})",
            peer_id.fmt_short(),
            reason.label()
        ));
        tracing::info!(
            "Relay health: refreshing {} ({})",
            peer_id.fmt_short(),
            reason.label()
        );

        let new_conn = match tokio::time::timeout(
            std::time::Duration::from_secs(10),
            connect_mesh(&self.endpoint, addr.clone()),
        )
        .await
        {
            Ok(Ok(conn)) => conn,
            Ok(Err(err)) => {
                tracing::debug!(
                    "Relay health refresh dial to {} failed: {err}",
                    peer_id.fmt_short()
                );
                return false;
            }
            Err(_) => {
                tracing::debug!(
                    "Relay health refresh dial to {} timed out",
                    peer_id.fmt_short()
                );
                return false;
            }
        };

        let gossip_ok = tokio::time::timeout(
            std::time::Duration::from_secs(10),
            self.initiate_gossip_inner(new_conn.clone(), peer_id, false),
        )
        .await
        .map(|result| result.is_ok())
        .unwrap_or(false);

        if !gossip_ok {
            tracing::debug!(
                "Relay health refresh gossip with {} failed",
                peer_id.fmt_short()
            );
            new_conn.close(0u32.into(), b"relay-health-gossip-failed");
            return false;
        }

        {
            let mut state = self.state.lock().await;
            if !should_remove_connection(
                state.connections.get(&peer_id).map(|conn| conn.stable_id()),
                existing_id,
            ) {
                tracing::debug!(
                    "Relay health refresh for {} raced with another reconnect; keeping newer connection",
                    peer_id.fmt_short()
                );
                drop(state);
                new_conn.close(0u32.into(), b"relay-health-raced");
                return false;
            }
            // Swap the tracked slot before closing the stale connection so its
            // dispatcher sees the newer stable_id and exits without reconnecting.
            state.connections.insert(peer_id, new_conn.clone());
        }

        let node_for_dispatch = self.clone();
        let conn_for_dispatch = new_conn.clone();
        tokio::spawn(async move {
            node_for_dispatch
                .dispatch_streams(conn_for_dispatch, peer_id)
                .await;
        });

        existing_conn.close(0u32.into(), b"relay-health-refresh");
        let _ =
            tokio::time::timeout(std::time::Duration::from_secs(1), existing_conn.closed()).await;

        true
    }
}
