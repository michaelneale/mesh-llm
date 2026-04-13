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

fn descriptors_share_exact_moe_identity(
    local: &[ServedModelDescriptor],
    remote: &[ServedModelDescriptor],
) -> bool {
    local.iter().any(|local_descriptor| {
        local_descriptor
            .topology
            .as_ref()
            .and_then(|topology| topology.moe.as_ref())
            .is_some()
            && remote.iter().any(|remote_descriptor| {
                remote_descriptor
                    .topology
                    .as_ref()
                    .and_then(|topology| topology.moe.as_ref())
                    .is_some()
                    && identities_match_exact(
                        &local_descriptor.identity,
                        &remote_descriptor.identity,
                    )
            })
    })
}

pub(super) fn heartbeat_failure_policy_for_peer(
    local_descriptors: &[ServedModelDescriptor],
    local_runtime: &[ModelRuntimeDescriptor],
    peer: &PeerInfo,
) -> HeartbeatFailurePolicy {
    if descriptors_share_exact_moe_identity(local_descriptors, &peer.served_model_descriptors) {
        if exact_moe_starting_during_convergence(local_descriptors, local_runtime, peer) {
            return HeartbeatFailurePolicy {
                allow_recent_inbound_grace: true,
                // Split convergence can spend minutes in split generation + shard
                // load. During that window, prefer serving continuity over fast
                // heartbeat-only fail-down; request-path failures still remove
                // dead shard peers immediately.
                failure_threshold: 4,
            };
        }
        HeartbeatFailurePolicy {
            allow_recent_inbound_grace: false,
            // Shared MoE peers should fail down faster than generic peers, but a
            // single heartbeat miss is too aggressive on relay-heavy or flaky
            // links. Request-path shard failures still trigger immediate
            // fail-down; heartbeat-only loss needs a second miss to avoid
            // tearing down a healthy split on one transient blip.
            failure_threshold: 2,
        }
    } else {
        HeartbeatFailurePolicy {
            allow_recent_inbound_grace: true,
            failure_threshold: 2,
        }
    }
}

pub(super) const MOE_RECOVERY_PROBATION_SECS: u64 = 30;

fn runtime_model_is_starting(runtimes: &[ModelRuntimeDescriptor], model_name: &str) -> bool {
    runtimes
        .iter()
        .any(|runtime| runtime.model_name == model_name && !runtime.ready)
}

fn exact_moe_starting_during_convergence(
    local_descriptors: &[ServedModelDescriptor],
    local_runtime: &[ModelRuntimeDescriptor],
    peer: &PeerInfo,
) -> bool {
    local_descriptors.iter().any(|local_descriptor| {
        let local_moe = local_descriptor
            .topology
            .as_ref()
            .and_then(|topology| topology.moe.as_ref())
            .is_some();
        if !local_moe {
            return false;
        }

        peer.served_model_descriptors
            .iter()
            .any(|remote_descriptor| {
                let remote_moe = remote_descriptor
                    .topology
                    .as_ref()
                    .and_then(|topology| topology.moe.as_ref())
                    .is_some();
                if !remote_moe
                    || !identities_match_exact(
                        &local_descriptor.identity,
                        &remote_descriptor.identity,
                    )
                {
                    return false;
                }

                let model_name = &local_descriptor.identity.model_name;
                runtime_model_is_starting(local_runtime, model_name)
                    || runtime_model_is_starting(&peer.served_model_runtime, model_name)
            })
    })
}

pub(crate) fn moe_recovery_ready_at(
    recovered_at: Option<std::time::Instant>,
    now: std::time::Instant,
) -> bool {
    recovered_at
        .map(|recovered_at| {
            now.duration_since(recovered_at).as_secs() >= MOE_RECOVERY_PROBATION_SECS
        })
        .unwrap_or(true)
}

pub(crate) fn descriptors_share_exact_moe_identity_for_model(
    local: &[ServedModelDescriptor],
    remote: &[ServedModelDescriptor],
    model_name: &str,
) -> Option<bool> {
    let local_moe: Vec<&ServedModelDescriptor> = local
        .iter()
        .filter(|descriptor| {
            descriptor.identity.model_name == model_name
                && descriptor
                    .topology
                    .as_ref()
                    .and_then(|topology| topology.moe.as_ref())
                    .is_some()
        })
        .collect();
    let remote_moe: Vec<&ServedModelDescriptor> = remote
        .iter()
        .filter(|descriptor| {
            descriptor.identity.model_name == model_name
                && descriptor
                    .topology
                    .as_ref()
                    .and_then(|topology| topology.moe.as_ref())
                    .is_some()
        })
        .collect();
    if local_moe.is_empty() || remote_moe.is_empty() {
        return None;
    }
    Some(local_moe.iter().any(|local_descriptor| {
        remote_moe.iter().any(|remote_descriptor| {
            identities_match_exact(&local_descriptor.identity, &remote_descriptor.identity)
        })
    }))
}

pub(crate) fn peer_is_eligible_for_active_moe(
    local_descriptors: &[ServedModelDescriptor],
    peer: &PeerInfo,
    model_name: &str,
) -> bool {
    let declares_model = peer.is_assigned_model(model_name)
        || peer
            .requested_models
            .iter()
            .any(|requested| requested == model_name);
    if !declares_model || matches!(peer.role, NodeRole::Client) {
        return false;
    }

    let identity_matches = descriptors_share_exact_moe_identity_for_model(
        local_descriptors,
        &peer.served_model_descriptors,
        model_name,
    )
    .unwrap_or(true);
    if !identity_matches {
        return false;
    }

    moe_recovery_ready_at(peer.moe_recovered_at, std::time::Instant::now())
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
                                    eprintln!(
                                        "💚 Heartbeat: reconnected to {}",
                                        peer_id.fmt_short()
                                    );
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
                            eprintln!(
                                "💚 Heartbeat: {} recovered (was {}/2)",
                                peer_id.fmt_short(),
                                fail_counts.get(&peer_id).unwrap_or(&0)
                            );
                            // Clear dead_peers if peer came back
                            node.state.lock().await.dead_peers.remove(&peer_id);
                        }
                        fail_counts.remove(&peer_id);
                    } else {
                        let (recently_seen, failure_policy) = {
                            let state = node.state.lock().await;
                            let peer = state.peers.get(&peer_id).cloned();
                            drop(state);
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
                                eprintln!("💚 Heartbeat: {} outbound failed but seen recently (inbound alive)", peer_id.fmt_short());
                                fail_counts.remove(&peer_id);
                            }
                        } else {
                            let count = fail_counts.entry(peer_id).or_default();
                            *count += 1;
                            if *count >= failure_policy.failure_threshold {
                                // Generic peers require 2 misses so a single timeout doesn't
                                // evict an otherwise-alive inbound-only peer. Shared MoE peers
                                // are stricter: one missed heartbeat should trigger re-election.
                                node.state.lock().await.dead_peers.insert(peer_id);
                                eprintln!(
                                    "💔 Heartbeat: {} unreachable ({} failure{}), removing + broadcasting death",
                                    peer_id.fmt_short(),
                                    count,
                                    if *count == 1 { "" } else { "s" }
                                );
                                fail_counts.remove(&peer_id);
                                node.handle_peer_death(peer_id).await;
                            } else {
                                eprintln!(
                                    "💛 Heartbeat: {} unreachable ({}/{}), will retry",
                                    peer_id.fmt_short(),
                                    count,
                                    failure_policy.failure_threshold
                                );
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
                    eprintln!(
                        "🧹 Pruning stale peer {} (no direct or transitive contact in {}s)",
                        stale_id.fmt_short(),
                        PEER_STALE_SECS * 2
                    );
                    node.remove_peer(stale_id).await;
                    // Also close any lingering connection
                    node.state.lock().await.connections.remove(&stale_id);
                }

                // GC expired demand entries to prevent unbounded map growth
                node.gc_demand().await;
            }
        });
    }

    /// Handle a peer death: remove from state, broadcast to all other peers.
    pub async fn handle_peer_death(&self, dead_id: EndpointId) {
        eprintln!(
            "⚠️  Peer {} died — removing and broadcasting",
            dead_id.fmt_short()
        );
        {
            let mut state = self.state.lock().await;
            // Keep the connection alive — if the peer recovers, their inbound
            // gossip will arrive on the existing connection and trigger recovery
            // via handle_gossip_stream → add_peer → clear dead_peers.
            // Don't remove: state.connections.remove(&dead_id);
            state.dead_peers.insert(dead_id);
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
                    match protocol {
                        ControlProtocol::ProtoV1 => {
                            let proto_msg = crate::proto::node::PeerDown {
                                peer_id: bytes,
                                gen: NODE_PROTOCOL_GENERATION,
                            };
                            write_len_prefixed(&mut send, &proto_msg.encode_to_vec()).await?;
                        }
                        ControlProtocol::JsonV0 => {
                            send.write_all(&bytes).await?;
                        }
                    }
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
                    match protocol {
                        ControlProtocol::ProtoV1 => {
                            let proto_msg = crate::proto::node::PeerLeaving {
                                peer_id: bytes,
                                gen: NODE_PROTOCOL_GENERATION,
                            };
                            write_len_prefixed(&mut send, &proto_msg.encode_to_vec()).await?;
                        }
                        ControlProtocol::JsonV0 => {
                            send.write_all(&bytes).await?;
                        }
                    }
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
}
