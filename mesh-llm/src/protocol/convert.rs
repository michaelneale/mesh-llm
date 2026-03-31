use crate::mesh::{ModelDemand, NodeRole, PeerAnnouncement, RouteEntry, RoutingTable};
use crate::protocol::NODE_PROTOCOL_GENERATION;
use iroh::{EndpointAddr, EndpointId};
use std::collections::HashMap;

pub(crate) fn local_role_to_proto(role: &NodeRole) -> (i32, Option<u32>) {
    match role {
        NodeRole::Worker => (crate::proto::node::NodeRole::Worker as i32, None),
        NodeRole::Host { http_port } => (
            crate::proto::node::NodeRole::Host as i32,
            Some(*http_port as u32),
        ),
        NodeRole::Client => (crate::proto::node::NodeRole::Client as i32, None),
    }
}

pub(crate) fn proto_role_to_local(role_int: i32, http_port: Option<u32>) -> NodeRole {
    match crate::proto::node::NodeRole::try_from(role_int).unwrap_or_default() {
        crate::proto::node::NodeRole::Host => NodeRole::Host {
            http_port: http_port.unwrap_or(0) as u16,
        },
        crate::proto::node::NodeRole::Client => NodeRole::Client,
        _ => NodeRole::Worker,
    }
}

pub(crate) fn local_ann_to_proto_ann(
    ann: &PeerAnnouncement,
) -> crate::proto::node::PeerAnnouncement {
    let (role_int, http_port) = local_role_to_proto(&ann.role);
    let serialized_addr = serde_json::to_vec(&ann.addr).unwrap_or_default();
    let demand: Vec<crate::proto::node::ModelDemandEntry> = ann
        .model_demand
        .iter()
        .map(
            |(name, d): (&String, &ModelDemand)| crate::proto::node::ModelDemandEntry {
                model_name: name.clone(),
                last_active: d.last_active,
                request_count: d.request_count,
            },
        )
        .collect();
    crate::proto::node::PeerAnnouncement {
        endpoint_id: ann.addr.id.as_bytes().to_vec(),
        role: role_int,
        http_port,
        version: ann.version.clone(),
        gpu_name: ann.gpu_name.clone(),
        hostname: ann.hostname.clone(),
        is_soc: ann.is_soc,
        gpu_vram: ann.gpu_vram.clone(),
        available_models: ann.available_models.clone(),
        serving_models: ann.serving_models.clone(),
        requested_models: ann.requested_models.clone(),
        available_model_metadata: ann.available_model_metadata.clone(),
        experts_summary: ann.experts_summary.clone(),
        rtt_ms: None,
        catalog_models: ann.models.clone(),
        vram_bytes: ann.vram_bytes,
        model_source: ann.model_source.clone(),
        primary_serving: ann.serving.clone(),
        mesh_id: ann.mesh_id.clone(),
        demand,
        available_model_sizes: ann.available_model_sizes.clone(),
        serialized_addr,
    }
}

pub(crate) fn build_gossip_frame(
    anns: &[PeerAnnouncement],
    sender_id: EndpointId,
) -> crate::proto::node::GossipFrame {
    let peers: Vec<crate::proto::node::PeerAnnouncement> =
        anns.iter().map(|ann| local_ann_to_proto_ann(ann)).collect();
    crate::proto::node::GossipFrame {
        gen: NODE_PROTOCOL_GENERATION,
        sender_id: sender_id.as_bytes().to_vec(),
        peers,
    }
}

pub(crate) fn proto_ann_to_local(
    pa: &crate::proto::node::PeerAnnouncement,
) -> Option<(EndpointAddr, PeerAnnouncement)> {
    let id_arr: [u8; 32] = pa.endpoint_id.as_slice().try_into().ok()?;
    let pk = iroh::PublicKey::from_bytes(&id_arr).ok()?;
    let peer_id = EndpointId::from(pk);
    let addr: EndpointAddr = if !pa.serialized_addr.is_empty() {
        serde_json::from_slice(&pa.serialized_addr).unwrap_or(EndpointAddr {
            id: peer_id,
            addrs: Default::default(),
        })
    } else {
        EndpointAddr {
            id: peer_id,
            addrs: Default::default(),
        }
    };
    let role = proto_role_to_local(pa.role, pa.http_port);
    let model_demand: HashMap<String, ModelDemand> = pa
        .demand
        .iter()
        .map(|e| {
            (
                e.model_name.clone(),
                ModelDemand {
                    last_active: e.last_active,
                    request_count: e.request_count,
                },
            )
        })
        .collect();
    let ann = PeerAnnouncement {
        addr: addr.clone(),
        role,
        models: pa.catalog_models.clone(),
        vram_bytes: pa.vram_bytes,
        model_source: pa.model_source.clone(),
        serving: pa.primary_serving.clone(),
        serving_models: pa.serving_models.clone(),
        available_models: pa.available_models.clone(),
        requested_models: pa.requested_models.clone(),
        version: pa.version.clone(),
        model_demand,
        mesh_id: pa.mesh_id.clone(),
        gpu_name: pa.gpu_name.clone(),
        hostname: pa.hostname.clone(),
        is_soc: pa.is_soc,
        gpu_vram: pa.gpu_vram.clone(),
        gpu_bandwidth_gbps: None,
        available_model_metadata: pa.available_model_metadata.clone(),
        experts_summary: pa.experts_summary.clone(),
        available_model_sizes: pa.available_model_sizes.clone(),
    };
    Some((addr, ann))
}

pub(crate) fn routing_table_to_proto(table: &RoutingTable) -> crate::proto::node::RouteTable {
    let entries = table
        .hosts
        .iter()
        .map(|e| crate::proto::node::RouteEntry {
            endpoint_id: e.endpoint_id.as_bytes().to_vec(),
            model: e.model.clone(),
        })
        .collect();
    crate::proto::node::RouteTable {
        entries,
        mesh_id: table.mesh_id.clone(),
        gen: NODE_PROTOCOL_GENERATION,
    }
}

pub(crate) fn proto_route_table_to_local(table: &crate::proto::node::RouteTable) -> RoutingTable {
    let hosts = table
        .entries
        .iter()
        .filter_map(|e| {
            let arr: [u8; 32] = e.endpoint_id.as_slice().try_into().ok()?;
            let pk = iroh::PublicKey::from_bytes(&arr).ok()?;
            let endpoint_id = EndpointId::from(pk);
            Some(RouteEntry {
                model: e.model.clone(),
                node_id: endpoint_id.fmt_short().to_string(),
                endpoint_id,
                vram_gb: 0.0,
            })
        })
        .collect();
    RoutingTable {
        hosts,
        mesh_id: table.mesh_id.clone(),
    }
}
