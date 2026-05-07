// v0 (legacy) compatibility — JSON-based protocol used before protobuf migration.

use anyhow::Result;
use std::collections::HashMap;

pub const ALPN_V0: &[u8] = b"mesh-llm/0";

pub fn decode_legacy_tunnel_map_frame(buf: &[u8]) -> Result<crate::proto::node::TunnelMap> {
    let serialized: HashMap<String, u16> = serde_json::from_slice(buf)?;
    let entries = serialized
        .into_iter()
        .filter_map(|(hex_id, port)| {
            let bytes = hex::decode(&hex_id).ok()?;
            let arr: [u8; 32] = bytes.try_into().ok()?;
            Some(crate::proto::node::TunnelEntry {
                target_peer_id: arr.to_vec(),
                relay_peer_id: None,
                tunnel_port: port as u32,
            })
        })
        .collect();
    Ok(crate::proto::node::TunnelMap {
        owner_peer_id: Vec::new(),
        entries,
    })
}
