//! REGISTER_PEER endpoint rewriting.
//!
//! The B2B fork's orchestrator sends RPC_CMD_REGISTER_PEER to tell each
//! worker about its peers. The endpoint string in that message is a
//! `host:port` that was valid on the orchestrator's machine but meaningless
//! on the worker's machine.
//!
//! This module intercepts that command in the QUIC→TCP relay path (inbound
//! tunnel to local rpc-server) and rewrites the endpoint to the local tunnel
//! port on this machine that routes to the correct peer.
//!
//! Wire format:
//!   Client→Server: | cmd (1 byte) | payload_size (8 bytes LE) | payload |
//!
//! REGISTER_PEER payload (132 bytes):
//!   | peer_id (4 bytes LE) | endpoint (128 bytes, null-terminated string) |
//!
//! We parse the port from the endpoint string, look it up in the
//! `remote_port → local_port` map, and rewrite the endpoint field.
//!
//! All other commands pass through as raw bytes.

use anyhow::Result;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::sync::RwLock;

const RPC_CMD_REGISTER_PEER: u8 = 18; // enum position after SET_TENSOR_GGUF (17)
const REGISTER_PEER_PAYLOAD_SIZE: usize = 4 + 128; // peer_id + endpoint
const MAX_RPC_PAYLOAD_BYTES: u64 = 256 * 1024 * 1024;
const RPC_PAYLOAD_GRACE_SECS: u64 = 10;
const RPC_PAYLOAD_RATE_BYTES_PER_SEC: u64 = 4 * 1024 * 1024;
const RPC_PAYLOAD_MAX_SECS: u64 = 120;

/// Shared map from orchestrator's tunnel port → worker's local tunnel port.
/// Built by combining the orchestrator's tunnel map (received via gossip)
/// with the worker's own tunnel map.
pub type PortRewriteMap = Arc<RwLock<HashMap<u16, u16>>>;

/// Create a new empty rewrite map.
pub fn new_rewrite_map() -> PortRewriteMap {
    Arc::new(RwLock::new(HashMap::new()))
}

/// Relay bytes from QUIC recv to TCP write, rewriting REGISTER_PEER commands.
///
/// RPC framing: | cmd (1 byte) | payload_size (8 bytes LE) | payload |
///
/// For REGISTER_PEER, we rewrite the endpoint field.
/// For everything else, we stream bytes through verbatim.
pub async fn relay_with_rewrite(
    quic_recv: iroh::endpoint::RecvStream,
    tcp_write: tokio::io::WriteHalf<tokio::net::TcpStream>,
    port_map: PortRewriteMap,
) -> Result<()> {
    relay_with_rewrite_inner(quic_recv, tcp_write, port_map).await
}

async fn relay_with_rewrite_inner<R, W>(
    mut reader: R,
    mut writer: W,
    port_map: PortRewriteMap,
) -> Result<()>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    loop {
        // Read command byte
        let mut cmd_buf = [0u8; 1];
        if reader.read_exact(&mut cmd_buf).await.is_err() {
            break; // stream closed
        }
        let cmd = cmd_buf[0];

        // Read payload size (8 bytes LE)
        let mut size_buf = [0u8; 8];
        reader.read_exact(&mut size_buf).await?;
        let payload_size = u64::from_le_bytes(size_buf);
        anyhow::ensure!(
            payload_size <= MAX_RPC_PAYLOAD_BYTES,
            "RPC payload too large: {payload_size} bytes exceeds {MAX_RPC_PAYLOAD_BYTES}"
        );
        let payload_deadline = rpc_payload_deadline(payload_size);
        let started = Instant::now();

        if cmd == RPC_CMD_REGISTER_PEER {
            anyhow::ensure!(
                payload_size == REGISTER_PEER_PAYLOAD_SIZE as u64,
                "invalid REGISTER_PEER payload size: expected {REGISTER_PEER_PAYLOAD_SIZE}, got {payload_size}"
            );
            // Read the full payload
            let mut payload = vec![0u8; payload_size as usize];
            read_exact_before_deadline(&mut reader, &mut payload, started, payload_deadline)
                .await?;

            // Extract peer_id (first 4 bytes LE)
            let peer_id = u32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]]);

            // Extract endpoint string (bytes 4..132) — copy to avoid borrow conflict
            let endpoint_bytes = &payload[4..132];
            let endpoint_str = std::str::from_utf8(
                &endpoint_bytes[..endpoint_bytes.iter().position(|&b| b == 0).unwrap_or(128)],
            )
            .unwrap_or("")
            .to_string();

            // Parse port from endpoint string like "127.0.0.1:49502"
            if let Some(port_str) = endpoint_str.rsplit(':').next() {
                if let Ok(remote_port) = port_str.parse::<u16>() {
                    let map = port_map.read().await;
                    if let Some(&local_port) = map.get(&remote_port) {
                        // Rewrite endpoint field
                        let new_endpoint = format!("127.0.0.1:{local_port}");
                        let mut new_endpoint_bytes = [0u8; 128];
                        let copy_len = new_endpoint.len().min(127);
                        new_endpoint_bytes[..copy_len]
                            .copy_from_slice(&new_endpoint.as_bytes()[..copy_len]);
                        payload[4..132].copy_from_slice(&new_endpoint_bytes);

                        tracing::info!(
                            "Rewrote REGISTER_PEER: peer_id={peer_id} \
                             {endpoint_str} → 127.0.0.1:{local_port}"
                        );
                    } else {
                        tracing::warn!(
                            "REGISTER_PEER: no rewrite mapping for port {remote_port} \
                             (peer_id={peer_id}, endpoint={endpoint_str}), passing through"
                        );
                    }
                }
            }

            // Forward (possibly rewritten) command
            writer.write_all(&[cmd]).await?;
            writer.write_all(&size_buf).await?;
            writer.write_all(&payload).await?;
        } else {
            // Not REGISTER_PEER — forward verbatim, streaming the payload
            writer.write_all(&[cmd]).await?;
            writer.write_all(&size_buf).await?;

            // Stream payload in chunks
            let mut remaining = payload_size;
            let mut buf = vec![0u8; 64 * 1024];
            while remaining > 0 {
                let to_read = (remaining as usize).min(buf.len());
                let n = read_before_deadline(
                    &mut reader,
                    &mut buf[..to_read],
                    started,
                    payload_deadline,
                )
                .await?
                .ok_or_else(|| anyhow::anyhow!("stream closed mid-payload"))?;
                writer.write_all(&buf[..n]).await?;
                remaining -= n as u64;
            }
        }

        writer.flush().await?;
    }

    Ok(())
}

fn rpc_payload_deadline(payload_size: u64) -> Duration {
    let throughput_seconds =
        (payload_size + RPC_PAYLOAD_RATE_BYTES_PER_SEC - 1) / RPC_PAYLOAD_RATE_BYTES_PER_SEC;
    Duration::from_secs((RPC_PAYLOAD_GRACE_SECS + throughput_seconds).min(RPC_PAYLOAD_MAX_SECS))
}

async fn read_exact_before_deadline<R>(
    reader: &mut R,
    buf: &mut [u8],
    started: Instant,
    deadline: Duration,
) -> Result<()>
where
    R: AsyncRead + Unpin,
{
    let remaining = deadline_remaining(started, deadline)?;
    tokio::time::timeout(remaining, reader.read_exact(buf))
        .await
        .map_err(|_| anyhow::anyhow!("RPC payload read timed out"))??;
    Ok(())
}

async fn read_before_deadline<R>(
    reader: &mut R,
    buf: &mut [u8],
    started: Instant,
    deadline: Duration,
) -> Result<Option<usize>>
where
    R: AsyncRead + Unpin,
{
    let remaining = deadline_remaining(started, deadline)?;
    let bytes_read = tokio::time::timeout(remaining, reader.read(buf))
        .await
        .map_err(|_| anyhow::anyhow!("RPC payload read timed out"))??;

    if bytes_read == 0 {
        Ok(None)
    } else {
        Ok(Some(bytes_read))
    }
}

fn deadline_remaining(started: Instant, deadline: Duration) -> Result<Duration> {
    deadline
        .checked_sub(started.elapsed())
        .ok_or_else(|| anyhow::anyhow!("RPC payload read timed out"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn register_peer_frame(remote_port: u16) -> Vec<u8> {
        let endpoint = format!("127.0.0.1:{remote_port}");
        let mut payload = vec![0u8; REGISTER_PEER_PAYLOAD_SIZE];
        payload[..4].copy_from_slice(&7u32.to_le_bytes());
        payload[4..4 + endpoint.len()].copy_from_slice(endpoint.as_bytes());

        let mut frame = vec![RPC_CMD_REGISTER_PEER];
        frame.extend_from_slice(&(REGISTER_PEER_PAYLOAD_SIZE as u64).to_le_bytes());
        frame.extend_from_slice(&payload);
        frame
    }

    #[tokio::test]
    async fn relay_with_rewrite_rewrites_register_peer_endpoints() {
        let port_map = new_rewrite_map();
        port_map.write().await.insert(49502, 41234);

        let (mut source_write, source_read) = tokio::io::duplex(4096);
        let (target_write, mut target_read) = tokio::io::duplex(4096);
        let sender = tokio::spawn(async move {
            source_write
                .write_all(&register_peer_frame(49502))
                .await
                .unwrap();
        });

        relay_with_rewrite_inner(source_read, target_write, port_map)
            .await
            .unwrap();
        sender.await.unwrap();

        let mut forwarded = Vec::new();
        target_read.read_to_end(&mut forwarded).await.unwrap();
        assert_eq!(forwarded[0], RPC_CMD_REGISTER_PEER);
        assert_eq!(
            &forwarded[1..9],
            &(REGISTER_PEER_PAYLOAD_SIZE as u64).to_le_bytes()
        );
        let endpoint = std::str::from_utf8(&forwarded[13..141])
            .unwrap()
            .trim_end_matches('\0');
        assert_eq!(endpoint, "127.0.0.1:41234");
    }

    #[tokio::test]
    async fn relay_with_rewrite_rejects_malformed_register_peer_frames() {
        let port_map = new_rewrite_map();
        let (mut source_write, source_read) = tokio::io::duplex(4096);
        let (target_write, mut target_read) = tokio::io::duplex(4096);
        let sender = tokio::spawn(async move {
            let mut frame = vec![RPC_CMD_REGISTER_PEER];
            frame.extend_from_slice(&131u64.to_le_bytes());
            frame.extend_from_slice(&vec![0u8; 131]);
            source_write.write_all(&frame).await.unwrap();
        });

        let err = relay_with_rewrite_inner(source_read, target_write, port_map)
            .await
            .unwrap_err();
        sender.await.unwrap();
        assert!(err
            .to_string()
            .contains("invalid REGISTER_PEER payload size"));

        let mut forwarded = Vec::new();
        target_read.read_to_end(&mut forwarded).await.unwrap();
        assert!(forwarded.is_empty());
    }

    #[tokio::test]
    async fn relay_with_rewrite_rejects_oversized_payloads() {
        let port_map = new_rewrite_map();
        let (mut source_write, source_read) = tokio::io::duplex(128);
        let (target_write, mut target_read) = tokio::io::duplex(128);
        let sender = tokio::spawn(async move {
            let mut frame = vec![0x01];
            frame.extend_from_slice(&(MAX_RPC_PAYLOAD_BYTES + 1).to_le_bytes());
            source_write.write_all(&frame).await.unwrap();
        });

        let err = relay_with_rewrite_inner(source_read, target_write, port_map)
            .await
            .unwrap_err();
        sender.await.unwrap();
        assert!(err.to_string().contains("RPC payload too large"));

        let mut forwarded = Vec::new();
        target_read.read_to_end(&mut forwarded).await.unwrap();
        assert!(forwarded.is_empty());
    }
}
