//! Abstract tunnel relay logic.
//!
//! Provides a `TunnelRelay` trait and a standalone `relay_with_rewrite`
//! function for bidirectional byte relay over any `TransportIo` pair.
//!
//! This module intentionally contains no TCP port binding, no `TcpListener`,
//! and no background-task spawning. Callers own concurrency and lifecycle.

use crate::network::transport::TransportIo;
use anyhow::Result;
use async_trait::async_trait;

/// Trait for types that can relay bytes between two `TransportIo` endpoints.
///
/// Implementations may apply protocol-level rewriting (e.g. REGISTER_PEER
/// port rewriting) or pass bytes through verbatim.
#[async_trait]
pub trait TunnelRelay: Send + Sync {
    /// Relay bytes from `incoming` to `outgoing`, returning the total byte count.
    async fn relay(
        &mut self,
        incoming: &mut dyn TransportIo,
        outgoing: &mut dyn TransportIo,
    ) -> Result<u64>;
}

/// Relay all bytes from `incoming` to `outgoing`, returning the total byte count.
///
/// Reads `incoming` in a loop until EOF (a zero-length read) and writes each
/// chunk to `outgoing` verbatim. No port binding, no spawning, no side-effects
/// beyond the two streams.
pub async fn relay_with_rewrite(
    incoming: &mut dyn TransportIo,
    outgoing: &mut dyn TransportIo,
) -> Result<u64> {
    let mut buf = vec![0u8; 64 * 1024];
    let mut total: u64 = 0;
    loop {
        let n = incoming.read(&mut buf).await?;
        if n == 0 {
            break; // EOF
        }
        outgoing.write_all(&buf[..n]).await?;
        total += n as u64;
    }
    Ok(total)
}
