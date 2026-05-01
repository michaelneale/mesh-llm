use async_trait::async_trait;

use crate::network::transport::TransportIo;

pub struct IrohBiStream {
    send: iroh::endpoint::SendStream,
    recv: iroh::endpoint::RecvStream,
}

impl IrohBiStream {
    pub fn new(send: iroh::endpoint::SendStream, recv: iroh::endpoint::RecvStream) -> Self {
        Self { send, recv }
    }

    pub async fn open(conn: &iroh::endpoint::Connection) -> anyhow::Result<Self> {
        let (send, recv) = conn.open_bi().await?;
        Ok(Self { send, recv })
    }
}

fn to_io_err<E: std::error::Error + Send + Sync + 'static>(e: E) -> std::io::Error {
    std::io::Error::other(e)
}

#[async_trait]
impl TransportIo for IrohBiStream {
    async fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        match self.recv.read(buf).await {
            Ok(Some(n)) => Ok(n),
            Ok(None) => Ok(0),
            Err(e) => Err(to_io_err(e)),
        }
    }

    async fn read_exact(&mut self, buf: &mut [u8]) -> std::io::Result<()> {
        match self.recv.read_exact(buf).await {
            Ok(()) => Ok(()),
            Err(e) => Err(to_io_err(e)),
        }
    }

    async fn write_all(&mut self, buf: &[u8]) -> std::io::Result<()> {
        match self.send.write_all(buf).await {
            Ok(()) => Ok(()),
            Err(e) => Err(to_io_err(e)),
        }
    }

    async fn shutdown(&mut self) -> std::io::Result<()> {
        match self.send.finish() {
            Ok(()) => Ok(()),
            Err(e) => Err(to_io_err(e)),
        }
    }
}
