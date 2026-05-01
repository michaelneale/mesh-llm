use async_trait::async_trait;

#[async_trait]
pub trait TransportIo: Send + Sync {
    async fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize>;
    async fn read_exact(&mut self, buf: &mut [u8]) -> std::io::Result<()>;
    async fn write_all(&mut self, buf: &[u8]) -> std::io::Result<()>;
    async fn shutdown(&mut self) -> std::io::Result<()>;
}

pub struct MockTransportIo {
    pub read_buf: std::io::Cursor<Vec<u8>>,
    pub write_buf: Vec<u8>,
}

impl MockTransportIo {
    pub fn new(data: Vec<u8>) -> Self {
        Self {
            read_buf: std::io::Cursor::new(data),
            write_buf: Vec::new(),
        }
    }
}

#[async_trait]
impl TransportIo for MockTransportIo {
    async fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        use std::io::Read;
        self.read_buf.read(buf)
    }

    async fn read_exact(&mut self, buf: &mut [u8]) -> std::io::Result<()> {
        use std::io::Read;
        self.read_buf.read_exact(buf)
    }

    async fn write_all(&mut self, buf: &[u8]) -> std::io::Result<()> {
        self.write_buf.extend_from_slice(buf);
        Ok(())
    }

    async fn shutdown(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
