use std::{borrow::Cow, sync::Arc, time::Instant};

use anyhow::{anyhow, Result};

#[derive(Debug, Clone)]
pub struct CacheBytes {
    pub(super) len: u64,
    pub(super) repr: CacheBytesRepr,
}

#[derive(Debug, Clone)]
pub(super) enum CacheBytesRepr {
    Inline(Arc<Vec<u8>>),
    Blocks(Arc<[CacheBlockRef]>),
}

#[derive(Debug, Clone)]
pub(super) struct CacheBlockRef {
    pub(super) bytes: Arc<Vec<u8>>,
}

impl CacheBlockRef {
    pub(super) fn new(bytes: Arc<Vec<u8>>) -> Self {
        Self { bytes }
    }
}

impl CacheBytes {
    pub fn inline(bytes: Vec<u8>) -> Self {
        Self {
            len: bytes.len() as u64,
            repr: CacheBytesRepr::Inline(Arc::new(bytes)),
        }
    }

    pub(super) fn blocks(len: u64, blocks: Vec<CacheBlockRef>) -> Self {
        Self {
            len,
            repr: CacheBytesRepr::Blocks(blocks.into()),
        }
    }

    pub fn len(&self) -> u64 {
        self.len
    }

    pub fn as_cow(&self) -> Result<Cow<'_, [u8]>> {
        match &self.repr {
            CacheBytesRepr::Inline(bytes) => Ok(Cow::Borrowed(bytes.as_slice())),
            CacheBytesRepr::Blocks(blocks) => {
                let capacity = usize::try_from(self.len)
                    .map_err(|_| anyhow!("cache payload too large to reconstruct"))?;
                let mut out = Vec::with_capacity(capacity);
                for block in blocks.iter() {
                    out.extend_from_slice(block.bytes.as_slice());
                }
                if out.len() as u64 != self.len {
                    return Err(anyhow!(
                        "cache payload reconstruction length mismatch: expected {} got {}",
                        self.len,
                        out.len()
                    ));
                }
                Ok(Cow::Owned(out))
            }
        }
    }

    pub fn as_cow_timed(&self) -> Result<(Cow<'_, [u8]>, CacheBytesReconstructStats)> {
        let started = Instant::now();
        let blocks = self.block_ref_count();
        let bytes = self.as_cow()?;
        Ok((
            bytes,
            CacheBytesReconstructStats {
                reconstruct_ms: started.elapsed().as_secs_f64() * 1000.0,
                reconstruct_bytes: self.len,
                reconstruct_blocks: blocks,
            },
        ))
    }

    fn block_ref_count(&self) -> usize {
        match &self.repr {
            CacheBytesRepr::Inline(_) => 0,
            CacheBytesRepr::Blocks(blocks) => blocks.len(),
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct CacheBytesReconstructStats {
    pub reconstruct_ms: f64,
    pub reconstruct_bytes: u64,
    pub reconstruct_blocks: usize,
}
