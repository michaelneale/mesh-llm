use crate::crypto::keys::OwnerKeypair;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum KeyProviderError {
    #[error("key not available: {0}")]
    NotAvailable(String),
}

pub trait KeyProvider: Send + Sync {
    fn owner_keypair(&self) -> Result<OwnerKeypair, KeyProviderError>;
    fn mesh_id(&self) -> Result<Option<Vec<u8>>, KeyProviderError>;
    fn node_id_seed(&self) -> Result<[u8; 32], KeyProviderError>;
}

pub struct InMemoryKeyProvider {
    owner: OwnerKeypair,
    mesh: Option<Vec<u8>>,
    seed: [u8; 32],
}

impl InMemoryKeyProvider {
    pub fn new(owner: OwnerKeypair, mesh: Option<Vec<u8>>, seed: [u8; 32]) -> Self {
        Self { owner, mesh, seed }
    }
}

impl KeyProvider for InMemoryKeyProvider {
    fn owner_keypair(&self) -> Result<OwnerKeypair, KeyProviderError> {
        Ok(self.owner.clone())
    }

    fn mesh_id(&self) -> Result<Option<Vec<u8>>, KeyProviderError> {
        Ok(self.mesh.clone())
    }

    fn node_id_seed(&self) -> Result<[u8; 32], KeyProviderError> {
        Ok(self.seed)
    }
}
