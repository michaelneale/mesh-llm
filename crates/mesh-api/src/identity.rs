#[derive(Debug, Clone)]
pub struct OwnerKeypair(mesh_client::OwnerKeypair);

impl OwnerKeypair {
    const ENCODED_HEX_LEN: usize = 64 * 2;

    pub fn generate() -> Self {
        Self(mesh_client::OwnerKeypair::generate())
    }

    pub fn owner_id(&self) -> String {
        self.0.owner_id()
    }

    pub fn from_bytes(signing_bytes: &[u8], encryption_bytes: &[u8]) -> Result<Self, String> {
        mesh_client::OwnerKeypair::from_bytes(signing_bytes, encryption_bytes)
            .map(Self)
            .map_err(|err| err.to_string())
    }

    pub fn from_hex(encoded: &str) -> Result<Self, String> {
        if encoded.len() != Self::ENCODED_HEX_LEN {
            return Err(format!(
                "owner keypair hex must be {} characters",
                Self::ENCODED_HEX_LEN
            ));
        }

        let bytes = hex::decode(encoded).map_err(|err| err.to_string())?;
        let (signing_bytes, encryption_bytes) = bytes.split_at(32);
        Self::from_bytes(signing_bytes, encryption_bytes)
    }

    pub fn to_hex(&self) -> String {
        let mut bytes = Vec::with_capacity(64);
        bytes.extend_from_slice(self.signing_bytes());
        bytes.extend_from_slice(&self.encryption_bytes());
        hex::encode(bytes)
    }

    pub fn signing_bytes(&self) -> &[u8; 32] {
        self.0.signing_bytes()
    }

    pub fn encryption_bytes(&self) -> [u8; 32] {
        self.0.encryption_bytes()
    }

    pub(crate) fn into_inner(self) -> mesh_client::OwnerKeypair {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::OwnerKeypair;

    #[test]
    fn owner_keypair_hex_roundtrip() {
        let keypair = OwnerKeypair::generate();
        let encoded = keypair.to_hex();
        let restored = OwnerKeypair::from_hex(&encoded).expect("hex roundtrip");
        assert_eq!(keypair.owner_id(), restored.owner_id());
        assert_eq!(keypair.to_hex(), restored.to_hex());
    }

    #[test]
    fn owner_keypair_hex_requires_full_key_material() {
        let err = OwnerKeypair::from_hex("deadbeef").expect_err("short hex must fail");
        assert!(err.contains("128"));
    }
}
