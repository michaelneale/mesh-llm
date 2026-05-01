pub fn canonical_config_hash(snapshot: &crate::proto::node::NodeConfigSnapshot) -> [u8; 32] {
    use prost::Message as _;
    use sha2::{Digest, Sha256};
    let bytes = snapshot.encode_to_vec();
    let hash = Sha256::digest(&bytes);
    hash.into()
}
