use mesh_llm_protocol::proto::node as proto_node;
use mesh_llm_protocol::protocol::NODE_PROTOCOL_GENERATION;
use semver::{BuildMetadata, Version};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

fn current_time_unix_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

const MESH_GENESIS_POLICY_VERSION: u32 = 1;
const MESH_GENESIS_POLICY_DOMAIN_TAG: &[u8] = b"mesh-llm-genesis-policy-v1:";
const SIGNED_MESH_GENESIS_POLICY_VERSION: u32 = 1;
const SIGNED_MESH_GENESIS_POLICY_DOMAIN_TAG: &[u8] = b"mesh-llm-signed-genesis-policy-v1:";
const SIGNED_BOOTSTRAP_TOKEN_VERSION: u32 = 1;
const SIGNED_BOOTSTRAP_TOKEN_DOMAIN_TAG: &[u8] = b"mesh-llm-bootstrap-token-v1:";
const RELEASE_BUILD_ATTESTATION_VERSION: u32 = 1;
const RELEASE_BUILD_ATTESTATION_DOMAIN_TAG: &[u8] = b"mesh-llm-release-attestation-v1:";
const DIRECT_NODE_ADMISSION_PROOF_VERSION: u32 = 1;
const DIRECT_NODE_ADMISSION_PROOF_DOMAIN_TAG: &[u8] = b"mesh-llm-direct-node-admission-proof-v1:";
const ED25519_SIGNATURE_ALGORITHM: &str = "ed25519";
pub const DIRECT_NODE_ADMISSION_PROOF_MAX_CLOCK_SKEW_MS: u64 = 30_000;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MeshGenesisPolicy {
    pub version: u32,
    pub origin_owner_id: String,
    pub created_at_unix_ms: u64,
    pub requirements: MeshRequirements,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SignedMeshGenesisPolicy {
    pub version: u32,
    pub policy: MeshGenesisPolicy,
    pub origin_sign_public_key: Vec<u8>,
    pub signature_algorithm: String,
    pub signature: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SignedBootstrapToken {
    pub version: u32,
    pub serialized_addrs: Vec<Vec<u8>>,
    pub mesh_id: String,
    pub policy_hash: String,
    pub genesis_policy: MeshGenesisPolicy,
    pub expires_at_unix_ms: Option<u64>,
    pub origin_sign_public_key: Vec<u8>,
    pub signature_algorithm: String,
    pub signature: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReleaseBuildAttestation {
    pub version: u32,
    pub node_version: String,
    pub build_id: String,
    pub commit: String,
    pub target_triple: String,
    pub supported_protocol_generation_min: Option<u32>,
    pub supported_protocol_generation_max: Option<u32>,
    pub artifact_digest: Option<String>,
    pub signer_key_id: String,
    pub signature_algorithm: String,
    pub signature: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DirectNodeAdmissionProof {
    pub version: u32,
    pub sender_id: Vec<u8>,
    pub mesh_id: String,
    pub policy_hash: String,
    pub attestation_hash: String,
    pub timestamp_unix_ms: u64,
    pub signature_algorithm: String,
    pub signature: Vec<u8>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct MeshRequirements {
    #[serde(default)]
    pub node_version: NodeVersionBounds,
    #[serde(default)]
    pub protocol_generation: ProtocolGenerationBounds,
    #[serde(default)]
    pub release_attestation: ReleaseAttestationRequirement,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct NodeVersionBounds {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProtocolGenerationBounds {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max: Option<u32>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReleaseAttestationRequirement {
    #[serde(default)]
    pub required: bool,
    #[serde(default)]
    pub allowed_signer_keys: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MeshRequirementDecision {
    Accepted,
    Rejected(MeshRequirementRejectReason),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MeshRequirementRejectReason {
    OriginOwnerMissing,
    NodeVersionBoundsInvalid,
    NodeVersionMalformed,
    NodeVersionBelowMinimum,
    NodeVersionAboveMaximum,
    ProtocolGenerationBoundsInvalid,
    ProtocolGenerationBelowMinimum,
    ProtocolGenerationAboveMaximum,
    ProtocolGenerationUnknown,
    CertifiedBinaryRequired,
    BuildProofMissing,
    BuildProofInvalid,
    ReleaseSignerUntrusted,
    AttestationPolicyMismatch,
    MeshPolicyMismatch,
    BootstrapTokenInvalid,
    BootstrapTokenExpired,
    DirectProofMissing,
    DirectProofStale,
    DirectProofSenderIdMismatch,
    TopologyDisclosureDenied,
    ReleaseSignerListEmpty,
    ReleaseSignerKeyMalformed,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MeshRequirementPolicySummary {
    pub policy_hash: String,
    pub requirements: MeshRequirements,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MeshRequirementRejectionSource {
    Join,
    Gossip,
    TopologyDisclosure,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MeshRequirementRejectionEvent {
    pub observed_at_unix_ms: u64,
    pub source: MeshRequirementRejectionSource,
    pub reason: MeshRequirementRejectReason,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub peer_id: Option<String>,
}

impl MeshRequirementRejectReason {
    pub const fn code(&self) -> &'static str {
        match self {
            Self::OriginOwnerMissing => "origin_owner_missing",
            Self::NodeVersionBoundsInvalid => "node_version_bounds_invalid",
            Self::NodeVersionMalformed => "node_version_malformed",
            Self::NodeVersionBelowMinimum => "node_version_below_minimum",
            Self::NodeVersionAboveMaximum => "node_version_above_maximum",
            Self::ProtocolGenerationBoundsInvalid => "protocol_generation_bounds_invalid",
            Self::ProtocolGenerationBelowMinimum => "protocol_generation_below_minimum",
            Self::ProtocolGenerationAboveMaximum => "protocol_generation_above_maximum",
            Self::ProtocolGenerationUnknown => "protocol_generation_unknown",
            Self::CertifiedBinaryRequired => "certified_binary_required",
            Self::BuildProofMissing => "build_proof_missing",
            Self::BuildProofInvalid => "build_proof_invalid",
            Self::ReleaseSignerUntrusted => "release_signer_untrusted",
            Self::AttestationPolicyMismatch => "attestation_policy_mismatch",
            Self::MeshPolicyMismatch => "mesh_policy_mismatch",
            Self::BootstrapTokenInvalid => "bootstrap_token_invalid",
            Self::BootstrapTokenExpired => "bootstrap_token_expired",
            Self::DirectProofMissing => "direct_proof_missing",
            Self::DirectProofStale => "direct_proof_stale",
            Self::DirectProofSenderIdMismatch => "direct_proof_sender_id_mismatch",
            Self::TopologyDisclosureDenied => "topology_disclosure_denied",
            Self::ReleaseSignerListEmpty => "release_signer_list_empty",
            Self::ReleaseSignerKeyMalformed => "release_signer_key_malformed",
        }
    }

    pub const fn message(&self) -> &'static str {
        match self {
            Self::OriginOwnerMissing => "the mesh genesis policy is missing its origin owner id.",
            Self::NodeVersionBoundsInvalid => {
                "the mesh node-version requirement range is invalid."
            }
            Self::NodeVersionMalformed => {
                "the peer advertised a malformed mesh-llm node version."
            }
            Self::NodeVersionBelowMinimum => {
                "the peer mesh-llm version is below this mesh's minimum allowed version."
            }
            Self::NodeVersionAboveMaximum => {
                "the peer mesh-llm version is above this mesh's maximum allowed version."
            }
            Self::ProtocolGenerationBoundsInvalid => {
                "the mesh protocol-generation requirement range is invalid."
            }
            Self::ProtocolGenerationBelowMinimum => {
                "the peer protocol generation is below this mesh's minimum allowed generation."
            }
            Self::ProtocolGenerationAboveMaximum => {
                "the peer protocol generation is above this mesh's maximum allowed generation."
            }
            Self::ProtocolGenerationUnknown => {
                "the peer did not advertise a protocol generation required by this mesh."
            }
            Self::CertifiedBinaryRequired => {
                "this mesh requires a certified mesh-llm binary; use a certified compiled binary to join."
            }
            Self::BuildProofMissing => {
                "the peer's certified build proof is missing required signer metadata."
            }
            Self::BuildProofInvalid => {
                "the peer's certified build proof could not be verified."
            }
            Self::ReleaseSignerUntrusted => {
                "the peer's certified build proof was signed by an untrusted release signer."
            }
            Self::AttestationPolicyMismatch => {
                "the certified build or policy attestation does not match this mesh's requirements."
            }
            Self::MeshPolicyMismatch => {
                "the peer or bootstrap token advertised a different mesh policy than this mesh requires."
            }
            Self::BootstrapTokenInvalid => {
                "the bootstrap token is invalid for this mesh."
            }
            Self::BootstrapTokenExpired => {
                "the bootstrap token has expired for this mesh."
            }
            Self::DirectProofMissing => {
                "the peer did not provide the required direct admission proof."
            }
            Self::DirectProofStale => "the peer's direct admission proof is stale.",
            Self::DirectProofSenderIdMismatch => {
                "the peer's direct admission proof does not match the live sender identity."
            }
            Self::TopologyDisclosureDenied => {
                "topology disclosure was denied until the peer completes mesh admission."
            }
            Self::ReleaseSignerListEmpty => {
                "release_attestation.required is true but release_signer_keys is empty; certified-build admission is not remote runtime attestation and refuses to trust self-signed builds."
            }
            Self::ReleaseSignerKeyMalformed => {
                "a release_signer_keys entry is not a valid 'ed25519:<32-byte-hex>' public key."
            }
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MeshRequirementEvaluationInput {
    pub advertised_node_version: Option<String>,
    pub negotiated_protocol_generation: Option<u32>,
    pub policy_hash: Option<String>,
    pub release_attestation: PeerReleaseAttestationStatus,
    pub direct_proof: DirectPeerProofStatus,
    pub bootstrap: BootstrapStatus,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum PeerReleaseAttestationStatus {
    #[default]
    Unsigned,
    Present {
        signer_key: Option<String>,
        attested_version: Option<String>,
    },
    Invalid,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum DirectPeerProofStatus {
    #[default]
    NotChecked,
    Verified,
    Missing,
    Invalid,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum BootstrapStatus {
    #[default]
    NotChecked,
    Valid,
    Invalid,
    Expired,
}

impl MeshGenesisPolicy {
    pub fn new(
        origin_owner_id: impl Into<String>,
        created_at_unix_ms: u64,
        requirements: MeshRequirements,
    ) -> Result<Self, MeshRequirementRejectReason> {
        let policy = Self {
            version: MESH_GENESIS_POLICY_VERSION,
            origin_owner_id: origin_owner_id.into(),
            created_at_unix_ms,
            requirements,
        };
        policy.validate()?;
        Ok(policy)
    }

    pub fn for_local_node(
        origin_owner_id: impl Into<String>,
        created_at_unix_ms: u64,
        requirements: MeshRequirements,
    ) -> Result<(Self, MeshRequirementEvaluationInput), MeshRequirementRejectReason> {
        let policy = Self::new(origin_owner_id, created_at_unix_ms, requirements)?;
        let input = MeshRequirementEvaluationInput {
            advertised_node_version: Some(crate::VERSION.to_string()),
            negotiated_protocol_generation: Some(NODE_PROTOCOL_GENERATION),
            policy_hash: Some(policy.canonical_hash_hex()?),
            release_attestation: PeerReleaseAttestationStatus::Unsigned,
            direct_proof: DirectPeerProofStatus::NotChecked,
            bootstrap: BootstrapStatus::NotChecked,
        };
        Ok((policy, input))
    }

    pub fn validate(&self) -> Result<(), MeshRequirementRejectReason> {
        if self.origin_owner_id.trim().is_empty() {
            return Err(MeshRequirementRejectReason::OriginOwnerMissing);
        }
        if self.version != MESH_GENESIS_POLICY_VERSION {
            return Err(MeshRequirementRejectReason::AttestationPolicyMismatch);
        }
        self.requirements.validate()
    }

    pub fn canonical_bytes(&self) -> Result<Vec<u8>, MeshRequirementRejectReason> {
        self.validate()?;

        let normalized_node_version = self.requirements.node_version.normalized()?;
        let normalized_protocol_generation = self.requirements.protocol_generation.normalized()?;
        let normalized_release_attestation = self.requirements.release_attestation.normalized()?;

        let mut buf = Vec::with_capacity(256);
        buf.extend_from_slice(MESH_GENESIS_POLICY_DOMAIN_TAG);
        buf.extend_from_slice(&self.version.to_le_bytes());
        write_string(&mut buf, self.origin_owner_id.trim());
        buf.extend_from_slice(&self.created_at_unix_ms.to_le_bytes());

        let normalized_node_version_min =
            normalized_node_version.min.as_ref().map(Version::to_string);
        let normalized_node_version_max =
            normalized_node_version.max.as_ref().map(Version::to_string);
        write_optional_string(&mut buf, normalized_node_version_min.as_deref());
        write_optional_string(&mut buf, normalized_node_version_max.as_deref());
        write_optional_u32(&mut buf, normalized_protocol_generation.min);
        write_optional_u32(&mut buf, normalized_protocol_generation.max);
        buf.push(u8::from(normalized_release_attestation.required));
        write_string_list(
            &mut buf,
            &normalized_release_attestation.allowed_signer_keys,
        );
        Ok(buf)
    }

    pub fn canonical_hash(&self) -> Result<[u8; 32], MeshRequirementRejectReason> {
        let digest = Sha256::digest(self.canonical_bytes()?);
        let mut hash = [0u8; 32];
        hash.copy_from_slice(&digest);
        Ok(hash)
    }

    pub fn canonical_hash_hex(&self) -> Result<String, MeshRequirementRejectReason> {
        Ok(hex::encode(self.canonical_hash()?))
    }

    pub fn policy_derived_mesh_id(&self) -> Result<String, MeshRequirementRejectReason> {
        self.canonical_hash_hex()
    }

    pub fn to_proto(&self) -> proto_node::MeshGenesisPolicy {
        proto_node::MeshGenesisPolicy {
            version: self.version,
            origin_owner_id: self.origin_owner_id.clone(),
            created_at_unix_ms: self.created_at_unix_ms,
            requirements: Some(self.requirements.to_proto()),
        }
    }

    pub fn from_proto(
        policy: &proto_node::MeshGenesisPolicy,
    ) -> Result<Self, MeshRequirementRejectReason> {
        let requirements = policy
            .requirements
            .as_ref()
            .map(MeshRequirements::from_proto)
            .transpose()?
            .unwrap_or_default();
        Self::new(
            policy.origin_owner_id.clone(),
            policy.created_at_unix_ms,
            requirements,
        )
    }

    pub fn evaluate(&self, input: &MeshRequirementEvaluationInput) -> MeshRequirementDecision {
        if let Err(reason) = self.validate() {
            return MeshRequirementDecision::Rejected(reason);
        }

        match input.bootstrap {
            BootstrapStatus::Invalid => {
                return MeshRequirementDecision::Rejected(
                    MeshRequirementRejectReason::BootstrapTokenInvalid,
                );
            }
            BootstrapStatus::Expired => {
                return MeshRequirementDecision::Rejected(
                    MeshRequirementRejectReason::BootstrapTokenExpired,
                );
            }
            BootstrapStatus::NotChecked | BootstrapStatus::Valid => {}
        }

        if let Some(policy_hash) = input.policy_hash.as_deref() {
            let expected_hash = match self.canonical_hash_hex() {
                Ok(hash) => hash,
                Err(reason) => return MeshRequirementDecision::Rejected(reason),
            };
            if policy_hash.trim() != expected_hash {
                return MeshRequirementDecision::Rejected(
                    MeshRequirementRejectReason::MeshPolicyMismatch,
                );
            }
        }

        self.requirements.evaluate(input)
    }
}

impl MeshRequirements {
    pub fn unrestricted() -> Self {
        Self::default()
    }

    pub fn is_unrestricted(&self) -> bool {
        self == &Self::default()
    }

    pub fn validate(&self) -> Result<(), MeshRequirementRejectReason> {
        self.node_version.normalized()?;
        self.protocol_generation.normalized()?;
        self.release_attestation.normalized()?;
        Ok(())
    }

    pub fn evaluate(&self, input: &MeshRequirementEvaluationInput) -> MeshRequirementDecision {
        if let Err(reason) = self.validate() {
            return MeshRequirementDecision::Rejected(reason);
        }

        let normalized_node_version = match self.node_version.normalized() {
            Ok(bounds) => bounds,
            Err(reason) => return MeshRequirementDecision::Rejected(reason),
        };
        if normalized_node_version.is_constrained() {
            let advertised = match input.advertised_node_version.as_deref() {
                Some(value) => value,
                None => {
                    return MeshRequirementDecision::Rejected(
                        MeshRequirementRejectReason::NodeVersionMalformed,
                    );
                }
            };
            let version = match parse_node_version(advertised) {
                Ok(version) => version,
                Err(reason) => return MeshRequirementDecision::Rejected(reason),
            };
            if let Some(min) = normalized_node_version.min.as_ref() {
                if version_precedence_cmp(&version, min).is_lt() {
                    return MeshRequirementDecision::Rejected(
                        MeshRequirementRejectReason::NodeVersionBelowMinimum,
                    );
                }
            }
            if let Some(max) = normalized_node_version.max.as_ref() {
                if version_precedence_cmp(&version, max).is_gt() {
                    return MeshRequirementDecision::Rejected(
                        MeshRequirementRejectReason::NodeVersionAboveMaximum,
                    );
                }
            }
        }

        let normalized_protocol_generation = match self.protocol_generation.normalized() {
            Ok(bounds) => bounds,
            Err(reason) => return MeshRequirementDecision::Rejected(reason),
        };
        if normalized_protocol_generation.is_constrained() {
            let protocol_generation = match input.negotiated_protocol_generation {
                Some(value) => value,
                None => {
                    return MeshRequirementDecision::Rejected(
                        MeshRequirementRejectReason::ProtocolGenerationUnknown,
                    );
                }
            };
            if let Some(min) = normalized_protocol_generation.min {
                if protocol_generation < min {
                    return MeshRequirementDecision::Rejected(
                        MeshRequirementRejectReason::ProtocolGenerationBelowMinimum,
                    );
                }
            }
            if let Some(max) = normalized_protocol_generation.max {
                if protocol_generation > max {
                    return MeshRequirementDecision::Rejected(
                        MeshRequirementRejectReason::ProtocolGenerationAboveMaximum,
                    );
                }
            }
        }

        let normalized_release_attestation = match self.release_attestation.normalized() {
            Ok(requirement) => requirement,
            Err(reason) => return MeshRequirementDecision::Rejected(reason),
        };
        if normalized_release_attestation.required {
            match &input.release_attestation {
                PeerReleaseAttestationStatus::Unsigned => {
                    return MeshRequirementDecision::Rejected(
                        MeshRequirementRejectReason::CertifiedBinaryRequired,
                    );
                }
                PeerReleaseAttestationStatus::Invalid => {
                    return MeshRequirementDecision::Rejected(
                        MeshRequirementRejectReason::BuildProofInvalid,
                    );
                }
                PeerReleaseAttestationStatus::Present {
                    signer_key,
                    attested_version,
                } => {
                    if let Some(attested) = attested_version.as_deref() {
                        if attested != crate::VERSION {
                            return MeshRequirementDecision::Rejected(
                                MeshRequirementRejectReason::BuildProofInvalid,
                            );
                        }
                    }
                    if !normalized_release_attestation
                        .allowed_signer_keys
                        .is_empty()
                    {
                        let Some(signer_key) = signer_key.as_deref() else {
                            return MeshRequirementDecision::Rejected(
                                MeshRequirementRejectReason::BuildProofMissing,
                            );
                        };
                        if !normalized_release_attestation
                            .allowed_signer_keys
                            .iter()
                            .any(|allowed| allowed == signer_key.trim())
                        {
                            return MeshRequirementDecision::Rejected(
                                MeshRequirementRejectReason::ReleaseSignerUntrusted,
                            );
                        }
                    }
                }
            }
        }

        MeshRequirementDecision::Accepted
    }

    pub fn to_proto(&self) -> proto_node::MeshRequirements {
        proto_node::MeshRequirements {
            node_version: Some(self.node_version.to_proto()),
            protocol_generation: Some(self.protocol_generation.to_proto()),
            release_attestation: Some(self.release_attestation.to_proto()),
        }
    }

    pub fn from_proto(
        value: &proto_node::MeshRequirements,
    ) -> Result<Self, MeshRequirementRejectReason> {
        let requirements = Self {
            node_version: value
                .node_version
                .as_ref()
                .map(NodeVersionBounds::from_proto)
                .unwrap_or_default(),
            protocol_generation: value
                .protocol_generation
                .as_ref()
                .map(ProtocolGenerationBounds::from_proto)
                .unwrap_or_default(),
            release_attestation: value
                .release_attestation
                .as_ref()
                .map(ReleaseAttestationRequirement::from_proto)
                .unwrap_or_default(),
        };
        requirements.validate()?;
        Ok(requirements)
    }
}

pub fn peer_release_attestation_status(
    attestation: Option<&ReleaseBuildAttestation>,
) -> PeerReleaseAttestationStatus {
    match attestation {
        None => PeerReleaseAttestationStatus::Unsigned,
        Some(attestation) => match attestation.verify() {
            Ok(()) => PeerReleaseAttestationStatus::Present {
                signer_key: Some(attestation.signer_key_id.trim().to_string()),
                attested_version: Some(attestation.node_version.clone()),
            },
            Err(_) => PeerReleaseAttestationStatus::Invalid,
        },
    }
}

pub fn evaluate_direct_peer_admission(
    policy: Option<&MeshGenesisPolicy>,
    input: &MeshRequirementEvaluationInput,
) -> MeshRequirementDecision {
    let Some(policy) = policy else {
        return MeshRequirementDecision::Accepted;
    };

    match input.direct_proof {
        DirectPeerProofStatus::Verified => policy.evaluate(input),
        DirectPeerProofStatus::Invalid => {
            MeshRequirementDecision::Rejected(MeshRequirementRejectReason::BuildProofInvalid)
        }
        DirectPeerProofStatus::Missing | DirectPeerProofStatus::NotChecked => {
            MeshRequirementDecision::Rejected(MeshRequirementRejectReason::DirectProofMissing)
        }
    }
}

impl NodeVersionBounds {
    pub fn normalized(&self) -> Result<NormalizedNodeVersionBounds, MeshRequirementRejectReason> {
        let min = self.min.as_deref().map(parse_node_version).transpose()?;
        let max = self.max.as_deref().map(parse_node_version).transpose()?;
        if let (Some(min), Some(max)) = (&min, &max) {
            if version_precedence_cmp(min, max).is_gt() {
                return Err(MeshRequirementRejectReason::NodeVersionBoundsInvalid);
            }
        }
        Ok(NormalizedNodeVersionBounds { min, max })
    }

    pub fn to_proto(&self) -> proto_node::NodeVersionBounds {
        proto_node::NodeVersionBounds {
            min: self.min.clone(),
            max: self.max.clone(),
        }
    }

    pub fn from_proto(value: &proto_node::NodeVersionBounds) -> Self {
        Self {
            min: value.min.clone(),
            max: value.max.clone(),
        }
    }
}

impl ProtocolGenerationBounds {
    pub fn normalized(
        &self,
    ) -> Result<NormalizedProtocolGenerationBounds, MeshRequirementRejectReason> {
        if let (Some(min), Some(max)) = (self.min, self.max) {
            if min > max {
                return Err(MeshRequirementRejectReason::ProtocolGenerationBoundsInvalid);
            }
        }
        Ok(NormalizedProtocolGenerationBounds {
            min: self.min,
            max: self.max,
        })
    }

    pub fn to_proto(&self) -> proto_node::ProtocolGenerationBounds {
        proto_node::ProtocolGenerationBounds {
            min: self.min,
            max: self.max,
        }
    }

    pub fn from_proto(value: &proto_node::ProtocolGenerationBounds) -> Self {
        Self {
            min: value.min,
            max: value.max,
        }
    }
}

impl ReleaseAttestationRequirement {
    pub fn normalized(
        &self,
    ) -> Result<NormalizedReleaseAttestationRequirement, MeshRequirementRejectReason> {
        let mut allowed_signer_keys = Vec::with_capacity(self.allowed_signer_keys.len());
        for signer_key in &self.allowed_signer_keys {
            let normalized = signer_key.trim();
            if normalized.is_empty() {
                return Err(MeshRequirementRejectReason::ReleaseSignerUntrusted);
            }
            allowed_signer_keys.push(normalized.to_string());
        }
        allowed_signer_keys.sort();
        allowed_signer_keys.dedup();
        // Refuse `required = true` without any trusted release signer.
        // Without an allowlist `evaluate()` would accept any self-consistent
        // attestation, defeating the point of `require_release_attestation`.
        // Certified-build admission is not remote runtime attestation; trust
        // must be anchored in a release signer the operator picked.
        if self.required && allowed_signer_keys.is_empty() {
            return Err(MeshRequirementRejectReason::ReleaseSignerListEmpty);
        }
        Ok(NormalizedReleaseAttestationRequirement {
            required: self.required,
            allowed_signer_keys,
        })
    }

    pub fn validate_signer_key_shapes(&self) -> Result<(), MeshRequirementRejectReason> {
        // Strict ed25519:<32-byte-hex> shape check. Run at config/CLI
        // policy-creation time so impossible policies cannot be persisted into
        // an immutable mesh id; do NOT run from `normalized()`/`evaluate()` so
        // peer announcements with already-baked-in allowlists keep evaluating
        // to a deterministic `release_signer_untrusted` rejection rather than
        // collapsing to a malformed-policy error.
        for signer_key in &self.allowed_signer_keys {
            let normalized = signer_key.trim();
            if normalized.is_empty() {
                return Err(MeshRequirementRejectReason::ReleaseSignerUntrusted);
            }
            parse_release_signer_public_key(normalized)
                .map_err(|_| MeshRequirementRejectReason::ReleaseSignerKeyMalformed)?;
        }
        Ok(())
    }

    pub fn to_proto(&self) -> proto_node::ReleaseAttestationRequirement {
        proto_node::ReleaseAttestationRequirement {
            required: Some(self.required),
            allowed_signer_keys: self.allowed_signer_keys.clone(),
        }
    }

    pub fn from_proto(value: &proto_node::ReleaseAttestationRequirement) -> Self {
        Self {
            required: value.required.unwrap_or(false),
            allowed_signer_keys: value.allowed_signer_keys.clone(),
        }
    }
}

impl SignedMeshGenesisPolicy {
    pub fn sign(
        policy: MeshGenesisPolicy,
        owner: &crate::crypto::OwnerKeypair,
    ) -> Result<Self, MeshRequirementRejectReason> {
        let mut signed = Self {
            version: SIGNED_MESH_GENESIS_POLICY_VERSION,
            policy,
            origin_sign_public_key: owner.verifying_key().as_bytes().to_vec(),
            signature_algorithm: ED25519_SIGNATURE_ALGORITHM.to_string(),
            signature: Vec::new(),
        };
        signed.signature = owner.sign_bytes(&signed.canonical_bytes()?).to_vec();
        signed.verify()?;
        Ok(signed)
    }

    pub fn to_proto(&self) -> proto_node::SignedMeshGenesisPolicy {
        proto_node::SignedMeshGenesisPolicy {
            version: self.version,
            policy: Some(self.policy.to_proto()),
            origin_sign_public_key: self.origin_sign_public_key.clone(),
            signature_algorithm: self.signature_algorithm.clone(),
            signature: self.signature.clone(),
        }
    }

    pub fn from_proto(
        value: &proto_node::SignedMeshGenesisPolicy,
    ) -> Result<Self, MeshRequirementRejectReason> {
        let policy = value
            .policy
            .as_ref()
            .ok_or(MeshRequirementRejectReason::AttestationPolicyMismatch)
            .and_then(MeshGenesisPolicy::from_proto)?;
        Ok(Self {
            version: value.version,
            policy,
            origin_sign_public_key: value.origin_sign_public_key.clone(),
            signature_algorithm: value.signature_algorithm.clone(),
            signature: value.signature.clone(),
        })
    }

    pub fn canonical_bytes(&self) -> Result<Vec<u8>, MeshRequirementRejectReason> {
        self.policy.validate()?;
        let mut buf = Vec::with_capacity(256);
        buf.extend_from_slice(SIGNED_MESH_GENESIS_POLICY_DOMAIN_TAG);
        buf.extend_from_slice(&self.version.to_le_bytes());
        buf.extend_from_slice(&self.policy.canonical_bytes()?);
        write_bytes(&mut buf, &self.origin_sign_public_key);
        write_string(&mut buf, self.signature_algorithm.trim());
        Ok(buf)
    }

    pub fn verify(&self) -> Result<(), MeshRequirementRejectReason> {
        if self.version != SIGNED_MESH_GENESIS_POLICY_VERSION {
            return Err(MeshRequirementRejectReason::AttestationPolicyMismatch);
        }
        if self.origin_sign_public_key.len() != 32 || self.signature.len() != 64 {
            return Err(MeshRequirementRejectReason::BuildProofInvalid);
        }
        if self.signature_algorithm.trim() != ED25519_SIGNATURE_ALGORITHM {
            return Err(MeshRequirementRejectReason::BuildProofInvalid);
        }
        let verifying_key = ed25519_dalek::VerifyingKey::from_bytes(
            &self
                .origin_sign_public_key
                .as_slice()
                .try_into()
                .map_err(|_| MeshRequirementRejectReason::BuildProofInvalid)?,
        )
        .map_err(|_| MeshRequirementRejectReason::BuildProofInvalid)?;
        if mesh_llm_identity::keys::owner_id_from_verifying_key(&verifying_key)
            != self.policy.origin_owner_id
        {
            return Err(MeshRequirementRejectReason::BuildProofInvalid);
        }
        let signature = ed25519_dalek::Signature::from_bytes(
            &self
                .signature
                .as_slice()
                .try_into()
                .map_err(|_| MeshRequirementRejectReason::BuildProofInvalid)?,
        );
        verifying_key
            .verify_strict(&self.canonical_bytes()?, &signature)
            .map_err(|_| MeshRequirementRejectReason::BuildProofInvalid)
    }
}

impl SignedBootstrapToken {
    pub fn sign(
        serialized_addrs: Vec<Vec<u8>>,
        signed_genesis_policy: &SignedMeshGenesisPolicy,
        expires_at_unix_ms: Option<u64>,
        owner: &crate::crypto::OwnerKeypair,
    ) -> Result<Self, MeshRequirementRejectReason> {
        signed_genesis_policy.verify()?;
        if owner.verifying_key().as_bytes()
            != signed_genesis_policy.origin_sign_public_key.as_slice()
        {
            return Err(MeshRequirementRejectReason::BootstrapTokenInvalid);
        }
        let mut token = Self {
            version: SIGNED_BOOTSTRAP_TOKEN_VERSION,
            serialized_addrs,
            mesh_id: signed_genesis_policy.policy.policy_derived_mesh_id()?,
            policy_hash: signed_genesis_policy.policy.canonical_hash_hex()?,
            genesis_policy: signed_genesis_policy.policy.clone(),
            expires_at_unix_ms,
            origin_sign_public_key: signed_genesis_policy.origin_sign_public_key.clone(),
            signature_algorithm: ED25519_SIGNATURE_ALGORITHM.to_string(),
            signature: Vec::new(),
        };
        token.signature = owner.sign_bytes(&token.canonical_bytes()?).to_vec();
        token.verify_at(
            token
                .expires_at_unix_ms
                .unwrap_or_else(current_time_unix_ms)
                .saturating_sub(1),
        )?;
        Ok(token)
    }

    pub fn to_proto(&self) -> proto_node::SignedBootstrapToken {
        proto_node::SignedBootstrapToken {
            version: self.version,
            serialized_addrs: self.serialized_addrs.clone(),
            mesh_id: self.mesh_id.clone(),
            policy_hash: self.policy_hash.clone(),
            genesis_policy: Some(self.genesis_policy.to_proto()),
            expires_at_unix_ms: self.expires_at_unix_ms,
            origin_sign_public_key: self.origin_sign_public_key.clone(),
            signature_algorithm: self.signature_algorithm.clone(),
            signature: self.signature.clone(),
        }
    }

    pub fn from_proto(
        value: &proto_node::SignedBootstrapToken,
    ) -> Result<Self, MeshRequirementRejectReason> {
        let genesis_policy = value
            .genesis_policy
            .as_ref()
            .ok_or(MeshRequirementRejectReason::BootstrapTokenInvalid)
            .and_then(MeshGenesisPolicy::from_proto)?;
        let token = Self {
            version: value.version,
            serialized_addrs: value.serialized_addrs.clone(),
            mesh_id: value.mesh_id.clone(),
            policy_hash: value.policy_hash.clone(),
            genesis_policy,
            expires_at_unix_ms: value.expires_at_unix_ms,
            origin_sign_public_key: value.origin_sign_public_key.clone(),
            signature_algorithm: value.signature_algorithm.clone(),
            signature: value.signature.clone(),
        };
        token.validate()?;
        Ok(token)
    }

    pub fn validate(&self) -> Result<(), MeshRequirementRejectReason> {
        self.validate_unsigned_shape()?;
        if self.signature.is_empty() {
            return Err(MeshRequirementRejectReason::BootstrapTokenInvalid);
        }
        Ok(())
    }

    fn validate_unsigned_shape(&self) -> Result<(), MeshRequirementRejectReason> {
        if self.version != SIGNED_BOOTSTRAP_TOKEN_VERSION
            || self.mesh_id.trim().is_empty()
            || self.policy_hash.trim().is_empty()
            || self.origin_sign_public_key.len() != 32
            || self.signature_algorithm.trim().is_empty()
        {
            return Err(MeshRequirementRejectReason::BootstrapTokenInvalid);
        }
        self.genesis_policy.validate()?;
        Ok(())
    }

    pub fn canonical_bytes(&self) -> Result<Vec<u8>, MeshRequirementRejectReason> {
        self.validate_unsigned_shape()?;
        let mut buf = Vec::with_capacity(256);
        buf.extend_from_slice(SIGNED_BOOTSTRAP_TOKEN_DOMAIN_TAG);
        buf.extend_from_slice(&self.version.to_le_bytes());
        write_bytes_list(&mut buf, &self.serialized_addrs);
        write_string(&mut buf, self.mesh_id.trim());
        write_string(&mut buf, self.policy_hash.trim());
        buf.extend_from_slice(&self.genesis_policy.canonical_bytes()?);
        write_optional_u64(&mut buf, self.expires_at_unix_ms);
        write_bytes(&mut buf, &self.origin_sign_public_key);
        write_string(&mut buf, self.signature_algorithm.trim());
        Ok(buf)
    }

    pub fn verify(&self) -> Result<(), MeshRequirementRejectReason> {
        self.verify_at(current_time_unix_ms())
    }

    pub fn verify_at(&self, now_unix_ms: u64) -> Result<(), MeshRequirementRejectReason> {
        self.validate()?;
        if self.signature_algorithm.trim() != ED25519_SIGNATURE_ALGORITHM
            || self.signature.len() != 64
        {
            return Err(MeshRequirementRejectReason::BootstrapTokenInvalid);
        }
        let expected_policy_hash = self.genesis_policy.canonical_hash_hex()?;
        if self.policy_hash.trim() != expected_policy_hash {
            return Err(MeshRequirementRejectReason::MeshPolicyMismatch);
        }
        let expected_mesh_id = self.genesis_policy.policy_derived_mesh_id()?;
        if self.mesh_id.trim() != expected_mesh_id {
            return Err(MeshRequirementRejectReason::MeshPolicyMismatch);
        }
        if self
            .expires_at_unix_ms
            .is_some_and(|expires_at| now_unix_ms > expires_at)
        {
            return Err(MeshRequirementRejectReason::BootstrapTokenExpired);
        }
        let verifying_key = ed25519_dalek::VerifyingKey::from_bytes(
            &self
                .origin_sign_public_key
                .as_slice()
                .try_into()
                .map_err(|_| MeshRequirementRejectReason::BootstrapTokenInvalid)?,
        )
        .map_err(|_| MeshRequirementRejectReason::BootstrapTokenInvalid)?;
        if mesh_llm_identity::keys::owner_id_from_verifying_key(&verifying_key)
            != self.genesis_policy.origin_owner_id
        {
            return Err(MeshRequirementRejectReason::BootstrapTokenInvalid);
        }
        let signature = ed25519_dalek::Signature::from_bytes(
            &self
                .signature
                .as_slice()
                .try_into()
                .map_err(|_| MeshRequirementRejectReason::BootstrapTokenInvalid)?,
        );
        verifying_key
            .verify_strict(&self.canonical_bytes()?, &signature)
            .map_err(|_| MeshRequirementRejectReason::BootstrapTokenInvalid)
    }
}

impl ReleaseBuildAttestation {
    pub fn to_proto(&self) -> proto_node::ReleaseBuildAttestation {
        proto_node::ReleaseBuildAttestation {
            version: self.version,
            node_version: self.node_version.clone(),
            build_id: self.build_id.clone(),
            commit: self.commit.clone(),
            target_triple: self.target_triple.clone(),
            supported_protocol_generation_min: self.supported_protocol_generation_min,
            supported_protocol_generation_max: self.supported_protocol_generation_max,
            artifact_digest: self.artifact_digest.clone(),
            signer_key_id: self.signer_key_id.clone(),
            signature_algorithm: self.signature_algorithm.clone(),
            signature: self.signature.clone(),
        }
    }

    pub fn from_proto(
        value: &proto_node::ReleaseBuildAttestation,
    ) -> Result<Self, MeshRequirementRejectReason> {
        Ok(Self {
            version: value.version,
            node_version: value.node_version.clone(),
            build_id: value.build_id.clone(),
            commit: value.commit.clone(),
            target_triple: value.target_triple.clone(),
            supported_protocol_generation_min: value.supported_protocol_generation_min,
            supported_protocol_generation_max: value.supported_protocol_generation_max,
            artifact_digest: value.artifact_digest.clone(),
            signer_key_id: value.signer_key_id.clone(),
            signature_algorithm: value.signature_algorithm.clone(),
            signature: value.signature.clone(),
        })
    }

    pub fn validate(&self) -> Result<(), MeshRequirementRejectReason> {
        if self.version != RELEASE_BUILD_ATTESTATION_VERSION
            || self.node_version.trim().is_empty()
            || self.build_id.trim().is_empty()
            || self.commit.trim().is_empty()
            || self.target_triple.trim().is_empty()
            || self.signer_key_id.trim().is_empty()
            || self.signature_algorithm.trim().is_empty()
            || self.signature.is_empty()
        {
            return Err(MeshRequirementRejectReason::BuildProofInvalid);
        }
        if let (Some(min), Some(max)) = (
            self.supported_protocol_generation_min,
            self.supported_protocol_generation_max,
        ) {
            if min > max {
                return Err(MeshRequirementRejectReason::BuildProofInvalid);
            }
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> Result<Vec<u8>, MeshRequirementRejectReason> {
        self.validate()?;
        let mut buf = Vec::with_capacity(256);
        buf.extend_from_slice(RELEASE_BUILD_ATTESTATION_DOMAIN_TAG);
        buf.extend_from_slice(&self.version.to_le_bytes());
        write_string(&mut buf, self.node_version.trim());
        write_string(&mut buf, self.build_id.trim());
        write_string(&mut buf, self.commit.trim());
        write_string(&mut buf, self.target_triple.trim());
        write_optional_u32(&mut buf, self.supported_protocol_generation_min);
        write_optional_u32(&mut buf, self.supported_protocol_generation_max);
        write_optional_string(&mut buf, self.artifact_digest.as_deref());
        write_string(&mut buf, self.signer_key_id.trim());
        write_string(&mut buf, self.signature_algorithm.trim());
        Ok(buf)
    }

    pub fn canonical_hash_hex(&self) -> Result<String, MeshRequirementRejectReason> {
        Ok(hex::encode(Sha256::digest(self.canonical_bytes()?)))
    }

    pub fn verify(&self) -> Result<(), MeshRequirementRejectReason> {
        self.validate()?;
        if self.signature_algorithm.trim() != ED25519_SIGNATURE_ALGORITHM
            || self.signature.len() != 64
        {
            return Err(MeshRequirementRejectReason::BuildProofInvalid);
        }
        let signer_public_key = parse_release_signer_public_key(self.signer_key_id.trim())?;
        let signature = ed25519_dalek::Signature::from_bytes(
            &self
                .signature
                .as_slice()
                .try_into()
                .map_err(|_| MeshRequirementRejectReason::BuildProofInvalid)?,
        );
        signer_public_key
            .verify_strict(&self.canonical_bytes()?, &signature)
            .map_err(|_| MeshRequirementRejectReason::BuildProofInvalid)
    }
}

impl DirectNodeAdmissionProof {
    pub fn to_proto(&self) -> proto_node::DirectNodeAdmissionProof {
        proto_node::DirectNodeAdmissionProof {
            version: self.version,
            sender_id: self.sender_id.clone(),
            mesh_id: self.mesh_id.clone(),
            policy_hash: self.policy_hash.clone(),
            attestation_hash: self.attestation_hash.clone(),
            timestamp_unix_ms: self.timestamp_unix_ms,
            signature_algorithm: self.signature_algorithm.clone(),
            signature: self.signature.clone(),
        }
    }

    pub fn from_proto(
        value: &proto_node::DirectNodeAdmissionProof,
    ) -> Result<Self, MeshRequirementRejectReason> {
        let proof = Self {
            version: value.version,
            sender_id: value.sender_id.clone(),
            mesh_id: value.mesh_id.clone(),
            policy_hash: value.policy_hash.clone(),
            attestation_hash: value.attestation_hash.clone(),
            timestamp_unix_ms: value.timestamp_unix_ms,
            signature_algorithm: value.signature_algorithm.clone(),
            signature: value.signature.clone(),
        };
        proof.validate_shape()?;
        Ok(proof)
    }

    pub fn canonical_bytes(&self) -> Result<Vec<u8>, MeshRequirementRejectReason> {
        self.validate_unsigned_shape()?;
        let mut buf = Vec::with_capacity(192);
        buf.extend_from_slice(DIRECT_NODE_ADMISSION_PROOF_DOMAIN_TAG);
        buf.extend_from_slice(&self.version.to_le_bytes());
        write_bytes(&mut buf, &self.sender_id);
        write_string(&mut buf, self.mesh_id.trim());
        write_string(&mut buf, self.policy_hash.trim());
        write_string(&mut buf, self.attestation_hash.trim());
        buf.extend_from_slice(&self.timestamp_unix_ms.to_le_bytes());
        write_string(&mut buf, self.signature_algorithm.trim());
        Ok(buf)
    }

    pub fn validate_shape(&self) -> Result<(), MeshRequirementRejectReason> {
        if self.version != DIRECT_NODE_ADMISSION_PROOF_VERSION
            || self.sender_id.len() != 32
            || self.mesh_id.trim().is_empty()
            || self.policy_hash.trim().is_empty()
            || self.attestation_hash.trim().is_empty()
            || self.signature_algorithm.trim() != ED25519_SIGNATURE_ALGORITHM
        {
            return Err(MeshRequirementRejectReason::BuildProofInvalid);
        }
        if self.signature.len() != 64 {
            return Err(MeshRequirementRejectReason::BuildProofInvalid);
        }
        Ok(())
    }

    fn validate_unsigned_shape(&self) -> Result<(), MeshRequirementRejectReason> {
        if self.version != DIRECT_NODE_ADMISSION_PROOF_VERSION
            || self.sender_id.len() != 32
            || self.mesh_id.trim().is_empty()
            || self.policy_hash.trim().is_empty()
            || self.attestation_hash.trim().is_empty()
            || self.signature_algorithm.trim() != ED25519_SIGNATURE_ALGORITHM
        {
            return Err(MeshRequirementRejectReason::BuildProofInvalid);
        }
        Ok(())
    }

    pub fn verify_for_live_sender(
        &self,
        live_sender_id: &[u8],
        now_unix_ms: u64,
    ) -> Result<(), MeshRequirementRejectReason> {
        self.validate_shape()?;
        if live_sender_id.len() != 32 || self.sender_id.as_slice() != live_sender_id {
            return Err(MeshRequirementRejectReason::DirectProofSenderIdMismatch);
        }
        let skew = self.timestamp_unix_ms.abs_diff(now_unix_ms);
        if skew > DIRECT_NODE_ADMISSION_PROOF_MAX_CLOCK_SKEW_MS {
            return Err(MeshRequirementRejectReason::DirectProofStale);
        }
        let verifying_key = ed25519_dalek::VerifyingKey::from_bytes(
            &live_sender_id
                .try_into()
                .map_err(|_| MeshRequirementRejectReason::BuildProofInvalid)?,
        )
        .map_err(|_| MeshRequirementRejectReason::BuildProofInvalid)?;
        let signature = ed25519_dalek::Signature::from_bytes(
            &self
                .signature
                .as_slice()
                .try_into()
                .map_err(|_| MeshRequirementRejectReason::BuildProofInvalid)?,
        );
        verifying_key
            .verify_strict(&self.canonical_bytes()?, &signature)
            .map_err(|_| MeshRequirementRejectReason::BuildProofInvalid)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NormalizedNodeVersionBounds {
    pub min: Option<Version>,
    pub max: Option<Version>,
}

impl NormalizedNodeVersionBounds {
    fn is_constrained(&self) -> bool {
        self.min.is_some() || self.max.is_some()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NormalizedProtocolGenerationBounds {
    pub min: Option<u32>,
    pub max: Option<u32>,
}

impl NormalizedProtocolGenerationBounds {
    fn is_constrained(&self) -> bool {
        self.min.is_some() || self.max.is_some()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NormalizedReleaseAttestationRequirement {
    pub required: bool,
    pub allowed_signer_keys: Vec<String>,
}

fn parse_node_version(raw: &str) -> Result<Version, MeshRequirementRejectReason> {
    let normalized = raw.trim();
    if normalized.is_empty() {
        return Err(MeshRequirementRejectReason::NodeVersionMalformed);
    }
    let normalized = normalized.strip_prefix(['v', 'V']).unwrap_or(normalized);
    Version::parse(normalized).map_err(|_| MeshRequirementRejectReason::NodeVersionMalformed)
}

fn parse_release_signer_public_key(
    signer_key_id: &str,
) -> Result<ed25519_dalek::VerifyingKey, MeshRequirementRejectReason> {
    let encoded = signer_key_id
        .strip_prefix("ed25519:")
        .ok_or(MeshRequirementRejectReason::BuildProofInvalid)?;
    let bytes = hex::decode(encoded).map_err(|_| MeshRequirementRejectReason::BuildProofInvalid)?;
    let bytes: [u8; 32] = bytes
        .try_into()
        .map_err(|_| MeshRequirementRejectReason::BuildProofInvalid)?;
    ed25519_dalek::VerifyingKey::from_bytes(&bytes)
        .map_err(|_| MeshRequirementRejectReason::BuildProofInvalid)
}

fn version_precedence_cmp(left: &Version, right: &Version) -> std::cmp::Ordering {
    let mut left = left.clone();
    let mut right = right.clone();
    left.build = BuildMetadata::EMPTY;
    right.build = BuildMetadata::EMPTY;
    left.cmp(&right)
}

fn write_string(buf: &mut Vec<u8>, value: &str) {
    buf.extend_from_slice(&(value.len() as u64).to_le_bytes());
    buf.extend_from_slice(value.as_bytes());
}

fn write_optional_string(buf: &mut Vec<u8>, value: Option<&str>) {
    match value {
        Some(value) => {
            buf.push(1);
            write_string(buf, value);
        }
        None => buf.push(0),
    }
}

fn write_optional_u32(buf: &mut Vec<u8>, value: Option<u32>) {
    match value {
        Some(value) => {
            buf.push(1);
            buf.extend_from_slice(&value.to_le_bytes());
        }
        None => buf.push(0),
    }
}

fn write_optional_u64(buf: &mut Vec<u8>, value: Option<u64>) {
    match value {
        Some(value) => {
            buf.push(1);
            buf.extend_from_slice(&value.to_le_bytes());
        }
        None => buf.push(0),
    }
}

fn write_bytes(buf: &mut Vec<u8>, value: &[u8]) {
    buf.extend_from_slice(&(value.len() as u64).to_le_bytes());
    buf.extend_from_slice(value);
}

fn write_bytes_list(buf: &mut Vec<u8>, values: &[Vec<u8>]) {
    buf.extend_from_slice(&(values.len() as u64).to_le_bytes());
    for value in values {
        write_bytes(buf, value);
    }
}

fn write_string_list(buf: &mut Vec<u8>, values: &[String]) {
    buf.extend_from_slice(&(values.len() as u64).to_le_bytes());
    for value in values {
        write_string(buf, value);
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    fn restricted_requirements() -> MeshRequirements {
        MeshRequirements {
            node_version: NodeVersionBounds {
                min: Some("0.65.0".into()),
                max: Some("0.65.2".into()),
            },
            protocol_generation: ProtocolGenerationBounds {
                min: Some(1),
                max: Some(2),
            },
            release_attestation: ReleaseAttestationRequirement {
                required: true,
                allowed_signer_keys: vec!["signer-b".into(), "signer-a".into()],
            },
        }
    }

    pub(crate) fn assert_mesh_requirements_policy_canonical_hash_is_stable() {
        let (policy, local_input) = MeshGenesisPolicy::for_local_node(
            "owner-123",
            1_717_171_717_000,
            restricted_requirements(),
        )
        .expect("policy should validate");

        let first = policy.canonical_hash_hex().expect("hash should compute");
        let second = policy
            .canonical_hash_hex()
            .expect("hash should compute twice");

        assert_eq!(first, second);
        assert_eq!(
            policy.evaluate(&local_input),
            MeshRequirementDecision::Rejected(MeshRequirementRejectReason::CertifiedBinaryRequired),
            "the local-input helper should still reflect unsigned default attestation state"
        );
        assert_eq!(
            first, "40a3e2b4d96294e47f443c74d0d8441bd3363efea1580eb82627253ae47363ee",
            "keep this hash stable unless the canonical encoding intentionally changes"
        );
    }

    pub(crate) fn assert_mesh_requirements_policy_change_changes_mesh_id() {
        let baseline =
            MeshGenesisPolicy::new("owner-123", 1_717_171_717_000, restricted_requirements())
                .expect("policy should validate");
        let changed = MeshGenesisPolicy::new(
            "owner-123",
            1_717_171_717_000,
            MeshRequirements {
                node_version: NodeVersionBounds {
                    min: Some("0.65.1".into()),
                    max: Some("0.65.2".into()),
                },
                ..restricted_requirements()
            },
        )
        .expect("changed policy should validate");

        assert_ne!(
            baseline
                .policy_derived_mesh_id()
                .expect("baseline mesh id should compute"),
            changed
                .policy_derived_mesh_id()
                .expect("changed mesh id should compute")
        );
    }

    pub(crate) fn assert_mesh_requirements_bootstrap_token_validates_origin_signature() {
        let owner = crate::crypto::OwnerKeypair::generate();
        let signed_policy = SignedMeshGenesisPolicy::sign(
            MeshGenesisPolicy::new(
                owner.owner_id(),
                1_717_171_717_000,
                restricted_requirements(),
            )
            .expect("policy should validate"),
            &owner,
        )
        .expect("signed policy should validate");
        let mut token = SignedBootstrapToken::sign(
            vec![serde_json::to_vec(&serde_json::json!({
                "id": hex::encode([1u8; 32]),
                "addrs": []
            }))
            .expect("json should serialize")],
            &signed_policy,
            Some(1_717_171_717_000 + 60_000),
            &owner,
        )
        .expect("signed token should validate");

        assert!(token.verify_at(1_717_171_717_000).is_ok());
        token.signature[0] ^= 0x55;
        assert_eq!(
            token.verify_at(1_717_171_717_000),
            Err(MeshRequirementRejectReason::BootstrapTokenInvalid)
        );
    }

    pub(crate) fn assert_mesh_requirements_bootstrap_rejects_expired_token() {
        let owner = crate::crypto::OwnerKeypair::generate();
        let signed_policy = SignedMeshGenesisPolicy::sign(
            MeshGenesisPolicy::new(
                owner.owner_id(),
                1_717_171_717_000,
                restricted_requirements(),
            )
            .expect("policy should validate"),
            &owner,
        )
        .expect("signed policy should validate");
        let token = SignedBootstrapToken::sign(
            vec![serde_json::to_vec(&serde_json::json!({
                "id": hex::encode([2u8; 32]),
                "addrs": []
            }))
            .expect("json should serialize")],
            &signed_policy,
            Some(1_717_171_717_000 + 5),
            &owner,
        )
        .expect("signed token should validate");

        assert_eq!(
            token.verify_at(1_717_171_717_000 + 6),
            Err(MeshRequirementRejectReason::BootstrapTokenExpired)
        );
    }

    pub(crate) fn assert_mesh_requirements_bootstrap_rejects_policy_hash_mismatch() {
        let owner = crate::crypto::OwnerKeypair::generate();
        let signed_policy = SignedMeshGenesisPolicy::sign(
            MeshGenesisPolicy::new(
                owner.owner_id(),
                1_717_171_717_000,
                restricted_requirements(),
            )
            .expect("policy should validate"),
            &owner,
        )
        .expect("signed policy should validate");
        let mut token = SignedBootstrapToken::sign(
            vec![serde_json::to_vec(&serde_json::json!({
                "id": hex::encode([3u8; 32]),
                "addrs": []
            }))
            .expect("json should serialize")],
            &signed_policy,
            Some(1_717_171_717_000 + 60_000),
            &owner,
        )
        .expect("signed token should validate");
        token.policy_hash = "deadbeef".to_string();

        assert_eq!(
            token.verify_at(1_717_171_717_000),
            Err(MeshRequirementRejectReason::MeshPolicyMismatch)
        );
    }

    pub(crate) fn assert_mesh_requirements_policy_hash_derives_mesh_id() {
        let policy =
            MeshGenesisPolicy::new("owner-123", 1_717_171_717_000, restricted_requirements())
                .expect("policy should validate");
        assert_eq!(
            policy
                .policy_derived_mesh_id()
                .expect("mesh id should compute"),
            policy.canonical_hash_hex().expect("hash should compute")
        );
    }

    pub(crate) fn assert_mesh_requirements_version_bounds_unset_min_only_max_only_and_exact() {
        let unrestricted = MeshRequirements::unrestricted();
        let stable_input = MeshRequirementEvaluationInput {
            advertised_node_version: Some("0.65.1".into()),
            negotiated_protocol_generation: Some(NODE_PROTOCOL_GENERATION),
            direct_proof: DirectPeerProofStatus::Verified,
            ..Default::default()
        };
        assert_eq!(
            unrestricted.evaluate(&stable_input),
            MeshRequirementDecision::Accepted
        );

        let min_only = MeshRequirements {
            node_version: NodeVersionBounds {
                min: Some("0.65.1".into()),
                max: None,
            },
            ..MeshRequirements::unrestricted()
        };
        assert_eq!(
            min_only.evaluate(&stable_input),
            MeshRequirementDecision::Accepted
        );
        assert_eq!(
            min_only.evaluate(&MeshRequirementEvaluationInput {
                advertised_node_version: Some("0.65.0".into()),
                ..stable_input.clone()
            }),
            MeshRequirementDecision::Rejected(MeshRequirementRejectReason::NodeVersionBelowMinimum)
        );

        let max_only = MeshRequirements {
            node_version: NodeVersionBounds {
                min: None,
                max: Some("0.65.1".into()),
            },
            ..MeshRequirements::unrestricted()
        };
        assert_eq!(
            max_only.evaluate(&stable_input),
            MeshRequirementDecision::Accepted
        );
        assert_eq!(
            max_only.evaluate(&MeshRequirementEvaluationInput {
                advertised_node_version: Some("0.65.2".into()),
                ..stable_input.clone()
            }),
            MeshRequirementDecision::Rejected(MeshRequirementRejectReason::NodeVersionAboveMaximum)
        );

        let exact = MeshRequirements {
            node_version: NodeVersionBounds {
                min: Some("0.65.1".into()),
                max: Some("0.65.1".into()),
            },
            ..MeshRequirements::unrestricted()
        };
        assert_eq!(
            exact.evaluate(&stable_input),
            MeshRequirementDecision::Accepted
        );
        assert_eq!(
            exact.evaluate(&MeshRequirementEvaluationInput {
                advertised_node_version: Some("0.65.1-alpha.1".into()),
                direct_proof: DirectPeerProofStatus::Missing,
                ..stable_input.clone()
            }),
            MeshRequirementDecision::Rejected(MeshRequirementRejectReason::NodeVersionBelowMinimum)
        );
        assert_eq!(
            exact.evaluate(&MeshRequirementEvaluationInput {
                advertised_node_version: Some("0.65.1+build.99".into()),
                direct_proof: DirectPeerProofStatus::Invalid,
                ..stable_input
            }),
            MeshRequirementDecision::Accepted,
            "exact precedence checks should still accept build metadata variants"
        );
    }

    pub(crate) fn assert_mesh_requirements_protocol_bounds_reject_unknown_only_when_constrained() {
        let unconstrained = MeshRequirements::unrestricted();
        let unrestricted_policy =
            MeshGenesisPolicy::new("owner-123", 1_717_171_717_000, unconstrained)
                .expect("unrestricted policy should validate");
        assert_eq!(
            unrestricted_policy.evaluate(&MeshRequirementEvaluationInput {
                bootstrap: BootstrapStatus::Valid,
                ..Default::default()
            }),
            MeshRequirementDecision::Accepted
        );

        let constrained = MeshRequirements {
            protocol_generation: ProtocolGenerationBounds {
                min: Some(1),
                max: Some(2),
            },
            ..MeshRequirements::unrestricted()
        };
        let constrained_policy =
            MeshGenesisPolicy::new("owner-123", 1_717_171_717_000, constrained)
                .expect("constrained policy should validate");
        assert_eq!(
            constrained_policy.evaluate(&MeshRequirementEvaluationInput::default()),
            MeshRequirementDecision::Rejected(
                MeshRequirementRejectReason::ProtocolGenerationUnknown
            )
        );
        assert_eq!(
            constrained_policy.evaluate(&MeshRequirementEvaluationInput {
                bootstrap: BootstrapStatus::Invalid,
                negotiated_protocol_generation: Some(0),
                ..Default::default()
            }),
            MeshRequirementDecision::Rejected(MeshRequirementRejectReason::BootstrapTokenInvalid)
        );
        assert_eq!(
            constrained_policy.evaluate(&MeshRequirementEvaluationInput {
                bootstrap: BootstrapStatus::Expired,
                negotiated_protocol_generation: Some(1),
                ..Default::default()
            }),
            MeshRequirementDecision::Rejected(MeshRequirementRejectReason::BootstrapTokenExpired)
        );
        assert_eq!(
            constrained_policy.evaluate(&MeshRequirementEvaluationInput {
                negotiated_protocol_generation: Some(0),
                ..Default::default()
            }),
            MeshRequirementDecision::Rejected(
                MeshRequirementRejectReason::ProtocolGenerationBelowMinimum
            )
        );
        assert_eq!(
            constrained_policy.evaluate(&MeshRequirementEvaluationInput {
                negotiated_protocol_generation: Some(3),
                ..Default::default()
            }),
            MeshRequirementDecision::Rejected(
                MeshRequirementRejectReason::ProtocolGenerationAboveMaximum
            )
        );
        assert_eq!(
            constrained_policy.evaluate(&MeshRequirementEvaluationInput {
                negotiated_protocol_generation: Some(1),
                ..Default::default()
            }),
            MeshRequirementDecision::Accepted
        );
    }

    pub(crate) fn assert_mesh_requirements_rejects_unsigned_when_attestation_required() {
        let constrained = MeshRequirements {
            release_attestation: ReleaseAttestationRequirement {
                required: true,
                allowed_signer_keys: vec!["trusted-signer".into()],
            },
            ..MeshRequirements::unrestricted()
        };

        assert_eq!(
            constrained.evaluate(&MeshRequirementEvaluationInput::default()),
            MeshRequirementDecision::Rejected(MeshRequirementRejectReason::CertifiedBinaryRequired)
        );
        assert_eq!(
            constrained.evaluate(&MeshRequirementEvaluationInput {
                release_attestation: PeerReleaseAttestationStatus::Invalid,
                ..Default::default()
            }),
            MeshRequirementDecision::Rejected(MeshRequirementRejectReason::BuildProofInvalid)
        );
        assert_eq!(
            constrained.evaluate(&MeshRequirementEvaluationInput {
                release_attestation: PeerReleaseAttestationStatus::Present {
                    signer_key: Some("untrusted-signer".into()),
                    attested_version: Some(crate::VERSION.to_string()),
                },
                ..Default::default()
            }),
            MeshRequirementDecision::Rejected(MeshRequirementRejectReason::ReleaseSignerUntrusted)
        );
        assert_eq!(
            constrained.evaluate(&MeshRequirementEvaluationInput {
                release_attestation: PeerReleaseAttestationStatus::Present {
                    signer_key: Some("trusted-signer".into()),
                    attested_version: Some(crate::VERSION.to_string()),
                },
                ..Default::default()
            }),
            MeshRequirementDecision::Accepted
        );
    }

    pub(crate) fn assert_mesh_requirements_rejection_reasons_are_stable() {
        let stable = [
            (
                MeshRequirementRejectReason::CertifiedBinaryRequired,
                "certified_binary_required",
                "this mesh requires a certified mesh-llm binary; use a certified compiled binary to join.",
            ),
            (
                MeshRequirementRejectReason::BuildProofMissing,
                "build_proof_missing",
                "the peer's certified build proof is missing required signer metadata.",
            ),
            (
                MeshRequirementRejectReason::BuildProofInvalid,
                "build_proof_invalid",
                "the peer's certified build proof could not be verified.",
            ),
            (
                MeshRequirementRejectReason::ReleaseSignerUntrusted,
                "release_signer_untrusted",
                "the peer's certified build proof was signed by an untrusted release signer.",
            ),
            (
                MeshRequirementRejectReason::AttestationPolicyMismatch,
                "attestation_policy_mismatch",
                "the certified build or policy attestation does not match this mesh's requirements.",
            ),
            (
                MeshRequirementRejectReason::MeshPolicyMismatch,
                "mesh_policy_mismatch",
                "the peer or bootstrap token advertised a different mesh policy than this mesh requires.",
            ),
            (
                MeshRequirementRejectReason::BootstrapTokenInvalid,
                "bootstrap_token_invalid",
                "the bootstrap token is invalid for this mesh.",
            ),
            (
                MeshRequirementRejectReason::BootstrapTokenExpired,
                "bootstrap_token_expired",
                "the bootstrap token has expired for this mesh."
            ),
            (
                MeshRequirementRejectReason::NodeVersionBelowMinimum,
                "node_version_below_minimum",
                "the peer mesh-llm version is below this mesh's minimum allowed version.",
            ),
            (
                MeshRequirementRejectReason::NodeVersionAboveMaximum,
                "node_version_above_maximum",
                "the peer mesh-llm version is above this mesh's maximum allowed version.",
            ),
            (
                MeshRequirementRejectReason::NodeVersionMalformed,
                "node_version_malformed",
                "the peer advertised a malformed mesh-llm node version.",
            ),
            (
                MeshRequirementRejectReason::ProtocolGenerationBelowMinimum,
                "protocol_generation_below_minimum",
                "the peer protocol generation is below this mesh's minimum allowed generation.",
            ),
            (
                MeshRequirementRejectReason::ProtocolGenerationAboveMaximum,
                "protocol_generation_above_maximum",
                "the peer protocol generation is above this mesh's maximum allowed generation.",
            ),
            (
                MeshRequirementRejectReason::ProtocolGenerationUnknown,
                "protocol_generation_unknown",
                "the peer did not advertise a protocol generation required by this mesh.",
            ),
            (
                MeshRequirementRejectReason::TopologyDisclosureDenied,
                "topology_disclosure_denied",
                "topology disclosure was denied until the peer completes mesh admission.",
            ),
        ];

        for (reason, expected_code, expected_message) in stable {
            assert_eq!(reason.code(), expected_code);
            assert_eq!(reason.message(), expected_message);
            assert_eq!(serde_json::to_value(&reason).unwrap(), expected_code);
        }
    }

    pub(crate) fn assert_mesh_requirements_direct_proof_rejects_stale_timestamp() {
        use ed25519_dalek::{Signer, SigningKey};

        let signing_key = SigningKey::from_bytes(&[7u8; 32]);
        let sender_id = signing_key.verifying_key().to_bytes().to_vec();
        let now = 1_717_171_717_000u64;
        let stale_timestamp = now - DIRECT_NODE_ADMISSION_PROOF_MAX_CLOCK_SKEW_MS - 1;
        let mut proof = DirectNodeAdmissionProof {
            version: DIRECT_NODE_ADMISSION_PROOF_VERSION,
            sender_id: sender_id.clone(),
            mesh_id: "mesh-1".into(),
            policy_hash: "policy-hash".into(),
            attestation_hash: "attestation-hash".into(),
            timestamp_unix_ms: stale_timestamp,
            signature_algorithm: ED25519_SIGNATURE_ALGORITHM.into(),
            signature: vec![],
        };
        proof.signature = signing_key
            .sign(&proof.canonical_bytes().unwrap())
            .to_bytes()
            .to_vec();

        assert_eq!(
            proof.verify_for_live_sender(&sender_id, now),
            Err(MeshRequirementRejectReason::DirectProofStale)
        );
    }

    pub(crate) fn assert_mesh_requirements_direct_proof_rejects_sender_id_mismatch() {
        use ed25519_dalek::{Signer, SigningKey};

        let signing_key = SigningKey::from_bytes(&[9u8; 32]);
        let other_key = SigningKey::from_bytes(&[10u8; 32]);
        let sender_id = signing_key.verifying_key().to_bytes().to_vec();
        let live_sender_id = other_key.verifying_key().to_bytes().to_vec();
        let mut proof = DirectNodeAdmissionProof {
            version: DIRECT_NODE_ADMISSION_PROOF_VERSION,
            sender_id: sender_id.clone(),
            mesh_id: "mesh-1".into(),
            policy_hash: "policy-hash".into(),
            attestation_hash: "attestation-hash".into(),
            timestamp_unix_ms: 1_717_171_717_000u64,
            signature_algorithm: ED25519_SIGNATURE_ALGORITHM.into(),
            signature: vec![],
        };
        proof.signature = signing_key
            .sign(&proof.canonical_bytes().unwrap())
            .to_bytes()
            .to_vec();

        assert_eq!(
            proof.verify_for_live_sender(&live_sender_id, 1_717_171_717_000u64),
            Err(MeshRequirementRejectReason::DirectProofSenderIdMismatch)
        );
    }
}
