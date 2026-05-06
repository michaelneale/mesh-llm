use crate::{ClientBuilder, InviteToken, MeshApiError, MeshClient, OwnerKeypair};

#[derive(Clone, Debug, Default)]
pub struct PublicMeshQuery {
    pub model: Option<String>,
    pub min_vram_gb: Option<f64>,
    pub region: Option<String>,
    pub target_name: Option<String>,
    pub relays: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct PublicMesh {
    pub invite_token: String,
    pub serving: Vec<String>,
    pub wanted: Vec<String>,
    pub on_disk: Vec<String>,
    pub total_vram_bytes: u64,
    pub node_count: usize,
    pub client_count: usize,
    pub max_clients: usize,
    pub name: Option<String>,
    pub region: Option<String>,
    pub mesh_id: Option<String>,
    pub publisher_npub: String,
    pub published_at: u64,
    pub expires_at: Option<u64>,
}

pub struct AutoConnectResult {
    pub client: MeshClient,
    pub selected_mesh: PublicMesh,
}

impl PublicMesh {
    pub fn invite_token(&self) -> &str {
        &self.invite_token
    }

    pub fn client_builder(
        &self,
        owner_keypair: OwnerKeypair,
    ) -> Result<ClientBuilder, MeshApiError> {
        ClientBuilder::from_public_mesh(owner_keypair, self)
    }
}

impl From<mesh_client::network::nostr::DiscoveredMesh> for PublicMesh {
    fn from(value: mesh_client::network::nostr::DiscoveredMesh) -> Self {
        Self {
            invite_token: value.listing.invite_token,
            serving: value.listing.serving,
            wanted: value.listing.wanted,
            on_disk: value.listing.on_disk,
            total_vram_bytes: value.listing.total_vram_bytes,
            node_count: value.listing.node_count,
            client_count: value.listing.client_count,
            max_clients: value.listing.max_clients,
            name: value.listing.name,
            region: value.listing.region,
            mesh_id: value.listing.mesh_id,
            publisher_npub: value.publisher_npub,
            published_at: value.published_at,
            expires_at: value.expires_at,
        }
    }
}

pub async fn discover_public_meshes(
    query: PublicMeshQuery,
) -> Result<Vec<PublicMesh>, MeshApiError> {
    let relays = resolve_relays(&query.relays);
    let filter = mesh_client::network::nostr::MeshFilter {
        model: query.model.clone(),
        min_vram_gb: query.min_vram_gb,
        region: query.region.clone(),
    };

    let discovered = mesh_client::network::nostr::discover(&relays, &filter, None)
        .await
        .map_err(|err| MeshApiError::Discovery(err.to_string()))?;

    Ok(discovered
        .into_iter()
        .filter(|mesh| matches_target_name(mesh, query.target_name.as_deref()))
        .map(Into::into)
        .collect())
}

pub async fn create_auto_client(
    owner_keypair: OwnerKeypair,
    query: PublicMeshQuery,
) -> Result<AutoConnectResult, MeshApiError> {
    let meshes = discover_public_meshes(query.clone()).await?;
    let discovered: Vec<mesh_client::network::nostr::DiscoveredMesh> =
        meshes.into_iter().map(public_mesh_to_discovered).collect();

    let selection = mesh_client::network::nostr::smart_auto(
        &discovered,
        0.0,
        query.target_name.as_deref(),
        None,
    );

    let mesh = match selection {
        mesh_client::network::nostr::AutoDecision::Join { mut candidates } => candidates
            .drain(..)
            .next()
            .map(|(_, mesh)| PublicMesh::from(mesh))
            .ok_or(MeshApiError::NoPublicMeshFound)?,
        mesh_client::network::nostr::AutoDecision::StartNew { .. } => {
            return Err(MeshApiError::NoPublicMeshFound)
        }
    };

    let client = mesh.client_builder(owner_keypair)?.build()?;

    Ok(AutoConnectResult {
        client,
        selected_mesh: mesh,
    })
}

impl ClientBuilder {
    pub fn from_public_mesh(
        owner_keypair: OwnerKeypair,
        mesh: &PublicMesh,
    ) -> Result<Self, MeshApiError> {
        let token = mesh
            .invite_token
            .parse::<InviteToken>()
            .map_err(MeshApiError::InvalidInviteToken)?;
        Ok(Self::new(owner_keypair, token))
    }
}

fn resolve_relays(relays: &[String]) -> Vec<String> {
    if relays.is_empty() {
        mesh_client::network::nostr::DEFAULT_RELAYS
            .iter()
            .map(|relay| (*relay).to_string())
            .collect()
    } else {
        relays.to_vec()
    }
}

fn matches_target_name(
    mesh: &mesh_client::network::nostr::DiscoveredMesh,
    target_name: Option<&str>,
) -> bool {
    let Some(target_name) = target_name else {
        return true;
    };

    mesh.listing
        .name
        .as_deref()
        .map(|name| name.eq_ignore_ascii_case(target_name))
        .unwrap_or(false)
}

fn public_mesh_to_discovered(mesh: PublicMesh) -> mesh_client::network::nostr::DiscoveredMesh {
    mesh_client::network::nostr::DiscoveredMesh {
        listing: mesh_client::network::nostr::MeshListing {
            invite_token: mesh.invite_token,
            serving: mesh.serving,
            wanted: mesh.wanted,
            on_disk: mesh.on_disk,
            total_vram_bytes: mesh.total_vram_bytes,
            node_count: mesh.node_count,
            client_count: mesh.client_count,
            max_clients: mesh.max_clients,
            name: mesh.name,
            region: mesh.region,
            mesh_id: mesh.mesh_id,
        },
        publisher_npub: mesh.publisher_npub,
        published_at: mesh.published_at,
        expires_at: mesh.expires_at,
    }
}

#[cfg(test)]
mod tests {
    use super::{matches_target_name, PublicMesh};

    fn sample_mesh(name: Option<&str>) -> mesh_client::network::nostr::DiscoveredMesh {
        mesh_client::network::nostr::DiscoveredMesh {
            listing: mesh_client::network::nostr::MeshListing {
                invite_token: "mesh-test:abc123".to_string(),
                serving: vec!["Qwen".to_string()],
                wanted: vec![],
                on_disk: vec![],
                total_vram_bytes: 32_000_000_000,
                node_count: 2,
                client_count: 1,
                max_clients: 0,
                name: name.map(str::to_string),
                region: Some("AU".to_string()),
                mesh_id: Some("mesh-1".to_string()),
            },
            publisher_npub: "npub1test".to_string(),
            published_at: 1,
            expires_at: Some(2),
        }
    }

    #[test]
    fn target_name_filter_is_case_insensitive() {
        let mesh = sample_mesh(Some("Mesh-LLM"));
        assert!(matches_target_name(&mesh, Some("mesh-llm")));
        assert!(!matches_target_name(&mesh, Some("other")));
    }

    #[test]
    fn public_mesh_can_build_client_builder() {
        let mesh = PublicMesh::from(sample_mesh(Some("mesh-llm")));
        let owner_keypair = crate::OwnerKeypair::generate();
        let builder = mesh.client_builder(owner_keypair);
        assert!(builder.is_ok());
    }
}
