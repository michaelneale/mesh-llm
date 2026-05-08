use anyhow::{Context, Result};
use hf_hub::HFClient;

/// Result of checking the authenticated user's permissions relative to the
/// meshllm organization.
#[derive(Debug, Clone)]
pub struct PermissionCheck {
    /// HuggingFace username.
    pub username: String,
    /// Whether the user is a member of the `meshllm` org.
    pub is_meshllm_member: bool,
    /// Namespace for job submission and target repos.
    /// `"meshllm"` for org members, username for everyone else.
    pub namespace: String,
    /// Whether catalog updates should be submitted as PRs.
    /// `true` for non-members, `false` for org members.
    pub catalog_create_pr: bool,
}

/// Call `whoami`, inspect org memberships, and decide direct vs PR mode.
pub async fn check_permissions(client: &HFClient) -> Result<PermissionCheck> {
    let user = client
        .whoami()
        .await
        .context("HF whoami failed — is your token valid?")?;

    let is_meshllm_member = user
        .orgs
        .as_ref()
        .map(|orgs| {
            orgs.iter()
                .any(|org| org.name.as_deref() == Some("meshllm"))
        })
        .unwrap_or(false);

    let namespace = if is_meshllm_member {
        "meshllm".to_string()
    } else {
        user.username.clone()
    };

    Ok(PermissionCheck {
        username: user.username,
        is_meshllm_member,
        catalog_create_pr: !is_meshllm_member,
        namespace,
    })
}
