#[derive(Clone, Debug)]
pub struct InviteToken(String);

impl InviteToken {
    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub(crate) fn into_inner(self) -> mesh_client::InviteToken {
        mesh_client::InviteToken(self.0)
    }
}

impl std::str::FromStr for InviteToken {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let token = s.parse::<mesh_client::InviteToken>()?;
        Ok(Self(token.0))
    }
}
