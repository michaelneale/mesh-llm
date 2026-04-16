use std::sync::Arc;

use tokio::sync::RwLock;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum WakeableState {
    Sleeping,
    Waking,
}

impl TryFrom<&str> for WakeableState {
    type Error = String;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "sleeping" => Ok(Self::Sleeping),
            "waking" => Ok(Self::Waking),
            other => Err(format!("unsupported wakeable state: {other}")),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct WakeableInventoryEntry {
    pub(crate) logical_id: String,
    pub(crate) models: Vec<String>,
    pub(crate) vram_gb: f32,
    pub(crate) provider: Option<String>,
    pub(crate) state: WakeableState,
    pub(crate) wake_eta_secs: Option<u32>,
}

#[derive(Clone, Default)]
pub(crate) struct WakeableInventory {
    entries: Arc<RwLock<Vec<WakeableInventoryEntry>>>,
}

impl WakeableInventory {
    pub(crate) async fn status_snapshot(&self) -> Vec<WakeableInventoryEntry> {
        self.entries.read().await.clone()
    }

    #[cfg(test)]
    pub(crate) async fn replace_for_tests(&self, entries: Vec<WakeableInventoryEntry>) {
        *self.entries.write().await = entries;
    }
}

#[cfg(test)]
mod tests {
    use super::WakeableState;

    #[test]
    fn wakeable_state_rejects_unknown_values() {
        assert_eq!(
            WakeableState::try_from("sleeping"),
            Ok(WakeableState::Sleeping)
        );
        assert_eq!(WakeableState::try_from("waking"), Ok(WakeableState::Waking));
        assert_eq!(
            WakeableState::try_from("invalid-state"),
            Err("unsupported wakeable state: invalid-state".to_string())
        );
    }
}
