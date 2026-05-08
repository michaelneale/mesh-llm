use super::snapshots::LocalInstancesSnapshot;
use crate::models::LocalModelInventorySnapshot;
use crate::runtime::instance::LocalInstanceSnapshot;
use tokio::sync::oneshot;

#[derive(Default)]
pub(crate) struct InventoryScanCoordinator {
    running: bool,
    waiters: Vec<oneshot::Sender<LocalModelInventorySnapshot>>,
}

impl InventoryScanCoordinator {
    pub(crate) fn begin_or_join(
        &mut self,
    ) -> (oneshot::Receiver<LocalModelInventorySnapshot>, bool) {
        let (tx, rx) = oneshot::channel();
        self.waiters.push(tx);
        if self.running {
            (rx, false)
        } else {
            self.running = true;
            (rx, true)
        }
    }

    pub(crate) fn finish(&mut self) -> Vec<oneshot::Sender<LocalModelInventorySnapshot>> {
        self.running = false;
        std::mem::take(&mut self.waiters)
    }
}

pub(crate) fn replace_local_instances_snapshot(
    current: &mut LocalInstancesSnapshot,
    replacement: Vec<LocalInstanceSnapshot>,
) -> bool {
    if current.instances == replacement {
        return false;
    }

    current.instances = replacement;
    true
}

pub(crate) fn replace_local_inventory_snapshot(
    current: &mut LocalModelInventorySnapshot,
    replacement: LocalModelInventorySnapshot,
) -> bool {
    if *current == replacement {
        return false;
    }

    *current = replacement;
    true
}
