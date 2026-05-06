//! Versioned dirty-bit subscriptions for runtime-data snapshots.
//!
//! The payload stays intentionally small so hot paths can publish without
//! awaiting and subscribers can coalesce updates by version.

use std::ops::{BitOr, BitOrAssign};
use tokio::sync::watch;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct RuntimeDataVersion(u64);

impl RuntimeDataVersion {
    #[cfg(test)]
    pub(crate) fn get(self) -> u64 {
        self.0
    }

    fn next(self) -> Self {
        Self(self.0.saturating_add(1))
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub(crate) struct RuntimeDataDirty(u8);

impl RuntimeDataDirty {
    pub(crate) const STATUS: Self = Self(1 << 0);
    pub(crate) const MODELS: Self = Self(1 << 1);
    pub(crate) const ROUTING: Self = Self(1 << 2);
    pub(crate) const PROCESSES: Self = Self(1 << 3);
    pub(crate) const INVENTORY: Self = Self(1 << 4);
    pub(crate) const PLUGINS: Self = Self(1 << 5);
    pub(crate) const RUNTIME: Self = Self(1 << 6);

    pub(crate) fn is_empty(self) -> bool {
        self.0 == 0
    }

    pub(crate) fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
}

impl BitOr for RuntimeDataDirty {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

impl BitOrAssign for RuntimeDataDirty {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct RuntimeDataSubscriptionState {
    pub version: RuntimeDataVersion,
    pub dirty: RuntimeDataDirty,
}

#[derive(Clone)]
pub(crate) struct RuntimeDataSubscriptions {
    sender: watch::Sender<RuntimeDataSubscriptionState>,
}

impl Default for RuntimeDataSubscriptions {
    fn default() -> Self {
        let (sender, _) = watch::channel(RuntimeDataSubscriptionState::default());
        Self { sender }
    }
}

impl RuntimeDataSubscriptions {
    pub(crate) fn subscribe(&self) -> watch::Receiver<RuntimeDataSubscriptionState> {
        self.sender.subscribe()
    }

    pub(crate) fn state(&self) -> RuntimeDataSubscriptionState {
        *self.sender.borrow()
    }

    pub(crate) fn publish(&self, dirty: RuntimeDataDirty) -> RuntimeDataSubscriptionState {
        if dirty.is_empty() {
            return self.state();
        }

        let mut published = None;
        let changed = self.sender.send_if_modified(|state| {
            state.version = state.version.next();
            state.dirty |= dirty;
            published = Some(*state);
            true
        });

        if changed {
            published.expect("published runtime data subscription state")
        } else {
            self.state()
        }
    }
}
