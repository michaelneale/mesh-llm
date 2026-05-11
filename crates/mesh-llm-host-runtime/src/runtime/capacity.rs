use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex};

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(super) enum RuntimeCapacityPool {
    Node,
    PinnedGpu(String),
}

impl fmt::Display for RuntimeCapacityPool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Node => f.write_str("node"),
            Self::PinnedGpu(stable_id) => write!(f, "pinned GPU {stable_id}"),
        }
    }
}

impl RuntimeCapacityPool {
    fn overlaps(&self, other: &Self) -> bool {
        // Node-wide local loads are not pinned to a backend device, so they
        // share the physical placement domain with every pinned GPU pool.
        // Pinned GPU pools remain independent from each other.
        self == other || matches!((self, other), (Self::Node, _) | (_, Self::Node))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct RuntimeCapacityRequest {
    pub(super) instance_id: String,
    pub(super) model_name: String,
    pub(super) pool: RuntimeCapacityPool,
    pub(super) capacity_bytes: u64,
    pub(super) required_bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct RuntimeCapacityAllocation {
    model_name: String,
    pool: RuntimeCapacityPool,
    required_bytes: u64,
    generation: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct RuntimeCapacityError {
    pub(super) model_name: String,
    pub(super) pool: RuntimeCapacityPool,
    pub(super) capacity_bytes: u64,
    pub(super) reserved_bytes: u64,
    pub(super) required_bytes: u64,
}

impl RuntimeCapacityError {
    pub(super) fn available_bytes(&self) -> u64 {
        self.capacity_bytes.saturating_sub(self.reserved_bytes)
    }

    pub(super) fn shortfall_bytes(&self) -> u64 {
        self.required_bytes.saturating_sub(self.available_bytes())
    }
}

impl fmt::Display for RuntimeCapacityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "runtime capacity for model '{}' exceeds {} pool: requires {}, available {}, reserved {}, capacity {}, short by {}",
            self.model_name,
            self.pool,
            format_gb(self.required_bytes),
            format_gb(self.available_bytes()),
            format_gb(self.reserved_bytes),
            format_gb(self.capacity_bytes),
            format_gb(self.shortfall_bytes())
        )
    }
}

impl std::error::Error for RuntimeCapacityError {}

#[derive(Clone, Debug, Default)]
pub(super) struct RuntimeCapacityLedger {
    inner: Arc<Mutex<RuntimeCapacityLedgerState>>,
}

#[derive(Debug, Default)]
struct RuntimeCapacityLedgerState {
    reservations: HashMap<String, RuntimeCapacityAllocation>,
    next_generation: u64,
}

#[derive(Debug)]
pub(super) struct RuntimeCapacityReservation {
    ledger: RuntimeCapacityLedger,
    instance_id: String,
    generation: u64,
    capacity_bytes: u64,
    reserved_bytes_excluding_self: u64,
}

impl RuntimeCapacityLedger {
    pub(super) fn reserve(
        &self,
        request: RuntimeCapacityRequest,
    ) -> Result<RuntimeCapacityReservation, RuntimeCapacityError> {
        let mut state = self.inner.lock().expect("runtime capacity ledger poisoned");
        let reserved_bytes = state
            .reservations
            .iter()
            .filter(|(instance_id, allocation)| {
                instance_id.as_str() != request.instance_id
                    && allocation.pool.overlaps(&request.pool)
            })
            .map(|(_, allocation)| allocation.required_bytes)
            .fold(0_u64, u64::saturating_add);

        if reserved_bytes.saturating_add(request.required_bytes) > request.capacity_bytes {
            return Err(RuntimeCapacityError {
                model_name: request.model_name,
                pool: request.pool,
                capacity_bytes: request.capacity_bytes,
                reserved_bytes,
                required_bytes: request.required_bytes,
            });
        }

        state.next_generation = state.next_generation.saturating_add(1);
        let generation = state.next_generation;
        state.reservations.insert(
            request.instance_id.clone(),
            RuntimeCapacityAllocation {
                model_name: request.model_name,
                pool: request.pool,
                required_bytes: request.required_bytes,
                generation,
            },
        );

        Ok(RuntimeCapacityReservation {
            ledger: self.clone(),
            instance_id: request.instance_id,
            generation,
            capacity_bytes: request.capacity_bytes,
            reserved_bytes_excluding_self: reserved_bytes,
        })
    }

    #[cfg(test)]
    pub(super) fn used_bytes(&self, pool: &RuntimeCapacityPool) -> u64 {
        let state = self.inner.lock().expect("runtime capacity ledger poisoned");
        state
            .reservations
            .values()
            .filter(|allocation| &allocation.pool == pool)
            .map(|allocation| allocation.required_bytes)
            .fold(0_u64, u64::saturating_add)
    }

    fn release_generation(&self, instance_id: &str, generation: u64) {
        let mut state = self.inner.lock().expect("runtime capacity ledger poisoned");
        let should_release = state
            .reservations
            .get(instance_id)
            .map(|allocation| allocation.generation == generation)
            .unwrap_or(false);
        if should_release {
            state.reservations.remove(instance_id);
        }
    }
}

impl RuntimeCapacityReservation {
    pub(super) fn capacity_budget_bytes(&self) -> u64 {
        self.capacity_bytes
            .saturating_sub(self.reserved_bytes_excluding_self)
    }
}

impl Drop for RuntimeCapacityReservation {
    fn drop(&mut self) {
        self.ledger
            .release_generation(&self.instance_id, self.generation);
    }
}

fn format_gb(bytes: u64) -> String {
    format!("{:.1}GB", bytes as f64 / 1e9)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(instance_id: &str, model_name: &str, required_bytes: u64) -> RuntimeCapacityRequest {
        pooled_request(
            instance_id,
            model_name,
            RuntimeCapacityPool::Node,
            1_000,
            required_bytes,
        )
    }

    fn pooled_request(
        instance_id: &str,
        model_name: &str,
        pool: RuntimeCapacityPool,
        capacity_bytes: u64,
        required_bytes: u64,
    ) -> RuntimeCapacityRequest {
        RuntimeCapacityRequest {
            instance_id: instance_id.to_string(),
            model_name: model_name.to_string(),
            pool,
            capacity_bytes,
            required_bytes,
        }
    }

    #[test]
    fn runtime_capacity_allows_duplicate_model_instances_when_capacity_remains() {
        let ledger = RuntimeCapacityLedger::default();

        let first = ledger
            .reserve(request("runtime-1", "Qwen", 400))
            .expect("first duplicate instance should reserve capacity");
        let second = ledger
            .reserve(request("runtime-2", "Qwen", 500))
            .expect("second duplicate instance should reserve remaining capacity");

        assert_eq!(ledger.used_bytes(&RuntimeCapacityPool::Node), 900);

        drop(first);
        assert_eq!(ledger.used_bytes(&RuntimeCapacityPool::Node), 500);

        drop(second);
        assert_eq!(ledger.used_bytes(&RuntimeCapacityPool::Node), 0);
    }

    #[test]
    fn runtime_capacity_rejects_instance_when_pool_is_short() {
        let ledger = RuntimeCapacityLedger::default();
        let _first = ledger
            .reserve(request("runtime-1", "Qwen", 700))
            .expect("first instance should reserve capacity");

        let err = ledger
            .reserve(request("runtime-2", "Qwen", 400))
            .expect_err("second instance should not overcommit local capacity");

        assert_eq!(err.model_name, "Qwen");
        assert_eq!(err.capacity_bytes, 1_000);
        assert_eq!(err.reserved_bytes, 700);
        assert_eq!(err.required_bytes, 400);
        assert_eq!(err.available_bytes(), 300);
        assert_eq!(err.shortfall_bytes(), 100);
        assert_eq!(ledger.used_bytes(&RuntimeCapacityPool::Node), 700);
    }

    #[test]
    fn runtime_capacity_replaces_same_instance_reservation() {
        let ledger = RuntimeCapacityLedger::default();
        let first = ledger
            .reserve(request("runtime-1", "Qwen", 400))
            .expect("initial reservation should succeed");
        let replacement = ledger
            .reserve(request("runtime-1", "Qwen", 600))
            .expect("same instance should be able to replace its reservation");

        assert_eq!(ledger.used_bytes(&RuntimeCapacityPool::Node), 600);
        assert_eq!(replacement.capacity_budget_bytes(), 1_000);

        drop(first);
        assert_eq!(
            ledger.used_bytes(&RuntimeCapacityPool::Node),
            600,
            "dropping a stale reservation must not release a newer reservation"
        );

        drop(replacement);
        assert_eq!(ledger.used_bytes(&RuntimeCapacityPool::Node), 0);
    }

    #[test]
    fn runtime_capacity_separates_pinned_gpu_pools() {
        let ledger = RuntimeCapacityLedger::default();

        let _first = ledger
            .reserve(pooled_request(
                "runtime-1",
                "Qwen",
                RuntimeCapacityPool::PinnedGpu("gpu-a".to_string()),
                1_000,
                800,
            ))
            .expect("first pinned GPU should reserve capacity");
        let _second = ledger
            .reserve(pooled_request(
                "runtime-2",
                "Qwen",
                RuntimeCapacityPool::PinnedGpu("gpu-b".to_string()),
                1_000,
                800,
            ))
            .expect("second pinned GPU should have independent capacity");

        assert_eq!(
            ledger.used_bytes(&RuntimeCapacityPool::PinnedGpu("gpu-a".to_string())),
            800
        );
        assert_eq!(
            ledger.used_bytes(&RuntimeCapacityPool::PinnedGpu("gpu-b".to_string())),
            800
        );
    }

    #[test]
    fn runtime_capacity_node_reservation_accounts_for_pinned_gpu_reservation() {
        let ledger = RuntimeCapacityLedger::default();
        let _pinned = ledger
            .reserve(pooled_request(
                "startup-gpu-a",
                "Qwen",
                RuntimeCapacityPool::PinnedGpu("gpu-a".to_string()),
                1_000,
                800,
            ))
            .expect("pinned startup model should reserve GPU capacity");

        let err = ledger
            .reserve(request("runtime-control", "Qwen", 300))
            .expect_err("node-wide runtime load should not bypass pinned GPU reservation");

        assert_eq!(err.pool, RuntimeCapacityPool::Node);
        assert_eq!(err.capacity_bytes, 1_000);
        assert_eq!(err.reserved_bytes, 800);
        assert_eq!(err.required_bytes, 300);
        assert_eq!(err.available_bytes(), 200);
        assert_eq!(err.shortfall_bytes(), 100);
    }

    #[test]
    fn runtime_capacity_node_reservation_allows_exact_remaining_pinned_capacity() {
        let ledger = RuntimeCapacityLedger::default();
        let _pinned = ledger
            .reserve(pooled_request(
                "startup-gpu-a",
                "Qwen",
                RuntimeCapacityPool::PinnedGpu("gpu-a".to_string()),
                1_000,
                700,
            ))
            .expect("pinned startup model should reserve GPU capacity");

        let node = ledger
            .reserve(request("runtime-control", "Qwen", 300))
            .expect("node-wide runtime load should use the exact remaining capacity");

        assert_eq!(node.capacity_budget_bytes(), 300);
        assert_eq!(ledger.used_bytes(&RuntimeCapacityPool::Node), 300);
    }

    #[test]
    fn runtime_capacity_pinned_gpu_reservation_accounts_for_node_reservation() {
        let ledger = RuntimeCapacityLedger::default();
        let _node = ledger
            .reserve(request("runtime-control", "Qwen", 700))
            .expect("node-wide runtime model should reserve aggregate local capacity");

        let err = ledger
            .reserve(pooled_request(
                "startup-gpu-a",
                "Qwen",
                RuntimeCapacityPool::PinnedGpu("gpu-a".to_string()),
                1_000,
                400,
            ))
            .expect_err("pinned GPU load should not bypass node-wide reservation");

        assert_eq!(
            err.pool,
            RuntimeCapacityPool::PinnedGpu("gpu-a".to_string())
        );
        assert_eq!(err.capacity_bytes, 1_000);
        assert_eq!(err.reserved_bytes, 700);
        assert_eq!(err.required_bytes, 400);
        assert_eq!(err.available_bytes(), 300);
        assert_eq!(err.shortfall_bytes(), 100);
    }

    #[test]
    fn runtime_capacity_node_reservation_counts_all_pinned_gpu_reservations() {
        let ledger = RuntimeCapacityLedger::default();
        let _first = ledger
            .reserve(pooled_request(
                "startup-gpu-a",
                "Qwen",
                RuntimeCapacityPool::PinnedGpu("gpu-a".to_string()),
                1_000,
                650,
            ))
            .expect("first pinned GPU should reserve capacity");
        let _second = ledger
            .reserve(pooled_request(
                "startup-gpu-b",
                "Qwen",
                RuntimeCapacityPool::PinnedGpu("gpu-b".to_string()),
                1_000,
                550,
            ))
            .expect("second pinned GPU should reserve independent capacity");

        let node = ledger
            .reserve(pooled_request(
                "runtime-control",
                "Qwen",
                RuntimeCapacityPool::Node,
                2_000,
                700,
            ))
            .expect("node-wide runtime load should fit in aggregate remaining capacity");

        assert_eq!(node.capacity_budget_bytes(), 800);

        let err = ledger
            .reserve(pooled_request(
                "runtime-control-2",
                "Qwen",
                RuntimeCapacityPool::Node,
                2_000,
                900,
            ))
            .expect_err("node-wide runtime load should include every pinned reservation");

        assert_eq!(err.reserved_bytes, 1_900);
        assert_eq!(err.available_bytes(), 100);
        assert_eq!(err.shortfall_bytes(), 800);
    }

    #[test]
    fn runtime_capacity_reservation_budget_excludes_other_instances() {
        let ledger = RuntimeCapacityLedger::default();
        let _first = ledger
            .reserve(request("runtime-1", "Qwen", 250))
            .expect("first instance should reserve capacity");
        let second = ledger
            .reserve(request("runtime-2", "Qwen", 400))
            .expect("second instance should reserve remaining capacity");

        assert_eq!(second.capacity_budget_bytes(), 750);
    }
}
