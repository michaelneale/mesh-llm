use skippy_cache::activation_page_id;
use skippy_protocol::{MessageBase, StageConfig};
use skippy_runtime::ActivationFrame;

use super::{KvStageIntegration, ResidentActivationRecord, ResidentActivationRestore};

impl KvStageIntegration {
    pub fn restore_resident_activation(
        &self,
        config: &StageConfig,
        base: &MessageBase,
        token_start: u64,
        token_ids: &[i32],
        activation_width: i32,
    ) -> Option<ResidentActivationRestore> {
        if !self.should_lookup() || token_ids.is_empty() {
            return None;
        }
        let identity = self.prefill_identity(config, base, token_start, token_ids);
        let page_id = activation_page_id(&identity.page_id, activation_width);
        let lookup = self
            .activations
            .lock()
            .expect("resident activation cache lock poisoned")
            .lookup(&page_id)?;
        Some(ResidentActivationRestore {
            identity,
            page_id,
            token_count: lookup.token_count as usize,
            payload_bytes: lookup.byte_size as usize,
            entries: lookup.entries,
            frame: lookup.frame,
        })
    }

    pub fn record_resident_activation(
        &self,
        config: &StageConfig,
        base: &MessageBase,
        token_start: u64,
        token_ids: &[i32],
        activation_width: i32,
        frame: &ActivationFrame,
    ) -> Option<ResidentActivationRecord> {
        if !self.should_record() || token_ids.is_empty() {
            return None;
        }
        let token_count = token_ids.len() as u64;
        if token_count < self.candidate_policy.min_tokens || frame.payload.is_empty() {
            return None;
        }
        let identity = self.prefill_identity(config, base, token_start, token_ids);
        let page_id = activation_page_id(&identity.page_id, activation_width);
        let mut cache = self
            .activations
            .lock()
            .expect("resident activation cache lock poisoned");
        let stored = cache.record(
            page_id.clone(),
            token_count,
            frame.payload.len() as u64,
            frame.clone(),
        );
        let stats = cache.stats();
        Some(ResidentActivationRecord {
            page_id,
            token_count: token_count as usize,
            payload_bytes: frame.payload.len(),
            evicted_entries: stored.evicted_entries,
            evicted_bytes: stored.evicted_bytes,
            entries: stats.entries,
            resident_bytes: stats.resident_bytes,
        })
    }
}
