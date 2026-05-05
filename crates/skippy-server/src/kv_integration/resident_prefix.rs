use anyhow::Result;

use crate::runtime_state::RuntimeState;

use super::{
    KvStageIntegration, PrefillKvIdentity, ResidentPrefixRecord, ResidentPrefixRestore,
    StagePrefixCachePayload,
};

impl KvStageIntegration {
    pub fn restore_resident_prefix(
        &self,
        runtime: &mut RuntimeState,
        session_id: &str,
        identities: &[PrefillKvIdentity],
        token_ids: &[i32],
    ) -> Result<Option<ResidentPrefixRestore>> {
        if !self.should_lookup() || self.payload != StagePrefixCachePayload::ResidentKv {
            return Ok(None);
        }
        for identity in identities {
            let token_count = identity
                .identity
                .token_count
                .try_into()
                .unwrap_or(usize::MAX)
                .min(token_ids.len());
            if token_count == 0 {
                continue;
            }
            let lookup = {
                self.resident
                    .lock()
                    .expect("resident prefix cache lock poisoned")
                    .lookup(&identity.page_id)
            };
            let Some(lookup) = lookup else {
                continue;
            };
            runtime.restore_resident_prefix(
                session_id,
                lookup.seq_id,
                &token_ids[..token_count],
            )?;
            return Ok(Some(ResidentPrefixRestore {
                page_id: identity.page_id.clone(),
                token_count,
                seq_id: lookup.seq_id,
                entries: lookup.entries,
            }));
        }
        Ok(None)
    }

    pub fn record_resident_prefix(
        &self,
        runtime: &mut RuntimeState,
        session_id: &str,
        identity: &PrefillKvIdentity,
        token_ids: &[i32],
    ) -> Result<Option<ResidentPrefixRecord>> {
        if !self.should_record() || self.payload != StagePrefixCachePayload::ResidentKv {
            return Ok(None);
        }
        let token_count = identity
            .identity
            .token_count
            .try_into()
            .unwrap_or(usize::MAX)
            .min(token_ids.len());
        if token_count == 0 || (token_count as u64) < self.candidate_policy.min_tokens {
            return Ok(None);
        }
        let layer_count = identity
            .identity
            .layer_end
            .saturating_sub(identity.identity.layer_start)
            .max(1);
        let estimated_bytes = resident_estimated_bytes(token_count as u64, layer_count);
        let mut cache = self
            .resident
            .lock()
            .expect("resident prefix cache lock poisoned");
        let allocation = cache.allocate_for_record(
            &identity.page_id,
            token_count as u64,
            estimated_bytes,
            |seq_id| runtime.drop_resident_prefix_sequence(session_id, seq_id),
        )?;
        runtime.save_resident_prefix(session_id, allocation.seq_id, token_count as u64)?;
        cache.commit_record(
            identity.page_id.clone(),
            allocation.seq_id,
            token_count as u64,
            estimated_bytes,
        );
        let stats = cache.stats();
        Ok(Some(ResidentPrefixRecord {
            page_id: identity.page_id.clone(),
            token_count,
            seq_id: allocation.seq_id,
            stored: true,
            evicted_entries: allocation.evictions.len(),
            evicted_tokens: allocation
                .evictions
                .iter()
                .map(|eviction| eviction.token_count)
                .sum(),
            entries: stats.entries,
            resident_tokens: stats.resident_tokens,
        }))
    }
}

fn resident_estimated_bytes(token_count: u64, layer_count: u32) -> u64 {
    token_count
        .saturating_mul(u64::from(layer_count))
        .saturating_mul(2)
}
