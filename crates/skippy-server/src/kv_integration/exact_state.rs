use anyhow::{Context, Result};
use skippy_cache::ExactStatePayload;

use crate::runtime_state::RuntimeState;

use super::{
    records::add_reconstruct_stats, ExactStateExtra, ExactStateRecord, ExactStateRestore,
    KvStageIntegration, PrefillKvIdentity, StagePrefixCachePayload,
};

impl KvStageIntegration {
    pub fn restore_exact_state(
        &self,
        runtime: &mut RuntimeState,
        session_id: &str,
        identities: &[PrefillKvIdentity],
    ) -> Result<Option<ExactStateRestore>> {
        if !self.should_lookup() || !self.payload.is_exact_state() {
            return Ok(None);
        }
        for identity in identities {
            let lookup = {
                self.exact_states
                    .lock()
                    .expect("exact state cache lock poisoned")
                    .lookup(&identity.page_id)
            };
            let Some(lookup) = lookup else {
                continue;
            };
            let mut reconstruct_ms = 0.0;
            let mut reconstruct_bytes = 0u64;
            let mut reconstruct_blocks = 0usize;
            match lookup.payload.kind().into() {
                StagePrefixCachePayload::FullState => {
                    let (full_state, stats) = lookup
                        .payload
                        .full_state_bytes_timed()
                        .context("reconstruct cached full-state payload")?;
                    add_reconstruct_stats(
                        &mut reconstruct_ms,
                        &mut reconstruct_bytes,
                        &mut reconstruct_blocks,
                        stats,
                    );
                    runtime.import_full_state_for_token_count(
                        session_id,
                        full_state.as_ref(),
                        lookup.token_count,
                    )?;
                }
                StagePrefixCachePayload::KvRecurrent => {
                    let Some(desc) = lookup.extra.kv_desc else {
                        continue;
                    };
                    if let Some((kv, stats)) = lookup
                        .payload
                        .kv_bytes_timed()
                        .context("reconstruct cached KV payload")?
                    {
                        add_reconstruct_stats(
                            &mut reconstruct_ms,
                            &mut reconstruct_bytes,
                            &mut reconstruct_blocks,
                            stats,
                        );
                        runtime.import_kv_page(session_id, &desc, kv.as_ref())?;
                    }
                    let (recurrent, stats) = lookup
                        .payload
                        .recurrent_state_bytes_timed()
                        .context("reconstruct cached recurrent payload")?;
                    add_reconstruct_stats(
                        &mut reconstruct_ms,
                        &mut reconstruct_bytes,
                        &mut reconstruct_blocks,
                        stats,
                    );
                    runtime.import_recurrent_state_for_token_count(
                        session_id,
                        recurrent.as_ref(),
                        lookup.token_count,
                    )?;
                }
                _ => continue,
            }
            return Ok(Some(ExactStateRestore {
                page_id: lookup.page_id,
                token_count: lookup.token_count as usize,
                payload_kind: lookup.payload.kind(),
                logical_bytes: lookup.logical_bytes,
                entries: lookup.entries,
                reconstruct_ms,
                reconstruct_bytes,
                reconstruct_blocks,
            }));
        }
        Ok(None)
    }

    pub fn record_exact_state(
        &self,
        runtime: &mut RuntimeState,
        session_id: &str,
        identity: &PrefillKvIdentity,
    ) -> Result<Option<ExactStateRecord>> {
        if !self.should_record() || !self.payload.is_exact_state() {
            return Ok(None);
        }
        let token_count = identity.identity.token_count;
        if token_count < self.candidate_policy.min_tokens {
            return Ok(None);
        }
        if !self.try_begin_record(&identity.page_id) {
            return Ok(None);
        }
        let result = (|| {
            let (payload, extra) = match self.payload {
                StagePrefixCachePayload::FullState => (
                    ExactStatePayload::full_state(runtime.export_full_state(session_id)?),
                    ExactStateExtra::default(),
                ),
                StagePrefixCachePayload::KvRecurrent => {
                    let kv = runtime.export_kv_page(session_id, 0, token_count)?;
                    let recurrent = runtime.export_recurrent_state(session_id)?;
                    (
                        ExactStatePayload::kv_recurrent(kv.payload, recurrent),
                        ExactStateExtra {
                            kv_desc: Some(kv.desc),
                        },
                    )
                }
                _ => return Ok(None),
            };
            let outcome = self
                .exact_states
                .lock()
                .expect("exact state cache lock poisoned")
                .record(identity.page_id.clone(), token_count, payload, extra);
            Ok(Some(ExactStateRecord {
                page_id: outcome.page_id,
                token_count: outcome.token_count as usize,
                payload_kind: self.payload.into(),
                stored: outcome.stored,
                logical_bytes: outcome.logical_bytes,
                physical_bytes: outcome.physical_bytes,
                entries: outcome.entries,
                evicted_entries: outcome.evicted_entries,
                evicted_logical_bytes: outcome.evicted_logical_bytes,
                dedupe: outcome.dedupe,
            }))
        })();
        self.finish_record(&identity.page_id);
        result
    }
}

impl StagePrefixCachePayload {
    pub(crate) fn is_exact_state(self) -> bool {
        matches!(self, Self::KvRecurrent | Self::FullState)
    }
}

impl From<skippy_cache::ExactStatePayloadKind> for StagePrefixCachePayload {
    fn from(kind: skippy_cache::ExactStatePayloadKind) -> Self {
        match kind {
            skippy_cache::ExactStatePayloadKind::FullState => Self::FullState,
            skippy_cache::ExactStatePayloadKind::KvRecurrent => Self::KvRecurrent,
            skippy_cache::ExactStatePayloadKind::RecurrentOnly => Self::Disabled,
        }
    }
}

impl From<StagePrefixCachePayload> for skippy_cache::ExactStatePayloadKind {
    fn from(payload: StagePrefixCachePayload) -> Self {
        match payload {
            StagePrefixCachePayload::FullState => Self::FullState,
            StagePrefixCachePayload::KvRecurrent => Self::KvRecurrent,
            StagePrefixCachePayload::Disabled | StagePrefixCachePayload::ResidentKv => {
                Self::FullState
            }
        }
    }
}
