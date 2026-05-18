use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
    time::Instant,
};

use anyhow::{bail, Context, Result};
use skippy_protocol::{FlashAttentionType, LoadMode, StageConfig};
use skippy_runtime::{
    parse_cache_type, ActivationFrame, FlashAttentionType as RuntimeFlashAttentionType,
    GenerationSignalWindow, MediaInput, MediaPrefill, MediaPrefillFrame, RuntimeConfig,
    RuntimeKvPage, RuntimeKvPageDesc, RuntimeLoadMode, SamplingConfig, StageModel, StageSession,
    StageSessionCheckpoint, TokenSignal,
};

use crate::package::select_package_parts;

pub struct RuntimeState {
    pub model: StageModel,
    layer_start: u32,
    layer_end: u32,
    lane_count: u32,
    /// High-water mark of lane indices ever handed out. Combined with
    /// [`Self::free_lane_indices`], the count of live lanes equals
    /// `next_lane_index - free_lane_indices.len()`.
    next_lane_index: usize,
    /// Lane indices that were previously handed out but are now free
    /// to reuse. An index lands here only when the lane's underlying
    /// StageSession has been dropped (which calls skippy_session_free
    /// on the C side, clearing that seq_id's KV cells).
    ///
    /// Without this list, a discarded lane (see
    /// [`Self::drop_session_timed`]) would permanently consume one of
    /// the slots represented by [`Self::next_lane_index`], leading to
    /// "all execution lanes are busy" errors long before the runtime
    /// has actually run out of capacity.
    free_lane_indices: Vec<usize>,
    sessions: BTreeMap<String, RuntimeLaneSession>,
    idle_sessions: Vec<RuntimeLaneSession>,
    session_token_counts: BTreeMap<String, u64>,
    session_checkpoints: BTreeMap<String, StageSessionCheckpoint>,
    session_resident_prefixes: BTreeMap<String, ResidentLanePrefix>,
}

struct RuntimeLaneSession {
    index: usize,
    session: StageSession,
    resident_prefix: Option<ResidentLanePrefix>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RuntimeSessionLaneStats {
    pub index: usize,
    pub active: bool,
    pub session_id: Option<String>,
    pub token_count: Option<u64>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RuntimeSessionStats {
    pub lane_count: usize,
    pub active_sessions: usize,
    pub idle_sessions: usize,
    pub idle_resident_prefixes: usize,
    pub tracked_token_counts: usize,
    pub max_session_tokens: u64,
    pub total_session_tokens: u64,
    pub checkpoints: usize,
    pub lanes: Vec<RuntimeSessionLaneStats>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct RuntimeSessionDropStats {
    pub reset_session: bool,
    pub reset_ms: f64,
    pub preserved_resident_prefix: bool,
    /// True when the lane could not be returned to the idle pool because
    /// the underlying StageSession failed to reset cleanly. The lane is
    /// dropped (which invokes the C-side skippy_session_free) and the
    /// pool capacity is restored on the next prewarm/admission cycle.
    pub lane_discarded: bool,
    /// Reset-error detail, when [`Self::lane_discarded`] is true.
    pub lane_discard_reason: Option<String>,
    pub stats_after: RuntimeSessionStats,
}

#[derive(Debug, Clone)]
struct ResidentLanePrefix {
    page_id: String,
    token_count: u64,
}

impl RuntimeState {
    pub fn prefill(&mut self, session_id: &str, token_ids: &[i32]) -> Result<()> {
        let session = self.session(session_id)?;
        session.prefill_chunked(token_ids)?;
        self.add_session_tokens(session_id, token_ids.len() as u64);
        Ok(())
    }

    pub fn media_marker(&self) -> String {
        self.model.media_marker()
    }

    pub fn has_media_projector(&self) -> bool {
        self.model.has_media_projector()
    }

    pub fn prefill_media(
        &mut self,
        session_id: &str,
        prompt: &str,
        media: &[MediaInput],
        sampling: Option<&SamplingConfig>,
    ) -> Result<MediaPrefill> {
        let model = &self.model as *const StageModel;
        let session = self.session(session_id)?;
        // `session()` mutably borrows the session map, while the projector lives
        // on the same RuntimeState. RuntimeState serializes access behind one
        // outer mutex, so this split borrow only aliases immutable model state.
        let prefill = unsafe { (&*model).prefill_media(session, prompt, media, sampling) }?;
        self.session_token_counts
            .insert(session_id.to_string(), prefill.position);
        Ok(prefill)
    }

    pub fn prefill_media_frame(
        &mut self,
        session_id: &str,
        prompt: &str,
        media: &[MediaInput],
    ) -> Result<MediaPrefillFrame> {
        let model = &self.model as *const StageModel;
        let session = self.session(session_id)?;
        // `session()` mutably borrows the session map, while the projector lives
        // on the same RuntimeState. RuntimeState serializes access behind one
        // outer mutex, so this split borrow only aliases immutable model state.
        let prefill = unsafe { (&*model).prefill_media_frame(session, prompt, media) }?;
        self.session_token_counts
            .insert(session_id.to_string(), prefill.position);
        Ok(prefill)
    }

    pub fn decode(&mut self, session_id: &str, token_id: i32) -> Result<i32> {
        self.decode_sampled(session_id, token_id, None)
    }

    pub fn decode_sampled(
        &mut self,
        session_id: &str,
        token_id: i32,
        sampling: Option<&SamplingConfig>,
    ) -> Result<i32> {
        let session = self.session(session_id)?;
        let token = session.decode_step_sampled(token_id, sampling)?;
        self.add_session_tokens(session_id, 1);
        Ok(token)
    }

    pub fn configure_chat_sampling(
        &mut self,
        session_id: &str,
        metadata_json: &str,
        prompt_token_count: u64,
        sampling: Option<&SamplingConfig>,
    ) -> Result<()> {
        self.session(session_id)?.configure_chat_sampling(
            metadata_json,
            prompt_token_count,
            sampling,
        )
    }

    pub fn last_token_signal(&mut self, session_id: &str) -> Result<TokenSignal> {
        self.session(session_id)?.last_token_signal()
    }

    pub fn signal_window(
        &mut self,
        session_id: &str,
        window_tokens: u32,
    ) -> Result<GenerationSignalWindow> {
        self.session(session_id)?.signal_window(window_tokens)
    }

    pub fn prefill_frame(
        &mut self,
        session_id: &str,
        token_ids: &[i32],
        input: Option<&ActivationFrame>,
    ) -> Result<ActivationFrame> {
        self.prefill_frame_with_positions(session_id, token_ids, &[], input)
    }

    pub fn prefill_frame_with_positions(
        &mut self,
        session_id: &str,
        token_ids: &[i32],
        positions: &[i32],
        input: Option<&ActivationFrame>,
    ) -> Result<ActivationFrame> {
        let session = self.session(session_id)?;
        let frame = session.prefill_chunk_frame_with_positions(token_ids, positions, input, 0)?;
        self.add_session_tokens(session_id, token_ids.len() as u64);
        Ok(frame)
    }

    pub fn prefill_final_frame_sampled(
        &mut self,
        session_id: &str,
        token_ids: &[i32],
        positions: &[i32],
        sampling: Option<&SamplingConfig>,
        input: Option<&ActivationFrame>,
    ) -> Result<(i32, ActivationFrame)> {
        let session = self.session(session_id)?;
        let (predicted, frame) = session
            .prefill_chunk_frame_sampled_with_positions(token_ids, positions, sampling, input, 0)?;
        self.add_session_tokens(session_id, token_ids.len() as u64);
        Ok((predicted, frame))
    }

    #[allow(dead_code)]
    pub fn decode_frame(
        &mut self,
        session_id: &str,
        token_id: i32,
        input: Option<&ActivationFrame>,
    ) -> Result<(i32, ActivationFrame)> {
        self.decode_frame_sampled(session_id, token_id, None, input)
    }

    pub fn decode_frame_sampled(
        &mut self,
        session_id: &str,
        token_id: i32,
        sampling: Option<&SamplingConfig>,
        input: Option<&ActivationFrame>,
    ) -> Result<(i32, ActivationFrame)> {
        let session = self.session(session_id)?;
        let output = session.decode_step_frame_sampled(token_id, sampling, input, 0)?;
        self.add_session_tokens(session_id, 1);
        Ok(output)
    }

    pub fn verify_frame(
        &mut self,
        session_id: &str,
        token_ids: &[i32],
        input: Option<&ActivationFrame>,
    ) -> Result<(Vec<i32>, ActivationFrame)> {
        let session = self.session(session_id)?;
        let output = session.verify_tokens_frame(token_ids, input, 0)?;
        self.add_session_tokens(session_id, token_ids.len() as u64);
        Ok(output)
    }

    pub fn checkpoint_session(&mut self, session_id: &str) -> Result<()> {
        let checkpoint = self.session(session_id)?.checkpoint()?;
        self.session_checkpoints
            .insert(session_id.to_string(), checkpoint);
        Ok(())
    }

    pub fn restore_session(&mut self, session_id: &str) -> Result<()> {
        let checkpoint = self
            .session_checkpoints
            .get(session_id)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("missing checkpoint for session {session_id}"))?;
        let token_count = {
            let session = self.session(session_id)?;
            session.restore_checkpoint(&checkpoint)?;
            session.token_count()
        };
        self.session_token_counts
            .insert(session_id.to_string(), token_count);
        Ok(())
    }

    pub fn trim_session(&mut self, session_id: &str, token_count: u64) -> Result<()> {
        let session = self.session(session_id)?;
        session.trim_session(token_count)?;
        self.session_token_counts
            .insert(session_id.to_string(), token_count);
        Ok(())
    }

    fn session(&mut self, session_id: &str) -> Result<&mut StageSession> {
        if !self.sessions.contains_key(session_id) {
            let lane_session = self.take_idle_session().map(Ok).unwrap_or_else(|| {
                if self.sessions.len() >= self.lane_count as usize {
                    bail!("all execution lanes are busy");
                }
                self.create_lane_session()
            })?;
            self.sessions.insert(session_id.to_string(), lane_session);
        }
        Ok(&mut self
            .sessions
            .get_mut(session_id)
            .expect("session inserted above")
            .session)
    }

    pub fn prewarm_idle_sessions(
        &mut self,
        target_idle_sessions: usize,
    ) -> Result<RuntimeSessionStats> {
        while self.idle_sessions.len() < target_idle_sessions {
            if self.sessions.len() + self.idle_sessions.len() >= self.lane_count as usize {
                break;
            }
            let lane_session = self.create_lane_session()?;
            self.idle_sessions.push(lane_session);
        }
        Ok(self.session_stats())
    }

    /// Release the session slot identified by `session_id`.
    ///
    /// This is the cleanup path called at the end of every chat
    /// completion (success, cancellation, or backend error). It must
    /// leave [`Self`] in a self-consistent state regardless of whether
    /// the underlying StageSession can be reset cleanly:
    ///
    ///  - The lane is either returned to `idle_sessions` (reset OK) or
    ///    dropped entirely (reset failed). Dropping the lane triggers
    ///    `StageSession::drop`, which calls `skippy_session_free` on
    ///    the C side — the authoritative path for releasing native KV
    ///    cells held by that sequence id.
    ///  - `session_token_counts`, `session_checkpoints`, and
    ///    `session_resident_prefixes` for `session_id` are always
    ///    removed.
    ///  - The function always returns `Ok` so per-request cleanup at
    ///    callsites never propagates a reset failure as a request
    ///    error. The outcome is reported via [`RuntimeSessionDropStats`]
    ///    fields (`lane_discarded`, `lane_discard_reason`) for
    ///    telemetry.
    ///
    /// Previously a reset error propagated `?` through this function,
    /// which left `session_token_counts` and `session_checkpoints`
    /// holding stale entries and dropped the lane on the floor without
    /// any record. That accumulated bookkeeping drift over time and
    /// could leave the native KV cache reporting "all slots in use"
    /// long after the owning sessions were gone, producing
    /// `failed to find a memory slot` errors on subsequent admissions.
    pub fn drop_session_timed(&mut self, session_id: &str) -> Result<RuntimeSessionDropStats> {
        let reset_started = Instant::now();
        let mut reset_session = false;
        let mut preserved_resident_prefix = false;
        let mut lane_discarded = false;
        let mut lane_discard_reason: Option<String> = None;

        if let Some(mut lane_session) = self.sessions.remove(session_id) {
            let lane_index = lane_session.index;
            if let Some(prefix) = self.session_resident_prefixes.remove(session_id) {
                match lane_session.session.trim_session(prefix.token_count) {
                    Ok(()) => {
                        preserved_resident_prefix = true;
                        lane_session.resident_prefix = Some(prefix);
                        self.idle_sessions.push(lane_session);
                    }
                    Err(trim_err) => {
                        reset_session = true;
                        match lane_session.session.reset() {
                            Ok(()) => {
                                lane_session.resident_prefix = None;
                                self.idle_sessions.push(lane_session);
                            }
                            Err(reset_err) => {
                                lane_discarded = true;
                                let reason = format!(
                                    "trim_session({}) failed ({trim_err:#}); fallback reset() also failed ({reset_err:#})",
                                    prefix.token_count
                                );
                                eprintln!(
                                    "skippy::runtime_state: drop_session_timed: discarding lane {lane_index} for session {session_id}: {reason}"
                                );
                                lane_discard_reason = Some(reason);
                                // `lane_session` falls out of scope and its
                                // StageSession::drop runs, which calls
                                // skippy_session_free on the C side. After
                                // the native KV cells for this seq_id are
                                // released, return the lane index to the
                                // free list so the next allocation can
                                // reuse it instead of consuming a fresh
                                // slot from next_lane_index.
                                drop(lane_session);
                                self.free_lane_indices.push(lane_index);
                            }
                        }
                    }
                }
            } else {
                reset_session = true;
                match lane_session.session.reset() {
                    Ok(()) => {
                        lane_session.resident_prefix = None;
                        self.idle_sessions.push(lane_session);
                    }
                    Err(reset_err) => {
                        lane_discarded = true;
                        let reason = format!("reset() failed ({reset_err:#})");
                        eprintln!(
                            "skippy::runtime_state: drop_session_timed: discarding lane {lane_index} for session {session_id}: {reason}"
                        );
                        lane_discard_reason = Some(reason);
                        // See sibling branch above: free the lane index
                        // after the StageSession drop releases the
                        // native KV cells for this seq_id.
                        drop(lane_session);
                        self.free_lane_indices.push(lane_index);
                    }
                }
            }
        }

        // Always clear per-session bookkeeping. The previous version
        // skipped these when reset returned Err, which leaked entries.
        //
        // session_resident_prefixes is also cleared here defensively:
        // it's already removed above on the active-session path, but
        // calling drop_session_timed for an id that's no longer in
        // `sessions` (idempotent cleanup, stale callers) must still
        // clear any stray resident-prefix entry under that id.
        self.session_token_counts.remove(session_id);
        self.session_checkpoints.remove(session_id);
        self.session_resident_prefixes.remove(session_id);

        Ok(RuntimeSessionDropStats {
            reset_session,
            reset_ms: reset_started.elapsed().as_secs_f64() * 1000.0,
            preserved_resident_prefix,
            lane_discarded,
            lane_discard_reason,
            stats_after: self.session_stats(),
        })
    }

    pub fn session_stats(&self) -> RuntimeSessionStats {
        let mut max_session_tokens = 0u64;
        let mut total_session_tokens = 0u64;
        let mut lanes = (0..self.lane_count as usize)
            .map(|index| RuntimeSessionLaneStats {
                index,
                active: false,
                session_id: None,
                token_count: None,
            })
            .collect::<Vec<_>>();

        for (session_id, lane_session) in &self.sessions {
            if let Some(token_count) = self.session_token_counts.get(session_id).copied() {
                max_session_tokens = max_session_tokens.max(token_count);
                total_session_tokens = total_session_tokens.saturating_add(token_count);
            }
            if let Some(lane) = lanes.get_mut(lane_session.index) {
                lane.active = true;
                lane.session_id = Some(session_id.clone());
                lane.token_count = self.session_token_counts.get(session_id).copied();
            }
        }

        RuntimeSessionStats {
            lane_count: self.lane_count as usize,
            active_sessions: self.sessions.len(),
            idle_sessions: self.idle_sessions.len(),
            idle_resident_prefixes: self
                .idle_sessions
                .iter()
                .filter(|idle| idle.resident_prefix.is_some())
                .count(),
            tracked_token_counts: self.session_token_counts.len(),
            max_session_tokens,
            total_session_tokens,
            checkpoints: self.session_checkpoints.len(),
            lanes,
        }
    }

    fn take_idle_session(&mut self) -> Option<RuntimeLaneSession> {
        if let Some(index) = self
            .idle_sessions
            .iter()
            .position(|idle| idle.resident_prefix.is_none())
        {
            return Some(self.idle_sessions.swap_remove(index));
        }
        self.idle_sessions.pop()
    }

    pub fn retain_resident_prefix_on_drop(
        &mut self,
        session_id: &str,
        page_id: String,
        token_count: u64,
    ) -> Result<()> {
        if !self.sessions.contains_key(session_id) {
            bail!("session {session_id} does not exist");
        }
        if self
            .session_resident_prefixes
            .get(session_id)
            .is_some_and(|current| current.token_count >= token_count)
        {
            return Ok(());
        }
        self.session_resident_prefixes.insert(
            session_id.to_string(),
            ResidentLanePrefix {
                page_id,
                token_count,
            },
        );
        Ok(())
    }

    pub fn acquire_resident_prefix_lane(
        &mut self,
        session_id: &str,
        page_id: &str,
        token_count: u64,
    ) -> Result<bool> {
        if self.sessions.contains_key(session_id) {
            bail!("session {session_id} already exists");
        }
        let Some(index) = self.idle_sessions.iter().position(|idle| {
            idle.resident_prefix.as_ref().is_some_and(|prefix| {
                prefix.page_id == page_id && prefix.token_count == token_count
            })
        }) else {
            return Ok(false);
        };
        let mut idle = self.idle_sessions.swap_remove(index);
        idle.resident_prefix = None;
        self.sessions.insert(session_id.to_string(), idle);
        self.session_token_counts
            .insert(session_id.to_string(), token_count);
        self.session_resident_prefixes.insert(
            session_id.to_string(),
            ResidentLanePrefix {
                page_id: page_id.to_string(),
                token_count,
            },
        );
        Ok(true)
    }

    pub fn has_session_range(&self, session_id: &str, token_start: u64, token_count: u64) -> bool {
        let Some(token_end) = token_start.checked_add(token_count) else {
            return false;
        };
        self.session_token_counts
            .get(session_id)
            .copied()
            .is_some_and(|known_tokens| token_end <= known_tokens)
    }

    #[allow(dead_code)]
    pub fn export_kv_page(
        &mut self,
        session_id: &str,
        token_start: u64,
        token_count: u64,
    ) -> Result<RuntimeKvPage> {
        self.validate_export_range(session_id, token_start, token_count)?;
        let layer_start = i32::try_from(self.model_layer_start())?;
        let layer_end = i32::try_from(self.model_layer_end())?;
        let session = self.session(session_id)?;
        session.export_kv_page(layer_start, layer_end, token_start, token_count)
    }

    #[allow(dead_code)]
    pub fn probe_kv_page(
        &mut self,
        session_id: &str,
        token_start: u64,
        token_count: u64,
    ) -> Result<RuntimeKvPageDesc> {
        self.validate_export_range(session_id, token_start, token_count)?;
        let layer_start = i32::try_from(self.model_layer_start())?;
        let layer_end = i32::try_from(self.model_layer_end())?;
        let session = self.session(session_id)?;
        let page = session.export_kv_page(layer_start, layer_end, token_start, token_count)?;
        Ok(page.desc)
    }

    pub fn import_kv_page(
        &mut self,
        session_id: &str,
        desc: &RuntimeKvPageDesc,
        bytes: &[u8],
    ) -> Result<()> {
        let session = self.session(session_id)?;
        session.import_kv_page(desc, bytes)?;
        let token_end = desc
            .token_start
            .checked_add(desc.token_count)
            .ok_or_else(|| anyhow::anyhow!("KV page token range overflows"))?;
        self.session_token_counts
            .entry(session_id.to_string())
            .and_modify(|current| *current = (*current).max(token_end))
            .or_insert(token_end);
        Ok(())
    }

    #[allow(dead_code)]
    pub fn export_state(&mut self, session_id: &str) -> Result<Vec<u8>> {
        let layer_start = i32::try_from(self.model_layer_start())?;
        let layer_end = i32::try_from(self.model_layer_end())?;
        let session = self.session(session_id)?;
        session.export_state(layer_start, layer_end)
    }

    pub fn import_state(&mut self, session_id: &str, bytes: &[u8]) -> Result<()> {
        let layer_start = i32::try_from(self.model_layer_start())?;
        let layer_end = i32::try_from(self.model_layer_end())?;
        let session = self.session(session_id)?;
        session.import_state(layer_start, layer_end, bytes)
    }

    pub fn import_state_for_token_count(
        &mut self,
        session_id: &str,
        bytes: &[u8],
        token_count: u64,
    ) -> Result<()> {
        let layer_start = i32::try_from(self.model_layer_start())?;
        let layer_end = i32::try_from(self.model_layer_end())?;
        let session = self.session(session_id)?;
        session.import_state_for_token_count(layer_start, layer_end, bytes, token_count)?;
        self.session_token_counts
            .entry(session_id.to_string())
            .and_modify(|current| *current = (*current).max(token_count))
            .or_insert(token_count);
        Ok(())
    }

    pub fn export_full_state(&mut self, session_id: &str) -> Result<Vec<u8>> {
        let layer_start = i32::try_from(self.model_layer_start())?;
        let layer_end = i32::try_from(self.model_layer_end())?;
        let session = self.session(session_id)?;
        session.export_full_state(layer_start, layer_end)
    }

    pub fn import_full_state(&mut self, session_id: &str, bytes: &[u8]) -> Result<()> {
        let layer_start = i32::try_from(self.model_layer_start())?;
        let layer_end = i32::try_from(self.model_layer_end())?;
        let session = self.session(session_id)?;
        session.import_full_state(layer_start, layer_end, bytes)
    }

    pub fn import_full_state_for_token_count(
        &mut self,
        session_id: &str,
        bytes: &[u8],
        token_count: u64,
    ) -> Result<()> {
        let layer_start = i32::try_from(self.model_layer_start())?;
        let layer_end = i32::try_from(self.model_layer_end())?;
        let session = self.session(session_id)?;
        session.import_full_state_for_token_count(layer_start, layer_end, bytes, token_count)?;
        self.session_token_counts
            .entry(session_id.to_string())
            .and_modify(|current| *current = (*current).max(token_count))
            .or_insert(token_count);
        Ok(())
    }

    pub fn export_recurrent_state(&mut self, session_id: &str) -> Result<Vec<u8>> {
        self.session(session_id)?.export_recurrent_state()
    }

    pub fn import_recurrent_state_for_token_count(
        &mut self,
        session_id: &str,
        bytes: &[u8],
        token_count: u64,
    ) -> Result<()> {
        self.session(session_id)?
            .import_recurrent_state_for_token_count(bytes, token_count)?;
        self.session_token_counts
            .entry(session_id.to_string())
            .and_modify(|current| *current = (*current).max(token_count))
            .or_insert(token_count);
        Ok(())
    }

    pub fn save_resident_prefix(
        &mut self,
        session_id: &str,
        cache_seq_id: i32,
        token_count: u64,
    ) -> Result<()> {
        self.session(session_id)?
            .save_prefix(cache_seq_id, token_count)
    }

    pub fn restore_resident_prefix(
        &mut self,
        session_id: &str,
        cache_seq_id: i32,
        token_ids: &[i32],
    ) -> Result<()> {
        let session = self.session(session_id)?;
        session.restore_prefix(cache_seq_id, token_ids)?;
        self.session_token_counts
            .insert(session_id.to_string(), token_ids.len() as u64);
        Ok(())
    }

    pub fn borrow_resident_prefix_session(
        &mut self,
        session_id: &str,
        cache_seq_id: i32,
        token_ids: &[i32],
    ) -> Result<()> {
        if self.sessions.contains_key(session_id) {
            bail!("session {session_id} already exists");
        }
        let model = &self.model;
        let (index, session) = create_indexed_lane_resource(
            &mut self.next_lane_index,
            &mut self.free_lane_indices,
            self.lane_count,
            || model.create_session_from_resident_prefix(cache_seq_id, token_ids),
        )?;
        let lane_session = RuntimeLaneSession {
            index,
            session,
            resident_prefix: None,
        };
        self.sessions.insert(session_id.to_string(), lane_session);
        self.session_token_counts
            .insert(session_id.to_string(), token_ids.len() as u64);
        Ok(())
    }

    pub fn drop_resident_prefix_sequence(
        &mut self,
        session_id: &str,
        cache_seq_id: i32,
    ) -> Result<()> {
        self.session(session_id)?.drop_sequence(cache_seq_id)
    }

    fn add_session_tokens(&mut self, session_id: &str, count: u64) {
        self.session_token_counts
            .entry(session_id.to_string())
            .and_modify(|current| *current = current.saturating_add(count))
            .or_insert(count);
    }

    fn validate_export_range(
        &self,
        session_id: &str,
        token_start: u64,
        token_count: u64,
    ) -> Result<()> {
        let token_end = token_start
            .checked_add(token_count)
            .ok_or_else(|| anyhow::anyhow!("KV page token range overflows"))?;
        let known_tokens = self
            .session_token_counts
            .get(session_id)
            .copied()
            .unwrap_or_default();
        if token_end > known_tokens {
            bail!(
                "cannot export KV page [{token_start}, {token_end}) from session with {known_tokens} known tokens"
            );
        }
        Ok(())
    }

    fn model_layer_start(&self) -> u32 {
        self.layer_start
    }

    fn model_layer_end(&self) -> u32 {
        self.layer_end
    }

    fn create_lane_session(&mut self) -> Result<RuntimeLaneSession> {
        let model = &self.model;
        let (index, session) = create_indexed_lane_resource(
            &mut self.next_lane_index,
            &mut self.free_lane_indices,
            self.lane_count,
            || model.create_session(),
        )?;
        Ok(RuntimeLaneSession {
            index,
            session,
            resident_prefix: None,
        })
    }
}

/// Allocate the next lane slot.
///
/// Prefers indices in `free_lane_indices` (lanes previously discarded
/// via [`RuntimeState::drop_session_timed`]) so they can be reused
/// without growing `next_lane_index` past `lane_count`. If the free
/// list is empty, falls through to bumping `next_lane_index`. If both
/// are exhausted, returns "all execution lanes are busy".
///
/// If `create()` fails after popping from the free list, the index is
/// pushed back so a retry can reuse it. The high-water counter is only
/// bumped on success, matching the prior behavior.
fn create_indexed_lane_resource<T>(
    next_lane_index: &mut usize,
    free_lane_indices: &mut Vec<usize>,
    lane_count: u32,
    create: impl FnOnce() -> Result<T>,
) -> Result<(usize, T)> {
    if let Some(index) = free_lane_indices.pop() {
        let resource = match create() {
            Ok(resource) => resource,
            Err(err) => {
                // Return the freed index so the next allocation can
                // still reuse it.
                free_lane_indices.push(index);
                return Err(err);
            }
        };
        return Ok((index, resource));
    }
    if *next_lane_index >= lane_count as usize {
        bail!("all execution lanes are busy");
    }
    let index = *next_lane_index;
    let resource = create()?;
    *next_lane_index = index + 1;
    Ok((index, resource))
}

impl Drop for RuntimeState {
    fn drop(&mut self) {
        self.sessions.clear();
        self.idle_sessions.clear();
    }
}

pub fn load_runtime(config: &StageConfig) -> Result<Option<Arc<Mutex<RuntimeState>>>> {
    let mut runtime_config = runtime_config_from_stage_config(config)?;

    let model = match config.load_mode {
        _ if std::env::var("MESH_LLM_BYPASS_SKIPPY_MODEL_LOAD").is_ok() => {
            skippy_runtime::StageModel::new_dummy()
        }
        LoadMode::LayerPackage => {
            let selected =
                select_package_parts(config).context("select layer package parts for stage")?;
            if runtime_config.projector_path.is_none() && should_attach_package_projector(config) {
                runtime_config.projector_path = selected
                    .projector_paths
                    .first()
                    .map(|path| path.to_string_lossy().to_string());
            }
            open_stage_model_from_parts(&selected.absolute_paths, &runtime_config)?
        }
        _ => {
            let Some(model_path) = config.model_path.as_ref().map(std::path::Path::new) else {
                return Ok(None);
            };
            open_stage_model(model_path, &runtime_config)?
        }
    };

    Ok(Some(Arc::new(Mutex::new(RuntimeState {
        model,
        layer_start: config.layer_start,
        layer_end: config.layer_end,
        lane_count: config.lane_count,
        next_lane_index: 0,
        free_lane_indices: Vec::new(),
        sessions: BTreeMap::new(),
        idle_sessions: Vec::new(),
        session_token_counts: BTreeMap::new(),
        session_checkpoints: BTreeMap::new(),
        session_resident_prefixes: BTreeMap::new(),
    }))))
}

fn should_attach_package_projector(config: &StageConfig) -> bool {
    config.stage_index == 0 && config.layer_start == 0
}

fn runtime_config_from_stage_config(config: &StageConfig) -> Result<RuntimeConfig> {
    let cache_type_k = parse_cache_type(&config.cache_type_k)
        .with_context(|| format!("parse cache_type_k for {}", config.stage_id))?;
    let cache_type_v = parse_cache_type(&config.cache_type_v)
        .with_context(|| format!("parse cache_type_v for {}", config.stage_id))?;
    Ok(RuntimeConfig {
        stage_index: config.stage_index,
        layer_start: config.layer_start,
        layer_end: config.layer_end,
        ctx_size: config.ctx_size,
        lane_count: config.lane_count,
        n_batch: config.n_batch,
        n_ubatch: config.n_ubatch,
        n_threads: None,
        n_threads_batch: None,
        n_gpu_layers: config.n_gpu_layers,
        selected_backend_device: config
            .selected_device
            .as_ref()
            .map(|device| device.backend_device.clone()),
        cache_type_k,
        cache_type_v,
        flash_attn_type: match config.flash_attn_type {
            FlashAttentionType::Auto => RuntimeFlashAttentionType::Auto,
            FlashAttentionType::Disabled => RuntimeFlashAttentionType::Disabled,
            FlashAttentionType::Enabled => RuntimeFlashAttentionType::Enabled,
        },
        load_mode: match config.load_mode {
            LoadMode::RuntimeSlice => RuntimeLoadMode::RuntimeSlice,
            LoadMode::LayerPackage => RuntimeLoadMode::LayerPackage,
            LoadMode::ArtifactSlice => RuntimeLoadMode::ArtifactSlice,
        },
        projector_path: config.projector_path.clone(),
        include_embeddings: config.layer_start == 0,
        include_output: config.downstream.is_none(),
        filter_tensors_on_load: config.filter_tensors_on_load,
    })
}

fn open_stage_model(path: &std::path::Path, runtime_config: &RuntimeConfig) -> Result<StageModel> {
    StageModel::open(path, runtime_config)
}

fn open_stage_model_from_parts(
    paths: &[std::path::PathBuf],
    runtime_config: &RuntimeConfig,
) -> Result<StageModel> {
    StageModel::open_from_parts(paths, runtime_config)
}

#[cfg(test)]
mod tests {
    use anyhow::{bail, Result};
    use skippy_protocol::{FlashAttentionType, LoadMode, PeerConfig, StageConfig, StageDevice};
    use skippy_runtime::FlashAttentionType as RuntimeFlashAttentionType;

    use super::{
        create_indexed_lane_resource, runtime_config_from_stage_config,
        should_attach_package_projector,
    };

    #[test]
    fn create_indexed_lane_resource_keeps_index_available_when_creation_fails() {
        let mut next_lane_index = 0;
        let mut free_lane_indices: Vec<usize> = Vec::new();

        let error = create_indexed_lane_resource(
            &mut next_lane_index,
            &mut free_lane_indices,
            2,
            || -> Result<()> { bail!("transient session creation failure") },
        )
        .expect_err("failed creation should propagate the original error");

        assert_eq!(error.to_string(), "transient session creation failure");
        assert_eq!(next_lane_index, 0);
        assert!(free_lane_indices.is_empty());

        let (index, resource) =
            create_indexed_lane_resource(&mut next_lane_index, &mut free_lane_indices, 2, || {
                Ok("lane")
            })
            .expect("successful retry should reuse the unconsumed lane index");

        assert_eq!(index, 0);
        assert_eq!(resource, "lane");
        assert_eq!(next_lane_index, 1);
    }

    #[test]
    fn create_indexed_lane_resource_reuses_freed_indices_before_growing() {
        // Simulate the wedge scenario: all lanes allocated, one lane
        // freed via the discard path, next allocation must reuse the
        // freed index rather than bailing with "all execution lanes
        // are busy".
        let mut next_lane_index = 0;
        let mut free_lane_indices: Vec<usize> = Vec::new();
        let lane_count = 2;

        // Allocate both lanes.
        let (a_idx, _) = create_indexed_lane_resource(
            &mut next_lane_index,
            &mut free_lane_indices,
            lane_count,
            || Ok("a"),
        )
        .expect("first allocation should succeed");
        let (b_idx, _) = create_indexed_lane_resource(
            &mut next_lane_index,
            &mut free_lane_indices,
            lane_count,
            || Ok("b"),
        )
        .expect("second allocation should succeed");
        assert_eq!(a_idx, 0);
        assert_eq!(b_idx, 1);
        assert_eq!(next_lane_index, 2);

        // Pool is full at the high-water mark. A third allocation must
        // fail.
        let error = create_indexed_lane_resource(
            &mut next_lane_index,
            &mut free_lane_indices,
            lane_count,
            || Ok("c"),
        )
        .expect_err("allocating past lane_count should fail when no slots are free");
        assert!(error.to_string().contains("all execution lanes are busy"));

        // Discard one lane: the caller pushes its freed index onto the
        // free list (this is what drop_session_timed does on the
        // discard branch).
        free_lane_indices.push(a_idx);

        // The next allocation MUST reuse the freed index instead of
        // bailing. This is the wedge regression: previously
        // next_lane_index stayed at lane_count and every allocation
        // failed forever.
        let (reused_idx, _) = create_indexed_lane_resource(
            &mut next_lane_index,
            &mut free_lane_indices,
            lane_count,
            || Ok("c"),
        )
        .expect("allocation must reuse a freed index, not stay wedged");
        assert_eq!(reused_idx, 0);
        assert_eq!(next_lane_index, 2);
        assert!(free_lane_indices.is_empty());
    }

    #[test]
    fn create_indexed_lane_resource_returns_freed_index_on_create_failure() {
        // If create() fails while consuming a freed index, the index
        // must go back onto the free list so a retry can use it.
        let mut next_lane_index = 1;
        let mut free_lane_indices: Vec<usize> = vec![0];

        let error = create_indexed_lane_resource(
            &mut next_lane_index,
            &mut free_lane_indices,
            2,
            || -> Result<()> { bail!("create failed mid-reuse") },
        )
        .expect_err("failed creation should propagate");
        assert_eq!(error.to_string(), "create failed mid-reuse");
        assert_eq!(next_lane_index, 1);
        assert_eq!(free_lane_indices, vec![0]);

        // A retry should now succeed using the same freed index.
        let (idx, _) =
            create_indexed_lane_resource(&mut next_lane_index, &mut free_lane_indices, 2, || {
                Ok("retry")
            })
            .expect("retry should succeed");
        assert_eq!(idx, 0);
        assert_eq!(next_lane_index, 1);
        assert!(free_lane_indices.is_empty());
    }

    #[test]
    fn runtime_config_preserves_selected_backend_device() {
        let config = StageConfig {
            run_id: "run-a".to_string(),
            topology_id: "topology-a".to_string(),
            model_id: "model-a".to_string(),
            package_ref: None,
            manifest_sha256: None,
            source_model_path: None,
            source_model_sha256: None,
            source_model_bytes: None,
            materialized_path: None,
            materialized_pinned: false,
            model_path: Some("/tmp/model.gguf".to_string()),
            projector_path: Some("/tmp/mmproj.gguf".to_string()),
            stage_id: "stage-0".to_string(),
            stage_index: 0,
            layer_start: 0,
            layer_end: 24,
            ctx_size: 512,
            lane_count: 2,
            n_batch: Some(1024),
            n_ubatch: Some(256),
            n_gpu_layers: -1,
            cache_type_k: "f16".to_string(),
            cache_type_v: "f16".to_string(),
            flash_attn_type: FlashAttentionType::Enabled,
            filter_tensors_on_load: true,
            selected_device: Some(StageDevice {
                backend_device: "Vulkan1".into(),
                stable_id: Some("pci:0000:65:00.0".into()),
                index: Some(1),
                vram_bytes: Some(16_000_000_000),
            }),
            kv_cache: None,
            load_mode: LoadMode::RuntimeSlice,
            bind_addr: "127.0.0.1:0".to_string(),
            upstream: None,
            downstream: None,
        };

        let runtime_config = runtime_config_from_stage_config(&config).unwrap();

        assert_eq!(
            runtime_config.selected_backend_device.as_deref(),
            Some("Vulkan1")
        );
        assert_eq!(runtime_config.lane_count, 2);
        assert_eq!(runtime_config.n_batch, Some(1024));
        assert_eq!(runtime_config.n_ubatch, Some(256));
        assert_eq!(
            runtime_config.flash_attn_type,
            RuntimeFlashAttentionType::Enabled
        );
    }

    #[test]
    fn runtime_config_omits_input_embeddings_for_final_non_first_stage() {
        let config = StageConfig {
            run_id: "run-a".to_string(),
            topology_id: "topology-a".to_string(),
            model_id: "model-a".to_string(),
            package_ref: Some("/tmp/package".to_string()),
            manifest_sha256: Some("manifest".to_string()),
            source_model_path: None,
            source_model_sha256: None,
            source_model_bytes: None,
            materialized_path: None,
            materialized_pinned: false,
            model_path: Some("/tmp/package".to_string()),
            projector_path: None,
            stage_id: "stage-2".to_string(),
            stage_index: 2,
            layer_start: 20,
            layer_end: 30,
            ctx_size: 512,
            lane_count: 1,
            n_batch: None,
            n_ubatch: None,
            n_gpu_layers: -1,
            cache_type_k: "f16".to_string(),
            cache_type_v: "f16".to_string(),
            flash_attn_type: FlashAttentionType::Auto,
            filter_tensors_on_load: true,
            selected_device: None,
            kv_cache: None,
            load_mode: LoadMode::LayerPackage,
            bind_addr: "127.0.0.1:0".to_string(),
            upstream: Some(PeerConfig {
                stage_id: "stage-1".to_string(),
                stage_index: 1,
                endpoint: "tcp://127.0.0.1:19001".to_string(),
            }),
            downstream: None,
        };

        let runtime_config = runtime_config_from_stage_config(&config).unwrap();

        assert!(!runtime_config.include_embeddings);
        assert!(runtime_config.include_output);
    }

    #[test]
    fn package_projector_fallback_is_stage_zero_only() {
        let mut config = StageConfig {
            run_id: "run-a".to_string(),
            topology_id: "topology-a".to_string(),
            model_id: "model-a".to_string(),
            package_ref: Some("/tmp/package".to_string()),
            manifest_sha256: Some("manifest".to_string()),
            source_model_path: None,
            source_model_sha256: None,
            source_model_bytes: None,
            materialized_path: None,
            materialized_pinned: false,
            model_path: Some("/tmp/package".to_string()),
            projector_path: None,
            stage_id: "stage-0".to_string(),
            stage_index: 0,
            layer_start: 0,
            layer_end: 10,
            ctx_size: 512,
            lane_count: 1,
            n_batch: None,
            n_ubatch: None,
            n_gpu_layers: -1,
            cache_type_k: "f16".to_string(),
            cache_type_v: "f16".to_string(),
            flash_attn_type: FlashAttentionType::Auto,
            filter_tensors_on_load: true,
            selected_device: None,
            kv_cache: None,
            load_mode: LoadMode::LayerPackage,
            bind_addr: "127.0.0.1:0".to_string(),
            upstream: None,
            downstream: Some(PeerConfig {
                stage_id: "stage-1".to_string(),
                stage_index: 1,
                endpoint: "tcp://127.0.0.1:19001".to_string(),
            }),
        };

        assert!(should_attach_package_projector(&config));

        config.stage_id = "stage-1".to_string();
        config.stage_index = 1;
        config.layer_start = 10;
        config.layer_end = 20;
        config.upstream = Some(PeerConfig {
            stage_id: "stage-0".to_string(),
            stage_index: 0,
            endpoint: "tcp://127.0.0.1:19000".to_string(),
        });
        config.downstream = None;

        assert!(!should_attach_package_projector(&config));
    }
}
