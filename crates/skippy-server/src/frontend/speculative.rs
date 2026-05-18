use super::*;

#[derive(Default)]
pub(super) struct OpenAiSpeculativeStats {
    pub(super) windows: usize,
    pub(super) draft_tokens: usize,
    pub(super) accepted_tokens: usize,
    pub(super) rejected_tokens: usize,
    pub(super) full_accept_windows: usize,
    pub(super) accepted_stop_windows: usize,
    pub(super) rejected_windows: usize,
    pub(super) early_reject_windows: usize,
    pub(super) tail_reject_windows: usize,
    pub(super) early_reject_stop_windows: usize,
    pub(super) repair_required_windows: usize,
    pub(super) first_reject_position_sum: usize,
    pub(super) primary_verify_requests: usize,
    pub(super) primary_verify_tokens: usize,
    pub(super) primary_verify_elapsed_ms: f64,
    pub(super) primary_verify_stage0_compute_ms: f64,
    pub(super) primary_verify_runtime_lock_wait_ms: f64,
    pub(super) primary_verify_runtime_lock_hold_ms: f64,
    pub(super) primary_verify_activation_encode_ms: f64,
    pub(super) primary_verify_forward_write_ms: f64,
    pub(super) primary_verify_downstream_wait_ms: f64,
    pub(super) primary_verify_output_activation_bytes: usize,
    pub(super) primary_verify_forward_activation_bytes: usize,
    pub(super) checkpoint_ms: f64,
    pub(super) draft_reset_ms: f64,
    pub(super) draft_propose_ms: f64,
    pub(super) mtp_propose_ms: f64,
    pub(super) mtp_ingest_ms: f64,
    pub(super) recovery_restores: usize,
    pub(super) recovery_decode_repairs: usize,
    pub(super) recovery_decode_elapsed_ms: f64,
    pub(super) recovery_reverify_tokens: usize,
    pub(super) recovery_ms: f64,
    pub(super) recovery_restore_ms: f64,
    pub(super) recovery_restore_local_ms: f64,
    pub(super) recovery_restore_downstream_write_ms: f64,
    pub(super) recovery_restore_downstream_wait_ms: f64,
    pub(super) recovery_reverify_elapsed_ms: f64,
    pub(super) adaptive_window_start: usize,
    pub(super) adaptive_window_final: usize,
    pub(super) adaptive_window_max: usize,
    pub(super) adaptive_window_min: usize,
    pub(super) adaptive_window_max_seen: usize,
    pub(super) adaptive_window_sum: usize,
    pub(super) adaptive_window_grows: usize,
    pub(super) adaptive_window_shrinks: usize,
    pub(super) adaptive_window_enabled: bool,
    pub(super) fused_target_decode_calls: usize,
    pub(super) fused_target_verify_decode_calls: usize,
    pub(super) fused_target_verify_tokens: usize,
    pub(super) fused_target_repair_decode_calls: usize,
    pub(super) fused_target_repair_tokens: usize,
    pub(super) fused_mtp_ingest_calls: usize,
    pub(super) fused_mtp_ingest_tokens: usize,
    pub(super) fused_mtp_draft_calls: usize,
    pub(super) fused_mtp_draft_decode_calls: usize,
    pub(super) fused_mtp_draft_tokens: usize,
    pub(super) fused_checkpoint_calls: usize,
    pub(super) fused_restore_calls: usize,
}

impl OpenAiSpeculativeStats {
    pub(super) fn observe_mtp_decode_result(&mut self, result: &MtpDecodeResult) {
        self.windows += result.windows as usize;
        self.draft_tokens += result.proposed_tokens as usize;
        self.accepted_tokens += result.accepted_tokens as usize;
        self.rejected_tokens += result.rejected_tokens as usize;
        self.rejected_windows += result.rejected_windows as usize;
        self.full_accept_windows += result.full_accept_windows as usize;
        self.tail_reject_windows += result.tail_reject_windows as usize;
        self.early_reject_windows += result.early_reject_windows as usize;
        self.repair_required_windows += result.repair_required_windows as usize;
        self.primary_verify_requests += result.windows as usize;
        self.primary_verify_tokens += result.proposed_tokens as usize;
        self.draft_propose_ms += result.mtp_propose_ms;
        self.mtp_propose_ms += result.mtp_propose_ms;
        self.mtp_ingest_ms += result.mtp_ingest_ms;
        self.primary_verify_elapsed_ms += result.primary_verify_ms;
        self.checkpoint_ms += result.checkpoint_ms;
        self.recovery_ms += result.repair_ms;
        self.recovery_restore_local_ms += result.restore_ms;
        self.fused_target_decode_calls += result.target_decode_calls as usize;
        self.fused_target_verify_decode_calls += result.target_verify_decode_calls as usize;
        self.fused_target_verify_tokens += result.target_verify_tokens as usize;
        self.fused_target_repair_decode_calls += result.target_repair_decode_calls as usize;
        self.fused_target_repair_tokens += result.target_repair_tokens as usize;
        self.fused_mtp_ingest_calls += result.mtp_ingest_calls as usize;
        self.fused_mtp_ingest_tokens += result.mtp_ingest_tokens as usize;
        self.fused_mtp_draft_calls += result.mtp_draft_calls as usize;
        self.fused_mtp_draft_decode_calls += result.mtp_draft_decode_calls as usize;
        self.fused_mtp_draft_tokens += result.mtp_draft_tokens as usize;
        self.fused_checkpoint_calls += result.checkpoint_calls as usize;
        self.fused_restore_calls += result.restore_calls as usize;
        if result.repair_required_windows > 0 {
            self.recovery_restores += result.repair_required_windows as usize;
        }
    }

    pub(super) fn observe_verify_decision(
        &mut self,
        decision: VerifySpanDecision,
        adaptive_window: &mut usize,
        adaptive_enabled: bool,
        max_speculative_window: usize,
    ) {
        self.accepted_tokens += decision.accepted_before_reject;
        if decision.rejected() {
            self.rejected_tokens += 1;
        }
        self.adaptive_window_sum += *adaptive_window;
        self.adaptive_window_min = nonzero_min(self.adaptive_window_min, *adaptive_window);
        self.adaptive_window_max_seen = self.adaptive_window_max_seen.max(*adaptive_window);
        match decision.kind {
            VerifySpanDecisionKind::FullAccept => {
                self.full_accept_windows += 1;
                self.grow_adaptive_window(
                    adaptive_window,
                    adaptive_enabled,
                    max_speculative_window,
                );
            }
            VerifySpanDecisionKind::AcceptedStop => {
                self.accepted_stop_windows += 1;
            }
            VerifySpanDecisionKind::TailReject => {
                self.observe_reject(decision);
                self.tail_reject_windows += 1;
                self.grow_adaptive_window(
                    adaptive_window,
                    adaptive_enabled,
                    max_speculative_window,
                );
            }
            VerifySpanDecisionKind::EarlyReject => {
                self.observe_reject(decision);
                self.early_reject_windows += 1;
                self.repair_required_windows += 1;
                self.shrink_adaptive_window(adaptive_window, adaptive_enabled, decision);
            }
            VerifySpanDecisionKind::EarlyRejectStop => {
                self.observe_reject(decision);
                self.early_reject_windows += 1;
                self.early_reject_stop_windows += 1;
            }
        }
    }

    pub(super) fn observe_reject(&mut self, decision: VerifySpanDecision) {
        if let Some(repair_input_count) = decision.repair_input_count {
            self.rejected_windows += 1;
            self.first_reject_position_sum += repair_input_count;
        }
    }

    pub(super) fn grow_adaptive_window(
        &mut self,
        adaptive_window: &mut usize,
        adaptive_enabled: bool,
        max_speculative_window: usize,
    ) {
        if adaptive_enabled && *adaptive_window < max_speculative_window {
            *adaptive_window += 1;
            self.adaptive_window_grows += 1;
        }
    }

    pub(super) fn shrink_adaptive_window(
        &mut self,
        adaptive_window: &mut usize,
        adaptive_enabled: bool,
        decision: VerifySpanDecision,
    ) {
        if !adaptive_enabled {
            return;
        }
        let Some(repair_input_count) = decision.repair_input_count else {
            return;
        };
        let next_window = (*adaptive_window)
            .saturating_sub(1)
            .max(repair_input_count)
            .max(1);
        if next_window < *adaptive_window {
            *adaptive_window = next_window;
            self.adaptive_window_shrinks += 1;
        }
    }

    pub(super) fn insert_attrs(&self, attrs: &mut BTreeMap<String, Value>) {
        if self.windows == 0 {
            attrs.insert("llama_stage.spec.enabled".to_string(), json!(false));
            return;
        }
        insert_bool_attrs(attrs, &[("llama_stage.spec.enabled", true)]);
        insert_usize_attrs(
            attrs,
            &[
                ("llama_stage.spec.windows", self.windows),
                ("llama_stage.spec.proposed", self.draft_tokens),
                ("llama_stage.spec.accepted", self.accepted_tokens),
                ("llama_stage.spec.rejected", self.rejected_tokens),
                (
                    "llama_stage.spec.full_accept_windows",
                    self.full_accept_windows,
                ),
                (
                    "llama_stage.spec.accepted_stop_windows",
                    self.accepted_stop_windows,
                ),
                ("llama_stage.spec.rejected_windows", self.rejected_windows),
                (
                    "llama_stage.spec.early_reject_windows",
                    self.early_reject_windows,
                ),
                (
                    "llama_stage.spec.tail_reject_windows",
                    self.tail_reject_windows,
                ),
                (
                    "llama_stage.spec.repair_required_windows",
                    self.repair_required_windows,
                ),
                ("llama_stage.spec.recovery_restores", self.recovery_restores),
                ("llama_stage.spec.window_start", self.adaptive_window_start),
                ("llama_stage.spec.window_final", self.adaptive_window_final),
                ("llama_stage.spec.window_max", self.adaptive_window_max),
                ("llama_stage.spec.window_min", self.adaptive_window_min),
                (
                    "llama_stage.spec.window_max_seen",
                    self.adaptive_window_max_seen,
                ),
                ("llama_stage.spec.window_grows", self.adaptive_window_grows),
                (
                    "llama_stage.spec.window_shrinks",
                    self.adaptive_window_shrinks,
                ),
                (
                    "llama_stage.spec.fused_target_decode_calls",
                    self.fused_target_decode_calls,
                ),
                (
                    "llama_stage.spec.primary_verify_output_activation_bytes",
                    self.primary_verify_output_activation_bytes,
                ),
                (
                    "llama_stage.spec.primary_verify_forward_activation_bytes",
                    self.primary_verify_forward_activation_bytes,
                ),
                (
                    "llama_stage.spec.fused_target_verify_decode_calls",
                    self.fused_target_verify_decode_calls,
                ),
                (
                    "llama_stage.spec.fused_target_verify_tokens",
                    self.fused_target_verify_tokens,
                ),
                (
                    "llama_stage.spec.fused_target_repair_decode_calls",
                    self.fused_target_repair_decode_calls,
                ),
                (
                    "llama_stage.spec.fused_target_repair_tokens",
                    self.fused_target_repair_tokens,
                ),
                (
                    "llama_stage.spec.fused_mtp_ingest_calls",
                    self.fused_mtp_ingest_calls,
                ),
                (
                    "llama_stage.spec.fused_mtp_ingest_tokens",
                    self.fused_mtp_ingest_tokens,
                ),
                (
                    "llama_stage.spec.fused_mtp_draft_calls",
                    self.fused_mtp_draft_calls,
                ),
                (
                    "llama_stage.spec.fused_mtp_draft_decode_calls",
                    self.fused_mtp_draft_decode_calls,
                ),
                (
                    "llama_stage.spec.fused_mtp_draft_tokens",
                    self.fused_mtp_draft_tokens,
                ),
                (
                    "llama_stage.spec.fused_checkpoint_calls",
                    self.fused_checkpoint_calls,
                ),
                (
                    "llama_stage.spec.fused_restore_calls",
                    self.fused_restore_calls,
                ),
            ],
        );
        insert_f64_attrs(
            attrs,
            &[
                ("llama_stage.spec.draft_reset_ms", self.draft_reset_ms),
                ("llama_stage.spec.draft_propose_ms", self.draft_propose_ms),
                ("llama_stage.spec.mtp_propose_ms", self.mtp_propose_ms),
                ("llama_stage.spec.mtp_ingest_ms", self.mtp_ingest_ms),
                (
                    "llama_stage.spec.primary_verify_elapsed_ms",
                    self.primary_verify_elapsed_ms,
                ),
                (
                    "llama_stage.spec.primary_verify_stage0_compute_ms",
                    self.primary_verify_stage0_compute_ms,
                ),
                (
                    "llama_stage.spec.primary_verify_runtime_lock_wait_ms",
                    self.primary_verify_runtime_lock_wait_ms,
                ),
                (
                    "llama_stage.spec.primary_verify_runtime_lock_hold_ms",
                    self.primary_verify_runtime_lock_hold_ms,
                ),
                (
                    "llama_stage.spec.primary_verify_activation_encode_ms",
                    self.primary_verify_activation_encode_ms,
                ),
                (
                    "llama_stage.spec.primary_verify_forward_write_ms",
                    self.primary_verify_forward_write_ms,
                ),
                (
                    "llama_stage.spec.primary_verify_downstream_wait_ms",
                    self.primary_verify_downstream_wait_ms,
                ),
                ("llama_stage.spec.checkpoint_ms", self.checkpoint_ms),
                ("llama_stage.spec.recovery_ms", self.recovery_ms),
                (
                    "llama_stage.spec.recovery_restore_local_ms",
                    self.recovery_restore_local_ms,
                ),
                (
                    "llama_stage.spec.recovery_restore_downstream_write_ms",
                    self.recovery_restore_downstream_write_ms,
                ),
                (
                    "llama_stage.spec.recovery_restore_downstream_wait_ms",
                    self.recovery_restore_downstream_wait_ms,
                ),
            ],
        );
        insert_bool_attrs(
            attrs,
            &[(
                "llama_stage.spec.adaptive_enabled",
                self.adaptive_window_enabled,
            )],
        );
        attrs.insert(
            "llama_stage.spec.accept_rate".to_string(),
            json!(if self.draft_tokens == 0 {
                0.0
            } else {
                self.accepted_tokens as f64 / self.draft_tokens as f64
            }),
        );
    }
}

fn insert_bool_attrs(attrs: &mut BTreeMap<String, Value>, values: &[(&str, bool)]) {
    for (key, value) in values {
        attrs.insert((*key).to_string(), json!(*value));
    }
}

fn insert_usize_attrs(attrs: &mut BTreeMap<String, Value>, values: &[(&str, usize)]) {
    for (key, value) in values {
        attrs.insert((*key).to_string(), json!(*value));
    }
}

fn insert_f64_attrs(attrs: &mut BTreeMap<String, Value>, values: &[(&str, f64)]) {
    for (key, value) in values {
        attrs.insert((*key).to_string(), json!(*value));
    }
}

pub(super) fn verify_inputs_for_proposals(current: i32, proposals: &[i32]) -> Vec<i32> {
    let mut tokens = Vec::with_capacity(proposals.len());
    if proposals.is_empty() {
        return tokens;
    }
    tokens.push(current);
    tokens.extend(proposals.iter().take(proposals.len().saturating_sub(1)));
    tokens
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum VerifySpanDecisionKind {
    FullAccept,
    AcceptedStop,
    TailReject,
    EarlyReject,
    EarlyRejectStop,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct VerifySpanDecision {
    pub(super) kind: VerifySpanDecisionKind,
    pub(super) accepted_before_reject: usize,
    pub(super) repair_input_count: Option<usize>,
    pub(super) commit_count: usize,
}

impl VerifySpanDecision {
    pub(super) fn rejected(self) -> bool {
        matches!(
            self.kind,
            VerifySpanDecisionKind::TailReject
                | VerifySpanDecisionKind::EarlyReject
                | VerifySpanDecisionKind::EarlyRejectStop
        )
    }

    pub(super) fn requires_repair(self) -> bool {
        self.kind == VerifySpanDecisionKind::EarlyReject
    }
}

pub(super) fn classify_verify_span<F>(
    draft_tokens: &[i32],
    predicted_tokens: &[i32],
    generated_len: usize,
    max_new_tokens: usize,
    mut token_is_eog: F,
) -> OpenAiResult<VerifySpanDecision>
where
    F: FnMut(i32) -> OpenAiResult<bool>,
{
    if predicted_tokens.len() < draft_tokens.len() {
        return Err(OpenAiError::backend(format!(
            "verify span returned too few tokens: got {} expected {}",
            predicted_tokens.len(),
            draft_tokens.len()
        )));
    }

    let mut accepted_before_reject = 0usize;
    let mut commit_count = 0usize;
    for (draft_token, predicted) in draft_tokens.iter().zip(predicted_tokens.iter()) {
        commit_count += 1;
        let accepted = *predicted == *draft_token;
        let reached_eog = token_is_eog(*predicted)?;
        let reached_limit = generated_len + commit_count >= max_new_tokens;
        if accepted {
            accepted_before_reject += 1;
            if (reached_eog || reached_limit) && commit_count < draft_tokens.len() {
                return Ok(VerifySpanDecision {
                    kind: VerifySpanDecisionKind::AcceptedStop,
                    accepted_before_reject,
                    repair_input_count: None,
                    commit_count,
                });
            }
            continue;
        }

        let repair_input_count = accepted_before_reject + 1;
        let kind = if repair_input_count == draft_tokens.len() {
            VerifySpanDecisionKind::TailReject
        } else if reached_eog || reached_limit {
            VerifySpanDecisionKind::EarlyRejectStop
        } else {
            VerifySpanDecisionKind::EarlyReject
        };
        return Ok(VerifySpanDecision {
            kind,
            accepted_before_reject,
            repair_input_count: Some(repair_input_count),
            commit_count,
        });
    }

    Ok(VerifySpanDecision {
        kind: VerifySpanDecisionKind::FullAccept,
        accepted_before_reject,
        repair_input_count: None,
        commit_count,
    })
}

pub(super) fn repaired_commit_tokens(
    draft_tokens: &[i32],
    accepted_before_reject: usize,
    repair_input_count: usize,
    repaired_predictions: &[i32],
) -> OpenAiResult<Vec<i32>> {
    if repaired_predictions.len() < repair_input_count {
        return Err(OpenAiError::backend(format!(
            "recovery verify returned too few tokens: expected {} got {:?}",
            repair_input_count, repaired_predictions
        )));
    }
    if accepted_before_reject > 0
        && repaired_predictions[..accepted_before_reject] != draft_tokens[..accepted_before_reject]
    {
        eprintln!(
            "recovery verify changed accepted prefix; committing restored target tokens: accepted {:?}, repaired {:?}",
            &draft_tokens[..accepted_before_reject],
            &repaired_predictions[..accepted_before_reject]
        );
    }
    Ok(repaired_predictions[..repair_input_count].to_vec())
}

pub(super) fn nonzero_min(current: usize, candidate: usize) -> usize {
    if current == 0 {
        candidate
    } else {
        current.min(candidate)
    }
}
