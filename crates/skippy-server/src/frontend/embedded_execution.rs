use super::*;

impl StageOpenAiBackend {
    pub(super) fn execute_embedded_stage_message(
        &self,
        request: &EmbeddedStageZeroGeneration<'_>,
        downstream: &mut TcpStream,
        session_key: &str,
        message: &StageWireMessage,
        token_ids: &[i32],
        expected_reply: WireReplyKind,
    ) -> OpenAiResult<EmbeddedStageExecution> {
        let timer = PhaseTimer::start();
        let mut stats = StageReplyStats::default();
        let stage0_timer = PhaseTimer::start();
        let output = {
            let lock_timer = PhaseTimer::start();
            let mut runtime = self
                .runtime
                .lock()
                .map_err(|_| OpenAiError::backend("runtime lock poisoned"))?;
            let lock_wait_ms = lock_timer.elapsed_ms();
            let hold_timer = PhaseTimer::start();
            if message.kind == WireMessageKind::VerifySpan
                && (message.state.flags & state_flags::SKIP_VERIFY_CHECKPOINT) == 0
            {
                let checkpoint_timer = PhaseTimer::start();
                runtime
                    .checkpoint_session(session_key)
                    .map_err(openai_backend_error)?;
                let checkpoint_us = ms_to_us(checkpoint_timer.elapsed_ms());
                stats.checkpoint_local_us += checkpoint_us;
                stats.checkpoint_total_us += checkpoint_us;
                stats.verify_span_checkpointed_requests += 1;
            } else if message.kind == WireMessageKind::VerifySpan {
                stats.verify_span_skip_checkpoint_requests += 1;
            }
            let output = run_binary_stage_message(
                &mut runtime,
                session_key,
                message,
                token_ids,
                None,
                false,
            )
            .map_err(openai_backend_error)?
            .2;
            let hold_ms = hold_timer.elapsed_ms();
            EmbeddedLocalOutput {
                output,
                runtime_lock_wait_ms: lock_wait_ms,
                runtime_lock_hold_ms: hold_ms,
            }
        };
        let stage0_compute_ms = stage0_timer.elapsed_ms();
        let forwarded = forwarded_stage_message_timed(
            request.config,
            message,
            &output.output,
            request.wire_dtype,
            request.activation_width,
        )
        .map_err(openai_backend_error)?;
        let write_timer = PhaseTimer::start();
        write_stage_message_conditioned(
            &mut *downstream,
            &forwarded.message,
            request.wire_dtype,
            request.downstream_wire_condition,
        )
        .map_err(openai_io_error)?;
        let forward_write_ms = write_timer.elapsed_ms();
        let wait_timer = PhaseTimer::start();
        let reply = recv_reply(&mut *downstream).map_err(openai_io_error)?;
        let downstream_wait_ms = wait_timer.elapsed_ms();
        if reply.kind != expected_reply {
            return Err(OpenAiError::backend(format!(
                "expected {expected_reply:?} reply from downstream, got {:?}",
                reply.kind
            )));
        }
        stats.merge(reply.stats);
        if message.kind == WireMessageKind::VerifySpan {
            stats.verify_span_compute_us += ms_to_us(stage0_compute_ms);
            stats.verify_span_forward_write_us += ms_to_us(forward_write_ms);
            stats.verify_span_downstream_wait_us += ms_to_us(downstream_wait_ms);
            stats.verify_span_total_us += ms_to_us(timer.elapsed_ms());
            stats.verify_span_stage_count += 1;
            stats.verify_span_request_count += 1;
            stats.verify_span_token_count += i64::from(message.token_count.max(0));
            stats.verify_span_max_tokens = stats
                .verify_span_max_tokens
                .max(i64::from(message.token_count.max(0)));
        }
        Ok(EmbeddedStageExecution {
            reply: StageReply { stats, ..reply },
            stats: EmbeddedExecutionStats {
                stage0_compute_ms,
                runtime_lock_wait_ms: output.runtime_lock_wait_ms,
                runtime_lock_hold_ms: output.runtime_lock_hold_ms,
                activation_encode_ms: forwarded.activation_encode_ms,
                output_activation_bytes: output.output.payload.len(),
                forward_activation_bytes: forwarded.message.activation.len(),
                forward_write_ms,
                downstream_wait_ms,
            },
            elapsed_ms: timer.elapsed_ms(),
        })
    }

    pub(super) fn restore_embedded_stage_session(
        &self,
        request: &EmbeddedStageZeroGeneration<'_>,
        downstream: &mut TcpStream,
        session_key: &str,
        request_id: u64,
        session_id: u64,
    ) -> OpenAiResult<EmbeddedSessionControl> {
        let timer = PhaseTimer::start();
        let local_timer = PhaseTimer::start();
        {
            let mut runtime = self
                .runtime
                .lock()
                .map_err(|_| OpenAiError::backend("runtime lock poisoned"))?;
            runtime
                .restore_session(session_key)
                .map_err(openai_backend_error)?;
        }
        let local_ms = local_timer.elapsed_ms();
        let message = embedded_session_control_message(
            request.wire_dtype,
            WireMessageKind::RestoreSession,
            request_id,
            session_id,
        );
        let write_timer = PhaseTimer::start();
        write_stage_message_conditioned(
            &mut *downstream,
            &message,
            request.wire_dtype,
            request.downstream_wire_condition,
        )
        .map_err(openai_io_error)?;
        let downstream_write_ms = write_timer.elapsed_ms();
        let wait_timer = PhaseTimer::start();
        let reply = recv_reply(&mut *downstream).map_err(openai_io_error)?;
        let downstream_wait_ms = wait_timer.elapsed_ms();
        if reply.kind != WireReplyKind::Ack {
            return Err(OpenAiError::backend(format!(
                "restore expected ACK from downstream, got {:?}",
                reply.kind
            )));
        }
        Ok(EmbeddedSessionControl {
            elapsed_ms: timer.elapsed_ms(),
            local_ms,
            downstream_write_ms,
            downstream_wait_ms,
        })
    }
}
