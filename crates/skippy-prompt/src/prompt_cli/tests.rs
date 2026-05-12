#[cfg(test)]
mod speculative_tests {
    use super::*;

    #[test]
    fn resolve_stage_ranges_supports_single_full_model_stage() {
        let ranges = resolve_stage_ranges(true, None, 4, 40).unwrap();
        assert_eq!(ranges, vec![(0, 40)]);
    }

    #[test]
    fn empty_splits_also_describe_one_full_range() {
        let ranges = resolve_stage_ranges(false, Some(""), 4, 40).unwrap();
        assert_eq!(ranges, vec![(0, 40)]);
    }

    #[test]
    fn even_stage_ranges_accepts_one_stage() {
        let ranges = even_stage_ranges(1, 40).unwrap();
        assert_eq!(ranges, vec![(0, 40)]);
    }

    #[test]
    fn resolve_stage_ranges_rejects_empty_layer_range() {
        let err = resolve_stage_ranges(true, None, 1, 0).unwrap_err();
        assert!(
            err.to_string()
                .contains("layer_end must be greater than zero"),
            "{err:#}"
        );
    }

    #[test]
    fn thinking_override_respects_no_think_and_budget_zero() {
        assert_eq!(normalized_prompt_thinking(false, None), None);
        assert_eq!(normalized_prompt_thinking(true, None), Some(false));
        assert_eq!(normalized_prompt_thinking(false, Some(0)), Some(false));
        assert_eq!(normalized_prompt_thinking(false, Some(128)), Some(true));
    }

    fn normalized_prompt_thinking(no_think: bool, budget: Option<usize>) -> Option<bool> {
        let reasoning = prompt_openai_reasoning_config(no_think, budget).unwrap();
        normalize_reasoning_template_options(reasoning.as_ref(), None, &BTreeMap::new())
            .unwrap()
            .enable_thinking
    }

    #[test]
    fn verify_inputs_align_with_draft_proposals() {
        assert_eq!(verify_inputs_for_proposals(10, &[]), Vec::<i32>::new());
        assert_eq!(verify_inputs_for_proposals(10, &[11]), vec![10]);
        assert_eq!(
            verify_inputs_for_proposals(10, &[11, 12, 13]),
            vec![10, 11, 12]
        );
    }

    #[test]
    fn classify_verify_span_full_accept() {
        let decision =
            classify_verify_span(&[10, 11, 12], &[10, 11, 12], 0, 16, |_| Ok(false)).unwrap();
        assert_eq!(
            decision,
            VerifySpanDecision {
                kind: VerifySpanDecisionKind::FullAccept,
                accepted_before_reject: 3,
                repair_input_count: None,
                commit_count: 3,
            }
        );
        assert!(!decision.rejected());
        assert!(!decision.requires_repair());
    }

    #[test]
    fn classify_verify_span_tail_reject_keeps_state() {
        let decision =
            classify_verify_span(&[10, 11, 12], &[10, 11, 42], 0, 16, |_| Ok(false)).unwrap();
        assert_eq!(
            decision,
            VerifySpanDecision {
                kind: VerifySpanDecisionKind::TailReject,
                accepted_before_reject: 2,
                repair_input_count: Some(3),
                commit_count: 3,
            }
        );
        assert!(decision.rejected());
        assert!(decision.tail_reject());
        assert!(!decision.requires_repair());
    }

    #[test]
    fn classify_verify_span_early_reject_requires_repair() {
        let decision =
            classify_verify_span(&[10, 11, 12, 13], &[10, 42, 77, 88], 0, 16, |_| Ok(false))
                .unwrap();
        assert_eq!(
            decision,
            VerifySpanDecision {
                kind: VerifySpanDecisionKind::EarlyReject,
                accepted_before_reject: 1,
                repair_input_count: Some(2),
                commit_count: 2,
            }
        );
        assert!(decision.rejected());
        assert!(!decision.tail_reject());
        assert!(decision.requires_repair());
    }

    #[test]
    fn classify_verify_span_accepted_eog_stops_without_growing_window() {
        let decision =
            classify_verify_span(&[10, 99, 12], &[10, 99, 12], 0, 16, |token| Ok(token == 99))
                .unwrap();
        assert_eq!(
            decision,
            VerifySpanDecision {
                kind: VerifySpanDecisionKind::AcceptedStop,
                accepted_before_reject: 2,
                repair_input_count: None,
                commit_count: 2,
            }
        );
        assert!(!decision.rejected());
        assert!(!decision.requires_repair());
    }

    #[test]
    fn classify_verify_span_early_reject_at_limit_does_not_repair() {
        let decision =
            classify_verify_span(&[10, 11, 12], &[10, 42, 77], 2, 4, |_| Ok(false)).unwrap();
        assert_eq!(
            decision,
            VerifySpanDecision {
                kind: VerifySpanDecisionKind::EarlyRejectStop,
                accepted_before_reject: 1,
                repair_input_count: Some(2),
                commit_count: 2,
            }
        );
        assert!(decision.rejected());
        assert!(!decision.tail_reject());
        assert!(!decision.requires_repair());
    }

    #[test]
    fn classify_verify_span_requires_complete_predictions() {
        let err = classify_verify_span(&[10, 11, 12], &[10, 11], 0, 16, |_| Ok(false)).unwrap_err();
        assert!(
            err.to_string()
                .contains("verify span returned too few tokens"),
            "{err:#}"
        );
    }

    #[test]
    fn observe_verify_decision_grows_on_full_accept_only() {
        let mut stats = SpeculativeStats::default();
        let mut adaptive_window = 4;
        stats.observe_verify_decision(
            VerifySpanDecision {
                kind: VerifySpanDecisionKind::FullAccept,
                accepted_before_reject: 4,
                repair_input_count: None,
                commit_count: 4,
            },
            &mut adaptive_window,
            true,
            8,
        );

        assert_eq!(adaptive_window, 5);
        assert_eq!(stats.full_accept_windows, 1);
        assert_eq!(stats.adaptive_window_grows, 1);
        assert_eq!(stats.accepted_tokens, 4);
    }

    #[test]
    fn observe_verify_decision_stop_outcomes_do_not_move_adaptive_window() {
        let mut stats = SpeculativeStats::default();
        let mut adaptive_window = 4;
        stats.observe_verify_decision(
            VerifySpanDecision {
                kind: VerifySpanDecisionKind::AcceptedStop,
                accepted_before_reject: 2,
                repair_input_count: None,
                commit_count: 2,
            },
            &mut adaptive_window,
            true,
            8,
        );
        stats.observe_verify_decision(
            VerifySpanDecision {
                kind: VerifySpanDecisionKind::EarlyRejectStop,
                accepted_before_reject: 1,
                repair_input_count: Some(2),
                commit_count: 2,
            },
            &mut adaptive_window,
            true,
            8,
        );

        assert_eq!(adaptive_window, 4);
        assert_eq!(stats.accepted_stop_windows, 1);
        assert_eq!(stats.early_reject_stop_windows, 1);
        assert_eq!(stats.repair_required_windows, 0);
        assert_eq!(stats.adaptive_window_grows, 0);
        assert_eq!(stats.adaptive_window_shrinks, 0);
    }

    #[test]
    fn observe_verify_decision_early_reject_shrinks_and_marks_repair() {
        let mut stats = SpeculativeStats::default();
        let mut adaptive_window = 6;
        stats.observe_verify_decision(
            VerifySpanDecision {
                kind: VerifySpanDecisionKind::EarlyReject,
                accepted_before_reject: 1,
                repair_input_count: Some(2),
                commit_count: 2,
            },
            &mut adaptive_window,
            true,
            8,
        );

        assert_eq!(adaptive_window, 5);
        assert_eq!(stats.early_reject_windows, 1);
        assert_eq!(stats.repair_required_windows, 1);
        assert_eq!(stats.adaptive_window_shrinks, 1);
        assert_eq!(stats.rejected_windows, 1);
        assert_eq!(stats.first_reject_position_sum, 2);
    }

    #[test]
    fn early_reject_commits_repaired_target_tokens() {
        let draft_tokens = [10, 11, 12, 13];
        let repaired = repaired_commit_tokens(&draft_tokens, 2, 3, &[10, 11, 42]).unwrap();
        assert_eq!(repaired, vec![10, 11, 42]);
    }

    #[test]
    fn repair_commits_changed_accepted_prefix_from_restored_state() {
        let draft_tokens = [10, 11, 12, 13];
        let repaired = repaired_commit_tokens(&draft_tokens, 2, 3, &[10, 99, 42]).unwrap();
        assert_eq!(repaired, vec![10, 99, 42]);
    }

    #[test]
    fn repair_requires_the_full_repaired_prefix() {
        let draft_tokens = [10, 11, 12, 13];
        let err = repaired_commit_tokens(&draft_tokens, 2, 3, &[10, 11]).unwrap_err();
        assert!(
            err.to_string()
                .contains("recovery verify returned too few tokens"),
            "{err:#}"
        );
    }

    #[test]
    fn stable_wire_ids_are_deterministic_and_namespaced() {
        let prompt_index = 7usize.to_le_bytes();
        let session = stable_wire_id(&[b"session-a"]);
        let request = stable_wire_id(&[b"session-a", &prompt_index]);
        assert_ne!(session, 0);
        assert_eq!(session, stable_wire_id(&[b"session-a"]));
        assert_ne!(session, request);
    }
}

fn configure_process_log(command: &mut Command, log_path: &Path) -> Result<()> {
    let stdout = fs::File::create(log_path)
        .with_context(|| format!("create child log {}", log_path.display()))?;
    let stderr = stdout
        .try_clone()
        .with_context(|| format!("clone child log {}", log_path.display()))?;
    command.stdout(stdout).stderr(stderr);
    Ok(())
}

fn path_str(path: &Path) -> Result<&str> {
    path.to_str()
        .with_context(|| format!("path is not valid UTF-8: {}", path.display()))
}

fn connect_ready(addr: &str, timeout_secs: u64) -> Result<TcpStream> {
    let deadline = Instant::now() + Duration::from_secs(timeout_secs.max(1));
    let mut last_error = None;
    while Instant::now() < deadline {
        match TcpStream::connect(addr) {
            Ok(mut stream) => {
                stream.set_nodelay(true).ok();
                stream
                    .set_read_timeout(Some(Duration::from_millis(500)))
                    .ok();
                match recv_ready_until_deadline(&mut stream, deadline) {
                    Ok(()) => {
                        stream.set_read_timeout(None).ok();
                        return Ok(stream);
                    }
                    Err(error) => {
                        last_error = Some(anyhow!(error).context("ready handshake failed"))
                    }
                }
            }
            Err(error) => last_error = Some(anyhow!(error).context("connect failed")),
        }
        thread::sleep(Duration::from_millis(500));
    }
    Err(last_error.unwrap_or_else(|| anyhow!("timed out")))
}

fn recv_ready_until_deadline(stream: &mut TcpStream, deadline: Instant) -> io::Result<()> {
    let mut bytes = [0_u8; 4];
    let mut offset = 0usize;
    while offset < bytes.len() {
        if Instant::now() >= deadline {
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "timed out waiting for ready handshake",
            ));
        }
        match stream.read(&mut bytes[offset..]) {
            Ok(0) => {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "ready handshake stream closed",
                ));
            }
            Ok(read) => offset += read,
            Err(error)
                if matches!(
                    error.kind(),
                    io::ErrorKind::Interrupted
                        | io::ErrorKind::WouldBlock
                        | io::ErrorKind::TimedOut
                ) =>
            {
                thread::sleep(Duration::from_millis(20));
            }
            Err(error) => return Err(error),
        }
    }
    let magic = i32::from_le_bytes(bytes);
    if magic != READY_MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "stage ready magic mismatch",
        ));
    }
    Ok(())
}

fn parse_wire_dtype(value: &str) -> Result<WireActivationDType> {
    match value {
        "fp32" | "f32" => Ok(WireActivationDType::F32),
        "fp16" | "f16" => Ok(WireActivationDType::F16),
        "q8" | "int8" | "i8" => Ok(WireActivationDType::Q8),
        _ => bail!("unsupported activation wire dtype {value}"),
    }
}

fn unix_millis() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock before Unix epoch")
        .as_millis()
}

#[cfg(test)]
mod package_tests {
    use super::*;

    #[test]
    fn package_artifact_checks_reads_manifest_artifacts() -> Result<()> {
        let package_dir = std::env::temp_dir().join(format!(
            "skippy-prompt-package-read-test-{}-{}",
            std::process::id(),
            unix_millis()
        ));
        fs::create_dir_all(&package_dir)?;
        let manifest = serde_json::json!({
            "shared": {
                "metadata": {"path": "shared/metadata.gguf", "artifact_bytes": 11},
                "embeddings": {"path": "shared/embeddings.gguf", "artifact_bytes": 22},
                "output": {"path": "shared/output.gguf", "artifact_bytes": 33}
            },
            "layers": [
                {"path": "layers/layer-000.gguf", "artifact_bytes": 44},
                {"path": "layers/layer-001.gguf", "artifact_bytes": 55}
            ]
        });
        fs::write(
            package_dir.join("model-package.json"),
            serde_json::to_vec(&manifest)?,
        )?;

        let checks = package_artifact_checks(&package_dir)?;
        fs::remove_dir_all(&package_dir).ok();

        assert_eq!(checks.len(), 5);
        assert_eq!(checks[0].path, "shared/metadata.gguf");
        assert_eq!(checks[0].artifact_bytes, 11);
        assert_eq!(checks[4].path, "layers/layer-001.gguf");
        assert_eq!(checks[4].artifact_bytes, 55);
        Ok(())
    }

    #[test]
    fn package_artifact_checks_rejects_unsafe_paths() -> Result<()> {
        let package_dir = std::env::temp_dir().join(format!(
            "skippy-prompt-package-unsafe-test-{}-{}",
            std::process::id(),
            unix_millis()
        ));
        fs::create_dir_all(&package_dir)?;
        let manifest = serde_json::json!({
            "shared": {
                "metadata": {"path": "../metadata.gguf", "artifact_bytes": 11},
                "embeddings": {"path": "shared/embeddings.gguf", "artifact_bytes": 22},
                "output": {"path": "shared/output.gguf", "artifact_bytes": 33}
            },
            "layers": []
        });
        fs::write(
            package_dir.join("model-package.json"),
            serde_json::to_vec(&manifest)?,
        )?;

        let result = package_artifact_checks(&package_dir);
        fs::remove_dir_all(&package_dir).ok();

        assert!(result.is_err());
        Ok(())
    }
}
