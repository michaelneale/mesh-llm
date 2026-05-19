#[test]
fn early_tui_spawns_before_llama_ready_in_active_flow() {
    runtime::assert_active_serve_path_spawn_gate_behavior();
}

#[test]
fn passive_path_tui_still_starts_immediately() {
    runtime::assert_passive_path_immediate_spawn_behavior();
}

#[tokio::test]
async fn non_serving_subcommands_retain_plain_output() {
    runtime::assert_non_serving_dispatch_short_circuit_behavior().await;
}

#[test]
fn startup_lifecycle_transitions_pending_partial_ready_failed() {
    cli::output::assert_startup_lifecycle_transitions_pending_partial_ready_failed();
}

#[test]
fn startup_lifecycle_keeps_runtime_ready_as_final_edge() {
    cli::output::assert_startup_lifecycle_keeps_runtime_ready_as_final_edge();
}

#[test]
fn startup_failures_surface_in_tui_events_and_status() {
    cli::output::assert_startup_failures_surface_in_tui_events_and_status();
}

#[test]
fn startup_failure_summary_sanitizes_multiline_detail() {
    cli::output::assert_startup_failure_summary_sanitizes_multiline_detail();
}

#[test]
fn rpc_and_llama_startup_failures_mark_components_failed() {
    cli::output::assert_rpc_and_llama_startup_failures_mark_components_failed();
}

#[test]
fn discovery_and_join_failures_mark_startup_mesh_component_failed() {
    cli::output::assert_discovery_and_join_failures_mark_startup_mesh_component_failed();
}

#[test]
fn post_ready_peer_churn_does_not_reopen_startup_failure() {
    cli::output::assert_post_ready_peer_churn_does_not_reopen_startup_failure();
}

#[test]
fn interactive_handler_spawns_once_across_startup_callbacks() {
    runtime::assert_interactive_handler_spawns_once_across_startup_callbacks();
}

#[test]
fn startup_history_is_visible_after_late_tui_attach() {
    cli::output::assert_startup_history_is_visible_after_late_tui_attach();
}

#[test]
fn startup_history_keeps_order_when_tui_attaches_late() {
    cli::output::assert_startup_history_keeps_order_when_tui_attaches_late();
}

#[test]
fn endpoint_rows_remain_starting_until_ready_events() {
    cli::output::assert_endpoint_rows_remain_starting_until_ready_events();
}

#[test]
fn startup_launch_plan_renders_not_ready_rows_before_actions() {
    cli::output::assert_startup_launch_plan_renders_not_ready_rows_before_actions();
}

#[test]
fn tui_model_progress_renders_dashboard_without_loading_screen() {
    cli::output::assert_tui_model_progress_renders_dashboard_without_loading_screen();
}

#[test]
fn tui_startup_progress_continues_in_dashboard_after_model_download_ready() {
    cli::output::assert_tui_startup_progress_continues_in_dashboard_after_model_download_ready();
}

#[test]
fn startup_progress_after_launch_plan_shows_dashboard_not_loader() {
    cli::output::assert_startup_progress_after_launch_plan_shows_dashboard_not_loader();
}

#[test]
fn planned_rows_transition_from_not_ready_to_ready_events() {
    cli::output::assert_planned_rows_transition_from_not_ready_to_ready_events();
}

#[test]
fn launch_plan_rows_survive_empty_startup_snapshot() {
    cli::output::assert_launch_plan_rows_survive_empty_startup_snapshot();
}

#[test]
fn launch_plan_preserves_distinct_port_zero_endpoint_rows() {
    cli::output::assert_launch_plan_preserves_distinct_port_zero_endpoint_rows();
}

#[test]
fn snapshot_upsert_preserves_distinct_port_zero_endpoint_rows() {
    cli::output::assert_snapshot_upsert_preserves_distinct_port_zero_endpoint_rows();
}

#[test]
fn planned_port_zero_process_rows_bind_to_concrete_startup_events() {
    cli::output::assert_planned_port_zero_process_rows_bind_to_concrete_startup_events();
}

#[test]
fn startup_launch_plan_describes_planned_runtime_before_process_start() {
    runtime::assert_startup_launch_plan_describes_planned_runtime_before_process_start();
}

#[test]
fn fallback_mode_surfaces_startup_failures_without_tui() {
    cli::output::assert_fallback_mode_surfaces_startup_failures_without_tui();
}

#[test]
fn quitting_during_startup_cancels_without_late_ready_render() {
    runtime::assert_quitting_during_startup_cancels_without_late_ready_render();
}

#[test]
fn interactive_preterminal_render_uses_plain_event_output() {
    cli::output::assert_interactive_preterminal_render_uses_plain_event_output();
}

#[test]
fn interactive_post_terminal_exit_resumes_plain_event_output() {
    cli::output::assert_interactive_post_terminal_exit_resumes_plain_event_output();
}

#[test]
fn tui_model_card_separates_name_from_metadata_columns() {
    cli::output::assert_tui_model_card_separates_name_from_metadata_columns();
}

#[test]
fn mesh_requirements_docs_examples_parse() {
    cli::assert_mesh_requirements_docs_examples_parse();
}

#[test]
fn mesh_requirements_policy_canonical_hash_is_stable() {
    mesh::requirements::tests::assert_mesh_requirements_policy_canonical_hash_is_stable();
}

#[test]
fn mesh_requirements_policy_change_changes_mesh_id() {
    mesh::requirements::tests::assert_mesh_requirements_policy_change_changes_mesh_id();
}

#[test]
fn mesh_requirements_bootstrap_token_validates_origin_signature() {
    mesh::requirements::tests::assert_mesh_requirements_bootstrap_token_validates_origin_signature(
    );
}

#[test]
fn mesh_requirements_bootstrap_rejects_expired_token() {
    mesh::requirements::tests::assert_mesh_requirements_bootstrap_rejects_expired_token();
}

#[test]
fn mesh_requirements_bootstrap_rejects_policy_hash_mismatch() {
    mesh::requirements::tests::assert_mesh_requirements_bootstrap_rejects_policy_hash_mismatch();
}

#[test]
fn mesh_requirements_policy_hash_derives_mesh_id() {
    mesh::requirements::tests::assert_mesh_requirements_policy_hash_derives_mesh_id();
}

#[test]
fn mesh_requirements_policy_change_creates_distinct_mesh() {
    mesh::requirements::tests::assert_mesh_requirements_policy_change_changes_mesh_id();
}

#[test]
fn mesh_requirements_version_bounds_unset_min_only_max_only_and_exact() {
    mesh::requirements::tests::assert_mesh_requirements_version_bounds_unset_min_only_max_only_and_exact();
}

#[test]
fn mesh_requirements_protocol_bounds_reject_unknown_only_when_constrained() {
    mesh::requirements::tests::assert_mesh_requirements_protocol_bounds_reject_unknown_only_when_constrained();
}

#[test]
fn mesh_requirements_rejects_unsigned_when_attestation_required() {
    mesh::requirements::tests::assert_mesh_requirements_rejects_unsigned_when_attestation_required(
    );
}

#[test]
fn mesh_requirements_rejection_reasons_are_stable() {
    mesh::requirements::tests::assert_mesh_requirements_rejection_reasons_are_stable();
}

#[test]
fn mesh_requirements_cli_accepts_each_bound_independently() {
    runtime::assert_mesh_requirements_cli_accepts_each_bound_independently();
}

#[test]
fn mesh_requirements_config_accepts_unset_min_only_max_only_and_full_ranges() {
    plugin::assert_mesh_requirements_config_accepts_unset_min_only_max_only_and_full_ranges();
}

#[test]
fn mesh_requirements_config_rejects_required_attestation_without_signer_keys() {
    plugin::assert_mesh_requirements_config_rejects_required_attestation_without_signer_keys();
}

#[test]
fn mesh_requirements_config_rejects_non_ed25519_signer_key() {
    plugin::assert_mesh_requirements_config_rejects_non_ed25519_signer_key();
}

#[test]
fn mesh_requirements_survive_owner_control_config_round_trip() {
    protocol::tests::mesh_requirements_survive_owner_control_config_round_trip();
}

#[test]
fn mesh_requirements_cli_overrides_config_per_field_before_genesis() {
    runtime::assert_mesh_requirements_cli_overrides_config_per_field_before_genesis();
}

#[test]
fn mesh_requirements_config_rejects_min_greater_than_max_after_merge() {
    runtime::assert_mesh_requirements_config_rejects_min_greater_than_max_after_merge();
}

#[test]
fn mesh_requirements_rejects_local_policy_mutation_on_existing_mesh() {
    runtime::assert_mesh_requirements_rejects_local_policy_mutation_on_existing_mesh();
}

#[test]
fn mesh_requirements_direct_proof_rejects_stale_timestamp() {
    mesh::requirements::tests::assert_mesh_requirements_direct_proof_rejects_stale_timestamp();
}

#[test]
fn mesh_requirements_direct_proof_rejects_sender_id_mismatch() {
    mesh::requirements::tests::assert_mesh_requirements_direct_proof_rejects_sender_id_mismatch();
}

#[test]
fn mesh_requirements_outbound_admits_compliant_peer_after_requirements_pass() {
    mesh::tests::assert_mesh_requirements_outbound_admits_compliant_peer_after_requirements_pass();
}

#[test]
fn mesh_requirements_inbound_rejects_before_topology_announcement() {
    mesh::tests::assert_mesh_requirements_inbound_rejects_before_topology_announcement();
}

#[test]
fn mesh_requirements_outbound_rejects_before_peer_promotion() {
    mesh::tests::assert_mesh_requirements_outbound_rejects_before_peer_promotion();
}

#[test]
fn mesh_requirements_add_peer_rejects_missing_direct_admission_proof() {
    mesh::tests::assert_mesh_requirements_add_peer_rejects_missing_direct_admission_proof();
}

#[test]
fn mesh_requirements_add_peer_rejects_invalid_direct_admission_proof() {
    mesh::tests::assert_mesh_requirements_add_peer_rejects_invalid_direct_admission_proof();
}

#[test]
fn mesh_requirements_add_peer_rejects_stale_direct_admission_proof() {
    mesh::tests::assert_mesh_requirements_add_peer_rejects_stale_direct_admission_proof();
}

#[test]
fn mesh_requirements_add_peer_rejects_direct_proof_sender_mismatch() {
    mesh::tests::assert_mesh_requirements_add_peer_rejects_direct_proof_sender_mismatch();
}

#[test]
fn requirement_aware_mesh_without_attestation_rejects_missing_direct_proof() {
    mesh::tests::assert_requirement_aware_mesh_without_attestation_rejects_missing_direct_proof();
}

#[test]
fn requirement_aware_mesh_without_attestation_rejects_invalid_direct_proof() {
    mesh::tests::assert_requirement_aware_mesh_without_attestation_rejects_invalid_direct_proof();
}

#[test]
fn requirement_aware_mesh_without_attestation_rejects_stale_direct_proof() {
    mesh::tests::assert_requirement_aware_mesh_without_attestation_rejects_stale_direct_proof();
}

#[test]
fn requirement_aware_mesh_without_attestation_rejects_sender_mismatch_direct_proof() {
    mesh::tests::assert_requirement_aware_mesh_without_attestation_rejects_sender_mismatch_direct_proof();
}

#[test]
fn requirement_aware_mesh_without_attestation_accepts_valid_direct_proof() {
    mesh::tests::assert_requirement_aware_mesh_without_attestation_accepts_valid_direct_proof();
}

#[test]
fn mesh_requirements_add_peer_rejects_untrusted_release_signer() {
    mesh::tests::assert_mesh_requirements_add_peer_rejects_untrusted_release_signer();
}

#[test]
fn mesh_requirements_add_peer_rejects_invalid_release_attestation_signature() {
    mesh::tests::assert_mesh_requirements_add_peer_rejects_invalid_release_attestation_signature();
}

#[test]
fn mesh_requirements_add_peer_rejects_wrong_mesh_id() {
    mesh::tests::assert_mesh_requirements_add_peer_rejects_wrong_mesh_id();
}

#[test]
fn mesh_requirements_transitive_gossip_never_admits_peer_without_direct_proof() {
    mesh::tests::assert_mesh_requirements_transitive_gossip_never_admits_peer_without_direct_proof(
    );
}

#[test]
fn mesh_requirements_rejected_peer_messages_have_no_mesh_effect() {
    mesh::tests::assert_mesh_requirements_rejected_peer_messages_have_no_mesh_effect();
}

#[test]
fn mesh_requirements_join_rejects_invalid_bootstrap_token() {
    mesh::tests::assert_mesh_requirements_join_rejects_invalid_bootstrap_token();
}

#[test]
fn mesh_requirements_unrestricted_legacy_mesh_join_stays_compatible() {
    mesh::tests::assert_mesh_requirements_unrestricted_legacy_mesh_join_stays_compatible();
}

#[test]
fn mesh_requirements_status_excludes_rejected_peers_from_admitted_list() {
    api::tests::assert_mesh_requirements_status_excludes_rejected_peers_from_admitted_list();
}

#[test]
fn mesh_requirements_status_reports_policy_hash_read_only() {
    api::tests::assert_mesh_requirements_status_reports_policy_hash_read_only();
}

#[test]
fn mesh_requirements_certified_binary_required_event_text() {
    api::tests::assert_mesh_requirements_certified_binary_required_event_text();
}

#[test]
fn mesh_requirements_rejection_events_do_not_expose_tokens() {
    api::tests::assert_mesh_requirements_rejection_events_do_not_expose_tokens();
}
