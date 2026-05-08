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
