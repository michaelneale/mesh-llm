use crate::api::{MeshApi, RuntimeControlRequest};
use crate::cli::output::{
    ConsoleSessionMode, OutputManager, TuiControlFlow, TuiEvent, TuiKeyEvent,
};
use crossterm::event::{
    self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers, MouseButton, MouseEvent,
    MouseEventKind,
};
use crossterm::terminal::{disable_raw_mode, enable_raw_mode, size};
use std::fmt;
#[cfg(test)]
use std::io::Write;
use std::io::{BufRead, IsTerminal};
use std::time::Duration;

pub(crate) const HELP_TEXT: &str = "help: h=help, q=quit, i=info snapshot";
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) const READY_PROMPT: &str = "> ";
const TUI_QUIT_FRAME_DELAY: Duration = Duration::from_millis(150);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum InitialPromptMode {
    Immediate,
    Deferred,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum InteractiveEntryKind {
    Tui,
    Line,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum InteractiveCommand {
    Help,
    Quit,
    Info,
}

fn parse_command(line: &str) -> Option<InteractiveCommand> {
    match line.trim() {
        "h" => Some(InteractiveCommand::Help),
        "q" => Some(InteractiveCommand::Quit),
        "i" => Some(InteractiveCommand::Info),
        _ => None,
    }
}

pub(crate) fn console_session_mode(stdin_is_tty: bool, stderr_is_tty: bool) -> ConsoleSessionMode {
    console_session_mode_for_term(
        stdin_is_tty,
        stderr_is_tty,
        std::env::var("TERM").ok().as_deref(),
    )
}

fn console_session_mode_for_term(
    stdin_is_tty: bool,
    stderr_is_tty: bool,
    term: Option<&str>,
) -> ConsoleSessionMode {
    if stdin_is_tty && stderr_is_tty && terminal_supports_dashboard(term) {
        ConsoleSessionMode::InteractiveDashboard
    } else {
        ConsoleSessionMode::Fallback
    }
}

fn terminal_supports_dashboard(term: Option<&str>) -> bool {
    match term.map(str::trim).filter(|term| !term.is_empty()) {
        Some(term) => term != "dumb",
        None => false,
    }
}

pub(crate) fn current_console_session_mode() -> ConsoleSessionMode {
    console_session_mode(
        std::io::stdin().is_terminal(),
        std::io::stderr().is_terminal(),
    )
}

#[cfg(test)]
fn write_ready_prompt<W: Write>(writer: &mut W) -> std::io::Result<()> {
    writer.write_all(READY_PROMPT.as_bytes())?;
    writer.flush()
}

#[cfg(test)]
fn maybe_write_initial_prompt<W: Write>(
    writer: &mut W,
    mode: InitialPromptMode,
) -> std::io::Result<()> {
    if matches!(mode, InitialPromptMode::Immediate) {
        write_ready_prompt(writer)?;
    }
    Ok(())
}

pub(crate) fn spawn_handler(
    control_tx: tokio::sync::mpsc::UnboundedSender<RuntimeControlRequest>,
    console_state: MeshApi,
    output_manager: &'static OutputManager,
    initial_prompt_mode: InitialPromptMode,
) {
    spawn_handler_with_first_paint_ack(
        control_tx,
        console_state,
        output_manager,
        initial_prompt_mode,
        None,
    );
}

pub(crate) fn spawn_handler_with_first_paint_ack(
    control_tx: tokio::sync::mpsc::UnboundedSender<RuntimeControlRequest>,
    console_state: MeshApi,
    output_manager: &'static OutputManager,
    initial_prompt_mode: InitialPromptMode,
    first_paint_ack: Option<tokio::sync::oneshot::Sender<std::io::Result<()>>>,
) {
    match interactive_entry_kind(output_manager.console_session_mode()) {
        InteractiveEntryKind::Tui => spawn_tui_handler(
            control_tx,
            console_state,
            output_manager,
            initial_prompt_mode,
            first_paint_ack,
        ),
        InteractiveEntryKind::Line => {
            if let Some(ack) = first_paint_ack {
                let _ = ack.send(Ok(()));
            }
            spawn_line_handler(
                control_tx,
                console_state,
                output_manager,
                initial_prompt_mode,
            );
        }
    }
}

pub(crate) fn interactive_entry_kind(
    console_session_mode: Option<ConsoleSessionMode>,
) -> InteractiveEntryKind {
    match console_session_mode {
        Some(ConsoleSessionMode::InteractiveDashboard) => InteractiveEntryKind::Tui,
        _ => InteractiveEntryKind::Line,
    }
}

#[cfg(test)]
pub(crate) fn assert_deferred_initial_prompt_waits_for_runtime_ready() {
    let mut output = Vec::new();
    maybe_write_initial_prompt(&mut output, InitialPromptMode::Deferred)
        .expect("deferred prompt should remain a no-op until RuntimeReady");
    assert!(
        output.is_empty(),
        "deferred prompt mode must not write the ready prompt during early interactive startup"
    );
}

fn spawn_line_handler(
    control_tx: tokio::sync::mpsc::UnboundedSender<RuntimeControlRequest>,
    console_state: MeshApi,
    output_manager: &'static OutputManager,
    initial_prompt_mode: InitialPromptMode,
) {
    let runtime_handle = tokio::runtime::Handle::current();
    spawn_line_handler_with_runtime(
        &runtime_handle,
        control_tx,
        console_state,
        output_manager,
        initial_prompt_mode,
    );
}

fn spawn_line_handler_with_runtime(
    runtime_handle: &tokio::runtime::Handle,
    control_tx: tokio::sync::mpsc::UnboundedSender<RuntimeControlRequest>,
    console_state: MeshApi,
    output_manager: &'static OutputManager,
    initial_prompt_mode: InitialPromptMode,
) {
    if matches!(initial_prompt_mode, InitialPromptMode::Immediate) {
        let _ = output_manager.write_ready_prompt();
    }

    let (line_tx, mut line_rx) =
        tokio::sync::mpsc::unbounded_channel::<Result<String, std::io::Error>>();
    if let Err(err) = std::thread::Builder::new()
        .name("mesh-llm-interactive-stdin".to_string())
        .spawn(move || {
            let stdin = std::io::stdin();
            let mut locked = stdin.lock();
            loop {
                let mut line = String::new();
                match locked.read_line(&mut line) {
                    Ok(0) => break,
                    Ok(_) => {
                        if line_tx.send(Ok(line)).is_err() {
                            break;
                        }
                    }
                    Err(err) => {
                        let _ = line_tx.send(Err(err));
                        break;
                    }
                }
            }
        })
    {
        tracing::warn!("interactive stdin thread failed to start: {err}");
        return;
    }

    runtime_handle.spawn(async move {
        loop {
            match line_rx.recv().await {
                Some(Ok(line)) => match parse_command(&line) {
                    Some(InteractiveCommand::Help) => {
                        eprintln!("{HELP_TEXT}");
                    }
                    Some(InteractiveCommand::Quit) => {
                        if control_tx.send(RuntimeControlRequest::Shutdown).is_err() {
                            tracing::warn!("interactive shutdown request dropped because runtime control is unavailable");
                        }
                        break;
                    }
                    Some(InteractiveCommand::Info) => {
                        eprintln!("{}", console_state.status_snapshot_string().await);
                    }
                    None => {}
                },
                None => break,
                Some(Err(err)) => {
                    tracing::warn!("interactive stdin read failed: {err}");
                    break;
                }
            }

            if output_manager.ready_prompt_active() {
                let _ = output_manager.write_ready_prompt();
            }
        }
    });
}

fn spawn_tui_handler(
    control_tx: tokio::sync::mpsc::UnboundedSender<RuntimeControlRequest>,
    console_state: MeshApi,
    output_manager: &'static OutputManager,
    initial_prompt_mode: InitialPromptMode,
    first_paint_ack: Option<tokio::sync::oneshot::Sender<std::io::Result<()>>>,
) {
    let runtime_handle = tokio::runtime::Handle::current();
    if let Err(err) = std::thread::Builder::new()
        .name("mesh-llm-interactive-tui".to_string())
        .spawn(move || {
            let fallback_control_tx = control_tx.clone();
            if let Err(err) =
                run_tui_loop(&runtime_handle, control_tx, output_manager, first_paint_ack)
            {
                let should_fallback = err.should_fallback_to_line_handler();
                tracing::warn!("interactive pretty loop failed: {err}");
                if should_fallback {
                    tracing::warn!(
                        "falling back to line-oriented pretty input after TUI startup failure"
                    );
                    spawn_line_handler_with_runtime(
                        &runtime_handle,
                        fallback_control_tx,
                        console_state,
                        output_manager,
                        initial_prompt_mode,
                    );
                }
            }
        })
    {
        tracing::warn!("interactive pretty stdin thread failed to start: {err}");
    }
}

fn run_tui_loop(
    runtime_handle: &tokio::runtime::Handle,
    control_tx: tokio::sync::mpsc::UnboundedSender<RuntimeControlRequest>,
    output_manager: &'static OutputManager,
    mut first_paint_ack: Option<tokio::sync::oneshot::Sender<std::io::Result<()>>>,
) -> Result<(), TuiLoopError> {
    enable_raw_mode()
        .map_err(std::io::Error::other)
        .map_err(TuiLoopError::startup)?;
    let mut cleanup_guard = TuiTerminalCleanupGuard::armed();
    let mut shutdown_requested = false;
    let mut shutdown_sent = false;

    let result = (|| -> Result<(), TuiLoopError> {
        if let Err(err) = runtime_handle.block_on(output_manager.enter_tui()) {
            send_first_paint_ack(&mut first_paint_ack, Err(clone_io_error(&err)));
            return Err(TuiLoopError::startup(err));
        }

        if let Ok((columns, rows)) = size() {
            let _ = runtime_handle
                .block_on(output_manager.dispatch_tui_event(TuiEvent::Resize { columns, rows }));
        }

        match runtime_handle.block_on(output_manager.render_tui_if_dirty()) {
            Ok(_) => send_first_paint_ack(&mut first_paint_ack, Ok(())),
            Err(err) => {
                send_first_paint_ack(&mut first_paint_ack, Err(clone_io_error(&err)));
                return Err(TuiLoopError::startup(err));
            }
        }

        loop {
            if !event::poll(Duration::from_millis(50))
                .map_err(std::io::Error::other)
                .map_err(TuiLoopError::runtime)?
            {
                continue;
            }

            let Some(event) = read_tui_event(
                event::read()
                    .map_err(std::io::Error::other)
                    .map_err(TuiLoopError::runtime)?,
            ) else {
                continue;
            };
            match runtime_handle
                .block_on(output_manager.dispatch_tui_event(event))
                .map_err(TuiLoopError::runtime)?
            {
                TuiControlFlow::Continue => {}
                TuiControlFlow::Quit => {
                    shutdown_requested = true;
                    if control_tx.send(RuntimeControlRequest::Shutdown).is_err() {
                        tracing::warn!(
                            "interactive shutdown request dropped because runtime control is unavailable"
                        );
                    }
                    shutdown_sent = true;
                    std::thread::sleep(TUI_QUIT_FRAME_DELAY);
                    if let Err(err) = runtime_handle.block_on(output_manager.render_tui_if_dirty())
                    {
                        tracing::warn!("interactive shutdown frame render failed: {err}");
                    }
                    break;
                }
            }
        }

        Ok(())
    })();

    let (exit_result, raw_result) = restore_tui_terminal_after_loop(
        runtime_handle.block_on(output_manager.exit_tui()),
        crate::cli::output::force_restore_tui_terminal,
        || disable_raw_mode().map_err(std::io::Error::other),
    );
    cleanup_guard.disarm();

    if shutdown_requested
        && !shutdown_sent
        && control_tx.send(RuntimeControlRequest::Shutdown).is_err()
    {
        tracing::warn!(
            "interactive shutdown request dropped because runtime control is unavailable"
        );
    }

    result?;
    exit_result.map_err(TuiLoopError::runtime)?;
    raw_result.map_err(TuiLoopError::runtime)
}

fn send_first_paint_ack(
    first_paint_ack: &mut Option<tokio::sync::oneshot::Sender<std::io::Result<()>>>,
    result: std::io::Result<()>,
) {
    if let Some(ack) = first_paint_ack.take() {
        let _ = ack.send(result);
    }
}

fn clone_io_error(err: &std::io::Error) -> std::io::Error {
    std::io::Error::new(err.kind(), err.to_string())
}

#[derive(Debug)]
struct TuiLoopError {
    source: std::io::Error,
    fallback_to_line_handler: bool,
}

impl TuiLoopError {
    fn startup(source: std::io::Error) -> Self {
        Self {
            source,
            fallback_to_line_handler: true,
        }
    }

    fn runtime(source: std::io::Error) -> Self {
        Self {
            source,
            fallback_to_line_handler: false,
        }
    }

    fn should_fallback_to_line_handler(&self) -> bool {
        self.fallback_to_line_handler
    }
}

impl fmt::Display for TuiLoopError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.source.fmt(formatter)
    }
}

impl std::error::Error for TuiLoopError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.source)
    }
}

struct TuiTerminalCleanupGuard {
    armed: bool,
}

impl TuiTerminalCleanupGuard {
    fn armed() -> Self {
        Self { armed: true }
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for TuiTerminalCleanupGuard {
    fn drop(&mut self) {
        if self.armed {
            tracing::warn!("interactive pretty loop unwound before normal terminal cleanup");
            let _ = crate::cli::output::force_restore_tui_terminal();
            let _ = disable_raw_mode();
        }
    }
}

fn restore_tui_terminal_after_loop<F, D>(
    worker_cleanup: std::io::Result<()>,
    mut force_restore: F,
    disable_raw: D,
) -> (std::io::Result<()>, std::io::Result<()>)
where
    F: FnMut() -> std::io::Result<()>,
    D: FnOnce() -> std::io::Result<()>,
{
    let exit_result = worker_cleanup.or_else(|err| {
        tracing::warn!("interactive pretty loop worker cleanup failed: {err}");
        force_restore()
    });
    let raw_result = disable_raw();
    if let Err(err) = &raw_result {
        tracing::warn!("interactive pretty loop raw-mode cleanup failed: {err}");
        let _ = force_restore();
    }

    (exit_result, raw_result)
}

fn read_tui_event(event: Event) -> Option<TuiEvent> {
    match event {
        Event::Resize(columns, rows) => Some(TuiEvent::Resize { columns, rows }),
        Event::Key(KeyEvent {
            code,
            modifiers,
            kind: KeyEventKind::Press,
            ..
        }) => map_key_event(code, modifiers),
        Event::Mouse(MouseEvent {
            kind: MouseEventKind::Down(MouseButton::Left),
            column,
            row,
            ..
        }) => Some(TuiEvent::MouseDown { column, row }),
        Event::Mouse(MouseEvent {
            kind: MouseEventKind::ScrollUp,
            ..
        }) => Some(TuiEvent::Key(TuiKeyEvent::PageUp)),
        Event::Mouse(MouseEvent {
            kind: MouseEventKind::ScrollDown,
            ..
        }) => Some(TuiEvent::Key(TuiKeyEvent::PageDown)),
        _ => None,
    }
}

fn map_key_event(code: KeyCode, modifiers: KeyModifiers) -> Option<TuiEvent> {
    let key = match code {
        KeyCode::Tab => TuiKeyEvent::Tab,
        KeyCode::BackTab => TuiKeyEvent::BackTab,
        KeyCode::Backspace => TuiKeyEvent::Backspace,
        KeyCode::Enter => TuiKeyEvent::Enter,
        KeyCode::Esc => TuiKeyEvent::Escape,
        KeyCode::Left => TuiKeyEvent::Left,
        KeyCode::Right => TuiKeyEvent::Right,
        KeyCode::Up => TuiKeyEvent::Up,
        KeyCode::Down => TuiKeyEvent::Down,
        KeyCode::PageUp => TuiKeyEvent::PageUp,
        KeyCode::PageDown => TuiKeyEvent::PageDown,
        KeyCode::Char('c') if modifiers.contains(KeyModifiers::CONTROL) => TuiKeyEvent::Interrupt,
        KeyCode::Char(_ch) if modifiers.contains(KeyModifiers::CONTROL) => return None,
        KeyCode::Char(ch) => TuiKeyEvent::Char(ch),
        _ => return None,
    };
    Some(TuiEvent::Key(key))
}

#[cfg(test)]
mod tests {
    use super::{
        console_session_mode_for_term, interactive_entry_kind, map_key_event,
        maybe_write_initial_prompt, parse_command, read_tui_event, restore_tui_terminal_after_loop,
        write_ready_prompt, InitialPromptMode, InteractiveCommand, InteractiveEntryKind,
        TuiLoopError, HELP_TEXT, READY_PROMPT,
    };
    use crate::cli::output::{ConsoleSessionMode, TuiEvent, TuiKeyEvent};
    use crossterm::event::{Event, KeyCode, KeyModifiers, MouseButton, MouseEvent, MouseEventKind};

    #[test]
    fn parse_command_accepts_supported_shortcuts() {
        assert_eq!(parse_command("h"), Some(InteractiveCommand::Help));
        assert_eq!(parse_command("q"), Some(InteractiveCommand::Quit));
        assert_eq!(parse_command("i"), Some(InteractiveCommand::Info));
    }

    #[test]
    fn parse_command_trims_surrounding_whitespace() {
        assert_eq!(parse_command("  h  \n"), Some(InteractiveCommand::Help));
        assert_eq!(parse_command("\ti\t"), Some(InteractiveCommand::Info));
    }

    #[test]
    fn parse_command_rejects_other_inputs() {
        assert_eq!(parse_command(""), None);
        assert_eq!(parse_command("help"), None);
        assert_eq!(parse_command("x"), None);
        assert_eq!(HELP_TEXT, "help: h=help, q=quit, i=info snapshot");
    }

    #[test]
    fn ready_prompt_is_exact_raw_prompt_bytes() {
        let mut output = Vec::new();
        write_ready_prompt(&mut output).expect("prompt write should succeed");
        assert_eq!(output, READY_PROMPT.as_bytes());
        assert_eq!(READY_PROMPT, "> ");
    }

    #[test]
    fn deferred_initial_prompt_does_not_write_immediately() {
        let mut output = Vec::new();
        maybe_write_initial_prompt(&mut output, InitialPromptMode::Deferred)
            .expect("deferred prompt should be a no-op");
        assert!(output.is_empty());
    }

    #[test]
    fn immediate_initial_prompt_writes_prompt_bytes() {
        let mut output = Vec::new();
        maybe_write_initial_prompt(&mut output, InitialPromptMode::Immediate)
            .expect("immediate prompt should write prompt bytes");
        assert_eq!(output, READY_PROMPT.as_bytes());
    }

    #[test]
    fn parse_command_quit_alias_still_supported_in_fallback_mode() {
        assert_eq!(parse_command("q\n"), Some(InteractiveCommand::Quit));
    }

    #[test]
    fn tui_uses_interactive_mode_only_when_stdin_and_stderr_are_ttys() {
        assert_eq!(
            console_session_mode_for_term(true, true, Some("xterm-256color")),
            ConsoleSessionMode::InteractiveDashboard
        );
        assert_eq!(
            console_session_mode_for_term(true, false, Some("xterm-256color")),
            ConsoleSessionMode::Fallback
        );
        assert_eq!(
            console_session_mode_for_term(false, true, Some("xterm-256color")),
            ConsoleSessionMode::Fallback
        );
        assert_eq!(
            console_session_mode_for_term(false, false, Some("xterm-256color")),
            ConsoleSessionMode::Fallback
        );
    }

    #[test]
    fn interactive_entry_kind_matches_console_session_mode() {
        assert_eq!(
            interactive_entry_kind(Some(ConsoleSessionMode::InteractiveDashboard)),
            InteractiveEntryKind::Tui
        );
        assert_eq!(
            interactive_entry_kind(Some(ConsoleSessionMode::Fallback)),
            InteractiveEntryKind::Line
        );
        assert_eq!(interactive_entry_kind(None), InteractiveEntryKind::Line);
    }

    #[test]
    fn tui_falls_back_for_unsupported_terminals() {
        assert_eq!(
            console_session_mode_for_term(true, true, Some("dumb")),
            ConsoleSessionMode::Fallback
        );
        assert_eq!(
            console_session_mode_for_term(true, true, Some("")),
            ConsoleSessionMode::Fallback
        );
        assert_eq!(
            console_session_mode_for_term(true, true, None),
            ConsoleSessionMode::Fallback
        );
    }

    #[test]
    fn tui_maps_ctrl_c_to_interrupt_quit() {
        assert_eq!(
            map_key_event(KeyCode::Char('c'), KeyModifiers::CONTROL),
            Some(TuiEvent::Key(TuiKeyEvent::Interrupt))
        );
    }

    #[test]
    fn tui_ignores_other_control_chars() {
        assert_eq!(
            map_key_event(KeyCode::Char('l'), KeyModifiers::CONTROL),
            None
        );
    }

    #[test]
    fn tui_maps_left_mouse_down_to_click_coordinates() {
        assert_eq!(
            read_tui_event(Event::Mouse(MouseEvent {
                kind: MouseEventKind::Down(MouseButton::Left),
                column: 42,
                row: 7,
                modifiers: KeyModifiers::empty(),
            })),
            Some(TuiEvent::MouseDown { column: 42, row: 7 })
        );
    }

    #[test]
    fn tui_maps_mouse_wheel_to_page_navigation() {
        assert_eq!(
            read_tui_event(Event::Mouse(MouseEvent {
                kind: MouseEventKind::ScrollUp,
                column: 42,
                row: 7,
                modifiers: KeyModifiers::empty(),
            })),
            Some(TuiEvent::Key(TuiKeyEvent::PageUp))
        );
        assert_eq!(
            read_tui_event(Event::Mouse(MouseEvent {
                kind: MouseEventKind::ScrollDown,
                column: 42,
                row: 7,
                modifiers: KeyModifiers::empty(),
            })),
            Some(TuiEvent::Key(TuiKeyEvent::PageDown))
        );
    }

    #[test]
    fn tui_cleanup_force_restores_when_worker_cleanup_fails() {
        let mut force_restore_calls = 0;
        let (exit_result, raw_result) = restore_tui_terminal_after_loop(
            Err(std::io::Error::other("worker cleanup failed")),
            || {
                force_restore_calls += 1;
                Ok(())
            },
            || Ok(()),
        );

        assert!(exit_result.is_ok());
        assert!(raw_result.is_ok());
        assert_eq!(force_restore_calls, 1);
    }

    #[test]
    fn tui_cleanup_force_restores_again_when_raw_mode_cleanup_fails() {
        let mut force_restore_calls = 0;
        let (exit_result, raw_result) = restore_tui_terminal_after_loop(
            Ok(()),
            || {
                force_restore_calls += 1;
                Ok(())
            },
            || Err(std::io::Error::other("raw cleanup failed")),
        );

        assert!(exit_result.is_ok());
        assert!(raw_result.is_err());
        assert_eq!(force_restore_calls, 1);
    }

    #[test]
    fn tui_startup_errors_request_line_handler_fallback() {
        assert!(
            TuiLoopError::startup(std::io::Error::other("raw mode failed"))
                .should_fallback_to_line_handler()
        );
        assert!(
            !TuiLoopError::runtime(std::io::Error::other("event read failed"))
                .should_fallback_to_line_handler()
        );
    }
}
