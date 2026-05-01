use anyhow::Result;
use std::ffi::OsStr;
use std::io::{self, IsTerminal, Write};
use std::process::{Command, Stdio};

const DEFAULT_PAGER: &str = "less";
const DEFAULT_PAGER_ARGS: &[&str] = &["-F", "-R", "-X"];

pub(crate) fn print_or_page(output: &str) -> Result<()> {
    if !should_use_pager(
        std::io::stdin().is_terminal(),
        std::io::stdout().is_terminal(),
        std::env::var_os("TERM").as_deref(),
    ) {
        return print_direct(output);
    }

    match page_with_less(output) {
        Ok(()) => Ok(()),
        Err(err) if pager_missing(&err) => print_direct(output),
        Err(err) => Err(err),
    }
}

fn should_use_pager(
    stdin_is_terminal: bool,
    stdout_is_terminal: bool,
    term: Option<&OsStr>,
) -> bool {
    stdin_is_terminal
        && stdout_is_terminal
        && term.is_none_or(|value| !value.eq_ignore_ascii_case(OsStr::new("dumb")))
}

fn print_direct(output: &str) -> Result<()> {
    let mut stdout = io::stdout().lock();
    stdout.write_all(output.as_bytes())?;
    stdout.flush()?;
    Ok(())
}

fn page_with_less(output: &str) -> Result<()> {
    let mut child = Command::new(DEFAULT_PAGER)
        .args(DEFAULT_PAGER_ARGS)
        .stdin(Stdio::piped())
        .spawn()?;

    if let Some(mut stdin) = child.stdin.take() {
        match stdin.write_all(output.as_bytes()) {
            Ok(()) => {}
            Err(err) if err.kind() == io::ErrorKind::BrokenPipe => {}
            Err(err) => return Err(err.into()),
        }
    }

    let _ = child.wait()?;
    Ok(())
}

fn pager_missing(err: &anyhow::Error) -> bool {
    err.downcast_ref::<io::Error>()
        .is_some_and(|io_err| io_err.kind() == io::ErrorKind::NotFound)
}

#[cfg(test)]
mod tests {
    use super::should_use_pager;
    use std::ffi::OsStr;

    #[test]
    fn pager_requires_tty_input_and_output() {
        assert!(!should_use_pager(
            false,
            true,
            Some(OsStr::new("xterm-256color"))
        ));
        assert!(!should_use_pager(
            true,
            false,
            Some(OsStr::new("xterm-256color"))
        ));
        assert!(should_use_pager(
            true,
            true,
            Some(OsStr::new("xterm-256color"))
        ));
    }

    #[test]
    fn pager_skips_dumb_terminals() {
        assert!(!should_use_pager(true, true, Some(OsStr::new("dumb"))));
        assert!(should_use_pager(true, true, None));
    }
}
