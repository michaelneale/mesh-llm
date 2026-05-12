pub(super) fn run() -> Result<()> {
    match Cli::parse().command {
        CommandKind::Prompt(args) => prompt_repl(*args),
        CommandKind::Binary(args) => binary_repl(*args),
    }
}
