struct PromptHistory {
    prompts: Vec<String>,
    path: Option<PathBuf>,
}

impl PromptHistory {
    fn load(path: Option<&Path>) -> Result<Self> {
        let prompts = match path {
            Some(path) if path.exists() => {
                let file = fs::File::open(path)
                    .with_context(|| format!("open prompt history {}", path.display()))?;
                io::BufReader::new(file)
                    .lines()
                    .collect::<io::Result<Vec<_>>>()
                    .with_context(|| format!("read prompt history {}", path.display()))?
                    .into_iter()
                    .filter(|line| !line.trim().is_empty())
                    .collect()
            }
            _ => Vec::new(),
        };
        Ok(Self {
            prompts,
            path: path.map(Path::to_path_buf),
        })
    }

    fn push(&mut self, prompt: &str) -> Result<()> {
        if self.prompts.last().is_some_and(|last| last == prompt) {
            return Ok(());
        }
        self.prompts.push(prompt.to_string());
        if let Some(path) = self.path.as_ref() {
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent)
                    .with_context(|| format!("create prompt history dir {}", parent.display()))?;
            }
            let mut file = fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
                .with_context(|| format!("open prompt history {}", path.display()))?;
            writeln!(file, "{prompt}")
                .with_context(|| format!("append prompt history {}", path.display()))?;
        }
        Ok(())
    }

    fn get(&self, one_based_index: usize) -> Option<&str> {
        one_based_index
            .checked_sub(1)
            .and_then(|index| self.prompts.get(index))
            .map(String::as_str)
    }

    fn print(&self) {
        for (index, prompt) in self.prompts.iter().enumerate() {
            println!("{:>4}: {}", index + 1, prompt);
        }
    }
}

enum PromptInput {
    Interactive(Box<DefaultEditor>),
    Stdin(io::Lines<BufReader<io::Stdin>>),
}

impl PromptInput {
    fn add_history_entry(&mut self, input: &str) {
        if let Self::Interactive(editor) = self {
            let _ = editor.add_history_entry(input);
        }
    }
}

fn prompt_input(history: &PromptHistory) -> Result<PromptInput> {
    if io::stdin().is_terminal() {
        return Ok(PromptInput::Interactive(Box::new(prompt_editor(history)?)));
    }
    Ok(PromptInput::Stdin(BufReader::new(io::stdin()).lines()))
}

fn prompt_editor(history: &PromptHistory) -> Result<DefaultEditor> {
    let mut editor = DefaultEditor::new().context("create prompt line editor")?;
    for prompt in &history.prompts {
        let _ = editor.add_history_entry(prompt);
    }
    Ok(editor)
}

fn read_history_prompt(input: &mut PromptInput, prompt: &str) -> Result<Option<String>> {
    match input {
        PromptInput::Interactive(editor) => match editor.readline(prompt) {
            Ok(input) => Ok(Some(input)),
            Err(ReadlineError::Interrupted) => {
                println!("^C");
                Ok(Some(String::new()))
            }
            Err(ReadlineError::Eof) => {
                println!();
                Ok(None)
            }
            Err(error) => Err(anyhow!(error)).context("read prompt line"),
        },
        PromptInput::Stdin(lines) => match lines.next() {
            Some(Ok(line)) => Ok(Some(line)),
            Some(Err(error)) => Err(error).context("read prompt line from stdin"),
            None => Ok(None),
        },
    }
}

fn parse_prompt_json_command(input: &str) -> Result<Option<(String, String)>> {
    const PREFIX: &str = ":prompt-json\t";
    let Some(rest) = input.strip_prefix(PREFIX) else {
        return Ok(None);
    };
    let (session_json, prompt_json) = rest
        .split_once('\t')
        .context(":prompt-json requires session and prompt JSON strings")?;
    let session_id: String =
        serde_json::from_str(session_json).context("parse :prompt-json session")?;
    let prompt: String = serde_json::from_str(prompt_json).context("parse :prompt-json prompt")?;
    Ok(Some((session_id, prompt)))
}

#[cfg(test)]
mod prompt_json_tests {
    use super::*;

    #[test]
    fn parse_prompt_json_preserves_multiline_prompt() -> Result<()> {
        let command = concat!(
            ":prompt-json\t",
            "\"trajectory-7\"\t",
            "\"fn main() {\\n    println!(\\\"hi\\\");\\n}\""
        );

        let parsed = parse_prompt_json_command(command)?.expect("prompt-json command");
        assert_eq!(parsed.0, "trajectory-7");
        assert_eq!(parsed.1, "fn main() {\n    println!(\"hi\");\n}");
        Ok(())
    }

    #[test]
    fn parse_prompt_json_ignores_regular_prompt() -> Result<()> {
        assert!(parse_prompt_json_command("plain prompt")?.is_none());
        Ok(())
    }

    #[test]
    fn default_max_new_tokens_uses_remaining_context_budget() -> Result<()> {
        assert_eq!(effective_prompt_max_new_tokens(0, 4096, 1024)?, 3072);
        assert_eq!(format_prompt_max_new_tokens(0), "context-budget");
        Ok(())
    }

    #[test]
    fn explicit_max_new_tokens_is_preserved() -> Result<()> {
        assert_eq!(effective_prompt_max_new_tokens(128, 4096, 1024)?, 128);
        assert_eq!(format_prompt_max_new_tokens(128), "128");
        Ok(())
    }

    #[test]
    fn exact_prefix_restore_only_runs_for_one_shot_large_prefixes() {
        assert!(!should_try_exact_prefix_restore(false, 0, 511));
        assert!(should_try_exact_prefix_restore(false, 0, 512));
        assert!(!should_try_exact_prefix_restore(true, 0, 512));
        assert!(!should_try_exact_prefix_restore(false, 1, 512));
    }

    #[test]
    fn live_resident_prefix_match_requires_nonempty_full_resident_prefix() {
        assert!(!live_resident_prefix_matches(&[], &[1, 2, 3], 2));
        assert!(live_resident_prefix_matches(&[1, 2], &[1, 2, 3, 4], 3));
        assert!(!live_resident_prefix_matches(&[1, 9], &[1, 2, 3, 4], 3));
        assert!(!live_resident_prefix_matches(&[1, 2, 3], &[1, 2, 3, 4], 2));
    }
}
