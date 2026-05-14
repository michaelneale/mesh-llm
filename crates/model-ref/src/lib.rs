use std::{error::Error, fmt, path::Path, str::FromStr};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelRef {
    pub repo: String,
    pub revision: Option<String>,
    pub selector: Option<String>,
}

impl ModelRef {
    pub fn parse(input: &str) -> Result<Self, ModelRefParseError> {
        parse_model_ref(input)
    }

    pub fn display_id(&self) -> String {
        format_model_ref(
            &self.repo,
            self.revision.as_deref(),
            self.selector.as_deref(),
        )
    }
}

impl fmt::Display for ModelRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.display_id())
    }
}

impl FromStr for ModelRef {
    type Err = ModelRefParseError;

    fn from_str(input: &str) -> Result<Self, Self::Err> {
        Self::parse(input)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelRefParseError {
    input: String,
}

impl ModelRefParseError {
    fn new(input: &str) -> Self {
        Self {
            input: input.to_string(),
        }
    }
}

impl fmt::Display for ModelRefParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "expected model ref like org/repo, org/repo:Q4_K_M, or org/repo@rev:Q4_K_M: {}",
            self.input
        )
    }
}

impl Error for ModelRefParseError {}

pub fn parse_model_ref(input: &str) -> Result<ModelRef, ModelRefParseError> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Err(ModelRefParseError::new(input));
    }

    if let Some(parsed) = parse_huggingface_repo_url(trimmed) {
        return Ok(parsed);
    }

    parse_huggingface_repo_ref(trimmed).ok_or_else(|| ModelRefParseError::new(input))
}

pub fn format_model_ref(repo: &str, revision: Option<&str>, selector: Option<&str>) -> String {
    match (revision, selector) {
        (Some(revision), Some(selector)) => format!("{repo}@{revision}:{selector}"),
        (Some(revision), None) => format!("{repo}@{revision}"),
        (None, Some(selector)) => format!("{repo}:{selector}"),
        (None, None) => repo.to_string(),
    }
}

pub fn format_gguf_selection_ref(repo: &str, file: &str, selector: &str) -> String {
    let directory_selector = Path::new(file)
        .parent()
        .and_then(Path::to_str)
        .filter(|directory| !directory.is_empty() && *directory != ".");
    format_model_ref(repo, None, directory_selector.or(Some(selector)))
}

pub fn format_canonical_ref(repo: &str, revision: &str, file: &str) -> String {
    format!("{repo}@{revision}/{file}")
}

pub fn quant_selector_from_gguf_file(file: &str) -> Option<String> {
    if !file.ends_with(".gguf") {
        return None;
    }

    if let Some((prefix, _)) = file.split_once('/') {
        if is_quant_like_selector(prefix) {
            return Some(prefix.to_string());
        }
    }

    let basename = Path::new(file).file_name()?.to_str()?;
    let mut stem = basename.strip_suffix(".gguf")?;
    if let Some(prefix) = split_gguf_shard_stem_prefix(stem) {
        stem = prefix;
    }

    for marker in [
        "-UD-", ".UD-", "-IQ", ".IQ", "-Q", ".Q", "-BF16", ".BF16", "-F16", ".F16", "-F32", ".F32",
    ] {
        if let Some(pos) = stem.rfind(marker) {
            return Some(stem[pos + 1..].to_string());
        }
    }
    None
}

pub fn normalize_gguf_distribution_id(file: &str) -> Option<String> {
    let basename = Path::new(file).file_name()?.to_str()?;
    let stem = basename.strip_suffix(".gguf")?;
    let stem = split_gguf_shard_stem_prefix(stem).unwrap_or(stem);
    (!stem.is_empty()).then(|| stem.to_string())
}

pub fn is_quant_like_selector(value: &str) -> bool {
    let upper = value.to_ascii_uppercase();
    upper.starts_with("UD-")
        || upper.starts_with('Q')
        || upper.starts_with("IQ")
        || upper == "BF16"
        || upper == "F16"
        || upper == "F32"
}

pub fn gguf_matches_quant_selector(file: &str, selector: &str) -> bool {
    let file_lower = file.to_ascii_lowercase();
    let selector_lower = selector.to_ascii_lowercase();
    if !file_lower.ends_with(".gguf") || selector_lower.is_empty() {
        return false;
    }
    file_lower.contains(&format!("/{selector_lower}/"))
        || file_lower.contains(&format!("-{selector_lower}-"))
        || file_lower.contains(&format!(".{selector_lower}-"))
        || file_lower.ends_with(&format!("-{selector_lower}.gguf"))
        || file_lower.ends_with(&format!(".{selector_lower}.gguf"))
        || file_lower.ends_with(&format!("/{selector_lower}.gguf"))
}

pub fn split_gguf_shard_info(file: &str) -> Option<SplitGgufShard<'_>> {
    let stem = file.strip_suffix(".gguf")?;
    let (prefix_and_part, total) = stem.rsplit_once("-of-")?;
    if total.len() != 5 || !total.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    let (prefix, part) = prefix_and_part.rsplit_once('-')?;
    if part.len() != 5 || !part.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    Some(SplitGgufShard {
        prefix,
        part,
        total,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SplitGgufShard<'a> {
    pub prefix: &'a str,
    pub part: &'a str,
    pub total: &'a str,
}

fn split_gguf_shard_stem_prefix(stem: &str) -> Option<&str> {
    let (prefix_and_part, total) = stem.rsplit_once("-of-")?;
    if total.len() != 5 || !total.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    let (prefix, part) = prefix_and_part.rsplit_once('-')?;
    if part.len() != 5 || !part.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    Some(prefix)
}

fn parse_huggingface_repo_ref(input: &str) -> Option<ModelRef> {
    let parts: Vec<&str> = input.splitn(2, '/').collect();
    if parts.len() != 2 {
        return None;
    }
    if parts[0].is_empty() || parts[1].is_empty() || parts[0].contains(':') {
        return None;
    }
    let (repo_tail, revision, selector) = parse_repo_tail_selector_and_revision(parts[1])?;
    if repo_tail.contains('/') {
        return None;
    }
    Some(ModelRef {
        repo: format!("{}/{}", parts[0], repo_tail),
        revision,
        selector,
    })
}

fn parse_huggingface_repo_url(input: &str) -> Option<ModelRef> {
    let tail = input
        .strip_prefix("https://huggingface.co/")
        .or_else(|| input.strip_prefix("http://huggingface.co/"))?;
    let clean = tail
        .split_once('?')
        .map(|(left, _)| left)
        .unwrap_or(tail)
        .split_once('#')
        .map(|(left, _)| left)
        .unwrap_or(tail)
        .trim_matches('/');
    let parts: Vec<&str> = clean.split('/').collect();
    if parts.len() < 2 || parts[0].is_empty() || parts[1].is_empty() {
        return None;
    }
    let (repo_tail, revision, selector) = parse_repo_tail_selector_and_revision(parts[1])?;
    let repo = format!("{}/{}", parts[0], repo_tail);
    if parts.len() >= 4 && parts[2] == "tree" && !parts[3].is_empty() {
        return Some(ModelRef {
            repo,
            revision: Some(parts[3].to_string()),
            selector,
        });
    }
    if parts.len() == 2 {
        return Some(ModelRef {
            repo,
            revision,
            selector,
        });
    }
    None
}

fn parse_repo_tail_selector_and_revision(
    tail: &str,
) -> Option<(String, Option<String>, Option<String>)> {
    let at_pos = tail.find('@');
    let colon_pos = tail.find(':');

    match (at_pos, colon_pos) {
        (Some(at), Some(colon)) if at < colon => {
            let repo_tail = &tail[..at];
            let revision = &tail[at + 1..colon];
            let selector = &tail[colon + 1..];
            nonempty_tail(repo_tail, Some(revision), Some(selector))
        }
        (Some(at), Some(colon)) if colon < at => {
            let repo_tail = &tail[..colon];
            let selector = &tail[colon + 1..at];
            let revision = &tail[at + 1..];
            nonempty_tail(repo_tail, Some(revision), Some(selector))
        }
        (Some(at), None) => {
            let repo_tail = &tail[..at];
            let revision = &tail[at + 1..];
            nonempty_tail(repo_tail, Some(revision), None)
        }
        (None, Some(colon)) => {
            let repo_tail = &tail[..colon];
            let selector = &tail[colon + 1..];
            nonempty_tail(repo_tail, None, Some(selector))
        }
        (None, None) if !tail.is_empty() => Some((tail.to_string(), None, None)),
        _ => None,
    }
}

fn nonempty_tail(
    repo_tail: &str,
    revision: Option<&str>,
    selector: Option<&str>,
) -> Option<(String, Option<String>, Option<String>)> {
    if repo_tail.is_empty()
        || revision.is_some_and(str::is_empty)
        || selector.is_some_and(str::is_empty)
    {
        return None;
    }
    Some((
        repo_tail.to_string(),
        revision.map(str::to_string),
        selector.map(str::to_string),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_public_model_refs() {
        assert_eq!(
            parse_model_ref("org/repo:Q4_K_M").unwrap(),
            ModelRef {
                repo: "org/repo".to_string(),
                revision: None,
                selector: Some("Q4_K_M".to_string()),
            }
        );

        assert_eq!(
            parse_model_ref("org/repo@abc123:Q4_K_M")
                .unwrap()
                .display_id(),
            "org/repo@abc123:Q4_K_M"
        );

        assert_eq!(
            parse_model_ref("org/repo:Q4_K_M@abc123")
                .unwrap()
                .display_id(),
            "org/repo@abc123:Q4_K_M"
        );
    }

    #[test]
    fn rejects_mesh_exact_file_refs() {
        assert!(parse_model_ref("org/repo/file.gguf").is_err());
        assert!(parse_model_ref("org/repo/model.safetensors").is_err());
        assert!(parse_model_ref("https://huggingface.co/org/repo/resolve/main/file.gguf").is_err());
    }

    #[test]
    fn parses_huggingface_repo_urls() {
        assert_eq!(
            parse_model_ref("https://huggingface.co/org/repo:BF16")
                .unwrap()
                .display_id(),
            "org/repo:BF16"
        );
        assert_eq!(
            parse_model_ref("https://huggingface.co/org/repo/tree/main")
                .unwrap()
                .display_id(),
            "org/repo@main"
        );
    }

    #[test]
    fn rejects_empty_or_non_repo_refs() {
        assert!(parse_model_ref("").is_err());
        assert!(parse_model_ref("repo-only").is_err());
        assert!(parse_model_ref("org/:Q4_K_M").is_err());
        assert!(parse_model_ref("org/repo:").is_err());
    }

    #[test]
    fn extracts_quant_selectors_from_gguf_files() {
        assert_eq!(
            quant_selector_from_gguf_file("gemma-4-31B-it-UD-Q4_K_XL.gguf"),
            Some("UD-Q4_K_XL".to_string())
        );
        assert_eq!(
            quant_selector_from_gguf_file("Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf"),
            Some("Q4_K_M".to_string())
        );
        assert_eq!(
            quant_selector_from_gguf_file("BF16/gemma-4-31B-it-BF16-00001-of-00002.gguf"),
            Some("BF16".to_string())
        );
        assert_eq!(
            quant_selector_from_gguf_file("qwen3.5-moe-0.87B-d0.8B.Q2_K.gguf"),
            Some("Q2_K".to_string())
        );
        assert_eq!(
            quant_selector_from_gguf_file("gemma-4-31B-it-Q4_0.gguf"),
            Some("Q4_0".to_string())
        );
    }

    #[test]
    fn normalizes_distribution_ids() {
        assert_eq!(
            normalize_gguf_distribution_id("GLM-5.1-UD-IQ2_M-00001-of-00006.gguf"),
            Some("GLM-5.1-UD-IQ2_M".to_string())
        );
        assert_eq!(
            normalize_gguf_distribution_id("Qwen3-30B-A3B-Q4_K_M.gguf"),
            Some("Qwen3-30B-A3B-Q4_K_M".to_string())
        );
        assert_eq!(
            normalize_gguf_distribution_id("UD-IQ2_M/GLM-5.1-UD-IQ2_M-00001-of-00006.gguf"),
            Some("GLM-5.1-UD-IQ2_M".to_string())
        );
        assert_eq!(normalize_gguf_distribution_id("README.md"), None);
    }

    #[test]
    fn matches_quant_selectors_against_gguf_paths() {
        assert!(gguf_matches_quant_selector(
            "UD-Q4_K_XL/gemma-4-31B-it-UD-Q4_K_XL-00001-of-00004.gguf",
            "UD-Q4_K_XL"
        ));
        assert!(gguf_matches_quant_selector(
            "Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf",
            "Q4_K_M"
        ));
        assert!(!gguf_matches_quant_selector(
            "Meta-Llama-3.1-8B-Instruct.Q5_K_M.gguf",
            "Q4_K_M"
        ));
    }

    #[test]
    fn formats_canonical_refs() {
        assert_eq!(
            format_canonical_ref("org/repo", "abc123", "model.gguf"),
            "org/repo@abc123/model.gguf"
        );
    }

    #[test]
    fn formats_selected_gguf_refs() {
        assert_eq!(
            format_gguf_selection_ref("unsloth/Qwen3-8B-GGUF", "Qwen3-8B-Q4_K_M.gguf", "Q4_K_M"),
            "unsloth/Qwen3-8B-GGUF:Q4_K_M"
        );
        assert_eq!(
            format_gguf_selection_ref(
                "unsloth/LTX-2.3-GGUF",
                "distilled-1.1/LTX-2.3-UD-Q4_K_M.gguf",
                "UD-Q4_K_M"
            ),
            "unsloth/LTX-2.3-GGUF:distilled-1.1"
        );
    }

    #[test]
    fn parses_split_gguf_shard_info() {
        assert_eq!(
            split_gguf_shard_info("Qwen3-30B-A3B-Q4_K_M-00001-of-00004.gguf"),
            Some(SplitGgufShard {
                prefix: "Qwen3-30B-A3B-Q4_K_M",
                part: "00001",
                total: "00004",
            })
        );
        assert_eq!(split_gguf_shard_info("Qwen3-30B-A3B-Q4_K_M.gguf"), None);
    }
}
