use std::path::Path;

use anyhow::{anyhow, Result};

/// Parses a Hugging Face resolve URL into repo, revision, and file path.
pub fn parse_hf_resolve_url(url: &str) -> Option<(String, Option<String>, String)> {
    let tail = url
        .strip_prefix("https://huggingface.co/")
        .or_else(|| url.strip_prefix("http://huggingface.co/"))?;
    let parts: Vec<&str> = tail.split('/').collect();
    if parts.len() < 5 || parts.get(2) != Some(&"resolve") {
        return None;
    }
    Some((
        format!("{}/{}", parts[0], parts[1]),
        parts.get(3).map(|value| value.to_string()),
        parts[4..].join("/"),
    ))
}

/// Parses an exact Hugging Face file reference.
///
/// Accepted forms include:
/// - `https://huggingface.co/org/repo/resolve/rev/file.gguf`
/// - `org/repo/file.gguf`
/// - `org/repo@rev/file.gguf`
pub fn parse_huggingface_file_ref(input: &str) -> Option<(String, Option<String>, String)> {
    if let Some(parsed) = parse_hf_resolve_url(input) {
        return Some(parsed);
    }

    let parts: Vec<&str> = input.splitn(3, '/').collect();
    if parts.len() != 3 {
        return None;
    }
    if parts[0].is_empty() || parts[1].is_empty() || parts[0].contains(':') {
        return None;
    }
    let (repo_tail, revision) = match parts[1].split_once('@') {
        Some((repo, revision)) => (repo, Some(revision.to_string())),
        None => (parts[1], None),
    };
    if repo_tail.is_empty() {
        return None;
    }
    Some((
        format!("{}/{}", parts[0], repo_tail),
        revision,
        parts[2].to_string(),
    ))
}

pub fn parse_huggingface_repo_ref(input: &str) -> Option<(String, Option<String>, Option<String>)> {
    if input.starts_with("http://") || input.starts_with("https://") {
        return None;
    }
    parse_model_ref_tuple(input)
}

pub fn parse_huggingface_repo_url(input: &str) -> Option<(String, Option<String>, Option<String>)> {
    if !input.starts_with("https://huggingface.co/") && !input.starts_with("http://huggingface.co/")
    {
        return None;
    }
    parse_model_ref_tuple(input)
}

fn parse_model_ref_tuple(input: &str) -> Option<(String, Option<String>, Option<String>)> {
    let parsed = model_ref::ModelRef::parse(input).ok()?;
    if parsed.repo.split('/').count() != 2 {
        return None;
    }
    Some((parsed.repo, parsed.revision, parsed.selector))
}

pub fn is_split_mlx_first_shard(file: &str) -> bool {
    let Some(rest) = file.strip_prefix("model-") else {
        return false;
    };
    let Some(rest) = rest.strip_suffix(".safetensors") else {
        return false;
    };
    let Some((left, right)) = rest.split_once("-of-") else {
        return false;
    };
    left == "00001" && right.len() == 5 && right.bytes().all(|byte| byte.is_ascii_digit())
}

pub fn is_primary_mlx_weight_file(file: &str) -> bool {
    let basename = file.rsplit('/').next().unwrap_or(file);
    basename.eq_ignore_ascii_case("model.safetensors") || is_split_mlx_first_shard(basename)
}

pub fn format_huggingface_exact_ref(repo: &str, revision: Option<&str>, file: &str) -> String {
    match revision {
        Some(revision) => format!("{repo}@{revision}/{file}"),
        None => format!("{repo}/{file}"),
    }
}

pub fn format_huggingface_display_ref(repo: &str, revision: Option<&str>, file: &str) -> String {
    if let Some(selector) = model_ref::quant_selector_from_gguf_file(file) {
        return model_ref::format_model_ref(repo, revision, Some(&selector));
    }
    if is_primary_mlx_weight_file(file) {
        return match revision {
            Some(revision) => format!("{repo}@{revision}"),
            None => repo.to_string(),
        };
    }
    format_huggingface_exact_ref(repo, revision, file)
}

pub fn huggingface_resolve_url(repo: &str, revision: Option<&str>, file: &str) -> String {
    let revision = revision.unwrap_or("main");
    format!("https://huggingface.co/{repo}/resolve/{revision}/{file}")
}

pub fn remote_filename(input: &str) -> Result<String> {
    input
        .rsplit('/')
        .next()
        .filter(|name| !name.is_empty())
        .map(str::to_string)
        .ok_or_else(|| anyhow!("Cannot extract filename from URL: {input}"))
}

pub fn file_basename(input: &str) -> &str {
    Path::new(input)
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or(input)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_hf_resolve_url_with_nested_file() {
        let parsed = parse_hf_resolve_url(
            "https://huggingface.co/org/repo/resolve/main/subdir/model-Q4_K_M.gguf",
        )
        .unwrap();

        assert_eq!(parsed.0, "org/repo");
        assert_eq!(parsed.1.as_deref(), Some("main"));
        assert_eq!(parsed.2, "subdir/model-Q4_K_M.gguf");
    }

    #[test]
    fn parses_exact_file_refs() {
        assert_eq!(
            parse_huggingface_file_ref("org/repo@rev/model.gguf"),
            Some((
                "org/repo".to_string(),
                Some("rev".to_string()),
                "model.gguf".to_string()
            ))
        );
    }

    #[test]
    fn formats_quant_and_mlx_display_refs() {
        assert_eq!(
            format_huggingface_display_ref("org/repo", Some("main"), "repo-Q4_K_M.gguf"),
            "org/repo@main:Q4_K_M"
        );
        assert_eq!(
            format_huggingface_display_ref("org/repo", Some("main"), "model.safetensors"),
            "org/repo@main"
        );
    }

    #[test]
    fn extracts_remote_filename() {
        assert_eq!(
            remote_filename("https://example.test/a/model.gguf").unwrap(),
            "model.gguf"
        );
        assert!(remote_filename("https://example.test/a/").is_err());
    }
}
