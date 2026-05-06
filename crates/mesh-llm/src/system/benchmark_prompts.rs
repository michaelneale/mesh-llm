use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

const DATASETS_SERVER_BASE: &str = "https://datasets-server.huggingface.co";

#[derive(Clone, Debug)]
pub(crate) struct ImportPromptsArgs {
    pub source: PromptImportSource,
    pub limit: usize,
    pub max_tokens: Option<u32>,
    pub output: PathBuf,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PromptImportSource {
    MtBench,
    Gsm8k,
    Humaneval,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct PromptCorpusEntry {
    pub id: String,
    pub source: String,
    pub category: String,
    pub messages: Vec<PromptMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub grader: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reference_answer: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reference_rationale: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reference_tests: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub entry_point: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct PromptMessage {
    pub role: String,
    pub content: String,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct PromptCorpusSummary {
    pub path: String,
    pub prompt_count: usize,
    pub multi_turn_prompt_count: usize,
    pub categories: BTreeMap<String, usize>,
    pub sources: BTreeMap<String, usize>,
}

#[derive(Debug, Deserialize)]
struct RowsResponse {
    rows: Vec<RowEnvelope>,
}

#[derive(Debug, Deserialize)]
struct RowEnvelope {
    row: serde_json::Value,
}

#[derive(Debug, Deserialize)]
struct SplitsResponse {
    splits: Vec<DatasetSplit>,
}

#[derive(Debug, Deserialize)]
struct DatasetSplit {
    config: String,
    split: String,
}

pub(crate) async fn import_prompt_corpus(args: ImportPromptsArgs) -> Result<()> {
    if args.limit == 0 {
        bail!("--limit must be at least 1");
    }

    let client = reqwest::Client::builder()
        .user_agent(format!("mesh-llm/{}", crate::VERSION))
        .build()
        .context("Build prompt import HTTP client")?;

    let source_spec = args.source.spec();
    let (config, split) = resolve_split(&client, source_spec.dataset, source_spec).await?;
    let prompts = fetch_prompt_entries(
        &client,
        args.source,
        source_spec.dataset,
        &config,
        &split,
        args.limit,
        args.max_tokens,
    )
    .await?;

    if prompts.is_empty() {
        bail!(
            "No prompt rows imported from {} ({config}/{split})",
            source_spec.dataset
        );
    }

    if let Some(parent) = args.output.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("Create prompt output directory {}", parent.display()))?;
    }

    let file = std::fs::File::create(&args.output)
        .with_context(|| format!("Create prompt corpus {}", args.output.display()))?;
    let mut writer = BufWriter::new(file);
    for prompt in &prompts {
        serde_json::to_writer(&mut writer, prompt).context("Serialize prompt corpus entry")?;
        writer
            .write_all(b"\n")
            .context("Write prompt corpus newline")?;
    }
    writer.flush().context("Flush prompt corpus")?;

    let summary = summarize_prompts(&prompts, &args.output);
    eprintln!(
        "📝 Imported {} prompts from {} to {}",
        summary.prompt_count,
        source_spec.dataset,
        args.output.display()
    );
    Ok(())
}

pub(crate) fn summarize_prompts(prompts: &[PromptCorpusEntry], path: &Path) -> PromptCorpusSummary {
    let mut categories = BTreeMap::new();
    let mut sources = BTreeMap::new();
    let mut multi_turn_prompt_count = 0;

    for prompt in prompts {
        *categories.entry(prompt.category.clone()).or_insert(0) += 1;
        *sources.entry(prompt.source.clone()).or_insert(0) += 1;
        if prompt.messages.len() > 1 {
            multi_turn_prompt_count += 1;
        }
    }

    PromptCorpusSummary {
        path: path.display().to_string(),
        prompt_count: prompts.len(),
        multi_turn_prompt_count,
        categories,
        sources,
    }
}

async fn resolve_split(
    client: &reqwest::Client,
    dataset: &str,
    spec: DatasetSpec,
) -> Result<(String, String)> {
    let url = format!("{DATASETS_SERVER_BASE}/splits");
    let response = client
        .get(url)
        .query(&[("dataset", dataset)])
        .send()
        .await
        .with_context(|| format!("Fetch splits for dataset {dataset}"))?
        .error_for_status()
        .with_context(|| format!("Dataset splits request failed for {dataset}"))?
        .json::<SplitsResponse>()
        .await
        .with_context(|| format!("Parse splits response for {dataset}"))?;

    let preferred = response
        .splits
        .iter()
        .find(|split| split.config == spec.config && split.split == spec.split)
        .ok_or_else(|| {
            anyhow!(
                "Dataset {} did not expose expected split {}/{}",
                dataset,
                spec.config,
                spec.split
            )
        })?;

    Ok((preferred.config.clone(), preferred.split.clone()))
}

async fn fetch_prompt_entries(
    client: &reqwest::Client,
    source: PromptImportSource,
    dataset: &str,
    config: &str,
    split: &str,
    limit: usize,
    max_tokens: Option<u32>,
) -> Result<Vec<PromptCorpusEntry>> {
    let mut prompts = Vec::with_capacity(limit);
    let mut offset = 0usize;

    while prompts.len() < limit {
        let remaining = limit - prompts.len();
        let page_len = remaining.min(100);
        let page = fetch_rows_page(client, dataset, config, split, offset, page_len).await?;
        if page.is_empty() {
            break;
        }
        offset += page.len();

        for row in page {
            prompts.push(prompt_from_row(source, row, max_tokens)?);
            if prompts.len() == limit {
                break;
            }
        }
    }

    Ok(prompts)
}

async fn fetch_rows_page(
    client: &reqwest::Client,
    dataset: &str,
    config: &str,
    split: &str,
    offset: usize,
    length: usize,
) -> Result<Vec<serde_json::Value>> {
    let url = format!("{DATASETS_SERVER_BASE}/rows");
    let response = client
        .get(url)
        .query(&[
            ("dataset", dataset),
            ("config", config),
            ("split", split),
            ("offset", &offset.to_string()),
            ("length", &length.to_string()),
        ])
        .send()
        .await
        .with_context(|| {
            format!("Fetch dataset rows for {dataset} ({config}/{split}) at offset {offset}")
        })?
        .error_for_status()
        .with_context(|| {
            format!(
                "Dataset rows request failed for {dataset} ({config}/{split}) at offset {offset}"
            )
        })?
        .json::<RowsResponse>()
        .await
        .with_context(|| {
            format!("Parse rows response for {dataset} ({config}/{split}) at offset {offset}")
        })?;
    Ok(response.rows.into_iter().map(|row| row.row).collect())
}

fn prompt_from_row(
    source: PromptImportSource,
    row: serde_json::Value,
    max_tokens: Option<u32>,
) -> Result<PromptCorpusEntry> {
    match source {
        PromptImportSource::MtBench => mt_bench_prompt_from_row(row, max_tokens),
        PromptImportSource::Gsm8k => gsm8k_prompt_from_row(row, max_tokens),
        PromptImportSource::Humaneval => humaneval_prompt_from_row(row, max_tokens),
    }
}

fn mt_bench_prompt_from_row(
    row: serde_json::Value,
    max_tokens: Option<u32>,
) -> Result<PromptCorpusEntry> {
    #[derive(Deserialize)]
    struct MtBenchRow {
        category: String,
        prompt: Vec<String>,
        prompt_id: u64,
    }

    let row: MtBenchRow = serde_json::from_value(row).context("Parse MT Bench row")?;
    let messages = row
        .prompt
        .into_iter()
        .map(|content| PromptMessage {
            role: "user".to_string(),
            content,
        })
        .collect();

    Ok(PromptCorpusEntry {
        id: format!("mt_bench_{}", row.prompt_id),
        source: "mt_bench".to_string(),
        category: row.category,
        messages,
        grader: None,
        reference_answer: None,
        reference_rationale: None,
        reference_tests: None,
        entry_point: None,
        max_tokens,
    })
}

fn gsm8k_prompt_from_row(
    row: serde_json::Value,
    max_tokens: Option<u32>,
) -> Result<PromptCorpusEntry> {
    #[derive(Deserialize)]
    struct Gsm8kRow {
        question: String,
        answer: String,
    }

    let row: Gsm8kRow = serde_json::from_value(row).context("Parse GSM8K row")?;
    let final_answer = row
        .answer
        .lines()
        .rev()
        .find_map(|line| {
            line.trim()
                .strip_prefix("####")
                .map(|v| v.trim().to_string())
        })
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| row.answer.clone());

    Ok(PromptCorpusEntry {
        id: stable_prompt_id("gsm8k", &row.question),
        source: "gsm8k".to_string(),
        category: "reasoning".to_string(),
        messages: vec![PromptMessage {
            role: "user".to_string(),
            content: row.question,
        }],
        grader: Some("exact_match".to_string()),
        reference_answer: Some(final_answer),
        reference_rationale: Some(row.answer),
        reference_tests: None,
        entry_point: None,
        max_tokens,
    })
}

fn humaneval_prompt_from_row(
    row: serde_json::Value,
    max_tokens: Option<u32>,
) -> Result<PromptCorpusEntry> {
    #[derive(Deserialize)]
    struct HumanEvalRow {
        task_id: String,
        prompt: String,
        test: String,
        entry_point: String,
    }

    let row: HumanEvalRow = serde_json::from_value(row).context("Parse HumanEval row")?;

    Ok(PromptCorpusEntry {
        id: row.task_id.replace('/', "_").to_lowercase(),
        source: "humaneval".to_string(),
        category: "code".to_string(),
        messages: vec![PromptMessage {
            role: "user".to_string(),
            content: row.prompt,
        }],
        grader: Some("python_tests".to_string()),
        reference_answer: None,
        reference_rationale: None,
        reference_tests: Some(row.test),
        entry_point: Some(row.entry_point),
        max_tokens,
    })
}

fn stable_prompt_id(prefix: &str, content: &str) -> String {
    use sha2::{Digest, Sha256};

    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    let digest = hasher.finalize();
    let short = hex::encode(digest)[..12].to_string();
    format!("{prefix}_{short}")
}

#[derive(Clone, Copy)]
struct DatasetSpec {
    dataset: &'static str,
    config: &'static str,
    split: &'static str,
}

impl PromptImportSource {
    fn spec(self) -> DatasetSpec {
        match self {
            Self::MtBench => DatasetSpec {
                dataset: "HuggingFaceH4/mt_bench_prompts",
                config: "default",
                split: "train",
            },
            Self::Gsm8k => DatasetSpec {
                dataset: "openai/gsm8k",
                config: "main",
                split: "test",
            },
            Self::Humaneval => DatasetSpec {
                dataset: "openai/openai_humaneval",
                config: "openai_humaneval",
                split: "test",
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn summarize_prompts_counts_categories_and_sources() {
        let prompts = vec![
            PromptCorpusEntry {
                id: "a".to_string(),
                source: "mt_bench".to_string(),
                category: "chat".to_string(),
                messages: vec![PromptMessage {
                    role: "user".to_string(),
                    content: "one".to_string(),
                }],
                grader: None,
                reference_answer: None,
                reference_rationale: None,
                reference_tests: None,
                entry_point: None,
                max_tokens: None,
            },
            PromptCorpusEntry {
                id: "b".to_string(),
                source: "mt_bench".to_string(),
                category: "chat".to_string(),
                messages: vec![
                    PromptMessage {
                        role: "user".to_string(),
                        content: "one".to_string(),
                    },
                    PromptMessage {
                        role: "user".to_string(),
                        content: "two".to_string(),
                    },
                ],
                grader: None,
                reference_answer: None,
                reference_rationale: None,
                reference_tests: None,
                entry_point: None,
                max_tokens: None,
            },
        ];

        let summary = summarize_prompts(&prompts, Path::new("/tmp/prompts.jsonl"));
        assert_eq!(summary.prompt_count, 2);
        assert_eq!(summary.multi_turn_prompt_count, 1);
        assert_eq!(summary.categories.get("chat"), Some(&2));
        assert_eq!(summary.sources.get("mt_bench"), Some(&2));
    }

    #[test]
    fn gsm8k_answer_extracts_final_answer() {
        let entry = gsm8k_prompt_from_row(
            serde_json::json!({
                "question": "What is 2 + 2?",
                "answer": "Reasoning\n#### 4"
            }),
            Some(128),
        )
        .expect("gsm8k entry");

        assert_eq!(entry.reference_answer.as_deref(), Some("4"));
        assert_eq!(entry.grader.as_deref(), Some("exact_match"));
    }
}
