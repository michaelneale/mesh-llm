use crate::models::{
    capabilities, catalog, huggingface_hub_cache_dir, ModelCapabilities, ModelDetails,
    SearchArtifactFilter, SearchHit, SearchSort,
};
use crate::system::hardware;
use anyhow::Result;
use chrono::{DateTime, Utc};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

pub(crate) trait SearchFormatter {
    fn is_json(&self) -> bool;
    fn render_catalog_empty(
        &self,
        query: &str,
        filter: SearchArtifactFilter,
        sort: SearchSort,
    ) -> Result<()>;
    fn render_catalog_results(
        &self,
        query: &str,
        filter: SearchArtifactFilter,
        results: &[&'static catalog::CatalogModel],
        limit: usize,
        sort: SearchSort,
    ) -> Result<()>;
    fn render_hf_empty(
        &self,
        query: &str,
        filter: SearchArtifactFilter,
        sort: SearchSort,
    ) -> Result<()>;
    fn render_hf_results(
        &self,
        query: &str,
        filter: SearchArtifactFilter,
        sort: SearchSort,
        results: &[SearchHit],
    ) -> Result<()>;
}

#[derive(Clone)]
pub(crate) struct InstalledRow {
    pub(crate) name: String,
    pub(crate) model_ref: String,
    pub(crate) path: PathBuf,
    pub(crate) size: Option<u64>,
    pub(crate) catalog_model: Option<&'static catalog::CatalogModel>,
    pub(crate) capabilities: ModelCapabilities,
    pub(crate) managed_by_mesh: bool,
    pub(crate) last_used_at: Option<String>,
}

pub(crate) trait ModelsFormatter: SearchFormatter {
    fn render_recommended(&self, models: &[&'static catalog::CatalogModel]) -> Result<()>;
    fn render_installed(&self, rows: &[InstalledRow]) -> Result<()>;
    fn render_show(&self, details: &ModelDetails, variants: Option<&[ModelDetails]>) -> Result<()>;
    fn render_download(
        &self,
        model_ref: &str,
        path: &Path,
        details: Option<&ModelDetails>,
        include_draft: bool,
        draft: Option<(&str, &Path)>,
    ) -> Result<()>;
    fn render_updates_status(&self, repo: Option<&str>, all: bool, check: bool) -> Result<()>;
}

pub(crate) struct ConsoleFormatter;
pub(crate) struct JsonFormatter;

pub(crate) fn search_formatter(json_output: bool) -> Box<dyn SearchFormatter> {
    if json_output {
        Box::new(JsonFormatter)
    } else {
        Box::new(ConsoleFormatter)
    }
}

pub(crate) fn models_formatter(json_output: bool) -> Box<dyn ModelsFormatter> {
    if json_output {
        Box::new(JsonFormatter)
    } else {
        Box::new(ConsoleFormatter)
    }
}

pub(crate) fn filter_label(filter: SearchArtifactFilter) -> &'static str {
    match filter {
        SearchArtifactFilter::Gguf => "GGUF",
        SearchArtifactFilter::Mlx => "MLX",
    }
}

pub(crate) fn filter_name(filter: SearchArtifactFilter) -> &'static str {
    match filter {
        SearchArtifactFilter::Gguf => "gguf",
        SearchArtifactFilter::Mlx => "mlx",
    }
}

pub(crate) fn sort_label(sort: SearchSort) -> &'static str {
    match sort {
        SearchSort::Trending => "trending",
        SearchSort::Downloads => "most downloads",
        SearchSort::Likes => "most likes",
        SearchSort::Created => "recently created",
        SearchSort::Updated => "recently updated",
        SearchSort::ParametersDesc => "most parameters",
        SearchSort::ParametersAsc => "least parameters",
    }
}

pub(crate) fn sort_name(sort: SearchSort) -> &'static str {
    match sort {
        SearchSort::Trending => "trending",
        SearchSort::Downloads => "downloads",
        SearchSort::Likes => "likes",
        SearchSort::Created => "created",
        SearchSort::Updated => "updated",
        SearchSort::ParametersDesc => "most_parameters",
        SearchSort::ParametersAsc => "least_parameters",
    }
}

pub(crate) fn print_json(value: Value) -> Result<()> {
    println!("{}", serde_json::to_string_pretty(&value)?);
    Ok(())
}

pub(crate) fn huggingface_repo_url(repo_id: &str) -> String {
    format!("https://huggingface.co/{repo_id}")
}

pub(crate) fn format_installed_size(bytes: u64) -> String {
    if bytes >= 1_000_000_000 {
        format!("{:.1}GB", bytes as f64 / 1e9)
    } else if bytes >= 1_000_000 {
        format!("{:.0}MB", bytes as f64 / 1e6)
    } else {
        format!("{}B", bytes)
    }
}

pub(crate) fn format_relative_timestamp(value: &str) -> Option<String> {
    let parsed = DateTime::parse_from_rfc3339(value).ok()?;
    let parsed = parsed.with_timezone(&Utc);
    let age = Utc::now().signed_duration_since(parsed);
    let age_label = if age.num_days() >= 1 {
        format!("{}d ago", age.num_days())
    } else if age.num_hours() >= 1 {
        format!("{}h ago", age.num_hours())
    } else if age.num_minutes() >= 1 {
        format!("{}m ago", age.num_minutes())
    } else {
        "just now".to_string()
    };
    Some(format!("{value} ({age_label})"))
}

pub(crate) fn installed_model_kind(path: &Path) -> &'static str {
    let text = path.to_string_lossy().to_ascii_lowercase();
    if text.ends_with(".safetensors")
        || text.ends_with(".safetensors.index.json")
        || text.contains("model.safetensors")
    {
        "🍎 MLX"
    } else {
        "🦙 GGUF"
    }
}

pub(crate) fn format_count(value: u64) -> String {
    let text = value.to_string();
    let mut out = String::with_capacity(text.len() + text.len() / 3);
    for (index, ch) in text.chars().enumerate() {
        if index > 0 && (text.len() - index).is_multiple_of(3) {
            out.push(',');
        }
        out.push(ch);
    }
    out
}

pub(crate) fn format_source_label(source: &str) -> &'static str {
    match source {
        "catalog" => "Catalog",
        "huggingface" => "Hugging Face",
        "url" => "Direct URL",
        _ => "Unknown",
    }
}

pub(crate) fn local_capacity_summary() -> Option<String> {
    let vram_gb = hardware::survey().vram_bytes as f64 / 1e9;
    if vram_gb <= 0.0 {
        None
    } else {
        Some(format!("🖥️ This machine: ~{vram_gb:.1}GB available"))
    }
}

pub(crate) fn local_capacity_json() -> Value {
    let vram_bytes = hardware::survey().vram_bytes;
    let vram_gb = vram_bytes as f64 / 1e9;
    json!({
        "vram_bytes": vram_bytes,
        "vram_gb": vram_gb,
    })
}

pub(crate) fn capabilities_json(caps: ModelCapabilities) -> Value {
    json!({
        "text": true,
        "multimodal": caps.multimodal_status(),
        "vision": caps.vision_status(),
        "audio": caps.audio_status(),
        "reasoning": caps.reasoning_status(),
        "tool_use": caps.tool_use_status(),
        "moe": caps.moe,
    })
}

pub(crate) fn moe_json(moe: Option<&catalog::MoeConfig>) -> Value {
    match moe {
        Some(moe) => json!({
            "n_expert": moe.n_expert,
            "n_expert_used": moe.n_expert_used,
            "min_experts_per_node": moe.min_experts_per_node,
            "ranking_len": moe.ranking.len(),
        }),
        None => Value::Null,
    }
}

pub(crate) fn fit_code_for_size_label(size_label: &str) -> Option<&'static str> {
    let model_gb = catalog::parse_size_gb(size_label);
    let vram_gb = hardware::survey().vram_bytes as f64 / 1e9;
    if model_gb <= 0.0 || vram_gb <= 0.0 {
        return None;
    }

    let code = if model_gb <= vram_gb * 0.6 {
        "comfortable"
    } else if model_gb <= vram_gb * 0.9 {
        "tight"
    } else if model_gb <= vram_gb * 1.1 {
        "tradeoff"
    } else {
        "too_large"
    };
    Some(code)
}

pub(crate) fn fit_hint_for_size_label(size_label: &str) -> Option<String> {
    let model_gb = catalog::parse_size_gb(size_label);
    let vram_gb = hardware::survey().vram_bytes as f64 / 1e9;
    if model_gb <= 0.0 || vram_gb <= 0.0 {
        return None;
    }

    let hint = if model_gb <= vram_gb * 0.6 {
        "✅ likely comfortable here"
    } else if model_gb <= vram_gb * 0.9 {
        "⚠️ likely fits, but tight"
    } else if model_gb <= vram_gb * 1.1 {
        "🟡 may load, but expect tradeoffs"
    } else {
        "⛔ likely too large for local serve"
    };
    Some(hint.to_string())
}

pub(crate) fn variant_selector_label(exact_ref: &str) -> String {
    if let Some((_, selector)) = exact_ref.split_once(':') {
        return selector
            .split_once('@')
            .map(|(value, _)| value)
            .unwrap_or(selector)
            .to_string();
    }
    Path::new(exact_ref)
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or(exact_ref)
        .to_string()
}

pub(crate) fn catalog_model_is_mlx(model: &catalog::CatalogModel) -> bool {
    model
        .source_file()
        .map(|file| {
            file.ends_with("model.safetensors") || file.ends_with("model.safetensors.index.json")
        })
        .unwrap_or(false)
        || model.url.contains("model.safetensors")
}

pub(crate) fn catalog_model_kind(model: &catalog::CatalogModel) -> &'static str {
    if catalog_model_is_mlx(model) {
        "🍎 MLX"
    } else {
        "🦙 GGUF"
    }
}

pub(crate) fn model_kind_code(kind: &str) -> &'static str {
    if kind.to_ascii_lowercase().contains("mlx") {
        "mlx"
    } else {
        "gguf"
    }
}

pub(crate) fn installed_model_kind_code(path: &Path) -> &'static str {
    model_kind_code(installed_model_kind(path))
}

pub(crate) fn catalog_model_kind_code(model: &catalog::CatalogModel) -> &'static str {
    model_kind_code(catalog_model_kind(model))
}

pub(crate) fn catalog_model_capabilities(model: &catalog::CatalogModel) -> ModelCapabilities {
    capabilities::infer_catalog_capabilities(model)
}

pub(crate) fn huggingface_cache_dir() -> PathBuf {
    huggingface_hub_cache_dir()
}
