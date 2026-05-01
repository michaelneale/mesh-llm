use serde::Deserialize;
use std::sync::LazyLock;

#[derive(Clone, Debug, Deserialize)]
pub struct CatalogAsset {
    pub file: String,
    pub url: String,
}

#[derive(Clone, Debug)]
pub struct CatalogModel {
    pub name: String,
    pub file: String,
    pub url: String,
    pub size: String,
    pub description: String,
    pub draft: Option<String>,
    pub moe: Option<MoeConfig>,
    pub extra_files: Vec<CatalogAsset>,
    pub mmproj: Option<CatalogAsset>,
}

impl CatalogModel {
    pub fn source_repo(&self) -> Option<&str> {
        parse_hf_resolve_url_parts(&self.url).map(|(repo, _, _)| repo)
    }

    pub fn source_revision(&self) -> Option<&str> {
        parse_hf_resolve_url_parts(&self.url).and_then(|(_, revision, _)| revision)
    }

    pub fn source_file(&self) -> Option<&str> {
        parse_hf_resolve_url_parts(&self.url).map(|(_, _, file)| file)
    }
}

#[derive(Clone, Debug)]
pub struct MoeConfig {
    pub n_expert: u32,
    pub n_expert_used: u32,
    pub min_experts_per_node: u32,
    pub ranking: Vec<u32>,
}

#[derive(Debug, Deserialize)]
struct CatalogModelJson {
    name: String,
    file: String,
    url: String,
    size: String,
    description: String,
    draft: Option<String>,
    moe: Option<MoeConfigJson>,
    #[serde(default)]
    extra_files: Vec<CatalogAsset>,
    mmproj: Option<CatalogAsset>,
}

#[derive(Clone, Debug, Deserialize)]
struct MoeConfigJson {
    n_expert: u32,
    n_expert_used: u32,
    min_experts_per_node: u32,
}

pub static MODEL_CATALOG: LazyLock<Vec<CatalogModel>> = LazyLock::new(load_catalog);

fn load_catalog() -> Vec<CatalogModel> {
    let raw: Vec<CatalogModelJson> =
        serde_json::from_str(include_str!("catalog.json")).expect("parse bundled catalog.json");
    raw.into_iter().map(CatalogModel::from_json).collect()
}

impl CatalogModel {
    fn from_json(raw: CatalogModelJson) -> Self {
        Self {
            name: raw.name,
            file: raw.file,
            url: raw.url,
            size: raw.size,
            description: raw.description,
            draft: raw.draft,
            moe: raw.moe.map(MoeConfig::from_json),
            extra_files: raw.extra_files,
            mmproj: raw.mmproj,
        }
    }
}

impl MoeConfig {
    fn from_json(raw: MoeConfigJson) -> Self {
        Self {
            n_expert: raw.n_expert,
            n_expert_used: raw.n_expert_used,
            min_experts_per_node: raw.min_experts_per_node,
            ranking: Vec::new(),
        }
    }
}

pub fn parse_size_gb(s: &str) -> f64 {
    let s = s.trim();
    if let Some(gb) = s.strip_suffix("GB") {
        gb.trim().parse().unwrap_or(0.0)
    } else if let Some(mb) = s.strip_suffix("MB") {
        mb.trim().parse::<f64>().unwrap_or(0.0) / 1000.0
    } else {
        0.0
    }
}

pub fn find_model(query: &str) -> Option<&'static CatalogModel> {
    let q = query.to_lowercase();
    MODEL_CATALOG
        .iter()
        .find(|m| m.name.to_lowercase() == q)
        .or_else(|| {
            MODEL_CATALOG
                .iter()
                .find(|m| m.name.to_lowercase().contains(&q))
        })
}

pub fn parse_hf_resolve_url_parts(url: &str) -> Option<(&str, Option<&str>, &str)> {
    let tail = url
        .strip_prefix("https://huggingface.co/")
        .or_else(|| url.strip_prefix("http://huggingface.co/"))?;
    let (repo, rest) = tail.split_once("/resolve/")?;
    if !repo.contains('/') {
        return None;
    }
    let (revision, file) = rest.split_once('/')?;
    if file.is_empty() {
        return None;
    }
    Some((repo, Some(revision), file))
}

pub fn huggingface_repo_url(url: &str) -> Option<String> {
    let (repo, _, _) = parse_hf_resolve_url_parts(url)?;
    Some(format!("https://huggingface.co/{repo}"))
}

pub fn list_models() {
    eprintln!("Available models:");
    eprintln!();
    for m in MODEL_CATALOG.iter() {
        let draft_info = if let Some(d) = m.draft.as_deref() {
            format!(" (draft: {})", d)
        } else {
            String::new()
        };
        eprintln!(
            "  {:40} {:>6}  {}{}",
            m.name, m.size, m.description, draft_info
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_identity_is_exposed_for_hf_catalog_entries() {
        let model = find_model("Qwen3-8B-Q4_K_M").unwrap();
        assert_eq!(model.source_repo(), Some("unsloth/Qwen3-8B-GGUF"));
        assert_eq!(model.source_revision(), Some("main"));
        assert_eq!(model.source_file(), Some("Qwen3-8B-Q4_K_M.gguf"));
        assert!(model.source_repo().is_some());
    }

    #[test]
    fn source_identity_is_absent_for_direct_url_entries() {
        let model = find_model("Qwen3.5-27B-Q4_K_M").unwrap();
        assert_eq!(model.source_repo(), None);
        assert_eq!(model.source_revision(), None);
        assert_eq!(model.source_file(), None);
        assert!(model.source_repo().is_none());
    }

    #[test]
    fn test_split_url_generation() {
        let filename = "Model-Q4_K_M-00001-of-00003.gguf";
        let url = "https://huggingface.co/org/repo/resolve/main/Model-Q4_K_M-00001-of-00003.gguf";

        let mut files = Vec::new();
        for i in 1..=3u32 {
            let part_filename = filename.replace("-00001-of-", &format!("-{i:05}-of-"));
            let part_url = url.replace("-00001-of-", &format!("-{i:05}-of-"));
            files.push((part_filename, part_url));
        }

        assert_eq!(files.len(), 3);
        assert_eq!(files[0].0, "Model-Q4_K_M-00001-of-00003.gguf");
        assert_eq!(files[1].0, "Model-Q4_K_M-00002-of-00003.gguf");
        assert_eq!(files[2].0, "Model-Q4_K_M-00003-of-00003.gguf");
        assert!(files[0].1.contains("-00001-of-"));
        assert!(files[1].1.contains("-00002-of-"));
        assert!(files[2].1.contains("-00003-of-"));
    }
}
