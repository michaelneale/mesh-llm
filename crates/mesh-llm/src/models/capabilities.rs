pub use mesh_client::models::capabilities::{
    merge_config_signals, merge_name_signals, merge_sibling_signals, CapabilityLevel,
    ModelCapabilities,
};

use super::build_hf_tokio_api;
use super::catalog;
use hf_hub::{RepoDownloadFileParams, RepoType};
use serde_json::Value;
use std::path::Path;

pub fn infer_catalog_capabilities(model: &catalog::CatalogModel) -> ModelCapabilities {
    let mut caps = ModelCapabilities::default();
    if model.mmproj.is_some() {
        caps.vision = CapabilityLevel::Supported;
        caps.multimodal = true;
    }
    caps.moe = model.moe.is_some();
    caps = merge_name_signals(
        caps,
        &[
            model.name.as_str(),
            model.file.as_str(),
            model.description.as_str(),
        ],
    );
    caps.normalize()
}

pub fn infer_local_model_capabilities(
    model_name: &str,
    path: &Path,
    catalog_entry: Option<&catalog::CatalogModel>,
) -> ModelCapabilities {
    let mut caps = catalog_entry
        .map(infer_catalog_capabilities)
        .unwrap_or_default();
    caps = merge_name_signals(
        caps,
        &[
            model_name,
            path.file_name()
                .and_then(|value| value.to_str())
                .unwrap_or_default(),
        ],
    );
    for config in read_local_metadata_jsons(path) {
        caps = merge_config_signals(caps, &config);
    }
    caps.normalize()
}

pub async fn infer_remote_hf_capabilities(
    repo: &str,
    revision: Option<&str>,
    file: &str,
    siblings: Option<&[String]>,
) -> ModelCapabilities {
    let metadata = fetch_remote_hf_metadata_jsons(repo, revision).await;
    infer_remote_hf_capabilities_with_metadata(repo, file, siblings, &metadata)
}

pub fn infer_remote_hf_capabilities_with_metadata(
    repo: &str,
    file: &str,
    siblings: Option<&[String]>,
    metadata: &[Value],
) -> ModelCapabilities {
    let mut caps = ModelCapabilities::default();
    caps = merge_name_signals(caps, &[repo, file]);
    if let Some(files) = siblings {
        caps = merge_sibling_signals(caps, files.iter().map(String::as_str));
    }
    for config in metadata {
        caps = merge_config_signals(caps, config);
    }
    caps.normalize()
}

fn read_local_metadata_jsons(path: &Path) -> Vec<Value> {
    let mut values = Vec::new();
    for dir in path.ancestors().skip(1).take(6) {
        for name in ["config.json", "tokenizer_config.json", "chat_template.json"] {
            let candidate = dir.join(name);
            if !candidate.is_file() {
                continue;
            }
            let Ok(text) = std::fs::read_to_string(&candidate) else {
                continue;
            };
            if let Ok(value) = serde_json::from_str(&text) {
                values.push(value);
            }
        }
    }
    values
}

pub async fn fetch_remote_hf_metadata_jsons(repo: &str, revision: Option<&str>) -> Vec<Value> {
    let Some(api) = build_hf_tokio_api(false).ok() else {
        return Vec::new();
    };
    let revision = revision.unwrap_or("main").to_string();
    let config = fetch_remote_json_with_api(
        api.clone(),
        repo.to_string(),
        revision.clone(),
        "config.json",
    );
    let tokenizer = fetch_remote_json_with_api(
        api.clone(),
        repo.to_string(),
        revision.clone(),
        "tokenizer_config.json",
    );
    let chat_template =
        fetch_remote_json_with_api(api, repo.to_string(), revision, "chat_template.json");

    let (config, tokenizer, chat_template) = tokio::join!(config, tokenizer, chat_template);
    let mut values = Vec::new();
    for value in [config, tokenizer, chat_template].into_iter().flatten() {
        values.push(value);
    }
    values
}

async fn fetch_remote_json_with_api(
    api: hf_hub::HFClient,
    repo: String,
    revision: String,
    file: &'static str,
) -> Option<Value> {
    let (owner, name) = repo.split_once('/').unwrap_or(("", repo.as_str()));
    let path = api
        .repo(RepoType::Model, owner, name)
        .download_file(
            &RepoDownloadFileParams::builder()
                .filename(file.to_string())
                .revision(revision)
                .build(),
        )
        .await
        .ok()?;
    let text = tokio::fs::read_to_string(path).await.ok()?;
    serde_json::from_str(&text).ok()
}
