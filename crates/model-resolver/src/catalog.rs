use std::collections::HashMap;

use serde::{de, Deserialize, Deserializer, Serialize};

#[derive(Debug, Clone, Serialize)]
pub struct CatalogEntry {
    pub schema_version: u32,
    pub source_repo: String,
    pub variants: HashMap<String, CatalogVariant>,
}

impl<'de> Deserialize<'de> for CatalogEntry {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct RawCatalogEntry {
            schema_version: u32,
            #[serde(default)]
            source_repo: Option<String>,
            variants: CatalogVariants,
        }

        let raw = RawCatalogEntry::deserialize(deserializer)?;
        let source_repo = raw
            .source_repo
            .or_else(|| {
                raw.variants
                    .0
                    .values()
                    .next()
                    .map(|variant| variant.source.repo.clone())
            })
            .ok_or_else(|| de::Error::missing_field("source_repo"))?;

        Ok(CatalogEntry {
            schema_version: raw.schema_version,
            source_repo,
            variants: raw.variants.0,
        })
    }
}

struct CatalogVariants(HashMap<String, CatalogVariant>);

impl<'de> Deserialize<'de> for CatalogVariants {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = serde_json::Value::deserialize(deserializer)?;
        match value {
            serde_json::Value::Object(object) => {
                let mut variants = HashMap::with_capacity(object.len());
                for (name, value) in object {
                    let variant = CatalogVariant::deserialize(value).map_err(de::Error::custom)?;
                    variants.insert(name, variant);
                }
                Ok(Self(variants))
            }
            serde_json::Value::Array(array) => {
                let mut variants = HashMap::with_capacity(array.len());
                for (index, value) in array.into_iter().enumerate() {
                    let variant = CatalogVariant::deserialize(value).map_err(de::Error::custom)?;
                    let key = catalog_variant_key_from_list_item(&variant, index);
                    variants.insert(key, variant);
                }
                Ok(Self(variants))
            }
            value => Err(de::Error::custom(format!(
                "catalog variants must be an object or array; got {value}"
            ))),
        }
    }
}

fn catalog_variant_key_from_list_item(variant: &CatalogVariant, index: usize) -> String {
    variant
        .source
        .file
        .as_deref()
        .map(|file| file.strip_suffix(".gguf").unwrap_or(file).to_string())
        .filter(|file| !file.is_empty())
        .unwrap_or_else(|| {
            if variant.curated.name.is_empty() {
                format!("variant-{index}")
            } else {
                variant.curated.name.clone()
            }
        })
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatalogVariant {
    pub source: CatalogSource,
    pub curated: CuratedMeta,
    #[serde(default)]
    pub packages: Vec<CatalogPackage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatalogSource {
    pub repo: String,
    #[serde(default)]
    pub revision: Option<String>,
    #[serde(default)]
    pub file: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CuratedMeta {
    pub name: String,
    #[serde(default)]
    pub size: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(
        default,
        deserialize_with = "deserialize_optional_string_or_legacy_bool"
    )]
    pub draft: Option<String>,
    #[serde(default, deserialize_with = "deserialize_optional_moe")]
    pub moe: Option<serde_json::Value>,
    #[serde(default)]
    pub extra_files: Vec<serde_json::Value>,
    #[serde(default)]
    pub mmproj: Option<CatalogSidecarRef>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(untagged)]
pub enum CatalogSidecarRef {
    Ref(String),
    Asset(CatalogSidecarAsset),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CatalogSidecarAsset {
    pub file: String,
    pub repo: String,
    #[serde(default)]
    pub revision: Option<String>,
    #[serde(default)]
    pub source_file: Option<String>,
}

fn deserialize_optional_string_or_legacy_bool<'de, D>(
    deserializer: D,
) -> std::result::Result<Option<String>, D::Error>
where
    D: Deserializer<'de>,
{
    let value = Option::<serde_json::Value>::deserialize(deserializer)?;
    match value {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(serde_json::Value::String(value)) => Ok(Some(value)),
        Some(serde_json::Value::Bool(_)) => Ok(None),
        Some(value) => Err(de::Error::custom(format!(
            "catalog curated.draft must be a string, null, or legacy bool; got {value}"
        ))),
    }
}

fn deserialize_optional_moe<'de, D>(
    deserializer: D,
) -> std::result::Result<Option<serde_json::Value>, D::Error>
where
    D: Deserializer<'de>,
{
    let value = Option::<serde_json::Value>::deserialize(deserializer)?;
    match value {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(serde_json::Value::String(value)) => Ok(Some(serde_json::Value::String(value))),
        Some(serde_json::Value::Object(value)) => Ok(Some(serde_json::Value::Object(value))),
        Some(value) => Err(de::Error::custom(format!(
            "catalog curated.moe must be a string, object, or null; got {value}"
        ))),
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatalogPackage {
    #[serde(rename = "type")]
    pub package_type: String,
    pub repo: String,
    #[serde(default)]
    pub layer_count: Option<u32>,
    #[serde(default)]
    pub total_bytes: Option<u64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deserializes_catalog_entry() {
        let json = r#"{
            "schema_version": 1,
            "source_repo": "unsloth/Qwen3-Coder-480B-A35B-Instruct-GGUF",
            "variants": {
                "Qwen3-Coder-480B-A35B-Instruct-UD-Q4_K_XL": {
                    "source": { "repo": "unsloth/Qwen3-Coder-480B-A35B-Instruct-GGUF", "revision": "main", "file": "Qwen3-Coder-480B-A35B-Instruct-UD-Q4_K_XL.gguf" },
                    "curated": { "name": "Qwen3 Coder 480B Q4_K_XL", "size": "294GB", "description": "Large MoE coding model", "draft": "Qwen3-Coder-Draft-Q4_K_M", "moe": "480B/35B", "extra_files": [], "mmproj": { "file": "mmproj-BF16.gguf", "repo": "unsloth/Qwen3-Coder-480B-A35B-Instruct-GGUF", "revision": "main" } },
                    "packages": [
                        { "type": "layer-package", "repo": "meshllm/Qwen3-Coder-480B-A35B-Instruct-UD-Q4_K_XL-layers", "layer_count": 62, "total_bytes": 315680000000 }
                    ]
                }
            }
        }"#;

        let entry: CatalogEntry = serde_json::from_str(json).unwrap();
        assert_eq!(entry.schema_version, 1);
        assert_eq!(
            entry.source_repo,
            "unsloth/Qwen3-Coder-480B-A35B-Instruct-GGUF"
        );
        assert_eq!(entry.variants.len(), 1);

        let variant = entry
            .variants
            .get("Qwen3-Coder-480B-A35B-Instruct-UD-Q4_K_XL")
            .unwrap();
        assert_eq!(variant.curated.name, "Qwen3 Coder 480B Q4_K_XL");
        assert_eq!(
            variant.curated.draft.as_deref(),
            Some("Qwen3-Coder-Draft-Q4_K_M")
        );
        assert_eq!(
            variant
                .curated
                .moe
                .as_ref()
                .and_then(|value| value.as_str()),
            Some("480B/35B")
        );
        assert!(matches!(
            variant.curated.mmproj.as_ref(),
            Some(CatalogSidecarRef::Asset(asset))
                if asset.file == "mmproj-BF16.gguf"
                    && asset.repo == "unsloth/Qwen3-Coder-480B-A35B-Instruct-GGUF"
                    && asset.revision.as_deref() == Some("main")
        ));
        assert_eq!(variant.packages.len(), 1);
        assert_eq!(variant.packages[0].package_type, "layer-package");
        assert_eq!(
            variant.packages[0].repo,
            "meshllm/Qwen3-Coder-480B-A35B-Instruct-UD-Q4_K_XL-layers"
        );
        assert_eq!(variant.packages[0].layer_count, Some(62));
    }

    #[test]
    fn serialization_round_trip() {
        let entry = CatalogEntry {
            schema_version: 1,
            source_repo: "test/repo".to_string(),
            variants: {
                let mut map = HashMap::new();
                map.insert(
                    "test-variant".to_string(),
                    CatalogVariant {
                        source: CatalogSource {
                            repo: "test/repo".to_string(),
                            revision: Some("abc123".to_string()),
                            file: Some("model.gguf".to_string()),
                        },
                        curated: CuratedMeta {
                            name: "Test Model".to_string(),
                            size: Some("4GB".to_string()),
                            description: Some("A test model".to_string()),
                            draft: Some("test-draft-Q4_K_M".to_string()),
                            moe: None,
                            extra_files: Vec::new(),
                            mmproj: Some(CatalogSidecarRef::Ref(
                                "org/repo/mmproj.gguf".to_string(),
                            )),
                        },
                        packages: vec![CatalogPackage {
                            package_type: "layer-package".to_string(),
                            repo: "meshllm/test-layers".to_string(),
                            layer_count: Some(32),
                            total_bytes: Some(4_000_000_000),
                        }],
                    },
                );
                map
            },
        };

        let json = serde_json::to_string(&entry).unwrap();
        let deserialized: CatalogEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.schema_version, entry.schema_version);
        assert_eq!(deserialized.source_repo, entry.source_repo);
        assert_eq!(deserialized.variants.len(), 1);

        let variant = deserialized.variants.get("test-variant").unwrap();
        assert_eq!(variant.source.repo, "test/repo");
        assert_eq!(variant.curated.name, "Test Model");
        assert_eq!(variant.curated.draft.as_deref(), Some("test-draft-Q4_K_M"));
        assert_eq!(variant.packages[0].layer_count, Some(32));
    }

    #[test]
    fn deserializes_legacy_bool_draft_as_no_draft_ref() {
        let json = r#"{
            "schema_version": 1,
            "source_repo": "org/repo",
            "variants": {
                "repo-Q4_K_M": {
                    "source": { "repo": "org/repo", "revision": "main", "file": "repo-Q4_K_M.gguf" },
                    "curated": { "name": "Repo Q4", "draft": true, "extra_files": [], "mmproj": null },
                    "packages": []
                }
            }
        }"#;

        let entry: CatalogEntry = serde_json::from_str(json).unwrap();
        let variant = entry.variants.get("repo-Q4_K_M").unwrap();
        assert_eq!(variant.curated.draft, None);
    }

    #[test]
    fn deserializes_moe_object() {
        let json = r#"{
            "schema_version": 1,
            "source_repo": "org/repo",
            "variants": {
                "repo-Q4_K_M": {
                    "source": { "repo": "org/repo", "revision": "main", "file": "repo-Q4_K_M.gguf" },
                    "curated": {
                        "name": "Repo Q4",
                        "moe": { "n_expert": 128, "n_expert_used": 8, "min_experts_per_node": 46 }
                    },
                    "packages": []
                }
            }
        }"#;

        let entry: CatalogEntry = serde_json::from_str(json).unwrap();
        let moe = entry
            .variants
            .get("repo-Q4_K_M")
            .unwrap()
            .curated
            .moe
            .as_ref()
            .and_then(|value| value.as_object())
            .unwrap();
        assert_eq!(
            moe.get("n_expert").and_then(|value| value.as_u64()),
            Some(128)
        );
        assert_eq!(
            moe.get("n_expert_used").and_then(|value| value.as_u64()),
            Some(8)
        );
    }

    #[test]
    fn deserializes_legacy_variant_list_and_infers_source_repo() {
        let json = r#"{
            "schema_version": 1,
            "variants": [
                {
                    "source": {
                        "repo": "unsloth/Kimi-K2-Thinking-GGUF",
                        "revision": "main",
                        "file": "UD-Q4_K_XL/Kimi-K2-Thinking-UD-Q4_K_XL-00001-of-00014.gguf"
                    },
                    "curated": {
                        "name": "unsloth/Kimi-K2-Thinking-GGUF:UD-Q4_K_XL",
                        "size": "61 layers",
                        "description": "Layer package"
                    },
                    "packages": [
                        { "type": "layer-package", "repo": "meshllm/Kimi-K2-Thinking-UD-Q4_K_XL-layers", "layer_count": 61 }
                    ]
                }
            ]
        }"#;

        let entry: CatalogEntry = serde_json::from_str(json).unwrap();
        assert_eq!(entry.source_repo, "unsloth/Kimi-K2-Thinking-GGUF");
        let variant = entry
            .variants
            .get("UD-Q4_K_XL/Kimi-K2-Thinking-UD-Q4_K_XL-00001-of-00014")
            .unwrap();
        assert_eq!(
            variant.curated.name,
            "unsloth/Kimi-K2-Thinking-GGUF:UD-Q4_K_XL"
        );
        assert_eq!(variant.packages[0].layer_count, Some(61));
    }
}
