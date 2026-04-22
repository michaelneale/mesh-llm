use super::*;
use serde::Deserialize;
use std::collections::HashMap;

#[derive(Debug, Deserialize)]
struct HfRepoFixture {
    repo: String,
    siblings: Vec<String>,
    size_bytes: HashMap<String, u64>,
}

fn load_gemma_live_fixture() -> HfRepoFixture {
    serde_json::from_str(include_str!(
        "../testdata/unsloth_gemma_4_31b_it_gguf.live.json"
    ))
    .expect("parse live Hugging Face fixture")
}

#[test]
fn primary_hf_ref_maps_to_full_catalog_download() {
    let model = matching_catalog_primary_for_huggingface(
        "unsloth/Qwen3.5-0.8B-GGUF",
        Some("main"),
        "Qwen3.5-0.8B-Q4_K_M.gguf",
    )
    .expect("primary model file should map to catalog download");
    assert_eq!(model.name, "Qwen3.5-0.8B-Vision-Q4_K_M");
    assert!(model.mmproj.is_some());
}

#[test]
fn mmproj_hf_ref_does_not_expand_to_full_catalog_download() {
    assert!(matching_catalog_primary_for_huggingface(
        "unsloth/Qwen3.5-0.8B-GGUF",
        Some("main"),
        "mmproj-BF16.gguf",
    )
    .is_none());
}

#[test]
fn primary_url_maps_to_full_catalog_download() {
    let model = matching_catalog_primary_for_url(
        "https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-Q4_K_M.gguf",
    )
    .expect("primary model url should map to catalog download");
    assert_eq!(model.name, "Qwen3.5-0.8B-Vision-Q4_K_M");
    assert!(model.mmproj.is_some());
}

#[test]
fn mmproj_url_does_not_expand_to_full_catalog_download() {
    assert!(matching_catalog_primary_for_url(
        "https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/mmproj-BF16.gguf",
    )
    .is_none());
}

#[test]
fn split_stem_resolves_to_first_part() {
    let siblings = vec![
        "zai-org.GLM-5.1.Q2_K-00002-of-00018.gguf".to_string(),
        "zai-org.GLM-5.1.Q2_K-00001-of-00018.gguf".to_string(),
    ];
    let resolved = resolve_hf_file_from_siblings("zai-org.GLM-5.1.Q2_K", &siblings).unwrap();
    assert_eq!(resolved, "zai-org.GLM-5.1.Q2_K-00001-of-00018.gguf");
}

#[test]
fn stem_without_split_resolves_to_gguf() {
    let siblings = vec![
        "Qwen3-8B-Q4_K_M.gguf".to_string(),
        "Qwen3-8B-Q8_0.gguf".to_string(),
    ];
    let resolved = resolve_hf_file_from_siblings("Qwen3-8B-Q4_K_M", &siblings).unwrap();
    assert_eq!(resolved, "Qwen3-8B-Q4_K_M.gguf");
}

#[test]
fn mlx_stem_resolves_to_model_safetensors() {
    let siblings = vec![
        "model.safetensors.index.json".to_string(),
        "model.safetensors".to_string(),
    ];
    let resolved = resolve_hf_file_from_siblings("model", &siblings).unwrap();
    assert_eq!(resolved, "model.safetensors");
}

#[test]
fn mlx_stem_resolves_to_first_split_shard() {
    let siblings = vec![
        "model-00002-of-00048.safetensors".to_string(),
        "model-00001-of-00048.safetensors".to_string(),
        "model.safetensors.index.json".to_string(),
    ];
    let resolved = resolve_hf_file_from_siblings("model", &siblings).unwrap();
    assert_eq!(resolved, "model-00001-of-00048.safetensors");
}

#[test]
fn repo_only_resolution_prefers_mlx_model_safetensors() {
    let siblings = vec![
        "Qwen3-8B-Q4_K_M.gguf".to_string(),
        "model.safetensors".to_string(),
        "model.safetensors.index.json".to_string(),
    ];
    let resolved = resolve_hf_file_from_siblings("", &siblings).unwrap();
    assert_eq!(resolved, "model.safetensors");
}

#[test]
fn repo_only_resolution_falls_back_to_gguf_when_no_mlx_weights() {
    let siblings = vec![
        "Qwen3-8B-Q8_0.gguf".to_string(),
        "Qwen3-8B-Q4_K_M.gguf".to_string(),
    ];
    let resolved = resolve_hf_file_from_siblings("", &siblings).unwrap();
    assert_eq!(resolved, "Qwen3-8B-Q4_K_M.gguf");
}

#[test]
fn parse_huggingface_ref_rejects_http_url() {
    assert!(parse_huggingface_ref("https://example.com/model.gguf").is_none());
}

#[test]
fn parse_huggingface_repo_ref_parses_repo_only() {
    let parsed = parse_huggingface_repo_ref("GreenBitAI/Llama-2-7B-layer-mix-bpw-2.2-mlx");
    assert_eq!(
        parsed,
        Some((
            "GreenBitAI/Llama-2-7B-layer-mix-bpw-2.2-mlx".to_string(),
            None,
            None
        ))
    );
}

#[test]
fn parse_huggingface_repo_ref_parses_quant_selector() {
    let parsed = parse_huggingface_repo_ref("unsloth/gemma-4-31B-it-GGUF:UD-Q4_K_XL");
    assert_eq!(
        parsed,
        Some((
            "unsloth/gemma-4-31B-it-GGUF".to_string(),
            None,
            Some("UD-Q4_K_XL".to_string())
        ))
    );
}

#[test]
fn parse_huggingface_repo_url_parses_repo_only() {
    let parsed = parse_huggingface_repo_url("https://huggingface.co/unsloth/gemma-4-31B-it-GGUF");
    assert_eq!(
        parsed,
        Some(("unsloth/gemma-4-31B-it-GGUF".to_string(), None, None))
    );
}

#[test]
fn parse_huggingface_repo_url_parses_tree_revision() {
    let parsed =
        parse_huggingface_repo_url("https://huggingface.co/unsloth/gemma-4-31B-it-GGUF/tree/main");
    assert_eq!(
        parsed,
        Some((
            "unsloth/gemma-4-31B-it-GGUF".to_string(),
            Some("main".to_string()),
            None
        ))
    );
}

#[test]
fn quant_selector_resolves_to_single_file_gguf() {
    let fixture = load_gemma_live_fixture();
    let resolved = resolve_hf_file_from_siblings("UD-Q4_K_XL", &fixture.siblings).unwrap();
    assert_eq!(resolved, "gemma-4-31B-it-UD-Q4_K_XL.gguf");
}

#[test]
fn dotted_quant_selector_resolves_to_single_file_gguf() {
    let siblings = vec![
        "qwen3.5-moe-0.87B-d0.8B.Q2_K.gguf".to_string(),
        "qwen3.5-moe-0.87B-d0.8B.Q4_K_M.gguf".to_string(),
    ];
    let resolved = resolve_hf_file_from_siblings("Q2_K", &siblings).unwrap();
    assert_eq!(resolved, "qwen3.5-moe-0.87B-d0.8B.Q2_K.gguf");
}

#[test]
fn gemma_bf16_selector_resolves_to_first_split_shard() {
    let fixture = load_gemma_live_fixture();
    let resolved = resolve_hf_file_from_siblings("BF16", &fixture.siblings).unwrap();
    assert_eq!(resolved, "BF16/gemma-4-31B-it-BF16-00001-of-00002.gguf");
}

#[test]
fn fit_aware_gguf_prefers_largest_comfortable_candidate() {
    let available = 20_000_000_000u64;
    let ordering = compare_gguf_candidates_by_fit(
        "repo/model-q4.gguf",
        Some(12_000_000_000),
        "repo/model-q5.gguf",
        Some(17_000_000_000),
        available,
    );
    assert_eq!(ordering, Ordering::Greater);
}

#[test]
fn fit_aware_gguf_prefers_smaller_when_both_too_large() {
    let available = 20_000_000_000u64;
    let ordering = compare_gguf_candidates_by_fit(
        "repo/model-q8.gguf",
        Some(29_000_000_000),
        "repo/model-bf16.gguf",
        Some(35_000_000_000),
        available,
    );
    assert_eq!(ordering, Ordering::Less);
}

#[test]
fn gemma_repo_default_prefers_q4_over_bf16_at_local_fit_budget() {
    let fixture = load_gemma_live_fixture();
    let q4 = fixture
        .size_bytes
        .get("gemma-4-31B-it-Q4_0.gguf")
        .copied()
        .expect("fixture Q4_0 size");
    let bf16 = fixture
        .size_bytes
        .get("BF16/gemma-4-31B-it-BF16-00001-of-00002.gguf")
        .copied()
        .expect("fixture BF16 size");
    let available = 19_300_000_000u64;
    let ordering = compare_gguf_candidates_by_fit(
        "unsloth/gemma-4-31B-it-GGUF/gemma-4-31B-it-Q4_0.gguf",
        Some(q4),
        "unsloth/gemma-4-31B-it-GGUF/BF16/gemma-4-31B-it-BF16-00001-of-00002.gguf",
        Some(bf16),
        available,
    );
    assert_eq!(ordering, Ordering::Less);
}

#[test]
fn repo_name_can_signal_gguf_intent() {
    assert!(repo_prefers_gguf_only("unsloth/gemma-4-31B-it-GGUF"));
    assert!(!repo_prefers_gguf_only(
        "mlx-community/Llama-3.2-3B-Instruct-4bit"
    ));
}

#[test]
fn parse_exact_model_ref_accepts_unsloth_gemma_repo_ref() {
    let parsed = parse_exact_model_ref("unsloth/gemma-4-31B-it-GGUF").unwrap();
    match parsed {
        ExactModelRef::HuggingFace {
            repo,
            revision,
            file,
        } => {
            assert_eq!(repo, "unsloth/gemma-4-31B-it-GGUF");
            assert_eq!(revision, None);
            assert_eq!(file, "");
        }
        other => panic!("expected HuggingFace repo ref, got {other:?}"),
    }
}

#[test]
fn parse_exact_model_ref_accepts_unsloth_gemma_repo_url() {
    let parsed =
        parse_exact_model_ref("https://huggingface.co/unsloth/gemma-4-31B-it-GGUF").unwrap();
    match parsed {
        ExactModelRef::HuggingFace {
            repo,
            revision,
            file,
        } => {
            assert_eq!(repo, "unsloth/gemma-4-31B-it-GGUF");
            assert_eq!(revision, None);
            assert_eq!(file, "");
        }
        other => panic!("expected HuggingFace repo ref from URL, got {other:?}"),
    }
}

#[test]
fn parse_exact_model_ref_accepts_unsloth_gemma_quant_selector() {
    let parsed = parse_exact_model_ref("unsloth/gemma-4-31B-it-GGUF:UD-Q4_K_XL").unwrap();
    match parsed {
        ExactModelRef::HuggingFace {
            repo,
            revision,
            file,
        } => {
            assert_eq!(repo, "unsloth/gemma-4-31B-it-GGUF");
            assert_eq!(revision, None);
            assert_eq!(file, "UD-Q4_K_XL");
        }
        other => panic!("expected HuggingFace quant selector ref, got {other:?}"),
    }
}

#[test]
fn simulated_name_and_repo_quant_inputs_converge_to_same_ref() {
    let fixture = load_gemma_live_fixture();
    let discovered_repo = fixture.repo.as_str();
    let selector = "UD-Q4_K_XL";

    let from_name = format!(
        "{}/{}",
        discovered_repo,
        resolve_hf_file_from_siblings(selector, &fixture.siblings).unwrap()
    );
    let from_repo = format!(
        "{}/{}",
        discovered_repo,
        resolve_hf_file_from_siblings(selector, &fixture.siblings).unwrap()
    );

    assert_eq!(
        from_name,
        "unsloth/gemma-4-31B-it-GGUF/gemma-4-31B-it-UD-Q4_K_XL.gguf"
    );
    assert_eq!(from_name, from_repo);
}

#[test]
fn parse_exact_model_ref_accepts_unsloth_gemma_repo_url_with_quant_selector() {
    let parsed =
        parse_exact_model_ref("https://huggingface.co/unsloth/gemma-4-31B-it-GGUF:UD-Q4_K_XL")
            .unwrap();
    match parsed {
        ExactModelRef::HuggingFace {
            repo,
            revision,
            file,
        } => {
            assert_eq!(repo, "unsloth/gemma-4-31B-it-GGUF");
            assert_eq!(revision, None);
            assert_eq!(file, "UD-Q4_K_XL");
        }
        other => panic!("expected HuggingFace repo URL quant selector ref, got {other:?}"),
    }
}

#[test]
fn split_bare_name_selector_supports_name_quant_shorthand() {
    assert_eq!(
        split_bare_name_selector("gemma-4-31B-it-GGUF:UD-Q4_K_XL"),
        ("gemma-4-31B-it-GGUF", Some("UD-Q4_K_XL"))
    );
    assert_eq!(
        split_bare_name_selector("unsloth/gemma-4-31B-it-GGUF:UD-Q4_K_XL"),
        ("unsloth/gemma-4-31B-it-GGUF:UD-Q4_K_XL", None)
    );
}

#[test]
fn select_strong_repo_hit_prefers_exact_leaf_name() {
    let repos = vec![
        "ggml-org/gemma-4-31B-it-GGUF".to_string(),
        "unsloth/gemma-4-31B-it-GGUF".to_string(),
        "bartowski/google_gemma-4-31B-it-GGUF".to_string(),
    ];
    let picked = select_strong_repo_hit("gemma-4-31B-it-GGUF", &repos);
    assert_eq!(picked, Some("ggml-org/gemma-4-31B-it-GGUF".to_string()));
}

#[test]
fn bare_name_quant_can_be_formatted_with_discovered_repo() {
    let (name, selector) = split_bare_name_selector("gemma-4-31B-it-GGUF:UD-Q4_K_XL");
    assert_eq!(name, "gemma-4-31B-it-GGUF");
    let selector = selector.expect("selector");
    let canonical = format!("{}:{}", "unsloth/gemma-4-31B-it-GGUF", selector);
    assert_eq!(canonical, "unsloth/gemma-4-31B-it-GGUF:UD-Q4_K_XL");
}

#[test]
fn quant_selector_from_gguf_file_extracts_expected_forms() {
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
        quant_selector_from_gguf_file("gemma-4-31B-it-Q4_0.gguf"),
        Some("Q4_0".to_string())
    );
    assert_eq!(
        quant_selector_from_gguf_file("qwen3.5-moe-0.87B-d0.8B.Q2_K.gguf"),
        Some("Q2_K".to_string())
    );
}

#[test]
fn format_huggingface_display_ref_prefers_selector_form_for_gguf() {
    assert_eq!(
        format_huggingface_display_ref(
            "unsloth/gemma-4-31B-it-GGUF",
            None,
            "gemma-4-31B-it-UD-Q4_K_XL.gguf"
        ),
        "unsloth/gemma-4-31B-it-GGUF:UD-Q4_K_XL"
    );
    assert_eq!(
        format_huggingface_display_ref(
            "QuantFactory/Meta-Llama-3.1-8B-Instruct-GGUF",
            None,
            "Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf"
        ),
        "QuantFactory/Meta-Llama-3.1-8B-Instruct-GGUF:Q4_K_M"
    );
}

#[test]
fn format_huggingface_display_ref_uses_selector_for_split_gguf() {
    assert_eq!(
        format_huggingface_display_ref(
            "unsloth/gemma-4-31B-it-GGUF",
            None,
            "BF16/gemma-4-31B-it-BF16-00001-of-00002.gguf"
        ),
        "unsloth/gemma-4-31B-it-GGUF:BF16"
    );
}

#[tokio::test]
async fn download_exact_ref_bf16_shorthand_downloads_full_split_model() {
    let fixture = load_gemma_live_fixture();
    let _siblings_guard = RepoSiblingEntriesOverrideGuard::set(Arc::new({
        let repo = fixture.repo.clone();
        let siblings = fixture
            .siblings
            .iter()
            .map(|file| (file.clone(), fixture.size_bytes.get(file).copied()))
            .collect::<Vec<_>>();
        move |requested_repo, requested_revision| {
            if requested_repo == repo && requested_revision == "main" {
                Some(siblings.clone())
            } else {
                None
            }
        }
    }));

    let planned = Arc::new(Mutex::new(Vec::<(bool, String)>::new()));
    let _plan_guard = catalog::DownloadPlanObserverGuard::set(Arc::new({
        let planned = Arc::clone(&planned);
        move |label, entries| {
            if label == "unsloth/gemma-4-31B-it-GGUF:BF16" {
                *planned.lock().unwrap() = entries;
            }
        }
    }));
    let _download_guard = catalog::set_download_hf_assets_label_override(
        "unsloth/gemma-4-31B-it-GGUF:BF16".to_string(),
        Arc::new(|_| {
            Ok(vec![
                PathBuf::from("/tmp/BF16/gemma-4-31B-it-BF16-00001-of-00002.gguf"),
                PathBuf::from("/tmp/BF16/gemma-4-31B-it-BF16-00002-of-00002.gguf"),
            ])
        }),
    );

    let resolved = download_exact_ref_with_progress("unsloth/gemma-4-31B-it-GGUF:BF16", false)
        .await
        .unwrap();

    assert_eq!(
        resolved,
        PathBuf::from("/tmp/BF16/gemma-4-31B-it-BF16-00001-of-00002.gguf")
    );
    assert_eq!(
        *planned.lock().unwrap(),
        vec![
            (false, "config.json".to_string()),
            (
                true,
                "BF16/gemma-4-31B-it-BF16-00001-of-00002.gguf".to_string()
            ),
            (
                true,
                "BF16/gemma-4-31B-it-BF16-00002-of-00002.gguf".to_string()
            ),
        ]
    );
}

#[tokio::test]
async fn show_model_variants_accepts_selected_quant_ref() {
    let fixture = load_gemma_live_fixture();
    let _siblings_guard = RepoSiblingEntriesOverrideGuard::set(Arc::new({
        let repo = fixture.repo.clone();
        let siblings = fixture
            .siblings
            .iter()
            .map(|file| (file.clone(), fixture.size_bytes.get(file).copied()))
            .collect::<Vec<_>>();
        move |requested_repo, requested_revision| {
            if requested_repo == repo && requested_revision == "main" {
                Some(siblings.clone())
            } else {
                None
            }
        }
    }));

    let variants = show_model_variants_with_progress("unsloth/gemma-4-31B-it-GGUF:BF16", |_| {})
        .await
        .unwrap()
        .expect("repo-backed GGUF refs should enumerate variants");

    assert!(!variants.is_empty());
    assert!(variants
        .iter()
        .any(|variant| { variant.exact_ref == "unsloth/gemma-4-31B-it-GGUF:BF16" }));
    assert!(variants
        .iter()
        .any(|variant| { variant.exact_ref == "unsloth/gemma-4-31B-it-GGUF:UD-Q4_K_XL" }));
}

#[test]
fn format_huggingface_display_ref_prefers_repo_form_for_mlx() {
    assert_eq!(
        format_huggingface_display_ref("mlx-community/SmolLM-135M-8bit", None, "model.safetensors"),
        "mlx-community/SmolLM-135M-8bit"
    );
    assert_eq!(
        format_huggingface_display_ref(
            "avlp12/GLM-5.1-Alis-MLX-Dynamic-2.7bpw",
            None,
            "model-00001-of-00010.safetensors"
        ),
        "avlp12/GLM-5.1-Alis-MLX-Dynamic-2.7bpw"
    );
}

#[test]
fn parse_exact_model_ref_accepts_legacy_mlx_model_path_shape() {
    let parsed = parse_exact_model_ref("mlx-community/SmolLM-135M-8bit/model").unwrap();
    match parsed {
        ExactModelRef::HuggingFace {
            repo,
            revision,
            file,
        } => {
            assert_eq!(repo, "mlx-community/SmolLM-135M-8bit");
            assert_eq!(revision, None);
            assert_eq!(file, "model");
        }
        _ => panic!("expected HuggingFace ref"),
    }
}

#[test]
fn collect_show_gguf_variants_excludes_mmproj_and_nonfirst_split() {
    let siblings = vec![
        ("mmproj-BF16.gguf".to_string(), Some(1_200_000_000)),
        (
            "gemma-4-26B-A4B-it-UD-Q3_K_S-00002-of-00009.gguf".to_string(),
            Some(12_500_000_000),
        ),
        (
            "gemma-4-26B-A4B-it-UD-Q3_K_S-00001-of-00009.gguf".to_string(),
            Some(12_500_000_000),
        ),
        (
            "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf".to_string(),
            Some(16_900_000_000),
        ),
    ];
    let files: Vec<_> = collect_show_gguf_variants_from_siblings(&siblings, 0)
        .into_iter()
        .map(|(file, _)| file)
        .collect();
    assert_eq!(
        files,
        vec![
            "gemma-4-26B-A4B-it-UD-Q3_K_S-00001-of-00009.gguf".to_string(),
            "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf".to_string(),
        ]
    );
}

#[test]
fn collect_show_gguf_variants_orders_by_fit_when_memory_known() {
    let siblings = vec![
        ("model-UD-Q5_K_M.gguf".to_string(), Some(21_200_000_000)),
        ("model-UD-Q4_K_M.gguf".to_string(), Some(16_900_000_000)),
        ("model-UD-Q3_K_S.gguf".to_string(), Some(12_500_000_000)),
    ];
    let files: Vec<_> = collect_show_gguf_variants_from_siblings(&siblings, 19_300_000_000)
        .into_iter()
        .map(|(file, _)| file)
        .collect();
    assert_eq!(
        files,
        vec![
            "model-UD-Q4_K_M.gguf".to_string(),
            "model-UD-Q3_K_S.gguf".to_string(),
            "model-UD-Q5_K_M.gguf".to_string(),
        ]
    );
}
