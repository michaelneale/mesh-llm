#[cfg(test)]
mod tests {
    use super::*;

    fn make_survey(
        gpu_count: u8,
        gpu_vram: Vec<u64>,
        gpu_name: Option<&str>,
        is_soc: bool,
    ) -> HardwareSurvey {
        HardwareSurvey {
            gpu_count,
            gpu_vram,
            gpu_name: gpu_name.map(str::to_owned),
            is_soc,
            ..Default::default()
        }
    }

    fn make_fingerprint(gpus: Vec<GpuBandwidth>, is_soc: bool) -> BenchmarkFingerprint {
        BenchmarkFingerprint {
            gpus,
            is_soc,
            timestamp_secs: 0,
        }
    }

    fn build_output(fp32: Option<f64>, fp16: Option<f64>) -> BenchmarkOutput {
        BenchmarkOutput {
            device: "Test GPU".into(),
            buffer_mb: 0,
            runs: 0,
            p50_gbps: 1.0,
            p90_gbps: 2.0,
            compute_tflops_fp32: fp32,
            compute_tflops_fp16: fp16,
            noise_pct: 0.0,
            runtime_s: 0.0,
            rated_gbps: None,
            rated_estimated: None,
            efficiency_pct: None,
            bus_width_bits: None,
            mem_clock_mhz: None,
            gcn_arch: None,
            hbm: None,
        }
    }

    fn make_hw_with_gpus() -> HardwareSurvey {
        HardwareSurvey {
            gpu_vram: vec![64_000_000_000],
            gpu_name: Some("Test GPU".into()),
            gpu_count: 1,
            is_soc: false,
            gpus: vec![GpuFacts {
                index: 0,
                display_name: "Test GPU".into(),
                backend_device: None,
                vram_bytes: 64_000_000_000,
                reserved_bytes: None,
                mem_bandwidth_gbps: None,
                compute_tflops_fp32: None,
                compute_tflops_fp16: None,
                unified_memory: false,
                stable_id: None,
                pci_bdf: None,
                vendor_uuid: None,
                metal_registry_id: None,
                dxgi_luid: None,
                pnp_instance_id: None,
            }],
            ..Default::default()
        }
    }

    // 1. Same hardware → false
    #[test]
    fn test_hardware_changed_same() {
        let hw = make_survey(1, vec![80_000_000_000], Some("A100"), false);
        let fp = make_fingerprint(
            vec![GpuBandwidth {
                name: "A100".into(),
                vram_bytes: 80_000_000_000,
                p50_gbps: 1935.0,
                p90_gbps: 1948.7,
                compute_tflops_fp32: None,
                compute_tflops_fp16: None,
            }],
            false,
        );
        assert!(!hardware_changed(&fp, &hw));
    }

    // 2. VRAM differs → true
    #[test]
    fn test_hardware_changed_vram() {
        let hw = make_survey(1, vec![40_000_000_000], Some("A100"), false);
        let fp = make_fingerprint(
            vec![GpuBandwidth {
                name: "A100".into(),
                vram_bytes: 80_000_000_000,
                p50_gbps: 1935.0,
                p90_gbps: 1948.7,
                compute_tflops_fp32: None,
                compute_tflops_fp16: None,
            }],
            false,
        );
        assert!(hardware_changed(&fp, &hw));
    }

    // 3. GPU count differs → true
    #[test]
    fn test_hardware_changed_gpu_count() {
        let hw = make_survey(
            2,
            vec![80_000_000_000, 80_000_000_000],
            Some("A100, A100"),
            false,
        );
        let fp = make_fingerprint(
            vec![GpuBandwidth {
                name: "A100".into(),
                vram_bytes: 80_000_000_000,
                p50_gbps: 1935.0,
                p90_gbps: 1948.7,
                compute_tflops_fp32: None,
                compute_tflops_fp16: None,
            }],
            false,
        );
        assert!(hardware_changed(&fp, &hw));
    }

    // 4. is_soc differs → true
    #[test]
    fn test_hardware_changed_soc_flag() {
        let hw = make_survey(1, vec![16_000_000_000], None, false);
        let fp = make_fingerprint(vec![], true); // is_soc: true vs false
        assert!(hardware_changed(&fp, &hw));
    }

    // 5. Parse single CUDA GPU JSON — assert p90_gbps == 1948.7
    #[test]
    fn test_benchmark_output_deserialize_cuda_single() {
        let json_str = r#"[{"device":"NVIDIA A100-SXM4-80GB","buffer_mb":512,"runs":20,"p50_gbps":1935.2,"p90_gbps":1948.7,"compute_tflops_fp32":19.5,"compute_tflops_fp16":312.0,"noise_pct":0.4,"runtime_s":1.23,"rated_gbps":2000,"rated_estimated":false,"efficiency_pct":96.8,"bus_width_bits":5120,"mem_clock_mhz":1215}]"#;
        let outputs: Vec<BenchmarkOutput> = serde_json::from_str(json_str).expect("should parse");
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].p90_gbps, 1948.7);
        assert_eq!(outputs[0].compute_tflops_fp32, Some(19.5));
        assert_eq!(outputs[0].compute_tflops_fp16, Some(312.0));
    }

    // 6. Parse 2-device JSON — assert both entries deserialize
    #[test]
    fn test_benchmark_output_deserialize_multi_gpu() {
        let json_str = r#"[{"device":"NVIDIA A100","buffer_mb":512,"runs":20,"p50_gbps":1935.2,"p90_gbps":1948.7,"compute_tflops_fp32":19.5,"compute_tflops_fp16":312.0,"noise_pct":0.4,"runtime_s":1.23,"rated_gbps":2000,"rated_estimated":false,"efficiency_pct":96.8,"bus_width_bits":5120,"mem_clock_mhz":1215},{"device":"NVIDIA A6000","buffer_mb":512,"runs":20,"p50_gbps":768.0,"p90_gbps":780.1,"compute_tflops_fp32":38.7,"compute_tflops_fp16":77.4,"noise_pct":0.6,"runtime_s":1.15,"rated_gbps":768,"rated_estimated":false,"efficiency_pct":100.0,"bus_width_bits":384,"mem_clock_mhz":2000}]"#;
        let outputs: Vec<BenchmarkOutput> = serde_json::from_str(json_str).expect("should parse");
        assert_eq!(outputs.len(), 2);
    }

    // 7. Error JSON (object, not array) → Err, no panic
    #[test]
    fn test_benchmark_output_deserialize_error_json() {
        let json_str = r#"{"error":"No CUDA-capable device found"}"#;
        let result = serde_json::from_str::<Vec<BenchmarkOutput>>(json_str);
        assert!(result.is_err(), "expected Err, got Ok");
    }

    // 8. parse_benchmark_output: single GPU → Some(vec with 1 entry, p90 == 1948.7)
    #[test]
    fn test_parse_benchmark_output_single_gpu() {
        let json = r#"[{"device":"NVIDIA A100-SXM4-80GB","buffer_mb":512,"runs":20,"p50_gbps":1935.2,"p90_gbps":1948.7,"compute_tflops_fp32":19.5,"compute_tflops_fp16":312.0,"noise_pct":0.4,"runtime_s":1.23,"rated_gbps":2000,"rated_estimated":false,"efficiency_pct":96.8,"bus_width_bits":5120,"mem_clock_mhz":1215}]"#;
        let result = parse_benchmark_output(json.as_bytes()).expect("should return Some");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].p90_gbps, 1948.7);
    }

    // 9. parse_benchmark_output: two GPUs → Some(vec with 2 entries), sum ~2728.8
    #[test]
    fn test_parse_benchmark_output_multi_gpu_sum() {
        let json = r#"[{"device":"NVIDIA A100","buffer_mb":512,"runs":20,"p50_gbps":1935.2,"p90_gbps":1948.7,"compute_tflops_fp32":19.5,"compute_tflops_fp16":312.0,"noise_pct":0.4,"runtime_s":1.23,"rated_gbps":2000,"rated_estimated":false,"efficiency_pct":96.8,"bus_width_bits":5120,"mem_clock_mhz":1215},{"device":"NVIDIA A6000","buffer_mb":512,"runs":20,"p50_gbps":768.0,"p90_gbps":780.1,"compute_tflops_fp32":38.7,"compute_tflops_fp16":77.4,"noise_pct":0.6,"runtime_s":1.15,"rated_gbps":768,"rated_estimated":false,"efficiency_pct":100.0,"bus_width_bits":384,"mem_clock_mhz":2000}]"#;
        let outputs = parse_benchmark_output(json.as_bytes()).expect("should return Some");
        assert_eq!(outputs.len(), 2);
        let sum: f64 = outputs.iter().map(|o| o.p90_gbps).sum();
        assert!(
            (sum - 2728.8_f64).abs() < 0.01,
            "expected ~2728.8, got {sum}"
        );
    }

    // 10. parse_benchmark_output: error object → None
    #[test]
    fn test_parse_benchmark_output_error_json() {
        let json = r#"{"error": "No CUDA devices found"}"#;
        let result = parse_benchmark_output(json.as_bytes());
        assert!(result.is_none());
    }

    // 11. parse_benchmark_output: empty array → None
    #[test]
    fn test_parse_benchmark_output_empty_array() {
        let result = parse_benchmark_output(b"[]");
        assert!(result.is_none());
    }

    // 12. detect_benchmark_binary: gpu_count == 0 → None (no process spawned)
    #[test]
    fn test_detect_benchmark_binary_gpu_count_zero() {
        let hw = HardwareSurvey {
            gpu_count: 0,
            ..Default::default()
        };
        let result = detect_benchmark_binary(&hw, Path::new("/tmp"));
        assert!(result.is_none());
    }

    #[test]
    fn test_benchmark_binary_name_for_windows() {
        assert_eq!(
            benchmark_binary_name_for("windows", "membench-fingerprint-cuda"),
            "membench-fingerprint-cuda.exe"
        );
        assert_eq!(
            benchmark_binary_name_for("linux", "membench-fingerprint-cuda"),
            "membench-fingerprint-cuda"
        );
    }

    #[test]
    fn test_detect_benchmark_binary_windows_cuda_missing_is_soft_failure() {
        let hw = make_survey(1, vec![24_000_000_000], Some("NVIDIA RTX 4090"), false);
        assert!(detect_benchmark_binary_for_with_exe_dir(
            "windows",
            &hw,
            Path::new("C:\\bench"),
            None,
        )
        .is_none());
    }

    #[test]
    fn test_detect_benchmark_binary_windows_hip_missing_is_soft_failure() {
        let hw = make_survey(
            1,
            vec![24_000_000_000],
            Some("AMD Radeon RX 7900 XTX"),
            false,
        );
        assert!(detect_benchmark_binary_for_with_exe_dir(
            "windows",
            &hw,
            Path::new("C:\\bench"),
            None,
        )
        .is_none());
    }

    #[test]
    fn test_detect_benchmark_binary_windows_intel_missing_is_soft_failure() {
        let hw = make_survey(1, vec![16_000_000_000], Some("Intel Arc A770"), false);
        assert!(detect_benchmark_binary_for_with_exe_dir(
            "windows",
            &hw,
            Path::new("C:\\bench"),
            None,
        )
        .is_none());
    }

    #[test]
    fn test_detect_benchmark_binary_linux_cuda_finds_crate_release_helper() {
        let root =
            std::env::temp_dir().join(format!("mesh-llm-benchmark-lookup-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let bin_dir = root.join("llama.cpp/build/bin");
        let exe_dir = root.join("target/release");
        let crate_release = root.join("target/release");
        std::fs::create_dir_all(&bin_dir).expect("create bin dir");
        std::fs::create_dir_all(&exe_dir).expect("create exe dir");
        std::fs::create_dir_all(&crate_release).expect("create crate release dir");

        let helper = crate_release.join("membench-fingerprint-cuda");
        std::fs::write(&helper, b"").expect("write helper");

        let hw = make_survey(1, vec![24_000_000_000], Some("NVIDIA RTX 4090"), false);
        let result = detect_benchmark_binary_for_with_exe_dir(
            "linux",
            &hw,
            &bin_dir,
            Some(exe_dir.as_path()),
        );
        let expected = helper.canonicalize().expect("canonical helper path");

        let _ = std::fs::remove_dir_all(&root);

        assert_eq!(result.as_deref(), Some(expected.as_path()));
    }

    #[test]
    fn test_detect_benchmark_binary_macos_soc_finds_crate_release_helper() {
        let root = std::env::temp_dir().join(format!(
            "mesh-llm-benchmark-lookup-macos-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        let bin_dir = root.join("llama.cpp/build/bin");
        let exe_dir = root.join("target/release");
        let crate_release = root.join("target/release");
        std::fs::create_dir_all(&bin_dir).expect("create bin dir");
        std::fs::create_dir_all(&exe_dir).expect("create exe dir");
        std::fs::create_dir_all(&crate_release).expect("create crate release dir");

        let helper = crate_release.join("membench-fingerprint");
        std::fs::write(&helper, b"").expect("write helper");

        let hw = make_survey(1, vec![24_000_000_000], Some("Apple M4 Pro"), true);
        let result = detect_benchmark_binary_for_with_exe_dir(
            "macos",
            &hw,
            &bin_dir,
            Some(exe_dir.as_path()),
        );
        let expected = helper.canonicalize().expect("canonical helper path");

        let _ = std::fs::remove_dir_all(&root);

        assert_eq!(result.as_deref(), Some(expected.as_path()));
    }

    // 13. hardware_changed: same VRAM, different GPU name → true
    #[test]
    fn test_hardware_changed_gpu_name() {
        let hw = make_survey(1, vec![80_000_000_000], Some("NVIDIA A6000"), false);
        let fp = make_fingerprint(
            vec![GpuBandwidth {
                name: "NVIDIA A100".into(),
                vram_bytes: 80_000_000_000,
                p50_gbps: 1935.0,
                p90_gbps: 1948.7,
                compute_tflops_fp32: None,
                compute_tflops_fp16: None,
            }],
            false,
        );
        assert!(
            hardware_changed(&fp, &hw),
            "name change should trigger hardware_changed"
        );
    }

    // 14. Cache round-trip: save → load → hardware_changed returns false for same hw
    #[test]
    fn test_fingerprint_cache_roundtrip() {
        let path = std::env::temp_dir().join("mesh-llm-test-fingerprint-roundtrip.json");
        let fp = make_fingerprint(
            vec![GpuBandwidth {
                name: "NVIDIA A100".into(),
                vram_bytes: 80_000_000_000,
                p50_gbps: 1935.2,
                p90_gbps: 1948.7,
                compute_tflops_fp32: Some(19.5),
                compute_tflops_fp16: Some(312.0),
            }],
            false,
        );
        save_fingerprint(&path, &fp);
        let loaded = load_fingerprint(&path).expect("fingerprint should round-trip");
        let _ = std::fs::remove_file(&path);

        let hw = make_survey(1, vec![80_000_000_000], Some("NVIDIA A100"), false);
        assert!(
            !hardware_changed(&loaded, &hw),
            "same hardware should not trigger hardware_changed after round-trip"
        );
    }

    #[test]
    fn test_try_save_fingerprint_overwrites_existing_cache() {
        let path = std::env::temp_dir().join("mesh-llm-test-fingerprint-overwrite.json");
        std::fs::write(&path, "stale").expect("seed existing cache");

        let fp = make_fingerprint(
            vec![GpuBandwidth {
                name: "NVIDIA A100".into(),
                vram_bytes: 80_000_000_000,
                p50_gbps: 1935.2,
                p90_gbps: 1948.7,
                compute_tflops_fp32: Some(19.5),
                compute_tflops_fp16: Some(312.0),
            }],
            false,
        );

        try_save_fingerprint(&path, &fp).expect("fingerprint should overwrite existing cache");
        let loaded = load_fingerprint(&path).expect("fingerprint should load after overwrite");
        let _ = std::fs::remove_file(&path);

        assert_eq!(loaded.gpus[0].p90_gbps, 1948.7);
    }

    #[cfg(unix)]
    #[test]
    fn test_run_and_save_rewrites_existing_cache() {
        use std::os::unix::fs::PermissionsExt;

        let root = std::env::temp_dir().join(format!(
            "mesh-llm-run-and-save-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system time")
                .as_nanos()
        ));
        let bin_dir = root.join("bin");
        let path = root.join("benchmark-fingerprint.json");
        std::fs::create_dir_all(&bin_dir).expect("create bin dir");

        let binary_name = if cfg!(target_os = "macos") {
            "membench-fingerprint"
        } else {
            "membench-fingerprint-cuda"
        };
        let binary = bin_dir.join(binary_name);
        std::fs::write(
            &binary,
            "#!/bin/sh\nprintf '%s\n' '[{\"device\":\"Test GPU\",\"buffer_mb\":512,\"runs\":2,\"p50_gbps\":111.0,\"p90_gbps\":222.0,\"noise_pct\":0.1,\"runtime_s\":0.5}]'\n",
        )
        .expect("write fake benchmark");
        std::fs::set_permissions(&binary, std::fs::Permissions::from_mode(0o755))
            .expect("chmod fake benchmark");

        let old = make_fingerprint(
            vec![GpuBandwidth {
                name: "Test GPU".into(),
                vram_bytes: 64_000_000_000,
                p50_gbps: 1.0,
                p90_gbps: 2.0,
                compute_tflops_fp32: None,
                compute_tflops_fp16: None,
            }],
            cfg!(target_os = "macos"),
        );
        try_save_fingerprint(&path, &old).expect("seed fingerprint cache");

        let hw = HardwareSurvey {
            gpu_count: 1,
            gpu_vram: vec![64_000_000_000],
            gpu_name: Some(if cfg!(target_os = "macos") {
                "Apple M4 Pro".into()
            } else {
                "NVIDIA RTX 4090".into()
            }),
            is_soc: cfg!(target_os = "macos"),
            ..Default::default()
        };

        let saved = run_and_save_to_path(&hw, &bin_dir, Duration::from_secs(2), &path)
            .expect("forced benchmark should succeed");
        let loaded = load_fingerprint(&path).expect("fingerprint should exist");
        let _ = std::fs::remove_dir_all(&root);

        assert_eq!(saved.result.mem_bandwidth_gbps, vec![222.0]);
        assert_eq!(loaded.gpus[0].p90_gbps, 222.0);
    }

    #[test]
    fn test_run_and_save_missing_binary_fails_cleanly() {
        let root = std::env::temp_dir().join(format!(
            "mesh-llm-run-and-save-missing-{}",
            std::process::id()
        ));
        let bin_dir = root.join("bin");
        let path = root.join("benchmark-fingerprint.json");
        std::fs::create_dir_all(&bin_dir).expect("create bin dir");

        let hw = HardwareSurvey {
            gpu_count: 1,
            gpu_vram: vec![64_000_000_000],
            gpu_name: Some(if cfg!(target_os = "macos") {
                "Apple M4 Pro".into()
            } else {
                "NVIDIA RTX 4090".into()
            }),
            is_soc: cfg!(target_os = "macos"),
            ..Default::default()
        };

        let err = run_and_save_to_path(&hw, &bin_dir, Duration::from_secs(1), &path)
            .expect_err("missing benchmark binary should fail");
        let _ = std::fs::remove_dir_all(&root);

        assert!(err.to_string().contains("benchmark binary"));
    }

    // 15. Old cache format (hardware_key field) fails to parse → load_fingerprint returns None
    #[test]
    fn test_old_cache_format_fails_parse() {
        let old_json = r#"{
            "hardware_key": {
                "gpu_count": 1,
                "gpu_vram": [80000000000],
                "gpu_name": "NVIDIA A100",
                "is_soc": false
            },
            "mem_bandwidth_gbps": 1948.7,
            "p50_gbps": 1935.2,
            "timestamp_secs": 1700000000
        }"#;
        let path = std::env::temp_dir().join("mesh-llm-test-fingerprint-old-format.json");
        std::fs::write(&path, old_json).expect("write should succeed");
        let result = load_fingerprint(&path);
        let _ = std::fs::remove_file(&path);
        assert!(
            result.is_none(),
            "old cache format should fail to parse and return None"
        );
    }

    #[test]
    fn test_benchmark_output_deserializes_without_tflops_fields() {
        let json = r#"[{"device":"NVIDIA A100","buffer_mb":512,"runs":20,"p50_gbps":1935.2,"p90_gbps":1948.7,"noise_pct":0.4,"runtime_s":1.23,"rated_gbps":2000,"rated_estimated":false,"efficiency_pct":96.8,"bus_width_bits":5120,"mem_clock_mhz":1215}]"#;
        let outputs: Vec<BenchmarkOutput> = serde_json::from_str(json).expect("should parse");

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].compute_tflops_fp32, None);
        assert_eq!(outputs[0].compute_tflops_fp16, None);
    }

    #[test]
    fn test_benchmark_output_deserializes_with_tflops_fields() {
        let json = r#"[{"device":"NVIDIA A100","buffer_mb":512,"runs":20,"p50_gbps":1935.2,"p90_gbps":1948.7,"compute_tflops_fp32":19.5,"compute_tflops_fp16":312.0,"noise_pct":0.4,"runtime_s":1.23,"rated_gbps":2000,"rated_estimated":false,"efficiency_pct":96.8,"bus_width_bits":5120,"mem_clock_mhz":1215}]"#;
        let outputs: Vec<BenchmarkOutput> = serde_json::from_str(json).expect("should parse");

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].compute_tflops_fp32, Some(19.5));
        assert_eq!(outputs[0].compute_tflops_fp16, Some(312.0));
    }

    #[test]
    fn test_benchmark_output_deserializes_fp32_only() {
        let json = r#"[{"device":"NVIDIA A100","buffer_mb":512,"runs":20,"p50_gbps":1935.2,"p90_gbps":1948.7,"compute_tflops_fp32":19.5,"noise_pct":0.4,"runtime_s":1.23,"rated_gbps":2000,"rated_estimated":false,"efficiency_pct":96.8,"bus_width_bits":5120,"mem_clock_mhz":1215}]"#;
        let outputs: Vec<BenchmarkOutput> = serde_json::from_str(json).expect("should parse");

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].compute_tflops_fp32, Some(19.5));
        assert_eq!(outputs[0].compute_tflops_fp16, None);
    }

    #[test]
    fn test_gpu_bandwidth_serde_round_trip_with_tflops() {
        let gpu = GpuBandwidth {
            name: "NVIDIA A100".into(),
            vram_bytes: 80_000_000_000,
            p50_gbps: 1935.2,
            p90_gbps: 1948.7,
            compute_tflops_fp32: Some(19.5),
            compute_tflops_fp16: Some(312.0),
        };

        let json = serde_json::to_string(&gpu).expect("should serialize");
        let round_trip: GpuBandwidth = serde_json::from_str(&json).expect("should deserialize");

        assert_eq!(round_trip, gpu);
    }

    #[test]
    fn test_gpu_bandwidth_omits_missing_tflops_fields_when_serializing() {
        let gpu = GpuBandwidth {
            name: "NVIDIA A100".into(),
            vram_bytes: 80_000_000_000,
            p50_gbps: 1935.2,
            p90_gbps: 1948.7,
            compute_tflops_fp32: None,
            compute_tflops_fp16: None,
        };

        let value = serde_json::to_value(&gpu).expect("should serialize");
        let object = value
            .as_object()
            .expect("GpuBandwidth should serialize as an object");

        assert!(!object.contains_key("compute_tflops_fp32"));
        assert!(!object.contains_key("compute_tflops_fp16"));
    }

    #[test]
    fn test_benchmark_result_tflops_none_when_binary_has_no_tflops() {
        let hw = make_hw_with_gpus();
        let output = build_output(None, None);
        let (_, result) = build_benchmark_result(&hw, &[output]);

        assert!(result.compute_tflops_fp32.is_none());
        assert!(result.compute_tflops_fp16.is_none());
    }

    #[test]
    fn test_benchmark_result_fp16_not_derived_when_fp32_available() {
        let hw = make_hw_with_gpus();
        let output = build_output(Some(19.5), None);
        let (_, result) = build_benchmark_result(&hw, &[output]);

        assert_eq!(result.compute_tflops_fp32, Some(vec![19.5]));
        assert!(result.compute_tflops_fp16.is_none());
    }

    #[test]
    fn test_benchmark_result_does_not_backfill_hardware_tflops() {
        let mut hw = make_hw_with_gpus();
        hw.gpus[0].compute_tflops_fp32 = Some(123.0);
        hw.gpus[0].compute_tflops_fp16 = Some(456.0);
        let output = build_output(None, None);
        let (_, result) = build_benchmark_result(&hw, &[output]);

        assert!(result.compute_tflops_fp32.is_none());
        assert!(result.compute_tflops_fp16.is_none());
    }

    #[test]
    fn test_build_benchmark_result_expands_identical_multi_gpu_names() {
        let hw = make_survey(
            2,
            vec![80_000_000_000, 80_000_000_000],
            Some("2× NVIDIA A100"),
            false,
        );
        let outputs = vec![
            BenchmarkOutput {
                device: "GPU 0".into(),
                buffer_mb: 512,
                runs: 2,
                p50_gbps: 100.0,
                p90_gbps: 110.0,
                compute_tflops_fp32: None,
                compute_tflops_fp16: None,
                noise_pct: 0.0,
                runtime_s: 0.0,
                rated_gbps: None,
                rated_estimated: None,
                efficiency_pct: None,
                bus_width_bits: None,
                mem_clock_mhz: None,
                gcn_arch: None,
                hbm: None,
            },
            BenchmarkOutput {
                device: "GPU 1".into(),
                buffer_mb: 512,
                runs: 2,
                p50_gbps: 120.0,
                p90_gbps: 130.0,
                compute_tflops_fp32: None,
                compute_tflops_fp16: None,
                noise_pct: 0.0,
                runtime_s: 0.0,
                rated_gbps: None,
                rated_estimated: None,
                efficiency_pct: None,
                bus_width_bits: None,
                mem_clock_mhz: None,
                gcn_arch: None,
                hbm: None,
            },
        ];

        let (gpus, result) = build_benchmark_result(&hw, &outputs);
        let fingerprint = make_fingerprint(gpus.clone(), false);

        assert_eq!(gpus.len(), 2);
        assert_eq!(gpus[0].name, "NVIDIA A100");
        assert_eq!(gpus[1].name, "NVIDIA A100");
        assert_eq!(result.mem_bandwidth_gbps, vec![110.0, 130.0]);
        assert!(!hardware_changed(&fingerprint, &hw));
    }

    #[test]
    fn test_old_fingerprint_cache_loads_without_tflops() {
        let json = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/pre-tops-fingerprint.json"
        ));
        let path = std::env::temp_dir().join("mesh-llm-test-fingerprint-pre-tops.json");
        std::fs::write(&path, json).expect("write should succeed");

        let fingerprint = load_fingerprint(&path).expect("old-format fingerprint should parse");
        let _ = std::fs::remove_file(&path);

        assert_eq!(fingerprint.gpus.len(), 1);
        assert_eq!(fingerprint.gpus[0].name, "NVIDIA A100");
        assert_eq!(fingerprint.gpus[0].compute_tflops_fp32, None);
        assert_eq!(fingerprint.gpus[0].compute_tflops_fp16, None);
    }

    #[test]
    fn test_fingerprint_path_filename() {
        let path = fingerprint_path();
        assert!(
            path.ends_with("benchmark-fingerprint.json"),
            "fingerprint_path() should use 'benchmark-fingerprint.json', got {:?}",
            path.file_name()
        );
        let parent = path.parent().expect("path should have parent");
        assert!(
            parent.ends_with("mesh-llm"),
            "fingerprint should be under mesh-llm cache directory, got {:?}",
            parent
        );
    }

    #[cfg(unix)]
    #[test]
    fn test_run_benchmark_kills_on_timeout() {
        use std::os::unix::fs::PermissionsExt;

        let dir = std::env::temp_dir().join("mesh-llm-test-bm-timeout");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create test dir");

        let script = dir.join("hang.sh");
        std::fs::write(&script, "#!/bin/sh\nsleep 999\n").expect("write script");
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755))
            .expect("set permissions");

        let start = std::time::Instant::now();
        let result = run_benchmark(&script, Duration::from_secs(1));
        let elapsed = start.elapsed();

        let _ = std::fs::remove_dir_all(&dir);

        assert!(
            result.is_none(),
            "hanging benchmark should return None on timeout"
        );
        assert!(
            elapsed < Duration::from_secs(5),
            "should kill subprocess promptly, took {:?}",
            elapsed
        );
    }
}
