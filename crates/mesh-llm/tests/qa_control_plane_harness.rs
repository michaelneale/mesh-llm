use std::fs;
#[cfg(unix)]
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("mesh-llm crate should live two levels below repo root")
        .to_path_buf()
}

fn harness_path() -> PathBuf {
    repo_root()
        .join("scripts")
        .join("qa-control-plane-mixed-version.sh")
}

fn unique_evidence_dir(test_name: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock should be after unix epoch")
        .as_nanos();
    std::env::temp_dir().join(format!(
        "mesh-llm-{test_name}-{}-{nanos}",
        std::process::id()
    ))
}

#[test]
fn mixed_version_qa_harness_exists_and_is_executable() {
    let path = harness_path();
    assert!(
        path.is_file(),
        "docs reference {}, but the harness is missing",
        path.display()
    );

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;

        let mode = std::fs::metadata(&path)
            .expect("harness metadata should be readable")
            .permissions()
            .mode();
        assert_ne!(mode & 0o111, 0, "harness should be executable");
    }
}

#[test]
fn mixed_version_qa_harness_help_documents_contract() {
    let output = Command::new("bash")
        .arg(harness_path())
        .arg("--help")
        .output()
        .expect("harness help should execute");

    assert!(
        output.status.success(),
        "harness --help failed: status={:?}, stderr={}",
        output.status.code(),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    for expected in [
        "--released-binary",
        "--current-binary",
        "--evidence-dir",
        "--local-only",
        "--config-only",
        "--print-plan",
        "manifest.json",
        "summary.json",
        "commands.jsonl",
        "config-missing-endpoint-required",
        "config-new-client-owner-control",
        "config-control-rejects-legacy-frames",
        "config-cargo-tests",
        "config-runtime-bootstrap",
        "mesh-llm-control/1",
    ] {
        assert!(
            stdout.contains(expected),
            "harness help should document {expected:?}; stdout was:\n{stdout}"
        );
    }
    for forbidden in [
        "PASS config-missing-endpoint-required",
        "PASS config-new-client-owner-control",
        "PASS config-control-rejects-legacy-frames",
    ] {
        assert!(
            !stdout.contains(forbidden),
            "harness help should not promise fixed PASS status for {forbidden:?}; stdout was:\n{stdout}"
        );
    }
}

#[test]
fn mixed_version_qa_harness_rejects_missing_required_binaries() {
    let output = Command::new("bash")
        .arg(harness_path())
        .arg("--local-only")
        .arg("--config-only")
        .output()
        .expect("harness argument validation should execute");

    assert!(
        !output.status.success(),
        "harness should reject missing binaries"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("--released-binary") && stderr.contains("--current-binary"),
        "missing-binary error should mention both required binary flags; stderr was:\n{stderr}"
    );
}

#[test]
fn mixed_version_qa_harness_rejects_unknown_arguments() {
    let output = Command::new("bash")
        .arg(harness_path())
        .arg("--definitely-unknown")
        .output()
        .expect("harness argument validation should execute");

    assert!(!output.status.success(), "unknown arguments should fail");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("unknown argument: --definitely-unknown"),
        "unknown-argument error should be explicit; stderr was:\n{stderr}"
    );
}

#[test]
fn mixed_version_qa_harness_print_plan_is_side_effect_free() {
    let evidence_dir = unique_evidence_dir("print-plan");
    let output = Command::new("bash")
        .arg(harness_path())
        .arg("--released-binary")
        .arg("/definitely/missing/released-mesh-llm")
        .arg("--current-binary")
        .arg("/definitely/missing/current-mesh-llm")
        .arg("--evidence-dir")
        .arg(&evidence_dir)
        .arg("--local-only")
        .arg("--config-only")
        .arg("--print-plan")
        .output()
        .expect("harness print-plan should execute");

    assert!(
        output.status.success(),
        "print-plan failed: status={:?}, stderr={}",
        output.status.code(),
        String::from_utf8_lossy(&output.stderr)
    );
    let plan: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("print-plan should emit valid JSON");
    let stdout = String::from_utf8_lossy(&output.stdout);
    for expected in [
        r#""config_only":true"#,
        r#""local_only":true"#,
        r#""public_mesh":false"#,
    ] {
        assert!(
            stdout.contains(expected),
            "print-plan should contain {expected:?}; stdout was:\n{stdout}"
        );
    }
    let checks = plan
        .get("checks")
        .and_then(serde_json::Value::as_array)
        .expect("print-plan should include a checks array");
    for expected in [
        "current-serves-released-client.loopback-coexistence",
        "released-serves-current-client.loopback-coexistence",
        "config-missing-endpoint-required",
        "config-new-client-owner-control",
        "config-control-rejects-legacy-frames",
        "config-runtime-bootstrap",
        "config-runtime-get-config",
        "cleanup",
    ] {
        assert!(
            checks.iter().any(|item| item.as_str() == Some(expected)),
            "print-plan checks should include exact result name {expected:?}; stdout was:\n{stdout}"
        );
    }
    for stale_name in [
        "public-current-client-auto",
        "loopback-current-to-released-peer-visible",
    ] {
        assert!(
            !checks.iter().any(|item| item.as_str() == Some(stale_name)),
            "print-plan checks should not include stale non-result name {stale_name:?}; stdout was:\n{stdout}"
        );
    }
    assert!(
        !evidence_dir.exists(),
        "print-plan must not create evidence directories"
    );
}

#[test]
fn mixed_version_qa_harness_starts_nodes_without_command_substitution() {
    let script = fs::read_to_string(harness_path()).expect("harness should be readable");

    assert!(
        !script.contains("$(start_node"),
        "start_node mutates parent-shell state, so callers must not use command substitution"
    );
    assert!(
        script.contains("START_NODE_PID"),
        "start_node should hand the child PID back through parent-shell state"
    );
}

#[cfg(unix)]
#[test]
fn mixed_version_qa_harness_records_empty_version_as_prereq() {
    use std::os::unix::fs::PermissionsExt;

    let temp_dir = tempfile::tempdir().expect("temp dir should be creatable");
    let fake_binary = temp_dir.path().join("fake-mesh-llm");
    let mut file = fs::File::create(&fake_binary).expect("fake binary should be writable");
    writeln!(
        file,
        "#!/usr/bin/env bash\nif [[ \"${{1:-}}\" == \"--version\" ]]; then\n  exit 0\nfi\nexit 1"
    )
    .expect("fake binary should be writable");
    drop(file);
    let mut permissions = fs::metadata(&fake_binary)
        .expect("fake binary metadata should be readable")
        .permissions();
    permissions.set_mode(0o755);
    fs::set_permissions(&fake_binary, permissions).expect("fake binary should be executable");

    let evidence_dir = unique_evidence_dir("empty-version");
    let output = Command::new("bash")
        .arg(harness_path())
        .arg("--released-binary")
        .arg(&fake_binary)
        .arg("--current-binary")
        .arg(&fake_binary)
        .arg("--evidence-dir")
        .arg(&evidence_dir)
        .arg("--local-only")
        .arg("--config-only")
        .arg("--skip-cargo-tests")
        .arg("--max-wait")
        .arg("1")
        .output()
        .expect("harness should execute");

    assert!(
        !output.status.success(),
        "fake binary should make the real harness run fail after prereq recording"
    );

    let run_dir = fs::read_dir(&evidence_dir)
        .expect("evidence directory should exist")
        .next()
        .expect("one evidence run should be present")
        .expect("evidence entry should be readable")
        .path();
    let results = fs::read_to_string(run_dir.join("results.jsonl"))
        .expect("results.jsonl should be readable");
    let records: Vec<serde_json::Value> = results
        .lines()
        .map(|line| serde_json::from_str(line).expect("result record should be JSON"))
        .collect();

    for name in ["prereq.released-binary", "prereq.current-binary"] {
        assert!(
            records.iter().any(|record| {
                record.get("name").and_then(serde_json::Value::as_str) == Some(name)
                    && record.get("status").and_then(serde_json::Value::as_str) == Some("PREREQ")
            }),
            "{name} should be recorded as PREREQ when --version is empty; results were:\n{results}"
        );
    }
    let _ = fs::remove_dir_all(&evidence_dir);
}
