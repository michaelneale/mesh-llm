#!/usr/bin/env python3
"""Run the shared GGUF/MLX validation matrix locally.

This orchestrates the deterministic exact suite and the MT-Bench-derived
behavior suite from one checked-in matrix definition. Each backend can be run
independently, or both can be run together, while preserving raw artifacts
under one stamped results tree.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = REPO_ROOT / "testdata" / "validation" / "matrix.json"
DEFAULT_BASELINES = REPO_ROOT / "testdata" / "validation" / "baselines.json"
DEFAULT_ROOT = REPO_ROOT / "MLX_VALIDATION_RESULTS"
DEFAULT_WAIT_SECONDS = 300
COMMON_BIN_DIRS = ["/opt/homebrew/bin", "/usr/local/bin"]


def log(message: str) -> None:
    sys.stderr.write(message + "\n")
    sys.stderr.flush()


def merged_env(env: dict[str, str] | None = None) -> dict[str, str]:
    base = dict(os.environ)
    if env:
        base.update(env)
    path_entries = [entry for entry in base.get("PATH", "").split(os.pathsep) if entry]
    for entry in reversed(COMMON_BIN_DIRS):
        if entry not in path_entries:
            path_entries.insert(0, entry)
    base["PATH"] = os.pathsep.join(path_entries)
    return base


def resolve_command(cmd: list[str], env: dict[str, str]) -> list[str]:
    if not cmd:
        return cmd
    executable = cmd[0]
    if "/" in executable:
        return cmd
    resolved = shutil.which(executable, path=env.get("PATH", ""))
    if resolved:
        return [resolved, *cmd[1:]]
    return cmd


def run(
    cmd: list[str],
    *,
    cwd: Path = REPO_ROOT,
    env: dict[str, str] | None = None,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    final_env = merged_env(env)
    return subprocess.run(
        resolve_command(cmd, final_env),
        cwd=cwd,
        env=final_env,
        text=True,
        capture_output=capture_output,
        check=False,
    )


def run_streaming(
    cmd: list[str],
    *,
    cwd: Path = REPO_ROOT,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    final_env = merged_env(env)
    proc = subprocess.Popen(
        resolve_command(cmd, final_env),
        cwd=cwd,
        env=final_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
    )
    chunks: list[bytes] = []
    assert proc.stdout is not None
    stderr_buffer = getattr(sys.stderr, "buffer", None)
    while True:
        chunk = proc.stdout.read(4096)
        if not chunk:
            break
        chunks.append(chunk)
        if stderr_buffer is not None:
            stderr_buffer.write(chunk)
            stderr_buffer.flush()
        else:
            sys.stderr.write(chunk.decode("utf-8", errors="replace"))
            sys.stderr.flush()
    returncode = proc.wait()
    return subprocess.CompletedProcess(
        cmd,
        returncode,
        b"".join(chunks).decode("utf-8", errors="replace"),
        "",
    )


def ensure_build(skip_build: bool) -> None:
    if skip_build:
        return
    rc = run(["just", "build"])
    if rc.returncode != 0:
        raise SystemExit(rc.returncode)


def load_matrix(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_baselines(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def selected_models(
    matrix: dict[str, Any],
    selectors: set[str],
    backend_filter: str,
) -> list[dict[str, Any]]:
    models: list[dict[str, Any]] = []
    for model in matrix["models"]:
        if selectors:
            candidate_keys = {
                model["id"],
                model["label"],
            }
            for backend_name in ("gguf", "mlx"):
                if backend_name in model:
                    candidate_keys.add(model[backend_name].get("exact_case_id", ""))
                    candidate_keys.add(model[backend_name].get("behavior_case_id", ""))
            if not candidate_keys.intersection(selectors):
                continue
        if backend_filter in ("gguf", "mlx") and backend_filter not in model:
            continue
        models.append(model)
    return models


def requested_backends(model: dict[str, Any], backend_filter: str) -> list[str]:
    if backend_filter == "both":
        return [backend for backend in ("gguf", "mlx") if backend in model]
    if backend_filter in model:
        return [backend_filter]
    return []


def parse_downloaded_gguf_path(output: str) -> str:
    for line in output.splitlines():
        trimmed = line.strip()
        if trimmed.startswith("/") and trimmed.endswith(".gguf"):
            return trimmed
    raise RuntimeError("could not determine downloaded gguf path")


def parse_downloaded_model_path(output: str) -> str:
    for line in output.splitlines():
        trimmed = line.strip()
        if not trimmed.startswith("/"):
            continue
        if trimmed.endswith(".gguf") or trimmed.endswith(".json") or trimmed.endswith(".safetensors"):
            return trimmed
    raise RuntimeError("could not determine downloaded model path")


def download_model_ref(model_ref: str, backend: str) -> str:
    flag = "--gguf" if backend == "gguf" else "--mlx"
    log(f"📥 Preflight download start [{backend}] {model_ref}")
    proc = run_streaming(
        ["./target/release/mesh-llm", "models", "download", model_ref, flag],
    )
    if proc.returncode != 0:
        log(f"❌ Preflight download failed [{backend}] {model_ref} (exit {proc.returncode})")
        raise SystemExit(proc.returncode)
    local_path = parse_downloaded_model_path(proc.stdout)
    log(f"✅ Preflight download complete [{backend}] {model_ref}")
    log(f"   ↳ {local_path}")
    return local_path


def summary_path(root: Path, stamp: str, suite: str) -> Path:
    return root / stamp / f"{suite}-summary.tsv"


def suite_stamp(stamp: str, suite: str) -> str:
    return f"{stamp}/{suite}"


def append_tsv(path: Path, header: list[str], row: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("\t".join(header) + "\n", encoding="utf-8")
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\t".join(row) + "\n")


def case_dir(root: Path, stamp: str, suite: str, case_id: str) -> Path:
    return root / stamp / suite / case_id


def run_exact_case(
    root: Path,
    stamp: str,
    matrix: dict[str, Any],
    model: dict[str, Any],
    backend: str,
    resolved_models: dict[tuple[str, str], str],
) -> int:
    exact_defaults = matrix["defaults"]["exact"]
    backend_cfg = model[backend]
    case_id = backend_cfg["exact_case_id"]
    env = {
        **os.environ,
        "VALIDATION_RESULTS_ROOT": str(root),
        "VALIDATION_RESULTS_STAMP": suite_stamp(stamp, "exact"),
    }

    run(["just", "stop"], env=env)

    prompt_suite_json = json.dumps(exact_defaults["prompt_suite"], separators=(",", ":"))
    cmd = [
        str(REPO_ROOT / "scripts" / "run-validation-case.sh"),
        backend,
        case_id,
        "python3",
        str(REPO_ROOT / "scripts" / "ci-exact-smoke.py"),
        "--backend",
        backend,
        "--mesh-llm",
        "target/release/mesh-llm",
        "--prompt",
        exact_defaults["prompt"],
        "--expect-contains",
        exact_defaults["expect_contains"],
        "--forbid-contains",
        exact_defaults["forbid_contains"],
        "--expect-exact",
        exact_defaults["expect_exact"],
        "--prompt-suite-json",
        prompt_suite_json,
    ]
    if backend == "gguf":
        gguf_path = resolved_models[(backend, backend_cfg["model_ref"])]
        cmd.extend(
            [
                "--bin-dir",
                "llama.cpp/build/bin",
                "--model",
                gguf_path,
            ]
        )
    else:
        mlx_path = resolved_models[(backend, backend_cfg["model_ref"])]
        cmd.extend(
            [
                "--model",
                mlx_path,
                "--expected-template-source",
                backend_cfg["template_source"],
            ]
        )

    rc = run(cmd, env=env).returncode
    append_tsv(
        summary_path(root, stamp, "exact"),
        ["model_id", "label", "expectation_class", "backend", "case_id", "exit"],
        [model["id"], model["label"], model["expectation_class"], backend, case_id, str(rc)],
    )
    return rc


def behavior_report_path(root: Path, stamp: str, case_id: str) -> Path:
    return case_dir(root, stamp, "behavior", case_id) / "report.json"


def exact_chat_dir(root: Path, stamp: str, case_id: str) -> Path:
    return case_dir(root, stamp, "exact", case_id) / "chat"


def normalize_exact_output(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip().lower()
    normalized = normalized.replace("**", "").replace("__", "").replace("`", "")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip(" \t\n\r.,;:!?")


def load_exact_prompt_artifacts(root: Path, stamp: str, case_id: str) -> dict[str, dict[str, Any]]:
    chat_dir = exact_chat_dir(root, stamp, case_id)
    artifacts: dict[str, dict[str, Any]] = {}
    if not chat_dir.exists():
        return artifacts
    for path in sorted(chat_dir.glob("*.json")):
        if path.stem.endswith(".thinking"):
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        label = str(payload.get("label", path.stem))
        artifacts[label] = payload
    return artifacts


def exact_artifact_content(payload: dict[str, Any]) -> str:
    content = payload.get("content")
    if content is not None:
        return str(content)
    response_text = payload.get("response_text")
    if response_text is not None:
        return str(response_text)
    return ""


def exact_prompt_snapshot(root: Path, stamp: str, case_id: str) -> dict[str, str]:
    return {
        label: normalize_exact_output(exact_artifact_content(payload))
        for label, payload in load_exact_prompt_artifacts(root, stamp, case_id).items()
    }


def satisfied_expectation_buckets(payload: dict[str, Any]) -> set[str]:
    expectations = payload.get("expectations", {})
    content = exact_artifact_content(payload)
    normalized_content = normalize_exact_output(content)
    buckets: set[str] = set()

    expect_exact = str(expectations.get("expect_exact", ""))
    if expect_exact and normalized_content == normalize_exact_output(expect_exact):
        buckets.add("expect_exact")

    expect_contains = str(expectations.get("expect_contains", ""))
    if expect_contains and expect_contains in content:
        buckets.add("expect_contains")

    expect_contains_ci = str(expectations.get("expect_contains_ci", ""))
    if expect_contains_ci and normalize_exact_output(expect_contains_ci) in normalized_content:
        buckets.add("expect_contains_ci")

    expect_contains_all_ci = [str(item) for item in expectations.get("expect_contains_all_ci", [])]
    if expect_contains_all_ci and all(normalize_exact_output(item) in normalized_content for item in expect_contains_all_ci):
        buckets.add("expect_contains_all_ci")

    expect_any_ci = [str(item) for item in expectations.get("expect_any_ci", [])]
    if expect_any_ci and any(normalize_exact_output(item) in normalized_content for item in expect_any_ci):
        buckets.add("expect_any_ci")

    return buckets


def compare_exact_prompt_payloads(
    gguf_payload: dict[str, Any],
    mlx_payload: dict[str, Any],
) -> tuple[str, str]:
    gguf_content = exact_artifact_content(gguf_payload)
    mlx_content = exact_artifact_content(mlx_payload)
    gguf_normalized = normalize_exact_output(gguf_content)
    mlx_normalized = normalize_exact_output(mlx_content)
    if gguf_normalized == mlx_normalized:
        return ("same-output", gguf_normalized)

    gguf_buckets = satisfied_expectation_buckets(gguf_payload)
    mlx_buckets = satisfied_expectation_buckets(mlx_payload)
    shared_buckets = sorted(gguf_buckets & mlx_buckets)
    if shared_buckets:
        return ("same-bucket", ",".join(shared_buckets))

    return ("backend-differs", f"gguf={gguf_content!r} mlx={mlx_content!r}")


def flagged_prompt_summary(report_path: Path) -> list[str]:
    if not report_path.exists():
        return []
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    flagged: list[str] = []
    for result in payload.get("results", []):
        if result.get("passed", True):
            continue
        prompt_id = result.get("prompt_id", "")
        category = result.get("category", "")
        flagged.append(f"{category}#{prompt_id}")
    return flagged


def run_behavior_case(
    root: Path,
    stamp: str,
    matrix: dict[str, Any],
    model: dict[str, Any],
    backend: str,
    resolved_models: dict[tuple[str, str], str],
    *,
    dataset: str,
    max_prompts: int,
    max_tokens: int,
    wait_seconds: int,
) -> int:
    behavior_defaults = matrix["defaults"]["behavior"]
    backend_cfg = model[backend]
    case_id = backend_cfg["behavior_case_id"]
    out_dir = case_dir(root, stamp, "behavior", case_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = behavior_report_path(root, stamp, case_id)
    mesh_log_path = out_dir / "mesh.log"
    env = {
        **os.environ,
        "VALIDATION_RESULTS_ROOT": str(root),
        "VALIDATION_RESULTS_STAMP": suite_stamp(stamp, "behavior"),
    }

    run(["just", "stop"], env=env)

    if backend == "gguf":
        model_arg = resolved_models[(backend, backend_cfg["model_ref"])]
        cmd = [
            str(REPO_ROOT / "scripts" / "run-validation-case.sh"),
            backend,
            case_id,
            "python3",
            str(REPO_ROOT / "scripts" / "ci-mt-bench-behavior.py"),
            "--backend",
            backend,
            "--mesh-llm",
            "target/release/mesh-llm",
            "--bin-dir",
            "llama.cpp/build/bin",
            "--model",
            model_arg,
            "--label",
            model["label"],
            "--dataset",
            dataset or behavior_defaults["dataset"],
            "--max-prompts",
            str(max_prompts),
            "--max-tokens",
            str(max_tokens or behavior_defaults["max_tokens"]),
            "--wait-seconds",
            str(wait_seconds or behavior_defaults["wait_seconds"]),
            "--mesh-log-output",
            str(mesh_log_path),
            "--output-json",
            str(report_path),
        ]
    else:
        model_arg = resolved_models[(backend, backend_cfg["model_ref"])]
        cmd = [
            str(REPO_ROOT / "scripts" / "run-validation-case.sh"),
            backend,
            case_id,
            "python3",
            str(REPO_ROOT / "scripts" / "ci-mt-bench-behavior.py"),
            "--backend",
            backend,
            "--mesh-llm",
            "target/release/mesh-llm",
            "--model",
            model_arg,
            "--label",
            model["label"],
            "--dataset",
            dataset or behavior_defaults["dataset"],
            "--max-prompts",
            str(max_prompts),
            "--max-tokens",
            str(max_tokens or behavior_defaults["max_tokens"]),
            "--wait-seconds",
            str(wait_seconds or behavior_defaults["wait_seconds"]),
            "--mesh-log-output",
            str(mesh_log_path),
            "--output-json",
            str(report_path),
        ]

    rc = run(cmd, env=env).returncode
    failed_prompt_count = ""
    prompt_count = ""
    if report_path.exists():
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        failed_prompt_count = str(payload.get("failed_prompt_count", ""))
        prompt_count = str(payload.get("prompt_count", ""))
    append_tsv(
        summary_path(root, stamp, "behavior"),
        [
            "model_id",
            "label",
            "expectation_class",
            "backend",
            "case_id",
            "exit",
            "failed_prompts",
            "prompt_count",
        ],
        [
            model["id"],
            model["label"],
            model["expectation_class"],
            backend,
            case_id,
            str(rc),
            failed_prompt_count,
            prompt_count,
        ],
    )
    return rc


def aggregate(root: Path, stamp: str, models: list[dict[str, Any]]) -> None:
    exact_rows: dict[tuple[str, str], str] = {}
    behavior_rows: dict[tuple[str, str], tuple[str, str]] = {}

    exact_path = summary_path(root, stamp, "exact")
    if exact_path.exists():
        for line in exact_path.read_text(encoding="utf-8").splitlines()[1:]:
            model_id, _label, _expectation, backend, _case_id, exit_code = line.split("\t")
            exact_rows[(model_id, backend)] = exit_code

    behavior_path = summary_path(root, stamp, "behavior")
    if behavior_path.exists():
        for line in behavior_path.read_text(encoding="utf-8").splitlines()[1:]:
            model_id, _label, _expectation, backend, _case_id, exit_code, failed_prompts, prompt_count = line.split("\t")
            behavior_rows[(model_id, backend)] = (exit_code, f"{failed_prompts}/{prompt_count}" if failed_prompts and prompt_count else "")

    aggregate_path = root / stamp / "validation-summary.tsv"
    header = [
        "model_id",
        "label",
        "expectation_class",
        "gguf_exact_exit",
        "mlx_exact_exit",
        "gguf_behavior",
        "mlx_behavior",
    ]
    lines = ["\t".join(header)]
    for model in models:
        gguf_behavior = behavior_rows.get((model["id"], "gguf"), ("", ""))
        mlx_behavior = behavior_rows.get((model["id"], "mlx"), ("", ""))
        lines.append(
            "\t".join(
                [
                    model["id"],
                    model["label"],
                    model["expectation_class"],
                    exact_rows.get((model["id"], "gguf"), ""),
                    exact_rows.get((model["id"], "mlx"), ""),
                    ":".join(filter(None, gguf_behavior)),
                    ":".join(filter(None, mlx_behavior)),
                ]
            )
        )
    aggregate_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def planned_cases(models: list[dict[str, Any]], backend_filter: str, suite: str) -> list[dict[str, str]]:
    backend_order = ["gguf", "mlx"] if backend_filter == "both" else [backend_filter]
    cases: list[dict[str, str]] = []
    if suite in ("exact", "all"):
        for backend in backend_order:
            for model in models:
                if backend not in requested_backends(model, backend_filter):
                    continue
                case_id = model[backend].get("exact_case_id")
                if not case_id:
                    continue
                cases.append(
                    {
                        "suite": "exact",
                        "backend": backend,
                        "model_id": model["id"],
                        "label": model["label"],
                        "case_id": case_id,
                    }
                )
    if suite in ("behavior", "all"):
        for backend in backend_order:
            for model in models:
                if backend not in requested_backends(model, backend_filter):
                    continue
                case_id = model[backend].get("behavior_case_id")
                if not case_id:
                    continue
                cases.append(
                    {
                        "suite": "behavior",
                        "backend": backend,
                        "model_id": model["id"],
                        "label": model["label"],
                        "case_id": case_id,
                    }
                )
    return cases


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_overall_progress(
    root: Path,
    stamp: str,
    *,
    total_cases: int,
    completed_cases: int,
    current: dict[str, Any] | None,
    overall_rc: int,
) -> None:
    payload = {
        "total_cases": total_cases,
        "completed_cases": completed_cases,
        "completion_ratio": (completed_cases / total_cases) if total_cases else 0.0,
        "current_case": current,
        "overall_exit_code": overall_rc,
    }
    write_json(root / stamp / "overall-progress.json", payload)


def preflight_models(
    root: Path,
    stamp: str,
    models: list[dict[str, Any]],
    backend_filter: str,
    suite: str,
) -> dict[tuple[str, str], str]:
    resolved: dict[tuple[str, str], str] = {}
    seen: set[tuple[str, str]] = set()
    required_refs: list[tuple[str, str]] = []

    model_by_id = {model["id"]: model for model in models}
    for case in planned_cases(models, backend_filter, suite):
        model = model_by_id[case["model_id"]]
        backend = case["backend"]
        model_ref = model[backend]["model_ref"]
        key = (backend, model_ref)
        if key in seen:
            continue
        seen.add(key)
        required_refs.append(key)

    out_path = root / stamp / "preflight.json"
    state: dict[str, Any] = {
        "status": "running",
        "total_models": len(required_refs),
        "completed_models": 0,
        "current_backend": None,
        "current_model_ref": None,
        "failed_backend": None,
        "failed_model_ref": None,
        "failure": None,
        "items": [],
    }
    write_json(out_path, state)

    for backend, model_ref in required_refs:
        state["current_backend"] = backend
        state["current_model_ref"] = model_ref
        write_json(out_path, state)
        try:
            local_path = download_model_ref(model_ref, backend)
        except Exception as exc:
            state["status"] = "failed"
            state["failed_backend"] = backend
            state["failed_model_ref"] = model_ref
            state["failure"] = str(exc)
            write_json(out_path, state)
            raise
        resolved[(backend, model_ref)] = local_path
        state["items"].append(
            {
                "backend": backend,
                "model_ref": model_ref,
                "local_path": local_path,
            }
        )
        state["completed_models"] = len(state["items"])
        write_json(out_path, state)

    state["status"] = "completed"
    state["current_backend"] = None
    state["current_model_ref"] = None
    write_json(out_path, state)
    return resolved


def compare_exact_against_baseline(
    baseline_cfg: dict[str, Any],
    root: Path,
    stamp: str,
    models: list[dict[str, Any]],
    backend_filter: str,
) -> None:
    exact_path = summary_path(root, stamp, "exact")
    if not exact_path.exists():
        return
    actual_rows: dict[tuple[str, str], tuple[str, str]] = {}
    for line in exact_path.read_text(encoding="utf-8").splitlines()[1:]:
        model_id, _label, _expectation, backend, case_id, exit_code = line.split("\t")
        actual_rows[(model_id, backend)] = (case_id, exit_code)

    compare_path = root / stamp / "exact-baseline-comparison.tsv"
    header = [
        "model_id",
        "backend",
        "case_id",
        "expected_exit",
        "actual_exit",
        "output_status",
        "status",
    ]
    lines = ["\t".join(header)]
    for model in models:
        for backend in requested_backends(model, backend_filter):
            expected = baseline_cfg.get("exact", {}).get(backend, {}).get(model["id"])
            actual = actual_rows.get((model["id"], backend))
            if expected is None:
                status = "no-baseline"
                lines.append(
                    "\t".join(
                        [
                            model["id"],
                            backend,
                            actual[0] if actual else "",
                            "",
                            actual[1] if actual else "",
                            "",
                            status,
                        ]
                    )
                )
                continue
            expected_exit = str(expected.get("exit", ""))
            actual_exit = actual[1] if actual else ""
            output_status = ""
            if actual is not None:
                expected_outputs = expected.get("prompt_outputs")
                if expected_outputs:
                    actual_outputs = exact_prompt_snapshot(root, stamp, actual[0])
                    output_status = "match" if actual_outputs == expected_outputs else "mismatch"
                else:
                    output_status = "no-output-baseline"
            status = (
                "match"
                if actual_exit == expected_exit and output_status in ("", "match", "no-output-baseline")
                else "mismatch"
            )
            lines.append(
                "\t".join(
                    [
                        model["id"],
                        backend,
                        actual[0] if actual else expected.get("case_id", ""),
                        expected_exit,
                        actual_exit,
                        output_status,
                        status,
                    ]
                )
            )
    compare_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compare_behavior_against_baseline(
    baseline_cfg: dict[str, Any],
    root: Path,
    stamp: str,
    models: list[dict[str, Any]],
    backend_filter: str,
) -> None:
    behavior_path = summary_path(root, stamp, "behavior")
    if not behavior_path.exists():
        return
    actual_rows: dict[tuple[str, str], dict[str, str]] = {}
    for line in behavior_path.read_text(encoding="utf-8").splitlines()[1:]:
        model_id, _label, _expectation, backend, case_id, exit_code, failed_prompts, prompt_count = line.split("\t")
        report_path = behavior_report_path(root, stamp, case_id)
        actual_rows[(model_id, backend)] = {
            "case_id": case_id,
            "exit": exit_code,
            "failed_prompt_count": failed_prompts,
            "prompt_count": prompt_count,
            "flagged": ",".join(flagged_prompt_summary(report_path)),
        }

    compare_path = root / stamp / "behavior-baseline-comparison.tsv"
    header = [
        "model_id",
        "backend",
        "case_id",
        "expected_exit",
        "actual_exit",
        "expected_failed_prompts",
        "actual_failed_prompts",
        "expected_flagged",
        "actual_flagged",
        "status",
    ]
    lines = ["\t".join(header)]
    for model in models:
        for backend in requested_backends(model, backend_filter):
            expected = baseline_cfg.get("behavior", {}).get(backend, {}).get(model["id"])
            actual = actual_rows.get((model["id"], backend))
            if expected is None:
                status = "no-baseline"
                lines.append(
                    "\t".join(
                        [
                            model["id"],
                            backend,
                            actual["case_id"] if actual else "",
                            "",
                            actual["exit"] if actual else "",
                            "",
                            actual["failed_prompt_count"] if actual else "",
                            "",
                            actual["flagged"] if actual else "",
                            status,
                        ]
                    )
                )
                continue
            expected_exit = str(expected.get("exit", ""))
            expected_failed = str(expected.get("failed_prompt_count", ""))
            expected_flagged = ",".join(expected.get("flagged_prompt_ids", []))
            actual_exit = actual["exit"] if actual else ""
            actual_failed = actual["failed_prompt_count"] if actual else ""
            actual_flagged = actual["flagged"] if actual else ""
            status = (
                "match"
                if actual_exit == expected_exit
                and actual_failed == expected_failed
                and actual_flagged == expected_flagged
                else "mismatch"
            )
            lines.append(
                "\t".join(
                    [
                        model["id"],
                        backend,
                        actual["case_id"] if actual else expected.get("case_id", ""),
                        expected_exit,
                        actual_exit,
                        expected_failed,
                        actual_failed,
                        expected_flagged,
                        actual_flagged,
                        status,
                    ]
                )
            )
    compare_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compare_parity_to_canonical(
    baseline_cfg: dict[str, Any],
    root: Path,
    stamp: str,
    models: list[dict[str, Any]],
) -> None:
    exact_path = summary_path(root, stamp, "exact")
    if not exact_path.exists():
        return
    canonical_backend = baseline_cfg.get("canonical_backend", "gguf")
    actual_rows: dict[tuple[str, str], tuple[str, str]] = {}
    for line in exact_path.read_text(encoding="utf-8").splitlines()[1:]:
        model_id, _label, _expectation, backend, case_id, exit_code = line.split("\t")
        actual_rows[(model_id, backend)] = (case_id, exit_code)

    compare_path = root / stamp / "parity-vs-canonical-baseline.tsv"
    header = [
        "model_id",
        "canonical_backend",
        "canonical_expected_exit",
        "actual_mlx_exit",
        "status",
    ]
    lines = ["\t".join(header)]
    for model in models:
        canonical = baseline_cfg.get("exact", {}).get(canonical_backend, {}).get(model["id"])
        mlx_actual = actual_rows.get((model["id"], "mlx"))
        if canonical is None or mlx_actual is None:
            continue
        canonical_exit = str(canonical.get("exit", ""))
        mlx_exit = mlx_actual[1]
        status = "within-threshold" if canonical_exit == mlx_exit else "mlx-differs"
        lines.append(
            "\t".join(
                [
                    model["id"],
                    canonical_backend,
                    canonical_exit,
                    mlx_exit,
                    status,
                ]
            )
        )
    compare_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compare_cross_backend_exact_parity(
    root: Path,
    stamp: str,
    models: list[dict[str, Any]],
) -> None:
    compare_path = root / stamp / "exact-cross-backend-parity.tsv"
    header = [
        "model_id",
        "label",
        "gguf_case_id",
        "mlx_case_id",
        "compared_prompts",
        "missing_labels",
        "status",
        "details",
    ]
    lines = ["\t".join(header)]

    for model in models:
        if "gguf" not in model or "mlx" not in model:
            continue
        gguf_case_id = model["gguf"].get("exact_case_id", "")
        mlx_case_id = model["mlx"].get("exact_case_id", "")
        if not gguf_case_id or not mlx_case_id:
            continue

        gguf_artifacts = load_exact_prompt_artifacts(root, stamp, gguf_case_id)
        mlx_artifacts = load_exact_prompt_artifacts(root, stamp, mlx_case_id)
        gguf_labels = set(gguf_artifacts)
        mlx_labels = set(mlx_artifacts)
        compared_labels = sorted(gguf_labels & mlx_labels)
        missing_labels = sorted(gguf_labels ^ mlx_labels)

        prompt_results: list[str] = []
        prompt_statuses: list[str] = []
        for prompt_label in compared_labels:
            status, detail = compare_exact_prompt_payloads(gguf_artifacts[prompt_label], mlx_artifacts[prompt_label])
            prompt_statuses.append(status)
            prompt_results.append(f"{prompt_label}={status}({detail})")

        if not compared_labels:
            overall_status = "no-shared-prompts"
        elif missing_labels:
            overall_status = "backend-differs"
        elif all(status == "same-output" for status in prompt_statuses):
            overall_status = "same-output"
        elif all(status in ("same-output", "same-bucket") for status in prompt_statuses):
            overall_status = "same-bucket"
        else:
            overall_status = "backend-differs"

        lines.append(
            "\t".join(
                [
                    model["id"],
                    model["label"],
                    gguf_case_id,
                    mlx_case_id,
                    ",".join(compared_labels),
                    ",".join(missing_labels),
                    overall_status,
                    " | ".join(prompt_results),
                ]
            )
        )

    compare_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def baseline_divergence_report(
    baseline_cfg: dict[str, Any],
    root: Path,
    stamp: str,
    models: list[dict[str, Any]],
) -> None:
    out_path = root / stamp / "baseline-divergence.tsv"
    header = [
        "suite",
        "model_id",
        "gguf_baseline",
        "mlx_baseline",
        "status",
    ]
    lines = ["\t".join(header)]

    for model in models:
        gguf_exact = baseline_cfg.get("exact", {}).get("gguf", {}).get(model["id"])
        mlx_exact = baseline_cfg.get("exact", {}).get("mlx", {}).get(model["id"])
        if gguf_exact is not None or mlx_exact is not None:
            gguf_value = "" if gguf_exact is None else str(gguf_exact.get("exit", ""))
            mlx_value = "" if mlx_exact is None else str(mlx_exact.get("exit", ""))
            status = "same" if gguf_value == mlx_value else "diverged"
            lines.append("\t".join(["exact", model["id"], gguf_value, mlx_value, status]))

        gguf_behavior = baseline_cfg.get("behavior", {}).get("gguf", {}).get(model["id"])
        mlx_behavior = baseline_cfg.get("behavior", {}).get("mlx", {}).get(model["id"])
        if gguf_behavior is not None or mlx_behavior is not None:
            gguf_failed = "" if gguf_behavior is None else str(gguf_behavior.get("failed_prompt_count", ""))
            mlx_failed = "" if mlx_behavior is None else str(mlx_behavior.get("failed_prompt_count", ""))
            gguf_flagged = "" if gguf_behavior is None else ",".join(gguf_behavior.get("flagged_prompt_ids", []))
            mlx_flagged = "" if mlx_behavior is None else ",".join(mlx_behavior.get("flagged_prompt_ids", []))
            gguf_value = f"{gguf_failed}:{gguf_flagged}".rstrip(":")
            mlx_value = f"{mlx_failed}:{mlx_flagged}".rstrip(":")
            status = "same" if gguf_value == mlx_value else "diverged"
            lines.append("\t".join(["behavior", model["id"], gguf_value, mlx_value, status]))

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def promote_baselines(
    baseline_cfg: dict[str, Any],
    baseline_path: Path,
    root: Path,
    stamp: str,
    models: list[dict[str, Any]],
    backend_filter: str,
    suite: str,
) -> None:
    for section in ("exact", "behavior"):
        baseline_cfg.setdefault(section, {})
        baseline_cfg[section].setdefault("gguf", {})
        baseline_cfg[section].setdefault("mlx", {})

    requested = {"gguf", "mlx"} if backend_filter == "both" else {backend_filter}

    model_by_id = {model["id"]: model for model in models}

    if suite in ("exact", "all"):
        exact_path = summary_path(root, stamp, "exact")
        if exact_path.exists():
            if "gguf" in requested:
                strict_failures: list[str] = []
                for line in exact_path.read_text(encoding="utf-8").splitlines()[1:]:
                    model_id, _label, expectation_class, backend, case_id, exit_code = line.split("\t")
                    if backend != "gguf" or expectation_class != "strict":
                        continue
                    if exit_code != "0":
                        strict_failures.append(f"{case_id}={exit_code}")
                if strict_failures:
                    raise SystemExit(
                        "❌ refusing to promote canonical GGUF exact baseline; strict rows failed: "
                        + ", ".join(strict_failures)
                    )
            for line in exact_path.read_text(encoding="utf-8").splitlines()[1:]:
                model_id, _label, _expectation, backend, case_id, exit_code = line.split("\t")
                if backend not in requested:
                    continue
                baseline_cfg["exact"][backend][model_id] = {
                    "exit": int(exit_code),
                    "case_id": case_id,
                    "prompt_outputs": exact_prompt_snapshot(root, stamp, case_id),
                }

    if suite in ("behavior", "all"):
        behavior_path = summary_path(root, stamp, "behavior")
        if behavior_path.exists():
            if "gguf" in requested:
                strict_failures: list[str] = []
                for line in behavior_path.read_text(encoding="utf-8").splitlines()[1:]:
                    model_id, _label, expectation_class, backend, case_id, exit_code, failed_prompts, _prompt_count = line.split("\t")
                    if backend != "gguf" or expectation_class != "strict":
                        continue
                    if exit_code != "0" or (failed_prompts and failed_prompts != "0"):
                        strict_failures.append(f"{case_id}=exit:{exit_code},failed:{failed_prompts or '0'}")
                if strict_failures:
                    raise SystemExit(
                        "❌ refusing to promote canonical GGUF behavior baseline; strict rows were flagged: "
                        + ", ".join(strict_failures)
                    )
            for line in behavior_path.read_text(encoding="utf-8").splitlines()[1:]:
                model_id, _label, _expectation, backend, case_id, exit_code, failed_prompts, prompt_count = line.split("\t")
                if backend not in requested:
                    continue
                report_path = behavior_report_path(root, stamp, case_id)
                baseline_cfg["behavior"][backend][model_id] = {
                    "exit": int(exit_code),
                    "case_id": case_id,
                    "failed_prompt_count": int(failed_prompts or "0"),
                    "prompt_count": int(prompt_count or "0"),
                    "flagged_prompt_ids": flagged_prompt_summary(report_path),
                }

    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_path.write_text(json.dumps(baseline_cfg, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=["exact", "behavior", "all"], default="all")
    parser.add_argument("--backend", choices=["gguf", "mlx", "both"], default="both")
    parser.add_argument("--stamp", default="")
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--matrix", default=str(DEFAULT_MATRIX))
    parser.add_argument("--baselines", default=str(DEFAULT_BASELINES))
    parser.add_argument("--cases", default="", help="Comma-separated model ids, labels, or case ids")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--dataset", default="")
    parser.add_argument("--max-prompts", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=0)
    parser.add_argument("--wait-seconds", type=int, default=DEFAULT_WAIT_SECONDS)
    parser.add_argument("--promote-baseline", action="store_true")
    args = parser.parse_args()

    if args.backend in ("mlx", "both") and os.uname().sysname != "Darwin":
        raise SystemExit("❌ MLX validation requires macOS")

    matrix = load_matrix(Path(args.matrix))
    baseline_path = Path(args.baselines) if args.baselines else None
    baselines = load_baselines(baseline_path)
    stamp = args.stamp or subprocess.check_output(["date", "+%Y%m%d-%H%M%S"], text=True).strip()
    root = Path(args.root)
    selectors = {item.strip() for item in args.cases.split(",") if item.strip()}
    models = selected_models(matrix, selectors, args.backend)
    if not models:
        print("No matrix entries matched the requested selectors.", file=sys.stderr)
        return 2

    ensure_build(args.skip_build)

    overall_rc = 0
    backend_order = ["gguf", "mlx"] if args.backend == "both" else [args.backend]
    cases = planned_cases(models, args.backend, args.suite)
    total_cases = len(cases)
    completed_cases = 0
    current_case_path = root / stamp / "current-case.json"
    write_overall_progress(
        root,
        stamp,
        total_cases=total_cases,
        completed_cases=completed_cases,
        current=None,
        overall_rc=overall_rc,
    )
    write_json(
        current_case_path,
        {
            "status": "preflight",
            "completed_cases": completed_cases,
            "total_cases": total_cases,
        },
    )
    resolved_models = preflight_models(root, stamp, models, args.backend, args.suite)
    write_json(
        current_case_path,
        {
            "status": "idle",
            "completed_cases": completed_cases,
            "total_cases": total_cases,
        },
    )

    if args.suite in ("exact", "all"):
        for backend in backend_order:
            for model in models:
                if backend not in requested_backends(model, args.backend):
                    continue
                case_id = model[backend].get("exact_case_id")
                if not case_id:
                    continue
                current = {
                    "suite": "exact",
                    "backend": backend,
                    "model_id": model["id"],
                    "label": model["label"],
                    "case_id": case_id,
                    "status": "running",
                }
                write_json(current_case_path, current)
                write_overall_progress(
                    root,
                    stamp,
                    total_cases=total_cases,
                    completed_cases=completed_cases,
                    current=current,
                    overall_rc=overall_rc,
                )
                print(f"\n=== Running {case_id} ({backend}) ===")
                rc = run_exact_case(root, stamp, matrix, model, backend, resolved_models)
                overall_rc = overall_rc or rc
                completed_cases += 1
                aggregate(root, stamp, models)
                compare_exact_against_baseline(baselines, root, stamp, models, args.backend)
                compare_behavior_against_baseline(baselines, root, stamp, models, args.backend)
                compare_parity_to_canonical(baselines, root, stamp, models)
                compare_cross_backend_exact_parity(root, stamp, models)
                baseline_divergence_report(baselines, root, stamp, models)
                write_overall_progress(
                    root,
                    stamp,
                    total_cases=total_cases,
                    completed_cases=completed_cases,
                    current={**current, "status": "completed", "exit_code": rc},
                    overall_rc=overall_rc,
                )

    if args.suite in ("behavior", "all"):
        for backend in backend_order:
            for model in models:
                if backend not in requested_backends(model, args.backend):
                    continue
                case_id = model[backend].get("behavior_case_id")
                if not case_id:
                    continue
                current = {
                    "suite": "behavior",
                    "backend": backend,
                    "model_id": model["id"],
                    "label": model["label"],
                    "case_id": case_id,
                    "status": "running",
                }
                write_json(current_case_path, current)
                write_overall_progress(
                    root,
                    stamp,
                    total_cases=total_cases,
                    completed_cases=completed_cases,
                    current=current,
                    overall_rc=overall_rc,
                )
                print(f"\n=== Running {case_id} ({backend}) ===")
                rc = run_behavior_case(
                    root,
                    stamp,
                    matrix,
                    model,
                    backend,
                    resolved_models,
                    dataset=args.dataset,
                    max_prompts=args.max_prompts,
                    max_tokens=args.max_tokens,
                    wait_seconds=args.wait_seconds,
                )
                overall_rc = overall_rc or rc
                completed_cases += 1
                aggregate(root, stamp, models)
                compare_exact_against_baseline(baselines, root, stamp, models, args.backend)
                compare_behavior_against_baseline(baselines, root, stamp, models, args.backend)
                compare_parity_to_canonical(baselines, root, stamp, models)
                compare_cross_backend_exact_parity(root, stamp, models)
                baseline_divergence_report(baselines, root, stamp, models)
                write_overall_progress(
                    root,
                    stamp,
                    total_cases=total_cases,
                    completed_cases=completed_cases,
                    current={**current, "status": "completed", "exit_code": rc},
                    overall_rc=overall_rc,
                )

    aggregate(root, stamp, models)
    compare_exact_against_baseline(baselines, root, stamp, models, args.backend)
    compare_behavior_against_baseline(baselines, root, stamp, models, args.backend)
    compare_parity_to_canonical(baselines, root, stamp, models)
    compare_cross_backend_exact_parity(root, stamp, models)
    baseline_divergence_report(baselines, root, stamp, models)

    if args.promote_baseline:
        if baseline_path is None:
            raise SystemExit("❌ --promote-baseline requires a writable --baselines path")
        promote_baselines(baselines, baseline_path, root, stamp, models, args.backend, args.suite)
        baseline_divergence_report(baselines, root, stamp, models)

    if args.suite in ("exact", "all"):
        exact_path = summary_path(root, stamp, "exact")
        if exact_path.exists():
            print("\n=== Exact summary ===")
            print(exact_path.read_text(encoding="utf-8"), end="")
    if args.suite in ("behavior", "all"):
        behavior_path = summary_path(root, stamp, "behavior")
        if behavior_path.exists():
            print("\n=== Behavior summary ===")
            print(behavior_path.read_text(encoding="utf-8"), end="")

    aggregate_path = root / stamp / "validation-summary.tsv"
    if aggregate_path.exists():
        print("\n=== Combined summary ===")
        print(aggregate_path.read_text(encoding="utf-8"), end="")
    exact_compare_path = root / stamp / "exact-baseline-comparison.tsv"
    if exact_compare_path.exists():
        print("\n=== Exact baseline comparison ===")
        print(exact_compare_path.read_text(encoding="utf-8"), end="")
    behavior_compare_path = root / stamp / "behavior-baseline-comparison.tsv"
    if behavior_compare_path.exists():
        print("\n=== Behavior baseline comparison ===")
        print(behavior_compare_path.read_text(encoding="utf-8"), end="")
    parity_compare_path = root / stamp / "parity-vs-canonical-baseline.tsv"
    if parity_compare_path.exists():
        print("\n=== Parity vs canonical baseline ===")
        print(parity_compare_path.read_text(encoding="utf-8"), end="")
    exact_parity_path = root / stamp / "exact-cross-backend-parity.tsv"
    if exact_parity_path.exists():
        print("\n=== Exact cross-backend parity ===")
        print(exact_parity_path.read_text(encoding="utf-8"), end="")
    divergence_path = root / stamp / "baseline-divergence.tsv"
    if divergence_path.exists():
        print("\n=== Baseline divergence ===")
        print(divergence_path.read_text(encoding="utf-8"), end="")
    write_json(
        current_case_path,
        {
            "status": "idle",
            "completed_cases": completed_cases,
            "total_cases": total_cases,
            "overall_exit_code": overall_rc,
        },
    )
    write_overall_progress(
        root,
        stamp,
        total_cases=total_cases,
        completed_cases=completed_cases,
        current=None,
        overall_rc=overall_rc,
    )
    print(f"\nRaw artifacts: {root / stamp}")
    return overall_rc


if __name__ == "__main__":
    raise SystemExit(main())
