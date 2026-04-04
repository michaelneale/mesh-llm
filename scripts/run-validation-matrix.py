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
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = REPO_ROOT / "scripts" / "validation-matrix.json"
DEFAULT_ROOT = REPO_ROOT / "MLX_VALIDATION_RESULTS"


def run(
    cmd: list[str],
    *,
    cwd: Path = REPO_ROOT,
    env: dict[str, str] | None = None,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        text=True,
        capture_output=capture_output,
        check=False,
    )


def ensure_build(skip_build: bool) -> None:
    if skip_build:
        return
    rc = run(["just", "build"])
    if rc.returncode != 0:
        raise SystemExit(rc.returncode)


def load_matrix(path: Path) -> dict[str, Any]:
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


def download_gguf_path(model_ref: str) -> str:
    proc = run(
        ["./target/release/mesh-llm", "models", "download", model_ref, "--gguf"],
        capture_output=True,
    )
    sys.stderr.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)
    return parse_downloaded_gguf_path(proc.stdout)


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
    if backend == "gguf":
        gguf_path = download_gguf_path(backend_cfg["model_ref"])
        cmd = [
            str(REPO_ROOT / "scripts" / "run-validation-case.sh"),
            backend,
            case_id,
            str(REPO_ROOT / "scripts" / "ci-gguf-smoke-test.sh"),
            "target/release/mesh-llm",
            "llama.cpp/build/bin",
            gguf_path,
            exact_defaults["prompt"],
            exact_defaults["expect_contains"],
            exact_defaults["forbid_contains"],
            exact_defaults["expect_exact"],
            prompt_suite_json,
        ]
    else:
        cmd = [
            str(REPO_ROOT / "scripts" / "run-validation-case.sh"),
            backend,
            case_id,
            str(REPO_ROOT / "scripts" / "ci-mlx-smoke-test.sh"),
            "target/release/mesh-llm",
            backend_cfg["model_ref"],
            backend_cfg["template_source"],
            exact_defaults["prompt"],
            exact_defaults["expect_contains"],
            exact_defaults["forbid_contains"],
            "",
            exact_defaults["expect_exact"],
            prompt_suite_json,
        ]

    rc = run(cmd, env=env).returncode
    append_tsv(
        summary_path(root, stamp, "exact"),
        ["model_id", "label", "expectation_class", "backend", "case_id", "exit"],
        [model["id"], model["label"], model["expectation_class"], backend, case_id, str(rc)],
    )
    return rc


def behavior_report_path(root: Path, stamp: str, case_id: str) -> Path:
    return case_dir(root, stamp, "behavior", case_id) / "report.json"


def run_behavior_case(
    root: Path,
    stamp: str,
    matrix: dict[str, Any],
    model: dict[str, Any],
    backend: str,
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
        model_arg = download_gguf_path(backend_cfg["model_ref"])
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
            backend_cfg["model_ref"],
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=["exact", "behavior", "all"], default="all")
    parser.add_argument("--backend", choices=["gguf", "mlx", "both"], default="both")
    parser.add_argument("--stamp", default="")
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--matrix", default=str(DEFAULT_MATRIX))
    parser.add_argument("--cases", default="", help="Comma-separated model ids, labels, or case ids")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--dataset", default="")
    parser.add_argument("--max-prompts", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=0)
    parser.add_argument("--wait-seconds", type=int, default=0)
    args = parser.parse_args()

    if args.backend in ("mlx", "both") and os.uname().sysname != "Darwin":
        raise SystemExit("❌ MLX validation requires macOS")

    matrix = load_matrix(Path(args.matrix))
    stamp = args.stamp or subprocess.check_output(["date", "+%Y%m%d-%H%M%S"], text=True).strip()
    root = Path(args.root)
    selectors = {item.strip() for item in args.cases.split(",") if item.strip()}
    models = selected_models(matrix, selectors, args.backend)
    if not models:
        print("No matrix entries matched the requested selectors.", file=sys.stderr)
        return 2

    ensure_build(args.skip_build)

    overall_rc = 0
    for model in models:
        for backend in requested_backends(model, args.backend):
            if args.suite in ("exact", "all"):
                print(f"\n=== Running {model[backend]['exact_case_id']} ({backend}) ===")
                rc = run_exact_case(root, stamp, matrix, model, backend)
                overall_rc = overall_rc or rc
            if args.suite in ("behavior", "all"):
                print(f"\n=== Running {model[backend]['behavior_case_id']} ({backend}) ===")
                rc = run_behavior_case(
                    root,
                    stamp,
                    matrix,
                    model,
                    backend,
                    dataset=args.dataset,
                    max_prompts=args.max_prompts,
                    max_tokens=args.max_tokens,
                    wait_seconds=args.wait_seconds,
                )
                overall_rc = overall_rc or rc

    aggregate(root, stamp, models)

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
    print(f"\nRaw artifacts: {root / stamp}")
    return overall_rc


if __name__ == "__main__":
    raise SystemExit(main())
