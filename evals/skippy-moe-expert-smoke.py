#!/usr/bin/env python3
"""Run MoE expert-route smoke checks for Skippy family certification.

This smoke combines two pieces of evidence:

1. The GGUF carries MoE expert tensors inside the stage layer range.
2. The Skippy state-handoff cache correctness path passes for that range while
   restoring into a different runtime lane/native sequence id.

ResidentKv keeps KV in the runtime cache instead of serializing it. For that
reason the payload digest can be empty while resident_state_bytes and
cache_storage_bytes are non-zero.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any


REPO = Path(__file__).resolve().parents[1]
PRODUCTION_BENCH = REPO / "evals/skippy-cache-production-bench.py"
EXPERT_NAME_MARKERS = (
    "_exps",
    "_shexp",
    ".expert",
    "_expert",
    "ffn_gate_inp",
    "expert_gate",
    "exp_probs",
)


def load_production_bench() -> Any:
    spec = importlib.util.spec_from_file_location("skippy_cache_production_bench", PRODUCTION_BENCH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {PRODUCTION_BENCH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def topology_case(case: Any, topology: str) -> Any:
    layer_end = int(case.layer_end)
    first = max(1, layer_end // 3)
    second = max(first + 1, (layer_end * 2) // 3)
    second = min(second, layer_end)
    if topology == "one-stage":
        return replace(case, state_layer_start=0, state_layer_end=layer_end, state_stage_index=0)
    if topology == "split-stage0":
        return replace(case, state_layer_start=0, state_layer_end=first, state_stage_index=0)
    if topology == "split-middle":
        return replace(case, state_layer_start=first, state_layer_end=second, state_stage_index=1)
    if topology == "split-final":
        return replace(case, state_layer_start=second, state_layer_end=layer_end, state_stage_index=2)
    raise ValueError(f"unknown topology {topology}")


def inspect_model(model_path: Path, inspect_bin: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [str(inspect_bin), "inspect", str(model_path)],
        cwd=REPO,
        text=True,
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return json.loads(completed.stdout)


def expert_tensors(inspect_report: dict[str, Any], layer_start: int, layer_end: int) -> dict[str, Any]:
    tensors = []
    for tensor in inspect_report.get("tensors", []):
        name = str(tensor.get("name", ""))
        layer_index = tensor.get("layer_index")
        if not isinstance(layer_index, int):
            continue
        if layer_index < layer_start or layer_index >= layer_end:
            continue
        if any(marker in name for marker in EXPERT_NAME_MARKERS):
            tensors.append(tensor)

    layers = sorted({tensor["layer_index"] for tensor in tensors})
    return {
        "expert_tensor_count": len(tensors),
        "expert_tensor_bytes": sum(int(tensor.get("byte_size") or 0) for tensor in tensors),
        "expert_layer_count": len(layers),
        "expert_layers": layers,
        "sample_expert_tensors": [tensor["name"] for tensor in tensors[:8]],
    }


def row_from_report(case: Any, topology: str, report: dict[str, Any], expert: dict[str, Any]) -> dict[str, Any]:
    pass_result = report.get("status") == "pass" and expert["expert_tensor_count"] > 0
    return {
        "family": case.family,
        "model_id": case.model_id,
        "payload": case.payload,
        "topology": topology,
        "layer_start": case.state_layer_start,
        "layer_end": case.state_layer_end or case.layer_end,
        "status": "pass" if pass_result else "fail",
        "correctness_status": report.get("status"),
        "native_seq_remapped": report.get("native_seq_remapped"),
        "source_native_seq_id": report.get("source_native_seq_id"),
        "restore_native_seq_id": report.get("restore_native_seq_id"),
        "prompt_tokens": report.get("prompt_token_count"),
        "suffix_prefill_matches": report.get("suffix_prefill_matches"),
        "cache_hit_matches": report.get("cache_hit_matches"),
        "resident_state_bytes": report.get("resident_state_bytes"),
        "cache_storage_bytes": report.get("cache_storage_bytes"),
        "serialized_payload_bytes": report.get("state_bytes"),
        "borrowed_resident_hits": report.get("borrowed_resident_hits"),
        "expert_tensor_count": expert["expert_tensor_count"],
        "expert_tensor_bytes": expert["expert_tensor_bytes"],
        "expert_layer_count": expert["expert_layer_count"],
        "expert_layers": expert["expert_layers"],
        "sample_expert_tensors": expert["sample_expert_tensors"],
    }


def write_markdown(rows: list[dict[str, Any]], output: Path) -> None:
    headers = [
        "Family",
        "Model ref",
        "Topology",
        "Layers",
        "Result",
        "Seq remap",
        "Suffix",
        "Hits",
        "Expert layers",
        "Expert tensors",
        "Expert bytes",
        "Resident bytes",
        "Serialized bytes",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        layers = f"{row['layer_start']}..{row['layer_end']}"
        expert_layers = f"{row['expert_layer_count']} ({row['expert_layers'][0]}..{row['expert_layers'][-1]})" if row["expert_layers"] else "0"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["family"]),
                    str(row["model_id"]),
                    str(row["topology"]),
                    layers,
                    str(row["status"]),
                    str(row["native_seq_remapped"]),
                    str(row["suffix_prefill_matches"]),
                    str(row["cache_hit_matches"]),
                    expert_layers,
                    str(row["expert_tensor_count"]),
                    str(row["expert_tensor_bytes"]),
                    str(row["resident_state_bytes"]),
                    str(row["serialized_payload_bytes"]),
                ]
            )
            + " |"
        )
    output.write_text("\n".join(lines) + "\n")


def main() -> int:
    bench = load_production_bench()
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/skippy-moe-expert-smoke"))
    parser.add_argument("--case", action="append", help="Run only this production-bench case key.")
    parser.add_argument(
        "--topology",
        action="append",
        choices=["one-stage", "split-stage0", "split-middle", "split-final"],
        help="Topology slice to test. Defaults to one-stage, split middle, and split final.",
    )
    parser.add_argument("--prefix-tokens", type=int, default=32)
    parser.add_argument("--suffix-token-count", type=int, default=3)
    parser.add_argument("--cache-hit-repeats", type=int, default=3)
    parser.add_argument("--runtime-lane-count", type=int, default=4)
    parser.add_argument("--n-gpu-layers", type=int)
    parser.add_argument("--correctness-timeout-secs", type=int, default=900)
    parser.add_argument(
        "--llama-stage-build-dir",
        type=Path,
        default=REPO / ".deps/llama-build/build-stage-abi-cpu",
    )
    parser.add_argument("--skippy-correctness-bin", type=Path, default=REPO / "target/debug/skippy-correctness")
    parser.add_argument("--skippy-model-package-bin", type=Path, default=REPO / "target/debug/skippy-model-package")
    args = parser.parse_args()

    wanted = set(args.case or ["olmoe", "qwen2moe", "qwen3moe"])
    selected = [case for case in bench.CASES if case.key in wanted]
    missing = wanted - {case.key for case in selected}
    if missing:
        raise SystemExit(f"unknown case(s): {', '.join(sorted(missing))}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    topologies = args.topology or ["one-stage", "split-middle", "split-final"]
    runner_args = SimpleNamespace(
        cache_hit_repeats=args.cache_hit_repeats,
        skippy_correctness_bin=args.skippy_correctness_bin,
        runtime_lane_count=args.runtime_lane_count,
        borrow_resident_hits=True,
        cache_decoded_result_hits=False,
        llama_stage_build_dir=args.llama_stage_build_dir,
        correctness_timeout_secs=args.correctness_timeout_secs,
        suffix_token_count=args.suffix_token_count,
    )

    rows: list[dict[str, Any]] = []
    raw: list[dict[str, Any]] = []
    inspect_cache: dict[Path, dict[str, Any]] = {}
    for base_case in selected:
        if base_case.model_path is None or not base_case.model_path.exists():
            raise SystemExit(f"missing model for {base_case.key}: {base_case.model_path}")
        inspect_cache[base_case.model_path] = inspect_model(base_case.model_path, args.skippy_model_package_bin)
        for topology in topologies:
            case = topology_case(base_case, topology)
            case = replace(
                case,
                prefix_tokens=args.prefix_tokens,
                ctx_size=max(case.ctx_size, args.prefix_tokens + args.suffix_token_count + 128),
                n_gpu_layers=args.n_gpu_layers if args.n_gpu_layers is not None else case.n_gpu_layers,
            )
            layer_start = int(case.state_layer_start)
            layer_end = int(case.state_layer_end or case.layer_end)
            expert = expert_tensors(inspect_cache[base_case.model_path], layer_start, layer_end)
            case_dir = args.output_dir / case.key / topology
            case_dir.mkdir(parents=True, exist_ok=True)
            print(
                f"==> {case.key} {topology} layers {layer_start}..{layer_end} "
                f"expert_tensors={expert['expert_tensor_count']}",
                flush=True,
            )
            report = bench.run_correctness(case, runner_args, case_dir)
            row = row_from_report(case, topology, report, expert)
            raw.append({"case": case.key, "topology": topology, "report": report, "expert": expert})
            rows.append(row)

    (args.output_dir / "moe-expert-smoke.json").write_text(json.dumps(raw, indent=2) + "\n")
    (args.output_dir / "moe-expert-smoke-table.json").write_text(json.dumps(rows, indent=2) + "\n")
    write_markdown(rows, args.output_dir / "moe-expert-smoke-table.md")
    return 0 if all(row["status"] == "pass" for row in rows) else 1


if __name__ == "__main__":
    os.environ.setdefault("CARGO_INCREMENTAL", "0")
    raise SystemExit(main())
