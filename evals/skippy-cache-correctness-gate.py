#!/usr/bin/env python3
"""Run the Skippy cache correctness exit gate by family/state-layout class."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any


REPO = Path(__file__).resolve().parents[1]
PRODUCTION_BENCH = REPO / "evals/skippy-cache-production-bench.py"


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


def row_from_report(case: Any, topology: str, report: dict[str, Any]) -> dict[str, Any]:
    digest = report.get("payload_digest") if isinstance(report.get("payload_digest"), dict) else {}
    return {
        "family": case.family,
        "model_id": case.model_id,
        "payload": case.payload,
        "topology": topology,
        "status": report.get("status"),
        "matches": report.get("matches"),
        "native_seq_remapped": report.get("native_seq_remapped"),
        "source_native_seq_id": report.get("source_native_seq_id"),
        "restore_native_seq_id": report.get("restore_native_seq_id"),
        "prompt_tokens": report.get("prompt_token_count"),
        "suffix_tokens": report.get("suffix_token_count"),
        "suffix_prefill_matches": report.get("suffix_prefill_matches"),
        "state_bytes": report.get("state_bytes"),
        "cache_storage_bytes": report.get("cache_storage_bytes"),
        "recurrent_bytes": digest.get("recurrent_bytes"),
        "kv_bytes": digest.get("kv_bytes"),
        "cache_hit_repeats": report.get("cache_hit_repeats"),
        "cache_hit_matches": report.get("cache_hit_matches"),
        "promotion_decision": "pass" if report.get("status") == "pass" else "disabled-or-recompute",
    }


def write_markdown(rows: list[dict[str, Any]], output: Path) -> None:
    headers = [
        "Family",
        "Model ref",
        "Payload",
        "Topology",
        "Result",
        "Seq remap",
        "Source seq",
        "Target seq",
        "Tokens",
        "Suffix",
        "Payload bytes",
        "Recurrent bytes",
        "Hits",
        "Promotion",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["family"]),
                    str(row["model_id"]),
                    str(row["payload"]),
                    str(row["topology"]),
                    str(row["status"]),
                    str(row["native_seq_remapped"]),
                    str(row["source_native_seq_id"]),
                    str(row["restore_native_seq_id"]),
                    str(row["prompt_tokens"]),
                    str(row["suffix_prefill_matches"]),
                    str(row["state_bytes"]),
                    str(row["recurrent_bytes"]),
                    str(row["cache_hit_matches"]),
                    str(row["promotion_decision"]),
                ]
            )
            + " |"
        )
    output.write_text("\n".join(lines) + "\n")


def main() -> int:
    bench = load_production_bench()
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/skippy-cache-correctness-gate"))
    parser.add_argument("--case", action="append", help="Run only this production-bench case key.")
    parser.add_argument(
        "--topology",
        action="append",
        choices=["one-stage", "split-stage0", "split-middle", "split-final"],
        help="Topology slice to test. Defaults to one-stage plus split middle/final.",
    )
    parser.add_argument("--prefix-tokens", type=int)
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
    parser.add_argument(
        "--skippy-correctness-bin",
        type=Path,
        default=REPO / "target/debug/skippy-correctness",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected = bench.CASES
    if args.case:
        wanted = set(args.case)
        selected = [case for case in selected if case.key in wanted]
        missing = wanted - {case.key for case in selected}
        if missing:
            raise SystemExit(f"unknown case(s): {', '.join(sorted(missing))}")
    else:
        wanted = {
            "llama",
            "qwen3_dense",
            "gemma3",
            "gemma",
            "olmo",
            "glm4",
            "falcon_h1",
            "mistral3",
            "gpt2",
            "mpt",
            "olmo2",
            "olmoe",
            "phi3",
            "granite",
            "bloom",
            "gptneox",
            "baichuan",
            "exaone",
            "exaone4",
            "command_r",
            "cohere2",
            "falcon",
            "internlm2",
            "stablelm",
            "starcoder2",
            "qwen2moe",
            "qwen3moe",
            "jamba",
            "lfm2",
            "mamba",
            "mamba2",
            "qwen3next",
            "rwkv6",
            "rwkv7",
        }
        selected = [case for case in selected if case.key in wanted]

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
    for base_case in selected:
        if base_case.model_path is None or not base_case.model_path.exists():
            rows.append(
                {
                    "family": base_case.family,
                    "model_id": base_case.model_id,
                    "payload": base_case.payload,
                    "topology": "all",
                    "status": "missing-model",
                    "matches": False,
                    "native_seq_remapped": False,
                    "source_native_seq_id": None,
                    "restore_native_seq_id": None,
                    "prompt_tokens": None,
                    "suffix_tokens": args.suffix_token_count,
                    "suffix_prefill_matches": False,
                    "state_bytes": None,
                    "cache_storage_bytes": None,
                    "recurrent_bytes": None,
                    "kv_bytes": None,
                    "cache_hit_repeats": 0,
                    "cache_hit_matches": False,
                    "promotion_decision": "disabled-or-recompute",
                }
            )
            continue
        for topology in topologies:
            case = topology_case(base_case, topology)
            prefix_tokens = args.prefix_tokens if args.prefix_tokens is not None else min(case.prefix_tokens, 32)
            case = replace(
                case,
                prefix_tokens=prefix_tokens,
                ctx_size=max(case.ctx_size, prefix_tokens + args.suffix_token_count + 128),
                n_gpu_layers=args.n_gpu_layers if args.n_gpu_layers is not None else case.n_gpu_layers,
            )
            case_dir = args.output_dir / case.key / topology
            case_dir.mkdir(parents=True, exist_ok=True)
            print(f"==> {case.key} {topology} {case.payload}", flush=True)
            report = bench.run_correctness(case, runner_args, case_dir)
            raw.append({"case": case.key, "topology": topology, "report": report})
            rows.append(row_from_report(case, topology, report))

    (args.output_dir / "cache-correctness-gate.json").write_text(json.dumps(raw, indent=2) + "\n")
    (args.output_dir / "cache-correctness-table.json").write_text(json.dumps(rows, indent=2) + "\n")
    write_markdown(rows, args.output_dir / "cache-correctness-table.md")
    return 0 if all(row["status"] in {"pass", "missing-model"} for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
