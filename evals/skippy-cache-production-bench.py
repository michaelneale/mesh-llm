#!/usr/bin/env python3
"""Run production Skippy cache correctness and llama-server baselines.

This runner intentionally benchmarks only production cache payloads:
ResidentKv and KvRecurrent. FullState is a correctness diagnostic and is not a
performance target.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
HOME = Path.home()


@dataclass(frozen=True)
class Case:
    key: str
    family: str
    model_id: str
    model_path: Path | None
    payload: str
    layer_end: int
    activation_width: int
    ctx_size: int = 512
    n_gpu_layers: int = 0
    prefix_tokens: int = 128
    cache_hit_repeats: int = 3
    stage_load_mode: str = "runtime-slice"
    state_layer_start: int = 0
    state_layer_end: int | None = None
    state_stage_index: int | None = None
    resident_kv_bytes_per_token: int | None = None
    skip_llama_server_reason: str | None = None


@dataclass(frozen=True)
class UseCase:
    key: str
    label: str
    prompt: str
    prefix_tokens: int = 128
    source_dataset: str | None = None
    source_config: str | None = None
    source_split: str | None = None
    source_row: int | None = None


def load_use_cases(path: Path) -> list[UseCase]:
    data = json.loads(path.read_text())
    use_cases = []
    for item in data.get("use_cases", []):
        source = item.get("source", {})
        use_cases.append(
            UseCase(
                key=item["key"],
                label=item["label"],
                prompt=item["prompt"],
                prefix_tokens=int(item.get("prefix_tokens", 128)),
                source_dataset=source.get("dataset"),
                source_config=source.get("config"),
                source_split=source.get("split"),
                source_row=source.get("row_idx"),
            )
        )
    return use_cases


CASES = [
    Case(
        "qwen3_dense",
        "Qwen3 dense",
        "Qwen/Qwen3-0.6B:Q8_0",
        HOME
        / ".cache/huggingface/hub/models--Qwen--Qwen3-0.6B-GGUF/snapshots/23749fefcc72300e3a2ad315e1317431b06b590a/Qwen3-0.6B-Q8_0.gguf",
        "resident-kv",
        28,
        1024,
        resident_kv_bytes_per_token=114_688,
    ),
    Case(
        "llama",
        "Llama",
        "hugging-quants/Llama-3.2-1B-Instruct-Q4_K_M-GGUF:Q4_K_M",
        HOME
        / ".cache/huggingface/hub/models--hugging-quants--Llama-3.2-1B-Instruct-Q4_K_M-GGUF/snapshots/7d1f70022fcab2038000074bd0342e03e1d8b755/llama-3.2-1b-instruct-q4_k_m.gguf",
        "resident-kv",
        16,
        2048,
        resident_kv_bytes_per_token=32_768,
    ),
    Case(
        "deepseek2",
        "DeepSeek2",
        "bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF:Q4_K_M",
        HOME
        / ".cache/huggingface/hub/models--bartowski--DeepSeek-Coder-V2-Lite-Instruct-GGUF/snapshots/8f248fa2072348f77a8bc37754e470de1f61866e/DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf",
        "resident-kv",
        27,
        2048,
        prefix_tokens=64,
        resident_kv_bytes_per_token=276_480,
    ),
    Case(
        "deepseek3",
        "DeepSeek3",
        "unsloth/DeepSeek-V3.2-GGUF:UD-Q4_K_XL",
        HOME
        / ".cache/huggingface/hub/models--meshllm--DeepSeek-V3.2-UD-Q4_K_XL-layers/snapshots/c7d74031a7201334b4550da6537d0b8734b81fe2",
        "resident-kv",
        61,
        7168,
        ctx_size=32,
        prefix_tokens=4,
        stage_load_mode="layer-package",
        state_layer_start=3,
        state_layer_end=4,
        state_stage_index=1,
        skip_llama_server_reason="Layer-package evidence only; no full GGUF is loaded for this DeepSeek3 gate.",
        resident_kv_bytes_per_token=2_176,
    ),
    Case(
        "glm47_flash",
        "GLM-4.7 Flash",
        "unsloth/GLM-4.7-Flash-GGUF:Q4_K_M",
        HOME
        / ".cache/huggingface/hub/models--unsloth--GLM-4.7-Flash-GGUF/snapshots/0d32489ecb9db6d2a4fc93bd27ef01519f95474d/GLM-4.7-Flash-Q4_K_M.gguf",
        "resident-kv",
        47,
        2048,
        prefix_tokens=32,
        resident_kv_bytes_per_token=102_272,
    ),
    Case(
        "glm4",
        "GLM4",
        "meshllm/glm-4-9b-0414-parity-q4_k_m-gguf:Q4_K_M",
        HOME
        / ".cache/huggingface/hub/models--meshllm--glm-4-9b-0414-parity-q4_k_m-gguf/snapshots/b15dd8df3957ace630d34943149a180282db4680/glm-4-9b-0414-q4_k_m.gguf",
        "resident-kv",
        40,
        4096,
        prefix_tokens=32,
        resident_kv_bytes_per_token=40_960,
    ),
    Case(
        "gemma4_a4b",
        "Gemma4 A4B",
        "batiai/Gemma-4-26B-A4B-it-GGUF:Q6_K",
        HOME
        / ".cache/huggingface/hub/models--batiai--Gemma-4-26B-A4B-it-GGUF/snapshots/45ad6023c1c79fe5813b34270bc4d44e392a0d17/google-gemma-4-26B-A4B-it-Q6_K.gguf",
        "resident-kv",
        30,
        2816,
        prefix_tokens=16,
        resident_kv_bytes_per_token=225_280,
    ),
    Case(
        "gemma4_e4b",
        "Gemma4 E4B",
        "unsloth/gemma-4-E4B-it-GGUF:Q4_K_M",
        HOME
        / ".cache/huggingface/hub/models--unsloth--gemma-4-E4B-it-GGUF/snapshots/315e03409eb1cdde302488d66e586dea1e82aad1/gemma-4-E4B-it-Q4_K_M.gguf",
        "resident-kv",
        42,
        2560,
        prefix_tokens=16,
        resident_kv_bytes_per_token=57_344,
    ),
    Case(
        "gemma3",
        "Gemma3",
        "ggml-org/gemma-3-1b-it-GGUF:Q4_K_M",
        HOME
        / ".cache/huggingface/hub/models--ggml-org--gemma-3-1b-it-GGUF/snapshots/f9c28bcd85737ffc5aef028638d3341d49869c27/gemma-3-1b-it-Q4_K_M.gguf",
        "resident-kv",
        26,
        1152,
        resident_kv_bytes_per_token=26_624,
    ),
    Case(
        "gemma2",
        "Gemma2",
        "bartowski/gemma-2-2b-it-GGUF:Q4_K_M",
        HOME
        / ".cache/huggingface/hub/models--bartowski--gemma-2-2b-it-GGUF/snapshots/855f67caed130e1befc571b52bd181be2e858883/gemma-2-2b-it-Q4_K_M.gguf",
        "resident-kv",
        26,
        2304,
        resident_kv_bytes_per_token=106_496,
    ),
    Case(
        "falcon_h1",
        "Falcon-H1",
        "tiiuae/Falcon-H1-1.5B-Instruct-GGUF:Q4_K_M",
        HOME
        / ".cache/huggingface/hub/models--tiiuae--Falcon-H1-1.5B-Instruct-GGUF/snapshots/0d3a6cfe25fb4eeab0153fb8623aac5b69d6bd0a/Falcon-H1-1.5B-Instruct-Q4_K_M.gguf",
        "kv-recurrent",
        24,
        2048,
    ),
    Case(
        "olmo",
        "OLMo",
        "meshllm/olmo-7b-instruct-hf-parity-f16-gguf:F16",
        HOME
        / ".cache/huggingface/hub/models--meshllm--olmo-7b-instruct-hf-parity-f16-gguf/snapshots/846c0ae38aff29ea8fce0959fb406cdcef858bac/olmo-7b-instruct-hf-f16.gguf",
        "resident-kv",
        32,
        4096,
        prefix_tokens=64,
        resident_kv_bytes_per_token=524_288,
    ),
    Case(
        "minimax_m27",
        "MiniMax M2.7",
        "unsloth/MiniMax-M2.7-GGUF:UD-Q2_K_XL",
        HOME
        / ".cache/huggingface/hub/models--unsloth--MiniMax-M2.7-GGUF/snapshots/d2a05ccf69491b03db0cc40b335aec14bdaf7198/UD-Q2_K_XL/MiniMax-M2.7-UD-Q2_K_XL-00001-of-00003.gguf",
        "resident-kv",
        62,
        3072,
        prefix_tokens=16,
        resident_kv_bytes_per_token=253_952,
    ),
    Case(
        "qwen3next",
        "Qwen3Next",
        "bartowski/Qwen_Qwen3-Coder-Next-GGUF:IQ2_XS",
        HOME
        / ".cache/huggingface/hub/models--bartowski--Qwen_Qwen3-Coder-Next-GGUF/snapshots/d32741c4b434bf1f927798d0c093564c7f4e92fd/Qwen_Qwen3-Coder-Next-IQ2_XS.gguf",
        "kv-recurrent",
        48,
        2048,
        prefix_tokens=16,
    ),
]


def http_json(url: str, payload: dict[str, Any] | None = None, timeout: float = 30.0) -> Any:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["content-type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_ready(port: int, proc: subprocess.Popen[str], timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server exited early with code {proc.returncode}")
        try:
            http_json(f"http://127.0.0.1:{port}/health", timeout=2.0)
            return
        except Exception as exc:  # noqa: BLE001 - readiness loop keeps last error.
            last_error = exc
            time.sleep(0.5)
    raise TimeoutError(f"llama-server did not become ready: {last_error}")


def warm_mean_ms(runs: list[dict[str, Any]]) -> float | None:
    values = [run.get("elapsed_ms") for run in runs[1:] if isinstance(run.get("elapsed_ms"), (int, float))]
    if not values:
        return None
    return sum(values) / len(values)


def median_ms(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def warm_median_ms(runs: list[dict[str, Any]]) -> float | None:
    values = [run.get("elapsed_ms") for run in runs[1:] if isinstance(run.get("elapsed_ms"), (int, float))]
    return median_ms(values)


def skippy_hit_median_ms(skippy: dict[str, Any]) -> float | None:
    imports = skippy.get("cache_hit_import_ms")
    decodes = skippy.get("cache_hit_decode_ms")
    if not isinstance(imports, list) or not isinstance(decodes, list):
        return None
    values = [
        float(import_ms) + float(decode_ms)
        for import_ms, decode_ms in zip(imports, decodes)
        if isinstance(import_ms, (int, float)) and isinstance(decode_ms, (int, float))
    ]
    return median_ms(values)


def run_correctness(case: Case, args: argparse.Namespace, case_dir: Path, prompt: str | None = None) -> dict[str, Any]:
    report_path = case_dir / "skippy-state-handoff.json"
    cache_hit_repeats = args.cache_hit_repeats or case.cache_hit_repeats
    cmd = [
        str(args.skippy_correctness_bin),
        "state-handoff",
        "--model",
        str(case.model_path),
        "--model-id",
        case.model_id,
        "--layer-end",
        str(case.layer_end),
        "--ctx-size",
        str(case.ctx_size),
        f"--n-gpu-layers={case.n_gpu_layers}",
        "--stage-load-mode",
        case.stage_load_mode,
        "--state-layer-end",
        str(case.state_layer_end or case.layer_end),
        "--state-payload-kind",
        case.payload,
        "--prefix-token-count",
        str(case.prefix_tokens),
        "--suffix-token-count",
        str(args.suffix_token_count),
        "--cache-hit-repeats",
        str(cache_hit_repeats),
        "--report-out",
        str(report_path),
    ]
    if prompt is not None:
        cmd.extend(["--prompt", prompt])
    if case.state_layer_start:
        cmd.extend(["--state-layer-start", str(case.state_layer_start)])
    if case.state_stage_index is not None:
        cmd.extend(["--state-stage-index", str(case.state_stage_index)])
    if args.runtime_lane_count is not None:
        cmd.extend(["--runtime-lane-count", str(args.runtime_lane_count)])
    if args.borrow_resident_hits:
        cmd.append("--borrow-resident-hits")
    if args.cache_decoded_result_hits:
        cmd.append("--cache-decoded-result-hits")
    env = os.environ.copy()
    env["LLAMA_STAGE_BUILD_DIR"] = str(args.llama_stage_build_dir)
    started = time.monotonic()
    completed = subprocess.run(
        cmd,
        cwd=REPO,
        env=env,
        text=True,
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=args.correctness_timeout_secs,
    )
    elapsed_ms = (time.monotonic() - started) * 1000
    (case_dir / "skippy-state-handoff.log").write_text(completed.stdout)
    if completed.returncode != 0:
        return {
            "status": "fail",
            "exit_code": completed.returncode,
            "elapsed_ms": elapsed_ms,
            "log": str(case_dir / "skippy-state-handoff.log"),
        }
    report = json.loads(report_path.read_text())
    report["runner_elapsed_ms"] = elapsed_ms
    return report


def run_llama_server(case: Case, prompt: str, args: argparse.Namespace, case_dir: Path) -> dict[str, Any]:
    port = free_port()
    log_path = case_dir / "llama-server.log"
    with log_path.open("w") as log:
        cmd = [
            str(args.llama_server_bin),
            "--model",
            str(case.model_path),
            "--ctx-size",
            str(case.ctx_size),
            "--n-gpu-layers",
            str(case.n_gpu_layers),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--parallel",
            str(args.llama_parallel),
            "--no-webui",
        ]
        proc = subprocess.Popen(
            cmd,
            cwd=REPO,
            text=True,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            wait_ready(port, proc, args.server_startup_timeout_secs)
            runs = []
            for index in range(args.llama_repeats):
                payload = {
                    "prompt": prompt,
                    "n_predict": 1,
                    "temperature": 0,
                    "top_k": 1,
                    "cache_prompt": True,
                }
                started = time.monotonic()
                response = http_json(f"http://127.0.0.1:{port}/completion", payload, timeout=args.request_timeout_secs)
                elapsed_ms = (time.monotonic() - started) * 1000
                timings = response.get("timings", {})
                runs.append(
                    {
                        "run": index + 1,
                        "elapsed_ms": elapsed_ms,
                        "content": response.get("content"),
                        "tokens_evaluated": timings.get("tokens_evaluated"),
                        "tokens_predicted": timings.get("tokens_predicted"),
                        "tokens_cached": timings.get("tokens_cached"),
                        "prompt_n": timings.get("prompt_n"),
                        "cache_n": timings.get("cache_n"),
                        "prompt_ms": timings.get("prompt_ms"),
                        "predicted_ms": timings.get("predicted_ms"),
                    }
                )
            return {
                "status": "ok",
                "log": str(log_path),
                "parallel": args.llama_parallel,
                "runs": runs,
                "warm_mean_ms": warm_mean_ms(runs),
                "warm_median_ms": warm_median_ms(runs),
            }
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)


def format_ms(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.1f}"
    return "n/a"


def format_bytes(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "n/a"
    if value == 0:
        return "0"
    if value < 1024 * 1024:
        kib = value / 1024
        return f"{kib:.1f} KiB"
    mib = value / (1024 * 1024)
    return f"{mib:.1f} MiB"


def cache_storage_bytes(row: dict[str, Any]) -> int | None:
    skippy = row.get("skippy", {})
    measured = skippy.get("cache_storage_bytes")
    if isinstance(measured, (int, float)):
        return int(measured)
    case = row.get("case", {})
    bytes_per_token = case.get("resident_kv_bytes_per_token")
    prefix_tokens = row.get("prefix_tokens")
    if row.get("payload") == "resident-kv" and isinstance(bytes_per_token, int) and isinstance(prefix_tokens, int):
        return bytes_per_token * prefix_tokens
    return None


def cache_storage_method(row: dict[str, Any]) -> str:
    skippy = row.get("skippy", {})
    if isinstance(skippy.get("cache_storage_bytes"), (int, float)):
        return "measured"
    if cache_storage_bytes(row) is not None:
        return "metadata-derived"
    return "n/a"


def markdown_table(results: list[dict[str, Any]]) -> str:
    include_use_case = any(row.get("use_case") for row in results)
    if include_use_case:
        lines = [
            "| Use case | Family | Payload | Correctness | Prefix tokens | Prompt tokens | llama-server warm median ms | Skippy hit median ms | Speedup | Cache bytes | Size method | Notes |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    else:
        lines = [
            "| Family | Payload | Correctness | Prefix tokens | Prompt tokens | llama-server warm median ms | Skippy hit median ms | Speedup | Cache bytes | Size method | Notes |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    for row in results:
        correctness = row.get("skippy", {}).get("status", "missing")
        llama_warm = row.get("llama_server", {}).get("warm_median_ms")
        if llama_warm is None:
            llama_warm = row.get("llama_server", {}).get("warm_mean_ms")
        skippy_hit = skippy_hit_median_ms(row.get("skippy", {}))
        if skippy_hit is None:
            skippy_hit = row.get("skippy", {}).get("cache_hit_total_ms")
        speedup = None
        if isinstance(llama_warm, (int, float)) and isinstance(skippy_hit, (int, float)) and skippy_hit > 0:
            speedup = llama_warm / skippy_hit
        notes = row.get("notes", "")
        cells = {
            "use_case": row.get("use_case_label", "n/a").replace("|", "/"),
            "family": row["family"],
            "payload": row["payload"],
            "correctness": correctness,
            "prefix": row.get("prefix_tokens", "n/a"),
            "tokens": row.get("benchmark_prompt_token_count", "n/a"),
            "llama": format_ms(llama_warm),
            "skippy": format_ms(skippy_hit),
            "speedup": f"{speedup:.2f}x" if speedup is not None else "n/a",
            "bytes": format_bytes(cache_storage_bytes(row)),
            "method": cache_storage_method(row),
            "notes": notes.replace("|", "/"),
        }
        if include_use_case:
            lines.append(
                "| {use_case} | {family} | `{payload}` | {correctness} | {prefix} | {tokens} | {llama} | {skippy} | {speedup} | {bytes} | {method} | {notes} |".format(
                    **cells
                )
            )
        else:
            lines.append(
                "| {family} | `{payload}` | {correctness} | {prefix} | {tokens} | {llama} | {skippy} | {speedup} | {bytes} | {method} | {notes} |".format(
                    **cells
                )
            )
    return "\n".join(lines) + "\n"


def parse_prefix_sweep(value: str | None) -> list[int | None]:
    if value is None:
        return [None]
    sizes = []
    for raw in value.split(","):
        raw = raw.strip()
        if not raw:
            continue
        size = int(raw)
        if size <= 0:
            raise SystemExit("--prefix-token-sweep values must be positive")
        sizes.append(size)
    if not sizes:
        raise SystemExit("--prefix-token-sweep did not contain any sizes")
    return sizes


def run_case(case: Case, args: argparse.Namespace, use_case: UseCase | None = None) -> dict[str, Any]:
    cache_hit_repeats = args.cache_hit_repeats or case.cache_hit_repeats
    prefix_tokens = args.prefix_tokens
    if prefix_tokens is None and use_case is not None:
        prefix_tokens = use_case.prefix_tokens
    n_gpu_layers = args.n_gpu_layers if args.n_gpu_layers is not None else case.n_gpu_layers
    if prefix_tokens is not None or n_gpu_layers != case.n_gpu_layers or cache_hit_repeats != case.cache_hit_repeats:
        case = Case(
            key=case.key,
            family=case.family,
            model_id=case.model_id,
            model_path=case.model_path,
            payload=case.payload,
            layer_end=case.layer_end,
            activation_width=case.activation_width,
            ctx_size=max(case.ctx_size, prefix_tokens + args.suffix_token_count + 128)
            if prefix_tokens is not None
            else case.ctx_size,
            n_gpu_layers=n_gpu_layers,
            prefix_tokens=prefix_tokens if prefix_tokens is not None else case.prefix_tokens,
            cache_hit_repeats=cache_hit_repeats,
            stage_load_mode=case.stage_load_mode,
            state_layer_start=case.state_layer_start,
            state_layer_end=case.state_layer_end,
            state_stage_index=case.state_stage_index,
            resident_kv_bytes_per_token=case.resident_kv_bytes_per_token,
            skip_llama_server_reason=case.skip_llama_server_reason,
        )
    case_dir = args.output_dir / f"{case.key}-p{case.prefix_tokens}"
    if use_case is not None:
        case_dir = args.output_dir / use_case.key / f"{case.key}-p{case.prefix_tokens}"
    case_dir.mkdir(parents=True, exist_ok=True)
    row: dict[str, Any] = {
        "key": case.key,
        "family": case.family,
        "model_id": case.model_id,
        "model_path": str(case.model_path) if case.model_path else None,
        "payload": case.payload,
        "prefix_tokens": case.prefix_tokens,
        "stage_load_mode": case.stage_load_mode,
        "state_layer_start": case.state_layer_start,
        "state_layer_end": case.state_layer_end or case.layer_end,
        "case": asdict(case) | {"model_path": str(case.model_path) if case.model_path else None},
    }
    if use_case is not None:
        row["use_case"] = use_case.key
        row["use_case_label"] = use_case.label
        row["use_case_source"] = {
            "dataset": use_case.source_dataset,
            "config": use_case.source_config,
            "split": use_case.source_split,
            "row_idx": use_case.source_row,
        }
    if case.model_path is None or not case.model_path.exists():
        row["skippy"] = {"status": "missing-model"}
        row["llama_server"] = {"status": "missing-model"}
        row["notes"] = "No local full GGUF available."
        return row

    print(f"==> {case.key}: Skippy {case.payload}", flush=True)
    try:
        skippy = run_correctness(case, args, case_dir, use_case.prompt if use_case is not None else None)
    except subprocess.TimeoutExpired as exc:
        skippy = {"status": "timeout", "timeout_secs": args.correctness_timeout_secs, "cmd": exc.cmd}
    except Exception as exc:  # noqa: BLE001 - benchmark continues across families.
        skippy = {"status": "error", "error": str(exc)}
    row["skippy"] = skippy
    row["benchmark_prompt_token_count"] = skippy.get("benchmark_prompt_token_count")

    if skippy.get("status") != "pass":
        row["llama_server"] = {"status": "skipped"}
        row["notes"] = "Skipped llama-server baseline because production cache correctness did not pass."
        return row
    if case.skip_llama_server_reason:
        row["llama_server"] = {"status": "skipped", "reason": case.skip_llama_server_reason}
        row["notes"] = case.skip_llama_server_reason
        return row
    if case.stage_load_mode != "runtime-slice":
        row["llama_server"] = {"status": "skipped", "reason": "llama-server requires a full GGUF"}
        row["notes"] = "llama-server baseline skipped because this case uses a layer package."
        return row
    if args.skip_llama_server:
        row["llama_server"] = {"status": "skipped"}
        row["notes"] = "llama-server baseline skipped by request."
        return row

    print(f"==> {case.key}: llama-server baseline", flush=True)
    try:
        row["llama_server"] = run_llama_server(case, skippy["benchmark_prompt_text"], args, case_dir)
        row["notes"] = ""
    except subprocess.TimeoutExpired as exc:
        row["llama_server"] = {"status": "timeout", "timeout_secs": args.request_timeout_secs, "cmd": exc.cmd}
        row["notes"] = "llama-server timed out."
    except Exception as exc:  # noqa: BLE001 - benchmark continues across families.
        row["llama_server"] = {"status": "error", "error": str(exc)}
        row["notes"] = "llama-server baseline failed."
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/skippy-cache-production-bench"))
    parser.add_argument("--case", action="append", help="Run only the named case; may be repeated.")
    parser.add_argument("--skip-llama-server", action="store_true")
    parser.add_argument("--llama-server-bin", type=Path, default=REPO / ".deps/llama-build/build-stage-abi-cpu/bin/llama-server")
    parser.add_argument("--skippy-correctness-bin", type=Path, default=REPO / "target/debug/skippy-correctness")
    parser.add_argument("--llama-stage-build-dir", type=Path, default=REPO / ".deps/llama-build/build-stage-abi-cpu")
    parser.add_argument("--correctness-timeout-secs", type=int, default=900)
    parser.add_argument("--server-startup-timeout-secs", type=int, default=600)
    parser.add_argument("--request-timeout-secs", type=int, default=600)
    parser.add_argument("--llama-repeats", type=int, default=3)
    parser.add_argument("--cache-hit-repeats", type=int, help="Override Skippy cache-hit repeats for every selected case.")
    parser.add_argument("--llama-parallel", type=int, default=1)
    parser.add_argument("--runtime-lane-count", type=int)
    parser.add_argument("--n-gpu-layers", type=int, help="Override n_gpu_layers for every selected case.")
    parser.add_argument("--borrow-resident-hits", action="store_true")
    parser.add_argument("--cache-decoded-result-hits", action="store_true")
    parser.add_argument("--prefix-tokens", type=int, help="Override the production prefix-token count for every selected case.")
    parser.add_argument("--suffix-token-count", type=int, default=0)
    parser.add_argument("--use-case", action="append", help="Run one named use case from the corpus; use 'all' for every use case.")
    parser.add_argument(
        "--use-case-corpus",
        type=Path,
        default=REPO / "evals/skippy-usecase-corpus.json",
        help="JSON corpus with HF-derived benchmark use-case prompts.",
    )
    parser.add_argument(
        "--prefix-token-sweep",
        help="Comma-separated prefix-token sizes to run as one benchmark sweep, for example 512,2048,8192.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected = CASES
    if args.case:
        wanted = set(args.case)
        selected = [case for case in CASES if case.key in wanted]
        missing = wanted - {case.key for case in selected}
        if missing:
            raise SystemExit(f"unknown case(s): {', '.join(sorted(missing))}")

    selected_use_cases: list[UseCase | None] = [None]
    if args.use_case:
        use_cases = load_use_cases(args.use_case_corpus)
        wanted_use_cases = set(args.use_case)
        if "all" in wanted_use_cases:
            selected_use_cases = use_cases
        else:
            selected_use_cases = [use_case for use_case in use_cases if use_case.key in wanted_use_cases]
            missing = wanted_use_cases - {use_case.key for use_case in selected_use_cases}
            if missing:
                raise SystemExit(f"unknown use case(s): {', '.join(sorted(missing))}")

    prefix_sweep = parse_prefix_sweep(args.prefix_token_sweep)
    if args.prefix_tokens is not None:
        prefix_sweep = [args.prefix_tokens]
    results = []
    for prefix_tokens in prefix_sweep:
        args.prefix_tokens = prefix_tokens
        for use_case in selected_use_cases:
            for case in selected:
                row = run_case(case, args, use_case)
                results.append(row)
                (args.output_dir / "production-cache-bench.json").write_text(json.dumps(results, indent=2))
                (args.output_dir / "production-cache-bench.md").write_text(markdown_table(results))

    print(markdown_table(results))
    print(f"Wrote {args.output_dir / 'production-cache-bench.json'}")
    print(f"Wrote {args.output_dir / 'production-cache-bench.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
