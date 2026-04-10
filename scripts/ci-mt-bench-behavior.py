#!/usr/bin/env python3
"""Run the full MT-Bench prompt dataset against one mesh-llm model/backend.

This is a behavior regression harness, not a correctness evaluator. It runs the
entire HuggingFaceH4/mt_bench_prompts dataset against a model and applies cheap
heuristics to catch empty outputs, reasoning leakage, and repetition / looping.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import re
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

DEFAULT_DATASET = "HuggingFaceH4/mt_bench_prompts"
DATASET_SERVER = "https://datasets-server.huggingface.co/rows"
DEFAULT_WAIT_SECONDS = 300
DEFAULT_REQUEST_TIMEOUT = 300


def pick_free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def http_json(url: str, payload: dict[str, Any] | None = None, timeout: int = 60) -> dict[str, Any]:
    if payload is None:
        request = urllib.request.Request(url)
    else:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def fetch_mt_bench_prompts(dataset: str) -> list[dict[str, Any]]:
    encoded = urllib.parse.quote(dataset, safe="")
    rows: list[dict[str, Any]] = []
    offset = 0
    page_size = 100
    while True:
        url = (
            f"{DATASET_SERVER}?dataset={encoded}&config=default&split=train"
            f"&offset={offset}&length={page_size}"
        )
        payload = http_json(url, timeout=60)
        page = payload.get("rows", [])
        rows.extend(item["row"] for item in page)
        if not payload.get("partial") and len(rows) >= int(payload.get("num_rows_total", len(rows))):
            break
        if not page:
            break
        offset += len(page)
    return rows


def tokenize_words(text: str) -> list[str]:
    return re.findall(r"\S+", text.lower())


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [part.strip().lower() for part in parts if part.strip()]


def repeated_ngram(tokens: list[str], size: int, threshold: int) -> str | None:
    if len(tokens) < size:
        return None
    counts = Counter(tuple(tokens[i : i + size]) for i in range(0, len(tokens) - size + 1))
    for ngram, count in counts.items():
        if count >= threshold:
            return " ".join(ngram)
    return None


def analyze_output(content: str) -> list[str]:
    issues: list[str] = []
    normalized = content.strip()
    if not normalized:
        return ["empty output"]
    if "<think>" in normalized or "</think>" in normalized:
        issues.append("reasoning markup leaked with enable_thinking=false")
    if len(normalized) > 6000:
        issues.append(f"output too long ({len(normalized)} chars)")

    lines = [line.strip() for line in normalized.splitlines() if line.strip()]
    repeated_lines = [line for line, count in Counter(lines).items() if count >= 3]
    if repeated_lines:
        issues.append(f"repeated line x3: {repeated_lines[0][:120]}")

    sentences = split_sentences(normalized)
    repeated_sentences = [s for s, count in Counter(sentences).items() if count >= 3]
    if repeated_sentences:
        issues.append(f"repeated sentence x3: {repeated_sentences[0][:120]}")

    tokens = tokenize_words(normalized)
    ngram = repeated_ngram(tokens, size=6, threshold=3)
    if ngram is not None:
        issues.append(f"repeated 6-gram x3: {ngram[:120]}")

    if len(tokens) >= 80:
        tail = tokens[-80:]
        unique_ratio = len(set(tail)) / len(tail)
        if unique_ratio < 0.30:
            issues.append(f"low tail token diversity ({unique_ratio:.2f})")

    return issues


def build_launch_command(args: argparse.Namespace, api_port: int, console_port: int) -> list[str]:
    command = [args.mesh_llm]
    if args.backend == "mlx":
        if os.path.isdir(args.model):
            command.extend(["--mlx-file", args.model])
        else:
            command.extend(["--model", args.model, "--mlx-file", args.model])
    else:
        command.extend(["--gguf-file", args.model, "--bin-dir", args.bin_dir])
    command.extend(["--no-draft", "--port", str(api_port), "--console", str(console_port)])
    return command


def behavior_case_dir() -> Path | None:
    raw = os.environ.get("VALIDATION_CASE_DIR", "").strip()
    if not raw:
        return None
    return Path(raw)


def sync_runtime_logs(case_dir: Path | None, mesh_log_path: Path) -> None:
    if case_dir is None:
        return

    case_dir.mkdir(parents=True, exist_ok=True)

    if mesh_log_path.exists():
        shutil.copyfile(mesh_log_path, case_dir / "mesh.log")

    temp_dir = Path(tempfile.gettempdir())
    for source_name, target_name in (
        ("mesh-llm-llama-server.log", "llama-server.log"),
        ("mesh-llm-rpc-server.log", "rpc-server.log"),
    ):
        source_path = temp_dir / source_name
        if source_path.exists():
            shutil.copyfile(source_path, case_dir / target_name)


def write_case_progress(
    case_dir: Path | None,
    *,
    status: str,
    backend: str,
    model: str,
    prompt_count: int,
    completed_prompts: int,
    failed_prompts: int,
    current_prompt_id: str = "",
    current_category: str = "",
) -> None:
    if case_dir is None:
        return
    payload = {
        "status": status,
        "backend": backend,
        "model": model,
        "prompt_count": prompt_count,
        "completed_prompts": completed_prompts,
        "failed_prompt_count": failed_prompts,
        "current_prompt_id": current_prompt_id,
        "current_category": current_category,
    }
    (case_dir / "progress.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def wait_until_ready(process: subprocess.Popen[str], console_port: int, log_path: Path, timeout: int) -> None:
    status_url = f"http://127.0.0.1:{console_port}/api/status"
    case_dir = behavior_case_dir()
    for second in range(1, timeout + 1):
        sync_runtime_logs(case_dir, log_path)
        if process.poll() is not None:
            print("❌ mesh-llm exited unexpectedly", file=sys.stderr)
            print(log_path.read_text(encoding="utf-8", errors="replace")[-8000:], file=sys.stderr)
            raise SystemExit(1)
        try:
            status = http_json(status_url, timeout=5)
            if bool(status.get("llama_ready", False)):
                print(f"✅ Model loaded in {second}s")
                return
        except Exception:
            pass
        if second % 15 == 0:
            print(f"  Still waiting... ({second}s)", flush=True)
        time.sleep(1)
    sync_runtime_logs(case_dir, log_path)
    print(f"❌ Model failed to load within {timeout}s", file=sys.stderr)
    print(log_path.read_text(encoding="utf-8", errors="replace")[-8000:], file=sys.stderr)
    raise SystemExit(1)


def run_chat(api_port: int, messages: list[dict[str, str]], max_tokens: int) -> dict[str, Any]:
    payload = {
        "model": "any",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "enable_thinking": False,
    }
    return http_json(
        f"http://127.0.0.1:{api_port}/v1/chat/completions",
        payload=payload,
        timeout=DEFAULT_REQUEST_TIMEOUT,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["gguf", "mlx"], required=True)
    parser.add_argument("--mesh-llm", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--bin-dir", default="")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--max-prompts", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument("--wait-seconds", type=int, default=DEFAULT_WAIT_SECONDS)
    parser.add_argument("--mesh-log-output", default="")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--label", default="")
    args = parser.parse_args()

    if args.backend == "gguf" and not args.bin_dir:
        parser.error("--bin-dir is required for gguf backend")

    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    print("=== MT-Bench Behavior Smoke ===", flush=True)
    print(f"  backend: {args.backend}", flush=True)
    print(f"  model:   {args.model}", flush=True)
    print(f"  dataset: {args.dataset}", flush=True)

    prompts = fetch_mt_bench_prompts(args.dataset)
    if args.max_prompts > 0:
        prompts = prompts[: args.max_prompts]
    print(f"  prompts: {len(prompts)}", flush=True)
    case_dir = behavior_case_dir()
    write_case_progress(
        case_dir,
        status="starting",
        backend=args.backend,
        model=args.label or args.model,
        prompt_count=len(prompts),
        completed_prompts=0,
        failed_prompts=0,
    )

    api_port = pick_free_port()
    console_port = pick_free_port()
    while api_port == console_port:
        console_port = pick_free_port()

    with tempfile.TemporaryDirectory(prefix="mesh-llm-behavior-") as temp_dir:
        log_path = Path(temp_dir) / "mesh-llm.log"
        log_file = open(log_path, "w", encoding="utf-8")
        process = subprocess.Popen(
            build_launch_command(args, api_port, console_port),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
            env={**os.environ, "RUST_LOG": os.environ.get("RUST_LOG", "info")},
        )
        try:
            sync_runtime_logs(case_dir, log_path)
            wait_until_ready(process, console_port, log_path, args.wait_seconds)
            write_case_progress(
                case_dir,
                status="running",
                backend=args.backend,
                model=args.label or args.model,
                prompt_count=len(prompts),
                completed_prompts=0,
                failed_prompts=0,
            )

            results: list[dict[str, Any]] = []
            failed = 0
            for index, row in enumerate(prompts, start=1):
                prompt_turns = row.get("prompt", [])
                messages: list[dict[str, str]] = []
                turn_results: list[dict[str, Any]] = []
                row_failed = False
                for turn_index, prompt_text in enumerate(prompt_turns, start=1):
                    messages.append({"role": "user", "content": prompt_text})
                    try:
                        response = run_chat(api_port, messages, args.max_tokens)
                    except Exception as exc:
                        row_failed = True
                        failed += 1
                        turn_results.append(
                            {
                                "turn": turn_index,
                                "prompt": prompt_text,
                                "failure": f"request failed: {exc}",
                            }
                        )
                        break

                    choice = response["choices"][0]
                    content = choice["message"]["content"]
                    issues = analyze_output(content)
                    finish_reason = choice.get("finish_reason", "")
                    if not finish_reason:
                        issues.append("missing finish_reason")
                    turn_results.append(
                        {
                            "turn": turn_index,
                            "prompt": prompt_text,
                            "content": content,
                            "finish_reason": finish_reason,
                            "issues": issues,
                        }
                    )
                    messages.append({"role": "assistant", "content": content})
                    if issues:
                        row_failed = True
                        failed += 1
                        break

                results.append(
                    {
                        "index": index,
                        "prompt_id": row.get("prompt_id"),
                        "category": row.get("category"),
                        "turns": turn_results,
                        "passed": not row_failed,
                    }
                )
                status = "PASS" if not row_failed else "FAIL"
                print(
                    f"[{index:02d}/{len(prompts):02d}] {status} {row.get('category')}#{row.get('prompt_id')}",
                    flush=True,
                )
                sync_runtime_logs(case_dir, log_path)
                write_case_progress(
                    case_dir,
                    status="running",
                    backend=args.backend,
                    model=args.label or args.model,
                    prompt_count=len(prompts),
                    completed_prompts=index,
                    failed_prompts=failed,
                    current_prompt_id=str(row.get("prompt_id", "")),
                    current_category=str(row.get("category", "")),
                )

            output = {
                "label": args.label or args.model,
                "backend": args.backend,
                "model": args.model,
                "dataset": args.dataset,
                "prompt_count": len(prompts),
                "failed_prompt_count": failed,
                "results": results,
            }
            output_path = Path(args.output_json)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
            if args.mesh_log_output:
                mesh_log_output = Path(args.mesh_log_output)
                mesh_log_output.parent.mkdir(parents=True, exist_ok=True)
                mesh_log_output.write_text(
                    log_path.read_text(encoding="utf-8", errors="replace"),
                    encoding="utf-8",
                )
            sync_runtime_logs(case_dir, log_path)
            write_case_progress(
                case_dir,
                status="completed",
                backend=args.backend,
                model=args.label or args.model,
                prompt_count=len(prompts),
                completed_prompts=len(prompts),
                failed_prompts=failed,
            )

            if failed:
                print(f"❌ Behavior smoke failed: {failed} prompt(s) flagged", file=sys.stderr)
                print(f"Summary written to {output_path}", file=sys.stderr)
                return 1
            print("✅ Behavior smoke passed")
            print(f"Summary written to {output_path}")
            return 0
        finally:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
            time.sleep(2)
            sync_runtime_logs(case_dir, log_path)
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            sync_runtime_logs(case_dir, log_path)
            log_file.close()


if __name__ == "__main__":
    raise SystemExit(main())
