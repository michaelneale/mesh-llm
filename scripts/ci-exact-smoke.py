#!/usr/bin/env python3
"""Run the deterministic exact smoke suite against one mesh-llm model/backend."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any

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


def case_dir() -> Path | None:
    raw = os.environ.get("VALIDATION_CASE_DIR", "").strip()
    if not raw:
        return None
    return Path(raw)


def temp_root_path() -> Path:
    custom = os.environ.get("TMPDIR")
    if custom:
        return Path(custom)
    return Path(tempfile.gettempdir())


def sync_runtime_logs(case_directory: Path | None, mesh_log_path: Path) -> None:
    if case_directory is None:
        return
    case_directory.mkdir(parents=True, exist_ok=True)
    if mesh_log_path.exists():
        shutil.copyfile(mesh_log_path, case_directory / "mesh.log")

    temp_dir = temp_root_path()
    for source_name, target_name in (
        ("mesh-llm-llama-server.log", "llama-server.log"),
        ("mesh-llm-rpc-server.log", "rpc-server.log"),
    ):
        source_path = temp_dir / source_name
        if source_path.exists():
            shutil.copyfile(source_path, case_directory / target_name)


def model_root_for(model_arg: str) -> Path:
    path = Path(model_arg)
    return path if path.is_dir() else path.parent


def ensure_expected_template_source(model_arg: str, expected_template_source: str) -> None:
    model_root = model_root_for(model_arg)
    expected_path = model_root / expected_template_source
    if not expected_path.exists():
        print(
            f"❌ Expected template source file not found in model directory: {expected_template_source}",
            file=sys.stderr,
        )
        print(f"Model directory: {model_root}", file=sys.stderr)
        raise SystemExit(1)


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


def wait_until_ready(
    process: subprocess.Popen[str],
    console_port: int,
    log_path: Path,
    timeout: int,
) -> None:
    status_url = f"http://127.0.0.1:{console_port}/api/status"
    for second in range(1, timeout + 1):
        sync_runtime_logs(case_dir(), log_path)
        if process.poll() is not None:
            print("❌ mesh-llm exited unexpectedly", file=sys.stderr)
            print(log_path.read_text(encoding="utf-8", errors="replace")[-8000:], file=sys.stderr)
            raise SystemExit(1)

        try:
            status = http_json(status_url, timeout=5)
            if bool(status.get("llama_ready", False)):
                print(f"✅ Model loaded in {second}s", flush=True)
                return
        except Exception:
            pass

        if second % 15 == 0:
            print(f"  Still waiting... ({second}s)", flush=True)
        time.sleep(1)

    sync_runtime_logs(case_dir(), log_path)
    print(f"❌ Model failed to load within {timeout}s", file=sys.stderr)
    print(log_path.read_text(encoding="utf-8", errors="replace")[-8000:], file=sys.stderr)
    raise SystemExit(1)


def normalize(text: str) -> str:
    return text.strip()


def record_chat_artifact(
    label: str,
    prompt_text: str,
    request_payload: dict[str, Any],
    response_payload: dict[str, Any],
    content: str,
    finish_reason: str,
    expectations: dict[str, Any],
) -> None:
    artifact_root = case_dir()
    if artifact_root is None:
        return

    chat_dir = artifact_root / "chat"
    chat_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "label": label,
        "prompt": prompt_text,
        "request": request_payload,
        "raw_response": response_payload,
        "content": content,
        "response_text": content,
        "finish_reason": finish_reason,
        "expectations": expectations,
    }
    (chat_dir / f"{label}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def fail(message: str, *, content: str = "", response: dict[str, Any] | None = None, log_path: Path | None = None) -> None:
    print(f"❌ {message}", file=sys.stderr)
    if content:
        print(f"Content: {content}", file=sys.stderr)
    if response is not None:
        print(f"Raw response: {json.dumps(response, ensure_ascii=False)}", file=sys.stderr)
    if log_path is not None and log_path.exists():
        print("--- Log tail ---", file=sys.stderr)
        print(log_path.read_text(encoding="utf-8", errors="replace")[-8000:], file=sys.stderr)
    raise SystemExit(1)


def case_failure(message: str, *, content: str = "", response: dict[str, Any] | None = None) -> str:
    details = message
    if content:
        details += f" | Content: {content}"
    if response is not None:
        details += f" | Raw response: {json.dumps(response, ensure_ascii=False)}"
    return details


def run_chat(
    api_port: int,
    prompt_text: str,
    *,
    max_tokens: int,
    enable_thinking: bool,
) -> tuple[dict[str, Any], dict[str, Any], str, str]:
    payload = {
        "model": "any",
        "messages": [{"role": "user", "content": prompt_text}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "top_p": 1,
        "top_k": 1,
        "seed": 123,
        "enable_thinking": enable_thinking,
    }
    response = http_json(
        f"http://127.0.0.1:{api_port}/v1/chat/completions",
        payload=payload,
        timeout=DEFAULT_REQUEST_TIMEOUT,
    )
    choice = response["choices"][0]
    content = choice["message"]["content"]
    finish_reason = choice.get("finish_reason", "")
    return payload, response, content, finish_reason


def validate_case(
    *,
    api_port: int,
    case_cfg: dict[str, Any],
    default_prompt: str,
    log_path: Path,
) -> tuple[bool, str]:
    label = case_cfg.get("label", "primary")
    prompt_text = case_cfg.get("prompt", default_prompt)
    expect_contains = str(case_cfg.get("expect_contains", ""))
    expect_contains_ci = str(case_cfg.get("expect_contains_ci", ""))
    expect_contains_all_ci = list(case_cfg.get("expect_contains_all_ci", []))
    expect_any_ci = list(case_cfg.get("expect_any_ci", []))
    forbid_contains = str(case_cfg.get("forbid_contains", ""))
    expect_exact = str(case_cfg.get("expect_exact", ""))
    thinking_mode = str(case_cfg.get("thinking_mode", ""))
    max_tokens = int(case_cfg.get("max_tokens", 32) or 32)

    print(f"Testing /v1/chat/completions ({label})...", flush=True)
    request_payload: dict[str, Any] = {}
    response_payload: dict[str, Any] | None = None
    content = ""
    finish_reason = ""
    error: str | None = None

    try:
        request_payload, response_payload, content, finish_reason = run_chat(
            api_port,
            prompt_text,
            max_tokens=max_tokens,
            enable_thinking=False,
        )
    except Exception as exc:
        error = case_failure(f"Request failed: {exc}")

    if error is None and not content:
        error = case_failure("Empty response from inference", response=response_payload)
    if error is None and "<think>" in content:
        error = case_failure("Unexpected reasoning output with enable_thinking=false", content=content)
    if error is None and expect_contains and expect_contains not in content:
        error = case_failure(f"Response did not contain expected text: {expect_contains}", content=content)
    if error is None and expect_contains_ci and expect_contains_ci.lower() not in content.lower():
        error = case_failure(
            f"Response did not contain expected text (case-insensitive): {expect_contains_ci}",
            content=content,
        )
    if error is None and expect_contains_all_ci:
        missing = [needle for needle in expect_contains_all_ci if needle.lower() not in content.lower()]
        if missing:
            error = case_failure(
                f"Response did not contain all expected terms (case-insensitive): {', '.join(missing)}",
                content=content,
            )
    if error is None and expect_any_ci and not any(needle.lower() in content.lower() for needle in expect_any_ci):
        error = case_failure(
            f"Response did not contain any expected text (case-insensitive): {json.dumps(expect_any_ci)}",
            content=content,
        )
    if error is None and expect_exact and normalize(content) != normalize(expect_exact):
        error = case_failure(
            "Response did not exactly match expected text",
            content=f"expected={normalize(expect_exact)!r} actual={normalize(content)!r}",
        )
    if error is None and forbid_contains and forbid_contains in content:
        error = case_failure(f"Response contained forbidden text: {forbid_contains}", content=content)
    if error is None and not finish_reason:
        error = case_failure("Missing finish_reason in response", response=response_payload)

    record_chat_artifact(
        label,
        prompt_text,
        request_payload,
        response_payload or {},
        content,
        finish_reason,
        {
            "expect_contains": expect_contains,
            "expect_contains_ci": expect_contains_ci,
            "expect_contains_all_ci": expect_contains_all_ci,
            "expect_any_ci": expect_any_ci,
            "forbid_contains": forbid_contains,
            "expect_exact": expect_exact,
        },
    )
    if error is not None:
        print(f"❌ {error}", flush=True)
        return False, error

    print(f"✅ Inference response: {content}", flush=True)

    if thinking_mode:
        print(f"Testing explicit reasoning output ({label})...", flush=True)
        try:
            think_request, think_response, think_content, _ = run_chat(
                api_port,
                prompt_text,
                max_tokens=64,
                enable_thinking=True,
            )
        except Exception as exc:
            error = case_failure(f"Explicit reasoning request failed: {exc}")
            print(f"❌ {error}", flush=True)
            return False, error
        if not think_content:
            error = case_failure("Empty response from explicit reasoning request", response=think_response)
        elif thinking_mode == "tagged":
            if "<think>" not in think_content:
                error = case_failure("Explicit reasoning response did not contain <think> tags", content=think_content)
        elif thinking_mode == "multiline":
            if think_content == content:
                error = case_failure("Explicit reasoning response matched non-thinking response", content=think_content)
            elif "\n" not in think_content:
                error = case_failure("Explicit reasoning response was not multiline", content=think_content)
        else:
            fail(f"Unknown thinking mode: {thinking_mode}")

        if error is not None:
            print(f"❌ {error}", flush=True)
            return False, error

        record_chat_artifact(
            f"{label}.thinking",
            prompt_text,
            think_request,
            think_response,
            think_content,
            "stop",
            {
                "expect_contains": "",
                "expect_contains_ci": "",
                "expect_contains_all_ci": [],
                "expect_any_ci": [],
                "forbid_contains": "",
                "expect_exact": "",
            },
        )
        print(f"✅ Explicit reasoning response: {think_content}", flush=True)

    return True, content


def write_models_artifact(api_port: int) -> None:
    models = http_json(f"http://127.0.0.1:{api_port}/v1/models", timeout=DEFAULT_REQUEST_TIMEOUT)
    model_count = len(models.get("data", []))
    if model_count == 0:
        fail("No models in /v1/models", response=models)
    artifact_root = case_dir()
    if artifact_root is not None:
        models_dir = artifact_root / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        (models_dir / "v1-models.json").write_text(
            json.dumps(models, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(f"✅ /v1/models returned {model_count} model(s)", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["gguf", "mlx"], required=True)
    parser.add_argument("--mesh-llm", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--bin-dir", default="")
    parser.add_argument("--expected-template-source", default="")
    parser.add_argument("--prompt", default="Reply with exactly: blue")
    parser.add_argument("--expect-contains", default="")
    parser.add_argument("--forbid-contains", default="")
    parser.add_argument("--expect-exact", default="")
    parser.add_argument("--prompt-suite-json", default="")
    parser.add_argument("--wait-seconds", type=int, default=DEFAULT_WAIT_SECONDS)
    args = parser.parse_args()

    if args.backend == "gguf" and not args.bin_dir:
        parser.error("--bin-dir is required for gguf backend")

    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    api_port = pick_free_port()
    console_port = pick_free_port()
    while api_port == console_port:
        console_port = pick_free_port()

    print("=== CI Exact Smoke Test ===", flush=True)
    print(f"  backend:   {args.backend}", flush=True)
    print(f"  mesh-llm:  {args.mesh_llm}", flush=True)
    if args.backend == "gguf":
        print(f"  bin-dir:   {args.bin_dir}", flush=True)
    print(f"  model:     {args.model}", flush=True)
    print(f"  api port:  {api_port}", flush=True)
    print(f"  os:        {os.uname().sysname}", flush=True)
    print(f"  prompt:    {args.prompt}", flush=True)

    if args.backend == "mlx" and args.expected_template_source:
        ensure_expected_template_source(args.model, args.expected_template_source)

    with tempfile.TemporaryDirectory(prefix="mesh-llm-exact-") as temp_dir:
        os.environ["TMPDIR"] = temp_dir
        log_path = Path(temp_dir) / "mesh-llm.log"
        with open(log_path, "w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                build_launch_command(args, api_port, console_port),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
                env={**os.environ, "RUST_LOG": os.environ.get("RUST_LOG", "info")},
            )
            try:
                wait_until_ready(
                    process,
                    console_port,
                    log_path,
                    args.wait_seconds,
                )

                primary_case = {
                    "label": "primary",
                    "prompt": args.prompt,
                    "expect_contains": args.expect_contains,
                    "forbid_contains": args.forbid_contains,
                    "expect_exact": args.expect_exact,
                }
                failures: list[tuple[str, str]] = []

                ok, details = validate_case(
                    api_port=api_port,
                    case_cfg=primary_case,
                    default_prompt=args.prompt,
                    log_path=log_path,
                )
                if not ok:
                    failures.append((primary_case["label"], details))

                if args.prompt_suite_json:
                    print("Running extra prompt suite...", flush=True)
                    suite = json.loads(args.prompt_suite_json)
                    for index, case_cfg in enumerate(suite, start=1):
                        case_cfg = dict(case_cfg)
                        case_cfg.setdefault("label", f"case-{index}")
                        ok, details = validate_case(
                            api_port=api_port,
                            case_cfg=case_cfg,
                            default_prompt=args.prompt,
                            log_path=log_path,
                        )
                        if not ok:
                            failures.append((str(case_cfg["label"]), details))

                print("Testing /v1/models...", flush=True)
                write_models_artifact(api_port)

                if failures:
                    print("\n=== Exact smoke test failed ===", flush=True)
                    print("Failed cases:", flush=True)
                    for label, details in failures:
                        print(f"  - {label}: {details}", flush=True)
                    return 1

                print("\n=== Exact smoke test passed ===", flush=True)
                return 0
            finally:
                sync_runtime_logs(case_dir(), log_path)
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    pass
                time.sleep(2)
                sync_runtime_logs(case_dir(), log_path)
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
                sync_runtime_logs(case_dir(), log_path)


if __name__ == "__main__":
    raise SystemExit(main())
