#!/usr/bin/env python3
"""Generate reproducible benchmark corpora from downloaded Hugging Face data."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "crates/skippy-bench/corpora/bench_corpus_sources.json"
DEFAULT_OUT_ROOT = ROOT / "target/bench-corpora"
DEFAULT_HF_DIR = ROOT / "target/hf-datasets"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tier")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--hf-dir", type=Path, default=DEFAULT_HF_DIR)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-prompt-chars", type=int)
    parser.add_argument("--sample-multiplier", type=int, default=20)
    args = parser.parse_args()

    require_hf()
    require_duckdb()

    config = read_json(args.config)
    seed = args.seed if args.seed is not None else int(config["seed"])
    tier = args.tier
    if tier not in config.get("tiers", {}):
        valid = ", ".join(sorted(config.get("tiers", {}).keys()))
        raise RuntimeError(f"unknown tier {tier!r}; valid tiers: {valid}")
    tier_config = config.get("tiers", {}).get(tier, {})
    max_prompt_chars = (
        args.max_prompt_chars
        if args.max_prompt_chars is not None
        else int(tier_config.get("max_prompt_chars", 6000))
    )
    target_prompt_chars = tier_config.get("target_prompt_chars")
    target_prompt_chars = int(target_prompt_chars) if target_prompt_chars is not None else None
    out_root = args.out_root if args.out_root.is_absolute() else ROOT / args.out_root
    hf_root = args.hf_dir if args.hf_dir.is_absolute() else ROOT / args.hf_dir
    out_dir = out_root / tier
    out_dir.mkdir(parents=True, exist_ok=True)
    hf_root.mkdir(parents=True, exist_ok=True)
    corpus_path = out_dir / "corpus.jsonl"
    manifest_path = out_dir / "manifest.json"

    rows: list[dict[str, Any]] = []
    source_manifests: list[dict[str, Any]] = []
    for source in config["sources"]:
        quota = int(source.get("quota", {}).get(tier, 0))
        if quota <= 0:
            continue
        info = hf_dataset_info(source["dataset"], source["revision"])
        resolved_revision = info.get("sha") or source["revision"]
        local_dir = download_source(source, resolved_revision, hf_root)
        parquet_files = find_parquet_files(local_dir, source)
        candidate_count = max(quota * args.sample_multiplier, quota + 25)
        raw_rows = sample_rows(
            parquet_files=parquet_files,
            source=source,
            seed=seed,
            limit=candidate_count,
        )
        accepted = 0
        generated = 0
        for raw in raw_rows:
            if accepted >= quota:
                break
            if tier == "coding-loop":
                normalized_rows = normalize_loop_rows(
                    tier=tier,
                    source=source,
                    resolved_revision=resolved_revision,
                    sample_idx=accepted,
                    row=raw,
                    max_prompt_chars=max_prompt_chars,
                    target_prompt_chars=target_prompt_chars,
                )
                if not normalized_rows:
                    continue
                rows.extend(normalized_rows)
                generated += len(normalized_rows)
            else:
                normalized = normalize_row(
                    tier=tier,
                    source=source,
                    resolved_revision=resolved_revision,
                    sample_idx=accepted,
                    row=raw,
                    max_prompt_chars=max_prompt_chars,
                    target_prompt_chars=target_prompt_chars,
                )
                if normalized is None:
                    continue
                rows.append(normalized)
                generated += 1
            accepted += 1
        if accepted < quota:
            raise RuntimeError(f"{source['name']} produced {accepted}/{quota} rows")
        source_manifests.append(
            {
                "name": source["name"],
                "dataset": source["dataset"],
                "config": source["config"],
                "split": source["split"],
                "revision": source["revision"],
                "resolved_revision": resolved_revision,
                "family": source["family"],
                "adapter": source["adapter"],
                "routing_hint": source.get("routing_hint"),
                "quota": quota,
                "generated_rows": generated,
                "download_dir": rel(local_dir),
                "parquet_files": [rel(path) for path in parquet_files],
            }
        )

    with corpus_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    manifest = {
        "schema_version": config["schema_version"],
        "tier": tier,
        "seed": seed,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "generator": "scripts/generate-bench-corpus.py",
        "generator_git_commit": git_commit(),
        "corpus_path": rel(corpus_path),
        "row_count": len(rows),
        "max_prompt_chars": max_prompt_chars,
        "target_prompt_chars": target_prompt_chars,
        "hf_download_root": rel(hf_root),
        "sources": source_manifests,
    }
    write_json(manifest_path, manifest)

    print(f"generated {len(rows)} rows")
    print(f"corpus:   {corpus_path}")
    print(f"manifest: {manifest_path}")
    return 0


def require_hf() -> None:
    try:
        subprocess.run(["hf", "--version"], check=True, stdout=subprocess.DEVNULL)
    except Exception as error:
        raise RuntimeError("missing Hugging Face CLI `hf`; install it before generating corpora") from error


def require_duckdb() -> None:
    if python_has_duckdb(sys.executable) or command_exists("uv"):
        return
    raise RuntimeError(
        "DuckDB is required to sample downloaded parquet files. Install the "
        "DuckDB Python package or install `uv` so the generator can run with "
        "`uv run --with duckdb`."
    )


def python_has_duckdb(python: str) -> bool:
    return (
        subprocess.run(
            [python, "-c", "import duckdb"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )


def command_exists(name: str) -> bool:
    return (
        subprocess.run(
            ["bash", "-lc", f"command -v {name} >/dev/null"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def git_commit() -> str | None:
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return output.strip()
    except Exception:
        return None


def hf_dataset_info(dataset: str, revision: str) -> dict[str, Any]:
    output = subprocess.check_output(
        ["hf", "datasets", "info", dataset, "--revision", revision, "--format", "json"],
        cwd=ROOT,
        text=True,
    )
    return json.loads(output)


def download_source(source: dict[str, Any], revision: str, hf_root: Path) -> Path:
    local_dir = hf_root / safe_name(source["dataset"]) / revision
    local_dir.mkdir(parents=True, exist_ok=True)
    include = parquet_include_patterns(source)
    command = [
        "hf",
        "download",
        source["dataset"],
        "--type",
        "dataset",
        "--revision",
        revision,
        "--local-dir",
        str(local_dir),
        "--quiet",
    ]
    for pattern in include:
        command.extend(["--include", pattern])
    subprocess.run(command, cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    try:
        find_parquet_files(local_dir, source)
    except RuntimeError:
        download_converted_parquet(source, local_dir)
    return local_dir


def download_converted_parquet(source: dict[str, Any], local_dir: Path) -> None:
    output = subprocess.check_output(
        [
            "hf",
            "datasets",
            "parquet",
            source["dataset"],
            "--subset",
            source["config"],
            "--split",
            source["split"],
            "--format",
            "json",
        ],
        cwd=ROOT,
        text=True,
    )
    entries = json.loads(output)
    if not entries:
        raise RuntimeError(f"no HF parquet URLs for {source['name']}")
    target_dir = local_dir / source["config"] / source["split"]
    target_dir.mkdir(parents=True, exist_ok=True)
    for index, entry in enumerate(entries):
        url = entry["url"]
        size = int(entry.get("size") or 0)
        path = target_dir / f"{index:04d}.parquet"
        if path.exists() and (size == 0 or path.stat().st_size == size):
            continue
        request = urllib.request.Request(url, headers=hf_headers())
        with urllib.request.urlopen(request, timeout=120) as response:
            path.write_bytes(response.read())


def hf_headers() -> dict[str, str]:
    headers = {"User-Agent": "skippy-runtime-bench-corpus/1"}
    token = os.environ.get("HF_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def parquet_include_patterns(source: dict[str, Any]) -> list[str]:
    config = source["config"]
    split = source["split"]
    return [
        f"{config}/{split}-*.parquet",
        f"{config}/{split}/*.parquet",
        f"{split}-*.parquet",
        f"**/{config}/{split}-*.parquet",
        f"**/{config}/{split}/*.parquet",
        f"**/{split}-*.parquet",
    ]


def find_parquet_files(local_dir: Path, source: dict[str, Any]) -> list[Path]:
    config = source["config"]
    split = source["split"]
    files = sorted(local_dir.rglob("*.parquet"))
    matches = [
        path
        for path in files
        if (path.name.startswith(f"{split}-") or path.parent.name == split)
        and (path.parent.name == config or config in path.parts)
    ]
    if not matches and config == "default":
        matches = [
            path
            for path in files
            if path.name.startswith(f"{split}-") or path.parent.name == split
        ]
    if not matches:
        raise RuntimeError(f"no parquet files found for {source['name']} in {local_dir}")
    return matches


def sample_rows(
    parquet_files: list[Path],
    source: dict[str, Any],
    seed: int,
    limit: int,
) -> list[dict[str, Any]]:
    table_expr = "[" + ",".join(sql_string(str(path)) for path in parquet_files) + "]"
    material = f"{seed}:{source['name']}:{source['dataset']}:{source['config']}:{source['split']}"
    source_seed = int.from_bytes(hashlib.sha256(material.encode()).digest()[:8], "little")
    query = f"""
WITH rows AS (
  SELECT *, row_number() OVER () - 1 AS __bench_row_idx
  FROM read_parquet({table_expr})
)
SELECT * EXCLUDE (__bench_row_idx)
FROM rows
ORDER BY hash(__bench_row_idx + {source_seed})
LIMIT {limit}
"""
    output = run_duckdb_json(query)
    return json.loads(output)


def run_duckdb_json(query: str) -> str:
    code = """
import duckdb
import json
import sys

query = sys.stdin.read()
connection = duckdb.connect()
result = connection.execute(query)
columns = [column[0] for column in result.description]
rows = [dict(zip(columns, row)) for row in result.fetchall()]
print(json.dumps(rows, ensure_ascii=False, default=str))
"""
    if python_has_duckdb(sys.executable):
        command = [sys.executable, "-c", code]
    else:
        command = ["uv", "run", "--with", "duckdb", "python", "-c", code]
    return subprocess.check_output(command, cwd=ROOT, input=query, text=True)


def sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def safe_name(value: str) -> str:
    return value.replace("/", "--")


def normalize_row(
    tier: str,
    source: dict[str, Any],
    resolved_revision: str,
    sample_idx: int,
    row: dict[str, Any],
    max_prompt_chars: int,
    target_prompt_chars: int | None,
) -> dict[str, Any] | None:
    adapter = source["adapter"]
    prompt, expected, metadata, session_group = ADAPTERS[adapter](row)
    if prompt is None:
        return None
    if target_prompt_chars is not None:
        prompt = expand_prompt_to_chars(prompt, target_prompt_chars)
        metadata = {
            "long_context_expanded": True,
            "long_context_target_chars": target_prompt_chars,
            **metadata,
        }
    prompt = truncate_text(prompt, max_prompt_chars)
    row_id = f"{source['name']}:{source['split']}:{sample_idx:05d}"
    return {
        "id": row_id,
        "tier": tier,
        "family": source["family"],
        "source": source["dataset"],
        "source_config": source["config"],
        "source_revision": resolved_revision,
        "split": source["split"],
        "session_group": session_group or f"{source['name']}:{source['family']}",
        "prompt": prompt,
        "expected_output": expected,
        "metadata": {
            "source_name": source["name"],
            "adapter": adapter,
            "routing_hint": source.get("routing_hint"),
            **metadata,
        },
    }


def normalize_loop_rows(
    tier: str,
    source: dict[str, Any],
    resolved_revision: str,
    sample_idx: int,
    row: dict[str, Any],
    max_prompt_chars: int,
    target_prompt_chars: int | None,
) -> list[dict[str, Any]]:
    adapter = source["adapter"]
    builder = LOOP_ADAPTERS.get(adapter)
    if builder is None:
        return []
    turns, expected, metadata, session_group = builder(row)
    if not turns:
        return []
    base_id = f"{source['name']}:{source['split']}:{sample_idx:05d}"
    group = session_group or f"{source['name']}:coding-loop:{sample_idx:05d}"
    normalized: list[dict[str, Any]] = []
    for turn_index, prompt in enumerate(turns, start=1):
        if target_prompt_chars is not None:
            prompt = expand_prompt_to_chars(prompt, target_prompt_chars)
        prompt = truncate_text(prompt, max_prompt_chars)
        normalized.append(
            {
                "id": f"{base_id}:turn-{turn_index:02d}",
                "tier": tier,
                "family": "coding_edit_loop",
                "source": source["dataset"],
                "source_config": source["config"],
                "source_revision": resolved_revision,
                "split": source["split"],
                "session_group": group,
                "prompt": prompt,
                "expected_output": expected if turn_index == 1 else None,
                "metadata": {
                    "source_name": source["name"],
                    "adapter": adapter,
                    "routing_hint": "ngram",
                    "benchmark_shape": "repeated_edit_loop",
                    "original_family": source["family"],
                    "loop_turn": turn_index,
                    "loop_turns": len(turns),
                    **(
                        {
                            "long_context_expanded": True,
                            "long_context_target_chars": target_prompt_chars,
                        }
                        if target_prompt_chars is not None
                        else {}
                    ),
                    **metadata,
                },
            }
        )
    return normalized


def expand_prompt_to_chars(prompt: str, target_chars: int) -> str:
    prompt = clean_text(prompt)
    if len(prompt) >= target_chars:
        return prompt
    sections = [
        "Long-context stress packet built from HF-sourced text. "
        "Use this tier for context-capacity and transport stress, not quality scoring."
    ]
    section_index = 1
    while sum(len(section) for section in sections) + len(sections) * 2 < target_chars:
        sections.append(f"Source excerpt repeat {section_index}:\n{prompt}")
        section_index += 1
    return "\n\n".join(sections)


def truncate_text(value: str, max_chars: int) -> str:
    value = clean_text(value)
    if len(value) <= max_chars:
        return value
    keep_head = max_chars * 2 // 3
    keep_tail = max_chars - keep_head - 80
    return (
        value[:keep_head].rstrip()
        + "\n\n...[truncated for benchmark prompt budget]...\n\n"
        + value[-keep_tail:].lstrip()
    )


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.replace("\r\n", "\n").replace("\r", "\n").strip()
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def commitpack_edit(row: dict[str, Any]) -> tuple[str | None, Any, dict[str, Any], str | None]:
    old = clean_text(row.get("old_contents"))
    new = clean_text(row.get("new_contents"))
    if not old or not new:
        return None, None, {}, None
    subject = clean_text(row.get("subject"))
    message = clean_text(row.get("message"))
    file_name = clean_text(row.get("old_file") or row.get("new_file"))
    prompt = f"""Apply the following code change.

File: {file_name}
Commit subject: {subject}
Commit message:
{message}

Current file contents:
```{row.get("lang", "")}
{old}
```

Return the updated file contents only."""
    metadata = {"language": row.get("lang"), "file": file_name}
    return prompt, new, metadata, f"commitpackft:{row.get('repos') or file_name}"


def code_refinement(row: dict[str, Any]) -> tuple[str | None, Any, dict[str, Any], str | None]:
    buggy = clean_text(row.get("buggy"))
    fixed = clean_text(row.get("fixed"))
    if not buggy or not fixed:
        return None, None, {}, None
    prompt = f"""Fix the bug in this code. Return the corrected code only.

```c
{buggy}
```"""
    return prompt, fixed, {"task": "code_refinement"}, "codexglue:code_refinement"


def swe_smith_trajectory_loop(row: dict[str, Any]) -> tuple[list[str], Any, dict[str, Any], str | None]:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return [], None, {}, None
    transcript: list[tuple[str, str]] = []
    turns: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = clean_text(message.get("role"))
        content = clean_text(message.get("content"))
        if not role or not content:
            continue
        if role == "system":
            continue
        if role == "assistant" and transcript:
            turns.append(agent_trajectory_prompt(transcript))
            if len(turns) >= 8:
                break
        transcript.append((role, content))
    if len(turns) < 2:
        return [], None, {}, None
    metadata = {
        "instance_id": row.get("instance_id"),
        "traj_id": row.get("traj_id"),
        "model": row.get("model"),
        "resolved": row.get("resolved"),
    }
    return turns, row.get("patch"), metadata, f"swe-smith:{row.get('instance_id') or row.get('traj_id') or sample_key(row)}"


def agent_trajectory_prompt(transcript: list[tuple[str, str]]) -> str:
    rendered: list[str] = []
    for role, content in transcript[-12:]:
        rendered.append(f"{role.upper()}:\n{content}")
    return (
        "Continue this software engineering agent session.\n\n"
        "Transcript so far:\n"
        + "\n\n".join(rendered)
        + "\n\nRespond with the next assistant action or code edit."
    )


def sample_key(row: dict[str, Any]) -> str:
    material = json.dumps(row, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(material.encode()).hexdigest()[:12]


def swe_bench_issue(row: dict[str, Any]) -> tuple[str | None, Any, dict[str, Any], str | None]:
    statement = clean_text(row.get("problem_statement"))
    if not statement:
        return None, None, {}, None
    repo = clean_text(row.get("repo"))
    hints = clean_text(row.get("hints_text"))
    prompt = f"""You are working in repository `{repo}`.

Resolve this GitHub issue:
{statement}
"""
    if hints:
        prompt += f"\nHints:\n{hints}\n"
    prompt += "\nDescribe the likely code changes and tests you would make."
    metadata = {"repo": repo, "difficulty": row.get("difficulty")}
    return prompt, row.get("patch"), metadata, f"swebench:{repo}"


def apps_codegen(row: dict[str, Any]) -> tuple[str | None, Any, dict[str, Any], str | None]:
    question = clean_text(row.get("question"))
    if not question:
        return None, None, {}, None
    starter = clean_text(row.get("starter_code"))
    prompt = "Solve this programming problem in Python.\n\n" + question
    if starter:
        prompt += f"\n\nStarter code:\n```python\n{starter}\n```"
    return prompt, row.get("solutions"), {"difficulty": row.get("difficulty")}, "apps:codegen"


def codesearchnet_explain(row: dict[str, Any]) -> tuple[str | None, Any, dict[str, Any], str | None]:
    code = clean_text(row.get("code"))
    comment = clean_text(row.get("comment"))
    if not code or not comment:
        return None, None, {}, None
    prompt = f"""Explain what this code does and identify any edge cases.

```
{code}
```"""
    return prompt, comment, {"task": "code_explain"}, "codesearchnet:explain"


def xlam_tool_call(row: dict[str, Any]) -> tuple[str | None, Any, dict[str, Any], str | None]:
    query = clean_text(row.get("query"))
    tools = clean_text(row.get("tools"))
    if not query or not tools:
        return None, None, {}, None
    prompt = f"""Choose the tool call or calls needed for the user request.
Return only JSON.

Available tools:
{tools}

User request:
{query}"""
    return prompt, row.get("answers"), {"task": "tool_call"}, "xlam:tool_call"


def spider_sql(row: dict[str, Any]) -> tuple[str | None, Any, dict[str, Any], str | None]:
    schema = clean_text(row.get("db_schema"))
    question = clean_text(row.get("question"))
    if not schema or not question:
        return None, None, {}, None
    prompt = f"""Write a SQL query for this database schema and question.
Return only SQL.

Schema:
{schema}

Question:
{question}"""
    return prompt, row.get("query"), {"db_id": row.get("db_id")}, f"spider:{row.get('db_id')}"


def oasst_prompt(row: dict[str, Any]) -> tuple[str | None, Any, dict[str, Any], str | None]:
    if row.get("role") != "prompter" or row.get("lang") != "en":
        return None, None, {}, None
    text = clean_text(row.get("text"))
    if not text:
        return None, None, {}, None
    return text, None, {"lang": row.get("lang")}, f"oasst2:{row.get('message_id')}"


def dolly_instruction(row: dict[str, Any]) -> tuple[str | None, Any, dict[str, Any], str | None]:
    instruction = clean_text(row.get("instruction"))
    context = clean_text(row.get("context"))
    if not instruction:
        return None, None, {}, None
    prompt = instruction
    if context:
        prompt += f"\n\nContext:\n{context}"
    return prompt, row.get("response"), {"category": row.get("category")}, f"dolly:{row.get('category')}"


def gsm8k_reasoning(row: dict[str, Any]) -> tuple[str | None, Any, dict[str, Any], str | None]:
    question = clean_text(row.get("question"))
    if not question:
        return None, None, {}, None
    prompt = f"Solve this math problem step by step.\n\n{question}"
    return prompt, row.get("answer"), {"task": "math_reasoning"}, "gsm8k:math"


def xsum_summarize(row: dict[str, Any]) -> tuple[str | None, Any, dict[str, Any], str | None]:
    document = clean_text(row.get("document"))
    if not document:
        return None, None, {}, None
    prompt = f"Summarize this article in one concise paragraph.\n\n{document}"
    return prompt, row.get("summary"), {"task": "summarization"}, "xsum:summarization"


ADAPTERS = {
    "commitpack_edit": commitpack_edit,
    "code_refinement": code_refinement,
    "swe_bench_issue": swe_bench_issue,
    "apps_codegen": apps_codegen,
    "codesearchnet_explain": codesearchnet_explain,
    "xlam_tool_call": xlam_tool_call,
    "spider_sql": spider_sql,
    "oasst_prompt": oasst_prompt,
    "dolly_instruction": dolly_instruction,
    "gsm8k_reasoning": gsm8k_reasoning,
    "xsum_summarize": xsum_summarize,
}

LOOP_ADAPTERS = {
    "swe_smith_trajectory_loop": swe_smith_trajectory_loop,
}


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
