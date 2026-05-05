#!/usr/bin/env python3
"""Render Skippy cache family benchmark results as README-ready tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]

FAMILY_ORDER = [
    "Qwen3Next",
    "Falcon-H1",
    "Llama",
    "Qwen3 dense",
    "DeepSeek2",
    "GLM-4.7 Flash",
    "GLM4",
    "Gemma4 A4B",
    "Gemma4 E4B",
    "Gemma3",
    "Gemma2",
    "OLMo",
    "MiniMax M2.7",
]

USE_CASE_ORDER = [
    "tool_calling",
    "text_to_sql",
    "coding_agent_loop",
    "issue_fixing",
    "code_refinement",
    "few_shot_reasoning",
    "open_chat",
    "summarization_rag",
]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def family_index(family: str) -> int:
    try:
        return FAMILY_ORDER.index(family)
    except ValueError:
        return len(FAMILY_ORDER)


def use_case_index(key: str) -> int:
    try:
        return USE_CASE_ORDER.index(key)
    except ValueError:
        return len(USE_CASE_ORDER)


def payload_label(value: Any) -> str:
    labels = {
        "resident-kv": "ResidentKv",
        "kv-recurrent": "KvRecurrent",
        "full-state": "FullState",
        "recurrent-only": "RecurrentOnly",
    }
    return labels.get(str(value), str(value))


def markdown_escape(value: Any) -> str:
    return str(value).replace("|", "/")


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
        return f"{value / 1024:.1f} KiB"
    return f"{value / (1024 * 1024):.1f} MiB"


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


def median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def skippy_hit_median_ms(row: dict[str, Any]) -> float | None:
    skippy = row.get("skippy", {})
    imports = skippy.get("cache_hit_import_ms")
    decodes = skippy.get("cache_hit_decode_ms")
    if isinstance(imports, list) and isinstance(decodes, list):
        values = [
            float(import_ms) + float(decode_ms)
            for import_ms, decode_ms in zip(imports, decodes)
            if isinstance(import_ms, (int, float)) and isinstance(decode_ms, (int, float))
        ]
        return median(values)
    total = skippy.get("cache_hit_total_ms")
    if isinstance(total, (int, float)):
        return float(total)
    return None


def llama_warm_median_ms(row: dict[str, Any]) -> float | None:
    llama_server = row.get("llama_server", {})
    value = llama_server.get("warm_median_ms")
    if isinstance(value, (int, float)):
        return float(value)
    value = llama_server.get("warm_mean_ms")
    if isinstance(value, (int, float)):
        return float(value)
    return None


def speedup(row: dict[str, Any], *, allow_skippy_recompute: bool = False) -> float | None:
    llama_ms = llama_warm_median_ms(row)
    if llama_ms is None and allow_skippy_recompute:
        recompute_ms = row.get("skippy", {}).get("recompute_total_ms")
        if isinstance(recompute_ms, (int, float)):
            llama_ms = float(recompute_ms)
    skippy_ms = skippy_hit_median_ms(row)
    if llama_ms is None or skippy_ms is None or skippy_ms <= 0:
        return None
    return llama_ms / skippy_ms


def speedup_cell(row: dict[str, Any], *, bold: bool = False, allow_skippy_recompute: bool = False) -> str:
    value = speedup(row, allow_skippy_recompute=allow_skippy_recompute)
    if value is None:
        return "n/a"
    text = f"{value:.2f}x"
    if bold:
        return f"**{text} faster**"
    return text


def correctness(row: dict[str, Any]) -> str:
    return str(row.get("skippy", {}).get("status", "missing"))


def full_gguf_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [
        row
        for row in results
        if not row.get("use_case")
        and row.get("stage_load_mode") == "runtime-slice"
        and row.get("llama_server", {}).get("status") == "ok"
    ]
    return sorted(rows, key=lambda row: (family_index(row.get("family", "")), row.get("family", "")))


def package_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [
        row
        for row in results
        if not row.get("use_case")
        and row.get("stage_load_mode") != "runtime-slice"
    ]
    return sorted(rows, key=lambda row: (family_index(row.get("family", "")), row.get("family", "")))


def use_case_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [
        row
        for row in results
        if row.get("use_case")
        and row.get("stage_load_mode") == "runtime-slice"
        and row.get("llama_server", {}).get("status") == "ok"
    ]
    return sorted(
        rows,
        key=lambda row: (
            use_case_index(row.get("use_case", "")),
            family_index(row.get("family", "")),
            row.get("family", ""),
        ),
    )


def render_full_gguf_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Family | Representative model ref | Production payload | Correctness | Prefix tokens | Prompt tokens | llama-server warm median ms | Skippy hit median ms | Skippy win | Cache bytes | Size method | Notes |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {family} | `{model}` | `{payload}` | {correctness} | {prefix} | {tokens} | {llama} | {skippy} | {win} | {bytes} | {method} | {notes} |".format(
                family=markdown_escape(row.get("family", "")),
                model=markdown_escape(row.get("model_id", "")),
                payload=payload_label(row.get("payload")),
                correctness=correctness(row),
                prefix=row.get("prefix_tokens", "n/a"),
                tokens=row.get("benchmark_prompt_token_count", "n/a"),
                llama=format_ms(llama_warm_median_ms(row)),
                skippy=format_ms(skippy_hit_median_ms(row)),
                win=speedup_cell(row, bold=True),
                bytes=format_bytes(cache_storage_bytes(row)),
                method=cache_storage_method(row),
                notes=markdown_escape(row.get("notes", "")),
            )
        )
    return "\n".join(lines)


def render_use_case_matrix(rows: list[dict[str, Any]]) -> str:
    families = sorted({row.get("family", "") for row in rows}, key=lambda family: (family_index(family), family))
    by_key = {(row.get("use_case"), row.get("family")): row for row in rows}
    labels = {
        row.get("use_case"): row.get("use_case_label", row.get("use_case"))
        for row in rows
    }
    use_cases = sorted({row.get("use_case", "") for row in rows}, key=use_case_index)
    lines = [
        "| Use case | " + " | ".join(markdown_escape(family) for family in families) + " |",
        "| --- | " + " | ".join("---:" for _ in families) + " |",
    ]
    for use_case in use_cases:
        cells = [markdown_escape(labels.get(use_case, use_case))]
        for family in families:
            row = by_key.get((use_case, family))
            cells.append(speedup_cell(row) if row is not None else "n/a")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def load_use_case_sources(path: Path) -> list[dict[str, Any]]:
    data = read_json(path)
    rows = []
    for item in data.get("use_cases", []):
        source = item.get("source", {})
        rows.append(
            {
                "key": item.get("key", ""),
                "label": item.get("label", item.get("key", "")),
                "dataset": source.get("dataset", ""),
                "config": source.get("config", ""),
                "split": source.get("split", ""),
                "row": source.get("row_idx", ""),
            }
        )
    return sorted(rows, key=lambda row: use_case_index(row["key"]))


def render_source_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Use case | Dataset | Config | Split | Row |",
        "| --- | --- | --- | --- | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {label} | `{dataset}` | `{config}` | `{split}` | {row_id} |".format(
                label=markdown_escape(row["label"]),
                dataset=markdown_escape(row["dataset"]),
                config=markdown_escape(row["config"]),
                split=markdown_escape(row["split"]),
                row_id=row["row"],
            )
        )
    return "\n".join(lines)


def render_package_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Family | Representative model ref | Production payload | Correctness | Prefix tokens | Prompt tokens | Baseline | Skippy hit median ms | Skippy win | Cache bytes | Size method | Notes |",
        "| --- | --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {family} | `{model}` | `{payload}` | {correctness} | {prefix} | {tokens} | {baseline} | {skippy} | {win} | {bytes} | {method} | {notes} |".format(
                family=markdown_escape(row.get("family", "")),
                model=markdown_escape(row.get("model_id", "")),
                payload=payload_label(row.get("payload")),
                correctness=correctness(row),
                prefix=row.get("prefix_tokens", "n/a"),
                tokens=row.get("benchmark_prompt_token_count", "n/a"),
                baseline="Skippy stage recompute",
                skippy=format_ms(skippy_hit_median_ms(row)),
                win=speedup_cell(row, bold=True, allow_skippy_recompute=True),
                bytes=format_bytes(cache_storage_bytes(row)),
                method=cache_storage_method(row),
                notes=markdown_escape(row.get("notes", "")),
            )
        )
    return "\n".join(lines)


def render_report(results: list[dict[str, Any]], use_case_corpus: Path) -> str:
    sections = [
        "### Full-GGUF llama-server vs Skippy",
        "",
        "Rows are ordered so related runtime/cache families appear next to each other.",
        "",
        render_full_gguf_table(full_gguf_rows(results)),
        "",
        "### Use-Case Benchmark Matrix",
        "",
        "This matrix uses one Hugging Face-sourced representative prompt per use case,",
        "the same requested prefix tokens, one generated token, Skippy",
        "`--runtime-lane-count 1`, llama-server `--parallel 1`, and the same full-GGUF",
        "family set as the table above. Values are Skippy warm-hit latency speedup over",
        "llama-server warm-cache latency. DeepSeek3 stays in the package-only section",
        "because there is no practical local full-GGUF llama-server baseline for that",
        "artifact.",
        "",
        render_use_case_matrix(use_case_rows(results)),
        "",
        "Prompt sources are checked in at `evals/skippy-usecase-corpus.json` with source",
        "dataset metadata:",
        "",
        render_source_table(load_use_case_sources(use_case_corpus)),
        "",
        "### Package-Only Giant Models",
        "",
        "These rows validate cache strategy for models where a full llama-server",
        "baseline is not operationally useful because monolithic residency is too large.",
        "",
        render_package_table(package_rows(results)),
        "",
    ]
    return "\n".join(sections)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        required=True,
        help="production-cache-bench.json from skippy-cache-production-bench.py; may be repeated.",
    )
    parser.add_argument("--output", type=Path, help="Write README-ready markdown tables here.")
    parser.add_argument(
        "--use-case-corpus",
        type=Path,
        default=REPO / "evals/skippy-usecase-corpus.json",
        help="HF-derived use-case corpus JSON.",
    )
    args = parser.parse_args()

    results = []
    for path in args.input:
        results.extend(read_json(path))
    report = render_report(results, args.use_case_corpus)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report)
        print(f"Wrote {args.output}")
    else:
        print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
