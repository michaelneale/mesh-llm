#!/usr/bin/env python3
"""Generate Hugging Face Dataset Viewer artifacts for meshllm/catalog.

The runtime catalog uses nested entry JSON files under entries/**/*.json. That
shape is good for model resolution but bad for the Hugging Face Dataset Viewer:
each entry has variant names as dynamic object keys, so the viewer cannot infer a
single table schema.

This script emits a flat JSONL table and a dataset-card README that pins the
Viewer to that JSONL file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


README = """---
pretty_name: Mesh-LLM Catalog
license: apache-2.0
configs:
- config_name: default
  data_files:
  - split: train
    path: catalog_rows.jsonl
---

# Mesh-LLM Catalog

This dataset is the Hugging Face-backed catalog for Mesh-LLM.

The runtime catalog entries live under `entries/**/*.json`. The Dataset Viewer
uses `catalog_rows.jsonl`, a flat generated table with one row per model variant.

The catalog deliberately excludes raw blob URLs. Entries should resolve to
Hugging Face repositories and canonical Mesh refs.
"""


def scalar_string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def scalar_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    return 0


def flatten_entry(path: Path, entries_dir: Path) -> list[dict[str, Any]]:
    entry = json.loads(path.read_text())
    relative_path = path.relative_to(entries_dir.parent).as_posix()
    source_repo = scalar_string(entry.get("source_repo"))
    schema_version = scalar_int(entry.get("schema_version"))
    variants = entry.get("variants") or {}
    rows: list[dict[str, Any]] = []

    if isinstance(variants, dict):
        variant_items = variants.items()
    elif isinstance(variants, list):
        variant_items = (
            ((variant.get("curated") or {}).get("name") or f"variant-{index}", variant)
            for index, variant in enumerate(variants)
            if isinstance(variant, dict)
        )
    else:
        variant_items = []

    for variant_id, variant in variant_items:
        if not isinstance(variant, dict):
            continue

        source = variant.get("source") or {}
        curated = variant.get("curated") or {}
        packages = variant.get("packages") or []
        if not isinstance(packages, list):
            packages = []

        layer_packages = [
            package
            for package in packages
            if isinstance(package, dict) and package.get("type") == "layer-package"
        ]
        package_repos = [
            scalar_string(package.get("repo"))
            for package in layer_packages
            if package.get("repo")
        ]
        primary_package = layer_packages[0] if layer_packages else {}

        rows.append(
            {
                "schema_version": schema_version,
                "entry_path": relative_path,
                "source_repo": source_repo,
                "variant_id": scalar_string(variant_id),
                "name": scalar_string(curated.get("name") or variant_id),
                "size": scalar_string(curated.get("size")),
                "description": scalar_string(curated.get("description")),
                "source_model_repo": scalar_string(source.get("repo") or source_repo),
                "source_revision": scalar_string(source.get("revision") or "main"),
                "source_file": scalar_string(source.get("file")),
                "draft_model": scalar_string(curated.get("draft")),
                "moe": scalar_string(curated.get("moe")),
                "mmproj_json": scalar_string(curated.get("mmproj")),
                "extra_files_json": scalar_string(curated.get("extra_files") or []),
                "runtime": "multi-machine" if layer_packages else "single-machine",
                "layer_package_available": bool(layer_packages),
                "package_count": len(layer_packages),
                "package_repos": ",".join(package_repos),
                "primary_package_repo": scalar_string(primary_package.get("repo")),
                "primary_package_layer_count": scalar_int(
                    primary_package.get("layer_count")
                ),
                "primary_package_total_bytes": scalar_int(
                    primary_package.get("total_bytes")
                ),
            }
        )

    return rows


def generate(entries_dir: Path, output_dir: Path) -> None:
    if not entries_dir.is_dir():
        raise SystemExit(f"entries directory does not exist: {entries_dir}")

    rows: list[dict[str, Any]] = []
    for path in sorted(entries_dir.rglob("*.json")):
        rows.extend(flatten_entry(path, entries_dir))

    rows.sort(key=lambda row: (row["source_repo"], row["variant_id"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "catalog_rows.jsonl"
    with rows_path.open("w") as file:
        for row in rows:
            file.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")

    (output_dir / "README.md").write_text(README)
    print(f"wrote {len(rows)} catalog rows to {rows_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entries-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    generate(args.entries_dir, args.output_dir)


if __name__ == "__main__":
    main()
