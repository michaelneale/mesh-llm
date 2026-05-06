#!/usr/bin/env python3
"""Cheap llama.cpp parity certification helper.

The script joins the pinned llama.cpp model implementation inventory with a
small GGUF candidate manifest. It resolves already-cached Hugging Face GGUFs,
prints missing download commands, and can run the existing family-certify
wrapper for locally available candidates.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "docs/skippy/llama-parity-candidates.json"
DEFAULT_UPSTREAM_PIN = ROOT / "third_party/llama.cpp/upstream.txt"


def repo_cache_dir(repo: str) -> Path:
    cache_root = os.environ.get("HF_HUB_CACHE")
    if cache_root:
        hub = Path(cache_root)
    elif os.environ.get("HF_HOME"):
        hub = Path(os.environ["HF_HOME"]) / "hub"
    else:
        hub = Path.home() / ".cache/huggingface/hub"
    return hub / ("models--" + repo.replace("/", "--"))


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run(args: list[str], *, cwd: Path | None = None, quiet: bool = False) -> str:
    proc = subprocess.run(
        args,
        cwd=str(cwd) if cwd else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        if quiet:
            return ""
        raise RuntimeError(
            f"command failed ({proc.returncode}): {' '.join(args)}\n{proc.stderr}"
        )
    return proc.stdout


def pinned_llama_models(llama_src: Path | None) -> list[str]:
    if llama_src:
        models_dir = llama_src / "src/models"
        if not models_dir.is_dir():
            raise SystemExit(f"llama source has no src/models: {llama_src}")
        return sorted(path.stem for path in models_dir.glob("*.cpp"))

    deps = ROOT / ".deps/llama.cpp/src/models"
    if deps.is_dir():
        return sorted(path.stem for path in deps.glob("*.cpp"))

    pin = DEFAULT_UPSTREAM_PIN.read_text(encoding="utf-8").strip()
    cache_root = Path(tempfile.mkdtemp(prefix="skippy-llama-pin."))
    try:
        run(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--no-checkout",
                "https://github.com/ggml-org/llama.cpp",
                str(cache_root),
            ],
            quiet=True,
        )
        run(["git", "fetch", "--depth", "1", "origin", pin], cwd=cache_root, quiet=True)
        listing = run(["git", "ls-tree", "-r", "--name-only", "FETCH_HEAD", "src/models"], cwd=cache_root)
    finally:
        shutil.rmtree(cache_root, ignore_errors=True)

    models = []
    for line in listing.splitlines():
        path = Path(line)
        if path.suffix == ".cpp" and path.parent.as_posix() == "src/models":
            models.append(path.stem)
    return sorted(models)


def candidate_index(manifest: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    index: dict[str, list[dict[str, Any]]] = {}
    for candidate in manifest.get("candidates", []):
        index.setdefault(candidate["llama_model"], []).append(candidate)
    return index


def resolve_candidate_file(candidate: dict[str, Any]) -> Path | None:
    repo = candidate.get("repo")
    include = candidate.get("include", "*.gguf")
    if not repo:
        return None
    includes = include if isinstance(include, list) else [include]
    base = repo_cache_dir(repo) / "snapshots"
    matches: list[Path] = []
    for pattern in includes:
        matches.extend(Path(path) for path in glob.glob(str(base / "*" / pattern), recursive=True))
    matches = [path for path in matches if path.exists()]
    if not matches:
        return None
    matches.sort(key=lambda path: (path.stat().st_size, str(path)))
    return matches[0]


def download_command(candidate: dict[str, Any]) -> str:
    repo = candidate.get("repo")
    include = candidate.get("include", "*.gguf")
    if not repo:
        return ""
    if isinstance(include, list):
        includes = " ".join(f"--include '{item}'" for item in include)
    else:
        includes = f"--include '{include}'"
    return f"hf download {repo} {includes}"


class GgufReader:
    def __init__(self, path: Path):
        self.handle = path.open("rb")

    def close(self) -> None:
        self.handle.close()

    def read(self, size: int) -> bytes:
        data = self.handle.read(size)
        if len(data) != size:
            raise EOFError("short GGUF read")
        return data

    def u32(self) -> int:
        return struct.unpack("<I", self.read(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.read(8))[0]

    def i32(self) -> int:
        return struct.unpack("<i", self.read(4))[0]

    def i64(self) -> int:
        return struct.unpack("<q", self.read(8))[0]

    def f32(self) -> float:
        return struct.unpack("<f", self.read(4))[0]

    def f64(self) -> float:
        return struct.unpack("<d", self.read(8))[0]

    def string(self) -> str:
        length = self.u64()
        return self.read(length).decode("utf-8", errors="replace")

    def value(self, typ: int) -> Any:
        if typ == 0:
            return self.read(1)[0]
        if typ == 1:
            return struct.unpack("<b", self.read(1))[0]
        if typ == 2:
            return struct.unpack("<H", self.read(2))[0]
        if typ == 3:
            return struct.unpack("<h", self.read(2))[0]
        if typ == 4:
            return self.u32()
        if typ == 5:
            return self.i32()
        if typ == 6:
            return self.f32()
        if typ == 7:
            return bool(self.read(1)[0])
        if typ == 8:
            return self.string()
        if typ == 9:
            item_type = self.u32()
            count = self.u64()
            return [self.value(item_type) for _ in range(count)]
        if typ == 10:
            return self.u64()
        if typ == 11:
            return self.i64()
        if typ == 12:
            return self.f64()
        raise ValueError(f"unsupported GGUF value type {typ}")


def gguf_metadata(path: Path) -> dict[str, Any]:
    reader = GgufReader(path)
    try:
        if reader.read(4) != b"GGUF":
            raise ValueError(f"not a GGUF file: {path}")
        version = reader.u32()
        if version < 2:
            raise ValueError(f"unsupported GGUF version {version}: {path}")
        tensor_count = reader.u64()
        kv_count = reader.u64()
        metadata: dict[str, Any] = {}
        for _ in range(kv_count):
            key = reader.string()
            typ = reader.u32()
            metadata[key] = reader.value(typ)
        metadata["_tensor_count"] = tensor_count
        return metadata
    finally:
        reader.close()


def infer_model_shape(path: Path) -> tuple[int, int, str | None]:
    metadata = gguf_metadata(path)
    arch = metadata.get("general.architecture")
    layer_count = None
    activation_width = None
    for key, value in metadata.items():
        if key.endswith(".block_count") and isinstance(value, int):
            layer_count = value
        if key.endswith(".embedding_length") and isinstance(value, int):
            activation_width = value
    if layer_count is None:
        raise ValueError(f"could not infer layer count from {path}")
    if activation_width is None:
        raise ValueError(f"could not infer embedding length from {path}")
    return layer_count, activation_width, arch if isinstance(arch, str) else None


def split_args(layer_count: int) -> tuple[int, str]:
    first = max(1, layer_count // 3)
    second = max(first + 1, (2 * layer_count) // 3)
    if second >= layer_count:
        second = layer_count - 1
    split_layer = max(1, layer_count // 2)
    if split_layer >= layer_count:
        split_layer = layer_count - 1
    return split_layer, f"{first},{second}"


def default_stage_build_dir() -> str | None:
    if os.environ.get("LLAMA_STAGE_BUILD_DIR"):
        return os.environ["LLAMA_STAGE_BUILD_DIR"]
    llama_root = ROOT / ".deps/llama.cpp"
    for name in (
        "build-stage-abi-metal",
        "build-stage-abi-static",
        "build-stage-abi-cuda",
        "build-stage-abi-vulkan",
        "build-stage-abi-rocm",
    ):
        candidate = llama_root / name
        if candidate.is_dir():
            return str(candidate)
    return None


def inventory(args: argparse.Namespace) -> list[dict[str, Any]]:
    manifest = load_json(args.manifest)
    candidates = candidate_index(manifest)
    rows = []
    for model in pinned_llama_models(args.llama_src):
        entries = candidates.get(model, [])
        if not entries:
            rows.append(
                {
                    "llama_model": model,
                    "family": model.replace("-", "_"),
                    "status": "missing_candidate",
                    "repo": None,
                    "local_path": None,
                    "download": "",
                }
            )
            continue
        for entry in entries:
            path = resolve_candidate_file(entry)
            row = {
                "llama_model": model,
                "family": entry.get("family", model.replace("-", "_")),
                "status": entry.get("status", "candidate"),
                "repo": entry.get("repo"),
                "include": entry.get("include"),
                "local_path": str(path) if path else None,
                "download": "" if path else download_command(entry),
                "notes": entry.get("notes", ""),
            }
            if path:
                try:
                    if path.suffix == ".gguf":
                        layer_count, activation_width, arch = infer_model_shape(path)
                        split_layer, splits = split_args(layer_count)
                        row.update(
                            {
                                "gguf_arch": arch,
                                "layer_end": layer_count,
                                "activation_width": activation_width,
                                "split_layer": split_layer,
                                "splits": splits,
                            }
                        )
                    else:
                        row["package_manifest"] = str(path)
                except Exception as exc:  # keep inventory useful for bad/corrupt downloads
                    row["inspect_error"] = str(exc)
            rows.append(row)
    return rows


def print_table(rows: list[dict[str, Any]]) -> None:
    print("| llama model | family | status | local | candidate/download |")
    print("| --- | --- | --- | --- | --- |")
    for row in rows:
        local = "yes" if row.get("local_path") else "no"
        target = row.get("repo") or row.get("download") or ""
        if row.get("download"):
            target = f"`{row['download']}`"
        print(
            f"| `{row['llama_model']}` | `{row['family']}` | {row['status']} | {local} | {target} |"
        )


def validate_inventory(rows: list[dict[str, Any]]) -> int:
    failures = 0
    missing = [row for row in rows if row.get("status") == "missing_candidate"]
    if missing:
        failures += len(missing)
        print("Missing parity manifest rows:", file=sys.stderr)
        for row in missing:
            print(f"  - {row['llama_model']}", file=sys.stderr)

    allowed_statuses = {
        "candidate",
        "candidate_stateful",
        "candidate_multimodal",
        "certified",
        "certified_package_only",
        "implementation_base",
        "needs_candidate",
        "needs_runtime_slice_support",
        "non_causal_aux",
        "package_or_remote_only",
    }
    unknown_statuses = [
        row
        for row in rows
        if row.get("status") not in allowed_statuses
        and row.get("status") != "missing_candidate"
    ]
    if unknown_statuses:
        failures += len(unknown_statuses)
        print("Unknown parity statuses:", file=sys.stderr)
        for row in unknown_statuses:
            print(
                f"  - {row['llama_model']}: {row.get('status')}",
                file=sys.stderr,
            )

    failures += validate_stage_abi_allowlist()

    return failures


def validate_stage_abi_allowlist() -> int:
    llama_src = ROOT / ".deps/llama.cpp/src"
    skippy_cpp = llama_src / "skippy.cpp"
    arch_cpp = llama_src / "llama-arch.cpp"
    models_dir = llama_src / "models"
    if not skippy_cpp.exists() or not arch_cpp.exists() or not models_dir.is_dir():
        return 0

    def normalized(name: str) -> str:
        return name.replace("_", "").replace("-", "")

    arch_names: dict[str, str] = {}
    for match in re.finditer(
        r'\{\s*LLM_ARCH_([A-Z0-9_]+),\s+"([^"]+)"\s*\}',
        arch_cpp.read_text(encoding="utf-8"),
    ):
        arch_names[match.group(1).lower()] = match.group(2)

    allowed = {
        arch_names.get(match.group(1).lower(), match.group(1).lower())
        for match in re.finditer(
            r"model->arch != LLM_ARCH_([A-Z0-9_]+)",
            skippy_cpp.read_text(encoding="utf-8"),
        )
    }
    if "gpt-oss" in allowed:
        allowed.add("openai-moe")

    stage_hooked = set()
    for path in models_dir.glob("*.cpp"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "skippy_graph_get_filter" in text and "stage_boundary" in text:
            stage_hooked.add(path.stem)

    # llama.cpp dispatches these architectures through another staged graph, or
    # uses a model file name that differs from the architecture string.
    stage_hooked.update(
        {
            "gpt-oss",
            "mamba2",
            "granite_moe",
            "hunyuan_dense",
            "phimoe",
            "glm-dsa",
            "lfm2moe",
            "minicpm",
            "nemotron_h_moe",
        }
    )

    allowed_by_norm = {normalized(name): name for name in allowed}
    hooked_by_norm = {normalized(name): name for name in stage_hooked}
    allowed_without_hook = sorted(set(allowed_by_norm) - set(hooked_by_norm))
    hooked_but_not_allowed = sorted(set(hooked_by_norm) - set(allowed_by_norm))

    failures = 0
    if allowed_without_hook:
        failures += len(allowed_without_hook)
        print(
            "Stage ABI allowlist contains architectures without detected staged graph support:",
            file=sys.stderr,
        )
        for key in allowed_without_hook:
            print(f"  - {allowed_by_norm[key]}", file=sys.stderr)
    if hooked_but_not_allowed:
        failures += len(hooked_but_not_allowed)
        print(
            "Staged graph implementations are missing from the stage ABI allowlist:",
            file=sys.stderr,
        )
        for key in hooked_but_not_allowed:
            print(f"  - {hooked_by_norm[key]}", file=sys.stderr)

    return failures


def run_certifications(args: argparse.Namespace, rows: list[dict[str, Any]]) -> int:
    defaults = load_json(args.manifest).get("defaults", {})
    statuses = set(args.status) if args.status else {
        "candidate",
        "candidate_stateful",
        "candidate_multimodal",
        "certified",
    }
    failures = 0
    selected = [
        row
        for row in rows
        if row.get("local_path")
        and row.get("layer_end")
        and row.get("activation_width")
        and row.get("status") in statuses
        and (not args.family or row.get("family") in args.family)
        and (not args.llama_model or row.get("llama_model") in args.llama_model)
    ]
    if args.limit:
        selected = selected[: args.limit]
    for row in selected:
        stage_build_dir = default_stage_build_dir()
        cmd = [
            str(ROOT / "scripts/family-certify.sh"),
            "--family",
            row["family"],
            "--target-model",
            row["local_path"],
            "--model-id",
            row.get("repo") or row["family"],
            "--layer-end",
            str(row["layer_end"]),
            "--split-layer",
            str(row["split_layer"]),
            "--splits",
            str(row["splits"]),
            "--activation-width",
            str(row["activation_width"]),
            "--ctx-size",
            str(args.ctx_size or defaults.get("ctx_size", 128)),
            "--n-gpu-layers",
            str(args.n_gpu_layers or defaults.get("n_gpu_layers", 999)),
            "--wire-dtype",
            str(defaults.get("wire_dtype", "f16")),
            "--wire-dtypes",
            str(defaults.get("wire_dtypes", "f32,f16,q8")),
            "--prompt",
            str(defaults.get("prompt", "Hello")),
            "--run-id",
            args.run_id,
        ]
        entry = next(
            (
                candidate
                for candidate in load_json(args.manifest).get("candidates", [])
                if candidate.get("family") == row["family"]
                and candidate.get("llama_model") == row["llama_model"]
            ),
            {},
        )
        if entry.get("recurrent") == "all":
            cmd.append("--recurrent-all")
        if args.skip_build:
            cmd.append("--skip-build")
        if args.skip_state:
            cmd.append("--skip-state")
        if args.skip_dtype:
            cmd.append("--skip-dtype")
        if args.state_payload_kind:
            cmd.extend(["--state-payload-kind", args.state_payload_kind])
        if args.prefix_token_count:
            cmd.extend(["--prefix-token-count", str(args.prefix_token_count)])
        if args.cache_hit_repeats:
            cmd.extend(["--cache-hit-repeats", str(args.cache_hit_repeats)])
        if args.borrow_resident_hits:
            cmd.append("--borrow-resident-hits")
        if args.cache_decoded_result_hits:
            cmd.append("--cache-decoded-result-hits")
        if args.dry_run:
            prefix = f"LLAMA_STAGE_BUILD_DIR={stage_build_dir} " if stage_build_dir else ""
            print(prefix + " ".join(cmd))
            continue
        print(f"==> certifying {row['family']} ({row['llama_model']})")
        env = os.environ.copy()
        if stage_build_dir:
            env["LLAMA_STAGE_BUILD_DIR"] = stage_build_dir
        proc = subprocess.run(cmd, cwd=str(ROOT), env=env, check=False)
        if proc.returncode != 0:
            failures += 1
            if args.stop_on_failure:
                return failures
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--llama-src", type=Path)
    sub = parser.add_subparsers(dest="command", required=True)

    inv = sub.add_parser("inventory", help="print joined llama/candidate inventory")
    inv.add_argument("--json", action="store_true")
    inv.add_argument("--missing-only", action="store_true")
    inv.add_argument("--local-only", action="store_true")

    commands = sub.add_parser("download-commands", help="print hf download commands for missing candidates")
    commands.add_argument("--all", action="store_true", help="include non-candidate/package-only rows")

    sub.add_parser("validate", help="fail if the pinned llama.cpp inventory is not fully classified")

    run_parser = sub.add_parser("run", help="run family-certify for local candidates")
    run_parser.add_argument("--status", action="append")
    run_parser.add_argument("--family", action="append")
    run_parser.add_argument("--llama-model", action="append")
    run_parser.add_argument("--limit", type=int)
    run_parser.add_argument("--dry-run", action="store_true")
    run_parser.add_argument("--skip-build", action="store_true")
    run_parser.add_argument("--skip-state", action="store_true")
    run_parser.add_argument("--skip-dtype", action="store_true")
    run_parser.add_argument("--state-payload-kind")
    run_parser.add_argument("--prefix-token-count", type=int)
    run_parser.add_argument("--cache-hit-repeats", type=int)
    run_parser.add_argument("--borrow-resident-hits", action="store_true")
    run_parser.add_argument("--cache-decoded-result-hits", action="store_true")
    run_parser.add_argument("--stop-on-failure", action="store_true")
    run_parser.add_argument("--ctx-size", type=int)
    run_parser.add_argument("--n-gpu-layers", type=int)
    run_parser.add_argument("--run-id", default="llama-parity-cheap")

    args = parser.parse_args()
    rows = inventory(args)
    if args.command == "inventory":
        if args.missing_only:
            rows = [row for row in rows if not row.get("local_path")]
        if args.local_only:
            rows = [row for row in rows if row.get("local_path")]
        if args.json:
            print(json.dumps(rows, indent=2, sort_keys=True))
        else:
            print_table(rows)
        return 0
    if args.command == "download-commands":
        statuses = {
            "candidate",
            "candidate_stateful",
            "candidate_multimodal",
            "certified",
        }
        seen = set()
        for row in rows:
            command = row.get("download")
            if not command or command in seen:
                continue
            if not args.all and row.get("status") not in statuses:
                continue
            seen.add(command)
            print(command)
        return 0
    if args.command == "validate":
        return 1 if validate_inventory(rows) else 0
    if args.command == "run":
        return 1 if run_certifications(args, rows) else 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
