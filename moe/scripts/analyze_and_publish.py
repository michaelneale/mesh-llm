#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "huggingface-hub>=0.33.0",
# ]
# ///

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import tarfile
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

try:
    from huggingface_hub import HfApi, hf_hub_download
except ModuleNotFoundError:  # pragma: no cover - depends on caller environment
    HfApi = None  # type: ignore[assignment]
    hf_hub_download = None  # type: ignore[assignment]


DEFAULT_MICRO_PROMPTS = [
    "Write a concise explanation of how a rainbow forms.",
    "Summarize the causes and effects of inflation in a paragraph.",
    "Explain why distributed systems are hard to debug.",
    "Give three practical tips for writing reliable shell scripts.",
    "Describe the water cycle for a middle school student.",
    "Compare TCP and QUIC in two short paragraphs.",
    "Explain the difference between RAM and disk storage.",
    "Write a short answer on why model evaluation matters.",
]

SHARD_SUFFIX_RE = re.compile(r"-(\d{5})-of-(\d{5})$")
DEFAULT_RELEASE_REPO = "michaelneale/mesh-llm"
RELEASE_TARGET_CHOICES = ["auto", "cpu", "cuda", "rocm", "vulkan", "metal"]


@dataclass(frozen=True)
class Distribution:
    source_repo: str
    source_revision: str
    requested_revision: str
    format: str
    distribution_id: str
    primary_file: str
    files: list[str]


@dataclass(frozen=True)
class AnalyzerBinary:
    path: Path
    source: str
    release_repo: str | None = None
    release_tag: str | None = None
    release_asset: str | None = None
    release_target: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a GGUF distribution from Hugging Face, run llama-moe-analyze, and publish artifacts."
    )
    parser.add_argument("--source-repo", required=True, help="Model repo, e.g. unsloth/GLM-5.1-GGUF")
    parser.add_argument(
        "--source-revision",
        default="main",
        help="Model revision to resolve. The exact commit SHA is recorded in metadata.",
    )
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument(
        "--filename",
        help="A GGUF file path inside the model repo. For sharded models, any shard in the distribution is acceptable.",
    )
    selector.add_argument(
        "--distribution-id",
        help="Normalized distribution id, e.g. GLM-5.1-UD-IQ2_M",
    )
    parser.add_argument(
        "--analyzer-source",
        choices=["local", "release"],
        default="local",
        help="Use a local llama-moe-analyze binary or bootstrap one from GitHub releases.",
    )
    parser.add_argument(
        "--analyzer-bin",
        help="Path to a local llama-moe-analyze binary. Required when --analyzer-source=local.",
    )
    parser.add_argument(
        "--analyzer-id",
        default="micro-v1",
        choices=["micro-v1", "full-v1"],
        help="Analysis method id and version.",
    )
    parser.add_argument(
        "--token-count",
        type=int,
        default=128,
        help="Token budget for micro analysis.",
    )
    parser.add_argument(
        "--context-size",
        type=int,
        default=4096,
        help="Context window passed to llama-moe-analyze.",
    )
    parser.add_argument(
        "--n-gpu-layers",
        type=int,
        default=0,
        help="Number of layers to offload to GPU. Use 0 for CPU-only runs.",
    )
    parser.add_argument(
        "--all-layers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to analyze all layers. Enabled by default.",
    )
    parser.add_argument(
        "--prompt-file",
        help="Reserved for future non-canonical analyzer ids. Not supported by the built-in analyzer ids.",
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=Path.cwd() / ".moe-cache" / "models",
        help="Local directory for downloaded model shards.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd() / ".moe-cache" / "artifacts",
        help="Local directory where dataset-layout artifacts are written before upload.",
    )
    parser.add_argument(
        "--tool-cache-dir",
        type=Path,
        default=Path.cwd() / ".moe-cache" / "tools",
        help="Local cache for bootstrapped analyzer binaries.",
    )
    parser.add_argument(
        "--release-repo",
        default=DEFAULT_RELEASE_REPO,
        help="GitHub repo used when --analyzer-source=release, e.g. michaelneale/mesh-llm.",
    )
    parser.add_argument(
        "--release-tag",
        default="latest",
        help="Release tag to download when --analyzer-source=release. Use `latest` for the latest release.",
    )
    parser.add_argument(
        "--release-target",
        choices=RELEASE_TARGET_CHOICES,
        default="auto",
        help="Release bundle target to use when --analyzer-source=release. Use `auto` to infer from the runtime platform.",
    )
    parser.add_argument(
        "--dataset-repo",
        help="Optional destination dataset repo for artifact upload.",
    )
    parser.add_argument(
        "--dataset-revision",
        default="main",
        help="Dataset repo branch or revision to upload to.",
    )
    parser.add_argument(
        "--dataset-private",
        action="store_true",
        help="Create the dataset repo as private if it does not exist.",
    )
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Generate artifacts locally only.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse an existing local artifact if metadata.json and ranking.csv are already present.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve inputs and print planned work without downloading or analyzing.",
    )
    args = parser.parse_args()
    if args.analyzer_source == "local" and not args.analyzer_bin:
        parser.error("--analyzer-bin is required when --analyzer-source=local")
    if args.analyzer_source == "release" and args.analyzer_bin:
        parser.error("--analyzer-bin cannot be combined with --analyzer-source=release")
    if args.prompt_file:
        parser.error(
            "--prompt-file is not supported by the built-in analyzer ids; add a new analyzer id for a custom prompt set"
        )
    if args.n_gpu_layers < 0:
        parser.error("--n-gpu-layers must be >= 0")
    if args.analyzer_source == "local" and args.release_target != "auto":
        parser.error("--release-target is only valid when --analyzer-source=release")
    return args


def require_hf_dependencies() -> None:
    if HfApi is None or hf_hub_download is None:
        raise SystemExit(
            "Missing dependency: huggingface_hub\n"
            "Run this script with `uv run moe/scripts/analyze_and_publish.py ...` "
            "or install `huggingface-hub` in your Python environment."
        )


def normalize_distribution_id(path: str) -> str:
    stem = Path(path).name
    if stem.endswith(".gguf"):
        stem = stem[:-5]
    return SHARD_SUFFIX_RE.sub("", stem)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def load_prompts(prompt_file: str | None) -> list[str]:
    if not prompt_file:
        return list(DEFAULT_MICRO_PROMPTS)
    prompts = [line.strip() for line in Path(prompt_file).read_text().splitlines()]
    prompts = [prompt for prompt in prompts if prompt]
    if not prompts:
        raise SystemExit(f"No prompts found in {prompt_file}")
    return prompts


def resolve_release_target(release_target: str) -> tuple[str, str]:
    if release_target == "auto":
        if sys.platform == "darwin" and platform.machine() == "arm64":
            return "metal", "aarch64-apple-darwin"
        if sys.platform.startswith("linux") and platform.machine() in {"x86_64", "amd64"}:
            return "cpu", "x86_64-unknown-linux-gnu"
        raise SystemExit(
            f"Unsupported platform for release bootstrap: platform={sys.platform} arch={platform.machine()}"
        )

    if release_target == "metal":
        if not (sys.platform == "darwin" and platform.machine() == "arm64"):
            raise SystemExit("--release-target=metal requires macOS arm64")
        return "metal", "aarch64-apple-darwin"

    if release_target in {"cpu", "cuda", "rocm", "vulkan"}:
        if not (sys.platform.startswith("linux") and platform.machine() in {"x86_64", "amd64"}):
            raise SystemExit(f"--release-target={release_target} requires Linux x86_64")
        return release_target, "x86_64-unknown-linux-gnu"

    raise SystemExit(f"Unsupported --release-target={release_target}")


def release_asset_name(release_tag: str, release_target: str) -> tuple[str, str]:
    resolved_target, target_triple = resolve_release_target(release_target)
    target_suffix = target_triple if resolved_target in {"cpu", "metal"} else f"{target_triple}-{resolved_target}"
    if release_tag == "latest":
        return f"mesh-llm-{target_suffix}.tar.gz", resolved_target
    return f"mesh-llm-{release_tag}-{target_suffix}.tar.gz", resolved_target


def release_download_url(release_repo: str, release_tag: str, asset_name: str) -> str:
    if release_tag == "latest":
        return f"https://github.com/{release_repo}/releases/latest/download/{asset_name}"
    return f"https://github.com/{release_repo}/releases/download/{release_tag}/{asset_name}"


def download_release_asset(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=120) as response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def extract_release_binary(archive_path: Path, extract_dir: Path, binary_name: str) -> Path:
    final_path = extract_dir / binary_name
    final_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = extract_dir / f".{binary_name}.tmp"
    with tarfile.open(archive_path, "r:gz") as tar:
        members = tar.getmembers()
        matches = [member for member in members if Path(member.name).name == binary_name]
        if not matches:
            raise SystemExit(f"{binary_name} not found in release archive {archive_path}")
        member = matches[0]
        if not member.isreg():
            raise SystemExit(f"Refusing to extract non-regular file from archive: {member.name}")
        source = tar.extractfile(member)
        if source is None:
            raise SystemExit(f"Failed to read {member.name} from release archive {archive_path}")
        try:
            with source, tmp_path.open("wb") as handle:
                shutil.copyfileobj(source, handle)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise
    tmp_path.chmod(0o755)
    tmp_path.replace(final_path)
    return final_path


def resolve_analyzer_binary(args: argparse.Namespace) -> AnalyzerBinary:
    if args.analyzer_source == "local":
        path = Path(args.analyzer_bin).expanduser().resolve()
        if not path.exists():
            raise SystemExit(f"Analyzer binary not found: {path}")
        return AnalyzerBinary(path=path, source="local")

    asset_name, resolved_release_target = release_asset_name(args.release_tag, args.release_target)
    cache_root = args.tool_cache_dir / "github-releases" / args.release_repo / args.release_tag / resolved_release_target
    archive_path = cache_root / asset_name
    binary_path = cache_root / "bin" / "llama-moe-analyze"
    if not binary_path.exists():
        url = release_download_url(args.release_repo, args.release_tag, asset_name)
        download_release_asset(url, archive_path)
        binary_path = extract_release_binary(archive_path, cache_root / "bin", "llama-moe-analyze")
    return AnalyzerBinary(
        path=binary_path.resolve(),
        source="release",
        release_repo=args.release_repo,
        release_tag=args.release_tag,
        release_asset=asset_name,
        release_target=resolved_release_target,
    )


def resolve_distribution(
    api: HfApi,
    source_repo: str,
    requested_revision: str,
    filename: str | None,
    distribution_id: str | None,
) -> Distribution:
    info = api.model_info(source_repo, revision=requested_revision, files_metadata=True)
    siblings = [s.rfilename for s in info.siblings or []]
    ggufs = [name for name in siblings if name.endswith(".gguf")]
    if not ggufs:
        raise SystemExit(f"No GGUF files found in {source_repo}@{requested_revision}")

    if filename:
        if filename not in ggufs:
            raise SystemExit(f"GGUF file not found in repo: {filename}")
        dist_id = normalize_distribution_id(filename)
        primary_file = filename
    else:
        assert distribution_id is not None
        matching = [name for name in ggufs if normalize_distribution_id(name) == distribution_id]
        if not matching:
            raise SystemExit(
                f"No GGUF distribution matched {distribution_id} in {source_repo}@{requested_revision}"
            )
        matching.sort()
        dist_id = distribution_id
        primary_file = matching[0]

    files = [name for name in ggufs if normalize_distribution_id(name) == dist_id]
    files.sort()
    if not files:
        raise SystemExit(f"No GGUF files resolved for distribution {dist_id}")

    return Distribution(
        source_repo=source_repo,
        source_revision=info.sha,
        requested_revision=requested_revision,
        format="gguf",
        distribution_id=dist_id,
        primary_file=primary_file,
        files=files,
    )


def download_distribution(
    distribution: Distribution,
    download_dir: Path,
) -> list[Path]:
    local_files: list[Path] = []
    for repo_file in distribution.files:
        local_path = Path(
            hf_hub_download(
                repo_id=distribution.source_repo,
                filename=repo_file,
                revision=distribution.source_revision,
                local_dir=str(download_dir),
            )
        )
        local_files.append(local_path)
    return local_files


def artifact_root(output_dir: Path, distribution: Distribution, analyzer_id: str) -> Path:
    namespace, repo_name = distribution.source_repo.split("/", 1)
    return (
        output_dir
        / "data"
        / namespace
        / repo_name
        / distribution.source_revision
        / distribution.format
        / distribution.distribution_id
        / analyzer_id
    )


def relative_artifact_prefix(distribution: Distribution, analyzer_id: str) -> str:
    namespace, repo_name = distribution.source_repo.split("/", 1)
    return "/".join(
        [
            "data",
            namespace,
            repo_name,
            distribution.source_revision,
            distribution.format,
            distribution.distribution_id,
            analyzer_id,
        ]
    )


def build_command(
    analyzer_bin: Path,
    args: argparse.Namespace,
    analyzer_output: Path,
    model_path: Path,
    prompt: str | None,
) -> list[str]:
    command = [
        str(analyzer_bin),
        "-m",
        str(model_path),
        "--export-ranking",
        str(analyzer_output),
        "-c",
        str(args.context_size),
        "-ngl",
        str(args.n_gpu_layers),
    ]
    if args.all_layers:
        command.append("--all-layers")
    if args.analyzer_id == "micro-v1":
        command.extend(["-n", str(args.token_count)])
        if prompt is None:
            raise ValueError("micro-v1 requires a prompt")
        command.extend(["-p", prompt])
    elif args.analyzer_id == "full-v1":
        command.extend(["-n", "32"])
    else:
        raise ValueError(f"Unsupported analyzer_id: {args.analyzer_id}")
    return command


def read_ranking_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = [
            row
            for row in csv.reader(handle)
            if row and row[0].strip() and not row[0].lstrip().startswith("#")
        ]

    if not rows:
        raise SystemExit(f"Ranking output was empty: {path}")

    first = [cell.strip() for cell in rows[0]]
    if first and first[0] == "expert_id":
        header = first
        data_rows = rows[1:]
    else:
        header = ["expert_id", "gate_mass", "mass_pct", "selection_count"]
        data_rows = rows

    normalized: list[dict[str, str]] = []
    for row in data_rows:
        values = [cell.strip() for cell in row]
        if len(values) < len(header):
            raise SystemExit(f"Ranking CSV row had too few columns in {path}: {row}")
        raw = dict(zip(header, values, strict=False))
        normalized.append(
            {
                "expert_id": raw["expert_id"],
                "total_mass": raw.get("total_mass", raw.get("gate_mass", "")),
                "mass_fraction": raw.get("mass_fraction", raw.get("mass_pct", "")),
                "selection_count": raw["selection_count"],
            }
        )

    required = {"expert_id", "total_mass", "mass_fraction", "selection_count"}
    for row in normalized:
        missing = [key for key in required if not row.get(key)]
        if missing:
            raise SystemExit(f"Ranking CSV missing expected columns {missing} in {path}")

    return normalized


def combine_rankings(per_prompt_paths: Iterable[Path], output_path: Path) -> None:
    totals: dict[str, dict[str, float | int]] = {}
    for path in per_prompt_paths:
        for row in read_ranking_rows(path):
            expert_id = row["expert_id"]
            entry = totals.setdefault(
                expert_id,
                {"total_mass": 0.0, "selection_count": 0},
            )
            entry["total_mass"] += float(row["total_mass"])
            entry["selection_count"] += int(row["selection_count"])

    ordered = sorted(
        totals.items(),
        key=lambda item: (-float(item[1]["total_mass"]), int(item[0])),
    )
    total_mass_sum = sum(float(values["total_mass"]) for _, values in ordered)
    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["expert_id", "total_mass", "mass_fraction", "selection_count"])
        for expert_id, values in ordered:
            total_mass = float(values["total_mass"])
            mass_fraction = (total_mass / total_mass_sum) if total_mass_sum else 0.0
            writer.writerow(
                [
                    expert_id,
                    f"{total_mass:.12f}",
                    f"{mass_fraction:.12f}",
                    int(values["selection_count"]),
                ]
            )


def run_analysis(
    analyzer: AnalyzerBinary,
    args: argparse.Namespace,
    distribution: Distribution,
    local_files: list[Path],
    artifact_dir: Path,
) -> tuple[Path, Path]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    ranking_path = artifact_dir / "ranking.csv"
    log_path = artifact_dir / "run.log"
    primary_local = next(path for path in local_files if path.as_posix().endswith(distribution.primary_file))

    if args.analyzer_id == "full-v1":
        command = build_command(analyzer.path, args, ranking_path, primary_local, None)
        proc = subprocess.run(command, capture_output=True, text=True, check=False)
        log_path.write_text(
            "".join(
                [
                    "$ ",
                    shlex.join(command),
                    "\n\n[stdout]\n",
                    proc.stdout,
                    "\n[stderr]\n",
                    proc.stderr,
                ]
            )
        )
        if proc.returncode != 0:
            raise SystemExit(
                f"llama-moe-analyze failed with exit code {proc.returncode} "
                f"(see {log_path} for details)"
            )
    else:
        prompts = load_prompts(args.prompt_file)
        temp_outputs: list[Path] = []
        with tempfile.TemporaryDirectory(prefix="moe-analyze-") as temp_dir:
            temp_root = Path(temp_dir)
            log_chunks: list[str] = []
            for idx, prompt in enumerate(prompts, start=1):
                partial = temp_root / f"prompt-{idx}.csv"
                command = build_command(analyzer.path, args, partial, primary_local, prompt)
                proc = subprocess.run(command, capture_output=True, text=True, check=False)
                log_chunks.append(
                    "".join(
                        [
                            f"$ {shlex.join(command)}\n",
                            f"[prompt {idx}]\n{prompt}\n\n",
                            "[stdout]\n",
                            proc.stdout,
                            "\n[stderr]\n",
                            proc.stderr,
                            "\n",
                        ]
                    )
                )
                if proc.returncode != 0:
                    log_path.write_text("\n".join(log_chunks))
                    raise SystemExit(
                        f"llama-moe-analyze failed for prompt {idx} with exit code {proc.returncode} "
                        f"(see {log_path} for details)"
                    )
                temp_outputs.append(partial)
            combine_rankings(temp_outputs, ranking_path)
            log_path.write_text("\n".join(log_chunks))

    return ranking_path, log_path


def write_metadata(
    analyzer: AnalyzerBinary,
    args: argparse.Namespace,
    distribution: Distribution,
    local_files: list[Path],
    artifact_dir: Path,
    ranking_path: Path,
) -> Path:
    metadata = {
        "schema_version": 1,
        "source_repo": distribution.source_repo,
        "source_revision": distribution.source_revision,
        "requested_revision": distribution.requested_revision,
        "format": distribution.format,
        "distribution_id": distribution.distribution_id,
        "analyzer_id": args.analyzer_id,
        "analysis_tool": "llama-moe-analyze",
        "analyzer_source": analyzer.source,
        "analyzer_release_repo": analyzer.release_repo,
        "analyzer_release_tag": analyzer.release_tag,
        "analyzer_release_asset": analyzer.release_asset,
        "analyzer_release_target": analyzer.release_target,
        "ranking_path": "ranking.csv",
        "primary_file": distribution.primary_file,
        "all_files": distribution.files,
        "file_hashes": {
            repo_file: sha256_file(local_path)
            for repo_file, local_path in zip(distribution.files, local_files, strict=True)
        },
        "llama_cpp_commit": os.environ.get("LLAMA_CPP_COMMIT"),
        "prompt_set": "meshllm-micro-v1" if args.analyzer_id == "micro-v1" else None,
        "prompt_file": args.prompt_file,
        "prompt_count": len(load_prompts(args.prompt_file)) if args.analyzer_id == "micro-v1" else None,
        "token_count": args.token_count if args.analyzer_id == "micro-v1" else 32,
        "all_layers": args.all_layers,
        "command": {
            "analyzer_bin": str(analyzer.path),
            "context_size": args.context_size,
            "token_count": args.token_count if args.analyzer_id == "micro-v1" else 32,
            "n_gpu_layers": args.n_gpu_layers,
            "analyzer_id": args.analyzer_id,
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
    }
    metadata_path = artifact_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    if not ranking_path.exists():
        raise SystemExit("ranking.csv was not created")
    return metadata_path


def upload_artifacts(
    api: HfApi,
    args: argparse.Namespace,
    distribution: Distribution,
    artifact_dir: Path,
) -> None:
    if not args.dataset_repo or args.skip_upload:
        return
    api.create_repo(
        repo_id=args.dataset_repo,
        repo_type="dataset",
        private=args.dataset_private,
        exist_ok=True,
    )
    api.upload_folder(
        repo_id=args.dataset_repo,
        repo_type="dataset",
        folder_path=str(artifact_dir),
        path_in_repo=relative_artifact_prefix(distribution, args.analyzer_id),
        revision=args.dataset_revision,
        commit_message=(
            f"Add {distribution.distribution_id} {args.analyzer_id} "
            f"for {distribution.source_repo}@{distribution.source_revision[:12]}"
        ),
    )


def remote_artifact_exists(
    api: HfApi,
    dataset_repo: str,
    dataset_revision: str,
    distribution: Distribution,
    analyzer_id: str,
) -> bool:
    prefix = relative_artifact_prefix(distribution, analyzer_id).rstrip("/")
    try:
        repo_tree = api.list_repo_tree(
            repo_id=dataset_repo,
            repo_type="dataset",
            revision=dataset_revision,
            path_in_repo=prefix,
            recursive=False,
        )
        return next(iter(repo_tree), None) is not None
    except Exception as exc:  # pragma: no cover - depends on remote state
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
        if status_code == 404:
            return False
        raise SystemExit(
            f"Failed to inspect dataset repo {dataset_repo}@{dataset_revision}: {exc}"
        ) from exc


def local_artifact_exists(artifact_dir: Path) -> bool:
    return (artifact_dir / "metadata.json").exists() and (artifact_dir / "ranking.csv").exists()


def main() -> int:
    args = parse_args()
    require_hf_dependencies()
    api = HfApi()
    analyzer = resolve_analyzer_binary(args)
    distribution = resolve_distribution(
        api=api,
        source_repo=args.source_repo,
        requested_revision=args.source_revision,
        filename=args.filename,
        distribution_id=args.distribution_id,
    )
    artifact_dir = artifact_root(args.output_dir, distribution, args.analyzer_id)

    print(
        json.dumps(
            {
                "source_repo": distribution.source_repo,
                "requested_revision": distribution.requested_revision,
                "source_revision": distribution.source_revision,
                "distribution_id": distribution.distribution_id,
                "files": distribution.files,
                "analyzer_source": analyzer.source,
                "analyzer_bin": str(analyzer.path),
                "analyzer_release_repo": analyzer.release_repo,
                "analyzer_release_tag": analyzer.release_tag,
                "analyzer_release_asset": analyzer.release_asset,
                "analyzer_id": args.analyzer_id,
                "artifact_dir": str(artifact_dir),
                "dataset_repo": args.dataset_repo,
            },
            indent=2,
        )
    )

    if args.dataset_repo and not args.skip_upload:
        if remote_artifact_exists(
            api=api,
            dataset_repo=args.dataset_repo,
            dataset_revision=args.dataset_revision,
            distribution=distribution,
            analyzer_id=args.analyzer_id,
        ):
            print(
                "Remote artifact already exists for "
                f"{distribution.source_repo}@{distribution.source_revision} "
                f"{distribution.distribution_id} {args.analyzer_id}"
            )
            return 0

    if args.dry_run:
        return 0

    if args.skip_existing and local_artifact_exists(artifact_dir):
        print(f"Reusing existing local artifact: {artifact_dir}")
        metadata_path = artifact_dir / "metadata.json"
        ranking_path = artifact_dir / "ranking.csv"
    else:
        local_files = download_distribution(distribution, args.download_dir)
        ranking_path, _ = run_analysis(analyzer, args, distribution, local_files, artifact_dir)
        metadata_path = write_metadata(analyzer, args, distribution, local_files, artifact_dir, ranking_path)

    upload_artifacts(api, args, distribution, artifact_dir)

    print(f"Wrote metadata: {metadata_path}")
    print(f"Wrote ranking: {ranking_path}")
    if args.dataset_repo and not args.skip_upload:
        print(f"Uploaded to dataset repo: {args.dataset_repo}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
