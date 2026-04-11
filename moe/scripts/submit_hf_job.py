#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "huggingface-hub>=0.33.0",
# ]
# ///

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path

try:
    from huggingface_hub import HfApi, get_token
except ModuleNotFoundError:  # pragma: no cover - depends on caller environment
    HfApi = None  # type: ignore[assignment]
    get_token = None  # type: ignore[assignment]


DEFAULT_RELEASE_REPO = "michaelneale/mesh-llm"
DEFAULT_JOB_FLAVOR = "cpu-xl"
DEFAULT_TIMEOUT = "1h"
RELEASE_TARGET_CHOICES = ["auto", "cpu", "cuda", "rocm", "vulkan", "metal"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit a Hugging Face Job that runs the canonical MoE analyze-and-publish workflow."
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
        help="Number of layers to offload to GPU inside the remote job. Use 0 for CPU-only runs.",
    )
    parser.add_argument(
        "--all-layers",
        action="store_true",
        default=True,
        help="Analyze all layers. Enabled by default.",
    )
    parser.add_argument(
        "--no-all-layers",
        dest="all_layers",
        action="store_false",
        help="Disable --all-layers for micro analysis.",
    )
    parser.add_argument(
        "--release-repo",
        default=DEFAULT_RELEASE_REPO,
        help="GitHub repo used to bootstrap llama-moe-analyze inside the job.",
    )
    parser.add_argument(
        "--release-tag",
        default="latest",
        help="Release tag to download inside the job. Use `latest` for the latest GitHub release.",
    )
    parser.add_argument(
        "--release-target",
        choices=RELEASE_TARGET_CHOICES,
        default="auto",
        help="Release bundle target to use inside the job. Use `cuda` for HF GPU jobs that should use the CUDA bundle.",
    )
    parser.add_argument(
        "--dataset-repo",
        required=True,
        help="Destination dataset repo for artifact upload, e.g. meshllm/moe-rankings.",
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
        "--skip-existing",
        action="store_true",
        help="Reuse an existing local artifact inside the job if metadata.json and ranking.csv are already present.",
    )
    parser.add_argument(
        "--flavor",
        default=DEFAULT_JOB_FLAVOR,
        help="HF Jobs hardware flavor, e.g. cpu-xl, cpu-performance, a10g-large.",
    )
    parser.add_argument(
        "--timeout",
        default=DEFAULT_TIMEOUT,
        help="HF Jobs timeout, e.g. 30m, 4h, 12h.",
    )
    parser.add_argument(
        "--namespace",
        help="Optional HF namespace that owns the job.",
    )
    parser.add_argument(
        "--image",
        help="Optional custom container image for the job runtime.",
    )
    parser.add_argument(
        "--python-version",
        default="3.11",
        help="Python version used by the UV job runtime.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the job configuration without submitting it.",
    )
    return parser.parse_args()


def require_hf_dependencies() -> None:
    if HfApi is None or get_token is None:
        raise SystemExit(
            "Missing dependency: huggingface_hub\n"
            "Run this script with `uv run moe/scripts/submit_hf_job.py ...` "
            "or install `huggingface-hub` in your Python environment."
        )


def require_hf_token() -> str:
    token = get_token()
    if not token:
        raise SystemExit("No Hugging Face token found. Run `hf auth login` before submitting a job.")
    return token


def build_script_args(args: argparse.Namespace) -> list[str]:
    script_args = [
        "--source-repo",
        args.source_repo,
        "--source-revision",
        args.source_revision,
        "--analyzer-source",
        "release",
        "--release-repo",
        args.release_repo,
        "--release-tag",
        args.release_tag,
        "--release-target",
        args.release_target,
        "--analyzer-id",
        args.analyzer_id,
        "--token-count",
        str(args.token_count),
        "--context-size",
        str(args.context_size),
        "--n-gpu-layers",
        str(args.n_gpu_layers),
        "--dataset-repo",
        args.dataset_repo,
        "--dataset-revision",
        args.dataset_revision,
    ]
    if args.filename:
        script_args.extend(["--filename", args.filename])
    if args.distribution_id:
        script_args.extend(["--distribution-id", args.distribution_id])
    if args.all_layers:
        script_args.append("--all-layers")
    else:
        script_args.append("--no-all-layers")
    if args.dataset_private:
        script_args.append("--dataset-private")
    if args.skip_existing:
        script_args.append("--skip-existing")
    return script_args


def build_labels(args: argparse.Namespace) -> dict[str, str]:
    labels = {
        "app": "mesh-llm",
        "workflow": "moe-analyze",
        "analyzer_id": args.analyzer_id,
        "source_repo": args.source_repo,
        "dataset_repo": args.dataset_repo,
    }
    if args.distribution_id:
        labels["distribution_id"] = args.distribution_id
    if args.filename:
        labels["filename"] = args.filename
    return labels


def main() -> int:
    args = parse_args()

    script_path = Path(__file__).with_name("analyze_and_publish.py")
    script_text = script_path.read_text()
    script_args = build_script_args(args)
    labels = build_labels(args)
    job_request = {
        "script_path": str(script_path),
        "script_args": script_args,
        "flavor": args.flavor,
        "timeout": args.timeout,
        "namespace": args.namespace,
        "image": args.image,
        "python_version": args.python_version,
        "labels": labels,
    }

    if args.dry_run:
        print(json.dumps(job_request, indent=2, sort_keys=True))
        print("\nCommand:")
        print(" ".join(shlex.quote(part) for part in ["uv", "run", str(script_path), *script_args]))
        return 0

    require_hf_dependencies()
    token = require_hf_token()
    api = HfApi(token=token)
    job = api.run_uv_job(
        script=script_text,
        script_args=script_args,
        python=args.python_version,
        image=args.image,
        secrets={"HF_TOKEN": token},
        flavor=args.flavor,
        timeout=args.timeout,
        labels=labels,
        namespace=args.namespace,
        token=token,
    )

    print(f"Submitted HF Job: {job.id}")
    print(f"Status: {job.status}")
    print(f"URL: {job.url}")
    print(f"Endpoint: {job.endpoint}")
    print(f"Flavor: {args.flavor}")
    print(f"Timeout: {args.timeout}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
