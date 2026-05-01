# MoE

This directory contains the MoE ranking work for `mesh-llm`.

## Contents

- [`MOE_ANALYZE_STORAGE_SPEC.md`](MOE_ANALYZE_STORAGE_SPEC.md)
  - Defines the canonical Hugging Face dataset layout for published `moe-analyze` artifacts.
  - Defines the optional colocated model-repo sidecar layout for `moe-analyze/` metadata next to GGUF files.
- [`MOE_PLACEMENT_PLAN.md`](MOE_PLACEMENT_PLAN.md)
  - Defines how deployment-oriented split planning is derived from canonical `moe-analyze` results rather than stored as part of the first artifact.
- [`MOE_CLI_PLAN.md`](MOE_CLI_PLAN.md)
  - Defines the `mesh-llm moe` command family, ranking resolution rules, UX decisions, and `serve` integration.

## Current Scope

- GGUF source models
- `micro-v1` and `full-v1` analyzer ids
- Canonical publication to the `meshllm/moe-rankings` Hugging Face dataset
- `mesh-llm moe plan` console and `--json` output modes
- `mesh-llm moe share` opening contribution PRs against `meshllm/moe-rankings`
- `mesh-llm moe analyze {full,micro} --hf-job` for remote analyze-and-share runs on Hugging Face Jobs

## Entry Points

- Read the storage contract in [`MOE_ANALYZE_STORAGE_SPEC.md`](MOE_ANALYZE_STORAGE_SPEC.md).
- Read the placement-planning note in [`MOE_PLACEMENT_PLAN.md`](MOE_PLACEMENT_PLAN.md).
- Read the CLI and UX plan in [`MOE_CLI_PLAN.md`](MOE_CLI_PLAN.md).
- Use `mesh-llm moe analyze full <model>` or `mesh-llm moe analyze micro <model>` to generate local ranking artifacts.
- Use `mesh-llm moe share <model>` to open a dataset PR with an existing local artifact.
- Use `mesh-llm moe analyze {full,micro} <model> --hf-job` to queue a remote analyze run that shares the result back through a dataset PR on success.
- For split GGUF distributions, prefer HF repo selector syntax such as `unsloth/gemma-4-26B-A4B-it-GGUF:BF16@main` instead of naming a specific shard file.

## Notes

- The Hugging Face dataset is the immutable system of record.
- Placement and split recommendations are derived later from canonical analysis artifacts.
- Model-repo sidecars are documented, but not automatically generated yet.
- `micro-v1` is bound to the built-in canonical prompt set.
