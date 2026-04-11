# MoE

This directory contains the MoE ranking work for `mesh-llm`.

## Contents

- [`MOE_ANALYZE_STORAGE_SPEC.md`](MOE_ANALYZE_STORAGE_SPEC.md)
  - Defines the canonical Hugging Face dataset layout for published `moe-analyze` artifacts.
  - Defines the optional colocated model-repo sidecar layout for `moe-analyze/` metadata next to GGUF files.
- [`MOE_PLACEMENT_PLAN.md`](MOE_PLACEMENT_PLAN.md)
  - Defines how deployment-oriented split planning is derived from canonical `moe-analyze` results rather than stored as part of the first artifact.
- [`scripts/`](scripts/)
  - Contains the dataset-generation tooling for downloading GGUF distributions, running `llama-moe-analyze`, publishing artifacts, and submitting the same workflow to Hugging Face Jobs.

## Current Scope

- GGUF source models
- `micro-v1` and `full-v1` analyzer ids
- Canonical publication to the `meshllm/moe-rankings` Hugging Face dataset

## Entry Points

- Read the storage contract in [`MOE_ANALYZE_STORAGE_SPEC.md`](MOE_ANALYZE_STORAGE_SPEC.md).
- Read the placement-planning note in [`MOE_PLACEMENT_PLAN.md`](MOE_PLACEMENT_PLAN.md).
- Use [`scripts/analyze_and_publish.py`](scripts/analyze_and_publish.py) to generate and publish canonical dataset artifacts.
- Use [`scripts/submit_hf_job.py`](scripts/submit_hf_job.py) to run the same workflow on Hugging Face Jobs.
- See [`scripts/README.md`](scripts/README.md) for usage examples.

## Notes

- The Hugging Face dataset is the immutable system of record.
- Placement and split recommendations are derived later from canonical analysis artifacts.
- Model-repo sidecars are documented, but not automatically generated yet.
- `micro-v1` is bound to the built-in canonical prompt set.
