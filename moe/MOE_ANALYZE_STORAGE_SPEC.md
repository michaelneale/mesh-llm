# MoE Analyze Storage Spec

This document defines how `moe-analyze` artifacts are stored in two places:

1. a canonical Hugging Face dataset repo used as the immutable system of record
2. an optional colocated layout inside model repos for easy discovery by tools and users

The goal is to make rankings easy to publish, discover, validate, and consume from `mesh-llm` and other tools without relying on filename guessing or ad hoc conventions.

## Scope

This spec currently covers:

- GGUF source models
- `moe-analyze` output artifacts
- `micro` and `full` analysis methods

This spec is designed to extend to future formats such as MLX without changing the core identity model.

## Terms

- `source_repo`: the Hugging Face repo containing the source model, for example `unsloth/GLM-5.1-GGUF`
- `source_revision`: the exact source commit hash used for analysis
- `format`: the source model format, currently `gguf`
- `distribution_id`: the normalized published model distribution identity, for example `GLM-5.1-UD-IQ2_M`
- `analyzer_id`: the analysis method and version, for example `micro-v1` or `full-v1`

## Identity

Each ranking artifact is identified by:

- `source_repo`
- `source_revision`
- `format`
- `distribution_id`
- `analyzer_id`

This identity is canonical for the dataset repo.

## Distribution ID

For GGUF repos, `distribution_id` should identify the logical published model distribution, not an individual shard file.

Examples:

- `GLM-5.1-UD-IQ2_M-00001-of-00006.gguf` -> `GLM-5.1-UD-IQ2_M`
- `GLM-5.1-UD-Q8_K_XL-00001-of-00018.gguf` -> `GLM-5.1-UD-Q8_K_XL`
- `Qwen3-30B-A3B-Q4_K_M.gguf` -> `Qwen3-30B-A3B-Q4_K_M`

Normalization rule for sharded GGUF files:

- remove the `.gguf` suffix
- remove the trailing shard suffix `-NNNNN-of-NNNNN`
- use the remaining stem as `distribution_id`

Normalization rule for unsharded GGUF files:

- remove the `.gguf` suffix
- use the remaining stem as `distribution_id`

## Canonical Dataset Layout

The Hugging Face dataset repo is the immutable system of record.

Layout:

```text
data/
  <source_namespace>/
    <source_repo_name>/
      <source_revision>/
        <format>/
          <distribution_id>/
            <analyzer_id>/
              metadata.json
              ranking.csv
              run.log
```

Example:

```text
data/
  unsloth/
    GLM-5.1-GGUF/
      6f3c1d9d4d8d1a2b3c4d5e6f7a8b9c0d1e2f3a4b/
        gguf/
          GLM-5.1-UD-IQ2_M/
            micro-v1/
              metadata.json
              ranking.csv
              run.log
```

Rules:

- dataset paths must include the exact `source_revision`
- dataset artifacts are immutable once published
- rerunning analysis with a different method or version must create a new `analyzer_id`
- rerunning analysis for a different source commit must create a new `source_revision` path

## Colocated Model Repo Layout

Model repos may optionally expose a colocated convenience view of `moe-analyze` artifacts next to the GGUF files they describe.

For repos organized with one directory per quant or distribution, the artifacts should live inside that distribution directory.

Layout:

```text
<distribution_dir>/
  *.gguf
  moe-analyze/
    index.json
    <analyzer_id>/
      metadata.json
      ranking.csv
      run.log
```

Example:

```text
UD-IQ2_M/
  GLM-5.1-UD-IQ2_M-00001-of-00006.gguf
  GLM-5.1-UD-IQ2_M-00002-of-00006.gguf
  GLM-5.1-UD-IQ2_M-00003-of-00006.gguf
  moe-analyze/
    index.json
    micro-v1/
      metadata.json
      ranking.csv
      run.log
    full-v1/
      metadata.json
      ranking.csv
      run.log
```

Rules:

- colocated artifacts should sit next to the `.gguf` files they describe
- the model repo layout is a projection for discovery and convenience, not the immutable archive
- exact `source_revision` must still be recorded in `metadata.json`
- multiple analyzer methods may coexist under the same `moe-analyze/` directory

## Discovery

Tools should not discover available colocated analyses by scanning arbitrary filenames.

Tools should read:

```text
<distribution_dir>/moe-analyze/index.json
```

first.

### `index.json`

Purpose:

- advertise available analyses
- point to the canonical `metadata.json` and `ranking.csv` paths for each method
- define preference order for consumers

Example:

```json
{
  "schema_version": 1,
  "format": "gguf",
  "distribution_id": "GLM-5.1-UD-IQ2_M",
  "analyses": [
    {
      "analyzer_id": "full-v1",
      "metadata_path": "full-v1/metadata.json",
      "ranking_path": "full-v1/ranking.csv",
      "status": "complete"
    },
    {
      "analyzer_id": "micro-v1",
      "metadata_path": "micro-v1/metadata.json",
      "ranking_path": "micro-v1/ranking.csv",
      "status": "complete"
    }
  ],
  "preferred_order": [
    "full-v1",
    "micro-v1"
  ]
}
```

Rules:

- `index.json` should be the first lookup point for colocated discovery
- `preferred_order` should be explicit rather than inferred
- `status` should be `complete` only when `metadata.json` and `ranking.csv` are ready to consume

## `metadata.json`

`metadata.json` is the validation and provenance record.

Required fields:

```json
{
  "schema_version": 1,
  "source_repo": "unsloth/GLM-5.1-GGUF",
  "source_revision": "6f3c1d9d4d8d1a2b3c4d5e6f7a8b9c0d1e2f3a4b",
  "format": "gguf",
  "distribution_id": "GLM-5.1-UD-IQ2_M",
  "analyzer_id": "micro-v1",
  "analysis_tool": "llama-moe-analyze",
  "ranking_path": "ranking.csv",
  "primary_file": "UD-IQ2_M/GLM-5.1-UD-IQ2_M-00001-of-00006.gguf",
  "all_files": [
    "UD-IQ2_M/GLM-5.1-UD-IQ2_M-00001-of-00006.gguf",
    "UD-IQ2_M/GLM-5.1-UD-IQ2_M-00002-of-00006.gguf"
  ],
  "file_hashes": {
    "UD-IQ2_M/GLM-5.1-UD-IQ2_M-00001-of-00006.gguf": "sha256:...",
    "UD-IQ2_M/GLM-5.1-UD-IQ2_M-00002-of-00006.gguf": "sha256:..."
  }
}
```

Recommended additional fields:

- `llama_cpp_commit`
- `prompt_set`
- `prompt_count`
- `token_count`
- `all_layers`
- `command`
- `created_at`
- `status`

Rules:

- `metadata.json` must include the exact `source_revision`
- `metadata.json` must include `distribution_id`
- `metadata.json` must include the exact file list used for the run
- file hashes should be included whenever practical

## `ranking.csv`

`ranking.csv` is the consumable output ranking.

Current expected CSV shape:

```text
expert_id,total_mass,mass_fraction,selection_count
```

If future formats evolve, the schema version should be recorded in `metadata.json`.

## `run.log`

`run.log` is the raw or lightly normalized execution log for the analysis run.

Purpose:

- debugging failed or suspicious runs
- auditing analysis arguments
- preserving warnings emitted by `llama-moe-analyze`

## Analyzer IDs

Analyzer IDs should describe the analysis method and version.

Examples:

- `micro-v1`
- `full-v1`

Future examples:

- `micro-v2`
- `full-v2`

Rules:

- changing the analysis method requires a new `analyzer_id`
- changing prompts, token budget, or semantics enough to affect compatibility should also require a new `analyzer_id`
- patching metadata generation without changing ranking semantics may keep the same `analyzer_id`

## Tool Behavior

Recommended consumer behavior for `mesh-llm` and other tools:

1. resolve the loaded model to `source_repo`, `source_revision`, `format`, and `distribution_id`
2. prefer the canonical dataset record when querying a global archive
3. when consuming a colocated model repo, read `moe-analyze/index.json`
4. select the best available `analyzer_id` using `preferred_order`
5. load `metadata.json`
6. verify that `distribution_id`, `source_revision`, and file identity match expectations
7. load `ranking.csv`

Tools should not rely on:

- guessing rankings by model family name alone
- assuming a single `ranking.csv` exists without consulting `index.json`
- inferring method preference from directory names alone

## Non-Goals

This spec does not currently define:

- a publication workflow
- a Hugging Face Job submission format
- MLX-specific normalization rules
- ranking merge or deduplication semantics across methods

Those can be added later without changing the storage identity model above.
