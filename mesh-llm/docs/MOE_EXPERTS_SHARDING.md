# MoE Package Repository Spec

Specification for publishing Mesh-LLM MoE artifacts as self-contained
Hugging Face repositories, with a separate Mesh catalog for resolution.

## Summary

This is a deliberate break from the legacy design.

We are no longer treating `meshllm/moe-analysis` as the canonical home for MoE
artifacts, and we are not extending the deleted bundled `catalog.json` model
list into the new system.

The new design has two layers:

- `meshllm/catalog`
  a small resolver dataset keyed by canonical Mesh model refs such as
  `unsloth/Qwen3.6-35B-A3B-GGUF:BF16`
- one Mesh-owned Hugging Face repo per upstream source repo
  containing active MoE artifacts for one or more variants from that source

This means:

- `meshllm/moe-analysis` is legacy and should be deprecated
- raw blob URLs are legacy and should be dropped
- the Hugging Face-backed `meshllm/catalog` dataset is now the catalog of
  record
- the new hot path for `mesh-llm serve` is:
  - resolve model ref in `meshllm/catalog`
  - load the package repo
  - read `meshllm.json`
  - load the selected variant's `manifest.json`
  - download `trunk.gguf` plus only the required experts

## Goals

- avoid combinatorial growth from publishing full topology-specific split sets
- publish one topology-independent MoE decomposition per model variant
- let one package repo hold multiple variants from the same upstream source repo
- keep package identity separate from runtime assembly metadata
- make `serve` resolve from a canonical Mesh model ref instead of a transport URL
- make the local fallback build the same package-shaped artifact layout that
  `moe publish` publishes

## Non-Goals

- supporting raw blob URLs as first-class catalog entries
- continuing to use `meshllm/moe-analysis` as the canonical artifact store
- supporting a root-level single-variant shortcut layout
- publishing one prebuilt split set per supported node count

## Canonical Model Ref

The canonical Mesh identifier remains:

```text
owner/repo:variant
```

Examples:

```text
unsloth/Qwen3.6-35B-A3B-GGUF:BF16
unsloth/Qwen3.6-35B-A3B-GGUF:Q4_K_XL
```

This format is the key used in `meshllm/catalog`.

## New Architecture

### `meshllm/catalog`

`meshllm/catalog` is the resolver index. It should stay small and only answer:

- does Mesh have a package for this canonical model ref?
- which repo contains it?
- which repo revision should runtime read?
- who published it?
- what trust level should runtime assign to it?

The catalog may contain multiple entries for the same `model_ref`.

For v1, the trust model is:

- `canonical`
  a Mesh-owned package repo, typically under `meshllm/...`
- `community`
  a user- or org-owned package repo that has been proposed into the catalog

Community publishers should be able to publish package repos in their own
namespace and open a PR adding those repos to `meshllm/catalog`.

Example row:

```json
{
  "schema_version": 1,
  "model_ref": "unsloth/Qwen3.6-35B-A3B-GGUF:Q4_K_XL",
  "package_repo": "meshllm/qwen3.6-35b-a3b-gguf-moe",
  "package_revision": "ab13dc98b9f1c3d4e5f60718293a4b5c6d7e8f90",
  "publisher": "meshllm",
  "trust": "canonical"
}
```

Example community row:

```json
{
  "schema_version": 1,
  "model_ref": "unsloth/Qwen3.6-35B-A3B-GGUF:Q4_K_XL",
  "package_repo": "jdumay/qwen3.6-35b-a3b-gguf-moe",
  "package_revision": "de45f6a7b8c9d0123456789abcdef0123456789",
  "publisher": "jdumay",
  "trust": "community"
}
```

### Per-source package repo

Each Mesh package repo is keyed by upstream source repo, not by one specific
variant.

Example repo:

```text
meshllm/qwen3.6-35b-a3b-gguf-moe
```

That repo may contain multiple variants from:

```text
unsloth/Qwen3.6-35B-A3B-GGUF
```

### Top-level package descriptor

Each package repo contains a top-level `meshllm.json` file describing:

- the upstream source repo
- the pinned upstream revision
- which variants are present
- where each variant's runtime manifest lives

Example:

```json
{
  "schema_version": 1,
  "source": {
    "repo": "unsloth/Qwen3.6-35B-A3B-GGUF",
    "revision": "9280dd353ab587157920d5bd391ada414d84e552"
  },
  "variants": {
    "BF16": {
      "distribution_id": "Qwen3.6-35B-A3B-BF16",
      "manifest": "variants/BF16/manifest.json"
    },
    "Q4_K_XL": {
      "distribution_id": "Qwen3.6-35B-A3B-UD-Q4_K_XL",
      "manifest": "variants/Q4_K_XL/manifest.json"
    }
  }
}
```

## Package Repo Layout

All active variant artifacts live under `variants/<variant>/`.

Even single-variant repos must use this layout. There is no root-level variant
shortcut.

```text
README.md
meshllm.json
variants/
  BF16/
    analysis.json
    ranking.csv
    manifest.json
    run.log
    trunk.gguf
    experts/
      expert-000.gguf
      expert-001.gguf
      ...
  Q4_K_XL/
    analysis.json
    ranking.csv
    manifest.json
    run.log
    trunk.gguf
    experts/
      expert-000.gguf
      expert-001.gguf
      ...
```

There is no `history/` directory in v1.

If we want retained publication history later, it can be added in a future
revision. For now, each variant directory only contains the active publication.

## Local Cache Layout

When `serve` cannot resolve a published package in `meshllm/catalog`, it
should build and cache the same package-shaped variant layout locally instead of
creating topology-specific split directories.

The local package cache should mirror the published package layout under:

```text
~/.cache/mesh-llm/moe/packages/<source-owner>/<source-repo>/<source-revision>/
  meshllm.json
  variants/
    <variant>/
      analysis.json
      ranking.csv
      manifest.json
      run.log
      trunk.gguf
      experts/
        expert-000.gguf
        expert-001.gguf
        ...
```

This local cache is the source of truth for:

- local runtime assembly
- later `moe publish` uploads
- topology changes after the initial local extraction

Legacy topology-specific GGUF split caches should be deleted rather than
retained or migrated forward.

## File Responsibilities

### `meshllm.json`

Repo-level descriptor.

Owns:

- source repo identity
- source revision
- available variants
- each variant's runtime manifest location

Does not own:

- analysis results
- ranking contents
- runtime expert inventory

### `variants/<variant>/manifest.json`

Runtime assembly manifest for one variant.

Owns:

- ranking hash used for validation
- MoE constants needed at runtime
- `trunk.gguf` path and hash
- expert file paths and hashes

Does not own:

- source repo identity
- source revision
- package repo identity
- descriptive metadata

Example:

```json
{
  "schema_version": 1,
  "format": "meshllm-moe-components",
  "ranking_sha256": "sha256:5f2b6f4d9b0e1a2c...",
  "n_expert": 128,
  "n_expert_used": 8,
  "min_experts_per_node": 46,
  "trunk": {
    "path": "trunk.gguf",
    "sha256": "sha256:19f2d5d6f2d2..."
  },
  "experts": [
    {
      "expert_id": 0,
      "path": "experts/expert-000.gguf",
      "sha256": "sha256:8a0d6c..."
    },
    {
      "expert_id": 1,
      "path": "experts/expert-001.gguf",
      "sha256": "sha256:1ff9a1..."
    }
  ]
}
```

Paths in the manifest are variant-relative because the manifest lives inside the
variant directory.

### `variants/<variant>/analysis.json`

Combined analysis results plus analysis-run metadata.

This file replaces the split between `analysis.json` and `metadata.json`.

Owns:

- analyzer id
- tool/version
- analysis parameters
- summary results
- planner notes
- pointers to the ranking and manifest files

Does not own:

- package identity
- source identity already present in `meshllm.json`

Example:

```json
{
  "schema_version": 1,
  "analyzer_id": "full-v1",
  "created_at": "2026-04-18T06:12:44Z",
  "tool": {
    "name": "llama-moe-analyze",
    "version": "mesh-llm-fork"
  },
  "parameters": {
    "all_layers": true,
    "context_size": 32768,
    "n_gpu_layers": 999
  },
  "summary": {
    "n_expert": 128,
    "n_expert_used": 8,
    "min_experts_per_node": 46
  },
  "planner": {
    "recommended_overlap": 1,
    "notes": [
      "46 experts per node preserved output quality in local benchmarks"
    ]
  },
  "artifacts": {
    "ranking": "ranking.csv",
    "manifest": "manifest.json"
  }
}
```

### `variants/<variant>/ranking.csv`

Canonical ranking artifact.

This remains a separate file.

### `variants/<variant>/run.log`

Latest analyze/share log for the active variant publication.

This is for debugging the current publication only. v1 does not preserve older
logs after a new publication replaces the active files.

## Why This Layout

- one source repo can naturally publish many variants
- variant boundaries already namespace the artifacts, so an extra `moe/`
  subdirectory is not needed
- `meshllm.json` is the single identity anchor
- runtime files do not need to duplicate package or source identity
- package repos remain self-contained and understandable on the Hub

## Legacy Breaks

This spec intentionally breaks with the legacy model in several ways.

### Deprecated: `meshllm/moe-analysis`

`meshllm/moe-analysis` should no longer be the canonical location for:

- rankings
- analysis output
- manifests
- trunks
- expert shards

Existing content there is legacy.

### Deprecated: bundled `catalog.json` as the MoE registry

The old bundled catalog at `mesh-llm/src/models/catalog.json` was a curated
download list with mixed transports and friendly labels.

That does not fit the new package-repo design because it:

- is keyed by display names instead of canonical Mesh refs
- points at direct transport URLs instead of Mesh package repos
- still contains non-HF blob-style entries

It has been replaced by the Hugging Face-backed `meshllm/catalog` dataset and
is not the source of
truth for MoE package resolution.

### Dropped: raw blob URLs

Raw blob URLs are not supported in the new catalog.

All catalog entries must resolve from canonical Mesh refs of the form:

```text
owner/repo:variant
```

Entries that only exist as raw transport blobs should be dropped rather than
ported into the new catalog.

## Runtime Resolution

For:

```bash
mesh-llm serve unsloth/Qwen3.6-35B-A3B-GGUF:Q4_K_XL
```

the intended runtime flow is:

1. Resolve `unsloth/Qwen3.6-35B-A3B-GGUF:Q4_K_XL` in `meshllm/catalog`.
2. Prefer a `canonical` entry when one exists; otherwise allow a `community`
   entry.
3. Fetch the selected package repo at `package_revision`.
4. Read `meshllm.json`.
5. Find the `Q4_K_XL` variant entry.
6. Load `variants/Q4_K_XL/manifest.json`.
7. Resolve the ranking and validate `ranking_sha256`.
8. Download `trunk.gguf` if missing.
9. Download only the required expert files for the local assignment.
10. Materialize a local runnable shard.
11. Launch `llama-server`.
12. If there is no catalog package, or package resolution fails, build or
    reuse the local package cache under `~/.cache/mesh-llm/moe/packages/...`.
13. Assemble the runnable shard from the cached `trunk.gguf` plus the selected
    experts for the current topology.
14. Delete legacy topology-specific split artifacts instead of reusing them.

## Hugging Face Client Behavior

Whenever mesh-llm contacts Hugging Face for this workflow, it should use:

- cache-aware clients and paths
- visible progress reporting for user-facing operations

This applies to both:

- metadata reads such as catalog lookup, package descriptor fetches, and
  manifest reads
- data transfer such as package uploads and artifact downloads

### Cache

All Hugging Face reads should reuse the local Hugging Face cache when possible.

That includes:

- `meshllm/catalog` fetches
- `meshllm.json` fetches
- variant `manifest.json` fetches
- `trunk.gguf` downloads
- expert artifact downloads

The runtime should not repeatedly re-fetch metadata or artifacts that already
exist in cache and match the required revision or hash.

### Progress

User-facing Hugging Face operations should show progress instead of silently
blocking.

That includes:

- upload progress for `mesh-llm moe publish`
- download progress for large artifacts
- lightweight progress indicators or spinners for metadata fetches when they
  are part of an interactive CLI flow

The intent is:

- if network work is happening, the user should see that it is happening
- if large files are moving, the user should see per-file and overall progress
- if cached data is reused, the CLI should make that visible where helpful

## Share Workflow

The preferred UX remains:

```bash
mesh-llm moe publish MODEL
```

- `mesh-llm moe publish MODEL`
  publishes the complete variant package:
  - `analysis.json`
  - `ranking.csv`
  - `manifest.json`
  - `run.log`
  - `trunk.gguf`
  - `experts/expert-*.gguf`

Under the new spec, share should publish to the package repo layout described
above and then update `meshllm/catalog` to point at that package revision.

If the publisher does not have permission to create or write the canonical
`meshllm/...` package repo, they should still be able to:

1. publish the package repo in their own namespace
2. open a PR adding a `community` entry to `meshllm/catalog`

Publishing order:

1. upload the package repo changes
2. obtain the resulting package revision
3. update `meshllm/catalog`

The catalog must not point at a package revision that does not exist yet.

## Upload Path

All MoE share uploads should use the Rust `huggingface_hub_rust` client.

Do not keep:

- one small-artifact NDJSON path
- another path for large-file expert uploads

The entire MoE share flow should use one Hub upload implementation with:

- contribution PR creation
- xet/LFS-backed large-file support
- shared progress reporting
- shared retry and resume behavior
- cache-aware metadata and branch inspection

## llama.cpp Tooling Decision

Keep `llama-moe-split` as the stable runnable-shard tool.

Do not turn it into the kitchen-sink interface for the new artifact model.

Instead:

- move shared internals into reusable code in the llama.cpp fork
- keep `llama-moe-split` as the stable legacy split tool rather than the new
  runtime fallback path
- use a sibling tool for component extraction and assembly

Required component operations:

- extract trunk
- extract one expert
- assemble `trunk + selected experts -> runnable shard.gguf`

Without that assembly step, `serve` cannot consume the new package format.

## Implementation Plan

### Phase 1: Define and freeze the new spec

1. Finalize this package layout and file schema.
2. Treat `meshllm/moe-analysis` as legacy.
3. Treat raw blob URL catalog entries as legacy.
4. Decide the exact schema for `meshllm/catalog`.

Deliverables:

- published spec
- agreed schema examples
- explicit legacy boundary

### Phase 2: Introduce the new package format in mesh-llm

1. Add data structures for:
   - `meshllm/catalog` entries
   - `meshllm.json`
   - variant `manifest.json`
   - variant `analysis.json`
2. Teach share to stage files into:
   - `meshllm.json`
   - `variants/<variant>/analysis.json`
   - `variants/<variant>/ranking.csv`
   - `variants/<variant>/manifest.json`
   - `variants/<variant>/run.log`
   - `variants/<variant>/trunk.gguf`
   - `variants/<variant>/experts/expert-*.gguf`
3. Remove assumptions that expert artifacts live under the old dataset prefix.

Deliverables:

- local staging of the new layout
- serialization for the new schema
- local cache layout that mirrors the published package layout

### Phase 3: Publish into per-source package repos

1. Change `moe publish` so it publishes into the per-source Mesh repo instead of
   the old analysis dataset.
2. Reuse the Rust Hub upload path for the complete package upload flow.
3. Preserve contribution PR creation.
4. Support publishing into either:
   - the canonical `meshllm/...` namespace
   - a publisher-owned namespace for community packages
5. After package upload succeeds, update `meshllm/catalog` to point to the
   new package revision with the correct `publisher` and `trust`.
6. Ensure all Hugging Face metadata reads and uploads use cache-aware clients
   and visible progress reporting.

Deliverables:

- package repo publication
- catalog publication
- no dependency on `meshllm/moe-analysis` for new uploads

### Phase 4: Teach `serve` the new resolver

1. Resolve canonical Mesh refs through `meshllm/catalog`.
2. Prefer `canonical` entries over `community` entries.
3. Fetch `meshllm.json` from the package repo revision.
4. Resolve the requested variant.
5. Load the variant `manifest.json`.
6. Download `trunk.gguf` plus only the needed experts.
7. Assemble a local shard and launch it.
8. If there is no catalog package, or package resolution fails, prefer a
   local package-shaped component cache:
   - write `meshllm.json` plus `variants/<variant>/...` into
     `~/.cache/mesh-llm/moe/packages/<source-owner>/<source-repo>/<source-revision>/`
   - reuse locally extracted `trunk.gguf`
   - reuse locally extracted `experts/expert-*.gguf`
   - assemble the required shard for the current topology from those local
     components
9. Delete any legacy topology-specific GGUF split cache entries instead of
   reusing them.
10. Ensure `moe publish` can upload directly from the local package cache when it
    already matches the current ranking and manifest.
11. Ensure Hugging Face metadata and artifact fetches use cache-first behavior
    and visible progress in interactive CLI flows.

Deliverables:

- runtime resolution from `meshllm/catalog`
- package-shaped local cache reuse when topology changes
- topology-independent expert reuse
- no new topology-specific split cache

### Phase 5: Legacy cleanup

1. Stop treating `meshllm/moe-analysis` as a publication target.
2. Stop relying on direct artifact URLs for MoE package resolution.
3. Remove or quarantine legacy catalog entries that cannot be represented as
   canonical `owner/repo:variant` Mesh refs.
4. Keep only the new Hugging Face-backed `meshllm/catalog` path at runtime.
5. Store catalog data as one file per source repo under
   `entries/<owner>/<repo>.json`, with:
   - curated variant metadata
   - zero or more package pointers per variant
6. Use local fixture entries under `mesh-llm/tests/fixtures/catalog/` for tests
   instead of any bundled runtime snapshot.

Deliverables:

- one canonical MoE publication path
- one canonical MoE resolver path
- one canonical per-source catalog entry format

## Open Questions

- Should `manifest.json` include any additional assembler-level compatibility
  fields beyond `ranking_sha256`, expert counts, and hashes?
- Should `run.log` always be uploaded, or should share allow a lighter-weight
  publication mode later?
- Do we want a separate discovery view later, or is the per-source catalog
  entry format enough on its own?

## Current Recommendation

Build the new system around:

- `meshllm/catalog` as the resolver index
- `entries/<owner>/<repo>.json` as the canonical catalog storage shape
- one Mesh-owned per-source package repo
- `meshllm.json` as the repo descriptor
- `variants/<variant>/...` as the active artifact layout
- `analysis.json`, `ranking.csv`, `manifest.json`, `run.log`, `trunk.gguf`,
  and `experts/` as the variant files

and treat the old `meshllm/moe-analysis` plus raw URL catalog model as legacy.
