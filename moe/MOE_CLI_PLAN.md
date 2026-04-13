# MoE CLI Plan

This document records the planned `mesh-llm` command surface and UX for MoE ranking resolution, planning, analysis, submission, and `serve` integration.

It is a planning document for the Rust CLI and runtime behavior. It builds on:

- [`MOE_ANALYZE_STORAGE_SPEC.md`](MOE_ANALYZE_STORAGE_SPEC.md)
- [`MOE_PLACEMENT_PLAN.md`](MOE_PLACEMENT_PLAN.md)

## Goals

- Make MoE analysis and planning available from the main `mesh-llm` CLI.
- Reuse published rankings from `meshllm/moe-rankings` when available.
- Prefer local cached rankings when they are strong enough.
- Use Hugging Face when it has a stronger published artifact.
- Keep local cache on ties; published data only wins when it is stronger.
- Make planner and analyze output obvious, inspectable, and operationally friendly.
- Allow users to contribute locally generated rankings back to `meshllm/moe-rankings`.

## Command Family

Planned command family:

```text
mesh-llm moe plan <model>
mesh-llm moe analyze full <model>
mesh-llm moe analyze micro <model>
mesh-llm moe share <model>
```

These are explicit subcommands, not flag modes.

## `mesh-llm moe plan`

Purpose:

- resolve the best available ranking artifact for a model
- determine whether the model is feasible as an MoE split on the current hardware or a given memory target
- show a placement/sizing recommendation derived from the ranking
- emit either human-readable console output or machine-readable JSON

### Ranking Resolution Order

Planner ranking resolution should follow this precedence:

1. `--ranking-file` override
2. local `~/.cache/mesh-llm/...`
3. Hugging Face dataset `meshllm/moe-rankings`

However, local cache should only win when it is current enough.

Final resolution rule:

1. If `--ranking-file` is provided, use it directly.
2. Otherwise inspect local cached rankings.
3. Inspect `meshllm/moe-rankings`.
4. If Hugging Face has a stronger artifact than local cache, use the Hugging Face artifact via the Hugging Face cache/download path. Do not copy it into `~/.cache/mesh-llm/...`. If local and published artifacts have equal analyzer strength, keep the local cache.
5. Otherwise use local cache.

This means planner behavior is:

- cached by default
- strength-aware
- able to work offline when a current local artifact already exists

### Ranking Preference

For the same model/distribution identity, planner resolution should prefer:

1. `full-v1`
2. `micro-v1`

More generally, full analysis should outrank micro analysis when both are compatible.

### Planner Output Requirements

Planner output must make provenance obvious.

It should clearly report:

- model identity
- distribution identity
- selected analyzer id
- ranking source:
  - explicit override
  - local cache
  - Hugging Face dataset
- reason for selection:
  - override
  - local cache current
  - Hugging Face stronger than local
  - local wins on equal analyzer strength
- source revision

Planner output should also answer:

- whether the model appears feasible on the target hardware/memory
- what assumptions were used
- the recommended split/placement summary

Planner should support a machine-readable mode:

```text
--json
```

Machine-readable requirements:

- output must be valid JSON with no progress preamble or trailing text
- JSON should include:
  - model identity
  - ranking identity and source
  - target memory inputs
  - feasibility result
  - assumptions
  - proposed node/expert assignments
- console-oriented progress and status lines should be suppressed when `--json` is used

### Planner Override

Planner should support:

```text
--ranking-file <path>
```

Behavior:

- bypass normal ranking resolution
- validate the supplied ranking file
- clearly state that an override was used

## `mesh-llm moe analyze full`

Purpose:

- run the canonical full MoE analysis for a model
- cache the resulting artifact locally
- optionally submit the analyze run to Hugging Face Jobs with `--hf-job`
- on remote success, share the resulting artifact back through a dataset PR

This should align to:

- `full-v1`

## `mesh-llm moe analyze micro`

Purpose:

- run the canonical micro analysis for a model
- cache the resulting artifact locally
- optionally submit the analyze run to Hugging Face Jobs with `--hf-job`
- on remote success, share the resulting artifact back through a dataset PR

This should align to:

- `micro-v1`

## Analyze UX Requirements

The analyze commands should feel alive and debuggable.

### Terminal UX

Use explicit, readable status output with emojis.

Examples:

- `📍` model/distribution selection
- `📦` checking local cache
- `☁️` checking Hugging Face
- `⬇️` downloading ranking/model data
- `🧠` planning or analyzing
- `✅` success
- `⚠️` warning or fallback
- `❌` failure

### Progress Bars

Progress bars should be used for:

- remote downloads
- long-running analysis work
- any other long blocking task

### Remote Job Mode

`--hf-job` should:

- require a Hugging Face-backed model identity so the remote worker can resolve the same exact distribution
- submit a Hugging Face Job from the Rust CLI
- download a release bundle inside the remote job
- run the public `mesh-llm moe analyze ...` command inside the job
- run the public `mesh-llm moe share ...` command after successful analysis
- open a dataset PR against `meshllm/moe-rankings` rather than writing directly to `main`

### Error Discoverability

If analysis fails, the error must be inspectable afterward.

Requirements:

- print a concise failure summary in the terminal
- always report the log path
- write durable logs containing:
  - exact command
  - stdout
  - stderr
  - relevant file paths

Expected failure shape:

```text
❌ MoE analysis failed
Log: ~/.cache/mesh-llm/moe/.../run.log
Cause: llama-moe-analyze exited with code 1
```

## `mesh-llm moe share`

Purpose:

- validate a locally generated ranking artifact
- open contribution PRs for new ranking artifacts against `meshllm/moe-rankings`
- show clearly whether the artifact already existed or a new PR was opened

The preferred model is contribution-oriented, not blind direct write for all users.

`submit` should:

- resolve or accept a local artifact
- validate schema and layout
- detect likely duplicates when possible
- open a dataset PR via the Hugging Face commit API when the target artifact is new
- report:
  - source artifact path
  - target dataset path
  - whether this appears to be new or already published

Expected override support later may include:

- `--ranking-file`
- `--metadata-file`
- `--dry-run`

## Hugging Face Data Use

The planner should use `meshllm/moe-rankings` as the canonical published ranking source.

Requirements:

- use the Hugging Face cache for fetched data
- do not redownload unnecessarily
- still detect and use newer published artifacts when available

If the current `hf-hub` integration is missing dataset support needed for this flow, update the dependency or integration path instead of building a separate ad hoc downloader.

## `mesh-llm serve` Integration

`mesh-llm serve` should remain separate from the explicit `moe` command family, but it should consume MoE rankings when useful.

### Intended Behavior

If a published or cached ranking exists for the model being served:

- `mesh-llm serve` should use it

If no ranking exists:

- `mesh-llm serve` may fall back to a heuristic or locally generated ranking path
- it should make the fallback visible

### Suggested Follow-Up After Local Ranking Generation

If `serve` generates a local ranking because no published ranking was available, it should suggest contributing it.

Example:

```text
🧠 Generated local MoE ranking for this model
📍 Cached at: ~/.cache/mesh-llm/...
☁️ No published ranking was found in meshllm/moe-rankings
📤 Consider contributing it with:
   mesh-llm moe share <model>
```

`serve` should not auto-submit.

## Placement Planner Scope

The placement planner is derived from ranking artifacts.

It should use the assumptions and framing in [`MOE_PLACEMENT_PLAN.md`](MOE_PLACEMENT_PLAN.md), including:

- target memory or hardware limits
- whether the model can run as an MoE split on the target hardware
- a clear explanation of the resulting recommendation

Placement outputs remain derived, not part of the first canonical ranking artifact.

## Memory Flag Consistency

The planner should use the same max-memory/max-VRAM naming already used elsewhere in `mesh-llm`.

Do not introduce a conflicting new spelling if an existing standard already exists in the CLI.
