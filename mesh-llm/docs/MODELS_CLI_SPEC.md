# Models CLI Spec

Status: Design draft (approved for implementation)

## 0. Problem Statement

The previous CLI surface was mismatched with how users actually choose and run models.

Core issues:

- It exposed low-level artifact details (exact shard filenames, `.gguf` suffixes, split-part names) instead of stable model identities.
- Search output was noisy and ambiguous about which command to run next.
- Gated Hugging Face repos could break metadata flows or produce opaque failures instead of actionable messaging.
- Download and serve semantics were conflated:
  - users could not clearly see when a model was downloadable vs locally runnable.
- Variant/quant selection lacked a clear contract:
  - users could not tell why one variant was picked over another.
- Capability variants in mixed repos (text/vision/audio) were not handled consistently:
  - users could not predict which capability family would be selected.
- Download reliability behavior (retry, resume, restart) was not explicit as a user-facing contract.

Resulting user pain:

- copy/paste friction due to shard-level refs
- confusing command output
- avoidable runtime failures
- unclear recovery behavior when downloads are interrupted

This spec redefines the CLI around stable refs, explicit selection outcomes, machine-aware recommendations, deterministic gating/fit rules, and explicit download resilience semantics.

## 1. Goals

- Make model refs human-friendly and stable.
- Keep search/show output explicit, non-lossy, and pasteable.
- Handle gated repos gracefully.
- Allow downloads even when a model is too large to serve locally.
- Block `serve`/`load` for non-MoE models that do not fit local capacity.
- Make downloads resilient with retry/backoff and recoverable resume behavior compatible with `huggingface_hub`.

## 2. Canonical Ref Format

User-facing canonical model ref:

- `repo/model-stem`

Examples:

- `unsloth/MiniMax-M2-GGUF/MiniMax-M2-Q4_K_M`
- `wimmmm/MiniMax-M2.5-REAP-172B-A10B-GGUF/Cerebras-MiniMax-M2.5-REAP-172B-Q4_K_M`

Notes:

- No required `.gguf` suffix in user refs.
- No shard suffix in user refs (`-00001-of-00003` is internal resolution detail).
- Resolver maps `repo/model-stem` to concrete file(s), including split shard expansion.

## 3. Commands

- `mesh-llm models search <query> [--limit N] [--json]`
- `mesh-llm models show <repo|repo/model-stem> [--all] [--all-files] [--json]`
- `mesh-llm models download <repo/model-stem> [--force-download]`
- `mesh-llm serve --model <repo/model-stem>`
- `mesh-llm load <repo/model-stem>`

Capability selectors (where supported):

- `--text`
- `--vision`
- `--audio`
- `--multimodal`

## 4. Search Behavior

Each entry prints:

1. Numbered repo heading (repo name is always the heading).
2. Repo URL.
3. Stats (`📦 files`, `⬇️`, `❤️`).
4. Short description (ellipsis when needed).
5. `✅ recommended for this machine`.
6. `🏆 highest quality`.
7. `🔎 models show ...`.
8. `⬇️ models download ...` pointing to recommended-for-machine ref.

Gated entries are listed (not dropped), with:

- `🟡 gated: additional info and downloads are unavailable until terms are accepted`

Capability behavior:

- Selection shown in `search` must respect the requested capability profile.
- If user passed capability selectors, include:
  - `🎯 capability: <profile>`
- If capability profile has no matching variants in a repo, skip recommendation lines for that repo and show:
  - `🟡 no variants matching requested capability profile`

### 4.1 Example

```text
$ mesh-llm models search minimax
🔎 Searching Hugging Face GGUF repos for 'minimax'...
   Inspecting 20 candidate repos...
   Inspected 20/20 candidate repos...

1. unsloth/MiniMax-M2-GGUF
   🔗 https://huggingface.co/unsloth/MiniMax-M2-GGUF
   📦 87 GGUF files  ⬇️ 1,474  ❤️ 87
   📝 Meet MiniMax-M2: compact MoE model for coding and agentic workflows.
   ✅ recommended for this machine: unsloth/MiniMax-M2-GGUF/MiniMax-M2-Q2_K_L
   🏆 highest quality: unsloth/MiniMax-M2-GGUF/MiniMax-M2-BF16
   🔎 mesh-llm models show unsloth/MiniMax-M2-GGUF
   ⬇️ mesh-llm models download unsloth/MiniMax-M2-GGUF/MiniMax-M2-Q2_K_L

2. Ex0bit/MiniMax-M2.5-PRISM-PRO
   🔗 https://huggingface.co/Ex0bit/MiniMax-M2.5-PRISM-PRO
   📦 7 GGUF files  ⬇️ 1,396  ❤️ 17
   🟡 gated: additional info and downloads are unavailable until terms are accepted
   🔎 mesh-llm models show Ex0bit/MiniMax-M2.5-PRISM-PRO
```

## 5. Show Behavior

### 5.1 `show <repo>`

Print:

- repo header + stats + short description
- capability availability summary
- recommended-for-machine block
- highest-quality block
- other variants table

Other variants heading:

- `🧾 Other variants (Found N)`

Columns:

- `🔗 ref`
- `⚖️ quant`
- `🎯 capability`
- `📏 size`
- `💻 fit`

All `ref` values are fully qualified and directly pasteable after:

- `mesh-llm models download `

### 5.2 `show <repo/model-stem>`

Print:

- resolved canonical ref
- quant
- total size
- fit
- resolved concrete file list (full shards if split)

### 5.3 Examples

```text
$ mesh-llm models show unsloth/MiniMax-M2-GGUF
📦 unsloth/MiniMax-M2-GGUF
🔗 https://huggingface.co/unsloth/MiniMax-M2-GGUF
📦 87 GGUF files  ⬇️ 1,474  ❤️ 87
💬 text-generation
📝 Meet MiniMax-M2: compact MoE model for coding and agentic workflows.
🎯 Capabilities available: text

✅ recommended for this machine
   🔗 unsloth/MiniMax-M2-GGUF/MiniMax-M2-Q2_K_L
   ⬇️ mesh-llm models download unsloth/MiniMax-M2-GGUF/MiniMax-M2-Q2_K_L

🏆 highest quality
   🔗 unsloth/MiniMax-M2-GGUF/MiniMax-M2-BF16
   ⬇️ mesh-llm models download unsloth/MiniMax-M2-GGUF/MiniMax-M2-BF16

🧾 Other variants (Found 25)
🔗 ref                                                         ⚖️ quant   🎯 capability  📏 size     💻 fit
unsloth/MiniMax-M2-GGUF/MiniMax-M2-Q8_0                       Q8_0       text          226.4 GB   ❌
unsloth/MiniMax-M2-GGUF/MiniMax-M2-Q6_K                       Q6_K       text          174.8 GB   ❌
unsloth/MiniMax-M2-GGUF/MiniMax-M2-Q5_K_M                     Q5_K_M     text          151.1 GB   ❌
...
```

```text
$ mesh-llm models show unsloth/MiniMax-M2-GGUF/MiniMax-M2-Q4_K_M
📦 unsloth/MiniMax-M2-GGUF
🔗 https://huggingface.co/unsloth/MiniMax-M2-GGUF
🔗 ref: unsloth/MiniMax-M2-GGUF/MiniMax-M2-Q4_K_M
⚖️ quant: Q4_K_M
📏 total size: 128.8 GB
💻 fit: ❌

Resolved files:
- Q4_K_M/MiniMax-M2-Q4_K_M-00001-of-00003.gguf
- Q4_K_M/MiniMax-M2-Q4_K_M-00002-of-00003.gguf
- Q4_K_M/MiniMax-M2-Q4_K_M-00003-of-00003.gguf
```

## 6. Variant Selection Policy

Two outputs are always distinct:

- `✅ recommended for this machine`: best locally runnable choice.
- `🏆 highest quality`: best quality ignoring local fit.

Capability matching is evaluated before quant/size ranking.

### 6.0 Capability Variant Handling

Repos may contain capability-specific variants (for example text-only and vision-capable files in the same repo). Selection order is:

1. Determine target capability profile.
2. Filter variants to those that satisfy that profile.
3. Apply ranking (`recommended for this machine` / `highest quality`) within the filtered set.

Profile rules:

- Default profile is text-oriented (`--text` behavior) unless user explicitly requests another capability.
- `--vision` requires vision-capable variants.
- `--audio` requires audio-capable variants.
- `--multimodal` requires multimodal-capable variants.
- Multiple selectors combine as intersection (all requested capabilities must be present).

Failure behavior:

- If no variants satisfy requested capability profile, fail with:
  - `🟡 No <capability>-capable variants found in <repo>. Run 'mesh-llm models show <repo>' to inspect available variants.`

Resolver policy:

1. Apply required selectors (family/ctx/capability if present).
2. Rank by profile (`recommended` vs `highest quality`).
3. Tie-break by smaller total bytes, then stable lexical fallback.

Split handling:

- Split is transparent to user refs.
- Resolver expands split to all required shards for download.

### 6.1 Variant Auto-Pick Examples

```text
$ mesh-llm models download MiniMax-M2.5-REAP-172B-A10B-GGUF
🔎 Resolving model ref: MiniMax-M2.5-REAP-172B-A10B-GGUF
📦 Repo: wimmmm/MiniMax-M2.5-REAP-172B-A10B-GGUF
🧾 Found 7 GGUF variants
💻 Local capacity: 96.0 GB VRAM (fit threshold includes 10% headroom)

✅ Picked variant that fits your machine:
   🔗 wimmmm/MiniMax-M2.5-REAP-172B-A10B-GGUF/Cerebras-MiniMax-M2.5-REAP-172B-Q4_K_M
   ⚖️ quant: Q4_K_M
   📏 size: 104.4 GB
   💻 fit: ✅

📥 Ensuring wimmmm/MiniMax-M2.5-REAP-172B-A10B-GGUF/Cerebras-MiniMax-M2.5-REAP-172B-Q4_K_M@main is available locally...
📥 Downloading model Cerebras-MiniMax-M2.5-REAP-172B-Q4_K_M
✅ Cached locally
```

```text
$ mesh-llm models download unsloth/MiniMax-M2-GGUF --vision
🔎 Resolving model ref: unsloth/MiniMax-M2-GGUF --vision
📦 Repo: unsloth/MiniMax-M2-GGUF
🧾 Found 26 GGUF variants
👁️ Requested capability: vision
🟡 No vision-capable variants found in unsloth/MiniMax-M2-GGUF. Run 'mesh-llm models show unsloth/MiniMax-M2-GGUF' to inspect available variants.
```

```text
$ mesh-llm serve --model MiniMax-M2.5-REAP-172B-A10B-GGUF
🔎 Resolving model ref: MiniMax-M2.5-REAP-172B-A10B-GGUF
📦 Repo: wimmmm/MiniMax-M2.5-REAP-172B-A10B-GGUF
🧾 Found 7 GGUF variants
💻 Local capacity: 96.0 GB VRAM (fit threshold includes 10% headroom)

✅ Picked variant that fits your machine:
   🔗 wimmmm/MiniMax-M2.5-REAP-172B-A10B-GGUF/Cerebras-MiniMax-M2.5-REAP-172B-Q4_K_M
   ⚖️ quant: Q4_K_M
   📏 size: 104.4 GB
   💻 fit: ✅

📥 Ensuring wimmmm/MiniMax-M2.5-REAP-172B-A10B-GGUF/Cerebras-MiniMax-M2.5-REAP-172B-Q4_K_M@main is available locally...
✅ Cached locally

🚀 Starting mesh-llm serve with:
   🔗 wimmmm/MiniMax-M2.5-REAP-172B-A10B-GGUF/Cerebras-MiniMax-M2.5-REAP-172B-Q4_K_M
✅ llama-server ready on :9337
```

## 7. Gated Repo Policy

### 7.1 Search

- Include gated repos in results.
- Mark capability limits with gated message.

### 7.2 Download

Hard stop with exact copy:

- `🟡 This Hugging Face repo is gated and cannot be downloaded until terms are accepted at https://huggingface.co/<repo>`

No `Error:` prefix.

### 7.3 Show

- Show repo-level information only when available.
- For blocked metadata, show gated notice.
- If capability-specific metadata is blocked, include that capability details are unavailable until terms are accepted.

### 7.4 Example

```text
$ mesh-llm models download Ex0bit/MiniMax-M2.5-PRISM-PRO/MiniMax-M2.5-PRISM-PRO-IQ2_XXS
🟡 This Hugging Face repo is gated and cannot be downloaded until terms are accepted at https://huggingface.co/Ex0bit/MiniMax-M2.5-PRISM-PRO
```

## 8. Fit Policy

Shared fit rule:

- `required_bytes = model_bytes * 1.10`
- `fits_local = available_vram_bytes >= required_bytes`

Operational recommendation for larger context windows:

- For `64k` context targets, plan for ~`1.30x` model-bytes headroom (about 30% extra) to leave practical KV/cache margin and avoid tight-runtime failures.
- The `1.10x` rule remains the baseline gate; `1.30x` is the recommended planning target for 64k operation.

Fit is evaluated only after capability-profile filtering chooses eligible variants.

### 8.1 Download

- Never blocked by local fit.
- If likely too large, print warning and continue:
  - `🟡 This model is likely too large for local serving on this machine.`

### 8.2 Serve / Load

- Non-MoE model not fitting local machine: hard fail.
- Message:
  - `🟡 <ref> is too large to serve on this machine (needs X GB, available Y GB).`

MoE exception:

- Over-capacity MoE models can proceed through MoE split path.

### 8.3 Examples

```text
$ mesh-llm models download unsloth/MiniMax-M2-GGUF/MiniMax-M2-Q8_0
🎯 capability: text
🟡 This model is likely too large for local serving on this machine.
📥 Ensuring unsloth/MiniMax-M2-GGUF/MiniMax-M2-Q8_0@main is available locally...
✅ Cached locally
```

```text
$ mesh-llm serve --model unsloth/MiniMax-M2-GGUF/MiniMax-M2-Q8_0
🟡 unsloth/MiniMax-M2-GGUF/MiniMax-M2-Q8_0 is too large to serve on this machine (needs 249.0 GB, available 96.0 GB).
```

## 9. Retry + Backoff Spec (Downloads)

This applies to both:

- explicit `models download`
- any serve/load path that triggers downloading

### 9.1 Parameters

- `max_attempts = 6` (attempt 1 + 5 retries)
- `base_delay = 1s`
- `max_delay = 30s`
- `jitter_factor = uniform(0.8, 1.2)`

### 9.2 Retryable Failures

- network/transient transport failures (timeout, reset, DNS/TLS transient)
- HTTP status: `429`, `500`, `502`, `503`, `504`

### 9.3 Non-Retryable Failures

- gated/terms-required
- explicit auth/access denial
- not found
- validation/parsing/integrity-class failures that require restart logic

### 9.4 Delay Function

- `raw = min(max_delay, base_delay * 2^(attempt-1))`
- `delay = raw * jitter_factor`

### 9.5 Logging

Format:

- `🟡 Download failed (<reason>). Retrying in <delay>s (attempt <n>/<max>).`

### 9.6 Example

```text
$ mesh-llm models download unsloth/MiniMax-M2-GGUF/MiniMax-M2-Q4_K_M
🎯 capability: text
📥 Downloading model MiniMax-M2-Q4_K_M
🟡 Download failed (HTTP 503). Retrying in 1.0s (attempt 2/6)
🟡 Download failed (timeout). Retrying in 2.1s (attempt 3/6)
✅ Cached locally
```

## 10. Recoverable Download Spec (HF-Compatible)

Behavior target aligns with `huggingface_hub` principles:

- resume by default when possible
- etag/version-scoped incomplete artifacts
- restart when resume is invalid
- `force` clears incomplete and restarts

### 10.1 Per-Asset State

- final path
- incomplete path (etag/version-scoped)
- metadata (etag, commit/revision, timestamp, expected size if known)
- lock file (single writer)

### 10.2 Recoverability Check

Given existing incomplete state, resume is allowed only if all pass:

1. remote identity matches local incomplete identity (etag/version)
2. partial size is sane (`<= expected_size` when known)
3. ranged request is accepted for current offset

If any check fails:

- `🟡 Existing partial download is not recoverable; clearing and restarting.`
- delete stale incomplete state
- restart from offset 0

### 10.3 Resume Protocol

If partial exists and is recoverable:

- request with `Range: bytes=<offset>-`

Response handling:

- `206`: valid resume, append
- `200` while resuming: range ignored, truncate and restart from 0

### 10.4 Integrity and Commit

After transfer:

1. verify expected size (and hash if available)
2. atomically move incomplete file to final path
3. persist metadata
4. release lock

### 10.5 Force Download

`--force-download` semantics:

- remove incomplete artifact first
- download from zero

### 10.6 Split GGUF

- Process per shard independently.
- Keep completed shards.
- Retry/resume/restart only failed shard(s).

### 10.7 Example (Backoff + Recoverable Resume, Detailed)

```text
$ mesh-llm models download unsloth/MiniMax-M2-GGUF/MiniMax-M2-Q4_K_M
🎯 capability: text
📥 Ensuring unsloth/MiniMax-M2-GGUF/MiniMax-M2-Q4_K_M@main is available locally...
📥 Resolved split GGUF: 3 files

📥 [1/3] Q4_K_M/MiniMax-M2-Q4_K_M-00001-of-00003.gguf
   Metadata: etag=2f4c... expected=46.1 GB
   Found partial: 18.3 GB
   Resuming at byte 19649499136 (attempt 1/6)
   🟡 Download interrupted (timeout). Retrying in 1.1s (attempt 2/6)
   Resuming at byte 24108732416 (attempt 2/6)
   ✅ Verified size 46.1 GB

📥 [2/3] Q4_K_M/MiniMax-M2-Q4_K_M-00002-of-00003.gguf
   Metadata: etag=9a71... expected=41.4 GB
   Found partial: 33.0 GB
   Server ignored range resume (HTTP 200 on ranged request)
   🟡 Existing partial download is not recoverable; clearing and restarting.
   Restarting from byte 0 (attempt 1/6)
   ✅ Verified size 41.4 GB

📥 [3/3] Q4_K_M/MiniMax-M2-Q4_K_M-00003-of-00003.gguf
   Metadata: etag=b83d... expected=41.3 GB
   Starting from byte 0 (attempt 1/6)
   🟡 Download failed (HTTP 503). Retrying in 1.0s (attempt 2/6)
   Resuming at byte 6.2 GB (attempt 2/6)
   🟡 Download failed (HTTP 503). Retrying in 2.2s (attempt 3/6)
   Resuming at byte 14.9 GB (attempt 3/6)
   ✅ Verified size 41.3 GB

✅ Cached locally
```
