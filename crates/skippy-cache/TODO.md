# skippy-cache TODO

## Serving-Path Exact Prefix Cache

The exact-state ABI and runtime wrappers exist, and Falcon-H1 has a passing
`KvRecurrent` handoff benchmark. The production OpenAI serving path still needs
to use those pieces instead of limiting exact-state cache work to correctness
tools.

Implement in `skippy-server`:

1. Build an exact prefix identity from model id, tokenizer/template, topology,
   stage/layer range, runtime ABI, KV dtype/layout, ctx/position config, token
   start, and token ids.
2. Use family policy to decide whether cache is disabled, resident KV,
   `KvRecurrent`, or `FullState`.
3. On lookup hit, import the cached payload into the request lane:
   - `FullState` -> `import_full_state`
   - `KvRecurrent` -> `import_kv_page` then `import_recurrent_state`
   - resident KV -> existing resident prefix restore path
4. Decode from the restored prefix state without recomputing the prefix.
5. On miss, prefill normally, export the selected payload, dedupe/store it, and
   record telemetry for bytes, import/export time, hit/miss, and evictions.
6. Keep unknown or uncertified families disabled until correctness evidence
   exists.

Add per-model caps for entries and bytes, and make cache policy visible in
runtime status/telemetry so we can debug live deployments.

## README Benchmark Table Evidence

The `skippy-cache` README table must be evidence-backed across model families,
not inferred from architecture alone. Build the table from repeatable
single-stage cache tests first, then add split-stage results separately.
DeepSeek3 is only one open certification item; every family bucket we claim in
the README or enable by default needs its own correctness and benchmark
evidence.

Benchmark protocol:

1. Pick one reviewed GGUF per family bucket:
   - dense attention: Llama, Qwen3 dense, DeepSeek2, GLM4, OLMo, MiniMax-M2.7
   - full-state bucket: Gemma2, Gemma3, Gemma4 A4B/E4B, GLM-4.7-Flash
   - recurrent/hybrid: Falcon-H1, Qwen3Next
   - newly classified/untested: DeepSeek3
2. For each family bucket, record the certification state before benchmarking:
   - already proven: Falcon-H1 `KvRecurrent` on
     `tiiuae/Falcon-H1-1.5B-Instruct-GGUF:Q4_K_M`
   - needs certification: Qwen3Next recurrent/hybrid
   - needs certification: DeepSeek3
   - needs certification: Gemma2/3/4 and GLM-4.7-Flash `FullState`
   - needs certification for README speedup claims: Llama, Qwen3 dense,
     DeepSeek2, GLM4, OLMo, MiniMax-M2.7
   - disabled: unknown families until explicitly certified
3. For each model, create a deterministic prompt that tokenizes to the same
   prefix length in llama-server and skippy. Start with 512 prefix tokens and
   repeat at 2k/8k where the model and hardware fit.
4. Run llama-server baseline:
   - same GGUF, ctx size, GPU layers/device, KV dtype, flash-attention setting,
     and one generated token
   - send the same prompt repeatedly with prompt cache enabled
   - record cold wall time, warm wall time, prompt eval tokens, prompt eval ms,
     and whether llama-server reused or reprocessed the prefix
5. Run `skippy-correctness state-handoff`:
   - `full-state` for every family as the reference exact payload
   - `kv-recurrent` for dense and hybrid families where policy wants
     KV-backed replay
   - `recurrent-only` only as a diagnostic; do not use it as the README success
     path unless the model is genuinely recurrent-only and proven exact
6. Verify pass/fail:
   - `matches = true`
   - source/restored predicted token match
   - activation/output payloads match when available
   - repeated cache-hit imports remain stable
7. Record cache economics:
   - total payload bytes
   - KV bytes and recurrent bytes where applicable
   - BLAKE3 block count, unique block count, duplicate block count
   - export time, import time, decode time, hit total time
8. Produce the README table with columns:
   - family
   - representative model ref
   - payload used
   - llama-server warm repeat
   - skippy cache hit
   - speedup
   - payload size
   - status: accepted / untested / rejected
   - notes
9. Only promote `exact_state_mobility` or family cache defaults after at least
   one representative model passes. Prefer model-ref-level promotion first when
   evidence is thin.

Keep raw JSON reports under `/tmp` while iterating, then copy summarized numbers
into README/docs only. Do not commit large benchmark artifacts.

## DeepSeek3 Exact-State Certification

DeepSeek3 is classified as its own topology family, but exact-state mobility is
still `Untested`. Before promoting it to accepted cache policy, run a real GGUF
state-handoff certification:

1. Run `skippy-correctness state-handoff` with `--state-payload-kind full-state`
   on a DeepSeek3 GGUF.
2. If full-state matches, run the same prompt with `--state-payload-kind
   kv-recurrent`.
3. Record `matches`, payload size, import/export timings, and cache-hit timing
   in the cache benchmark table.
4. Promote only the tested model/ref, or the family if multiple representative
   refs pass.

Keep `q8_wire_validation` and `exact_state_mobility` as untested until this
evidence exists.

## Cache Optimizations To Try

These are follow-on experiments from the cache literature pass. Keep them behind
feature/config flags until they have correctness tests, benchmark numbers, and
clear failure behavior.

1. Radix prefix index:
   - Replace or augment flat exact-prefix lookup with a token radix tree.
   - Return the longest exact prefix hit, not only full-prompt hits.
   - Track model ref, tokenizer/template, topology, layer range, ABI, KV config,
     and position config at each cacheable node.
   - Measure lookup overhead and reuse rate against the current BLAKE3 exact-key
     path.
   - Expected outcome: repeated prompts with shared long prefixes hit earlier
     prefixes instead of falling back to full recompute.
   - Exit criteria: accept if longest-prefix reuse improves benchmark hit rate
     or p50 latency by at least 10% with less than 1% lookup overhead; reject if
     memory overhead or invalidation complexity makes flat exact keys simpler
     and faster.
2. Page-aligned KV/state chunks:
   - Store cache payloads in page-sized chunks aligned with llama/skippy KV
     pages where possible.
   - Deduplicate at the chunk/page layer instead of one monolithic state blob.
   - Measure import/export latency, duplicate block rate, and memory
     fragmentation.
   - Expected outcome: less copying and better dedupe for prompts that share
     prefix pages.
   - Exit criteria: accept if physical cache bytes drop by at least 20% or
     import/export latency drops by at least 10% without correctness fallout;
     reject if page bookkeeping costs more than block-level BLAKE3 dedupe saves.
3. Hot/warm/cold cache tiers:
   - Keep hot exact states resident in GPU or runtime memory.
   - Keep warm deduped payloads in host memory.
   - Optionally spill cold payloads to disk only for benchmark exploration; do
     not put serialized disk restore on the serving hot path by default.
   - Add per-model byte caps and eviction telemetry before enabling this in
     normal serving.
   - Expected outcome: high-value entries stay fast while total cache size is
     bounded per model.
   - Exit criteria: accept hot/warm tiers if p95 latency stays stable under
     eviction pressure and byte caps are honored; keep disk cold tier
     benchmark-only unless it beats recompute for large prefixes by at least 2x.
4. Async prefetch and staged restore:
   - Begin importing the best prefix hit while OpenAI request parsing and route
     setup continue.
   - For split serving, prefetch downstream stage state before stage 0 finishes
     frontend setup.
   - Benchmark whether prefetch hides import latency or simply adds scheduling
     contention.
   - Expected outcome: restore time is overlapped with request setup and
     downstream stage readiness.
   - Exit criteria: accept if p50/p95 cache-hit latency improves by at least 10%
     without increasing miss latency or lock contention; reject if prefetch makes
     misses or concurrent requests noisier.
5. Batched cache movement:
   - Batch export/import of KV pages, recurrent state, and full-state payloads
     into fewer ABI calls.
   - Prefer contiguous memory movement when the runtime can expose it.
   - Measure CPU overhead and tail latency for concurrent lane restores.
   - Expected outcome: fewer ABI crossings and better tail latency when many
     lanes restore cache entries concurrently.
   - Exit criteria: accept if cache-hit p95 improves by at least 10% or CPU time
     per restore drops measurably; reject if batching adds latency for small
     payloads or complicates error recovery.
6. Chunk-aware prefix reuse:
   - Try chunked prefix matching for very long prompts so partial reuse can
     happen at chunk boundaries.
   - Validate that restored position/state is exactly equivalent before decode.
   - Compare against the radix longest-prefix hit path.
   - Expected outcome: very long prompts can reuse a safe chunk boundary even
     when the full prefix is not cached.
   - Exit criteria: accept only if correctness matches full recompute for every
     tested family and long-context benchmarks beat radix-only lookup; reject if
     boundary state is ambiguous or family-specific.
7. Cold-tier compression:
   - Try lossless compression for cold exact payloads after BLAKE3 dedupe.
   - Keep hot and warm cache hits uncompressed unless measurements show import
     still wins.
   - Record compression ratio, compression time, decompression time, and net hit
     latency.
   - Expected outcome: larger cold caches fit within the same byte budget
     without affecting hot path latency.
   - Exit criteria: accept only for cold entries if compressed restore still
     beats recompute and saves at least 25% physical bytes; reject for hot/warm
     entries unless decompression overhead is negligible.
8. Approximate KV compression and eviction:
   - Explore H2O/SnapKV-style approximate retention only as a separate
     non-exact mode.
   - Never mix approximate entries into the exact cache namespace.
   - Require output-quality benchmarks before considering serving use.
   - Expected outcome: optional approximate mode may trade exactness for larger
     effective context/cache capacity.
   - Exit criteria: keep disabled unless quality benchmarks show acceptable
     degradation for explicit opt-in workloads; reject as a default path because
     exact cache correctness must remain binary.
9. Non-prefix document/RAG reuse:
   - Investigate CacheBlend-style reuse for shared documents that appear inside
     different prompts.
   - Treat this as a separate cache mode because it is not exact token-prefix
     restore.
   - Start with correctness tooling only, not the OpenAI serving path.
   - Expected outcome: document-heavy RAG workloads reuse shared context even
     when the user prompt changes around it.
   - Exit criteria: accept only as an explicit experimental mode if outputs are
     validated against recompute and latency beats normal prefill on real RAG
     traces; reject for the exact prefix serving path.
10. Native MLA/latent-state handling:
   - For DeepSeek3 and other MLA families, identify whether the backend exposes
     native latent KV/state or only expanded KV-compatible state.
   - Certify exact replay with the actual exported representation before
     enabling cache policy.
   - Document whether payload economics differ from normal dense attention KV.
   - Expected outcome: MLA families get the smallest exact replay payload the
     backend can safely expose.
   - Exit criteria: accept family policy only after state-handoff correctness
     passes on representative GGUFs and payload/timing data is recorded; leave
     DeepSeek3/MLA cache disabled if native latent state cannot be proven exact.
