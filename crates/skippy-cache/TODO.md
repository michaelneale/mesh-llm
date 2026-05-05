# skippy-cache TODO

## Completed In This PR

- Serving-path exact prefix cache is wired into local OpenAI serving:
  `ResidentKv` and `KvRecurrent` policies restore before decode, record after
  prefill, dedupe exact payloads, evict by entry/byte caps, and emit cache
  telemetry. `FullState` remains a correctness/certification payload only; it
  is not selected by production family policy.
- README benchmark evidence now has an explicit production-payload table.
  Every reviewed family with a local full GGUF is benchmarked with either
  `ResidentKv` or `KvRecurrent`; `FullState` rows are excluded from production
  performance evidence.
- Benchmark evidence now uses matched single-request settings:
  Skippy `--runtime-lane-count 1` and llama-server `--parallel 1`. Resident KV
  rows report measured cache footprint when native KV-page export is available;
  hybrid/SWA layouts that can resident-copy but cannot export KV pages for
  sizing are marked `TBD` instead of failing correctness.

## DeepSeek3 Exact-State Certification

DeepSeek3 is classified as its own topology family, but exact-state mobility is
still `Untested`. Before promoting it to accepted cache policy, run a real GGUF
state-handoff certification:

1. Run `skippy-correctness state-handoff` with `--state-payload-kind full-state`
   on a DeepSeek3 GGUF.
2. If full-state matches, run the same prompt with `--state-payload-kind
   resident-kv` or `kv-recurrent`, depending on the certified compact payload.
3. Record `matches`, payload size, import/export timings, and cache-hit timing
   in the cache benchmark table.
4. Promote only the tested model/ref, or the family if multiple representative
   refs pass.

Keep `q8_wire_validation` and `exact_state_mobility` as untested until this
evidence exists.

## Follow-Up Certification

- DeepSeek3 needs a local full GGUF before it can be benchmarked against
  llama-server. The local layer-shard package is enough for staged serving work
  but not for the matched llama-server baseline in
  `evals/skippy-cache-production-bench.py`.
- MiniMax M2.7 passes `ResidentKv`, and the exported recurrent component is
  zero bytes for the tested GGUF. Add a multi-token continuation certification
  and a compact recurrent-payload review before treating that as proof for every
  MiniMax-style recurrent variant.
- Extend the production benchmark to longer prompts and multi-token generation.
  The current smoke matrix shows dense-family parity/wins against llama-server
  warm slots, but longer prompts and generation lengths should be tracked before
  making release-level claims.
- Promote exact decoded-result caching from the correctness harness into serving:
  cache full-prompt state/logits for exact repeated prompts, return the cached
  first token without re-running the final prompt-token decode, and continue
  normal decode only when `max_tokens > 1`.

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
