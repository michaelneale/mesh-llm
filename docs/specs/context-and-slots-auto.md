# Spec: Improved Heuristics for Context Size and Parallel Slots in `--auto`

## Background

When `mesh-llm serve --auto` launches llama-server, it picks two key resource parameters:

1. **Context size (`-c`)** — token budget for all active sequences combined.
2. **Parallel slots (`--parallel`)** — number of concurrent decode streams.

### Current state

**Context size** (`compute_context_size` in `inference/launch.rs`):

```rust
fn compute_context_size(ctx_size_override, model_bytes, my_vram, total_group_vram) -> u32 {
    let vram_after_model = my_vram.saturating_sub(host_model_bytes);
    if vram_after_model >= 30 * GB { 65536 }
    else if vram_after_model >= 12 * GB { 32768 }
    else if vram_after_model >= 6 * GB  { 16384 }
    else if vram_after_model >= 3 * GB  { 8192 }
    else                                { 4096 }
}
```

The thresholds are VRAM-only and completely ignore:
- The model's native max context length (`GgufCompactMeta::context_length`)
- The number of attention heads and head dimension (which drives per-token KV byte cost)
- KV cache quantization in use (Q4/Q8/F16 — already computed separately)
- Number of parallel slots (more slots × context = more KV)

**Parallel slots** (`runtime/mod.rs`):

```rust
let slots = primary_startup_model
    .and_then(|m| m.parallel)
    .or(config.gpu.parallel)
    .unwrap_or(4);  // ← hardcoded default
```

The default of 4 is a fixed number with no relationship to available VRAM, context size chosen, or model size. A node with 80 GB free VRAM gets the same 4 slots as a node with 4 GB free.

---

## Problems

### Context size

1. **Ignores model's architectural max.** A model trained with a 4 096-token context window cannot usefully run at 65 536. Capping at `min(computed, gguf.context_length)` would be free correctness.

2. **Ignores KV cost per token.** The actual bytes consumed per token per layer is:
   ```
   kv_bytes_per_token = 2 × n_kv_heads × head_dim × bytes_per_element × n_layers
   ```
   With GQA and Q4/Q8 KV quantization, this can be 4–8× cheaper than naive F16 assumptions. A model with 8 KV heads at Q4 has vastly different headroom than a full-attention F16 model of the same size.

3. **VRAM thresholds are not anchored to real KV math.** The `30 GB → 65 536` bracket was hand-tuned for a specific generation of models (likely 7-13B dense). A 70B model filling 30 GB free VRAM has 4× more layers and larger heads, so 65 536 tokens will OOM where the heuristic says it won't.

4. **Does not clamp to a power of two or a model-supported max.** Some models have native rope limits; exceeding them requires rope scaling which degrades quality silently.

### Parallel slots

1. **Hardcoded 4 is both too low and too high depending on hardware.** A 96 GB node serving a 7B model has room for 16+ concurrent streams. A 4 GB node serving a 3B model at 8 192 ctx barely has room for 2 without thrashing.

2. **Slots × context = total KV budget.** Context size and slots are not independent. Today `compute_context_size` has no idea how many slots will be requested, so it can accidentally allocate all remaining VRAM to context while the slot count then multiplies the KV footprint.

3. **No relationship to expected concurrency.** A node on the public mesh expects more concurrent callers than a private single-user node. The `--auto` path knows it is joining the mesh and could default higher.

---

## Proposed Design

### 1. Expose KV byte cost from `GgufCompactMeta`

K and V caches are **separate allocations** with potentially different element widths and
— importantly — different tensor dimensions (`key_length` vs `value_length` are
independent GGUF fields). Any helper that folds both into a single scalar loses the
ability to correctly price asymmetric quantization pairs such as Q8K/Q4V.

The fix is to compute K cost and V cost independently and sum them. Add helpers to
`models/gguf.rs`:

```rust
impl GgufCompactMeta {
    /// Bytes consumed by the K cache per token across all layers under f16.
    /// Returns None if required metadata fields are zero / missing.
    pub fn k_cache_bytes_per_token_f16(&self) -> Option<u64> {
        if self.layer_count == 0 || self.key_length == 0 { return None; }
        // GQA: kv_heads ≤ q_heads. Fall back to full head_count if the
        // head_count_kv field is not present.
        let kv_heads = self.head_count_kv.unwrap_or(self.head_count) as u64;
        Some(kv_heads * self.key_length as u64 * 2 /* f16 */ * self.layer_count as u64)
    }

    /// Bytes consumed by the V cache per token across all layers under f16.
    pub fn v_cache_bytes_per_token_f16(&self) -> Option<u64> {
        if self.layer_count == 0 || self.value_length == 0 { return None; }
        let kv_heads = self.head_count_kv.unwrap_or(self.head_count) as u64;
        Some(kv_heads * self.value_length as u64 * 2 /* f16 */ * self.layer_count as u64)
    }
}
```

`head_count_kv` should be added as an `Option<u32>` field read from
`.attention.head_count_kv` — it is already parsed locally in `scan_gguf_compact_meta`
but not stored in the struct.

`KvCacheQuant` gets a method that takes the compact meta and returns the **actual
quantized bytes per token**, keeping K and V costs separate until the final sum:

```rust
impl KvCacheQuant {
    /// Quantized KV bytes per token across all layers.
    /// Returns None when the metadata is too incomplete to estimate.
    pub fn kv_bytes_per_token(&self, meta: &GgufCompactMeta) -> Option<u64> {
        let k_f16 = meta.k_cache_bytes_per_token_f16()?;
        let v_f16 = meta.v_cache_bytes_per_token_f16()?;
        let k_bytes = apply_quant_scale(k_f16, self.k_type);
        let v_bytes = apply_quant_scale(v_f16, self.v_type);
        Some(k_bytes + v_bytes)
    }
}

fn apply_quant_scale(f16_bytes: u64, t: KvType) -> u64 {
    match t {
        KvType::F16  => f16_bytes,           // 2 bytes/elem, scale 1.0
        KvType::Q8_0 => f16_bytes / 2,       // 1 byte/elem,  scale 0.5
        KvType::Q4_0 => f16_bytes / 4,       // 0.5 byte/elem, scale 0.25
    }
}
```

**Why separate K/V matters for asymmetric pairs:**

| Pair | K scale | V scale | Blended if key_len≠val_len |
|---|---|---|---|
| F16 / F16 | 1.0 | 1.0 | n/a, trivially correct |
| Q8_0 / Q8_0 | 0.5 | 0.5 | n/a, trivially correct |
| Q4_0 / Q4_0 | 0.25 | 0.25 | n/a, trivially correct |
| **Q8_0 / Q4_0** (medium tier) | **0.5** | **0.25** | total = K×0.5 + V×0.25, **not** (K+V)×0.375 when key_len≠val_len |

The medium-tier asymmetric pair (Q8K/Q4V) is the case that breaks a simple average.
For models where `key_length == value_length` the difference is numerically zero, but
that equality is architecture-specific — e.g. some architectures use longer K projections
— so the correct formula costs each tensor independently regardless.

`head_count_kv` should be added as an optional field read from `.attention.head_count_kv` — it is already parsed locally in `scan_gguf_compact_meta` but not stored in the struct. Promoting it to a named field completes the picture.

### 2. Revised `compute_context_size`

Replace the VRAM-threshold ladder with a capacity-driven formula. Signature expands to accept `GgufCompactMeta`:

```rust
fn compute_context_size(
    ctx_size_override: Option<u32>,
    model_bytes: u64,
    my_vram: u64,
    total_group_vram: Option<u64>,
    meta: Option<&GgufCompactMeta>,  // new
    kv_quant: KvCacheQuant,          // new — already computed before this call
    slots: usize,                    // new — so context is co-planned with slots
) -> u32
```

**Algorithm:**

```
1. If ctx_size_override is set, return it (existing behaviour, no change).

2. Compute vram_after_model (existing split/local logic, no change).

3. Estimate the quantized KV byte cost per (token × slot):
     kv_bytes_per_token = kv_quant.kv_bytes_per_token(meta)
     This computes K and V costs independently using their actual quantized widths
     (see §1), then sums them. Asymmetric pairs such as Q8K/Q4V are correctly priced
     without assuming key_length == value_length.
     If meta is None (scan failed, shard, etc.), fall back to a conservative estimate:
       kv_bytes_per_token = 512 bytes  (safe for most 7–13B F16 models)

4. Reserve a headroom fraction for activations and system overhead:
     usable_vram = vram_after_model × 0.85

5. Solve for context size given the slot count:
     ctx_size = usable_vram / (kv_bytes_per_token × slots)

6. Snap to the nearest power-of-two step from the set
   {512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072}
   rounding down.

7. Clamp to [MIN_CTX, model_native_max]:
     MIN_CTX = 512
     model_native_max = meta.context_length if > 0, else 131072

8. Return the clamped value.
```

The `compression_factor()` concept is **not introduced**. K and V costs are always
computed separately via `kv_quant.kv_bytes_per_token(meta)` as defined in §1. This
avoids the implicit assumption that K and V tensors have equal element counts, which
does not hold when `key_length ≠ value_length`.

### 3. Revised parallel slots heuristic

Replace the `unwrap_or(4)` default with a function that co-plans slots with context:

```rust
fn default_parallel_slots(
    vram_after_model: u64,
    ctx_size: u32,          // the context size already chosen
    kv_bytes_per_token: u64, // from kv_quant.kv_bytes_per_token(meta), or fallback constant
) -> usize
```

**Algorithm:**

```
1. usable_vram = vram_after_model × 0.85

2. total_kv_budget = usable_vram

3. kv_per_slot = ctx_size × kv_bytes_per_token
     where kv_bytes_per_token = kv_quant.kv_bytes_per_token(meta)  (K and V priced separately)

4. raw_slots = total_kv_budget / kv_per_slot

5. Clamp raw_slots to [1, MAX_SLOTS]:
     MAX_SLOTS = 32   (hard ceiling — beyond this the scheduler overhead dominates)

6. Snap down to the nearest value in {1, 2, 4, 8, 16, 32}.

7. Return the result.
```

Caller site in `runtime/mod.rs` changes from:

```rust
let slots = primary_startup_model
    .and_then(|m| m.parallel)
    .or(config.gpu.parallel)
    .unwrap_or(4);
```

to:

```rust
let slots = primary_startup_model
    .as_ref()
    .and_then(|m| m.parallel)
    .or(config.gpu.parallel)
    .unwrap_or_else(|| {
        let vram_after = my_vram.saturating_sub(host_model_bytes);
        let kv_bpt = kv_quant.kv_bytes_per_token(meta.as_ref()).unwrap_or(512);
        inference::launch::default_parallel_slots(vram_after, ctx_size, kv_bpt)
    });
```

`ctx_size` must be computed first, then `slots` derived from it, so the two are consistent.

### 4. Co-planning order

The two values must be computed in a defined order to avoid circular dependency:

```
1. Compute kv_bytes_per_token from GGUF metadata (or fallback).
2. Compute a preliminary ctx_size using a provisional slot count of 1
   to find the maximum single-slot headroom.
3. Compute slots from that headroom and the chosen ctx_size.
4. Recompute ctx_size using the final slot count.
   (One iteration is sufficient; the relationship is monotone.)
```

Alternatively, pick a target utilisation: allocate 70% of VRAM budget to context and 30% to multi-slot overhead. This avoids iteration entirely:

```
kv_bpt      = kv_quant.kv_bytes_per_token(meta).unwrap_or(512)
ctx_budget  = usable_vram × 0.70 / kv_bpt     // largest single-slot context
slot_budget = usable_vram × 0.30 / (ctx_budget × kv_bpt)  // remaining headroom → slots
```

`kv_bpt` already encodes the full quantization picture (K type, V type, GQA heads,
per-head dimensions) so the same formula applies for all three tiers — F16/F16,
Q8K/Q8V, Q8K/Q4V — without any special-casing.

The split approach is simpler to implement and test.

---

## Fallback Behaviour

When GGUF metadata is unavailable (scan failed, file not yet downloaded, split-model shard):

- Use the existing VRAM-threshold ladder for context size (preserves current behaviour for the unknown case).
- Use `slots = max(1, vram_after_model / (4096 × 512))` clamped to `[1, 8]` as the slot default.

This ensures the new logic is strictly opt-in based on data availability and never regresses existing behaviour.

---

## Affected Call Sites

| File | Change |
|---|---|
| `inference/launch.rs` | `compute_context_size` gains `meta`, `kv_quant`, `slots` params. New `default_parallel_slots` function. No `compression_factor` — K/V costs computed separately. |
| `models/gguf.rs` | `GgufCompactMeta` gains `head_count_kv: Option<u32>` field. New `k_cache_bytes_per_token_f16()` and `v_cache_bytes_per_token_f16()` methods. `scan_gguf_compact_meta` stores `head_count_kv`. |
| `inference/launch.rs` (KvCacheQuant) | New `kv_bytes_per_token(meta)` method on `KvCacheQuant` that prices K and V independently via `apply_quant_scale`. Replaces any future use of a `compression_factor` average. |
| `runtime/mod.rs` | `slots` computed via new function rather than `unwrap_or(4)`. Both primary (line ~1940) and extra-model (line ~2111) sites updated identically. |
| `inference/election.rs` | `ModelLaunchSpec` and callsites: `slots` is still passed in as a `usize`, no structural change; only the value fed in changes. |

---

## Tests to Add / Update

- `KvCacheQuant::kv_bytes_per_token` — unit tests covering all six meaningful combinations:
  - Symmetric F16/F16: result == k_f16 + v_f16 (both at full width)
  - Symmetric Q8_0/Q8_0: result == (k_f16 + v_f16) / 2
  - Symmetric Q4_0/Q4_0: result == (k_f16 + v_f16) / 4
  - Asymmetric Q8_0/Q4_0 (medium tier), key_len == val_len: result == k_f16/2 + v_f16/4
  - Asymmetric Q8_0/Q4_0, key_len ≠ val_len: result must **not** equal (k_f16+v_f16)×0.375
    — this is the case that breaks the averaging approach
  - meta with head_count_kv present (GQA): verify kv_heads used, not head_count

- `compute_context_size` — add cases for:
  - GQA model (8 KV heads, 32 Q heads) with symmetric Q8 KV: expect higher ctx than F16 baseline
  - Asymmetric Q8K/Q4V: result must be between symmetric Q8 and symmetric Q4 results,
    and must match the hand-computed K+V sum (not the average)
  - context_length clamp: model with native max 4096 must never return > 4096
  - Fallback (meta = None): must produce same output as current VRAM-threshold ladder

- `default_parallel_slots` — add cases for:
  - 80 GB free, 7B model, symmetric Q4/Q4: expect ≥ 8 slots
  - 80 GB free, 7B model, asymmetric Q8K/Q4V: expect fewer slots than symmetric Q4/Q4
    (K is more expensive under asymmetric pair)
  - 4 GB free, 3B model, F16/F16: expect 1–2 slots
  - config override still wins

- Update `model_launch_spec_slots_defaults_to_runtime_value` — the comment about `unwrap_or(4)` will no longer reflect reality; update accordingly.

---

## Non-Goals

- Does not change the `--ctx-size` / `--parallel` config file override path. Explicit user values always win.
- Does not attempt to dynamically resize context or slots at runtime (that would require llama-server restart).
- Does not model CPU-offload memory separately. CPU RAM is ignored for now; a future improvement could count it.
- Does not change MoE expert-shard launch, which has separate resource accounting.
