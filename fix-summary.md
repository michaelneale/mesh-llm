# Fix Summary: `ggml_backend_load_all()` Guard

**Branch:** `codex/skippy-device-gpu-survey`  
**Commit:** `f1e1b91f`  
**Base:** `57380e3b`  

---

## Problem

When a second model loaded (e.g. the 4B following the 27B), `ggml_backend_load_all()` was called unconditionally in three places inside `skippy.cpp`:

1. `skippy_model_open` — called per model open
2. `skippy_model_open_from_parts` — called per staged/split model open
3. `skippy_enumerate_backend_devices` — called on GPU device survey

Each call re-registered all backends (CUDA, CPU, etc.) from scratch, even when they were already registered from a previous call. This caused redundant initialization work on every model load after the first.

---

## Root Cause

`ggml_backend_load_all()` in llama.cpp's backend registry has no idempotency guard — repeated calls re-register all backends. The three `skippy.cpp` call sites had no guard against this.

---

## Fix

Added a registration count check before each `ggml_backend_load_all()` call:

```cpp
if (!ggml_backend_reg_count()) {
    ggml_backend_load_all();
}
```

Applied in two patch files:
- `third_party/llama.cpp/patches/0032-Expose-selected-backend-device-ABI.patch` — guards in `skippy_model_open` and `skippy_model_open_from_parts`
- `third_party/llama.cpp/patches/0080-Expose-skippy-backend-device-enumeration-ABI.patch` — guard in `skippy_enumerate_backend_devices`

---

## Benchmark Results

Timed on a clean `just release-build-cuda-blackwell` build (8 CUDA architectures: sm75→sm120a) from a fully wiped `target/` and `.deps/`, measured from first log event to `model_loaded`.

| Model | Baseline (`3cc524b8`) | Fixed (`f1e1b91f`) | Delta |
|-------|----------------------|---------------------|-------|
| Qwen3.6-27B-UD-Q4_K_XL (RTX 5090) | 11.4s | **10.4s** | −1.0s |
| Qwen3.5-4B-UD-Q4_K_XL (RTX 3080) | 17.7s | **16.5s** | −1.2s |
| Runtime ready (wall clock) | 17.7s | **16.5s** | −1.2s |

No regression. Marginal improvement on both models across a fully clean build.
