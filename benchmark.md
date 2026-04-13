# Plan: add `mesh-llm gpu benchmark`

## Goal

Add a CLI command at `mesh-llm gpu benchmark` that forces a fresh benchmark run on the current platform and rewrites `~/.mesh-llm/benchmark-fingerprint.json`.

## Proposed approach

### 1. Change the GPU CLI shape

Update the existing GPU command from a bare top-level variant into a real subcommand surface.

- Today, `mesh-llm/src/cli/mod.rs` defines `Command::Gpus` as a bare variant with the alias `gpu`.
- Change that to a subcommand-bearing variant so `mesh-llm gpu benchmark` becomes valid.
- Add a new `GpuCommand` enum for GPU-specific actions.

Expected command shape:

- `mesh-llm gpus` — keep existing GPU inspection behavior
- `mesh-llm gpu benchmark` — force rerun benchmark and rewrite cache

Optional compatibility decision during implementation:

- either preserve bare `mesh-llm gpu` as an alias for listing GPUs
- or require an explicit listing subcommand such as `mesh-llm gpu list`

## Files to change

### `mesh-llm/src/cli/mod.rs`

- Replace the bare `Command::Gpus` variant with a subcommand-bearing form.
- Add a `GpuCommand` enum.
- Keep user-facing help text concise and consistent with nearby commands.

### `mesh-llm/src/cli/commands/mod.rs`

- Change dispatch from direct `run_gpus()` invocation to a GPU command dispatcher.
- Route `gpu benchmark` to a dedicated handler.

### `mesh-llm/src/cli/commands/gpus.rs`

- Keep `run_gpus()` for the current read-only inspection path.
- Add something like `dispatch_gpu_command()`.
- Add `run_gpu_benchmark()` to perform the forced benchmark flow and print a short result summary.

### `mesh-llm/src/system/benchmark.rs`

- Add a helper for a forced rerun path.
- Do not reuse `run_or_load()` unchanged, because it prefers the cache when hardware matches.

Best implementation options:

1. Add a helper like `run_and_save(...)` that always:
   - detects the benchmark binary
   - runs it
   - builds the result
   - writes `benchmark-fingerprint.json`
2. Or extend `run_or_load(...)` with a force flag and bypass:
   - `load_fingerprint()`
   - `hardware_changed()`

Preferred direction: extract a dedicated helper rather than overloading `run_or_load()` too much, so runtime startup and explicit CLI forcing stay easy to reason about.

## Desired CLI behavior

`mesh-llm gpu benchmark` should:

1. Survey current hardware.
2. Exit cleanly with a clear message if no GPUs are present.
3. Detect the correct platform-specific benchmark binary.
4. Run the benchmark with the existing timeout behavior.
5. Atomically rewrite `~/.mesh-llm/benchmark-fingerprint.json`.
6. Print a short success summary including:
   - GPU count
   - total measured bandwidth
   - fingerprint cache path

## Important constraints

- Keep `mesh-llm gpus` read-only.
- Do not silently reuse the cache for `gpu benchmark`.
- Reuse existing benchmark binary discovery and parsing logic.
- Preserve current atomic write behavior using the temp-file-plus-rename path.
- Surface soft failures clearly to the user.

## Edge cases to handle

### No GPU present

- Current runtime and benchmark code already short-circuit when `gpu_count == 0`.
- The new CLI should print a clear user-facing message and avoid writing a new fingerprint file.

### Missing benchmark binary

- `detect_benchmark_binary()` can return `None` for unsupported or missing platform binaries.
- The CLI should report that explicitly instead of failing silently.

### Benchmark timeout

- Reuse existing timeout behavior from `run_benchmark()`.
- If the benchmark times out or exits unsuccessfully, do not claim success and do not present stale results as fresh.

### Parse failures or invalid output

- If benchmark JSON is empty, malformed, or reports an error object, surface that as a benchmark failure.

### Existing cache present

- The command should overwrite the existing fingerprint file with a newly generated one.
- The force path must bypass normal cache reuse.

## Validation plan

### Unit tests

Add or extend tests in `mesh-llm/src/system/benchmark.rs` to cover:

- forced rerun path bypasses cache reuse
- forced path rewrites the fingerprint file
- no-GPU path does not write a new cache
- missing-binary path fails cleanly

### CLI verification

Verify manually or with command-level tests that:

- `mesh-llm gpus` still shows current GPU inspection output
- `mesh-llm gpu benchmark` is accepted by clap
- `mesh-llm gpu benchmark` rewrites the fingerprint file even when one already exists
- error cases produce clear output

### Regression checks

- Ensure startup benchmarking still uses the normal cached path.
- Ensure existing `gpus` output formatting remains unchanged unless intentionally improved.

## Summary

This should be implemented as a small CLI expansion plus a focused benchmark helper in `system/benchmark.rs`. The key design requirement is that `mesh-llm gpu benchmark` must bypass cache reuse and always regenerate and rewrite `benchmark-fingerprint.json`, while the existing `mesh-llm gpus` path remains a read-only inspector of cached data.
