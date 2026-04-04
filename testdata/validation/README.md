# Validation Testdata

This directory contains the checked-in data that drives local and CI validation
for the GGUF and MLX backends.

## Files

- `matrix.json`
  - The canonical model matrix.
  - Pins the exact GGUF and MLX artifacts to test.
  - Describes the model label, expectation class, and the exact/behavior case
    ids used by the runner.

- `baselines.json`
  - The checked-in expected results.
  - `GGUF` is the canonical baseline.
  - `MLX` baselines are tracked secondarily for backend self-consistency.

## Strategy

The validation system has two suites:

1. `exact`
   - Deterministic prompts such as `blue / green / red`
   - Used for strict backend parity and prompt-following checks

2. `behavior`
   - MT-Bench-derived prompts with heuristic health checks
   - Used to catch repetition, garbling, empty outputs, leaked reasoning,
     and timeout/liveness failures

## Baseline policy

- New `GGUF` runs are compared against the checked-in `GGUF` baseline.
- New `MLX` runs are compared against:
  - the checked-in `MLX` baseline for backend regression detection
  - the checked-in `GGUF` baseline for parity

Canonical baseline rule:

- `GGUF` is only promotable as the canonical baseline when the `strict` GGUF
  rows are clean for the suite being promoted.
- Weak canaries may remain weak, but they do not define what “good GGUF”
  means for the reference baseline.

This gives three useful comparisons:

1. `GGUF run` vs `GGUF baseline`
2. `MLX run` vs `MLX baseline`
3. `MLX run` vs `GGUF baseline`

## Expectation classes

- `strict`
  - Should remain clean and deterministic.

- `weak-but-stable`
  - Known tiny-model weirdness is tolerated if it remains stable.

- `informational`
  - Useful for tracking parity, but not a hard quality gate.

## Artifact drift

Avoid changing `matrix.json` casually.

If the artifact under test changes, update the pinned ref explicitly and treat
that as a baseline change, not a routine rerun.

## Behavior baselines

Behavior baselines should stay summary-based rather than full-output goldens.

Prefer recording:

- exit code
- failed prompt count
- flagged prompt ids/categories

Do not check in large generated outputs as the baseline.
