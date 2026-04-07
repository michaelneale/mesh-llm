# MLX Validation Matrix

Local-first backend-parity ledger for model families. The point is not to judge
MLX in isolation; it is to compare `🦙 GGUF` against `🍎 MLX` on the same family /
model / case so we can tell shared model weakness from MLX-specific regressions.

## Legend

| Status | Meaning |
|---|---|
| `PASS` | Validated locally and behaved acceptably for the checks listed |
| `FAIL` | Reproduced a real issue locally |
| `PARTIAL` | Loads and answers basic prompts, but has behavior issues or incomplete coverage |
| `BLOCKED` | Could not be validated locally on this machine |
| `PENDING` | Not checked yet |

## GGUF Parity

| Status | Meaning |
|---|---|
| `MATCH` | GGUF showed the same behavior, so the issue is likely not MLX-specific |
| `DIFFERS` | GGUF and MLX both ran, but they diverged in ways that need source-model context to interpret |
| `MLX WORSE` | GGUF handled the same case better than MLX |
| `MLX BETTER` | MLX handled the same case better than GGUF |
| `PENDING` | GGUF comparison not run yet |
| `BLOCKED` | Could not get a meaningful GGUF comparison locally |

## Pair Quality

| Status | Meaning |
|---|---|
| `HIGH` | Same family, same size, same instruct/chat target, and close quant class; good parity signal |
| `MEDIUM` | Same family and roughly same target, but quant or conversion path differs materially |
| `LOW` | Only approximate family parity; useful for triage, but not a strong apples-to-apples comparison |
| `PENDING` | Pair quality not assessed yet |

## Models

| Family | Model Pair | GGUF Target | MLX Target | Pair Quality | Last Checked | GGUF Exact | MLX Exact | GGUF Behavior | MLX Behavior | Parity | Status | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5 | 0.5B instruct | `meshllm/qwen2.5-0.5b-instruct-parity-q8_0-gguf/qwen2.5-0.5b-instruct-q8_0.gguf` | `meshllm/qwen2.5-0.5b-instruct-parity-8bit-mlx` | `HIGH` | 2026-04-06 | `PASS` | `PASS` | `STALE` | `STALE` | `MATCH` | `PARTIAL` | Published same-origin parity pair derived from `Qwen/Qwen2.5-0.5B-Instruct`. Local exact validation passed on both backends with matching outputs across the full checked-in exact suite, including `after-monday -> Tuesday`. Older behavior numbers were collected against the public pair before the Qwen2.5 MLX template-rendering fix, so behavior should be rerun on this canonical pair before drawing new parity conclusions. |
| Qwen3 | 0.6B instruct | `meshllm/qwen3-0.6b-parity-q8_0-gguf/qwen3-0.6b-q8_0.gguf` | `meshllm/qwen3-0.6b-parity-8bit-mlx` | `HIGH` | 2026-04-06 | `FAIL` | `FAIL` | `STALE` | `STALE` | `DIFFERS` | `PARTIAL` | Published same-origin parity pair derived from `Qwen/Qwen3-0.6B`. Local exact validation still fails on both backends, but the important result is that MLX now matches the original checkpoint behavior while GGUF drifts from it on multiple prompts (`after-monday`, `banana-color`, `largest-planet`). This row remains useful for backend-drift tracking, but Qwen3 is a weak parity canary and the old behavior numbers should not be carried forward to the new canonical pair. |
| Llama | 3.2 1B instruct | `meshllm/llama-3.2-1b-instruct-parity-f16-gguf/llama-3.2-1b-instruct-f16.gguf` | `meshllm/llama-3.2-1b-instruct-parity-bf16-mlx` | `HIGH` | 2026-04-06 | `FAIL` | `FAIL` | `PENDING` | `PENDING` | `MATCH` | `PARTIAL` | Published same-origin high-fidelity parity pair derived from `meta-llama/Llama-3.2-1B-Instruct`. Local exact validation shows clean agreement on all semantic prompts, with only the known shared capitalization drift on `blue/green/red`. The earlier low-bit MLX `banana-color -> Green` miss does not reproduce at `bf16`, so the canonical row now uses `f16`/`bf16` instead of the noisier public low-bit pair. |
| Gemma 2 | 2B instruct | `meshllm/gemma-2-2b-it-parity-q8_0-gguf/gemma-2-2b-it-q8_0.gguf` | `meshllm/gemma-2-2b-it-parity-8bit-mlx` | `HIGH` | 2026-04-06 | `PASS` | `PASS` | `STALE` | `STALE` | `MATCH` | `PARTIAL` | Published same-origin parity pair derived from `google/gemma-2-2b-it`. Local exact validation passed on both backends with matching outputs across the full checked-in exact suite; the only minor formatting difference was `2 + 2 = **4**` vs `2 + 2 = 4`, which stayed in the same acceptance bucket. Older behavior numbers came from the public pair and should be rerun against this canonical pair before drawing new parity conclusions. |
| Gemma 3 | 1B instruct | `meshllm/gemma-3-1b-it-parity-f16-gguf/gemma-3-1b-it-f16.gguf` | `meshllm/gemma-3-1b-it-parity-bf16-mlx` | `HIGH` | 2026-04-06 | `PASS` | `PASS` | `PENDING` | `PENDING` | `MATCH` | `PARTIAL` | Published same-origin high-fidelity parity pair derived from `google/gemma-3-1b-it`. Validated on `studio54.local`: both backends passed the full exact suite with identical outputs, including `primary-colors -> Red, Green, Blue`. This replaces the noisier public low-bit Gemma3 pair for future parity checks. |
| Gemma 4 | E4B instruct | `meshllm/gemma-4-e4b-it-parity-q8_0-gguf/gemma-4-e4b-it-q8_0.gguf` | `meshllm/gemma-4-e4b-it-parity-8bit-mlx` | `HIGH` | 2026-04-06 | `PASS` | `PASS` | `PENDING` | `PENDING` | `MATCH` | `PARTIAL` | Published same-origin parity pair derived from `google/gemma-4-E4B-it`. Local exact validation passed on both backends with matching outputs across the full checked-in exact suite. The MLX side originally exposed a mixed dense/quantized Gemma 4 loader bug in mesh-llm (`missing language_model.model.per_layer_model_projection.scales`); after fixing that loader path, the same-origin 8bit/Q8_0 pair matched cleanly. |
| GLM4 | 9B 0414 | `meshllm/glm-4-9b-0414-parity-q4_k_m-gguf/glm-4-9b-0414-q4_k_m.gguf` | `meshllm/glm-4-9b-0414-parity-4bit-mlx` | `HIGH` | 2026-04-06 | `PASS` | `PASS` | `PENDING` | `PENDING` | `MATCH` | `PARTIAL` | Published same-origin parity pair derived from `THUDM/GLM-4-9B-0414`. Local exact validation passed on both backends with matching outputs across the full checked-in exact suite, including `primary-colors -> red, green, blue` and `banana-color -> Yellow`. The converted MLX artifact carries its prompt template in `chat_template.jinja`, so the canonical row now points there instead of the older public `tokenizer_config.json`-driven pair. |
| LFM2 | 350M | `meshllm/lfm2-350m-parity-q4_k_m-gguf/lfm2-350m-q4_k_m.gguf` | `meshllm/lfm2-350m-parity-4bit-mlx` | `HIGH` | 2026-04-06 | `FAIL` | `FAIL` | `PENDING` | `PENDING` | `DIFFERS` | `PARTIAL` | Published same-origin backend-drift pair derived from `LiquidAI/LFM2-350M`. Local exact validation shows the GGUF side is materially worse than the MLX side on simple prompts: GGUF answered `primary` and `alt-green` with explanatory prose instead of the requested one-word colors, while MLX returned `blue` and `green` cleanly. We keep this row to track a likely llama/GGUF-side issue rather than as a parity-clean canary. |
| OLMo2 | 7B instruct | `meshllm/olmo2-7b-instruct-parity-q8_0-gguf/olmo2-7b-instruct-q8_0.gguf` | `meshllm/olmo2-7b-instruct-parity-8bit-mlx` | `HIGH` | 2026-04-07 | `PASS` | `PASS` | `PENDING` | `PENDING` | `PENDING` | `PARTIAL` | Published same-origin OLMo2 parity pair derived from `allenai/OLMo-2-1124-7B-Instruct`. Fresh exact rerun against the canonical Q8_0 GGUF and 8-bit MLX artifacts passed on both backends. The branch also carries the MLX runtime/template fixes needed for OLMo2 prompt formatting and stability, but behavior baselines have not been accepted into the checked-in ledger yet. |
| Mamba | 2.8B | `/Users/jdumay/code/worktrees/mesh-llm-validation/output/mamba-debug/mamba-f16.gguf` | `/Users/jdumay/code/worktrees/mesh-llm-validation/mlx/mamba-8bit` | `MEDIUM` | 2026-04-07 | `FAIL` | `FAIL` | `PENDING` | `PENDING` | `BLOCKED` | `BLOCKED` | Local-only candidate pair for `state-spaces/mamba-2.8b-hf`. Exact validation failed on both sides: GGUF drifted badly on one-word/completion prompts, and the MLX path never reached inference because mesh-llm routed the MLX directory into `llama-server` and died on `gguf_init_from_file_ptr: failed to read magic`. Keep this row as an explicit failure record, not a publish target. |
| SmolLM2 | 135M instruct | `/Users/jdumay/code/worktrees/mesh-llm-validation/output/smollm2-135m/SmolLM2-135M-Instruct-Q8_0.gguf` | `/Users/jdumay/code/worktrees/mesh-llm-validation/mlx/smollm2-135m-instruct-4bit` | `MEDIUM` | 2026-04-07 | `FAIL` | `FAIL` | `PENDING` | `PENDING` | `BLOCKED` | `BLOCKED` | Local-only candidate pair for `HuggingFaceTB/SmolLM2-135M-Instruct`. Exact validation failed overall: GGUF passed the factual prompts but missed `primary`, `alt-green`, `alt-red`, `banana-color`, and `after-monday`, while the MLX path never reached inference because mesh-llm routed the MLX directory into `llama-server` and died on `gguf_init_from_file_ptr: failed to read magic`. Keep this row as an explicit failure record, not a publish target. |
| DeepSeekV3 / Kimi-K2 | K2 instruct | `public GGUF target TBD` | `mlx-community/Kimi-K2-Instruct-4bit` | `LOW` | — | `PENDING` | `PENDING` | `PENDING` | `PENDING` | `PENDING` | `PENDING` | Public GGUF target is still unresolved and likely impractical locally, so parity here will be approximate even when we can run it. |
| gpt-oss | 20B-ish | `unsloth/gpt-oss-20b-GGUF/gpt-oss-20b-Q2_K.gguf` | `concrete MLX target TBD` | `LOW` | — | `PENDING` | `PENDING` | `PENDING` | `PENDING` | `PENDING` | `PENDING` | Need concrete MLX repo target before this becomes a meaningful parity pair. |
| Kimi Linear | 48B A3B | `public GGUF target TBD` | `mlx-community/Kimi-Linear-48B-A3B-Instruct-4bit` | `LOW` | — | `PENDING` | `PENDING` | `PENDING` | `PENDING` | `PENDING` | `PENDING` | Public GGUF target is unresolved and likely too large locally, so parity will be approximate even if we can run it. |

## Notes

- Exact smoke means the deterministic `blue / green / red` style suite plus reasoning-on probe where relevant.
- Behavior means the MT-Bench-derived behavior harness in [`scripts/ci-mt-bench-behavior.py`](/Users/jdumay/.codex/worktrees/e497/mesh-llm/scripts/ci-mt-bench-behavior.py).
- Raw rebuilt-engine exact rerun artifacts are stored under [`MLX_VALIDATION_RESULTS/rerun-20260404-buildsync`](/Users/jdumay/.codex/worktrees/e497/mesh-llm/MLX_VALIDATION_RESULTS/rerun-20260404-buildsync).
- The judgment rule is simple:
  - `🦙 GGUF FAIL` + `🍎 MLX FAIL` = probably shared model weakness
  - `🦙 GGUF PASS` + `🍎 MLX FAIL` = MLX-specific problem and not OK
  - `🦙 GGUF FAIL` + `🍎 MLX PASS` = MLX at least not worse there
- Record enough detail in `Notes` to make the next fix obvious.
