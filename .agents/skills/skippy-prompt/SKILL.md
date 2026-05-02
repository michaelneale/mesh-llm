---
name: skippy-prompt
description: Use this skill when running, debugging, or migrating the skippy prompt CLI, local staged prompt launcher, binary REPL client, prompt history commands, speculative prompt mode, or prompt-owned process lifecycle.
metadata:
  short-description: Run and migrate staged prompt workflows
---

# skippy-prompt

Use this skill for staged prompt workflows and any future prompt CLI migration.

## Mesh Migration Notes

Do not bring back `kv-server` or `ngram-pool` as part of prompt migration.
Prompt flows in mesh should exercise the same embedded skippy runtime and
mesh stage-control lifecycle used by normal serving.

Public OpenAI compatibility belongs in `openai-frontend`, not in prompt
tooling. Prompt tools are for development, diagnostics, and reproducible model
checks.

## Commands

Before using source-repo prompt commands, verify the crate exists here:

```bash
cargo metadata --no-deps --format-version 1 | jq -r '.packages[].name' | sort
```

Current mesh smoke paths should go through `mesh-llm` serving and the
management/OpenAI APIs rather than old standalone prompt launchers.
