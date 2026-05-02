# Documentation

This directory holds project documentation that is not owned by a single Rust crate.

## Start Here

| Doc | What it covers |
|---|---|
| [USAGE.md](USAGE.md) | Service installs, model commands, storage, and runtime control |
| [AGENTS.md](AGENTS.md) | Goose, Claude Code, pi, OpenCode, curl, and blackboard usage |
| [BENCHMARKS.md](BENCHMARKS.md) | Current benchmark numbers and performance context |
| [SKIPPY.md](SKIPPY.md) | Plan for replacing llama-server/rpc-server serving with skippy |
| [SKIPPY_RUNTIME_README.md](SKIPPY_RUNTIME_README.md) | Imported standalone skippy runtime README for background context |
| [CLI.md](CLI.md) | CLI reference |
| [CI_GUIDANCE.md](CI_GUIDANCE.md) | CI workflow responsibilities and path filtering guidance |

## Topic Areas

| Directory | What belongs there |
|---|---|
| [design/](design/) | Architecture notes, protocol design, testing playbooks, and carried llama.cpp patch documentation |
| [moe/](moe/) | MoE analyzer, placement, and CLI planning notes |
| [plugins/](plugins/) | Plugin architecture and implementation planning |
| [plans/](plans/) | Narrow implementation plans that are not yet general design docs |
| [specs/](specs/) | Focused behavior specs for individual features |

Per-crate docs stay with their crates. The main binary crate overview lives at [../crates/mesh-llm/README.md](../crates/mesh-llm/README.md).
