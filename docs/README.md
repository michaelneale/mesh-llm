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
| [specs/layer-package-repos.md](specs/layer-package-repos.md) | Layer package repository layout, manifest, publishing, and validation spec |

## Topic Areas

| Directory | What belongs there |
|---|---|
| [design/](design/) | Architecture notes, protocol design, testing playbooks, and carried llama.cpp patch documentation |
| [plugins/](plugins/) | Plugin architecture and implementation planning |
| [plans/](plans/) | Narrow implementation plans that are not yet general design docs |
| [specs/](specs/) | Focused behavior specs for individual features |

Per-crate docs stay with their crates. The main binary crate overview lives at
[../crates/mesh-llm/README.md](../crates/mesh-llm/README.md), and the web
console/embedded asset crate overview lives at
[../crates/mesh-llm-ui/README.md](../crates/mesh-llm-ui/README.md).
Shared protocol-facing model/type ownership lives at
[../crates/mesh-llm-types/README.md](../crates/mesh-llm-types/README.md).
Shared owner identity and envelope crypto lives at
[../crates/mesh-llm-identity/README.md](../crates/mesh-llm-identity/README.md).
Shared wire protocol ownership lives at
[../crates/mesh-llm-protocol/README.md](../crates/mesh-llm-protocol/README.md).
Shared routing target ownership lives at
[../crates/mesh-llm-routing/README.md](../crates/mesh-llm-routing/README.md).
