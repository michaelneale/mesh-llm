# Documentation

Use this hub to find project guides that are not owned by a single Rust crate.

## Start here

| Need | Doc |
|---|---|
| Install, run, service mode, model storage | [USAGE.md](USAGE.md) |
| Private meshes, published meshes, public joining | [MESHES.md](MESHES.md) |
| Run big models with Skippy layer splits | [SKIPPY_SPLITS.md](SKIPPY_SPLITS.md) |
| Contribute or publish layer package repositories | [LAYER_PACKAGE_REPOS.md](LAYER_PACKAGE_REPOS.md) |
| Goose, Claude Code, OpenCode, Pi, curl, blackboard | [AGENTS.md](AGENTS.md) |
| Command-by-command CLI reference | [CLI.md](CLI.md) |
| Exo comparison | [EXO_COMPARISON.md](EXO_COMPARISON.md) |

## Skippy and model-package docs

| Doc | What it covers |
|---|---|
| [skippy/FAMILY_STATUS.md](skippy/FAMILY_STATUS.md) | Certified family/split/wire-dtype status |
| [skippy/FAMILY_CERTIFY.md](skippy/FAMILY_CERTIFY.md) | Certification workflow for new families |
| [skippy/TOPOLOGY_PLANNER.md](skippy/TOPOLOGY_PLANNER.md) | Stage topology planning behavior |
| [skippy/DATA_FLOW.md](skippy/DATA_FLOW.md) | Stage data flow and transport details |
| [skippy/LLAMA_PARITY.md](skippy/LLAMA_PARITY.md) | Remaining llama.cpp parity queue |
| [specs/layer-package-repos.md](specs/layer-package-repos.md) | Manifest schema and package artifact rules |
| [SKIPPY.md](SKIPPY.md) | Skippy integration readiness and parity notes |

Use [SKIPPY_SPLITS.md](SKIPPY_SPLITS.md) for Skippy split-serving workflows.

## Other references

| Doc or directory | What belongs there |
|---|---|
| [BENCHMARKS.md](BENCHMARKS.md) | Current benchmark numbers and performance context |
| [design/](design/) | Architecture notes, protocol design, testing playbooks, carried llama.cpp patch documentation |
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
