```mermaid
flowchart TD
    subgraph Triggers["Pull request triggers"]
        PR["opened / synchronize / reopened / ready_for_review"]
    end

    subgraph Changes["compute-changes"]
        Files["changed files"]
        Affected["affected crates + reverse deps"]
        ClippyBins["clippy binpack\nplan-clippy-batches.sh"]
        Backend["backend_changed?"]
        SDK["sdk_smoke_required?"]
        Docs["docs_only?"]
    end

    PR --> Files --> Affected
    Affected --> ClippyBins
    Files --> Backend
    Affected --> SDK
    Files --> Docs

    subgraph Quality["pr_quality.yml · PR Quality Checks"]
        direction TB
        Fmt["rust-fmt"]
        Clippy["rust-clippy matrix\nweighted affected-crate bins"]
        UIQ["ui-quality"]
        QSummary["summary"]
        Fmt --> QSummary
        Clippy --> QSummary
        UIQ --> QSummary
    end

    ClippyBins --> Clippy
    Affected --> Fmt
    Files --> UIQ

subgraph PRCI["pr_builds.yml · PR Builds"]
        direction TB
        subgraph Producers["top-level target matrices"]
            LinuxTargets["linux_targets matrix\nCPU row: crate tests · debug mesh-llm · CLI/client smoke\nCUDA / ROCm / Vulkan rows build when backend_changed\nCPU → ci-linux-inference-binaries"]
            WindowsTargets["windows_targets matrix\nCPU / CUDA / ROCm / Vulkan\nCPU checks unless Windows CPU changed"]
            MacTargets["macos_targets matrix\nCPU row: macOS Metal build · crate tests · CLI smoke\nCUDA / ROCm / Vulkan explicit skips\nCPU → ci-macos-inference-binaries"]
        end

        subgraph Smokes["artifact-consuming smokes"]
            Restore["restore-smoke-inputs action\ndownload artifact · stage binary · restore model"]
            Inference["smoke.yml\nLinux inference + OpenAI + split serving"]
            Scripted["scripted-binary-smoke.yml\ntwo-node client/serving"]
            SDKSmoke["sdk-smoke.yml\nnative · Kotlin · Swift"]
        end
    end

    Docs -. "true: gate heavy jobs" .-> PRCI
    Affected --> LinuxTargets
    Affected --> MacTargets
    Backend --> LinuxTargets
    Backend --> WindowsTargets
    Backend --> MacTargets
    LinuxTargets -- "CPU artifact: ci-linux-inference-binaries" --> Restore
    MacTargets -- "CPU artifact: ci-macos-inference-binaries" --> Restore
    Restore --> Inference
    Restore --> Scripted
    Restore --> SDKSmoke
    SDK --> SDKSmoke

    subgraph PRDocker["pr_docker.yml · PR Docker Build"]
        DockerBuild["Build Docker client image\npush: false"]
    end

    Files --> DockerBuild

    subgraph Cleanup["pr_cleanup.yml · PR Cache Cleanup"]
        Closed["pull_request_target closed"]
        DeleteCaches["delete caches for\nrefs/pull/<PR>/merge"]
        DeleteArtifacts["delete artifacts from\nmatched PR workflow runs"]
        Closed --> DeleteCaches
        Closed --> DeleteArtifacts
    end

    subgraph MainRelease["non-PR workflows"]
        MainCI["ci.yml\npush main / dispatch"]
        DockerPublish["docker.yml\ntag / dispatch publish"]
        Release["release.yml\nrelease artifacts + publish gates"]
    end

    style Quality fill:#1a3a5c,stroke:#4a90d9,color:#e8f4fd
    style PRCI fill:#1a3d2e,stroke:#2ecc71,color:#eaffef
    style Producers fill:#1a3d2e,stroke:#2ecc71,color:#eaffef
    style Smokes fill:#17324d,stroke:#4a90d9,color:#e8f4fd
    style PRDocker fill:#3d2b00,stroke:#f39c12,color:#fff8e1
    style Cleanup fill:#3d2b00,stroke:#f39c12,color:#fff8e1
    style MainRelease fill:#2a2a2a,stroke:#888,color:#ddd
```

## Current PR Builds contract

- `pr_quality.yml` is named **PR Quality Checks** and owns the earliest feedback:
  formatting, UI quality when relevant, and deterministic clippy bins from
  `scripts/plan-clippy-batches.sh`.
- `pr_builds.yml` is named **PR Builds** and owns PR target matrices plus integration
  and smoke validation. Linux, macOS, and Windows are top-level matrices; Linux
  and macOS CPU rows upload the binaries that downstream smoke jobs consume.
- `pr_docker.yml` validates the PR Docker client image without publishing.
- `pr_cleanup.yml` deletes PR merge-ref caches and artifacts from positively
  matched PR workflow runs when a pull request closes.
- Non-PR workflows (`ci.yml`, `docker.yml`, `release.yml`) own main, dispatch,
  tag, and release-grade publishing behavior.

## Artifact and smoke reuse

- Smoke jobs restore binaries through `.github/actions/restore-smoke-inputs` and
  reusable workflows instead of rebuilding `mesh-llm` or patched llama.cpp.
- Linux CPU artifacts feed inference, two-node, native SDK, and Kotlin SDK
  smokes. macOS CPU artifacts feed Swift SDK smokes.
- PR and smoke-only CI artifacts use `retention-days: 1`; PR cleanup removes
  matched PR-run artifacts proactively.
- Direct `mesh-llm` invocations in workflows and CI scripts must include
  `--log-format json`.

For agent-facing workflow editing rules, see `.github/AGENTS.md`.
