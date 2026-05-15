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
        PRWorkflow["pr_workflow_only?"]
        SDK["sdk_smoke_required?"]
        Docs["docs_only?"]
    end

    PR --> Files --> Affected
    Affected --> ClippyBins
    Files --> Backend
    Files --> PRWorkflow
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

subgraph PRCI["pr_ci.yml · PR Builds"]
        direction TB
        subgraph Producers["top-level target matrices"]
            LinuxCrateTests["linux_crate_tests matrix\nSDK/API · Skippy · unit/protocol bins"]
            LinuxTargets["linux_targets matrix\nCPU row: debug mesh-llm · binary smokes\nCUDA / ROCm / Vulkan rows build when backend_changed\nCPU → ci-linux-inference-binaries"]
            WindowsTargets["windows_targets matrix\nCPU / CUDA / ROCm / Vulkan\nCPU checks unless Windows CPU changed"]
            MacTargets["macos_targets matrix\nmacOS/UI/Swift/backend-scoped CPU build · CLI smoke\nCUDA / ROCm / Vulkan explicit skips\nCPU → ci-macos-inference-binaries"]
            SwiftXCFramework["swift_xcframework\nhost macOS XCFramework artifact"]
        end

        subgraph Smokes["artifact-consuming smokes"]
            Restore["restore-smoke-inputs action\ndownload artifact · stage binary · restore model"]
            Inference["smoke.yml\nLinux inference + OpenAI + split serving"]
            Scripted["scripted-binary-smoke.yml\ntwo-node client/serving"]
            SDKSmoke["sdk-smoke.yml\nnative · Kotlin · Swift"]
        end
    end

    Docs -. "true: gate heavy jobs" .-> PRCI
    PRWorkflow -. "true: keep to routing validation" .-> PRCI
    Affected --> LinuxCrateTests
    Affected --> LinuxTargets
    Affected --> MacTargets
    Backend --> LinuxTargets
    Backend --> WindowsTargets
    Backend --> MacTargets
    LinuxTargets -- "CPU artifact: ci-linux-inference-binaries" --> Restore
    MacTargets -- "CPU artifact: ci-macos-inference-binaries" --> Restore
    SwiftXCFramework -- "ci-swift-xcframework" --> SDKSmoke
    Restore --> Inference
    Restore --> Scripted
    Restore --> SDKSmoke
    SDK --> SDKSmoke

    subgraph PRDocker["pr_docker.yml · PR Docker Build"]
        DockerBuild["Build Docker client image\nDocker-input scoped · push: false\nGHA layer cache"]
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
- `pr_ci.yml` is named **PR Builds** and owns PR target matrices plus integration
  and smoke validation. Linux crate tests run in a separate matrix so Linux CPU
  can produce the smoke artifact without serializing every Rust test on the
  binary-build critical path. Linux, macOS, and Windows are top-level matrices;
  Linux and macOS CPU rows upload the binaries that downstream smoke jobs consume.
  Swift XCFramework production runs in parallel with the macOS CPU row so Swift
  SDK smoke consumes a built XCFramework artifact instead of rebuilding it after
  the macOS binary is ready.
- `pr_docker.yml` validates the PR Docker client image without publishing when
  Docker packaging inputs change. Workflow-only edits are covered by the shared
  YAML/consistency validation instead of self-triggering a heavyweight image
  build. Docker builds use GitHub Actions layer cache scoped to the PR cache ref
  so repeated Docker checks do not rebuild every layer.
- `pr_cleanup.yml` deletes PR merge-ref caches and artifacts from positively
  matched PR workflow runs when a pull request closes.
- Non-PR workflows (`ci.yml`, `docker.yml`, `release.yml`) own main, dispatch,
  tag, and release-grade publishing behavior.

## Rust cache reuse

- PR Rust builds use `Swatinem/rust-cache` alongside `sccache` so repeated runs
  on the same pull request can restore Cargo registry/git state while compiler
  outputs are primarily handled by `sccache`.
- PR `rust-cache` saves avoid workspace `target/` uploads because the post-run
  archive cost was longer than the reuse benefit for PR Builds; main-branch
  cache saves may still include target data.
- Rust caches stay platform/backend scoped. Linux CPU, Linux backend rows, macOS,
  Linux crate-test groups, Windows backend rows, clippy, and HuggingFace download
  smoke each keep their own compatible cache namespace instead of sharing a
  single prebuilt dependency artifact across incompatible runners or SDK
  environments.
- PR cache writes use the standard GitHub Actions cache service under
  `refs/pull/<PR>/merge`; `pr_cleanup.yml` deletes that ref's caches when the PR
  closes so PR-lifetime Rust caches do not linger unbounded.
- PR workflow/control-plane-only changes use `pr_workflow_only` routing so they
  validate the CI surface without forcing every Rust crate and backend lane.

## Artifact and smoke reuse

- Smoke jobs restore binaries through `.github/actions/restore-smoke-inputs` and
  reusable workflows instead of rebuilding `mesh-llm` or patched llama.cpp.
- Linux CPU artifacts feed inference, two-node, native SDK, and Kotlin SDK
  smokes. macOS CPU artifacts and the parallel Swift XCFramework artifact feed
  Swift SDK smokes.
- PR and smoke-only CI artifacts use `retention-days: 1`; PR cleanup removes
  matched PR-run artifacts proactively.
- Direct `mesh-llm` invocations in workflows and CI scripts must include
  `--log-format json`.

For agent-facing workflow editing rules, see `.github/AGENTS.md`.
