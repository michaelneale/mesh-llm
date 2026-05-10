```mermaid
flowchart TD
    subgraph Triggers["Triggers"]
        PR["Pull Request / push main"]
    end
    subgraph Changes["changes (path filter)"]
        F_UI["ui changed?"]
        F_RUST["rust changed?"]
        F_SDK["sdk changed?"]
        F_BENCH["benchmarks changed?"]
    end
    PR --> Changes
    %% ── Producers ──
    subgraph Producers["Producers (parallel, independent)"]
        direction TB
        subgraph UI_Target["Target: ui"]
            UI["Build UI\nnpm ci + build + test\n→ upload: ci-ui-dist"]
        end
        subgraph Core_Target["Target: rust-core (ubuntu-latest)"]
            CORE["fmt check · clippy\ncargo build -p mesh-llm (debug)\nunit tests · protocol compat\nbuild llama.cpp CPU+RPC\nCLI smoke · client-auto boot\n→ upload: ci-linux-inference-binaries"]
        end
        subgraph FFI_Target["Target: ffi-sdk (ubuntu-latest)"]
            FFI["Build mesh-api / mesh-api-ffi / mesh-client\nembedded dep purity check\ncompile+lint only, no artifact"]
        end
        subgraph Vulkan_Target["Target: vulkan (ubuntu-latest)"]
            VULKAN["Install libvulkan-dev + glslc\nDownload ci-ui-dist\njust release-build-vulkan\nCLI smoke"]
        end
    end
    F_UI -- "true" --> UI_Target
    F_RUST -- "true" --> Core_Target
    F_RUST -- "true" --> FFI_Target
    F_RUST -- "true" --> Vulkan_Target
    UI_Target -- "artifact: ci-ui-dist" --> Vulkan_Target
    %% ── CUDA (self-hosted GPU runner) ──
    subgraph CUDA_Target["Target: cuda\n🖥️ self-hosted GPU runner"]
        direction TB
        CUDA_BUILD["Build llama.cpp CUDA\njust release-build-cuda 89\ncargo build mesh-llm (debug)"]
        CUDA_SMOKE["CLI smoke\nmesh-llm --version / --help"]
        CUDA_BUILD --> CUDA_SMOKE
    end
    F_RUST -- "true" --> CUDA_Target
    UI_Target -- "artifact: ci-ui-dist" --> CUDA_Target
    %% ── ROCm (self-hosted GPU runner) ──
    subgraph ROCm_Target["Target: rocm\n🖥️ self-hosted GPU runner"]
        direction TB
        ROCM_BUILD["Build llama.cpp ROCm\njust release-build-rocm gfx1100\ncargo build mesh-llm (debug)"]
        ROCM_SMOKE["CLI smoke"]
        ROCM_BUILD --> ROCM_SMOKE
    end
    F_RUST -- "true" --> ROCm_Target
    UI_Target -- "artifact: ci-ui-dist" --> ROCm_Target
    %% ── Smoke tests (consume artifact) ──
    subgraph Smoke["smoke.yml (reusable, ubuntu-latest)"]
        SMOKE["Download ci-linux-inference-binaries\nReal inference · OpenAI compat\nSplit-mode · MoE split + mesh"]
    end
    CORE -- "artifact" --> SMOKE
    subgraph SDK_Smokes["SDK Smokes (consume artifact)"]
        direction LR
        NATIVE["Native SDK\n(Linux)"]
        KOTLIN["Kotlin SDK\n(Linux)"]
        SWIFT["Swift SDK\n(macOS)\nbuild llama.cpp Metal"]
    end
    SMOKE -- "success" --> SDK_Smokes
    F_SDK -- "true" --> SDK_Smokes
    %% ── Benchmark smokes (optional, self-hosted) ──
    subgraph Bench_Smokes["Benchmark Smokes (optional)\n🖥️ self-hosted runners"]
        direction LR
        SWIFT_BENCH["macOS benchmark"]
        CUDA_BENCH["CUDA benchmark"]
        ROCM_BENCH["ROCm benchmark"]
    end
    F_BENCH -- "true" --> Bench_Smokes
    CUDA_Target --> CUDA_BENCH
    ROCm_Target --> ROCM_BENCH
    %% ════════════════════════════════════════════════════
    %% Release (separate shape, separate trigger)
    %% ════════════════════════════════════════════════════
    subgraph Release["release.yml (workflow_dispatch, tag push)"]
        REL_PREP["prepare_release\nversion bump · tag · push"]
        subgraph REL_Builds["Release builds (full shape, parallel)"]
            REL_CPU["Linux CPU\n(ubuntu-latest)"]
            REL_ARM["Linux ARM64\n(ubuntu-24.04-arm)"]
            REL_MACOS["macOS Metal\n(macos-14)"]
            REL_VULKAN["Linux Vulkan\n(ubuntu-latest)"]
        end
        subgraph REL_GPU["Release GPU builds\n🖥️ self-hosted runners"]
            REL_CUDA_126["CUDA 12.6\nfull arch, FA=ON"]
            REL_CUDA_127["CUDA 12.7\nfull arch, FA=ON"]
            REL_CUDA_129["CUDA 12.9\nfull arch, FA=ON"]
            REL_CUDA_132["CUDA 13.2\nfull arch, FA=ON"]
            REL_ROCM["ROCm\nfull gfx matrix"]
        end
        REL_SMOKE["Release smoke\n(release-shape binaries)"]
        PUBLISH["publish GitHub release\ngated on smoke success"]
        PUB_CRATES["publish crates.io"]
        PUB_ANDROID["publish Android Maven"]
        REL_PREP --> REL_Builds & REL_GPU
        REL_CPU --> REL_SMOKE --> PUBLISH
        PUBLISH --> PUB_CRATES & PUB_ANDROID
    end
    style UI_Target fill:#1a3a5c,stroke:#4a90d9,color:#e8f4fd
    style Core_Target fill:#1a3a5c,stroke:#4a90d9,color:#e8f4fd
    style FFI_Target fill:#1a3a5c,stroke:#4a90d9,color:#e8f4fd
    style Vulkan_Target fill:#1a3a5c,stroke:#4a90d9,color:#e8f4fd
    style CUDA_Target fill:#2d1b4e,stroke:#9b59b6,color:#f5eeff
    style ROCm_Target fill:#2d1b4e,stroke:#9b59b6,color:#f5eeff
    style Smoke fill:#1a3d2e,stroke:#2ecc71,color:#eaffef
    style SDK_Smokes fill:#1a3d2e,stroke:#2ecc71,color:#eaffef
    style Bench_Smokes fill:#3d2b00,stroke:#f39c12,color:#fff8e1
    style Release fill:#2a2a2a,stroke:#888,color:#ddd
    style REL_Builds fill:#3a3a3a,stroke:#888,color:#ddd
    style REL_GPU fill:#3a2a4a,stroke:#9b59b6,color:#f5eeff
```
