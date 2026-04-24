param(
    [string]$Backend = "",
    [string]$CudaArch = "",
    [string]$RocmArch = ""
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $scriptDir ".."))
$llamaDir = Join-Path $repoRoot "llama.cpp"
$buildDir = Join-Path $llamaDir "build"
$meshUiDir = Join-Path $repoRoot "mesh-llm\ui"
$compilerLauncherArgs = @()
$compilerCacheBin = $null

function Add-ToPath {
    param([string]$Directory)

    if (-not $Directory -or -not (Test-Path $Directory)) {
        return
    }

    $pathEntries = @($env:PATH -split [System.IO.Path]::PathSeparator)
    if ($pathEntries -contains $Directory) {
        return
    }

    $env:PATH = "$Directory$([System.IO.Path]::PathSeparator)$env:PATH"
}

function Test-CommandSuccess {
    param(
        [string]$Command,
        [string[]]$Arguments = @()
    )

    try {
        $null = & $Command @Arguments 2>$null
        return $LASTEXITCODE -eq 0
    } catch {
        return $false
    }
}

function Resolve-CommandPath {
    param([string]$Name)

    $command = Get-Command $Name -ErrorAction SilentlyContinue
    if ($command) {
        return $command.Source
    }
    return $null
}

function Configure-CompilerCache {
    $script:compilerCacheBin = Resolve-CommandPath "sccache"
    if (-not $script:compilerCacheBin) {
        $script:compilerCacheBin = Resolve-CommandPath "ccache"
    }
    if (-not $script:compilerCacheBin) {
        $script:compilerLauncherArgs = @()
        return
    }

    Write-Host "Using compiler cache: $script:compilerCacheBin"
    $script:compilerLauncherArgs = @(
        "-DCMAKE_C_COMPILER_LAUNCHER=$script:compilerCacheBin",
        "-DCMAKE_CXX_COMPILER_LAUNCHER=$script:compilerCacheBin"
    )
}

function Import-CmdEnvironment {
    param(
        [Parameter(Mandatory = $true)]
        [string]$CommandLine
    )

    $output = & cmd.exe /s /c "$CommandLine && set"
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to initialize Windows build environment with command: $CommandLine"
    }

    foreach ($line in $output) {
        if ($line -match '^(?<name>[^=]+)=(?<value>.*)$') {
            Set-Item -Path "env:$($Matches.name)" -Value $Matches.value
        }
    }
}

function Invoke-NativeCommand {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Command,
        [string[]]$Arguments = @()
    )

    & $Command @Arguments
    if ($LASTEXITCODE -ne 0) {
        $argString = if ($Arguments.Count -gt 0) { " " + ($Arguments -join " ") } else { "" }
        throw "Command failed with exit code ${LASTEXITCODE}: $Command$argString"
    }
}

function Test-UiBuildRequired {
    param(
        [Parameter(Mandatory = $true)]
        [string]$UiDirectory
    )

    $distDir = Join-Path $UiDirectory "dist"
    if (-not (Test-Path $distDir)) {
        return $true
    }

    $distFiles = Get-ChildItem -Path $distDir -File -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $distFiles) {
        return $true
    }

    $distTimestampUtc = (Get-Item $distDir).LastWriteTimeUtc
    $uiBuildInputs = @(
        (Join-Path $UiDirectory "package.json"),
        (Join-Path $UiDirectory "package-lock.json"),
        (Join-Path $UiDirectory "vite.config.ts"),
        (Join-Path $UiDirectory "tsconfig.json"),
        (Join-Path $UiDirectory "postcss.config.cjs"),
        (Join-Path $UiDirectory "tailwind.config.ts"),
        (Join-Path $UiDirectory "index.html"),
        (Join-Path $UiDirectory "src"),
        (Join-Path $UiDirectory "public")
    )

    foreach ($path in $uiBuildInputs) {
        if (-not (Test-Path $path)) {
            continue
        }

        $item = Get-Item $path
        if ($item.PSIsContainer) {
            $newerInput = Get-ChildItem -Path $path -File -Recurse -ErrorAction SilentlyContinue |
                Where-Object { $_.LastWriteTimeUtc -gt $distTimestampUtc } |
                Select-Object -First 1
            if ($newerInput) {
                return $true
            }
            continue
        }

        if ($item.LastWriteTimeUtc -gt $distTimestampUtc) {
            return $true
        }
    }

    return $false
}

function Test-NpmInstallRequired {
    param(
        [Parameter(Mandatory = $true)]
        [string]$UiDirectory
    )

    $nodeModulesDir = Join-Path $UiDirectory "node_modules"
    if (-not (Test-Path $nodeModulesDir)) {
        return $true
    }

    $nodeModulesTimestampUtc = (Get-Item $nodeModulesDir).LastWriteTimeUtc
    foreach ($manifestName in @("package.json", "package-lock.json")) {
        $manifestPath = Join-Path $UiDirectory $manifestName
        if (-not (Test-Path $manifestPath)) {
            continue
        }

        if ((Get-Item $manifestPath).LastWriteTimeUtc -gt $nodeModulesTimestampUtc) {
            return $true
        }
    }

    return $false
}

function Normalize-RecipeArgument {
    param(
        [AllowEmptyString()]
        [string]$Value,
        [string[]]$KnownNames = @()
    )

    if ($null -eq $Value) {
        return $Value
    }

    $normalized = $Value.Trim()
    if (-not $normalized) {
        return ""
    }

    if ($normalized -match '^(?<name>[A-Za-z_][A-Za-z0-9_-]*)=(?<value>.*)$') {
        $matchedName = $Matches.name
        $isKnownName = $KnownNames.Count -eq 0
        foreach ($knownName in $KnownNames) {
            if ($matchedName.Equals($knownName, [System.StringComparison]::OrdinalIgnoreCase)) {
                $isKnownName = $true
                break
            }
        }

        if ($isKnownName) {
            $normalized = $Matches.value
        }
    }

    if ($normalized.Length -ge 2) {
        $first = $normalized[0]
        $last = $normalized[$normalized.Length - 1]
        if (($first -eq '"' -and $last -eq '"') -or ($first -eq "'" -and $last -eq "'")) {
            $normalized = $normalized.Substring(1, $normalized.Length - 2)
        }
    }

    return $normalized.Trim()
}

function Ensure-MsvcToolchain {
    if ((Resolve-CommandPath "cl") -and (Resolve-CommandPath "link") -and (Resolve-CommandPath "lib")) {
        return
    }

    $programFilesX86 = ${env:ProgramFiles(x86)}
    $vswhereCandidates = @()
    if ($programFilesX86) {
        $vswhereCandidates += (Join-Path $programFilesX86 "Microsoft Visual Studio\Installer\vswhere.exe")
    }
    if ($env:ProgramFiles) {
        $vswhereCandidates += (Join-Path $env:ProgramFiles "Microsoft Visual Studio\Installer\vswhere.exe")
    }
    $vswhereFromPath = Resolve-CommandPath "vswhere"
    if ($vswhereFromPath) {
        $vswhereCandidates += $vswhereFromPath
    }

    $vcvars64 = $null
    $vswhere = $vswhereCandidates | Where-Object { $_ -and (Test-Path $_) } | Select-Object -Unique -First 1
    if ($vswhere) {
        $installationPathOutput = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath | Select-Object -First 1
        $installationPath = if ($installationPathOutput) { $installationPathOutput.Trim() } else { "" }
        if ($installationPath) {
            $candidate = Join-Path $installationPath "VC\Auxiliary\Build\vcvars64.bat"
            if (Test-Path $candidate) {
                $vcvars64 = $candidate
            }
        }
    }

    if (-not $vcvars64) {
        $searchRoots = @($programFilesX86, $env:ProgramFiles) | Where-Object { $_ } | Select-Object -Unique
        foreach ($searchRoot in $searchRoots) {
            $candidate = Get-ChildItem -Path $searchRoot -Filter vcvars64.bat -Recurse -ErrorAction SilentlyContinue |
                Where-Object { $_.FullName -like '*Microsoft Visual Studio*VC\Auxiliary\Build\vcvars64.bat' } |
                Select-Object -First 1
            if ($candidate) {
                $vcvars64 = $candidate.FullName
                break
            }
        }
    }

    if (-not (Test-Path $vcvars64)) {
        throw "Visual Studio Build Tools with vcvars64.bat were not found on this Windows runner."
    }

    Import-CmdEnvironment "`"$vcvars64`" >nul"

    if (-not (Resolve-CommandPath "cl")) {
        throw "MSVC toolchain initialization completed, but cl.exe is still not available in PATH."
    }
}

function Resolve-HipPackageRoot {
    $roots = @()
    if ($env:HIP_PATH) {
        $roots += $env:HIP_PATH
    }
    if ($env:ROCM_PATH) {
        $roots += $env:ROCM_PATH
    }

    $roots = $roots | Where-Object { $_ } | Select-Object -Unique

    foreach ($root in $roots) {
        if (-not (Test-Path $root)) {
            continue
        }

        $directConfig = Join-Path $root "lib\cmake\hip\hip-config.cmake"
        if (Test-Path $directConfig) {
            return [PSCustomObject]@{
                Root   = $root
                HipDir = Split-Path -Parent $directConfig
            }
        }

        $versionedRoot = Get-ChildItem -Path $root -Directory -ErrorAction SilentlyContinue |
            Where-Object {
                Test-Path (Join-Path $_.FullName "lib\cmake\hip\hip-config.cmake")
            } |
            Sort-Object Name -Descending |
            Select-Object -First 1

        if ($versionedRoot) {
            $configPath = Join-Path $versionedRoot.FullName "lib\cmake\hip\hip-config.cmake"
            return [PSCustomObject]@{
                Root   = $versionedRoot.FullName
                HipDir = Split-Path -Parent $configPath
            }
        }
    }

    return $null
}

function Resolve-RocmRoot {
    $hipPackage = Resolve-HipPackageRoot
    if ($hipPackage) {
        return $hipPackage.Root
    }
    if ($env:ProgramFiles) {
        foreach ($candidate in @(
            (Join-Path $env:ProgramFiles "AMD\ROCm"),
            (Join-Path $env:ProgramFiles "AMD\HIP")
        )) {
            if (Test-Path $candidate) {
                return $candidate
            }
        }
    }
    return $null
}

function Resolve-VulkanSdkRoot {
    if ($env:VULKAN_SDK -and (Test-Path $env:VULKAN_SDK)) {
        return $env:VULKAN_SDK
    }

    if ($env:ProgramFiles) {
        $sdkBase = Join-Path $env:ProgramFiles "VulkanSDK"
        if (Test-Path $sdkBase) {
            $latest = Get-ChildItem -Path $sdkBase -Directory | Sort-Object Name -Descending | Select-Object -First 1
            if ($latest) {
                return $latest.FullName
            }
        }
    }

    return $null
}

function Resolve-Backend {
    param([string]$Requested)

    if ($Requested) {
        $normalized = $Requested.ToLowerInvariant()
        switch ($normalized) {
            "hip" { return "rocm" }
            "rocm" { return "rocm" }
            default { return $normalized }
        }
    }

    if (Test-CommandSuccess "nvidia-smi" @("--query-gpu=name", "--format=csv,noheader")) {
        return "cuda"
    }

    if (Resolve-RocmRoot) {
        return "rocm"
    }

    if ((Resolve-CommandPath "hipInfo") -or (Resolve-CommandPath "hipconfig")) {
        return "rocm"
    }

    if (Test-CommandSuccess "vulkaninfo" @("--summary")) {
        return "vulkan"
    }

    if (Resolve-VulkanSdkRoot) {
        return "vulkan"
    }

    return "cpu"
}

function Ensure-CudaToolchain {
    if (Resolve-CommandPath "nvcc") {
        return
    }

    $candidates = @()
    if ($env:CUDA_PATH) {
        $candidates += (Join-Path $env:CUDA_PATH "bin")
    }
    if ($env:ProgramFiles) {
        $toolkitRoot = Join-Path $env:ProgramFiles "NVIDIA GPU Computing Toolkit\CUDA"
        if (Test-Path $toolkitRoot) {
            $candidates += Get-ChildItem -Path $toolkitRoot -Directory | Sort-Object Name -Descending | ForEach-Object {
                Join-Path $_.FullName "bin"
            }
        }
    }

    foreach ($candidate in $candidates) {
        if (Test-Path (Join-Path $candidate "nvcc.exe")) {
            Add-ToPath $candidate
            return
        }
    }

    throw "nvcc not found. Install the CUDA toolkit and ensure nvcc.exe is in PATH."
}

function Ensure-RocmToolchain {
    $rocmRoot = Resolve-RocmRoot
    $hipPackage = Resolve-HipPackageRoot
    if ($rocmRoot) {
        $binDir = Join-Path $rocmRoot "bin"
        $llvmBinDir = Join-Path $rocmRoot "llvm\bin"
        Add-ToPath $binDir
        Add-ToPath $llvmBinDir
        $env:ROCM_PATH = $rocmRoot
        $env:HIP_PATH = $rocmRoot
        $env:CMAKE_PREFIX_PATH = if ($env:CMAKE_PREFIX_PATH) {
            "$rocmRoot;$env:CMAKE_PREFIX_PATH"
        } else {
            $rocmRoot
        }
        if ($hipPackage) {
            $env:hip_DIR = $hipPackage.HipDir
        }
        if (-not $env:HIPCC -or -not $env:HIPCXX) {
            foreach ($candidate in @(
                (Join-Path $llvmBinDir "clang++.exe"),
                (Join-Path $binDir "clang++.exe"),
                (Join-Path $llvmBinDir "clang.exe"),
                (Join-Path $binDir "clang.exe")
            )) {
                if ($candidate -like "*clang++.exe" -and -not $env:HIPCXX -and (Test-Path $candidate)) {
                    $env:HIPCXX = $candidate
                } elseif ($candidate -like "*clang.exe" -and -not $env:HIPCC -and (Test-Path $candidate)) {
                    $env:HIPCC = $candidate
                }

                if ($env:HIPCC -and $env:HIPCXX) {
                    break
                }
            }
        }
    }

    $hipConfig = Resolve-CommandPath "hipconfig"
    if ($hipConfig) {
        try {
            $hipCompilerRoot = (& $hipConfig -l).Trim()
            if ($hipCompilerRoot) {
                $clangxx = Join-Path $hipCompilerRoot "clang++.exe"
                $clang = Join-Path $hipCompilerRoot "clang.exe"
                if (Test-Path $clangxx) {
                    $env:HIPCXX = $clangxx
                }
                if (Test-Path $clang) {
                    $env:HIPCC = $clang
                    if (-not $env:HIPCXX) {
                        $env:HIPCXX = $clang
                    }
                }
            }
        } catch {
        }

        try {
            $hipRoot = (& $hipConfig -R).Trim()
            if ($hipRoot -and (Test-Path $hipRoot)) {
                $env:HIP_PATH = $hipRoot
                $env:ROCM_PATH = $hipRoot
                if (-not $hipPackage) {
                    $hipPackage = Resolve-HipPackageRoot
                    if ($hipPackage) {
                        $env:hip_DIR = $hipPackage.HipDir
                    }
                }
            }
        } catch {
        }
    }

    if (-not (Resolve-CommandPath "hipconfig") -and -not (Resolve-CommandPath "hipInfo") -and -not $rocmRoot) {
        throw "ROCm/HIP toolchain not found. Install ROCm on Windows and ensure hipconfig or hipInfo is available."
    }

    if (-not $hipPackage) {
        $hipPackage = Resolve-HipPackageRoot
    }
    if (-not $hipPackage) {
        throw "HIP package config not found. Expected hip-config.cmake under the HIP SDK installation."
    }
    $env:hip_DIR = $hipPackage.HipDir
    if (-not $env:HIPCC) {
        throw "HIP C compiler not found. Expected clang.exe in the HIP SDK installation."
    }
    if (-not $env:HIPCXX) {
        throw "HIP C++ compiler not found. Expected clang++.exe in the HIP SDK installation."
    }
}

function Ensure-VulkanToolchain {
    $sdkRoot = Resolve-VulkanSdkRoot
    if ($sdkRoot) {
        Add-ToPath (Join-Path $sdkRoot "Bin")
        Add-ToPath (Join-Path $sdkRoot "Bin32")
        if (-not $env:VULKAN_SDK) {
            $env:VULKAN_SDK = $sdkRoot
        }
        $env:CMAKE_PREFIX_PATH = if ($env:CMAKE_PREFIX_PATH) {
            "$sdkRoot;$env:CMAKE_PREFIX_PATH"
        } else {
            $sdkRoot
        }
    }

    $hasVulkanHeaders =
        ($env:VULKAN_SDK -and (Test-Path (Join-Path $env:VULKAN_SDK "Include\vulkan\vulkan.h"))) -or
        ($sdkRoot -and (Test-Path (Join-Path $sdkRoot "Include\vulkan\vulkan.h")))
    if (-not $hasVulkanHeaders) {
        throw "Vulkan SDK/development files not found. Install the Vulkan SDK and ensure VULKAN_SDK is configured."
    }

    if (-not (Resolve-CommandPath "glslc")) {
        throw "glslc not found. Install the Vulkan SDK and ensure its Bin directory is in PATH."
    }
}

function Copy-DevRuntimeBinaries {
    param(
        [Parameter(Mandatory = $true)]
        [string]$BackendName,
        [Parameter(Mandatory = $true)]
        [string]$BuildDir,
        [Parameter(Mandatory = $true)]
        [string]$RepoRoot
    )

    $sourceBinDir = Join-Path $BuildDir "bin"
    $targetDir = Join-Path $RepoRoot "target\release"
    New-Item -ItemType Directory -Force -Path $targetDir | Out-Null

    $flavoredCopies = @(
        @{ Source = "rpc-server.exe"; Target = "rpc-server-$BackendName.exe" },
        @{ Source = "llama-server.exe"; Target = "llama-server-$BackendName.exe" }
    )

    foreach ($copy in $flavoredCopies) {
        $source = Join-Path $sourceBinDir $copy.Source
        if (-not (Test-Path $source)) {
            throw "Expected llama.cpp binary not found: $source"
        }
        Copy-Item -LiteralPath $source -Destination (Join-Path $targetDir $copy.Target) -Force
    }

    foreach ($name in @("llama-moe-analyze.exe", "llama-moe-split.exe")) {
        $source = Join-Path $sourceBinDir $name
        if (Test-Path $source) {
            Copy-Item -LiteralPath $source -Destination (Join-Path $targetDir $name) -Force
        }
    }

    Write-Host "Staged llama.cpp runtime binaries in target\release with '$BackendName' flavor names."
}

function Invoke-InRepo {
    param(
        [scriptblock]$Script
    )

    Push-Location $repoRoot
    try {
        & $Script
    } finally {
        Pop-Location
    }
}

$Backend = Normalize-RecipeArgument $Backend @("backend")
$CudaArch = Normalize-RecipeArgument $CudaArch @("cuda_arch", "cudaarch")
$RocmArch = Normalize-RecipeArgument $RocmArch @("rocm_arch", "rocmarch", "amd_arch", "amdarch")

$backendName = Resolve-Backend $Backend
Write-Host "Using Windows backend: $backendName"

Ensure-MsvcToolchain
Configure-CompilerCache

switch ($backendName) {
    "cuda" {
        Ensure-CudaToolchain
        if ($CudaArch) {
            Write-Host "Using CUDA architectures: $CudaArch"
        } else {
            Write-Host "Using CUDA toolkit at: $(Split-Path -Parent (Resolve-CommandPath 'nvcc'))"
        }
    }
    "rocm" {
        Ensure-RocmToolchain
        if ($RocmArch) {
            Write-Host "Using AMDGPU targets: $RocmArch"
        }
    }
    "vulkan" {
        Ensure-VulkanToolchain
    }
    "cpu" {
        Write-Host "Building Windows backend: CPU only"
    }
    default {
        throw "Unsupported backend '$backendName'. Use one of: cuda, rocm, hip, vulkan, cpu."
    }
}

Invoke-InRepo {
    $llamaRepo = "https://github.com/Mesh-LLM/llama.cpp.git"
    $llamaPinSha = $env:MESH_LLM_LLAMA_PIN_SHA
    $llamaPinFile = Join-Path $repoRoot "LLAMA_CPP_SHA"
    if (-not $llamaPinSha -and (Test-Path $llamaPinFile)) {
        $llamaPinSha = (Get-Content -Path $llamaPinFile -Raw).Trim()
    }

    if (-not (Test-Path $llamaDir)) {
        if ($llamaPinSha) {
            Write-Host "Cloning Mesh-LLM/llama.cpp pinned to $llamaPinSha..."
            Invoke-NativeCommand "git" @("clone", "-b", "master", "--depth", "1", $llamaRepo, $llamaDir)
            Push-Location $llamaDir
            try {
                if (-not (Test-CommandSuccess "git" @("cat-file", "-e", "${llamaPinSha}^{commit}"))) {
                    Write-Host "Pinned SHA not on master tip, fetching explicitly..."
                    Invoke-NativeCommand "git" @("fetch", "--depth", "1", "origin", $llamaPinSha)
                }
                Invoke-NativeCommand "git" @("checkout", "--detach", $llamaPinSha)
            } finally {
                Pop-Location
            }
        } else {
            Write-Host "Cloning Mesh-LLM/llama.cpp (master)..."
            Invoke-NativeCommand "git" @("clone", "-b", "master", $llamaRepo, $llamaDir)
        }
    } else {
        Push-Location $llamaDir
        try {
            if (Test-CommandSuccess "git" @("remote", "get-url", "origin")) {
                Invoke-NativeCommand "git" @("remote", "set-url", "origin", $llamaRepo)
            } else {
                Invoke-NativeCommand "git" @("remote", "add", "origin", $llamaRepo)
            }

            if ($llamaPinSha) {
                if (-not (Test-CommandSuccess "git" @("cat-file", "-e", "${llamaPinSha}^{commit}"))) {
                    Write-Host "Fetching pinned llama.cpp SHA $llamaPinSha..."
                    Invoke-NativeCommand "git" @("fetch", "--depth", "1", "origin", $llamaPinSha)
                }
                $currentSha = (& git rev-parse HEAD).Trim()
                if ($currentSha -ne $llamaPinSha) {
                    Write-Host "Checking out pinned llama.cpp SHA $llamaPinSha (was $currentSha)..."
                    Invoke-NativeCommand "git" @("checkout", "--detach", $llamaPinSha)
                } else {
                    Write-Host "llama.cpp already at pinned SHA $llamaPinSha, no checkout needed"
                }
            } else {
                $currentBranch = (& git branch --show-current).Trim()
                if ($currentBranch -ne "master") {
                    Write-Host "Switching llama.cpp from '$currentBranch' to master..."
                    Invoke-NativeCommand "git" @("checkout", "master")
                }
                Write-Host "Pulling latest master from origin..."
                Invoke-NativeCommand "git" @("pull", "--ff-only", "origin", "master")
            }
        } finally {
            Pop-Location
        }
    }

    $cmakeArgs = @(
        "-B", $buildDir,
        "-S", $llamaDir,
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_CXX_FLAGS=/DPATH_MAX=4096",
        "-DGGML_RPC=ON",
        "-DGGML_METAL=OFF",
        "-DGGML_CUDA=OFF",
        "-DGGML_HIP=OFF",
        "-DGGML_VULKAN=OFF",
        "-DBUILD_SHARED_LIBS=OFF",
        "-DLLAMA_OPENSSL=OFF",
        "-DLLAMA_BUILD_TESTS=OFF",
        "-DGGML_BUILD_TESTS=OFF"
    )

    if (Resolve-CommandPath "ninja") {
        $cmakeArgs = @("-G", "Ninja") + $cmakeArgs
    }

    switch ($backendName) {
        "cuda" {
            $cmakeArgs += "-DGGML_CUDA=ON"
            if ($compilerCacheBin) {
                $cmakeArgs += "-DCMAKE_CUDA_COMPILER_LAUNCHER=$compilerCacheBin"
            }
            if ($CudaArch) {
                $cmakeArgs += "-DCMAKE_CUDA_ARCHITECTURES=$CudaArch"
            }
        }
        "rocm" {
            $cmakeArgs += "-DGGML_HIP=ON"
            if ($compilerCacheBin) {
                $cmakeArgs += "-DCMAKE_HIP_COMPILER_LAUNCHER=$compilerCacheBin"
            }
            if ($env:HIPCC) {
                $cmakeArgs += "-DCMAKE_C_COMPILER=$env:HIPCC"
            }
            if ($env:HIPCXX) {
                $cmakeArgs += "-DCMAKE_CXX_COMPILER=$env:HIPCXX"
            }
            if ($env:hip_DIR) {
                $cmakeArgs += "-Dhip_DIR=$env:hip_DIR"
            }
            if ($env:ROCM_PATH) {
                $cmakeArgs += "-DCMAKE_PREFIX_PATH=$env:ROCM_PATH"
            }
            if ($RocmArch) {
                $cmakeArgs += "-DAMDGPU_TARGETS=$RocmArch"
            }
        }
        "vulkan" {
            $cmakeArgs += "-DGGML_VULKAN=ON"
        }
        "cpu" {
        }
    }

    $cmakeArgs += $compilerLauncherArgs

    $parallelJobs = [Environment]::ProcessorCount
    Invoke-NativeCommand "cmake" $cmakeArgs
    Invoke-NativeCommand "cmake" @("--build", $buildDir, "--config", "Release", "--parallel", "$parallelJobs")
    Write-Host "Build complete: $buildDir\bin\"

    if (Test-Path $meshUiDir) {
        if (Test-UiBuildRequired -UiDirectory $meshUiDir) {
            Write-Host "Building mesh-llm UI..."
            Push-Location $meshUiDir
            try {
                if (Test-NpmInstallRequired -UiDirectory $meshUiDir) {
                    Invoke-NativeCommand "npm" @("ci")
                }
                Invoke-NativeCommand "npm" @("run", "build")
            } finally {
                Pop-Location
            }
        } else {
            Write-Host "Skipping mesh-llm UI build; dist is up to date."
        }
    }

    Write-Host "Building mesh-llm..."
    Invoke-NativeCommand "cargo" @("build", "--release", "--locked", "-p", "mesh-llm")
    Copy-DevRuntimeBinaries -BackendName $backendName -BuildDir $buildDir -RepoRoot $repoRoot
    Write-Host "Mesh binary: target\release\mesh-llm.exe"
}
