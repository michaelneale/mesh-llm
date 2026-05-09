param(
    [Parameter(Position = 0)]
    [string]$Tool,

    [Parameter(Position = 1, ValueFromRemainingArguments = $true)]
    [string[]]$ToolArguments = @()
)

$ErrorActionPreference = "Stop"

function Show-FastLinkerInstallInstructions {
    Write-Error @"
Rust fast linker was not found for the Windows MSVC target.

Install one of these, then rerun the just command:
  rustup component add llvm-tools-preview

Or install LLVM lld-link:
  winget install LLVM.LLVM
  choco install llvm

The build looks for rust-lld.exe in the active Rust sysroot first, then falls
back to rust-lld.exe or lld-link.exe on PATH.
"@
}

function Resolve-FastRustLinker {
    try {
        $sysroot = (& rustc --print sysroot).Trim()
        if ($LASTEXITCODE -eq 0 -and $sysroot) {
            foreach ($target in @("x86_64-pc-windows-msvc", "aarch64-pc-windows-msvc")) {
                $candidate = Join-Path $sysroot "lib\rustlib\$target\bin\rust-lld.exe"
                if (Test-Path $candidate) {
                    return $candidate
                }
            }
        }
    } catch {
    }

    foreach ($name in @("rust-lld.exe", "lld-link.exe")) {
        $command = Get-Command $name -ErrorAction SilentlyContinue
        if ($command) {
            return $command.Source
        }
    }

    return $null
}

$linker = Resolve-FastRustLinker
if (-not $linker) {
    Show-FastLinkerInstallInstructions
    exit 1
}

$env:CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_LINKER = $linker
$env:CARGO_TARGET_AARCH64_PC_WINDOWS_MSVC_LINKER = $linker
Write-Host "Using Rust linker: $linker"

if ($Tool) {
    & $Tool @ToolArguments
    exit $LASTEXITCODE
}
