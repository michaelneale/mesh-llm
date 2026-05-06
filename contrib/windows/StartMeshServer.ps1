param(
    [string]$Model = "Qwen2.5-3B-Instruct-Q4_K_M",
    [string]$Device = "Vulkan1",
    [int]$Port = 9337,
    [int]$ConsolePort = 3131,
    [string]$MeshLlm = "",
    [string[]]$ExtraArgs = @()
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $scriptDir "..\.."))

if (-not $MeshLlm) {
    $candidate = Join-Path $repoRoot "target\release\mesh-llm.exe"
    if (Test-Path $candidate) {
        $MeshLlm = $candidate
    } else {
        $MeshLlm = "mesh-llm"
    }
}

& $MeshLlm serve `
    --model $Model `
    --device $Device `
    --port $Port `
    --console $ConsolePort `
    @ExtraArgs
