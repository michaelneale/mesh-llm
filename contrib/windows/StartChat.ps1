param(
    [string]$Model = "Qwen2.5-3B-Instruct-Q4_K_M",
    [string]$BaseUrl = "http://localhost:9337/v1",
    [string]$LogFile = "",
    [int]$MaxTurns = 20
)

$ErrorActionPreference = "Stop"

if (-not $LogFile) {
    $LogFile = Join-Path $env:TEMP "mesh-llm-chat.jsonl"
}

if (-not (Test-Path $LogFile)) {
    New-Item -Path $LogFile -ItemType File -Force | Out-Null
}

$history = @()

while ($true) {
    $message = Read-Host "`nYou"
    if ($message -in @("exit", "quit")) {
        break
    }

    $messages = $history + @(@{ role = "user"; content = $message })
    $body = @{
        model = $Model
        messages = $messages
    } | ConvertTo-Json -Depth 16

    try {
        $response = Invoke-RestMethod `
            -Uri "$BaseUrl/chat/completions" `
            -Method Post `
            -ContentType "application/json" `
            -Body $body `
            -ErrorAction Stop

        $content = $response.choices[0].message.content
        Write-Host "Assistant: $content" -ForegroundColor Green

        $history += @{ role = "user"; content = $message }
        $history += @{ role = "assistant"; content = $content }
        if ($history.Count -gt ($MaxTurns * 2)) {
            $history = $history[($history.Count - ($MaxTurns * 2))..($history.Count - 1)]
        }

        @{
            user = $message
            assistant = $content
            timestamp = (Get-Date -Format "o")
        } | ConvertTo-Json -Compress | Add-Content -Path $LogFile
    } catch {
        Write-Host "Could not reach Mesh-LLM at $BaseUrl" -ForegroundColor Red
        Write-Host $_.Exception.Message -ForegroundColor DarkRed
    }
}
