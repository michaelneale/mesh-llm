$model = "qwen2.5-3b-instruct-q4_k_m"
$logFile = "D:\IAs\mesh_chat_log.jsonl"
if (-not (Test-Path $logFile)) { New-Item $logFile -ItemType File | Out-Null }

$historial = @()
$turnos = 0

while ($true) {
    $msg = Read-Host "`nTú"
    if ($msg -eq "salir") { break }

    # Buscar recuerdos en MemPalace
    $recuerdos = python -m mempalace search $msg --wing mipalacio --max-results 3 2>&1 | Out-String
    if ($recuerdos -match "No results") { $recuerdos = "Sin recuerdos relevantes." }

    $systemMsg = @{
        role = "system"
        content = "Recuerdos de conversaciones pasadas:`n$recuerdos"
    }

    $body = @{
        model = $model
        messages = @($systemMsg) + $historial + @(@{ role = "user"; content = $msg })
    } | ConvertTo-Json -Depth 10

    try {
        $resp = Invoke-RestMethod -Uri http://localhost:9337/v1/chat/completions `
                                  -Method Post `
                                  -ContentType "application/json" `
                                  -Body $body `
                                  -ErrorAction Stop
        $content = $resp.choices[0].message.content
        Write-Host "IA: $content" -ForegroundColor Green

        $historial += @{ role = "user"; content = $msg }
        $historial += @{ role = "assistant"; content = $content }

        # Guardar en log
        $turn = @{ user = $msg; assistant = $content; timestamp = (Get-Date -Format "o") }
        $turn | ConvertTo-Json -Compress | Add-Content -Path $logFile

        # Re-indexar cada 3 turnos
        $turnos++
        if ($turnos % 3 -eq 0) {
            Write-Host "Actualizando memoria..." -ForegroundColor DarkGray
            python -m mempalace mine $logFile --mode convos --wing mipalacio *>$null
        }
    }
    catch {
        Write-Host "Error al conectar con el servidor. ¿Está corriendo Mesh-LLM?" -ForegroundColor Red
    }
}