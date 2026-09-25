$ErrorActionPreference = 'Stop'

$root = 'D:\ReachyAI'
$daemonStarter = Join-Path $root 'scripts\start-reachy-daemon.ps1'

if (-not (Test-Path -LiteralPath $daemonStarter)) {
    throw "Required file is missing: $daemonStarter"
}

& $daemonStarter -ForceRestart

$deadline = (Get-Date).AddSeconds(45)
do {
    Start-Sleep -Seconds 1
    try {
        $status = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/api/daemon/status' -TimeoutSec 2
    } catch {
        $status = $null
    }
} while ((-not $status -or $status.state -notin @('running', 'error')) -and (Get-Date) -lt $deadline)

if (-not $status) {
    throw 'Reachy daemon did not start within 45 seconds.'
}
if ($status.state -eq 'error') {
    throw "Reachy daemon still reports: $($status.error). Reconnect USB data and power, then retry."
}

Write-Host "Reachy daemon recovered: $($status.state)" -ForegroundColor Green
