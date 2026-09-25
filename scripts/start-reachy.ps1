$ErrorActionPreference = 'Stop'

$root = 'D:\ReachyAI'
$python = Join-Path $root '.venv\Scripts\python.exe'
$daemonStarter = Join-Path $root 'scripts\start-reachy-daemon.ps1'
$statusUrl = 'http://127.0.0.1:8000/api/daemon/status'

foreach ($required in @($daemonStarter, $python)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Required file is missing: $required"
    }
}

function Get-ReachyStatus {
    try {
        return Invoke-RestMethod -Uri $statusUrl -TimeoutSec 3
    } catch {
        return $null
    }
}

$status = Get-ReachyStatus
if (-not $status) {
    & $daemonStarter
    $status = Get-ReachyStatus
}

if (-not $status) {
    throw 'Reachy daemon did not expose its API.'
}

if ($status.state -eq 'error') {
    Write-Host "Daemon error detected; rebuilding process: $($status.error)" -ForegroundColor Yellow
    & 'D:\ReachyAI\scripts\repair-reachy-daemon.ps1'
    $status = Get-ReachyStatus
}

if ($status.state -eq 'stopped') {
    Invoke-RestMethod -Method Post `
        -Uri 'http://127.0.0.1:8000/api/daemon/start?wake_up=false' `
        -TimeoutSec 5 | Out-Null
    $deadline = (Get-Date).AddSeconds(35)
    do {
        Start-Sleep -Milliseconds 750
        $status = Get-ReachyStatus
    } while ($status.state -notin @('running', 'error') -and (Get-Date) -lt $deadline)
}

if ($status.state -ne 'running' -or $status.error) {
    throw "Reachy failed to start: state=$($status.state), error=$($status.error)"
}

# Lift the head over the control WebSocket only. Constructing ReachyMini here
# would also create a media client and can either race Windows audio (LOCAL) or
# wait on WebRTC shutdown. This helper never touches camera or audio.
& $python 'D:\ReachyAI\scripts\wake-reachy-control.py'
if ($LASTEXITCODE -ne 0) {
    throw "Reachy wake-up action failed with exit code $LASTEXITCODE"
}

Write-Host 'Reachy is awake and ready.' -ForegroundColor Green
