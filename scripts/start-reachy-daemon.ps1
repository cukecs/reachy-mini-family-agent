param(
    [switch]$ForceRestart
)

$ErrorActionPreference = 'Stop'

$root = 'D:\ReachyAI'
$python = Join-Path $root '.venv\Scripts\python.exe'
$logs = Join-Path $root 'logs'
$statusUrl = 'http://127.0.0.1:8000/api/daemon/status'
$moduleName = 'reachy_mini.daemon.app.main'

if (-not (Test-Path -LiteralPath $python)) {
    throw "Required Python is missing: $python"
}
New-Item -ItemType Directory -Path $logs -Force | Out-Null

function Get-ReachyStatus {
    try {
        return Invoke-RestMethod -Uri $statusUrl -TimeoutSec 2
    } catch {
        return $null
    }
}

function Get-ReachyDaemonProcesses {
    return @(Get-CimInstance Win32_Process | Where-Object {
        $_.CommandLine -and (
            $_.CommandLine.Contains($moduleName) -or
            $_.CommandLine.Contains('reachy_daemon_resilient.py') -or
            $_.Name -eq 'reachy-mini-daemon.exe'
        )
    })
}

if ($ForceRestart) {
    foreach ($target in (Get-ReachyDaemonProcesses)) {
        Write-Host "Stopping Reachy daemon PID $($target.ProcessId)"
        Stop-Process -Id $target.ProcessId -Force -ErrorAction SilentlyContinue
    }
    $deadline = (Get-Date).AddSeconds(10)
    do {
        Start-Sleep -Milliseconds 250
        $listener = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
    } while ($listener -and (Get-Date) -lt $deadline)
}

$status = Get-ReachyStatus
if ($status) {
    return
}

$listener = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
if ($listener) {
    throw 'Port 8000 is occupied by another process that is not a Reachy daemon.'
}

Write-Host 'Starting official Reachy Mini daemon'
Start-Process -FilePath $python -ArgumentList @(
    '-m', $moduleName,
    '--desktop-app-daemon', '--fastapi-host', '127.0.0.1',
    '--fastapi-port', '8000', '--autostart',
    '--no-wake-up-on-start', '--no-goto-sleep-on-stop', '--no-media',
    '--log-level', 'INFO', '--log-file', (Join-Path $logs 'reachy-daemon.log')
) -WorkingDirectory $root -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $logs 'reachy-daemon.stdout.log') `
    -RedirectStandardError (Join-Path $logs 'reachy-daemon.stderr.log')

$deadline = (Get-Date).AddSeconds(45)
do {
    Start-Sleep -Milliseconds 500
    $status = Get-ReachyStatus
} while (-not $status -and (Get-Date) -lt $deadline)

if (-not $status) {
    $details = ''
    $stderr = Join-Path $logs 'reachy-daemon.stderr.log'
    if (Test-Path -LiteralPath $stderr) {
        $details = (Get-Content -LiteralPath $stderr -Tail 20 -ErrorAction SilentlyContinue) -join "`n"
    }
    throw "Official Reachy daemon did not expose its API within 45 seconds.`n$details"
}

Write-Host "Reachy daemon API is ready: $($status.state)" -ForegroundColor Green
