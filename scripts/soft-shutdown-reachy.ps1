$ErrorActionPreference = 'Stop'

$statusUrl = 'http://127.0.0.1:8000/api/daemon/status'
$startUrl = 'http://127.0.0.1:8000/api/daemon/start?wake_up=false'
$stopUrl = 'http://127.0.0.1:8000/api/daemon/stop?goto_sleep=true'

function Get-ReachyStatus {
    Invoke-RestMethod -Uri $statusUrl -TimeoutSec 3
}

# Stop the autonomous owner first so it releases camera and audio and cannot
# issue another search movement while the robot is entering its sleep pose.
$agents = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -eq 'python.exe' -and $_.CommandLine -and
    $_.CommandLine.Contains('D:\ReachyAI\scripts\robot-agent.py')
}
foreach ($agent in $agents) {
    Stop-Process -Id $agent.ProcessId -Force -ErrorAction SilentlyContinue
}
if ($agents) {
    Start-Sleep -Seconds 2
}

$status = Get-ReachyStatus
if ($status.error) {
    Write-Host "Daemon error detected; rebuilding process: $($status.error)" -ForegroundColor Yellow
    & 'D:\ReachyAI\scripts\repair-reachy-daemon.ps1'
    $status = Get-ReachyStatus
}

if ($status.state -eq 'stopped') {
    # Reconnect without playing the wake-up movement; a live backend is required
    # to execute the physical sleep trajectory.
    Invoke-RestMethod -Method Post -Uri $startUrl -TimeoutSec 5 | Out-Null
    $deadline = (Get-Date).AddSeconds(35)
    do {
        Start-Sleep -Milliseconds 750
        $status = Get-ReachyStatus
    } while ($status.state -in @('starting', 'stopped') -and (Get-Date) -lt $deadline)
}

if ($status.state -ne 'running') {
    throw "Reachy cannot enter a controllable state: state=$($status.state), error=$($status.error)"
}

Invoke-RestMethod -Method Post -Uri $stopUrl -TimeoutSec 5 | Out-Null
$deadline = (Get-Date).AddSeconds(25)
do {
    Start-Sleep -Milliseconds 750
    $status = Get-ReachyStatus
} while ($status.state -notin @('stopped', 'error') -and (Get-Date) -lt $deadline)

if ($status.state -ne 'stopped' -or $status.error) {
    throw "Soft shutdown failed: state=$($status.state), error=$($status.error)"
}

Write-Host 'Reachy is in the sleep pose, motors are disabled, and the hardware backend is stopped.' -ForegroundColor Green
