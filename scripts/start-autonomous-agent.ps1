$ErrorActionPreference = 'Stop'

$root = 'D:\ReachyAI'
$python = Join-Path $root '.venv\Scripts\python.exe'
$agent = Join-Path $root 'scripts\robot-agent.py'
$env:HF_HOME = Join-Path $root 'cache\huggingface'

& (Join-Path $root 'scripts\start-reachy.ps1')
if (-not $?) {
    throw 'start-reachy.ps1 failed'
}

$daemonDeadline = (Get-Date).AddSeconds(45)
do {
    Start-Sleep -Milliseconds 500
    $daemonReady = Get-NetTCPConnection -LocalPort 8000 -State Listen `
        -ErrorAction SilentlyContinue
} while (-not $daemonReady -and (Get-Date) -lt $daemonDeadline)
if (-not $daemonReady) {
    throw 'Reachy daemon did not become ready on port 8000'
}

& (Join-Path $root 'scripts\start-stack-hidden.ps1')

$running = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -eq 'python.exe' -and $_.CommandLine -and
    $_.CommandLine.Contains('D:\ReachyAI\scripts\robot-agent.py')
}

if (-not $running) {
    $env:PYTHONPATH = Join-Path $root 'app'
    $env:PYTHONUNBUFFERED = '1'
    $env:PYTHONIOENCODING = 'utf-8'
    Start-Process -FilePath $python -ArgumentList @($agent) `
        -WorkingDirectory $root -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $root 'logs\robot-agent.stdout.log') `
        -RedirectStandardError (Join-Path $root 'logs\robot-agent.stderr.log')
}
