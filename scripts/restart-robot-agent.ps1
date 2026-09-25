$ErrorActionPreference = 'Stop'

$root = 'D:\ReachyAI'
$agentPath = 'D:\ReachyAI\scripts\robot-agent.py'

$targets = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -eq 'python.exe' -and $_.CommandLine -and
    $_.CommandLine.Contains($agentPath)
}
foreach ($target in $targets) {
    Stop-Process -Id $target.ProcessId -Force -ErrorAction SilentlyContinue
}

Start-Sleep -Milliseconds 700
& (Join-Path $root 'scripts\start-autonomous-agent.ps1')
if (-not $?) {
    throw 'start-autonomous-agent.ps1 failed'
}
Write-Host 'Reachy robot Agent restarted.'
