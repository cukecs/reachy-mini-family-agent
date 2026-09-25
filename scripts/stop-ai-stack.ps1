$ErrorActionPreference = 'Stop'

$targets = Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -and (
        ($_.Name -eq 'python.exe' -and $_.CommandLine.Contains('reachy_ai.server')) -or
        ($_.Name -eq 'llama-server.exe' -and $_.CommandLine.Contains('D:\ReachyAI')) -or
        ($_.Name -eq 'python.exe' -and $_.CommandLine.Contains('D:\ReachyAI\scripts\robot-agent.py'))
    )
}

foreach ($target in $targets) {
    Stop-Process -Id $target.ProcessId -Force -ErrorAction SilentlyContinue
}

Write-Host 'Reachy AI agent, router, and local model have stopped.'
