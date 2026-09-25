$ErrorActionPreference = 'Stop'

$targets = Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -and (
        ($_.Name -eq 'python.exe' -and $_.CommandLine.Contains('reachy_ai.server')) -or
        ($_.Name -eq 'llama-server.exe' -and $_.CommandLine.Contains('D:\ReachyAI'))
    )
}
foreach ($target in $targets) {
    Stop-Process -Id $target.ProcessId -Force -ErrorAction SilentlyContinue
}

Start-Sleep -Seconds 1
& 'D:\ReachyAI\scripts\start-stack-hidden.ps1'
Write-Host 'Reachy AI model and router restarted.'
