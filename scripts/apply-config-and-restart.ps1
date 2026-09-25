$ErrorActionPreference = 'Stop'

& 'D:\ReachyAI\scripts\restart-ai-stack.ps1'
if (-not $?) {
    throw 'restart-ai-stack.ps1 failed'
}

& 'D:\ReachyAI\scripts\restart-robot-agent.ps1'
if (-not $?) {
    throw 'restart-robot-agent.ps1 failed'
}
Write-Host 'Reachy AI configuration applied; model router and Agent restarted.'
