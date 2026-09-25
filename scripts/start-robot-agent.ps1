$ErrorActionPreference = 'Stop'

# Compatibility entry point only.  The GUI/start-autonomous-agent.ps1 owns the
# startup flow; do not keep a foreground PowerShell window alive around Agent.
& 'D:\ReachyAI\scripts\start-autonomous-agent.ps1'
