$ErrorActionPreference = 'Stop'
$env:PYTHONPATH = 'D:\ReachyAI\app'
$env:REACHY_LOCAL_API_KEY = [Environment]::GetEnvironmentVariable(
    'REACHY_LOCAL_API_KEY', 'User'
)
$env:AUTOWAKE_API_KEY = [Environment]::GetEnvironmentVariable(
    'AUTOWAKE_API_KEY', 'User'
)
$env:DEEPSEEK_API_KEY = [Environment]::GetEnvironmentVariable(
    'DEEPSEEK_API_KEY', 'User'
)
$env:ZHIPU_API_KEY = [Environment]::GetEnvironmentVariable(
    'ZHIPU_API_KEY', 'User'
)
$env:MODELSCOPE_API_KEY = [Environment]::GetEnvironmentVariable(
    'MODELSCOPE_API_KEY', 'User'
)
$python = 'D:\ReachyAI\.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    throw "Required Python environment is missing: $python"
}
& $python `
    -m reachy_ai.server --config 'D:\ReachyAI\config\settings.json'
