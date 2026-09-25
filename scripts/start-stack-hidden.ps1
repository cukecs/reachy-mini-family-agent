$ErrorActionPreference = 'Stop'

$root = 'D:\ReachyAI'
$env:HF_HOME = Join-Path $root 'cache\huggingface'
$python = Join-Path $root '.venv\Scripts\python.exe'
$modelConfigPath = Join-Path $root 'config\local-model.json'
$modelConfig = Get-Content -LiteralPath $modelConfigPath -Raw | ConvertFrom-Json
$server = [string]$modelConfig.server_path
$model = [string]$modelConfig.model_path
$mmproj = [string]$modelConfig.mmproj_path

$localApiKey = [Environment]::GetEnvironmentVariable('REACHY_LOCAL_API_KEY', 'User')
if ([string]::IsNullOrWhiteSpace($localApiKey)) {
    $localApiKey = [Convert]::ToBase64String(
        [Security.Cryptography.RandomNumberGenerator]::GetBytes(32)
    )
    [Environment]::SetEnvironmentVariable('REACHY_LOCAL_API_KEY', $localApiKey, 'User')
}
$env:REACHY_LOCAL_API_KEY = $localApiKey
$env:LLAMA_API_KEY = $localApiKey
$env:DEEPSEEK_API_KEY = [Environment]::GetEnvironmentVariable(
    'DEEPSEEK_API_KEY', 'User'
)

if (-not (Get-NetTCPConnection -LocalPort $modelConfig.port -State Listen -ErrorAction SilentlyContinue)) {
    $llamaArgs = @(
        '--model', $model, '--mmproj', $mmproj,
        '--alias', [string]$modelConfig.alias,
        '--host', [string]$modelConfig.host, '--port', [string]$modelConfig.port,
        '--ctx-size', [string]$modelConfig.context_size,
        '--n-gpu-layers', [string]$modelConfig.gpu_layers,
        '--flash-attn', $(if ($modelConfig.flash_attention) { 'on' } else { 'off' }),
        '--cache-type-k', [string]$modelConfig.cache_type_k,
        '--cache-type-v', [string]$modelConfig.cache_type_v,
        '--parallel', [string]$modelConfig.parallel,
        '--image-min-tokens', [string]$modelConfig.image_min_tokens, '--jinja'
    )
    if ($modelConfig.PSObject.Properties.Name -contains 'image_max_tokens') {
        $llamaArgs += @('--image-max-tokens', [string]$modelConfig.image_max_tokens)
    }
    if (($modelConfig.PSObject.Properties.Name -contains 'mmproj_offload') -and -not [bool]$modelConfig.mmproj_offload) {
        $llamaArgs += '--no-mmproj-offload'
    }
    if ($modelConfig.PSObject.Properties.Name -contains 'reasoning_budget') {
        $llamaArgs += @('--reasoning-budget', [string]$modelConfig.reasoning_budget)
    }
    Start-Process -FilePath $server -ArgumentList $llamaArgs -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $root 'logs\llama-server.stdout.log') `
        -RedirectStandardError (Join-Path $root 'logs\llama-server.stderr.log')
}

$deadline = (Get-Date).AddMinutes(2)
do {
    Start-Sleep -Seconds 1
    try {
        $headers = @{ Authorization = "Bearer $localApiKey" }
        $ready = Invoke-RestMethod -Uri ("http://$($modelConfig.host):$($modelConfig.port)/health") `
            -Headers $headers -TimeoutSec 2
    } catch {
        $ready = $null
    }
} while (-not $ready -and (Get-Date) -lt $deadline)

if (-not $ready) { throw 'llama.cpp did not become ready within two minutes' }

if (-not (Get-NetTCPConnection -LocalPort 8077 -State Listen -ErrorAction SilentlyContinue)) {
    $env:PYTHONPATH = Join-Path $root 'app'
    Start-Process -FilePath $python -ArgumentList @(
        '-m', 'reachy_ai.server', '--config', (Join-Path $root 'config\settings.json')
    ) -WorkingDirectory (Join-Path $root 'app') -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $root 'logs\router.stdout.log') `
        -RedirectStandardError (Join-Path $root 'logs\router.stderr.log')
}
