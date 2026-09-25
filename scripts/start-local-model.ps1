$ErrorActionPreference = 'Stop'

$root = 'D:\ReachyAI'
$modelConfigPath = Join-Path $root 'config\local-model.json'
$modelConfig = Get-Content -LiteralPath $modelConfigPath -Raw | ConvertFrom-Json
$server = [string]$modelConfig.server_path
$model = [string]$modelConfig.model_path
$mmproj = [string]$modelConfig.mmproj_path
$log = Join-Path $root 'logs\llama-server.log'

$localApiKey = [Environment]::GetEnvironmentVariable('REACHY_LOCAL_API_KEY', 'User')
if ([string]::IsNullOrWhiteSpace($localApiKey)) {
    $localApiKey = [Convert]::ToBase64String(
        [Security.Cryptography.RandomNumberGenerator]::GetBytes(32)
    )
    [Environment]::SetEnvironmentVariable('REACHY_LOCAL_API_KEY', $localApiKey, 'User')
}
$env:REACHY_LOCAL_API_KEY = $localApiKey
$env:LLAMA_API_KEY = $localApiKey

foreach ($required in @($server, $model, $mmproj)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Required file is missing: $required"
    }
}

$arguments = @(
    '--model', $model,
    '--mmproj', $mmproj,
    '--alias', [string]$modelConfig.alias,
    '--host', [string]$modelConfig.host,
    '--port', [string]$modelConfig.port,
    '--ctx-size', [string]$modelConfig.context_size,
    '--n-gpu-layers', [string]$modelConfig.gpu_layers,
    '--flash-attn', $(if ($modelConfig.flash_attention) { 'on' } else { 'off' }),
    '--cache-type-k', [string]$modelConfig.cache_type_k,
    '--cache-type-v', [string]$modelConfig.cache_type_v,
    '--parallel', [string]$modelConfig.parallel,
    '--image-min-tokens', [string]$modelConfig.image_min_tokens,
    '--jinja'
)

if ($modelConfig.PSObject.Properties.Name -contains 'image_max_tokens') {
    $arguments += @('--image-max-tokens', [string]$modelConfig.image_max_tokens)
}
if (($modelConfig.PSObject.Properties.Name -contains 'mmproj_offload') -and -not [bool]$modelConfig.mmproj_offload) {
    $arguments += '--no-mmproj-offload'
}
if ($modelConfig.PSObject.Properties.Name -contains 'reasoning_budget') {
    $arguments += @('--reasoning-budget', [string]$modelConfig.reasoning_budget)
}

& $server @arguments 2>&1 | Tee-Object -FilePath $log -Append
