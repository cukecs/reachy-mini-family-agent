$ErrorActionPreference = 'Stop'

$root = 'D:\ReachyAI'
$startup = [Environment]::GetFolderPath('Startup')
$backup = Join-Path $root 'config\disabled-startup-shortcuts'
$names = @(
    'Reachy AI Stack.lnk',
    'Reachy Autonomous Agent.lnk',
    'Reachy Mini Daemon.lnk'
)

$resolvedRoot = [IO.Path]::GetFullPath($root).TrimEnd('\') + '\'
$resolvedBackup = [IO.Path]::GetFullPath($backup).TrimEnd('\') + '\'
if (-not $resolvedBackup.StartsWith($resolvedRoot, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing to move startup shortcuts outside the ReachyAI workspace: $resolvedBackup"
}

New-Item -ItemType Directory -Force -Path $backup | Out-Null
foreach ($name in $names) {
    $source = Join-Path $startup $name
    $destination = Join-Path $backup $name
    if (Test-Path -LiteralPath $source) {
        Move-Item -LiteralPath $source -Destination $destination -Force
        Write-Host "Disabled Windows logon startup: $name"
    } elseif (Test-Path -LiteralPath $destination) {
        Write-Host "Already disabled: $name"
    } else {
        Write-Host "Startup shortcut not found: $name"
    }
}

Write-Host 'Reachy startup is now managed by ReachyAI-Control.exe.' -ForegroundColor Green
