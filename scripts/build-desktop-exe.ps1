$ErrorActionPreference = 'Stop'

$root = 'D:\ReachyAI'
$python = Join-Path $root '.venv\Scripts\python.exe'
$entry = Join-Path $root 'scripts\desktop-console.py'
$dist = Join-Path $root 'dist'
$build = Join-Path $root 'build\desktop-console'
$spec = Join-Path $root 'build\ReachyAI-Control.spec'

& $python -m PyInstaller `
    --noconfirm --clean --onefile --windowed `
    --name 'ReachyAI-Control' `
    --distpath $dist --workpath $build --specpath (Join-Path $root 'build') `
    --collect-all PIL `
    $entry

if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller failed with exit code $LASTEXITCODE"
}

Write-Host "Built: $dist\ReachyAI-Control.exe" -ForegroundColor Green
