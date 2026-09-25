$ErrorActionPreference = 'Stop'

$root = 'D:\ReachyAI'
$runtime = Join-Path $root 'runtime\llama.cpp\b11011'
$modelDir = Join-Path $root 'models\agents-a1-4b'
$binDir = Join-Path $runtime 'bin'

$downloads = @(
    @{
        Url = 'https://github.com/ggml-org/llama.cpp/releases/download/b11011/llama-b11011-bin-win-cuda-12.4-x64.zip'
        Path = Join-Path $runtime 'llama-b11011-bin-win-cuda-12.4-x64.zip'
    },
    @{
        Url = 'https://github.com/ggml-org/llama.cpp/releases/download/b11011/cudart-llama-bin-win-cuda-12.4-x64.zip'
        Path = Join-Path $runtime 'cudart-llama-bin-win-cuda-12.4-x64.zip'
    },
    @{
        Url = 'https://huggingface.co/InternScience/Agents-A1-4B-Q4_K_M-GGUF/resolve/main/Agents-A1-4B-Q4_K_M.gguf'
        Path = Join-Path $modelDir 'Agents-A1-4B-Q4_K_M.gguf'
    },
    @{
        Url = 'https://huggingface.co/InternScience/Agents-A1-4B-Q4_K_M-GGUF/resolve/main/Agents-A1-4B-mmproj.gguf'
        Path = Join-Path $modelDir 'Agents-A1-4B-mmproj.gguf'
    }
)

New-Item -ItemType Directory -Path $runtime, $modelDir, $binDir -Force | Out-Null
foreach ($item in $downloads) {
    Write-Host "Downloading/resuming $($item.Path)"
    & curl.exe -L --fail --retry 20 --retry-delay 3 --retry-all-errors `
        -C - -o $item.Path $item.Url
    if ($LASTEXITCODE -ne 0) { throw "Download failed: $($item.Path)" }
}

Write-Host 'Extracting llama.cpp CUDA runtime'
Expand-Archive -LiteralPath $downloads[0].Path -DestinationPath $binDir -Force
Expand-Archive -LiteralPath $downloads[1].Path -DestinationPath $binDir -Force

$server = Join-Path $binDir 'llama-server.exe'
if (-not (Test-Path -LiteralPath $server)) {
    $found = Get-ChildItem -LiteralPath $binDir -Filter 'llama-server.exe' -Recurse | Select-Object -First 1
    if (-not $found) { throw 'llama-server.exe was not found after extraction' }
    Write-Host "llama-server path: $($found.FullName)"
} else {
    & $server --version
}
