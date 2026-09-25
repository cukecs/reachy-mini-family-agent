$ErrorActionPreference = 'Stop'

$installer = 'D:\ReachyAI\runtime\ryzen-ai\NPU_RAI_376_WHQL\npu_sw_installer.exe'
$expectedSha256 = 'AA836CBFCAD5D0782C79B58F197AA50624AF37E7CB8311C5F94D85B0DC3CCAAD'
$archive = 'D:\ReachyAI\runtime\ryzen-ai\NPU_RAI_376_WHQL.zip'

if (-not (Test-Path -LiteralPath $installer)) {
    throw "AMD NPU driver installer not found: $installer"
}

if (Test-Path -LiteralPath $archive) {
    $actualSha256 = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash
    if ($actualSha256 -ne $expectedSha256) {
        throw 'AMD NPU driver archive checksum failed. Installation aborted.'
    }
}

$principal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
$isAdmin = $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    Write-Host 'Windows UAC will open. Approve the official AMD NPU driver, then reboot.' -ForegroundColor Yellow
    Start-Process -FilePath 'powershell.exe' -Verb RunAs -ArgumentList @(
        '-NoProfile',
        '-ExecutionPolicy', 'Bypass',
        '-File', ('"{0}"' -f $PSCommandPath)
    )
    exit 0
}

Write-Host 'Starting the official AMD NPU driver installer. Reboot when it finishes.'
$process = Start-Process -FilePath $installer -Wait -PassThru
if ($process.ExitCode -ne 0) {
    throw "NPU driver installer exit code: $($process.ExitCode)"
}

Write-Host 'Driver installation finished. Reboot, then run check-npu.ps1.' -ForegroundColor Green
