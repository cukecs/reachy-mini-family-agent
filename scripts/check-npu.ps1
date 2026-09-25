$ErrorActionPreference = 'Stop'

$requiredDriver = [version]'32.0.203.280'
$npu = Get-PnpDevice -Class ComputeAccelerator -ErrorAction SilentlyContinue |
    Where-Object { $_.FriendlyName -match 'NPU|Neural' } |
    Select-Object -First 1

if (-not $npu) {
    Write-Host 'NPU compute accelerator was not found.' -ForegroundColor Red
    exit 1
}

$driver = Get-PnpDeviceProperty -InstanceId $npu.InstanceId -KeyName 'DEVPKEY_Device_DriverVersion' |
    Select-Object -ExpandProperty Data
$driverVersion = [version]$driver
$runtimePresent = Test-Path -LiteralPath 'C:\Program Files\RyzenAI'

Write-Host ('NPU: {0}' -f $npu.FriendlyName)
Write-Host ('Status: {0}' -f $npu.Status)
Write-Host ('Driver: {0}' -f $driverVersion)
Write-Host ('Ryzen AI runtime: {0}' -f $(if ($runtimePresent) { 'installed' } else { 'not installed' }))
Write-Host '8845H/Hawk Point role: INT8 CNN only; not LLM, VLM, ASR, or TTS.'

if ($driverVersion -lt $requiredDriver) {
    Write-Host ('Driver is older than Ryzen AI 1.8 minimum {0}. Run install-npu-driver.ps1.' -f $requiredDriver) -ForegroundColor Yellow
    exit 2
}

Write-Host 'NPU driver meets the Ryzen AI 1.8 minimum.' -ForegroundColor Green
