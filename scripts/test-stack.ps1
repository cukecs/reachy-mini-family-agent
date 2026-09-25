$ErrorActionPreference = 'Stop'

$health = Invoke-RestMethod -Uri 'http://127.0.0.1:8077/health' -TimeoutSec 10
$health | ConvertTo-Json -Depth 8

$simple = @{
    messages = @(@{ role = 'user'; content = 'hello' })
} | ConvertTo-Json -Depth 6
Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:8077/route' `
    -ContentType 'application/json; charset=utf-8' -Body $simple | ConvertTo-Json

$complex = @{
    messages = @(@{ role = 'user'; content = 'Perform a deep architecture and security audit.' })
} | ConvertTo-Json -Depth 6
Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:8077/route' `
    -ContentType 'application/json; charset=utf-8' -Body $complex | ConvertTo-Json
