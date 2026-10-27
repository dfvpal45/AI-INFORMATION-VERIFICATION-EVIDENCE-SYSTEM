Set-Location $PSScriptRoot
Write-Host "Starting AIVES using Docker Compose. Start Docker Desktop first." -ForegroundColor Cyan
docker compose up --build
