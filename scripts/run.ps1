<#
.SYNOPSIS
    AS Code Run Script
.DESCRIPTION
    Starts the AS Code Local AI Server.
#>

$ErrorActionPreference = "Stop"

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "         Starting AS Code Server         " -ForegroundColor Cyan
Write-Host "      Fast Local AI for Real Hardware    " -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan

if (-not (Test-Path "venv\Scripts\Activate.ps1")) {
    Write-Error "Virtual environment not found. Please run .\scripts\install.ps1 first."
    exit 1
}

# Activate venv
Write-Host "Activating virtual environment..."
.\venv\Scripts\Activate.ps1

# Force UTF-8 console encoding
chcp 65001 > $null
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

# Ensure Python can find 'api', 'core', 'providers', etc.
$env:PYTHONPATH = (Get-Location).Path

# Ensure uploads/ directory exists (for document sessions)
if (-not (Test-Path "uploads")) {
    New-Item -ItemType Directory -Path "uploads" | Out-Null
}

# Start FastAPI server
Write-Host "Starting FastAPI server..." -ForegroundColor Green
Write-Host "Press Ctrl+C to stop the server." -ForegroundColor Yellow
Write-Host "UI: http://localhost:8000" -ForegroundColor Cyan
Write-Host "-----------------------------------------"

if (Test-Path "api\main.py") {
    python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
} else {
    Write-Error "api\main.py not found. Are you running from the repository root?"
    exit 1
}
