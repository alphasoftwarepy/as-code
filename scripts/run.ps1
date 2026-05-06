<#
.SYNOPSIS
    AS Code Run Script
.DESCRIPTION
    Starts the AS Code FastAPI server.
#>

$ErrorActionPreference = "Stop"

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "         Starting AS Code Server         " -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan

if (-not (Test-Path "venv\Scripts\Activate.ps1")) {
    Write-Error "Virtual environment not found. Please run .\scripts\install.ps1 first."
    exit 1
}

# Activate venv
Write-Host "Activating virtual environment..."
.\venv\Scripts\Activate.ps1

# Optional: Open browser logic (can be expanded later if a specific port/URL is known)
# Write-Host "Opening local UI in browser..."
# Start-Process "http://127.0.0.1:8000"

# Start FastAPI server
Write-Host "Starting FastAPI server..." -ForegroundColor Green
Write-Host "Press Ctrl+C to stop the server." -ForegroundColor Yellow
Write-Host "-----------------------------------------"

# Ensure module can find 'api' and 'core'
$env:PYTHONPATH = (Get-Location).Path

# Adjust the command below if your main entrypoint differs. Assuming api/main.py or similar.
# A generic uvicorn command for FastAPI is typical. If there's a specific run script, use that.
# For now, falling back to a safe standard for a FastAPI app named app.main or api.main:
if (Test-Path "api\main.py") {
    uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
} else {
    # If the exact entrypoint isn't api/main.py, try core or app
    python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
}
