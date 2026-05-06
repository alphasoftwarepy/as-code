<#
.SYNOPSIS
    AS Code Installation Script
.DESCRIPTION
    Sets up the AS Code environment on Windows.
#>

$ErrorActionPreference = "Stop"

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "       AS Code - Setup Script            " -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan

# 1. Verify Python
Write-Host "`n[1/6] Verifying Python installation..."
if (-not (Get-Command "python" -ErrorAction SilentlyContinue)) {
    Write-Error "Python is not installed or not in PATH. Please install Python 3.10+."
    exit 1
}
$pythonVersion = python --version
Write-Host "Found: $pythonVersion" -ForegroundColor Green

# 2. Create venv
Write-Host "`n[2/6] Creating virtual environment (venv)..."
if (-not (Test-Path "venv")) {
    python -m venv venv
    Write-Host "Virtual environment created." -ForegroundColor Green
} else {
    Write-Host "Virtual environment already exists." -ForegroundColor Yellow
}

# 3. Install requirements
Write-Host "`n[3/6] Installing dependencies..."
.\venv\Scripts\Activate.ps1
if (Test-Path "requirements.txt") {
    pip install -r requirements.txt
    Write-Host "Dependencies installed." -ForegroundColor Green
} else {
    Write-Host "requirements.txt not found. Skipping." -ForegroundColor Yellow
}

# 4. Create folders if missing
Write-Host "`n[4/6] Ensuring required directories exist..."
$directories = @("models", "logs", "cache")
foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir | Out-Null
        Write-Host "Created directory: $dir" -ForegroundColor Green
    }
}

# 5. Generate .env if missing
Write-Host "`n[5/6] Checking for .env file..."

if (-not (Test-Path ".env")) {

    Set-Content -Path ".env" -Value "HOST=127.0.0.1`nPORT=8000"

    Write-Host "Created default .env file." -ForegroundColor Green

} else {

    Write-Host ".env file already exists." -ForegroundColor Yellow

}

# 6. Verify model folder and manual instructions
Write-Host "`n[6/6] Checking models directory..."
$modelsCount = (Get-ChildItem -Path "models" -Filter "*.litertlm" -ErrorAction SilentlyContinue | Measure-Object).Count
if ($modelsCount -eq 0) {
    Write-Host "`n=========================================" -ForegroundColor Yellow
    Write-Host "          ACTION REQUIRED                " -ForegroundColor Yellow
    Write-Host "=========================================" -ForegroundColor Yellow
    Write-Host "No .litertlm models found in the 'models' directory."
    Write-Host "Automatic model downloading is currently NOT supported."
    Write-Host ""
    Write-Host "MANUAL MODEL SETUP INSTRUCTIONS:"
    Write-Host "1. Download a .litertlm model file from HuggingFace or your preferred source."
    Write-Host "2. Place the downloaded file into the 'models' folder in this repository."
    Write-Host "3. The runtime will automatically detect it when you start the server."
    Write-Host "=========================================" -ForegroundColor Yellow
} else {
    Write-Host "Found $modelsCount model(s) in the 'models' directory." -ForegroundColor Green
}

Write-Host "`nSetup complete! You can now run the server using '.\scripts\run.ps1'." -ForegroundColor Cyan
