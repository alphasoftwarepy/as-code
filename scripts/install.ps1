<#
.SYNOPSIS
    AS Code Setup Script
.DESCRIPTION
    Fast, lightweight, general-purpose local AI runtime setup for Windows.
    Prepares venv, dependencies, directories, .env, and validates litert-lm CLI.
#>

$ErrorActionPreference = "Stop"

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "       AS Code - Local AI Setup          " -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan

# ── 1. Verify Python ───────────────────────────────────────────
Write-Host "`n[1/8] Verifying Python installation..."
if (-not (Get-Command "python" -ErrorAction SilentlyContinue)) {
    Write-Error "Python is not installed or not in PATH. Please install Python 3.10+."
    exit 1
}
$pythonVersion = python --version
Write-Host "Found: $pythonVersion" -ForegroundColor Green

# ── 2. Create venv ─────────────────────────────────────────────
Write-Host "`n[2/8] Creating virtual environment (venv)..."
if (-not (Test-Path "venv")) {
    python -m venv venv
    Write-Host "Virtual environment created." -ForegroundColor Green
} else {
    Write-Host "Virtual environment already exists." -ForegroundColor Yellow
}

# ── 3. Install requirements ────────────────────────────────────
Write-Host "`n[3/8] Installing dependencies..."
.\venv\Scripts\Activate.ps1
if (Test-Path "requirements.txt") {
    pip install -r requirements.txt --quiet
    Write-Host "Dependencies installed." -ForegroundColor Green
} else {
    Write-Host "requirements.txt not found. Skipping." -ForegroundColor Yellow
}

# ── 4. Create required directories ────────────────────────────
Write-Host "`n[4/8] Ensuring required directories exist..."
$directories = @("models", "models\gemma", "uploads", "logs", "cache")
foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir | Out-Null
        Write-Host "  Created: $dir" -ForegroundColor Green
    }
}
Write-Host "Directories ready." -ForegroundColor Green

# ── 5. Generate .env from .env.example ────────────────────────
Write-Host "`n[5/8] Checking for .env file..."
if (-not (Test-Path ".env")) {
    if (Test-Path ".env.example") {
        Copy-Item ".env.example" ".env"
        Write-Host "Created .env from .env.example." -ForegroundColor Green
    } else {
        # Minimal fallback
        Set-Content -Path ".env" -Value "HOST=127.0.0.1`nPORT=8000`nASCODE_LITERT_BACKEND=gpu"
        Write-Host "Created minimal .env (no .env.example found)." -ForegroundColor Yellow
    }
} else {
    Write-Host ".env already exists." -ForegroundColor Yellow
}

# ── 6. Detect GPU and adjust backend ──────────────────────────
Write-Host "`n[6/8] Detecting GPU..."
$hasGpu = $false
try {
    $nvidiaSmi = Get-Command "nvidia-smi" -ErrorAction SilentlyContinue
    if ($nvidiaSmi) {
        $gpuInfo = nvidia-smi --query-gpu=name --format=csv,noheader 2>$null
        if ($gpuInfo) {
            Write-Host "  GPU detected: $gpuInfo" -ForegroundColor Green
            $hasGpu = $true
        }
    }
} catch {}

if (-not $hasGpu) {
    Write-Host "  No NVIDIA GPU detected. Setting backend to CPU." -ForegroundColor Yellow
    # Update ASCODE_LITERT_BACKEND in .env to cpu
    if (Test-Path ".env") {
        $envContent = Get-Content ".env"
        $envContent = $envContent -replace "ASCODE_LITERT_BACKEND=gpu", "ASCODE_LITERT_BACKEND=cpu"
        if ($envContent -notmatch "ASCODE_LITERT_BACKEND") {
            $envContent += "`nASCODE_LITERT_BACKEND=cpu"
        }
        Set-Content -Path ".env" -Value $envContent
        Write-Host "  Set ASCODE_LITERT_BACKEND=cpu in .env" -ForegroundColor Yellow
    }
} else {
    Write-Host "  GPU backend enabled (ASCODE_LITERT_BACKEND=gpu)." -ForegroundColor Green
}

# ── 7. Check litert-lm CLI ────────────────────────────────────
Write-Host "`n[7/8] Checking litert-lm CLI..."
$litert = Get-Command "litert-lm" -ErrorAction SilentlyContinue
if (-not $litert) {
    Write-Host ""
    Write-Host "  WARNING: 'litert-lm' not found in PATH." -ForegroundColor Yellow
    Write-Host "  AS Code requires the LiteRT-LM CLI to run inference." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  Install it with:" -ForegroundColor Cyan
    Write-Host "    uv tool install litert-lm" -ForegroundColor White
    Write-Host ""
    Write-Host "  If you don't have 'uv', install it first:" -ForegroundColor Cyan
    Write-Host "    winget install astral-sh.uv" -ForegroundColor White
    Write-Host "    (or: pip install uv)" -ForegroundColor White
    Write-Host ""
} else {
    Write-Host "  litert-lm found: $($litert.Source)" -ForegroundColor Green
}

# ── 8. Check for models ───────────────────────────────────────
Write-Host "`n[8/8] Checking models directory..."
$modelsCount = (Get-ChildItem -Path "models" -Filter "*.litertlm" -Recurse -ErrorAction SilentlyContinue | Measure-Object).Count
if ($modelsCount -eq 0) {
    Write-Host ""
    Write-Host "=========================================" -ForegroundColor Yellow
    Write-Host "           ACTION REQUIRED               " -ForegroundColor Yellow
    Write-Host "=========================================" -ForegroundColor Yellow
    Write-Host "No .litertlm models found."
    Write-Host ""
    Write-Host "Download the model file and place it here:"
    Write-Host "  models\gemma\gemma-3n-E2B-it-int4.litertlm" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Download from HuggingFace:"
    Write-Host "  https://huggingface.co/google/gemma-3n-E2B-it-litert-lm" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  (Look for the file ending in '-int4.litertlm')"
    Write-Host "=========================================" -ForegroundColor Yellow
} else {
    Write-Host "  Found $modelsCount model(s)." -ForegroundColor Green
}

Write-Host ""
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "  Setup complete!" -ForegroundColor Cyan
Write-Host "  Run the server:  .\scripts\run.ps1" -ForegroundColor White
Write-Host "  Open browser:    http://localhost:8000" -ForegroundColor White
Write-Host "=========================================" -ForegroundColor Cyan
