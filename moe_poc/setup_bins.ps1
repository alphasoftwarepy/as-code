param(
    [string]$BinsDir = "C:\as-code\moe_poc\bins",
    [string]$ModelsDir = "C:\as-code\moe_poc\models"
)

$ErrorActionPreference = "Stop"

Write-Host "============================================" -ForegroundColor Cyan
Write-Host " AS-Core MoE POC - Setup" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan

$llamaZip  = Join-Path $BinsDir "llama-cuda.zip"
$cudartZip = Join-Path $BinsDir "cudart.zip"

foreach ($zip in @($llamaZip, $cudartZip)) {
    if (-not (Test-Path $zip)) {
        Write-Host "ERROR: Missing $zip" -ForegroundColor Red
        exit 1
    }
    $item = Get-Item $zip
    $sizeMb = [math]::Round($item.Length / 1MB, 1)
    $leaf = Split-Path $zip -Leaf
    Write-Host "  Found: $leaf ($sizeMb MB)"
}

Write-Host ""
Write-Host "[1/3] Extracting llama-cuda.zip..."
Expand-Archive -Path $llamaZip -DestinationPath $BinsDir -Force
Write-Host "  Done."

Write-Host "[2/3] Extracting cudart.zip..."
Expand-Archive -Path $cudartZip -DestinationPath $BinsDir -Force
Write-Host "  Done."

Write-Host "[3/3] Verifying extracted files..."
$required = @(
    "llama-server.exe",
    "llama.dll",
    "ggml.dll"
)

foreach ($f in $required) {
    $p = Join-Path $BinsDir $f
    if (Test-Path $p) {
        Write-Host "  [OK] $f" -ForegroundColor Green
    } else {
        Write-Host "  [MISSING] $f" -ForegroundColor Red
    }
}

$cudaDlls = Get-ChildItem $BinsDir -Filter "*cuda*.dll" -ErrorAction SilentlyContinue
$cublasDlls = Get-ChildItem $BinsDir -Filter "*cublas*.dll" -ErrorAction SilentlyContinue

Write-Host "  CUDA/cublas DLLs found: $(($cudaDlls.Count + $cublasDlls.Count))"

Write-Host ""
Write-Host "[VERSION CHECK]"
$serverExe = Join-Path $BinsDir "llama-server.exe"
& $serverExe --version

Write-Host ""
Write-Host "[CUDA DEVICE CHECK]"
& $serverExe --list-devices

New-Item -ItemType Directory -Force -Path $ModelsDir | Out-Null
Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host " Setup complete." -ForegroundColor Green
Write-Host " Bins dir:   $BinsDir"
Write-Host " Models dir: $ModelsDir"
Write-Host "============================================" -ForegroundColor Cyan
