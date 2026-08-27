param(
    [string]$ModelPath    = "C:\as-code\moe_poc\models\qwen1.5-moe-a2.7b-q4_k_m.gguf",
    [int]   $NGL          = 8,
    [int]   $Port         = 8766,
    [int]   $WarmupTokens = 10,
    [int]   $BenchTokens  = 50,
    [int]   $NRuns        = 3,
    [string]$BinsDir      = "C:\as-code\moe_poc\bins",
    [string]$LogFile      = "C:\as-code\moe_poc\qwen_server_stderr.log",
    [string]$OutputJson   = "C:\as-code\moe_poc\qwen_benchmark_results.json"
)

$ErrorActionPreference = "Stop"

function Get-VRAMFree {
    try {
        $out = & nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>$null
        return [int]$out.Trim()
    } catch { return -1 }
}

function Get-VRAMUsed {
    try {
        $out = & nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>$null
        return [int]$out.Trim()
    } catch { return -1 }
}

function Get-GPUUtil {
    try {
        $out = & nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>$null
        return [int]$out.Trim()
    } catch { return -1 }
}

function Get-GPUPower {
    try {
        $out = & nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits 2>$null
        return [double]$out.Trim()
    } catch { return -1 }
}

function Get-RAMAvailableGB {
    try {
        $out = python -c "import psutil; print(f'{psutil.virtual_memory().available / 1024**3:.2f}')" 2>$null
        return [double]$out.Trim()
    } catch { return -1 }
}

function Get-CPUUtilPct {
    try {
        $out = python -c "import psutil; print(f'{psutil.cpu_percent(interval=0.2):.1f}')" 2>$null
        return [double]$out.Trim()
    } catch { return -1 }
}

function Wait-ServerReady([int]$port, [int]$timeoutSec = 120) {
    $start = Get-Date
    Write-Host "  Waiting for llama-server on port $port..." -NoNewline
    while ((Get-Date) -lt $start.AddSeconds($timeoutSec)) {
        try {
            $resp = Invoke-RestMethod -Uri "http://127.0.0.1:$port/health" -TimeoutSec 2 -ErrorAction Stop
            if ($resp.status -eq "ok" -or $resp -match "ok") {
                Write-Host " READY" -ForegroundColor Green
                return $true
            }
        } catch {}
        Start-Sleep -Milliseconds 500
        Write-Host "." -NoNewline
    }
    Write-Host " TIMEOUT" -ForegroundColor Red
    return $false
}

function Invoke-ChatCompletion([int]$port, [string]$prompt, [int]$maxTokens) {
    $body = @{
        model       = "qwen-moe"
        messages    = @(@{ role = "user"; content = $prompt })
        max_tokens  = $maxTokens
        temperature = 0.0
        stream      = $false
    } | ConvertTo-Json -Depth 5

    $start = Get-Date
    $resp  = Invoke-RestMethod -Uri "http://127.0.0.1:$port/v1/chat/completions" `
                               -Method POST `
                               -ContentType "application/json" `
                               -Body $body `
                               -TimeoutSec 300
    $elapsed = ((Get-Date) - $start).TotalSeconds

    $usage = $resp.usage
    $tokensGenerated = if ($usage.completion_tokens) { [int]$usage.completion_tokens } else { $maxTokens }

    return @{
        tokens_generated = $tokensGenerated
        elapsed_sec      = $elapsed
        tok_s            = [math]::Round($tokensGenerated / $elapsed, 2)
        prompt_tokens    = if ($usage.prompt_tokens) { [int]$usage.prompt_tokens } else { 0 }
        content          = $resp.choices[0].message.content
    }
}

Write-Host ""
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host " AS-Core MoE POC - FASE 0.5: MODEL > VRAM BENCHMARK" -ForegroundColor Cyan
Write-Host " Qwen1.5-MoE-A2.7B (8.84 GB) | GTX 1650 Ti (4 GB VRAM)" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host ""

$serverExe = Join-Path $BinsDir "llama-server.exe"
if (-not (Test-Path $serverExe)) {
    Write-Host "ERROR: llama-server.exe not found at $serverExe" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $ModelPath)) {
    Write-Host "ERROR: Model not found: $ModelPath" -ForegroundColor Red
    exit 1
}

$modelItem = Get-Item $ModelPath
$modelSizeGB = [math]::Round($modelItem.Length / 1GB, 2)
Write-Host "[MODEL INFO]"
Write-Host "  File:         $ModelPath"
Write-Host "  Size:         $modelSizeGB GB"

Write-Host ""
Write-Host "[HARDWARE BASELINE]"
$gpuInfo = & nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader,nounits
Write-Host "  GPU:          $gpuInfo"
$vramFreeStart = Get-VRAMFree
Write-Host "  VRAM Free:    $vramFreeStart MB"
$ramFreeStart = Get-RAMAvailableGB
Write-Host "  RAM Avail:    $ramFreeStart GB"
Write-Host ""

Write-Host "[SERVER STARTUP] Launching with -ngl $NGL --ctx-size 2048..."
$serverArgs = @(
    "-m", $ModelPath,
    "--port", $Port,
    "-ngl", $NGL,
    "--ctx-size", "2048",
    "--threads", "6",
    "-v"
)

if (Test-Path $LogFile) { Remove-Item $LogFile -Force }

$loadStart = Get-Date

$serverProc = Start-Process -FilePath $serverExe `
                             -ArgumentList $serverArgs `
                             -WorkingDirectory $BinsDir `
                             -PassThru `
                             -WindowStyle Hidden `
                             -RedirectStandardError $LogFile

Write-Host "  Process PID: $($serverProc.Id)"

$ready = Wait-ServerReady -port $Port -timeoutSec 120
if (-not $ready) {
    Write-Host "Server failed to start or crashed on allocation. Checking log..." -ForegroundColor Red
    if (Test-Path $LogFile) {
        Get-Content $LogFile | Select-Object -Last 30
    }
    Stop-Process -Id $serverProc.Id -Force -ErrorAction SilentlyContinue
    exit 1
}

$loadTime = ((Get-Date) - $loadStart).TotalSeconds
$vramAfter = Get-VRAMFree
$vramUsedByModel = $vramFreeStart - $vramAfter
$ramAfter = Get-RAMAvailableGB
$ramConsumed = [math]::Round($ramFreeStart - $ramAfter, 2)

Write-Host ""
Write-Host "[MEMORY DISTRIBUTION OBSERVED]"
Write-Host "  Model Load Time:   $([math]::Round($loadTime, 1)) s"
Write-Host "  VRAM allocated:    $vramUsedByModel MB (in GPU)"
Write-Host "  VRAM free now:     $vramAfter MB"
Write-Host "  RAM allocated:     $ramConsumed GB (in Host RAM)"
Write-Host "  RAM free now:      $ramAfter GB"
Write-Host ""

Write-Host "[WARMUP] Generating $WarmupTokens tokens..."
$warmup = Invoke-ChatCompletion -port $Port -prompt "Count from 1 to 5." -maxTokens $WarmupTokens
Write-Host "  Warmup speed: $($warmup.tok_s) tok/s ($($warmup.tokens_generated) tokens in $([math]::Round($warmup.elapsed_sec,2)) s)"
Write-Host ""

Write-Host "[BENCHMARK RUNS] $NRuns runs x $BenchTokens tokens each"
Write-Host ""

$prompts = @(
    "Explain the concept of entropy in information theory.",
    "Write a quicksort algorithm in Python with comments.",
    "What are the main architectural components of a Transformer model?"
)

$results = @()
for ($i = 0; $i -lt $NRuns; $i++) {
    $prompt = $prompts[$i % $prompts.Count]
    $gpuUtil = Get-GPUUtil
    $cpuUtil = Get-CPUUtilPct
    $gpuPower = Get-GPUPower
    $vramNow = Get-VRAMUsed

    Write-Host "  Run $($i+1)/$NRuns..." -NoNewline
    $r = Invoke-ChatCompletion -port $Port -prompt $prompt -maxTokens $BenchTokens

    $tokColor = if ($r.tok_s -ge 10) { "Green" } else { "Yellow" }
    Write-Host " $($r.tok_s) tok/s | $($r.tokens_generated) tok in $([math]::Round($r.elapsed_sec,2))s | GPU: ${gpuUtil}% | CPU: ${cpuUtil}% | Power: ${gpuPower}W" -ForegroundColor $tokColor

    $results += @{
        run               = $i + 1
        prompt_tokens     = $r.prompt_tokens
        completion_tokens = $r.tokens_generated
        elapsed_sec       = $r.elapsed_sec
        tok_s             = $r.tok_s
        gpu_util_pct      = $gpuUtil
        cpu_util_pct      = $cpuUtil
        gpu_power_w       = $gpuPower
        vram_used_mb      = $vramNow
    }
}

Write-Host ""
Write-Host "[TTFT] Measuring Time-To-First-Token..."
$ttftStart = Get-Date
$ttftBody  = @{
    model       = "qwen-moe"
    messages    = @(@{ role = "user"; content = "Hi" })
    max_tokens  = 1
    stream      = $false
    temperature = 0.0
} | ConvertTo-Json -Depth 5

try {
    Invoke-RestMethod -Uri "http://127.0.0.1:$Port/v1/chat/completions" -Method POST -ContentType "application/json" -Body $ttftBody -TimeoutSec 30 | Out-Null
    $ttftMs = ((Get-Date) - $ttftStart).TotalMilliseconds
    Write-Host "  TTFT: $([math]::Round($ttftMs,1)) ms"
} catch {
    $ttftMs = -1
    Write-Host "  TTFT: measurement failed"
}
Write-Host ""

$tokValues = $results | ForEach-Object { [double]$_.tok_s }
$avgTokS   = [math]::Round(($tokValues | Measure-Object -Average).Average, 2)
$minTokS   = [math]::Round(($tokValues | Measure-Object -Minimum).Minimum, 2)
$maxTokS   = [math]::Round(($tokValues | Measure-Object -Maximum).Maximum, 2)
$sortedTok = $tokValues | Sort-Object
$medTokS   = $sortedTok[[math]::Floor($sortedTok.Count / 2)]

$avgGPU    = [math]::Round(($results | ForEach-Object { [double]$_.gpu_util_pct } | Measure-Object -Average).Average, 0)
$avgCPU    = [math]::Round(($results | ForEach-Object { [double]$_.cpu_util_pct } | Measure-Object -Average).Average, 0)
$avgPower  = [math]::Round(($results | ForEach-Object { [double]$_.gpu_power_w } | Measure-Object -Average).Average, 1)

Write-Host "========================================================" -ForegroundColor Cyan
Write-Host " BASELINE 1 RESULTS SUMMARY (MODEL > VRAM)" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "  Model:               Qwen1.5-MoE-A2.7B Q4_K_M (8.84 GB)"
Write-Host "  Backend:             llama-server b10649 CUDA 12.4"
Write-Host "  GPU Layers Offloaded: -ngl $NGL"
Write-Host "  Load Time:           $([math]::Round($loadTime, 1)) s"
Write-Host "  VRAM in GPU:         $vramUsedByModel MB"
Write-Host "  RAM in Host:         $ramConsumed GB"
Write-Host "  TTFT:                $([math]::Round($ttftMs, 1)) ms"
Write-Host ""
Write-Host "  Generation tok/s:"
Write-Host "    Average:           $avgTokS tok/s" -ForegroundColor $(if ($avgTokS -ge 10) { "Green" } else { "Yellow" })
Write-Host "    Median:            $medTokS tok/s"
Write-Host "    Minimum:           $minTokS tok/s"
Write-Host "    Maximum:           $maxTokS tok/s"
Write-Host ""
Write-Host "  Utilization:"
Write-Host "    GPU Active Avg:    $avgGPU%"
Write-Host "    CPU Active Avg:    $avgCPU%"
Write-Host "    GPU Power Draw:    ${avgPower} W"
Write-Host ""

$output = @{
    timestamp            = (Get-Date -Format "o")
    phase                = "FASE_0_5_BASELINE_1"
    model                = "Qwen1.5-MoE-A2.7B-Q4_K_M"
    model_size_gb        = $modelSizeGB
    backend              = "llama-server-b10649-cuda12.4"
    gpu                  = $gpuInfo
    ngl                  = $NGL
    load_time_sec        = [math]::Round($loadTime, 2)
    vram_used_mb         = $vramUsedByModel
    ram_consumed_gb      = $ramConsumed
    ttft_ms              = [math]::Round($ttftMs, 1)
    tok_s_avg            = $avgTokS
    tok_s_med            = $medTokS
    tok_s_min            = $minTokS
    tok_s_max            = $maxTokS
    gpu_util_avg_pct     = $avgGPU
    cpu_util_avg_pct     = $avgCPU
    gpu_power_avg_w      = $avgPower
    gate_10_tok_s        = ($avgTokS -ge 10)
    runs                 = $results
}

$output | ConvertTo-Json -Depth 5 | Out-File -Encoding utf8 -FilePath $OutputJson
Write-Host "  Results saved to: $OutputJson"

Stop-Process -Id $serverProc.Id -Force -ErrorAction SilentlyContinue
Write-Host "  Server stopped."
Write-Host "========================================================" -ForegroundColor Cyan
