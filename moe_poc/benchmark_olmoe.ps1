param(
    [string]$ModelPath    = "C:\as-code\moe_poc\models\OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf",
    [int]   $NGL          = 99,
    [int]   $Port         = 8765,
    [int]   $WarmupTokens = 50,
    [int]   $BenchTokens  = 200,
    [int]   $NRuns        = 3,
    [string]$BinsDir      = "C:\as-code\moe_poc\bins",
    [string]$OutputJson   = "C:\as-code\moe_poc\benchmark_results.json"
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

function Wait-ServerReady([int]$port, [int]$timeoutSec = 60) {
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
        model       = "olmoe"
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
Write-Host "============================================" -ForegroundColor Cyan
Write-Host " AS-Core MoE POC - FASE 0 Benchmark" -ForegroundColor Cyan
Write-Host " OLMoE-1B-7B-Q4_K_M | GTX 1650 Ti" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
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

Write-Host "[HARDWARE]"
$gpuInfo = & nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader,nounits
Write-Host "  GPU: $gpuInfo"
Write-Host "  VRAM libre antes de cargar: $(Get-VRAMFree) MB"

$ramInfo = python -c "import psutil; v=psutil.virtual_memory(); print(f'{v.total//(1024**3)} GB total, {v.available//(1024**3)} GB available')" 2>$null
Write-Host "  RAM: $ramInfo"
Write-Host ""

Write-Host "[SERVER] Launching llama-server with -ngl $NGL (GPU-first)..."
$serverArgs = @(
    "-m", $ModelPath,
    "--port", $Port,
    "-ngl", $NGL,
    "--ctx-size", "2048",
    "--threads", "4",
    "--log-disable"
)

$vramBefore = Get-VRAMFree
$loadStart  = Get-Date

$serverProc = Start-Process -FilePath $serverExe `
                             -ArgumentList $serverArgs `
                             -WorkingDirectory $BinsDir `
                             -PassThru `
                             -WindowStyle Hidden `
                             -RedirectStandardError "C:\as-code\moe_poc\server_stderr.log"

Write-Host "  PID: $($serverProc.Id)"

$ready = Wait-ServerReady -port $Port -timeoutSec 90
if (-not $ready) {
    Write-Host "Server failed to start. Check C:\as-code\moe_poc\server_stderr.log" -ForegroundColor Red
    Stop-Process -Id $serverProc.Id -Force -ErrorAction SilentlyContinue
    exit 1
}

$loadTime   = ((Get-Date) - $loadStart).TotalSeconds
$vramAfter  = Get-VRAMFree
$vramLoaded = $vramBefore - $vramAfter

Write-Host ""
Write-Host "[LOAD STATS]"
Write-Host "  Load time:     $([math]::Round($loadTime, 1)) s"
Write-Host "  VRAM delta:    $vramLoaded MB"
Write-Host "  VRAM free now: $vramAfter MB"
Write-Host ""

Write-Host "[WARMUP] Generating $WarmupTokens tokens..."
$warmup = Invoke-ChatCompletion -port $Port -prompt "Write a short poem about stars." -maxTokens $WarmupTokens
Write-Host "  Warmup: $($warmup.tok_s) tok/s ($($warmup.tokens_generated) tokens in $([math]::Round($warmup.elapsed_sec,2))s)"
Write-Host ""

Write-Host "[BENCHMARK] $NRuns runs x $BenchTokens tokens each"
Write-Host ""

$prompts = @(
    "Explain in detail how neural networks learn from data. Include backpropagation.",
    "Write a complete Python implementation of a binary search tree with insert, search and delete.",
    "Describe the differences between supervised, unsupervised and reinforcement learning."
)

$results = @()
for ($i = 0; $i -lt $NRuns; $i++) {
    $prompt = $prompts[$i % $prompts.Count]
    $gpuUtil = Get-GPUUtil
    $vramNow  = Get-VRAMUsed

    Write-Host "  Run $($i+1)/$NRuns..." -NoNewline
    $r = Invoke-ChatCompletion -port $Port -prompt $prompt -maxTokens $BenchTokens

    $tokColor = if ($r.tok_s -ge 10) { "Green" } else { "Red" }
    Write-Host " $($r.tok_s) tok/s | $($r.tokens_generated) tokens | $([math]::Round($r.elapsed_sec,2))s | GPU util: ${gpuUtil}%" -ForegroundColor $tokColor

    $results += @{
        run               = $i + 1
        prompt_tokens     = $r.prompt_tokens
        completion_tokens = $r.tokens_generated
        elapsed_sec       = $r.elapsed_sec
        tok_s             = $r.tok_s
        gpu_util_pct      = $gpuUtil
        vram_used_mb      = $vramNow
    }
}

Write-Host ""
Write-Host "[TTFT] Measuring Time-To-First-Token..."
$ttftStart = Get-Date
$ttftBody  = @{
    model       = "olmoe"
    messages    = @(@{ role = "user"; content = "Hello" })
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

$avgTokS   = [math]::Round(($results | Measure-Object -Property tok_s -Average).Average, 2)
$minTokS   = [math]::Round(($results | Measure-Object -Property tok_s -Minimum).Minimum, 2)
$maxTokS   = [math]::Round(($results | Measure-Object -Property tok_s -Maximum).Maximum, 2)
$avgGPU    = [math]::Round(($results | Measure-Object -Property gpu_util_pct -Average).Average, 0)
$avgVRAM   = [math]::Round(($results | Measure-Object -Property vram_used_mb -Average).Average, 0)

Write-Host "============================================" -ForegroundColor Cyan
Write-Host " RESULTS SUMMARY" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Model:         OLMoE-1B-7B Q4_K_M"
Write-Host "  Backend:       llama-server b10649 CUDA 12.4"
Write-Host "  GPU layers:    -ngl $NGL"
Write-Host "  VRAM loaded:   ~$vramLoaded MB"
Write-Host "  Load time:     $([math]::Round($loadTime,1)) s"
Write-Host "  TTFT:          $([math]::Round($ttftMs,1)) ms"
Write-Host ""
Write-Host "  tok/s avg:     $avgTokS" -ForegroundColor $(if ($avgTokS -ge 10) { "Green" } else { "Red" })
Write-Host "  tok/s min:     $minTokS"
Write-Host "  tok/s max:     $maxTokS"
Write-Host "  GPU util avg:  $avgGPU%"
Write-Host "  VRAM used avg: $avgVRAM MB"
Write-Host ""

$gate = if ($avgTokS -ge 10) { "PASS - GATE >=10 tok/s SATISFIED" } else { "FAIL - GATE >=10 tok/s NOT REACHED" }
$gateColor = if ($avgTokS -ge 10) { "Green" } else { "Red" }
Write-Host "  GATE:          $gate" -ForegroundColor $gateColor
Write-Host ""

$output = @{
    timestamp        = (Get-Date -Format "o")
    phase            = "FASE_0_CONDICION_1"
    model            = "OLMoE-1B-7B-0924-Instruct-Q4_K_M"
    backend          = "llama-server-b10649-cuda12.4"
    gpu              = $gpuInfo
    ngl              = $NGL
    vram_loaded_mb   = $vramLoaded
    load_time_sec    = [math]::Round($loadTime, 2)
    ttft_ms          = [math]::Round($ttftMs, 1)
    tok_s_avg        = $avgTokS
    tok_s_min        = $minTokS
    tok_s_max        = $maxTokS
    gpu_util_avg_pct = $avgGPU
    vram_used_avg_mb = $avgVRAM
    gate_10_tok_s    = ($avgTokS -ge 10)
    runs             = $results
}

$output | ConvertTo-Json -Depth 5 | Out-File -Encoding utf8 -FilePath $OutputJson
Write-Host "  Results saved: $OutputJson"

Stop-Process -Id $serverProc.Id -Force -ErrorAction SilentlyContinue
Write-Host "  Server stopped."
