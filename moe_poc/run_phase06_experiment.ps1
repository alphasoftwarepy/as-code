param(
    [string]$ModelPath  = "C:\as-code\moe_poc\models\qwen1.5-moe-a2.7b-q4_k_m.gguf",
    [string]$BinsDir    = "C:\as-code\moe_poc\bins",
    [string]$OutputJson = "C:\as-code\moe_poc\EXPERT_RESIDENCY_RESULTS.json"
)

$ErrorActionPreference = "Stop"

$prompts = @(
    @{ id = "P1_Reasoning"; prompt = "Explain step by step why the sky is blue and how Rayleigh scattering differs from Mie scattering." },
    @{ id = "P2_Coding";    prompt = "Write a complete Python implementation of Dijkstra algorithm with priority queue." },
    @{ id = "P3_Logic";     prompt = "Three boxes are labeled incorrectly: Apples, Oranges, Both. You pick 1 fruit from 'Both'. Explain how to label all." },
    @{ id = "P4_Systems";   prompt = "Describe the key architectural differences between monolithic and microservices systems." },
    @{ id = "P5_MoE";       prompt = "Summarize the key benefits of Mixture of Experts MoE models compared to dense transformers." }
)

function Get-VRAMFree {
    try {
        $out = & nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>$null
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

function Get-CPUUtilPct {
    try {
        $out = python -c "import psutil; print(f'{psutil.cpu_percent(interval=0.2):.1f}')" 2>$null
        return [double]$out.Trim()
    } catch { return -1 }
}

function Wait-ServerReady([int]$port, [int]$timeoutSec = 120) {
    $start = Get-Date
    while ((Get-Date) -lt $start.AddSeconds($timeoutSec)) {
        try {
            $resp = Invoke-RestMethod -Uri "http://127.0.0.1:$port/health" -TimeoutSec 2 -ErrorAction Stop
            if ($resp.status -eq "ok" -or $resp -match "ok") {
                return $true
            }
        } catch {}
        Start-Sleep -Milliseconds 400
    }
    return $false
}

function Run-Workload([int]$port, [string]$testName) {
    Write-Host "  -> Running Workload on $testName (5 multi-domain prompts)..." -ForegroundColor Cyan
    $runResults = @()

    foreach ($item in $prompts) {
        $body = @{
            model       = "qwen-moe"
            messages    = @(@{ role = "user"; content = $item.prompt })
            max_tokens  = 80
            temperature = 0.0
            stream      = $false
        } | ConvertTo-Json -Depth 5

        $start = Get-Date
        $resp  = Invoke-RestMethod -Uri "http://127.0.0.1:$port/v1/chat/completions" `
                                   -Method POST `
                                   -ContentType "application/json" `
                                   -Body $body `
                                   -TimeoutSec 180
        $elapsed = ((Get-Date) - $start).TotalSeconds
        $usage = $resp.usage
        $toks = if ($usage.completion_tokens) { [int]$usage.completion_tokens } else { 80 }
        $spd  = [math]::Round($toks / $elapsed, 2)
        $gpuU = Get-GPUUtil
        $cpuU = Get-CPUUtilPct
        $pwr  = Get-GPUPower

        Write-Host "     [$($item.id)] $spd tok/s ($toks tok in $([math]::Round($elapsed,2))s) | GPU: ${gpuU}% | CPU: ${cpuU}% | Power: ${pwr}W"

        $runResults += @{
            prompt_id         = $item.id
            completion_tokens = $toks
            elapsed_sec       = $elapsed
            tok_s             = $spd
            gpu_util_pct      = $gpuU
            cpu_util_pct      = $cpuU
            gpu_power_w       = $pwr
        }
    }

    $tokValues = $runResults | ForEach-Object { [double]$_.tok_s }
    $avgTok = [math]::Round(($tokValues | Measure-Object -Average).Average, 2)
    $minTok = [math]::Round(($tokValues | Measure-Object -Minimum).Minimum, 2)
    $maxTok = [math]::Round(($tokValues | Measure-Object -Maximum).Maximum, 2)
    $sortedTok = $tokValues | Sort-Object
    $medTok = $sortedTok[[math]::Floor($sortedTok.Count / 2)]
    $avgGpu = [math]::Round(($runResults | ForEach-Object { [double]$_.gpu_util_pct } | Measure-Object -Average).Average, 0)
    $avgCpu = [math]::Round(($runResults | ForEach-Object { [double]$_.cpu_util_pct } | Measure-Object -Average).Average, 0)
    $avgPwr = [math]::Round(($runResults | ForEach-Object { [double]$_.gpu_power_w } | Measure-Object -Average).Average, 1)

    return @{
        runs        = $runResults
        tok_s_avg   = $avgTok
        tok_s_med   = $medTok
        tok_s_min   = $minTok
        tok_s_max   = $maxTok
        gpu_avg_pct = $avgGpu
        cpu_avg_pct = $avgCpu
        power_avg_w = $avgPwr
    }
}

Write-Host "========================================================" -ForegroundColor Cyan
Write-Host " AS-Core FASE 0.6: EXPERT RESIDENCY EXPERIMENTAL SUITE" -ForegroundColor Cyan
Write-Host " Hardware: GTX 1650 Ti (4GB VRAM) / 16GB RAM" -ForegroundColor Cyan
Write-Host " Model:    Qwen1.5-MoE-A2.7B (8.84 GB)" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host ""

$serverExe = Join-Path $BinsDir "llama-server.exe"

# ==============================================================================
# TEST 1: RE-CONFIRM BASELINE 1A (-ngl 10, Standard Layer Offload)
# ==============================================================================
Write-Host "[EXPERIMENT 1] BASELINE 1A: Layer Offload (-ngl 10)..." -ForegroundColor Yellow
$port1 = 8770
$log1  = "C:\as-code\moe_poc\exp_baseline1a.log"
$args1 = @("-m", $ModelPath, "--port", $port1, "-ngl", "10", "--ctx-size", "2048", "--threads", "6", "--log-disable")

$vFreeBefore1 = Get-VRAMFree
$proc1 = Start-Process -FilePath $serverExe -ArgumentList $args1 -WorkingDirectory $BinsDir -PassThru -WindowStyle Hidden -RedirectStandardError $log1
Wait-ServerReady -port $port1 | Out-Null
$vFreeAfter1 = Get-VRAMFree
$vramUsed1 = $vFreeBefore1 - $vFreeAfter1

$base1aResults = Run-Workload -port $port1 -testName "BASELINE 1A (-ngl 10)"
Stop-Process -Id $proc1.Id -Force -ErrorAction SilentlyContinue
Write-Host "  -> Baseline 1A Average: $($base1aResults.tok_s_avg) tok/s (Min: $($base1aResults.tok_s_min), Max: $($base1aResults.tok_s_max), GPU: $($base1aResults.gpu_avg_pct)%)" -ForegroundColor Green
Write-Host ""

# ==============================================================================
# TEST 2: ALL DENSE ATTENTION IN GPU + EXPERTS IN CPU (-ngl 25 --cpu-moe)
# ==============================================================================
Write-Host "[EXPERIMENT 2] DENSE ATTENTION IN GPU + CPU EXPERTS (--cpu-moe -ngl 25)..." -ForegroundColor Yellow
$port2 = 8771
$log2  = "C:\as-code\moe_poc\exp_cpumoe.log"
$args2 = @("-m", $ModelPath, "--port", $port2, "-ngl", "25", "--cpu-moe", "--ctx-size", "2048", "--threads", "6", "--log-disable")

$vFreeBefore2 = Get-VRAMFree
$proc2 = Start-Process -FilePath $serverExe -ArgumentList $args2 -WorkingDirectory $BinsDir -PassThru -WindowStyle Hidden -RedirectStandardError $log2
Wait-ServerReady -port $port2 | Out-Null
$vFreeAfter2 = Get-VRAMFree
$vramUsed2 = $vFreeBefore2 - $vFreeAfter2

$cpuMoeResults = Run-Workload -port $port2 -testName "ALL DENSE IN GPU (--cpu-moe)"
Stop-Process -Id $proc2.Id -Force -ErrorAction SilentlyContinue
Write-Host "  -> Dense GPU + CPU Experts Average: $($cpuMoeResults.tok_s_avg) tok/s (GPU: $($cpuMoeResults.gpu_avg_pct)%)" -ForegroundColor Green
Write-Host ""

# ==============================================================================
# TEST 3: HYBRID PARTIAL LAYER ALLOCATION (-ngl 6, conservative)
# ==============================================================================
Write-Host "[EXPERIMENT 3] CONSERVATIVE LAYER OFFLOAD (-ngl 6)..." -ForegroundColor Yellow
$port3 = 8772
$log3  = "C:\as-code\moe_poc\exp_ngl6.log"
$args3 = @("-m", $ModelPath, "--port", $port3, "-ngl", "6", "--ctx-size", "2048", "--threads", "6", "--log-disable")

$vFreeBefore3 = Get-VRAMFree
$proc3 = Start-Process -FilePath $serverExe -ArgumentList $args3 -WorkingDirectory $BinsDir -PassThru -WindowStyle Hidden -RedirectStandardError $log3
Wait-ServerReady -port $port3 | Out-Null
$vFreeAfter3 = Get-VRAMFree
$vramUsed3 = $vFreeBefore3 - $vFreeAfter3

$ngl6Results = Run-Workload -port $port3 -testName "CONSERVATIVE (-ngl 6)"
Stop-Process -Id $proc3.Id -Force -ErrorAction SilentlyContinue
Write-Host "  -> NGL 6 Average: $($ngl6Results.tok_s_avg) tok/s (GPU: $($ngl6Results.gpu_avg_pct)%)" -ForegroundColor Green
Write-Host ""

# ==============================================================================
# SAVE COMPLETE RESULTS
# ==============================================================================
$outputData = @{
    timestamp = (Get-Date -Format "o")
    phase     = "FASE_0_6_EXPERT_RESIDENCY_EXPERIMENT"
    model     = "Qwen1.5-MoE-A2.7B-Q4_K_M"
    hardware  = @{
        gpu          = "NVIDIA GeForce GTX 1650 Ti (4GB VRAM)"
        vram_total   = 4096
        ram_total_gb = 16
    }
    workload = $prompts
    experiments = @{
        baseline_1a_ngl10 = @{
            config       = "-ngl 10 (10 GPU layers / 14 CPU layers)"
            vram_used_mb = $vramUsed1
            results      = $base1aResults
        }
        dense_gpu_cpu_moe = @{
            config       = "-ngl 25 --cpu-moe (24 Dense Attention in GPU / All Experts in CPU)"
            vram_used_mb = $vramUsed2
            results      = $cpuMoeResults
        }
        conservative_ngl6 = @{
            config       = "-ngl 6 (6 GPU layers / 18 CPU layers)"
            vram_used_mb = $vramUsed3
            results      = $ngl6Results
        }
    }
}

$outputData | ConvertTo-Json -Depth 6 | Out-File -Encoding utf8 -FilePath $OutputJson
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host " Experiment suite completed. Results saved to $OutputJson" -ForegroundColor Green
Write-Host "========================================================" -ForegroundColor Cyan
